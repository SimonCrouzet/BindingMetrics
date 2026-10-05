"""Coulomb cross-chain energy: D-residues, HIP, phospho residues and the coverage counters."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics.electrostatics import (  # noqa: E402
    _FORMAL_CHARGES,
    compute_coulomb_cross_chain,
)

DATA = Path(__file__).resolve().parents[1] / "data"
K_KJ = 1389.35  # e^2 / (4 pi eps0) in kJ Å / mol
EPSILON = 4.0


def _write_pdb(tmp_path, records, name="complex.pdb"):
    """Write (chain, res_id, res_name, atom_name, element, xyz) records as a PDB file."""
    arr = struc.AtomArray(len(records))
    arr.chain_id = np.array([r[0] for r in records])
    arr.res_id = np.array([r[1] for r in records], dtype=int)
    arr.res_name = np.array([r[2] for r in records])
    arr.atom_name = np.array([r[3] for r in records])
    arr.element = np.array([r[4] for r in records])
    arr.coord = np.array([r[5] for r in records], dtype=float)
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, arr)
    path = tmp_path / name
    pdb_file.write(str(path))
    return path


def _receptor_with(res_name, atoms_and_xyz, first_res_id=11):
    """Chain A: ten inert ALA residues far away plus one residue with the given atoms."""
    inert = [("A", i + 1, "ALA", "CA", "C", (200.0 + 5 * i, 0.0, 0.0)) for i in range(10)]
    special = [("A", first_res_id, res_name, atom, atom[0], xyz) for atom, xyz in atoms_and_xyz]
    return inert + special


def _expected_kj(pairs):
    """Σ q1 q2 / r over (q1, q2, r) triples, in kJ/mol at ε = 4."""
    return sum(q1 * q2 / r for q1, q2, r in pairs) * K_KJ / EPSILON


class TestDResidues:
    def _complex(self, peptide_lys, peptide_ala, receptor_asp):
        receptor = _receptor_with(
            receptor_asp, [("OD1", (0.0, 0.0, 0.0)), ("OD2", (1.0, 0.0, 0.0))]
        )
        peptide = [
            ("B", 1, peptide_lys, "NZ", "N", (4.0, 0.0, 0.0)),
            ("B", 2, peptide_ala, "CA", "C", (60.0, 0.0, 0.0)),
            ("B", 3, peptide_ala, "CA", "C", (64.0, 0.0, 0.0)),
        ]
        return receptor + peptide

    def test_all_d_peptide_gets_the_energy_of_its_l_twin(self, tmp_path):
        d_path = _write_pdb(tmp_path, self._complex("DLY", "DAL", "ASP"), "d.pdb")
        l_path = _write_pdb(tmp_path, self._complex("LYS", "ALA", "ASP"), "l.pdb")

        d_result = compute_coulomb_cross_chain(d_path)  # chains auto-detected
        l_result = compute_coulomb_cross_chain(l_path)

        assert d_result["coulomb_energy_kJ"] != 0.0
        assert d_result["coulomb_energy_kJ"] == pytest.approx(l_result["coulomb_energy_kJ"])
        assert d_result["n_attractive"] == 2
        assert [a["atom"] for a in d_result["charged_atoms_peptide"]] == ["NZ"]
        assert d_result["charged_atoms_peptide"][0]["residue"] == "DLY:B:1"

    def test_energy_matches_the_coulomb_sum(self, tmp_path):
        path = _write_pdb(tmp_path, self._complex("DLY", "DAL", "DAS"), "d_asp.pdb")
        result = compute_coulomb_cross_chain(path, "B", "A")
        expected = _expected_kj([(+1.0, -0.5, 4.0), (+1.0, -0.5, 3.0)])
        assert result["coulomb_energy_kJ"] == pytest.approx(expected, rel=1e-5)
        assert result["coulomb_energy_kcal"] == pytest.approx(expected / 4.184, rel=1e-5)

    def test_d_arginine_and_d_glutamate_are_charged(self, tmp_path):
        records = [
            ("A", 1, "DAR", "NH1", "N", (0.0, 0.0, 0.0)),
            ("A", 1, "DAR", "NH2", "N", (0.0, 3.0, 0.0)),
            ("B", 1, "DGL", "OE1", "O", (4.0, 0.0, 0.0)),
            ("B", 1, "DGL", "OE2", "O", (4.0, 3.0, 0.0)),
        ]
        result = compute_coulomb_cross_chain(_write_pdb(tmp_path, records), "B", "A")
        assert result["n_charged_pairs"] == 4
        assert result["n_attractive"] == 4
        assert result["coulomb_energy_kJ"] < 0.0


class TestHistidineAndPhospho:
    def test_hip_is_positive_and_plain_his_is_neutral(self, tmp_path):
        def energy(his_name):
            records = _receptor_with("ASP", [("OD1", (0.0, 0.0, 0.0))]) + [
                ("B", 1, his_name, "ND1", "N", (4.0, 0.0, 0.0)),
                ("B", 1, his_name, "NE2", "N", (0.0, 5.0, 0.0)),
            ]
            path = _write_pdb(tmp_path, records, f"{his_name}.pdb")
            return compute_coulomb_cross_chain(path, "B", "A")

        hip, his = energy("HIP"), energy("HIS")
        assert hip["coulomb_energy_kJ"] == pytest.approx(
            _expected_kj([(+0.5, -0.5, 4.0), (+0.5, -0.5, 5.0)]), rel=1e-5
        )
        assert his["n_charged_pairs"] == 0
        assert his["coulomb_energy_kJ"] == 0.0

    @pytest.mark.parametrize("residue", ["SEP", "TPO", "PTR"])
    def test_phosphate_charge_is_minus_two_over_three_oxygens(self, tmp_path, residue):
        records = _receptor_with("LYS", [("NZ", (0.0, 0.0, 0.0))]) + [
            ("B", 1, residue, "O1P", "O", (3.0, 0.0, 0.0)),
            ("B", 1, residue, "O2P", "O", (4.0, 0.0, 0.0)),
            ("B", 1, residue, "O3P", "O", (5.0, 0.0, 0.0)),
            ("B", 1, residue, "P", "P", (4.0, 1.0, 0.0)),
        ]
        result = compute_coulomb_cross_chain(_write_pdb(tmp_path, records), "B", "A")
        q = -2.0 / 3.0
        assert result["coulomb_energy_kJ"] == pytest.approx(
            _expected_kj([(1.0, q, 3.0), (1.0, q, 4.0), (1.0, q, 5.0)]), rel=1e-5
        )
        assert sum(a["charge"] for a in result["charged_atoms_peptide"]) == pytest.approx(-2.0)

    def test_net_charge_per_residue(self):
        net = defaultdict(float)
        for (res, _atom), charge in _FORMAL_CHARGES.items():
            net[res] += charge
        expected = {
            "LYS": 1.0,
            "ARG": 1.0,
            "HIP": 1.0,
            "ASP": -1.0,
            "GLU": -1.0,
            "SEP": -2.0,
            "TPO": -2.0,
            "PTR": -2.0,
        }
        assert dict(net) == pytest.approx(expected)


class TestCoverageCounters:
    def test_counts_on_a_synthetic_complex(self, tmp_path):
        records = [
            ("A", 1, "LYS", "NZ", "N", (0.0, 0.0, 0.0)),
            ("A", 2, "ALA", "CA", "C", (30.0, 0.0, 0.0)),
            ("A", 3, "MLE", "CA", "C", (35.0, 0.0, 0.0)),  # unrecognised ncAA
            ("A", 4, "HOH", "O", "O", (40.0, 0.0, 0.0)),  # solvent: not an amino acid
            ("B", 1, "DAS", "OD1", "O", (4.0, 0.0, 0.0)),
            ("B", 2, "SEP", "O1P", "O", (5.0, 0.0, 0.0)),
            ("B", 3, "BMT", "CA", "C", (50.0, 0.0, 0.0)),  # unrecognised ncAA
            ("B", 4, "DAL", "CA", "C", (55.0, 0.0, 0.0)),
            ("B", 5, "ABA", "CA", "C", (60.0, 0.0, 0.0)),  # unrecognised ncAA
        ]
        result = compute_coulomb_cross_chain(_write_pdb(tmp_path, records), "B", "A")
        assert result["n_ionisable_residues_seen"] == 3  # LYS, DAS, SEP
        assert result["n_residues_unrecognised"] == 3  # MLE, BMT, ABA

    def test_bundled_p53_mdm2_energy_is_unchanged(self):
        result = compute_coulomb_cross_chain(DATA / "example_linear_p53_1YCR.pdb")
        assert result["coulomb_energy_kJ"] == pytest.approx(-91.58911325183033, rel=1e-9)
        assert result["n_charged_pairs"] == 16
        assert result["n_residues_unrecognised"] == 0
        assert result["n_ionisable_residues_seen"] > 0

    def test_cyclosporin_reports_its_ncaa_residues(self):
        result = compute_coulomb_cross_chain(DATA / "example_ncaa_cyclosporin_1CWA.cif")
        # Cyclosporin A: four N-methyl-Leu, N-methyl-Val, MeBmt, Abu and Sar are outside the
        # charge table; D-Ala maps to Ala and Val is standard.
        assert result["n_residues_unrecognised"] == 8
        assert result["n_ionisable_residues_seen"] > 0

    def test_default_result_carries_the_counters(self):
        # One protein chain only: no receptor, so the zero default is returned.
        result = compute_coulomb_cross_chain(DATA / "example_lactam_somatostatin_1XY4.cif")
        assert result["n_ionisable_residues_seen"] == 0
        assert result["n_residues_unrecognised"] == 0
        assert result["coulomb_energy_kJ"] == 0.0
