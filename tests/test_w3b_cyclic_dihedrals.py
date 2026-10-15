"""Ring-closing phi/psi/omega of head-to-tail cyclic peptides.

The closing amide bond links the last residue's C to the first residue's N.
Its angles must be scored like every other backbone angle, so a cyclic chain
has as many evaluated residues and peptide bonds as it has residues.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdbx as pdbx  # noqa: E402

from binding_metrics.metrics.geometry import (  # noqa: E402
    _load_structure,
    compute_omega_planarity,
    compute_ramachandran,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"  # chain C: 11 residues, D-Ala first
SFTI1 = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"  # chain I: 14 residues, head-to-tail
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"  # chain B: 13 residues, linear
SOMATOSTATIN = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"  # 12 residues, side-chain lactam


# Documented cut-offs of compute_omega_planarity: an outlier is more than 15 degrees from trans,
# a cis bond has |omega| below 30 degrees.
OMEGA_OUTLIER_DEG = 15.0
CIS_OMEGA_DEG = 30.0


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} not bundled")
    return path


def _amino_acid_chain(path: Path, chain: str):
    atoms = _load_structure(path)
    return atoms[(atoms.chain_id == chain) & struc.filter_amino_acids(atoms)]


def _rotate_residues(peptide, shift: int):
    """The residues of ``peptide`` in cyclic order from residue ``shift``.

    Coordinates and residue numbers are untouched, only the order of the residues changes.
    """
    bounds = [*struc.get_residue_starts(peptide), peptide.array_length()]
    residues = [peptide[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    order = residues[shift:] + residues[:shift]
    rotated = order[0]
    for residue in order[1:]:
        rotated = rotated + residue
    return rotated


def backbone_angles_deg(path: Path, chain: str, ring: bool = False):
    """phi, psi and omega in degrees, one row per amino-acid residue of a chain, from biotite alone.

    ``dihedral_backbone`` walks the chain in the order of the file and never closes a ring. For a
    head-to-tail ring the angles that involve the closing amide are read from the same residues
    listed from the second one: the last residue is then followed by the first, so the closing
    bond is a sequential one. The rows are then those of the file order, the closing phi in the
    first row, the closing psi and omega in the last. The torsions come from float32 coordinates
    and differ by up to 2e-5 degree between machines; a reference computed on the machine that
    runs the test cancels that noise.
    """
    peptide = _amino_acid_chain(path, chain)
    phi, psi, omega = (np.degrees(angles) for angles in struc.dihedral_backbone(peptide))
    if ring:
        rotated = struc.dihedral_backbone(_rotate_residues(peptide, 1))
        phi_ring, psi_ring, omega_ring = (np.degrees(angles) for angles in rotated)
        # rows of the rotated order: residues 2 to n, then residue 1
        phi[0] = phi_ring[-1]
        psi[-1] = psi_ring[-2]
        omega[-1] = omega_ring[-2]
    return phi, psi, omega


def deviation_from_trans_deg(omega_deg):
    """|omega - 180| wrapped to [0, 180], for one angle or an array of them."""
    return np.abs(np.asarray(omega_deg) % 360.0 - 180.0)


class TestCyclosporin:
    def test_every_residue_is_evaluated(self):
        rama = compute_ramachandran(_need(CYCLOSPORIN), chain="C")
        assert rama["cyclic_closure_detected"] is True
        assert rama["cyclic_closure_evaluated"] is True
        assert rama["n_residues_evaluated"] == 11

    def test_d_alanine_at_position_one_is_scored(self):
        rama = compute_ramachandran(_need(CYCLOSPORIN), chain="C")
        assert rama["n_d_residues"] == 1
        first = rama["per_residue"][0]
        assert (first["res_id"], first["res_name"], first["is_d_aa"]) == (1, "DAL", True)
        # D-residue phi is positive; it is judged on the mirrored plot.
        assert 60.0 < first["phi"] < 120.0
        assert first["region"] == "favoured"

    def test_last_residue_psi_comes_from_the_closing_bond(self):
        rama = compute_ramachandran(_need(CYCLOSPORIN), chain="C")
        last = rama["per_residue"][-1]
        assert (last["res_id"], last["res_name"]) == (11, "ALA")
        # psi of the last residue is N-CA-C-N(first): the bond that only the ring-opened order of
        # the residues makes sequential
        _, psi, _ = backbone_angles_deg(CYCLOSPORIN, "C", ring=True)
        assert last["psi"] == pytest.approx(psi[-1], abs=1e-3)

    def test_closing_peptide_bond_is_evaluated(self):
        omega = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        assert omega["cyclic_closure_evaluated"] is True
        assert omega["n_bonds_evaluated"] == 11
        closing = omega["per_residue"][-1]
        assert closing["res_id"] == 11
        _, _, reference = backbone_angles_deg(CYCLOSPORIN, "C", ring=True)
        # the last row carries the angle of the ring-closing bond, not a NaN or another bond's
        assert closing["omega"] == pytest.approx(reference[-1], abs=1e-3)
        # the closing bond is judged by the same rule as the sequential ones
        outliers = deviation_from_trans_deg(reference) > OMEGA_OUTLIER_DEG
        assert closing["is_outlier"] == outliers[-1]
        assert omega["omega_outlier_count"] == outliers.sum()

    def test_sequential_angles_are_untouched(self):
        """The first 10 bonds alone give the mean deviation of a linear chain."""
        omega = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        deviations = [r["deviation"] for r in omega["per_residue"][:-1]]
        _, _, reference = backbone_angles_deg(CYCLOSPORIN, "C", ring=True)
        sequential = deviation_from_trans_deg(reference[:-1])
        assert len(deviations) == len(sequential)
        # the ring closure leaves the sequential bonds as biotite gives them
        assert sum(deviations) / len(deviations) == pytest.approx(sequential.mean(), abs=1e-3)


class TestSfti1:
    def test_all_fourteen_residues_and_bonds_are_evaluated(self):
        rama = compute_ramachandran(_need(SFTI1), chain="I")
        omega = compute_omega_planarity(_need(SFTI1), chain="I")
        assert rama["cyclic_closure_evaluated"] is True
        assert rama["n_residues_evaluated"] == 14
        assert omega["n_bonds_evaluated"] == 14

    def test_closing_bond_is_a_slightly_twisted_trans_amide(self):
        omega = compute_omega_planarity(_need(SFTI1), chain="I")
        closing = omega["per_residue"][-1]
        assert (closing["res_id"], closing["res_name"]) == (14, "ASP")
        _, _, reference = backbone_angles_deg(SFTI1, "I", ring=True)
        # the last row carries the angle of the ring-closing bond
        assert closing["omega"] == pytest.approx(reference[-1], abs=1e-3)
        # in this structure the closing amide is beyond the outlier cut-off and still closer to
        # trans than to cis
        assert OMEGA_OUTLIER_DEG < deviation_from_trans_deg(reference[-1]) < 90.0
        assert closing["is_outlier"] is True
        # the cis Pro is the only cis bond
        cis_bonds = np.abs(reference) < CIS_OMEGA_DEG
        assert cis_bonds.sum() == 1
        assert omega["omega_cis_count"] == cis_bonds.sum()


class TestLinearPeptidesAreUnchanged:
    def test_p53_terminal_residues_stay_unscored(self):
        rama = compute_ramachandran(_need(P53), chain="B")
        omega = compute_omega_planarity(_need(P53), chain="B")
        assert rama["cyclic_closure_detected"] is False
        assert rama["cyclic_closure_evaluated"] is False
        assert rama["n_residues_evaluated"] == 11  # 13 residues minus the two termini
        assert omega["n_bonds_evaluated"] == 12
        _, _, reference = backbone_angles_deg(P53, "B")
        deviations = deviation_from_trans_deg(reference[~np.isnan(reference)])
        assert len(deviations) == omega["n_bonds_evaluated"]
        # a linear chain keeps the mean over its sequential bonds, with no closing bond added
        assert omega["omega_mean_dev"] == pytest.approx(deviations.mean(), abs=1e-3)

    def test_side_chain_lactam_is_not_a_backbone_closure(self):
        rama = compute_ramachandran(_need(SOMATOSTATIN), chain="A")
        omega = compute_omega_planarity(_need(SOMATOSTATIN), chain="A")
        assert rama["cyclic_closure_evaluated"] is False
        assert rama["n_residues_evaluated"] == 10
        assert omega["n_bonds_evaluated"] == 11


def _write_rotated_ring(source: Path, chain: str, shift: int, out: Path) -> Path:
    """Write the chain with its residues cyclically re-ordered by ``shift``.

    Coordinates and residue numbers are untouched, only the order in the file
    changes, so a different residue is "first" and the ring closes at another
    bond. Whatever the metrics say for a residue must not depend on that order.
    """
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, _rotate_residues(_amino_acid_chain(source, chain), shift))
    cif.write(str(out))
    return out


class TestRingClosureAgreesWithSequentialAngles:
    """The closing angles equal what biotite gives when that bond is not the closure."""

    @pytest.mark.parametrize("shift", [1, 5, 10])
    def test_cyclosporin_angles_do_not_depend_on_where_the_ring_is_opened(self, shift, tmp_path):
        _need(CYCLOSPORIN)
        rotated = _write_rotated_ring(CYCLOSPORIN, "C", shift, tmp_path / "rotated.cif")

        reference_rama = compute_ramachandran(CYCLOSPORIN, chain="C")
        reference_omega = compute_omega_planarity(CYCLOSPORIN, chain="C")
        rama = compute_ramachandran(rotated, chain="C")
        omega = compute_omega_planarity(rotated, chain="C")

        assert rama["cyclic_closure_evaluated"] is True
        assert rama["n_residues_evaluated"] == 11
        assert omega["n_bonds_evaluated"] == 11
        expected = {r["res_id"]: r for r in reference_rama["per_residue"]}
        for entry in rama["per_residue"]:
            assert entry["phi"] == pytest.approx(expected[entry["res_id"]]["phi"], abs=1e-3)
            assert entry["psi"] == pytest.approx(expected[entry["res_id"]]["psi"], abs=1e-3)
        expected_omega = {r["res_id"]: r["omega"] for r in reference_omega["per_residue"]}
        for entry in omega["per_residue"]:
            assert entry["omega"] == pytest.approx(expected_omega[entry["res_id"]], abs=1e-3)
        assert rama["n_d_residues"] == reference_rama["n_d_residues"] == 1
