"""D-amino-acid and phospho residues in the salt-bridge and solvation charge tables."""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402

from binding_metrics.metrics.interface import _solvation_types  # noqa: E402
from binding_metrics.metrics.polar_contacts import (  # noqa: E402
    compute_saltbridges,
    l_equivalent_residue_names,
)


def _atoms(records):
    """AtomArray from (chain, res_id, res_name, atom_name, element, xyz) tuples."""
    arr = struc.AtomArray(len(records))
    arr.chain_id = np.array([r[0] for r in records])
    arr.res_id = np.array([r[1] for r in records], dtype=int)
    arr.res_name = np.array([r[2] for r in records])
    arr.atom_name = np.array([r[3] for r in records])
    arr.element = np.array([r[4] for r in records])
    arr.coord = np.array([r[5] for r in records], dtype=float)
    return arr


def _pair(pos, neg):
    """Chain A holds the positive residue, chain B the negative one, 3.0 Å apart."""
    return _atoms([("A", 1, *pos, (0.0, 0.0, 0.0)), ("B", 1, *neg, (3.0, 0.0, 0.0))])


class TestLEquivalentNames:
    def test_d_codes_map_to_l_names_and_others_pass_through(self):
        names = l_equivalent_residue_names(["dly", "DAR ", "DAS", "DGL", "ALA", "MLE", "SEP"])
        assert list(names) == ["LYS", "ARG", "ASP", "GLU", "ALA", "MLE", "SEP"]

    def test_empty_input(self):
        assert l_equivalent_residue_names([]).size == 0


class TestSaltBridgesWithDResidues:
    @pytest.mark.parametrize(
        "positive, negative",
        [
            (("LYS", "NZ", "N"), ("ASP", "OD1", "O")),
            (("DLY", "NZ", "N"), ("ASP", "OD1", "O")),
            (("LYS", "NZ", "N"), ("DAS", "OD1", "O")),
            (("DAR", "NH1", "N"), ("DGL", "OE1", "O")),
        ],
    )
    def test_d_residue_pairs_form_the_same_bridge_as_l_residues(self, positive, negative):
        result = compute_saltbridges(_pair(positive, negative), "A", "B")
        assert result["saltbridges"] == 1
        assert result["saltbridge_energy"] == pytest.approx(-83.0159 / 3.0, rel=1e-3)

    def test_all_d_peptide_against_l_receptor_has_bridges(self):
        atoms = _atoms(
            [
                ("A", 1, "ARG", "NH1", "N", (0.0, 0.0, 0.0)),
                ("A", 2, "GLU", "OE1", "O", (20.0, 0.0, 0.0)),
                ("B", 1, "DGL", "OE2", "O", (3.0, 0.0, 0.0)),
                ("B", 2, "DLY", "NZ", "N", (23.0, 0.0, 0.0)),
            ]
        )
        assert compute_saltbridges(atoms, "A", "B")["saltbridges"] == 2


class TestPhosphoSaltBridges:
    @pytest.mark.parametrize("residue", ["SEP", "TPO", "PTR"])
    @pytest.mark.parametrize("oxygen", ["O1P", "O2P", "O3P"])
    def test_phosphate_oxygens_pair_with_arginine(self, residue, oxygen):
        result = compute_saltbridges(_pair(("ARG", "NH1", "N"), (residue, oxygen, "O")), "A", "B")
        assert result["saltbridges"] == 1

    def test_phosphorus_and_bridging_oxygen_do_not_count(self):
        for atom_name, element in (("P", "P"), ("OG", "O")):
            atoms = _pair(("ARG", "NH1", "N"), ("SEP", atom_name, element))
            assert compute_saltbridges(atoms, "A", "B")["saltbridges"] == 0

    def test_default_hetero_filter_keeps_phospho_and_d_residues(self):
        atoms = _pair(("DLY", "NZ", "N"), ("SEP", "O1P", "O"))
        assert compute_saltbridges(atoms, "A", "B", hetero="ignore")["saltbridges"] == 1


class TestSolvationTypesWithDResidues:
    def test_charged_groups_of_d_and_phospho_residues_are_typed_as_charged(self):
        atoms = _atoms(
            [
                ("A", 1, "DLY", "NZ", "N", (0, 0, 0)),
                ("A", 1, "DLY", "N", "N", (0, 0, 1)),
                ("A", 2, "DAS", "OD1", "O", (0, 0, 2)),
                ("A", 2, "DAS", "O", "O", (0, 0, 3)),
                ("A", 3, "SEP", "O1P", "O", (0, 0, 4)),
                ("A", 3, "SEP", "OG", "O", (0, 0, 5)),
                ("A", 3, "SEP", "P", "P", (0, 0, 6)),
            ]
        )
        assert list(_solvation_types(atoms)) == ["N+", "N/O", "O-", "N/O", "O-", "N/O", ""]
