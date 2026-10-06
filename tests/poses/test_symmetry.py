"""The symmetric-atom table of the pose agreement: content, residue mapping and assignment."""

import itertools

import numpy as np
import pytest

from binding_metrics.core.nonstandard import D_AA_MAP, NME_AA_MAP
from binding_metrics.core.residues import STANDARD_AMINO_ACIDS
from binding_metrics.poses import symmetry
from binding_metrics.poses.symmetry import (
    C_TERMINAL_CARBOXYLATE,
    N_METHYL_PARENT,
    SYMMETRIC_ATOM_GROUPS,
    SwapGroup,
    best_labelling,
    count_swapped,
    resolve_group,
    symmetry_groups,
    symmetry_parent,
    symmetry_status,
)

SWAPPABLE_STANDARD = {"ASP", "GLU", "LEU", "VAL", "ARG", "PHE", "TYR"}
CODES_WITH_A_PARENT_ENTRY = sorted(
    code for code in set(D_AA_MAP) | set(N_METHYL_PARENT) if symmetry_groups(code)
)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


class TestTable:
    def test_the_standard_residues_with_symmetric_atoms_are_exactly_the_documented_ones(self):
        in_table = {name for name in STANDARD_AMINO_ACIDS if name in SYMMETRIC_ATOM_GROUPS}
        assert in_table == SWAPPABLE_STANDARD

    @pytest.mark.parametrize(
        ("residue", "atoms", "alternatives"),
        [
            ("ASP", ("OD1", "OD2"), (("OD2", "OD1"),)),
            ("GLU", ("OE1", "OE2"), (("OE2", "OE1"),)),
            ("LEU", ("CD1", "CD2"), (("CD2", "CD1"),)),
            ("VAL", ("CG1", "CG2"), (("CG2", "CG1"),)),
            ("ARG", ("NH1", "NH2"), (("NH2", "NH1"),)),
            ("PHE", ("CD1", "CD2", "CE1", "CE2"), (("CD2", "CD1", "CE2", "CE1"),)),
            ("TYR", ("CD1", "CD2", "CE1", "CE2"), (("CD2", "CD1", "CE2", "CE1"),)),
        ],
    )
    def test_entries_of_the_standard_residues(self, residue, atoms, alternatives):
        (group,) = SYMMETRIC_ATOM_GROUPS[residue]
        assert group.atoms == atoms
        assert group.alternatives == alternatives

    def test_a_ring_flip_is_one_group_and_not_two_independent_swaps(self):
        (group,) = SYMMETRIC_ATOM_GROUPS["PHE"]
        # no alternative exchanges only one of the two pairs
        assert len(group.alternatives) == 1
        assert group.alternatives[0][0] == "CD2" and group.alternatives[0][2] == "CE2"

    @pytest.mark.parametrize("residue", ["SEP", "TPO", "PTR"])
    def test_the_phosphate_oxygens_take_every_permutation(self, residue):
        phosphate = SYMMETRIC_ATOM_GROUPS[residue][-1]
        assert phosphate.atoms == ("O1P", "O2P", "O3P")
        assert len(phosphate.alternatives) == 5
        assert set(phosphate.alternatives) | {phosphate.atoms} == set(
            itertools.permutations(phosphate.atoms)
        )

    def test_a_phosphotyrosine_has_the_ring_flip_and_the_phosphate(self):
        assert [g.atoms for g in SYMMETRIC_ATOM_GROUPS["PTR"]] == [
            ("CD1", "CD2", "CE1", "CE2"),
            ("O1P", "O2P", "O3P"),
        ]

    def test_asn_gln_and_his_are_not_swapped(self):
        for residue in ("ASN", "GLN", "HIS"):
            assert symmetry_groups(residue) == ()
            assert symmetry_status(residue) == "none"

    def test_the_table_cannot_be_modified(self):
        with pytest.raises(TypeError):
            SYMMETRIC_ATOM_GROUPS["ALA"] = ()

    def test_the_carboxylate_group_is_not_a_residue_entry(self):
        # it depends on the position in the chain, so no residue has it in the table
        assert C_TERMINAL_CARBOXYLATE.atoms == ("O", "OXT")
        assert C_TERMINAL_CARBOXYLATE.alternatives == (("OXT", "O"),)
        for groups in SYMMETRIC_ATOM_GROUPS.values():
            assert C_TERMINAL_CARBOXYLATE not in groups
        assert symmetry_groups("ALA") == ()

    def test_every_alternative_is_a_reordering_of_the_group_atoms(self):
        for groups in SYMMETRIC_ATOM_GROUPS.values():
            for group in groups:
                for alternative in group.alternatives:
                    assert sorted(alternative) == sorted(group.atoms)
                    assert alternative != group.atoms


class TestTableAgainstTheChemicalComponentDictionary:
    """Each entry is an automorphism of the heavy-atom bond graph of its residue."""

    @pytest.fixture
    def info(self):
        pytest.importorskip("biotite")
        import biotite.structure.info as info

        return info

    def _graph(self, info, residue):
        template = info.residue(residue)
        template = template[template.element != "H"]
        names = [str(name) for name in template.atom_name]
        elements = dict(zip(names, (str(e).upper() for e in template.element)))
        bonds = {frozenset((names[int(a)], names[int(b)])) for a, b, _ in template.bonds.as_array()}
        return elements, bonds

    @pytest.mark.parametrize("residue", sorted(SYMMETRIC_ATOM_GROUPS))
    def test_the_atoms_exist_and_the_exchange_keeps_every_bond_and_element(self, info, residue):
        elements, bonds = self._graph(info, residue)
        for group in SYMMETRIC_ATOM_GROUPS[residue]:
            for name in group.atoms:
                assert name in elements, f"{name} is not an atom of {residue} in the CCD"
            for alternative in group.alternatives:
                image = dict(zip(group.atoms, alternative))
                for name, other in image.items():
                    assert elements[name] == elements[other]
                for bond in bonds:
                    mapped = frozenset(image.get(name, name) for name in bond)
                    assert mapped in bonds, f"{residue}: {sorted(bond)} is not kept by {image}"

    @pytest.mark.parametrize("residue", ["ALA", "ASN", "LEU", "PRO", "GLY", "SEP", "MLE"])
    def test_the_c_terminal_carboxylate_exchange_keeps_every_bond_and_element(self, info, residue):
        elements, bonds = self._graph(info, residue)
        image = dict(zip(C_TERMINAL_CARBOXYLATE.atoms, C_TERMINAL_CARBOXYLATE.alternatives[0]))
        assert set(image) == {"O", "OXT"} and set(image) <= set(elements)
        assert elements["O"] == elements["OXT"]
        for bond in bonds:
            assert frozenset(image.get(name, name) for name in bond) in bonds

    @pytest.mark.parametrize("residue", CODES_WITH_A_PARENT_ENTRY)
    def test_d_forms_and_n_methyl_codes_have_the_atoms_of_their_parents_entry(self, info, residue):
        try:
            elements, _ = self._graph(info, residue)
        except KeyError:
            pytest.skip(f"{residue} is not in the biotite component dictionary")
        for group in symmetry_groups(residue):
            assert set(group.atoms) <= set(elements)


# ---------------------------------------------------------------------------
# residue names
# ---------------------------------------------------------------------------


class TestResidueMapping:
    @pytest.mark.parametrize(
        ("name", "parent"),
        [
            ("LEU", "LEU"),
            ("leu", "LEU"),
            (" ASP ", "ASP"),
            ("DLE", "LEU"),
            ("DPN", "PHE"),
            ("DAR", "ARG"),
            ("MLE", "LEU"),
            ("MVA", "VAL"),
            ("SAR", "GLY"),
            ("NMG", "GLY"),
            ("NMA", "ALA"),
            ("MAA", "ALA"),
            ("ASH", "ASP"),
            ("GLH", "GLU"),
            ("HID", "HIS"),
            ("BMT", "BMT"),
        ],
    )
    def test_parent(self, name, parent):
        assert symmetry_parent(name) == parent

    def test_every_d_form_follows_its_l_parent(self):
        for code, parent in D_AA_MAP.items():
            assert symmetry_groups(code) == SYMMETRIC_ATOM_GROUPS.get(parent, ())

    def test_every_n_methyl_code_of_the_nonstandard_module_has_a_parent(self):
        assert set(N_METHYL_PARENT) == set(NME_AA_MAP)

    def test_n_methyl_leucine_and_valine_use_the_entry_of_their_parent(self):
        assert symmetry_groups("MLE") == SYMMETRIC_ATOM_GROUPS["LEU"]
        assert symmetry_groups("MVA") == SYMMETRIC_ATOM_GROUPS["VAL"]
        assert symmetry_groups("SAR") == ()

    @pytest.mark.parametrize("name", sorted(SWAPPABLE_STANDARD | {"SEP", "TPO", "PTR"}))
    def test_status_swappable(self, name):
        assert symmetry_status(name) == "swappable"

    @pytest.mark.parametrize("name", ["DLE", "MLE", "MVA", "ASH", "DAS", "DTY"])
    def test_status_swappable_through_a_parent(self, name):
        assert symmetry_status(name) == "swappable"

    @pytest.mark.parametrize(
        "name", ["ALA", "ASN", "CYS", "GLN", "GLY", "HIS", "ILE", "LYS", "MET", "PRO", "SER"]
    )
    def test_status_none_for_standard_residues_without_symmetric_atoms(self, name):
        assert symmetry_status(name) == "none"

    @pytest.mark.parametrize(
        "name", ["THR", "TRP", "DAL", "SAR", "NMA", "HIE", "MSE", "ACE", "NME"]
    )
    def test_status_none_for_known_residues_without_symmetric_atoms(self, name):
        assert symmetry_status(name) == "none"

    @pytest.mark.parametrize("name", ["BMT", "ABA", "ASPL", "GLUL", "LYSL", "MK8", "UNK", "XYZ"])
    def test_status_unknown_for_any_other_residue(self, name):
        assert symmetry_status(name) == "unknown"
        assert symmetry_groups(name) == ()


# ---------------------------------------------------------------------------
# the assignment
# ---------------------------------------------------------------------------


def _asp_pair():
    """Rows 0 CG, 1 OD1, 2 OD2 of a first pose; the second pose labels the oxygens the other way."""
    coord_a = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]])
    coord_b = coord_a[[0, 2, 1]]
    return coord_a, coord_b


def _resolved(residue, rows_by_name):
    return [resolve_group(group, rows_by_name) for group in SYMMETRIC_ATOM_GROUPS[residue]]


class TestAssignment:
    def test_resolve_group_returns_none_when_an_atom_is_missing(self):
        (group,) = SYMMETRIC_ATOM_GROUPS["ASP"]
        assert resolve_group(group, {"OD1": 0}) is None

    def test_resolve_group_locates_the_rows_and_the_alternatives(self):
        (group,) = SYMMETRIC_ATOM_GROUPS["ASP"]
        resolved = resolve_group(group, {"CG": 0, "OD1": 1, "OD2": 2})
        assert resolved.rows.tolist() == [1, 2]
        assert resolved.alternative_rows.tolist() == [[2, 1]]

    def test_the_best_labelling_undoes_a_swap(self):
        coord_a, coord_b = _asp_pair()
        groups = _resolved("ASP", {"CG": 0, "OD1": 1, "OD2": 2})
        order = best_labelling(coord_a, coord_b, groups)
        assert order.tolist() == [0, 2, 1]
        np.testing.assert_array_equal(coord_b[order], coord_a)
        assert count_swapped(order, groups) == 1

    def test_the_identity_wins_when_it_is_already_best(self):
        coord_a, _ = _asp_pair()
        groups = _resolved("ASP", {"CG": 0, "OD1": 1, "OD2": 2})
        order = best_labelling(coord_a, coord_a.copy(), groups)
        assert order.tolist() == [0, 1, 2]
        assert count_swapped(order, groups) == 0

    def test_a_tie_goes_to_the_identity(self):
        # both oxygens of the second pose sit at the same point: the two labellings cost the same
        coord_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        coord_b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        groups = _resolved("ASP", {"CG": 0, "OD1": 1, "OD2": 2})
        assert best_labelling(coord_a, coord_b, groups).tolist() == [0, 1, 2]

    def test_the_inputs_are_not_modified(self):
        coord_a, coord_b = _asp_pair()
        before = coord_b.copy()
        best_labelling(coord_a, coord_b, _resolved("ASP", {"CG": 0, "OD1": 1, "OD2": 2}))
        np.testing.assert_array_equal(coord_b, before)

    def test_the_phosphate_takes_the_best_of_six_permutations(self):
        coord_a = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        # a labelling that is a 3-cycle of the first
        coord_b = coord_a[[1, 2, 0]]
        groups = _resolved("SEP", {"O1P": 0, "O2P": 1, "O3P": 2})
        order = best_labelling(coord_a, coord_b, groups)
        np.testing.assert_array_equal(coord_b[order], coord_a)
        assert count_swapped(order, groups) == 1

    def test_a_ring_flip_moves_both_pairs_together(self):
        # rows 0..3 are CD1, CD2, CE1, CE2 of a ring in a plane
        coord_a = np.array([[1.2, 0.7, 0], [1.2, -0.7, 0], [2.4, 1.2, 0], [2.4, -1.2, 0]])
        flipped = coord_a[[1, 0, 3, 2]]
        groups = _resolved("PHE", {"CD1": 0, "CD2": 1, "CE1": 2, "CE2": 3})
        np.testing.assert_array_equal(flipped[best_labelling(coord_a, flipped, groups)], coord_a)
        # the swap of one pair only is not a labelling of the table: it cannot be undone
        half = coord_a[[1, 0, 2, 3]]
        order = best_labelling(coord_a, half, groups)
        assert np.sum((coord_a - half[order]) ** 2) > 0.0

    def test_groups_are_chosen_independently(self):
        coord_a = np.array(
            [[0.0, 0, 0], [1.2, 0, 0], [-1.2, 0, 0], [10, 0, 0], [11.2, 0, 0], [8.8, 0, 0]]
        )
        # the second residue is swapped, the first is not
        coord_b = coord_a[[0, 1, 2, 3, 5, 4]]
        groups = _resolved("ASP", {"CG": 0, "OD1": 1, "OD2": 2}) + _resolved(
            "GLU", {"CD": 3, "OE1": 4, "OE2": 5}
        )
        order = best_labelling(coord_a, coord_b, groups)
        assert order.tolist() == [0, 1, 2, 3, 5, 4]
        assert count_swapped(order, groups) == 1


def test_the_module_documents_where_the_table_comes_from():
    doc = symmetry.__doc__
    for phrase in ("Chemical Component", "Leu CD1/CD2", "guanidinium", "ring turns"):
        assert phrase.lower() in doc.lower()


def test_swap_group_is_hashable_and_immutable():
    group = SwapGroup(("A", "B"), (("B", "A"),))
    assert hash(group) == hash(SwapGroup(("A", "B"), (("B", "A"),)))
    with pytest.raises(AttributeError):
        group.atoms = ("C",)
