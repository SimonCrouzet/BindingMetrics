"""``profile_input`` on small synthetic structures: closures, residue classes, size and type."""

from __future__ import annotations

import json

import numpy as np
import pytest

from binding_metrics.capabilities import (
    InputProfile,
    classify_residue,
    detect_closures,
    estimate_binder_type,
    profile_input,
)
from tests.test_pre_structures import add_waters_and_ion, build_chain, head_to_tail_ring

struc = pytest.importorskip("biotite.structure")

LINEAR = ["ALA", "GLY", "CYS", "ASP", "LYS", "GLU", "ALA", "GLY", "CYS", "LYS"]


def _kinds(atoms, chain="B"):
    return sorted(c.kind for c in detect_closures(atoms, chain))


class TestLinearBinder:
    def test_a_linear_peptide_has_no_closure_and_canonical_residues(self):
        profile = profile_input(build_chain(LINEAR), "B")
        assert profile.closures == frozenset({"none"})
        assert profile.closure_bonds == ()
        assert profile.residue_classes == frozenset({"canonical"})
        assert profile.n_binder_residues == 10
        assert profile.binder_type == "peptide"
        assert profile.binder_type_source == "estimated"
        assert profile.receptor_chain is None

    def test_waters_and_ions_under_the_chain_id_are_not_residues_of_the_binder(self):
        atoms = add_waters_and_ion(build_chain(LINEAR), n_waters=4)
        profile = profile_input(atoms, "B")
        assert profile.n_binder_residues == 10
        assert profile.residue_classes == frozenset({"canonical"})
        assert profile.closures == frozenset({"none"})

    def test_a_cysteine_pair_out_of_reach_and_without_a_bond_is_not_a_disulfide(self):
        atoms = build_chain(LINEAR, close=[(8, "SG", 2, "SG", 3.4)])
        assert profile_input(atoms, "B").closures == frozenset({"none"})

    def test_a_single_residue_has_no_closure(self):
        assert profile_input(build_chain(["ALA"]), "B").closures == frozenset({"none"})


class TestClosures:
    def test_head_to_tail_ring(self):
        profile = profile_input(head_to_tail_ring(LINEAR), "B")
        assert profile.closures == frozenset({"head_to_tail"})
        (closure,) = profile.closure_bonds
        assert closure.kind == "head_to_tail"
        assert (closure.end1.residue_index, closure.end1.atom_name) == (9, "C")
        assert (closure.end2.residue_index, closure.end2.atom_name) == (0, "N")

    def test_disulfide_by_distance(self):
        atoms = build_chain(LINEAR, close=[(8, "SG", 2, "SG", 2.05)])
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"disulfide"})
        (closure,) = profile.closure_bonds
        assert (closure.end1.residue_number, closure.end2.residue_number) == (3, 9)

    @pytest.mark.parametrize("distance, found", [(1.95, True), (2.05, False)])
    def test_the_amide_cut_off_is_2_angstrom(self, distance, found):
        atoms = build_chain(LINEAR, close=[(9, "C", 0, "N", distance)])
        assert (profile_input(atoms, "B").closures == frozenset({"head_to_tail"})) is found

    @pytest.mark.parametrize("distance, found", [(2.55, True), (2.65, False)])
    def test_the_disulfide_cut_off_is_2_6_angstrom(self, distance, found):
        atoms = build_chain(LINEAR, close=[(8, "SG", 2, "SG", distance)])
        assert (profile_input(atoms, "B").closures == frozenset({"disulfide"})) is found

    def test_disulfide_declared_in_the_bond_table_but_stretched(self):
        atoms = build_chain(LINEAR, bonds=[(2, "SG", 8, "SG")])
        assert profile_input(atoms, "B").closures == frozenset({"disulfide"})

    def test_a_disulfide_between_amber_cyx_residues_counts(self):
        names = ["ALA", "CYX", "GLY", "GLY", "CYX", "ALA"]
        atoms = build_chain(names, close=[(4, "SG", 1, "SG", 2.05)])
        assert _kinds(atoms) == ["disulfide"]

    def test_bicyclic_head_to_tail_and_disulfide(self):
        atoms = head_to_tail_ring(LINEAR, close=[(8, "SG", 2, "SG", 2.05)])
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"head_to_tail", "disulfide"})
        assert len(profile.closure_bonds) == 2

    @pytest.mark.parametrize(
        "names, close, kind",
        [
            (["GLY", "ASP", "GLY", "GLY", "ALA"], [(1, "CG", 0, "N", 1.35)], "lactam_n_asp"),
            (["GLY", "GLU", "GLY", "GLY", "ALA"], [(1, "CD", 0, "N", 1.35)], "lactam_n_glu"),
            (["GLY", "ALA", "LYS", "GLY", "ALA"], [(2, "NZ", 4, "C", 1.35)], "lactam_c_lys"),
            (["GLY", "LYS", "GLY", "GLY", "ASP"], [(1, "NZ", 4, "CG", 1.35)], "lactam_sc_lys_asp"),
            (["GLY", "LYS", "GLY", "GLY", "GLU"], [(1, "NZ", 4, "CD", 1.35)], "lactam_sc_lys_glu"),
            (["GLY", "GLU", "GLY", "GLY", "LYS"], [(4, "NZ", 1, "CD", 1.35)], "lactam_sc_lys_glu"),
        ],
    )
    def test_lactams(self, names, close, kind):
        atoms = build_chain(names, close=close)
        assert _kinds(atoms) == [kind]
        assert profile_input(atoms, "B").closures == frozenset({"lactam"})

    def test_a_lysine_that_closes_on_the_c_terminus_is_not_counted_twice_as_a_staple(self):
        atoms = build_chain(
            ["GLY", "ALA", "LYS", "GLY", "ALA"],
            close=[(2, "NZ", 4, "C", 1.35)],
            bonds=[(2, "NZ", 4, "C")],
        )
        assert _kinds(atoms) == ["lactam_c_lys"]

    def test_hydrocarbon_staple_from_the_bond_table(self):
        names = ["ALA", "GLY", "S5A", "GLY", "GLY", "GLY", "R8A", "ALA"]
        atoms = build_chain(
            names,
            side_chain_atoms={"S5A": ("CB", "CE"), "R8A": ("CB", "CT")},
            bonds=[(2, "CE", 6, "CT")],
        )
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"staple"})
        assert profile.closure_bonds[0].kind == "hydrocarbon_staple"
        assert profile.residue_classes == frozenset({"canonical", "other_ncaa"})
        assert profile.residue_names["other_ncaa"] == ("R8A", "S5A")

    def test_a_link_that_is_not_all_carbon_is_another_cross_link(self):
        names = ["ALA", "CYS", "GLY", "GLY", "GLY", "XAA", "ALA"]
        atoms = build_chain(
            names,
            side_chain_atoms={"XAA": ("CB", "CE")},
            bonds=[(1, "SG", 5, "CE")],
        )
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"other"})
        assert profile.closure_bonds[0].kind == "unsupported_crosslink"

    def test_a_side_chain_to_backbone_link_is_another_cross_link(self):
        atoms = build_chain(
            ["ALA", "GLY", "ALA", "GLY", "ALA"],
            bonds=[(0, "CB", 3, "N")],
        )
        assert profile_input(atoms, "B").closures == frozenset({"other"})

    def test_neighbouring_residues_are_never_a_cross_link(self):
        atoms = build_chain(LINEAR, bonds=[(2, "C", 3, "N"), (3, "CB", 4, "N")])
        assert profile_input(atoms, "B").closures == frozenset({"none"})

    def test_a_bond_table_is_needed_for_staples_and_the_profile_says_so(self):
        names = ["ALA", "GLY", "S5A", "GLY", "GLY", "GLY", "R8A", "ALA"]
        atoms = build_chain(
            names, side_chain_atoms={"S5A": ("CB", "CE"), "R8A": ("CB", "CT")}, bond_table=False
        )
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"none"})
        assert any("no bond table" in note for note in profile.notes)

    def test_a_metal_coordination_bond_is_not_a_closure(self):
        atoms = build_chain(LINEAR)
        atoms.bonds = struc.BondList(
            atoms.array_length(),
            np.array([[0, atoms.array_length() - 1, struc.BondType.COORDINATION]]),
        )
        assert profile_input(atoms, "B").closures == frozenset({"none"})

    def test_a_missing_terminal_atom_is_noted_and_does_not_hide_the_disulfide(self):
        atoms = build_chain(LINEAR, close=[(8, "SG", 2, "SG", 2.05)])
        atoms = atoms[~((atoms.res_id == 1) & (atoms.atom_name == "N"))]
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"disulfide"})
        assert any("no atom N" in note for note in profile.notes)


class TestResidueClasses:
    def test_every_class_is_recognised(self):
        names = ["ACE", "ALA", "DAL", "SAR", "MLE", "SEP", "BMT", "XYZ", "GSH", "NME"]
        atoms = build_chain(
            names,
            residue_atoms={
                "ACE": ("C", "O", "CH3"),
                "NME": ("N", "C"),
                "GSH": ("S1", "C1", "O1"),
            },
        )
        profile = profile_input(atoms, "B")
        assert profile.residue_classes == frozenset(
            {"canonical", "d_amino", "n_methyl", "phospho", "other_ncaa", "cap", "ligand"}
        )
        assert profile.residue_names["n_methyl"] == ("MLE", "SAR")
        assert profile.residue_names["other_ncaa"] == ("BMT", "XYZ")
        assert profile.residue_names["cap"] == ("ACE", "NME")
        assert profile.residue_names["ligand"] == ("GSH",)
        # ACE, NME and the glutathione are not residues of the sequence
        assert profile.n_binder_residues == 7

    def test_caps_do_not_move_the_ends_of_the_chain(self):
        names = ["ACE", "ALA", "GLY", "GLY", "ALA", "NME"]
        atoms = build_chain(
            names,
            residue_atoms={"ACE": ("C", "O", "CH3"), "NME": ("N", "C")},
            close=[(4, "C", 1, "N", 1.33)],
        )
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"head_to_tail"})
        assert profile.closure_bonds[0].end1.residue_index == 3  # the last amino acid

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("ALA", "canonical"),
            ("HIE", "canonical"),
            ("CYX", "canonical"),
            ("HSD", "canonical"),
            ("DAL", "d_amino"),
            ("DTR", "d_amino"),
            ("SAR", "n_methyl"),
            ("MVA", "n_methyl"),
            ("SEP", "phospho"),
            ("PTR", "phospho"),
            ("ACE", "cap"),
            ("NH2", "cap"),
            ("HOH", None),
            ("SOL", None),
        ],
    )
    def test_classify_residue_by_name(self, name, expected):
        assert classify_residue(name) == expected

    def test_an_unlisted_name_is_an_ncaa_or_a_ligand_by_its_backbone(self):
        assert classify_residue("BMT", is_amino_acid=True) == "other_ncaa"
        assert classify_residue("BMT", is_amino_acid=False) == "ligand"

    def test_only_the_binder_chain_is_classified(self):
        binder = build_chain(["ALA", "GLY", "ALA"], "B")
        receptor = build_chain(["DAL", "SEP", "GLY"], "A", x_offset=500.0)
        profile = profile_input(binder + receptor, "B", "A")
        assert profile.residue_classes == frozenset({"canonical"})


class TestSizeAndType:
    @pytest.mark.parametrize(
        "n, expected",
        [(1, "peptide"), (40, "peptide"), (41, "miniprotein"), (100, "miniprotein")],
    )
    def test_the_size_classes(self, n, expected):
        assert estimate_binder_type(n) == expected
        atoms = build_chain(["ALA"] * n)
        assert profile_input(atoms, "B").binder_type == expected

    def test_a_long_binder_has_an_unknown_type_and_a_note(self):
        profile = profile_input(build_chain(["ALA"] * 101), "B")
        assert estimate_binder_type(101) == "unknown"
        assert profile.binder_type == "unknown"
        assert profile.binder_type_source == "estimated"
        assert any("skipped" in note for note in profile.notes)

    def test_a_given_type_is_kept_whatever_the_size(self):
        profile = profile_input(build_chain(["ALA"] * 120), "B", binder_type="nanobody")
        assert (profile.binder_type, profile.binder_type_source) == ("nanobody", "given")
        assert profile.notes == ()

    def test_a_wrong_type_is_refused(self):
        with pytest.raises(ValueError, match="binder_type"):
            profile_input(build_chain(LINEAR), "B", binder_type="protein")
        with pytest.raises(ValueError, match="binder_type"):
            profile_input(build_chain(LINEAR), "B", binder_type="unknown")


class TestChains:
    def test_receptor_and_chain_ids_are_recorded(self):
        atoms = build_chain(LINEAR, "B") + build_chain(["ALA"] * 30, "A", x_offset=900.0)
        profile = profile_input(atoms, "B", "A")
        assert profile.receptor_chain == "A"
        assert profile.chain_ids == ("B", "A")
        assert profile.n_chains == 2

    def test_a_binder_of_two_chains_adds_up(self):
        atoms = build_chain(LINEAR, "H") + build_chain(["ALA"] * 5, "L", x_offset=900.0)
        profile = profile_input(atoms, ("H", "L"))
        assert profile.binder_chains == ("H", "L")
        assert profile.binder_chain == "H"
        assert profile.n_binder_residues == 15

    def test_an_unknown_binder_chain_lists_the_chains(self):
        with pytest.raises(ValueError, match=r"chain 'Z' not found.*\['B'\]"):
            profile_input(build_chain(LINEAR), "Z")

    def test_an_unknown_receptor_chain_is_refused(self):
        with pytest.raises(ValueError, match="receptor chain 'Q' not found"):
            profile_input(build_chain(LINEAR), "B", "Q")

    def test_a_repeated_or_empty_binder_chain_is_refused(self):
        with pytest.raises(ValueError, match="distinct"):
            profile_input(build_chain(LINEAR), ("B", "B"))
        with pytest.raises(ValueError, match="distinct"):
            profile_input(build_chain(LINEAR), ())


class TestInputs:
    def test_a_stack_is_read_as_its_first_model(self):
        first = head_to_tail_ring(LINEAR)
        second = build_chain(LINEAR)
        stack = struc.stack([first, second])
        assert profile_input(stack, "B").closures == frozenset({"head_to_tail"})

    def test_a_pdb_path(self, tmp_path):
        import biotite.structure.io.pdb as pdb_io

        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, head_to_tail_ring(LINEAR))
        path = tmp_path / "ring.pdb"
        pdb_file.write(str(path))
        profile = profile_input(path, "B")
        assert profile.closures == frozenset({"head_to_tail"})
        assert profile.n_binder_residues == 10

    def test_a_cif_path_given_as_a_string(self, tmp_path):
        import biotite.structure.io.pdbx as pdbx

        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, build_chain(LINEAR, close=[(8, "SG", 2, "SG", 2.05)]))
        path = tmp_path / "loop.cif"
        cif_file.write(str(path))
        profile = profile_input(str(path), "B")
        assert profile.closures == frozenset({"disulfide"})

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            profile_input(tmp_path / "absent.pdb", "B")


class TestProfileObject:
    def test_to_dict_is_json_ready(self):
        atoms = head_to_tail_ring(LINEAR, close=[(8, "SG", 2, "SG", 2.05)])
        text = json.dumps(profile_input(atoms, "B").to_dict())
        loaded = json.loads(text)
        assert loaded["closures"] == ["head_to_tail", "disulfide"]
        assert loaded["binder_chains"] == ["B"]
        assert len(loaded["closure_bonds"]) == 2

    def test_describe_names_the_facts(self):
        text = profile_input(head_to_tail_ring(LINEAR), "B").describe()
        for part in ("binder chain B", "10 residues", "peptide", "head_to_tail", "canonical"):
            assert part in text

    def test_a_hand_built_profile_needs_only_the_binder_chain(self):
        profile = InputProfile("B")
        assert profile.closures == frozenset({"none"}) and profile.binder_type == "unknown"
        assert InputProfile(binder_chains="B").binder_chains == ("B",)

    def test_an_unknown_binder_type_is_refused(self):
        with pytest.raises(ValueError, match="binder_type"):
            InputProfile("B", binder_type="protein")

    def test_the_profile_is_immutable(self):
        profile = profile_input(build_chain(LINEAR), "B")
        with pytest.raises(AttributeError):
            profile.binder_type = "antibody"
