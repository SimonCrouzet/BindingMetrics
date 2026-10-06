"""Reference, Partner and ResidueKey: construction, validation, identity and serialisation."""

import dataclasses
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose
from binding_metrics.poses.reference import (
    RESERVED_REGION_NAMES,
    Partner,
    Reference,
    ResidueKey,
    group_by_frame,
    normalize_references,
    polymer_residues,
)

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"

# The 22 MDM2 residues within 4.5 angstrom of the crystal peptide (design note, section 1.2).
YCR_EPITOPE = [25, 26, 50, 51, 54, 57, 58, 61, 62, 67, 70, 71, 72, 73, 75, 91, 93, 94, 96, 99]
YCR_EPITOPE += [100, 104]


@pytest.fixture(scope="module")
def ycr_atoms():
    from binding_metrics.metrics._common import load_structure

    atoms = load_structure(YCR)
    atoms.coord.flags.writeable = False
    return atoms


# ---------------------------------------------------------------------------
# ResidueKey
# ---------------------------------------------------------------------------


class TestResidueKey:
    def test_fields_default_icode_and_label(self):
        key = ResidueKey("A", 52)
        assert (key.chain, key.number, key.icode) == ("A", 52, "")
        assert key.label == "A:52" == str(key)
        assert ResidueKey("A", 52, "B").label == "A:52B"

    def test_insertion_code_makes_a_different_residue(self):
        assert ResidueKey("A", 52) != ResidueKey("A", 52, "A")
        assert len({ResidueKey("A", 52), ResidueKey("A", 52, "A"), ResidueKey("A", 52)}) == 2

    def test_sorts_by_chain_number_icode(self):
        keys = [
            ResidueKey("B", 1),
            ResidueKey("A", 52, "A"),
            ResidueKey("A", 52),
            ResidueKey("A", 9),
        ]
        assert [k.label for k in sorted(keys)] == ["A:9", "A:52", "A:52A", "B:1"]

    def test_numpy_integer_becomes_int_and_blank_icode_is_empty(self):
        key = ResidueKey("A", np.int32(7), " ")
        assert type(key.number) is int and key.icode == ""

    def test_negative_numbers_are_residues(self):
        assert ResidueKey.parse("A:-3") == ResidueKey("A", -3)
        assert ResidueKey.parse("A:-3B") == ResidueKey("A", -3, "B")

    @pytest.mark.parametrize(
        "args, error, match",
        [
            ((1, 5), TypeError, "chain must be a string"),
            (("", 5), ValueError, "chain must not be empty"),
            (("A", True), TypeError, "must be an integer"),
            (("A", 5.0), TypeError, "must be an integer"),
            (("A", 5, 1), TypeError, "insertion code must be a string"),
            (("A", 5, "AB"), ValueError, "one character"),
        ],
    )
    def test_invalid_fields(self, args, error, match):
        with pytest.raises(error, match=match):
            ResidueKey(*args)

    def test_coerce_forms(self):
        expected = ResidueKey("A", 52, "B")
        assert ResidueKey.coerce(expected) is expected
        assert ResidueKey.coerce("A:52B") == expected
        assert ResidueKey.coerce(("A", 52, "B")) == expected
        assert ResidueKey.coerce(["A", 52, "B"]) == expected
        assert ResidueKey.coerce(("A", 52)) == ResidueKey("A", 52)

    @pytest.mark.parametrize("bad", ["A52", "A:", ":52", "A:5B2", 52, ("A",), None])
    def test_coerce_rejects_other_forms(self, bad):
        with pytest.raises((TypeError, ValueError), match="residue key"):
            ResidueKey.coerce(bad)

    def test_list_round_trip(self):
        key = ResidueKey("A", 60, "C")
        assert key.to_list() == ["A", 60, "C"]
        assert ResidueKey.from_list(key.to_list()) == key
        with pytest.raises(ValueError, match="chain, number, icode"):
            ResidueKey.from_list(["A", 60])

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ResidueKey("A", 1).number = 2


# ---------------------------------------------------------------------------
# polymer_residues
# ---------------------------------------------------------------------------


class TestPolymerResidues:
    def test_one_entry_per_residue_with_unique_keys(self, ycr_atoms):
        receptor = polymer_residues(ycr_atoms, "A")
        peptide = polymer_residues(ycr_atoms, "B")
        assert len(receptor) == 85 and len(peptide) == 13
        assert [r.key.number for r in receptor] == list(range(25, 110))
        assert len({r.key for r in receptor}) == 85
        first = receptor[0]
        assert ycr_atoms.atom_name[first.ca_index] == "CA"
        assert set(ycr_atoms.res_id[first.atom_indices]) == {25}

    def test_insertion_codes_are_separate_residues(self):
        from binding_metrics.metrics._common import load_structure

        atoms = load_structure(SFTI)
        keys = [r.key for r in polymer_residues(atoms, "A")]
        assert ResidueKey("A", 60) in keys
        assert {ResidueKey("A", 60, code) for code in "ABCDEFGHI"} <= set(keys)
        assert len(keys) == len(set(keys))

    def test_absent_chain_and_non_polymer_chain(self, ycr_atoms):
        assert polymer_residues(ycr_atoms, "Z") == []
        struc = pytest.importorskip("biotite.structure")
        ion = struc.AtomArray(1)
        ion.chain_id[:] = "I"
        ion.res_id[:] = 1
        ion.res_name[:] = "CA"
        ion.atom_name[:] = "CA"
        ion.element[:] = "CA"
        assert polymer_residues(ion, "I") == []


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_path_array_and_pose_give_one_content(self, ycr_atoms):
        from_path = Reference(YCR, "A", "B")
        from_str = Reference(str(YCR), "A", "B")
        from_pose = Reference(ExperimentalPose.from_file(YCR), "A", "B")
        assert from_path.content_hash() == from_str.content_hash() == from_pose.content_hash()
        from_array = Reference(ycr_atoms, "A", "B")
        assert from_array.content_hash() != from_path.content_hash()  # a file and its atoms differ
        assert from_array.atoms().array_length() == ycr_atoms.array_length()

    def test_a_bare_structure_is_stored_as_an_experimental_pose(self, ycr_atoms):
        assert isinstance(Reference(YCR, "A").pose, ExperimentalPose)
        assert isinstance(Reference(ycr_atoms, "A").structure, ExperimentalPose)

    def test_a_given_pose_keeps_its_kind_and_chain_map(self):
        pose = DockedPose.from_file(YCR, chain_map={"A": "R", "B": "P"})
        reference = Reference(pose, "R", "P")
        assert reference.pose is pose
        assert set(reference.atoms().chain_id) == {"R", "P"}

    def test_construction_reads_no_structure(self, tmp_path):
        script = textwrap.dedent(
            f"""
            import sys
            sys.modules["biotite"] = None
            from binding_metrics.poses.reference import Reference, Partner
            r = Reference({str(YCR)!r}, "A", "B", residue_sets={{"s": ["A:25"]}},
                          partners={{"p": {str(YCR)!r}}})
            r.to_dict()
            r.content_hash()
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_defaults(self):
        reference = Reference(YCR, "A")
        assert reference.name == "reference"
        assert reference.binder_chain is None
        assert dict(reference.residue_map) == {}
        assert dict(reference.residue_sets) == {}
        assert dict(reference.partners) == {}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="reference structure file not found"):
            Reference(tmp_path / "absent.pdb", "A")

    def test_wrong_structure_type(self):
        with pytest.raises(TypeError, match="structure path or a biotite AtomArray"):
            Reference(12, "A")

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            ({"receptor_chain": 1}, TypeError, "receptor_chain must be a chain ID"),
            ({"receptor_chain": ""}, ValueError, "receptor_chain must not be empty"),
            ({"binder_chain": "A"}, ValueError, "are both 'A'"),
            ({"binder_chain": ""}, ValueError, "binder_chain must not be empty"),
            ({"binder_chain": 3}, TypeError, "binder_chain must be"),
            ({"name": ""}, ValueError, "name must not be empty"),
            ({"name": 3}, TypeError, "name must be a string"),
        ],
    )
    def test_chain_and_name_checks(self, kwargs, error, match):
        arguments = {"structure": YCR, "receptor_chain": "A"} | kwargs
        with pytest.raises(error, match=match):
            Reference(**arguments)

    def test_residue_sets_are_normalised(self):
        reference = Reference(
            YCR,
            "A",
            residue_sets={
                "pocket": ["A:50", ("A", 51), ResidueKey("A", 50)],
                "epitope": {ResidueKey("A", 25)},
            },
        )
        assert list(reference.residue_sets) == ["epitope", "pocket"]  # sorted by name
        assert reference.residue_sets["pocket"] == frozenset(
            {ResidueKey("A", 50), ResidueKey("A", 51)}
        )

    @pytest.mark.parametrize("name", RESERVED_REGION_NAMES)
    def test_reserved_set_names(self, name):
        with pytest.raises(ValueError, match="reserved"):
            Reference(YCR, "A", residue_sets={name: ["A:50"]})

    def test_residue_set_errors(self):
        with pytest.raises(ValueError, match="'empty' is empty"):
            Reference(YCR, "A", residue_sets={"empty": []})
        with pytest.raises(TypeError, match="collection of ResidueKey"):
            Reference(YCR, "A", residue_sets={"s": "A:50"})
        with pytest.raises(ValueError, match="off the receptor chain 'A'"):
            Reference(YCR, "A", "B", residue_sets={"s": ["B:20"]})
        with pytest.raises(ValueError, match="non-empty strings"):
            Reference(YCR, "A", residue_sets={"": ["A:50"]})
        with pytest.raises(TypeError, match="mapping of set name"):
            Reference(YCR, "A", residue_sets=[("s", ["A:50"])])

    def test_residue_map_is_normalised_and_one_to_one(self):
        reference = Reference(YCR, "A", residue_map={("A", 2): "A:26", "A:1": ("A", 25)})
        assert dict(reference.residue_map) == {
            ResidueKey("A", 1): ResidueKey("A", 25),
            ResidueKey("A", 2): ResidueKey("A", 26),
        }
        with pytest.raises(ValueError, match="not one-to-one"):
            Reference(YCR, "A", residue_map={"A:1": "A:25", "A:2": "A:25"})
        with pytest.raises(ValueError, match="must be on the receptor chain"):
            Reference(YCR, "A", residue_map={"A:1": "B:20"})
        with pytest.raises(TypeError, match="mapping of pose residue"):
            Reference(YCR, "A", residue_map=[("A:1", "A:25")])

    def test_partners_accept_paths_poses_and_partners(self, ycr_atoms):
        reference = Reference(
            YCR,
            "A",
            partners={
                "from_path": YCR,
                "from_array": ycr_atoms,
                "from_pose": DockedPose.from_file(YCR),
                "explicit": Partner(YCR, chains="B"),
            },
        )
        assert list(reference.partners) == ["explicit", "from_array", "from_path", "from_pose"]
        assert all(isinstance(p, Partner) for p in reference.partners.values())
        with pytest.raises(ValueError, match="partner names"):
            Reference(YCR, "A", partners={"": YCR})
        with pytest.raises(TypeError, match="mapping of partner name"):
            Reference(YCR, "A", partners=[YCR])

    def test_frozen_and_read_only_views(self):
        reference = Reference(YCR, "A", residue_sets={"s": ["A:25"]})
        with pytest.raises(dataclasses.FrozenInstanceError):
            reference.name = "other"
        with pytest.raises(TypeError):
            reference.residue_sets["t"] = frozenset()
        with pytest.raises(TypeError):
            reference.residue_map[ResidueKey("A", 1)] = ResidueKey("A", 2)
        with pytest.raises(AttributeError):
            reference.residue_sets["s"].add(ResidueKey("A", 26))

    def test_the_arguments_are_copied(self):
        members = [ResidueKey("A", 25)]
        sets = {"s": members}
        reference = Reference(YCR, "A", residue_sets=sets)
        members.append(ResidueKey("A", 26))
        sets["t"] = [ResidueKey("A", 27)]
        assert list(reference.residue_sets) == ["s"]
        assert reference.residue_sets["s"] == frozenset({ResidueKey("A", 25)})

    def test_repr_is_short(self):
        reference = Reference(YCR, "A", "B", residue_sets={"epitope": ["A:25", "A:26"]})
        text = repr(reference)
        assert "residue_sets={'epitope': 2}" in text and "binder_chain='B'" in text


# ---------------------------------------------------------------------------
# validate against the structure
# ---------------------------------------------------------------------------


class TestValidate:
    def test_a_good_reference_passes(self):
        Reference(
            YCR,
            "A",
            "B",
            residue_sets={"epitope": [("A", n) for n in YCR_EPITOPE]},
            residue_map={("A", 1): ("A", 25)},
            partners={"peptide": Partner(YCR, "B")},
        ).validate()

    def test_insertion_codes_are_checked(self):
        good = Reference(SFTI, "A", "I", residue_sets={"loop": ["A:60", "A:60B", "A:221A"]})
        good.validate()
        bad = Reference(SFTI, "A", "I", residue_sets={"loop": ["A:60", "A:61B"]})
        with pytest.raises(ValueError, match="1 of 2 residues.*A:61B"):
            bad.validate()

    def test_missing_chains_are_listed_with_what_the_structure_has(self):
        with pytest.raises(ValueError, match=r"receptor chain 'X' is not in .*chains \['A', 'B'\]"):
            Reference(YCR, "X").validate()
        with pytest.raises(ValueError, match=r"binder chain 'Q' is not in"):
            Reference(YCR, "A", "Q").validate()

    def test_every_problem_is_reported_in_one_error(self):
        reference = Reference(
            YCR,
            "A",
            residue_sets={"s": ["A:1", "A:2", "A:25"], "t": ["A:300"]},
            residue_map={"A:1": "A:999"},
            partners={"p": Partner(YCR, "Z")},
        )
        with pytest.raises(ValueError) as error:
            reference.validate()
        text = str(error.value)
        assert "reference 'reference' is not valid" in text
        assert "residue set 's' names 2 of 3 residues" in text and "A:1, A:2" in text
        assert "residue set 't' names 1 of 1" in text
        assert "residue_map names 1 of 1 residues" in text
        assert "partner 'p': partner chains ['Z']" in text

    def test_a_receptor_chain_without_polymer_residues(self, ycr_atoms):
        struc = pytest.importorskip("biotite.structure")
        water = struc.AtomArray(1)
        water.chain_id[:] = "W"
        water.res_id[:] = 1
        water.res_name[:] = "HOH"
        water.atom_name[:] = "O"
        water.element[:] = "O"
        with pytest.raises(ValueError, match="has no polymer residues"):
            Reference(ycr_atoms + water, "W").validate()


# ---------------------------------------------------------------------------
# Partner
# ---------------------------------------------------------------------------


class TestPartner:
    def test_chain_selection(self, ycr_atoms):
        only_b = Partner(YCR, chains="B").atoms()
        assert set(only_b.chain_id) == {"B"} and only_b.array_length() == 113
        both = Partner(YCR, chains=("A", "B")).atoms()
        assert both.array_length() == ycr_atoms.array_length()

    def test_default_is_every_polymer_chain(self, ycr_atoms):
        struc = pytest.importorskip("biotite.structure")
        water = struc.AtomArray(2)
        water.chain_id[:] = "W"
        water.res_id[:] = [1, 2]
        water.res_name[:] = "HOH"
        water.atom_name[:] = "O"
        water.element[:] = "O"
        atoms = Partner(ycr_atoms + water).atoms()
        assert set(atoms.chain_id) == {"A", "B"}

    def test_the_structure_is_not_modified(self, ycr_atoms):
        before = ycr_atoms.coord.copy()
        partner = Partner(ycr_atoms, "B")
        selected = partner.atoms()
        selected.coord[:] = 0.0
        assert np.array_equal(ycr_atoms.coord, before)

    def test_errors(self):
        with pytest.raises(ValueError, match=r"partner chains \['Z'\] are not in"):
            Partner(YCR, "Z").atoms()
        with pytest.raises(ValueError, match="empty or repeats|non-empty|repeats"):
            Partner(YCR, ["B", "B"])
        with pytest.raises(ValueError, match="non-empty chain ID"):
            Partner(YCR, [])
        with pytest.raises(TypeError, match="chain ID or a sequence"):
            Partner(YCR, 5)
        with pytest.raises(FileNotFoundError, match="partner structure file not found"):
            Partner("absent.pdb")

    def test_hash_depends_on_structure_and_chains(self):
        assert Partner(YCR, "B").content_hash() == Partner(str(YCR), ["B"]).content_hash()
        assert Partner(YCR, "B").content_hash() != Partner(YCR, "A").content_hash()
        assert Partner(YCR).content_hash() != Partner(YCR, "B").content_hash()

    def test_dict_round_trip(self):
        partner = Partner(YCR, ("A", "B"))
        data = partner.to_dict()
        json.dumps(data)
        again = Partner.from_dict(data)
        assert again.to_dict() == data and again.content_hash() == partner.content_hash()
        with pytest.raises(ValueError, match="unknown keys"):
            Partner.from_dict({**data, "extra": 1})


# ---------------------------------------------------------------------------
# identity: content hash and frame key
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_hash_is_stable_and_ignores_the_name_and_the_order(self):
        one = Reference(
            YCR, "A", "B", residue_sets={"a": ["A:25", "A:26"], "b": ["A:50"]}, name="one"
        )
        two = Reference(
            YCR, "A", "B", residue_sets={"b": ["A:50"], "a": ["A:26", "A:25"]}, name="two"
        )
        assert one.content_hash() == two.content_hash()
        assert len(one.content_hash()) == 64

    @pytest.mark.parametrize(
        "change",
        [
            {"receptor_chain": "B", "binder_chain": "A", "residue_sets": {"a": ["B:20"]}},
            {"binder_chain": None},
            {"residue_sets": {"a": ["A:25"]}},
            {"residue_sets": {"other": ["A:25", "A:26"]}},
            {"residue_map": {"A:1": "A:25"}},
            {"partners": {"p": YCR}},
        ],
    )
    def test_hash_changes_with_content(self, change):
        base = {
            "structure": YCR,
            "receptor_chain": "A",
            "binder_chain": "B",
            "residue_sets": {"a": ["A:25", "A:26"]},
        }
        assert Reference(**base).content_hash() != Reference(**(base | change)).content_hash()

    def test_hash_changes_with_the_file_and_the_chain_renaming(self, tmp_path):
        copy = tmp_path / "copy.pdb"
        copy.write_bytes(YCR.read_bytes() + b"REMARK extra\n")
        assert Reference(copy, "A").content_hash() != Reference(YCR, "A").content_hash()
        renamed = DockedPose.from_file(YCR, chain_map={"A": "R", "B": "P"})
        assert Reference(renamed, "R").content_hash() != Reference(YCR, "A").content_hash()

    def test_partner_content_is_in_the_hash(self, tmp_path):
        copy = tmp_path / "copy.pdb"
        copy.write_bytes(YCR.read_bytes() + b"REMARK extra\n")
        a = Reference(YCR, "A", partners={"p": YCR})
        b = Reference(YCR, "A", partners={"p": copy})
        assert a.content_hash() != b.content_hash()

    def test_frame_key_shares_the_superposition_of_equal_receptors(self):
        a = Reference(YCR, "A", "B", residue_sets={"s": ["A:25"]}, name="a")
        b = Reference(str(YCR), "A", None, residue_sets={"t": ["A:26"]}, name="b")
        assert a.frame_key() == b.frame_key()
        other_chain = Reference(YCR, "B", name="c")
        with_map = Reference(YCR, "A", residue_map={"A:1": "A:25"}, name="d")
        assert len({a.frame_key(), other_chain.frame_key(), with_map.frame_key()}) == 3

    def test_frame_key_with_a_restricted_fit(self):
        a = Reference(YCR, "A", "B", residue_sets={"s": ["A:25"]})
        b = Reference(YCR, "A", "B", residue_sets={"s": ["A:26"]})
        assert a.frame_key() == b.frame_key()
        assert a.frame_key("s") != b.frame_key("s")
        assert a.frame_key("site") == b.frame_key("site")
        assert a.frame_key("site") != a.frame_key()
        with pytest.raises(ValueError, match="not a residue set"):
            a.frame_key("missing")
        with pytest.raises(ValueError, match="needs a binder_chain"):
            Reference(YCR, "A").frame_key("site")


# ---------------------------------------------------------------------------
# dictionary round trip
# ---------------------------------------------------------------------------


class TestSerialisation:
    @staticmethod
    def rich():
        return Reference(
            YCR,
            "A",
            "B",
            residue_map={("A", 1): ("A", 25), ("A", 2): ("A", 26)},
            residue_sets={"epitope": [("A", n) for n in YCR_EPITOPE], "tip": ["A:25"]},
            partners={"peptide": Partner(YCR, "B")},
            name="strict",
        )

    def test_round_trip_is_exact_and_json_ready(self):
        reference = self.rich()
        data = reference.to_dict()
        text = json.dumps(data)
        again = Reference.from_dict(json.loads(text))
        assert again.to_dict() == data
        assert again.content_hash() == reference.content_hash()
        assert again.name == "strict" and again.binder_chain == "B"
        assert dict(again.residue_map) == dict(reference.residue_map)
        assert dict(again.residue_sets) == dict(reference.residue_sets)

    def test_to_dict_is_deterministic_and_sorted(self):
        data = self.rich().to_dict()
        assert list(data["residue_sets"]) == ["epitope", "tip"]
        assert data["residue_sets"]["tip"] == [["A", 25, ""]]
        assert data["residue_map"][0] == [["A", 1, ""], ["A", 25, ""]]
        assert data["schema"] == "binding_metrics.reference" and data["schema_version"] == 1
        assert data["structure"]["path"] == str(YCR) and data["structure"]["kind"] == "experimental"

    def test_insertion_codes_survive(self):
        reference = Reference(SFTI, "A", "I", residue_sets={"loop": ["A:60", "A:60B"]})
        again = Reference.from_dict(reference.to_dict())
        assert again.residue_sets["loop"] == frozenset(
            {ResidueKey("A", 60), ResidueKey("A", 60, "B")}
        )

    def test_a_docked_pose_is_read_back_as_a_docked_pose(self):
        reference = Reference(DockedPose.from_file(YCR, chain_map={"A": "R"}), "R")
        again = Reference.from_dict(reference.to_dict())
        assert isinstance(again.pose, DockedPose) and again.pose.chain_map == {"A": "R"}
        assert again.content_hash() == reference.content_hash()

    def test_an_in_memory_structure_needs_the_structure_back(self, ycr_atoms):
        reference = Reference(ycr_atoms, "A", partners={"p": Partner(ycr_atoms, "B")})
        data = reference.to_dict()
        assert data["structure"]["path"] is None
        with pytest.raises(ValueError, match="was in memory"):
            Reference.from_dict(data)
        with pytest.raises(ValueError, match="was in memory"):
            Reference.from_dict(data, structure=ycr_atoms)
        again = Reference.from_dict(data, structure=ycr_atoms, partner_structures={"p": ycr_atoms})
        assert again.content_hash() == reference.content_hash()

    def test_a_changed_file_is_refused(self, tmp_path):
        copy = tmp_path / "ref.pdb"
        copy.write_bytes(YCR.read_bytes())
        data = Reference(copy, "A").to_dict()
        copy.write_bytes(YCR.read_bytes() + b"REMARK extra\n")
        with pytest.raises(ValueError, match="does not match the recorded content hash"):
            Reference.from_dict(data)

    def test_a_missing_file_is_a_clear_error(self, tmp_path):
        copy = tmp_path / "ref.pdb"
        copy.write_bytes(YCR.read_bytes())
        data = Reference(copy, "A").to_dict()
        copy.unlink()
        with pytest.raises(FileNotFoundError, match="reference structure file not found"):
            Reference.from_dict(data)

    def test_unknown_keys_schema_and_version(self):
        data = self.rich().to_dict()
        with pytest.raises(ValueError, match="unknown keys"):
            Reference.from_dict({**data, "extra": 1})
        with pytest.raises(ValueError, match="not a reference dictionary"):
            Reference.from_dict({**data, "schema": "other"})
        with pytest.raises(ValueError, match="schema_version 2"):
            Reference.from_dict({**data, "schema_version": 2})
        with pytest.raises(ValueError, match="names partners"):
            Reference.from_dict(data, partner_structures={"ghost": YCR})


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


class TestVariants:
    def test_none_one_and_many(self):
        assert normalize_references(None) == {}
        one = Reference(YCR, "A")
        assert normalize_references(one) == {"reference": one}
        variants = normalize_references(
            {"strict": Reference(YCR, "A", "B"), "loose": Reference(YCR, "A")}
        )
        assert list(variants) == ["strict", "loose"]

    def test_the_mapping_key_is_the_variant_name(self):
        named = Reference(YCR, "A", name="mine")
        same = Reference(YCR, "A", name="strict")
        variants = normalize_references({"strict": same, "renamed": named})
        assert variants["strict"] is same
        assert variants["renamed"].name == "renamed"
        assert variants["renamed"] is not named and named.name == "mine"
        assert variants["renamed"].content_hash() == named.content_hash()

    def test_errors(self):
        with pytest.raises(ValueError, match="empty mapping"):
            normalize_references({})
        with pytest.raises(TypeError, match=r"reference\['x'\] must be a Reference"):
            normalize_references({"x": YCR})
        with pytest.raises(ValueError, match="variant names"):
            normalize_references({"": Reference(YCR, "A")})
        with pytest.raises(TypeError, match="Reference, a mapping"):
            normalize_references([Reference(YCR, "A")])

    def test_group_by_frame_keeps_first_variant_first(self):
        variants = normalize_references(
            {
                "first": Reference(YCR, "A", "B"),
                "other_receptor": Reference(SFTI, "A", "I"),
                "same_as_first": Reference(YCR, "A", residue_sets={"s": ["A:25"]}),
                "other_map": Reference(YCR, "A", residue_map={"A:1": "A:25"}),
            }
        )
        assert group_by_frame(variants) == [
            ("first", "same_as_first"),
            ("other_receptor",),
            ("other_map",),
        ]
