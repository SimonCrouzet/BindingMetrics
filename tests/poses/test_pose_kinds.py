"""Pose, DockedPose and ExperimentalPose: construction, identity, structure and scores."""

import dataclasses
import hashlib

import numpy as np
import pytest

from binding_metrics.poses import DockedPose, ExperimentalPose, Pose
from binding_metrics.poses.pose import atoms_sha256
from binding_metrics.predictors.store import file_sha256
from tests.predictors import synth


@pytest.fixture
def atoms():
    return synth.synthetic_complex().atoms


@pytest.fixture
def structure_file(tmp_path, atoms):
    return synth.write_structure(atoms, tmp_path / "decoy_01.cif")


# ---------------------------------------------------------------------------
# the base class
# ---------------------------------------------------------------------------


class TestBaseClass:
    def test_pose_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Pose()

    def test_a_kind_without_native_scores_cannot_be_built(self):
        class Incomplete(Pose):
            kind = "docked"

        with pytest.raises(TypeError, match="native_scores"):
            Incomplete()

    def test_the_kinds_are_poses_with_their_own_kind(self):
        assert issubclass(DockedPose, Pose) and DockedPose.kind == "docked"
        assert issubclass(ExperimentalPose, Pose) and ExperimentalPose.kind == "experimental"

    def test_poses_compare_by_identity(self, structure_file):
        first = DockedPose.from_file(structure_file)
        second = DockedPose.from_file(structure_file)
        assert first != second
        assert first.content_hash() == second.content_hash()
        assert len({first, second}) == 2


# ---------------------------------------------------------------------------
# content hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_a_path_hashes_the_bytes_of_the_file(self, structure_file):
        digest = DockedPose.from_file(structure_file).content_hash()
        assert digest == hashlib.sha256(structure_file.read_bytes()).hexdigest()
        assert digest == file_sha256(structure_file)

    def test_two_paths_with_one_content_share_the_hash(self, tmp_path, structure_file):
        copy = tmp_path / "elsewhere.cif"
        copy.write_bytes(structure_file.read_bytes())
        assert (
            DockedPose.from_file(copy).content_hash()
            == DockedPose.from_file(structure_file).content_hash()
        )

    def test_a_compressed_file_is_hashed_as_stored(self, tmp_path, atoms):
        path = synth.write_structure(atoms, tmp_path / "decoy.pdb.gz")
        assert DockedPose.from_file(path).content_hash() == file_sha256(path)

    def test_a_path_and_its_loading_into_memory_differ(self, structure_file):
        from binding_metrics.metrics._common import load_structure

        loaded = load_structure(structure_file, purpose="a test")
        on_disk = DockedPose.from_file(structure_file).content_hash()
        assert DockedPose.from_atoms(loaded).content_hash() != on_disk

    def test_atoms_hash_is_stable_and_follows_a_copy(self, atoms):
        assert atoms_sha256(atoms) == atoms_sha256(atoms.copy())
        assert len(atoms_sha256(atoms)) == 64

    def test_a_shift_below_a_thousandth_of_an_angstrom_is_ignored(self, atoms):
        nudged = atoms.copy()
        nudged.coord = nudged.coord + 0.0001
        assert atoms_sha256(nudged) == atoms_sha256(atoms)

    def test_a_shift_of_a_hundredth_of_an_angstrom_changes_it(self, atoms):
        moved = atoms.copy()
        moved.coord[0, 0] += 0.01
        assert atoms_sha256(moved) != atoms_sha256(atoms)

    @pytest.mark.parametrize(
        "annotation, value",
        [
            ("chain_id", "Z"),
            ("res_id", 99),
            ("ins_code", "A"),
            ("res_name", "GLY"),
            ("atom_name", "N"),
            ("element", "N"),
        ],
    )
    def test_every_annotation_of_the_serialisation_counts(self, atoms, annotation, value):
        changed = atoms.copy()
        column = getattr(changed, annotation).copy()
        column[3] = value
        changed.set_annotation(annotation, column)
        assert atoms_sha256(changed) != atoms_sha256(atoms)

    def test_the_order_of_the_atoms_counts(self, atoms):
        assert atoms_sha256(atoms[::-1]) != atoms_sha256(atoms)

    def test_a_b_factor_is_not_part_of_the_hash(self, atoms):
        other = atoms.copy()
        other.set_annotation("b_factor", np.asarray(other.b_factor) + 5.0)
        assert atoms_sha256(other) == atoms_sha256(atoms)

    def test_a_non_finite_coordinate_hashes_without_a_cast_warning(self, atoms, recwarn):
        broken = atoms.copy()
        broken.coord[2, 1] = np.nan
        assert atoms_sha256(broken) != atoms_sha256(atoms)
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]

    def test_the_chain_map_does_not_change_the_hash(self, atoms):
        plain = DockedPose.from_atoms(atoms)
        renamed = DockedPose.from_atoms(atoms, chain_map={"A": "R"})
        assert plain.content_hash() == renamed.content_hash()

    def test_a_missing_file_is_an_error_at_hash_time_not_at_construction(self, tmp_path):
        pose = DockedPose(tmp_path / "later.pdb")
        with pytest.raises(FileNotFoundError):
            pose.content_hash()


# ---------------------------------------------------------------------------
# id
# ---------------------------------------------------------------------------


class TestId:
    def test_a_given_id_wins(self, structure_file):
        assert DockedPose.from_file(structure_file, id="rank-1").id == "rank-1"

    def test_a_path_gets_its_stem_and_eight_hex_digits(self, structure_file):
        pose = DockedPose.from_file(structure_file)
        assert pose.id == f"decoy_01-{pose.content_hash()[:8]}"

    def test_the_stem_ignores_gz_and_the_structure_suffix(self, tmp_path, atoms):
        path = synth.write_structure(atoms, tmp_path / "model.v2.pdb.gz")
        assert DockedPose.from_file(path).id.startswith("model.v2-")

    def test_an_in_memory_docked_pose_is_named_after_its_kind(self, atoms):
        pose = DockedPose.from_atoms(atoms)
        assert pose.id == f"docked-{atoms_sha256(atoms)[:8]}"

    def test_an_in_memory_experimental_pose_prefers_the_pdb_id(self, atoms):
        assert ExperimentalPose.from_atoms(atoms, pdb_id="1YCR").id.startswith("1YCR-")
        assert ExperimentalPose.from_atoms(atoms).id.startswith("experimental-")

    def test_the_id_is_the_same_on_every_read(self, structure_file):
        pose = DockedPose.from_file(structure_file)
        assert pose.id == pose.id

    @pytest.mark.parametrize("bad", ["", 3])
    def test_a_given_id_must_be_a_non_empty_string(self, structure_file, bad):
        with pytest.raises(TypeError, match="id must be a non-empty string"):
            DockedPose.from_file(structure_file, id=bad)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_atoms_are_read_once(self, structure_file):
        pose = DockedPose.from_file(structure_file)
        assert pose.atoms() is pose.atoms()
        assert pose.atoms().array_length() == synth.N_ATOMS

    @pytest.mark.parametrize("suffix", [".pdb", ".cif.gz", ".pdb.gz"])
    def test_reads_pdb_and_gzip_compressed_files(self, tmp_path, atoms, suffix):
        path = synth.write_structure(atoms, tmp_path / f"m{suffix}")
        assert DockedPose.from_file(path).atoms().array_length() == synth.N_ATOMS

    def test_the_chain_map_renames_chains_from_a_file(self, structure_file):
        pose = ExperimentalPose.from_file(structure_file, chain_map={"A": "R", "B": "P"})
        assert set(pose.atoms().chain_id) == {"R", "P"}

    def test_a_swap_is_applied_at_once(self, atoms):
        renamed = DockedPose.from_atoms(atoms, chain_map={"A": "B", "B": "A"}).atoms()
        assert (renamed.chain_id == "B").sum() == 8
        assert (renamed.chain_id == "A").sum() == 6

    def test_an_in_memory_array_is_returned_as_given_without_a_chain_map(self, atoms):
        assert DockedPose.from_atoms(atoms).atoms() is atoms

    def test_an_in_memory_array_is_never_modified_by_a_chain_map(self, atoms):
        before = atoms.chain_id.copy()
        renamed = DockedPose.from_atoms(atoms, chain_map={"A": "R"}).atoms()
        assert set(renamed.chain_id) == {"R", "B"}
        assert (atoms.chain_id == before).all()

    def test_a_chain_the_structure_does_not_have_is_an_error(self, atoms):
        pose = DockedPose.from_atoms(atoms, chain_map={"Z": "R"})
        with pytest.raises(ValueError, match=r"names chains \['Z'\].*has \['A', 'B'\]"):
            pose.atoms()

    def test_renaming_into_an_existing_chain_would_merge_and_is_refused(self, atoms):
        with pytest.raises(ValueError, match="would merge chains"):
            DockedPose.from_atoms(atoms, chain_map={"A": "B"}).atoms()

    def test_the_structure_file_is_not_modified(self, structure_file):
        before = structure_file.read_bytes()
        DockedPose.from_file(structure_file, chain_map={"A": "R"}).atoms()
        assert structure_file.read_bytes() == before

    def test_structure_path_is_the_path_or_none(self, structure_file, atoms):
        assert DockedPose.from_file(str(structure_file)).structure_path == structure_file
        assert DockedPose.from_atoms(atoms).structure_path is None

    def test_a_stack_of_models_is_refused(self, atoms):
        from biotite.structure import stack

        with pytest.raises(TypeError, match="one model"):
            DockedPose.from_atoms(stack([atoms, atoms]))

    def test_a_source_that_is_not_a_structure_is_refused(self):
        with pytest.raises(TypeError, match="path or a biotite AtomArray"):
            DockedPose(42)

    def test_from_file_names_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nothing.pdb"):
            DockedPose.from_file(tmp_path / "nothing.pdb")
        with pytest.raises(FileNotFoundError, match="nothing.cif"):
            ExperimentalPose.from_file(tmp_path / "nothing.cif")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


class TestInputs:
    def test_keyword_arguments_left_over_are_provenance(self, structure_file):
        pose = DockedPose.from_file(structure_file, program="haddock", run=7, protocol="p1")
        assert pose.provenance == {"run": 7, "protocol": "p1"}
        assert pose.program == "haddock"

    def test_provenance_must_be_json_scalars(self, structure_file):
        with pytest.raises(TypeError, match=r"provenance\['box'\] must be a JSON scalar"):
            DockedPose.from_file(structure_file, box=[1, 2, 3])
        with pytest.raises(TypeError, match="JSON cannot hold"):
            ExperimentalPose.from_file(structure_file, temperature=float("nan"))

    def test_the_inputs_are_copied(self, structure_file):
        scores, chain_map = {"s": 1.0}, {"A": "R"}
        pose = DockedPose.from_file(structure_file, scores=scores, chain_map=chain_map)
        scores["s"], chain_map["A"] = 99.0, "X"
        assert pose.scores == {"s": 1.0} and pose.chain_map == {"A": "R"}

    def test_an_invalid_chain_map_is_refused(self, structure_file):
        with pytest.raises(ValueError, match="renamed to the same ID"):
            DockedPose.from_file(structure_file, chain_map={"A": "R", "B": "R"})

    def test_the_pose_is_frozen(self, structure_file):
        pose = DockedPose.from_file(structure_file)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pose.decoy = True

    def test_the_repr_does_not_print_the_atoms(self, atoms, structure_file):
        assert "14 atoms" in repr(DockedPose.from_atoms(atoms, decoy=True))
        assert "decoy_01.cif" in repr(ExperimentalPose.from_file(structure_file, pdb_id="1YCR"))


# ---------------------------------------------------------------------------
# DockedPose
# ---------------------------------------------------------------------------


class TestDockedPose:
    def test_native_scores_are_the_programs_scores(self, structure_file):
        pose = DockedPose.from_file(
            structure_file, program="haddock", scores={"haddock_score": -112.4, "n_clusters": 3}
        )
        assert pose.native_scores() == {"haddock_score": -112.4, "n_clusters": 3.0}

    def test_a_missing_score_is_none_and_nan_becomes_none(self, structure_file):
        pose = DockedPose.from_file(structure_file, scores={"a": None, "b": float("nan")})
        assert pose.native_scores() == {"a": None, "b": None}

    def test_native_scores_are_a_new_dictionary(self, structure_file):
        pose = DockedPose.from_file(structure_file, scores={"a": 1.0})
        pose.native_scores()["a"] = 5.0
        assert pose.native_scores() == {"a": 1.0}

    def test_a_pose_without_scores_reports_none(self, structure_file):
        assert DockedPose.from_file(structure_file).native_scores() == {}

    def test_numpy_scalars_are_accepted_and_become_floats(self, structure_file):
        pose = DockedPose.from_file(structure_file, scores={"a": np.float32(2.5), "n": np.int64(4)})
        assert [type(v) for v in pose.native_scores().values()] == [float, float]

    @pytest.mark.parametrize("bad", ["-12", True, [1.0]])
    def test_a_score_that_is_not_a_number_is_refused(self, structure_file, bad):
        with pytest.raises(TypeError, match=r"scores\['s'\] must be a number or None"):
            DockedPose.from_file(structure_file, scores={"s": bad})

    def test_an_infinite_score_is_refused_with_the_advice(self, structure_file):
        with pytest.raises(ValueError, match="use None for a missing value"):
            DockedPose.from_file(structure_file, scores={"s": float("inf")})

    def test_a_bare_decoy_has_no_program(self, atoms):
        decoy = DockedPose.from_atoms(atoms, decoy=True)
        assert decoy.decoy is True and decoy.program is None
        assert decoy.native_scores() == {}
        assert decoy.kind == "docked"

    def test_decoy_must_be_a_bool(self, atoms):
        with pytest.raises(TypeError, match="decoy must be True or False"):
            DockedPose.from_atoms(atoms, decoy=1)

    def test_program_must_be_a_string(self, atoms):
        with pytest.raises(TypeError, match="program"):
            DockedPose.from_atoms(atoms, program=3)


# ---------------------------------------------------------------------------
# ExperimentalPose
# ---------------------------------------------------------------------------


class TestExperimentalPose:
    def test_native_scores_hold_the_resolution_then_the_quality_values(self, structure_file):
        pose = ExperimentalPose.from_file(
            structure_file,
            method="X-RAY DIFFRACTION",
            resolution_angstrom=2.2,
            pdb_id="1YCR",
            quality={"r_free": 0.25, "r_work": 0.2},
        )
        assert pose.native_scores() == {"resolution_angstrom": 2.2, "r_free": 0.25, "r_work": 0.2}
        assert list(pose.native_scores()) == ["resolution_angstrom", "r_free", "r_work"]
        assert pose.kind == "experimental"

    def test_an_unknown_resolution_is_none_under_its_stable_name(self, structure_file):
        pose = ExperimentalPose.from_file(structure_file, method="SOLUTION NMR")
        assert pose.native_scores() == {"resolution_angstrom": None}

    def test_the_resolution_becomes_a_float(self, structure_file):
        assert (
            ExperimentalPose.from_file(structure_file, resolution_angstrom=2).resolution_angstrom
            == 2.0
        )

    @pytest.mark.parametrize("bad", [0, -1.5, float("nan"), float("inf")])
    def test_the_resolution_must_be_positive_and_finite(self, structure_file, bad):
        with pytest.raises(ValueError, match="resolution_angstrom must be positive and finite"):
            ExperimentalPose.from_file(structure_file, resolution_angstrom=bad)

    def test_the_resolution_must_be_a_number(self, structure_file):
        with pytest.raises(TypeError, match="resolution_angstrom must be a number"):
            ExperimentalPose.from_file(structure_file, resolution_angstrom="2.2 A")

    def test_quality_may_not_reuse_the_resolution_key(self, structure_file):
        with pytest.raises(ValueError, match="resolution_angstrom"):
            ExperimentalPose.from_file(structure_file, quality={"resolution_angstrom": 2.0})

    def test_a_missing_quality_value_is_none(self, structure_file):
        pose = ExperimentalPose.from_file(structure_file, quality={"r_free": float("nan")})
        assert pose.native_scores() == {"resolution_angstrom": None, "r_free": None}

    def test_method_and_pdb_id_are_kept_as_given(self, atoms):
        pose = ExperimentalPose.from_atoms(atoms, method="ELECTRON MICROSCOPY", pdb_id="7ABC")
        assert (pose.method, pose.pdb_id) == ("ELECTRON MICROSCOPY", "7ABC")
        with pytest.raises(TypeError, match="pdb_id"):
            ExperimentalPose.from_atoms(atoms, pdb_id=1234)
