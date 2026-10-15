"""Synthetic pose sets with a known pairwise RMSD (binding_metrics.testing.synthetic)."""

import importlib
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import binding_metrics.testing as testing
from binding_metrics.metrics._common import load_structure
from binding_metrics.poses.aggregate import (
    PoseDistances,
    aggregate_poses,
    select_representatives,
)
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose
from binding_metrics.testing import synthetic
from binding_metrics.testing.synthetic import make_synthetic_poses

#: Documented tolerance of the closed form against coordinates stored as float32 (angstrom).
TOLERANCE = 1e-4

ROTATIONS = {"backend_a": (0.0, 20.0, 35.0), "backend_b": (10.0, -15.0, 60.0)}


def kabsch(mobile, fixed):
    """Rotation and translation that best map ``mobile`` onto ``fixed`` (``x @ R.T + t``)."""
    mobile_centre, fixed_centre = mobile.mean(axis=0), fixed.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mobile_centre).T @ (fixed - fixed_centre))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, fixed_centre - mobile_centre @ rotation.T


def rmsd_after_receptor_fit(pose_a, pose_b, kind="ca"):
    """Peptide RMSD of two poses after fitting pose a's receptor C-alpha onto pose b's."""
    atoms_a, atoms_b = pose_a.atoms(), pose_b.atoms()
    receptor = (atoms_a.chain_id == "A") & (atoms_a.atom_name == "CA")
    rotation, translation = kabsch(
        atoms_a.coord[receptor].astype(float), atoms_b.coord[receptor].astype(float)
    )
    binder = atoms_a.chain_id == "B"
    if kind == "ca":
        binder = binder & (atoms_a.atom_name == "CA")
    elif kind == "backbone":
        binder = binder & np.isin(atoms_a.atom_name, ["N", "CA", "C", "O"])
    moved = atoms_a.coord[binder].astype(float) @ rotation.T + translation
    difference = moved - atoms_b.coord[binder].astype(float)
    return float(np.sqrt((difference**2).sum(axis=1).mean()))


def computed_matrix(poses, kind="ca"):
    return np.array([[rmsd_after_receptor_fit(a, b, kind) for b in poses] for a in poses])


@pytest.fixture(scope="module")
def default_set():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def rotated_set():
    return make_synthetic_poses(rotation_degrees=ROTATIONS)


class TestDefaultSet:
    def test_three_seeds_for_each_of_two_backends(self, default_set):
        assert [pose.id for pose in default_set.poses] == [
            "backend_a-s1",
            "backend_a-s2",
            "backend_a-s3",
            "backend_b-s1",
            "backend_b-s2",
            "backend_b-s3",
        ]
        assert default_set.truth.ids == tuple(pose.id for pose in default_set.poses)
        assert (default_set.binder_chain, default_set.target_chain) == ("B", "A")

    def test_poses_are_docked_poses_tagged_with_backend_and_seed(self, default_set):
        for pose, backend, seed in zip(
            default_set.poses, default_set.truth.backends, default_set.truth.seeds
        ):
            assert isinstance(pose, DockedPose) and pose.kind == "docked"
            assert pose.decoy is False
            assert pose.provenance["backend"] == backend and pose.provenance["seed"] == seed
            assert pose.provenance["synthetic"] is True
            assert pose.program == backend

    def test_native_scores_are_round_numbers_and_the_best_is_not_the_medoid(self, default_set):
        scores = [pose.native_scores()["ranking_score"] for pose in default_set.poses]
        assert scores == [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        assert tuple(scores) == default_set.truth.scores

    def test_the_peptide_and_receptor_are_the_two_chains_of_1ycr(self, default_set):
        atoms = default_set.poses[0].atoms()
        assert set(atoms.chain_id) == {"A", "B"}
        assert atoms.array_length() == 818
        assert (
            atoms.res_id[atoms.chain_id == "B"].min(),
            atoms.res_id[atoms.chain_id == "B"].max(),
        ) == (17, 29)

    def test_the_default_offsets_give_the_analytic_pairwise_matrix_of_the_design(self, default_set):
        truth = default_set.truth
        assert truth.offsets_angstrom == (0.0, 0.4, 0.8, 2.0, 2.3, 2.9)
        offsets = np.array(truth.offsets_angstrom)
        expected = np.abs(offsets[:, None] - offsets[None, :])
        for kind in synthetic.ATOM_SETS:
            assert np.allclose(truth.rmsd[kind], expected, atol=1e-12)
        assert truth.pair("backend_a-s1", "backend_a-s3") == pytest.approx(0.8)
        assert truth.pair("backend_b-s1", "backend_b-s3", "heavy") == pytest.approx(0.9)
        assert truth.pair("backend_a-s3", "backend_b-s1") == pytest.approx(1.2)
        assert truth.pair("backend_a-s1", "backend_b-s3") == pytest.approx(2.9)

    def test_the_matrices_are_symmetric_with_a_zero_diagonal(self, rotated_set):
        for kind, matrix in rotated_set.truth.rmsd.items():
            assert np.array_equal(matrix, matrix.T)
            assert np.all(np.diag(matrix) == 0.0)
            assert np.all(matrix >= 0.0), kind

    def test_analytic_rmsd_matches_the_computed_one_within_the_tolerance(self, default_set):
        for kind in synthetic.ATOM_SETS:
            difference = computed_matrix(default_set.poses, kind) - default_set.truth.rmsd[kind]
            assert np.abs(difference).max() < TOLERANCE, kind

    def test_the_truth_is_a_read_only_view(self, default_set):
        with pytest.raises(TypeError):
            default_set.truth.rmsd["ca"] = np.zeros((6, 6))
        assert default_set.truth.ids_of("backend_b") == [
            "backend_b-s1",
            "backend_b-s2",
            "backend_b-s3",
        ]


class TestRotations:
    def test_closed_form_with_rotations_matches_the_computed_rmsd(self, rotated_set):
        for kind in synthetic.ATOM_SETS:
            difference = computed_matrix(rotated_set.poses, kind) - rotated_set.truth.rmsd[kind]
            assert np.abs(difference).max() < TOLERANCE, kind

    def test_a_rotation_changes_the_atom_sets_differently(self, rotated_set):
        truth = rotated_set.truth
        assert not np.allclose(truth.rmsd["ca"], truth.rmsd["heavy"], atol=1e-3)

    def test_a_rotation_about_the_centroid_alone_follows_the_small_angle_scale(self):
        angles = {"backend_a": (0.0, 20.0, 0.0), "backend_b": (0.0, 0.0, 0.0)}
        zero_offsets = {"backend_a": (0.0, 0.0, 0.0), "backend_b": (0.0, 0.0, 0.0)}
        result = make_synthetic_poses(offsets_angstrom=zero_offsets, rotation_degrees=angles)
        # RMSD^2 = 4 sin^2(theta/2) <d_perp^2>, so it is proportional to sin(theta/2)
        small = make_synthetic_poses(
            offsets_angstrom=zero_offsets,
            rotation_degrees={"backend_a": (0.0, 10.0, 0.0), "backend_b": (0.0, 0.0, 0.0)},
        )
        ratio = result.truth.pair("backend_a-s1", "backend_a-s2") / small.truth.pair(
            "backend_a-s1", "backend_a-s2"
        )
        assert ratio == pytest.approx(np.sin(np.radians(10.0)) / np.sin(np.radians(5.0)), rel=1e-12)

    def test_no_motion_gives_a_zero_matrix(self):
        zero = {"backend_a": (0.0, 0.0, 0.0), "backend_b": (0.0, 0.0, 0.0)}
        result = make_synthetic_poses(offsets_angstrom=zero)
        assert not np.any(result.truth.rmsd["ca"])
        assert np.abs(computed_matrix(result.poses)).max() < TOLERANCE


class TestDeterminismAndSeeds:
    def test_the_same_arguments_give_the_same_poses_and_matrices(self):
        first = make_synthetic_poses(rotation_degrees=ROTATIONS, rng_seed=5)
        second = make_synthetic_poses(rotation_degrees=ROTATIONS, rng_seed=5)
        for kind in synthetic.ATOM_SETS:
            assert np.array_equal(first.truth.rmsd[kind], second.truth.rmsd[kind])
        for pose_a, pose_b in zip(first.poses, second.poses):
            assert pose_a.id == pose_b.id
            assert np.array_equal(pose_a.atoms().coord, pose_b.atoms().coord)
            assert pose_a.content_hash() == pose_b.content_hash()
        for (rot_a, shift_a), (rot_b, shift_b) in zip(
            first.truth.frame_transforms, second.truth.frame_transforms
        ):
            assert np.array_equal(rot_a, rot_b) and np.array_equal(shift_a, shift_b)

    def test_different_seeds_give_different_perturbations(self, default_set):
        first, second = default_set.poses[0], default_set.poses[1]
        binder = first.atoms().chain_id == "B"
        assert not np.allclose(first.atoms().coord[binder], second.atoms().coord[binder])
        assert default_set.truth.pair(first.id, second.id) == pytest.approx(0.4)

    def test_without_frame_noise_the_binder_moves_by_the_offset_along_the_direction(self):
        result = make_synthetic_poses(frame_noise=False)
        first, third = (result.poses[0].atoms(), result.poses[2].atoms())
        binder = first.chain_id == "B"
        shift = (third.coord[binder] - first.coord[binder]).astype(float)
        expected = 0.8 * np.array(result.truth.direction)
        assert np.allclose(shift, expected, atol=1e-5)
        assert result.truth.frame_transforms is None

    def test_the_receptor_is_not_moved_by_the_perturbation(self):
        result = make_synthetic_poses(frame_noise=False, rotation_degrees=ROTATIONS)
        reference = result.reference.atoms()
        for pose in result.poses:
            atoms = pose.atoms()
            receptor = atoms.chain_id == "A"
            assert np.array_equal(atoms.coord[receptor], reference.coord[receptor])

    def test_the_default_pose_with_no_motion_is_the_file_when_there_is_no_frame_noise(self):
        result = make_synthetic_poses(frame_noise=False)
        assert np.array_equal(result.poses[0].atoms().coord, result.reference.atoms().coord)

    def test_frame_noise_puts_every_pose_in_its_own_frame_and_the_transform_is_recorded(
        self, default_set
    ):
        reference = default_set.reference.atoms()
        centroids = [pose.atoms().coord.mean(axis=0) for pose in default_set.poses]
        assert (
            min(np.linalg.norm(a - b) for i, a in enumerate(centroids) for b in centroids[:i]) > 1.0
        )
        pose = default_set.poses[0]  # offset 0: only the frame transform differs from the file
        rotation, translation = default_set.truth.frame_transforms[0]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        expected = reference.coord.astype(float) @ rotation.T + translation
        assert np.allclose(pose.atoms().coord, expected, atol=1e-4)

    def test_the_rng_seed_changes_the_frames_and_not_the_truth(self):
        first = make_synthetic_poses(rng_seed=0)
        second = make_synthetic_poses(rng_seed=1)
        assert not np.allclose(first.poses[0].atoms().coord, second.poses[0].atoms().coord)
        assert np.array_equal(first.truth.rmsd["ca"], second.truth.rmsd["ca"])
        assert np.abs(computed_matrix(second.poses) - second.truth.rmsd["ca"]).max() < TOLERANCE

    def test_a_pose_does_not_depend_on_the_other_backends_or_seeds_of_the_call(self):
        small = make_synthetic_poses(backends=("backend_a",), seeds=(1, 2, 3))
        full = make_synthetic_poses()
        for pose_small, pose_full in zip(small.poses, full.poses[:3]):
            assert np.array_equal(pose_small.atoms().coord, pose_full.atoms().coord)


class TestParameters:
    def test_other_backends_and_seeds_use_a_regular_grid_of_offsets(self):
        result = make_synthetic_poses(backends=("a", "b", "c"), seeds=(7, 8, 9, 10))
        assert len(result.poses) == 12
        assert result.poses[0].id == "a-s7" and result.poses[-1].id == "c-s10"
        assert result.truth.offsets_angstrom[:5] == pytest.approx((0.0, 0.4, 0.8, 1.2, 2.0))
        assert result.truth.rmsd["ca"].shape == (12, 12)
        assert np.abs(computed_matrix(result.poses) - result.truth.rmsd["ca"]).max() < TOLERANCE

    def test_one_backend_one_seed(self):
        result = make_synthetic_poses(backends=("solo",), seeds=(42,))
        assert [pose.id for pose in result.poses] == ["solo-s42"]
        assert result.truth.rmsd["ca"].tolist() == [[0.0]]

    def test_explicit_offsets(self):
        offsets = {"backend_a": (0.0, 1.0, 5.0), "backend_b": (0.5, 0.5, 9.0)}
        result = make_synthetic_poses(offsets_angstrom=offsets)
        assert result.truth.pair("backend_a-s1", "backend_a-s3") == pytest.approx(5.0)
        assert result.truth.pair("backend_b-s1", "backend_b-s2") == pytest.approx(0.0)
        assert result.truth.pair("backend_a-s2", "backend_b-s3") == pytest.approx(8.0)

    def test_the_direction_is_normalised(self):
        result = make_synthetic_poses(direction=(0.0, 0.0, 10.0), frame_noise=False)
        assert result.truth.direction == (0.0, 0.0, 1.0)
        first, third = result.poses[0].atoms(), result.poses[2].atoms()
        shift = (third.coord - first.coord)[first.chain_id == "B"]
        assert np.allclose(shift, [0.0, 0.0, 0.8], atol=1e-5)

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"binder_chain": "Z"}, "binder_chain"),
            ({"target_chain": "Z"}, "target_chain"),
            ({"binder_chain": "A"}, "differ"),
            ({"backends": ()}, "empty"),
            ({"seeds": ()}, "empty"),
            ({"backends": ("a", "a")}, "unique"),
            ({"seeds": (1, 1)}, "unique"),
            ({"backends": ("a/b",)}, "backend names"),
            ({"seeds": (1.5,)}, "integers"),
            ({"rng_seed": -1}, "rng_seed"),
            ({"direction": (0.0, 0.0, 0.0)}, "direction"),
            ({"rotation_axis": (1.0, 2.0)}, "rotation_axis"),
            ({"offsets_angstrom": {"backend_a": (0.0, 1.0, 2.0)}}, "no entry"),
            ({"offsets_angstrom": {"backend_a": (0.0,), "backend_b": (0.0,)}}, "3 seeds"),
            (
                {"rotation_degrees": {"backend_a": (0, 0, float("nan")), "backend_b": (0, 0, 0)}},
                "finite",
            ),
        ],
    )
    def test_bad_arguments_are_refused(self, arguments, message):
        with pytest.raises(ValueError, match=message):
            make_synthetic_poses(**arguments)


class TestReferenceAndFiles:
    def test_the_reference_is_the_unmoved_complex_with_its_provenance(self, default_set):
        reference = default_set.reference
        assert isinstance(reference, ExperimentalPose) and reference.id == "reference"
        assert (reference.pdb_id, reference.method) == ("1YCR", "X-RAY DIFFRACTION")
        assert reference.resolution_angstrom == 2.6
        original = load_structure(synthetic.bundled_complex_path())
        assert np.array_equal(reference.atoms().coord, original.coord)

    def test_write_gives_one_mmcif_per_pose_that_reads_back_identically(
        self, default_set, tmp_path
    ):
        paths = default_set.write(tmp_path / "poses")
        assert [path.name for path in paths] == [f"{pose.id}.cif" for pose in default_set.poses]
        for path, pose in zip(paths, default_set.poses):
            atoms = load_structure(path)
            assert np.array_equal(atoms.coord, pose.atoms().coord)
            assert (atoms.atom_name == pose.atoms().atom_name).all()
            assert (atoms.res_id == pose.atoms().res_id).all()
            assert (atoms.chain_id == pose.atoms().chain_id).all()

    def test_a_pose_read_from_its_file_gives_the_same_rmsd(self, rotated_set, tmp_path):
        paths = rotated_set.write(tmp_path)
        poses = [DockedPose.from_file(path) for path in paths]
        assert np.abs(computed_matrix(poses) - rotated_set.truth.rmsd["ca"]).max() < TOLERANCE

    def test_complex_path_reads_another_file(self, tmp_path):
        copy = tmp_path / "other_name.pdb"
        shutil.copy(synthetic.bundled_complex_path(), copy)
        result = make_synthetic_poses(copy)
        assert np.array_equal(result.truth.rmsd["ca"], make_synthetic_poses().truth.rmsd["ca"])
        assert result.reference.pdb_id is None

    def test_the_bundled_file_is_found_in_a_source_tree(self):
        path = synthetic.bundled_complex_path()
        assert path.name == "example_linear_p53_1YCR.pdb" and path.is_file()

    def test_a_missing_bundled_file_raises_with_the_way_out(self, monkeypatch, tmp_path):
        monkeypatch.setattr(synthetic, "_candidate_paths", lambda: [tmp_path / "absent.pdb"])
        with pytest.raises(FileNotFoundError, match="complex_path"):
            make_synthetic_poses()
        with pytest.raises(FileNotFoundError, match="complex_path"):
            synthetic.bundled_complex_path()

    def test_an_explicit_complex_path_never_consults_the_bundled_location(
        self, monkeypatch, tmp_path
    ):
        source = synthetic.bundled_complex_path()
        monkeypatch.setattr(synthetic, "_candidate_paths", lambda: [tmp_path / "absent.pdb"])
        explicit = tmp_path / "x.pdb"
        shutil.copy(source, explicit)
        assert len(make_synthetic_poses(explicit).poses) == 6


class TestPoseSetLayerOnTheSyntheticSet:
    """The synthetic truth against the aggregates and representatives of the design (11.2)."""

    @staticmethod
    def rows(result):
        return [
            {
                "id": pose.id,
                "status": "ok",
                "provenance": pose.provenance,
                "native": pose.native_scores(),
            }
            for pose in result.poses
        ]

    def test_medoids_follow_from_the_analytic_matrix(self, default_set):
        distances = PoseDistances(default_set.truth.ids, default_set.truth.rmsd["ca"])
        chosen = select_representatives(self.rows(default_set), distances)
        assert chosen["backend_a"] == "backend_a-s2"
        assert chosen["backend_b"] == "backend_b-s2"
        # a3 and b1 are tied at a distance sum of 6.0: the lowest id wins
        assert chosen["all"] == "backend_a-s3"

    def test_the_best_ranking_score_per_backend_is_the_last_seed(self, default_set):
        chosen = select_representatives(self.rows(default_set), rule="confidence")
        assert chosen["backend_a"] == "backend_a-s3" and chosen["backend_b"] == "backend_b-s3"

    def test_native_scores_aggregate_per_backend(self, default_set):
        aggregates = aggregate_poses(self.rows(default_set))
        assert aggregates["backend_a"]["native.ranking_score"]["mean"] == pytest.approx(0.6)
        assert aggregates["backend_b"]["native.ranking_score"]["mean"] == pytest.approx(0.75)
        assert aggregates["all"]["native.ranking_score"]["n_total"] == 6


class TestPackageExports:
    def test_the_export_table_matches_all_and_the_defining_module(self):
        assert set(testing._EXPORTS) == set(testing.__all__)
        for name, module_name in testing._EXPORTS.items():
            assert getattr(testing, name) is getattr(importlib.import_module(module_name), name)

    def test_importing_the_package_loads_no_structure_library(self):
        code = textwrap.dedent(
            """
            import sys
            import binding_metrics.testing
            from binding_metrics.testing import make_synthetic_poses
            libraries = ("biotite", "scipy", "openmm", "gemmi", "mdtraj")
            print([m for m in libraries if m in sys.modules])
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        assert result.stdout.strip() == "[]"
