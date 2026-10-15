"""Common-frame alignment: pairing, superposition, region RMSD and applying the transform."""

import copy
import dataclasses
import json
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose
from binding_metrics.poses.frame import (
    FrameOptions,
    FrameResult,
    Transform,
    canonical_residue_name,
    is_degenerate,
    kabsch_transform,
    one_letter_code,
    pair_residues,
    place_in_reference_frame,
    prepare_reference,
    renumber_to_reference,
    residue_names_compatible,
    rmsd,
    superpose_receptor,
)
from binding_metrics.poses.reference import Reference, ResidueKey, polymer_residues

struc = pytest.importorskip("biotite.structure")

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"

SITE_CUTOFF_ANGSTROM = 4.5  # the default of FrameOptions.site_cutoff_angstrom


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load(path):
    from binding_metrics.metrics._common import load_structure

    return load_structure(path)


def brute_force_site(atoms, cutoff_angstrom=SITE_CUTOFF_ANGSTROM):
    """Numbers of the MDM2 residues (chain A) with a heavy atom near the p53 peptide (chain B).

    The binding site of 1YCR at ``cutoff_angstrom``, from the full distance matrix between the
    heavy atoms of the two chains.
    """
    heavy = ~np.isin(atoms.element, ["H", "D"]) & struc.filter_amino_acids(atoms)
    receptor = atoms[heavy & (atoms.chain_id == "A")]
    peptide = atoms[heavy & (atoms.chain_id == "B")]
    distances = np.linalg.norm(
        receptor.coord[:, None, :].astype(np.float64) - peptide.coord[None, :, :], axis=-1
    )
    near = (distances <= cutoff_angstrom).any(axis=1)
    return sorted({int(number) for number in receptor.res_id[near]})


@pytest.fixture(scope="module")
def ycr():
    return _load(YCR)


@pytest.fixture(scope="module")
def ycr_site(ycr):
    return brute_force_site(ycr)


@pytest.fixture(scope="module")
def sfti():
    atoms = _load(SFTI)
    return atoms[~atoms.hetero]


@pytest.fixture(scope="module")
def ycr_reference():
    return Reference(YCR, "A", "B")


@pytest.fixture(scope="module")
def renumbered(ycr):
    """1YCR with both chains numbered from 1, as the co-folding models write them."""
    return consecutive(consecutive(ycr, "A"), "B")


@pytest.fixture(scope="module")
def reference():
    return Reference(SFTI, "A", "I")


def random_rigid(seed):
    """A proper rotation and a translation, seeded."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.normal(size=3) * 15.0


def moved(atoms, rotation, translation):
    out = atoms.copy()
    out.coord = (atoms.coord.astype(np.float64) @ rotation.T + translation).astype(np.float32)
    return out


def residue_runs(atoms, chain):
    """Atom-index arrays of the residue runs of a chain, in order."""
    index = np.flatnonzero(atoms.chain_id == chain)
    changes = np.ones(len(index), dtype=bool)
    changes[1:] = (
        (atoms.res_id[index][1:] != atoms.res_id[index][:-1])
        | (atoms.ins_code[index][1:] != atoms.ins_code[index][:-1])
        | (atoms.res_name[index][1:] != atoms.res_name[index][:-1])
    )
    starts = np.flatnonzero(changes)
    return np.split(index, starts[1:])


def consecutive(atoms, chain, start=1):
    """A copy whose residues of ``chain`` are numbered start, start+1, ... with no icode."""
    out = atoms.copy()
    for number, run in enumerate(residue_runs(atoms, chain), start=start):
        out.res_id[run] = number
        out.ins_code[run] = ""
    return out


def drop(atoms, chain, drop_numbers):
    """A copy without the residues of ``chain`` whose number is in ``drop_numbers``."""
    remove = (atoms.chain_id == chain) & np.isin(atoms.res_id, list(drop_numbers))
    return atoms[~remove]


def ca_chain(coords, names=None, *, chain="A", first=1):
    """A chain of CA-only residues at ``coords`` with consecutive numbers from ``first``."""
    coords = np.asarray(coords, dtype=np.float64)
    atoms = struc.AtomArray(len(coords))
    atoms.coord = coords.astype(np.float32)
    atoms.chain_id[:] = chain
    atoms.res_id[:] = np.arange(first, first + len(coords))
    atoms.res_name[:] = names if names is not None else "ALA"
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    return atoms


def region(result, name):
    return result.region_rmsd_angstrom[name]


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


class TestOptions:
    def test_defaults_are_the_documented_ones(self):
        options = FrameOptions()
        assert (options.pairing, options.frame_atoms) == ("auto", "ca")
        assert options.frame_residues is None and options.outliers is None
        assert options.min_frame_residues == 10
        assert options.min_frame_fraction == 0.5
        assert options.min_sequence_identity == 0.8
        assert options.min_pair_fraction == 0.9
        assert options.site_cutoff_angstrom == 4.5
        assert (options.iqr_threshold, options.iqr_max_iterations) == (1.5, 10)
        assert options.iqr_min_deviation_angstrom == 0.01
        assert options.alignment_matrix == "BLOSUM62"
        assert options.alignment_gap_penalty == (-10, -1)
        assert options.renumber_receptor is True and options.metrics_frame == "reference"

    def test_dict_round_trip_is_json_and_exact(self):
        options = FrameOptions(
            pairing="sequence",
            frame_atoms="backbone",
            frame_residues="site",
            outliers="iqr",
            min_frame_residues=7,
            alignment_gap_penalty=(-12, -2),
        )
        data = options.to_dict()
        assert data["alignment_gap_penalty"] == [-12, -2]
        assert FrameOptions.from_dict(json.loads(json.dumps(data))) == options
        assert FrameOptions.from_dict({}) == FrameOptions()

    def test_unknown_option_is_refused(self):
        with pytest.raises(ValueError, match="unknown frame options.*typo"):
            FrameOptions.from_dict({"typo": 1})

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            ({"pairing": "fuzzy"}, ValueError, "pairing must be one of"),
            ({"frame_atoms": "all"}, ValueError, "frame_atoms must be one of"),
            ({"outliers": "mad"}, ValueError, "outliers must be one of"),
            ({"metrics_frame": "x"}, ValueError, "metrics_frame must be one of"),
            ({"frame_residues": ""}, ValueError, "residue set name"),
            ({"min_frame_residues": 2}, ValueError, "at least 3"),
            ({"min_frame_residues": 3.5}, TypeError, "integer"),
            ({"min_frame_residues": True}, TypeError, "integer"),
            ({"iqr_max_iterations": 0}, ValueError, "at least 1"),
            ({"min_frame_fraction": 1.5}, ValueError, "at most 1.0"),
            ({"min_sequence_identity": -0.1}, ValueError, "at least 0.0"),
            ({"min_pair_fraction": float("nan")}, ValueError, "finite"),
            ({"site_cutoff_angstrom": 0}, ValueError, "above 0.0"),
            ({"iqr_threshold": -1}, ValueError, "at least 0.0"),
            ({"alignment_gap_penalty": (-10,)}, ValueError, "open, extend"),
            ({"alignment_gap_penalty": (5, -1)}, ValueError, "not positive"),
            ({"alignment_matrix": ""}, ValueError, "matrix name"),
            ({"renumber_receptor": 1}, TypeError, "True or False"),
        ],
    )
    def test_invalid_values(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            FrameOptions(**kwargs)

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            FrameOptions().pairing = "map"

    def test_a_mapping_is_accepted_where_options_are(self, ycr, ycr_reference):
        result = superpose_receptor(ycr, ycr_reference, options={"min_frame_residues": 5})
        assert result.ok
        with pytest.raises(TypeError, match="FrameOptions, a mapping or None"):
            superpose_receptor(ycr, ycr_reference, options=5)


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------


class TestTransform:
    def test_apply_and_inverse(self):
        rotation, translation = random_rigid(3)
        transform = Transform(rotation, translation)
        points = np.random.default_rng(0).normal(size=(5, 3))
        expected = np.array([rotation @ p + translation for p in points])
        assert np.allclose(transform.apply(points), expected, atol=1e-12)
        assert np.allclose(transform.inverse().apply(transform.apply(points)), points, atol=1e-12)
        assert transform.inverse().inverse().allclose(transform, atol=1e-12)

    def test_apply_keeps_leading_axes_and_does_not_modify(self):
        transform = Transform(*random_rigid(1))
        stack = np.random.default_rng(1).normal(size=(2, 4, 3)).astype(np.float32)
        before = stack.copy()
        assert transform.apply(stack).shape == (2, 4, 3)
        assert np.array_equal(stack, before)

    def test_arrays_are_float64_and_read_only(self):
        transform = Transform(np.eye(3, dtype=np.float32), [1, 2, 3])
        assert transform.rotation.dtype == np.float64 and transform.translation.dtype == np.float64
        with pytest.raises(ValueError):
            transform.rotation[0, 0] = 2.0

    def test_invalid_transforms(self):
        with pytest.raises(ValueError, match="3 x 3 rotation and a 3-vector"):
            Transform(np.eye(2), np.zeros(3))
        with pytest.raises(ValueError, match="proper rotation"):
            Transform(2 * np.eye(3), np.zeros(3))
        with pytest.raises(ValueError, match="proper rotation"):
            Transform(np.diag([1.0, 1.0, -1.0]), np.zeros(3))  # a reflection
        with pytest.raises(ValueError, match="finite"):
            Transform(np.eye(3), [np.nan, 0, 0])

    def test_dict_round_trip(self):
        transform = Transform(*random_rigid(2))
        data = transform.to_dict()
        assert Transform.from_dict(json.loads(json.dumps(data))).allclose(transform, atol=1e-15)
        with pytest.raises(ValueError, match="unknown keys"):
            Transform.from_dict({**data, "scale": 1})


class TestKabsch:
    def test_recovers_a_known_rigid_motion(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(20, 3)) * 10
        rotation, translation = random_rigid(5)
        target = points @ rotation.T + translation
        found = kabsch_transform(points, target)
        assert np.allclose(found.rotation, rotation, atol=1e-12)
        assert np.allclose(found.translation, translation, atol=1e-11)

    def test_agrees_with_scipy(self):
        spatial = pytest.importorskip("scipy.spatial.transform")
        rng = np.random.default_rng(6)
        mobile = rng.normal(size=(30, 3)) * 8
        fixed = mobile @ random_rigid(7)[0].T + rng.normal(scale=0.5, size=(30, 3))
        found = kabsch_transform(mobile, fixed)
        centred_mobile = mobile - mobile.mean(axis=0)
        centred_fixed = fixed - fixed.mean(axis=0)
        reference, _ = spatial.Rotation.align_vectors(centred_fixed, centred_mobile)
        assert np.allclose(found.rotation, reference.as_matrix(), atol=1e-10)

    def test_a_mirror_image_gets_a_rotation_not_a_reflection(self):
        rng = np.random.default_rng(8)
        points = rng.normal(size=(15, 3)) * 5
        mirrored = points * np.array([-1.0, 1.0, 1.0])
        found = kabsch_transform(points, mirrored)
        assert math_isclose(np.linalg.det(found.rotation), 1.0)
        assert rmsd(found.apply(points), mirrored) > 1.0

    def test_planar_points_are_fine_and_collinear_ones_are_degenerate(self):
        rng = np.random.default_rng(9)
        plane = np.column_stack([rng.normal(size=8) * 5, rng.normal(size=8) * 5, np.zeros(8)])
        assert not is_degenerate(plane)
        line = np.outer(np.arange(6.0), [1.0, 2.0, 3.0])
        assert is_degenerate(line)
        assert is_degenerate(np.ones((6, 3)))
        assert is_degenerate(rng.normal(size=(2, 3)))
        assert not is_degenerate(rng.normal(size=(3, 3)))

    def test_shape_errors(self):
        with pytest.raises(ValueError, match="shape"):
            kabsch_transform(np.zeros((3, 3)), np.zeros((4, 3)))


def math_isclose(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# residue names
# ---------------------------------------------------------------------------


class TestResidueNames:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("HIE", "HIS"),
            ("hid", "HIS"),
            ("CYX", "CYS"),
            ("DAL", "ALA"),
            ("DPN", "PHE"),
            ("MSE", "MET"),
            ("SEP", "SER"),
            ("ALA", "ALA"),
            ("NMG", "NMG"),
            (" lys ", "LYS"),
        ],
    )
    def test_canonical_names(self, name, expected):
        assert canonical_residue_name(name) == expected

    def test_compatibility_and_one_letter_codes(self):
        assert residue_names_compatible("HIE", "HIS") and residue_names_compatible("DAL", "ALA")
        assert not residue_names_compatible("ALA", "GLY")
        assert one_letter_code("TRP") == "W" and one_letter_code("MSE") == "M"
        assert one_letter_code("NMG") == "X" and one_letter_code("ZZZ") == "X"


# ---------------------------------------------------------------------------
# rigid motion: the transform is recovered exactly
# ---------------------------------------------------------------------------


class TestRigidMotion:
    def test_the_transform_and_a_zero_rmsd_are_recovered(self, ycr, ycr_reference):
        rotation, translation = random_rigid(11)
        pose = moved(ycr, rotation, translation)
        result = superpose_receptor(pose, ycr_reference)
        assert result.status == "ok" and result.reason is None
        assert result.pairing == "number" and result.n_paired == 85
        assert result.paired_fraction == 1.0 and result.sequence_identity == 1.0
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        expected = Transform(rotation, translation).inverse()
        assert result.transform.allclose(expected, atol=1e-5)
        assert result.n_rejected == 0 and result.n_fit_residues == 85 and result.n_fit_atoms == 85

    def test_placing_the_pose_gives_the_reference_coordinates(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(12))
        before = pose.coord.copy()
        result = superpose_receptor(pose, ycr_reference)
        placed = place_in_reference_frame(pose, result)
        assert np.abs(placed.coord - ycr.coord).max() < 1e-4  # receptor and peptide alike
        assert np.array_equal(pose.coord, before)  # the input is not modified
        assert placed is not pose and placed.coord.dtype == np.float32
        assert (placed.res_id == ycr.res_id).all() and (placed.chain_id == ycr.chain_id).all()

    def test_the_binder_follows_the_receptor_transform(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(13))
        placed = place_in_reference_frame(pose, superpose_receptor(pose, ycr_reference))
        peptide = placed.chain_id == "B"
        assert rmsd(placed.coord[peptide], ycr.coord[peptide]) < 1e-4

    def test_backbone_atoms(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(14))
        result = superpose_receptor(pose, ycr_reference, options={"frame_atoms": "backbone"})
        assert result.ok and result.n_fit_atoms == 4 * 85 and result.n_fit_residues == 85
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)

    def test_identity_for_the_reference_itself(self, ycr, ycr_reference):
        result = superpose_receptor(ycr, ycr_reference)
        assert result.transform.allclose(Transform.identity(), atol=1e-9)
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-9)

    def test_a_mirror_image_pose_is_not_fitted_with_a_reflection(self, ycr, ycr_reference):
        mirrored = ycr.copy()
        mirrored.coord = mirrored.coord * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        result = superpose_receptor(mirrored, ycr_reference)
        assert result.ok
        assert math_isclose(np.linalg.det(result.transform.rotation), 1.0)
        assert result.receptor_rmsd_angstrom > 1.0

    def test_the_result_is_deterministic(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(15))
        first = superpose_receptor(pose, ycr_reference).to_dict(include_pairs=True)
        second = superpose_receptor(pose, ycr_reference).to_dict(include_pairs=True)
        assert first == second


class TestReferenceForms:
    def test_path_array_and_pose_give_the_same_frame(self, ycr):
        pose = moved(ycr, *random_rigid(16))
        results = [
            superpose_receptor(pose, Reference(YCR, "A", "B")).to_dict(),
            superpose_receptor(pose, Reference(ycr, "A", "B")).to_dict(),
            superpose_receptor(
                pose, Reference(ExperimentalPose.from_file(YCR), "A", "B")
            ).to_dict(),
            superpose_receptor(pose, Reference(DockedPose.from_file(YCR), "A", "B")).to_dict(),
        ]
        assert all(result == results[0] for result in results)
        assert results[0]["status"] == "ok"

    def test_a_pose_object_in_place_of_atoms(self, ycr, ycr_reference):
        rotation, translation = random_rigid(17)
        pose = DockedPose.from_atoms(moved(ycr, rotation, translation))
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.receptor_rmsd_angstrom < 1e-4
        placed = place_in_reference_frame(pose, result)
        assert np.abs(placed.coord - ycr.coord).max() < 1e-4

    def test_a_pose_from_a_file(self, ycr_reference):
        pose = DockedPose.from_file(YCR)
        assert superpose_receptor(pose, ycr_reference).transform.allclose(Transform.identity())

    def test_the_pose_chain_may_differ_from_the_reference_chain(self, ycr, ycr_reference):
        pose = DockedPose.from_atoms(moved(ycr, *random_rigid(18)), chain_map={"A": "R", "B": "P"})
        assert not superpose_receptor(pose, ycr_reference).ok  # chain 'A' is not in the pose
        result = superpose_receptor(pose, ycr_reference, receptor_chain="R")
        assert result.ok and result.n_paired == 85

    def test_a_prepared_reference_is_reused(self, ycr, ycr_reference):
        prepared = prepare_reference(ycr_reference)
        assert prepare_reference(ycr_reference) is prepared  # cached on the reference
        pose = moved(ycr, *random_rigid(19))
        assert (
            superpose_receptor(pose, prepared).to_dict()
            == superpose_receptor(pose, ycr_reference).to_dict()
        )
        other_cutoff = superpose_receptor(
            pose, prepared, options=FrameOptions(site_cutoff_angstrom=3.5)
        )
        assert len(other_cutoff.region_n_residues) == 2
        assert other_cutoff.region_n_residues["site"] < len(prepared.site)

    def test_stacks_and_wrong_types_are_refused(self, ycr, ycr_reference):
        stack = struc.stack([ycr, ycr])
        with pytest.raises(ValueError, match="one model"):
            superpose_receptor(stack, ycr_reference)
        with pytest.raises(TypeError, match="must be a Reference"):
            superpose_receptor(ycr, "not a reference")


# ---------------------------------------------------------------------------
# trimmed constructs
# ---------------------------------------------------------------------------


class TestTrimmedConstructs:
    DROPPED = list(range(25, 30)) + list(range(105, 110)) + [60, 61, 62, 63]

    def test_residues_missing_at_both_ends_and_in_the_middle(self, ycr, ycr_reference):
        pose = moved(drop(ycr, "A", self.DROPPED), *random_rigid(21))
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.pairing == "number"
        assert result.n_receptor_residues_pose == 71 and result.n_paired == 71
        assert result.paired_fraction == pytest.approx(71 / 85)
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        assert result.transform.allclose(Transform(*random_rigid(21)).inverse(), atol=1e-5)
        paired_numbers = {key.number for _, key in result.pairs}
        assert paired_numbers.isdisjoint(self.DROPPED)

    def test_trimmed_and_renumbered_from_one_pairs_by_sequence(self, ycr, ycr_reference):
        trimmed = drop(ycr, "A", self.DROPPED)
        pose = moved(consecutive(consecutive(trimmed, "A"), "B"), *random_rigid(22))
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.pairing == "sequence"
        assert result.n_paired == 71 and result.sequence_identity == 1.0
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        assert not result.alignment_ambiguous
        # pose residue i is the i-th kept reference residue, across the 4-residue gap
        kept = [n for n in range(25, 110) if n not in self.DROPPED]
        assert [key.number for _, key in result.pairs] == kept
        assert [key.number for key, _ in result.pairs] == list(range(1, 72))

    def test_a_trimmed_reference(self, ycr):
        reference = Reference(drop(ycr, "A", range(25, 35)), "A", "B")
        result = superpose_receptor(moved(ycr, *random_rigid(23)), reference)
        assert result.ok and result.n_paired == 75
        assert result.n_receptor_residues_pose == 85 and result.n_receptor_residues_reference == 75
        assert result.paired_fraction == 1.0

    def test_a_short_construct_fails_on_the_fraction(self, ycr, ycr_reference):
        pose = drop(ycr, "A", range(51, 110))  # 26 residues, 31 percent of the reference
        result = superpose_receptor(pose, ycr_reference)
        assert result.status == "failed" and result.transform is None
        assert result.reason.startswith("only 26 of the 85 reference receptor residues")
        assert "min_frame_fraction is 0.50" in result.reason
        assert result.n_paired == 26 and result.paired_fraction == pytest.approx(26 / 85)
        assert result.receptor_rmsd_angstrom is None and result.region_rmsd_angstrom == {}
        relaxed = superpose_receptor(
            pose, ycr_reference, options={"min_frame_fraction": 0.3, "min_frame_residues": 10}
        )
        assert relaxed.ok and relaxed.n_paired == 26

    def test_too_few_residues_to_fit(self, ycr, ycr_reference):
        pose = drop(ycr, "A", range(35, 110))  # 10 residues: 25..34
        result = superpose_receptor(pose, ycr_reference, options={"min_frame_fraction": 0.0})
        assert result.ok and result.n_paired == 10
        pose = drop(ycr, "A", range(34, 110))  # 9 residues
        result = superpose_receptor(pose, ycr_reference, options={"min_frame_fraction": 0.0})
        assert result.status == "failed"
        assert "only 9 paired receptor residues to fit; min_frame_residues is 10" in result.reason


# ---------------------------------------------------------------------------
# renumbered constructs
# ---------------------------------------------------------------------------


class TestRenumberedConstructs:
    def test_both_chains_from_one_pair_with_an_offset_of_24(self, renumbered, ycr_reference):
        pose = moved(renumbered, *random_rigid(31))
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.pairing == "sequence"
        assert result.n_paired == 85 and result.sequence_identity == 1.0
        assert not result.alignment_ambiguous
        assert all(r.number == p.number + 24 for p, r in result.pairs)
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        assert result.transform.allclose(Transform(*random_rigid(31)).inverse(), atol=1e-5)

    def test_the_site_region_is_in_reference_numbering(self, renumbered, ycr_reference, ycr_site):
        result = superpose_receptor(renumbered, ycr_reference)
        # every site residue of the reference is paired although the pose is numbered from 1
        assert result.region_n_residues["site"] == len(ycr_site)
        assert result.region_n_paired["site"] == len(ycr_site)
        assert region(result, "site") == pytest.approx(0.0, abs=1e-4)

    def test_the_note_says_why_number_pairing_did_not_apply(self, renumbered, ycr_reference):
        pairing = pair_residues(renumbered, ycr_reference)
        assert pairing.method == "sequence"
        assert len(pairing.notes) == 1 and pairing.notes[0].startswith("pairing by number: ")

    def test_renumbered_without_the_reference_overlap_in_numbers(self, ycr, ycr_reference):
        pose = consecutive(ycr, "A", start=1000)
        result = superpose_receptor(pose, ycr_reference)
        assert result.pairing == "sequence" and result.n_paired == 85

    def test_forcing_the_number_method_on_renumbered_residues_fails(
        self, renumbered, ycr_reference
    ):
        result = superpose_receptor(renumbered, ycr_reference, options={"pairing": "number"})
        assert result.status == "failed" and result.pairing == "number"
        assert "pairing by number" in result.reason and "min_pair_fraction is 0.90" in result.reason

    def test_forcing_the_sequence_method_on_matching_numbers(self, ycr, ycr_reference):
        result = superpose_receptor(ycr, ycr_reference, options={"pairing": "sequence"})
        assert result.ok and result.pairing == "sequence" and result.n_paired == 85

    def test_position_pairing_is_opt_in(self, renumbered, ycr_reference):
        result = superpose_receptor(renumbered, ycr_reference, options={"pairing": "position"})
        assert result.ok and result.pairing == "position" and result.n_paired == 85
        trimmed = drop(renumbered, "A", [1, 2])
        failed = superpose_receptor(trimmed, ycr_reference, options={"pairing": "position"})
        assert failed.status == "failed" and failed.pairing == "position"
        assert "equal length: the pose has 83 residues, the reference 85" in failed.reason

    def test_renamed_variants_still_pair(self, ycr, ycr_reference):
        pose = ycr.copy()
        pose.res_name[(pose.chain_id == "A") & (pose.res_name == "HIS")] = "HIE"
        first_ala = np.flatnonzero((pose.chain_id == "A") & (pose.res_name == "ALA"))[:1]
        pose.res_name[first_ala] = "DAL"
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.pairing == "number" and result.n_paired == 85


# ---------------------------------------------------------------------------
# insertion codes
# ---------------------------------------------------------------------------


class TestInsertionCodes:
    def test_the_reference_has_insertion_codes(self, sfti, reference):
        keys = [r.key for r in polymer_residues(sfti, "A")]
        assert ResidueKey("A", 60, "B") in keys and len(keys) == len(set(keys)) == 241

    def test_same_numbering_pairs_every_residue_once(self, sfti, reference):
        result = superpose_receptor(moved(sfti, *random_rigid(41)), reference)
        assert result.ok and result.pairing == "number" and result.n_paired == 241
        assert all(p == r for p, r in result.pairs)
        assert any(r.icode for _, r in result.pairs)
        assert len({r for _, r in result.pairs}) == 241  # 60, 60A, ... 60I stay apart
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)

    def test_a_model_that_numbers_residues_consecutively(self, sfti, reference):
        pose = moved(consecutive(consecutive(sfti, "A"), "I"), *random_rigid(42))
        result = superpose_receptor(pose, reference)
        assert result.ok and result.pairing == "sequence" and result.n_paired == 241
        expected = [r.key for r in polymer_residues(sfti, "A")]
        assert [r for _, r in result.pairs] == expected
        assert ResidueKey("A", 60, "I") in [r for _, r in result.pairs]
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)

    def test_a_residue_map_with_insertion_codes(self, sfti, reference):
        pose = consecutive(sfti, "A")
        reference_keys = [r.key for r in polymer_residues(sfti, "A")]
        mapping = {("A", i + 1): key for i, key in enumerate(reference_keys)}
        result = superpose_receptor(pose, reference, residue_map=mapping)
        assert result.ok and result.pairing == "map" and result.n_paired == 241
        assert ResidueKey("A", 60, "C") in [r for _, r in result.pairs]

    def test_number_pairing_does_not_confuse_52_and_52a(self):
        base = np.random.default_rng(5).normal(size=(12, 3)) * 8
        reference_atoms = ca_chain(base, first=50)
        reference_atoms.ins_code[3:] = "A"  # residues 53.. become 53A, 54A, ...
        pose_atoms = ca_chain(base, first=50)  # the same coordinates, numbered 50..61
        reference = Reference(reference_atoms, "A")
        refused = pair_residues(pose_atoms, reference, options={"pairing": "number"})
        assert refused.n_paired == 0 and "min_pair_fraction is 0.90" in refused.reason
        result = pair_residues(
            pose_atoms, reference, options={"pairing": "number", "min_pair_fraction": 0.2}
        )
        assert result.method == "number" and result.n_paired == 3  # only 50, 51, 52 agree
        assert [key.label for key in result.reference_keys] == ["A:50", "A:51", "A:52"]


# ---------------------------------------------------------------------------
# the residue map
# ---------------------------------------------------------------------------


class TestResidueMap:
    def test_a_map_pairs_a_renumbered_construct(self, ycr, ycr_reference):
        pose = consecutive(consecutive(ycr, "A"), "B")
        mapping = {("A", i): ("A", i + 24) for i in range(1, 61)}
        result = superpose_receptor(pose, Reference(YCR, "A", "B", residue_map=mapping))
        assert result.ok and result.pairing == "map" and result.n_paired == 60
        assert result.paired_fraction == pytest.approx(60 / 85)
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        assert result.sequence_identity == 1.0

    def test_the_map_wins_over_numbers_that_would_pair(self, ycr, ycr_reference):
        mapping = {("A", n): ("A", n) for n in range(30, 81)}
        result = superpose_receptor(ycr, Reference(YCR, "A", residue_map=mapping))
        assert result.pairing == "map" and result.n_paired == 51
        assert {p.number for p, _ in result.pairs} == set(range(30, 81))

    def test_a_per_call_map_replaces_the_reference_map(self, ycr):
        reference = Reference(YCR, "A", residue_map={("A", 999): ("A", 25)})
        broken = superpose_receptor(ycr, reference)
        assert broken.status == "failed" and "no entry of the residue_map" in broken.reason
        better = superpose_receptor(
            ycr, reference, residue_map={("A", n): ("A", n) for n in range(25, 110)}
        )
        assert better.ok and better.pairing == "map" and better.n_paired == 85
        removed = superpose_receptor(ycr, reference, residue_map={})
        assert removed.ok and removed.pairing == "number"

    def test_a_low_identity_map_is_not_refused_by_identity(self, ycr):
        # the user states the correspondence: a shift by one residue pairs unlike residues
        mapping = {("A", n): ("A", n + 1) for n in range(25, 109)}
        result = superpose_receptor(ycr, Reference(YCR, "A", residue_map=mapping))
        assert result.pairing == "map" and result.sequence_identity < 0.5
        assert result.ok or "min_sequence_identity" not in result.reason

    def test_entries_on_other_chains_are_ignored(self, ycr):
        identity = {("A", n): ("A", n) for n in range(25, 110)}
        reference = Reference(YCR, "A", residue_map=identity)
        stray = {**identity, ("B", 17): ("A", 25)}  # a peptide residue: not a receptor pair
        result = superpose_receptor(ycr, reference, residue_map=stray)
        assert result.ok and result.n_paired == 85

    def test_the_map_method_without_a_map(self, ycr, ycr_reference):
        result = superpose_receptor(ycr, ycr_reference, options={"pairing": "map"})
        assert result.status == "failed" and "needs a residue_map" in result.reason

    def test_a_residue_that_occurs_twice_is_refused_in_a_map(self, ycr):
        pose = ycr.copy()
        runs = residue_runs(pose, "A")
        pose.res_id[runs[1]] = pose.res_id[runs[0]][0]  # two residues numbered 25
        reference = Reference(YCR, "A", residue_map={("A", 25): ("A", 25), ("A", 26): ("A", 26)})
        result = superpose_receptor(pose, reference)
        assert result.status == "failed" and "A:25, which occurs more than once" in result.reason


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_a_missing_receptor_chain(self, ycr, ycr_reference):
        result = superpose_receptor(ycr[ycr.chain_id == "B"], ycr_reference)
        assert result.status == "failed"
        assert "receptor chain 'A' is not in the pose: it has chains ['B']" in result.reason
        assert result.pairing is None and result.transform is None

    def test_a_chain_without_polymer_residues(self, ycr, ycr_reference):
        water = struc.AtomArray(3)
        water.chain_id[:] = "A"
        water.res_name[:] = "HOH"
        water.atom_name[:] = "O"
        water.element[:] = "O"
        result = superpose_receptor(water, ycr_reference)
        assert (
            result.status == "failed"
            and "pose receptor chain has no polymer residues" in result.reason
        )

    def test_another_protein_is_refused(self, sfti, ycr_reference):
        # trypsin against MDM2: the free end gaps let the alignment keep only a short overlap,
        # so the paired fraction (and not the identity of the few aligned columns) refuses it
        result = superpose_receptor(sfti, ycr_reference)
        assert result.status == "failed" and result.pairing == "sequence"
        assert re.match(r"only \d+ of the 85 reference receptor residues are paired", result.reason)
        assert result.n_receptor_residues_pose == 241 and result.transform is None

    def test_a_scrambled_construct_fails_and_a_lower_limit_accepts_it(self, ycr, ycr_reference):
        pose = ycr.copy()
        for run in residue_runs(pose, "A")[::5]:  # every fifth residue becomes glycine
            pose.res_name[run] = "GLY"
        strict = superpose_receptor(pose, ycr_reference)
        assert strict.ok  # 20 percent changed: the number method (0.80 < 0.90) gave way to sequence
        assert strict.pairing == "sequence" and strict.sequence_identity == pytest.approx(
            0.8, abs=0.05
        )
        for run in residue_runs(pose, "A")[1::5]:
            pose.res_name[run] = "GLY"
        failed = superpose_receptor(pose, ycr_reference)
        assert failed.status == "failed" and "identity" in failed.reason
        accepted = superpose_receptor(
            pose, ycr_reference, options={"min_sequence_identity": 0.5, "min_pair_fraction": 0.5}
        )
        assert accepted.ok and accepted.n_paired < 85

    def test_point_mutations_leave_the_number_pairing_alone(self, ycr, ycr_reference):
        pose = ycr.copy()
        for run in residue_runs(pose, "A")[:2]:
            pose.res_name[run] = "TRP"
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.pairing == "number" and result.n_paired == 83
        assert result.sequence_identity == pytest.approx(83 / 85)

    def test_a_failed_frame_cannot_be_applied(self, ycr, ycr_reference):
        failed = superpose_receptor(ycr[ycr.chain_id == "B"], ycr_reference)
        with pytest.raises(ValueError, match="no transform: receptor chain 'A' is not in the pose"):
            place_in_reference_frame(ycr, failed)
        with pytest.raises(ValueError, match="the frame failed"):
            renumber_to_reference(ycr, failed, ycr_reference)

    def test_configuration_errors_raise_even_for_a_bad_pose(self, ycr, ycr_reference):
        with pytest.raises(ValueError, match="not a residue set of reference"):
            superpose_receptor(ycr, ycr_reference, options={"frame_residues": "nope"})
        with pytest.raises(ValueError, match="needs a binder_chain"):
            superpose_receptor(ycr, Reference(YCR, "A"), options={"frame_residues": "site"})
        bad = Reference(YCR, "A", residue_sets={"s": ["A:1"]})
        with pytest.raises(ValueError, match="reference 'reference' is not valid"):
            superpose_receptor(ycr, bad)
        with pytest.raises(ValueError, match="substitution matrix"):
            superpose_receptor(
                consecutive(ycr, "A"), Reference(YCR, "A"), options={"alignment_matrix": "NOPE"}
            )


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_fewer_than_three_paired_residues(self):
        coords = np.array([[0.0, 0, 0], [3.8, 0, 0]])
        tiny = ca_chain(coords)
        result = superpose_receptor(
            tiny, Reference(tiny, "A"), options={"min_frame_residues": 3, "min_frame_fraction": 0.0}
        )
        assert result.status == "failed"
        assert "only 2 paired receptor residues to fit; min_frame_residues is 3" in result.reason
        assert result.transform is None

    def test_collinear_atoms_have_no_unique_rotation(self):
        coords = np.outer(np.arange(6.0), [3.8, 0.0, 0.0])
        line = ca_chain(coords)
        result = superpose_receptor(line, Reference(line, "A"), options={"min_frame_residues": 3})
        assert result.status == "failed" and "collinear or coincident" in result.reason
        assert "the rotation is not determined" in result.reason

    def test_a_collinear_pose_against_a_good_reference(self):
        coords = np.random.default_rng(2).normal(size=(8, 3)) * 6
        reference = ca_chain(coords)
        line = ca_chain(np.outer(np.arange(8.0), [3.8, 0.0, 0.0]))
        result = superpose_receptor(
            line, Reference(reference, "A"), options={"min_frame_residues": 3}
        )
        assert result.status == "failed" and "fit atoms of the pose are collinear" in result.reason

    def test_a_planar_set_is_fitted_with_a_proper_rotation(self):
        rng = np.random.default_rng(3)
        plane = np.column_stack([rng.normal(size=8) * 6, rng.normal(size=8) * 6, np.zeros(8)])
        reference = ca_chain(plane)
        rotation, translation = random_rigid(4)
        pose = ca_chain(plane @ rotation.T + translation)
        result = superpose_receptor(
            pose, Reference(reference, "A"), options={"min_frame_residues": 3}
        )
        assert result.ok
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        assert math_isclose(np.linalg.det(result.transform.rotation), 1.0)

    def test_residues_with_non_finite_coordinates_are_left_out(self, ycr, ycr_reference):
        pose = ycr.copy()
        first = residue_runs(pose, "A")[0]
        pose.coord[first] = np.nan
        result = superpose_receptor(pose, ycr_reference)
        assert result.ok and result.n_paired == 84
        assert any("1 pose receptor residues with non-finite CA" in w for w in result.warnings)
        assert np.isfinite(result.transform.rotation).all()


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


class TestRegions:
    @staticmethod
    def pair_scaled_outward():
        """Twelve CA atoms in six +-pairs; the pair of residues 1 and 7 is pushed out by 20 percent.

        The displacement of each moved atom is proportional to its position, so the covariance
        matrix of the two sets stays symmetric and positive definite: the optimal fit is the
        identity, and every RMSD below is known by hand.
        """
        half = np.array(
            [[10, 0, 0], [0, 8, 0], [0, 0, 6], [5, 5, 5], [-4, 6, 3], [7, -2, 5]], dtype=float
        )
        reference = np.vstack([half, -half])
        pose = reference.copy()
        pose[[0, 6]] *= (
            1.2  # residues 1 and 7: displaced by 2 angstrom each, in opposite directions
        )
        names = ["ALA", "GLY", "SER", "THR", "VAL", "LEU", "ILE", "PHE", "TYR", "TRP", "LYS", "ARG"]
        return ca_chain(reference, names), ca_chain(pose, names)

    def test_region_rmsd_against_a_hand_computed_value(self):
        reference_atoms, pose_atoms = self.pair_scaled_outward()
        reference = Reference(
            reference_atoms,
            "A",
            residue_sets={
                "moved": ["A:1", "A:7"],
                "still": [("A", n) for n in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)],
            },
        )
        result = superpose_receptor(pose_atoms, reference)
        assert result.ok and result.pairing == "number" and result.n_paired == 12
        # the fit is the identity because the covariance is symmetric positive definite
        assert result.transform.allclose(Transform.identity(), atol=1e-6)
        assert region(result, "moved") == pytest.approx(2.0, abs=1e-5)  # sqrt((2^2 + 2^2) / 2)
        assert region(result, "still") == pytest.approx(0.0, abs=1e-5)
        assert region(result, "all") == pytest.approx(np.sqrt(8 / 12), abs=1e-5)
        assert result.receptor_rmsd_angstrom == pytest.approx(np.sqrt(8 / 12), abs=1e-5)
        assert result.region_n_residues == {"all": 12, "moved": 2, "still": 10}
        assert result.region_n_paired == {"all": 12, "moved": 2, "still": 10}
        assert result.region_reasons == {}

    def test_fitting_only_the_still_residues_leaves_the_moved_ones_displaced(self):
        reference_atoms, pose_atoms = self.pair_scaled_outward()
        reference = Reference(
            reference_atoms,
            "A",
            residue_sets={
                "moved": ["A:1", "A:7"],
                "still": [("A", n) for n in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)],
            },
        )
        result = superpose_receptor(pose_atoms, reference, options={"frame_residues": "still"})
        assert result.ok and result.n_fit_residues == 10
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-5)
        assert region(result, "moved") == pytest.approx(2.0, abs=1e-5)
        assert region(result, "all") == pytest.approx(np.sqrt(8 / 12), abs=1e-5)

    def test_a_region_with_no_paired_residue_is_none_with_a_reason(self, ycr):
        reference = Reference(
            YCR,
            "A",
            "B",
            residue_sets={"low": ["A:25", "A:26"], "high": [("A", n) for n in range(100, 110)]},
        )
        pose = drop(ycr, "A", range(100, 110))
        result = superpose_receptor(pose, reference)
        assert result.ok
        assert result.region_rmsd_angstrom["high"] is None
        assert result.region_reasons == {
            "high": "none of the 10 residues of region 'high' is paired"
        }
        assert result.region_n_paired["high"] == 0 and result.region_n_paired["low"] == 2
        assert result.region_rmsd_angstrom["low"] == pytest.approx(0.0, abs=1e-5)

    def test_the_site_region_exists_only_with_a_binder_chain(self, ycr):
        with_binder = superpose_receptor(ycr, Reference(YCR, "A", "B"))
        without = superpose_receptor(ycr, Reference(YCR, "A"))
        assert set(with_binder.region_rmsd_angstrom) == {"all", "site"}
        assert set(without.region_rmsd_angstrom) == {"all"}

    def test_the_site_is_the_residues_near_the_reference_peptide(
        self, ycr, ycr_reference, ycr_site
    ):
        prepared = prepare_reference(ycr_reference)
        # the site is the set of residues of the distance matrix, at the default and at a cutoff
        assert [key.number for key in prepared.site] == ycr_site
        assert all(key.chain == "A" for key in prepared.site)
        tighter = prepare_reference(ycr_reference, site_cutoff_angstrom=3.5)
        assert [key.number for key in tighter.site] == brute_force_site(ycr, 3.5)
        assert 0 < len(tighter.site) < len(prepared.site) and set(tighter.site) < set(prepared.site)

    def test_the_site_cutoff_is_an_option(self, ycr, ycr_reference):
        wide = superpose_receptor(ycr, ycr_reference, options={"site_cutoff_angstrom": 6.0})
        narrow = superpose_receptor(ycr, ycr_reference, options={"site_cutoff_angstrom": 3.5})
        # the option reaches the site of each result
        assert wide.region_n_residues["site"] == len(brute_force_site(ycr, 6.0))
        assert narrow.region_n_residues["site"] == len(brute_force_site(ycr, 3.5))

    def test_regions_use_alpha_carbons_whatever_the_fit_atoms(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(51))
        result = superpose_receptor(pose, ycr_reference, options={"frame_atoms": "backbone"})
        assert region(result, "all") == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# the fit options
# ---------------------------------------------------------------------------


class TestFitOptions:
    @staticmethod
    def hinge(atoms, keep_numbers, angle_degrees=15.0):
        """Rotate every receptor residue that is not in ``keep_numbers`` about the centroid of
        those residues, leaving the kept residues (and the peptide) where they are."""
        pose = atoms.copy()
        move = (pose.chain_id == "A") & ~np.isin(pose.res_id, keep_numbers)
        centre = pose.coord[move].astype(np.float64).mean(axis=0)
        angle = np.radians(angle_degrees)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        pose.coord[move] = ((pose.coord[move] - centre) @ rotation.T + centre).astype(np.float32)
        return pose

    def test_fitting_the_site_only_gives_a_local_frame(self, ycr, ycr_reference, ycr_site):
        pose = self.hinge(ycr, ycr_site)
        everywhere = superpose_receptor(pose, ycr_reference)
        local = superpose_receptor(pose, ycr_reference, options={"frame_residues": "site"})
        assert everywhere.ok and local.ok
        # the local fit uses the site residues alone, the global one every receptor residue
        assert local.n_fit_residues == len(ycr_site) and everywhere.n_fit_residues == 85
        assert region(local, "site") == pytest.approx(0.0, abs=1e-4)
        assert local.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)
        assert region(everywhere, "site") > 0.3  # the global fit is pulled by the hinge
        assert region(local, "all") > region(everywhere, "all")  # and the rest moved instead

    def test_fitting_a_named_residue_set(self, ycr, ycr_site):
        reference = Reference(YCR, "A", "B", residue_sets={"core": [("A", n) for n in ycr_site]})
        pose = self.hinge(ycr, ycr_site)
        result = superpose_receptor(pose, reference, options={"frame_residues": "core"})
        # the fit uses the named set alone
        assert result.n_fit_residues == len(ycr_site) and region(result, "core") == pytest.approx(
            0.0, abs=1e-4
        )

    def test_too_few_residues_in_the_fit_set(self, ycr):
        reference = Reference(YCR, "A", residue_sets={"few": ["A:25", "A:26", "A:50"]})
        result = superpose_receptor(ycr, reference, options={"frame_residues": "few"})
        assert result.status == "failed"
        assert "only 3 paired residues of frame_residues='few' to fit" in result.reason

    @staticmethod
    def with_outliers(atoms, numbers, seed=0):
        """Noise of 0.2 angstrom on every receptor atom; the residues in ``numbers`` moved by 10."""
        rng = np.random.default_rng(seed)
        pose = atoms.copy()
        receptor = pose.chain_id == "A"
        pose.coord[receptor] += rng.normal(scale=0.2, size=(receptor.sum(), 3)).astype(np.float32)
        for number in numbers:
            direction = rng.normal(size=3)
            rows = receptor & (pose.res_id == number)
            pose.coord[rows] += (10.0 * direction / np.linalg.norm(direction)).astype(np.float32)
        return pose

    def test_iqr_rejection_ignores_displaced_residues(self, ycr, ycr_reference):
        displaced = [30, 45, 60, 75, 90]
        pose = self.with_outliers(ycr, displaced)
        plain = superpose_receptor(pose, ycr_reference)
        robust = superpose_receptor(pose, ycr_reference, options={"outliers": "iqr"})
        assert plain.n_rejected == 0 and robust.n_rejected >= 5
        rejected = {r.number for _, r in robust.rejected_pairs}
        assert set(displaced) <= rejected
        assert robust.n_fit_residues == 85 - robust.n_rejected
        assert robust.receptor_rmsd_angstrom < 0.4 < plain.receptor_rmsd_angstrom
        # the region RMSD of all paired residues still sees the displaced ones
        assert region(robust, "all") > robust.receptor_rmsd_angstrom

    def test_iqr_rejects_nothing_in_a_perfect_copy(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(61))
        result = superpose_receptor(pose, ycr_reference, options={"outliers": "iqr"})
        assert result.n_rejected == 0 and result.rejected_pairs == ()
        assert result.receptor_rmsd_angstrom < 1e-4

    def test_iqr_stops_before_the_fit_set_gets_too_small(self, ycr, ycr_reference):
        pose = self.with_outliers(ycr, [30, 45, 60, 75, 90])
        result = superpose_receptor(
            pose, ycr_reference, options={"outliers": "iqr", "min_frame_residues": 84}
        )
        assert result.ok and result.n_rejected == 0
        assert any(w.startswith("outlier rejection stopped") for w in result.warnings)

    def test_iqr_with_backbone_atoms_rejects_whole_residues(self, ycr, ycr_reference):
        pose = self.with_outliers(ycr, [30, 45, 60, 75, 90])
        result = superpose_receptor(
            pose, ycr_reference, options={"outliers": "iqr", "frame_atoms": "backbone"}
        )
        assert result.n_rejected >= 5
        assert result.n_fit_atoms == 4 * result.n_fit_residues

    def test_a_single_round_of_rejection(self, ycr, ycr_reference):
        pose = self.with_outliers(ycr, [30, 45, 60, 75, 90])
        one = superpose_receptor(
            pose, ycr_reference, options={"outliers": "iqr", "iqr_max_iterations": 1}
        )
        many = superpose_receptor(pose, ycr_reference, options={"outliers": "iqr"})
        assert 5 <= one.n_rejected <= many.n_rejected


# ---------------------------------------------------------------------------
# the sequence alignment
# ---------------------------------------------------------------------------


class TestSequenceAlignment:
    SEQUENCE = (
        "MET LYS THR ALA TYR ILE ALA LYS GLN ARG GLY GLY GLY GLY SER LEU VAL GLU GLN "
        "HIS LEU ALA GLY"
    ).split()

    def test_a_tie_between_alignments_is_flagged(self):
        coords = np.random.default_rng(7).normal(size=(len(self.SEQUENCE), 3)) * 8
        reference = ca_chain(coords, self.SEQUENCE, first=101)
        keep = [i for i in range(len(self.SEQUENCE)) if i != 12]  # one glycine of the run of four
        pose = ca_chain(coords[keep], [self.SEQUENCE[i] for i in keep])
        result = superpose_receptor(
            pose, Reference(reference, "A"), options={"min_frame_residues": 10}
        )
        assert result.pairing == "sequence" and result.n_paired == 22
        assert result.alignment_ambiguous
        assert any("more than one optimal solution" in w for w in result.warnings)
        assert result.status == "ok"

    def test_a_unique_alignment_is_not_flagged(self):
        coords = np.random.default_rng(8).normal(size=(len(self.SEQUENCE), 3)) * 8
        reference = ca_chain(coords, self.SEQUENCE, first=101)
        keep = list(range(3, len(self.SEQUENCE)))  # trimmed at the start
        pose = ca_chain(coords[keep], [self.SEQUENCE[i] for i in keep])
        result = superpose_receptor(
            pose, Reference(reference, "A"), options={"min_frame_residues": 10}
        )
        assert result.pairing == "sequence" and not result.alignment_ambiguous
        assert [r.number for _, r in result.pairs] == list(range(104, 101 + len(self.SEQUENCE)))
        assert result.receptor_rmsd_angstrom == pytest.approx(0.0, abs=1e-4)

    def test_non_standard_residues_are_x_and_pair_only_when_named_alike(self):
        names = list(self.SEQUENCE)
        names[5] = "NMG"
        coords = np.random.default_rng(9).normal(size=(len(names), 3)) * 8
        reference = ca_chain(coords, names, first=101)
        same = ca_chain(coords, names)
        other = ca_chain(coords, [*names[:5], "MLE", *names[6:]])
        options = {"min_frame_residues": 10}
        assert superpose_receptor(same, Reference(reference, "A"), options=options).n_paired == 23
        assert superpose_receptor(other, Reference(reference, "A"), options=options).n_paired == 22


# ---------------------------------------------------------------------------
# result schema, application and renumbering
# ---------------------------------------------------------------------------


class TestResultSchema:
    def test_the_frame_block_has_the_design_keys(self, ycr, ycr_reference):
        data = superpose_receptor(moved(ycr, *random_rigid(71)), ycr_reference).to_dict()
        for key in (
            "pairing",
            "n_receptor_residues_pose",
            "n_paired",
            "paired_fraction",
            "sequence_identity",
            "receptor_rmsd_angstrom",
            "n_rejected",
            "region_rmsd_angstrom",
            "transform",
            "alignment_ambiguous",
        ):
            assert key in data
        assert set(data["transform"]) == {"rotation", "translation"}
        assert np.array(data["transform"]["rotation"]).shape == (3, 3)
        json.dumps(data, allow_nan=False)

    def test_options_for_the_dictionary(self, ycr, ycr_reference):
        result = superpose_receptor(ycr, ycr_reference)
        assert "transform" not in result.to_dict(include_transform=False)
        assert "pairs" not in result.to_dict()
        pairs = result.to_dict(include_pairs=True)["pairs"]
        assert pairs[0] == [["A", 25, ""], ["A", 25, ""]] and len(pairs) == 85

    def test_a_failed_result_is_json_ready(self, ycr, ycr_reference):
        data = superpose_receptor(ycr[ycr.chain_id == "B"], ycr_reference).to_dict()
        assert data["status"] == "failed" and data["transform"] is None
        assert data["receptor_rmsd_angstrom"] is None and data["region_rmsd_angstrom"] == {}
        json.dumps(data, allow_nan=False)

    def test_results_are_frozen(self, ycr, ycr_reference):
        result = superpose_receptor(ycr, ycr_reference)
        assert isinstance(result, FrameResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = "failed"


class TestPickling:
    @pytest.mark.parametrize(
        "clone", [lambda o: pickle.loads(pickle.dumps(o)), copy.deepcopy, copy.copy]
    )
    def test_a_frame_result_survives_with_every_value(self, ycr, ycr_reference, ycr_site, clone):
        reference = Reference(YCR, "A", "B", residue_sets={"core": [("A", n) for n in ycr_site]})
        result = superpose_receptor(moved(ycr, *random_rigid(91)), reference)
        again = clone(result)
        assert again is not result and again.ok
        assert again.to_dict(include_pairs=True) == result.to_dict(include_pairs=True)
        assert again.transform.allclose(result.transform, atol=0.0)
        assert again.pairs == result.pairs and again.warnings == result.warnings
        assert set(again.region_rmsd_angstrom) == {"all", "site", "core"}

    def test_the_mapping_views_stay_read_only(self, ycr, ycr_reference):
        again = pickle.loads(pickle.dumps(superpose_receptor(ycr, ycr_reference)))
        for name in (
            "region_rmsd_angstrom",
            "region_n_residues",
            "region_n_paired",
            "region_reasons",
        ):
            with pytest.raises(TypeError):
                getattr(again, name)["x"] = 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            again.status = "failed"

    def test_a_failed_result_survives(self, ycr, ycr_reference):
        failed = superpose_receptor(ycr[ycr.chain_id == "B"], ycr_reference)
        again = pickle.loads(pickle.dumps(failed))
        assert again.status == "failed" and again.reason == failed.reason
        assert again.transform is None and again.to_dict() == failed.to_dict()

    def test_a_transform_stays_read_only_and_equal_in_value(self):
        transform = Transform(*random_rigid(92))
        again = pickle.loads(pickle.dumps(transform))
        assert again.allclose(transform, atol=0.0)
        assert not again.rotation.flags.writeable and not again.translation.flags.writeable
        with pytest.raises(ValueError):
            again.rotation[0, 0] = 2.0

    def test_options_and_pairings_survive(self, ycr, ycr_reference):
        options = FrameOptions(
            frame_atoms="backbone", outliers="iqr", alignment_gap_penalty=(-12, -2)
        )
        assert pickle.loads(pickle.dumps(options)) == options
        pairing = pair_residues(ycr, ycr_reference)
        again = pickle.loads(pickle.dumps(pairing))
        assert again.pose_keys == pairing.pose_keys and again.method == pairing.method

    def test_a_copied_reference_gives_the_same_frame(self, ycr, ycr_reference):
        pose = moved(ycr, *random_rigid(93))
        again = pickle.loads(pickle.dumps(ycr_reference))
        assert (
            superpose_receptor(pose, again).to_dict()
            == superpose_receptor(pose, ycr_reference).to_dict()
        )


class TestPlaceInReferenceFrame:
    def test_a_transform_may_be_given_directly_and_a_stack_is_moved(self, ycr):
        transform = Transform(*random_rigid(81))
        stack = struc.stack([ycr, ycr])
        placed = place_in_reference_frame(stack, transform)
        expected = transform.apply(ycr.coord).astype(np.float32)
        assert np.allclose(placed.coord[1], expected, atol=1e-5)
        assert np.array_equal(stack.coord[0], ycr.coord)

    def test_the_annotations_and_bonds_are_kept(self, ycr):
        atoms = ycr.copy()
        atoms.bonds = struc.connect_via_residue_names(atoms)
        placed = place_in_reference_frame(atoms, Transform(*random_rigid(82)))
        assert placed.bonds == atoms.bonds
        assert (placed.atom_name == atoms.atom_name).all()
        assert placed.coord.dtype == atoms.coord.dtype

    def test_wrong_frame_type(self, ycr):
        with pytest.raises(TypeError, match="FrameResult or a Transform"):
            place_in_reference_frame(ycr, "frame")


class TestRenumberToReference:
    def test_paired_residues_get_the_reference_numbers_and_the_peptide_keeps_its_own(
        self, ycr, ycr_reference
    ):
        pose = consecutive(consecutive(ycr, "A"), "B")
        result = superpose_receptor(pose, ycr_reference)
        before = (pose.res_id.copy(), pose.ins_code.copy(), pose.coord.copy())
        renumbered = renumber_to_reference(pose, result, ycr_reference)
        assert (renumbered.res_id == ycr.res_id)[ycr.chain_id == "A"].all()
        assert (renumbered.res_id[pose.chain_id == "B"] == pose.res_id[pose.chain_id == "B"]).all()
        assert np.array_equal(renumbered.coord, pose.coord)
        assert np.array_equal(pose.res_id, before[0]) and np.array_equal(pose.ins_code, before[1])
        assert np.array_equal(pose.coord, before[2])

    def test_insertion_codes_of_the_reference_are_written(self, sfti):
        reference = Reference(SFTI, "A", "I")
        pose = consecutive(consecutive(sfti, "A"), "I")
        result = superpose_receptor(pose, reference)
        renumbered = renumber_to_reference(pose, result, reference)
        receptor = renumbered.chain_id == "A"
        assert (renumbered.res_id[receptor] == sfti.res_id[sfti.chain_id == "A"]).all()
        assert (renumbered.ins_code[receptor] == sfti.ins_code[sfti.chain_id == "A"]).all()
        assert {"B", "I"} <= set(renumbered.ins_code[receptor])

    def test_unpaired_residues_get_numbers_above_the_reference_maximum(self, ycr, ycr_reference):
        pose = consecutive(ycr, "A")
        mapping = {("A", i): ("A", i + 24) for i in range(1, 61)}  # residues 61..85 are unpaired
        result = superpose_receptor(pose, ycr_reference, residue_map=mapping)
        renumbered = renumber_to_reference(pose, result, ycr_reference)
        receptor = renumbered.chain_id == "A"
        runs = residue_runs(renumbered, "A")
        numbers = [int(renumbered.res_id[run][0]) for run in runs]
        assert numbers[:60] == list(range(25, 85))
        assert numbers[60:] == list(range(110, 135))  # one number per unpaired residue, file order
        assert (renumbered.ins_code[receptor] == "").all()
        assert len(set(numbers)) == 85  # labels cannot collide


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------


def test_importing_and_the_numpy_parts_need_no_biotite():
    script = textwrap.dedent(
        """
        import sys
        sys.modules["biotite"] = None
        import numpy as np
        from binding_metrics.poses import frame
        points = np.random.default_rng(0).normal(size=(10, 3))
        shifted = points + 1.0
        transform = frame.kabsch_transform(points, shifted)
        assert np.allclose(transform.apply(points), shifted)
        assert frame.FrameOptions().pairing == "auto"
        assert frame.canonical_residue_name("HIE") == "HIS"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
