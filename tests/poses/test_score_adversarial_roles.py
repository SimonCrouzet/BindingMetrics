"""``evobind_adversarial`` in a pose set: the reference is the design, the pose is the prediction.

The check compares the design with a second, predicted structure and divides by the pLDDT of that
second structure. In a pose set the second structure is the pose, so the layer gives the reference
as ``design_structure_path`` and the pose as ``afm_structure_path`` with the pose's per-atom pLDDT.
The displacement of the binder centre of mass is symmetric in the two structures and keeps its
value. A pose without pLDDT has no score, and the pre-flight check says why.
"""

import importlib

import numpy as np
import pytest

from binding_metrics.capabilities import IncompatibleInputError
from binding_metrics.metrics.evobind import (
    compute_evobind_adversarial_check,
    compute_evobind_adversarial_from_records,
)
from binding_metrics.poses import Reference, score_pose_set
from binding_metrics.predictors import get_parser
from binding_metrics.testing import make_synthetic_poses
from tests.predictors import synth
from tests.test_feat_c_support import EXAMPLE_1YCR, complex_from

pytest.importorskip("biotite.structure")

CHECK = "evobind_adversarial"
SHIFT_ANGSTROM = 3.0


def write(root, model, name, truth):
    out = root / model
    out.mkdir(exist_ok=True)
    importlib.import_module(f"tests.predictors.synth_{model}").write_prediction(out, name, truth)
    return get_parser(model).load(out, name)


@pytest.fixture(scope="module")
def poses(tmp_path_factory):
    """OpenFold3 poses of 1YCR: one on the crystal, one with the binder moved by 3 angstrom."""
    root = tmp_path_factory.mktemp("adversary")
    truth = complex_from(EXAMPLE_1YCR)
    moved = truth.atoms.copy()
    moved.coord[moved.chain_id == "B"] += np.array([SHIFT_ANGSTROM, 0.0, 0.0], dtype=np.float32)
    shifted = synth.SyntheticComplex(
        atoms=moved,
        plddt_per_atom=truth.plddt_per_atom,
        pae=truth.pae,
        pde=truth.pde,
        scalars=truth.scalars,
        chain_ptm=truth.chain_ptm,
        chain_pair_iptm=truth.chain_pair_iptm,
    )
    return {
        "same": write(root, "of3", "same", truth),
        "moved": write(root, "of3", "moved", shifted),
    }


def score(selected, **kwargs):
    kwargs.setdefault("reference", Reference(EXAMPLE_1YCR, "A", "B"))
    kwargs.setdefault("metrics", [CHECK])
    return score_pose_set(list(selected), "B", "A", **kwargs)


def block_of(result, name):
    (row,) = [row for row in result["poses"] if row["provenance"].get("name") == name]
    return row[CHECK]


class TestPredictedPoses:
    def test_the_score_is_a_number_and_the_pose_supplies_the_plddt(self, poses):
        result = score(poses.values())
        for row in result["poses"]:
            assert row["status"] == "ok" and not row["reasons"]
            block = row[CHECK]
            assert isinstance(block["evobind_adversarial_score"], float)
            assert np.isfinite(block["evobind_adversarial_score"])
            assert np.isfinite(block["afm_mean_plddt_binder"])
            assert "reason" not in block
        entry = result["aggregates"]["of3"][f"{CHECK}.evobind_adversarial_score"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 2)

    def test_the_binder_pldt_is_the_one_of_the_pose(self, poses):
        # the synthetic pLDDT rises along the atoms, so the binder (the last chain) is high
        block = block_of(score(poses.values()), "same")
        reference = compute_evobind_adversarial_from_records(EXAMPLE_1YCR, poses["same"], "B", "A")
        assert block["afm_mean_plddt_binder"] == pytest.approx(
            reference["afm_mean_plddt_binder"], rel=1e-6
        )
        assert block["afm_mean_plddt_binder"] > 60.0

    def test_the_numbers_are_those_of_the_reference_as_design_and_the_pose_as_adversary(
        self, poses
    ):
        result = score(poses.values())
        for name in ("same", "moved"):
            block = block_of(result, name)
            expected = compute_evobind_adversarial_from_records(EXAMPLE_1YCR, poses[name], "B", "A")
            for key in (
                "delta_com_angstrom",
                "afm_if_dist_pep_to_rec",
                "afm_if_dist_rec_to_pep",
                "afm_mean_if_dist",
                "afm_mean_plddt_binder",
                "evobind_adversarial_score",
            ):
                assert block[key] == pytest.approx(expected[key], rel=1e-4, abs=1e-4), (name, key)

    def test_the_displacement_is_the_one_of_the_moved_binder(self, poses):
        result = score(poses.values())
        assert block_of(result, "same")["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)
        assert block_of(result, "moved")["delta_com_angstrom"] == pytest.approx(
            SHIFT_ANGSTROM, abs=1e-2
        )

    def test_the_displacement_does_not_depend_on_which_structure_is_the_design(self, poses):
        swapped = compute_evobind_adversarial_check(
            design_structure_path=poses["moved"].structure_path,
            afm_structure_path=EXAMPLE_1YCR,
            binder_chain="B",
            receptor_chain="A",
        )
        assert block_of(score(poses.values()), "moved")["delta_com_angstrom"] == pytest.approx(
            swapped["delta_com_angstrom"], abs=1e-3
        )

    def test_the_design_origin_still_leaves_the_check_out_for_a_matching_back_end(
        self, poses, tmp_path
    ):
        boltz = write(tmp_path, "boltz2", "boltzpose", complex_from(EXAMPLE_1YCR))
        result = score([poses["same"], boltz], design_source="boltzgen")
        rows = {row["provenance"]["backend"]: row for row in result["poses"]}
        assert CHECK not in rows["boltz2"] and CHECK in rows["of3"]
        assert "the design was made with boltzgen" in rows["boltz2"]["reasons"][CHECK]
        assert np.isfinite(rows["of3"][CHECK]["evobind_adversarial_score"])
        assert [row["independent_of_design"] for row in result["poses"]] == [False, True]


@pytest.fixture(scope="module")
def docked():
    return make_synthetic_poses(backends=("backend_a",), seeds=(1, 2)).poses


class TestPosesWithoutPlddt:
    def test_a_set_of_docked_poses_is_refused_with_the_reason(self, docked):
        with pytest.raises(IncompatibleInputError) as caught:
            score(docked)
        message = str(caught.value)
        assert "predicted poses with per-atom pLDDT" in message
        assert "the reference is the design" in message
        assert "leave 'evobind_adversarial' out of the metric list" in message
        assert [v.name for v in caught.value.violations] == [CHECK]

    def test_with_warn_the_score_is_none_and_the_row_says_why(self, docked):
        result = score(docked, preflight="warn")
        for row in result["poses"]:
            block = row[CHECK]
            assert block["evobind_adversarial_score"] is None
            assert block["afm_mean_plddt_binder"] is None
            assert np.isfinite(block["delta_com_angstrom"])
            assert "no per-atom pLDDT" in block["reason"]
            assert row["status"] == "partial" and row["reasons"][CHECK] == block["reason"]
        assert [v["name"] for v in result["preflight"]["violations"]] == [CHECK]
        assert (
            "predicted poses with per-atom pLDDT" in result["preflight"]["violations"][0]["reason"]
        )

    def test_the_other_checks_of_the_set_are_not_held_up(self, docked):
        result = score(docked, preflight="skip", metrics=[CHECK, "delta_sasa_static"])
        assert [m["name"] for m in result["preflight"]["metrics_skipped"]] == [CHECK]
        for row in result["poses"]:
            assert "delta_sasa_static" in row and CHECK not in row
            assert row["reasons"][CHECK].startswith("left out by the pre-flight check")

    def test_a_docked_pose_among_predicted_ones_has_a_score_of_none_and_a_reason(
        self, poses, docked
    ):
        result = score([*poses.values(), docked[0]], preflight="warn")
        rows = {row["id"]: row for row in result["poses"]}
        assert rows[docked[0].id][CHECK]["evobind_adversarial_score"] is None
        assert "no per-atom pLDDT" in rows[docked[0].id]["reasons"][CHECK]
        for name in ("same", "moved"):
            assert block_of(result, name)["evobind_adversarial_score"] > 0
