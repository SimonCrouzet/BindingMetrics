"""The binder profile comes from a pose that can be placed on the reference (issue #189).

The pre-flight profile and the "binder profile differs" warnings were taken from the readable
pose with the lowest id, even when that pose then failed the frame stage, so a stray pose that
sorted first made every good pose differ from it.
"""

import pytest

from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.reference import Reference
from binding_metrics.poses.score import PoseScoringError, score_pose_set
from binding_metrics.testing import make_synthetic_poses
from tests.conftest import DATA_DIR
from tests.poses.test_frame import drop

pytest.importorskip("biotite.structure")

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
STRAY = "a-stray"


@pytest.fixture(scope="module")
def reference():
    return Reference(YCR, "A", "B")


@pytest.fixture(scope="module")
def good():
    return make_synthetic_poses().poses[:3]


def _stray(pose, pose_id=STRAY):
    """A pose that cannot be placed on 1YCR (another receptor sequence) with a shorter binder."""
    atoms = drop(pose.atoms(), "B", range(25, 30))
    atoms.res_name[atoms.chain_id == "A"] = "GLY"
    return DockedPose.from_atoms(atoms, id=pose_id, **pose.provenance)


def _differing(result):
    return [text for text in result["warnings"] if "binder profile differs" in text]


class TestProfileReference:
    def test_the_stray_pose_is_the_only_one_that_fails(self, good, reference):
        result = score_pose_set([_stray(good[0]), *good], "B", "A", ["interface"], reference)
        statuses = {row["id"]: row["status"] for row in result["poses"]}
        assert statuses.pop(STRAY) == "failed"
        assert set(statuses.values()) == {"ok"}

    def test_the_profile_is_that_of_a_pose_that_passes_the_frame(self, good, reference):
        result = score_pose_set([_stray(good[0]), *good], "B", "A", ["interface"], reference)
        assert result["preflight"]["profile"]["n_binder_residues"] == 13

    def test_the_good_poses_do_not_differ_from_a_failed_pose(self, good, reference):
        result = score_pose_set([_stray(good[0]), *good], "B", "A", ["interface"], reference)
        warned = _differing(result)
        assert all(text.startswith(STRAY) for text in warned)
        assert not any(good_pose.id in text.split(":")[0] for text in warned for good_pose in good)
        for row in result["poses"]:
            if row["id"] != STRAY:
                assert row["profile"]["differs_from_first"] == []

    def test_the_warning_names_the_pose_the_others_are_compared_with(self, good, reference):
        result = score_pose_set([_stray(good[0]), *good], "B", "A", ["interface"], reference)
        assert any(f"of {good[0].id} " in text for text in _differing(result))

    def test_a_set_where_no_pose_can_be_placed_keeps_the_first_readable_pose(self, good, reference):
        strays = [_stray(good[0], "a-stray"), _stray(good[1], "b-stray")]
        result = score_pose_set(strays, "B", "A", ["interface"], reference)
        assert [row["status"] for row in result["poses"]] == ["failed", "failed"]
        assert result["preflight"]["profile"]["n_binder_residues"] == 8
        assert result["references"]["implicit"] is False

    def test_the_first_pose_that_fails_the_frame_is_skipped_under_raise_too(self, good, reference):
        with pytest.raises(PoseScoringError) as caught:
            score_pose_set(
                [_stray(good[0]), *good], "B", "A", ["interface"], reference, on_error="raise"
            )
        assert caught.value.pose_id == STRAY
        assert caught.value.stage == "frame"

    def test_without_a_reference_the_first_readable_pose_is_still_the_reference(self, good):
        result = score_pose_set([_stray(good[0]), *good], "B", "A", ["interface"], preflight="off")
        assert result["references"]["implicit"] is True
        assert result["references"]["implicit_pose_id"] == STRAY
        assert result["poses"][0]["frame"]["status"] == "ok"
