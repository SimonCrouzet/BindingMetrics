"""Poses with the same content are named in a warning (issue #190).

Two stored predictions with one structure file (a run that also wrote embeddings repeats the
structure of the plain run) are two poses of the set: both count in every ``n_total`` and in the
agreement block, where they sit at distance 0. The counts do not change, the result says so.
"""

import pytest

from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose
from binding_metrics.poses.score import score_pose_set
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")


@pytest.fixture(scope="module")
def distinct():
    return make_synthetic_poses().poses[:3]


def _copy(pose, pose_id):
    return DockedPose.from_atoms(pose.atoms().copy(), id=pose_id, **pose.provenance)


def _same_content(result):
    return [text for text in result["warnings"] if "same structure" in text]


class TestDuplicateContent:
    def test_two_poses_with_one_structure_give_one_warning_naming_both(self, distinct):
        poses = [*distinct, _copy(distinct[0], "copy-of-s1")]
        result = score_pose_set(poses, "B", "A", ["interface"])
        (warning,) = _same_content(result)
        assert "backend_a-s1" in warning and "copy-of-s1" in warning
        assert "backend_a-s2" not in warning and "backend_a-s3" not in warning

    def test_the_warning_gives_the_content_hash(self, distinct):
        poses = [*distinct, _copy(distinct[0], "copy-of-s1")]
        result = score_pose_set(poses, "B", "A", ["interface"])
        digest = result["poses"][0]["content_hash"]
        assert digest[:12] in _same_content(result)[0]

    def test_poses_are_grouped_by_hash(self, distinct):
        poses = [
            *distinct,
            _copy(distinct[0], "copy-of-s1"),
            _copy(distinct[0], "second-copy-of-s1"),
            _copy(distinct[1], "copy-of-s2"),
        ]
        result = score_pose_set(poses, "B", "A", ["interface"])
        warnings = _same_content(result)
        assert len(warnings) == 2
        triple = next(text for text in warnings if "second-copy-of-s1" in text)
        assert all(pose_id in triple for pose_id in ("backend_a-s1", "copy-of-s1"))
        pair = next(text for text in warnings if "copy-of-s2" in text)
        assert "backend_a-s2" in pair and "backend_a-s1" not in pair

    def test_the_ids_of_a_group_are_listed_in_id_order(self, distinct):
        poses = [*distinct, _copy(distinct[0], "a-copy"), _copy(distinct[0], "z-copy")]
        warning = _same_content(score_pose_set(poses, "B", "A", ["interface"]))[0]
        assert warning.index("a-copy") < warning.index("backend_a-s1") < warning.index("z-copy")

    def test_the_counts_do_not_change(self, distinct):
        poses = [*distinct, _copy(distinct[0], "copy-of-s1")]
        result = score_pose_set(poses, "B", "A", ["interface"])
        entry = result["aggregates"]["all"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"]) == (4, 4)
        assert len(result["poses"]) == 4

    def test_distinct_poses_give_no_warning(self, distinct):
        result = score_pose_set(distinct, "B", "A", ["interface"])
        assert _same_content(result) == []

    def test_poses_without_a_hash_are_not_grouped(self, distinct, tmp_path):
        gone = []
        for name in ("x-1", "x-2"):
            path = tmp_path / f"{name}.cif"
            path.write_text("data_x\n", encoding="utf-8")
            gone.append(ExperimentalPose.from_file(path, id=name))
            path.unlink()
        result = score_pose_set([*distinct, *gone], "B", "A", ["interface"])
        rows = {row["id"]: row for row in result["poses"]}
        assert rows["x-1"]["content_hash"] is None and rows["x-2"]["content_hash"] is None
        assert _same_content(result) == []
