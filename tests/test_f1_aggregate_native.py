"""The native scores are aggregated within a group, not in ``all`` (issue #191).

The ``all`` group pools every group, and the native scores of one model (its ranking score, its
pTM) are not comparable with those of another, so the means over both back-ends of a set meant
nothing. With one group per back-end the ``native`` block is aggregated in each group only.
"""

import pytest

from binding_metrics.poses.aggregate import aggregate_poses
from binding_metrics.poses.reference import Reference
from binding_metrics.poses.score import score_pose_set
from tests.test_f1_aggregate_keys import YCR, _predicted_poses, _row, _with

pytest.importorskip("biotite.structure")


@pytest.fixture(scope="module")
def poses(tmp_path_factory):
    return _predicted_poses(tmp_path_factory.mktemp("two_models"))


class TestScorePoseSet:
    def test_the_native_scores_are_formed_within_a_back_end(self, poses):
        result = score_pose_set(
            poses, "B", "A", metrics=["prediction"], reference=Reference(YCR, "A", "B")
        )
        aggregates = result["aggregates"]
        for group in ("boltz2", "of3"):
            assert aggregates[group]["native.ranking_score"]["n_ok"] == 2
        assert _with(aggregates["all"], "native.") == []
        assert _with(aggregates["all"], "prediction.") != []

    def test_without_groups_the_all_group_holds_the_native_scores(self, poses):
        result = score_pose_set(
            poses,
            "B",
            "A",
            metrics=["prediction"],
            reference=Reference(YCR, "A", "B"),
            group_by=(),
        )
        assert list(result["aggregates"]) == ["all"]
        assert result["aggregates"]["all"]["native.ranking_score"]["n_total"] == 4


class TestAggregatePoses:
    def test_a_per_group_block_is_not_pooled_in_all(self):
        rows = [
            _row("a1", "a", native={"iptm": 0.5}, interface={"delta_sasa": 1.0}),
            _row("b1", "b", native={"iptm": 0.6}, interface={"delta_sasa": 2.0}),
        ]
        aggregates = aggregate_poses(rows, per_group_blocks=("native",))
        assert "native.iptm" in aggregates["a"] and "native.iptm" in aggregates["b"]
        assert "native.iptm" not in aggregates["all"]
        assert aggregates["all"]["interface.delta_sasa"]["n_ok"] == 2

    def test_a_per_group_block_stays_in_all_without_groups(self):
        rows = [_row("a1", "a", native={"iptm": 0.5}), _row("b1", "b", native={"iptm": 0.6})]
        aggregates = aggregate_poses(rows, (), per_group_blocks=("native",))
        assert aggregates["all"]["native.iptm"]["n_ok"] == 2

    def test_the_default_pools_every_block(self):
        rows = [_row("a1", "a", native={"iptm": 0.5}), _row("b1", "b", native={"iptm": 0.6})]
        assert aggregate_poses(rows)["all"]["native.iptm"]["n_ok"] == 2
