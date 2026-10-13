"""Medoid caveats in the pose-set result: where they appear and what they must not break."""

import json
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics._common import load_structure
from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses.agreement import INCOMPLETE_MEDOID_CAVEAT
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.report import write_pose_report
from binding_metrics.poses.score import (
    REPRESENTATIVE_META_KEYS,
    PoseSetResult,
    score_pose_set,
)
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")

SFTI = Path(__file__).resolve().parents[2] / "data" / "example_bicyclic_sfti1_3P8F.cif"
HEAVY = {"agreement": {"kind": "heavy"}}
CALLS: list[str] = []


def _stub_heavy(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {"value": 1.0}


@pytest.fixture
def heavy_stub(monkeypatch):
    CALLS.clear()
    monkeypatch.setitem(
        METRICS_BY_NAME,
        "stub_heavy",
        MetricSpec(
            name="stub_heavy",
            import_path=f"{__name__}:_stub_heavy",
            description="stub metric of the caveat tests",
            input_type="static_structure",
            chain_mode="interface",
            path_arg="cif_path",
            peptide_chain_arg="binder_chain",
            receptor_chain_arg="receptor_chain",
            cost_class="structural",
        ),
    )
    yield
    CALLS.clear()


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def stripped(synthetic):
    """Three poses of one back-end; the middle one has no side chains on the peptide.

    For the heavy atoms its pairs share 65 of 113 atoms (0.58), below the default 0.8, so
    two of the three distances are undefined and no pose has a complete row.
    """
    atoms = synthetic.poses[1].atoms()
    keep = ~((atoms.chain_id == "B") & ~np.isin(atoms.atom_name, ["N", "CA", "C", "O", "CB"]))
    pose = DockedPose.from_atoms(atoms[keep], id="backend_a-s2", **synthetic.poses[1].provenance)
    return [synthetic.poses[0], pose, synthetic.poses[2]]


@pytest.fixture(scope="module")
def caveat_result(stripped):
    return score_pose_set(stripped, "B", "A", options=HEAVY)


@pytest.fixture(scope="module")
def open_ring_poses():
    """The 3P8F complex twice and a copy whose head-to-tail amide is pulled open."""
    atoms = load_structure(SFTI)
    opened = atoms.copy()
    last_c = np.flatnonzero((opened.chain_id == "I") & (opened.atom_name == "C"))[-1]
    opened.coord[last_c] += np.array([4.0, 0.0, 0.0], dtype=np.float32)
    return [
        DockedPose.from_atoms(atoms, id="a_closed", backend="x"),
        DockedPose.from_atoms(atoms.copy(), id="b_closed", backend="x"),
        DockedPose.from_atoms(opened, id="c_open", backend="x"),
    ]


class TestWhereTheCaveatAppears:
    def test_the_setup_has_undefined_pairs_and_no_complete_row(self, caveat_result):
        pairs = caveat_result["agreement"]["rmsd"]["heavy"]["matrix"]["undefined_pairs"]
        assert [p["pair"] for p in pairs] == [
            ["backend_a-s1", "backend_a-s2"],
            ["backend_a-s2", "backend_a-s3"],
        ]
        assert "below min_shared_atom_fraction 0.8" in pairs[0]["reason"]

    def test_the_representatives_carry_the_caveat_per_group(self, caveat_result):
        representatives = caveat_result["representatives"]
        assert representatives["caveats"] == {
            "all": INCOMPLETE_MEDOID_CAVEAT,
            "backend_a": INCOMPLETE_MEDOID_CAVEAT,
        }
        assert representatives["backend_a"] == "backend_a-s1"  # a tie with s3 goes to the lowest id
        assert representatives["reasons"] == {}

    def test_the_medoids_of_the_agreement_block_carry_it_too(self, caveat_result):
        medoids = caveat_result["agreement"]["medoids"]
        assert medoids["overall"]["caveat"] == INCOMPLETE_MEDOID_CAVEAT
        assert medoids["per_group"]["backend_a"]["caveat"] == INCOMPLETE_MEDOID_CAVEAT
        assert medoids["per_group"]["backend_a"]["pose_id"] == "backend_a-s1"
        assert medoids["per_group"]["backend_a"]["excluded"] == ["backend_a-s2"]

    def test_the_warnings_name_each_group_once(self, caveat_result):
        warnings = [w for w in caveat_result["warnings"] if INCOMPLETE_MEDOID_CAVEAT in w]
        assert sorted(warnings) == [
            f"medoid of 'all': {INCOMPLETE_MEDOID_CAVEAT}",
            f"medoid of 'backend_a': {INCOMPLETE_MEDOID_CAVEAT}",
        ]

    def test_only_the_groups_without_a_complete_row_are_named(self, stripped, synthetic):
        result = score_pose_set(list(stripped) + list(synthetic.poses[3:]), "B", "A", options=HEAVY)
        warnings = [w for w in result["warnings"] if INCOMPLETE_MEDOID_CAVEAT in w]
        # the stripped pose has an undefined distance to every other pose, so no pose of the
        # whole set has a complete row; inside backend_b every row is complete
        assert [w.split(":")[0] for w in warnings] == ["medoid of 'all'", "medoid of 'backend_a'"]
        assert sorted(result["representatives"]["caveats"]) == ["all", "backend_a"]
        assert result["representatives"]["backend_b"] == "backend_b-s2"

    def test_a_different_pairing_rule_makes_the_pair_defined_and_the_caveat_go(self, stripped):
        options = {"agreement": {"kind": "heavy", "min_shared_atom_fraction": 0.4}}
        result = score_pose_set(stripped, "B", "A", options=options)
        assert "caveats" not in result["representatives"]
        assert not any(INCOMPLETE_MEDOID_CAVEAT in w for w in result["warnings"])
        assert result["representatives"]["backend_a"] == "backend_a-s2"

    def test_a_clean_set_has_no_caveat_key_and_no_warning(self, synthetic):
        result = score_pose_set(synthetic.poses, "B", "A")
        assert "caveats" not in result["representatives"]
        assert result["warnings"] == []
        assert all("caveat" not in m for m in result["agreement"]["medoids"]["per_group"].values())

    def test_the_other_atom_sets_of_the_same_poses_do_not_trigger_it(self, stripped):
        # kind="ca" is complete for the same poses: the caveat is about the chosen matrix
        result = score_pose_set(stripped, "B", "A")
        assert "caveats" not in result["representatives"]

    def test_a_closure_mismatch_that_removes_the_pairs_gives_the_caveat(self, open_ring_poses):
        result = score_pose_set(
            open_ring_poses, "I", "A", options={"agreement": {"exclude_closure_mismatch": True}}
        )
        assert [row["profile"]["closures"] for row in result["poses"]][2] == ["disulfide"]
        assert result["agreement"]["closure_mismatch_pairs"] == [
            ["a_closed", "c_open"],
            ["b_closed", "c_open"],
        ]
        assert result["representatives"]["caveats"] == {
            "all": INCOMPLETE_MEDOID_CAVEAT,
            "x": INCOMPLETE_MEDOID_CAVEAT,
        }
        # the closed rings are 0 apart and the open one is not compared: the lowest id wins
        assert result["representatives"]["x"] == "a_closed"
        warnings = [w for w in result["warnings"] if INCOMPLETE_MEDOID_CAVEAT in w]
        assert len(warnings) == 2

    def test_without_the_exclusion_the_pairs_have_a_value_and_no_caveat(self, open_ring_poses):
        result = score_pose_set(open_ring_poses, "I", "A")
        assert result["agreement"]["closure_mismatch_pairs"]  # still flagged
        assert "caveats" not in result["representatives"]

    def test_the_confidence_rule_has_no_representative_caveat_but_the_medoids_still_warn(
        self, stripped
    ):
        result = score_pose_set(stripped, "B", "A", representative="confidence", options=HEAVY)
        assert "caveats" not in result["representatives"]
        warnings = [w for w in result["warnings"] if INCOMPLETE_MEDOID_CAVEAT in w]
        assert len(warnings) == 2


class TestNothingBreaks:
    def test_the_meta_keys_are_the_ones_that_are_not_groups(self):
        assert REPRESENTATIVE_META_KEYS == ("rule", "reasons", "caveats")

    def test_caveats_is_not_a_group_of_the_aggregates(self, caveat_result):
        assert list(caveat_result["aggregates"]) == ["backend_a", "all"]
        assert list(caveat_result["agreement"]["medoids"]["per_group"]) == ["backend_a"]

    def test_the_heavy_metrics_run_on_the_chosen_poses_and_not_on_the_caveat(
        self, stripped, heavy_stub
    ):
        result = score_pose_set(stripped, "B", "A", metrics=["stub_heavy"], options=HEAVY)
        assert sorted(set(CALLS)) == ["pose_0000.cif"]
        assert [row["id"] for row in result["poses"] if "stub_heavy" in row] == ["backend_a-s1"]
        assert [row["status"] for row in result["poses"]] == ["ok"] * 3

    def test_the_flat_rows_and_the_csv_are_unchanged_in_shape(self, caveat_result, tmp_path):
        flat = caveat_result.to_flat_rows()
        assert len({tuple(row) for row in flat}) == 1
        assert not any("caveat" in name for name in flat[0])
        table = write_pose_report(caveat_result, "csv", tmp_path / "r.csv")
        assert table.read_text(encoding="utf-8").startswith("id,kind,status")

    def test_the_markdown_lists_the_group_names_without_the_meta_keys(self, caveat_result):
        text = caveat_result.summary()
        section = text.split("## Representatives")[1].split("\n## ")[0]
        assert "- backend_a: backend_a-s1" in section and "- all: backend_a-s1" in section
        assert "- caveats:" not in section and "- rule:" not in section
        assert f"caveat for backend_a: {INCOMPLETE_MEDOID_CAVEAT}" in section
        assert "## Warnings" in text and f"medoid of 'all': {INCOMPLETE_MEDOID_CAVEAT}" in text

    def test_the_html_and_the_json_carry_it(self, caveat_result, tmp_path):
        html = write_pose_report(caveat_result, "html", tmp_path / "r.html").read_text(
            encoding="utf-8"
        )
        assert INCOMPLETE_MEDOID_CAVEAT in html
        written = write_pose_report(caveat_result, "json", tmp_path / "r.json")
        data = json.loads(written.read_text(encoding="utf-8"))
        assert data["representatives"]["caveats"]["backend_a"] == INCOMPLETE_MEDOID_CAVEAT
        again = PoseSetResult.from_dict(data)
        assert again.to_json(include_timing=False) == caveat_result.to_json(include_timing=False)

    def test_the_result_does_not_depend_on_the_order_of_the_poses(self, stripped, caveat_result):
        again = score_pose_set(list(reversed(stripped)), "B", "A", options=HEAVY)
        assert again.to_json(include_timing=False) == caveat_result.to_json(include_timing=False)

    @pytest.mark.slow
    def test_a_pool_gives_the_same_result(self, stripped, caveat_result):
        pooled = score_pose_set(stripped, "B", "A", options=HEAVY, n_workers=2)
        assert pooled["representatives"] == caveat_result["representatives"]
        assert pooled["warnings"] == caveat_result["warnings"]
