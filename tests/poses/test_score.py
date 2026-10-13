"""score_pose_set and PoseSetResult: layout, synthetic truth, determinism, errors, arguments."""

import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.capabilities import IncompatibleInputError
from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses.aggregate import RESERVED_GROUP_KEYS
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.score import (
    RESERVED_ROW_NAMES,
    RESULT_KEYS,
    SCHEMA,
    SCHEMA_VERSION,
    AgreementOptions,
    PoseScoringError,
    PoseSetOptions,
    PoseSetResult,
    score_pose_set,
)
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")

#: Cheap static metrics: a pose takes a fraction of a second for all four.
METRICS = ("delta_sasa_static", "hbonds", "ramachandran", "omega")

IDS = (
    "backend_a-s1",
    "backend_a-s2",
    "backend_a-s3",
    "backend_b-s1",
    "backend_b-s2",
    "backend_b-s3",
)


# ---------------------------------------------------------------------------
# stub metrics, registered under a test name
# ---------------------------------------------------------------------------

STUB_CALLS: list[str] = []


def _stub_value(cif_path, binder_chain=None, receptor_chain=None):
    STUB_CALLS.append(Path(cif_path).name)
    return {"value": 2.5, "binder_chain": binder_chain, "receptor_chain": receptor_chain}


def _stub_raises(cif_path, binder_chain=None, receptor_chain=None):
    STUB_CALLS.append(Path(cif_path).name)
    if Path(cif_path).name == "pose_0001.cif":
        raise ValueError("stub failure")
    return {"value": 1.0}


def _stub_nan(cif_path, binder_chain=None, receptor_chain=None):
    return {"value": float("nan"), "other": 1.0}


def _stub_reason(cif_path, binder_chain=None, receptor_chain=None):
    return {"value": 4.0, "reason": "half of it could not be computed"}


def _stub_arrays(cif_path, binder_chain=None, receptor_chain=None):
    return {"values": np.arange(3, dtype=np.float32), "count": np.int64(7), "ok": np.bool_(True)}


def _stub_dotted(cif_path, binder_chain=None, receptor_chain=None):
    return {"min_distance": 1.5, "clash_atoms": {"2.2": 3, "3.0": 5}}


def _stub_lists(cif_path, binder_chain=None, receptor_chain=None):
    return {
        "head": [1.0, 2.0],
        "grid": [[1, 2], [3, 4]],
        "long": list(range(40)),
        "table": [{"a": 1}, {"a": 2}],
        "names": ["x", "y"],
    }


def _spec(name, function, cost_class="static"):
    return MetricSpec(
        name=name,
        import_path=f"{__name__}:{function}",
        description="stub metric of the score tests",
        input_type="static_structure",
        chain_mode="interface",
        path_arg="cif_path",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
        cost_class=cost_class,
    )


@pytest.fixture
def stubs(monkeypatch):
    """Register the stub metrics for one test and clear the call log."""
    STUB_CALLS.clear()
    for name, function, cost in (
        ("stub_value", "_stub_value", "static"),
        ("stub_raises", "_stub_raises", "static"),
        ("stub_nan", "_stub_nan", "static"),
        ("stub_reason", "_stub_reason", "static"),
        ("stub_arrays", "_stub_arrays", "static"),
        ("stub_dotted", "_stub_dotted", "static"),
        ("stub_lists", "_stub_lists", "static"),
        ("stub_heavy", "_stub_value", "structural"),
        ("stub_md", "_stub_value", "md"),
    ):
        monkeypatch.setitem(METRICS_BY_NAME, name, _spec(name, function, cost))
    yield
    STUB_CALLS.clear()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def result(synthetic):
    return score_pose_set(synthetic.poses, "B", "A", metrics=METRICS)


def comparable(result):
    """The result without what describes the run: timing and provenance."""
    data = json.loads(result.to_json(include_timing=False))
    data.pop("provenance")
    return data


# ---------------------------------------------------------------------------
# the layout of the result
# ---------------------------------------------------------------------------


class TestLayout:
    def test_the_keys_are_the_documented_ones_in_order(self, result):
        assert list(result) == list(RESULT_KEYS)

    def test_schema_and_version(self, result):
        assert result["schema"] == SCHEMA == "binding_metrics.pose_set_result"
        assert result["schema_version"] == SCHEMA_VERSION == 1
        assert isinstance(result["schema_version"], int)
        assert result.schema_version == 1

    def test_package_version_and_provenance(self, result):
        provenance = result["provenance"]
        assert result["package_version"] == provenance["package_version"]
        assert provenance["pose_set_schema_version"] == SCHEMA_VERSION
        assert provenance["n_poses"] == 6
        assert provenance["run"]["n_workers"] == 1
        assert provenance["schema_version"] == 1  # the provenance block's own version

    def test_chains_construct_and_references(self, synthetic):
        result = score_pose_set(synthetic.poses[:2], "B", "A", construct="1YCR")
        assert (result["construct"], result["binder_chain"], result["target_chain"]) == (
            "1YCR",
            "B",
            "A",
        )
        references = result["references"]
        assert references["implicit"] is True
        assert references["implicit_pose_id"] == "backend_a-s1"
        assert references["metrics_frame_variant"] == "first_pose"

    def test_it_is_a_plain_dict_that_json_writes_without_a_hook(self, result):
        assert isinstance(result, dict) and isinstance(result, PoseSetResult)
        assert json.loads(json.dumps(result)) == json.loads(result.to_json())

    def test_to_dict_is_a_deep_copy_of_plain_values(self, result):
        plain = result.to_dict()
        assert type(plain) is dict and type(plain["poses"][0]) is dict
        plain["poses"][0]["status"] = "changed"
        assert result["poses"][0]["status"] == "ok"

    def test_json_round_trip_through_from_dict(self, result):
        again = PoseSetResult.from_dict(json.loads(result.to_json()))
        assert isinstance(again, PoseSetResult)
        assert again.to_json(include_timing=False) == result.to_json(include_timing=False)

    def test_from_dict_rejects_another_schema_and_missing_keys(self, result):
        data = result.to_dict()
        with pytest.raises(ValueError, match="not a pose-set result"):
            PoseSetResult.from_dict({**data, "schema": "other"})
        del data["aggregates"]
        with pytest.raises(ValueError, match="aggregates"):
            PoseSetResult.from_dict(data)
        with pytest.raises(ValueError, match="mapping"):
            PoseSetResult.from_dict([])

    def test_the_rows_are_sorted_by_id(self, result):
        assert [row["id"] for row in result["poses"]] == list(IDS)
        assert result.rows is result["poses"]

    def test_a_row_has_the_reserved_fields_then_the_metric_blocks_in_request_order(self, result):
        row = result["poses"][0]
        assert list(row) == [
            "id",
            "kind",
            "content_hash",
            "provenance",
            "native",
            "status",
            "reason",
            "reasons",
            "independent_of_design",
            "profile",
            "frame",
            *METRICS,
        ]
        assert set(row) - set(METRICS) <= set(RESERVED_ROW_NAMES)

    def test_a_row_describes_its_pose(self, result, synthetic):
        pose = synthetic.poses[0]
        row = result.row(pose.id)
        assert row["kind"] == "docked"
        assert row["content_hash"] == pose.content_hash()
        assert row["provenance"]["backend"] == "backend_a"
        assert row["native"] == pose.native_scores()
        assert (row["status"], row["reason"], row["reasons"]) == ("ok", None, {})
        assert row["independent_of_design"] is None

    def test_row_raises_for_an_unknown_id(self, result):
        with pytest.raises(KeyError, match="no pose 'nope'"):
            result.row("nope")

    def test_the_profile_is_the_binder_of_the_pose(self, result):
        profile = result["poses"][0]["profile"]
        assert profile["n_binder_residues"] == 13
        assert profile["closures"] == ["none"]
        assert profile["closures_as_expected"] is True
        assert profile["differs_from_first"] == []

    def test_the_metric_blocks_are_what_the_registry_function_returned(self, result):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        block = result["poses"][0]["delta_sasa_static"]
        assert set(block) == {"delta_sasa", "sasa_peptide", "sasa_receptor", "sasa_complex"}
        # the first pose is the reference: its metrics frame is its own, so the temporary
        # file reproduces the plain call on the pose's coordinates
        reference = compute_delta_sasa_static(
            _written(make_synthetic_poses().poses[0]), peptide_chain="B", receptor_chain="A"
        )
        assert block["delta_sasa"] == pytest.approx(reference["delta_sasa"], rel=3e-3)

    def test_numpy_values_of_a_metric_become_plain_python(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["stub_arrays"])
        block = result["poses"][0]["stub_arrays"]
        assert block == {"values": [0.0, 1.0, 2.0], "count": 7, "ok": True}
        assert type(block["count"]) is int and type(block["ok"]) is bool
        json.dumps(result)


def _written(pose):
    """The pose's atoms in a CIF file, for a plain registry call."""
    import tempfile

    from binding_metrics.testing.synthetic import write_cif

    path = Path(tempfile.mkdtemp(prefix="bm_score_test_")) / "pose.cif"
    write_cif(pose.atoms(), path)
    return path


# ---------------------------------------------------------------------------
# what the synthetic set pins: analytic RMSD, medoids, clusters, aggregates
# ---------------------------------------------------------------------------


class TestSyntheticSet:
    def test_every_pose_is_placed_on_the_first_one(self, result):
        for row in result["poses"]:
            frame = row["frame"]
            assert row["status"] == "ok"
            assert frame["pairing"] == "number"
            assert (frame["n_paired"], frame["paired_fraction"]) == (85, 1.0)
            assert frame["receptor_rmsd_angstrom"] < 1e-3
            assert frame["variant"] == "first_pose"
            assert "variants" not in frame
            assert frame["region_rmsd_angstrom"]["all"] < 1e-3

    @pytest.mark.parametrize("kind", ["ca", "backbone", "heavy"])
    def test_the_pairwise_rmsd_is_the_closed_form(self, result, synthetic, kind):
        block = result["agreement"]["rmsd"][kind]["matrix"]
        assert block["pose_ids"] == list(IDS)
        matrix = np.array(block["matrix"], dtype=float)
        assert np.allclose(matrix, synthetic.truth.rmsd[kind], atol=1e-4)

    def test_the_shape_rmsd_of_a_translation_is_zero(self, result):
        matrix = np.array(result["agreement"]["shape_rmsd"]["ca"]["matrix"]["matrix"], dtype=float)
        assert np.allclose(matrix, 0.0, atol=1e-4)

    def test_within_and_between_statistics_are_the_offsets(self, result):
        statistics = result["agreement"]["rmsd"]["ca"]["statistics"]
        within = statistics["within"]
        assert within["backend_a"]["min"] == pytest.approx(0.4, abs=1e-4)
        assert within["backend_a"]["max"] == pytest.approx(0.8, abs=1e-4)
        assert within["backend_a"]["mean"] == pytest.approx(0.8 / 1.5, abs=1e-4)
        assert within["backend_b"]["max"] == pytest.approx(0.9, abs=1e-4)
        between = statistics["between"]["backend_a|backend_b"]
        assert between["n_pairs"] == 9
        assert between["min"] == pytest.approx(1.2, abs=1e-4)
        assert between["max"] == pytest.approx(2.9, abs=1e-4)
        assert between["min_pair"] == ["backend_a-s3", "backend_b-s1"]
        assert statistics["between_reason"] is None

    def test_the_medoids_are_the_middle_poses(self, result):
        medoids = result["agreement"]["medoids"]
        assert medoids["per_group"]["backend_a"]["pose_id"] == "backend_a-s2"
        assert medoids["per_group"]["backend_b"]["pose_id"] == "backend_b-s2"
        assert medoids["between_groups"]["backend_a|backend_b"]["distance_angstrom"] == (
            pytest.approx(2.3 - 0.4, abs=1e-4)
        )

    def test_the_default_clusters_are_the_two_backends(self, result):
        clusters = result["agreement"]["clusters"]
        assert list(clusters) == ["1.0", "2.0"]
        assert clusters["1.0"]["clusters"] == [list(IDS[:3]), list(IDS[3:])]
        assert clusters["1.0"]["mixed_clusters"] == []
        assert clusters["1.0"]["linkage"] == "complete"

    def test_the_clusters_at_other_cutoffs_follow_the_tie_rule(self, synthetic):
        options = {"agreement": {"cluster_cutoffs_angstrom": [0.5, 3.0]}}
        result = score_pose_set(synthetic.poses, "B", "A", options=options)
        clusters = result["agreement"]["clusters"]
        assert clusters["0.5"]["sizes"] == [2, 2, 1, 1]
        # the pairs (a1, a2) and (a2, a3) tie at 0.4 angstrom: the lower ranks merge first
        assert clusters["0.5"]["clusters"][0] == ["backend_a-s1", "backend_a-s2"]
        assert clusters["0.5"]["clusters"][1] == ["backend_b-s1", "backend_b-s2"]
        assert clusters["3.0"]["n_clusters"] == 1
        assert clusters["3.0"]["mixed_clusters"] == [0]

    def test_the_representatives_are_the_group_medoids_and_the_overall_one(self, result):
        # the offsets 0, 0.4, 0.8, 2.0, 2.3, 2.9: a3 and b1 both have a distance sum of 6.0,
        # and a tie goes to the lowest id
        assert result["representatives"] == {
            "backend_a": "backend_a-s2",
            "backend_b": "backend_b-s2",
            "all": "backend_a-s3",
            "rule": "medoid",
            "reasons": {},
        }

    def test_the_aggregates_count_and_average_the_poses_of_a_group(self, result):
        entry = result["aggregates"]["backend_a"]["delta_sasa_static.delta_sasa"]
        values = [r["delta_sasa_static"]["delta_sasa"] for r in result["poses"][:3]]
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (3, 3, True)
        assert entry["mean"] == pytest.approx(np.mean(values))
        assert entry["std"] == pytest.approx(np.std(values, ddof=1))
        assert (entry["min"], entry["max"]) == (min(values), max(values))
        assert entry["median"] == pytest.approx(np.median(values))
        assert list(result["aggregates"]) == ["backend_a", "backend_b", "all"]
        assert result["aggregates"]["all"]["frame.n_paired"]["n_total"] == 6

    def test_the_native_scores_are_aggregated_too(self, result):
        entry = result["aggregates"]["all"]["native.ranking_score"]
        assert entry["n_ok"] == 6

    def test_the_spread_per_position_and_the_overlap_of_the_interfaces(self, result):
        spread = result["agreement"]["spread"]
        assert set(spread["per_group"]) == {"backend_a", "backend_b"}
        # a translation moves every alpha carbon by the same amount: the spread is the
        # population standard deviation of the offsets of the group (0, 0.4, 0.8)
        expected = float(np.std([0.0, 0.4, 0.8]))
        assert spread["per_group"]["backend_a"]["mean_spread_angstrom"] == pytest.approx(
            expected, abs=1e-4
        )
        contacts = result["agreement"]["contacts"]
        assert set(contacts["interface_jaccard"]) == {"all", "receptor", "binder"}
        overall = contacts["contact_map_jaccard"]["statistics"]["overall"]
        assert 0.0 < overall["min"] <= overall["mean"] <= overall["max"] <= 1.0

    def test_a_single_back_end_has_no_between_block_and_says_why(self, synthetic):
        result = score_pose_set(synthetic.poses[:3], "B", "A")
        statistics = result["agreement"]["rmsd"]["ca"]["statistics"]
        assert statistics["between"] == {}
        assert statistics["between_reason"] == "only one back-end"
        assert result["agreement"]["medoids"]["between_groups"] == {}

    def test_one_pose_alone_gives_none_with_a_reason(self, synthetic):
        result = score_pose_set(synthetic.poses[:1], "B", "A", metrics=["hbonds"])
        within = result["agreement"]["rmsd"]["ca"]["statistics"]["within"]["backend_a"]
        assert within["min"] is None and "at least 2 poses" in within["reason"]
        entry = result["aggregates"]["backend_a"]["hbonds.hbonds"]
        assert entry["n_ok"] == 1 and entry["std"] is None
        assert "at least 2" in entry["statistic_reasons"]["std"]
        assert result["agreement"]["clusters"]["1.0"]["n_clusters"] is None
        assert result["representatives"]["backend_a"] == "backend_a-s1"

    def test_the_group_by_option_groups_by_provenance_keys(self, synthetic):
        result = score_pose_set(
            synthetic.poses, "B", "A", metrics=["hbonds"], group_by=("backend", "seed")
        )
        assert "backend_a|seed=1" in result["aggregates"]
        assert result["aggregates"]["backend_a|seed=1"]["hbonds.hbonds"]["n_total"] == 1
        assert result["representatives"]["backend_b|seed=3"] == "backend_b-s3"

    def test_without_groups_there_is_the_group_all_only(self, synthetic):
        result = score_pose_set(synthetic.poses, "B", "A", metrics=["hbonds"], group_by=())
        assert list(result["aggregates"]) == ["all"]
        assert result["agreement"]["groups"] is None
        assert result["agreement"]["medoids"]["per_group"] == {}

    @pytest.mark.parametrize("backend", RESERVED_GROUP_KEYS)
    def test_a_backend_with_a_reserved_name_is_refused_before_any_work(self, synthetic, backend):
        pose = DockedPose.from_atoms(synthetic.poses[0].atoms(), id="x", backend=backend)
        with pytest.raises(ValueError, match="reserved"):
            score_pose_set([pose], "B", "A")

    def test_the_confidence_rule_picks_the_best_native_score_per_backend(self, synthetic):
        result = score_pose_set(synthetic.poses, "B", "A", representative="confidence")
        best = {
            backend: max(
                (p for p in synthetic.poses if p.provenance["backend"] == backend),
                key=lambda p: p.native_scores()["ranking_score"],
            ).id
            for backend in ("backend_a", "backend_b")
        }
        assert result["representatives"]["backend_a"] == best["backend_a"]
        assert result["representatives"]["backend_b"] == best["backend_b"]
        assert result["representatives"]["all"] is None
        assert "not comparable" in result["representatives"]["reasons"]["all"]

    def test_a_callable_rule_chooses_the_representative(self, synthetic):
        def last(rows, distances):
            return rows[-1]["id"]

        result = score_pose_set(synthetic.poses, "B", "A", representative=last)
        assert result["representatives"]["rule"] == "custom"
        assert result["representatives"]["backend_a"] == "backend_a-s3"
        assert result["parameters"]["representative"] == "custom"

    def test_the_options_are_echoed_with_their_defaults(self, result):
        options = result["parameters"]["options"]
        assert options["frame"]["pairing"] == "auto"
        assert options["frame"]["min_frame_fraction"] == 0.5
        assert options["frame"]["renumber_receptor"] is True
        assert options["frame"]["metrics_frame"] == "reference"
        assert options["agreement"] == {
            "kind": "ca",
            "cluster_cutoffs_angstrom": [1.0, 2.0],
            "linkage": "complete",
            "symmetry": "standard",
            "cyclic_shift": "off",
            "residue_pairing": "auto",
            "expected_closures": None,
            "exclude_closure_mismatch": False,
            "contact_cutoff_angstrom": 5.0,
            "min_shared_atom_fraction": 0.8,
            "shape_rmsd": True,
        }
        assert options["aggregation"] == {"min_n_ok": 1, "reason_policy": "exclude"}
        assert options["representative"] == {
            "confidence_key": "ranking_score",
            "confidence_direction": None,
        }
        parameters = result["parameters"]
        assert parameters["aggregate"] == ["mean", "std", "min", "max", "median"]
        assert parameters["group_by"] == ["backend"]
        assert parameters["heavy"] == "representative"
        assert parameters["preflight"] == "error"

    def test_the_agreement_kind_decides_the_clusters_not_the_three_matrices(self, synthetic):
        result = score_pose_set(synthetic.poses, "B", "A", options={"agreement": {"kind": "heavy"}})
        assert result["agreement"]["kind"] == "heavy"
        assert set(result["agreement"]["rmsd"]) == {"ca", "backbone", "heavy"}

    def test_the_shape_rmsd_can_be_left_out(self, synthetic):
        result = score_pose_set(
            synthetic.poses[:3], "B", "A", options={"agreement": {"shape_rmsd": False}}
        )
        assert result["agreement"]["shape_rmsd"] is None


# ---------------------------------------------------------------------------
# the options objects
# ---------------------------------------------------------------------------


class TestOptions:
    def test_defaults_round_trip(self):
        options = PoseSetOptions()
        assert PoseSetOptions.from_dict(options.to_dict()) == options

    def test_a_mapping_and_an_object_are_the_same_call(self, synthetic):
        as_object = PoseSetOptions(agreement=AgreementOptions(cluster_cutoffs_angstrom=(0.7,)))
        as_mapping = {"agreement": {"cluster_cutoffs_angstrom": [0.7]}}
        a = score_pose_set(synthetic.poses[:3], "B", "A", options=as_object)
        b = score_pose_set(synthetic.poses[:3], "B", "A", options=as_mapping)
        assert comparable(a) == comparable(b)
        assert list(a["agreement"]["clusters"]) == ["0.7"]

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"nope": {}}, "unknown option groups"),
            ({"agreement": {"nope": 1}}, "unknown agreement options"),
            ({"frame": {"nope": 1}}, "unknown frame options"),
            ({"agreement": {"kind": "all"}}, "kind must be one of"),
            ({"agreement": {"cluster_cutoffs_angstrom": [0.0]}}, "positive finite"),
            ({"agreement": {"cluster_cutoffs_angstrom": []}}, "positive finite"),
            (
                {"agreement": {"min_shared_atom_fraction": 1.5}},
                "min_shared_atom_fraction must be in",
            ),
            ({"agreement": {"expected_closures": "disulfide"}}, "list of closure family names"),
            ({"aggregation": {"min_n_ok": 0}}, "min_n_ok"),
            ({"aggregation": {"reason_policy": "x"}}, "reason_policy must be one of"),
            ({"representative": {"confidence_direction": "up"}}, "confidence_direction"),
            ({"frame": {"min_frame_residues": 2}}, "at least 3"),
        ],
    )
    def test_a_bad_option_raises_before_any_pose_is_read(self, synthetic, options, message):
        with pytest.raises(ValueError, match=message):
            score_pose_set(synthetic.poses[:2], "B", "A", options=options)

    def test_options_of_another_type_raise(self, synthetic):
        with pytest.raises(TypeError, match="options must be"):
            score_pose_set(synthetic.poses[:2], "B", "A", options=3)
        with pytest.raises(TypeError, match="options.frame"):
            PoseSetOptions(frame=3)

    def test_an_explicit_expected_closure_is_compared_with_every_pose(self, synthetic):
        options = {"agreement": {"expected_closures": ["head_to_tail"]}}
        result = score_pose_set(synthetic.poses[:2], "B", "A", options=options)
        assert result["agreement"]["expected_closures"] == ["head_to_tail"]
        assert result["poses"][0]["profile"]["closures_as_expected"] is False


# ---------------------------------------------------------------------------
# determinism and parallelism
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDeterminism:
    def test_any_order_of_the_input_gives_the_same_text(self, synthetic, result):
        reference = result.to_json(include_timing=False)
        for order in (list(reversed(synthetic.poses)), synthetic.poses[3:] + synthetic.poses[:3]):
            again = score_pose_set(order, "B", "A", metrics=METRICS)
            assert again.to_json(include_timing=False) == reference

    def test_a_pool_of_workers_gives_the_same_values(self, synthetic, result):
        pooled = score_pose_set(synthetic.poses, "B", "A", metrics=METRICS, n_workers=2)
        assert pooled["provenance"]["run"]["n_workers"] == 2
        assert comparable(pooled) == comparable(result)

    def test_the_input_key_is_the_same_for_the_same_inputs(self, synthetic, result):
        pooled = score_pose_set(synthetic.poses, "B", "A", metrics=METRICS, n_workers=2)
        assert pooled.input_key == result.input_key
        assert len(result.input_key) == 64

    def test_the_input_key_changes_with_the_inputs_and_the_parameters(self, synthetic, result):
        fewer = score_pose_set(synthetic.poses[:5], "B", "A", metrics=METRICS)
        other_metrics = score_pose_set(synthetic.poses, "B", "A", metrics=METRICS[:2])
        other_option = score_pose_set(
            synthetic.poses, "B", "A", metrics=METRICS, options={"agreement": {"kind": "heavy"}}
        )
        keys = {r.input_key for r in (result, fewer, other_metrics, other_option)}
        assert len(keys) == 4

    def test_timing_is_the_only_thing_that_varies_and_it_is_in_its_own_block(self, result):
        timing = result["timing"]
        assert set(timing["stages_seconds"]) >= {"arguments", "poses", "agreement", "total"}
        assert set(timing["poses_seconds"]) == set(IDS)
        assert "timing" not in json.loads(result.to_json(include_timing=False))


# ---------------------------------------------------------------------------
# flat rows
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat(synthetic, tmp_path_factory):
    """Flat rows of a set with one pose that could not be read."""
    corrupt = tmp_path_factory.mktemp("flat") / "corrupt.pdb"
    corrupt.write_text("this is not a structure\n", encoding="utf-8")
    poses = list(synthetic.poses[:3]) + [DockedPose.from_file(corrupt, id="zz_corrupt")]
    return score_pose_set(poses, "B", "A", metrics=["hbonds"]).to_flat_rows()


class TestFlatRows:
    def test_one_row_per_pose_in_id_order(self, flat):
        assert [row["id"] for row in flat] == [
            "backend_a-s1",
            "backend_a-s2",
            "backend_a-s3",
            "zz_corrupt",
        ]

    def test_the_leading_columns_are_fixed_and_every_row_has_every_column(self, flat):
        assert list(flat[0])[:5] == ["id", "kind", "status", "reason", "content_hash"]
        assert len({tuple(row) for row in flat}) == 1

    def test_nested_keys_are_joined_with_a_dot(self, flat):
        row = flat[0]
        assert row["hbonds.hbonds"] == 3
        assert row["native.ranking_score"] == 0.55
        assert row["frame.receptor_rmsd_angstrom"] < 1e-3
        assert row["provenance.backend"] == "backend_a"

    def test_short_lists_of_scalars_become_one_column_per_item(self, flat):
        row = flat[0]
        assert row["frame.transform.translation[2]"] is not None
        assert row["frame.transform.rotation[2][2]"] == pytest.approx(1.0)
        assert "frame.transform.rotation[3][0]" not in row

    def test_a_row_has_none_where_a_pose_has_no_value(self, flat):
        assert flat[-1]["hbonds.hbonds"] is None and flat[0]["hbonds.hbonds"] == 3

    def test_no_column_holds_a_list_or_a_dictionary(self, flat):
        assert all(not isinstance(v, (list, dict)) for row in flat for v in row.values())

    def test_long_lists_and_lists_of_dictionaries_stay_in_the_json(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["stub_lists"])
        row = result.to_flat_rows()[0]
        assert (row["stub_lists.head[0]"], row["stub_lists.head[1]"]) == (1.0, 2.0)
        assert row["stub_lists.grid[1][0]"] == 3
        assert (row["stub_lists.names[0]"], row["stub_lists.names[1]"]) == ("x", "y")
        assert not any(name.startswith(("stub_lists.long", "stub_lists.table")) for name in row)
        wide = result.to_flat_rows(max_list_items=64)[0]
        assert wide["stub_lists.long[39]"] == 39
        assert not any(name.startswith("stub_lists.table") for name in wide)
        none = result.to_flat_rows(max_list_items=0)[0]
        assert not any(name.startswith("stub_lists.head") for name in none)

    def test_the_residue_table_is_never_flattened(self, synthetic):
        result = score_pose_set(synthetic.poses[:2], "B", "A")
        result["poses"][0]["residue_table"] = [{"role": "binder"}]
        assert not any(name.startswith("residue_table") for name in result.to_flat_rows()[0])

    def test_a_bad_limit_raises(self, result):
        with pytest.raises(ValueError, match="max_list_items"):
            result.to_flat_rows(max_list_items=-1)

    def test_a_key_that_contains_a_dot_is_kept_in_the_column_and_in_the_aggregate_path(
        self, synthetic, stubs
    ):
        """The clash cutoffs of the partner descriptors are the keys ``"2.2"`` and ``"3.0"``.

        One rule serves the flat rows, the CSV columns and the aggregate paths: keys are joined
        with ``.`` and a key is never changed or escaped, so ``clash_atoms.2.2`` is the key
        ``2.2`` under ``clash_atoms``. The names are unique, and the flat columns are the
        aggregate paths of the numeric values.
        """
        result = score_pose_set(synthetic.poses, "B", "A", metrics=["stub_dotted"])
        row = result.to_flat_rows()[0]
        assert row["stub_dotted.clash_atoms.2.2"] == 3
        assert row["stub_dotted.clash_atoms.3.0"] == 5
        entry = result["aggregates"]["backend_a"]["stub_dotted.clash_atoms.2.2"]
        assert (entry["n_ok"], entry["mean"]) == (3, 3.0)
        columns = {name for name in row if name.startswith("stub_dotted.")}
        paths = {name for name in result["aggregates"]["all"] if name.startswith("stub_dotted.")}
        assert (
            columns
            == paths
            == {
                "stub_dotted.min_distance",
                "stub_dotted.clash_atoms.2.2",
                "stub_dotted.clash_atoms.3.0",
            }
        )
        json.dumps(result)


# ---------------------------------------------------------------------------
# failures: rows with a reason, never a silent NaN
# ---------------------------------------------------------------------------


@pytest.fixture
def bad_file(tmp_path):
    path = tmp_path / "corrupt.pdb"
    path.write_text("this is not a structure\n", encoding="utf-8")
    return path


class TestFailedRows:
    def test_a_corrupt_file_is_a_failed_row_that_counts_in_n_total(self, synthetic, bad_file):
        poses = list(synthetic.poses[:3]) + [DockedPose.from_file(bad_file, id="zz_corrupt")]
        result = score_pose_set(poses, "B", "A", metrics=["delta_sasa_static"])
        row = result.row("zz_corrupt")
        assert row["status"] == "failed"
        assert row["reason"].startswith("ValueError")
        assert row["reasons"] == {"load": row["reason"]}
        assert row["frame"] is None and row["profile"] is None
        assert "delta_sasa_static" not in row
        entry = result["aggregates"]["all"]["delta_sasa_static.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (3, 4, False)
        assert entry["reasons"] == {row["reason"]: 1}
        # the survivors are still compared with each other
        assert result["agreement"]["pose_ids"] == list(IDS[:3])
        assert result["agreement"]["excluded"] == {"zz_corrupt": row["reason"]}

    def test_a_missing_chain_fails_the_row_and_names_the_chains(self, synthetic):
        result = score_pose_set(synthetic.poses[:3], "Z", "A")
        for row in result["poses"]:
            assert row["status"] == "failed"
            assert "chains ['Z'] not in the structure" in row["reason"]
        assert result["agreement"]["status"] == "skipped"
        assert result["representatives"]["all"] is None

    def test_a_pose_that_cannot_be_placed_is_a_failed_row_with_the_frame_reason(self, synthetic):
        tryp = DockedPose.from_file(
            Path(__file__).resolve().parents[2] / "data" / "example_bicyclic_sfti1_3P8F.cif",
            chain_map={"I": "B"},
            id="zz_trypsin",
        )
        poses = list(synthetic.poses[:3]) + [tryp]
        result = score_pose_set(poses, "B", "A", metrics=["hbonds"])
        row = result.row("zz_trypsin")
        assert row["status"] == "failed"
        assert row["reason"] == row["reasons"]["frame"]
        assert "reference receptor residues are paired" in row["reason"]
        assert row["frame"]["status"] == "failed" and row["frame"]["n_paired"] == 2
        assert "hbonds" not in row
        entry = result["aggregates"]["all"]["frame.n_paired"]
        assert (entry["n_ok"], entry["n_total"]) == (3, 4)
        assert result["agreement"]["excluded"]["zz_trypsin"] == row["reason"]

    def test_every_pose_unreadable_gives_failed_rows_and_a_result(self, bad_file):
        poses = [DockedPose.from_file(bad_file, id="p1"), DockedPose.from_file(bad_file, id="p2")]
        result = score_pose_set(poses, "B", "A", metrics=["hbonds"])
        assert [row["status"] for row in result["poses"]] == ["failed", "failed"]
        assert result["agreement"]["status"] == "skipped"
        assert result["references"]["variants"] == {}
        assert result["aggregates"]["all"] == {}
        json.dumps(result)

    def test_a_single_chain_structure_has_no_receptor_and_fails_with_a_reason(self):
        path = Path(__file__).resolve().parents[2] / "data" / "example_lactam_somatostatin_1XY4.cif"
        result = score_pose_set([DockedPose.from_file(path)], metrics=["hbonds"])
        row = result["poses"][0]
        assert row["status"] == "failed"
        assert "could not tell the binder and the target chain apart" in row["reason"]

    def test_a_metric_that_raises_gives_a_partial_row_and_its_reason(self, synthetic, stubs):
        result = score_pose_set(
            synthetic.poses[:4], "B", "A", metrics=["stub_raises", "stub_value"]
        )
        row = result["poses"][1]
        assert row["status"] == "partial"
        assert row["reasons"] == {"stub_raises": "ValueError: stub failure"}
        assert "stub_raises" not in row and "stub_value" in row
        assert result["poses"][0]["status"] == "ok"
        entry = result["aggregates"]["backend_a"]["stub_raises.value"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 3)
        assert entry["reasons"] == {"ValueError: stub failure": 1}

    def test_a_metric_that_reports_a_reason_is_partial_and_leaves_the_aggregate(
        self, synthetic, stubs
    ):
        result = score_pose_set(synthetic.poses[:3], "B", "A", metrics=["stub_reason"])
        row = result["poses"][0]
        assert row["status"] == "partial"
        assert row["reasons"] == {"stub_reason": "half of it could not be computed"}
        assert row["stub_reason"]["value"] == 4.0
        entry = result["aggregates"]["backend_a"]["stub_reason.value"]
        assert (entry["n_ok"], entry["n_total"]) == (0, 3)
        kept = score_pose_set(
            synthetic.poses[:3],
            "B",
            "A",
            metrics=["stub_reason"],
            options={"aggregation": {"reason_policy": "keep"}},
        )
        entry = kept["aggregates"]["backend_a"]["stub_reason.value"]
        assert (entry["n_ok"], entry["mean"]) == (3, 4.0)

    def test_a_nan_of_a_metric_is_kept_and_listed(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["stub_nan"])
        assert result["nonfinite_fields"] == [
            "poses[0].stub_nan.value",
            "poses[1].stub_nan.value",
        ]
        assert "NaN" in result.to_json()
        entry = result["aggregates"]["backend_a"]["stub_nan.value"]
        assert (entry["n_ok"], entry["n_total"]) == (0, 2)

    def test_a_clean_result_lists_no_nonfinite_field(self, result):
        assert result["nonfinite_fields"] == []

    def test_a_pose_whose_binder_cannot_be_read_stays_a_partial_row(self, synthetic):
        # a binder that is only water has no heavy atom: the agreement cannot read it
        atoms = synthetic.poses[0].atoms().copy()
        in_binder = atoms.chain_id == "B"
        atoms.res_name[in_binder] = "HOH"
        broken = DockedPose.from_atoms(atoms, id="zz_water", backend="backend_a")
        result = score_pose_set(list(synthetic.poses[:3]) + [broken], "B", "A")
        row = result.row("zz_water")
        assert row["status"] == "partial"
        assert "agreement" in row["reasons"]
        assert "zz_water" not in result["agreement"]["pose_ids"]
        assert "zz_water" in result["agreement"]["excluded"]
        # the others still have a medoid
        assert result["agreement"]["medoids"]["per_group"]["backend_a"]["pose_id"]


class TestOnErrorRaise:
    def test_a_metric_that_raises_stops_at_the_first_pose_and_names_it(self, synthetic, stubs):
        with pytest.raises(PoseScoringError, match="'backend_a-s2'.*stub_raises") as info:
            score_pose_set(synthetic.poses, "B", "A", metrics=["stub_raises"], on_error="raise")
        assert (info.value.pose_id, info.value.stage) == ("backend_a-s2", "stub_raises")
        assert isinstance(info.value.__cause__, ValueError)
        assert "stub failure" in str(info.value)
        # the poses after the failing one were not processed
        assert STUB_CALLS == ["pose_0000.cif", "pose_0001.cif"]

    def test_record_runs_every_pose(self, synthetic, stubs):
        score_pose_set(synthetic.poses, "B", "A", metrics=["stub_raises"], on_error="record")
        assert len(STUB_CALLS) == 6

    def test_a_corrupt_file_raises_chained(self, synthetic, bad_file):
        poses = list(synthetic.poses[:2]) + [DockedPose.from_file(bad_file, id="zz_corrupt")]
        with pytest.raises(PoseScoringError, match="'zz_corrupt'") as info:
            score_pose_set(poses, "B", "A", on_error="raise")
        assert info.value.stage == "load"
        assert isinstance(info.value.__cause__, ValueError)

    def test_a_pose_that_cannot_be_placed_raises_without_a_cause(self, synthetic):
        tryp = DockedPose.from_file(
            Path(__file__).resolve().parents[2] / "data" / "example_bicyclic_sfti1_3P8F.cif",
            chain_map={"I": "B"},
            id="zz_trypsin",
        )
        with pytest.raises(PoseScoringError, match="paired") as info:
            score_pose_set(list(synthetic.poses[:2]) + [tryp], "B", "A", on_error="raise")
        assert (info.value.pose_id, info.value.stage) == ("zz_trypsin", "frame")
        assert info.value.__cause__ is None

    def test_a_failure_before_the_first_good_pose_raises_too(self, synthetic, bad_file):
        poses = list(synthetic.poses[:2]) + [DockedPose.from_file(bad_file, id="0_corrupt")]
        with pytest.raises(PoseScoringError, match="'0_corrupt'"):
            score_pose_set(poses, "B", "A", on_error="raise")

    @pytest.mark.slow
    def test_a_pool_raises_the_first_failure_in_id_order(self, synthetic, stubs):
        with pytest.raises(PoseScoringError, match="'backend_a-s2'"):
            score_pose_set(
                synthetic.poses,
                "B",
                "A",
                metrics=["stub_raises"],
                on_error="raise",
                n_workers=2,
            )

    def test_a_metric_that_returns_a_reason_does_not_raise(self, synthetic, stubs):
        result = score_pose_set(
            synthetic.poses[:2], "B", "A", metrics=["stub_reason"], on_error="raise"
        )
        assert result["poses"][0]["status"] == "partial"


# ---------------------------------------------------------------------------
# which metrics run on which poses
# ---------------------------------------------------------------------------


class TestHeavyMetrics:
    REPRESENTATIVE_FILES = {"pose_0001.cif", "pose_0002.cif", "pose_0004.cif"}

    def test_a_heavy_metric_runs_on_the_representatives_only_by_default(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses, "B", "A", metrics=["stub_heavy", "stub_value"])
        assert set(STUB_CALLS) == self.REPRESENTATIVE_FILES | {
            f"pose_{i:04d}.cif" for i in range(6)
        }
        with_block = [row["id"] for row in result["poses"] if "stub_heavy" in row]
        representatives = result["representatives"]
        assert with_block == sorted({representatives[k] for k in ("backend_a", "backend_b", "all")})
        for row in result["poses"]:
            assert "stub_heavy" in row or "stub_heavy" not in row["reasons"]
            assert row["status"] == "ok"
        parameters = result["parameters"]
        assert parameters["metric_scope"] == {
            "stub_heavy": "representatives",
            "stub_value": "all",
        }

    def test_the_aggregate_of_a_heavy_metric_counts_the_poses_it_ran_on(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses, "B", "A", metrics=["stub_heavy"])
        assert result["aggregates"]["all"]["stub_heavy.value"]["n_total"] == 3
        assert result["aggregates"]["backend_a"]["stub_heavy.value"]["n_total"] == 2
        assert result["aggregates"]["backend_b"]["stub_heavy.value"]["n_total"] == 1

    def test_heavy_all_runs_it_on_every_pose(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses, "B", "A", metrics=["stub_md"], heavy="all")
        assert all("stub_md" in row for row in result["poses"])
        assert result["aggregates"]["all"]["stub_md.value"]["n_ok"] == 6

    def test_heavy_skip_leaves_it_out(self, synthetic, stubs):
        result = score_pose_set(
            synthetic.poses, "B", "A", metrics=["stub_heavy", "stub_value"], heavy="skip"
        )
        assert all("stub_heavy" not in row for row in result["poses"])
        assert "stub_heavy.value" not in result["aggregates"]["all"]
        assert result["parameters"]["metric_scope"]["stub_heavy"] == "none"

    def test_a_metric_without_a_declared_cost_class_is_heavy(self, synthetic, monkeypatch, stubs):
        monkeypatch.setitem(
            METRICS_BY_NAME, "stub_undeclared", _spec("stub_undeclared", "_stub_value", None)
        )
        result = score_pose_set(synthetic.poses, "B", "A", metrics=["stub_undeclared"])
        assert result["parameters"]["metric_scope"] == {"stub_undeclared": "representatives"}
        assert sum("stub_undeclared" in row for row in result["poses"]) == 3

    def test_a_failed_pose_is_never_a_representative(self, synthetic, stubs, bad_file):
        poses = [DockedPose.from_file(bad_file, id="0_corrupt", backend="backend_a")]
        poses += list(synthetic.poses)
        result = score_pose_set(poses, "B", "A", metrics=["stub_heavy"])
        assert result.row("0_corrupt")["status"] == "failed"
        assert "0_corrupt" not in {
            v for k, v in result["representatives"].items() if k not in ("rule", "reasons")
        }

    def test_the_chain_arguments_reach_the_function(self, synthetic, stubs):
        result = score_pose_set(synthetic.poses[:1], "B", "A", metrics=["stub_value"])
        block = result["poses"][0]["stub_value"]
        assert (block["binder_chain"], block["receptor_chain"]) == ("B", "A")

    def test_metric_kwargs_are_passed_verbatim(self, synthetic):
        result = score_pose_set(
            synthetic.poses[:1],
            "B",
            "A",
            metrics=["delta_sasa_static"],
            metric_kwargs={"delta_sasa_static": {"probe_radius": 3.0}},
        )
        plain = score_pose_set(synthetic.poses[:1], "B", "A", metrics=["delta_sasa_static"])
        a = result["poses"][0]["delta_sasa_static"]["delta_sasa"]
        b = plain["poses"][0]["delta_sasa_static"]["delta_sasa"]
        assert a != b
        assert result["parameters"]["metric_kwargs"] == {"delta_sasa_static": {"probe_radius": 3.0}}


# ---------------------------------------------------------------------------
# the registry names and the reserved row names
# ---------------------------------------------------------------------------


class TestReservedNames:
    def test_no_registered_metric_has_a_reserved_row_name(self):
        """A metric added to the registry later that takes one of these names fails here."""
        clashes = sorted(set(METRICS_BY_NAME) & set(RESERVED_ROW_NAMES))
        assert clashes == []

    def test_the_reserved_names_are_the_fields_of_a_row(self, result):
        row_fields = [name for name in result["poses"][0] if name not in METRICS]
        assert set(row_fields) <= set(RESERVED_ROW_NAMES)
        assert len(set(RESERVED_ROW_NAMES)) == len(RESERVED_ROW_NAMES)

    @pytest.mark.parametrize("name", RESERVED_ROW_NAMES)
    def test_a_metric_named_like_a_row_field_fails_loudly(self, synthetic, monkeypatch, name):
        monkeypatch.setitem(METRICS_BY_NAME, name, _spec(name, "_stub_value"))
        with pytest.raises(ValueError, match=f"{name!r} is a reserved row name"):
            score_pose_set(synthetic.poses[:2], "B", "A", metrics=[name])

    def test_a_reserved_name_is_refused_even_when_the_registry_does_not_know_it(self, synthetic):
        with pytest.raises(ValueError, match="reserved row name"):
            score_pose_set(synthetic.poses[:2], "B", "A", metrics=["frame"])


# ---------------------------------------------------------------------------
# arguments are checked before any pose is read
# ---------------------------------------------------------------------------


class Unreadable(DockedPose):
    """A pose that fails the test if anyone reads it."""

    def atoms(self):  # pragma: no cover - reached only when an argument check is missing
        raise AssertionError("a structure was read before the arguments were checked")


class TestArgumentChecks:
    @pytest.fixture
    def poses(self, bad_file):
        return [Unreadable(bad_file, given_id="p1"), Unreadable(bad_file, given_id="p2")]

    @pytest.mark.parametrize(
        "kwargs, error, message",
        [
            ({"metrics": ["nope"]}, KeyError, "Unknown metric 'nope'"),
            ({"metrics": "interface"}, ValueError, "not the string"),
            ({"metrics": [3]}, TypeError, "must be strings"),
            ({"metrics": ["hbonds", "hbonds"]}, ValueError, "repeats"),
            ({"metrics": ["rmsd"]}, ValueError, "reads a trajectory"),
            ({"metrics": ["openfold"]}, ValueError, "OpenFold3 run"),
            ({"metrics": ["interface_pae"]}, ValueError, "OpenFold3 run"),
            ({"metrics": ["dockq"]}, ValueError, "give reference="),
            ({"metrics": ["structure_rmsd"]}, ValueError, "give reference="),
            ({"on_error": "ignore"}, ValueError, "on_error must be one of"),
            ({"heavy": "some"}, ValueError, "heavy must be one of"),
            ({"preflight": "never"}, ValueError, "preflight must be one of"),
            ({"n_workers": 0}, ValueError, "n_workers"),
            ({"n_workers": 1.5}, ValueError, "n_workers"),
            ({"aggregate": ["mode"]}, ValueError, "mode"),
            ({"group_by": [""]}, ValueError, "group_by"),
            ({"representative": "best"}, ValueError, "representative must be"),
            (
                {
                    "representative": "confidence",
                    "options": {"representative": {"confidence_key": "mine"}},
                },
                ValueError,
                "direction of native score 'mine'",
            ),
            ({"metric_kwargs": {"hbonds": {}}}, ValueError, "not one of the requested"),
            (
                {"metrics": ["hbonds"], "metric_kwargs": {"hbonds": {"atoms": 1}}},
                ValueError,
                "passes itself",
            ),
            ({"metric_kwargs": 3}, TypeError, "metric_kwargs must be"),
            ({"binder_chain": "A", "target_chain": "A"}, ValueError, "both 'A'"),
            ({"binder_chain": ""}, ValueError, "binder_chain"),
            ({"tmp_dir": "/no/such/directory"}, ValueError, "not an existing directory"),
            ({"keep_tmp": "yes"}, TypeError, "keep_tmp"),
            ({"construct": 3}, TypeError, "construct"),
            ({"reference": 3}, TypeError, "reference must be"),
            ({"reference": {}}, ValueError, "empty mapping"),
            ({"residue_maps": {"nobody": {}}}, ValueError, "not pose ids"),
            ({"residue_maps": 3}, TypeError, "residue_maps"),
        ],
    )
    def test_the_argument_is_refused_without_reading_a_structure(
        self, poses, kwargs, error, message
    ):
        with pytest.raises(error, match=message):
            score_pose_set(poses, **{"binder_chain": "B", "target_chain": "A", **kwargs})

    @pytest.mark.parametrize(
        "poses, error, message",
        [
            ([], ValueError, "poses is empty"),
            ("a.pdb", TypeError, "sequence of Pose"),
            (3, TypeError, "sequence of Pose"),
            (["a.pdb"], TypeError, "not a Pose"),
        ],
    )
    def test_the_poses_are_checked(self, poses, error, message):
        with pytest.raises(error, match=message):
            score_pose_set(poses, "B", "A")

    def test_a_repeated_id_lists_the_ids(self, bad_file):
        poses = [Unreadable(bad_file, given_id="same"), Unreadable(bad_file, given_id="same")]
        with pytest.raises(ValueError, match=r"repeated: \['same'\]"):
            score_pose_set(poses, "B", "A")

    def test_a_reference_that_is_not_valid_raises_with_every_problem(self, synthetic):
        from binding_metrics.poses.reference import Reference

        data = Path(__file__).resolve().parents[2] / "data"
        reference = Reference(data / "example_linear_p53_1YCR.pdb", "Z", "Y")
        with pytest.raises(ValueError, match="receptor chain 'Z'.*binder chain 'Y'|chain 'Y'"):
            score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)

    def test_an_unknown_frame_residue_set_raises_before_any_pose_is_read(self, synthetic):
        with pytest.raises(ValueError, match="frame_residues='nope'"):
            score_pose_set(
                synthetic.poses[:2],
                "B",
                "A",
                options={"frame": {"frame_residues": "nope"}},
                reference=_ycr_reference(),
            )


def _ycr_reference(**kwargs):
    from binding_metrics.poses.reference import Reference

    data = Path(__file__).resolve().parents[2] / "data"
    return Reference(data / "example_linear_p53_1YCR.pdb", "A", "B", **kwargs)


# ---------------------------------------------------------------------------
# temporary files and the working directory
# ---------------------------------------------------------------------------


class TestTemporaryFiles:
    def test_nothing_is_written_to_the_working_directory(self, synthetic, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        score_pose_set(synthetic.poses[:3], "B", "A", metrics=["delta_sasa_static", "hbonds"])
        assert list(work.iterdir()) == []

    def test_the_temporary_directory_is_made_in_tmp_dir_and_removed(self, synthetic, tmp_path):
        score_pose_set(
            synthetic.poses[:3], "B", "A", metrics=["delta_sasa_static"], tmp_dir=tmp_path
        )
        assert list(tmp_path.iterdir()) == []

    def test_the_directory_is_removed_when_a_pose_raises(self, synthetic, tmp_path, stubs):
        with pytest.raises(PoseScoringError):
            score_pose_set(
                synthetic.poses,
                "B",
                "A",
                metrics=["stub_raises"],
                on_error="raise",
                tmp_dir=tmp_path,
            )
        assert list(tmp_path.iterdir()) == []

    def test_keep_tmp_leaves_one_file_per_pose_in_the_reference_frame(self, synthetic, tmp_path):
        from binding_metrics.metrics._common import load_structure

        result = score_pose_set(
            synthetic.poses[:3],
            "B",
            "A",
            metrics=["delta_sasa_static"],
            tmp_dir=tmp_path,
            keep_tmp=True,
        )
        kept = Path(result["provenance"]["temp_dir"])
        assert kept.parent == tmp_path and kept.name.startswith("bm_poseset_")
        assert sorted(p.name for p in kept.iterdir()) == [
            "pose_0000.cif",
            "pose_0001.cif",
            "pose_0002.cif",
        ]
        # every pose was moved onto the receptor of the first: the receptor atoms coincide
        first = load_structure(kept / "pose_0000.cif")
        other = load_structure(kept / "pose_0002.cif")
        receptor = first.chain_id == "A"
        assert np.abs(first.coord[receptor] - other.coord[receptor]).max() < 1e-3
        assert len(first.atom_name) == synthetic.poses[0].atoms().array_length()

    def test_a_metric_that_takes_an_array_needs_no_file(self, synthetic, tmp_path):
        result = score_pose_set(
            synthetic.poses[:2], "B", "A", metrics=["hbonds"], tmp_dir=tmp_path, keep_tmp=True
        )
        assert "temp_dir" not in result["provenance"]
        assert list(tmp_path.iterdir()) == []

    def test_the_original_frame_reads_the_pose_file_as_it_is(self, synthetic, tmp_path, stubs):
        files = synthetic.write(tmp_path / "poses")
        poses = [DockedPose.from_file(path, id=path.stem, backend="x") for path in files]
        score_pose_set(
            poses[:3],
            "B",
            "A",
            metrics=["stub_value"],
            options={"frame": {"metrics_frame": "original"}},
        )
        assert set(STUB_CALLS) == {path.name for path in files[:3]}

    def test_the_original_frame_writes_a_pose_that_has_no_file(self, synthetic, stubs):
        score_pose_set(
            synthetic.poses[:2],
            "B",
            "A",
            metrics=["stub_value"],
            options={"frame": {"metrics_frame": "original"}},
        )
        assert STUB_CALLS == ["pose_0000.cif", "pose_0001.cif"]

    def test_the_original_frame_gives_the_values_of_a_plain_call(self, synthetic, tmp_path):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        path = synthetic.write(tmp_path)[0]
        pose = DockedPose.from_file(path, id="only", backend="x")
        result = score_pose_set(
            [pose],
            "B",
            "A",
            metrics=["delta_sasa_static"],
            options={"frame": {"metrics_frame": "original"}},
        )
        plain = compute_delta_sasa_static(path, peptide_chain="B", receptor_chain="A")
        assert result["poses"][0]["delta_sasa_static"] == plain


# ---------------------------------------------------------------------------
# the pre-flight check
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_a_metric_the_input_cannot_take_is_refused_before_any_pose_is_scored(self, synthetic):
        # the 'prediction' block needs a predicted pose; these are docked poses
        with pytest.raises(IncompatibleInputError, match="prediction"):
            score_pose_set(synthetic.poses[:2], "B", "A", metrics=["prediction"])

    def test_skip_leaves_the_metric_out_and_lists_it(self, synthetic):
        result = score_pose_set(
            synthetic.poses[:3], "B", "A", metrics=["prediction", "hbonds"], preflight="skip"
        )
        report = result["preflight"]
        assert report["policy"] == "skip"
        assert report["metrics_to_run"] == ["hbonds"]
        assert [m["name"] for m in report["metrics_skipped"]] == ["prediction"]
        row = result["poses"][0]
        assert "prediction" not in row and "hbonds" in row
        assert row["reasons"]["prediction"].startswith("left out by the pre-flight check")
        assert row["status"] == "partial"
        entry = result["aggregates"]["backend_a"]["hbonds.hbonds"]
        assert entry["n_ok"] == 3

    def test_off_does_not_check_and_the_pose_says_why_it_has_no_block(self, synthetic):
        result = score_pose_set(
            synthetic.poses[:2], "B", "A", metrics=["prediction"], preflight="off"
        )
        assert result["preflight"] is None
        row = result["poses"][0]
        assert "predicted pose" in row["reasons"]["prediction"]
        assert row["status"] == "partial" and "prediction" not in row
        assert row["reason"] is None  # a metric that does not apply is not a failure of the pose

    def test_a_compatible_input_has_a_report_in_the_result(self, result):
        report = result["preflight"]
        assert report["compatible"] is True and report["policy"] == "error"
        assert report["metrics_to_run"] == list(METRICS)
        assert report["profile"]["binder_chains"] == ["B"]

    def test_the_input_is_refused_when_the_metric_needs_a_reference(self, synthetic):
        with pytest.raises(ValueError, match="give reference="):
            score_pose_set(synthetic.poses[:2], "B", "A", metrics=["dockq"])


# ---------------------------------------------------------------------------
# what importing and running need
# ---------------------------------------------------------------------------


class TestImports:
    @staticmethod
    def _run(code: str) -> str:
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return completed.stdout.strip()

    def test_importing_the_module_loads_no_structure_library(self):
        out = self._run(
            """
            import sys
            import binding_metrics.poses.score
            libraries = ("biotite", "scipy", "openmm", "gemmi", "mdtraj", "hydride")
            print([m for m in libraries if m in sys.modules])
            """
        )
        assert out == "[]"

    @pytest.mark.slow
    def test_static_metrics_run_without_openmm_mdtraj_simtk_and_pdbfixer(self):
        out = self._run(
            """
            import sys
            for name in ("openmm", "simtk", "mdtraj", "pdbfixer"):
                sys.modules[name] = None  # an import of them now raises ImportError
            from binding_metrics.poses import score_pose_set
            from binding_metrics.testing import make_synthetic_poses
            synthetic = make_synthetic_poses(seeds=(1, 2))
            result = score_pose_set(
                synthetic.poses, "B", "A", metrics=["delta_sasa_static", "hbonds", "omega"]
            )
            statuses = sorted({row["status"] for row in result["poses"]})
            blocked = [m for m in ("openmm", "simtk", "mdtraj", "pdbfixer") if sys.modules[m]]
            print(statuses, blocked, result["representatives"]["rule"])
            """
        )
        assert out == "['ok'] [] medoid"


def test_the_result_survives_a_deep_copy(result):
    assert copy.deepcopy(result).to_json(include_timing=False) == result.to_json(
        include_timing=False
    )
