"""Per-group aggregates of pose rows and the representative pose (poses/aggregate.py)."""

import json
import math
import random

import numpy as np
import pytest

from binding_metrics.poses.aggregate import (
    DEFAULT_STATISTICS,
    ENTRY_KEYS,
    STATISTICS,
    PoseDistances,
    aggregate_poses,
    aggregate_values,
    group_key,
    native_score_direction,
    rank_by,
    resolve_statistics,
    select_representatives,
)

SASA = [1266.8, 1280.1, 1324.7, 1209.5, 1264.0, 1378.3]


def make_row(pose_id, backend="a", seed=1, **blocks):
    return {
        "id": pose_id,
        "kind": "predicted",
        "status": "ok",
        "reason": None,
        "reasons": {},
        "provenance": {"backend": backend, "seed": seed},
        "native": blocks.pop("native", {}),
        **blocks,
    }


def sasa_rows(values, backend="a"):
    return [
        make_row(f"{backend}{i}", backend, i, interface={"delta_sasa": value, "n_hbonds": i})
        for i, value in enumerate(values, start=1)
    ]


class TestStatistics:
    def test_every_statistic_matches_numpy(self):
        entry = aggregate_poses(sasa_rows(SASA))["a"]["interface.delta_sasa"]
        values = np.array(SASA)
        assert entry["mean"] == pytest.approx(np.mean(values), rel=1e-12)
        assert entry["std"] == pytest.approx(np.std(values, ddof=1), rel=1e-12)
        assert entry["min"] == min(SASA)
        assert entry["max"] == max(SASA)
        assert entry["median"] == pytest.approx(np.median(values), rel=1e-12)
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (6, 6, True)
        assert entry["reasons"] == {} and entry["statistic_reasons"] == {}

    def test_std_is_the_sample_deviation_not_the_population_one(self):
        entry = aggregate_values([1.0, 2.0, 3.0])
        assert entry["std"] == pytest.approx(1.0)
        assert entry["std"] != pytest.approx(np.std([1.0, 2.0, 3.0]))

    def test_median_of_an_even_count_is_the_mean_of_the_two_middle_values(self):
        assert aggregate_values([4.0, 1.0, 3.0, 2.0])["median"] == 2.5

    def test_the_default_statistics_and_the_order_of_the_entry(self):
        entry = aggregate_values([1.0, 2.0])
        assert DEFAULT_STATISTICS == ("mean", "std", "min", "max", "median")
        assert list(entry) == [*DEFAULT_STATISTICS, *ENTRY_KEYS]

    def test_a_subset_of_statistics_keeps_the_order_given(self):
        entry = aggregate_values([1.0, 2.0], statistics=("max", "mean"))
        assert list(entry)[:2] == ["max", "mean"] and "std" not in entry

    def test_the_documented_numbers_of_the_design_example(self):
        rows = sasa_rows([1266.8, 1279.9, 1324.7])
        entry = aggregate_poses(rows)["a"]["interface.delta_sasa"]
        assert entry["mean"] == pytest.approx(1290.5, abs=0.05)
        assert entry["std"] == pytest.approx(30.3, abs=0.1)
        assert entry["median"] == 1279.9

    def test_integers_are_aggregated_as_floats(self):
        entry = aggregate_poses(sasa_rows(SASA))["a"]["interface.n_hbonds"]
        assert entry["mean"] == 3.5 and isinstance(entry["max"], float)

    def test_numpy_scalars_are_numbers(self):
        rows = [
            make_row("a1", interface={"x": np.float32(1.5)}),
            make_row("a2", interface={"x": np.int64(2)}),
        ]
        entry = aggregate_poses(rows)["a"]["interface.x"]
        assert entry["n_ok"] == 2 and entry["mean"] == 1.75


class TestCounting:
    def test_a_failed_pose_counts_in_n_total_and_never_in_n_ok(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        rows[2]["status"] = "failed"
        rows[2]["reason"] = "ValueError: chain C is absent"
        entry = aggregate_poses(rows)["a"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (2, 3, False)
        assert entry["mean"] == 105.0
        assert entry["reasons"] == {"ValueError: chain C is absent": 1}

    def test_a_failed_pose_without_a_reason_says_so(self):
        rows = sasa_rows([100.0, 110.0])
        rows[1]["status"] = "failed"
        rows[1]["reason"] = None
        assert aggregate_poses(rows)["a"]["interface.delta_sasa"]["reasons"] == {"pose failed": 1}

    def test_nan_none_inf_bool_and_string_are_not_valid_values(self):
        values = [1.0, float("nan"), None, float("inf"), True, "n/a", [1.0], 3.0]
        entry = aggregate_values(values)
        assert (entry["n_ok"], entry["n_total"]) == (2, 8)
        assert entry["mean"] == 2.0
        assert entry["reasons"] == {
            "value is NaN": 1,
            "value is None": 1,
            "value is infinite": 1,
            "value is a bool, not a number": 1,
            "value is not a number (str)": 1,
            "value is not a number (list)": 1,
        }

    def test_rows_with_nan_and_none_count_in_n_total(self):
        rows = sasa_rows([100.0, float("nan"), 120.0, 130.0])
        rows[3]["interface"]["delta_sasa"] = None
        entry = aggregate_poses(rows)["a"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 4)
        assert entry["reasons"] == {"value is NaN": 1, "value is None": 1}
        assert entry["mean"] == 110.0

    def test_a_metric_that_raised_is_counted_with_its_reason_from_the_row(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        del rows[1]["interface"]
        rows[1]["status"] = "partial"
        rows[1]["reasons"] = {"interface": "RuntimeError: boom"}
        entry = aggregate_poses(rows)["a"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 3)
        assert entry["reasons"] == {"RuntimeError: boom": 1}

    def test_a_pose_that_was_not_asked_for_a_metric_is_not_counted(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        for row in rows[1:]:
            del row["interface"]  # a heavy metric run on the representative only
        entry = aggregate_poses(rows)["a"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (1, 1, True)

    def test_a_key_present_in_some_poses_only_counts_the_whole_block(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        rows[0]["interface"]["extra"] = 5.0
        entry = aggregate_poses(rows)["a"]["interface.extra"]
        assert (entry["n_ok"], entry["n_total"]) == (1, 3)
        assert entry["reasons"] == {"no value for this key": 2}

    def test_bools_and_strings_never_make_a_key(self):
        rows = [
            make_row("a1", interface={"flag": True, "label": "x", "per_residue": [1.0, 2.0]}),
            make_row("a2", interface={"flag": False, "label": "y", "per_residue": [3.0]}),
        ]
        assert aggregate_poses(rows)["a"] == {}

    def test_excluded_blocks_are_not_aggregated_and_can_be_changed(self):
        rows = sasa_rows([100.0, 110.0])
        keys = set(aggregate_poses(rows)["a"])
        assert not any(key.startswith(("provenance", "id", "reasons")) for key in keys)
        with_provenance = aggregate_poses(rows, exclude_blocks=("id", "kind"))["a"]
        assert with_provenance["provenance.seed"]["mean"] == 1.5

    def test_forced_keys_are_reported_even_with_no_number(self):
        rows = [make_row("a1", native={"gpde": None}), make_row("a2", native={"gpde": None})]
        assert aggregate_poses(rows)["a"] == {}
        entry = aggregate_poses(rows, keys=["native.gpde"])["a"]["native.gpde"]
        assert (entry["n_ok"], entry["n_total"], entry["mean"]) == (0, 2, None)
        assert entry["statistic_reasons"]["mean"] == "no valid value"

    def test_a_group_whose_poses_all_failed_reports_every_key_with_no_value(self):
        rows = sasa_rows([100.0, 110.0], "a") + sasa_rows([200.0, 210.0], "b")
        for row in rows[2:]:
            row["status"] = "failed"
            row["reason"] = "no structure"
        entry = aggregate_poses(rows)["b"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"], entry["mean"]) == (0, 2, None)
        assert entry["reasons"] == {"no structure": 2}


class TestReasonPolicy:
    def reason_rows(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        rows[1]["interface"]["reason"] = "hydrogen-bond step failed"
        return rows

    def test_a_dict_level_reason_excludes_every_key_of_the_metric_for_that_pose(self):
        aggregates = aggregate_poses(self.reason_rows())["a"]
        for key in ("interface.delta_sasa", "interface.n_hbonds"):
            assert (aggregates[key]["n_ok"], aggregates[key]["n_total"]) == (2, 3)
            assert aggregates[key]["reasons"] == {"hydrogen-bond step failed": 1}
        assert aggregates["interface.delta_sasa"]["mean"] == 110.0

    def test_the_row_keeps_its_values(self):
        rows = self.reason_rows()
        aggregate_poses(rows)
        assert rows[1]["interface"]["delta_sasa"] == 110.0

    def test_other_blocks_of_that_pose_still_count(self):
        rows = self.reason_rows()
        rows[1]["native"] = {"iptm": 0.8}
        rows[0]["native"] = {"iptm": 0.7}
        assert aggregate_poses(rows)["a"]["native.iptm"]["n_ok"] == 2

    def test_a_reason_in_a_nested_dict_covers_only_that_subtree(self):
        rows = [
            make_row("a1", descriptors={"v": {"s": {"x": 1.0}, "t": {"x": 5.0}}}),
            make_row(
                "a2",
                descriptors={"v": {"s": {"x": 3.0, "reason": "set is empty"}, "t": {"x": 7.0}}},
            ),
        ]
        aggregates = aggregate_poses(rows)["a"]
        assert (
            aggregates["descriptors.v.s.x"]["n_ok"],
            aggregates["descriptors.v.s.x"]["n_total"],
        ) == (1, 2)
        assert aggregates["descriptors.v.t.x"]["n_ok"] == 2

    def test_keep_policy_keeps_the_values_and_lists_the_reasons_as_caveats(self):
        entry = aggregate_poses(self.reason_rows(), reason_policy="keep")["a"][
            "interface.delta_sasa"
        ]
        assert (entry["n_ok"], entry["n_total"], entry["mean"]) == (3, 3, 110.0)
        assert entry["caveats"] == {"hydrogen-bond step failed": 1}
        assert entry["reasons"] == {}

    def test_an_empty_or_missing_reason_is_not_a_reason(self):
        rows = sasa_rows([100.0, 110.0])
        rows[0]["interface"]["reason"] = ""
        rows[1]["interface"]["reason"] = None
        assert aggregate_poses(rows)["a"]["interface.delta_sasa"]["n_ok"] == 2

    def test_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="reason_policy"):
            aggregate_poses(sasa_rows([1.0]), reason_policy="drop")


class TestMinNOk:
    def test_default_reports_a_single_survivor_with_its_count_and_no_std(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        for row in rows[1:]:
            row["status"] = "failed"
        entry = aggregate_poses(rows)["a"]["interface.delta_sasa"]
        assert entry["mean"] == 100.0 and entry["std"] is None
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (1, 3, False)
        assert entry["statistic_reasons"] == {"std": "needs at least 2 values"}

    def test_fewer_valid_values_than_min_n_ok_gives_none_for_every_statistic(self):
        rows = sasa_rows([100.0, 110.0, 120.0])
        rows[1]["status"] = "failed"
        entry = aggregate_poses(rows, min_n_ok=3)["a"]["interface.delta_sasa"]
        assert all(entry[name] is None for name in DEFAULT_STATISTICS)
        assert entry["n_ok"] == 2
        assert set(entry["statistic_reasons"]) == set(DEFAULT_STATISTICS)
        assert all("below min_n_ok 3" in text for text in entry["statistic_reasons"].values())

    def test_exactly_min_n_ok_values_are_enough(self):
        entry = aggregate_poses(sasa_rows([100.0, 110.0]), min_n_ok=2)["a"]["interface.delta_sasa"]
        assert entry["mean"] == 105.0 and entry["statistic_reasons"] == {}

    def test_no_valid_value_at_all(self):
        entry = aggregate_values([None, float("nan")])
        assert all(entry[name] is None for name in DEFAULT_STATISTICS)
        assert entry["statistic_reasons"]["mean"] == "no valid value"
        assert (entry["n_ok"], entry["n_total"]) == (0, 2)

    def test_std_needs_two_values_even_with_min_n_ok_one(self):
        entry = aggregate_values([5.0], min_n_ok=1)
        assert entry["mean"] == 5.0 and entry["std"] is None

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
    def test_min_n_ok_below_one_or_not_an_integer_is_refused(self, bad):
        with pytest.raises(ValueError, match="min_n_ok"):
            aggregate_values([1.0], min_n_ok=bad)


class TestGroups:
    def two_backends(self):
        return sasa_rows([100.0, 110.0, 120.0], "boltz2") + sasa_rows([200.0, 220.0], "of3")

    def test_one_block_per_backend_and_one_for_all_poses(self):
        aggregates = aggregate_poses(self.two_backends())
        assert list(aggregates) == ["boltz2", "of3", "all"]
        assert aggregates["boltz2"]["interface.delta_sasa"]["mean"] == 110.0
        assert aggregates["of3"]["interface.delta_sasa"]["mean"] == 210.0
        assert aggregates["all"]["interface.delta_sasa"]["n_total"] == 5
        assert aggregates["all"]["interface.delta_sasa"]["mean"] == pytest.approx(150.0)

    def test_group_by_backend_and_seed(self):
        rows = [
            make_row("a1", "boltz2", 42, interface={"x": 1.0}),
            make_row("a2", "boltz2", 42, interface={"x": 3.0}),
            make_row("a3", "boltz2", 43, interface={"x": 5.0}),
        ]
        aggregates = aggregate_poses(rows, group_by=("backend", "seed"))
        assert list(aggregates) == ["boltz2|seed=42", "boltz2|seed=43", "all"]
        assert aggregates["boltz2|seed=42"]["interface.x"]["mean"] == 2.0
        assert group_key(rows[0], ("backend", "seed")) == "boltz2|seed=42"

    def test_group_by_a_key_other_than_backend_and_by_nothing(self):
        rows = sasa_rows([100.0, 110.0], "a")
        assert list(aggregate_poses(rows, group_by="seed")) == ["seed=1", "seed=2", "all"]
        assert list(aggregate_poses(rows, group_by=())) == ["all"]

    def test_a_pose_without_the_group_key_forms_its_own_group(self):
        rows = sasa_rows([100.0, 110.0], "a")
        rows[1]["provenance"] = {}
        assert list(aggregate_poses(rows)) == ["a", "backend=None", "all"]

    def test_one_backend_has_one_group_plus_all(self):
        aggregates = aggregate_poses(sasa_rows([100.0, 110.0]))
        assert list(aggregates) == ["a", "all"]
        assert aggregates["a"] == aggregates["all"]

    def test_a_group_does_not_report_a_key_none_of_its_poses_was_asked_for(self):
        rows = sasa_rows([100.0, 110.0], "a") + sasa_rows([200.0], "b")
        del rows[2]["interface"]
        aggregates = aggregate_poses(rows)
        assert "interface.delta_sasa" in aggregates["a"]
        assert aggregates["b"] == {}

    @pytest.mark.parametrize("name", ["all", "rule", "reasons"])
    def test_a_reserved_group_key_is_refused(self, name):
        with pytest.raises(ValueError, match="reserved"):
            aggregate_poses(sasa_rows([100.0], name))

    def test_keys_are_sorted_and_the_result_does_not_depend_on_the_row_order(self):
        rows = self.two_backends()
        for row in rows:
            row["frame"] = {"receptor_rmsd_angstrom": 0.5, "n_paired": 80}
            row["native"] = {"iptm": 0.5}
        reference = json.dumps(aggregate_poses(rows))
        assert list(aggregate_poses(rows)["all"]) == sorted(aggregate_poses(rows)["all"])
        shuffler = random.Random(3)
        for _ in range(5):
            shuffled = list(rows)
            shuffler.shuffle(shuffled)
            assert json.dumps(aggregate_poses(shuffled)) == reference

    def test_rows_that_are_not_mappings_are_refused(self):
        with pytest.raises(ValueError, match="row 1"):
            aggregate_poses([make_row("a1"), "a2"])


class TestPluggableAggregators:
    def test_a_mapping_of_callables(self):
        statistics = {
            "mean": STATISTICS["mean"],
            "spread": lambda values: max(values) - min(values),
        }
        entry = aggregate_poses(sasa_rows(SASA), statistics=statistics)["a"]["interface.delta_sasa"]
        assert list(entry)[:2] == ["mean", "spread"]
        assert entry["spread"] == pytest.approx(max(SASA) - min(SASA))

    def test_the_callable_receives_a_sorted_list_of_floats(self):
        seen = []

        def record(values):
            seen.append(values)
            return 0.0

        aggregate_values([3, 1.5, np.float32(2.0), None], statistics={"record": record})
        assert seen == [[1.5, 2.0, 3.0]] and all(type(v) is float for v in seen[0])

    def test_a_callable_that_raises_gives_none_and_a_reason(self):
        def broken(values):
            raise RuntimeError("no luck")

        entry = aggregate_values(
            [1.0, 2.0], statistics={"broken": broken, "mean": STATISTICS["mean"]}
        )
        assert entry["broken"] is None and entry["mean"] == 1.5
        assert entry["statistic_reasons"] == {"broken": "RuntimeError: no luck"}

    def test_a_callable_that_returns_none_or_a_non_finite_number(self):
        entry = aggregate_values(
            [1.0, 2.0],
            statistics={
                "none": lambda v: None,
                "nan": lambda v: float("nan"),
                "text": lambda v: "x",
            },
        )
        assert entry["none"] is None and entry["nan"] is None and entry["text"] is None
        assert entry["statistic_reasons"]["none"] == "the aggregator returned None"
        assert "not a finite number" in entry["statistic_reasons"]["nan"]
        assert "not a finite number" in entry["statistic_reasons"]["text"]

    def test_min_values_attribute_keeps_a_callable_from_running_on_too_few_values(self):
        calls = []

        def third_moment(values):
            calls.append(values)
            return float(np.mean(np.array(values) ** 3))

        third_moment.min_values = 3
        entry = aggregate_values([1.0, 2.0], statistics={"m3": third_moment})
        assert entry["m3"] is None and calls == []
        assert entry["statistic_reasons"] == {"m3": "needs at least 3 values"}

    def test_a_name_that_is_neither_registered_nor_callable_is_refused_before_any_work(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            aggregate_poses([object()], statistics=("mean", "mode"))
        with pytest.raises(ValueError, match="not callable"):
            resolve_statistics({"mean": "mean"})

    def test_names_that_clash_with_the_entry_are_refused(self):
        with pytest.raises(ValueError, match="reserved"):
            resolve_statistics({"n_ok": lambda values: 1.0})
        with pytest.raises(ValueError, match="repeated"):
            resolve_statistics(("mean", "mean"))
        with pytest.raises(ValueError):
            resolve_statistics("mean")

    def test_the_registered_statistics_are_read_only(self):
        with pytest.raises(TypeError):
            STATISTICS["mode"] = len


def offsets_rows():
    """Six poses on a line: a1-a3 at 0.0, 0.4, 0.8 and b1-b3 at 2.0, 2.3, 2.9 (design 11.2)."""
    offsets = {"a1": 0.0, "a2": 0.4, "a3": 0.8, "b1": 2.0, "b2": 2.3, "b3": 2.9}
    rows = [
        make_row(
            pose_id,
            "backend_a" if pose_id[0] == "a" else "backend_b",
            int(pose_id[1]),
            native={"ranking_score": score, "gpde": score},
        )
        for (pose_id, _), score in zip(offsets.items(), (0.60, 0.70, 0.65, 0.50, 0.90, 0.80))
    ]
    values = np.array(list(offsets.values()))
    return rows, PoseDistances(tuple(offsets), np.abs(values[:, None] - values[None, :]))


class TestMedoid:
    def test_one_medoid_per_backend_and_one_overall(self):
        rows, distances = offsets_rows()
        chosen = select_representatives(rows, distances)
        assert chosen["backend_a"] == "a2"
        assert chosen["backend_b"] == "b2"
        assert chosen["rule"] == "medoid" and chosen["reasons"] == {}
        assert list(chosen) == ["backend_a", "backend_b", "all", "rule", "reasons"]

    def test_the_overall_medoid_is_decided_by_the_tie_rule(self):
        # a3 and b1 both have a distance sum of 6.0 to the other five poses
        rows, distances = offsets_rows()
        assert select_representatives(rows, distances)["all"] == "a3"

    def test_a_tie_goes_to_the_lowest_id_whatever_the_row_order(self):
        rows, distances = offsets_rows()
        for order in (rows, rows[::-1], rows[3:] + rows[:3]):
            assert select_representatives(order, distances)["all"] == "a3"

    def test_two_poses_at_equal_distance_are_a_tie(self):
        rows = [make_row("p2", "m"), make_row("p1", "m")]
        distances = PoseDistances(("p1", "p2"), [[0.0, 1.5], [1.5, 0.0]])
        assert select_representatives(rows, distances)["m"] == "p1"

    def test_a_tie_that_rounding_would_break_is_still_a_tie(self):
        rows = [make_row(f"q{i}", "m") for i in (1, 2)]
        near = 1.0 + 1e-13
        distances = PoseDistances(("q1", "q2"), [[0.0, near], [1.0, 0.0]])
        assert select_representatives(rows, distances)["m"] == "q1"

    def test_a_failed_pose_is_not_a_candidate_and_need_not_be_in_the_matrix(self):
        rows, distances = offsets_rows()
        rows[1]["status"] = "failed"  # a2
        rows.append(make_row("c1", "backend_c", status="failed"))
        rows[-1]["status"] = "failed"
        chosen = select_representatives(rows, distances)
        assert chosen["backend_a"] in {"a1", "a3"}
        assert chosen["backend_c"] is None
        assert "failed" in chosen["reasons"]["backend_c"]
        assert chosen["all"] != "a2"

    def test_a_group_of_one_pose_is_represented_by_it(self):
        rows, distances = offsets_rows()
        rows = [row for row in rows if row["id"] in ("a1", "b1", "b2")]
        assert select_representatives(rows, distances)["backend_a"] == "a1"

    def test_a_pose_with_no_comparable_pose_is_not_a_candidate(self):
        rows = [make_row(pose_id, "m") for pose_id in ("p1", "p2", "p3")]
        nan = float("nan")
        distances = PoseDistances(
            ("p1", "p2", "p3"), [[0.0, 1.0, nan], [1.0, 0.0, nan], [nan, nan, 0.0]]
        )
        assert select_representatives(rows, distances)["m"] == "p1"

    def test_medoid_needs_the_matrix(self):
        rows, _ = offsets_rows()
        with pytest.raises(ValueError, match="distance matrix"):
            select_representatives(rows)

    def test_a_pose_missing_from_the_matrix_is_refused(self):
        rows, distances = offsets_rows()
        rows.append(make_row("z9", "backend_a"))
        with pytest.raises(ValueError, match="z9"):
            select_representatives(rows, distances)

    def test_the_matrix_is_checked(self):
        with pytest.raises(ValueError, match="shape"):
            PoseDistances(("a", "b"), [[0.0, 1.0]])
        with pytest.raises(ValueError, match="unique"):
            PoseDistances(("a", "a"), [[0.0, 1.0], [1.0, 0.0]])
        distances = PoseDistances(("a", "b"), [[0.0, 1.0], [1.0, 0.0]])
        assert distances.distance("a", "b") == 1.0
        assert distances.submatrix(["b", "a"]).tolist() == [[0.0, 1.0], [1.0, 0.0]]
        with pytest.raises(ValueError, match="no row"):
            distances.submatrix(["c"])

    def test_the_rows_need_unique_ids(self):
        rows, distances = offsets_rows()
        rows[1]["id"] = "a1"
        with pytest.raises(ValueError, match="not unique"):
            select_representatives(rows, distances)


class TestConfidence:
    def test_the_best_native_score_per_backend_largest_for_the_ranking_score(self):
        rows, _ = offsets_rows()
        chosen = select_representatives(rows, rule="confidence")
        assert chosen["backend_a"] == "a2"  # 0.70 against 0.60 and 0.65
        assert chosen["backend_b"] == "b2"  # 0.90 against 0.50 and 0.80
        assert chosen["rule"] == "confidence"

    def test_smallest_for_a_score_where_lower_is_better(self):
        rows, _ = offsets_rows()  # gpde equals the ranking score here
        chosen = select_representatives(rows, rule="confidence", confidence_key="gpde")
        assert chosen["backend_a"] == "a1"  # 0.60 is the smallest of 0.60, 0.70, 0.65
        assert chosen["backend_b"] == "b1"  # 0.50 is the smallest of 0.50, 0.90, 0.80
        ranking = select_representatives(rows, rule="confidence", confidence_key="ranking_score")
        assert (ranking["backend_a"], ranking["backend_b"]) == ("a2", "b2")

    def test_the_direction_can_be_given_for_a_score_with_no_known_direction(self):
        rows, _ = offsets_rows()
        for row in rows:
            row["native"]["haddock_score"] = -row["native"]["ranking_score"]
        with pytest.raises(ValueError, match="direction"):
            select_representatives(rows, rule="confidence", confidence_key="haddock_score")
        low = select_representatives(
            rows, rule="confidence", confidence_key="haddock_score", confidence_direction="min"
        )
        high = select_representatives(
            rows, rule="confidence", confidence_key="haddock_score", confidence_direction="max"
        )
        assert (low["backend_a"], high["backend_a"]) == ("a2", "a1")

    def test_an_invalid_direction_is_refused(self):
        rows, _ = offsets_rows()
        with pytest.raises(ValueError, match="'max' or 'min'"):
            select_representatives(
                rows, rule="confidence", confidence_key="x", confidence_direction="best"
            )

    def test_the_overall_representative_is_none_when_scores_are_not_comparable(self):
        rows, _ = offsets_rows()
        chosen = select_representatives(rows, rule="confidence")
        assert chosen["all"] is None and "not comparable" in chosen["reasons"]["all"]

    def test_one_backend_makes_the_overall_representative_comparable(self):
        rows, _ = offsets_rows()
        one_backend = [row for row in rows if row["id"].startswith("a")]
        chosen = select_representatives(one_backend, rule="confidence")
        assert chosen["all"] == chosen["backend_a"] == "a2"

    def test_a_tie_goes_to_the_lowest_id(self):
        rows = [make_row(pose_id, "m", native={"iptm": 0.8}) for pose_id in ("p3", "p1", "p2")]
        assert select_representatives(rows, rule="confidence", confidence_key="iptm")["m"] == "p1"

    def test_poses_without_a_finite_score_are_skipped_and_a_group_without_any_is_none(self):
        rows = [
            make_row("p1", "m", native={"ranking_score": None}),
            make_row("p2", "m", native={"ranking_score": float("nan")}),
            make_row("p3", "m", native={"ranking_score": 0.2}),
            make_row("q1", "n", native={}),
        ]
        chosen = select_representatives(rows, rule="confidence")
        assert chosen["m"] == "p3" and chosen["n"] is None
        assert "ranking_score" in chosen["reasons"]["n"]

    def test_a_failed_pose_is_never_chosen_even_with_the_best_score(self):
        rows, _ = offsets_rows()
        rows[1]["status"] = "failed"  # a2 has the best ranking score of backend_a
        assert select_representatives(rows, rule="confidence")["backend_a"] == "a3"

    def test_native_score_directions(self):
        assert native_score_direction("iptm") == "max"
        assert native_score_direction("gpde") == "min"
        assert native_score_direction("chain_pair_iptm.A-B") == "max"
        assert native_score_direction("haddock_score") is None


class TestCustomRule:
    def test_rank_by_a_key_function_smallest_and_largest(self):
        rows = sasa_rows([1266.8, 1280.1, 1324.7])
        smallest = select_representatives(
            rows, rule=rank_by(lambda row: row["interface"]["delta_sasa"])
        )
        largest = select_representatives(
            rows, rule=rank_by(lambda row: row["interface"]["delta_sasa"], largest=True)
        )
        assert (smallest["a"], largest["a"]) == ("a1", "a3")
        assert smallest["rule"] == "custom"

    def test_rank_by_serves_the_overall_pose_across_backends(self):
        rows = sasa_rows([10.0, 20.0], "a") + sasa_rows([5.0, 30.0], "b")
        chosen = select_representatives(
            rows, rule=rank_by(lambda row: row["interface"]["delta_sasa"])
        )
        assert (chosen["a"], chosen["b"], chosen["all"]) == ("a1", "b1", "b1")

    def test_rank_by_skips_rows_whose_key_fails_and_breaks_ties_by_id(self):
        rows = [
            make_row("p3", "m", interface={"x": 1.0}),
            make_row("p1", "m", interface={"x": 1.0}),
            make_row("p2", "m", interface={}),
            make_row("p4", "m", interface={"x": float("nan")}),
        ]
        assert (
            select_representatives(rows, rule=rank_by(lambda r: r["interface"]["x"]))["m"] == "p1"
        )

    def test_a_callable_gets_the_rows_that_did_not_fail_sorted_by_id_and_the_distances(self):
        rows, distances = offsets_rows()
        rows[0]["status"] = "failed"
        seen = []

        def rule(group_rows, matrix):
            seen.append(([row["id"] for row in group_rows], matrix))
            return group_rows[-1]["id"]

        chosen = select_representatives(rows, distances, rule=rule)
        assert seen[0][0] == ["a2", "a3"] and seen[0][1] is distances
        assert (chosen["backend_a"], chosen["backend_b"], chosen["all"]) == ("a3", "b3", "b3")

    def test_a_callable_that_raises_or_returns_a_stranger_gives_none_and_a_reason(self):
        rows, _ = offsets_rows()

        def raising(group_rows, matrix):
            raise RuntimeError("boom")

        chosen = select_representatives(rows, rule=raising)
        assert (
            chosen["backend_a"] is None and chosen["reasons"]["backend_a"] == "RuntimeError: boom"
        )
        chosen = select_representatives(rows, rule=lambda group_rows, matrix: "zzz")
        assert "not a pose of the group" in chosen["reasons"]["all"]
        chosen = select_representatives(rows, rule=lambda group_rows, matrix: None)
        assert chosen["reasons"]["all"] == "the rule returned None"

    def test_unknown_rule_is_refused(self):
        rows, distances = offsets_rows()
        with pytest.raises(ValueError, match="rule"):
            select_representatives(rows, distances, rule="best")

    def test_the_representative_is_a_member_of_its_group(self):
        rows, distances = offsets_rows()
        by_group = {}
        for row in rows:
            by_group.setdefault(group_key(row), set()).add(row["id"])
        chosen = select_representatives(rows, distances)
        for group, members in by_group.items():
            assert chosen[group] in members
        assert chosen["all"] in {row["id"] for row in rows}
        assert math.isfinite(distances.distance(chosen["all"], "a1"))
