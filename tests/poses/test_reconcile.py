"""Gaps, consensus interfaces, rank agreement and hotspot consistency (poses/reconcile.py)."""

import json
import math

import numpy as np
import pytest

from binding_metrics.poses.aggregate import aggregate_poses
from binding_metrics.poses.reconcile import gap_between_backends

KEY = "interface.delta_sasa"


def make_row(pose_id, backend, seed, status="ok", **blocks):
    return {
        "id": pose_id,
        "kind": "predicted",
        "status": status,
        "reason": None if status == "ok" else "pose failed",
        "reasons": {},
        "provenance": {"backend": backend, "seed": seed},
        "native": {},
        **blocks,
    }


def value_rows(values_by_backend, key_block="interface", key_name="delta_sasa"):
    rows = []
    for backend, values in values_by_backend.items():
        for seed, value in enumerate(values, start=1):
            rows.append(
                make_row(f"{backend}{seed}", backend, seed, **{key_block: {key_name: value}})
            )
    return rows


def aggregates_of(values_by_backend, **kwargs):
    return aggregate_poses(value_rows(values_by_backend), **kwargs)


def entry(n_ok, mean=None, std=None, **extra):
    """An aggregate entry as ``aggregate_poses`` writes it, for the fields a gap reads."""
    return {"n_ok": n_ok, "n_total": n_ok, "mean": mean, "std": std, **extra}


def pooled_by_hand(values_a, values_b):
    """The pooled deviation as the residual of each back-end around its own mean."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    residuals = np.concatenate([a - a.mean(), b - b.mean()])
    return float(np.sqrt(np.sum(residuals**2) / (a.size + b.size - 2)))


class TestGap:
    def test_gap_and_standardised_gap_follow_the_formulas(self):
        rng = np.random.default_rng(7)
        values_a = rng.normal(1290.0, 40.0, size=5)
        values_b = rng.normal(1280.0, 70.0, size=4)
        result = gap_between_backends(aggregates_of({"a": values_a, "b": values_b}), "a", "b")
        got = result[KEY]
        pooled = pooled_by_hand(values_a, values_b)
        assert got["gap"] == pytest.approx(values_a.mean() - values_b.mean(), rel=1e-12)
        assert got["scale"] == pytest.approx(pooled, rel=1e-12)
        assert got["standardised_gap"] == pytest.approx(
            (values_a.mean() - values_b.mean()) / pooled, rel=1e-12
        )
        assert got["scale_kind"] == "pooled"
        assert (got["mean_a"], got["mean_b"]) == (
            pytest.approx(values_a.mean()),
            pytest.approx(values_b.mean()),
        )
        assert (got["n_ok_a"], got["n_ok_b"]) == (5, 4)
        assert got["reasons"] == {}

    def test_pooled_scale_is_the_stated_expression_of_the_sample_deviations(self):
        values_a, values_b = [10.0, 12.0, 17.0], [4.0, 9.0]
        got = gap_between_backends(aggregates_of({"a": values_a, "b": values_b}), "a", "b")[KEY]
        s_a, s_b = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
        expected = math.sqrt((2 * s_a**2 + 1 * s_b**2) / (3 + 2 - 2))
        assert got["scale"] == pytest.approx(expected, rel=1e-12)

    def test_worked_example_numbers_on_made_up_values(self):
        # Three values per back-end built as mean - s, mean, mean + s so that the sample
        # deviation is exactly s. The means and deviations are those of the design note's
        # worked example (rounded there to +6.5, 64.6 and 0.10); the values are made up.
        values_a = [1290.5 - 30.3, 1290.5, 1290.5 + 30.3]
        values_b = [1284.0 - 86.1, 1284.0, 1284.0 + 86.1]
        got = gap_between_backends(aggregates_of({"a": values_a, "b": values_b}), "a", "b")[KEY]
        assert got["gap"] == pytest.approx(6.5, abs=1e-9)
        assert got["scale"] == pytest.approx(math.sqrt((30.3**2 + 86.1**2) / 2), rel=1e-12)
        assert got["scale"] == pytest.approx(64.6, abs=0.1)
        assert round(got["standardised_gap"], 2) == 0.10

    def test_sign_follows_the_order_of_the_back_ends(self):
        aggregates = aggregates_of({"a": [10.0, 12.0, 14.0], "b": [5.0, 6.0, 7.0]})
        forward = gap_between_backends(aggregates, "a", "b")[KEY]
        backward = gap_between_backends(aggregates, "b", "a")[KEY]
        assert forward["gap"] == pytest.approx(6.0)
        assert forward["gap"] > 0
        assert backward["gap"] == pytest.approx(-forward["gap"])
        assert backward["standardised_gap"] == pytest.approx(-forward["standardised_gap"])
        assert backward["scale"] == pytest.approx(forward["scale"])
        assert (backward["mean_a"], backward["n_ok_a"]) == (forward["mean_b"], forward["n_ok_b"])

    def test_every_numeric_key_is_reported_in_sorted_order(self):
        rows = [
            make_row(
                f"{backend}{seed}",
                backend,
                seed,
                interface={"delta_sasa": 100.0 * seed + offset, "n_hbonds": seed},
                native={"iptm": 0.5 + 0.1 * seed},
            )
            for backend, offset in (("a", 10.0), ("b", 0.0))
            for seed in (1, 2, 3)
        ]
        result = gap_between_backends(aggregate_poses(rows), "a", "b")
        assert list(result) == ["interface.delta_sasa", "interface.n_hbonds", "native.iptm"]
        assert result["interface.delta_sasa"]["gap"] == pytest.approx(10.0)
        assert result["interface.n_hbonds"]["gap"] == pytest.approx(0.0)
        assert result["interface.n_hbonds"]["standardised_gap"] == pytest.approx(0.0)
        assert result["native.iptm"]["gap"] == pytest.approx(0.0)

    def test_keys_argument_restricts_and_may_name_a_missing_key(self):
        aggregates = aggregates_of({"a": [1.0, 2.0], "b": [3.0, 5.0]})
        result = gap_between_backends(aggregates, "a", "b", keys=[KEY, "native.iptm"])
        assert list(result) == [KEY, "native.iptm"]
        missing = result["native.iptm"]
        assert missing["gap"] is None and missing["standardised_gap"] is None
        assert "no aggregate for this key" in missing["reasons"]["gap"]
        assert result[KEY]["gap"] == pytest.approx(-2.5)


class TestGapMappingScale:
    def test_scale_mapping_replaces_the_pooled_deviation(self):
        aggregates = aggregates_of({"a": [10.0, 12.0, 14.0], "b": [5.0, 6.0, 7.0]})
        got = gap_between_backends(aggregates, "a", "b", scale={KEY: 4.0})[KEY]
        assert got["gap"] == pytest.approx(6.0)
        assert got["scale"] == 4.0 and got["scale_kind"] == "given"
        assert got["standardised_gap"] == pytest.approx(1.5)
        assert got["reasons"] == {}

    def test_given_scale_needs_neither_two_seeds_nor_a_spread(self):
        aggregates = aggregates_of({"a": [10.0], "b": [4.0]})
        got = gap_between_backends(aggregates, "a", "b", scale={KEY: 3.0})[KEY]
        assert got["standardised_gap"] == pytest.approx(2.0)
        pooled = gap_between_backends(aggregates, "a", "b")[KEY]
        assert pooled["gap"] == pytest.approx(6.0) and pooled["standardised_gap"] is None

    def test_a_key_missing_from_the_mapping_is_none_with_a_reason(self):
        aggregates = aggregates_of({"a": [10.0, 12.0], "b": [5.0, 6.0]})
        got = gap_between_backends(aggregates, "a", "b", scale={"other.key": 2.0})[KEY]
        assert got["gap"] == pytest.approx(5.5)
        assert got["standardised_gap"] is None and got["scale"] is None
        assert got["reasons"] == {"standardised_gap": f"no scale given for {KEY!r}"}

    @pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("nan"), float("inf"), None, "2", True])
    def test_a_scale_that_is_not_positive_and_finite_raises(self, bad):
        aggregates = aggregates_of({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="positive finite"):
            gap_between_backends(aggregates, "a", "b", scale={KEY: bad})

    @pytest.mark.parametrize("bad", ["standard", 2.0, None, ["pooled"]])
    def test_scale_must_be_pooled_or_a_mapping(self, bad):
        aggregates = aggregates_of({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="scale must be"):
            gap_between_backends(aggregates, "a", "b", scale=bad)

    def test_scale_mapping_keys_must_be_strings(self):
        aggregates = aggregates_of({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="scale keys"):
            gap_between_backends(aggregates, "a", "b", scale={3: 1.0})


class TestGapNoneWithReason:
    def test_a_back_end_without_valid_values_gives_no_gap(self):
        rows = value_rows({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        for row in rows:
            if row["provenance"]["backend"] == "a":
                row["status"], row["reason"] = "failed", "could not be aligned"
        # a failed pose gives no key of its own: ask for the key explicitly
        aggregates = aggregate_poses(rows, keys=[KEY])
        assert aggregates["a"][KEY]["n_ok"] == 0
        got = gap_between_backends(aggregates, "a", "b")[KEY]
        assert got["gap"] is None and got["standardised_gap"] is None
        assert "'a'" in got["reasons"]["gap"] and "n_ok = 0" in got["reasons"]["gap"]
        assert got["reasons"]["standardised_gap"] == got["reasons"]["gap"]
        assert got["n_ok_a"] == 0 and got["n_ok_b"] == 2
        assert got["mean_a"] is None and got["mean_b"] == pytest.approx(3.5)

    def test_both_back_ends_empty_names_both_in_the_reason(self):
        aggregates = {"a": {KEY: entry(0, reasons={"x": 1})}, "b": {KEY: entry(0)}}
        got = gap_between_backends(aggregates, "a", "b")[KEY]
        assert got["gap"] is None
        assert "'a'" in got["reasons"]["gap"] and "'b'" in got["reasons"]["gap"]

    def test_one_value_per_back_end_gives_the_gap_but_no_standardised_gap(self):
        got = gap_between_backends(aggregates_of({"a": [10.0], "b": [4.0]}), "a", "b")[KEY]
        assert got["gap"] == pytest.approx(6.0)
        assert got["standardised_gap"] is None and got["scale"] is None
        assert "at least 2 valid values per back-end" in got["reasons"]["standardised_gap"]
        assert "n_ok: 1 and 1" in got["reasons"]["standardised_gap"]
        assert "gap" not in got["reasons"]

    def test_one_back_end_with_a_single_value_is_enough_to_block_the_scale(self):
        got = gap_between_backends(aggregates_of({"a": [1.0, 2.0, 3.0], "b": [4.0]}), "a", "b")[KEY]
        assert got["gap"] == pytest.approx(-2.0)
        assert got["standardised_gap"] is None
        assert "n_ok: 3 and 1" in got["reasons"]["standardised_gap"]

    def test_identical_values_make_the_pooled_deviation_zero(self):
        got = gap_between_backends(aggregates_of({"a": [5.0, 5.0], "b": [3.0, 3.0]}), "a", "b")[KEY]
        assert got["gap"] == pytest.approx(2.0)
        assert got["scale"] == 0.0
        assert got["standardised_gap"] is None
        assert "pooled standard deviation is 0" in got["reasons"]["standardised_gap"]

    def test_zero_gap_over_zero_spread_is_still_none_not_nan(self):
        got = gap_between_backends(aggregates_of({"a": [5.0, 5.0], "b": [5.0, 5.0]}), "a", "b")[KEY]
        assert got["gap"] == 0.0 and got["standardised_gap"] is None

    def test_min_n_ok_that_blanks_the_mean_blanks_the_gap_with_its_reason(self):
        aggregates = aggregates_of({"a": [1.0, 2.0], "b": [3.0, 4.0, 5.0]}, min_n_ok=3)
        got = gap_between_backends(aggregates, "a", "b")[KEY]
        assert got["gap"] is None and got["mean_a"] is None
        assert got["n_ok_a"] == 2 and got["n_ok_b"] == 3
        assert "'a' has no mean" in got["reasons"]["gap"]
        assert "below min_n_ok" in got["reasons"]["gap"]

    def test_a_missing_std_is_reported_when_the_scale_is_pooled(self):
        aggregates = {"a": {KEY: entry(3, 10.0)}, "b": {KEY: entry(3, 4.0, 1.0)}}
        got = gap_between_backends(aggregates, "a", "b")[KEY]
        assert got["gap"] == pytest.approx(6.0) and got["standardised_gap"] is None
        assert "'a' has no standard deviation" in got["reasons"]["standardised_gap"]

    def test_a_single_back_end_gives_none_with_a_reason_for_every_key(self):
        aggregates = aggregates_of({"a": [1.0, 2.0, 3.0]})
        result = gap_between_backends(aggregates, "a", "b")
        assert list(result) == [KEY]
        got = result[KEY]
        assert got["gap"] is None and got["standardised_gap"] is None
        assert "no aggregates for back-end 'b'" in got["reasons"]["gap"]
        assert got["n_ok_a"] == 3 and got["n_ok_b"] == 0

    def test_nothing_in_a_result_is_nan(self):
        cases = [
            aggregates_of({"a": [5.0, 5.0], "b": [5.0, 5.0]}),
            aggregates_of({"a": [10.0], "b": [4.0]}),
            aggregates_of({"a": [1.0, 2.0, 3.0]}),
            {"a": {KEY: entry(0)}, "b": {KEY: entry(2, 1.0, 0.5)}},
        ]
        for aggregates in cases:
            result = gap_between_backends(aggregates, "a", "b")
            json.dumps(result, allow_nan=False)


class TestGapValidation:
    def test_neither_back_end_in_the_aggregates_raises_and_lists_the_groups(self):
        aggregates = aggregates_of({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match=r"neither 'x' nor 'y'.*\['a', 'all'\]"):
            gap_between_backends(aggregates, "x", "y")

    def test_empty_aggregates_raise(self):
        with pytest.raises(ValueError, match="neither"):
            gap_between_backends({}, "a", "b")

    def test_two_groups_without_keys_give_an_empty_result(self):
        assert gap_between_backends({"a": {}, "b": {}}, "a", "b") == {}

    @pytest.mark.parametrize("a, b", [("a", "a"), ("", "b"), (None, "b"), ("a", 3)])
    def test_back_end_names_must_be_distinct_strings(self, a, b):
        with pytest.raises(ValueError, match="backend_[ab]"):
            gap_between_backends({"a": {}, "b": {}}, a, b)

    def test_aggregates_must_be_a_mapping_of_mappings(self):
        with pytest.raises(ValueError, match="aggregate_poses"):
            gap_between_backends([1, 2], "a", "b")
        with pytest.raises(ValueError, match=r"aggregates\['a'\]"):
            gap_between_backends({"a": [1], "b": {}}, "a", "b")

    def test_an_entry_without_n_ok_raises(self):
        with pytest.raises(ValueError, match="n_ok"):
            gap_between_backends({"a": {KEY: {"mean": 1.0}}, "b": {KEY: entry(1, 2.0)}}, "a", "b")
        with pytest.raises(ValueError, match="non-negative integer"):
            gap_between_backends({"a": {KEY: entry(-1, 1.0)}, "b": {KEY: entry(1, 2.0)}}, "a", "b")

    def test_keys_must_be_a_list_of_key_paths(self):
        aggregates = aggregates_of({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="list of key paths"):
            gap_between_backends(aggregates, "a", "b", keys=KEY)
        with pytest.raises(ValueError, match="non-empty key paths"):
            gap_between_backends(aggregates, "a", "b", keys=[KEY, ""])

    def test_the_input_is_not_modified(self):
        aggregates = aggregates_of({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.5]})
        before = json.dumps(aggregates, sort_keys=True)
        gap_between_backends(aggregates, "a", "b")
        assert json.dumps(aggregates, sort_keys=True) == before
