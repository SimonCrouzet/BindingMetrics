"""Gaps, consensus interfaces, rank agreement and hotspot consistency (poses/reconcile.py)."""

import json
import math

import numpy as np
import pytest

from binding_metrics.poses.aggregate import aggregate_poses
from binding_metrics.poses.reconcile import (
    ConsensusInterface,
    compare_consensus_interfaces,
    consensus_interface,
    gap_between_backends,
    hotspot_consistency,
    rank_agreement,
)
from binding_metrics.poses.reference import ResidueKey

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


# ---------------------------------------------------------------------- consensus interface


def residue_table(binder=(), receptor=(), n_binder=8, unmapped=0, chain="A"):
    """A residue table in the format of the design note (8.5): the columns that are read.

    ``binder`` holds the 1-based positions in the interface, ``receptor`` the reference
    residues in the interface, as numbers or ``(number, icode)`` pairs on ``chain``;
    ``unmapped`` adds receptor interface residues that have no reference key.
    """
    table = [
        {"role": "binder", "binder_index": i, "in_interface": i in set(binder)}
        for i in range(1, n_binder + 1)
    ]
    for item in receptor:
        number, icode = item if isinstance(item, tuple) else (item, "")
        table.append(
            {
                "role": "receptor",
                "binder_index": None,
                "reference_chain": chain,
                "reference_number": number,
                "reference_icode": icode,
                "in_interface": True,
            }
        )
    table.append(  # a receptor residue outside the interface never counts
        {
            "role": "receptor",
            "reference_chain": chain,
            "reference_number": 999,
            "reference_icode": "",
            "in_interface": False,
        }
    )
    for _ in range(unmapped):
        table.append(
            {
                "role": "receptor",
                "reference_chain": None,
                "reference_number": None,
                "reference_icode": None,
                "in_interface": True,
            }
        )
    return table


def pose_row(pose_id, backend, binder=(), receptor=(), **kwargs):
    return make_row(
        pose_id,
        backend,
        1,
        residue_table=residue_table(binder, receptor, **kwargs),
    )


def four_poses(backend="a"):
    """Binder 1 in 4 of 4 poses, 2 in 2, 3 in 1; receptor 25 in 4, 26 in 3, 30 in 2, 31 in 1."""
    return [
        pose_row(f"{backend}1", backend, [1, 2, 3], [25, 26, 30, 31]),
        pose_row(f"{backend}2", backend, [1, 2], [25, 26, 30]),
        pose_row(f"{backend}3", backend, [1], [25, 26]),
        pose_row(f"{backend}4", backend, [1], [25]),
    ]


class TestConsensusInterface:
    def test_default_threshold_keeps_residues_held_by_at_least_half_of_the_poses(self):
        consensus = consensus_interface(four_poses(), "a")
        assert isinstance(consensus, ConsensusInterface)
        assert consensus.binder == {1, 2}
        assert consensus.receptor == {ResidueKey("A", n) for n in (25, 26, 30)}
        assert (consensus.n_poses, consensus.n_total, consensus.reason) == (4, 4, None)
        assert consensus.binder_fractions == {1: 1.0, 2: 0.5, 3: 0.25}
        assert consensus.receptor_fractions[ResidueKey("A", 26)] == 0.75
        assert consensus.receptor_fractions[ResidueKey("A", 31)] == 0.25

    @pytest.mark.parametrize(
        "min_fraction, binder, receptor",
        [
            (0.25, {1, 2, 3}, {25, 26, 30, 31}),  # the fraction is reached exactly
            (0.2500001, {1, 2}, {25, 26, 30}),  # just above the fraction of the rarest
            (0.5, {1, 2}, {25, 26, 30}),  # exactly half is in
            (0.5000001, {1}, {25, 26}),  # just above half is out
            (0.75, {1}, {25, 26}),
            (1.0, {1}, {25}),
        ],
    )
    def test_boundary_of_min_fraction(self, min_fraction, binder, receptor):
        consensus = consensus_interface(four_poses(), "a", min_fraction=min_fraction)
        assert consensus.binder == binder
        assert consensus.receptor == {ResidueKey("A", n) for n in receptor}

    def test_a_decimal_fraction_is_reached_by_the_matching_count(self):
        rows = [pose_row(f"p{i}", "a", [1] if i < 7 else [2]) for i in range(10)]
        assert consensus_interface(rows, "a", 0.7).binder == {1}
        assert consensus_interface(rows, "a", 0.71).binder == frozenset()
        # 0.1 + 0.2 is 0.30000000000000004 in floating point; 3 of 10 must still reach it
        assert consensus_interface(rows, "a", 0.1 + 0.2).binder == {1, 2}

    def test_failed_poses_and_rows_without_a_table_do_not_vote_and_are_counted(self):
        rows = [
            pose_row("a1", "a", [1, 2], [25]),
            pose_row("a2", "a", [1], [26]),
            make_row("a3", "a", 3, status="failed"),
            make_row("a4", "a", 4),
        ]
        consensus = consensus_interface(rows, "a")
        assert (consensus.n_poses, consensus.n_total) == (2, 4)
        assert consensus.binder == {1, 2}  # 2 of the 2 voters hold 1, 1 of 2 holds 2
        assert consensus.receptor == {ResidueKey("A", 25), ResidueKey("A", 26)}
        assert consensus.skipped == {"pose failed": 1, "the row has no residue_table": 1}

    def test_rows_of_other_back_ends_are_ignored(self):
        rows = four_poses("a") + [pose_row("b1", "b", [7], [90])]
        consensus = consensus_interface(rows, "a")
        assert 7 not in consensus.binder and ResidueKey("A", 90) not in consensus.receptor
        assert consensus_interface(rows, "b").binder == {7}

    def test_insertion_codes_make_distinct_residues(self):
        rows = [
            pose_row("a1", "a", receptor=[52, (52, "A")]),
            pose_row("a2", "a", receptor=[(52, "A")]),
        ]
        consensus = consensus_interface(rows, "a", min_fraction=1.0)
        assert consensus.receptor == {ResidueKey("A", 52, "A")}
        assert consensus_interface(rows, "a", 0.5).receptor == {
            ResidueKey("A", 52),
            ResidueKey("A", 52, "A"),
        }

    def test_unmapped_receptor_interface_residues_are_counted_and_left_out(self):
        rows = [pose_row("a1", "a", [1], [25], unmapped=2), pose_row("a2", "a", [1], [25])]
        consensus = consensus_interface(rows, "a")
        assert consensus.n_unmapped_receptor == 2
        assert consensus.receptor == {ResidueKey("A", 25)}

    def test_without_a_valid_pose_the_sets_are_empty_and_the_reason_says_why(self):
        none_for_backend = consensus_interface(four_poses("a"), "b")
        assert (none_for_backend.n_poses, none_for_backend.n_total) == (0, 0)
        assert none_for_backend.binder == frozenset() and none_for_backend.receptor == frozenset()
        assert "no row for back-end 'b'" in none_for_backend.reason
        only_failed = consensus_interface([make_row("a1", "a", 1, status="failed")], "a")
        assert only_failed.n_poses == 0 and only_failed.n_total == 1
        assert "no valid pose" in only_failed.reason
        assert consensus_interface([], "a").n_total == 0

    def test_an_empty_interface_in_every_pose_is_a_valid_empty_consensus(self):
        consensus = consensus_interface([pose_row("a1", "a"), pose_row("a2", "a")], "a")
        assert consensus.n_poses == 2 and consensus.reason is None
        assert consensus.binder == frozenset() and consensus.receptor == frozenset()

    def test_to_dict_is_json_ready_and_sorted(self):
        data = consensus_interface(four_poses(), "a").to_dict()
        json.dumps(data, allow_nan=False)
        assert data["binder"] == [1, 2]
        assert data["receptor"] == [["A", 25, ""], ["A", 26, ""], ["A", 30, ""]]
        assert data["binder_fractions"] == [[1, 1.0], [2, 0.5], [3, 0.25]]
        assert data["reason"] is None and data["n_poses"] == 4

    def test_input_order_does_not_matter(self):
        rows = four_poses()
        assert consensus_interface(rows[::-1], "a") == consensus_interface(rows, "a")

    @pytest.mark.parametrize("bad", [0, 0.0, -0.1, 1.01, float("nan"), None, True, "0.5"])
    def test_min_fraction_must_be_in_the_half_open_unit_interval(self, bad):
        with pytest.raises(ValueError, match="min_fraction"):
            consensus_interface(four_poses(), "a", min_fraction=bad)

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_backend_must_be_a_non_empty_string(self, bad):
        with pytest.raises(ValueError, match="backend"):
            consensus_interface(four_poses(), bad)

    def test_malformed_rows_and_table_lines_raise_with_the_pose_and_line(self):
        with pytest.raises(ValueError, match="row 0 is a str"):
            consensus_interface(["a1"], "a")
        base = pose_row("a1", "a", [1])

        def with_line(line):
            row = dict(base)
            row["residue_table"] = [dict(base["residue_table"][0]), line]
            return [row]

        with pytest.raises(ValueError, match=r"line 1 .*'a1'.*role 'ligand'"):
            consensus_interface(with_line({"role": "ligand", "in_interface": True}), "a")
        with pytest.raises(ValueError, match="in_interface None"):
            consensus_interface(with_line({"role": "binder", "binder_index": 1}), "a")
        with pytest.raises(ValueError, match="binder_index 'x'"):
            consensus_interface(
                with_line({"role": "binder", "binder_index": "x", "in_interface": True}), "a"
            )
        with pytest.raises(ValueError, match="not a mapping"):
            consensus_interface(with_line("binder"), "a")
        with pytest.raises(ValueError, match="no valid reference residue"):
            line = {
                "role": "receptor",
                "reference_chain": "A",
                "reference_number": "25",
                "in_interface": True,
            }
            consensus_interface(with_line(line), "a")


class TestCompareConsensus:
    @staticmethod
    def consensus(backend, binder, receptor):
        rows = [pose_row(f"{backend}1", backend, binder, receptor)]
        return consensus_interface(rows, backend, min_fraction=1.0)

    def test_jaccard_and_difference_lists_on_hand_built_sets(self):
        a = self.consensus("a", [1, 2, 3], [25, 26, 50])
        b = self.consensus("b", [2, 3, 4], [26, 51, 52])
        got = compare_consensus_interfaces(a, b)
        assert (got["backend_a"], got["backend_b"]) == ("a", "b")
        assert got["jaccard_binder"] == pytest.approx(2 / 4)
        assert got["jaccard_receptor"] == pytest.approx(1 / 5)
        assert got["jaccard"] == pytest.approx(3 / 9)  # 2 + 1 shared of 4 + 5 in the union
        key = lambda n: ResidueKey("A", n)  # noqa: E731
        assert got["only_a"] == {"binder": [1], "receptor": [key(25), key(50)]}
        assert got["only_b"] == {"binder": [4], "receptor": [key(51), key(52)]}
        assert got["reasons"] == {} and "below_limit" not in got

    def test_swapping_the_back_ends_swaps_the_lists_and_keeps_the_value(self):
        a = self.consensus("a", [1, 2], [25])
        b = self.consensus("b", [2, 3], [25, 26])
        forward, backward = compare_consensus_interfaces(a, b), compare_consensus_interfaces(b, a)
        assert forward["jaccard"] == pytest.approx(backward["jaccard"])
        assert forward["only_a"] == backward["only_b"]
        assert forward["only_b"] == backward["only_a"]

    def test_identical_consensus_sets_have_jaccard_one_and_no_difference(self):
        a = self.consensus("a", [1, 2], [25, 26])
        b = self.consensus("b", [1, 2], [25, 26])
        got = compare_consensus_interfaces(a, b)
        assert got["jaccard"] == got["jaccard_binder"] == got["jaccard_receptor"] == 1.0
        assert got["only_a"] == got["only_b"] == {"binder": [], "receptor": []}

    def test_disjoint_sets_have_jaccard_zero(self):
        got = compare_consensus_interfaces(
            self.consensus("a", [1], [25]), self.consensus("b", [2], [26])
        )
        assert got["jaccard"] == 0.0

    def test_two_empty_consensus_sets_have_no_jaccard(self):
        a = self.consensus("a", [], [])
        b = self.consensus("b", [], [])
        got = compare_consensus_interfaces(a, b)
        assert got["jaccard"] is None and got["only_a"] == got["only_b"]
        assert got["reasons"]["jaccard"] == "both consensus interfaces are empty"
        assert got["reasons"]["jaccard_binder"] == "both consensus binder sets are empty"
        assert got["reasons"]["jaccard_receptor"] == "both consensus receptor sets are empty"

    def test_an_empty_side_with_a_non_empty_other_is_zero_not_none(self):
        got = compare_consensus_interfaces(
            self.consensus("a", [], [25]), self.consensus("b", [], [26])
        )
        assert got["jaccard_receptor"] == 0.0
        assert got["jaccard_binder"] is None
        assert list(got["reasons"]) == ["jaccard_binder"]

    def test_limit_adds_the_boolean_next_to_the_value(self):
        a = self.consensus("a", [1, 2, 3], [25, 26, 50])
        b = self.consensus("b", [2, 3, 4], [26, 51, 52])  # jaccard 1/3
        assert compare_consensus_interfaces(a, b, limit=0.5)["below_limit"] is True
        assert compare_consensus_interfaces(a, b, limit=0.2)["below_limit"] is False
        at_value = compare_consensus_interfaces(a, b, limit=3 / 9)
        assert at_value["below_limit"] is False  # strictly below
        assert at_value["jaccard"] == pytest.approx(1 / 3)

    def test_limit_with_no_jaccard_is_none_with_a_reason(self):
        a, b = self.consensus("a", [], []), self.consensus("b", [], [])
        got = compare_consensus_interfaces(a, b, limit=0.5)
        assert got["below_limit"] is None
        assert got["reasons"]["below_limit"] == got["reasons"]["jaccard"]

    def test_a_back_end_without_a_valid_pose_gives_none_everywhere(self):
        a = self.consensus("a", [1], [25])
        nothing = consensus_interface([], "b")
        got = compare_consensus_interfaces(a, nothing, limit=0.5)
        for name in ("jaccard", "jaccard_binder", "jaccard_receptor", "only_a", "only_b"):
            assert got[name] is None
            assert "no row for back-end 'b'" in got["reasons"][name]
        assert got["below_limit"] is None
        json.dumps(got, allow_nan=False)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), "0.5", True])
    def test_limit_must_be_in_the_unit_interval(self, bad):
        a = self.consensus("a", [1], [25])
        with pytest.raises(ValueError, match="limit"):
            compare_consensus_interfaces(a, a, limit=bad)

    def test_arguments_must_be_consensus_objects(self):
        a = self.consensus("a", [1], [25])
        with pytest.raises(ValueError, match="consensus_b must be a ConsensusInterface"):
            compare_consensus_interfaces(a, {"binder": [1]})
        with pytest.raises(ValueError, match="consensus_a"):
            compare_consensus_interfaces(None, a)


# ---------------------------------------------------------------------- rank agreement


def population(values_a, values_b, metric="score", constructs=None):
    """A long table: one row per construct and back-end; ``None`` leaves the value out."""
    names = constructs or [f"c{i}" for i in range(len(values_a))]
    rows = []
    for name, a, b in zip(names, values_a, values_b):
        rows.append({"construct": name, "backend": "a", metric: a})
        rows.append({"construct": name, "backend": "b", metric: b})
    return rows


class TestRankAgreementClosedForms:
    def test_spearman_without_ties_is_one_minus_six_sum_d_squared_over_n_n_squared_minus_one(self):
        x, y = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
        got = rank_agreement(population(x, y), "a", "b")["score"]
        assert got["correlation"] == pytest.approx(1 - 6 * 4 / (5 * 24), abs=1e-14)
        assert got["correlation"] == pytest.approx(0.8)
        assert (got["method"], got["n"], got["n_dropped"]) == ("spearman", 5, 0)
        assert "reason" not in got

    def test_spearman_with_ties_uses_midranks(self):
        # ranks of x: 1, 2.5, 2.5, 4; of y: 1, 3, 2, 4; Pearson of the ranks = 4.5 / sqrt(4.5 * 5)
        got = rank_agreement(population([1, 2, 2, 3], [1, 3, 2, 4]), "a", "b")["score"]
        assert got["correlation"] == pytest.approx(4.5 / math.sqrt(22.5), rel=1e-14)
        assert got["correlation"] == pytest.approx(0.9486832980505138)

    def test_spearman_is_one_for_any_increasing_relation_and_minus_one_for_a_decreasing_one(self):
        x = [0.1, 0.5, 2.0, 7.0, 40.0]
        up = rank_agreement(population(x, [3 * v**3 + 1 for v in x]), "a", "b")["score"]
        down = rank_agreement(population(x, [-math.exp(v) for v in x]), "a", "b")["score"]
        assert up["correlation"] == pytest.approx(1.0)
        assert down["correlation"] == pytest.approx(-1.0)

    def test_kendall_without_ties_is_concordant_minus_discordant_over_pairs(self):
        x, y = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]  # 8 concordant and 2 discordant pairs of 10
        got = rank_agreement(population(x, y), "a", "b", method="kendall")["score"]
        assert got["correlation"] == pytest.approx((8 - 2) / 10)
        assert got["method"] == "kendall"

    def test_kendall_tau_b_corrects_for_ties(self):
        # 5 concordant pairs, 0 discordant, one pair tied in x: tau-b = 5 / sqrt((6-1)(6-0))
        got = rank_agreement(population([1, 2, 2, 3], [1, 3, 2, 4]), "a", "b", method="kendall")
        assert got["score"]["correlation"] == pytest.approx(5 / math.sqrt(30), rel=1e-14)

    def test_kendall_with_ties_in_both_lists(self):
        # x = 1 1 2 2, y = 1 2 2 2. Pairs: (1,3) and (1,4) are concordant; (1,2) is tied in x;
        # (2,3) and (2,4) are tied in y; (3,4) is tied in both. 2 concordant, 0 discordant,
        # P_x = 4 pairs untied in x, P_y = 3 untied in y: tau-b = 2 / sqrt(4 * 3).
        got = rank_agreement(population([1, 1, 2, 2], [1, 2, 2, 2]), "a", "b", method="kendall")
        assert got["score"]["correlation"] == pytest.approx(2 / math.sqrt(12), rel=1e-14)

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_swapping_the_back_ends_keeps_the_coefficient(self, method):
        table = population([3.0, 1.0, 2.0, 5.0, 4.0], [10.0, 30.0, 20.0, 50.0, 40.0])
        forward = rank_agreement(table, "a", "b", method=method)["score"]
        backward = rank_agreement(table, "b", "a", method=method)["score"]
        assert forward["correlation"] == pytest.approx(backward["correlation"], abs=1e-14)
        assert (forward["n_missing_a"], forward["n_missing_b"]) == (0, 0)

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_agrees_with_scipy_on_seeded_data_with_ties(self, method):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(11)
        for n in (3, 4, 8, 25, 60):
            x = rng.integers(0, 6, size=n).astype(float)
            y = (x + rng.normal(0, 1.5, size=n)).round()
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            got = rank_agreement(population(list(x), list(y)), "a", "b", method=method)["score"]
            function = stats.spearmanr if method == "spearman" else stats.kendalltau
            assert got["correlation"] == pytest.approx(function(x, y)[0], abs=1e-12)

    def test_the_coefficient_never_leaves_minus_one_to_one(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            x = rng.normal(size=12)
            for method in ("spearman", "kendall"):
                for y in (x, -x, 5 * x + 2):
                    got = rank_agreement(population(list(x), list(y)), "a", "b", method=method)
                    assert -1.0 <= got["score"]["correlation"] <= 1.0


class TestRankAgreementDropsAndReasons:
    def test_constructs_without_a_value_for_either_back_end_are_dropped_and_counted(self):
        table = population(
            [1.0, 2.0, None, 4.0, 5.0, float("nan"), 7.0],
            [1.5, None, 3.0, 4.5, 5.5, 6.0, 7.5],
        )
        table.append({"construct": "only_a", "backend": "a", "score": 9.0})
        got = rank_agreement(table, "a", "b")["score"]
        assert got["n"] == 4  # c0, c3, c4, c6
        assert got["n_constructs"] == 8
        assert got["n_dropped"] == 4
        assert got["n_missing_a"] == 2  # c2 (None) and c5 (NaN)
        assert got["n_missing_b"] == 2  # c1 (None) and only_a (no row)
        assert got["correlation"] == pytest.approx(1.0)

    def test_infinity_bool_and_text_count_as_missing(self):
        table = population([1.0, 2.0, 3.0, float("inf"), True, "x", 6.0], [1, 2, 3, 4, 5, 6, 7])
        got = rank_agreement(table, "a", "b")["score"]
        assert (got["n"], got["n_missing_a"], got["n_missing_b"]) == (4, 3, 0)

    def test_a_missing_key_in_a_row_is_a_missing_value(self):
        table = population([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        del table[2]["score"]  # c1, back-end a
        got = rank_agreement(table, "a", "b")["score"]
        assert (got["n"], got["n_dropped"], got["n_missing_a"]) == (3, 1, 1)

    def test_fewer_than_three_constructs_gives_none_with_a_reason(self):
        got = rank_agreement(population([1.0, 2.0], [2.0, 1.0]), "a", "b")["score"]
        assert got["correlation"] is None and got["n"] == 2
        assert "at least 3 constructs" in got["reason"] and "n = 2" in got["reason"]
        three = rank_agreement(population([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]), "a", "b")["score"]
        assert three["correlation"] == pytest.approx(0.5) and "reason" not in three

    def test_the_reason_counts_the_dropped_constructs(self):
        table = population([1.0, None, None, 4.0], [1.0, 2.0, 3.0, 4.0])
        reason = rank_agreement(table, "a", "b")["score"]["reason"]
        assert "n = 2" in reason and "2 dropped" in reason

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_a_constant_list_gives_none_with_the_back_end_named(self, method):
        got = rank_agreement(population([5, 5, 5, 5], [1, 2, 3, 4]), "a", "b", method=method)
        assert got["score"]["correlation"] is None and got["score"]["n"] == 4
        assert "back-end 'a'" in got["score"]["reason"] and "tied" in got["score"]["reason"]
        got = rank_agreement(population([1, 2, 3, 4], [7, 7, 7, 7]), "a", "b", method=method)
        assert "back-end 'b'" in got["score"]["reason"]

    def test_no_row_for_either_back_end_gives_n_zero_and_a_reason(self):
        got = rank_agreement(
            [{"construct": "c", "backend": "z", "score": 1.0}], "a", "b", ["score"]
        )
        assert got["score"]["n"] == 0 and got["score"]["n_constructs"] == 0
        assert got["score"]["correlation"] is None and "n = 0" in got["score"]["reason"]

    def test_nothing_is_nan(self):
        tables = [
            population([5, 5, 5], [1, 2, 3]),
            population([1.0, 2.0], [2.0, 1.0]),
            population([float("nan")] * 4, [1, 2, 3, 4]),
        ]
        for table in tables:
            for method in ("spearman", "kendall"):
                json.dumps(rank_agreement(table, "a", "b", method=method), allow_nan=False)

    def test_other_back_ends_are_ignored_and_order_does_not_matter(self):
        table = population([1.0, 3.0, 2.0, 5.0, 4.0], [2.0, 1.0, 3.0, 5.0, 4.0])
        expected = rank_agreement(table, "a", "b")
        noisy = table + [{"construct": "c0", "backend": "z", "score": -1.0}]
        shuffled = np.random.default_rng(0).permutation(len(noisy))
        assert rank_agreement([noisy[i] for i in shuffled], "a", "b") == expected


class TestRankAgreementColumns:
    def table(self):
        rows = []
        for i in range(5):
            rows.append({"construct": f"c{i}", "backend": "a", "m1": i, "m2": 10 - i, "label": "x"})
            rows.append(
                {"construct": f"c{i}", "backend": "b", "m1": 2 * i, "m2": i * i, "label": "y"}
            )
        return rows

    def test_default_metrics_are_the_numeric_columns_sorted(self):
        got = rank_agreement(self.table(), "a", "b")
        assert list(got) == ["m1", "m2"]
        assert got["m1"]["correlation"] == pytest.approx(1.0)
        assert got["m2"]["correlation"] == pytest.approx(-1.0)

    def test_explicit_metrics_keep_their_order_and_may_be_absent_columns(self):
        got = rank_agreement(self.table(), "a", "b", metrics=["m2", "nope", "m1"])
        assert list(got) == ["m2", "nope", "m1"]
        assert got["nope"]["n"] == 0 and got["nope"]["correlation"] is None
        assert got["nope"]["n_missing_a"] == got["nope"]["n_missing_b"] == 5
        assert "reason" in got["nope"]

    def test_an_empty_table_gives_an_empty_result_or_n_zero(self):
        assert rank_agreement([], "a", "b") == {}
        got = rank_agreement([], "a", "b", metrics=["score"])
        assert got["score"]["n"] == 0 and "reason" in got["score"]

    def test_constructs_may_be_numbers(self):
        table = population([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], constructs=[10, 2, 33])
        assert rank_agreement(table, "a", "b")["score"]["n"] == 3


class TestRankAgreementValidation:
    def test_back_end_names_and_method(self):
        table = population([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="differ"):
            rank_agreement(table, "a", "a")
        with pytest.raises(ValueError, match="backend_b"):
            rank_agreement(table, "a", "")
        with pytest.raises(ValueError, match="method must be one of"):
            rank_agreement(table, "a", "b", method="pearson")

    def test_table_must_be_a_list_of_mappings_with_construct_and_backend(self):
        with pytest.raises(ValueError, match="list of row dictionaries"):
            rank_agreement("rows", "a", "b")
        with pytest.raises(ValueError, match="list of row dictionaries"):
            rank_agreement({"construct": "c"}, "a", "b")
        with pytest.raises(ValueError, match="table row 0 is a list"):
            rank_agreement([[1, 2]], "a", "b")
        with pytest.raises(ValueError, match="table row 0 has no 'construct'"):
            rank_agreement([{"backend": "a", "score": 1.0}], "a", "b")
        with pytest.raises(ValueError, match="table row 1 has no 'backend'"):
            rank_agreement(
                [{"construct": "c", "backend": "a"}, {"construct": "c", "score": 1.0}], "a", "b"
            )
        with pytest.raises(ValueError, match="expected a string"):
            rank_agreement([{"construct": "c", "backend": 3, "score": 1.0}], "a", "b")
        with pytest.raises(ValueError, match="unhashable"):
            rank_agreement([{"construct": ["c"], "backend": "a", "score": 1.0}], "a", "b")

    def test_two_rows_for_one_construct_and_back_end_raise(self):
        table = population([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        table.append({"construct": "c1", "backend": "a", "score": 9.0})
        with pytest.raises(ValueError, match="construct 'c1' appears twice for back-end 'a'"):
            rank_agreement(table, "a", "b")

    def test_metrics_argument_is_checked(self):
        table = population([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="list of column names"):
            rank_agreement(table, "a", "b", metrics="score")
        with pytest.raises(ValueError, match="non-empty column names"):
            rank_agreement(table, "a", "b", metrics=["score", ""])
        with pytest.raises(ValueError, match="identifies the row"):
            rank_agreement(table, "a", "b", metrics=["backend"])
        with pytest.raises(ValueError, match="repeats"):
            rank_agreement(table, "a", "b", metrics=["score", "score"])

    def test_the_table_is_not_modified(self):
        table = population([1.0, None, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        before = json.dumps(table, sort_keys=True)
        rank_agreement(table, "a", "b")
        assert json.dumps(table, sort_keys=True) == before


# ---------------------------------------------------------------------- hotspot consistency


def binder_table(areas, interface=None, threshold=0.5):
    """Binder lines of a residue table for positions 1..n; None for an unknown area.

    ``interface`` lists the positions in the interface; by default those with an area above
    ``threshold`` (0.5, the interface threshold of the toolkit).
    """
    if interface is None:
        interface = [i for i, a in enumerate(areas, start=1) if a is not None and a > threshold]
    return [
        {
            "role": "binder",
            "binder_index": i,
            "buried_area_angstrom2": area,
            "in_interface": i in set(interface),
        }
        for i, area in enumerate(areas, start=1)
    ]


def sensitivities_of(values):
    return {i: v for i, v in enumerate(values, start=1)}


class TestHotspotSpearman:
    def test_spearman_without_ties_matches_the_closed_form(self):
        areas = [10.0, 20.0, 30.0, 40.0, 50.0]
        scan = sensitivities_of([1.0, 3.0, 2.0, 5.0, 4.0])  # rank differences 0, 1, 1, 1, 1
        got = hotspot_consistency(scan, binder_table(areas))
        assert got["spearman_area_vs_sensitivity"] == pytest.approx(1 - 6 * 4 / (5 * 24))
        assert got["n_positions"] == 5 and got["reasons"] == {}

    def test_spearman_with_tied_areas_uses_midranks(self):
        # area ranks 1.5, 1.5, 3, 4, 5 and sensitivity ranks 1, 3, 2, 5, 4:
        # Pearson of the ranks = 7 / sqrt(9.5 * 10)
        areas = [0.0, 0.0, 30.0, 40.0, 50.0]
        scan = sensitivities_of([0.1, 0.3, 0.2, 0.5, 0.4])
        got = hotspot_consistency(scan, binder_table(areas))
        assert got["spearman_area_vs_sensitivity"] == pytest.approx(7 / math.sqrt(95), rel=1e-14)

    def test_more_buried_and_more_sensitive_gives_one_and_the_reverse_minus_one(self):
        areas = [5.0, 80.0, 20.0, 150.0, 0.0, 60.0]
        scan = sensitivities_of([0.2, 1.5, 0.7, 4.0, 0.0, 1.1])
        up = hotspot_consistency(scan, binder_table(areas))
        down = hotspot_consistency({k: -v for k, v in scan.items()}, binder_table(areas))
        assert up["spearman_area_vs_sensitivity"] == pytest.approx(1.0)
        assert down["spearman_area_vs_sensitivity"] == pytest.approx(-1.0)

    def test_agrees_with_scipy_on_seeded_data(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(21)
        for n in (3, 5, 13, 40):
            areas = rng.integers(0, 6, size=n).astype(float) * 20.0
            sens = (areas / 20.0 + rng.normal(0, 1.2, size=n)).round(1)
            if np.ptp(areas) == 0 or np.ptp(sens) == 0:
                continue
            got = hotspot_consistency(sensitivities_of(sens), binder_table(list(areas)))
            assert got["spearman_area_vs_sensitivity"] == pytest.approx(
                stats.spearmanr(areas, sens)[0], abs=1e-12
            )

    def test_receptor_lines_and_positions_without_a_sensitivity_are_ignored(self):
        areas = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        table = binder_table(areas) + [
            {"role": "receptor", "buried_area_angstrom2": 999.0, "in_interface": True},
            {"role": "receptor"},
        ]
        scan = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}  # positions 5 and 6 were not scanned
        got = hotspot_consistency(scan, table)
        assert got["n_positions"] == 4 and got["spearman_area_vs_sensitivity"] == pytest.approx(1.0)
        assert got["n_unmatched_positions"] == 0

    def test_order_of_the_table_and_of_the_mapping_does_not_matter(self):
        areas = [0.0, 12.0, 0.0, 44.0, 30.0, 7.0, 90.0]
        scan = sensitivities_of([0.0, 0.6, 0.1, 2.5, 1.0, 0.3, 3.0])
        expected = hotspot_consistency(scan, binder_table(areas))
        order = np.random.default_rng(2).permutation(len(areas))
        table = binder_table(areas)
        shuffled_table = [table[i] for i in order]
        shuffled_scan = {k: scan[k] for k in (int(i) + 1 for i in order)}
        assert hotspot_consistency(shuffled_scan, shuffled_table) == expected


class TestHotspotTopM:
    scan = sensitivities_of([5.0, 4.0, 3.0, 2.0, 1.0])

    def test_fraction_of_the_top_m_in_the_interface(self):
        table = binder_table([9.0] * 5, interface=[1, 2, 4])
        three = hotspot_consistency(self.scan, table, top_m=3)
        assert three["top_in_interface_fraction"] == pytest.approx(2 / 3)
        assert (three["n_top"], three["n_top_in_interface"]) == (3, 2)
        assert three["top_positions"] == [1, 2, 3] and three["top_m"] == 3
        assert hotspot_consistency(self.scan, table, top_m=2)["top_in_interface_fraction"] == 1.0
        assert hotspot_consistency(self.scan, table, top_m=1)["top_in_interface_fraction"] == 1.0
        five = hotspot_consistency(self.scan, table, top_m=5)
        assert five["top_in_interface_fraction"] == pytest.approx(3 / 5)

    def test_default_top_m_is_three(self):
        table = binder_table([9.0] * 5, interface=[1, 2, 4])
        assert hotspot_consistency(self.scan, table) == hotspot_consistency(
            self.scan, table, top_m=3
        )
        assert hotspot_consistency(self.scan, table)["top_m"] == 3

    def test_ties_at_the_cut_all_enter_the_top_set(self):
        scan = sensitivities_of([5.0, 4.0, 4.0, 4.0, 1.0])
        table = binder_table([9.0] * 5, interface=[1, 2])
        for top_m in (2, 3, 4):  # the cut falls inside the tie of positions 2, 3, 4
            got = hotspot_consistency(scan, table, top_m=top_m)
            assert got["top_positions"] == [1, 2, 3, 4]
            assert (got["n_top"], got["n_top_in_interface"]) == (4, 2)
            assert got["top_in_interface_fraction"] == pytest.approx(0.5)
        one = hotspot_consistency(scan, table, top_m=1)
        assert one["top_positions"] == [1] and one["top_in_interface_fraction"] == 1.0
        five = hotspot_consistency(scan, table, top_m=5)
        assert five["n_top"] == 5 and five["top_in_interface_fraction"] == pytest.approx(2 / 5)

    def test_the_tie_rule_does_not_pick_by_position(self):
        scan = sensitivities_of([3.0, 3.0, 3.0, 3.0])
        table = binder_table([9.0] * 4, interface=[4])
        got = hotspot_consistency(scan, table, top_m=1)
        assert got["top_positions"] == [1, 2, 3, 4]
        assert got["top_in_interface_fraction"] == pytest.approx(0.25)
        assert got["spearman_area_vs_sensitivity"] is None

    def test_top_positions_are_most_sensitive_first_with_ties_by_position(self):
        scan = {7: 1.0, 2: 3.0, 5: 3.0, 9: 2.0}
        table = [dict(line, binder_index=p) for line, p in zip(binder_table([5.0] * 4), scan)]
        got = hotspot_consistency(scan, table, top_m=3)
        assert got["top_positions"] == [2, 5, 9]

    def test_negative_sensitivities_are_ordinary_numbers(self):
        scan = sensitivities_of([-1.0, -3.0, -2.0])
        table = binder_table([9.0, 9.0, 0.0], interface=[1, 2])
        got = hotspot_consistency(scan, table, top_m=1)
        assert got["top_positions"] == [1]


class TestHotspotNoneWithReason:
    def test_fewer_than_three_overlapping_positions(self):
        got = hotspot_consistency({1: 2.0, 2: 1.0}, binder_table([30.0, 10.0]), top_m=2)
        assert got["n_positions"] == 2
        assert got["spearman_area_vs_sensitivity"] is None
        assert "at least 3 positions" in got["reasons"]["spearman_area_vs_sensitivity"]
        assert "n = 2" in got["reasons"]["spearman_area_vs_sensitivity"]
        assert got["top_in_interface_fraction"] == 1.0  # the fraction needs only top_m positions

    def test_top_m_larger_than_the_positions_gives_no_fraction(self):
        scan = sensitivities_of([3.0, 2.0, 1.0, 0.5])
        got = hotspot_consistency(scan, binder_table([9.0, 8.0, 7.0, 6.0]), top_m=5)
        assert got["top_in_interface_fraction"] is None
        assert (got["n_top"], got["n_top_in_interface"], got["top_positions"]) == (None,) * 3
        assert "fewer than top_m = 5" in got["reasons"]["top_in_interface_fraction"]
        assert got["spearman_area_vs_sensitivity"] is not None

    def test_a_constant_buried_area_gives_no_rank_correlation(self):
        got = hotspot_consistency(
            sensitivities_of([1.0, 2.0, 3.0, 4.0]), binder_table([0.0, 0.0, 0.0, 0.0]), top_m=2
        )
        assert got["spearman_area_vs_sensitivity"] is None
        reason = got["reasons"]["spearman_area_vs_sensitivity"]
        assert "the buried area" in reason and "tied" in reason
        assert got["top_in_interface_fraction"] == 0.0  # nothing is buried, nothing is in

    def test_a_constant_sensitivity_gives_no_rank_correlation(self):
        got = hotspot_consistency({1: 1.0, 2: 1.0, 3: 1.0}, binder_table([5.0, 50.0, 500.0]))
        assert got["spearman_area_vs_sensitivity"] is None
        assert "the sensitivity" in got["reasons"]["spearman_area_vs_sensitivity"]
        assert got["top_positions"] == [1, 2, 3]

    def test_missing_values_are_dropped_and_counted(self):
        scan = {1: 1.0, 2: None, 3: float("nan"), 4: 4.0, 5: 5.0, 6: 6.0, 8: 2.0, 7: float("inf")}
        areas = [10.0, 20.0, 30.0, None, 50.0, 60.0, 70.0]  # no row for position 8; 4 unknown
        got = hotspot_consistency(scan, binder_table(areas), top_m=2)
        assert got["n_missing_sensitivity"] == 3  # positions 2, 3 and 7
        assert got["n_unmatched_positions"] == 2  # position 4 (area None) and 8 (no line)
        assert got["n_positions"] == 3  # positions 1, 5 and 6
        assert got["top_positions"] == [6, 5]
        assert got["spearman_area_vs_sensitivity"] == pytest.approx(1.0)

    def test_nan_infinite_and_none_areas_are_unmatched_not_errors(self):
        areas = [10.0, float("nan"), float("inf"), None, 40.0, 50.0]
        got = hotspot_consistency(sensitivities_of([1, 2, 3, 4, 5, 6]), binder_table(areas))
        assert got["n_positions"] == 3 and got["n_unmatched_positions"] == 3

    def test_empty_inputs_give_none_with_reasons_not_an_error(self):
        got = hotspot_consistency({}, [])
        assert got["n_positions"] == 0
        assert got["spearman_area_vs_sensitivity"] is None
        assert got["top_in_interface_fraction"] is None
        assert set(got["reasons"]) == {"spearman_area_vs_sensitivity", "top_in_interface_fraction"}
        only_scan = hotspot_consistency(sensitivities_of([1.0, 2.0, 3.0]), [])
        assert only_scan["n_unmatched_positions"] == 3
        only_table = hotspot_consistency({}, binder_table([1.0, 2.0, 3.0]))
        assert only_table["n_positions"] == 0

    def test_nothing_is_nan_and_the_result_is_json_ready(self):
        cases = [
            hotspot_consistency({}, []),
            hotspot_consistency({1: 1.0, 2: 1.0, 3: 1.0}, binder_table([1.0, 2.0, 3.0])),
            hotspot_consistency(
                {1: float("nan"), 2: 1.0}, binder_table([float("nan"), 2.0, 3.0]), top_m=1
            ),
        ]
        for result in cases:
            json.dumps(result, allow_nan=False)


class TestHotspotInRows:
    def test_the_block_is_aggregated_like_any_other_numeric_block(self):
        scan = sensitivities_of([0.1, 0.4, 0.2, 0.9, 0.7])
        good = hotspot_consistency(scan, binder_table([0.0, 30.0, 5.0, 90.0, 60.0]))
        blank = hotspot_consistency({}, [])
        rows = [
            make_row("a1", "a", 1, hotspots=good),
            make_row("a2", "a", 2, hotspots=good),
            make_row("a3", "a", 3, hotspots=blank),
        ]
        entry = aggregate_poses(rows)["a"]["hotspots.spearman_area_vs_sensitivity"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 3)
        assert entry["mean"] == pytest.approx(good["spearman_area_vs_sensitivity"])
        fraction = aggregate_poses(rows)["a"]["hotspots.top_in_interface_fraction"]
        assert (fraction["n_ok"], fraction["n_total"]) == (2, 3)


class TestHotspotValidation:
    table = binder_table([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, None, "3"])
    def test_top_m_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="top_m"):
            hotspot_consistency({1: 1.0}, self.table, top_m=bad)

    @pytest.mark.parametrize("bad", [[1.0, 2.0], None, "scan"])
    def test_sensitivities_must_be_a_mapping(self, bad):
        with pytest.raises(ValueError, match="mapping"):
            hotspot_consistency(bad, self.table)

    @pytest.mark.parametrize("key", ["3", 2.0, True, 0, -2, None])
    def test_sensitivity_keys_must_be_one_based_integer_positions(self, key):
        with pytest.raises(ValueError, match="binder position"):
            hotspot_consistency({key: 1.0}, self.table)

    @pytest.mark.parametrize("value", ["high", True, [1.0]])
    def test_sensitivity_values_must_be_numbers_or_none(self, value):
        with pytest.raises(ValueError, match="must be a number"):
            hotspot_consistency({1: value}, self.table)

    def test_numpy_integers_and_floats_are_accepted(self):
        scan = {np.int64(1): np.float64(1.0), np.int32(2): np.float32(2.0), 3: 3}
        got = hotspot_consistency(scan, self.table)
        assert got["n_positions"] == 3

    @pytest.mark.parametrize("bad", ["rows", {"role": "binder"}, None, 3])
    def test_the_table_must_be_a_list_of_rows(self, bad):
        with pytest.raises(ValueError, match="list of row dictionaries"):
            hotspot_consistency({1: 1.0}, bad)

    def test_malformed_binder_lines_raise_with_the_line(self):
        good = binder_table([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="line 3 of the residue table is a str"):
            hotspot_consistency({1: 1.0}, [*good, "binder"])
        with pytest.raises(ValueError, match="line 3 .*role 'peptide'"):
            hotspot_consistency({1: 1.0}, [*good, {"role": "peptide"}])
        for column in ("binder_index", "buried_area_angstrom2", "in_interface"):
            line = dict(good[0])
            del line[column]
            with pytest.raises(ValueError, match=f"line 0 .*without '{column}'"):
                hotspot_consistency({1: 1.0}, [line, *good[1:]])
        with pytest.raises(ValueError, match="binder_index 'x'"):
            hotspot_consistency({1: 1.0}, [dict(good[0], binder_index="x")])
        with pytest.raises(ValueError, match="in_interface None"):
            hotspot_consistency({1: 1.0}, [dict(good[0], in_interface=None)])
        with pytest.raises(ValueError, match="buried_area_angstrom2 'big'"):
            hotspot_consistency({1: 1.0}, [dict(good[0], buried_area_angstrom2="big")])

    def test_a_binder_position_twice_in_the_table_raises(self):
        with pytest.raises(ValueError, match="binder position 1 appears twice"):
            hotspot_consistency({1: 1.0}, [*self.table, dict(self.table[0])])

    def test_the_inputs_are_not_modified(self):
        scan = sensitivities_of([1.0, None, 3.0, 2.0])
        table = binder_table([5.0, 6.0, None, 8.0])
        before = json.dumps([scan, table], sort_keys=True)
        hotspot_consistency(scan, table)
        assert json.dumps([scan, table], sort_keys=True) == before
