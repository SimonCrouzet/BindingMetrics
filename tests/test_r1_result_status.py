"""The status that a metric's own result gives a pose in ``score_pose_set``.

A registry metric can report that it failed without raising: ``compute_interaction_energy``
catches its errors and returns ``success=False`` with ``error_message``, other metrics return
an ``error`` key or ``{"skipped": True}``. The pose row has to say so in ``reasons``, the
aggregates have to count the pose as not valid for that metric, and ``on_error="raise"`` has to
stop at it as it stops at a metric that raised (issue #168).
"""

from pathlib import Path

import pytest

from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses.score import PoseScoringError, score_pose_set
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")

FIRST_POSE = "backend_a-s1"
N_POSES = 6

CALLS: list[str] = []


def _stub_error(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {"value": float("nan"), "count": 3, "error": "the solver did not start"}


def _stub_success_false(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {
        "success": False,
        "error_message": "RuntimeError: forcefield unavailable",
        "raw_interaction_energy": None,
        "num_contacts": 42,
    }


def _stub_error_and_reason(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {"count": 3, "error": "short", "reason": "the longer explanation"}


def _stub_one_mode_fails(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {
        "success": True,
        "error_message": "relaxed: minimisation did not converge",
        "raw_interaction_energy": -310.5,
        "relaxed_interaction_energy": None,
        "num_contacts": 42,
    }


def _stub_mode_fails_on_one_pose(cif_path, binder_chain=None, receptor_chain=None):
    """The relaxed mode fails on the first pose scored only."""
    CALLS.append(Path(cif_path).name)
    if len(CALLS) == 1:
        return _stub_one_mode_fails(cif_path)
    return {
        "success": True,
        "error_message": None,
        "raw_interaction_energy": -310.5,
        "relaxed_interaction_energy": -280.0,
        "num_contacts": 42,
    }


def _stub_skipped(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {"skipped": True}


def _stub_skipped_with_reason(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {"skipped": True, "reason": "the step needs a GPU"}


def _stub_clean(cif_path, binder_chain=None, receptor_chain=None):
    CALLS.append(Path(cif_path).name)
    return {
        "success": True,
        "error_message": None,
        "error": None,
        "raw_interaction_energy": -310.5,
        "num_contacts": 42,
    }


def _spec(name, function, cost_class="static"):
    return MetricSpec(
        name=name,
        import_path=f"{__name__}:{function}",
        description="stub metric of the result-status tests",
        input_type="static_structure",
        chain_mode="interface",
        path_arg="cif_path",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
        cost_class=cost_class,
    )


#: name -> (function, the sentence the row must carry in ``reasons``, a field of the block)
FAILING = {
    "stub_error": ("_stub_error", "the solver did not start", "count"),
    "stub_success_false": (
        "_stub_success_false",
        "RuntimeError: forcefield unavailable",
        "num_contacts",
    ),
}
SKIPPED = {
    "stub_skipped": ("_stub_skipped", "skipped by the metric, which gave no reason"),
    "stub_skipped_reason": ("_stub_skipped_with_reason", "the step needs a GPU"),
}
PARTIAL = {
    "stub_one_mode_fails": ("_stub_one_mode_fails", "relaxed: minimisation did not converge"),
}
MODE_FAILS_ONCE = "stub_mode_fails_on_one_pose"


@pytest.fixture
def stubs(monkeypatch):
    CALLS.clear()
    for table in (FAILING, SKIPPED, PARTIAL):
        for name, (function, *_) in table.items():
            monkeypatch.setitem(METRICS_BY_NAME, name, _spec(name, function))
    monkeypatch.setitem(
        METRICS_BY_NAME, MODE_FAILS_ONCE, _spec(MODE_FAILS_ONCE, "_stub_mode_fails_on_one_pose")
    )
    monkeypatch.setitem(
        METRICS_BY_NAME,
        "stub_error_and_reason",
        _spec("stub_error_and_reason", "_stub_error_and_reason"),
    )
    monkeypatch.setitem(METRICS_BY_NAME, "stub_clean", _spec("stub_clean", "_stub_clean"))
    monkeypatch.setitem(
        METRICS_BY_NAME, "stub_failing_md", _spec("stub_failing_md", "_stub_success_false", "md")
    )
    yield
    CALLS.clear()


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


def _score(synthetic, name, **options):
    return score_pose_set(synthetic.poses, "B", "A", metrics=[name], **options)


# ---------------------------------------------------------------------------
# the one interpreter
# ---------------------------------------------------------------------------


class TestResultStatus:
    @pytest.mark.parametrize(
        ("block", "kind", "reason"),
        [
            ({"value": 1.0}, "ok", None),
            ([1.0, 2.0], "ok", None),
            (None, "ok", None),
            ({"value": 1.0, "error": None}, "ok", None),
            ({"value": 1.0, "error": ""}, "ok", None),
            ({"success": True, "error_message": None}, "ok", None),
            ({"error": "no solver"}, "failed", "no solver"),
            ({"error": "no solver", "reason": "no solver"}, "failed", "no solver"),
            ({"success": False, "error_message": "boom"}, "failed", "boom"),
            (
                {"success": False, "error_message": None},
                "failed",
                "the metric reported success=False without a message",
            ),
            ({"success": False, "reason": "no chains"}, "failed", "no chains"),
            (
                {"success": True, "error_message": "raw: clash"},
                "caveat",
                "raw: clash",
            ),
            ({"value": 1.0, "reason": "half of it"}, "caveat", "half of it"),
            (
                {"skipped": True},
                "skipped",
                "skipped by the metric, which gave no reason",
            ),
            ({"skipped": True, "reason": "needs a GPU"}, "skipped", "needs a GPU"),
            (
                {"skipped": True, "error": "ignored"},
                "skipped",
                "skipped by the metric, which gave no reason",
            ),
            ({"skipped": False, "value": 1.0}, "ok", None),
        ],
    )
    def test_each_result_shape_has_one_reading(self, block, kind, reason):
        from binding_metrics.poses.score import _result_status

        status = _result_status(block)
        assert (status.kind, status.reason) == (kind, reason)

    @pytest.mark.parametrize(
        "block",
        [
            {"value": 1.0},
            {"error": "no solver"},
            {"success": False, "error_message": "boom"},
            {"success": False},
            {"success": True, "error_message": "raw: clash"},
            {"success": True, "error_message": None},
            {"skipped": True},
            {"skipped": True, "error": "ignored"},
            {"value": 1.0, "reason": "half of it"},
        ],
    )
    def test_the_failures_are_those_the_single_run_counts(self, block):
        from binding_metrics.cli.run import _collect_failures
        from binding_metrics.poses.score import _result_status

        assert bool(_collect_failures({"step": block})) == (_result_status(block).kind == "failed")


# ---------------------------------------------------------------------------
# the row
# ---------------------------------------------------------------------------


class TestRow:
    @pytest.mark.parametrize("name", sorted(FAILING))
    def test_a_failed_result_is_a_partial_row_with_its_reason(self, synthetic, stubs, name):
        result = _score(synthetic, name)
        expected = FAILING[name][1]
        assert [row["status"] for row in result.rows] == ["partial"] * N_POSES
        for row in result.rows:
            assert row["reasons"] == {name: expected}
            assert row["reason"] is None

    @pytest.mark.parametrize("name", sorted(FAILING))
    def test_the_block_stays_and_carries_the_reason(self, synthetic, stubs, name):
        result = _score(synthetic, name)
        for row in result.rows:
            assert row[name]["reason"] == FAILING[name][1]
            assert row[name][FAILING[name][2]] is not None

    @pytest.mark.parametrize("name", sorted(SKIPPED))
    def test_a_skipped_result_is_a_partial_row_with_its_reason(self, synthetic, stubs, name):
        result = _score(synthetic, name)
        for row in result.rows:
            assert row["status"] == "partial"
            assert row["reasons"] == {name: SKIPPED[name][1]}
            assert row[name]["skipped"] is True
            assert row[name]["reason"] == SKIPPED[name][1]

    def test_a_clean_result_stays_ok(self, synthetic, stubs):
        result = _score(synthetic, "stub_clean")
        for row in result.rows:
            assert row["status"] == "ok"
            assert row["reasons"] == {}
            assert "reason" not in row["stub_clean"]

    def test_a_block_keeps_its_own_reason_when_it_has_one(self, synthetic, stubs):
        row = _score(synthetic, "stub_error_and_reason").row(FIRST_POSE)
        assert row["stub_error_and_reason"]["reason"] == "the longer explanation"
        assert row["reasons"]["stub_error_and_reason"] == "short"

    def test_one_failed_mode_keeps_the_valid_modes_and_names_the_failed_one(self, synthetic, stubs):
        name = "stub_one_mode_fails"
        result = _score(synthetic, name)
        for row in result.rows:
            assert row["status"] == "partial"
            assert row["reasons"] == {name: PARTIAL[name][1]}
            assert row[name]["raw_interaction_energy"] == -310.5
            assert row[name]["relaxed_interaction_energy"] is None
            assert row[name]["success"] is True
            assert "reason" not in row[name]
        assert "relaxed" in result.row(FIRST_POSE)["reasons"][name]

    def test_a_mode_that_fails_on_one_pose_marks_that_pose_only(self, synthetic, stubs):
        result = _score(synthetic, MODE_FAILS_ONCE)
        statuses = {row["id"]: row["status"] for row in result.rows}
        assert statuses.pop(FIRST_POSE) == "partial"
        assert set(statuses.values()) == {"ok"}
        reason = result.row(FIRST_POSE)["reasons"][MODE_FAILS_ONCE]
        assert reason == "relaxed: minimisation did not converge"

    def test_the_failure_is_in_the_summary(self, synthetic, stubs):
        text = _score(synthetic, "stub_success_false").summary()
        assert "0 ok, 6 partial, 0 failed" in text
        assert "forcefield unavailable" in text


# ---------------------------------------------------------------------------
# the aggregates
# ---------------------------------------------------------------------------


class TestAggregates:
    @pytest.mark.parametrize("name", sorted(FAILING))
    def test_a_failed_result_counts_in_n_total_and_never_in_n_ok(self, synthetic, stubs, name):
        aggregates = _score(synthetic, name)["aggregates"]["all"]
        keys = [key for key in aggregates if key.startswith(f"{name}.")]
        assert keys, "the block has numeric fields"
        for key in keys:
            entry = aggregates[key]
            assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (0, N_POSES, False)
            assert entry["reasons"] == {FAILING[name][1]: N_POSES}

    def test_the_groups_count_the_failed_poses_too(self, synthetic, stubs):
        aggregates = _score(synthetic, "stub_success_false")["aggregates"]
        for group in ("backend_a", "backend_b"):
            entry = aggregates[group]["stub_success_false.num_contacts"]
            assert (entry["n_ok"], entry["n_total"]) == (0, 3)

    def test_one_failed_mode_leaves_the_other_modes_valid(self, synthetic, stubs):
        aggregates = _score(synthetic, MODE_FAILS_ONCE)["aggregates"]["all"]
        counts = {
            mode: (entry["n_ok"], entry["n_total"], entry["complete"])
            for mode, entry in (
                (key.split(".", 1)[1], aggregates[key])
                for key in aggregates
                if key.startswith(f"{MODE_FAILS_ONCE}.")
            )
        }
        assert counts == {
            "raw_interaction_energy": (N_POSES, N_POSES, True),
            "relaxed_interaction_energy": (N_POSES - 1, N_POSES, False),
            "num_contacts": (N_POSES, N_POSES, True),
        }
        raw = aggregates[f"{MODE_FAILS_ONCE}.raw_interaction_energy"]
        assert raw["mean"] == pytest.approx(-310.5)
        relaxed = aggregates[f"{MODE_FAILS_ONCE}.relaxed_interaction_energy"]
        assert relaxed["mean"] == pytest.approx(-280.0)

    def test_a_clean_result_counts_every_pose(self, synthetic, stubs):
        entry = _score(synthetic, "stub_clean")["aggregates"]["all"]["stub_clean.num_contacts"]
        assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (N_POSES, N_POSES, True)


# ---------------------------------------------------------------------------
# on_error="raise"
# ---------------------------------------------------------------------------


class TestOnErrorRaise:
    @pytest.mark.parametrize("name", sorted(FAILING))
    def test_a_failed_result_raises_at_the_first_pose(self, synthetic, stubs, name):
        with pytest.raises(PoseScoringError) as caught:
            _score(synthetic, name, on_error="raise")
        assert caught.value.pose_id == FIRST_POSE
        assert caught.value.stage == name
        assert FAILING[name][1] in str(caught.value)
        assert caught.value.__cause__ is None
        assert len(CALLS) == 1, "the poses after the failing one are not processed"

    def test_a_failed_result_of_a_heavy_metric_raises_as_well(self, synthetic, stubs):
        with pytest.raises(PoseScoringError) as caught:
            _score(synthetic, "stub_failing_md", on_error="raise")
        assert caught.value.stage == "stub_failing_md"
        assert "forcefield unavailable" in str(caught.value)

    @pytest.mark.parametrize("name", [*sorted(SKIPPED), *sorted(PARTIAL), "stub_clean"])
    def test_a_skipped_or_partial_result_does_not_raise(self, synthetic, stubs, name):
        result = _score(synthetic, name, on_error="raise")
        assert len(result.rows) == N_POSES

    def test_record_mode_runs_every_pose(self, synthetic, stubs):
        _score(synthetic, "stub_success_false", on_error="record")
        assert len(CALLS) == N_POSES


# ---------------------------------------------------------------------------
# the counterexample through the real energy wrapper
# ---------------------------------------------------------------------------


class TestEnergyWrapperFailure:
    """A force-field setup failure inside ``compute_interaction_energy`` itself."""

    NAME = "structure_interaction_energy"

    @pytest.fixture
    def broken_force_field(self, monkeypatch):
        pytest.importorskip("openmm")
        import binding_metrics.metrics.energy as energy

        def unavailable(*args, **kwargs):
            raise RuntimeError("forcefield unavailable")

        monkeypatch.setattr(energy, "_create_implicit_system", unavailable)

    def test_the_wrapper_reports_the_failure_in_its_result(self, synthetic, broken_force_field):
        result = score_pose_set(synthetic.poses, metrics=[self.NAME], heavy="all")
        block = result.row(FIRST_POSE)[self.NAME]
        assert block["success"] is False
        assert "forcefield unavailable" in block["error_message"]

    def test_every_pose_that_was_asked_carries_the_failure(self, synthetic, broken_force_field):
        result = score_pose_set(synthetic.poses, metrics=[self.NAME], heavy="all")
        for row in result.rows:
            assert row["status"] == "partial"
            assert "forcefield unavailable" in row["reasons"][self.NAME]

    def test_the_representatives_carry_the_failure_and_the_others_were_not_asked(
        self, synthetic, broken_force_field
    ):
        result = score_pose_set(synthetic.poses, metrics=[self.NAME])
        chosen = {
            pose_id
            for key, pose_id in result["representatives"].items()
            if key not in ("rule", "reasons", "caveats") and pose_id
        }
        assert chosen
        for row in result.rows:
            if row["id"] in chosen:
                assert row["status"] == "partial"
                assert "forcefield unavailable" in row["reasons"][self.NAME]
            else:
                assert row["status"] == "ok"
                assert self.NAME not in row["reasons"]

    def test_no_pose_counts_as_valid_for_the_energy(self, synthetic, broken_force_field):
        result = score_pose_set(synthetic.poses, metrics=[self.NAME], heavy="all")
        entries = {
            key: entry
            for key, entry in result["aggregates"]["all"].items()
            if key.startswith(f"{self.NAME}.")
        }
        assert entries
        assert {entry["n_ok"] for entry in entries.values()} == {0}
        assert {entry["n_total"] for entry in entries.values()} == {N_POSES}

    def test_on_error_raise_stops_at_the_failed_energy(self, synthetic, broken_force_field):
        with pytest.raises(PoseScoringError) as caught:
            score_pose_set(synthetic.poses, metrics=[self.NAME], heavy="all", on_error="raise")
        assert caught.value.pose_id == FIRST_POSE
        assert caught.value.stage == self.NAME
        assert "forcefield unavailable" in str(caught.value)

    def test_on_error_raise_stops_at_a_representative_with_the_default_heavy(
        self, synthetic, broken_force_field
    ):
        with pytest.raises(PoseScoringError) as caught:
            score_pose_set(synthetic.poses, metrics=[self.NAME], on_error="raise")
        assert caught.value.stage == self.NAME
