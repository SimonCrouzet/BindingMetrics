"""The pre-flight block in the report and the CSV row, and ``check_input`` on its own."""

from __future__ import annotations

from pathlib import Path

from binding_metrics.preflight_cli import check_input, refusal_block, steps_that_run
from binding_metrics.protocols.report import _build_summary, _flatten

DATA = Path(__file__).resolve().parent.parent / "data"
LINEAR = DATA / "example_linear_p53_1YCR.pdb"
BICYCLE = DATA / "example_bicyclic_sfti1_3P8F.cif"
ONE_CHAIN = DATA / "example_lactam_somatostatin_1XY4.cif"


class TestCheckInput:
    def test_a_compatible_input_is_ok(self):
        outcome = check_input(LINEAR, "B", "A", steps=frozenset({"interface"}))
        assert outcome.block["status"] == "ok" and not outcome.refused
        assert outcome.report.compatible and outcome.skipped_steps == {}

    def test_it_returns_the_error_instead_of_raising(self):
        outcome = check_input(ONE_CHAIN, "A", None, steps=frozenset({"interface"}))
        assert outcome.refused and outcome.block["status"] == "refused"
        assert "no receptor chain was given" in outcome.block["reason"]
        assert outcome.error.report is outcome.report

    def test_no_binder_chain_means_not_checked(self):
        outcome = check_input(LINEAR, None, None, steps=frozenset({"interface"}))
        assert outcome.block["status"] == "not_checked" and not outcome.refused

    def test_an_input_that_cannot_be_read_runs_as_it_did_before(self, tmp_path, caplog):
        broken = tmp_path / "broken.pdb"
        broken.write_text("not a structure\n", encoding="utf-8")
        outcome = check_input(broken, "B", "A", steps=frozenset({"interface"}))
        assert outcome.block["status"] == "not_checked" and not outcome.refused
        assert "could not be profiled" in outcome.block["reason"]

    def test_caveats_alone_make_the_status_warn(self):
        outcome = check_input(
            BICYCLE.parent / "example_ncaa_cyclosporin_1CWA.cif",
            "C",
            "A",
            model=("of3", "openfold", False),
        )
        assert outcome.block["status"] == "warn"
        assert "cyclic: true" in outcome.block["reason"]

    def test_skip_names_the_steps_and_the_metrics_of_geometry(self):
        outcome = check_input(
            ONE_CHAIN,
            "A",
            None,
            on_incompatible="skip",
            steps=frozenset({"interface", "geometry", "electrostatics"}),
        )
        assert set(outcome.skipped_steps) == {"interface", "electrostatics"}
        assert set(outcome.skipped_geometry) == {"shape_complementarity"}
        assert outcome.block["status"] == "skipped"

    def test_geometry_is_left_out_as_a_step_only_when_all_its_metrics_are(self):
        # only shape complementarity has a limit, so the step stays and runs the other two
        outcome = check_input(
            ONE_CHAIN, "A", None, on_incompatible="skip", steps=frozenset({"geometry"})
        )
        assert "geometry" not in outcome.skipped_steps
        assert set(outcome.skipped_geometry) == {"shape_complementarity"}

    def test_the_block_is_json_ready(self):
        import json

        outcome = check_input(
            BICYCLE, "I", "A", model=("of3", "openfold", False), include_plan=True
        )
        json.dumps(outcome.block)
        assert outcome.block["plan"].startswith("Pre-flight check failed")

    def test_a_refusal_block_keeps_the_report(self):
        outcome = check_input(BICYCLE, "I", "A", model=("of3", "openfold", False))
        block = refusal_block(outcome.error)
        assert block["status"] == "refused" and block["report"]["violations"]


class TestTheReport:
    def _results(self, **block):
        return {
            "sample_id": "s",
            "input": "s.cif",
            "preflight": {
                "status": "skipped",
                "reason": "metric 'interface': needs: no receptor chain",
                "policy": "skip",
                "skipped_steps": {"interface": "no receptor chain"},
                "skipped_geometry": {"shape_complementarity": "no receptor chain"},
                "report": {"warnings": ["a caveat"], "notes": ["a note"]},
                **block,
            },
            "interface": {"skipped": True, "reason": "no receptor chain"},
            "geometry": {
                "ramachandran": {"ramachandran_favoured_pct": 90.0},
                "omega": {"omega_mean_dev": 1.0},
                "shape_complementarity": {"skipped": True, "reason": "no receptor chain"},
            },
        }

    def test_the_csv_row_has_the_status_and_the_reason_only(self):
        flat = _flatten(self._results())
        assert flat["preflight_status"] == "skipped"
        assert flat["preflight_reason"].startswith("metric 'interface'")
        assert [k for k in flat if k.startswith("preflight_")] == [
            "preflight_status",
            "preflight_reason",
        ]

    def test_a_skipped_geometry_metric_has_its_own_columns(self):
        flat = _flatten(self._results())
        assert flat["geometry_shape_complementarity_skipped"] is True
        assert flat["geometry_shape_complementarity_reason"] == "no receptor chain"
        assert "geometry_skipped" not in flat
        assert flat["geometry_ramachandran_favoured_pct"] == 90.0

    def test_no_preflight_block_no_columns(self):
        flat = _flatten({"sample_id": "s", "input": "s.cif"})
        assert not [k for k in flat if k.startswith("preflight_")]

    def test_the_summary_shows_a_short_block(self):
        text = _build_summary(self._results())
        assert "## Pre-flight check" in text
        assert "Status: `skipped` (policy: `skip`)." in text
        assert "- `interface`: no receptor chain" in text
        assert "- `geometry.shape_complementarity`: no receptor chain" in text
        assert "- a caveat" in text and "- a note" in text

    def test_the_summary_without_the_block_is_unchanged(self):
        assert "Pre-flight" not in _build_summary({"sample_id": "s", "input": "s.cif"})


def test_steps_that_run_ignores_the_steps_that_have_no_declared_limit():
    assert steps_that_run({"openfold", "dockq"}, skip_relax=True) == frozenset()
