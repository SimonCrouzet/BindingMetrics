"""``--prediction-mode`` on ``binding-metrics-run`` and ``-batch``, and the mode in the results.

The mode is checked in the pre-flight check against what the model supports, before anything
runs. Without the option nothing changes: OpenFold3 run from here takes ``--openfold-mode``, and
an output read with ``--prediction-dir`` has a mode that is not known and not checked.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pytest

from binding_metrics.capabilities import MODES, IncompatibleInputError
from binding_metrics.cli import batch, prediction, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import run_pipeline
from tests.test_feat_c_support import StubOpenFold, complex_from, write_of3_output
from tests.test_pre_cli_run import LINEAR, Tripwires

STEM = LINEAR.stem


def pipeline(path, output_dir, **kwargs):
    kwargs.setdefault("skip_prep", True)
    kwargs.setdefault("skip_relax", True)
    kwargs.setdefault("metrics", frozenset({"openfold"}))
    kwargs.setdefault("openfold_conda_env", None)
    kwargs.setdefault("peptide_chain", "B" if path == LINEAR else None)
    kwargs.setdefault("receptor_chain", "A" if path == LINEAR else None)
    return run_pipeline(path, output_dir, **kwargs)


class TestTheOption:
    def _parser(self):
        parser = argparse.ArgumentParser()
        prediction.add_prediction_args(parser)
        return parser

    def test_the_choices_are_the_vocabulary_and_the_default_is_none(self):
        parser = self._parser()
        assert parser.parse_args([]).prediction_mode is None
        for mode in MODES:
            assert parser.parse_args(["--prediction-mode", mode]).prediction_mode == mode
        with pytest.raises(SystemExit):
            parser.parse_args(["--prediction-mode", "dock"])

    def test_the_help_says_what_each_mode_is_and_what_the_default_is(self):
        text = " ".join(self._parser().format_help().split())
        for part in (
            "predict (sequences only)",
            "refold (receptor templated",
            "score (every chain",
        ):
            assert part in text
        assert "score-lock (score, with the pose pinned to the input)" in text
        assert "for --predictor of3 the value of --openfold-mode" in text
        # the default of a run from here is the mode of its runner
        assert "score for boltz2 and predict for af2 and protenix" in text

    def test_it_needs_a_predictor(self):
        parser = self._parser()
        args = parser.parse_args(["--prediction-mode", "score-lock"])
        with pytest.raises(SystemExit):
            prediction.check_prediction_args(parser, args)

    def test_predict_cannot_be_run_from_here_but_can_be_read(self, tmp_path):
        parser = self._parser()
        run_from_here = parser.parse_args(["--predictor", "of3", "--prediction-mode", "predict"])
        with pytest.raises(SystemExit):
            prediction.check_prediction_args(parser, run_from_here)
        read = parser.parse_args(
            [
                "--predictor",
                "of3",
                "--prediction-mode",
                "predict",
                "--prediction-dir",
                str(tmp_path),
            ]
        )
        prediction.check_prediction_args(parser, read)  # no error

    def test_the_python_api_checks_the_same(self):
        with pytest.raises(ValueError, match="prediction_mode must be one of"):
            prediction.check_predictor("of3", None, "dock")
        with pytest.raises(ValueError, match="needs a predictor"):
            prediction.check_predictor(None, None, "score-lock")
        with pytest.raises(ValueError, match="cannot be run from here"):
            prediction.check_predictor("of3", None, "predict")
        prediction.check_predictor("boltz2", Path("d"), "score-lock")

    def test_the_effective_mode(self):
        mode = prediction.effective_prediction_mode
        assert mode(None, None, None, "refold") == "refold"  # the legacy step
        assert mode("of3", None, None, "score") == "score"  # a run from here
        assert mode("of3", None, "refold", "score") == "refold"  # the option wins
        assert mode("boltz2", Path("d"), None, "score") is None  # made elsewhere: not known
        assert mode("boltz2", Path("d"), "score-lock", "score") == "score-lock"
        assert mode("of3", Path("d"), None, "score") is None


class TestThePreflightRefusesALock:
    def test_score_lock_for_openfold3_run_from_here_is_refused_before_anything_runs(
        self, tmp_path, monkeypatch
    ):
        tripwires = Tripwires(monkeypatch)
        out = tmp_path / "out"
        with pytest.raises(IncompatibleInputError) as caught:
            pipeline(LINEAR, out, predictor="of3", prediction_mode="score-lock")
        message = str(caught.value)
        assert tripwires.calls == [] and not out.exists()
        assert "predictor OpenFold3 0.5.0: modes" in message
        assert "mode 'score-lock' was requested" in message
        assert "supported modes: predict, refold, score" in message
        assert (
            "Mode: score-lock (score, with the pose of the chains pinned to the input)" in message
        )
        # the alternatives come from the registry; Boltz-2 can be run from here, so it is not marked
        assert "Boltz-2 (boltz2)" in message and "[no runner here" not in message
        assert "OpenFold3 (of3)" not in message.split("fix:")[1]

    def test_the_same_through_the_openfold_step_is_not_reachable_by_the_option(self):
        parser = argparse.ArgumentParser()
        prediction.add_prediction_args(parser)
        with pytest.raises(SystemExit):  # --prediction-mode needs --predictor
            prediction.check_prediction_args(
                parser, parser.parse_args(["--prediction-mode", "score-lock"])
            )

    def test_the_plan_names_the_mode(self, tmp_path):
        result = pipeline(
            LINEAR,
            tmp_path / "out",
            predictor="of3",
            prediction_mode="score-lock",
            preflight_only=True,
        )
        block = result["preflight"]
        assert block["status"] == "refused" and block["mode"] == "score-lock"
        assert "Mode: score-lock" in block["plan"]


class TestTheDefaultsAreUnchanged:
    def test_the_mode_of_a_run_is_the_openfold_mode(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        results = pipeline(LINEAR, tmp_path / "score", predictor="of3")
        assert stub.calls[0]["kind"] == "scoring"
        assert results["prediction"]["mode"] == "score"
        assert results["preflight"]["mode"] == "score"
        refold = pipeline(LINEAR, tmp_path / "refold", predictor="of3", openfold_mode="refold")
        assert stub.calls[1]["kind"] == "refolding"
        assert refold["prediction"]["mode"] == refold["preflight"]["mode"] == "refold"

    def test_the_option_wins_over_the_openfold_mode(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        results = pipeline(
            LINEAR,
            tmp_path / "out",
            predictor="of3",
            openfold_mode="score",
            prediction_mode="refold",
        )
        assert stub.calls[0]["kind"] == "refolding"
        assert results["prediction"]["mode"] == "refold"

    def test_the_legacy_step_records_the_mode_in_the_preflight_block(self, tmp_path, monkeypatch):
        from tests.test_feat_c_support import EXAMPLE_1YCR  # noqa: F401

        StubOpenFold(monkeypatch)
        results = pipeline(LINEAR, tmp_path / "out", openfold_mode="refold", preflight_only=True)
        assert results["preflight"]["mode"] == "refold"

    def test_no_metric_that_needs_a_model_means_no_mode(self, tmp_path):
        results = pipeline(
            LINEAR, tmp_path / "out", metrics=frozenset({"interface"}), preflight_only=True
        )
        assert results["preflight"]["mode"] is None


class TestAnOutputMadeElsewhere:
    @pytest.fixture
    def boltz2_output(self, tmp_path):
        writer = importlib.import_module("tests.predictors.synth_boltz2")
        root = tmp_path / "boltz_out"
        root.mkdir()
        writer.write_prediction(root, STEM, complex_from(LINEAR))
        return root

    def test_score_lock_is_accepted_for_boltz2_with_a_warning_and_recorded(
        self, tmp_path, boltz2_output
    ):
        results = pipeline(
            LINEAR,
            tmp_path / "out",
            predictor="boltz2",
            prediction_dir=boltz2_output,
            prediction_mode="score-lock",
        )
        block = results["preflight"]
        assert block["status"] == "warn" and block["mode"] == "score-lock"
        assert "force: true" in block["reason"] and "measured once" in block["reason"]
        assert results["prediction"]["mode"] == "score-lock"
        assert results["prediction"]["model"] == "boltz2" and "error" not in results["prediction"]

    def test_without_the_option_the_mode_is_not_known_and_not_checked(
        self, tmp_path, boltz2_output
    ):
        results = pipeline(
            LINEAR, tmp_path / "out", predictor="boltz2", prediction_dir=boltz2_output
        )
        assert results["prediction"]["mode"] is None
        assert results["preflight"]["mode"] is None
        assert results["preflight"]["status"] == "ok"

    def test_a_lock_stated_for_openfold3_output_only_warns(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        out = write_of3_output(tmp_path / "out", STEM, LINEAR)
        results = pipeline(
            LINEAR,
            tmp_path / "run",
            predictor="of3",
            prediction_dir=out,
            prediction_mode="score-lock",
        )
        block = results["preflight"]
        assert stub.starts == 0
        assert block["status"] == "warn" and "mode 'score-lock' was requested" in block["reason"]
        assert block["report"]["predictor_policy"] == "warn"
        assert results["prediction"]["mode"] == "score-lock"

    def test_the_mode_is_part_of_the_request_key_of_an_adopted_output(self, tmp_path):
        base = dict(binder_chain="B", receptor_chain="A", adopt=True)
        plain = prediction.make_request("boltz2", "s", LINEAR, **base)
        locked = prediction.make_request(
            "boltz2", "s", LINEAR, prediction_mode="score-lock", **base
        )
        assert plain.mode == "predict" and locked.mode == "score-lock"
        assert plain.key() != locked.key()


class TestInABatch:
    @pytest.fixture
    def samples(self, tmp_path):
        import shutil

        folder = tmp_path / "in"
        folder.mkdir()
        paths = []
        for name in ("a", "b"):
            path = folder / f"{name}.pdb"
            shutil.copy(LINEAR, path)
            paths.append(path)
        return paths

    def batch_of(self, samples, out, **kwargs):
        kwargs.setdefault("skip_prep", True)
        kwargs.setdefault("skip_relax", True)
        kwargs.setdefault("metrics", {"openfold"})
        kwargs.setdefault("openfold_conda_env", None)
        return run_batch(samples, out, **kwargs)

    def test_a_lock_for_openfold3_refuses_every_sample_and_the_model_never_starts(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = self.batch_of(
            samples, tmp_path / "out", predictor="of3", prediction_mode="score-lock"
        )
        assert stub.starts == 0
        for row in rows:
            assert row["batch_status"] == "error" and row["preflight_status"] == "refused"
            assert "mode 'score-lock' was requested" in row["preflight_reason"]

    def test_under_skip_the_model_step_is_left_out_with_its_mode(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = self.batch_of(
            samples,
            tmp_path / "out",
            predictor="of3",
            prediction_mode="score-lock",
            on_incompatible="skip",
        )
        assert stub.starts == 0
        for row in rows:
            assert row["batch_status"] == "ok"
            assert row["prediction_skipped"] is True and row["prediction_mode"] == "score-lock"
            assert "mode 'score-lock' was requested" in row["prediction_reason"]

    def test_the_mode_is_a_column_of_the_row(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch)
        rows = self.batch_of(samples, tmp_path / "out", predictor="of3", prediction_mode="refold")
        assert {row["prediction_mode"] for row in rows} == {"refold"}
        assert {row["batch_status"] for row in rows} == {"ok"}

    def test_the_default_mode_of_the_batch_is_the_openfold_mode(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = self.batch_of(samples, tmp_path / "out", predictor="of3")
        assert stub.starts >= 1 and all(c["kind"] != "refolding" for c in stub.calls)
        assert {row["prediction_mode"] for row in rows} == {"score"}

    def test_preflight_only_names_the_mode_and_refuses_a_lock(self, tmp_path, samples):
        rows = self.batch_of(
            samples,
            tmp_path / "out",
            predictor="of3",
            prediction_mode="score-lock",
            preflight_only=True,
        )
        assert {row["batch_status"] for row in rows} == {"error"}
        assert all("Mode: score-lock" in row["preflight_plan"] for row in rows)

    def test_the_command_line_prints_the_plan_with_the_alternatives(
        self, tmp_path, samples, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-batch",
                "--input-dir", str(samples[0].parent),
                "--output-csv", str(tmp_path / "res" / "out.csv"),
                "--metrics", "openfold",
                "--predictor", "of3",
                "--prediction-mode", "score-lock",
                "--preflight-only",
            ],
        )  # fmt: skip
        with pytest.raises(SystemExit) as exit_info:
            batch.main()
        out = capsys.readouterr().out
        assert exit_info.value.code == 1
        assert "mode 'score-lock' was requested" in out and "Boltz-2 (boltz2)" in out
        assert not (tmp_path / "res").exists()


class TestTheRunCommandLine:
    def test_the_preflight_only_plan_of_a_lock(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-run",
                "--input", str(LINEAR),
                "--output-dir", str(tmp_path / "out"),
                "--metrics", "openfold",
                "--predictor", "of3",
                "--prediction-mode", "score-lock",
                "--preflight-only",
            ],
        )  # fmt: skip
        with pytest.raises(SystemExit) as exit_info:
            run.main()
        out = capsys.readouterr().out
        assert exit_info.value.code == 1
        assert "Pre-flight check failed" in out
        assert "mode 'score-lock' was requested" in out
        assert "Boltz-2 (boltz2)" in out and "[no runner here" not in out
        assert not (tmp_path / "out").exists()
