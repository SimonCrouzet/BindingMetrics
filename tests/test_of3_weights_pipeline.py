"""``--prediction-weights`` in ``binding-metrics-run`` and ``-batch``.

The model is a stub (``tests/test_feat_c_support.StubOpenFold``); the store, the session, the
pre-flight check, the adapters and the report are the real code. What OpenFold3 does with
``--inference-ckpt-path`` rests on its source, not on a run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest

from binding_metrics.capabilities import IncompatibleInputError
from binding_metrics.cli import batch, run
from binding_metrics.cli import prediction as cli_prediction
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import openfold
from binding_metrics.predictors.weights import CACHE_FILE
from binding_metrics.protocols.report import _flatten, write_report
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
    write_of3_output,
)

NOTE_START = "custom weights: the limits declared for OpenFold3"


def checkpoint(directory: Path, name="of3-finetuned.pt", content=b"fine-tuned weights") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def pipeline(tmp_path, **kwargs):
    return run_pipeline(
        EXAMPLE_1YCR,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        peptide_chain=PEPTIDE_CHAIN,
        receptor_chain=RECEPTOR_CHAIN,
        openfold_conda_env=None,
        **kwargs,
    )


def capture_parser(module: str, monkeypatch) -> argparse.ArgumentParser:
    holder = {}

    def stop(self, *args, **kwargs):
        holder["parser"] = self
        raise SystemExit

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        importlib.import_module(module).main()
    return holder["parser"]


class TestTheOption:
    @pytest.mark.parametrize("module", ["binding_metrics.cli.run", "binding_metrics.cli.batch"])
    def test_it_takes_a_path_and_is_off_by_default(self, module, monkeypatch):
        parser = capture_parser(module, monkeypatch)
        action = next(a for a in parser._actions if "--prediction-weights" in a.option_strings)
        assert action.default is None and action.type is Path and action.metavar == "PATH"
        help_text = " ".join(action.help.split())
        for stated in ("a file for a model that takes a checkpoint file", "--ckpt", "SHA-256"):
            assert stated in help_text
        assert "Cannot be combined with --prediction-dir" in help_text

    @pytest.mark.parametrize("module", ["run", "batch"])
    def test_it_reaches_the_api(self, module, monkeypatch, tmp_path):
        captured = {}
        target = {"run": "run_pipeline", "batch": "run_batch"}[module]
        monkeypatch.setattr(
            {"run": run, "batch": batch}[module],
            target,
            lambda *a, **kwargs: captured.update(kwargs) or ({} if module == "run" else []),
        )
        weights = checkpoint(tmp_path / "w")
        input_args = ["-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o")]
        if module == "batch":
            input_args = ["-i", str(EXAMPLE_1YCR.parent), "--output-csv", str(tmp_path / "m.csv")]
        for extra, expected in (([], None), (["--prediction-weights", str(weights)], weights)):
            monkeypatch.setattr(sys, "argv", ["prog", *input_args, *extra])
            try:
                {"run": run, "batch": batch}[module].main()
            except SystemExit:
                pass
            assert captured["prediction_weights"] == expected


class TestUsageErrors:
    @staticmethod
    def _args(**kwargs):
        defaults = dict(
            predictor=None,
            prediction_dir=None,
            prediction_mode=None,
            prediction_binder_chain=None,
            prediction_target_chain=None,
            prediction_cache=None,
            rerun_predictions=False,
            prediction_weights=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _parser():
        parser = argparse.ArgumentParser(prog="prog")
        return parser

    def test_weights_with_a_prediction_dir_are_a_usage_error(self, tmp_path, capsys):
        weights = checkpoint(tmp_path / "w")
        args = self._args(predictor="of3", prediction_dir=tmp_path, prediction_weights=weights)
        with pytest.raises(SystemExit) as caught:
            cli_prediction.check_prediction_args(self._parser(), args)
        assert caught.value.code == 2
        message = " ".join(capsys.readouterr().err.split())
        assert "cannot be combined with --prediction-dir" in message
        assert "inference_ckpt_path" in message  # where the weights of an output are recorded

    def test_the_api_raises_a_value_error_for_the_same_combination(self, tmp_path):
        weights = checkpoint(tmp_path / "w")
        with pytest.raises(ValueError, match="cannot be combined with --prediction-dir"):
            pipeline(
                tmp_path / "o", predictor="of3", prediction_dir=tmp_path, prediction_weights=weights
            )
        with pytest.raises(ValueError, match="cannot be combined with --prediction-dir"):
            batch.run_batch(
                [EXAMPLE_1YCR],
                tmp_path / "b",
                predictor="of3",
                prediction_dir=tmp_path,
                prediction_weights=weights,
            )
        assert not (tmp_path / "b").exists()

    def test_a_missing_path_ends_with_exit_code_1_and_a_message(self, tmp_path, capsys):
        args = self._args(predictor="of3", prediction_weights=tmp_path / "nowhere.pt")
        with pytest.raises(SystemExit) as caught:
            cli_prediction.check_prediction_args(self._parser(), args)
        assert caught.value.code == 1
        assert "--prediction-weights" in capsys.readouterr().err

    def test_the_legacy_step_is_checked_too(self, tmp_path, capsys):
        args = self._args(predictor=None, prediction_weights=tmp_path / "nowhere.pt")
        with pytest.raises(SystemExit) as caught:
            cli_prediction.check_prediction_args(self._parser(), args)
        assert caught.value.code == 1

    def test_a_directory_is_refused_for_the_checkpoint_file_of_openfold3(self, tmp_path, capsys):
        directory = tmp_path / "d"
        checkpoint(directory)
        args = self._args(predictor="of3", prediction_weights=directory)
        with pytest.raises(SystemExit) as caught:
            cli_prediction.check_prediction_args(self._parser(), args)
        assert caught.value.code == 1
        assert "one checkpoint file" in capsys.readouterr().err

    def test_a_good_path_passes_and_none_is_free(self, tmp_path):
        weights = checkpoint(tmp_path / "w")
        cli_prediction.check_prediction_args(
            self._parser(), self._args(predictor="of3", prediction_weights=weights)
        )
        cli_prediction.check_prediction_args(self._parser(), self._args(predictor="of3"))

    def test_a_bad_path_raises_before_the_output_directory_exists(self, tmp_path):
        with pytest.raises(ValueError, match="do not exist"):
            pipeline(tmp_path / "o", prediction_weights=tmp_path / "nowhere.pt")
        assert not (tmp_path / "o").exists()


class TestPredictorRoute:
    def test_the_checkpoint_reaches_the_runner_function_and_the_block_describes_it(
        self, tmp_path, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        block = pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)["prediction"]
        assert stub.calls[0]["kwargs"]["inference_ckpt_path"] == str(weights)
        described = block["weights"]
        assert described["custom"] is True and described["name"] == weights.name
        assert described["path"] == str(weights) and described["kind"] == "file"
        assert described["size"] == len(b"fine-tuned weights") and len(described["sha256"]) == 64
        assert described["n_files"] == 1

    def test_the_provenance_records_the_weights(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        results = pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)
        assert results["provenance"]["prediction_weights"] == results["prediction"]["weights"]

    def test_a_default_run_has_no_inference_checkpoint_and_no_weights_key(
        self, tmp_path, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        results = pipeline(tmp_path / "o", predictor="of3")
        assert "inference_ckpt_path" not in stub.calls[0]["kwargs"]
        assert "weights" not in results["prediction"]
        assert "prediction_weights" not in results["provenance"]

    def test_the_key_follows_the_content_and_the_store_serves_a_copy(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        one = checkpoint(tmp_path / "one")
        blocks = {}
        for label, kwargs in {
            "default": {},
            "custom": {"prediction_weights": one},
            "copy": {"prediction_weights": checkpoint(tmp_path / "other", "renamed.ckpt")},
            "changed": {
                "prediction_weights": checkpoint(tmp_path / "third", content=b"another fine-tune")
            },
        }.items():
            blocks[label] = pipeline(
                tmp_path / label, predictor="of3", prediction_cache=store, **kwargs
            )["prediction"]
        keys = {label: block["cache"]["request_key"] for label, block in blocks.items()}
        assert keys["custom"] == keys["copy"]  # the same content under another name
        assert len({keys["default"], keys["custom"], keys["changed"]}) == 3
        assert stub.starts == 3 and blocks["copy"]["cache"]["hits"] == 1

    def test_the_hash_cache_is_kept_in_the_store_root(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)
        cache = json.loads(
            (tmp_path / "o" / "predictions" / CACHE_FILE).read_text(encoding="utf-8")
        )
        assert str(weights) in cache["files"]

    def test_the_plan_carries_the_note_and_no_violation(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        plan = pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)["preflight"]
        assert plan["status"] == "ok"
        assert any(n.startswith(NOTE_START) for n in plan["report"]["notes"])

    def test_a_runner_without_custom_weights_is_refused_before_the_model_starts(
        self, tmp_path, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)

        class NoWeights:
            supports_custom_weights = False

        monkeypatch.setattr(cli_prediction, "RUNNERS", {"of3": f"{__name__}:NoWeights"})
        weights = checkpoint(tmp_path / "w")
        with pytest.raises(IncompatibleInputError) as caught:
            pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)
        (violation,) = caught.value.violations
        assert violation.constraint == "weights" and violation.name == "of3"
        assert stub.starts == 0
        assert not (tmp_path / "o" / "predictions").exists()  # nothing was stored

    def test_under_skip_the_model_step_is_left_out(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        monkeypatch.setattr(cli_prediction, "RUNNERS", {"of3": f"{__name__}:NoWeights"})
        weights = checkpoint(tmp_path / "w")
        results = pipeline(
            tmp_path / "o", predictor="of3", prediction_weights=weights, on_incompatible="skip"
        )
        assert results["prediction"]["skipped"] is True and stub.starts == 0

    def test_preflight_only_shows_the_note_and_runs_nothing(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        results = pipeline(
            tmp_path / "o", predictor="of3", prediction_weights=weights, preflight_only=True
        )
        assert NOTE_START in results["preflight"]["plan"] and stub.starts == 0


class NoWeights:
    """A runner class that does not take custom weights (see the tests above)."""

    supports_custom_weights = False


class TestLegacyStep:
    @staticmethod
    def _stub(monkeypatch, tmp_path, *, write_config=False):
        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            predictions = Path(kwargs["output_dir"]) / "predictions"
            write_of3_output(predictions, kwargs["query_name"], kwargs["complex_structure_path"])
            if write_config:
                (predictions / "experiment_config.json").write_text(
                    json.dumps(
                        {
                            "inference_ckpt_path": "/home/me/.openfold3/of3-ob-2025-06-30-174k.pt",
                            "inference_ckpt_name": "openbind-2025-06-30-174k",
                        }
                    ),
                    encoding="utf-8",
                )
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_scoring", record)
        monkeypatch.setattr(openfold, "run_openfold_refolding", record)
        return seen

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_checkpoint_goes_to_inference_ckpt_path(self, tmp_path, monkeypatch, mode):
        seen = self._stub(monkeypatch, tmp_path)
        weights = checkpoint(tmp_path / "w")
        results = pipeline(tmp_path / "o", openfold_mode=mode, prediction_weights=weights)
        assert seen["inference_ckpt_path"] == str(weights)
        described = results["openfold"]["weights"]
        assert described["custom"] is True and described["path"] == str(weights)
        assert results["provenance"]["prediction_weights"] == described

    def test_without_weights_the_call_and_the_block_are_as_before(self, tmp_path, monkeypatch):
        seen = self._stub(monkeypatch, tmp_path)
        results = pipeline(tmp_path / "o")
        assert "inference_ckpt_path" not in seen and "weights" not in results["openfold"]
        assert "prediction_weights" not in results["provenance"]

    def test_the_default_checkpoint_that_openfold3_recorded_is_shown(self, tmp_path, monkeypatch):
        self._stub(monkeypatch, tmp_path, write_config=True)
        described = pipeline(tmp_path / "o")["openfold"]["weights"]
        assert described == {
            "custom": False,
            "name": "openbind-2025-06-30-174k",
            "path": "/home/me/.openfold3/of3-ob-2025-06-30-174k.pt",
            "sha256": None,
            "size": None,
        }

    def test_a_directory_is_refused_for_the_checkpoint_file(self, tmp_path, monkeypatch):
        seen = self._stub(monkeypatch, tmp_path)
        directory = tmp_path / "d"
        checkpoint(directory)
        with pytest.raises(ValueError, match="one checkpoint file"):
            pipeline(tmp_path / "o", prediction_weights=directory)
        assert seen == {}

    def test_a_runner_without_custom_weights_is_refused_for_the_legacy_step_too(
        self, tmp_path, monkeypatch
    ):
        seen = self._stub(monkeypatch, tmp_path)
        monkeypatch.setattr(cli_prediction, "RUNNERS", {"of3": f"{__name__}:NoWeights"})
        with pytest.raises(IncompatibleInputError):
            pipeline(tmp_path / "o", prediction_weights=checkpoint(tmp_path / "w"))
        assert seen == {}


class TestAnOutputYouMade:
    def test_the_checkpoint_the_output_names_is_shown_as_default_weights(self, tmp_path):
        output = tmp_path / "mine"
        write_of3_output(output, EXAMPLE_1YCR.stem, EXAMPLE_1YCR)
        (output / "experiment_config.json").write_text(
            json.dumps({"inference_ckpt_path": "/w/ft.pt", "inference_ckpt_name": "ft"}),
            encoding="utf-8",
        )
        results = pipeline(tmp_path / "o", predictor="of3", prediction_dir=output)
        described = results["prediction"]["weights"]
        assert described["custom"] is False and described["name"] == "ft"
        assert described["path"] == "/w/ft.pt"
        assert results["provenance"]["prediction_weights"] == described


class TestRowsAndReport:
    def test_the_csv_row_has_the_hash_and_the_path(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        results = pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)
        flat = _flatten(results)
        assert flat["prediction_weights_sha256"] == results["prediction"]["weights"]["sha256"]
        assert flat["prediction_weights_path"] == str(weights)
        assert flat["prediction_weights_custom"] is True

    def test_the_markdown_report_has_a_weights_row(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        results = pipeline(tmp_path / "o", predictor="of3", prediction_weights=weights)
        write_report(results, tmp_path / "o", "s", fmt="json", summary=True)
        summary = (tmp_path / "o" / "s_report.md").read_text(encoding="utf-8")
        digest = results["prediction"]["weights"]["sha256"][:12]
        assert f"custom: {weights.name} (sha256 {digest}" in summary

    def test_a_default_checkpoint_is_labelled_default(self):
        from binding_metrics.protocols.report import _weights_label

        assert (
            _weights_label({"custom": False, "name": "openbind-174k"}) == "default (openbind-174k)"
        )

    def test_a_run_without_weights_has_no_row(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        results = pipeline(tmp_path / "o", predictor="of3")
        write_report(results, tmp_path / "o", "s", fmt="json", summary=True)
        assert "| Weights |" not in (tmp_path / "o" / "s_report.md").read_text(encoding="utf-8")


class TestBatch:
    @staticmethod
    def _eligible(monkeypatch):
        monkeypatch.setattr(
            batch,
            "_detect_sample_chains",
            lambda *a, **k: [(0, EXAMPLE_1YCR.stem, EXAMPLE_1YCR, PEPTIDE_CHAIN, RECEPTOR_CHAIN)],
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))

    def test_the_batched_prediction_hashes_once_and_fills_the_columns(self, tmp_path, monkeypatch):
        self._eligible(monkeypatch)
        stub = StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_prediction(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            predictor="of3",
            peptide_chain=None,
            receptor_chain=None,
            prediction_weights=weights,
        )
        row = rows[0]
        assert stub.calls[0]["kwargs"]["inference_ckpt_path"] == str(weights)
        assert row["prediction_weights_path"] == str(weights)
        assert row["prediction_weights_sha256"] == row["provenance_prediction_weights_sha256"]
        assert row["provenance_prediction_weights_path"] == str(weights)
        assert not any(isinstance(value, dict) for value in row.values())

    def test_the_batched_openfold_step_passes_the_checkpoint(self, tmp_path, monkeypatch):
        self._eligible(monkeypatch)
        seen = {}

        def record(*, samples, output_dir, **kwargs):
            seen.update(kwargs)
            predictions = Path(output_dir) / "predictions"
            for sample in samples:
                write_of3_output(predictions, sample.query_name, sample.complex_structure_path)
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_batched", record)
        weights = checkpoint(tmp_path / "w")
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
            prediction_weights=weights,
        )
        assert seen["inference_ckpt_path"] == str(weights)
        assert rows[0]["openfold_weights_path"] == str(weights)
        assert rows[0]["provenance_prediction_weights_sha256"] == rows[0]["openfold_weights_sha256"]
        assert (tmp_path / "_predictions" / CACHE_FILE).is_file()

    def test_without_weights_the_batched_calls_are_as_before(self, tmp_path, monkeypatch):
        self._eligible(monkeypatch)
        stub = StubOpenFold(monkeypatch)
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_prediction(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            predictor="of3",
            peptide_chain=None,
            receptor_chain=None,
        )
        assert "inference_ckpt_path" not in stub.calls[0]["kwargs"]
        assert not any("weights" in key for key in rows[0])

    def test_the_provenance_helper_flattens_a_dict_and_keeps_a_scalar(self):
        row = {}
        batch._add_provenance(row, "prediction_weights", {"path": "/w", "sha256": "ab"})
        batch._add_provenance(row, "openfold3_checkpoint", "ft")
        assert row == {
            "provenance_prediction_weights_path": "/w",
            "provenance_prediction_weights_sha256": "ab",
            "provenance_openfold3_checkpoint": "ft",
        }

    def test_preflight_only_rows_show_the_note(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        weights = checkpoint(tmp_path / "w")
        rows = batch.run_batch(
            [EXAMPLE_1YCR],
            tmp_path / "out",
            metrics={"openfold"},
            predictor="of3",
            prediction_weights=weights,
            preflight_only=True,
        )
        assert NOTE_START in rows[0]["preflight_plan"]
