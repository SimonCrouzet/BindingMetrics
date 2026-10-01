"""``run_batch`` and ``binding-metrics-batch`` with ``--predictor``: one shared prediction store.

The workers do the steps they always did; the prediction step then runs once for all samples
in the main process. The model is a stub (``StubOpenFold``) that writes synthetic OpenFold3
output for the samples it is asked for; everything else is the real code. The samples are
copies of 1YCR with a different comment line each, so their inputs differ in content and each
one is its own prediction request.
"""

import csv
import json
import math
import sys

import pytest

from binding_metrics.cli import batch
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.prediction import RUNNERS
from tests.test_feat_c_support import EXAMPLE_1YCR, StubOpenFold, write_of3_output

NAMES = ("a", "b", "c")


@pytest.fixture
def samples(tmp_path):
    """Three inputs that differ only in a REMARK line, so each has its own request key."""
    folder = tmp_path / "in"
    folder.mkdir()
    text = EXAMPLE_1YCR.read_text(encoding="utf-8")
    paths = []
    for name in NAMES:
        path = folder / f"{name}.pdb"
        path.write_text(f"REMARK   9 sample {name}\n{text}", encoding="utf-8")
        paths.append(path)
    return paths


def batch_of(samples, out, **kwargs):
    kwargs.setdefault("skip_prep", True)
    kwargs.setdefault("skip_relax", True)
    kwargs.setdefault("metrics", {"openfold"})
    kwargs.setdefault("openfold_conda_env", None)
    return run_batch(samples, out, **kwargs)


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


class TestOneModelStart:
    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_the_model_starts_once_for_all_samples(self, tmp_path, samples, monkeypatch, n_workers):
        stub = StubOpenFold(monkeypatch)
        rows = batch_of(samples, tmp_path / "out", predictor="of3", n_workers=n_workers)

        assert stub.starts == 1
        assert stub.calls[0]["kind"] == "batched" and stub.predicted == list(NAMES)
        assert [row["sample_id"] for row in rows] == list(NAMES)
        for row in rows:
            assert row["batch_status"] == "ok"
            assert row["prediction_model"] == "of3"
            assert finite(row["prediction_avg_plddt"]) and finite(row["prediction_evobind_score"])
            assert finite(row["prediction_mean_interface_pae"])
            assert row["prediction_delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)
            assert len(row["prediction_cache_request_key"]) == 64

    def test_each_sample_finds_its_prediction_in_the_shared_store(
        self, tmp_path, samples, monkeypatch
    ):
        StubOpenFold(monkeypatch)
        rows = batch_of(samples, tmp_path / "out", predictor="of3")
        for row in rows:
            assert row["prediction_cache_hits"] == 1 and row["prediction_cache_runs"] == 0
            assert row["prediction_cache_parsed"] == 1
        keys = {row["prediction_cache_request_key"] for row in rows}
        assert len(keys) == 3
        store = tmp_path / "out" / "_predictions" / "of3"
        assert len(list(store.glob("*/*/STATUS.json"))) == 3

    def test_the_rows_carry_no_array_column_and_no_openfold_metrics(
        self, tmp_path, samples, monkeypatch
    ):
        StubOpenFold(monkeypatch)
        (row, *_) = batch_of(samples, tmp_path / "out", predictor="of3")
        assert "prediction_plddt_per_atom" not in row
        assert "prediction_binder_plddt_per_residue" not in row
        assert [key for key in row if key.startswith("openfold_")] == ["openfold_skipped"]

    def test_the_sample_reports_get_the_prediction_block(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch)
        batch_of(samples, tmp_path / "out", predictor="of3")
        report = json.loads((tmp_path / "out" / "b" / "b_results.json").read_text(encoding="utf-8"))
        assert report["openfold"] == {"skipped": True}
        assert report["prediction"]["model"] == "of3" and report["prediction"]["query_name"] == "b"
        assert len(report["prediction"]["plddt_per_atom"]) == 818
        assert report["sample_id"] == "b"

    def test_the_version_is_asked_once_for_the_batch(self, tmp_path, samples, monkeypatch):
        from binding_metrics.metrics import _openfold_run

        StubOpenFold(monkeypatch)
        asked = []

        def probe(python_cmd=None):
            asked.append(python_cmd)
            return "0.5.0"

        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", probe)
        rows = batch_of(samples, tmp_path / "out", predictor="of3", openfold_conda_env="of3env")
        assert len(asked) == 1 and asked[0][:4] == ["conda", "run", "-n", "of3env"]
        assert {row["provenance_openfold3_version"] for row in rows} == {"0.5.0"}

    def test_the_run_options_reach_the_batched_call(self, tmp_path, samples, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        batch_of(
            samples,
            tmp_path / "out",
            predictor="of3",
            openfold_seeds=[3],
            openfold_conda_env="of3env",
            on_unmappable_residue="x",
        )
        kwargs = stub.calls[0]["kwargs"]
        assert kwargs["seeds"] == (3,) and kwargs["conda_env"] == "of3env"
        assert kwargs["on_unmappable_residue"] == "x" and kwargs["mode"] == "score"


class TestStoreReuse:
    def test_a_second_batch_starts_no_model(self, tmp_path, samples, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        first = batch_of(samples, tmp_path / "out", predictor="of3")
        second = batch_of(samples, tmp_path / "out", predictor="of3")
        assert stub.starts == 1
        assert [r["prediction_cache_request_key"] for r in second] == [
            r["prediction_cache_request_key"] for r in first
        ]
        assert [r["prediction_avg_plddt"] for r in second] == [
            r["prediction_avg_plddt"] for r in first
        ]

    def test_an_explicit_cache_is_shared_between_output_directories(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        cache = tmp_path / "cache"
        batch_of(samples, tmp_path / "one", predictor="of3", prediction_cache=cache)
        batch_of(samples, tmp_path / "two", predictor="of3", prediction_cache=cache)
        assert stub.starts == 1
        assert not (tmp_path / "one" / "_predictions").exists()

    def test_only_the_missing_samples_are_predicted(self, tmp_path, samples, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        batch_of(samples[:2], tmp_path / "out", predictor="of3")
        batch_of(samples, tmp_path / "out", predictor="of3")
        assert [call["names"] for call in stub.calls] == [["a", "b"], ["c"]]

    def test_rerun_predictions_starts_the_model_again_once_for_the_batch(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        batch_of(samples, tmp_path / "out", predictor="of3")
        rows = batch_of(samples, tmp_path / "out", predictor="of3", rerun_predictions=True)
        assert stub.starts == 2 and stub.calls[1]["names"] == list(NAMES)
        assert {row["batch_status"] for row in rows} == {"ok"}


class TestFailures:
    def test_a_failed_batch_call_marks_every_sample_partial(self, tmp_path, samples, monkeypatch):
        stub = StubOpenFold(monkeypatch, error=RuntimeError("GPU out of memory"))
        rows = batch_of(samples, tmp_path / "out", predictor="of3")
        assert stub.starts == 1
        for row in rows:
            assert row["batch_status"] == "partial"
            assert row["batch_failed_steps"] == "prediction"
            assert "GPU out of memory" in row["prediction_error"]
            assert row["batch_failed_reasons"].startswith("prediction: ")

    def test_a_failed_batch_is_not_started_again_without_rerun(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch, error=RuntimeError("GPU out of memory"))
        batch_of(samples, tmp_path / "out", predictor="of3")
        rows = batch_of(samples, tmp_path / "out", predictor="of3")
        assert stub.starts == 1
        assert all("--rerun-predictions" in row["prediction_error"] for row in rows)
        stub.error = None
        rows = batch_of(samples, tmp_path / "out", predictor="of3", rerun_predictions=True)
        assert stub.starts == 2 and {row["batch_status"] for row in rows} == {"ok"}

    def test_a_model_that_cannot_start_records_every_sample(self, tmp_path, samples, monkeypatch):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        stub = StubOpenFold(monkeypatch)
        monkeypatch.setattr(OpenFold3Runner, "is_available", lambda runner: False)
        rows = batch_of(samples, tmp_path / "out", predictor="of3")
        assert stub.starts == 0
        assert all("cannot be started" in row["prediction_error"] for row in rows)
        assert {row["batch_status"] for row in rows} == {"partial"}

    def test_one_sample_that_predicted_nothing_does_not_stop_the_others(
        self, tmp_path, samples, monkeypatch
    ):
        from binding_metrics.metrics import openfold

        StubOpenFold(monkeypatch)

        def without_b(*, samples, output_dir, **kwargs):
            predictions = tmp_path / "partial" / "predictions"
            for sample in samples:
                if sample.query_name != "b":
                    write_of3_output(predictions, sample.query_name, sample.complex_structure_path)
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_batched", without_b)
        rows = {r["sample_id"]: r for r in batch_of(samples, tmp_path / "out", predictor="of3")}
        assert rows["b"]["batch_status"] == "partial"
        assert "wrote no output" in rows["b"]["prediction_error"]
        for name in ("a", "c"):
            assert rows[name]["batch_status"] == "ok" and finite(rows[name]["prediction_iptm"])

    def test_the_other_steps_of_a_sample_are_not_affected(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch, error=RuntimeError("GPU out of memory"))
        rows = batch_of(
            samples[:1], tmp_path / "out", predictor="of3", metrics={"openfold", "geometry"}
        )
        (row,) = rows
        assert row["batch_failed_steps"] == "prediction"
        assert not [key for key in row if key.startswith("geometry_error")]
        assert any(key.startswith("geometry_") for key in row)

    def test_a_worker_that_failed_is_left_out(self, tmp_path, samples, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        broken = tmp_path / "in" / "broken.pdb"
        broken.write_text("not a structure\n", encoding="utf-8")
        rows = batch_of([*samples[:1], broken], tmp_path / "out", predictor="of3")
        by_id = {row["sample_id"]: row for row in rows}
        assert by_id["broken"]["batch_status"] == "error"
        assert "prediction_model" not in by_id["broken"]
        assert by_id["a"]["batch_status"] == "ok" and stub.predicted == ["a"]


class TestParseOnly:
    @staticmethod
    def _root(tmp_path, names=NAMES, **kwargs):
        root = tmp_path / "outputs"
        for name in names:
            write_of3_output(root, name, EXAMPLE_1YCR, **kwargs)
        return root

    def test_each_sample_is_adopted_from_the_root_and_the_model_never_runs(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = batch_of(
            samples, tmp_path / "out", predictor="of3", prediction_dir=self._root(tmp_path)
        )
        assert stub.starts == 0
        for row in rows:
            assert row["batch_status"] == "ok"
            assert row["prediction_cache_adopted"] == 1 and row["prediction_cache_runs"] == 0
            assert finite(row["prediction_iptm"])

    def test_each_sample_gets_its_own_output_even_with_identical_inputs(
        self, tmp_path, monkeypatch
    ):
        StubOpenFold(monkeypatch)
        folder = tmp_path / "in"
        folder.mkdir()
        twins = []
        for name in ("x", "y"):  # the same input file twice
            path = folder / f"{name}.pdb"
            path.write_text(EXAMPLE_1YCR.read_text(encoding="utf-8"), encoding="utf-8")
            twins.append(path)
        root = tmp_path / "outputs"
        write_of3_output(root, "x", EXAMPLE_1YCR, plddt_low=50.0, plddt_high=60.0)
        write_of3_output(root, "y", EXAMPLE_1YCR, plddt_low=70.0, plddt_high=80.0)
        rows = {
            r["sample_id"]: r
            for r in batch_of(twins, tmp_path / "out", predictor="of3", prediction_dir=root)
        }
        assert rows["x"]["prediction_query_name"] == "x"
        assert rows["y"]["prediction_query_name"] == "y"
        assert rows["x"]["prediction_avg_plddt"] == pytest.approx(55.0, abs=1.0)
        assert rows["y"]["prediction_avg_plddt"] == pytest.approx(75.0, abs=1.0)

    def test_a_sample_without_output_fails_alone(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch)
        root = self._root(tmp_path, names=("a", "c"))
        rows = {
            r["sample_id"]: r
            for r in batch_of(samples, tmp_path / "out", predictor="of3", prediction_dir=root)
        }
        assert rows["b"]["batch_status"] == "partial"
        assert "no OpenFold3 output for 'b'" in rows["b"]["prediction_error"]
        assert rows["a"]["batch_status"] == rows["c"]["batch_status"] == "ok"

    def test_the_version_of_an_installation_that_ran_nothing_is_not_recorded(
        self, tmp_path, samples, monkeypatch
    ):
        from binding_metrics.metrics import _openfold_run

        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", lambda cmd=None: "0.5.0")
        StubOpenFold(monkeypatch)
        (row, *_) = batch_of(
            samples, tmp_path / "out", predictor="of3", prediction_dir=self._root(tmp_path)
        )
        assert "provenance_openfold3_version" not in row

    @pytest.mark.parametrize("model", ["af2", "boltz2", "protenix"])
    def test_a_model_without_a_runner_is_read_from_disk(
        self, tmp_path, samples, monkeypatch, model
    ):
        import importlib

        from tests.test_feat_c_support import complex_from

        writer = importlib.import_module(f"tests.predictors.synth_{model}")
        root = tmp_path / "outputs"
        root.mkdir()
        for name in NAMES:
            writer.write_prediction(root, name, complex_from(EXAMPLE_1YCR))
        rows = batch_of(samples, tmp_path / "out", predictor=model, prediction_dir=root)
        for row in rows:
            assert row["prediction_model"] == model and row["batch_status"] == "ok"
            assert finite(row["prediction_avg_plddt"])
            assert row["prediction_cache_adopted"] == 1


class TestApiChecks:
    def test_an_unknown_predictor_raises_before_any_worker_starts(self, tmp_path, samples):
        with pytest.raises(ValueError, match="Unknown predictor 'nope'"):
            batch_of(samples, tmp_path / "out", predictor="nope")
        assert not (tmp_path / "out").exists()

    def test_a_model_without_a_runner_needs_a_directory(self, tmp_path, samples, monkeypatch):
        monkeypatch.delitem(RUNNERS, "af2")  # every registered model has one: take it away
        with pytest.raises(ValueError, match="has no runner yet"):
            batch_of(samples, tmp_path / "out", predictor="af2")
        assert not (tmp_path / "out").exists()

    def test_without_a_predictor_the_batched_openfold_step_is_called(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(batch, "_run_one", lambda **kw: {"batch_status": "ok"})
        monkeypatch.setattr(batch, "_run_batched_openfold", lambda **kw: seen.append("openfold"))
        monkeypatch.setattr(
            batch, "_run_batched_prediction", lambda **kw: seen.append("prediction")
        )
        run_batch([tmp_path / "a.pdb"], tmp_path, metrics={"openfold"})
        run_batch([tmp_path / "a.pdb"], tmp_path, metrics={"openfold"}, predictor="of3")
        assert seen == ["openfold", "prediction"]

    def test_a_predictor_without_the_openfold_step_runs_no_prediction(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(batch, "_run_one", lambda **kw: {"batch_status": "ok"})
        monkeypatch.setattr(batch, "_run_batched_prediction", lambda **kw: seen.append(kw))
        run_batch([tmp_path / "a.pdb"], tmp_path, metrics={"interface"}, predictor="of3")
        assert seen == []


class TestMarkStepFailed:
    def test_an_ok_row_becomes_partial(self):
        row = {"batch_status": "ok"}
        batch._mark_step_failed(row, "prediction", "no output")
        assert row == {
            "batch_status": "partial",
            "batch_failed_steps": "prediction",
            "batch_failed_reasons": "prediction: no output",
        }

    def test_an_earlier_failure_is_kept_and_extended(self):
        row = {
            "batch_status": "partial",
            "batch_failed_steps": "energy",
            "batch_failed_reasons": "energy: boom",
        }
        batch._mark_step_failed(row, "prediction", "x" * 300)
        assert row["batch_failed_steps"] == "energy;prediction"
        assert row["batch_failed_reasons"] == "energy: boom | prediction: " + "x" * 200


class TestCommandLine:
    @staticmethod
    def _main(monkeypatch, tmp_path, samples, extra, expect_exit=None):
        out_csv = tmp_path / "out" / "metrics.csv"
        argv = [
            "binding-metrics-batch",
            "-i",
            str(samples[0].parent),
            "--output-csv",
            str(out_csv),
            "--skip-prep",
            "--skip-relax",
            "--metrics",
            "openfold",
            "--openfold-conda-env",
            "",
            *extra,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as stop:
            batch.main()
        return stop.value.code, out_csv

    def test_predictor_of3_with_two_workers_writes_the_prediction_columns(
        self, tmp_path, samples, monkeypatch, capsys
    ):
        stub = StubOpenFold(monkeypatch)
        code, out_csv = self._main(
            monkeypatch, tmp_path, samples, ["--predictor", "of3", "--workers", "2"]
        )
        assert code == 0 and stub.starts == 1
        with open(out_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["sample_id"] for row in rows] == list(NAMES)
        assert {row["batch_status"] for row in rows} == {"ok"}
        assert all(row["prediction_model"] == "of3" for row in rows)
        assert all(row["prediction_cache_request_key"] for row in rows)
        assert not [name for name in rows[0] if name.startswith("prediction_plddt")]
        assert "DONE in" in capsys.readouterr().out

    def test_the_prediction_options_reach_run_batch(self, tmp_path, samples, monkeypatch):
        seen = {}

        def fake_run_batch(paths, output_dir, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(batch, "run_batch", fake_run_batch)
        root = tmp_path / "outputs"
        root.mkdir()
        self._main(
            monkeypatch,
            tmp_path,
            samples,
            [
                "--predictor",
                "boltz2",
                "--prediction-dir",
                str(root),
                "--prediction-binder-chain",
                "P",
                "--prediction-target-chain",
                "R",
                "--prediction-cache",
                str(tmp_path / "cache"),
                "--rerun-predictions",
            ],
        )
        assert seen["predictor"] == "boltz2" and seen["prediction_dir"] == root
        assert seen["prediction_binder_chain"] == "P" and seen["prediction_target_chain"] == "R"
        assert seen["prediction_cache"] == tmp_path / "cache"
        assert seen["rerun_predictions"] is True

    def test_the_defaults_leave_the_step_as_it_was(self, tmp_path, samples, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            batch, "run_batch", lambda paths, output_dir, **kw: seen.update(kw) or []
        )
        self._main(monkeypatch, tmp_path, samples, [])
        assert seen["predictor"] is None and seen["prediction_dir"] is None
        assert seen["rerun_predictions"] is False

    @pytest.mark.parametrize("model", ["af2", "boltz2", "of3", "protenix"])
    def test_a_model_without_a_runner_fails_while_the_arguments_are_checked(
        self, tmp_path, samples, monkeypatch, capsys, model
    ):
        monkeypatch.delitem(RUNNERS, model)  # every registered model has one: take it away
        code, _ = self._main(monkeypatch, tmp_path, samples, ["--predictor", model])
        assert code == 2
        err = capsys.readouterr().err
        assert "has no runner yet" in err and "--prediction-dir" in err
        assert not (tmp_path / "out" / "metrics.csv").exists()

    @pytest.mark.parametrize(
        "option, value",
        [
            ("--prediction-dir", "somewhere"),
            ("--prediction-binder-chain", "P"),
            ("--prediction-cache", "cache"),
            ("--rerun-predictions", None),
        ],
    )
    def test_the_options_need_a_predictor(
        self, tmp_path, samples, monkeypatch, capsys, option, value
    ):
        code, _ = self._main(monkeypatch, tmp_path, samples, [option, *([value] if value else [])])
        assert code == 2
        assert f"{option} needs --predictor" in capsys.readouterr().err

    def test_a_root_that_does_not_exist_is_refused(self, tmp_path, samples, monkeypatch, capsys):
        missing = tmp_path / "nowhere"
        code, _ = self._main(
            monkeypatch, tmp_path, samples, ["--predictor", "of3", "--prediction-dir", str(missing)]
        )
        assert code == 1
        assert f"--prediction-dir is not a directory: {missing}" in capsys.readouterr().err

    def test_the_config_file_sets_the_options(self, tmp_path, samples, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            batch, "run_batch", lambda paths, output_dir, **kw: seen.update(kw) or []
        )
        root = tmp_path / "outputs"
        root.mkdir()
        config = tmp_path / "batch.toml"
        config.write_text(f'predictor = "protenix"\nprediction-dir = "{root}"\n', encoding="utf-8")
        self._main(monkeypatch, tmp_path, samples, ["--config", str(config)])
        assert seen["predictor"] == "protenix" and seen["prediction_dir"] == root
