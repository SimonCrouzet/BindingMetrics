"""``run_batch`` and ``binding-metrics-batch`` with the pre-flight check.

Each worker checks its sample first, the model step included, so a refused sample is an error
row before anything is prepared and the batch goes on. The whole-batch model step checks again
for ``--on-incompatible skip``. The model is the stub of the FEAT-C tests.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import batch
from binding_metrics.cli.batch import run_batch
from tests.test_feat_c_support import StubOpenFold, write_of3_output

DATA = Path(__file__).resolve().parent.parent / "data"
LINEAR = DATA / "example_linear_p53_1YCR.pdb"
BICYCLE = DATA / "example_bicyclic_sfti1_3P8F.cif"
ONE_CHAIN = DATA / "example_lactam_somatostatin_1XY4.cif"


@pytest.fixture
def inputs(tmp_path):
    """The linear complex, the bicyclic one and a single chain, named after their stems."""
    folder = tmp_path / "in"
    folder.mkdir()
    paths = {}
    for label, source in (("linear", LINEAR), ("bicycle", BICYCLE), ("single", ONE_CHAIN)):
        paths[label] = folder / f"{label}{source.suffix}"
        shutil.copy(source, paths[label])
    return paths


def batch_of(paths, out, **kwargs):
    kwargs.setdefault("skip_prep", True)
    kwargs.setdefault("skip_relax", True)
    kwargs.setdefault("openfold_conda_env", None)
    return run_batch(paths, out, **kwargs)


def by_sample(rows):
    return {row["sample_id"]: row for row in rows}


class TestARefusedSampleDoesNotStopTheBatch:
    def test_it_is_an_error_row_with_the_reason_and_the_others_run(self, tmp_path, inputs):
        rows = by_sample(
            batch_of([inputs["linear"], inputs["single"]], tmp_path / "out", metrics={"interface"})
        )
        good, bad = rows["linear"], rows["single"]
        assert good["batch_status"] == "ok" and good["preflight_status"] == "ok"
        assert good["preflight_reason"] == ""
        assert "interface_delta_sasa" in good
        assert bad["batch_status"] == "error"
        assert "IncompatibleInputError" in bad["batch_error"]
        assert bad["preflight_status"] == "refused"
        assert "metric 'interface': needs: no receptor chain was given" in bad["preflight_reason"]

    def test_the_refused_sample_prepared_nothing(self, tmp_path, inputs):
        batch_of([inputs["single"]], tmp_path / "out", metrics={"interface"})
        assert list((tmp_path / "out" / "single").glob("*_cleaned.cif")) == []

    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_with_workers_too(self, tmp_path, inputs, n_workers):
        rows = by_sample(
            batch_of(
                [inputs["linear"], inputs["single"]],
                tmp_path / "out",
                metrics={"interface"},
                n_workers=n_workers,
            )
        )
        assert rows["linear"]["batch_status"] == "ok"
        assert rows["single"]["batch_status"] == "error"
        assert rows["single"]["preflight_status"] == "refused"

    def test_skip_records_the_left_out_step_and_the_row_is_ok(self, tmp_path, inputs):
        rows = by_sample(
            batch_of(
                [inputs["linear"], inputs["single"]],
                tmp_path / "out",
                metrics={"interface", "geometry"},
                on_incompatible="skip",
            )
        )
        row = rows["single"]
        assert row["batch_status"] == "ok" and row["preflight_status"] == "skipped"
        assert row["interface_skipped"] is True
        assert row["geometry_shape_complementarity_skipped"] is True
        assert "ramachandran_favoured_pct" in {k.removeprefix("geometry_") for k in row}
        assert "no receptor chain was given" in row["preflight_reason"]


class TestTheWholeBatchModelStep:
    def test_a_refused_sample_never_reaches_the_legacy_openfold_call(
        self, tmp_path, inputs, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = by_sample(
            batch_of([inputs["linear"], inputs["bicycle"]], tmp_path / "out", metrics={"openfold"})
        )
        assert stub.predicted == ["linear"]
        assert rows["linear"]["batch_status"] == "ok"
        assert rows["bicycle"]["batch_status"] == "error"
        assert "predictor OpenFold3 0.5.0: closures" in rows["bicycle"]["preflight_reason"]

    def test_under_skip_the_sample_is_left_out_of_the_call_and_says_why(
        self, tmp_path, inputs, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = by_sample(
            batch_of(
                [inputs["linear"], inputs["bicycle"]],
                tmp_path / "out",
                metrics={"openfold"},
                on_incompatible="skip",
            )
        )
        assert stub.predicted == ["linear"]
        bicycle = rows["bicycle"]
        assert bicycle["batch_status"] == "ok"
        assert bicycle["openfold_skipped"] is True
        assert "a disulfide bond" in bicycle["openfold_reason"]
        assert bicycle["preflight_status"] == "skipped"

    def test_the_predictor_route_refuses_before_a_request_is_built(
        self, tmp_path, inputs, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = by_sample(
            batch_of(
                [inputs["linear"], inputs["bicycle"]],
                tmp_path / "out",
                metrics={"openfold"},
                predictor="of3",
            )
        )
        assert stub.predicted == ["linear"]
        assert rows["bicycle"]["batch_status"] == "error"
        assert rows["linear"]["prediction_model"] == "of3"

    def test_under_skip_the_prediction_columns_carry_the_reason(
        self, tmp_path, inputs, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = by_sample(
            batch_of(
                [inputs["linear"], inputs["bicycle"]],
                tmp_path / "out",
                metrics={"openfold"},
                predictor="of3",
                on_incompatible="skip",
            )
        )
        assert stub.predicted == ["linear"]
        bicycle = rows["bicycle"]
        assert bicycle["batch_status"] == "ok"
        assert bicycle["prediction_skipped"] is True
        assert "a disulfide bond" in bicycle["prediction_reason"]

    def test_the_whole_batch_check_is_not_repeated_under_warn(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(batch, "check_input", lambda *a, **k: seen.append(a))
        allowed, why = batch._model_step_allowed(
            "s", tmp_path / "x.pdb", "B", "A", ("of3", "openfold", False), "auto", "warn"
        )
        assert allowed and why == "" and seen == []

    def test_an_output_made_elsewhere_is_only_warned_about(self, tmp_path, inputs, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        given = tmp_path / "given"
        write_of3_output(given, "bicycle", inputs["bicycle"])
        rows = by_sample(
            batch_of(
                [inputs["bicycle"]],
                tmp_path / "out",
                metrics={"openfold"},
                predictor="of3",
                prediction_dir=given,
            )
        )
        row = rows["bicycle"]
        assert stub.starts == 0
        assert row["batch_status"] == "ok" and row["preflight_status"] == "warn"
        assert row["prediction_model"] == "of3"


class TestPreflightOnly:
    def test_one_row_per_sample_and_nothing_created(self, tmp_path, inputs, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        out = tmp_path / "out"
        rows = by_sample(
            run_batch(
                list(inputs.values()),
                out,
                metrics={"interface", "openfold"},
                skip_relax=True,
                preflight_only=True,
                openfold_conda_env=None,
            )
        )
        assert stub.starts == 0 and not out.exists()
        assert rows["linear"]["batch_status"] == "ok"
        assert "Runs: metrics interface" in rows["linear"]["preflight_plan"]
        assert rows["bicycle"]["batch_status"] == "error"  # the model step is refused
        assert "predictor OpenFold3 0.5.0" in rows["bicycle"]["preflight_reason"]
        assert rows["single"]["batch_status"] == "error"  # no receptor for interface

    def test_under_skip_every_sample_can_go_on(self, tmp_path, inputs):
        rows = run_batch(
            list(inputs.values()),
            tmp_path / "out",
            metrics={"interface", "openfold"},
            skip_relax=True,
            preflight_only=True,
            on_incompatible="skip",
            openfold_conda_env=None,
        )
        assert {row["batch_status"] for row in rows} == {"ok"}
        assert {row["preflight_status"] for row in rows} == {"ok", "skipped"}


class TestTheCommandLine:
    def _main(self, monkeypatch, capsys, *argv):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-batch", *argv])
        try:
            batch.main()
            code = 0
        except SystemExit as exit_info:
            code = exit_info.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_preflight_only_prints_every_plan_and_exits_1_on_a_refusal(
        self, tmp_path, inputs, monkeypatch, capsys
    ):
        code, out, _ = self._main(
            monkeypatch,
            capsys,
            "--input-dir", str(inputs["linear"].parent),
            "--output-csv", str(tmp_path / "res" / "out.csv"),
            "--metrics", "interface",
            "--skip-relax",
            "--preflight-only",
        )  # fmt: skip
        assert code == 1
        assert "--- linear: ok" in out and "--- single: refused" in out
        assert "--- bicycle: ok" in out
        assert "Pre-flight: 2 of 3 samples can run" in out
        assert not (tmp_path / "res").exists()  # no CSV, no directory

    def test_a_batch_with_skip_writes_the_csv_columns(self, tmp_path, inputs, monkeypatch, capsys):
        folder = tmp_path / "only"
        folder.mkdir()
        shutil.copy(LINEAR, folder / "linear.pdb")
        shutil.copy(ONE_CHAIN, folder / "single.cif")
        csv_path = tmp_path / "res" / "out.csv"
        code, _, _ = self._main(
            monkeypatch,
            capsys,
            "--input-dir", str(folder),
            "--output-csv", str(csv_path),
            "--metrics", "interface",
            "--skip-prep", "--skip-relax",
            "--on-incompatible", "skip",
        )  # fmt: skip
        assert code == 0
        import csv

        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
        assert rows["linear"]["preflight_status"] == "ok"
        assert rows["single"]["preflight_status"] == "skipped"
        assert "no receptor chain" in rows["single"]["preflight_reason"]
