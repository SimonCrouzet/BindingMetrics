"""``--designed-with`` and ``--design-policy`` on ``binding-metrics-run`` and ``-batch``.

Everything but the model is the real code: the pre-flight check, the store, the adapters and the
EvoBind metrics run on a synthetic output of the model (``--prediction-dir``) or on the stub of the
OpenFold3 tests. A design pipeline that ran the model of the run has its adversarial check left out
with a loud warning and the rest of the run goes on; any other model runs the check as before.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import sys

import pytest

from binding_metrics.capabilities import POLICIES, IncompatibleInputError
from binding_metrics.cli import batch, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import run_pipeline
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
    complex_from,
)

STEM = EXAMPLE_1YCR.stem
SKIPPED = "the design was made with"


def synthetic_output(tmp_path, model):
    """A synthetic output of ``model`` for 1YCR, in the directory ``--prediction-dir`` reads."""
    out = tmp_path / f"{model}_out"
    out.mkdir()
    writer = importlib.import_module(f"tests.predictors.synth_{model}")
    writer.write_prediction(out, STEM, complex_from(EXAMPLE_1YCR))
    return out


def pipeline(tmp_path, model, **kwargs):
    if "prediction_dir" not in kwargs:
        kwargs["prediction_dir"] = synthetic_output(tmp_path, model)
    return run_pipeline(
        EXAMPLE_1YCR,
        tmp_path / "run",
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        peptide_chain=PEPTIDE_CHAIN,
        receptor_chain=RECEPTOR_CHAIN,
        predictor=model,
        **kwargs,
    )


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


class TestOneRun:
    @pytest.mark.parametrize(
        "source, model",
        [("boltzgen", "boltz2"), ("BindCraft", "af2"), ("colabdesign", "af2")],
    )
    def test_the_model_that_the_design_ran_has_its_adversarial_check_left_out(
        self, tmp_path, caplog, source, model
    ):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            results = pipeline(tmp_path, model, designed_with=source)
        block = results["prediction"]
        assert block["model"] == model and "error" not in block
        assert block["independent_of_design"] is False
        assert SKIPPED in block["adversarial_skipped"] and "itself" in block["adversarial_skipped"]
        assert not {"delta_com_angstrom", "evobind_adversarial_score"} & set(block)
        assert "adversarial_error" not in block
        # the rest of the step is intact
        assert finite(block["avg_plddt"]) and finite(block["evobind_score"])
        assert finite(block["mean_interface_pae"])
        # the pre-flight block says what was left out and why, and the warning was loud
        decision = results["preflight"]
        assert decision["status"] == "skipped" and decision["policy"] == "error"
        design = decision["report"]["design"]
        assert design["source"] == source.lower() and design["not_independent"] == [model]
        assert [(c["check"], c["action"]) for c in design["checks"]] == [
            ("evobind_adversarial", "skipped")
        ]
        assert [m["name"] for m in decision["report"]["metrics_skipped"]] == ["evobind_adversarial"]
        assert any(
            r.levelno == logging.WARNING and "evobind_adversarial is skipped because" in r.message
            for r in caplog.records
        )

    @pytest.mark.parametrize("source, model", [("boltzgen", "af2"), ("bindcraft", "boltz2")])
    def test_another_model_runs_the_check_and_is_labelled_independent(
        self, tmp_path, source, model
    ):
        block = pipeline(tmp_path, model, designed_with=source)["prediction"]
        assert block["independent_of_design"] is True
        assert "adversarial_skipped" not in block
        if model == "af2":  # Boltz-2 numbers the residues from 1; see test_feat_c_run_prediction
            assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_without_a_source_the_block_is_what_it_always_was(self, tmp_path):
        results = pipeline(tmp_path, "af2")
        block = results["prediction"]
        assert "independent_of_design" not in block and "adversarial_skipped" not in block
        assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
        assert "design" not in results["preflight"]["report"]
        assert results["preflight"]["status"] in ("ok", "warn")

    def test_warn_runs_the_check_and_marks_the_result_not_independent(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            block = pipeline(tmp_path, "af2", designed_with="bindcraft", design_policy="warn")[
                "prediction"
            ]
        assert block["independent_of_design"] is False
        assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
        assert "adversarial_skipped" not in block
        assert any("evobind_adversarial runs on a" in r.message for r in caplog.records)

    def test_error_refuses_before_anything_is_written(self, tmp_path):
        out = synthetic_output(tmp_path, "af2")
        with pytest.raises(IncompatibleInputError) as caught:
            pipeline(
                tmp_path,
                "af2",
                prediction_dir=out,
                designed_with="bindcraft",
                design_policy="error",
            )
        assert not (tmp_path / "run").exists()
        assert [v.kind for v in caught.value.violations] == ["design"]
        assert "design_origin" in str(caught.value)

    def test_on_incompatible_does_not_change_the_rule(self, tmp_path):
        for policy in POLICIES:
            sub = tmp_path / policy
            sub.mkdir()
            block = pipeline(sub, "af2", designed_with="bindcraft", on_incompatible=policy)[
                "prediction"
            ]
            assert "adversarial_skipped" in block, policy

    def test_the_openfold3_step_without_a_predictor_follows_the_same_rule(
        self, tmp_path, monkeypatch
    ):
        StubOpenFold(monkeypatch)

        def step(designed_with):
            return run_pipeline(
                EXAMPLE_1YCR,
                tmp_path / str(designed_with),
                skip_prep=True,
                skip_relax=True,
                metrics=frozenset({"openfold"}),
                peptide_chain=PEPTIDE_CHAIN,
                receptor_chain=RECEPTOR_CHAIN,
                openfold_conda_env=None,
                designed_with=designed_with,
            )["openfold"]

        circular = step(("of3",))
        assert circular["independent_of_design"] is False
        assert "adversarial_skipped" in circular and "delta_com_angstrom" not in circular
        independent = step("boltzgen")
        assert independent["independent_of_design"] is True
        assert independent["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)

    def test_an_unknown_source_warns_and_runs_everything(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            block = pipeline(tmp_path, "af2", designed_with=("a_model_nobody_registered",))[
                "prediction"
            ]
        assert "adversarial_skipped" not in block and "independent_of_design" in block
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            sub = tmp_path / "second"
            sub.mkdir()
            block = pipeline(sub, "af2", designed_with="rfdiffusion")["prediction"]
        assert "adversarial_skipped" not in block and "independent_of_design" not in block
        assert any("'rfdiffusion' is not one of" in r.message for r in caplog.records)

    def test_the_options_are_checked_before_anything_runs(self, tmp_path):
        out = synthetic_output(tmp_path, "af2")
        with pytest.raises(ValueError, match="design_policy must be one of"):
            pipeline(
                tmp_path, "af2", prediction_dir=out, designed_with="boltzgen", design_policy="x"
            )
        with pytest.raises(TypeError, match="design_source"):
            pipeline(tmp_path, "af2", prediction_dir=out, designed_with=5)
        assert not (tmp_path / "run").exists()

    def test_the_plan_of_preflight_only_shows_it(self, tmp_path):
        result = pipeline(tmp_path, "boltz2", designed_with="boltzgen", preflight_only=True)
        plan = result["preflight"]["plan"]
        assert "Design origin: boltzgen, which ran Boltz-2 itself" in plan
        assert "Left out: metrics evobind_adversarial" in plan
        assert "design: skip" in plan
        assert not (tmp_path / "run").exists()


class TestCommandLine:
    @staticmethod
    def parsed(monkeypatch, tmp_path, extra, module=run):
        captured = {}

        def fake(*args, **kwargs):
            captured.update(kwargs)
            return {"sample_id": "x", "provenance": {}, "rows": []}

        monkeypatch.setattr(module, "run_pipeline" if module is run else "run_batch", fake)
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o"), *extra]
            if module is run
            else ["prog", "-i", str(tmp_path), "-o", str(tmp_path / "o"), *extra],
        )
        module.main()
        return captured

    def test_the_defaults_decide_nothing(self, monkeypatch, tmp_path):
        parsed = self.parsed(monkeypatch, tmp_path, [])
        assert parsed["designed_with"] is None and parsed["design_policy"] == "skip"

    @pytest.mark.parametrize("name", ["boltzgen", "BoltzGen", "BOLTZGEN"])
    def test_the_name_is_read_in_any_case(self, monkeypatch, tmp_path, name):
        parsed = self.parsed(
            monkeypatch, tmp_path, ["--designed-with", name, "--design-policy", "warn"]
        )
        assert parsed["designed_with"] == "boltzgen" and parsed["design_policy"] == "warn"

    def test_the_choices_are_the_pipelines_of_the_table_and_the_three_policies(
        self, monkeypatch, tmp_path, capsys
    ):
        for extra, shown in (
            (["--designed-with", "rfdiffusion"], "boltzgen"),
            (["--design-policy", "ignore"], "skip"),
        ):
            with pytest.raises(SystemExit) as stop:
                self.parsed(monkeypatch, tmp_path, extra)
            assert stop.value.code == 2 and shown in capsys.readouterr().err

    def test_the_config_file_sets_them(self, monkeypatch, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('designed-with = "ColabDesign"\ndesign-policy = "error"\n', "utf-8")
        parsed = self.parsed(monkeypatch, tmp_path, ["--config", str(config)])
        assert parsed["designed_with"] == "colabdesign" and parsed["design_policy"] == "error"

    def test_a_real_run_writes_the_decision_to_the_results(self, tmp_path, monkeypatch):
        out = synthetic_output(tmp_path, "af2")
        argv = [
            "binding-metrics-run",
            "-i", str(EXAMPLE_1YCR),
            "-o", str(tmp_path / "results"),
            "--skip-prep", "--skip-relax",
            "--metrics", "openfold",
            "--peptide-chain", PEPTIDE_CHAIN,
            "--receptor-chain", RECEPTOR_CHAIN,
            "--predictor", "af2",
            "--prediction-dir", str(out),
            "--designed-with", "BindCraft",
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        run.main()
        written = json.loads((tmp_path / "results" / f"{STEM}_results.json").read_text("utf-8"))
        block = written["prediction"]
        assert block["independent_of_design"] is False and "adversarial_skipped" in block
        assert written["preflight"]["report"]["design"]["source"] == "bindcraft"

    def test_preflight_only_prints_the_plan_with_the_design_and_exits_with_the_policy(
        self, tmp_path, monkeypatch, capsys
    ):
        out = synthetic_output(tmp_path, "af2")
        base = [
            "binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o"),
            "--metrics", "openfold", "--predictor", "af2", "--prediction-dir", str(out),
            "--designed-with", "bindcraft", "--preflight-only",
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", base)
        with pytest.raises(SystemExit) as stop:
            run.main()
        shown = capsys.readouterr().out
        assert stop.value.code == 0
        assert "Design origin: bindcraft, which ran AlphaFold2 / ColabFold itself" in shown
        assert "Left out: metrics evobind_adversarial" in shown
        monkeypatch.setattr(sys, "argv", [*base, "--design-policy", "error"])
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 1
        assert "Pre-flight check failed" in capsys.readouterr().out
        assert not (tmp_path / "o").exists()


@pytest.fixture
def samples(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    text = EXAMPLE_1YCR.read_text(encoding="utf-8")
    paths = []
    for name in ("a", "b"):
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


class TestBatch:
    def test_the_prediction_step_leaves_the_check_out_for_every_sample(
        self, tmp_path, samples, monkeypatch, caplog
    ):
        StubOpenFold(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="binding_metrics.cli.batch"):
            rows = batch_of(
                samples, tmp_path / "out", predictor="of3", designed_with=("of3",), n_workers=1
            )
        for row in rows:
            assert row["batch_status"] == "ok"
            assert row["prediction_independent_of_design"] is False
            assert "itself" in row["prediction_adversarial_skipped"]
            assert "prediction_delta_com_angstrom" not in row
            assert finite(row["prediction_evobind_score"])
        report = json.loads((tmp_path / "out" / "a" / "a_results.json").read_text("utf-8"))
        assert "adversarial_skipped" in report["prediction"]
        lines = [r for r in caplog.records if "evobind_adversarial is skipped" in r.message]
        assert len([r for r in lines if r.name == "binding_metrics.cli.batch"]) == 1
        assert len(lines) == 3  # once for the batch, then once for each sample's own pre-flight

    def test_an_independent_model_runs_it(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch)
        rows = batch_of(samples, tmp_path / "out", predictor="of3", designed_with="boltzgen")
        for row in rows:
            assert row["prediction_independent_of_design"] is True
            assert row["prediction_delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)
            assert "prediction_adversarial_skipped" not in row

    def test_the_batched_openfold_step_follows_the_rule(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch)
        rows = batch_of(samples, tmp_path / "out", designed_with=("of3",))
        for row in rows:
            assert row["openfold_independent_of_design"] is False
            assert "itself" in row["openfold_adversarial_skipped"]
            assert "openfold_delta_com_angstrom" not in row

    def test_error_makes_every_sample_an_error_row_and_runs_no_model(
        self, tmp_path, samples, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        rows = batch_of(
            samples,
            tmp_path / "out",
            predictor="of3",
            designed_with=("of3",),
            design_policy="error",
        )
        assert [row["batch_status"] for row in rows] == ["error", "error"]
        assert all(row["preflight_status"] == "refused" for row in rows)
        assert stub.starts == 0

    def test_without_a_source_the_rows_are_unchanged(self, tmp_path, samples, monkeypatch):
        StubOpenFold(monkeypatch)
        rows = batch_of(samples, tmp_path / "out", predictor="of3")
        for row in rows:
            assert not [
                k for k in row if "independent_of_design" in k or "adversarial_skipped" in k
            ]
            assert row["prediction_delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)

    def test_preflight_only_rows_show_the_decision(self, tmp_path, samples):
        rows = batch_of(
            samples,
            tmp_path / "out",
            predictor="boltz2",
            prediction_dir=tmp_path,
            designed_with="boltzgen",
            preflight_only=True,
        )
        for row in rows:
            assert "Left out: metrics evobind_adversarial" in row["preflight_plan"]
        assert not (tmp_path / "out").exists()

    def test_the_options_are_checked_first(self, tmp_path, samples):
        with pytest.raises(ValueError, match="design_policy must be one of"):
            batch_of(samples, tmp_path / "out", design_policy="ignore")
        with pytest.raises(TypeError):
            batch_of(samples, tmp_path / "out", designed_with=3)
        assert not (tmp_path / "out").exists()

    def test_the_command_line_passes_the_options(self, monkeypatch, tmp_path):
        folder = tmp_path / "designs"
        folder.mkdir()
        (folder / "one.pdb").write_text(EXAMPLE_1YCR.read_text(encoding="utf-8"), "utf-8")
        captured = {}

        def fake(paths, output_dir, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(batch, "run_batch", fake)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--input-dir",
                str(folder),
                "--output-csv",
                str(tmp_path / "o.csv"),
                "--designed-with",
                "ColabDesign",
            ],
        )
        with pytest.raises(SystemExit):
            batch.main()
        assert captured["designed_with"] == "colabdesign" and captured["design_policy"] == "skip"
