"""``run_pipeline`` and ``binding-metrics-run`` with ``--predictor``: one run per prediction.

The model is a stub (``StubOpenFold``) that writes a synthetic OpenFold3 output; everything
else (the store, the session, the adapters, the confidence and EvoBind metrics, the report)
is the real code. The options that read an existing output run for every registered adapter
with its synthetic writer.
"""

import importlib
import json
import math
import sys

import pytest

from binding_metrics.cli import run
from binding_metrics.cli.run import _collect_failures, run_pipeline
from binding_metrics.predictors import PARSERS
from binding_metrics.protocols.report import write_report
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
    write_of3_output,
)

STEM = EXAMPLE_1YCR.stem


def pipeline(tmp_path, *, metrics=frozenset({"openfold"}), **kwargs):
    kwargs.setdefault("peptide_chain", PEPTIDE_CHAIN)
    kwargs.setdefault("receptor_chain", RECEPTOR_CHAIN)
    kwargs.setdefault("openfold_conda_env", None)
    return run_pipeline(
        EXAMPLE_1YCR, tmp_path, skip_prep=True, skip_relax=True, metrics=metrics, **kwargs
    )


def finite(value) -> bool:
    return value is not None and math.isfinite(value)


class TestRunOnce:
    def test_one_model_start_feeds_every_consumer(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3")["prediction"]

        assert stub.starts == 1 and stub.predicted == [STEM]
        assert block["cache"]["runs"] == 1
        assert block["cache"]["parsed"] == 1  # the record is parsed once for all four consumers
        # the consumers: confidence scalars, interface PAE/PDE, EvoBind score and adversarial
        assert finite(block["avg_plddt"]) and finite(block["iptm"])
        assert finite(block["mean_interface_pae"]) and finite(block["mean_interface_pde"])
        assert finite(block["evobind_score"])
        assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)
        assert "error" not in block and "reason" not in block

    def test_the_block_holds_the_summary_keys_and_the_model(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        assert list(block)[:3] == ["model", "query_name", "seed"]
        assert block["model"] == "of3" and block["query_name"] == STEM
        assert block["n_atoms"] == 818 and len(block["binder_plddt_per_residue"]) == 13
        assert block["adversary_model"] == "of3"
        assert len(block["cache"]["request_key"]) == 64

    def test_the_openfold_entry_is_skipped_and_the_others_are_untouched(
        self, tmp_path, monkeypatch
    ):
        StubOpenFold(monkeypatch)
        results = pipeline(tmp_path, predictor="of3")
        assert results["openfold"] == {"skipped": True}
        assert results["energy"] == {"skipped": True}

    def test_without_a_predictor_there_is_no_prediction_entry(self, tmp_path):
        results = pipeline(tmp_path, metrics=frozenset())
        assert "prediction" not in results

    def test_the_step_not_selected_gives_a_skipped_prediction(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        results = pipeline(tmp_path, metrics=frozenset(), predictor="of3")
        assert results["prediction"] == {"skipped": True}
        assert stub.starts == 0

    def test_the_store_is_below_the_output_directory(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        key = block["cache"]["request_key"]
        entry = tmp_path / "predictions" / "of3" / key[:2] / key
        assert (entry / "request.json").is_file() and (entry / "STATUS.json").is_file()
        assert json.loads((entry / "STATUS.json").read_text(encoding="utf-8"))["status"] == "done"

    def test_a_second_run_over_the_same_store_starts_no_model(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        first = pipeline(tmp_path / "a", predictor="of3", prediction_cache=store)["prediction"]
        second = pipeline(tmp_path / "b", predictor="of3", prediction_cache=store)["prediction"]

        assert stub.starts == 1
        assert first["cache"]["runs"] == 1 and second["cache"]["runs"] == 0
        assert second["cache"]["hits"] == 1
        assert second["cache"]["request_key"] == first["cache"]["request_key"]
        assert second["avg_plddt"] == first["avg_plddt"]

    def test_rerun_predictions_starts_the_model_again_once(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        pipeline(tmp_path / "a", predictor="of3", prediction_cache=store)
        rerun = pipeline(
            tmp_path / "b", predictor="of3", prediction_cache=store, rerun_predictions=True
        )["prediction"]
        assert stub.starts == 2
        assert rerun["cache"]["runs"] == 1 and rerun["cache"]["requests"] == 1

    def test_another_seed_is_another_prediction(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        one = pipeline(tmp_path / "a", predictor="of3", prediction_cache=store)["prediction"]
        two = pipeline(tmp_path / "b", predictor="of3", prediction_cache=store, openfold_seeds=[7])[
            "prediction"
        ]
        assert stub.starts == 2
        assert one["cache"]["request_key"] != two["cache"]["request_key"]

    def test_the_results_and_the_report_are_written(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        results = pipeline(tmp_path, predictor="of3")
        path = write_report(results, tmp_path, "s", fmt="json", summary=True)
        stored = json.loads(path.read_text(encoding="utf-8"))["prediction"]
        assert len(stored["plddt_per_atom"]) == 818 and stored["model"] == "of3"
        summary = (tmp_path / "s_report.md").read_text(encoding="utf-8")
        assert "## Structure prediction (OpenFold3)" in summary
        assert "_Prediction store: 1 run(s), 0 store hit(s), 0 adopted output(s)._" in summary


class TestRunnerArguments:
    def test_mode_score_uses_the_scoring_function(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        pipeline(tmp_path, predictor="of3")
        assert stub.calls[0]["kind"] == "scoring"
        assert "seeds" not in stub.calls[0]["kwargs"]

    def test_mode_refold_uses_the_refolding_function_and_measures_the_rmsd(
        self, tmp_path, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3", openfold_mode="refold")["prediction"]
        assert stub.calls[0]["kind"] == "refolding"
        assert block["binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)

    def test_the_rmsd_is_measured_in_score_mode_too_in_the_receptor_frame(
        self, tmp_path, monkeypatch
    ):
        # A template holds one chain and no cross-chain geometry, so OpenFold3 places the binder
        # itself in score mode as well, and the input is the reference in both modes.
        StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        assert block["binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)

    def test_seeds_reach_the_run(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        pipeline(tmp_path, predictor="of3", openfold_seeds=[7, 8])
        assert stub.calls[0]["kwargs"]["seeds"] == (7, 8)

    def test_the_conda_environment_reaches_the_run(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        pipeline(tmp_path, predictor="of3", openfold_conda_env="of3env")
        assert stub.calls[0]["kwargs"]["conda_env"] == "of3env"

    def test_an_empty_conda_environment_means_the_current_one(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        pipeline(tmp_path, predictor="of3", openfold_conda_env="")
        assert "conda_env" not in stub.calls[0]["kwargs"]

    def test_on_unmappable_residue_reaches_the_run(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        pipeline(tmp_path, predictor="of3", on_unmappable_residue="x")
        assert stub.calls[0]["kwargs"]["on_unmappable_residue"] == "x"

    def test_the_run_uses_the_input_structure_and_the_chain_roles(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        pipeline(tmp_path, predictor="of3")
        kwargs = stub.calls[0]["kwargs"]
        assert kwargs["complex_structure_path"] == EXAMPLE_1YCR
        assert (kwargs["binder_chain"], kwargs["receptor_chain"]) == ("B", "A")


class TestProvenance:
    def test_the_checkpoint_of_the_run_is_recorded(self, tmp_path, monkeypatch):
        from binding_metrics.metrics import openfold

        stub = StubOpenFold(monkeypatch)
        single = openfold.run_openfold_scoring

        def with_config(**kwargs):
            predictions = single(**kwargs)
            (predictions / "experiment_config.json").write_text(
                json.dumps({"inference_ckpt_name": "of3-ob-2025-06-30-174k.pt"}), encoding="utf-8"
            )
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_scoring", with_config)
        results = pipeline(tmp_path, predictor="of3")
        assert stub.starts == 1
        assert results["provenance"]["openfold3_checkpoint"] == "of3-ob-2025-06-30-174k.pt"

    def test_a_record_without_a_checkpoint_adds_no_key(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        assert "openfold3_checkpoint" not in pipeline(tmp_path, predictor="of3")["provenance"]

    def test_the_version_is_asked_when_the_model_is_run_here(self, tmp_path, monkeypatch):
        from binding_metrics.metrics import _openfold_run

        StubOpenFold(monkeypatch)
        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", lambda cmd=None: "0.5.0")
        assert pipeline(tmp_path, predictor="of3")["provenance"]["openfold3_version"] == "0.5.0"

    def test_the_version_is_not_asked_for_output_read_from_disk(self, tmp_path, monkeypatch):
        from binding_metrics.metrics import _openfold_run

        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", lambda cmd=None: "0.5.0")
        out = write_of3_output(tmp_path / "out", STEM, EXAMPLE_1YCR)
        results = pipeline(tmp_path / "run", predictor="of3", prediction_dir=out)
        assert "openfold3_version" not in results["provenance"]


class TestFailure:
    def test_a_failed_run_is_recorded_and_the_other_steps_go_on(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch, error=RuntimeError("GPU out of memory"))
        results = pipeline(tmp_path, metrics=frozenset({"openfold", "geometry"}), predictor="of3")
        block = results["prediction"]
        assert "GPU out of memory" in block["error"] and block["model"] == "of3"
        assert block["cache"]["failed"] == 1
        assert "error" not in results["geometry"] and results["geometry"] != {"skipped": True}
        assert [step for step, _ in _collect_failures(results)] == ["prediction"]
        assert stub.starts == 1

    def test_a_failed_run_is_not_started_again_without_rerun(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch, error=RuntimeError("GPU out of memory"))
        store = tmp_path / "shared"
        pipeline(tmp_path / "a", predictor="of3", prediction_cache=store)
        again = pipeline(tmp_path / "b", predictor="of3", prediction_cache=store)["prediction"]
        assert stub.starts == 1
        assert "GPU out of memory" in again["error"] and "--rerun-predictions" in again["error"]

    def test_rerun_predictions_retries_a_failed_run(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch, error=RuntimeError("GPU out of memory"))
        store = tmp_path / "shared"
        pipeline(tmp_path / "a", predictor="of3", prediction_cache=store)
        stub.error = None
        block = pipeline(
            tmp_path / "b", predictor="of3", prediction_cache=store, rerun_predictions=True
        )["prediction"]
        assert stub.starts == 2 and "error" not in block

    def test_a_run_that_writes_nothing_is_a_failure(self, tmp_path, monkeypatch):
        from binding_metrics.metrics import openfold

        StubOpenFold(monkeypatch)
        monkeypatch.setattr(openfold, "run_openfold_scoring", lambda **kw: tmp_path)
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        assert "wrote no output" in block["error"]

    def test_a_model_that_cannot_start_is_reported_and_records_nothing(self, tmp_path, monkeypatch):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        stub = StubOpenFold(monkeypatch)
        monkeypatch.setattr(OpenFold3Runner, "is_available", lambda runner: False)
        results = pipeline(tmp_path, predictor="of3")
        assert "cannot be started on this machine" in results["prediction"]["error"]
        assert stub.starts == 0
        assert not list((tmp_path / "predictions").glob("of3/*/*/STATUS.json"))

    def test_a_missing_chain_skips_the_step(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        # detect_chains_from_file finds both chains itself; a stub of it leaves none
        monkeypatch.setattr(
            "binding_metrics.io.structures.detect_chains_from_file",
            lambda *a, **kw: {
                "peptide_chain": None,
                "receptor_chain": None,
                "peptide_chain_label": None,
                "receptor_chain_label": None,
                "all_chains": [],
            },
        )
        results = pipeline(tmp_path, predictor="of3", peptide_chain=None, receptor_chain=None)
        assert results["prediction"] == {"skipped": True}
        assert stub.starts == 0


class TestParseOnly:
    def test_the_directory_is_adopted_and_the_model_never_runs(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        out = write_of3_output(tmp_path / "out", STEM, EXAMPLE_1YCR)
        block = pipeline(tmp_path / "run", predictor="of3", prediction_dir=out)["prediction"]
        assert stub.starts == 0
        assert block["cache"]["adopted"] == 1 and block["cache"]["runs"] == 0
        assert finite(block["avg_plddt"]) and finite(block["evobind_score"])
        assert block["structure_path"].startswith(str(out.resolve()))  # read where it lies

    def test_a_rerun_never_replaces_adopted_outputs(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        out = write_of3_output(tmp_path / "out", STEM, EXAMPLE_1YCR)
        block = pipeline(
            tmp_path / "run", predictor="of3", prediction_dir=out, rerun_predictions=True
        )["prediction"]
        assert stub.starts == 0 and block["cache"]["adopted"] == 1

    def test_a_directory_without_the_sample_is_an_error(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        out = write_of3_output(tmp_path / "out", "another_sample", EXAMPLE_1YCR)
        block = pipeline(tmp_path / "run", predictor="of3", prediction_dir=out)["prediction"]
        assert f"no OpenFold3 output for '{STEM}'" in block["error"]
        assert stub.starts == 0

    def test_a_directory_that_does_not_exist_is_an_error(self, tmp_path):
        block = pipeline(tmp_path, predictor="of3", prediction_dir=tmp_path / "nowhere")[
            "prediction"
        ]
        assert f"no OpenFold3 output for '{STEM}'" in block["error"]

    def test_the_chain_ids_of_the_prediction_are_renamed(self, tmp_path, monkeypatch):
        from tests.predictors import synth, synth_of3
        from tests.test_feat_c_support import complex_from

        StubOpenFold(monkeypatch)
        truth = complex_from(EXAMPLE_1YCR)
        renamed = synth.SyntheticComplex(
            atoms=synth.renamed_atoms(truth, {"A": "R", "B": "P"}),
            plddt_per_atom=truth.plddt_per_atom,
            pae=truth.pae,
            pde=truth.pde,
            scalars=truth.scalars,
            chain_ptm=truth.chain_ptm,
            chain_pair_iptm=truth.chain_pair_iptm,
        )
        out = tmp_path / "out"
        synth_of3.write_prediction(out, STEM, renamed)

        block = pipeline(
            tmp_path / "run",
            predictor="of3",
            prediction_dir=out,
            prediction_binder_chain="P",
            prediction_target_chain="R",
        )["prediction"]
        assert "error" not in block
        assert len(block["binder_plddt_per_residue"]) == 13
        assert finite(block["mean_interface_pae"]) and finite(block["evobind_score"])
        assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)

    def test_without_the_chain_options_renamed_chains_are_reported(self, tmp_path, monkeypatch):
        from tests.predictors import synth, synth_of3
        from tests.test_feat_c_support import complex_from

        StubOpenFold(monkeypatch)
        truth = complex_from(EXAMPLE_1YCR)
        renamed = synth.SyntheticComplex(
            atoms=synth.renamed_atoms(truth, {"A": "R", "B": "P"}),
            plddt_per_atom=truth.plddt_per_atom,
            pae=truth.pae,
            pde=truth.pde,
            scalars=truth.scalars,
        )
        out = tmp_path / "out"
        synth_of3.write_prediction(out, STEM, renamed)
        block = pipeline(tmp_path / "run", predictor="of3", prediction_dir=out)["prediction"]
        assert "evobind_error" in block and "adversarial_error" in block

    @pytest.mark.parametrize("model", sorted(PARSERS))
    def test_every_registered_adapter_runs_through_the_pipeline(self, tmp_path, monkeypatch, model):
        from tests.test_feat_c_support import complex_from

        writer = importlib.import_module(f"tests.predictors.synth_{model}")
        out = tmp_path / "out"
        out.mkdir()
        writer.write_prediction(out, STEM, complex_from(EXAMPLE_1YCR))

        results = pipeline(tmp_path / "run", predictor=model, prediction_dir=out)
        block = results["prediction"]
        assert block["model"] == model and "error" not in block, block.get("error")
        assert block["cache"]["adopted"] == 1 and block["cache"]["runs"] == 0
        assert finite(block["avg_plddt"]) and finite(block["binder_avg_plddt"])
        assert finite(block["evobind_score"])
        assert finite(block["mean_interface_pae"])
        assert _collect_failures(results) == []
        # Boltz-2 numbers residues from 1; the residue pairing of the adversarial check
        # cannot follow that for a design whose numbers start elsewhere (a known limit)
        if model != "boltz2":
            assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
            assert "adversarial_error" not in block


class TestApiChecks:
    def test_an_unknown_predictor_raises_before_any_step(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown predictor 'nope'"):
            pipeline(tmp_path, metrics=frozenset(), predictor="nope")
        assert not any(tmp_path.iterdir())

    @pytest.mark.parametrize("model", [m for m in sorted(PARSERS) if m != "of3"])
    def test_a_model_without_a_runner_needs_a_directory(self, tmp_path, model):
        with pytest.raises(ValueError, match="has no runner yet"):
            pipeline(tmp_path, metrics=frozenset(), predictor=model)
        assert not any(tmp_path.iterdir())


class TestCommandLine:
    @staticmethod
    def _parsed(monkeypatch, tmp_path, extra):
        captured = {}

        def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return {"sample_id": "x", "provenance": {}}

        monkeypatch.setattr(run, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o"), *extra],
        )
        run.main()
        return captured

    def test_the_defaults_leave_the_step_as_it_was(self, monkeypatch, tmp_path):
        parsed = self._parsed(monkeypatch, tmp_path, [])
        assert parsed["predictor"] is None
        assert parsed["prediction_dir"] is None
        assert parsed["prediction_binder_chain"] is None
        assert parsed["prediction_target_chain"] is None
        assert parsed["prediction_cache"] is None
        assert parsed["rerun_predictions"] is False

    def test_the_options_reach_the_pipeline(self, monkeypatch, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        parsed = self._parsed(
            monkeypatch,
            tmp_path,
            [
                "--predictor",
                "af2",
                "--prediction-dir",
                str(out),
                "--prediction-binder-chain",
                "P",
                "--prediction-target-chain",
                "R",
                "--prediction-cache",
                str(tmp_path / "cache"),
                "--rerun-predictions",
            ],
        )
        assert parsed["predictor"] == "af2" and parsed["prediction_dir"] == out
        assert parsed["prediction_binder_chain"] == "P"
        assert parsed["prediction_target_chain"] == "R"
        assert parsed["prediction_cache"] == tmp_path / "cache"
        assert parsed["rerun_predictions"] is True

    def test_the_choices_are_the_registered_models(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as stop:
            self._parsed(monkeypatch, tmp_path, ["--predictor", "nope"])
        assert stop.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice" in err
        assert all(model in err for model in PARSERS)

    @pytest.mark.parametrize("model", [m for m in sorted(PARSERS) if m != "of3"])
    def test_a_model_without_a_runner_fails_while_the_arguments_are_checked(
        self, monkeypatch, tmp_path, capsys, model
    ):
        with pytest.raises(SystemExit) as stop:
            self._parsed(monkeypatch, tmp_path, ["--predictor", model])
        assert stop.value.code == 2
        err = capsys.readouterr().err
        assert "has no runner yet" in err and "--prediction-dir" in err

    def test_of3_may_be_run_without_a_directory(self, monkeypatch, tmp_path):
        assert self._parsed(monkeypatch, tmp_path, ["--predictor", "of3"])["predictor"] == "of3"

    @pytest.mark.parametrize(
        "option, value",
        [
            ("--prediction-dir", "somewhere"),
            ("--prediction-binder-chain", "P"),
            ("--prediction-target-chain", "R"),
            ("--prediction-cache", "cache"),
            ("--rerun-predictions", None),
        ],
    )
    def test_the_options_need_a_predictor(self, monkeypatch, tmp_path, capsys, option, value):
        with pytest.raises(SystemExit) as stop:
            self._parsed(monkeypatch, tmp_path, [option, *([value] if value else [])])
        assert stop.value.code == 2
        assert f"{option} needs --predictor" in capsys.readouterr().err

    def test_a_directory_that_does_not_exist_is_refused(self, monkeypatch, tmp_path, capsys):
        missing = tmp_path / "nowhere"
        with pytest.raises(SystemExit) as stop:
            self._parsed(
                monkeypatch, tmp_path, ["--predictor", "of3", "--prediction-dir", str(missing)]
            )
        assert stop.value.code == 1
        assert f"--prediction-dir is not a directory: {missing}" in capsys.readouterr().err

    def test_the_config_file_sets_the_options(self, monkeypatch, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        config = tmp_path / "run.toml"
        config.write_text(
            f'predictor = "boltz2"\nprediction-dir = "{out}"\nrerun-predictions = true\n',
            encoding="utf-8",
        )
        parsed = self._parsed(monkeypatch, tmp_path, ["--config", str(config)])
        assert parsed["predictor"] == "boltz2" and parsed["prediction_dir"] == out
        assert parsed["rerun_predictions"] is True

    def test_a_config_predictor_outside_the_choices_is_refused(self, monkeypatch, tmp_path, capsys):
        config = tmp_path / "run.toml"
        config.write_text('predictor = "nope"\n', encoding="utf-8")
        with pytest.raises(SystemExit) as stop:
            self._parsed(monkeypatch, tmp_path, ["--config", str(config)])
        assert stop.value.code == 2
        assert "'nope' is not one of" in capsys.readouterr().err
