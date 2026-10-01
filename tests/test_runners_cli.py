"""``--predictor MODEL`` without ``--prediction-dir``: the same checks for every registered runner.

The tests are parametrized over ``RUNNERS``, so a runner added to the registry is run through the
arguments, the store and the metrics without a new line here. The model is a stand-in process
(``tests/test_runners_support.ModelStub``); the runner, its input files and its command line, the
store, the session, the adapter and the metrics are the real code. What the tests pin:

* the registered runners are the four models, and each says which modes it runs and which it
  runs by default; a mode it does not run is a usage error that names the ones it does;
* the run starts the model once for all consumers, a second call finds the stored output,
  ``results["prediction"]`` names the model and its mode, and ``binder_ca_rmsd`` is measured;
* the chain IDs of the prediction are the input's, whatever the model calls them;
* the generic settings (``--prediction-cyclic``, ``--prediction-no-msa-server``,
  ``--prediction-conda-env``, ``--prediction-lock-threshold``) reach the command of the model that
  has the setting and are a usage error for the one that has not, and the ``--openfold-*``
  spellings stay the settings of OpenFold3.
"""

import argparse
import math
import sys
from pathlib import Path

import pytest

from binding_metrics.capabilities import MODES, IncompatibleInputError
from binding_metrics.cli import batch, prediction, run
from binding_metrics.cli.prediction import RUNNERS, runner_class
from binding_metrics.cli.run import run_pipeline
from binding_metrics.predictors.af2_runner import ColabFoldRunner
from binding_metrics.predictors.registry import PARSERS
from tests.test_feat_c_support import EXAMPLE_1YCR
from tests.test_runners_support import MODELS, ModelStub, default_mode_of, renamed_input

STEM = EXAMPLE_1YCR.stem


def pipeline(tmp_path, model, input_path=EXAMPLE_1YCR, **kwargs):
    kwargs.setdefault("peptide_chain", "B")
    kwargs.setdefault("receptor_chain", "A")
    return run_pipeline(
        input_path,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        predictor=model,
        **kwargs,
    )


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


#: (model, mode) the runner does not run, split by who refuses it: the model's declared limits
#: (the pre-flight check) or the runner (the argument check).
RUNNER_REFUSES = [
    (model, mode)
    for model in MODELS
    for mode in MODES
    if mode not in prediction.runner_modes(model) and not prediction.model_refuses_mode(model, mode)
]
MODEL_REFUSES = [
    (model, mode)
    for model in MODELS
    for mode in MODES
    if mode not in prediction.runner_modes(model) and prediction.model_refuses_mode(model, mode)
]


class TestTheRegistry:
    def test_every_parser_has_a_runner(self):
        assert sorted(RUNNERS) == sorted(PARSERS) == ["af2", "boltz2", "of3", "protenix"]

    def test_the_runners_are_the_classes_of_the_lanes(self):
        names = {model: runner_class(model).__name__ for model in MODELS}
        assert names == {
            "af2": "ColabFoldRunner",
            "boltz2": "Boltz2Runner",
            "of3": "OpenFold3Runner",
            "protenix": "ProtenixRunner",
        }
        for model in MODELS:
            assert runner_class(model).name == model

    @pytest.mark.parametrize("model", MODELS)
    def test_a_runner_runs_its_default_mode(self, model):
        assert prediction.runner_default_mode(model) == default_mode_of(model)
        assert prediction.runner_default_mode(model) in prediction.runner_modes(model)

    def test_the_modes_each_runner_runs(self):
        assert {model: sorted(prediction.runner_modes(model)) for model in MODELS} == {
            "af2": ["predict"],
            "boltz2": ["predict", "refold", "score", "score-lock"],
            "of3": ["refold", "score"],
            "protenix": ["predict"],
        }

    def test_the_weights_kinds_are_read_from_the_classes(self):
        assert prediction.runner_weights_kinds() == {
            "af2": "directory",
            "boltz2": "file",
            "of3": "file",
            "protenix": "directory",
        }

    def test_every_mode_that_is_not_run_is_refused_by_the_runner_or_the_model(self):
        for model in MODELS:
            missing = set(MODES) - prediction.runner_modes(model)
            refused = {m for mm, m in RUNNER_REFUSES + MODEL_REFUSES if mm == model}
            assert missing == refused


class TestOneRun:
    @pytest.mark.parametrize("model", MODELS)
    def test_the_model_is_started_once_for_every_consumer(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        block = pipeline(tmp_path / "o", model)["prediction"]

        assert "error" not in block, block.get("error")
        assert stub.starts == 1
        assert block["model"] == model
        assert block["mode"] == default_mode_of(model)
        assert block["cache"]["runs"] == 1 and block["cache"]["parsed"] == 1
        # the consumers: confidence scalars, interface PAE, EvoBind score and adversarial check
        assert finite(block["avg_plddt"]) and block["iptm"] == pytest.approx(0.76)
        assert finite(block["mean_interface_pae"]) and finite(block["evobind_score"])
        assert block["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-3)
        assert "evobind_error" not in block and "adversarial_error" not in block

    @pytest.mark.parametrize("model", MODELS)
    def test_the_binder_is_measured_against_the_input_pose(self, tmp_path, monkeypatch, model):
        ModelStub(model, tmp_path, monkeypatch)
        block = pipeline(tmp_path / "o", model)["prediction"]
        assert block["binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)
        assert "binder RMSD" not in block.get("reason", "")

    @pytest.mark.parametrize("model", MODELS)
    def test_a_second_run_over_the_same_store_starts_no_model(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        store = tmp_path / "shared"
        first = pipeline(tmp_path / "a", model, prediction_cache=store)["prediction"]
        second = pipeline(tmp_path / "b", model, prediction_cache=store)["prediction"]

        assert stub.starts == 1
        assert first["cache"]["runs"] == 1 and second["cache"]["runs"] == 0
        assert second["cache"]["hits"] == 1
        assert second["cache"]["request_key"] == first["cache"]["request_key"]
        assert second["avg_plddt"] == first["avg_plddt"]

    @pytest.mark.parametrize("model", MODELS)
    def test_the_rerun_flag_starts_the_model_again(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        store = tmp_path / "shared"
        pipeline(tmp_path / "a", model, prediction_cache=store)
        pipeline(tmp_path / "b", model, prediction_cache=store, rerun_predictions=True)
        assert stub.starts == 2

    @pytest.mark.parametrize("model", MODELS)
    def test_the_preflight_names_the_model_and_its_mode(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        results = pipeline(tmp_path / "o", model, preflight_only=True)
        assert stub.starts == 0 and not (tmp_path / "o").exists()
        assert results["preflight"]["status"] in ("ok", "warn")
        assert f"Mode: {default_mode_of(model)}" in results["preflight"]["plan"]

    @pytest.mark.parametrize("model", MODELS)
    def test_the_result_and_the_preflight_record_the_mode(self, tmp_path, monkeypatch, model):
        ModelStub(model, tmp_path, monkeypatch)
        results = pipeline(tmp_path / "o", model)
        assert results["preflight"]["report"]["mode"] == default_mode_of(model)
        assert results["prediction"]["mode"] == default_mode_of(model)

    @pytest.mark.parametrize("model", MODELS)
    def test_a_failed_run_names_the_model_and_the_sample_goes_on(
        self, tmp_path, monkeypatch, caplog, model
    ):
        stub = ModelStub(model, tmp_path, monkeypatch)
        stub.fail()
        results = pipeline(tmp_path / "o", model)
        block = results["prediction"]
        assert block["model"] == model and block["mode"] == default_mode_of(model)
        assert "stand-in failure" in block["error"] and f"the {model} prediction" in block["error"]
        assert results["energy"] == {"skipped": True}
        assert any(
            f"{prediction.display_name(model)} prediction failed" in record.getMessage()
            for record in caplog.records
        )


class TestTheModeOfARun:
    @pytest.mark.parametrize("model", MODELS)
    def test_every_mode_the_runner_runs_runs_through_the_stub(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        keys = set()
        for mode in sorted(prediction.runner_modes(model)):
            results = pipeline(tmp_path / mode, model, prediction_mode=mode, on_incompatible="warn")
            block = results["prediction"]
            assert "error" not in block, (mode, block.get("error"))
            assert block["mode"] == mode and results["preflight"]["report"]["mode"] == mode
            keys.add(block["cache"]["request_key"])
        assert len(keys) == len(prediction.runner_modes(model))  # a mode is part of the key
        assert stub.starts == len(keys)

    @pytest.mark.parametrize("model, mode", RUNNER_REFUSES)
    def test_a_mode_the_runner_does_not_run_is_a_usage_error(
        self, tmp_path, monkeypatch, capsys, model, mode
    ):
        stub = ModelStub(model, tmp_path, monkeypatch)
        have = sorted(prediction.runner_modes(model))
        with pytest.raises(ValueError) as caught:
            pipeline(tmp_path / "o", model, prediction_mode=mode)
        message = str(caught.value)
        assert f"--prediction-mode {mode} cannot be run from here" in message
        assert prediction.display_name(model) in message
        assert all(name in message for name in have)
        assert "--prediction-dir DIR" in message
        assert stub.starts == 0 and not (tmp_path / "o").exists()

        parser = argparse.ArgumentParser()
        prediction.add_prediction_args(parser)
        args = parser.parse_args(["--predictor", model, "--prediction-mode", mode])
        with pytest.raises(SystemExit) as stop:
            prediction.check_prediction_args(parser, args)
        assert stop.value.code == 2
        assert f"--prediction-mode {mode} cannot be run from here" in capsys.readouterr().err

    @pytest.mark.parametrize("model, mode", MODEL_REFUSES)
    def test_a_mode_the_model_does_not_have_is_left_to_the_preflight_check(
        self, tmp_path, monkeypatch, model, mode
    ):
        """The pre-flight check says why, and which models do have the mode."""
        stub = ModelStub(model, tmp_path, monkeypatch)
        with pytest.raises(IncompatibleInputError) as caught:
            pipeline(tmp_path / "o", model, prediction_mode=mode)
        assert f"mode '{mode}' was requested" in str(caught.value)
        assert stub.starts == 0

    @pytest.mark.parametrize("model", MODELS)
    def test_a_mode_read_from_a_directory_is_not_limited_by_the_runner(self, tmp_path, model):
        parser = argparse.ArgumentParser()
        prediction.add_prediction_args(parser)
        read = parser.parse_args(
            ["--predictor", model, "--prediction-mode", "score-lock", "--prediction-dir", "."]
        )
        prediction.check_prediction_args(parser, read)  # no error
        prediction.check_predictor(model, tmp_path, "predict")

    @pytest.mark.parametrize("model", MODELS)
    def test_the_effective_mode_is_the_default_of_the_runner(self, model):
        mode = prediction.effective_prediction_mode
        assert mode(model, None, None, "score") == default_mode_of(model)
        assert mode(model, None, "score-lock", "score") == "score-lock"
        assert mode(model, Path("."), None, "score") is None  # made elsewhere: not known

    def test_openfold_mode_stays_the_mode_of_openfold3_only(self):
        mode = prediction.effective_prediction_mode
        assert mode("of3", None, None, "refold") == "refold"
        assert mode("boltz2", None, None, "refold") == "score"  # the runner's own default
        with pytest.raises(ValueError, match="--openfold-mode.*use --prediction-mode"):
            prediction.check_predictor("boltz2", None, openfold_mode="refold")


class TestTheChainsOfThePrediction:
    """The input calls its chains R and L; ColabFold writes A and B, the others keep R and L."""

    @pytest.fixture
    def renamed(self, tmp_path):
        return renamed_input(tmp_path / "ycr_rl.pdb", {"A": "R", "B": "L"})

    @pytest.mark.parametrize("model", MODELS)
    def test_the_metrics_find_the_chains_under_the_ids_of_the_input(
        self, tmp_path, monkeypatch, renamed, model
    ):
        ModelStub(model, tmp_path, monkeypatch, renamed, "R", "L", names=[renamed.stem])
        block = pipeline(tmp_path / "o", model, renamed, peptide_chain="L", receptor_chain="R")[
            "prediction"
        ]
        assert "error" not in block, block.get("error")
        assert block["binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)
        assert finite(block["evobind_score"]) and finite(block["mean_interface_pae"])
        assert "evobind_error" not in block and "adversarial_error" not in block
        assert "reason" not in block, block.get("reason")

    def test_without_the_map_the_chains_of_colabfold_are_not_found(
        self, tmp_path, monkeypatch, renamed
    ):
        ModelStub("af2", tmp_path, monkeypatch, renamed, "R", "L", names=[renamed.stem])
        monkeypatch.setattr(ColabFoldRunner, "output_chain_map", staticmethod(lambda request: None))
        block = pipeline(tmp_path / "o", "af2", renamed, peptide_chain="L", receptor_chain="R")[
            "prediction"
        ]
        assert "evobind_error" in block and "adversarial_error" in block
        assert math.isnan(block["binder_ca_rmsd"])
        # the reason names the model and what to do
        assert "AlphaFold2 / ColabFold" in block["reason"]
        assert "--prediction-binder-chain" in block["reason"]

    def test_the_chain_options_win_over_the_map_of_the_runner(self, tmp_path, monkeypatch, renamed):
        ModelStub("af2", tmp_path, monkeypatch, renamed, "R", "L", names=[renamed.stem])
        monkeypatch.setattr(ColabFoldRunner, "output_chain_map", staticmethod(lambda request: None))
        block = pipeline(
            tmp_path / "o",
            "af2",
            renamed,
            peptide_chain="L",
            receptor_chain="R",
            prediction_binder_chain="B",
            prediction_target_chain="A",
        )["prediction"]
        assert finite(block["evobind_score"]) and block["binder_ca_rmsd"] == pytest.approx(
            0, abs=1e-2
        )

    def test_the_default_map_of_a_runner_is_none(self):
        for model in ("of3", "boltz2", "protenix"):
            assert prediction.runner_chain_map(prediction.make_runner(model), object()) is None
        assert prediction.runner_chain_map(None, object()) is None

    def test_the_runner_of_colabfold_names_its_chains(self, tmp_path, monkeypatch, renamed):
        ModelStub("af2", tmp_path, monkeypatch, renamed, "R", "L", names=[renamed.stem])
        runner = prediction.make_runner("af2")
        request = prediction.make_request(
            "af2", "p", renamed, binder_chain="L", receptor_chain="R", runner=runner
        )
        assert prediction.runner_chain_map(runner, request) == {"A": "R", "B": "L"}


class TestTheGenericSettings:
    """What each setting does to the command of each model, read from what the stand-in got."""

    @pytest.mark.parametrize("model", MODELS)
    def test_no_msa_server_reaches_every_model(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        block = pipeline(tmp_path / "o", model, prediction_use_msa_server=False)["prediction"]
        assert "error" not in block, block.get("error")
        call = stub.calls[-1]
        if model == "af2":
            assert stub.argument("--msa-mode") == "single_sequence"
        elif model == "boltz2":
            assert 'msa: "empty"' in call["input"] and "--use_msa_server" not in call["argv"]
        elif model == "protenix":
            assert stub.argument("--use_msa") == "false"
        else:
            assert call["kwargs"]["use_msa_server"] is False

    @pytest.mark.parametrize("model", MODELS)
    def test_the_server_is_used_unless_the_option_is_given(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        pipeline(tmp_path / "o", model)
        call = stub.calls[-1]
        if model == "af2":
            assert stub.argument("--msa-mode") == "mmseqs2_uniref_env"
        elif model == "boltz2":
            assert "--use_msa_server" in call["argv"]
        elif model == "protenix":
            assert stub.argument("--use_msa") == "true"
        else:
            assert "use_msa_server" not in call["kwargs"]

    def test_the_server_option_changes_the_request_key(self, tmp_path, monkeypatch):
        keys = set()
        for model in MODELS:
            ModelStub(model, tmp_path / model, monkeypatch)
            for use in (True, False):
                block = pipeline(tmp_path / model / str(use), model, prediction_use_msa_server=use)
                keys.add((model, use, block["prediction"]["cache"]["request_key"]))
        assert len({key for _, _, key in keys}) == 2 * len(MODELS)

    @pytest.mark.parametrize("model", ["boltz2", "protenix", "of3"])
    def test_cyclic_on_reaches_the_models_that_take_it(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        block = pipeline(tmp_path / "o", model, prediction_cyclic="on")["prediction"]
        assert "error" not in block, block.get("error")
        call = stub.calls[-1]
        if model == "boltz2":
            assert "      cyclic: true\n" in call["input"]
        elif model == "protenix":
            (bond,) = call["input"][0]["covalent_bonds"]
            assert (bond["atom1"], bond["atom2"]) == ("C", "N")
        else:
            assert call["kwargs"]["binder_cyclic"] is True

    @pytest.mark.parametrize("model", ["boltz2", "protenix", "of3"])
    def test_cyclic_off_and_auto_write_no_closure_for_a_linear_binder(
        self, tmp_path, monkeypatch, model
    ):
        stub = ModelStub(model, tmp_path, monkeypatch)
        for value in ("off", "auto", None):
            pipeline(tmp_path / str(value), model, prediction_cyclic=value)
        for call in stub.calls:
            if model == "boltz2":
                assert "cyclic: true" not in call["input"]
            elif model == "protenix":
                assert "covalent_bonds" not in call["input"][0]

    @pytest.mark.parametrize("value", ["on", "off"])
    def test_colabfold_has_no_cyclic_setting_and_refuses_the_option(
        self, tmp_path, monkeypatch, capsys, value
    ):
        stub = ModelStub("af2", tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="--prediction-cyclic is not available for AlphaFold2"):
            pipeline(tmp_path / "o", "af2", prediction_cyclic=value)
        assert stub.starts == 0 and not (tmp_path / "o").exists()
        assert _usage_error(["--predictor", "af2", "--prediction-cyclic", value], capsys) == 2
        assert "--prediction-cyclic is not available for AlphaFold2" in capsys.readouterr().err

    @pytest.mark.parametrize("model", MODELS)
    def test_the_conda_environment_reaches_the_command(self, tmp_path, monkeypatch, model):
        stub = ModelStub(model, tmp_path, monkeypatch)
        if model == "of3":
            monkeypatch.setattr(
                "binding_metrics.predictors.of3_runner.OpenFold3Runner.version", lambda r: "0.5.0"
            )
        block = pipeline(tmp_path / "o", model, prediction_conda_env="myenv")["prediction"]
        assert "error" not in block, block.get("error")
        call = stub.calls[-1]
        assert (call["kwargs"]["conda_env"] if model == "of3" else call["conda_env"]) == "myenv"

    @pytest.mark.parametrize("model", [m for m in MODELS if m != "of3"])
    def test_without_the_option_a_model_runs_in_the_current_environment(
        self, tmp_path, monkeypatch, model
    ):
        """``--openfold-conda-env`` has the default ``openfold3``: it is not the environment of
        another model."""
        stub = ModelStub(model, tmp_path, monkeypatch)
        pipeline(tmp_path / "o", model, openfold_conda_env=None)
        assert stub.calls[-1]["conda_env"] is None

    def test_the_lock_threshold_reaches_boltz2_in_score_lock(self, tmp_path, monkeypatch):
        stub = ModelStub("boltz2", tmp_path, monkeypatch)
        block = pipeline(
            tmp_path / "o",
            "boltz2",
            prediction_mode="score-lock",
            prediction_lock_threshold=3.5,
            on_incompatible="warn",
        )["prediction"]
        assert "error" not in block, block.get("error")
        assert "    force: true\n    threshold: 3.5\n" in stub.calls[-1]["input"]

    def test_the_lock_threshold_defaults_to_the_one_of_the_runner(self, tmp_path, monkeypatch):
        from binding_metrics.predictors.boltz2_runner import DEFAULT_LOCK_THRESHOLD_ANGSTROM

        stub = ModelStub("boltz2", tmp_path, monkeypatch)
        pipeline(tmp_path / "o", "boltz2", prediction_mode="score-lock", on_incompatible="warn")
        assert f"threshold: {DEFAULT_LOCK_THRESHOLD_ANGSTROM}\n" in stub.calls[-1]["input"]

    @pytest.mark.parametrize("model", [m for m in MODELS if m != "boltz2"])
    def test_the_lock_threshold_is_refused_for_a_runner_without_it(
        self, tmp_path, monkeypatch, model
    ):
        stub = ModelStub(model, tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="--prediction-lock-threshold is not available for"):
            pipeline(tmp_path / "o", model, prediction_lock_threshold=3.0)
        assert stub.starts == 0

    @pytest.mark.parametrize("mode", [None, "predict", "refold", "score"])
    def test_the_lock_threshold_belongs_to_score_lock(self, tmp_path, monkeypatch, mode):
        stub = ModelStub("boltz2", tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="threshold of the mode score-lock"):
            pipeline(tmp_path / "o", "boltz2", prediction_mode=mode, prediction_lock_threshold=3.0)
        assert stub.starts == 0

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "two"])
    def test_the_lock_threshold_is_a_positive_number(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            prediction.lock_threshold_arg(value)
        assert prediction.lock_threshold_arg("2.5") == 2.5

    def test_seeds_and_the_unmappable_choice_go_to_every_runner(self, tmp_path, monkeypatch):
        stub = ModelStub("boltz2", tmp_path, monkeypatch)
        pipeline(tmp_path / "o", "boltz2", openfold_seeds=[7])
        assert stub.argument("--seed") == "7"
        stub = ModelStub("af2", tmp_path / "x", monkeypatch)
        pipeline(tmp_path / "p", "af2", openfold_seeds=[0, 1])
        assert stub.argument("--num-seeds") == "2"

    def test_a_seed_list_the_runner_refuses_is_recorded_as_the_error_of_the_step(
        self, tmp_path, monkeypatch
    ):
        stub = ModelStub("af2", tmp_path, monkeypatch)
        block = pipeline(tmp_path / "o", "af2", openfold_seeds=[1, 3])["prediction"]
        assert "consecutive integers from 0" in block["error"] and stub.starts == 0
        stub = ModelStub("boltz2", tmp_path / "x", monkeypatch)
        block = pipeline(tmp_path / "p", "boltz2", openfold_seeds=[1, 3])["prediction"]
        assert "one seed" in block["error"] and stub.starts == 0


class TestTheTwoSpellings:
    def test_the_openfold_spelling_is_the_setting_of_openfold3(self, tmp_path, monkeypatch):
        stub = ModelStub("of3", tmp_path, monkeypatch)
        pipeline(tmp_path / "o", "of3", openfold_cyclic="on", openfold_use_msa_server=False)
        kwargs = stub.calls[-1]["kwargs"]
        assert kwargs["binder_cyclic"] is True and kwargs["use_msa_server"] is False

    def test_both_spellings_with_one_value_are_one_setting(self, tmp_path, monkeypatch):
        stub = ModelStub("of3", tmp_path, monkeypatch)
        pipeline(
            tmp_path / "o",
            "of3",
            openfold_cyclic="on",
            prediction_cyclic="on",
            openfold_use_msa_server=False,
            prediction_use_msa_server=False,
            openfold_conda_env="e",
            prediction_conda_env="e",
        )
        kwargs = stub.calls[-1]["kwargs"]
        assert kwargs["binder_cyclic"] is True and kwargs["conda_env"] == "e"

    @pytest.mark.parametrize(
        "legacy, generic, match",
        [
            (
                {"openfold_cyclic": "on"},
                {"prediction_cyclic": "off"},
                "--openfold-cyclic on and --prediction-cyclic off set the same thing",
            ),
            (
                {"openfold_conda_env": "a"},
                {"prediction_conda_env": "b"},
                "--openfold-conda-env a and --prediction-conda-env b set the same thing",
            ),
        ],
    )
    def test_different_values_are_a_usage_error(
        self, tmp_path, monkeypatch, legacy, generic, match
    ):
        stub = ModelStub("of3", tmp_path, monkeypatch)
        with pytest.raises(ValueError, match=match):
            pipeline(tmp_path / "o", "of3", **legacy, **generic)
        assert stub.starts == 0

    def test_the_prediction_spelling_wins_over_the_default_environment(self, tmp_path, monkeypatch):
        stub = ModelStub("of3", tmp_path, monkeypatch)
        monkeypatch.setattr(
            "binding_metrics.predictors.of3_runner.OpenFold3Runner.version", lambda r: "0.5.0"
        )
        pipeline(tmp_path / "o", "of3", openfold_conda_env="openfold3", prediction_conda_env="mine")
        assert stub.calls[-1]["kwargs"]["conda_env"] == "mine"

    @pytest.mark.parametrize("model", [m for m in MODELS if m != "of3"])
    @pytest.mark.parametrize(
        "legacy, option, generic",
        [
            ({"openfold_cyclic": "on"}, "--openfold-cyclic", "--prediction-cyclic"),
            (
                {"openfold_use_msa_server": False},
                "--openfold-no-msa-server",
                "--prediction-no-msa-server",
            ),
            ({"openfold_conda_env": "boltz"}, "--openfold-conda-env", "--prediction-conda-env"),
            ({"openfold_mode": "refold"}, "--openfold-mode", "--prediction-mode"),
        ],
    )
    def test_the_openfold_spelling_is_refused_for_another_model(
        self, tmp_path, monkeypatch, model, legacy, option, generic
    ):
        stub = ModelStub(model, tmp_path, monkeypatch)
        with pytest.raises(ValueError) as caught:
            pipeline(tmp_path / "o", model, **legacy)
        message = str(caught.value)
        assert f"{option} sets how OpenFold3 is run" in message
        assert f"--predictor {model}" in message and f"use {generic}" in message
        assert stub.starts == 0

    def test_the_legacy_step_without_a_predictor_is_unchanged(self):
        route = prediction.check_predictor(
            None, None, openfold_cyclic="on", openfold_use_msa_server=False, openfold_conda_env="x"
        )
        assert (route.cyclic, route.use_msa_server, route.conda_env) == (True, False, "x")

    @pytest.mark.parametrize(
        "setting",
        [
            {"prediction_cyclic": "on"},
            {"prediction_use_msa_server": False},
            {"prediction_conda_env": "boltz"},
            {"prediction_lock_threshold": 2.0},
        ],
    )
    def test_the_settings_of_a_run_need_a_predictor_and_a_run(self, tmp_path, setting):
        with pytest.raises(ValueError, match="needs --predictor"):
            prediction.check_predictor(None, None, **setting)
        with pytest.raises(ValueError, match="cannot be combined with --prediction-dir"):
            prediction.check_predictor("boltz2", tmp_path, **setting)


def _usage_error(argv, capsys=None) -> int:
    """The exit code of ``check_prediction_args`` for these options (0 when they pass)."""
    parser = argparse.ArgumentParser()
    prediction.add_prediction_args(parser)
    parser.add_argument("--openfold-cyclic", default="auto")
    parser.add_argument("--openfold-no-msa-server", action="store_true")
    parser.add_argument("--openfold-conda-env", default="openfold3")
    parser.add_argument("--openfold-mode", default="score")
    args = parser.parse_args(argv)
    try:
        prediction.check_prediction_args(parser, args)
    except SystemExit as stop:
        return int(stop.code)
    return 0


class TestTheCommandLine:
    @pytest.mark.parametrize("model", MODELS)
    def test_a_model_may_be_run_without_a_directory(self, model):
        assert _usage_error(["--predictor", model]) == 0

    @pytest.mark.parametrize(
        "option, value",
        [
            ("--prediction-cyclic", "on"),
            ("--prediction-no-msa-server", None),
            ("--prediction-conda-env", "boltz"),
            ("--prediction-lock-threshold", "2.0"),
        ],
    )
    def test_the_settings_need_a_predictor(self, capsys, option, value):
        assert _usage_error([option, *([value] if value else [])]) == 2
        assert f"{option} needs --predictor" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "option, value",
        [
            ("--prediction-cyclic", "on"),
            ("--prediction-no-msa-server", None),
            ("--prediction-conda-env", "boltz"),
            ("--prediction-lock-threshold", "2.0"),
        ],
    )
    def test_the_settings_of_a_run_are_refused_with_a_directory(
        self, capsys, tmp_path, option, value
    ):
        argv = ["--predictor", "boltz2", "--prediction-dir", str(tmp_path), option]
        assert _usage_error(argv + ([value] if value else [])) == 2
        assert f"{option} cannot be combined with --prediction-dir" in capsys.readouterr().err

    def test_a_threshold_that_is_not_a_positive_number_is_refused_by_the_parser(self, capsys):
        with pytest.raises(SystemExit):
            _usage_error(["--predictor", "boltz2", "--prediction-lock-threshold", "-2"])
        assert "positive number of angstrom" in capsys.readouterr().err

    def test_the_mode_the_runner_does_not_run_is_a_usage_error_with_the_modes_it_runs(self, capsys):
        assert _usage_error(["--predictor", "protenix", "--prediction-mode", "score"]) == 2
        err = " ".join(capsys.readouterr().err.split())
        assert "--prediction-mode score cannot be run from here for Protenix" in err
        assert "its runner runs predict on a complex structure" in err

    def test_the_options_reach_the_pipeline(self, monkeypatch, tmp_path):
        captured = {}

        def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return {"sample_id": "x", "provenance": {}}

        monkeypatch.setattr(run, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-run",
                "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o"),
                "--predictor", "boltz2",
                "--prediction-mode", "score-lock",
                "--prediction-lock-threshold", "3.5",
                "--prediction-cyclic", "off",
                "--prediction-no-msa-server",
                "--prediction-conda-env", "boltz",
            ],
        )  # fmt: skip
        run.main()
        assert captured["prediction_mode"] == "score-lock"
        assert captured["prediction_lock_threshold"] == 3.5
        assert captured["prediction_cyclic"] == "off"
        assert captured["prediction_use_msa_server"] is False
        assert captured["prediction_conda_env"] == "boltz"
        assert (
            captured["openfold_cyclic"] == "auto" and captured["openfold_conda_env"] == "openfold3"
        )

    @pytest.mark.parametrize("model", MODELS)
    def test_the_command_runs_the_model_and_writes_the_results(
        self, monkeypatch, tmp_path, model, capsys
    ):
        import json

        stub = ModelStub(model, tmp_path, monkeypatch)
        out = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-run",
                "-i", str(EXAMPLE_1YCR), "-o", str(out),
                "--skip-prep", "--skip-relax", "--metrics", "openfold",
                "--peptide-chain", "B", "--receptor-chain", "A",
                "--predictor", model, "--prediction-no-msa-server",
            ],
        )  # fmt: skip
        run.main()
        capsys.readouterr()
        results = json.loads((out / f"{STEM}_results.json").read_text(encoding="utf-8"))
        block = results["prediction"]
        assert block["model"] == model and block["mode"] == default_mode_of(model)
        assert "error" not in block and stub.starts == 1


class TestInABatch:
    NAMES = ("a", "b")

    @pytest.fixture
    def samples(self, tmp_path):
        folder = tmp_path / "in"
        folder.mkdir()
        text = EXAMPLE_1YCR.read_text(encoding="utf-8")
        paths = []
        for name in self.NAMES:
            path = folder / f"{name}.pdb"
            path.write_text(f"REMARK   9 sample {name}\n{text}", encoding="utf-8")
            paths.append(path)
        return paths

    @staticmethod
    def batch_of(samples, out, **kwargs):
        kwargs.setdefault("skip_prep", True)
        kwargs.setdefault("skip_relax", True)
        kwargs.setdefault("metrics", {"openfold"})
        return batch.run_batch(samples, out, **kwargs)

    @pytest.mark.parametrize("model", MODELS)
    def test_every_sample_is_predicted_and_a_second_batch_starts_no_model(
        self, tmp_path, samples, monkeypatch, model
    ):
        stub = ModelStub(model, tmp_path, monkeypatch, names=list(self.NAMES))
        rows = self.batch_of(samples, tmp_path / "out", predictor=model)
        assert [row["sample_id"] for row in rows] == list(self.NAMES)
        for row in rows:
            assert row["batch_status"] == "ok", row.get("batch_failed_reasons")
            assert row["prediction_model"] == model
            assert row["prediction_mode"] == default_mode_of(model)
            assert finite(row["prediction_avg_plddt"]) and finite(row["prediction_evobind_score"])
            assert row["prediction_binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)
        started = stub.starts
        assert started >= 1
        again = self.batch_of(
            samples,
            tmp_path / "out2",
            predictor=model,
            prediction_cache=(tmp_path / "out" / "_predictions"),
        )
        assert stub.starts == started
        assert all(row["prediction_cache_runs"] == 0 for row in again)

    @pytest.mark.parametrize("model, mode", RUNNER_REFUSES)
    def test_a_mode_the_runner_does_not_run_is_refused_before_a_sample_starts(
        self, tmp_path, samples, monkeypatch, model, mode
    ):
        stub = ModelStub(model, tmp_path, monkeypatch, names=list(self.NAMES))
        with pytest.raises(ValueError, match="cannot be run from here"):
            self.batch_of(samples, tmp_path / "out", predictor=model, prediction_mode=mode)
        assert stub.starts == 0 and not (tmp_path / "out").exists()

    def test_the_generic_settings_reach_the_batched_runs(self, tmp_path, samples, monkeypatch):
        stub = ModelStub("boltz2", tmp_path, monkeypatch, names=list(self.NAMES))
        self.batch_of(
            samples,
            tmp_path / "out",
            predictor="boltz2",
            prediction_mode="score-lock",
            prediction_lock_threshold=4.0,
            prediction_use_msa_server=False,
            prediction_cyclic="off",
            prediction_conda_env="boltz",
            on_incompatible="warn",
        )
        assert stub.starts == len(self.NAMES)
        for call in stub.calls:
            assert "threshold: 4.0\n" in call["input"] and 'msa: "empty"' in call["input"]
            assert call["conda_env"] == "boltz"

    def test_colabfold_refuses_a_closure_setting_before_a_sample_starts(
        self, tmp_path, samples, monkeypatch
    ):
        stub = ModelStub("af2", tmp_path, monkeypatch, names=list(self.NAMES))
        with pytest.raises(ValueError, match="--prediction-cyclic is not available"):
            self.batch_of(samples, tmp_path / "out", predictor="af2", prediction_cyclic="on")
        assert stub.starts == 0

    def test_the_chain_map_of_colabfold_reaches_every_sample(self, tmp_path, samples, monkeypatch):
        renamed = [renamed_input(tmp_path / f"{n}.pdb", {"A": "R", "B": "L"}) for n in self.NAMES]
        stub = ModelStub("af2", tmp_path, monkeypatch, renamed[0], "R", "L", names=list(self.NAMES))
        rows = self.batch_of(
            renamed, tmp_path / "out", predictor="af2", peptide_chain="L", receptor_chain="R"
        )
        # the request of ColabFold holds the sequences and no file: the samples share one run
        assert stub.starts == 1
        for row in rows:
            assert row["batch_status"] == "ok", row.get("batch_failed_reasons")
            assert finite(row["prediction_evobind_score"])
            assert row["prediction_binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)
