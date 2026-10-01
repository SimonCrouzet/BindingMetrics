"""PredictionRunner: the abstract steps and the defaults of the optional ones."""

import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_run, openfold
from binding_metrics.metrics._openfold_run import (
    OpenFoldQueryError,
    OpenFoldRunError,
    UnmappableResidueError,
)
from binding_metrics.predictors import of3_runner
from binding_metrics.predictors.of3 import OpenFold3Parser
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import (
    PredictionFailedError,
    PredictionRequest,
    PredictionStore,
    PredictionUnavailableError,
)

P53 = Path(__file__).resolve().parents[2] / "data" / "example_linear_p53_1YCR.pdb"


class _Minimal(PredictionRunner):
    name = "minimal"

    def prepare(self, request, work_dir):
        return Path(work_dir) / "input.txt"

    def run(self, request, work_dir):
        return Path(work_dir)


class TestTheAbstractClass:
    def test_cannot_be_instantiated_without_prepare_and_run(self):
        class OnlyRun(PredictionRunner):
            name = "only_run"

            def run(self, request, work_dir):
                return Path(work_dir)

        with pytest.raises(TypeError, match="abstract"):
            OnlyRun()

    def test_a_subclass_with_the_two_steps_can_be_instantiated(self):
        assert _Minimal().name == "minimal"

    def test_no_capabilities_are_declared_by_default(self):
        """The pre-flight lane defines the class later; a runner declares None until then."""
        assert PredictionRunner.capabilities is None
        assert _Minimal.capabilities is None

    def test_a_runner_may_declare_capabilities_and_a_sibling_is_unaffected(self):
        marker = object()

        class Constrained(_Minimal):
            capabilities = marker

        assert Constrained.capabilities is marker
        assert _Minimal.capabilities is None


class TestTheOptionalSteps:
    def test_a_runner_is_available_and_has_no_version_by_default(self):
        runner = _Minimal()
        assert runner.is_available() is True
        assert runner.version() is None

    def test_no_request_can_be_batched_by_default(self):
        assert _Minimal().supports_batch(object()) is False

    def test_run_many_says_that_the_runner_has_no_batched_mode(self):
        with pytest.raises(NotImplementedError, match="minimal runner has no batched mode"):
            _Minimal().run_many([], Path("."))

    def test_the_message_survives_a_runner_without_a_name(self):
        class Nameless(PredictionRunner):
            def prepare(self, request, work_dir):
                return Path(work_dir)

            def run(self, request, work_dir):
                return Path(work_dir)

        with pytest.raises(NotImplementedError, match="Nameless runner"):
            Nameless().run_many([], Path("."))


class TestModesAndTheChainMap:
    def test_a_runner_runs_predict_only_unless_it_says_more(self):
        assert PredictionRunner.supported_modes == frozenset({"predict"})
        assert _Minimal.supported_modes == frozenset({"predict"})

    def test_the_default_mode_is_left_to_make_request_unless_the_runner_names_one(self):
        assert PredictionRunner.default_mode is None
        assert _Minimal.default_mode is None

    def test_a_runner_may_declare_its_modes_and_a_sibling_is_unaffected(self):
        class Templated(_Minimal):
            supported_modes = frozenset({"score", "refold"})
            default_mode = "score"

        assert Templated.supported_modes == {"score", "refold"}
        assert Templated.default_mode == "score"
        assert _Minimal.supported_modes == frozenset({"predict"})

    def test_the_prediction_keeps_the_chain_ids_of_the_input_by_default(self):
        assert _Minimal().output_chain_map(object()) is None

    def test_a_runner_may_name_the_chains_of_its_prediction(self):
        class Renaming(_Minimal):
            def output_chain_map(self, request):
                return {"A": request.receptor_chain, "B": request.binder_chain}

        request = type("Request", (), {"receptor_chain": "R", "binder_chain": "L"})()
        assert Renaming().output_chain_map(request) == {"A": "R", "B": "L"}

    def test_openfold3_runs_score_and_refold_on_a_structure_and_score_by_default(self):
        """Mode predict takes a query file of the caller's, score-lock cannot be run."""
        assert OpenFold3Runner.supported_modes == frozenset({"score", "refold"})
        assert OpenFold3Runner.default_mode == "score"
        assert OpenFold3Runner().output_chain_map(object()) is None

    def test_the_default_mode_of_openfold3_is_the_default_of_make_request(self):
        parameter = inspect.signature(OpenFold3Runner.make_request).parameters["mode"]
        assert parameter.default == OpenFold3Runner.default_mode


# ---------------------------------------------------------------------------- OpenFold3Runner


@pytest.fixture(autouse=True)
def of3_environment(tmp_path, monkeypatch):
    """No OpenFold3 and no user-default runner.yml on this machine, whatever the machine has."""
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "openfold_cache"))
    versions = []

    def fake_version(python_cmd=None):
        versions.append(python_cmd)
        return "0.5.0"

    monkeypatch.setattr(_openfold_run, "installed_openfold3_version", fake_version)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *args, **kwargs: (
            "/fake/run_openfold" if name == "run_openfold" else real_which(name, *args, **kwargs)
        ),
    )
    return versions


def write_of3_output(predictions, name, *, seed=42, samples=1):
    """The smallest OpenFold3 output the parser accepts as the output of a query."""
    seed_dir = Path(predictions) / name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for sample in range(1, samples + 1):
        prefix = f"{name}_seed_{seed}_sample_{sample}"
        (seed_dir / f"{prefix}_confidences_aggregated.json").write_text(
            json.dumps({"avg_plddt": 81.5, "ptm": 0.77, "iptm": 0.66, "gpde": 1.5}),
            encoding="utf-8",
        )
    return Path(predictions)


def complex_file(tmp_path, name="c", content=b"data_x\n"):
    path = tmp_path / "inputs" / f"{name}.cif"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return path


def score_request(tmp_path, runner=None, **kwargs):
    runner = runner or OpenFold3Runner()
    kwargs.setdefault("binder_chain", "B")
    kwargs.setdefault("receptor_chain", "A")
    return runner.make_request(
        complex_file(tmp_path, content=kwargs.pop("content", b"data_x\n")),
        name=kwargs.pop("name", "q"),
        **kwargs,
    )


class TestMakeRequest:
    def test_every_default_is_written_out(self, tmp_path):
        request = score_request(tmp_path)
        assert (request.model, request.name, request.mode) == ("of3", "q", "score")
        assert request.model_version == "0.5.0"
        assert request.seeds == (42,) and request.num_samples == 5
        assert (request.binder_chain, request.receptor_chain) == ("B", "A")
        assert request.options == {
            "presets": ["predict", "low_mem"],
            "use_msa_server": True,
            "num_model_seeds": None,
            "on_unmappable_residue": "error",
            "binder_cyclic": "auto",
            "template_mode": "alignment",
            "query_builder_version": _openfold_run.QUERY_BUILDER_VERSION,
            "extra_args": [],
            "inference_ckpt_path": None,
            "inference_ckpt_size_bytes": None,
        }
        assert set(request.extra_files) == set()

    def test_the_same_run_gives_the_same_key(self, tmp_path):
        assert score_request(tmp_path).key() == score_request(tmp_path).key()

    @pytest.mark.parametrize(
        "change",
        [
            {"seeds": (7, 8)},
            {"num_samples": 3},
            {"presets": ["predict"]},
            {"use_msa_server": False},
            {"num_model_seeds": 2},
            {"on_unmappable_residue": "x"},
            {"extra_args": ["--data_seed=1"]},
            {"inference_ckpt_path": "/weights/other.pt"},
            {"template_mode": "structure"},
            {"mode": "refold"},
            {"binder_chain": "P"},
            {"content": b"another complex"},
            {"name": "renamed"},
        ],
        ids=lambda change: next(iter(change)),
    )
    def test_a_setting_that_changes_the_output_changes_the_key(self, tmp_path, change):
        same_name = change.keys() == {"name"}
        assert (score_request(tmp_path, **change).key() != score_request(tmp_path).key()) is (
            not same_name
        )

    def test_predict_is_added_to_the_presets_and_the_order_is_kept(self, tmp_path):
        assert score_request(tmp_path, presets=["low_mem"]).options["presets"] == [
            "predict",
            "low_mem",
        ]
        assert score_request(tmp_path, presets=["low_mem"]).key() == score_request(tmp_path).key()

    def test_a_runner_yaml_replaces_the_presets_and_is_hashed_by_content(self, tmp_path):
        config = tmp_path / "runner.yml"
        config.write_text("model_update:\n  presets: [predict]\n", encoding="utf-8")
        first = score_request(tmp_path, runner_yaml=config, presets=["predict"])
        assert first.options["presets"] is None
        assert first.key() == score_request(tmp_path, runner_yaml=config).key()
        config.write_text("model_update:\n  presets: [predict, low_mem]\n", encoding="utf-8")
        assert score_request(tmp_path, runner_yaml=config).key() != first.key()

    def test_a_template_file_is_hashed_by_content(self, tmp_path):
        template = tmp_path / "relaxed.cif"
        template.write_bytes(b"one")
        first = score_request(tmp_path, template_cif_path=template)
        template.write_bytes(b"two")
        assert score_request(tmp_path, template_cif_path=template).key() != first.key()
        assert first.key() != score_request(tmp_path).key()

    def test_the_user_default_runner_yaml_is_part_of_the_key_when_it_exists(
        self, tmp_path, monkeypatch
    ):
        without = score_request(tmp_path)
        cache = tmp_path / "openfold_cache"
        cache.mkdir()
        (cache / "runner.yml").write_text("msa_computation_settings: {}\n", encoding="utf-8")
        first = score_request(tmp_path)
        assert first.key() != without.key() and "user_default_runner_yaml" in first.extra_files
        (cache / "runner.yml").write_text("seeds: [1]\n", encoding="utf-8")
        assert score_request(tmp_path).key() != first.key()

    def test_the_size_of_the_checkpoint_is_recorded(self, tmp_path):
        checkpoint = tmp_path / "weights.pt"
        checkpoint.write_bytes(b"x" * 1234)
        request = score_request(tmp_path, inference_ckpt_path=checkpoint)
        assert request.options["inference_ckpt_path"] == str(checkpoint)
        assert request.options["inference_ckpt_size_bytes"] == 1234
        checkpoint.write_bytes(b"x" * 999)
        assert score_request(tmp_path, inference_ckpt_path=checkpoint).key() != request.key()

    def test_a_checkpoint_that_is_not_there_yet_is_recorded_by_path(self, tmp_path):
        request = score_request(tmp_path, inference_ckpt_path=tmp_path / "later.pt")
        assert request.options["inference_ckpt_size_bytes"] is None

    def test_another_openfold3_version_is_another_key(self, tmp_path, monkeypatch):
        first = score_request(tmp_path)
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.4.1"
        )
        second = score_request(tmp_path)
        assert second.model_version == "0.4.1" and second.key() != first.key()

    def test_an_unknown_version_is_an_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: None
        )
        assert score_request(tmp_path).model_version == ""

    def test_the_conda_environment_is_not_part_of_the_key(self, tmp_path):
        assert (
            score_request(tmp_path, OpenFold3Runner(conda_env="of3")).key()
            == score_request(tmp_path).key()
        )

    def test_predict_takes_a_query_file_and_no_chains(self, tmp_path):
        query = tmp_path / "query.json"
        query.write_text('{"queries": {}}', encoding="utf-8")
        request = OpenFold3Runner().make_request(query, name="q", mode="predict")
        assert request.mode == "predict" and request.binder_chain is None

    def test_a_score_or_refold_request_needs_both_chains(self, tmp_path):
        for mode in ("score", "refold"):
            with pytest.raises(ValueError, match="needs binder_chain and receptor_chain"):
                OpenFold3Runner().make_request(complex_file(tmp_path), name="q", mode=mode)

    def test_a_query_file_cannot_take_a_template(self, tmp_path):
        with pytest.raises(ValueError, match="names its own templates"):
            OpenFold3Runner().make_request(
                complex_file(tmp_path), name="q", mode="predict", template_cif_path=tmp_path
            )

    def test_an_unknown_unmappable_choice_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="on_unmappable_residue"):
            score_request(tmp_path, on_unmappable_residue="skip")

    def test_the_mode_is_checked_by_the_request(self, tmp_path):
        with pytest.raises(ValueError, match="mode must be one of"):
            score_request(tmp_path, mode="dock")

    def test_the_version_is_asked_once_and_through_conda_when_an_environment_is_set(
        self, tmp_path, of3_environment
    ):
        runner = OpenFold3Runner(conda_env="of3")
        assert runner.version() == "0.5.0" and runner.version() == "0.5.0"
        score_request(tmp_path, runner)
        assert of3_environment == [["conda", "run", "-n", "of3", "python"]]

    def test_the_current_interpreter_is_asked_without_an_environment(self, of3_environment):
        OpenFold3Runner().version()
        assert of3_environment == [None]


class TestTheDefaultsMirrorTheRunFunctions:
    """The runner leaves out arguments that have their default, so the defaults must match."""

    @pytest.mark.parametrize(
        "function",
        [openfold.run_openfold, openfold.run_openfold_scoring, openfold.run_openfold_refolding],
    )
    def test_the_signature_defaults(self, function):
        parameters = inspect.signature(function).parameters
        assert parameters["num_diffusion_samples"].default == of3_runner._DEFAULT_NUM_SAMPLES
        assert parameters["num_model_seeds"].default == of3_runner._DEFAULT_NUM_MODEL_SEEDS
        assert parameters["use_msa_server"].default is of3_runner._DEFAULT_USE_MSA_SERVER
        if "on_unmappable_residue" in parameters:
            assert parameters["on_unmappable_residue"].default == of3_runner._DEFAULT_ON_UNMAPPABLE
        if "binder_cyclic" in parameters:
            assert parameters["binder_cyclic"].default == of3_runner._DEFAULT_BINDER_CYCLIC
        if "template_mode" in parameters:
            assert parameters["template_mode"].default == of3_runner._DEFAULT_TEMPLATE_MODE
            assert of3_runner._DEFAULT_TEMPLATE_MODE == _openfold_run.TEMPLATE_MODES[0]

    def test_the_unmappable_choices(self):
        assert of3_runner._ON_UNMAPPABLE_CHOICES == _openfold_run._ON_UNMAPPABLE_CHOICES


class RecordingWrappers:
    """Patches the openfold run functions with fakes that record their keyword arguments."""

    def __init__(self, monkeypatch, *, write_output=True, batch_names_without_output=()):
        self.calls = []
        self.write_output = write_output
        self.without_output = set(batch_names_without_output)
        for name in ("run_openfold_scoring", "run_openfold_refolding"):
            monkeypatch.setattr(openfold, name, self._single(name))
        monkeypatch.setattr(openfold, "run_openfold", self._predict)
        monkeypatch.setattr(openfold, "run_openfold_batched", self._batched)

    def _single(self, function):
        def fake(**kwargs):
            self.calls.append((function, kwargs))
            predictions = Path(kwargs["output_dir"]) / "predictions"
            if self.write_output:
                write_of3_output(predictions, kwargs["query_name"])
            predictions.mkdir(parents=True, exist_ok=True)
            return predictions

        return fake

    def _predict(self, **kwargs):
        self.calls.append(("run_openfold", kwargs))
        predictions = Path(kwargs["output_dir"])
        query = json.loads(Path(kwargs["query_json"]).read_text(encoding="utf-8"))
        for name in query["queries"]:
            write_of3_output(predictions, name)
        return predictions

    def _batched(self, **kwargs):
        self.calls.append(("run_openfold_batched", kwargs))
        predictions = Path(kwargs["output_dir"]) / "predictions"
        predictions.mkdir(parents=True, exist_ok=True)
        (predictions / "experiment_config.json").write_text(
            json.dumps({"inference_ckpt_name": "fake-ckpt"}), encoding="utf-8"
        )
        for sample in kwargs["samples"]:
            if sample.query_name not in self.without_output:
                write_of3_output(predictions, sample.query_name)
        return predictions


class TestRun:
    def test_a_default_score_run_makes_the_call_the_pipeline_makes_today(
        self, tmp_path, monkeypatch
    ):
        wrappers = RecordingWrappers(monkeypatch)
        request = score_request(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        predictions = OpenFold3Runner().run(request, work)
        assert predictions == work / "predictions"
        assert wrappers.calls == [
            (
                "run_openfold_scoring",
                {
                    "complex_structure_path": request.input_path,
                    "receptor_chain": "A",
                    "binder_chain": "B",
                    "query_name": "q",
                    "output_dir": work,
                },
            )
        ]

    def test_refold_calls_the_refolding_wrapper(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        (tmp_path / "w").mkdir()
        OpenFold3Runner().run(score_request(tmp_path, mode="refold"), tmp_path / "w")
        assert [name for name, _ in wrappers.calls] == ["run_openfold_refolding"]

    def test_every_setting_that_is_not_a_default_reaches_the_wrapper(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        config = tmp_path / "runner.yml"
        config.write_text("model_update: {}\n", encoding="utf-8")
        template = tmp_path / "relaxed.cif"
        template.write_bytes(b"t")
        runner = OpenFold3Runner(conda_env="of3")
        request = score_request(
            tmp_path,
            runner,
            seeds=(7, 8),
            num_samples=3,
            use_msa_server=False,
            on_unmappable_residue="x",
            extra_args=["--a=1"],
            inference_ckpt_path="/weights/x.pt",
            runner_yaml=config,
            template_cif_path=template,
        )
        (tmp_path / "w").mkdir()
        runner.run(request, tmp_path / "w")
        ((_, kwargs),) = wrappers.calls
        assert kwargs["conda_env"] == "of3" and kwargs["seeds"] == (7, 8)
        assert kwargs["num_diffusion_samples"] == 3 and "num_model_seeds" not in kwargs
        assert kwargs["use_msa_server"] is False and kwargs["on_unmappable_residue"] == "x"
        assert kwargs["extra_args"] == ["--a=1"]
        assert kwargs["inference_ckpt_path"] == "/weights/x.pt"
        assert kwargs["runner_yaml"] == config and kwargs["template_cif_path"] == template
        assert "model_presets" not in kwargs  # the runner YAML replaces them

    def test_generated_seeds_reach_the_wrapper_without_seed_values(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        (tmp_path / "w").mkdir()
        OpenFold3Runner().run(score_request(tmp_path, num_model_seeds=2), tmp_path / "w")
        ((_, kwargs),) = wrappers.calls
        assert kwargs["num_model_seeds"] == 2 and "seeds" not in kwargs

    def test_other_presets_are_passed(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        (tmp_path / "w").mkdir()
        OpenFold3Runner().run(score_request(tmp_path, presets=["predict"]), tmp_path / "w")
        assert wrappers.calls[0][1]["model_presets"] == ["predict"]

    def test_predict_runs_a_query_file(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        query = tmp_path / "query.json"
        query.write_text(json.dumps({"seeds": [1], "queries": {"q": {}}}), encoding="utf-8")
        runner = OpenFold3Runner()
        request = runner.make_request(query, name="q", mode="predict", num_samples=2)
        (tmp_path / "w").mkdir()
        predictions = runner.run(request, tmp_path / "w")
        assert predictions == tmp_path / "w" / "predictions"
        ((_, kwargs),) = wrappers.calls
        assert kwargs == {
            "query_json": query,
            "output_dir": tmp_path / "w" / "predictions",
            "num_diffusion_samples": 2,
        }

    def test_the_functions_are_looked_up_when_the_run_starts(self, tmp_path, monkeypatch):
        runner = OpenFold3Runner()
        request = score_request(tmp_path)
        wrappers = RecordingWrappers(monkeypatch)  # patched after the runner was made
        (tmp_path / "w").mkdir()
        runner.run(request, tmp_path / "w")
        assert len(wrappers.calls) == 1

    def test_a_run_that_wrote_nothing_is_an_error(self, tmp_path, monkeypatch):
        RecordingWrappers(monkeypatch, write_output=False)
        (tmp_path / "w").mkdir()
        with pytest.raises(RuntimeError, match="wrote no output for query 'q'"):
            OpenFold3Runner().run(score_request(tmp_path), tmp_path / "w")

    def test_the_recorded_reason_holds_no_path_of_the_work_directory(self, tmp_path, monkeypatch):
        """The store renames its temporary work directory, so a path in a reason points nowhere."""
        RecordingWrappers(monkeypatch, write_output=False)
        store = PredictionStore(tmp_path / "store")
        (entry,) = store.run_missing([score_request(tmp_path)], OpenFold3Runner())
        assert entry.status == "failed"
        assert "wrote no output for query 'q'" in entry.reason
        assert ".tmp-" not in entry.reason and str(tmp_path) not in entry.reason
        assert "outputs/predictions of a stored entry" in entry.reason

    def test_the_reason_openfold_logged_is_in_the_error(self, tmp_path, monkeypatch):
        def failing(**kwargs):
            predictions = Path(kwargs["output_dir"]) / "predictions"
            (predictions / "logs").mkdir(parents=True)
            (predictions / "summary.txt").write_text(
                "Total Queries Processed: 1\n  - Successful Queries:  0\n"
                "  - Failed Queries:      1\n\nFailed Queries: q\n",
                encoding="utf-8",
            )
            (predictions / "logs" / "predict_err_rank0.log").write_text(
                "Query ID(s): q\nError Type: OutOfMemoryError\nError Message: CUDA out of memory\n"
                + "-" * 50
                + "\nTraceback:x",
                encoding="utf-8",
            )
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_scoring", failing)
        (tmp_path / "w").mkdir()
        with pytest.raises(RuntimeError, match="OutOfMemoryError: CUDA out of memory"):
            OpenFold3Runner().run(score_request(tmp_path), tmp_path / "w")

    def test_a_request_for_another_model_is_refused(self, tmp_path):
        request = PredictionRequest("boltz2", "q", sequences={"A": "GG"})
        with pytest.raises(ValueError, match="cannot run a 'boltz2' request"):
            OpenFold3Runner().run(request, tmp_path)

    def test_prepare_returns_the_query_and_starts_nothing(self, tmp_path, monkeypatch):
        seen = {}

        def fake_prepare(**kwargs):
            seen.update(kwargs)
            return Path(kwargs["output_dir"]) / "q_query.json"

        monkeypatch.setattr(openfold, "prepare_scoring_query", fake_prepare)
        monkeypatch.setattr(
            openfold, "run_openfold", lambda **kw: pytest.fail("prepare must not run the model")
        )
        request = score_request(tmp_path, seeds=(3,))
        query = OpenFold3Runner().prepare(request, tmp_path / "w")
        assert query == tmp_path / "w" / "query" / "q_query.json"
        # seeds are not part of the query file; the run writes them to the runner YAML
        assert "seeds" not in seen and seen["query_name"] == "q"

    def test_prepare_raises_the_input_errors_of_a_run(self, tmp_path):
        pytest.importorskip("gemmi")
        request = OpenFold3Runner().make_request(
            P53, name="q", binder_chain="B", receptor_chain="Z"
        )
        with pytest.raises(ValueError, match="Z"):
            OpenFold3Runner().prepare(request, tmp_path / "w")

    def test_prepare_of_a_query_file_is_the_file(self, tmp_path):
        query = tmp_path / "query.json"
        query.write_text("{}", encoding="utf-8")
        runner = OpenFold3Runner()
        request = runner.make_request(query, name="q", mode="predict")
        assert runner.prepare(request, tmp_path / "w") == query


class TestAvailability:
    def test_run_openfold_on_path(self, monkeypatch):
        monkeypatch.setattr(of3_runner.shutil, "which", lambda name: "/bin/run_openfold")
        assert OpenFold3Runner().is_available() is True
        monkeypatch.setattr(of3_runner.shutil, "which", lambda name: None)
        assert OpenFold3Runner().is_available() is False

    def test_a_conda_environment_needs_openfold3_in_it(self, monkeypatch):
        monkeypatch.setattr(of3_runner.shutil, "which", lambda name: None)  # not consulted
        assert OpenFold3Runner(conda_env="of3").is_available() is True
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: None
        )
        assert OpenFold3Runner(conda_env="of3").is_available() is False

    def test_the_capabilities_are_undeclared_for_now(self):
        assert OpenFold3Runner.capabilities is None
        assert issubclass(OpenFold3Runner, PredictionRunner) and OpenFold3Runner.name == "of3"


# ---------------------------------------------------------------------------- with the store


class TestThroughTheStore:
    def test_the_model_runs_once_for_two_metrics_and_the_parser_reads_the_result(
        self, tmp_path, monkeypatch
    ):
        wrappers = RecordingWrappers(monkeypatch)
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = score_request(tmp_path, runner)
        first = store.get_or_run(request, runner)
        second = store.get_or_run(score_request(tmp_path, runner), runner)
        assert len(wrappers.calls) == 1 and first.run_id == second.run_id
        assert first.prediction_dir == first.directory / "outputs" / "predictions"
        record = OpenFold3Parser().load(second.prediction_dir, second.name)
        assert (record.avg_plddt, record.ptm, record.iptm) == (81.5, 0.77, 0.66)
        assert first.runner_name == "of3" and first.runner_version == "0.5.0"

    def test_a_request_without_a_version_is_stored_under_the_installed_one(
        self, tmp_path, monkeypatch
    ):
        RecordingWrappers(monkeypatch)
        store = PredictionStore(tmp_path / "store")
        request = score_request(tmp_path)
        entry = store.get_or_run(request.with_model_version(""), OpenFold3Runner())
        assert entry.key == request.key()

    @pytest.mark.parametrize(
        "error, expected",
        [
            (
                OpenFoldRunError(
                    1, ["run_openfold"], "ValueError: cowardly refusing to perform inference"
                ),
                "OpenFoldRunError: OpenFold3 exited with status 1: ValueError: cowardly refusing",
            ),
            (
                OpenFoldQueryError(Path("/out"), {"q": "OutOfMemoryError: CUDA out of memory"}),
                "OpenFoldQueryError: OpenFold3 exited normally but failed on every query",
            ),
            (
                UnmappableResidueError([("", "B", ["XYZ 5"])]),
                "UnmappableResidueError: OpenFold3 cannot take these residues",
            ),
            (
                FileNotFoundError("run_openfold not found on PATH. Pass conda_env='openfold3'"),
                "FileNotFoundError: run_openfold not found on PATH",
            ),
        ],
        ids=["run-error", "query-error", "unmappable", "not-installed"],
    )
    def test_a_failed_run_is_recorded_with_the_exception_text_and_not_retried(
        self, tmp_path, monkeypatch, error, expected
    ):
        calls = []

        def failing(**kwargs):
            calls.append(kwargs)
            raise error

        monkeypatch.setattr(openfold, "run_openfold_scoring", failing)
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = score_request(tmp_path, runner)
        with pytest.raises(PredictionFailedError) as info:
            store.get_or_run(request, runner)
        assert info.value.reason.startswith(expected) and info.value.__cause__ is error
        assert store.lookup(request).reason.startswith(expected)
        with pytest.raises(PredictionFailedError, match=expected.split(":")[0]):
            store.get_or_run(request, runner)
        assert len(calls) == 1

    def test_the_hint_of_a_known_failure_is_kept_in_the_reason(self, tmp_path, monkeypatch):
        def failing(**kwargs):
            raise OpenFoldRunError(
                1, ["run_openfold"], "ValueError: Default checkpoint x not found, cowardly refusing"
            )

        monkeypatch.setattr(openfold, "run_openfold_scoring", failing)
        store = PredictionStore(tmp_path / "store")
        entry = store.ensure(score_request(tmp_path), OpenFold3Runner())
        assert "setup_openfold --non-interactive" in entry.reason

    def test_rerun_retries_a_failed_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            openfold, "run_openfold_scoring", lambda **kw: (_ for _ in ()).throw(OSError("full"))
        )
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = score_request(tmp_path, runner)
        assert store.ensure(request, runner).status == "failed"
        RecordingWrappers(monkeypatch)
        assert store.get_or_run(request, runner, rerun=True).status == "done"

    def test_openfold3_that_is_not_installed_records_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(of3_runner.shutil, "which", lambda name: None)
        wrappers = RecordingWrappers(monkeypatch)
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = score_request(tmp_path, runner)
        with pytest.raises(PredictionUnavailableError, match="cannot be started"):
            store.get_or_run(request, runner)
        assert wrappers.calls == [] and store.lookup(request) is None
        assert not store.path_for(request).exists()

    def test_adopted_openfold3_output_is_parsed_and_never_run(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        outputs = write_of3_output(tmp_path / "my_predictions", "q")
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = score_request(tmp_path, runner).with_model_version("")
        store.adopt(request, outputs)
        entry = store.get_or_run(request, runner)
        assert wrappers.calls == [] and entry.status == "adopted"
        assert OpenFold3Parser().load(entry.prediction_dir, entry.name).iptm == 0.66


class TestBatches:
    def _requests(self, tmp_path, names, **kwargs):
        runner = OpenFold3Runner()
        return [
            score_request(tmp_path, runner, name=name, content=name.encode(), **kwargs)
            for name in names
        ]

    def test_which_requests_can_be_batched(self, tmp_path):
        runner = OpenFold3Runner()
        template = tmp_path / "t.cif"
        template.write_bytes(b"t")
        query = tmp_path / "query.json"
        query.write_text("{}", encoding="utf-8")
        assert runner.supports_batch(score_request(tmp_path)) is True
        assert runner.supports_batch(score_request(tmp_path, mode="refold")) is True
        assert runner.supports_batch(score_request(tmp_path, template_cif_path=template)) is False
        assert runner.supports_batch(runner.make_request(query, name="q", mode="predict")) is False

    def test_one_call_runs_every_query_and_each_gets_its_own_folder(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        runner = OpenFold3Runner(conda_env="of3")
        requests = self._requests(tmp_path, ["a", "b"], seeds=(5,))
        work = tmp_path / "work"
        work.mkdir()
        results = runner.run_many(requests, work)
        ((function, kwargs),) = wrappers.calls
        assert function == "run_openfold_batched"
        assert [(s.query_name, s.receptor_chain, s.binder_chain) for s in kwargs["samples"]] == [
            ("a", "A", "B"),
            ("b", "A", "B"),
        ]
        assert kwargs["samples"][0].complex_structure_path == requests[0].input_path
        assert kwargs["output_dir"] == work and kwargs["mode"] == "score"
        assert kwargs["conda_env"] == "of3" and kwargs["seeds"] == (5,)

        assert sorted(results) == sorted(r.key() for r in requests)
        for request in requests:
            folder = results[request.key()]
            assert folder == work / "split" / request.key()
            assert {p.name for p in folder.iterdir()} == {"experiment_config.json", request.name}
            record = OpenFold3Parser().load(folder, request.name)
            assert record.ptm == 0.77 and record.extras["inference_ckpt_name"] == "fake-ckpt"
        assert not (results[requests[0].key()] / "b").exists()  # nothing of the other query

    def test_the_files_that_say_what_became_of_the_templates_go_with_each_query(
        self, tmp_path, monkeypatch
    ):
        wrappers = RecordingWrappers(monkeypatch)
        recorded = openfold.run_openfold_batched

        def batched(**kwargs):
            predictions = recorded(**kwargs)
            for name in ("inference_query_set.json", "template_accounting.json"):
                (predictions / name).write_text("{}", encoding="utf-8")
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_batched", batched)
        requests = self._requests(tmp_path, ["a", "b"])
        work = tmp_path / "work"
        work.mkdir()
        results = OpenFold3Runner().run_many(requests, work)
        assert len(wrappers.calls) == 1
        for request in requests:
            assert {p.name for p in results[request.key()].iterdir()} == {
                "experiment_config.json",
                "inference_query_set.json",
                "template_accounting.json",
                request.name,
            }

    def test_refold_is_a_mode_of_the_batched_call(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        (tmp_path / "w").mkdir()
        OpenFold3Runner().run_many(
            self._requests(tmp_path, ["a", "b"], mode="refold"), tmp_path / "w"
        )
        assert wrappers.calls[0][1]["mode"] == "refold"

    def test_a_query_without_output_is_an_exception_with_the_logged_reason(
        self, tmp_path, monkeypatch
    ):
        RecordingWrappers(monkeypatch, batch_names_without_output=["b"])
        requests = self._requests(tmp_path, ["a", "b", "c"])
        (tmp_path / "w").mkdir()
        results = OpenFold3Runner().run_many(requests, tmp_path / "w")
        assert isinstance(results[requests[0].key()], Path)
        assert isinstance(results[requests[1].key()], RuntimeError)
        assert "wrote no output for query 'b'" in str(results[requests[1].key()])
        assert isinstance(results[requests[2].key()], Path)

    def test_a_batch_that_cannot_be_formed_is_refused(self, tmp_path):
        runner = OpenFold3Runner()
        (tmp_path / "w").mkdir()
        mixed = self._requests(tmp_path, ["a"]) + self._requests(tmp_path, ["b"], mode="refold")
        with pytest.raises(ValueError, match="cannot share a batch"):
            runner.run_many(mixed, tmp_path / "w")
        twins = [
            score_request(tmp_path, runner, name="same", content=b"1"),
            score_request(tmp_path, runner, name="same", content=b"2"),
        ]
        with pytest.raises(ValueError, match="names of a batch must be distinct"):
            runner.run_many(twins, tmp_path / "w")
        template = tmp_path / "t.cif"
        template.write_bytes(b"t")
        with pytest.raises(ValueError, match="cannot be batched"):
            runner.run_many(
                [score_request(tmp_path, runner, template_cif_path=template)], tmp_path / "w"
            )

    def test_nothing_to_run_is_an_empty_result(self, tmp_path):
        assert OpenFold3Runner().run_many([], tmp_path) == {}

    def test_the_store_batches_only_the_missing_requests(self, tmp_path, monkeypatch):
        wrappers = RecordingWrappers(monkeypatch)
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        requests = self._requests(tmp_path, ["a", "b", "c", "d"])
        store.get_or_run(requests[0], runner)
        wrappers.calls.clear()
        entries = store.run_missing(requests, runner)
        assert [name for name, _ in wrappers.calls] == ["run_openfold_batched"]
        assert [s.query_name for s in wrappers.calls[0][1]["samples"]] == ["b", "c", "d"]
        assert [e.status for e in entries] == ["done"] * 4
        parser = OpenFold3Parser()
        for entry in entries:
            assert parser.load(entry.prediction_dir, entry.name).avg_plddt == 81.5

    def test_a_query_that_failed_in_the_batch_is_recorded_with_its_reason(
        self, tmp_path, monkeypatch
    ):
        RecordingWrappers(monkeypatch, batch_names_without_output=["b"])
        store = PredictionStore(tmp_path / "store")
        requests = self._requests(tmp_path, ["a", "b"])
        entries = store.run_missing(requests, OpenFold3Runner())
        assert [e.status for e in entries] == ["done", "failed"]
        assert "wrote no output for query 'b'" in entries[1].reason

    def test_a_failed_batch_is_recorded_for_every_query(self, tmp_path, monkeypatch):
        def failing(**kwargs):
            raise OpenFoldRunError(
                1, ["run_openfold"], "torch.OutOfMemoryError: CUDA out of memory"
            )

        monkeypatch.setattr(openfold, "run_openfold_batched", failing)
        store = PredictionStore(tmp_path / "store")
        entries = store.run_missing(self._requests(tmp_path, ["a", "b"]), OpenFold3Runner())
        assert {e.status for e in entries} == {"failed"}
        assert all("CUDA out of memory" in e.reason for e in entries)


# ---------------------------------------------------------------------------- a stub run_openfold

STUB_SCRIPT = """#!{python}
import json, pathlib, sys
argv = sys.argv[1:]
with open({record!r}, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\\n")
if {fail!r}:
    sys.stderr.write("Traceback (most recent call last):\\n")
    sys.stderr.write("ValueError: Default checkpoint openbind-2025-06-30-174k not found in /w, "
                     "cowardly refusing to perform inference.\\n")
    sys.exit(1)
args = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
query = json.loads(pathlib.Path(args["query_json"]).read_text(encoding="utf-8"))
for name in query["queries"]:
    seed_dir = pathlib.Path(args["output_dir"]) / name / "seed_42"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / (name + "_seed_42_sample_1_confidences_aggregated.json")).write_text(
        json.dumps({{"avg_plddt": 77.0, "ptm": 0.5, "iptm": 0.25}}), encoding="utf-8")
"""


@pytest.fixture
def stub_openfold(tmp_path, monkeypatch):
    """A ``run_openfold`` on PATH that writes the smallest output; returns its argv log."""
    pytest.importorskip("gemmi")
    record = tmp_path / "argv.jsonl"

    def install(fail=False):
        script = tmp_path / "bin" / "run_openfold"
        script.parent.mkdir(exist_ok=True)
        script.write_text(
            STUB_SCRIPT.format(python=sys.executable, record=str(record), fail=fail),
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")

    def calls():
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]

    install()
    return type("Stub", (), {"install": staticmethod(install), "calls": staticmethod(calls)})


class TestWithAStubExecutable:
    """The real run functions and query builders, with a fake ``run_openfold`` on PATH."""

    def test_score_run_stored_and_parsed(self, tmp_path, stub_openfold):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = runner.make_request(
            P53, name="p53", binder_chain="B", receptor_chain="A", num_samples=2, seeds=(1, 2)
        )
        entry = store.get_or_run(request, runner)
        store.get_or_run(request, runner)  # a second metric: no second process
        (argv,) = stub_openfold.calls()
        assert argv[0] == "predict" and argv[1].startswith("--query_json=")
        assert argv[1].endswith("/outputs/query/p53_query.json")
        assert "--num_diffusion_samples=2" in argv
        query = json.loads(
            (entry.directory / "outputs" / "query" / "p53_query.json").read_text(encoding="utf-8")
        )
        assert "seeds" not in query and list(query["queries"]) == ["p53"]
        assert not any(a.startswith("--num_model_seeds") for a in argv)
        yaml = pytest.importorskip("yaml")
        runner_yaml = entry.directory / "outputs" / "predictions" / "runner_config.yaml"
        config = yaml.safe_load(runner_yaml.read_text(encoding="utf-8"))
        assert config["experiment_settings"] == {"seeds": [1, 2]}
        record = OpenFold3Parser().load(entry.prediction_dir, "p53")
        assert (record.avg_plddt, record.ptm, record.iptm) == (77.0, 0.5, 0.25)

    def test_a_process_that_fails_is_recorded_with_the_hint_and_not_run_again(
        self, tmp_path, stub_openfold
    ):
        stub_openfold.install(fail=True)
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = runner.make_request(P53, name="p53", binder_chain="B", receptor_chain="A")
        with pytest.raises(PredictionFailedError, match="cowardly refusing") as info:
            store.get_or_run(request, runner)
        assert isinstance(info.value.__cause__, OpenFoldRunError)
        assert "setup_openfold --non-interactive" in info.value.reason
        with pytest.raises(PredictionFailedError):
            store.get_or_run(request, runner)
        assert len(stub_openfold.calls()) == 1
        stub_openfold.install(fail=False)
        assert store.get_or_run(request, runner, rerun=True).status == "done"
        assert len(stub_openfold.calls()) == 2

    def test_a_batch_is_one_process(self, tmp_path, stub_openfold):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        requests = [
            runner.make_request(P53, name=name, binder_chain="B", receptor_chain="A", seeds=(1,))
            for name in ("one",)
        ]
        other = tmp_path / "p53_copy.pdb"
        other.write_text(
            P53.read_text(encoding="utf-8") + "REMARK another file\n", encoding="utf-8"
        )
        requests.append(
            runner.make_request(other, name="two", binder_chain="B", receptor_chain="A", seeds=(1,))
        )
        entries = store.run_missing(requests, runner)
        (argv,) = stub_openfold.calls()
        assert [e.status for e in entries] == ["done", "done"]
        assert argv[0] == "predict"
        parser = OpenFold3Parser()
        assert parser.load(entries[0].prediction_dir, "one").iptm == 0.25
        assert parser.load(entries[1].prediction_dir, "two").iptm == 0.25
        assert not (entries[0].prediction_dir / "two").exists()

    def test_a_second_run_of_the_batch_runs_nothing(self, tmp_path, stub_openfold):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = runner.make_request(P53, name="p53", binder_chain="B", receptor_chain="A")
        store.run_missing([request], runner)
        store.run_missing([request], runner)
        assert len(stub_openfold.calls()) == 1


class TestSessionWithOpenFold3:
    """Session, store and runner together, with the OpenFold3 parser of the registry."""

    def test_two_metrics_and_a_second_session_start_one_process(self, tmp_path, stub_openfold):
        from binding_metrics.predictors.session import PredictionSession

        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        request = runner.make_request(P53, name="p53", binder_chain="B", receptor_chain="A")
        session = PredictionSession(store, [runner])
        assert session.record(request).ptm == 0.5  # metric one
        assert session.record(request).iptm == 0.25  # metric two
        again = PredictionSession(store, [runner])  # a restarted pipeline
        assert again.record(request).avg_plddt == 77.0
        assert len(stub_openfold.calls()) == 1
        assert session.stats()["runs"] == 1 and again.stats()["runs"] == 0
        assert again.stats()["hits"] == 1

    def test_prefetch_starts_one_process_for_a_batch(self, tmp_path, stub_openfold):
        from binding_metrics.predictors.session import PredictionSession

        runner = OpenFold3Runner()
        other = tmp_path / "p53_other.pdb"
        other.write_text(P53.read_text(encoding="utf-8") + "REMARK other\n", encoding="utf-8")
        requests = [
            runner.make_request(path, name=name, binder_chain="B", receptor_chain="A")
            for path, name in ((P53, "one"), (other, "two"))
        ]
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        session.prefetch(requests)
        assert [session.record(request).iptm for request in requests] == [0.25, 0.25]
        assert len(stub_openfold.calls()) == 1
        assert session.stats()["runs"] == 2

    def test_a_failure_reaches_the_caller_as_one_readable_error(self, tmp_path, stub_openfold):
        from binding_metrics.predictors.session import PredictionSession

        stub_openfold.install(fail=True)
        runner = OpenFold3Runner()
        request = runner.make_request(P53, name="p53", binder_chain="B", receptor_chain="A")
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        with pytest.raises(PredictionFailedError, match="cowardly refusing"):
            session.record(request)
        with pytest.raises(PredictionFailedError):
            session.record(request)
        assert len(stub_openfold.calls()) == 1 and session.stats()["failed"] == 1


# ---------------------------------------------------------------------------- module facts


def test_the_new_modules_import_nothing_heavy():
    """Importing the store, session and runners, and building a request, needs no numpy."""
    code = textwrap.dedent(
        """
        import sys
        import binding_metrics.predictors.runners
        import binding_metrics.predictors.store
        import binding_metrics.predictors.session
        import binding_metrics.predictors.of3_runner as m
        import tempfile, pathlib
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "openfold3", "gemmi", "simtk")
        print(sorted(x for x in sys.modules if x.split(".")[0] in heavy))
        path = pathlib.Path(tempfile.mkdtemp()) / "c.cif"
        path.write_text("x", encoding="utf-8")
        m.OpenFold3Runner().make_request(path, name="q", binder_chain="B", receptor_chain="A")
        print(sorted(x for x in sys.modules if x.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:2] == ["[]", "[]"]


def test_importing_the_module_imports_only_the_standard_library():
    code = textwrap.dedent(
        """
        import sys
        import binding_metrics.predictors.runners
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "openfold3", "gemmi")
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
