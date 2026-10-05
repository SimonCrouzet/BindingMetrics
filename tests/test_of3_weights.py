"""Custom weights for OpenFold3: the checkpoint reaches ``run_openfold`` and the key follows it.

A stub ``run_openfold`` on PATH records its command line and writes the smallest output; what
OpenFold3 does with ``--inference_ckpt_path`` rests on its source (``run_openfold.py`` at v0.5.0),
not on a run.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_run
from binding_metrics.predictors.of3 import OpenFold3Parser
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import PredictionStore
from binding_metrics.predictors.weights import WeightsRef

P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

STUB = """#!{python}
import json, pathlib, sys
argv = sys.argv[1:]
with open({record!r}, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\\n")
args = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
query = json.loads(pathlib.Path(args["query_json"]).read_text(encoding="utf-8"))
out = pathlib.Path(args["output_dir"])
for name in query["queries"]:
    seed_dir = out / name / "seed_42"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / (name + "_seed_42_sample_1_confidences_aggregated.json")).write_text(
        json.dumps({{"avg_plddt": 77.0, "ptm": 0.5, "iptm": 0.25}}), encoding="utf-8")
(out / "experiment_config.json").write_text(
    json.dumps({{"inference_ckpt_path": args.get("inference_ckpt_path"),
                "inference_ckpt_name": None}}), encoding="utf-8")
"""


@pytest.fixture(autouse=True)
def _openfold3_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "openfold_cache"))
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )
    monkeypatch.setattr(_openfold_run, "_VERSION_BY_PYTHON", {})


@pytest.fixture
def stub_openfold(tmp_path, monkeypatch):
    """A ``run_openfold`` on PATH that records its arguments; returns a reader of the log."""
    pytest.importorskip("gemmi")
    record = tmp_path / "argv.jsonl"
    script = tmp_path / "bin" / "run_openfold"
    script.parent.mkdir()
    script.write_text(STUB.format(python=sys.executable, record=str(record)), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")

    def calls():
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]

    return calls


def checkpoint(directory: Path, name="of3-finetuned.pt", content=b"fine-tuned weights") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def request_for(runner, **kwargs):
    return runner.make_request(P53, name="p53", binder_chain="B", receptor_chain="A", **kwargs)


def ckpt_options(argv):
    return [a for a in argv if a.startswith("--inference_ckpt_path")]


class TestTheRunnerSupportsWeights:
    def test_the_class_declares_it(self):
        assert OpenFold3Runner.supports_custom_weights is True
        assert OpenFold3Runner.weights_kind == "file"

    def test_the_request_holds_the_content_and_no_path_in_the_key(self, tmp_path):
        weights = checkpoint(tmp_path / "w")
        request = request_for(OpenFold3Runner(), weights=weights)
        assert isinstance(request.weights, WeightsRef) and request.weights.kind == "file"
        assert request.canonical()["weights"]["sha256"] == request.weights.sha256
        assert str(tmp_path) not in json.dumps(request.canonical())
        assert request.options["inference_ckpt_path"] is None  # the older route is not used

    def test_the_key_follows_the_content_not_the_path(self, tmp_path):
        runner = OpenFold3Runner()
        plain = request_for(runner)
        one = request_for(runner, weights=checkpoint(tmp_path / "one"))
        copy = request_for(runner, weights=checkpoint(tmp_path / "two", "other-name.ckpt"))
        changed = request_for(runner, weights=checkpoint(tmp_path / "three", content=b"another"))
        assert one.key() == copy.key()
        assert len({plain.key(), one.key(), changed.key()}) == 3

    def test_a_request_without_weights_has_the_key_of_the_default_model(self, tmp_path):
        runner = OpenFold3Runner()
        assert request_for(runner).key() == request_for(runner, weights=None).key()
        assert request_for(runner).weights is None

    def test_the_older_checkpoint_argument_still_works_with_its_own_key(self, tmp_path):
        runner = OpenFold3Runner()
        weights = checkpoint(tmp_path / "w")
        old = request_for(runner, inference_ckpt_path=weights)
        assert old.options["inference_ckpt_path"] == str(weights) and old.weights is None
        assert old.key() != request_for(runner, weights=weights).key()
        assert runner._run_arguments(old)["inference_ckpt_path"] == str(weights)

    def test_both_ways_of_naming_a_checkpoint_are_refused(self, tmp_path):
        weights = checkpoint(tmp_path / "w")
        with pytest.raises(ValueError, match="weights or inference_ckpt_path, not both"):
            request_for(OpenFold3Runner(), weights=weights, inference_ckpt_path=weights)

    def test_a_missing_file_is_refused_when_the_request_is_made(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="do not exist"):
            request_for(OpenFold3Runner(), weights=tmp_path / "nowhere.pt")

    def test_a_weights_reference_is_taken_as_it_is(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        weights = checkpoint(tmp_path / "w")
        request = request_for(OpenFold3Runner(), weights=store.weights_reference(weights))
        assert request.weights == store.weights_reference(weights)


class TestTheCheckpointReachesOpenFold3:
    def test_the_run_arguments_carry_the_checkpoint(self, tmp_path):
        weights = checkpoint(tmp_path / "w")
        runner = OpenFold3Runner()
        assert runner._run_arguments(request_for(runner, weights=weights)) == {
            "inference_ckpt_path": str(weights)
        }
        assert runner._run_arguments(request_for(runner)) == {}

    def test_the_command_has_the_checkpoint_and_the_default_command_has_none(
        self, tmp_path, stub_openfold
    ):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        weights = checkpoint(tmp_path / "w")
        store.get_or_run(request_for(runner, weights=store.weights_reference(weights)), runner)
        store.get_or_run(request_for(runner), runner)
        custom, default = stub_openfold()
        assert ckpt_options(custom) == [f"--inference_ckpt_path={weights}"]
        assert ckpt_options(default) == []

    def test_the_other_checkpoint_is_another_run_and_the_same_one_is_not(
        self, tmp_path, stub_openfold
    ):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        one = checkpoint(tmp_path / "one")
        two = checkpoint(tmp_path / "two", content=b"weights of another fine-tune")
        copy_of_one = checkpoint(tmp_path / "copy", "renamed.pt")
        for weights in (one, two, copy_of_one):
            store.get_or_run(request_for(runner, weights=store.weights_reference(weights)), runner)
        assert len(stub_openfold()) == 2  # the copy of the first checkpoint found its entry

    def test_a_batch_runs_with_the_checkpoint(self, tmp_path, stub_openfold):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        weights = store.weights_reference(checkpoint(tmp_path / "w"))
        other = tmp_path / "p53_copy.pdb"
        other.write_text(
            P53.read_text(encoding="utf-8") + "REMARK another file\n", encoding="utf-8"
        )
        requests = [
            runner.make_request(
                path, name=name, binder_chain="B", receptor_chain="A", weights=weights
            )
            for name, path in (("one", P53), ("two", other))
        ]
        entries = store.run_missing(requests, runner)
        (argv,) = stub_openfold()
        assert [e.status for e in entries] == ["done", "done"]
        assert ckpt_options(argv) == [f"--inference_ckpt_path={weights.path}"]

    def test_a_directory_is_refused_for_a_checkpoint_file_before_openfold3_starts(
        self, tmp_path, stub_openfold
    ):
        store = PredictionStore(tmp_path / "store")
        directory = tmp_path / "d"
        checkpoint(directory)
        runner = OpenFold3Runner()
        with pytest.raises(ValueError, match="takes its weights as a file, and .* is a directory"):
            store.get_or_run(request_for(runner, weights=directory), runner)
        assert stub_openfold() == []
        with pytest.raises(ValueError, match="takes its weights as a file"):
            runner.prepare(request_for(runner, weights=directory), tmp_path / "work")


class TestTheRecord:
    def test_the_record_has_the_request_weights_and_what_openfold3_recorded(
        self, tmp_path, stub_openfold
    ):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        weights = checkpoint(tmp_path / "w")
        request = request_for(runner, weights=store.weights_reference(weights))
        record = PredictionSession(store, [runner]).record(request)
        assert record.extras["weights"] == request.weights.to_dict()
        # the adapter reads experiment_config.json of the run; the stub writes the path there
        assert record.extras["inference_ckpt_path"] == str(weights)
        assert record.model == OpenFold3Parser.name

    def test_a_default_run_has_no_weights_extra(self, tmp_path, stub_openfold):
        store = PredictionStore(tmp_path / "store")
        runner = OpenFold3Runner()
        record = PredictionSession(store, [runner]).record(request_for(runner))
        assert "weights" not in record.extras
