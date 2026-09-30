"""PredictionSession: one model run and one parse per request, however many metrics ask.

A stub runner writes the made-up ``stub`` model's files (``synth_stub``) and the session parses
them with ``StubParser``; nothing needs a model, a GPU or the network.
"""

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from binding_metrics.predictors import registry
from binding_metrics.predictors.registry import PARSERS, ParserSpec
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import (
    PredictionFailedError,
    PredictionRequest,
    PredictionStore,
    PredictionUnavailableError,
)
from tests.predictors import synth
from tests.predictors.synth_stub import StubParser, write_prediction

TRUTH = synth.synthetic_complex()
SECOND_SAMPLE = synth.synthetic_complex(plddt_shift=10.0)


class ParsingRunner(PredictionRunner):
    """Writes two stub-model samples of ``request.name``; counts runs and batches."""

    name = "stub"

    def __init__(self, *, fail=None, batch_fail=(), gate=None, batchable=True):
        self.fail = fail
        self.batch_fail = set(batch_fail)
        self.gate = gate
        self.batchable = batchable
        self.runs = []
        self.batches = []

    def _write(self, folder, name):
        write_prediction(folder, name, TRUTH)
        write_prediction(folder, name, SECOND_SAMPLE, sample=2)

    def prepare(self, request, work_dir):
        return Path(work_dir)

    def run(self, request, work_dir):
        self.runs.append(request.name)
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail is not None:
            raise self.fail
        self._write(work_dir, request.name)
        return Path(work_dir)

    def supports_batch(self, request):
        return self.batchable

    def run_many(self, requests, work_dir):
        self.batches.append([r.name for r in requests])
        results = {}
        for request in requests:
            if request.name in self.batch_fail:
                results[request.key()] = RuntimeError(f"{request.name} ran out of memory")
                continue
            folder = Path(work_dir) / request.key()[:12]
            folder.mkdir()
            self._write(folder, request.name)
            results[request.key()] = folder
        return results

    def version(self):
        return "1.0"


def make_request(tmp_path, name="s1", content=b"ATOM 1", **kwargs):
    path = Path(tmp_path) / "in" / f"{name}.cif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    fields = {
        "input_path": path,
        "binder_chain": "B",
        "receptor_chain": "A",
        "model_version": "1.0",
    }
    fields.update(kwargs)
    return PredictionRequest("stub", name, **fields)


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "store")


def make_session(store, runner=None, **kwargs):
    runners = {} if runner is None else {"stub": runner}
    return PredictionSession(store, runners, {"stub": StubParser()}, **kwargs)


def assert_counts_add_up(stats):
    assert stats["requests"] == (
        stats["memo_hits"] + stats["hits"] + stats["adopted"] + stats["misses"]
    ), stats


# ---------------------------------------------------------------------------- run once


class TestRunOnce:
    def test_two_metrics_asking_for_one_request_run_the_model_once(self, store, tmp_path):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path)
        first = session.record(request)  # metric one
        second = session.record(make_request(tmp_path))  # metric two, same request
        assert runner.runs == ["s1"]
        assert first is second  # parsed once
        assert first.model == "stub" and first.avg_plddt == pytest.approx(
            TRUTH.scalars["avg_plddt"]
        )
        assert session.stats() == {
            "requests": 2,
            "memo_hits": 1,
            "hits": 0,
            "adopted": 0,
            "misses": 1,
            "runs": 1,
            "failed": 0,
            "parsed": 1,
        }

    def test_the_second_ask_does_not_touch_the_disk(self, store, tmp_path, monkeypatch):
        session = make_session(store, ParsingRunner())
        request = make_request(tmp_path)
        session.record(request)

        def forbidden(*args, **kwargs):
            raise AssertionError("the store was touched for a remembered request")

        for method in ("lookup", "ensure", "get_or_run"):
            monkeypatch.setattr(store, method, forbidden)
        assert session.record(request).avg_plddt > 0
        assert session.entry(request).status == "done"

    def test_the_entry_and_the_record_share_the_run(self, store, tmp_path):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path)
        entry = session.entry(request)
        record = session.record(request)
        assert runner.runs == ["s1"]
        assert record.structure_path.parent == entry.prediction_dir
        assert record.files.directory == entry.prediction_dir

    def test_a_new_session_on_the_same_store_finds_the_run(self, store, tmp_path):
        request = make_request(tmp_path)
        make_session(store, ParsingRunner()).record(request)
        runner = ParsingRunner()
        session = make_session(store, runner)
        session.record(request)
        assert runner.runs == []
        stats = session.stats()
        assert (stats["hits"], stats["misses"], stats["runs"]) == (1, 0, 0)
        assert_counts_add_up(stats)

    def test_samples_and_chain_maps_are_parsed_separately_from_one_run(self, store, tmp_path):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path)
        first = session.record(request)
        second = session.record(request, sample=2)
        renamed = session.record(request, chain_map={"A": "R", "B": "L"})
        assert runner.runs == ["s1"]
        assert second.avg_plddt == pytest.approx(SECOND_SAMPLE.scalars["avg_plddt"])
        assert first.chain_map == {} and renamed.chain_map == {"A": "R", "B": "L"}
        assert session.record(request, chain_map={"B": "L", "A": "R"}) is renamed
        assert session.record(request, sample=2) is second
        stats = session.stats()
        assert (stats["runs"], stats["parsed"]) == (1, 3)
        assert_counts_add_up(stats)

    def test_a_request_with_another_seed_is_another_run(self, store, tmp_path):
        runner = ParsingRunner()
        session = make_session(store, runner)
        session.record(make_request(tmp_path, seeds=(1,)))
        session.record(make_request(tmp_path, seeds=(2,)))
        assert runner.runs == ["s1", "s1"]
        assert session.stats()["runs"] == 2

    def test_the_stats_are_a_copy(self, store, tmp_path):
        session = make_session(store, ParsingRunner())
        session.record(make_request(tmp_path))
        session.stats()["runs"] = 99
        assert session.stats()["runs"] == 1

    def test_the_parser_comes_from_the_registry_when_none_is_given(
        self, store, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(
            PARSERS,
            "stub",
            ParserSpec(
                name="stub",
                import_path="tests.predictors.synth_stub:StubParser",
                display_name="Stub model",
                family="af3",
            ),
        )
        session = PredictionSession(store, [ParsingRunner()])  # runners as a list
        assert session.record(make_request(tmp_path)).model == "stub"
        assert registry.get_parser("stub").name == "stub"

    def test_an_unknown_model_names_the_available_ones(self, store, tmp_path):
        session = PredictionSession(store, {})
        request = PredictionRequest("ghost", "s1", sequences={"A": "GG"})
        store.adopt(request, tmp_path)
        with pytest.raises(KeyError, match="Unknown predictor 'ghost'"):
            session.record(request)


class TestWithoutARunner:
    def test_a_stored_run_is_served(self, store, tmp_path):
        request = make_request(tmp_path)
        make_session(store, ParsingRunner()).record(request)
        session = make_session(store)
        assert session.record(request).avg_plddt > 0
        assert session.stats()["hits"] == 1

    def test_a_miss_is_an_error_and_is_counted(self, store, tmp_path):
        session = make_session(store)
        with pytest.raises(PredictionUnavailableError, match="no runner"):
            session.record(make_request(tmp_path))
        stats = session.stats()
        assert (stats["requests"], stats["misses"], stats["runs"]) == (1, 1, 0)
        assert_counts_add_up(stats)

    def test_a_miss_is_not_remembered(self, store, tmp_path):
        request = make_request(tmp_path)
        session = make_session(store)
        with pytest.raises(PredictionUnavailableError):
            session.entry(request)
        make_session(store, ParsingRunner()).record(request)  # someone else makes it
        assert session.entry(request).status == "done"


# ---------------------------------------------------------------------------- failures


class TestFailures:
    def test_a_failure_is_raised_for_each_ask_and_the_model_ran_once(self, store, tmp_path):
        runner = ParsingRunner(fail=RuntimeError("CUDA out of memory"))
        session = make_session(store, runner)
        request = make_request(tmp_path)
        for _ in range(3):
            with pytest.raises(PredictionFailedError, match="CUDA out of memory"):
                session.record(request)
        assert runner.runs == ["s1"]
        stats = session.stats()
        assert (stats["runs"], stats["failed"], stats["misses"], stats["memo_hits"]) == (1, 1, 1, 2)
        assert_counts_add_up(stats)

    def test_a_new_session_does_not_retry_it(self, store, tmp_path):
        request = make_request(tmp_path)
        with pytest.raises(PredictionFailedError):
            make_session(store, ParsingRunner(fail=RuntimeError("bad"))).record(request)
        runner = ParsingRunner()
        session = make_session(store, runner)
        with pytest.raises(PredictionFailedError, match="rerun=True"):
            session.record(request)
        assert runner.runs == []
        assert (session.stats()["hits"], session.stats()["failed"]) == (1, 1)

    def test_rerun_predictions_retries_it_once_per_session(self, store, tmp_path):
        request = make_request(tmp_path)
        with pytest.raises(PredictionFailedError):
            make_session(store, ParsingRunner(fail=RuntimeError("bad"))).record(request)
        runner = ParsingRunner()
        session = make_session(store, runner, rerun=True)
        assert session.record(request).avg_plddt > 0
        session.record(request)
        assert runner.runs == ["s1"]
        assert session.stats()["runs"] == 1

    def test_rerun_predictions_runs_a_finished_request_again_once(self, store, tmp_path):
        request = make_request(tmp_path)
        make_session(store, ParsingRunner()).record(request)
        runner = ParsingRunner()
        session = make_session(store, runner, rerun=True)
        session.record(request)
        session.record(request)
        assert runner.runs == ["s1"]


# ---------------------------------------------------------------------------- adoption


class TestAdoption:
    @pytest.fixture
    def outputs(self, tmp_path):
        folder = tmp_path / "user_outputs"
        write_prediction(folder, "s1", TRUTH)
        return folder

    def test_adopted_outputs_are_parsed_and_never_run(self, store, tmp_path, outputs):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path)
        assert session.adopt(request, outputs).status == "adopted"
        record = session.record(request)
        assert runner.runs == [] and record.avg_plddt == pytest.approx(TRUTH.scalars["avg_plddt"])
        assert record.structure_path.parent == outputs.resolve()
        stats = session.stats()
        assert (stats["adopted"], stats["misses"], stats["runs"], stats["hits"]) == (1, 0, 0, 0)
        assert_counts_add_up(stats)

    def test_a_wrong_directory_is_refused_by_the_check(self, store, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        session = make_session(store, ParsingRunner())
        with pytest.raises(ValueError, match="no Stub model output for 's1'"):
            session.adopt(make_request(tmp_path), empty)
        with pytest.raises(ValueError, match="'other'"):
            session.adopt(make_request(tmp_path, name="other"), tmp_path)
        assert not store.root.exists()

    def test_check_can_be_switched_off(self, store, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        session = make_session(store)
        assert session.adopt(make_request(tmp_path), empty, check=False).status == "adopted"

    def test_adoption_replaces_what_the_session_remembers(self, store, tmp_path, outputs):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path)
        with pytest.raises(PredictionFailedError):
            make_session(store, ParsingRunner(fail=RuntimeError("bad"))).record(request)
        with pytest.raises(PredictionFailedError):
            session.record(request)  # the failure is now remembered
        session.adopt(request, outputs)
        assert session.record(request).structure_path.parent == outputs.resolve()
        assert runner.runs == []

    def test_a_rerun_does_not_replace_adopted_outputs(self, store, tmp_path, outputs):
        runner = ParsingRunner()
        session = make_session(store, runner, rerun=True)
        request = make_request(tmp_path)
        session.adopt(request, outputs)
        session.record(request)
        assert runner.runs == []

    def test_a_runner_with_a_version_finds_outputs_adopted_without_one(
        self, store, tmp_path, outputs
    ):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path, model_version="")
        session.adopt(request, outputs)
        session.record(request)
        assert runner.runs == []

    def test_a_directory_that_cannot_be_adopted_raises_before_the_check_needs_a_parser(
        self, store, tmp_path
    ):
        with pytest.raises(FileNotFoundError):
            make_session(store).adopt(make_request(tmp_path), tmp_path / "nowhere", check=False)


# ---------------------------------------------------------------------------- prefetch


class TestPrefetch:
    def _requests(self, tmp_path, count):
        return [
            make_request(tmp_path, name=f"s{i}", content=f"c{i}".encode()) for i in range(count)
        ]

    def test_the_missing_requests_run_in_one_batch_and_then_cost_nothing(self, store, tmp_path):
        requests = self._requests(tmp_path, 4)
        make_session(store, ParsingRunner()).record(requests[0])  # one is already there
        runner = ParsingRunner()
        session = make_session(store, runner)
        session.prefetch(requests)
        assert runner.batches == [["s1", "s2", "s3"]] and runner.runs == []
        for request in requests:
            assert session.record(request).avg_plddt > 0
        assert runner.batches == [["s1", "s2", "s3"]]
        stats = session.stats()
        assert (stats["hits"], stats["misses"], stats["runs"]) == (1, 3, 3)
        assert stats["memo_hits"] == 4 and stats["requests"] == 8
        assert_counts_add_up(stats)

    def test_a_request_that_failed_in_the_batch_raises_for_that_request_only(self, store, tmp_path):
        requests = self._requests(tmp_path, 3)
        session = make_session(store, ParsingRunner(batch_fail=["s1"]))
        session.prefetch(requests)  # does not raise
        assert session.record(requests[0]).avg_plddt > 0
        with pytest.raises(PredictionFailedError, match="s1 ran out of memory"):
            session.record(requests[1])
        assert session.record(requests[2]).avg_plddt > 0
        assert session.stats()["failed"] == 1

    def test_requests_already_remembered_are_not_asked_again(self, store, tmp_path):
        requests = self._requests(tmp_path, 3)
        runner = ParsingRunner()
        session = make_session(store, runner)
        session.record(requests[0])
        session.prefetch(requests)
        assert runner.runs == ["s0"] and runner.batches == [["s1", "s2"]]

    def test_a_repeated_request_is_prefetched_once(self, store, tmp_path):
        first, second = self._requests(tmp_path, 2)
        runner = ParsingRunner()
        make_session(store, runner).prefetch([first, second, first])
        assert runner.batches == [["s0", "s1"]]

    def test_prefetch_without_a_runner_raises_when_something_is_missing(self, store, tmp_path):
        with pytest.raises(PredictionUnavailableError):
            make_session(store).prefetch(self._requests(tmp_path, 2))

    def test_prefetch_of_nothing_is_fine(self, store):
        session = make_session(store, ParsingRunner())
        session.prefetch([])
        assert session.stats()["requests"] == 0

    def test_prefetch_of_adopted_outputs_counts_them(self, store, tmp_path):
        request = make_request(tmp_path)
        outputs = tmp_path / "outs"
        write_prediction(outputs, "s1", TRUTH)
        store.adopt(request, outputs)
        session = make_session(store, ParsingRunner())
        session.prefetch([request])
        assert session.stats()["adopted"] == 1


# ---------------------------------------------------------------------------- threads


class TestThreads:
    def test_two_threads_asking_for_one_request_run_it_once(self, store, tmp_path):
        runner = ParsingRunner()
        session = make_session(store, runner)
        request = make_request(tmp_path)
        records = []

        def ask():
            records.append(session.record(request))

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert runner.runs == ["s1"]
        assert len({id(record) for record in records}) == 1
        stats = session.stats()
        assert (stats["runs"], stats["parsed"], stats["memo_hits"]) == (1, 1, 3)

    def test_requests_for_different_samples_are_not_serialised(self, store, tmp_path):
        """Both runs must be inside the model at once for the barrier to open."""
        gate = threading.Barrier(2)
        runner = ParsingRunner(gate=gate)
        session = make_session(store, runner)
        requests = [
            make_request(tmp_path, name=f"s{i}", content=f"c{i}".encode()) for i in range(2)
        ]
        errors = []

        def ask(request):
            try:
                session.record(request)
            except Exception as exc:  # noqa: BLE001 - reported by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=ask, args=(r,)) for r in requests]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert errors == [] and sorted(runner.runs) == ["s0", "s1"]


def test_the_module_imports_only_the_standard_library():
    code = textwrap.dedent(
        """
        import sys
        import binding_metrics.predictors.session
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "openfold3", "gemmi")
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
