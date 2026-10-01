"""PredictionRequest and PredictionStore: the key, the entry layout, run-once, failures, adoption.

Nothing here needs a model, a GPU or the network: a stub runner writes small text files, and the
processes that must not run a model twice are started with ``multiprocessing`` (fork).
"""

import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from binding_metrics.predictors import store as store_module
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import (
    KEY_FORMAT,
    MODES,
    PredictionFailedError,
    PredictionRequest,
    PredictionStore,
    PredictionStoreError,
    PredictionUnavailableError,
    StoredPrediction,
    file_sha256,
)

FORK = "fork" in multiprocessing.get_all_start_methods()
needs_fork = pytest.mark.skipif(not FORK, reason="these tests start processes with fork")


class StubRunner(PredictionRunner):
    """Writes ``<name>.txt`` in a ``predictions`` folder; counts its runs in memory and in a file.

    ``counter_file`` lets processes that do not share memory count together. ``fail`` is an
    exception to raise after some output was written; ``batch_fail`` maps a request name to the
    exception ``run_many`` returns for it.
    """

    name = "stub"

    def __init__(
        self,
        counter_file=None,
        *,
        fail=None,
        sleep=0.0,
        version="1.0",
        available=True,
        batchable=True,
        batch_fail=None,
        batch_raise=None,
    ):
        self.counter_file = None if counter_file is None else Path(counter_file)
        self.fail = fail
        self.sleep = sleep
        self._version = version
        self.available = available
        self.batchable = batchable
        self.batch_fail = dict(batch_fail or {})
        self.batch_raise = batch_raise
        self.run_names = []
        self.batches = []

    def _count(self, names):
        self.run_names.extend(names)
        if self.counter_file is not None:
            with open(self.counter_file, "a", encoding="utf-8") as handle:
                handle.write("".join(f"{os.getpid()} {name}\n" for name in names))

    def prepare(self, request, work_dir):
        return Path(work_dir) / "input.txt"

    def run(self, request, work_dir):
        self._count([request.name])
        out = Path(work_dir) / "predictions"
        out.mkdir()
        (out / f"{request.name}.txt").write_text(
            f"model output of {request.name}", encoding="utf-8"
        )
        time.sleep(self.sleep)
        if self.fail is not None:
            raise self.fail
        return out

    def supports_batch(self, request):
        return self.batchable

    def run_many(self, requests, work_dir):
        self.batches.append([r.name for r in requests])
        self._count([r.name for r in requests])
        if self.batch_raise is not None:
            raise self.batch_raise
        results = {}
        for request in requests:
            if request.name in self.batch_fail:
                results[request.key()] = self.batch_fail[request.name]
                continue
            folder = Path(work_dir) / "split" / request.key()[:10]
            folder.mkdir(parents=True)
            (folder / f"{request.name}.txt").write_text("batched output", encoding="utf-8")
            results[request.key()] = folder
        return results

    def is_available(self):
        return self.available

    def version(self):
        return self._version


def make_request(tmp_path, name="s1", content=b"ATOM 1", *, subdir="in", **kwargs):
    """A score request for a small input file ``<tmp_path>/<subdir>/<name>.cif``."""
    folder = Path(tmp_path) / subdir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.cif"
    path.write_bytes(content)
    fields = {
        "input_path": path,
        "binder_chain": "B",
        "receptor_chain": "A",
        "model_version": "1.0",
    }
    fields.update(kwargs)
    return PredictionRequest("stub", name, **fields)


def same_input_other_name(request, name):
    """A request for the same input file and settings under another name."""
    return PredictionRequest(
        request.model,
        name,
        input_path=request.input_path,
        binder_chain=request.binder_chain,
        receptor_chain=request.receptor_chain,
        model_version=request.model_version,
    )


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "store")


# ---------------------------------------------------------------------------- the key


class TestTheKey:
    def test_it_is_a_sha256_hex_digest_and_stable(self, tmp_path):
        request = make_request(tmp_path)
        assert len(request.key()) == 64 and set(request.key()) <= set("0123456789abcdef")
        assert make_request(tmp_path).key() == request.key()

    def test_it_hashes_the_documented_canonical_json(self, tmp_path):
        """The canonical form is written out here, so a change to it fails this test."""
        request = make_request(
            tmp_path, seeds=(1, 2), num_samples=3, options={"presets": ["predict"], "a": 1}
        )
        canonical = {
            "format": KEY_FORMAT,
            "model": "stub",
            "model_version": "1.0",
            "mode": "score",
            "file_sha256": {"input": hashlib.sha256(b"ATOM 1").hexdigest()},
            "binder_chain": "B",
            "receptor_chain": "A",
            "sequences": {},
            "seeds": [1, 2],
            "num_samples": 3,
            "options": {"a": 1, "presets": ["predict"]},
        }
        text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        assert request.canonical() == canonical
        assert request.key() == hashlib.sha256(text.encode("ascii")).hexdigest()

    def test_a_key_is_pinned_so_that_a_new_version_does_not_orphan_old_stores(self, tmp_path):
        request = make_request(tmp_path, seeds=(42,), options={"presets": ["predict", "low_mem"]})
        assert request.key() == (
            "d02ce3ae2903fa523097fbfd4af2ffa53c14d1d0a73f9e2eb8b01a82ea8c0081"
        ), "the canonical form changed: raise KEY_FORMAT so that old entries are not reused"

    @pytest.mark.parametrize(
        "change",
        [
            {"seeds": (43,)},
            {"num_samples": 6},
            {"options": {"presets": ["predict"]}},
            {"model_version": "1.1"},
            {"mode": "refold"},
            {"binder_chain": "C"},
            {"receptor_chain": "C"},
            {"sequences": {"B": "AAAA"}},
        ],
        ids=lambda change: next(iter(change)),
    )
    def test_a_changed_field_gives_another_key(self, tmp_path, change):
        assert make_request(tmp_path, **change).key() != make_request(tmp_path).key()

    def test_changed_input_file_content_gives_another_key(self, tmp_path):
        first = make_request(tmp_path, content=b"ATOM 1")
        second = make_request(tmp_path, content=b"ATOM 2")
        assert first.key() != second.key()

    def test_a_moved_or_renamed_identical_file_gives_the_same_key(self, tmp_path):
        first = make_request(tmp_path, name="s1", subdir="one")
        moved = make_request(tmp_path, name="s1", subdir="elsewhere/deeper")
        renamed = PredictionRequest(
            "stub",
            "s1",
            input_path=first.input_path.rename(first.input_path.with_name("renamed.pdb")),
            binder_chain="B",
            receptor_chain="A",
            model_version="1.0",
        )
        assert first.key() == moved.key() == renamed.key()
        assert first.input_path != moved.input_path

    def test_the_name_is_not_part_of_the_key_but_of_equality(self, tmp_path):
        first = make_request(tmp_path, name="s1")
        second = PredictionRequest(
            "stub",
            "other_name",
            input_path=first.input_path,
            binder_chain="B",
            receptor_chain="A",
            model_version="1.0",
        )
        assert first.key() == second.key()
        assert first != second
        assert first == make_request(tmp_path, name="s1")
        assert len({first, make_request(tmp_path, name="s1"), second}) == 2

    def test_for_adoption_puts_the_name_into_the_key(self, tmp_path):
        first = make_request(tmp_path, name="a", content=b"same")
        second = same_input_other_name(first, "b")
        assert first.key() == second.key()
        assert first.for_adoption().key() != first.key()
        assert first.for_adoption().key() != second.for_adoption().key()
        assert (
            first.for_adoption().key()
            == make_request(tmp_path, name="a", content=b"same").for_adoption().key()
        )
        assert first.for_adoption().canonical()["adopted_name"] == "a"
        assert "adopted_name" not in first.canonical()

    def test_for_adoption_is_idempotent_and_survives_the_other_copies(self, tmp_path):
        request = make_request(tmp_path, model_version="")
        adopted = request.for_adoption()
        assert adopted.for_adoption() is adopted
        assert adopted.with_model_version("2.0").canonical()["adopted_name"] == "s1"
        assert pickle.loads(pickle.dumps(adopted)).key() == adopted.key()
        assert request.adoption_name is None and adopted.adoption_name == "s1"

    def test_a_request_options_key_named_adopted_name_does_not_collide(self, tmp_path):
        request = make_request(tmp_path)
        lookalike = make_request(tmp_path, options={"adopted_name": "s1"})
        assert lookalike.key() != request.for_adoption().key()

    def test_dict_order_and_tuples_do_not_matter(self, tmp_path):
        first = make_request(
            tmp_path, options={"a": 1, "b": [1, 2]}, sequences={"A": "GG", "B": "K"}
        )
        second = make_request(
            tmp_path, options={"b": (1, 2), "a": 1}, sequences={"B": "K", "A": "GG"}
        )
        assert first.key() == second.key()

    def test_the_content_is_hashed_when_the_request_is_made(self, tmp_path):
        request = make_request(tmp_path)
        key = request.key()
        request.input_path.write_bytes(b"changed afterwards")
        assert request.key() == key

    def test_extra_files_are_hashed_by_content_too(self, tmp_path):
        template = tmp_path / "template.cif"
        template.write_bytes(b"T1")
        first = make_request(tmp_path, extra_files={"template": template})
        template.write_bytes(b"T2")
        second = make_request(tmp_path, extra_files={"template": template})
        assert first.key() != second.key() != make_request(tmp_path).key()

    def test_the_key_is_the_same_in_another_process(self, tmp_path):
        request = make_request(tmp_path, options={"presets": ["predict"]})
        code = textwrap.dedent(
            f"""
            from binding_metrics.predictors.store import PredictionRequest
            r = PredictionRequest("stub", "zzz", input_path={str(request.input_path)!r},
                binder_chain="B", receptor_chain="A", model_version="1.0",
                options={{"presets": ["predict"]}})
            print(r.key())
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True
        )
        assert result.stdout.strip() == request.key()

    def test_with_model_version_changes_the_key_without_reading_the_file_again(self, tmp_path):
        request = make_request(tmp_path, model_version="")
        request.input_path.unlink()
        newer = request.with_model_version("2.0")
        assert newer.model_version == "2.0" and request.model_version == ""
        assert newer.key() != request.key()
        assert newer.key() == make_request(tmp_path, model_version="2.0").key()

    def test_a_request_can_be_pickled_for_a_worker_process(self, tmp_path):
        request = make_request(tmp_path, options={"presets": ["predict"]}, seeds=(1, 2))
        again = pickle.loads(pickle.dumps(request))
        assert again == request and again.key() == request.key()


class TestBatchSignature:
    def test_samples_of_one_batch_share_it(self, tmp_path):
        first = make_request(tmp_path, name="s1", content=b"one", binder_chain="B")
        second = make_request(tmp_path, name="s2", content=b"two", binder_chain="P")
        assert first.batch_signature() == second.batch_signature()
        assert first.key() != second.key()

    @pytest.mark.parametrize(
        "change",
        [
            {"seeds": (7,)},
            {"num_samples": 1},
            {"options": {"x": 1}},
            {"mode": "refold"},
            {"model_version": "9"},
        ],
        ids=lambda change: next(iter(change)),
    )
    def test_a_shared_setting_splits_a_batch(self, tmp_path, change):
        assert (
            make_request(tmp_path, **change).batch_signature()
            != make_request(tmp_path, name="s2").batch_signature()
        )

    def test_a_per_sample_extra_file_splits_a_batch(self, tmp_path):
        template = tmp_path / "t.cif"
        template.write_bytes(b"T")
        with_template = make_request(tmp_path, extra_files={"template": template})
        assert with_template.batch_signature() != make_request(tmp_path).batch_signature()


class TestValidation:
    def test_a_request_needs_an_input(self):
        with pytest.raises(ValueError, match="needs an input file or sequences"):
            PredictionRequest("stub", "s1")

    def test_sequences_are_an_input(self):
        request = PredictionRequest("stub", "s1", mode="predict", sequences={"A": "GGG"})
        assert request.input_path is None and request.content_hashes == {}
        assert request.key() != PredictionRequest("stub", "s1", sequences={"A": "GGA"}).key()

    @pytest.mark.parametrize("model", ["", "Stub", "with space", "a/b", "../x", "2fold"])
    def test_the_model_is_a_safe_directory_name(self, model):
        with pytest.raises(ValueError, match="lower case letters"):
            PredictionRequest(model, "s1", sequences={"A": "G"})

    def test_the_mode_is_one_of_the_known_four(self):
        assert MODES == ("predict", "score", "refold", "lock")
        with pytest.raises(ValueError, match="mode must be one of"):
            PredictionRequest("stub", "s1", mode="dock", sequences={"A": "G"})

    def test_lock_is_a_mode_of_its_own_in_the_key(self):
        keys = {
            mode: PredictionRequest("stub", "s1", mode=mode, sequences={"A": "G"}).key()
            for mode in MODES
        }
        assert len(set(keys.values())) == 4

    def test_the_name_must_not_be_empty(self):
        with pytest.raises(ValueError, match="name"):
            PredictionRequest("stub", "", sequences={"A": "G"})

    def test_seeds_are_integers_not_a_string(self):
        with pytest.raises(TypeError, match="seeds"):
            PredictionRequest("stub", "s1", sequences={"A": "G"}, seeds="42")

    @pytest.mark.parametrize("num_samples", [0, -1, 2.5])
    def test_the_number_of_samples_is_a_positive_integer(self, num_samples):
        with pytest.raises(ValueError, match="num_samples"):
            PredictionRequest("stub", "s1", sequences={"A": "G"}, num_samples=num_samples)

    @pytest.mark.parametrize(
        "options", [{"path": Path("x")}, {"array": {1, 2}}, {"nan": float("nan")}, {"f": len}]
    )
    def test_options_must_be_plain_json_values(self, options):
        with pytest.raises(TypeError, match="JSON values"):
            PredictionRequest("stub", "s1", sequences={"A": "G"}, options=options)

    def test_options_are_copied_so_later_edits_do_not_change_the_key(self):
        options = {"presets": ["predict"]}
        request = PredictionRequest("stub", "s1", sequences={"A": "G"}, options=options)
        key = request.key()
        options["presets"].append("low_mem")
        options["new"] = 1
        assert request.key() == key

    def test_the_input_role_is_reserved(self, tmp_path):
        extra = tmp_path / "x"
        extra.write_bytes(b"x")
        with pytest.raises(ValueError, match="reserved"):
            make_request(tmp_path, extra_files={"input": extra})

    def test_a_missing_input_file_raises_when_the_request_is_made(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PredictionRequest("stub", "s1", input_path=tmp_path / "absent.cif")

    def test_file_sha256_is_the_content_digest(self, tmp_path):
        path = tmp_path / "f"
        path.write_bytes(b"x" * 3_000_000)
        assert file_sha256(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


# ---------------------------------------------------------------------------- run once


class TestRunOnce:
    def test_two_metrics_asking_the_same_request_run_the_model_once(self, store, tmp_path):
        runner = StubRunner()
        first = store.get_or_run(make_request(tmp_path), runner)
        second = store.get_or_run(make_request(tmp_path), runner)  # a second metric, same request
        assert runner.run_names == ["s1"]
        assert first.executed_here is True and second.executed_here is False
        assert first.key == second.key and first.run_id == second.run_id
        assert (second.prediction_dir / "s1.txt").read_text(encoding="utf-8").startswith("model")

    def test_two_names_for_one_input_share_one_run(self, store, tmp_path):
        first = make_request(tmp_path, name="a", content=b"the shared input")
        second = same_input_other_name(first, "b")
        runner = StubRunner()
        ran = store.get_or_run(first, runner)
        again = store.get_or_run(second, runner)
        assert runner.run_names == ["a"] and again.run_id == ran.run_id
        assert again.name == "a"  # the entry keeps the name it was run under

    def test_a_new_store_object_on_the_same_root_finds_the_entry(self, store, tmp_path):
        request = make_request(tmp_path)
        store.get_or_run(request, StubRunner())
        runner = StubRunner()
        entry = PredictionStore(store.root).get_or_run(request, runner)
        assert runner.run_names == [] and entry.status == "done"

    def test_the_input_may_have_moved_since(self, store, tmp_path):
        store.get_or_run(make_request(tmp_path, subdir="a"), StubRunner())
        runner = StubRunner()
        store.get_or_run(make_request(tmp_path, subdir="b"), runner)
        assert runner.run_names == []

    def test_a_different_input_is_another_run(self, store, tmp_path):
        runner = StubRunner()
        store.get_or_run(make_request(tmp_path, content=b"one"), runner)
        store.get_or_run(make_request(tmp_path, content=b"two"), runner)
        assert runner.run_names == ["s1", "s1"]

    def test_the_layout_of_an_entry(self, store, tmp_path):
        request = make_request(tmp_path)
        entry = store.get_or_run(request, StubRunner())
        expected = store.root / "stub" / request.key()[:2] / request.key()
        assert store.path_for(request) == expected and entry.directory == expected
        assert sorted(p.name for p in expected.iterdir()) == [
            "STATUS.json",
            "outputs",
            "request.json",
        ]
        assert entry.prediction_dir == expected / "outputs" / "predictions"

        written = json.loads((expected / "request.json").read_text(encoding="utf-8"))
        assert written["key"] == request.key() and written["name"] == "s1"
        assert written["input_path"] == str(request.input_path)
        assert written["file_sha256"] == {"input": file_sha256(request.input_path)}

        status = json.loads((expected / "STATUS.json").read_text(encoding="utf-8"))
        assert status["status"] == "done" and status["reason"] is None
        assert status["key"] == request.key() and status["prediction_subdir"] == "predictions"
        assert status["runner"] == {"name": "stub", "version": "1.0"}
        assert status["run_id"] == entry.run_id
        assert status["started_at"] <= status["finished_at"]

    def test_only_the_entry_and_its_lock_are_left_behind(self, store, tmp_path):
        request = make_request(tmp_path)
        store.get_or_run(request, StubRunner())
        names = sorted(p.name for p in store.path_for(request).parent.iterdir())
        assert names == [request.key(), f"{request.key()}.lock"]

    def test_a_relative_root_does_not_move_with_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        relative = PredictionStore("cache")
        assert relative.root == tmp_path / "cache"
        request = make_request(tmp_path)
        entry = relative.get_or_run(request, StubRunner())
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert entry.directory.is_absolute() and entry.prediction_dir.is_dir()
        assert relative.lookup(request).key == entry.key

    def test_a_home_relative_root_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert PredictionStore("~/predictions").root == tmp_path / "predictions"

    def test_a_lookup_writes_nothing(self, store, tmp_path):
        assert store.lookup(make_request(tmp_path)) is None
        assert not store.root.exists()

    def test_the_runner_that_writes_into_the_work_dir_directly_is_supported(self, store, tmp_path):
        class Direct(StubRunner):
            def run(self, request, work_dir):
                (Path(work_dir) / "out.txt").write_text("x", encoding="utf-8")
                return Path(work_dir)

        entry = store.get_or_run(make_request(tmp_path), Direct())
        assert entry.prediction_dir == entry.directory / "outputs"
        assert (entry.prediction_dir / "out.txt").exists()

    def test_a_request_without_a_version_is_stored_under_the_runners_version(self, store, tmp_path):
        request = make_request(tmp_path, model_version="")
        entry = store.get_or_run(request, StubRunner(version="3.4"))
        assert entry.runner_version == "3.4"
        assert entry.key == request.with_model_version("3.4").key() != request.key()
        runner = StubRunner(version="3.4")
        store.get_or_run(request, runner)
        assert runner.run_names == []

    def test_a_run_made_under_an_unknown_version_is_not_reused_once_it_is_known(
        self, store, tmp_path
    ):
        request = make_request(tmp_path, model_version="")
        unknown = StubRunner(version=None)
        first = store.get_or_run(request, unknown)
        assert first.key == request.key()  # stored without a version
        assert store.get_or_run(request, StubRunner(version=None)).run_id == first.run_id
        known = StubRunner(version="2.0")
        second = store.get_or_run(request, known)
        assert known.run_names == ["s1"] and second.run_id != first.run_id
        assert second.key == request.with_model_version("2.0").key()

    def test_another_model_version_is_another_run(self, store, tmp_path):
        runner = StubRunner(version="1.0")
        store.get_or_run(make_request(tmp_path, model_version=""), runner)
        runner._version = "1.1"
        store.get_or_run(make_request(tmp_path, model_version=""), runner)
        assert runner.run_names == ["s1", "s1"]

    def test_the_runner_must_be_for_the_model_of_the_request(self, store, tmp_path):
        class Other(StubRunner):
            name = "other"

        with pytest.raises(ValueError, match="'other' runner cannot run a request for the model"):
            store.get_or_run(make_request(tmp_path), Other())

    def test_a_miss_without_a_runner_is_an_error_and_a_hit_is_not(self, store, tmp_path):
        request = make_request(tmp_path)
        with pytest.raises(PredictionUnavailableError, match="no runner"):
            store.get_or_run(request, None)
        store.get_or_run(request, StubRunner())
        assert store.get_or_run(request, None).status == "done"


# ---------------------------------------------------------------------------- failures


class TestFailedRuns:
    def test_a_failure_is_recorded_with_the_exception_text_and_raised(self, store, tmp_path):
        request = make_request(tmp_path)
        boom = RuntimeError("CUDA out of memory (2 GiB)")
        with pytest.raises(PredictionFailedError, match="CUDA out of memory") as info:
            store.get_or_run(request, StubRunner(fail=boom))
        assert info.value.__cause__ is boom
        assert info.value.reason == "RuntimeError: CUDA out of memory (2 GiB)"
        assert info.value.key == request.key()

        entry = store.lookup(request)
        assert entry.status == "failed" and not entry.ok
        assert entry.reason == "RuntimeError: CUDA out of memory (2 GiB)"
        status = json.loads((entry.directory / "STATUS.json").read_text(encoding="utf-8"))
        assert status["status"] == "failed" and "CUDA out of memory" in status["reason"]

    def test_the_partial_output_of_a_failed_run_is_kept_for_diagnosis(self, store, tmp_path):
        entry = store.ensure(make_request(tmp_path), StubRunner(fail=ValueError("bad")))
        assert (entry.directory / "outputs" / "predictions" / "s1.txt").exists()

    def test_a_failed_run_is_not_retried_without_rerun(self, store, tmp_path):
        request = make_request(tmp_path)
        runner = StubRunner(fail=ValueError("bad input"))
        for _ in range(3):
            with pytest.raises(PredictionFailedError, match="bad input"):
                store.get_or_run(request, runner)
        assert runner.run_names == ["s1"]

    def test_a_recorded_failure_message_says_how_to_retry(self, store, tmp_path):
        request = make_request(tmp_path)
        with pytest.raises(PredictionFailedError):
            store.get_or_run(request, StubRunner(fail=ValueError("bad")))
        with pytest.raises(PredictionFailedError, match="rerun=True") as info:
            store.get_or_run(request, StubRunner())
        assert info.value.__cause__ is None  # found on disk, not raised by this call

    def test_rerun_runs_again_and_replaces_the_failure(self, store, tmp_path):
        request = make_request(tmp_path)
        store.ensure(request, StubRunner(fail=ValueError("bad")))
        runner = StubRunner()
        entry = store.get_or_run(request, runner, rerun=True)
        assert runner.run_names == ["s1"] and entry.status == "done" and entry.reason == ""
        assert store.lookup(request).status == "done"
        assert not list(entry.directory.parent.glob("*.old-*"))

    def test_rerun_replaces_a_finished_run_and_its_outputs(self, store, tmp_path):
        request = make_request(tmp_path)
        first = store.get_or_run(request, StubRunner())
        (first.prediction_dir / "stale.txt").write_text("from the first run", encoding="utf-8")
        second = store.get_or_run(request, StubRunner(), rerun=True)
        assert second.run_id != first.run_id and second.executed_here
        assert not (second.prediction_dir / "stale.txt").exists()

    def test_ensure_returns_a_failure_instead_of_raising(self, store, tmp_path):
        entry = store.ensure(make_request(tmp_path), StubRunner(fail=OSError("disk full")))
        assert entry.status == "failed" and entry.executed_here
        with pytest.raises(PredictionFailedError):
            entry.require_ok()

    def test_an_exception_without_text_is_recorded_by_its_type(self, store, tmp_path):
        entry = store.ensure(make_request(tmp_path), StubRunner(fail=TimeoutError()))
        assert entry.reason == "TimeoutError"

    def test_a_long_reason_is_cut(self, store, tmp_path):
        entry = store.ensure(make_request(tmp_path), StubRunner(fail=ValueError("x" * 9000)))
        assert len(entry.reason) == 4000

    def test_a_subprocess_error_is_recorded_with_its_own_text(self, store, tmp_path):
        error = subprocess.CalledProcessError(3, ["run_openfold"], stderr="boom")
        entry = store.ensure(make_request(tmp_path), StubRunner(fail=error))
        assert entry.reason.startswith("CalledProcessError: Command '['run_openfold']'")

    def test_a_runner_that_returns_a_path_outside_its_work_dir_fails_the_run(self, store, tmp_path):
        class Escaping(StubRunner):
            def run(self, request, work_dir):
                return tmp_path

        entry = store.ensure(make_request(tmp_path), Escaping())
        assert entry.status == "failed" and "not inside its work directory" in entry.reason

    def test_a_runner_that_returns_nothing_usable_fails_the_run(self, store, tmp_path):
        class Ghost(StubRunner):
            def run(self, request, work_dir):
                return Path(work_dir) / "never_written"

        entry = store.ensure(make_request(tmp_path), Ghost())
        assert entry.status == "failed" and "not a directory" in entry.reason

    def test_an_unavailable_runner_records_nothing(self, store, tmp_path):
        request = make_request(tmp_path)
        runner = StubRunner(available=False)
        with pytest.raises(PredictionUnavailableError, match="cannot be started"):
            store.get_or_run(request, runner)
        assert runner.run_names == [] and store.lookup(request) is None
        assert not store.path_for(request).exists()
        runner.available = True  # the user fixed the environment: no stale failure is remembered
        assert store.get_or_run(request, runner).status == "done"

    def test_an_unavailable_runner_does_not_hide_a_stored_result(self, store, tmp_path):
        request = make_request(tmp_path)
        store.get_or_run(request, StubRunner())
        assert store.get_or_run(request, StubRunner(available=False)).status == "done"


# ---------------------------------------------------------------------------- interrupted runs


def _list_entries(store, request):
    return sorted(p.name for p in store.path_for(request).parent.glob(f"{request.key()}*"))


class TestInterruptedRuns:
    def test_an_interrupt_leaves_no_done_marker_and_no_leftover(self, store, tmp_path):
        request = make_request(tmp_path)
        with pytest.raises(KeyboardInterrupt):
            store.get_or_run(request, StubRunner(fail=KeyboardInterrupt()))
        assert store.lookup(request) is None
        assert _list_entries(store, request) == [f"{request.key()}.lock"]

    def test_the_run_writes_into_a_temporary_directory_until_it_is_finished(self, store, tmp_path):
        request = make_request(tmp_path)
        seen = {}

        class Watching(StubRunner):
            def run(self, request, work_dir):
                seen["entry_exists"] = store.path_for(request).exists()
                seen["lookup"] = store.lookup(request)
                seen["work_dir"] = Path(work_dir)
                return super().run(request, work_dir)

        store.get_or_run(request, Watching())
        assert seen["entry_exists"] is False and seen["lookup"] is None
        assert ".tmp-" in str(seen["work_dir"])
        assert not seen["work_dir"].parent.exists()  # renamed into the entry

    @needs_fork
    def test_a_killed_process_leaves_no_done_marker(self, store, tmp_path):
        request = make_request(tmp_path)

        class Killed(StubRunner):
            def run(self, request, work_dir):
                (Path(work_dir) / "half_written.bin").write_bytes(b"\x00" * 100)
                os.kill(os.getpid(), signal.SIGKILL)

        child = multiprocessing.get_context("fork").Process(
            target=lambda: store.get_or_run(request, Killed())
        )
        child.start()
        child.join(60)
        assert child.exitcode == -signal.SIGKILL

        assert store.lookup(request) is None
        assert not store.path_for(request).exists()
        leftovers = [n for n in _list_entries(store, request) if ".tmp-" in n]
        assert len(leftovers) == 1  # the half-written outputs

        runner = StubRunner()  # the dead process's lock is gone: this neither hangs nor reuses
        entry = store.get_or_run(request, runner)
        assert runner.run_names == ["s1"] and entry.status == "done"
        assert not [n for n in _list_entries(store, request) if ".tmp-" in n]

    def test_a_leftover_temporary_directory_is_removed_by_the_next_run(self, store, tmp_path):
        request = make_request(tmp_path)
        leftover = store.path_for(request).parent / f"{request.key()}.tmp-99999-deadbeef"
        (leftover / "outputs").mkdir(parents=True)
        store.get_or_run(request, StubRunner())
        assert not leftover.exists()

    def test_an_entry_whose_status_is_unreadable_counts_as_absent(self, store, tmp_path, caplog):
        request = make_request(tmp_path)
        entry = store.get_or_run(request, StubRunner())
        (entry.directory / "STATUS.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=store_module.logger.name):
            assert store.lookup(request) is None
        assert "unreadable" in caplog.text
        runner = StubRunner()
        assert store.get_or_run(request, runner).status == "done"
        assert runner.run_names == ["s1"]

    def test_an_entry_whose_outputs_were_deleted_is_run_again(self, store, tmp_path):
        import shutil

        request = make_request(tmp_path)
        entry = store.get_or_run(request, StubRunner())
        shutil.rmtree(entry.directory / "outputs")
        assert store.lookup(request) is None
        runner = StubRunner()
        assert store.get_or_run(request, runner).prediction_dir.is_dir()
        assert runner.run_names == ["s1"]


# ---------------------------------------------------------------------------- processes


def _ask(store_root, counter, request, rerun, barrier, results):
    """Body of one worker process: ask for the request at the same moment as its peer."""
    runner = StubRunner(counter, sleep=0.6)
    barrier.wait(30)
    entry = PredictionStore(store_root).get_or_run(request, runner, rerun=rerun)
    results.put((os.getpid(), entry.executed_here, entry.status, entry.run_id))


@needs_fork
class TestTwoProcessesAskingAtOnce:
    def _race(self, store, request, tmp_path, *, rerun):
        context = multiprocessing.get_context("fork")
        counter = tmp_path / "runs.txt"
        barrier, results = context.Barrier(2), context.Queue()
        workers = [
            context.Process(
                target=_ask, args=(store.root, counter, request, rerun, barrier, results)
            )
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        answers = [results.get(timeout=60) for _ in workers]
        for worker in workers:
            worker.join(60)
            assert worker.exitcode == 0
        runs = counter.read_text(encoding="utf-8").splitlines() if counter.exists() else []
        return answers, runs

    def test_the_model_runs_once_and_both_get_the_output(self, store, tmp_path):
        request = make_request(tmp_path)
        answers, runs = self._race(store, request, tmp_path, rerun=False)
        assert len(runs) == 1
        assert sorted(executed for _, executed, _, _ in answers) == [False, True]
        assert {status for _, _, status, _ in answers} == {"done"}
        assert len({run_id for _, _, _, run_id in answers}) == 1
        runner_pid = next(pid for pid, executed, _, _ in answers if executed)
        assert runs[0].split()[0] == str(runner_pid)  # the counter agrees with executed_here

    def test_a_rerun_asked_by_both_at_once_reruns_once(self, store, tmp_path):
        request = make_request(tmp_path)
        store.get_or_run(request, StubRunner())  # something to rerun
        answers, runs = self._race(store, request, tmp_path, rerun=True)
        assert len(runs) == 1
        assert sorted(executed for _, executed, _, _ in answers) == [False, True]

    def test_a_failure_is_shared_and_not_run_twice(self, store, tmp_path):
        request = make_request(tmp_path)
        context = multiprocessing.get_context("fork")
        counter = tmp_path / "runs.txt"
        barrier, results = context.Barrier(2), context.Queue()

        def failing(rank):
            runner = StubRunner(counter, sleep=0.4, fail=ValueError("bad"))
            barrier.wait(30)
            entry = PredictionStore(store.root).ensure(request, runner)
            results.put((entry.status, entry.executed_here))

        workers = [context.Process(target=failing, args=(rank,)) for rank in range(2)]
        for worker in workers:
            worker.start()
        answers = [results.get(timeout=60) for _ in workers]
        for worker in workers:
            worker.join(60)
        assert len(counter.read_text(encoding="utf-8").splitlines()) == 1
        assert sorted(answers) == [("failed", False), ("failed", True)]


# ---------------------------------------------------------------------------- adoption


class TestAdoption:
    @pytest.fixture
    def outputs(self, tmp_path):
        folder = tmp_path / "my_own_run"
        (folder / "s1").mkdir(parents=True)
        (folder / "s1" / "result.txt").write_text("produced by the user", encoding="utf-8")
        return folder

    def test_adopted_outputs_are_never_run(self, store, tmp_path, outputs):
        request = make_request(tmp_path)
        adopted = store.adopt(request, outputs)
        runner = StubRunner()
        entry = store.get_or_run(request, runner)
        assert runner.run_names == []
        assert entry.status == "adopted" and entry.ok and not entry.executed_here
        assert entry.prediction_dir == outputs.resolve()
        assert entry.run_id == adopted.run_id
        assert entry.runner_name == ""

    def test_the_outputs_stay_where_they_are_by_default(self, store, tmp_path, outputs):
        entry = store.adopt(make_request(tmp_path), outputs)
        assert list((entry.directory / "outputs").iterdir()) == []
        status = json.loads((entry.directory / "STATUS.json").read_text(encoding="utf-8"))
        assert status["status"] == "adopted" and status["outputs_path"] == str(outputs.resolve())

    def test_the_outputs_can_be_copied_into_the_store(self, store, tmp_path, outputs):
        import shutil

        request = make_request(tmp_path)
        entry = store.adopt(request, outputs, copy_outputs=True)
        shutil.rmtree(outputs)
        assert entry.prediction_dir == entry.directory / "outputs"
        assert (store.lookup(request).prediction_dir / "s1" / "result.txt").exists()

    def test_adopting_again_changes_nothing(self, store, tmp_path, outputs):
        request = make_request(tmp_path)
        first = store.adopt(request, outputs)
        assert store.adopt(request, outputs).run_id == first.run_id

    def test_adopting_another_directory_replaces_the_reference(self, store, tmp_path, outputs):
        request = make_request(tmp_path)
        store.adopt(request, outputs)
        other = tmp_path / "other_run"
        other.mkdir()
        assert store.adopt(request, other).prediction_dir == other.resolve()

    def test_adopting_leaves_a_run_of_the_same_request_alone_and_wins_the_lookup(
        self, store, tmp_path, outputs
    ):
        request = make_request(tmp_path)
        ran = store.get_or_run(request, StubRunner())
        adopted = store.adopt(request, outputs)
        assert adopted.status == "adopted" and adopted.run_id != ran.run_id
        assert store.lookup(request).status == "adopted"
        status = json.loads((store.path_for(request) / "STATUS.json").read_text(encoding="utf-8"))
        assert status["status"] == "done" and status["run_id"] == ran.run_id  # the run is intact

    def test_two_samples_with_one_input_file_keep_their_own_adopted_outputs(self, store, tmp_path):
        first = make_request(tmp_path, name="a", content=b"the shared input")
        second = same_input_other_name(first, "b")
        assert first.key() == second.key()  # a run of either is the same run
        dir_a, dir_b = tmp_path / "out_a", tmp_path / "out_b"
        dir_a.mkdir()
        dir_b.mkdir()
        entry_a, entry_b = store.adopt(first, dir_a), store.adopt(second, dir_b)
        assert (entry_a.name, entry_b.name) == ("a", "b") and entry_a.key != entry_b.key
        assert store.lookup(first).prediction_dir == dir_a.resolve()
        assert store.lookup(second).prediction_dir == dir_b.resolve()
        runner = StubRunner()
        assert store.get_or_run(first, runner).prediction_dir == dir_a.resolve()
        assert store.get_or_run(second, runner).prediction_dir == dir_b.resolve()
        assert runner.run_names == []

    def test_a_name_that_was_not_adopted_does_not_see_another_names_outputs(
        self, store, tmp_path, outputs
    ):
        first = make_request(tmp_path, name="a", content=b"the shared input")
        second = same_input_other_name(first, "b")
        store.adopt(first, outputs)
        assert store.lookup(second) is None
        runner = StubRunner()
        assert store.get_or_run(second, runner).status == "done"
        assert runner.run_names == ["b"]

    def test_the_adopted_entry_records_the_name_in_its_key(self, store, tmp_path, outputs):
        request = make_request(tmp_path)
        entry = store.adopt(request, outputs)
        assert entry.key == request.for_adoption().key() != request.key()
        assert entry.directory == store.path_for(request.for_adoption())
        written = json.loads((entry.directory / "request.json").read_text(encoding="utf-8"))
        assert written["adopted_name"] == "s1" and written["key"] == entry.key

    def test_adoption_replaces_a_recorded_failure(self, store, tmp_path, outputs):
        request = make_request(tmp_path)
        store.ensure(request, StubRunner(fail=ValueError("bad")))
        assert store.adopt(request, outputs).status == "adopted"
        assert store.get_or_run(request, None).ok

    def test_a_missing_directory_is_refused(self, store, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            store.adopt(make_request(tmp_path), tmp_path / "nowhere")
        assert not store.root.exists()

    def test_a_file_is_not_a_directory(self, store, tmp_path):
        request = make_request(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.adopt(request, request.input_path)

    def test_outputs_adopted_without_a_version_are_found_by_a_runner_that_has_one(
        self, store, tmp_path, outputs
    ):
        request = make_request(tmp_path, model_version="")
        store.adopt(request, outputs)
        runner = StubRunner(version="5.0")
        entry = store.get_or_run(request, runner)
        assert runner.run_names == [] and entry.status == "adopted"

    def test_adopted_outputs_win_over_a_run_of_the_installed_version(
        self, store, tmp_path, outputs
    ):
        request = make_request(tmp_path, model_version="")
        runner = StubRunner(version="5.0")
        store.get_or_run(request, runner)  # a run exists under version 5.0
        store.adopt(request, outputs)  # the user then points at their own outputs
        entry = store.get_or_run(request, runner)
        assert entry.status == "adopted" and entry.prediction_dir == outputs.resolve()
        assert runner.run_names == ["s1"]

    def test_rerun_of_adopted_outputs_runs_the_model_and_leaves_the_users_files(
        self, store, tmp_path, outputs
    ):
        request = make_request(tmp_path)
        store.adopt(request, outputs)
        runner = StubRunner()
        entry = store.get_or_run(request, runner, rerun=True)
        assert runner.run_names == ["s1"] and entry.status == "done"
        assert (outputs / "s1" / "result.txt").exists()
        assert store.lookup(request).run_id == entry.run_id  # the fresh run is what is found now

    def test_adopted_outputs_that_vanished_count_as_absent(self, store, tmp_path, outputs):
        import shutil

        request = make_request(tmp_path)
        store.adopt(request, outputs)
        shutil.rmtree(outputs)
        assert store.lookup(request) is None


# ---------------------------------------------------------------------------- run_missing


class TestRunMissing:
    def _requests(self, tmp_path, count, **kwargs):
        return [
            make_request(tmp_path, name=f"s{i}", content=f"complex {i}".encode(), **kwargs)
            for i in range(count)
        ]

    def test_only_the_missing_requests_are_batched(self, store, tmp_path):
        requests = self._requests(tmp_path, 5)
        store.get_or_run(requests[1], StubRunner())
        store.adopt(requests[3], tmp_path)  # adopted: never run either
        runner = StubRunner()
        entries = store.run_missing(requests, runner)
        assert runner.batches == [["s0", "s2", "s4"]]
        assert [e.name for e in entries] == [f"s{i}" for i in range(5)]
        assert [e.status for e in entries] == ["done", "done", "done", "adopted", "done"]
        assert [e.executed_here for e in entries] == [True, False, True, False, True]

    def test_adopted_twins_keep_their_outputs_and_run_twins_share_one_run(self, store, tmp_path):
        a = make_request(tmp_path, name="a", content=b"twin input")
        b = same_input_other_name(a, "b")
        dir_a, dir_b = tmp_path / "out_a", tmp_path / "out_b"
        dir_a.mkdir()
        dir_b.mkdir()
        store.adopt(a, dir_a)
        store.adopt(b, dir_b)
        c = make_request(tmp_path, name="c", content=b"other input")
        d = same_input_other_name(c, "d")  # the same run under two names
        runner = StubRunner()
        entries = store.run_missing([a, b, c, d], runner)
        assert [e.status for e in entries] == ["adopted", "adopted", "done", "done"]
        assert [e.prediction_dir for e in entries[:2]] == [dir_a.resolve(), dir_b.resolve()]
        assert entries[2].key == entries[3].key and entries[2].run_id == entries[3].run_id
        assert runner.run_names == ["c"] and runner.batches == []

    def test_rerun_drops_the_adoption_and_runs_the_model(self, store, tmp_path):
        request = make_request(tmp_path)
        store.adopt(request, tmp_path)
        runner = StubRunner()
        (entry,) = store.run_missing([request], runner, rerun=True)
        assert entry.status == "done" and runner.run_names == ["s1"]
        assert store.lookup(request).status == "done"

    def test_every_entry_is_a_normal_entry_that_a_later_lookup_finds(self, store, tmp_path):
        requests = self._requests(tmp_path, 3)
        store.run_missing(requests, StubRunner())
        runner = StubRunner()
        for request in requests:
            entry = store.get_or_run(request, runner)
            assert entry.prediction_dir == entry.directory / "outputs"
            assert (entry.prediction_dir / f"{request.name}.txt").read_text(encoding="utf-8") == (
                "batched output"
            )
        assert runner.run_names == []

    def test_the_batch_workspace_is_removed(self, store, tmp_path):
        store.run_missing(self._requests(tmp_path, 3), StubRunner())
        assert not list((store.root / "stub").glob(".batch-*"))

    def test_nothing_missing_runs_nothing(self, store, tmp_path):
        requests = self._requests(tmp_path, 3)
        store.run_missing(requests, StubRunner())
        runner = StubRunner()
        entries = store.run_missing(requests, runner)
        assert runner.batches == [] and runner.run_names == []
        assert all(not e.executed_here for e in entries)

    def test_a_repeated_request_runs_once_and_is_returned_at_each_position(self, store, tmp_path):
        first, second = self._requests(tmp_path, 2)
        runner = StubRunner()
        entries = store.run_missing([first, second, first], runner)
        assert runner.batches == [["s0", "s1"]]
        assert entries[0].key == entries[2].key and entries[0].key != entries[1].key

    def test_a_single_missing_request_is_run_alone(self, store, tmp_path):
        runner = StubRunner()
        store.run_missing(self._requests(tmp_path, 1), runner)
        assert runner.batches == [] and runner.run_names == ["s0"]

    def test_a_runner_without_a_batched_mode_runs_them_one_by_one(self, store, tmp_path):
        runner = StubRunner(batchable=False)
        entries = store.run_missing(self._requests(tmp_path, 3), runner)
        assert runner.batches == [] and runner.run_names == ["s0", "s1", "s2"]
        assert all(e.status == "done" for e in entries)

    def test_a_request_the_runner_cannot_batch_runs_alone(self, store, tmp_path):
        class Picky(StubRunner):
            def supports_batch(self, request):
                return request.name != "s1"

        runner = Picky()
        store.run_missing(self._requests(tmp_path, 4), runner)
        assert runner.batches == [["s0", "s2", "s3"]] and runner.run_names.count("s1") == 1

    def test_requests_with_different_settings_are_batched_separately(self, store, tmp_path):
        score = self._requests(tmp_path, 2)
        refold = self._requests(tmp_path, 2, mode="refold")
        runner = StubRunner()
        store.run_missing(score + refold, runner)
        assert sorted(runner.batches) == [["s0", "s1"], ["s0", "s1"]]

    def test_a_long_list_is_split_and_a_last_single_request_runs_alone(self, store, tmp_path):
        runner = StubRunner()
        store.run_missing(self._requests(tmp_path, 5), runner, max_batch=2)
        assert runner.batches == [["s0", "s1"], ["s2", "s3"]]
        assert runner.run_names[-1] == "s4"

    def test_two_requests_with_one_name_never_share_a_batch(self, store, tmp_path):
        """The model would see one query for both."""
        a = make_request(tmp_path, name="same", content=b"first", subdir="a")
        b = make_request(tmp_path, name="same", content=b"second", subdir="b")
        c = make_request(tmp_path, name="other", content=b"third")
        runner = StubRunner()
        entries = store.run_missing([a, b, c], runner)
        assert runner.batches == [["same", "other"]]
        assert runner.run_names.count("same") == 2 and all(e.status == "done" for e in entries)

    def test_a_request_the_batch_failed_for_is_recorded_and_the_others_are_kept(
        self, store, tmp_path
    ):
        requests = self._requests(tmp_path, 3)
        runner = StubRunner(batch_fail={"s1": RuntimeError("query s1 failed: out of memory")})
        entries = store.run_missing(requests, runner)
        assert [e.status for e in entries] == ["done", "failed", "done"]
        assert entries[1].reason == "RuntimeError: query s1 failed: out of memory"
        with pytest.raises(PredictionFailedError, match="out of memory"):
            entries[1].require_ok()
        assert store.lookup(requests[1]).status == "failed"
        again = StubRunner()
        assert [e.status for e in store.run_missing(requests, again)] == ["done", "failed", "done"]
        assert again.batches == []  # the failure is not retried

    def test_a_batch_that_raises_fails_every_request_with_its_text(self, store, tmp_path):
        runner = StubRunner(batch_raise=OSError("no space left on device"))
        entries = store.run_missing(self._requests(tmp_path, 3), runner)
        assert {e.status for e in entries} == {"failed"}
        assert all("no space left on device" in e.reason for e in entries)

    def test_a_request_the_runner_forgot_is_a_failure(self, store, tmp_path):
        class Forgetful(StubRunner):
            def run_many(self, requests, work_dir):
                results = super().run_many(requests, work_dir)
                results.pop(requests[0].key())
                return results

        entries = store.run_missing(self._requests(tmp_path, 2), Forgetful())
        assert [e.status for e in entries] == ["failed", "done"]
        assert "no result" in entries[0].reason

    def test_a_result_outside_the_workspace_is_a_failure(self, store, tmp_path):
        class Escaping(StubRunner):
            def run_many(self, requests, work_dir):
                return {r.key(): tmp_path for r in requests}

        entries = store.run_missing(self._requests(tmp_path, 2), Escaping())
        assert all(e.status == "failed" and "not inside" in e.reason for e in entries)

    def test_rerun_runs_everything_again(self, store, tmp_path):
        requests = self._requests(tmp_path, 3)
        store.run_missing(requests, StubRunner())
        runner = StubRunner()
        entries = store.run_missing(requests, runner, rerun=True)
        assert runner.batches == [["s0", "s1", "s2"]] and all(e.executed_here for e in entries)

    def test_without_a_runner_only_a_complete_store_is_served(self, store, tmp_path):
        requests = self._requests(tmp_path, 2)
        store.get_or_run(requests[0], StubRunner())
        with pytest.raises(PredictionUnavailableError, match="no runner"):
            store.run_missing(requests, None)
        assert len(store.run_missing(requests[:1], None)) == 1

    def test_an_unavailable_runner_is_refused_before_anything_runs(self, store, tmp_path):
        runner = StubRunner(available=False)
        with pytest.raises(PredictionUnavailableError):
            store.run_missing(self._requests(tmp_path, 3), runner)
        assert runner.batches == [] and not store.root.exists()

    def test_a_runner_for_another_model_is_refused(self, store, tmp_path):
        class Other(StubRunner):
            name = "other"

        with pytest.raises(ValueError, match="cannot run a request"):
            store.run_missing(self._requests(tmp_path, 2), Other())

    def test_max_batch_must_be_positive(self, store, tmp_path):
        with pytest.raises(ValueError, match="max_batch"):
            store.run_missing(self._requests(tmp_path, 2), StubRunner(), max_batch=0)

    def test_the_workspace_of_a_batch_that_died_is_cleaned_up_by_the_next_one(
        self, store, tmp_path
    ):
        dead = store.root / "stub" / ".batch-1-dead"
        (dead / "work").mkdir(parents=True)
        (dead / ".lock").write_text("", encoding="utf-8")  # nobody holds its lock
        store.run_missing(self._requests(tmp_path, 2), StubRunner())
        assert not dead.exists()

    def test_the_workspace_of_a_batch_in_progress_is_left_alone(self, store, tmp_path):
        import fcntl

        alive = store.root / "stub" / ".batch-2-alive"
        alive.mkdir(parents=True)
        descriptor = os.open(alive / ".lock", os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            store.run_missing(self._requests(tmp_path, 2), StubRunner())
            assert alive.exists()
        finally:
            os.close(descriptor)

    def test_the_locks_of_the_batch_are_released_afterwards(self, store, tmp_path):
        import fcntl

        requests = self._requests(tmp_path, 3)
        store.run_missing(requests, StubRunner())
        for request in requests:
            descriptor = os.open(store.path_for(request).with_suffix(".lock"), os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)  # would raise if still held
            finally:
                os.close(descriptor)


# ---------------------------------------------------------------------------- module facts


def test_the_module_imports_only_the_standard_library():
    code = textwrap.dedent(
        """
        import sys
        import binding_metrics.predictors.store
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "openfold3", "gemmi")
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_the_errors_share_one_base_class():
    assert issubclass(PredictionFailedError, PredictionStoreError)
    assert issubclass(PredictionUnavailableError, PredictionStoreError)
    assert issubclass(PredictionStoreError, RuntimeError)


def test_a_stored_prediction_reports_ok_by_status():
    def entry(status):
        return StoredPrediction("k", "stub", "s", status, Path("d"), Path("p"))

    assert [entry(s).ok for s in ("done", "adopted", "failed")] == [True, True, False]
