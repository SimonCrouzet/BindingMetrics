"""Custom weights in a prediction request: the key, the hash cache and the runner contract.

Nothing here needs a model or a GPU. The weights are small files; what matters is that the key
follows their content, that a request without weights keeps the key it always had, and that a
runner that cannot use weights is refused before it runs and before anything is stored.
"""

import json
import logging
import os
import pickle
from pathlib import Path

import pytest

from binding_metrics.predictors import weights as weights_module
from binding_metrics.predictors.runners import PredictionRunner, require_weights_support
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import PredictionRequest, PredictionStore
from binding_metrics.predictors.weights import CACHE_FILE, WeightsRef, weights_reference
from tests.predictors.test_store import StubRunner, make_request

# Keys that the request produced before the ``weights`` field existed. They must not move: an
# existing prediction store stays valid.
KEY_OF_SEQUENCES_REQUEST = "374842fe1c1257bba5962351ca74ce25422cf7eef7dead29e9aad2015bca1cbd"
KEY_OF_FILE_REQUEST = "bcb3a1c99e523ae5b4c00a536099b5b764b21754b54f277e30f9886feb2ac72f"
BATCH_SIGNATURE_OF_FILE_REQUEST = "4a110b652f3092be1d2148526a9c67e1bac7f0ea038652e936925d35526e1511"


class WeightsRunner(StubRunner):
    """A stub runner that takes a weights file and records what it was given."""

    supports_custom_weights = True
    weights_kind = "file"

    def run(self, request, work_dir):
        self.weights_seen = None if request.weights is None else request.weights.path
        return super().run(request, work_dir)


class DirectoryRunner(WeightsRunner):
    weights_kind = "directory"


def write_weights(directory: Path, name="model.pt", content=b"weights v1") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def sha_reads(monkeypatch):
    """Count the files that ``weights`` reads (not those it takes from the cache)."""
    reads = []
    real = weights_module._sha256_of_file

    def counting(path):
        reads.append(Path(path))
        return real(path)

    monkeypatch.setattr(weights_module, "_sha256_of_file", counting)
    return reads


class TestKeyWithoutWeightsIsUnchanged:
    def test_a_request_of_sequences_has_its_old_key(self):
        request = PredictionRequest(
            "stub",
            "s1",
            sequences={"A": "GG", "B": "AC"},
            seeds=(1, 2),
            num_samples=3,
            model_version="1.0",
            options={
                "presets": ["predict", "low_mem"],
                "use_msa_server": True,
                "num_model_seeds": None,
            },
        )
        assert request.key() == KEY_OF_SEQUENCES_REQUEST

    def test_a_request_with_files_has_its_old_key_and_batch_signature(self, tmp_path):
        complex_file = tmp_path / "c.cif"
        complex_file.write_bytes(b"data_x\nloop_\n")
        template = tmp_path / "t.cif"
        template.write_bytes(b"template\n")
        request = PredictionRequest(
            "of3",
            "q",
            mode="refold",
            input_path=complex_file,
            binder_chain="B",
            receptor_chain="A",
            extra_files={"template_cif": template},
            seeds=(42,),
            num_samples=5,
            model_version="0.5.0",
            options={
                "presets": ["predict", "low_mem"],
                "use_msa_server": False,
                "binder_cyclic": "auto",
            },
        )
        assert request.key() == KEY_OF_FILE_REQUEST
        assert request.batch_signature() == BATCH_SIGNATURE_OF_FILE_REQUEST

    def test_the_canonical_form_has_no_weights_field_without_weights(self, tmp_path):
        request = make_request(tmp_path)
        assert "weights" not in request.canonical() and "weights" not in request.describe()
        assert request.weights is None


class TestKeyWithWeights:
    def test_weights_change_the_key(self, tmp_path):
        weights = write_weights(tmp_path / "w")
        plain = make_request(tmp_path)
        custom = make_request(tmp_path, weights=weights)
        assert custom.key() != plain.key()
        assert custom.key() == make_request(tmp_path, weights=weights).key()

    def test_the_content_decides_not_the_path(self, tmp_path):
        first = write_weights(tmp_path / "one", "a.pt", b"same bytes")
        second = write_weights(tmp_path / "elsewhere" / "deeper", "renamed.ckpt", b"same bytes")
        assert (
            make_request(tmp_path, weights=first).key()
            == make_request(tmp_path, weights=second).key()
        )

    def test_a_moved_file_gives_the_same_key(self, tmp_path):
        weights = write_weights(tmp_path / "old")
        before = make_request(tmp_path, weights=weights).key()
        moved = tmp_path / "new" / "model.pt"
        moved.parent.mkdir()
        weights.rename(moved)
        assert make_request(tmp_path, weights=moved).key() == before

    def test_changed_content_gives_another_key(self, tmp_path):
        weights = write_weights(tmp_path / "w", content=b"version one")
        before = make_request(tmp_path, weights=weights).key()
        weights.write_bytes(b"version two")
        assert make_request(tmp_path, weights=weights).key() != before

    def test_a_path_and_its_reference_give_one_key(self, tmp_path):
        weights = write_weights(tmp_path / "w")
        by_path = make_request(tmp_path, weights=weights)
        by_reference = make_request(tmp_path, weights=weights_reference(weights))
        assert by_path.key() == by_reference.key()
        assert isinstance(by_path.weights, WeightsRef)

    def test_the_canonical_form_holds_content_and_no_path(self, tmp_path):
        weights = write_weights(tmp_path / "w", content=b"abc")
        canonical = make_request(tmp_path, weights=weights).canonical()["weights"]
        assert set(canonical) == {"kind", "sha256", "size"}
        assert canonical["kind"] == "file" and canonical["size"] == 3
        assert str(tmp_path) not in json.dumps(canonical)

    def test_the_description_names_the_path_for_request_json(self, tmp_path):
        weights = write_weights(tmp_path / "w")
        described = make_request(tmp_path, weights=weights).describe()["weights"]
        assert described["path"] == str(weights) and described["n_files"] == 1
        assert len(described["sha256"]) == 64

    def test_a_file_and_a_directory_with_it_are_two_keys(self, tmp_path):
        weights = write_weights(tmp_path / "w")
        as_file = make_request(tmp_path, weights=weights)
        as_directory = make_request(tmp_path, weights=weights.parent)
        assert as_file.key() != as_directory.key()

    def test_the_batch_signature_follows_the_weights(self, tmp_path):
        one = write_weights(tmp_path / "one", content=b"one")
        two = write_weights(tmp_path / "two", content=b"two")
        a = make_request(tmp_path, "a", weights=one)
        b = make_request(tmp_path, "b", content=b"ATOM 2", weights=one)
        c = make_request(tmp_path, "c", weights=two)
        assert a.batch_signature() == b.batch_signature()  # samples that share the weights
        assert a.batch_signature() != c.batch_signature()  # other weights cannot share a batch
        assert a.batch_signature() != make_request(tmp_path, "a").batch_signature()

    def test_a_request_with_weights_can_be_pickled(self, tmp_path):
        request = make_request(tmp_path, weights=write_weights(tmp_path / "w"))
        clone = pickle.loads(pickle.dumps(request))
        assert clone.key() == request.key() and clone.weights == request.weights

    def test_for_adoption_and_a_new_version_keep_the_weights(self, tmp_path):
        request = make_request(tmp_path, weights=write_weights(tmp_path / "w"))
        assert request.for_adoption().weights == request.weights
        assert request.with_model_version("2.0").weights == request.weights

    def test_a_missing_file_is_a_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="do not exist"):
            make_request(tmp_path, weights=tmp_path / "nowhere.pt")


class TestDirectories:
    @staticmethod
    def _tree(root: Path):
        write_weights(root, "config.json", b"{}")
        write_weights(root / "shards", "part-1.bin", b"one")
        write_weights(root / "shards", "part-2.bin", b"two")
        return root

    def test_a_directory_is_described_by_a_manifest(self, tmp_path):
        reference = weights_reference(self._tree(tmp_path / "d"))
        assert reference.kind == "directory" and reference.n_files == 3
        assert reference.size == len(b"{}") + len(b"one") + len(b"two")
        assert len(reference.sha256) == 64

    def test_the_location_of_the_directory_does_not_matter(self, tmp_path):
        first = weights_reference(self._tree(tmp_path / "a"))
        second = weights_reference(self._tree(tmp_path / "elsewhere" / "b"))
        assert first.sha256 == second.sha256 and first.path != second.path

    @pytest.mark.parametrize(
        "change",
        ["edit", "rename", "add", "remove"],
    )
    def test_any_change_below_the_directory_changes_the_digest(self, tmp_path, change):
        root = self._tree(tmp_path / "d")
        before = weights_reference(root).sha256
        if change == "edit":
            (root / "shards" / "part-1.bin").write_bytes(b"uno")
        elif change == "rename":
            (root / "shards" / "part-2.bin").rename(root / "shards" / "part-3.bin")
        elif change == "add":
            write_weights(root, "extra.bin", b"x")
        else:
            (root / "config.json").unlink()
        assert weights_reference(root).sha256 != before

    def test_the_digest_is_the_hash_of_the_sorted_manifest(self, tmp_path):
        import hashlib

        root = self._tree(tmp_path / "d")
        rows = sorted(
            [
                p.relative_to(root).as_posix(),
                p.stat().st_size,
                hashlib.sha256(p.read_bytes()).hexdigest(),
            ]
            for p in root.rglob("*")
            if p.is_file()
        )
        expected = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        assert weights_reference(root).sha256 == expected

    def test_an_empty_directory_is_refused(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="holds no files"):
            weights_reference(tmp_path / "empty")

    @pytest.mark.parametrize(
        "expect, target, found", [("file", "dir", "directory"), ("directory", "file", "file")]
    )
    def test_the_expected_kind_is_enforced(self, tmp_path, expect, target, found):
        path = self._tree(tmp_path / "d") if target == "dir" else write_weights(tmp_path / "w")
        with pytest.raises(ValueError, match=f"are a {found}, and a {expect} is expected"):
            weights_reference(path, expect=expect)

    def test_each_file_of_a_directory_goes_through_the_cache(self, tmp_path, monkeypatch):
        root = self._tree(tmp_path / "d")
        cache = tmp_path / "cache"
        reads = sha_reads(monkeypatch)
        weights_reference(root, cache_dir=cache)
        assert len(reads) == 3
        weights_reference(root, cache_dir=cache)
        assert len(reads) == 3  # nothing was read the second time
        (root / "shards" / "part-1.bin").write_bytes(b"UNO")
        weights_reference(root, cache_dir=cache)
        assert len(reads) == 4  # only the file that changed


class TestTheHashCache:
    def test_a_second_call_does_not_read_the_file(self, tmp_path, monkeypatch):
        weights = write_weights(tmp_path / "w")
        reads = sha_reads(monkeypatch)
        first = weights_reference(weights, cache_dir=tmp_path / "cache")
        second = weights_reference(weights, cache_dir=tmp_path / "cache")
        assert len(reads) == 1 and first == second

    def test_the_cache_is_a_json_file_in_the_cache_directory(self, tmp_path):
        weights = write_weights(tmp_path / "w")
        weights_reference(weights, cache_dir=tmp_path / "cache")
        data = json.loads((tmp_path / "cache" / CACHE_FILE).read_text(encoding="utf-8"))
        entry = data["files"][str(weights)]
        assert (
            set(entry) == {"size", "mtime_ns", "sha256"} and entry["size"] == weights.stat().st_size
        )
        assert not list((tmp_path / "cache").glob("*.tmp-*"))  # written atomically

    def test_a_cache_hit_gives_the_hash_of_the_content(self, tmp_path):
        import hashlib

        weights = write_weights(tmp_path / "w", content=b"payload")
        reference = weights_reference(weights, cache_dir=tmp_path / "cache")
        assert reference.sha256 == hashlib.sha256(b"payload").hexdigest()
        assert weights_reference(weights, cache_dir=tmp_path / "cache").sha256 == reference.sha256

    def test_a_changed_size_is_a_miss(self, tmp_path, monkeypatch):
        weights = write_weights(tmp_path / "w", content=b"short")
        cache = tmp_path / "cache"
        weights_reference(weights, cache_dir=cache)
        reads = sha_reads(monkeypatch)
        weights.write_bytes(b"much longer than before")
        assert weights_reference(weights, cache_dir=cache).size == len(b"much longer than before")
        assert len(reads) == 1

    def test_a_changed_modification_time_is_a_miss(self, tmp_path, monkeypatch):
        weights = write_weights(tmp_path / "w", content=b"12345")
        cache = tmp_path / "cache"
        before = weights_reference(weights, cache_dir=cache)
        reads = sha_reads(monkeypatch)
        weights.write_bytes(b"54321")  # same size, another content
        stat = weights.stat()
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        after = weights_reference(weights, cache_dir=cache)
        assert len(reads) == 1 and after.sha256 != before.sha256

    def test_an_edit_in_place_with_the_same_size_and_mtime_is_not_detected(self, tmp_path):
        """The documented limit of the cache: the old hash is used."""
        weights = write_weights(tmp_path / "w", content=b"12345")
        cache = tmp_path / "cache"
        before = weights_reference(weights, cache_dir=cache)
        stat = weights.stat()
        weights.write_bytes(b"54321")
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert weights_reference(weights, cache_dir=cache).sha256 == before.sha256
        assert weights_reference(weights).sha256 != before.sha256  # without the cache it is seen

    @pytest.mark.parametrize(
        "content",
        [
            "not json at all",
            "[1, 2, 3]",
            '{"format": 99, "files": {}}',
            '{"format": 1, "files": 5}',
        ],
    )
    def test_a_corrupt_cache_is_ignored_and_rebuilt(self, tmp_path, monkeypatch, content, caplog):
        weights = write_weights(tmp_path / "w")
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / CACHE_FILE).write_text(content, encoding="utf-8")
        reads = sha_reads(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=weights_module.logger.name):
            reference = weights_reference(weights, cache_dir=cache)
        assert len(reads) == 1 and len(reference.sha256) == 64
        rebuilt = json.loads((cache / CACHE_FILE).read_text(encoding="utf-8"))
        assert rebuilt["format"] == 1 and str(weights) in rebuilt["files"]
        weights_reference(weights, cache_dir=cache)
        assert len(reads) == 1  # the rebuilt cache works

    def test_a_malformed_entry_is_dropped_and_the_others_are_kept(self, tmp_path, monkeypatch):
        good = write_weights(tmp_path / "g", "good.pt", b"good")
        bad = write_weights(tmp_path / "b", "bad.pt", b"bad")
        cache = tmp_path / "cache"
        weights_reference(good, cache_dir=cache)
        weights_reference(bad, cache_dir=cache)
        data = json.loads((cache / CACHE_FILE).read_text(encoding="utf-8"))
        data["files"][str(bad)]["sha256"] = "too short"
        (cache / CACHE_FILE).write_text(json.dumps(data), encoding="utf-8")
        reads = sha_reads(monkeypatch)
        weights_reference(good, cache_dir=cache)
        weights_reference(bad, cache_dir=cache)
        assert reads == [bad]

    def test_without_a_cache_directory_nothing_is_written(self, tmp_path):
        weights_reference(write_weights(tmp_path / "w"))
        assert not list(tmp_path.rglob(CACHE_FILE))

    def test_a_cache_directory_that_cannot_be_written_does_not_stop_the_hash(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where a directory is needed", encoding="utf-8")
        reference = weights_reference(write_weights(tmp_path / "w"), cache_dir=blocker / "cache")
        assert len(reference.sha256) == 64

    def test_what_another_process_wrote_meanwhile_is_merged(self, tmp_path):
        first = write_weights(tmp_path / "a", "a.pt", b"a")
        second = write_weights(tmp_path / "b", "b.pt", b"b")
        cache = tmp_path / "cache"
        late = weights_module._HashCache(cache)  # reads an empty cache
        weights_reference(first, cache_dir=cache)  # another process finishes first
        stat = second.stat()
        late.remember(second, stat.st_size, stat.st_mtime_ns, "0" * 64)
        late.save()
        files = json.loads((cache / CACHE_FILE).read_text(encoding="utf-8"))["files"]
        assert set(files) == {str(first), str(second)}

    def test_the_store_keeps_the_cache_in_its_root(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        weights = write_weights(tmp_path / "w")
        reference = store.weights_reference(weights)
        assert (tmp_path / "store" / CACHE_FILE).is_file()
        assert reference == weights_reference(weights)

    def test_a_large_file_is_announced_in_the_log(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(weights_module, "_ANNOUNCE_BYTES", 1)
        with caplog.at_level(logging.INFO, logger=weights_module.logger.name):
            weights_reference(write_weights(tmp_path / "w"))
        assert "hashing the weights file" in caplog.text


class TestTheRunnerContract:
    def test_the_base_class_does_not_support_weights(self):
        assert PredictionRunner.supports_custom_weights is False
        assert PredictionRunner.weights_kind == "file"

    def test_a_runner_without_support_is_refused_before_it_runs(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        runner = StubRunner()
        request = make_request(tmp_path, weights=write_weights(tmp_path / "w"))
        with pytest.raises(ValueError, match="the stub runner does not take custom weights"):
            store.get_or_run(request, runner)
        assert runner.run_names == []
        assert not (tmp_path / "store").exists()  # nothing was stored, not even a failure

    def test_the_message_names_the_weights_and_the_remedy(self, tmp_path):
        weights = write_weights(tmp_path / "w")
        request = make_request(tmp_path, weights=weights)
        with pytest.raises(ValueError) as caught:
            require_weights_support(StubRunner(), request)
        assert str(weights) in str(caught.value) and "supports_custom_weights" in str(caught.value)

    def test_ensure_and_run_missing_refuse_too(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        request = make_request(tmp_path, weights=write_weights(tmp_path / "w"))
        with pytest.raises(ValueError, match="custom weights"):
            store.ensure(request, StubRunner())
        with pytest.raises(ValueError, match="custom weights"):
            store.run_missing([request], StubRunner())

    def test_check_weights_is_available_to_a_runner_that_calls_it_itself(self, tmp_path):
        request = make_request(tmp_path, weights=write_weights(tmp_path / "w"))
        with pytest.raises(ValueError, match="custom weights"):
            StubRunner().check_weights(request)
        WeightsRunner().check_weights(request)  # a runner that supports them accepts

    def test_a_request_without_weights_is_never_refused(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        assert store.get_or_run(make_request(tmp_path), StubRunner()).status == "done"

    def test_a_duck_typed_runner_counts_as_not_supporting(self, tmp_path):
        class Plain:
            name = "stub"

        request = make_request(tmp_path, weights=write_weights(tmp_path / "w"))
        with pytest.raises(ValueError, match="custom weights"):
            require_weights_support(Plain(), request)

    def test_a_supporting_runner_runs_and_the_entry_keeps_the_weights(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        weights = write_weights(tmp_path / "w")
        runner = WeightsRunner()
        request = make_request(tmp_path, weights=store.weights_reference(weights))
        entry = store.get_or_run(request, runner)
        assert entry.status == "done" and runner.weights_seen == weights
        description = json.loads((entry.directory / "request.json").read_text(encoding="utf-8"))
        assert description["weights"]["path"] == str(weights)
        assert description["weights"]["sha256"] == request.weights.sha256

    def test_the_other_weights_are_another_entry_and_a_second_run(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        runner = WeightsRunner()
        one = write_weights(tmp_path / "one", content=b"one")
        two = write_weights(tmp_path / "two", content=b"two")
        for weights in (one, two, one):
            store.get_or_run(make_request(tmp_path, weights=weights), runner)
        assert len(runner.run_names) == 2  # the third request found the first entry
        store.get_or_run(make_request(tmp_path), runner)
        assert len(runner.run_names) == 3  # no weights: the default model, its own entry

    def test_the_kind_must_be_the_one_the_runner_takes(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        directory = tmp_path / "d"
        write_weights(directory, "a.bin")
        with pytest.raises(ValueError, match="takes its weights as a file, and .* is a directory"):
            store.get_or_run(make_request(tmp_path, weights=directory), WeightsRunner())
        file_request = make_request(tmp_path, weights=write_weights(tmp_path / "f"))
        with pytest.raises(ValueError, match="takes its weights as a directory, and .* is a file"):
            store.get_or_run(file_request, DirectoryRunner())
        assert store.get_or_run(make_request(tmp_path, weights=directory), DirectoryRunner()).ok


class TestTheRecord:
    def test_extras_hold_the_request_weights(self, tmp_path, monkeypatch):
        from binding_metrics.predictors.record import PredictionRecord

        class Parser:
            name = "stub"

            def load(self, directory, name, **kwargs):
                return PredictionRecord("stub", name)

            def complete(self, record):
                return record

        store = PredictionStore(tmp_path / "store")
        runner = WeightsRunner()
        weights = write_weights(tmp_path / "w")
        request = make_request(tmp_path, weights=store.weights_reference(weights))
        session = PredictionSession(store, [runner], {"stub": Parser()})
        record = session.record(request)
        assert record.extras["weights"] == request.weights.to_dict()
        assert record.extras["weights"]["path"] == str(weights)

    def test_a_record_without_weights_has_no_key(self, tmp_path):
        from binding_metrics.predictors.record import PredictionRecord

        class Parser:
            name = "stub"

            def load(self, directory, name, **kwargs):
                return PredictionRecord("stub", name)

            def complete(self, record):
                return record

        session = PredictionSession(
            PredictionStore(tmp_path / "s"), [StubRunner()], {"stub": Parser()}
        )
        assert "weights" not in session.record(make_request(tmp_path)).extras
