"""A run is refused when a file of the request changed since the request was made.

The request hashes its input and extra files when it is made and a runner reads them later; an
edit in between would store the output of the new content under the key of the old one. The
stub runner of ``test_m4_provenance`` records every run, so "the model is not started" is a
statement about ``runner.runs``.
"""

import pickle

import pytest

from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import KEY_FORMAT, PredictionRequest, PredictionStore
from tests.predictors import synth_of3
from tests.predictors.test_m4_provenance import SynthRunner


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "store")


@pytest.fixture
def runner():
    return SynthRunner("of3", synth_of3)


@pytest.fixture
def files(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    structure = folder / "complex.cif"
    structure.write_text("data_complex\n", encoding="utf-8")
    msa = folder / "receptor.a3m"
    msa.write_text(">query\nACDEFGH\n", encoding="utf-8")
    return structure, msa


def make_request(files, name="design", **overrides):
    structure, msa = files
    fields = {
        "mode": "score",
        "input_path": structure,
        "extra_files": {"receptor_msa": msa},
        "binder_chain": "B",
        "receptor_chain": "A",
        "seeds": (9,),
        "num_samples": 1,
        "model_version": "1.0",
    }
    fields.update(overrides)
    return PredictionRequest("of3", name, **fields)


def nothing_is_stored(store):
    return not list(store.root.rglob("STATUS.json"))


# ---------------------------------------------------------------------------- verify_files


class TestVerifyFiles:
    def test_unchanged_files_pass(self, files):
        request = make_request(files)
        assert request.verify_files() is None

    def test_a_request_with_sequences_only_has_no_file_to_check(self):
        request = PredictionRequest("of3", "q", mode="predict", sequences={"A": "ACDE"})
        request.verify_files()

    def test_an_edited_input_names_its_role(self, files):
        request = make_request(files)
        files[0].write_text("data_other\n", encoding="utf-8")
        with pytest.raises(
            ValueError, match=r"the input file .* changed since the request was made"
        ):
            request.verify_files()

    def test_an_edited_extra_file_names_its_role(self, files):
        request = make_request(files)
        files[1].write_text(">query\nWWWWWWW\n", encoding="utf-8")
        with pytest.raises(
            ValueError, match=r"the receptor_msa file .* changed since the request was made"
        ):
            request.verify_files()

    def test_a_file_with_the_same_content_written_again_passes(self, files):
        request = make_request(files)
        files[0].write_text(files[0].read_text(encoding="utf-8"), encoding="utf-8")
        request.verify_files()

    def test_a_missing_file_is_refused_unless_allowed(self, files):
        request = make_request(files)
        files[1].unlink()
        with pytest.raises(ValueError, match=r"the receptor_msa file .* cannot be read now"):
            request.verify_files()
        request.verify_files(allow_missing=True)

    def test_allowing_a_missing_file_still_refuses_a_changed_one(self, files):
        request = make_request(files)
        files[1].unlink()
        files[0].write_text("data_other\n", encoding="utf-8")
        with pytest.raises(ValueError, match="the input file"):
            request.verify_files(allow_missing=True)

    def test_the_message_names_the_request_and_says_nothing_ran(self, files):
        request = make_request(files, name="my_design")
        files[0].write_text("data_other\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"of3 request 'my_design'.*nothing was run"):
            request.verify_files()

    def test_the_key_does_not_change(self, files):
        request = make_request(files)
        key = request.key()
        request.verify_files()
        files[0].write_text("data_other\n", encoding="utf-8")
        with pytest.raises(ValueError):
            request.verify_files()
        assert request.key() == key  # still the key of the content it was made from
        assert KEY_FORMAT == 1
        assert "verify" not in " ".join(request.canonical())

    def test_copies_of_a_request_are_checked_too(self, files):
        request = make_request(files)
        files[0].write_text("data_other\n", encoding="utf-8")
        for copy in (request.with_model_version("2.0"), request.for_adoption()):
            with pytest.raises(ValueError, match="changed since the request was made"):
                copy.verify_files()


# ---------------------------------------------------------------------------- the store


class TestEnsure:
    def test_an_unchanged_request_runs(self, store, runner, files):
        entry = store.ensure(make_request(files), runner)
        assert entry.status == "done" and runner.runs == ["design"]

    @pytest.mark.parametrize("which, role", [(0, "input"), (1, "receptor_msa")])
    def test_a_file_edited_after_the_request_refuses_the_run(
        self, store, runner, files, which, role
    ):
        request = make_request(files)
        files[which].write_text("edited after the request was made\n", encoding="utf-8")
        with pytest.raises(ValueError, match=f"the {role} file"):
            store.ensure(request, runner)
        assert runner.runs == []  # the model was not started
        assert nothing_is_stored(store)
        assert store.lookup(request) is None

    def test_get_or_run_refuses_too(self, store, runner, files):
        request = make_request(files)
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError, match="changed since the request was made"):
            store.get_or_run(request, runner)
        assert runner.runs == [] and nothing_is_stored(store)

    def test_a_deleted_file_refuses_the_run(self, store, runner, files):
        request = make_request(files)
        files[1].unlink()
        with pytest.raises(ValueError, match="cannot be read now"):
            store.ensure(request, runner)
        assert runner.runs == [] and nothing_is_stored(store)

    def test_a_request_without_a_version_is_checked_as_it_is_run(self, store, runner, files):
        request = make_request(files, model_version="")  # the runner's version is filled in
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError, match="changed since the request was made"):
            store.ensure(request, runner)
        assert runner.runs == []

    def test_a_request_made_again_from_the_new_content_runs(self, store, runner, files):
        files[0].write_text("edited\n", encoding="utf-8")
        entry = store.ensure(make_request(files), runner)
        assert entry.status == "done"

    def test_a_request_restored_from_a_pickle_is_checked(self, store, runner, files):
        restored = pickle.loads(pickle.dumps(make_request(files)))
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError, match="changed since the request was made"):
            store.ensure(restored, runner)
        assert runner.runs == []


class TestAFinishedRunNeedsNoFile:
    def test_a_stored_run_is_found_after_its_input_moved(self, store, runner, files, tmp_path):
        request = make_request(files)
        first = store.ensure(request, runner)
        moved = tmp_path / "moved"
        moved.mkdir()
        files[0].rename(moved / "complex.cif")
        files[1].unlink()
        again = store.ensure(request, runner)
        assert again.run_id == first.run_id and runner.runs == ["design"]
        assert store.lookup(request) is not None

    def test_a_session_serves_a_stored_run_after_the_input_changed(self, store, runner, files):
        request = make_request(files)
        store.ensure(request, runner)
        files[0].write_text("edited\n", encoding="utf-8")
        pose = PredictionSession(store, [runner]).record(request)
        assert pose.provenance["request_key"] == request.key()
        assert runner.runs == ["design"]

    def test_a_rerun_with_a_moved_input_is_refused_and_keeps_the_entry(self, store, runner, files):
        request = make_request(files)
        first = store.ensure(request, runner)
        files[0].unlink()
        with pytest.raises(ValueError, match="cannot be read now"):
            store.ensure(request, runner, rerun=True)
        assert store.lookup(request).run_id == first.run_id
        assert runner.runs == ["design"]

    def test_a_rerun_with_an_edited_input_keeps_the_adopted_outputs(
        self, store, runner, files, tmp_path
    ):
        outputs = tmp_path / "mine"
        outputs.mkdir()
        request = make_request(files)
        store.adopt(request, outputs)
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError, match="changed since the request was made"):
            store.ensure(request, runner, rerun=True)
        adopted = store.lookup(request)
        assert adopted is not None and adopted.status == "adopted"  # the adoption was not dropped
        assert runner.runs == []


class TestRunMissing:
    def test_nothing_runs_when_one_request_of_the_group_changed(self, store, runner, files):
        first = make_request(files, name="first")
        second = make_request(files, name="second", extra_files={})
        files[1].write_text(">query\nWWWWWWW\n", encoding="utf-8")  # only `first` hashed it
        with pytest.raises(ValueError, match="the receptor_msa file"):
            store.run_missing([second, first], runner)
        assert runner.runs == [] and nothing_is_stored(store)

    def test_an_unchanged_group_runs(self, store, runner, files):
        requests = [
            make_request(files, name="first"),
            make_request(files, name="second", seeds=(10,)),
        ]
        entries = store.run_missing(requests, runner)
        assert [entry.status for entry in entries] == ["done", "done"]
        assert sorted(runner.runs) == ["first", "second"]

    def test_requests_that_are_stored_need_no_file(self, store, runner, files):
        request = make_request(files)
        store.ensure(request, runner)
        files[0].unlink()
        (entry,) = store.run_missing([request], runner)
        assert entry.status == "done" and runner.runs == ["design"]


class TestSession:
    def test_entry_and_record_refuse_and_the_model_is_not_started(self, store, runner, files):
        session = PredictionSession(store, [runner])
        request = make_request(files)
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError, match="changed since the request was made"):
            session.entry(request)
        with pytest.raises(ValueError, match="changed since the request was made"):
            session.record(request)
        assert runner.runs == [] and nothing_is_stored(store)

    def test_prefetch_refuses_before_any_run(self, store, runner, files):
        session = PredictionSession(store, [runner])
        request = make_request(files)
        files[1].write_text(">query\nWWWWWWW\n", encoding="utf-8")
        with pytest.raises(ValueError, match="the receptor_msa file"):
            session.prefetch([request])
        assert runner.runs == [] and nothing_is_stored(store)

    def test_a_refusal_is_not_remembered_as_a_failed_run(self, store, runner, files):
        session = PredictionSession(store, [runner])
        request = make_request(files)
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError):
            session.entry(request)
        assert session.stats()["failed"] == 0
        assert session.lookup(request) is None


class TestAdopt:
    def test_an_edited_file_refuses_the_adoption(self, store, files, tmp_path):
        outputs = tmp_path / "mine"
        outputs.mkdir()
        request = make_request(files)
        files[0].write_text("edited\n", encoding="utf-8")
        with pytest.raises(ValueError, match="changed since the request was made"):
            store.adopt(request, outputs)
        assert store.lookup(request) is None

    def test_a_missing_input_does_not_refuse_it(self, store, files, tmp_path):
        """Parse-only usage may happen where the inputs are not."""
        outputs = tmp_path / "mine"
        outputs.mkdir()
        request = make_request(files)
        files[0].unlink()
        files[1].unlink()
        assert store.adopt(request, outputs).status == "adopted"

    def test_an_unchanged_request_is_adopted(self, store, files, tmp_path):
        outputs = tmp_path / "mine"
        outputs.mkdir()
        assert store.adopt(make_request(files), outputs).status == "adopted"
