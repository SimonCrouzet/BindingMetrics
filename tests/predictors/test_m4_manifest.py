"""``PredictionSession.records``, ``lookup`` and the manifest of a set of predictions.

The samples are the synthetic outputs of the four models (``synth_<model>``) written by stub
runners and read back by the real adapters. A runner whose ``run`` raises stands for "no model
is started".
"""

import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from binding_metrics import predictors
from binding_metrics.predictors import manifest as manifest_module
from binding_metrics.predictors.manifest import COLUMNS, STATUS_MISSING, manifest
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import (
    PredictionFailedError,
    PredictionRequest,
    PredictionStore,
    PredictionUnavailableError,
    file_sha256,
)
from tests.predictors import synth_af2, synth_boltz2, synth_of3, synth_protenix, synth_stub
from tests.predictors.synth_stub import StubParser
from tests.predictors.test_m4_provenance import SynthRunner, make_request, truth_for


class FailingRunner(PredictionRunner):
    """A runner whose model runs out of memory."""

    name = "protenix"

    def prepare(self, request, work_dir):
        return Path(work_dir)

    def run(self, request, work_dir):
        raise RuntimeError("out of memory on the GPU")

    def version(self):
        return "1.0"


def boltz_score(seed_index, sample):
    return round(0.9 - 0.1 * (sample - 1), 3)  # the files are numbered by rank


def of3_score(seed_index, sample):
    return round(0.4 + 0.1 * seed_index + 0.01 * sample, 3)


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "store")


@pytest.fixture
def mixed(tmp_path, store):
    """Two models run, one run that failed, one set of outputs adopted, one request never made.

    boltz2: one request per seed (42 and 43), two samples each. of3: one request, the seeds
    9, 10 and 11, two samples each. protenix: a failed run. af2: adopted outputs, two samples.
    """
    boltz = SynthRunner("boltz2", synth_boltz2, score=boltz_score)
    of3 = SynthRunner("of3", synth_of3, score=of3_score)
    runners = [boltz, of3, FailingRunner()]
    session = PredictionSession(store, runners)

    boltz_requests = [
        make_request("boltz2", name=f"d_s{seed}", mode="score", seeds=(seed,)) for seed in (42, 43)
    ]
    of3_request = make_request("of3", name="d", mode="refold", seeds=(9, 10, 11))
    failed_request = make_request("protenix", name="d", seeds=(101,))
    for request in (*boltz_requests, of3_request):
        session.entry(request)
    assert session.prefetch([failed_request]) is None  # a failed run is recorded, not raised

    outputs = tmp_path / "mine"
    outputs.mkdir()
    for sample in (1, 2):
        synth_af2.write_prediction(outputs, "adopted", truth_for(1, sample), sample=sample)
    adopted_request = make_request("af2", name="adopted", seeds=(99,))
    session.adopt(adopted_request, outputs)

    never_run = make_request("of3", name="never", sequences={"A": "WWWW", "B": "YY"}, seeds=(1,))
    requests = [*boltz_requests, of3_request, failed_request, adopted_request, never_run]
    return SimpleNamespace(
        session=session,
        runners=runners,
        requests=requests,
        boltz=boltz_requests,
        of3=of3_request,
        failed=failed_request,
        adopted=adopted_request,
        never_run=never_run,
        run_counts={runner.name: list(runner.runs) for runner in (boltz, of3)},
    )


def count_by(rows, column):
    counts = {}
    for row in rows:
        counts[row[column]] = counts.get(row[column], 0) + 1
    return counts


def only(rows, **where):
    return [row for row in rows if all(row[key] == value for key, value in where.items())]


# ---------------------------------------------------------------------------- records, lookup


class TestRecords:
    def test_every_sample_comes_back_in_the_natural_order(self, store):
        session = PredictionSession(store, [SynthRunner("of3", synth_of3)])
        request = make_request("of3", seeds=(9, 10, 11), num_samples=2)
        poses = session.records(request)
        assert [(p.seed_index, p.sample) for p in poses] == [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
        ]
        assert [p.provenance["seed"] for p in poses] == [9, 9, 10, 10, 11, 11]
        assert all(p.provenance["backend"] == "of3" for p in poses)

    def test_the_seeds_9_10_11_are_in_numeric_not_string_order(self, store):
        session = PredictionSession(store, [SynthRunner("of3", synth_of3)])
        poses = session.records(make_request("of3", seeds=(9, 10, 11), num_samples=1))
        assert [p.extras["seed_value"] for p in poses] == ["9", "10", "11"]
        assert [p.provenance["seed"] for p in poses] == [9, 10, 11]

    def test_a_record_is_the_one_that_record_returns(self, store):
        session = PredictionSession(store, [SynthRunner("of3", synth_of3)])
        request = make_request("of3", seeds=(9, 10), num_samples=2)
        poses = session.records(request)
        assert session.record(request, seed_index=2, sample=1) is poses[2]
        assert session.stats()["parsed"] == 4
        assert session.records(request) == poses  # parsed once, same objects
        assert session.stats()["parsed"] == 4

    def test_the_model_runs_once(self, store):
        runner = SynthRunner("of3", synth_of3)
        session = PredictionSession(store, [runner])
        request = make_request("of3", seeds=(9,))
        session.records(request)
        session.records(request)
        assert runner.runs == ["design"]

    def test_a_model_without_a_seed_dimension_lists_its_samples(self, store):
        session = PredictionSession(store, [SynthRunner("boltz2", synth_boltz2)])
        poses = session.records(make_request("boltz2", mode="score", seeds=(5,), num_samples=3))
        assert [(p.seed_index, p.sample) for p in poses] == [(1, 1), (1, 2), (1, 3)]

    def test_a_failed_request_raises_as_record_does(self, store):
        session = PredictionSession(store, [FailingRunner()])
        with pytest.raises(PredictionFailedError, match="out of memory"):
            session.records(make_request("protenix", seeds=(101,)))

    def test_a_run_without_a_sample_gives_an_empty_list(self, store, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        session = PredictionSession(store)
        request = make_request("of3", seeds=(9,))
        session.adopt(request, empty, check=False)
        assert session.records(request) == []

    def test_the_chain_map_reaches_the_poses(self, store):
        session = PredictionSession(store, [SynthRunner("of3", synth_of3)])
        request = make_request("of3", seeds=(9,), num_samples=1)
        poses = session.records(request, chain_map={"A": "R", "B": "P"})
        assert poses[0].chain_map == {"A": "R", "B": "P"}


class TestLookup:
    def test_nothing_stored_is_none_and_nothing_runs(self, store):
        runner = SynthRunner("of3", synth_of3)
        session = PredictionSession(store, [runner])
        assert session.lookup(make_request("of3", seeds=(9,))) is None
        assert runner.runs == []
        assert session.stats()["requests"] == 0

    def test_a_stored_run_is_found_by_a_new_session(self, store):
        request = make_request("of3", seeds=(9,), model_version="1.0")
        PredictionSession(store, [SynthRunner("of3", synth_of3)]).entry(request)
        reader = PredictionSession(store)  # no runner at all
        entry = reader.lookup(request)
        assert entry is not None and entry.status == "done"
        assert reader.stats() == dict.fromkeys(reader.stats(), 0)

    def test_a_failed_run_is_returned_not_raised(self, store):
        request = make_request("protenix", seeds=(101,), model_version="1.0")
        session = PredictionSession(store, [FailingRunner()])
        with pytest.raises(PredictionFailedError):
            session.entry(request)
        entry = PredictionSession(store).lookup(request)
        assert entry.status == "failed" and "out of memory" in entry.reason

    def test_a_request_without_a_version_finds_the_run_made_under_the_runner_version(self, store):
        runner = SynthRunner("of3", synth_of3, version="0.5.0")
        session = PredictionSession(store, [runner])
        request = make_request("of3", seeds=(9,))  # no model_version
        session.entry(request)
        found = PredictionSession(store, [runner]).lookup(request)
        assert found is not None and found.status == "done"

    def test_the_memory_of_the_session_comes_first(self, store):
        session = PredictionSession(store, [SynthRunner("of3", synth_of3)])
        request = make_request("of3", seeds=(9,))
        entry = session.entry(request)
        assert session.lookup(request) is entry


# ---------------------------------------------------------------------------- the manifest


class TestTheRows:
    def test_one_row_per_sample_and_one_for_each_request_without_one(self, mixed):
        rows = manifest(mixed.session, mixed.requests)
        assert len(rows) == 2 + 4 + 6 + 1 + 1  # af2, boltz2, of3, never run, failed
        assert count_by(rows, "backend") == {"af2": 2, "boltz2": 4, "of3": 7, "protenix": 1}
        assert count_by(rows, "status") == {"adopted": 2, "done": 10, "failed": 1, "missing": 1}

    def test_every_row_has_the_columns_and_is_json(self, mixed):
        rows = manifest(mixed.session, mixed.requests)
        for row in rows:
            assert list(row)[: len(COLUMNS)] == list(COLUMNS)
        assert json.loads(json.dumps(rows)) == rows

    def test_the_rows_are_sorted_by_backend_seed_and_sample(self, mixed):
        rows = manifest(mixed.session, mixed.requests)
        assert [(r["backend"], r["seed"], r["sample"]) for r in rows] == [
            ("af2", 99, 1),
            ("af2", 99, 2),
            ("boltz2", 42, 1),
            ("boltz2", 42, 2),
            ("boltz2", 43, 1),
            ("boltz2", 43, 2),
            ("of3", 9, 1),
            ("of3", 9, 2),
            ("of3", 10, 1),
            ("of3", 10, 2),
            ("of3", 11, 1),
            ("of3", 11, 2),
            ("of3", None, None),  # the request that was never made follows those with a seed
            ("protenix", None, None),
        ]

    def test_openfold3_seeds_9_10_11_are_not_sorted_as_strings(self, mixed):
        rows = only(manifest(mixed.session, mixed.requests), backend="of3", status="done")
        assert [row["seed"] for row in rows] == [9, 9, 10, 10, 11, 11]
        assert [row["seed_index"] for row in rows] == [1, 1, 2, 2, 3, 3]

    def test_the_identity_columns_are_those_of_the_request(self, mixed):
        rows = manifest(mixed.session, mixed.requests)
        first = only(rows, backend="boltz2", seed=43, sample=2)[0]
        request = mixed.boltz[1]
        assert first["request_key"] == request.key()
        assert first["request_name"] == "d_s43"
        assert first["model_version"] == "1.0"
        assert first["status"] == "done"
        assert first["reason"] is None

    def test_the_file_columns_name_the_structure_and_its_hash(self, mixed):
        for row in manifest(mixed.session, mixed.requests):
            if row["sample"] is None:
                assert row["structure_path"] is None and row["content_hash"] is None
                continue
            path = Path(row["structure_path"])
            assert path.is_file()
            assert row["content_hash"] == file_sha256(path)

    def test_the_native_scores_are_flat_columns(self, mixed):
        rows = manifest(mixed.session, mixed.requests)
        row = only(rows, backend="of3", seed=9, sample=1)[0]
        pose = mixed.session.records(mixed.of3)[0]
        for name, value in pose.native_scores().items():
            assert row[name] == value, name
        assert "chain_ptm.A" in row and "chain_pair_iptm.(A, B)" in row

    def test_the_scalar_columns_cover_the_native_scalars(self, mixed):
        pose = mixed.session.records(mixed.of3)[0]
        scalars = {key for key in pose.native_scores() if "." not in key}
        assert scalars <= set(COLUMNS)
        row = only(manifest(mixed.session, mixed.of3), seed=9, sample=1)[0]
        assert [key for key in row if key not in COLUMNS] == [
            key for key in pose.native_scores() if "." in key
        ]

    def test_a_single_request_is_accepted_and_a_repeated_one_counts_once(self, mixed):
        once = manifest(mixed.session, mixed.of3)
        assert len(once) == 6
        twice = manifest(mixed.session, [mixed.of3, mixed.of3])
        assert twice == once

    def test_the_provenance_of_a_pose_and_its_row_agree(self, mixed):
        rows = manifest(mixed.session, mixed.of3)
        for row, pose in zip(rows, mixed.session.records(mixed.of3)):
            assert row["seed"] == pose.provenance["seed"]
            assert row["model_version"] == pose.provenance["model_version"]
            assert row["request_key"] == pose.provenance["request_key"]


class TestAnUnknownSeed:
    def _session(self, store):
        return PredictionSession(store, [SynthRunner("stub", synth_stub)], {"stub": StubParser()})

    def test_the_rows_of_a_request_with_unsorted_seeds_have_no_seed(self, store):
        session = self._session(store)
        request = make_request("stub", seeds=(8, 3), num_samples=2)
        session.entry(request)
        rows = manifest(session, request)
        assert [(r["seed"], r["seed_index"], r["sample"]) for r in rows] == [
            (None, 1, 1),
            (None, 1, 2),
            (None, 2, 1),
            (None, 2, 2),
        ]
        assert all(r["status"] == "done" and r["reason"] is None for r in rows)
        assert json.loads(json.dumps(rows)) == rows

    def test_sorted_seeds_give_their_values(self, store):
        session = self._session(store)
        request = make_request("stub", seeds=(3, 8), num_samples=1)
        session.entry(request)
        assert [r["seed"] for r in manifest(session, request)] == [3, 8]

    def test_a_row_without_a_seed_follows_the_rows_that_have_one(self, store):
        session = self._session(store)
        known = make_request("stub", name="a", seeds=(5,), num_samples=1)
        unknown = make_request("stub", name="b", seeds=(9, 2), num_samples=1)
        for request in (known, unknown):
            session.entry(request)
        rows = manifest(session, [unknown, known])
        assert [(r["request_name"], r["seed"]) for r in rows] == [
            ("a", 5),
            ("b", None),
            ("b", None),
        ]


class TestFailedAndMissingRequests:
    def test_a_failed_request_gives_one_row_with_its_stored_reason(self, mixed):
        rows = only(manifest(mixed.session, mixed.requests), backend="protenix")
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "failed"
        assert row["reason"] == "RuntimeError: out of memory on the GPU"
        assert row["request_key"] == mixed.failed.key()
        assert row["model_version"] == "1.0"  # the version the failed runner reported
        for column in ("seed", "seed_index", "sample", "ranking_score", "rank", "structure_path"):
            assert row[column] is None, column

    def test_a_fresh_session_reads_the_same_failure_from_the_store(self, mixed):
        reader = PredictionSession(mixed.session.store, mixed.runners)
        rows = only(manifest(reader, mixed.failed), status="failed")
        assert rows[0]["reason"] == "RuntimeError: out of memory on the GPU"

    def test_a_request_with_nothing_stored_is_a_missing_row_and_nothing_runs(self, mixed):
        rows = only(manifest(mixed.session, mixed.requests), status=STATUS_MISSING)
        assert len(rows) == 1
        assert rows[0]["request_key"] == mixed.never_run.key()
        assert rows[0]["reason"]
        assert mixed.runners[1].runs == mixed.run_counts["of3"]  # of3 did not run again

    def test_an_output_without_a_sample_is_one_row_with_a_reason(self, store, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        session = PredictionSession(store)
        request = make_request("of3", seeds=(9,))
        session.adopt(request, empty, check=False)
        (row,) = manifest(session, request)
        assert row["status"] == "adopted" and row["sample"] is None
        assert row["reason"] == "the output holds no sample"


class TestRanks:
    def test_boltz2_ranks_by_its_own_score_within_the_seed(self, mixed):
        rows = only(manifest(mixed.session, mixed.requests), backend="boltz2")
        assert [(r["seed"], r["sample"], r["ranking_score"], r["rank"]) for r in rows] == [
            (42, 1, 0.9, 1),
            (42, 2, 0.8, 2),
            (43, 1, 0.9, 1),
            (43, 2, 0.8, 2),
        ]

    def test_a_score_that_disagrees_with_the_numbering_wins(self, store):
        runner = SynthRunner("boltz2", synth_boltz2, score=lambda i, s: [0.5, 0.9, 0.7][s - 1])
        session = PredictionSession(store, [runner])
        request = make_request("boltz2", mode="score", seeds=(7,), num_samples=3)
        session.entry(request)
        rows = manifest(session, request)
        assert [(r["sample"], r["rank"]) for r in rows] == [(1, 3), (2, 1), (3, 2)]

    def test_ties_keep_the_natural_order(self, store):
        runner = SynthRunner("boltz2", synth_boltz2, score=lambda i, s: 0.6)
        session = PredictionSession(store, [runner])
        request = make_request("boltz2", mode="score", seeds=(7,), num_samples=3)
        session.entry(request)
        assert [r["rank"] for r in manifest(session, request)] == [1, 2, 3]

    def test_openfold3_samples_are_unranked_although_they_have_a_score(self, mixed):
        rows = only(manifest(mixed.session, mixed.requests), backend="of3", status="done")
        assert all(row["rank"] is None for row in rows)
        assert [row["ranking_score"] for row in rows] == [0.51, 0.52, 0.61, 0.62, 0.71, 0.72]

    def test_protenix_ranks_inside_each_seed(self, store):
        runner = SynthRunner("protenix", synth_protenix, score=lambda i, s: 0.9 - 0.1 * s - i)
        session = PredictionSession(store, [runner])
        request = make_request("protenix", seeds=(9, 10), num_samples=2)
        session.entry(request)
        rows = manifest(session, request)
        assert [(r["seed"], r["sample"], r["rank"]) for r in rows] == [
            (9, 1, 1),
            (9, 2, 2),
            (10, 1, 1),
            (10, 2, 2),
        ]

    def test_colabfold_takes_the_rank_from_its_numbering_when_it_has_no_score(self, mixed):
        rows = only(manifest(mixed.session, mixed.requests), backend="af2")
        assert [row["ranking_score"] for row in rows] == [None, None]
        assert [row["rank"] for row in rows] == [1, 2]


class TestNoModelIsStarted:
    def _tripwire(self, monkeypatch, runners):
        def boom(*args, **kwargs):
            raise AssertionError("a model was started")

        for runner in runners:
            monkeypatch.setattr(runner, "run", boom)
            monkeypatch.setattr(runner, "run_many", boom, raising=False)
            monkeypatch.setattr(runner, "prepare", boom)

    def test_the_manifest_of_a_mixed_set_runs_nothing(self, mixed, monkeypatch):
        self._tripwire(monkeypatch, mixed.runners)
        before = mixed.session.stats()
        rows = manifest(mixed.session, mixed.requests)
        assert len(rows) == 14
        assert mixed.session.stats()["runs"] == before["runs"]

    def test_a_rerun_session_runs_nothing_either(self, mixed, monkeypatch):
        self._tripwire(monkeypatch, mixed.runners)
        rerun = PredictionSession(mixed.session.store, mixed.runners, rerun=True)
        rows = manifest(rerun, mixed.requests)
        assert len(rows) == 14
        assert rerun.stats()["runs"] == 0 and rerun.stats()["misses"] == 0

    def test_a_session_with_runners_that_cannot_start_reads_the_store(self, mixed, monkeypatch):
        for runner in mixed.runners:
            monkeypatch.setattr(runner, "is_available", lambda: False)
        rows = manifest(PredictionSession(mixed.session.store, mixed.runners), mixed.requests)
        assert len(rows) == 14

    def test_a_missing_request_does_not_raise_when_there_is_no_runner(self, store):
        (row,) = manifest(PredictionSession(store), make_request("of3", seeds=(9,)))
        assert row["status"] == STATUS_MISSING
        with pytest.raises(PredictionUnavailableError):
            PredictionSession(store).entry(make_request("of3", seeds=(9,)))


class TestDeterminism:
    def test_the_order_of_the_requests_changes_nothing(self, mixed):
        expected = manifest(mixed.session, mixed.requests)
        for seed in range(5):
            shuffled = list(mixed.requests)
            random.Random(seed).shuffle(shuffled)
            assert manifest(mixed.session, shuffled) == expected

    def test_a_second_session_on_the_same_store_gives_the_same_rows(self, mixed):
        expected = manifest(mixed.session, mixed.requests)
        reader = PredictionSession(mixed.session.store, mixed.runners)
        assert manifest(reader, mixed.requests) == expected

    def test_two_requests_of_one_model_with_one_seed_are_told_apart_by_name(self, store):
        runner = SynthRunner("boltz2", synth_boltz2)
        session = PredictionSession(store, [runner])
        one = make_request("boltz2", name="b", mode="score", seeds=(42,), num_samples=1)
        other = make_request(
            "boltz2", name="a", mode="score", seeds=(42,), num_samples=1, sequences={"A": "WWW"}
        )
        for request in (one, other):
            session.entry(request)
        rows = manifest(session, [one, other])
        assert [row["request_name"] for row in rows] == ["a", "b"]
        assert manifest(session, [other, one]) == rows


class TestThePackageExport:
    def test_the_function_is_exported_as_prediction_manifest(self):
        assert predictors.prediction_manifest is manifest_module.manifest
        assert "prediction_manifest" in predictors.__all__
        assert predictors._EXPORTS["prediction_manifest"] == "binding_metrics.predictors.manifest"

    def test_the_name_manifest_stays_the_submodule(self):
        assert predictors.manifest is manifest_module  # importing the module binds its name

    def test_importing_the_package_does_not_import_the_manifest_module(self):
        code = textwrap.dedent(
            """
            import sys
            import binding_metrics.predictors as p
            assert "binding_metrics.predictors.manifest" not in sys.modules
            p.prediction_manifest
            assert "binding_metrics.predictors.manifest" in sys.modules
            """
        )
        subprocess.run([sys.executable, "-c", code], check=True, encoding="utf-8")


class TestRequestsAreNotChanged:
    def test_reading_a_manifest_leaves_the_keys_of_the_requests(self, mixed):
        before = [request.key() for request in mixed.requests]
        manifest(mixed.session, mixed.requests)
        assert [request.key() for request in mixed.requests] == before
        assert isinstance(mixed.requests[0], PredictionRequest)
