"""Seeds of Protenix and ColabFold requests: ``make_requests``, ``records`` and the manifest.

Protenix takes the seeds as one list, so one request carries them all. ColabFold runs
``range(random_seed, random_seed + num_seeds)``: ascending consecutive seeds are one request and
any other list is one request per seed. The runners below are the real ones with a ``run`` that
writes the synthetic output of the model (``synth_protenix``, ``synth_af2``) under the seed
values of the request, so that the real adapters read it back through a session. No model, GPU
or network is involved.
"""

import contextlib
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from binding_metrics.predictors import prediction_manifest  # noqa: E402
from binding_metrics.predictors.af2_runner import ColabFoldRunner  # noqa: E402
from binding_metrics.predictors.protenix_runner import ProtenixRunner  # noqa: E402
from binding_metrics.predictors.runners import PredictionRunner  # noqa: E402
from binding_metrics.predictors.session import PredictionSession  # noqa: E402
from binding_metrics.predictors.store import PredictionStore  # noqa: E402
from tests.predictors import synth_af2, synth_protenix  # noqa: E402
from tests.predictors.test_m4_provenance import truth_for  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
ROLES = {"name": "p53", "binder_chain": "B", "receptor_chain": "A"}


@pytest.fixture(autouse=True)
def installed_models(tmp_path, monkeypatch):
    """A version for both models and no weights, whatever this machine has."""
    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(tmp_path / "protenix_root"))
    monkeypatch.setattr(ProtenixRunner, "version", lambda self: "2.0.0")
    monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.6.3")


@contextlib.contextmanager
def seed_values(writer, values):
    """Make a synthetic writer name the seed at position ``i`` ``values[i - 1]``."""
    original = writer.seed_value
    writer.seed_value = lambda seed_index: values[seed_index - 1]
    try:
        yield
    finally:
        writer.seed_value = original


def score_of(seed_index, sample):
    """The second sample of a seed scores higher, so its rank is 1."""
    return round(0.5 + 0.1 * sample - 0.01 * seed_index, 3)


class WritingRunner:
    """Mixin: ``run`` writes the synthetic output of the model under the seeds of the request.

    The adapters order the seeds by value, so the output lists ``sorted(request.seeds)``
    whatever the order of the request.
    """

    writer = None

    def is_available(self):
        return True

    def prepare(self, request, work_dir):
        return Path(work_dir)

    def run(self, request, work_dir):
        values = sorted(request.seeds)
        self.runs = [*getattr(self, "runs", []), request.name]
        with seed_values(self.writer, values):
            for seed_index in range(1, len(values) + 1):
                for sample in range(1, request.num_samples + 1):
                    self.writer.write_prediction(
                        work_dir,
                        request.name,
                        truth_for(seed_index, sample, score_of(seed_index, sample)),
                        seed_index=seed_index,
                        sample=sample,
                    )
        return Path(work_dir)


class WritingProtenix(WritingRunner, ProtenixRunner):
    writer = synth_protenix


class WritingColabFold(WritingRunner, ColabFoldRunner):
    writer = synth_af2


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "store")


def seeds_of(requests):
    return [tuple(request.seeds) for request in requests]


def names_of(requests):
    return [request.name for request in requests]


def after(command, flag):
    return command[command.index(flag) + 1]


# ---------------------------------------------------------------------------- Protenix


class TestProtenixCarriesAllTheSeeds:
    def test_the_runner_has_a_seed_dimension(self):
        assert ProtenixRunner.seed_dimension is True

    def test_one_request_holds_the_seeds_in_the_order_given(self):
        (request,) = ProtenixRunner().make_requests(P53, seeds=[42, 7, 11], **ROLES)
        assert tuple(request.seeds) == (42, 7, 11)
        assert request.name == "p53"

    def test_one_seed_keeps_the_name(self):
        (request,) = ProtenixRunner().make_requests(P53, seeds=[3], **ROLES)
        assert tuple(request.seeds) == (3,) and request.name == "p53"

    def test_the_request_is_the_one_make_request_builds(self):
        runner = ProtenixRunner()
        (request,) = runner.make_requests(P53, seeds=[1, 2, 3], num_samples=2, **ROLES)
        direct = runner.make_request(P53, seeds=[1, 2, 3], num_samples=2, **ROLES)
        assert request == direct and request.key() == direct.key()

    def test_another_set_of_seeds_is_another_key(self):
        runner = ProtenixRunner()
        keys = {
            runner.make_request(P53, seeds=seeds, **ROLES).key()
            for seeds in ([1, 2], [2, 1], [1, 2, 3], [1], [2])
        }
        assert len(keys) == 5

    def test_the_command_passes_the_seeds_joined_by_commas(self):
        runner = ProtenixRunner()
        (request,) = runner.make_requests(P53, seeds=[42, 7, 11], **ROLES)
        command = runner._command(request, Path("in.json"), Path("out"))
        assert after(command, "--seeds") == "42,7,11"

    @pytest.mark.parametrize(
        "seeds, message",
        [([], "at least one seed"), ([1, 1], "distinct"), ([-1, 2], "not be negative")],
    )
    def test_a_list_the_model_cannot_take_is_refused(self, seeds, message):
        with pytest.raises(ValueError, match=message):
            ProtenixRunner().make_requests(P53, seeds=seeds, **ROLES)


# ---------------------------------------------------------------------------- ColabFold


class TestColabFoldGroupsConsecutiveSeeds:
    def test_the_runner_groups_by_rule_not_by_dimension(self):
        assert ColabFoldRunner.seed_dimension is False
        assert "_group_seeds" in vars(ColabFoldRunner)

    @pytest.mark.parametrize(
        "seeds, groups",
        [
            ([0, 1, 2], [(0, 1, 2)]),
            ([5, 6, 7], [(5, 6, 7)]),
            ([4], [(4,)]),
            ([0, 2], [(0,), (2,)]),
            ([0, 1, 3], [(0,), (1,), (3,)]),
            ([2, 1, 0], [(2,), (1,), (0,)]),
            ([1, 0], [(1,), (0,)]),
            ([], []),
        ],
    )
    def test_the_groups(self, seeds, groups):
        assert ColabFoldRunner()._group_seeds(seeds) == groups

    def test_consecutive_seeds_are_one_request_named_by_the_job(self):
        (request,) = ColabFoldRunner().make_requests(P53, seeds=[0, 1], **ROLES)
        assert tuple(request.seeds) == (0, 1) and request.name == "p53"

    def test_consecutive_seeds_from_a_first_value_other_than_zero(self):
        (request,) = ColabFoldRunner().make_requests(P53, seeds=[5, 6, 7], **ROLES)
        assert tuple(request.seeds) == (5, 6, 7)

    def test_other_seeds_get_a_request_each_named_after_the_seed(self):
        requests = ColabFoldRunner().make_requests(P53, seeds=[0, 2], **ROLES)
        assert seeds_of(requests) == [(0,), (2,)]
        assert names_of(requests) == ["p53_s0", "p53_s2"]
        assert requests[0].key() != requests[1].key()

    def test_one_seed_is_named_after_it_however_it_is_asked(self):
        runner = ColabFoldRunner()
        (alone,) = runner.make_requests(P53, seeds=[2], **ROLES)
        apart = runner.make_requests(P53, seeds=[0, 2], **ROLES)
        assert alone.name == "p53_s2" == apart[1].name
        assert alone.key() == apart[1].key()

    def test_the_request_of_a_group_is_the_one_make_request_builds(self):
        runner = ColabFoldRunner()
        for seeds in ([0, 1, 2], [3]):
            (request,) = runner.make_requests(P53, seeds=seeds, **ROLES)
            direct = runner.make_request(P53, seeds=seeds, **ROLES)
            assert request.key() == direct.key()

    def test_the_command_runs_the_range_of_each_request(self):
        runner = ColabFoldRunner()
        fasta, results = Path("q.fasta"), Path("out")
        (one,) = runner.make_requests(P53, seeds=[5, 6, 7], **ROLES)
        command = runner._command(one, fasta, results)
        assert (after(command, "--random-seed"), after(command, "--num-seeds")) == ("5", "3")
        first, second = runner.make_requests(P53, seeds=[0, 2], **ROLES)
        pairs = [
            (after(c, "--random-seed"), after(c, "--num-seeds"))
            for c in (runner._command(r, fasta, results) for r in (first, second))
        ]
        assert pairs == [("0", "1"), ("2", "1")]

    @pytest.mark.parametrize(
        "seeds, message",
        [([], "at least one seed"), ([1, 1], "distinct"), ([-1, 0], "not be negative")],
    )
    def test_a_list_the_model_cannot_take_is_refused(self, seeds, message):
        with pytest.raises(ValueError, match=message):
            ColabFoldRunner().make_requests(P53, seeds=seeds, **ROLES)

    def test_every_request_it_builds_passes_the_check_of_the_runner(self):
        runner = ColabFoldRunner()
        for seeds in ([0, 1, 2], [0, 2], [3, 1]):
            for request in runner.make_requests(P53, seeds=seeds, **ROLES):
                runner._check_request(request)


class TestColabFoldRefusesSeedsThatAreNotIntegers:
    """``make_request`` used ``int(seed)``, which turned 0.7 into 0 and True into 1."""

    @pytest.mark.parametrize(
        "seeds, shown",
        [
            ([0.7, 1.2], "0.7"),
            ([True, 2], "True"),
            (["1", "2"], "'1'"),
            ([0, 1.5], "1.5"),
            ([1.0], "1.0"),
            ([None], "None"),
        ],
    )
    def test_the_values_the_bug_report_lists_are_refused_and_named(self, seeds, shown):
        with pytest.raises(ValueError) as caught:
            ColabFoldRunner().make_request(P53, seeds=seeds, **ROLES)
        assert str(caught.value) == f"seeds must be integers, got {shown}"

    @pytest.mark.parametrize("seeds", [[0.7, 1.2], [True, 2], ["1", "2"], [0, 1.5]])
    def test_protenix_says_the_same_for_the_same_list(self, seeds):
        with pytest.raises(ValueError) as protenix:
            ProtenixRunner().make_request(P53, seeds=seeds, **ROLES)
        with pytest.raises(ValueError) as colabfold:
            ColabFoldRunner().make_request(P53, seeds=seeds, **ROLES)
        assert str(colabfold.value) == str(protenix.value)

    @pytest.mark.parametrize("seeds", ["12", b"12"])
    def test_a_string_is_not_a_list_of_digits(self, seeds):
        with pytest.raises(ValueError, match="not a string"):
            ColabFoldRunner().make_request(P53, seeds=seeds, **ROLES)

    def test_make_requests_refuses_the_same_lists_before_the_runner_is_asked(self):
        for seeds in ([0.7, 1.2], [True, 2], ["1", "2"]):
            with pytest.raises(ValueError, match="seeds must be integers"):
                ColabFoldRunner().make_requests(P53, seeds=seeds, **ROLES)

    def test_integers_numpy_integers_and_a_tuple_still_work(self):
        import numpy as np

        runner = ColabFoldRunner()
        assert runner.make_request(P53, seeds=[1, 0], **ROLES).seeds == (0, 1)
        assert runner.make_request(P53, seeds=(np.int64(2), np.int32(3)), **ROLES).seeds == (2, 3)
        assert runner.make_request(P53, seeds=np.arange(3), **ROLES).seeds == (0, 1, 2)
        assert runner.make_request(P53, seeds=None, **ROLES).seeds == (0,)

    def test_the_key_of_a_request_with_integer_seeds_is_unchanged(self):
        runner = ColabFoldRunner()
        assert (
            runner.make_request(P53, seeds=[0, 1], num_samples=2, **ROLES).key()
            == "415fb60014dcd6f0ed90e7bd1505126a34409d5ddd98fe205a0d646a86f938d1"
        )


# ---------------------------------------------------------------------------- records, manifest


def rows_of(session, requests):
    """Run the requests, then list them: the manifest itself never starts a model."""
    for request in requests:
        session.entry(request)
    return prediction_manifest(session, requests)


def columns(rows, *names):
    return [tuple(row[name] for name in names) for row in rows]


class TestProtenixSeedsInTheManifest:
    def test_each_seed_is_listed_under_its_value_whatever_the_order_of_the_request(self, store):
        runner = WritingProtenix()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[42, 7, 11], num_samples=2, **ROLES)
        assert len(requests) == 1
        rows = rows_of(session, requests)
        assert columns(rows, "seed", "seed_index", "sample") == [
            (7, 1, 1),
            (7, 1, 2),
            (11, 2, 1),
            (11, 2, 2),
            (42, 3, 1),
            (42, 3, 2),
        ]
        assert runner.runs == ["p53"]

    def test_the_poses_carry_the_seed_the_model_sampled_with(self, store):
        runner = WritingProtenix()
        session = PredictionSession(store, [runner])
        (request,) = runner.make_requests(P53, seeds=[9, 10, 100], num_samples=1, **ROLES)
        poses = session.records(request)
        assert [pose.provenance["seed"] for pose in poses] == [9, 10, 100]
        assert [pose.extras["seed_value"] for pose in poses] == ["9", "10", "100"]
        assert all(pose.provenance["backend"] == "protenix" for pose in poses)

    def test_the_rank_counts_inside_each_seed_by_the_model_score(self, store):
        runner = WritingProtenix()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[1, 2], num_samples=2, **ROLES)
        rows = rows_of(session, requests)
        assert columns(rows, "seed", "sample", "rank") == [
            (1, 1, 2),
            (1, 2, 1),
            (2, 1, 2),
            (2, 2, 1),
        ]

    def test_the_manifest_shares_the_run_of_the_session(self, store):
        runner = WritingProtenix()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[1, 2], **ROLES)
        session.records(requests[0])
        rows_of(session, requests)
        assert runner.runs == ["p53"]

    def test_a_request_that_was_not_run_is_missing_and_the_manifest_starts_nothing(self, store):
        runner = WritingProtenix()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[1, 2], **ROLES)
        rows = prediction_manifest(session, requests)
        assert columns(rows, "status", "seed") == [("missing", None)]
        assert getattr(runner, "runs", []) == []


class TestColabFoldSeedsInTheManifest:
    def test_consecutive_seeds_are_one_run_listing_both_seeds(self, store):
        runner = WritingColabFold()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[0, 1], num_samples=2, **ROLES)
        assert len(requests) == 1
        rows = rows_of(session, requests)
        assert columns(rows, "backend", "seed", "seed_index", "sample") == [
            ("af2", 0, 1, 1),
            ("af2", 0, 1, 2),
            ("af2", 1, 2, 1),
            ("af2", 1, 2, 2),
        ]
        assert runner.runs == ["p53"]
        assert {row["request_name"] for row in rows} == {"p53"}

    def test_a_first_seed_above_zero_is_read_from_the_file_names(self, store):
        runner = WritingColabFold()
        session = PredictionSession(store, [runner])
        (request,) = runner.make_requests(P53, seeds=[5, 6, 7], num_samples=1, **ROLES)
        poses = session.records(request)
        assert [pose.provenance["seed"] for pose in poses] == [5, 6, 7]
        assert [pose.extras["seed_value"] for pose in poses] == ["5", "6", "7"]

    def test_seeds_that_are_not_consecutive_are_one_run_each_and_all_listed(self, store):
        runner = WritingColabFold()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[0, 2], num_samples=2, **ROLES)
        assert names_of(requests) == ["p53_s0", "p53_s2"]
        rows = rows_of(session, requests)
        assert columns(rows, "request_name", "seed", "seed_index", "sample") == [
            ("p53_s0", 0, 1, 1),
            ("p53_s0", 0, 1, 2),
            ("p53_s2", 2, 1, 1),
            ("p53_s2", 2, 1, 2),
        ]
        assert runner.runs == ["p53_s0", "p53_s2"]

    def test_the_rows_of_a_run_carry_its_own_key(self, store):
        runner = WritingColabFold()
        session = PredictionSession(store, [runner])
        requests = runner.make_requests(P53, seeds=[0, 2], num_samples=1, **ROLES)
        rows = rows_of(session, requests)
        assert [row["request_key"] for row in rows] == [r.key() for r in requests]
        assert all(row["status"] == "done" for row in rows)

    def test_the_rank_is_the_position_inside_the_seed_by_the_colabfold_rank(self, store):
        runner = WritingColabFold()
        session = PredictionSession(store, [runner])
        (request,) = runner.make_requests(P53, seeds=[0, 1], num_samples=2, **ROLES)
        rows = rows_of(session, [request])
        assert sorted(columns(rows, "seed", "rank")) == [(0, 1), (0, 2), (1, 1), (1, 2)]


class TestTheRunnersKeepTheContract:
    @pytest.mark.parametrize("runner_class", [ProtenixRunner, ColabFoldRunner])
    def test_make_requests_is_the_one_of_the_base_class(self, runner_class):
        assert "make_requests" not in vars(runner_class)
        assert runner_class.make_requests is PredictionRunner.make_requests
