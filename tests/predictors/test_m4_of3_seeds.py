"""OpenFold3 seeds: one request carries a list of seeds, and the session lists each seed.

OpenFold3 samples every seed of a request in one process and writes a ``seed_<value>``
directory for each. The tests use the real runner for the requests and a subclass that writes the
synthetic OpenFold3 output (``synth_of3``) in place of the model, with the directories named
after the seeds of the request.
"""

from pathlib import Path

import pytest

from binding_metrics.predictors import prediction_manifest
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import PredictionStore
from tests.predictors import synth_of3
from tests.predictors.test_m4_provenance import truth_for

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CHAINS = {"binder_chain": "B", "receptor_chain": "A"}
SEEDS = [42, 7, 11]


class WritesSeedDirectories(OpenFold3Runner):
    """The OpenFold3 runner with the model replaced by the synthetic output of its layout."""

    def __init__(self):
        super().__init__()
        self.runs = []

    def is_available(self):
        return True

    def version(self):
        return "0.5.0"

    def prepare(self, request, work_dir):
        return Path(work_dir)

    def run(self, request, work_dir):
        self.runs.append(request.name)
        # the directory of the seed at position `position` of the request is `seed_<value>`
        original = synth_of3.seed_value
        synth_of3.seed_value = lambda position: request.seeds[position - 1]
        try:
            for position in range(1, len(request.seeds) + 1):
                for sample in range(1, request.num_samples + 1):
                    # the ranking score tells the seeds apart: 0.1 per seed value, plus the sample
                    score = round(0.01 * request.seeds[position - 1] + 0.001 * sample, 4)
                    synth_of3.write_prediction(
                        work_dir,
                        request.name,
                        truth_for(position, sample, score),
                        seed_index=position,
                        sample=sample,
                    )
        finally:
            synth_of3.seed_value = original
        return Path(work_dir)


@pytest.fixture
def runner():
    return WritesSeedDirectories()


@pytest.fixture
def session(tmp_path, runner):
    return PredictionSession(PredictionStore(tmp_path / "store"), [runner])


@pytest.fixture
def one_request(runner):
    (request,) = runner.make_requests(
        P53, seeds=SEEDS, name="p53", mode="refold", num_samples=2, **CHAINS
    )
    return request


class TestTheRequest:
    def test_the_runner_has_a_seed_dimension(self):
        assert OpenFold3Runner.seed_dimension is True
        assert OpenFold3Runner()._group_seeds(SEEDS) == [(42, 7, 11)]

    def test_a_list_of_seeds_is_one_request_that_carries_them_in_the_order_given(self, runner):
        requests = runner.make_requests(P53, seeds=SEEDS, name="p53", mode="refold", **CHAINS)
        assert len(requests) == 1
        assert requests[0].seeds == (42, 7, 11)
        assert requests[0].name == "p53"  # no `_s<seed>` suffix: the request holds every seed

    def test_one_seed_is_one_request_with_the_name_it_was_given(self, runner):
        (request,) = runner.make_requests(P53, seeds=[7], name="p53", mode="refold", **CHAINS)
        assert (request.seeds, request.name) == ((7,), "p53")

    def test_the_request_has_the_key_that_make_request_gives(self, runner):
        (request,) = runner.make_requests(P53, seeds=SEEDS, name="p53", mode="refold", **CHAINS)
        direct = runner.make_request(P53, seeds=SEEDS, name="other", mode="refold", **CHAINS)
        assert request.key() == direct.key()

    def test_the_seed_dimension_adds_nothing_to_the_key_of_a_request_without_seeds(self, runner):
        # the default is the single seed 42, as before the attribute was set
        (via_make_requests,) = runner.make_requests(
            P53, seeds=[42], name="p53", mode="refold", **CHAINS
        )
        default = runner.make_request(P53, name="p53", mode="refold", **CHAINS)
        assert via_make_requests.key() == default.key()

    def test_other_seed_lists_are_other_requests(self, runner):
        keys = {
            runner.make_requests(P53, seeds=seeds, name="p53", mode="refold", **CHAINS)[0].key()
            for seeds in ([42, 7, 11], [7, 11, 42], [42, 7], [42])
        }
        assert len(keys) == 4  # the seeds are in the key, in the order given

    @pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [1.5]])
    def test_seeds_that_cannot_be_run_are_refused(self, runner, seeds):
        with pytest.raises(ValueError, match="seed"):
            runner.make_requests(P53, seeds=seeds, name="p53", mode="refold", **CHAINS)

    def test_seeds_cannot_be_combined_with_a_generated_number_of_seeds(self, runner):
        with pytest.raises(ValueError, match="num_model_seeds"):
            runner.make_requests(
                P53, seeds=SEEDS, name="p53", mode="refold", num_model_seeds=2, **CHAINS
            )


class TestTheSessionListsEachSeed:
    def test_the_model_is_started_once_for_the_three_seeds(self, session, runner, one_request):
        poses = session.records(one_request)
        assert runner.runs == ["p53"]
        assert len(poses) == 6

    def test_the_poses_name_the_seed_value_of_the_directory(self, session, one_request):
        poses = session.records(one_request)
        # the directories are seed_7, seed_11 and seed_42: numeric order, not the order requested
        assert [(p.seed_index, p.sample) for p in poses] == [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
        ]
        assert [p.extras["seed_value"] for p in poses] == ["7", "7", "11", "11", "42", "42"]
        assert [p.provenance["seed"] for p in poses] == [7, 7, 11, 11, 42, 42]
        assert {p.provenance["seed_index"] for p in poses} == {1, 2, 3}

    def test_each_pose_comes_from_the_files_of_its_own_seed(self, session, one_request):
        poses = session.records(one_request)
        assert [p.ranking_score for p in poses] == pytest.approx(
            [0.071, 0.072, 0.111, 0.112, 0.421, 0.422]
        )
        assert [p.structure_path.parent.name for p in poses] == [
            "seed_7",
            "seed_7",
            "seed_11",
            "seed_11",
            "seed_42",
            "seed_42",
        ]


class TestTheManifest:
    @pytest.fixture
    def stored(self, session, one_request):
        session.entry(one_request)  # the manifest reads what a run stored
        return one_request

    def test_it_lists_the_three_seeds_with_their_values(self, session, stored):
        rows = prediction_manifest(session, [stored])
        assert len(rows) == 6
        assert [(row["seed"], row["seed_index"], row["sample"]) for row in rows] == [
            (7, 1, 1),
            (7, 1, 2),
            (11, 2, 1),
            (11, 2, 2),
            (42, 3, 1),
            (42, 3, 2),
        ]
        assert {row["seed"] for row in rows} == set(SEEDS)

    def test_the_rows_carry_the_request_and_the_scores_of_their_seed(self, session, stored):
        rows = prediction_manifest(session, [stored])
        assert {row["backend"] for row in rows} == {"of3"}
        assert {row["request_key"] for row in rows} == {stored.key()}
        assert {row["request_name"] for row in rows} == {"p53"}
        assert {row["status"] for row in rows} == {"done"}
        assert [row["ranking_score"] for row in rows] == pytest.approx(
            [0.071, 0.072, 0.111, 0.112, 0.421, 0.422]
        )
        assert all(row["rank"] is None for row in rows)  # OpenFold3 samples are not ranked
        assert all(Path(row["structure_path"]).is_file() for row in rows)

    def test_the_manifest_starts_no_model(self, session, runner, stored):
        runs_before = list(runner.runs)
        prediction_manifest(session, [stored])
        assert runner.runs == runs_before == ["p53"]

    def test_a_request_that_was_not_run_is_one_row_without_a_seed(
        self, session, runner, one_request
    ):
        (row,) = prediction_manifest(session, [one_request])
        assert (row["status"], row["seed"]) == ("missing", None)
        assert runner.runs == []
