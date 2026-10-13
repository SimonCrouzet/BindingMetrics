"""``PredictionRunner.make_requests``: one request for all seeds, or one request per seed.

Stub runners stand for the two kinds of model (with and without a seed dimension) and for the
ColabFold rule (consecutive seeds in one request). The four real runners are called with their
own ``make_request``, with the model version patched, so that the test needs no installation.
"""

from pathlib import Path

import numpy as np
import pytest

from binding_metrics.predictors.af2_runner import ColabFoldRunner
from binding_metrics.predictors.boltz2_runner import Boltz2Runner
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.protenix_runner import ProtenixRunner
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import KEY_FORMAT, PredictionRequest

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"


class OneSeedRunner(PredictionRunner):
    """A model that runs one seed at a time (the default of the base class)."""

    name = "oneseed"

    def prepare(self, request, work_dir):
        return Path(work_dir)

    def run(self, request, work_dir):
        return Path(work_dir)

    def make_request(self, input_path, *, name, seeds=None, num_samples=5, **extra):
        return PredictionRequest(
            self.name,
            name,
            mode="predict",
            input_path=input_path,
            seeds=(42,) if seeds is None else seeds,
            num_samples=num_samples,
            options=extra,
        )


class SeedListRunner(OneSeedRunner):
    """A model that takes any list of seeds in one run."""

    name = "seedlist"
    seed_dimension = True


class ConsecutiveRunner(OneSeedRunner):
    """The ColabFold rule: consecutive seeds share a request, any other seed has its own."""

    name = "consecutive"

    def _group_seeds(self, seeds):
        if list(seeds) == list(range(seeds[0], seeds[0] + len(seeds))):
            return [tuple(seeds)]
        return [(seed,) for seed in seeds]


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "complex.cif"
    path.write_text("data_x\n", encoding="utf-8")
    return path


def seeds_of(requests):
    return [tuple(request.seeds) for request in requests]


def names_of(requests):
    return [request.name for request in requests]


# ---------------------------------------------------------------------------- the two kinds


class TestARunnerWithoutASeedDimension:
    def test_the_base_class_says_one_request_per_seed(self):
        assert PredictionRunner.seed_dimension is False
        assert OneSeedRunner.seed_dimension is False

    def test_each_seed_gets_its_own_request_in_the_order_given(self, input_file):
        requests = OneSeedRunner().make_requests(input_file, seeds=[44, 42, 43], name="d")
        assert seeds_of(requests) == [(44,), (42,), (43,)]

    def test_the_names_carry_the_seed_so_output_folders_differ(self, input_file):
        requests = OneSeedRunner().make_requests(input_file, seeds=[42, 43, 44], name="d")
        assert names_of(requests) == ["d_s42", "d_s43", "d_s44"]
        assert len(set(names_of(requests))) == 3

    def test_a_single_seed_is_named_after_its_seed_too(self, input_file):
        """A seed has the same name whether it is asked alone or with others."""
        (alone,) = OneSeedRunner().make_requests(input_file, seeds=[43], name="d")
        together = OneSeedRunner().make_requests(input_file, seeds=[42, 43], name="d")
        assert alone.name == "d_s43" == together[1].name
        assert alone == together[1]  # equal key and equal name

    def test_the_other_options_reach_every_request(self, input_file):
        requests = OneSeedRunner().make_requests(
            input_file, seeds=[1, 2], name="d", num_samples=3, preset="fast"
        )
        assert [r.num_samples for r in requests] == [3, 3]
        assert [r.options for r in requests] == [{"preset": "fast"}] * 2

    def test_the_requests_differ_in_their_key_by_the_seed_only(self, input_file):
        runner = OneSeedRunner()
        requests = runner.make_requests(input_file, seeds=[42, 43], name="d")
        assert requests[0].key() != requests[1].key()
        for request, seed in zip(requests, (42, 43)):
            assert (
                request.key() == runner.make_request(input_file, name="other", seeds=[seed]).key()
            )


class TestARunnerWithASeedDimension:
    def test_one_request_carries_all_the_seeds_in_the_order_given(self, input_file):
        (request,) = SeedListRunner().make_requests(input_file, seeds=[42, 7, 11], name="d")
        assert tuple(request.seeds) == (42, 7, 11)

    def test_the_name_is_kept(self, input_file):
        (request,) = SeedListRunner().make_requests(input_file, seeds=[42, 7, 11], name="d")
        assert request.name == "d"

    def test_one_seed_is_one_request_with_the_name_kept(self, input_file):
        (request,) = SeedListRunner().make_requests(input_file, seeds=[9], name="d")
        assert request.name == "d" and tuple(request.seeds) == (9,)

    def test_the_request_is_the_one_make_request_builds(self, input_file):
        runner = SeedListRunner()
        (request,) = runner.make_requests(input_file, seeds=[1, 2, 3], name="d", num_samples=2)
        direct = runner.make_request(input_file, name="d", seeds=[1, 2, 3], num_samples=2)
        assert request == direct and request.key() == direct.key()


class TestTheGroupingRule:
    """ColabFold overrides ``_group_seeds``; the base class only supplies the hook."""

    def test_consecutive_seeds_share_one_request_named_by_the_job(self, input_file):
        (request,) = ConsecutiveRunner().make_requests(input_file, seeds=[0, 1, 2], name="d")
        assert tuple(request.seeds) == (0, 1, 2) and request.name == "d"

    def test_seeds_that_are_not_consecutive_get_a_request_each(self, input_file):
        requests = ConsecutiveRunner().make_requests(input_file, seeds=[0, 2], name="d")
        assert seeds_of(requests) == [(0,), (2,)]
        assert names_of(requests) == ["d_s0", "d_s2"]

    def test_consecutive_from_a_first_value_other_than_zero(self, input_file):
        (request,) = ConsecutiveRunner().make_requests(input_file, seeds=[5, 6, 7], name="d")
        assert tuple(request.seeds) == (5, 6, 7)

    def test_one_seed_is_a_group_of_one_and_is_named_after_it(self, input_file):
        (request,) = ConsecutiveRunner().make_requests(input_file, seeds=[3], name="d")
        assert request.name == "d_s3"

    def test_the_default_groups(self):
        assert OneSeedRunner()._group_seeds([3, 1]) == [(3,), (1,)]
        assert SeedListRunner()._group_seeds([3, 1]) == [(3, 1)]

    def test_a_grouping_that_loses_a_seed_is_refused(self, input_file):
        class Lossy(OneSeedRunner):
            def _group_seeds(self, seeds):
                return [tuple(seeds[:-1])]

        with pytest.raises(ValueError, match="does not cover each seed once"):
            Lossy().make_requests(input_file, seeds=[1, 2], name="d")

    def test_a_grouping_that_reorders_the_seeds_is_refused(self, input_file):
        class Reordering(OneSeedRunner):
            def _group_seeds(self, seeds):
                return [(seed,) for seed in reversed(seeds)]

        with pytest.raises(ValueError, match="does not cover each seed once"):
            Reordering().make_requests(input_file, seeds=[1, 2], name="d")


# ---------------------------------------------------------------------------- bad input


class TestTheSeeds:
    def test_no_seed_raises(self, input_file):
        with pytest.raises(ValueError, match="at least one seed"):
            OneSeedRunner().make_requests(input_file, seeds=[], name="d")

    def test_a_repeated_seed_raises_and_names_it(self, input_file):
        with pytest.raises(ValueError, match=r"distinct.*\[42\]"):
            OneSeedRunner().make_requests(input_file, seeds=[42, 43, 42], name="d")

    def test_a_negative_seed_raises_and_names_it(self, input_file):
        with pytest.raises(ValueError, match=r"not be negative.*\[-1\]"):
            OneSeedRunner().make_requests(input_file, seeds=[3, -1], name="d")

    @pytest.mark.parametrize("runner_class", [OneSeedRunner, SeedListRunner])
    def test_the_checks_do_not_depend_on_the_dimension(self, input_file, runner_class):
        for seeds in ([], [1, 1], [-5]):
            with pytest.raises(ValueError):
                runner_class().make_requests(input_file, seeds=seeds, name="d")

    @pytest.mark.parametrize("seeds", ["42", b"42", {42, 43}, {42: 1}, 42, None])
    def test_something_that_is_not_a_sequence_raises_a_type_error(self, input_file, seeds):
        with pytest.raises(TypeError, match="ordered sequence of integers"):
            OneSeedRunner().make_requests(input_file, seeds=seeds, name="d")

    @pytest.mark.parametrize("bad", [1.5, "7", True, None])
    def test_a_value_that_is_not_an_integer_raises(self, input_file, bad):
        with pytest.raises(ValueError, match="must be integers"):
            OneSeedRunner().make_requests(input_file, seeds=[1, bad], name="d")

    def test_numpy_integers_and_a_tuple_are_accepted(self, input_file):
        requests = OneSeedRunner().make_requests(input_file, seeds=(np.int64(4), 5), name="d")
        assert seeds_of(requests) == [(4,), (5,)]
        assert all(type(seed) is int for request in requests for seed in request.seeds)

    def test_nothing_is_built_when_a_seed_is_wrong(self, input_file):
        built = []

        class Counting(OneSeedRunner):
            def make_request(self, input_path, **kwargs):
                built.append(kwargs)
                return super().make_request(input_path, **kwargs)

        with pytest.raises(ValueError):
            Counting().make_requests(input_file, seeds=[1, 2, 2], name="d")
        assert built == []

    def test_a_missing_name_is_the_error_of_make_request(self, input_file):
        with pytest.raises(TypeError, match="name"):
            OneSeedRunner().make_requests(input_file, seeds=[1])

    def test_a_runner_without_make_request_says_so(self, input_file):
        class Bare(PredictionRunner):
            name = "bare"

            def prepare(self, request, work_dir):
                return Path(work_dir)

            def run(self, request, work_dir):
                return Path(work_dir)

        with pytest.raises(NotImplementedError, match="bare runner defines no make_request"):
            Bare().make_requests(input_file, seeds=[1], name="d")


# ---------------------------------------------------------------------------- the key


class TestTheKeyIsUnchanged:
    def test_a_request_built_the_old_way_keeps_its_key(self):
        request = PredictionRequest(
            "boltz2",
            "pin",
            mode="predict",
            sequences={"A": "ACDEFGHIK", "B": "LMNPQ"},
            seeds=(42,),
            num_samples=5,
            model_version="2.2.1",
            options={"use_msa_server": True, "binder_cyclic": "auto", "extra_args": []},
        )
        assert KEY_FORMAT == 1
        assert request.key() == "1a3ae76651f9ac9c339e40dabc321b08a35eee311642a098e4daf5e57590d33a"

    def test_make_requests_adds_nothing_to_the_options_of_a_request(self, input_file):
        runner = OneSeedRunner()
        (request,) = runner.make_requests(input_file, seeds=[42], name="d", preset="fast")
        assert request.options == {"preset": "fast"}


# ---------------------------------------------------------------------------- the real runners

#: Runner class and the arguments its make_request needs for the 1YCR complex.
REAL_RUNNERS = [
    (Boltz2Runner, {"mode": "score"}),
    (OpenFold3Runner, {"mode": "score"}),
    (ProtenixRunner, {"mode": "predict"}),
    (ColabFoldRunner, {"mode": "predict"}),
]


@pytest.fixture
def versions(monkeypatch):
    """A version for every runner, whatever is installed."""
    for runner_class, _ in REAL_RUNNERS:
        monkeypatch.setattr(runner_class, "version", lambda self: "1.0")


def real_requests(runner_class, extra, seeds):
    return runner_class().make_requests(
        P53, seeds=seeds, name="p53", binder_chain="B", receptor_chain="A", **extra
    )


class TestTheFourRunners:
    @pytest.mark.parametrize("runner_class, extra", REAL_RUNNERS)
    def test_the_requests_follow_the_seed_dimension_of_the_runner(
        self, versions, runner_class, extra
    ):
        seeds = [0, 1, 2]
        requests = real_requests(runner_class, extra, seeds)
        groups = runner_class()._group_seeds(seeds)
        assert seeds_of(requests) == [tuple(group) for group in groups]
        assert all(request.model == runner_class.name for request in requests)

    @pytest.mark.parametrize("runner_class, extra", REAL_RUNNERS)
    def test_a_runner_without_the_dimension_gives_one_request_per_seed(
        self, versions, runner_class, extra
    ):
        """What is true of every runner that has not set ``seed_dimension`` or a grouping."""
        if runner_class.seed_dimension or "_group_seeds" in vars(runner_class):
            pytest.skip(f"{runner_class.__name__} declares its own grouping")
        requests = real_requests(runner_class, extra, [0, 1, 2])
        assert seeds_of(requests) == [(0,), (1,), (2,)]
        assert names_of(requests) == ["p53_s0", "p53_s1", "p53_s2"]

    @pytest.mark.parametrize("runner_class, extra", REAL_RUNNERS)
    def test_the_request_of_a_seed_has_the_key_make_request_gives(
        self, versions, runner_class, extra
    ):
        runner = runner_class()
        for request in real_requests(runner_class, extra, [0, 1]):
            direct = runner.make_request(
                P53,
                name="p53",
                binder_chain="B",
                receptor_chain="A",
                seeds=list(request.seeds),
                **extra,
            )
            assert request.key() == direct.key()

    def test_boltz2_takes_one_seed_per_request_so_a_seed_list_is_a_list_of_requests(self, versions):
        requests = real_requests(Boltz2Runner, {"mode": "score"}, [42, 43, 44])
        assert seeds_of(requests) == [(42,), (43,), (44,)]
        assert names_of(requests) == ["p53_s42", "p53_s43", "p53_s44"]
        assert len({request.key() for request in requests}) == 3
