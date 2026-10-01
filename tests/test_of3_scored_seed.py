"""The pipeline scores the first seed given with ``--openfold-seeds``, not the lowest value.

OpenFold3 writes one ``seed_<value>`` directory per seed, and the adapters count the directories
in the numeric order of the values (``seed`` is that position). ``--openfold-seeds 11 9`` must
score the sample of seed 11, which is the second directory. The model is a stub that writes
synthetic outputs under the seed values 9, 10 and 11 (``tests/predictors/synth_of3.py``).
"""

import math
from pathlib import Path

import pytest

from binding_metrics.cli import batch
from binding_metrics.cli.prediction import scored_seed_index, scored_seed_kwargs
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import openfold
from tests.predictors import synth_of3
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
    complex_from,
)

#: seed index -> (seed value, average pLDDT) of the synthetic outputs
SEEDS = {1: (9, 61.0), 2: (10, 62.0), 3: (11, 63.0)}


def write_three_seeds(directory, name, input_path, leftovers=()):
    """Outputs of the seeds 9, 10 and 11 of ``name``, each with its own pLDDT."""
    for index, (_, plddt) in SEEDS.items():
        synthetic = complex_from(input_path)
        synthetic.scalars["avg_plddt"] = plddt
        synth_of3.write_prediction(Path(directory), name, synthetic, seed_index=index)
    for value in leftovers:  # directories that an earlier run left behind
        (Path(directory) / name / f"seed_{value}").mkdir(parents=True, exist_ok=True)


class TestScoredSeedIndex:
    @pytest.mark.parametrize(
        "seeds, expected",
        [
            (None, 1),
            ([], 1),
            ([5], 1),
            ([3, 9], 1),
            ([9, 3], 2),
            ([9, 3, 5], 3),
            ([11, 9, 10], 3),
            ([9, 9, 3], 2),
            ([2746317213, 42], 2),
        ],
    )
    def test_the_position_of_the_first_given_seed_in_numeric_order(self, seeds, expected):
        assert scored_seed_index(seeds) == expected

    def test_the_directories_that_exist_decide_when_they_are_given(self, tmp_path):
        for value in (5, 42, 50, 60):
            (tmp_path / "q" / f"seed_{value}").mkdir(parents=True)
        # an earlier run left seed_5 and seed_42; the formula alone would say 1
        assert scored_seed_index([50, 60], tmp_path, "q") == 3
        assert scored_seed_index([60, 50], tmp_path, "q") == 4

    def test_a_missing_directory_falls_back_to_the_formula(self, tmp_path):
        assert scored_seed_index([9, 3], tmp_path, "nowhere") == 2
        (tmp_path / "q" / "seed_3").mkdir(parents=True)
        assert scored_seed_index([9, 3], tmp_path, "q") == 2  # seed_9 is not there (yet)

    def test_kwargs_are_empty_without_seeds_so_the_call_is_unchanged(self, tmp_path):
        assert scored_seed_kwargs(None, tmp_path, "q") == {}
        assert scored_seed_kwargs([], tmp_path, "q") == {}
        assert scored_seed_kwargs([9, 3], tmp_path, "q") == {"seed": 2}


def pipeline(tmp_path, **kwargs):
    return run_pipeline(
        EXAMPLE_1YCR,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        peptide_chain=PEPTIDE_CHAIN,
        receptor_chain=RECEPTOR_CHAIN,
        openfold_conda_env=None,
        **kwargs,
    )


@pytest.fixture
def three_seed_model(monkeypatch):
    """The stub model, writing the seeds 9, 10 and 11 (and stale seed 5 and 42 directories)."""
    stub = StubOpenFold(monkeypatch)
    leftovers = []

    def run(**kwargs):
        stub.calls.append({"kind": "scoring", "names": [kwargs["query_name"]], "kwargs": kwargs})
        predictions = Path(kwargs["output_dir"]) / "predictions"
        write_three_seeds(
            predictions, kwargs["query_name"], kwargs["complex_structure_path"], leftovers
        )
        return predictions

    def run_batched(*, samples, output_dir, **kwargs):
        stub.calls.append(
            {"kind": "batched", "names": [s.query_name for s in samples], "kwargs": kwargs}
        )
        predictions = Path(output_dir) / "predictions"
        for sample in samples:
            write_three_seeds(
                predictions, sample.query_name, sample.complex_structure_path, leftovers
            )
        return predictions

    monkeypatch.setattr(openfold, "run_openfold_scoring", run)
    monkeypatch.setattr(openfold, "run_openfold_refolding", run)
    monkeypatch.setattr(openfold, "run_openfold_batched", run_batched)
    stub.leftovers = leftovers
    return stub


class TestLegacyStep:
    def test_without_seeds_the_first_directory_is_scored_as_before(
        self, tmp_path, three_seed_model
    ):
        block = pipeline(tmp_path)["openfold"]
        assert (block["seed"], block["seed_value"]) == (1, 9)
        assert block["avg_plddt"] == pytest.approx(61.0)

    @pytest.mark.parametrize(
        "seeds, seed_index, seed_value, plddt",
        [([9], 1, 9, 61.0), ([10, 9], 2, 10, 62.0), ([11, 9, 10], 3, 11, 63.0)],
    )
    def test_the_first_given_seed_is_scored(
        self, tmp_path, three_seed_model, seeds, seed_index, seed_value, plddt
    ):
        block = pipeline(tmp_path, openfold_seeds=seeds)["openfold"]
        assert (block["seed"], block["seed_value"]) == (seed_index, seed_value)
        assert block["avg_plddt"] == pytest.approx(plddt)

    def test_stale_seed_directories_do_not_shift_the_choice(self, tmp_path, three_seed_model):
        three_seed_model.leftovers.extend([5, 42])  # numeric order: 5, 9, 10, 11, 42
        block = pipeline(tmp_path, openfold_seeds=[10])["openfold"]
        assert (block["seed"], block["seed_value"]) == (3, 10)
        assert block["avg_plddt"] == pytest.approx(62.0)


class TestPredictionStore:
    def test_without_seeds_the_first_directory_is_read(self, tmp_path, three_seed_model):
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        assert (block["seed"], block["seed_value"]) == (1, 9)

    @pytest.mark.parametrize(
        "seeds, seed_index, seed_value", [([10, 9], 2, 10), ([11, 9, 10], 3, 11)]
    )
    def test_the_first_given_seed_is_read(
        self, tmp_path, three_seed_model, seeds, seed_index, seed_value
    ):
        block = pipeline(tmp_path, predictor="of3", openfold_seeds=seeds)["prediction"]
        assert (block["seed"], block["seed_value"]) == (seed_index, seed_value)
        assert block["avg_plddt"] == pytest.approx(SEEDS[seed_index][1])
        assert math.isfinite(block["mean_interface_pae"])  # the other consumers read it too

    def test_an_adopted_output_keeps_its_first_directory(self, tmp_path):
        output = tmp_path / "mine"
        write_three_seeds(output, EXAMPLE_1YCR.stem, EXAMPLE_1YCR)
        block = pipeline(
            tmp_path / "o", predictor="of3", prediction_dir=output, openfold_seeds=[11]
        )["prediction"]
        # --openfold-seeds configures a run made here; an output you made is read as it is
        assert (block["seed"], block["seed_value"]) == (1, 9)


class TestBatch:
    @staticmethod
    def _eligible(monkeypatch):
        monkeypatch.setattr(
            batch,
            "_detect_sample_chains",
            lambda *a, **k: [(0, EXAMPLE_1YCR.stem, EXAMPLE_1YCR, PEPTIDE_CHAIN, RECEPTOR_CHAIN)],
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))

    def test_the_batched_openfold_step_scores_the_first_given_seed(
        self, tmp_path, monkeypatch, three_seed_model
    ):
        self._eligible(monkeypatch)
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
            openfold_seeds=[11, 9],
        )
        assert (rows[0]["openfold_seed"], rows[0]["openfold_seed_value"]) == (3, 11)
        assert rows[0]["openfold_avg_plddt"] == pytest.approx(63.0)

    def test_the_batched_prediction_step_scores_the_first_given_seed(
        self, tmp_path, monkeypatch, three_seed_model
    ):
        self._eligible(monkeypatch)
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_prediction(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            predictor="of3",
            peptide_chain=None,
            receptor_chain=None,
            openfold_seeds=[10, 9],
        )
        assert (rows[0]["prediction_seed"], rows[0]["prediction_seed_value"]) == (2, 10)
        assert rows[0]["prediction_avg_plddt"] == pytest.approx(62.0)

    def test_without_seeds_the_batched_steps_score_the_first_directory(
        self, tmp_path, monkeypatch, three_seed_model
    ):
        self._eligible(monkeypatch)
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
        )
        assert (rows[0]["openfold_seed"], rows[0]["openfold_seed_value"]) == (1, 9)
