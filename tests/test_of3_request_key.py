"""The version of the query builders is part of the OpenFold3 request key.

A prediction that the store kept was made from the query that the builders of that day wrote. When
the builders change in a way that can change a prediction (a repaired template CIF, another rule
for the cyclic flag, a dummy MSA), ``QUERY_BUILDER_VERSION`` is raised, the key of every score and
refold request changes once, and the store makes the prediction again instead of reusing the old
one. A mode ``predict`` request names a query file of the caller, which no builder writes: its key
does not depend on the version.
"""

from pathlib import Path

import pytest

from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import _openfold_run, openfold
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from tests.test_feat_c_support import StubOpenFold

P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


@pytest.fixture(autouse=True)
def _version(monkeypatch):
    monkeypatch.setattr(OpenFold3Runner, "version", lambda runner: "0.5.0")


def _request(mode="score", **kwargs):
    runner = OpenFold3Runner()
    if mode == "predict":
        return runner.make_request(P53.parent / "q.json", name="q", mode="predict", **kwargs)
    return runner.make_request(
        P53, name="q", binder_chain="B", receptor_chain="A", mode=mode, **kwargs
    )


class TestTheVersionIsInTheKey:
    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_options_say_which_builders_wrote_the_query(self, mode):
        assert (
            _request(mode).options["query_builder_version"] == _openfold_run.QUERY_BUILDER_VERSION
        )

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_another_version_is_another_key(self, mode, monkeypatch):
        before = _request(mode).key()
        monkeypatch.setattr(
            _openfold_run, "QUERY_BUILDER_VERSION", _openfold_run.QUERY_BUILDER_VERSION + 1
        )
        assert _request(mode).key() != before

    def test_the_same_version_gives_the_same_key(self):
        assert _request().key() == _request().key()

    def test_a_query_file_of_the_caller_does_not_depend_on_the_builders(
        self, tmp_path, monkeypatch
    ):
        query = tmp_path / "q.json"
        query.write_text("{}", encoding="utf-8")
        runner = OpenFold3Runner()
        request = runner.make_request(query, name="q", mode="predict")
        assert request.options["query_builder_version"] is None
        before = request.key()
        monkeypatch.setattr(
            _openfold_run, "QUERY_BUILDER_VERSION", _openfold_run.QUERY_BUILDER_VERSION + 1
        )
        assert runner.make_request(query, name="q", mode="predict").key() == before

    def test_the_version_is_an_integer_and_is_importable_from_the_openfold_module(self):
        assert isinstance(openfold.QUERY_BUILDER_VERSION, int)
        assert openfold.QUERY_BUILDER_VERSION >= 3  # 3: the builders of this repair


class TestAStoredPredictionOfAnOlderBuilderIsNotReused:
    @staticmethod
    def _pipeline(tmp_path, store):
        return run_pipeline(
            P53,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain="B",
            receptor_chain="A",
            predictor="of3",
            prediction_cache=store,
            openfold_conda_env=None,
        )["prediction"]

    def test_the_model_runs_again_after_a_bump_and_not_without_one(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        store = tmp_path / "store"
        first = self._pipeline(tmp_path / "a", store)
        assert stub.starts == 1
        again = self._pipeline(tmp_path / "b", store)
        assert stub.starts == 1 and again["cache"]["request_key"] == first["cache"]["request_key"]
        monkeypatch.setattr(
            _openfold_run, "QUERY_BUILDER_VERSION", _openfold_run.QUERY_BUILDER_VERSION + 1
        )
        newer = self._pipeline(tmp_path / "c", store)
        assert stub.starts == 2
        assert newer["cache"]["request_key"] != first["cache"]["request_key"]
