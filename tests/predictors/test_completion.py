"""PredictionParser.complete: the step that needs the structure, and who calls it.

The Protenix fixture (a modified residue and an ion, so the matrices have 9 tokens for 6
residues) shows the difference: through ``load`` alone the interface block is refused, through
``compute_prediction_metrics`` and through ``PredictionSession.record`` (the pipeline and
``binding-metrics-prediction``) it is cut from the token layout.
"""

import sys
from unittest import mock

import numpy as np
import pytest

from binding_metrics.metrics import openfold
from binding_metrics.metrics.prediction import compute_prediction_metrics, summarize_prediction
from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.protenix import ProtenixParser
from binding_metrics.predictors.record import PredictionRecord
from binding_metrics.predictors.registry import PARSERS
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import PredictionRequest, PredictionStore
from tests.predictors import contract, synth_protenix, synth_stub
from tests.predictors.test_protenix import (
    NAME,
    _complex_with_a_modified_residue_and_an_ion,
    _load,
    _write,
    registered,  # noqa: F401 - a fixture
)

CHAINS = {"binder_chain": "B", "receptor_chain": "A"}


@pytest.fixture
def truth(tmp_path):
    return _write(tmp_path, _complex_with_a_modified_residue_and_an_ion())


def _block(truth):
    """The binder rows by receptor columns of the 9-token matrices."""
    return truth.pae[6:8, 0:6]


class TestTheDefault:
    def test_the_base_class_returns_the_record_unchanged(self):
        class Plain(synth_stub.StubParser):
            pass

        record = PredictionRecord("stub", "q", avg_plddt=80.0)
        assert PredictionParser.complete(Plain(), record) is record
        assert record.reasons == [] and record.tokens is None

    def test_load_does_not_complete(self, truth, tmp_path, registered):  # noqa: F811
        record = _load(tmp_path)
        assert record.tokens is None


class TestProtenixCompletion:
    def test_load_alone_leaves_the_interface_refused(self, truth, tmp_path):
        record = _load(tmp_path)
        with pytest.warns(UserWarning, match="interface PAE skipped"):
            result = summarize_prediction(record, **CHAINS)
        assert np.isnan(result["mean_interface_pae"])

    def test_complete_attaches_the_layout_and_returns_the_same_record(self, truth, tmp_path):
        record = _load(tmp_path)
        assert ProtenixParser().complete(record) is record
        assert len(record.tokens) == 9
        assert record.tokens.token_ranges() == {"A": (0, 6), "B": (6, 8), "C": (8, 9)}
        assert record.reasons == [] or all("token layout" not in r for r in record.reasons)
        record.validate(check_structure=True)

    def test_complete_is_idempotent(self, truth, tmp_path):
        record = _load(tmp_path)
        parser = ProtenixParser()
        parser.complete(record)
        tokens, reasons = record.tokens, list(record.reasons)
        parser.complete(record)
        assert record.tokens is tokens and record.reasons == reasons

    def test_compute_prediction_metrics_cuts_the_interface_block_from_the_layout(
        self,
        truth,
        tmp_path,
        registered,  # noqa: F811
    ):
        result = compute_prediction_metrics(
            tmp_path, "protenix", NAME, include_matrices=True, **CHAINS
        )
        assert "reason" not in result
        np.testing.assert_allclose(result["pae_interface"], _block(truth))
        assert result["mean_interface_pae"] == pytest.approx(
            (truth.pae[6:8, 0:6].mean() + truth.pae[0:6, 6:8].mean()) / 2
        )
        assert np.isfinite(result["mean_interface_pde"])

    def test_the_chains_of_the_user_reach_the_layout(self, truth, tmp_path, registered):  # noqa: F811
        result = compute_prediction_metrics(
            tmp_path,
            "protenix",
            NAME,
            binder_chain="P",
            receptor_chain="R",
            chain_map={"A": "R", "B": "P"},
            include_matrices=True,
        )
        assert "reason" not in result
        np.testing.assert_allclose(result["pae_interface"], _block(truth))

    def test_a_record_without_the_full_data_file_is_returned_as_it_is(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_protenix, "WRITE_FULL_DATA", False)
        _write(tmp_path, _complex_with_a_modified_residue_and_an_ion())
        record = _load(tmp_path)
        reasons = list(record.reasons)
        ProtenixParser().complete(record)
        assert record.tokens is None and record.reasons == reasons

    def test_a_record_without_a_structure_is_returned_as_it_is(self, truth, tmp_path):
        record = _load(tmp_path)
        record.structure_path = None
        assert ProtenixParser().complete(record).tokens is None

    def test_files_of_two_samples_give_a_reason_and_no_layout(self, truth, tmp_path):
        record = _load(tmp_path)
        record.extras["atom_to_token_idx"] = record.extras["atom_to_token_idx"][:-1]
        ProtenixParser().complete(record)
        assert record.tokens is None
        assert record.reasons[-1].startswith("token layout not built")
        assert "not from the same sample" in record.reasons[-1]
        parser_reasons = list(record.reasons)
        ProtenixParser().complete(record)  # no second sentence
        assert record.reasons == parser_reasons

    def test_a_missing_biotite_leaves_the_record_alone(self, truth, tmp_path):
        record = _load(tmp_path)
        blocked = {
            name: None
            for name in (
                "biotite",
                "biotite.structure",
                "biotite.structure.io",
                "biotite.structure.io.pdb",
                "biotite.structure.io.pdbx",
            )
        }
        with mock.patch.dict(sys.modules, blocked):
            ProtenixParser().complete(record)
        assert record.tokens is None and record.reasons == []

    def test_a_record_of_another_model_is_returned_as_it_is(self, truth, tmp_path):
        record = _load(tmp_path)
        record.model = "of3"
        assert ProtenixParser().complete(record).tokens is None


class TestThePipelinePath:
    """``PredictionSession.record`` is the one place the pipeline and the CLI read from."""

    @pytest.fixture
    def session(self, tmp_path):
        store = PredictionStore(tmp_path / "store")
        return PredictionSession(store, {}, {"protenix": ProtenixParser()})

    def _request(self, session, tmp_path):
        request = PredictionRequest("protenix", NAME, sequences={"A": "AAA"})
        session.store.adopt(request, tmp_path)
        return request

    def test_the_session_hands_out_completed_records_parsed_once(
        self, truth, tmp_path, session, monkeypatch
    ):
        calls = []
        original = ProtenixParser.complete

        def _spy(self, record):
            calls.append(record)
            return original(self, record)

        monkeypatch.setattr(ProtenixParser, "complete", _spy)
        request = self._request(session, tmp_path)
        first = session.record(request)
        assert session.record(request) is first
        assert len(calls) == 1
        assert len(first.tokens) == 9

    def test_the_summary_of_a_session_record_has_the_interface_block(
        self, truth, tmp_path, session
    ):
        record = session.record(self._request(session, tmp_path))
        result = summarize_prediction(record, include_matrices=True, **CHAINS)
        assert "reason" not in result
        np.testing.assert_allclose(result["pae_interface"], _block(truth))

    def test_the_command_line_step_reports_the_interface_values(self, truth, tmp_path, session):
        from binding_metrics.cli.prediction import _analyse

        request = self._request(session, tmp_path)
        model_file = next(tmp_path.rglob("*_sample_0.cif"))
        block, _ = _analyse(
            session,
            request,
            input_path=model_file,
            chain_map=None,
            reference_path=None,
            adopted=True,
            **CHAINS,
        )
        assert np.isfinite(block["mean_interface_pae"])
        assert "PAE matrix has" not in block.get("reason", "")
        assert block["model"] == "protenix"


class TestOpenFold3Path:
    def test_compute_openfold_metrics_still_matches(self, tmp_path):
        from tests.predictors import synth, synth_of3

        synth_of3.write_prediction(tmp_path, contract.NAME, synth.synthetic_complex())
        legacy = openfold.compute_openfold_metrics(tmp_path, contract.NAME, **CHAINS)
        assert legacy["mean_interface_pae"] == pytest.approx(3.4375)
        assert "model" not in legacy

    def test_the_registered_adapters_of_the_default_kind_return_records_unchanged(self):
        # every adapter that does not override complete returns its record as it is
        for spec in PARSERS.values():
            cls = spec.load()
            if cls.complete is PredictionParser.complete:
                record = PredictionRecord(cls.name, "q")
                assert cls().complete(record) is record
