"""Two cleanups of ``cli/prediction.py`` after the store and the EvoBind score gained public API.

An adopted request is keyed by its name in the store itself, so the CLI no longer puts the name
into the request options; and the primary EvoBind score of a record comes from
``compute_evobind_score_from_record`` and not from a private function of the metric module.
"""

from __future__ import annotations

import inspect
import math

import pytest

from binding_metrics.cli import prediction
from binding_metrics.cli.prediction import (
    make_request,
    make_session,
    make_store,
    run_prediction_step,
)
from binding_metrics.metrics.evobind import compute_evobind_score_from_record
from binding_metrics.predictors import get_parser
from tests.test_feat_c_support import EXAMPLE_1YCR, write_of3_output


class TestAdoptedRequestsAreKeyedByName:
    def test_the_request_carries_no_workaround_option(self):
        request = make_request(
            "of3", "a", EXAMPLE_1YCR, binder_chain="B", receptor_chain="A", adopt=True
        )
        assert "adopted_name" not in request.options
        assert dict(request.options) == {}

    def test_two_names_over_one_input_file_have_two_keys(self):
        first = make_request(
            "of3", "a", EXAMPLE_1YCR, binder_chain="B", receptor_chain="A", adopt=True
        )
        second = make_request(
            "of3", "b", EXAMPLE_1YCR, binder_chain="B", receptor_chain="A", adopt=True
        )
        assert first.for_adoption().key() != second.for_adoption().key()

    def test_each_sample_reads_its_own_record_through_the_step_helper(self, tmp_path):
        outputs = tmp_path / "outputs"
        write_of3_output(outputs, "a", EXAMPLE_1YCR, plddt_low=50.0, plddt_high=60.0)
        write_of3_output(outputs, "b", EXAMPLE_1YCR, plddt_low=70.0, plddt_high=80.0)
        store = make_store(tmp_path / "store")
        blocks = {}
        for name in ("a", "b"):  # the same input file, two sample names
            request = make_request(
                "of3", name, EXAMPLE_1YCR, binder_chain="B", receptor_chain="A", adopt=True
            )
            blocks[name], _ = run_prediction_step(
                make_session(store, None),
                request,
                input_path=EXAMPLE_1YCR,
                binder_chain="B",
                receptor_chain="A",
                prediction_dir=outputs,
            )
        assert blocks["a"]["query_name"] == "a" and blocks["b"]["query_name"] == "b"
        assert blocks["a"]["avg_plddt"] == pytest.approx(55.0, abs=1.0)
        assert blocks["b"]["avg_plddt"] == pytest.approx(75.0, abs=1.0)
        assert blocks["a"]["cache"]["request_key"] != blocks["b"]["cache"]["request_key"]


class TestThePublicEvobindScore:
    def _record(self, tmp_path):
        outputs = tmp_path / "outputs"
        write_of3_output(outputs, "s", EXAMPLE_1YCR)
        return get_parser("of3").load(outputs, "s")

    def test_the_helper_calls_the_public_function(self):
        source = inspect.getsource(prediction._evobind_score_of)
        assert "compute_evobind_score_from_record" in source
        assert "_score_from_atoms" not in source

    def test_it_gives_the_score_of_the_public_function(self, tmp_path):
        record = self._record(tmp_path)
        ours = prediction._evobind_score_of(record, "B", "A")
        theirs = compute_evobind_score_from_record(record, "B", "A", interface_cutoff_angstrom=8.0)
        assert ours == theirs
        assert math.isfinite(ours["evobind_score"]) and ours["model"] == "of3"

    def test_a_record_without_plddt_gives_distances_a_none_score_and_a_reason(self, tmp_path):
        record = self._record(tmp_path)
        record.plddt_per_atom = None
        result = prediction._evobind_score_of(record, "B", "A")
        assert result["evobind_score"] is None
        assert "per-atom pLDDT" in result["reason"]
        assert math.isfinite(result["if_dist_pep_to_rec"])

    def test_the_cutoff_is_the_one_of_the_step(self):
        assert prediction._INTERFACE_CUTOFF_ANGSTROM == 8.0
