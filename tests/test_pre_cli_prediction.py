"""Two cleanups of ``cli/prediction.py`` after the store and the EvoBind score gained public API.

An adopted request is keyed by its name in the store itself, so the CLI no longer puts the name
into the request options; and the primary EvoBind score of a record comes from
``compute_evobind_score_from_record`` and not from a private function of the metric module.
"""

from __future__ import annotations

import pytest

from binding_metrics.cli.prediction import (
    make_request,
    make_session,
    make_store,
    run_prediction_step,
)
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
