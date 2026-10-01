"""The ``results["prediction"]`` block in the CSV row and the text report.

The block is what ``binding-metrics-run --predictor`` writes: the keys of
``summarize_prediction`` (with ``model``), the EvoBind keys and ``cache``. ``_md_openfold``
shares its table with the new section and must not change.
"""

import csv

import numpy as np

from binding_metrics.protocols.report import (
    _build_summary,
    _flatten,
    _md_openfold,
    _md_prediction,
    write_report,
)


def _prediction(**overrides) -> dict:
    block = {
        "model": "boltz2",
        "query_name": "s1",
        "seed": 1,
        "sample": 1,
        "structure_path": "store/model.cif",
        "avg_plddt": 81.234,
        "ptm": 0.77,
        "iptm": 0.66,
        "gpde": 1.5,
        "chain_ptm": {"A": 0.9, "B": 0.7},
        "plddt_per_atom": np.linspace(50.0, 90.0, 1500),
        "binder_plddt_per_residue": np.array([90.0, 65.5, 88.0, 40.25]),
        "mean_interface_pae": 4.256,
        "binder_ca_rmsd": float("nan"),
        "evobind_score": 3.5,
        "delta_com_angstrom": 1.25,
        "adversary_model": "boltz2",
        "timing": {"runtime_s": 12.5},
        "cache": {"runs": 1, "hits": 0, "adopted": 0, "request_key": "ab" * 32},
    }
    block.update(overrides)
    return block


class TestCsvRow:
    def test_scalars_become_prediction_columns(self):
        flat = _flatten({"sample_id": "s1", "prediction": _prediction()})
        assert flat["prediction_model"] == "boltz2"
        assert flat["prediction_avg_plddt"] == 81.234
        assert flat["prediction_iptm"] == 0.66
        assert flat["prediction_evobind_score"] == 3.5
        assert flat["prediction_delta_com_angstrom"] == 1.25
        assert flat["prediction_chain_ptm_A"] == 0.9
        assert flat["prediction_timing_runtime_s"] == 12.5

    def test_the_cache_counters_are_columns(self):
        flat = _flatten({"prediction": _prediction()})
        assert flat["prediction_cache_runs"] == 1
        assert flat["prediction_cache_hits"] == 0
        assert flat["prediction_cache_request_key"] == "ab" * 32

    def test_arrays_stay_out(self):
        flat = _flatten({"prediction": _prediction()})
        assert "prediction_plddt_per_atom" not in flat
        assert "prediction_binder_plddt_per_residue" not in flat

    def test_the_openfold_columns_are_untouched(self):
        flat = _flatten({"openfold": {"iptm": 0.5}, "prediction": _prediction()})
        assert flat["openfold_iptm"] == 0.5
        assert "openfold_model" not in flat

    def test_an_error_block_gives_error_and_model_columns(self):
        block = {"model": "af2", "error": "no output", "cache": {"runs": 0}}
        flat = _flatten({"prediction": block})
        assert flat["prediction_error"] == "no output"
        assert flat["prediction_model"] == "af2"

    def test_a_skipped_block_gives_one_column(self):
        assert _flatten({"prediction": {"skipped": True}})["prediction_skipped"] is True

    def test_the_single_sample_csv_has_the_columns(self, tmp_path):
        path = write_report(
            {"sample_id": "s1", "input": "s1.cif", "prediction": _prediction()},
            tmp_path,
            "s1",
            fmt="csv",
        )
        with open(path, newline="", encoding="utf-8") as handle:
            (row,) = list(csv.DictReader(handle))
        assert row["prediction_model"] == "boltz2"
        assert row["prediction_iptm"] == "0.66"
        assert "prediction_plddt_per_atom" not in row


class TestTextReport:
    def test_the_heading_names_the_model_by_its_display_name(self):
        text = _md_prediction(_prediction(model="boltz2"))
        assert text.startswith("## Structure prediction (Boltz-2)\n")

    def test_of3_is_named_openfold3(self):
        assert "## Structure prediction (OpenFold3)" in _md_prediction(_prediction(model="of3"))

    def test_an_unregistered_model_keeps_its_key(self):
        assert "## Structure prediction (mystery)" in _md_prediction(_prediction(model="mystery"))

    def test_the_confidence_table_has_the_openfold_rows(self):
        text = _md_prediction(_prediction())
        assert "| avg pLDDT" in text and "81.23" in text
        assert "| pTM" in text and "0.770" in text
        assert "| ipTM" in text and "0.660" in text
        assert "| gPDE" in text and "1.50 Å" in text

    def test_the_interface_and_evobind_rows_follow(self):
        text = _md_prediction(_prediction())
        assert "Mean interface PAE" in text and "4.26 Å" in text
        assert "EvoBind score" in text and "3.50" in text
        assert "Adversarial ΔCOM" in text and "1.25 Å" in text

    def test_a_value_that_was_not_computed_has_no_row(self):
        text = _md_prediction(
            _prediction(
                mean_interface_pae=float("nan"), evobind_score=None, delta_com_angstrom=None
            )
        )
        assert "Mean interface PAE" not in text
        assert "EvoBind score" not in text
        assert "ΔCOM" not in text

    def test_low_binder_plddt_is_flagged(self):
        text = _md_prediction(_prediction())
        assert "Low binder pLDDT (< 70):** res2 (65.5), res4 (40.2)" in text

    def test_the_reason_is_shown(self):
        text = _md_prediction(_prediction(reason="interface PAE: no PAE matrix"))
        assert "_Not computed: interface PAE: no PAE matrix_" in text

    def test_the_store_line_says_how_often_the_model_ran(self):
        text = _md_prediction(_prediction())
        assert "_Prediction store: 1 run(s), 0 store hit(s), 0 adopted output(s)._" in text

    def test_a_failed_prediction_shows_the_reason_only(self):
        text = _md_prediction({"model": "of3", "error": "OpenFold3 wrote no output"})
        assert text == (
            "## Structure prediction (OpenFold3)\n_Failed: OpenFold3 wrote no output_\n"
        )

    def test_skipped_and_absent(self):
        assert _md_prediction({"skipped": True}) == "## Structure prediction\n_Skipped._\n"
        assert _md_prediction({}) == "## Structure prediction\n_Absent._\n"
        assert _md_prediction(None) == "## Structure prediction\n_Absent._\n"

    def test_the_summary_places_it_after_the_openfold_section(self):
        summary = _build_summary(
            {"sample_id": "s1", "openfold": {"skipped": True}, "prediction": _prediction()}
        )
        assert summary.index("## OpenFold") < summary.index("## Structure prediction")
        assert summary.index("## Structure prediction") < summary.index("## Summary Scorecard")

    def test_the_summary_omits_the_section_when_the_key_is_absent(self):
        assert "Structure prediction" not in _build_summary({"sample_id": "s1"})


class TestOpenFoldSectionIsUnchanged:
    """``_md_openfold`` shares its table with the prediction block; its text is pinned."""

    def test_full_block(self):
        block = {
            "avg_plddt": 81.234,
            "ptm": 0.77,
            "iptm": 0.66,
            "gpde": 1.5,
            "binder_ca_rmsd": 2.345,
            "binder_plddt_per_residue": np.array([90.0, 65.5, 88.0, 40.25]),
        }
        assert _md_openfold(block) == (
            "## OpenFold\n\n"
            "| Metric         | Value  |\n"
            "| -------------- | ------ |\n"
            "| avg pLDDT      | 81.23  |\n"
            "| pTM            | 0.770  |\n"
            "| ipTM           | 0.660  |\n"
            "| gPDE           | 1.50 Å |\n"
            "| Refolding RMSD | 2.35 Å |\n\n"
            "⚠️ **Low binder pLDDT (< 70):** res2 (65.5), res4 (40.2)\n"
        )

    def test_skipped_absent_and_nan(self):
        assert _md_openfold({"skipped": True}) == "## OpenFold\n_Skipped._\n"
        assert _md_openfold({}) == "## OpenFold\n_Absent._\n"
        assert _md_openfold({"avg_plddt": float("nan"), "binder_ca_rmsd": float("nan")}) == (
            "## OpenFold\n\n"
            "| Metric    | Value |\n"
            "| --------- | ----- |\n"
            "| avg pLDDT | —     |\n"
            "| pTM       | —     |\n"
            "| ipTM      | —     |\n"
            "| gPDE      | — Å   |\n"
        )
