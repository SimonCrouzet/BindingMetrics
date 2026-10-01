"""``_detect_sample_chains``: which samples a whole-batch step (OpenFold, prediction) covers."""

import logging
from pathlib import Path

from binding_metrics.cli import batch

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def test_a_sample_qualifies_with_its_detected_chains():
    rows = [{"sample_id": "s1", "batch_status": "ok"}]
    eligible = batch._detect_sample_chains(rows, {"s1": EXAMPLE_1YCR}, None, None, "prediction")
    assert eligible == [(0, "s1", EXAMPLE_1YCR, "B", "A")]


def test_given_chains_are_used():
    rows = [{"sample_id": "s1", "batch_status": "ok"}]
    (sample,) = batch._detect_sample_chains(rows, {"s1": EXAMPLE_1YCR}, "A", "B", "prediction")
    assert sample[3:] == ("A", "B")


def test_a_failed_worker_an_unknown_input_and_a_missing_id_are_left_out():
    rows = [
        {"sample_id": "bad", "batch_status": "error"},
        {"sample_id": "unknown", "batch_status": "ok"},
        {"batch_status": "ok"},
        {"sample_id": "good", "batch_status": "partial"},
    ]
    inputs = {"bad": EXAMPLE_1YCR, "good": EXAMPLE_1YCR}
    eligible = batch._detect_sample_chains(rows, inputs, None, None, "prediction")
    assert [sample[:2] for sample in eligible] == [(3, "good")]


def test_an_unreadable_input_is_named_after_the_step_and_skipped(tmp_path, caplog):
    broken = tmp_path / "broken.pdb"
    broken.write_text("not a structure\n", encoding="utf-8")
    rows = [{"sample_id": "b", "batch_status": "ok"}, {"sample_id": "g", "batch_status": "ok"}]
    with caplog.at_level(logging.WARNING, logger="binding_metrics"):
        eligible = batch._detect_sample_chains(
            rows, {"b": broken, "g": EXAMPLE_1YCR}, None, None, "OpenFold"
        )
    assert [sample[1] for sample in eligible] == ["g"]
    assert "b: skipped for OpenFold, chain detection failed" in caplog.text
