"""A failed batched OpenFold3 call leaves the rows ``partial``, not ``ok`` (#107)."""

from __future__ import annotations

from pathlib import Path

from binding_metrics.cli import batch
from binding_metrics.metrics import openfold

EXAMPLE_1YCR = Path(__file__).resolve().parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def _rows(*ids):
    return [{"sample_id": sid, "batch_status": "ok"} for sid in ids]


def _run(rows, tmp_path):
    batch._run_batched_openfold(
        rows=rows,
        sid_to_input={row["sample_id"]: EXAMPLE_1YCR for row in rows},
        output_dir=tmp_path,
        openfold_mode="score",
        openfold_conda_env=None,
        peptide_chain="B",
        receptor_chain="A",
    )


class TestAFailedCall:
    def test_every_covered_row_is_partial_with_the_reason(self, tmp_path, monkeypatch):
        def fail(**kwargs):
            raise RuntimeError("no gpu")

        monkeypatch.setattr(openfold, "run_openfold_batched", fail)
        rows = _rows("s1", "s2")
        _run(rows, tmp_path)
        for row in rows:
            assert row["batch_status"] == "partial"
            assert row["batch_failed_steps"] == "openfold"
            assert row["batch_failed_reasons"] == "openfold: no gpu"
            assert row["openfold_error"] == "no gpu"

    def test_a_row_that_had_failed_another_step_keeps_both(self, tmp_path, monkeypatch):
        def fail(**kwargs):
            raise RuntimeError("no gpu")

        monkeypatch.setattr(openfold, "run_openfold_batched", fail)
        rows = [
            {
                "sample_id": "s1",
                "batch_status": "partial",
                "batch_failed_steps": "relax",
                "batch_failed_reasons": "relax: diverged",
            }
        ]
        _run(rows, tmp_path)
        assert rows[0]["batch_failed_steps"] == "relax;openfold"
        assert rows[0]["batch_failed_reasons"] == "relax: diverged | openfold: no gpu"


class TestAFailedSample:
    def test_only_the_sample_whose_metrics_failed_is_partial(self, tmp_path, monkeypatch):
        monkeypatch.setattr(openfold, "run_openfold_batched", lambda **kw: tmp_path)

        def metrics(**kwargs):
            if kwargs["query_name"] == "bad":
                raise ValueError("no confidences file")
            return {"iptm": 0.5}

        monkeypatch.setattr(openfold, "compute_openfold_metrics", metrics)
        rows = _rows("good", "bad")
        _run(rows, tmp_path)
        good, bad = rows
        assert good["batch_status"] == "ok" and "batch_failed_steps" not in good
        assert bad["batch_status"] == "partial"
        assert bad["batch_failed_steps"] == "openfold"
        assert "no confidences file" in bad["batch_failed_reasons"]
