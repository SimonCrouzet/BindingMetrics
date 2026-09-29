"""Tests for binding-metrics-batch helpers.

Focus on the DockQ reference-matching logic (input sample → native structure by
filename stem), which is pure and testable without running the pipeline.
"""

import csv
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import batch
from binding_metrics.cli.batch import _build_reference_map, _run_one

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def _touch(path: Path) -> Path:
    path.write_text("REMARK dummy\n")
    return path


class TestBuildReferenceMap:
    def test_matches_structures_by_stem(self, tmp_path):
        _touch(tmp_path / "target1.pdb")
        _touch(tmp_path / "target2.cif")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"target1", "target2"}
        assert refs["target1"].name == "target1.pdb"
        assert refs["target2"].name == "target2.cif"

    def test_ignores_non_structure_files(self, tmp_path):
        _touch(tmp_path / "target1.pdb")
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / "scores.csv")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"target1"}

    def test_ignores_subdirectories(self, tmp_path):
        _touch(tmp_path / "target1.pdb")
        (tmp_path / "subdir").mkdir()
        _touch(tmp_path / "subdir" / "target2.pdb")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"target1"}

    def test_duplicate_stem_is_deterministic(self, tmp_path):
        # Two references share the stem 'target1'; .cif sorts before .pdb, so it wins.
        _touch(tmp_path / "target1.pdb")
        _touch(tmp_path / "target1.cif")
        refs = _build_reference_map(tmp_path)
        assert refs["target1"].suffix == ".cif"

    def test_suffix_matching_is_case_insensitive(self, tmp_path):
        _touch(tmp_path / "TARGET.PDB")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"TARGET"}

    def test_empty_dir_gives_empty_map(self, tmp_path):
        assert _build_reference_map(tmp_path) == {}

    def test_resolution_against_input_stems(self, tmp_path):
        """A sample is matched iff a reference shares its stem."""
        _touch(tmp_path / "sampleA.pdb")
        _touch(tmp_path / "sampleC.pdb")
        refs = _build_reference_map(tmp_path)
        inputs = [Path("in/sampleA.cif"), Path("in/sampleB.cif"), Path("in/sampleC.cif")]
        resolved = {p.stem: refs.get(p.stem) for p in inputs}
        assert resolved["sampleA"] is not None
        assert resolved["sampleB"] is None  # no matching reference
        assert resolved["sampleC"] is not None


def _worker_kwargs(tmp_path, **overrides):
    """Arguments for ``_run_one`` that reach the pipeline's chain check and stop."""
    kwargs = dict(
        input_path=EXAMPLE_1YCR,
        output_dir=tmp_path,
        sample_id=None,
        skip_prep=True,
        ph=7.4,
        keep_water=False,
        canonicalize=False,
        skip_relax=True,
        md_duration_ps=0.0,
        device="cuda",
        peptide_chain=None,
        receptor_chain=None,
        metrics=frozenset(),
        energy_modes=("relaxed",),
        openfold_mode="score",
        openfold_conda_env=None,
        log_file=None,
    )
    kwargs.update(overrides)
    return kwargs


class TestRunOneChains:
    def test_unknown_chain_marks_the_sample_as_error(self, tmp_path):
        row = _run_one(**_worker_kwargs(tmp_path, peptide_chain="Z"))
        assert row["batch_status"] == "error"
        assert "chain 'Z' not found; available: B (13), A (85)" in row["batch_error"]

    def test_valid_chains_run_to_completion(self, tmp_path):
        row = _run_one(**_worker_kwargs(tmp_path, peptide_chain="B", receptor_chain="A"))
        assert row["batch_status"] == "ok"
        assert row["sample_id"] == "example_linear_p53_1YCR"


class TestRunOneFailedSteps:
    """A sample whose pipeline step failed must not be reported as ok."""

    def _run(self, tmp_path, monkeypatch, pipeline_results):
        monkeypatch.setattr(batch, "run_pipeline", lambda **_: dict(pipeline_results))
        return _run_one(**_worker_kwargs(tmp_path, sample_id="s1"))

    def test_clean_pipeline_is_ok(self, tmp_path, monkeypatch):
        row = self._run(
            tmp_path,
            monkeypatch,
            {
                "sample_id": "s1",
                "energy": {"success": True, "relaxed_interaction_energy": -3.0},
                "interface": {"skipped": True},
            },
        )
        assert row["batch_status"] == "ok"
        assert "batch_failed_steps" not in row

    def test_step_error_makes_the_sample_partial(self, tmp_path, monkeypatch):
        row = self._run(
            tmp_path,
            monkeypatch,
            {
                "sample_id": "s1",
                "energy": {"error": "no template for residue BMT"},
                "geometry": {"error": "empty chain"},
                "interface": {"delta_sasa": 1200.0},
            },
        )
        assert row["batch_status"] == "partial"
        assert row["batch_failed_steps"] == "energy;geometry"
        assert "energy: no template for residue BMT" in row["batch_failed_reasons"]
        assert "geometry: empty chain" in row["batch_failed_reasons"]
        assert "batch_error" not in row
        assert row["interface_delta_sasa"] == 1200.0  # the good step is kept

    def test_success_false_makes_the_sample_partial(self, tmp_path, monkeypatch):
        row = self._run(
            tmp_path,
            monkeypatch,
            {"sample_id": "s1", "relax": {"success": False, "error_message": "KeyError: 'N'"}},
        )
        assert row["batch_status"] == "partial"
        assert row["batch_failed_steps"] == "relax"

    def test_worker_exception_stays_an_error(self, tmp_path, monkeypatch):
        def boom(**_):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(batch, "run_pipeline", boom)
        row = _run_one(**_worker_kwargs(tmp_path, sample_id="s1"))
        assert row["batch_status"] == "error"
        assert row["batch_error"] == "RuntimeError: kaboom"
        assert "batch_failed_steps" not in row


def _run_main(monkeypatch, tmp_path, rows_by_sample):
    """Run ``batch.main`` with a stubbed worker; return (exit_code, csv_rows)."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for name in rows_by_sample:
        (input_dir / f"{name}.cif").write_text("data_x\n")
    out_csv = tmp_path / "out" / "metrics.csv"

    def fake_run_one(input_path, **_):
        return dict(rows_by_sample[input_path.stem], sample_id=input_path.stem)

    monkeypatch.setattr(batch, "_run_one", fake_run_one)
    monkeypatch.setattr(
        sys,
        "argv",
        ["binding-metrics-batch", "-i", str(input_dir), "--output-csv", str(out_csv)]
        + ["--metrics", "energy"],
    )
    with pytest.raises(SystemExit) as exc:
        batch.main()
    with open(out_csv, newline="") as fh:
        return exc.value.code, list(csv.DictReader(fh))


class TestMainStatusAccounting:
    def test_partial_samples_are_not_counted_ok(self, tmp_path, monkeypatch, capsys):
        code, rows = _run_main(
            monkeypatch,
            tmp_path,
            {
                "a": {"batch_status": "ok"},
                "b": {"batch_status": "partial", "batch_failed_steps": "energy"},
            },
        )
        out = capsys.readouterr().out
        assert code == 0  # one sample is fully ok
        assert "1 ok, 0 error(s), 1 with failed steps" in out
        assert {r["sample_id"]: r["batch_status"] for r in rows} == {"a": "ok", "b": "partial"}
        assert {r["sample_id"]: r["batch_failed_steps"] for r in rows} == {"a": "", "b": "energy"}

    def test_exit_code_is_nonzero_when_no_sample_is_fully_ok(self, tmp_path, monkeypatch):
        code, _ = _run_main(
            monkeypatch,
            tmp_path,
            {"a": {"batch_status": "partial", "batch_failed_steps": "relax"}},
        )
        assert code == 1

    def test_banner_is_unchanged_without_partial_samples(self, tmp_path, monkeypatch, capsys):
        code, _ = _run_main(monkeypatch, tmp_path, {"a": {"batch_status": "ok"}})
        assert code == 0
        assert "1 ok, 0 error(s)\n" in capsys.readouterr().out
