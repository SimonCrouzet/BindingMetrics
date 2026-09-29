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


class TestPerSampleLog:
    def test_default_is_one_log_inside_each_sample_dir(self, tmp_path):
        path = batch._resolve_log_path(tmp_path / "s1", "s1", None)
        assert path == tmp_path / "s1" / "s1.log"

    def test_explicit_log_file_wins(self, tmp_path):
        shared = tmp_path / "all.log"
        assert batch._resolve_log_path(tmp_path / "s1", "s1", shared) == shared

    def test_worker_writes_its_own_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "run_pipeline", lambda **_: {"sample_id": "s1"})
        _run_one(**_worker_kwargs(tmp_path, sample_id="s1"))
        assert "binding-metrics-batch worker: s1" in (tmp_path / "s1" / "s1.log").read_text()

    def test_flag_is_accepted_and_reported_ignored_with_log_file(
        self, tmp_path, monkeypatch, capsys
    ):
        rows = {"a": {"batch_status": "ok"}}
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.cif").write_text("data_x\n")
        monkeypatch.setattr(
            batch, "_run_one", lambda input_path, **_: dict(rows["a"], sample_id="a")
        )
        argv = [
            "binding-metrics-batch",
            "-i",
            str(input_dir),
            "--output-csv",
            str(tmp_path / "m.csv"),
        ]
        monkeypatch.setattr(sys, "argv", argv + ["--per-sample-log"])
        with pytest.raises(SystemExit) as exc:
            batch.main()
        assert exc.value.code == 0
        assert "ignored" not in capsys.readouterr().err

        monkeypatch.setattr(
            sys, "argv", argv + ["--per-sample-log", "--log-file", str(tmp_path / "x.log")]
        )
        with pytest.raises(SystemExit):
            batch.main()
        assert "--per-sample-log is ignored because --log-file was given" in capsys.readouterr().err


class TestSharedLogFile:
    def test_log_to_file_append_mode_keeps_earlier_content(self, tmp_path):
        from binding_metrics.cli import log_to_file

        log = tmp_path / "x.log"
        with log_to_file(log):
            print("first")
        with log_to_file(log, mode="a"):
            print("second")
        assert log.read_text().split() == ["first", "second"]
        with log_to_file(log):  # default mode still starts afresh
            print("third")
        assert log.read_text().split() == ["third"]

    def test_every_sample_stays_in_a_shared_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "run_pipeline", lambda **_: {})
        shared = tmp_path / "logs" / "all.log"
        for sid in ("s1", "s2"):
            _run_one(**_worker_kwargs(tmp_path, sample_id=sid, log_file=shared))
        text = shared.read_text()
        assert "worker: s1" in text
        assert "worker: s2" in text

    def test_main_starts_the_shared_log_file_afresh(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.cif").write_text("data_x\n")
        shared = tmp_path / "all.log"
        shared.write_text("STALE LOG FROM AN EARLIER RUN\n")
        monkeypatch.setattr(batch, "_run_one", lambda input_path, **_: {"batch_status": "ok"})
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-batch", "-i", str(input_dir), "--output-csv", str(tmp_path / "m.csv")]
            + ["--log-file", str(shared)],
        )
        with pytest.raises(SystemExit):
            batch.main()
        assert shared.read_text() == ""


class TestRandomSeed:
    def _seen_seeds(self, tmp_path, monkeypatch, extra_args):
        """Run batch.main over two samples and collect the seed each worker got."""
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        for name in ("a", "b"):
            (input_dir / f"{name}.cif").write_text("data_x\n")
        seeds = []

        def fake_run_one(input_path, random_seed, **_):
            seeds.append(random_seed)
            return {"sample_id": input_path.stem, "batch_status": "ok"}

        monkeypatch.setattr(batch, "_run_one", fake_run_one)
        argv = [
            "binding-metrics-batch",
            "-i",
            str(input_dir),
            "--output-csv",
            str(tmp_path / "m.csv"),
        ]
        monkeypatch.setattr(sys, "argv", argv + extra_args)
        with pytest.raises(SystemExit):
            batch.main()
        return seeds

    def test_default_is_the_library_default_seed(self, tmp_path, monkeypatch):
        from binding_metrics.core.system import DEFAULT_RANDOM_SEED

        assert self._seen_seeds(tmp_path, monkeypatch, []) == [DEFAULT_RANDOM_SEED] * 2

    def test_explicit_seed_reaches_every_worker(self, tmp_path, monkeypatch):
        assert self._seen_seeds(tmp_path, monkeypatch, ["--random-seed", "7"]) == [7, 7]

    @pytest.mark.parametrize("value", ["none", "random", "OFF"])
    def test_none_asks_for_fresh_randomness(self, tmp_path, monkeypatch, value):
        assert self._seen_seeds(tmp_path, monkeypatch, ["--random-seed", value]) == [None, None]

    def test_worker_passes_the_seed_to_the_pipeline_and_records_it(self, tmp_path, monkeypatch):
        received = {}

        def fake_pipeline(**kwargs):
            received.update(kwargs)
            return {"sample_id": "s1", "provenance": {"seed": kwargs["random_seed"]}}

        monkeypatch.setattr(batch, "run_pipeline", fake_pipeline)
        _run_one(**_worker_kwargs(tmp_path, sample_id="s1", random_seed=11))
        assert received["random_seed"] == 11
        report = (tmp_path / "s1" / "s1_results.json").read_text()
        assert '"seed": 11' in report

    def test_worker_default_matches_the_pipeline_default(self):
        import inspect

        from binding_metrics.cli.run import run_pipeline

        worker_default = inspect.signature(_run_one).parameters["random_seed"].default
        pipeline_default = inspect.signature(run_pipeline).parameters["random_seed"].default
        assert worker_default == pipeline_default
