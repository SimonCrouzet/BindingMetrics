"""``run_batch``: the in-process counterpart of ``binding-metrics-batch`` (issue #32)."""

import csv
import shutil
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

from binding_metrics.cli import batch
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import ChainNotFoundError

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

# Nothing is prepped, relaxed or scored: each item only detects its chains, which
# keeps a sample well under a second and still exercises the real worker.
CHEAP = dict(skip_prep=True, skip_relax=True, metrics=frozenset())


@pytest.fixture
def structures(tmp_path):
    """Four copies of 1YCR (chain B is the 13-residue peptide, chain A is MDM2)."""
    folder = tmp_path / "in"
    folder.mkdir()
    paths = []
    for name in ("d", "b", "a", "c"):  # deliberately not alphabetical
        target = folder / f"{name}.pdb"
        shutil.copy(EXAMPLE_1YCR, target)
        paths.append(target)
    return paths


@pytest.fixture
def broken(tmp_path):
    path = tmp_path / "in" / "broken.cif"
    path.parent.mkdir(exist_ok=True)
    path.write_text("this is not a structure\n")
    return path


class TestResultOrder:
    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_rows_follow_the_order_of_paths(self, tmp_path, structures, n_workers):
        rows = run_batch(structures, tmp_path / "out", n_workers=n_workers, **CHEAP)
        assert [row["sample_id"] for row in rows] == ["d", "b", "a", "c"]
        assert {row["batch_status"] for row in rows} == {"ok"}

    def test_each_item_has_its_own_directory_and_report(self, tmp_path, structures):
        run_batch(structures, tmp_path / "out", **CHEAP)
        for name in ("a", "b", "c", "d"):
            assert (tmp_path / "out" / name / f"{name}_results.json").exists()

    def test_rows_are_the_rows_the_csv_gets(self, tmp_path, structures):
        rows = run_batch(structures[:1], tmp_path / "out", **CHEAP)
        (row,) = rows
        assert row["sample_id"] == "d"
        assert row["provenance_seed"] == 1  # the library default seed

    def test_no_paths_give_no_rows(self, tmp_path):
        assert run_batch([], tmp_path / "out", **CHEAP) == []


class TestErrorIsolation:
    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_a_failing_item_is_recorded_and_the_others_still_run(
        self, tmp_path, structures, broken, n_workers
    ):
        paths = [structures[0], broken, structures[1]]
        rows = run_batch(paths, tmp_path / "out", n_workers=n_workers, **CHEAP)
        assert [row["sample_id"] for row in rows] == ["d", "broken", "b"]
        assert [row["batch_status"] for row in rows] == ["ok", "error", "ok"]
        assert rows[1]["batch_error"]  # "<ExceptionType>: <message>"

    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_raise_mode_re_raises_the_items_exception(self, tmp_path, structures, n_workers):
        with pytest.raises(ChainNotFoundError, match="chain 'Z' not found"):
            run_batch(
                structures,
                tmp_path / "out",
                peptide_chain="Z",
                on_error="raise",
                n_workers=n_workers,
                **CHEAP,
            )

    def test_record_mode_reports_the_same_failure_as_an_error_row(self, tmp_path, structures):
        rows = run_batch(structures[:2], tmp_path / "out", peptide_chain="Z", **CHEAP)
        assert [row["batch_status"] for row in rows] == ["error", "error"]
        assert all("chain 'Z' not found" in row["batch_error"] for row in rows)

    def test_a_failure_outside_the_pipeline_is_recorded_too(self, tmp_path, monkeypatch):
        good = {"sample_id": "ok", "batch_status": "ok"}

        def flaky(input_path, **_):
            if input_path.stem == "bad":
                raise RuntimeError("kaboom")
            return dict(good)

        monkeypatch.setattr(batch, "_run_one", flaky)
        rows = run_batch([Path("ok.cif"), Path("bad.cif"), Path("ok2.cif")], tmp_path, **CHEAP)
        assert [row["batch_status"] for row in rows] == ["ok", "error", "ok"]
        assert rows[1] == {
            "sample_id": "bad",
            "batch_status": "error",
            "batch_error": "RuntimeError: kaboom",
        }
        with pytest.raises(RuntimeError, match="kaboom"):
            run_batch([Path("bad.cif")], tmp_path, on_error="raise", **CHEAP)

    def test_partial_rows_are_not_exceptions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            batch,
            "run_pipeline",
            lambda **kw: {"sample_id": kw["sample_id"], "energy": {"error": "no template"}},
        )
        for mode in ("record", "raise"):
            (row,) = run_batch([EXAMPLE_1YCR], tmp_path / mode, on_error=mode, **CHEAP)
            assert row["batch_status"] == "partial"
            assert row["batch_failed_steps"] == "energy"


class TestCallbacks:
    def test_on_result_gets_each_row_as_it_finishes(self, tmp_path, structures):
        seen = []
        rows = run_batch(structures, tmp_path / "out", on_result=seen.append, **CHEAP)
        # One worker: the order of completion is the order of the paths.
        assert [row["sample_id"] for row in seen] == ["d", "b", "a", "c"]
        assert all(a is b for a, b in zip(seen, rows))

    def test_on_result_runs_between_items_not_after_the_last(self, tmp_path, monkeypatch):
        events = []

        def worker(input_path, **_):
            events.append(f"run {input_path.stem}")
            return {"batch_status": "ok"}

        monkeypatch.setattr(batch, "_run_one", worker)
        run_batch(
            [Path("a.cif"), Path("b.cif")],
            tmp_path,
            on_start=lambda path: events.append(f"start {path.stem}"),
            on_result=lambda row: events.append(f"done {row['sample_id']}"),
            **CHEAP,
        )
        assert events == ["start a", "run a", "done a", "start b", "run b", "done b"]

    def test_on_result_also_sees_recorded_errors(self, tmp_path, structures, broken):
        seen = []
        run_batch([broken, structures[0]], tmp_path / "out", on_result=seen.append, **CHEAP)
        assert [row["batch_status"] for row in seen] == ["error", "ok"]

    def test_with_workers_every_row_is_delivered_once(self, tmp_path, structures):
        seen = []
        rows = run_batch(structures, tmp_path / "out", n_workers=2, on_result=seen.append, **CHEAP)
        assert sorted(row["sample_id"] for row in seen) == ["a", "b", "c", "d"]
        assert [row["sample_id"] for row in rows] == ["d", "b", "a", "c"]

    def test_a_callback_that_raises_stops_the_batch(self, tmp_path, structures):
        def stop(row):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_batch(structures, tmp_path / "out", on_result=stop, **CHEAP)


class TestArguments:
    def test_bad_arguments_are_refused_before_anything_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "_run_one", lambda **kw: pytest.fail("ran an item"))
        for kwargs, message in (
            ({"n_workers": 0}, "n_workers must be at least 1"),
            ({"on_error": "ignore"}, "on_error must be 'record' or 'raise'"),
            ({"metrics": {"energy", "nonsense"}}, "Unknown metric.*nonsense"),
            ({"peptide_chain": "B", "binder_chain": "A"}, "peptide_chain='B' and binder_chain='A'"),
        ):
            with pytest.raises(ValueError, match=message):
                run_batch([EXAMPLE_1YCR], tmp_path, **{**CHEAP, **kwargs})

    def test_binder_and_target_aliases_select_the_chains(self, tmp_path, structures):
        (row,) = run_batch(
            structures[:1], tmp_path / "out", binder_chain="B", target_chain="A", **CHEAP
        )
        assert row["batch_status"] == "ok"

    def test_references_enable_dockq_and_are_matched_by_stem(self, tmp_path, monkeypatch):
        seen = {}

        def worker(input_path, reference_path, metrics, **_):
            seen[input_path.stem] = (reference_path, metrics)
            return {"batch_status": "ok"}

        monkeypatch.setattr(batch, "_run_one", worker)
        native = Path("native_a.pdb")
        run_batch(
            [Path("a.cif"), Path("b.cif")],
            tmp_path,
            references={"a": native},
            skip_prep=True,
            metrics={"interface"},
        )
        assert seen["a"] == (native, frozenset({"interface", "dockq"}))
        assert seen["b"] == (None, frozenset({"interface", "dockq"}))

    def test_openfold_runs_once_after_the_items_and_not_in_the_workers(self, tmp_path, monkeypatch):
        calls = []

        def worker(input_path, metrics, **_):
            calls.append(("item", input_path.stem, "openfold" in metrics))
            return {"sample_id": input_path.stem, "batch_status": "ok"}

        def batched(rows, openfold_seeds, **kw):
            calls.append(("openfold", [row["sample_id"] for row in rows], openfold_seeds))
            rows[0]["openfold_iptm"] = 0.8

        monkeypatch.setattr(batch, "_run_one", worker)
        monkeypatch.setattr(batch, "_run_batched_openfold", batched)
        rows = run_batch(
            [Path("a.cif"), Path("b.cif")],
            tmp_path,
            metrics={"openfold"},
            openfold_seeds=[5],
        )
        assert calls == [
            ("item", "a", False),
            ("item", "b", False),
            ("openfold", ["a", "b"], [5]),
        ]
        assert rows[0]["openfold_iptm"] == 0.8

    def test_a_shared_log_file_is_truncated_at_the_start(self, tmp_path, monkeypatch):
        shared = tmp_path / "logs" / "all.log"
        shared.parent.mkdir()
        shared.write_text("STALE\n")
        monkeypatch.setattr(batch, "_run_one", lambda **kw: {"batch_status": "ok"})
        run_batch([Path("a.cif")], tmp_path, log_file=shared, **CHEAP)
        assert shared.read_text() == ""


class _InlineExecutor:
    """Runs submitted calls immediately in this process; stands in for the process pool."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class TestCommandLineOnRunBatch:
    """``main`` prints the lines it always printed, now through the run_batch callbacks."""

    def _main(self, monkeypatch, tmp_path, worker, workers):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        for name in ("a", "b"):
            (input_dir / f"{name}.cif").write_text("data_x\n")
        out_csv = tmp_path / "m.csv"
        monkeypatch.setattr(batch, "_run_one", worker)
        monkeypatch.setattr(batch, "ProcessPoolExecutor", _InlineExecutor)
        argv = ["binding-metrics-batch", "-i", str(input_dir), "--output-csv", str(out_csv)]
        monkeypatch.setattr(sys, "argv", [*argv, "--workers", str(workers), "--metrics", "energy"])
        with pytest.raises(SystemExit) as exit_request:
            batch.main()
        with open(out_csv, newline="") as handle:
            return exit_request.value.code, list(csv.DictReader(handle))

    @staticmethod
    def _worker(input_path, **_):
        if input_path.stem == "b":
            raise RuntimeError("worker died")
        return {"sample_id": input_path.stem, "batch_status": "ok", "total_elapsed_s": 1.5}

    def test_sequential_progress_lines(self, monkeypatch, tmp_path, capsys):
        code, rows = self._main(monkeypatch, tmp_path, self._worker, workers=1)
        out = capsys.readouterr().out
        assert code == 0
        assert (
            "[1/2] Processing: a\n  -> ok  (1.5s)\n"
            "[2/2] Processing: b\n  -> ERROR: RuntimeError: worker died\n"
        ) in out
        assert "DONE in" in out and "1 ok, 1 error(s)\n" in out
        assert [row["sample_id"] for row in rows] == ["a", "b"]

    def test_parallel_progress_lines(self, monkeypatch, tmp_path, capsys):
        code, rows = self._main(monkeypatch, tmp_path, self._worker, workers=2)
        out = capsys.readouterr().out
        assert code == 0
        assert "[1/2] a -> ok (1.5s)\n" in out
        assert "[2/2] b -> FATAL: worker died\n" in out
        assert "Processing:" not in out
        assert [row["batch_status"] for row in rows] == ["ok", "error"]
        assert rows[1]["batch_error"] == "RuntimeError: worker died"
