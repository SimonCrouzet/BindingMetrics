"""Two inputs with the same file stem stop a batch before anything is created (issue #170).

The sample ID is the file stem and names the output directory, the report, the log and the
key of the prediction store, so two inputs with one stem would write the same files.
"""

import csv
import json
import shutil
import sys

import pytest
from conftest import DATA_DIR

from binding_metrics.cli import batch
from binding_metrics.cli.batch import run_batch

EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"

CHEAP = dict(skip_prep=True, skip_relax=True, metrics=frozenset())


def copy_to(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    shutil.copy(EXAMPLE_1YCR, target)
    return target


class Calls:
    """Replaces the steps that do work, to see that none of them starts."""

    def __init__(self, monkeypatch):
        self.started = []
        for name in ("_run_one", "_run_batched_prediction", "_run_batched_openfold"):
            monkeypatch.setattr(batch, name, self.record(name))

    def record(self, name):
        def stand_in(*args, **kwargs):
            self.started.append(name)
            raise AssertionError(f"{name} started")

        return stand_in


@pytest.fixture
def same_stem_in_two_folders(tmp_path):
    return [
        copy_to(tmp_path / "in" / "first", "sample.pdb"),
        copy_to(tmp_path / "in" / "second", "sample.pdb"),
    ]


@pytest.fixture
def same_stem_two_suffixes(tmp_path):
    return [
        copy_to(tmp_path / "in", "sample.pdb"),
        copy_to(tmp_path / "in", "sample.cif"),
    ]


class TestRunBatch:
    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_the_same_stem_in_two_directories_is_refused(
        self, tmp_path, same_stem_in_two_folders, n_workers
    ):
        with pytest.raises(ValueError, match="sample") as caught:
            run_batch(same_stem_in_two_folders, tmp_path / "out", n_workers=n_workers, **CHEAP)
        message = str(caught.value)
        assert all(str(path) in message for path in same_stem_in_two_folders)
        assert not (tmp_path / "out").exists()

    def test_the_same_stem_with_two_suffixes_is_refused(self, tmp_path, same_stem_two_suffixes):
        with pytest.raises(ValueError) as caught:
            run_batch(same_stem_two_suffixes, tmp_path / "out", **CHEAP)
        assert all(str(path) in str(caught.value) for path in same_stem_two_suffixes)
        assert not (tmp_path / "out").exists()

    def test_the_same_path_listed_twice_is_refused(self, tmp_path):
        path = copy_to(tmp_path / "in", "sample.pdb")
        with pytest.raises(ValueError, match="sample"):
            run_batch([path, path], tmp_path / "out", **CHEAP)
        assert not (tmp_path / "out").exists()

    def test_every_colliding_id_is_listed_with_its_paths(self, tmp_path):
        clashing = [
            copy_to(tmp_path / "in" / "first", "x.pdb"),
            copy_to(tmp_path / "in" / "second", "x.pdb"),
            copy_to(tmp_path / "in" / "first", "y.pdb"),
            copy_to(tmp_path / "in" / "first", "y.cif"),
            copy_to(tmp_path / "in" / "third", "y.pdb"),
        ]
        alone = copy_to(tmp_path / "in" / "first", "z.pdb")
        with pytest.raises(ValueError) as caught:
            run_batch([*clashing, alone], tmp_path / "out", **CHEAP)
        message = str(caught.value)
        assert all(str(path) in message for path in clashing)
        assert str(alone) not in message

    def test_nothing_starts_and_nothing_is_written(
        self, tmp_path, monkeypatch, same_stem_in_two_folders
    ):
        calls = Calls(monkeypatch)
        log_file = tmp_path / "logs" / "all.log"
        with pytest.raises(ValueError):
            run_batch(
                same_stem_in_two_folders,
                tmp_path / "out",
                metrics={"interface", "openfold"},
                log_file=log_file,
                n_workers=2,
            )
        assert calls.started == []
        assert sorted(path.name for path in tmp_path.iterdir()) == ["in"]

    @pytest.mark.parametrize("preflight_only", [False, True])
    def test_a_prediction_directory_does_not_pair_two_inputs_with_one_output(
        self, tmp_path, monkeypatch, same_stem_in_two_folders, preflight_only
    ):
        calls = Calls(monkeypatch)
        predictions = tmp_path / "predictions"
        (predictions / "sample").mkdir(parents=True)
        with pytest.raises(ValueError, match="sample"):
            run_batch(
                same_stem_in_two_folders,
                tmp_path / "out",
                skip_relax=True,
                metrics={"interface", "openfold"},
                predictor="boltz2",
                prediction_dir=predictions,
                preflight_only=preflight_only,
            )
        assert calls.started == []
        assert not (tmp_path / "out").exists()

    def test_a_predictor_run_from_here_is_refused_before_the_model_starts(
        self, tmp_path, monkeypatch, same_stem_in_two_folders
    ):
        calls = Calls(monkeypatch)
        with pytest.raises(ValueError, match="sample"):
            run_batch(
                same_stem_in_two_folders,
                tmp_path / "out",
                skip_relax=True,
                metrics={"interface", "openfold"},
                predictor="boltz2",
            )
        assert calls.started == []
        assert not (tmp_path / "out").exists()


class TestBatchWithoutCollisions:
    def test_directories_do_not_matter_when_the_stems_differ(self, tmp_path):
        paths = [
            copy_to(tmp_path / "in" / "first", "a.pdb"),
            copy_to(tmp_path / "in" / "second", "b.pdb"),
        ]
        rows = run_batch(paths, tmp_path / "out", **CHEAP)
        assert [row["sample_id"] for row in rows] == ["a", "b"]
        assert [row["batch_status"] for row in rows] == ["ok", "ok"]
        assert [row["input"] for row in rows] == [str(path) for path in paths]
        reports = sorted(
            path.relative_to(tmp_path / "out") for path in (tmp_path / "out").rglob("*.json")
        )
        assert [str(path) for path in reports] == ["a/a_results.json", "b/b_results.json"]
        for name, path in zip("ab", paths):
            report = json.loads(
                (tmp_path / "out" / name / f"{name}_results.json").read_text(encoding="utf-8")
            )
            assert report["input"] == str(path)

    def test_a_stem_that_contains_another_is_not_a_collision(self, tmp_path):
        paths = [copy_to(tmp_path / "in", "sample.pdb"), copy_to(tmp_path / "in", "sample_2.pdb")]
        rows = run_batch(paths, tmp_path / "out", **CHEAP)
        assert [row["sample_id"] for row in rows] == ["sample", "sample_2"]
        assert {row["batch_status"] for row in rows} == {"ok"}


def run_main(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["binding-metrics-batch", *map(str, args)])
    with pytest.raises(SystemExit) as exited:
        batch.main()
    captured = capsys.readouterr()
    return exited.value.code, captured.out, captured.err


class TestMain:
    def test_the_same_stem_with_two_suffixes_is_a_usage_error(
        self, tmp_path, monkeypatch, capsys, same_stem_two_suffixes
    ):
        calls = Calls(monkeypatch)
        csv_path = tmp_path / "res" / "out.csv"
        code, _, err = run_main(
            monkeypatch,
            capsys,
            "-i",
            same_stem_two_suffixes[0].parent,
            "--output-csv",
            csv_path,
            "--metrics",
            "interface",
        )
        assert code == 2
        assert "usage:" in err
        assert all(str(path) in err for path in same_stem_two_suffixes)
        assert calls.started == []
        assert not (tmp_path / "res").exists()

    @pytest.mark.parametrize("workers", ["1", "2"])
    def test_a_recursive_glob_finds_the_same_stem_in_two_directories(
        self, tmp_path, monkeypatch, capsys, same_stem_in_two_folders, workers
    ):
        calls = Calls(monkeypatch)
        out = tmp_path / "res"
        code, _, err = run_main(
            monkeypatch,
            capsys,
            "-i",
            tmp_path / "in",
            "--glob",
            "**/*.pdb",
            "--output-csv",
            out / "out.csv",
            "--output-dir",
            out / "samples",
            "--workers",
            workers,
            "--metrics",
            "interface",
        )
        assert code == 2
        assert all(str(path) in err for path in same_stem_in_two_folders)
        assert calls.started == []
        assert not out.exists()

    def test_a_prediction_directory_does_not_hide_the_collision(
        self, tmp_path, monkeypatch, capsys, same_stem_in_two_folders
    ):
        calls = Calls(monkeypatch)
        predictions = tmp_path / "predictions"
        (predictions / "sample").mkdir(parents=True)
        code, _, err = run_main(
            monkeypatch,
            capsys,
            "-i",
            tmp_path / "in",
            "--glob",
            "*/*.pdb",
            "--output-csv",
            tmp_path / "res" / "out.csv",
            "--metrics",
            "interface,openfold",
            "--predictor",
            "boltz2",
            "--prediction-dir",
            predictions,
        )
        assert code == 2
        assert all(str(path) in err for path in same_stem_in_two_folders)
        assert calls.started == []
        assert not (tmp_path / "res").exists()

    def test_preflight_only_checks_the_stems_too(
        self, tmp_path, monkeypatch, capsys, same_stem_in_two_folders
    ):
        code, out, err = run_main(
            monkeypatch,
            capsys,
            "-i",
            tmp_path / "in",
            "--glob",
            "*/*.pdb",
            "--output-csv",
            tmp_path / "res" / "out.csv",
            "--preflight-only",
        )
        assert code == 2
        assert all(str(path) in err for path in same_stem_in_two_folders)
        assert "Pre-flight:" not in out

    def test_a_batch_without_collisions_writes_the_same_csv_and_reports(
        self, tmp_path, monkeypatch, capsys
    ):
        copy_to(tmp_path / "in" / "first", "a.pdb")
        copy_to(tmp_path / "in" / "second", "b.pdb")
        out = tmp_path / "res"
        code, stdout, err = run_main(
            monkeypatch,
            capsys,
            "-i",
            tmp_path / "in",
            "--glob",
            "*/*.pdb",
            "--output-csv",
            out / "out.csv",
            "--metrics",
            "interface",
            "--skip-prep",
            "--skip-relax",
        )
        assert code == 0, err
        assert "2 ok, 0 error(s)" in stdout
        with open(out / "out.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [row["sample_id"] for row in rows] == ["a", "b"]
        assert (out / "a" / "a_results.json").is_file()
        assert (out / "b" / "b_results.json").is_file()
