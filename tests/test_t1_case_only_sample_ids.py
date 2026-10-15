"""Sample IDs that differ only in case are refused like equal ones (#219).

The sample ID is the file stem and names the output directory and the report. On a
case-insensitive file system ``Sample`` and ``sample`` are one directory, so the second sample
overwrote the first and both rows read ``ok``. The stems are compared case-folded, on every file
system, so a batch names the same outputs wherever it runs.
"""

import shutil
import sys
from pathlib import Path

import pytest
from conftest import DATA_DIR

from binding_metrics.cli import batch
from binding_metrics.cli.batch import _check_unique_sample_ids, run_batch

EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"

CHEAP = dict(skip_prep=True, skip_relax=True, metrics=frozenset())


class TestNormalisation:
    """The comparison of the stems; it reads no file."""

    def test_stems_that_differ_only_in_case_are_refused_with_their_paths(self):
        paths = [Path("first/Sample.pdb"), Path("second/sample.pdb")]
        with pytest.raises(ValueError) as caught:
            _check_unique_sample_ids(paths)
        message = str(caught.value)
        assert all(str(path) in message for path in paths)
        assert "Sample" in message
        assert "case" in message

    def test_the_ids_of_a_group_are_listed_once_each(self):
        paths = [Path("a/Sample.pdb"), Path("b/sample.pdb"), Path("c/sample.cif")]
        with pytest.raises(ValueError) as caught:
            _check_unique_sample_ids(paths)
        heading = [line for line in str(caught.value).splitlines() if line.startswith("  S")]
        assert heading == ["  Sample, sample"]

    def test_every_group_is_listed_and_a_lone_stem_is_not(self):
        clashing = [
            Path("a/X.pdb"),
            Path("b/x.pdb"),
            Path("a/y.pdb"),
            Path("b/Y.cif"),
            Path("c/y.pdb"),
        ]
        alone = Path("a/z.pdb")
        with pytest.raises(ValueError) as caught:
            _check_unique_sample_ids([*clashing, alone])
        message = str(caught.value)
        assert all(str(path) in message for path in clashing)
        assert str(alone) not in message

    @pytest.mark.parametrize(
        ("first", "second"),
        [("é", "É"), ("Straße", "STRASSE"), ("ΣΑΣ", "σας"), ("run_1A", "RUN_1a")],
    )
    def test_the_folding_is_the_one_of_unicode(self, first, second):
        paths = [Path(f"a/{first}.pdb"), Path(f"b/{second}.pdb")]
        with pytest.raises(ValueError):
            _check_unique_sample_ids(paths)

    @pytest.mark.parametrize(
        "names",
        [
            ["sample.pdb", "sample_2.pdb"],
            ["Sample.pdb", "Other.pdb"],
            ["sample.pdb", "sample1.pdb", "SampleA.pdb"],
            ["a.b.pdb", "a.pdb"],
        ],
    )
    def test_stems_that_differ_by_more_than_case_pass(self, names):
        _check_unique_sample_ids([Path("in") / name for name in names])

    def test_equal_stems_are_still_refused(self):
        paths = [Path("a/sample.pdb"), Path("b/sample.pdb")]
        with pytest.raises(ValueError, match="sample") as caught:
            _check_unique_sample_ids(paths)
        assert "  sample\n" in str(caught.value)


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
def case_only_pair(tmp_path):
    """Two inputs in two folders, so the pair exists on a file system that ignores case too."""
    return [
        copy_to(tmp_path / "in" / "first", "Sample.pdb"),
        copy_to(tmp_path / "in" / "second", "sample.pdb"),
    ]


class TestRunBatch:
    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_a_case_only_pair_is_refused_before_anything_is_written(
        self, tmp_path, monkeypatch, case_only_pair, n_workers
    ):
        calls = Calls(monkeypatch)
        with pytest.raises(ValueError) as caught:
            run_batch(
                case_only_pair,
                tmp_path / "out",
                metrics={"interface", "openfold"},
                log_file=tmp_path / "logs" / "all.log",
                n_workers=n_workers,
            )
        assert all(str(path) in str(caught.value) for path in case_only_pair)
        assert calls.started == []
        assert sorted(path.name for path in tmp_path.iterdir()) == ["in"]

    def test_a_prediction_directory_does_not_pair_them_with_one_output(
        self, tmp_path, monkeypatch, case_only_pair
    ):
        calls = Calls(monkeypatch)
        predictions = tmp_path / "predictions"
        (predictions / "sample").mkdir(parents=True)
        with pytest.raises(ValueError, match="Sample"):
            run_batch(
                case_only_pair,
                tmp_path / "out",
                skip_relax=True,
                metrics={"interface", "openfold"},
                predictor="boltz2",
                prediction_dir=predictions,
            )
        assert calls.started == []
        assert not (tmp_path / "out").exists()

    def test_the_pre_flight_check_refuses_them_too(self, tmp_path, case_only_pair):
        with pytest.raises(ValueError, match="Sample"):
            run_batch(case_only_pair, tmp_path / "out", preflight_only=True, **CHEAP)
        assert not (tmp_path / "out").exists()


def run_main(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["binding-metrics-batch", *map(str, args)])
    with pytest.raises(SystemExit) as exited:
        batch.main()
    captured = capsys.readouterr()
    return exited.value.code, captured.out, captured.err


class TestMain:
    def test_a_case_only_pair_is_a_usage_error(self, tmp_path, monkeypatch, capsys, case_only_pair):
        calls = Calls(monkeypatch)
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
            "interface",
        )
        assert code == 2
        assert "usage:" in err
        assert all(str(path) in err for path in case_only_pair)
        assert calls.started == []
        assert not (tmp_path / "res").exists()


def ignores_case(folder: Path) -> bool:
    """Whether ``folder`` is on a file system that gives ``a`` and ``A`` one name."""
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / "a"
    probe.write_text("", encoding="utf-8")
    try:
        return (folder / "A").exists()
    finally:
        probe.unlink()


@pytest.fixture
def case_insensitive_destination(tmp_path):
    if not ignores_case(tmp_path / "probe"):
        pytest.skip("the destination tells a from A, so the two IDs would not share an output")
    return tmp_path


@pytest.mark.integration
class TestCaseInsensitiveDestination:
    """The report that the second sample overwrote, on the file system where it happened."""

    def test_the_pair_is_refused_before_an_output_directory_exists(
        self, case_insensitive_destination, case_only_pair
    ):
        out = case_insensitive_destination / "out"
        with pytest.raises(ValueError) as caught:
            run_batch(case_only_pair, out, **CHEAP)
        assert all(str(path) in str(caught.value) for path in case_only_pair)
        assert not out.exists()

    def test_the_other_samples_of_a_batch_run_as_before(self, case_insensitive_destination):
        paths = [
            copy_to(case_insensitive_destination / "in", "Sample.pdb"),
            copy_to(case_insensitive_destination / "in", "Other.pdb"),
        ]
        out = case_insensitive_destination / "out"
        rows = run_batch(paths, out, skip_prep=True, skip_relax=True, metrics={"interface"})
        assert [row["sample_id"] for row in rows] == ["Sample", "Other"]
        assert {row["batch_status"] for row in rows} == {"ok"}
        assert (out / "Sample" / "Sample_results.json").is_file()
        assert (out / "Other" / "Other_results.json").is_file()
