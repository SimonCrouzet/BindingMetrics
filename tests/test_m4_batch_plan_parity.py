"""The plan of ``binding-metrics-batch --preflight-only`` accepts and refuses what the run does.

Both commands check the prediction options before anything runs, and ``--preflight-only`` prints
the plan of the check. For one set of options the two commands must give the same answer: the
same usage error (exit status 2), the same refusal in the plan (exit status 1), or the same plan
(exit status 0). The table below runs each case through the ``main`` of both commands and through
``run_batch`` and ``run_pipeline`` with ``preflight_only=True``.
"""

from __future__ import annotations

import shutil
import sys

import pytest

from binding_metrics.cli import batch, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import run_pipeline
from tests.test_pre_cli_run import LINEAR

#: (id, command-line options, exit status of both commands)
CASES = [
    ("cyclic on, ColabFold has no such setting", "--predictor af2 --prediction-cyclic on", 2),
    ("cyclic off, ColabFold", "--predictor af2 --prediction-cyclic off", 2),
    ("cyclic on, Boltz-2", "--predictor boltz2 --prediction-cyclic on", 0),
    ("lock threshold without a lock mode", "--predictor boltz2 --prediction-lock-threshold 1.0", 2),
    (
        "lock threshold in score-lock",
        "--predictor boltz2 --prediction-mode score-lock --prediction-lock-threshold 1.0",
        0,
    ),
    (
        "lock threshold in refold-lock, which Boltz-2 does not declare",
        "--predictor boltz2 --prediction-mode refold-lock --prediction-lock-threshold 1.0",
        1,
    ),
    (
        "lock threshold, ColabFold has no such setting",
        "--predictor af2 --prediction-lock-threshold 1.0",
        2,
    ),
    (
        "lock threshold for OpenFold3 in refold",
        "--predictor of3 --prediction-mode refold --prediction-lock-threshold 1.0",
        2,
    ),
    ("conda environment named", "--predictor af2 --prediction-conda-env someenv", 0),
    ("conda environment, empty string", "--predictor af2 --prediction-conda-env ''", 0),
    (
        "conda environment of OpenFold3 under both names",
        "--predictor of3 --prediction-conda-env a --openfold-conda-env b",
        2,
    ),
    ("legacy cyclic for another model", "--predictor boltz2 --openfold-cyclic on", 2),
    (
        "legacy conda environment for another model",
        "--predictor boltz2 --openfold-conda-env other",
        2,
    ),
    ("alignment templates for another model", "--predictor af2 --openfold-templates alignment", 2),
    ("alignment templates for OpenFold3", "--predictor of3 --openfold-templates alignment", 0),
    ("a mode the runner does not run", "--predictor af2 --prediction-mode score", 2),
    ("a mode the model does not have", "--predictor of3 --prediction-mode score-lock", 1),
    ("no MSA server", "--predictor af2 --prediction-no-msa-server", 0),
    (
        "the policy without a server setting, runner lacks it",
        "--predictor af2 --prediction-msa-policy forbid-server",
        2,
    ),
    ("an option without a predictor", "--prediction-cyclic on", 2),
]


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "in"
    path.mkdir()
    shutil.copy(LINEAR, path / "a.pdb")
    return path


def plan_exit(main, argv, monkeypatch, capsys):
    """Run ``main`` with ``argv``; return the exit status and the usage error, if any."""
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        main()
    streams = capsys.readouterr()
    error = streams.err.partition("error: ")[2].strip() if exit_info.value.code == 2 else ""
    return exit_info.value.code, error


def options_of(text):
    import shlex

    return [word for word in shlex.split(text)]


class TestTheCommandLines:
    @pytest.mark.parametrize("name, options, status", CASES, ids=[c[0] for c in CASES])
    def test_the_batch_plan_answers_as_the_run_plan(
        self, name, options, status, folder, tmp_path, monkeypatch, capsys
    ):
        words = options_of(options)
        run_status, run_error = plan_exit(
            run.main,
            [
                "binding-metrics-run",
                "--input", str(LINEAR),
                "--output-dir", str(tmp_path / "out"),
                "--metrics", "openfold",
                *words,
                "--preflight-only",
            ],
            monkeypatch,
            capsys,
        )  # fmt: skip
        batch_status, batch_error = plan_exit(
            batch.main,
            [
                "binding-metrics-batch",
                "--input-dir", str(folder),
                "--output-csv", str(tmp_path / "res" / "out.csv"),
                "--metrics", "openfold",
                *words,
                "--preflight-only",
            ],
            monkeypatch,
            capsys,
        )  # fmt: skip
        assert (run_status, batch_status) == (status, status), (run_error, batch_error)
        assert run_error == batch_error
        assert not (tmp_path / "out").exists() and not (tmp_path / "res").exists()


#: the same cases as keywords of the Python functions (those that have a keyword)
KEYWORDS = [
    ("cyclic on, ColabFold", dict(predictor="af2", prediction_cyclic="on"), "binder_cyclic"),
    ("cyclic off, ColabFold", dict(predictor="af2", prediction_cyclic="off"), "binder_cyclic"),
    ("cyclic on, Boltz-2", dict(predictor="boltz2", prediction_cyclic="on"), None),
    (
        "lock threshold, no lock mode",
        dict(predictor="boltz2", prediction_lock_threshold=1.0),
        "score-lock or refold-lock",
    ),
    (
        "lock threshold, score-lock",
        dict(predictor="boltz2", prediction_mode="score-lock", prediction_lock_threshold=1.0),
        None,
    ),
    (
        "lock threshold, ColabFold",
        dict(predictor="af2", prediction_lock_threshold=1.0),
        "lock_threshold_angstrom",
    ),
    ("conda environment, empty", dict(predictor="af2", prediction_conda_env=""), None),
    (
        "legacy cyclic, Boltz-2",
        dict(predictor="boltz2", openfold_cyclic="on"),
        "--prediction-cyclic",
    ),
    (
        "alignment templates, ColabFold",
        dict(predictor="af2", openfold_templates="alignment"),
        "--openfold-templates",
    ),
    (
        "a mode the runner does not run",
        dict(predictor="af2", prediction_mode="score"),
        "cannot be run from here",
    ),
]


class TestThePythonFunctions:
    @pytest.mark.parametrize("name, keywords, message", KEYWORDS, ids=[c[0] for c in KEYWORDS])
    def test_run_batch_and_run_pipeline_answer_alike(
        self, name, keywords, message, folder, tmp_path
    ):
        common = dict(
            metrics={"openfold"},
            skip_prep=True,
            skip_relax=True,
            openfold_conda_env=None,
            peptide_chain="B",
            receptor_chain="A",
            preflight_only=True,
        )
        if message is None:
            (row,) = run_batch([folder / "a.pdb"], tmp_path / "out", **common, **keywords)
            block = run_pipeline(LINEAR, tmp_path / "out2", **common, **keywords)["preflight"]
            assert row["batch_status"] == "ok" and block["status"] in ("ok", "warn")
            assert row["preflight_status"] == block["status"]
        else:
            with pytest.raises(ValueError, match=message) as from_batch:
                run_batch([folder / "a.pdb"], tmp_path / "out", **common, **keywords)
            with pytest.raises(ValueError, match=message) as from_run:
                run_pipeline(LINEAR, tmp_path / "out2", **common, **keywords)
            assert str(from_batch.value) == str(from_run.value)
        assert not (tmp_path / "out").exists() and not (tmp_path / "out2").exists()
