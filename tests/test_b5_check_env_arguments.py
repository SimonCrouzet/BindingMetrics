"""``binding-metrics-check-env`` parses its command line (#161).

The command read no argument: ``--help`` ran the whole check (OpenMM's test installation, which
loads CUDA, an MDTraj run and the OpenFold3 probe, about ten seconds) and any other argument was
ignored. It now has an argument parser: ``--help`` prints the usage and what is checked and runs
nothing, an unknown argument exits with status 2 before any check, and a call without argument
does what it did.

The checks are replaced by a recorder in the in-process tests and never run in the subprocess
tests, which stop at the parser, so none of these tests loads CUDA.
"""

import subprocess
import sys

import pytest

from binding_metrics.cli import check_env


@pytest.fixture
def recorded_checks(monkeypatch):
    """Replace the checks by one that records that it ran and passes."""
    calls: list[str] = []

    def check() -> bool:
        calls.append("ran")
        return True

    monkeypatch.setattr(check_env, "CHECKS", [("stub", check)])
    return calls


def _main(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["binding-metrics-check-env", *arguments])
    check_env.main()


class TestArguments:
    def test_help_prints_what_is_checked_and_runs_nothing(
        self, monkeypatch, recorded_checks, capsys
    ):
        with pytest.raises(SystemExit) as stop:
            _main(monkeypatch, "--help")
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith("usage: binding-metrics-check-env")
        for name in ("OpenMM", "MDTraj", "OpenFold3"):
            assert name in out
        assert "Exit status" in out
        assert recorded_checks == []

    @pytest.mark.parametrize("argument", ["--bogus", "extra", "-x"])
    def test_an_unknown_argument_exits_with_2_before_any_check(
        self, monkeypatch, recorded_checks, capsys, argument
    ):
        with pytest.raises(SystemExit) as stop:
            _main(monkeypatch, argument)
        assert stop.value.code == 2
        captured = capsys.readouterr()
        assert "unrecognized arguments" in captured.err
        assert "environment check" not in captured.out
        assert recorded_checks == []

    def test_without_argument_the_checks_run_as_before(self, monkeypatch, recorded_checks, capsys):
        with pytest.raises(SystemExit) as stop:
            _main(monkeypatch)
        assert stop.value.code == 0
        assert recorded_checks == ["ran"]
        assert "All 1 check(s) passed" in capsys.readouterr().out

    def test_a_failed_check_still_exits_with_1(self, monkeypatch, capsys):
        monkeypatch.setattr(check_env, "CHECKS", [("stub", lambda: False)])
        with pytest.raises(SystemExit) as stop:
            _main(monkeypatch)
        assert stop.value.code == 1
        assert "0 passed, 1 failed" in capsys.readouterr().out


def _command(*arguments: str) -> subprocess.CompletedProcess:
    code = "import sys; from binding_metrics.cli.check_env import main; main()"
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


@pytest.mark.slow
class TestCommandLine:
    def test_help_exits_with_0_and_checks_nothing(self):
        result = _command("--help")
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert "environment check" not in result.stdout
        assert "Running openmm" not in result.stdout

    def test_an_unknown_argument_exits_with_2_and_checks_nothing(self):
        result = _command("--bogus")
        assert result.returncode == 2
        assert "unrecognized arguments: --bogus" in result.stderr
        assert "environment check" not in result.stdout
