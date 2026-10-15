"""``binding-metrics-energy`` exits with 1 when no sample produced an energy (#160).

``compute_interaction_energy`` returns ``success=False`` and an ``error_message`` for a sample it
cannot evaluate, and the command used to print that row and exit 0, so a script that checks the
exit status saw a failed run as a good one. The rule is the one of ``binding-metrics-batch``: the
status is 1 only when every sample failed. When some failed and some succeeded the status is 0, the
table has the failed rows and a warning names them.
"""

import logging
import subprocess
import sys

import pytest
from conftest import DATA_DIR

from binding_metrics.metrics import energy
from binding_metrics.utils import _CurrentStreamHandler

EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"


@pytest.fixture
def package_logger():
    """The package logger with the console handlers of the test removed afterwards."""
    logger = logging.getLogger("binding_metrics")
    saved_level = logger.level
    yield logger
    for handler in [h for h in logger.handlers if isinstance(h, _CurrentStreamHandler)]:
        logger.removeHandler(handler)
    logger.setLevel(saved_level)


def _result(sample_id: str, success: bool) -> dict:
    return {
        "sample_id": sample_id,
        "success": success,
        "error_message": None if success else "ValueError: chain A has no terminal oxygen",
        "raw_interaction_energy": -16.0 if success else None,
    }


def _run_main(monkeypatch, tmp_path, outcomes: dict[str, bool], *extra: str):
    """Run ``main`` on one file per sample, with an energy function that returns ``outcomes``."""
    for name in outcomes:
        (tmp_path / f"{name}.pdb").touch()

    def fake_energy(path, **kwargs):
        return _result(path.stem, outcomes[path.stem])

    monkeypatch.setattr(energy, "compute_interaction_energy", fake_energy)
    argv = ["binding-metrics-energy", "--input-dir", str(tmp_path), "--glob-pattern", "*.pdb"]
    argv += ["--modes", "raw", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    energy.main()


class TestExitStatus:
    def test_every_sample_failed_exits_with_1(self, monkeypatch, tmp_path, package_logger, capsys):
        with pytest.raises(SystemExit) as stop:
            _run_main(monkeypatch, tmp_path, {"a": False, "b": False})
        assert stop.value.code == 1
        captured = capsys.readouterr()
        assert "success" in captured.out, "the table is still printed"
        assert "no sample produced an energy (2 failed)" in captured.out

    def test_the_csv_is_written_before_the_exit(self, monkeypatch, tmp_path, package_logger):
        csv = tmp_path / "energies.csv"
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, tmp_path, {"a": False}, "--output", str(csv))
        assert csv.read_text(encoding="utf-8").splitlines()[1].startswith("a,False")

    def test_every_sample_succeeded_exits_normally(
        self, monkeypatch, tmp_path, package_logger, caplog
    ):
        _run_main(monkeypatch, tmp_path, {"a": True, "b": True})
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_some_failed_exits_normally_and_names_them(
        self, monkeypatch, tmp_path, package_logger, capsys
    ):
        _run_main(monkeypatch, tmp_path, {"good": True, "bad": False, "worse": False})
        captured = capsys.readouterr()
        assert "2 of 3 sample(s) produced no energy: bad, worse" in captured.out
        assert "ValueError: chain A has no terminal oxygen" in captured.out, "the table shows why"

    def test_no_file_found_keeps_the_status(self, monkeypatch, tmp_path, package_logger):
        monkeypatch.setattr(
            sys, "argv", ["binding-metrics-energy", "--input-dir", str(tmp_path), "--modes", "raw"]
        )
        energy.main()


def test_unprepared_example_exits_with_1_from_the_command_line(tmp_path):
    """1YCR as deposited has no terminal oxygen, which the energy step refuses (before: 0)."""
    result = subprocess.run(
        [sys.executable, "-m", "binding_metrics.metrics.energy"]
        + ["--input", str(EXAMPLE_1YCR), "--modes", "raw"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert result.returncode == 1, result.stdout[-800:] + result.stderr[-800:]
    assert "without its terminal oxygen" in result.stdout + result.stderr
    assert "no sample produced an energy (1 failed)" in result.stdout
