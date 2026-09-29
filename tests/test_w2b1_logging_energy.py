"""energy.py reports progress and failures through logging, with the same text as before."""

import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

from binding_metrics.metrics import energy  # noqa: E402
from binding_metrics.utils import _CurrentStreamHandler  # noqa: E402

LOGGER_NAME = "binding_metrics.metrics.energy"


@pytest.fixture
def package_logger():
    """The package logger, with handlers installed by a CLI ``main()`` removed afterwards."""
    package = logging.getLogger("binding_metrics")
    saved_level = package.level
    yield package
    for handler in [h for h in package.handlers if isinstance(h, _CurrentStreamHandler)]:
        package.removeHandler(handler)
    package.setLevel(saved_level)


class _FailingContext:
    def setPositions(self, positions):  # noqa: N802 (OpenMM naming)
        raise RuntimeError("boom")


class _FailingSimulation:
    context = _FailingContext()


def _evaluate_with_failure(failures=None):
    return energy._evaluate_subsystem_energies(
        _FailingSimulation(), None, None, "B", "A", "obc2", "cpu", failures=failures
    )


def _records(caplog):
    return [r for r in caplog.records if r.name == LOGGER_NAME]


def test_no_print_call_outside_main():
    """Library code must log; only the CLI ``main()`` may print."""
    tree = ast.parse(Path(energy.__file__).read_text())
    offenders = []
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if func.name == "main":
            continue
        for call in ast.walk(func):
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "print":
                offenders.append((func.name, call.lineno))
    assert not offenders
    assert "print_exc" not in Path(energy.__file__).read_text()


def test_failed_subsystem_evaluation_is_logged_with_traceback(caplog):
    failures = []
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = _evaluate_with_failure(failures)
    assert result == (None, None, None)
    assert failures == ["RuntimeError: boom"]
    records = _records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING, logging.ERROR]
    assert records[0].getMessage() == "  Warning: subsystem energy evaluation failed: boom"
    assert records[1].getMessage().startswith("Traceback (most recent call last):")
    assert records[1].getMessage().endswith("RuntimeError: boom")


def test_missing_input_logs_the_error_line_and_traceback(caplog, tmp_path):
    missing = tmp_path / "absent.pdb"
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = energy.compute_interaction_energy(missing, modes=("raw",), sample_id="s1")
    assert result["success"] is False
    records = _records(caplog)
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == f"[s1] ERROR: {result['error_message']}"
    assert records[0].getMessage().startswith("[s1] ERROR: FileNotFoundError")
    assert records[-1].levelno == logging.ERROR
    assert "Traceback (most recent call last)" in records[-1].getMessage()


def test_orphaned_cysteine_repair_is_logged_with_its_label(caplog):
    from openmm.app import Element, Topology

    topology = Topology()
    residue = topology.addResidue("CYS", topology.addChain("B"))
    for name, symbol in (("N", "N"), ("CA", "C"), ("CB", "C"), ("SG", "S")):
        topology.addAtom(name, Element.getBySymbol(symbol), residue)
    positions = np.array([[0.0, 0, 0], [0.15, 0, 0], [0.2, 0.14, 0], [0.3, 0.2, 0]])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        energy._repair_orphaned_cys(topology, positions, label="s2")
    (record,) = _records(caplog)
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        "[s2]   Repairing orphaned CYS (cross-chain disulfide severed): ['CYS1']"
    )


def test_cli_prints_the_same_text_on_the_same_streams(
    package_logger, monkeypatch, capsys, tmp_path
):
    """The ERROR line stays on stdout and the traceback on stderr, as with print."""
    missing = tmp_path / "absent.pdb"
    monkeypatch.setattr(
        sys,
        "argv",
        ["binding-metrics-energy", "--input", str(missing), "--modes", "raw"],
    )
    energy.main()
    captured = capsys.readouterr()
    assert captured.out.startswith("[absent] ERROR: FileNotFoundError: Structure file not found: ")
    assert "\nResults:\n" in captured.out
    assert captured.err.startswith("Traceback (most recent call last):\n")
    assert captured.err.endswith(
        "FileNotFoundError: Structure file not found: " + str(missing) + "\n"
    )
    assert "\n\n" not in captured.err


def test_main_installs_the_console_handlers_first(package_logger, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["binding-metrics-energy", "--help"])
    with pytest.raises(SystemExit):
        energy.main()
    assert any(isinstance(h, _CurrentStreamHandler) for h in package_logger.handlers)
