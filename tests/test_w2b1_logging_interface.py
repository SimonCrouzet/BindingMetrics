"""interface.py reports its warnings through logging, with the same text as before."""

import ast
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics import interface, polar_contacts  # noqa: E402
from binding_metrics.utils import _CurrentStreamHandler  # noqa: E402

P53_MDM2 = Path(__file__).resolve().parents[1] / "data" / "example_linear_p53_1YCR.pdb"
LOGGER_NAME = "binding_metrics.metrics.interface"


@pytest.fixture
def package_logger():
    """The package logger, with handlers installed by a CLI ``main()`` removed afterwards."""
    package = logging.getLogger("binding_metrics")
    saved_level = package.level
    yield package
    for handler in [h for h in package.handlers if isinstance(h, _CurrentStreamHandler)]:
        package.removeHandler(handler)
    package.setLevel(saved_level)


def _messages(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == LOGGER_NAME]


def test_no_print_call_outside_main():
    """Library code must log; only the CLI ``main()`` may print."""
    tree = ast.parse(Path(interface.__file__).read_text(encoding="utf-8"))
    offenders = []
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if func.name == "main":
            continue
        for call in ast.walk(func):
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "print":
                offenders.append((func.name, call.lineno))
    assert not offenders


def test_empty_chain_warning_reaches_the_logger(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = interface.compute_interface_metrics(P53_MDM2, design_chain="Z", receptor_chain="A")
    assert "'Z'" in result["reason"]
    assert _messages(caplog) == [(logging.WARNING, f"  Warning: Empty chain(s) in {P53_MDM2}")]


def test_sasa_failure_warning_reaches_the_logger(caplog, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("sasa exploded")

    monkeypatch.setattr(interface, "_per_atom_sasa", broken)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = interface.compute_interface_metrics(P53_MDM2)
    assert result["reason"] == "SASA computation failed: RuntimeError: sasa exploded"
    assert _messages(caplog) == [
        (logging.WARNING, "  Warning: SASA computation failed: sasa exploded")
    ]


def test_polar_contact_failure_warning_reaches_the_logger(caplog, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no hydrogens")

    monkeypatch.setattr(polar_contacts, "compute_hbonds", broken)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = interface.compute_interface_metrics(P53_MDM2)
    assert result["reason"] == "H-bond/salt bridge computation failed: RuntimeError: no hydrogens"
    assert _messages(caplog) == [
        (logging.WARNING, "  Warning: H-bond/salt bridge computation failed: no hydrogens")
    ]


def test_cli_keeps_warning_on_stdout_between_the_printed_lines(package_logger, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binding-metrics-interface",
            "--input",
            str(P53_MDM2),
            "--design-chain",
            "Z",
            "--receptor-chain",
            "A",
        ],
    )
    interface.main()
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[:2] == [
        f"Computing interface metrics for: {P53_MDM2}",
        f"  Warning: Empty chain(s) in {P53_MDM2}",
    ]
    assert lines[2:4] == ["", "Interface summary:"]
    assert captured.err == ""


def test_main_installs_the_console_handlers_first(package_logger, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["binding-metrics-interface", "--help"])
    with pytest.raises(SystemExit):
        interface.main()
    assert any(isinstance(h, _CurrentStreamHandler) for h in package_logger.handlers)
