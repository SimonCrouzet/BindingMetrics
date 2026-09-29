"""Every metrics console script configures logging before it does anything else.

Library code logs through ``logging``; the record only reaches the console when a
CLI ``main()`` has called ``configure_logging``. These tests pin that the call is
there, and that the printed CLI text is unchanged by it.
"""

import ast
import importlib
import inspect
import logging
import sys
import tomllib
from pathlib import Path

import pytest

from binding_metrics.utils import _CurrentStreamHandler

ROOT = Path(__file__).resolve().parents[1]
P53_MDM2 = ROOT / "data" / "example_linear_p53_1YCR.pdb"


def _metrics_scripts() -> dict[str, str]:
    """Console scripts of pyproject.toml that live in ``binding_metrics.metrics``."""
    with open(ROOT / "pyproject.toml", "rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]
    return {
        name: target
        for name, target in scripts.items()
        if target.startswith("binding_metrics.metrics.")
    }


SCRIPTS = _metrics_scripts()


def _resolve(target: str):
    module_name, func_name = target.split(":")
    return getattr(importlib.import_module(module_name), func_name)


@pytest.fixture
def package_logger():
    """The package logger, with handlers installed by a CLI ``main()`` removed afterwards."""
    package = logging.getLogger("binding_metrics")
    saved_level = package.level
    yield package
    for handler in [h for h in package.handlers if isinstance(h, _CurrentStreamHandler)]:
        package.removeHandler(handler)
    package.setLevel(saved_level)


def test_the_metrics_scripts_are_found():
    assert {"binding-metrics-energy", "binding-metrics-openfold"} <= set(SCRIPTS)


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_main_starts_with_configure_logging(script):
    main = _resolve(SCRIPTS[script])
    tree = ast.parse(Path(inspect.getsourcefile(main)).read_text())
    (func,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == main.__name__]
    first = func.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert getattr(first.value.func, "id", None) == "configure_logging"


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_running_main_installs_the_console_handlers(script, package_logger, monkeypatch):
    for handler in [h for h in package_logger.handlers if isinstance(h, _CurrentStreamHandler)]:
        package_logger.removeHandler(handler)
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit):
        _resolve(SCRIPTS[script])()
    assert any(isinstance(h, _CurrentStreamHandler) for h in package_logger.handlers)


def test_compare_cli_prints_the_same_text_on_stdout(package_logger, monkeypatch, capsys):
    pytest.importorskip("gemmi")
    from binding_metrics.metrics.comparison import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["binding-metrics-compare", "--initial", str(P53_MDM2), "--processed", str(P53_MDM2)],
    )
    main()
    captured = capsys.readouterr()
    assert captured.out.splitlines()[:5] == [
        "Comparing structures:",
        f"  Initial:   {P53_MDM2}",
        f"  Processed: {P53_MDM2}",
        "",
        "Results:",
    ]
    assert "  rmsd: 0.000 Å" in captured.out
    assert captured.err == ""
