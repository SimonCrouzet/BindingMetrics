"""sasa.py reports a failed SASA computation through logging, with the same text as before."""

import ast
import importlib
import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics import sasa  # noqa: E402

P53_MDM2 = Path(__file__).resolve().parents[1] / "data" / "example_linear_p53_1YCR.pdb"
LOGGER_NAME = "binding_metrics.metrics.sasa"


def test_module_has_no_print_calls():
    tree = ast.parse(Path(sasa.__file__).read_text(encoding="utf-8"))
    prints = [
        call.lineno
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "print"
    ]
    assert not prints


def test_failed_static_sasa_warns_through_the_logger(caplog, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("sasa exploded")

    # biotite.structure re-exports the function under the module's own name, so the
    # module has to be fetched through importlib.
    monkeypatch.setattr(importlib.import_module("biotite.structure.sasa"), "sasa", broken)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = sasa.compute_delta_sasa_static(P53_MDM2, peptide_chain="B", receptor_chain="A")
    assert np.isnan(result["delta_sasa"])
    assert result["reason"] == "SASA computation failed: RuntimeError: sasa exploded"
    records = [r for r in caplog.records if r.name == LOGGER_NAME]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.WARNING, "  Warning: biotite SASA computation failed: sasa exploded")
    ]
