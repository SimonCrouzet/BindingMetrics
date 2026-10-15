"""Strict JSON export of ``binding-metrics-receptor-quality --output result.json`` (#220)."""

import json
import sys

import numpy as np
import pytest

from binding_metrics.metrics import receptor_quality
from tests.test_feat_c_support import EXAMPLE_1YCR
from tests.test_t3_support import (  # noqa: F401  (the fixture is used by name)
    read,
    restore_package_logger,
    strict_loads,
)

pytestmark = pytest.mark.usefixtures("restore_package_logger")

RESULT = {
    "receptor_chain": "A",
    "n_models": 1,
    "models": [],
    "summary": {
        "energy_kJ_mol": float("nan"),
        "clashscore": np.float32("inf"),
        "series": [1.0, float("-inf")],
    },
}


def _main(monkeypatch, tmp_path, *extra):
    result = {**RESULT, "summary": dict(RESULT["summary"])}
    monkeypatch.setattr(receptor_quality, "compute_receptor_quality", lambda *a, **kw: result)
    out = tmp_path / "result.json"
    argv = ["binding-metrics-receptor-quality", "-i", str(EXAMPLE_1YCR), "-o", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, *extra])
    receptor_quality.main()
    return read(out), result


def test_the_default_text_is_the_text_json_dump_gives(monkeypatch, tmp_path):
    text, result = _main(monkeypatch, tmp_path)
    assert text == json.dumps(result, indent=2, default=receptor_quality._json_default)
    assert '"energy_kJ_mol": NaN' in text and "nonfinite_fields" not in text


def test_the_flag_writes_null_and_lists_the_fields(monkeypatch, tmp_path):
    text, _ = _main(monkeypatch, tmp_path, "--nonfinite", "null")
    data = strict_loads(text)
    assert data["summary"] == {"energy_kJ_mol": None, "clashscore": None, "series": [1.0, None]}
    assert data["nonfinite_fields"] == [
        "summary.energy_kJ_mol",
        "summary.clashscore",
        "summary.series[1]",
    ]
    assert data["receptor_chain"] == "A" and data["input_filename"] == EXAMPLE_1YCR.name
