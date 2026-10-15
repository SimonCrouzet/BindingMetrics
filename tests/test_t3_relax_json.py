"""Strict JSON export of ``binding-metrics-relax --results-json`` (#220)."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from binding_metrics.protocols import relaxation
from binding_metrics.protocols.relaxation import RelaxationResult
from binding_metrics.protocols.report import _json_default
from tests.test_t3_support import read, restore_package_logger, strict_loads  # noqa: F401

pytestmark = pytest.mark.usefixtures("restore_package_logger")


class StubRelaxer:
    def run(self, input_path, output_dir, sample_id=None):
        return RelaxationResult(
            sample_id="s",
            success=True,
            potential_energy_minimized=float("nan"),
            potential_energy_md_avg=np.float32(-12.5),
        )


def _write(tmp_path, **kwargs):
    target = tmp_path / "relax.json"
    relaxation._run_one(StubRelaxer(), tmp_path / "in.cif", tmp_path, "s", target, None, **kwargs)
    return read(target)


def test_the_default_text_is_the_text_json_dumps_gives(tmp_path):
    expected = json.dumps(StubRelaxer().run(None, None).to_dict(), indent=2, default=_json_default)
    text = _write(tmp_path)
    assert text == expected
    assert '"potential_energy_minimized": NaN' in text
    assert _write(tmp_path, nonfinite="keep") == expected


def test_null_writes_strict_json_and_lists_the_field(tmp_path):
    data = strict_loads(_write(tmp_path, nonfinite="null"))
    assert data["potential_energy_minimized"] is None
    assert data["potential_energy_md_avg"] == -12.5
    assert data["nonfinite_fields"] == ["potential_energy_minimized"]
    assert data["success"] is True


def test_null_changes_no_other_value(tmp_path):
    kept = json.loads(_write(tmp_path))
    nulled = strict_loads(_write(tmp_path, nonfinite="null"))
    kept["potential_energy_minimized"] = None
    kept["nonfinite_fields"] = ["potential_energy_minimized"]
    assert nulled == kept


def test_another_value_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match=r"keep.*null"):
        _write(tmp_path, nonfinite="drop")
    assert not (tmp_path / "relax.json").exists()


class TestCommand:
    @staticmethod
    def _main(monkeypatch, tmp_path, *extra):
        seen = {}

        def fake_run_one(relaxer, *args, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(success=True)

        monkeypatch.setattr(relaxation, "ImplicitRelaxation", lambda config: None)
        monkeypatch.setattr(relaxation, "_run_one", fake_run_one)
        argv = ["binding-metrics-relax", "-i", str(tmp_path / "in.cif"), "-o", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        relaxation.main()
        return seen

    def test_the_default_is_keep(self, monkeypatch, tmp_path):
        assert self._main(monkeypatch, tmp_path)["nonfinite"] == "keep"

    def test_the_flag_reaches_the_writer(self, monkeypatch, tmp_path):
        seen = self._main(monkeypatch, tmp_path, "--nonfinite", "null", "--results-json", "r.json")
        assert seen["nonfinite"] == "null" and str(seen["results_json"]) == "r.json"

    def test_the_flag_is_read_from_a_config_file(self, monkeypatch, tmp_path):
        config = tmp_path / "relax.toml"
        config.write_text('nonfinite = "null"\n', encoding="utf-8")
        assert self._main(monkeypatch, tmp_path, "--config", str(config))["nonfinite"] == "null"

    def test_another_choice_is_a_usage_error(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as stop:
            self._main(monkeypatch, tmp_path, "--nonfinite", "drop")
        assert stop.value.code == 2
        assert "invalid choice: 'drop'" in capsys.readouterr().err
