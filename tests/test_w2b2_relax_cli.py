"""``binding-metrics-relax`` command-line parsing and console output."""

import sys
from types import SimpleNamespace

import pytest

from binding_metrics.protocols import relaxation


class _RecordingRelaxer:
    """Stands in for ImplicitRelaxation: keeps the config main() built, runs nothing."""

    configs: list = []

    def __init__(self, config):
        type(self).configs.append(config)


@pytest.fixture
def relax_main(monkeypatch, tmp_path):
    """Run ``relaxation.main()`` with the given argv, without relaxing anything."""
    _RecordingRelaxer.configs = []
    monkeypatch.setattr(relaxation, "ImplicitRelaxation", _RecordingRelaxer)
    monkeypatch.setattr(
        relaxation, "_run_one", lambda *args, **kwargs: SimpleNamespace(success=True)
    )

    def run(*extra):
        argv = ["binding-metrics-relax", "-i", str(tmp_path / "in.cif"), "-o", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        relaxation.main()
        return _RecordingRelaxer.configs[-1]

    return run


class TestSmallMoleculesFlag:
    def test_default_is_auto(self, relax_main):
        assert relax_main().small_molecules == "auto"

    @pytest.mark.parametrize("value", ["auto", "AUTO", " Auto "])
    def test_auto_is_case_insensitive(self, relax_main, value):
        assert relax_main("--small-molecules", value).small_molecules == "auto"

    @pytest.mark.parametrize("value", ["none", "None", "NONE"])
    def test_none_disables_parameterisation(self, relax_main, value):
        assert relax_main("--small-molecules", value).small_molecules is None

    @pytest.mark.parametrize("value", ["aut", "yes", "CCO"])
    def test_other_strings_are_rejected(self, relax_main, capsys, value):
        with pytest.raises(SystemExit) as exc:
            relax_main("--small-molecules", value)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert f"argument --small-molecules: invalid value {value!r}" in err
        assert "expected one of auto, none" in err
