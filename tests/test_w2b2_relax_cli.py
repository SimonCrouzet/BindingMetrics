"""``binding-metrics-relax`` command-line parsing and console output."""

import logging
import sys
from types import SimpleNamespace

import pytest

from binding_metrics.protocols import relaxation
from binding_metrics.utils import _CurrentStreamHandler


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


@pytest.fixture
def package_logging_restored():
    """Remove the console handlers ``main()`` installs, and restore the level."""
    package_logger = logging.getLogger("binding_metrics")
    saved_level = package_logger.level
    yield package_logger
    for name in ("binding_metrics", "__main__"):
        target = logging.getLogger(name)
        for handler in [h for h in target.handlers if isinstance(h, _CurrentStreamHandler)]:
            target.removeHandler(handler)
    package_logger.setLevel(saved_level)


class TestMainConfiguresLogging:
    def test_library_records_reach_stdout_with_the_former_text(
        self, relax_main, monkeypatch, capsys, package_logging_restored
    ):
        def run_one_that_logs(*args, **kwargs):
            library = logging.getLogger("binding_metrics.protocols.relaxation")
            library.info("  Platform: CPU")
            library.warning("  Warning: CUDA unavailable (probe failed), falling back to CPU.")
            return SimpleNamespace(success=True)

        monkeypatch.setattr(relaxation, "_run_one", run_one_that_logs)
        relax_main()
        assert capsys.readouterr().out == (
            "  Platform: CPU\n  Warning: CUDA unavailable (probe failed), falling back to CPU.\n"
        )

    def test_configuration_happens_before_argument_parsing(
        self, monkeypatch, package_logging_restored
    ):
        """A usage error exits before any relaxation, but logging is already set up."""
        monkeypatch.setattr(sys, "argv", ["binding-metrics-relax", "--bogus"])
        with pytest.raises(SystemExit):
            relaxation.main()
        assert any(isinstance(h, _CurrentStreamHandler) for h in package_logging_restored.handlers)
