"""configure_logging: where package log records go, and that it is safe to repeat."""

import io
import logging
import sys
import types

import pytest

from binding_metrics.utils import _CurrentStreamHandler, configure_logging


@pytest.fixture
def package_logger():
    """The package logger with our handlers removed again after the test."""
    logger = logging.getLogger("binding_metrics")
    saved_level = logger.level
    yield logger
    for name in ("binding_metrics", "__main__"):
        target = logging.getLogger(name)
        for handler in [h for h in target.handlers if isinstance(h, _CurrentStreamHandler)]:
            target.removeHandler(handler)
    logger.setLevel(saved_level)


def test_info_and_warning_go_to_stdout_error_to_stderr(package_logger, capsys):
    configure_logging()
    log = logging.getLogger("binding_metrics.some.module")
    log.info("progress line")
    log.warning("WARNING: careful")
    log.error("failed step")
    out, err = capsys.readouterr()
    assert out == "progress line\nWARNING: careful\n"
    assert err == "failed step\n"


def test_debug_is_hidden_at_the_default_level(package_logger, capsys):
    configure_logging()
    logging.getLogger("binding_metrics.x").debug("noise")
    assert capsys.readouterr() == ("", "")


def test_second_call_adds_no_handlers_and_updates_the_level(package_logger, capsys):
    configure_logging()
    n_handlers = len(package_logger.handlers)
    configure_logging(logging.DEBUG)
    assert len(package_logger.handlers) == n_handlers
    logging.getLogger("binding_metrics.x").debug("now visible")
    assert capsys.readouterr().out == "now visible\n"


def test_records_follow_a_swapped_stdout(package_logger, capsys, monkeypatch):
    """cli.log_to_file replaces sys.stdout after configure_logging ran."""
    configure_logging()
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stdout", replacement)
    logging.getLogger("binding_metrics.x").info("to the log file")
    assert replacement.getvalue() == "to the log file\n"


def test_other_loggers_are_left_alone(package_logger, capsys):
    configure_logging()
    logging.getLogger("some_other_library").warning("not ours")
    assert capsys.readouterr().out == ""


def test_module_run_with_dash_m_is_covered(package_logger, capsys, monkeypatch):
    """`python -m binding_metrics.x` logs under the name __main__, outside the package."""
    main_module = sys.modules["__main__"]
    monkeypatch.setattr(
        main_module,
        "__spec__",
        types.SimpleNamespace(name="binding_metrics.cli.run"),
        raising=False,
    )
    configure_logging()
    logging.getLogger("__main__").info("from a module run as a script")
    assert capsys.readouterr().out == "from a module run as a script\n"


def test_a_user_script_named_main_is_not_touched(package_logger, capsys, monkeypatch):
    main_module = sys.modules["__main__"]
    monkeypatch.setattr(
        main_module, "__spec__", types.SimpleNamespace(name="my_analysis"), raising=False
    )
    configure_logging()
    assert not any(
        isinstance(h, _CurrentStreamHandler) for h in logging.getLogger("__main__").handlers
    )
