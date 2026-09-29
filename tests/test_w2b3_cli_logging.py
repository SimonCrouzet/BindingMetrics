"""Pipeline messages go through the package logger and read the same on the console.

``binding-metrics-run`` and the batch worker used to ``print`` their progress lines.
They now log through ``binding_metrics.cli.*`` loggers; a CLI ``main()`` installs the
console handlers (``configure_logging``), so the text on stdout/stderr and in a
``--log-file`` must stay byte-identical to the former ``print`` output.
"""

import logging
import re
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import log_to_file
from binding_metrics.cli import run as run_cli
from binding_metrics.utils import _CurrentStreamHandler, configure_logging

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

BAR = "=" * 60
HASHES = "#" * 60


@pytest.fixture
def package_logger():
    """The package logger with our handlers removed again after the test."""
    logger = logging.getLogger("binding_metrics")
    saved_level = logger.level
    yield logger
    for handler in [h for h in logger.handlers if isinstance(h, _CurrentStreamHandler)]:
        logger.removeHandler(handler)
    logger.setLevel(saved_level)


def _run_records(caplog):
    return [
        (r.levelno, r.getMessage()) for r in caplog.records if r.name == "binding_metrics.cli.run"
    ]


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

DOCKQ_STEP = f"\n{BAR}\n  Step: DockQ CAPRI accuracy (vs reference)\n{BAR}\n"
DOCKQ_WARNING = (
    "  [warning] DockQ requested but no reference structure was provided for this run; skipping.\n"
)
SKIP_LINES = (
    "\n  [skip] Prep skipped — using raw input.\n"
    "\n  [skip] Relaxation skipped — using raw input for downstream steps.\n"
)


def _skip_prep_and_relax(tmp_path, **kwargs):
    return run_cli.run_pipeline(
        EXAMPLE_1YCR,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"dockq"}),
        **kwargs,
    )


class TestRunPipelineMessages:
    def test_console_text_is_the_former_print_output(self, tmp_path, package_logger, capsys):
        configure_logging()
        _skip_prep_and_relax(tmp_path)
        out, err = capsys.readouterr()
        assert out.endswith(SKIP_LINES + DOCKQ_STEP + DOCKQ_WARNING)
        assert "[warning]" not in err

    def test_messages_reach_the_logger_with_their_severity(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="binding_metrics")
        _skip_prep_and_relax(tmp_path)
        records = _run_records(caplog)
        assert (logging.INFO, "\n  [skip] Prep skipped — using raw input.") in records
        assert (logging.INFO, DOCKQ_STEP.rstrip("\n")) in records
        assert (logging.WARNING, DOCKQ_WARNING.rstrip("\n")) in records

    def test_step_banner_is_a_single_record(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="binding_metrics")
        _skip_prep_and_relax(tmp_path)
        banners = [msg for _, msg in _run_records(caplog) if "Step:" in msg]
        assert banners == [DOCKQ_STEP.rstrip("\n")]

    def test_nothing_is_written_to_a_stream_unless_a_handler_is_installed(
        self, tmp_path, package_logger, capsys
    ):
        assert not any(isinstance(h, _CurrentStreamHandler) for h in package_logger.handlers)
        _skip_prep_and_relax(tmp_path)
        out, err = capsys.readouterr()
        assert "Prep skipped" not in out + err
        assert "DockQ requested" not in out + err


class _FakeRelaxationResult:
    """Just the attributes ``run_pipeline`` reads from a relaxation result."""

    def __init__(self, success: bool, structure: str | None = None):
        self.success = success
        self.error_message = None if success else "boom"
        self.md_final_structure_path = structure
        self.minimized_structure_path = structure

    def to_dict(self):
        return {"success": self.success}


def _fake_relaxer(result):
    class FakeRelaxation:
        def __init__(self, config):
            self.config = config

        def run(self, path, output_dir, sample_id=None):
            return result

    return FakeRelaxation


class TestRunPipelineRelaxationMessages:
    def _run(self, tmp_path, monkeypatch, result, **kwargs):
        from binding_metrics.protocols import relaxation

        monkeypatch.setattr(relaxation, "ImplicitRelaxation", _fake_relaxer(result))
        return run_cli.run_pipeline(
            EXAMPLE_1YCR, tmp_path, skip_prep=True, metrics=frozenset(), **kwargs
        )

    def test_failed_relaxation_is_a_warning_on_stdout(
        self, tmp_path, monkeypatch, package_logger, caplog, capsys
    ):
        caplog.set_level(logging.INFO, logger="binding_metrics")
        configure_logging()
        self._run(tmp_path, monkeypatch, _FakeRelaxationResult(success=False))
        out, err = capsys.readouterr()
        assert "\n[FAILED] Relaxation failed: boom\n" in out
        assert "  Continuing with prepped input for downstream steps...\n" in out
        assert err == ""
        levels = dict((msg, level) for level, msg in _run_records(caplog))
        assert levels["\n[FAILED] Relaxation failed: boom"] == logging.WARNING

    def test_relaxed_structure_path_is_reported(
        self, tmp_path, monkeypatch, package_logger, capsys
    ):
        configure_logging()
        result = _FakeRelaxationResult(success=True, structure=str(tmp_path / "relaxed.cif"))
        self._run(tmp_path, monkeypatch, result)
        assert f"\n  Relaxed structure: {tmp_path / 'relaxed.cif'}\n" in capsys.readouterr().out

    def test_cpu_md_warning_keeps_its_banner_text(
        self, tmp_path, monkeypatch, package_logger, caplog, capsys
    ):
        caplog.set_level(logging.INFO, logger="binding_metrics")
        configure_logging()
        self._run(
            tmp_path,
            monkeypatch,
            _FakeRelaxationResult(success=False),
            device="cpu",
            md_duration_ps=10.0,
        )
        expected = (
            "\n  *** WARNING: running MD on CPU is extremely slow and not recommended. ***\n"
            "  *** For production use, run on a CUDA-capable GPU (--device cuda).   ***\n"
            "  *** Use --md-duration-ps 0 to minimize only if GPU is unavailable.   ***\n\n"
        )
        assert expected in capsys.readouterr().out
        warnings = [msg for level, msg in _run_records(caplog) if level == logging.WARNING]
        assert any("running MD on CPU" in msg for msg in warnings)


# ---------------------------------------------------------------------------
# binding-metrics-run main(): console and --log-file
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        ["binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "out")]
        + ["--skip-prep", "--skip-relax", "--metrics", "dockq"]
        + list(extra),
    )
    run_cli.main()


def _normalise(text, tmp_path):
    return re.sub(r"DONE in [\d.]+s", "DONE in Ns", text.replace(str(tmp_path), "TMP"))


class TestRunMain:
    def test_main_installs_the_console_handlers(self, tmp_path, monkeypatch, package_logger):
        _run_main(monkeypatch, tmp_path)
        assert any(isinstance(h, _CurrentStreamHandler) for h in package_logger.handlers)

    def test_console_output_matches_the_print_era_text(
        self, tmp_path, monkeypatch, package_logger, capsys
    ):
        _run_main(monkeypatch, tmp_path)
        out, err = capsys.readouterr()
        text = _normalise(out, tmp_path)
        results = "TMP/out/example_linear_p53_1YCR_results.json"
        assert text.startswith(
            f"\n{HASHES}\n  binding-metrics-run: example_linear_p53_1YCR\n"
            f"  Input:  {EXAMPLE_1YCR}\n  Output: TMP/out\n{HASHES}\n"
        )
        assert text.endswith(
            SKIP_LINES
            + DOCKQ_STEP
            + DOCKQ_WARNING
            + f"\n{HASHES}\n  DONE in Ns\n  Results: {results}\n{HASHES}\n\n"
        )
        assert "[warning]" not in err

    def test_log_file_receives_the_same_text_as_the_console(
        self, tmp_path, monkeypatch, package_logger, capsys
    ):
        _run_main(monkeypatch, tmp_path / "console")
        console = _normalise(capsys.readouterr().out, tmp_path / "console")
        log = tmp_path / "logs" / "run.log"
        _run_main(monkeypatch, tmp_path / "file", "--log-file", str(log))
        assert capsys.readouterr() == ("", "")  # everything went to the file
        logged = _normalise(log.read_text(encoding="utf-8"), tmp_path / "file")
        # Same lines, except the header line that names the log file itself.
        header_log_line = f"  Log:    {log}\n"
        assert header_log_line in logged
        assert logged.replace(header_log_line, "") == console


class TestLogToFile:
    """``log_to_file`` swaps sys.stdout/sys.stderr; the handlers follow the swap."""

    def test_records_land_in_the_file_and_the_streams_come_back(
        self, tmp_path, package_logger, capsys
    ):
        configure_logging()
        log = logging.getLogger("binding_metrics.cli.run")
        path = tmp_path / "run.log"
        with log_to_file(path):
            log.info("progress line")
            log.warning("  [warning] careful")
            log.error("a failed step")
            print("printed in main")
        assert path.read_text(encoding="utf-8") == (
            "progress line\n  [warning] careful\na failed step\nprinted in main\n"
        )
        assert capsys.readouterr() == ("", "")
        log.info("after the context")
        assert capsys.readouterr().out == "after the context\n"

    def test_a_log_file_opened_in_append_mode_keeps_earlier_records(self, tmp_path, package_logger):
        configure_logging()
        log = logging.getLogger("binding_metrics.cli.batch")
        path = tmp_path / "shared.log"
        for sample in ("s1", "s2"):
            with log_to_file(path, mode="a" if sample != "s1" else "w"):
                log.info("worker %s", sample)
        assert path.read_text(encoding="utf-8") == "worker s1\nworker s2\n"


class TestRunAsModule:
    def test_python_dash_m_shows_the_progress_lines(self, tmp_path):
        """``python -m binding_metrics.cli.run`` runs the file as ``__main__``."""
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-m", "binding_metrics.cli.run"]
            + ["-i", str(EXAMPLE_1YCR), "-o", str(tmp_path)]
            + ["--skip-prep", "--skip-relax", "--metrics", "dockq"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        assert SKIP_LINES + DOCKQ_STEP + DOCKQ_WARNING in proc.stdout
