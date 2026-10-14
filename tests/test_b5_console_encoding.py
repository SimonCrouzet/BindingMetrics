"""The command-line tools run on a console that cannot encode every character they print.

Messages and help texts hold an em dash, an arrow, an Angstrom sign and Greek letters. When the
standard output is ASCII (``LC_ALL=C`` with the locale coercion switched off, or
``PYTHONIOENCODING=ascii``), ``print`` raised ``UnicodeEncodeError``: ``--help`` of eight tools
died, ``binding-metrics-batch`` exited 1 after a good run, and ``binding-metrics-run`` lost the
log lines that held such a character (#53).

``configure_logging`` is the one call every ``main()`` makes before it parses its arguments, so it
is where the console streams are made tolerant: a stream that cannot encode a character writes a
``?`` in its place. The first class checks that rule on streams built in the test. The second runs
every console script of ``pyproject.toml`` in a subprocess under an ASCII console, with ``--help``
and with a small real run, and expects exit status 0 and no traceback.
"""

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import DATA_DIR, HAS_CUDA

from binding_metrics.utils import _CurrentStreamHandler, configure_logging

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"

DASH_LINE = "a — b → c Å"
DASH_LINE_REPLACED = "a ? b ? c ?"


@pytest.fixture
def package_logger():
    """The package logger with the console handlers of the test removed afterwards."""
    logger = logging.getLogger("binding_metrics")
    saved_level = logger.level
    yield logger
    for handler in [h for h in logger.handlers if isinstance(h, _CurrentStreamHandler)]:
        logger.removeHandler(handler)
    logger.setLevel(saved_level)


def _console(monkeypatch, *, encoding: str, errors: str, stream: str = "stdout"):
    """Install a text stream with the given encoding in place of ``sys.<stream>``."""
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors=errors)
    monkeypatch.setattr(sys, stream, wrapper)
    return wrapper


def _written(wrapper) -> bytes:
    wrapper.flush()
    return wrapper.buffer.getvalue()


class TestConfigureLoggingMakesTheConsoleTolerant:
    @pytest.mark.parametrize("errors", ["strict", "surrogateescape"])
    def test_print_to_an_ascii_stdout_replaces_what_it_cannot_encode(
        self, monkeypatch, package_logger, errors
    ):
        stdout = _console(monkeypatch, encoding="ascii", errors=errors)
        configure_logging()
        print(DASH_LINE)
        assert _written(stdout) == (DASH_LINE_REPLACED + "\n").encode("ascii")

    def test_log_record_to_an_ascii_stdout_is_written_with_replacements(
        self, monkeypatch, package_logger
    ):
        stdout = _console(monkeypatch, encoding="ascii", errors="strict")
        configure_logging()
        logging.getLogger("binding_metrics.b5").info(DASH_LINE)
        assert _written(stdout) == (DASH_LINE_REPLACED + "\n").encode("ascii")

    def test_ascii_stderr_replaces_too(self, monkeypatch, package_logger):
        stderr = _console(monkeypatch, encoding="ascii", errors="strict", stream="stderr")
        configure_logging()
        logging.getLogger("binding_metrics.b5").error(DASH_LINE)
        assert _written(stderr) == (DASH_LINE_REPLACED + "\n").encode("ascii")

    def test_a_utf8_console_is_not_touched(self, monkeypatch, package_logger):
        stdout = _console(monkeypatch, encoding="utf-8", errors="surrogateescape")
        configure_logging()
        assert stdout.errors == "surrogateescape"
        print(DASH_LINE)
        assert _written(stdout) == (DASH_LINE + "\n").encode("utf-8")

    def test_a_console_that_already_escapes_keeps_its_handler(self, monkeypatch, package_logger):
        stderr = _console(monkeypatch, encoding="ascii", errors="backslashreplace", stream="stderr")
        configure_logging()
        assert stderr.errors == "backslashreplace"

    def test_a_stream_that_cannot_be_reconfigured_is_accepted(self, monkeypatch, package_logger):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        configure_logging()
        configure_logging()
        print(DASH_LINE)
        assert sys.stdout.getvalue() == DASH_LINE + "\n"


# ---------------------------------------------------------------------------
# Every console script, in a subprocess
# ---------------------------------------------------------------------------


def _console_scripts() -> dict[str, tuple[str, str]]:
    """The ``[project.scripts]`` table as ``{command: (module, function)}``."""
    with open(REPO / "pyproject.toml", "rb") as handle:
        table = tomllib.load(handle)["project"]["scripts"]
    return {name: tuple(target.split(":")) for name, target in table.items()}


CONSOLE_SCRIPTS = _console_scripts()

#: A console that encodes ASCII only, whatever the locale.
ASCII_CONSOLE = {"LC_ALL": "C", "PYTHONIOENCODING": "ascii"}
#: The C locale with Python's coercion to UTF-8 switched off: stdout is ASCII with surrogateescape.
C_LOCALE = {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}

_LOCALE_VARIABLES = ("PYTHONUTF8", "PYTHONIOENCODING", "PYTHONCOERCECLOCALE", "LANG", "LANGUAGE")


def _console_environment(console: dict[str, str]) -> dict[str, str]:
    """The environment of this process with its locale settings replaced by ``console``."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _LOCALE_VARIABLES and not key.startswith("LC_")
    }
    env.update(console)
    return env


#: ``binding-metrics-check-env`` has no ``--help`` and its checks load OpenMM and CUDA for ten
#: seconds; the stub replaces them with one check that prints through the tool's own helper.
_CHECK_ENV_STUB = """
from binding_metrics.cli import check_env
def stub_check():
    check_env._ok("CUDA platform active \\u2014 stub \\u2714")
    return True
check_env.CHECKS[:] = [("stub", stub_check)]
"""


def _run_console_script(
    command: str, args: list[str], console: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess:
    module, function = CONSOLE_SCRIPTS[command]
    prelude = _CHECK_ENV_STUB if command == "binding-metrics-check-env" else ""
    code = (
        f"import sys\n{prelude}\nfrom {module} import {function}\n"
        f"sys.argv[0] = {command!r}\nsys.exit({function}())\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=_console_environment(console),
        cwd=cwd,
        capture_output=True,
        timeout=300,
    )


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _assert_ran_cleanly(result: subprocess.CompletedProcess) -> None:
    stderr = _text(result.stderr)
    assert "UnicodeEncodeError" not in stderr, stderr[-1500:]
    assert "Logging error" not in stderr, stderr[-1500:]
    assert "Traceback" not in stderr, stderr[-1500:]
    assert result.returncode == 0, stderr[-1500:]


def _prepared_input(tmp_path: Path) -> Path:
    """The 1YCR complex after ``binding-metrics-prep``: the tools below refuse an unprepared one."""
    prepared = tmp_path / "prepared.pdb"
    prep = _run_console_script(
        "binding-metrics-prep",
        ["--input", str(EXAMPLE_1YCR), "--output", str(prepared)],
        {"PYTHONUTF8": "1"},
        tmp_path,
    )
    _assert_ran_cleanly(prep)
    return prepared


def _tiny_run_args(command: str, tmp_path: Path) -> list[str]:
    """Arguments of a small real run of ``command`` on the bundled 1YCR complex."""
    pdb = str(EXAMPLE_1YCR)
    out = tmp_path / "out"
    if command == "binding-metrics-run":
        return [
            *["--input", pdb, "--output-dir", str(out), "--skip-prep", "--skip-relax"],
            *["--metrics", "dockq"],
        ]
    if command == "binding-metrics-batch":
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        shutil.copy(EXAMPLE_1YCR, input_dir)
        return [
            *["--input-dir", str(input_dir), "--output-csv", str(tmp_path / "batch.csv")],
            *["--output-dir", str(out), "--skip-prep", "--skip-relax", "--metrics", "dockq"],
        ]
    if command == "binding-metrics-interface":
        return ["--input", pdb]
    if command == "binding-metrics-compare":
        return ["--initial", pdb, "--processed", pdb]
    if command == "binding-metrics-dockq":
        return ["--model", pdb, "--reference", pdb]
    if command == "binding-metrics-electrostatics":
        return ["--input", pdb]
    if command == "binding-metrics-geometry":
        return ["--input", pdb, "--metric", "ramachandran"]
    if command == "binding-metrics-prep":
        return ["--input", pdb, "--output", str(tmp_path / "prepared.pdb")]
    if command == "binding-metrics-solvate":
        return [
            "--input",
            str(_prepared_input(tmp_path)),
            "--output",
            str(tmp_path / "solvated.pdb"),
        ]
    if command == "binding-metrics-openfold":
        return [
            *["prepare-query", "--complex", pdb, "--receptor-chain", "A", "--binder-chain", "B"],
            *["--query-name", "q", "--output-dir", str(out), "--dummy-msa"],
            *["--openfold-cyclic", "off"],
        ]
    if command == "binding-metrics-report":
        results = tmp_path / "x_results.json"
        results.write_text(json.dumps({"sample_id": "x", "input": "x.pdb"}), encoding="utf-8")
        return ["--results", str(results), "--summary", "--output-dir", str(out)]
    if command == "binding-metrics-relax":
        prepared = str(_prepared_input(tmp_path))
        return ["--input", prepared, "--output-dir", str(out), "--md-duration-ps", "0"]
    if command == "binding-metrics-energy":
        return ["--input", str(_prepared_input(tmp_path)), "--modes", "raw"]
    if command == "binding-metrics-receptor-quality":
        return ["--input", str(_prepared_input(tmp_path))]
    if command == "binding-metrics-check-env":
        return []
    raise ValueError(f"no small run is defined for {command}")


#: Commands whose small run uses the GPU (their device defaults to cuda).
NEEDS_GPU = {
    "binding-metrics-relax",
    "binding-metrics-energy",
    "binding-metrics-receptor-quality",
}


def _tiny_run_cases():
    cases = []
    for command in sorted(CONSOLE_SCRIPTS):
        marks = []
        if command in NEEDS_GPU:
            marks = [pytest.mark.gpu, pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")]
        cases.append(pytest.param(command, ASCII_CONSOLE, id=command, marks=marks))
    # The C locale gives stdout a different error handler (surrogateescape): the two tools whose
    # messages were lost or whose exit status was wrong are checked under it as well.
    cases.append(pytest.param("binding-metrics-run", C_LOCALE, id="binding-metrics-run-c-locale"))
    cases.append(
        pytest.param("binding-metrics-batch", C_LOCALE, id="binding-metrics-batch-c-locale")
    )
    return cases


@pytest.mark.slow
class TestConsoleScriptsOnAnAsciiConsole:
    @pytest.mark.parametrize("command", sorted(CONSOLE_SCRIPTS))
    @pytest.mark.parametrize("console", [ASCII_CONSOLE, C_LOCALE], ids=["ascii-io", "c-locale"])
    def test_help_exits_with_status_0(self, command, console, tmp_path):
        if command == "binding-metrics-check-env" and console is C_LOCALE:
            pytest.skip("check-env has no --help: its stubbed run is the small run below")
        args = [] if command == "binding-metrics-check-env" else ["--help"]
        result = _run_console_script(command, args, console, tmp_path)
        _assert_ran_cleanly(result)
        if args:
            assert "usage:" in _text(result.stdout)

    @pytest.mark.parametrize(("command", "console"), _tiny_run_cases())
    def test_small_run_exits_with_status_0(self, command, console, tmp_path):
        args = _tiny_run_args(command, tmp_path)
        result = _run_console_script(command, args, console, tmp_path)
        _assert_ran_cleanly(result)
        assert _text(result.stdout).strip()

    def test_run_keeps_the_lines_that_hold_a_dash(self, tmp_path):
        args = _tiny_run_args("binding-metrics-run", tmp_path)
        result = _run_console_script("binding-metrics-run", args, C_LOCALE, tmp_path)
        _assert_ran_cleanly(result)
        stdout = _text(result.stdout)
        assert "[skip] Prep skipped ? using raw input." in stdout
        assert "[skip] Relaxation skipped ? using raw input for downstream steps." in stdout

    def test_batch_reports_its_summary_and_writes_the_csv(self, tmp_path):
        args = _tiny_run_args("binding-metrics-batch", tmp_path)
        result = _run_console_script("binding-metrics-batch", args, C_LOCALE, tmp_path)
        _assert_ran_cleanly(result)
        assert "DONE in" in _text(result.stdout)
        assert " ? 1 ok, 0 error(s)" in _text(result.stdout)
        assert (tmp_path / "batch.csv").stat().st_size > 0
