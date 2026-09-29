"""Text files are read and written as UTF-8 whatever the locale says.

``Path.read_text()``, ``open()`` and ``subprocess.run(text=True)`` without ``encoding`` use the
locale. Under an ASCII locale (``LC_ALL=C`` with UTF-8 mode off) they fail on the first non-ASCII
character, as ``pyproject.toml`` has, and a force-field file written that way loses its
non-ASCII text. The guard scans every Python file of ``src/``, ``tests/``, ``scripts/`` and
``benchmarks/``; the behavioural tests run Python in an ASCII locale.
"""

import ast
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

#: Trees whose text I/O must name its encoding. ``src`` and ``tests`` must exist.
SCANNED_TREES = ("src", "tests", "scripts", "benchmarks")

#: Sites that may stay without an encoding, as ``"path/from/repo/root.py:what"`` -> reason.
#: A site belongs here only when the file it touches is not UTF-8 text. None so far.
ALLOWED_SITES: dict[str, str] = {}

ASCII_LOCALE = {
    "LC_ALL": "C",
    "LANG": "C",
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
}

#: Calls that open a file in text mode unless the mode says ``b``:
#: ``(owner, name)`` -> (position of ``mode``, position of ``encoding``).
_TEXT_UNLESS_BINARY = {
    (None, "open"): (1, 3),
    ("io", "open"): (1, 3),
    (None, "fdopen"): (1, 3),
    ("os", "fdopen"): (1, 3),
}

#: Calls that open a file in binary mode unless the mode names text (``t`` or no ``b``).
_BINARY_UNLESS_TEXT = {
    "NamedTemporaryFile": (0, 2),
    "TemporaryFile": (0, 2),
    "SpooledTemporaryFile": (1, 3),
}
_COMPRESSED_MODULES = ("gzip", "bz2", "lzma")
_FILE_HANDLERS = ("FileHandler", "WatchedFileHandler", "RotatingFileHandler")
_MODE_LETTERS = frozenset("rwxa+bt")


def _has_keyword(call: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in call.keywords)


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _names_encoding(call: ast.Call, position: int) -> bool:
    """True when ``encoding`` is given by keyword, by position, or may come from ``**options``."""
    return len(call.args) > position or any(
        keyword.arg in ("encoding", None) for keyword in call.keywords
    )


def _mode(call: ast.Call, position: int):
    """The ``mode`` argument: a string, ``""`` when absent, ``None`` when it is not a literal."""
    node = call.args[position] if len(call.args) > position else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            node = keyword.value
    if node is None:
        return ""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _call_without_encoding(call: ast.Call):
    """What kind of text-mode access ``call`` is when it names no encoding, else ``None``."""
    function = call.func
    if isinstance(function, ast.Attribute):
        # "?" marks an owner that is not a plain name (``Path(x).open()``); None is a bare call.
        name, owner = function.attr, getattr(function.value, "id", "?")
    else:
        name, owner = getattr(function, "id", ""), None

    if name in ("read_text", "write_text"):
        if not _names_encoding(call, 1 if name == "write_text" else 0):
            return name
    elif (owner, name) in _TEXT_UNLESS_BINARY:
        mode_position, encoding_position = _TEXT_UNLESS_BINARY[(owner, name)]
        mode = _mode(call, mode_position)
        is_binary = mode is not None and "b" in mode
        if not is_binary and not _names_encoding(call, encoding_position):
            return name
    elif name == "open" and owner in _COMPRESSED_MODULES:
        # gzip.open("f") is binary; only an explicit text mode ("rt", "wt") decodes.
        mode = _mode(call, 1)
        if mode and "t" in mode and not _names_encoding(call, 3):
            return f"{owner}.open"
    elif name == "open" and isinstance(function, ast.Attribute) and owner != "os":
        # path.open(...): text when no mode is given or the mode is a text-mode literal. A first
        # argument that is not a mode (Image.open(path), zipfile.open("a.txt")) is not this call.
        first = call.args[0] if call.args else None
        no_mode = first is None and not _has_keyword(call, "mode")
        mode = _mode(call, 0)
        is_mode_literal = bool(mode) and set(mode) <= _MODE_LETTERS and "b" not in mode
        if (no_mode or is_mode_literal) and not _names_encoding(call, 2):
            return ".open"
    elif name in _BINARY_UNLESS_TEXT:
        mode_position, encoding_position = _BINARY_UNLESS_TEXT[name]
        mode = _mode(call, mode_position)
        if mode and "b" not in mode and not _names_encoding(call, encoding_position):
            return name
    elif name in _FILE_HANDLERS:
        if not _names_encoding(call, 2):
            return name
    elif owner == "subprocess" and name in ("run", "check_output", "check_call", "call", "Popen"):
        text = any(
            keyword.arg in ("text", "universal_newlines") and _is_true(keyword.value)
            for keyword in call.keywords
        )
        if text and not _has_keyword(call, "encoding"):
            return f"subprocess.{name}(text=True)"
    return None


def text_io_calls(tree: ast.AST) -> list:
    """``(call, what)`` for each text-mode file access in ``tree`` that names no encoding."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (what := _call_without_encoding(node)):
            found.append((node, what))
    return found


def text_io_without_encoding(source: str) -> list:
    """``(line, what)`` for each text-mode file access in ``source`` that names no encoding."""
    return [(call.lineno, what) for call, what in text_io_calls(ast.parse(source))]


class TestTheScanner:
    """The guard finds what it claims to find."""

    @pytest.mark.parametrize(
        "snippet",
        [
            "path.read_text()",
            "path.write_text('x')",
            "open('f')",
            "open('f', 'w')",
            "open('f', 'w', newline='')",
            "io.open('f')",
            "os.fdopen(fd, 'w')",
            "path.open()",
            "path.open('w')",
            "Path(name).open(mode='r')",
            "tempfile.NamedTemporaryFile(mode='w', suffix='.cif')",
            "tempfile.TemporaryFile('w+')",
            "tempfile.SpooledTemporaryFile(1024, 'w+')",
            "gzip.open(path, 'rt')",
            "logging.FileHandler(path)",
            "subprocess.run(['x'], text=True)",
            "subprocess.run(['x'], capture_output=True, universal_newlines=True)",
            "subprocess.check_output(['x'], text=True)",
            "subprocess.Popen(['x'], text=True)",
        ],
    )
    def test_flags_a_missing_encoding(self, snippet):
        assert text_io_without_encoding(snippet)

    @pytest.mark.parametrize(
        "snippet",
        [
            "path.read_text(encoding='utf-8')",
            "path.read_text('utf-8')",
            "path.write_text('x', encoding='utf-8')",
            "path.write_text('x', 'utf-8')",
            "open('f', encoding='utf-8')",
            "open('f', 'r', -1, 'utf-8')",
            "open('f', 'rb')",
            "open('f', mode='wb')",
            "os.fdopen(fd, 'wb')",
            "os.fdopen(fd, 'w', encoding='utf-8')",
            "path.open('wb')",
            "path.open(encoding='utf-8')",
            "os.open('a', os.O_RDONLY)",
            "Image.open(path)",
            "archive.open('a.txt')",
            "tempfile.NamedTemporaryFile(suffix='.json', delete=False)",
            "tempfile.NamedTemporaryFile(mode='w+b')",
            "tempfile.NamedTemporaryFile(mode='w', encoding='utf-8')",
            "gzip.open(path)",
            "gzip.open(path, 'rt', encoding='utf-8')",
            "logging.FileHandler(path, encoding='utf-8')",
            "subprocess.run(['x'], encoding='utf-8')",
            "subprocess.run(['x'], text=True, encoding='utf-8')",
            "subprocess.run(['x'], capture_output=True)",
        ],
    )
    def test_accepts_an_explicit_encoding_or_binary_mode(self, snippet):
        assert not text_io_without_encoding(snippet)

    def test_reports_the_line_of_the_call(self):
        source = "x = 1\n\nwith open('f') as handle:\n    pass\n"
        assert text_io_without_encoding(source) == [(3, "open")]


def _python_files(directory: Path) -> list:
    return sorted(
        path
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts and ".egg-info" not in str(path)
    )


def _sites_without_encoding(directory: Path, root: Path = ROOT) -> list:
    """``path:line what`` for every unallowed text-mode file access under ``directory``."""
    found = []
    for path in _python_files(directory):
        relative = path.relative_to(root).as_posix()
        for line, what in text_io_without_encoding(path.read_text(encoding="utf-8")):
            if f"{relative}:{what}" not in ALLOWED_SITES:
                found.append(f"{relative}:{line} {what}")
    return found


@pytest.mark.parametrize("tree", SCANNED_TREES)
def test_text_io_names_its_encoding(tree):
    """No source, test, script or benchmark opens a text file with the locale's encoding."""
    directory = ROOT / tree
    if not directory.is_dir() and tree not in ("src", "tests"):
        pytest.skip(f"{tree}/ is not in this checkout")
    assert _python_files(directory), f"no Python file found under {directory}"
    found = _sites_without_encoding(directory)
    assert found == [], "add encoding='utf-8' to:\n" + "\n".join(found)


def test_a_new_site_without_an_encoding_fails_the_scan(tmp_path):
    (tmp_path / "clean.py").write_text("open('f', encoding='utf-8')\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("x = 1\nopen('f')\n", encoding="utf-8")
    assert _sites_without_encoding(tmp_path, root=tmp_path) == ["new.py:2 open"]


def _python_in_an_ascii_locale(code: str, extra_env=None) -> subprocess.CompletedProcess:
    env = {**os.environ, **ASCII_LOCALE, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


class TestInAnAsciiLocale:
    def test_the_locale_is_ascii(self):
        result = _python_in_an_ascii_locale("import locale; print(locale.getencoding())")
        assert result.stdout.strip() == "ANSI_X3.4-1968"

    def test_a_force_field_file_keeps_its_non_ascii_text(self):
        result = _python_in_an_ascii_locale(
            """
            from binding_metrics.core import gaff_ncaa

            class ForceField:
                def loadFile(self, path):
                    with open(path, "rb") as handle:
                        self.text = handle.read().decode("utf-8")

            ff = ForceField()
            gaff_ncaa._load_ffxml(ff, "<ForceField><!-- caf\\u00e9 --></ForceField>")
            assert "caf\\u00e9" in ff.text, ff.text
            """
        )
        assert result.returncode == 0, result.stderr

    def test_antechamber_output_with_non_ascii_text_is_reported(self, tmp_path):
        fake = tmp_path / "antechamber"
        fake.write_bytes(b"#!/bin/sh\nprintf 'caf\\303\\251 failed\\n' >&2\nexit 1\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        result = _python_in_an_ascii_locale(
            f"""
            from binding_metrics.core import gaff_ncaa

            try:
                gaff_ncaa._run_antechamber(["-h"], {str(tmp_path)!r})
            except RuntimeError as error:
                assert "caf\\u00e9 failed" in str(error), str(error)
            else:
                raise SystemExit("no RuntimeError")
            """,
            extra_env={"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        )
        assert result.returncode == 0, result.stderr
