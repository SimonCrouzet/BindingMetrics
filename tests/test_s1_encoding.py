"""Text files are read and written as UTF-8 whatever the locale says.

``Path.read_text()``, ``open()`` and ``subprocess.run(text=True)`` without ``encoding`` use the
locale. Under an ASCII locale (``LC_ALL=C`` with UTF-8 mode off) they fail on the first non-ASCII
character, as ``pyproject.toml`` has, and a force-field file written that way loses its
non-ASCII text. The guard scans the files of this lane; the behavioural tests run Python in an
ASCII locale.
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
TESTS = Path(__file__).parent

#: Files whose text I/O must name its encoding.
CHECKED_FILES = sorted(
    {ROOT / "src" / "binding_metrics" / "core" / "gaff_ncaa.py"}
    | set(TESTS.glob("test_i2_*.py"))
    | set(TESTS.glob("test_s1_*.py"))
    | set(TESTS.glob("test_l10_*.py"))
    | {TESTS / "test_gaff_ncaa.py", TESTS / "test_l5_gaff_charge.py"}
)

ASCII_LOCALE = {
    "LC_ALL": "C",
    "LANG": "C",
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
}


def _has_keyword(call: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in call.keywords)


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def text_io_without_encoding(source: str) -> list:
    """``(line, what)`` for each text-mode file access in ``source`` that names no encoding."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
        owner = getattr(function.value, "id", None) if isinstance(function, ast.Attribute) else None
        if name in ("read_text", "write_text"):
            encoding_position = 1 if name == "write_text" else 0
            if not _has_keyword(node, "encoding") and len(node.args) <= encoding_position:
                found.append((node.lineno, name))
        elif name in ("open", "fdopen") and owner in (None, "os", "io"):
            mode_position = 1
            mode = node.args[mode_position] if len(node.args) > mode_position else None
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    mode = keyword.value
            is_binary = isinstance(mode, ast.Constant) and "b" in str(mode.value)
            if not is_binary and not _has_keyword(node, "encoding"):
                found.append((node.lineno, name))
        elif owner == "subprocess" and name in ("run", "check_output", "Popen"):
            text = any(
                k.arg in ("text", "universal_newlines") and _is_true(k.value) for k in node.keywords
            )
            if text and not _has_keyword(node, "encoding"):
                found.append((node.lineno, f"subprocess.{name}(text=True)"))
    return found


class TestTheScanner:
    """The guard finds what it claims to find."""

    @pytest.mark.parametrize(
        "snippet",
        [
            "path.read_text()",
            "path.write_text('x')",
            "open('f')",
            "open('f', 'w')",
            "os.fdopen(fd, 'w')",
            "subprocess.run(['x'], text=True)",
        ],
    )
    def test_flags_a_missing_encoding(self, snippet):
        assert text_io_without_encoding(snippet)

    @pytest.mark.parametrize(
        "snippet",
        [
            "path.read_text(encoding='utf-8')",
            "path.write_text('x', encoding='utf-8')",
            "open('f', encoding='utf-8')",
            "open('f', 'rb')",
            "os.fdopen(fd, 'wb')",
            "subprocess.run(['x'], encoding='utf-8')",
            "subprocess.run(['x'], capture_output=True)",
        ],
    )
    def test_accepts_an_explicit_encoding_or_binary_mode(self, snippet):
        assert not text_io_without_encoding(snippet)


@pytest.mark.parametrize("path", CHECKED_FILES, ids=lambda p: p.name)
def test_text_io_names_its_encoding(path):
    assert text_io_without_encoding(path.read_text(encoding="utf-8")) == []


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
