"""Tests find the bundled examples from their own location, not from the working directory.

``Path("data/example_linear_p53_1YCR.pdb")`` is right only when pytest runs from the repository
root. From any other directory the file is missing, and a test that guards on ``.exists()``
skips: the run is green with the checks missing (#149). The examples are reached through
``conftest.DATA_DIR``. The guard parses every test file and rejects a path into ``data/`` that
is built from a literal, which is the form that depends on the working directory.

A scan of the source is used instead of a run from a foreign directory: it takes milliseconds,
needs no subprocess or imports, and names the file and line to change.
"""

import ast
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pytest
from conftest import DATA_DIR

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent

#: Calls that turn their first argument into a path or open it. The name is the last part of the
#: callee, so ``Path``, ``pathlib.Path``, ``os.path.join`` and ``os.path.isfile`` all match.
PATH_CALLS = frozenset(
    {"Path", "PurePath", "open", "join", "joinpath", "exists", "isfile", "isdir"}
)

#: Cheap test before parsing: an opening parenthesis followed by a string that starts with data.
OPENS_DATA_PATH = re.compile(r"""\(\s*[fF]?["'](?:\.{1,2}/)*data(?:["']|/)""")


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _leading_text(node: ast.expr) -> str | None:
    """The literal text an argument starts with: ``"data/x"``, ``f"data/{n}"``, ``"data" + s``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        return _leading_text(node.values[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _leading_text(node.left)
    return None


def _enters_data_directory(text: str) -> bool:
    """True for ``data``, ``data/x``, ``./data/x`` and ``../data/x``: a path from the cwd."""
    parts = [part for part in PurePosixPath(text).parts if part != ".."]
    return bool(parts) and parts[0] == "data" and not PurePosixPath(text).is_absolute()


def relative_data_paths(source: str) -> list[int]:
    """Line numbers where the source builds a path into ``data/`` from a string literal."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        if _callee_name(node.func) not in PATH_CALLS:
            continue
        text = _leading_text(node.args[0])
        if text is not None and _enters_data_directory(text):
            lines.append(node.lineno)
    return sorted(lines)


class TestDataDir:
    def test_points_at_the_bundled_examples(self):
        assert DATA_DIR.is_absolute()
        assert DATA_DIR == REPO / "data"
        assert (DATA_DIR / "example_linear_p53_1YCR.pdb").is_file()


class TestScanner:
    @pytest.mark.parametrize(
        "source",
        [
            'Path("data/example_linear_p53_1YCR.pdb")',
            'Path("data")',
            'Path("./data/x.cif")',
            'Path("../data/x.cif")',
            'pathlib.Path("data/x.cif")',
            'PurePath(f"data/{name}")',
            'Path("data" + "/x.cif")',
            'open("data/x.pdb", encoding="utf-8")',
            'os.path.join("data", "x.pdb")',
            'os.path.isfile("data/x.pdb")',
            'base.joinpath("data/x.pdb")',
        ],
    )
    def test_flags_a_path_from_the_working_directory(self, source):
        assert relative_data_paths(source) == [1]
        assert OPENS_DATA_PATH.search(source), "the test before parsing would skip this file"

    @pytest.mark.parametrize(
        "source",
        [
            'Path(__file__).parent.parent / "data" / "x.pdb"',
            'Path(__file__).resolve().parents[1] / "data/x.pdb"',
            'DATA_DIR / "x.pdb"',
            'tmp_path / "data"',
            'os.path.join(str(ROOT), "data")',
            'Path("database.txt")',
            'Path("/data/run")',
            'Path("out/data/x.pdb")',
            'f"data/{name}" in listed',
            '"data/x.pdb" in text',
        ],
    )
    def test_accepts_a_path_that_does_not_depend_on_the_working_directory(self, source):
        assert relative_data_paths(source) == []

    def test_reports_the_line(self):
        source = 'import os\n\nA = 1\nB = open("data/x.pdb", encoding="utf-8")\n'
        assert relative_data_paths(source) == [4]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_test_builds_a_path_into_data_from_the_working_directory():
    files = sorted(TESTS_DIR.rglob("*.py"))
    assert Path(__file__).resolve() in files
    # Reading some 290 files from a slow mount is the cost, so it is done in threads, and only a
    # file with a string that starts with ``data`` right after a parenthesis is parsed.
    with ThreadPoolExecutor() as pool:
        sources = list(pool.map(_read, files))
    offenders = [
        f"{path.relative_to(REPO)}:{line}"
        for path, source in zip(files, sources)
        if OPENS_DATA_PATH.search(source)
        for line in relative_data_paths(source)
    ]
    assert not offenders, (
        "a path into data/ built from a literal depends on the directory pytest runs from; "
        "use DATA_DIR from conftest (from conftest import DATA_DIR): " + ", ".join(offenders)
    )
