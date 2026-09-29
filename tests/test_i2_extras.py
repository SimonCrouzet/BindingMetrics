"""The optional-dependency extras of pyproject.toml match what the code imports."""

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "binding_metrics"


@pytest.fixture(scope="module")
def extras() -> dict:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not found")
    return tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]


def _distribution(requirement: str) -> str:
    """``"pandas>=1.5"`` -> ``"pandas"``."""
    for sign in "<>=!~[ ":
        requirement = requirement.split(sign)[0]
    return requirement


def _imported_top_level_names() -> set:
    """Top-level names of every ``import`` and ``from ... import`` under src/."""
    names: set = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


class TestReportExtra:
    def test_every_package_of_the_extra_is_imported_somewhere(self, extras):
        imported = _imported_top_level_names()
        unused = [d for d in map(_distribution, extras["report"]) if d not in imported]
        assert not unused, f"in the report extra but imported nowhere under src/: {unused}"

    def test_matplotlib_is_not_a_dependency_of_any_extra(self, extras):
        """Nothing imports it; the colour strings of protocols/plots.py are plain text."""
        assert "matplotlib" not in _imported_top_level_names()
        for name, requirements in extras.items():
            assert "matplotlib" not in map(_distribution, requirements), name

    def test_all_extra_carries_the_packages_of_the_report_extra(self, extras):
        report = set(map(_distribution, extras["report"]))
        assert report <= set(map(_distribution, extras["all"]))
