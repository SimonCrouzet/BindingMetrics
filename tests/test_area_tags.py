"""Every test carries area markers, and the table that derives them stays complete.

``tests/conftest.py`` adds one marker per area of the code a test file exercises, from the
imports of that file (``tests/_areas.py``), so that ``pytest -m poses`` runs one part of the
suite. The checks here keep that partition honest: no test without an area, every marker
registered in ``pyproject.toml`` (``--strict-markers`` needs it), a row in the table for each
module of the package, and a scan that reads the imports the way the module docstring says.
"""

import itertools
import subprocess
import sys
import textwrap
import tomllib

import _areas
import pytest

REPO = _areas.REPO_ROOT
ADDING_A_ROW = "add a row for it to AREA_OF_MODULE in tests/_areas.py"
PYTEST_OWN_MARKERS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "tryfirst",
    "trylast",
}
EARLIER_MARKERS = ("integration", "gpu", "slow", "openfold")
CASE_NUMBERS = itertools.count()


def registered_markers(config):
    return {line.split(":")[0].split("(")[0].strip() for line in config.getini("markers")}


def test_every_collected_test_has_an_area(request):
    untagged = [
        item.nodeid
        for item in request.session.items
        if not {mark.name for mark in item.iter_markers()} & set(_areas.ALL_AREAS)
    ]
    assert not untagged, f"tests without an area marker: {untagged[:5]}"


def test_every_test_file_has_an_area():
    project = _areas.DEFAULT_PROJECT
    assert project.test_files(), "no test file found"
    for path in project.test_files():
        areas = project.areas_of_test_file(path)
        assert areas and areas <= set(_areas.ALL_AREAS), f"{path}: {sorted(areas)}"


def test_every_marker_in_use_is_registered(request):
    registered = registered_markers(request.config)
    used = {mark.name for item in request.session.items for mark in item.iter_markers()}
    assert used - registered - PYTEST_OWN_MARKERS == set()


def test_pyproject_registers_every_area_and_keeps_the_earlier_markers():
    with (REPO / "pyproject.toml").open("rb") as handle:
        lines = tomllib.load(handle)["tool"]["pytest"]["ini_options"]["markers"]
    registered = {line.split(":")[0]: line for line in lines}
    for area in _areas.ALL_AREAS:
        assert registered.get(area) == f"{area}: {_areas.AREA_DESCRIPTIONS[area]}", (
            f"pyproject.toml [tool.pytest.ini_options] markers needs the line "
            f'"{area}: {_areas.AREA_DESCRIPTIONS.get(area, "<description>")}"'
        )
    for earlier in EARLIER_MARKERS:
        assert earlier in registered
        assert earlier not in _areas.ALL_AREAS, f"{earlier!r} is already a marker, not an area"
    assert not set(_areas.ALL_AREAS) & PYTEST_OWN_MARKERS


def test_every_area_is_described():
    assert set(_areas.AREA_DESCRIPTIONS) == set(_areas.ALL_AREAS)
    assert set(_areas.AREA_OF_MODULE.values()) | set(_areas.SPECIAL_AREAS) == set(_areas.ALL_AREAS)
    assert all(text and text == text.strip() for text in _areas.AREA_DESCRIPTIONS.values())


def test_the_table_covers_every_first_level_module_of_the_package():
    project = _areas.DEFAULT_PROJECT
    missing = [
        name for name in project.first_level_modules() if _areas.area_of_module(name) is None
    ]
    assert not missing, "".join(
        f"\nbinding_metrics.{name} has no area: {ADDING_A_ROW}" for name in missing
    )


def test_the_table_has_no_row_for_a_module_that_is_gone():
    project = _areas.DEFAULT_PROJECT
    existing = project.module_names() | {"__init__"} | set(project.first_level_modules())
    stale = [row for row in _areas.AREA_OF_MODULE if row not in existing]
    assert not stale, f"rows of AREA_OF_MODULE for modules that no longer exist: {stale}"


@pytest.mark.parametrize(
    ("test_file", "expected"),
    [
        ("tests/poses/test_frame.py", {"poses"}),
        ("tests/test_w3a_cli_help.py", {"cli"}),
        ("tests/test_registry.py", {"metrics"}),
        ("tests/test_of3_runner.py", {"of3"}),
        ("tests/test_pre_capabilities.py", {"preflight"}),
        ("tests/test_feat_c_docs.py", {"docs"}),
        ("tests/poses/test_golden_pose_set.py", {"poses", "golden"}),
        ("tests/test_s1_encoding.py", {"misc"}),
    ],
)
def test_known_files_carry_the_expected_areas(test_file, expected):
    areas = _areas.areas_of_test_file(REPO / test_file)
    assert expected <= areas, f"{test_file}: {sorted(areas)}"
    if expected == {"misc"}:
        assert areas == {"misc"}


def test_the_marker_selects_the_partition_from_the_command_line():
    files = ["tests/poses/test_frame.py", "tests/test_l7_cli_args.py"]
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]

    def collected(expression):
        done = subprocess.run(
            [*command, "-m", expression, *files],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
        return {line.split("::")[0] for line in done.stdout.splitlines() if "::" in line}

    assert collected("poses") == {files[0]}
    assert collected("cli or docs") == {files[1]}


# ---------------------------------------------------------------------------------------------
# The scan itself, on a small project written to a temporary directory.
# ---------------------------------------------------------------------------------------------

PACKAGE_FILES = {
    "src/binding_metrics/__init__.py": """
        from typing import TYPE_CHECKING
        _EXPORTS = {"compute_ramachandran": "binding_metrics.metrics.geometry"}
        if TYPE_CHECKING:
            from binding_metrics.cli.run import run_pipeline
            from binding_metrics.metrics.geometry import compute_ramachandran
        """,
    "src/binding_metrics/core/__init__.py": "from binding_metrics import _lazy_exports\n",
    "src/binding_metrics/core/system.py": "def prep_structure(): ...\n",
    "src/binding_metrics/io/__init__.py": "",
    "src/binding_metrics/io/structures.py": "def load_structure(): ...\n",
    "src/binding_metrics/metrics/__init__.py": """
        _EXPORTS = {"compute_openfold_metrics": "binding_metrics.metrics.openfold"}
        """,
    "src/binding_metrics/metrics/geometry.py": "from binding_metrics.io import structures\n",
    "src/binding_metrics/metrics/openfold.py": "def compute_openfold_metrics(): ...\n",
    "src/binding_metrics/metrics/registry.py": """
        SPEC = "binding_metrics.metrics.geometry:compute_ramachandran"
        """,
    "src/binding_metrics/poses/__init__.py": "",
    "src/binding_metrics/poses/frame.py": "def frame(): ...\n",
    "src/binding_metrics/poses/report.py": "from . import frame\n",
    "src/binding_metrics/cli/__init__.py": "",
    "src/binding_metrics/cli/run.py": """
        from binding_metrics.core import system
        from binding_metrics.metrics import registry
        """,
    "src/binding_metrics/newthing.py": "VALUE = 1\n",
    "pyproject.toml": """
        [project.scripts]
        binding-metrics-run = "binding_metrics.cli.run:main"
        """,
    "tests/__init__.py": "",
    "tests/conftest.py": """
        import pytest

        @pytest.fixture
        def prepped():
            from binding_metrics.core.system import prep_structure
            return prep_structure
        """,
    "tests/support.py": "from binding_metrics.poses.frame import frame\n",
    "tests/golden/__init__.py": "",
    "tests/golden/cases.py": "CASES = {}\n",
    "tests/poses/__init__.py": "",
    "tests/poses/helpers.py": "import binding_metrics.poses\n",
}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    for name, text in PACKAGE_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return _areas.Project(root)


def areas_of_source(project, source, directory="tests"):
    """Areas of a test file with this source (each call writes a new file: scans are cached)."""
    path = project.root / directory / f"test_case_{next(CASE_NUMBERS)}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return set(project.areas_of_test_file(path))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import binding_metrics.core.system\n", {"core"}),
        ("import binding_metrics.core.system as system\n", {"core"}),
        ("from binding_metrics.poses.frame import frame\n", {"poses"}),
        ("from binding_metrics import poses\n", {"poses"}),
        ("from binding_metrics.metrics import geometry\n", {"metrics"}),
        # a name that the package re-exports lazily belongs to the module that defines it
        ("from binding_metrics import compute_ramachandran\n", {"metrics"}),
        ("from binding_metrics.metrics import compute_openfold_metrics\n", {"of3"}),
        ("from binding_metrics.metrics import openfold\n", {"of3"}),
        # an import inside a function counts
        ("def test_x():\n    from binding_metrics.io.structures import load_structure\n", {"io"}),
        # the union over several imports
        (
            "import binding_metrics.core.system\nimport binding_metrics.poses.frame\n",
            {"core", "poses"},
        ),
        # a dotted path in a string, as monkeypatch.setattr and python -m take it
        ('PATH = "binding_metrics.core.system.prep_structure"\n', {"core"}),
        ('ARGS = ["-m", "binding_metrics.cli.run"]\n', {"cli"}),
        ('ARGS = ["binding-metrics-run", "--help"]\n', {"cli"}),
        ('COMMAND = "binding-metrics-run --help"\n', {"cli"}),
        # test helpers, absolute and relative, bring their own imports
        ("from tests.support import frame\n", {"poses"}),
        ("from tests import support\n", {"poses"}),
        # a fixture of conftest.py brings the imports of its body
        ("def test_x(prepped):\n    pass\n", {"core"}),
        # an import that only a type checker reads is not a use
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            "    from binding_metrics.cli import run\n",
            {"misc"},
        ),
        ("import binding_metrics\n", {"package"}),
        # a module without a row in the table is still tagged, as misc
        ("import binding_metrics.newthing\n", {"misc"}),
        ("import json\n\ndef test_x():\n    assert json\n", {"misc"}),
        ("def test_x(:\n", {"misc"}),
        # docs and golden are not modules
        ('from pathlib import Path\nTEXT = Path("docs") / "metrics.md"\n', {"docs"}),
        ('TEXT = open("README.md").read()\n', {"docs"}),
        ('FOLDER = ROOT / "docs"\n', {"docs"}),
        # the word alone is not a use of the docs folder
        ('KINDS = {"docs"}\n', {"misc"}),
        ("from tests.golden import cases\n", {"golden"}),
        (
            "from tests.golden import cases\nimport binding_metrics.io.structures\n",
            {"golden", "io"},
        ),
    ],
)
def test_the_scan_reads_imports_the_way_the_docstring_says(project, source, expected):
    assert areas_of_source(project, source) == expected


def test_a_relative_helper_in_a_subpackage_is_followed(project):
    source = "from .helpers import something\n"
    assert areas_of_source(project, source, directory="tests/poses") == {"poses"}


def test_dependents_of_a_module_are_those_that_import_it(project):
    closure = project.dependents_closure({"core.system"})
    assert "cli.run" in closure
    assert closure == {"core.system", "cli.run"}
    # the registry names the metric functions by string, so a metric change reaches it and the CLI
    closure = project.dependents_closure({"metrics.geometry"})
    assert {"metrics.registry", "cli.run"} <= closure
    # a relative import is a dependency, and a type-checking import in the root is not
    assert "poses.report" in project.dependents_closure({"poses.frame"})
    assert "" not in project.dependents_closure({"core.system"})


def test_a_source_file_maps_to_its_module_and_data_to_its_package(project):
    assert project.modules_of_source_file("src/binding_metrics/core/system.py") == {"core.system"}
    data = project.modules_of_source_file("src/binding_metrics/poses/data/frame.json")
    assert data == {"poses", "poses.frame", "poses.report"}
    assert project.modules_of_source_file("src/binding_metrics/core/gone.py") == {
        "core",
        "core.system",
    }
    assert project.areas_of_modules(["core.system", "newthing", ""]) == {"core", "misc", "package"}
