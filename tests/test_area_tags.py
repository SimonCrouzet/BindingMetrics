"""Every test carries area markers, and the table that derives them stays complete.

``tests/conftest.py`` adds a fine and a coarse marker for each area of the code a test file is
about, from the imports and the names it uses (``tests/_areas.py``), so that ``pytest -m
poses_frame`` runs one part of the suite and ``-m poses`` a larger one. The checks here keep that
partition honest: no test without an area, every marker registered in ``pyproject.toml``
(``--strict-markers`` needs it), a row in the table for each module of the package, and a scan
that reads the imports the way the module docstring says.
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
    fine = set(_areas.AREA_OF_MODULE.values())
    assert fine | {_areas.coarse_of(a) for a in fine} | set(_areas.SPECIAL_AREAS) == set(
        _areas.ALL_AREAS
    )
    assert all(text and text == text.strip() for text in _areas.AREA_DESCRIPTIONS.values())


def test_a_fine_area_is_named_after_its_coarse_area():
    for area in _areas.FINE_AREAS:
        coarse = _areas.coarse_of(area)
        assert coarse in _areas.AREA_DESCRIPTIONS, f"{area}: no coarse area {coarse!r}"
        assert area == coarse or area.startswith(coarse + "_")
    assert 30 <= len(_areas.FINE_AREAS) <= 50, "keep the fine areas few enough to remember"


def test_every_test_file_carries_the_coarse_area_of_each_fine_one():
    project = _areas.DEFAULT_PROJECT
    for path in project.test_files():
        areas = project.areas_of_test_file(path)
        fine = areas & set(_areas.FINE_AREAS)
        assert {_areas.coarse_of(a) for a in fine} <= areas, path


def test_the_testing_doc_lists_every_area_with_its_description():
    text = (REPO / "docs" / "testing.md").read_text(encoding="utf-8")
    for area in _areas.COARSE_AREAS + _areas.SPECIAL_AREAS:
        description = _areas.AREA_DESCRIPTIONS[area]
        row = f"| `{area}` | {description[0].upper()}{description[1:]} |"
        assert row in text, f"docs/testing.md needs the table row {row} ..."
    for area in _areas.FINE_AREAS:
        assert f"`{area}`" in text, f"docs/testing.md does not list the area {area}"


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
        ("tests/poses/test_frame.py", {"poses_frame", "poses"}),
        ("tests/test_l7_cli_args.py", {"cli_env", "cli"}),
        ("tests/test_run.py", {"cli_run", "cli"}),
        ("tests/test_registry.py", {"metrics_registry", "metrics"}),
        ("tests/test_of3_runner.py", {"of3_run", "of3"}),
        ("tests/test_pre_capabilities.py", {"preflight_capabilities", "preflight"}),
        ("tests/test_feat_c_docs.py", {"docs"}),
        ("tests/poses/test_golden_pose_set.py", {"poses", "golden"}),
        ("tests/test_w3b_dependabot.py", {"misc"}),
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
    assert collected("poses_frame") == {files[0]}
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
        import importlib
        SPEC = "binding_metrics.metrics.geometry:compute_ramachandran"
        def load(spec):
            return importlib.import_module(spec.split(":")[0])
        """,
    "src/binding_metrics/metrics/loader.py": """
        import importlib
        importlib.import_module("binding_metrics.metrics.geometry")
        """,
    "src/binding_metrics/poses/__init__.py": "",
    "src/binding_metrics/poses/frame.py": "def frame(): ...\n",
    "src/binding_metrics/poses/report.py": "from . import frame\n",
    "src/binding_metrics/cli/__init__.py": "def add_flag(): ...\n",
    "src/binding_metrics/cli/run.py": """
        from binding_metrics.core import system
        from binding_metrics.metrics import registry
        """,
    "src/binding_metrics/newthing.py": "VALUE = 1\n",
    "scripts/tool.py": "from binding_metrics.core import system\n",
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


def with_coarse(*areas):
    """The fine areas given and their coarse areas, which come with them."""
    return set(areas) | {_areas.coarse_of(area) for area in areas}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import binding_metrics.core.system\n", with_coarse("core_system")),
        ("import binding_metrics.core.system as system\n", with_coarse("core_system")),
        ("from binding_metrics.poses.frame import frame\n", with_coarse("poses_frame")),
        ("from binding_metrics import poses\n", with_coarse("poses_pose")),
        ("from binding_metrics.metrics import geometry\n", with_coarse("metrics_geometry")),
        # a name that the package re-exports lazily belongs to the module that defines it
        ("from binding_metrics import compute_ramachandran\n", with_coarse("metrics_geometry")),
        (
            "from binding_metrics.metrics import compute_openfold_metrics\n",
            with_coarse("of3_run"),
        ),
        ("from binding_metrics.metrics import openfold\n", with_coarse("of3_run")),
        # an import inside a function counts
        (
            "def test_x():\n    from binding_metrics.io.structures import load_structure\n",
            with_coarse("io"),
        ),
        # the union over several imports, none of them used more than the other
        (
            "import binding_metrics.core.system\nimport binding_metrics.poses.frame\n",
            with_coarse("core_system", "poses_frame"),
        ),
        # a dotted path in a string, as monkeypatch.setattr and python -m take it
        ('PATH = "binding_metrics.core.system.prep_structure"\n', with_coarse("core_system")),
        ('ARGS = ["-m", "binding_metrics.cli.run"]\n', with_coarse("cli_run")),
        ('ARGS = ["binding-metrics-run", "--help"]\n', with_coarse("cli_run")),
        ('COMMAND = "binding-metrics-run --help"\n', with_coarse("cli_run")),
        # test helpers, absolute and relative, bring their own imports
        ("from tests.support import frame\n", with_coarse("poses_frame")),
        ("from tests import support\n", with_coarse("poses_frame")),
        # a fixture of conftest.py brings the imports of its body
        ("def test_x(prepped):\n    pass\n", with_coarse("core_system")),
        # an import that only a type checker reads is not a use
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            "    from binding_metrics.cli import run\n",
            {"misc"},
        ),
        ("import binding_metrics\n", with_coarse("package_exports")),
        # a module without a row in the table is still tagged, as misc
        ("import binding_metrics.newthing\n", {"misc"}),
        ("import json\n\ndef test_x():\n    assert json\n", {"misc"}),
        # code that a test runs from a string is not what the test is about
        ('CODE = """\nimport binding_metrics.core.system\nprint(1)\n"""\n', {"misc"}),
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
            with_coarse("io") | {"golden"},
        ),
    ],
)
def test_the_scan_reads_imports_the_way_the_docstring_says(project, source, expected):
    assert areas_of_source(project, source) == expected


USES_FOUR_TO_ONE = """
from binding_metrics.core.system import prep_structure
from binding_metrics.poses.frame import frame

def test_x():
    prep_structure(); prep_structure(); prep_structure(); prep_structure()
    frame()
"""
USES_THREE_TO_ONE = USES_FOUR_TO_ONE.replace("prep_structure(); ", "", 1)


def test_a_file_is_about_the_areas_that_hold_a_quarter_of_its_references(project):
    # one use in five is an incidental one: the test is about the system, not about the frame
    assert areas_of_source(project, USES_FOUR_TO_ONE) == with_coarse("core_system")
    # one use in four is not
    assert areas_of_source(project, USES_THREE_TO_ONE) == with_coarse("core_system", "poses_frame")


def test_the_area_with_most_references_always_counts(project):
    source = "from binding_metrics.core.system import prep_structure\n" + "\n".join(
        f"from binding_metrics.metrics import geometry as g{i}" for i in range(1)
    )
    source += "\nfor _ in range(2):\n    prep_structure()\n    prep_structure()\n"
    assert with_coarse("core_system") <= areas_of_source(project, source)


def test_a_relative_helper_in_a_subpackage_is_followed(project):
    source = "from .helpers import something\n"
    assert areas_of_source(project, source, directory="tests/poses") == with_coarse("poses_pose")


@pytest.fixture
def fresh(tmp_path):
    """The same small project, for a test that adds test files and reads them all back."""
    for name, text in PACKAGE_FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return _areas.Project(tmp_path)


def chosen_for(project, *seeds, **files):
    """Write the test files given as name=source, and say which ones a change to seeds selects."""
    for name, source in files.items():
        (project.root / "tests" / f"test_{name}.py").write_text(source, encoding="utf-8")
    return {
        path.name.removeprefix("test_").removesuffix(".py"): reason
        for path, reason in project.select_tests(seeds).items()
    }


def test_a_test_is_chosen_for_a_module_it_imports_or_depends_on(fresh):
    chosen = chosen_for(
        fresh,
        "core.system",
        direct="import binding_metrics.core.system\n",
        through="from binding_metrics.cli.run import main\n",
        unrelated="from binding_metrics.poses.frame import frame\n",
        nothing="import json\n",
    )
    assert chosen == {
        "direct": "imports core.system",
        "through": "imports cli.run -> core.system",
    }


def test_a_module_that_a_table_loads_by_name_selects_the_importers_of_the_loader(fresh):
    files = {
        "loader": "from binding_metrics.metrics import registry\n",
        "other": "from binding_metrics.poses.frame import frame\n",
    }
    chosen = chosen_for(fresh, "metrics.geometry", **files)
    assert chosen == {"loader": "imports metrics.registry, which loads metrics.geometry by name"}
    # a module that no table names has no such tests
    assert chosen_for(fresh, "poses.frame") == {"other": "imports poses.frame"}


def test_the_names_of_a_package_init_are_not_the_children_of_the_package(fresh):
    chosen = chosen_for(
        fresh,
        "cli",
        flag="from binding_metrics.cli import add_flag\n",
        child="from binding_metrics.cli.run import main\n",
    )
    assert chosen == {"flag": "imports cli"}


def test_a_test_that_reads_the_console_scripts_uses_every_script_module(fresh):
    source = 'import tomllib\nSCRIPTS = tomllib.load(open("pyproject.toml", "rb"))\n'
    assert chosen_for(fresh, "cli.run", reader=source) == {"reader": "imports cli.run"}


def test_a_test_that_runs_a_script_or_code_in_a_string_uses_what_they_import(fresh):
    script = 'import subprocess\nRUN = ["python", "tool.py"]\n'
    snippet = 'CODE = """\nimport binding_metrics.core.system\nprint(1)\n"""\n'
    chosen = chosen_for(fresh, "core.system", script=script, snippet=snippet)
    assert chosen == {"script": "imports core.system", "snippet": "imports core.system"}


def test_a_second_import_inside_a_function_does_not_extend_the_chain(fresh):
    package = fresh.package_dir
    (package / "lazy_a.py").write_text(
        "def use():\n    from binding_metrics.core import system\n", encoding="utf-8"
    )
    (package / "lazy_b.py").write_text(
        "def use():\n    from binding_metrics import lazy_a\n", encoding="utf-8"
    )
    (package / "eager_c.py").write_text("from binding_metrics import lazy_b\n", encoding="utf-8")
    fresh._modules = None  # the module list is read once: read it again
    reached = fresh.affected_modules({"core.system"})
    assert "lazy_a" in reached  # one import inside a function: the function may be called
    assert "lazy_b" not in reached and "eager_c" not in reached  # a second one is dropped
    assert "lazy_b" in fresh.affected_modules({"core.system"}, max_lazy=2)


def test_the_chain_of_a_dependent_is_read_back_to_the_seed(fresh):
    chain = fresh.affected_modules({"core.system"})
    assert chain == {"core.system": (0, None), "cli.run": (1, "core.system")}


def test_dependents_of_a_module_are_those_that_import_it(project):
    closure = project.dependents_closure({"core.system"})
    assert "cli.run" in closure
    assert closure == {"core.system", "cli.run"}
    # a call to import_module is a dependency, a path kept as data (the registry) is not
    closure = project.dependents_closure({"metrics.geometry"})
    assert "metrics.loader" in closure
    assert not closure & {"metrics.registry", "cli.run"}
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
    assert project.areas_of_modules(["core.system", "newthing", ""]) == {
        "core_system",
        "misc",
        "package_exports",
    }
