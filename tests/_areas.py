"""Areas of the code that each test exercises, derived from the imports of its file.

The suite has thousands of tests and takes 20 minutes, so a change to one module should not
mean running all of them. Every test is therefore tagged with one pytest marker per *area* (for
instance ``poses`` or ``cli``), and ``pytest -m poses`` runs the tests that touch that area.
The tag is derived, never written by hand: the source of the test file is parsed with ``ast``
(nothing is executed) and each ``binding_metrics`` module it imports is mapped to an area by
``AREA_OF_MODULE``. A test file carries the union of the areas of its imports.

What counts as an import of the package, in the file itself or in the test helpers it imports:

- ``import binding_metrics.x``, ``from binding_metrics.x import y``, ``from binding_metrics import
  x`` and a name re-exported by a package ``__init__`` (``from binding_metrics import
  compute_ramachandran`` resolves to ``binding_metrics.metrics.geometry`` through ``_EXPORTS``);
- the same inside a function or a fixture, since the whole tree is walked;
- a string that is a dotted module path (``"binding_metrics.core.system.prep_structure"`` given to
  ``monkeypatch.setattr`` or ``python -m``), and a console-script name of ``pyproject.toml``;
- a test helper (``from tests.predictors.synth import X``, ``from .support import Y``), whose own
  imports are added;
- a fixture of ``tests/conftest.py`` that the file requests, with the imports of the fixture body.

``docs``, ``golden`` and ``misc`` are not modules of the package. ``docs`` marks a test that reads
a Markdown file, ``golden`` one that uses stored reference output, and ``misc`` one that imports
nothing from the package.

This module imports only the standard library, because ``scripts/pytest_for_diff.py`` imports it
to choose the tests for a diff with the same table and the same scan as the markers. Run
``python tests/_areas.py`` to print the partition without collecting anything.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "binding_metrics"
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# First-level module (or a dotted prefix that refines one) -> area. The longest prefix wins, so
# ``"metrics.openfold"`` overrides ``"metrics"`` for that one file. Every first-level module of
# the package needs its own row: tests/test_area_tags.py fails, and says where to add the row,
# when a new module has none. A refinement may only point at a module that exists.
AREA_OF_MODULE = {
    # core: what turns a structure into an OpenMM system (preparation, force fields, GAFF and
    # phosphorylated residues, cyclic and non-standard residues, simulation). Its tests need
    # OpenMM and GAFF tools and are the slow ones, so they are worth a run of their own.
    "core": "core",
    # io: reading and writing structures. Almost every other module imports it, so it is its own
    # area to make "only the I/O tests" a possible run.
    "io": "io",
    # metrics: the static and MD metrics, their registry and the confidence metrics of a
    # prediction. The largest area of the suite.
    "metrics": "metrics",
    # of3: the OpenFold3 route. Its run, templates and flags (metrics/openfold.py, _openfold_*)
    # and its adapter and runner (predictors/of3*.py) have dozens of test files of their own, with
    # docs and cyclic-peptide cases that a change elsewhere in metrics does not reach.
    "metrics.openfold": "of3",
    "metrics._openfold_cli": "of3",
    "metrics._openfold_run": "of3",
    "metrics._openfold_templates": "of3",
    "predictors.of3": "of3",
    "predictors.of3_runner": "of3",
    # predictors: model adapters, runners, the run-once prediction store and session.
    "predictors": "predictors",
    # poses: the pose layer (kinds, reference, frame, scoring, pose-set reports). The `testing`
    # package only builds synthetic pose sets, so it follows it.
    "poses": "poses",
    "testing": "poses",
    # protocols: relaxation, preparation, solvation, structural QC and the report.
    "protocols": "protocols",
    # cli: the binding-metrics-run, -batch and -check-env entry points and their shared flags.
    "cli": "cli",
    # preflight: what a model or metric accepts, and the check of an input against it, before
    # anything runs. Both modules sit at the top of the package, outside any subpackage.
    "capabilities": "preflight",
    "preflight_cli": "preflight",
    # package: the root __init__ (lazy exports, version), the constants importable without OpenMM,
    # the logging setup and the provenance of result files. Small, used by everything.
    "__init__": "package",
    "_constants": "package",
    "utils": "package",
    "provenance": "package",
}

# One line per area, copied into the ``markers`` list of pyproject.toml as "<area>: <line>".
AREA_DESCRIPTIONS = {
    "core": "system preparation, force fields, simulation, GAFF, cyclic and non-standard residues",
    "io": "reading and writing structures (load_structure, save_cif, chain detection)",
    "metrics": "the static and MD metrics, their registry and the prediction confidence metrics",
    "of3": "the OpenFold3 route: metrics/openfold.py with its helpers, the of3 adapter and runner",
    "predictors": "model adapters, runners, the prediction store and session (OpenFold3 aside)",
    "poses": "the pose layer (kinds, reference, scoring, pose-set reports) and the testing package",
    "protocols": "relaxation, preparation, solvation, structural QC and the report",
    "cli": "the binding-metrics-run, -batch and -check-env entry points and their flags",
    "preflight": "what a model or metric accepts and the check of an input before a run",
    "package": "the package root, constants, logging setup and provenance",
    "docs": "the text of docs/, README.md and CHANGELOG.md, checked against the code",
    "golden": "comparisons with stored reference output (tests/golden, OpenFold3 golden files)",
    "misc": "tests that import nothing from the package",
}
SPECIAL_AREAS = ("docs", "golden", "misc")
CODE_AREAS = tuple(sorted(set(AREA_OF_MODULE.values())))
ALL_AREAS = CODE_AREAS + SPECIAL_AREAS

_DOTTED = re.compile(rf"{PACKAGE}(\.[A-Za-z_]\w*)+(:\w+)?")
_MARKDOWN_PATH = re.compile(r"[\w./-]*\.md")
_CONSOLE_SCRIPT = re.compile(r"(binding-metrics-[\w-]+)(\s.*)?", re.DOTALL)


def area_of_module(module: str) -> str | None:
    """Area of a dotted module name below the package (``"core.system"``; ``""`` is the root).

    Returns None for a module whose first level has no row in ``AREA_OF_MODULE``.
    """
    parts = module.split(".") if module else ["__init__"]
    for end in range(len(parts), 0, -1):
        area = AREA_OF_MODULE.get(".".join(parts[:end]))
        if area is not None:
            return area
    return None


@dataclass(frozen=True)
class FileScan:
    """What one Python file imports, found without executing it."""

    modules: frozenset[str]  # package modules, dotted below the package ("" is the root)
    helpers: frozenset[Path]  # test helper files it imports
    names: frozenset[str]  # argument names and strings: candidates for fixture names
    reads_docs: bool


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def walk_runtime(tree: ast.AST) -> Iterator[ast.AST]:
    """Walk a tree like ``ast.walk`` but skip ``if TYPE_CHECKING:`` bodies: they never run."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
        else:
            stack.extend(ast.iter_child_nodes(node))


def _names_docs_folder(node: ast.AST) -> bool:
    """True for ``root / "docs"``, ``Path("docs")`` and ``root.joinpath("docs")``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        operands = [node.left, node.right]
    elif isinstance(node, ast.Call) and _called_name(node) in ("Path", "joinpath"):
        operands = node.args
    else:
        return False
    return any(isinstance(o, ast.Constant) and o.value == "docs" for o in operands)


def _called_name(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")


def _parse(path: Path) -> ast.Module | None:
    """Parse a Python file; None when it cannot be read or parsed (pytest reports that itself)."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
        return None


def _is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


class Project:
    """A checkout (the real one, or a temporary one in a test) and the scans of its files.

    Results are cached per instance, so a long run parses each file once.
    """

    def __init__(self, root: Path = REPO_ROOT):
        self.root = Path(root).resolve()
        self.tests_dir = self.root / "tests"
        self.package_dir = self.root / "src" / PACKAGE
        self._scans: dict[Path, FileScan] = {}
        self._exports: dict[str, dict[str, str]] = {}
        self._texts: dict[Path, str] = {}
        self._closures: dict[Path, tuple[frozenset[str], frozenset[Path]]] = {}
        self._modules: frozenset[str] | None = None
        self._packages: frozenset[str] = frozenset()
        self._first_level: list[str] = []
        self._test_python: frozenset[Path] | None = None
        self._graph: dict[str, frozenset[str]] | None = None
        self._fixtures: dict[str, frozenset[str]] | None = None
        self._scripts: dict[str, str] | None = None

    # ------------------------------------------------------------------ the package

    def _index_package(self) -> None:
        """List the modules once: stat calls are slow on a mounted drive."""
        modules, packages, first_level = set(), set(), set()
        for folder, subfolders, files in os.walk(self.package_dir):
            subfolders[:] = [d for d in subfolders if d != "__pycache__"]
            relative = Path(folder).relative_to(self.package_dir).parts
            for name in files:
                if not name.endswith(".py"):
                    continue
                parts = [*relative, name[: -len(".py")]]
                if parts[-1] == "__init__":
                    parts.pop()
                    packages.add(".".join(parts))
                modules.add(".".join(parts))
                first_level.add(parts[0] if parts else "__init__")
        # a subpackage is a first-level module of its own, whatever its files are called
        self._modules = frozenset(modules)
        self._packages = frozenset(packages)
        self._first_level = sorted(first_level)

    def module_names(self) -> frozenset[str]:
        """Every module and package below ``src/binding_metrics``, dotted (``""`` is the root)."""
        if self._modules is None:
            self._index_package()
        return self._modules

    def first_level_modules(self) -> list[str]:
        """Names a table row must exist for: the ``.py`` files and the subpackages at the top."""
        self.module_names()
        return self._first_level

    def module_of_source_path(self, relative_path: str) -> str | None:
        """Dotted module of a repository path such as ``src/binding_metrics/core/system.py``."""
        prefix = f"src/{PACKAGE}/"
        if not relative_path.startswith(prefix) or not relative_path.endswith(".py"):
            return None
        parts = relative_path[len(prefix) : -len(".py")].split("/")
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _deepest_module(self, dotted: str) -> str:
        """The longest prefix of ``dotted`` that is a module (``core.system.f`` -> core.system)."""
        parts = dotted.split(".") if dotted else []
        while parts and ".".join(parts) not in self.module_names():
            parts.pop()
        return ".".join(parts)

    def exports_of(self, package: str) -> dict[str, str]:
        """Public name -> defining module for the ``_EXPORTS`` table of a package ``__init__``."""
        if package not in self._exports:
            table: dict[str, str] = {}
            tree = _parse(self._path_of_module(package))
            for node in tree.body if tree else []:
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                    continue
                if not any(isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in node.targets):
                    continue
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                        module = str(value.value)
                        if module == PACKAGE or module.startswith(PACKAGE + "."):
                            table[str(key.value)] = module[len(PACKAGE) :].lstrip(".")
            self._exports[package] = table
        return self._exports[package]

    def resolve_from_import(self, base: str, names: Iterable[str]) -> set[str]:
        """Package modules that ``from binding_metrics.<base> import <names>`` loads.

        A name is a submodule (``from binding_metrics.metrics import geometry``), a name that
        the package re-exports lazily (``from binding_metrics import compute_ramachandran``), or
        something defined in the package ``__init__`` itself, in which case the package is it.
        """
        base = self._deepest_module(base)
        exports = self.exports_of(base)
        found = set()
        for name in names:
            child = f"{base}.{name}" if base else name
            if child in self.module_names():
                found.add(child)
            elif name in exports:
                found.add(exports[name])
            else:
                found.add(base)
        return found

    def package_imports(
        self, tree: ast.AST, current_module: str | None = None, *, strings: str = "all"
    ) -> tuple[set[str], list[ast.ImportFrom]]:
        """Package modules a tree imports, and its imports that are not of the package.

        ``current_module`` (dotted, for a file of the package) resolves relative imports; in a
        test file a relative import is a helper, so the caller leaves it None and gets the
        ``ImportFrom`` node back in the second item. ``strings`` says which strings also count
        when they are a dotted module path or a console-script name: ``"all"`` of them, only those
        that are the argument of an ``import_module`` call (``"calls"``), or ``"none"``.
        """
        modules: set[str] = set()
        others: list[ast.ImportFrom] = []
        for node in walk_runtime(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == PACKAGE or alias.name.startswith(PACKAGE + "."):
                        modules.add(self._deepest_module(alias.name[len(PACKAGE) :].lstrip(".")))
            elif isinstance(node, ast.ImportFrom):
                if node.level and current_module is not None:
                    base = self._relative_base(current_module, node.level, node.module)
                elif (
                    node.level == 0
                    and node.module
                    and (node.module == PACKAGE or node.module.startswith(PACKAGE + "."))
                ):
                    base = node.module[len(PACKAGE) :].lstrip(".")
                else:
                    others.append(node)
                    continue
                modules |= self.resolve_from_import(base, [a.name for a in node.names])
            elif strings == "all" and isinstance(node, ast.Constant):
                modules |= self._modules_named_by(node.value)
            elif strings == "calls" and isinstance(node, ast.Call):
                if _called_name(node) == "import_module" and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant):
                        modules |= self._modules_named_by(argument.value)
        return modules, others

    def _relative_base(self, current_module: str, level: int, module: str | None) -> str:
        """Dotted package a relative import of a package file starts from."""
        parts = current_module.split(".") if current_module else []
        if current_module and not self._is_package(current_module):
            parts.pop()
        if level > 1:
            parts = parts[: max(len(parts) - (level - 1), 0)]
        return ".".join(parts + (module.split(".") if module else []))

    def _path_of_module(self, module: str) -> Path:
        """File of a module of the package: its ``.py``, or the ``__init__.py`` of a package."""
        base = self.package_dir.joinpath(*module.split(".")) if module else self.package_dir
        return base / "__init__.py" if self._is_package(module) else base.with_suffix(".py")

    def _is_package(self, module: str) -> bool:
        self.module_names()
        return module in self._packages

    def _modules_named_by(self, text: object) -> set[str]:
        """Modules a string constant refers to: a dotted path or a console-script name."""
        if not isinstance(text, str):
            return set()
        if _DOTTED.fullmatch(text):
            return {self._deepest_module(text[len(PACKAGE) + 1 :].split(":")[0])}
        named = _CONSOLE_SCRIPT.fullmatch(text)
        scripts = self.console_scripts()
        return {scripts[named.group(1)]} if named and named.group(1) in scripts else set()

    def console_scripts(self) -> dict[str, str]:
        """Console-script name -> module (``binding-metrics-run`` -> ``cli.run``)."""
        if self._scripts is None:
            self._scripts = {}
            try:
                with (self.root / "pyproject.toml").open("rb") as handle:
                    entries = tomllib.load(handle).get("project", {}).get("scripts", {})
            except (OSError, tomllib.TOMLDecodeError):
                entries = {}
            for name, target in entries.items():
                module = target.split(":")[0]
                if module.startswith(PACKAGE + "."):
                    self._scripts[name] = self._deepest_module(module[len(PACKAGE) + 1 :])
        return self._scripts

    # ------------------------------------------------------- the dependencies inside src

    def source_graph(self) -> dict[str, frozenset[str]]:
        """Module -> modules it depends on when it runs, from its imports (lazy ones included).

        Importing ``a.b.c`` also imports ``a.b``, ``a`` and the root, so a module depends on the
        parent packages of everything it imports. ``if TYPE_CHECKING:`` imports are left out. A
        string counts only as the argument of ``import_module``: the ``import_path="pkg.mod:func"``
        strings of the metric registry and the ``_EXPORTS`` tables name modules that load on first
        use, and following them would make every metric a dependency of the whole CLI.
        """
        if self._graph is None:
            graph = {}
            for module in self.module_names():
                tree = _parse(self._path_of_module(module))
                imported = self.package_imports(tree, module, strings="calls")[0] if tree else set()
                depends = set(imported)
                for name in imported:
                    parts = name.split(".") if name else []
                    depends |= {".".join(parts[:end]) for end in range(len(parts))}
                depends.discard(module)
                graph[module] = frozenset(depends)
            self._graph = graph
        return self._graph

    def dependents_closure(self, seeds: Iterable[str]) -> set[str]:
        """The seeds and every module that depends on one of them, directly or not."""
        reverse: dict[str, set[str]] = {}
        for module, dependencies in self.source_graph().items():
            for dependency in dependencies:
                reverse.setdefault(dependency, set()).add(module)
        closure, stack = set(), list(seeds)
        while stack:
            module = stack.pop()
            if module in closure:
                continue
            closure.add(module)
            stack.extend(reverse.get(module, ()))
        return closure

    def modules_of_source_file(self, relative_path: str) -> set[str]:
        """Modules a change to a file below ``src/binding_metrics`` can affect (before dependents).

        A ``.py`` file that exists is its module. A deleted or non-Python file (package data)
        stands for the package it sits in and everything below it.
        """
        module = self.module_of_source_path(relative_path)
        if module is not None and module in self.module_names():
            return {module}
        folder = Path(relative_path).parent.parts[2:]  # drop "src" and the package name
        package = self._deepest_module(".".join(folder))
        return {
            m
            for m in self.module_names()
            if not package or m == package or m.startswith(package + ".")
        }

    @staticmethod
    def areas_of_modules(modules: Iterable[str]) -> set[str]:
        """Areas of modules; a module without a table row stands for ``misc``."""
        return {area_of_module(m) or "misc" for m in modules}

    # ------------------------------------------------------------------ the test files

    def _python_files_of_tests(self) -> frozenset[Path]:
        if self._test_python is None:
            found = set()
            for folder, subfolders, files in os.walk(self.tests_dir):
                subfolders[:] = [d for d in subfolders if d != "__pycache__"]
                found |= {Path(folder, name) for name in files if name.endswith(".py")}
            self._test_python = frozenset(found)
        return self._test_python

    def test_files(self) -> list[Path]:
        return sorted(p for p in self._python_files_of_tests() if p.name.startswith("test_"))

    def _text(self, path: Path) -> str:
        if path not in self._texts:
            try:
                self._texts[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self._texts[path] = ""
        return self._texts[path]

    def fixture_modules(self) -> dict[str, frozenset[str]]:
        """Fixture of ``tests/conftest.py`` -> package modules its body imports."""
        if self._fixtures is None:
            self._fixtures = {}
            tree = _parse(self.tests_dir / "conftest.py")
            for node in ast.walk(tree) if tree else []:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_fixture(node):
                    self._fixtures[node.name] = frozenset(self.package_imports(node)[0])
        return self._fixtures

    def scan(self, path: Path) -> FileScan:
        """Imports of one test or helper file (cached)."""
        path = Path(os.path.abspath(path))
        if path not in self._scans:
            tree = _parse(path)
            modules: set[str] = set()
            helpers: set[Path] = set()
            names: set[str] = set()
            reads_docs = False
            if tree is not None:
                modules, others = self.package_imports(tree)
                for node in others:
                    helpers |= self._helper_files(path, node)
                for node in ast.walk(tree):
                    if isinstance(node, ast.arg):
                        names.add(node.arg)
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        names.add(node.value)
                        reads_docs |= bool(_MARKDOWN_PATH.fullmatch(node.value))
                    else:
                        reads_docs |= _names_docs_folder(node)
            self._scans[path] = FileScan(
                frozenset(modules), frozenset(helpers), frozenset(names), reads_docs
            )
        return self._scans[path]

    def _module_files(self, base: Path) -> set[Path]:
        """Files of the tests folder that ``base`` names as a module: ``base.py`` or a package."""
        candidates = {base.parent / f"{base.name}.py", base / "__init__.py"}
        return candidates & self._python_files_of_tests()

    def _helper_module(self, importer: Path, dotted: str) -> set[Path]:
        """Test files that a dotted module name of ``from x.y import`` stands for."""
        parts = dotted.split(".")
        if parts[0] == "conftest":
            return set()  # its fixtures are followed by name, see fixture_modules
        if parts[0] == "tests":
            parts = parts[1:]
        for root in (self.tests_dir, importer.parent):
            found = self._module_files(root.joinpath(*parts))
            if found:
                return found
        return set()

    def _helper_files(self, importer: Path, node: ast.ImportFrom) -> set[Path]:
        """Test helper files that an import which is not of the package refers to."""
        if node.level:
            base = importer.parent
            for _ in range(node.level - 1):
                base = base.parent
            module_path = base.joinpath(*(node.module.split(".") if node.module else []))
            found = self._module_files(module_path) if node.module else set()
            for alias in node.names:
                found |= self._module_files(module_path / alias.name)
            return found
        found = set()
        if node.module:
            found |= self._helper_module(importer, node.module)
            for alias in node.names:
                found |= self._helper_module(importer, f"{node.module}.{alias.name}")
        return found

    def closure(self, path: Path) -> tuple[frozenset[str], frozenset[Path]]:
        """Package modules a test file uses, and the files (itself and its helpers) behind them."""
        path = Path(os.path.abspath(path))
        if path not in self._closures:
            fixtures = self.fixture_modules()
            modules: set[str] = set()
            seen: set[Path] = set()
            stack = [path]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                scan = self.scan(current)
                modules |= scan.modules
                for fixture in scan.names & fixtures.keys():
                    modules |= fixtures[fixture]
                stack.extend(scan.helpers)
            self._closures[path] = (frozenset(modules), frozenset(seen))
        return self._closures[path]

    def areas_of_test_file(self, path: Path) -> frozenset[str]:
        """The areas a test file exercises: always at least one."""
        modules, files = self.closure(path)
        areas = {area_of_module(m) or "misc" for m in modules}
        if self.scan(path).reads_docs:
            areas.add("docs")
        if any(self.tests_dir / "golden" in f.parents or "golden" in f.stem for f in files):
            areas.add("golden")
        return frozenset(areas or {"misc"})

    def tests_using(self, helper: Path) -> list[Path]:
        """Test files that import ``helper`` (a test helper), directly or through another."""
        helper = Path(os.path.abspath(helper))
        return [t for t in self.test_files() if t != helper and helper in self.closure(t)[1]]

    def tests_mentioning(self, name: str) -> list[Path]:
        """Test files, with their helpers, whose text contains ``name`` (a file name)."""
        return [
            t for t in self.test_files() if any(name in self._text(f) for f in self.closure(t)[1])
        ]

    def partition(self) -> dict[str, list[Path]]:
        """Area -> test files that carry it."""
        by_area: dict[str, list[Path]] = {area: [] for area in ALL_AREAS}
        for path in self.test_files():
            for area in self.areas_of_test_file(path):
                by_area.setdefault(area, []).append(path)
        return by_area


DEFAULT_PROJECT = Project()


def areas_of_test_file(path: Path | str) -> frozenset[str]:
    """Areas of a test file of this checkout (the function the pytest hook calls)."""
    return DEFAULT_PROJECT.areas_of_test_file(Path(path))


def main(argv: list[str]) -> int:
    """Print the partition: files per area, or the files of the areas named on the command line."""
    partition = DEFAULT_PROJECT.partition()
    unknown = [a for a in argv if a not in partition]
    if unknown:
        print(f"unknown area: {', '.join(unknown)}; areas are {', '.join(ALL_AREAS)}")
        return 2
    if argv:
        for area in argv:
            for path in partition[area]:
                print(path.relative_to(REPO_ROOT).as_posix())
        return 0
    for area in ALL_AREAS:
        print(f"{area:<11}{len(partition[area]):>4} files  {AREA_DESCRIPTIONS[area]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
