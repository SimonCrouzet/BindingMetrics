"""Areas of the code that each test file is about, derived from what it imports and uses.

The suite has thousands of tests and takes 20 minutes, so a change to one module should not
mean running all of them. Every test is therefore tagged with pytest markers, one per *area* of
the code the test file is about (``poses_frame``, ``cli_run``), and ``pytest -m poses_frame``
runs those tests. The tags are derived, never written by hand: the source of the test file is
parsed with ``ast`` (nothing is executed) and each ``binding_metrics`` module it refers to is
mapped to a fine area by ``AREA_OF_MODULE``.

An area is named ``<coarse>_<family>``. The coarse area, the part before the first underscore,
is a second marker on every test of the family, so ``pytest -m metrics`` runs all of the metrics
and ``pytest -m metrics_geometry`` one family. An area without a family (``io``) is its own
coarse area.

What counts as a reference to the package, in the file itself or in the test helpers it imports:

- a name imported with ``from binding_metrics.x import y``, ``from binding_metrics import x`` or
  ``import binding_metrics.x as x``, including a name that a package re-exports lazily
  (``from binding_metrics import compute_ramachandran`` resolves to
  ``binding_metrics.metrics.geometry`` through ``_EXPORTS``), and a dotted chain from
  ``binding_metrics`` in the code;
- the same inside a function or a fixture, since the whole tree is walked;
- a string that is a dotted module path (``"binding_metrics.core.system.prep_structure"`` given to
  ``monkeypatch.setattr`` or ``python -m``), and a console-script name of ``pyproject.toml``;
- a test helper (``from tests.predictors.synth import X``, ``from .support import Y``), whose own
  references are added;
- a fixture of ``tests/conftest.py`` that the file requests, with the imports of the fixture body.

A test file takes the fine areas that hold at least ``PRIMARY_SHARE`` of its references, and the
one with the most whatever its share, so a CLI test that also builds a structure is about the
CLI. The markers say what a test is about; the tests that a change can break are a wider set.

``docs``, ``golden`` and ``misc`` are not modules of the package. ``docs`` marks a test that reads
a Markdown file, ``golden`` one that uses stored reference output, and ``misc`` one that imports
nothing from the package.

This module imports only the standard library, because ``scripts/pytest_for_diff.py`` imports it
to choose the tests for a diff with the same table and the same scan as the markers. Run
``python tests/_areas.py`` to print the partition without collecting anything.
"""

from __future__ import annotations

import ast
import copy
import heapq
import os
import re
import sys
import textwrap
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

PACKAGE = "binding_metrics"
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Module (or a dotted prefix) -> fine area. The longest prefix wins, so ``"metrics.geometry"``
# overrides ``"metrics"`` for that file. A fine area is named ``<coarse>_<family>``; the part
# before the first underscore is its coarse area, and a test file carries both markers, so
# ``pytest -m metrics`` runs every metrics family and ``pytest -m metrics_geometry`` one of them.
# An area with no family (``io``) is its own coarse area. Every first-level module of the package
# needs its own row: tests/test_area_tags.py fails, and says where to add the row, when a new
# module has none. A row may only name a module that exists.
AREA_OF_MODULE = {
    # core: what turns a structure into an OpenMM system. The slowest tests of the suite sit here.
    "core": "core_system",  # system preparation, the force field set-up and the simulation
    "core.system": "core_system",
    "core.simulation": "core_system",
    "core.forcefields": "core_system",
    "core.gaff_ncaa": "core_gaff",  # parameters of non-standard residues and phosphates
    "core.phosaa": "core_gaff",
    "core.cyclic": "core_cyclic",  # head-to-tail, disulfide, lactam and staple closures
    "core.residues": "core_residues",  # residue name tables, D-amino acids, N-methylation
    "core.nonstandard": "core_residues",
    # io: reading and writing structures. Almost every other module imports it.
    "io": "io",
    # metrics: one area per family of metrics, so that a change to the SASA code does not run the
    # electrostatics tests. The registry and the shared helpers are hubs: most metric tests
    # import them, and their areas are large.
    "metrics": "metrics_common",
    "metrics._common": "metrics_common",
    "metrics.interface": "metrics_interface",  # interface residues and polar contacts
    "metrics.contacts": "metrics_interface",
    "metrics.polar_contacts": "metrics_interface",
    "metrics.sasa": "metrics_sasa",
    "metrics.energy": "metrics_energy",  # interaction energy and the Coulomb term
    "metrics.mlff_energy": "metrics_energy",
    "metrics.electrostatics": "metrics_energy",
    "metrics.geometry": "metrics_geometry",
    "metrics.comparison": "metrics_comparison",  # RMSD against a reference, DockQ
    "metrics.rmsd": "metrics_comparison",
    "metrics.dockq": "metrics_comparison",
    "metrics.evobind": "metrics_evobind",
    "metrics.receptor_quality": "metrics_receptor",
    "metrics.registry": "metrics_registry",
    "metrics.prediction": "predictors_base",  # it reads what the adapters return
    # of3: the OpenFold3 route has dozens of test files of its own, with docs and cyclic-peptide
    # cases that a change elsewhere in metrics does not reach.
    "metrics.openfold": "of3_run",
    "metrics._openfold_run": "of3_run",
    "predictors.of3_runner": "of3_run",
    "metrics._openfold_templates": "of3_templates",  # template modes and the command-line flags
    "metrics._openfold_cli": "of3_templates",
    "predictors.of3": "of3_adapter",
    # predictors: one area per model adapter, one for the store and runners, one for the contract.
    "predictors": "predictors_base",
    "predictors.af2": "predictors_af2",
    "predictors.af2_runner": "predictors_af2",
    "predictors.boltz2": "predictors_boltz2",
    "predictors.boltz2_runner": "predictors_boltz2",
    "predictors.protenix": "predictors_protenix",
    "predictors.protenix_runner": "predictors_protenix",
    "predictors.store": "predictors_store",  # the run-once store, session, manifest and runners
    "predictors.session": "predictors_store",
    "predictors.manifest": "predictors_store",
    "predictors.runners": "predictors_store",
    "predictors.weights": "predictors_store",
    "predictors._closures": "predictors_store",
    "predictors._msa": "predictors_store",
    # poses: the pose layer, split along its stages. The `testing` package only builds synthetic
    # pose sets, so it follows the pose kinds.
    "poses": "poses_pose",
    "poses.pose": "poses_pose",
    "poses.docked": "poses_pose",
    "poses.experimental": "poses_pose",
    "testing": "poses_pose",
    "poses.frame": "poses_frame",
    "poses.agreement": "poses_agreement",
    "poses.aggregate": "poses_agreement",
    "poses.symmetry": "poses_agreement",
    "poses.descriptors": "poses_descriptors",
    "poses.reference": "poses_descriptors",
    "poses.score": "poses_score",
    "poses.report": "poses_score",
    "poses.reconcile": "poses_score",
    # protocols: relaxation with preparation and solvation, structural QC, the report.
    "protocols": "protocols_relaxation",
    "protocols.qc": "protocols_qc",
    "protocols.report": "protocols_report",
    "protocols.plots": "protocols_report",
    # cli: one area per command, and one for the shared flags and the environment check.
    "cli": "cli_env",
    "cli.run": "cli_run",
    "cli.batch": "cli_batch",
    "cli.prediction": "cli_prediction",
    # preflight: what a model or metric accepts, and the check of an input against it, before
    # anything runs. Both modules sit at the top of the package, outside any subpackage.
    "capabilities": "preflight_capabilities",
    "preflight_cli": "preflight_cli",
    # package: the root __init__ (lazy exports, version), then the constants importable without
    # OpenMM, the logging setup and the provenance of result files.
    "__init__": "package_exports",
    "_constants": "package_support",
    "utils": "package_support",
    "provenance": "package_support",
}

# One line per area, copied into the ``markers`` list of pyproject.toml as "<area>: <line>".
AREA_DESCRIPTIONS = {
    # coarse areas
    "cli": "all command-line entry points: run, batch, check-env and the shared flags",
    "core": "system preparation, force fields, simulation, GAFF, cyclic and non-standard residues",
    "io": "reading and writing structures (load_structure, save_cif, chain detection)",
    "metrics": "the static and MD metrics and their registry",
    "of3": "the OpenFold3 route: its run, templates, flags, adapter and runner",
    "package": "the package root, constants, logging setup and provenance",
    "poses": "the pose layer (kinds, reference, scoring, pose-set reports) and the testing package",
    "predictors": "model adapters, runners, the prediction store and session (OpenFold3 aside)",
    "preflight": "what a model or metric accepts and the check of an input before a run",
    "protocols": "relaxation, preparation, solvation, structural QC and the report",
    # fine areas
    "cli_batch": "binding-metrics-batch (cli/batch.py)",
    "cli_env": "the shared command-line flags and binding-metrics-check-env (cli/__init__.py)",
    "cli_prediction": "the --predictor step of run and batch (cli/prediction.py)",
    "cli_run": "binding-metrics-run and the pipeline behind it (cli/run.py)",
    "core_cyclic": "head-to-tail, disulfide, lactam and staple closures (core/cyclic.py)",
    "core_gaff": "GAFF2 parameters of non-standard residues and phospho-residues",
    "core_residues": "residue name tables, D-amino acids, N-methylated residues",
    "core_system": "system preparation, force-field set-up and simulation (core/system.py)",
    "metrics_common": "helpers shared by the metrics: structure loader, chain aliases, OpenMM",
    "metrics_comparison": "RMSD against a reference and DockQ",
    "metrics_energy": "interaction energy, the Coulomb term and the MLFF energy",
    "metrics_evobind": "the EvoBind score and its adversarial check",
    "metrics_geometry": "Ramachandran, omega planarity, shape complementarity, void volume",
    "metrics_interface": "interface residues, contacts, hydrogen bonds and salt bridges",
    "metrics_receptor": "receptor quality and receptor drift",
    "metrics_registry": "the metric registry, its metadata and its documentation",
    "metrics_sasa": "buried and static solvent-accessible surface area",
    "of3_adapter": "the OpenFold3 output adapter (predictors/of3.py)",
    "of3_run": "running OpenFold3 and reading its result (metrics/openfold.py, the runner)",
    "of3_templates": "OpenFold3 template modes and their command-line flags",
    "poses_agreement": "agreement between poses, aggregation and symmetry",
    "poses_descriptors": "reference-site descriptors and the reference (poses/descriptors.py)",
    "poses_frame": "the common frame of a pose set (poses/frame.py)",
    "poses_pose": "the pose kinds and the synthetic pose sets",
    "poses_score": "scoring a pose set, its report and the reconciliation of two back-ends",
    "predictors_af2": "the AlphaFold2 and ColabFold adapter and runner",
    "predictors_base": "the adapter contract: base class, registry, record, confidence helpers",
    "predictors_boltz2": "the Boltz-2 adapter and runner",
    "predictors_protenix": "the Protenix adapter and runner",
    "predictors_store": "the run-once store, session, manifest, runners and weights",
    "preflight_capabilities": "the declared limits of steps and models and the pre-flight check",
    "preflight_cli": "the pre-flight check of the command-line pipeline (preflight_cli.py)",
    "protocols_qc": "structural QC of relaxed structures",
    "protocols_relaxation": "relaxation, preparation and solvation protocols",
    "protocols_report": "the report and its plots",
    "package_exports": "lazy exports and the version of the package root",
    "package_support": "constants, logging setup and provenance of result files",
    # not modules of the package
    "docs": "the text of docs/, README.md and CHANGELOG.md, checked against the code",
    "golden": "comparisons with stored reference output (tests/golden, OpenFold3 golden files)",
    "misc": "tests that import nothing from the package",
}
SPECIAL_AREAS = ("docs", "golden", "misc")


def coarse_of(area: str) -> str:
    """The coarse area of a fine one: the part before the first underscore."""
    return area.split("_")[0]


FINE_AREAS = tuple(sorted(set(AREA_OF_MODULE.values())))
COARSE_AREAS = tuple(sorted({coarse_of(a) for a in FINE_AREAS}))
CODE_AREAS = tuple(sorted(set(FINE_AREAS) | set(COARSE_AREAS)))
ALL_AREAS = CODE_AREAS + SPECIAL_AREAS

# A chain of imports counts for a change when at most this many of its links are imports inside a
# function (see ``Project.affected_modules``).
MAX_LAZY_IMPORTS = 1

# A fine area tags a test file when it holds this share of the file's references to the package.
PRIMARY_SHARE = 0.25
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
    usage: Mapping[str, int]  # module -> how many times the file refers to it
    through: frozenset[str]  # packages whose lazy attribute lookup the file goes through


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


class Closure(NamedTuple):
    """What a test file uses: package modules, the files that use them, and lazy lookups."""

    modules: frozenset[str]
    files: frozenset[Path]
    through: frozenset[str]


class _FunctionBodies(ast.NodeTransformer):
    """Replaces the body of every function by ``pass``, so that what is left runs on import."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def _drop_function_bodies(tree: ast.AST) -> ast.AST:
    return _FunctionBodies().visit(copy.deepcopy(tree))


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
        self._closures: dict[Path, Closure] = {}
        self._modules: frozenset[str] | None = None
        self._packages: frozenset[str] = frozenset()
        self._first_level: list[str] = []
        self._test_python: frozenset[Path] | None = None
        self._edges: dict[str, dict[str, int]] | None = None
        self._dispatch: tuple[set[str], set[str]] | None = None
        self._fixtures: dict[str, frozenset[str]] | None = None
        self._scripts: dict[str, str] | None = None
        self._scripts_dir: frozenset[str] | None = None

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
        self,
        tree: ast.AST,
        current_module: str | None = None,
        *,
        strings: str = "all",
        through: set[str] | None = None,
    ) -> tuple[set[str], list[ast.ImportFrom]]:
        """Package modules a tree imports, and its imports that are not of the package.

        ``current_module`` (dotted, for a file of the package) resolves relative imports; in a
        test file a relative import is a helper, so the caller leaves it None and gets the
        ``ImportFrom`` node back in the second item. ``strings`` says which strings also count
        when they are a dotted module path or a console-script name: ``"all"`` of them, only those
        that are the argument of an ``import_module`` call (``"calls"``), or ``"none"``. A name
        that a package re-exports lazily goes through the package's ``__getattr__`` (PEP 562):
        the package joins ``through`` when that set is given, and the modules otherwise.
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
                # a lazily re-exported name goes through the package's __getattr__ (PEP 562)
                package = self._deepest_module(base)
                if any(a.name in self.exports_of(package) for a in node.names):
                    (modules if through is None else through).add(package)
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

    @staticmethod
    def _parse_snippet(text: str) -> ast.Module | None:
        try:
            return ast.parse(textwrap.dedent(text))
        except (SyntaxError, ValueError):
            return None

    def _script_names(self) -> frozenset[str]:
        if self._scripts_dir is None:
            folder = self.root / "scripts"
            self._scripts_dir = frozenset(os.listdir(folder)) if folder.is_dir() else frozenset()
        return self._scripts_dir

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

    def import_edges(self) -> dict[str, dict[str, int]]:
        """Module -> {module it imports: 0 for an import at the top of a file, 1 inside a function}.

        ``if TYPE_CHECKING:`` imports are left out. A string counts only as the argument of
        ``import_module``: the ``import_path="pkg.mod:func"`` strings of the metric registry and
        the ``_EXPORTS`` tables name modules that load on first use, and following them would make
        every metric a dependency of the whole CLI (``select_tests`` handles them as dispatch).
        Importing ``a.b.c`` also runs ``a.b`` and ``a``, which is not an edge: a change to the
        ``__init__`` of a package concerns the importers of its names, not those of its children.
        """
        if self._edges is None:
            edges: dict[str, dict[str, int]] = {}
            for module in self.module_names():
                tree = _parse(self._path_of_module(module))
                if tree is None:
                    edges[module] = {}
                    continue
                everywhere = self.package_imports(tree, module, strings="calls")[0]
                at_the_top = self.package_imports(
                    _drop_function_bodies(tree), module, strings="calls"
                )[0]
                edges[module] = {m: 0 if m in at_the_top else 1 for m in everywhere - {module}}
            self._edges = edges
        return self._edges

    def affected_modules(
        self, seeds: Iterable[str], max_lazy: int = MAX_LAZY_IMPORTS
    ) -> dict[str, tuple[int, str | None]]:
        """The seeds and every module that depends on one, each with its distance and next hop.

        A module depends on what it imports. A dependency reached through more than ``max_lazy``
        imports inside functions is dropped: such an import runs only when its function is called,
        and a second one in a chain almost never leads to a test that runs the seed (see the
        figures in docs/testing.md). The next hop is the module one step closer to a seed (None
        for a seed), so that a chain ``a -> b -> seed`` can be read back.
        """
        importers: dict[str, list[tuple[str, int]]] = {}
        for module, dependencies in self.import_edges().items():
            for dependency, lazy in dependencies.items():
                importers.setdefault(dependency, []).append((module, lazy))
        found: dict[str, tuple[int, str | None]] = {}
        best: dict[str, tuple[int, int]] = {}  # module -> (lazy imports, hops) of the best chain
        queue = [(0, 0, module, None) for module in sorted(set(seeds))]
        heapq.heapify(queue)
        while queue:
            lazy, hops, module, previous = heapq.heappop(queue)
            if module in found:
                continue
            found[module] = (hops, previous)
            best[module] = (lazy, hops)
            for importer, is_lazy in sorted(importers.get(module, ())):
                if importer not in found and lazy + is_lazy <= max_lazy:
                    heapq.heappush(queue, (lazy + is_lazy, hops + 1, importer, module))
        return found

    def dependents_closure(self, seeds: Iterable[str]) -> set[str]:
        """The seeds and every module that depends on one of them, directly or not."""
        return set(self.affected_modules(seeds))

    def dispatch_tables(self) -> tuple[set[str], set[str]]:
        """(modules named by ``"pkg.mod:name"`` strings, modules that import a module by variable).

        The first are loaded by name from a table (the metric registry, the parsers, the runners);
        the second hold the code that does it. A test that imports a loader can run any entry of
        a table, whichever module it sits in, which no import statement shows.
        """
        if self._dispatch is None:
            named: set[str] = set()
            loaders: set[str] = set()
            for module in self.module_names():
                tree = _parse(self._path_of_module(module))
                holds_table = loads_by_variable = False
                for node in ast.walk(tree) if tree else []:
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if _DOTTED.fullmatch(node.value) and ":" in node.value:
                            holds_table = True
                            named.add(
                                self._deepest_module(node.value[len(PACKAGE) + 1 :].split(":")[0])
                            )
                    elif isinstance(node, ast.Call) and _called_name(node) in (
                        "import_module",
                        "__import__",
                    ):
                        loads_by_variable |= bool(node.args) and not isinstance(
                            node.args[0], ast.Constant
                        )
                if holds_table and loads_by_variable:
                    loaders.add(module)
            self._dispatch = (named, loaders)
        return self._dispatch

    def select_tests(self, seeds: Iterable[str]) -> dict[Path, str]:
        """Test files that a change to the ``seeds`` modules can affect, each with the reason.

        A test file is chosen when it imports a seed or a module that depends on one (see
        ``affected_modules``), and, for a seed that a table loads by name, when it imports a
        module that does the loading. Imports are those of ``closure``: the file, its helpers and
        the conftest fixtures it asks for.
        """
        seeds = set(seeds)
        affected = self.affected_modules(seeds)
        named, loaders = self.dispatch_tables()
        loaded_by_name = sorted(seeds & named)
        chosen: dict[Path, str] = {}
        for test in self.test_files():
            closure = self.closure(test)
            modules = closure.modules | closure.through
            hits = modules & affected.keys()
            if hits:
                first = min(hits, key=lambda m: (affected[m][0], m))
                chain = [first]
                while affected[chain[-1]][1] is not None:
                    chain.append(affected[chain[-1]][1])
                chosen[test] = "imports " + " -> ".join(m or "binding_metrics" for m in chain)
            elif loaded_by_name and modules & loaders:
                loader = sorted(modules & loaders)[0]
                chosen[test] = f"imports {loader}, which loads {loaded_by_name[0]} by name"
        return chosen

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
            usage: Mapping[str, int] = {}
            through: set[str] = set()
            if tree is not None:
                modules, others = self.package_imports(tree, through=through)
                usage = self.module_usage(tree)
                for node in others:
                    helpers |= self._helper_files(path, node)
                for node in ast.walk(tree):
                    if isinstance(node, ast.arg):
                        names.add(node.arg)
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        names.add(node.value)
                        reads_docs |= bool(_MARKDOWN_PATH.fullmatch(node.value))
                        if node.value == "pyproject.toml":
                            # a test that reads the console scripts runs through all of them
                            modules = modules | set(self.console_scripts().values())
                        elif node.value.endswith(".py") and node.value in self._script_names():
                            # a test that runs a script of scripts/ uses what the script imports
                            helpers.add(self.root / "scripts" / node.value)
                        elif (
                            PACKAGE in node.value and "import" in node.value and "\n" in node.value
                        ):
                            # a test that runs code given as a string (python -c) uses its imports
                            snippet = self._parse_snippet(node.value)
                            if snippet:
                                modules = (
                                    modules | self.package_imports(snippet, through=through)[0]
                                )
                    else:
                        reads_docs |= _names_docs_folder(node)
            self._scans[path] = FileScan(
                frozenset(modules),
                frozenset(helpers),
                frozenset(names),
                reads_docs,
                usage,
                frozenset(through),
            )
        return self._scans[path]

    def module_usage(self, tree: ast.AST) -> Counter:
        """How often a file refers to each module of the package.

        A reference is a use of a name that the file imported from the module (the name itself, or
        the module bound by ``from binding_metrics.metrics import geometry``), a dotted chain from
        ``binding_metrics`` or a string that names the module. Imports that are never used count
        nothing, which is what separates a test about a module from one that only needs it.
        """
        aliases: dict[str, str] = {}
        names: list[ast.Name] = []
        attributes: list[ast.Attribute] = []
        strings: list[object] = []
        for node in walk_runtime(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname and (
                        alias.name == PACKAGE or alias.name.startswith(PACKAGE + ".")
                    ):
                        aliases[alias.asname] = self._deepest_module(
                            alias.name[len(PACKAGE) :].lstrip(".")
                        )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == PACKAGE or node.module.startswith(PACKAGE + "."):
                    base = node.module[len(PACKAGE) :].lstrip(".")
                    for alias in node.names:
                        if alias.name != "*":
                            (module,) = self.resolve_from_import(base, [alias.name])
                            aliases[alias.asname or alias.name] = module
            elif isinstance(node, ast.Name):
                names.append(node)
            elif isinstance(node, ast.Attribute):
                attributes.append(node)
            elif isinstance(node, ast.Constant):
                strings.append(node.value)
        counts: Counter = Counter(
            aliases[n.id] for n in names if isinstance(n.ctx, ast.Load) and n.id in aliases
        )
        inside_a_chain = {id(a.value) for a in attributes}
        for node in attributes:
            if id(node) in inside_a_chain:
                continue
            chain, inner = [], node
            while isinstance(inner, ast.Attribute):
                chain.append(inner.attr)
                inner = inner.value
            if isinstance(inner, ast.Name) and inner.id == PACKAGE:
                counts[self._deepest_module(".".join(reversed(chain)))] += 1
        for value in strings:
            counts.update(self._modules_named_by(value))
        return counts

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

    def closure(self, path: Path) -> Closure:
        """Package modules a test file uses, and the files (itself and its helpers) behind them."""
        path = Path(os.path.abspath(path))
        if path not in self._closures:
            fixtures = self.fixture_modules()
            modules: set[str] = set()
            through: set[str] = set()
            seen: set[Path] = set()
            stack = [path]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                scan = self.scan(current)
                modules |= scan.modules
                through |= scan.through
                for fixture in scan.names & fixtures.keys():
                    modules |= fixtures[fixture]
                stack.extend(scan.helpers)
            self._closures[path] = Closure(frozenset(modules), frozenset(seen), frozenset(through))
        return self._closures[path]

    def usage_of(self, path: Path) -> Counter:
        """Module -> references, over a test file, its helpers and the fixtures it asks for."""
        modules, files, _ = self.closure(path)
        counts: Counter = Counter()
        fixtures = self.fixture_modules()
        for file in files:
            scan = self.scan(file)
            counts.update(scan.usage)
            for fixture in scan.names & fixtures.keys():
                counts.update(dict.fromkeys(fixtures[fixture], 1))
        return counts

    def areas_of_test_file(self, path: Path) -> frozenset[str]:
        """What a test file is about: its markers, always at least one.

        The fine areas are those that hold at least ``PRIMARY_SHARE`` of the file's references to
        the package, and the one with the most references whatever its share; a file that refers
        to nothing by name (it only imports) takes the areas of all its imports. The coarse area
        of each fine one comes with it, then ``docs`` and ``golden`` where they apply.
        """
        modules, files, _ = self.closure(path)
        by_area: Counter = Counter()
        for module, count in self.usage_of(path).items():
            by_area[area_of_module(module) or "misc"] += count
        total = sum(by_area.values())
        if total:
            fine = {a for a, count in by_area.items() if count / total >= PRIMARY_SHARE}
            fine.add(max(sorted(by_area), key=by_area.__getitem__))
        else:
            fine = {area_of_module(m) or "misc" for m in modules}
        areas = fine | {coarse_of(a) for a in fine}
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
