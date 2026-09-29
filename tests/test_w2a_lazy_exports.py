"""Lazy (PEP 562) exports of the five package ``__init__`` files.

The tables that drive the lazy loading, the ``TYPE_CHECKING`` blocks that keep IDEs
working and ``__all__`` are three copies of one list; these tests keep them equal.
Checks that need a clean interpreter (nothing imported yet, dependencies blocked)
run in a subprocess so they cannot leak into the rest of the test session.
"""

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import binding_metrics

PACKAGES = [
    "binding_metrics",
    "binding_metrics.core",
    "binding_metrics.io",
    "binding_metrics.protocols",
    "binding_metrics.metrics",
]


def _type_checking_imports(package):
    """Map name -> module from the ``if TYPE_CHECKING:`` block of a package ``__init__``."""
    tree = ast.parse(Path(package.__file__).read_text())
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom)
                for alias in statement.names:
                    imports[alias.asname or alias.name] = statement.module
    return imports


@pytest.mark.parametrize("package_name", PACKAGES)
def test_export_table_matches_all_and_type_checking_block(package_name):
    package = importlib.import_module(package_name)

    assert set(package._EXPORTS) == set(package.__all__)
    assert package._EXPORTS == _type_checking_imports(package)


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_export_is_the_object_its_module_defines(package_name):
    pytest.importorskip("openmm")
    package = importlib.import_module(package_name)

    for name, module_name in package._EXPORTS.items():
        expected = getattr(importlib.import_module(module_name), name)
        assert getattr(package, name) is expected, name
        # Cached on the package after the first access.
        assert vars(package)[name] is expected, name


@pytest.mark.parametrize("package_name", PACKAGES)
def test_dir_lists_every_export(package_name):
    package = importlib.import_module(package_name)

    assert set(package.__all__) <= set(dir(package))


@pytest.mark.parametrize("package_name", PACKAGES)
def test_unknown_attribute_raises_attribute_error(package_name):
    package = importlib.import_module(package_name)

    with pytest.raises(AttributeError, match="no_such_name"):
        package.no_such_name  # noqa: B018
    assert not hasattr(package, "__wrapped__")


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """A throwaway package whose one module imports a dependency that is not installed."""
    root = tmp_path / "w2a_fake_package"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "needs_absent.py").write_text("import w2a_absent_dependency\nthing = 1\n")
    (root / "needs_unlisted.py").write_text("import w2a_unlisted_dependency\nthing = 1\n")
    (root / "fine.py").write_text("thing = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        binding_metrics._EXTRA_FOR_DEPENDENCY, "w2a_absent_dependency", "simulation"
    )
    yield "w2a_fake_package"
    for name in [m for m in sys.modules if m.startswith("w2a_fake_package")]:
        del sys.modules[name]


def test_missing_optional_dependency_names_the_extra(fake_package):
    namespace = {}
    getter, _ = binding_metrics._lazy_exports(
        fake_package, {"thing": f"{fake_package}.needs_absent"}, namespace
    )

    with pytest.raises(ImportError) as excinfo:
        getter("thing")

    assert "pip install binding-metrics[simulation]" in str(excinfo.value)
    assert "'w2a_absent_dependency'" in str(excinfo.value)
    assert excinfo.value.name == "w2a_absent_dependency"
    assert "thing" not in namespace


def test_unlisted_missing_dependency_is_not_reworded(fake_package):
    getter, _ = binding_metrics._lazy_exports(
        fake_package, {"thing": f"{fake_package}.needs_unlisted"}, {}
    )

    with pytest.raises(ModuleNotFoundError, match="w2a_unlisted_dependency") as excinfo:
        getter("thing")

    assert "pip install" not in str(excinfo.value)


def test_export_is_cached_after_first_access(fake_package):
    namespace = {}
    getter, lister = binding_metrics._lazy_exports(
        fake_package, {"thing": f"{fake_package}.fine"}, namespace
    )

    assert getter("thing") == 42
    assert namespace["thing"] == 42
    assert "thing" in lister()


def test_submodule_fallback_imports_the_submodule(fake_package):
    getter, _ = binding_metrics._lazy_exports(fake_package, {}, {})

    assert getter("fine").thing == 42
    with pytest.raises(AttributeError):
        getter("not_a_module")


def _run_in_clean_interpreter(script, tmp_path):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=180,
    )


def test_importing_the_package_loads_no_submodule(tmp_path):
    result = _run_in_clean_interpreter(
        """
        import sys
        import binding_metrics
        import binding_metrics.metrics

        loaded = sorted(m for m in sys.modules if m.startswith("binding_metrics."))
        assert loaded == ["binding_metrics.metrics"], loaded
        assert "openmm" not in sys.modules
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr[-2000:]


def test_submodules_resolve_as_attributes_without_an_explicit_import(tmp_path):
    """``package.submodule`` worked when the ``__init__`` files imported everything."""
    result = _run_in_clean_interpreter(
        """
        import binding_metrics

        assert binding_metrics.metrics.geometry.__name__ == "binding_metrics.metrics.geometry"
        assert binding_metrics.core.nonstandard.D_AA_MAP
        assert binding_metrics.utils.__name__ == "binding_metrics.utils"
        assert hasattr(binding_metrics._constants, "DEFAULT_RANDOM_SEED")
        assert "geometry" in dir(binding_metrics.metrics)
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr[-2000:]
