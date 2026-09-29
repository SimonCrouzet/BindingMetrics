"""``metrics/energy.py`` does not need OpenMM until a function that uses it runs.

The blocked-import checks run in a subprocess so the fake "OpenMM is not
installed" state cannot leak into the rest of the test session. The plain
``from binding_metrics.metrics import energy`` runs there, so the parent
packages' ``__init__`` files are part of what is checked.
"""

import subprocess
import sys
import textwrap

import pytest

_BLOCKED_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys


    class _BlockOpenMM(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("openmm", "simtk"):
                raise ImportError(f"{name} is blocked for this test")
            return None


    sys.meta_path.insert(0, _BlockOpenMM())

    from binding_metrics.metrics import energy

    assert "openmm" not in sys.modules, "energy.py imported openmm at module level"
    assert not any(m.startswith("simtk") for m in sys.modules)

    args = energy._build_parser().parse_args(["--input", "x.cif", "--random-seed", "5"])
    assert args.random_seed == 5

    try:
        energy._get_platform("cpu")
    except ImportError:
        pass
    else:
        raise AssertionError("_get_platform should need OpenMM")

    print("DEFAULT_RANDOM_SEED=" + repr(energy.DEFAULT_RANDOM_SEED))
    """
)


@pytest.fixture(scope="module")
def blocked_import_run(tmp_path_factory):
    return subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path_factory.mktemp("blocked_import"),
        timeout=120,
    )


def test_module_imports_without_openmm(blocked_import_run):
    assert blocked_import_run.returncode == 0, blocked_import_run.stderr[-2000:]


def test_fallback_seed_matches_core_system(blocked_import_run):
    """Without OpenMM the module has the seed default that ``core.system`` re-exports."""
    pytest.importorskip("openmm")
    from binding_metrics.core.system import DEFAULT_RANDOM_SEED

    assert f"DEFAULT_RANDOM_SEED={DEFAULT_RANDOM_SEED!r}" in blocked_import_run.stdout


def test_openmm_names_still_resolve_from_the_module():
    """The names the module used to import eagerly keep working as attributes."""
    pytest.importorskip("openmm")
    import openmm
    import openmm.app
    import openmm.unit

    from binding_metrics.core.forcefields import get_forcefield
    from binding_metrics.metrics import energy

    assert energy.openmm is openmm
    assert energy.unit is openmm.unit
    assert energy.ForceField is openmm.app.ForceField
    assert energy.PDBFile is openmm.app.PDBFile
    assert energy.get_forcefield is get_forcefield
    with pytest.raises(AttributeError):
        energy.no_such_name  # noqa: B018
