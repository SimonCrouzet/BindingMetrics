"""``core/forcefields.py`` imports OpenMM only when ``get_forcefield`` builds a ForceField.

The blocked-import checks run in a subprocess so the fake "OpenMM is not
installed" state cannot leak into the rest of the test session. The module is
loaded straight from its file so the check does not depend on what the package
``__init__`` files import.
"""

import subprocess
import sys
import textwrap

import pytest

_BLOCKED_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import importlib.util
    import os


    class _BlockOpenMM(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("openmm", "simtk"):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None


    sys.meta_path.insert(0, _BlockOpenMM())

    package_dir = os.path.join(
        os.path.dirname(importlib.util.find_spec("binding_metrics").origin), "core"
    )
    spec = importlib.util.spec_from_file_location(
        "forcefields_standalone", os.path.join(package_dir, "forcefields.py")
    )
    forcefields = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forcefields)

    assert "openmm" not in sys.modules, "forcefields.py imported openmm at module level"

    assert forcefields.get_forcefield_config("charmm").protein_ff == "charmm36.xml"
    assert set(forcefields.FORCEFIELD_CONFIGS) == {"amber", "charmm"}

    try:
        forcefields.get_forcefield("gromos")
    except ValueError as exc:
        assert "Valid options" in str(exc)
    else:
        raise AssertionError("an unknown name must raise ValueError before OpenMM is needed")

    try:
        forcefields.get_forcefield("amber")
    except ImportError as exc:
        print("MESSAGE=" + str(exc))
    else:
        raise AssertionError("get_forcefield should need OpenMM")

    try:
        forcefields.ForceField
    except ImportError:
        pass
    else:
        raise AssertionError("ForceField should resolve through OpenMM")
    """
)


@pytest.fixture(scope="module")
def blocked_run(tmp_path_factory):
    return subprocess.run(
        [sys.executable, "-c", "import sys\n" + _BLOCKED_SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path_factory.mktemp("forcefields_blocked"),
        timeout=120,
    )


def test_config_api_works_without_openmm(blocked_run):
    assert blocked_run.returncode == 0, blocked_run.stderr[-2000:]


def test_get_forcefield_error_names_the_extra(blocked_run):
    assert "pip install binding-metrics[simulation]" in blocked_run.stdout


def test_openmm_backed_names_still_resolve():
    """``ForceField`` stays reachable as a module attribute and get_forcefield still works."""
    openmm_app = pytest.importorskip("openmm.app")

    from binding_metrics.core import forcefields

    assert forcefields.ForceField is openmm_app.ForceField
    assert isinstance(forcefields.get_forcefield("amber"), openmm_app.ForceField)
    with pytest.raises(AttributeError):
        forcefields.no_such_name  # noqa: B018
