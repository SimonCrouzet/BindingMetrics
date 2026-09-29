"""OpenMM-only metric entry points name the extra to install when OpenMM is missing.

The check runs in a subprocess whose ``sys.meta_path`` refuses to import OpenMM, so the
fake "not installed" state does not leak into the test session.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLE_PDB = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import json
    import sys
    import traceback

    BLOCKED = ("openmm", "simtk")


    class _BlockOpenMM(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None


    sys.meta_path.insert(0, _BlockOpenMM())

    EXAMPLE = {example!r}
    results = {{}}


    def check(function):
        try:
            function()
        except BaseException:
            results[function.__name__] = traceback.format_exc()[-1500:]
        else:
            results[function.__name__] = "ok"


    def _expect_hint(call, needle):
        try:
            call()
        except ImportError as error:
            text = str(error)
            assert "pip install binding-metrics[simulation]" in text, text
            assert needle in text, text
            assert getattr(error, "name", None) == "openmm", repr(error.name)
        else:
            raise AssertionError("no ImportError")


    @check
    def compute_interaction_energy():
        from binding_metrics.metrics.energy import compute_interaction_energy as fn

        _expect_hint(lambda: fn(EXAMPLE, modes=("raw",)), "compute_interaction_energy")


    @check
    def compute_interaction_energy_with_aliases():
        from binding_metrics.metrics.energy import compute_interaction_energy as fn

        _expect_hint(lambda: fn(EXAMPLE, binder_chain="B", target_chain="A"), "OpenMM")


    def _pretend_mdtraj_is_installed():
        # On an install without mdtraj the mdtraj message would come first.
        from binding_metrics.metrics import energy

        if energy.md is None:
            energy.md = object()


    @check
    def calculate_interaction_energy():
        _pretend_mdtraj_is_installed()
        from binding_metrics.metrics.energy import calculate_interaction_energy as fn

        _expect_hint(lambda: fn("t.dcd", EXAMPLE, [0], [1]), "calculate_interaction_energy")


    @check
    def calculate_component_energies():
        _pretend_mdtraj_is_installed()
        from binding_metrics.metrics.energy import calculate_component_energies as fn

        _expect_hint(lambda: fn("t.dcd", EXAMPLE, [0], [1]), "calculate_component_energies")


    @check
    def mdtraj_message_still_comes_first_when_mdtraj_is_missing():
        from binding_metrics.metrics import energy

        original = energy.md
        energy.md = None
        try:
            energy.calculate_interaction_energy("t.dcd", EXAMPLE, [0], [1])
        except ImportError as error:
            assert "mdtraj is required" in str(error), str(error)
        else:
            raise AssertionError("no ImportError")
        finally:
            energy.md = original


    @check
    def static_metrics_still_run():
        from binding_metrics.metrics.interface import compute_interface_metrics

        assert compute_interface_metrics(EXAMPLE)["delta_sasa"] > 0


    print("RESULTS=" + json.dumps(results))
    """
)

CHECKS = [
    "compute_interaction_energy",
    "compute_interaction_energy_with_aliases",
    "calculate_interaction_energy",
    "calculate_component_energies",
    "mdtraj_message_still_comes_first_when_mdtraj_is_missing",
    "static_metrics_still_run",
]


@pytest.fixture(scope="module")
def blocked_results(tmp_path_factory):
    pytest.importorskip("biotite")
    script = _SCRIPT.format(example=str(EXAMPLE_PDB))
    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path_factory.mktemp("no_openmm"),
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(x for x in run.stdout.splitlines() if x.startswith("RESULTS="))
    return json.loads(line[len("RESULTS=") :])


@pytest.mark.parametrize("check", CHECKS)
def test_openmm_entry_points_name_the_extra(blocked_results, check):
    assert blocked_results[check] == "ok", blocked_results[check]
