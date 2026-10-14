"""Automatic platform selection on a machine whose CUDA plugin has no device (#173).

OpenMM registers its CUDA platform when the plugin loads, whether or not a device is visible, and
the error shows only when a context is created on it. ``MDSimulation`` with ``platform="auto"``
took the CUDA platform as soon as it was registered, so the run ended in CUDA error 100 instead of
using the CPU platform. The energy metrics, which fall back from ``device="cuda"`` to the CPU
platform, looked the same way and recorded that error.

Each call runs in a subprocess with ``CUDA_VISIBLE_DEVICES=-1``: no device is visible there, on a
machine with a GPU as well, and the process the tests run in keeps its own CUDA state. On an
install without the CUDA plugin the platform is unregistered and the same assertions hold.
"""

import json
import math
import os
import subprocess
import sys
import textwrap

import pytest
from conftest import DATA_DIR

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

pytestmark = [pytest.mark.slow, pytest.mark.integration]

EXAMPLE = DATA_DIR / "example_linear_p53_1YCR.pdb"

SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import openmm
    from openmm import Vec3, app, unit

    from binding_metrics.core.forcefields import get_forcefield
    from binding_metrics.core.simulation import MDSimulation, SimulationConfig
    from binding_metrics.metrics.energy import compute_interaction_energy
    from binding_metrics.metrics.receptor_quality import compute_receptor_quality

    prepped, example = sys.argv[1:3]
    out = {
        "registered": [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
    }

    # The platforms that the metrics hand to openmm.app.Simulation.
    used = []
    real_simulation = app.Simulation


    class Recording(real_simulation):
        def __init__(self, topology, system, integrator, platform=None, *args, **kwargs):
            used.append(platform.getName())
            super().__init__(topology, system, integrator, platform, *args, **kwargs)


    app.Simulation = Recording


    def attempt(name, function):
        try:
            out[name] = function()
        except Exception as exc:
            out[name] = {"raised": type(exc).__name__, "message": str(exc)}


    def water_box(platform):
        forcefield = get_forcefield("amber")
        modeller = app.Modeller(app.Topology(), [])
        modeller.addSolvent(forcefield, boxSize=Vec3(1.2, 1.2, 1.2) * unit.nanometer)
        config = SimulationConfig(platform=platform, pressure=None, nonbonded_cutoff=0.5)
        return MDSimulation(modeller, forcefield, config)


    def auto():
        simulation = water_box("auto")
        simulation.setup()
        return simulation.simulation.context.getPlatform().getName()


    attempt("auto", auto)
    attempt("explicit", lambda: water_box("CUDA").setup())

    used.clear()


    def energy():
        result = compute_interaction_energy(prepped, modes=("raw",))
        return {
            "success": result["success"],
            "error_message": result["error_message"],
            "used": list(used),
        }


    def receptor_quality():
        used.clear()
        energy_term = compute_receptor_quality(example, "A")["models"][0]["energy"]
        return {
            "energy_kJ_mol": energy_term["energy_kJ_mol"],
            "error": energy_term["error"],
            "used": list(used),
        }


    attempt("energy", energy)
    attempt("receptor_quality", receptor_quality)
    print("RESULT " + json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def without_a_device(prepped_example_cif):
    """What each call did in a process that sees no CUDA device."""
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(prepped_example_cif), str(EXAMPLE)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    line = next(line for line in run.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_auto_uses_the_cpu_platform_when_cuda_has_no_device(without_a_device):
    assert without_a_device["auto"] == "CPU"


def test_an_explicit_cuda_request_fails_and_names_the_platform(without_a_device):
    failure = without_a_device["explicit"]

    assert failure["raised"] == "OpenMMException"
    assert "CUDA" in failure["message"]
    if "CUDA" in without_a_device["registered"]:
        assert "OpenMM platform CUDA is not usable" in failure["message"]
        assert "CUDA_ERROR_NO_DEVICE" in failure["message"]


def test_the_interaction_energy_runs_on_the_cpu_platform(without_a_device):
    result = without_a_device["energy"]

    assert result["success"], result
    assert result["error_message"] is None
    assert set(result["used"]) == {"CPU"}


def test_the_receptor_energy_runs_on_the_cpu_platform(without_a_device):
    result = without_a_device["receptor_quality"]

    assert result["error"] is None, result
    assert math.isfinite(result["energy_kJ_mol"])
    assert set(result["used"]) == {"CPU"}
