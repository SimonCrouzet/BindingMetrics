"""The package and the static (single-structure) metrics need no OpenMM.

The checks run in a subprocess with a ``sys.meta_path`` finder that refuses to import
OpenMM and the other simulation-only packages, so the fake "not installed" state cannot
leak into the rest of the test session. On an install that really lacks them (the CI
job ``Static import``) the finder changes nothing and the same assertions hold.

The second half checks, with OpenMM present, that every name importable before the
package exports became lazy is still importable through every path.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLE_PDB = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

# Packages that only the simulation and force-field routes use.
SIMULATION_ONLY = ("openmm", "simtk", "pdbfixer", "openmmforcefields", "openff", "mdtraj", "rdkit")

_SCRIPT = textwrap.dedent(
    """
    import contextlib
    import importlib
    import importlib.abc
    import io
    import json
    import sys
    import traceback

    BLOCKED = {blocked!r}


    class _BlockSimulationPackages(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None


    sys.meta_path.insert(0, _BlockSimulationPackages())

    EXAMPLE = {example!r}
    results = {{}}


    def check(function):
        try:
            function()
        except BaseException:
            results[function.__name__] = traceback.format_exc()[-1500:]
        else:
            results[function.__name__] = "ok"


    @check
    def import_package():
        import binding_metrics
        import binding_metrics.core
        import binding_metrics.io
        import binding_metrics.metrics
        import binding_metrics.protocols

        assert binding_metrics.__version__


    @check
    def import_static_metric_modules():
        for name in (
            "comparison", "contacts", "dockq", "electrostatics", "energy", "evobind",
            "geometry", "interface", "openfold", "polar_contacts", "receptor_quality",
            "registry", "rmsd", "sasa",
        ):
            importlib.import_module("binding_metrics.metrics." + name)


    @check
    def import_static_exports():
        from binding_metrics import (
            compute_buried_void_volume, compute_coulomb_cross_chain, compute_delta_sasa_static,
            compute_hbonds, compute_interface_metrics, compute_omega_planarity,
            compute_ramachandran, compute_receptor_drift, compute_saltbridges,
            compute_shape_complementarity, compute_structure_rmsd, ForceFieldConfig,
        )
        from binding_metrics.metrics import calculate_rmsd, compute_receptor_quality  # noqa: F401


    @check
    def import_light_modules():
        from binding_metrics import _constants, provenance, utils  # noqa: F401
        from binding_metrics.core import D_AA_MAP, CyclicBondInfo, NonstandardInfo  # noqa: F401
        from binding_metrics.core.nonstandard import D_AA_MAP as same_map

        assert D_AA_MAP is same_map and D_AA_MAP


    @check
    def interface_metrics_run():
        from binding_metrics import compute_interface_metrics

        metrics = compute_interface_metrics(EXAMPLE)
        # MDM2 (A) - p53 peptide (B): about 1500 A^2 buried on a 5400 A^2 receptor.
        assert (metrics["receptor_chain"], metrics["peptide_chain"]) == ("A", "B")
        assert 800.0 < metrics["delta_sasa"] < 3000.0, metrics["delta_sasa"]
        assert metrics["delta_g_int"] < 0.0
        assert metrics["hbonds"] >= 1


    @check
    def shape_complementarity_runs():
        from binding_metrics import compute_shape_complementarity

        result = compute_shape_complementarity(EXAMPLE)
        # Protein-peptide interfaces score about 0.6-0.8 on the Lawrence-Colman scale.
        assert 0.4 < result["sc"] < 0.9, result["sc"]


    @check
    def structure_rmsd_runs():
        from binding_metrics import compute_structure_rmsd

        assert compute_structure_rmsd(EXAMPLE, EXAMPLE)["rmsd"] < 1e-6


    @check
    def registry_loads():
        from binding_metrics.metrics.registry import METRICS, get_metric

        static = [spec for spec in METRICS if spec.cost_class == "static"]
        assert len(static) >= 8
        for spec in static:
            assert callable(spec.load()), spec.name
        assert callable(get_metric("interface").load())


    @check
    def metric_command_line_interfaces():
        for name in (
            "interface", "comparison", "electrostatics", "geometry", "receptor_quality",
            "dockq", "openfold",
        ):
            module = importlib.import_module("binding_metrics.metrics." + name)
            sys.argv = [name, "--help"]
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                try:
                    module.main()
                except SystemExit as exit_request:
                    assert exit_request.code in (0, None), name
                else:
                    raise AssertionError(f"{{name}} --help did not exit")
            assert "usage" in buffer.getvalue().lower(), name


    @check
    def interface_command_line_runs():
        from binding_metrics.metrics import interface

        sys.argv = ["binding-metrics-interface", "--input", EXAMPLE]
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            interface.main()
        assert "delta_sasa" in buffer.getvalue()


    @check
    def simulation_names_explain_the_missing_extra():
        import binding_metrics
        import binding_metrics.core

        for owner, name in (
            (binding_metrics, "MDSimulation"),
            (binding_metrics, "prepare_system"),
            (binding_metrics, "PeptideBindingProtocol"),
            (binding_metrics.core, "SimulationConfig"),
        ):
            try:
                getattr(owner, name)
            except ImportError as error:
                assert "pip install binding-metrics[simulation]" in str(error), str(error)
            else:
                raise AssertionError(f"{{owner.__name__}}.{{name}} needs OpenMM")


    @check
    def openmm_function_explains_the_missing_extra():
        from binding_metrics import get_forcefield

        try:
            get_forcefield("amber")
        except ImportError as error:
            assert "pip install binding-metrics[simulation]" in str(error), str(error)
        else:
            raise AssertionError("get_forcefield needs OpenMM")


    @check
    def no_simulation_package_was_imported():
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        assert not loaded, loaded


    print("RESULTS=" + json.dumps(results))
    """
)

CHECKS = [
    "import_package",
    "import_static_metric_modules",
    "import_static_exports",
    "import_light_modules",
    "interface_metrics_run",
    "shape_complementarity_runs",
    "structure_rmsd_runs",
    "registry_loads",
    "metric_command_line_interfaces",
    "interface_command_line_runs",
    "simulation_names_explain_the_missing_extra",
    "openmm_function_explains_the_missing_extra",
    "no_simulation_package_was_imported",
]


@pytest.fixture(scope="module")
def blocked_results(tmp_path_factory):
    script = _SCRIPT.format(blocked=SIMULATION_ONLY, example=str(EXAMPLE_PDB))
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
def test_without_simulation_packages(blocked_results, check):
    assert blocked_results[check] == "ok", blocked_results[check]


# Names each namespace exposed while the ``__init__`` files imported everything at load
# time. Listed by hand so that dropping a name from an export table and from ``__all__``
# together still fails.
PREVIOUSLY_IMPORTABLE = {
    "binding_metrics": [
        "ForceFieldConfig",
        "ImplicitRelaxation",
        "MDSimulation",
        "PeptideBindingProtocol",
        "ProtocolResults",
        "RelaxationConfig",
        "RelaxationResult",
        "SimulationConfig",
        "compute_buried_void_volume",
        "compute_coulomb_cross_chain",
        "compute_delta_sasa_static",
        "compute_dockq_metrics",
        "compute_evobind_adversarial_check",
        "compute_evobind_score",
        "compute_hbonds",
        "compute_interaction_energy",
        "compute_interface_metrics",
        "compute_omega_planarity",
        "compute_openfold_metrics",
        "compute_ramachandran",
        "compute_receptor_drift",
        "compute_saltbridges",
        "compute_shape_complementarity",
        "compute_structure_rmsd",
        "detect_chains",
        "get_forcefield",
        "load_structure",
        "prepare_system",
        "run_openfold",
        "run_simulation",
        "save_cif",
        # submodules
        "_constants",
        "core",
        "io",
        "metrics",
        "protocols",
        "utils",
    ],
    "binding_metrics.core": [
        "CyclicBondInfo",
        "CyclizationError",
        "D_AA_MAP",
        "ForceFieldConfig",
        "MDSimulation",
        "NME_AA_MAP",
        "NonstandardInfo",
        "SimulationConfig",
        "get_addh_variants",
        "get_forcefield",
        "prepare_system",
        "cyclic",
        "forcefields",
        "nonstandard",
        "simulation",
        "system",
    ],
    "binding_metrics.io": ["get_chain_atom_indices", "load_complex", "structures"],
    "binding_metrics.protocols": [
        "BaseProtocol",
        "PeptideBindingProtocol",
        "ProtocolResults",
        "base",
        "peptide",
        "relaxation",
    ],
    "binding_metrics.metrics": [
        "_openfold_cli",
        "_openfold_run",
        "calculate_buried_sasa",
        "calculate_contacts",
        "calculate_interaction_energy",
        "calculate_rmsd",
        "compute_buried_void_volume",
        "compute_coulomb_cross_chain",
        "compute_evobind_adversarial_check",
        "compute_evobind_score",
        "compute_interface_pae",
        "compute_omega_planarity",
        "compute_openfold_metrics",
        "compute_ramachandran",
        "compute_receptor_drift",
        "compute_receptor_quality",
        "compute_shape_complementarity",
        "prepare_refolding_query",
        "prepare_scoring_query",
        "run_openfold",
        "run_openfold_refolding",
        "run_openfold_scoring",
        "comparison",
        "contacts",
        "dockq",
        "electrostatics",
        "energy",
        "evobind",
        "geometry",
        "interface",
        "openfold",
        "polar_contacts",
        "receptor_quality",
        "rmsd",
        "sasa",
    ],
}


@pytest.mark.parametrize("package_name", sorted(PREVIOUSLY_IMPORTABLE))
def test_previously_importable_names_still_resolve(package_name):
    pytest.importorskip("openmm")
    import importlib

    package = importlib.import_module(package_name)

    for name in PREVIOUSLY_IMPORTABLE[package_name]:
        assert getattr(package, name) is not None, f"{package_name}.{name}"
        namespace = {}
        exec(f"from {package_name} import {name}", namespace)
        assert namespace[name] is getattr(package, name)


def test_star_import_binds_every_public_name():
    pytest.importorskip("openmm")
    import binding_metrics

    namespace = {}
    exec("from binding_metrics import *", namespace)

    assert set(binding_metrics.__all__) <= set(namespace)
