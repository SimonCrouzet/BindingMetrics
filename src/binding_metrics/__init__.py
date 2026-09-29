"""BindingMetrics: GPU-compatible binding metrics evaluation through MD simulations.

The public names listed in ``__all__`` are loaded on first access (PEP 562), so
``import binding_metrics`` does not import OpenMM and the static, single-structure
metrics work on an install without it. ``from binding_metrics import X``,
``binding_metrics.X`` and ``from binding_metrics import *`` behave as before. A
name whose optional dependency is missing raises an ``ImportError`` that names the
extra to install.
"""

import importlib
import importlib.metadata
import importlib.util
from typing import TYPE_CHECKING

# Extra in pyproject.toml that installs each optional third-party package. Used to
# tell the user what to install when a lazily loaded name cannot be imported.
_EXTRA_FOR_DEPENDENCY = {
    "openmm": "simulation",
    "simtk": "simulation",
    "mdtraj": "analysis",
    "pdbfixer": "structure",
    "gemmi": "static",
    "biotite": "static",
    "hydride": "static",
    "scipy": "static",
    "DockQ": "dockq",
    "openfold": "openfold",
    "pandas": "report",
    "matplotlib": "report",
    "markdown": "report",
}
# No pip requirement exists for these; environment.yml provides them.
_CONDA_ONLY_DEPENDENCIES = frozenset({"openmmforcefields", "openff", "rdkit"})


def _missing_dependency_message(attribute: str, dependency: str) -> str | None:
    """Explain how to install ``dependency``, or return None if it is not a known optional one."""
    root = dependency.split(".")[0]
    if root in _EXTRA_FOR_DEPENDENCY:
        return (
            f"{attribute} needs the optional dependency {root!r}, which is not installed. "
            f"Install it with `pip install binding-metrics[{_EXTRA_FOR_DEPENDENCY[root]}]`."
        )
    if root in _CONDA_ONLY_DEPENDENCIES:
        return (
            f"{attribute} needs the optional dependency {root!r}, which is not installed. "
            "It is available from conda-forge only: create the environment from "
            "environment.yml."
        )
    return None


def _import_module(module_name: str, attribute: str):
    """Import ``module_name``; a missing optional dependency gets an install hint."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        message = _missing_dependency_message(attribute, exc.name or "")
        if message is None:
            raise
        raise ModuleNotFoundError(message, name=exc.name) from exc


def _lazy_exports(package: str, exports: dict[str, str], namespace: dict):
    """Build the module-level ``__getattr__`` and ``__dir__`` of a package (PEP 562).

    ``exports`` maps each public name to the module that defines it. The first
    access imports that module and caches the value in ``namespace`` (the package's
    ``globals()``). Names that are not exports fall back to the package's own
    submodules, so ``package.submodule`` still resolves without an explicit import,
    as it did when the ``__init__`` files imported everything at load time.
    """

    def module_getattr(name: str):
        module_name = exports.get(name)
        if module_name is not None:
            module = _import_module(module_name, f"{package}.{name}")
            value = getattr(module, name)
        elif name.startswith("__") or "." in name:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        else:
            submodule = f"{package}.{name}"
            if importlib.util.find_spec(submodule) is None:
                raise AttributeError(f"module {package!r} has no attribute {name!r}")
            value = _import_module(submodule, submodule)
        namespace[name] = value
        return value

    def module_dir() -> list[str]:
        return sorted({*namespace, *exports})

    return module_getattr, module_dir


# Public name -> defining module. ``tests/test_w2a_lazy_exports.py`` checks this
# table against the TYPE_CHECKING block and ``__all__``.
_EXPORTS = {
    # Core
    "ForceFieldConfig": "binding_metrics.core.forcefields",
    "get_forcefield": "binding_metrics.core.forcefields",
    "MDSimulation": "binding_metrics.core.simulation",
    "SimulationConfig": "binding_metrics.core.simulation",
    "run_simulation": "binding_metrics.core.simulation",
    "prepare_system": "binding_metrics.core.system",
    # I/O
    "detect_chains": "binding_metrics.io.structures",
    "load_structure": "binding_metrics.io.structures",
    "save_cif": "binding_metrics.io.structures",
    # Metrics
    "compute_structure_rmsd": "binding_metrics.metrics.comparison",
    "compute_dockq_metrics": "binding_metrics.metrics.dockq",
    "compute_coulomb_cross_chain": "binding_metrics.metrics.electrostatics",
    "compute_interaction_energy": "binding_metrics.metrics.energy",
    "compute_evobind_adversarial_check": "binding_metrics.metrics.evobind",
    "compute_evobind_score": "binding_metrics.metrics.evobind",
    "compute_buried_void_volume": "binding_metrics.metrics.geometry",
    "compute_omega_planarity": "binding_metrics.metrics.geometry",
    "compute_ramachandran": "binding_metrics.metrics.geometry",
    "compute_shape_complementarity": "binding_metrics.metrics.geometry",
    "compute_interface_metrics": "binding_metrics.metrics.interface",
    "compute_openfold_metrics": "binding_metrics.metrics.openfold",
    "run_openfold": "binding_metrics.metrics.openfold",
    "compute_hbonds": "binding_metrics.metrics.polar_contacts",
    "compute_saltbridges": "binding_metrics.metrics.polar_contacts",
    "compute_receptor_drift": "binding_metrics.metrics.rmsd",
    "compute_delta_sasa_static": "binding_metrics.metrics.sasa",
    # Protocols
    "ProtocolResults": "binding_metrics.protocols.base",
    "PeptideBindingProtocol": "binding_metrics.protocols.peptide",
    "ImplicitRelaxation": "binding_metrics.protocols.relaxation",
    "RelaxationConfig": "binding_metrics.protocols.relaxation",
    "RelaxationResult": "binding_metrics.protocols.relaxation",
    "Relaxer": "binding_metrics.protocols.relaxer",
    # Pipeline API
    "run_pipeline": "binding_metrics.cli.run",
    "run_batch": "binding_metrics.cli.batch",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.cli.batch import run_batch
    from binding_metrics.cli.run import run_pipeline
    from binding_metrics.core.forcefields import ForceFieldConfig, get_forcefield
    from binding_metrics.core.simulation import MDSimulation, SimulationConfig, run_simulation
    from binding_metrics.core.system import prepare_system
    from binding_metrics.io.structures import detect_chains, load_structure, save_cif
    from binding_metrics.metrics.comparison import compute_structure_rmsd
    from binding_metrics.metrics.dockq import compute_dockq_metrics
    from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain
    from binding_metrics.metrics.energy import compute_interaction_energy
    from binding_metrics.metrics.evobind import (
        compute_evobind_adversarial_check,
        compute_evobind_score,
    )
    from binding_metrics.metrics.geometry import (
        compute_buried_void_volume,
        compute_omega_planarity,
        compute_ramachandran,
        compute_shape_complementarity,
    )
    from binding_metrics.metrics.interface import compute_interface_metrics
    from binding_metrics.metrics.openfold import compute_openfold_metrics, run_openfold
    from binding_metrics.metrics.polar_contacts import compute_hbonds, compute_saltbridges
    from binding_metrics.metrics.rmsd import compute_receptor_drift
    from binding_metrics.metrics.sasa import compute_delta_sasa_static
    from binding_metrics.protocols.base import ProtocolResults
    from binding_metrics.protocols.peptide import PeptideBindingProtocol
    from binding_metrics.protocols.relaxation import (
        ImplicitRelaxation,
        RelaxationConfig,
        RelaxationResult,
    )
    from binding_metrics.protocols.relaxer import Relaxer

# pyproject.toml is the one place the version is written; an installed package reads
# it back from its metadata. A source tree that was never installed has no metadata.
try:
    __version__ = importlib.metadata.version("binding-metrics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    # Core
    "ForceFieldConfig",
    "get_forcefield",
    "MDSimulation",
    "SimulationConfig",
    "run_simulation",
    "prepare_system",
    # I/O
    "load_structure",
    "detect_chains",
    "save_cif",
    # Protocols
    "ProtocolResults",
    "PeptideBindingProtocol",
    "ImplicitRelaxation",
    "RelaxationConfig",
    "RelaxationResult",
    "Relaxer",
    # Pipeline API
    "run_pipeline",
    "run_batch",
    # Metrics
    "compute_interaction_energy",
    "compute_structure_rmsd",
    "compute_dockq_metrics",
    "compute_hbonds",
    "compute_saltbridges",
    "compute_interface_metrics",
    "compute_openfold_metrics",
    "run_openfold",
    "compute_delta_sasa_static",
    # EvoBind metrics
    "compute_evobind_score",
    "compute_evobind_adversarial_check",
    # New metrics
    "compute_coulomb_cross_chain",
    "compute_ramachandran",
    "compute_omega_planarity",
    "compute_shape_complementarity",
    "compute_buried_void_volume",
    "compute_receptor_drift",
]
