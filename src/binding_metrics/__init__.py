"""BindingMetrics: GPU-compatible binding metrics evaluation through MD simulations."""

from binding_metrics.core.forcefields import ForceFieldConfig, get_forcefield
from binding_metrics.core.simulation import MDSimulation, SimulationConfig, run_simulation
from binding_metrics.core.system import prepare_system
from binding_metrics.io.structures import detect_chains, load_structure, save_cif
from binding_metrics.metrics.comparison import compute_structure_rmsd
from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain
from binding_metrics.metrics.energy import compute_interaction_energy
from binding_metrics.metrics.geometry import (
    compute_buried_void_volume,
    compute_omega_planarity,
    compute_ramachandran,
    compute_shape_complementarity,
)
from binding_metrics.metrics.hbonds import compute_hbonds, compute_saltbridges
from binding_metrics.metrics.interface import compute_interface_metrics
from binding_metrics.metrics.openfold import compute_openfold_metrics, run_openfold
from binding_metrics.metrics.rmsd import compute_receptor_drift
from binding_metrics.metrics.sasa import compute_delta_sasa_static
from binding_metrics.protocols.base import ProtocolResults
from binding_metrics.protocols.peptide import PeptideBindingProtocol
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig, RelaxationResult

__version__ = "0.1.0"

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
    # Metrics
    "compute_interaction_energy",
    "compute_structure_rmsd",
    "compute_hbonds",
    "compute_saltbridges",
    "compute_interface_metrics",
    "compute_openfold_metrics",
    "run_openfold",
    "compute_delta_sasa_static",
    # New metrics
    "compute_coulomb_cross_chain",
    "compute_ramachandran",
    "compute_omega_planarity",
    "compute_shape_complementarity",
    "compute_buried_void_volume",
    "compute_receptor_drift",
]
