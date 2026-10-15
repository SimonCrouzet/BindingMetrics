"""Binding metrics calculations.

Names load on first access (PEP 562), so importing this package or one static metric
module does not import OpenMM. See ``binding_metrics``.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "calculate_contacts": "binding_metrics.metrics.contacts",
    "compute_coulomb_cross_chain": "binding_metrics.metrics.electrostatics",
    "calculate_interaction_energy": "binding_metrics.metrics.energy",
    "compute_evobind_adversarial_check": "binding_metrics.metrics.evobind",
    "compute_evobind_score": "binding_metrics.metrics.evobind",
    "compute_buried_void_volume": "binding_metrics.metrics.geometry",
    "compute_omega_planarity": "binding_metrics.metrics.geometry",
    "compute_ramachandran": "binding_metrics.metrics.geometry",
    "compute_shape_complementarity": "binding_metrics.metrics.geometry",
    "compute_interface_pae": "binding_metrics.metrics.openfold",
    "compute_openfold_metrics": "binding_metrics.metrics.openfold",
    "prepare_refolding_query": "binding_metrics.metrics.openfold",
    "prepare_scoring_query": "binding_metrics.metrics.openfold",
    "run_openfold": "binding_metrics.metrics.openfold",
    "run_openfold_refolding": "binding_metrics.metrics.openfold",
    "run_openfold_scoring": "binding_metrics.metrics.openfold",
    "compute_prediction_metrics": "binding_metrics.metrics.prediction",
    "compute_receptor_quality": "binding_metrics.metrics.receptor_quality",
    "calculate_rmsd": "binding_metrics.metrics.rmsd",
    "compute_receptor_drift": "binding_metrics.metrics.rmsd",
    "calculate_buried_sasa": "binding_metrics.metrics.sasa",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.metrics.contacts import calculate_contacts
    from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain
    from binding_metrics.metrics.energy import calculate_interaction_energy
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
    from binding_metrics.metrics.openfold import (
        compute_interface_pae,
        compute_openfold_metrics,
        prepare_refolding_query,
        prepare_scoring_query,
        run_openfold,
        run_openfold_refolding,
        run_openfold_scoring,
    )
    from binding_metrics.metrics.prediction import compute_prediction_metrics
    from binding_metrics.metrics.receptor_quality import compute_receptor_quality
    from binding_metrics.metrics.rmsd import calculate_rmsd, compute_receptor_drift
    from binding_metrics.metrics.sasa import calculate_buried_sasa

__all__ = [
    "compute_evobind_score",
    "compute_evobind_adversarial_check",
    "calculate_buried_sasa",
    "calculate_contacts",
    "calculate_interaction_energy",
    "calculate_rmsd",
    "compute_buried_void_volume",
    "compute_coulomb_cross_chain",
    "compute_omega_planarity",
    "compute_interface_pae",
    "compute_openfold_metrics",
    "compute_prediction_metrics",
    "compute_ramachandran",
    "compute_receptor_drift",
    "compute_receptor_quality",
    "compute_shape_complementarity",
    "prepare_refolding_query",
    "prepare_scoring_query",
    "run_openfold",
    "run_openfold_refolding",
    "run_openfold_scoring",
]
