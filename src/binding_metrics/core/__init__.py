"""Core simulation engine components.

Names load on first access (PEP 562); see ``binding_metrics``. The configuration and
non-standard residue helpers need no OpenMM; ``MDSimulation`` and ``prepare_system`` do.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "CyclicBondInfo": "binding_metrics.core.cyclic",
    "CyclizationError": "binding_metrics.core.cyclic",
    "get_addh_variants": "binding_metrics.core.cyclic",
    "ForceFieldConfig": "binding_metrics.core.forcefields",
    "get_forcefield": "binding_metrics.core.forcefields",
    "D_AA_MAP": "binding_metrics.core.nonstandard",
    "NME_AA_MAP": "binding_metrics.core.nonstandard",
    "NonstandardInfo": "binding_metrics.core.nonstandard",
    "MDSimulation": "binding_metrics.core.simulation",
    "SimulationConfig": "binding_metrics.core.simulation",
    "prepare_system": "binding_metrics.core.system",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.core.cyclic import CyclicBondInfo, CyclizationError, get_addh_variants
    from binding_metrics.core.forcefields import ForceFieldConfig, get_forcefield
    from binding_metrics.core.nonstandard import D_AA_MAP, NME_AA_MAP, NonstandardInfo
    from binding_metrics.core.simulation import MDSimulation, SimulationConfig
    from binding_metrics.core.system import prepare_system

__all__ = [
    "CyclicBondInfo",
    "CyclizationError",
    "D_AA_MAP",
    "ForceFieldConfig",
    "get_addh_variants",
    "get_forcefield",
    "MDSimulation",
    "NME_AA_MAP",
    "NonstandardInfo",
    "SimulationConfig",
    "prepare_system",
]
