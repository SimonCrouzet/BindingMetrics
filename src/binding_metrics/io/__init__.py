"""I/O utilities for structure handling.

Names load on first access (PEP 562); see ``binding_metrics``.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "get_chain_atom_indices": "binding_metrics.io.structures",
    "load_complex": "binding_metrics.io.structures",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.io.structures import get_chain_atom_indices, load_complex

__all__ = ["load_complex", "get_chain_atom_indices"]
