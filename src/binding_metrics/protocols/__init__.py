"""Simulation protocols for binding metrics evaluation.

Names load on first access (PEP 562); see ``binding_metrics``.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "BaseProtocol": "binding_metrics.protocols.base",
    "ProtocolResults": "binding_metrics.protocols.base",
    "PeptideBindingProtocol": "binding_metrics.protocols.peptide",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.protocols.base import BaseProtocol, ProtocolResults
    from binding_metrics.protocols.peptide import PeptideBindingProtocol

__all__ = [
    "BaseProtocol",
    "ProtocolResults",
    "PeptideBindingProtocol",
]
