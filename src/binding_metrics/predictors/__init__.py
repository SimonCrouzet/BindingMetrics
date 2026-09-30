"""Readers for the output of structure-prediction models.

Each supported model gets an adapter (``PredictionParser``) that turns its files into one
neutral ``PredictionRecord``, so the confidence metrics and the EvoBind check need no
per-model code. Adapters read output only and never run a model. Names load on first
access (PEP 562); importing this package imports nothing heavy, and biotite is imported only
when a structure is read.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "PredictionFiles": "binding_metrics.predictors.record",
    "PredictionRecord": "binding_metrics.predictors.record",
    "SampleRef": "binding_metrics.predictors.record",
    "TokenLayout": "binding_metrics.predictors.record",
    "PredictionParser": "binding_metrics.predictors.base",
    "PARSERS": "binding_metrics.predictors.registry",
    "ParserSpec": "binding_metrics.predictors.registry",
    "get_parser": "binding_metrics.predictors.registry",
    "register_parser": "binding_metrics.predictors.registry",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.predictors.base import PredictionParser
    from binding_metrics.predictors.record import (
        PredictionFiles,
        PredictionRecord,
        SampleRef,
        TokenLayout,
    )
    from binding_metrics.predictors.registry import (
        PARSERS,
        ParserSpec,
        get_parser,
        register_parser,
    )

__all__ = [
    "PARSERS",
    "ParserSpec",
    "PredictionFiles",
    "PredictionParser",
    "PredictionRecord",
    "SampleRef",
    "TokenLayout",
    "get_parser",
    "register_parser",
]
