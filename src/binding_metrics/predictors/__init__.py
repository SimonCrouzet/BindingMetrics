"""Readers for the output of structure-prediction models, and the store that runs each once.

Each supported model gets an adapter (``PredictionParser``) that turns its files into one
neutral ``PredictionRecord``, so the confidence metrics and the EvoBind check need no
per-model code. Adapters read output only and never run a model.

A run-once store sits beside them: ``PredictionRequest`` describes a prediction,
``PredictionStore`` keeps finished ones under a key of that description, ``PredictionSession``
is what a pipeline asks for a parsed record (running the model on the first miss only), and a
``PredictionRunner`` starts the model: ``OpenFold3Runner``, ``ColabFoldRunner`` (AlphaFold2),
``Boltz2Runner`` and ``ProtenixRunner``.

Names load on first access (PEP 562); importing this package imports nothing heavy, and
biotite is imported only when a structure is read.
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
    "PredictionRunner": "binding_metrics.predictors.runners",
    "OpenFold3Runner": "binding_metrics.predictors.of3_runner",
    "ColabFoldRunner": "binding_metrics.predictors.af2_runner",
    "Boltz2Runner": "binding_metrics.predictors.boltz2_runner",
    "ProtenixRunner": "binding_metrics.predictors.protenix_runner",
    "PredictionRequest": "binding_metrics.predictors.store",
    "StoredPrediction": "binding_metrics.predictors.store",
    "PredictionStore": "binding_metrics.predictors.store",
    "PredictionFailedError": "binding_metrics.predictors.store",
    "PredictionUnavailableError": "binding_metrics.predictors.store",
    "PredictionSession": "binding_metrics.predictors.session",
    "PARSERS": "binding_metrics.predictors.registry",
    "ParserSpec": "binding_metrics.predictors.registry",
    "get_parser": "binding_metrics.predictors.registry",
    "register_parser": "binding_metrics.predictors.registry",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.predictors.af2_runner import ColabFoldRunner
    from binding_metrics.predictors.base import PredictionParser
    from binding_metrics.predictors.boltz2_runner import Boltz2Runner
    from binding_metrics.predictors.of3_runner import OpenFold3Runner
    from binding_metrics.predictors.protenix_runner import ProtenixRunner
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
    from binding_metrics.predictors.runners import PredictionRunner
    from binding_metrics.predictors.session import PredictionSession
    from binding_metrics.predictors.store import (
        PredictionFailedError,
        PredictionRequest,
        PredictionStore,
        PredictionUnavailableError,
        StoredPrediction,
    )

__all__ = [
    "Boltz2Runner",
    "ColabFoldRunner",
    "OpenFold3Runner",
    "PARSERS",
    "ParserSpec",
    "PredictionFailedError",
    "PredictionFiles",
    "PredictionParser",
    "PredictionRecord",
    "PredictionRequest",
    "PredictionRunner",
    "PredictionSession",
    "PredictionStore",
    "PredictionUnavailableError",
    "ProtenixRunner",
    "SampleRef",
    "StoredPrediction",
    "TokenLayout",
    "get_parser",
    "register_parser",
]
