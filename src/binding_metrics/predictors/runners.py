"""The contract of a prediction runner: what starts a model for the prediction store.

A runner turns a ``binding_metrics.predictors.store.PredictionRequest`` into the files the
model writes and never reads them (the parser of the same model does that). It is separate from
the parser on purpose: parsing needs no model installed, running does. The store
(``PredictionStore``) is the only caller in the pipeline; metrics never call a runner.

Public names and signatures (this module imports the standard library only)::

    class PredictionRunner(ABC):
        name: ClassVar[str]                    # the parser name of the same model, "of3"
        capabilities: ClassVar[Optional[Any]] = None
        supports_custom_weights: ClassVar[bool] = False
        weights_kind: ClassVar[str] = "file"   # "file" or "directory"
        @abstractmethod
        def prepare(self, request: PredictionRequest, work_dir: Path) -> Path
        @abstractmethod
        def run(self, request: PredictionRequest, work_dir: Path) -> Path
        def supports_batch(self, request: PredictionRequest) -> bool          # False
        def run_many(self, requests: Sequence[PredictionRequest], work_dir: Path)
                -> dict[str, Union[Path, BaseException]]                      # raises
        def is_available(self) -> bool                                        # True
        def version(self) -> Optional[str]                                    # None
        def check_weights(self, request: PredictionRequest) -> None           # raises ValueError

    require_weights_support(runner, request) -> None     # what the store calls; any runner object

How the ``PredictionRunner`` of the design note (section B) maps here: the note listed the
arguments of a query (structure file, chains, name, seeds); they all live in the request now, so
``prepare`` and ``run`` take the request, and settings that belong to the machine rather than to
the prediction (a conda environment) are arguments of the runner's constructor.

Contract, checked for the stub runners of ``tests/predictors/test_runners.py``:

* ``run(request, work_dir)`` writes everything it produces below ``work_dir``, which exists and
  is empty, and returns the directory (``work_dir`` itself or one inside it) that the parser of
  ``request.model`` loads with ``parser.load(directory, request.name)``. The store keeps the
  whole ``work_dir`` and remembers where that directory is. A runner that returns a path outside
  ``work_dir`` gets its run recorded as failed. ``work_dir`` is a temporary directory that is
  renamed when the run is stored, so nothing may depend on its path after ``run`` returns.
* A failure raises; the exception text becomes the reason recorded in the store, so it should
  say what went wrong and what to do (``OpenFoldRunError`` does). A run that exits normally but
  wrote no output is a failure too: raise, so that no empty result is stored as done.
* ``prepare(request, work_dir)`` writes the model's input files below ``work_dir`` and returns
  the main one, raising the errors that ``run`` would raise about the input (for example an
  unmappable residue) without starting the model. The store does not call it; a pre-flight check
  or a dry run may.
* ``supports_batch(request)`` and ``run_many(requests, work_dir)`` are optional. A runner that
  can predict several requests in one process (one model load) says so per request, and
  ``run_many`` returns, for each request key, the directory that holds only that request's
  output (inside ``work_dir``, not shared with another request), or an exception instance
  explaining why that request has no output. The requests it receives share ``model``, ``mode``,
  ``seeds``, ``num_samples``, ``options`` and the hashes of their extra files
  (``PredictionRequest.batch_signature``) and have distinct ``name`` values. An exception it
  raises fails the whole batch.
* ``is_available()`` is a cheap check that the model can start on this machine. The store asks
  it before a run and refuses without recording a failure, so a missing installation is not
  remembered as a failed prediction.
* ``version()`` is the model's version string, or None when it cannot be told. It is part of the
  request key, so predictions of two versions never share a store entry.
* Custom weights (a fine-tuned model). ``request.weights`` is a ``WeightsRef`` (content hash in
  the key, see ``binding_metrics.predictors.weights``) or None. A runner that can start its
  model with weights the user chooses sets ``supports_custom_weights = True`` and ``weights_kind``
  to ``"file"`` (a checkpoint file) or ``"directory"``, and passes ``request.weights.path`` to
  the model. The default is False: a request with weights for such a runner is refused with a
  ``ValueError`` that names the model, in ``PredictionStore.ensure``/``get_or_run``/``run_missing``
  before anything runs (through ``require_weights_support``, which also works for a runner object
  that does not subclass this class), and a runner's ``prepare`` and ``run`` call
  ``self.check_weights(request)`` so that a direct call is refused too. A runner that ignored the
  weights would run the default model and store the result under the key of the custom one.
* ``capabilities`` stays None on a runner. The limits of a model (the input classes it cannot
  handle) live on ``PredictionParser.capabilities``, which is what ``preflight`` reads, so
  ``OpenFold3Runner.capabilities`` is None while ``OpenFold3Parser.capabilities`` carries the
  limits. The attribute exists so that a runner can declare a limit of its own, such as one
  that depends on how the model is started.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence, Union

if TYPE_CHECKING:
    from binding_metrics.predictors.store import PredictionRequest


class PredictionRunner(ABC):
    """Starts one structure-prediction model. Subclasses set ``name`` and implement two steps.

    A runner holds machine settings (a conda environment name, an executable path) but no
    per-prediction state, so one instance can serve any number of requests.
    """

    #: Name of the model; equal to the name of its ``PredictionParser`` (``"of3"``) and to
    #: ``PredictionRequest.model``. The store refuses a request for another model.
    name: ClassVar[str]
    #: A limit of the runner itself, as a ``binding_metrics.capabilities.Capabilities``; None
    #: (the value on every runner) declares none. The limits of the model are on the parser.
    capabilities: ClassVar[Optional[Any]] = None
    #: True when the runner starts its model with ``request.weights`` (see the module docstring).
    supports_custom_weights: ClassVar[bool] = False
    #: How the model takes its weights when ``supports_custom_weights``: ``"file"`` (one
    #: checkpoint) or ``"directory"``.
    weights_kind: ClassVar[str] = "file"

    @abstractmethod
    def prepare(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Write the model's input files for ``request`` below ``work_dir``; return the main one."""

    @abstractmethod
    def run(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Run the model and return the directory its parser loads (see the module docstring)."""

    def supports_batch(self, request: PredictionRequest) -> bool:
        """True when ``request`` can be predicted inside ``run_many``. The default says no."""
        return False

    def run_many(
        self, requests: Sequence[PredictionRequest], work_dir: Path
    ) -> dict[str, Union[Path, BaseException]]:
        """Predict several requests in one process; see the module docstring for the result.

        Raises:
            NotImplementedError: The runner has no batched mode (``supports_batch`` is False
                for every request).
        """
        name = getattr(self, "name", type(self).__name__)
        raise NotImplementedError(f"the {name} runner has no batched mode")

    def check_weights(self, request: PredictionRequest) -> None:
        """Raise ``ValueError`` when ``request.weights`` is set and this runner cannot use it.

        Call it first in ``prepare`` and ``run``. The store calls ``require_weights_support``
        before a run, so a runner started through the store never sees such a request.
        """
        require_weights_support(self, request)

    def is_available(self) -> bool:
        """True when the model can be started here. The default assumes it can."""
        return True

    def version(self) -> Optional[str]:
        """The installed model's version string, or None when it cannot be told."""
        return None


def require_weights_support(runner: Any, request: PredictionRequest) -> None:
    """Raise ``ValueError`` when ``request`` has weights that ``runner`` cannot use.

    A runner object need not subclass ``PredictionRunner``: a missing ``supports_custom_weights``
    counts as False and a missing ``weights_kind`` as ``"file"``.

    Raises:
        ValueError: The runner does not support custom weights (the message names the model), or
            takes them as another kind than the request holds (a file or a directory).
    """
    weights = getattr(request, "weights", None)
    if weights is None:
        return
    model = getattr(runner, "name", None) or getattr(request, "model", "this model")
    if not getattr(runner, "supports_custom_weights", False):
        raise ValueError(
            f"the {model} runner does not take custom weights: it starts the model with its own "
            f"weights, so a run would ignore {weights.path} and the result would be stored under "
            "the key of the custom weights. Leave the weights out, or use a runner with "
            "supports_custom_weights"
        )
    wanted = getattr(runner, "weights_kind", "file")
    if weights.kind != wanted:
        raise ValueError(
            f"the {model} runner takes its weights as a {wanted}, and {weights.path} is a "
            f"{weights.kind}"
        )
