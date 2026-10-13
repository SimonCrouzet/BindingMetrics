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
        supported_modes: ClassVar[frozenset[str]] = frozenset({"predict"})
        default_mode: ClassVar[Optional[str]] = None
        seed_dimension: ClassVar[bool] = False  # one request can carry several seeds
        supports_given_msa: ClassVar[bool] = True   # receptor_msa=PATH is accepted
        supports_single: ClassVar[bool] = True      # receptor_msa="single" is accepted
        msa_conflicts_with_server: ClassVar[bool] = False  # a receptor_msa needs the server off
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
        def output_chain_map(self, request: PredictionRequest) -> Optional[dict[str, str]]  # None
        def make_requests(self, input_path, *, seeds: Sequence[int], **options)
                -> list[PredictionRequest]                                    # see "Seeds"
        def _group_seeds(self, seeds: Sequence[int]) -> list[tuple[int, ...]]

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
* Modes. ``supported_modes`` is the set of modes (``binding_metrics.predictors.store.MODES``) the
  runner can run when ``request.input_path`` is a complex structure, which is what the command
  line gives it: ``predict`` (the model folds the sequences), ``refold`` (the receptor is given
  as template), ``score`` (every chain is given its own structure as template), ``score-lock``
  (``score`` with the pose pinned) and ``refold-lock`` (``refold`` with the receptor pinned by a
  forced template and the binder free). The default is ``{"predict"}``, the one mode that needs
  nothing but sequences. ``default_mode`` is the mode used when the caller names none and must
  be in ``supported_modes``; None (the default) means the default of the ``mode`` parameter of
  the runner's ``make_request``. A request for a mode outside ``supported_modes`` raises
  ``ValueError`` in ``make_request``, and ``binding_metrics.cli.prediction`` refuses it while the
  command line is checked, with the modes the runner has. A runner that has not declared
  ``supported_modes`` but names the modes it builds an input for ``run_modes``
  (``Boltz2Runner``) is read through that name.
* ``output_chain_map(request)`` is the chain IDs of the prediction when they are not the input's:
  ``{chain ID in the prediction: chain ID in the input}``, or None (the default) when the
  prediction keeps the input's IDs. A reader that wants the input's IDs on the atoms passes the
  map as ``chain_map`` to ``PredictionSession.record``; ``ColabFoldRunner`` gives the receptor
  the ID ``A`` and the binder ``B`` whatever their IDs in the input.
* Seeds. ``make_requests(input_path, *, seeds, **options)`` turns a list of seeds into the
  requests that run them, calling the runner's own ``make_request`` (``options`` are its
  keyword arguments, ``name`` included). A runner whose model takes several seeds in one run
  (OpenFold3, Protenix) sets ``seed_dimension = True`` and gets one request that carries all
  the seeds; the default, False, is a model that runs one seed at a time (Boltz-2), and gets one
  request per seed. A request that carries one seed of a runner without the dimension is named
  ``f"{name}_s{seed}"``, so that the output directories of the requests of one call do not
  collide; a request that carries several seeds keeps ``name``. ``_group_seeds(seeds)`` decides
  which seeds share a request and returns the groups in the order of ``seeds``: a runner
  overrides it when its model takes only some lists (ColabFold runs consecutive seeds from a
  first value). ``seeds`` must be a non-empty sequence of distinct non-negative integers. The
  seeds of a request are part of its key, so ``make_requests`` changes no key: the request of
  one seed is the request that ``make_request(..., seeds=[seed])`` builds.
* The receptor MSA. A runner whose ``make_request`` has a ``receptor_msa`` keyword says what it
  accepts with the class attributes ``supports_given_msa`` (the path of an alignment file) and
  ``supports_single`` (the word ``"single"``, a receptor predicted from its own sequence); both
  are True by default. They are the two flags the runner gives ``_msa.check_msa_request``, and
  the command line reads them to refuse a ``--prediction-receptor-msa`` the runner cannot take
  before anything runs (``ColabFoldRunner`` accepts ``"single"`` only).
  ``msa_conflicts_with_server`` (False by default) is True for a runner whose model ignores the
  alignment while its MSA server is on and so refuses ``receptor_msa`` with
  ``use_msa_server=True`` (``OpenFold3Runner``); the command line refuses that pair too, naming
  ``--prediction-no-msa-server``.
* ``capabilities`` stays None on a runner. The limits of a model (the input classes it cannot
  handle) live on ``PredictionParser.capabilities``, which is what ``preflight`` reads, so
  ``OpenFold3Runner.capabilities`` is None while ``OpenFold3Parser.capabilities`` carries the
  limits. The attribute exists so that a runner can declare a limit of its own, such as one
  that depends on how the model is started.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Set
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Optional, Sequence, Union

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
    #: The modes the runner can run for a complex structure (see the module docstring).
    supported_modes: ClassVar[frozenset[str]] = frozenset({"predict"})
    #: The mode used when the caller names none, one of ``supported_modes``; None takes the
    #: default of the ``mode`` parameter of ``make_request``.
    default_mode: ClassVar[Optional[str]] = None
    #: True when one request can carry several seeds (the model runs them in one process and
    #: writes a seed dimension); False when one run takes one seed (see ``make_requests``).
    seed_dimension: ClassVar[bool] = False
    #: True when ``make_request(receptor_msa=PATH)`` can give an alignment file to the model.
    supports_given_msa: ClassVar[bool] = True
    #: True when ``make_request(receptor_msa="single")`` can run a receptor without an alignment.
    supports_single: ClassVar[bool] = True
    #: True when ``make_request`` refuses a ``receptor_msa`` together with ``use_msa_server=True``.
    msa_conflicts_with_server: ClassVar[bool] = False

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

    def output_chain_map(self, request: PredictionRequest) -> Optional[dict[str, str]]:
        """Chain ID in the prediction to chain ID in the input, or None when they are the same.

        The default says the prediction keeps the input's chain IDs. A runner whose model names
        the chains itself overrides it (``ColabFoldRunner``). The pipeline passes the map to the
        reader as ``chain_map``, so that the metrics find the chains under the IDs of the input.
        """
        return None

    def make_requests(
        self, input_path: str | Path, *, seeds: Sequence[int], **options: Any
    ) -> list[PredictionRequest]:
        """The requests that run ``seeds`` of one input: one for all, or one per seed.

        ``options`` are the keyword arguments of this runner's ``make_request`` (``name`` and
        the chain roles, for the four runners of this package), given to every request. The
        seeds are grouped by ``_group_seeds``: all in one request when ``seed_dimension`` is
        True, one request per seed otherwise. A request that carries a single seed of a runner
        without the dimension is named ``f"{name}_s{seed}"``, so the output directories of the
        requests of one call differ (Boltz-2 names its folder after the input); any other
        request keeps ``name``. Nothing is run and no key changes: the request of one seed
        equals ``make_request(input_path, seeds=[seed], **options)`` except for its name, which
        is not part of the key.

        Args:
            input_path: The input file, as for ``make_request``.
            seeds: Distinct non-negative integers, at least one.
            **options: The other arguments of ``make_request``.

        Returns:
            The requests, in the order of the groups.

        Raises:
            ValueError: ``seeds`` is empty, repeats a value, holds a negative or a non-integer
                value, or whatever ``make_request`` raises for the options.
            TypeError: ``seeds`` is a string, or not a sequence.
            NotImplementedError: The runner defines no ``make_request``.
        """
        runner_name = getattr(self, "name", type(self).__name__)
        make_request = getattr(self, "make_request", None)
        if make_request is None:
            raise NotImplementedError(f"the {runner_name} runner defines no make_request")
        values = _checked_seeds(seeds)
        groups = [tuple(group) for group in self._group_seeds(values)]
        if [seed for group in groups for seed in group] != values or not all(groups):
            raise ValueError(
                f"the {runner_name} runner grouped the seeds {values} as {groups}, which does "
                "not cover each seed once, in order"
            )
        name = options.get("name")
        requests = []
        for group in groups:
            group_options = dict(options)
            if isinstance(name, str) and len(group) == 1 and not self.seed_dimension:
                group_options["name"] = f"{name}_s{group[0]}"
            requests.append(make_request(input_path, seeds=list(group), **group_options))
        return requests

    def _group_seeds(self, seeds: Sequence[int]) -> list[tuple[int, ...]]:
        """Which seeds share a request: the groups, covering ``seeds`` once, in order.

        The default is one group of all the seeds when ``seed_dimension`` is True and one group
        per seed otherwise. A runner whose model runs only some lists of seeds in one process
        overrides it (ColabFold: consecutive seeds in one group, any other seed on its own).
        """
        if self.seed_dimension:
            return [tuple(seeds)]
        return [(seed,) for seed in seeds]

    def is_available(self) -> bool:
        """True when the model can be started here. The default assumes it can."""
        return True

    def version(self) -> Optional[str]:
        """The installed model's version string, or None when it cannot be told."""
        return None


def _checked_seeds(seeds: Any) -> list[int]:
    """``seeds`` as a list of distinct non-negative integers; raises on anything else."""
    if isinstance(seeds, (str, bytes, Mapping, Set)) or not isinstance(seeds, Iterable):
        raise TypeError(
            f"seeds must be an ordered sequence of integers, got {type(seeds).__name__}"
        )
    values = list(seeds)
    if not values:
        raise ValueError("seeds must name at least one seed")
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError(f"seeds must be integers, got {seed!r}")
    values = [int(seed) for seed in values]
    negative = sorted({seed for seed in values if seed < 0})
    if negative:
        raise ValueError(f"seeds must not be negative, got {negative} in {values}")
    repeated = sorted({seed for seed in values if values.count(seed) > 1})
    if repeated:
        raise ValueError(
            f"seeds must be distinct, but {repeated} appear more than once in {values}: a "
            "repeated seed would be the same prediction twice"
        )
    return values


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
