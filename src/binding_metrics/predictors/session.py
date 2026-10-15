"""The per-sample handle that metrics and pipeline steps use to get a parsed prediction.

A ``PredictionSession`` joins a ``PredictionStore``, the runners and the parsers. A metric asks
the session for the record of a ``PredictionRequest``; the session runs the model only on the
first miss, parses the files once and remembers both, so two metrics of one process never touch
the disk twice and nothing calls a runner directly.

Public names and signatures (the module imports the standard library only; the parsers import
numpy when they are used)::

    class PredictionSession:
        PredictionSession(store: PredictionStore,
            runners: Optional[Mapping[str, PredictionRunner] | Iterable[PredictionRunner]] = None,
            parsers: Optional[Mapping[str, PredictionParser]] = None, *, rerun: bool = False)
        .store, .rerun
        .entry(request) -> StoredPrediction
        .record(request, *, seed_index: int = 1, sample: int = 1,
                chain_map: Optional[Mapping[str, str]] = None) -> PredictedPose
        .records(request, *, chain_map: Optional[Mapping[str, str]] = None)
                -> list[PredictedPose]
        .lookup(request) -> Optional[StoredPrediction]
        .prefetch(requests: Iterable[PredictionRequest]) -> None
        .adopt(request, directory, *, copy_outputs: bool = False, check: bool = True)
                -> StoredPrediction
        .stats() -> dict[str, int]

    receptor_forced(request) -> Optional[bool]      # was the receptor pinned by a forced template
    receptor_templated(request) -> Optional[bool]   # was the receptor given as a template

What a pipeline calls, in this order: build the requests (``OpenFold3Runner.make_request``),
``prefetch`` them all when a batch of samples is about to be processed (one batched model run
for the missing ones), and ``record`` for every metric that needs the prediction. ``records``
returns every sample of one request in the model's natural order, and ``lookup`` says what the
store holds for a request without running anything (``binding_metrics.predictors.manifest``
lists the samples of a set of requests with it). Parse-only
usage (``--predictor X --prediction-dir D``) calls ``adopt`` first; ``record`` then treats the
directory exactly like a run.

``runners`` maps a model name to its runner (a list of runners is keyed by their ``name``);
a model without a runner can be served from what the store already holds. ``parsers`` maps a
model name to a ``PredictionParser`` and defaults to the registry (``get_parser``). ``rerun``
runs each request once even when the store has an entry (``--rerun-predictions``); asking again
in the same session returns the fresh entry. Outputs adopted through the session are exempt: a
rerun never replaces what the user pointed at.

The record is parsed from the directory of the stored entry with the name the outputs were
written under. For a model run that is the name of the request that produced it, so ``record.name``
is that name when a second request with another name but the same content hits the entry (the
model ran once for both). Adopted outputs belong to a name: two samples with one input file each
get their own record. The session memory tells requests apart by key and name. The returned
record is shared between callers: treat it as read-only.

How the prediction was requested. ``record`` adds the facts of the request to
``record.provenance`` (JSON scalars; other keys that the record carries are kept)::

    backend              the model (``request.model``)
    mode                 ``request.mode``
    receptor_forced      ``receptor_forced(request)``: the receptor was pinned by a forced template
    receptor_templated   ``receptor_templated(request)``: the receptor was given as a template
    seed                 the seed value of the sample (None when it cannot be told)
    seed_index, sample   the 1-based positions of the sample
    msa_policy, msa_receptor, msa_binder
                         only when the request names an MSA policy or a receptor MSA
    embeddings_path      only when the request asked for embeddings (Boltz-2 write_embeddings)
                         and the file exists: the path of embeddings_<name>.npz in the store
    request_key          ``request.key()``
    model_version        ``request.model_version``, else the version the runner reported
    request_name         ``request.name``

``seed`` is the number the model sampled with. The adapters that read it from the output
(OpenFold3 and Protenix from the ``seed_<value>`` directory, ColabFold from the ``seed_<value>``
of the file name) put it in ``extras["seed_value"]``, and that is what is used. For the others
(Boltz-2 has one seed per request) it is ``request.seeds[seed_index - 1]``, but only when the
request's seeds are strictly increasing, because every adapter counts seeds in the numeric order
of their values. Otherwise ``seed`` is None and ``record.reasons`` says "seed value unknown".

``stats()`` counts, since the session was made (all integers, ready for a JSON result)::

    requests    asks: calls of ``entry`` and ``record``, and each request of ``prefetch``
    memo_hits   answered from this session's memory, with no disk access
    hits        a finished run was found in the store (an earlier process or session made it,
                or a peer finished it while this one waited for the lock)
    adopted     adopted outputs were found
    misses      nothing usable was stored: the model was run, or a run was forced or attempted
    runs        predictions the model computed for this session (a batched run of five counts
                five; a failed run counts)
    failed      requests whose entry is a recorded failure, counted once each
    parsed      records parsed from files

``requests == memo_hits + hits + adopted + misses`` always. "This model ran once" reads as
``runs == 1``.

Thread safety: one lock per request key serialises the work of two threads that ask for the
same request; different requests run in parallel. Processes are protected by the store's locks.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping as _MappingABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from binding_metrics.predictors.store import (
    STATUS_ADOPTED,
    STATUS_FAILED,
    PredictionRequest,
    PredictionStore,
    PredictionUnavailableError,
    StoredPrediction,
)

if TYPE_CHECKING:
    from binding_metrics.predictors.base import PredictionParser
    from binding_metrics.predictors.record import PredictedPose
    from binding_metrics.predictors.runners import PredictionRunner

_COUNTERS = (
    "requests",
    "memo_hits",
    "hits",
    "adopted",
    "misses",
    "runs",
    "failed",
    "parsed",
)


#: What the session tells requests apart by: the request key and the name. A run does not depend
#: on the name and the store shares it, but adopted outputs belong to a name, and so does the
#: record a caller expects back.
_Ident = tuple[str, str]


def _ident(request: PredictionRequest) -> _Ident:
    return (request.key(), request.name)


# ---------------------------------------------------------------------- what the request says

#: The models whose request vocabulary this module can read: the parser names of the four
#: adapters. Any other model gets ``None`` for the facts below, because nothing says what it does.
_KNOWN_MODELS = frozenset({"af2", "boltz2", "of3", "protenix"})

#: The models whose runner writes a forced template (Boltz-2: ``force: true`` with a threshold).
#: The other three cannot pin the receptor: OpenFold3 templates carry one chain's fold,
#: Protenix takes templates as alignments, ColabFold has no per-chain template route.
_FORCING_MODELS = frozenset({"boltz2"})

#: Modes are compared as strings, so a mode that the vocabulary (``store.MODES``) gains later
#: is read here without a change to this table.
_LOCK_MODES = frozenset({"refold-lock", "score-lock"})
_TEMPLATED_MODES = frozenset({"refold", "score", "refold-lock", "score-lock"})
_MODES_WITH_A_FACT = _TEMPLATED_MODES | {"predict"}


def receptor_forced(request: PredictionRequest) -> Optional[bool]:
    """Whether the receptor was pinned to the input by a forced template; derived, not stored.

    True for the modes ``refold-lock`` and ``score-lock`` of a model whose runner forces
    templates (Boltz-2), False for every other mode of the four known models (``af2``,
    ``boltz2``, ``of3``, ``protenix``), including a lock mode of a model that cannot force, and
    None for any other model or a mode this function does not know. It reads ``request.model``
    and ``request.mode`` only, so it adds nothing to the request key.
    """
    model, mode = getattr(request, "model", None), getattr(request, "mode", None)
    if model not in _KNOWN_MODELS or mode not in _MODES_WITH_A_FACT:
        return None
    return model in _FORCING_MODELS and mode in _LOCK_MODES


def receptor_templated(request: PredictionRequest) -> Optional[bool]:
    """Whether the receptor was given to the model as a template (forced or not).

    True for ``refold``, ``score``, ``refold-lock`` and ``score-lock``, False for ``predict``,
    None for a model other than the four known ones or a mode this function does not know. An
    OpenFold3 ``refold`` pose therefore reads ``receptor_forced=False, receptor_templated=True``.
    """
    model, mode = getattr(request, "model", None), getattr(request, "mode", None)
    if model not in _KNOWN_MODELS or mode not in _MODES_WITH_A_FACT:
        return None
    return mode in _TEMPLATED_MODES


def _seed_value(request: PredictionRequest, record: PredictedPose) -> tuple[Optional[int], str]:
    """The seed a sample was made with, and why it is unknown (empty when it is known).

    The output's own seed comes first: OpenFold3, Protenix and ColabFold name it in the
    directory or file name and the adapter reports it in ``extras["seed_value"]``, which is
    right even for seeds the model generated or outputs adopted from elsewhere. Without it the
    request's seeds can be used only when they are strictly increasing: the adapters count seeds
    in the numeric order of their values, so the ``seed_index``-th seed of a request in another
    order is not the sample's. Otherwise the seed is None and the second item says why.
    """
    reported = record.extras.get("seed_value")
    if reported is not None and str(reported).isdecimal():
        return int(reported), ""
    seeds = tuple(request.seeds)
    start = "seed value unknown: the adapter does not report it and "
    if any(later <= earlier for earlier, later in zip(seeds, seeds[1:])):
        return None, start + "the request seeds are not sorted"
    if not 1 <= record.seed_index <= len(seeds):
        return None, start + f"the request has no seed at position {record.seed_index}"
    return seeds[record.seed_index - 1], ""


def _msa_facts(request: PredictionRequest) -> dict[str, Any]:
    """The MSA the request asked for, as flat provenance keys; empty when it asked for none.

    Read from ``options["msa_policy"]`` (present only when it is not ``"default"``),
    ``options["receptor_msa"] == "single"`` and ``extra_files["receptor_msa"]``, the key names
    of the MSA request. The receptor is the first 12 hex digits of the file's content hash, or
    ``"single"``, or None; the binder is always single-sequence. The values are scalars because
    ``PredictedPose.validate`` accepts nothing else in ``provenance``.
    """
    options = request.options
    file_hash = request.content_hashes.get("receptor_msa")
    single = options.get("receptor_msa") == "single"
    if "msa_policy" not in options and file_hash is None and not single:
        return {}
    return {
        "msa_policy": options.get("msa_policy", "default"),
        "msa_receptor": file_hash[:12] if file_hash is not None else ("single" if single else None),
        "msa_binder": "single",
    }


def _embedding_facts(request: PredictionRequest, record: PredictedPose) -> dict[str, Any]:
    """The embeddings file of a request that asked for one, as a provenance key.

    ``options["write_embeddings"]`` is present only when it is True. The adapter puts the path in
    ``record.extras["embeddings_path"]`` when the file exists, so the path follows the store's
    rename; the arrays are never loaded here. A request that asked for embeddings and finds no
    file gets a reason on the record and no key.
    """
    if not request.options.get("write_embeddings"):
        return {}
    path = record.extras.get("embeddings_path")
    if path is None:
        record.reasons.append(
            "the request asked for embeddings (write_embeddings) but no embeddings file was "
            "found beside the structures of the run"
        )
        return {}
    return {"embeddings_path": str(path)}


def _request_facts(
    request: PredictionRequest, entry: StoredPrediction, record: PredictedPose, seed: Optional[int]
) -> dict[str, Any]:
    """The facts of ``request`` for ``record.provenance`` (see the module docstring)."""
    facts: dict[str, Any] = {
        "backend": request.model,
        "mode": request.mode,
        "receptor_forced": receptor_forced(request),
        "receptor_templated": receptor_templated(request),
        "seed": seed,
        "seed_index": record.seed_index,
        "sample": record.sample,
    }
    facts.update(_msa_facts(request))
    facts.update(
        request_key=request.key(),
        model_version=request.model_version or entry.runner_version,
        request_name=request.name,
    )
    return facts


class PredictionSession:
    """Runs each model at most once per request and parses each prediction once."""

    def __init__(
        self,
        store: PredictionStore,
        runners: Optional[Mapping[str, PredictionRunner] | Iterable[PredictionRunner]] = None,
        parsers: Optional[Mapping[str, PredictionParser]] = None,
        *,
        rerun: bool = False,
    ):
        self.store = store
        self.rerun = rerun
        if runners is None:
            self._runners: dict[str, PredictionRunner] = {}
        elif isinstance(runners, _MappingABC):
            self._runners = dict(runners)
        else:
            self._runners = {runner.name: runner for runner in runners}
        self._parsers: dict[str, PredictionParser] = dict(parsers or {})
        self._entries: dict[_Ident, StoredPrediction] = {}
        self._records: dict[tuple, PredictedPose] = {}
        self._counts = dict.fromkeys(_COUNTERS, 0)
        self._guard = threading.Lock()  # counters, memos and the table of key locks
        self._key_locks: dict[_Ident, threading.Lock] = {}
        self._adopted: set[_Ident] = set()  # ``rerun`` does not replace what the user adopted

    # ------------------------------------------------------------------ asking

    def entry(self, request: PredictionRequest) -> StoredPrediction:
        """The finished store entry of ``request``; runs the model on the first miss only.

        Returns:
            The entry (status ``done`` or ``adopted``).

        Raises:
            PredictionFailedError: The run failed, now or earlier (``rerun`` forces a new run
                once per session).
            PredictionUnavailableError: Nothing is stored and no runner can produce it.
            ValueError: A run is needed and a file of the request changed or vanished since the
                request was made (``PredictionRequest.verify_files``); nothing is run.
        """
        key = _ident(request)
        with self._key_lock(key):
            with self._guard:
                self._counts["requests"] += 1
                remembered = self._entries.get(key)
                if remembered is not None:
                    self._counts["memo_hits"] += 1
            if remembered is not None:
                return remembered.require_ok()
            try:
                entry = self.store.ensure(
                    request,
                    self._runners.get(request.model),
                    rerun=self.rerun and key not in self._adopted,
                )
            except PredictionUnavailableError:
                with self._guard:
                    self._counts["misses"] += 1
                raise
            self._remember(key, entry)
        return entry.require_ok()

    def record(
        self,
        request: PredictionRequest,
        *,
        seed_index: int = 1,
        sample: int = 1,
        chain_map: Optional[Mapping[str, str]] = None,
    ) -> PredictedPose:
        """The parsed, completed record of one sample of ``request``; runs and parses at most once.

        The record has been through ``PredictionParser.complete`` of its model, so a consumer
        gets the token layout and chain names that need the structure. A request with custom
        weights puts ``request.weights.to_dict()`` (path, kind, sha256, size, n_files) in
        ``record.extras["weights"]``. ``record.provenance`` gets the facts of the request
        (``backend``, ``mode``, ``receptor_forced``, ``seed``, ``request_key``, ...; see the
        module docstring). This is the one place the pipeline and
        ``binding-metrics-prediction`` read records from.

        Args:
            request: What to predict.
            seed_index, sample: 1-based positions in the model's natural order of outputs.
            chain_map: Model chain ID to user chain ID (see ``PredictionParser.load``).

        Raises:
            PredictionFailedError, PredictionUnavailableError: See ``entry``.
            ValueError: ``chain_map`` is not a valid map.
            KeyError: No parser is registered for the model.
        """
        return self._parsed(request, self.entry(request), seed_index, sample, chain_map)

    def records(
        self,
        request: PredictionRequest,
        *,
        chain_map: Optional[Mapping[str, str]] = None,
    ) -> list[PredictedPose]:
        """Every sample of ``request`` as a parsed, completed record, in the model's natural order.

        The samples come from ``PredictionParser.list_samples`` (seed position, then sample
        position, as the adapter counts them) and each is read with ``record``, so a sample that
        was parsed before is not parsed again and the model runs on the first miss only. The
        records carry the provenance described in the module docstring. A run that wrote no
        sample gives an empty list.

        Args:
            request: What to predict.
            chain_map: Model chain ID to user chain ID, as for ``record``.

        Raises:
            PredictionFailedError, PredictionUnavailableError: See ``entry``.
            ValueError: ``chain_map`` is not a valid map.
            KeyError: No parser is registered for the model.
        """
        return self._parsed_samples(request, self.entry(request), chain_map)

    def lookup(self, request: PredictionRequest) -> Optional[StoredPrediction]:
        """The entry the store (or this session's memory) holds for ``request``; never runs a model.

        Unlike ``entry`` it returns a failed entry instead of raising, returns None when nothing
        is stored instead of running the model or raising ``PredictionUnavailableError``, and
        does not change the counters or the memory of the session. It resolves the request as
        ``entry`` does: outputs adopted for the name first, and a request without a
        ``model_version`` under the version its runner reports (``runner.version()`` is called,
        which may start a subprocess that reads the installed version, never the model).
        """
        with self._guard:
            remembered = self._entries.get(_ident(request))
        if remembered is not None:
            return remembered
        found, _ = self.store._find(request, self._runners.get(request.model))
        return found

    def prefetch(self, requests: Iterable[PredictionRequest]) -> None:
        """Make sure the entries of all ``requests`` exist, in as few model runs as possible.

        The requests that the store has no finished entry for and that share a model go to
        ``PredictionStore.run_missing``, which batches them when the runner can. A failed run
        is remembered, not raised here: ``entry`` and ``record`` raise it for that request.

        Raises:
            PredictionUnavailableError: Something has to run and there is no usable runner.
            ValueError: A request that has to run has a file that changed or vanished since it
                was made; nothing is run, for any request of its model.
        """
        wanted: dict[str, list[PredictionRequest]] = {}
        seen = set()
        with self._guard:
            already = set(self._entries)
            if self.rerun:
                already |= self._adopted  # a rerun never replaces what the user adopted
        for request in requests:
            key = _ident(request)
            if key in already or key in seen:
                continue
            seen.add(key)
            wanted.setdefault(request.model, []).append(request)
        for model, group in wanted.items():
            entries = self.store.run_missing(group, self._runners.get(model), rerun=self.rerun)
            for request, entry in zip(group, entries):
                key = _ident(request)
                with self._guard:
                    self._counts["requests"] += 1
                self._remember(key, entry)

    def adopt(
        self,
        request: PredictionRequest,
        directory: str | Path,
        *,
        copy_outputs: bool = False,
        check: bool = True,
    ) -> StoredPrediction:
        """Register outputs that the user produced, for parse-only usage.

        See ``PredictionStore.adopt``.

        Args:
            request: What the outputs are a prediction of.
            directory: The directory the model's parser loads with ``request.name``.
            copy_outputs: Copy it into the store instead of pointing at it.
            check: Refuse a directory in which the parser finds no file of the sample (seed
                index 1, sample 1), so a wrong path fails here and not as a record of NaN.

        Raises:
            FileNotFoundError: ``directory`` is not a directory.
            ValueError: ``check`` is set and the directory holds no output of the model.
        """
        if check:
            parser = self._parser(request.model)
            if not parser.find_files(Path(directory), request.name).has_output():
                raise ValueError(
                    f"no {parser.display_name} output for '{request.name}' in {directory}: "
                    "check the directory and the prediction name"
                )
        entry = self.store.adopt(request, directory, copy_outputs=copy_outputs)
        key = _ident(request)
        with self._guard:  # what was remembered for this request describes the older entry
            self._adopted.add(key)
            self._entries.pop(key, None)
            for memo_key in [k for k in self._records if k[0] == key]:
                del self._records[memo_key]
        return entry

    def stats(self) -> dict[str, int]:
        """A copy of the counters described in the module docstring."""
        with self._guard:
            return dict(self._counts)

    # ------------------------------------------------------------------ internals

    def _parsed(
        self,
        request: PredictionRequest,
        entry: StoredPrediction,
        seed_index: int,
        sample: int,
        chain_map: Optional[Mapping[str, str]],
    ) -> PredictedPose:
        """The record of one sample of a finished ``entry``, parsed once and then remembered.

        ``record`` and ``records`` get the entry from ``entry`` (which may run the model);
        ``binding_metrics.predictors.manifest`` gets it from ``lookup``, which never does.
        """
        key = _ident(request)
        chains = tuple(sorted((chain_map or {}).items()))
        memo_key = (key, seed_index, sample, chains)
        with self._key_lock(key):
            record = self._records.get(memo_key)
            if record is None:
                parser = self._parser(entry.model)
                record = parser.complete(
                    parser.load(
                        entry.prediction_dir,
                        entry.name,
                        seed_index=seed_index,
                        sample=sample,
                        chain_map=chain_map,
                    )
                )
                if request.weights is not None:
                    # what the run was started with; an adapter may add what the model reports
                    # (OpenFold3: inference_ckpt_path and inference_ckpt_name)
                    record.extras["weights"] = request.weights.to_dict()
                seed, seed_problem = _seed_value(request, record)
                if seed_problem:
                    record.reasons.append(seed_problem)
                record.provenance.update(_request_facts(request, entry, record, seed))
                record.provenance.update(_embedding_facts(request, record))
                with self._guard:
                    self._records[memo_key] = record
                    self._counts["parsed"] += 1
        return record

    def _parsed_samples(
        self,
        request: PredictionRequest,
        entry: StoredPrediction,
        chain_map: Optional[Mapping[str, str]],
    ) -> list[PredictedPose]:
        """All samples of a finished ``entry`` in the adapter's natural order."""
        parser = self._parser(entry.model)
        refs = parser.list_samples(entry.prediction_dir, entry.name)
        return [self._parsed(request, entry, ref.seed_index, ref.sample, chain_map) for ref in refs]

    def _key_lock(self, key: _Ident) -> threading.Lock:
        with self._guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def _parser(self, model: str) -> PredictionParser:
        with self._guard:
            parser = self._parsers.get(model)
            if parser is None:
                from binding_metrics.predictors.registry import get_parser

                parser = self._parsers[model] = get_parser(model)
        return parser

    def _remember(self, key: _Ident, entry: StoredPrediction) -> None:
        """Memoise ``entry`` and count how it was obtained."""
        with self._guard:
            if key in self._entries:
                self._counts["memo_hits"] += 1
                return
            self._entries[key] = entry
            if entry.executed_here:
                self._counts["misses"] += 1
                self._counts["runs"] += 1
            elif entry.status == STATUS_ADOPTED:
                self._counts["adopted"] += 1
            else:
                self._counts["hits"] += 1
            if entry.status == STATUS_FAILED:
                self._counts["failed"] += 1
