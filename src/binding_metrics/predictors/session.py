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
                chain_map: Optional[Mapping[str, str]] = None) -> PredictionRecord
        .prefetch(requests: Iterable[PredictionRequest]) -> None
        .adopt(request, directory, *, copy_outputs: bool = False, check: bool = True)
                -> StoredPrediction
        .stats() -> dict[str, int]

What a pipeline calls, in this order: build the requests (``OpenFold3Runner.make_request``),
``prefetch`` them all when a batch of samples is about to be processed (one batched model run
for the missing ones), and ``record`` for every metric that needs the prediction. Parse-only
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
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

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
    from binding_metrics.predictors.record import PredictionRecord
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
        self._records: dict[tuple, PredictionRecord] = {}
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
    ) -> PredictionRecord:
        """The parsed, completed record of one sample of ``request``; runs and parses at most once.

        The record has been through ``PredictionParser.complete`` of its model, so a consumer
        gets the token layout and chain names that need the structure. This is the one place
        the pipeline and ``binding-metrics-prediction`` read records from.

        Args:
            request: What to predict.
            seed_index, sample: 1-based positions in the model's natural order of outputs.
            chain_map: Model chain ID to user chain ID (see ``PredictionParser.load``).

        Raises:
            PredictionFailedError, PredictionUnavailableError: See ``entry``.
            ValueError: ``chain_map`` is not a valid map.
            KeyError: No parser is registered for the model.
        """
        entry = self.entry(request)
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
                with self._guard:
                    self._records[memo_key] = record
                    self._counts["parsed"] += 1
        return record

    def prefetch(self, requests: Iterable[PredictionRequest]) -> None:
        """Make sure the entries of all ``requests`` exist, in as few model runs as possible.

        The requests that the store has no finished entry for and that share a model go to
        ``PredictionStore.run_missing``, which batches them when the runner can. A failed run
        is remembered, not raised here: ``entry`` and ``record`` raise it for that request.

        Raises:
            PredictionUnavailableError: Something has to run and there is no usable runner.
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
