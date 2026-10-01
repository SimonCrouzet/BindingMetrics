"""The run-once store of model predictions (design addendum G1).

One prediction feeds several consumers (the confidence scalars, the interface PAE and PDE, the
EvoBind score, the adversarial check). The store makes each model run happen at most once for one
request, across metrics, batch workers, processes and restarts of the pipeline.

Public names and signatures (the module imports the standard library only)::

    KEY_FORMAT = 1
    MODES = ("predict", "score", "refold")
    STATUS_DONE = "done"; STATUS_FAILED = "failed"; STATUS_ADOPTED = "adopted"

    file_sha256(path) -> str

    @dataclass(frozen=True, eq=False)
    class PredictionRequest:
        PredictionRequest(model: str, name: str, *, mode: str = "score",
            input_path: Optional[str | Path] = None, binder_chain: Optional[str] = None,
            receptor_chain: Optional[str] = None, sequences: Optional[Mapping[str, str]] = None,
            extra_files: Optional[Mapping[str, str | Path]] = None,
            seeds: Sequence[int] = (42,), num_samples: int = 5, model_version: str = "",
            options: Optional[Mapping[str, Any]] = None)
        .key() -> str                       # SHA-256 hex of the canonical JSON of .canonical()
        .canonical() -> dict
        .batch_signature() -> str           # the key with the per-sample fields left out
        .describe() -> dict                 # canonical() plus name and paths, for request.json
        .with_model_version(version: str) -> PredictionRequest
        .for_adoption() -> PredictionRequest    # the same request with the name in its key

    @dataclass(frozen=True)
    class StoredPrediction:
        key, model, name, status, directory, prediction_dir, reason, run_id, started_at,
        finished_at, runner_name, runner_version, executed_here, cause
        .ok -> bool                         # status is "done" or "adopted"
        .require_ok() -> StoredPrediction   # raises PredictionFailedError for "failed"

    class PredictionStore:
        PredictionStore(root: str | Path)
        .path_for(request) -> Path          # <root>/<model>/<key[:2]>/<key>
        .lookup(request) -> Optional[StoredPrediction]      # adopted outputs first, then a run
        .get_or_run(request, runner, *, rerun: bool = False) -> StoredPrediction
        .ensure(request, runner, *, rerun: bool = False) -> StoredPrediction
        .adopt(request, directory, *, copy_outputs: bool = False) -> StoredPrediction
        .run_missing(requests, runner, *, rerun: bool = False, max_batch: int = 256)
                -> list[StoredPrediction]

    class PredictionStoreError(RuntimeError)
    class PredictionFailedError(PredictionStoreError)        # .entry, .key, .reason
    class PredictionUnavailableError(PredictionStoreError)

The key. ``request.key()`` is the SHA-256 of a canonical JSON (sorted keys, no spaces, ASCII) of
everything that changes the output: the model, its version, the mode, the seeds, the number of
samples, the options, the chain roles, the sequences and the CONTENT hash of the input file (and
of each extra file), never its path. The same request gives the same key on every machine, a
moved or renamed identical file gives the same key, and a changed seed, option, model version or
file content gives another. ``name`` is not in the key of a RUN: it is the label of the output
files, so two samples that hold the same structure share one run, and a stored entry keeps the
name of the request that produced it (``StoredPrediction.name``, which the parser must be given).

Adopted outputs are the opposite: the user's directory holds one output per name, so the name
selects it. ``adopt`` stores the entry under ``request.for_adoption().key()``, the key of the
same request with its name added, and ``lookup`` (hence ``get_or_run``, ``ensure`` and
``run_missing``) tries that entry first and the run entry second. Two samples with one input
file each keep their own adopted outputs, and two run requests that differ only in the name
still run once.

The layout of one entry, ``<root>/<model>/<key[:2]>/<key>/``::

    request.json    what was asked (the canonical fields, the name and the input paths)
    STATUS.json     status (done | failed | adopted), reason, run_id, timestamps, runner and
                    where the parser's directory is (prediction_subdir, or outputs_path for
                    adopted outputs kept where they are)
    outputs/        the model's own files, untouched

Crash safety. A run works in a temporary directory ``<key>.tmp-<pid>-<random>`` next to the
entry and the finished directory is renamed into place as one step, so a process that is killed
(SIGKILL, power of the job scheduler) leaves no ``<key>/`` directory and the request looks as
if it never ran; the next run for that key removes the leftovers. The rename is atomic against
a killed process, not against loss of power: nothing is flushed to disk before it. The runner
works inside the temporary directory, so a file it wrote that names its own paths (an OpenFold3
query file lists its alignment files) keeps the temporary path after the rename; only the
directory the parser loads is meant to be read.

Concurrency. An advisory lock (``fcntl.flock``) per key, in ``<key>.lock`` beside the entry,
is held around a run. A process that finds the lock taken waits, then reuses the finished
outputs, so two processes asking for the same request at once run the model once. The lock is
released by the kernel when its holder dies. It needs a POSIX file system on which ``flock``
works, one host at a time (a network file system may not honour it) and is not available on
Windows. Lookups need no lock.

Failures. A run that raises is recorded as ``failed`` with the exception text as its reason
and is not retried silently: asking again raises ``PredictionFailedError`` with that reason, and
``rerun=True`` (the ``--rerun-predictions`` option) forces a new run and drops the adoption of the
same request, so that the fresh run is what is found afterwards. A runner that says it is
unavailable on this machine (``PredictionRunner.is_available()``) raises
``PredictionUnavailableError`` and records nothing, so a missing installation is not remembered
as a failed prediction. ``KeyboardInterrupt`` and ``SystemExit`` stop the run, remove its
temporary directory and propagate.

Model versions. A request made without ``model_version`` takes the runner's ``version()`` when a
runner is given and that version is known. Outputs adopted without a declared version are found
under the key without a version first, so a session that has a runner uses what the user pointed
at; a run made while the version was unknown is not reused once it is known.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Sequence

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

if TYPE_CHECKING:
    from binding_metrics.predictors.runners import PredictionRunner

logger = logging.getLogger(__name__)

#: Version of the canonical JSON that the key hashes; a change to its fields raises it, so old
#: entries are not mistaken for new ones.
KEY_FORMAT = 1

#: Values of ``PredictionRequest.mode``: a prediction from the model's own input (``predict``), a
#: prediction of an existing complex with each chain given its own structure as a template
#: (``score``), and one with the binder folded from its sequence next to a receptor given as
#: template (``refold``).
MODES: tuple[str, ...] = ("predict", "score", "refold")

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_ADOPTED = "adopted"
_STATUSES = (STATUS_DONE, STATUS_FAILED, STATUS_ADOPTED)

#: Longest reason kept in ``STATUS.json`` (characters). An OpenFold3 error carries several lines.
_REASON_CHARS = 4000

_MODEL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

#: The input file's role in ``PredictionRequest.content_hashes``; an extra file may not use it.
_INPUT_ROLE = "input"

#: Fields of ``canonical()`` that belong to one sample and so are not part of ``batch_signature``.
_PER_SAMPLE_FIELDS = ("binder_chain", "receptor_chain", "sequences")


# ---------------------------------------------------------------------------- the request


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of the content of ``path`` (read in 1 MiB blocks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _plain_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``options`` made of JSON values only; anything else raises."""
    try:
        return json.loads(json.dumps(dict(options), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "request options must be JSON values (str, int, float, bool, None, list, dict with "
            f"str keys; no NaN); {exc}"
        ) from None


@dataclass(frozen=True, eq=False)
class PredictionRequest:
    """What to predict, described by everything that changes the output.

    Attributes:
        model: Adapter name (``"of3"``); lower case letters, digits and underscores.
        name: Label of the output files (the query name for OpenFold3). Not part of the key.
        mode: One of ``MODES``.
        input_path: The input file: the complex structure for ``score`` and ``refold``, or
            the model's own input (a query file) for ``predict``. Its content is hashed when
            the request is made, so a file that changes later does not change the key.
        binder_chain, receptor_chain: Chain roles inside ``input_path``.
        sequences: Chain ID to sequence, for a model that predicts from sequences.
        extra_files: Role to file, for other inputs that change the output (a template
            structure, a runner configuration); hashed by content like ``input_path``. The role
            ``"input"`` is reserved.
        seeds: Seed values.
        num_samples: Structures per seed.
        model_version: The model's version string; empty when unknown.
        options: Everything else that changes the output, as JSON values (presets, MSA mode,
            checkpoint, ...). Copied when the request is made.
        content_hashes: Role to the SHA-256 of the file's content, filled in from ``input_path``
            and ``extra_files`` (the input file has the role ``"input"``).

    A request needs an input file or sequences (otherwise every request of a model would share
    one key). Treat it as immutable; ``with_model_version`` and ``for_adoption`` return changed
    copies. Two requests are equal when their key and name are equal. It can be pickled, so
    batch workers can be sent one.

    Raises:
        ValueError: A malformed field, no input, or a reserved extra-file role.
        TypeError: ``options`` holds something JSON cannot represent.
        OSError: An input file cannot be read.
    """

    model: str
    name: str
    _: KW_ONLY
    mode: str = "score"
    input_path: Optional[Path] = None
    binder_chain: Optional[str] = None
    receptor_chain: Optional[str] = None
    sequences: Mapping[str, str] = field(default_factory=dict)
    extra_files: Mapping[str, Path] = field(default_factory=dict)
    seeds: Sequence[int] = (42,)
    num_samples: int = 5
    model_version: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)
    content_hashes: Mapping[str, str] = field(init=False, default_factory=dict, repr=False)
    adoption_name: Optional[str] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        def put(field_name: str, value: Any) -> None:
            object.__setattr__(self, field_name, value)

        if not isinstance(self.model, str) or not _MODEL_NAME.match(self.model):
            raise ValueError(
                f"model {self.model!r} must be lower case letters, digits and underscores"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if isinstance(self.seeds, (str, bytes)):
            raise TypeError("seeds must be a sequence of integers, not a string")
        if not isinstance(self.num_samples, int) or self.num_samples < 1:
            raise ValueError(f"num_samples must be a positive integer, got {self.num_samples!r}")

        input_path = None if self.input_path is None else Path(self.input_path)
        sequences = {str(chain): str(sequence) for chain, sequence in dict(self.sequences).items()}
        extra_files = {str(role): Path(path) for role, path in dict(self.extra_files).items()}
        if input_path is None and not sequences:
            raise ValueError("a prediction request needs an input file or sequences")
        if _INPUT_ROLE in extra_files:
            raise ValueError(f"the extra-file role {_INPUT_ROLE!r} is reserved for input_path")

        hashes = {}
        if input_path is not None:
            hashes[_INPUT_ROLE] = file_sha256(input_path)
        for role, path in extra_files.items():
            hashes[role] = file_sha256(path)

        put("input_path", input_path)
        put("sequences", sequences)
        put("extra_files", extra_files)
        put("seeds", tuple(int(seed) for seed in self.seeds))
        put("options", _plain_options(self.options))
        put("model_version", str(self.model_version or ""))
        put("content_hashes", hashes)

    def canonical(self) -> dict[str, Any]:
        """The fields that the key hashes, as plain JSON values.

        A request made by ``for_adoption`` has one more, ``adopted_name``.
        """
        fields = {
            "format": KEY_FORMAT,
            "model": self.model,
            "model_version": self.model_version,
            "mode": self.mode,
            "file_sha256": dict(sorted(self.content_hashes.items())),
            "binder_chain": self.binder_chain,
            "receptor_chain": self.receptor_chain,
            "sequences": dict(sorted(self.sequences.items())),
            "seeds": list(self.seeds),
            "num_samples": self.num_samples,
            "options": copy.deepcopy(self.options),
        }
        if self.adoption_name is not None:
            fields["adopted_name"] = self.adoption_name
        return fields

    def key(self) -> str:
        """SHA-256 hex digest of the canonical JSON: the same request gives the same key."""
        return hashlib.sha256(_canonical_json(self.canonical()).encode("ascii")).hexdigest()

    def batch_signature(self) -> str:
        """Digest of what several requests must share to be predicted in one batched run.

        The key without the chain roles, the sequences and the input file's hash, which are what
        distinguishes the samples of a batch. A per-sample extra file (a template) still splits
        a batch, because its hash is part of the signature.
        """
        fields = self.canonical()
        fields.pop("adopted_name", None)
        for name in _PER_SAMPLE_FIELDS:
            fields.pop(name)
        fields["file_sha256"] = {
            role: sha for role, sha in fields["file_sha256"].items() if role != _INPUT_ROLE
        }
        return hashlib.sha256(_canonical_json(fields).encode("ascii")).hexdigest()

    def describe(self) -> dict[str, Any]:
        """``canonical()`` plus the key, the name and the input paths (``request.json``)."""
        return {
            **self.canonical(),
            "key": self.key(),
            "name": self.name,
            "input_path": None if self.input_path is None else str(self.input_path),
            "extra_files": {role: str(path) for role, path in self.extra_files.items()},
        }

    def with_model_version(self, version: str) -> PredictionRequest:
        """A copy with ``model_version`` set; the files are not read again."""
        clone = copy.copy(self)
        object.__setattr__(clone, "model_version", str(version or ""))
        return clone

    def for_adoption(self) -> PredictionRequest:
        """The request whose key also holds the name; adopted outputs are stored under it.

        A model run does not depend on the name, so run requests that differ only in it share
        one key. Adopted outputs belong to a name: the directory the user points at holds one
        output per name. The copy is what the store uses for the adopted entry; the files are
        not read again.
        """
        if self.adoption_name is not None:
            return self
        clone = copy.copy(self)
        object.__setattr__(clone, "adoption_name", self.name)
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PredictionRequest):
            return NotImplemented
        return (self.key(), self.name) == (other.key(), other.name)

    def __hash__(self) -> int:
        return hash((self.key(), self.name))


# ---------------------------------------------------------------------------- the entries


class PredictionStoreError(RuntimeError):
    """Base class of the errors the store raises on purpose."""


class PredictionFailedError(PredictionStoreError):
    """A prediction that the store has recorded as failed was asked for.

    ``entry`` is the stored record, ``reason`` the text of the exception that failed the run
    (also ``entry.reason``) and ``key`` the request key. When the run failed in this very call,
    ``__cause__`` is the original exception.
    """

    def __init__(self, entry: StoredPrediction):
        self.entry = entry
        self.key = entry.key
        self.reason = entry.reason
        super().__init__(
            f"the {entry.model} prediction '{entry.name}' failed (recorded in {entry.directory}): "
            f"{entry.reason}\nIt is not retried automatically; pass rerun=True "
            "(--rerun-predictions) to run it again."
        )


class PredictionUnavailableError(PredictionStoreError):
    """The request has no stored output and the model cannot be run here. Nothing is recorded."""


@dataclass(frozen=True)
class StoredPrediction:
    """One entry of the store, as ``STATUS.json`` describes it.

    Attributes:
        key: The request key the entry was written under.
        model, name: The model, and the name the output files were written under; give this
            name (not necessarily the asking request's) to the parser.
        status: ``STATUS_DONE``, ``STATUS_FAILED`` or ``STATUS_ADOPTED``.
        directory: The entry directory.
        prediction_dir: The directory the parser loads; exists unless the run failed.
        reason: Why the run failed; empty otherwise.
        run_id: Random identifier of the run (or adoption) that wrote the entry.
        started_at, finished_at: UTC timestamps (ISO 8601).
        runner_name, runner_version: The runner of the run; empty for adopted outputs.
        executed_here: True on the object returned by the call that ran the model, False for
            an entry that was found. Not stored.
        cause: The exception of a run that failed in this call. Not stored.
    """

    key: str
    model: str
    name: str
    status: str
    directory: Path
    prediction_dir: Path
    reason: str = ""
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    runner_name: str = ""
    runner_version: str = ""
    executed_here: bool = False
    cause: Optional[BaseException] = field(default=None, compare=False, repr=False)

    @property
    def ok(self) -> bool:
        """True for ``done`` and ``adopted``: the outputs can be parsed."""
        return self.status in (STATUS_DONE, STATUS_ADOPTED)

    def require_ok(self) -> StoredPrediction:
        """Return the entry, or raise ``PredictionFailedError`` when the run failed."""
        if not self.ok:
            raise PredictionFailedError(self) from self.cause
        return self


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _reason_of(exc: BaseException) -> str:
    """The recorded reason of a failed run: the exception type and its text."""
    text = str(exc).strip()
    name = type(exc).__name__
    return (f"{name}: {text}" if text else name)[:_REASON_CHARS]


# ---------------------------------------------------------------------------- the store


class PredictionStore:
    """A directory of model predictions, each run at most once (see the module docstring).

    The root is created when something is first written; a store on a read-only file system
    can be looked up. It is made absolute (and ``~`` expanded) when the store is made, so a
    later change of the working directory does not move it. One instance can be shared by
    threads, and any number of instances and processes may use one root.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().absolute()

    # ------------------------------------------------------------------ reading

    def path_for(self, request: PredictionRequest) -> Path:
        """The entry directory of ``request`` (it exists once the request has a record).

        It is the directory of a model run; ``path_for(request.for_adoption())`` is where the
        outputs adopted for the request's name are recorded.
        """
        return self._parent(request) / request.key()

    def lookup(self, request: PredictionRequest) -> Optional[StoredPrediction]:
        """The entry stored for ``request``, or None: adopted outputs for its name first.

        Then the run stored under exactly ``request.key()``. A failed run is returned (check
        ``ok``). An entry whose ``STATUS.json`` cannot be read, or whose output directory has
        vanished, counts as absent and is logged.
        """
        adopted = self._read_entry(self.path_for(request.for_adoption()))
        if adopted is not None:
            return adopted
        return self._read_entry(self.path_for(request))

    # ------------------------------------------------------------------ running

    def get_or_run(
        self,
        request: PredictionRequest,
        runner: Optional[PredictionRunner],
        *,
        rerun: bool = False,
    ) -> StoredPrediction:
        """The finished entry of ``request``; the model runs only when there is none.

        Args:
            request: What to predict. Without a ``model_version`` the runner's is used.
            runner: Runs the model on a miss; None makes a miss an error.
            rerun: Run again even when an entry exists (a peer that finished a run while this
                call waited for the lock counts as that run).

        Returns:
            The entry; ``executed_here`` says whether this call ran the model.

        Raises:
            PredictionFailedError: The run failed now, or failed earlier and ``rerun`` is False.
            PredictionUnavailableError: Nothing is stored and the model cannot be run here.
            ValueError: ``runner`` is for another model.
        """
        return self.ensure(request, runner, rerun=rerun).require_ok()

    def ensure(
        self,
        request: PredictionRequest,
        runner: Optional[PredictionRunner],
        *,
        rerun: bool = False,
    ) -> StoredPrediction:
        """Like ``get_or_run``, but a failed run is returned (``status == "failed"``), not raised.

        Raises:
            PredictionUnavailableError, ValueError: As ``get_or_run``.
        """
        if runner is not None:
            _check_runner(request, runner)
        entry, request = self._find(request, runner)
        if entry is not None and not rerun:
            return entry
        if runner is None:
            raise PredictionUnavailableError(
                f"no stored {request.model} prediction for '{request.name}' "
                f"(key {request.key()[:12]}) and no runner to compute it"
            )
        _require_available(request, runner)
        seen = self._read_entry(self.path_for(request))
        seen_run_id = seen.run_id if seen is not None else None
        if rerun:
            self._drop_adoption(request)
        with self._locked(request):
            current = self._read_entry(self.path_for(request)) if rerun else self.lookup(request)
            if current is not None and (not rerun or current.run_id != seen_run_id):
                return current  # another process ran it while this one waited
            return self._execute(request, runner)

    def run_missing(
        self,
        requests: Sequence[PredictionRequest],
        runner: Optional[PredictionRunner],
        *,
        rerun: bool = False,
        max_batch: int = 256,
    ) -> list[StoredPrediction]:
        """Run every request that has no entry, batched when the runner can.

        Requests that the runner can batch (``supports_batch``) and that share a
        ``batch_signature`` go to one ``run_many`` call each, in chunks of at most
        ``max_batch`` (one lock file descriptor is held per request while a chunk runs) and with
        distinct names inside a chunk; the others run one at a time as ``ensure`` does. Requests
        with the same key are run once.

        Returns:
            One entry per request, in the order of ``requests``, whatever its status: a failed
            run is recorded, not raised (call ``require_ok``).

        Raises:
            PredictionUnavailableError: Something has to run and the model cannot be run here.
            ValueError: ``runner`` is for another model, or ``max_batch`` is not positive.
        """
        if max_batch < 1:
            raise ValueError(f"max_batch must be at least 1, got {max_batch}")
        requests = list(requests)
        if runner is not None:
            for request in requests:
                _check_runner(request, runner)
        resolved: list[PredictionRequest] = []
        found: list[Optional[StoredPrediction]] = []
        seen_run_ids: dict[str, Optional[str]] = {}
        todo: dict[str, PredictionRequest] = {}  # by run key: equal runs are made once
        for request in requests:
            entry, effective = self._find(request, runner)
            resolved.append(effective)
            if entry is not None and not rerun:
                found.append(entry)
                continue
            found.append(None)
            if effective.key() not in todo:
                todo[effective.key()] = effective
                seen = self._read_entry(self.path_for(effective))
                seen_run_ids[effective.key()] = seen.run_id if seen is not None else None

        runs: dict[str, StoredPrediction] = {}
        if todo:
            if runner is None:
                raise PredictionUnavailableError(
                    f"{len(todo)} prediction(s) have no stored output and there is no runner"
                )
            _require_available(next(iter(todo.values())), runner)
            if rerun:
                for request in resolved:
                    self._drop_adoption(request)
            singles = [r for r in todo.values() if not runner.supports_batch(r)]
            groups: dict[str, list[PredictionRequest]] = {}
            for request in todo.values():
                if runner.supports_batch(request):
                    groups.setdefault(request.batch_signature(), []).append(request)
            for group in groups.values():
                for chunk in _chunks(group, max_batch):
                    if len(chunk) == 1:
                        singles.extend(chunk)
                    else:
                        self._run_batch(chunk, runner, rerun, seen_run_ids, runs)
            for request in singles:
                runs[request.key()] = self.ensure(request, runner, rerun=rerun)
        return [
            entry if entry is not None else runs[request.key()]
            for entry, request in zip(found, resolved)
        ]

    # ------------------------------------------------------------------ adopting

    def adopt(
        self,
        request: PredictionRequest,
        directory: str | Path,
        *,
        copy_outputs: bool = False,
    ) -> StoredPrediction:
        """Register outputs that the user produced, so that they count as a finished run.

        The entry belongs to the request's name (``request.for_adoption()``): two samples with
        one input file each keep their own outputs. A run of the same request stays as it is and
        is found only when nothing is adopted for the name.

        Args:
            request: What the outputs are a prediction of. A request without a
                ``model_version`` is stored without one (a session that has a runner still finds
                it).
            directory: The directory the parser of ``request.model`` loads with
                ``request.name``. It is kept where it is (a reference in ``STATUS.json``) unless
                ``copy_outputs`` is True, which copies it into the entry.
            copy_outputs: Copy the directory into the store.

        Returns:
            The entry, with status ``adopted``. Adopting the same directory again returns the
            entry unchanged; another directory replaces it.

        Raises:
            FileNotFoundError: ``directory`` is not a directory.
        """
        source = Path(directory).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"cannot adopt {directory}: it is not a directory")
        slot = request.for_adoption()
        with self._locked(slot):
            current = self._read_entry(self.path_for(slot))
            if current is not None and not copy_outputs and current.prediction_dir == source:
                return current
            self._discard_leftovers(slot)
            tmp = self._new_temp(slot)
            if copy_outputs:
                shutil.copytree(source, tmp / "outputs")
            else:
                (tmp / "outputs").mkdir()
            return self._commit(
                slot,
                tmp,
                status=STATUS_ADOPTED,
                started_at=_now(),
                seconds=None,
                outputs_path=None if copy_outputs else str(source),
            )

    # ------------------------------------------------------------------ internals

    def _parent(self, request: PredictionRequest) -> Path:
        key = request.key()
        return self.root / request.model / key[:2]

    def _find(
        self, request: PredictionRequest, runner: Optional[PredictionRunner]
    ) -> tuple[Optional[StoredPrediction], PredictionRequest]:
        """The stored entry (or None) and the request as it will be run.

        A request without a version is run and stored under the runner's version. Outputs that
        the user adopted without declaring a version are found first: the user pointed at them,
        whatever version made them. A run made under an unknown version is not reused once the
        version is known.
        """
        entry = self.lookup(request)
        if not request.model_version and runner is not None:
            version = runner.version()
            if version:
                if entry is not None and entry.status == STATUS_ADOPTED:
                    return entry, request
                request = request.with_model_version(version)
                return self.lookup(request), request
        return entry, request

    def _drop_adoption(self, request: PredictionRequest) -> None:
        """Forget the outputs adopted for ``request``'s name (a rerun replaces them).

        The user's own files are not touched; a copy made by ``copy_outputs`` is removed.
        """
        slot = request.for_adoption()
        entry_dir = self.path_for(slot)
        if not entry_dir.exists():
            return
        with self._locked(slot):  # a leaf lock: nothing else is taken while it is held
            if entry_dir.exists():
                old = entry_dir.with_name(f"{entry_dir.name}.old-{uuid.uuid4().hex[:8]}")
                os.rename(entry_dir, old)
                shutil.rmtree(old, ignore_errors=True)

    def _read_entry(self, directory: Path) -> Optional[StoredPrediction]:
        status_path = directory / "STATUS.json"
        try:
            with open(status_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("ignoring the unreadable prediction record %s: %s", status_path, exc)
            return None
        if not isinstance(data, dict) or data.get("status") not in _STATUSES:
            logger.warning("ignoring the malformed prediction record %s", status_path)
            return None
        outputs_path = data.get("outputs_path")
        if outputs_path:
            prediction_dir = Path(outputs_path)
        else:
            prediction_dir = directory / "outputs" / str(data.get("prediction_subdir") or "")
        status = data["status"]
        if status != STATUS_FAILED and not prediction_dir.is_dir():
            logger.warning(
                "the outputs of the stored prediction %s are gone (%s); treating it as absent",
                directory,
                prediction_dir,
            )
            return None
        runner = data.get("runner") or {}
        return StoredPrediction(
            key=str(data.get("key") or directory.name),
            model=str(data.get("model") or directory.parent.parent.name),
            name=str(data.get("name") or ""),
            status=status,
            directory=directory,
            prediction_dir=prediction_dir,
            reason=str(data.get("reason") or ""),
            run_id=str(data.get("run_id") or ""),
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            runner_name=str(runner.get("name") or ""),
            runner_version=str(runner.get("version") or ""),
        )

    @contextlib.contextmanager
    def _locked(self, request: PredictionRequest) -> Iterator[None]:
        """Hold the advisory lock of ``request``'s key; wait when another process has it."""
        if fcntl is None:
            raise RuntimeError(
                "the prediction store needs fcntl.flock (Linux, macOS, WSL); it is not "
                "available on this platform"
            )
        parent = self._parent(request)
        parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(parent / f"{request.key()}.lock", os.O_RDWR | os.O_CREAT, 0o666)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info(
                    "waiting for another process that runs the %s prediction '%s' (%s)",
                    request.model,
                    request.name,
                    request.key()[:12],
                )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)  # closing the descriptor releases the lock

    def _new_temp(self, request: PredictionRequest) -> Path:
        tmp = self._parent(request) / f"{request.key()}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        return tmp

    def _discard_leftovers(self, request: PredictionRequest) -> None:
        """Remove what a killed run of this key left; the caller holds the key's lock."""
        parent = self._parent(request)
        for pattern in (f"{request.key()}.tmp-*", f"{request.key()}.old-*"):
            for leftover in parent.glob(pattern):
                logger.info("removing the leftover of an interrupted run: %s", leftover)
                shutil.rmtree(leftover, ignore_errors=True)

    def _execute(self, request: PredictionRequest, runner: PredictionRunner) -> StoredPrediction:
        """Run the model in a temporary directory and install the result; the lock is held."""
        self._discard_leftovers(request)
        tmp = self._new_temp(request)
        outputs = tmp / "outputs"
        outputs.mkdir()
        started_at, clock = _now(), time.monotonic()
        status, reason, subdir, cause = STATUS_DONE, "", "", None
        try:
            subdir = _subdirectory(outputs, Path(runner.run(request, outputs)), runner)
        except Exception as exc:  # noqa: BLE001 - any runner failure is recorded and re-raised later
            status, reason, cause = STATUS_FAILED, _reason_of(exc), exc
            logger.warning(
                "the %s prediction '%s' failed: %s",
                request.model,
                request.name,
                reason.split("\n")[0],
            )
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self._commit(
            request,
            tmp,
            status=status,
            reason=reason,
            subdir=subdir,
            started_at=started_at,
            seconds=time.monotonic() - clock,
            runner=runner,
            cause=cause,
            executed=True,
        )

    def _run_batch(
        self,
        chunk: list[PredictionRequest],
        runner: PredictionRunner,
        rerun: bool,
        seen_run_ids: dict[str, Optional[str]],
        results: dict[str, StoredPrediction],
    ) -> None:
        """One ``run_many`` call for ``chunk``; installs an entry for each request."""
        with contextlib.ExitStack() as stack:
            for request in sorted(chunk, key=lambda r: r.key()):  # one order: no deadlock
                stack.enter_context(self._locked(request))
            pending = []
            for request in chunk:
                current = self._read_entry(self.path_for(request))
                if current is not None and (
                    not rerun or current.run_id != seen_run_ids.get(request.key())
                ):
                    results[request.key()] = current  # finished by another process meanwhile
                else:
                    pending.append(request)
            if not pending:
                return
            workspace = stack.enter_context(self._workspace(chunk[0].model))
            started_at, clock = _now(), time.monotonic()
            try:
                outcome = dict(runner.run_many(pending, workspace / "work"))
                batch_error: Optional[BaseException] = None
            except Exception as exc:  # noqa: BLE001 - a failed batch is recorded for every request
                outcome, batch_error = {}, exc
                logger.warning("the batched %s run failed: %s", chunk[0].model, _reason_of(exc))
            seconds = time.monotonic() - clock
            for request in pending:
                self._discard_leftovers(request)
                tmp = self._new_temp(request)
                outputs = tmp / "outputs"
                result = batch_error or outcome.get(request.key())
                if result is None:
                    result = RuntimeError("the runner returned no result for this request")
                status, reason, cause = STATUS_DONE, "", None
                if isinstance(result, BaseException):
                    status, reason, cause = STATUS_FAILED, _reason_of(result), result
                else:
                    try:
                        os.replace(_inside(Path(result), workspace, runner), outputs)
                    except Exception as exc:  # noqa: BLE001 - recorded as this request's failure
                        status, reason, cause = STATUS_FAILED, _reason_of(exc), exc
                outputs.mkdir(exist_ok=True)
                results[request.key()] = self._commit(
                    request,
                    tmp,
                    status=status,
                    reason=reason,
                    started_at=started_at,
                    seconds=seconds,
                    runner=runner,
                    cause=cause,
                    executed=True,
                    batch_size=len(pending),
                )

    def _commit(
        self,
        request: PredictionRequest,
        tmp: Path,
        *,
        status: str,
        started_at: str,
        seconds: Optional[float],
        reason: str = "",
        subdir: str = "",
        runner: Optional[PredictionRunner] = None,
        outputs_path: Optional[str] = None,
        cause: Optional[BaseException] = None,
        executed: bool = False,
        batch_size: Optional[int] = None,
    ) -> StoredPrediction:
        """Write ``request.json`` and ``STATUS.json`` into ``tmp`` and rename it to the entry."""
        payload = {
            "format": KEY_FORMAT,
            "status": status,
            "reason": reason or None,
            "key": request.key(),
            "model": request.model,
            "name": request.name,
            "run_id": uuid.uuid4().hex,
            "prediction_subdir": subdir,
            "outputs_path": outputs_path,
            "started_at": started_at,
            "finished_at": _now(),
            "duration_seconds": None if seconds is None else round(seconds, 3),
            "batch_size": batch_size,
            "runner": None
            if runner is None
            else {"name": getattr(runner, "name", ""), "version": runner.version() or ""},
        }
        (tmp / "request.json").write_text(
            json.dumps(request.describe(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (tmp / "STATUS.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._install(tmp, self.path_for(request))
        entry = self._read_entry(self.path_for(request))
        if entry is None:  # cannot happen: the directory was just written
            raise PredictionStoreError(f"the entry {self.path_for(request)} cannot be read back")
        return dataclasses.replace(entry, executed_here=executed, cause=cause)

    @contextlib.contextmanager
    def _workspace(self, model: str) -> Iterator[Path]:
        """A directory for one batched run, locked while it is in use so that a crash can be told.

        ``.batch-*`` directories whose lock nobody holds belong to a run that died; they are
        removed the next time a batch starts for the same model.
        """
        model_dir = self.root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for stale in model_dir.glob(".batch-*"):
            try:
                probe = os.open(stale / ".lock", os.O_RDWR)
            except OSError:
                continue
            try:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # in use by another process
                logger.info("removing the workspace of a batched run that died: %s", stale)
                shutil.rmtree(stale, ignore_errors=True)
            finally:
                os.close(probe)
        workspace = model_dir / f".batch-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        workspace.mkdir()
        (workspace / "work").mkdir()
        descriptor = os.open(workspace / ".lock", os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield workspace
        finally:
            os.close(descriptor)
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _install(tmp: Path, entry_dir: Path) -> None:
        """Rename ``tmp`` to ``entry_dir`` in one step; an existing entry is moved aside first."""
        if not entry_dir.exists():
            os.rename(tmp, entry_dir)
            return
        old = entry_dir.with_name(f"{entry_dir.name}.old-{uuid.uuid4().hex[:8]}")
        os.rename(entry_dir, old)
        try:
            os.rename(tmp, entry_dir)
        except BaseException:
            os.rename(old, entry_dir)
            raise
        shutil.rmtree(old, ignore_errors=True)


# ---------------------------------------------------------------------------- helpers


def _check_runner(request: PredictionRequest, runner: PredictionRunner) -> None:
    runner_model = getattr(runner, "name", None)
    if runner_model and runner_model != request.model:
        raise ValueError(
            f"the '{runner_model}' runner cannot run a request for the model '{request.model}'"
        )


def _require_available(request: PredictionRequest, runner: PredictionRunner) -> None:
    if not runner.is_available():
        raise PredictionUnavailableError(
            f"the {request.model} model cannot be started on this machine, and there is no "
            f"stored prediction for '{request.name}' (key {request.key()[:12]}); nothing was "
            "recorded"
        )


def _chunks(group: list[PredictionRequest], size: int) -> list[list[PredictionRequest]]:
    """Split ``group`` into chunks of at most ``size`` with distinct names inside each chunk.

    Two requests with the same name but another key (different inputs) cannot share a batched
    run, because the model would see one query for both.
    """
    chunks: list[list[PredictionRequest]] = []
    for request in group:
        for chunk in chunks:
            if len(chunk) < size and all(other.name != request.name for other in chunk):
                chunk.append(request)
                break
        else:
            chunks.append([request])
    return chunks


def _inside(path: Path, base: Path, runner: PredictionRunner) -> Path:
    """``path`` as a directory below ``base``; anything else is the runner's contract broken."""
    resolved = path.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        raise ValueError(
            f"the {getattr(runner, 'name', 'runner')} runner returned {path}, which is not "
            f"inside its work directory {base}"
        ) from None
    if not resolved.is_dir():
        raise FileNotFoundError(f"the runner returned {path}, which is not a directory")
    return resolved


def _subdirectory(outputs: Path, returned: Path, runner: PredictionRunner) -> str:
    """Where ``returned`` lies inside ``outputs``, as a relative POSIX path ('' for itself)."""
    relative = _inside(returned, outputs, runner).relative_to(outputs.resolve())
    return "" if str(relative) == "." else relative.as_posix()
