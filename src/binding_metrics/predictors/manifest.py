"""A table of the samples that a set of predictions holds: one row per backend, seed and sample.

``manifest(session, requests)`` answers "what did these runs produce?" from the store and the
session memory. It never starts a model: a request with nothing stored gets a row that says so,
and a request whose run failed gets a row with the stored reason. The rows are plain dictionaries
of JSON values (``json.dumps(manifest(...))`` works), in a fixed order, so two calls on the same
store give equal lists whatever order the requests came in.

Public names and signatures (the module imports the standard library only)::

    manifest(session: PredictionSession, requests: Iterable[PredictionRequest] | PredictionRequest,
             *, chain_map: Optional[Mapping[str, str]] = None) -> list[dict[str, Any]]
    prediction_manifest = manifest      # the name the package exports (see below)
    COLUMNS: tuple[str, ...]            # the keys every row has, in order
    STATUS_MISSING = "missing"

``binding_metrics.predictors`` exports the function as ``prediction_manifest``, because its
attribute ``manifest`` is this module: importing a submodule binds its name on the package, so a
function of the same name would be replaced by the module as soon as anything imports it. Call
``from binding_metrics.predictors import prediction_manifest`` or ``from
binding_metrics.predictors.manifest import manifest``.

Columns (every row has all of them; ``None`` where a value does not apply)::

    backend         the model (``request.model``: "of3", "boltz2", "protenix", "af2")
    model_version   the request's version, else the version the runner reported when it ran
    request_key     ``request.key()``
    request_name    ``request.name``
    status          "done" (a run is stored), "adopted" (outputs the user pointed at), "failed"
                    (the stored run failed) or "missing" (nothing is stored for the request)
    reason          why the run failed, as stored; a sentence for a request with nothing stored
                    or an output without a sample; None otherwise
    seed            the seed value the sample was made with (an integer, ``provenance["seed"]``);
                    None when it is unknown: the pose's ``reasons`` say why
    seed_index      1-based position of the seed in the model's natural order of outputs
    sample          1-based position of the sample inside its seed
    ranking_score   the model's own score of the sample (None when it has none)
    rank            1-based rank inside the seed by the model's own score; None for OpenFold3,
                    whose sample numbers are not a ranking, and for a model not listed here
    structure_path  the predicted structure file, as a string
    content_hash    SHA-256 of the structure file's bytes (``PredictedPose.content_hash``)
    avg_plddt, ptm, iptm, gpde, has_clash, disorder
                    the scalars of ``PredictedPose.native_scores()``

and, after those, the keys of ``native_scores()`` that depend on the chains of the sample
(``chain_ptm.<chain>``, ``chain_pair_iptm.<pair>``), which a sample has only when the model
wrote them. A row for a request without a sample (failed, missing, or an output with no sample)
has seed, seed_index, sample and the sample's columns at None.

Order. Rows are sorted by ``(backend, seed, sample)``; a row without a seed follows those with
one, in the order of seed index and sample. Rows that share the triple (two requests of one model
with the same seed) are ordered by request name, request key and seed index, so the order depends
on the content of the requests and nothing else. A request given twice makes one set of rows.

Ranks. A sample's rank is its position inside its seed when the samples are sorted by the
model's score, highest first, with ties in the model's natural order. Boltz-2, Protenix and
ColabFold number their samples by rank, so a sample without a score takes its position. OpenFold3
writes a score for each sample but its sample numbers are not a ranking, and the manifest leaves
its rank empty rather than invent an order that the model does not define.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

from binding_metrics.predictors.store import STATUS_FAILED, PredictionRequest

if TYPE_CHECKING:
    from binding_metrics.predictors.record import PredictedPose
    from binding_metrics.predictors.session import PredictionSession
    from binding_metrics.predictors.store import StoredPrediction

logger = logging.getLogger(__name__)

#: The status of a request for which the store holds nothing (and the manifest runs nothing).
STATUS_MISSING = "missing"

#: The columns every row has, in this order. The scalars after ``content_hash`` are the ones of
#: ``PredictedPose.native_scores()`` that the ``ranking_score`` column does not already hold.
COLUMNS: tuple[str, ...] = (
    "backend",
    "model_version",
    "request_key",
    "request_name",
    "status",
    "reason",
    "seed",
    "seed_index",
    "sample",
    "ranking_score",
    "rank",
    "structure_path",
    "content_hash",
    "avg_plddt",
    "ptm",
    "iptm",
    "gpde",
    "has_clash",
    "disorder",
)

#: The models that rank the samples of a seed (see the module docstring for OpenFold3).
_RANKED_MODELS = frozenset({"af2", "boltz2", "protenix"})

#: The models whose sample numbers are the rank, so a sample without a score takes its position.
#: ColabFold output is recognised by the rank the adapter read from the file name instead.
_NUMBERED_BY_RANK = frozenset({"boltz2", "protenix"})


def manifest(
    session: PredictionSession,
    requests: Union[Iterable[PredictionRequest], PredictionRequest],
    *,
    chain_map: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """One row per ``(backend, seed, sample)`` of ``requests``; see the module docstring.

    The entries come from ``session.lookup`` (the session memory, then the store), the samples
    from the adapter of each model through the session, so a sample that the session parsed
    before is not parsed again. No model is started, whatever the session's ``rerun`` setting.

    Args:
        session: The session whose store and memory are read.
        requests: The requests to list (one request is accepted too).
        chain_map: Model chain ID to user chain ID, as for ``PredictionSession.record``; it
            changes the atoms of the poses and no value of a row.

    Returns:
        A new list of new dictionaries, ready for ``json.dumps``.

    Raises:
        ValueError: ``chain_map`` is not a valid map, or an output file is corrupt (the adapter
            raises, and the manifest does not hide it).
        KeyError: No parser is registered for the model of a stored request.
    """
    if isinstance(requests, PredictionRequest):
        requests = [requests]
    rows: list[dict[str, Any]] = []
    seen = set()
    for request in requests:
        identity = (request.key(), request.name)
        if identity in seen:
            continue
        seen.add(identity)
        rows.extend(_rows_of(session, request, chain_map))
    rows.sort(key=_sort_key)
    return rows


prediction_manifest = manifest


def _rows_of(
    session: PredictionSession,
    request: PredictionRequest,
    chain_map: Optional[Mapping[str, str]],
) -> list[dict[str, Any]]:
    entry = session.lookup(request)
    if entry is None:
        return [
            _row_without_sample(
                request, None, STATUS_MISSING, "no prediction is stored for this request"
            )
        ]
    if entry.status == STATUS_FAILED:
        return [_row_without_sample(request, entry, entry.status, entry.reason)]
    poses = session._parsed_samples(request, entry, chain_map)  # the session's own, no run
    if not poses:
        return [_row_without_sample(request, entry, entry.status, "the output holds no sample")]
    ranks = _ranks(request.model, poses)
    return [_sample_row(request, entry, pose, rank) for pose, rank in zip(poses, ranks)]


def _row_without_sample(
    request: PredictionRequest,
    entry: Optional[StoredPrediction],
    status: str,
    reason: str,
) -> dict[str, Any]:
    row = dict.fromkeys(COLUMNS)
    row.update(
        backend=request.model,
        model_version=request.model_version or (entry.runner_version if entry else ""),
        request_key=request.key(),
        request_name=request.name,
        status=status,
        reason=reason or None,
    )
    return row


def _sample_row(
    request: PredictionRequest, entry: StoredPrediction, pose: PredictedPose, rank: Optional[int]
) -> dict[str, Any]:
    facts = pose.provenance  # written by PredictionSession.record
    scores = pose.native_scores()
    row = dict.fromkeys(COLUMNS)
    row.update(
        backend=request.model,
        model_version=facts["model_version"],
        request_key=request.key(),
        request_name=request.name,
        status=entry.status,
        seed=facts["seed"],
        seed_index=pose.seed_index,
        sample=pose.sample,
        ranking_score=scores["ranking_score"],
        rank=rank,
        structure_path=None if pose.structure_path is None else str(pose.structure_path),
        content_hash=_content_hash(pose),
    )
    for name, value in scores.items():
        if name != "ranking_score":
            row[name] = value
    return row


def _content_hash(pose: PredictedPose) -> Optional[str]:
    """The digest of the structure file, or None when the sample has none or it has vanished."""
    try:
        return pose.content_hash()
    except (ValueError, OSError) as exc:
        logger.debug("no content hash for the %s sample '%s': %s", pose.model, pose.name, exc)
        return None


def _ranks(model: str, poses: list[PredictedPose]) -> list[Optional[int]]:
    """The rank of each pose inside its seed (``poses`` are in the model's natural order)."""
    ranks: list[Optional[int]] = [None] * len(poses)
    if model not in _RANKED_MODELS:
        return ranks
    by_seed: dict[int, list[int]] = {}
    for position, pose in enumerate(poses):
        by_seed.setdefault(pose.seed_index, []).append(position)
    for positions in by_seed.values():
        scores = [_finite(poses[position].ranking_score) for position in positions]
        if any(score is not None for score in scores):
            order = sorted(
                range(len(positions)),
                key=lambda i: (scores[i] is None, -(scores[i] or 0.0), i),
            )
        elif model in _NUMBERED_BY_RANK or all(
            "colabfold_rank" in poses[position].extras for position in positions
        ):
            order = list(range(len(positions)))
        else:
            continue  # the natural order is not a ranking and no score says otherwise
        for rank, i in enumerate(order, start=1):
            ranks[positions[i]] = rank
    return ranks


def _finite(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _sort_key(row: dict[str, Any]) -> tuple:
    seed, sample, seed_index = row["seed"], row["sample"], row["seed_index"]
    return (
        row["backend"],
        seed is None,
        0 if seed is None else seed,
        # samples of an unknown seed keep the model's seed order; known seeds sort by value
        0 if seed is not None or seed_index is None else seed_index,
        sample is None,
        0 if sample is None else sample,
        row["request_name"],
        row["request_key"],
        0 if seed_index is None else seed_index,
    )
