"""Per-group aggregates of per-pose rows, and the representative pose of each group.

The rows are the dictionaries of a pose-set result (one per pose, registry metric blocks at
the top level next to ``id``, ``status``, ``provenance``, ``native``, ``reason`` and
``reasons``). Two jobs, both pure Python plus numpy:

* ``aggregate_poses`` summarises every numeric key of every block over the poses of each
  group (the back-end by default) and over all poses, and always says how many poses gave
  a value: ``n_ok`` next to ``n_total``. A failed pose counts in ``n_total`` and never in
  ``n_ok``, so a mean over the survivors is never shown as a mean over the set.
* ``select_representatives`` names one pose per group and one overall: the medoid of a
  distance matrix, the pose with the best native score, or the choice of a function.

Which values are valid
----------------------
A value of key ``K`` in the block ``B`` of a pose is valid when all of these hold:

1. the row is not ``"status": "failed"``;
2. no dictionary on the path to ``K`` carries a ``reason`` (a non-empty string); with
   ``reason_policy="exclude"`` (the default) a dictionary-level ``reason`` excludes every
   value under it, because the toolkit's convention is a sentinel plus a ``reason`` and some
   metrics set it when only one step failed, so the keys it covers are not known. The row
   keeps its values. ``reason_policy="keep"`` keeps them and lists the reasons under
   ``caveats``;
3. the value is a finite number: not ``None``, NaN, infinite, a bool or a string.

Keys come from the numeric leaves (int or float, not bool) of the rows that did not fail;
strings, bools and lists never make a key, and ``exclude_blocks`` keeps whole blocks out
(``provenance``, ``residue_table`` ... by default). A pose counts in ``n_total`` of a key
when it failed, or its block is in the row, or the row's ``reasons`` names the block (a
metric that raised). A pose that has none of these was not asked for that metric (a heavy
metric run on the representatives only) and is not counted. Values are sorted before any
statistic is computed, so permuting the rows changes no digit.

Importing this module imports numpy and the standard library only.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Statistics reported by default, in this order.
DEFAULT_STATISTICS = ("mean", "std", "min", "max", "median")

#: Blocks of a row that are never aggregated: identity, labels, run facts and long tables.
DEFAULT_EXCLUDED_BLOCKS = (
    "id",
    "kind",
    "content_hash",
    "provenance",
    "status",
    "reason",
    "reasons",
    "independent_of_design",
    "profile",
    "residue_table",
)

#: Names of the entries that every aggregate carries next to the statistics.
ENTRY_KEYS = ("n_ok", "n_total", "complete", "reasons", "caveats", "statistic_reasons")

#: Group keys that the result uses for other things.
RESERVED_GROUP_KEYS = ("all", "rule", "reasons")

REASON_POLICIES = ("exclude", "keep")

#: Two scores closer than this (relative, plus a tiny absolute floor) are tied, so that the
#: float rounding of analytically equal sums does not decide a medoid.
TIE_REL_TOLERANCE = 1e-9
TIE_ABS_TOLERANCE = 1e-12

_NOT_ASKED = object()


# ---------------------------------------------------------------------- statistics


class Aggregator(Protocol):
    """A statistic: a function from the valid values of a group to one number.

    It receives a list of finite Python floats in ascending order and returns a float, or
    None when it cannot say. An exception it raises becomes ``None`` with the exception as
    the reason; a result that is not a finite number becomes ``None`` with a reason. A
    function may set a ``min_values`` attribute (default 1): with fewer values it is not
    called and the reason is "needs at least N values".
    """

    def __call__(self, values: Sequence[float]) -> Optional[float]: ...


@dataclass(frozen=True)
class _Statistic:
    function: Callable[[Sequence[float]], Optional[float]]
    min_values: int = 1

    def __call__(self, values: Sequence[float]) -> Optional[float]:
        return self.function(values)


#: The registered statistics. ``std`` is the sample standard deviation (``ddof=1``);
#: ``median`` is the mean of the two middle values for an even count.
STATISTICS = MappingProxyType(
    {
        "mean": _Statistic(lambda values: float(np.mean(values))),
        "std": _Statistic(lambda values: float(np.std(values, ddof=1)), min_values=2),
        "min": _Statistic(lambda values: float(np.min(values))),
        "max": _Statistic(lambda values: float(np.max(values))),
        "median": _Statistic(lambda values: float(np.median(values))),
    }
)


def resolve_statistics(
    statistics: Sequence[str] | Mapping[str, Aggregator] = DEFAULT_STATISTICS,
) -> dict[str, Aggregator]:
    """Return the statistics to compute as ``{name: aggregator}``, in the given order.

    Args:
        statistics: Names from ``STATISTICS`` (``"mean"``, ``"std"``, ``"min"``, ``"max"``,
            ``"median"``), or a mapping ``{name: aggregator}`` for custom ones (pass
            ``STATISTICS["mean"]`` to keep a registered one next to them).

    Raises:
        ValueError: If a name is unknown, repeated or one of ``ENTRY_KEYS``, or a mapping
            value is not callable. Call it before any pose is read.
    """
    if isinstance(statistics, str):
        raise ValueError(f"statistics must be a list of names or a mapping, got {statistics!r}")
    if isinstance(statistics, Mapping):
        named = dict(statistics)
        for name, function in named.items():
            if not callable(function):
                raise ValueError(f"statistic {name!r} is not callable: {function!r}")
    else:
        names = list(statistics)
        unknown = [name for name in names if name not in STATISTICS]
        if unknown:
            raise ValueError(
                f"unknown statistic {unknown}; registered: {sorted(STATISTICS)}, "
                "or pass a mapping {name: callable}"
            )
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"statistic names repeated: {repeated}")
        named = {name: STATISTICS[name] for name in names}
    for name in named:
        if not isinstance(name, str) or not name:
            raise ValueError(f"statistic names must be non-empty strings, got {name!r}")
        if name in ENTRY_KEYS:
            raise ValueError(f"statistic name {name!r} is reserved (reserved: {list(ENTRY_KEYS)})")
    return named


def _check_min_n_ok(min_n_ok: int) -> int:
    if isinstance(min_n_ok, bool) or not isinstance(min_n_ok, (int, np.integer)) or min_n_ok < 1:
        raise ValueError(f"min_n_ok must be an integer of at least 1, got {min_n_ok!r}")
    return int(min_n_ok)


def _check_reason_policy(reason_policy: str) -> str:
    if reason_policy not in REASON_POLICIES:
        raise ValueError(f"reason_policy must be one of {REASON_POLICIES}, got {reason_policy!r}")
    return reason_policy


def _classify(value: Any) -> tuple[Optional[float], Optional[str]]:
    """``(number, None)`` for a finite number, ``(None, why)`` for anything else."""
    if isinstance(value, (bool, np.bool_)):
        return None, "value is a bool, not a number"
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            number = float(value)
        except OverflowError:
            return None, "value is infinite"
        if math.isnan(number):
            return None, "value is NaN"
        if math.isinf(number):
            return None, "value is infinite"
        return number, None
    if value is None:
        return None, "value is None"
    return None, f"value is not a number ({type(value).__name__})"


def _is_numeric(value: Any) -> bool:
    """True for an int or float (NaN and infinity included), False for a bool or anything else."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _apply(
    function: Aggregator, values: list[float], min_n_ok: int
) -> tuple[Optional[float], Optional[str]]:
    n_ok = len(values)
    if n_ok < min_n_ok or n_ok == 0:
        return None, "no valid value" if n_ok == 0 else f"n_ok {n_ok} is below min_n_ok {min_n_ok}"
    needed = int(getattr(function, "min_values", 1))
    if n_ok < needed:
        return None, f"needs at least {needed} values"
    try:
        result = function(list(values))
    except Exception as exc:  # noqa: BLE001 - a user aggregator may raise anything; the failure is recorded as the reason and logged
        logger.warning("aggregator failed on %d values: %s: %s", n_ok, type(exc).__name__, exc)
        return None, f"{type(exc).__name__}: {exc}"
    if result is None:
        return None, "the aggregator returned None"
    number, why = _classify(result)
    if number is None:
        return None, f"the aggregator returned a result that is not a finite number ({why})"
    return number, None


def _counted(texts: Sequence[str]) -> dict[str, int]:
    """``{text: count}``, most frequent first, ties in alphabetical order."""
    counts = Counter(texts)
    return {text: counts[text] for text in sorted(counts, key=lambda text: (-counts[text], text))}


def _summarise(
    valid: Sequence[float],
    reasons: Sequence[str],
    caveats: Sequence[str],
    aggregators: Mapping[str, Aggregator],
    min_n_ok: int,
) -> dict[str, Any]:
    ordered = sorted(valid)
    entry: dict[str, Any] = {}
    statistic_reasons: dict[str, str] = {}
    for name, function in aggregators.items():
        value, reason = _apply(function, ordered, min_n_ok)
        entry[name] = value
        if reason is not None:
            statistic_reasons[name] = reason
    n_ok = len(ordered)
    n_total = n_ok + len(reasons)
    entry["n_ok"] = n_ok
    entry["n_total"] = n_total
    entry["complete"] = n_ok == n_total
    entry["reasons"] = _counted(reasons)
    entry["caveats"] = _counted(caveats)
    entry["statistic_reasons"] = statistic_reasons
    return entry


def aggregate_values(
    values: Sequence[Any],
    *,
    statistics: Sequence[str] | Mapping[str, Aggregator] = DEFAULT_STATISTICS,
    min_n_ok: int = 1,
) -> dict[str, Any]:
    """Summarise one list of values: the statistics, and how many of them were usable.

    A value that is not a finite number (None, NaN, infinity, a bool, a string) is not
    valid; it counts in ``n_total`` and its cause is counted under ``reasons``.

    Returns:
        ``{<statistic>: float | None, ..., "n_ok": int, "n_total": int, "complete": bool,
        "reasons": {text: count}, "caveats": {}, "statistic_reasons": {statistic: text}}``.
        ``n_total`` is ``len(values)``, ``complete`` is ``n_ok == n_total``. With fewer than
        ``min_n_ok`` valid values every statistic is None with a reason; ``std`` needs two
        values whatever ``min_n_ok`` is. Keys are in the order of ``statistics``, then the
        entries above.

    Raises:
        ValueError: For an unknown statistic or a ``min_n_ok`` below 1.
    """
    aggregators = resolve_statistics(statistics)
    min_n_ok = _check_min_n_ok(min_n_ok)
    valid: list[float] = []
    reasons: list[str] = []
    for value in values:
        number, why = _classify(value)
        if number is None:
            reasons.append(why)
        else:
            valid.append(number)
    return _summarise(valid, reasons, [], aggregators, min_n_ok)


# ---------------------------------------------------------------------- groups


def _check_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(rows)
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {position} is a {type(row).__name__}, not a mapping")
    return rows


def _check_group_by(group_by: Sequence[str] | str) -> tuple[str, ...]:
    names = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"group_by names must be non-empty strings, got {name!r}")
    return names


def group_key(row: Mapping[str, Any], group_by: Sequence[str] = ("backend",)) -> str:
    """The group a row belongs to, from its ``provenance``.

    The ``backend`` value is written bare and every other key as ``name=value``, joined by
    ``|``: ``group_by=("backend",)`` gives ``"boltz2"`` and ``("backend", "seed")`` gives
    ``"boltz2|seed=42"``. A key the pose does not have reads as None (``"seed=None"``).
    """
    provenance = row.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    parts = []
    for name in _check_group_by(group_by):
        value = provenance.get(name)
        parts.append(str(value) if name == "backend" and value is not None else f"{name}={value}")
    return "|".join(parts)


def _group_members(
    rows: Sequence[Mapping[str, Any]], group_by: tuple[str, ...]
) -> dict[str, list[int]]:
    """Row positions of every group (keys in sorted order), then of ``"all"``."""
    members: dict[str, list[int]] = {}
    if group_by:
        for position, row in enumerate(rows):
            members.setdefault(group_key(row, group_by), []).append(position)
    clash = sorted(set(members) & set(RESERVED_GROUP_KEYS))
    if clash:
        raise ValueError(
            f"group keys {clash} are reserved for the result "
            f"(reserved: {list(RESERVED_GROUP_KEYS)}); "
            "rename the provenance value or group by another key"
        )
    ordered = {key: members[key] for key in sorted(members)}
    ordered["all"] = list(range(len(rows)))
    return ordered


# ---------------------------------------------------------------------- flattening


@dataclass
class _RowView:
    """What the aggregates read from one row."""

    failed_reason: Optional[str]
    leaves: dict[str, tuple[str, Any]]  # dotted path -> (block, value)
    reason_nodes: dict[str, str]  # dotted path of a dictionary with a reason -> the reason
    blocks: set[str]
    stage_reasons: dict[str, str]

    def covering_reason(self, path: str) -> Optional[str]:
        """The reason of the outermost dictionary on the way to ``path``, if there is one."""
        found = [
            (len(node), reason)
            for node, reason in self.reason_nodes.items()
            if path == node or path.startswith(node + ".")
        ]
        return min(found)[1] if found else None


def _reason_of(node: Mapping[str, Any]) -> Optional[str]:
    reason = node.get("reason")
    return reason if isinstance(reason, str) and reason else None


def _walk(
    value: Any,
    path: str,
    block: str,
    view: _RowView,
    covered: bool,
) -> None:
    if not isinstance(value, Mapping):
        view.leaves[path] = (block, value)
        return
    if not covered:
        reason = _reason_of(value)
        if reason is not None:
            view.reason_nodes[path] = reason
            covered = True
    for key, child in value.items():
        if key == "reason":
            continue  # the reason is text about the values, never one of them
        _walk(child, f"{path}.{key}", block, view, covered)


def _view_of(row: Mapping[str, Any], exclude_blocks: Sequence[str]) -> _RowView:
    failed = row.get("status") == "failed"
    stage_reasons = row.get("reasons")
    stage_reasons = stage_reasons if isinstance(stage_reasons, Mapping) else {}
    view = _RowView(
        failed_reason=(str(row.get("reason") or "pose failed") if failed else None),
        leaves={},
        reason_nodes={},
        blocks=set(),
        stage_reasons={
            str(stage): text for stage, text in stage_reasons.items() if isinstance(text, str)
        },
    )
    for name, value in row.items():
        if name in exclude_blocks:
            continue
        view.blocks.add(str(name))
        _walk(value, str(name), str(name), view, covered=False)
    return view


def _outcome(view: _RowView, path: str, block: str, reason_policy: str):
    """``(number, None, caveat)``, ``(None, reason, None)``, or ``_NOT_ASKED``."""
    if view.failed_reason is not None:
        return None, view.failed_reason, None
    covering = view.covering_reason(path)
    if covering is not None and reason_policy == "exclude":
        return None, covering, None
    if path in view.leaves:
        number, why = _classify(view.leaves[path][1])
        if number is None:
            return None, why, None
        return number, None, covering
    if block in view.blocks:
        return None, "no value for this key", None
    if block in view.stage_reasons:
        return None, view.stage_reasons[block], None
    return _NOT_ASKED


def aggregate_poses(
    rows: Sequence[Mapping[str, Any]],
    group_by: Sequence[str] = ("backend",),
    *,
    statistics: Sequence[str] | Mapping[str, Aggregator] = DEFAULT_STATISTICS,
    min_n_ok: int = 1,
    reason_policy: str = "exclude",
    exclude_blocks: Sequence[str] = DEFAULT_EXCLUDED_BLOCKS,
    keys: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate every numeric key of the rows per group and over all poses.

    Args:
        rows: Per-pose row dictionaries (see the module docstring for what is read).
        group_by: ``provenance`` keys that define the groups; ``("backend",)`` by default,
            ``("backend", "seed")`` aggregates over the samples of each seed, ``()`` gives
            the ``"all"`` group only. Group keys are written as ``group_key`` does.
        statistics: Names from ``STATISTICS`` or a mapping ``{name: aggregator}``.
        min_n_ok: A key with fewer valid values yields None for every statistic and a
            reason. The default 1 reports a mean over a single survivor, with ``n_ok`` and
            ``complete`` next to it and ``std`` None ("needs at least 2 values").
        reason_policy: ``"exclude"`` (default) drops every value under a dictionary that
            has a ``reason``; ``"keep"`` keeps them and counts the reasons under ``caveats``.
        exclude_blocks: Top-level row names that are not aggregated.
        keys: Dotted key paths to report even when no pose has a number there.

    Returns:
        ``{group: {key path: entry}}``: groups in sorted order and ``"all"`` last, key paths
        sorted (``"interface.delta_sasa"``, ``"native.iptm"``, ``"frame.n_paired"``). An
        entry is the dictionary of ``aggregate_values`` plus ``n_ok``, ``n_total``,
        ``complete``, ``reasons`` (``{text: count}`` of the poses without a valid value),
        ``caveats`` and ``statistic_reasons``. A key a group was not asked for (see the
        module docstring) is absent from that group.

    Raises:
        ValueError: For an unknown statistic or policy, ``min_n_ok`` below 1, a row that is
            not a mapping, or a group key that is reserved (``"all"``, ``"rule"``,
            ``"reasons"``). The check happens before any value is read.
    """
    aggregators = resolve_statistics(statistics)
    min_n_ok = _check_min_n_ok(min_n_ok)
    reason_policy = _check_reason_policy(reason_policy)
    rows = _check_rows(rows)
    groups = _group_members(rows, _check_group_by(group_by))
    views = [_view_of(row, exclude_blocks) for row in rows]

    block_of: dict[str, str] = {}
    for view in views:
        if view.failed_reason is not None:
            continue
        for path, (block, value) in view.leaves.items():
            if _is_numeric(value):
                block_of.setdefault(path, block)
    for path in keys or ():
        block_of.setdefault(path, path.split(".", 1)[0])
    paths = sorted(block_of)

    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for group, positions in groups.items():
        entries: dict[str, dict[str, Any]] = {}
        for path in paths:
            valid: list[float] = []
            missing: list[str] = []
            caveats: list[str] = []
            for position in positions:
                outcome = _outcome(views[position], path, block_of[path], reason_policy)
                if outcome is _NOT_ASKED:
                    continue
                number, reason, caveat = outcome
                if number is None:
                    missing.append(reason)
                else:
                    valid.append(number)
                    if caveat is not None:
                        caveats.append(caveat)
            if valid or missing:
                entries[path] = _summarise(valid, missing, caveats, aggregators, min_n_ok)
        aggregates[group] = entries
    return aggregates


# ---------------------------------------------------------------------- representatives


@dataclass(frozen=True, eq=False)
class PoseDistances:
    """A distance matrix between poses, with the id of each row and column.

    ``matrix[i, j]`` is the distance from pose ``ids[i]`` to pose ``ids[j]``; a pair that
    cannot be compared is NaN or None. The diagonal is never read.

    Raises:
        ValueError: If the ids are not unique non-empty strings or the matrix is not
            ``len(ids)`` by ``len(ids)`` numbers.
    """

    ids: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self):
        ids = tuple(self.ids)
        if not all(isinstance(pose_id, str) and pose_id for pose_id in ids):
            raise ValueError("ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            repeated = sorted({pose_id for pose_id in ids if ids.count(pose_id) > 1})
            raise ValueError(f"ids must be unique; repeated: {repeated}")
        try:
            matrix = np.array(self.matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"matrix must hold numbers or None: {exc}") from exc
        if matrix.shape != (len(ids), len(ids)):
            raise ValueError(f"matrix has shape {matrix.shape}, expected {(len(ids), len(ids))}")
        matrix.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "matrix", matrix)

    def submatrix(self, ids: Sequence[str]) -> np.ndarray:
        """The distances among ``ids``, rows and columns in the order given.

        Raises:
            ValueError: If an id is not in the matrix.
        """
        position = {pose_id: index for index, pose_id in enumerate(self.ids)}
        absent = [pose_id for pose_id in ids if pose_id not in position]
        if absent:
            raise ValueError(f"the distance matrix has no row for poses {absent}")
        order = [position[pose_id] for pose_id in ids]
        return self.matrix[np.ix_(order, order)]

    def distance(self, pose_a: str, pose_b: str) -> float:
        """The distance from ``pose_a`` to ``pose_b`` (NaN when they cannot be compared)."""
        return float(self.submatrix([pose_a, pose_b])[0, 1])


#: The direction in which a producing model calls a native score better. Other names need
#: an explicit ``confidence_direction``.
NATIVE_SCORE_DIRECTION = MappingProxyType(
    {
        "avg_plddt": "max",
        "ptm": "max",
        "iptm": "max",
        "ranking_score": "max",
        "gpde": "min",
        "has_clash": "min",
        "disorder": "min",
    }
)
_NATIVE_SCORE_FAMILIES = {"chain_ptm.": "max", "chain_pair_iptm.": "max"}


def native_score_direction(name: str) -> Optional[str]:
    """``"max"`` or ``"min"`` for a native score of a predicted pose, None for another name.

    ``avg_plddt``, ``ptm``, ``iptm``, ``ranking_score`` and the per-chain ``chain_ptm.*`` and
    ``chain_pair_iptm.*`` values are better when larger; ``gpde``, ``has_clash`` and
    ``disorder`` are better when smaller.
    """
    if name in NATIVE_SCORE_DIRECTION:
        return NATIVE_SCORE_DIRECTION[name]
    for prefix, direction in _NATIVE_SCORE_FAMILIES.items():
        if name.startswith(prefix):
            return direction
    return None


def _ids_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [row.get("id") for row in rows]
    if not all(isinstance(pose_id, str) and pose_id for pose_id in ids):
        raise ValueError("every row needs a non-empty string 'id' to name a representative")
    repeated = sorted({pose_id for pose_id in ids if ids.count(pose_id) > 1})
    if repeated:
        raise ValueError(f"pose ids are not unique: {repeated}")
    return ids


def _first_of_best(scores: Sequence[float], largest: bool) -> int:
    """Index of the best score; scores that tie within the tolerance go to the first index."""
    best = max(scores) if largest else min(scores)
    for index, score in enumerate(scores):
        if math.isclose(score, best, rel_tol=TIE_REL_TOLERANCE, abs_tol=TIE_ABS_TOLERANCE):
            return index
    raise AssertionError("the best score is among the scores")  # pragma: no cover


def _medoid(
    candidates: Sequence[Mapping[str, Any]], distances: PoseDistances
) -> tuple[Optional[str], Optional[str]]:
    ids = [row["id"] for row in candidates]
    if len(ids) == 1:
        distances.submatrix(ids)  # the pose must be in the matrix
        return ids[0], None
    matrix = distances.submatrix(ids)
    off_diagonal = ~np.eye(len(ids), dtype=bool)
    usable = np.isfinite(matrix) & off_diagonal
    counts = usable.sum(axis=1)
    comparable = np.flatnonzero(counts > 0)
    if comparable.size == 0:
        return None, "no pair of poses in the group has a finite distance"
    totals = np.where(usable, matrix, 0.0).sum(axis=1)
    scores = [float(totals[i] / counts[i]) for i in comparable]
    return ids[comparable[_first_of_best(scores, largest=False)]], None


def _best_native(
    candidates: Sequence[Mapping[str, Any]], key: str, direction: str
) -> tuple[Optional[str], Optional[str]]:
    scored: list[tuple[str, float]] = []
    for row in candidates:
        native = row.get("native")
        value = native.get(key) if isinstance(native, Mapping) else None
        number, _ = _classify(value)
        if number is not None:
            scored.append((row["id"], number))
    if not scored:
        return None, f"no pose has a finite native score '{key}'"
    index = _first_of_best([score for _, score in scored], largest=direction == "max")
    return scored[index][0], None


def rank_by(key: Callable[[Mapping[str, Any]], float], *, largest: bool = False):
    """Make a representative rule from a key function of a row.

    The rule picks the pose whose ``key(row)`` is smallest (largest with ``largest=True``);
    a row whose key raises or is not a finite number is skipped, and ties go to the lowest
    id. Use it as ``rule=rank_by(lambda row: row["interface"]["delta_sasa"], largest=True)``.
    """

    def rule(rows: Sequence[Mapping[str, Any]], distances: Optional[PoseDistances]):
        scored: list[tuple[str, float]] = []
        for row in rows:
            try:
                number, _ = _classify(key(row))
            except Exception as exc:  # noqa: BLE001 - a user key function may raise anything; the row is skipped and the failure logged
                logger.debug(
                    "rank_by key failed on %r: %s: %s", row.get("id"), type(exc).__name__, exc
                )
                continue
            if number is not None:
                scored.append((row["id"], number))
        if not scored:
            return None
        return scored[_first_of_best([score for _, score in scored], largest)][0]

    return rule


def _check_rule(rule, distances, confidence_key, confidence_direction) -> tuple[str, Optional[str]]:
    """Validate the arguments of ``select_representatives``; return the rule name and direction."""
    if callable(rule):
        return "custom", None
    if rule == "medoid":
        if distances is None:
            raise ValueError("rule 'medoid' needs a distance matrix (PoseDistances)")
        return "medoid", None
    if rule == "confidence":
        direction = confidence_direction or native_score_direction(confidence_key)
        if direction is None:
            raise ValueError(
                f"the direction of native score {confidence_key!r} is not known; "
                "pass confidence_direction='max' or 'min'"
            )
        if direction not in ("max", "min"):
            raise ValueError(f"confidence_direction must be 'max' or 'min', got {direction!r}")
        return "confidence", direction
    raise ValueError(f"rule must be 'medoid', 'confidence' or a callable, got {rule!r}")


def select_representatives(
    rows: Sequence[Mapping[str, Any]],
    distances: Optional[PoseDistances] = None,
    *,
    group_by: Sequence[str] = ("backend",),
    rule: str | Callable = "medoid",
    confidence_key: str = "ranking_score",
    confidence_direction: Optional[str] = None,
) -> dict[str, Any]:
    """Name one representative pose per group and one over all poses.

    Rules:

    * ``"medoid"`` (default): the pose with the smallest mean distance to the other poses of
      its group (the smallest sum when every pair has a distance, so the same pose as the
      sum). Needs ``distances``. A pose with no finite distance to any other member is not a
      candidate; a group of one pose is represented by it.
    * ``"confidence"``: the pose with the best ``native[confidence_key]`` in its group,
      "best" being the largest for ``ranking_score``, ``iptm``, ``ptm`` and ``avg_plddt``
      and the smallest for ``gpde``, ``has_clash`` and ``disorder`` (``native_score_direction``);
      another name needs ``confidence_direction``. A model's own scores are not comparable
      across models, so with more than one group the overall representative is None with a
      reason: give a callable or use the medoid for it.
    * a callable ``rule(rows_of_group, distances) -> pose id``: it receives the rows of
      the group that did not fail, sorted by id, and the ``PoseDistances`` (or None). It
      must return the id of one of them; ``rank_by`` makes one from a key function.

    A pose with ``"status": "failed"`` is never a candidate. Ties (scores within
    ``TIE_REL_TOLERANCE``) go to the lowest id in string order, so the answer does not
    depend on the order of ``rows``.

    Args:
        rows: Per-pose rows with ``id``, ``status``, ``provenance`` and ``native``.
        distances: Distances among the poses; every pose that did not fail must be in it
            when the rule is ``"medoid"``.
        group_by: ``provenance`` keys that define the groups, as for ``aggregate_poses``.

    Returns:
        ``{group: id or None, ..., "all": id or None, "rule": "medoid" | "confidence" |
        "custom", "reasons": {key: sentence}}``; ``reasons`` has an entry for every None.

    Raises:
        ValueError: For an unknown rule, a ``medoid`` without distances, a native score with
            no known direction, rows without unique ids, a reserved group key, or a pose
            that is not in ``distances``. A callable that raises or returns a bad id gives
            None and a reason instead.
    """
    group_by = _check_group_by(group_by)
    rule_name, direction = _check_rule(rule, distances, confidence_key, confidence_direction)
    rows = _check_rows(rows)
    _ids_of(rows)
    groups = _group_members(rows, group_by)

    chosen: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    n_groups = len(groups) - 1
    for group, positions in groups.items():
        members = sorted(
            (rows[position] for position in positions if rows[position].get("status") != "failed"),
            key=lambda row: row["id"],
        )
        pose_id: Optional[str] = None
        reason: Optional[str] = None
        if not members:
            reason = "no pose has a status other than 'failed'"
        elif rule_name == "confidence" and group == "all" and n_groups > 1:
            reason = (
                "native scores are not comparable across groups; "
                "use rule='medoid' or a callable for the overall pose"
            )
        elif rule_name == "medoid":
            pose_id, reason = _medoid(members, distances)
        elif rule_name == "confidence":
            pose_id, reason = _best_native(members, confidence_key, direction)
        else:
            pose_id, reason = _call_rule(rule, members, distances)
        chosen[group] = pose_id
        if pose_id is None:
            reasons[group] = reason
    return {**chosen, "rule": rule_name, "reasons": reasons}


def _call_rule(rule, members, distances) -> tuple[Optional[str], Optional[str]]:
    try:
        pose_id = rule(list(members), distances)
    except Exception as exc:  # noqa: BLE001 - a user rule may raise anything; the failure is recorded as the reason and logged
        logger.warning("representative rule failed: %s: %s", type(exc).__name__, exc)
        return None, f"{type(exc).__name__}: {exc}"
    if pose_id is None:
        return None, "the rule returned None"
    if pose_id not in {row["id"] for row in members}:
        return None, f"the rule returned {pose_id!r}, which is not a pose of the group"
    return pose_id, None
