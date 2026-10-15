"""Cross-model reconciliation: how far two back-ends agree on the same constructs.

Pure functions on the plain dictionaries of the pose-set layer (numpy, no pandas or scipy).
They read the aggregates of ``poses.aggregate`` and the per-pose rows of a pose-set result and
report numbers; thresholds are arguments.

* ``gap_between_backends``: the difference of two back-ends' means per aggregated key, raw and
  in units of the spread over seeds.
* ``consensus_interface`` and ``compare_consensus_interfaces``: the residues that most poses
  of a back-end hold in the interface, and the Jaccard index of two back-ends' consensus sets.
* ``rank_agreement``: Spearman's rho or Kendall's tau between two back-ends' values of a
  metric over a population of constructs.
* ``hotspot_consistency``: whether the peptide positions that an alanine scan finds sensitive
  are the positions a pose buries and holds in the interface.

A value that cannot be computed is None, never NaN, with a sentence under the name of the
field in ``reasons`` (``reasons["standardised_gap"]``); ``reasons`` is empty when every field
was computed. A back-end absent from the aggregates (a set with a single model) gives None
with a reason, not an error. A malformed argument raises ``ValueError`` before any value is
computed.

Importing this module imports numpy and the standard library only.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from binding_metrics.poses.aggregate import group_key
from binding_metrics.poses.reference import ResidueKey

#: Value of ``scale`` that standardises a gap by the standard deviation pooled over the seeds
#: of the two back-ends.
POOLED_SCALE = "pooled"


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None for None, NaN, infinity, a bool or a non-number."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _check_backends(backend_a: Any, backend_b: Any) -> None:
    for name, value in (("backend_a", backend_a), ("backend_b", backend_b)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    if backend_a == backend_b:
        raise ValueError(f"backend_a and backend_b must differ, both are {backend_a!r}")


# ---------------------------------------------------------------------- gap


def _check_scale(scale: Any) -> Optional[dict[str, float]]:
    """None for the pooled scale, else a validated ``{key path: positive finite scale}``."""
    if isinstance(scale, str):
        if scale != POOLED_SCALE:
            raise ValueError(
                f"scale must be {POOLED_SCALE!r} or a mapping {{key path: scale}}, got {scale!r}"
            )
        return None
    if not isinstance(scale, Mapping):
        raise ValueError(
            f"scale must be {POOLED_SCALE!r} or a mapping {{key path: scale}}, got {scale!r}"
        )
    checked: dict[str, float] = {}
    for key, value in scale.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"scale keys must be non-empty key paths, got {key!r}")
        number = _number(value)
        if number is None or number <= 0:
            raise ValueError(f"scale for {key!r} must be a positive finite number, got {value!r}")
        checked[key] = number
    return checked


def _check_keys(keys: Optional[Iterable[str]]) -> Optional[list[str]]:
    if keys is None:
        return None
    if isinstance(keys, str):
        raise ValueError(f"keys must be a list of key paths, got the string {keys!r}")
    checked = list(keys)
    for key in checked:
        if not isinstance(key, str) or not key:
            raise ValueError(f"keys must be non-empty key paths, got {key!r}")
    return sorted(set(checked))


def _group_of(aggregates: Mapping[str, Any], backend: str) -> Optional[Mapping[str, Any]]:
    group = aggregates.get(backend)
    if group is None:
        return None
    if not isinstance(group, Mapping):
        raise ValueError(
            f"aggregates[{backend!r}] must map key paths to entries (the output of "
            f"aggregate_poses), got {type(group).__name__}"
        )
    return group


class _Side:
    """What one back-end contributes to a gap: its count, mean and standard deviation."""

    def __init__(self, backend: str, group: Optional[Mapping[str, Any]], groups, key: str):
        self.backend = backend
        self.n_ok = 0
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.mean_reason: Optional[str] = None
        self.std_reason: Optional[str] = None
        if group is None:
            self.mean_reason = f"no aggregates for back-end {backend!r} (groups: {sorted(groups)})"
            return
        entry = group.get(key)
        if entry is None:
            self.mean_reason = f"back-end {backend!r} has no aggregate for this key"
            return
        if not isinstance(entry, Mapping) or "n_ok" not in entry:
            raise ValueError(
                f"the entry of {key!r} for back-end {backend!r} must be a mapping with 'n_ok' "
                f"(the output of aggregate_poses), got {entry!r}"
            )
        n_ok = entry["n_ok"]
        if isinstance(n_ok, bool) or not isinstance(n_ok, (int, np.integer)) or n_ok < 0:
            raise ValueError(
                f"n_ok of {key!r} for back-end {backend!r} must be a non-negative integer, "
                f"got {n_ok!r}"
            )
        self.n_ok = int(n_ok)
        statistic_reasons = entry.get("statistic_reasons")
        statistic_reasons = statistic_reasons if isinstance(statistic_reasons, Mapping) else {}
        if self.n_ok == 0:
            self.mean_reason = f"back-end {backend!r} has no valid value (n_ok = 0)"
            return
        self.mean = _number(entry.get("mean"))
        if self.mean is None:
            self.mean_reason = f"back-end {backend!r} has no mean" + _because(
                statistic_reasons.get("mean")
            )
        self.std = _number(entry.get("std"))
        if self.std is None:
            self.std_reason = f"back-end {backend!r} has no standard deviation" + _because(
                statistic_reasons.get("std")
            )


def _because(reason: Any) -> str:
    return f" ({reason})" if isinstance(reason, str) and reason else ""


def _pooled_scale(side_a: _Side, side_b: _Side) -> tuple[Optional[float], Optional[str]]:
    """``(s_pooled, None)`` or ``(None, why)``; ``s_pooled`` may be 0 when every value ties."""
    if side_a.n_ok < 2 or side_b.n_ok < 2:
        return None, (
            "the pooled standard deviation needs at least 2 valid values per back-end "
            f"(n_ok: {side_a.n_ok} and {side_b.n_ok})"
        )
    for side in (side_a, side_b):
        if side.std is None:
            return None, side.std_reason
    degrees_of_freedom = side_a.n_ok + side_b.n_ok - 2
    variance = (
        (side_a.n_ok - 1) * side_a.std**2 + (side_b.n_ok - 1) * side_b.std**2
    ) / degrees_of_freedom
    return math.sqrt(variance), None


def gap_between_backends(
    aggregates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    backend_a: str,
    backend_b: str,
    *,
    keys: Optional[Iterable[str]] = None,
    scale: Any = POOLED_SCALE,
) -> dict[str, dict[str, Any]]:
    """The gap between two back-ends on every aggregated key, raw and in units of the spread.

    ``gap = mean_a - mean_b``; the sign follows the order of ``backend_a`` and ``backend_b``.

    ``standardised_gap = gap / s_pooled``, the denominator of Cohen's d (*Statistical Power
    Analysis for the Behavioral Sciences*, 2nd ed., 1988):
    ``s_pooled = sqrt(((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2))``, where ``s_a``
    and ``s_b`` are the sample standard deviations (ddof = 1) of the back-ends' valid values
    and ``n_a``, ``n_b`` their counts. It needs ``n_a >= 2``, ``n_b >= 2`` and
    ``s_pooled > 0``. ``scale`` can also be a mapping with one scale per key, for instance the
    spread of the same key over constructs.

    Args:
        aggregates: The output of ``aggregate_poses``, ``{group: {key path: entry}}``. The
            entries need ``n_ok`` and, for the gap, ``mean``; for the pooled scale also
            ``std``. Statistics that a custom ``statistics`` argument left out read as missing.
        backend_a: The group (back-end) whose mean is the minuend.
        backend_b: The group whose mean is subtracted; it must differ from ``backend_a``.
        keys: The key paths to report; by default every key of the two groups, sorted. A key a
            group does not have gives None with a reason.
        scale: ``"pooled"`` (default), or a mapping ``{key path: scale}`` of positive finite
            numbers used as given. A key missing from the mapping is None with a reason; a
            mapping key that no entry uses is ignored.

    Returns:
        ``{key path: entry}`` with the keys sorted. An entry is ``{"gap": float | None,
        "standardised_gap": float | None, "mean_a": float | None, "mean_b": float | None,
        "n_ok_a": int, "n_ok_b": int, "scale": float | None, "scale_kind": "pooled" |
        "given", "reasons": {field: sentence}}``. ``scale`` is the pooled deviation (also when
        it is 0, which makes the standardised gap None) or the given scale. ``reasons`` has an
        entry for every field that is None and is empty when everything was computed. A
        back-end that is absent from ``aggregates`` (a set with a single model) gives every
        entry the reason that it has no aggregates.

    Raises:
        ValueError: If neither back-end is in ``aggregates``, the two names are equal or not
            strings, ``scale`` is not ``"pooled"`` or a mapping of positive finite numbers,
            ``keys`` is a bare string, or an entry is not the output of ``aggregate_poses``.
    """
    if not isinstance(aggregates, Mapping):
        raise ValueError(
            "aggregates must be the output of aggregate_poses, {group: {key path: entry}}, "
            f"got {type(aggregates).__name__}"
        )
    _check_backends(backend_a, backend_b)
    scales = _check_scale(scale)
    wanted = _check_keys(keys)
    group_a = _group_of(aggregates, backend_a)
    group_b = _group_of(aggregates, backend_b)
    if group_a is None and group_b is None:
        raise ValueError(
            f"neither {backend_a!r} nor {backend_b!r} is a group of the aggregates "
            f"(groups: {sorted(aggregates)})"
        )
    if wanted is None:
        wanted = sorted(set(group_a or ()) | set(group_b or ()))

    result: dict[str, dict[str, Any]] = {}
    for key in wanted:
        side_a = _Side(backend_a, group_a, aggregates, key)
        side_b = _Side(backend_b, group_b, aggregates, key)
        result[key] = _gap_entry(key, side_a, side_b, scales)
    return result


def _gap_entry(
    key: str, side_a: _Side, side_b: _Side, scales: Optional[dict[str, float]]
) -> dict[str, Any]:
    reasons: dict[str, str] = {}
    gap = None
    gap_blockers = [s.mean_reason for s in (side_a, side_b) if s.mean_reason is not None]
    if gap_blockers:
        reasons["gap"] = "; ".join(gap_blockers)
    else:
        gap = side_a.mean - side_b.mean

    scale_value: Optional[float] = None
    standardised = None
    scale_reason: Optional[str] = None
    if scales is None:
        scale_kind = "pooled"
        scale_value, scale_reason = _pooled_scale(side_a, side_b)
        if scale_value is not None and not scale_value > 0:
            scale_reason = "the pooled standard deviation is 0 (the values of each back-end tie)"
    else:
        scale_kind = "given"
        scale_value = scales.get(key)
        if scale_value is None:
            scale_reason = f"no scale given for {key!r}"
    if gap is None:
        reasons["standardised_gap"] = reasons["gap"]
    elif scale_reason is not None:
        reasons["standardised_gap"] = scale_reason
    else:
        standardised = gap / scale_value

    return {
        "gap": gap,
        "standardised_gap": standardised,
        "mean_a": side_a.mean,
        "mean_b": side_b.mean,
        "n_ok_a": side_a.n_ok,
        "n_ok_b": side_b.n_ok,
        "scale": scale_value,
        "scale_kind": scale_kind,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------- interface consensus

#: A consensus residue must reach this fraction of the valid poses, up to this relative slack:
#: ``0.1 + 0.2`` is not ``0.3`` in floating point and must not decide a residue.
_FRACTION_SLACK = 1e-9


def _check_fraction(value: Any, name: str) -> float:
    number = _number(value)
    if number is None or not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be a number in (0, 1], got {value!r}")
    return number


@dataclass(frozen=True)
class ConsensusInterface:
    """The interface residues that at least ``min_fraction`` of a back-end's valid poses share.

    Attributes:
        backend: The back-end the poses came from.
        min_fraction: The fraction of the valid poses a residue has to reach.
        n_poses: The valid poses that voted (the denominator of every fraction).
        n_total: The rows of the back-end, valid or not.
        binder: Binder positions (1-based, as ``binder_index``) in the consensus.
        receptor: Receptor residues in the consensus, as ``ResidueKey`` in the numbering of
            the reference.
        binder_fractions: Binder position -> fraction of the valid poses with it in the
            interface, for every position that is in the interface of at least one pose.
        receptor_fractions: The same for the receptor residues.
        n_unmapped_receptor: Receptor interface residues without a reference key, summed over
            the poses. They are left out of the vote: no other pose can be compared on them.
        skipped: ``{reason: count}`` of the rows that did not vote.
        reason: Why the consensus is empty because no pose voted, else None.
    """

    backend: str
    min_fraction: float
    n_poses: int
    n_total: int
    binder: frozenset
    receptor: frozenset
    binder_fractions: dict = field(default_factory=dict)
    receptor_fractions: dict = field(default_factory=dict)
    n_unmapped_receptor: int = 0
    skipped: dict = field(default_factory=dict)
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dictionary: sorted lists, receptor keys as ``[chain, number, icode]``."""
        return {
            "backend": self.backend,
            "min_fraction": self.min_fraction,
            "n_poses": self.n_poses,
            "n_total": self.n_total,
            "binder": sorted(self.binder),
            "receptor": [key.to_list() for key in sorted(self.receptor)],
            "binder_fractions": [[i, f] for i, f in sorted(self.binder_fractions.items())],
            "receptor_fractions": [
                [key.to_list(), f] for key, f in sorted(self.receptor_fractions.items())
            ],
            "n_unmapped_receptor": self.n_unmapped_receptor,
            "skipped": dict(self.skipped),
            "reason": self.reason,
        }


def _interface_of_table(table: Sequence[Any], pose_id: str):
    """``(binder positions, receptor keys, n unmapped receptor)`` of one residue table."""
    binder: set[int] = set()
    receptor: set[ResidueKey] = set()
    unmapped = 0
    for position, line in enumerate(table):
        where = f"line {position} of the residue_table of pose {pose_id!r}"
        if not isinstance(line, Mapping):
            raise ValueError(f"{where} is a {type(line).__name__}, not a mapping")
        role = line.get("role")
        if role not in ("binder", "receptor"):
            raise ValueError(f"{where} has role {role!r}, expected 'binder' or 'receptor'")
        flag = line.get("in_interface")
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{where} has in_interface {flag!r}, expected a bool")
        if not flag:
            continue
        if role == "binder":
            index = line.get("binder_index")
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError(f"{where} has binder_index {index!r}, expected an integer")
            binder.add(int(index))
        elif line.get("reference_chain") is None or line.get("reference_number") is None:
            unmapped += 1
        else:
            try:
                key = ResidueKey(
                    line["reference_chain"],
                    line["reference_number"],
                    line.get("reference_icode") or "",
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{where} has no valid reference residue: {error}") from error
            receptor.add(key)
    return binder, receptor, unmapped


def consensus_interface(
    rows: Sequence[Mapping[str, Any]], backend: str, min_fraction: float = 0.5
) -> ConsensusInterface:
    """The interface residues shared by at least ``min_fraction`` of a back-end's valid poses.

    The interface of a pose is read from the ``residue_table`` of its row (one dictionary per
    residue; columns used: ``role``, ``in_interface``, ``binder_index`` and, for a receptor
    residue, ``reference_chain``, ``reference_number``, ``reference_icode``). The receptor is
    compared in the numbering of the reference; a residue the pose cannot map to it is left
    out and counted in ``n_unmapped_receptor``. The peptide is compared by position. A pose
    is valid when its row is not ``"status": "failed"`` and has a ``residue_table``; the others
    are counted in ``skipped`` and do not vote. A residue is in the consensus when
    ``count / n_poses >= min_fraction`` (a residue held by exactly half of the poses is in the
    default consensus).

    Args:
        rows: Per-pose rows of a pose-set result, with ``provenance["backend"]``.
        backend: The back-end whose poses vote.
        min_fraction: A number in (0, 1].

    Returns:
        A ``ConsensusInterface``. With no valid pose both sets are empty, ``n_poses`` is 0 and
        ``reason`` says why.

    Raises:
        ValueError: If ``min_fraction`` is not in (0, 1], ``backend`` is not a non-empty
            string, a row is not a mapping, or a residue table line is malformed (it names
            the pose and the line).
    """
    if not isinstance(backend, str) or not backend:
        raise ValueError(f"backend must be a non-empty string, got {backend!r}")
    min_fraction = _check_fraction(min_fraction, "min_fraction")
    members = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {position} is a {type(row).__name__}, not a mapping")
        if group_key(row, ("backend",)) == backend:
            members.append((str(row.get("id", f"#{position}")), row))

    skipped: list[str] = []
    binder_counts: Counter = Counter()
    receptor_counts: Counter = Counter()
    unmapped = 0
    n_poses = 0
    for pose_id, row in members:
        if row.get("status") == "failed":
            skipped.append(str(row.get("reason") or "pose failed"))
            continue
        table = row.get("residue_table")
        if isinstance(table, (str, bytes)) or not isinstance(table, Sequence):
            skipped.append("the row has no residue_table")
            continue
        binder, receptor, pose_unmapped = _interface_of_table(table, pose_id)
        n_poses += 1
        binder_counts.update(binder)
        receptor_counts.update(receptor)
        unmapped += pose_unmapped

    def kept(counts: Counter) -> frozenset:
        return frozenset(
            item for item, count in counts.items() if _reaches(count / n_poses, min_fraction)
        )

    reason = None
    if n_poses == 0:
        reason = (
            f"no row for back-end {backend!r}"
            if not members
            else f"no valid pose of back-end {backend!r} has a residue_table"
        )
    return ConsensusInterface(
        backend=backend,
        min_fraction=min_fraction,
        n_poses=n_poses,
        n_total=len(members),
        binder=kept(binder_counts) if n_poses else frozenset(),
        receptor=kept(receptor_counts) if n_poses else frozenset(),
        binder_fractions={i: c / n_poses for i, c in sorted(binder_counts.items())},
        receptor_fractions={k: c / n_poses for k, c in sorted(receptor_counts.items())},
        n_unmapped_receptor=unmapped,
        skipped=_counted(skipped),
        reason=reason,
    )


def _reaches(fraction: float, minimum: float) -> bool:
    return fraction >= minimum or math.isclose(fraction, minimum, rel_tol=_FRACTION_SLACK)


def _counted(texts: Sequence[str]) -> dict[str, int]:
    """``{text: count}``, most frequent first, ties in alphabetical order."""
    counts = Counter(texts)
    return {text: counts[text] for text in sorted(counts, key=lambda text: (-counts[text], text))}


def _tagged(consensus: ConsensusInterface) -> frozenset:
    """Both sides in one set, each item tagged with the side it belongs to."""
    return frozenset(("binder", i) for i in consensus.binder) | frozenset(
        ("receptor", key) for key in consensus.receptor
    )


def _jaccard(first: frozenset, second: frozenset) -> Optional[float]:
    union = len(first | second)
    return len(first & second) / union if union else None


def compare_consensus_interfaces(
    consensus_a: ConsensusInterface,
    consensus_b: ConsensusInterface,
    *,
    limit: Optional[float] = None,
) -> dict[str, Any]:
    """The Jaccard index of two back-ends' consensus interfaces, and what each has alone.

    ``jaccard = |A & B| / |A | B|`` over the binder positions and the receptor residues
    together; ``jaccard_binder`` and ``jaccard_receptor`` give each side alone.

    Args:
        consensus_a: The consensus of the first back-end (``consensus_interface``).
        consensus_b: The consensus of the second.
        limit: Optional number in [0, 1]. When given, the result also carries
            ``below_limit``, True when ``jaccard < limit``.

    Returns:
        ``{"backend_a", "backend_b", "jaccard", "jaccard_binder", "jaccard_receptor",
        "only_a": {"binder": [positions], "receptor": [ResidueKey]}, "only_b": {...},
        "reasons": {field: sentence}}`` plus ``"below_limit"`` with a ``limit``. The lists are
        sorted (``ResidueKey.label`` or ``to_list()`` for text). A Jaccard of two empty sets is
        None ("both ... are empty"). When a back-end has no valid pose every field is None
        with that reason, so that an empty consensus does not read as "no interface".

    Raises:
        ValueError: If an argument is not a ``ConsensusInterface`` or ``limit`` is not a
            number in [0, 1].
    """
    for name, value in (("consensus_a", consensus_a), ("consensus_b", consensus_b)):
        if not isinstance(value, ConsensusInterface):
            raise ValueError(f"{name} must be a ConsensusInterface, got {type(value).__name__}")
    if limit is not None:
        limit_number = _number(limit)
        if limit_number is None or not 0.0 <= limit_number <= 1.0:
            raise ValueError(f"limit must be a number in [0, 1], got {limit!r}")
        limit = limit_number

    result: dict[str, Any] = {
        "backend_a": consensus_a.backend,
        "backend_b": consensus_b.backend,
    }
    fields = ("jaccard", "jaccard_binder", "jaccard_receptor", "only_a", "only_b")
    unusable = [c.reason for c in (consensus_a, consensus_b) if c.n_poses == 0]
    if unusable:
        why = "; ".join(unusable)
        result.update({name: None for name in fields})
        result["reasons"] = {name: why for name in fields}
    else:
        a, b = consensus_a, consensus_b
        result["jaccard"] = _jaccard(_tagged(a), _tagged(b))
        result["jaccard_binder"] = _jaccard(a.binder, b.binder)
        result["jaccard_receptor"] = _jaccard(a.receptor, b.receptor)
        result["only_a"] = {
            "binder": sorted(a.binder - b.binder),
            "receptor": sorted(a.receptor - b.receptor),
        }
        result["only_b"] = {
            "binder": sorted(b.binder - a.binder),
            "receptor": sorted(b.receptor - a.receptor),
        }
        result["reasons"] = {
            name: f"both consensus {label} are empty"
            for name, label in (
                ("jaccard", "interfaces"),
                ("jaccard_binder", "binder sets"),
                ("jaccard_receptor", "receptor sets"),
            )
            if result[name] is None
        }
    if limit is not None:
        result["below_limit"] = None if result["jaccard"] is None else result["jaccard"] < limit
        if result["below_limit"] is None:
            result["reasons"]["below_limit"] = result["reasons"]["jaccard"]
    return result


# ---------------------------------------------------------------------- rank correlation

RANK_METHODS = ("spearman", "kendall")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share the mean of the ranks they span (the midrank)."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def _clip_correlation(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _rank_correlation(
    x: Sequence[float], y: Sequence[float], method: str, names: tuple[str, str]
) -> tuple[Optional[float], Optional[str]]:
    """``(coefficient, None)`` or ``(None, why)`` for two equally long lists of numbers.

    Spearman's rho is the Pearson correlation of the midranks. Kendall's tau-b is
    ``sum_{i<j} sign(dx) sign(dy) / sqrt(P_x P_y)``, with ``P_x`` the number of pairs not tied
    in ``x`` and ``P_y`` the same for ``y``. Either is undefined when one list is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for values, name in zip((x, y), names):
        if np.ptp(values) == 0:
            return None, f"every value of {name} is tied, so the rank correlation is undefined"
    if method == "spearman":
        dx = _average_ranks(x)
        dy = _average_ranks(y)
        dx = dx - dx.mean()
        dy = dy - dy.mean()
        return _clip_correlation(float(dx @ dy / math.sqrt((dx @ dx) * (dy @ dy)))), None
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    return (
        _clip_correlation(float((sx * sy).sum() / math.sqrt(np.abs(sx).sum() * np.abs(sy).sum()))),
        None,
    )


MIN_CONSTRUCTS_FOR_RANK = 3


def rank_agreement(
    table: Sequence[Mapping[str, Any]],
    backend_a: str,
    backend_b: str,
    metrics: Optional[Iterable[str]] = None,
    method: str = "spearman",
) -> dict[str, dict[str, Any]]:
    """Do two back-ends order the constructs of a population the same way?

    For each metric, the rank correlation between the values of ``backend_a`` and
    ``backend_b`` over the constructs that have a valid value for both. Spearman's rho is the
    Pearson correlation of the midranks; Kendall's tau is tau-b. Both match
    ``scipy.stats.spearmanr`` and ``kendalltau``. Only exact ties count as ties.

    Args:
        table: Rows ``{"construct": id, "backend": name, "<metric>": value, ...}``, one per
            construct and back-end: a list of dictionaries, or the aggregated means of
            ``aggregate_poses`` written out in that shape. Rows of other back-ends are ignored.
        backend_a: The first back-end.
        backend_b: The second; it must differ from ``backend_a``.
        metrics: Column names to correlate, in the order reported. By default every column
            (other than ``construct`` and ``backend``) that has a number in a row of either
            back-end, sorted.
        method: ``"spearman"`` (default) or ``"kendall"``.

    Returns:
        ``{metric: {"method", "correlation": float | None, "n", "n_constructs", "n_dropped",
        "n_missing_a", "n_missing_b", "reason"}}``. A valid value is a finite number (not
        None, NaN, infinity, a bool or a string). ``n`` is the number of constructs used,
        ``n_constructs`` the constructs that appear in a row of either back-end, ``n_dropped``
        the difference, and ``n_missing_a`` and ``n_missing_b`` how many constructs lack a
        valid value for each back-end (a missing row included). ``reason`` is present only
        when ``correlation`` is None: fewer than 3 constructs, or one back-end's values all
        tie. With ``metrics=None`` and no row for the two back-ends the result is empty.

    Raises:
        ValueError: If a back-end name is not a non-empty string or the names are equal,
            ``method`` is unknown, ``metrics`` is a bare string, repeats a name or names
            ``construct`` or ``backend``, ``table`` is not a list of mappings, a row has no
            ``construct`` or no ``backend``, or a construct appears twice for one back-end.
    """
    _check_backends(backend_a, backend_b)
    if method not in RANK_METHODS:
        raise ValueError(f"method must be one of {RANK_METHODS}, got {method!r}")
    if isinstance(table, (str, bytes, Mapping)) or not isinstance(table, Iterable):
        raise ValueError("table must be a list of row dictionaries")
    rows_of: dict[str, dict[Any, Mapping[str, Any]]] = {backend_a: {}, backend_b: {}}
    for position, row in enumerate(table):
        if not isinstance(row, Mapping):
            raise ValueError(f"table row {position} is a {type(row).__name__}, not a mapping")
        for column in ("construct", "backend"):
            if row.get(column) is None:
                raise ValueError(f"table row {position} has no {column!r}")
        backend = row["backend"]
        if not isinstance(backend, str):
            raise ValueError(f"table row {position} has backend {backend!r}, expected a string")
        if backend not in rows_of:
            continue
        construct = row["construct"]
        try:
            hash(construct)
        except TypeError as error:
            raise ValueError(
                f"table row {position} has an unhashable construct {construct!r}"
            ) from error
        if construct in rows_of[backend]:
            raise ValueError(
                f"construct {construct!r} appears twice for back-end {backend!r} "
                f"(table row {position}); give one row per construct and back-end"
            )
        rows_of[backend][construct] = row

    constructs = sorted(
        set(rows_of[backend_a]) | set(rows_of[backend_b]),
        key=lambda construct: (type(construct).__name__, str(construct)),
    )
    names = _metric_names(metrics, rows_of[backend_a], rows_of[backend_b])

    result: dict[str, dict[str, Any]] = {}
    for metric in names:
        values_a = _valid_values(rows_of[backend_a], constructs, metric)
        values_b = _valid_values(rows_of[backend_b], constructs, metric)
        used = [c for c in constructs if values_a[c] is not None and values_b[c] is not None]
        entry: dict[str, Any] = {
            "method": method,
            "correlation": None,
            "n": len(used),
            "n_constructs": len(constructs),
            "n_dropped": len(constructs) - len(used),
            "n_missing_a": sum(values_a[c] is None for c in constructs),
            "n_missing_b": sum(values_b[c] is None for c in constructs),
        }
        if len(used) < MIN_CONSTRUCTS_FOR_RANK:
            entry["reason"] = (
                f"needs at least {MIN_CONSTRUCTS_FOR_RANK} constructs with a valid value for "
                f"both back-ends (n = {len(used)}, {entry['n_dropped']} dropped)"
            )
        else:
            entry["correlation"], why = _rank_correlation(
                [values_a[c] for c in used],
                [values_b[c] for c in used],
                method,
                (f"back-end {backend_a!r}", f"back-end {backend_b!r}"),
            )
            if why is not None:
                entry["reason"] = why
        result[metric] = entry
    return result


def _metric_names(metrics, rows_a: Mapping, rows_b: Mapping) -> list[str]:
    reserved = ("construct", "backend")
    if metrics is None:
        columns = {
            column
            for row in (*rows_a.values(), *rows_b.values())
            for column, value in row.items()
            if column not in reserved and _number(value) is not None
        }
        return sorted(str(column) for column in columns)
    if isinstance(metrics, str):
        raise ValueError(f"metrics must be a list of column names, got the string {metrics!r}")
    names = list(metrics)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"metrics must be non-empty column names, got {name!r}")
        if name in reserved:
            raise ValueError(f"{name!r} is not a metric: it identifies the row")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"metrics repeats {repeated}")
    return names


def _valid_values(rows: Mapping, constructs: Sequence, metric: str) -> dict[Any, Optional[float]]:
    """``{construct: finite float or None}``; None when the row or the value is missing."""
    return {
        construct: _number(rows[construct].get(metric)) if construct in rows else None
        for construct in constructs
    }


# ---------------------------------------------------------------------- hotspot consistency

#: Default size of the set of most sensitive positions.
DEFAULT_TOP_M = 3

#: Positions needed for a rank correlation between buried area and sensitivity.
MIN_POSITIONS_FOR_RANK = 3


def _read_sensitivities(sensitivities: Any) -> tuple[dict[int, float], int]:
    """``({position: finite sensitivity}, number of positions with no usable value)``."""
    if not isinstance(sensitivities, Mapping):
        raise ValueError(
            "sensitivities must be a mapping {binder position: sensitivity}, "
            f"got {type(sensitivities).__name__}"
        )
    values: dict[int, float] = {}
    missing = 0
    for position, value in sensitivities.items():
        if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
            raise ValueError(
                f"sensitivity keys must be binder positions (integers), got {position!r}"
            )
        if position < 1:
            raise ValueError(f"binder positions are 1-based, got {position!r}")
        if value is None:
            missing += 1
        elif isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"the sensitivity of position {position} must be a number, got {value!r}"
            )
        elif _number(value) is None:
            missing += 1
        else:
            values[int(position)] = float(value)
    return values, missing


def _read_binder_rows(residue_table: Any) -> dict[int, tuple[Optional[float], bool]]:
    """``{binder_index: (buried area or None, in_interface)}`` from a per-residue table."""
    if isinstance(residue_table, (str, bytes, Mapping)) or not isinstance(residue_table, Iterable):
        raise ValueError("residue_table must be a list of row dictionaries")
    binder: dict[int, tuple[Optional[float], bool]] = {}
    for position, line in enumerate(residue_table):
        where = f"line {position} of the residue table"
        if not isinstance(line, Mapping):
            raise ValueError(f"{where} is a {type(line).__name__}, not a mapping")
        role = line.get("role")
        if role not in ("binder", "receptor"):
            raise ValueError(f"{where} has role {role!r}, expected 'binder' or 'receptor'")
        if role == "receptor":
            continue
        for column in ("binder_index", "buried_area_angstrom2", "in_interface"):
            if column not in line:
                raise ValueError(f"{where} is a binder line without {column!r}")
        index = line["binder_index"]
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"{where} has binder_index {index!r}, expected an integer")
        flag = line["in_interface"]
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{where} has in_interface {flag!r}, expected a bool")
        area = line["buried_area_angstrom2"]
        if area is not None and (
            isinstance(area, (bool, np.bool_))
            or not isinstance(area, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"{where} has buried_area_angstrom2 {area!r}, expected a number")
        if int(index) in binder:
            raise ValueError(f"binder position {int(index)} appears twice in the residue table")
        binder[int(index)] = (_number(area), bool(flag))
    return binder


def hotspot_consistency(
    sensitivities: Mapping[int, Optional[float]],
    residue_table: Iterable[Mapping[str, Any]],
    top_m: int = DEFAULT_TOP_M,
) -> dict[str, Any]:
    """Do the peptide positions that an alanine scan finds sensitive sit in the interface?

    Two numbers for one pose, from the sensitivities of the scan and the per-residue interface
    table of that pose (``buried_area_angstrom2`` and ``in_interface`` of the binder lines):

    * ``spearman_area_vs_sensitivity``: Spearman's rho between the buried area and the
      sensitivity over the positions that have both.
    * ``top_in_interface_fraction``: the fraction of the ``top_m`` most sensitive positions
      that are in the interface.

    A larger sensitivity means a more important position; pass a scan that scores the other
    way with the sign changed. Positions are 1-based positions in the binder chain, as
    ``binder_index``. Only positions with a finite sensitivity and a finite buried area are
    used; the others are counted.

    Tie rule at the cut: every position whose sensitivity is at least that of the
    ``top_m``-th most sensitive position is in the top set, so the set has more than
    ``top_m`` members when a tie straddles the cut, and the fraction is taken over the whole
    set. When all sensitivities tie, the top set is every position. The result does not
    depend on the order of the table or of the mapping.

    Args:
        sensitivities: ``{binder position: sensitivity}``. None and non-finite values are
            positions without a sensitivity (a glycine or alanine that cannot be scanned).
        residue_table: The per-residue table of one pose, one dictionary per residue; receptor
            lines are ignored. Binder lines need ``binder_index``, ``buried_area_angstrom2``
            (None for an unknown area) and ``in_interface``.
        top_m: Size of the top set, at least 1 (default ``DEFAULT_TOP_M``, 3).

    Returns:
        A JSON-ready dictionary: ``spearman_area_vs_sensitivity`` and
        ``top_in_interface_fraction`` (each None with a reason), ``top_m``, ``n_top`` (the size
        of the top set), ``n_top_in_interface``, ``top_positions`` (most sensitive first, ties
        by position), ``n_positions`` (positions used), ``n_missing_sensitivity`` (positions
        of ``sensitivities`` with no usable value), ``n_unmatched_positions`` (positions with
        a sensitivity but no usable area in the table) and ``reasons`` (``{field: sentence}``,
        empty when both numbers were computed). The Spearman value needs ``n_positions >= 3``
        and a spread in both lists; the fraction needs ``n_positions >= top_m``. When the
        fraction is None, ``n_top``, ``n_top_in_interface`` and ``top_positions`` are None.
        Stored in a row (``row["hotspots"]``), the numbers are summarised by
        ``aggregate_poses`` like any other block.

    Raises:
        ValueError: If ``sensitivities`` is not a mapping with positive integer keys and
            numeric or None values, ``residue_table`` is not a list of well-formed lines, a
            binder position occurs twice in it, or ``top_m`` is not an integer of at least 1.
    """
    if isinstance(top_m, (bool, np.bool_)) or not isinstance(top_m, (int, np.integer)) or top_m < 1:
        raise ValueError(f"top_m must be an integer of at least 1, got {top_m!r}")
    top_m = int(top_m)
    scan, n_missing = _read_sensitivities(sensitivities)
    table = _read_binder_rows(residue_table)

    used = sorted(p for p in scan if p in table and table[p][0] is not None)
    n_positions = len(used)
    reasons: dict[str, str] = {}
    result: dict[str, Any] = {
        "spearman_area_vs_sensitivity": None,
        "top_in_interface_fraction": None,
        "top_m": top_m,
        "n_top": None,
        "n_top_in_interface": None,
        "top_positions": None,
        "n_positions": n_positions,
        "n_missing_sensitivity": n_missing,
        "n_unmatched_positions": len(scan) - n_positions,
        "reasons": reasons,
    }

    if n_positions < MIN_POSITIONS_FOR_RANK:
        reasons["spearman_area_vs_sensitivity"] = (
            f"needs at least {MIN_POSITIONS_FOR_RANK} positions with both a sensitivity and a "
            f"buried area (n = {n_positions})"
        )
    else:
        rho, why = _rank_correlation(
            [table[p][0] for p in used],
            [scan[p] for p in used],
            "spearman",
            ("the buried area", "the sensitivity"),
        )
        result["spearman_area_vs_sensitivity"] = rho
        if why is not None:
            reasons["spearman_area_vs_sensitivity"] = why

    if n_positions < top_m:
        reasons["top_in_interface_fraction"] = (
            f"only {n_positions} positions have both a sensitivity and a buried area, "
            f"fewer than top_m = {top_m}"
        )
    else:
        ranked = sorted(used, key=lambda p: (-scan[p], p))
        cut = scan[ranked[top_m - 1]]
        top = [p for p in ranked if scan[p] >= cut]
        inside = sum(table[p][1] for p in top)
        result["top_in_interface_fraction"] = inside / len(top)
        result["n_top"] = len(top)
        result["n_top_in_interface"] = inside
        result["top_positions"] = top
    return result
