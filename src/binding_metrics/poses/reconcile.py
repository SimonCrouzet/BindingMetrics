"""Cross-model reconciliation: how far two back-ends agree on the same constructs.

Pure functions on the plain dictionaries of the pose-set layer, in Python and numpy (no
pandas, no scipy). They read the aggregates of ``poses.aggregate`` and the per-pose rows of a
pose-set result, report numbers, and never judge them: a threshold is an argument, and the
engine decides what a gap or a correlation means.

* ``gap_between_backends``: the difference of two back-ends' means per aggregated key, and
  the same difference in units of the spread over seeds.

A value that cannot be computed is None with a reason, never NaN. A reason is a short sentence
under the name of the field it explains (``reasons["standardised_gap"]``); the dictionary is
empty when every field was computed. Missing data is not an error: a back-end that is absent
from the aggregates (a set with a single model) gives None with a reason. A malformed argument
raises ``ValueError`` before any value is computed.

Importing this module imports numpy and the standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import numpy as np

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

    ``gap = mean_a - mean_b``: a positive gap means back-end ``a`` has the larger mean, and the
    sign follows the order of ``backend_a`` and ``backend_b`` in the call.

    ``standardised_gap = gap / s_pooled`` with the standard deviation pooled over the seeds of
    the two back-ends (the denominator of Cohen's d),
    ``s_pooled = sqrt(((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2))``, where ``s_a``
    and ``s_b`` are the sample standard deviations (ddof = 1) of the back-ends' valid values
    and ``n_a``, ``n_b`` their counts. It needs ``n_a >= 2``, ``n_b >= 2`` and
    ``s_pooled > 0``. With three seeds per back-end the pooled value has 4 degrees of freedom
    and is unstable, so ``scale`` can also be a mapping with a scale of the caller's choosing
    for each key, for instance the spread of the same key over constructs. Cohen, *Statistical
    Power Analysis for the Behavioral Sciences*, 2nd ed., 1988.

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
