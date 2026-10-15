"""Strict JSON export of result files: NaN and infinity written as ``null``.

``json.dumps`` writes a NaN or an infinity as the token ``NaN``, ``Infinity`` or ``-Infinity``.
Python's ``json`` reads the tokens back; the parsers of JavaScript, ``jq`` and most databases
refuse them. Every writer of a result file takes ``nonfinite``:

``"keep"`` (default)
    The tokens are written as they always were.

``"null"``
    Every NaN and infinity is replaced by ``null`` in a copy, the paths replaced join the
    top-level list ``nonfinite_fields`` (an entry that is already listed stays, none is listed
    twice, and the list is added when the result has none), and the text is made with
    ``allow_nan=False``, so no token can reach the file.

A path is the dotted key path with ``name[3]`` for a list item: ``relax.rmsd_md_final``,
``poses[0].interface.delta_sasa``, ``geometry.series[1]``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

#: The values of ``nonfinite``; the first is the default.
NONFINITE_MODES = ("keep", "null")

#: The top-level key that lists the paths of the values that are NaN or infinite.
NONFINITE_FIELDS_KEY = "nonfinite_fields"


def check_nonfinite(nonfinite: str) -> str:
    """Return ``nonfinite`` when it is one of ``NONFINITE_MODES``.

    Raises:
        ValueError: For any other value; the message names the allowed ones.
    """
    if nonfinite not in NONFINITE_MODES:
        raise ValueError(f"nonfinite must be one of {list(NONFINITE_MODES)}, got {nonfinite!r}")
    return nonfinite


def is_nonfinite(value: Any) -> bool:
    """True for NaN and +-infinity, as a Python float or a numpy floating scalar."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, np.floating):
        return not bool(np.isfinite(value))
    return False


def _walk(value: Any, path: str, found: list[str]) -> Any:
    """A copy of ``value`` with None for every nonfinite number; the paths join ``found``."""
    if isinstance(value, Mapping):
        return {
            key: _walk(child, f"{path}.{key}" if path else str(key), found)
            for key, child in value.items()
        }
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_walk(child, f"{path}[{index}]", found) for index, child in enumerate(value)]
    if is_nonfinite(value):
        found.append(path)
        return None
    return value


def null_nonfinite(value: Any) -> tuple[Any, list[str]]:
    """A copy of a nested result with None in place of every NaN and infinity.

    Dictionaries stay dictionaries; lists, tuples and numpy arrays become lists. A number
    that is finite and any other object are returned as they are, so a ``default`` function of
    ``json.dumps`` still sees them.

    Returns:
        ``(copy, paths)``: the paths of the replaced values in the order of the traversal.
    """
    found: list[str] = []
    return _walk(value, "", found), found


def nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    """The paths of every NaN and infinity in a nested result (``prefix`` is put before each)."""
    found: list[str] = []
    _walk(value, prefix, found)
    return found


def strict_json_text(data: Mapping[str, Any], **dumps_kwargs: Any) -> str:
    """The JSON text of a result that a strict parser accepts.

    NaN and infinity become ``null``, and ``data["nonfinite_fields"]`` lists them after the
    entries it already holds. ``data`` is not changed.

    Args:
        data: The result, a mapping.
        **dumps_kwargs: Keywords of ``json.dumps`` (``indent``, ``default``, ``ensure_ascii``).
            ``allow_nan`` is always False.

    Raises:
        ValueError: If a nonfinite number reaches the encoder anyway, for example from the
            ``default`` function.
    """
    copy, replaced = null_nonfinite(data)
    listed = copy.get(NONFINITE_FIELDS_KEY)
    listed = list(listed) if isinstance(listed, list) else []
    copy[NONFINITE_FIELDS_KEY] = listed + [path for path in replaced if path not in listed]
    return json.dumps(copy, allow_nan=False, **dumps_kwargs)


def add_nonfinite_arg(parser) -> None:
    """Add ``--nonfinite {keep,null}`` (default ``keep``) to an argparse parser or group."""
    parser.add_argument(
        "--nonfinite",
        choices=NONFINITE_MODES,
        default=NONFINITE_MODES[0],
        help="NaN and infinity in the JSON results: 'keep' writes the tokens NaN, Infinity and "
        "-Infinity; 'null' writes null and lists the paths in nonfinite_fields, for parsers "
        "that accept only standard JSON (default: %(default)s)",
    )
