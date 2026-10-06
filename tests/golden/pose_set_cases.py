"""Golden pose-set results: the cases, how they are compared and how the files are written.

A golden file holds the ``score_pose_set`` result of one fixed call, written by
``scripts/regen_pose_golden.py``; the tests in ``tests/poses/test_golden_*.py`` rebuild the
result and compare it with the file. This module is the one place that defines the calls, the
tolerance and the file layout, so the tests and the script cannot drift apart.

Cases
-----
``synthetic_1ycr``
    ``make_synthetic_poses()`` (two back-ends, three seeds, every pose in its own random frame)
    scored against the unmoved 1YCR complex with the epitope as a residue set and the crystal
    peptide as a partner, with a few cheap static metrics (``SYNTHETIC_METRICS``). It is what an
    engine reproduces with ``make_synthetic_poses()`` and its own call.
``descriptors_1ycr``
    The crystal peptide and the same peptide translated by 6 angstrom, on the public complex
    1YCR, with two residue sets and the crystal peptide as partner. No frame noise: the
    transform is the identity, so the surface sampling is the one of a plain ``compute_*`` call
    on the file. ``tests/poses/test_golden_descriptors.py`` recomputes the set overlap, the
    shared buried area and the contact columns from the file with independent code.

What is compared
----------------
Everything in the result except what describes the run or the machine: ``timing``, the
top-level ``provenance`` (git SHA, Python, OS), ``package_version`` and the file paths of the
reference structures (``IGNORED``).
The result is flattened to ``dotted.path[index]`` leaves. Booleans, integers, strings and None
must be equal and of the same type. A float agrees when
``|live - golden| <= atol + rtol * |golden|`` (the rule of ``numpy.isclose``), with
``rtol = atol = 1e-6`` unless a pattern of ``TOLERANCE_OVERRIDES`` names the path (none is
needed today). The default is far above the rounding noise of the computation (about 1e-12
relative) and far below any change of an algorithm: a flipped surface sampling point moves a
buried area by 0.1 square angstrom, 1e-4 relative. A leaf that exists on one side only, or a
list of another length, is a mismatch. Key order is checked by
``tests/poses/test_contract_result.py``, not here.

Golden files are written with floats rounded to 10 significant digits and values below 1e-12
written as 0.0, so that regenerating on another machine does not rewrite noise.
"""

from __future__ import annotations

import fnmatch
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

GOLDEN_DIR = Path(__file__).resolve().parent / "pose_sets"
FORMAT_VERSION = 1

#: Default tolerance of a float: ``|live - golden| <= ATOL + RTOL * |golden|``.
RTOL = 1e-6
ATOL = 1e-6
#: ``{path pattern: (rtol, atol)}`` for a value that needs more than the default. A pattern is
#: matched with ``fnmatch.fnmatchcase`` against the leaf path, ``*`` matching any characters.
#: Each entry needs a comment that says why the value is noisier than the rest.
TOLERANCE_OVERRIDES: dict[str, tuple[float, float]] = {}

#: Leaf paths (``fnmatch`` patterns) that describe the run or the machine, not the result.
IGNORED = (
    "package_version",
    "timing*",
    "provenance*",
    "references.*.path",
)

_SIGNIFICANT_DIGITS = 10
_SNAP_TO_ZERO = 1e-12

#: Receptor residues of 1YCR chain A with a heavy atom within 4.5 angstrom of a heavy atom of the
#: crystal peptide (chain B); the 22 residues of the design's worked example.
EPITOPE_4_5_ANGSTROM = (
    25, 26, 50, 51, 54, 57, 58, 61, 62, 67, 70,
    71, 72, 73, 75, 91, 93, 94, 96, 99, 100, 104,
)  # fmt: skip

#: Receptor residues of 1YCR chain A that the crystal peptide buries by at least 0.5 square
#: angstrom (the interface of ``compute_interface_metrics``).
CRYSTAL_INTERFACE = (
    25, 26, 49, 50, 51, 54, 57, 58, 61, 62, 67, 70, 71,
    72, 73, 75, 86, 91, 93, 94, 96, 99, 100, 103, 104,
)  # fmt: skip

SYNTHETIC_METRICS = ("delta_sasa_static", "omega", "coulomb")
DESCRIPTOR_METRICS = ("delta_sasa_static",)

DECOY_OFFSETS_ANGSTROM = {"crystal": (0.0,), "decoy_6A": (6.0,)}


# ---------------------------------------------------------------------------- the cases


def reference_1ycr(complex_pose):
    """The reference of both cases: the unmoved 1YCR complex, two sets, the peptide as partner."""
    from binding_metrics.poses import Partner, Reference, ResidueKey

    return Reference(
        complex_pose,
        "A",
        "B",
        residue_sets={
            "epitope": [ResidueKey("A", number) for number in EPITOPE_4_5_ANGSTROM],
            "crystal_interface": [ResidueKey("A", number) for number in CRYSTAL_INTERFACE],
        },
        partners={"crystal_peptide": Partner(complex_pose, chains=["B"])},
    )


def synthetic_poses():
    """The synthetic set of the first case: the default call of ``make_synthetic_poses``."""
    from binding_metrics.testing import make_synthetic_poses

    return make_synthetic_poses()


def decoy_poses():
    """The poses of the second case: the crystal peptide and the same peptide shifted 6 angstrom.

    One back-end each, a shift along the generator's default direction, no frame noise.
    """
    from binding_metrics.testing import make_synthetic_poses

    return make_synthetic_poses(
        backends=tuple(DECOY_OFFSETS_ANGSTROM),
        seeds=(1,),
        offsets_angstrom=DECOY_OFFSETS_ANGSTROM,
        frame_noise=False,
    )


def run_synthetic_1ycr():
    from binding_metrics.poses import score_pose_set

    synthetic = synthetic_poses()
    return score_pose_set(
        synthetic.poses,
        "B",
        "A",
        metrics=SYNTHETIC_METRICS,
        reference=reference_1ycr(synthetic.reference),
        construct="synthetic-1ycr",
    )


def run_descriptors_1ycr():
    from binding_metrics.poses import score_pose_set

    decoys = decoy_poses()
    return score_pose_set(
        decoys.poses,
        "B",
        "A",
        metrics=DESCRIPTOR_METRICS,
        reference=reference_1ycr(decoys.reference),
        construct="1ycr-descriptors",
    )


@dataclass(frozen=True)
class GoldenCase:
    """One golden file: its name, what it holds and the function that makes the result."""

    name: str
    description: str
    run: Callable[[], Any]

    @property
    def path(self) -> Path:
        return GOLDEN_DIR / f"{self.name}.json"


def _metrics_phrase(names) -> str:
    """``metric a`` or ``metrics a, b and c``, from the tuple the case runs."""
    names = list(names)
    if len(names) == 1:
        return f"metric {names[0]}"
    return f"metrics {', '.join(names[:-1])} and {names[-1]}"


CASES = {
    case.name: case
    for case in (
        GoldenCase(
            "synthetic_1ycr",
            "score_pose_set on make_synthetic_poses() (backend_a and backend_b, seeds 1 to 3, "
            "random frames) against the unmoved 1YCR complex, with the 4.5 angstrom epitope and "
            "the buried-area interface as residue sets, the crystal peptide as partner and the "
            f"{_metrics_phrase(SYNTHETIC_METRICS)}.",
            run_synthetic_1ycr,
        ),
        GoldenCase(
            "descriptors_1ycr",
            "score_pose_set on the crystal peptide of 1YCR and the same peptide shifted 6 "
            "angstrom along (2/3, -1/3, 2/3), identity frame, with the 4.5 angstrom epitope and "
            "the buried-area interface as residue sets, the crystal peptide as partner and the "
            f"{_metrics_phrase(DESCRIPTOR_METRICS)}.",
            run_descriptors_1ycr,
        ),
    )
}


# ---------------------------------------------------------------------------- the file format


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    if abs(value) < _SNAP_TO_ZERO:
        return 0.0
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def canonical(value: Any) -> Any:
    """Plain JSON values with floats rounded for the file (see the module docstring)."""
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        return _round_float(value)
    return value


def strip_ignored(value: Any, path: str = "") -> Any:
    """The result without the keys of ``IGNORED``, so that the file holds no run-specific text."""
    if isinstance(value, dict):
        return {
            key: strip_ignored(item, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
            if not is_ignored(f"{path}.{key}" if path else str(key))
        }
    if isinstance(value, (list, tuple)):
        return [strip_ignored(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return value


def golden_document(case: GoldenCase, result: Any) -> dict:
    """The content of the golden file of ``case`` for a ``PoseSetResult`` (not yet written)."""
    return {
        "golden_format": FORMAT_VERSION,
        "name": case.name,
        "description": case.description,
        "regenerate": "python scripts/regen_pose_golden.py --write",
        "tolerance": {"rtol": RTOL, "atol": ATOL, "overrides": TOLERANCE_OVERRIDES},
        "ignored": list(IGNORED),
        "result": canonical(strip_ignored(result.to_dict())),
    }


_ONE_LINE_LIMIT = 260


def _is_flat_list(value: Any) -> bool:
    """A list of scalars, or of lists of scalars (a vector or a matrix)."""
    return isinstance(value, list) and all(
        not isinstance(item, dict)
        and (not isinstance(item, list) or all(not isinstance(x, (dict, list)) for x in item))
        for item in value
    )


def _is_simple(value: Any) -> bool:
    """A scalar, a flat list, or a dict of those (at any depth)."""
    if isinstance(value, dict):
        return all(_is_simple(item) for item in value.values())
    return _is_flat_list(value) if isinstance(value, list) else True


def dumps(value: Any, level: int = 0) -> str:
    """JSON text: a key per line; vectors, matrices and short flat dicts on one line.

    Readable in a diff, and the text is the same for the same data on every platform.
    """
    pad = "  " * level
    if _is_simple(value):
        flat = json.dumps(value, ensure_ascii=False)
        if not isinstance(value, dict) or len(flat) <= _ONE_LINE_LIMIT:
            return flat
    if isinstance(value, dict):
        inner = ",\n".join(
            f"{pad}  {json.dumps(key, ensure_ascii=False)}: {dumps(item, level + 1)}"
            for key, item in value.items()
        )
        return "{\n" + inner + "\n" + pad + "}"
    inner = ",\n".join(f"{pad}  {dumps(item, level + 1)}" for item in value)
    return "[\n" + inner + "\n" + pad + "]"


def write_golden(path: Path, document: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dumps(document) + "\n", encoding="utf-8", newline="\n")


def read_golden(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- comparison


def flatten(value: Any, path: str = "") -> dict[str, Any]:
    """``{leaf path: leaf}`` of a nested result; an empty dict or list is a leaf."""
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(flatten(item, f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(value, (list, tuple)) and value:
        out = {}
        for index, item in enumerate(value):
            out.update(flatten(item, f"{path}[{index}]"))
        return out
    if isinstance(value, (list, tuple)):
        return {path: []}
    return {path: value}


def is_ignored(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in IGNORED)


def tolerance_for(path: str) -> tuple[float, float]:
    for pattern, tolerance in TOLERANCE_OVERRIDES.items():
        if fnmatch.fnmatchcase(path, pattern):
            return tolerance
    return RTOL, ATOL


@dataclass(frozen=True)
class Mismatch:
    """One leaf that differs: ``kind`` is ``changed``, ``type``, ``missing`` or ``extra``."""

    path: str
    kind: str
    golden: Any = None
    live: Any = None

    def describe(self) -> str:
        if self.kind == "missing":
            return f"{self.path}: in the golden file ({self.golden!r}), absent from the result"
        if self.kind == "extra":
            return f"{self.path}: in the result ({self.live!r}), absent from the golden file"
        if self.kind == "type":
            return (
                f"{self.path}: type changed, golden {self.golden!r} "
                f"({type(self.golden).__name__}), result {self.live!r} ({type(self.live).__name__})"
            )
        if isinstance(self.golden, float) and isinstance(self.live, float):
            rtol, atol = tolerance_for(self.path)
            difference = abs(self.live - self.golden)
            relative = difference / abs(self.golden) if self.golden else math.inf
            return (
                f"{self.path}: golden {self.golden!r}, result {self.live!r} "
                f"(difference {difference:.3g}, relative {relative:.3g}; "
                f"tolerance rtol={rtol:g} atol={atol:g})"
            )
        return f"{self.path}: golden {self.golden!r}, result {self.live!r}"


def _same(golden: Any, live: Any, path: str) -> str | None:
    """None when the two leaves agree, else the mismatch kind."""
    if isinstance(golden, float) or isinstance(live, float):
        if not (isinstance(golden, float) and isinstance(live, float)):
            return "type"
        if math.isnan(golden) or math.isnan(live):
            return None if math.isnan(golden) and math.isnan(live) else "changed"
        rtol, atol = tolerance_for(path)
        return None if abs(live - golden) <= atol + rtol * abs(golden) else "changed"
    if type(golden) is not type(live):
        return "type"
    return None if golden == live else "changed"


def compare(golden: Any, live: Any) -> list[Mismatch]:
    """Every difference between a golden result and a live one, in the order of the golden file."""
    gold = {p: v for p, v in flatten(golden).items() if not is_ignored(p)}
    now = {p: v for p, v in flatten(live).items() if not is_ignored(p)}
    mismatches = []
    for path, value in gold.items():
        if path not in now:
            mismatches.append(Mismatch(path, "missing", golden=value))
            continue
        kind = _same(value, now[path], path)
        if kind is not None:
            mismatches.append(Mismatch(path, kind, golden=value, live=now[path]))
    mismatches.extend(Mismatch(p, "extra", live=v) for p, v in now.items() if p not in gold)
    return mismatches


def failure_message(case_name: str, mismatches: list[Mismatch], limit: int = 12) -> str:
    """The text of a failed comparison: the keys that differ and what to do about it."""
    shown = "\n".join("  " + mismatch.describe() for mismatch in mismatches[:limit])
    more = f"\n  ... and {len(mismatches) - limit} more" if len(mismatches) > limit else ""
    return (
        f"the pose-set result of golden case '{case_name}' differs from tests/golden/pose_sets/"
        f"{case_name}.json in {len(mismatches)} value(s):\n{shown}{more}\n"
        "If the change is intended, run `python scripts/regen_pose_golden.py --write`, read "
        "the diff, and give the commit a regression note (one named value, before -> after)."
    )
