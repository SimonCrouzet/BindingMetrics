"""Common-frame alignment: place every pose in the frame of one reference receptor.

For one pose and one ``Reference`` the stage

1. pairs the receptor residues of the pose with those of the reference (``pair_residues``),
2. superposes the pose on the reference with a Kabsch fit on the paired receptor C-alpha
   atoms (``superpose_receptor``), and
3. reports the frame: the pairing method and counts, the receptor RMSD of the fit, and the
   C-alpha RMSD of named regions in the final frame.

``place_in_reference_frame`` applies the transform to the atoms of the pose;
``renumber_to_reference`` gives the paired receptor residues the numbers of the reference, and
the binder residues too when ``pair_binder_residues`` paired them with the reference binder.
Nothing here modifies its input.

Pairing (first success wins; the method used is reported)
---------------------------------------------------------
``map``
    A ``residue_map`` (``Reference.residue_map``, or the one given to the call) pairs the
    residues it names that both structures have. With ``pairing="auto"`` a map wins over the
    other methods.
``number``
    Same ``(number, insertion code)`` and compatible residue name, when at least
    ``min_pair_fraction`` of the shorter chain pairs this way. Two names are compatible when
    they are equal after protonation variants (HID, HIE, ... to HIS), D-amino-acid codes (DAL
    to ALA) and four common modified residues (MSE, SEP, TPO, PTR) are replaced by the amino
    acid they stand for.
``sequence``
    Global pairwise alignment of the one-letter sequences of the two receptor chains (a
    residue that is not one of the 20 amino acids or a variant above is ``X``) with
    ``biotite.sequence.align.align_optimal``: substitution matrix BLOSUM62, affine gap penalty
    open -10 and extend -1 (biotite's convention for ``(-10, -1)``), no penalty for gaps at
    the ends, so a trimmed construct aligns inside the longer chain. The aligned columns whose
    two residues have compatible names are paired. When several alignments score the same,
    the first one in biotite's order is used and ``alignment_ambiguous`` is True.
``position``
    Opt-in (``pairing="position"``): the i-th residue to the i-th, equal lengths required.

A pose fails (``status="failed"``, a ``reason``, no transform) when its receptor chain is
missing, when the sequence identity of the pairing is below ``min_sequence_identity``, when
fewer than ``min_frame_fraction`` of the reference receptor residues are paired, when fewer
than ``min_frame_residues`` residues are left to fit, or when the fit atoms of either structure
are collinear. The identity limit does not apply to a ``residue_map``, which states the
correspondence itself.

Fit
---
Plain Kabsch (a proper rotation; a reflection is never used) on the C-alpha atoms of all paired
residues, no outlier rejection. The transform maps a pose coordinate ``x`` to the reference
frame as ``rotation @ x + translation`` (column vectors), which for an array of row
coordinates is ``coords @ rotation.T + translation``. Options: ``frame_atoms="backbone"`` fits
the N, CA, C and O atoms that both residues have; ``frame_residues`` fits only the paired
residues of a named residue set of the reference, or of the binding site (``"site"``);
``outliers="iqr"`` rejects residues iteratively (the algorithm of
``biotite.structure.superimpose_without_outliers``: a residue is an outlier when its squared
deviation exceeds the upper quartile by ``iqr_threshold`` interquartile ranges, at most
``iqr_max_iterations`` rounds), with two guards: a residue that deviates by less than
``iqr_min_deviation_angstrom`` is never rejected, and rejection stops before fewer than
``min_frame_residues`` residues would be left.

Determinism
-----------
No random numbers; everything is computed in float64. Degenerate inputs are refused with a
reason, never fitted: ``min_frame_residues`` is at least 3, and a fit set whose atoms are
collinear or coincident, in the pose or in the reference, has no unique rotation and fails the
pose. Sequence-alignment ties are broken by biotite's order, which biotite does not document as
stable across versions; they are flagged by ``alignment_ambiguous``.

Regions
-------
``region_rmsd_angstrom`` holds the C-alpha RMSD of the paired residues of a region in the final
frame, with no refit: ``"all"`` (every paired residue), ``"site"`` (receptor residues with a
heavy atom within ``site_cutoff_angstrom`` of a heavy atom of the reference binder chain;
present only when the reference has a ``binder_chain``) and every named residue set of the
reference. A region with no paired residue has ``None`` and a reason in ``region_reasons``.

Importing this module imports numpy and the standard library (and the reference module);
biotite is imported when a structure is read and when a sequence alignment is needed.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import asdict, dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from binding_metrics.poses.pose import Pose
from binding_metrics.poses.reference import (
    Reference,
    ResidueKey,
    polymer_residues,
)

logger = logging.getLogger(__name__)

PAIRING_METHODS = ("auto", "map", "number", "sequence", "position")
FRAME_ATOMS = ("ca", "backbone")
OUTLIER_METHODS = (None, "iqr")
METRICS_FRAMES = ("reference", "original")
_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
_COLLINEAR_RATIO = 1e-6


# ---------------------------------------------------------------------------- options


@dataclass(frozen=True)
class FrameOptions:
    """Every threshold and choice of the frame stage, with its default.

    Attributes:
        pairing: ``"auto"`` (map, then number, then sequence), or one method: ``"map"``,
            ``"number"``, ``"sequence"``, ``"position"``.
        frame_atoms: ``"ca"`` (C-alpha of the paired residues) or ``"backbone"`` (N, CA, C, O).
        frame_residues: None (fit all paired residues), the name of a residue set of the
            reference, or ``"site"`` (the binding site; the reference needs a binder chain):
            fit only the paired residues of that set.
        outliers: None (no rejection) or ``"iqr"``.
        min_frame_residues: Fewest residues to fit, at least 3; a pose with fewer fails.
        min_frame_fraction: Smallest fraction of the reference receptor residues that must be
            paired (0 to 1); a pose below it fails.
        min_sequence_identity: Smallest fraction of aligned or same-numbered residues with
            compatible names (0 to 1); a pose below it fails. Not applied to a ``residue_map``.
        min_pair_fraction: ``"number"`` pairing is used only when at least this fraction of
            the shorter chain pairs by number and name (0 to 1).
        site_cutoff_angstrom: Distance (any heavy atom to any heavy atom of the reference
            binder) that makes a receptor residue part of the binding site.
        iqr_threshold: Interquartile ranges above the upper quartile that make an outlier.
        iqr_max_iterations: Most rounds of outlier rejection.
        iqr_min_deviation_angstrom: A residue that deviates by less is never an outlier.
        alignment_matrix: Name of the biotite substitution matrix of the sequence alignment.
        alignment_gap_penalty: ``(open, extend)``, both negative or zero integers.
        renumber_receptor: Read by the scoring stage: write the structure of the registry
            metrics with the reference numbers on the paired receptor residues and, when the
            reference has a binder chain that pairs with the binder of the pose, on the binder
            residues (see ``renumber_to_reference`` and ``pair_binder_residues``).
        metrics_frame: Read by the scoring stage: ``"reference"`` computes path-based metrics
            on the transformed pose, ``"original"`` on the pose's own file.
    """

    pairing: str = "auto"
    frame_atoms: str = "ca"
    frame_residues: Optional[str] = None
    outliers: Optional[str] = None
    min_frame_residues: int = 10
    min_frame_fraction: float = 0.5
    min_sequence_identity: float = 0.8
    min_pair_fraction: float = 0.9
    site_cutoff_angstrom: float = 4.5
    iqr_threshold: float = 1.5
    iqr_max_iterations: int = 10
    iqr_min_deviation_angstrom: float = 0.01
    alignment_matrix: str = "BLOSUM62"
    alignment_gap_penalty: tuple = (-10, -1)
    renumber_receptor: bool = True
    metrics_frame: str = "reference"

    def __post_init__(self):
        _one_of("pairing", self.pairing, PAIRING_METHODS)
        _one_of("frame_atoms", self.frame_atoms, FRAME_ATOMS)
        _one_of("outliers", self.outliers, OUTLIER_METHODS)
        _one_of("metrics_frame", self.metrics_frame, METRICS_FRAMES)
        if self.frame_residues is not None and not (
            isinstance(self.frame_residues, str) and self.frame_residues
        ):
            raise ValueError(
                f"frame_residues must be None or a residue set name, got {self.frame_residues!r}"
            )
        _check_int("min_frame_residues", self.min_frame_residues, minimum=3)
        _check_int("iqr_max_iterations", self.iqr_max_iterations, minimum=1)
        for name in ("min_frame_fraction", "min_sequence_identity", "min_pair_fraction"):
            _check_number(name, getattr(self, name), minimum=0.0, maximum=1.0)
        _check_number(
            "site_cutoff_angstrom", self.site_cutoff_angstrom, minimum=0.0, exclusive=True
        )
        _check_number("iqr_threshold", self.iqr_threshold, minimum=0.0)
        _check_number("iqr_min_deviation_angstrom", self.iqr_min_deviation_angstrom, minimum=0.0)
        if not (isinstance(self.alignment_matrix, str) and self.alignment_matrix):
            raise ValueError(
                f"alignment_matrix must be a matrix name, got {self.alignment_matrix!r}"
            )
        gap = self.alignment_gap_penalty
        try:
            gap = tuple(gap)
        except TypeError:
            gap = ()
        if len(gap) != 2 or not all(
            isinstance(g, (int, np.integer)) and not isinstance(g, bool) and g <= 0 for g in gap
        ):
            raise ValueError(
                "alignment_gap_penalty must be (open, extend), two integers that are not "
                f"positive, got {self.alignment_gap_penalty!r}"
            )
        object.__setattr__(self, "alignment_gap_penalty", (int(gap[0]), int(gap[1])))
        if not isinstance(self.renumber_receptor, bool):
            raise TypeError(
                f"renumber_receptor must be True or False, got {self.renumber_receptor!r}"
            )

    def to_dict(self) -> dict:
        """The options as a JSON-ready dictionary (the gap penalty as a list)."""
        data = asdict(self)
        data["alignment_gap_penalty"] = list(self.alignment_gap_penalty)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameOptions:
        """Inverse of ``to_dict``; a missing key takes its default.

        Raises:
            ValueError: If a key is not an option.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown frame options {unknown}; the options are {sorted(known)}")
        return cls(**dict(data))


def _as_options(options: Any) -> FrameOptions:
    if options is None:
        return FrameOptions()
    if isinstance(options, FrameOptions):
        return options
    if isinstance(options, Mapping):
        return FrameOptions.from_dict(options)
    raise TypeError(
        f"options must be FrameOptions, a mapping or None, got {type(options).__name__}"
    )


def _one_of(name: str, value: Any, allowed: Sequence) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _check_int(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _check_number(
    name: str, value: Any, *, minimum: float, maximum: float = math.inf, exclusive: bool = False
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    too_low = value <= minimum if exclusive else value < minimum
    if not math.isfinite(value) or too_low or value > maximum:
        low = f"above {minimum}" if exclusive else f"at least {minimum}"
        high = "" if maximum == math.inf else f" and at most {maximum}"
        raise ValueError(f"{name} must be finite, {low}{high}; got {value}")


# ---------------------------------------------------------------------------- transform


@dataclass(frozen=True, eq=False)
class Transform:
    """A rigid motion: ``x -> rotation @ x + translation`` for a column vector ``x``.

    ``rotation`` is a proper rotation (determinant +1), shape (3, 3); ``translation`` has shape
    (3,). Both are float64 and read-only. Equality is identity; ``allclose`` compares values.

    Raises:
        ValueError: If ``rotation`` is not a 3 x 3 proper rotation (to 1e-6) or ``translation``
            is not a 3-vector.
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        rotation = np.array(self.rotation, dtype=np.float64)
        translation = np.array(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                f"a transform is a 3 x 3 rotation and a 3-vector, got shapes "
                f"{rotation.shape} and {translation.shape}"
            )
        if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
            raise ValueError("a transform must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
        ):
            raise ValueError("rotation is not a proper rotation (orthogonal, determinant +1)")
        rotation.flags.writeable = False
        translation.flags.writeable = False
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> Transform:
        return cls(np.eye(3), np.zeros(3))

    def apply(self, coords) -> np.ndarray:
        """The coordinates moved by the transform: a float64 array of the same shape.

        ``coords`` has shape (..., 3), one row per point; it is not modified.
        """
        return np.asarray(coords, dtype=np.float64) @ self.rotation.T + self.translation

    def inverse(self) -> Transform:
        """The transform that undoes this one."""
        rotation = self.rotation.T
        return Transform(rotation, -rotation @ self.translation)

    def allclose(self, other: Transform, *, atol: float = 1e-8) -> bool:
        """True when both rotation and translation agree within ``atol``."""
        return bool(
            np.allclose(self.rotation, other.rotation, atol=atol, rtol=0.0)
            and np.allclose(self.translation, other.translation, atol=atol, rtol=0.0)
        )

    def to_dict(self) -> dict:
        """``{"rotation": 3 x 3 nested list, "translation": 3-list}``."""
        return {"rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Transform:
        """Inverse of ``to_dict``."""
        unknown = sorted(set(data) - {"rotation", "translation"})
        if unknown:
            raise ValueError(f"unknown keys {unknown} in a transform dictionary")
        return cls(data["rotation"], data["translation"])

    def __reduce__(self):
        # rebuilt through the constructor so that the arrays are read-only again
        return (Transform, (self.rotation, self.translation))

    def __repr__(self) -> str:
        return (
            f"Transform(rotation={self.rotation.tolist()}, translation={self.translation.tolist()})"
        )


def kabsch_transform(mobile: np.ndarray, fixed: np.ndarray) -> Transform:
    """The proper rotation and translation that best map ``mobile`` onto ``fixed`` (Kabsch, 1976).

    Least squares over the rows (point i of ``mobile`` corresponds to point i of ``fixed``).
    The rotation comes from the SVD of the covariance matrix; a reflection is corrected by
    flipping the last singular direction. The inputs must not be degenerate (see
    ``is_degenerate``), or the rotation is not unique.

    Args:
        mobile: Coordinates, shape (n, 3).
        fixed: Coordinates, shape (n, 3).
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.float64)
    if mobile.shape != fixed.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            f"mobile and fixed must both have shape (n, 3), got {mobile.shape} and {fixed.shape}"
        )
    mobile_centre = mobile.mean(axis=0)
    fixed_centre = fixed.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (fixed - fixed_centre)
    left, _, right_t = np.linalg.svd(covariance)
    sign = 1.0 if np.linalg.det(right_t.T @ left.T) >= 0 else -1.0
    rotation = right_t.T @ np.diag([1.0, 1.0, sign]) @ left.T
    return Transform(rotation, fixed_centre - rotation @ mobile_centre)


def is_degenerate(coords: np.ndarray) -> bool:
    """True when the points do not fix a rotation: fewer than 3, coincident, or collinear.

    Collinear means a ratio of the second to the first singular value of the centred
    coordinates below 1e-6; planar sets are fine.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 3:
        return True
    singular = np.linalg.svd(coords - coords.mean(axis=0), compute_uv=False)
    return bool(singular[0] <= 0.0 or singular[1] <= _COLLINEAR_RATIO * singular[0])


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square deviation of two (n, 3) coordinate arrays, row against row."""
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


# ---------------------------------------------------------------------------- residue names

_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

#: Modified residues that stand for a standard amino acid: selenomethionine and three
#: phosphorylated residues. MSE in a crystal pairs with MET in a model.
_MODIFIED_PARENTS = {"MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR"}


@functools.lru_cache(maxsize=1)
def _parent_names() -> Mapping[str, str]:
    """Residue name to the amino acid it stands for: modified, protonation variant, D form."""
    from binding_metrics.core.nonstandard import D_AA_MAP
    from binding_metrics.core.residues import VARIANT_TO_PARENT_RESIDUE

    return {**_MODIFIED_PARENTS, **VARIANT_TO_PARENT_RESIDUE, **D_AA_MAP}


def canonical_residue_name(name: str) -> str:
    """The amino acid a residue name stands for (HIE to HIS, DAL to ALA, MSE to MET), else the
    upper-case name itself."""
    upper = str(name).strip().upper()
    return _parent_names().get(upper, upper)


def residue_names_compatible(name_a: str, name_b: str) -> bool:
    """True when two residue names are equal after ``canonical_residue_name``."""
    return canonical_residue_name(name_a) == canonical_residue_name(name_b)


def one_letter_code(name: str) -> str:
    """One-letter code of the canonical amino acid, ``"X"`` for anything else."""
    return _THREE_TO_ONE.get(canonical_residue_name(name), "X")


# ---------------------------------------------------------------------------- residue tables


@dataclass(frozen=True, eq=False)
class _ReceptorTable:
    """The polymer residues of one chain with a finite CA, with what the frame stage reads."""

    chain: str
    keys: tuple
    names: tuple
    atom_indices: tuple
    ca: np.ndarray
    backbone: tuple
    n_nonfinite: int
    duplicate_keys: frozenset

    def __len__(self) -> int:
        return len(self.keys)


def _single_model(atoms, what: str):
    if getattr(atoms.coord, "ndim", 2) != 2:
        raise ValueError(f"{what} must be one model: take one model of the stack with stack[i]")
    return atoms


def _build_table(atoms, chain: str) -> _ReceptorTable:
    coords = np.asarray(atoms.coord, dtype=np.float64)
    atom_names = np.asarray(atoms.atom_name)
    keys, names, indices, ca, backbone = [], [], [], [], []
    n_nonfinite = 0
    for residue in polymer_residues(atoms, chain):
        position = coords[residue.ca_index]
        if not np.isfinite(position).all():
            n_nonfinite += 1
            continue
        atoms_by_name: dict[str, np.ndarray] = {}
        for index in residue.atom_indices:
            name = str(atom_names[index])
            if name in _BACKBONE_ATOM_NAMES and name not in atoms_by_name:
                if np.isfinite(coords[index]).all():
                    atoms_by_name[name] = coords[index]
        atoms_by_name["CA"] = position
        keys.append(residue.key)
        names.append(residue.name)
        indices.append(residue.atom_indices)
        ca.append(position)
        backbone.append(atoms_by_name)
    counts: dict[ResidueKey, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return _ReceptorTable(
        chain=chain,
        keys=tuple(keys),
        names=tuple(names),
        atom_indices=tuple(indices),
        ca=np.array(ca, dtype=np.float64).reshape(-1, 3),
        backbone=tuple(backbone),
        n_nonfinite=n_nonfinite,
        duplicate_keys=frozenset(key for key, count in counts.items() if count > 1),
    )


# ---------------------------------------------------------------------------- pairing


@dataclass(frozen=True, eq=False)
class ResiduePairing:
    """How the receptor residues of a pose were paired with those of the reference.

    Attributes:
        method: ``"map"``, ``"number"``, ``"sequence"`` or ``"position"``; None when no method
            produced a pairing.
        pose_keys: Keys of the paired pose residues, in the order of the pose chain.
        reference_keys: The reference residue of each pair.
        n_receptor_residues_pose, n_receptor_residues_reference: Polymer residues of the two
            receptor chains.
        sequence_identity: Fraction of the aligned (sequence, position) or same-numbered
            (number) residues with compatible names, or the fraction of the pairs for a map;
            None when nothing was compared.
        alignment_ambiguous: True when more than one sequence alignment had the best score.
        reason: Why the pairing was refused, or None when it can be used.
        notes: Why earlier methods of the ``auto`` cascade did not apply.
    """

    method: Optional[str]
    pose_keys: tuple
    reference_keys: tuple
    n_receptor_residues_pose: int
    n_receptor_residues_reference: int
    sequence_identity: Optional[float]
    alignment_ambiguous: bool
    reason: Optional[str]
    notes: tuple
    pose_index: np.ndarray
    reference_index: np.ndarray

    @property
    def n_paired(self) -> int:
        return len(self.pose_keys)

    @property
    def paired_fraction(self) -> Optional[float]:
        """Paired residues over the residues of the reference receptor; None without any."""
        if self.n_receptor_residues_reference == 0:
            return None
        return self.n_paired / self.n_receptor_residues_reference

    @property
    def ok(self) -> bool:
        return self.reason is None and self.method is not None


def _make_pairing(
    method: Optional[str],
    pairs: Sequence[tuple[int, int]],
    pose: _ReceptorTable,
    reference: _ReceptorTable,
    *,
    identity: Optional[float] = None,
    ambiguous: bool = False,
    reason: Optional[str] = None,
    notes: Sequence[str] = (),
) -> ResiduePairing:
    pose_index = np.array([p for p, _ in pairs], dtype=np.int64)
    reference_index = np.array([r for _, r in pairs], dtype=np.int64)
    return ResiduePairing(
        method=method,
        pose_keys=tuple(pose.keys[i] for i in pose_index),
        reference_keys=tuple(reference.keys[j] for j in reference_index),
        n_receptor_residues_pose=len(pose),
        n_receptor_residues_reference=len(reference),
        sequence_identity=identity,
        alignment_ambiguous=ambiguous,
        reason=reason,
        notes=tuple(notes),
        pose_index=pose_index,
        reference_index=reference_index,
    )


def _pair_by_map(pose, reference, residue_map, notes):
    pose_lookup = {key: i for i, key in reversed(list(enumerate(pose.keys)))}
    reference_lookup = {key: j for j, key in reversed(list(enumerate(reference.keys)))}
    pairs, applicable = [], 0
    for pose_key, reference_key in residue_map.items():
        if pose_key.chain != pose.chain or reference_key.chain != reference.chain:
            continue
        applicable += 1
        for key, table, side in (
            (pose_key, pose, "pose"),
            (reference_key, reference, "reference"),
        ):
            if key in table.duplicate_keys:
                return _make_pairing(
                    "map",
                    (),
                    pose,
                    reference,
                    reason=(
                        f"residue_map names {key.label}, which occurs more than once in the "
                        f"{side} receptor chain: the residue is ambiguous"
                    ),
                    notes=notes,
                )
        i, j = pose_lookup.get(pose_key), reference_lookup.get(reference_key)
        if i is not None and j is not None:
            pairs.append((i, j))
    if not pairs:
        return _make_pairing(
            "map",
            (),
            pose,
            reference,
            reason=(
                f"no entry of the residue_map pairs a residue of the pose chain '{pose.chain}' "
                f"with one of the reference chain '{reference.chain}' ({len(residue_map)} "
                f"entries, {applicable} on these chains, none with both residues present)"
            ),
            notes=notes,
        )
    pairs.sort()
    compatible = sum(residue_names_compatible(pose.names[i], reference.names[j]) for i, j in pairs)
    return _make_pairing(
        "map", pairs, pose, reference, identity=compatible / len(pairs), notes=notes
    )


def _pair_by_number(pose, reference, options, notes):
    """The pairing by number and name, or None and the reason it does not apply."""
    if pose.duplicate_keys or reference.duplicate_keys:
        which = "pose" if pose.duplicate_keys else "reference"
        repeated = sorted((pose.duplicate_keys or reference.duplicate_keys))[0].label
        return None, f"pairing by number: residue {repeated} occurs twice in the {which} chain"
    lookup = {(key.number, key.icode): j for j, key in enumerate(reference.keys)}
    same_number = [
        (i, lookup[(key.number, key.icode)])
        for i, key in enumerate(pose.keys)
        if (key.number, key.icode) in lookup
    ]
    pairs = [
        (i, j)
        for i, j in same_number
        if residue_names_compatible(pose.names[i], reference.names[j])
    ]
    shorter = min(len(pose), len(reference))
    fraction = len(pairs) / shorter if shorter else 0.0
    if not pairs or fraction < options.min_pair_fraction - 1e-12:
        return None, (
            f"pairing by number: {len(pairs)} of the {shorter} residues of the shorter chain "
            f"share number, insertion code and residue name ({fraction:.2f}; "
            f"min_pair_fraction is {options.min_pair_fraction:.2f})"
        )
    identity = len(pairs) / len(same_number)
    return _make_pairing("number", pairs, pose, reference, identity=identity, notes=notes), None


def _align_columns(pose_sequence: str, reference_sequence: str, options: FrameOptions):
    """Aligned (pose index, reference index) columns of the best global alignment, and whether
    another alignment scored the same."""
    try:
        import biotite.sequence as bio_sequence
        import biotite.sequence.align as bio_align
    except ImportError as error:
        raise ImportError(
            "biotite is required for pairing receptor residues by sequence alignment. "
            "Install with: pip install binding-metrics[biotite]"
        ) from error
    alphabet = bio_sequence.ProteinSequence.alphabet
    try:
        matrix = bio_align.SubstitutionMatrix(alphabet, alphabet, options.alignment_matrix)
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise ValueError(
            f"alignment_matrix {options.alignment_matrix!r} is not a substitution matrix that "
            "biotite knows (for example 'BLOSUM62')"
        ) from error
    alignments = bio_align.align_optimal(
        bio_sequence.ProteinSequence(pose_sequence),
        bio_sequence.ProteinSequence(reference_sequence),
        matrix,
        gap_penalty=options.alignment_gap_penalty,
        terminal_penalty=False,
        local=False,
        max_number=2,
    )
    trace = np.asarray(alignments[0].trace)
    columns = trace[(trace[:, 0] >= 0) & (trace[:, 1] >= 0)]
    return columns, len(alignments) > 1


def _pair_by_sequence(pose, reference, options, notes):
    columns, ambiguous = _align_columns(
        "".join(one_letter_code(name) for name in pose.names),
        "".join(one_letter_code(name) for name in reference.names),
        options,
    )
    pairs = [
        (int(i), int(j))
        for i, j in columns
        if residue_names_compatible(pose.names[i], reference.names[j])
    ]
    identity = len(pairs) / len(columns) if len(columns) else 0.0
    return _make_pairing(
        "sequence", pairs, pose, reference, identity=identity, ambiguous=ambiguous, notes=notes
    )


def _pair_by_position(pose, reference, notes):
    if len(pose) != len(reference):
        return _make_pairing(
            "position",
            (),
            pose,
            reference,
            reason=(
                f"pairing='position' needs receptor chains of equal length: the pose has "
                f"{len(pose)} residues, the reference {len(reference)}"
            ),
            notes=notes,
        )
    pairs = [(i, i) for i in range(len(pose))]
    compatible = sum(
        residue_names_compatible(pose.names[i], reference.names[i]) for i in range(len(pose))
    )
    return _make_pairing(
        "position", pairs, pose, reference, identity=compatible / len(pose), notes=notes
    )


def _pair_tables(pose, reference, residue_map, options) -> ResiduePairing:
    """The cascade of the module docstring, then the refusals that apply to every method."""
    notes: list[str] = []
    if len(pose) == 0 or len(reference) == 0:
        side = "pose" if len(pose) == 0 else "reference"
        return _make_pairing(
            None, (), pose, reference, reason=f"the {side} receptor chain has no polymer residues"
        )
    method = options.pairing
    pairing: Optional[ResiduePairing] = None
    if method == "map" and not residue_map:
        pairing = _make_pairing(
            "map",
            (),
            pose,
            reference,
            reason="pairing='map' needs a residue_map and none was given",
        )
    elif residue_map and method in ("auto", "map"):
        pairing = _pair_by_map(pose, reference, residue_map, notes)
    if pairing is None and method in ("auto", "number"):
        pairing, note = _pair_by_number(pose, reference, options, notes)
        if pairing is None:
            if method == "number":
                pairing = _make_pairing("number", (), pose, reference, reason=note)
            else:
                notes.append(note)
    if pairing is None and method in ("auto", "sequence"):
        pairing = _pair_by_sequence(pose, reference, options, notes)
    if pairing is None:
        pairing = _pair_by_position(pose, reference, notes)
    return _refuse_if_poor(pairing, options)


def _refuse_if_poor(pairing: ResiduePairing, options: FrameOptions) -> ResiduePairing:
    """Set the reason when a pairing that exists is too poor to place a pose on."""
    if pairing.reason is not None:
        return pairing
    reason = None
    identity = pairing.sequence_identity
    if pairing.method != "map" and (identity is None or identity < options.min_sequence_identity):
        shown = "none" if identity is None else f"{identity:.2f}"
        reason = (
            f"pairing by {pairing.method}: residue name identity {shown} is below "
            f"min_sequence_identity {options.min_sequence_identity:.2f}; the pose receptor does "
            "not look like the reference receptor"
        )
    elif pairing.n_paired == 0:
        reason = f"pairing by {pairing.method} paired no residue"
    elif pairing.paired_fraction < options.min_frame_fraction - 1e-12:
        reason = (
            f"only {pairing.n_paired} of the {pairing.n_receptor_residues_reference} reference "
            f"receptor residues are paired ({pairing.paired_fraction:.2f}); min_frame_fraction "
            f"is {options.min_frame_fraction:.2f}"
        )
    if reason is None:
        return pairing
    return ResiduePairing(**{**_shallow(pairing), "reason": reason})


def _shallow(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def _normalise_map(residue_map: Optional[Mapping]) -> dict:
    if not residue_map:
        return {}
    return {ResidueKey.coerce(k): ResidueKey.coerce(v) for k, v in residue_map.items()}


# ---------------------------------------------------------------------------- reference


@dataclass(frozen=True, eq=False)
class PreparedReference:
    """A reference read once: its receptor residue table, its binding site and its binder table.

    Build with ``prepare_reference``; ``superpose_receptor`` accepts it in place of the
    ``Reference`` so that the reference structure is read once for a whole set of poses.
    ``binder_table`` is None when the reference has no binder chain.
    """

    reference: Reference
    table: _ReceptorTable
    site: Optional[tuple]
    site_cutoff_angstrom: float
    binder_table: Optional[_ReceptorTable] = None


def prepare_reference(
    reference: Reference, *, site_cutoff_angstrom: float = FrameOptions.site_cutoff_angstrom
) -> PreparedReference:
    """Validate a reference against its structure and read what the frame stage needs.

    The result is cached on the ``Reference`` (per cutoff) with the content hash of its
    structure, so repeated calls cost one ``os.stat`` and a rewritten structure file is read
    again.

    Args:
        reference: The reference.
        site_cutoff_angstrom: Heavy-atom distance to the reference binder chain that puts a
            receptor residue in the binding site (default 4.5, ``FrameOptions``).

    Raises:
        ValueError: If ``Reference.validate`` finds a problem (listed in the message), or the
            structure holds more than one model.
    """
    cache = reference.__dict__.setdefault("_prepared_cache", {})
    cutoff = float(site_cutoff_angstrom)
    # The prepared data depend on the structure only (the other fields of the reference are
    # frozen), so the entry is valid while the structure hash matches.
    structure_hash = reference.structure.content_hash()
    entry = cache.get(cutoff)
    if entry is not None and entry[0] == structure_hash:
        return entry[1]
    reference.validate()
    atoms = _single_model(reference.atoms(), "the reference structure")
    table = _build_table(atoms, reference.receptor_chain)
    site = binder_table = None
    if reference.binder_chain is not None:
        site = _site_keys(atoms, table, reference.binder_chain, cutoff)
        binder_table = _build_table(atoms, reference.binder_chain)
    prepared = PreparedReference(reference, table, site, cutoff, binder_table)
    cache[cutoff] = (structure_hash, prepared)
    return prepared


def _heavy(atoms, indices: np.ndarray) -> np.ndarray:
    elements = (
        np.asarray(atoms.element)[indices]
        if "element" in atoms.get_annotation_categories()
        else None
    )
    if elements is None:
        return np.ones(len(indices), dtype=bool)
    return ~np.isin(elements, ("H", "D"))


def _site_keys(atoms, table: _ReceptorTable, binder_chain: str, cutoff: float) -> tuple:
    """Receptor residues with a heavy atom within ``cutoff`` of a heavy atom of the binder."""
    coords = np.asarray(atoms.coord, dtype=np.float64)
    binder_indices = np.flatnonzero(np.asarray(atoms.chain_id) == binder_chain)
    binder = coords[binder_indices[_heavy(atoms, binder_indices)]]
    binder = binder[np.isfinite(binder).all(axis=1)]
    if len(binder) == 0:
        return ()
    limit = cutoff * cutoff
    site = []
    for key, indices in zip(table.keys, table.atom_indices):
        residue = coords[indices[_heavy(atoms, indices)]]
        residue = residue[np.isfinite(residue).all(axis=1)]
        if len(residue) == 0:
            continue
        squared = ((residue[:, None, :] - binder[None, :, :]) ** 2).sum(axis=2)
        if (squared <= limit).any():
            site.append(key)
    return tuple(site)


# ---------------------------------------------------------------------------- result


@dataclass(frozen=True, eq=False)
class FrameResult:
    """The outcome of placing one pose on one reference.

    A failed result has a ``reason``, no transform and no RMSD, and keeps the pairing counts
    that were found. Region RMSDs are C-alpha RMSDs of the paired residues of a region in the
    final frame, with no refit.

    Attributes:
        status, reason: ``"ok"`` or ``"failed"``, and why it failed (None when ok).
        pairing: The method that paired the residues (``"map"``, ``"number"``, ``"sequence"``,
            ``"position"``), or None.
        n_receptor_residues_pose, n_receptor_residues_reference: Polymer residues of the two
            receptor chains.
        n_paired, paired_fraction: Paired residues, and their fraction of the reference
            receptor residues.
        sequence_identity: See ``ResiduePairing``.
        alignment_ambiguous: True when the sequence alignment had a tie.
        n_fit_residues, n_fit_atoms: Residues and atoms of the fit set, after rejection.
        n_rejected: Residues rejected by ``outliers="iqr"``.
        receptor_rmsd_angstrom: RMSD over the fit atoms after the fit (after rejection).
        region_rmsd_angstrom: ``{"all": ..., "site": ..., "<set>": ...}``, a float or None.
        region_n_residues, region_n_paired: Size of each region in the reference, and how many
            of its residues are paired.
        region_reasons: Why a region RMSD is None.
        transform: Maps pose coordinates into the reference frame, or None when failed.
        pairs: ``(pose ResidueKey, reference ResidueKey)`` of every paired residue.
        rejected_pairs: The pairs that ``outliers="iqr"`` rejected.
        warnings: Things that did not stop the fit but that a reader should know.
    """

    status: str
    reason: Optional[str]
    pairing: Optional[str]
    n_receptor_residues_pose: int
    n_receptor_residues_reference: int
    n_paired: int
    paired_fraction: Optional[float]
    sequence_identity: Optional[float]
    alignment_ambiguous: bool
    n_fit_residues: int
    n_fit_atoms: int
    n_rejected: int
    receptor_rmsd_angstrom: Optional[float]
    region_rmsd_angstrom: Mapping[str, Optional[float]]
    region_n_residues: Mapping[str, int]
    region_n_paired: Mapping[str, int]
    region_reasons: Mapping[str, str]
    transform: Optional[Transform]
    pairs: tuple
    rejected_pairs: tuple
    warnings: tuple

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def __getstate__(self) -> dict:
        """The state for pickle and deepcopy: the read-only mapping views as plain dicts."""
        state = dict(self.__dict__)
        for name in _RESULT_MAPPING_FIELDS:
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state: dict) -> None:
        state = dict(state)
        for name in _RESULT_MAPPING_FIELDS:
            state[name] = MappingProxyType(state[name])
        self.__dict__.update(state)

    def to_dict(self, *, include_transform: bool = True, include_pairs: bool = False) -> dict:
        """The ``frame`` block of a row, JSON-ready: every attribute except ``rejected_pairs``.

        Args:
            include_transform: Include the rotation and translation (None when failed).
            include_pairs: Include the paired residues as ``[[pose key], [reference key]]``
                lists (long; off by default).
        """
        data: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "pairing": self.pairing,
            "n_receptor_residues_pose": self.n_receptor_residues_pose,
            "n_receptor_residues_reference": self.n_receptor_residues_reference,
            "n_paired": self.n_paired,
            "paired_fraction": self.paired_fraction,
            "sequence_identity": self.sequence_identity,
            "alignment_ambiguous": self.alignment_ambiguous,
            "n_fit_residues": self.n_fit_residues,
            "n_fit_atoms": self.n_fit_atoms,
            "n_rejected": self.n_rejected,
            "receptor_rmsd_angstrom": self.receptor_rmsd_angstrom,
            "region_rmsd_angstrom": dict(self.region_rmsd_angstrom),
            "region_n_residues": dict(self.region_n_residues),
            "region_n_paired": dict(self.region_n_paired),
            "region_reasons": dict(self.region_reasons),
            "warnings": list(self.warnings),
        }
        if include_transform:
            data["transform"] = None if self.transform is None else self.transform.to_dict()
        if include_pairs:
            data["pairs"] = [[p.to_list(), r.to_list()] for p, r in self.pairs]
        return data


#: The fields of ``FrameResult`` that hold read-only mapping views.
_RESULT_MAPPING_FIELDS = (
    "region_rmsd_angstrom",
    "region_n_residues",
    "region_n_paired",
    "region_reasons",
)


def _failed(
    pairing: Optional[ResiduePairing], reason: str, *, pose_n: int = 0, ref_n: int = 0
) -> FrameResult:
    if pairing is not None:
        pose_n, ref_n = pairing.n_receptor_residues_pose, pairing.n_receptor_residues_reference
    return FrameResult(
        status="failed",
        reason=reason,
        pairing=None if pairing is None else pairing.method,
        n_receptor_residues_pose=pose_n,
        n_receptor_residues_reference=ref_n,
        n_paired=0 if pairing is None else pairing.n_paired,
        paired_fraction=None if pairing is None else pairing.paired_fraction,
        sequence_identity=None if pairing is None else pairing.sequence_identity,
        alignment_ambiguous=False if pairing is None else pairing.alignment_ambiguous,
        n_fit_residues=0,
        n_fit_atoms=0,
        n_rejected=0,
        receptor_rmsd_angstrom=None,
        region_rmsd_angstrom=MappingProxyType({}),
        region_n_residues=MappingProxyType({}),
        region_n_paired=MappingProxyType({}),
        region_reasons=MappingProxyType({}),
        transform=None,
        pairs=() if pairing is None else tuple(zip(pairing.pose_keys, pairing.reference_keys)),
        rejected_pairs=(),
        warnings=() if pairing is None else _pairing_warnings(pairing),
    )


def _pairing_warnings(pairing: ResiduePairing) -> tuple:
    found = []
    if pairing.alignment_ambiguous:
        found.append(
            "the receptor sequence alignment has more than one optimal solution; the first one "
            "was used"
        )
    return tuple(found)


# ---------------------------------------------------------------------------- the fit


def _fit_atoms(pose, reference, pose_index, reference_index, frame_atoms):
    """Matching coordinates of the fit atoms and the pair (row of ``pose_index``) of each."""
    if frame_atoms == "ca":
        return pose.ca[pose_index], reference.ca[reference_index], np.arange(len(pose_index))
    mobile, fixed, owner = [], [], []
    for row, (i, j) in enumerate(zip(pose_index, reference_index)):
        for name in _BACKBONE_ATOM_NAMES:
            if name in pose.backbone[i] and name in reference.backbone[j]:
                mobile.append(pose.backbone[i][name])
                fixed.append(reference.backbone[j][name])
                owner.append(row)
    return (
        np.array(mobile, dtype=np.float64).reshape(-1, 3),
        np.array(fixed, dtype=np.float64).reshape(-1, 3),
        np.array(owner, dtype=np.int64),
    )


def _fit(mobile, fixed, owner, n_pairs, options):
    """Kabsch fit, with iterative interquartile rejection of residues when asked.

    Returns ``(transform, inlier pair mask, warnings)``. A pair whose atoms are all gone from
    the fit set (backbone atoms missing) is not an inlier.
    """
    in_fit = np.zeros(n_pairs, dtype=bool)
    in_fit[owner] = True
    inliers = in_fit.copy()
    found_warnings: list[str] = []
    transform = kabsch_transform(mobile[inliers[owner]], fixed[inliers[owner]])
    if options.outliers is None:
        return transform, inliers, found_warnings
    n_per_pair = np.bincount(owner, minlength=n_pairs).astype(np.float64)
    guard = max(options.min_frame_residues, 3)
    fitted_current = True
    for _ in range(options.iqr_max_iterations):
        atom_mask = inliers[owner]
        transform = kabsch_transform(mobile[atom_mask], fixed[atom_mask])
        fitted_current = True
        squared = np.sum((transform.apply(mobile) - fixed) ** 2, axis=1)
        per_pair = np.bincount(owner, weights=squared, minlength=n_pairs) / np.maximum(
            n_per_pair, 1.0
        )
        lower, upper = np.quantile(per_pair[inliers], (0.25, 0.75))
        limit = upper + options.iqr_threshold * (upper - lower)
        outlier = inliers & (per_pair > limit) & (per_pair > options.iqr_min_deviation_angstrom**2)
        if not outlier.any():
            break
        remaining = inliers & ~outlier
        if remaining.sum() < guard:
            found_warnings.append(
                f"outlier rejection stopped: rejecting {int(outlier.sum())} more residues would "
                f"leave {int(remaining.sum())}, fewer than {guard}"
            )
            break
        keep_mask = remaining[owner]
        if is_degenerate(mobile[keep_mask]) or is_degenerate(fixed[keep_mask]):
            found_warnings.append("outlier rejection stopped: the remaining atoms are collinear")
            break
        inliers = remaining
        fitted_current = False
    if not fitted_current:
        atom_mask = inliers[owner]
        transform = kabsch_transform(mobile[atom_mask], fixed[atom_mask])
    return transform, inliers, found_warnings


def _resolve_fit_set(prepared: PreparedReference, options: FrameOptions) -> Optional[frozenset]:
    """The reference residues that the fit is restricted to, or None for all paired residues."""
    name = options.frame_residues
    if name is None:
        return None
    reference = prepared.reference
    if name == "site":
        if prepared.site is None:
            raise ValueError(
                f"frame_residues='site' needs a binder_chain on reference '{reference.name}'"
            )
        return frozenset(prepared.site)
    members = reference.residue_sets.get(name)
    if members is None:
        raise ValueError(
            f"frame_residues='{name}' is not a residue set of reference '{reference.name}' "
            f"(it has {list(reference.residue_sets)}) and not 'site'"
        )
    return members


def pair_residues(
    pose,
    reference,
    *,
    receptor_chain: Optional[str] = None,
    residue_map: Optional[Mapping] = None,
    options: Any = None,
) -> ResiduePairing:
    """Pair the receptor residues of a pose with those of a reference (no fit).

    See the module docstring for the methods. The refusals that depend only on the pairing
    (chain without residues, sequence identity, paired fraction) set ``reason``.

    Args:
        pose: A ``Pose`` or a biotite ``AtomArray`` of one model.
        reference: A ``Reference`` or a ``PreparedReference``.
        receptor_chain: The receptor chain of the pose (user chain ID); default is the receptor
            chain of the reference.
        residue_map: ``{pose ResidueKey: reference ResidueKey}`` replacing the reference's own
            map; an empty mapping means no map. None uses ``Reference.residue_map``.
        options: ``FrameOptions``, a mapping of options, or None for the defaults.

    Raises:
        ValueError: For an invalid option or reference, or when the pose has more than one model.
    """
    options = _as_options(options)
    prepared = _prepare(reference, options)
    return _pair_pose(pose, prepared, receptor_chain, residue_map, options)[1]


def _pair_pose(pose, prepared: PreparedReference, receptor_chain, residue_map, options):
    """The receptor table of the pose and its pairing with the prepared reference."""
    atoms = _single_model(pose.atoms() if isinstance(pose, Pose) else pose, "the pose")
    chain = receptor_chain or prepared.reference.receptor_chain
    table = _build_table(atoms, chain)
    present = sorted({str(c) for c in np.unique(np.asarray(atoms.chain_id))})
    if chain not in present:
        return table, _make_pairing(
            None,
            (),
            table,
            prepared.table,
            reason=f"receptor chain '{chain}' is not in the pose: it has chains {present}",
        )
    mapping = prepared.reference.residue_map if residue_map is None else _normalise_map(residue_map)
    return table, _pair_tables(table, prepared.table, mapping, options)


def _empty_table(chain: str) -> _ReceptorTable:
    return _ReceptorTable(
        chain=chain,
        keys=(),
        names=(),
        atom_indices=(),
        ca=np.zeros((0, 3)),
        backbone=(),
        n_nonfinite=0,
        duplicate_keys=frozenset(),
    )


def pair_binder_residues(
    pose,
    reference,
    *,
    binder_chain: str,
    options: Any = None,
) -> ResiduePairing:
    """Pair the binder residues of a pose with those of the binder chain of the reference.

    The pairing says which numbers ``renumber_to_reference`` gives the binder. The two chains
    must have the same number of polymer residues and every residue must pair, by number and
    name where the numbers agree and otherwise by the sequence alignment of the module
    docstring (``FrameOptions.alignment_*``; the identity and fraction limits of the receptor
    do not apply, a complete pairing has an identity of 1). A ``residue_map`` does not apply:
    its reference side is the receptor. ``reason`` says why the binder cannot be renumbered.

    Args:
        pose: A ``Pose`` or a biotite ``AtomArray`` of one model.
        reference: A ``Reference`` or a ``PreparedReference``; it needs a ``binder_chain``.
        binder_chain: The binder chain of the pose (user chain ID).
        options: ``FrameOptions``, a mapping of options, or None for the defaults.

    Raises:
        ValueError: For an invalid option or reference, or when the pose has more than one model.
    """
    options = _as_options(options)
    prepared = _prepare(reference, options)
    atoms = _single_model(pose.atoms() if isinstance(pose, Pose) else pose, "the pose")
    table = _build_table(atoms, binder_chain)
    reference_table = prepared.binder_table
    if reference_table is None:
        return _make_pairing(
            None, (), table, _empty_table(""), reason="the reference has no binder chain"
        )
    own, theirs = binder_chain, reference_table.chain
    if len(table) == 0:
        reason = f"the pose has no polymer residue in the binder chain '{own}'"
    elif len(table) != len(reference_table):
        reason = (
            f"the pose binder chain '{own}' has {len(table)} polymer residues and the reference "
            f"binder chain '{theirs}' has {len(reference_table)}"
        )
    elif table.duplicate_keys:
        reason = (
            f"residue {sorted(table.duplicate_keys)[0].label} occurs twice in the pose binder "
            f"chain '{own}'"
        )
    else:
        # the receptor limits (identity, fraction of the reference) do not apply: below, every
        # residue has to pair, and a pairing by names of a different peptide fails that
        auto = replace(options, pairing="auto", min_sequence_identity=0.0, min_frame_fraction=0.0)
        pairing = _pair_tables(table, reference_table, None, auto)
        if pairing.reason is None and pairing.n_paired != len(table):
            reason = (
                f"only {pairing.n_paired} of the {len(table)} residues of the pose binder chain "
                f"'{own}' pair with the reference binder chain '{theirs}' "
                f"(pairing by {pairing.method})"
            )
            return ResiduePairing(**{**_shallow(pairing), "reason": reason})
        return pairing
    return _make_pairing(None, (), table, reference_table, reason=reason)


def _prepare(reference, options: FrameOptions) -> PreparedReference:
    if isinstance(reference, PreparedReference):
        if reference.site_cutoff_angstrom == float(options.site_cutoff_angstrom):
            return reference
        reference = reference.reference
    if not isinstance(reference, Reference):
        raise TypeError(f"reference must be a Reference, got {type(reference).__name__}")
    return prepare_reference(reference, site_cutoff_angstrom=options.site_cutoff_angstrom)


def superpose_receptor(
    pose,
    reference,
    *,
    receptor_chain: Optional[str] = None,
    residue_map: Optional[Mapping] = None,
    options: Any = None,
) -> FrameResult:
    """Place a pose in the frame of a reference receptor and report the quality of the frame.

    Pairs the receptor residues (``pair_residues``), fits the pose onto the reference by
    Kabsch on the paired atoms, and measures the receptor RMSD of the fit and the C-alpha RMSD
    of each region (see the module docstring). A pose that cannot be placed gives a failed
    result with a reason; an invalid argument raises.

    Args:
        pose: A ``Pose`` (its ``atoms()`` are used) or a biotite ``AtomArray`` of one model.
        reference: A ``Reference`` or a ``PreparedReference``.
        receptor_chain: The receptor chain of the pose (user chain ID); default is the receptor
            chain of the reference.
        residue_map: Per-pose ``{pose ResidueKey: reference ResidueKey}`` that replaces
            ``Reference.residue_map``; None keeps the reference's map, an empty mapping removes
            it.
        options: ``FrameOptions``, a mapping of options, or None for the defaults.

    Returns:
        A ``FrameResult``. The pose is not modified; ``place_in_reference_frame`` applies the
        transform.

    Raises:
        ValueError: For an invalid option, an invalid reference (``Reference.validate``), a
            ``frame_residues`` that the reference does not have, or a pose of more than one model.
        TypeError: For an argument of the wrong type.
    """
    options = _as_options(options)
    prepared = _prepare(reference, options)
    fit_set = _resolve_fit_set(prepared, options)
    pose_table, pairing = _pair_pose(pose, prepared, receptor_chain, residue_map, options)
    if not pairing.ok:
        return _failed(pairing, pairing.reason or "no pairing", ref_n=len(prepared.table))
    reference_table = prepared.table

    pair_selected = np.ones(pairing.n_paired, dtype=bool)
    if fit_set is not None:
        pair_selected = np.array([key in fit_set for key in pairing.reference_keys], dtype=bool)
    n_selected = int(pair_selected.sum())
    if n_selected < options.min_frame_residues:
        scope = (
            "paired receptor residues"
            if fit_set is None
            else f"paired residues of frame_residues='{options.frame_residues}'"
        )
        return _failed(
            pairing,
            f"only {n_selected} {scope} to fit; min_frame_residues is {options.min_frame_residues}",
        )

    fit_pose_index = pairing.pose_index[pair_selected]
    fit_reference_index = pairing.reference_index[pair_selected]
    mobile, fixed, owner = _fit_atoms(
        pose_table, reference_table, fit_pose_index, fit_reference_index, options.frame_atoms
    )
    n_with_atoms = len(np.unique(owner))
    if n_with_atoms < options.min_frame_residues:
        return _failed(
            pairing,
            f"only {n_with_atoms} paired residues have {', '.join(_BACKBONE_ATOM_NAMES)} atoms "
            f"in both structures; min_frame_residues is {options.min_frame_residues}",
        )
    if is_degenerate(mobile) or is_degenerate(fixed):
        where = "pose" if is_degenerate(mobile) else "reference"
        return _failed(
            pairing,
            f"the {len(mobile)} fit atoms of the {where} are collinear or coincident: the "
            "rotation is not determined",
        )

    transform, inlier_pairs, fit_warnings = _fit(mobile, fixed, owner, len(fit_pose_index), options)
    inlier_atoms = inlier_pairs[owner]
    fit_rmsd = rmsd(transform.apply(mobile[inlier_atoms]), fixed[inlier_atoms])
    has_atoms = np.zeros(len(fit_pose_index), dtype=bool)
    has_atoms[owner] = True
    rejected_rows = np.flatnonzero(pair_selected)[has_atoms & ~inlier_pairs]

    pair_list = tuple(zip(pairing.pose_keys, pairing.reference_keys))
    moved_ca = transform.apply(pose_table.ca[pairing.pose_index])
    deviation = np.sqrt(
        np.sum((moved_ca - reference_table.ca[pairing.reference_index]) ** 2, axis=1)
    )
    regions = _regions(prepared, pairing, deviation)

    found_warnings = list(_pairing_warnings(pairing)) + fit_warnings
    if pose_table.n_nonfinite:
        found_warnings.append(
            f"{pose_table.n_nonfinite} pose receptor residues with non-finite CA coordinates "
            "were left out"
        )
    logger.debug(
        "frame: %s pairing, %d paired of %d, receptor rmsd %.3f A, %d rejected",
        pairing.method,
        pairing.n_paired,
        len(reference_table),
        fit_rmsd,
        len(rejected_rows),
    )
    return FrameResult(
        status="ok",
        reason=None,
        pairing=pairing.method,
        n_receptor_residues_pose=pairing.n_receptor_residues_pose,
        n_receptor_residues_reference=pairing.n_receptor_residues_reference,
        n_paired=pairing.n_paired,
        paired_fraction=pairing.paired_fraction,
        sequence_identity=pairing.sequence_identity,
        alignment_ambiguous=pairing.alignment_ambiguous,
        n_fit_residues=int(inlier_pairs.sum()),
        n_fit_atoms=int(inlier_atoms.sum()),
        n_rejected=len(rejected_rows),
        receptor_rmsd_angstrom=fit_rmsd,
        region_rmsd_angstrom=MappingProxyType(regions[0]),
        region_n_residues=MappingProxyType(regions[1]),
        region_n_paired=MappingProxyType(regions[2]),
        region_reasons=MappingProxyType(regions[3]),
        transform=transform,
        pairs=pair_list,
        rejected_pairs=tuple(pair_list[row] for row in rejected_rows),
        warnings=tuple(found_warnings),
    )


def _regions(prepared: PreparedReference, pairing: ResiduePairing, deviation: np.ndarray):
    """C-alpha RMSD per region in the final frame: ``all``, ``site`` and the named sets."""
    reference = prepared.reference
    definitions: dict[str, Optional[frozenset]] = {"all": None}
    sizes = {"all": len(prepared.table)}
    if prepared.site is not None:
        definitions["site"] = frozenset(prepared.site)
        sizes["site"] = len(prepared.site)
    for name, members in reference.residue_sets.items():
        definitions[name] = members
        sizes[name] = len(members)
    values: dict[str, Optional[float]] = {}
    n_paired: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for name, members in definitions.items():
        if members is None:
            mask = np.ones(len(deviation), dtype=bool)
        else:
            mask = np.array([key in members for key in pairing.reference_keys], dtype=bool)
        n_paired[name] = int(mask.sum())
        if not mask.any():
            values[name] = None
            reasons[name] = f"none of the {sizes[name]} residues of region '{name}' is paired"
        else:
            values[name] = float(np.sqrt(np.mean(deviation[mask] ** 2)))
    return values, sizes, n_paired, reasons


# ---------------------------------------------------------------------------- applying


def place_in_reference_frame(structure, frame) -> Any:
    """The atoms of a pose moved into the reference frame: a transformed copy.

    Args:
        structure: A ``Pose`` (its ``atoms()``) or a biotite ``AtomArray`` (or stack).
        frame: A ``FrameResult`` or a ``Transform``.

    Returns:
        A new ``AtomArray`` (same annotations and bonds, float32 coordinates as biotite
        stores them) with every atom, receptor and binder alike, moved by the transform.

    Raises:
        ValueError: If ``frame`` is a failed ``FrameResult`` (it has no transform).
    """
    transform = frame.transform if isinstance(frame, FrameResult) else frame
    if transform is None:
        raise ValueError(f"the frame has no transform: {frame.reason}")
    if not isinstance(transform, Transform):
        raise TypeError(f"frame must be a FrameResult or a Transform, got {type(frame).__name__}")
    atoms = structure.atoms() if isinstance(structure, Pose) else structure
    placed = atoms.copy()
    placed.coord = transform.apply(atoms.coord).astype(atoms.coord.dtype)
    return placed


def _renumber_chain(
    atoms,
    chain: str,
    targets: Mapping,
    first_free: int,
    new_numbers: np.ndarray,
    new_icodes: np.ndarray,
) -> None:
    """Write the new numbers of the residues of ``chain`` into ``new_numbers``/``new_icodes``.

    A residue whose key is in ``targets`` takes the number and insertion code of its value; any
    other residue (unpaired, ligand, water) takes the next unused number from ``first_free``,
    in file order, with no insertion code.
    """
    has_icode = "ins_code" in atoms.get_annotation_categories()
    chain_atoms = np.flatnonzero(np.asarray(atoms.chain_id) == chain)
    numbers = np.asarray(atoms.res_id)[chain_atoms]
    icodes = np.asarray(atoms.ins_code)[chain_atoms] if has_icode else np.full(len(chain_atoms), "")
    names = np.asarray(atoms.res_name)[chain_atoms]
    boundary = np.ones(len(chain_atoms), dtype=bool)
    boundary[1:] = (
        (numbers[1:] != numbers[:-1]) | (icodes[1:] != icodes[:-1]) | (names[1:] != names[:-1])
    )
    starts = np.flatnonzero(boundary)
    stops = np.append(starts[1:], len(chain_atoms))
    next_free = first_free
    for start, stop in zip(starts, stops):
        target = targets.get(ResidueKey(chain, int(numbers[start]), str(icodes[start])))
        rows = chain_atoms[start:stop]
        if target is None:
            number, icode = next_free, ""
            next_free += 1
        else:
            number, icode = target.number, target.icode
        new_numbers[rows] = number
        new_icodes[rows] = icode


def renumber_to_reference(
    structure,
    frame: FrameResult,
    reference,
    *,
    receptor_chain: Optional[str] = None,
    binder_pairing: Optional[ResiduePairing] = None,
):
    """A copy of the atoms with the reference numbers on the paired residues.

    Makes labels such as ``interface_residues_receptor`` share one numbering across poses.
    Residues of the receptor chain that are not paired (unpaired polymer residues, ligands,
    waters) get unused numbers above the highest residue number of the reference receptor
    chain, one per residue in file order, with no insertion code. The binder chain keeps its
    numbers unless ``binder_pairing`` is given; other chains keep their numbers. Coordinates
    are unchanged (apply ``place_in_reference_frame`` separately); the input is never modified.

    Args:
        structure: A ``Pose`` or an ``AtomArray`` of one model.
        frame: The ``FrameResult`` of this pose (it carries the pairs).
        reference: The ``Reference`` or ``PreparedReference`` the frame was computed against.
        receptor_chain: The receptor chain of the pose; default the reference's.
        binder_pairing: The ``pair_binder_residues`` result for this pose. When it is ok, the
            binder chain is renumbered as the receptor chain is, with the numbers of the
            reference binder chain and unused numbers above its highest for the residues
            that are not paired. A pairing with a ``reason`` leaves the binder as it is.

    Raises:
        ValueError: If the frame failed.
    """
    if not frame.ok:
        raise ValueError(f"the frame failed, so there is no pairing to renumber by: {frame.reason}")
    prepared = (
        reference if isinstance(reference, PreparedReference) else prepare_reference(reference)
    )
    atoms = _single_model(
        structure.atoms() if isinstance(structure, Pose) else structure, "the pose"
    )
    chain = receptor_chain or prepared.reference.receptor_chain
    table = _build_table(atoms, chain)
    unique_pose_keys = {key for key in table.keys if key not in table.duplicate_keys}
    new_key = {p: r for p, r in frame.pairs if p in unique_pose_keys}
    renumbered = atoms.copy()
    new_numbers = np.asarray(renumbered.res_id).copy()
    has_icode = "ins_code" in atoms.get_annotation_categories()
    new_icodes = (
        np.asarray(renumbered.ins_code).copy()
        if has_icode
        else np.full(renumbered.array_length(), "", dtype="U1")
    )
    highest = max((key.number for key in prepared.table.keys), default=0)
    _renumber_chain(atoms, chain, new_key, highest + 1, new_numbers, new_icodes)
    if binder_pairing is not None and binder_pairing.ok and prepared.binder_table is not None:
        binder = binder_pairing.pose_keys[0].chain
        binder_key = dict(zip(binder_pairing.pose_keys, binder_pairing.reference_keys))
        highest = max((key.number for key in prepared.binder_table.keys), default=0)
        _renumber_chain(atoms, binder, binder_key, highest + 1, new_numbers, new_icodes)
    if not has_icode:
        renumbered.add_annotation("ins_code", dtype="U1")
    renumbered.res_id = new_numbers
    renumbered.ins_code = new_icodes
    return renumbered


__all__ = [
    "FrameOptions",
    "FrameResult",
    "PreparedReference",
    "ResiduePairing",
    "Transform",
    "canonical_residue_name",
    "is_degenerate",
    "kabsch_transform",
    "one_letter_code",
    "pair_binder_residues",
    "pair_residues",
    "place_in_reference_frame",
    "prepare_reference",
    "renumber_to_reference",
    "residue_names_compatible",
    "rmsd",
    "superpose_receptor",
]
