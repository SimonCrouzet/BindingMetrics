"""Pose agreement: how much the peptides of a pose set agree with each other.

The functions here take poses that are ALREADY in one common frame (the receptors are
superposed, which is the job of the frame stage) and compare the peptides in that frame. The
peptide is never refitted: the RMSD measures displacement and conformation together, which is
the quantity that says whether two models put the peptide in the same place. ``refit=True``
adds a superposition of the peptide on the atoms it measures and gives the shape RMSD, which
separates conformation from position.

Input. A pose is ``(pose_id, atoms)``: a biotite ``AtomArray`` in the common frame and the
chain ID of the binder, which is the same for every pose (use the chain IDs of the user, after
any chain renaming). A set is a mapping ``{pose_id: atoms}`` or a sequence of pairs. Poses are
sorted by id, so the order of the input never changes an output. Nothing here imports biotite:
the arrays are read through their annotations and ``coord`` only.

Atom sets (``kind``).

* ``"ca"``: the alpha carbon of each residue.
* ``"backbone"``: N, CA, C and O of each residue; OXT is not part of it.
* ``"heavy"``: every non-hydrogen atom of the binder residues. Atoms are matched by residue and
  atom name; with ``symmetry="standard"`` the interchangeable side-chain atoms of
  ``binding_metrics.poses.symmetry`` are matched in the labelling with the smallest RMSD.

Water residues and hydrogens are never part of the binder, even when they carry the binder's
chain ID (an author-numbered file lists the waters of a chain under its ID). Atoms with the same
residue and atom name twice (alternate locations) count once: the first one.

Residue correspondence between two poses (``residue_pairing``). A residue is identified by its
key, the tuple ``(chain ID, residue number, insertion code)`` (the three fields of the layer's
``ResidueKey``), so ``52`` and ``52A`` are two residues.

* ``"key"``: residues with the same key are paired; a residue present in one pose only is
  dropped.
* ``"position"``: the i-th residue of one chain is paired with the i-th of the other, whatever
  the numbers; the two chains need the same number of residues.
* ``"auto"`` (default): ``"key"`` when the keys of one pose are all in the other pose (equal
  numbering, or one pose lacks some residues), otherwise ``"position"`` when the lengths are
  equal. A numbering that is shifted between the poses is therefore read by position, and two
  poses that each lack residues the other has are refused with a reason, since neither
  correspondence can be trusted.

Residues that are paired must be the same amino acid: the names are compared after the AMBER
and CHARMM protonation variants and the D-amino acid codes are mapped to their L parent
(HID is His, DAL is Ala), so a model that flips a D-residue to L is measured and not refused,
while Leu against N-methyl-Leu (MLE), or any other change of the sequence, is refused.

Missing residues and atoms. Residues and atoms present in one pose only are dropped from that
pair. The share of atoms kept is ``n_shared / max(n_a, n_b)``, with ``n_a`` and ``n_b`` the
atoms of the chosen set in each binder; below ``min_shared_atom_fraction`` (default 0.8) the
pair has no value and a reason. The same rule serves the three kinds: for ``"ca"`` it is a
share of residues. A pose whose binder cannot be read, or whose sequence differs, has no value
in any of its pairs and says why; the other pairs are computed.

RMSD. For the matched atoms ``i = 1..m``, ``sqrt(sum_i |x_a,i - x_b,i|^2 / m)`` in angstrom, an
average over atoms (not over residues). The matrix is symmetric with a zero diagonal. Because
the symmetric-atom choice is a minimum, a heavy-atom RMSD with ``symmetry="standard"`` need not
obey the triangle inequality.

What "cyclic-aware" means here, and what it does not mean.

The code does these things.

1. No terminus is assumed. Every binder residue takes part; none is trimmed by position, and
   the first and the last residue of the file are treated like any other. A free C-terminal
   OXT is compared in the ``"heavy"`` set only when both poses have it and is never required.
2. Closure atoms are ordinary atoms. The atoms that close a ring (C of the last residue and N
   of the first for a head-to-tail amide, SG of a disulfide, the side-chain atoms of a lactam or
   a staple) belong to the sets like every other atom: C and N are in ``"backbone"``, the
   side-chain atoms in ``"heavy"``. Nothing is added or removed because of a closure.
3. A head-to-tail ring has no chain break. The correspondence is one-to-one by key or position
   and never looks at which residue follows which, so the bond between the last and the first
   residue is neither required nor missed.
4. The origin of a ring can be left free (``cyclic_shift="auto"``, opt-in). For two poses that
   are both closed head-to-tail (C of the last residue within 2.0 angstrom of N of the first,
   the amide threshold of ``capabilities.detect_closures``), have the same number of residues
   and the same residue names under the shift, the pair value is the smallest over all the
   cyclic shifts ``s`` of the pairing "residue i with residue (i + s) mod n"; a tie goes to the
   smallest shift. A pose that is not closed, or a pair without a shift that keeps the residue
   names, falls back to the ordinary pairing. ``shift`` is recorded per pair.
5. Closure families can be compared. ``closure_families`` (or the helper
   ``binder_closure_families``) names the ring-closure families of each pose, and a pair
   carries ``closure_mismatch`` when the two sets differ; ``exclude_closure_mismatch=True``
   gives such pairs no value.

The code does not do these things: it does not detect closures itself, apart from the distance
test of item 4; it does not judge whether a ring is well formed or closed; it does not try the
reversed direction of a ring (the mirror sequence); it applies no weight to closure atoms; and
the cyclic shift is not used by the positional spread or by the contact maps.

Cannot compute. A quantity that cannot be computed is None with a reason, never NaN: in the
dictionaries of ``to_dict`` and in the results of the statistics. The one place where NaN
appears is ``PairMatrix.values``, which is the numeric array of the matrix; NaN there marks an
undefined pair, and ``PairMatrix.reasons`` says why.

Ties. Two distances closer than ``TIE_TOLERANCE_ANGSTROM`` (1e-9 angstrom) are equal wherever a
rule breaks a tie, so that rounding differences (2.3 - 2.0 against 0.3) do not decide.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional, Sequence

import numpy as np

from binding_metrics.core.nonstandard import D_AA_MAP
from binding_metrics.core.residues import (
    BACKBONE_HEAVY_ATOM_NAMES,
    VARIANT_TO_PARENT_RESIDUE,
    WATER_NAMES_ALL,
)
from binding_metrics.poses.symmetry import (
    best_labelling,
    count_swapped,
    resolve_group,
    symmetry_groups,
    symmetry_status,
)

logger = logging.getLogger(__name__)

#: The atom sets of the peptide RMSD.
KINDS = ("ca", "backbone", "heavy")
#: How residues of two poses are paired.
RESIDUE_PAIRINGS = ("auto", "key", "position")
#: ``"off"``: residue i with residue i; ``"auto"``: free ring origin for closed head-to-tail rings.
CYCLIC_SHIFTS = ("off", "auto")
#: ``"standard"``: the symmetric atoms of ``binding_metrics.poses.symmetry``; ``"off"``: names
#: as given.
SYMMETRY_MODES = ("standard", "off")

DEFAULT_MIN_SHARED_ATOM_FRACTION = 0.8
#: Two distances closer than this are equal in every tie rule (angstrom).
TIE_TOLERANCE_ANGSTROM = 1e-9
#: C-N distance below which a binder counts as closed head-to-tail: the amide threshold of
#: ``capabilities.detect_closures`` (a stretched closure bond of a predicted model still counts).
HEAD_TO_TAIL_CUTOFF_ANGSTROM = 2.0

_REFIT_MAX_ITERATIONS = 10
_BACKBONE_ORDER = tuple(sorted(BACKBONE_HEAVY_ATOM_NAMES))


# --------------------------------------------------------------------------- the binder of a pose


@dataclass(frozen=True)
class BinderResidue:
    """One residue of the binder: its key, its name and the rows of its heavy atoms.

    Attributes:
        key: ``(chain ID, residue number, insertion code)``.
        name: Residue name, upper case.
        atom_rows: Heavy atom name to its row in ``BinderView.coord`` (the first of a repeated
            name).
    """

    key: tuple[str, int, str]
    name: str
    atom_rows: Mapping[str, int]


@dataclass(frozen=True, eq=False)
class BinderView:
    """The heavy atoms of the binder chain of one pose, grouped by residue.

    Attributes:
        pose_id: The pose's id (used in messages).
        coord: Heavy-atom coordinates of the binder, shape (n, 3), float64, in file order.
        residues: The binder residues in file order; waters and hydrogens are not part of it.
        keys_unique: Whether no two residues share a key.
        closed_head_to_tail: Whether C of the last residue is within
            ``HEAD_TO_TAIL_CUTOFF_ANGSTROM`` of N of the first.
    """

    pose_id: str
    coord: np.ndarray
    residues: tuple[BinderResidue, ...]
    keys_unique: bool
    closed_head_to_tail: bool

    @classmethod
    def from_atoms(cls, atoms, binder_chain: str, pose_id: str = "") -> "BinderView":
        """Read the binder chain of a biotite ``AtomArray``.

        Raises:
            ValueError: If the structure is a stack, has no atom of the chain, or the chain
                has no heavy atom outside water.
        """
        coord = np.asarray(atoms.coord)
        if coord.ndim != 2:
            raise ValueError("a pose must be one model (an AtomArray), not a stack")
        chain_id = _strings(atoms.chain_id)
        in_chain = chain_id == str(binder_chain)
        if not in_chain.any():
            raise ValueError(f"no atom of chain {binder_chain!r}")
        res_name = np.char.upper(_strings(atoms.res_name))
        keep = in_chain & ~_hydrogen_mask(atoms) & ~np.isin(res_name, sorted(WATER_NAMES_ALL))
        rows = np.flatnonzero(keep)
        if rows.size == 0:
            raise ValueError(f"chain {binder_chain!r} has no heavy atom outside water")
        res_id = np.asarray(atoms.res_id)[rows]
        ins_code = _insertion_codes(atoms)[rows]
        names = res_name[rows]
        atom_name = _strings(atoms.atom_name)[rows]
        changed = np.ones(rows.size, dtype=bool)
        changed[1:] = (
            (res_id[1:] != res_id[:-1])
            | (ins_code[1:] != ins_code[:-1])
            | (names[1:] != names[:-1])
        )
        starts = np.flatnonzero(changed)
        stops = np.append(starts[1:], rows.size)
        residues = []
        for start, stop in zip(starts, stops):
            atom_rows: dict[str, int] = {}
            for row in range(start, stop):
                atom_rows.setdefault(str(atom_name[row]), int(row))
            key = (str(binder_chain), int(res_id[start]), str(ins_code[start]))
            residues.append(BinderResidue(key, str(names[start]), atom_rows))
        keys = [residue.key for residue in residues]
        view_coord = np.asarray(coord[rows], dtype=np.float64)
        return cls(
            pose_id=pose_id,
            coord=view_coord,
            residues=tuple(residues),
            keys_unique=len(set(keys)) == len(keys),
            closed_head_to_tail=_is_closed_head_to_tail(residues, view_coord),
        )


def _strings(values) -> np.ndarray:
    return np.char.strip(np.asarray(values).astype(str))


def _insertion_codes(atoms) -> np.ndarray:
    if "ins_code" in atoms.get_annotation_categories():
        return _strings(atoms.ins_code)
    return np.full(atoms.array_length(), "", dtype="U1")


def _hydrogen_mask(atoms) -> np.ndarray:
    """Hydrogens and deuteriums; the atom name decides only where the element is empty."""
    name = np.char.upper(_strings(atoms.atom_name))
    by_name = np.char.startswith(name, "H")
    if "element" not in atoms.get_annotation_categories():
        return by_name
    element = np.char.upper(_strings(atoms.element))
    return np.where(element != "", np.isin(element, ("H", "D")), by_name)


def _is_closed_head_to_tail(residues: Sequence[BinderResidue], coord: np.ndarray) -> bool:
    nitrogen = next((r.atom_rows["N"] for r in residues if "N" in r.atom_rows), None)
    carbon = next((r.atom_rows["C"] for r in reversed(residues) if "C" in r.atom_rows), None)
    if nitrogen is None or carbon is None or len(residues) < 2:
        return False
    return bool(np.linalg.norm(coord[nitrogen] - coord[carbon]) <= HEAD_TO_TAIL_CUTOFF_ANGSTROM)


def binder_closure_families(atoms, binder_chain: str) -> frozenset[str]:
    """The ring-closure families of the binder of a pose, from ``capabilities.detect_closures``.

    Args:
        atoms: A biotite ``AtomArray`` (with a bond table to also find staples).
        binder_chain: The binder chain ID.

    Returns:
        The closure families found (``head_to_tail``, ``disulfide``, ``lactam``, ...); empty for
        a linear chain. Use it to fill ``closure_families`` of ``pairwise_peptide_rmsd``.
    """
    from binding_metrics.capabilities import detect_closures

    return frozenset(closure.family for closure in detect_closures(atoms, binder_chain))


# --------------------------------------------------------------------------- residue pairing


def _identity_name(res_name: str) -> str:
    """The amino acid a name stands for, with protonation variants and D-forms mapped to L."""
    name = res_name.strip().upper()
    name = VARIANT_TO_PARENT_RESIDUE.get(name, name)
    return D_AA_MAP.get(name, name)


def _label(key: tuple[str, int, str]) -> str:
    return f"{key[0]}:{key[1]}{key[2]}"


def _name_conflict(a: BinderView, b: BinderView, pairs: Sequence[tuple[int, int]]) -> Optional[str]:
    for index_a, index_b in pairs:
        residue_a, residue_b = a.residues[index_a], b.residues[index_b]
        if _identity_name(residue_a.name) != _identity_name(residue_b.name):
            return (
                f"residue {_label(residue_a.key)} is {residue_a.name} in {a.pose_id!r} but "
                f"residue {_label(residue_b.key)} is {residue_b.name} in {b.pose_id!r}"
            )
    return None


def _pair_by_key(a: BinderView, b: BinderView):
    if not (a.keys_unique and b.keys_unique):
        return None, "residue keys are not unique in a binder"
    index_b = {residue.key: j for j, residue in enumerate(b.residues)}
    pairs = [(i, index_b[r.key]) for i, r in enumerate(a.residues) if r.key in index_b]
    if not pairs:
        return None, "no residue key is shared"
    conflict = _name_conflict(a, b, pairs)
    return (None, conflict) if conflict else (pairs, None)


def _pair_by_position(a: BinderView, b: BinderView):
    if len(a.residues) != len(b.residues):
        return None, (
            f"binder lengths differ ({len(a.residues)} residues in {a.pose_id!r}, "
            f"{len(b.residues)} in {b.pose_id!r})"
        )
    pairs = [(i, i) for i in range(len(a.residues))]
    conflict = _name_conflict(a, b, pairs)
    return (None, conflict) if conflict else (pairs, None)


def _pair_residues(a: BinderView, b: BinderView, mode: str):
    """``(pairs, method, reason)``: the residue pairing of two binders, or why there is none."""
    if mode == "key":
        pairs, reason = _pair_by_key(a, b)
        return (pairs, "key", None) if pairs else (None, None, reason)
    if mode == "position":
        pairs, reason = _pair_by_position(a, b)
        return (pairs, "position", None) if pairs else (None, None, reason)
    keys_a = {residue.key for residue in a.residues}
    keys_b = {residue.key for residue in b.residues}
    if a.keys_unique and b.keys_unique and (keys_a <= keys_b or keys_b <= keys_a):
        pairs, key_reason = _pair_by_key(a, b)
        if pairs:
            return pairs, "key", None
    else:
        key_reason = "the residue numbers of one binder are not all in the other"
    pairs, position_reason = _pair_by_position(a, b)
    if pairs:
        return pairs, "position", None
    return None, None, f"no residue pairing: by key, {key_reason}; by position, {position_reason}"


# --------------------------------------------------------------------------- one pair


@dataclass(frozen=True)
class PairRmsd:
    """The RMSD of the binders of two poses, with how it was obtained.

    Attributes:
        pose_a, pose_b: The ids, ``pose_a`` before ``pose_b`` in sorted order.
        rmsd_angstrom: The RMSD, or None when it cannot be computed (then ``reason`` says why).
        reason: Why ``rmsd_angstrom`` is None; None otherwise.
        n_atoms: Matched atoms the RMSD was computed on.
        n_residues: Paired residues that contributed at least one atom.
        shared_atom_fraction: ``n_atoms / max(atoms of the set in a, atoms in b)``.
        pairing: ``"key"`` or ``"position"``; None when no pairing was found.
        shift: The cyclic shift used (0 unless ``cyclic_shift="auto"`` chose another).
        n_swapped_groups: Symmetric-atom groups that are not in the labelling of the file.
        unknown_residues: Residue names of the paired residues that the symmetry table does not
            cover (compared as named); empty unless ``kind="heavy"`` and ``symmetry="standard"``.
        closure_mismatch: Whether the two poses have different closure families; None when
            ``closure_families`` was not given for both.
    """

    pose_a: str
    pose_b: str
    rmsd_angstrom: Optional[float]
    reason: Optional[str] = None
    n_atoms: int = 0
    n_residues: int = 0
    shared_atom_fraction: Optional[float] = None
    pairing: Optional[str] = None
    shift: int = 0
    n_swapped_groups: int = 0
    unknown_residues: tuple[str, ...] = ()
    closure_mismatch: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary; ``reason`` is present only when the RMSD is None."""
        out = dataclasses.asdict(self)
        out["unknown_residues"] = list(self.unknown_residues)
        if self.reason is None:
            del out["reason"]
        return out


@dataclass(frozen=True)
class _Options:
    kind: str
    symmetry: str
    refit: bool
    min_shared_atom_fraction: float


def _kind_atom_names(kind: str, residue: BinderResidue) -> list[str]:
    if kind == "ca":
        return ["CA"] if "CA" in residue.atom_rows else []
    if kind == "backbone":
        return [name for name in _BACKBONE_ORDER if name in residue.atom_rows]
    return list(residue.atom_rows)


def _count_selected(view: BinderView, kind: str) -> int:
    return sum(len(_kind_atom_names(kind, residue)) for residue in view.residues)


def _fit_onto(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """``mobile`` moved onto ``reference`` by the optimal proper rotation and translation.

    Kabsch (Acta Cryst. A32, 922-923, 1976): SVD of the covariance matrix, with the sign of the
    last singular direction corrected so that the transform is never a reflection (the
    convention of ``metrics.comparison``).
    """
    reference_center = reference.mean(axis=0)
    centered_mobile = mobile - mobile.mean(axis=0)
    centered_reference = reference - reference_center
    u, _, vt = np.linalg.svd(centered_mobile.T @ centered_reference)
    sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return centered_mobile @ rotation + reference_center


def _sum_squares(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def _rmsd_of_matched(a: np.ndarray, b: np.ndarray, groups, refit: bool):
    """``(rmsd, n_swapped_groups, reason)`` for matched coordinates and resolved swap groups."""
    n_atoms = len(a)
    if not refit:
        if groups:
            order = best_labelling(a, b, groups)
            return (
                math.sqrt(_sum_squares(a, b[order]) / n_atoms),
                count_swapped(order, groups),
                None,
            )
        return math.sqrt(_sum_squares(a, b) / n_atoms), 0, None
    if n_atoms < 3:
        return None, 0, f"a fit needs at least 3 atoms, {n_atoms} are shared"
    # Alternating minimisation: fit, choose the labelling in the fitted frame, fit again. Each
    # step keeps or lowers the sum of squares, so it stops; the choice inside a step is the
    # deterministic one of ``best_labelling``.
    order = np.arange(n_atoms, dtype=np.intp)
    fitted = _fit_onto(a, b)
    cost = _sum_squares(a, fitted)
    for _ in range(_REFIT_MAX_ITERATIONS if groups else 0):
        step = best_labelling(a, fitted, groups)
        if np.array_equal(step, np.arange(n_atoms)):
            break
        candidate = _fit_onto(a, fitted[step])
        candidate_cost = _sum_squares(a, candidate)
        if candidate_cost >= cost - 1e-12:
            break
        order, fitted, cost = order[step], candidate, candidate_cost
    swapped = count_swapped(order, groups) if groups else 0
    return math.sqrt(cost / n_atoms), swapped, None


def _failure(a: BinderView, b: BinderView, reason: str, **extra) -> PairRmsd:
    return PairRmsd(a.pose_id, b.pose_id, None, reason=reason, **extra)


def _measure(
    a: BinderView, b: BinderView, pairs, method: str, shift: int, options: _Options
) -> PairRmsd:
    """The RMSD of two binders under a given residue pairing."""
    expected = max(_count_selected(a, options.kind), _count_selected(b, options.kind))
    if expected == 0:
        return _failure(a, b, f"the binders have no {options.kind} atom", pairing=method)
    rows_a: list[int] = []
    rows_b: list[int] = []
    residues = []
    for index_a, index_b in pairs:
        residue_a, residue_b = a.residues[index_a], b.residues[index_b]
        rows_by_name: dict[str, int] = {}
        for name in _kind_atom_names(options.kind, residue_a):
            if name in residue_b.atom_rows:
                rows_by_name[name] = len(rows_a)
                rows_a.append(residue_a.atom_rows[name])
                rows_b.append(residue_b.atom_rows[name])
        residues.append((residue_a, residue_b, rows_by_name))
    n_shared = len(rows_a)
    fraction = n_shared / expected
    common = dict(pairing=method, shift=shift, n_atoms=n_shared, shared_atom_fraction=fraction)
    if n_shared == 0:
        return _failure(a, b, "the paired residues share no atom of the set", **common)
    if fraction < options.min_shared_atom_fraction:
        return _failure(
            a,
            b,
            f"only {n_shared} of {expected} atoms are shared ({fraction:.2f}, below "
            f"min_shared_atom_fraction {options.min_shared_atom_fraction:g})",
            **common,
        )
    coord_a, coord_b = a.coord[rows_a], b.coord[rows_b]
    if not (np.isfinite(coord_a).all() and np.isfinite(coord_b).all()):
        return _failure(a, b, "a matched atom has a non-finite coordinate", **common)
    groups = []
    unknown: set[str] = set()
    if options.kind == "heavy" and options.symmetry == "standard":
        for residue_a, residue_b, rows_by_name in residues:
            for residue in (residue_a, residue_b):
                if symmetry_status(residue.name) == "unknown":
                    unknown.add(residue.name)
            for group in symmetry_groups(residue_a.name):
                resolved = resolve_group(group, rows_by_name)
                if resolved is not None:
                    groups.append(resolved)
    rmsd, n_swapped, reason = _rmsd_of_matched(coord_a, coord_b, groups, options.refit)
    if rmsd is None:
        return _failure(a, b, reason, **common)
    return PairRmsd(
        pose_a=a.pose_id,
        pose_b=b.pose_id,
        rmsd_angstrom=rmsd,
        n_residues=sum(1 for _, _, rows in residues if rows),
        n_swapped_groups=n_swapped,
        unknown_residues=tuple(sorted(unknown)),
        **common,
    )


def _compare(
    a: BinderView, b: BinderView, options: _Options, residue_pairing: str, cyclic_shift: str
) -> PairRmsd:
    n_residues = len(a.residues)
    if (
        cyclic_shift == "auto"
        and n_residues == len(b.residues)
        and a.closed_head_to_tail
        and b.closed_head_to_tail
    ):
        best: Optional[PairRmsd] = None
        fallback: Optional[PairRmsd] = None
        for shift in range(n_residues):
            pairs = [(i, (i + shift) % n_residues) for i in range(n_residues)]
            if _name_conflict(a, b, pairs) is not None:
                continue
            result = _measure(a, b, pairs, "position", shift, options)
            fallback = fallback or result
            if result.rmsd_angstrom is None:
                continue
            if best is None or result.rmsd_angstrom < best.rmsd_angstrom - TIE_TOLERANCE_ANGSTROM:
                best = result
        if best is not None or fallback is not None:
            return best or fallback
    pairs, method, reason = _pair_residues(a, b, residue_pairing)
    if pairs is None:
        return _failure(a, b, reason)
    return _measure(a, b, pairs, method, 0, options)


# --------------------------------------------------------------------------- matrices


@dataclass(frozen=True, eq=False)
class PairMatrix:
    """A symmetric ``n x n`` matrix over the poses of a set, with a reason for each gap.

    Attributes:
        name: What the values are (``"rmsd_ca"``, ``"jaccard_interface"``, ...).
        unit: ``"angstrom"`` for a distance, None for a ratio.
        direction: ``"distance"`` (0 on the diagonal; small means close) or ``"similarity"``
            (1 on the diagonal; large means close).
        pose_ids: The ids, sorted; row and column ``i`` belong to ``pose_ids[i]``.
        values: The array, shape (n, n), float64, read-only. NaN marks an undefined entry:
            this array is the only place where NaN is used, and ``reasons`` explains each one.
        reasons: ``{(id_i, id_j): reason}`` for each undefined pair with ``i < j``.
    """

    name: str
    unit: Optional[str]
    direction: str
    pose_ids: tuple[str, ...]
    values: np.ndarray
    reasons: Mapping[tuple[str, str], str]

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    def index(self, pose_id: str) -> int:
        """Row of a pose; raises ``KeyError`` for an id that is not in the matrix."""
        try:
            return self.pose_ids.index(pose_id)
        except ValueError:
            raise KeyError(f"no pose {pose_id!r} in the matrix {self.pose_ids}") from None

    def value(self, pose_a: str, pose_b: str) -> Optional[float]:
        """The entry of two poses, None where it is undefined."""
        entry = self.values[self.index(pose_a), self.index(pose_b)]
        return None if np.isnan(entry) else float(entry)

    def reason(self, pose_a: str, pose_b: str) -> Optional[str]:
        """Why the entry of two poses is undefined; None when it is defined."""
        key = tuple(sorted((pose_a, pose_b)))
        return self.reasons.get(key)

    def subset(self, pose_ids: Collection[str]) -> "PairMatrix":
        """The matrix restricted to some poses (sorted), for a group or after dropping a pose."""
        wanted = sorted(set(pose_ids))
        rows = [self.index(pose_id) for pose_id in wanted]
        reasons = {key: why for key, why in self.reasons.items() if set(key) <= set(wanted)}
        return PairMatrix(
            self.name,
            self.unit,
            self.direction,
            tuple(wanted),
            self.values[np.ix_(rows, rows)],
            reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary: NaN becomes None, the undefined pairs are listed with reasons."""
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "pose_ids": list(self.pose_ids),
            "matrix": [[None if np.isnan(v) else float(v) for v in row] for row in self.values],
            "undefined_pairs": [
                {"pair": list(pair), "reason": why} for pair, why in sorted(self.reasons.items())
            ],
        }


@dataclass(frozen=True, eq=False)
class PairwiseRmsd(PairMatrix):
    """The all-pairs peptide RMSD matrix of a set, with the detail of each pair.

    Attributes beyond ``PairMatrix``:
        atom_kind: The atom set, one of ``KINDS``.
        pairs: ``{(id_i, id_j): PairRmsd}`` for every pair with ``i < j``.
        unknown_residues: The residue names found in the binders that the symmetry table does
            not cover, compared as named; empty unless ``kind="heavy"`` and
            ``symmetry="standard"``.
        pose_reasons: ``{pose_id: reason}`` for the poses whose binder could not be read; each of
            their pairs is undefined.
        parameters: The resolved options, for the result's ``parameters`` block.
    """

    atom_kind: str
    pairs: Mapping[tuple[str, str], PairRmsd]
    unknown_residues: tuple[str, ...]
    pose_reasons: Mapping[str, str]
    parameters: Mapping[str, Any]

    def to_dict(self, include_pairs: bool = False) -> dict[str, Any]:
        """JSON-ready dictionary; ``include_pairs`` adds the detail of every pair."""
        out = PairMatrix.to_dict(self)
        out.update(
            atom_kind=self.atom_kind,
            unknown_residues=list(self.unknown_residues),
            pose_reasons=dict(self.pose_reasons),
            parameters=dict(self.parameters),
        )
        if include_pairs:
            out["pairs"] = [pair.to_dict() for _, pair in sorted(self.pairs.items())]
        return out


def _items(poses) -> list[tuple[str, Any]]:
    """The poses as a list of ``(id, atoms)``, ids checked."""
    items = list(poses.items()) if isinstance(poses, Mapping) else [tuple(p) for p in poses]
    ids = []
    for item in items:
        if len(item) != 2 or not isinstance(item[0], str) or not item[0]:
            raise TypeError("poses must be {id: atoms} or (id, atoms) pairs with a non-empty id")
        ids.append(item[0])
    duplicated = sorted({pose_id for pose_id in ids if ids.count(pose_id) > 1})
    if duplicated:
        raise ValueError(f"pose ids are not unique: {duplicated}")
    return sorted(items, key=lambda item: item[0])


def _check_choice(name: str, value: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def pairwise_peptide_rmsd(
    poses,
    binder_chain: str,
    *,
    kind: str = "ca",
    symmetry: str = "standard",
    residue_pairing: str = "auto",
    cyclic_shift: str = "off",
    refit: bool = False,
    min_shared_atom_fraction: float = DEFAULT_MIN_SHARED_ATOM_FRACTION,
    closure_families: Optional[Mapping[str, Collection[str]]] = None,
    exclude_closure_mismatch: bool = False,
) -> PairwiseRmsd:
    """The RMSD of the binder between every two poses of a set, in the frame the poses are in.

    See the module docstring for the atom sets, the residue correspondence, the missing-residue
    policy and what "cyclic-aware" does and does not mean. Deterministic: no random numbers.

    Args:
        poses: ``{pose_id: atoms}`` or a sequence of ``(pose_id, atoms)``, each ``atoms`` a
            biotite ``AtomArray`` already in the common frame.
        binder_chain: Chain ID of the binder in every pose.
        kind: ``"ca"`` (default), ``"backbone"`` or ``"heavy"``.
        symmetry: ``"standard"`` (default) or ``"off"``; only ``kind="heavy"`` is affected.
        residue_pairing: ``"auto"`` (default), ``"key"`` or ``"position"``.
        cyclic_shift: ``"off"`` (default) or ``"auto"``.
        refit: False (default) gives the displacement RMSD in the common frame; True superposes
            the matched atoms of the two peptides first (shape RMSD; at least 3 atoms).
        min_shared_atom_fraction: A pair that shares a smaller fraction of the atoms of the
            larger set has no value.
        closure_families: ``{pose_id: closure families}``, for the ``closure_mismatch`` flag of
            the pairs (see ``binder_closure_families``).
        exclude_closure_mismatch: Give a pair with different closure families no value.

    Returns:
        A ``PairwiseRmsd``. With fewer than two poses the matrix has no off-diagonal entry.

    Raises:
        ValueError: For an unknown option, a threshold outside (0, 1], or repeated pose ids.
        TypeError: If ``poses`` is not in one of the two forms.
    """
    _check_choice("kind", kind, KINDS)
    _check_choice("symmetry", symmetry, SYMMETRY_MODES)
    _check_choice("residue_pairing", residue_pairing, RESIDUE_PAIRINGS)
    _check_choice("cyclic_shift", cyclic_shift, CYCLIC_SHIFTS)
    if not 0.0 < min_shared_atom_fraction <= 1.0:
        raise ValueError(
            f"min_shared_atom_fraction must be in (0, 1], got {min_shared_atom_fraction!r}"
        )
    options = _Options(kind, symmetry, bool(refit), float(min_shared_atom_fraction))
    items = _items(poses)
    pose_ids = tuple(pose_id for pose_id, _ in items)
    views: dict[str, BinderView] = {}
    pose_reasons: dict[str, str] = {}
    for pose_id, atoms in items:
        try:
            views[pose_id] = BinderView.from_atoms(atoms, binder_chain, pose_id)
        except ValueError as error:
            pose_reasons[pose_id] = f"pose {pose_id!r}: {error}"
            logger.warning("%s", pose_reasons[pose_id])
    families = {pose_id: frozenset(found) for pose_id, found in (closure_families or {}).items()}
    n_poses = len(pose_ids)
    values = np.zeros((n_poses, n_poses), dtype=np.float64)
    reasons: dict[tuple[str, str], str] = {}
    pairs: dict[tuple[str, str], PairRmsd] = {}
    for i in range(n_poses):
        for j in range(i + 1, n_poses):
            id_a, id_b = pose_ids[i], pose_ids[j]
            broken = [pose_reasons[p] for p in (id_a, id_b) if p in pose_reasons]
            if broken:
                result = PairRmsd(id_a, id_b, None, reason="; ".join(broken))
            else:
                result = _compare(views[id_a], views[id_b], options, residue_pairing, cyclic_shift)
            if id_a in families and id_b in families:
                mismatch = families[id_a] != families[id_b]
                result = dataclasses.replace(result, closure_mismatch=mismatch)
                if mismatch and exclude_closure_mismatch:
                    result = dataclasses.replace(
                        result,
                        rmsd_angstrom=None,
                        reason=(
                            "closure families differ: "
                            f"{sorted(families[id_a])} against {sorted(families[id_b])}"
                        ),
                    )
            pairs[(id_a, id_b)] = result
            if result.rmsd_angstrom is None:
                values[i, j] = values[j, i] = np.nan
                reasons[(id_a, id_b)] = result.reason or "no value"
            else:
                values[i, j] = values[j, i] = result.rmsd_angstrom
    unknown: tuple[str, ...] = ()
    if kind == "heavy" and symmetry == "standard":
        unknown = tuple(
            sorted(
                {
                    residue.name
                    for view in views.values()
                    for residue in view.residues
                    if symmetry_status(residue.name) == "unknown"
                }
            )
        )
        if unknown:
            logger.info(
                "residues without a symmetric-atom entry, compared as named: %s", list(unknown)
            )
    return PairwiseRmsd(
        name=f"rmsd_{kind}",
        unit="angstrom",
        direction="distance",
        pose_ids=pose_ids,
        values=values,
        reasons=reasons,
        atom_kind=kind,
        pairs=pairs,
        unknown_residues=unknown,
        pose_reasons=pose_reasons,
        parameters={
            "kind": kind,
            "symmetry": symmetry,
            "residue_pairing": residue_pairing,
            "cyclic_shift": cyclic_shift,
            "refit": bool(refit),
            "min_shared_atom_fraction": float(min_shared_atom_fraction),
            "exclude_closure_mismatch": bool(exclude_closure_mismatch),
        },
    )


def peptide_rmsd(atoms_a, atoms_b, binder_chain: str, **options) -> PairRmsd:
    """The RMSD of the binder of two poses already in one frame (the poses are called
    ``"first"`` and ``"second"`` in the reasons).

    Takes the options of ``pairwise_peptide_rmsd`` except the closure ones.
    """
    matrix = pairwise_peptide_rmsd({"first": atoms_a, "second": atoms_b}, binder_chain, **options)
    return matrix.pairs[("first", "second")]


# --------------------------------------------------------------------------- statistics of a matrix


def _statistic(matrix: PairMatrix, index_pairs, empty_reason: str) -> dict[str, Any]:
    """min, mean, max and the extreme pairs of the defined entries of some pairs."""
    n_pairs = len(index_pairs)
    entries = [(float(matrix.values[i, j]), (i, j)) for i, j in index_pairs]
    defined = [entry for entry in entries if not math.isnan(entry[0])]
    out: dict[str, Any] = {
        "n_pairs": n_pairs,
        "n_defined": len(defined),
        "min": None,
        "mean": None,
        "max": None,
        "min_pair": None,
        "max_pair": None,
    }
    if n_pairs == 0:
        out["reason"] = empty_reason
        return out
    if not defined:
        first = matrix.reasons.get(
            (matrix.pose_ids[index_pairs[0][0]], matrix.pose_ids[index_pairs[0][1]]), "no value"
        )
        out["reason"] = f"no pair has a value (first reason: {first})"
        return out
    low = min(defined, key=lambda entry: entry[0])
    high = max(defined, key=lambda entry: entry[0])
    out["min"], out["max"] = low[0], high[0]
    out["mean"] = float(np.mean([value for value, _ in defined]))
    out["min_pair"] = [matrix.pose_ids[low[1][0]], matrix.pose_ids[low[1][1]]]
    out["max_pair"] = [matrix.pose_ids[high[1][0]], matrix.pose_ids[high[1][1]]]
    return out


def pair_statistics(
    matrix: PairMatrix, groups: Optional[Mapping[str, str]] = None
) -> dict[str, Any]:
    """Minimum, mean and maximum of a matrix over the pairs inside and between groups.

    Each unordered pair of distinct poses counts once, and the diagonal is never used. A pair
    whose entry is undefined is counted in ``n_pairs`` and left out of ``n_defined`` and of the
    statistics. A group of one pose has no pair, so its statistics are None with a reason.

    Args:
        matrix: Any ``PairMatrix`` (the RMSD matrix, a Jaccard matrix, ...).
        groups: ``{pose_id: group label}``, for example the back-end of each pose. None puts all
            poses in one group called ``"all"``. Every pose of the matrix needs a label; extra
            keys are ignored.

    Returns:
        ``{"n_poses", "overall", "within", "between", "between_reason"}``. Each statistic is
        ``{"n_pairs", "n_defined", "min", "mean", "max", "min_pair", "max_pair"}`` plus
        ``"reason"`` when the values are None; the pairs are lists of two ids. ``within`` is keyed
        by group, ``between`` by ``"a|b"`` for two groups in sorted order. With a single group
        ``between`` is empty and ``between_reason`` is ``"only one group"``.

    Raises:
        ValueError: If a pose of the matrix has no group label.
    """
    ids = matrix.pose_ids
    labels = {pose_id: "all" for pose_id in ids} if groups is None else dict(groups)
    missing = [pose_id for pose_id in ids if pose_id not in labels]
    if missing:
        raise ValueError(f"poses without a group label: {missing}")
    n_poses = len(ids)
    all_pairs = [(i, j) for i in range(n_poses) for j in range(i + 1, n_poses)]
    names = sorted({labels[pose_id] for pose_id in ids})
    within = {}
    for name in names:
        members = [i for i, pose_id in enumerate(ids) if labels[pose_id] == name]
        pairs = [(i, j) for i, j in all_pairs if i in members and j in members]
        within[name] = _statistic(
            matrix, pairs, f"needs at least 2 poses (group {name!r} has {len(members)})"
        )
    between = {}
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            pairs = [
                (i, j) for i, j in all_pairs if {labels[ids[i]], labels[ids[j]]} == {first, second}
            ]
            between[f"{first}|{second}"] = _statistic(matrix, pairs, "no pair")
    return {
        "n_poses": n_poses,
        "overall": _statistic(matrix, all_pairs, f"needs at least 2 poses (got {n_poses})"),
        "within": within,
        "between": between,
        "between_reason": "only one group" if len(names) < 2 else None,
    }
