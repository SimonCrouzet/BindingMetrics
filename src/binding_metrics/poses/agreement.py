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

Statistics of a set. Every statistic takes a ``PairMatrix`` (or atoms) and says what it needs.

* ``pair_statistics``: minimum, mean and maximum over the pairs inside each group and between
  groups; needs one pair.
* ``complete_linkage_clusters``: agglomerative clustering with complete linkage, implemented
  here. Start with one cluster per pose, ordered by pose id; at each step merge the two clusters
  with the smallest linkage distance (the largest distance between a member of one and a member
  of the other); stop when that distance exceeds the cutoff, a distance within the tolerance of
  the cutoff counting as within it. Ties go to the pair of clusters with the smallest
  ``(rank of the lowest member of the first cluster, rank of the lowest member of the second)``,
  the rank being the position in the sorted ids. Cluster labels run from 0 and follow
  ``(-size, rank of the lowest member)``. Per cutoff: the number of clusters, the sizes, the
  fraction of poses in the largest cluster, the label of each pose and, when groups are given,
  the clusters that mix groups. An undefined distance counts as infinite, so those two poses
  never share a cluster. The linkage is complete and fixed (``CLUSTER_LINKAGE``); the cutoffs
  default to ``DEFAULT_CLUSTER_CUTOFFS_ANGSTROM``. Needs two poses.
* ``medoid`` and ``medoids``: the pose with the smallest sum of distances to the other poses
  (the smallest mean, which orders the poses the same way); a tie, within the tolerance of the
  sums, goes to the lowest pose id. A pose with an undefined distance to another pose is not a
  candidate. One pose is its own medoid, with no distance. ``medoids`` gives the overall one,
  one per group and the RMSD between the medoids of two groups.
* ``jaccard_matrix`` (and ``interface_jaccard``, ``contact_map_jaccard``): the Jaccard index
  ``|A & B| / |A | B|`` of two sets; two empty sets give no value, with a reason. The interface
  and contact sets come from the caller (``heavy_atom_contacts`` and ``interface_residues``
  build them from atoms with a plain distance rule: a residue is in the interface, and a residue
  pair is in contact, when two of their heavy atoms are closer than ``contact_cutoff_angstrom``,
  default 5.0, the contact distance of the CAPRI fraction of native contacts). Binder residues
  are named by their 1-based position in the binder chain, receptor residues by their key; the
  numbers of the receptor residues must already be those of one numbering for all poses.
* ``positional_spread``: for each binder position, the root mean square distance of the alpha
  carbons of the poses from their mean position, ``sqrt(mean_i |x_i - mean(x)|^2)`` in angstrom
  (divisor n, the population form, like the RMSF of the toolkit), and the mean of that over the
  positions. The poses are in the common frame, so no further superposition is applied; it
  describes the poses given and does not estimate a population spread. A position needs two
  poses with an alpha carbon.

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
from typing import Any, Callable, Collection, Mapping, NamedTuple, Optional, Sequence

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

#: The linkage of ``complete_linkage_clusters``; it is not a parameter.
CLUSTER_LINKAGE = "complete"
#: Cutoffs (angstrom) of the clustering when the caller gives none: the scale of the seed spread
#: of one model and of the gap between two models for a peptide of about a dozen residues.
DEFAULT_CLUSTER_CUTOFFS_ANGSTROM = (1.0, 2.0)
#: Heavy-atom distance (angstrom) below which two residues are in contact.
DEFAULT_CONTACT_CUTOFF_ANGSTROM = 5.0

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
        atom_residue: Index into ``residues`` of the residue of each row of ``coord``.
        keys_unique: Whether no two residues share a key.
        closed_head_to_tail: Whether C of the last residue is within
            ``HEAD_TO_TAIL_CUTOFF_ANGSTROM`` of N of the first.
    """

    pose_id: str
    coord: np.ndarray
    residues: tuple[BinderResidue, ...]
    atom_residue: np.ndarray
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
            atom_residue=np.cumsum(changed) - 1,
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


# --------------------------------------------------------------------------- medoid


def _require_distance(matrix: PairMatrix, what: str) -> None:
    if matrix.direction != "distance":
        raise ValueError(f"{what} needs a distance matrix, got a {matrix.direction} matrix")


def medoid(matrix: PairMatrix) -> dict[str, Any]:
    """The pose with the smallest sum of distances to the other poses of a distance matrix.

    A tie, within ``TIE_TOLERANCE_ANGSTROM`` per distance, goes to the lowest pose id. A pose
    with an undefined distance to another pose of the matrix is not a candidate; take a
    ``subset`` first to leave a pose out.

    Returns:
        ``{"pose_id", "sum_distance_angstrom", "mean_distance_angstrom", "tied_with",
        "n_poses", "excluded"}``; ``tied_with`` lists the other poses that tie, ``excluded`` the
        poses that were not candidates. One pose is its own medoid with no distance and a
        ``reason``; with no candidate ``pose_id`` is None with a ``reason``.

    Raises:
        ValueError: If the matrix is not a distance matrix.
    """
    _require_distance(matrix, "medoid")
    ids = matrix.pose_ids
    n_poses = len(ids)
    out: dict[str, Any] = {
        "pose_id": None,
        "sum_distance_angstrom": None,
        "mean_distance_angstrom": None,
        "tied_with": [],
        "n_poses": n_poses,
        "excluded": [],
    }
    if n_poses == 0:
        out["reason"] = "no pose"
        return out
    if n_poses == 1:
        out["pose_id"] = ids[0]
        out["reason"] = "one pose: it is its own medoid and has no distance to compare"
        return out
    off_diagonal = matrix.values.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    complete = ~np.isnan(off_diagonal).any(axis=1)
    out["excluded"] = [pose_id for pose_id, ok in zip(ids, complete) if not ok]
    if not complete.any():
        out["reason"] = "no pose has a defined distance to every other pose"
        return out
    sums = off_diagonal.sum(axis=1)
    tolerance = TIE_TOLERANCE_ANGSTROM * (n_poses - 1)
    smallest = sums[complete].min()
    winners = [i for i in range(n_poses) if complete[i] and sums[i] <= smallest + tolerance]
    chosen = winners[0]  # the ids are sorted, so the first winner has the lowest id
    out["pose_id"] = ids[chosen]
    out["sum_distance_angstrom"] = float(sums[chosen])
    out["mean_distance_angstrom"] = float(sums[chosen] / (n_poses - 1))
    out["tied_with"] = [ids[i] for i in winners[1:]]
    return out


def medoids(matrix: PairMatrix, groups: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """The medoid of the whole set, of each group, and the RMSD between the medoids of two groups.

    Args:
        matrix: A distance matrix.
        groups: ``{pose_id: group label}``; None gives only the overall medoid.

    Returns:
        ``{"overall": medoid(matrix), "per_group": {group: medoid}, "between_groups":
        {"a|b": {"medoid_a", "medoid_b", "distance_angstrom"}}}``; ``distance_angstrom`` is None
        with a ``reason`` when a group has no medoid or the two medoids have no defined
        distance.

    Raises:
        ValueError: If the matrix is not a distance matrix, or a pose has no group label.
    """
    _require_distance(matrix, "medoids")
    out: dict[str, Any] = {"overall": medoid(matrix), "per_group": {}, "between_groups": {}}
    if groups is None:
        return out
    missing = [pose_id for pose_id in matrix.pose_ids if pose_id not in groups]
    if missing:
        raise ValueError(f"poses without a group label: {missing}")
    names = sorted({groups[pose_id] for pose_id in matrix.pose_ids})
    for name in names:
        members = [pose_id for pose_id in matrix.pose_ids if groups[pose_id] == name]
        out["per_group"][name] = medoid(matrix.subset(members))
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            id_a = out["per_group"][first]["pose_id"]
            id_b = out["per_group"][second]["pose_id"]
            entry: dict[str, Any] = {"medoid_a": id_a, "medoid_b": id_b, "distance_angstrom": None}
            if id_a is None or id_b is None:
                entry["reason"] = "a group has no medoid"
            else:
                distance = matrix.value(id_a, id_b)
                entry["distance_angstrom"] = distance
                if distance is None:
                    entry["reason"] = matrix.reason(id_a, id_b) or "no value"
            out["between_groups"][f"{first}|{second}"] = entry
    return out


# --------------------------------------------------------------------------- clustering


@dataclass(frozen=True)
class ClusterResult:
    """Complete-linkage clusters of a set at one cutoff.

    Attributes:
        cutoff_angstrom: The cutoff.
        n_clusters: Number of clusters; None when the set has fewer than two poses.
        sizes: Cluster sizes in label order (largest first, then by lowest pose id).
        largest_cluster_fraction: ``max(sizes) / n_poses``.
        labels: ``{pose_id: label}``.
        clusters: Pose ids of each cluster, in label order, each sorted.
        mixed_clusters: Labels of the clusters that contain poses of more than one group; None
            when no groups were given.
        n_undefined_pairs: Pairs without a distance, which never share a cluster.
        reason: Why the numbers are None; None otherwise.
    """

    cutoff_angstrom: float
    n_clusters: Optional[int]
    sizes: Optional[tuple[int, ...]]
    largest_cluster_fraction: Optional[float]
    labels: Optional[dict[str, int]]
    clusters: Optional[tuple[tuple[str, ...], ...]]
    mixed_clusters: Optional[tuple[int, ...]]
    n_undefined_pairs: int = 0
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary; ``reason`` is present only when the numbers are None."""
        out: dict[str, Any] = {
            "cutoff_angstrom": self.cutoff_angstrom,
            "linkage": CLUSTER_LINKAGE,
            "n_clusters": self.n_clusters,
            "sizes": None if self.sizes is None else list(self.sizes),
            "largest_cluster_fraction": self.largest_cluster_fraction,
            "labels": None if self.labels is None else dict(self.labels),
            "clusters": None if self.clusters is None else [list(c) for c in self.clusters],
            "mixed_clusters": None if self.mixed_clusters is None else list(self.mixed_clusters),
            "n_undefined_pairs": self.n_undefined_pairs,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def _merge_sequence(distances: np.ndarray) -> list[tuple[float, list[int], list[int]]]:
    """Every complete-linkage merge of the poses, in order: ``(height, first, second)``.

    Poses are the rows of ``distances`` (rank = row number). The clusters stay in a list ordered
    by their lowest member, so the first pair of the row-major scan of the upper triangle among
    the tied smallest linkage distances is the one with the smallest ``(rank of the lowest member
    of the first cluster, rank of the lowest member of the second)``. The merged cluster keeps the
    place of the first one, which keeps the order. The sequence does not depend on a cutoff: a
    cutoff only decides how many merges are applied.
    """
    linkage = np.where(np.isnan(distances), np.inf, distances).astype(np.float64)
    np.fill_diagonal(linkage, np.inf)
    members: list[list[int]] = [[i] for i in range(len(linkage))]
    merges: list[tuple[float, list[int], list[int]]] = []
    while len(members) > 1:
        upper_rows, upper_columns = np.triu_indices(len(members), 1)
        values = linkage[upper_rows, upper_columns]
        smallest = values.min()
        if not np.isfinite(smallest):
            break
        chosen = int(np.flatnonzero(values <= smallest + TIE_TOLERANCE_ANGSTROM)[0])
        i, j = int(upper_rows[chosen]), int(upper_columns[chosen])
        merges.append((float(linkage[i, j]), list(members[i]), list(members[j])))
        merged = np.maximum(linkage[i], linkage[j])  # complete linkage: the farthest member
        linkage[i, :] = merged
        linkage[:, i] = merged
        linkage[i, i] = np.inf
        linkage = np.delete(np.delete(linkage, j, axis=0), j, axis=1)
        members[i] = sorted(members[i] + members[j])
        del members[j]
    return merges


def _clusters_at(n_poses: int, merges, cutoff: float) -> list[list[int]]:
    members: list[list[int]] = [[i] for i in range(n_poses)]
    position = {i: i for i in range(n_poses)}  # lowest member -> index in ``members``
    for height, first, second in merges:
        if height > cutoff + TIE_TOLERANCE_ANGSTROM:
            break
        i, j = position[first[0]], position[second[0]]
        members[i] = sorted(members[i] + members[j])
        del members[j]
        position = {cluster[0]: index for index, cluster in enumerate(members)}
    return members


def complete_linkage_clusters(
    matrix: PairMatrix,
    cutoffs_angstrom: Sequence[float] = DEFAULT_CLUSTER_CUTOFFS_ANGSTROM,
    *,
    groups: Optional[Mapping[str, str]] = None,
) -> list[ClusterResult]:
    """Cluster the poses of a distance matrix by complete linkage, once per cutoff.

    The module docstring gives the rule and the tie rule. The result is deterministic and does
    not depend on the order in which the poses were given. Poses within a cluster are all within
    the cutoff of each other (the cutoff is the largest allowed distance inside a cluster).

    Args:
        matrix: A distance matrix (``PairwiseRmsd`` for example).
        cutoffs_angstrom: The cutoffs, finite and positive; one result for each, in order.
        groups: ``{pose_id: group label}``, to find the clusters that mix groups.

    Returns:
        One ``ClusterResult`` per cutoff. With fewer than two poses the numbers are None with a
        reason.

    Raises:
        ValueError: For a cutoff that is not a positive finite number, a matrix that is not a
            distance matrix, or a pose without a group label.
    """
    _require_distance(matrix, "clustering")
    cutoffs = [
        float(c)
        for c in ([cutoffs_angstrom] if np.isscalar(cutoffs_angstrom) else cutoffs_angstrom)
    ]
    bad = [c for c in cutoffs if not (math.isfinite(c) and c > 0)]
    if bad or not cutoffs:
        raise ValueError(
            f"cutoffs_angstrom must be positive finite numbers, got {list(cutoffs_angstrom)}"
        )
    ids = matrix.pose_ids
    n_poses = len(ids)
    if groups is not None:
        missing = [pose_id for pose_id in ids if pose_id not in groups]
        if missing:
            raise ValueError(f"poses without a group label: {missing}")
    if n_poses < 2:
        reason = f"needs at least 2 poses (got {n_poses})"
        return [ClusterResult(c, None, None, None, None, None, None, 0, reason) for c in cutoffs]
    n_undefined = int(np.isnan(np.triu(matrix.values, 1)).sum())
    merges = _merge_sequence(matrix.values)
    results = []
    for cutoff in cutoffs:
        clusters = _clusters_at(n_poses, merges, cutoff)
        clusters.sort(key=lambda members: (-len(members), members[0]))
        labels = {ids[i]: label for label, members in enumerate(clusters) for i in members}
        mixed = None
        if groups is not None:
            mixed = tuple(
                label
                for label, members in enumerate(clusters)
                if len({groups[ids[i]] for i in members}) > 1
            )
        results.append(
            ClusterResult(
                cutoff_angstrom=cutoff,
                n_clusters=len(clusters),
                sizes=tuple(len(members) for members in clusters),
                largest_cluster_fraction=len(clusters[0]) / n_poses,
                labels=labels,
                clusters=tuple(tuple(ids[i] for i in members) for members in clusters),
                mixed_clusters=mixed,
                n_undefined_pairs=n_undefined,
            )
        )
    return results


# --------------------------------------------------------------------------- Jaccard


def jaccard_matrix(
    sets_by_pose: Mapping[str, Collection],
    *,
    name: str = "jaccard",
    empty_reason: str = "both sets are empty",
) -> PairMatrix:
    """The Jaccard index ``|A & B| / |A | B|`` of the set of each pose with that of each other pose.

    Args:
        sets_by_pose: ``{pose_id: collection of hashable items}``, for example the interface
            residues of each pose.
        name: Name of the matrix.
        empty_reason: The reason given for a pair of two empty sets, which has no value.

    Returns:
        A similarity ``PairMatrix`` (1 on the diagonal, except for an empty set, which has no
        value there either).
    """
    ids = tuple(sorted(sets_by_pose))
    sets = [frozenset(sets_by_pose[pose_id]) for pose_id in ids]
    n_poses = len(ids)
    values = np.full((n_poses, n_poses), np.nan)
    reasons: dict[tuple[str, str], str] = {}
    for i in range(n_poses):
        if sets[i]:
            values[i, i] = 1.0
        for j in range(i + 1, n_poses):
            union = sets[i] | sets[j]
            if not union:
                reasons[(ids[i], ids[j])] = empty_reason
                continue
            values[i, j] = values[j, i] = len(sets[i] & sets[j]) / len(union)
    return PairMatrix(name, None, "similarity", ids, values, reasons)


class InterfaceResidues(NamedTuple):
    """The interface of one pose.

    Attributes:
        binder: 1-based positions in the binder chain of the binder residues in the interface.
        receptor: Keys of the receptor residues in the interface (the key type of the caller;
            plain ``(chain, number, insertion code)`` tuples from ``interface_residues``).
    """

    binder: frozenset
    receptor: frozenset

    @property
    def combined(self) -> frozenset:
        """Both sides in one set, tagged ``("binder", position)`` or ``("receptor", key)``."""
        return frozenset(("binder", i) for i in self.binder) | frozenset(
            ("receptor", key) for key in self.receptor
        )


def interface_jaccard(interfaces: Mapping[str, InterfaceResidues]) -> dict[str, PairMatrix]:
    """Jaccard matrices of the interface residue sets: both sides, receptor only, binder only.

    Args:
        interfaces: ``{pose_id: InterfaceResidues}``; computing the interface (by buried area, by
            distance, ...) is the caller's job, and ``interface_residues`` is a simple helper.

    Returns:
        ``{"all": matrix, "receptor": matrix, "binder": matrix}``; a pair of two empty
        interfaces has no value, with the reason ``"both interfaces empty"``.
    """
    reason = "both interfaces empty"
    return {
        "all": jaccard_matrix(
            {k: v.combined for k, v in interfaces.items()},
            name="jaccard_interface",
            empty_reason=reason,
        ),
        "receptor": jaccard_matrix(
            {k: v.receptor for k, v in interfaces.items()},
            name="jaccard_interface_receptor",
            empty_reason=reason,
        ),
        "binder": jaccard_matrix(
            {k: v.binder for k, v in interfaces.items()},
            name="jaccard_interface_binder",
            empty_reason=reason,
        ),
    }


def contact_map_jaccard(contacts_by_pose: Mapping[str, Collection]) -> PairMatrix:
    """Jaccard index of the contact maps of the poses.

    A contact map is the set of residue pairs ``(binder position, receptor residue key)`` that
    have two heavy atoms closer than the contact cutoff (``heavy_atom_contacts``). Two poses
    with the same contacts score 1; a pair of two empty maps has no value.
    """
    return jaccard_matrix(
        contacts_by_pose, name="contact_map_jaccard", empty_reason="both contact maps empty"
    )


def _plain_key(chain: str, number: int, insertion_code: str) -> tuple[str, int, str]:
    """The default residue key of the contact helpers: ``(chain ID, number, insertion code)``."""
    return (chain, number, insertion_code)


def heavy_atom_contacts(
    atoms,
    binder_chain: str,
    *,
    target_chain: Optional[str] = None,
    cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_ANGSTROM,
    key_factory: Callable[[str, int, str], Any] = _plain_key,
) -> frozenset:
    """The residue pairs of the binder and the receptor with two heavy atoms closer than a cutoff.

    Args:
        atoms: A biotite ``AtomArray`` of one pose, in the common frame, with the receptor
            numbered as it should be compared across poses.
        binder_chain: The binder chain ID.
        target_chain: The receptor chain ID; None takes every atom outside the binder chain.
            Hydrogens and waters are never part of either side.
        cutoff_angstrom: Contact distance, default 5.0 (the heavy-atom contact distance of the
            CAPRI fraction of native contacts); a pair at exactly the cutoff is not in contact.
        key_factory: Builds the receptor residue key from ``(chain ID, number, insertion
            code)``; the default gives a plain tuple, ``ResidueKey`` gives the layer's key.

    Returns:
        A frozenset of ``(binder position, receptor key)``, the position being the 1-based index
        of the residue in the binder chain.

    Raises:
        ValueError: If the binder chain is missing or empty, or ``target_chain`` is not in the
            structure.
    """
    view = BinderView.from_atoms(atoms, binder_chain)
    chain_id = _strings(atoms.chain_id)
    if target_chain is None:
        in_target = chain_id != str(binder_chain)
    else:
        in_target = chain_id == str(target_chain)
        if not in_target.any():
            raise ValueError(f"no atom of chain {target_chain!r}")
    res_name = np.char.upper(_strings(atoms.res_name))
    rows = np.flatnonzero(
        in_target & ~_hydrogen_mask(atoms) & ~np.isin(res_name, sorted(WATER_NAMES_ALL))
    )
    if rows.size == 0:
        return frozenset()
    target = np.asarray(atoms.coord, dtype=np.float64)[rows]
    # only the receptor atoms near the binder can be in contact
    low, high = view.coord.min(axis=0) - cutoff_angstrom, view.coord.max(axis=0) + cutoff_angstrom
    near = np.all((target >= low) & (target <= high), axis=1)
    rows, target = rows[near], target[near]
    if rows.size == 0:
        return frozenset()
    res_id = np.asarray(atoms.res_id)[rows]
    ins_code = _insertion_codes(atoms)[rows]
    chains = chain_id[rows]
    contacts = set()
    limit = cutoff_angstrom**2
    for start in range(0, len(view.coord), 128):
        block = view.coord[start : start + 128]
        squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        for binder_row, target_row in np.argwhere(squared < limit):
            position = int(view.atom_residue[start + binder_row]) + 1
            key = key_factory(
                str(chains[target_row]), int(res_id[target_row]), str(ins_code[target_row])
            )
            contacts.add((position, key))
    return frozenset(contacts)


def interface_residues(
    atoms,
    binder_chain: str,
    *,
    target_chain: Optional[str] = None,
    cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_ANGSTROM,
    key_factory: Callable[[str, int, str], Any] = _plain_key,
) -> InterfaceResidues:
    """The residues of the binder and of the receptor that touch the other partner.

    A simple distance rule, not the buried-area definition of the interface metrics: a residue
    is in the interface when one of its heavy atoms is closer than ``cutoff_angstrom`` (default
    5.0) to a heavy atom of the other partner. Computing the interface by another rule and
    passing it as ``InterfaceResidues`` is the caller's choice. The arguments are those of
    ``heavy_atom_contacts``.
    """
    contacts = heavy_atom_contacts(
        atoms,
        binder_chain,
        target_chain=target_chain,
        cutoff_angstrom=cutoff_angstrom,
        key_factory=key_factory,
    )
    return InterfaceResidues(
        binder=frozenset(position for position, _ in contacts),
        receptor=frozenset(key for _, key in contacts),
    )


# --------------------------------------------------------------------------- positional spread


@dataclass(frozen=True)
class PositionSpread:
    """The spread of the alpha carbons of the poses at one binder position.

    Attributes:
        binder_index: 1-based position in the binder chain of the reference pose.
        chain, number, insertion_code, residue: The reference pose's residue at that position.
        n_poses: Poses with an alpha carbon at the position.
        spread_angstrom: ``sqrt(mean_i |x_i - mean(x)|^2)``; None with a ``reason`` when fewer
            than two poses have an alpha carbon there.
        reason: Why ``spread_angstrom`` is None.
    """

    binder_index: int
    chain: str
    number: int
    insertion_code: str
    residue: str
    n_poses: int
    spread_angstrom: Optional[float]
    reason: Optional[str] = None


@dataclass(frozen=True)
class PositionalSpread:
    """Per-position spread of the alpha carbons of a set of poses.

    Attributes:
        pose_ids: The poses that were used, sorted.
        reference_id: The pose whose residues name the positions (the first usable id unless
            one was given); the other poses are paired with it by residue key or position.
        positions: One ``PositionSpread`` per residue of the reference pose that has an alpha
            carbon; empty when the spread cannot be computed.
        mean_spread_angstrom: Mean of the spreads that exist; None with a ``reason`` otherwise.
        excluded: ``{pose_id: reason}`` for the poses that could not be used.
        reason: Why the whole result is empty or None.
    """

    pose_ids: tuple[str, ...]
    reference_id: Optional[str]
    positions: tuple[PositionSpread, ...]
    mean_spread_angstrom: Optional[float]
    excluded: dict[str, str]
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary; ``reason`` keys appear only where a value is None."""
        out: dict[str, Any] = {
            "pose_ids": list(self.pose_ids),
            "reference_id": self.reference_id,
            "positions": [
                {
                    k: v
                    for k, v in dataclasses.asdict(p).items()
                    if not (k == "reason" and v is None)
                }
                for p in self.positions
            ],
            "mean_spread_angstrom": self.mean_spread_angstrom,
            "excluded": dict(self.excluded),
        }
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def positional_spread(
    poses,
    binder_chain: str,
    *,
    residue_pairing: str = "auto",
    reference_id: Optional[str] = None,
) -> PositionalSpread:
    """How far the alpha carbons of the poses scatter around their mean, per binder position.

    For each residue of the reference pose that has an alpha carbon, ``spread = sqrt(mean_i
    |x_i - mean(x)|^2)`` over the poses that have an alpha carbon at the paired residue (divisor
    n). The poses are taken as they are in the common frame. The residues of the other poses
    are paired with those of the reference pose by ``residue_pairing`` (see the module
    docstring); a pose that cannot be paired is listed in ``excluded``. The cyclic shift is not
    applied: a ring with another origin is excluded or paired by position.

    Call it once per group (for example per back-end) for the spread inside a group.

    Args:
        poses: ``{pose_id: atoms}`` or ``(pose_id, atoms)`` pairs, in the common frame.
        binder_chain: The binder chain ID.
        residue_pairing: ``"auto"``, ``"key"`` or ``"position"``.
        reference_id: The pose that names the positions; the first usable id by default.

    Returns:
        A ``PositionalSpread``. With fewer than two usable poses it has no positions and a
        reason.

    Raises:
        ValueError: For an unknown ``residue_pairing`` or a ``reference_id`` that is not a pose.
    """
    _check_choice("residue_pairing", residue_pairing, RESIDUE_PAIRINGS)
    items = _items(poses)
    views: dict[str, BinderView] = {}
    excluded: dict[str, str] = {}
    for pose_id, atoms in items:
        try:
            views[pose_id] = BinderView.from_atoms(atoms, binder_chain, pose_id)
        except ValueError as error:
            excluded[pose_id] = f"pose {pose_id!r}: {error}"
    if reference_id is not None and reference_id not in {pose_id for pose_id, _ in items}:
        raise ValueError(f"reference_id {reference_id!r} is not one of the poses")
    if reference_id is not None and reference_id in excluded:
        raise ValueError(excluded[reference_id])
    reference = reference_id or next(iter(views), None)
    usable: dict[str, dict[int, np.ndarray]] = {}
    if reference is not None:
        ref_view = views[reference]
        for pose_id, view in views.items():
            if pose_id == reference:
                pairs = [(i, i) for i in range(len(ref_view.residues))]
            else:
                pairs, _, why = _pair_residues(ref_view, view, residue_pairing)
                if pairs is None:
                    excluded[pose_id] = why
                    continue
            usable[pose_id] = {
                i: view.coord[view.residues[j].atom_rows["CA"]]
                for i, j in pairs
                if "CA" in view.residues[j].atom_rows
            }
    pose_ids = tuple(sorted(usable))
    if len(pose_ids) < 2:
        return PositionalSpread(
            pose_ids,
            reference,
            (),
            None,
            excluded,
            f"needs at least 2 usable poses (got {len(pose_ids)})",
        )
    positions = []
    for index, residue in enumerate(views[reference].residues):
        if "CA" not in residue.atom_rows:
            continue
        coords = np.array([usable[p][index] for p in pose_ids if index in usable[p]])
        spread: Optional[float] = None
        why = None
        if len(coords) >= 2:
            spread = float(np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0)) ** 2, axis=1))))
        else:
            why = f"needs at least 2 poses with an alpha carbon here (got {len(coords)})"
        positions.append(
            PositionSpread(
                index + 1,
                residue.key[0],
                residue.key[1],
                residue.key[2],
                residue.name,
                len(coords),
                spread,
                why,
            )
        )
    defined = [p.spread_angstrom for p in positions if p.spread_angstrom is not None]
    mean = float(np.mean(defined)) if defined else None
    return PositionalSpread(
        pose_ids,
        reference,
        tuple(positions),
        mean,
        excluded,
        None if defined else "no position has two poses with an alpha carbon",
    )


def positional_spread_by_group(
    poses,
    binder_chain: str,
    groups: Mapping[str, str],
    *,
    residue_pairing: str = "auto",
) -> dict[str, PositionalSpread]:
    """``positional_spread`` of each group of poses.

    Args:
        poses: ``{pose_id: atoms}`` or ``(pose_id, atoms)`` pairs.
        binder_chain: The binder chain ID.
        groups: ``{pose_id: group label}``; every pose needs one.
        residue_pairing: As in ``positional_spread``.

    Returns:
        ``{group: PositionalSpread}`` in sorted group order.

    Raises:
        ValueError: If a pose has no group label.
    """
    items = _items(poses)
    missing = [pose_id for pose_id, _ in items if pose_id not in groups]
    if missing:
        raise ValueError(f"poses without a group label: {missing}")
    out = {}
    for name in sorted({groups[pose_id] for pose_id, _ in items}):
        members = [(pose_id, atoms) for pose_id, atoms in items if groups[pose_id] == name]
        out[name] = positional_spread(members, binder_chain, residue_pairing=residue_pairing)
    return out
