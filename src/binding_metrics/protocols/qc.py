"""Structural QC for relaxed structures.

A minimization can report ``success=True`` with a finite energy while having
blown up the geometry: exploded coordinates, fused atoms, an inverted
stereocentre, NaNs. :func:`check_relaxed_structure` runs seven checks on a
structure before and after relaxation so that a normal-looking result is not
mistaken for a sound one:

1. ``energy``: finite, inside a sane range, and not higher than before
   minimization.
2. ``rmsd``: heavy-atom RMSD to the input stays below a bound.
3. ``coordinates_finite``: no NaN or inf coordinate.
4. ``min_heavy_distance``: no two heavy atoms of different residues overlap.
5. ``bond_lengths``: no covalent bond of the topology was stretched or broken.
6. ``chirality``: no C-alpha stereocentre inverted (matters for D-amino acids).
7. ``composition``: no heavy atom was added, dropped or renamed.

The same functions run inside :class:`~binding_metrics.protocols.relaxation.ImplicitRelaxation`
(where they only annotate the result) and in ``tests/test_structural_qc.py``.

Structures are compared through :class:`AtomSnapshot`, which holds coordinates
plus the identifiers needed to pair atoms between the two structures. Both
snapshots of one comparison must come from the same loader
(:meth:`AtomSnapshot.from_topology` or :meth:`AtomSnapshot.from_file`), because
atoms are paired by residue key and atom name.

The ``bond_lengths`` check takes its bond list from the topology, never from
the geometry: a snapshot built from an OpenMM topology carries the topology
bonds, and one built from a file carries the bonds of the residue templates in
the wwPDB Chemical Component Dictionary (through biotite). Perceiving bonds
from the input distances would count every clash between atoms that PDBFixer
rebuilt as a bond, and a relaxation that resolves the clash would then look
like a stretched bond. The bonds are measured in the relaxed structure only.

Return schema of every ``check_*`` function::

    {
        "passed": bool,        # False only when the check found a problem
        "evaluated": bool,     # False when there was nothing to evaluate
        "value": float | None, # the headline number the limit applies to
        "limit": str,          # the acceptance rule in words and units
        "detail": str,         # one line for a log or an assertion message
        "reason": str,         # only when evaluated is False
    }

and of :func:`check_relaxed_structure`::

    {"passed": bool, "failed": [check names], "checks": {name: check dict}}

Measured values on the bundled examples (short minimizations on CUDA)::

                 energy(min)   energy(pre)   heavy RMSD   min inter-res dist
    1YCR        -14460 kJ/mol  +5082         0.409 A      1.330 A
    3P8F        -33108 kJ/mol -27312         0.293 A      1.329 A
    cyclosporin -21515 kJ/mol   n/a          0.297 A      1.327 A

                min bond   max bond   C-alpha centres   heavy atoms
    1YCR        1.218 A    1.822 A     94          819
    3P8F        1.217 A    2.052 A    225          1970
    cyclosporin 1.219 A    1.821 A    152          1351

The longest topology bonds are the methionine C-S bond (1.82 A) and disulfides
(2.05 A), so the 2.5 A limit keeps a margin of 0.45 A over any real bond.

The limits below are wide on purpose: each passes any genuinely relaxed
structure and fails an exploded one. They can shift across GPU models and
force-field versions, so do not tighten them to the numbers above.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from binding_metrics.core.residues import WATER_NAMES_WITH_H2O

logger = logging.getLogger(__name__)

# --- Limits -----------------------------------------------------------------

#: Blown-up structures show |E| far above 1e6 kJ/mol (or inf).
ENERGY_MAX_KJ_MOL = 1.0e6
#: Sane lower bound for a small implicit-solvent system.
ENERGY_MIN_KJ_MOL = -1.0e8
#: Minimization only lowers the energy. This slack absorbs mixed-precision
#: noise when the input was already at a minimum.
ENERGY_INCREASE_TOLERANCE_KJ_MOL = 1.0

#: Heavy-atom RMSD to the input above this means the structure exploded; a
#: short minimization moves atoms by well under 1 A.
RMSD_MAX_ANGSTROM = 5.0

#: Closest pair of heavy atoms in different residues. The tightest legitimate
#: contacts are the peptide bond C-N at about 1.33 A and the disulfide S-S at
#: about 2.05 A. Only same-residue pairs are excluded, so bonded neighbours are
#: included; 0.8 A sits below every real bond and above fused atoms.
MIN_HEAVY_DISTANCE_ANGSTROM = 0.8

#: Bonds between residues of a file snapshot (peptide C-N, disulfide S-S) exist
#: only when the two atoms are inside this window in the file: C-N is about
#: 1.33 A and S-S 2.05 A (Engh and Huber, Acta Cryst. A47, 392, 1991), while a
#: chain break leaves C and N several angstrom apart. Bonds inside a residue
#: never use the window; they come from the residue template.
BOND_PERCEIVE_MIN_ANGSTROM = 0.9
BOND_PERCEIVE_MAX_ANGSTROM = 2.1
#: Allowed length of a topology bond after relaxation: wide enough for any
#: real heavy-heavy bond with margin, tight enough to fail a stretched or
#: broken one.
BOND_LENGTH_MIN_ANGSTROM = 0.5
BOND_LENGTH_MAX_ANGSTROM = 2.5

#: A real C-alpha stereocentre has a signed tetrahedral volume of about
#: 2.2-2.6 A^3 in magnitude. An inversion swings the sign between two such
#: values. Only a near-planar centre (|V| near 0) has a sign that
#: mixed-precision noise can tip, so a sign change counts as an inversion only
#: when both volumes exceed this magnitude.
CHIRALITY_MIN_VOLUME_ANGSTROM3 = 0.5

#: Residue names excluded from the heavy-atom checks.
WATER_NAMES = WATER_NAMES_WITH_H2O

#: Rows of the pairwise distance matrix computed at once; bounds memory to
#: about ``_CHUNK_ROWS * n_atoms * 8`` bytes.
_CHUNK_ROWS = 128

PathLike = Union[str, Path]


# --- Snapshot ---------------------------------------------------------------


@dataclass(frozen=True)
class AtomSnapshot:
    """Coordinates and identifiers of every atom of one structure.

    Attributes:
        coords: (n, 3) array in angstrom, all atoms including hydrogens.
        atom_names: Atom name per atom.
        residue_keys: Hashable key per atom, unique per residue and equal for
            the same residue in two structures of one comparison.
        is_hydrogen: True for hydrogen and deuterium atoms.
        is_water: True for atoms of water residues.
        bonds: Covalent bonds between heavy atoms as ``(row, row)`` pairs, taken
            from the topology or the residue templates, never from the
            geometry. ``None`` when no bond source was available; the
            ``bond_lengths`` check is then not evaluated.
    """

    coords: np.ndarray
    atom_names: tuple
    residue_keys: tuple
    is_hydrogen: np.ndarray
    is_water: np.ndarray
    bonds: Optional[tuple] = None

    @classmethod
    def from_topology(cls, topology, positions) -> "AtomSnapshot":
        """Snapshot an OpenMM topology with its positions (nm Quantity or array)."""
        if hasattr(positions, "value_in_unit"):
            from openmm import unit

            xyz_nm = np.array(positions.value_in_unit(unit.nanometer), dtype=float)
        else:
            xyz_nm = np.asarray(positions, dtype=float)

        n_atoms = topology.getNumAtoms()
        if n_atoms != len(xyz_nm):
            raise ValueError(f"topology has {n_atoms} atoms but positions has {len(xyz_nm)} rows")
        names = [""] * n_atoms
        keys: list = [None] * n_atoms
        hydrogen = np.zeros(n_atoms, dtype=bool)
        water = np.zeros(n_atoms, dtype=bool)
        seen: dict = {}
        for residue in topology.residues():
            base = (residue.chain.id, str(residue.id))
            occurrence = seen.get(base, 0)
            seen[base] = occurrence + 1
            key = base + (occurrence,)
            is_water = residue.name in WATER_NAMES
            for atom in residue.atoms():
                # ``positions`` is indexed by atom.index.
                names[atom.index] = atom.name
                keys[atom.index] = key
                hydrogen[atom.index] = atom.element is not None and atom.element.symbol in (
                    "H",
                    "D",
                )
                water[atom.index] = is_water
        heavy = ~hydrogen & ~water
        bonds = sorted(
            {
                (min(bond.atom1.index, bond.atom2.index), max(bond.atom1.index, bond.atom2.index))
                for bond in topology.bonds()
                if heavy[bond.atom1.index] and heavy[bond.atom2.index]
            }
        )
        return cls(
            coords=xyz_nm * 10.0,
            atom_names=tuple(names),
            residue_keys=tuple(keys),
            is_hydrogen=hydrogen,
            is_water=water,
            bonds=tuple(bonds),
        )

    @classmethod
    def from_file(cls, path: PathLike) -> "AtomSnapshot":
        """Snapshot the first model of a CIF or PDB file (needs gemmi).

        Residues are keyed by ``(chain, sequence number, insertion code,
        occurrence)``. The occurrence counter is needed because a prepared file
        can repeat a sequence number within one chain, and both structures of a
        comparison list their residues in the same order.

        A file has no topology, so ``bonds`` come from residue templates (see
        :func:`_template_bonds`) and stay ``None`` when biotite is missing.
        Bonds that only a topology knows, such as a cyclization or a staple
        between residues, are not in the list.
        """
        try:
            import gemmi
        except ImportError as exc:
            raise ImportError("gemmi is required to read structure files for QC") from exc

        model = gemmi.read_structure(str(path))[0]
        coords, names, keys, hydrogen, water = [], [], [], [], []
        residues: list = []
        seen: dict = {}
        for chain_index, chain in enumerate(model):
            for residue in chain:
                base = (chain.name, residue.seqid.num, residue.seqid.icode)
                occurrence = seen.get(base, 0)
                seen[base] = occurrence + 1
                key = base + (occurrence,)
                is_water = residue.name in WATER_NAMES
                heavy_rows: dict = {}
                for atom in residue:
                    is_hydrogen = atom.element.name in ("H", "D")
                    if not (is_hydrogen or is_water):
                        heavy_rows[atom.name] = len(coords)
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    names.append(atom.name)
                    keys.append(key)
                    hydrogen.append(is_hydrogen)
                    water.append(is_water)
                if heavy_rows:
                    residues.append((chain_index, residue.name, heavy_rows))
        xyz = np.array(coords, dtype=float).reshape(-1, 3)
        return cls(
            coords=xyz,
            atom_names=tuple(names),
            residue_keys=tuple(keys),
            is_hydrogen=np.array(hydrogen, dtype=bool),
            is_water=np.array(water, dtype=bool),
            bonds=_template_bonds(residues, xyz),
        )

    @property
    def heavy_mask(self) -> np.ndarray:
        """Non-hydrogen atoms of non-water residues."""
        return ~self.is_hydrogen & ~self.is_water


Structure = Union[AtomSnapshot, PathLike]


def _as_snapshot(structure: Structure) -> AtomSnapshot:
    return structure if isinstance(structure, AtomSnapshot) else AtomSnapshot.from_file(structure)


# --- Shared helpers ---------------------------------------------------------


def _result(
    passed: bool,
    value: Optional[float],
    limit: str,
    detail: str,
    *,
    evaluated: bool = True,
    reason: Optional[str] = None,
) -> dict:
    out = {
        "passed": bool(passed),
        "evaluated": bool(evaluated),
        "value": None if value is None else float(value),
        "limit": limit,
        "detail": detail,
    }
    if reason is not None:
        out["reason"] = reason
    return out


def _format_residue(key: tuple) -> str:
    """Readable residue label from a key: ``chain:number`` plus insertion code."""
    label = f"{key[0]}:{key[1]}"
    if len(key) == 4 and str(key[2]).strip():
        label += str(key[2]).strip()
    return label


#: Force-field names of protonation variants. The Chemical Component Dictionary
#: reads HIE, HID and the like as unrelated ligands, so they are looked up under
#: the residue they are a variant of; the heavy-atom bonds are those of the parent.
_TEMPLATE_ALIASES = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
}


def _template_bonds(residues: list, coords: np.ndarray) -> Optional[tuple]:
    """Heavy-atom bonds of a structure file, from residue templates.

    Bonds inside a residue are the ones the wwPDB Chemical Component Dictionary
    lists for its name (through biotite), restricted to the atoms present. Three
    kinds of bond between residues are added, each only when the two atoms are
    inside the covalent window of the file: the peptide bond from C of one
    residue (an acetyl cap included) to N of the next in the same chain, which
    a chain break leaves out; the head-to-tail bond from C of the last residue
    of a chain to N of its first; and disulfides between SG atoms.

    Args:
        residues: ``(chain index, residue name, {heavy atom name: row})`` per
            non-water residue, in file order.
        coords: (n, 3) coordinates the rows index.

    Returns:
        Sorted ``(row, row)`` pairs, or ``None`` when biotite is not installed.
    """
    try:
        from biotite.structure.info import bonds_in_residue
    except ImportError:
        logger.warning("biotite is not installed: no bond list for the bond_lengths check")
        return None

    def in_window(row_a: int, row_b: int) -> bool:
        length = float(np.linalg.norm(coords[row_a] - coords[row_b]))
        return BOND_PERCEIVE_MIN_ANGSTROM <= length <= BOND_PERCEIVE_MAX_ANGSTROM

    def is_amino_acid(rows: dict) -> bool:
        return {"N", "CA", "C"} <= rows.keys()

    bonds: set = set()
    sulfurs: list = []
    chain_ends: dict = {}
    previous: Optional[tuple] = None
    for chain_index, name, rows in residues:
        for atom_a, atom_b in bonds_in_residue(_TEMPLATE_ALIASES.get(name, name)):
            if atom_a in rows and atom_b in rows:
                bonds.add(tuple(sorted((rows[atom_a], rows[atom_b]))))
        if (
            previous is not None
            and previous[0] == chain_index
            and "C" in previous[1]
            and "N" in rows
            and in_window(previous[1]["C"], rows["N"])
        ):
            bonds.add((previous[1]["C"], rows["N"]))
        first, _ = chain_ends.get(chain_index, (rows, rows))
        chain_ends[chain_index] = (first, rows)
        if "SG" in rows:
            sulfurs.append(rows["SG"])
        previous = (chain_index, rows)
    for first, last in chain_ends.values():
        if first is not last and is_amino_acid(first) and is_amino_acid(last):
            if in_window(last["C"], first["N"]):
                bonds.add(tuple(sorted((last["C"], first["N"]))))
    for k, row_a in enumerate(sulfurs):
        bonds.update((row_a, row_b) for row_b in sulfurs[k + 1 :] if in_window(row_a, row_b))
    return tuple(sorted(bonds))


def _skipped(limit: str, reason: str) -> dict:
    """A check with nothing to evaluate: it does not flag a problem."""
    return _result(True, None, limit, reason, evaluated=False, reason=reason)


def _heavy_atom_index(snapshot: AtomSnapshot) -> dict:
    """Map ``(residue key, atom name)`` to the row of that heavy atom."""
    return {
        (snapshot.residue_keys[i], snapshot.atom_names[i]): i
        for i in np.nonzero(snapshot.heavy_mask)[0]
    }


def _distance_block(block: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Distances between every row of ``block`` and every row of ``coords``.

    Accumulated one axis at a time, which keeps the temporaries at
    ``len(block) * len(coords)`` values instead of three times that.
    """
    squared = np.zeros((len(block), len(coords)))
    for axis in range(3):
        squared += (block[:, axis, None] - coords[None, :, axis]) ** 2
    return np.sqrt(squared)


def _kabsch_rmsd(moving: np.ndarray, fixed: np.ndarray) -> float:
    """RMSD after optimal superposition (Kabsch, Acta Cryst. A32, 922, 1976).

    ``H = P^T Q = U S V^T`` gives the rotation ``R = V U^T`` for column vectors;
    the coordinates are rows, so the rotated set is ``P @ R.T``.
    """
    p = moving - moving.mean(axis=0)
    q = fixed - fixed.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    return float(np.sqrt(np.mean(np.sum((p @ rot.T - q) ** 2, axis=1))))


# --- The seven checks -------------------------------------------------------


def check_energy(
    energy_kj_mol: Optional[float], energy_before_kj_mol: Optional[float] = None
) -> dict:
    """Check 1: the energy is finite, in range and did not rise.

    Args:
        energy_kj_mol: Potential energy after relaxation.
        energy_before_kj_mol: Potential energy before minimization. When given,
            the energy must not exceed it by more than
            ``ENERGY_INCREASE_TOLERANCE_KJ_MOL``. Leave it out for MD, whose
            mean energy is legitimately above a minimum.
    """
    limit = f"{ENERGY_MIN_KJ_MOL:g} < E < {ENERGY_MAX_KJ_MOL:g} kJ/mol"
    if energy_kj_mol is None:
        return _skipped(limit, "no energy available")
    if not math.isfinite(energy_kj_mol):
        return _result(False, energy_kj_mol, limit, f"non-finite energy {energy_kj_mol}")
    if not ENERGY_MIN_KJ_MOL < energy_kj_mol < ENERGY_MAX_KJ_MOL:
        return _result(
            False,
            energy_kj_mol,
            limit,
            f"energy {energy_kj_mol:.1f} kJ/mol outside ({ENERGY_MIN_KJ_MOL:g}, "
            f"{ENERGY_MAX_KJ_MOL:g}): structure likely exploded",
        )
    if energy_before_kj_mol is not None:
        if not math.isfinite(energy_before_kj_mol):
            return _result(
                False, energy_kj_mol, limit, f"non-finite energy before: {energy_before_kj_mol}"
            )
        if energy_kj_mol > energy_before_kj_mol + ENERGY_INCREASE_TOLERANCE_KJ_MOL:
            return _result(
                False,
                energy_kj_mol,
                limit + " and not above the energy before minimization",
                f"energy {energy_kj_mol:.1f} kJ/mol exceeds the {energy_before_kj_mol:.1f} "
                "kJ/mol before minimization",
            )
    return _result(True, energy_kj_mol, limit, f"energy {energy_kj_mol:.1f} kJ/mol")


def check_rmsd(
    before: Structure, after: Structure, max_rmsd_angstrom: float = RMSD_MAX_ANGSTROM
) -> dict:
    """Check 2: heavy-atom RMSD between the two structures (superposed) is bounded."""
    limit = f"heavy-atom RMSD < {max_rmsd_angstrom:g} A"
    b, a = _as_snapshot(before), _as_snapshot(after)
    index_b, index_a = _heavy_atom_index(b), _heavy_atom_index(a)
    shared = sorted(set(index_b) & set(index_a), key=index_b.get)
    if len(shared) < 3:
        return _skipped(limit, f"only {len(shared)} heavy atoms in common")
    xyz_b = b.coords[[index_b[k] for k in shared]]
    xyz_a = a.coords[[index_a[k] for k in shared]]
    if not (np.isfinite(xyz_b).all() and np.isfinite(xyz_a).all()):
        return _result(False, math.nan, limit, "non-finite coordinates, RMSD undefined")
    rmsd = _kabsch_rmsd(xyz_b, xyz_a)
    passed = rmsd < max_rmsd_angstrom
    return _result(
        passed,
        rmsd,
        limit,
        f"heavy-atom RMSD {rmsd:.3f} A over {len(shared)} atoms"
        + ("" if passed else f" (limit {max_rmsd_angstrom:g} A)"),
    )


def check_coordinates_finite(after: Structure) -> dict:
    """Check 3: no NaN or inf coordinate among all atoms."""
    limit = "all coordinates finite"
    snap = _as_snapshot(after)
    bad = int((~np.isfinite(snap.coords).all(axis=1)).sum())
    return _result(
        bad == 0,
        bad,
        limit,
        f"{bad} atom(s) with non-finite coordinates" if bad else "all coordinates finite",
    )


def check_min_heavy_distance(after: Structure) -> dict:
    """Check 4: no heavy atoms of different residues are closer than 0.8 A."""
    limit = f"closest inter-residue heavy-atom pair > {MIN_HEAVY_DISTANCE_ANGSTROM:g} A"
    snap = _as_snapshot(after)
    heavy = np.nonzero(snap.heavy_mask)[0]
    if len(heavy) == 0:
        return _skipped(limit, "no heavy atoms")
    xyz = snap.coords[heavy]
    if not np.isfinite(xyz).all():
        return _skipped(limit, "non-finite coordinates")
    residue_code = {}
    codes = np.array(
        [residue_code.setdefault(snap.residue_keys[i], len(residue_code)) for i in heavy]
    )
    closest = math.inf
    for start in range(0, len(xyz), _CHUNK_ROWS):
        dist = _distance_block(xyz[start : start + _CHUNK_ROWS], xyz)
        dist[codes[start : start + _CHUNK_ROWS, None] == codes[None, :]] = np.inf
        closest = min(closest, float(dist.min()))
    if not math.isfinite(closest):
        return _skipped(limit, "single residue: no inter-residue pair")
    passed = closest > MIN_HEAVY_DISTANCE_ANGSTROM
    return _result(
        passed,
        closest,
        limit,
        f"closest inter-residue heavy pair {closest:.3f} A"
        + ("" if passed else ": fused or overlapping atoms indicate an explosion"),
    )


def _format_atom(residue_key: tuple, atom_name: str) -> str:
    return f"{_format_residue(residue_key)}:{atom_name}"


def check_bond_lengths(before: Structure, after: Structure) -> dict:
    """Check 5: no covalent bond of the topology is stretched or broken in ``after``.

    The bond list is ``before.bonds`` (the topology bonds of an OpenMM
    snapshot, or the residue-template bonds of a file snapshot) and the lengths
    are measured in ``after`` only. The lengths the same bonds have in
    ``before`` are informative: a bond that was already longer than the limit
    there (a chain break the force field then closes, an atom PDBFixer placed
    badly) is named in the detail but does not fail the check, because the
    relaxation is judged on the structure it returns.
    """
    limit = f"topology bonds within [{BOND_LENGTH_MIN_ANGSTROM:g}, {BOND_LENGTH_MAX_ANGSTROM:g}] A"
    b, a = _as_snapshot(before), _as_snapshot(after)
    if not b.bonds:
        return _skipped(
            limit,
            "the input snapshot has no bonds (needs a topology with bonds, or biotite for a file)",
        )
    index_a = _heavy_atom_index(a)

    # Atom identity is (residue key, atom name); a bond whose atoms are absent
    # from ``after`` is left to the composition check.
    ends_b: list = []
    ends_a: list = []
    for i, j in b.bonds:
        id_i = (b.residue_keys[i], b.atom_names[i])
        id_j = (b.residue_keys[j], b.atom_names[j])
        if id_i in index_a and id_j in index_a:
            ends_b.append((i, j))
            ends_a.append((index_a[id_i], index_a[id_j]))
    if not ends_a:
        return _skipped(limit, "no bond of the input found in the relaxed structure")

    pairs_b, pairs_a = np.array(ends_b), np.array(ends_a)
    lengths = np.linalg.norm(a.coords[pairs_a[:, 0]] - a.coords[pairs_a[:, 1]], axis=1)
    lengths_before = np.linalg.norm(b.coords[pairs_b[:, 0]] - b.coords[pairs_b[:, 1]], axis=1)
    # Distance outside the window; NaN lengths count as infinitely far out.
    excess = np.where(
        np.isfinite(lengths),
        np.maximum(BOND_LENGTH_MIN_ANGSTROM - lengths, lengths - BOND_LENGTH_MAX_ANGSTROM),
        np.inf,
    )
    n_bad = int((excess > 0).sum())
    finite = lengths[np.isfinite(lengths)]
    longest = float(finite.max()) if len(finite) else math.nan

    def label(k: int) -> str:
        i, j = pairs_b[k]
        return (
            f"{_format_atom(b.residue_keys[i], b.atom_names[i])}-"
            f"{_format_atom(b.residue_keys[j], b.atom_names[j])}"
        )

    if n_bad:
        worst = int(np.argmax(excess))
        detail = (
            f"{n_bad} of {len(lengths)} bonds outside the limit, worst {label(worst)} "
            f"= {lengths[worst]:.2f} A"
        )
    else:
        detail = f"{len(lengths)} bonds, lengths {finite.min():.3f}-{longest:.3f} A"
    long_before = np.nonzero(lengths_before > BOND_LENGTH_MAX_ANGSTROM)[0]
    if len(long_before):
        widest = int(long_before[np.argmax(lengths_before[long_before])])
        detail += (
            f"; {len(long_before)} bonds were already longer than {BOND_LENGTH_MAX_ANGSTROM:g} A "
            f"in the input, longest {label(widest)} = {lengths_before[widest]:.2f} A"
        )
    return _result(n_bad == 0, longest, limit, detail)


def _ca_signed_volumes(snapshot: AtomSnapshot) -> dict:
    """Signed tetrahedral volume ``(N-CA) . ((C-CA) x (CB-CA))`` per residue.

    Its sign encodes the C-alpha handedness. Residues without all four of N,
    CA, C and CB (glycine, sarcosine) are skipped.
    """
    wanted = {"N", "CA", "C", "CB"}
    per_residue: dict = {}
    for i in np.nonzero(~snapshot.is_hydrogen)[0]:
        name = snapshot.atom_names[i]
        if name in wanted:
            per_residue.setdefault(snapshot.residue_keys[i], {})[name] = snapshot.coords[i]
    volumes = {}
    for key, atoms in per_residue.items():
        if len(atoms) < 4:
            continue
        ca = atoms["CA"]
        volumes[key] = float(np.dot(atoms["N"] - ca, np.cross(atoms["C"] - ca, atoms["CB"] - ca)))
    return volumes


def check_chirality(before: Structure, after: Structure) -> dict:
    """Check 6: no C-alpha stereocentre changed handedness.

    Compares the sign of the signed tetrahedral volume at each C-alpha between
    the two structures. The sign is compared, never the absolute L or D
    configuration, so D-amino acids need no special handling.
    """
    limit = f"no sign change of C-alpha volume (|V| >= {CHIRALITY_MIN_VOLUME_ANGSTROM3:g} A^3)"
    vol_b, vol_a = _ca_signed_volumes(_as_snapshot(before)), _ca_signed_volumes(_as_snapshot(after))
    shared = set(vol_b) & set(vol_a)
    if not shared:
        return _skipped(limit, "no residue with N, CA, C and CB in both structures")
    flipped = sorted(
        (k, round(vol_b[k], 2), round(vol_a[k], 2))
        for k in shared
        if (vol_b[k] > 0) != (vol_a[k] > 0)
        and abs(vol_b[k]) >= CHIRALITY_MIN_VOLUME_ANGSTROM3
        and abs(vol_a[k]) >= CHIRALITY_MIN_VOLUME_ANGSTROM3
    )
    detail = (
        f"{len(flipped)} of {len(shared)} C-alpha centres inverted, e.g. residue "
        f"{_format_residue(flipped[0][0])}: volume {flipped[0][1]} -> {flipped[0][2]} A^3"
        if flipped
        else f"{len(shared)} C-alpha centres, none inverted"
    )
    return _result(not flipped, len(flipped), limit, detail)


def _heavy_composition(snapshot: AtomSnapshot) -> dict:
    composition: dict = {}
    for i in np.nonzero(snapshot.heavy_mask)[0]:
        composition.setdefault(snapshot.residue_keys[i], Counter())[snapshot.atom_names[i]] += 1
    return composition


def check_composition(before: Structure, after: Structure) -> dict:
    """Check 7: same residues and same heavy-atom names per residue."""
    limit = "identical heavy-atom composition per residue"
    comp_b = _heavy_composition(_as_snapshot(before))
    comp_a = _heavy_composition(_as_snapshot(after))
    if not comp_b:
        return _skipped(limit, "no heavy atoms before relaxation")
    n_b, n_a = (
        sum(sum(c.values()) for c in comp_b.values()),
        sum(sum(c.values()) for c in comp_a.values()),
    )
    added = sorted(set(comp_a) - set(comp_b))
    dropped = sorted(set(comp_b) - set(comp_a))
    changed = sorted(k for k in set(comp_b) & set(comp_a) if comp_b[k] != comp_a[k])
    passed = n_b == n_a and not (added or dropped or changed)

    def _labels(keys: list) -> list:
        return [_format_residue(k) for k in keys[:3]]

    detail = (
        f"{n_a} heavy atoms in {len(comp_a)} residues, unchanged"
        if passed
        else f"heavy atoms {n_b} -> {n_a}; residues added {_labels(added)}, "
        f"dropped {_labels(dropped)}, changed {_labels(changed)}"
    )
    return _result(passed, n_a - n_b, limit, detail)


# --- Entry point ------------------------------------------------------------


def check_relaxed_structure(
    before: Structure,
    after: Structure,
    *,
    energy_kj_mol: Optional[float] = None,
    energy_before_kj_mol: Optional[float] = None,
    max_rmsd_angstrom: Optional[float] = RMSD_MAX_ANGSTROM,
) -> dict:
    """Run the seven structural checks on a relaxed structure.

    Args:
        before: Structure handed to the relaxation (an :class:`AtomSnapshot` or
            a path to a CIF or PDB file).
        after: Relaxed structure, in the same form as ``before``.
        energy_kj_mol: Potential energy after relaxation, for the ``energy``
            check. Omit it and that check is reported as not evaluated.
        energy_before_kj_mol: Potential energy before minimization.
        max_rmsd_angstrom: RMSD bound for the ``rmsd`` check. Pass ``None`` to
            leave that check out, as for an MD frame, which legitimately moves
            away from its start.

    Returns:
        ``{"passed": bool, "failed": [names], "checks": {name: check dict}}``.
        ``passed`` is False if any evaluated check found a problem. See the
        module docstring for the check schema. The result is advisory:
        callers decide what to do with it.
    """
    snap_before, snap_after = _as_snapshot(before), _as_snapshot(after)
    checks = {"energy": check_energy(energy_kj_mol, energy_before_kj_mol)}
    if max_rmsd_angstrom is not None:
        checks["rmsd"] = check_rmsd(snap_before, snap_after, max_rmsd_angstrom)
    checks["coordinates_finite"] = check_coordinates_finite(snap_after)
    checks["min_heavy_distance"] = check_min_heavy_distance(snap_after)
    checks["bond_lengths"] = check_bond_lengths(snap_before, snap_after)
    checks["chirality"] = check_chirality(snap_before, snap_after)
    checks["composition"] = check_composition(snap_before, snap_after)
    failed = [name for name, check in checks.items() if not check["passed"]]
    for name in failed:
        logger.warning("structural QC: %s failed: %s", name, checks[name]["detail"])
    return {"passed": not failed, "failed": failed, "checks": checks}
