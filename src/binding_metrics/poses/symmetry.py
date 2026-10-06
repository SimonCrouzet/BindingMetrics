"""Symmetric atoms of amino-acid side chains, for heavy-atom RMSD between poses.

Two atoms that the chemistry cannot tell apart carry names that a structure-prediction model,
a docking tool or a crystallographer assigns arbitrarily (OD1 and OD2 of an aspartate, CD1 and
CD2 of a leucine or of a phenyl ring). A heavy-atom RMSD that matches atoms by name then adds a
difference that is a labelling choice and not a geometric one. ``SYMMETRIC_ATOM_GROUPS`` lists,
for each residue, the atoms whose names may be exchanged; the RMSD code of
``binding_metrics.poses.agreement`` compares each residue under every exchange and keeps the one
with the smallest sum of squared distances.

Source of the table. The residues and atom names are those of the Chemical Component
Dictionary (wwPDB), whose atom naming every residue template of this package follows. Each
entry is an automorphism of the heavy-atom bond graph of the residue (the exchange leaves every
bond and every element in place), which is what makes the two labellings the same molecule; a
test checks that for every entry against the bond table of the bundled Chemical Component
Dictionary of biotite. The chemistry behind each entry:

* Asp OD1/OD2 and Glu OE1/OE2: the two carboxylate oxygens are equivalent by resonance.
* Arg NH1/NH2: the two terminal nitrogens of the guanidinium group are equivalent by resonance.
* Leu CD1/CD2 and Val CG1/CG2: the two methyl carbons of an isopropyl group.
* Phe and Tyr: the ring turns by 180 degrees about the CB-CG-CZ axis, which exchanges CD1 with
  CD2 and CE1 with CE2 together; the two exchanges form one group, because swapping only one
  pair would break the ring. The hydroxyl oxygen of Tyr lies on the axis and stays put.
* Sep, Tpo and Ptr: the three non-bridging phosphate oxygens O1P, O2P and O3P are equivalent
  (all six permutations).

Not in the table: Asn, Gln and His (the two end atoms are different elements or
differently bonded, so a swap is not a relabelling), and the O/OXT pair of a free C-terminal
carboxylate, which depends on the position of the residue in the chain and not on its type; an
atom of either kind is compared as named.

Which residues are covered. D-amino acids (``D_AA_MAP``), the AMBER and CHARMM protonation
variants (``VARIANT_TO_PARENT_RESIDUE``) and the N-methylated residues (``N_METHYL_PARENT``)
use the entry of their L parent: MVA as Val, MLE as Leu, SAR/NMG as Gly, NMA/MAA as Ala. A
residue that has no entry and is not known to lack symmetric atoms (any other non-canonical
residue: BMT, ABA, the lactam templates, a stapling residue, ...) is "unknown": its atoms are
compared as named, which can only overestimate the RMSD, and ``symmetry_status`` says so, so
that the callers list it in their output.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from binding_metrics.core.nonstandard import D_AA_MAP, NME_AA_MAP
from binding_metrics.core.residues import STANDARD_AMINO_ACIDS, VARIANT_TO_PARENT_RESIDUE


@dataclass(frozen=True)
class SwapGroup:
    """Atoms of one residue whose names may be exchanged together.

    Attributes:
        atoms: The atom names of the group, in a fixed order.
        alternatives: Every other labelling of the group. Each alternative lists, for each
            position of ``atoms``, the name that is compared with the atom at that position (the
            identity labelling is implied and is tried first).
    """

    atoms: tuple[str, ...]
    alternatives: tuple[tuple[str, ...], ...]


def _exchange(*pairs: tuple[str, str]) -> SwapGroup:
    """One group in which every given pair of names is exchanged at the same time."""
    exchanged: dict[str, str] = {}
    for first, second in pairs:
        exchanged[first] = second
        exchanged[second] = first
    atoms = tuple(name for pair in pairs for name in pair)
    return SwapGroup(atoms, (tuple(exchanged[name] for name in atoms),))


def _all_permutations(*atoms: str) -> SwapGroup:
    """One group whose atoms may be assigned in any order (lexicographic by position)."""
    return SwapGroup(atoms, tuple(p for p in itertools.permutations(atoms) if p != atoms))


#: Residue name (an L parent, a standard residue or a phosphorylated one) to its swap groups.
SYMMETRIC_ATOM_GROUPS: Mapping[str, tuple[SwapGroup, ...]] = MappingProxyType(
    {
        "ASP": (_exchange(("OD1", "OD2")),),
        "GLU": (_exchange(("OE1", "OE2")),),
        "LEU": (_exchange(("CD1", "CD2")),),
        "VAL": (_exchange(("CG1", "CG2")),),
        "ARG": (_exchange(("NH1", "NH2")),),
        "PHE": (_exchange(("CD1", "CD2"), ("CE1", "CE2")),),
        "TYR": (_exchange(("CD1", "CD2"), ("CE1", "CE2")),),
        "SEP": (_all_permutations("O1P", "O2P", "O3P"),),
        "TPO": (_all_permutations("O1P", "O2P", "O3P"),),
        "PTR": (
            _exchange(("CD1", "CD2"), ("CE1", "CE2")),
            _all_permutations("O1P", "O2P", "O3P"),
        ),
    }
)

# The template each N-methylated code of ``NME_AA_MAP`` stands for, as its L parent.
_N_METHYL_TEMPLATE_PARENT = {"NMG": "GLY", "NMA": "ALA", "MVA": "VAL", "MLE": "LEU"}

#: N-methylated residue codes (``NME_AA_MAP`` keys and template names) to the L amino acid that
#: they methylate: SAR and NMG to Gly, NMA and MAA to Ala, MVA to Val, MLE to Leu.
N_METHYL_PARENT: Mapping[str, str] = MappingProxyType(
    {
        code: _N_METHYL_TEMPLATE_PARENT[template]
        for code, template in NME_AA_MAP.items()
        if template in _N_METHYL_TEMPLATE_PARENT
    }
)

#: Residues known to have no pair of interchangeable heavy atoms: the standard residues that are
#: not in the table, selenomethionine and the terminal caps (acetyl, N-methylamide, amide,
#: formyl).
NO_SYMMETRIC_ATOMS_RESIDUES: frozenset[str] = frozenset(
    (STANDARD_AMINO_ACIDS - frozenset(SYMMETRIC_ATOM_GROUPS)) | {"MSE", "ACE", "NME", "NH2", "FOR"}
)


def symmetry_parent(res_name: str) -> str:
    """The residue whose table entry applies to ``res_name``.

    Follows the AMBER and CHARMM protonation variants (HID to His, ASH to Asp, ...), the D-amino
    acid codes (DLE to Leu) and the N-methylated codes (MLE to Leu); any other name is returned
    in upper case as it is.
    """
    name = res_name.strip().upper()
    name = VARIANT_TO_PARENT_RESIDUE.get(name, name)
    name = D_AA_MAP.get(name, name)
    return N_METHYL_PARENT.get(name, name)


def symmetry_groups(res_name: str) -> tuple[SwapGroup, ...]:
    """The swap groups of a residue, resolved through ``symmetry_parent``; empty when none."""
    return SYMMETRIC_ATOM_GROUPS.get(symmetry_parent(res_name), ())


def symmetry_status(res_name: str) -> str:
    """How the table covers a residue: ``"swappable"``, ``"none"`` or ``"unknown"``.

    ``"swappable"``: the residue (or its L parent) has swap groups. ``"none"``: the residue is
    known to have no interchangeable heavy atoms. ``"unknown"``: the table says nothing about
    it, so its atoms are compared as named and the RMSD can only be overestimated.
    """
    parent = symmetry_parent(res_name)
    if parent in SYMMETRIC_ATOM_GROUPS:
        return "swappable"
    if parent in NO_SYMMETRIC_ATOMS_RESIDUES:
        return "none"
    return "unknown"


@dataclass(frozen=True)
class ResolvedGroup:
    """A swap group located in a pair of coordinate arrays.

    Attributes:
        rows: Row numbers of the group atoms, in the order of ``SwapGroup.atoms``.
        alternative_rows: One row of numbers per alternative labelling: entry ``[p, j]`` is the
            row of the second array that alternative ``p`` compares with ``rows[j]`` of the
            first.
    """

    rows: np.ndarray
    alternative_rows: np.ndarray


def resolve_group(group: SwapGroup, rows_by_name: Mapping[str, int]) -> Optional[ResolvedGroup]:
    """Locate a swap group by atom name; None unless every atom of the group has a row."""
    if any(name not in rows_by_name for name in group.atoms):
        return None
    rows = np.array([rows_by_name[name] for name in group.atoms], dtype=np.intp)
    alternatives = np.array(
        [[rows_by_name[name] for name in alternative] for alternative in group.alternatives],
        dtype=np.intp,
    )
    return ResolvedGroup(rows, alternatives)


def best_labelling(
    coord_a: np.ndarray, coord_b: np.ndarray, groups: Sequence[ResolvedGroup]
) -> np.ndarray:
    """The row order of ``coord_b`` that puts every group in its best labelling.

    For each group the sum of squared distances to ``coord_a`` is computed for the identity
    labelling and for each alternative, and the labelling with the smallest sum is kept; the
    first one wins a tie, so the identity wins a tie and the choice is deterministic. Each group
    contributes to the total sum of squares independently of the others, so the choice per group
    gives the labelling with the smallest RMSD of all combinations.

    Args:
        coord_a: Coordinates of the first pose, shape (n, 3).
        coord_b: Coordinates of the second pose, same atoms in the same order, shape (n, 3).
        groups: The groups, resolved against the rows of these two arrays.

    Returns:
        ``order``, shape (n,): ``coord_b[order]`` is the second pose with its atoms labelled as
        in the first. Neither input is modified.
    """
    order = np.arange(len(coord_b), dtype=np.intp)
    for group in groups:
        reference = coord_a[group.rows]
        costs = [float(np.sum((reference - coord_b[group.rows]) ** 2))]
        costs.extend(
            float(np.sum((reference - coord_b[alternative]) ** 2))
            for alternative in group.alternative_rows
        )
        best = int(np.argmin(costs))
        if best > 0:
            order[group.rows] = group.alternative_rows[best - 1]
    return order


def count_swapped(order: np.ndarray, groups: Sequence[ResolvedGroup]) -> int:
    """How many of the groups are not in the identity labelling under ``order``."""
    return sum(1 for group in groups if np.any(order[group.rows] != group.rows))
