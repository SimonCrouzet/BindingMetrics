"""Symmetric atoms of amino-acid side chains, for heavy-atom RMSD between poses.

Some atoms are indistinguishable by chemistry, and prediction models, docking tools and
crystallographers name them arbitrarily (OD1 and OD2 of Asp, CD1 and CD2 of Leu or of a phenyl
ring). ``SYMMETRIC_ATOM_GROUPS`` lists, per residue, the atoms whose names may be exchanged. The
RMSD code of ``binding_metrics.poses.agreement`` compares each residue under every exchange and
keeps the one with the smallest sum of squared distances.

Source of the table. Residues and atom names follow the Chemical Component Dictionary (wwPDB).
Each entry is an automorphism of the heavy-atom bond graph of the residue; a test checks this
against the bond table of the Chemical Component Dictionary bundled with biotite. Entries:

* Asp OD1/OD2 and Glu OE1/OE2: the carboxylate oxygens, equivalent by resonance.
* Arg NH1/NH2: the terminal nitrogens of the guanidinium group, equivalent by resonance.
* Leu CD1/CD2 and Val CG1/CG2: the methyl carbons of an isopropyl group.
* Phe and Tyr: the ring turns by 180 degrees about the CB-CG-CZ axis, which exchanges CD1/CD2
  and CE1/CE2 together (one group, since swapping one pair would break the ring). The Tyr
  hydroxyl oxygen lies on the axis. Ptr has this group too.
* Sep, Tpo and Ptr: the non-bridging phosphate oxygens O1P, O2P and O3P (all six permutations).

The free C-terminal carboxylate: O and OXT of the last residue of a linear chain are equivalent
by resonance, whatever the residue. The group depends on the position in the chain, so it is
``C_TERMINAL_CARBOXYLATE`` and not an entry of ``SYMMETRIC_ATOM_GROUPS``.
``binding_metrics.poses.agreement`` applies it per pair of poses, to the last residue of both
binders, when both binders are linear (not closed head to tail) and both poses carry O and OXT
there. Only the ``"heavy"`` atom set uses it; the ``"backbone"`` set holds O and not OXT.

Not in the table: Asn, Gln and His (the two end atoms differ in element or bonding).

Coverage. D-amino acids (``D_AA_MAP``), AMBER and CHARMM protonation variants
(``VARIANT_TO_PARENT_RESIDUE``) and N-methylated residues (``N_METHYL_PARENT``) use the entry of
their L parent: MVA as Val, MLE as Leu, SAR/NMG as Gly, NMA/MAA as Ala. A residue with no entry
that is not known to lack symmetric atoms (BMT, ABA, the lactam templates, a stapling residue,
any other non-canonical residue) has status "unknown": its atoms are compared as named, which
can only overestimate the RMSD, and ``symmetry_status`` reports it so that callers list it.
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

#: O and OXT of a free C-terminal carboxylate (see the module docstring for when it applies).
C_TERMINAL_CARBOXYLATE: SwapGroup = _exchange(("O", "OXT"))

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

    For each group, the sum of squared distances to ``coord_a`` is computed for the identity
    labelling and for each alternative; the smallest sum wins and the identity wins a tie, so the
    choice is deterministic. Groups contribute independently to the total sum of squares, so the
    per-group choice gives the smallest RMSD over all combinations.

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
