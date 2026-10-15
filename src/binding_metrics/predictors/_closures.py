"""Where the ring closures of a binder sit in the sequence that a runner sends to the model.

``capabilities.detect_closures`` finds the covalent links that close a ring inside a chain and
names each end by residue (name, number, and the index among the amino-acid residues of the
chain, counted from 0) and atom. A runner that writes a bond between two residues needs the
1-based position of each residue in the sequence it sends, and the atom names. The Protenix and
Boltz-2 runners share that step here.

A residue is found by its name and its number, among the ``(name, number)`` pairs of the residues
that were sent (the rule of ``metrics._openfold_run._extract_query_chain``, the reader every
runner uses for the sequence). The ``residue_index`` of an end is not used: the detector and the
query reader count residues by different rules, and a bond must point at a residue that was
sent. A residue that is not among them, or is there twice (an insertion code, a repeated number),
raises ``ValueError``; the caller adds how to switch the closures off through ``way_out``.

Only the families a model documents are placed; the others come back as text for a warning.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from binding_metrics.capabilities import Closure, ClosureEnd


@dataclasses.dataclass(frozen=True)
class PlacedBond:
    """One closure bond, with 1-based positions in the sequence sent for the binder.

    Attributes:
        family: ``head_to_tail``, ``disulfide`` or ``lactam`` (a ``CLOSURE_FAMILIES`` member).
        position1, atom1, position2, atom2: The two ends. A head-to-tail bond is always written
            as C of the last residue to N of the first, and the positions of a disulfide in
            increasing order; a lactam keeps the order of ``detect_closures``.
    """

    family: str
    position1: int
    atom1: str
    position2: int
    atom2: str


def find_closures(
    structure_path: Path, binder_chain: str, *, fallback_advice: str
) -> list[Closure]:
    """The closures of ``binder_chain`` in ``structure_path``, as ``detect_closures`` finds them.

    Args:
        structure_path: The complex structure.
        binder_chain: The binder chain ID.
        fallback_advice: What the caller can pass to skip the search, for the error text.

    Raises:
        ValueError: The structure cannot be read or searched; a linear prediction of a ring is
            not a fallback, so the failure is not swallowed.
    """
    from binding_metrics.capabilities import _read_atoms, detect_closures

    try:
        return detect_closures(_read_atoms(structure_path), binder_chain)
    except Exception as exc:  # noqa: BLE001 - external reader; raised again with the way out
        raise ValueError(
            f"could not look for ring closures in chain {binder_chain} of {structure_path} "
            f"({exc}), and a linear prediction of a ring is not the fallback: {fallback_advice}"
        ) from exc


def place_closures(
    closures: Sequence[Closure],
    residues: Sequence[tuple[str, int]],
    binder_chain: str,
    *,
    families: Sequence[str],
    way_out: str,
) -> tuple[list[PlacedBond], list[str]]:
    """Place the closures of ``families`` in the sequence sent; describe the others.

    Args:
        closures: What ``find_closures`` returned.
        residues: ``(name, number)`` of each position of the sequence sent for the binder; the
            i-th entry is position i + 1.
        binder_chain: The binder chain ID, for messages.
        families: The families to place (``head_to_tail``, ``disulfide``, ``lactam``).
        way_out: The sentence that ends an error, naming the argument that switches closures off.

    Returns:
        The placed bonds in the order of ``closures``, and the description of each closure whose
        family is not in ``families`` (a staple, another cross-link).

    Raises:
        ValueError: A closure end is not in ``residues`` or is there twice, a head-to-tail bond
            is not between C of the last residue and N of the first, or a disulfide is not
            between two SG atoms.
    """
    length = len(residues)
    places: dict[tuple[str, int], list[int]] = {}
    for position, key in enumerate(residues, start=1):
        places.setdefault(key, []).append(position)

    def position_of(end: ClosureEnd, closure: Closure) -> int:
        found = places.get((end.residue_name, end.residue_number), [])
        if len(found) != 1:
            raise ValueError(
                f"cannot place the {closure.describe()} of chain {binder_chain}: residue "
                f"{end.residue_name} {end.residue_number} is {'not' if not found else 'twice'} "
                f"in the {length} residues sent for the chain. {way_out}"
            )
        return found[0]

    placed: list[PlacedBond] = []
    skipped: list[str] = []
    for closure in closures:
        ends = (closure.end1, closure.end2)
        if closure.family not in families:
            skipped.append(closure.describe())
        elif closure.family == "head_to_tail":
            by_atom = {end.atom_name: position_of(end, closure) for end in ends}
            if set(by_atom) != {"C", "N"} or by_atom["C"] != length or by_atom["N"] != 1:
                raise ValueError(
                    f"the {closure.describe()} of chain {binder_chain} is not between C of the "
                    f"last residue ({length}) and N of the first: {by_atom}. {way_out}"
                )
            placed.append(PlacedBond("head_to_tail", length, "C", 1, "N"))
        elif closure.family == "disulfide":
            if any(end.atom_name != "SG" for end in ends):
                raise ValueError(
                    f"the {closure.describe()} of chain {binder_chain} is not between two SG "
                    f"atoms. {way_out}"
                )
            first, second = sorted(position_of(end, closure) for end in ends)
            placed.append(PlacedBond("disulfide", first, "SG", second, "SG"))
        else:
            first = position_of(closure.end1, closure)
            second = position_of(closure.end2, closure)
            placed.append(
                PlacedBond(
                    closure.family, first, closure.end1.atom_name, second, closure.end2.atom_name
                )
            )
    return placed, skipped
