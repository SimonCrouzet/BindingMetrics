"""Reference-site descriptors: where a pose sits relative to what is known about the target.

The functions take the atoms of ONE pose that are already in the frame of the reference (the
output of ``binding_metrics.poses.frame.place_in_reference_frame``), the ``FrameResult`` of that
pose and the ``Reference`` it was placed on, and return plain dictionaries. Nothing here moves a
structure, reads a file or judges a value: the thresholds are parameters with documented
defaults, and a quantity that cannot be computed is ``None`` with a reason, never NaN.

What the atoms must look like. Chains carry the user's chain IDs, and the receptor residues keep
the numbering that the frame stage read: ``frame.pairs`` names each paired residue by its pose
key ``(chain, number, insertion code)`` and the descriptors look the residues up by that key.
Pass the transformed copy of the pose, not a copy renumbered to the reference
(``renumber_to_reference``): the reference numbers are reached through ``frame.pairs``.

The interface. A residue is in the interface when its buried area is at least
``interface_threshold_angstrom2`` (default 0.5, the ``interface_threshold`` of
``compute_interface_metrics``). The buried area of an atom is its solvent-accessible area
(Shrake-Rupley, 960 points per atom, 1.4 angstrom probe, as in ``metrics.interface``) with the
chain alone minus its area in the binder-receptor complex, clamped at zero; the buried area of a
residue is the sum over its heavy atoms. Waters, ligands and hydrogens are not part of the
surface, by the same filters as ``compute_interface_metrics``. ``interface_areas`` computes this
once; every function takes the result as ``areas`` so that one pose is not measured twice.

Residue sets (``residue_set_descriptors``, design section 8.3). Notation: ``S`` is a named set in
reference numbering, ``P`` the receptor residues of the pose that the frame pairs with reference
residues (a trimmed construct lacks some), ``I`` the residues of ``P`` that are in the
interface. A set with more than ``max_unmapped_fraction`` (default 0.5) of its residues outside
``P`` has no value, and says so.

Residue identity everywhere is ``(chain, number, insertion code)``: ``52`` and ``52A`` are two
residues.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from binding_metrics.metrics.interface import (
    _collect_per_residue,
    _import_biotite,
    _per_atom_sasa,
    filter_hetero_atoms,
    filter_hydrogens,
)
from binding_metrics.poses.frame import FrameResult
from binding_metrics.poses.reference import Reference, ResidueKey

logger = logging.getLogger(__name__)

#: Smallest buried area (angstrom^2) of a residue in the interface; the default of
#: ``compute_interface_metrics``.
DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2 = 0.5
#: Largest fraction of a residue set that the pose may lack before the set has no value.
DEFAULT_MAX_UNMAPPED_FRACTION = 0.5
#: Radius of the solvent probe (angstrom): water.
DEFAULT_PROBE_RADIUS = 1.4

#: The columns of a residue-set block, in order (design section 8.3).
SET_COLUMNS = (
    "n_set",
    "n_set_mapped",
    "n_interface",
    "n_overlap",
    "covered_fraction",
    "interface_in_set_fraction",
    "jaccard",
    "receptor_buried_area_in_set_angstrom2",
    "receptor_buried_area_in_set_fraction",
)

# Slack on comparisons of fractions, so that 11 unmapped residues of 22 are exactly one half.
_FRACTION_TOLERANCE = 1e-12


class _EmptyChainError(ValueError):
    """A chain has no heavy polymer atom, so its interface cannot be measured."""


# ---------------------------------------------------------------------------- arguments


def _check_chain(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a chain ID, got {value!r}")


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


def _check_chains(binder_chain: Any, target_chain: Any) -> None:
    _check_chain("binder_chain", binder_chain)
    _check_chain("target_chain", target_chain)
    if binder_chain == target_chain:
        raise ValueError(f"binder_chain and target_chain are both '{binder_chain}': two chains")


def _check_one_model(atoms) -> None:
    if getattr(atoms.coord, "ndim", 2) != 2:
        raise ValueError("atoms must be one model: take one model of the stack with stack[i]")


def _check_common(frame, reference, atoms, binder_chain, target_chain, probe_radius) -> None:
    if not isinstance(reference, Reference):
        raise TypeError(f"reference must be a Reference, got {type(reference).__name__}")
    if not isinstance(frame, FrameResult):
        raise TypeError(f"frame must be a FrameResult, got {type(frame).__name__}")
    _check_chains(binder_chain, target_chain)
    _check_number("probe_radius", probe_radius, minimum=0.0, exclusive=True)
    _check_one_model(atoms)


# ---------------------------------------------------------------------------- interface areas


@dataclass(frozen=True, eq=False)
class InterfaceAreas:
    """The buried area of one pose, per atom and per residue, in the frame of its atoms.

    Build with ``interface_areas``. The atoms are the heavy polymer atoms of each chain in file
    order; ``buried_binder[i]`` is the area (angstrom^2) of ``binder_atoms[i]`` that the receptor
    covers, and ``buried_receptor`` the same for the receptor.

    Attributes:
        binder_chain, target_chain: The chain IDs the areas were computed for.
        probe_radius: The probe radius used (angstrom).
        binder_atoms, receptor_atoms: biotite ``AtomArray`` of the two chains after the polymer
            and hydrogen filters.
        buried_binder, buried_receptor: Per-atom buried area, float arrays.
        binder_residues, receptor_residues: One dict per residue in file order, as
            ``_collect_per_residue`` returns them with threshold 0: ``residue``, ``chain``,
            ``res_name``, ``res_id``, ``ins_code``, ``buried_sasa``, ``delta_g_res``,
            ``polar_area``, ``apolar_area``. Residues without buried area are included.
    """

    binder_chain: str
    target_chain: str
    probe_radius: float
    binder_atoms: Any
    receptor_atoms: Any
    buried_binder: np.ndarray
    buried_receptor: np.ndarray
    binder_residues: tuple
    receptor_residues: tuple


def interface_areas(
    atoms, binder_chain: str, target_chain: str, *, probe_radius: float = DEFAULT_PROBE_RADIUS
) -> InterfaceAreas:
    """Measure the buried area of the binder-receptor interface of one pose.

    The same calculation as ``compute_interface_metrics`` (same filters, same per-atom
    Shrake-Rupley areas, clamp at zero), so the sums equal its ``delta_sasa`` on the same atoms;
    the per-residue lists are its ``_collect_per_residue`` with threshold 0.

    Args:
        atoms: A biotite ``AtomArray`` of one model.
        binder_chain: The binder (peptide) chain ID.
        target_chain: The receptor chain ID.
        probe_radius: Solvent probe radius in angstrom (default 1.4).

    Raises:
        ValueError: If a chain ID is empty or the two are equal, ``probe_radius`` is not
            positive, the atoms are a stack, or a chain has no heavy polymer atom.
    """
    _check_chains(binder_chain, target_chain)
    _check_number("probe_radius", probe_radius, minimum=0.0, exclusive=True)
    _check_one_model(atoms)
    _, _, _, sasa_fn, vdw_fn = _import_biotite()
    heavy = filter_hydrogens(filter_hetero_atoms(atoms))
    binder_mask = np.asarray(heavy.chain_id) == binder_chain
    receptor_mask = np.asarray(heavy.chain_id) == target_chain
    empty = [
        chain
        for chain, mask in ((binder_chain, binder_mask), (target_chain, receptor_mask))
        if not mask.any()
    ]
    if empty:
        raise _EmptyChainError(
            f"no heavy polymer atom left for chain(s) {empty}: the interface cannot be measured"
        )
    binder_atoms = heavy[binder_mask]
    receptor_atoms = heavy[receptor_mask]
    complex_atoms = heavy[binder_mask | receptor_mask]
    sasa_binder = _per_atom_sasa(binder_atoms, probe_radius, sasa_fn, vdw_fn)
    sasa_receptor = _per_atom_sasa(receptor_atoms, probe_radius, sasa_fn, vdw_fn)
    sasa_complex = _per_atom_sasa(complex_atoms, probe_radius, sasa_fn, vdw_fn)
    in_complex = np.asarray(complex_atoms.chain_id)
    # The isolated and the complex areas are sampled independently, so a buried atom can come
    # out marginally negative: clamp, as compute_interface_metrics does.
    buried_binder = np.maximum(sasa_binder - sasa_complex[in_complex == binder_chain], 0.0)
    buried_receptor = np.maximum(sasa_receptor - sasa_complex[in_complex == target_chain], 0.0)
    return InterfaceAreas(
        binder_chain=binder_chain,
        target_chain=target_chain,
        probe_radius=float(probe_radius),
        binder_atoms=binder_atoms,
        receptor_atoms=receptor_atoms,
        buried_binder=buried_binder,
        buried_receptor=buried_receptor,
        binder_residues=tuple(_collect_per_residue(binder_atoms, buried_binder, 0.0)),
        receptor_residues=tuple(_collect_per_residue(receptor_atoms, buried_receptor, 0.0)),
    )


def _areas_or_reason(atoms, binder_chain, target_chain, probe_radius, areas):
    """``(areas, None)``, or ``(None, reason)`` when a chain has no heavy polymer atom."""
    if areas is not None:
        if (areas.binder_chain, areas.target_chain) != (binder_chain, target_chain):
            raise ValueError(
                f"areas were computed for binder '{areas.binder_chain}' and target "
                f"'{areas.target_chain}', not '{binder_chain}' and '{target_chain}'"
            )
        if areas.probe_radius != float(probe_radius):
            raise ValueError(
                f"areas were computed with probe_radius {areas.probe_radius}, not {probe_radius}"
            )
        return areas, None
    try:
        return interface_areas(atoms, binder_chain, target_chain, probe_radius=probe_radius), None
    except _EmptyChainError as error:
        return None, str(error)


def _frame_reason(frame: FrameResult) -> Optional[str]:
    return None if frame.ok else f"the frame failed: {frame.reason}"


def _residue_key(row: Mapping) -> ResidueKey:
    return ResidueKey(row["chain"], row["res_id"], row["ins_code"])


# ---------------------------------------------------------------------------- B1: residue sets


def _set_failure(n_set: int, n_mapped: Optional[int], reason: str) -> dict:
    block: dict[str, Any] = dict.fromkeys(SET_COLUMNS)
    block["n_set"] = n_set
    block["n_set_mapped"] = n_mapped
    block["reason"] = reason
    return block


def _ratio(numerator: float, denominator: float, reasons: dict, column: str, why: str):
    if denominator > 0:
        return numerator / denominator
    reasons[column] = why
    return None


def residue_set_descriptors(
    atoms,
    frame: FrameResult,
    reference: Reference,
    *,
    binder_chain: str,
    target_chain: str,
    interface_threshold_angstrom2: float = DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2,
    max_unmapped_fraction: float = DEFAULT_MAX_UNMAPPED_FRACTION,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    areas: Optional[InterfaceAreas] = None,
) -> dict[str, dict]:
    """How the interface of a pose overlaps each named residue set of a reference.

    For each set ``S`` of ``reference.residue_sets`` (reference numbering), with ``P`` and ``I``
    as defined in the module docstring, the block has these columns:

    ========================================  ===================================================
    ``n_set``                                 ``|S|``
    ``n_set_mapped``                          ``|S & P|``: set residues that the pose has
    ``n_interface``                           ``|I|``
    ``n_overlap``                             ``|I & S|``
    ``covered_fraction``                      ``n_overlap / n_set_mapped``
    ``interface_in_set_fraction``             ``n_overlap / n_interface``
    ``jaccard``                               ``n_overlap / |I | (S & P)|``
    ``receptor_buried_area_in_set_angstrom2`` buried area of the receptor residues in ``S & P``
    ``receptor_buried_area_in_set_fraction``  that area over the total receptor buried area
    ========================================  ===================================================

    The area is the receptor side only (``compute_interface_metrics`` adds both partners).

    A set that the pose lacks too much of (more than ``max_unmapped_fraction`` of its residues
    are not in ``P``) has every column except ``n_set`` and ``n_set_mapped`` set to None and a
    ``reason`` such as ``"22 set residues, 14 not in the pose"``. The same holds for every set
    when the frame failed, a chain has no heavy polymer atom, or none of the paired residues is
    in the atoms. A single column with a zero denominator is None and named in
    ``column_reasons`` (``{column: why}``), which is absent when every column has a value; the
    key ``reason`` is used only when the whole block has no value.

    Args:
        atoms: The pose in the frame of the reference: a biotite ``AtomArray`` of one model,
            with the numbering the frame stage read (see the module docstring).
        frame: The ``FrameResult`` of this pose against ``reference``; its ``pairs`` map pose
            residues to reference residues.
        reference: The ``Reference`` the pose was placed on; its ``residue_sets`` are measured.
        binder_chain: The binder chain ID in ``atoms``.
        target_chain: The receptor chain ID in ``atoms``.
        interface_threshold_angstrom2: Smallest residue buried area that puts it in the
            interface (default 0.5).
        max_unmapped_fraction: Largest fraction of a set that may be missing from the pose
            (default 0.5, between 0 and 1); a set at exactly this fraction is computed.
        probe_radius: Solvent probe radius in angstrom (default 1.4).
        areas: The result of ``interface_areas`` for these atoms and chains, when the caller
            has it already.

    Returns:
        ``{set name: block}`` in the order of ``reference.residue_sets``; empty when the
        reference has no set.

    Raises:
        TypeError: For an argument of the wrong type.
        ValueError: For an invalid chain ID or threshold, or ``areas`` computed for other
            chains or another probe.
    """
    _check_common(frame, reference, atoms, binder_chain, target_chain, probe_radius)
    _check_number("interface_threshold_angstrom2", interface_threshold_angstrom2, minimum=0.0)
    _check_number("max_unmapped_fraction", max_unmapped_fraction, minimum=0.0, maximum=1.0)
    sets = reference.residue_sets
    if not sets:
        return {}
    reason = _frame_reason(frame)
    if reason is None:
        areas, reason = _areas_or_reason(atoms, binder_chain, target_chain, probe_radius, areas)
    if reason is None:
        receptor_area = {_residue_key(row): row["buried_sasa"] for row in areas.receptor_residues}
        area_of_reference = {
            reference_key: receptor_area[pose_key]
            for pose_key, reference_key in frame.pairs
            if pose_key in receptor_area
        }
        if not area_of_reference:
            reason = (
                f"none of the {frame.n_paired} paired receptor residues is on chain "
                f"'{target_chain}' of the atoms: pass the atoms with the numbering the frame "
                "stage read"
            )
    if reason is not None:
        return {name: _set_failure(len(members), None, reason) for name, members in sets.items()}

    mapped_keys = frozenset(area_of_reference)
    interface_keys = frozenset(
        key for key, area in area_of_reference.items() if area >= interface_threshold_angstrom2
    )
    total_area = float(areas.buried_receptor.sum())
    return {
        name: _set_block(
            members,
            mapped_keys,
            interface_keys,
            area_of_reference,
            total_area,
            max_unmapped_fraction,
        )
        for name, members in sets.items()
    }


def _set_block(members, mapped_keys, interface_keys, area_of_reference, total_area, limit) -> dict:
    n_set = len(members)
    in_pose = members & mapped_keys
    n_mapped = len(in_pose)
    n_unmapped = n_set - n_mapped
    if n_unmapped / n_set > limit + _FRACTION_TOLERANCE:
        return _set_failure(
            n_set,
            n_mapped,
            f"{n_set} set residues, {n_unmapped} not in the pose "
            f"(max_unmapped_fraction is {limit:g})",
        )
    overlap = interface_keys & members
    n_overlap = len(overlap)
    n_interface = len(interface_keys)
    area_in_set = float(sum(area_of_reference[key] for key in in_pose))
    reasons: dict[str, str] = {}
    block: dict[str, Any] = {
        "n_set": n_set,
        "n_set_mapped": n_mapped,
        "n_interface": n_interface,
        "n_overlap": n_overlap,
        "covered_fraction": _ratio(
            n_overlap, n_mapped, reasons, "covered_fraction", "no residue of the set is in the pose"
        ),
        "interface_in_set_fraction": _ratio(
            n_overlap,
            n_interface,
            reasons,
            "interface_in_set_fraction",
            "the pose has no interface residue",
        ),
        "jaccard": _ratio(
            n_overlap,
            len(interface_keys | in_pose),
            reasons,
            "jaccard",
            "the interface and the mapped set are both empty",
        ),
        "receptor_buried_area_in_set_angstrom2": area_in_set,
        "receptor_buried_area_in_set_fraction": _ratio(
            area_in_set,
            total_area,
            reasons,
            "receptor_buried_area_in_set_fraction",
            "no receptor area is buried",
        ),
    }
    if reasons:
        block["column_reasons"] = reasons
    return block


__all__ = [
    "DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2",
    "DEFAULT_MAX_UNMAPPED_FRACTION",
    "DEFAULT_PROBE_RADIUS",
    "SET_COLUMNS",
    "InterfaceAreas",
    "interface_areas",
    "residue_set_descriptors",
]
