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

Steric exclusion (``partner_descriptors``, design section 8.4). The partner is a known binder
that is already placed on the reference receptor (``Reference.partners``). The descriptors say
how much of the pose's binder lies where the partner lies (distances between heavy atoms, a
sphere-overlap volume) and how much of the receptor surface that the partner covers the pose
covers too (buried area shared over the receptor atoms that the pose and the reference have in
common). The partner and the binder keep only heavy polymer atoms, as the interface metric does.

Per-residue table (``interface_residue_table``, design section 8.5). One row per binder residue
and per receptor residue with buried area, in a fixed column order and sort order, for a reader
that wants the residue-level picture rather than the set-level one. The areas are those of
``_collect_per_residue`` with threshold 0, labelled with the numbering of the pose and, for the
receptor residues that the frame pairs, with the reference numbering. There is no
per-residue interaction energy: the only energy-like column is the Eisenberg-McLachlan solvation
term, which is a surface-area model.

Residue identity everywhere is ``(chain, number, insertion code)``: ``52`` and ``52A`` are two
residues.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from binding_metrics.metrics._common import import_biotite
from binding_metrics.metrics.interface import (
    _BIOTITE_PURPOSE,
    _collect_per_residue,
    _get_vdw_radii,
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

#: Heavy-atom distances (angstrom) below which an atom, or a pair of atoms, counts as a clash.
#: 2.2 is the cutoff of the validation work for clashing pairs; 3.0 also catches strongly
#: compressed carbon pairs and some hydrogen-bonded N/O pairs, so both are reported.
DEFAULT_CLASH_CUTOFFS_ANGSTROM = (2.2, 3.0)
#: A binder atom within this distance (angstrom) of a partner atom is in the partner's shell.
DEFAULT_SHELL_ANGSTROM = 4.0

#: The columns of a partner block, in order (design section 8.4).
PARTNER_COLUMNS = (
    "min_distance_angstrom",
    "clash_atoms",
    "clash_pairs",
    "shell_fraction",
    "shared_buried_area_angstrom2",
    "partner_buried_area_angstrom2",
    "pose_buried_area_angstrom2",
    "shared_fraction_of_partner",
    "overlap_volume_angstrom3",
)

#: The columns of a row of the per-residue table, in order (design section 8.5).
TABLE_COLUMNS = (
    "chain",
    "role",
    "number",
    "icode",
    "name",
    "binder_index",
    "reference_chain",
    "reference_number",
    "reference_icode",
    "buried_area_angstrom2",
    "polar_area_angstrom2",
    "apolar_area_angstrom2",
    "solvation_dg_kcal_mol",
    "in_interface",
    "in_sets",
)

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
    import_biotite(_BIOTITE_PURPOSE)
    from biotite.structure.info import vdw_radius_single as vdw_fn
    from biotite.structure.sasa import sasa as sasa_fn

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


def _check_areas(areas: InterfaceAreas, binder_chain: str, target_chain: str, probe_radius) -> None:
    if (areas.binder_chain, areas.target_chain) != (binder_chain, target_chain):
        raise ValueError(
            f"areas were computed for binder '{areas.binder_chain}' and target "
            f"'{areas.target_chain}', not '{binder_chain}' and '{target_chain}'"
        )
    if areas.probe_radius != float(probe_radius):
        raise ValueError(
            f"areas were computed with probe_radius {areas.probe_radius}, not {probe_radius}"
        )


def _areas_or_reason(atoms, binder_chain, target_chain, probe_radius, areas):
    """``(areas, None)``, or ``(None, reason)`` when a chain has no heavy polymer atom."""
    if areas is not None:
        _check_areas(areas, binder_chain, target_chain, probe_radius)
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
        logger.debug("residue sets have no value: %s", reason)
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


# ---------------------------------------------------------------------------- B2: partners

# Rows of the binder-by-partner distance matrix built at a time; bounds the memory of a large
# partner.
_DISTANCE_BLOCK_ELEMENTS = 500_000


def sphere_overlap_volume(radius_a, radius_b, distance):
    """Volume (angstrom^3 for radii and distance in angstrom) shared by two spheres.

    The closed form for the lens of two spheres of radii ``R`` and ``r`` whose centres are ``d``
    apart, ``R + r > d > |R - r|``::

        pi (R + r - d)^2 (d^2 + 2 d (R + r) - 3 (R - r)^2) / (12 d)

    When ``d <= |R - r|`` the smaller sphere lies inside the larger one and the volume is that
    of the smaller sphere, ``4/3 pi min(R, r)^3`` (also the volume for ``d = 0``); when
    ``d >= R + r`` the spheres do not overlap and the volume is 0. The two branches agree at
    ``d = |R - r|``, so the function is continuous in ``d``.

    Args:
        radius_a, radius_b, distance: Numbers or arrays that broadcast together; radii are
            positive and distances are not negative.

    Returns:
        A float for scalar inputs, otherwise an array of the broadcast shape.
    """
    radius_a, radius_b, distance = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in (radius_a, radius_b, distance))
    )
    if np.any(radius_a <= 0) or np.any(radius_b <= 0):
        raise ValueError("sphere radii must be positive")
    if np.any(distance < 0):
        raise ValueError("a distance cannot be negative")
    small = np.minimum(radius_a, radius_b)
    reach = radius_a + radius_b
    inside = distance <= np.abs(radius_a - radius_b)
    lens = (distance < reach) & ~inside
    volume = np.zeros(distance.shape)
    volume[inside] = 4.0 / 3.0 * np.pi * small[inside] ** 3
    d, total = distance[lens], reach[lens]
    volume[lens] = (
        np.pi
        * (total - d) ** 2
        * (d**2 + 2 * d * total - 3 * (radius_a[lens] - radius_b[lens]) ** 2)
    ) / (12 * d)
    return float(volume) if volume.ndim == 0 else volume


def _check_cutoffs(cutoffs: Any) -> tuple:
    try:
        values = tuple(cutoffs)
    except TypeError as error:
        raise TypeError(
            f"clash_cutoffs_angstrom must be a sequence of numbers, got {cutoffs!r}"
        ) from error
    if not values:
        raise ValueError("clash_cutoffs_angstrom must not be empty")
    for value in values:
        _check_number("clash_cutoffs_angstrom", value, minimum=0.0, exclusive=True)
    if len({float(value) for value in values}) != len(values):
        raise ValueError(f"clash_cutoffs_angstrom repeats a cutoff: {values!r}")
    return tuple(float(value) for value in values)


def _partner_failure(reason: str) -> dict:
    block: dict[str, Any] = dict.fromkeys(PARTNER_COLUMNS)
    block["reason"] = reason
    return block


def _heavy_polymer(atoms):
    return filter_hydrogens(filter_hetero_atoms(atoms))


def _finite(atoms):
    return atoms[np.isfinite(np.asarray(atoms.coord, dtype=np.float64)).all(axis=1)]


def _contact_columns(binder, partner, vdw_fn, cutoffs, shell_angstrom) -> dict:
    """Distances, clash counts, shell fraction and overlap volume of two heavy-atom sets."""
    binder_xyz = np.asarray(binder.coord, dtype=np.float64)
    partner_xyz = np.asarray(partner.coord, dtype=np.float64)
    binder_radii = _get_vdw_radii(binder, vdw_fn)
    partner_radii = _get_vdw_radii(partner, vdw_fn)
    nearest = np.empty(len(binder_xyz))
    pair_counts = dict.fromkeys(cutoffs, 0)
    overlap = 0.0
    step = max(1, _DISTANCE_BLOCK_ELEMENTS // len(partner_xyz))
    for start in range(0, len(binder_xyz), step):
        rows = slice(start, start + step)
        offsets = binder_xyz[rows, None, :] - partner_xyz[None, :, :]
        distance = np.sqrt(np.sum(offsets * offsets, axis=2))
        nearest[rows] = distance.min(axis=1)
        for cutoff in cutoffs:
            pair_counts[cutoff] += int(np.count_nonzero(distance < cutoff))
        reach = binder_radii[rows, None] + partner_radii[None, :]
        close = distance < reach
        if close.any():
            first, second = np.nonzero(close)
            overlap += float(
                sphere_overlap_volume(
                    binder_radii[rows][first], partner_radii[second], distance[first, second]
                ).sum()
            )
    return {
        "min_distance_angstrom": float(nearest.min()),
        "clash_atoms": {str(c): int(np.count_nonzero(nearest < c)) for c in cutoffs},
        "clash_pairs": {str(c): pair_counts[c] for c in cutoffs},
        "shell_fraction": float(np.count_nonzero(nearest <= shell_angstrom) / len(nearest)),
        "overlap_volume_angstrom3": overlap,
    }


def _matched_receptor_atoms(pose_receptor, reference_receptor, frame: FrameResult):
    """Rows of the pose receptor and of the reference receptor that are the same atom.

    Same atom: the residues are paired by the frame and the atom name is the same. The first
    atom of a repeated name in a residue (an alternate location) is used.
    """
    reference_row: dict[tuple, int] = {}
    for row, (number, icode, name) in enumerate(
        zip(
            reference_receptor.res_id.tolist(),
            np.char.strip(reference_receptor.ins_code.astype(str)).tolist(),
            np.char.strip(reference_receptor.atom_name.astype(str)).tolist(),
        )
    ):
        reference_row.setdefault((number, icode, name), row)
    reference_of = {
        (pose_key.chain, pose_key.number, pose_key.icode): (key.number, key.icode)
        for pose_key, key in frame.pairs
    }
    pose_rows, reference_rows = [], []
    for row, (chain, number, icode, name) in enumerate(
        zip(
            pose_receptor.chain_id.tolist(),
            pose_receptor.res_id.tolist(),
            np.char.strip(pose_receptor.ins_code.astype(str)).tolist(),
            np.char.strip(pose_receptor.atom_name.astype(str)).tolist(),
        )
    ):
        paired = reference_of.get((chain, number, icode))
        if paired is None:
            continue
        match = reference_row.get((paired[0], paired[1], name))
        if match is not None:
            pose_rows.append(row)
            reference_rows.append(match)
    return np.array(pose_rows, dtype=np.int64), np.array(reference_rows, dtype=np.int64)


def partner_descriptors(
    atoms,
    frame: FrameResult,
    reference: Reference,
    *,
    binder_chain: str,
    target_chain: str,
    clash_cutoffs_angstrom=DEFAULT_CLASH_CUTOFFS_ANGSTROM,
    shell_angstrom: float = DEFAULT_SHELL_ANGSTROM,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    areas: Optional[InterfaceAreas] = None,
) -> dict[str, dict]:
    """How the binder of a pose lies against each partner of a reference.

    Let ``B`` be the heavy atoms of the pose's binder and ``Q`` those of a partner (both in the
    reference frame, polymer atoms only), ``d_ij`` the distance of binder atom ``i`` to partner
    atom ``j`` and ``r_i`` the van der Waals radius that the surface-area code uses for the
    atom (biotite's single-radius table by element, 1.8 for an element without an entry). The
    block of one partner has these columns:

    ====================================  =====================================================
    ``min_distance_angstrom``             ``min d_ij``
    ``clash_atoms``                       ``{cutoff: binder atoms with some d_ij < cutoff}``
    ``clash_pairs``                       ``{cutoff: pairs with d_ij < cutoff}``
    ``shell_fraction``                    binder atoms with ``min_j d_ij <= shell`` over ``|B|``
    ``shared_buried_area_angstrom2``      ``sum_a min(b_pose(a), b_partner(a))``
    ``partner_buried_area_angstrom2``     ``sum_a b_partner(a)``
    ``pose_buried_area_angstrom2``        ``sum_a b_pose(a)``
    ``shared_fraction_of_partner``        ``shared / partner buried``
    ``overlap_volume_angstrom3``          sum over pairs with ``d_ij < r_i + r_j`` of the volume
                                          that the two spheres share (``sphere_overlap_volume``)
    ====================================  =====================================================

    The keys of the two clash dictionaries are the cutoffs written as floats (``"2.2"``,
    ``"3.0"``). ``b_X(a)`` is the area of receptor atom ``a`` that ``X`` buries: the area of the
    receptor alone minus its area with ``X`` added, clamped at zero. The sums run over the
    receptor atoms ``a`` that the pose and the reference both have (residues paired by the
    frame, same atom name): ``b_pose`` is measured on the pose's receptor with the pose's binder
    and ``b_partner`` on the reference's receptor with the partner, so the shared area is the
    receptor surface that the pose and the partner both cover. When the partner is the pose's
    own binder in the pose's own receptor, the shared area is the partner's whole area.

    The overlap volume is a proxy. A triple overlap is counted once per pair, so a crowded site
    is overestimated; atoms at van der Waals distance contribute nothing, and compression adds
    volume. Two atoms of one hydrogen bond (N...O about 2.9 angstrom, radii 1.55 and 1.52)
    overlap a little, as they count in any distance-based clash count.

    A partner with no value (the frame failed, a chain has no heavy polymer atom, the partner
    has none) has every column None and a ``reason``. A column that cannot be computed alone is
    None and named in ``column_reasons`` (``{column: why}``): ``shared_fraction_of_partner``
    when the partner buries no area on the matched atoms, and the four area columns when no
    receptor atom is matched.

    Args:
        atoms: The pose in the frame of the reference, as for ``residue_set_descriptors``.
        frame: The ``FrameResult`` of this pose against ``reference``.
        reference: The ``Reference``; its ``partners`` are measured, and its receptor chain
            carries the partner's buried area.
        binder_chain: The binder chain ID in ``atoms``.
        target_chain: The receptor chain ID in ``atoms``.
        clash_cutoffs_angstrom: Distances for the clash counts (default ``(2.2, 3.0)``); a
            sequence of distinct positive numbers.
        shell_angstrom: Distance of the contact shell (default 4.0); a binder atom at exactly
            this distance is in the shell.
        probe_radius: Solvent probe radius in angstrom (default 1.4).
        areas: The result of ``interface_areas`` for these atoms and chains, when the caller
            has it already.

    Returns:
        ``{partner name: block}`` in the order of ``reference.partners``; empty when the
        reference has no partner. The reference receptor is measured once per call and the
        partner once per partner.

    Raises:
        TypeError: For an argument of the wrong type.
        ValueError: For an invalid chain ID, cutoff or threshold, ``areas`` computed for other
            chains or another probe, or a partner chain that its structure does not have.
    """
    _check_common(frame, reference, atoms, binder_chain, target_chain, probe_radius)
    cutoffs = _check_cutoffs(clash_cutoffs_angstrom)
    _check_number("shell_angstrom", shell_angstrom, minimum=0.0, exclusive=True)
    partners = reference.partners
    if not partners:
        return {}
    reason = _frame_reason(frame)
    if reason is None:
        areas, reason = _areas_or_reason(atoms, binder_chain, target_chain, probe_radius, areas)
    reference_receptor = None
    if reason is None:
        reference_atoms = _heavy_polymer(reference.atoms())
        reference_receptor = reference_atoms[
            np.asarray(reference_atoms.chain_id) == reference.receptor_chain
        ]
        if len(reference_receptor) == 0:
            reason = (
                f"the reference receptor chain '{reference.receptor_chain}' has no heavy "
                "polymer atom"
            )
    binder = None if reason is not None else _finite(areas.binder_atoms)
    if binder is not None and len(binder) == 0:
        reason = "the binder has no atom with finite coordinates"
    if reason is not None:
        logger.debug("partner descriptors have no value: %s", reason)
        return {name: _partner_failure(reason) for name in partners}

    import_biotite(_BIOTITE_PURPOSE)
    from biotite.structure.info import vdw_radius_single as vdw_fn
    from biotite.structure.sasa import sasa as sasa_fn

    pose_rows, reference_rows = _matched_receptor_atoms(
        areas.receptor_atoms, reference_receptor, frame
    )
    pose_burial = areas.buried_receptor[pose_rows]
    receptor_alone = _per_atom_sasa(reference_receptor, probe_radius, sasa_fn, vdw_fn)
    blocks = {}
    for name, partner in partners.items():
        partner_atoms = _finite(_heavy_polymer(partner.atoms()))
        if len(partner_atoms) == 0:
            blocks[name] = _partner_failure(f"partner '{name}' has no heavy polymer atom")
            continue
        with_partner = _per_atom_sasa(
            reference_receptor + partner_atoms, probe_radius, sasa_fn, vdw_fn
        )
        partner_burial = np.maximum(receptor_alone - with_partner[: len(reference_receptor)], 0.0)
        columns = _contact_columns(binder, partner_atoms, vdw_fn, cutoffs, shell_angstrom)
        reasons: dict[str, str] = {}
        columns.update(_area_columns(pose_burial, partner_burial[reference_rows], reasons))
        block = {column: columns[column] for column in PARTNER_COLUMNS}
        if reasons:
            block["column_reasons"] = reasons
        blocks[name] = block
    return blocks


def _area_columns(pose_burial: np.ndarray, partner_burial: np.ndarray, reasons: dict) -> dict:
    """The four area columns over the matched receptor atoms (both arrays are in one order)."""
    columns: dict[str, Any] = {
        "shared_buried_area_angstrom2": None,
        "partner_buried_area_angstrom2": None,
        "pose_buried_area_angstrom2": None,
        "shared_fraction_of_partner": None,
    }
    if len(pose_burial) == 0:
        why = "no receptor atom of the pose matches the reference receptor"
        reasons.update(dict.fromkeys(columns, why))
        return columns
    shared = float(np.minimum(pose_burial, partner_burial).sum())
    partner_area = float(partner_burial.sum())
    columns["shared_buried_area_angstrom2"] = shared
    columns["partner_buried_area_angstrom2"] = partner_area
    columns["pose_buried_area_angstrom2"] = float(pose_burial.sum())
    columns["shared_fraction_of_partner"] = _ratio(
        shared,
        partner_area,
        reasons,
        "shared_fraction_of_partner",
        "the partner buries no area on the receptor atoms that the pose has",
    )
    return columns


# ---------------------------------------------------------------------------- C3: residue table


def _table_row(row: Mapping, role: str, threshold: float) -> dict:
    return {
        "chain": row["chain"],
        "role": role,
        "number": row["res_id"],
        "icode": row["ins_code"],
        "name": row["res_name"],
        "binder_index": None,
        "reference_chain": None,
        "reference_number": None,
        "reference_icode": None,
        "buried_area_angstrom2": float(row["buried_sasa"]),
        "polar_area_angstrom2": float(row["polar_area"]),
        "apolar_area_angstrom2": float(row["apolar_area"]),
        "solvation_dg_kcal_mol": float(row["delta_g_res"]),
        "in_interface": bool(row["buried_sasa"] >= threshold),
        "in_sets": None,
    }


def interface_residue_table(
    atoms,
    frame: Optional[FrameResult] = None,
    reference: Optional[Reference] = None,
    *,
    binder_chain: str,
    target_chain: str,
    interface_threshold_angstrom2: float = DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    areas: Optional[InterfaceAreas] = None,
) -> list[dict]:
    """The interface of a pose residue by residue, in a long format with fixed columns.

    One dict per residue, with the keys of ``TABLE_COLUMNS`` in this order:

    * ``chain``, ``role`` (``"binder"`` or ``"receptor"``), ``number``, ``icode``, ``name``:
      the residue as the atoms label it;
    * ``binder_index``: 1-based position in the binder chain, in file order (None for a
      receptor residue);
    * ``reference_chain``, ``reference_number``, ``reference_icode``: the reference residue that
      the frame pairs with a receptor residue (None for a binder residue, an unpaired receptor
      residue, or when there is no frame);
    * ``buried_area_angstrom2``, ``polar_area_angstrom2`` (N and O atoms),
      ``apolar_area_angstrom2`` (C and S atoms): the buried area of the residue, from
      ``_collect_per_residue`` with threshold 0;
    * ``solvation_dg_kcal_mol``: the solvation term of the residue (``delta_g_res`` of
      ``_collect_per_residue``; Eisenberg & McLachlan, Nature 319:199-203, 1986), negative when
      burial is favourable;
    * ``in_interface``: whether the buried area is at least ``interface_threshold_angstrom2``;
    * ``in_sets``: for a receptor residue, the sorted names of the reference residue sets that
      contain its paired reference residue (an empty list when there are none, the residue is
      unpaired or there is no reference); None for a binder residue.

    Rows: every binder residue (zero area allowed) and every receptor residue with buried area
    above zero, sorted by ``(role, chain, number, icode)``, so the binder rows come first. The
    table does not depend on the frame or the reference except through the reference columns
    and ``in_sets``; a missing or failed ``frame`` leaves those empty.

    Args:
        atoms: The pose, as for ``residue_set_descriptors``.
        frame: The ``FrameResult`` of this pose, for the reference columns; optional.
        reference: The ``Reference`` whose ``residue_sets`` fill ``in_sets``; optional, and
            needs ``frame``.
        binder_chain: The binder chain ID in ``atoms``.
        target_chain: The receptor chain ID in ``atoms``.
        interface_threshold_angstrom2: Smallest residue buried area for ``in_interface``
            (default 0.5). It does not remove rows.
        probe_radius: Solvent probe radius in angstrom (default 1.4).
        areas: The result of ``interface_areas`` for these atoms and chains, when the caller
            has it already.

    Raises:
        TypeError: For an argument of the wrong type.
        ValueError: For an invalid chain ID or threshold, ``reference`` without ``frame``,
            ``areas`` computed for other chains or another probe, or a chain with no heavy
            polymer atom (a table cannot carry a reason).
    """
    if reference is not None and not isinstance(reference, Reference):
        raise TypeError(f"reference must be a Reference, got {type(reference).__name__}")
    if frame is not None and not isinstance(frame, FrameResult):
        raise TypeError(f"frame must be a FrameResult or None, got {type(frame).__name__}")
    if reference is not None and frame is None:
        raise ValueError("reference needs the frame: its sets are read through the residue pairing")
    _check_chains(binder_chain, target_chain)
    _check_number("interface_threshold_angstrom2", interface_threshold_angstrom2, minimum=0.0)
    _check_number("probe_radius", probe_radius, minimum=0.0, exclusive=True)
    _check_one_model(atoms)
    if areas is None:
        areas = interface_areas(atoms, binder_chain, target_chain, probe_radius=probe_radius)
    else:
        _check_areas(areas, binder_chain, target_chain, probe_radius)

    paired = {}
    if frame is not None and frame.ok:
        paired = {(key.chain, key.number, key.icode): other for key, other in frame.pairs}
    sets = {} if reference is None else reference.residue_sets
    rows = []
    for index, residue in enumerate(areas.binder_residues, start=1):
        row = _table_row(residue, "binder", interface_threshold_angstrom2)
        row["binder_index"] = index
        rows.append(row)
    for residue in areas.receptor_residues:
        if residue["buried_sasa"] <= 0:
            continue
        row = _table_row(residue, "receptor", interface_threshold_angstrom2)
        other = paired.get((residue["chain"], residue["res_id"], residue["ins_code"]))
        if other is not None:
            row["reference_chain"] = other.chain
            row["reference_number"] = other.number
            row["reference_icode"] = other.icode
        row["in_sets"] = sorted(
            name for name, members in sets.items() if other is not None and other in members
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["role"], row["chain"], row["number"], row["icode"]))
    return rows


__all__ = [
    "DEFAULT_CLASH_CUTOFFS_ANGSTROM",
    "DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2",
    "DEFAULT_MAX_UNMAPPED_FRACTION",
    "DEFAULT_PROBE_RADIUS",
    "DEFAULT_SHELL_ANGSTROM",
    "PARTNER_COLUMNS",
    "SET_COLUMNS",
    "TABLE_COLUMNS",
    "InterfaceAreas",
    "interface_areas",
    "interface_residue_table",
    "partner_descriptors",
    "residue_set_descriptors",
    "sphere_overlap_volume",
]
