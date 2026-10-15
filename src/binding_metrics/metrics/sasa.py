"""Solvent accessible surface area calculations.

The trajectory functions use mdtraj's ``shrake_rupley`` (radii in nm); the
static function uses biotite. Both implement the Shrake-Rupley algorithm
(J. Mol. Biol. 79:351-371, 1973).

The static function reports the buried area of heavy atoms by default
(``hydrogens="ignore"``), the convention of PISA and of the atomic solvation
parameters used for ΔG_int in ``interface``. The trajectory functions do not
filter hydrogens: they use the atoms of the index lists as given. Their areas are
in nm^2 (mdtraj); the static function reports Å^2.

A trajectory that stores a unit cell is imaged first (``_periodic.image_complex``): molecules
are made whole and the ligand is placed next to the receptor, so the areas do not depend on
the periodic image in which either partner was written. A trajectory without a unit cell is
used as stored.
"""

import logging
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from binding_metrics.metrics._common import import_biotite, load_structure, resolve_chain_role
from binding_metrics.metrics._periodic import has_unit_cell, image_complex

logger = logging.getLogger(__name__)


def _import_mdtraj():
    """Import mdtraj for the trajectory functions; the static function never needs it.

    A module-level import would load mdtraj for every caller of
    ``compute_delta_sasa_static``, including a pose-set call on the ``static`` extra.
    """
    try:
        import mdtraj
    except ImportError as exc:
        raise ImportError(
            "mdtraj is required for SASA calculations. "
            "Install with: pip install binding-metrics[analysis]"
        ) from exc
    return mdtraj


def _check_selections(
    function_name: str,
    ligand_indices: list[int],
    receptor_indices: list[int],
    on_empty: str,
) -> None:
    """Refuse overlapping index lists and report empty ones.

    An atom in both lists would be counted in the ligand term, the receptor term and once in
    the complex, so overlapping lists raise ``ValueError``. An empty list gives a buried area
    of zero ("nothing was evaluated"); ``on_empty`` selects a ``RuntimeWarning`` or a
    ``ValueError``. ``stacklevel=4`` attributes the warning to the caller of the public
    function (this function, ``_interface_sasa_nm2``, the public function, its caller).
    """
    from binding_metrics.metrics.rmsd import _check_on_empty

    _check_on_empty(on_empty)
    shared = sorted(set(ligand_indices) & set(receptor_indices))
    if shared:
        raise ValueError(
            f"{function_name}: ligand_indices and receptor_indices share {len(shared)} "
            f"atom(s), first {shared[:5]}; the selections must be disjoint"
        )
    empty = [
        name
        for name, indices in (
            ("ligand_indices", ligand_indices),
            ("receptor_indices", receptor_indices),
        )
        if len(indices) == 0
    ]
    if empty:
        message = (
            f"{function_name}: {' and '.join(empty)} contain no atoms; the buried area is "
            "zero because nothing was evaluated"
        )
        if on_empty == "raise":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=4)


def _interface_sasa_nm2(
    function_name: str,
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    probe_radius: float,
    on_empty: str,
) -> dict[str, np.ndarray]:
    """Per-frame SASA in nm^2 of the ligand, the receptor and their complex.

    The complex is the union of the two index lists. Atoms outside both (the water and
    ions of a solvated trajectory) are left out of every term; with them in, the complex
    term would hold the outer surface of the solvent box and the buried area would be
    negative.

    With a unit cell, the areas are taken on the imaged complex of ``image_complex``, which
    holds the two selections only; the index lists are then positions in that copy.
    """
    md = _import_mdtraj()
    _check_selections(function_name, ligand_indices, receptor_indices, on_empty)

    traj = md.load(str(trajectory_path), top=str(topology_path))
    if len(ligand_indices) and len(receptor_indices) and has_unit_cell(traj):
        traj, ligand_indices, receptor_indices = image_complex(
            traj, ligand_indices, receptor_indices
        )

    def total_sasa(atom_indices) -> np.ndarray:
        sliced = traj.atom_slice(atom_indices)
        # One frame per call: mdtraj 1.11.1 computes the areas of a frame from state left by the
        # previous frame of the same OpenMP thread, so a multi-frame call depends on the frame
        # order and on the thread count.
        return np.array(
            [
                md.shrake_rupley(sliced[i], probe_radius=probe_radius).sum()
                for i in range(sliced.n_frames)
            ]
        )

    sasa_ligand = total_sasa(ligand_indices)
    sasa_receptor = total_sasa(receptor_indices)
    sasa_complex = total_sasa(sorted(set(ligand_indices) | set(receptor_indices)))

    return {
        "ligand": sasa_ligand,
        "receptor": sasa_receptor,
        "complex": sasa_complex,
        "buried": sasa_ligand + sasa_receptor - sasa_complex,
    }


def calculate_buried_sasa(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    probe_radius: float = 0.14,
    *,
    on_empty: Literal["warn", "raise"] = "warn",
) -> np.ndarray:
    """Calculate buried solvent accessible surface area upon binding.

    SASA_buried = SASA_ligand_alone + SASA_receptor_alone - SASA_complex

    The complex is the ligand and receptor atoms only; other atoms of the trajectory
    (water, ions) are ignored, so a solvated trajectory gives the same area as a dry one.
    Hydrogens are not filtered, unlike in ``compute_delta_sasa_static``.

    When the trajectory stores a unit cell (orthorhombic or triclinic), molecules are made
    whole and the ligand is placed in the periodic image next to the receptor before the
    areas are taken, so the result does not depend on the image in which either partner was
    written; without a cell the stored coordinates are used.

    Args:
        trajectory_path: Path to trajectory file (DCD, XTC, etc.)
        topology_path: Path to topology file (PDB)
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        probe_radius: Probe radius in nm (default 0.14 nm = 1.4 A)
        on_empty: What to do when ``ligand_indices`` or ``receptor_indices`` is empty.
            "warn" (default) emits a ``RuntimeWarning`` naming the empty selection and
            returns zeros ("nothing evaluated"); "raise" raises ``ValueError``.

    Returns:
        Array of buried SASA values (in nm^2, not Angstrom^2: multiply by 100 for A^2)
        for each frame

    Raises:
        ImportError: If mdtraj is not installed
        ValueError: If the two index lists share an atom, if ``on_empty`` is not "warn"
            or "raise", or is "raise" and a selection is empty.
    """
    return _interface_sasa_nm2(
        "calculate_buried_sasa",
        trajectory_path,
        topology_path,
        ligand_indices,
        receptor_indices,
        probe_radius,
        on_empty,
    )["buried"]


def calculate_interface_sasa(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    probe_radius: float = 0.14,
    *,
    on_empty: Literal["warn", "raise"] = "warn",
) -> dict[str, np.ndarray]:
    """Calculate per-component SASA values at the interface.

    The complex is the ligand and receptor atoms only, as in ``calculate_buried_sasa``.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        probe_radius: Probe radius in nm
        on_empty: As in ``calculate_buried_sasa``.

    Returns:
        Dictionary with per-frame SASA arrays of shape (n_frames,) in nm^2 (multiply
        by 100 for A^2): "ligand" and "receptor" (each alone), "complex" (ligand and
        receptor atoms together), and "buried" (ligand + receptor - complex)

    Raises:
        ImportError: If mdtraj is not installed
        ValueError: As in ``calculate_buried_sasa``.
    """
    return _interface_sasa_nm2(
        "calculate_interface_sasa",
        trajectory_path,
        topology_path,
        ligand_indices,
        receptor_indices,
        probe_radius,
        on_empty,
    )


# ---------------------------------------------------------------------------
# Biotite-based SASA for static structures (no trajectory required)
# ---------------------------------------------------------------------------


def compute_delta_sasa_static(
    cif_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    probe_radius: float = 1.4,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    hetero: Literal["ignore", "keep"] = "ignore",
    hydrogens: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Compute delta SASA (buried surface area on binding) for a static structure.

    Uses biotite's SASA implementation; no trajectory is needed.

    delta_SASA = SASA(peptide alone) + SASA(receptor alone) - SASA(complex)

    Positive values indicate surface buried upon binding. The areas are
    heavy-atom areas unless ``hydrogens="keep"``, so ``delta_sasa`` equals the
    value from ``interface.compute_interface_metrics`` and does not depend on
    the protonation of the input.

    Args:
        cif_path: Path to CIF structure file
        peptide_chain: Chain ID of the peptide. Required, through this
            parameter or ``binder_chain``.
        receptor_chain: Chain ID of the receptor. Required, through this
            parameter or ``target_chain``.
        probe_radius: Solvent probe radius in Ångström (default 1.4 = water)
        binder_chain: Alias of ``peptide_chain``; different IDs in both raise
            ``ValueError``.
        target_chain: Alias of ``receptor_chain``, same rule.
        hetero: "ignore" (default) keeps only the polymer before the chain
            selection (see ``interface.filter_hetero_atoms``), so waters,
            ions, ligands and glycans that carry a protein chain ID are
            dropped. "keep" uses every atom with the chain ID; atoms without
            a defined SASA (water, ions) count as zero area.
        hydrogens: "ignore" (default) drops hydrogen and deuterium atoms
            before the areas are computed (see ``interface.filter_hydrogens``).
            "keep" includes them.

    Returns:
        Dictionary with keys:
            - delta_sasa (float, Å²)
            - sasa_peptide (float, Å²)
            - sasa_receptor (float, Å²)
            - sasa_complex (float, Å²)
            - reason (str): only when a value could not be computed (empty
              chain, failed SASA); the areas are then 0.0 for an empty chain
              and NaN for a failed calculation.
    """
    from binding_metrics.metrics.interface import (
        SASA_POINT_NUMBER,
        filter_hetero_atoms,
        filter_hydrogens,
    )

    peptide_chain = resolve_chain_role(
        "peptide_chain", peptide_chain, "binder_chain", binder_chain, required=True
    )
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    import_biotite("static SASA")
    from biotite.structure.info import vdw_radius_single
    from biotite.structure.sasa import sasa as biotite_sasa

    atoms = load_structure(cif_path, purpose="static SASA")
    atoms = filter_hydrogens(filter_hetero_atoms(atoms, hetero), hydrogens)

    peptide_mask = atoms.chain_id == peptide_chain
    receptor_mask = atoms.chain_id == receptor_chain
    complex_mask = peptide_mask | receptor_mask

    peptide_atoms = atoms[peptide_mask]
    receptor_atoms = atoms[receptor_mask]
    complex_atoms = atoms[complex_mask]

    if len(peptide_atoms) == 0 or len(receptor_atoms) == 0:
        return {
            "delta_sasa": 0.0,
            "sasa_peptide": 0.0,
            "sasa_receptor": 0.0,
            "sasa_complex": 0.0,
            "reason": (
                f"peptide chain {peptide_chain!r} or receptor chain {receptor_chain!r} "
                f"has no atoms (hetero={hetero!r})"
            ),
        }

    def _get_radii(atom_array):
        radii = []
        for atom in atom_array:
            element = str(atom.element).strip()
            if element == "D":  # the table has no entry for deuterium
                element = "H"
            r = vdw_radius_single(element)
            radii.append(r if r is not None else 1.8)
        return np.array(radii, dtype=float)

    def _total_sasa(atom_array) -> float:
        # biotite marks atoms it does not sample (water, monoatomic ions) with NaN;
        # they carry no area, so nansum keeps them from poisoning the total.
        per_atom = biotite_sasa(
            atom_array,
            probe_radius=probe_radius,
            point_number=SASA_POINT_NUMBER,
            vdw_radii=_get_radii(atom_array),
        )
        return float(np.nansum(per_atom))

    reason = None
    try:
        sasa_peptide = _total_sasa(peptide_atoms)
        sasa_receptor = _total_sasa(receptor_atoms)
        sasa_complex = _total_sasa(complex_atoms)
        delta_sasa = sasa_peptide + sasa_receptor - sasa_complex
    except Exception as e:  # noqa: BLE001 - one bad structure must not abort a batch; see reason
        logger.warning(f"  Warning: biotite SASA computation failed: {e}")
        delta_sasa = sasa_peptide = sasa_receptor = sasa_complex = np.nan
        reason = f"SASA computation failed: {type(e).__name__}: {e}"

    result = {
        "delta_sasa": delta_sasa,
        "sasa_peptide": sasa_peptide,
        "sasa_receptor": sasa_receptor,
        "sasa_complex": sasa_complex,
    }
    if reason is not None:
        result["reason"] = reason
    return result
