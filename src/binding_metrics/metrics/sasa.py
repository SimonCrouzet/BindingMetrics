"""Solvent accessible surface area calculations.

The trajectory functions use mdtraj's ``shrake_rupley`` (radii in nm); the
static function uses biotite. Both implement the Shrake-Rupley algorithm
(J. Mol. Biol. 79:351-371, 1973).
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from binding_metrics.metrics._common import resolve_chain_role
from binding_metrics.utils import backfill_auth_columns

logger = logging.getLogger(__name__)

try:
    import mdtraj as md
except ImportError:
    md = None


def calculate_buried_sasa(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    probe_radius: float = 0.14,
) -> np.ndarray:
    """Calculate buried solvent accessible surface area upon binding.

    The buried SASA is computed as:
        SASA_buried = SASA_ligand_alone + SASA_receptor_alone - SASA_complex

    Args:
        trajectory_path: Path to trajectory file (DCD, XTC, etc.)
        topology_path: Path to topology file (PDB)
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        probe_radius: Probe radius in nm (default 0.14 nm = 1.4 A)

    Returns:
        Array of buried SASA values (in nm^2) for each frame

    Raises:
        ImportError: If mdtraj is not installed
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for SASA calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    sasa_complex = md.shrake_rupley(traj, probe_radius=probe_radius)
    sasa_complex_total = sasa_complex.sum(axis=1)

    ligand_traj = traj.atom_slice(ligand_indices)
    sasa_ligand = md.shrake_rupley(ligand_traj, probe_radius=probe_radius)
    sasa_ligand_total = sasa_ligand.sum(axis=1)

    receptor_traj = traj.atom_slice(receptor_indices)
    sasa_receptor = md.shrake_rupley(receptor_traj, probe_radius=probe_radius)
    sasa_receptor_total = sasa_receptor.sum(axis=1)

    buried_sasa = sasa_ligand_total + sasa_receptor_total - sasa_complex_total

    return buried_sasa


def calculate_interface_sasa(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    probe_radius: float = 0.14,
) -> dict[str, np.ndarray]:
    """Calculate per-component SASA values at the interface.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        probe_radius: Probe radius in nm

    Returns:
        Dictionary with per-frame SASA arrays of shape (n_frames,) in nm^2:
        "ligand" and "receptor" (each alone), "complex", and "buried"
        (ligand + receptor - complex)
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for SASA calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    sasa_complex = md.shrake_rupley(traj, probe_radius=probe_radius)
    sasa_complex_total = sasa_complex.sum(axis=1)

    ligand_traj = traj.atom_slice(ligand_indices)
    sasa_ligand = md.shrake_rupley(ligand_traj, probe_radius=probe_radius)
    sasa_ligand_total = sasa_ligand.sum(axis=1)

    receptor_traj = traj.atom_slice(receptor_indices)
    sasa_receptor = md.shrake_rupley(receptor_traj, probe_radius=probe_radius)
    sasa_receptor_total = sasa_receptor.sum(axis=1)

    buried = sasa_ligand_total + sasa_receptor_total - sasa_complex_total

    return {
        "ligand": sasa_ligand_total,
        "receptor": sasa_receptor_total,
        "complex": sasa_complex_total,
        "buried": buried,
    }


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
) -> dict:
    """Compute delta SASA (buried surface area on binding) for a static structure.

    Uses biotite's SASA implementation, which does not require a trajectory.
    Suitable for evaluating single energy-minimized or predicted structures.

    The buried area is defined as:
        delta_SASA = SASA(peptide alone) + SASA(receptor alone) - SASA(complex)

    Positive values indicate surface buried upon binding.

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
            a defined SASA (water, ions) then count as zero area instead of
            turning the sums into NaN.

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
    from binding_metrics.metrics.interface import SASA_POINT_NUMBER, filter_hetero_atoms

    peptide_chain = resolve_chain_role(
        "peptide_chain", peptide_chain, "binder_chain", binder_chain, required=True
    )
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    try:
        import biotite.structure.io.pdbx as pdbx
        from biotite.structure.info import vdw_radius_single
        from biotite.structure.sasa import sasa as biotite_sasa
    except ImportError:
        raise ImportError(
            "biotite is required for static SASA. "
            "Install with: pip install binding-metrics[biotite]"
        )

    path = Path(cif_path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        pdbx_file = pdbx.CIFFile.read(str(path))
        backfill_auth_columns(pdbx_file)
        atoms = pdbx.get_structure(pdbx_file, model=1)
    else:
        import biotite.structure.io.pdb as pdb_io

        pdb_file = pdb_io.PDBFile.read(str(path))
        atoms = pdb_io.get_structure(pdb_file, model=1)

    atoms = filter_hetero_atoms(atoms, hetero)

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
    except Exception as e:  # kept broad: one bad structure must not abort a batch (see reason)
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
