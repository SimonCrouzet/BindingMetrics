"""Solvent accessible surface area calculations."""

from pathlib import Path
from typing import Optional

import numpy as np

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

    # Calculate SASA for the complex
    sasa_complex = md.shrake_rupley(traj, probe_radius=probe_radius)
    sasa_complex_total = sasa_complex.sum(axis=1)

    # Calculate SASA for ligand alone
    ligand_traj = traj.atom_slice(ligand_indices)
    sasa_ligand = md.shrake_rupley(ligand_traj, probe_radius=probe_radius)
    sasa_ligand_total = sasa_ligand.sum(axis=1)

    # Calculate SASA for receptor alone
    receptor_traj = traj.atom_slice(receptor_indices)
    sasa_receptor = md.shrake_rupley(receptor_traj, probe_radius=probe_radius)
    sasa_receptor_total = sasa_receptor.sum(axis=1)

    # Buried SASA = isolated components - complex
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
        Dictionary with SASA arrays for ligand, receptor, complex, and buried
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for SASA calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    # Complex SASA
    sasa_complex = md.shrake_rupley(traj, probe_radius=probe_radius)
    sasa_complex_total = sasa_complex.sum(axis=1)

    # Ligand SASA
    ligand_traj = traj.atom_slice(ligand_indices)
    sasa_ligand = md.shrake_rupley(ligand_traj, probe_radius=probe_radius)
    sasa_ligand_total = sasa_ligand.sum(axis=1)

    # Receptor SASA
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
    peptide_chain: str,
    receptor_chain: str,
    probe_radius: float = 1.4,
) -> dict:
    """Compute delta SASA (buried surface area on binding) for a static structure.

    Uses biotite's SASA implementation, which does not require a trajectory.
    Suitable for evaluating single energy-minimized or predicted structures.

    The buried area is defined as:
        delta_SASA = SASA(peptide alone) + SASA(receptor alone) - SASA(complex)

    Positive values indicate surface buried upon binding.

    Args:
        cif_path: Path to CIF structure file
        peptide_chain: Chain ID of the peptide
        receptor_chain: Chain ID of the receptor
        probe_radius: Solvent probe radius in Ångström (default 1.4 = water)

    Returns:
        Dictionary with keys:
            - delta_sasa (float, Å²)
            - sasa_peptide (float, Å²)
            - sasa_receptor (float, Å²)
            - sasa_complex (float, Å²)
    """
    try:
        import biotite.structure as structure
        import biotite.structure.io.pdbx as pdbx
        from biotite.structure.sasa import sasa as biotite_sasa
        from biotite.structure.info import vdw_radius_single
    except ImportError:
        raise ImportError(
            "biotite is required for static SASA. "
            "Install with: pip install binding-metrics[biotite]"
        )

    pdbx_file = pdbx.CIFFile.read(str(cif_path))
    atoms = pdbx.get_structure(pdbx_file, model=1)

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
        }

    def _get_radii(atom_array):
        radii = []
        for atom in atom_array:
            element = str(atom.element).strip()
            r = vdw_radius_single(element)
            radii.append(r if r is not None else 1.8)
        return np.array(radii, dtype=float)

    try:
        sasa_peptide = float(
            biotite_sasa(peptide_atoms, probe_radius=probe_radius,
                         point_number=960, vdw_radii=_get_radii(peptide_atoms)).sum()
        )
        sasa_receptor = float(
            biotite_sasa(receptor_atoms, probe_radius=probe_radius,
                         point_number=960, vdw_radii=_get_radii(receptor_atoms)).sum()
        )
        sasa_complex = float(
            biotite_sasa(complex_atoms, probe_radius=probe_radius,
                         point_number=960, vdw_radii=_get_radii(complex_atoms)).sum()
        )
        delta_sasa = sasa_peptide + sasa_receptor - sasa_complex
    except Exception as e:
        print(f"  Warning: biotite SASA computation failed: {e}")
        delta_sasa = sasa_peptide = sasa_receptor = sasa_complex = np.nan

    return {
        "delta_sasa": delta_sasa,
        "sasa_peptide": sasa_peptide,
        "sasa_receptor": sasa_receptor,
        "sasa_complex": sasa_complex,
    }
