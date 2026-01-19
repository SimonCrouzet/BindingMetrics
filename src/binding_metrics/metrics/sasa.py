"""Solvent accessible surface area calculations."""

from pathlib import Path

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
