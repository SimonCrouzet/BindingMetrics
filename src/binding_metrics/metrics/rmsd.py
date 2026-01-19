"""RMSD calculations for structural stability analysis."""

from pathlib import Path

import numpy as np

try:
    import mdtraj as md
except ImportError:
    md = None


def calculate_rmsd(
    trajectory_path: str | Path,
    topology_path: str | Path,
    atom_indices: list[int] | None = None,
    reference_frame: int = 0,
) -> np.ndarray:
    """Calculate RMSD relative to reference frame.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        atom_indices: Atom indices to include in RMSD calculation.
            If None, uses all non-water, non-ion heavy atoms.
        reference_frame: Frame index to use as reference (default 0)

    Returns:
        Array of RMSD values (in nm) for each frame
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for RMSD calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    # Select atoms if not specified
    if atom_indices is None:
        # Select protein heavy atoms (exclude water and ions)
        atom_indices = traj.topology.select("protein and not type H")

    if len(atom_indices) == 0:
        return np.zeros(traj.n_frames)

    # Slice to selected atoms
    traj_subset = traj.atom_slice(atom_indices)

    # Superpose to reference and calculate RMSD
    traj_subset.superpose(traj_subset, frame=reference_frame)
    rmsd = md.rmsd(traj_subset, traj_subset, frame=reference_frame)

    return rmsd


def calculate_rmsf(
    trajectory_path: str | Path,
    topology_path: str | Path,
    atom_indices: list[int] | None = None,
) -> np.ndarray:
    """Calculate root mean square fluctuation per atom.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        atom_indices: Atom indices to include. If None, uses all
            non-water, non-ion heavy atoms.

    Returns:
        Array of RMSF values (in nm) for each selected atom
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for RMSF calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    if atom_indices is None:
        atom_indices = traj.topology.select("protein and not type H")

    if len(atom_indices) == 0:
        return np.array([])

    traj_subset = traj.atom_slice(atom_indices)

    # Superpose to average structure
    traj_subset.superpose(traj_subset, frame=0)

    # Calculate mean positions
    mean_positions = traj_subset.xyz.mean(axis=0)

    # Calculate RMSF
    diff = traj_subset.xyz - mean_positions
    rmsf = np.sqrt((diff ** 2).sum(axis=2).mean(axis=0))

    return rmsf


def calculate_ligand_rmsd(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    reference_frame: int = 0,
) -> dict[str, np.ndarray]:
    """Calculate RMSD for ligand after aligning on receptor.

    This measures ligand movement relative to the receptor binding site.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor (for alignment)
        reference_frame: Frame index to use as reference

    Returns:
        Dictionary with 'ligand_rmsd' and 'receptor_rmsd' arrays
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for RMSD calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    # Align on receptor
    traj.superpose(traj, frame=reference_frame, atom_indices=receptor_indices)

    # Calculate receptor RMSD (should be ~0 after alignment)
    receptor_traj = traj.atom_slice(receptor_indices)
    receptor_rmsd = md.rmsd(receptor_traj, receptor_traj, frame=reference_frame)

    # Calculate ligand RMSD (relative to receptor-aligned reference)
    ligand_traj = traj.atom_slice(ligand_indices)
    ligand_rmsd = md.rmsd(ligand_traj, ligand_traj, frame=reference_frame)

    return {
        "ligand_rmsd": ligand_rmsd,
        "receptor_rmsd": receptor_rmsd,
    }
