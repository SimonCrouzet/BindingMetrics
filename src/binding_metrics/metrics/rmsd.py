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


def compute_receptor_drift(
    trajectory_path: str | Path,
    topology_path: str | Path,
    receptor_chain: str,
    reference_frame: int = 0,
) -> dict:
    """Compute receptor backbone drift over a trajectory.

    Measures how much the receptor chain drifts from the reference frame,
    both as aligned (conformational) drift and absolute (raw) drift.
    Absolute drift is set to NaN when periodic boundary conditions are
    detected, as PBC makes raw displacement physically meaningless.

    Type: score

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology/PDB file
        receptor_chain: Chain ID of the receptor (e.g. "A")
        reference_frame: Frame index to use as reference (default 0)

    Returns:
        Dictionary with keys:

        Scores:
            drift_aligned_mean (float): Mean conformational drift in Å
                (after superposition on receptor Cα atoms)
            drift_aligned_max (float): Maximum conformational drift in Å
            drift_raw_mean (float): Mean absolute drift in Å; NaN if PBC detected
            drift_raw_max (float): Maximum absolute drift in Å; NaN if PBC detected
            pbc_detected (bool): True if periodic boundary conditions found

        Features:
            drift_aligned_per_frame (np.ndarray): Per-frame aligned drift (n_frames,) in Å
            drift_raw_per_frame (np.ndarray): Per-frame raw drift (n_frames,) in Å;
                NaN array if PBC detected
            n_receptor_ca (int): Number of receptor Cα atoms used
            n_frames (int): Total number of frames in trajectory
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for receptor drift calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    # Select receptor Cα atoms
    ca_indices = []
    for atom in traj.topology.atoms:
        if atom.name != "CA":
            continue
        chain_id = getattr(atom.residue.chain, "chain_id", str(atom.residue.chain.index))
        if chain_id == receptor_chain:
            ca_indices.append(atom.index)

    if len(ca_indices) == 0:
        nan_arr = np.full(traj.n_frames, np.nan)
        return {
            "drift_aligned_mean": np.nan,
            "drift_aligned_max": np.nan,
            "drift_raw_mean": np.nan,
            "drift_raw_max": np.nan,
            "pbc_detected": traj.unitcell_lengths is not None,
            "drift_aligned_per_frame": nan_arr,
            "drift_raw_per_frame": nan_arr,
            "n_receptor_ca": 0,
            "n_frames": traj.n_frames,
        }

    ca_idx_arr = np.array(ca_indices)

    # Aligned drift: MDTraj superpose on Cα and compute RMSD (nm → Å)
    drift_aligned = md.rmsd(traj, traj, reference_frame, atom_indices=ca_idx_arr) * 10.0

    # PBC detection
    pbc_detected = traj.unitcell_lengths is not None

    # Raw drift: per-frame RMSD of positions relative to reference frame (nm → Å)
    if pbc_detected:
        drift_raw = np.full(traj.n_frames, np.nan)
        drift_raw_mean = np.nan
        drift_raw_max = np.nan
    else:
        # xyz shape: (n_frames, n_atoms, 3) in nm
        pos_all = traj.xyz[:, ca_idx_arr, :]  # (n_frames, n_ca, 3)
        pos_ref = pos_all[reference_frame]     # (n_ca, 3)
        diff = pos_all - pos_ref[np.newaxis, :, :]  # (n_frames, n_ca, 3)
        drift_raw = np.sqrt(np.mean(np.sum(diff ** 2, axis=2), axis=1)) * 10.0
        drift_raw_mean = float(np.mean(drift_raw))
        drift_raw_max = float(np.max(drift_raw))

    return {
        # scores
        "drift_aligned_mean": float(np.mean(drift_aligned)),
        "drift_aligned_max": float(np.max(drift_aligned)),
        "drift_raw_mean": drift_raw_mean if not pbc_detected else np.nan,
        "drift_raw_max": drift_raw_max if not pbc_detected else np.nan,
        "pbc_detected": pbc_detected,
        # features
        "drift_aligned_per_frame": drift_aligned,
        "drift_raw_per_frame": drift_raw,
        "n_receptor_ca": len(ca_indices),
        "n_frames": traj.n_frames,
    }
