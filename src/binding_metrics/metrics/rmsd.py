"""RMSD calculations for structural stability analysis.

RMSDs are computed with mdtraj (McGibbon et al., 2015, Biophys. J. 109, 1528),
whose ``rmsd`` superposes every frame on the reference with the QCP algorithm
(Theobald, 2005, Acta Cryst. A61, 478), so values are free of overall rotation
and translation. ``calculate_ligand_rmsd`` is the exception: it fits on the
receptor and leaves the ligand unfitted. Distances are in nm unless a function
says otherwise.
"""

import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from binding_metrics.metrics._common import resolve_chain_role

try:
    import mdtraj as md
except ImportError:
    md = None

_ON_EMPTY_MODES = ("warn", "raise")


def _report_empty_selection(message: str, on_empty: str) -> None:
    """Warn or raise for an atom selection that matched nothing.

    A zero-filled result reads as a perfect fit or as no contacts, so the
    caller has to be told that nothing was evaluated. ``stacklevel=3`` points
    the warning at the caller of the public function.

    Raises:
        ValueError: If ``on_empty`` is "raise" (message is the error text).
    """
    if on_empty == "raise":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _check_on_empty(on_empty: str) -> None:
    if on_empty not in _ON_EMPTY_MODES:
        raise ValueError(f"on_empty must be one of {_ON_EMPTY_MODES}, got {on_empty!r}")


def calculate_rmsd(
    trajectory_path: str | Path,
    topology_path: str | Path,
    atom_indices: list[int] | None = None,
    reference_frame: int = 0,
    *,
    on_empty: Literal["warn", "raise"] = "warn",
) -> np.ndarray:
    """Calculate RMSD relative to reference frame.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        atom_indices: Atom indices to include in RMSD calculation.
            If None, uses the mdtraj selection ``protein and not type H``.
            That selection knows the standard residue names only, so a chain
            made of unrecognised non-canonical residues matches nothing.
        reference_frame: Frame index to use as reference (default 0)
        on_empty: What to do when the selection contains no atoms. "warn"
            (default) emits a ``RuntimeWarning`` naming the selection and
            returns zeros, the historical result, which is not a real RMSD;
            "raise" raises ``ValueError`` instead.

    Returns:
        Array of RMSD values (in nm) for each frame

    Raises:
        ValueError: If ``on_empty`` is not "warn" or "raise", or is "raise"
            and the selection is empty.
    """
    _check_on_empty(on_empty)
    if md is None:
        raise ImportError(
            "mdtraj is required for RMSD calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    if atom_indices is None:
        selection = "default selection 'protein and not type H'"
        atom_indices = traj.topology.select("protein and not type H")
    else:
        selection = "atom_indices"

    if len(atom_indices) == 0:
        _report_empty_selection(
            f"calculate_rmsd: {selection} matched no atoms; returning zeros for "
            f"{traj.n_frames} frames, which is not an RMSD",
            on_empty,
        )
        return np.zeros(traj.n_frames)

    traj_subset = traj.atom_slice(atom_indices)

    # md.rmsd fits each frame onto the reference itself, so the RMSD does not
    # depend on how the frames are oriented in the file.
    traj_subset.superpose(traj_subset, frame=reference_frame)
    rmsd = md.rmsd(traj_subset, traj_subset, frame=reference_frame)

    return rmsd


def calculate_rmsf(
    trajectory_path: str | Path,
    topology_path: str | Path,
    atom_indices: list[int] | None = None,
) -> np.ndarray:
    """Calculate root mean square fluctuation per atom.

    Frames are superposed on frame 0 and the fluctuation is taken about the
    mean position of each atom after that fit.

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

    # Fitted on frame 0, not iteratively on the mean structure: a frame-0 fit
    # that is far from the mean adds to the fluctuation of every atom.
    traj_subset.superpose(traj_subset, frame=0)

    mean_positions = traj_subset.xyz.mean(axis=0)
    diff = traj_subset.xyz - mean_positions
    rmsf = np.sqrt((diff**2).sum(axis=2).mean(axis=0))

    return rmsf


def calculate_ligand_rmsd(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    reference_frame: int = 0,
) -> dict[str, np.ndarray]:
    """Calculate RMSD for ligand after aligning on receptor.

    Every frame is superposed on the reference frame using the receptor atoms
    only. The ligand RMSD is then the plain root-mean-square displacement of
    the ligand atoms in that receptor frame, with no further fit, as in the
    CAPRI ligand RMSD (Mendez et al., 2003, Proteins 52, 51). It therefore
    includes the rigid-body motion of the ligand relative to the receptor and
    is 0 only when the ligand keeps its pose in the receptor frame.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor (for alignment)
        reference_frame: Frame index to use as reference

    Returns:
        Dictionary with 'ligand_rmsd' (receptor-frame ligand displacement,
        nm) and 'receptor_rmsd' (receptor RMSD after its own fit, nm) arrays,
        one value per frame.
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

    # md.rmsd would fit the ligand onto its own reference again and hide any
    # displacement relative to the receptor, so the RMSD is taken directly on
    # the receptor-aligned coordinates.
    ligand_xyz = traj.xyz[:, np.asarray(ligand_indices, dtype=int), :]
    ligand_displacement = ligand_xyz - ligand_xyz[reference_frame]
    ligand_rmsd = np.sqrt((ligand_displacement**2).sum(axis=2).mean(axis=1))

    return {
        "ligand_rmsd": ligand_rmsd,
        "receptor_rmsd": receptor_rmsd,
    }


def compute_receptor_drift(
    trajectory_path: str | Path,
    topology_path: str | Path,
    receptor_chain: Optional[str] = None,
    reference_frame: int = 0,
    *,
    target_chain: Optional[str] = None,
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
        receptor_chain: Chain ID of the receptor (e.g. "A"). Required,
            through this parameter or ``target_chain``.
        reference_frame: Frame index to use as reference (default 0)
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

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
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    if md is None:
        raise ImportError(
            "mdtraj is required for receptor drift calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

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

    # A stored unit cell means the coordinates may be wrapped, which makes raw
    # displacements meaningless.
    pbc_detected = traj.unitcell_lengths is not None

    # Raw drift: per-frame RMSD of positions relative to reference frame (nm → Å)
    if pbc_detected:
        drift_raw = np.full(traj.n_frames, np.nan)
        drift_raw_mean = np.nan
        drift_raw_max = np.nan
    else:
        # xyz shape: (n_frames, n_atoms, 3) in nm
        pos_all = traj.xyz[:, ca_idx_arr, :]  # (n_frames, n_ca, 3)
        pos_ref = pos_all[reference_frame]  # (n_ca, 3)
        diff = pos_all - pos_ref[np.newaxis, :, :]  # (n_frames, n_ca, 3)
        drift_raw = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1)) * 10.0
        drift_raw_mean = float(np.mean(drift_raw))
        drift_raw_max = float(np.max(drift_raw))

    return {
        "drift_aligned_mean": float(np.mean(drift_aligned)),
        "drift_aligned_max": float(np.max(drift_aligned)),
        "drift_raw_mean": drift_raw_mean if not pbc_detected else np.nan,
        "drift_raw_max": drift_raw_max if not pbc_detected else np.nan,
        "pbc_detected": pbc_detected,
        "drift_aligned_per_frame": drift_aligned,
        "drift_raw_per_frame": drift_raw,
        "n_receptor_ca": len(ca_indices),
        "n_frames": traj.n_frames,
    }
