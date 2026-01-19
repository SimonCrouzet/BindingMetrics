"""Interface contact analysis."""

from pathlib import Path

import numpy as np

try:
    import mdtraj as md
except ImportError:
    md = None


def calculate_contacts(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    cutoff: float = 0.45,
) -> np.ndarray:
    """Calculate the number of interface contacts per frame.

    A contact is defined as any heavy atom pair (ligand-receptor)
    within the cutoff distance.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        cutoff: Distance cutoff in nm (default 0.45 nm = 4.5 A)

    Returns:
        Array of contact counts for each frame
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for contact analysis. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    # Build pairs of ligand-receptor atoms
    pairs = []
    for lig_idx in ligand_indices:
        for rec_idx in receptor_indices:
            pairs.append((lig_idx, rec_idx))

    if not pairs:
        return np.zeros(traj.n_frames)

    pairs = np.array(pairs)

    # Calculate distances
    distances = md.compute_distances(traj, pairs)

    # Count contacts per frame
    contacts = (distances < cutoff).sum(axis=1)

    return contacts.astype(np.float64)


def calculate_contact_residues(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    cutoff: float = 0.45,
) -> dict:
    """Identify residues involved in interface contacts.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        cutoff: Distance cutoff in nm

    Returns:
        Dictionary with contact residue information
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for contact analysis. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))
    topology = traj.topology

    # Map atom indices to residues
    ligand_residues = set()
    receptor_residues = set()
    contact_pairs = []

    # Build pairs and track residues
    pairs = []
    pair_residue_map = []

    for lig_idx in ligand_indices:
        lig_atom = topology.atom(lig_idx)
        lig_res = (lig_atom.residue.chain.index, lig_atom.residue.index, lig_atom.residue.name)

        for rec_idx in receptor_indices:
            rec_atom = topology.atom(rec_idx)
            rec_res = (rec_atom.residue.chain.index, rec_atom.residue.index, rec_atom.residue.name)

            pairs.append((lig_idx, rec_idx))
            pair_residue_map.append((lig_res, rec_res))

    if not pairs:
        return {"ligand_residues": [], "receptor_residues": [], "contact_frequency": {}}

    pairs = np.array(pairs)

    # Calculate distances for all frames
    distances = md.compute_distances(traj, pairs)

    # Find contacts and their frequencies
    contact_freq = {}
    for i, (lig_res, rec_res) in enumerate(pair_residue_map):
        n_contacts = (distances[:, i] < cutoff).sum()
        if n_contacts > 0:
            ligand_residues.add(lig_res)
            receptor_residues.add(rec_res)
            key = (lig_res, rec_res)
            contact_freq[key] = contact_freq.get(key, 0) + n_contacts

    # Normalize by number of frames
    n_frames = traj.n_frames
    contact_freq = {k: v / n_frames for k, v in contact_freq.items()}

    return {
        "ligand_residues": list(ligand_residues),
        "receptor_residues": list(receptor_residues),
        "contact_frequency": contact_freq,
    }
