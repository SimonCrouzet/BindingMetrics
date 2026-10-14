"""Periodic-boundary policy shared by the trajectory metrics.

A trajectory that stores a unit cell is periodic. The MD engine keeps each molecule whole but
may write the receptor and the binder in different images of the box, so a raw coordinate
difference between them depends on the image. Pair separations, used by the pair energies,
take the minimum image of each pair, for orthorhombic and triclinic cells
(``ligand_receptor_distances_nm``).

A trajectory without unit-cell information is used as stored, with the same arithmetic as before
this module existed. A crystallographic cell (``CRYST1``) counts as a unit cell.

References:
    Tuckerman, Statistical Mechanics: Theory and Molecular Simulation (Oxford, 2010), scheme
        B.9: minimum image convention for triclinic cells, as implemented in mdtraj.
    McGibbon et al., Biophys. J. 109, 1528 (2015): mdtraj.
"""

from __future__ import annotations

import numpy as np


def has_unit_cell(traj) -> bool:
    """True when the trajectory stores a unit cell with positive lengths in every frame."""
    lengths = traj.unitcell_lengths
    return lengths is not None and bool(np.all(np.asarray(lengths) > 0))


def ligand_receptor_distances_nm(
    traj, frame_index: int, ligand_indices, receptor_indices
) -> np.ndarray:
    """Distances in nm between every ligand atom and every receptor atom of one frame.

    With a unit cell each distance is the minimum-image distance of the cell of that frame
    (mdtraj's ``compute_distances(periodic=True)``). Without one it is the norm of the stored
    coordinate difference.

    Args:
        traj: ``mdtraj.Trajectory``.
        frame_index: Frame to measure.
        ligand_indices: Atom indices of the ligand.
        receptor_indices: Atom indices of the receptor.

    Returns:
        Float32 array of shape (n_ligand, n_receptor); element [i, j] pairs
        ``ligand_indices[i]`` with ``receptor_indices[j]``.
    """
    ligand = np.asarray(ligand_indices, dtype=np.int64)
    receptor = np.asarray(receptor_indices, dtype=np.int64)
    if has_unit_cell(traj):
        import mdtraj as md

        pairs = np.stack(np.meshgrid(ligand, receptor, indexing="ij"), axis=-1).reshape(-1, 2)
        distances = md.compute_distances(traj[frame_index], pairs, periodic=True)[0]
        return distances.reshape(len(ligand), len(receptor))
    xyz = traj.xyz[frame_index]
    return np.linalg.norm(xyz[ligand][:, None, :] - xyz[receptor][None, :, :], axis=-1)
