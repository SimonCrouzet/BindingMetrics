"""Periodic-boundary policy shared by the trajectory metrics.

A trajectory that stores a unit cell is periodic. The MD engine keeps each molecule whole but
may write the receptor and the binder in different images of the box, so a raw coordinate
difference between them depends on the image. The pair energies, the SASA and the
receptor-frame RMSD use one image of the complex: molecules are made whole and the ligand is
placed in the image next to the receptor (``image_complex``), then plain distances and areas
are taken. A pair farther apart than half a cell edge keeps its distance in the complex.

A trajectory without unit-cell information is used as stored. A crystallographic cell
(``CRYST1``) counts as a unit cell.

References:
    McGibbon et al., Biophys. J. 109, 1528 (2015): mdtraj, whose ``image_molecules`` does the
        imaging.
"""

from __future__ import annotations

import numpy as np


def has_unit_cell(traj) -> bool:
    """True when the trajectory stores a unit cell with positive lengths in every frame."""
    lengths = traj.unitcell_lengths
    return lengths is not None and bool(np.all(np.asarray(lengths) > 0))


def ligand_receptor_distances_nm(traj, ligand_indices, receptor_indices):
    """Yield, per frame, the distances in nm between every ligand atom and every receptor atom.

    With a unit cell, the complex is imaged first (``image_complex``) and the distances are
    plain norms in that image; without one they are the norms of the stored coordinates.

    Args:
        traj: ``mdtraj.Trajectory``.
        ligand_indices: Atom indices of the ligand.
        receptor_indices: Atom indices of the receptor.

    Yields:
        Float64 array of shape (n_ligand, n_receptor); element [i, j] pairs
        ``ligand_indices[i]`` with ``receptor_indices[j]``. The float32 coordinates are cast
        before the differences, so the pair sums that use them are float64 on every NumPy.
    """
    ligand = np.asarray(ligand_indices, dtype=np.int64)
    receptor = np.asarray(receptor_indices, dtype=np.int64)
    if len(ligand) and len(receptor) and has_unit_cell(traj):
        traj, ligand, receptor = image_complex(traj, ligand, receptor)
    for frame_index in range(traj.n_frames):
        xyz = traj.xyz[frame_index].astype(np.float64)
        yield np.linalg.norm(xyz[ligand][:, None, :] - xyz[receptor][None, :, :], axis=-1)


def image_complex(traj, ligand_indices, receptor_indices):
    """Copy of the two selections with every molecule whole and the ligand next to the receptor.

    The copy holds only the atoms of the two selections, in index order, and keeps the unit
    cell. When the trajectory has a cell, the molecules of the copy (connected components of the
    topology bonds) are made whole with mdtraj's ``image_molecules``; the molecules that hold a
    receptor atom are the anchors and every other molecule goes to the image that mdtraj
    places next to them by the molecule centre. A ligand covalently bonded to the receptor is an
    anchor itself. The result is deterministic and the given trajectory is not modified.
    Without a cell, or with an empty selection, the atoms are returned as stored.

    Limit: a molecule whose centre is more than half a cell edge from the anchors goes to the
    image on the other side, and atoms that the topology does not bond are placed one by one.

    Args:
        traj: ``mdtraj.Trajectory``.
        ligand_indices: Atom indices of the ligand.
        receptor_indices: Atom indices of the receptor.

    Returns:
        Tuple ``(imaged, ligand, receptor)``: the new trajectory and the positions of the
        ligand and receptor atoms in it, in the order given.
    """
    union = np.array(
        sorted({int(i) for i in ligand_indices} | {int(i) for i in receptor_indices}), dtype=int
    )
    imaged = traj.atom_slice(union)
    ligand = np.searchsorted(union, np.asarray(ligand_indices, dtype=int))
    receptor = np.searchsorted(union, np.asarray(receptor_indices, dtype=int))
    if len(ligand) == 0 or len(receptor) == 0 or not has_unit_cell(imaged):
        return imaged, ligand, receptor

    receptor_atoms = set(receptor.tolist())
    anchors, others = [], []
    for molecule in imaged.topology.find_molecules():
        holds_receptor = any(atom.index in receptor_atoms for atom in molecule)
        (anchors if holds_receptor else others).append(molecule)
    _image_in_place(imaged, anchors, others)
    return imaged, ligand, receptor


def image_selection(traj, atom_indices):
    """Copy of one selection with every molecule whole and the molecules gathered.

    For a selection that is not split into receptor and ligand, such as the protein atoms of a
    complex. The copy holds the selected atoms in file order (``atom_slice`` sorts them). When
    the trajectory has a cell, the largest molecule of the copy (the lowest first atom index
    among equals) is the anchor and the other molecules go to the image next to it, with the
    rule and the limit of ``image_complex``. Without a cell, or with an empty selection, the
    atoms are returned as stored.

    Args:
        traj: ``mdtraj.Trajectory``.
        atom_indices: Atom indices of the selection.

    Returns:
        New ``mdtraj.Trajectory``; the given trajectory is not modified.
    """
    imaged = traj.atom_slice(atom_indices)
    if imaged.n_atoms == 0 or not has_unit_cell(imaged):
        return imaged
    molecules = imaged.topology.find_molecules()
    anchor = max(molecules, key=lambda molecule: (len(molecule), -min(a.index for a in molecule)))
    others = [molecule for molecule in molecules if molecule is not anchor]
    _image_in_place(imaged, [anchor], others)
    return imaged


def _image_in_place(traj, anchors, others) -> None:
    """Make the molecules whole and place ``others`` next to ``anchors``, in ``traj``."""
    # mdtraj fails on the empty bond list of a topology without bonds; nothing to make whole there.
    traj.image_molecules(
        inplace=True,
        anchor_molecules=anchors,
        other_molecules=others,
        make_whole=traj.topology.n_bonds > 0,
    )
