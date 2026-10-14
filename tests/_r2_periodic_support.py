"""Builders shared by the tests of the periodic-image policy (#169).

Every trajectory is made from one reference frame by translating the ligand by lattice vectors
of an orthorhombic or a triclinic cell, so the physical configuration is the same in each frame.
"""

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

# Cell lengths (nm) and angles (degrees). Every length is at least twice the largest separation
# used below, in the triclinic cell too.
CELLS = {
    "orthorhombic": ((6.0, 6.5, 7.0), (90.0, 90.0, 90.0)),
    "triclinic": ((6.0, 6.5, 7.0), (80.0, 70.0, 75.0)),
}
CELL_NAMES = sorted(CELLS)
# Integer combinations of the three cell vectors.
LATTICE_COMBINATIONS = ((1, 0, 0), (0, -1, 0), (0, 0, 1), (1, -1, 1))

ION_DISTANCE_NM = 0.4
ION_PAIR_NM = [[1.0, 1.0, 1.0], [1.0 + ION_DISTANCE_NM, 1.0, 1.0]]
# Carbon radius of mdtraj plus the default probe radius; two such spheres at 0.40 nm bury
# 2 pi R (2 R - d) = 0.4285 nm^2.
TWO_SPHERE_BURIED_NM2 = 0.4285
TWO_SPHERE_TOLERANCE_NM2 = 0.005

N_RECEPTOR = 12
N_LIGAND = 5
RECEPTOR = list(range(N_RECEPTOR))
LIGAND = list(range(N_RECEPTOR, N_RECEPTOR + N_LIGAND))
# Two copies of the alanine dipeptide of conftest's sample, the second 5 A off along each axis:
# the nearest atoms are 0.30 nm apart.
SECOND_DIPEPTIDE_SHIFT_ANGSTROM = 5.0


def cell_vectors_nm(cell: str) -> np.ndarray:
    lengths, angles = CELLS[cell]
    return np.array(md.utils.lengths_and_angles_to_box_vectors(*lengths, *angles), dtype=float)


def lattice_offsets_nm(cell: str) -> list[np.ndarray]:
    vectors = cell_vectors_nm(cell)
    return [np.array(combination) @ vectors for combination in LATTICE_COMBINATIONS]


def frames_with_images(reference_xyz, moved_indices, cell: str) -> list[np.ndarray]:
    """The reference frame, then one frame per lattice offset with ``moved_indices`` translated."""
    frames = [np.array(reference_xyz, dtype=float)]
    for offset in lattice_offsets_nm(cell):
        frame = np.array(reference_xyz, dtype=float)
        frame[list(moved_indices)] += offset
        frames.append(frame)
    return frames


def write_trajectory(tmp_path, topology, frames, cell=None, topology_path=None):
    """Save frames as a DCD (with the cell when ``cell`` is given) plus a PDB topology."""
    kwargs = {}
    if cell is not None:
        lengths, angles = CELLS[cell]
        kwargs = {
            "unitcell_lengths": np.tile(lengths, (len(frames), 1)),
            "unitcell_angles": np.tile(angles, (len(frames), 1)),
        }
    traj = md.Trajectory(np.asarray(frames, dtype=np.float32), topology, **kwargs)
    dcd = tmp_path / "frames.dcd"
    traj.save_dcd(str(dcd))
    if topology_path is None:
        topology_path = tmp_path / "topology.pdb"
        traj[0].save_pdb(str(topology_path))
    return dcd, topology_path


def ion_topology() -> "md.Topology":
    """Atom 0 is Na+ (ligand) and atom 1 is Cl- (receptor)."""
    topology = md.Topology()
    for name, element in (("NA", md.element.sodium), ("CL", md.element.chlorine)):
        residue = topology.add_residue(name, topology.add_chain())
        topology.add_atom(name, element, residue)
    return topology


def carbon_pair_topology() -> "md.Topology":
    """Atom 0 is the receptor carbon and atom 1 the ligand carbon."""
    topology = md.Topology()
    for name in ("REC", "LIG"):
        residue = topology.add_residue(name, topology.add_chain())
        topology.add_atom("C", md.element.carbon, residue)
    return topology


def write_dipeptide_pair(path, sample_pdb_content: str):
    """PDB with the alanine dipeptide of the sample as chain A and a shifted copy as chain B."""
    chain_a = [
        line
        for line in sample_pdb_content.splitlines()
        if line.startswith("ATOM") and line[21] == "A"
    ]
    shift = SECOND_DIPEPTIDE_SHIFT_ANGSTROM
    chain_b = []
    for serial, line in enumerate(chain_a, start=len(chain_a) + 1):
        x, y, z = (float(line[start : start + 8]) + shift for start in (30, 38, 46))
        chain_b.append(
            f"{line[:6]}{serial:5d}{line[11:21]}B{line[22:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        )
    lines = [*chain_a, "TER", *chain_b, "TER", "END"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def peptide_pair(pair_path):
    """Topology, coordinates (nm), ligand and receptor indices of the dipeptide pair."""
    reference = md.load(str(pair_path))
    chains = list(reference.topology.chains)
    receptor = [atom.index for atom in chains[0].atoms]
    ligand = [atom.index for atom in chains[1].atoms]
    return reference.topology, reference.xyz[0].astype(float), ligand, receptor


def wrap_into_cell(xyz, cell: str) -> np.ndarray:
    """Wrap each atom into the primary cell: molecules that straddle a face are cut apart."""
    vectors = cell_vectors_nm(cell)
    fractional = xyz @ np.linalg.inv(vectors)
    return (fractional - np.floor(fractional)) @ vectors


def rigid_bodies_topology() -> "md.Topology":
    """A 12-atom receptor chain and a 5-atom ligand chain, one residue per atom, no bonds."""
    topology = md.Topology()
    receptor_chain = topology.add_chain()
    ligand_chain = topology.add_chain()
    for i in range(N_RECEPTOR + N_LIGAND):
        chain = receptor_chain if i < N_RECEPTOR else ligand_chain
        residue = topology.add_residue("ALA", chain, resSeq=i + 1)
        topology.add_atom("CA", md.element.carbon, residue)
    return topology


def rigid_bodies_xyz() -> np.ndarray:
    """Receptor around the origin, ligand about 1.5 nm away."""
    rng = np.random.default_rng(7)
    receptor = rng.normal(scale=0.6, size=(N_RECEPTOR, 3))
    ligand = rng.normal(scale=0.3, size=(N_LIGAND, 3)) + np.array([1.5, 0.0, 0.0])
    return np.vstack([receptor, ligand])


def trajectory_with_cell(topology, frames, cell: str):
    """In-memory ``mdtraj.Trajectory`` of the frames, every frame in the given cell."""
    lengths, angles = CELLS[cell]
    return md.Trajectory(
        np.asarray(frames, dtype=np.float32),
        topology,
        unitcell_lengths=np.tile(lengths, (len(frames), 1)),
        unitcell_angles=np.tile(angles, (len(frames), 1)),
    )
