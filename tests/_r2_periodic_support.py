"""Builders shared by the tests of the periodic-image policy (#169).

Every trajectory is made from one reference frame by translating the ligand by lattice vectors
of an orthorhombic or a triclinic cell, so the physical configuration is the same in each frame.
"""

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from binding_metrics import _constants  # noqa: E402

# Cell lengths (nm) and angles (degrees). Every length is at least twice the largest separation
# of the system it is used with, in the triclinic cell too.
CELLS = {
    "orthorhombic": ((6.0, 6.5, 7.0), (90.0, 90.0, 90.0)),
    "triclinic": ((6.0, 6.5, 7.0), (80.0, 70.0, 75.0)),
    # Smaller than the dipeptide pair: some pairs are farther apart than half an edge.
    "snug": ((2.2, 2.2, 2.2), (90.0, 90.0, 90.0)),
    # For the bundled protein complexes, which are about 4 nm across.
    "orthorhombic_large": ((12.0, 12.5, 13.0), (90.0, 90.0, 90.0)),
    "triclinic_large": ((12.0, 12.5, 13.0), (80.0, 70.0, 75.0)),
}
CELL_NAMES = ["orthorhombic", "triclinic"]
LARGE_CELL_NAMES = ["orthorhombic_large", "triclinic_large"]
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


def write_dipeptides(path, sample_pdb_content: str, shifts_angstrom):
    """PDB of alanine dipeptides, chains A, B, ...: the sample's chain A shifted by each vector."""
    chain_a = [
        line
        for line in sample_pdb_content.splitlines()
        if line.startswith("ATOM") and line[21] == "A"
    ]
    lines = []
    for chain_number, shift in enumerate(shifts_angstrom):
        chain_id = "ABCDEFGH"[chain_number]
        for position, line in enumerate(chain_a):
            serial = chain_number * len(chain_a) + position + 1
            x, y, z = (float(line[start : start + 8]) + d for start, d in zip((30, 38, 46), shift))
            lines.append(
                f"{line[:6]}{serial:5d}{line[11:21]}{chain_id}{line[22:30]}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            )
        lines.append("TER")
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="utf-8")
    return path


def write_dipeptide_pair(path, sample_pdb_content: str):
    """Chain A and a copy 5 A off along each axis (nearest atoms 0.30 nm apart)."""
    return write_dipeptides(path, sample_pdb_content, [(0, 0, 0), (5, 5, 5)])


def write_dipeptide_trio(path, sample_pdb_content: str):
    """The pair plus a third copy, 0.27 nm from chain A and 0.64 nm from chain B."""
    return write_dipeptides(path, sample_pdb_content, [(0, 0, 0), (5, 5, 5), (-5, -5, 5)])


def chain_atom_indices(topology) -> list[list[int]]:
    """Atom indices of each chain."""
    return [[atom.index for atom in chain.atoms] for chain in topology.chains]


def peptide_pair(pair_path):
    """Topology, coordinates (nm), ligand and receptor indices of a dipeptide pair or trio.

    The ligand is chain B and the receptor is chain A, plus chain C when there is one.
    """
    reference = md.load(str(pair_path))
    chains = chain_atom_indices(reference.topology)
    receptor = [*chains[0], *(chains[2] if len(chains) > 2 else [])]
    return reference.topology, reference.xyz[0].astype(float), chains[1], receptor


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


def pair_sum_reference_kj_mol(pdb_path, xyz_nm, ligand, receptor):
    """Coulomb, Lennard-Jones and total of the ligand x receptor pair sum, in float64."""
    import openmm
    import openmm.unit as unit
    from openmm.app import PDBFile

    from binding_metrics.core.forcefields import get_forcefield

    pdb = PDBFile(str(pdb_path))
    system = get_forcefield("amber").createSystem(
        pdb.topology, nonbondedMethod=openmm.app.NoCutoff, constraints=None
    )
    force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))

    def parameters(indices):
        rows = [force.getParticleParameters(i) for i in indices]
        return (
            np.array([q.value_in_unit(unit.elementary_charge) for q, _, _ in rows]),
            np.array([s.value_in_unit(unit.nanometer) for _, s, _ in rows]),
            np.array([e.value_in_unit(unit.kilojoule_per_mole) for _, _, e in rows]),
        )

    q_l, s_l, e_l = parameters(ligand)
    q_r, s_r, e_r = parameters(receptor)
    delta = xyz_nm[ligand][:, None, :].astype(float) - xyz_nm[receptor][None, :, :].astype(float)
    r = np.linalg.norm(delta, axis=-1)
    coulomb = _constants.COULOMB_KJ_NM_MOL_E2 * q_l[:, None] * q_r[None, :] / r
    sigma = (s_l[:, None] + s_r[None, :]) / 2
    epsilon = np.sqrt(e_l[:, None] * e_r[None, :])
    sr6 = (sigma / r) ** 6
    lennard_jones = np.where((epsilon > 0) & (sigma > 0), 4 * epsilon * (sr6**2 - sr6), 0.0)
    return coulomb.sum(), lennard_jones.sum(), (coulomb + lennard_jones).sum()
