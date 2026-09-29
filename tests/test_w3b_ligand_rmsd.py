"""Ligand RMSD is measured in the receptor frame (CAPRI L-RMSD idea).

Synthetic trajectories: a rigid "receptor" of 12 atoms and a 5-atom "ligand",
coordinates fixed by a seeded generator. Distances from mdtraj are in nm.
"""

from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from binding_metrics.metrics.rmsd import calculate_ligand_rmsd  # noqa: E402

N_RECEPTOR = 12
N_LIGAND = 5
RECEPTOR = list(range(N_RECEPTOR))
LIGAND = list(range(N_RECEPTOR, N_RECEPTOR + N_LIGAND))


def _reference_xyz() -> np.ndarray:
    """One frame, nm: receptor around the origin, ligand about 1 nm away."""
    rng = np.random.default_rng(7)
    receptor = rng.normal(scale=0.6, size=(N_RECEPTOR, 3))
    ligand = rng.normal(scale=0.3, size=(N_LIGAND, 3)) + np.array([1.5, 0.0, 0.0])
    return np.vstack([receptor, ligand]).astype(np.float32)


def _write(tmp_path: Path, frames: list[np.ndarray]) -> tuple[Path, Path]:
    """Save frames as a DCD plus a one-frame PDB topology, return their paths."""
    topology = md.Topology()
    receptor_chain = topology.add_chain()
    ligand_chain = topology.add_chain()
    for i in range(N_RECEPTOR + N_LIGAND):
        chain = receptor_chain if i < N_RECEPTOR else ligand_chain
        residue = topology.add_residue("ALA", chain, resSeq=i + 1)
        topology.add_atom("CA", md.element.carbon, residue)
    traj = md.Trajectory(np.stack(frames), topology)
    dcd, pdb = tmp_path / "synthetic.dcd", tmp_path / "synthetic.pdb"
    traj.save_dcd(str(dcd))
    traj[0].save_pdb(str(pdb))
    return dcd, pdb


def _rotation_z(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_identical_frames_give_zero(tmp_path):
    ref = _reference_xyz()
    dcd, pdb = _write(tmp_path, [ref, ref.copy(), ref.copy()])
    result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)
    np.testing.assert_allclose(result["ligand_rmsd"], 0.0, atol=1e-4)
    np.testing.assert_allclose(result["receptor_rmsd"], 0.0, atol=1e-4)


def test_ligand_moved_5_angstrom_relative_to_receptor(tmp_path):
    ref = _reference_xyz()
    moved = ref.copy()
    moved[N_RECEPTOR:] += np.array([0.0, 0.5, 0.0], dtype=np.float32)  # 5 A in nm
    dcd, pdb = _write(tmp_path, [ref, moved])

    result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)

    # A pure translation of every ligand atom by 0.5 nm has an RMSD of 0.5 nm.
    assert result["ligand_rmsd"][0] == pytest.approx(0.0, abs=1e-4)
    assert result["ligand_rmsd"][1] == pytest.approx(0.5, abs=5e-3)
    # The receptor did not move, so its fitted RMSD stays at zero.
    np.testing.assert_allclose(result["receptor_rmsd"], 0.0, atol=1e-4)


def test_rigid_motion_of_the_whole_complex_gives_zero(tmp_path):
    ref = _reference_xyz()
    rotation = _rotation_z(60.0).astype(np.float32)
    shift = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    moved = ref @ rotation.T + shift
    dcd, pdb = _write(tmp_path, [ref, moved])

    result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)

    np.testing.assert_allclose(result["ligand_rmsd"], 0.0, atol=2e-3)
    np.testing.assert_allclose(result["receptor_rmsd"], 0.0, atol=2e-3)


def test_ligand_moved_together_with_a_rotating_complex_still_shows_displacement(tmp_path):
    """Whole complex rotates and the ligand also slides 3 A: L-RMSD is 3 A."""
    ref = _reference_xyz()
    rotation = _rotation_z(-40.0).astype(np.float32)
    moved = ref @ rotation.T + np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Slide the ligand along the rotated x axis by 0.3 nm.
    slide = (np.array([0.3, 0.0, 0.0]) @ rotation.T).astype(np.float32)
    moved[N_RECEPTOR:] += slide
    dcd, pdb = _write(tmp_path, [ref, moved])

    result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)

    assert result["ligand_rmsd"][1] == pytest.approx(0.3, abs=5e-3)


def test_reference_frame_can_be_any_frame(tmp_path):
    ref = _reference_xyz()
    moved = ref.copy()
    moved[N_RECEPTOR:] += np.array([0.0, 0.0, 0.4], dtype=np.float32)
    dcd, pdb = _write(tmp_path, [ref, moved, ref.copy()])

    result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR, reference_frame=1)

    np.testing.assert_allclose(result["ligand_rmsd"], [0.4, 0.0, 0.4], atol=5e-3)
