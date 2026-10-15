"""Trajectory SASA of a frame does not depend on the other frames or the thread count (#201)."""

import os
import subprocess
import sys

import numpy as np
import pytest
from _r2_periodic_support import carbon_pair_topology, write_trajectory

from binding_metrics.metrics.sasa import calculate_buried_sasa

NEAR_NM = [[1.0, 1.0, 1.0], [1.4, 1.0, 1.0]]
FAR_NM = [[1.0, 1.0, 1.0], [7.4, 1.0, 1.0]]

SCRIPT = """
import sys
import numpy as np
from binding_metrics.metrics.sasa import calculate_buried_sasa
values = calculate_buried_sasa(sys.argv[1], sys.argv[2], ligand_indices=[1], receptor_indices=[0])
print("RESULT", *(repr(float(v)) for v in values))
"""


def _frames(order):
    return np.array([order[0], order[1]], dtype=np.float32)


@pytest.mark.parametrize("order", [(NEAR_NM, FAR_NM), (FAR_NM, NEAR_NM)])
def test_buried_sasa_of_a_frame_is_the_same_with_one_thread_and_with_two(tmp_path, order):
    dcd, pdb = write_trajectory(tmp_path, carbon_pair_topology(), _frames(order), cell=None)
    results = {}
    for threads in ("1", "2"):
        run = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(dcd), str(pdb)],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": threads},
            check=True,
        )
        line = next(x for x in run.stdout.splitlines() if x.startswith("RESULT "))
        results[threads] = np.array([float(v) for v in line.split()[1:]])

    assert results["1"] == pytest.approx(results["2"], abs=1e-7)
    far = 1 if order[0] is NEAR_NM else 0
    assert results["1"][far] == pytest.approx(0.0, abs=1e-6)


def test_the_in_process_values_match_the_frame_by_frame_areas(tmp_path):
    dcd, pdb = write_trajectory(
        tmp_path, carbon_pair_topology(), _frames((NEAR_NM, FAR_NM)), cell=None
    )

    buried = calculate_buried_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

    assert buried[1] == pytest.approx(0.0, abs=1e-6)


REFERENCE_SCRIPT = """
import sys
import mdtraj as md
import numpy as np
from binding_metrics.metrics.sasa import calculate_buried_sasa

traj = md.load(sys.argv[1], top=sys.argv[2])
receptor = [a.index for a in traj.topology.atoms if a.residue.chain.index == 0]
ligand = [a.index for a in traj.topology.atoms if a.residue.chain.index == 1]


def area(frame, atoms):
    return md.shrake_rupley(frame.atom_slice(atoms)).sum()


reference = [
    area(traj[i], ligand) + area(traj[i], receptor) - area(traj[i], ligand + receptor)
    for i in range(traj.n_frames)
]
values = calculate_buried_sasa(
    sys.argv[1], sys.argv[2], ligand_indices=ligand, receptor_indices=receptor
)
print("RESULT", float(np.abs(np.asarray(values) - np.asarray(reference)).max()))
"""


def test_a_protein_trajectory_with_more_frames_than_threads_gives_the_frame_by_frame_areas(
    tmp_path,
):
    import mdtraj as md

    from binding_metrics.testing import bundled_complex_path

    start = md.load(str(bundled_complex_path()))
    noise = np.random.default_rng(0).normal(0, 0.03, (5, *start.xyz.shape[1:]))
    frames = (start.xyz[0] + noise).astype(np.float32)
    dcd, pdb = tmp_path / "frames.dcd", tmp_path / "frames.pdb"
    md.Trajectory(frames, start.topology).save_dcd(str(dcd))
    start[0].save_pdb(str(pdb))

    run = subprocess.run(
        [sys.executable, "-c", REFERENCE_SCRIPT, str(dcd), str(pdb)],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "2"},
        check=True,
    )
    line = next(x for x in run.stdout.splitlines() if x.startswith("RESULT "))

    assert float(line.split()[1]) < 1e-5
