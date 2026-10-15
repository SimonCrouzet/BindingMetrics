"""``PeptideBindingProtocol.analyze`` hands mdtraj an ascending atom selection.

In the bundled 1YCR file the ligand (chain B) follows the receptor (chain A), so the list
``ligand_indices + receptor_indices`` descends where the two lists meet and mdtraj warns
"atom_indices are not monotonically increasing" on every call. ``Trajectory.atom_slice`` sorts the
atoms it keeps, so the sorted selection gives the same coordinates: the warning goes away and the
values do not change.

The trajectory is built by hand from the bundled structure (four noisy frames, seeded), without
a simulation. The interaction energy needs OpenMM and a force field and does not take part in the
selection, so it is replaced; the SASA, the contacts and the RMSD are the real functions.
"""

import warnings

import numpy as np
import pytest
from conftest import DATA_DIR

from binding_metrics.metrics.rmsd import calculate_rmsd
from binding_metrics.protocols import peptide

md = pytest.importorskip("mdtraj")

EXAMPLE = DATA_DIR / "example_linear_p53_1YCR.pdb"
N_FRAMES = 4
NOISE_NM = 0.02
CELL_NM = 8.0
SEED = 0


def build_trajectory(tmp_path, *, with_cell: bool):
    """Write ``frames.dcd`` and ``topology.pdb`` of the bundled 1YCR complex, noisy frames."""
    reference = md.load(str(EXAMPLE))
    rng = np.random.default_rng(SEED)
    noise = rng.normal(scale=NOISE_NM, size=(N_FRAMES,) + reference.xyz.shape[1:])
    cell = {}
    if with_cell:
        cell = {
            "unitcell_lengths": np.full((N_FRAMES, 3), CELL_NM),
            "unitcell_angles": np.full((N_FRAMES, 3), 90.0),
        }
    traj = md.Trajectory((reference.xyz + noise).astype(np.float32), reference.topology, **cell)
    dcd, pdb = tmp_path / "frames.dcd", tmp_path / "topology.pdb"
    traj.save_dcd(str(dcd))
    traj[0].save_pdb(str(pdb))
    return dcd, pdb


@pytest.fixture(params=[False, True], ids=["no_cell", "with_cell"])
def protocol_and_files(request, tmp_path, monkeypatch):
    dcd, pdb = build_trajectory(tmp_path, with_cell=request.param)
    frames = np.zeros(N_FRAMES)
    monkeypatch.setattr(peptide, "calculate_interaction_energy", lambda *a, **k: frames)
    protocol = peptide.PeptideBindingProtocol(pdb_path=pdb, ligand_chain="B", receptor_chains=["A"])
    protocol._topology_path = pdb  # what run() sets
    return protocol, dcd, pdb


def analysis_warnings(protocol, dcd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = protocol.analyze(trajectory_path=dcd)
    return results, [str(w.message) for w in caught]


class TestTheSelectionIsAscending:
    def test_the_ligand_follows_the_receptor_in_the_file(self, protocol_and_files):
        protocol, _, pdb = protocol_and_files

        ligand, receptor = protocol._partner_atom_indices(pdb)

        assert min(ligand) > max(receptor)
        assert ligand + receptor != sorted(ligand + receptor)

    def test_analyze_emits_no_monotonicity_warning(self, protocol_and_files):
        protocol, dcd, _ = protocol_and_files

        _, messages = analysis_warnings(protocol, dcd)

        assert [m for m in messages if "monotonically" in m] == []

    def test_the_rmsd_is_that_of_the_unsorted_selection_bit_for_bit(self, protocol_and_files):
        protocol, dcd, pdb = protocol_and_files
        ligand, receptor = protocol._partner_atom_indices(pdb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unsorted_selection = calculate_rmsd(dcd, pdb, atom_indices=ligand + receptor)

        results, _ = analysis_warnings(protocol, dcd)

        np.testing.assert_array_equal(results.rmsd, unsorted_selection)
        assert results.rmsd[0] == 0.0
        # Two independent frames with an isotropic noise of NOISE_NM per coordinate differ by
        # sqrt(2 * 3) * NOISE_NM per atom, a little less after the fit.
        assert results.rmsd[1:] == pytest.approx(NOISE_NM * np.sqrt(6), rel=0.1)
        assert results.rmsd_mean == pytest.approx(float(np.mean(unsorted_selection)))
