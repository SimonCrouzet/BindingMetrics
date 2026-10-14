"""Pair energies of a trajectory whose partners sit in different periodic images (#169).

The pair energy of a trajectory with a unit cell uses minimum-image separations, so translating
one partner by a lattice vector of an orthorhombic or a triclinic cell leaves it unchanged. The
ion pair has an analytic Coulomb term, -k / d for charges +1 and -1. The contact count already
used the minimum image; its tests guard that.
"""

import numpy as np
import pytest
from _r2_periodic_support import (
    CELL_NAMES,
    CELLS,
    ION_DISTANCE_NM,
    ION_PAIR_NM,
    carbon_pair_topology,
    frames_with_images,
    ion_topology,
    peptide_pair,
    write_dipeptide_pair,
    write_trajectory,
)

md = pytest.importorskip("mdtraj")

from binding_metrics import _constants  # noqa: E402
from binding_metrics.metrics.contacts import calculate_contacts  # noqa: E402

COULOMB_K_KJ_NM_MOL_E2 = _constants.COULOMB_KJ_NM_MOL_E2


@pytest.fixture
def pair_path(tmp_path, sample_pdb_content):
    return write_dipeptide_pair(tmp_path / "pair.pdb", sample_pdb_content)


# ---------------------------------------------------------------------------
# Pair energy
# ---------------------------------------------------------------------------


class TestPairEnergy:
    @pytest.fixture(autouse=True)
    def _needs_openmm(self):
        pytest.importorskip("openmm")

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_coulomb_of_an_ion_pair_is_the_same_in_every_image(self, tmp_path, cell):
        from binding_metrics.metrics.energy import calculate_component_energies

        frames = frames_with_images(ION_PAIR_NM, moved_indices=[0], cell=cell)
        dcd, pdb = write_trajectory(tmp_path, ion_topology(), frames, cell)

        energies = calculate_component_energies(dcd, pdb, [0], [1])

        expected = -COULOMB_K_KJ_NM_MOL_E2 / ION_DISTANCE_NM
        assert energies["electrostatic"] == pytest.approx([expected] * len(frames), rel=1e-5)
        assert energies["vdw"] == pytest.approx([energies["vdw"][0]] * len(frames), rel=1e-4)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_interaction_energy_of_an_ion_pair_is_the_same_in_every_image(self, tmp_path, cell):
        from binding_metrics.metrics.energy import (
            calculate_component_energies,
            calculate_interaction_energy,
        )

        frames = frames_with_images(ION_PAIR_NM, moved_indices=[0], cell=cell)
        dcd, pdb = write_trajectory(tmp_path, ion_topology(), frames, cell)

        total = calculate_interaction_energy(dcd, pdb, [0], [1])

        components = calculate_component_energies(dcd, pdb, [0], [1])
        assert total == pytest.approx([total[0]] * len(frames), rel=1e-4)
        assert total == pytest.approx(components["total"], rel=1e-6)
        assert total[0] < 0

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_peptide_pair_energies_match_the_frame_without_translation(
        self, tmp_path, pair_path, cell
    ):
        from binding_metrics.metrics.energy import (
            calculate_component_energies,
            calculate_interaction_energy,
        )

        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        frames = frames_with_images(reference_xyz, moved_indices=ligand, cell=cell)
        dcd, pdb = write_trajectory(tmp_path, topology, frames, cell, pair_path)

        components = calculate_component_energies(dcd, pdb, ligand, receptor)
        total = calculate_interaction_energy(dcd, pdb, ligand, receptor)

        for key in ("electrostatic", "vdw", "total"):
            assert components[key] == pytest.approx([components[key][0]] * len(frames), rel=1e-4)
        assert total == pytest.approx([total[0]] * len(frames), rel=1e-4)

    def test_peptide_pair_in_a_cell_equals_the_same_frame_without_a_cell(self, tmp_path, pair_path):
        from binding_metrics.metrics.energy import calculate_component_energies

        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        with_cell = tmp_path / "cell"
        without_cell = tmp_path / "plain"
        with_cell.mkdir()
        without_cell.mkdir()
        dcd_cell, _ = write_trajectory(
            with_cell, topology, [reference_xyz], "orthorhombic", pair_path
        )
        dcd_plain, _ = write_trajectory(without_cell, topology, [reference_xyz], None, pair_path)

        periodic = calculate_component_energies(dcd_cell, pair_path, ligand, receptor)
        plain = calculate_component_energies(dcd_plain, pair_path, ligand, receptor)

        for key in ("electrostatic", "vdw", "total"):
            assert periodic[key] == pytest.approx(plain[key], rel=1e-5)
        assert plain["electrostatic"][0] != 0.0

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_partners_far_apart_in_one_image_keep_their_far_apart_energy(self, tmp_path, cell):
        from binding_metrics.metrics.energy import calculate_component_energies

        far_nm = 2.0
        frames = [[[1.0, 1.0, 1.0], [1.0 + far_nm, 1.0, 1.0]]]
        dcd, pdb = write_trajectory(tmp_path, ion_topology(), frames, cell)

        energies = calculate_component_energies(dcd, pdb, [0], [1])

        assert energies["electrostatic"][0] == pytest.approx(
            -COULOMB_K_KJ_NM_MOL_E2 / far_nm, rel=1e-5
        )

    def test_a_pair_beyond_half_a_box_length_takes_the_nearest_image(self, tmp_path):
        from binding_metrics.metrics.energy import calculate_component_energies

        box_x_nm = CELLS["orthorhombic"][0][0]
        separation_nm = 0.6 * box_x_nm
        frames = [[[1.0, 1.0, 1.0], [1.0 + separation_nm, 1.0, 1.0]]]
        dcd, pdb = write_trajectory(tmp_path, ion_topology(), frames, "orthorhombic")

        energies = calculate_component_energies(dcd, pdb, [0], [1])

        assert energies["electrostatic"][0] == pytest.approx(
            -COULOMB_K_KJ_NM_MOL_E2 / (box_x_nm - separation_nm), rel=1e-5
        )

    def test_trajectory_without_a_cell_uses_the_stored_separations(self, tmp_path):
        from binding_metrics.metrics.energy import calculate_component_energies

        box_x_nm = CELLS["orthorhombic"][0][0]
        frames = frames_with_images(ION_PAIR_NM, moved_indices=[0], cell="orthorhombic")[:2]
        dcd, pdb = write_trajectory(tmp_path, ion_topology(), frames, cell=None)

        energies = calculate_component_energies(dcd, pdb, [0], [1])

        raw_separation_nm = box_x_nm - ION_DISTANCE_NM
        assert energies["electrostatic"] == pytest.approx(
            [
                -COULOMB_K_KJ_NM_MOL_E2 / ION_DISTANCE_NM,
                -COULOMB_K_KJ_NM_MOL_E2 / raw_separation_nm,
            ],
            rel=1e-5,
        )


# ---------------------------------------------------------------------------
# Contacts: counted under the minimum image already; these tests guard that
# ---------------------------------------------------------------------------


class TestContacts:
    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_contact_count_is_the_same_in_every_image(self, tmp_path, cell):
        frames = frames_with_images(ION_PAIR_NM, moved_indices=[1], cell=cell)
        dcd, pdb = write_trajectory(tmp_path, carbon_pair_topology(), frames, cell)

        contacts = calculate_contacts(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        np.testing.assert_array_equal(contacts, np.ones(len(frames)))


# ---------------------------------------------------------------------------
# The shared policy
# ---------------------------------------------------------------------------


def random_two_selection_trajectory(cell: str, seed: int):
    """One frame of 6 ligand and 9 receptor atoms scattered over a few cells, in a cell."""
    rng = np.random.default_rng(seed)
    n_ligand, n_receptor = 6, 9
    topology = md.Topology()
    for _ in range(n_ligand + n_receptor):
        residue = topology.add_residue("ALA", topology.add_chain())
        topology.add_atom("CA", md.element.carbon, residue)
    xyz = rng.uniform(-5.0, 5.0, size=(1, n_ligand + n_receptor, 3)).astype(np.float32)
    lengths, angles = CELLS[cell]
    traj = md.Trajectory(
        xyz, topology, unitcell_lengths=np.array([lengths]), unitcell_angles=np.array([angles])
    )
    return traj, list(range(n_ligand)), list(range(n_ligand, n_ligand + n_receptor))


def brute_force_minimum_image_nm(traj, ligand, receptor) -> np.ndarray:
    """Smallest distance over the 7 x 7 x 7 neighbouring cells, in float64."""
    vectors = traj.unitcell_vectors[0].astype(float)
    shifts = [
        np.array((i, j, k)) @ vectors
        for i in range(-3, 4)
        for j in range(-3, 4)
        for k in range(-3, 4)
    ]
    xyz = traj.xyz[0].astype(float)
    out = np.empty((len(ligand), len(receptor)))
    for a, lig_atom in enumerate(ligand):
        for b, rec_atom in enumerate(receptor):
            delta = xyz[lig_atom] - xyz[rec_atom]
            out[a, b] = min(np.linalg.norm(delta + shift) for shift in shifts)
    return out


class TestMinimumImageDistances:
    def test_a_cell_is_detected_only_when_the_trajectory_has_one(self):
        from binding_metrics.metrics import _periodic

        with_cell, _, _ = random_two_selection_trajectory("triclinic", 0)
        without_cell = md.Trajectory(with_cell.xyz, with_cell.topology)

        assert _periodic.has_unit_cell(with_cell)
        assert not _periodic.has_unit_cell(without_cell)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_pair_distances_match_a_search_over_neighbouring_cells(self, cell, seed):
        from binding_metrics.metrics import _periodic

        traj, ligand, receptor = random_two_selection_trajectory(cell, seed)

        distances = _periodic.ligand_receptor_distances_nm(traj, 0, ligand, receptor)

        assert distances.shape == (len(ligand), len(receptor))
        np.testing.assert_allclose(
            distances, brute_force_minimum_image_nm(traj, ligand, receptor), atol=2e-5
        )

    def test_pair_distances_without_a_cell_are_the_plain_norm(self):
        from binding_metrics.metrics import _periodic

        traj, ligand, receptor = random_two_selection_trajectory("orthorhombic", 3)
        plain = md.Trajectory(traj.xyz, traj.topology)

        distances = _periodic.ligand_receptor_distances_nm(plain, 0, ligand, receptor)

        expected = np.array(
            [[np.linalg.norm(plain.xyz[0][i] - plain.xyz[0][j]) for j in receptor] for i in ligand]
        )
        np.testing.assert_array_equal(distances, expected)
