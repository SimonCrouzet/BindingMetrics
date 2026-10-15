"""Pair energies of a trajectory whose partners sit in different periodic images (#169).

With a unit cell the pair energy is taken on the complex made contiguous (molecules whole,
ligand in the image next to the receptor) with raw distances, so translating one partner by a
lattice vector of an orthorhombic or a triclinic cell leaves it unchanged, and so does a cell
smaller than the complex. The ion pair has an analytic Coulomb term, -k / d for charges +1 and
-1. The contact count already used the minimum image; its tests guard that.
"""

import numpy as np
import pytest
from _r2_periodic_support import (
    CELL_NAMES,
    CELLS,
    ION_DISTANCE_NM,
    ION_PAIR_NM,
    LIGAND,
    RECEPTOR,
    carbon_pair_topology,
    frames_with_images,
    ion_topology,
    pair_sum_reference_kj_mol,
    peptide_pair,
    rigid_bodies_topology,
    rigid_bodies_xyz,
    trajectory_with_cell,
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


@pytest.fixture
def numpy2_promotion():
    """Scalar promotion of NumPy 2: a Python float combined with a float32 scalar stays float32.

    NumPy 1.x can switch to that rule; on NumPy 2 it is the rule already.
    """
    if not hasattr(np, "_set_promotion_state"):
        yield
        return
    previous = np._get_promotion_state()
    np._set_promotion_state("weak")
    try:
        yield
    finally:
        np._set_promotion_state(previous)


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

    def test_a_ligand_beyond_half_a_cell_from_the_receptor_takes_the_image_on_the_other_side(
        self, tmp_path
    ):
        from binding_metrics.metrics.energy import calculate_component_energies

        box_x_nm = CELLS["orthorhombic"][0][0]
        separation_nm = 0.6 * box_x_nm
        frames = [[[1.0, 1.0, 1.0], [1.0 + separation_nm, 1.0, 1.0]]]
        dcd, pdb = write_trajectory(tmp_path, ion_topology(), frames, "orthorhombic")

        energies = calculate_component_energies(dcd, pdb, [0], [1])

        assert energies["electrostatic"][0] == pytest.approx(
            -COULOMB_K_KJ_NM_MOL_E2 / (box_x_nm - separation_nm), rel=1e-5
        )

    def test_pairs_farther_apart_than_half_a_cell_keep_the_whole_complex_energy(
        self, tmp_path, pair_path
    ):
        from binding_metrics.metrics.energy import calculate_component_energies

        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        frames = frames_with_images(reference_xyz, ligand, "snug")
        snug_dir, plain_dir = tmp_path / "snug", tmp_path / "plain"
        snug_dir.mkdir()
        plain_dir.mkdir()
        dcd, _ = write_trajectory(snug_dir, topology, frames, "snug", pair_path)
        plain_dcd, _ = write_trajectory(plain_dir, topology, frames[:1], None, pair_path)
        half_cell_nm = CELLS["snug"][0][0] / 2
        delta = reference_xyz[ligand][:, None, :] - reference_xyz[receptor][None, :, :]
        assert np.linalg.norm(delta, axis=-1).max() > half_cell_nm

        periodic = calculate_component_energies(dcd, pair_path, ligand, receptor)
        plain = calculate_component_energies(plain_dcd, pair_path, ligand, receptor)

        for key in ("electrostatic", "vdw", "total"):
            assert periodic[key] == pytest.approx([plain[key][0]] * len(frames), rel=1e-4), key

    def test_sums_of_a_float32_trajectory_are_the_float64_pair_sums(
        self, tmp_path, pair_path, numpy2_promotion
    ):
        from binding_metrics.metrics.energy import (
            calculate_component_energies,
            calculate_interaction_energy,
        )

        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        dcd, _ = write_trajectory(tmp_path, topology, [reference_xyz], None, pair_path)
        stored_xyz = md.load(str(dcd), top=str(pair_path)).xyz[0]
        assert stored_xyz.dtype == np.float32

        components = calculate_component_energies(dcd, pair_path, ligand, receptor)
        total = calculate_interaction_energy(dcd, pair_path, ligand, receptor)

        coulomb, lennard_jones, both = pair_sum_reference_kj_mol(
            pair_path, stored_xyz, ligand, receptor
        )
        assert components["electrostatic"].dtype == np.float64
        assert total.dtype == np.float64
        assert components["electrostatic"][0] == pytest.approx(coulomb, rel=1e-9)
        assert components["vdw"][0] == pytest.approx(lennard_jones, rel=1e-9)
        assert total[0] == pytest.approx(both, rel=1e-9)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_sums_in_a_cell_are_float64(self, tmp_path, pair_path, cell, numpy2_promotion):
        from binding_metrics.metrics.energy import calculate_component_energies

        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        dcd, _ = write_trajectory(tmp_path, topology, [reference_xyz], cell, pair_path)

        components = calculate_component_energies(dcd, pair_path, ligand, receptor)

        assert {value.dtype for value in components.values()} == {np.dtype("float64")}

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


def rigid_bodies_trajectory(cell):
    """Rigid-body frames; in a cell, frames 1 and later have the ligand a lattice vector away."""
    if cell is None:
        return md.Trajectory(rigid_bodies_xyz()[None].astype(np.float32), rigid_bodies_topology())
    frames = frames_with_images(rigid_bodies_xyz(), LIGAND, cell)
    return trajectory_with_cell(rigid_bodies_topology(), frames, cell)


class TestComplexDistances:
    def test_a_cell_is_detected_only_when_the_trajectory_has_one(self):
        from binding_metrics.metrics import _periodic

        assert _periodic.has_unit_cell(rigid_bodies_trajectory("triclinic"))
        assert not _periodic.has_unit_cell(rigid_bodies_trajectory(None))

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_every_frame_gives_the_distances_of_the_adjacent_image(self, cell):
        from binding_metrics.metrics import _periodic

        traj = rigid_bodies_trajectory(cell)

        matrices = list(_periodic.ligand_receptor_distances_nm(traj, LIGAND, RECEPTOR))

        assert len(matrices) == traj.n_frames
        assert matrices[0].shape == (len(LIGAND), len(RECEPTOR))
        reference = rigid_bodies_xyz()
        expected = np.linalg.norm(
            reference[LIGAND][:, None, :] - reference[RECEPTOR][None, :, :], axis=-1
        )
        for matrix in matrices:
            np.testing.assert_allclose(matrix, expected, atol=2e-5)

    def test_distances_without_a_cell_are_the_float64_norm_of_the_stored_coordinates(self):
        from binding_metrics.metrics import _periodic

        traj = rigid_bodies_trajectory(None)

        (matrix,) = _periodic.ligand_receptor_distances_nm(traj, LIGAND, RECEPTOR)

        xyz = traj.xyz[0].astype(np.float64)
        expected = np.array([[np.linalg.norm(xyz[i] - xyz[j]) for j in RECEPTOR] for i in LIGAND])
        assert matrix.dtype == np.float64
        np.testing.assert_allclose(matrix, expected, rtol=1e-13)

    def test_distances_follow_the_order_of_the_index_lists(self):
        from binding_metrics.metrics import _periodic

        traj = rigid_bodies_trajectory("triclinic")
        ligand, receptor = LIGAND[::-1], RECEPTOR[2:9][::-1]

        (first, *_) = _periodic.ligand_receptor_distances_nm(traj, ligand, receptor)
        (reference, *_) = _periodic.ligand_receptor_distances_nm(traj, LIGAND, RECEPTOR)

        np.testing.assert_allclose(first, reference[::-1][:, 2:9][:, ::-1], atol=1e-6)
