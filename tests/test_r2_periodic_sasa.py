"""Buried SASA of a trajectory whose partners sit in different periodic images (#169).

With a unit cell the complex is imaged before the areas are taken: molecules are made whole
and the ligand is placed next to the receptor, for orthorhombic and triclinic cells.
"""

import numpy as np
import pytest
from _r2_periodic_support import (
    CELL_NAMES,
    CELLS,
    ION_PAIR_NM,
    LIGAND,
    N_LIGAND,
    N_RECEPTOR,
    RECEPTOR,
    TWO_SPHERE_BURIED_NM2,
    TWO_SPHERE_TOLERANCE_NM2,
    carbon_pair_topology,
    chain_atom_indices,
    frames_with_images,
    peptide_pair,
    rigid_bodies_topology,
    rigid_bodies_xyz,
    trajectory_with_cell,
    wrap_into_cell,
    write_dipeptide_pair,
    write_dipeptide_trio,
    write_trajectory,
)

md = pytest.importorskip("mdtraj")

from binding_metrics.metrics.sasa import (  # noqa: E402
    calculate_buried_sasa,
    calculate_interface_sasa,
)

# mdtraj gives a lone carbon 1.2076 nm^2 in some frames and 1.2091 nm^2 in others (0.13 %).
SPHERE_TOLERANCE_NM2 = 0.003


@pytest.fixture
def pair_path(tmp_path, sample_pdb_content):
    return write_dipeptide_pair(tmp_path / "pair.pdb", sample_pdb_content)


@pytest.fixture
def trio_path(tmp_path, sample_pdb_content):
    return write_dipeptide_trio(tmp_path / "trio.pdb", sample_pdb_content)


# ---------------------------------------------------------------------------
# Buried SASA
# ---------------------------------------------------------------------------


class TestBuriedSasa:
    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_two_spheres_bury_the_same_area_in_every_image(self, tmp_path, cell):
        frames = frames_with_images(ION_PAIR_NM, moved_indices=[1], cell=cell)
        dcd, pdb = write_trajectory(tmp_path, carbon_pair_topology(), frames, cell)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        assert buried == pytest.approx(
            [TWO_SPHERE_BURIED_NM2] * len(frames), abs=TWO_SPHERE_TOLERANCE_NM2
        )

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_every_term_of_the_interface_sasa_is_the_same_in_every_image(self, tmp_path, cell):
        frames = frames_with_images(ION_PAIR_NM, moved_indices=[1], cell=cell)
        dcd, pdb = write_trajectory(tmp_path, carbon_pair_topology(), frames, cell)

        terms = calculate_interface_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        for key in ("ligand", "receptor", "complex", "buried"):
            expected = [terms[key][0]] * len(frames)
            assert terms[key] == pytest.approx(expected, abs=SPHERE_TOLERANCE_NM2), key
        assert terms["buried"][0] == pytest.approx(
            TWO_SPHERE_BURIED_NM2, abs=TWO_SPHERE_TOLERANCE_NM2
        )

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_peptide_pair_buries_the_same_area_in_every_image(self, tmp_path, pair_path, cell):
        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        frames = frames_with_images(reference_xyz, moved_indices=ligand, cell=cell)
        dcd, pdb = write_trajectory(tmp_path, topology, frames, cell)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=ligand, receptor_indices=receptor)

        assert buried[0] > 0.1
        assert buried == pytest.approx([buried[0]] * len(frames), abs=1e-3)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_molecules_cut_apart_by_the_cell_faces_are_made_whole(self, tmp_path, pair_path, cell):
        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        centred = reference_xyz - reference_xyz.mean(axis=0)
        frames = [centred, wrap_into_cell(centred, cell)]
        assert not np.allclose(frames[0], frames[1])
        dcd, pdb = write_trajectory(tmp_path, topology, frames, cell)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=ligand, receptor_indices=receptor)

        assert buried[0] > 0.1
        assert buried[1] == pytest.approx(buried[0], abs=1e-3)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_receptor_of_two_molecules_in_different_images_is_gathered(
        self, tmp_path, trio_path, cell
    ):
        topology, reference_xyz, ligand, receptor = peptide_pair(trio_path)
        receptor_second_molecule = chain_atom_indices(topology)[2]
        frames = frames_with_images(reference_xyz, receptor_second_molecule, cell)
        frames += frames_with_images(reference_xyz, ligand + receptor_second_molecule, cell)[1:]
        dcd, pdb = write_trajectory(tmp_path, topology, frames, cell)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=ligand, receptor_indices=receptor)

        assert buried[0] > 0.1
        assert buried == pytest.approx([buried[0]] * len(frames), abs=1e-3)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_water_outside_the_selections_stays_out_of_the_complex(self, tmp_path, cell):
        topology = carbon_pair_topology()
        water = topology.add_residue("HOH", topology.add_chain())
        for name, element in (("O", md.element.oxygen), ("H1", md.element.hydrogen)):
            topology.add_atom(name, element, water)
        topology.add_atom("H2", md.element.hydrogen, water)
        reference = [*ION_PAIR_NM, [4.0, 4.0, 4.0], [4.1, 4.0, 4.0], [3.9, 4.0, 4.0]]
        frames = frames_with_images(reference, moved_indices=[1], cell=cell)
        dcd, pdb = write_trajectory(tmp_path, topology, frames, cell)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        assert buried == pytest.approx(
            [TWO_SPHERE_BURIED_NM2] * len(frames), abs=TWO_SPHERE_TOLERANCE_NM2
        )

    def test_area_in_a_cell_equals_the_same_frame_without_a_cell(self, tmp_path, pair_path):
        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        with_cell = tmp_path / "cell"
        without_cell = tmp_path / "plain"
        with_cell.mkdir()
        without_cell.mkdir()
        dcd_cell, pdb_cell = write_trajectory(with_cell, topology, [reference_xyz], "triclinic")
        dcd_plain, pdb_plain = write_trajectory(without_cell, topology, [reference_xyz], None)

        periodic = calculate_buried_sasa(dcd_cell, pdb_cell, ligand, receptor)
        plain = calculate_buried_sasa(dcd_plain, pdb_plain, ligand, receptor)

        assert periodic == pytest.approx(plain, abs=1e-4)

    def test_partners_far_apart_bury_nothing(self, tmp_path):
        far_nm = 1.5
        frames = frames_with_images(
            [[1.0, 1.0, 1.0], [1.0 + far_nm, 1.0, 1.0]], moved_indices=[1], cell="triclinic"
        )
        dcd, pdb = write_trajectory(tmp_path, carbon_pair_topology(), frames, "triclinic")

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        assert buried == pytest.approx([0.0] * len(frames), abs=1e-6)

    def test_trajectory_without_a_cell_uses_the_stored_coordinates(self, tmp_path):
        box_x_nm = CELLS["orthorhombic"][0][0]
        frames = frames_with_images(ION_PAIR_NM, moved_indices=[1], cell="orthorhombic")[:2]
        assert frames[1][1, 0] - frames[0][1, 0] == pytest.approx(box_x_nm)
        dcd, pdb = write_trajectory(tmp_path, carbon_pair_topology(), frames, cell=None)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        assert buried[0] == pytest.approx(TWO_SPHERE_BURIED_NM2, abs=TWO_SPHERE_TOLERANCE_NM2)
        assert buried[1] == pytest.approx(0.0, abs=1e-6)


class TestImageComplex:
    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_imaging_leaves_the_given_trajectory_untouched(self, cell):
        from binding_metrics.metrics import _periodic

        frames = frames_with_images(rigid_bodies_xyz(), moved_indices=LIGAND, cell=cell)
        traj = trajectory_with_cell(rigid_bodies_topology(), frames, cell)
        before = traj.xyz.copy()

        _periodic.image_complex(traj, LIGAND, RECEPTOR)

        np.testing.assert_array_equal(traj.xyz, before)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_imaged_selections_are_the_given_atoms_and_the_ligand_touches_the_receptor(self, cell):
        from binding_metrics.metrics import _periodic

        frames = frames_with_images(rigid_bodies_xyz(), moved_indices=LIGAND, cell=cell)
        traj = trajectory_with_cell(rigid_bodies_topology(), frames, cell)

        imaged, ligand, receptor = _periodic.image_complex(traj, LIGAND, RECEPTOR)

        assert imaged.n_atoms == N_LIGAND + N_RECEPTOR
        assert [imaged.topology.atom(i).residue.resSeq for i in ligand] == [
            traj.topology.atom(i).residue.resSeq for i in LIGAND
        ]
        assert [imaged.topology.atom(i).residue.resSeq for i in receptor] == [
            traj.topology.atom(i).residue.resSeq for i in RECEPTOR
        ]
        nearest_nm = [
            np.linalg.norm(
                imaged.xyz[f][ligand][:, None, :] - imaged.xyz[f][receptor][None, :, :], axis=-1
            ).min()
            for f in range(imaged.n_frames)
        ]
        assert nearest_nm == pytest.approx([nearest_nm[0]] * len(frames), abs=1e-4)
        assert nearest_nm[0] < 1.5

    def test_imaging_a_trajectory_without_a_cell_returns_the_selected_atoms_as_they_are(self):
        from binding_metrics.metrics import _periodic

        xyz = np.asarray([rigid_bodies_xyz()], dtype=np.float32)
        traj = md.Trajectory(xyz, rigid_bodies_topology())

        imaged, ligand, receptor = _periodic.image_complex(traj, LIGAND, RECEPTOR)

        np.testing.assert_array_equal(imaged.xyz[:, ligand], xyz[:, LIGAND])
        np.testing.assert_array_equal(imaged.xyz[:, receptor], xyz[:, RECEPTOR])
