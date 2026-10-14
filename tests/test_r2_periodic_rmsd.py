"""Receptor-frame ligand RMSD of a trajectory whose ligand sits in another periodic image (#169).

The ligand RMSD is taken on coordinates in which the ligand is next to the receptor, for
orthorhombic and triclinic cells; a trajectory without a cell is used as stored.
"""

import numpy as np
import pytest
from _r2_periodic_support import (
    CELL_NAMES,
    CELLS,
    LIGAND,
    RECEPTOR,
    frames_with_images,
    lattice_offsets_nm,
    peptide_pair,
    rigid_bodies_topology,
    rigid_bodies_xyz,
    wrap_into_cell,
    write_dipeptide_pair,
    write_trajectory,
)

pytest.importorskip("mdtraj")

from binding_metrics.metrics.rmsd import calculate_ligand_rmsd  # noqa: E402


@pytest.fixture
def pair_path(tmp_path, sample_pdb_content):
    return write_dipeptide_pair(tmp_path / "pair.pdb", sample_pdb_content)


class TestLigandRmsd:
    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_ligand_in_another_image_has_the_rmsd_of_the_adjacent_image(self, tmp_path, cell):
        reference = rigid_bodies_xyz()
        frames = frames_with_images(reference, moved_indices=LIGAND, cell=cell)
        dcd, pdb = write_trajectory(tmp_path, rigid_bodies_topology(), frames, cell)

        result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)

        np.testing.assert_allclose(result["ligand_rmsd"], 0.0, atol=1e-3)
        np.testing.assert_allclose(result["receptor_rmsd"], 0.0, atol=1e-3)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_displacement_of_the_ligand_is_kept_when_it_is_also_in_another_image(
        self, tmp_path, cell
    ):
        reference = rigid_bodies_xyz()
        moved = reference.copy()
        moved[LIGAND] += np.array([0.0, 0.5, 0.0])
        moved[LIGAND] += lattice_offsets_nm(cell)[3]
        dcd, pdb = write_trajectory(tmp_path, rigid_bodies_topology(), [reference, moved], cell)

        result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)

        assert result["ligand_rmsd"][0] == pytest.approx(0.0, abs=1e-3)
        assert result["ligand_rmsd"][1] == pytest.approx(0.5, abs=5e-3)

    @pytest.mark.parametrize("cell", CELL_NAMES)
    def test_bonded_molecules_cut_by_the_cell_faces_keep_their_rmsd(
        self, tmp_path, pair_path, cell
    ):
        topology, reference_xyz, ligand, receptor = peptide_pair(pair_path)
        centred = reference_xyz - reference_xyz.mean(axis=0)
        displaced = centred.copy()
        displaced[ligand] += np.array([0.0, 0.3, 0.0])
        frames = [centred, wrap_into_cell(centred, cell), wrap_into_cell(displaced, cell)]
        dcd, pdb = write_trajectory(tmp_path, topology, frames, cell)

        result = calculate_ligand_rmsd(dcd, pdb, ligand, receptor)

        np.testing.assert_allclose(result["ligand_rmsd"][:2], 0.0, atol=2e-3)
        assert result["ligand_rmsd"][2] == pytest.approx(0.3, abs=5e-3)
        np.testing.assert_allclose(result["receptor_rmsd"], 0.0, atol=2e-3)

    def test_trajectory_without_a_cell_reports_the_stored_displacement(self, tmp_path):
        reference = rigid_bodies_xyz()
        moved = reference.copy()
        shift_nm = np.array([CELLS["orthorhombic"][0][0], 0.0, 0.0])
        moved[LIGAND] += shift_nm
        dcd, pdb = write_trajectory(tmp_path, rigid_bodies_topology(), [reference, moved], None)

        result = calculate_ligand_rmsd(dcd, pdb, LIGAND, RECEPTOR)

        assert result["ligand_rmsd"][1] == pytest.approx(np.linalg.norm(shift_nm), abs=1e-3)
