"""The periodic-image policy on the bundled complexes (#169).

Without a unit cell the trajectory metrics give the value of an independent computation from the
stored coordinates. With a cell that holds the complex, a ligand written in another image gives
the value of the adjacent image, for the real complexes 1YCR, 3P8F (a bicyclic peptide with
trypsin) and 1CWA (cyclosporin with cyclophilin, non-canonical residues included).
"""

from pathlib import Path

import numpy as np
import pytest
from _r2_periodic_support import (
    LARGE_CELL_NAMES,
    chain_atom_indices,
    frames_with_images,
    lattice_offsets_nm,
    pair_sum_reference_kj_mol,
    wrap_into_cell,
    write_trajectory,
)
from conftest import DATA_DIR

md = pytest.importorskip("mdtraj")

from binding_metrics.metrics.rmsd import calculate_ligand_rmsd  # noqa: E402
from binding_metrics.metrics.sasa import calculate_buried_sasa  # noqa: E402

pytestmark = pytest.mark.integration

# (file, index of the ligand chain, index of the receptor chain) in the order of the chains of
# the PDB written from the file.
COMPLEXES = [
    ("example_linear_p53_1YCR.pdb", 1, 0),
    ("example_bicyclic_sfti1_3P8F.cif", 1, 0),
    ("example_ncaa_cyclosporin_1CWA.cif", 1, 0),
]
# The areas of one structure differ by about 0.15 % between translated copies of it.
SASA_RELATIVE_TOLERANCE = 5e-3


@pytest.fixture(scope="module")
def complex_pdbs(tmp_path_factory):
    """Each bundled complex written as PDB once, keyed by file name."""
    pytest.importorskip("openmm")
    from binding_metrics.io.structures import load_structure, save_structure

    directory = tmp_path_factory.mktemp("r2_complexes")
    paths = {}
    for name, _, _ in COMPLEXES:
        source = DATA_DIR / name
        topology, positions = load_structure(source)
        paths[name] = directory / f"{Path(name).stem}.pdb"
        save_structure(topology, positions, paths[name], source_path=source)
    return paths


@pytest.mark.parametrize("cell", LARGE_CELL_NAMES)
@pytest.mark.parametrize(("name", "ligand_chain", "receptor_chain"), COMPLEXES)
def test_sasa_and_ligand_rmsd_of_a_bundled_complex_do_not_depend_on_the_image(
    tmp_path, complex_pdbs, name, ligand_chain, receptor_chain, cell
):
    topology = md.load(str(complex_pdbs[name])).topology
    chains = chain_atom_indices(topology)
    ligand, receptor = chains[ligand_chain], chains[receptor_chain]
    xyz = md.load(str(complex_pdbs[name])).xyz[0].astype(float)
    xyz -= xyz[ligand + receptor].mean(axis=0)
    frames = frames_with_images(xyz, ligand, cell)
    frames.append(wrap_into_cell(xyz, cell))
    shifted = xyz.copy()
    shifted[ligand] += lattice_offsets_nm(cell)[3]
    frames.append(wrap_into_cell(shifted, cell))
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    dcd, pdb = write_trajectory(tmp_path, topology, frames, cell)
    plain_dcd, plain_pdb = write_trajectory(plain_dir, topology, frames[:1], None)

    buried = calculate_buried_sasa(dcd, pdb, ligand, receptor)
    stored = calculate_buried_sasa(plain_dcd, plain_pdb, ligand, receptor)
    ligand_rmsd = calculate_ligand_rmsd(dcd, pdb, ligand, receptor)["ligand_rmsd"]

    assert stored[0] > 5.0
    assert buried == pytest.approx([stored[0]] * len(frames), rel=SASA_RELATIVE_TOLERANCE)
    np.testing.assert_allclose(ligand_rmsd, 0.0, atol=2e-3)


def test_sasa_with_the_crystal_cell_of_1ycr_is_the_area_without_a_cell(
    tmp_path, prepped_example_pdb
):
    reference = md.load(str(prepped_example_pdb))
    assert reference.unitcell_lengths is not None
    chains = chain_atom_indices(reference.topology)
    ligand, receptor = chains[1], chains[0]
    with_cell = tmp_path / "cell.dcd"
    reference.save_dcd(str(with_cell))
    without_cell = tmp_path / "plain.dcd"
    md.Trajectory(reference.xyz, reference.topology).save_dcd(str(without_cell))

    periodic = calculate_buried_sasa(with_cell, prepped_example_pdb, ligand, receptor)
    plain = calculate_buried_sasa(without_cell, prepped_example_pdb, ligand, receptor)

    assert periodic == pytest.approx(plain, rel=SASA_RELATIVE_TOLERANCE)


def test_energy_without_a_cell_is_the_independent_pair_sum(tmp_path, prepped_example_pdb):
    pytest.importorskip("openmm")
    from binding_metrics.metrics.energy import (
        calculate_component_energies,
        calculate_interaction_energy,
    )

    reference = md.load(str(prepped_example_pdb))
    chains = chain_atom_indices(reference.topology)
    ligand, receptor = chains[1], chains[0]
    dcd = tmp_path / "plain.dcd"
    md.Trajectory(reference.xyz, reference.topology).save_dcd(str(dcd))
    stored_xyz = md.load(str(dcd), top=str(prepped_example_pdb)).xyz[0]

    components = calculate_component_energies(dcd, prepped_example_pdb, ligand, receptor)
    total = calculate_interaction_energy(dcd, prepped_example_pdb, ligand, receptor)

    coulomb, lennard_jones, both = pair_sum_reference_kj_mol(
        prepped_example_pdb, stored_xyz, ligand, receptor
    )
    assert components["electrostatic"][0] == pytest.approx(coulomb, rel=1e-9)
    assert components["vdw"][0] == pytest.approx(lennard_jones, rel=1e-9)
    assert total[0] == pytest.approx(both, rel=1e-9)


@pytest.fixture(scope="module")
def cell_less_1ycr(tmp_path_factory, prepped_example_pdb):
    """Prepared 1YCR as a cell-less one-frame DCD: topology, selections and its pair energies."""
    pytest.importorskip("openmm")
    from binding_metrics.metrics.energy import calculate_component_energies

    reference = md.load(str(prepped_example_pdb))
    chains = chain_atom_indices(reference.topology)
    ligand, receptor = chains[1], chains[0]
    dcd = tmp_path_factory.mktemp("r2_plain") / "plain.dcd"
    md.Trajectory(reference.xyz, reference.topology).save_dcd(str(dcd))
    energies = calculate_component_energies(dcd, prepped_example_pdb, ligand, receptor)
    return reference, ligand, receptor, energies


def test_energy_with_the_crystal_cell_of_1ycr_is_the_energy_without_a_cell(
    tmp_path, prepped_example_pdb, cell_less_1ycr
):
    from binding_metrics.metrics.energy import calculate_component_energies

    reference, ligand, receptor, plain = cell_less_1ycr
    assert reference.unitcell_lengths is not None
    dcd = tmp_path / "cell.dcd"
    reference.save_dcd(str(dcd))

    periodic = calculate_component_energies(dcd, prepped_example_pdb, ligand, receptor)

    for key in ("electrostatic", "vdw", "total"):
        assert periodic[key] == pytest.approx(plain[key], rel=1e-5), key


def test_energy_in_a_box_padded_by_1_nm_is_the_whole_complex_energy(
    tmp_path, prepped_example_pdb, cell_less_1ycr
):
    from binding_metrics.metrics.energy import calculate_component_energies

    reference, ligand, receptor, plain = cell_less_1ycr
    xyz = reference.xyz[0].astype(float)
    edge_nm = (xyz.max(axis=0) - xyz.min(axis=0)).max() + 2.0
    xyz = xyz - xyz.mean(axis=0) + edge_nm / 2
    cube = md.Trajectory(
        xyz[None].astype(np.float32),
        reference.topology,
        unitcell_lengths=np.full((1, 3), edge_nm),
        unitcell_angles=np.full((1, 3), 90.0),
    )
    dcd = tmp_path / "cube.dcd"
    cube.save_dcd(str(dcd))

    periodic = calculate_component_energies(dcd, prepped_example_pdb, ligand, receptor)

    for key in ("electrostatic", "vdw", "total"):
        assert periodic[key] == pytest.approx(plain[key], rel=1e-5), key


@pytest.mark.parametrize("cell", LARGE_CELL_NAMES)
def test_energy_of_1ycr_does_not_depend_on_the_image_of_the_ligand(
    tmp_path, prepped_example_pdb, cell
):
    pytest.importorskip("openmm")
    from binding_metrics.metrics.energy import calculate_component_energies

    reference = md.load(str(prepped_example_pdb))
    chains = chain_atom_indices(reference.topology)
    ligand = chains[1]
    xyz = reference.xyz[0].astype(float)
    xyz -= xyz[chains[0] + ligand].mean(axis=0)
    gap = np.linalg.norm(xyz[chains[0]][:, None, :] - xyz[ligand][None, :, :], axis=-1).min(axis=1)
    receptor = [atom for atom, nearest in zip(chains[0], gap) if nearest < 0.8]
    frames = frames_with_images(xyz, ligand, cell)[:3]
    dcd, _ = write_trajectory(tmp_path, reference.topology, frames, cell, prepped_example_pdb)
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    plain_dcd, _ = write_trajectory(
        plain_dir, reference.topology, frames[:1], None, prepped_example_pdb
    )

    periodic = calculate_component_energies(dcd, prepped_example_pdb, ligand, receptor)
    plain = calculate_component_energies(plain_dcd, prepped_example_pdb, ligand, receptor)

    assert len(receptor) > 100
    for key in ("electrostatic", "vdw", "total"):
        assert periodic[key] == pytest.approx([plain[key][0]] * len(frames), rel=1e-4), key
