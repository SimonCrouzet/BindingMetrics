"""The heterogens of a crystal pose in the file that the metrics read (#197).

The waters, ions and ligands of a crystal sit in chains of their own in a deposited file (a
label ID per group, the author ID of the chain they belong to). The file written for the metrics
named them by the chain of the polymer, so a reader that names the chains by label ID (OpenMM)
put them into the polymer, and the interaction energy counted them. The written file keeps the
author chain ID of every atom and gives each group of heterogens a label of its own.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR

from binding_metrics.io.structures import load_structure, strip_heterogens, topology_chain_id
from binding_metrics.poses import ExperimentalPose, Reference, score_pose_set
from binding_metrics.poses.score import _write_structure

CWA = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"
ENERGY = "structure_interaction_energy"


def _place(path, binder, target, tmp_path, **options):
    pose = ExperimentalPose.from_file(path, id="crystal")
    reference = Reference(path, receptor_chain=target, binder_chain=binder)
    return score_pose_set(
        [pose],
        binder,
        target,
        metrics=options.pop("metrics", ["interface"]),
        reference=reference,
        keep_tmp=True,
        tmp_dir=tmp_path,
        **options,
    )


def _written(source, target):
    """The file that ``_write_structure`` makes from the atoms of a bundled structure."""
    from binding_metrics.metrics._common import load_structure as load_atoms

    _write_structure(load_atoms(source), target)
    return target


def _chain_names(topology, chain_ids):
    """chain ID in the topology -> the names of its residues, for the author IDs given."""
    wanted = {topology_chain_id(topology, chain_id) for chain_id in chain_ids}
    return {c.id: {r.name for r in c.residues()} for c in topology.chains() if c.id in wanted}


def _label_and_author(path):
    import biotite.structure.io.pdbx as pdbx

    site = pdbx.CIFFile.read(str(path)).block["atom_site"]
    return site["label_asym_id"].as_array(), site["auth_asym_id"].as_array()


def test_the_waters_of_a_crystal_are_outside_the_polymer_chains(tmp_path):
    topology, _ = load_structure(_written(CWA, tmp_path / "written.cif"))

    for names in _chain_names(topology, ["A", "C"]).values():
        assert "HOH" not in names
    assert sum(1 for r in topology.residues() if r.name == "HOH") == 144


def test_the_author_chain_of_every_atom_is_kept(tmp_path):
    from binding_metrics.metrics._common import load_structure as load_atoms

    written = _written(CWA, tmp_path / "written.cif")

    def census(atoms):
        return Counter(zip(atoms.chain_id.tolist(), atoms.res_name.tolist()))

    assert census(load_atoms(written)) == census(load_atoms(CWA))
    labels, authors = _label_and_author(written)
    assert set(authors) == {"A", "C"}
    assert set(labels) > {"A", "C"}


def test_strip_heterogens_drops_the_waters_as_for_the_original_file(tmp_path):
    topology, positions = load_structure(_written(CWA, tmp_path / "written.cif"))
    original, original_positions = load_structure(CWA)

    stripped, _ = strip_heterogens(
        topology, positions, topology_chain_id(topology, "C"), topology_chain_id(topology, "A")
    )
    reference, _ = strip_heterogens(
        original,
        original_positions,
        topology_chain_id(original, "C"),
        topology_chain_id(original, "A"),
    )

    assert not any(r.name == "HOH" for r in stripped.residues())
    assert [r.name for r in stripped.residues()] == [r.name for r in reference.residues()]


def test_a_ligand_has_a_chain_of_its_own(tmp_path):
    topology, _ = load_structure(_written(SFTI, tmp_path / "written.cif"))

    for names in _chain_names(topology, ["A", "I"]).values():
        assert not names & {"HOH", "GSH"}
    assert any("GSH" in {r.name for r in c.residues()} for c in topology.chains())


def test_a_structure_without_heterogens_keeps_one_label_per_chain(tmp_path):
    labels, authors = _label_and_author(_written(P53, tmp_path / "written.cif"))

    assert (labels == authors).all()


def test_a_modified_residue_inside_the_chain_stays_in_the_chain(tmp_path):
    """A residue the amino-acid list does not know, bonded to its neighbours, is polymer."""
    from binding_metrics.metrics._common import load_structure as load_atoms

    atoms = load_atoms(CWA)
    atoms = atoms[atoms.res_name != "HOH"]
    atoms.res_name[(atoms.chain_id == "C") & (atoms.res_id == 5)] = "XYZ"
    target = tmp_path / "modified.cif"

    _write_structure(atoms, target)

    topology, _ = load_structure(target)
    chain_id = topology_chain_id(topology, "C")
    peptide = [r for c in topology.chains() if c.id == chain_id for r in c.residues()]
    assert len(peptide) == 11
    assert "XYZ" in {r.name for r in peptide}
    labels, authors = _label_and_author(target)
    assert (labels == authors).all()


def test_the_written_array_is_not_modified(tmp_path):
    from binding_metrics.metrics._common import load_structure as load_atoms

    atoms = load_atoms(CWA)
    before = atoms.chain_id.copy()

    _write_structure(atoms, tmp_path / "written.cif")

    assert np.array_equal(atoms.chain_id, before)


@pytest.mark.slow
def test_the_file_of_a_pose_set_keeps_the_waters_out_of_the_polymer(tmp_path):
    result = _place(CWA, "C", "A", tmp_path)

    topology, _ = load_structure(Path(result["provenance"]["temp_dir"]) / "pose_0000.cif")

    for names in _chain_names(topology, ["A", "C"]).values():
        assert "HOH" not in names
    assert sum(1 for r in topology.residues() if r.name == "HOH") == 144


@pytest.mark.slow
@pytest.mark.integration
def test_the_energy_of_a_crystal_is_the_same_in_both_frames(tmp_path):
    cpu = {ENERGY: {"modes": ("raw",), "device": "cpu"}}
    energies = {}
    for frame in ("reference", "original"):
        (tmp_path / frame).mkdir()
        result = _place(
            CWA,
            "C",
            "A",
            tmp_path / frame,
            metrics=[ENERGY],
            options={"frame": {"metrics_frame": frame}},
            metric_kwargs=cpu,
        )
        row = result["poses"][0]
        assert row["status"] == "ok", row["reasons"]
        energies[frame] = row[ENERGY]

    assert energies["reference"]["raw_interaction_energy"] == pytest.approx(
        energies["original"]["raw_interaction_energy"], abs=0.01
    )
    assert energies["reference"]["num_contacts"] == energies["original"]["num_contacts"]
