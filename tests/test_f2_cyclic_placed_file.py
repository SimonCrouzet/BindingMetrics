"""The structure that ``score_pose_set`` writes for the metrics keeps the covalent links (#186).

A model of a cyclic peptide with non-standard residues has no ``_struct_conn``: nothing in the
file says that its residues are bonded to their neighbours or that the ring is closed. The pose
layer writes the links it finds by geometry, which is what a reader that builds bonds from the
residue names (OpenMM) cannot do for a residue it does not know.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR

from binding_metrics.poses import ExperimentalPose, Reference, score_pose_set
from binding_metrics.poses.score import _write_structure
from binding_metrics.testing.synthetic import write_cif

CWA = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
ENERGY = "structure_interaction_energy"


@pytest.fixture(scope="module")
def ring_model(tmp_path_factory):
    """1CWA without waters, written as a bare model: no bond records of any kind."""
    from binding_metrics.metrics._common import load_structure

    atoms = load_structure(CWA)
    path = tmp_path_factory.mktemp("model") / "ring_model.cif"
    write_cif(atoms[atoms.res_name != "HOH"], path)
    assert "_struct_conn" not in path.read_text(encoding="utf-8")
    return path


def _place(path, tmp_path, **options):
    pose = ExperimentalPose.from_file(path, id="ring")
    reference = Reference(CWA, receptor_chain="A", binder_chain="C")
    return score_pose_set(
        [pose],
        "C",
        "A",
        metrics=options.pop("metrics", ["interface"]),
        reference=reference,
        keep_tmp=True,
        **options,
    )


def _links(topology, chain):
    """The bonds that join two residues of the chain: ((position, atom), (position, atom))."""
    start = next(r.index for c in topology.chains() if c.id == chain for r in c.residues())
    return {
        tuple(sorted(((a.residue.index - start, a.name), (b.residue.index - start, b.name))))
        for a, b in topology.bonds()
        if a.residue is not b.residue and chain in (a.residue.chain.id, b.residue.chain.id)
    }


def test_the_placed_file_has_the_links_of_the_bundled_structure(ring_model, tmp_path):
    from binding_metrics.io.structures import load_structure, topology_chain_id

    result = _place(ring_model, tmp_path)

    placed = Path(result["provenance"]["temp_dir"]) / "pose_0000.cif"
    topology, _ = load_structure(placed)
    bundled, _ = load_structure(CWA)
    chain = topology_chain_id(topology, "C")
    expected = _links(bundled, topology_chain_id(bundled, "C"))
    assert len(expected) == 11  # ten peptide links and the head-to-tail closure
    assert _links(topology, chain) == expected


def test_a_bond_table_is_written_to_struct_conn(tmp_path):
    import biotite.structure as struc

    from binding_metrics.metrics._common import load_structure

    atoms = load_structure(CWA)
    atoms = atoms[atoms.res_name != "HOH"]
    last = int(
        np.flatnonzero((atoms.chain_id == "C") & (atoms.res_id == 11) & (atoms.atom_name == "C"))[0]
    )
    first = int(
        np.flatnonzero((atoms.chain_id == "C") & (atoms.res_id == 1) & (atoms.atom_name == "N"))[0]
    )
    atoms.bonds = struc.BondList(
        atoms.array_length(), np.array([[last, first, int(struc.BondType.SINGLE)]])
    )
    target = tmp_path / "with_bond.cif"

    _write_structure(atoms, target)

    text = target.read_text(encoding="utf-8")
    assert "_struct_conn.ptnr1_label_atom_id" in text
    assert "covale" in text


def test_an_array_without_a_bond_table_is_written_without_struct_conn(tmp_path):
    from binding_metrics.metrics._common import load_structure

    target = tmp_path / "no_bond.cif"

    _write_structure(load_structure(CWA), target)

    assert "_struct_conn" not in target.read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.integration
def test_the_energy_of_the_ring_in_the_frame_of_the_reference(ring_model, tmp_path):
    cpu = {"modes": ("raw",), "device": "cpu"}
    result = _place(
        ring_model, tmp_path, metrics=[ENERGY], heavy="all", metric_kwargs={ENERGY: cpu}
    )

    row = result["poses"][0]
    assert row["status"] == "ok", row["reasons"]
    energy = row[ENERGY]["raw_interaction_energy"]
    assert math.isfinite(energy)
    assert energy < 0  # the ring binds cyclophilin


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("BM_SAVED_PREDICTIONS"),
    reason="BM_SAVED_PREDICTIONS names no directory of saved predictions",
)
def test_the_energy_of_a_saved_prediction_of_the_ring(tmp_path):
    pattern = (
        "boltz2/*/*/outputs/boltz_results_cwa_auto_s42/predictions/cwa_auto_s42/"
        "cwa_auto_s42_model_0.cif"
    )
    found = sorted(Path(os.environ["BM_SAVED_PREDICTIONS"]).glob(pattern))
    if not found:
        pytest.skip(f"no saved prediction matches {pattern}")
    cpu = {"modes": ("raw",), "device": "cpu"}

    result = _place(found[0], tmp_path, metrics=[ENERGY], heavy="all", metric_kwargs={ENERGY: cpu})

    row = result["poses"][0]
    assert row["status"] == "ok", row["reasons"]
    assert math.isfinite(row[ENERGY]["raw_interaction_energy"])
    assert row[ENERGY]["terminal_oxygens_added"] == ["A:GLU165"]


def test_the_file_is_written_when_the_closure_search_fails(ring_model, tmp_path, monkeypatch):
    from binding_metrics.io.structures import load_structure

    def broken(atoms, chain_id):
        raise RuntimeError("closure search failed")

    monkeypatch.setattr("binding_metrics.capabilities.detect_closures", broken)

    result = _place(ring_model, tmp_path)

    assert result["poses"][0]["status"] == "ok", result["poses"][0]["reasons"]
    placed = Path(result["provenance"]["temp_dir"]) / "pose_0000.cif"
    topology, _ = load_structure(placed)
    peptide = [c for c in topology.chains() if len(list(c.residues())) == 11][0]
    assert len(_links(topology, peptide.id)) == 10  # the peptide links; the ring is not closed
