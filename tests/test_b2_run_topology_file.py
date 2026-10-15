"""``PeptideBindingProtocol.run`` closes the solvated topology it writes (#52).

The simulation is replaced by a stand-in; the file that ``run`` writes comes from the bundled
1YCR structure, so the content checked is that of a real PDB file.
"""

import gc
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import DATA_DIR
from openmm.app import PDBFile

from binding_metrics.protocols import peptide

EXAMPLE = DATA_DIR / "example_linear_p53_1YCR.pdb"


class StandInSimulation:
    """Takes the place of ``MDSimulation``: no force field, no dynamics."""

    def __init__(self, modeller, forcefield, config):
        pass

    def setup(self):
        pass

    def minimize(self):
        pass

    def equilibrate(self):
        pass

    def run(self, output_dir):
        return Path(output_dir) / "trajectory.dcd"


@pytest.fixture
def protocol(monkeypatch):
    pdb = PDBFile(str(EXAMPLE))
    monkeypatch.setattr(peptide, "load_complex", lambda path: pdb)
    monkeypatch.setattr(peptide, "get_forcefield", lambda name: None)
    monkeypatch.setattr(
        peptide,
        "prepare_system",
        lambda pdb, forcefield: SimpleNamespace(topology=pdb.topology, positions=pdb.positions),
    )
    monkeypatch.setattr(peptide, "MDSimulation", StandInSimulation)
    return peptide.PeptideBindingProtocol(pdb_path=EXAMPLE, ligand_chain="B", receptor_chains=["A"])


def test_run_leaves_no_file_object_open(protocol, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        protocol.run(tmp_path)
        gc.collect()

    unclosed = [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)]
    assert unclosed == []


def test_the_writer_receives_a_file_that_run_closes(protocol, tmp_path, monkeypatch):
    handed = []
    write_file = PDBFile.writeFile

    def remember(topology, positions, file, *args, **kwargs):
        handed.append(file)
        return write_file(topology, positions, file, *args, **kwargs)

    monkeypatch.setattr(peptide.PDBFile, "writeFile", staticmethod(remember))

    protocol.run(tmp_path)

    assert len(handed) == 1
    assert handed[0].closed


def test_the_solvated_topology_is_complete_when_run_returns(protocol, tmp_path):
    protocol.run(tmp_path)

    written = tmp_path / "solvated.pdb"
    reread = PDBFile(str(written))
    assert reread.topology.getNumAtoms() == PDBFile(str(EXAMPLE)).topology.getNumAtoms()
    assert written.read_text(encoding="utf-8").rstrip().endswith("END")
