"""The energy of a receptor chain without PDBFixer (#165).

``receptor_quality._receptor_energy`` prepares the chain with PDBFixer when it is installed.
Without it the chain used to go to ``Modeller.addHydrogens`` as it was, which stops on a
chain that has no terminal oxygen (every raw crystal chain). It now adds the OXT itself, and
a chain that still lacks atoms returns NaN with an ``error`` that says PDBFixer is needed.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")
pytest.importorskip("biotite")

import biotite.structure as struc
import biotite.structure.io as sio
from conftest import DATA_DIR

from binding_metrics.metrics import receptor_quality

pytestmark = pytest.mark.slow

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"

#: Runs in a fresh interpreter where ``import pdbfixer`` raises: the energy of one chain of
#: a file, optionally without one atom (``RESIDUE_NUMBER:ATOM_NAME``), as JSON on the last line.
_WITHOUT_PDBFIXER = """
import json, sys, warnings
warnings.simplefilter("ignore")
sys.modules["pdbfixer"] = None
import biotite.structure as struc
import biotite.structure.io as sio
from binding_metrics.metrics.receptor_quality import _receptor_energy

path, chain, dropped_atom = sys.argv[1:4]
atoms = sio.load_structure(path)
atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == chain)]
if dropped_atom != "-":
    number, name = dropped_atom.split(":")
    atoms = atoms[~((atoms.res_id == int(number)) & (atoms.atom_name == name))]
print(json.dumps(_receptor_energy(atoms, device="cpu")))
"""


def _energy_without_pdbfixer(path, chain: str, dropped_atom: str = "-") -> dict:
    run = subprocess.run(
        [sys.executable, "-c", _WITHOUT_PDBFIXER, str(path), chain, dropped_atom],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(run.stdout.strip().splitlines()[-1])


def _chain_atoms(path, chain: str):
    atoms = sio.load_structure(str(path))
    atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
    return atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == chain)]


class TestWithoutPdbfixer:
    def test_a_raw_chain_without_its_terminal_oxygen_gets_an_energy(self):
        """1YCR chain A ends in VAL85 without OXT; PDBFixer completes it to 1449 atoms."""
        without = _energy_without_pdbfixer(YCR, "A")

        assert without["error"] is None, without["error"]
        assert np.isfinite(without["energy_kJ_mol"])
        with_pdbfixer = receptor_quality._receptor_energy(_chain_atoms(YCR, "A"), device="cpu")
        assert without["n_atoms_with_h"] == with_pdbfixer["n_atoms_with_h"] == 1449
        # The chain is complete but for the OXT, so the two routes build the same system. They
        # place the OXT 0.7 A apart (PDBFixer from the residue template, here in the plane of
        # CA, C and O), which moves the energy by 556 of 4372 kJ/mol through the carboxylate charge.
        assert without["energy_kJ_mol"] == pytest.approx(with_pdbfixer["energy_kJ_mol"], rel=0.2)

    def test_a_chain_that_lacks_atoms_says_that_pdbfixer_is_needed(self):
        """The CB of residue 50 is missing: only PDBFixer can rebuild it."""
        result = _energy_without_pdbfixer(YCR, "A", "50:CB")

        assert np.isnan(result["energy_kJ_mol"])
        assert "PDBFixer" in result["error"]
        assert "binding-metrics[structure]" in result["error"]

    def test_a_chain_that_is_already_complete_is_unchanged(self):
        """Chain B of 1YCR already had an energy without PDBFixer and keeps it."""
        result = _energy_without_pdbfixer(YCR, "B")
        assert result["error"] is None, result["error"]
        assert np.isfinite(result["energy_kJ_mol"])


class TestTerminalOxygen:
    @pytest.fixture
    def open_chain(self, tmp_path):
        """Chain A of 1YCR as OpenMM reads it: VAL85 is the last residue and has no OXT."""
        import biotite.structure.io.pdb as pdb_io
        from openmm.app import PDBFile

        handle = pdb_io.PDBFile()
        pdb_io.set_structure(handle, _chain_atoms(YCR, "A"))
        handle.write(str(tmp_path / "chain_a.pdb"))
        pdb = PDBFile(str(tmp_path / "chain_a.pdb"))
        return pdb.topology, pdb.positions

    def test_the_oxygen_is_in_the_carboxylate_plane_at_1_25_angstrom(self, open_chain):
        import openmm.unit as unit

        from binding_metrics.core.system import add_terminal_oxygens, find_open_c_termini

        assert len(find_open_c_termini(open_chain[0])) == 1
        topology, positions = add_terminal_oxygens(*open_chain)

        assert find_open_c_termini(topology) == []
        last = list(topology.residues())[-1]
        by_name = {a.name: a.index for a in last.atoms()}
        xyz = np.array(positions.value_in_unit(unit.angstrom))
        carbon, alpha, oxygen, oxt = (xyz[by_name[n]] for n in ("C", "CA", "O", "OXT"))
        assert np.linalg.norm(oxt - carbon) == pytest.approx(1.25, abs=1e-6)
        assert np.linalg.norm(oxt - oxygen) == pytest.approx(2.17, abs=0.05)  # 120 degrees apart
        normal = np.cross(alpha - carbon, oxygen - carbon)
        assert abs(np.dot(oxt - carbon, normal / np.linalg.norm(normal))) < 1e-6
        bonds = {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()}
        assert frozenset((by_name["C"], by_name["OXT"])) in bonds
        assert topology.getNumAtoms() == open_chain[0].getNumAtoms() + 1
        assert len(positions) == topology.getNumAtoms()

    def test_a_closed_chain_is_returned_as_it_is(self, open_chain):
        from binding_metrics.core.system import add_terminal_oxygens

        closed, closed_positions = add_terminal_oxygens(*open_chain)
        again, again_positions = add_terminal_oxygens(closed, closed_positions)
        assert again is closed
        assert again_positions is closed_positions
