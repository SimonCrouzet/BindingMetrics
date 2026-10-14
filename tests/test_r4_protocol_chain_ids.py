"""``PeptideBindingProtocol`` finds the caller's chains in the solvated system it writes (#171).

``run`` writes ``solvated.pdb`` and ``analyze`` selects the ligand and receptor atoms from it by
the chain IDs the caller gave. The preparation is replaced by the bundled 1YCR structure with
its chains renamed and a water and an ion chain appended, named as ``Modeller`` names them
(the chain count plus one); the simulation is replaced by a stand-in and the metric functions
by recorders of the atom indices they receive.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR
from openmm import Vec3, unit
from openmm.app import Element, Modeller, PDBFile, Topology

from binding_metrics.protocols import peptide

EXAMPLE = DATA_DIR / "example_linear_p53_1YCR.pdb"

#: Atom indices of the two chains of the bundled structure, by the IDs of the file.
SOURCE = PDBFile(str(EXAMPLE)).topology
RECEPTOR_ATOMS = [a.index for c in SOURCE.chains() if c.id == "A" for a in c.atoms()]
PEPTIDE_ATOMS = [a.index for c in SOURCE.chains() if c.id == "B" for a in c.atoms()]

CARBON = Element.getBySymbol("C")
NITROGEN = Element.getBySymbol("N")
HYDROGEN = Element.getBySymbol("H")
OXYGEN = Element.getBySymbol("O")
SODIUM = Element.getBySymbol("Na")
ALANINE_BACKBONE = [("N", NITROGEN), ("CA", CARBON), ("C", CARBON)]


class StandInSimulation:
    """Takes the place of ``MDSimulation``: no force field, no dynamics."""

    def __init__(self, modeller, forcefield, config):
        self.modeller = modeller

    def setup(self):
        pass

    def minimize(self):
        pass

    def equilibrate(self):
        pass

    def run(self, output_dir):
        return Path(output_dir) / "trajectory.dcd"


def small_chain(chain_id, residue_name, atoms):
    """A topology of one chain with one residue, and the positions of its atoms (nm)."""
    topology = Topology()
    residue = topology.addResidue(residue_name, topology.addChain(chain_id))
    for name, element in atoms:
        topology.addAtom(name, element, residue)
    positions = [Vec3(0.1 * k, 0.0, 0.0) for k in range(len(atoms))] * unit.nanometer
    return topology, positions


def renamed_input(renames, extra_chain_ids=(), extra_atoms=ALANINE_BACKBONE):
    """The bundled structure with its chains renamed and one alanine chain per extra ID."""
    pdb = PDBFile(str(EXAMPLE))
    for chain in pdb.topology.chains():
        chain.id = renames.get(chain.id, chain.id)
    modeller = Modeller(pdb.topology, pdb.positions)
    for chain_id in extra_chain_ids:
        modeller.add(*small_chain(chain_id, "ALA", extra_atoms))
    return modeller


def solvated(pdb):
    """``pdb`` plus a water chain and an ion chain, as ``Modeller.addSolvent`` names them."""
    modeller = Modeller(pdb.topology, pdb.positions)
    for residue_name, atoms in (
        ("HOH", [("O", OXYGEN), ("H1", HYDROGEN)]),
        ("NA", [("NA", SODIUM)]),
    ):
        chain_id = str(modeller.topology.getNumChains() + 1)
        modeller.add(*small_chain(chain_id, residue_name, atoms))
    return modeller


def written_ids(path):
    return [chain.id for chain in PDBFile(str(path)).topology.chains()]


@pytest.fixture
def recorded(monkeypatch):
    """Atom indices that ``analyze`` hands to the metric functions."""
    seen = {}

    def record(trajectory, topology, ligand_indices, receptor_indices, **kwargs):
        seen.update(ligand=list(ligand_indices), receptor=list(receptor_indices))
        return np.zeros(1)

    for name in ("calculate_buried_sasa", "calculate_contacts", "calculate_interaction_energy"):
        monkeypatch.setattr(peptide, name, record)
    monkeypatch.setattr(peptide, "calculate_rmsd", lambda *a, **k: np.zeros(1))
    return seen


@pytest.fixture
def run_protocol(monkeypatch, tmp_path, recorded):
    """``run`` then ``analyze`` on the structure with its chains renamed; returns the protocol."""

    def go(
        renames, ligand_chain, receptor_chains, extra_chain_ids=(), extra_atoms=ALANINE_BACKBONE
    ):
        pdb = renamed_input(renames, extra_chain_ids, extra_atoms)
        monkeypatch.setattr(peptide, "load_complex", lambda path: pdb)
        monkeypatch.setattr(peptide, "get_forcefield", lambda name: None)
        monkeypatch.setattr(peptide, "prepare_system", lambda pdb, forcefield: solvated(pdb))
        monkeypatch.setattr(peptide, "MDSimulation", StandInSimulation)
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, ligand_chain, receptor_chains)
        protocol.run(tmp_path, minimize=False, equilibrate=False)
        protocol.analyze()
        return protocol

    return go


class TestChainsThatRoundTrip:
    def test_a_and_b_select_the_atoms_the_regenerated_ids_selected(
        self, run_protocol, recorded, tmp_path
    ):
        protocol = run_protocol({}, "B", ["A"])

        legacy = tmp_path / "legacy.pdb"
        modeller = solvated(PDBFile(str(EXAMPLE)))
        with open(legacy, "w", encoding="utf-8") as handle:
            PDBFile.writeFile(modeller.topology, modeller.positions, handle)
        assert recorded["ligand"] == peptide.get_chain_atom_indices(legacy, ["B"])
        assert recorded["receptor"] == peptide.get_chain_atom_indices(legacy, ["A"])
        assert recorded["ligand"] == PEPTIDE_ATOMS
        assert recorded["receptor"] == RECEPTOR_ATOMS
        assert written_ids(protocol._topology_path)[:2] == ["A", "B"]

    def test_the_residue_numbers_of_the_input_are_written(self, run_protocol, tmp_path):
        run_protocol({}, "B", ["A"])

        reread = PDBFile(str(tmp_path / "solvated.pdb")).topology
        written = [c for c in reread.chains() if c.id == "B"][0]
        source = [c for c in SOURCE.chains() if c.id == "B"][0]
        assert [r.id for r in written.residues()] == [r.id for r in source.residues()]
        assert next(iter(source.residues())).id == "17"  # the file does not number from 1

    def test_several_receptor_chains_are_selected_together(self, run_protocol, recorded):
        run_protocol({}, "B", ["A", "C"], extra_chain_ids=["C"])

        assert recorded["ligand"] == PEPTIDE_ATOMS
        first_extra_atom = SOURCE.getNumAtoms()
        assert recorded["receptor"] == [
            *RECEPTOR_ATOMS,
            *range(first_extra_atom, first_extra_atom + 3),
        ]

    def test_a_chain_of_alpha_carbons_is_not_taken_for_solvent(self, run_protocol, recorded):
        run_protocol({}, "B", ["A", "C"], extra_chain_ids=["C"], extra_atoms=[("CA", CARBON)])

        assert recorded["receptor"] == [*RECEPTOR_ATOMS, SOURCE.getNumAtoms()]


class TestChainsThatDoNotRoundTrip:
    @pytest.mark.parametrize(
        ("renames", "ligand", "receptors", "ligand_atoms", "receptor_atoms"),
        [
            pytest.param({"A": "R", "B": "L"}, "L", ["R"], PEPTIDE_ATOMS, RECEPTOR_ATOMS, id="R-L"),
            pytest.param(
                {"A": "R", "B": "L"}, "R", ["L"], RECEPTOR_ATOMS, PEPTIDE_ATOMS, id="swapped"
            ),
            pytest.param(
                {"A": "B", "B": "A"}, "A", ["B"], PEPTIDE_ATOMS, RECEPTOR_ATOMS, id="IDs-exchanged"
            ),
            pytest.param(
                {"A": "MDM2", "B": "P53"},
                "P53",
                ["MDM2"],
                PEPTIDE_ATOMS,
                RECEPTOR_ATOMS,
                id="several-characters",
            ),
            pytest.param(
                {"A": "3", "B": "4"}, "4", ["3"], PEPTIDE_ATOMS, RECEPTOR_ATOMS, id="solvent-IDs"
            ),
            pytest.param({"A": " "}, "B", [" "], PEPTIDE_ATOMS, RECEPTOR_ATOMS, id="blank-ID"),
            pytest.param(
                {"A": "a", "B": "Z"}, "Z", ["a"], PEPTIDE_ATOMS, RECEPTOR_ATOMS, id="case"
            ),
        ],
    )
    def test_the_partners_are_the_chains_the_caller_named(
        self, run_protocol, recorded, renames, ligand, receptors, ligand_atoms, receptor_atoms
    ):
        run_protocol(renames, ligand, receptors)

        assert recorded["ligand"] == ligand_atoms
        assert recorded["receptor"] == receptor_atoms

    def test_a_valid_id_is_written_as_given(self, run_protocol):
        protocol = run_protocol({"A": "R", "B": "L"}, "L", ["R"])

        assert written_ids(protocol._topology_path) == ["R", "L", "3", "4"]

    def test_ids_a_pdb_file_cannot_hold_are_written_as_distinct_characters(self, run_protocol):
        protocol = run_protocol({"A": "MDM2", "B": "P53"}, "P53", ["MDM2"])

        written = written_ids(protocol._topology_path)
        assert len(written) == 4
        assert len(set(written)) == 4
        assert all(len(chain_id) == 1 for chain_id in written)

    def test_a_solvent_chain_with_the_id_of_the_solute_gets_another_one(self, run_protocol):
        protocol = run_protocol({"A": "3", "B": "4"}, "4", ["3"])

        written = written_ids(protocol._topology_path)
        assert written[:2] == ["3", "4"]
        assert len(set(written)) == 4

    def test_more_chain_ids_than_a_pdb_file_holds_fail_before_the_simulation(
        self, monkeypatch, tmp_path
    ):
        pdb = renamed_input({}, extra_chain_ids=[f"chain{k}" for k in range(70)])
        monkeypatch.setattr(peptide, "load_complex", lambda path: pdb)
        monkeypatch.setattr(peptide, "get_forcefield", lambda name: None)
        monkeypatch.setattr(peptide, "prepare_system", lambda pdb, forcefield: solvated(pdb))
        monkeypatch.setattr(peptide, "MDSimulation", StandInSimulation)
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, "B", ["A"])

        with pytest.raises(ValueError, match=r"74 distinct chain IDs.*62"):
            protocol.run(tmp_path, minimize=False, equilibrate=False)
        assert not (tmp_path / "solvated.pdb").exists()


@pytest.mark.slow
@pytest.mark.integration
class TestWithThePreparedSystem:
    """The real ``prepare_system``: protonation, solvation and ions, on the 1YCR example."""

    @pytest.fixture
    def prepared(self, monkeypatch, tmp_path, recorded):
        pytest.importorskip("pdbfixer")

        def go(renames, ligand_chain, receptor_chains):
            pdb = PDBFile(str(EXAMPLE))
            for chain in pdb.topology.chains():
                chain.id = renames.get(chain.id, chain.id)
            monkeypatch.setattr(peptide, "load_complex", lambda path: pdb)
            monkeypatch.setattr(peptide, "MDSimulation", StandInSimulation)
            protocol = peptide.PeptideBindingProtocol(EXAMPLE, ligand_chain, receptor_chains)
            protocol.run(tmp_path, minimize=False, equilibrate=False)
            protocol.analyze()
            return protocol

        return go

    def test_a_and_b_select_the_atoms_the_regenerated_ids_selected(
        self, prepared, recorded, tmp_path
    ):
        protocol = prepared({}, "B", ["A"])

        modeller = protocol._simulation.modeller
        legacy = tmp_path / "legacy.pdb"
        with open(legacy, "w", encoding="utf-8") as handle:
            PDBFile.writeFile(modeller.topology, modeller.positions, handle)
        assert recorded["ligand"] == peptide.get_chain_atom_indices(legacy, ["B"])
        assert recorded["receptor"] == peptide.get_chain_atom_indices(legacy, ["A"])
        assert len(written_ids(protocol._topology_path)) == 4  # peptide, receptor, water, ions

    def test_r_and_l_select_the_atoms_that_a_and_b_select(self, prepared, recorded):
        prepared({}, "B", ["A"])
        as_a_and_b = dict(recorded)
        prepared({"A": "R", "B": "L"}, "L", ["R"])

        assert recorded["ligand"] == as_a_and_b["ligand"]
        assert recorded["receptor"] == as_a_and_b["receptor"]
