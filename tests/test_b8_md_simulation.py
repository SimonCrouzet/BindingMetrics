"""MDSimulation.setup leaves a chain break unbonded (#60).

The explicit-solvent protocol built its system with ``forcefield.createSystem(topology)``,
which bonds each residue to the next one of its chain by name, whatever the distance: a gap
in the prepared model was closed by the minimisation as in the implicit-solvent relaxation.
``setup`` now builds the system with ``create_system_respecting_chain_breaks`` and gives the
``Simulation`` the topology that the system was built from.

The fixture is the p53 peptide of 1YCR (chain B) without residues 22-24, prepared, solvated
and ionised in a small box, and the same peptide without a gap.
"""

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

import openmm
import openmm.unit as unit
from conftest import DATA_DIR
from openmm import app

from binding_metrics.core.forcefields import get_forcefield
from binding_metrics.core.simulation import MDSimulation, SimulationConfig

pytestmark = pytest.mark.slow

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"


def _solvated_peptide(deleted: tuple):
    """Chain B of 1YCR without the residues numbered ``deleted``, prepared and solvated."""
    from binding_metrics.core.system import prep_structure, solvate
    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(YCR)
    modeller = app.Modeller(topology, positions)
    modeller.delete(
        [r for r in modeller.topology.residues() if r.chain.id != "B" or int(r.id) in deleted]
    )
    prepped, prepped_positions = prep_structure(modeller.topology, modeller.positions)
    return solvate(prepped, prepped_positions, forcefield=get_forcefield("amber"), padding=0.7)


@pytest.fixture(scope="module")
def gapped_box():
    return _solvated_peptide((22, 23, 24))


@pytest.fixture(scope="module")
def whole_box():
    return _solvated_peptide(())


def _config() -> SimulationConfig:
    return SimulationConfig(platform="CPU", pressure=None)


def _harmonic_bond_pairs(system) -> set:
    force = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
    return {frozenset(force.getBondParameters(i)[:2]) for i in range(force.getNumBonds())}


def _break_atoms(modeller) -> tuple:
    from binding_metrics.core.system import _chain_breaks

    ((_, before, after, _),) = list(_chain_breaks(modeller.topology, modeller.positions))
    return (
        next(a.index for a in before.atoms() if a.name == "C"),
        next(a.index for a in after.atoms() if a.name == "N"),
    )


class TestSetup:
    def test_the_two_sides_of_the_break_are_not_bonded(self, gapped_box):
        pair = _break_atoms(gapped_box)
        simulation = MDSimulation(gapped_box, get_forcefield("amber"), _config())

        simulation.setup()

        assert frozenset(pair) not in _harmonic_bond_pairs(simulation.system)
        simulation_bonds = {
            frozenset((b.atom1.index, b.atom2.index))
            for b in simulation.simulation.topology.bonds()
        }
        assert frozenset(pair) not in simulation_bonds

    def test_a_minimisation_leaves_the_gap_open(self, gapped_box):
        pair = _break_atoms(gapped_box)
        simulation = MDSimulation(gapped_box, get_forcefield("amber"), _config())
        simulation.setup()
        start = np.asarray(simulation.get_positions())

        simulation.minimize(max_iterations=200)
        relaxed = np.asarray(simulation.get_positions())

        assert np.linalg.norm(start[pair[0]] - start[pair[1]]) * 10 > 4.0  # nm -> A
        assert np.linalg.norm(relaxed[pair[0]] - relaxed[pair[1]]) * 10 > 3.0

    def test_the_modeller_of_the_caller_is_left_alone(self, gapped_box):
        pair = _break_atoms(gapped_box)
        MDSimulation(gapped_box, get_forcefield("amber"), _config()).setup()

        bonds = {frozenset((b.atom1.index, b.atom2.index)) for b in gapped_box.topology.bonds()}
        assert frozenset(pair) in bonds


class TestWithoutABreak:
    def test_the_system_is_the_one_createsystem_gives(self, whole_box):
        forcefield = get_forcefield("amber")
        config = _config()
        expected = forcefield.createSystem(
            whole_box.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=config.nonbonded_cutoff * unit.nanometer,
            constraints=app.HBonds,
        )
        simulation = MDSimulation(whole_box, forcefield, config)

        simulation.setup()

        assert openmm.XmlSerializer.serialize(simulation.system) == openmm.XmlSerializer.serialize(
            expected
        )
        assert simulation.simulation.topology is whole_box.topology
