"""The system builders do not bond across a chain break (#60).

OpenMM bonds every residue to the next one of its chain by atom name, whatever the
distance, so a gap in the model (an unresolved loop) used to be closed by the
minimisation: the two ends were pulled together, and the residues next to the gap could
come out with an inverted C-alpha. The relaxation, the implicit-solvent energy and the
energy of the receptor now leave the pair of atoms across a break unbonded and match the
residues on both sides to their templates with ``ignoreExternalBonds=True``, for an input
that has a break and for no other.

The fixtures are the bundled 1YCR (MDM2 chain A, p53 peptide chain B) with a stretch
deleted: residues 60-62 of the receptor (C-N 3.95 A across the gap) or 22-24 of the
peptide (4.96 A), then prepared the way the pipeline prepares an input.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

import openmm
import openmm.unit as unit
from conftest import DATA_DIR
from openmm import app

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"

pytestmark = pytest.mark.slow


def _prepped_with_gap(directory: Path, chain: str, first: int, last: int) -> Path:
    """1YCR without residues ``first``..``last`` of ``chain``, prepared and saved as PDB."""
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_structure

    topology, positions = load_structure(YCR)
    modeller = app.Modeller(topology, positions)
    modeller.delete(
        [
            residue
            for residue in modeller.topology.residues()
            if residue.chain.id == chain and first <= int(residue.id) <= last
        ]
    )
    prepped, prepped_positions = prep_structure(modeller.topology, modeller.positions)
    path = directory / f"1YCR_{chain}_{first}_{last}.pdb"
    save_structure(prepped, prepped_positions, path)
    return path


@pytest.fixture(scope="module")
def peptide_gap(tmp_path_factory) -> Path:
    return _prepped_with_gap(tmp_path_factory.mktemp("peptide_gap"), "B", 22, 24)


@pytest.fixture(scope="module")
def receptor_gap(tmp_path_factory) -> Path:
    return _prepped_with_gap(tmp_path_factory.mktemp("receptor_gap"), "A", 60, 62)


def _harmonic_bond_pairs(system) -> set:
    force = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
    return {frozenset(force.getBondParameters(i)[:2]) for i in range(force.getNumBonds())}


def _topology_bond_pairs(topology) -> set:
    return {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()}


def _break_atoms(topology, positions) -> list:
    """``(index of C, index of N)`` across each chain break of a structure."""
    from binding_metrics.core.system import _chain_breaks

    return [
        (
            next(a.index for a in before.atoms() if a.name == "C"),
            next(a.index for a in after.atoms() if a.name == "N"),
        )
        for _, before, after, _ in _chain_breaks(topology, positions)
    ]


def _c_n_distance_angstrom(positions, pair) -> float:
    xyz = np.asarray(positions.value_in_unit(unit.angstrom))
    return float(np.linalg.norm(xyz[pair[0]] - xyz[pair[1]]))


def _minimise_on_cpu(system, topology, positions, iterations: int = 300):
    simulation = app.Simulation(
        topology,
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=iterations)
    return simulation.context.getState(getPositions=True).getPositions()


class TestRelaxationSystem:
    def test_the_two_sides_of_the_break_are_not_bonded(self, peptide_gap):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        relaxer = ImplicitRelaxation(RelaxationConfig(peptide_chain_id="B", receptor_chain_id="A"))
        system, topology, positions, _ = relaxer._setup_system(peptide_gap)

        (pair,) = _break_atoms(topology, positions)
        assert frozenset(pair) not in _topology_bond_pairs(topology)
        assert frozenset(pair) not in _harmonic_bond_pairs(system)

    def test_a_minimisation_leaves_the_gap_open(self, peptide_gap):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        relaxer = ImplicitRelaxation(RelaxationConfig(peptide_chain_id="B", receptor_chain_id="A"))
        system, topology, positions, _ = relaxer._setup_system(peptide_gap)
        (pair,) = _break_atoms(topology, positions)

        minimised = _minimise_on_cpu(system, topology, positions)

        assert _c_n_distance_angstrom(positions, pair) > 4.0
        # Bonded, the same minimisation closes the gap to the 1.34 A of an amide bond.
        assert _c_n_distance_angstrom(minimised, pair) > 3.0

    def test_the_break_is_still_warned_about(self, peptide_gap, caplog):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        relaxer = ImplicitRelaxation(RelaxationConfig(peptide_chain_id="B", receptor_chain_id="A"))
        with caplog.at_level(logging.WARNING, logger="binding_metrics.protocols.relaxation"):
            relaxer._setup_system(peptide_gap)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Chain break in chain B" in m and "not bonded" in m for m in messages)
        assert not any("pulls them together" in m for m in messages)


class TestEnergySystems:
    def test_the_implicit_system_has_no_bond_across_the_break(self, receptor_gap):
        from binding_metrics.io.structures import load_structure
        from binding_metrics.metrics import energy

        topology, positions = load_structure(receptor_gap)
        system, topology_h, positions_h, _, _ = energy._create_implicit_system(
            topology, positions, "obc2", peptide_chain="B"
        )

        (pair,) = _break_atoms(topology_h, positions_h)
        assert frozenset(pair) not in _topology_bond_pairs(topology_h)
        assert frozenset(pair) not in _harmonic_bond_pairs(system)

    def test_the_isolated_receptor_is_built_without_the_bond_as_well(
        self, receptor_gap, monkeypatch
    ):
        from binding_metrics.io.structures import load_structure
        from binding_metrics.metrics import energy

        topology, positions = load_structure(receptor_gap)
        system, topology_h, positions_h, bond_info, xmls = energy._create_implicit_system(
            topology, positions, "obc2", peptide_chain="B"
        )
        (pair,) = _break_atoms(topology_h, positions_h)
        built = []
        original = energy._build_subsystem

        def recording(subsystem_topology, *args, **kwargs):
            subsystem = original(subsystem_topology, *args, **kwargs)
            built.append(subsystem)
            return subsystem

        monkeypatch.setattr(energy, "_build_subsystem", recording)
        simulation = app.Simulation(
            topology_h,
            system,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("CPU"),
        )
        simulation.context.setPositions(positions_h)
        failures: list = []
        e_complex, e_peptide, e_receptor = energy._evaluate_subsystem_energies(
            simulation,
            topology_h,
            positions_h,
            "B",
            "A",
            "obc2",
            "cpu",
            bond_info=bond_info,
            ncaa_xmls=xmls,
            failures=failures,
        )

        assert failures == []
        assert np.isfinite([e_complex, e_peptide, e_receptor]).all()
        # Chain A comes first in 1YCR, so its atoms keep their indices in the extracted chain.
        _, receptor_system = built
        assert frozenset(pair) not in _harmonic_bond_pairs(receptor_system)


def _receptor_energy_of_chain_without(path, chain: str, dropped: list, monkeypatch):
    """Run ``_receptor_energy`` on ``chain`` of ``path`` minus the residues ``dropped``.

    Returns ``(result, topology, system)``, the last two being the system whose energy is
    reported.
    """
    import biotite.structure as struc
    import biotite.structure.io as sio

    from binding_metrics.metrics import receptor_quality

    atoms = sio.load_structure(str(path))
    atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
    receptor = atoms[
        struc.filter_amino_acids(atoms)
        & (atoms.chain_id == chain)
        & ~np.isin(atoms.res_id, dropped)
    ]

    created = []
    original = app.ForceField.createSystem

    def recording(self, topology, *args, **kwargs):
        # OpenMM checks the keyword arguments against self.createSystem.
        monkeypatch.setattr(app.ForceField, "createSystem", original)
        try:
            system = original(self, topology, *args, **kwargs)
        finally:
            monkeypatch.setattr(app.ForceField, "createSystem", recording)
        created.append((topology, system))
        return system

    monkeypatch.setattr(app.ForceField, "createSystem", recording)
    result = receptor_quality._receptor_energy(receptor, device="cpu")
    topology, system = created[-1]
    return result, topology, system


def _bonded_across(topology, system, id_before: str, id_after: str) -> bool:
    residues = {r.id: r for r in topology.residues()}
    c_before = next(a.index for a in residues[id_before].atoms() if a.name == "C")
    n_after = next(a.index for a in residues[id_after].atoms() if a.name == "N")
    return frozenset((c_before, n_after)) in _harmonic_bond_pairs(system)


class TestReceptorQualityEnergy:
    def test_the_energy_system_of_a_receptor_with_a_gap_has_no_bond_across_it(self, monkeypatch):
        result, topology, system = _receptor_energy_of_chain_without(
            YCR, "A", [60, 61, 62], monkeypatch
        )

        assert result["error"] is None
        assert np.isfinite(result["energy_kJ_mol"])
        assert not _bonded_across(topology, system, "59", "63")

    def test_a_receptor_with_disulfides_and_a_gap_is_built(self, monkeypatch):
        """``ignoreExternalBonds`` alone lets a disulfide cysteine fit both CYM and CYX."""
        import biotite.structure as struc
        import biotite.structure.io as sio

        sfti1_complex = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
        atoms = sio.load_structure(str(sfti1_complex))
        atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
        matriptase = atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == "A")]
        ids = sorted(set(matriptase.res_id.tolist()))
        dropped = ids[100:103]

        result, topology, system = _receptor_energy_of_chain_without(
            sfti1_complex, "A", dropped, monkeypatch
        )

        assert result["error"] is None, result["error"]
        assert np.isfinite(result["energy_kJ_mol"])
        assert not _bonded_across(topology, system, str(ids[99]), str(ids[103]))


@pytest.fixture(scope="module")
def force_field():
    return app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml", "implicit/obc2.xml")


class TestHelper:
    def test_an_input_without_a_break_gives_the_system_and_topology_of_createsystem(
        self, prepped_example_cif, force_field
    ):
        from binding_metrics.core.system import create_system_respecting_chain_breaks
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(prepped_example_cif)
        expected = force_field.createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        system, returned = create_system_respecting_chain_breaks(
            force_field, topology, positions, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )

        assert returned is topology
        assert openmm.XmlSerializer.serialize(system) == openmm.XmlSerializer.serialize(expected)

    def test_openmm_still_checks_the_bonds_of_an_input_without_a_break(
        self, prepped_example_cif, force_field
    ):
        """A missing bond between residues that are 1.3 A apart is not a chain break."""
        from binding_metrics.core.system import create_system_respecting_chain_breaks
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(prepped_example_cif)
        peptide_bond = next(
            b
            for b in topology.bonds()
            if {b.atom1.name, b.atom2.name} == {"C", "N"} and b.atom1.residue != b.atom2.residue
        )
        unbonded = app.Topology()
        atoms = {}
        for chain in topology.chains():
            new_chain = unbonded.addChain(chain.id)
            for residue in chain.residues():
                new_residue = unbonded.addResidue(residue.name, new_chain, residue.id)
                for atom in residue.atoms():
                    atoms[atom.index] = unbonded.addAtom(atom.name, atom.element, new_residue)
        for bond in topology.bonds():
            if bond is not peptide_bond:
                unbonded.addBond(atoms[bond.atom1.index], atoms[bond.atom2.index])

        with pytest.raises(ValueError, match="No template found"):
            create_system_respecting_chain_breaks(
                force_field, unbonded, positions, nonbondedMethod=app.NoCutoff
            )

    def test_a_break_gives_a_topology_without_the_bond_and_keeps_everything_else(
        self, peptide_gap, force_field
    ):
        from binding_metrics.core.system import create_system_respecting_chain_breaks
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(peptide_gap)
        (pair,) = _break_atoms(topology, positions)
        assert frozenset(pair) in _topology_bond_pairs(topology)  # PDBFile bonds it by name

        system, returned = create_system_respecting_chain_breaks(
            force_field, topology, positions, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )

        assert _topology_bond_pairs(topology) - _topology_bond_pairs(returned) == {frozenset(pair)}
        assert _topology_bond_pairs(returned) < _topology_bond_pairs(topology)
        assert [
            (a.name, a.residue.name, a.residue.id, a.residue.chain.id) for a in returned.atoms()
        ] == [(a.name, a.residue.name, a.residue.id, a.residue.chain.id) for a in topology.atoms()]
        assert frozenset(pair) not in _harmonic_bond_pairs(system)
        assert system.getNumParticles() == topology.getNumAtoms()

    def test_the_break_is_logged_once_unless_the_caller_asks_for_silence(
        self, peptide_gap, force_field, caplog
    ):
        from binding_metrics.core.system import create_system_respecting_chain_breaks
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(peptide_gap)
        with caplog.at_level(logging.INFO, logger="binding_metrics.core.system"):
            create_system_respecting_chain_breaks(
                force_field, topology, positions, nonbondedMethod=app.NoCutoff
            )
        warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings_logged) == 1
        assert "not bonded" in warnings_logged[0].getMessage()

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="binding_metrics.core.system"):
            create_system_respecting_chain_breaks(
                force_field, topology, positions, warn=False, nonbondedMethod=app.NoCutoff
            )
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
