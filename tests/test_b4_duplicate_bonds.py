"""A bond that PDBFixer lists twice is listed once in the prepared topology.

``PDBFixer.addMissingAtoms`` calls ``createStandardBonds`` and then adds again every
bond of the old topology that touches a residue the standard table does not know, so
the peptide bond next to a phosphoserine or an N-methylated residue appears twice.
The duplicate gives OpenMM one harmonic bond term too many and the GAFF2 cap logic a
carbon with two caps.
"""

import collections
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

import openmm
from openmm import unit
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Topology, element

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHOSPHO = DATA_DIR / "example_phospho_1QJB.pdb"
LACTAM = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"


def _duplicated_pairs(topology) -> dict:
    """Atom-index pairs that ``topology.bonds()`` lists more than once, with their counts."""
    counts = collections.Counter(
        frozenset((bond.atom1.index, bond.atom2.index)) for bond in topology.bonds()
    )
    return {pair: n for pair, n in counts.items() if n > 1}


def _bond_names(topology) -> collections.Counter:
    """Every bond as ((chain, residue, atom), (chain, residue, atom)), unordered."""

    def end(atom):
        return (atom.residue.chain.index, atom.residue.index, atom.name)

    return collections.Counter(
        frozenset((end(bond.atom1), end(bond.atom2))) for bond in topology.bonds()
    )


def _small_topology():
    """Two chains, an insertion code, atom IDs, a bond type and a box."""
    topology = Topology()
    topology.setPeriodicBoxVectors(
        [openmm.Vec3(3.0, 0, 0), openmm.Vec3(0, 3.0, 0), openmm.Vec3(0, 0, 3.0)] * unit.nanometer
    )
    chain_a = topology.addChain("A")
    ala = topology.addResidue("ALA", chain_a, "7")
    thr = topology.addResidue("THR", chain_a, "7", "A")
    n1 = topology.addAtom("N", element.nitrogen, ala, "11")
    ca1 = topology.addAtom("CA", element.carbon, ala, "12")
    c1 = topology.addAtom("C", element.carbon, ala, "13")
    n2 = topology.addAtom("N", element.nitrogen, thr, "21")
    ca2 = topology.addAtom("CA", element.carbon, thr, "22")
    chain_b = topology.addChain("B")
    water = topology.addResidue("HOH", chain_b, "900")
    oxygen = topology.addAtom("O", element.oxygen, water, "31")
    return topology, (n1, ca1, c1, n2, ca2, oxygen)


class TestDropDuplicateBonds:
    def test_a_pair_listed_twice_is_listed_once_in_either_order(self):
        from binding_metrics.core.system import _drop_duplicate_bonds

        topology, (n1, ca1, c1, n2, ca2, _) = _small_topology()
        topology.addBond(n1, ca1)
        topology.addBond(ca1, c1)
        topology.addBond(c1, n2)
        topology.addBond(n2, c1)  # the same pair, written the other way round
        topology.addBond(ca1, n1)
        topology.addBond(n2, ca2)

        cleaned, n_dropped = _drop_duplicate_bonds(topology)

        assert n_dropped == 2
        assert cleaned.getNumBonds() == 4
        assert _duplicated_pairs(cleaned) == {}

    def test_the_first_listing_is_kept_and_the_order_of_the_others_is_not_changed(self):
        from openmm.app import Double, Single

        from binding_metrics.core.system import _drop_duplicate_bonds

        topology, (n1, ca1, c1, n2, ca2, _) = _small_topology()
        topology.addBond(n1, ca1, Single, 1)
        topology.addBond(c1, n2, Double, 2)
        topology.addBond(ca1, c1, Single, 1)
        topology.addBond(n2, c1, Single, 1)  # later copy with another type: dropped
        topology.addBond(n2, ca2, Single, 1)

        cleaned, _ = _drop_duplicate_bonds(topology)

        bonds = [(b.atom1.name, b.atom2.name, b.type, b.order) for b in cleaned.bonds()]
        assert bonds == [
            ("N", "CA", Single, 1),
            ("C", "N", Double, 2),
            ("CA", "C", Single, 1),
            ("N", "CA", Single, 1),
        ]

    def test_atoms_residues_chains_ids_and_box_are_unchanged(self):
        from binding_metrics.core.system import _drop_duplicate_bonds

        topology, (n1, ca1, c1, n2, _, _) = _small_topology()
        topology.addBond(n1, ca1)
        topology.addBond(c1, n2)
        topology.addBond(n2, c1)

        cleaned, n_dropped = _drop_duplicate_bonds(topology)

        assert n_dropped == 1
        assert [(c.id, c.index) for c in cleaned.chains()] == [("A", 0), ("B", 1)]
        assert [(r.name, r.id, r.insertionCode, r.index) for r in cleaned.residues()] == [
            ("ALA", "7", "", 0),
            ("THR", "7", "A", 1),
            ("HOH", "900", "", 2),
        ]
        assert [(a.name, a.element, a.id, a.index) for a in cleaned.atoms()] == [
            (a.name, a.element, a.id, a.index) for a in topology.atoms()
        ]
        assert cleaned.getPeriodicBoxVectors() == topology.getPeriodicBoxVectors()

    def test_a_topology_without_duplicates_is_returned_as_it_is(self):
        from binding_metrics.core.system import _drop_duplicate_bonds

        topology, (n1, ca1, c1, _, _, _) = _small_topology()
        topology.addBond(n1, ca1)
        topology.addBond(ca1, c1)

        same, n_dropped = _drop_duplicate_bonds(topology)

        assert same is topology
        assert n_dropped == 0


class TestOpenMMSystem:
    """The duplicate is one harmonic bond term too many in the system OpenMM builds."""

    @staticmethod
    def _bond_terms_and_energy(topology, positions, force_field):
        system = force_field.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None)
        bond_force = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
        context = openmm.Context(
            system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        return bond_force.getNumBonds(), energy.value_in_unit(unit.kilojoule_per_mole)

    def test_removing_the_duplicate_gives_back_the_system_of_the_clean_topology(self):
        from binding_metrics.core.system import _drop_duplicate_bonds

        # The p53 peptide of 1YCR with hydrogens, the peptide bond of residues 3 and 4
        # then listed a second time.
        pdb = PDBFile(str(P53_MDM2))
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.delete([r for r in modeller.topology.residues() if r.chain.id != "B"])
        force_field = ForceField("amber14-all.xml")
        modeller.addHydrogens(force_field)
        topology, positions = modeller.topology, modeller.positions
        clean = self._bond_terms_and_energy(topology, positions, force_field)

        residues = list(topology.residues())
        carbon = next(a for a in residues[2].atoms() if a.name == "C")
        nitrogen = next(a for a in residues[3].atoms() if a.name == "N")
        topology.addBond(carbon, nitrogen)
        duplicated = self._bond_terms_and_energy(topology, positions, force_field)
        cleaned, n_dropped = _drop_duplicate_bonds(topology)
        after = self._bond_terms_and_energy(cleaned, positions, force_field)

        assert n_dropped == 1
        assert duplicated[0] == clean[0] + 1  # the symptom: one bond term too many
        assert after[0] == clean[0]
        assert after[1] == pytest.approx(clean[1], abs=1e-6)


@pytest.mark.slow
class TestPrepStructure:
    def test_phosphoserine_peptide_bond_is_listed_once(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(PHOSPHO)
        prepped, _ = prep_structure(topology, positions)

        assert _duplicated_pairs(prepped) == {}
        sep_c_tyr_n = [
            b
            for b in prepped.bonds()
            if {(b.atom1.residue.name, b.atom1.name), (b.atom2.residue.name, b.atom2.name)}
            == {("SEP", "C"), ("TYR", "N")}
        ]
        assert len(sep_c_tyr_n) == 1  # dropped to one copy, not to none

    def test_only_the_duplicate_goes_and_the_structure_is_the_same(self, monkeypatch):
        """Prep with the helper switched off differs from prep with it by that one bond."""
        pytest.importorskip("pdbfixer")
        from binding_metrics.core import system
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(PHOSPHO)
        with_helper, positions_with = system.prep_structure(topology, positions)

        monkeypatch.setattr(system, "_drop_duplicate_bonds", lambda t: (t, 0))
        topology, positions = load_structure(PHOSPHO)
        without_helper, positions_without = system.prep_structure(topology, positions)

        duplicated = _duplicated_pairs(without_helper)
        assert len(duplicated) == 1  # the case this test stands on: SEP C to TYR N
        assert without_helper.getNumBonds() - with_helper.getNumBonds() == 1
        names_without = _bond_names(without_helper)
        names_with = _bond_names(with_helper)
        assert set(names_with) == set(names_without)  # the same bonds ...
        assert sum(names_without.values()) - sum(names_with.values()) == 1  # ... one copy fewer
        assert [(a.name, a.residue.name) for a in with_helper.atoms()] == [
            (a.name, a.residue.name) for a in without_helper.atoms()
        ]
        # Heavy atoms are where they were. The hydrogens settle in a minimisation that sees one
        # bond term fewer; the shift depends on the pdbfixer version, and stays under half an
        # X-H bond length.
        shift_nm = np.abs(
            np.array(positions_with.value_in_unit(unit.nanometer))
            - np.array(positions_without.value_in_unit(unit.nanometer))
        ).max(axis=1)
        heavy = np.array([a.element is not element.hydrogen for a in with_helper.atoms()])
        assert shift_nm[heavy].max() == pytest.approx(0.0, abs=1e-9)
        assert shift_nm.max() < 0.05

    def test_a_lactam_peptide_with_a_non_canonical_residue_has_no_duplicate(self):
        pytest.importorskip("pdbfixer")
        pytest.importorskip("rdkit")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(LACTAM)
        prepped, _ = prep_structure(topology, positions)

        assert _duplicated_pairs(prepped) == {}
