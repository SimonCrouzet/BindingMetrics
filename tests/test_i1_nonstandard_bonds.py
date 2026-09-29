"""Bonds around a non-standard residue that OpenMM loads without any bond.

``createStandardBonds`` knows no bond for an unknown residue name (S-palmitoyl-cysteine
P1L in 6SBA, an NCAA, a D-amino acid), and ``PDBxFile`` drops the ``struct_conn`` row that
would give the C(i-1)-N(i) peptide bond when label and author residue numbers differ.
The fixture is the bundled 1YCR (MDM2 chain A, p53 peptide chain B) with one residue
renamed and stripped of exactly those bonds, so the expected result is the original
bond list.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")

from openmm import app

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"

UNKNOWN = "XYZ"


def _bond_keys(topology):
    return {
        frozenset(
            {
                (b.atom1.residue.chain.id, b.atom1.residue.id, b.atom1.name),
                (b.atom2.residue.chain.id, b.atom2.residue.id, b.atom2.name),
            }
        )
        for b in topology.bonds()
    }


def _rename_and_unbond(chain_id: str, residue_id: str, new_name: str = UNKNOWN):
    """1YCR with one residue renamed and without the bonds OpenMM leaves out for it.

    Returns ``(topology, positions, original_bond_keys)``. The original keys keep the
    residue's chain and number; only names differ from the topology handed back.
    """
    pdb = app.PDBFile(str(P53_MDM2))
    original = pdb.topology
    target = next(r for r in original.residues() if r.chain.id == chain_id and r.id == residue_id)
    target_index = target.index
    topology = app.Topology()
    atoms = {}
    for chain in original.chains():
        new_chain = topology.addChain(chain.id)
        for res in chain.residues():
            name = new_name if res.index == target_index else res.name
            new_res = topology.addResidue(name, new_chain, res.id)
            for atom in res.atoms():
                atoms[atom.index] = topology.addAtom(atom.name, atom.element, new_res)
    for bond in original.bonds():
        residues = {bond.atom1.residue.index, bond.atom2.residue.index}
        if target_index in residues:
            continue  # its internal bonds and the peptide bonds on both sides
        topology.addBond(atoms[bond.atom1.index], atoms[bond.atom2.index])
    return topology, pdb.positions, _bond_keys(original)


def _n_bonds_of(topology, chain_id: str, residue_id: str) -> int:
    return sum(
        1
        for b in topology.bonds()
        if any(
            a.residue.chain.id == chain_id and a.residue.id == residue_id
            for a in (b.atom1, b.atom2)
        )
    )


class TestReconstructNonstandardResidueBonds:
    def test_restores_the_internal_bonds_and_both_peptide_bonds(self):
        from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds

        topology, positions, original = _rename_and_unbond("B", "22")
        assert _bond_keys(topology) != original
        assert _n_bonds_of(topology, "B", "22") == 0  # the fault: an isolated cloud of atoms

        added = reconstruct_nonstandard_residue_bonds(topology, positions)

        assert _bond_keys(topology) == original
        assert added == 9
        # LEU has 8 heavy atoms and 7 bonds inside, plus C(21)-N(22) and C(22)-N(23).
        assert _n_bonds_of(topology, "B", "22") == 9

    def test_a_second_call_adds_nothing(self):
        from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds

        topology, positions, _ = _rename_and_unbond("B", "22")
        reconstruct_nonstandard_residue_bonds(topology, positions)
        n_bonds = topology.getNumBonds()
        assert reconstruct_nonstandard_residue_bonds(topology, positions) == 0
        assert topology.getNumBonds() == n_bonds

    def test_a_fully_standard_structure_gains_no_bond(self):
        from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds

        pdb = app.PDBFile(str(P53_MDM2))
        n_bonds = pdb.topology.getNumBonds()
        assert reconstruct_nonstandard_residue_bonds(pdb.topology, pdb.positions) == 0
        assert pdb.topology.getNumBonds() == n_bonds

    def test_a_chain_break_is_not_bridged(self):
        """With the next residue gone, C(i) and N(i+2) are about 0.5 nm apart: no bond."""
        from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds

        topology, positions, _ = _rename_and_unbond("B", "22")
        following = next(r for r in topology.residues() if r.chain.id == "B" and r.id == "23")
        modeller = app.Modeller(topology, positions)
        modeller.delete(list(following.atoms()))
        topology, positions = modeller.topology, modeller.positions

        reconstruct_nonstandard_residue_bonds(topology, positions)

        assert _n_bonds_of(topology, "B", "22") == 7 + 1  # internal bonds and C(21)-N(22) only

    def test_a_residue_in_the_receptor_chain_is_repaired(self):
        """The residue sits in chain A, not in the peptide chain B."""
        from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds

        topology, positions, original = _rename_and_unbond("A", "50")
        added = reconstruct_nonstandard_residue_bonds(topology, positions)
        assert added > 0
        assert _bond_keys(topology) == original

    def test_a_chain_of_unknown_residues_is_repaired_only_when_named(self):
        """A peptide made only of D-residues holds no standard amino acid."""
        from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds

        pdb = app.PDBFile(str(P53_MDM2))
        original = pdb.topology
        residues = [r for r in original.residues() if r.chain.id == "B"][3:6]
        topology = app.Topology()
        chain = topology.addChain("Z")
        atoms = {}
        positions = []
        for i, res in enumerate(residues):
            new_res = topology.addResidue(f"X{i}", chain, str(i + 1))
            for atom in res.atoms():
                atoms[atom.index] = topology.addAtom(atom.name, atom.element, new_res)
                positions.append(pdb.positions[atom.index])
        assert topology.getNumBonds() == 0

        assert reconstruct_nonstandard_residue_bonds(topology, positions) == 0
        added = reconstruct_nonstandard_residue_bonds(topology, positions, include_chain="Z")
        wanted = sum(
            1
            for b in original.bonds()
            if b.atom1.index in atoms and b.atom2.index in atoms  # both in the three residues
        )
        assert added == wanted
        assert topology.getNumBonds() == wanted


class TestPatchCyclicTopologyRepairsEveryChain:
    def test_a_receptor_residue_gets_its_bonds_before_the_cyclic_detection(self):
        from binding_metrics.core.cyclic import patch_cyclic_topology

        topology, positions, original = _rename_and_unbond("A", "50")

        _, _, bond_info = patch_cyclic_topology(topology, positions, "B")

        assert bond_info == []  # the p53 peptide is linear
        assert _bond_keys(topology) == original

    def test_the_peptide_chain_is_still_repaired(self):
        from binding_metrics.core.cyclic import patch_cyclic_topology

        topology, positions, original = _rename_and_unbond("B", "22")

        patch_cyclic_topology(topology, positions, "B")

        assert _bond_keys(topology) == original
