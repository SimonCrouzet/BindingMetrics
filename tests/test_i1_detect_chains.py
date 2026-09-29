"""``io.structures.detect_chains`` must see chains made only of D-amino acids.

``compute_interaction_energy`` uses it to find the peptide and the receptor when the
caller names neither. It used to recognise proteins by ``PROTEIN_RESIDUES``, which has no
D-residue, so a D-peptide binder was not a chain at all.
"""

import pytest

pytest.importorskip("openmm")

from openmm.app import Topology, element


def _topology(chains: dict) -> Topology:
    """One backbone-free residue per entry: ``{chain_id: [residue names]}``."""
    topology = Topology()
    for chain_id, names in chains.items():
        chain = topology.addChain(chain_id)
        for i, name in enumerate(names, 1):
            residue = topology.addResidue(name, chain, str(i))
            topology.addAtom("CA", element.carbon, residue)
    return topology


class TestDetectChains:
    def test_a_d_peptide_is_the_ligand_chain(self):
        from binding_metrics.io.structures import detect_chains

        topology = _topology({"A": ["ALA"] * 20, "B": ["DAL"] * 5})
        assert detect_chains(topology) == ("B", "A")

    def test_a_d_only_receptor_is_still_a_chain(self):
        from binding_metrics.io.structures import detect_chains

        topology = _topology({"A": ["DLE"] * 12, "B": ["ALA"] * 4})
        assert detect_chains(topology) == ("B", "A")

    def test_phosphorylated_and_non_canonical_residues_count(self):
        from binding_metrics.io.structures import detect_chains

        topology = _topology({"A": ["SEP", "TPO", "PTR", "MSE", "ALA", "ALA"], "B": ["ALA"] * 2})
        assert detect_chains(topology) == ("B", "A")

    def test_waters_ions_and_ligands_are_not_chains(self):
        from binding_metrics.io.structures import detect_chains

        topology = _topology({"A": ["ALA"] * 6, "B": ["HOH"] * 30, "C": ["ZN"], "D": ["GLC"] * 3})
        assert detect_chains(topology) == ("A", None)

    def test_only_a_ligand_gives_no_chain(self):
        from binding_metrics.io.structures import detect_chains

        assert detect_chains(_topology({"A": ["HOH"] * 3})) == (None, None)

    def test_l_amino_acids_behave_as_before(self):
        from binding_metrics.io.structures import detect_chains

        topology = _topology({"A": ["ALA"] * 20, "B": ["GLY"] * 5, "C": ["VAL"] * 9})
        assert detect_chains(topology) == ("B", "A")
