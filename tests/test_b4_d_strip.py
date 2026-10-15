"""A chain of D-amino acids is a protein chain for the heterogen strip.

``strip_heterogens`` kept the residues of ``PROTEIN_RESIDUES`` and the phosphorylated ones
in a chain that is neither the peptide nor the receptor. Neither set has a D-amino acid, so
a third chain of D residues was removed as heterogens and never reached
``drop_other_protein_chains``, which reports it as a dropped protein chain. The strip now
keeps what ``detect_chains`` and ``drop_other_protein_chains`` call an amino acid.
"""

import pytest

pytest.importorskip("openmm")

from openmm import Vec3, app, unit

from binding_metrics.io.structures import drop_other_protein_chains, strip_heterogens


def _chain_sizes(topology):
    return {c.id: len(list(c.residues())) for c in topology.chains()}


def _topology(chains):
    """One carbon atom per residue; ``chains`` maps a chain ID to its residue names."""
    topology = app.Topology()
    coordinates = []
    for chain_id, names in chains.items():
        chain = topology.addChain(id=chain_id)
        for name in names:
            residue = topology.addResidue(name, chain)
            topology.addAtom("CA", app.element.carbon, residue)
            coordinates.append(Vec3(0.3 * len(coordinates), 0.0, 0.0))
    return topology, unit.Quantity(coordinates, unit.nanometer)


class TestStripKeepsADChain:
    def _strip(self, chains):
        topology, positions = _topology(chains)
        report: dict = {}
        stripped, stripped_positions = strip_heterogens(
            topology, positions, "B", "A", report=report
        )
        stripped, _ = drop_other_protein_chains(
            stripped, stripped_positions, "B", "A", report=report
        )
        return stripped, report

    def test_a_third_chain_of_d_residues_is_a_dropped_protein_chain(self):
        stripped, report = self._strip({"A": ["ALA"] * 30, "B": ["DAL"] * 4, "C": ["DLE"] * 6})

        assert report["removed_heterogens"] == []
        assert report["dropped_protein_chains"] == ["C"]
        assert _chain_sizes(stripped) == {"A": 30, "B": 4}

    def test_a_third_chain_of_l_residues_is_reported_as_before(self):
        _, report = self._strip({"A": ["ALA"] * 30, "B": ["DAL"] * 4, "C": ["LEU"] * 6})

        assert report["removed_heterogens"] == []
        assert report["dropped_protein_chains"] == ["C"]

    def test_the_d_residues_of_an_unselected_chain_are_not_cut_out_of_it(self):
        """Removing them as heterogens would leave the chain in two pieces."""
        topology, positions = _topology(
            {"A": ["ALA"] * 30, "B": ["DAL"] * 4, "C": ["LEU", "DLE", "LEU", "DAL", "GLY"]}
        )
        stripped, _ = strip_heterogens(topology, positions, "B", "A")

        assert _chain_sizes(stripped)["C"] == 5

    def test_a_ligand_and_a_water_are_still_removed(self):
        stripped, report = self._strip(
            {"A": ["ALA"] * 30, "B": ["DAL"] * 4, "L": ["EDO"], "W": ["HOH", "HOH"]}
        )

        assert report["removed_heterogens"] == ["EDO (chain L)"]
        assert report["n_removed_waters"] == 2
        assert _chain_sizes(stripped) == {"A": 30, "B": 4}
