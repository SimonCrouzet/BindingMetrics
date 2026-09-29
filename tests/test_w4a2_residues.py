"""Residue and water name sets of core, io and protocols (``binding_metrics.core.residues``).

The reference literals below are the hand-written sets that ``io.structures``,
``core.system``, ``core.gaff_ncaa``, ``protocols.relaxation`` and ``protocols.qc`` carried
before they imported the shared constants. Each test pins the shared constant to its old
literal, so the refactor is documented and no residue is silently added or dropped. The
behaviour tests check that the consuming functions still classify residues the same way.
"""

import pytest

from binding_metrics.core import residues

# --- old literals -----------------------------------------------------------------

# io.structures.detect_chains, io.structures.strip_heterogens and
# ImplicitRelaxation._identify_chains each spelled out this same set.
OLD_PROTEIN_RESIDUES = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "CYX HID HIE HIP ASPL GLUL LYSL NMG NMA MVA MLE"
    ).split()
)


def _build_topology(chains):
    """Topology with one carbon atom per residue; ``chains`` maps chain id to residue names."""
    pytest.importorskip("openmm")
    from openmm import Vec3, app, unit

    topology = app.Topology()
    coordinates = []
    for chain_id, names in chains.items():
        chain = topology.addChain(id=chain_id)
        for name in names:
            residue = topology.addResidue(name, chain)
            topology.addAtom("CA", app.element.carbon, residue)
            coordinates.append(Vec3(0.3 * len(coordinates), 0.0, 0.0))
    return topology, unit.Quantity(coordinates, unit.nanometer)


class TestProteinResidues:
    def test_constant_equals_the_old_literal(self):
        assert residues.PROTEIN_RESIDUES == OLD_PROTEIN_RESIDUES
        assert len(residues.PROTEIN_RESIDUES) == 31

    def test_building_blocks_are_disjoint_from_the_standard_amino_acids(self):
        assert not residues.STANDARD_AMINO_ACIDS & residues.LACTAM_TEMPLATE_RESIDUES
        assert not residues.STANDARD_AMINO_ACIDS & residues.N_METHYLATED_RESIDUES
        assert residues.LACTAM_TEMPLATE_RESIDUES == {"ASPL", "GLUL", "LYSL"}
        assert residues.N_METHYLATED_RESIDUES == {"NMG", "NMA", "MVA", "MLE"}

    def test_variants_left_out_of_the_set_stay_out(self):
        assert not {"HIN", "ASH", "GLH", "LYN", "CYM"} & residues.PROTEIN_RESIDUES

    def test_detect_chains_counts_lactam_and_n_methylated_residues(self):
        from binding_metrics.io.structures import detect_chains

        topology, _ = _build_topology(
            {
                "A": ["ASPL", "MLE", "NMG"],
                "B": ["ALA"] * 5,
                "C": ["HOH"] * 10 + ["HIN"] * 8,
            }
        )

        assert detect_chains(topology) == ("A", "B")

    def test_identify_chains_counts_lactam_and_n_methylated_residues(self):
        pytest.importorskip("openmm")
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        topology, _ = _build_topology(
            {
                "A": ["LYSL", "NMA", "MVA"],
                "B": ["GLY"] * 5,
                "C": ["HOH"] * 10 + ["CYM"] * 8,
            }
        )

        relaxer = ImplicitRelaxation(RelaxationConfig())

        assert relaxer._identify_chains(topology) == ("A", "B")

    def test_strip_heterogens_keeps_protein_named_residues_outside_the_chains(self):
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = _build_topology(
            {
                "A": ["ALA", "GLY"],
                "B": ["ALA"] * 4,
                "C": ["MLE", "ASPL", "HIN", "ZN"],
            }
        )
        report: dict = {}

        stripped, _ = strip_heterogens(topology, positions, "A", "B", report=report)

        names = [residue.name for residue in stripped.residues()]
        assert names == ["ALA", "GLY"] + ["ALA"] * 4 + ["MLE", "ASPL"]
        assert report["removed_heterogens"] == ["HIN (chain C)", "ZN (chain C)"]
