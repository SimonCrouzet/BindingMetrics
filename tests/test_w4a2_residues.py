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

# io.structures.strip_heterogens: local ``_water_names``.
OLD_STRIP_HETEROGENS_WATERS = frozenset({"HOH", "WAT", "TIP", "TIP3", "SOL"})

# core.system._METAL_ELEMENTS and core.gaff_ncaa._METAL_SYMBOLS were the same 30 symbols.
OLD_METAL_ELEMENTS = frozenset(
    (
        "Li Na K Rb Cs Mg Ca Sr Ba V Cr Mn Fe Co Ni Cu Zn Mo Ru Rh Pd Ag Cd W Re Os Ir Pt Au Hg"
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


class TestStripHeterogensWaters:
    def test_constant_equals_the_old_literal(self):
        assert residues.WATER_NAMES_STRIP_HETEROGENS == OLD_STRIP_HETEROGENS_WATERS

    def test_h2o_is_not_one_of_the_stripped_solvent_names(self):
        assert "H2O" not in residues.WATER_NAMES_STRIP_HETEROGENS
        assert residues.WATER_MODEL_NAMES == {"SOL", "TIP", "TIP3"}

    def test_waters_are_removed_and_counted_but_h2o_is_a_heterogen(self):
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = _build_topology(
            {
                "A": ["ALA", "GLY"],
                "B": ["ALA"] * 4,
                "C": ["HOH", "WAT", "TIP", "TIP3", "SOL", "H2O"],
            }
        )
        report: dict = {}

        stripped, _ = strip_heterogens(topology, positions, "A", "B", report=report)

        assert [residue.name for residue in stripped.residues()] == ["ALA", "GLY"] + ["ALA"] * 4
        assert report["n_removed_waters"] == 5
        assert report["removed_heterogens"] == ["H2O (chain C)"]


class TestMetalElements:
    def test_constant_equals_the_old_literal(self):
        assert residues.METAL_ELEMENTS == OLD_METAL_ELEMENTS
        assert len(residues.METAL_ELEMENTS) == 30

    def test_gaff_skips_metal_only_residues_and_keeps_organic_ones(self):
        pytest.importorskip("openmm")
        from openmm import app

        from binding_metrics.core.gaff_ncaa import _is_ncaa

        topology = app.Topology()
        chain = topology.addChain(id="A")
        cluster = topology.addResidue("FES", chain)
        topology.addAtom("FE1", app.element.iron, cluster)
        topology.addAtom("FE2", app.element.iron, cluster)
        organic = topology.addResidue("BMT", chain)
        topology.addAtom("C1", app.element.carbon, organic)
        topology.addAtom("N1", app.element.nitrogen, organic)

        residues_by_name = {residue.name: residue for residue in topology.residues()}

        assert not _is_ncaa(residues_by_name["FES"])
        assert _is_ncaa(residues_by_name["BMT"])


class TestPrepStructureClassification:
    """``prep_structure`` sorts residues with the shared standard, metal and water sets."""

    def test_waters_are_removed_a_metal_is_kept_and_a_ligand_is_stripped(self, example_pdb_path):
        pytest.importorskip("pdbfixer")
        from openmm import Vec3, app, unit

        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(example_pdb_path)
        extra = app.Topology()
        coordinates = []

        def add_residue(chain_id, name, atoms):
            residue = extra.addResidue(name, extra.addChain(id=chain_id))
            for atom_name, element in atoms:
                extra.addAtom(atom_name, element, residue)
                coordinates.append(Vec3(6.0 + 0.4 * len(coordinates), 6.0, 6.0))

        for name in ["HOH", "WAT", "SOL", "TIP3", "TIP", "H2O"]:
            add_residue("W", name, [("O", app.element.oxygen)])
        add_residue("Z", "ZN", [("ZN", app.element.zinc)])
        add_residue("L", "LIG", [("C1", app.element.carbon), ("O1", app.element.oxygen)])
        modeller = app.Modeller(topology, positions)
        modeller.add(extra, unit.Quantity(coordinates, unit.nanometer))
        report: dict = {}

        prepped, _ = prep_structure(
            modeller.topology, modeller.positions, keep_water=False, report=report
        )

        assert report["n_removed_waters"] == 6
        assert report["kept_nonstandard"] == ["ZN (metal, chain Z)"]
        assert report["removed_heterogens"] == ["LIG (chain L)"]
        assert {residue.name for residue in prepped.residues()} >= {"ZN"}
        assert not {"HOH", "WAT", "SOL", "TIP3", "TIP", "H2O", "LIG"} & {
            residue.name for residue in prepped.residues()
        }
