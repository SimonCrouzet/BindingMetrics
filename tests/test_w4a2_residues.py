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

# core.system._STANDARD_RESIDUES: amino acids, protonation variants, nucleotides.
OLD_SYSTEM_STANDARD_RESIDUES = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "HIE HID HIP CYX ASH GLH LYN "
        "DA DC DG DT A C G T U"
    ).split()
)

# core.system._WATER_NAMES.
OLD_SYSTEM_WATER_NAMES = frozenset({"HOH", "WAT", "SOL", "TIP", "TIP3", "H2O"})

# core.system.get_system_info: local ``ion_names``.
OLD_SYSTEM_ION_NAMES = frozenset({"NA", "CL", "K", "MG", "CA", "ZN"})

# core.cyclic.get_addh_variants: local ``_custom_h_residues`` tuple.
OLD_CUSTOM_HYDROGEN_RESIDUES = frozenset(
    ("CYX", "ASPL", "GLUL", "LYSL", "NMG", "NMA", "MVA", "MLE")
)

# core.cyclic.rename_disulfide_cys_to_cyx: inline ``("CYS", "CYX")``.
OLD_CYCLIC_CYSTEINE_NAMES = frozenset(("CYS", "CYX"))

# core.gaff_ncaa._BACKBONE_HEAVY, ImplicitRelaxation._add_restraints (``backbone_names``)
# and, with OXT added, core.cyclic._BACKBONE_ATOM_NAMES.
OLD_BACKBONE_HEAVY_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})
OLD_CYCLIC_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

# protocols.qc.WATER_NAMES.
OLD_QC_WATER_NAMES = frozenset({"HOH", "WAT", "H2O"})

# protocols.relaxation.ImplicitRelaxation._AMBER_STANDARD.
OLD_RELAXATION_AMBER_STANDARD = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "CYX HID HIE HIP HIN LYN ASH GLH ASPL GLUL LYSL ACE NME NMA FOR "
        "HOH WAT H2O NA CL K MG CA ZN"
    ).split()
)

# core.gaff_ncaa.GAFF_SKIP_RESIDUES: standard residues and variants, curated templates,
# phospho residues, caps, nucleotides, waters and ions.
OLD_GAFF_SKIP_RESIDUES = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "CYX HID HIE HIP HIN LYN ASH GLH "
        "NMG NMA MVA MLE ASPL GLUL LYSL "
        "SEP TPO PTR S1P T1P Y1P H1D H2D H1E H2E "
        "ACE NME FOR "
        "DA DC DG DT A C G T U "
        "HOH WAT H2O SOL TIP TIP3 NA CL K MG CA ZN LI RB CS FE MN CU"
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


class TestSystemSets:
    def test_standard_residues_equal_the_old_literal(self):
        # CYM (deprotonated cysteine) joined the standard variants afterwards.
        assert residues.AMBER_STANDARD_RESIDUES == OLD_SYSTEM_STANDARD_RESIDUES | {"CYM"}
        assert len(residues.AMBER_STANDARD_RESIDUES) == 37

    def test_standard_residues_leave_out_hin(self):
        assert "HIN" not in residues.AMBER_STANDARD_RESIDUES
        assert residues.AMBER_STANDARD_VARIANTS == residues.AMBER_PROTONATION_VARIANTS

    def test_nucleotides(self):
        assert residues.NUCLEOTIDE_RESIDUES == {"DA", "DC", "DG", "DT", "A", "C", "G", "T", "U"}

    def test_all_water_names_equal_the_old_literal(self):
        assert residues.WATER_NAMES_ALL == OLD_SYSTEM_WATER_NAMES

    def test_ion_names_equal_the_old_literal(self):
        assert residues.ION_NAMES_COMMON == OLD_SYSTEM_ION_NAMES

    def test_the_water_sets_nest(self):
        assert residues.WATER_NAMES_PDB_AMBER < residues.WATER_NAMES_WITH_H2O
        assert residues.WATER_NAMES_WITH_H2O < residues.WATER_NAMES_ALL
        assert residues.WATER_NAMES_STRIP_HETEROGENS < residues.WATER_NAMES_ALL

    def test_system_info_counts_hoh_and_wat_as_water_and_the_common_ions(self):
        from types import SimpleNamespace

        from binding_metrics.core.system import get_system_info

        names = ["ALA", "HOH", "WAT", "SOL", "H2O", "NA", "CL", "K", "MG", "CA", "ZN", "FE"]
        topology, _ = _build_topology({"A": names})

        info = get_system_info(SimpleNamespace(topology=topology))

        assert info["n_waters"] == 2
        assert info["n_ions"] == 6


class TestGaffSkipResidues:
    def test_skip_list_equals_the_old_literal(self):
        from binding_metrics.core.gaff_ncaa import GAFF_SKIP_RESIDUES

        assert GAFF_SKIP_RESIDUES == OLD_GAFF_SKIP_RESIDUES | {"CYM"}
        assert len(GAFF_SKIP_RESIDUES) == 76

    def test_cap_names_differ_from_the_terminal_caps_by_nh2_and_for(self):
        assert residues.FORCE_FIELD_CAP_NAMES == {"ACE", "NME", "FOR"}
        assert "NH2" in residues.TERMINAL_CAP_NAMES
        assert "NH2" not in residues.FORCE_FIELD_CAP_NAMES

    @pytest.mark.parametrize(
        ("residue_name", "is_ncaa"),
        [
            ("ACE", False),
            ("S1P", False),
            ("MLE", False),
            ("LIG", True),
        ],
    )
    def test_multi_atom_residues_are_ncaa_unless_their_name_is_skipped(self, residue_name, is_ncaa):
        pytest.importorskip("openmm")
        from openmm import app

        from binding_metrics.core.gaff_ncaa import _is_ncaa

        topology = app.Topology()
        residue = topology.addResidue(residue_name, topology.addChain(id="A"))
        topology.addAtom("C1", app.element.carbon, residue)
        topology.addAtom("N1", app.element.nitrogen, residue)

        assert _is_ncaa(residue) is is_ncaa


class TestQcWaterNames:
    def test_qc_water_names_equal_the_old_literal(self):
        from binding_metrics.protocols import qc

        assert qc.WATER_NAMES == OLD_QC_WATER_NAMES
        assert residues.WATER_NAMES_WITH_H2O == OLD_QC_WATER_NAMES

    def test_snapshot_marks_hoh_wat_and_h2o_as_water_but_not_sol_or_tip3(self):
        from binding_metrics.protocols.qc import AtomSnapshot

        topology, positions = _build_topology({"A": ["ALA", "HOH", "WAT", "H2O", "SOL", "TIP3"]})

        snapshot = AtomSnapshot.from_topology(topology, positions)

        assert snapshot.is_water.tolist() == [False, True, True, True, False, False]


class TestRelaxationAmberStandard:
    def test_set_equals_the_old_literal(self):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation

        assert ImplicitRelaxation._AMBER_STANDARD == OLD_RELAXATION_AMBER_STANDARD | {"CYM"}
        assert len(ImplicitRelaxation._AMBER_STANDARD) == 45

    def test_only_residues_outside_the_set_become_gaff_molecules(self):
        pytest.importorskip("openff.toolkit")
        pytest.importorskip("rdkit")
        from openmm import app

        from binding_metrics.protocols.relaxation import ImplicitRelaxation

        topology = app.Topology()
        chain = topology.addChain(id="A")
        for name in ["ALA", "HIN", "NMA", "ACE", "HOH", "NA"]:
            topology.addAtom("C1", app.element.carbon, topology.addResidue(name, chain))
        ligand = topology.addResidue("LIG", chain)
        first = topology.addAtom("C1", app.element.carbon, ligand)
        second = topology.addAtom("N1", app.element.nitrogen, ligand)
        topology.addBond(first, second)

        molecules = ImplicitRelaxation._discover_heterogens(topology)

        assert [molecule.n_atoms for molecule in molecules] == [2]


class TestCyclicNameSets:
    def test_custom_hydrogen_residues_equal_the_old_literal(self):
        assert residues.CUSTOM_HYDROGEN_RESIDUES == OLD_CUSTOM_HYDROGEN_RESIDUES

    def test_cysteine_names_equal_the_old_literal(self):
        assert residues.CYSTEINE_NAMES == OLD_CYCLIC_CYSTEINE_NAMES

    def test_addh_variants_are_given_to_the_custom_residues_only(self):
        from binding_metrics.core.cyclic import get_addh_variants

        names = ["ALA", "CYX", "ASPL", "GLUL", "LYSL", "NMG", "NMA", "MVA", "MLE", "HIN", "CYS"]
        topology, _ = _build_topology({"A": names})

        variants = get_addh_variants(topology, [], "A")

        assert [variant is not None for variant in variants] == [
            name in residues.CUSTOM_HYDROGEN_RESIDUES for name in names
        ]
        assert sum(variant is not None for variant in variants) == 8


class TestBackboneAtomNames:
    def test_constant_equals_the_old_literal(self):
        assert residues.BACKBONE_HEAVY_ATOM_NAMES == OLD_BACKBONE_HEAVY_ATOM_NAMES

    def test_gaff_and_cyclic_sets_equal_their_old_literals(self):
        from binding_metrics.core import cyclic, gaff_ncaa

        assert gaff_ncaa._BACKBONE_HEAVY == OLD_BACKBONE_HEAVY_ATOM_NAMES
        assert cyclic._BACKBONE_ATOM_NAMES == OLD_CYCLIC_BACKBONE_ATOM_NAMES

    def test_backbone_restraint_covers_the_four_backbone_atoms_only(self):
        pytest.importorskip("openmm")
        import openmm
        from openmm import app

        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        topology = app.Topology()
        residue = topology.addResidue("ALA", topology.addChain(id="A"))
        for name in ["N", "CA", "C", "O", "CB", "OXT"]:
            topology.addAtom(name, app.element.carbon, residue)
        positions = [openmm.Vec3(0.1 * i, 0.0, 0.0) for i in range(6)]
        system = openmm.System()
        for _ in positions:
            system.addParticle(12.0)

        relaxer = ImplicitRelaxation(RelaxationConfig())
        force_index = relaxer._add_restraints(system, topology, positions, backbone_only=True)

        restrained = [system.getForce(force_index).getParticleParameters(i)[0] for i in range(4)]
        assert system.getForce(force_index).getNumParticles() == 4
        assert restrained == [0, 1, 2, 3]
