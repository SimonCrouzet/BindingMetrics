"""Chain auto-detection of the relaxation counts D-amino acids as protein (#56).

``ImplicitRelaxation._identify_chains`` ranked the chains by their residues in
``PROTEIN_RESIDUES``, a set without any D-amino acid. On 1YCR with the 13 peptide residues
renamed to their D forms (DGL, DTH, DPN, ...) it returned ``("A", None)``: the receptor as
the peptide and no receptor. ``strip_heterogens`` then removed the whole D-peptide as 13
heterogens, so the standalone relaxation (no chain IDs given) ran on the receptor alone.
The chains now come from ``io.structures.detect_chains``, which counts the D-amino acids, the
phosphorylated residues and the peptide-linking components of the Chemical Component Dictionary.
"""

import pytest
from conftest import DATA_DIR, requires_cuda

pytest.importorskip("openmm")

from binding_metrics.core.nonstandard import D_AA_MAP  # noqa: E402
from binding_metrics.io.structures import (  # noqa: E402
    detect_chains,
    load_structure,
    save_structure,
)
from binding_metrics.protocols.relaxation import (  # noqa: E402
    ImplicitRelaxation,
    RelaxationConfig,
)

P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"

#: Residues of 1YCR: receptor chain A, peptide chain B.
P53_RECEPTOR_RESIDUES = 85
P53_PEPTIDE_RESIDUES = 13

L_TO_D = {l_name: d_name for d_name, l_name in D_AA_MAP.items()}


def _load(path):
    return load_structure(path)


def _flip_to_d(topology, chain_id, every=1):
    """Rename every ``every``-th residue of a chain to its D form; return how many changed."""
    changed = 0
    for index, residue in enumerate(
        r for c in topology.chains() if c.id == chain_id for r in c.residues()
    ):
        if index % every == 0 and residue.name in L_TO_D:
            residue.name = L_TO_D[residue.name]
            changed += 1
    return changed


def _identify(topology, peptide=None, receptor=None):
    config = RelaxationConfig(peptide_chain_id=peptide, receptor_chain_id=receptor)
    return ImplicitRelaxation(config)._identify_chains(topology)


def _strip(topology, positions, pair):
    """The relaxation's own strip step; returns the topology and the report."""
    report: dict = {}
    stripped, _ = ImplicitRelaxation(RelaxationConfig())._strip_heterogens(
        topology, positions, *pair, report=report
    )
    return stripped, report


def _residues_by_chain(topology):
    return {c.id: [r.name for r in c.residues()] for c in topology.chains()}


def _build_topology(chains):
    """Topology with one carbon atom per residue; ``chains`` maps chain id to residue names."""
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


class TestAllDPeptide:
    def test_the_pair_is_found(self):
        topology, _ = _load(P53)
        assert _flip_to_d(topology, "B") == P53_PEPTIDE_RESIDUES

        assert _identify(topology) == ("B", "A")

    def test_it_is_the_pair_that_detect_chains_finds(self):
        topology, _ = _load(P53)
        _flip_to_d(topology, "B")

        assert _identify(topology) == detect_chains(topology)

    def test_the_peptide_survives_the_heterogen_strip(self):
        topology, positions = _load(P53)
        _flip_to_d(topology, "B")

        stripped, report = _strip(topology, positions, _identify(topology))

        sizes = {c.id: len(list(c.residues())) for c in stripped.chains()}
        assert sizes == {"A": P53_RECEPTOR_RESIDUES, "B": P53_PEPTIDE_RESIDUES}
        assert report.get("removed_heterogens", []) == []
        assert report.get("dropped_protein_chains", []) == []

    def test_chain_ids_named_by_the_caller_still_win(self):
        topology, _ = _load(P53)
        _flip_to_d(topology, "B")

        assert _identify(topology, peptide="B", receptor="A") == ("B", "A")
        assert _identify(topology, peptide="A", receptor="B") == ("A", "B")

    def test_one_chain_named_leaves_the_other_to_the_detection(self):
        topology, _ = _load(P53)
        _flip_to_d(topology, "B")

        assert _identify(topology, peptide="B") == ("B", "A")
        assert _identify(topology, receptor="A") == ("B", "A")


class TestMixedPeptide:
    @pytest.mark.parametrize("every", [2, 3, 13])
    def test_the_pair_is_found_and_kept(self, every):
        topology, positions = _load(P53)
        flipped = _flip_to_d(topology, "B", every=every)
        assert 0 < flipped < P53_PEPTIDE_RESIDUES

        pair = _identify(topology)
        stripped, report = _strip(topology, positions, pair)

        assert pair == ("B", "A")
        assert len(_residues_by_chain(stripped)["B"]) == P53_PEPTIDE_RESIDUES
        assert report.get("removed_heterogens", []) == []

    def test_the_d_residues_count_in_the_size_of_the_chain(self):
        """A short receptor chain is the smaller one only when the D-residues are counted."""
        topology, _ = _build_topology(
            {
                "A": ["ALA"] * 10,
                "B": ["DAL", "DLE", "ALA", "DPN", "DSN", "DVA", "LEU", "DTH", "DGL", "DAS"]
                + ["DLY", "DAR", "DTY", "DPR"],
                "C": ["HOH"] * 20,
            }
        )

        assert _identify(topology) == detect_chains(topology) == ("A", "B")


class TestDResiduesNextToModifiedResidues:
    def test_phosphorylated_residues_count_with_the_d_residues(self):
        topology, positions = _build_topology(
            {
                "A": ["ALA"] * 30,
                "B": ["DAL", "SEP", "DPN", "TPO", "DLY", "PTR"],
                "C": ["HOH"] * 15,
            }
        )

        pair = _identify(topology)
        stripped, report = _strip(topology, positions, pair)

        assert pair == ("B", "A")
        assert _residues_by_chain(stripped) == {
            "A": ["ALA"] * 30,
            "B": ["DAL", "SEP", "DPN", "TPO", "DLY", "PTR"],
        }
        assert report["n_removed_waters"] == 15

    def test_a_noncanonical_amino_acid_from_the_component_dictionary_counts(self):
        pytest.importorskip("biotite")
        topology, positions = _build_topology(
            {
                "A": ["ALA"] * 30,
                "B": ["DAL", "ABA", "DPN", "ORN", "DLY"],
            }
        )

        pair = _identify(topology)
        stripped, report = _strip(topology, positions, pair)

        assert pair == ("B", "A")
        assert len(_residues_by_chain(stripped)["B"]) == 5
        assert report.get("removed_heterogens", []) == []

    def test_a_ligand_chain_does_not_become_the_peptide(self):
        topology, positions = _build_topology(
            {
                "A": ["ALA"] * 30,
                "B": ["DAL", "DPN", "DLY", "DSN"],
                "C": ["ZN", "ZN"],
            }
        )

        pair = _identify(topology)
        stripped, report = _strip(topology, positions, pair)

        assert pair == ("B", "A")
        assert set(_residues_by_chain(stripped)) == {"A", "B"}
        assert report["removed_heterogens"] == ["ZN (chain C)", "ZN (chain C)"]


class TestNoProteinChain:
    def test_a_structure_without_amino_acids_raises(self):
        topology, _ = _build_topology({"A": ["HOH"] * 5, "B": ["ZN"]})

        with pytest.raises(ValueError, match="No protein chains found in structure"):
            _identify(topology)

    def test_it_raises_also_when_the_ids_are_named(self):
        topology, _ = _build_topology({"A": ["HOH"] * 5})

        with pytest.raises(ValueError, match="No protein chains found in structure"):
            _identify(topology, peptide="A", receptor="B")

    def test_a_single_protein_chain_has_no_receptor(self):
        topology, _ = _build_topology({"A": ["DAL", "DLE", "DPN"], "B": ["HOH"] * 4})

        assert _identify(topology) == ("A", None)


class TestOneDefinitionOfAProteinChain:
    """The relaxation reads the chains as ``detect_chains``, on every bundled example."""

    @pytest.mark.parametrize(
        "name",
        [
            "example_linear_p53_1YCR.pdb",
            "example_ncaa_cyclosporin_1CWA.cif",
            "example_bicyclic_sfti1_3P8F.cif",
            "example_lactam_somatostatin_1XY4.cif",
            "example_phospho_1QJB.pdb",
            "example_staple_3V3B.pdb",
        ],
    )
    def test_the_bundled_example(self, name):
        topology, _ = _load(DATA_DIR / name)

        assert _identify(topology) == detect_chains(topology)

    def test_cyclosporin_with_its_d_alanine_keeps_the_ring(self):
        topology, positions = _load(CYCLOSPORIN)
        peptide, receptor = _identify(topology)

        stripped, report = _strip(topology, positions, (peptide, receptor))

        assert (peptide, receptor) == ("B", "A")
        assert _residues_by_chain(topology)[peptide][0] == "DAL"
        assert len(_residues_by_chain(stripped)[peptide]) == 11
        assert report.get("removed_heterogens", []) == []

    def test_cyclosporin_with_every_standard_residue_in_d_form(self):
        """The ring has two standard L residues (VAL, ALA) besides the N-methylated ones."""
        topology, positions = _load(CYCLOSPORIN)
        before = detect_chains(topology)
        assert _flip_to_d(topology, "B") == 2

        assert _identify(topology) == before == ("B", "A")
        stripped, _ = _strip(topology, positions, before)
        assert len(_residues_by_chain(stripped)["B"]) == 11


@pytest.fixture(scope="module")
def all_d_prepped(prepped_example_cif, tmp_path_factory):
    """The prepped 1YCR with its 13 peptide residues renamed to the D forms."""
    topology, positions = load_structure(prepped_example_cif)
    assert _flip_to_d(topology, "B") == P53_PEPTIDE_RESIDUES
    path = tmp_path_factory.mktemp("all_d") / "all_d_1YCR_prepped.cif"
    save_structure(topology, positions, path, source_path=prepped_example_cif)
    return path


@pytest.mark.integration
class TestStandaloneRelaxationOfAnAllDPeptide:
    """No chain ID is given, as in ``binding-metrics-relax`` on a file."""

    def test_setup_keeps_the_peptide(self, all_d_prepped):
        relaxer = ImplicitRelaxation(RelaxationConfig(md_duration_ps=0.0))

        _, topology, _, _ = relaxer._setup_system(all_d_prepped)

        sizes = {chain.id: len(list(chain.residues())) for chain in topology.chains()}
        assert relaxer._chain_ids == ("B", "A")
        assert relaxer._dropped_protein_chains == []
        assert sizes == {"A": P53_RECEPTOR_RESIDUES, "B": P53_PEPTIDE_RESIDUES}
        assert len(relaxer._ns_info.d_residues) == P53_PEPTIDE_RESIDUES

    @requires_cuda
    def test_the_relaxed_file_holds_receptor_and_d_peptide(self, all_d_prepped, tmp_path):
        import gemmi

        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=200,
            min_steps_restrained=100,
            min_steps_final=200,
            device="cuda",
        )

        result = ImplicitRelaxation(config).run(all_d_prepped, tmp_path / "relaxed")

        assert result.success, result.error_message
        model = gemmi.read_structure(str(result.minimized_structure_path))[0]
        assert {chain.name: len(chain) for chain in model} == {
            "A": P53_RECEPTOR_RESIDUES,
            "B": P53_PEPTIDE_RESIDUES,
        }
        assert [residue.name for residue in model["B"]][:3] == ["DGL", "DTH", "DPN"]
