"""The relaxation reads its chain options as author IDs and reports author IDs (#64).

1CWA: the peptide is author chain C and chain B of the OpenMM topology, whose chain C holds
140 waters. Before the fix, ``RelaxationConfig(peptide_chain_id="C")`` on the raw file took the
waters for the peptide, dropped the real peptide as a third protein chain and finished with
``success`` true.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

sys.path.insert(0, str(Path(__file__).parent))
from conftest import requires_cuda  # noqa: E402

from binding_metrics.io.structures import author_chain_ids, load_structure  # noqa: E402
from binding_metrics.protocols.qc import AtomSnapshot  # noqa: E402
from binding_metrics.protocols.relaxation import (  # noqa: E402
    ImplicitRelaxation,
    RelaxationConfig,
    RelaxationResult,
)

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"

try:
    import openmmforcefields  # noqa: F401

    HAS_OMMFF = True
except ImportError:
    HAS_OMMFF = False
requires_ommff = pytest.mark.skipif(not HAS_OMMFF, reason="openmmforcefields not installed")


def _load(path):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return load_structure(path)


def _identified(path, peptide, receptor):
    topology, _ = _load(path)
    config = RelaxationConfig(peptide_chain_id=peptide, receptor_chain_id=receptor)
    return ImplicitRelaxation(config)._identify_chains(topology)


class TestIdentifyChains:
    def test_1cwa_author_ids(self):
        assert _identified(CWA, "C", "A") == ("B", "A")

    def test_1cwa_topology_ids_still_work(self):
        assert _identified(CWA, "B", "A") == ("B", "A")

    def test_1cwa_auto_detection(self):
        assert _identified(CWA, None, None) == ("B", "A")

    def test_3p8f_author_ids(self):
        assert _identified(SFTI, "I", "A") == ("B", "A")

    def test_a_pdb_input_is_unchanged(self):
        assert _identified(P53, "B", "A") == ("B", "A")

    def test_an_unknown_id_is_handed_on_as_given(self):
        assert _identified(CWA, "Z", "A") == ("Z", "A")

    def test_letters_swapped_between_label_and_author_ids(self, tmp_path):
        from tests.test_fix_64_chain_ids import _write_cif

        path = _write_cif(
            tmp_path / "swapped.cif",
            [("GLY", "A", "B")] * 3 + [("GLY", "B", "A")] * 5 + [("HOH", "C", "A")],
        )
        assert _identified(path, "B", "A") == ("A", "B")


class TestQcNames:
    def test_the_snapshot_names_the_chains_by_author_id(self):
        topology, positions = _load(CWA)
        snapshot = AtomSnapshot.from_topology(topology, positions)
        assert {key[0] for key in snapshot.residue_keys} == {"A", "C"}
        # the 11 residues of the peptide, and the 4 waters, are in author chain C
        peptide = {key for key in snapshot.residue_keys if key[0] == "C"}
        assert len(peptide) == 11 + 4

    def test_a_topology_without_author_ids_keeps_its_own(self):
        topology, positions = _load(P53)
        snapshot = AtomSnapshot.from_topology(topology, positions)
        assert {key[0] for key in snapshot.residue_keys} == set(author_chain_ids(topology))


@pytest.fixture(scope="module")
def cwa_system():
    """The relaxer and its ``_setup_system`` result for the raw 1CWA (peptide: author ID C)."""
    if not CWA.exists():
        pytest.skip(f"bundled example not found: {CWA}")
    config = RelaxationConfig(peptide_chain_id="C", receptor_chain_id="A", small_molecules="auto")
    relaxer = ImplicitRelaxation(config)
    return relaxer, relaxer._setup_system(CWA)


@pytest.mark.integration
@requires_ommff
class TestSetupSystem:
    def test_the_peptide_is_kept_and_nothing_is_dropped(self, cwa_system):
        relaxer, (_, topology, _, bond_info) = cwa_system
        assert relaxer._chain_ids == ("B", "A")
        assert relaxer._dropped_protein_chains == []
        chains = {chain.id: sum(1 for _ in chain.residues()) for chain in topology.chains()}
        assert chains == {"A": 165, "B": 11}
        assert [b.cyclic_type for b in bond_info] == ["head_to_tail"]

    def test_the_returned_topology_carries_the_author_ids(self, cwa_system):
        _, (_, topology, _, _) = cwa_system
        assert author_chain_ids(topology) == ["A", "C"]


class TestPipelineHandsOverAuthorIds:
    def test_the_relaxer_and_the_hints_get_the_author_ids(self, tmp_path, monkeypatch):
        pytest.importorskip("biotite")
        from binding_metrics.cli.run import run_pipeline
        from binding_metrics.protocols import relaxation

        seen: dict = {}

        class RecordingRelaxer:
            def __init__(self, config):
                seen["config"] = config

            def run(self, input_path, output_dir, sample_id=None):
                return RelaxationResult(sample_id=sample_id, success=False, error_message="stub")

        monkeypatch.setattr(relaxation, "ImplicitRelaxation", RecordingRelaxer)
        _load(CWA)
        run_pipeline(CWA, tmp_path, skip_prep=True, md_duration_ps=0, metrics=frozenset())
        config = seen["config"]
        assert (config.peptide_chain_id, config.receptor_chain_id) == ("C", "A")
        assert [(h.cyclic_type, h.atom1_id[0]) for h in config.cyclic_bond_hints] == [
            ("head_to_tail", "B")
        ]


@pytest.fixture(scope="module")
def relaxed_raw_cwa(tmp_path_factory):
    """A short minimisation of the raw 1CWA with the peptide named by its author ID."""
    if not CWA.exists():
        pytest.skip(f"bundled example not found: {CWA}")
    config = RelaxationConfig(
        peptide_chain_id="C",
        receptor_chain_id="A",
        md_duration_ps=0.0,
        min_steps_initial=200,
        min_steps_restrained=100,
        min_steps_final=200,
        device="cuda",
        small_molecules="auto",
    )
    return ImplicitRelaxation(config).run(CWA, tmp_path_factory.mktemp("relax_raw"))


@requires_cuda
@requires_ommff
@pytest.mark.integration
class TestRelaxRawFile:
    def test_it_relaxes_the_peptide_and_names_its_chain_c(self, relaxed_raw_cwa):
        assert relaxed_raw_cwa.success, relaxed_raw_cwa.error_message
        assert relaxed_raw_cwa.dropped_protein_chains == []
        assert relaxed_raw_cwa.peptide_cyclic_bonds == [
            {"type": "head_to_tail", "atom1": "C:11:C", "atom2": "C:1:N"}
        ]

    def test_the_relaxed_file_has_the_peptide_in_author_chain_c(self, relaxed_raw_cwa):
        import gemmi

        model = gemmi.read_structure(relaxed_raw_cwa.minimized_structure_path)[0]
        assert sorted(chain.name for chain in model) == ["A", "C"]
        assert len(model["C"]) == 11
