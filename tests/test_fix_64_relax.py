"""The relaxation reads its chain options as author IDs (#64).

1CWA: the peptide is author chain C and chain B of the OpenMM topology, whose chain C holds
140 waters. Before the fix, ``RelaxationConfig(peptide_chain_id="C")`` on the raw file took the
waters for the peptide, dropped the real peptide as a third protein chain and finished with
``success`` true.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")


from binding_metrics.io.structures import load_structure  # noqa: E402
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
