"""Chain IDs in ``run_pipeline``: author, label and OpenMM names of one chain.

3P8F (trypsin with the bicyclic SFTI-1 inhibitor) has the peptide as author chain I and
label chain B, and the receptor as author chain A and label chain A; waters and a ligand
give the file more label IDs than author IDs, so OpenMM names the chains by label.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("biotite")

from binding_metrics.cli.run import _collect_failures, run_pipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BICYCLIC_CIF = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"


@pytest.fixture
def recorded_energy_chains(monkeypatch):
    """Replace the energy metric by a recorder of the chains it is called with."""
    calls: list = []

    def fake_energy(path, peptide_chain=None, receptor_chain=None, **kwargs):
        calls.append({"path": Path(path), "peptide": peptide_chain, "receptor": receptor_chain})
        return {"stub": True}

    monkeypatch.setattr("binding_metrics.metrics.energy.compute_interaction_energy", fake_energy)
    return calls


class TestSkippedPrepAndRelax:
    """The metrics read the raw input, whose biotite chain IDs are the author IDs."""

    def test_biotite_metrics_find_the_author_chains(self, tmp_path):
        results = run_pipeline(
            BICYCLIC_CIF,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"geometry", "interface", "electrostatics"}),
        )
        assert results["chains"]["peptide_chain"] == "I"
        assert results["chains"]["peptide_chain_label"] == "B"
        assert _collect_failures(results) == []
        assert results["interface"]["peptide_chain"] == "I"
        assert results["interface"]["receptor_chain"] == "A"

    def test_energy_reads_the_raw_file_through_openmm_names(self, tmp_path, recorded_energy_chains):
        run_pipeline(
            BICYCLIC_CIF,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"energy"}),
        )
        (call,) = recorded_energy_chains
        assert call["path"] == BICYCLIC_CIF
        assert (call["peptide"], call["receptor"]) == ("B", "A")


class TestPreppedFile:
    def test_energy_reads_the_prepped_file_through_its_own_names(
        self, tmp_path, recorded_energy_chains
    ):
        """Prep writes the author IDs into both the author and the label columns."""
        run_pipeline(BICYCLIC_CIF, tmp_path, skip_relax=True, metrics=frozenset({"energy"}))
        (call,) = recorded_energy_chains
        assert call["path"].name.endswith("_cleaned.cif")
        assert (call["peptide"], call["receptor"]) == ("I", "A")

    def test_cyclic_hints_name_the_chain_of_the_raw_topology(self, tmp_path, monkeypatch):
        """The hints are found on the raw file, so the chain is the raw file's OpenMM name (B).

        The prepped file calls the peptide I; looked up in the raw topology that name finds
        no chain, and the hints were silently dropped.
        """
        from binding_metrics.core import cyclic

        seen: list = []
        real = cyclic.detect_cyclization

        def recording(topology, positions, chain_id):
            hints = real(topology, positions, chain_id)
            seen.append((chain_id, len(hints)))
            return hints

        monkeypatch.setattr(cyclic, "detect_cyclization", recording)
        run_pipeline(BICYCLIC_CIF, tmp_path, skip_relax=True, metrics=frozenset())
        # Once inside prep and once for the hints; SFTI-1 has two closure bonds.
        assert seen == [("B", 2), ("B", 2)]


class TestRelaxedFile:
    def test_relaxed_file_from_a_raw_input_carries_the_author_ids(
        self, tmp_path, recorded_energy_chains
    ):
        """With prep skipped, the relaxer reads the raw file and writes author IDs back."""
        from binding_metrics.io.structures import load_structure, save_cif

        class WritingRelaxer:
            def run(self, input_path, output_dir, sample_id=None):
                from binding_metrics.protocols.relaxation import RelaxationResult

                topology, positions = load_structure(input_path)
                out = Path(output_dir) / "relaxed.cif"
                save_cif(topology, positions, out, source_cif_path=input_path)
                return RelaxationResult(
                    sample_id=sample_id, success=True, minimized_structure_path=str(out)
                )

        results = run_pipeline(
            BICYCLIC_CIF,
            tmp_path,
            skip_prep=True,
            relaxer=WritingRelaxer(),
            metrics=frozenset({"energy", "interface"}),
        )
        (call,) = recorded_energy_chains
        assert (call["peptide"], call["receptor"]) == ("I", "A")
        assert "error" not in results["interface"]
        assert results["interface"]["peptide_chain"] == "I"
