"""Tests for the protocol module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from binding_metrics.core.simulation import SimulationConfig
from binding_metrics.protocols.base import BaseProtocol, ProtocolResults
from binding_metrics.protocols.peptide import PeptideBindingProtocol


class TestProtocolResults:
    """Tests for ProtocolResults dataclass."""

    def test_default_values(self):
        """Should initialize with empty arrays and zero means."""
        results = ProtocolResults()

        assert len(results.sasa_buried) == 0
        assert results.sasa_buried_mean == 0.0
        assert results.sasa_buried_std == 0.0
        assert len(results.interface_contacts) == 0
        assert results.interface_contacts_mean == 0.0
        assert len(results.interaction_energy) == 0
        assert results.interaction_energy_mean == 0.0
        assert len(results.rmsd) == 0
        assert results.rmsd_mean == 0.0
        assert results.raw_data == {}

    def test_to_dict(self):
        """to_dict should return summary statistics."""
        results = ProtocolResults(
            sasa_buried=np.array([100.0, 110.0, 105.0]),
            sasa_buried_mean=105.0,
            sasa_buried_std=5.0,
            interface_contacts=np.array([10, 12, 11]),
            interface_contacts_mean=11.0,
            interaction_energy=np.array([-50.0, -55.0, -52.0]),
            interaction_energy_mean=-52.3,
            interaction_energy_std=2.5,
            rmsd=np.array([0.1, 0.15, 0.12]),
            rmsd_mean=0.123,
        )

        d = results.to_dict()

        assert "sasa_buried_mean" in d
        assert "interface_contacts_mean" in d
        assert "interaction_energy_mean" in d
        assert "rmsd_mean" in d
        assert "n_frames" in d
        assert d["n_frames"] == 3

    def test_summary_string(self):
        """summary should return formatted string."""
        results = ProtocolResults(
            sasa_buried=np.array([100.0]),
            sasa_buried_mean=100.0,
            sasa_buried_std=5.0,
            interface_contacts=np.array([10]),
            interface_contacts_mean=10.0,
            interaction_energy=np.array([-50.0]),
            interaction_energy_mean=-50.0,
            interaction_energy_std=2.0,
            rmsd=np.array([0.1]),
            rmsd_mean=0.1,
        )

        summary = results.summary()

        assert "Buried SASA" in summary
        assert "100.0" in summary
        assert "Interface contacts" in summary
        assert "10.0" in summary
        assert "Interaction energy" in summary
        assert "-50.0" in summary
        assert "RMSD" in summary
        assert "0.100" in summary

    def test_raw_data_storage(self):
        """raw_data should store arbitrary metadata."""
        results = ProtocolResults(
            raw_data={
                "ligand_chain": "B",
                "receptor_chains": ["A"],
                "custom_metric": 42.0,
            }
        )

        assert results.raw_data["ligand_chain"] == "B"
        assert results.raw_data["custom_metric"] == 42.0


class TestBaseProtocol:
    """Tests for BaseProtocol abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseProtocol is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseProtocol("test.pdb", "A", ["B"])

    def test_subclass_must_implement_run(self):
        """Subclass without run() should fail."""

        class IncompleteProtocol(BaseProtocol):
            def analyze(self, trajectory_path=None):
                return ProtocolResults()

        with pytest.raises(TypeError):
            IncompleteProtocol("test.pdb", "A", ["B"])

    def test_subclass_must_implement_analyze(self):
        """Subclass without analyze() should fail."""

        class IncompleteProtocol(BaseProtocol):
            def run(self, output_dir, **kwargs):
                return Path("traj.dcd")

        with pytest.raises(TypeError):
            IncompleteProtocol("test.pdb", "A", ["B"])

    def test_complete_subclass_works(self):
        """Complete subclass should instantiate."""

        class CompleteProtocol(BaseProtocol):
            def run(self, output_dir, **kwargs):
                return Path("traj.dcd")

            def analyze(self, trajectory_path=None):
                return ProtocolResults()

        protocol = CompleteProtocol("test.pdb", "A", ["B"])
        assert protocol.pdb_path == Path("test.pdb")
        assert protocol.ligand_chain == "A"
        assert protocol.receptor_chains == ["B"]


class TestPeptideBindingProtocol:
    """Tests for PeptideBindingProtocol."""

    def test_init_with_defaults(self, sample_pdb_path: Path):
        """Should initialize with default configuration."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        assert protocol.pdb_path == sample_pdb_path
        assert protocol.ligand_chain == "B"
        assert protocol.receptor_chains == ["A"]
        assert protocol.forcefield_name == "amber"
        assert protocol.simulation_config is not None

    def test_init_with_custom_config(self, sample_pdb_path: Path):
        """Should accept custom simulation config."""
        config = SimulationConfig(duration_ns=1.0, platform="CPU")
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
            simulation_config=config,
        )

        assert protocol.simulation_config.duration_ns == 1.0
        assert protocol.simulation_config.platform == "CPU"

    def test_init_with_charmm(self, sample_pdb_path: Path):
        """Should accept CHARMM force field."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
            forcefield="charmm",
        )

        assert protocol.forcefield_name == "charmm"

    def test_trajectory_path_initially_none(self, sample_pdb_path: Path):
        """trajectory_path should be None before run()."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        assert protocol.trajectory_path is None

    def test_results_initially_none(self, sample_pdb_path: Path):
        """results should be None before analyze()."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        assert protocol.results is None

    def test_analyze_without_run_raises(self, sample_pdb_path: Path):
        """analyze() without run() or trajectory_path should raise."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        with pytest.raises(RuntimeError, match="No trajectory"):
            protocol.analyze()

    def test_multiple_receptor_chains(self, sample_pdb_path: Path):
        """Should accept multiple receptor chains."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A", "C", "D"],
        )

        assert protocol.receptor_chains == ["A", "C", "D"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_creates_output(
        self, example_pdb_path: Path, example_pdb_chains: dict, output_dir: Path
    ):
        """run() should create trajectory and topology files."""
        pytest.importorskip("openmm")
        pytest.importorskip(
            "pdbfixer", reason="pdbfixer required to fix incomplete PDB structures"
        )

        config = SimulationConfig(
            duration_ns=0.0001,  # Very short for testing
            equilibration_ns=0.00001,
            save_interval_ps=0.01,
            platform="CPU",
        )

        protocol = PeptideBindingProtocol(
            pdb_path=example_pdb_path,
            ligand_chain=example_pdb_chains["ligand"],
            receptor_chains=example_pdb_chains["receptor"],
            simulation_config=config,
        )

        traj_path = protocol.run(output_dir)

        # Generic assertions - file existence only
        assert traj_path.exists()
        assert traj_path.suffix == ".dcd"
        assert (output_dir / "solvated.pdb").exists()
        assert (output_dir / "state.csv").exists()


class TestRunAndAnalyze:
    """Tests for the combined run_and_analyze workflow."""

    def test_run_and_analyze_returns_results(self, sample_pdb_path: Path):
        """run_and_analyze should return ProtocolResults."""

        class MockProtocol(BaseProtocol):
            def run(self, output_dir, **kwargs):
                self._trajectory_path = Path("mock.dcd")
                return self._trajectory_path

            def analyze(self, trajectory_path=None):
                self._results = ProtocolResults(
                    sasa_buried_mean=100.0,
                    interaction_energy_mean=-50.0,
                )
                return self._results

        protocol = MockProtocol(sample_pdb_path, "B", ["A"])
        results = protocol.run_and_analyze("/tmp/output")

        assert isinstance(results, ProtocolResults)
        assert results.sasa_buried_mean == 100.0
        assert results.interaction_energy_mean == -50.0
