"""Tests for package-level imports and exports."""

import pytest


class TestPackageImports:
    """Test that all public API is importable."""

    def test_import_main_package(self):
        """Should be able to import binding_metrics."""
        import binding_metrics

        assert binding_metrics is not None

    def test_version_exists(self):
        """Package should have __version__."""
        import binding_metrics

        assert hasattr(binding_metrics, "__version__")
        assert binding_metrics.__version__ == "0.1.0"

    def test_import_simulation_config(self):
        """SimulationConfig should be importable from main package."""
        from binding_metrics import SimulationConfig

        config = SimulationConfig()
        assert config.temperature == 300.0

    def test_import_md_simulation(self):
        """MDSimulation should be importable from main package."""
        from binding_metrics import MDSimulation

        assert MDSimulation is not None

    def test_import_run_simulation(self):
        """run_simulation should be importable from main package."""
        from binding_metrics import run_simulation

        assert callable(run_simulation)

    def test_import_forcefield_config(self):
        """ForceFieldConfig should be importable from main package."""
        from binding_metrics import ForceFieldConfig

        assert ForceFieldConfig is not None

    def test_import_get_forcefield(self):
        """get_forcefield should be importable from main package."""
        from binding_metrics import get_forcefield

        assert callable(get_forcefield)

    def test_import_prepare_system(self):
        """prepare_system should be importable from main package."""
        from binding_metrics import prepare_system

        assert callable(prepare_system)

    def test_import_protocol_results(self):
        """ProtocolResults should be importable from main package."""
        from binding_metrics import ProtocolResults

        results = ProtocolResults()
        assert results.sasa_buried_mean == 0.0

    def test_import_peptide_binding_protocol(self):
        """PeptideBindingProtocol should be importable from main package."""
        from binding_metrics import PeptideBindingProtocol

        assert PeptideBindingProtocol is not None

    def test_all_exports(self):
        """__all__ should contain expected exports."""
        import binding_metrics

        expected = {
            "ForceFieldConfig",
            "get_forcefield",
            "MDSimulation",
            "SimulationConfig",
            "run_simulation",
            "prepare_system",
            "ProtocolResults",
            "PeptideBindingProtocol",
        }

        assert set(binding_metrics.__all__) == expected


class TestSubmoduleImports:
    """Test that submodules are importable."""

    def test_import_core_module(self):
        """Core module should be importable."""
        from binding_metrics import core

        assert core is not None

    def test_import_protocols_module(self):
        """Protocols module should be importable."""
        from binding_metrics import protocols

        assert protocols is not None

    def test_import_metrics_module(self):
        """Metrics module should be importable."""
        from binding_metrics import metrics

        assert metrics is not None

    def test_import_io_module(self):
        """IO module should be importable."""
        from binding_metrics import io

        assert io is not None

    def test_import_sasa_function(self):
        """SASA function should be importable from metrics."""
        from binding_metrics.metrics import calculate_buried_sasa

        assert callable(calculate_buried_sasa)

    def test_import_contacts_function(self):
        """Contacts function should be importable from metrics."""
        from binding_metrics.metrics import calculate_contacts

        assert callable(calculate_contacts)

    def test_import_energy_function(self):
        """Energy function should be importable from metrics."""
        from binding_metrics.metrics import calculate_interaction_energy

        assert callable(calculate_interaction_energy)

    def test_import_rmsd_function(self):
        """RMSD function should be importable from metrics."""
        from binding_metrics.metrics import calculate_rmsd

        assert callable(calculate_rmsd)

    def test_import_load_complex(self):
        """load_complex should be importable from io."""
        from binding_metrics.io import load_complex

        assert callable(load_complex)

    def test_import_get_chain_atom_indices(self):
        """get_chain_atom_indices should be importable from io."""
        from binding_metrics.io import get_chain_atom_indices

        assert callable(get_chain_atom_indices)


class TestTypicalUsage:
    """Test typical usage patterns work."""

    def test_create_config_and_protocol(self, sample_pdb_path):
        """Typical workflow should not raise on setup."""
        from binding_metrics import PeptideBindingProtocol, SimulationConfig

        config = SimulationConfig(
            duration_ns=1.0,
            temperature=300.0,
            platform="CPU",
        )

        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
            forcefield="amber",
            simulation_config=config,
        )

        assert protocol is not None
        assert protocol.simulation_config.duration_ns == 1.0

    def test_protocol_results_workflow(self):
        """ProtocolResults should support typical analysis workflow."""
        import numpy as np

        from binding_metrics import ProtocolResults

        # Simulate analysis results
        results = ProtocolResults(
            sasa_buried=np.array([95.0, 100.0, 105.0, 98.0, 102.0]),
            sasa_buried_mean=100.0,
            sasa_buried_std=3.5,
            interface_contacts=np.array([15, 16, 14, 15, 16]),
            interface_contacts_mean=15.2,
            interaction_energy=np.array([-45.0, -50.0, -48.0, -52.0, -47.0]),
            interaction_energy_mean=-48.4,
            interaction_energy_std=2.5,
            rmsd=np.array([0.05, 0.08, 0.10, 0.09, 0.11]),
            rmsd_mean=0.086,
        )

        # Get summary
        summary = results.summary()
        assert "Buried SASA" in summary
        assert "100.0" in summary

        # Get dict for serialization
        d = results.to_dict()
        assert d["sasa_buried_mean"] == 100.0
        assert d["n_frames"] == 5
