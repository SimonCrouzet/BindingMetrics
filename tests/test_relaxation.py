"""Tests for the implicit solvent MD relaxation protocol."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_cuda
from binding_metrics.protocols.relaxation import (
    ImplicitRelaxation,
    RelaxationConfig,
    RelaxationResult,
)

# Real CIF test data (available in the repo)
EXAMPLE_CIF = Path("data/rank001_design_spec_457.cif")


class TestRelaxationConfig:
    """Tests for RelaxationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RelaxationConfig()
        assert config.md_duration_ps == 200.0
        assert config.md_timestep_fs == 2.0
        assert config.md_temperature_k == 300.0
        assert config.solvent_model == "obc2"
        assert config.device == "cuda"
        assert config.peptide_chain_id is None
        assert config.receptor_chain_id is None
        assert config.custom_bond_handler is None

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            solvent_model="gbn2",
            peptide_chain_id="A",
        )
        assert config.md_duration_ps == 0.0
        assert config.device == "cpu"
        assert config.solvent_model == "gbn2"
        assert config.peptide_chain_id == "A"

    def test_custom_bond_handler_callable(self):
        """Should accept a callable for custom_bond_handler."""
        handler = lambda topo, pos, chain: (topo, pos, [])
        config = RelaxationConfig(custom_bond_handler=handler)
        assert callable(config.custom_bond_handler)


class TestRelaxationResult:
    """Tests for RelaxationResult dataclass."""

    def test_failed_result(self):
        """Should represent a failed run correctly."""
        result = RelaxationResult(sample_id="test", success=False, error_message="Boom")
        assert result.sample_id == "test"
        assert not result.success
        assert result.error_message == "Boom"
        assert result.potential_energy_minimized is None

    def test_to_dict_keys(self):
        """to_dict() should contain required keys."""
        result = RelaxationResult(sample_id="test", success=True)
        d = result.to_dict()
        assert "sample_id" in d
        assert "success" in d
        assert "potential_energy_minimized" in d
        assert "rmsd_md_final" in d
        assert "minimization_time_s" in d

    def test_to_dict_rmsf_json(self):
        """to_dict() should serialize per-residue RMSF as JSON string."""
        result = RelaxationResult(
            sample_id="test",
            success=True,
            peptide_rmsf_per_residue=[1.0, 2.0, 3.0],
        )
        d = result.to_dict()
        assert "peptide_rmsf_per_residue" in d
        parsed = json.loads(d["peptide_rmsf_per_residue"])
        assert parsed == [1.0, 2.0, 3.0]

    def test_to_dict_no_rmsf_key_when_none(self):
        """to_dict() should omit rmsf key when not set."""
        result = RelaxationResult(sample_id="test", success=True)
        d = result.to_dict()
        assert "peptide_rmsf_per_residue" not in d


class TestImplicitRelaxation:
    """Tests for ImplicitRelaxation class."""

    def test_init(self):
        """Should initialize with config."""
        config = RelaxationConfig()
        relaxer = ImplicitRelaxation(config)
        assert relaxer.config is config
        assert not relaxer._openmm_imported

    @requires_cuda
    @pytest.mark.integration
    def test_run_minimize_only_cif(self, tmp_path: Path):
        """Should successfully minimize a CIF structure (no MD)."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=10,
            min_steps_restrained=5,
            min_steps_final=10,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(EXAMPLE_CIF, tmp_path / "out")

        assert result.success, result.error_message
        assert result.potential_energy_minimized is not None
        assert result.minimized_structure_path is not None
        assert Path(result.minimized_structure_path).exists()
        assert result.md_final_structure_path is None

    @requires_cuda
    @pytest.mark.slow
    @pytest.mark.integration
    def test_run_with_md_cif(self, tmp_path: Path):
        """Should run a very short MD simulation on a CIF structure."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        config = RelaxationConfig(
            md_duration_ps=2.0,
            md_save_interval_ps=1.0,
            md_timestep_fs=1.0,  # 1 fs for stability after clash resolution
            min_steps_initial=200,
            min_steps_restrained=100,
            min_steps_final=200,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(EXAMPLE_CIF, tmp_path / "out")

        assert result.success, result.error_message
        assert result.md_final_structure_path is not None
        assert result.rmsd_md_final is not None
        assert result.rmsd_md_final >= 0.0

    @requires_cuda
    @pytest.mark.integration
    def test_run_missing_file_fails_gracefully(self, tmp_path: Path):
        """Should return failed result for missing input."""
        config = RelaxationConfig(md_duration_ps=0.0)
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(tmp_path / "missing.pdb", tmp_path / "out")
        assert not result.success
        assert result.error_message is not None

    @requires_cuda
    @pytest.mark.integration
    def test_sample_id_defaults_to_file_stem(self, tmp_path: Path):
        """sample_id should default to the input file stem."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=5, min_steps_restrained=5, min_steps_final=5,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(EXAMPLE_CIF, tmp_path / "out")
        assert result.sample_id == EXAMPLE_CIF.stem

    @pytest.mark.integration
    def test_cpu_platform_available(self, tmp_path: Path):
        """CPU platform should work as a fallback for non-GPU environments."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            min_steps_initial=5, min_steps_restrained=5, min_steps_final=5,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(EXAMPLE_CIF, tmp_path / "out")
        assert result.success, result.error_message

    @pytest.mark.integration
    def test_kabsch_rmsd_identical(self):
        """_compute_rmsd should return 0 for identical positions."""
        config = RelaxationConfig()
        relaxer = ImplicitRelaxation(config)

        from openmm import Vec3
        import openmm.unit as unit

        pos = [Vec3(float(i) * 0.1 + 0.1, 0.0, 0.0) for i in range(5)] * unit.nanometers
        rmsd = relaxer._compute_rmsd(pos, pos)
        assert abs(rmsd) < 1e-6

    @pytest.mark.integration
    def test_rmsf_zero_for_static_trajectory(self):
        """_compute_rmsf should return 0 for identical frames."""
        config = RelaxationConfig()
        relaxer = ImplicitRelaxation(config)

        from openmm import Vec3
        import openmm.unit as unit

        pos = [Vec3(float(i) * 0.1 + 0.1, 0.0, 0.0) for i in range(5)] * unit.nanometers
        trajectory = [pos, pos, pos]
        rmsf = relaxer._compute_rmsf(trajectory, list(range(5)))
        assert np.allclose(rmsf, 0.0, atol=1e-6)
