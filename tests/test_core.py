"""Tests for the core simulation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from binding_metrics.core.forcefields import (
    AMBER_CONFIG,
    CHARMM_CONFIG,
    FORCEFIELD_CONFIGS,
    ForceFieldConfig,
    get_forcefield,
    get_forcefield_config,
)
from binding_metrics.core.simulation import MDSimulation, SimulationConfig


class TestForceFieldConfig:
    """Tests for ForceFieldConfig dataclass."""

    def test_amber_config_exists(self):
        """AMBER configuration should be predefined."""
        assert AMBER_CONFIG is not None
        assert AMBER_CONFIG.name == "amber"
        assert "amber" in AMBER_CONFIG.protein_ff.lower()

    def test_charmm_config_exists(self):
        """CHARMM configuration should be predefined."""
        assert CHARMM_CONFIG is not None
        assert CHARMM_CONFIG.name == "charmm"
        assert "charmm" in CHARMM_CONFIG.protein_ff.lower()

    def test_forcefield_config_is_frozen(self):
        """ForceFieldConfig should be immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            AMBER_CONFIG.name = "modified"

    def test_all_configs_have_required_fields(self):
        """All configs should have name, protein_ff, water_model, description."""
        for name, config in FORCEFIELD_CONFIGS.items():
            assert config.name == name
            assert config.protein_ff
            assert config.water_model
            assert config.description


class TestGetForcefield:
    """Tests for get_forcefield function."""

    @pytest.mark.integration
    def test_get_amber_forcefield(self):
        """Should return an OpenMM ForceField for AMBER."""
        ff = get_forcefield("amber")
        assert ff is not None

    @pytest.mark.integration
    def test_get_charmm_forcefield(self):
        """Should return an OpenMM ForceField for CHARMM."""
        ff = get_forcefield("charmm")
        assert ff is not None

    def test_invalid_forcefield_raises(self):
        """Should raise ValueError for unknown force field."""
        with pytest.raises(ValueError, match="Unknown force field"):
            get_forcefield("invalid_ff")

    def test_get_forcefield_config_amber(self):
        """Should return AMBER config."""
        config = get_forcefield_config("amber")
        assert config == AMBER_CONFIG

    def test_get_forcefield_config_charmm(self):
        """Should return CHARMM config."""
        config = get_forcefield_config("charmm")
        assert config == CHARMM_CONFIG


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_values(self):
        """Default configuration should have sensible values."""
        config = SimulationConfig()

        assert config.temperature == 300.0
        assert config.pressure == 1.0
        assert config.timestep == 2.0
        assert config.duration_ns == 10.0
        assert config.equilibration_ns == 0.1
        assert config.save_interval_ps == 10.0
        assert config.friction == 1.0
        assert config.nonbonded_cutoff == 1.0
        assert config.constraints == "hbonds"
        assert config.platform == "auto"

    def test_custom_values(self):
        """Should accept custom configuration values."""
        config = SimulationConfig(
            temperature=310.0,
            pressure=None,  # NVT
            duration_ns=1.0,
            platform="CUDA",
        )

        assert config.temperature == 310.0
        assert config.pressure is None
        assert config.duration_ns == 1.0
        assert config.platform == "CUDA"

    def test_nvt_ensemble(self):
        """Setting pressure=None should configure NVT ensemble."""
        config = SimulationConfig(pressure=None)
        assert config.pressure is None

    def test_short_simulation_config(self):
        """Quick evaluation config for testing."""
        config = SimulationConfig(
            duration_ns=0.001,  # 1 ps
            equilibration_ns=0.0001,
            save_interval_ps=0.1,
        )
        assert config.duration_ns == 0.001


class TestMDSimulation:
    """Tests for MDSimulation class."""

    def test_init_with_default_config(self):
        """Should initialize with default config if none provided."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()

        sim = MDSimulation(mock_modeller, mock_forcefield)

        assert sim.config is not None
        assert isinstance(sim.config, SimulationConfig)
        assert sim.modeller is mock_modeller
        assert sim.forcefield is mock_forcefield
        assert sim.simulation is None

    def test_init_with_custom_config(self):
        """Should use provided config."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        config = SimulationConfig(duration_ns=5.0)

        sim = MDSimulation(mock_modeller, mock_forcefield, config)

        assert sim.config.duration_ns == 5.0

    def test_minimize_before_setup_raises(self):
        """Calling minimize before setup should raise RuntimeError."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        sim = MDSimulation(mock_modeller, mock_forcefield)

        with pytest.raises(RuntimeError, match="Call setup"):
            sim.minimize()

    def test_equilibrate_before_setup_raises(self):
        """Calling equilibrate before setup should raise RuntimeError."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        sim = MDSimulation(mock_modeller, mock_forcefield)

        with pytest.raises(RuntimeError, match="Call setup"):
            sim.equilibrate()

    def test_run_before_setup_raises(self):
        """Calling run before setup should raise RuntimeError."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        sim = MDSimulation(mock_modeller, mock_forcefield)

        with pytest.raises(RuntimeError, match="Call setup"):
            sim.run("/tmp/output")

    def test_get_positions_before_setup_raises(self):
        """Calling get_positions before setup should raise RuntimeError."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        sim = MDSimulation(mock_modeller, mock_forcefield)

        with pytest.raises(RuntimeError, match="Call setup"):
            sim.get_positions()

    def test_get_potential_energy_before_setup_raises(self):
        """Calling get_potential_energy before setup should raise RuntimeError."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        sim = MDSimulation(mock_modeller, mock_forcefield)

        with pytest.raises(RuntimeError, match="Call setup"):
            sim.get_potential_energy()

    def test_system_property_initially_none(self):
        """System property should be None before setup."""
        mock_modeller = MagicMock()
        mock_forcefield = MagicMock()
        sim = MDSimulation(mock_modeller, mock_forcefield)

        assert sim.system is None
