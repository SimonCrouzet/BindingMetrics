"""Tests for the subsystem interaction energy computation."""

from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics.energy import (
    _extract_chain,
    compute_interaction_energy,
)

# Real CIF test data
EXAMPLE_CIF = Path("data/rank001_design_spec_457.cif")


class TestExtractChain:
    """Tests for the _extract_chain helper."""

    @pytest.mark.integration
    def test_extracts_correct_atoms(self, sample_pdb_path: Path):
        """Should extract only atoms belonging to the specified chain."""
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(sample_pdb_path)
        pep_topo, pep_pos = _extract_chain(topology, positions, "B")

        for atom in pep_topo.atoms():
            assert atom.residue.chain.id == "B"

    @pytest.mark.integration
    def test_extracted_positions_shape(self, sample_pdb_path: Path):
        """Extracted positions should match atom count in new topology."""
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(sample_pdb_path)
        pep_topo, pep_pos = _extract_chain(topology, positions, "B")

        n_atoms = pep_topo.getNumAtoms()
        assert len(pep_pos) == n_atoms


class TestComputeInteractionEnergy:
    """Tests for compute_interaction_energy function."""

    @pytest.mark.integration
    def test_returns_dict_with_base_keys(self):
        """Should return a dict with the base (non-mode) keys."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(EXAMPLE_CIF, device="cpu", modes=("raw",))

        base_keys = {"sample_id", "success", "error_message", "num_contacts", "num_close_contacts"}
        assert base_keys.issubset(result.keys())

    @pytest.mark.integration
    def test_raw_mode_keys(self):
        """raw mode should produce raw_* keys."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(EXAMPLE_CIF, device="cpu", modes=("raw",))

        raw_keys = {"raw_interaction_energy", "raw_e_complex", "raw_e_peptide", "raw_e_receptor"}
        assert raw_keys.issubset(result.keys())

    @pytest.mark.integration
    def test_relaxed_mode_keys(self):
        """relaxed mode should produce relaxed_* keys."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("relaxed",),
            relaxed_min_steps_restrained=20,
            relaxed_min_steps_full=50,
        )

        relaxed_keys = {
            "relaxed_interaction_energy", "relaxed_e_complex",
            "relaxed_e_peptide", "relaxed_e_receptor",
        }
        assert relaxed_keys.issubset(result.keys())

    @pytest.mark.integration
    def test_relaxed_succeeds_on_cif_structure(self):
        """relaxed mode should successfully compute energies."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("relaxed",),
            relaxed_min_steps_restrained=20,
            relaxed_min_steps_full=50,
        )
        assert result["success"], result.get("error_message")
        assert result["relaxed_interaction_energy"] is not None
        assert np.isfinite(result["relaxed_interaction_energy"])

    @pytest.mark.integration
    def test_no_clash_keys_for_relaxed_mode(self):
        """relaxed mode should NOT produce num_close_contacts per mode (raw-only metric)."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("relaxed",),
            relaxed_min_steps_restrained=20,
            relaxed_min_steps_full=50,
        )
        assert "relaxed_num_close_contacts" not in result
        assert "relaxed_num_contacts" not in result

    @pytest.mark.integration
    def test_contact_counts_are_raw_only(self):
        """num_contacts and num_close_contacts reflect raw input structure."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("relaxed",),
            relaxed_min_steps_restrained=20,
            relaxed_min_steps_full=50,
        )
        assert result["num_contacts"] is not None
        assert result["num_contacts"] >= 0
        assert result["num_close_contacts"] is not None
        assert result["num_close_contacts"] >= 0
        assert result["num_close_contacts"] <= result["num_contacts"]

    @pytest.mark.integration
    def test_sample_id_from_file_stem(self):
        """sample_id should default to the input file stem."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("relaxed",),
            relaxed_min_steps_restrained=10,
            relaxed_min_steps_full=20,
        )
        assert result["sample_id"] == EXAMPLE_CIF.stem

    @pytest.mark.integration
    def test_custom_sample_id(self):
        """Should use custom sample_id when provided."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("relaxed",),
            relaxed_min_steps_restrained=10,
            relaxed_min_steps_full=20,
            sample_id="my_sample",
        )
        assert result["sample_id"] == "my_sample"

    @pytest.mark.integration
    def test_missing_file_fails_gracefully(self, tmp_path: Path):
        """Should return failed result for missing file."""
        result = compute_interaction_energy(tmp_path / "missing.pdb", device="cpu")
        assert not result["success"]
        assert result["error_message"] is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_after_md_mode(self):
        """after_md mode should run MD and return finite energies."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_interaction_energy(
            EXAMPLE_CIF, device="cpu",
            modes=("after_md",),
            relaxed_min_steps_restrained=50,
            relaxed_min_steps_full=100,
            after_md_duration_ps=1.0,
            after_md_timestep_fs=1.0,
        )
        assert result["success"], result.get("error_message")
        ie = result["after_md_interaction_energy"]
        assert ie is not None
        assert np.isfinite(ie)
