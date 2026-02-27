"""Tests for interface analysis metrics (SASA, ΔG_int, H-bonds, salt bridges)."""

from pathlib import Path

import numpy as np
import pytest

try:
    import biotite
    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False

requires_biotite = pytest.mark.skipif(not HAS_BIOTITE, reason="biotite not installed")

# CIF fixture shared with other integration tests
EXAMPLE_CIF = Path("data/rank001_design_spec_457.cif")


class TestDetectInterfaceChains:
    """Tests for detect_interface_chains."""

    @requires_biotite
    def test_returns_tuple(self):
        from binding_metrics.metrics.interface import detect_interface_chains, load_biotite_structure

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        atoms = load_biotite_structure(EXAMPLE_CIF)
        pep, rec = detect_interface_chains(atoms)
        assert isinstance(pep, str)
        assert isinstance(rec, str)
        assert pep != rec

    @requires_biotite
    def test_explicit_design_chain(self):
        from binding_metrics.metrics.interface import detect_interface_chains, load_biotite_structure

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        atoms = load_biotite_structure(EXAMPLE_CIF)
        all_chains = list(np.unique(atoms.chain_id))
        if len(all_chains) < 2:
            pytest.skip("Need at least 2 chains")

        pep, rec = detect_interface_chains(atoms, design_chain=all_chains[0])
        assert pep == all_chains[0]
        assert rec != all_chains[0]

    @requires_biotite
    def test_empty_structure_returns_none(self):
        import biotite.structure as struc
        from binding_metrics.metrics.interface import detect_interface_chains

        empty = struc.AtomArray(0)
        pep, rec = detect_interface_chains(empty)
        assert pep is None
        assert rec is None


class TestComputeInterfaceMetrics:
    """Tests for compute_interface_metrics."""

    @requires_biotite
    @pytest.mark.integration
    def test_returns_expected_keys(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        result = compute_interface_metrics(EXAMPLE_CIF)

        expected_keys = {
            "peptide_chain", "receptor_chain",
            "delta_sasa", "sasa_peptide", "sasa_receptor", "sasa_complex",
            "delta_g_int", "delta_g_int_kJ",
            "polar_area", "apolar_area", "fraction_polar",
            "n_interface_residues_peptide", "n_interface_residues_receptor",
            "interface_residues_peptide", "interface_residues_receptor",
            "per_residue",
            "hbonds", "saltbridges",
        }
        assert set(result.keys()) == expected_keys

    @requires_biotite
    @pytest.mark.integration
    def test_sasa_consistency(self):
        """delta_sasa should approximately equal sasa_pep + sasa_rec - sasa_complex."""
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)

        expected_delta = r["sasa_peptide"] + r["sasa_receptor"] - r["sasa_complex"]
        # Allow small numerical difference due to per-atom clamping of negatives
        assert abs(r["delta_sasa"] - expected_delta) < 1.0

    @requires_biotite
    @pytest.mark.integration
    def test_delta_sasa_positive(self):
        """Buried SASA should be positive for a real complex."""
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        assert r["delta_sasa"] > 0

    @requires_biotite
    @pytest.mark.integration
    def test_delta_g_int_is_finite(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        assert np.isfinite(r["delta_g_int"])

    @requires_biotite
    @pytest.mark.integration
    def test_delta_g_int_kJ_consistent(self):
        """kJ/mol value should be kcal/mol × 4.184."""
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        assert abs(r["delta_g_int_kJ"] - r["delta_g_int"] * 4.184) < 1e-6

    @requires_biotite
    @pytest.mark.integration
    def test_polar_apolar_sum_leq_delta_sasa(self):
        """polar_area + apolar_area should not exceed delta_sasa (other elements excluded)."""
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        assert r["polar_area"] + r["apolar_area"] <= r["delta_sasa"] + 1e-3

    @requires_biotite
    @pytest.mark.integration
    def test_fraction_polar_in_range(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        assert 0.0 <= r["fraction_polar"] <= 1.0

    @requires_biotite
    @pytest.mark.integration
    def test_interface_residues_are_strings(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        for label in r["interface_residues_peptide"] + r["interface_residues_receptor"]:
            assert isinstance(label, str)
            # Format: RES:CHAIN:NUM
            assert label.count(":") == 2

    @requires_biotite
    @pytest.mark.integration
    def test_per_residue_fields(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        for entry in r["per_residue"]:
            assert "residue" in entry
            assert "buried_sasa" in entry
            assert "delta_g_res" in entry
            assert "polar_area" in entry
            assert "apolar_area" in entry
            assert entry["buried_sasa"] >= 0.5  # threshold default

    @requires_biotite
    @pytest.mark.integration
    def test_hbonds_and_saltbridges_are_ints(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        r = compute_interface_metrics(EXAMPLE_CIF)
        assert isinstance(r["hbonds"], int)
        assert isinstance(r["saltbridges"], int)
        assert r["hbonds"] >= 0
        assert r["saltbridges"] >= 0

    def test_missing_biotite_raises(self, tmp_path):
        """Should raise ImportError when biotite is not available."""
        import sys
        import unittest.mock as mock

        dummy_cif = tmp_path / "dummy.cif"
        dummy_cif.write_text("data_dummy\n")

        with mock.patch.dict(sys.modules, {"biotite": None, "biotite.structure": None}):
            with pytest.raises(ImportError, match="biotite"):
                from binding_metrics.metrics import interface as iface_mod
                import importlib
                importlib.reload(iface_mod)
                iface_mod.compute_interface_metrics(dummy_cif)


class TestLoadBiotiteStructure:
    """Tests for load_biotite_structure."""

    @requires_biotite
    @pytest.mark.integration
    def test_loads_atomarray(self):
        import biotite.structure as struc
        from binding_metrics.metrics.interface import load_biotite_structure

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        atoms = load_biotite_structure(EXAMPLE_CIF)
        assert isinstance(atoms, struc.AtomArray)
        assert len(atoms) > 0
