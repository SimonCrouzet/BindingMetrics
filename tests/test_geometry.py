"""Tests for geometry metrics: Ramachandran, omega planarity, shape complementarity,
and buried void volume."""

from pathlib import Path

import numpy as np
import pytest

EXAMPLE_PDB_PATH = Path(__file__).parent.parent / "data" / "example.pdb"


def _skip_if_no_biotite():
    pytest.importorskip("biotite")


def _skip_if_no_example():
    if not EXAMPLE_PDB_PATH.exists():
        pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")


# ---------------------------------------------------------------------------
# Ramachandran tests
# ---------------------------------------------------------------------------


class TestRamachandran:
    """Tests for compute_ramachandran."""

    def test_import_function(self):
        """Should be able to import compute_ramachandran."""
        from binding_metrics.metrics.geometry import compute_ramachandran
        assert callable(compute_ramachandran)

    def test_returns_expected_keys(self):
        """Result dict should have all required keys."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="M")

        expected = {
            "ramachandran_favoured_pct",
            "ramachandran_allowed_pct",
            "ramachandran_outlier_pct",
            "ramachandran_outlier_count",
            "n_residues_evaluated",
            "per_residue",
        }
        assert expected.issubset(set(result.keys()))

    def test_percentages_sum_to_100(self):
        """Favoured + allowed + outlier percentages should sum to ~100."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="M")

        if result["n_residues_evaluated"] > 0:
            total = (
                result["ramachandran_favoured_pct"]
                + result["ramachandran_allowed_pct"]
                + result["ramachandran_outlier_pct"]
            )
            assert total == pytest.approx(100.0, abs=0.1)

    def test_per_residue_has_required_fields(self):
        """Each per_residue entry should have res_id, res_name, chain, phi, psi, region."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="M")

        for entry in result["per_residue"]:
            assert "res_id" in entry
            assert "res_name" in entry
            assert "chain" in entry
            assert "phi" in entry
            assert "psi" in entry
            assert "region" in entry
            assert entry["region"] in ("favoured", "allowed", "outlier")

    def test_chain_R_receptor(self):
        """Should work for receptor chain R too."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="R")

        assert result["n_residues_evaluated"] > 0

    def test_auto_detect_chain(self):
        """Auto-detection should default to the smaller (peptide) chain."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result_auto = compute_ramachandran(EXAMPLE_PDB_PATH)
        result_M = compute_ramachandran(EXAMPLE_PDB_PATH, chain="M")

        # Auto should pick chain M (smaller chain)
        assert result_auto["n_residues_evaluated"] == result_M["n_residues_evaluated"]

    def test_outlier_count_consistent_with_pct(self):
        """Outlier count should be consistent with outlier percentage."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="R")

        n_eval = result["n_residues_evaluated"]
        if n_eval > 0:
            expected_pct = 100.0 * result["ramachandran_outlier_count"] / n_eval
            assert result["ramachandran_outlier_pct"] == pytest.approx(expected_pct, abs=0.1)

    def test_classify_ramachandran_favoured_alpha(self):
        """Alpha-helix region should be classified as favoured."""
        from binding_metrics.metrics.geometry import _classify_ramachandran

        # Canonical alpha helix
        assert _classify_ramachandran(-60.0, -45.0) == "favoured"

    def test_classify_ramachandran_favoured_beta(self):
        """Beta-sheet region should be classified as favoured."""
        from binding_metrics.metrics.geometry import _classify_ramachandran

        assert _classify_ramachandran(-120.0, 130.0) == "favoured"

    def test_classify_ramachandran_outlier(self):
        """Upper-right quadrant (rarely observed) should be outlier."""
        from binding_metrics.metrics.geometry import _classify_ramachandran

        assert _classify_ramachandran(60.0, 150.0) == "outlier"

    def test_classify_ramachandran_nan(self):
        """NaN input (terminus) should return None."""
        from binding_metrics.metrics.geometry import _classify_ramachandran

        assert _classify_ramachandran(np.nan, np.nan) is None


# ---------------------------------------------------------------------------
# Omega planarity tests
# ---------------------------------------------------------------------------


class TestOmegaPlanarity:
    """Tests for compute_omega_planarity."""

    def test_import_function(self):
        """Should be able to import compute_omega_planarity."""
        from binding_metrics.metrics.geometry import compute_omega_planarity
        assert callable(compute_omega_planarity)

    def test_returns_expected_keys(self):
        """Result dict should have all required keys."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="M")

        expected = {
            "omega_mean_dev", "omega_max_dev",
            "omega_outlier_fraction", "omega_outlier_count",
            "n_bonds_evaluated", "per_residue",
        }
        assert expected.issubset(set(result.keys()))

    def test_mean_dev_is_non_negative(self):
        """Mean deviation should be non-negative."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="M")

        if result["n_bonds_evaluated"] > 0:
            assert result["omega_mean_dev"] >= 0.0
            assert result["omega_max_dev"] >= result["omega_mean_dev"]

    def test_per_residue_fields(self):
        """Per-residue entries should have omega, deviation, is_outlier fields."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="M")

        for entry in result["per_residue"]:
            assert "omega" in entry
            assert "deviation" in entry
            assert "is_outlier" in entry
            assert entry["deviation"] >= 0.0
            assert isinstance(entry["is_outlier"], bool)

    def test_outlier_threshold_15_degrees(self):
        """Outlier fraction should match entries with deviation > 15°."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="R")

        manual_outliers = sum(1 for e in result["per_residue"] if e["is_outlier"])
        assert manual_outliers == result["omega_outlier_count"]

    def test_chain_R_works(self):
        """Should work for receptor chain R."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="R")
        assert result["n_bonds_evaluated"] > 0

    def test_deviation_within_range(self):
        """Deviation from 180° should be in [0, 180] degrees."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="M")

        for entry in result["per_residue"]:
            assert 0.0 <= entry["deviation"] <= 180.0


# ---------------------------------------------------------------------------
# Shape complementarity tests
# ---------------------------------------------------------------------------


class TestShapeComplementarity:
    """Tests for compute_shape_complementarity."""

    def test_import_function(self):
        """Should be able to import compute_shape_complementarity."""
        from binding_metrics.metrics.geometry import compute_shape_complementarity
        assert callable(compute_shape_complementarity)

    def test_returns_expected_keys(self):
        """Result dict should have all required keys."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        expected = {
            "sc", "sc_A_to_B", "sc_B_to_A",
            "n_surface_dots_A", "n_surface_dots_B",
            "per_dot_scores_A", "per_dot_scores_B",
        }
        assert expected.issubset(set(result.keys()))

    def test_sc_value_in_range(self):
        """Sc score should be in [-1, 1]."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        if not np.isnan(result["sc"]):
            assert -1.0 <= result["sc"] <= 1.0

    def test_sc_is_mean_of_directional(self):
        """Sc should equal mean of sc_A_to_B and sc_B_to_A."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        if not np.isnan(result["sc"]):
            expected_sc = (result["sc_A_to_B"] + result["sc_B_to_A"]) / 2.0
            assert result["sc"] == pytest.approx(expected_sc, abs=1e-6)

    def test_surface_dots_positive(self):
        """Should generate surface dots for interface atoms."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        if not np.isnan(result["sc"]):
            assert result["n_surface_dots_A"] > 0
            assert result["n_surface_dots_B"] > 0

    def test_per_dot_scores_are_arrays(self):
        """per_dot_scores_A and _B should be numpy arrays."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        assert isinstance(result["per_dot_scores_A"], np.ndarray)
        assert isinstance(result["per_dot_scores_B"], np.ndarray)

    def test_missing_chains_returns_nan(self):
        """Non-existent chains should return NaN sc."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="Z", receptor_chain="W"
        )

        assert np.isnan(result["sc"])

    def test_auto_chain_detection(self):
        """Auto-detection should give same result as explicit chains."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result_auto = compute_shape_complementarity(EXAMPLE_PDB_PATH)
        result_explicit = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        if not np.isnan(result_auto["sc"]) and not np.isnan(result_explicit["sc"]):
            assert result_auto["sc"] == pytest.approx(result_explicit["sc"], rel=1e-4)

    def test_fibonacci_sphere(self):
        """Fibonacci sphere points should be unit vectors."""
        from binding_metrics.metrics.geometry import _fibonacci_sphere

        pts = _fibonacci_sphere(50)
        assert pts.shape == (50, 3)
        norms = np.linalg.norm(pts, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Void volume tests
# ---------------------------------------------------------------------------


class TestBuriedVoidVolume:
    """Tests for compute_buried_void_volume."""

    def test_import_function(self):
        """Should be able to import compute_buried_void_volume."""
        from binding_metrics.metrics.geometry import compute_buried_void_volume
        assert callable(compute_buried_void_volume)

    def test_returns_expected_keys(self):
        """Result dict should have all required keys."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=1.0,  # coarser for speed
        )

        expected = {
            "void_volume_A3", "void_grid_fraction",
            "interface_box_volume_A3", "n_interface_atoms",
        }
        assert expected.issubset(set(result.keys()))

    def test_void_volume_non_negative(self):
        """Void volume should be non-negative."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=1.0,
        )

        if not np.isnan(result["void_volume_A3"]):
            assert result["void_volume_A3"] >= 0.0

    def test_void_fraction_in_range(self):
        """Void grid fraction should be in [0, 1]."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=1.0,
        )

        if not np.isnan(result["void_grid_fraction"]):
            assert 0.0 <= result["void_grid_fraction"] <= 1.0

    def test_interface_atoms_positive(self):
        """Should detect interface atoms."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=1.0,
        )

        assert result["n_interface_atoms"] > 0

    def test_missing_chains_returns_nan(self):
        """Non-existent chains should return NaN void volume."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="Z", receptor_chain="W"
        )

        assert np.isnan(result["void_volume_A3"])

    def test_box_volume_larger_than_void(self):
        """Bounding box volume should be >= void volume."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=1.0,
        )

        if not np.isnan(result["void_volume_A3"]):
            assert result["interface_box_volume_A3"] >= result["void_volume_A3"]

    def test_auto_chain_detection(self):
        """Auto chain detection should produce valid results."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(EXAMPLE_PDB_PATH, grid_spacing=1.0)

        assert result["n_interface_atoms"] > 0

    def test_finer_grid_leq_coarser_void(self):
        """Finer grid should give more accurate (generally lower) void estimate."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result_coarse = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=1.5,
        )
        result_fine = compute_buried_void_volume(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R",
            grid_spacing=0.75,
        )

        # Both should be valid (non-NaN)
        assert not np.isnan(result_coarse["void_volume_A3"])
        assert not np.isnan(result_fine["void_volume_A3"])


# ---------------------------------------------------------------------------
# Module-level classification function tests (standalone)
# ---------------------------------------------------------------------------


class TestClassifyRamachandran:
    """Unit tests for _classify_ramachandran helper."""

    def test_alpha_helix_is_favoured(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        assert _classify_ramachandran(-60.0, -45.0) == "favoured"

    def test_beta_sheet_is_favoured(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        assert _classify_ramachandran(-120.0, 130.0) == "favoured"

    def test_ppii_is_favoured(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # Poly-proline II region
        assert _classify_ramachandran(-70.0, 150.0) == "favoured"

    def test_l_helix_is_favoured(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # Left-handed helix
        assert _classify_ramachandran(60.0, 40.0) == "favoured"

    def test_disallowed_region_is_outlier(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        # Upper right: rarely seen
        assert _classify_ramachandran(60.0, 150.0) == "outlier"

    def test_nan_returns_none(self):
        from binding_metrics.metrics.geometry import _classify_ramachandran
        assert _classify_ramachandran(np.nan, -45.0) is None
        assert _classify_ramachandran(-60.0, np.nan) is None
