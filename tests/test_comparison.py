"""Tests for structure comparison utilities (RMSD between structures)."""

from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics.comparison import (
    _kabsch_rmsd,
    _matched_rmsd,
    compute_structure_rmsd,
)

try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

requires_gemmi = pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")

EXAMPLE_CIF = Path("data/rank001_design_spec_457.cif")
EXAMPLE_CIF2 = Path("data/rank002_design_spec_384.cif")


class TestKabschRmsd:
    """Tests for the Kabsch RMSD helper."""

    def test_identical_structures(self):
        """RMSD of identical structures should be 0."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert abs(_kabsch_rmsd(coords, coords)) < 1e-6

    def test_translated_structure(self):
        """Pure translation should give 0 RMSD after Kabsch alignment."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        translated = coords + np.array([5.0, 3.0, 1.0])
        assert abs(_kabsch_rmsd(coords, translated)) < 1e-6

    def test_nonzero_rmsd(self):
        """Perturbed coordinates should give nonzero RMSD."""
        rng = np.random.default_rng(42)
        coords = rng.random((10, 3))
        perturbed = coords + rng.random((10, 3)) * 0.5
        assert _kabsch_rmsd(coords, perturbed) > 0.0


class TestMatchedRmsd:
    """Tests for atom-matched RMSD computation."""

    def test_same_length_uses_kabsch(self):
        """Same-length arrays should use direct Kabsch and return 0 for identical coords."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        keys = [("A", 1, "CA"), ("A", 2, "CA")]
        result = _matched_rmsd(coords, keys, coords, keys)
        assert result is not None
        assert abs(result) < 1e-6

    def test_empty_coords_returns_none(self):
        """Empty coordinates should return None."""
        coords = np.zeros((0, 3))
        result = _matched_rmsd(coords, [], coords, [])
        assert result is None

    def test_different_lengths_no_common_atoms_returns_none(self):
        """Differing lengths with no common atom keys should return None."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        keys1 = [("A", 1, "CA"), ("A", 2, "CA")]
        coords2 = np.array([[0.0, 0.0, 0.0]])
        keys2 = [("B", 99, "CB")]  # no overlap with keys1
        result = _matched_rmsd(coords1, keys1, coords2, keys2)
        assert result is None

    def test_subset_matching(self):
        """Should match on common atoms when counts differ."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        keys1 = [("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C")]
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        keys2 = [("A", 1, "N"), ("A", 1, "CA")]
        result = _matched_rmsd(coords1, keys1, coords2, keys2)
        assert result is not None
        assert result >= 0.0


class TestComputeStructureRmsd:
    """Tests for compute_structure_rmsd function."""

    @requires_gemmi
    @pytest.mark.integration
    def test_same_structure_zero_rmsd(self):
        """Comparing a structure to itself should give ~0 RMSD."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        result = compute_structure_rmsd(EXAMPLE_CIF, EXAMPLE_CIF)

        assert result["rmsd"] is not None
        assert abs(result["rmsd"]) < 1e-4
        assert result["bb_rmsd"] is not None
        assert abs(result["bb_rmsd"]) < 1e-4

    @requires_gemmi
    @pytest.mark.integration
    def test_returns_expected_keys(self):
        """Should always return the four expected keys."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        result = compute_structure_rmsd(EXAMPLE_CIF, EXAMPLE_CIF)
        assert set(result.keys()) == {"rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"}

    @requires_gemmi
    @pytest.mark.integration
    def test_design_chain_rmsd_not_none(self):
        """Should compute design-chain RMSD when design_chain is given."""
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")
        result = compute_structure_rmsd(EXAMPLE_CIF, EXAMPLE_CIF, design_chain="A")
        assert result["rmsd_design"] is not None
        assert result["bb_rmsd_design"] is not None

    def test_missing_gemmi_raises(self, tmp_path: Path):
        """Should raise ImportError if gemmi is not installed."""
        import sys
        import unittest.mock as mock

        # Temporarily hide gemmi even if installed
        with mock.patch.dict(sys.modules, {"gemmi": None}):
            with pytest.raises(ImportError, match="gemmi"):
                # Create dummy CIF paths (they won't be opened before gemmi check)
                compute_structure_rmsd(tmp_path / "a.cif", tmp_path / "b.cif")
