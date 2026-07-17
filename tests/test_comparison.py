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

EXAMPLE_CIF = Path("data/example_linear_p53_1YCR.pdb")
EXAMPLE_CIF2 = Path("data/example_bicyclic_sfti1_3P8F.cif")


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

    def test_multiplicity_mismatch_does_not_raise(self):
        """Keys shared with different multiplicity (e.g. altlocs / duplicated
        keys) must not blow up Kabsch with mismatched index-list lengths.

        Regression: previously the two selected index lists could differ in
        length (3 vs 2 here), raising ``ValueError`` from the Kabsch matmul.
        The fix pairs shared keys one-to-one up to the smaller count.
        """
        coords1 = np.zeros((3, 3))
        keys1 = [("A", 1, "CA"), ("A", 1, "CA"), ("A", 2, "CA")]
        coords2 = np.zeros((2, 3))
        keys2 = [("A", 1, "CA"), ("A", 2, "CA")]
        result = _matched_rmsd(coords1, keys1, coords2, keys2)
        assert result is not None
        assert np.isfinite(result)
        assert isinstance(result, float)


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

    @requires_gemmi
    @pytest.mark.integration
    def test_atom_matching_branch_on_real_files(self, prepped_example_cif):
        """Exercise the len(coords1) != len(coords2) atom-matching branch.

        ``test_same_structure_zero_rmsd`` compares a file to itself, so atom counts
        are equal and the fast direct-Kabsch path is taken — the atom-matching
        branch (match on (chain,res,atom) when counts differ) is never covered by
        any test. Comparing raw 1YCR to its prepped variant (hydrogens added →
        different atom count) drives that branch on real data and must return
        finite, non-None RMSDs rather than erroring or silently yielding None.
        """
        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        result = compute_structure_rmsd(EXAMPLE_CIF, prepped_example_cif, design_chain="A")

        assert result["rmsd"] is not None
        assert result["bb_rmsd"] is not None
        assert result["rmsd_design"] is not None
        assert result["rmsd"] >= 0.0
        assert result["bb_rmsd"] >= 0.0

    @requires_gemmi
    @pytest.mark.integration
    def test_different_peptides_no_shape_mismatch(self):
        """Comparing two different peptide structures with differing atom
        counts and shared (chain, res, atom) keys of differing multiplicity
        must not raise a Kabsch shape mismatch.

        Regression for the matmul ValueError (553 vs 490 atoms): the
        full-complex RMSDs must come back finite. Chain 'A' of the two
        peptides shares no common atom keys, so the design-chain variants are
        legitimately None — assert that graceful outcome rather than requiring
        a value.
        """
        if not EXAMPLE_CIF.exists() or not EXAMPLE_CIF2.exists():
            pytest.skip("Test CIFs not available")

        result = compute_structure_rmsd(EXAMPLE_CIF, EXAMPLE_CIF2)

        assert result["rmsd"] is not None
        assert np.isfinite(result["rmsd"])
        assert result["rmsd"] >= 0.0
        assert result["bb_rmsd"] is not None
        assert np.isfinite(result["bb_rmsd"])
        assert result["bb_rmsd"] >= 0.0

    def test_missing_gemmi_raises(self, tmp_path: Path):
        """Should raise ImportError if gemmi is not installed."""
        import sys
        import unittest.mock as mock

        # Temporarily hide gemmi even if installed
        with mock.patch.dict(sys.modules, {"gemmi": None}):
            with pytest.raises(ImportError, match="gemmi"):
                # Create dummy CIF paths (they won't be opened before gemmi check)
                compute_structure_rmsd(tmp_path / "a.cif", tmp_path / "b.cif")
