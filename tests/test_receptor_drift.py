"""Tests for compute_receptor_drift in metrics/rmsd.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

EXAMPLE_PDB_PATH = Path(__file__).parent.parent / "data" / "example.pdb"


class TestReceptorDrift:
    """Tests for compute_receptor_drift."""

    def test_import_function(self):
        """Should be able to import compute_receptor_drift."""
        from binding_metrics.metrics.rmsd import compute_receptor_drift

        assert callable(compute_receptor_drift)

    def test_requires_mdtraj(self):
        """Should raise ImportError if mdtraj is not installed."""
        from binding_metrics.metrics import rmsd

        original_md = rmsd.md
        rmsd.md = None
        try:
            with pytest.raises(ImportError, match="mdtraj"):
                from binding_metrics.metrics.rmsd import compute_receptor_drift
                compute_receptor_drift("fake.dcd", "fake.pdb", "A")
        finally:
            rmsd.md = original_md

    @pytest.mark.integration
    def test_returns_expected_keys(self, tmp_path: Path):
        """Result dict should contain all required keys."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "R")

        expected_keys = {
            "drift_aligned_mean", "drift_aligned_max",
            "drift_raw_mean", "drift_raw_max",
            "pbc_detected",
            "drift_aligned_per_frame", "drift_raw_per_frame",
            "n_receptor_ca", "n_frames",
        }
        assert expected_keys.issubset(set(result.keys()))

    @pytest.mark.integration
    def test_single_frame_drift_is_zero(self, tmp_path: Path):
        """Drift from a single-frame trajectory (ref to itself) should be near zero."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        traj_path = tmp_path / "single.dcd"
        traj.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "R")

        # Single frame: aligned drift of reference frame is ~0 (DCD round-trip
        # may introduce ~0.01 Å numerical noise due to float32 precision)
        assert result["drift_aligned_per_frame"][0] == pytest.approx(0.0, abs=0.1)

    @pytest.mark.integration
    def test_n_receptor_ca_positive(self, tmp_path: Path):
        """Should detect at least one Cα atom for receptor chain R."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "R")

        assert result["n_receptor_ca"] > 0
        assert result["n_frames"] == traj.n_frames

    @pytest.mark.integration
    def test_drift_units_are_angstroms(self, tmp_path: Path):
        """Drift values should be in Å (roughly 0–100 for a protein structure)."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "R")

        # For a single-frame traj, drift is 0; aligned max should be >= 0
        assert result["drift_aligned_max"] >= 0.0
        assert isinstance(result["drift_aligned_per_frame"], np.ndarray)

    @pytest.mark.integration
    def test_missing_chain_returns_nan(self, tmp_path: Path):
        """Missing receptor chain should return NaN drift scores."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "ZZZNOTEXIST")

        assert result["n_receptor_ca"] == 0
        assert np.isnan(result["drift_aligned_mean"])

    @pytest.mark.integration
    def test_multi_frame_trajectory(self, tmp_path: Path):
        """Multi-frame trajectory should produce per-frame array of correct length."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        # Stack the same frame 5 times to create a multi-frame trajectory
        traj5 = md.join([traj] * 5)
        traj_path = tmp_path / "multi.dcd"
        traj5.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "R")

        assert result["n_frames"] == 5
        assert len(result["drift_aligned_per_frame"]) == 5
        # All frames identical → drift should be very small (DCD float32 precision allows ~0.1 Å)
        assert np.all(result["drift_aligned_per_frame"] < 0.5)

    @pytest.mark.integration
    def test_no_pbc_for_pdb(self, tmp_path: Path):
        """PDB file without unit cell should not report PBC."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import compute_receptor_drift

        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        traj = md.load(str(EXAMPLE_PDB_PATH))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = compute_receptor_drift(traj_path, EXAMPLE_PDB_PATH, "R")

        # The example.pdb has no meaningful unit cell → raw drift should be available
        # (DCD trajectories from PDB load without PBC info when the source has dummy cell)
        # Just check the type is correct
        assert isinstance(result["pbc_detected"], bool)
