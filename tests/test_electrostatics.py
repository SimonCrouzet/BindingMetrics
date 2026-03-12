"""Tests for compute_coulomb_cross_chain in metrics/electrostatics.py."""

from pathlib import Path

import numpy as np
import pytest

EXAMPLE_PDB_PATH = Path(__file__).parent.parent / "data" / "example.pdb"

requires_biotite = pytest.mark.skipif(
    not pytest.importorskip("biotite", reason="biotite not installed"),
    reason="biotite not installed",
)


def _skip_if_no_biotite():
    pytest.importorskip("biotite")


class TestCoulombCrossChain:
    """Tests for compute_coulomb_cross_chain."""

    def test_import_function(self):
        """Should be able to import compute_coulomb_cross_chain."""
        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain
        assert callable(compute_coulomb_cross_chain)

    def test_requires_biotite_import(self):
        """Module should raise ImportError if biotite is not available."""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"biotite": None, "biotite.structure": None,
                                       "biotite.structure.io": None,
                                       "biotite.structure.io.pdbx": None,
                                       "biotite.structure.io.pdb": None}):
            from binding_metrics.metrics.electrostatics import _import_biotite
            with pytest.raises(ImportError, match="biotite"):
                _import_biotite()

    def test_with_example_pdb(self):
        """Should compute Coulomb energy for example.pdb chains M and R."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH,
            peptide_chain="M",
            receptor_chain="R",
        )

        expected_keys = {
            "coulomb_energy_kJ", "coulomb_energy_kcal",
            "n_charged_pairs", "n_attractive", "n_repulsive",
            "charged_atoms_peptide", "charged_atoms_receptor",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_energy_is_finite(self):
        """Coulomb energy should be a finite float for valid input."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH,
            peptide_chain="M",
            receptor_chain="R",
        )

        assert np.isfinite(result["coulomb_energy_kJ"])
        assert np.isfinite(result["coulomb_energy_kcal"])

    def test_energy_kcal_vs_kj_conversion(self):
        """kcal/mol and kJ/mol values should have correct 4.184 ratio."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH,
            peptide_chain="M",
            receptor_chain="R",
        )

        kJ = result["coulomb_energy_kJ"]
        kcal = result["coulomb_energy_kcal"]
        assert kcal == pytest.approx(kJ / 4.184, rel=1e-4)

    def test_charged_atoms_have_expected_format(self):
        """Charged atom dicts should contain required keys."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH,
            peptide_chain="M",
            receptor_chain="R",
        )

        for info in result["charged_atoms_peptide"]:
            assert "residue" in info
            assert "atom" in info
            assert "charge" in info
            assert "coords" in info
            assert len(info["coords"]) == 3

    def test_n_attractive_plus_repulsive_leq_total(self):
        """Attractive + repulsive pairs should not exceed total charged pairs."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH,
            peptide_chain="M",
            receptor_chain="R",
        )

        assert result["n_attractive"] + result["n_repulsive"] <= result["n_charged_pairs"]

    def test_auto_chain_detection(self):
        """Should auto-detect chains when not specified."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result_auto = compute_coulomb_cross_chain(EXAMPLE_PDB_PATH)
        result_explicit = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R"
        )

        assert result_auto["coulomb_energy_kJ"] == pytest.approx(
            result_explicit["coulomb_energy_kJ"], rel=1e-4
        )

    def test_cutoff_reduces_pairs(self):
        """Smaller cutoff should reduce number of charged pairs."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result_large = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R", cutoff_ang=20.0
        )
        result_small = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R", cutoff_ang=5.0
        )

        assert result_small["n_charged_pairs"] <= result_large["n_charged_pairs"]

    def test_dielectric_scales_energy(self):
        """Doubling dielectric constant should halve the energy."""
        _skip_if_no_biotite()
        if not EXAMPLE_PDB_PATH.exists():
            pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result_d4 = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R", dielectric=4.0
        )
        result_d8 = compute_coulomb_cross_chain(
            EXAMPLE_PDB_PATH, peptide_chain="M", receptor_chain="R", dielectric=8.0
        )

        # Energy should be inversely proportional to dielectric
        if abs(result_d4["coulomb_energy_kJ"]) > 1e-6:
            assert result_d8["coulomb_energy_kJ"] == pytest.approx(
                result_d4["coulomb_energy_kJ"] / 2.0, rel=1e-3
            )

    def test_synthetic_attractive_pair(self, tmp_path: Path):
        """Synthetic structure: one LYS NZ and one ASP OD1 should be attractive."""
        _skip_if_no_biotite()
        # Build minimal PDB with one charged pair at ~5 Å distance
        pdb_content = """\
ATOM      1  NZ  LYS M   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  LYS M   1      -1.000   0.000   0.000  1.00  0.00           C
ATOM      3  OD1 ASP R   1       4.000   0.000   0.000  1.00  0.00           O
ATOM      4  CA  ASP R   1       5.000   0.000   0.000  1.00  0.00           C
END
"""
        pdb_path = tmp_path / "synthetic.pdb"
        pdb_path.write_text(pdb_content)

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            pdb_path, peptide_chain="M", receptor_chain="R", cutoff_ang=12.0
        )

        assert result["n_attractive"] >= 1
        assert result["coulomb_energy_kJ"] < 0.0, "LYS-ASP should be attractive (negative energy)"

    def test_synthetic_repulsive_pair(self, tmp_path: Path):
        """Synthetic structure: two LYS NZ groups should be repulsive."""
        _skip_if_no_biotite()
        pdb_content = """\
ATOM      1  NZ  LYS M   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  LYS M   1      -1.000   0.000   0.000  1.00  0.00           C
ATOM      3  NZ  LYS R   1       4.000   0.000   0.000  1.00  0.00           N
ATOM      4  CA  LYS R   1       5.000   0.000   0.000  1.00  0.00           C
END
"""
        pdb_path = tmp_path / "repulsive.pdb"
        pdb_path.write_text(pdb_content)

        from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

        result = compute_coulomb_cross_chain(
            pdb_path, peptide_chain="M", receptor_chain="R", cutoff_ang=12.0
        )

        assert result["n_repulsive"] >= 1
        assert result["coulomb_energy_kJ"] > 0.0, "LYS-LYS should be repulsive (positive energy)"
