"""Tests for interface analysis metrics (SASA, ΔG_int, H-bonds, salt bridges)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HAS_BIOTITE = importlib.util.find_spec("biotite") is not None

requires_biotite = pytest.mark.skipif(not HAS_BIOTITE, reason="biotite not installed")

# CIF fixture shared with other integration tests
EXAMPLE_CIF = Path("data/example_linear_p53_1YCR.pdb")


class TestDetectInterfaceChains:
    """Tests for detect_interface_chains."""

    @requires_biotite
    def test_returns_tuple(self):
        from binding_metrics.metrics.interface import (
            detect_interface_chains,
            load_biotite_structure,
        )

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        atoms = load_biotite_structure(EXAMPLE_CIF)
        pep, rec = detect_interface_chains(atoms)
        assert isinstance(pep, str)
        assert isinstance(rec, str)
        assert pep != rec

    @requires_biotite
    def test_explicit_design_chain(self):
        from binding_metrics.metrics.interface import (
            detect_interface_chains,
            load_biotite_structure,
        )

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
            "hbonds", "hbond_energy",
            "saltbridges", "saltbridges_bidentate", "saltbridge_energy",
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
        assert isinstance(r["saltbridges_bidentate"], int)
        assert isinstance(r["hbond_energy"], float)
        assert isinstance(r["saltbridge_energy"], float)
        assert r["hbonds"] >= 0
        assert r["saltbridges"] >= 0
        assert r["saltbridges_bidentate"] >= 0
        assert r["hbond_energy"] <= 0.0
        assert r["saltbridge_energy"] <= 0.0


def _make_atom_array(records):
    """Build an AtomArray from (chain, res_id, res_name, atom_name, element, xyz) tuples."""
    import biotite.structure as struc
    n = len(records)
    arr = struc.AtomArray(n)
    arr.chain_id  = np.array([r[0] for r in records])
    arr.res_id    = np.array([r[1] for r in records], dtype=int)
    arr.res_name  = np.array([r[2] for r in records])
    arr.atom_name = np.array([r[3] for r in records])
    arr.element   = np.array([r[4] for r in records])
    arr.coord     = np.array([r[5] for r in records], dtype=float)
    return arr


@requires_biotite
class TestPolarContacts:
    """Unit tests for compute_saltbridges / compute_hbonds (in metrics.polar_contacts)."""

    def test_saltbridges_synthetic_all_cases(self):
        """One synthetic complex covering: ARG-GLU bidentate, LYS-ASP monodentate,
        HIS-as-neutral, HIP-as-positive, intra-chain ignored, far pair excluded."""
        from binding_metrics.metrics.polar_contacts import compute_saltbridges

        atoms = _make_atom_array([
            # ARG res 1 on A — bidentate target for GLU on B
            ("A", 1, "ARG", "NH1", "N", (0.0, 0.0, 0.0)),
            ("A", 1, "ARG", "NH2", "N", (0.0, 1.5, 0.0)),
            ("A", 1, "ARG", "NE",  "N", (0.0, -1.5, 0.0)),
            # HIS res 2 on A — must be treated as neutral
            ("A", 2, "HIS", "ND1", "N", (10.0, 0.0, 0.0)),
            ("A", 2, "HIS", "NE2", "N", (10.0, 1.5, 0.0)),
            # HIP res 3 on A — positive; NE2 placed far so HIP-ASP is monodentate
            ("A", 3, "HIP", "ND1", "N", (20.0, 0.0, 0.0)),
            ("A", 3, "HIP", "NE2", "N", (20.0, 5.0, 0.0)),
            # LYS res 4 on A — monodentate to ASP res 2 on B
            ("A", 4, "LYS", "NZ", "N", (30.0, 0.0, 0.0)),
            # Intra-chain ASP res 5 on A — must NOT count against LYS-A4
            ("A", 5, "ASP", "OD1", "O", (32.0, 0.0, 0.0)),
            # GLU res 1 on B — bidentate partner of ARG-A1
            ("B", 1, "GLU", "OE1", "O", (2.8, 0.0, 0.0)),
            ("B", 1, "GLU", "OE2", "O", (3.0, 1.5, 0.0)),
            # ASP res 2 on B — monodentate to LYS-A4
            ("B", 2, "ASP", "OD1", "O", (33.0, 0.0, 0.0)),
            ("B", 2, "ASP", "OD2", "O", (38.0, 0.0, 0.0)),  # far → not bidentate
            # ASP res 3 on B — close to HIS-A2 (neutral, must be ignored)
            ("B", 3, "ASP", "OD1", "O", (13.0, 0.0, 0.0)),
            # ASP res 4 on B — close to HIP-A3 (must form salt bridge)
            ("B", 4, "ASP", "OD1", "O", (23.0, 0.0, 0.0)),
            # ASP res 5 on B — far from everything
            ("B", 5, "ASP", "OD1", "O", (100.0, 0.0, 0.0)),
        ])

        r = compute_saltbridges(atoms, "A", "B")

        # Expected residue-pair contacts:
        #   ARG-A1 ↔ GLU-B1  (bidentate, r_min = 2.8 Å)
        #   LYS-A4 ↔ ASP-B2  (monodentate, r_min = 3.0 Å)
        #   HIP-A3 ↔ ASP-B4  (monodentate, r_min = 3.0 Å; HIP counts)
        # Excluded: HIS-A2 (neutral), LYS-A4↔ASP-A5 (intra-chain), ASP-B5 (far).
        assert r["saltbridges"] == 3
        assert r["saltbridges_bidentate"] == 1
        expected = -83.0159 * (1.0 / 2.8 + 1.0 / 3.0 + 1.0 / 3.0)
        assert r["saltbridge_energy"] == pytest.approx(expected, rel=1e-3)

    @pytest.mark.integration
    def test_hbonds_integration_with_and_without_explicit_h(self):
        """End-to-end: hbond detection runs on the example CIF, and stripping
        hydrogens still produces non-zero output (hydride re-adds them)."""
        from binding_metrics.metrics.interface import load_biotite_structure
        from binding_metrics.metrics.polar_contacts import compute_hbonds

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        atoms = load_biotite_structure(EXAMPLE_CIF)
        chains = list(np.unique(atoms.chain_id))
        if len(chains) < 2:
            pytest.skip("Test CIF has fewer than 2 chains")

        r_full = compute_hbonds(atoms, chains[0], chains[1])
        assert set(r_full.keys()) == {"hbond_energy", "hbonds"}
        assert isinstance(r_full["hbonds"], int) and r_full["hbonds"] >= 0
        assert isinstance(r_full["hbond_energy"], float) and r_full["hbond_energy"] <= 0.0

        try:
            import hydride  # noqa: F401
        except ImportError:
            pytest.skip("hydride not installed for the no-H path")

        atoms_heavy = atoms[atoms.element != "H"]
        r_heavy = compute_hbonds(atoms_heavy, chains[0], chains[1])
        assert r_heavy["hbonds"] >= 0
        assert r_heavy["hbond_energy"] <= 0.0

    def test_missing_biotite_raises(self, tmp_path):
        """Should raise ImportError when biotite is not available."""
        import sys
        import unittest.mock as mock

        dummy_cif = tmp_path / "dummy.cif"
        dummy_cif.write_text("data_dummy\n")

        with mock.patch.dict(sys.modules, {"biotite": None, "biotite.structure": None}):
            with pytest.raises(ImportError, match="biotite"):
                import importlib

                from binding_metrics.metrics import interface as iface_mod
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

    @requires_biotite
    def test_loads_pdb(self, sample_pdb_path):
        import biotite.structure as struc

        from binding_metrics.metrics.interface import load_biotite_structure

        atoms = load_biotite_structure(sample_pdb_path)
        assert isinstance(atoms, struc.AtomArray)
        assert len(atoms) > 0

    @requires_biotite
    def test_pdb_chains_preserved(self, sample_pdb_path):
        import numpy as np

        from binding_metrics.metrics.interface import load_biotite_structure

        atoms = load_biotite_structure(sample_pdb_path)
        chains = set(np.unique(atoms.chain_id))
        assert "A" in chains
        assert "B" in chains


class TestComputeInterfaceMetricsPDB:
    """Verify compute_interface_metrics accepts PDB in addition to CIF."""

    @requires_biotite
    def test_runs_on_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.interface import compute_interface_metrics

        result = compute_interface_metrics(
            sample_pdb_path, design_chain="B", receptor_chain="A"
        )
        assert isinstance(result, dict)

    @requires_biotite
    def test_returns_expected_keys_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.interface import compute_interface_metrics

        result = compute_interface_metrics(
            sample_pdb_path, design_chain="B", receptor_chain="A"
        )
        expected_keys = {
            "peptide_chain", "receptor_chain",
            "delta_sasa", "sasa_peptide", "sasa_receptor", "sasa_complex",
            "delta_g_int", "delta_g_int_kJ",
            "polar_area", "apolar_area", "fraction_polar",
            "n_interface_residues_peptide", "n_interface_residues_receptor",
            "interface_residues_peptide", "interface_residues_receptor",
            "per_residue",
            "hbonds", "hbond_energy",
            "saltbridges", "saltbridges_bidentate", "saltbridge_energy",
        }
        assert set(result.keys()) == expected_keys

    @requires_biotite
    def test_chain_ids_correct_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.interface import compute_interface_metrics

        result = compute_interface_metrics(
            sample_pdb_path, design_chain="B", receptor_chain="A"
        )
        assert result["peptide_chain"] == "B"
        assert result["receptor_chain"] == "A"

    @requires_biotite
    def test_sasa_values_finite_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.interface import compute_interface_metrics

        result = compute_interface_metrics(
            sample_pdb_path, design_chain="B", receptor_chain="A"
        )
        assert np.isfinite(result["sasa_peptide"])
        assert np.isfinite(result["sasa_receptor"])
        assert np.isfinite(result["sasa_complex"])
        assert result["sasa_peptide"] > 0
        assert result["sasa_receptor"] > 0


class TestComputeDeltaSasaStatic:
    """Tests for compute_delta_sasa_static — CIF and PDB support."""

    @requires_biotite
    def test_runs_on_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        result = compute_delta_sasa_static(
            sample_pdb_path, peptide_chain="B", receptor_chain="A"
        )
        assert isinstance(result, dict)

    @requires_biotite
    def test_returns_expected_keys_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        result = compute_delta_sasa_static(
            sample_pdb_path, peptide_chain="B", receptor_chain="A"
        )
        assert set(result.keys()) == {
            "delta_sasa", "sasa_peptide", "sasa_receptor", "sasa_complex"
        }

    @requires_biotite
    def test_sasa_values_positive_pdb(self, sample_pdb_path):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        result = compute_delta_sasa_static(
            sample_pdb_path, peptide_chain="B", receptor_chain="A"
        )
        assert result["sasa_peptide"] > 0
        assert result["sasa_receptor"] > 0
        assert result["sasa_complex"] > 0

    @requires_biotite
    @pytest.mark.integration
    def test_runs_on_cif(self):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        if not EXAMPLE_CIF.exists():
            pytest.skip("Test CIF not available")

        result = compute_delta_sasa_static(
            EXAMPLE_CIF, peptide_chain="A", receptor_chain="B"
        )
        assert np.isfinite(result["delta_sasa"])
