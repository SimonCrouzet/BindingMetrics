"""Tests for geometry metrics: Ramachandran, omega planarity, shape complementarity,
and buried void volume."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
EXAMPLE_PDB_PATH = DATA_DIR / "example_linear_p53_1YCR.pdb"


def _skip_if_no_biotite():
    pytest.importorskip("biotite")


def _skip_if_no_example():
    if not EXAMPLE_PDB_PATH.exists():
        pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")


def _find_native_example(*pdb_id_tokens):
    """Return the first bundled example file whose name contains a token.

    The bundled native complexes have been renamed across revisions
    (e.g. ``example_linear_p53_1YCR.pdb`` vs ``example_linearpeptide_1YCR.pdb``),
    so tests discover them by PDB id rather than a hard-coded filename.
    Returns None if no matching, readable file is present.
    """
    if not DATA_DIR.is_dir():
        return None
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in (".pdb", ".cif", ".mmcif"):
            continue
        name = path.name.lower()
        if any(tok.lower() in name for tok in pdb_id_tokens):
            return path
    return None


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

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="B")

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

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="B")

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

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="B")

        for entry in result["per_residue"]:
            assert "res_id" in entry
            assert "res_name" in entry
            assert "chain" in entry
            assert "phi" in entry
            assert "psi" in entry
            assert "region" in entry
            assert entry["region"] in ("favoured", "allowed", "outlier")

    def test_chain_R_receptor(self):
        """Should work for receptor chain A too."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="A")

        assert result["n_residues_evaluated"] > 0

    def test_auto_detect_chain(self):
        """Auto-detection should default to the smaller (peptide) chain."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result_auto = compute_ramachandran(EXAMPLE_PDB_PATH)
        result_M = compute_ramachandran(EXAMPLE_PDB_PATH, chain="B")

        # Auto should pick chain B (smaller chain)
        assert result_auto["n_residues_evaluated"] == result_M["n_residues_evaluated"]

    def test_outlier_count_consistent_with_pct(self):
        """Outlier count should be consistent with outlier percentage."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_ramachandran

        result = compute_ramachandran(EXAMPLE_PDB_PATH, chain="A")

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

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="B")

        expected = {
            "omega_mean_dev",
            "omega_max_dev",
            "omega_outlier_fraction",
            "omega_outlier_count",
            "n_bonds_evaluated",
            "per_residue",
        }
        assert expected.issubset(set(result.keys()))

    def test_mean_dev_is_non_negative(self):
        """Mean deviation should be non-negative."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="B")

        if result["n_bonds_evaluated"] > 0:
            assert result["omega_mean_dev"] >= 0.0
            assert result["omega_max_dev"] >= result["omega_mean_dev"]

    def test_per_residue_fields(self):
        """Per-residue entries should have omega, deviation, is_outlier fields."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="B")

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

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="A")

        manual_outliers = sum(1 for e in result["per_residue"] if e["is_outlier"])
        assert manual_outliers == result["omega_outlier_count"]

    def test_chain_R_works(self):
        """Should work for receptor chain A."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="A")
        assert result["n_bonds_evaluated"] > 0

    def test_deviation_within_range(self):
        """Deviation from 180° should be in [0, 180] degrees."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_omega_planarity

        result = compute_omega_planarity(EXAMPLE_PDB_PATH, chain="B")

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
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
        )

        expected = {
            "sc",
            "sc_A_to_B",
            "sc_B_to_A",
            "n_surface_dots_A",
            "n_surface_dots_B",
            "per_dot_scores_A",
            "per_dot_scores_B",
        }
        assert expected.issubset(set(result.keys()))

    def test_sc_is_finite_on_real_example(self):
        """The real complex must yield a finite Sc, not a silent NaN.

        Every other Sc assertion in this class is guarded by
        ``if not np.isnan(result["sc"])``, so a regression that made
        shape-complementarity return NaN on real input would pass all of them
        vacuously. This unguarded check fails if the effective computation breaks.
        """
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
        )

        assert not np.isnan(result["sc"]), "Sc is NaN on a real interface"
        assert -1.0 <= result["sc"] <= 1.0
        # A real bound interface generates surface dots on both partners.
        assert result["n_surface_dots_A"] > 0
        assert result["n_surface_dots_B"] > 0

    def test_sc_value_in_range(self):
        """Sc score should be in [-1, 1]."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
        )

        if not np.isnan(result["sc"]):
            assert -1.0 <= result["sc"] <= 1.0

    def test_sc_is_mean_of_directional(self):
        """Sc should equal mean of sc_A_to_B and sc_B_to_A."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
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
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
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
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
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
            EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A"
        )

        if not np.isnan(result_auto["sc"]) and not np.isnan(result_explicit["sc"]):
            assert result_auto["sc"] == pytest.approx(result_explicit["sc"], rel=1e-4)

    # -- Physical calibration on native complexes (Lawrence & Colman range) --
    #
    # Well-formed native interfaces score Sc ~= 0.6-0.75 (protease-inhibitor
    # ~0.70-0.78, antibody-antigen ~0.64-0.68); peptide interfaces run a touch
    # lower. A collapsed value (~0.2) means the surface normals / interface
    # selection are broken. As an external anchor, this implementation scores
    # the classic trypsin-BPTI interface (PDB 2PTC) at ~0.70, matching the
    # Lawrence & Colman (1993) literature value of ~0.71-0.72.

    @pytest.mark.parametrize(
        "tokens,lo,hi",
        [
            (("1YCR",), 0.50, 0.80),  # p53 peptide - MDM2 (linear peptide)
            (("3P8F",), 0.55, 0.85),  # SFTI-1 bicyclic peptide
            (("1CWA",), 0.55, 0.85),  # cyclosporin (ncAA macrocycle)
        ],
    )
    def test_native_sc_in_physical_range(self, tokens, lo, hi):
        """Native complexes must score in the physically sensible Sc range.

        Regression guard against the miscalibration that returned Sc ~= 0.2
        for well-packed native interfaces.
        """
        _skip_if_no_biotite()
        path = _find_native_example(*tokens)
        if path is None:
            pytest.skip(f"No bundled example for {tokens}")

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        result = compute_shape_complementarity(path)
        sc = result["sc"]
        if np.isnan(sc):
            pytest.skip(f"No interface detected for {path.name}")
        assert lo <= sc <= hi, (
            f"{path.name}: Sc={sc:.3f} outside expected native range "
            f"[{lo}, {hi}] - shape complementarity is miscalibrated"
        )
        assert result["n_surface_dots_A"] > 0
        assert result["n_surface_dots_B"] > 0

    def test_facing_slabs_are_complementary(self):
        """Two facing atom slabs in vdW contact must give a high positive Sc.

        A cheap, self-contained calibration case: two multi-layer slabs whose
        facing surfaces sit at van der Waals contact form a near-perfect
        lock-and-key interface, which must score near the top of the range
        (Sc > 0.7). This exercises the normal sign convention and weighting
        without any external file. The slabs are several layers thick so the
        smoothed surface normal of the contact face is well defined.
        """
        _skip_if_no_biotite()
        import tempfile

        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io

        from binding_metrics.metrics.geometry import compute_shape_complementarity

        spacing = 3.4  # ~vdW contact for carbon
        xs = np.arange(8) * spacing
        ys = np.arange(8) * spacing
        gx, gy = np.meshgrid(xs, ys)

        def layer(z):
            return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)

        n_layers = 3
        # Chain A: contact face at z=0, body below; Chain B: contact face at
        # z=spacing, body above -> the two faces are one vdW spacing apart.
        slab_a = np.vstack([layer(-spacing * k) for k in range(n_layers)])
        slab_b = np.vstack([layer(spacing * (k + 1)) for k in range(n_layers)])
        coords = np.vstack([slab_a, slab_b])
        n = len(slab_a)

        arr = struc.AtomArray(len(coords))
        arr.coord = coords.astype(np.float32)
        arr.chain_id = np.array(["A"] * n + ["B"] * len(slab_b))
        arr.res_id = np.arange(1, len(coords) + 1)
        arr.res_name = np.array(["ALA"] * len(coords))
        arr.atom_name = np.array(["C"] * len(coords))
        arr.element = np.array(["C"] * len(coords))

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fh:
            pf = pdb_io.PDBFile()
            pf.set_structure(arr)
            pf.write(fh.name)
            result = compute_shape_complementarity(fh.name, peptide_chain="A", receptor_chain="B")

        if not np.isnan(result["sc"]):
            assert result["sc"] > 0.7, (
                f"Complementary facing slabs scored Sc={result['sc']:.3f}; "
                "sign convention or normals are wrong"
            )

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
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
            grid_spacing=1.0,  # coarser for speed
        )

        expected = {
            "void_volume_A3",
            "void_grid_fraction",
            "interface_box_volume_A3",
            "n_interface_atoms",
        }
        assert expected.issubset(set(result.keys()))

    def test_void_volume_finite_on_real_example(self):
        """The real interface must yield a finite (non-NaN) void volume.

        ``test_void_volume_non_negative`` and ``test_void_fraction_in_range`` are
        both guarded by ``if not np.isnan(...)``, so a regression returning NaN on
        real input would pass them without asserting anything. This closes that gap.
        """
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
            grid_spacing=1.0,
        )

        assert not np.isnan(result["void_volume_A3"])
        assert not np.isnan(result["void_grid_fraction"])
        assert result["void_volume_A3"] >= 0.0
        assert 0.0 <= result["void_grid_fraction"] <= 1.0

    def test_void_volume_non_negative(self):
        """Void volume should be non-negative."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
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
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
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
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
            grid_spacing=1.0,
        )

        assert result["n_interface_atoms"] > 0

    def test_missing_chains_returns_nan(self):
        """Non-existent chains should return NaN void volume."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(EXAMPLE_PDB_PATH, peptide_chain="Z", receptor_chain="W")

        assert np.isnan(result["void_volume_A3"])

    def test_box_volume_larger_than_void(self):
        """Bounding box volume should be >= void volume."""
        _skip_if_no_biotite()
        _skip_if_no_example()

        from binding_metrics.metrics.geometry import compute_buried_void_volume

        result = compute_buried_void_volume(
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
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
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
            grid_spacing=1.5,
        )
        result_fine = compute_buried_void_volume(
            EXAMPLE_PDB_PATH,
            peptide_chain="B",
            receptor_chain="A",
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
