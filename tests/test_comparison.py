"""Tests for structure comparison utilities (RMSD between structures)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics.comparison import (
    _get_coords,
    _kabsch_rmsd,
    _matched_rmsd,
    compute_structure_rmsd,
)

HAS_GEMMI = importlib.util.find_spec("gemmi") is not None

requires_gemmi = pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")

EXAMPLE_CIF = Path("data/example_linear_p53_1YCR.pdb")
EXAMPLE_CIF2 = Path("data/example_bicyclic_sfti1_3P8F.cif")


def _pdb_lines(chains: dict[str, np.ndarray]) -> list[str]:
    """ATOM records for backbone-only residues; each chain maps to (n_res, 4, 3) coordinates."""
    lines = []
    serial = 1
    for chain_id, residues in chains.items():
        for res_num, residue in enumerate(residues, start=1):
            for atom_name, xyz in zip((" N  ", " CA ", " C  ", " O  "), residue):
                element = atom_name.strip()[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name} ALA {chain_id}{res_num:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
                )
                serial += 1
    return lines


def _write_pdb(path: Path, models: list[dict[str, np.ndarray]]) -> Path:
    """Write a (multi-)model PDB file from per-model chain coordinates."""
    out = []
    for i, chains in enumerate(models, start=1):
        if len(models) > 1:
            out.append(f"MODEL     {i:4d}")
        out.extend(_pdb_lines(chains))
        if len(models) > 1:
            out.append("ENDMDL")
    out.append("END")
    path.write_text("\n".join(out) + "\n")
    return path


def _rotation_matrix(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for column vectors (apply to rows as ``p @ R.T``)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * cross + (1 - np.cos(angle_rad)) * cross @ cross


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

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_rigidly_rotated_and_translated_copy_is_zero(self, seed):
        """A rigid-body copy of a point set superposes exactly.

        Regression: the rotation was applied as ``p @ R`` instead of
        ``p @ R.T``, so any pair that was not already co-oriented came back
        with an RMSD of 1-2 Angstrom instead of 0.
        """
        rng = np.random.default_rng(seed)
        coords = rng.normal(size=(50, 3)) * 5.0
        rotation = _rotation_matrix(rng.normal(size=3), rng.uniform(0.3, 3.0))
        moved = coords @ rotation.T + rng.normal(size=3) * 10.0
        assert _kabsch_rmsd(coords, moved) < 1e-6
        # symmetric in its arguments
        assert _kabsch_rmsd(moved, coords) < 1e-6

    @pytest.mark.parametrize("angle_deg", [5.0, 60.0, 90.0, 179.0])
    def test_known_rotation_angle_is_zero(self, angle_deg):
        """A pure rotation about z by a known angle gives zero, including 5 degrees.

        Before the fix a 5 degree rotation of a unit-normal cloud reported
        an RMSD of about 0.22.
        """
        rng = np.random.default_rng(7)
        coords = rng.normal(size=(30, 3))
        rotation = _rotation_matrix([0.0, 0.0, 1.0], np.deg2rad(angle_deg))
        assert _kabsch_rmsd(coords, coords @ rotation.T) < 1e-9

    def test_known_nonzero_rmsd_survives_rotation(self):
        """Stretching a rectangle by 0.5 A per end atom gives RMSD 0.5 in any pose.

        The identity is the optimal superposition of a rectangle on its
        symmetric stretched copy, so the value is known in closed form.
        """
        rectangle = np.array(
            [[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]]
        )
        stretched = rectangle * np.array([2.5 / 2.0, 1.0, 1.0])
        rotation = _rotation_matrix([1.0, 2.0, 3.0], 1.1)
        moved = stretched @ rotation.T + np.array([4.0, -7.0, 2.0])
        assert _kabsch_rmsd(rectangle, stretched) == pytest.approx(0.5, abs=1e-9)
        assert _kabsch_rmsd(rectangle, moved) == pytest.approx(0.5, abs=1e-9)

    def test_mirror_image_is_not_superposed(self):
        """Only proper rotations are allowed: a chiral set never matches its mirror image."""
        rng = np.random.default_rng(3)
        coords = rng.normal(size=(20, 3)) * 3.0
        mirrored = coords * np.array([-1.0, 1.0, 1.0])
        assert _kabsch_rmsd(coords, mirrored) > 0.5


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


class TestMultiModelInput:
    """Only the first model of an ensemble file enters the comparison."""

    @staticmethod
    def _ensemble(tmp_path: Path):
        """Two-chain structure (A: 4 residues, B: 2 residues) in two very different models."""
        rng = np.random.default_rng(11)
        model1 = {"A": rng.normal(size=(4, 4, 3)) * 4.0, "B": rng.normal(size=(2, 4, 3)) * 4.0}
        model2 = {
            "A": model1["A"] + rng.normal(size=(4, 4, 3)) * 3.0,
            "B": model1["B"] + rng.normal(size=(2, 4, 3)) * 3.0,
        }
        single = _write_pdb(tmp_path / "single.pdb", [model1])
        multi = _write_pdb(tmp_path / "multi.pdb", [model1, model2])
        return model1, single, multi

    @requires_gemmi
    def test_get_coords_reads_first_model_only(self, tmp_path):
        import gemmi

        model1, _, multi = self._ensemble(tmp_path)
        coords, keys = _get_coords(gemmi.read_structure(str(multi)))
        assert coords.shape == (24, 3)  # (4 + 2) residues x 4 backbone atoms, one model
        assert len(set(keys)) == len(keys)  # no duplicated (chain, res, atom) keys
        expected = np.concatenate([model1["A"].reshape(-1, 3), model1["B"].reshape(-1, 3)])
        np.testing.assert_allclose(coords, expected, atol=1e-3)  # PDB has 3 decimals

    @requires_gemmi
    def test_ensembles_with_equal_first_model_give_zero(self, tmp_path):
        """Two ensembles whose model 1 agree but whose later models differ compare as 0.

        With equal atom counts the direct-Kabsch path used to run on the
        concatenation of all models, so the differing model 2 leaked in.
        """
        model1, _, multi = self._ensemble(tmp_path)
        rng = np.random.default_rng(12)
        other_model2 = {k: v + rng.normal(size=v.shape) * 5.0 for k, v in model1.items()}
        other = _write_pdb(tmp_path / "other.pdb", [model1, other_model2])
        result = compute_structure_rmsd(multi, other, design_chain="B")
        for key in ("rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"):
            assert result[key] is not None
            assert result[key] < 1e-2, key

    @requires_gemmi
    def test_design_chain_autodetect_ignores_later_models(self, tmp_path):
        """The auto-detected design chain is the smallest chain of model 1 (chain B).

        Model 2 carries an extra one-residue chain C that must not win.
        """
        model1, single, _ = self._ensemble(tmp_path)
        rng = np.random.default_rng(13)
        model2 = {**model1, "C": rng.normal(size=(1, 4, 3))}
        multi = _write_pdb(tmp_path / "with_extra_chain.pdb", [model1, model2])
        auto = compute_structure_rmsd(multi, single)
        explicit = compute_structure_rmsd(multi, single, design_chain="B")
        assert auto["rmsd_design"] is not None
        assert auto["rmsd_design"] == pytest.approx(explicit["rmsd_design"])
        assert auto["bb_rmsd_design"] == pytest.approx(explicit["bb_rmsd_design"])


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
        are equal and the fast direct-Kabsch path is taken. ``TestMatchedRmsd``
        covers the atom-matching branch (match on (chain,res,atom) when counts
        differ) on synthetic arrays only. Comparing raw 1YCR to its prepped
        variant (hydrogens added → different atom count) drives that branch on
        real data and must return finite, non-None RMSDs rather than erroring or
        silently yielding None.
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
