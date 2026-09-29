"""`reason` key on sentinel results of the geometry and comparison metrics.

When a value cannot be computed the sentinel (NaN, 0 or None) is unchanged and
a short `reason` string says why. On success the key is absent.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("scipy")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics import geometry  # noqa: E402
from binding_metrics.metrics.geometry import (  # noqa: E402
    compute_buried_void_volume,
    compute_omega_planarity,
    compute_ramachandran,
    compute_shape_complementarity,
)

DATA_DIR = Path(__file__).parent.parent / "data"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"

HAS_GEMMI = importlib.util.find_spec("gemmi") is not None


def _write_pdb(path: Path, atoms: list[tuple]) -> Path:
    """Each atom is (chain, res_id, res_name, atom_name, element, (x, y, z)); hetero for HOH."""
    arr = struc.AtomArray(len(atoms))
    arr.chain_id = np.array([a[0] for a in atoms])
    arr.res_id = np.array([a[1] for a in atoms])
    arr.res_name = np.array([a[2] for a in atoms])
    arr.atom_name = np.array([a[3] for a in atoms])
    arr.element = np.array([a[4] for a in atoms])
    arr.coord = np.array([a[5] for a in atoms], dtype=np.float32)
    arr.hetero = np.array([a[2] == "HOH" for a in atoms])
    pdb = pdb_io.PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(path))
    return path


def _residue(chain: str, res_id: int, offset: tuple[float, float, float] = (0, 0, 0)) -> list:
    """One alanine backbone (N, CA, C, O) placed at ``offset``."""
    sites = [
        ("N", "N", (0.0, 0.0, 0.0)),
        ("CA", "C", (1.46, 0.0, 0.0)),
        ("C", "C", (2.0, 1.4, 0.0)),
        ("O", "O", (1.4, 2.4, 0.0)),
    ]
    return [(chain, res_id, "ALA", name, el, tuple(np.add(xyz, offset))) for name, el, xyz in sites]


@pytest.fixture
def lone_residues(tmp_path) -> Path:
    """Two single-residue chains far apart: no phi/psi/omega, no interface."""
    atoms = _residue("A", 1) + _residue("B", 1, offset=(100.0, 0.0, 0.0))
    return _write_pdb(tmp_path / "lone.pdb", atoms)


@pytest.fixture
def waters_only(tmp_path) -> Path:
    atoms = [("A", i, "HOH", "O", "O", (3.0 * i, 0.0, 0.0)) for i in range(1, 4)]
    return _write_pdb(tmp_path / "waters.pdb", atoms)


class TestBackboneMetrics:
    def test_no_protein_chain(self, waters_only):
        for func, nan_key in (
            (compute_ramachandran, "ramachandran_favoured_pct"),
            (compute_omega_planarity, "omega_mean_dev"),
        ):
            result = func(waters_only)
            assert np.isnan(result[nan_key])
            assert result["per_residue"] == []
            assert "auto-detected" in result["reason"]

    def test_chain_without_dihedrals(self, lone_residues):
        rama = compute_ramachandran(lone_residues, chain="A")
        assert np.isnan(rama["ramachandran_outlier_pct"])
        assert rama["n_residues_evaluated"] == 0
        assert "phi/psi" in rama["reason"] and "'A'" in rama["reason"]

        omega = compute_omega_planarity(lone_residues, chain="A")
        assert np.isnan(omega["omega_max_dev"])
        assert omega["n_bonds_evaluated"] == 0
        assert "omega" in omega["reason"] and "'A'" in omega["reason"]

    @pytest.mark.skipif(not P53_MDM2.exists(), reason="1YCR example not bundled")
    def test_no_reason_on_success(self):
        assert "reason" not in compute_ramachandran(P53_MDM2, chain="B")
        assert "reason" not in compute_omega_planarity(P53_MDM2, chain="B")


class TestShapeComplementarityReason:
    def test_single_chain_structure(self, tmp_path):
        path = _write_pdb(tmp_path / "one.pdb", _residue("A", 1) + _residue("A", 2, (3.8, 0, 0)))
        result = compute_shape_complementarity(path)
        assert np.isnan(result["sc"])
        assert result["n_surface_dots_A"] == 0
        assert len(result["per_dot_scores_A"]) == 0
        assert "could not be determined" in result["reason"]

    @pytest.mark.skipif(not P53_MDM2.exists(), reason="1YCR example not bundled")
    def test_missing_chain_is_named(self):
        result = compute_shape_complementarity(P53_MDM2, peptide_chain="Z", receptor_chain="A")
        assert np.isnan(result["sc"])
        assert "peptide chain 'Z'" in result["reason"]
        result = compute_shape_complementarity(P53_MDM2, peptide_chain="B", receptor_chain="Z")
        assert "receptor chain 'Z'" in result["reason"]

    def test_chains_too_far_apart(self, lone_residues):
        result = compute_shape_complementarity(lone_residues, peptide_chain="A", receptor_chain="B")
        assert np.isnan(result["sc"])
        assert "interface_cutoff" in result["reason"]

    @pytest.mark.skipif(not P53_MDM2.exists(), reason="1YCR example not bundled")
    def test_no_reason_on_success(self):
        result = compute_shape_complementarity(P53_MDM2, peptide_chain="B", receptor_chain="A")
        assert np.isfinite(result["sc"])
        assert "reason" not in result


class TestVoidVolumeReason:
    def test_single_chain_structure(self, tmp_path):
        path = _write_pdb(tmp_path / "one.pdb", _residue("A", 1) + _residue("A", 2, (3.8, 0, 0)))
        result = compute_buried_void_volume(path)
        assert np.isnan(result["void_volume_A3"])
        assert result["n_interface_atoms"] == 0
        assert "could not be determined" in result["reason"]

    @pytest.mark.skipif(not P53_MDM2.exists(), reason="1YCR example not bundled")
    def test_missing_chain_is_named(self):
        result = compute_buried_void_volume(P53_MDM2, peptide_chain="Z", receptor_chain="A")
        assert np.isnan(result["void_volume_A3"])
        assert "peptide chain 'Z'" in result["reason"]

    def test_chains_too_far_apart(self, lone_residues):
        result = compute_buried_void_volume(lone_residues, peptide_chain="A", receptor_chain="B")
        assert np.isnan(result["void_grid_fraction"])
        assert "interface_cutoff" in result["reason"]

    def test_grid_out_of_memory_is_reported(self, tmp_path, monkeypatch):
        atoms = _residue("A", 1) + _residue("B", 1, offset=(3.0, 0.0, 0.0))
        path = _write_pdb(tmp_path / "close.pdb", atoms)

        def no_memory(*args, **kwargs):
            raise MemoryError

        monkeypatch.setattr(geometry, "_occupancy_grid", no_memory)
        result = compute_buried_void_volume(path, peptide_chain="A", receptor_chain="B")
        assert np.isnan(result["void_volume_A3"])
        assert "memory" in result["reason"]

    def test_other_errors_are_not_swallowed(self, tmp_path, monkeypatch):
        atoms = _residue("A", 1) + _residue("B", 1, offset=(3.0, 0.0, 0.0))
        path = _write_pdb(tmp_path / "close.pdb", atoms)

        def broken(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(geometry, "_occupancy_grid", broken)
        with pytest.raises(RuntimeError, match="boom"):
            compute_buried_void_volume(path, peptide_chain="A", receptor_chain="B")

    @pytest.mark.skipif(not P53_MDM2.exists(), reason="1YCR example not bundled")
    def test_no_reason_on_success(self):
        assert "reason" not in compute_buried_void_volume(P53_MDM2)


requires_gemmi = pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")


@requires_gemmi
class TestStructureRmsdReason:
    @pytest.fixture
    def disjoint_pair(self, tmp_path):
        first = _write_pdb(tmp_path / "first.pdb", _residue("A", 1) + _residue("A", 2, (3.8, 0, 0)))
        second = _write_pdb(tmp_path / "second.pdb", _residue("B", 7))
        return first, second

    def test_variants_that_cannot_be_matched_get_a_reason(self, disjoint_pair):
        from binding_metrics.metrics.comparison import compute_structure_rmsd

        result = compute_structure_rmsd(*disjoint_pair, design_chain="A")
        for key in ("rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"):
            assert result[key] is None
        reason = result["reason"]
        assert reason.startswith("not computed:")
        assert "no (chain, residue number, atom name) key is shared" in reason
        assert "rmsd_design (no atoms selected in the processed structure)" in reason

    def test_no_reason_when_everything_is_computed(self, disjoint_pair):
        from binding_metrics.metrics.comparison import compute_structure_rmsd

        first, _ = disjoint_pair
        result = compute_structure_rmsd(first, first)
        assert "reason" not in result
        assert set(result) == {"rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"}

    def test_cli_keeps_stdout_and_reports_the_reason_on_stderr(
        self, disjoint_pair, monkeypatch, capsys
    ):
        from binding_metrics.metrics.comparison import main

        first, second = disjoint_pair
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-compare", "--initial", str(first), "--processed", str(second)],
        )
        main()
        captured = capsys.readouterr()
        assert "  rmsd: N/A" in captured.out
        assert "reason" not in captured.out
        assert "reason: not computed" in captured.err
