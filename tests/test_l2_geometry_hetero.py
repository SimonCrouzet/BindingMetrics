"""Heteroatom policy of the Sc and void-volume metrics (`hetero` keyword, `--hetero` flag).

Waters carrying a protein chain ID used to be counted as protein atoms by the
chain-ID masks. On 1CWA (144 waters on chain A) that moved Sc from 0.722 to
0.750 and the void interface atom count from 153 to 123.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("scipy")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402
import biotite.structure.io.pdbx as pdbx  # noqa: E402

from binding_metrics.metrics.geometry import (  # noqa: E402
    _load_structure,
    compute_buried_void_volume,
    compute_shape_complementarity,
    main,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CWA = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"

SC_KEYS = ("sc", "sc_A_to_B", "sc_B_to_A", "n_surface_dots_A", "n_surface_dots_B")
VOID_KEYS = ("void_volume_A3", "void_grid_fraction", "interface_box_volume_A3", "n_interface_atoms")


@pytest.fixture(scope="module")
def cwa_dry(tmp_path_factory) -> Path:
    """1CWA with every non-amino-acid atom (waters) removed, written as CIF."""
    if not CWA.exists():
        pytest.skip(f"{CWA.name} not bundled")
    atoms = _load_structure(CWA)
    dry = atoms[struc.filter_amino_acids(atoms)]
    assert len(dry) < len(atoms)  # the example really does contain waters
    path = tmp_path_factory.mktemp("cwa_dry") / "cwa_dry.cif"
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, dry)
    cif.write(str(path))
    return path


def _assert_same(a: dict, b: dict, keys) -> None:
    for key in keys:
        assert a[key] == pytest.approx(b[key], rel=1e-6, abs=1e-9), key


class TestHeteroOnCyclosporin:
    def test_sc_ignore_on_raw_equals_dry(self, cwa_dry):
        raw = compute_shape_complementarity(CWA, hetero="ignore")
        dry = compute_shape_complementarity(cwa_dry, hetero="ignore")
        _assert_same(raw, dry, SC_KEYS)

    def test_void_ignore_on_raw_equals_dry(self, cwa_dry):
        raw = compute_buried_void_volume(CWA, hetero="ignore")
        dry = compute_buried_void_volume(cwa_dry, hetero="ignore")
        _assert_same(raw, dry, VOID_KEYS)

    def test_default_is_ignore(self, cwa_dry):
        _assert_same(
            compute_shape_complementarity(CWA),
            compute_shape_complementarity(CWA, hetero="ignore"),
            SC_KEYS,
        )
        _assert_same(
            compute_buried_void_volume(CWA),
            compute_buried_void_volume(CWA, hetero="ignore"),
            VOID_KEYS,
        )

    def test_keep_reproduces_the_water_contaminated_inputs(self, cwa_dry):
        keep_sc = compute_shape_complementarity(CWA, hetero="keep")
        dry_sc = compute_shape_complementarity(cwa_dry)
        # waters on the receptor chain raise the number of dotted atoms and change Sc
        assert keep_sc["sc"] == pytest.approx(0.7215, abs=5e-3)
        assert dry_sc["sc"] == pytest.approx(0.7500, abs=5e-3)
        assert keep_sc["sc"] != pytest.approx(dry_sc["sc"], abs=1e-3)

        keep_void = compute_buried_void_volume(CWA, hetero="keep")
        dry_void = compute_buried_void_volume(cwa_dry)
        assert keep_void["n_interface_atoms"] == 153
        assert dry_void["n_interface_atoms"] == 123

    def test_dry_structure_is_unchanged_by_the_policy(self, cwa_dry):
        """No heteroatoms present: keep and ignore are identical."""
        _assert_same(
            compute_shape_complementarity(cwa_dry, hetero="keep"),
            compute_shape_complementarity(cwa_dry, hetero="ignore"),
            SC_KEYS,
        )
        _assert_same(
            compute_buried_void_volume(cwa_dry, hetero="keep"),
            compute_buried_void_volume(cwa_dry, hetero="ignore"),
            VOID_KEYS,
        )

    @pytest.mark.parametrize("func", [compute_shape_complementarity, compute_buried_void_volume])
    def test_invalid_mode_raises(self, func):
        with pytest.raises(ValueError, match="hetero"):
            func(CWA, hetero="drop")


def _slab(n: int, z_values, spacing: float = 3.4) -> np.ndarray:
    """n x n carbon lattice repeated at each z (Angstrom), one atom per lattice point."""
    xs = np.arange(n) * spacing
    gx, gy = np.meshgrid(xs, xs)
    return np.vstack(
        [np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1) for z in z_values]
    )


def _two_slab_pdb(path: Path, with_water: bool) -> Path:
    """Two facing carbon slabs (chains A, B); optionally a HOH labelled chain A in the gap."""
    slab_a = _slab(6, [0.0, -3.4])
    slab_b = _slab(6, [3.4, 6.8])
    coords = [slab_a, slab_b]
    chains = ["A"] * len(slab_a) + ["B"] * len(slab_b)
    res_names = ["ALA"] * (len(slab_a) + len(slab_b))
    atom_names = ["C"] * len(res_names)
    elements = ["C"] * len(res_names)
    hetero = [False] * len(res_names)
    if with_water:
        coords.append(np.array([[6.8, 6.8, 1.7], [10.2, 10.2, 1.7], [3.4, 10.2, 1.7]]))
        for _ in range(3):
            chains.append("A")
            res_names.append("HOH")
            atom_names.append("O")
            elements.append("O")
            hetero.append(True)
    xyz = np.vstack(coords)
    arr = struc.AtomArray(len(xyz))
    arr.coord = xyz.astype(np.float32)
    arr.chain_id = np.array(chains)
    arr.res_id = np.arange(1, len(xyz) + 1)
    arr.res_name = np.array(res_names)
    arr.atom_name = np.array(atom_names)
    arr.element = np.array(elements)
    arr.hetero = np.array(hetero)
    pdb = pdb_io.PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(path))
    return path


class TestHeteroSynthetic:
    """Three waters labelled with the peptide chain ID sit in the interface gap."""

    def test_void_ignore_drops_the_waters(self, tmp_path):
        wet = _two_slab_pdb(tmp_path / "wet.pdb", with_water=True)
        dry = _two_slab_pdb(tmp_path / "dry.pdb", with_water=False)
        kw = dict(peptide_chain="A", receptor_chain="B", grid_spacing=1.0)
        ignored = compute_buried_void_volume(wet, hetero="ignore", **kw)
        reference = compute_buried_void_volume(dry, **kw)
        kept = compute_buried_void_volume(wet, hetero="keep", **kw)
        _assert_same(ignored, reference, VOID_KEYS)
        assert kept["n_interface_atoms"] == reference["n_interface_atoms"] + 3

    def test_sc_ignore_drops_the_waters(self, tmp_path):
        wet = _two_slab_pdb(tmp_path / "wet.pdb", with_water=True)
        dry = _two_slab_pdb(tmp_path / "dry.pdb", with_water=False)
        kw = dict(peptide_chain="A", receptor_chain="B")
        ignored = compute_shape_complementarity(wet, hetero="ignore", **kw)
        reference = compute_shape_complementarity(dry, **kw)
        kept = compute_shape_complementarity(wet, hetero="keep", **kw)
        _assert_same(ignored, reference, SC_KEYS)
        assert kept["n_surface_dots_A"] != reference["n_surface_dots_A"]


class TestPolymerNamesKeptUnderIgnore:
    """AMBER protonation variants and caps are polymer, not heteroatoms.

    Structures written by OpenMM or other modelling tools name histidines HID,
    HIE or HIN; biotite's CCD amino-acid filter does not know those names, so
    dropping "non-amino-acid" atoms with it alone would delete the residue.
    """

    @staticmethod
    def _renamed_two_slabs(tmp_path: Path, name_a: str, name_b: str) -> Path:
        slab_a = _slab(6, [0.0, -3.4])
        slab_b = _slab(6, [3.4, 6.8])
        coords = np.vstack([slab_a, slab_b])
        n = len(coords)
        arr = struc.AtomArray(n)
        arr.coord = coords.astype(np.float32)
        arr.chain_id = np.array(["A"] * len(slab_a) + ["B"] * len(slab_b))
        arr.res_id = np.arange(1, n + 1)
        arr.res_name = np.array([name_a] * len(slab_a) + [name_b] * len(slab_b))
        arr.atom_name = np.array(["C"] * n)
        arr.element = np.array(["C"] * n)
        pdb = pdb_io.PDBFile()
        pdb.set_structure(arr)
        path = tmp_path / f"{name_a}_{name_b}.pdb"
        pdb.write(str(path))
        return path

    @pytest.mark.parametrize(
        "variant_a, variant_b", [("HIE", "ALA"), ("ALA", "HID"), ("ACE", "NME")]
    )
    def test_variant_residues_give_the_same_result_as_alanine(self, tmp_path, variant_a, variant_b):
        plain = self._renamed_two_slabs(tmp_path, "ALA", "ALA")
        variant = self._renamed_two_slabs(tmp_path, variant_a, variant_b)
        kw = dict(peptide_chain="A", receptor_chain="B")
        sc = compute_shape_complementarity(variant, **kw)
        assert np.isfinite(sc["sc"])
        _assert_same(sc, compute_shape_complementarity(plain, **kw), SC_KEYS)
        void = compute_buried_void_volume(variant, grid_spacing=1.0, **kw)
        assert void["n_interface_atoms"] > 0
        _assert_same(void, compute_buried_void_volume(plain, grid_spacing=1.0, **kw), VOID_KEYS)


class TestHeteroCli:
    def _run(self, monkeypatch, capsys, *args):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-geometry", *map(str, args)])
        main()
        return capsys.readouterr().out

    def test_void_flag_switches_the_policy(self, monkeypatch, capsys):
        ignore = self._run(monkeypatch, capsys, "--input", CWA, "--metric", "void")
        keep = self._run(
            monkeypatch, capsys, "--input", CWA, "--metric", "void", "--hetero", "keep"
        )
        explicit = self._run(
            monkeypatch, capsys, "--input", CWA, "--metric", "void", "--hetero", "ignore"
        )
        assert ignore == explicit
        assert ignore != keep

    def test_sc_flag_switches_the_policy(self, monkeypatch, capsys):
        ignore = self._run(monkeypatch, capsys, "--input", CWA, "--metric", "sc")
        keep = self._run(monkeypatch, capsys, "--input", CWA, "--metric", "sc", "--hetero", "keep")
        assert ignore != keep

    def test_rejects_unknown_value(self, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, capsys, "--input", CWA, "--metric", "sc", "--hetero", "drop")
