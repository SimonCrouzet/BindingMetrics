"""Void volume with a real solvent probe (`probe_radius`).

The synthetic system is a dense carbon lattice (spacing 1.5 A, van der Waals
radius 1.7 A, so no gap between neighbours is open to any probe) with a
spherical hole. Chain A holds the lattice atoms with z < 0 and chain B the
rest, so the hole is a cavity that only the two chains together close off.
The expected volumes come from an independent brute-force computation on a
0.2 A grid with a KD-tree, not from the grid code under test.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("scipy")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from binding_metrics.metrics.geometry import compute_buried_void_volume  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"

VDW_CARBON = 1.7
HOLE_RADIUS = 5.0
HALF_WIDTH = 9.0
SPACING = 1.5


def _lattice(hole_center, hole_radius: float, half: float) -> np.ndarray:
    """Cubic carbon lattice of half-width ``half`` minus a ball around ``hole_center``."""
    axis = np.arange(-half, half + 1e-9, SPACING)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return grid[np.linalg.norm(grid - hole_center, axis=1) >= hole_radius]


def _write_pdb(path: Path, coords: np.ndarray, chain_ids) -> Path:
    arr = struc.AtomArray(len(coords))
    arr.coord = coords.astype(np.float32)
    arr.chain_id = np.asarray(chain_ids)
    arr.res_id = np.arange(1, len(coords) + 1)
    arr.res_name = np.array(["ALA"] * len(coords))
    arr.atom_name = np.array(["C"] * len(coords))
    arr.element = np.array(["C"] * len(coords))
    pdb = pdb_io.PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(path))
    return path


def _brute_force_cavity_volume(
    atoms: np.ndarray, hole_center, hole_radius: float, probe: float, cell: float = 0.2
) -> float:
    """Empty volume of the hole reachable by the probe body from a probe-centre seed.

    Fine-grid reference: seeds are the points at least ``vdw + probe`` from
    every atom; the volume is the set of empty points within ``probe`` of a
    seed. ``probe=0`` gives the plain empty volume of the hole.
    """
    axis = np.arange(-hole_radius, hole_radius, cell) + cell / 2
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    points = points[np.linalg.norm(points, axis=1) < hole_radius] + hole_center
    dist_atom, _ = cKDTree(atoms).query(points)
    empty = dist_atom >= VDW_CARBON
    if probe == 0:
        return float(empty.sum() * cell**3)
    seeds = points[dist_atom >= VDW_CARBON + probe]
    if len(seeds) == 0:
        return 0.0
    dist_seed, _ = cKDTree(seeds).query(points)
    return float(((dist_seed <= probe) & empty).sum() * cell**3)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    """Lattice with a central hole, split into chain A (z < 0) and chain B."""
    center = np.zeros(3)
    atoms = _lattice(center, HOLE_RADIUS, HALF_WIDTH)
    chains = np.where(atoms[:, 2] < 0, "A", "B")
    path = _write_pdb(tmp_path_factory.mktemp("cavity") / "cavity.pdb", atoms, chains)
    return path, atoms


def _void(path, **kwargs) -> dict:
    return compute_buried_void_volume(path, peptide_chain="A", receptor_chain="B", **kwargs)


@pytest.fixture(scope="module")
def results_by_probe(cavity):
    """Void result dict of the synthetic cavity for each probe radius (computed once)."""
    path, _ = cavity
    return {p: _void(path, probe_radius=p) for p in (0.0, 0.5, 1.4, 2.0, 3.0, 6.0)}


class TestSyntheticCavity:
    @pytest.mark.parametrize("probe", [0.0, 1.4, 2.0])
    def test_volume_matches_brute_force_reference(self, cavity, results_by_probe, probe):
        _, atoms = cavity
        expected = _brute_force_cavity_volume(atoms, np.zeros(3), HOLE_RADIUS, probe)
        assert expected > 100.0  # the reference cavity is not degenerate
        assert results_by_probe[probe]["void_volume_A3"] == pytest.approx(expected, rel=0.10)

    def test_default_probe_is_a_water_probe(self, cavity, results_by_probe):
        path, _ = cavity
        assert _void(path) == results_by_probe[1.4]

    def test_probe_larger_than_the_cavity_finds_nothing(self, results_by_probe):
        """A probe of radius 6 A does not fit into a hole with ~3.3 A of empty radius."""
        assert results_by_probe[6.0]["void_volume_A3"] == 0.0

    def test_probe_radius_changes_the_result(self, results_by_probe):
        volumes = [results_by_probe[p]["void_volume_A3"] for p in (0.5, 1.4, 3.0, 6.0)]
        assert volumes[0] > volumes[1] > volumes[2] > volumes[3]

    def test_box_volume_does_not_depend_on_the_probe(self, results_by_probe):
        """The analysis box is set by the interface atoms and the padding only."""
        assert len({r["interface_box_volume_A3"] for r in results_by_probe.values()}) == 1
        assert len({r["n_interface_atoms"] for r in results_by_probe.values()}) == 1

    def test_void_fraction_is_voxels_over_box_voxels(self, results_by_probe):
        result = results_by_probe[1.4]
        assert result["void_grid_fraction"] == pytest.approx(
            result["void_volume_A3"] / result["interface_box_volume_A3"]
        )
        assert result["void_volume_A3"] % 0.5**3 == pytest.approx(0.0, abs=1e-9)

    def test_negative_probe_is_rejected(self, cavity):
        path, _ = cavity
        with pytest.raises(ValueError, match="probe_radius"):
            _void(path, probe_radius=-1.0)


class TestCavityNeedsBothChains:
    def test_cavity_walled_by_one_chain_is_not_an_interface_void(self, tmp_path):
        """The same hole inside chain A, with chain B a small cap on the outside, is not counted.

        Removing chain A trivially opens a cavity that chain A alone encloses,
        so that space says nothing about the interface.
        """
        center = np.array([0.0, 0.0, HALF_WIDTH - 6.0])
        lattice = _lattice(center, 4.0, HALF_WIDTH)
        cap_axis = np.array([-1.5, 0.0, 1.5])
        cap = np.array([[x, y, HALF_WIDTH + SPACING] for x in cap_axis for y in cap_axis])
        coords = np.vstack([lattice, cap])
        chains = ["A"] * len(lattice) + ["B"] * len(cap)
        path = _write_pdb(tmp_path / "one_chain_cavity.pdb", coords, chains)

        result = _void(path, probe_radius=1.4)
        assert result["n_interface_atoms"] > len(cap)  # the cap really touches the lattice
        assert result["void_volume_A3"] == 0.0

        # the same hole is a large cavity for the reference computation
        assert _brute_force_cavity_volume(lattice, center, 4.0, 1.4) > 50.0


@pytest.fixture(scope="module")
def by_probe():
    if not P53_MDM2.exists():
        pytest.skip("1YCR example not bundled")
    return {p: compute_buried_void_volume(P53_MDM2, probe_radius=p) for p in (0.5, 1.4, 3.0)}


class TestRealComplex:
    """1YCR (p53 peptide bound to MDM2); the old probe-free value was 0.25 A^3 for every probe."""

    def test_probe_changes_the_value(self, by_probe):
        volumes = [r["void_volume_A3"] for r in by_probe.values()]
        assert len(set(volumes)) == 3

    def test_default_probe_value_is_finite_and_in_the_expected_range(self, by_probe):
        volume = by_probe[1.4]["void_volume_A3"]
        assert np.isfinite(volume)
        assert 10.0 < volume < 150.0

    def test_box_and_interface_atoms_do_not_depend_on_the_probe(self, by_probe):
        assert len({r["interface_box_volume_A3"] for r in by_probe.values()}) == 1
        assert len({r["n_interface_atoms"] for r in by_probe.values()}) == 1
