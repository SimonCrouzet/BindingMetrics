"""Unit tests for the structural QC of relaxed structures (protocols/qc.py).

Every failure mode is built by damaging a real structure (1YCR) or a small
hand-made backbone, so the tests need no GPU and no relaxation run.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.protocols import qc
from binding_metrics.protocols.qc import AtomSnapshot

gemmi = pytest.importorskip("gemmi", reason="gemmi reads the example structure")

EXAMPLE_PDB = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


@pytest.fixture(scope="module")
def ycr() -> AtomSnapshot:
    return AtomSnapshot.from_file(EXAMPLE_PDB)


def _moved(snapshot: AtomSnapshot, coords: np.ndarray) -> AtomSnapshot:
    return dataclasses.replace(snapshot, coords=coords)


def _row(snapshot: AtomSnapshot, residue_key: tuple, atom_name: str) -> int:
    for i in np.nonzero(snapshot.heavy_mask)[0]:
        if snapshot.residue_keys[i] == residue_key and snapshot.atom_names[i] == atom_name:
            return int(i)
    raise KeyError((residue_key, atom_name))


def _first_residue_with_cb(snapshot: AtomSnapshot) -> tuple:
    for i in np.nonzero(snapshot.heavy_mask)[0]:
        if snapshot.atom_names[i] == "CB":
            return snapshot.residue_keys[i]
    raise AssertionError("no CB in example")


class TestIdenticalStructure:
    def test_all_seven_checks_pass_and_are_evaluated(self, ycr):
        result = qc.check_relaxed_structure(
            ycr, ycr, energy_kj_mol=-1000.0, energy_before_kj_mol=0.0
        )
        assert result["passed"] is True
        assert result["failed"] == []
        assert list(result["checks"]) == [
            "energy",
            "rmsd",
            "coordinates_finite",
            "min_heavy_distance",
            "bond_lengths",
            "chirality",
            "composition",
        ]
        for name, check in result["checks"].items():
            assert check["evaluated"], name
            assert check["passed"], name
        assert result["checks"]["rmsd"]["value"] < 1e-6
        # Backbone peptide bond C-N is about 1.33 A in a real structure.
        assert 1.0 < result["checks"]["min_heavy_distance"]["value"] < 1.6

    def test_md_mode_leaves_out_the_rmsd_check(self, ycr):
        result = qc.check_relaxed_structure(ycr, ycr, max_rmsd_angstrom=None)
        assert "rmsd" not in result["checks"]

    def test_energy_is_reported_as_not_evaluated_when_missing(self, ycr):
        check = qc.check_relaxed_structure(ycr, ycr)["checks"]["energy"]
        assert check["passed"] is True
        assert check["evaluated"] is False
        assert check["reason"] == "no energy available"


class TestDamagedStructures:
    def test_mirror_image_inverts_every_ca_centre(self, ycr):
        mirrored = _moved(ycr, ycr.coords * np.array([-1.0, 1.0, 1.0]))
        result = qc.check_relaxed_structure(ycr, mirrored, energy_kj_mol=-1000.0)
        assert "chirality" in result["failed"]
        assert result["checks"]["chirality"]["value"] == 94  # every Ca centre of 1YCR
        # Bond lengths and contacts are unchanged by a reflection.
        for name in ("bond_lengths", "min_heavy_distance", "coordinates_finite", "composition"):
            assert result["checks"][name]["passed"], name

    def test_single_reflected_side_chain_is_caught_alone(self, ycr):
        key = _first_residue_with_cb(ycr)
        n, ca, c = (_row(ycr, key, name) for name in ("N", "CA", "C"))
        normal = np.cross(ycr.coords[n] - ycr.coords[ca], ycr.coords[c] - ycr.coords[ca])
        normal /= np.linalg.norm(normal)
        side_chain = [
            i
            for i in np.nonzero(ycr.heavy_mask)[0]
            if ycr.residue_keys[i] == key and ycr.atom_names[i] not in ("N", "CA", "C", "O")
        ]
        coords = ycr.coords.copy()
        # Reflect the side chain through the N-CA-C plane: its internal bonds
        # keep their lengths but the C-alpha centre changes handedness.
        offsets = coords[side_chain] - coords[ca]
        coords[side_chain] -= 2.0 * (offsets @ normal)[:, None] * normal
        damaged = _moved(ycr, coords)
        check = qc.check_chirality(ycr, damaged)
        assert check["passed"] is False
        assert check["value"] == 1
        assert qc.check_bond_lengths(ycr, damaged)["passed"]

    def test_exploded_structure_fails_rmsd_and_bonds(self, ycr):
        centre = ycr.coords.mean(axis=0)
        exploded = _moved(ycr, centre + (ycr.coords - centre) * 3.0)
        result = qc.check_relaxed_structure(ycr, exploded)
        assert {"rmsd", "bond_lengths"} <= set(result["failed"])
        assert result["checks"]["bond_lengths"]["value"] > qc.BOND_LENGTH_MAX_ANGSTROM

    def test_fused_atoms_fail_the_distance_check(self, ycr):
        heavy = np.nonzero(ycr.heavy_mask)[0]
        first, last = heavy[5], heavy[-5]
        assert ycr.residue_keys[first] != ycr.residue_keys[last]
        coords = ycr.coords.copy()
        coords[first] = coords[last]
        check = qc.check_min_heavy_distance(_moved(ycr, coords))
        assert check["passed"] is False
        assert check["value"] == pytest.approx(0.0, abs=1e-9)

    def test_nan_coordinate_is_flagged_and_does_not_break_other_checks(self, ycr):
        coords = ycr.coords.copy()
        coords[10] = np.nan
        result = qc.check_relaxed_structure(ycr, _moved(ycr, coords))
        assert "coordinates_finite" in result["failed"]
        assert result["checks"]["coordinates_finite"]["value"] == 1
        assert result["checks"]["min_heavy_distance"]["evaluated"] is False
        assert "rmsd" in result["failed"]

    def test_dropped_atom_fails_the_composition_check(self, ycr):
        heavy = np.nonzero(ycr.heavy_mask)[0]
        drop = heavy[40]
        keep = np.array([i for i in range(len(ycr.coords)) if i != drop])
        smaller = AtomSnapshot(
            coords=ycr.coords[keep],
            atom_names=tuple(ycr.atom_names[i] for i in keep),
            residue_keys=tuple(ycr.residue_keys[i] for i in keep),
            is_hydrogen=ycr.is_hydrogen[keep],
            is_water=ycr.is_water[keep],
        )
        check = qc.check_composition(ycr, smaller)
        assert check["passed"] is False
        assert check["value"] == -1
        assert "changed" in check["detail"]


class TestEnergy:
    @pytest.mark.parametrize("energy", [math.nan, math.inf, -math.inf, 5.0e6, -2.0e8])
    def test_out_of_range_energy_fails(self, energy):
        assert qc.check_energy(energy)["passed"] is False

    def test_energy_above_the_starting_energy_fails(self):
        assert qc.check_energy(-100.0, energy_before_kj_mol=-500.0)["passed"] is False

    def test_energy_within_tolerance_of_the_start_passes(self):
        tolerance = qc.ENERGY_INCREASE_TOLERANCE_KJ_MOL
        assert qc.check_energy(-500.0 + 0.5 * tolerance, energy_before_kj_mol=-500.0)["passed"]

    def test_lower_energy_passes(self):
        assert qc.check_energy(-14460.0, energy_before_kj_mol=5082.0)["passed"]


class TestKabschOrientation:
    def test_rotated_and_translated_copy_has_zero_rmsd(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(50, 3)) * 5.0
        angle = np.deg2rad(60.0)
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
        moved = points @ rotation.T + np.array([3.0, -2.0, 1.0])
        assert qc._kabsch_rmsd(points, moved) < 1e-9


def _three_alanines():
    """Topology and positions (nm) of three ALA backbones plus one water.

    Each residue has N, CA, C and CB placed so the C-alpha volume is about
    -2.6 A^3.
    """
    from openmm import Vec3, app
    from openmm import unit as u
    from openmm.app import element

    template = {
        "N": (-1.2, 0.8, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (1.4, 0.5, 0.2),
        "CB": (-0.3, -0.6, 1.4),
    }
    symbols = {
        "N": element.nitrogen,
        "CA": element.carbon,
        "C": element.carbon,
        "CB": element.carbon,
    }
    topology = app.Topology()
    chain = topology.addChain("A")
    coords_angstrom = []
    for i in range(3):
        residue = topology.addResidue("ALA", chain)
        for name, (x, y, z) in template.items():
            topology.addAtom(name, symbols[name], residue)
            coords_angstrom.append((x + 4.5 * i, y, z))
    water = topology.addResidue("HOH", topology.addChain("W"))
    topology.addAtom("O", element.oxygen, water)
    coords_angstrom.append((30.0, 30.0, 30.0))
    positions = u.Quantity([Vec3(*(np.array(c) / 10.0)) for c in coords_angstrom], u.nanometer)
    return topology, positions, np.array(coords_angstrom)


@pytest.mark.integration
class TestFromTopology:
    def test_coordinates_are_in_angstrom_and_water_is_marked(self):
        topology, positions, coords = _three_alanines()
        snap = AtomSnapshot.from_topology(topology, positions)
        assert np.allclose(snap.coords, coords)
        assert snap.is_water[-1] and not snap.is_water[:-1].any()
        assert not snap.is_hydrogen.any()
        assert snap.heavy_mask.sum() == 12

    def test_position_count_mismatch_raises(self):
        topology, positions, _ = _three_alanines()
        with pytest.raises(ValueError, match="atoms"):
            AtomSnapshot.from_topology(topology, positions[:-1])

    def test_identical_snapshots_pass_and_a_mirror_image_fails_chirality(self):
        topology, positions, coords = _three_alanines()
        before = AtomSnapshot.from_topology(topology, positions)
        assert qc.check_relaxed_structure(before, before)["passed"]
        mirrored = dataclasses.replace(before, coords=coords * np.array([1.0, 1.0, -1.0]))
        check = qc.check_chirality(before, mirrored)
        assert check["passed"] is False and check["value"] == 3


class TestFromFile:
    def test_paths_are_accepted_and_a_mirrored_file_fails_chirality(self, tmp_path):
        structure = gemmi.read_structure(str(EXAMPLE_PDB))
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.pos = gemmi.Position(-atom.pos.x, atom.pos.y, atom.pos.z)
        mirrored = tmp_path / "mirrored.pdb"
        structure.write_pdb(str(mirrored))

        assert qc.check_relaxed_structure(EXAMPLE_PDB, EXAMPLE_PDB)["passed"]
        result = qc.check_relaxed_structure(EXAMPLE_PDB, mirrored)
        assert "chirality" in result["failed"]
        assert "composition" not in result["failed"]
