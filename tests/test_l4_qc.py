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
pytest.importorskip("biotite", reason="biotite supplies the residue templates of a file's bonds")

EXAMPLE_PDB = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
CYCLIC_CIF = Path(__file__).parent.parent / "data" / "example_bicyclic_sfti1_3P8F.cif"


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


def _nonbonded_pair(snapshot: AtomSnapshot, same_residue: bool) -> tuple:
    """Two heavy atoms 3-4.5 A apart that share no bond, in one residue or in two."""
    bonded = set(snapshot.bonds)
    heavy = np.nonzero(snapshot.heavy_mask)[0]
    for i in heavy[10:]:
        for j in heavy[heavy > i]:
            if (snapshot.residue_keys[i] == snapshot.residue_keys[j]) != same_residue:
                continue
            distance = np.linalg.norm(snapshot.coords[i] - snapshot.coords[j])
            if 3.0 < distance < 4.5 and (int(i), int(j)) not in bonded:
                return int(i), int(j)
    raise AssertionError("no non-bonded pair found")


def _bond_between(snapshot: AtomSnapshot, same_residue: bool) -> tuple:
    """A bond of the snapshot inside one residue or between two."""
    for i, j in snapshot.bonds[len(snapshot.bonds) // 3 :]:
        if (snapshot.residue_keys[i] == snapshot.residue_keys[j]) == same_residue:
            return i, j
    raise AssertionError("no such bond")


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


class TestBondLengthsUseTheBondList:
    """The bonds come from the topology or the residue templates, not from distances.

    PDBFixer rebuilds missing atoms with little regard for their neighbours, so
    a prepared input holds clashing atoms 1.5-2.1 A apart that are not bonded.
    Perceiving bonds from those distances flagged the relaxation for resolving
    the clash (ARG497 CD to ARG509 CG, 1.82 A before and 5.91 A after in 4KRL).
    """

    @pytest.mark.parametrize("same_residue", [False, True], ids=["between_residues", "in_residue"])
    def test_a_clash_resolved_by_relaxation_is_not_a_stretched_bond(self, ycr, same_residue):
        i, j = _nonbonded_pair(ycr, same_residue)
        direction = ycr.coords[j] - ycr.coords[i]
        coords = ycr.coords.copy()
        coords[j] = ycr.coords[i] + 1.8 * direction / np.linalg.norm(direction)
        clashing_input = _moved(ycr, coords)
        assert (i, j) not in set(ycr.bonds)

        check = qc.check_bond_lengths(clashing_input, ycr)

        assert check["evaluated"] and check["passed"], check["detail"]
        assert check["value"] < 2.0  # the longest real bond, a C-S bond, is 1.82 A

    @pytest.mark.parametrize("same_residue", [True, False], ids=["in_residue", "peptide"])
    def test_a_genuinely_stretched_bond_still_fails(self, ycr, same_residue):
        i, j = _bond_between(ycr, same_residue)
        direction = ycr.coords[j] - ycr.coords[i]
        coords = ycr.coords.copy()
        coords[j] = ycr.coords[i] + 3.2 * direction / np.linalg.norm(direction)

        check = qc.check_bond_lengths(ycr, _moved(ycr, coords))

        assert check["passed"] is False
        assert check["value"] == pytest.approx(3.2, abs=0.01)
        assert f"{ycr.atom_names[i]}" in check["detail"] and "1 of" in check["detail"]

    def test_a_bond_already_stretched_in_the_input_is_reported_but_not_failed(self, ycr):
        i, j = _bond_between(ycr, same_residue=False)
        direction = ycr.coords[j] - ycr.coords[i]
        coords = ycr.coords.copy()
        coords[j] = ycr.coords[i] + 7.4 * direction / np.linalg.norm(direction)

        check = qc.check_bond_lengths(_moved(ycr, coords), ycr)

        assert check["passed"] is True
        assert "already longer" in check["detail"] and "7.40" in check["detail"]

    def test_file_bonds_are_chemically_sensible(self, ycr):
        lengths = [np.linalg.norm(ycr.coords[i] - ycr.coords[j]) for i, j in ycr.bonds]
        assert len(ycr.bonds) > 800
        assert 1.1 < min(lengths) and max(lengths) < 2.0  # crystal geometry, no S-S in 1YCR

    def test_a_chain_break_is_not_a_bond(self, tmp_path):
        structure = gemmi.read_structure(str(EXAMPLE_PDB))
        chain = structure[0][0]
        for residue in list(chain)[40:]:
            for atom in residue:
                atom.pos = gemmi.Position(atom.pos.x + 10.0, atom.pos.y, atom.pos.z)
        broken = tmp_path / "broken.pdb"
        structure.write_pdb(str(broken))

        whole, split = AtomSnapshot.from_file(EXAMPLE_PDB), AtomSnapshot.from_file(broken)

        assert len(whole.bonds) - len(split.bonds) == 1  # the C-N link across the break

    def test_head_to_tail_bond_of_a_cyclic_peptide_is_listed(self):
        snapshot = AtomSnapshot.from_file(CYCLIC_CIF)
        closing = [
            (i, j)
            for i, j in snapshot.bonds
            if {snapshot.atom_names[i], snapshot.atom_names[j]} == {"C", "N"}
            and abs(snapshot.residue_keys[i][1] - snapshot.residue_keys[j][1]) > 5
        ]
        assert len(closing) == 1

    def test_protonation_variant_names_get_the_bonds_of_their_parent(self, ycr, tmp_path):
        """HIE, CYX and the like are ligands in the dictionary, not variants."""
        variants = {"HIS": "HIE", "CYS": "CYX", "LYS": "LYN", "ASP": "ASH", "GLU": "GLH"}
        structure = gemmi.read_structure(str(EXAMPLE_PDB))
        renamed = 0
        for residue in structure[0][0]:
            if residue.name in variants:
                residue.name = variants[residue.name]
                renamed += 1
        assert renamed > 10
        variant_file = tmp_path / "variants.pdb"
        structure.write_pdb(str(variant_file))

        assert AtomSnapshot.from_file(variant_file).bonds == ycr.bonds

    def test_a_snapshot_without_bonds_is_not_evaluated(self, ycr):
        check = qc.check_bond_lengths(dataclasses.replace(ycr, bonds=None), ycr)
        assert check["evaluated"] is False and check["passed"] is True
        assert "no bonds" in check["reason"]


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


@pytest.mark.integration
class TestTopologyBonds:
    def _bonded_alanines(self):
        """The three backbones of ``_three_alanines`` with their bonds added."""
        topology, positions, coords = _three_alanines()
        residues = list(topology.residues())[:3]
        atoms = [{a.name: a for a in residue.atoms()} for residue in residues]
        for k, by_name in enumerate(atoms):
            for name in ("N", "C", "CB"):
                topology.addBond(by_name[name], by_name["CA"])
            if k:
                topology.addBond(atoms[k - 1]["C"], by_name["N"])
        return topology, positions, coords

    def test_bonds_are_the_heavy_heavy_topology_bonds(self):
        topology, positions, _ = self._bonded_alanines()
        snapshot = AtomSnapshot.from_topology(topology, positions)
        assert len(snapshot.bonds) == 3 * 3 + 2  # N, C, CB to CA per residue, two peptide links
        assert all(snapshot.heavy_mask[i] and snapshot.heavy_mask[j] for i, j in snapshot.bonds)

    def test_a_clash_between_residues_is_not_a_bond_but_a_broken_bond_still_fails(self):
        topology, positions, coords = self._bonded_alanines()
        cb_first, cb_second = 3, 7  # CB of residue 0 and of residue 1, in atom order
        clash = coords.copy()
        clash[cb_first] = coords[cb_second] + np.array([1.7, 0.0, 0.0])
        before = AtomSnapshot.from_topology(topology, clash / 10.0)
        relaxed = AtomSnapshot.from_topology(topology, coords / 10.0)

        assert qc.check_bond_lengths(before, relaxed)["passed"]

        torn = coords.copy()
        torn[cb_first] += np.array([0.0, 0.0, 3.0])  # CA-CB of residue 0 now 4.5 A
        result = qc.check_bond_lengths(relaxed, dataclasses.replace(relaxed, coords=torn))
        assert result["passed"] is False


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
