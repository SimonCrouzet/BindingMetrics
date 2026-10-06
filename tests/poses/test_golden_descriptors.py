"""Golden descriptors on the public complex 1YCR, checked against a calculation from the file.

``tests/golden/pose_sets/descriptors_1ycr.json`` holds the pose-set result for the crystal
peptide of 1YCR (MDM2 chain A, p53 peptide chain B) and the same peptide moved 6 angstrom along
the unit vector (2/3, -1/3, 2/3), against the crystal complex as reference: two residue sets (the
22 receptor residues within 4.5 angstrom of the peptide and the 25 that it buries by at least 0.5
square angstrom) and the crystal peptide as partner. The values in the file come from the
committed code. This module recomputes the set overlap, the shared buried area and the contact
columns from the PDB file with code of its own, using only numpy and biotite's Shrake-Rupley
routine (960 points, 1.4 angstrom probe, element radii), and asserts the file agrees. The
comparison of the file with a live run is in ``test_golden_pose_set.py``.

Tolerance: counts are exact. Areas are sums of float32 numbers computed in a different order,
so they agree to 1e-5 relative (observed: below 1e-7).
"""

import math

import numpy as np
import pytest

from binding_metrics.testing.synthetic import bundled_complex_path
from tests.golden import pose_set_cases as golden

biotite_pdb = pytest.importorskip("biotite.structure.io.pdb")
struc = pytest.importorskip("biotite.structure")

from biotite.structure.info import vdw_radius_single  # noqa: E402

RELATIVE = 1e-5
DECOY_SHIFT_ANGSTROM = 6.0
DIRECTION = np.array([2.0, -1.0, 2.0]) / 3.0
PROBE_ANGSTROM = 1.4
POINTS_PER_ATOM = 960
INTERFACE_THRESHOLD_ANGSTROM2 = 0.5
CLASH_CUTOFFS = (2.2, 3.0)
SHELL_ANGSTROM = 4.0


# ---------------------------------------------------------------------------
# the calculation from the file
# ---------------------------------------------------------------------------


def radii_of(atoms):
    return np.array([vdw_radius_single(str(e).strip()) or 1.8 for e in atoms.element])


def sasa_per_atom(atoms):
    values = struc.sasa(
        atoms, probe_radius=PROBE_ANGSTROM, vdw_radii=radii_of(atoms), point_number=POINTS_PER_ATOM
    )
    return np.where(np.isfinite(values), values, 0.0)


def burial_of_receptor(receptor, binder):
    """Per receptor atom, the area ``binder`` covers: alone minus with it, floored at 0."""
    together = sasa_per_atom(receptor + binder)[: len(receptor)]
    return np.maximum(sasa_per_atom(receptor) - together, 0.0)


def per_residue(receptor, per_atom):
    totals = {}
    for number, value in zip(receptor.res_id.tolist(), per_atom.tolist()):
        totals[number] = totals.get(number, 0.0) + value
    return totals


def sphere_overlap_volumes(radius_a, radius_b, distance):
    """Volume shared by two spheres, element by element (closed form of the lens)."""
    radius_a, radius_b, distance = np.broadcast_arrays(radius_a, radius_b, distance)
    reach = radius_a + radius_b
    gap = np.abs(radius_a - radius_b)
    volume = np.zeros(distance.shape)
    inside = distance <= gap
    volume[inside] = 4.0 / 3.0 * math.pi * np.minimum(radius_a, radius_b)[inside] ** 3
    lens = (distance < reach) & ~inside
    d, big, r_a, r_b = distance[lens], reach[lens], radius_a[lens], radius_b[lens]
    volume[lens] = math.pi * (big - d) ** 2 * (d**2 + 2 * d * big - 3 * (r_a - r_b) ** 2) / (12 * d)
    return volume


@pytest.fixture(scope="module")
def complex_atoms():
    atoms = biotite_pdb.PDBFile.read(str(bundled_complex_path())).get_structure(model=1)
    keep = (~atoms.hetero) & (atoms.element != "H") & np.isin(atoms.chain_id, ["A", "B"])
    return atoms[keep]


@pytest.fixture(scope="module")
def receptor(complex_atoms):
    return complex_atoms[complex_atoms.chain_id == "A"]


@pytest.fixture(scope="module")
def crystal_peptide(complex_atoms):
    return complex_atoms[complex_atoms.chain_id == "B"]


def shifted(peptide, angstrom):
    moved = peptide.copy()
    moved.coord = (peptide.coord.astype(float) + angstrom * DIRECTION).astype(np.float32)
    return moved


@pytest.fixture(scope="module")
def peptides(crystal_peptide):
    return {
        "crystal-s1": crystal_peptide,
        "decoy_6A-s1": shifted(crystal_peptide, DECOY_SHIFT_ANGSTROM),
    }


@pytest.fixture(scope="module")
def partner_burial(receptor, crystal_peptide):
    return burial_of_receptor(receptor, crystal_peptide)


@pytest.fixture(scope="module")
def document():
    return golden.read_golden(golden.CASES["descriptors_1ycr"].path)


@pytest.fixture(scope="module")
def rows(document):
    return {row["id"]: row for row in document["result"]["poses"]}


def approx(value):
    return pytest.approx(value, rel=RELATIVE, abs=1e-9)


# ---------------------------------------------------------------------------
# what the sets are
# ---------------------------------------------------------------------------


class TestTheResidueSets:
    def test_the_epitope_is_the_receptor_residues_within_4_5_angstrom_of_the_peptide(
        self, receptor, crystal_peptide
    ):
        distances = np.linalg.norm(
            receptor.coord[:, None, :] - crystal_peptide.coord[None, :, :], axis=2
        )
        close = {int(n) for n in receptor.res_id[(distances <= 4.5).any(axis=1)]}
        assert sorted(close) == list(golden.EPITOPE_4_5_ANGSTROM)

    def test_the_interface_is_the_residues_that_the_peptide_buries_by_half_a_square_angstrom(
        self, receptor, partner_burial
    ):
        areas = per_residue(receptor, partner_burial)
        interface = sorted(n for n, area in areas.items() if area >= INTERFACE_THRESHOLD_ANGSTROM2)
        assert interface == list(golden.CRYSTAL_INTERFACE)

    def test_the_two_sets_overlap_in_22_residues(self):
        epitope, interface = set(golden.EPITOPE_4_5_ANGSTROM), set(golden.CRYSTAL_INTERFACE)
        assert len(epitope) == 22 and len(interface) == 25
        assert epitope <= interface  # every residue near the peptide is buried by it
        assert sorted(interface - epitope) == [49, 86, 103]


class TestThePoses:
    def test_the_decoy_is_the_crystal_peptide_translated_by_6_angstrom(
        self, peptides, crystal_peptide
    ):
        decoys = golden.decoy_poses()
        by_id = {pose.id: pose.atoms() for pose in decoys.poses}
        assert set(by_id) == set(peptides)
        for pose_id, peptide in peptides.items():
            atoms = by_id[pose_id]
            moved = atoms.coord[atoms.chain_id == "B"].astype(float)
            assert np.abs(moved - peptide.coord.astype(float)).max() < 1e-5
        shift = np.linalg.norm(peptides["decoy_6A-s1"].coord - crystal_peptide.coord, axis=1)
        assert shift == pytest.approx(DECOY_SHIFT_ANGSTROM, abs=1e-4)

    @pytest.mark.parametrize("kind", ["ca", "backbone", "heavy"])
    def test_the_two_poses_are_6_angstrom_apart_after_the_fit(self, document, kind):
        block = document["result"]["agreement"]["rmsd"][kind]["matrix"]
        assert block["pose_ids"] == ["crystal-s1", "decoy_6A-s1"]
        assert block["matrix"][0][1] == pytest.approx(DECOY_SHIFT_ANGSTROM, abs=1e-4)

    def test_both_poses_sit_in_the_frame_of_the_reference(self, rows):
        for row in rows.values():
            frame = row["frame"]
            assert row["status"] == "ok" and frame["pairing"] == "number"
            assert frame["receptor_rmsd_angstrom"] < 1e-5
            assert np.allclose(frame["transform"]["rotation"], np.eye(3), atol=1e-6)
            assert np.allclose(frame["transform"]["translation"], 0.0, atol=1e-5)

    def test_the_buried_area_of_the_metric_block_is_the_sum_of_the_three_areas(
        self, rows, receptor, peptides
    ):
        for pose_id, peptide in peptides.items():
            expected = (
                sasa_per_atom(receptor).sum()
                + sasa_per_atom(peptide).sum()
                - sasa_per_atom(receptor + peptide).sum()
            )
            block = rows[pose_id]["delta_sasa_static"]
            assert block["delta_sasa"] == approx(expected)
            assert block["sasa_complex"] == approx(sasa_per_atom(receptor + peptide).sum())


# ---------------------------------------------------------------------------
# residue sets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pose_id", ["crystal-s1", "decoy_6A-s1"])
@pytest.mark.parametrize("set_name", ["epitope", "crystal_interface"])
class TestSetOverlapByHand:
    def test_counts_fractions_and_area(self, rows, receptor, peptides, pose_id, set_name):
        members = {
            "epitope": set(golden.EPITOPE_4_5_ANGSTROM),
            "crystal_interface": set(golden.CRYSTAL_INTERFACE),
        }[set_name]
        burial = burial_of_receptor(receptor, peptides[pose_id])
        areas = per_residue(receptor, burial)
        interface = {n for n, area in areas.items() if area >= INTERFACE_THRESHOLD_ANGSTROM2}
        overlap = interface & members
        in_set_area = sum(areas[n] for n in members)

        block = rows[pose_id]["descriptors"]["reference"]["sets"][set_name]
        assert block["n_set"] == len(members) == block["n_set_mapped"]
        assert block["n_interface"] == len(interface)
        assert block["n_overlap"] == len(overlap)
        assert block["covered_fraction"] == approx(len(overlap) / len(members))
        assert block["interface_in_set_fraction"] == approx(len(overlap) / len(interface))
        assert block["jaccard"] == approx(len(overlap) / len(interface | members))
        assert block["receptor_buried_area_in_set_angstrom2"] == approx(in_set_area)
        assert block["receptor_buried_area_in_set_fraction"] == approx(in_set_area / burial.sum())

    def test_the_crystal_peptide_covers_its_own_set_and_the_decoy_covers_less(
        self, rows, pose_id, set_name
    ):
        block = rows[pose_id]["descriptors"]["reference"]["sets"][set_name]
        if pose_id == "crystal-s1":
            assert block["covered_fraction"] == 1.0 and block["n_overlap"] == block["n_set"]
        else:
            own = rows["crystal-s1"]["descriptors"]["reference"]["sets"][set_name]
            assert block["n_overlap"] < own["n_overlap"]
            assert (
                block["receptor_buried_area_in_set_angstrom2"]
                < (own["receptor_buried_area_in_set_angstrom2"])
            )


# ---------------------------------------------------------------------------
# partner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pose_id", ["crystal-s1", "decoy_6A-s1"])
class TestPartnerByHand:
    @pytest.fixture
    def block(self, rows, pose_id):
        return rows[pose_id]["descriptors"]["reference"]["partners"]["crystal_peptide"]

    def test_the_shared_buried_area_is_the_overlap_of_the_two_buried_surfaces(
        self, block, receptor, peptides, partner_burial, pose_id
    ):
        pose_burial = burial_of_receptor(receptor, peptides[pose_id])
        shared = np.minimum(pose_burial, partner_burial).sum()
        assert block["shared_buried_area_angstrom2"] == approx(shared)
        assert block["partner_buried_area_angstrom2"] == approx(partner_burial.sum())
        assert block["pose_buried_area_angstrom2"] == approx(pose_burial.sum())
        assert block["shared_fraction_of_partner"] == approx(shared / partner_burial.sum())

    def test_distances_clashes_shell_and_overlap_volume(
        self, block, peptides, crystal_peptide, pose_id
    ):
        binder = peptides[pose_id].coord.astype(float)
        partner = crystal_peptide.coord.astype(float)
        distance = np.linalg.norm(binder[:, None, :] - partner[None, :, :], axis=2)
        nearest = distance.min(axis=1)
        radii = radii_of(crystal_peptide)

        assert block["min_distance_angstrom"] == pytest.approx(distance.min(), abs=1e-4)
        for cutoff in CLASH_CUTOFFS:
            assert block["clash_atoms"][f"{cutoff}"] == int((nearest < cutoff).sum())
            assert block["clash_pairs"][f"{cutoff}"] == int((distance < cutoff).sum())
        assert block["shell_fraction"] == approx(float((nearest <= SHELL_ANGSTROM).mean()))
        volume = sphere_overlap_volumes(radii[:, None], radii[None, :], distance).sum()
        assert block["overlap_volume_angstrom3"] == approx(volume)


class TestTheCrystalPeptideAgainstItself:
    """The pose is the partner: everything the partner covers is shared."""

    @pytest.fixture
    def block(self, rows):
        return rows["crystal-s1"]["descriptors"]["reference"]["partners"]["crystal_peptide"]

    def test_the_whole_buried_area_is_shared(self, block):
        assert block["shared_buried_area_angstrom2"] == block["partner_buried_area_angstrom2"]
        assert block["shared_buried_area_angstrom2"] == block["pose_buried_area_angstrom2"]
        assert block["shared_fraction_of_partner"] == 1.0

    def test_every_atom_sits_on_its_partner_atom(self, block, crystal_peptide):
        n_atoms = len(crystal_peptide)
        assert block["min_distance_angstrom"] == 0.0
        assert block["clash_atoms"] == {"2.2": n_atoms, "3.0": n_atoms}
        assert block["shell_fraction"] == 1.0

    def test_the_decoy_shares_part_of_the_surface_and_pushes_into_the_receptor(self, rows):
        decoy = rows["decoy_6A-s1"]["descriptors"]["reference"]["partners"]["crystal_peptide"]
        assert 0.0 < decoy["shared_fraction_of_partner"] < 1.0
        assert decoy["shell_fraction"] < 1.0 and decoy["min_distance_angstrom"] > 0.0
        assert decoy["pose_buried_area_angstrom2"] < decoy["partner_buried_area_angstrom2"]
