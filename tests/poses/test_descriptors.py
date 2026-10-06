"""Reference-site descriptors: residue-set overlap, steric exclusion against a partner, and the
per-residue interface table.

The constructed complex has one atom per residue and residues 10 angstrom apart, so every
buried area is a sphere cap that can be computed by hand: a receptor carbon (radius 1.7) and a
binder carbon at distance d bury a cap of height 3.1 - d / 2 on each accessible sphere (radius
1.7 + 1.4 = 3.1 angstrom), of area 2 pi 3.1 (3.1 - d / 2). Residues farther than 6.2 angstrom
from every binder atom have exactly zero buried area.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.poses.descriptors import (
    DEFAULT_CLASH_CUTOFFS_ANGSTROM,
    DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2,
    DEFAULT_MAX_UNMAPPED_FRACTION,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_SHELL_ANGSTROM,
    PARTNER_COLUMNS,
    SET_COLUMNS,
    InterfaceAreas,
    interface_areas,
    partner_descriptors,
    residue_set_descriptors,
    sphere_overlap_volume,
)
from binding_metrics.poses.frame import (
    FrameOptions,
    place_in_reference_frame,
    renumber_to_reference,
    superpose_receptor,
)
from binding_metrics.poses.reference import Partner, Reference, ResidueKey

struc = pytest.importorskip("biotite.structure")

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"

# Accessible-sphere radius of a carbon (1.7 + 1.4 probe) and the cap that a binder atom at
# height 4 above a receptor atom takes from it.
SPHERE_RADIUS = 3.1
HEIGHT = 4.0
CAP_AREA = 2 * math.pi * SPHERE_RADIUS * (SPHERE_RADIUS - HEIGHT / 2)

RESIDUE_NAMES = ("ALA", "GLY", "LEU", "VAL", "SER", "THR", "ASP", "GLU", "LYS", "ARG", "PHE", "TYR")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load(path):
    from binding_metrics.metrics._common import load_structure

    return load_structure(path)


def tiny_complex(contacts=(1, 2, 6), height=HEIGHT, numbering=None, binder_xyz=None):
    """Twelve receptor residues on a 4 x 3 grid (10 angstrom apart, z = 0), chain A, and one
    binder atom (chain B) at ``height`` above each receptor residue of ``contacts``.

    ``numbering`` maps the residue index 1..12 to ``(number, icode)``; default ``(index, "")``.
    ``binder_xyz`` places the binder atoms at these coordinates instead.
    Every residue is one carbon named CA with an amino-acid name, so the residue is a polymer
    residue and the sequence is not periodic.
    """
    numbering = numbering or {i: (i, "") for i in range(1, 13)}
    grid = {i: np.array([((i - 1) % 4) * 10.0, ((i - 1) // 4) * 10.0, 0.0]) for i in range(1, 13)}
    rows = []
    for i in range(1, 13):
        number, icode = numbering[i]
        rows.append(("A", number, icode, RESIDUE_NAMES[i - 1], grid[i]))
    if binder_xyz is None:
        binder_xyz = [grid[i] + np.array([0.0, 0.0, height]) for i in contacts]
    for position, xyz in enumerate(binder_xyz, start=1):
        rows.append(("B", position, "", "ALA", np.asarray(xyz, dtype=float)))
    atoms = struc.AtomArray(len(rows))
    for row, (chain, number, icode, name, xyz) in enumerate(rows):
        atoms.chain_id[row] = chain
        atoms.res_id[row] = number
        atoms.ins_code[row] = icode
        atoms.res_name[row] = name
        atoms.atom_name[row] = "CA"
        atoms.element[row] = "C"
        atoms.coord[row] = xyz
    return atoms


def random_rigid(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, rng.normal(size=3) * 15.0


def moved(atoms, seed=3):
    rotation, translation = random_rigid(seed)
    out = atoms.copy()
    out.coord = (atoms.coord.astype(np.float64) @ rotation.T + translation).astype(np.float32)
    return out


def without(atoms, chain, numbers):
    remove = (atoms.chain_id == chain) & np.isin(atoms.res_id, list(numbers))
    return atoms[~remove]


def keys(*numbers, chain="A"):
    return [ResidueKey(chain, n) for n in numbers]


def place(pose_atoms, reference, **frame_options):
    """The frame of a pose against a reference, and the pose in the reference frame."""
    frame = superpose_receptor(pose_atoms, reference, options=FrameOptions(**frame_options))
    assert frame.ok, frame.reason
    return frame, place_in_reference_frame(pose_atoms, frame)


SETS = {
    "s_a": keys(2, 3, 6, 7),
    "s_b": keys(6, 7, 8),
    "contacts": keys(1, 2, 6),
    "far": keys(9, 10, 11, 12),
}


@pytest.fixture(scope="module")
def reference_atoms():
    return tiny_complex()


@pytest.fixture(scope="module")
def reference(reference_atoms):
    return Reference(reference_atoms, "A", "B", residue_sets=SETS)


@pytest.fixture(scope="module")
def pose(reference_atoms, reference):
    """The reference complex in an arbitrary frame, placed back on the reference."""
    return place(moved(reference_atoms), reference)


def describe(placed, frame, reference, **kwargs):
    return residue_set_descriptors(
        placed, frame, reference, binder_chain="B", target_chain="A", **kwargs
    )


# ---------------------------------------------------------------------------
# interface areas
# ---------------------------------------------------------------------------


class TestInterfaceAreas:
    def test_buried_area_is_the_sphere_cap_and_zero_elsewhere(self, reference_atoms):
        areas = interface_areas(reference_atoms, "B", "A")
        assert isinstance(areas, InterfaceAreas)
        assert areas.buried_receptor[[0, 1, 5]] == pytest.approx(CAP_AREA, rel=0.03)
        far = np.delete(areas.buried_receptor, [0, 1, 5])
        assert np.all(far == 0.0)
        assert areas.buried_binder == pytest.approx(CAP_AREA, rel=0.03)

    def test_residue_lists_follow_the_atoms(self, reference_atoms):
        areas = interface_areas(reference_atoms, "B", "A")
        assert [row["res_id"] for row in areas.receptor_residues] == list(range(1, 13))
        assert [row["res_id"] for row in areas.binder_residues] == [1, 2, 3]
        assert sum(row["buried_sasa"] for row in areas.receptor_residues) == pytest.approx(
            areas.buried_receptor.sum()
        )

    def test_sums_equal_compute_interface_metrics(self):
        from binding_metrics.metrics.interface import compute_interface_metrics

        atoms = _load(YCR)
        areas = interface_areas(atoms, "B", "A")
        result = compute_interface_metrics(YCR, binder_chain="B", target_chain="A")
        total = float(areas.buried_binder.sum() + areas.buried_receptor.sum())
        assert total == pytest.approx(result["delta_sasa"], rel=1e-9)

    def test_empty_chain_raises(self, reference_atoms):
        with pytest.raises(ValueError, match=r"no heavy polymer atom left for chain\(s\) \['Z'\]"):
            interface_areas(reference_atoms, "Z", "A")

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"binder_chain": "A", "target_chain": "A"}, "both 'A'"),
            ({"binder_chain": "", "target_chain": "A"}, "chain ID"),
            ({"binder_chain": "B", "target_chain": "A", "probe_radius": 0.0}, "probe_radius"),
            ({"binder_chain": "B", "target_chain": "A", "probe_radius": float("nan")}, "finite"),
        ],
    )
    def test_invalid_arguments(self, reference_atoms, kwargs, message):
        with pytest.raises(ValueError, match=message):
            interface_areas(reference_atoms, **kwargs)

    def test_stack_is_refused(self, reference_atoms):
        with pytest.raises(ValueError, match="one model"):
            interface_areas(struc.stack([reference_atoms, reference_atoms]), "B", "A")


# ---------------------------------------------------------------------------
# B1: residue sets
# ---------------------------------------------------------------------------


class TestResidueSets:
    def test_defaults_are_the_documented_ones(self):
        assert DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2 == 0.5
        assert DEFAULT_MAX_UNMAPPED_FRACTION == 0.5
        assert DEFAULT_PROBE_RADIUS == 1.4

    def test_hand_computed_overlap(self, reference, pose):
        # The interface is residues 1, 2 and 6. Set s_a = {2, 3, 6, 7}: two of its four
        # residues are in the interface, which has three residues; |I u S| = |{1,2,3,6,7}| = 5.
        frame, placed = pose
        block = describe(placed, frame, reference)["s_a"]
        assert list(block) == list(SET_COLUMNS)
        assert block["n_set"] == 4
        assert block["n_set_mapped"] == 4
        assert block["n_interface"] == 3
        assert block["n_overlap"] == 2
        assert block["covered_fraction"] == pytest.approx(2 / 4)
        assert block["interface_in_set_fraction"] == pytest.approx(2 / 3)
        assert block["jaccard"] == pytest.approx(2 / 5)
        assert block["receptor_buried_area_in_set_angstrom2"] == pytest.approx(
            2 * CAP_AREA, rel=0.03
        )
        assert block["receptor_buried_area_in_set_fraction"] == pytest.approx(2 / 3, rel=0.03)
        assert "reason" not in block
        assert "column_reasons" not in block

    def test_blocks_for_every_set_in_reference_order(self, reference, pose):
        frame, placed = pose
        blocks = describe(placed, frame, reference)
        assert list(blocks) == list(reference.residue_sets)
        contacts = blocks["contacts"]
        assert (contacts["n_overlap"], contacts["covered_fraction"]) == (3, 1.0)
        assert contacts["interface_in_set_fraction"] == 1.0
        assert contacts["jaccard"] == 1.0
        assert contacts["receptor_buried_area_in_set_fraction"] == pytest.approx(1.0)
        far = blocks["far"]
        assert (far["n_overlap"], far["covered_fraction"], far["jaccard"]) == (0, 0.0, 0.0)
        assert far["receptor_buried_area_in_set_angstrom2"] == 0.0
        assert far["receptor_buried_area_in_set_fraction"] == 0.0

    def test_a_set_that_does_not_move_with_the_frame(self, reference, reference_atoms):
        # The descriptors read the atoms in the reference frame, so any frame gives one answer.
        results = []
        for seed in (1, 2):
            frame, placed = place(moved(reference_atoms, seed), reference)
            results.append(describe(placed, frame, reference)["s_a"])
        for column in ("n_overlap", "n_interface", "jaccard", "covered_fraction"):
            assert results[0][column] == pytest.approx(results[1][column])

    def test_trimmed_pose_counts_only_mapped_residues(self, reference, reference_atoms):
        # Residues 6 and 7 are missing: the interface is {1, 2}; s_a keeps {2, 3}, exactly half
        # of the set is unmapped, which is not more than max_unmapped_fraction.
        trimmed = without(moved(reference_atoms), "A", (6, 7))
        frame, placed = place(trimmed, reference)
        block = describe(placed, frame, reference)["s_a"]
        assert block["n_set"] == 4
        assert block["n_set_mapped"] == 2
        assert block["n_interface"] == 2
        assert block["n_overlap"] == 1
        assert block["covered_fraction"] == pytest.approx(1 / 2)
        assert block["interface_in_set_fraction"] == pytest.approx(1 / 2)
        assert block["jaccard"] == pytest.approx(1 / 3)
        assert block["receptor_buried_area_in_set_angstrom2"] == pytest.approx(CAP_AREA, rel=0.03)
        assert block["receptor_buried_area_in_set_fraction"] == pytest.approx(1 / 2, rel=0.05)

    def test_set_lacking_too_much_has_no_value_and_a_reason(self, reference, reference_atoms):
        trimmed = without(moved(reference_atoms), "A", (6, 7))
        frame, placed = place(trimmed, reference)
        block = describe(placed, frame, reference)["s_b"]  # {6, 7, 8}: 2 of 3 are not in the pose
        assert block["reason"].startswith("3 set residues, 2 not in the pose")
        assert (block["n_set"], block["n_set_mapped"]) == (3, 1)
        assert all(block[c] is None for c in SET_COLUMNS if c not in ("n_set", "n_set_mapped"))
        assert list(block) == [*SET_COLUMNS, "reason"]

    def test_max_unmapped_fraction_is_a_parameter(self, reference, reference_atoms):
        trimmed = without(moved(reference_atoms), "A", (6, 7))
        frame, placed = place(trimmed, reference)
        strict = describe(placed, frame, reference, max_unmapped_fraction=0.4)["s_a"]
        assert strict["reason"].startswith("4 set residues, 2 not in the pose")
        assert strict["jaccard"] is None
        loose = describe(placed, frame, reference, max_unmapped_fraction=0.7)["s_b"]
        assert "reason" not in loose
        # {8} is the only mapped residue and is not in the interface {1, 2}.
        assert (loose["n_set_mapped"], loose["n_overlap"]) == (1, 0)
        assert loose["covered_fraction"] == 0.0
        assert loose["interface_in_set_fraction"] == 0.0
        assert loose["jaccard"] == 0.0

    def test_set_with_no_mapped_residue_names_the_columns_it_cannot_give(
        self, reference, reference_atoms
    ):
        trimmed = without(moved(reference_atoms), "A", (6, 7))
        frame, placed = place(trimmed, reference)
        only_missing = Reference(
            reference_atoms, "A", "B", residue_sets={"gone": keys(6, 7)}, name="gone"
        )
        block = describe(placed, frame, only_missing, max_unmapped_fraction=1.0)["gone"]
        assert "reason" not in block
        assert (block["n_set"], block["n_set_mapped"], block["n_overlap"]) == (2, 0, 0)
        assert block["covered_fraction"] is None
        assert block["column_reasons"] == {
            "covered_fraction": "no residue of the set is in the pose"
        }
        assert block["interface_in_set_fraction"] == 0.0
        assert block["jaccard"] == 0.0
        assert block["receptor_buried_area_in_set_angstrom2"] == 0.0

    def test_no_interface(self, reference, reference_atoms):
        apart = tiny_complex(height=34.0)
        frame, placed = place(moved(apart), reference)
        block = describe(placed, frame, reference)["s_a"]
        assert block["n_interface"] == 0
        assert block["n_overlap"] == 0
        assert block["covered_fraction"] == 0.0
        assert block["jaccard"] == 0.0
        assert block["receptor_buried_area_in_set_angstrom2"] == 0.0
        assert block["interface_in_set_fraction"] is None
        assert block["receptor_buried_area_in_set_fraction"] is None
        assert sorted(block["column_reasons"]) == [
            "interface_in_set_fraction",
            "receptor_buried_area_in_set_fraction",
        ]
        assert "reason" not in block

    def test_interface_threshold_is_a_parameter(self, reference, pose):
        frame, placed = pose
        none = describe(placed, frame, reference, interface_threshold_angstrom2=CAP_AREA * 2)
        assert none["s_a"]["n_interface"] == 0
        everything = describe(placed, frame, reference, interface_threshold_angstrom2=0.0)
        assert everything["s_a"]["n_interface"] == 12  # every residue has area >= 0
        assert everything["far"]["covered_fraction"] == 1.0

    def test_failed_frame_gives_every_set_a_reason(self, reference, reference_atoms):
        short = without(reference_atoms, "A", range(6, 13))  # 5 residues: below the fit minimum
        frame = superpose_receptor(short, reference)
        assert not frame.ok
        blocks = describe(short, frame, reference)
        assert list(blocks) == list(reference.residue_sets)
        for name, block in blocks.items():
            assert block["reason"].startswith("the frame failed: ")
            assert block["n_set"] == len(reference.residue_sets[name])
            assert block["n_set_mapped"] is None
            assert block["jaccard"] is None

    def test_missing_binder_chain_gives_every_set_a_reason(self, reference, pose):
        frame, placed = pose
        receptor_only = placed[placed.chain_id == "A"]
        for block in describe(receptor_only, frame, reference).values():
            assert "no heavy polymer atom left for chain(s) ['B']" in block["reason"]

    def test_atoms_renumbered_to_the_reference_are_recognised(self, reference_atoms):
        shifted = tiny_complex(numbering={i: (i + 100, "") for i in range(1, 13)})
        mapping = {("A", i + 100): ("A", i) for i in range(1, 13)}
        ref = Reference(reference_atoms, "A", "B", residue_map=mapping, residue_sets=SETS)
        frame, placed = place(moved(shifted), ref)
        # The numbering the frame read is the pose's own, so the sets still resolve ...
        assert describe(placed, frame, ref)["s_a"]["n_overlap"] == 2
        # ... and a copy renumbered to the reference is told apart from it.
        renumbered = renumber_to_reference(placed, frame, ref)
        block = describe(renumbered, frame, ref)["s_a"]
        assert "pass the atoms with the numbering the frame stage read" in block["reason"]

    def test_empty_reference_sets_give_an_empty_dict(self, reference_atoms, pose):
        frame, placed = pose
        plain = Reference(reference_atoms, "A", "B")
        assert describe(placed, frame, plain) == {}

    def test_result_is_plain_json(self, reference, pose):
        frame, placed = pose
        text = json.dumps(describe(placed, frame, reference))
        assert json.loads(text)["s_a"]["n_overlap"] == 2

    def test_areas_can_be_shared(self, reference, pose):
        frame, placed = pose
        areas = interface_areas(placed, "B", "A")
        assert describe(placed, frame, reference, areas=areas) == describe(placed, frame, reference)

    def test_areas_for_other_chains_or_probe_are_refused(self, reference, pose):
        frame, placed = pose
        areas = interface_areas(placed, "B", "A")
        with pytest.raises(ValueError, match="probe_radius"):
            describe(placed, frame, reference, areas=areas, probe_radius=1.0)
        with pytest.raises(ValueError, match="binder 'B' and target 'A', not 'A' and 'B'"):
            residue_set_descriptors(
                placed, frame, reference, binder_chain="A", target_chain="B", areas=areas
            )

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"max_unmapped_fraction": 1.5}, ValueError),
            ({"max_unmapped_fraction": -0.1}, ValueError),
            ({"interface_threshold_angstrom2": -1.0}, ValueError),
            ({"interface_threshold_angstrom2": "0.5"}, TypeError),
            ({"probe_radius": 0.0}, ValueError),
        ],
    )
    def test_invalid_parameters(self, reference, pose, kwargs, error):
        frame, placed = pose
        with pytest.raises(error):
            describe(placed, frame, reference, **kwargs)

    def test_wrong_object_types(self, reference, pose):
        frame, placed = pose
        with pytest.raises(TypeError, match="reference must be a Reference"):
            describe(placed, frame, "reference")
        with pytest.raises(TypeError, match="frame must be a FrameResult"):
            describe(placed, None, reference)


class TestInsertionCodes:
    """52 and 52A are two residues."""

    NUMBERING = {i: (i, "") for i in range(1, 13)} | {5: (52, ""), 6: (52, "A"), 7: (53, "")}

    def test_set_names_one_of_two_residues_with_one_number(self):
        atoms = tiny_complex(contacts=(6,), numbering=self.NUMBERING)  # binder above 52A
        sets = {"plain": [ResidueKey("A", 52)], "coded": [ResidueKey("A", 52, "A")]}
        ref = Reference(atoms, "A", "B", residue_sets=sets)
        frame, placed = place(moved(atoms), ref)
        assert ("A", 52, "A") in {(p.chain, p.number, p.icode) for p, _ in frame.pairs}
        blocks = describe(placed, frame, ref)
        assert blocks["plain"]["n_overlap"] == 0
        assert blocks["plain"]["n_interface"] == 1
        assert blocks["coded"]["n_overlap"] == 1
        assert blocks["coded"]["covered_fraction"] == 1.0
        assert blocks["coded"]["receptor_buried_area_in_set_angstrom2"] == pytest.approx(
            CAP_AREA, rel=0.03
        )
        assert blocks["plain"]["receptor_buried_area_in_set_angstrom2"] == 0.0

    def test_real_structure_with_insertion_codes(self):
        # 3P8F numbers trypsin ... 60, 60A, ..., 60I. Its interface (the per-residue labels of
        # compute_interface_metrics) holds 60, 60B and 60G, and 60A and 60H are outside it.
        from binding_metrics.metrics.interface import compute_interface_metrics

        atoms = _load(SFTI)
        result = compute_interface_metrics(SFTI, binder_chain="I", target_chain="A")
        sets = {
            "coded": [ResidueKey("A", 60, c) for c in ("A", "B", "G", "H")],
            "all_interface": [
                ResidueKey.parse(label.split(":", 1)[1])
                for label in result["interface_residues_receptor"]
            ],
        }
        ref = Reference(SFTI, "A", "I", residue_sets=sets)
        frame = superpose_receptor(atoms, ref)
        assert frame.ok and frame.pairing == "number"
        blocks = residue_set_descriptors(atoms, frame, ref, binder_chain="I", target_chain="A")
        coded = blocks["coded"]
        assert result["n_interface_residues_receptor"] == 32
        assert coded["n_interface"] == 32
        assert (coded["n_set"], coded["n_set_mapped"], coded["n_overlap"]) == (4, 4, 2)
        assert coded["covered_fraction"] == pytest.approx(2 / 4)
        assert coded["interface_in_set_fraction"] == pytest.approx(2 / 32)
        assert coded["jaccard"] == pytest.approx(2 / 34)
        everything = blocks["all_interface"]
        assert (everything["n_overlap"], everything["jaccard"]) == (32, 1.0)
        # Residues with less than 0.5 square angstrom are outside the interface and the set but
        # inside the total, so the fraction is just below one.
        assert 0.999 < everything["receptor_buried_area_in_set_fraction"] < 1.0


# ---------------------------------------------------------------------------
# B2: steric exclusion against a partner
# ---------------------------------------------------------------------------


def lens_equal_spheres(radius, distance):
    """Volume shared by two spheres of one radius: pi (4 r + d) (2 r - d)^2 / 12."""
    return math.pi * (4 * radius + distance) * (2 * radius - distance) ** 2 / 12


def lens_by_integration(radius_a, radius_b, distance, points=2_000_001):
    """Volume shared by two spheres from the area of the common disc at each position along
    the axis (a trapezoid rule on a fine grid), independent of the closed form."""
    x = np.linspace(-radius_a, distance + radius_b, points)
    disc = np.minimum(radius_a**2 - x**2, radius_b**2 - (x - distance) ** 2)
    area = math.pi * np.clip(disc, 0.0, None)
    return float(np.sum((area[1:] + area[:-1]) / 2 * np.diff(x)))


def carbon_atoms(xyz, chain="P", residue_name="ALA"):
    """One carbon per residue (numbered from 1) at the given coordinates."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    atoms = struc.AtomArray(len(xyz))
    atoms.chain_id[:] = chain
    atoms.res_id[:] = np.arange(1, len(xyz) + 1)
    atoms.res_name[:] = residue_name
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.coord = xyz.astype(np.float32)
    return atoms


def measure_partners(pose_atoms, frame, reference, **kwargs):
    return partner_descriptors(
        pose_atoms, frame, reference, binder_chain="B", target_chain="A", **kwargs
    )


def frame_of(pose_atoms, reference):
    frame = superpose_receptor(pose_atoms, reference)
    assert frame.ok, frame.reason
    return frame


class TestSphereOverlapVolume:
    def test_equal_spheres_match_the_textbook_lens(self):
        for distance in (0.5, 1.7, 2.5, 3.3):
            assert sphere_overlap_volume(1.7, 1.7, distance) == pytest.approx(
                lens_equal_spheres(1.7, distance), rel=1e-12
            )

    @pytest.mark.parametrize(
        "radius_a, radius_b, distance", [(1.7, 1.52, 2.0), (2.0, 1.0, 2.2), (1.55, 1.8, 0.4)]
    )
    def test_unequal_spheres_match_numerical_integration(self, radius_a, radius_b, distance):
        assert sphere_overlap_volume(radius_a, radius_b, distance) == pytest.approx(
            lens_by_integration(radius_a, radius_b, distance), rel=1e-5
        )

    def test_contained_sphere_is_the_smaller_sphere(self):
        smaller = 4 / 3 * math.pi * 1.0**3
        assert sphere_overlap_volume(2.0, 1.0, 0.5) == pytest.approx(smaller, rel=1e-12)
        assert sphere_overlap_volume(1.0, 2.0, 0.0) == pytest.approx(smaller, rel=1e-12)
        # two equal spheres on one centre coincide
        assert sphere_overlap_volume(1.7, 1.7, 0.0) == pytest.approx(
            4 / 3 * math.pi * 1.7**3, rel=1e-12
        )

    def test_continuous_where_the_branches_meet(self):
        smaller = 4 / 3 * math.pi * 1.0**3
        assert sphere_overlap_volume(2.0, 1.0, 1.0) == pytest.approx(smaller, rel=1e-12)
        assert sphere_overlap_volume(2.0, 1.0, 1.0 + 1e-7) == pytest.approx(smaller, rel=1e-5)

    def test_no_overlap_at_or_beyond_contact(self):
        assert sphere_overlap_volume(1.7, 1.52, 3.22) == 0.0
        assert sphere_overlap_volume(1.7, 1.52, 5.0) == 0.0

    def test_falls_as_the_spheres_part(self):
        distances = np.linspace(0.0, 3.5, 36)
        volumes = sphere_overlap_volume(1.7, 1.52, distances)
        assert np.all(np.diff(volumes) <= 1e-12)
        assert volumes[0] > 0 and volumes[-1] == 0.0

    def test_arrays_broadcast_and_scalars_are_floats(self):
        assert isinstance(sphere_overlap_volume(1.7, 1.7, 2.0), float)
        volumes = sphere_overlap_volume([1.7, 1.52], 1.7, [[1.0], [2.0]])
        assert volumes.shape == (2, 2)

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="radii"):
            sphere_overlap_volume(0.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="negative"):
            sphere_overlap_volume(1.0, 1.0, -0.1)


ANCHOR_DISTANCES = (1.0, 2.5, 3.9, 4.0, 5.0)


@pytest.fixture(scope="module")
def anchor_setup(reference_atoms):
    """Binder atoms at known distances from a one-atom partner, away from the receptor."""
    anchor = np.array([30.0, 30.0, 30.0])
    pose_atoms = tiny_complex(binder_xyz=[anchor + [d, 0.0, 0.0] for d in ANCHOR_DISTANCES])
    reference = Reference(
        reference_atoms, "A", "B", partners={"anchor": Partner(carbon_atoms([anchor]))}
    )
    return pose_atoms, reference, frame_of(pose_atoms, reference)


@pytest.fixture(scope="module")
def cap_reference(reference_atoms):
    """A reference whose partner is one atom above receptor residue 1."""
    partner = Partner(carbon_atoms([[0.0, 0.0, HEIGHT]]))
    return Reference(reference_atoms, "A", "B", partners={"cap": partner})


@pytest.fixture(scope="module")
def two_partner_reference(reference_atoms):
    partners = {
        "cap": Partner(carbon_atoms([[0.0, 0.0, HEIGHT]])),
        "second": Partner(carbon_atoms([[10.0, 0.0, HEIGHT]])),
    }
    return Reference(reference_atoms, "A", "B", partners=partners)


class TestPartnerContacts:
    def test_distances_clashes_shell_and_volume_by_hand(self, anchor_setup):
        pose_atoms, reference, frame = anchor_setup
        block = measure_partners(pose_atoms, frame, reference)["anchor"]
        assert list(block)[: len(PARTNER_COLUMNS)] == list(PARTNER_COLUMNS)
        assert block["min_distance_angstrom"] == 1.0
        # 1.0 is within both cutoffs, 2.5 within 3.0 only; the partner has one atom, so atoms
        # and pairs agree.
        assert block["clash_atoms"] == {"2.2": 1, "3.0": 2}
        assert block["clash_pairs"] == {"2.2": 1, "3.0": 2}
        # 1.0, 2.5, 3.9 and 4.0 are in the shell (4.0 exactly: the shell is closed), 5.0 is not.
        assert block["shell_fraction"] == pytest.approx(4 / 5)
        # Carbon radius 1.7: the pairs closer than 3.4 are the first two.
        expected = lens_equal_spheres(1.7, 1.0) + lens_equal_spheres(1.7, 2.5)
        assert block["overlap_volume_angstrom3"] == pytest.approx(expected, rel=1e-12)

    def test_partner_far_from_the_receptor_buries_nothing(self, anchor_setup):
        pose_atoms, reference, frame = anchor_setup
        block = measure_partners(pose_atoms, frame, reference)["anchor"]
        assert block["shared_buried_area_angstrom2"] == 0.0
        assert block["partner_buried_area_angstrom2"] == 0.0
        assert block["pose_buried_area_angstrom2"] == 0.0
        assert block["shared_fraction_of_partner"] is None
        assert list(block["column_reasons"]) == ["shared_fraction_of_partner"]
        assert "reason" not in block

    def test_cutoffs_and_shell_are_parameters(self, anchor_setup):
        pose_atoms, reference, frame = anchor_setup
        block = measure_partners(
            pose_atoms, frame, reference, clash_cutoffs_angstrom=(1.5, 4.0), shell_angstrom=3.0
        )["anchor"]
        # 1.0, 2.5 and 3.9 are closer than 4.0; the atom at exactly 4.0 is not (clashes are strict)
        assert block["clash_atoms"] == {"1.5": 1, "4.0": 3}
        assert block["clash_pairs"] == {"1.5": 1, "4.0": 3}
        assert block["shell_fraction"] == pytest.approx(2 / 5)

    def test_defaults_are_the_documented_ones(self):
        assert DEFAULT_CLASH_CUTOFFS_ANGSTROM == (2.2, 3.0)
        assert DEFAULT_SHELL_ANGSTROM == 4.0

    def test_clash_pairs_count_every_partner_atom(self, reference_atoms):
        # Two partner atoms 1.2 apart and one binder atom 2.0 from the first (2.33 from the
        # second): one pair under 2.2, two pairs under 3.0, and one binder atom either way.
        anchor = np.array([30.0, 30.0, 30.0])
        pose_atoms = tiny_complex(binder_xyz=[anchor + [2.0, 0.0, 0.0]])
        partner = Partner(carbon_atoms([anchor, anchor + [0.0, 1.2, 0.0]]))
        reference = Reference(reference_atoms, "A", "B", partners={"two": partner})
        block = measure_partners(pose_atoms, frame_of(pose_atoms, reference), reference)["two"]
        assert block["clash_atoms"] == {"2.2": 1, "3.0": 1}
        assert block["clash_pairs"] == {"2.2": 1, "3.0": 2}
        assert block["min_distance_angstrom"] == 2.0

    def test_result_is_plain_json(self, anchor_setup):
        pose_atoms, reference, frame = anchor_setup
        text = json.dumps(measure_partners(pose_atoms, frame, reference))
        assert json.loads(text)["anchor"]["clash_atoms"]["3.0"] == 2


class TestPartnerBuriedArea:
    """One partner atom above receptor residue 1 buries a cap there and nowhere else."""

    def test_pose_covering_the_partner_surface_and_more(self, cap_reference):
        pose_atoms = tiny_complex(contacts=(1, 2))
        block = measure_partners(pose_atoms, frame_of(pose_atoms, cap_reference), cap_reference)[
            "cap"
        ]
        assert block["partner_buried_area_angstrom2"] == pytest.approx(CAP_AREA, rel=0.03)
        assert block["pose_buried_area_angstrom2"] == pytest.approx(2 * CAP_AREA, rel=0.03)
        assert block["shared_buried_area_angstrom2"] == pytest.approx(CAP_AREA, rel=0.03)
        # the atom that both bury is covered by the same sphere in both: the whole partner area
        assert block["shared_fraction_of_partner"] == pytest.approx(1.0, rel=1e-9)
        assert block["shared_buried_area_angstrom2"] == block["partner_buried_area_angstrom2"]
        assert block["min_distance_angstrom"] == 0.0

    def test_pose_covering_other_surface_shares_nothing(self, cap_reference):
        pose_atoms = tiny_complex(contacts=(2, 3))
        block = measure_partners(pose_atoms, frame_of(pose_atoms, cap_reference), cap_reference)[
            "cap"
        ]
        assert block["shared_buried_area_angstrom2"] == 0.0
        assert block["shared_fraction_of_partner"] == 0.0
        assert block["pose_buried_area_angstrom2"] == pytest.approx(2 * CAP_AREA, rel=0.03)

    def test_areas_are_summed_over_matched_receptor_atoms_only(self, cap_reference):
        # Residue 1 is missing from the pose: the partner's area on it is not counted.
        pose_atoms = without(tiny_complex(contacts=(1, 2)), "A", (1,))
        block = measure_partners(pose_atoms, frame_of(pose_atoms, cap_reference), cap_reference)[
            "cap"
        ]
        assert block["partner_buried_area_angstrom2"] == 0.0
        assert block["shared_buried_area_angstrom2"] == 0.0
        assert block["pose_buried_area_angstrom2"] == pytest.approx(CAP_AREA, rel=0.03)
        assert block["shared_fraction_of_partner"] is None
        assert "partner buries no area" in block["column_reasons"]["shared_fraction_of_partner"]

    def test_atoms_of_a_renumbered_pose_are_matched_through_the_pairing(self, reference_atoms):
        shifted = tiny_complex(contacts=(1, 2), numbering={i: (i + 100, "") for i in range(1, 13)})
        mapping = {("A", i + 100): ("A", i) for i in range(1, 13)}
        ref = Reference(
            reference_atoms,
            "A",
            "B",
            residue_map=mapping,
            partners={"cap": Partner(carbon_atoms([[0.0, 0.0, HEIGHT]]))},
        )
        block = measure_partners(shifted, frame_of(shifted, ref), ref)["cap"]
        assert block["shared_fraction_of_partner"] == pytest.approx(1.0, rel=1e-9)

    def test_no_matched_atom_gives_the_area_columns_a_reason(self, cap_reference):
        import dataclasses

        pose_atoms = tiny_complex(contacts=(1, 2))
        frame = dataclasses.replace(frame_of(pose_atoms, cap_reference), pairs=())
        block = measure_partners(pose_atoms, frame, cap_reference)["cap"]
        area_columns = [c for c in PARTNER_COLUMNS if "area" in c or c.endswith("of_partner")]
        assert sorted(block["column_reasons"]) == sorted(area_columns)
        assert all(block[c] is None for c in area_columns)
        assert block["min_distance_angstrom"] == 0.0

    def test_areas_can_be_shared(self, cap_reference):
        pose_atoms = tiny_complex(contacts=(1, 2))
        frame = frame_of(pose_atoms, cap_reference)
        areas = interface_areas(pose_atoms, "B", "A")
        assert measure_partners(pose_atoms, frame, cap_reference, areas=areas) == measure_partners(
            pose_atoms, frame, cap_reference
        )


class TestPartnerFailures:
    def test_blocks_follow_the_partners_of_the_reference(self, two_partner_reference):
        pose_atoms = tiny_complex(contacts=(1, 2))
        blocks = measure_partners(
            pose_atoms, frame_of(pose_atoms, two_partner_reference), two_partner_reference
        )
        assert list(blocks) == ["cap", "second"]
        assert blocks["second"]["shared_buried_area_angstrom2"] == pytest.approx(CAP_AREA, rel=0.03)

    def test_no_partner_gives_an_empty_dict(self, reference_atoms):
        plain = Reference(reference_atoms, "A", "B")
        pose_atoms = tiny_complex()
        assert measure_partners(pose_atoms, frame_of(pose_atoms, plain), plain) == {}

    def test_failed_frame(self, two_partner_reference, reference_atoms):
        short = without(reference_atoms, "A", range(6, 13))
        frame = superpose_receptor(short, two_partner_reference)
        assert not frame.ok
        for block in measure_partners(short, frame, two_partner_reference).values():
            assert block["reason"].startswith("the frame failed: ")
            assert all(block[c] is None for c in PARTNER_COLUMNS)

    def test_missing_binder_chain(self, two_partner_reference):
        pose_atoms = tiny_complex()
        frame = frame_of(pose_atoms, two_partner_reference)
        blocks = measure_partners(
            pose_atoms[pose_atoms.chain_id == "A"], frame, two_partner_reference
        )
        for block in blocks.values():
            assert "no heavy polymer atom left for chain(s) ['B']" in block["reason"]

    def test_partner_without_polymer_atoms(self, reference_atoms):
        water = carbon_atoms([[0.0, 0.0, 3.0]], chain="W", residue_name="HOH")
        water.element[:] = "O"
        water.atom_name[:] = "O"
        ref = Reference(
            reference_atoms,
            "A",
            "B",
            partners={
                "water": Partner(water, chains="W"),
                "cap": Partner(carbon_atoms([[0, 0, 4]])),
            },
        )
        pose_atoms = tiny_complex()
        blocks = measure_partners(pose_atoms, frame_of(pose_atoms, ref), ref)
        assert blocks["water"]["reason"] == "partner 'water' has no heavy polymer atom"
        assert "reason" not in blocks["cap"]  # the other partner is still measured

    def test_partner_chain_missing_from_its_structure_raises(self, reference_atoms):
        ref = Reference(
            reference_atoms,
            "A",
            "B",
            partners={"x": Partner(carbon_atoms([[0, 0, 4]]), chains="Z")},
        )
        pose_atoms = tiny_complex()
        with pytest.raises(ValueError, match="partner chains"):
            measure_partners(pose_atoms, frame_of(pose_atoms, ref), ref)

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"clash_cutoffs_angstrom": ()}, ValueError),
            ({"clash_cutoffs_angstrom": (2.2, 2.2)}, ValueError),
            ({"clash_cutoffs_angstrom": (0.0,)}, ValueError),
            ({"clash_cutoffs_angstrom": 2.2}, TypeError),
            ({"shell_angstrom": 0.0}, ValueError),
            ({"shell_angstrom": "4"}, TypeError),
            ({"probe_radius": -1.0}, ValueError),
        ],
    )
    def test_invalid_parameters(self, two_partner_reference, kwargs, error):
        pose_atoms = tiny_complex()
        with pytest.raises(error):
            measure_partners(
                pose_atoms,
                frame_of(pose_atoms, two_partner_reference),
                two_partner_reference,
                **kwargs,
            )


SCALE_SHIFTS = (0.0, 3.0, 6.0, 10.0)


@pytest.fixture(scope="module")
def results():
    """The crystal peptide of 1YCR as the partner, and the peptide translated away from the
    receptor by each shift (along the line from the centre of the binding site to the peptide)."""
    atoms = _load(YCR)
    reference = Reference(YCR, "A", "B", partners={"crystal": Partner(YCR, chains="B")})
    frame = frame_of(atoms, reference)
    peptide = atoms.chain_id == "B"
    receptor = (atoms.chain_id == "A") & ~atoms.hetero
    offsets = atoms.coord[receptor][:, None, :] - atoms.coord[peptide][None, :, :]
    near = np.sqrt((offsets**2).sum(axis=2)).min(axis=1) < 4.5
    away = atoms.coord[peptide].mean(axis=0) - atoms.coord[receptor][near].mean(axis=0)
    away /= np.linalg.norm(away)
    measured = {}
    for shift in SCALE_SHIFTS:
        pose_atoms = atoms.copy()
        pose_atoms.coord[peptide] += (shift * away).astype(np.float32)
        measured[shift] = measure_partners(pose_atoms, frame, reference)["crystal"]
    return measured


class TestScaleCheck1YCR:
    """The crystal peptide as the partner; the poses are translations of the peptide away from
    the receptor (design section 8.4): the shared area is the partner's at 0 and falls to 0."""

    SHIFTS = SCALE_SHIFTS

    def test_pose_equal_to_the_partner_shares_all_of_its_area(self, results):
        at_zero = results[0.0]
        assert at_zero["shared_buried_area_angstrom2"] == at_zero["partner_buried_area_angstrom2"]
        assert at_zero["pose_buried_area_angstrom2"] == at_zero["partner_buried_area_angstrom2"]
        assert at_zero["shared_fraction_of_partner"] == 1.0
        # the design note measured 662 square angstrom for the crystal peptide on MDM2
        assert at_zero["partner_buried_area_angstrom2"] == pytest.approx(662.3, abs=1.0)

    def test_shared_area_falls_monotonically_to_zero(self, results):
        shared = [results[s]["shared_buried_area_angstrom2"] for s in self.SHIFTS]
        assert shared[0] > shared[1] > shared[2] > shared[3]
        assert shared[3] == pytest.approx(0.0, abs=1e-9)
        fractions = [results[s]["shared_fraction_of_partner"] for s in self.SHIFTS]
        assert fractions[0] == 1.0
        assert 0.3 < fractions[1] < 0.8
        assert 0.05 < fractions[2] < 0.4
        assert fractions[3] == pytest.approx(0.0, abs=1e-12)

    def test_partner_area_does_not_depend_on_the_pose(self, results):
        areas = {r["partner_buried_area_angstrom2"] for r in results.values()}
        assert len(areas) == 1

    def test_contact_columns_follow_the_translation(self, results):
        at_zero, at_ten = results[0.0], results[10.0]
        assert at_zero["min_distance_angstrom"] == 0.0
        assert at_zero["clash_atoms"] == {
            "2.2": 113,
            "3.0": 113,
        }  # the 113 heavy atoms of the peptide
        assert at_zero["shell_fraction"] == 1.0
        assert at_zero["overlap_volume_angstrom3"] > 0
        assert at_ten["shell_fraction"] < 0.5
        assert at_ten["clash_atoms"]["2.2"] < at_zero["clash_atoms"]["2.2"]
        volumes = [results[s]["overlap_volume_angstrom3"] for s in self.SHIFTS]
        assert volumes[0] > volumes[1] > volumes[3]
