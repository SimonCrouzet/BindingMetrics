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
    DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2,
    DEFAULT_MAX_UNMAPPED_FRACTION,
    DEFAULT_PROBE_RADIUS,
    SET_COLUMNS,
    InterfaceAreas,
    interface_areas,
    residue_set_descriptors,
)
from binding_metrics.poses.frame import (
    FrameOptions,
    place_in_reference_frame,
    renumber_to_reference,
    superpose_receptor,
)
from binding_metrics.poses.reference import Reference, ResidueKey

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


def tiny_complex(contacts=(1, 2, 6), height=HEIGHT, numbering=None):
    """Twelve receptor residues on a 4 x 3 grid (10 angstrom apart, z = 0), chain A, and one
    binder atom (chain B) at ``height`` above each receptor residue of ``contacts``.

    ``numbering`` maps the residue index 1..12 to ``(number, icode)``; default ``(index, "")``.
    Every residue is one carbon named CA with an amino-acid name, so the residue is a polymer
    residue and the sequence is not periodic.
    """
    numbering = numbering or {i: (i, "") for i in range(1, 13)}
    grid = {i: np.array([((i - 1) % 4) * 10.0, ((i - 1) // 4) * 10.0, 0.0]) for i in range(1, 13)}
    rows = []
    for i in range(1, 13):
        number, icode = numbering[i]
        rows.append(("A", number, icode, RESIDUE_NAMES[i - 1], grid[i]))
    for position, i in enumerate(contacts, start=1):
        rows.append(("B", position, "", "ALA", grid[i] + np.array([0.0, 0.0, height])))
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
