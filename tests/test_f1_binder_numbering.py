"""The binder of a pose takes the numbers of the reference binder (issue #187).

A co-folding model numbers both chains from 1, the crystal peptide of 1YCR is 17-29. The
receptor was renumbered to the reference and the binder was not, so ``structure_rmsd`` had no
(chain, residue number, atom name) key in common for the binder and gave no binder RMSD.
"""

import numpy as np
import pytest

from binding_metrics.poses import frame as frame_module
from binding_metrics.poses.aggregate import aggregate_poses
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.frame import renumber_to_reference, superpose_receptor
from binding_metrics.poses.reference import Reference
from binding_metrics.poses.score import score_pose_set
from binding_metrics.testing import make_synthetic_poses
from tests.conftest import DATA_DIR
from tests.poses.test_frame import consecutive, drop

pytest.importorskip("biotite.structure")

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
PEPTIDE_NUMBERS = list(range(17, 30))
RMSD_KEYS = ("rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design")


@pytest.fixture(scope="module")
def reference():
    return Reference(YCR, "A", "B")


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


def _from_one(pose, binder_drop=()):
    """The pose with both chains numbered from 1, without the given binder positions."""
    atoms = consecutive(consecutive(pose.atoms(), "A"), "B")
    if binder_drop:
        atoms = drop(atoms, "B", binder_drop)
    return DockedPose.from_atoms(atoms, id=pose.id, **pose.provenance)


@pytest.fixture(scope="module")
def cofolded(synthetic):
    return [_from_one(pose) for pose in synthetic.poses[:3]]


@pytest.fixture(scope="module")
def one_residue_short(synthetic):
    return [_from_one(pose, binder_drop=(13,)) for pose in synthetic.poses[:3]]


def _numbers(atoms, chain):
    chain_atoms = atoms[atoms.chain_id == chain]
    return list(dict.fromkeys(int(number) for number in chain_atoms.res_id))


class TestBinderPairing:
    def test_a_binder_numbered_from_1_is_paired_with_the_reference_peptide(
        self, cofolded, reference
    ):
        pairing = frame_module.pair_binder_residues(cofolded[0], reference, binder_chain="B")
        assert pairing.ok
        assert pairing.n_paired == 13
        assert [key.number for key in pairing.pose_keys] == list(range(1, 14))
        assert [key.number for key in pairing.reference_keys] == PEPTIDE_NUMBERS
        assert {key.chain for key in pairing.reference_keys} == {"B"}

    def test_the_binder_pairing_writes_the_reference_numbers(self, cofolded, reference):
        pose = cofolded[0]
        frame = superpose_receptor(pose, reference)
        pairing = frame_module.pair_binder_residues(pose, reference, binder_chain="B")
        renumbered = renumber_to_reference(pose, frame, reference, binder_pairing=pairing)
        assert _numbers(renumbered, "B") == PEPTIDE_NUMBERS
        assert _numbers(renumbered, "A") == _numbers(
            reference.atoms()[reference.atoms().chain_id == "A"], "A"
        )
        assert np.array_equal(renumbered.coord, pose.atoms().coord)

    def test_the_binder_keeps_its_numbers_without_a_pairing(self, cofolded, reference):
        pose = cofolded[0]
        frame = superpose_receptor(pose, reference)
        renumbered = renumber_to_reference(pose, frame, reference)
        assert _numbers(renumbered, "B") == list(range(1, 14))

    def test_a_binder_of_another_length_is_not_paired_and_the_reason_gives_both_lengths(
        self, one_residue_short, reference
    ):
        pairing = frame_module.pair_binder_residues(
            one_residue_short[0], reference, binder_chain="B"
        )
        assert not pairing.ok
        assert "12" in pairing.reason and "13" in pairing.reason
        pose = one_residue_short[0]
        frame = superpose_receptor(pose, reference)
        renumbered = renumber_to_reference(pose, frame, reference, binder_pairing=pairing)
        assert _numbers(renumbered, "B") == list(range(1, 13))

    def test_a_reference_without_a_binder_chain_has_nothing_to_pair(self, cofolded):
        receptor_only = Reference(YCR, "A")
        pairing = frame_module.pair_binder_residues(cofolded[0], receptor_only, binder_chain="B")
        assert not pairing.ok
        assert "no binder chain" in pairing.reason

    def test_a_binder_already_numbered_like_the_reference_is_unchanged(self, synthetic, reference):
        pose = synthetic.poses[0]
        pairing = frame_module.pair_binder_residues(pose, reference, binder_chain="B")
        assert pairing.ok and pairing.method == "number"
        assert [key.number for key in pairing.pose_keys] == PEPTIDE_NUMBERS
        assert pairing.pose_keys == pairing.reference_keys

    def test_a_binder_with_a_different_sequence_is_not_paired(self, cofolded, reference):
        atoms = cofolded[0].atoms().copy()
        atoms.res_name[atoms.chain_id == "B"] = "GLY"
        pairing = frame_module.pair_binder_residues(atoms, reference, binder_chain="B")
        assert not pairing.ok
        assert pairing.n_paired < 13


class TestScorePoseSet:
    def test_the_binder_rmsd_is_computed_for_a_binder_numbered_from_1(self, cofolded, reference):
        result = score_pose_set(cofolded, "B", "A", reference=reference, metrics=["structure_rmsd"])
        for row in result["poses"]:
            block = row["structure_rmsd"]
            assert row["status"] == "ok" and row["reasons"] == {}
            assert "reason" not in block
            assert all(block[key] is not None for key in RMSD_KEYS)
            # the poses move the peptide as a rigid body, which a fit on the peptide undoes
            assert block["rmsd_design"] < 1e-3 and block["bb_rmsd_design"] < 1e-3

    def test_the_aggregates_count_every_pose_for_every_rmsd(self, cofolded, reference):
        result = score_pose_set(cofolded, "B", "A", reference=reference, metrics=["structure_rmsd"])
        entries = result["aggregates"]["all"]
        for key in RMSD_KEYS:
            entry = entries[f"structure_rmsd.{key}"]
            assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (3, 3, True)

    def test_the_interface_labels_of_the_binder_use_the_reference_numbers(
        self, cofolded, reference
    ):
        result = score_pose_set(cofolded[:1], "B", "A", reference=reference, metrics=["interface"])
        labels = result["poses"][0]["interface"]["interface_residues_peptide"]
        numbers = {int(label.split(":")[-1]) for label in labels}
        assert numbers and numbers <= set(PEPTIDE_NUMBERS)

    def test_a_binder_of_another_length_keeps_its_numbers_and_the_row_says_so(
        self, one_residue_short, reference
    ):
        result = score_pose_set(
            one_residue_short, "B", "A", reference=reference, metrics=["structure_rmsd"]
        )
        for row in result["poses"]:
            assert row["status"] == "partial"
            assert "12" in row["reasons"]["binder_numbering"]
            assert "13" in row["reasons"]["binder_numbering"]
            assert row["structure_rmsd"]["rmsd_design"] is None

    def test_a_partial_result_keeps_its_valid_values_in_the_aggregates(
        self, cofolded, one_residue_short, reference
    ):
        poses = [cofolded[0], one_residue_short[1], one_residue_short[2]]
        result = score_pose_set(poses, "B", "A", reference=reference, metrics=["structure_rmsd"])
        assert [row["status"] for row in result["poses"]] == ["ok", "partial", "partial"]
        entries = result["aggregates"]["all"]
        for key in ("rmsd", "bb_rmsd"):
            entry = entries[f"structure_rmsd.{key}"]
            assert (entry["n_ok"], entry["n_total"]) == (3, 3)
            assert entry["mean"] is not None
        for key in ("rmsd_design", "bb_rmsd_design"):
            entry = entries[f"structure_rmsd.{key}"]
            assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (1, 3, False)
            assert list(entry["reasons"].values()) == [2]
            assert key in next(iter(entry["reasons"]))

    def test_a_set_with_no_binder_rmsd_at_all_lists_only_the_complex_rmsd(
        self, one_residue_short, reference
    ):
        result = score_pose_set(
            one_residue_short, "B", "A", reference=reference, metrics=["structure_rmsd"]
        )
        entries = result["aggregates"]["all"]
        listed = sorted(key for key in entries if key.startswith("structure_rmsd."))
        assert listed == ["structure_rmsd.bb_rmsd", "structure_rmsd.rmsd"]
        assert all(entries[key]["n_ok"] == 3 for key in listed)

    def test_the_binder_numbering_is_not_reported_when_no_metric_reads_the_reference(
        self, one_residue_short, reference
    ):
        result = score_pose_set(
            one_residue_short, "B", "A", reference=reference, metrics=["interface"]
        )
        assert [row["status"] for row in result["poses"]] == ["ok"] * 3
        assert all(row["reasons"] == {} for row in result["poses"])

    def test_renumber_receptor_false_leaves_the_binder_numbers(self, cofolded, reference):
        result = score_pose_set(
            cofolded[:1],
            "B",
            "A",
            reference=reference,
            metrics=["structure_rmsd"],
            options={"frame": {"renumber_receptor": False}},
        )
        assert result["poses"][0]["structure_rmsd"]["rmsd_design"] is None


class TestPartialBlocks:
    @staticmethod
    def _rows():
        text = "not computed: rmsd_design (no shared key)"
        partial = {"rmsd": 1.5, "rmsd_design": None, "reason": text}
        return text, [
            {"id": "p1", "status": "partial", "structure_rmsd": partial, "reasons": {}},
            {
                "id": "p2",
                "status": "ok",
                "structure_rmsd": {"rmsd": 2.5, "rmsd_design": 0.5},
                "reasons": {},
            },
        ]

    def test_without_the_option_a_top_level_reason_excludes_the_block(self):
        _, rows = self._rows()
        entries = aggregate_poses(rows, ())["all"]
        assert entries["structure_rmsd.rmsd"]["n_ok"] == 1

    def test_a_partial_block_keeps_its_valid_values_and_gives_the_none_its_reason(self):
        text, rows = self._rows()
        entries = aggregate_poses(rows, (), partial_blocks=("structure_rmsd",))["all"]
        assert entries["structure_rmsd.rmsd"]["n_ok"] == 2
        assert entries["structure_rmsd.rmsd"]["mean"] == 2.0
        assert entries["structure_rmsd.rmsd"]["reasons"] == {}
        design = entries["structure_rmsd.rmsd_design"]
        assert (design["n_ok"], design["n_total"]) == (1, 2)
        assert design["reasons"] == {text: 1}

    def test_a_reason_deeper_in_a_partial_block_still_excludes_what_is_under_it(self):
        rows = [
            {
                "id": "p1",
                "status": "ok",
                "structure_rmsd": {"rmsd": 1.0, "term": {"value": 3.0, "reason": "no term"}},
                "reasons": {},
            }
        ]
        entries = aggregate_poses(rows, (), partial_blocks=("structure_rmsd",))["all"]
        assert entries["structure_rmsd.rmsd"]["n_ok"] == 1
        assert entries["structure_rmsd.term.value"]["n_ok"] == 0
