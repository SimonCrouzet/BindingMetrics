"""score_pose_set with reference sets, partners, several variants and the residue table."""

import json
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.poses import (
    PARTNER_COLUMNS,
    SET_COLUMNS,
    TABLE_COLUMNS,
    compare_consensus_interfaces,
    consensus_interface,
)
from binding_metrics.poses.descriptors import (
    interface_residue_table,
    partner_descriptors,
    residue_set_descriptors,
)
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.frame import place_in_reference_frame, superpose_receptor
from binding_metrics.poses.reference import Partner, Reference, ResidueKey
from binding_metrics.poses.score import DescriptorOptions, PoseSetResult, score_pose_set
from binding_metrics.testing import make_synthetic_poses
from tests.poses.test_frame import consecutive, drop

pytest.importorskip("biotite.structure")

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"

EPITOPE = [25, 26, 50, 51, 54, 57, 58, 61, 62, 67, 70, 71, 72, 73, 75, 91, 93, 94, 96, 99, 100, 104]
OFFSETS = {
    "backend_a-s1": 0.0,
    "backend_a-s2": 0.4,
    "backend_a-s3": 0.8,
    "backend_b-s1": 2.0,
    "backend_b-s2": 2.3,
    "backend_b-s3": 2.9,
}


def epitope_reference(**kwargs):
    return Reference(
        YCR,
        "A",
        "B",
        residue_sets={"epitope": [ResidueKey("A", n) for n in EPITOPE]},
        partners={"crystal_peptide": Partner(YCR, chains=["B"])},
        **kwargs,
    )


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def reference():
    return epitope_reference()


@pytest.fixture(scope="module")
def result(synthetic, reference):
    return score_pose_set(synthetic.poses, "B", "A", reference=reference, residue_table=True)


def direct(pose, reference):
    """The descriptors of one pose from the three functions, called as their lane documents."""
    frame = superpose_receptor(pose, reference, receptor_chain="A")
    placed = place_in_reference_frame(pose.atoms(), frame)
    common = dict(binder_chain="B", target_chain="A")
    return (
        residue_set_descriptors(placed, frame, reference, **common),
        partner_descriptors(placed, frame, reference, **common),
        interface_residue_table(placed, frame, reference, **common),
    )


def plain(value):
    return json.loads(json.dumps(value))


class TestBlocks:
    def test_the_row_has_the_descriptors_per_variant_and_the_table(self, result):
        row = result["poses"][0]
        assert list(row)[-2:] == ["descriptors", "residue_table"]
        assert list(row["descriptors"]) == ["reference"]
        assert list(row["descriptors"]["reference"]) == ["sets", "partners"]
        assert list(row["descriptors"]["reference"]["sets"]) == ["epitope"]
        assert list(row["descriptors"]["reference"]["partners"]) == ["crystal_peptide"]

    def test_the_columns_are_those_of_the_descriptor_functions(self, result):
        block = result["poses"][0]["descriptors"]["reference"]
        assert tuple(block["sets"]["epitope"]) == SET_COLUMNS
        assert tuple(block["partners"]["crystal_peptide"]) == PARTNER_COLUMNS

    def test_the_values_are_those_of_a_direct_call_on_the_placed_atoms(
        self, result, synthetic, reference
    ):
        for pose in (synthetic.poses[0], synthetic.poses[4]):
            sets, partners, table = direct(pose, reference)
            row = result.row(pose.id)
            assert row["descriptors"]["reference"]["sets"] == plain(sets)
            assert row["descriptors"]["reference"]["partners"] == plain(partners)
            assert row["residue_table"] == plain(table)

    def test_the_epitope_is_covered_by_the_interface(self, result):
        for row in result["poses"]:
            block = row["descriptors"]["reference"]["sets"]["epitope"]
            assert (block["n_set"], block["n_set_mapped"]) == (22, 22)
            assert block["covered_fraction"] > 0.9
            assert 0.0 < block["receptor_buried_area_in_set_fraction"] <= 1.0
            assert "reason" not in block

    def test_the_unmoved_peptide_shares_all_of_the_partner_area_and_it_falls_with_the_shift(
        self, result
    ):
        shared = {
            row["id"]: row["descriptors"]["reference"]["partners"]["crystal_peptide"][
                "shared_buried_area_angstrom2"
            ]
            for row in result["poses"]
        }
        # the pose with no shift is the crystal peptide: its partner area is 662.3 A^2 (the
        # scale check of the design note on 1YCR) and the fraction is exactly one
        assert shared["backend_a-s1"] == pytest.approx(662.3, abs=0.5)
        fractions = [
            row["descriptors"]["reference"]["partners"]["crystal_peptide"][
                "shared_fraction_of_partner"
            ]
            for row in result["poses"]
        ]
        assert fractions[0] == pytest.approx(1.0, abs=1e-6)
        ordered = sorted(shared, key=OFFSETS.get)
        values = [shared[pose_id] for pose_id in ordered]
        assert values == sorted(values, reverse=True) and values[-1] < 0.85 * values[0]

    def test_the_clash_keys_are_the_cutoffs_as_text(self, result):
        partner = result["poses"][0]["descriptors"]["reference"]["partners"]["crystal_peptide"]
        assert list(partner["clash_atoms"]) == ["2.2", "3.0"]
        assert list(partner["clash_pairs"]) == ["2.2", "3.0"]

    def test_a_clean_row_is_ok_and_json_ready(self, result):
        assert [row["status"] for row in result["poses"]] == ["ok"] * 6
        assert result["nonfinite_fields"] == []
        json.dumps(result)

    def test_nothing_is_added_without_a_reference_that_asks_for_it(self, synthetic):
        implicit = score_pose_set(synthetic.poses[:2], "B", "A")
        assert "descriptors" not in implicit["poses"][0]
        assert "residue_table" not in implicit["poses"][0]
        plain_reference = score_pose_set(
            synthetic.poses[:2], "B", "A", reference=Reference(YCR, "A", "B")
        )
        assert "descriptors" not in plain_reference["poses"][0]

    def test_the_descriptors_follow_the_pose_numbering_the_frame_read(self, synthetic, reference):
        poses = [
            DockedPose.from_atoms(
                consecutive(consecutive(p.atoms(), "A"), "B"), id=p.id, **p.provenance
            )
            for p in synthetic.poses[:2]
        ]
        result = score_pose_set(poses, "B", "A", reference=reference, residue_table=True)
        block = result["poses"][0]["descriptors"]["reference"]["sets"]["epitope"]
        assert block["n_set_mapped"] == 22 and "reason" not in block
        numbers = {
            line["reference_number"]
            for line in result["poses"][0]["residue_table"]
            if line["role"] == "receptor"
        }
        assert {25, 26, 50} <= numbers
        original = score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)
        a = block["covered_fraction"]
        b = original["poses"][0]["descriptors"]["reference"]["sets"]["epitope"]["covered_fraction"]
        assert a == pytest.approx(b, abs=0.05)


class TestAggregatesAndFlatRows:
    def test_the_descriptor_columns_are_aggregated_per_group(self, result):
        key = "descriptors.reference.sets.epitope.covered_fraction"
        entry = result["aggregates"]["backend_a"][key]
        values = [
            row["descriptors"]["reference"]["sets"]["epitope"]["covered_fraction"]
            for row in result["poses"][:3]
        ]
        assert (entry["n_ok"], entry["n_total"]) == (3, 3)
        assert entry["mean"] == pytest.approx(np.mean(values))

    def test_a_key_with_a_dot_is_the_same_name_in_the_flat_row_and_the_aggregates(self, result):
        base = "descriptors.reference.partners.crystal_peptide.clash_atoms."
        flat = result.to_flat_rows()[0]
        assert flat[base + "2.2"] == 113 and flat[base + "3.0"] == 113
        entry = result["aggregates"]["backend_a"][base + "2.2"]
        assert (entry["n_ok"], entry["mean"]) == (3, 113.0)
        columns = {name for name in flat if name.startswith("descriptors.")}
        paths = {name for name in result["aggregates"]["all"] if name.startswith("descriptors.")}
        # the numeric columns are exactly the aggregate keys: one name for both
        numeric = {
            name
            for name in columns
            if isinstance(flat[name], (int, float)) and not isinstance(flat[name], bool)
        }
        assert paths == numeric

    def test_the_csv_has_the_dotted_columns_and_no_table(self, result, tmp_path):
        from binding_metrics.poses.report import write_pose_report

        header = (
            write_pose_report(result, "csv", tmp_path / "r.csv")
            .read_text(encoding="utf-8")
            .splitlines()[0]
            .split(",")
        )
        assert "descriptors.reference.partners.crystal_peptide.clash_atoms.2.2" in header
        assert not any(name.startswith("residue_table") for name in header)

    def test_the_markdown_summary_still_reads(self, result):
        assert isinstance(result.summary(), str)


class TestResidueTable:
    def test_the_columns_and_the_order_of_the_rows(self, result):
        table = result["poses"][0]["residue_table"]
        assert all(tuple(line) == TABLE_COLUMNS for line in table)
        roles = [line["role"] for line in table]
        assert roles == sorted(roles)  # binder first
        binder = [line for line in table if line["role"] == "binder"]
        assert [line["binder_index"] for line in binder] == list(range(1, 14))

    def test_the_reference_numbers_and_the_sets_of_the_receptor_residues(self, result):
        table = result["poses"][0]["residue_table"]
        by_number = {line["reference_number"]: line for line in table if line["role"] == "receptor"}
        assert by_number[25]["in_sets"] == ["epitope"] and by_number[25]["in_interface"] is True
        assert all(line["in_sets"] is None for line in table if line["role"] == "binder")

    def test_it_is_a_parameter_that_is_off_by_default(self, synthetic, reference):
        off = score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)
        assert "residue_table" not in off["poses"][0]
        assert off["parameters"]["residue_table"] is False
        assert (
            score_pose_set(synthetic.poses[:1], "B", "A", residue_table=True)["parameters"][
                "residue_table"
            ]
            is True
        )

    def test_without_a_reference_the_numbers_are_those_of_the_first_pose(self, synthetic):
        result = score_pose_set(synthetic.poses[:2], "B", "A", residue_table=True)
        table = result["poses"][1]["residue_table"]
        assert all(line["in_sets"] == [] for line in table if line["role"] == "receptor")
        assert any(line["reference_number"] == 25 for line in table)

    def test_the_flag_must_be_a_bool(self, synthetic):
        with pytest.raises(TypeError, match="residue_table"):
            score_pose_set(synthetic.poses[:2], "B", "A", residue_table="yes")

    def test_the_consensus_interface_reads_the_rows(self, result):
        a = consensus_interface(result["poses"], "backend_a")
        b = consensus_interface(result["poses"], "backend_b")
        assert (a.n_poses, b.n_poses) == (3, 3)
        assert ResidueKey("A", 25) in a.receptor and ResidueKey("A", 54) in a.receptor
        assert a.binder and b.binder
        comparison = compare_consensus_interfaces(a, b)
        assert 0.8 < comparison["jaccard"] <= 1.0
        # a failed pose does not vote
        rows = [dict(row) for row in result["poses"]]
        rows[0]["status"] = "failed"
        assert consensus_interface(rows, "backend_a").n_poses == 2

    def test_the_table_is_json_ready_and_survives_the_round_trip(self, result):
        again = PoseSetResult.from_dict(json.loads(result.to_json()))
        assert again["poses"][0]["residue_table"] == result["poses"][0]["residue_table"]


class TestOptions:
    def test_the_defaults_are_the_documented_ones(self, result):
        assert result["parameters"]["options"]["descriptors"] == {
            "interface_threshold_angstrom2": 0.5,
            "clash_cutoffs_angstrom": [2.2, 3.0],
            "shell_angstrom": 4.0,
            "max_unmapped_fraction": 0.5,
            "probe_radius": 1.4,
        }

    def test_a_larger_threshold_shrinks_the_interface(self, synthetic, reference, result):
        strict = score_pose_set(
            synthetic.poses[:2],
            "B",
            "A",
            reference=reference,
            options={"descriptors": {"interface_threshold_angstrom2": 20.0}},
        )
        small = strict["poses"][0]["descriptors"]["reference"]["sets"]["epitope"]["n_interface"]
        large = result["poses"][0]["descriptors"]["reference"]["sets"]["epitope"]["n_interface"]
        assert small < large

    def test_other_cutoffs_give_other_keys(self, synthetic, reference):
        result = score_pose_set(
            synthetic.poses[:2],
            "B",
            "A",
            reference=reference,
            options={"descriptors": {"clash_cutoffs_angstrom": [2.0, 3.5, 4.5]}},
        )
        partner = result["poses"][0]["descriptors"]["reference"]["partners"]["crystal_peptide"]
        assert list(partner["clash_atoms"]) == ["2.0", "3.5", "4.5"]
        flat = result.to_flat_rows()[0]
        assert "descriptors.reference.partners.crystal_peptide.clash_atoms.4.5" in flat

    def test_the_options_object_and_the_mapping_agree(self):
        options = DescriptorOptions(clash_cutoffs_angstrom=(2.5,))
        assert DescriptorOptions.from_dict(options.to_dict()) == options

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"nope": 1}, "unknown descriptors options"),
            ({"clash_cutoffs_angstrom": []}, "clash_cutoffs_angstrom"),
            ({"clash_cutoffs_angstrom": [0.0]}, "clash_cutoffs_angstrom"),
            ({"interface_threshold_angstrom2": -1}, "interface_threshold_angstrom2"),
            ({"shell_angstrom": 0}, "shell_angstrom"),
            ({"probe_radius": float("nan")}, "probe_radius"),
            ({"max_unmapped_fraction": 1.5}, "max_unmapped_fraction"),
        ],
    )
    def test_a_bad_option_raises_before_any_pose_is_read(self, synthetic, options, message):
        with pytest.raises(ValueError, match=message):
            score_pose_set(synthetic.poses[:2], "B", "A", options={"descriptors": options})


class TestMissingValues:
    def test_a_set_the_pose_lacks_too_much_of_has_a_reason_and_makes_the_row_partial(
        self, synthetic, reference
    ):
        atoms = drop(synthetic.poses[1].atoms(), "A", range(25, 80))
        trimmed = DockedPose.from_atoms(atoms, id="a_trimmed", backend="backend_a")
        result = score_pose_set(
            [synthetic.poses[0], trimmed],
            "B",
            "A",
            reference=reference,
            options={"frame": {"min_frame_fraction": 0.3}},
        )
        row = result.row("a_trimmed")
        block = row["descriptors"]["reference"]["sets"]["epitope"]
        assert row["status"] == "partial"
        assert block["covered_fraction"] is None and "not in the pose" in block["reason"]
        assert row["reasons"]["descriptors.reference.sets.epitope"] == block["reason"]
        entry = result["aggregates"]["backend_a"][
            "descriptors.reference.sets.epitope.covered_fraction"
        ]
        assert (entry["n_ok"], entry["n_total"]) == (1, 2)
        assert block["reason"] in entry["reasons"]

    def test_a_variant_whose_frame_failed_has_blocks_with_the_frame_reason(self, synthetic):
        residues = {"A": [ResidueKey("A", 20)]}
        trypsin = Reference(SFTI, "A", "I", residue_sets={"s": residues["A"]})
        result = score_pose_set(
            synthetic.poses[:2],
            "B",
            "A",
            reference={"mdm2": epitope_reference(), "trypsin": trypsin},
        )
        row = result["poses"][0]
        assert row["status"] == "partial" and "frame:trypsin" in row["reasons"]
        block = row["descriptors"]["trypsin"]["sets"]["s"]
        assert block["reason"].startswith("the frame failed")
        assert row["descriptors"]["mdm2"]["sets"]["epitope"]["covered_fraction"] > 0.9

    def test_a_pose_whose_binder_has_no_heavy_atom_has_reasons_and_no_table(
        self, synthetic, reference
    ):
        atoms = synthetic.poses[0].atoms().copy()
        atoms.res_name[atoms.chain_id == "B"] = "HOH"
        water = DockedPose.from_atoms(atoms, id="zz_water", backend="backend_a")
        result = score_pose_set(
            [synthetic.poses[0], water], "B", "A", reference=reference, residue_table=True
        )
        row = result.row("zz_water")
        assert row["status"] == "partial"
        assert (
            "residue_table" not in row
            and "no heavy polymer atom" in row["reasons"]["residue_table"]
        )
        assert (
            "no heavy polymer atom" in row["descriptors"]["reference"]["sets"]["epitope"]["reason"]
        )


@pytest.fixture(scope="module")
def variants():
    return {
        "strict": epitope_reference(),
        "same_site": Reference(
            YCR, "A", "B", residue_sets={"site_again": [ResidueKey("A", n) for n in EPITOPE]}
        ),
        "bare": Reference(YCR, "A", "B"),
    }


class TestVariants:
    def test_a_variant_without_sets_or_partners_has_no_descriptor_block(self, synthetic, variants):
        result = score_pose_set(synthetic.poses[:2], "B", "A", reference=variants)
        assert list(result["poses"][0]["descriptors"]) == ["strict", "same_site"]

    def test_variants_that_share_a_receptor_give_the_same_numbers(self, synthetic, variants):
        result = score_pose_set(synthetic.poses[:2], "B", "A", reference=variants)
        blocks = result["poses"][0]["descriptors"]
        assert blocks["strict"]["sets"]["epitope"] == blocks["same_site"]["sets"]["site_again"]

    def test_the_aggregates_are_keyed_by_variant(self, synthetic, variants):
        result = score_pose_set(synthetic.poses[:3], "B", "A", reference=variants)
        paths = result["aggregates"]["backend_a"]
        assert "descriptors.strict.sets.epitope.covered_fraction" in paths
        assert "descriptors.same_site.sets.site_again.covered_fraction" in paths
        assert "descriptors.strict.partners.crystal_peptide.shared_buried_area_angstrom2" in paths


class TestParallel:
    @pytest.mark.slow
    def test_a_pool_gives_the_same_descriptors_and_table(self, synthetic, reference, result):
        pooled = score_pose_set(
            synthetic.poses, "B", "A", reference=reference, residue_table=True, n_workers=2
        )
        a = json.loads(pooled.to_json(include_timing=False))
        b = json.loads(result.to_json(include_timing=False))
        a.pop("provenance"), b.pop("provenance")
        assert a == b


class TestReturnType:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"residue_table": True},
            {"include_transform": False},
            {"heavy": "all"},
            {"heavy": "skip"},
            {"preflight": "off"},
            {"on_error": "raise"},
            {"group_by": ()},
            {"representative": "confidence"},
            {"aggregate": ["mean"]},
            {"n_workers": 2},
            {"keep_tmp": False, "metrics": ["hbonds"]},
        ],
    )
    def test_the_return_type_never_depends_on_an_argument(self, synthetic, kwargs):
        result = score_pose_set(synthetic.poses[:3], "B", "A", **kwargs)
        assert type(result) is PoseSetResult
        assert isinstance(result.summary(), str)
        assert isinstance(result.to_flat_rows(), list)
