"""score_pose_set against a reference: numbering, variants, real complexes, predicted poses."""

import json
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics._common import load_structure
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose
from binding_metrics.poses.frame import Transform
from binding_metrics.poses.reference import Reference, ResidueKey
from binding_metrics.poses.score import score_pose_set
from binding_metrics.testing import make_synthetic_poses
from tests.poses.test_frame import consecutive, drop

struc = pytest.importorskip("biotite.structure")

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"

# MDM2 residues within 4.5 angstrom of the crystal peptide (design note, section 1.2).
EPITOPE = [25, 26, 50, 51, 54, 57, 58, 61, 62, 67, 70, 71, 72, 73, 75, 91, 93, 94, 96, 99, 100, 104]


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def reference():
    return Reference(YCR, "A", "B", residue_sets={"epitope": [ResidueKey("A", n) for n in EPITOPE]})


def renumbered(pose):
    """The pose with both chains numbered from 1, as the co-folding models write them."""
    atoms = consecutive(consecutive(pose.atoms(), "A"), "B")
    return DockedPose.from_atoms(atoms, id=pose.id, **pose.provenance)


@pytest.fixture(scope="module")
def renumbered_poses(synthetic):
    return [renumbered(pose) for pose in synthetic.poses]


# ---------------------------------------------------------------------------
# an explicit reference
# ---------------------------------------------------------------------------


class TestExplicitReference:
    @pytest.mark.slow
    def test_a_pose_renumbered_from_1_is_paired_by_sequence(self, renumbered_poses, reference):
        result = score_pose_set(renumbered_poses, "B", "A", reference=reference)
        assert result["references"]["implicit"] is False
        assert result["references"]["metrics_frame_variant"] == "reference"
        for row in result["poses"]:
            frame = row["frame"]
            assert row["status"] == "ok"
            assert frame["pairing"] == "sequence"
            assert (frame["n_paired"], frame["paired_fraction"]) == (85, 1.0)
            assert frame["receptor_rmsd_angstrom"] < 1e-3
            assert set(frame["region_rmsd_angstrom"]) == {"all", "site", "epitope"}
            assert frame["region_n_paired"]["epitope"] == 22

    @pytest.mark.slow
    def test_the_agreement_does_not_depend_on_the_reference_or_the_numbering(
        self, synthetic, renumbered_poses, reference
    ):
        implicit = score_pose_set(synthetic.poses, "B", "A")
        explicit = score_pose_set(renumbered_poses, "B", "A", reference=reference)
        for kind in ("ca", "backbone", "heavy"):
            a = np.array(implicit["agreement"]["rmsd"][kind]["matrix"]["matrix"], dtype=float)
            b = np.array(explicit["agreement"]["rmsd"][kind]["matrix"]["matrix"], dtype=float)
            assert np.allclose(a, b, atol=1e-4)
        assert implicit["agreement"]["clusters"] == explicit["agreement"]["clusters"]

    def test_the_frame_numbers_are_those_of_the_original_frame(self, synthetic, reference):
        result = score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)
        assert result["poses"][0]["frame"]["pairing"] == "number"

    def test_a_trimmed_receptor_is_paired_on_what_it_has(self, renumbered_poses, reference):
        atoms = renumbered_poses[1].atoms()
        atoms = drop(drop(atoms, "A", range(1, 6)), "A", range(40, 44))
        trimmed = DockedPose.from_atoms(atoms, id="a_trimmed", backend="backend_a")
        result = score_pose_set(
            [renumbered_poses[0], trimmed], "B", "A", reference=reference, metrics=["hbonds"]
        )
        frame = result.row("a_trimmed")["frame"]
        assert result.row("a_trimmed")["status"] == "ok"
        assert frame["pairing"] == "sequence"
        assert (frame["n_receptor_residues_pose"], frame["n_paired"]) == (76, 76)
        assert frame["paired_fraction"] == pytest.approx(76 / 85)
        assert frame["receptor_rmsd_angstrom"] < 1e-3
        # the epitope residues that the pose lost are not counted as paired
        assert frame["region_n_paired"]["epitope"] < 22
        assert result.row("a_trimmed")["hbonds"]["hbonds"] >= 0

    def test_the_reference_numbering_reaches_the_labels_of_the_metrics(
        self, renumbered_poses, reference
    ):
        result = score_pose_set(
            renumbered_poses[:1], "B", "A", reference=reference, metrics=["interface"]
        )
        labels = result["poses"][0]["interface"]["interface_residues_receptor"]
        assert "GLU:A:25" in labels and "MET:A:50" in labels

    def test_renumber_receptor_false_keeps_the_numbers_of_the_pose(
        self, renumbered_poses, reference
    ):
        result = score_pose_set(
            renumbered_poses[:1],
            "B",
            "A",
            reference=reference,
            metrics=["interface"],
            options={"frame": {"renumber_receptor": False}},
        )
        labels = result["poses"][0]["interface"]["interface_residues_receptor"]
        assert "GLU:A:1" in labels and "GLU:A:25" not in labels

    def test_a_per_pose_residue_map_replaces_the_pairing(self, renumbered_poses, reference):
        pose = renumbered_poses[0]
        mapping = {ResidueKey("A", n - 24): ResidueKey("A", n) for n in range(25, 110)}
        result = score_pose_set(
            [pose], "B", "A", reference=reference, residue_maps={pose.id: mapping}
        )
        frame = result["poses"][0]["frame"]
        assert (frame["pairing"], frame["n_paired"]) == ("map", 85)
        assert result["parameters"]["residue_maps"][pose.id][0] == [["A", 1, ""], ["A", 25, ""]]

    def test_the_fit_can_be_restricted_to_a_residue_set(self, synthetic, reference):
        result = score_pose_set(
            synthetic.poses[:2],
            "B",
            "A",
            reference=reference,
            options={"frame": {"frame_residues": "epitope"}},
        )
        assert result["poses"][0]["frame"]["n_fit_residues"] == 22

    def test_the_transform_places_the_receptor_on_the_reference(self, synthetic, reference):
        result = score_pose_set(synthetic.poses[3:4], "B", "A", reference=reference)
        transform = Transform.from_dict(result["poses"][0]["frame"]["transform"])
        pose_atoms = synthetic.poses[3].atoms()
        crystal = load_structure(YCR)
        is_ca = lambda a: (a.chain_id == "A") & (a.atom_name == "CA")  # noqa: E731
        moved = transform.apply(pose_atoms.coord[is_ca(pose_atoms)])
        assert np.abs(moved - crystal.coord[is_ca(crystal)]).max() < 2e-3

    def test_the_transform_can_be_left_out_of_the_rows(self, synthetic, reference):
        result = score_pose_set(
            synthetic.poses[:2], "B", "A", reference=reference, include_transform=False
        )
        assert "transform" not in result["poses"][0]["frame"]
        assert result["parameters"]["include_transform"] is False

    def test_the_reference_is_recorded_with_its_hash(self, synthetic, reference):
        result = score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)
        variant = result["references"]["variants"]["reference"]
        assert variant["content_hash"] == reference.content_hash()
        assert variant["reference"]["residue_sets"]["epitope"][0] == ["A", 25, ""]
        json.dumps(result)

    def test_a_reference_that_is_a_pose_of_another_kind_works(self, synthetic):
        crystal = ExperimentalPose.from_file(YCR, method="X-RAY DIFFRACTION")
        reference = Reference(crystal, "A", "B")
        result = score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)
        assert result["poses"][0]["frame"]["variant"] == "reference"

    def test_a_metric_that_compares_with_the_reference_gets_its_file(self, synthetic, reference):
        result = score_pose_set(
            synthetic.poses[:3], "B", "A", reference=reference, metrics=["structure_rmsd"]
        )
        rmsd = [row["structure_rmsd"]["rmsd"] for row in result["poses"]]
        # the poses are the crystal with the peptide shifted by 0, 0.4 and 0.8 angstrom
        assert rmsd[0] < 1e-3 < rmsd[1] < rmsd[2]
        assert all(row["structure_rmsd"]["rmsd_design"] < 1e-3 for row in result["poses"])
        assert [row["status"] for row in result["poses"]] == ["ok"] * 3


# ---------------------------------------------------------------------------
# several reference variants in one call: one frame per distinct receptor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moved_crystal():
    crystal = load_structure(YCR)
    angle = np.radians(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    )
    moved = crystal.copy()
    moved.coord = (crystal.coord @ rotation.T + np.array([5.0, -3.0, 2.0])).astype(np.float32)
    return moved


@pytest.fixture(scope="module")
def variants(moved_crystal):
    return {
        "strict": Reference(
            YCR, "A", "B", residue_sets={"epitope": [ResidueKey("A", n) for n in EPITOPE]}
        ),
        "loose": Reference(YCR, "A", "B"),
        "moved": Reference(ExperimentalPose.from_atoms(moved_crystal), "A", "B"),
    }


@pytest.fixture(scope="module")
def variants_result(synthetic, variants):
    return score_pose_set(synthetic.poses[:3], "B", "A", reference=variants)


class TestReferenceVariants:
    def test_every_variant_has_a_frame_block_and_the_first_is_the_row_frame(self, variants_result):
        frame = variants_result["poses"][0]["frame"]
        assert frame["variant"] == "strict"
        assert list(frame["variants"]) == ["strict", "loose", "moved"]
        assert frame["variants"]["strict"]["pairing"] == frame["pairing"]
        assert variants_result["references"]["metrics_frame_variant"] == "strict"
        assert list(variants_result["references"]["variants"]) == ["strict", "loose", "moved"]

    def test_variants_with_one_receptor_share_a_transform_and_another_receptor_has_its_own(
        self, variants_result
    ):
        blocks = variants_result["poses"][0]["frame"]["variants"]
        strict, loose, moved = (
            Transform.from_dict(blocks[name]["transform"]) for name in ("strict", "loose", "moved")
        )
        assert strict.allclose(loose, atol=1e-9)
        assert not strict.allclose(moved, atol=1e-3)

    def test_each_transform_places_the_pose_on_its_own_receptor(
        self, variants_result, synthetic, moved_crystal
    ):
        pose_atoms = synthetic.poses[0].atoms()
        is_ca = lambda a: (a.chain_id == "A") & (a.atom_name == "CA")  # noqa: E731
        blocks = variants_result["poses"][0]["frame"]["variants"]
        crystal = load_structure(YCR)
        for name, target in (("strict", crystal), ("moved", moved_crystal)):
            moved = Transform.from_dict(blocks[name]["transform"]).apply(
                pose_atoms.coord[is_ca(pose_atoms)]
            )
            assert np.abs(moved - target.coord[is_ca(target)]).max() < 2e-3, name

    def test_a_region_belongs_to_the_variant_that_names_it(self, variants_result):
        blocks = variants_result["poses"][0]["frame"]["variants"]
        assert "epitope" in blocks["strict"]["region_rmsd_angstrom"]
        assert "epitope" not in blocks["loose"]["region_rmsd_angstrom"]

    def test_the_metrics_run_in_the_frame_of_the_first_variant(self, synthetic, variants, tmp_path):
        first_only = {"loose": variants["loose"], "moved": variants["moved"]}
        result = score_pose_set(
            synthetic.poses[:2],
            "B",
            "A",
            reference=first_only,
            metrics=["delta_sasa_static"],
            tmp_dir=tmp_path,
            keep_tmp=True,
        )
        kept = Path(result["provenance"]["temp_dir"])
        placed = load_structure(kept / "pose_0001.cif")
        crystal = load_structure(YCR)
        is_ca = lambda a: (a.chain_id == "A") & (a.atom_name == "CA")  # noqa: E731
        assert np.abs(placed.coord[is_ca(placed)] - crystal.coord[is_ca(crystal)]).max() < 2e-3

    def test_a_variant_that_cannot_place_a_pose_leaves_a_partial_row(self, synthetic):
        # trypsin is not the receptor of these poses: that variant fails, the first one does not
        trypsin = Reference(SFTI, "A", "I")
        result = score_pose_set(
            synthetic.poses[:3],
            "B",
            "A",
            reference={"mdm2": Reference(YCR, "A", "B"), "trypsin": trypsin},
        )
        row = result["poses"][0]
        assert row["status"] == "partial"
        assert "frame:trypsin" in row["reasons"] and "frame" not in row["reasons"]
        assert row["frame"]["variants"]["trypsin"]["status"] == "failed"
        assert row["frame"]["variants"]["mdm2"]["status"] == "ok"
        assert result["agreement"]["status"] == "ok"

    def test_the_input_key_depends_on_the_variants(self, synthetic, variants, variants_result):
        one = score_pose_set(synthetic.poses[:3], "B", "A", reference=variants["strict"])
        assert one.input_key != variants_result.input_key

    def test_a_single_variant_has_no_variants_block(self, synthetic, variants):
        reference = Reference(YCR, "A", "B", name="loose")
        result = score_pose_set(synthetic.poses[:2], "B", "A", reference=reference)
        assert "variants" not in result["poses"][0]["frame"]
        assert result["poses"][0]["frame"]["variant"] == "loose"

    def test_a_mapping_with_a_wrong_value_raises(self, synthetic):
        with pytest.raises(TypeError, match="must be a Reference"):
            score_pose_set(synthetic.poses[:2], "B", "A", reference={"x": 3})


# ---------------------------------------------------------------------------
# real complexes from data/
# ---------------------------------------------------------------------------


def three_poses(path, binder):
    """The complex as it is, rotated and shifted as a whole, and with the peptide moved 0.5 A."""
    atoms = load_structure(path)
    angle = np.radians(40.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    )
    rotated = atoms.copy()
    rotated.coord = (atoms.coord @ rotation.T + np.array([10.0, -7.0, 3.0])).astype(np.float32)
    shifted = atoms.copy()
    shifted.coord[shifted.chain_id == binder] += np.array([0.5, 0.0, 0.0], dtype=np.float32)
    return [
        DockedPose.from_atoms(atoms, id="orig", backend="a"),
        DockedPose.from_atoms(rotated, id="rot", backend="b"),
        DockedPose.from_atoms(shifted, id="shift", backend="b"),
    ]


class TestRealComplexes:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "path, binder, target, closures, classes",
        [
            pytest.param(
                SFTI, "I", "A", {"head_to_tail", "disulfide"}, {"canonical"}, marks=pytest.mark.slow
            ),
            (
                CYCLOSPORIN,
                "C",
                "A",
                {"head_to_tail"},
                {"canonical", "d_amino", "n_methyl", "other_ncaa"},
            ),
        ],
    )
    def test_a_cyclic_peptide_with_hetero_atoms_is_scored_like_a_plain_call(
        self, path, binder, target, closures, classes
    ):
        from binding_metrics.metrics.sasa import compute_delta_sasa_static

        result = score_pose_set(
            three_poses(path, binder),
            binder,
            target,
            metrics=["delta_sasa_static", "ramachandran", "omega"],
        )
        assert [row["status"] for row in result["poses"]] == ["ok"] * 3
        profile = result.row("orig")["profile"]
        assert set(profile["closures"]) == closures
        assert set(profile["residue_classes"]) == classes
        assert profile["closures_as_expected"] is True
        # the rotated copy is placed back on the original: its numbers are the original's
        plain = compute_delta_sasa_static(path, peptide_chain=binder, receptor_chain=target)
        for pose_id in ("orig", "rot"):
            block = result.row(pose_id)["delta_sasa_static"]
            assert block["delta_sasa"] == pytest.approx(plain["delta_sasa"], rel=2e-3)
        # the peptide moved by 0.5 angstrom is 0.5 angstrom from the two others, for every atom set
        for kind in ("ca", "backbone", "heavy"):
            matrix = np.array(result["agreement"]["rmsd"][kind]["matrix"]["matrix"], dtype=float)
            ids = result["agreement"]["rmsd"][kind]["matrix"]["pose_ids"]
            shift = ids.index("shift")
            others = [i for i in range(3) if i != shift]
            assert np.allclose(matrix[shift, others], 0.5, atol=2e-3), kind
            assert matrix[ids.index("orig"), ids.index("rot")] < 2e-3
        assert result["agreement"]["expected_closures"] == sorted(closures)

    def test_the_residues_without_a_symmetry_entry_are_listed(self):
        result = score_pose_set(three_poses(CYCLOSPORIN, "C"), "C", "A")
        assert result["agreement"]["unknown_residues"] == ["ABA", "BMT"]

    @pytest.mark.parametrize(
        "name", ["example_lactam_somatostatin_1XY4.cif", "example_phospho_1QJB.pdb"]
    )
    def test_a_structure_with_no_receptor_is_a_failed_row_with_a_reason(self, name):
        result = score_pose_set([DockedPose.from_file(DATA / name)], metrics=["hbonds"])
        row = result["poses"][0]
        assert row["status"] == "failed" and row["reason"]
        assert result["agreement"]["status"] == "skipped"
        json.dumps(result)

    def test_a_pose_set_of_one_structure_against_itself_has_the_identity_frame(self):
        pose = DockedPose.from_file(YCR, id="crystal", backend="x")
        result = score_pose_set(
            [pose], "B", "A", reference=Reference(YCR, "A", "B"), metrics=["delta_sasa_static"]
        )
        frame = result["poses"][0]["frame"]
        assert frame["receptor_rmsd_angstrom"] < 1e-6
        assert np.allclose(frame["transform"]["rotation"], np.eye(3), atol=1e-9)
        assert np.allclose(frame["transform"]["translation"], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# predicted poses built from the adapter fixtures
# ---------------------------------------------------------------------------


def build_predicted_poses(root):
    """Two Boltz-2 samples of the real 1YCR complex, written under ``root`` and parsed.

    The writer numbers every chain from 1, as Boltz does, so the crystal reference needs the
    sequence pairing; the confidences are made-up round numbers.
    """
    from binding_metrics.predictors import get_parser
    from tests.predictors import synth, synth_boltz2

    atoms = load_structure(YCR)
    atoms = atoms[(~atoms.hetero) & np.isin(atoms.chain_id, ["A", "B"]) & (atoms.element != "H")]
    n_tokens = int(synth_boltz2.token_index_of_atoms(atoms).max()) + 1
    poses = []
    for name, shift in (("sampleone", 0.0), ("sampletwo", 8.0)):
        plddt = np.linspace(60.0, 95.0, atoms.array_length()) - shift
        sample_atoms = atoms.copy()
        sample_atoms.set_annotation("b_factor", plddt)
        i, j = np.arange(n_tokens)[:, None], np.arange(n_tokens)[None, :]
        truth = synth.SyntheticComplex(
            atoms=sample_atoms,
            plddt_per_atom=plddt,
            pae=1.0 + 0.01 * i + 0.01 * j,
            pde=0.5 + 0.01 * i + 0.005 * j,
            scalars={
                "avg_plddt": float(plddt.mean()),
                "ptm": 0.88,
                "iptm": 0.76,
                "gpde": 1.2,
                "ranking_score": 0.82 - shift / 100,
                "has_clash": 0.0,
                "disorder": 0.1,
            },
            chain_ptm={"A": 0.88, "B": 0.8},
            chain_pair_iptm={"A-B": 0.76, "B-A": 0.74},
        )
        synth_boltz2.write_prediction(root, name, truth)
        poses.append(get_parser("boltz2").load(root, name))
    return poses


@pytest.fixture(scope="module")
def predicted_poses(tmp_path_factory):
    return build_predicted_poses(tmp_path_factory.mktemp("boltz"))


@pytest.fixture(scope="module")
def predicted_result(predicted_poses):
    return score_pose_set(
        predicted_poses,
        "B",
        "A",
        metrics=["delta_sasa_static", "prediction"],
        reference=Reference(YCR, "A", "B"),
    )


class TestPredictedPoses:
    def test_the_group_is_the_model_and_the_provenance_says_where_the_pose_came_from(
        self, predicted_result
    ):
        row = predicted_result["poses"][0]
        assert row["kind"] == "predicted"
        assert row["id"] == "boltz2:sampleone:s1:1"
        assert row["provenance"] == {
            "backend": "boltz2",
            "name": "sampleone",
            "seed_index": 1,
            "sample": 1,
        }
        assert list(predicted_result["aggregates"]) == ["boltz2", "all"]
        assert predicted_result["representatives"]["boltz2"] in {
            p["id"] for p in predicted_result["poses"]
        }

    def test_the_native_scores_are_on_the_row_and_in_the_aggregates(
        self, predicted_result, predicted_poses
    ):
        row = predicted_result["poses"][0]
        assert row["native"] == predicted_poses[0].native_scores()
        entry = predicted_result["aggregates"]["boltz2"]["native.ranking_score"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 2)
        assert entry["max"] == pytest.approx(0.82) and entry["min"] == pytest.approx(0.74)

    def test_the_prediction_block_is_served_from_the_pose(self, predicted_result):
        block = predicted_result["poses"][0]["prediction"]
        assert block["model"] == "boltz2" and block["query_name"] == "sampleone"
        assert isinstance(block["plddt_per_atom"], list)
        assert 60.0 < block["binder_avg_plddt"] < 95.0
        assert block["mean_interface_pae"] == pytest.approx(1.0 + 0.01 * 49 + 0.01 * 49, abs=0.5)
        second = predicted_result["poses"][1]["prediction"]
        assert second["binder_avg_plddt"] == pytest.approx(
            block["binder_avg_plddt"] - 8.0, abs=0.01
        )
        json.dumps(predicted_result)

    def test_the_renumbered_model_is_placed_on_the_crystal_by_sequence(self, predicted_result):
        for row in predicted_result["poses"]:
            assert row["status"] == "ok"
            assert row["frame"]["pairing"] == "sequence"
            assert row["frame"]["n_paired"] == 85
            assert row["frame"]["receptor_rmsd_angstrom"] < 2e-3

    def test_the_two_samples_are_the_same_structure(self, predicted_result):
        matrix = np.array(
            predicted_result["agreement"]["rmsd"]["heavy"]["matrix"]["matrix"], dtype=float
        )
        assert matrix[0, 1] < 2e-3

    def test_a_predicted_pose_without_a_structure_is_a_failed_row_with_the_adapter_reason(
        self, predicted_poses
    ):
        from dataclasses import replace

        bare = replace(
            predicted_poses[0],
            structure_path=None,
            given_id="bare",
            reasons=["no structure written"],
        )
        result = score_pose_set(
            [predicted_poses[1], bare], "B", "A", reference=Reference(YCR, "A", "B")
        )
        row = result.row("bare")
        assert row["status"] == "failed" and "no structure file" in row["reason"]
        assert row["content_hash"] is None
        assert row["native"] == {}
