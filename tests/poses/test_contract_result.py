"""Contract of the pose-set result: layout, determinism, order, side effects, inputs, input key.

``test_score.py``, ``test_report.py`` and the other pose tests check what each stage computes.
This module pins what an engine relies on when it stores the result: the exact keys at each
level (a committed list, not the constants of the code), the reserved row names, that the same
input gives the same bytes, that the order of the poses does not matter, that a call writes
nothing outside the directories it was given, that it changes none of its inputs, and what the
``input_key`` does and does not depend on.

Versioning rule (design 9.5): adding a key does not change ``schema_version``; renaming or
removing one, or changing a meaning or a unit, does. The pinned lists below fail on both, with
a message that says which of the two it is.
"""

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from binding_metrics import provenance as provenance_module
from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses import (
    Partner,
    PoseSetResult,
    Reference,
    ResidueKey,
    score_pose_set,
    write_pose_report,
)
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.score import RESERVED_ROW_NAMES, SCHEMA, SCHEMA_VERSION
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")

METRICS = ("delta_sasa_static", "omega")
SITE = (25, 26, 50, 51, 54, 57, 58, 61, 62, 67)


def pin(actual, pinned, where):
    """Assert the keys of one block are exactly ``pinned``, in order, with a pointed message."""
    actual, pinned = list(actual), list(pinned)
    removed = [key for key in pinned if key not in actual]
    added = [key for key in actual if key not in pinned]
    assert not removed, (
        f"{where}: {removed} removed or renamed. That changes schema_version {SCHEMA_VERSION}: "
        "bump SCHEMA_VERSION and list the change in the changelog."
    )
    assert not added, (
        f"{where}: new key(s) {added}. Adding a key does not change schema_version; add it to "
        "the pinned list in this test and to docs/pose_sets.md."
    )
    assert actual == pinned, f"{where}: the keys are in another order: {actual}"


class TestThePinHelper:
    def test_it_accepts_the_same_keys_in_the_same_order(self):
        pin({"a": 1, "b": 2}, ["a", "b"], "block")

    def test_a_removed_key_says_that_the_schema_version_changes(self):
        with pytest.raises(
            AssertionError, match=r"\['b'\] removed or renamed.*bump SCHEMA_VERSION"
        ):
            pin({"a": 1}, ["a", "b"], "block")

    def test_an_added_key_says_that_the_version_stays_and_the_list_grows(self):
        with pytest.raises(AssertionError, match=r"new key\(s\) \['c'\].*does not change"):
            pin({"a": 1, "c": 3}, ["a"], "block")

    def test_another_order_is_a_failure_too(self):
        with pytest.raises(AssertionError, match="another order"):
            pin(["b", "a"], ["a", "b"], "block")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses(seeds=(1, 2))


def reference_of(synthetic):
    return Reference(
        synthetic.reference,
        "A",
        "B",
        residue_sets={"epitope": [ResidueKey("A", n) for n in SITE]},
        partners={"crystal": Partner(synthetic.reference, chains=["B"])},
    )


def score(synthetic, poses=None, **kwargs):
    kwargs.setdefault("metrics", METRICS)
    kwargs.setdefault("reference", reference_of(synthetic))
    kwargs.setdefault("residue_table", True)
    return score_pose_set(synthetic.poses if poses is None else poses, "B", "A", **kwargs)


@pytest.fixture(scope="module")
def result(synthetic):
    return score(synthetic)


def comparable(result):
    """The result as JSON text without the timing block: the run-to-run invariant."""
    return result.to_json(include_timing=False)


# ---------------------------------------------------------------------------
# the layout, pinned
# ---------------------------------------------------------------------------


class TestPinnedLayout:
    def test_the_reserved_row_names_are_the_thirteen_of_the_design(self):
        assert RESERVED_ROW_NAMES == (
            "id",
            "kind",
            "content_hash",
            "provenance",
            "native",
            "status",
            "reason",
            "reasons",
            "independent_of_design",
            "profile",
            "frame",
            "descriptors",
            "residue_table",
        )
        assert not set(RESERVED_ROW_NAMES) & set(METRICS_BY_NAME)

    def test_the_schema_name_and_version(self, result):
        assert SCHEMA == "binding_metrics.pose_set_result"
        assert SCHEMA_VERSION == 1 and type(SCHEMA_VERSION) is int
        assert result["schema"] == SCHEMA
        assert result["schema_version"] == 1 and type(result["schema_version"]) is int

    def test_a_result_of_another_schema_version_is_refused_by_from_dict(self, result):
        for version in (0, 2, "1", None):
            data = result.to_dict()
            data["schema_version"] = version
            with pytest.raises(ValueError, match="schema_version"):
                PoseSetResult.from_dict(data)

    def test_the_top_level_keys(self, result):
        pin(
            result,
            [
                "schema",
                "schema_version",
                "package_version",
                "construct",
                "binder_chain",
                "target_chain",
                "input_key",
                "provenance",
                "parameters",
                "references",
                "preflight",
                "poses",
                "agreement",
                "aggregates",
                "representatives",
                "warnings",
                "nonfinite_fields",
                "timing",
            ],
            "result",
        )

    def test_the_provenance_the_layer_adds_and_the_run_block(self, result):
        provenance = result["provenance"]
        assert {"package_version", "git_sha", "pose_set_schema_version", "n_poses", "run"} <= set(
            provenance
        )
        assert provenance["pose_set_schema_version"] == SCHEMA_VERSION
        pin(
            provenance["run"],
            ["n_workers", "start_method", "on_error", "tmp_dir", "keep_tmp"],
            "provenance.run",
        )

    def test_the_parameters_block_and_the_name_of_every_option(self, result):
        parameters = result["parameters"]
        pin(
            parameters,
            [
                "options",
                "metrics",
                "metric_scope",
                "metric_kwargs",
                "aggregate",
                "group_by",
                "representative",
                "heavy",
                "preflight",
                "residue_maps",
                "residue_table",
                "include_transform",
            ],
            "parameters",
        )
        options = parameters["options"]
        pin(options, ["frame", "agreement", "aggregation", "representative", "descriptors"], "o")
        pin(
            options["frame"],
            [
                "pairing",
                "frame_atoms",
                "frame_residues",
                "outliers",
                "min_frame_residues",
                "min_frame_fraction",
                "min_sequence_identity",
                "min_pair_fraction",
                "site_cutoff_angstrom",
                "iqr_threshold",
                "iqr_max_iterations",
                "iqr_min_deviation_angstrom",
                "alignment_matrix",
                "alignment_gap_penalty",
                "renumber_receptor",
                "metrics_frame",
            ],
            "options.frame",
        )
        pin(
            options["agreement"],
            [
                "kind",
                "cluster_cutoffs_angstrom",
                "linkage",
                "symmetry",
                "cyclic_shift",
                "residue_pairing",
                "expected_closures",
                "exclude_closure_mismatch",
                "contact_cutoff_angstrom",
                "min_shared_atom_fraction",
                "shape_rmsd",
            ],
            "options.agreement",
        )
        pin(options["aggregation"], ["min_n_ok", "reason_policy"], "options.aggregation")
        pin(
            options["representative"],
            ["confidence_key", "confidence_direction"],
            "options.representative",
        )
        pin(
            options["descriptors"],
            [
                "interface_threshold_angstrom2",
                "clash_cutoffs_angstrom",
                "shell_angstrom",
                "max_unmapped_fraction",
                "probe_radius",
            ],
            "options.descriptors",
        )

    def test_the_references_and_the_preflight_blocks(self, result):
        references = result["references"]
        pin(
            references,
            ["implicit", "implicit_pose_id", "metrics_frame_variant", "variants"],
            "references",
        )
        variant = references["variants"]["reference"]
        pin(variant, ["content_hash", "reference"], "references.variants.reference")
        pin(
            variant["reference"],
            [
                "schema",
                "schema_version",
                "name",
                "structure",
                "receptor_chain",
                "binder_chain",
                "residue_map",
                "residue_sets",
                "partners",
            ],
            "references.variants.reference.reference",
        )
        pin(
            result["preflight"],
            [
                "policy",
                "predictor_policy",
                "mode",
                "compatible",
                "profile",
                "metrics_requested",
                "metrics_to_run",
                "metrics_skipped",
                "predictors",
                "predictor_usable",
                "violations",
                "warnings",
                "notes",
            ],
            "preflight",
        )

    def test_a_design_source_adds_one_block_to_parameters_and_to_preflight_and_nothing_else(
        self, synthetic, result
    ):
        # Adding keys does not change schema_version. They are present only with a source, so the
        # default layout above, the golden results and every reader of them stay as they are.
        with_source = score(synthetic, design_source="boltzgen")
        assert list(with_source) == list(result)
        pin(
            with_source["parameters"],
            [*result["parameters"], "design"],
            "parameters with a design source",
        )
        pin(
            with_source["parameters"]["design"],
            ["source", "predictors", "policy"],
            "parameters.design",
        )
        pin(
            with_source["preflight"],
            [*result["preflight"], "design"],
            "preflight with a design source",
        )
        pin(
            with_source["preflight"]["design"],
            ["source", "known", "predictors", "policy", "models", "not_independent", "checks"],
            "preflight.design",
        )
        assert [list(row) for row in with_source["poses"]] == [list(row) for row in result["poses"]]
        assert with_source["schema_version"] == result["schema_version"] == 1

    def test_a_row_and_its_frame_profile_and_descriptor_blocks(self, result):
        row = result["poses"][0]
        pin(
            row,
            [
                "id",
                "kind",
                "content_hash",
                "provenance",
                "native",
                "status",
                "reason",
                "reasons",
                "independent_of_design",
                "profile",
                "frame",
                *METRICS,
                "descriptors",
                "residue_table",
            ],
            "row",
        )
        pin(
            row["frame"],
            [
                "status",
                "reason",
                "pairing",
                "n_receptor_residues_pose",
                "n_receptor_residues_reference",
                "n_paired",
                "paired_fraction",
                "sequence_identity",
                "alignment_ambiguous",
                "n_fit_residues",
                "n_fit_atoms",
                "n_rejected",
                "receptor_rmsd_angstrom",
                "region_rmsd_angstrom",
                "region_n_residues",
                "region_n_paired",
                "region_reasons",
                "warnings",
                "transform",
                "variant",
            ],
            "row.frame",
        )
        pin(row["frame"]["transform"], ["rotation", "translation"], "row.frame.transform")
        pin(
            row["profile"],
            [
                "n_binder_residues",
                "binder_type",
                "closures",
                "closure_bonds",
                "residue_classes",
                "residue_names",
                "notes",
                "closures_as_expected",
                "differs_from_first",
            ],
            "row.profile",
        )
        block = row["descriptors"]["reference"]
        pin(block, ["sets", "partners"], "row.descriptors.<variant>")
        pin(
            block["sets"]["epitope"],
            [
                "n_set",
                "n_set_mapped",
                "n_interface",
                "n_overlap",
                "covered_fraction",
                "interface_in_set_fraction",
                "jaccard",
                "receptor_buried_area_in_set_angstrom2",
                "receptor_buried_area_in_set_fraction",
            ],
            "row.descriptors.<variant>.sets.<set>",
        )
        pin(
            block["partners"]["crystal"],
            [
                "min_distance_angstrom",
                "clash_atoms",
                "clash_pairs",
                "shell_fraction",
                "shared_buried_area_angstrom2",
                "partner_buried_area_angstrom2",
                "pose_buried_area_angstrom2",
                "shared_fraction_of_partner",
                "overlap_volume_angstrom3",
            ],
            "row.descriptors.<variant>.partners.<partner>",
        )
        pin(
            row["residue_table"][0],
            [
                "chain",
                "role",
                "number",
                "icode",
                "name",
                "binder_index",
                "reference_chain",
                "reference_number",
                "reference_icode",
                "buried_area_angstrom2",
                "polar_area_angstrom2",
                "apolar_area_angstrom2",
                "solvation_dg_kcal_mol",
                "in_interface",
                "in_sets",
            ],
            "row.residue_table[]",
        )

    def test_the_agreement_block(self, result):
        agreement = result["agreement"]
        pin(
            agreement,
            [
                "status",
                "reason",
                "kind",
                "linkage",
                "pose_ids",
                "groups",
                "excluded",
                "expected_closures",
                "rmsd",
                "shape_rmsd",
                "unknown_residues",
                "closure_mismatch_pairs",
                "medoids",
                "clusters",
                "spread",
                "contacts",
            ],
            "agreement",
        )
        for name in ("rmsd", "shape_rmsd"):
            pin(agreement[name], ["ca", "backbone", "heavy"], f"agreement.{name}")
        block = agreement["rmsd"]["ca"]
        pin(block, ["matrix", "statistics"], "agreement.rmsd.ca")
        pin(
            block["matrix"],
            ["name", "unit", "direction", "pose_ids", "matrix", "undefined_pairs"],
            "agreement.rmsd.ca.matrix",
        )
        pin(
            block["statistics"],
            ["n_poses", "overall", "within", "between", "between_reason"],
            "agreement.rmsd.ca.statistics",
        )
        medoids = agreement["medoids"]
        pin(medoids, ["overall", "per_group", "between_groups"], "agreement.medoids")
        pin(
            medoids["overall"],
            [
                "pose_id",
                "sum_distance_angstrom",
                "mean_distance_angstrom",
                "tied_with",
                "n_poses",
                "excluded",
            ],
            "agreement.medoids.overall",
        )
        pin(
            medoids["between_groups"]["backend_a|backend_b"],
            ["medoid_a", "medoid_b", "distance_angstrom"],
            "agreement.medoids.between_groups.<a|b>",
        )
        pin(
            agreement["clusters"]["1.0"],
            [
                "cutoff_angstrom",
                "linkage",
                "n_clusters",
                "sizes",
                "largest_cluster_fraction",
                "labels",
                "clusters",
                "mixed_clusters",
                "n_undefined_pairs",
            ],
            "agreement.clusters.<cutoff>",
        )
        spread = agreement["spread"]
        pin(spread, ["all", "per_group"], "agreement.spread")
        pin(
            spread["all"],
            ["pose_ids", "reference_id", "positions", "mean_spread_angstrom", "excluded"],
            "agreement.spread.all",
        )
        pin(
            spread["all"]["positions"][0],
            ["binder_index", "chain", "number", "icode", "residue", "n_poses", "spread_angstrom"],
            "agreement.spread.all.positions[]",
        )
        contacts = agreement["contacts"]
        pin(
            contacts,
            [
                "definition",
                "cutoff_angstrom",
                "excluded",
                "interface_jaccard",
                "contact_map_jaccard",
            ],
            "agreement.contacts",
        )
        pin(contacts["interface_jaccard"], ["all", "receptor", "binder"], "contacts.jaccard")

    def test_the_aggregates_and_the_representatives(self, result):
        aggregates = result["aggregates"]
        pin(aggregates, ["backend_a", "backend_b", "all"], "aggregates")
        statistic = aggregates["backend_a"]["delta_sasa_static.delta_sasa"]
        pin(
            statistic,
            [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "n_ok",
                "n_total",
                "complete",
                "reasons",
                "caveats",
                "statistic_reasons",
            ],
            "aggregates.<group>.<column>",
        )
        assert "native.ranking_score" in aggregates["backend_a"]
        assert any(
            key.startswith("descriptors.reference.sets.epitope.") for key in aggregates["all"]
        )
        pin(
            result["representatives"],
            ["backend_a", "backend_b", "all", "rule", "reasons"],
            "representatives",
        )

    def test_the_key_names_of_the_flat_rows_are_the_paths_of_the_nested_ones(self, result):
        flat = result.to_flat_rows()[0]
        assert list(flat)[:5] == ["id", "kind", "status", "reason", "content_hash"]
        for column in (
            "frame.receptor_rmsd_angstrom",
            "delta_sasa_static.delta_sasa",
            "native.ranking_score",
            "descriptors.reference.sets.epitope.n_overlap",
            "descriptors.reference.partners.crystal.clash_atoms.2.2",
        ):
            assert column in flat
        assert not any(column.startswith("residue_table") for column in flat)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_input_twice_gives_identical_dictionaries_and_json_bytes(self):
        first = score(make_synthetic_poses(seeds=(1, 2)))
        second = score(make_synthetic_poses(seeds=(1, 2)))
        a, b = first.to_dict(), second.to_dict()
        assert set(a) - {"timing"} == set(b) - {"timing"}
        a.pop("timing"), b.pop("timing")
        assert a == b  # provenance included: nothing in it depends on the run
        assert comparable(first).encode("utf-8") == comparable(second).encode("utf-8")
        assert first.input_key == second.input_key

    def test_the_same_poses_scored_twice_give_the_same_result(self, synthetic, result):
        assert comparable(score(synthetic)) == comparable(result)

    def test_the_writers_give_the_same_bytes_for_two_runs(self, tmp_path):
        outputs = []
        for run in ("first", "second"):
            made = score(make_synthetic_poses(seeds=(1, 2)))
            paths = {
                "json": write_pose_report(
                    made, "json", tmp_path / run / "r.json", include_timing=False
                ),
                "csv": write_pose_report(made, "csv", tmp_path / run / "r.csv"),
                "md": write_pose_report(made, "md", tmp_path / run / "r.md"),
            }
            outputs.append({fmt: path.read_bytes() for fmt, path in paths.items()})
        assert outputs[0] == outputs[1]
        assert all(outputs[0].values())

    def test_other_calls_in_between_leave_nothing_behind(self, synthetic, result):
        """A call with other poses, options and metrics does not change the next call."""
        other = make_synthetic_poses(
            seeds=(3, 4), rotation_degrees={"backend_a": (5.0, 10.0), "backend_b": (0.0, 20.0)}
        )
        score(other, metrics=("delta_sasa_static",), options={"agreement": {"kind": "heavy"}})
        score_pose_set(other.poses[:1], "B", "A", group_by=())
        assert comparable(score(synthetic)) == comparable(result)

    def test_the_result_survives_a_round_trip_through_json_and_from_dict(self, result):
        text = comparable(result)
        again = PoseSetResult.from_dict(json.loads(text))
        assert again.to_dict() == json.loads(text)
        assert comparable(again) == text


# ---------------------------------------------------------------------------
# order of the poses
# ---------------------------------------------------------------------------


def renamed(synthetic, names):
    """The poses of ``synthetic`` under other ids that do not sort like the construction."""
    return [
        DockedPose.from_atoms(
            pose.atoms(),
            id=name,
            program=pose.provenance["backend"],
            backend=pose.provenance["backend"],
            seed=pose.provenance["seed"],
        )
        for pose, name in zip(synthetic.poses, names)
    ]


class TestOrderIndependence:
    def test_every_order_of_the_poses_gives_the_same_aggregates_and_agreement(
        self, synthetic, result
    ):
        reference = reference_of(synthetic)
        for seed in range(4):
            order = np.random.default_rng(seed).permutation(len(synthetic.poses))
            shuffled = [synthetic.poses[i] for i in order]
            again = score(synthetic, shuffled, reference=reference)
            assert again["aggregates"] == result["aggregates"]
            assert again["agreement"] == result["agreement"]
            assert again["representatives"] == result["representatives"]
            assert [row["id"] for row in again["poses"]] == [row["id"] for row in result["poses"]]
            assert comparable(again) == comparable(result)

    def test_rows_follow_the_ids_whatever_the_order_of_construction(self, synthetic):
        poses = renamed(synthetic, ["q-1", "b-2", "m-3", "a-4"])
        results = [
            score(synthetic, order, group_by=())
            for order in (poses, poses[::-1], [poses[2], poses[0], poses[3], poses[1]])
        ]
        for made in results:
            assert [row["id"] for row in made["poses"]] == ["a-4", "b-2", "m-3", "q-1"]
            assert made["agreement"]["pose_ids"] == ["a-4", "b-2", "m-3", "q-1"]
        assert len({comparable(made) for made in results}) == 1

    def test_the_matrix_rows_move_with_the_ids_not_with_the_input(self, synthetic, result):
        """Poses shuffled: the pair of two named poses has the same distance in every order."""
        shuffled = score(synthetic, synthetic.poses[::-1])
        block = shuffled["agreement"]["rmsd"]["ca"]["matrix"]
        assert block["pose_ids"] == sorted(block["pose_ids"])
        index = {pose_id: i for i, pose_id in enumerate(block["pose_ids"])}
        expected = synthetic.truth.pair("backend_a-s2", "backend_b-s1")
        assert block["matrix"][index["backend_a-s2"]][index["backend_b-s1"]] == pytest.approx(
            expected, abs=1e-4
        )


# ---------------------------------------------------------------------------
# side effects
# ---------------------------------------------------------------------------

STUB_SEEN: list[Path] = []


def _stub_records_its_file(cif_path, binder_chain=None, receptor_chain=None):
    STUB_SEEN.append(Path(cif_path))
    return {"value": 1.0, "exists": Path(cif_path).is_file()}


@pytest.fixture
def file_stub(monkeypatch):
    """A metric that reads a file, and notes where the layer put it."""
    STUB_SEEN.clear()
    spec = MetricSpec(
        name="stub_file",
        import_path=f"{__name__}:_stub_records_its_file",
        description="records the path it is given",
        input_type="static_structure",
        chain_mode="interface",
        path_arg="cif_path",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
        cost_class="static",
    )
    monkeypatch.setitem(METRICS_BY_NAME, "stub_file", spec)
    yield STUB_SEEN
    STUB_SEEN.clear()


def tree(directory):
    return sorted(str(path.relative_to(directory)) for path in Path(directory).rglob("*"))


class TestNoSideEffects:
    def test_a_full_call_in_an_empty_working_directory_creates_nothing_there(
        self, synthetic, tmp_path, monkeypatch
    ):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        made = score(
            synthetic,
            metrics=("delta_sasa_static", "hbonds", "structure_rmsd"),
            n_workers=2,
            keep_tmp=False,
        )
        assert [row["status"] for row in made["poses"]] == ["ok"] * 4
        write_pose_report(made, "json", tmp_path / "out" / "r.json")
        assert tree(work) == []
        assert tree(tmp_path / "out") == ["r.json"]

    def test_a_failing_call_leaves_nothing_in_the_working_directory_either(
        self, synthetic, tmp_path, monkeypatch, file_stub
    ):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError, match="binder_chain"):
            score_pose_set(synthetic.poses, "B", "B", metrics=["stub_file"])
        with pytest.raises(Exception, match="pose"):
            score_pose_set(synthetic.poses, "Z", "A", metrics=["stub_file"], on_error="raise")
        assert tree(work) == []

    def test_the_files_go_to_tmp_dir_and_the_system_temp_directory_stays_empty(
        self, synthetic, tmp_path, monkeypatch, file_stub
    ):
        system_temp = tmp_path / "system_temp"
        chosen = tmp_path / "chosen"
        system_temp.mkdir(), chosen.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
        score(synthetic, metrics=["stub_file"], tmp_dir=chosen, reference=None)
        assert len(file_stub) == 4
        assert {path.parent.parent for path in file_stub} == {chosen}
        assert all(path.parent.name.startswith("bm_poseset_") for path in file_stub)
        assert tree(chosen) == [] and tree(system_temp) == []

    def test_without_tmp_dir_it_uses_the_system_temp_directory_and_cleans_up(
        self, synthetic, tmp_path, monkeypatch, file_stub
    ):
        system_temp = tmp_path / "system_temp"
        system_temp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
        score(synthetic, metrics=["stub_file"], reference=None)
        assert {path.parent.parent for path in file_stub} == {system_temp}
        assert tree(system_temp) == []

    def test_keep_tmp_names_the_directory_that_it_kept_inside_tmp_dir(
        self, synthetic, tmp_path, file_stub
    ):
        made = score(synthetic, metrics=["stub_file"], tmp_dir=tmp_path, keep_tmp=True)
        kept = Path(made["provenance"]["temp_dir"])
        assert kept.parent == tmp_path and kept.name.startswith("bm_poseset_")
        assert sorted(path.name for path in kept.iterdir()) == [
            f"pose_{i:04d}.cif" for i in range(4)
        ]
        assert made["provenance"]["run"]["tmp_dir"] == str(tmp_path)

    def test_the_input_files_and_their_directory_are_not_touched(self, synthetic, tmp_path):
        inputs = tmp_path / "inputs"
        paths = synthetic.write(inputs)
        poses = [DockedPose.from_file(path, backend="x", seed=i) for i, path in enumerate(paths)]
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        listing = tree(tmp_path)
        score_pose_set(
            poses,
            "B",
            "A",
            metrics=("delta_sasa_static", "hbonds"),
            reference=reference_of(synthetic),
            residue_table=True,
        )
        assert tree(tmp_path) == listing
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        assert after == before


# ---------------------------------------------------------------------------
# the inputs are not changed
# ---------------------------------------------------------------------------


def snapshot_of(pose):
    atoms = pose.atoms()
    return {
        "id": pose.id,
        "hash": pose.content_hash(),
        "provenance": copy.deepcopy(dict(pose.provenance)),
        "native": copy.deepcopy(pose.native_scores()),
        "coord": atoms.coord.copy(),
        "chain": atoms.chain_id.copy(),
        "res_id": atoms.res_id.copy(),
        "names": atoms.atom_name.copy(),
        "writeable": atoms.coord.flags.writeable,
    }


def same_snapshot(before, after):
    for key in ("id", "hash", "provenance", "native", "writeable"):
        assert before[key] == after[key], key
    for key in ("coord", "chain", "res_id", "names"):
        assert np.array_equal(before[key], after[key]), key


class TestNoMutation:
    def test_the_poses_the_reference_and_every_argument_come_back_unchanged(self, synthetic):
        poses = list(synthetic.poses)
        reference = reference_of(synthetic)
        arguments = {
            "metrics": list(METRICS),
            "options": {
                "frame": {"min_pair_fraction": 0.8},
                "agreement": {"cluster_cutoffs_angstrom": [0.5, 1.5]},
                "descriptors": {"shell_angstrom": 5.0},
            },
            "aggregate": ["mean", "max"],
            "group_by": ["backend"],
            "metric_kwargs": {"omega": {}},
            "residue_maps": {},
        }
        frozen_arguments = copy.deepcopy(arguments)
        before_poses = [snapshot_of(pose) for pose in poses]
        before_reference = (
            reference.content_hash(),
            {name: list(members) for name, members in reference.residue_sets.items()},
            reference.partners["crystal"].pose.content_hash(),
            snapshot_of(reference.pose),
        )
        score_pose_set(
            poses, "B", "A", reference=reference, residue_table=True, n_workers=2, **arguments
        )
        assert arguments == frozen_arguments
        assert poses == list(synthetic.poses)
        for pose, before in zip(poses, before_poses):
            same_snapshot(before, snapshot_of(pose))
        assert before_reference[0] == reference.content_hash()
        assert before_reference[1] == {
            name: list(members) for name, members in reference.residue_sets.items()
        }
        assert before_reference[2] == reference.partners["crystal"].pose.content_hash()
        same_snapshot(before_reference[3], snapshot_of(reference.pose))

    def test_the_atoms_of_a_pose_in_another_frame_are_not_moved_by_the_superposition(
        self, synthetic
    ):
        pose = synthetic.poses[3]  # b-s1: its own random frame, so the fit has work to do
        before = pose.atoms().coord.copy()
        made = score(synthetic, metrics=("delta_sasa_static",))
        transform = made.row(pose.id)["frame"]["transform"]
        assert not np.allclose(transform["rotation"], np.eye(3), atol=1e-3)
        assert np.array_equal(pose.atoms().coord, before)

    def test_a_pose_that_is_a_file_keeps_its_path_and_provenance(self, synthetic, tmp_path):
        paths = synthetic.write(tmp_path)
        poses = [DockedPose.from_file(path, backend="x", seed=i) for i, path in enumerate(paths)]
        before = [(pose.id, pose.structure_path, dict(pose.provenance)) for pose in poses]
        score_pose_set(poses, "B", "A", metrics=["delta_sasa_static"])
        assert before == [(pose.id, pose.structure_path, dict(pose.provenance)) for pose in poses]

    def test_changing_the_result_does_not_change_the_poses_or_the_next_result(
        self, synthetic, result
    ):
        made = score(synthetic)
        made["poses"][0]["provenance"]["backend"] = "changed"
        made["poses"][0]["native"]["ranking_score"] = -1.0
        made["references"]["variants"]["reference"]["reference"]["residue_sets"].clear()
        assert synthetic.poses[0].provenance["backend"] == "backend_a"
        assert synthetic.poses[0].native_scores()["ranking_score"] == 0.55
        assert comparable(score(synthetic)) == comparable(result)


# ---------------------------------------------------------------------------
# the input key
# ---------------------------------------------------------------------------


class TestInputKey:
    def test_it_does_not_depend_on_the_package_version_or_how_the_call_was_run(
        self, synthetic, result, tmp_path, monkeypatch
    ):
        real = provenance_module.collect_provenance

        def other_release():
            data = real()
            data.update(package_version="99.0.0", git_sha="0" * 40, python="0.0")
            return data

        monkeypatch.setattr(provenance_module, "collect_provenance", other_release)
        moved = score(synthetic, tmp_dir=tmp_path, keep_tmp=True, n_workers=2)
        assert moved["package_version"] == "99.0.0" and moved["provenance"]["git_sha"] == "0" * 40
        assert moved.input_key == result.input_key
        assert Path(moved["provenance"]["temp_dir"]).parent == tmp_path

    @pytest.mark.parametrize(
        "keyword",
        [
            {"options": {"frame": {"min_pair_fraction": 0.8}}},
            {"options": {"agreement": {"contact_cutoff_angstrom": 4.5}}},
            {"options": {"agreement": {"cluster_cutoffs_angstrom": [0.5, 1.0]}}},
            {"options": {"descriptors": {"shell_angstrom": 5.0}}},
            {"options": {"aggregation": {"min_n_ok": 2}}},
            {"group_by": ()},
            {"heavy": "all"},
            {"residue_table": False},
            {"include_transform": False},
            {"aggregate": ["mean"]},
            {"metrics": ("delta_sasa_static",)},
        ],
    )
    def test_it_changes_with_a_threshold_or_a_choice(self, synthetic, result, keyword):
        assert score(synthetic, **keyword).input_key != result.input_key

    def test_it_changes_with_the_reference_the_chains_and_the_poses(self, synthetic, result):
        fewer_sites = Reference(
            synthetic.reference,
            "A",
            "B",
            residue_sets={"epitope": [ResidueKey("A", n) for n in SITE[:5]]},
            partners={"crystal": Partner(synthetic.reference, chains=["B"])},
        )
        keys = {
            result.input_key,
            score(synthetic, reference=fewer_sites).input_key,
            score(synthetic, synthetic.poses[:3]).input_key,
            score(synthetic, reference=None, residue_table=False).input_key,
        }
        assert len(keys) == 4
        assert len(result.input_key) == 64 and int(result.input_key, 16) >= 0
