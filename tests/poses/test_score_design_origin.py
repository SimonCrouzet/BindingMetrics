"""``score_pose_set`` and the design origin: which back-ends are independent of the design.

A model that took part in the design agrees with the design partly because the design was
selected for it. ``design_source`` names the pipeline; the layer labels each pose
(``independent_of_design``), does not run a check that rests on independence
(``evobind_adversarial``) on the poses of a back-end that the pipeline ran itself, and says so in
the row, the pre-flight block and the warnings. The poses are three predicted poses of 1YCR
written by the synthetic writers of the adapters: Boltz-2, OpenFold3 and AlphaFold2.
"""

import importlib
import inspect
import json
import logging

import pytest

from binding_metrics.capabilities import IncompatibleInputError
from binding_metrics.poses import PoseSetResult, Reference, score_pose_set
from binding_metrics.poses.docked import DockedPose
from binding_metrics.predictors import get_parser
from binding_metrics.testing import make_synthetic_poses
from tests.test_feat_c_support import EXAMPLE_1YCR, complex_from

pytest.importorskip("biotite.structure")

CHECK = "evobind_adversarial"
LOGGER = "binding_metrics.capabilities"


@pytest.fixture(scope="module")
def predicted(tmp_path_factory):
    """One predicted pose per model, all of the same complex."""
    root = tmp_path_factory.mktemp("models")
    poses = {}
    for model in ("boltz2", "of3", "af2"):
        out = root / model
        out.mkdir()
        writer = importlib.import_module(f"tests.predictors.synth_{model}")
        writer.write_prediction(out, "design", complex_from(EXAMPLE_1YCR))
        poses[model] = get_parser(model).load(out, "design")
    return poses


def score(poses, **kwargs):
    kwargs.setdefault("metrics", [CHECK])
    kwargs.setdefault("reference", Reference(EXAMPLE_1YCR, "A", "B"))
    return score_pose_set(list(poses), "B", "A", **kwargs)


def labels(result):
    return {row["provenance"]["backend"]: row["independent_of_design"] for row in result["poses"]}


def by_backend(result):
    return {row["provenance"]["backend"]: row for row in result["poses"]}


@pytest.fixture(scope="module")
def mixed(predicted):
    return score(predicted.values(), design_source="BoltzGen")


class TestWithoutASource:
    def test_nothing_is_decided_and_the_result_has_no_design_keys(self, predicted):
        result = score(predicted.values())
        assert set(labels(result).values()) == {None}
        assert "design" not in result["parameters"] and "design" not in result["preflight"]
        for row in result["poses"]:
            assert row["status"] == "ok" and CHECK in row and not row["reasons"]
        assert result["preflight"]["metrics_to_run"] == [CHECK]
        assert result["preflight"]["notes"] == [] or not any(
            "independence" in n for n in result["preflight"]["notes"]
        )
        assert not any("design origin" in text for text in result["warnings"])

    def test_the_arguments_default_to_no_source_and_the_skip_policy(self):
        parameters = inspect.signature(score_pose_set).parameters
        assert parameters["design_source"].default is None
        assert parameters["design_policy"].default == "skip"
        assert list(parameters)[-2:] == ["design_source", "design_policy"]


class TestSkip:
    def test_each_pose_is_labelled_by_its_back_end(self, mixed):
        assert labels(mixed) == {"af2": True, "boltz2": False, "of3": True}

    def test_the_check_is_not_run_on_the_model_the_design_ran_and_the_row_says_why(self, mixed):
        rows = by_backend(mixed)
        assert CHECK not in rows["boltz2"]
        assert rows["boltz2"]["status"] == "partial" and rows["boltz2"]["reason"] is None
        reason = rows["boltz2"]["reasons"][CHECK]
        assert "the design was made with boltzgen, which ran Boltz-2 itself" in reason
        for model in ("of3", "af2"):
            assert rows[model]["status"] == "ok" and rows[model][CHECK]["delta_com_angstrom"] >= 0
            assert not rows[model]["reasons"]

    def test_the_groups_count_the_skipped_pose_in_n_total_and_not_in_n_ok(self, mixed):
        key = f"{CHECK}.delta_com_angstrom"
        entries = {group: mixed["aggregates"][group][key] for group in ("boltz2", "of3", "af2")}
        assert [(e["n_ok"], e["n_total"]) for e in entries.values()] == [(0, 1), (1, 1), (1, 1)]
        assert list(entries["boltz2"]["reasons"]) == [mixed["poses"][1]["reasons"][CHECK]]
        assert (
            mixed["aggregates"]["all"][key]["n_ok"],
            mixed["aggregates"]["all"][key]["n_total"],
        ) == (2, 3)

    def test_the_other_checks_and_the_agreement_run_on_every_pose(self, predicted):
        result = score(
            predicted.values(), metrics=[CHECK, "delta_sasa_static"], design_source="boltzgen"
        )
        for row in result["poses"]:
            assert "delta_sasa_static" in row
        assert result["agreement"]["status"] == "ok"
        assert len(result["agreement"]["pose_ids"]) == 3

    def test_the_check_stays_in_the_set_while_one_back_end_can_run_it(self, mixed):
        assert mixed["preflight"]["metrics_to_run"] == [CHECK]
        assert mixed["parameters"]["metric_scope"] == {CHECK: "all"}

    def test_the_preflight_block_says_which_check_which_model_and_why(self, mixed):
        design = mixed["preflight"]["design"]
        assert design["source"] == "boltzgen" and design["policy"] == "skip"
        assert design["predictors"] == ["boltz2"] and design["known"] is True
        assert design["models"] == ["af2", "boltz2", "of3"]
        assert design["not_independent"] == ["boltz2"]
        (entry,) = design["checks"]
        assert (entry["check"], entry["model"], entry["action"]) == (CHECK, "boltz2", "skipped")
        assert entry["reason"] == mixed["poses"][1]["reasons"][CHECK]
        assert [v["kind"] for v in mixed["preflight"]["violations"]] == ["design"]

    def test_the_warnings_name_the_check_and_the_back_ends_that_are_not_independent(self, mixed):
        sentences = [w for w in mixed["warnings"] if "evobind_adversarial is skipped" in w]
        assert len(sentences) == 1 and "design_policy='warn'" in sentences[0]
        (named,) = [w for w in mixed["warnings"] if w.startswith("design origin:")]
        assert named == (
            "design origin: the poses of boltz2 (1 pose) are not independent of the design "
            "(boltzgen, which ran Boltz-2 itself); their independent_of_design is False"
        )
        summary = mixed.summary()
        assert sentences[0] in summary and named in summary

    def test_the_parameters_record_the_source_and_change_the_input_key(self, predicted, mixed):
        assert mixed["parameters"]["design"] == {
            "source": "boltzgen",
            "predictors": ["boltz2"],
            "policy": "skip",
        }
        plain = score(predicted.values())
        other = score(predicted.values(), design_source="bindcraft")
        assert len({plain["input_key"], mixed["input_key"], other["input_key"]}) == 3
        assert (
            score(predicted.values(), design_source="BOLTZGEN")["input_key"] == mixed["input_key"]
        )

    def test_the_log_carries_the_loud_sentence(self, predicted, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            score(predicted.values(), design_source="boltzgen")
        assert any(
            r.levelno == logging.WARNING and f"{CHECK} is skipped because" in r.getMessage()
            for r in caplog.records
        )

    def test_when_every_back_end_is_circular_the_check_is_left_out_of_the_set(self, predicted):
        result = score([predicted["boltz2"]], design_source="boltzgen")
        assert result["preflight"]["metrics_to_run"] == []
        assert [m["name"] for m in result["preflight"]["metrics_skipped"]] == [CHECK]
        assert result["parameters"]["metric_scope"] == {CHECK: "none"}
        row = result["poses"][0]
        assert CHECK not in row and row["independent_of_design"] is False
        assert row["reasons"][CHECK].startswith("left out by the pre-flight check: the design")

    def test_two_models_the_pipeline_ran_are_both_labelled(self, predicted):
        result = score(predicted.values(), design_source=("boltz2", "AF2"))
        assert labels(result) == {"af2": False, "boltz2": False, "of3": True}
        rows = by_backend(result)
        assert CHECK in rows["of3"] and CHECK not in rows["af2"] and CHECK not in rows["boltz2"]


class TestOtherPolicies:
    def test_warn_runs_the_check_on_every_pose_and_still_labels_and_warns(self, predicted):
        result = score(predicted.values(), design_source="boltzgen", design_policy="warn")
        assert labels(result) == {"af2": True, "boltz2": False, "of3": True}
        for row in result["poses"]:
            assert CHECK in row and row["status"] == "ok"
        assert any(
            "evobind_adversarial runs on a Boltz-2 structure" in w for w in result["warnings"]
        )
        assert result["preflight"]["design"]["checks"][0]["action"] == "run"
        assert result["parameters"]["design"]["policy"] == "warn"

    def test_error_refuses_before_a_pose_is_scored(self, predicted, tmp_path):
        with pytest.raises(IncompatibleInputError) as caught:
            score(
                predicted.values(),
                design_source="boltzgen",
                design_policy="error",
                tmp_dir=tmp_path,
            )
        assert [v.kind for v in caught.value.violations] == ["design"]
        assert "design_origin" in str(caught.value)
        assert not list(tmp_path.iterdir())

    def test_error_refuses_nothing_when_no_back_end_is_circular(self, predicted):
        result = score(
            [predicted["of3"], predicted["af2"]], design_source="boltzgen", design_policy="error"
        )
        assert set(labels(result).values()) == {True}

    @pytest.mark.parametrize("mode", ["error", "skip", "warn", "off"])
    def test_the_preflight_argument_does_not_change_the_rule(self, predicted, mode):
        result = score(predicted.values(), design_source="boltzgen", preflight=mode)
        assert labels(result) == {"af2": True, "boltz2": False, "of3": True}
        assert CHECK in by_backend(result)["of3"] and CHECK not in by_backend(result)["boltz2"]
        assert (result["preflight"] is None) == (mode == "off")
        assert any("evobind_adversarial is skipped" in w for w in result["warnings"])

    def test_off_with_error_still_refuses(self, predicted):
        with pytest.raises(IncompatibleInputError):
            score(
                predicted.values(),
                design_source="boltzgen",
                design_policy="error",
                preflight="off",
            )


class TestUnknownSources:
    def test_a_name_outside_the_table_warns_and_blocks_nothing(self, predicted, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = score(predicted.values(), design_source="rfdiffusion", design_policy="error")
        assert set(labels(result).values()) == {None}
        assert all(CHECK in row for row in result["poses"])
        assert any("'rfdiffusion' is not one of" in w for w in result["warnings"])
        assert result["parameters"]["design"]["source"] == "rfdiffusion"
        assert result["preflight"]["design"]["known"] is False
        assert any("'rfdiffusion' is not one of" in r.getMessage() for r in caplog.records)

    def test_a_pose_without_a_back_end_has_no_label(self):
        synthetic = make_synthetic_poses(backends=("boltz2", "of3"), seeds=(1,))
        result = score_pose_set(synthetic.poses, "B", "A", design_source="boltzgen")
        assert labels(result) == {"boltz2": False, "of3": True}
        bare = DockedPose.from_file(EXAMPLE_1YCR, id="bare")
        result = score_pose_set([bare, *synthetic.poses], "B", "A", design_source="boltzgen")
        assert result.row("bare")["independent_of_design"] is None
        assert sorted(
            row["independent_of_design"] for row in result["poses"] if row["id"] != "bare"
        ) == [
            False,
            True,
        ]


class TestTheArguments:
    def test_a_policy_outside_the_three_is_refused_before_any_pose_is_read(self):
        class Unreadable:
            pass

        with pytest.raises(ValueError, match="design_policy must be one of"):
            score_pose_set([Unreadable()], "B", "A", design_policy="ignore")

    def test_a_source_of_another_type_is_a_type_error(self, predicted):
        with pytest.raises(TypeError, match="design_source"):
            score(predicted.values(), design_source=3)

    def test_the_result_is_json_and_survives_a_round_trip(self, mixed):
        again = PoseSetResult.from_dict(json.loads(mixed.to_json()))
        assert again["parameters"]["design"] == mixed["parameters"]["design"]
        assert [row["independent_of_design"] for row in again["poses"]] == [
            row["independent_of_design"] for row in mixed["poses"]
        ]

    def test_the_label_is_a_column_of_the_flat_rows(self, mixed):
        flat = mixed.to_flat_rows()
        assert {row["id"]: row["independent_of_design"] for row in flat} == {
            row["id"]: row["independent_of_design"] for row in mixed["poses"]
        }

    def test_the_workers_skip_the_same_poses(self, predicted, mixed):
        pooled = score(predicted.values(), design_source="BoltzGen", n_workers=2)
        assert pooled["input_key"] == mixed["input_key"]
        assert pooled["poses"] == mixed["poses"]
        assert pooled["warnings"] == mixed["warnings"]
