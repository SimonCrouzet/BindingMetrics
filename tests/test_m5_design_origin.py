"""The design origin in the pre-flight check: a check that rests on independence is not run
against the model that made the design.

``evobind_adversarial`` asks whether a model other than the designer puts the binder where the
design has it. When the design pipeline ran that model itself (BoltzGen and Boltz-2, BindCraft and
ColabDesign and AlphaFold2), the design was selected for that model and its agreement says little,
so the default policy leaves the check out with a loud warning and runs everything else. An
unknown source never blocks.
"""

from __future__ import annotations

import logging
import pickle
from types import SimpleNamespace

import pytest

from binding_metrics.capabilities import (
    DEFAULT_DESIGN_POLICY,
    DESIGN_SOURCE_PREDICTORS,
    INDEPENDENCE_DEPENDENT_CHECKS,
    POLICIES,
    DesignOrigin,
    IncompatibleInputError,
    InputProfile,
    check_design_origin,
    design_origin,
    preflight,
)
from binding_metrics.metrics.registry import METRICS_BY_NAME
from binding_metrics.predictors.registry import PARSERS

PEPTIDE = InputProfile("B", "A", n_binder_residues=13, binder_type="peptide")
CHECK = "evobind_adversarial"
PROVIDED = {"reference_structure", "predicted_structure"}
LOGGER = "binding_metrics.capabilities"


def run(source, predictor, metrics=(CHECK, "omega"), **kwargs):
    kwargs.setdefault("provided", PROVIDED)
    return preflight(PEPTIDE, list(metrics), predictor, design_source=source, **kwargs)


class TestTheTables:
    def test_the_mapping_is_the_one_of_the_design_pipelines(self):
        assert DESIGN_SOURCE_PREDICTORS == {
            "boltzgen": frozenset({"boltz2"}),
            "bindcraft": frozenset({"af2"}),
            "colabdesign": frozenset({"af2"}),
        }
        assert DEFAULT_DESIGN_POLICY == "skip"

    def test_every_model_of_the_table_is_a_registered_predictor(self):
        used = set().union(*DESIGN_SOURCE_PREDICTORS.values())
        assert used <= set(PARSERS)

    def test_every_independent_check_is_a_registry_metric_that_compares_two_structures(self):
        for name, why in INDEPENDENCE_DEPENDENT_CHECKS.items():
            spec = METRICS_BY_NAME[name]
            assert spec.secondary_path_arg, name
            assert "selected for that model" in why

    def test_the_confidence_metrics_are_not_among_them(self):
        for name in ("prediction", "openfold", "interface_pae", "evobind_score"):
            assert name not in INDEPENDENCE_DEPENDENT_CHECKS


class TestTheOrigin:
    @pytest.mark.parametrize("name", ["boltzgen", "BoltzGen", "  BOLTZGEN "])
    def test_a_name_is_read_in_any_case(self, name):
        origin = design_origin(name)
        assert origin == DesignOrigin("boltzgen", frozenset({"boltz2"}))
        assert origin.known

    def test_the_models_can_be_given_directly(self):
        origin = design_origin(("Boltz2", "af2"))
        assert origin.source is None and origin.predictors == frozenset({"boltz2", "af2"})

    def test_none_and_an_unknown_name_have_no_models(self):
        assert design_origin(None) == DesignOrigin()
        unknown = design_origin("rfdiffusion")
        assert unknown.source == "rfdiffusion" and not unknown.known

    def test_an_origin_passes_through(self):
        origin = DesignOrigin("x", frozenset({"of3"}))
        assert design_origin(origin) is origin

    @pytest.mark.parametrize("bad", [3, 1.5, {"a": 1}, [1, 2]])
    def test_a_value_of_another_type_is_a_type_error(self, bad):
        with pytest.raises(TypeError):
            design_origin(bad)

    def test_it_names_the_models_it_ran(self):
        origin = design_origin("bindcraft")
        assert origin.shares_model_with("af2") and origin.shares_model_with(" AF2 ")
        assert not origin.shares_model_with("boltz2")
        assert not origin.shares_model_with(None)
        assert "bindcraft, which ran AlphaFold2 / ColabFold itself" in origin.short()

    def test_the_origin_pickles(self):
        origin = design_origin("boltzgen")
        assert pickle.loads(pickle.dumps(origin)) == origin


class TestTheDecision:
    def test_a_circular_model_leaves_the_check_out_under_skip(self):
        decision = check_design_origin("boltzgen", [CHECK, "omega"], ["boltz2"])
        assert decision.skipped == (CHECK,)
        assert decision.checks == (CHECK,)  # omega rests on nothing
        assert decision.independent("boltz2") is False and decision.independent("of3") is True
        (violation,) = decision.violations
        assert (violation.kind, violation.name, violation.constraint) == (
            "design",
            CHECK,
            "design_origin",
        )

    def test_the_loud_sentence_says_which_check_why_and_how_to_override(self):
        (text,) = check_design_origin("boltzgen", [CHECK], ["boltz2"]).warnings
        assert text.startswith(f"{CHECK} is skipped because the design was made with boltzgen")
        assert "Boltz-2" in text and "the design was selected for that model" in text
        assert "OpenFold3 (of3)" in text and "Boltz-2 (boltz2)" not in text.split("predict with")[1]
        assert "design_policy='warn' (--design-policy warn)" in text

    def test_warn_runs_the_check_and_says_so(self):
        decision = check_design_origin("boltzgen", [CHECK], ["boltz2"], policy="warn")
        assert decision.skipped == () and len(decision.violations) == 1
        assert decision.warnings[0].startswith(f"{CHECK} runs on a Boltz-2 structure although")

    def test_error_leaves_the_refusal_to_the_caller_and_adds_no_warning(self):
        decision = check_design_origin("bindcraft", [CHECK], ["af2"], policy="error")
        assert decision.skipped == () and decision.warnings == ()
        assert len(decision.violations) == 1

    def test_an_independent_model_changes_nothing(self):
        decision = check_design_origin("boltzgen", [CHECK], ["of3"])
        assert decision.skipped == () and decision.violations == () and decision.warnings == ()

    def test_the_check_goes_when_every_model_that_makes_its_structure_is_circular(self):
        mixed = check_design_origin("boltzgen", [CHECK], ["boltz2", "of3"])
        assert mixed.skipped == () and len(mixed.violations) == 1  # of3 can still run it
        both = check_design_origin("colabdesign", [CHECK], ["af2", "AF2"])
        assert both.skipped == (CHECK,)

    def test_an_unknown_source_is_a_warning_and_never_a_block(self):
        for policy in POLICIES:
            decision = check_design_origin("rfdiffusion", [CHECK], ["boltz2"], policy=policy)
            assert decision.skipped == () and decision.violations == ()
            (text,) = decision.warnings
            assert "'rfdiffusion' is not one of boltzgen, bindcraft, colabdesign" in text
            assert "design_source=('boltz2',)" in text

    def test_no_source_says_so_in_a_note_only_when_a_check_needs_it(self):
        with_check = check_design_origin(None, [CHECK], ["boltz2"])
        assert "no design source was given" in with_check.notes[0] and with_check.warnings == ()
        assert check_design_origin(None, ["omega"], ["boltz2"]).notes == ()

    def test_a_model_that_is_not_named_gets_a_note(self):
        decision = check_design_origin("boltzgen", [CHECK], [])
        assert "was not named" in decision.notes[0] and decision.skipped == ()

    def test_a_single_name_is_not_taken_for_a_list_of_letters(self):
        decision = check_design_origin("boltzgen", CHECK, "boltz2")
        assert decision.skipped == (CHECK,)

    def test_the_policy_is_checked(self):
        with pytest.raises(ValueError, match="design_policy must be one of"):
            check_design_origin("boltzgen", [CHECK], ["boltz2"], policy="ignore")


class TestInThePreflight:
    def test_the_default_skips_only_the_circular_check_and_keeps_the_rest(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            report = run("boltzgen", "boltz2", metrics=(CHECK, "omega", "interface"))
        assert report.metrics_to_run == ("omega", "interface")
        assert report.skipped_metrics == (CHECK,)
        assert report.predictor_usable
        assert any(
            r.levelno == logging.WARNING and f"{CHECK} is skipped because" in r.getMessage()
            for r in caplog.records
        )

    def test_the_report_says_what_was_skipped_and_why_in_its_dictionary(self):
        data = run("BoltzGen", "boltz2").to_dict()
        assert data["metrics_to_run"] == ["omega"]
        (skipped,) = data["metrics_skipped"]
        assert skipped["name"] == CHECK and "ran Boltz-2 itself" in skipped["reasons"][0]
        design = data["design"]
        assert design["source"] == "boltzgen" and design["known"] is True
        assert design["predictors"] == ["boltz2"] and design["policy"] == "skip"
        assert design["models"] == ["boltz2"] and design["not_independent"] == ["boltz2"]
        (entry,) = design["checks"]
        assert entry["check"] == CHECK and entry["model"] == "boltz2"
        assert entry["action"] == "skipped" and "ran Boltz-2 itself" in entry["reason"]
        (violation,) = [v for v in data["violations"] if v["kind"] == "design"]
        assert violation["name"] == CHECK and violation["fix"]
        assert any(f"{CHECK} is skipped" in text for text in data["warnings"])
        assert data["compatible"] is False

    def test_the_plan_names_the_origin_and_the_check_that_was_left_out(self):
        text = run("boltzgen", "boltz2").format()
        assert "(policy: error, design: skip)" in text
        assert "Design origin: boltzgen, which ran Boltz-2 itself" in text
        assert f"Left out: metrics {CHECK}" in text
        assert "design_origin" in text and "Warnings:" in text

    def test_warn_runs_it_and_labels_the_finding(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            report = run("bindcraft", "af2", design_policy="warn")
        assert report.metrics_to_run == (CHECK, "omega")
        assert report.to_dict()["design"]["checks"][0]["action"] == "run"
        assert any(
            "runs on a AlphaFold2 / ColabFold structure" in r.getMessage() for r in caplog.records
        )

    def test_error_refuses_with_the_whole_explanation(self):
        with pytest.raises(IncompatibleInputError) as caught:
            run("boltzgen", "boltz2", design_policy="error")
        report = caught.value.report
        assert report.refused and [v.kind for v in report.violations] == ["design"]
        message = str(caught.value)
        assert (
            "design_origin" in message and "found:    the design was made with boltzgen" in message
        )
        assert "why:" in message and "fix:" in message

    def test_the_policy_of_the_other_rules_does_not_change_it(self):
        # --on-incompatible error is the default and the design rule still skips
        assert run("boltzgen", "boltz2", policy="error").metrics_to_run == ("omega",)
        assert run("boltzgen", "boltz2", policy="warn").metrics_to_run == ("omega",)
        # and the design policy does not touch the rest: warn here, error there
        report = run("boltzgen", "boltz2", policy="skip", design_policy="warn")
        assert report.metrics_to_run == (CHECK, "omega")

    def test_a_model_that_the_design_did_not_run_leaves_everything_in(self):
        report = run("boltzgen", "of3")
        assert report.metrics_to_run == (CHECK, "omega")
        assert report.to_dict()["design"]["checks"] == []
        assert report.to_dict()["design"]["not_independent"] == []

    def test_the_models_can_be_passed_without_a_predictor_to_check(self):
        report = run("boltzgen", None, prediction_models=["boltz2", "of3"])
        assert report.metrics_to_run == (CHECK, "omega")  # of3 can still run it
        assert report.to_dict()["design"]["not_independent"] == ["boltz2"]
        only = run("boltzgen", None, prediction_models=["boltz2"])
        assert only.metrics_to_run == ("omega",)

    def test_a_runner_or_adapter_object_is_matched_by_its_name(self):
        stand_in = SimpleNamespace(name="boltz2", display_name="Boltz-2", capabilities=None)
        assert run("boltzgen", stand_in).metrics_to_run == ("omega",)

    def test_models_given_directly_work_for_a_pipeline_that_is_not_in_the_table(self):
        report = run(("of3",), "of3")
        assert report.metrics_to_run == ("omega",)
        assert report.to_dict()["design"]["source"] is None
        assert "the design pipeline ran OpenFold3 itself" in report.warnings[0]

    def test_an_unknown_source_runs_everything_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            report = run("rfdiffusion", "boltz2", policy="error", design_policy="error")
        assert report.metrics_to_run == (CHECK, "omega") and not report.refused
        assert report.to_dict()["design"]["known"] is False
        assert any("'rfdiffusion' is not one of" in r.getMessage() for r in caplog.records)

    def test_without_a_source_the_dictionary_is_the_one_it_always_was(self):
        data = run(None, "boltz2").to_dict()
        assert "design" not in data
        assert list(data) == [
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
        ]
        assert data["metrics_to_run"] == [CHECK, "omega"]
        assert any("no design source was given" in note for note in data["notes"])

    def test_a_policy_that_is_not_one_of_the_three_is_refused_first(self):
        with pytest.raises(ValueError, match="design_policy must be one of"):
            run("boltzgen", "boltz2", design_policy="ignore")

    def test_a_source_of_another_type_is_a_type_error(self):
        with pytest.raises(TypeError, match="design_source"):
            run(7, "boltz2")

    def test_the_report_pickles_with_the_design_block(self):
        report = run("boltzgen", "boltz2")
        again = pickle.loads(pickle.dumps(report))
        assert again.to_dict() == report.to_dict()
