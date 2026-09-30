"""``preflight``: collected violations, policies, message content, and no work before a refusal."""

from __future__ import annotations

import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest

from binding_metrics.capabilities import (
    POLICIES,
    Capabilities,
    Closure,
    ClosureEnd,
    IncompatibleInputError,
    InputProfile,
    PreflightReport,
    preflight,
)
from binding_metrics.predictors.registry import PARSERS, ParserSpec
from tests.predictors.synth_stub import StubParser

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _closure(kind, family, a, b):
    return Closure(kind, family, ClosureEnd(*a), ClosureEnd(*b))


RING = _closure("head_to_tail", "head_to_tail", (13, "ASP", 14, "C"), (0, "GLY", 1, "N"))
DISULFIDE = _closure("disulfide", "disulfide", (2, "CYS", 3, "SG"), (10, "CYS", 11, "SG"))

LINEAR = InputProfile("B", "A", n_binder_residues=14, binder_type="peptide")
BICYCLE = dataclasses.replace(
    LINEAR, closures=frozenset({"head_to_tail", "disulfide"}), closure_bonds=(RING, DISULFIDE)
)
D_RESIDUES = dataclasses.replace(
    LINEAR,
    residue_classes=frozenset({"canonical", "d_amino"}),
    residue_names={"canonical": ("ALA",), "d_amino": ("DAL",)},
)

# ---------------------------------------------------------------------------
# Predictors: three registered adapters and a runner that records every call
# ---------------------------------------------------------------------------

OF3_LIKE_CAPABILITIES = Capabilities(
    closures={"none", "head_to_tail"},
    residue_classes={"canonical"},
    reasons={
        "closures": "The query builder writes a head-to-tail closure and nothing else.",
        "closures:disulfide": "A disulfide has no field in the query.",
        "residue_classes": "Other residues are not written into the query.",
    },
    version="9.9",
)


class NarrowFold(StubParser):
    name = "narrow"
    display_name = "NarrowFold"
    capabilities = OF3_LIKE_CAPABILITIES


class WideFold(StubParser):
    name = "wide"
    display_name = "WideFold"
    capabilities = Capabilities(
        closures={"none", "head_to_tail", "disulfide"},
        reasons={"closures": "Reads any of these."},
    )


class VagueFold(StubParser):
    name = "vague"
    display_name = "VagueFold"


@pytest.fixture
def three_models(monkeypatch):
    """Register NarrowFold, WideFold and VagueFold for one test."""
    for cls in (NarrowFold, WideFold, VagueFold):
        spec = ParserSpec(
            name=cls.name,
            import_path=f"{cls.__module__}:{cls.__name__}",
            display_name=cls.display_name,
            family=cls.family,
        )
        monkeypatch.setitem(PARSERS, cls.name, spec)


class Spy:
    """A runner that records every attribute read and every call, and declares limits."""

    log: list = []
    capabilities = OF3_LIKE_CAPABILITIES
    display_name = "SpyRunner"
    name = "spy"

    def __init__(self):
        Spy.log.append("__init__")

    def __getattribute__(self, attribute):
        if not attribute.startswith("__"):
            Spy.log.append(attribute)
        return object.__getattribute__(self, attribute)

    def run(self):
        Spy.log.append("run")

    def prepare(self):
        Spy.log.append("prepare")


@pytest.fixture(autouse=True)
def _clear_spy():
    Spy.log = []


# ---------------------------------------------------------------------------
# The three policies
# ---------------------------------------------------------------------------


class TestCompatibleInput:
    def test_nothing_declared_means_nothing_refused(self):
        report = preflight(BICYCLE, ["interface", "coulomb"], VagueFold)
        assert isinstance(report, PreflightReport)
        assert report.compatible and report.violations == ()
        assert report.metrics_to_run == ("interface", "coulomb") and report.predictor_usable

    def test_an_accepted_input_passes_a_constrained_predictor(self):
        report = preflight(LINEAR, ["interface"], NarrowFold)
        assert report.compatible
        assert report.predictors == ("NarrowFold 9.9",)

    def test_no_metric_and_no_predictor(self):
        assert preflight(BICYCLE, []).compatible
        assert preflight(BICYCLE).compatible

    def test_a_metric_name_can_be_given_bare(self):
        assert preflight(LINEAR, "interface").metrics_requested == ("interface",)

    def test_the_policies_are_listed(self):
        assert POLICIES == ("error", "skip", "warn")


class TestErrorPolicy:
    def test_a_refusal_raises_a_value_error_with_the_report(self, three_models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(BICYCLE, ["interface"], "narrow")
        error = caught.value
        assert isinstance(error, ValueError)
        assert error.report.policy == "error" and not error.report.compatible
        assert error.violations == error.report.violations

    def test_the_message_names_the_model_the_closure_found_and_the_alternatives(self, three_models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(BICYCLE, ["interface"], "narrow")
        message = str(caught.value)
        assert "Pre-flight check failed: 1 incompatibility" in message
        assert "predictor NarrowFold 9.9: closures" in message
        assert "the binder has a disulfide bond (CYS 3.SG - CYS 11.SG)" in message
        assert "closures limited to: none, head_to_tail" in message
        assert "A disulfide has no field in the query." in message
        assert "WideFold (wide)" in message  # accepts this input
        assert "VagueFold (vague)" in message and "no declared limits" in message
        assert "policy='skip'" in message
        assert "binder chain B: 14 residues" in message  # the input, so the fact can be checked

    def test_every_violation_is_listed_at_once(self, three_models):
        profile = dataclasses.replace(
            BICYCLE,
            residue_classes=D_RESIDUES.residue_classes,
            residue_names=D_RESIDUES.residue_names,
        )
        needing = SimpleNamespace(
            name="needs_receptor",
            capabilities=Capabilities(needs={"receptor_chain"}, reasons={"needs": "Interface."}),
        )
        profile = dataclasses.replace(profile, receptor_chain=None)
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(profile, [needing], "narrow")
        violations = caught.value.violations
        assert [(v.kind, v.constraint) for v in violations] == [
            ("metric", "needs"),
            ("predictor", "closures"),
            ("predictor", "residue_classes"),
        ]
        assert "3 incompatibilities" in str(caught.value)
        for violation in violations:
            assert violation.fact in str(caught.value)
        # the two violations of the predictor share one fix line
        assert str(caught.value).count("fix:") == 2

    def test_a_metric_violation_names_the_metric_and_the_fix(self):
        needing = SimpleNamespace(
            name="dockq",
            capabilities=Capabilities(
                needs={"reference_structure"}, reasons={"needs": "DockQ compares to a native."}
            ),
        )
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(LINEAR, [needing], provided=set())
        (violation,) = caught.value.violations
        assert violation.subject == "metric 'dockq'"
        assert violation.fact == "no reference structure was provided"
        assert violation.reason == "DockQ compares to a native."
        assert "leave 'dockq' out of the metric list" in violation.fix

    def test_no_alternative_is_said_plainly(self, monkeypatch):
        monkeypatch.setattr("binding_metrics.predictors.registry.PARSERS", {})
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(BICYCLE, [], NarrowFold)
        assert "no other registered predictor declares support for this input" in str(caught.value)


class TestSkipPolicy:
    def test_the_compatible_metrics_go_on_and_the_refused_ones_are_named(self):
        cyclic_only = SimpleNamespace(
            name="closure_check",
            capabilities=Capabilities(
                closures={"head_to_tail", "disulfide", "lactam", "staple", "other"},
                reasons={"closures": "It scores the closing bond."},
            ),
        )
        report = preflight(LINEAR, ["interface", cyclic_only, "coulomb"], policy="skip")
        assert report.metrics_to_run == ("interface", "coulomb")
        assert report.skipped_metrics == ("closure_check",)
        assert not report.compatible
        assert report.to_dict()["metrics_skipped"] == [
            {"name": "closure_check", "reasons": ["the binder is linear (no ring closure found)"]}
        ]

    def test_a_refused_predictor_is_marked_unusable_and_the_metrics_stay(self, three_models):
        report = preflight(BICYCLE, ["interface"], "narrow", policy="skip")
        assert report.predictor_usable is False
        assert report.metrics_to_run == ("interface",)
        text = report.format()
        assert "policy: skip" in text and "Left out: predictor NarrowFold 9.9" in text

    def test_the_skipped_metric_is_logged(self, caplog):
        needing = SimpleNamespace(
            name="dockq",
            capabilities=Capabilities(needs={"reference_structure"}, reasons={"needs": "Native."}),
        )
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            preflight(LINEAR, [needing], policy="skip", provided=set())
        assert "leaving it out" in caplog.text and "metric 'dockq'" in caplog.text


class TestWarnPolicy:
    def test_everything_runs_and_the_violations_are_logged(self, three_models, caplog):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            report = preflight(BICYCLE, ["interface"], "narrow", policy="warn")
        assert report.violations and report.metrics_to_run == ("interface",)
        assert report.predictor_usable is True
        assert "running anyway" in caplog.text and "predictor NarrowFold 9.9" in caplog.text

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="policy must be one of"):
            preflight(LINEAR, [], policy="ignore")


# ---------------------------------------------------------------------------
# Nothing runs before a refusal
# ---------------------------------------------------------------------------


class TestNothingRunsBeforeARefusal:
    @staticmethod
    def _pipeline(profile, runner, prep, calls):
        """What a caller does: pre-flight first, then prepare, then run the model."""
        preflight(profile, ["interface"], runner)
        prep()
        runner.run() if not isinstance(runner, type) else runner().run()
        calls.append("finished")

    @pytest.mark.parametrize("as_instance", [False, True])
    def test_a_refused_input_never_reaches_prep_or_the_runner(self, as_instance):
        calls = []
        runner = Spy() if as_instance else Spy
        Spy.log = []
        with pytest.raises(IncompatibleInputError):
            self._pipeline(BICYCLE, runner, lambda: calls.append("prep"), calls)
        assert calls == []
        assert "run" not in Spy.log and "prepare" not in Spy.log
        # preflight reads the declaration and the name; it builds nothing and calls nothing
        assert set(Spy.log) <= {"capabilities", "display_name", "name"}
        assert "__init__" not in Spy.log

    def test_an_accepted_input_goes_through_to_the_end(self):
        calls = []
        self._pipeline(LINEAR, Spy(), lambda: calls.append("prep"), calls)
        assert calls == ["prep", "finished"]

    def test_a_registered_adapter_is_not_instantiated(self, three_models):
        made = []
        original = NarrowFold.__init__

        def recording_init(self, *args, **kwargs):
            made.append(True)
            original(self, *args, **kwargs)

        NarrowFold.__init__ = recording_init
        try:
            with pytest.raises(IncompatibleInputError):
                preflight(BICYCLE, [], "narrow")
        finally:
            NarrowFold.__init__ = original
        assert made == []


# ---------------------------------------------------------------------------
# Inputs to preflight
# ---------------------------------------------------------------------------


class TestResolvingSteps:
    def test_a_registry_name_reads_the_capabilities_of_the_spec(self, monkeypatch):
        spec = SimpleNamespace(
            name="antibody_numbering",
            capabilities=Capabilities(
                binder_types={"nanobody", "antibody"}, reasons={"binder_types": "Ig domains only."}
            ),
        )
        monkeypatch.setattr("binding_metrics.metrics.registry.METRICS", [spec])
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(LINEAR, ["antibody_numbering", "interface"])
        (violation,) = caught.value.violations
        assert violation.name == "antibody_numbering" and "peptide (estimated" in violation.fact

    def test_a_metric_the_registry_does_not_know_has_no_limit(self):
        assert preflight(BICYCLE, ["no_such_metric"]).compatible

    def test_the_real_registry_metrics_declare_no_limit_yet_are_accepted(self):
        assert preflight(BICYCLE, ["interface", "ramachandran", "omega"]).compatible

    def test_a_metric_needs_a_name(self):
        with pytest.raises(TypeError, match="registry name"):
            preflight(LINEAR, [object()])

    def test_an_unknown_predictor_name_lists_the_registered_ones(self, three_models):
        with pytest.raises(KeyError, match=r"Unknown predictor 'nope'.*narrow"):
            preflight(LINEAR, [], "nope")

    def test_the_real_of3_adapter_is_found_by_name(self):
        if "of3" not in PARSERS:
            pytest.skip("the of3 adapter is not registered")
        assert preflight(LINEAR, [], "of3").compatible

    def test_a_capabilities_object_can_stand_for_the_predictor(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(BICYCLE, [], OF3_LIKE_CAPABILITIES)
        assert caught.value.violations[0].subject == "predictor (unnamed) 9.9"

    def test_a_parser_and_a_runner_are_both_checked(self, three_models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(BICYCLE, [], [NarrowFold, Spy])
        subjects = {v.subject for v in caught.value.violations}
        assert subjects == {"predictor NarrowFold 9.9", "predictor SpyRunner 9.9"}

    def test_a_wrong_capabilities_attribute_is_a_type_error(self):
        broken = type("Broken", (), {"capabilities": {"closures": ["none"]}})
        with pytest.raises(TypeError, match="must be None or a Capabilities"):
            preflight(LINEAR, [], broken)
        with pytest.raises(TypeError, match="must be None or a Capabilities"):
            preflight(LINEAR, [SimpleNamespace(name="m", capabilities="none")])


class TestWarningsAndNotes:
    def test_a_caveat_is_a_warning_and_never_a_refusal(self):
        careful = SimpleNamespace(
            name="mlff",
            capabilities=Capabilities(
                caveats={"closures:head_to_tail": "Never validated on macrocycles."}
            ),
        )
        ring = dataclasses.replace(
            LINEAR, closures=frozenset({"head_to_tail"}), closure_bonds=(RING,)
        )
        report = preflight(ring, [careful])
        assert report.compatible
        assert report.warnings == ("metric 'mlff': Never validated on macrocycles.",)
        assert preflight(LINEAR, [careful]).warnings == ()

    def test_an_unknown_binder_type_skips_the_type_check_and_says_so(self, monkeypatch, caplog):
        antibody_only = SimpleNamespace(
            name="numbering",
            capabilities=Capabilities(binder_types={"antibody"}, reasons={"binder_types": "Ig."}),
        )
        profile = dataclasses.replace(LINEAR, binder_type="unknown", n_binder_residues=240)
        with caplog.at_level(logging.INFO, logger="binding_metrics.capabilities"):
            report = preflight(profile, [antibody_only])
        assert report.compatible
        assert any("binder type is unknown" in note for note in report.notes)
        assert "binder-type checks" in caplog.text

    def test_needs_the_caller_did_not_describe_are_reported_as_unchecked(self):
        needing = SimpleNamespace(
            name="dockq",
            capabilities=Capabilities(needs={"reference_structure"}, reasons={"needs": "Native."}),
        )
        report = preflight(LINEAR, [needing])
        assert report.compatible
        assert any("did not say what it provides" in note for note in report.notes)
        assert preflight(LINEAR, [needing], provided={"reference_structure"}).notes == ()

    def test_the_notes_of_the_profile_are_kept(self):
        profile = dataclasses.replace(LINEAR, notes=("no bond table",))
        assert "no bond table" in preflight(profile, []).notes


class TestReport:
    def test_to_dict_is_json_ready_and_complete(self, three_models):
        report = preflight(BICYCLE, ["interface"], "narrow", policy="warn")
        data = json.loads(json.dumps(report.to_dict()))
        assert data["policy"] == "warn" and data["compatible"] is False
        assert data["metrics_to_run"] == ["interface"]
        assert data["predictors"] == ["NarrowFold 9.9"]
        assert data["profile"]["closures"] == ["head_to_tail", "disulfide"]
        assert {v["constraint"] for v in data["violations"]} == {"closures"}
        assert data["violations"][0]["subject"] == "predictor NarrowFold 9.9"

    def test_a_compatible_report_formats_the_plan(self):
        text = preflight(LINEAR, ["interface", "coulomb"], VagueFold).format()
        assert text.startswith("Pre-flight check: the input fits everything requested")
        assert "Runs: metrics interface, coulomb; predictor VagueFold" in text

    def test_the_report_is_immutable(self):
        report = preflight(LINEAR, [])
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.policy = "warn"  # type: ignore[misc]
