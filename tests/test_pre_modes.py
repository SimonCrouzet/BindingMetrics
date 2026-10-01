"""The modes a model supports: ``Capabilities.modes``, its check, and the declarations.

A mode is how a model is used for a complex: ``predict`` (sequences only), ``refold`` (receptor
templated, binder free), ``score`` (every chain templated on its own, the pose not given) and
``lock`` (the pose pinned to the input). A hard limit is declared only where a source says the
input cannot be given; "soft", "guidance" and "not reliable" are caveats. The evidence tests read
the model source when a clone is named by ``BINDING_METRICS_MODEL_SOURCES``.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from binding_metrics.capabilities import (
    MODES,
    Capabilities,
    IncompatibleInputError,
    InputProfile,
    preflight,
)
from binding_metrics.predictors.registry import PARSERS, ParserSpec
from tests.predictors.synth_stub import StubParser
from tests.test_pre_structures import model_source_text

PROFILE = InputProfile("B", "A", n_binder_residues=12, binder_type="peptide")


class TestTheField:
    def test_the_vocabulary(self):
        assert MODES == ("predict", "refold", "score", "lock")

    def test_the_store_accepts_the_same_modes(self):
        from binding_metrics.predictors.store import MODES as STORE_MODES

        assert set(STORE_MODES) == set(MODES)

    def test_nothing_is_declared_by_default(self):
        caps = Capabilities()
        assert caps.modes == frozenset() and caps.is_unconstrained
        assert caps.check(PROFILE, mode="lock") == []
        assert not caps.declares_mode("lock")

    def test_a_mode_left_out_needs_a_reason(self):
        with pytest.raises(ValueError, match="modes is constrained but reasons has no sentence"):
            Capabilities(modes={"score"})
        caps = Capabilities(modes={"score"}, reasons={"modes": "Only re-docking."})
        assert caps.constrained_fields() == ("modes",)

    def test_the_full_set_declares_support_and_refuses_nothing(self):
        caps = Capabilities(modes=set(MODES))
        assert caps.is_unconstrained and caps.constrained_fields() == ()
        assert all(caps.declares_mode(mode) for mode in MODES)
        assert all(caps.check(PROFILE, mode=mode) == [] for mode in MODES)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="modes has unknown value"):
            Capabilities(modes={"dock"})
        with pytest.raises(ValueError, match="mode must be one of"):
            Capabilities().check(PROFILE, mode="dock")
        with pytest.raises(ValueError, match="mode must be one of"):
            preflight(PROFILE, [], mode="dock")

    def test_a_reason_and_a_caveat_may_name_a_mode(self):
        caps = Capabilities(
            modes={"score"},
            reasons={"modes": "General.", "modes:lock": "About the pose."},
            caveats={"modes:score": "Not validated."},
        )
        assert caps.reason_for("modes", "lock") == "About the pose."
        assert caps.reason_for("modes", "refold") == "General."
        with pytest.raises(ValueError, match="unknown key 'modes:dock'"):
            Capabilities(caveats={"modes:dock": "x"})


class TestTheCheck:
    caps = Capabilities(
        modes={"predict", "refold", "score"},
        reasons={"modes": "The pose cannot be given."},
    )

    @pytest.mark.parametrize("mode", ["predict", "refold", "score"])
    def test_a_listed_mode_passes(self, mode):
        assert self.caps.check(PROFILE, mode=mode) == []

    def test_a_mode_left_out_is_refused_with_the_fact_and_the_supported_modes(self):
        (violation,) = self.caps.check(PROFILE, mode="lock")
        assert violation.constraint == "modes"
        assert violation.fact == "mode 'lock' was requested"
        assert violation.requirement == "supported modes: predict, refold, score"
        assert violation.reason == "The pose cannot be given."

    def test_no_mode_means_not_checked(self):
        assert self.caps.check(PROFILE) == []

    def test_a_caveat_applies_to_the_requested_mode_only(self):
        caps = Capabilities(caveats={"modes:lock": "Soft."})
        assert caps.caveats_for(PROFILE, "lock") == ["Soft."]
        assert caps.caveats_for(PROFILE, "score") == []
        assert caps.caveats_for(PROFILE) == []
        assert caps.check(PROFILE, mode="lock") == []  # a caveat never refuses


# ---------------------------------------------------------------------------
# preflight with a mode
# ---------------------------------------------------------------------------


class LockFold(StubParser):
    name = "lockfold"
    display_name = "LockFold"
    capabilities = Capabilities(modes={"predict", "lock"}, reasons={"modes": "Predicts and pins."})


class ScoreFold(StubParser):
    name = "scorefold"
    display_name = "ScoreFold"
    capabilities = Capabilities(modes={"score"}, reasons={"modes": "Only re-docking."})


class VagueFold(StubParser):
    name = "vague"
    display_name = "VagueFold"


class NoModesFold(StubParser):
    """Declares limits, but no modes: it accepts the input and says nothing about the mode."""

    name = "nomodes"
    display_name = "NoModesFold"
    capabilities = Capabilities(closures={"none"}, reasons={"closures": "Linear only."})


class SoftFold(StubParser):
    name = "softfold"
    display_name = "SoftFold"
    capabilities = Capabilities(caveats={"modes:lock": "The pose is only guided."})


@pytest.fixture
def models(monkeypatch):
    for cls in (LockFold, ScoreFold, VagueFold, NoModesFold, SoftFold):
        spec = ParserSpec(
            name=cls.name,
            import_path=f"{cls.__module__}:{cls.__name__}",
            display_name=cls.display_name,
            family=cls.family,
        )
        monkeypatch.setitem(PARSERS, cls.name, spec)
    # the models of the package are out of the way of the lists these tests read
    for name in ("af2", "boltz2", "of3", "protenix"):
        monkeypatch.delitem(PARSERS, name, raising=False)


class TestPreflightWithAMode:
    def test_a_mode_the_model_does_not_list_is_refused_with_the_two_lists(self, models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "scorefold", mode="lock")
        (violation,) = caught.value.violations
        assert (violation.kind, violation.constraint) == ("predictor", "modes")
        message = str(caught.value)
        assert "predictor ScoreFold: modes" in message
        assert "found:    mode 'lock' was requested" in message
        assert "requires: supported modes: score" in message
        assert "Only re-docking." in message
        assert "Mode: lock (the pose of the chains pinned to the input)" in message
        declared = "predictors that declare the mode 'lock' whose declared limits accept this input"
        assert f"{declared}: LockFold (lockfold)" in message
        assert "predictors that declare no limits (not validated for this input)" in message
        # the models that declare a different set, the rejected one and the lister of no modes
        assert "ScoreFold (scorefold)" not in message.split("fix:")[1]

    def test_the_undeclared_list_holds_the_models_with_no_declaration_of_modes(self, models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "scorefold", mode="lock")
        fix = caught.value.violations[0].fix
        undeclared = fix.split("not validated for this input): ")[1]
        for label in ("NoModesFold (nomodes)", "SoftFold (softfold)", "VagueFold (vague)"):
            assert label in undeclared
        assert "LockFold" not in undeclared

    def test_a_mode_nobody_declares_is_said_plainly(self, models, monkeypatch):
        monkeypatch.delitem(PARSERS, "lockfold")
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "scorefold", mode="lock")
        assert (
            "no other registered predictor declares support for this input in the mode 'lock'"
            in str(caught.value)
        )

    def test_a_listed_mode_passes(self, models):
        report = preflight(PROFILE, [], "scorefold", mode="score")
        assert report.compatible and report.mode == "score"

    def test_no_mode_is_not_checked(self, models):
        report = preflight(PROFILE, [], "scorefold")
        assert report.compatible and report.mode is None

    def test_a_model_without_declared_modes_is_never_refused_for_one(self, models):
        assert preflight(PROFILE, [], "vague", mode="lock").compatible
        assert preflight(PROFILE, [], "nomodes", mode="lock").compatible

    def test_a_caveat_is_a_warning_for_the_requested_mode(self, models):
        report = preflight(PROFILE, [], "softfold", mode="lock")
        assert report.compatible
        assert report.warnings == ("predictor SoftFold: The pose is only guided.",)
        assert preflight(PROFILE, [], "softfold", mode="score").warnings == ()

    def test_models_that_cannot_be_run_from_here_are_marked(self, models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "scorefold", mode="lock", runnable={"scorefold", "vague"})
        fix = caught.value.violations[0].fix
        assert "LockFold (lockfold) [no runner here: give its output with --prediction-dir]" in fix
        assert "VagueFold (vague)" in fix and "VagueFold (vague) [no runner" not in fix

    def test_without_the_runnable_list_nothing_is_marked(self, models):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "scorefold", mode="lock")
        assert "no runner here" not in caught.value.violations[0].fix

    def test_a_mode_does_not_apply_to_metrics(self, models):
        needing = type(
            "Spec",
            (),
            {"name": "m", "capabilities": Capabilities(modes={"score"}, reasons={"modes": "x"})},
        )()
        assert preflight(PROFILE, [needing], mode="lock").compatible

    def test_the_report_names_the_mode(self, models):
        data = preflight(PROFILE, [], "scorefold", mode="score").to_dict()
        assert data["mode"] == "score"
        json.dumps(data)
        assert preflight(PROFILE, []).to_dict()["mode"] is None

    def test_the_other_alternatives_keep_their_rule_without_a_mode(self, models):
        closed = dataclasses.replace(
            PROFILE, closures=frozenset({"disulfide"}), residue_names={}, closure_bonds=()
        )
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(closed, [], "nomodes")
        assert (
            "predictors whose declared limits accept this input" in caught.value.violations[0].fix
        )

    def test_the_log_names_the_mode_violation(self, models, caplog):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            preflight(PROFILE, [], "scorefold", mode="lock", policy="warn")
        assert "mode 'lock' was requested" in caplog.text


# ---------------------------------------------------------------------------
# What the models of the package declare
# ---------------------------------------------------------------------------


class TestTheDeclarations:
    def _caps(self, name):
        return PARSERS[name].load_capabilities()

    def test_openfold3_supports_predict_refold_and_score_but_not_lock(self):
        caps = self._caps("of3")
        assert caps.modes == frozenset({"predict", "refold", "score"})
        reason = caps.reason_for("modes", "lock")
        for cited in (
            "create_template_distogram",
            "template_embedders.py",
            "template_how_to.md",
            "input_format_reference.md, section 4",
            "was not confirmed",
        ):
            assert cited in reason

    def test_lock_is_refused_for_openfold3_with_the_alternatives(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "of3", mode="lock")
        message = str(caught.value)
        assert "predictor OpenFold3 0.5.0: modes" in message
        assert "supported modes: predict, refold, score" in message
        assert "Boltz-2 (boltz2)" in message
        assert "OpenFold3 (of3)" not in message.split("fix:")[1]

    @pytest.mark.parametrize("mode", ["predict", "refold", "score"])
    def test_the_other_modes_pass_for_openfold3(self, mode):
        assert preflight(PROFILE, [], "of3", mode=mode).compatible

    def test_boltz2_declares_all_four_and_warns_for_lock(self):
        caps = self._caps("boltz2")
        assert caps.modes == frozenset(MODES) and caps.is_unconstrained
        report = preflight(PROFILE, [], "boltz2", mode="lock")
        assert report.compatible
        (warning,) = [w for w in report.warnings if "force: true" in w]
        assert "guidance term" in warning and "never run" in warning
        assert preflight(PROFILE, [], "boltz2", mode="score").warnings == ()

    def test_protenix_declares_no_modes_and_warns_for_lock(self):
        caps = self._caps("protenix")
        assert caps.modes == frozenset() and caps.is_unconstrained
        (warning,) = preflight(PROFILE, [], "protenix", mode="lock").warnings
        assert "soft constraint" in warning and "docs/infer_json_format.md" in warning
        for mode in ("predict", "refold", "score"):
            assert preflight(PROFILE, [], "protenix", mode=mode).warnings == ()

    def test_alphafold2_declares_nothing_about_modes(self):
        caps = self._caps("af2")
        assert caps.modes == frozenset() and not [k for k in caps.caveats if k.startswith("modes")]
        assert preflight(PROFILE, [], "af2", mode="lock").compatible

    def test_the_lists_for_lock_name_boltz2_as_declared_and_the_others_as_undeclared(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "of3", mode="lock", runnable={"of3"})
        fix = caught.value.violations[0].fix
        declared, undeclared = fix.split("; predictors that declare no limits")
        assert (
            "Boltz-2 (boltz2) [no runner here: give its output with --prediction-dir]" in declared
        )
        for model in ("AlphaFold2 / ColabFold (af2)", "Protenix (protenix)"):
            assert model in undeclared


class TestTheEvidence:
    """Re-read the model sources when the clones are at hand (OpenFold3 v0.5.0, Boltz v2.2.1)."""

    def test_openfold3_zeroes_the_template_pair_features_between_chains(self):
        features = model_source_text(
            "openfold-3", "openfold3/core/data/primitives/featurization/template.py"
        )
        assert "multichain_pair_mask: torch.Tensor" in features
        assert "* multichain_pair_mask" in features
        pipeline = model_source_text(
            "openfold-3", "openfold3/core/data/pipelines/featurization/template.py"
        )
        assert "multichain_pair_mask = (asym_id[..., None] == asym_id[..., None, :])" in pipeline
        assert pipeline.count("multichain_pair_mask,") >= 2  # distogram and unit vector
        embedder = model_source_text(
            "openfold-3", "openfold3/core/model/feature_embedders/template_embedders.py"
        )
        assert 'batch["asym_id"][..., None] == batch["asym_id"][..., None, :]' in embedder

    def test_an_openfold3_cif_template_gives_one_chain_and_the_pocket_is_for_ligands(self):
        how_to = model_source_text("openfold-3", "docs/source/template_how_to.md")
        assert "For multi-chain CIF files, only the best matching chain per file is used." in how_to
        reference = model_source_text("openfold-3", "docs/source/input_format_reference.md")
        assert "Optional ligand-to-pocket constraint for a small-molecule ligand" in reference

    def test_openfold3_reads_covalent_bonds_nowhere_so_no_other_channel_exists(self):
        import os
        from pathlib import Path

        model_source_text("openfold-3", "openfold3/__init__.py")  # skips without the clone
        root = Path(os.environ["BINDING_METRICS_MODEL_SOURCES"]) / "openfold-3" / "openfold3"
        users = [
            path.name
            for path in root.rglob("*.py")
            if "covalent_bonds" in path.read_text(encoding="utf-8")
        ]
        assert users == ["inference_query_format.py"]

    def test_boltz2_templates_carry_no_pose_unless_forced(self):
        trunk = model_source_text("boltz", "src/boltz/model/modules/trunkv2.py")
        assert "template features only attend within the same chain" in trunk

    def test_boltz2_forces_a_template_with_a_guidance_potential(self):
        potentials = model_source_text("boltz", "src/boltz/model/potentials/potentials.py")
        start = potentials.index("TemplateReferencePotential(\n                    parameters=")
        assert '"guidance_weight": 0.1' in potentials[start : start + 400]
        assert "weighted_rigid_align(" in potentials
        featurizer = model_source_text("boltz", "src/boltz/data/feature/featurizerv2.py")
        assert (
            "name_to_templates.setdefault(template_info.name, []).append(template_info)"
            in featurizer
        )
        docs = model_source_text("boltz", "docs/prediction.md")
        assert "force: true" in docs and "controls the distance (in Angstroms)" in docs
        assert "chain_id: CHAIN_ID" in docs

    def test_protenix_calls_its_constraints_soft(self):
        docs = model_source_text("Protenix", "docs/infer_json_format.md")
        assert (
            "This is a **soft constraint**: the model is encouraged, but not strictly required"
            in docs
        )
        assert "templatesPath" in docs and "`.a3m`" in docs


class TestTheRunnerStopsALock:
    """The pre-flight check normally refuses ``lock`` first; the runner is the second stop."""

    def test_the_request_cannot_be_made(self):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner
        from tests.test_pre_cli_run import LINEAR

        with pytest.raises(ValueError, match="OpenFold3 cannot run mode 'lock'") as caught:
            OpenFold3Runner().make_request(
                LINEAR, name="s", binder_chain="B", receptor_chain="A", mode="lock"
            )
        assert "create_template_distogram" in str(caught.value)  # the text of the declaration

    def test_a_request_of_mode_lock_is_not_prepared_or_run(self, tmp_path):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner
        from binding_metrics.predictors.store import PredictionRequest
        from tests.test_pre_cli_run import LINEAR

        request = PredictionRequest(
            "of3", "s", mode="lock", input_path=LINEAR, binder_chain="B", receptor_chain="A"
        )
        runner = OpenFold3Runner()
        for action in (runner.prepare, runner.run):
            with pytest.raises(ValueError, match="cannot run mode 'lock'"):
                action(request, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_the_other_modes_are_not_affected(self):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner
        from tests.test_pre_cli_run import LINEAR

        request = OpenFold3Runner().make_request(
            LINEAR, name="s", binder_chain="B", receptor_chain="A", mode="refold"
        )
        assert request.mode == "refold"
