"""The fifth mode, ``refold-lock``: refold with the receptor pinned by a forced template.

Only the vocabulary is under test here: the word is in the store, in the capabilities and on the
command line, OpenFold3 refuses it with a sentence of its own, Boltz-2 declares it, and a model
that does not run it is refused before anything runs. The keys of the four older modes are pinned
to the values they had before the word existed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from binding_metrics import capabilities
from binding_metrics.capabilities import (
    LOCK_MODES,
    MODES,
    IncompatibleInputError,
    InputProfile,
    preflight,
)
from binding_metrics.cli import prediction
from binding_metrics.predictors import store
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.registry import PARSERS
from binding_metrics.predictors.store import PredictionRequest
from tests.test_pre_cli_run import LINEAR, Tripwires

PROFILE = InputProfile("B", "A", n_binder_residues=12, binder_type="peptide")

#: Keys of requests built before ``refold-lock`` existed (the same construction, see
#: ``build_request``): a request of an older mode must keep its key.
KEYS_BEFORE = {
    "predict": "031693385b8b5dd8ac106f29d593d1568ac23859d2dd3d12bc798e0ae61cce59",
    "score": "d471a11ff556433f9c0c56ed46982c4f87d2b5ff41dfe100962cbb68b8a32715",
    "refold": "56fb84455b4b2abc8de1dec6d098a312ee64b5540f14da35bb6f594fddc75f70",
    "score-lock": "817c3f683493db3209c1e0310db58c082fbab82abb8365d6f59c5098eca83f18",
}
KEY_OF_SEQUENCES_BEFORE = "44e4e705ee6ffe99daaa8ce7f1341f538ccba24bf3fce3ee8f7e88fef5a45b8a"
KEY_WITHOUT_SERVER_BEFORE = "1fc5fb3645a09d02c2dabb9f796542bbb8a24733c5a393aa7483aacc9dffa634"

OPTIONS = {
    "use_msa_server": True,
    "binder_cyclic": "auto",
    "extra_args": [],
    "presets": ["predict"],
}


def build_request(tmp_path: Path, mode: str, **changes) -> PredictionRequest:
    complex_file = tmp_path / "complex.cif"
    complex_file.write_bytes(b"data_x\n#pinned complex\n")
    fields = dict(
        mode=mode,
        input_path=complex_file,
        binder_chain="B",
        receptor_chain="A",
        seeds=(42,),
        num_samples=5,
        model_version="1.0",
        options=OPTIONS,
    )
    fields.update(changes)
    return PredictionRequest("stub", "s1", **fields)


class TestTheVocabulary:
    def test_the_store_and_the_capabilities_have_the_five_modes(self):
        assert store.MODES == ("predict", "score", "refold", "score-lock", "refold-lock")
        assert MODES == ("predict", "refold", "score", "score-lock", "refold-lock")
        assert set(store.MODES) == set(MODES)

    def test_the_lock_modes_are_the_two_that_force_a_template(self):
        assert LOCK_MODES == ("score-lock", "refold-lock")
        assert set(LOCK_MODES) <= set(MODES)

    def test_every_mode_has_its_plan_text(self):
        assert set(capabilities._MODE_TEXT) == set(MODES)
        text = capabilities._MODE_TEXT["refold-lock"]
        assert text.startswith("refold-lock (") and "receptor" in text and "forced template" in text

    def test_a_request_can_have_the_mode_and_it_is_part_of_the_key(self, tmp_path):
        keys = {mode: build_request(tmp_path, mode).key() for mode in MODES}
        assert len(set(keys.values())) == len(MODES)
        assert build_request(tmp_path, "refold-lock").mode == "refold-lock"

    @pytest.mark.parametrize("mode", sorted(KEYS_BEFORE))
    def test_a_request_of_an_older_mode_keeps_its_key(self, tmp_path, mode):
        assert build_request(tmp_path, mode).key() == KEYS_BEFORE[mode]

    def test_other_requests_keep_their_keys(self, tmp_path):
        assert (
            PredictionRequest("stub", "s1", sequences={"A": "GGG", "B": "AAA"}, options={}).key()
            == KEY_OF_SEQUENCES_BEFORE
        )
        no_server = build_request(tmp_path, "refold", options={**OPTIONS, "use_msa_server": False})
        assert no_server.key() == KEY_WITHOUT_SERVER_BEFORE

    def test_an_unknown_mode_is_still_refused(self, tmp_path):
        with pytest.raises(ValueError, match="mode must be one of"):
            build_request(tmp_path, "pin")


class TestOpenFold3:
    caps = PARSERS["of3"].load_capabilities()

    def test_the_declaration_keeps_its_three_modes_and_the_runner_its_two(self):
        assert self.caps.modes == frozenset({"predict", "refold", "score"})
        assert OpenFold3Runner.supported_modes == frozenset({"score", "refold"})

    def test_the_reason_says_it_cannot_force_a_template(self):
        reason = self.caps.reason_for("modes", "refold-lock")
        assert "cannot force a template" in reason and "only guides it per chain" in reason
        assert "'refold'" in reason
        # the sentence of score-lock is the general one and has not changed
        general = self.caps.reason_for("modes", "score-lock")
        assert general == self.caps.reasons["modes"] and general != reason
        assert "create_template_distogram" in general

    def test_the_preflight_refuses_it_with_that_sentence(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "of3", mode="refold-lock")
        (violation,) = caught.value.violations
        assert violation.constraint == "modes"
        assert violation.fact == "mode 'refold-lock' was requested"
        assert violation.requirement == "supported modes: predict, refold, score"
        assert violation.reason == self.caps.reason_for("modes", "refold-lock")
        assert "Mode: refold-lock (refold, with the receptor pinned" in str(caught.value)

    def test_the_runner_refuses_it_with_the_same_sentence(self):
        sentence = self.caps.reason_for("modes", "refold-lock")
        with pytest.raises(ValueError, match="OpenFold3 cannot run mode 'refold-lock'") as caught:
            OpenFold3Runner().make_request(
                LINEAR, name="s", binder_chain="B", receptor_chain="A", mode="refold-lock"
            )
        assert sentence in str(caught.value)

    @pytest.mark.parametrize("step", ["prepare", "run"])
    def test_a_request_made_by_hand_is_refused_before_anything_starts(self, tmp_path, step):
        request = PredictionRequest(
            "of3",
            "s",
            mode="refold-lock",
            input_path=LINEAR,
            binder_chain="B",
            receptor_chain="A",
        )
        with pytest.raises(ValueError, match="OpenFold3 cannot run mode 'refold-lock'"):
            getattr(OpenFold3Runner(), step)(request, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_score_lock_is_refused_as_before(self):
        with pytest.raises(ValueError, match="OpenFold3 cannot run mode 'score-lock'"):
            OpenFold3Runner().make_request(
                LINEAR, name="s", binder_chain="B", receptor_chain="A", mode="score-lock"
            )


class TestOnlyBoltz2RunsIt:
    """Three of the four models are refused, by their declaration or by their runner."""

    #: who refuses the mode in a run from here: the pre-flight check (the declaration does not
    #: list it) or the argument check (the runner does not run it and the model declares nothing)
    REFUSED_BY = {
        "of3": "preflight",
        "af2": "arguments",
        "protenix": "arguments",
    }

    def test_every_model_but_boltz2_is_covered(self):
        assert sorted(self.REFUSED_BY) == sorted(set(PARSERS) - {"boltz2"})

    @pytest.mark.parametrize("model", sorted(REFUSED_BY))
    def test_the_arguments_are_refused_or_left_to_the_preflight_check(self, model):
        if self.REFUSED_BY[model] == "arguments":
            with pytest.raises(ValueError, match="cannot be run from here"):
                prediction.check_predictor(model, None, "refold-lock")
        else:
            assert prediction.check_predictor(model, None, "refold-lock").mode == "refold-lock"

    def test_the_arguments_of_boltz2_are_accepted(self):
        assert prediction.check_predictor("boltz2", None, "refold-lock").mode == "refold-lock"

    def test_the_preflight_check_refuses_it_and_names_boltz2_as_the_model_that_declares_it(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(PROFILE, [], "of3", mode="refold-lock")
        (violation,) = caught.value.violations
        assert violation.fact == "mode 'refold-lock' was requested"
        assert violation.reason  # a refusal always says why
        assert violation.fix.startswith(
            "predictors that declare the mode 'refold-lock' whose declared limits accept this "
            "input: Boltz-2 (boltz2);"
        )

    def test_boltz2_is_accepted_in_all_five_modes(self):
        for mode in MODES:
            assert preflight(PROFILE, [], "boltz2", mode=mode).compatible

    def test_boltz2_is_accepted_in_this_mode_with_the_caveat_of_a_guidance_term(self):
        report = preflight(PROFILE, [], "boltz2", mode="refold-lock")
        (warning,) = report.warnings
        assert warning.startswith("predictor Boltz-2 2.2.1: ")
        assert "force: true" in warning and "guidance term" in warning
        assert "weight 0.1" in warning and "measured once" in warning

    def test_the_runner_of_boltz2_builds_an_input_for_it(self):
        from binding_metrics.predictors.boltz2_runner import RUN_MODES

        assert set(RUN_MODES) == set(MODES)

    @pytest.mark.parametrize("model", ["of3"])
    def test_a_run_is_refused_before_anything_starts(self, model, tmp_path, monkeypatch):
        from binding_metrics.cli.run import run_pipeline

        tripwires = Tripwires(monkeypatch)
        out = tmp_path / "out"
        with pytest.raises(IncompatibleInputError, match="mode 'refold-lock' was requested"):
            run_pipeline(
                LINEAR,
                out,
                skip_prep=True,
                skip_relax=True,
                metrics=frozenset({"openfold"}),
                openfold_conda_env=None,
                peptide_chain="B",
                receptor_chain="A",
                predictor=model,
                prediction_mode="refold-lock",
            )
        assert tripwires.calls == [] and not out.exists()

    @pytest.mark.parametrize("model", ["boltz2", *sorted(REFUSED_BY)])
    def test_an_output_made_elsewhere_can_be_read_in_the_mode(self, model, tmp_path):
        prediction.check_predictor(model, tmp_path, "refold-lock")  # no error
        request = prediction.make_request(
            model,
            "s",
            LINEAR,
            binder_chain="B",
            receptor_chain="A",
            adopt=True,
            prediction_mode="refold-lock",
        )
        assert request.mode == "refold-lock"


class TestTheOption:
    def _parser(self):
        parser = argparse.ArgumentParser()
        prediction.add_prediction_args(parser)
        return parser

    def test_the_mode_is_a_choice_and_the_help_says_what_it_is(self):
        parser = self._parser()
        assert parser.parse_args(["--prediction-mode", "refold-lock"]).prediction_mode == (
            "refold-lock"
        )
        action = parser._option_string_actions["--prediction-mode"]
        assert tuple(action.choices) == MODES
        assert "refold-lock (refold, with the receptor pinned to the input" in action.help
        assert "the binder free" in action.help

    @pytest.mark.parametrize("command", ["run", "batch"])
    def test_both_commands_take_the_mode(self, command, monkeypatch):
        from tests.test_w3a_cli_help import CLI_MODULES, capture_parser

        parser = capture_parser(CLI_MODULES[command], monkeypatch)
        action = parser._option_string_actions["--prediction-mode"]
        assert "refold-lock" in action.choices

    def test_it_needs_a_predictor(self):
        parser = self._parser()
        args = parser.parse_args(["--prediction-mode", "refold-lock"])
        with pytest.raises(SystemExit):
            prediction.check_prediction_args(parser, args)

    def test_the_lock_threshold_goes_with_both_lock_modes_only(self):
        for mode in LOCK_MODES:
            options = prediction.resolve_prediction_options(
                "boltz2", None, prediction_mode=mode, prediction_lock_threshold=1.5
            )
            assert options.mode == mode and options.lock_threshold == 1.5
        with pytest.raises(ValueError, match="threshold of the mode score-lock or refold-lock"):
            prediction.resolve_prediction_options(
                "boltz2", None, prediction_mode="refold", prediction_lock_threshold=1.5
            )

    def test_a_runner_without_the_threshold_refuses_it_in_a_lock_mode(self):
        with pytest.raises(ValueError, match="--prediction-lock-threshold is not available"):
            prediction.resolve_prediction_options(
                "of3", None, prediction_mode="refold", prediction_lock_threshold=1.5
            )
