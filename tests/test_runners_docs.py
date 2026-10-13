"""The documentation of the runners says what the runners and the command line do.

The table of ``docs/metrics.md`` ("Runners") is compared with the class attributes of the
runners, the options it names with the parsers of ``binding-metrics-run`` and ``-batch``, and the
modes that the docs combine with a model with the modes of its runner. Sentences that the
runners made false are listed in ``STALE`` below.
"""

import argparse
import importlib
import re
import sys
from pathlib import Path

import pytest

from binding_metrics.capabilities import MODES
from binding_metrics.cli import prediction
from binding_metrics.cli.prediction import RUNNERS, runner_class
from binding_metrics.predictors.runners import PredictionRunner
from tests.test_runners_support import MODELS

ROOT = Path(__file__).parent.parent
METRICS = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
PREFLIGHT = (ROOT / "docs" / "preflight.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Sentences that were true while OpenFold3 was the only model that could be run from here, and
# are false now that every adapter has a runner. Add a row when a text is found to say it again;
# do not add a row for a rewording. Matched on the text with its white space folded.
STALE = {
    "Only OpenFold3 can be run from here": "af2, boltz2 and protenix have runners",
    "and is the only runner": "af2, boltz2 and protenix have runners",
    "None of the adapters starts a model: OpenFold3 is the only model": "every adapter has one",
    "`predict` cannot be run from here": "af2 and protenix run predict",
    "has no runner yet and that its output is passed": "every adapter has a runner",
    "marked as having no runner here": "every adapter has a runner",
}


def flat(text: str) -> str:
    return " ".join(text.split())


def runner_rows() -> dict[str, list[str]]:
    """The rows of the table under "#### Runners", by model key."""
    section = METRICS.split("#### Runners", 1)[1].split("\n### ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        match = re.fullmatch(r"`(\w+)`", cells[0]) if cells else None
        if line.startswith("|") and match and len(cells) >= 9:
            rows[match.group(1)] = cells
    return rows


def command_line_parsers():
    parsers = {}
    for name in ("run", "batch"):
        module = importlib.import_module(f"binding_metrics.cli.{name}")
        parsers[name] = _captured_parser(module)
    return parsers


def _captured_parser(module):
    """The parser that ``module.main`` builds, taken at the call of ``parse_args``."""
    holder = {}

    def stop(self, *args, **kwargs):
        holder["parser"] = self
        raise SystemExit(0)

    mp = pytest.MonkeyPatch()
    mp.setattr(argparse.ArgumentParser, "parse_args", stop)
    mp.setattr(sys, "argv", ["prog"])
    try:
        with pytest.raises(SystemExit):
            module.main()
    finally:
        mp.undo()
    return holder["parser"]


class TestTheRunnerTable:
    def test_there_is_a_row_for_every_registered_runner(self):
        assert sorted(runner_rows()) == MODELS == sorted(RUNNERS)

    @pytest.mark.parametrize("model", MODELS)
    def test_the_modes_and_the_default_are_those_of_the_class(self, model):
        cell = runner_rows()[model][3]
        listed = set(re.findall(r"`([a-z-]+)`", cell.split("(")[0]))
        default = re.search(r"\(`([a-z-]+)`\)", cell).group(1)
        assert listed == prediction.runner_modes(model)
        assert default == prediction.runner_default_mode(model)

    @pytest.mark.parametrize("model", MODELS)
    def test_the_weights_kind_is_the_one_of_the_class(self, model):
        kind = runner_class(model).weights_kind
        assert runner_class(model).supports_custom_weights
        assert kind in runner_rows()[model][5]

    @pytest.mark.parametrize("model", MODELS)
    def test_the_runner_class_and_module_are_named(self, model):
        cell = runner_rows()[model][1]
        module, cls = RUNNERS[model].split(":")
        assert f"`{cls}`" in cell and module.rsplit(".", 1)[1] in cell

    def test_only_colabfold_renames_the_chains(self):
        rows = runner_rows()
        for model in MODELS:
            maps = runner_class(model).output_chain_map
            renames = model == "af2"
            assert ("receptor `A`, binder `B`" in rows[model][8]) is renames
            assert (maps.__qualname__.split(".")[0] != "PredictionRunner") is renames

    def test_the_cyclic_setting_is_where_the_runner_has_the_keyword(self):
        rows = runner_rows()
        for model in MODELS:
            has_keyword = prediction.takes_setting(runner_class(model), "binder_cyclic")
            assert has_keyword == ("no setting" not in rows[model][7])

    def test_the_default_mode_help_says_what_the_classes_say(self):
        text = flat(command_line_parsers()["run"].format_help())
        for model in MODELS:
            modes = prediction.runner_modes(model)
            if modes == set(MODES):
                assert f"{model}: all five" in text
            elif model in ("of3", "boltz2", "af2"):
                assert f"{model}: {', '.join(sorted(modes))}" in text or f"{model} and" in text
        assert "af2 and protenix: predict" in text


class TestTheOptionsAreDocumented:
    NEW = (
        "--prediction-cyclic",
        "--prediction-no-msa-server",
        "--prediction-conda-env",
        "--prediction-lock-threshold",
    )

    @pytest.mark.parametrize("command", ["run", "batch"])
    @pytest.mark.parametrize("option", NEW)
    def test_the_option_exists_on_both_commands_and_in_the_docs(self, command, option):
        parser = command_line_parsers()[command]
        assert option in {o for action in parser._actions for o in action.option_strings}
        assert f"`{option}" in METRICS and f"`{option}" in README

    def test_the_python_keywords_are_documented(self):
        for keyword in (
            "prediction_cyclic",
            "prediction_use_msa_server",
            "prediction_conda_env",
            "prediction_lock_threshold",
        ):
            assert keyword in METRICS
        for command in ("run", "batch"):
            function = {"run": "run_pipeline", "batch": "run_batch"}[command]
            module = importlib.import_module(f"binding_metrics.cli.{command}")
            import inspect

            names = inspect.signature(getattr(module, function)).parameters
            assert {
                "prediction_cyclic",
                "prediction_use_msa_server",
                "prediction_conda_env",
                "prediction_lock_threshold",
            } <= set(names)

    def test_the_runner_attributes_are_documented(self):
        for text in ("`supported_modes`", "`default_mode`", "`output_chain_map(request)`"):
            assert text in METRICS
            assert hasattr(PredictionRunner, text.strip("`").split("(")[0]), text


class TestTheDocsAgreeWithTheRunners:
    def test_no_text_says_again_what_the_runners_made_false(self):
        texts = {"README.md": README, "docs/metrics.md": METRICS, "docs/preflight.md": PREFLIGHT}
        found = {
            f"{sentence!r} ({reason})": [
                name for name, text in texts.items() if sentence in flat(text)
            ]
            for sentence, reason in STALE.items()
        }
        assert not {key: names for key, names in found.items() if names}

    def test_every_mode_the_docs_combine_with_an_option_is_in_the_vocabulary(self):
        """In a code span, ``--prediction-mode MODE`` names a mode of the vocabulary, and with
        ``--predictor MODEL`` in front of it a mode that the runner of MODEL runs."""
        documents = (
            ("README.md", README),
            ("docs/metrics.md", METRICS),
            ("docs/preflight.md", PREFLIGHT),
        )
        examples = 0
        for name, text in documents:
            for span in re.findall(r"`([^`]*--prediction-mode [a-z][^`]*)`", text):
                mode = re.search(r"--prediction-mode ([a-z][a-z-]*)", span).group(1)
                assert mode in MODES, f"{name}: {mode!r} is not a mode ({span!r})"
                model = re.search(r"--predictor (\w+)", span)
                if model:
                    assert mode in prediction.runner_modes(model.group(1)), f"{name}: {span!r}"
                examples += 1
        assert examples, "no example with --prediction-mode in a code span"
