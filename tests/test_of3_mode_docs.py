"""What the text says about the OpenFold3 modes matches what the code does.

A template carries the fold of one chain and no cross-chain geometry: each query chain gets its
own template structures (the template CIFs here are single-chain files), and OpenFold3's
``_embed_feats`` applies the same-chain mask to the validity indicators of the template pair
features. So ``score`` does not hand OpenFold3 the pose of the input. The text used to say that
both chains were given as templates "so that OF3 evaluates the known conformation", and that an
adversary run in score mode agrees with the design "partly by construction". The tests keep that
wording out and check the facts the text names: the result keys, the options and the arguments.
The binder RMSD against the input, in the receptor frame, is measured in both modes, next to
``delta_com_angstrom`` (``TestTheClaimsAboutTheResultKeys`` runs it).
"""

import argparse
import inspect
import math
import re
from pathlib import Path

import pytest

from binding_metrics.cli import OPENFOLD_MODE_HELP, OPENFOLD_TEMPLATE_CHOICES
from binding_metrics.cli.run import run_pipeline
from tests.test_feat_c_docs import _capture_parser
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
)

ROOT = Path(__file__).parent.parent

# Wording that a real run, or a reading of the OpenFold3 source, disproved. Add a row when a text
# is found to carry such a claim; do not add a row for a rewording. Scanned in README.md,
# docs/metrics.md and every file of src/.
BANNED = {
    # The old reading of the modes: the template does not carry the pose of the input.
    "both chains as templates": "a template carries one chain's fold and no cross-chain geometry",
    "known conformation": "OpenFold3 places the binder itself in score mode",
    "partly by construction": "the adversary run in score mode is not told the pose",
    "templated on the design pose": "the templates are single-chain files of the input",
    "templated on the input pose": "the templates are single-chain files of the input",
    "receptor fixed as template": "the receptor is one template among the chains, not fixed",
    "fixed as template": "same: no template fixes the pose",
    # The result keys are measured in both modes (TestTheClaimsAboutTheResultKeys).
    "refold` mode only": "binder_ca_rmsd and delta_com_angstrom are given in score and refold",
    "NaN in `score`": "binder_ca_rmsd is a number in score mode",
    # What the template embedder does (read from the source of v0.5.0, not run): only the
    # validity indicators of the template pair features are masked.
    "every template feature": "the mask applies to the validity indicators only",
    "all template features": "the mask applies to the validity indicators only",
    "restricts the pair masks": "the mask applies to the validity indicators only",
}

# The result keys that the mode help and the modes paragraph name must be keys of the block that a
# run writes (test_the_predictor_step_measures_both_keys_in_both_modes).
RESULT_KEY = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
BACKTICKED_KEY = re.compile(r"`([a-z]+(?:_[a-z]+)+)`")


def _texts():
    files = [ROOT / "README.md", ROOT / "docs" / "metrics.md"]
    files += sorted((ROOT / "src" / "binding_metrics").rglob("*.py"))
    return [(path, path.read_text(encoding="utf-8")) for path in files]


def _paragraph(marker: str) -> str:
    text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    start = text.index(marker)
    return text[start : text.index("\n\n", start)]


def _openfold_subcommands(monkeypatch) -> dict:
    parser = _capture_parser("binding_metrics.metrics.openfold", monkeypatch)
    (subparsers,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return subparsers.choices


def test_the_wording_a_real_run_disproved_is_gone_from_the_source_and_the_docs():
    texts = _texts()
    found = {
        f"{phrase!r} ({reason})": [
            str(path.relative_to(ROOT)) for path, text in texts if phrase in text
        ]
        for phrase, reason in BANNED.items()
    }
    assert not {key: paths for key, paths in found.items() if paths}


@pytest.mark.parametrize("module", ["binding_metrics.cli.run", "binding_metrics.cli.batch"])
def test_the_mode_help_of_both_commands_is_the_shared_text(module, monkeypatch):
    parser = _capture_parser(module, monkeypatch)
    action = next(a for a in parser._actions if "--openfold-mode" in a.option_strings)
    assert action.help == OPENFOLD_MODE_HELP


def test_the_template_and_server_options_the_docs_name_exist_with_the_documented_values(
    monkeypatch,
):
    section = _paragraph("**template mode (`template_mode`")
    assert f"`--openfold-templates {{{','.join(OPENFOLD_TEMPLATE_CHOICES)}}}`" in section
    for module in ("binding_metrics.cli.run", "binding_metrics.cli.batch"):
        options = _capture_parser(module, monkeypatch)._option_string_actions
        templates = options["--openfold-templates"]
        assert tuple(templates.choices) == OPENFOLD_TEMPLATE_CHOICES
        assert templates.default == OPENFOLD_TEMPLATE_CHOICES[0], "the paragraph says structure"
        assert "--openfold-no-msa-server" in options
        assert "--prediction-templates" not in options, (
            f"{module}: the paragraph says there is none"
        )
    subcommands = _openfold_subcommands(monkeypatch)
    for name in ("prepare-query", "prepare-scoring-query", "refold", "score"):
        assert "--openfold-templates" in subcommands[name]._option_string_actions, name
    for name in ("run", "refold", "score"):
        assert "--no-msa-server" in subcommands[name]._option_string_actions, name


def test_the_arguments_the_docs_name_are_parameters_of_the_functions():
    from binding_metrics.cli.batch import run_batch
    from binding_metrics.metrics import openfold

    section = _paragraph("**template mode (`template_mode`")
    for name in (
        "prepare_refolding_query",
        "prepare_scoring_query",
        "run_openfold_scoring",
        "run_openfold_refolding",
        "run_openfold_batched",
    ):
        assert f"`{name}`" in section, name
        parameters = inspect.signature(getattr(openfold, name)).parameters
        assert "template_mode" in parameters, name
        assert parameters["template_mode"].default == OPENFOLD_TEMPLATE_CHOICES[0], name
        if name.startswith("run_openfold"):
            assert "use_msa_server" in parameters, name
    for function in (run_pipeline, run_batch):
        parameters = inspect.signature(function).parameters
        assert {"openfold_templates", "openfold_use_msa_server"} <= set(parameters), function


def _displaced_binder_run():
    """A stub run function that writes the input with the binder moved 2.0 A along z."""
    from tests.predictors import synth_of3
    from tests.test_feat_c_support import complex_from

    def write(*, samples=None, output_dir=None, **kw):
        names_and_inputs = (
            [(s.query_name, s.complex_structure_path) for s in samples]
            if samples is not None
            else [(kw["query_name"], kw["complex_structure_path"])]
        )
        predictions = Path(output_dir if output_dir is not None else kw["output_dir"])
        predictions = predictions / "predictions"
        for name, input_path in names_and_inputs:
            synthetic = complex_from(input_path)
            synthetic.atoms.coord[synthetic.atoms.chain_id == PEPTIDE_CHAIN, 2] += 2.0
            synth_of3.write_prediction(predictions, name, synthetic)
        return predictions

    return write


class TestTheClaimsAboutTheResultKeys:
    """The binder RMSD against the input is measured in both modes, in the receptor frame."""

    @staticmethod
    def _run(tmp_path, **kwargs):
        return run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain=PEPTIDE_CHAIN,
            receptor_chain=RECEPTOR_CHAIN,
            openfold_conda_env=None,
            **kwargs,
        )

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_predictor_step_measures_both_keys_in_both_modes(self, tmp_path, monkeypatch, mode):
        StubOpenFold(monkeypatch)
        block = self._run(tmp_path, predictor="of3", openfold_mode=mode)["prediction"]
        assert math.isfinite(block["delta_com_angstrom"])
        assert block["binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-2)
        paragraph = _paragraph("**the two modes.**")
        for name, named in (
            ("the mode help", set(RESULT_KEY.findall(OPENFOLD_MODE_HELP))),
            ("the modes paragraph", set(BACKTICKED_KEY.findall(paragraph))),
        ):
            assert {"binder_ca_rmsd", "delta_com_angstrom"} <= named, name
            assert named <= set(block), f"{name} names {sorted(named - set(block))}, not keys"

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_a_displaced_binder_is_seen_in_both_modes(self, tmp_path, monkeypatch, mode):
        from binding_metrics.metrics import openfold

        stub = StubOpenFold(monkeypatch)
        monkeypatch.setattr(openfold, "run_openfold_scoring", _displaced_binder_run())
        monkeypatch.setattr(openfold, "run_openfold_refolding", _displaced_binder_run())
        block = self._run(tmp_path, predictor="of3", openfold_mode=mode)["prediction"]
        assert stub.starts == 0  # the patched functions replaced the stub's
        # the receptor does not move, so the receptor-frame RMSD is the displacement
        assert block["binder_ca_rmsd"] == pytest.approx(2.0, abs=1e-2)
        assert block["delta_com_angstrom"] == pytest.approx(2.0, abs=1e-2)

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_legacy_step_agrees(self, tmp_path, monkeypatch, mode):
        from binding_metrics.metrics import openfold

        monkeypatch.setattr(openfold, "run_openfold_scoring", _displaced_binder_run())
        monkeypatch.setattr(openfold, "run_openfold_refolding", _displaced_binder_run())
        block = self._run(tmp_path, openfold_mode=mode)["openfold"]
        assert block["binder_ca_rmsd"] == pytest.approx(2.0, abs=1e-2)
        assert block["delta_com_angstrom"] == pytest.approx(2.0, abs=1e-2)

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_batched_openfold_step_agrees(self, tmp_path, monkeypatch, mode):
        from binding_metrics.cli import batch
        from binding_metrics.metrics import openfold

        monkeypatch.setattr(openfold, "run_openfold_batched", _displaced_binder_run())
        monkeypatch.setattr(
            batch,
            "_detect_sample_chains",
            lambda *a, **k: [(0, EXAMPLE_1YCR.stem, EXAMPLE_1YCR, PEPTIDE_CHAIN, RECEPTOR_CHAIN)],
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode=mode,
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
        )
        assert rows[0]["openfold_binder_ca_rmsd"] == pytest.approx(2.0, abs=1e-2)

    def test_the_reference_is_the_input_for_every_model(self):
        from binding_metrics.cli.prediction import reference_for
        from binding_metrics.predictors import PARSERS

        for model in sorted(PARSERS):
            assert reference_for(model, EXAMPLE_1YCR) == EXAMPLE_1YCR
