"""What the text says about the OpenFold3 modes matches what the code does.

A template carries the fold of one chain and no cross-chain geometry: each query chain gets its
own template structures (the template CIFs here are single-chain files), and OpenFold3's
``_embed_feats`` applies the same-chain mask to the validity indicators of the template pair
features. So ``score`` does not hand OpenFold3 the pose of the input. The text used to say that
both chains were given as templates "so that OF3 evaluates the known conformation", and that an
adversary run in score mode agrees with the design "partly by construction". These tests keep
that wording out and check what the text now says about the result keys: the binder RMSD against
the input, in the receptor frame, is measured in both modes, next to ``delta_com_angstrom``.
"""

import argparse
import importlib
import math
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import OPENFOLD_MODE_HELP
from binding_metrics.cli.run import run_pipeline
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
)

ROOT = Path(__file__).parent.parent

#: Phrases that stated the old reading of the modes.
STALE = (
    "both chains as templates",
    "known conformation",
    "partly by construction",
    "templated on the design pose",
    "templated on the input pose",
    "receptor fixed as template",
    "fixed as template",
)


def _texts():
    files = [ROOT / "README.md", ROOT / "docs" / "metrics.md"]
    files += sorted((ROOT / "src" / "binding_metrics").rglob("*.py"))
    return [(path, path.read_text(encoding="utf-8")) for path in files]


@pytest.mark.parametrize("phrase", STALE)
def test_the_old_wording_is_gone_from_the_source_and_the_docs(phrase):
    found = [str(path.relative_to(ROOT)) for path, text in _texts() if phrase in text]
    assert not found, f"{phrase!r} is still in {found}"


@pytest.mark.parametrize("module", ["binding_metrics.cli.run", "binding_metrics.cli.batch"])
def test_the_mode_help_of_both_commands_is_the_shared_text(module, monkeypatch):
    holder = {}

    def stop(self, *args, **kwargs):
        holder["parser"] = self
        raise SystemExit

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        importlib.import_module(module).main()
    action = next(a for a in holder["parser"]._actions if "--openfold-mode" in a.option_strings)
    assert action.help == OPENFOLD_MODE_HELP
    for stated in ("its own structure", "places the binder itself", "delta_com_angstrom"):
        assert stated in OPENFOLD_MODE_HELP
    assert "only the receptor is templated" in OPENFOLD_MODE_HELP


def test_the_docs_name_the_keys_that_measure_the_pose():
    text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    section = text[text.index("**the two modes.**") :]
    section = section[: section.index("\n\n")]
    assert "`delta_com_angstrom`" in section and "`binder_ca_rmsd`" in section
    assert "given in both modes by `binder_ca_rmsd`" in section
    assert "refold` mode only" not in section and "NaN in `score`" not in section


def test_the_docs_state_exactly_what_the_template_embedder_does():
    """Only the validity indicators are masked; the distogram and unit-vector tensors are not."""
    text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    section = text[text.index("**the two modes.**") :]
    section = section[: section.index("\n\n")]
    assert "validity indicators of the template pair features" in section
    assert "each query chain gets its own template structures" in section
    assert "not multiplied by that mask" in section
    for overstated in (
        "every template feature",
        "all template features",
        "restricts the pair masks",
    ):
        assert overstated not in section


def test_the_docs_give_the_measured_msa_server_effect_and_the_ways_to_keep_the_template():
    text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    start = text.index("**the MSA server and the templates (issue #68")
    section = text[start : text.index("\n\n", start)]
    assert "#68" in section and "measured on one complex" in section
    # what the run showed: the default has no template at all, with the numbers of one complex
    assert "the default `score` run has no template at all" in section
    assert "135 template hits" in section and "70 for the 13-residue binder" in section
    assert "1.62 A" in section and "1.12 A" in section
    # the two ways to keep it
    assert "`use_msa_server=False`" in section and "`--no-msa-server`" in section
    assert "`--openfold-no-msa-server`" in section
    assert (
        '`template_mode="structure"`' in section and "`--openfold-templates structure`" in section
    )
    # not tried is said
    assert "Not tried" in section and "fetch_missing_structures: true" in section


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
