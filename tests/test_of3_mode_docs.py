"""What the text says about the OpenFold3 modes matches what the code does.

A template carries the fold of one chain and no inter-chain geometry (the template CIFs here are
single-chain files, and OpenFold3's template embedder keeps only same-chain pairs of the template
features), so ``score`` does not hand OpenFold3 the pose of the input. The text used to say that
both chains were given as templates "so that OF3 evaluates the known conformation", and that an
adversary run in score mode agrees with the design "partly by construction". These tests keep
that wording out and check the two claims that the text now makes about the result keys.
"""

import argparse
import importlib
import math
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import OPENFOLD_MODE_HELP
from binding_metrics.cli.run import run_pipeline
from tests.test_feat_c_support import EXAMPLE_1YCR, StubOpenFold

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


def test_the_docs_name_the_key_that_measures_the_pose_and_the_mode_of_the_rmsd():
    text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    section = text[text.index("**the two modes.**") :]
    section = section[: section.index("\n\n")]
    assert "`delta_com_angstrom`" in section
    assert "`binder_ca_rmsd` is computed in `refold` mode only" in section


def test_the_docs_describe_the_msa_server_limitation_as_open_and_give_the_workaround():
    text = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
    start = text.index("**known limitation: the MSA server and the templates")
    section = text[start : text.index("\n\n", start)]
    assert "#68" in section and "open" in section
    assert "`use_msa_server=False`" in section and "`--no-msa-server`" in section


class TestTheClaimsAboutTheResultKeys:
    """In score mode the pose is reported by ``delta_com_angstrom``, not by ``binder_ca_rmsd``."""

    def test_score_mode_has_the_com_displacement_and_no_binder_rmsd(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        block = run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain="B",
            receptor_chain="A",
            openfold_conda_env=None,
            predictor="of3",
        )["prediction"]
        assert math.isfinite(block["delta_com_angstrom"])
        assert math.isnan(block["binder_ca_rmsd"])

    def test_refold_mode_measures_the_binder_rmsd(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        block = run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain="B",
            receptor_chain="A",
            openfold_conda_env=None,
            predictor="of3",
            openfold_mode="refold",
        )["prediction"]
        assert math.isfinite(block["binder_ca_rmsd"])

    def test_the_legacy_step_agrees(self, tmp_path, monkeypatch):
        from binding_metrics.metrics import openfold
        from tests.test_feat_c_support import write_of3_output

        def write(**kwargs):
            predictions = Path(kwargs["output_dir"]) / "predictions"
            write_of3_output(predictions, kwargs["query_name"], kwargs["complex_structure_path"])
            return predictions

        monkeypatch.setattr(openfold, "run_openfold_scoring", write)
        block = run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain="B",
            receptor_chain="A",
            openfold_conda_env=None,
        )["openfold"]
        assert math.isfinite(block["delta_com_angstrom"]) and math.isnan(block["binder_ca_rmsd"])
