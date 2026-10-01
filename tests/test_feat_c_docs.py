"""The prediction options and the ``results["prediction"]`` keys named in docs/metrics.md exist.

The documentation table of the options and of the keys is read from the file, so a rename in
the code or a stale row in the docs fails here.
"""

import argparse
import fnmatch
import importlib
import re
import sys
from pathlib import Path

import pytest

from binding_metrics.cli.run import run_pipeline
from tests.test_feat_c_support import EXAMPLE_1YCR, StubOpenFold

DOCS = Path(__file__).parent.parent / "docs" / "metrics.md"
HEADING = '### Pipeline: `--predictor`, the prediction store and `results["prediction"]`'


def _section() -> str:
    text = DOCS.read_text(encoding="utf-8")
    start = text.index(HEADING)
    return text[start : text.index("\n---\n", start)]


def _first_column(table_after: str) -> list[list[str]]:
    """The backticked names in the first column of the table that follows ``table_after``."""
    section = _section()
    lines = section[section.index(table_after) :].splitlines()
    names = []
    started = False
    for line in lines[1:]:
        if line.startswith("|"):
            started = True
            cell = line.split("|")[1]
            if set(cell.strip()) <= {"-", " "}:
                continue
            names.append(re.findall(r"`([^`]+)`", cell))
        elif started:
            break
    return names[1:] if names and not names[0] else names  # the header has no backticks


def _capture_parser(module_name, monkeypatch):
    holder = {}

    def stop(self, *args, **kwargs):
        holder["parser"] = self
        raise SystemExit

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        importlib.import_module(module_name).main()
    return holder["parser"]


@pytest.mark.parametrize("module", ["binding_metrics.cli.run", "binding_metrics.cli.batch"])
def test_every_documented_option_is_an_option_of_both_commands(module, monkeypatch):
    parser = _capture_parser(module, monkeypatch)
    documented = [
        re.match(r"(--[a-z0-9-]+)", name).group(1)
        for cell in _first_column("| option | default | effect |")
        for name in cell
        if name.startswith("--")
    ]
    assert len(documented) >= 9
    missing = [option for option in documented if option not in parser._option_string_actions]
    assert not missing, f"documented but not an option of {module}: {missing}"


CONDITIONAL_KEYS = {"evobind_error", "adversarial_error", "reason"}
# read from the files that OpenFold3 writes next to its output, which the stand-in does not write
FILE_DEPENDENT_KEYS = {"templates"}


def _documented_keys() -> list[str]:
    return [
        name
        for cell in _first_column("| key | description |")
        for name in cell
        if re.fullmatch(r"[a-z_*]+", name)
    ]


def _prediction_block(tmp_path, **kwargs):
    return run_pipeline(
        EXAMPLE_1YCR,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        peptide_chain="B",
        receptor_chain="A",
        openfold_conda_env=None,
        predictor="of3",
        **kwargs,
    )["prediction"]


def test_every_documented_key_is_in_a_real_block(tmp_path, monkeypatch):
    StubOpenFold(monkeypatch)
    block = _prediction_block(tmp_path)
    documented = _documented_keys()
    assert len(documented) >= 20
    unmatched = [
        name
        for name in documented
        if name not in CONDITIONAL_KEYS | FILE_DEPENDENT_KEYS
        and not fnmatch.filter(list(block), name)
    ]
    assert not unmatched, f"documented but not in results['prediction']: {unmatched}"
    assert {
        "requests",
        "memo_hits",
        "hits",
        "adopted",
        "misses",
        "runs",
        "failed",
        "parsed",
    } <= set(block["cache"])
    assert "request_key" in block["cache"]


def test_the_keys_that_appear_only_on_failure_are_documented_and_real(tmp_path, monkeypatch):
    from tests.predictors import synth, synth_of3
    from tests.test_feat_c_support import complex_from

    StubOpenFold(monkeypatch)
    truth = complex_from(EXAMPLE_1YCR)
    renamed = synth.SyntheticComplex(
        atoms=synth.renamed_atoms(truth, {"A": "R", "B": "P"}),
        plddt_per_atom=truth.plddt_per_atom,
        pae=truth.pae,
        pde=truth.pde,
        scalars=truth.scalars,
    )
    out = tmp_path / "out"
    synth_of3.write_prediction(out, EXAMPLE_1YCR.stem, renamed)
    block = _prediction_block(
        tmp_path / "run", prediction_dir=out
    )  # the chains are not renamed back
    assert CONDITIONAL_KEYS <= set(block)
    assert CONDITIONAL_KEYS <= set(_documented_keys())


def test_the_templates_key_is_documented_and_real_for_an_output_that_has_the_files(
    tmp_path, monkeypatch
):
    import json

    from tests.predictors import synth_of3
    from tests.test_feat_c_support import complex_from

    StubOpenFold(monkeypatch)
    out = tmp_path / "out"
    synth_of3.write_prediction(out, EXAMPLE_1YCR.stem, complex_from(EXAMPLE_1YCR))
    chains = [
        {"chain_ids": ["A"], "template_entry_chain_ids": ["receptor_A"]},
        {"chain_ids": ["B"], "template_entry_chain_ids": []},
    ]
    query_set = {"queries": {EXAMPLE_1YCR.stem: {"chains": chains}}}
    (out / "inference_query_set.json").write_text(json.dumps(query_set), encoding="utf-8")
    block = _prediction_block(tmp_path / "run", prediction_dir=out)
    assert FILE_DEPENDENT_KEYS <= set(block) and FILE_DEPENDENT_KEYS <= set(_documented_keys())
    assert block["templates"]["A"]["used"] is True and block["templates"]["B"]["used"] is False
