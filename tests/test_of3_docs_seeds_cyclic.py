"""The text about the OpenFold3 seeds and the ``cyclic`` flag matches the code (#67, #77).

OpenFold3 0.5.0 does not read seeds from the query JSON, so no text may say that the seeds are
written there; ``--num-seeds 1`` is passed as ``--num_model_seeds=1``, which samples with the seed
that OpenFold3 generates from 42 and not with 42, so it is not the example of a default run
(the docs may say so, but no command line example may use it).
"""

import inspect
import re
from pathlib import Path

import pytest

from binding_metrics.cli import OPENFOLD_CYCLIC_CHOICES
from binding_metrics.metrics.openfold import run_openfold
from tests.test_feat_c_docs import _capture_parser

ROOT = Path(__file__).parent.parent
METRICS = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
PREFLIGHT = (ROOT / "docs" / "preflight.md").read_text(encoding="utf-8")

# Wording that a real run disproved. Add a row when a text is found to carry such a claim; do not
# add a row for a rewording. The match ignores case and covers README.md, the docs and src/.
BANNED = {
    "query JSON carries the seed": "OpenFold3 0.5.0 does not read seeds from the query JSON (#67)",
    "seed values written to the query json": "same: the seeds are not read from the query JSON",
    "written to the openfold3 query json": "same: the seeds are not read from the query JSON",
    "sets the seed values of the query json": "same: the seeds are not read from the query JSON",
    "--num-seeds 1 \\": (
        "an example that suggested one seed is the default: it is passed as --num_model_seeds=1,"
        " which samples with the seed OpenFold3 generates from 42, not with 42 (#67)"
    ),
    "do not write `cyclic: true`": (
        "the pre-flight page said that the query builder writes no cyclic field; it writes it"
        " for a head-to-tail binder of standard residues by default (#77)"
    ),
}


def _texts():
    files = [ROOT / "README.md", ROOT / "docs" / "metrics.md", ROOT / "docs" / "preflight.md"]
    files += sorted((ROOT / "src" / "binding_metrics").rglob("*.py"))
    return [(path, path.read_text(encoding="utf-8").lower()) for path in files]


def test_the_wording_a_real_run_disproved_is_gone_from_the_docs_and_the_source():
    texts = _texts()
    found = {
        f"{phrase!r} ({reason})": [
            str(path.relative_to(ROOT)) for path, text in texts if phrase.lower() in text
        ]
        for phrase, reason in BANNED.items()
    }
    assert not {key: paths for key, paths in found.items() if paths}


def test_the_documented_signature_of_run_openfold_is_the_real_one():
    heading = next(line for line in METRICS.splitlines() if line.startswith("### `run_openfold("))
    documented = heading[heading.index("(") + 1 : heading.rindex(")")]
    parts = []
    for name, parameter in inspect.signature(run_openfold).parameters.items():
        parts.append(
            name
            if parameter.default is inspect.Parameter.empty
            else f"{name}={parameter.default!r}"
        )
    assert documented == ", ".join(parts)


@pytest.mark.parametrize("argument", ["num_model_seeds", "seeds", "runner_yaml", "template_dir"])
def test_the_argument_table_of_run_openfold_has_the_row(argument):
    start = METRICS.index("| argument | default | description |")
    table = METRICS[start : METRICS.index("\n\n", start)]
    row = next(line for line in table.splitlines() if line.startswith(f"| `{argument}` |"))
    default = inspect.signature(run_openfold).parameters[argument].default
    assert row.split("|")[2].strip() == (f"`{default}`" if default is not None else "None")


def test_the_pre_flight_page_names_the_cyclic_option_and_its_default_of_the_code(monkeypatch):
    from binding_metrics.metrics import openfold

    defaults = {
        name: inspect.signature(function).parameters["binder_cyclic"].default
        for name, function in vars(openfold).items()
        if (name.startswith("prepare_") or name.startswith("run_openfold"))
        and callable(function)
        and "binder_cyclic" in inspect.signature(function).parameters
    }
    assert {"prepare_scoring_query", "run_openfold_scoring"} <= set(defaults)
    assert set(defaults.values()) == {OPENFOLD_CYCLIC_CHOICES[0]}
    assert f'binder_cyclic="{OPENFOLD_CYCLIC_CHOICES[0]}"' in PREFLIGHT
    parser = _capture_parser("binding_metrics.cli.run", monkeypatch)
    assert "--openfold-cyclic" in parser._option_string_actions
    assert "--openfold-cyclic" in PREFLIGHT


def test_the_docs_name_the_cyclic_options_and_keys_that_exist():
    from binding_metrics.metrics import openfold

    for name in ("decide_binder_cyclic", "BinderCyclicDecision"):
        assert hasattr(openfold, name) and f"`{name}" in METRICS
    assert f"`--openfold-cyclic {{{','.join(OPENFOLD_CYCLIC_CHOICES)}}}`" in METRICS
    assert re.search(r"`binder_cyclic`\s*\(bool", METRICS)
