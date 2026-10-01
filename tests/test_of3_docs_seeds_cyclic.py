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

from binding_metrics.metrics.openfold import run_openfold

ROOT = Path(__file__).parent.parent
METRICS = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")

STALE = (
    "query JSON carries the seed",
    "seed values written to the query json",
    "written to the openfold3 query json",
    "sets the seed values of the query json",
    "--num-seeds 1 \\",  # the example that suggested a single seed is the default
)


def _texts():
    files = [ROOT / "README.md", ROOT / "docs" / "metrics.md", ROOT / "docs" / "preflight.md"]
    files += sorted((ROOT / "src" / "binding_metrics").rglob("*.py"))
    return [(path, path.read_text(encoding="utf-8").lower()) for path in files]


@pytest.mark.parametrize("phrase", STALE)
def test_no_text_says_that_the_seeds_go_to_the_query_json(phrase):
    found = [str(path.relative_to(ROOT)) for path, text in _texts() if phrase.lower() in text]
    assert not found, f"{phrase!r} is still in {found}"


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


def test_the_pre_flight_page_no_longer_says_the_builder_writes_no_cyclic_field():
    text = (ROOT / "docs" / "preflight.md").read_text(encoding="utf-8")
    assert "do not write `cyclic: true`" not in text
    row = next(line for line in text.splitlines() if "head-to-tail binder is sent" in line)
    assert "binder_cyclic" in row and "does not enforce the closure bond" in row


def test_the_docs_name_the_cyclic_options_and_keys_that_exist():
    from binding_metrics.metrics import openfold

    for name in ("decide_binder_cyclic", "BinderCyclicDecision"):
        assert hasattr(openfold, name) and f"`{name}" in METRICS
    assert "`--openfold-cyclic {auto,on,off}`" in METRICS
    assert re.search(r"`binder_cyclic`\s*\(bool", METRICS)
