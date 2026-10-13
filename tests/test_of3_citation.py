"""The OpenFold3 reference and install command agree across README, docs and code (#87, #89).

Offline checks of what the places agree on, not of what the DOIs resolve to (no network in
tests): the README, ``docs/metrics.md`` and the docstring of ``metrics.openfold`` cite the same
Zenodo records, and the install command is the same in the README, ``environment_openfold3.yml``
and the message of ``binding-metrics-check-env``. A release that changes a DOI or the pin needs
the documents edited, not this file.
"""

import re
from pathlib import Path

from binding_metrics.metrics import openfold

ROOT = Path(__file__).parent.parent
ZENODO_DOI = re.compile(r"10\.5281/zenodo\.\d+")
SETUP_COMMAND = "setup_openfold --non-interactive"

# Wording that was wrong for a released openfold3 and must not come back. Add a row when a
# document is found to carry a claim that a real run disproved; do not add a row for a rewording.
BANNED = {
    "10.5281/zenodo.19001000": "the record of openfold3 0.4.0; the toolkit cites the 0.5.0 record",
    "OpenFold3-preview": "the name of the 0.4 series; the reference is OpenFold3 v0.5.0",
    "OpenFold3: An open-source, trainable implementation": (
        "the title of the OpenFold (v1) paper, which is not the OpenFold3 reference"
    ),
    "That environment runs Python 3.10": "openfold3 0.5 supports Python 3.10 to 3.13 (#89)",
}


def _documents() -> dict[str, str]:
    return {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/metrics.md": (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8"),
        "metrics/openfold.py docstring": openfold.__doc__,
    }


def test_the_readme_the_docs_and_the_module_cite_the_same_records():
    documents = _documents()
    dois = {name: set(ZENODO_DOI.findall(text)) for name, text in documents.items()}
    assert dois["README.md"], "the README cites no Zenodo record"
    assert dois["README.md"] == dois["docs/metrics.md"], dois
    assert dois["metrics/openfold.py docstring"], "the module docstring cites no Zenodo record"
    assert dois["metrics/openfold.py docstring"] <= dois["README.md"], dois


def test_the_wording_a_real_run_disproved_is_gone():
    found = {
        phrase: [name for name, text in _documents().items() if phrase in text] for phrase in BANNED
    }
    still_there = {
        f"{phrase!r} ({BANNED[phrase]})": names for phrase, names in found.items() if names
    }
    assert not still_there


def test_the_install_command_is_the_same_in_the_readme_the_environment_file_and_check_env():
    readme = _documents()["README.md"]
    command = re.search(r'pip install "(openfold3[^"]*)"', readme)
    assert command is not None, "the README has no pip install line for openfold3"
    requirement = command.group(1)
    environment = (ROOT / "environment_openfold3.yml").read_text(encoding="utf-8")
    assert f'"{requirement}"' in environment
    check_env = (ROOT / "src" / "binding_metrics" / "cli" / "check_env.py").read_text(
        encoding="utf-8"
    )
    assert requirement in check_env
    assert SETUP_COMMAND in readme and SETUP_COMMAND in check_env
