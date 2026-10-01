"""The OpenFold3 citation and install wording of the README and docs (#87, #89).

Text checks only: they show which DOI and Python statement the documents carry, not that the
DOIs resolve (no network in tests). The DOIs were read from the Zenodo records of
openfold3 v0.5.0 (version) and the project (concept) on 2026-09-30.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
VERSION_DOI = "10.5281/zenodo.22042719"  # openfold3 v0.5.0
CONCEPT_DOI = "10.5281/zenodo.17485509"  # all versions
OLD_DOI = "10.5281/zenodo.19001000"  # the 0.4.0 record

DOCUMENTS = ["README.md", "docs/metrics.md"]


@pytest.mark.parametrize("name", DOCUMENTS)
def test_the_reference_is_the_0_5_0_record_with_its_concept_doi(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert VERSION_DOI in text
    assert CONCEPT_DOI in text
    assert OLD_DOI not in text
    assert "OpenFold3-preview" not in text


@pytest.mark.parametrize("name", DOCUMENTS)
def test_alphafold3_stays_cited(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "Accurate structure prediction of biomolecular interactions with AlphaFold 3" in text


def test_the_readme_no_longer_says_openfold3_runs_python_3_10_only():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "That environment runs Python 3.10" not in text
    assert re.search(r"Python 3\.10 to 3\.13", text)
    assert 'pip install "openfold3>=0.5.0,<0.6"' in text
    assert "setup_openfold --non-interactive" in text
