"""The licence is an SPDX expression (PEP 639) and the build floor supports it (issue #176).

`license = {text = ...}` is deprecated in setuptools 77.0.0 and stops being supported on
2027-02-18. The string form and `license-files` need setuptools 77.0.0 or newer, and PEP 639
forbids a `License ::` classifier next to a licence expression.
"""

import re
import tomllib
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
FIRST_SETUPTOOLS_WITH_LICENSE_EXPRESSION = (77, 0, 0)

pytestmark = pytest.mark.skipif(not PYPROJECT.is_file(), reason="needs a source checkout")


def _pyproject() -> dict:
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _setuptools_floor() -> tuple[int, ...]:
    (requirement,) = [
        r for r in _pyproject()["build-system"]["requires"] if re.match(r"setuptools\b", r)
    ]
    match = re.fullmatch(r"setuptools\s*>=\s*(\d+(?:\.\d+)*)\s*", requirement)
    assert match, f"the setuptools requirement is not a plain '>=' floor: {requirement!r}"
    return _version_tuple(match.group(1))


def test_the_licence_is_an_spdx_expression_string():
    assert _pyproject()["project"]["license"] == "Apache-2.0"


def test_no_license_classifier_accompanies_the_expression():
    classifiers = _pyproject()["project"]["classifiers"]
    assert [c for c in classifiers if c.startswith("License ::")] == []


def test_the_licence_file_is_listed_and_exists():
    listed = _pyproject()["project"]["license-files"]
    assert listed == ["LICENSE"]
    assert (REPO_ROOT / "LICENSE").is_file()


def test_the_setuptools_floor_supports_the_licence_expression():
    assert _setuptools_floor() >= FIRST_SETUPTOOLS_WITH_LICENSE_EXPRESSION


def test_setuptools_reads_the_declaration_without_a_licence_warning():
    setuptools = pytest.importorskip("setuptools")
    if _version_tuple(setuptools.__version__.split("+")[0]) < _setuptools_floor():
        pytest.skip("the installed setuptools is older than the floor of pyproject.toml")
    from setuptools.config.pyprojecttoml import apply_configuration
    from setuptools.dist import Distribution

    distribution = Distribution({"src_root": str(REPO_ROOT)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_configuration(distribution, PYPROJECT)
    licence_warnings = [str(w.message) for w in caught if "license" in str(w.message).lower()]
    assert licence_warnings == []
    assert distribution.metadata.license_expression == "Apache-2.0"
