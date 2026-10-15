"""The OpenFold3 conda environment file pins the release the toolkit is written against (#82).

Only the text of the file is checked: nothing is solved or installed here, so whether the
torch and CUDA wheels that pip picks for this pin work with ``cuda-version=12.4`` is not tested.
"""

import re
from pathlib import Path

import pytest

ENV_FILE = Path(__file__).parent.parent / "environment_openfold3.yml"


@pytest.fixture(scope="module")
def pip_requirements():
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(ENV_FILE.read_text(encoding="utf-8"))
    pip_block = next(dep for dep in spec["dependencies"] if isinstance(dep, dict))["pip"]
    return pip_block


def test_openfold3_is_pinned_to_the_0_5_series(pip_requirements):
    specifiers = pytest.importorskip("packaging.specifiers")
    requirement = next(r for r in pip_requirements if r.startswith("openfold3"))
    spec = specifiers.SpecifierSet(re.sub(r"^openfold3", "", requirement))
    assert spec.contains("0.5.0") and spec.contains("0.5.7")
    assert not spec.contains("0.4.5")
    assert not spec.contains("0.6.0")


def test_the_comment_names_the_checkpoint_and_the_download_command():
    text = ENV_FILE.read_text(encoding="utf-8")
    assert "OpenBind-0" in text
    assert "of3-ob-2025-06-30-174k.pt" in text
    assert "setup_openfold --non-interactive" in text
    assert "OpenFold3-preview" not in text


def test_the_python_requirement_is_what_openfold3_states():
    text = ENV_FILE.read_text(encoding="utf-8")
    assert re.search(r"python=3\.10\b.*>=3\.10.*3\.13", text)
