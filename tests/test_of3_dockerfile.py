"""Static checks of the Docker files and README instructions for OpenFold3 (#72, #81).

The image is not built here (no Docker, no GPU), so these tests read the text: they show that
the instructions agree with each other and no longer describe the DeepSpeed path that
openfold3 0.5 does not use by default. They do not show that the image builds or that Triton
finds its cache at the mounted path.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("stale", ["CUTLASS", "evoformer", "torch_extensions", "of3-jit-cache"])
def test_the_deepspeed_jit_path_is_gone_from_the_dockerfile_and_readme(stale):
    assert stale not in DOCKERFILE
    assert stale not in README


def test_the_dockerfile_still_sets_the_triton_cache_the_mount_points_at():
    match = re.search(r"^ENV TRITON_CACHE_DIR=(\S+)", DOCKERFILE, re.MULTILINE)
    assert match is not None
    cache = match.group(1)
    assert f":{cache} " in README
    assert f":{cache} " in DOCKERFILE


def test_the_readme_and_dockerfile_name_the_openbind_checkpoint_as_the_default():
    for text in (README, DOCKERFILE):
        assert "openbind-2025-06-30-174k" in text
    assert "openfold3-p2-155k" not in README
    assert "openfold3-p2-155k" not in DOCKERFILE


def test_the_entrypoint_is_copied_into_the_image_from_the_path_that_exists():
    match = re.search(r"^COPY (\S+) /usr/local/bin/binding-metrics-entrypoint", DOCKERFILE, re.M)
    assert match is not None
    assert (ROOT / match.group(1)).is_file()


def test_the_dockerfile_installs_the_pinned_environment_file():
    assert "mamba env create -f environment_openfold3.yml" in DOCKERFILE
    assert (ROOT / "environment_openfold3.yml").is_file()
