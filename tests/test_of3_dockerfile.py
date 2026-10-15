"""Static checks of the Docker files and README instructions for OpenFold3 (#72, #81).

The image is not built here (no Docker, no GPU), so these tests read the text: they show that
the instructions agree with each other and with the code, and no longer describe the DeepSpeed
path that openfold3 0.5 does not use by default. They do not show that the image builds or that
Triton finds its cache at the mounted path.
"""

import re
from pathlib import Path

from binding_metrics.metrics._openfold_run import _OPENFOLD_DEFAULT_CHECKPOINT_NAME

ROOT = Path(__file__).parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

# Words that described a path or a default that a real run disproved. Add a row when a document
# is found to carry such a claim; do not add a row for a rewording.
BANNED = {
    "CUTLASS": "the CUTLASS layer of the DeepSpeed path; DeepSpeed is an opt-in extra (#72)",
    "evoformer": "the DeepSpeed evoformer_attn path, which openfold3 0.5 does not use by default",
    "torch_extensions": "the JIT cache mount of the DeepSpeed path (#72, #81)",
    "of3-jit-cache": "the JIT cache mount of the DeepSpeed path (#72, #81)",
    "openfold3-p2-155k": "the Preview2 default; openfold3 0.5 defaults to OpenBind-0 (#72)",
}


def test_the_wording_a_real_run_disproved_is_gone_from_the_dockerfile_and_readme():
    found = {
        f"{word!r} ({reason})": [
            name
            for name, text in (("Dockerfile", DOCKERFILE), ("README.md", README))
            if word in text
        ]
        for word, reason in BANNED.items()
    }
    assert not {key: names for key, names in found.items() if names}


def test_the_dockerfile_still_sets_the_triton_cache_the_mount_points_at():
    match = re.search(r"^ENV TRITON_CACHE_DIR=(\S+)", DOCKERFILE, re.MULTILINE)
    assert match is not None
    cache = match.group(1)
    assert f":{cache} " in README
    assert f":{cache} " in DOCKERFILE


def test_the_readme_and_dockerfile_name_the_default_checkpoint_of_the_code():
    for name, text in (("README.md", README), ("Dockerfile", DOCKERFILE)):
        assert _OPENFOLD_DEFAULT_CHECKPOINT_NAME in text, name


def test_the_files_the_dockerfile_installs_and_copies_exist():
    environments = re.findall(r"env create -f (\S+)", DOCKERFILE)
    assert "environment_openfold3.yml" in environments
    copied = re.findall(r"^COPY (\S+) \S+binding-metrics-entrypoint", DOCKERFILE, re.MULTILINE)
    assert copied, "no COPY line for the entrypoint"
    for relative in [*environments, *copied]:
        assert (ROOT / relative).is_file(), relative
