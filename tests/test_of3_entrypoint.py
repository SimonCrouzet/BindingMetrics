"""The Docker entrypoint fetches the OpenBind-0 checkpoint (issues #70, #71, #72).

The script runs for real under bash, with a stub ``conda`` first on PATH that records its
command line and stdin and can create the checkpoint file. Nothing is downloaded and no
Docker image is built, so what these tests cannot show is that ``setup_openfold
--non-interactive`` behaves as documented in openfold3 0.5.0 inside the image.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).parent.parent / "docker" / "entrypoint.sh"
DEFAULT_FILE = "of3-ob-2025-06-30-174k.pt"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

_FAKE_CONDA = """#!/bin/bash
echo "$@" >> "$FAKE_CONDA_LOG"
echo "stdin bytes: $(cat | wc -c)" >> "$FAKE_CONDA_LOG"
if [ "${FAKE_CONDA_CREATE:-0}" = "1" ]; then
    mkdir -p "$HOME/.openfold3"
    echo -n fake > "$HOME/.openfold3/of3-ob-2025-06-30-174k.pt"
fi
exit "${FAKE_CONDA_EXIT:-0}"
"""


@pytest.fixture
def sandbox(tmp_path):
    """A HOME, a stub ``conda`` and a log of its calls."""
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda = bin_dir / "conda"
    conda.write_text(_FAKE_CONDA, encoding="utf-8")
    conda.chmod(0o755)
    return {"home": home, "bin": bin_dir, "log": tmp_path / "conda.log"}


def _run(sandbox, *, create=False, exit_code=0, skip_check=False):
    env = {
        "PATH": f"{sandbox['bin']}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(sandbox["home"]),
        "FAKE_CONDA_LOG": str(sandbox["log"]),
        "FAKE_CONDA_CREATE": "1" if create else "0",
        "FAKE_CONDA_EXIT": str(exit_code),
    }
    if skip_check:
        env["BINDING_METRICS_SKIP_WEIGHTS_CHECK"] = "1"
    return subprocess.run(
        ["bash", str(ENTRYPOINT), "echo", "command-ran"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _conda_calls(sandbox) -> list[str]:
    if not sandbox["log"].exists():
        return []
    return sandbox["log"].read_text(encoding="utf-8").splitlines()


def _weights(sandbox, name, root=None):
    directory = root or sandbox["home"] / ".openfold3"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"fake")
    return directory


def test_the_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(ENTRYPOINT)]).returncode == 0


class TestDownload:
    def test_an_empty_volume_runs_setup_without_prompt_answers(self, sandbox):
        result = _run(sandbox, create=True)
        assert result.returncode == 0, result.stderr
        calls = _conda_calls(sandbox)
        assert calls[0] == "run -n openfold3 --no-capture-output setup_openfold --non-interactive"
        # nothing is piped in: the old canned answers broke with setup_openfold >= 0.4.2
        assert calls[1] == "stdin bytes: 0"
        assert "command-ran" in result.stdout
        assert "Setup complete" in result.stdout

    def test_the_banner_names_the_openbind_checkpoint_not_preview2(self, sandbox):
        result = _run(sandbox, create=True)
        assert "openbind-2025-06-30-174k" in result.stdout
        assert "openfold3-p2-155k" not in result.stdout

    def test_preview2_weights_alone_do_not_count_as_present(self, sandbox):
        _weights(sandbox, "of3-p2-155k.pt")
        (sandbox["home"] / ".openfold3" / "ckpt_root").write_text(
            f"{sandbox['home'] / '.openfold3'}\n", encoding="utf-8"
        )
        result = _run(sandbox, create=True)
        assert result.returncode == 0, result.stderr
        assert len(_conda_calls(sandbox)) == 2  # setup ran
        assert "Preview weights" in result.stdout
        assert "do not load into" in result.stdout

    def test_a_setup_that_exits_non_zero_stops_with_the_reason(self, sandbox):
        result = _run(sandbox, exit_code=3)
        assert result.returncode == 1
        assert "setup_openfold exited non-zero" in result.stdout
        assert "command-ran" not in result.stdout

    def test_a_setup_that_leaves_the_file_missing_stops(self, sandbox):
        result = _run(sandbox, create=False)
        assert result.returncode == 1
        assert f"{DEFAULT_FILE} is still missing" in result.stdout
        assert "command-ran" not in result.stdout


class TestPresentWeights:
    def test_the_default_checkpoint_skips_setup(self, sandbox):
        _weights(sandbox, DEFAULT_FILE)
        result = _run(sandbox)
        assert result.returncode == 0, result.stderr
        assert _conda_calls(sandbox) == []
        assert result.stdout.strip() == "command-ran"

    def test_the_default_checkpoint_counts_next_to_preview2_files(self, sandbox):
        _weights(sandbox, DEFAULT_FILE)
        _weights(sandbox, "of3-p2-155k.pt")
        _run(sandbox)
        assert _conda_calls(sandbox) == []

    def test_ckpt_root_redirects_the_search(self, sandbox, tmp_path):
        elsewhere = _weights(sandbox, DEFAULT_FILE, root=tmp_path / "weights")
        cache = sandbox["home"] / ".openfold3"
        cache.mkdir()
        (cache / "ckpt_root").write_text(f"{elsewhere}\n", encoding="utf-8")
        result = _run(sandbox)
        assert result.returncode == 0, result.stderr
        assert _conda_calls(sandbox) == []

    def test_the_skip_variable_bypasses_the_check(self, sandbox):
        result = _run(sandbox, skip_check=True)
        assert result.returncode == 0
        assert _conda_calls(sandbox) == []
        assert result.stdout.strip() == "command-ran"
