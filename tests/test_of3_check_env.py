"""``binding-metrics-check-env``: the OpenFold3 check reports the version and the checkpoint.

Importing ``openfold3`` says nothing about the weights: openfold3 >= 0.5.0 stops without the
OpenBind-0 file, and Preview2 weights of a 0.4.x install do not load into it. The tests use
tiny fake checkpoint files and a stubbed version probe; a real install is not needed, so what
they cannot show is that the file names match a real download.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import check_env
from binding_metrics.metrics import _openfold_run

DEFAULT_FILE = "of3-ob-2025-06-30-174k.pt"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty OpenFold3 cache directory that ``$OPENFOLD_CACHE`` points to."""
    directory = tmp_path / "of3-cache"
    directory.mkdir()
    monkeypatch.setenv("OPENFOLD_CACHE", str(directory))
    return directory


@pytest.fixture
def version(monkeypatch):
    """Set the version that the probe reports (None means not readable)."""
    box = {"value": "0.5.0"}
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: box["value"]
    )
    return box


def _report(capsys, where="current environment"):
    ready = check_env._report_openfold_readiness([sys.executable], where)
    return ready, capsys.readouterr().out


class TestReadiness:
    def test_current_version_with_the_default_checkpoint_passes(self, cache, version, capsys):
        (cache / DEFAULT_FILE).write_bytes(b"fake")
        ready, out = _report(capsys)
        assert ready
        assert "openfold3 0.5.0 available in current environment" in out
        assert f"default checkpoint {DEFAULT_FILE} found in {cache}" in out

    def test_a_missing_default_checkpoint_fails_and_says_how_to_get_it(
        self, cache, version, capsys
    ):
        ready, out = _report(capsys)
        assert not ready
        assert f"Default checkpoint {DEFAULT_FILE} not found" in out
        assert "setup_openfold --non-interactive" in out
        assert "cowardly refusing" in out

    def test_preview2_weights_alone_do_not_count(self, cache, version, capsys):
        (cache / "of3-p2-155k.pt").write_bytes(b"fake")
        ready, out = _report(capsys)
        assert not ready
        assert "Only Preview weights are there (of3-p2-155k.pt)" in out
        assert "do not load into openfold3 >= 0.5" in out

    def test_the_checkpoint_root_file_redirects_the_search(self, tmp_path, cache, version, capsys):
        weights = tmp_path / "elsewhere"
        weights.mkdir()
        (weights / DEFAULT_FILE).write_bytes(b"fake")
        (cache / "ckpt_root").write_text(f"{weights}\n", encoding="utf-8")
        ready, out = _report(capsys)
        assert ready
        assert str(weights) in out

    def test_an_old_installation_passes_with_a_warning(self, cache, version, capsys):
        version["value"] = "0.4.5"
        (cache / "of3-p2-155k.pt").write_bytes(b"fake")
        ready, out = _report(capsys)
        assert ready
        assert "openfold3 0.4.5 available" in out
        assert "predates 0.5.0" in out
        assert "setup_openfold --non-interactive" in out

    def test_an_unreadable_version_passes_with_a_warning(self, cache, version, capsys):
        version["value"] = None
        ready, out = _report(capsys)
        assert ready
        assert "version not readable" in out
        assert "cannot be judged" in out

    def test_the_conda_env_name_is_reported(self, cache, version, capsys):
        (cache / DEFAULT_FILE).write_bytes(b"fake")
        _, out = _report(capsys, where="conda env 'openfold3'")
        assert "available in conda env 'openfold3'" in out


class TestCheckOpenfold:
    """The whole check, with the import probes stubbed."""

    @pytest.fixture
    def importable(self, monkeypatch):
        def _run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(check_env.subprocess, "run", _run)

    def test_a_present_import_without_weights_fails_the_check(
        self, cache, version, importable, capsys
    ):
        assert check_env._check_openfold() is False
        assert "Default checkpoint" in capsys.readouterr().out

    def test_a_present_import_with_weights_passes_the_check(
        self, cache, version, importable, capsys
    ):
        (cache / DEFAULT_FILE).write_bytes(b"fake")
        assert check_env._check_openfold() is True

    def test_the_install_command_still_names_the_package(self):
        source = Path(check_env.__file__).read_text(encoding="utf-8")
        assert "pip install openfold3" in source
        assert "setup_openfold --non-interactive" in source


class TestWithoutConda:
    """A missing ``conda`` executable means the dedicated env cannot exist (#97)."""

    @pytest.fixture
    def no_conda(self, monkeypatch):
        calls = []

        def _run(cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "conda" or Path(cmd[0]).name == "conda":
                raise FileNotFoundError(2, "No such file or directory: 'conda'")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")  # not importable

        monkeypatch.setattr(check_env.subprocess, "run", _run)
        monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
        return calls

    def test_the_check_reports_openfold3_as_not_found(self, no_conda, capsys):
        assert check_env._check_openfold() is False
        out = capsys.readouterr().out
        assert "OpenFold3 not found" in out
        assert "pip install openfold3" in out
        assert "optional" in out

    def test_conda_is_not_asked_a_second_time(self, no_conda):
        check_env._check_openfold()
        assert no_conda.count("conda") == 1

    def test_the_whole_command_ends_with_a_report_not_a_traceback(
        self, no_conda, monkeypatch, capsys
    ):
        monkeypatch.setattr(check_env, "CHECKS", [("OpenFold3", check_env._check_openfold)])
        with pytest.raises(SystemExit) as info:
            check_env.main()
        assert info.value.code == 1
        assert "1 failed" in capsys.readouterr().out
