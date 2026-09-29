"""collect_provenance: keys, JSON safety and best-effort behaviour."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from binding_metrics import provenance
from binding_metrics.provenance import SCHEMA_VERSION, collect_provenance

EXPECTED_KEYS = {
    "schema_version",
    "package_version",
    "git_sha",
    "python",
    "os",
    "openmm_version",
    "platform",
    "seed",
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    """The version and sha lookups are cached; keep tests independent."""
    provenance._package_version.cache_clear()
    provenance._git_sha.cache_clear()
    yield
    provenance._package_version.cache_clear()
    provenance._git_sha.cache_clear()


def test_all_keys_present_with_expected_types():
    prov = collect_provenance(seed=7, platform="CUDA")
    assert set(prov) == EXPECTED_KEYS
    assert prov["schema_version"] == SCHEMA_VERSION == 1
    assert prov["seed"] == 7
    assert prov["platform"] == "CUDA"
    assert re.fullmatch(r"\d+\.\d+\.\d+", prov["python"])
    assert isinstance(prov["package_version"], str) and prov["package_version"]


def test_defaults_are_none():
    prov = collect_provenance()
    assert prov["seed"] is None
    assert prov["platform"] is None


def test_json_round_trip():
    prov = collect_provenance(seed=1, platform="CPU")
    assert json.loads(json.dumps(prov)) == prov


def test_package_version_matches_the_package():
    import binding_metrics

    assert collect_provenance()["package_version"] == binding_metrics.__version__


def test_package_version_falls_back_to_dunder_version(monkeypatch):
    import importlib.metadata as md

    import binding_metrics

    def missing(_name):
        raise md.PackageNotFoundError

    monkeypatch.setattr(md, "version", missing)
    assert collect_provenance()["package_version"] == binding_metrics.__version__


def test_git_sha_is_a_full_sha_or_none():
    sha = collect_provenance()["git_sha"]
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha)


@pytest.mark.parametrize("failure", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 5)])
def test_missing_or_hanging_git_gives_none_and_never_raises(monkeypatch, failure):
    def boom(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    prov = collect_provenance(seed=3)
    assert prov["git_sha"] is None
    assert prov["seed"] == 3
    json.dumps(prov)


def test_failing_git_command_gives_none(monkeypatch):
    def not_a_repo(cmd, **_kwargs):
        return subprocess.CompletedProcess(
            cmd, 128, stdout="", stderr="fatal: not a git repository"
        )

    monkeypatch.setattr(provenance.subprocess, "run", not_a_repo)
    assert collect_provenance()["git_sha"] is None


def _fake_git(toplevel, sha="a" * 40):
    def run(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{toplevel}\n{sha}\n", stderr="")

    return run


def test_sha_is_reported_for_this_packages_own_checkout(monkeypatch):
    repo_root = Path(provenance.__file__).resolve().parents[2]
    monkeypatch.setattr(provenance.subprocess, "run", _fake_git(repo_root))
    assert collect_provenance()["git_sha"] == "a" * 40


def test_sha_of_an_unrelated_enclosing_repository_is_not_reported(monkeypatch, tmp_path):
    """An installed copy inside someone else's git repo must not borrow its sha."""
    monkeypatch.setattr(provenance.subprocess, "run", _fake_git(tmp_path))
    assert collect_provenance()["git_sha"] is None


def test_unparseable_git_output_gives_none(monkeypatch):
    def garbage(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", garbage)
    assert collect_provenance()["git_sha"] is None


def test_openmm_missing_gives_none(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openmm", None)  # makes `import openmm` raise
    assert collect_provenance()["openmm_version"] is None
