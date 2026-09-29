"""Exception handling in core, io, protocols, cli and the helper modules.

The behaviour is unchanged; these tests pin what a caller can observe when a step fails on
purpose-broad catches: the run carries on with a documented sentinel, and the failure
leaves a record (a log line or a result key) whatever the exception type.
"""

import logging
import subprocess
import sys

import pytest

from binding_metrics import provenance
from binding_metrics.provenance import collect_provenance

# A catch that isolates one step must hold for any exception type the step can raise.
STEP_FAILURES = [RuntimeError("boom"), ValueError("boom"), OSError("boom"), KeyError("boom")]


@pytest.fixture
def fresh_provenance_caches():
    provenance._package_version.cache_clear()
    provenance._git_sha.cache_clear()
    yield
    provenance._package_version.cache_clear()
    provenance._git_sha.cache_clear()


@pytest.mark.usefixtures("fresh_provenance_caches")
class TestProvenance:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_git_failure_gives_none_and_a_debug_record(self, monkeypatch, caplog, failure):
        def boom(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(provenance.subprocess, "run", boom)
        with caplog.at_level(logging.DEBUG, logger="binding_metrics.provenance"):
            prov = collect_provenance()

        assert prov["git_sha"] is None
        assert "git sha unavailable" in caplog.text

    def test_timeout_gives_none_and_a_debug_record(self, monkeypatch, caplog):
        def hang(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("git", 5)

        monkeypatch.setattr(provenance.subprocess, "run", hang)
        with caplog.at_level(logging.DEBUG, logger="binding_metrics.provenance"):
            assert collect_provenance()["git_sha"] is None

        assert "git sha unavailable" in caplog.text

    def test_broken_metadata_gives_no_version_and_a_debug_record(self, monkeypatch, caplog):
        import importlib.metadata as metadata

        def broken(_name):
            raise ValueError("malformed metadata")

        monkeypatch.setattr(metadata, "version", broken)
        with caplog.at_level(logging.DEBUG, logger="binding_metrics.provenance"):
            prov = collect_provenance()

        assert prov["package_version"] is None
        assert "package version unavailable" in caplog.text

    def test_missing_openmm_gives_none_and_a_debug_record(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "openmm", None)
        with caplog.at_level(logging.DEBUG, logger="binding_metrics.provenance"):
            prov = collect_provenance()

        assert prov["openmm_version"] is None
        assert "OpenMM version unavailable" in caplog.text

    def test_failing_platform_probe_gives_no_os_name_and_a_debug_record(self, monkeypatch, caplog):
        def boom():
            raise OSError("no uname")

        monkeypatch.setattr(provenance._platform, "platform", boom)
        with caplog.at_level(logging.DEBUG, logger="binding_metrics.provenance"):
            prov = collect_provenance()

        assert prov["os"] is None
        assert "OS name unavailable" in caplog.text
