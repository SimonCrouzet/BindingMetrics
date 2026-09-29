"""Exception handling in core, io, protocols, cli and the helper modules.

The behaviour is unchanged; these tests pin what a caller can observe when a step fails on
purpose-broad catches: the run carries on with a documented sentinel, and the failure
leaves a record (a log line or a result key) whatever the exception type.
"""

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from binding_metrics import provenance
from binding_metrics.provenance import collect_provenance

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "run.py"
TEST_RUNNER = ROOT / "scripts" / "run_tests.py"
P53_MDM2 = ROOT / "data" / "example_linear_p53_1YCR.pdb"

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


@pytest.fixture(scope="module")
def benchmark():
    if not BENCHMARK.exists():
        pytest.skip("benchmarks/run.py is not part of this checkout")
    spec = importlib.util.spec_from_file_location("bm_benchmark_run_w4a2", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkRun:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_chain_detection_failure_keeps_the_counts_and_is_logged(
        self, benchmark, monkeypatch, caplog, failure
    ):
        def broken(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr("binding_metrics.metrics.interface.detect_interface_chains", broken)
        with caplog.at_level(logging.WARNING, logger=benchmark.logger.name):
            props = benchmark.compute_structure_properties(P53_MDM2, design_chain="B")

        assert props["n_atoms"] > 0
        assert props["peptide_chain"] == "B"
        assert props["receptor_chain"] is None
        assert "Could not detect the interface chains" in caplog.text

    def test_unreadable_trajectory_keeps_the_entry_and_is_logged(self, benchmark, tmp_path, caplog):
        entry = {"ligand_chain": "B", "receptor_chain": "A"}
        with caplog.at_level(logging.WARNING, logger=benchmark.logger.name):
            props = benchmark.compute_trajectory_properties(
                tmp_path / "missing.xtc", tmp_path / "missing.pdb", entry
            )

        assert props["n_frames"] is None
        assert props["ligand_chain"] == "B"
        assert "Could not read the trajectory properties" in caplog.text

    def test_argument_inspection_failure_gives_an_empty_set(self, benchmark, caplog):
        def broken_load():
            raise ImportError("no module")

        spec = SimpleNamespace(name="demo", load=broken_load)
        with caplog.at_level(logging.DEBUG, logger=benchmark.logger.name):
            assert benchmark._fn_argnames(spec) == set()

        assert "Could not inspect the arguments of demo" in caplog.text

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_static_metric_is_recorded_in_the_error_key(
        self, benchmark, monkeypatch, failure
    ):
        def broken_load():
            raise failure

        monkeypatch.setattr(benchmark, "_build_static_kwargs", lambda *_args: {})
        spec = SimpleNamespace(load=broken_load, formats=("pdb",))

        result = benchmark.bench_static_metric(spec, Path("x.pdb"), {}, n_runs=1)

        assert result == {"error": str(failure)}

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_trajectory_metric_is_recorded_in_the_error_key(
        self, benchmark, monkeypatch, failure
    ):
        def broken_load():
            raise failure

        monkeypatch.setattr(benchmark, "_build_trajectory_kwargs", lambda *_args: {})
        spec = SimpleNamespace(load=broken_load)

        result = benchmark.bench_trajectory_metric(spec, Path("x.xtc"), Path("x.pdb"), {}, n_runs=1)

        assert result == {"error": str(failure)}

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_md_run_is_recorded_in_the_error_key(
        self, benchmark, monkeypatch, tmp_path, failure
    ):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation

        def broken_run(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(ImplicitRelaxation, "run", broken_run)

        result = benchmark.bench_md_simulation(P53_MDM2, {}, tmp_path)

        assert result == {"error": str(failure)}


class TestTestRunnerScript:
    def test_gpu_probe_failure_is_reported_as_cpu(self, monkeypatch, capsys):
        if not TEST_RUNNER.exists():
            pytest.skip("scripts/run_tests.py is not part of this checkout")
        spec = importlib.util.spec_from_file_location("bm_run_tests_w4a2", TEST_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, "openmm", None)  # makes `import openmm` raise

        module.check_environment()

        output = capsys.readouterr().out
        assert "GPU" in output
        assert "probe failed (ModuleNotFoundError), using CPU" in output
