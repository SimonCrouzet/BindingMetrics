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


class TestBatchOpenFold:
    @staticmethod
    def _run(tmp_path, monkeypatch, of_metrics=None, compute_error=None, evobind_error=None):
        from binding_metrics.cli import batch
        from binding_metrics.metrics import evobind, openfold

        def compute(**_kwargs):
            if compute_error is not None:
                raise compute_error
            return dict(of_metrics or {})

        def score(*_args, **_kwargs):
            raise evobind_error

        monkeypatch.setattr(openfold, "run_openfold_batched", lambda **_kw: tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", compute)
        if evobind_error is not None:
            monkeypatch.setattr(evobind, "compute_evobind_score", score)
        rows = [{"sample_id": "s1", "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={"s1": P53_MDM2},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain="B",
            receptor_chain="A",
        )
        return rows[0]

    def test_chain_detection_failure_skips_the_sample_and_is_logged(
        self, tmp_path, monkeypatch, caplog
    ):
        from binding_metrics.cli import batch

        def broken(*_args, **_kwargs):
            raise ValueError("no chains")

        monkeypatch.setattr("binding_metrics.io.structures.detect_chains_from_file", broken)
        rows = [{"sample_id": "s1", "batch_status": "ok"}]
        with caplog.at_level(logging.WARNING, logger="binding_metrics"):
            batch._run_batched_openfold(
                rows=rows,
                sid_to_input={"s1": P53_MDM2},
                output_dir=tmp_path,
                openfold_mode="score",
                openfold_conda_env=None,
                peptide_chain=None,
                receptor_chain=None,
            )

        assert rows == [{"sample_id": "s1", "batch_status": "ok"}]
        assert "s1: skipped for OpenFold, chain detection failed: no chains" in caplog.text

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_evobind_failure_is_recorded_in_the_row(self, tmp_path, monkeypatch, failure):
        row = self._run(
            tmp_path, monkeypatch, of_metrics={"structure_path": "of3.cif"}, evobind_error=failure
        )

        assert row["openfold_evobind_error"] == str(failure)

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_metrics_failure_is_recorded_in_the_row_and_logged(
        self, tmp_path, monkeypatch, caplog, failure
    ):
        with caplog.at_level(logging.WARNING, logger="binding_metrics"):
            row = self._run(tmp_path, monkeypatch, compute_error=failure)

        assert row["openfold_error"] == str(failure)
        assert "s1: OpenFold metrics failed" in caplog.text

    @pytest.mark.parametrize(
        "content", ["{not json", "[1, 2]"], ids=["invalid-json", "not-an-object"]
    )
    def test_unusable_sample_report_is_left_alone_and_logged(self, tmp_path, caplog, content):
        from binding_metrics.cli import batch

        report = tmp_path / "s1_results.json"
        report.write_text(content)

        with caplog.at_level(logging.WARNING, logger="binding_metrics"):
            batch._update_sample_json(tmp_path, "s1", {"iptm": 0.5})

        assert report.read_text() == content
        assert "s1: could not update s1_results.json" in caplog.text


class TestRunPipelineCatches:
    METRIC_ENTRY_POINTS = [
        ("energy", "binding_metrics.metrics.energy", "compute_interaction_energy"),
        ("interface", "binding_metrics.metrics.interface", "compute_interface_metrics"),
        ("geometry", "binding_metrics.metrics.geometry", "compute_ramachandran"),
        ("electrostatics", "binding_metrics.metrics.electrostatics", "compute_coulomb_cross_chain"),
        ("dockq", "binding_metrics.metrics.dockq", "compute_dockq_metrics"),
    ]

    @staticmethod
    def _run(tmp_path, metrics, **kwargs):
        from binding_metrics.cli.run import run_pipeline

        return run_pipeline(
            P53_MDM2,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset(metrics),
            **kwargs,
        )

    @pytest.mark.parametrize(("metric", "module", "function"), METRIC_ENTRY_POINTS)
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_metric_is_recorded_and_the_run_carries_on(
        self, tmp_path, monkeypatch, caplog, metric, module, function, failure
    ):
        def broken(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(f"{module}.{function}", broken)
        with caplog.at_level(logging.WARNING, logger="binding_metrics"):
            results = self._run(tmp_path, {metric}, reference_path=P53_MDM2)

        assert results[metric] == {"error": str(failure)}
        assert "failed" in caplog.text

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_prep_is_recorded_and_the_raw_input_is_used(
        self, tmp_path, monkeypatch, failure
    ):
        from binding_metrics.cli.run import run_pipeline
        from binding_metrics.core import system

        def broken(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(system, "prep_structure", broken)

        results = run_pipeline(P53_MDM2, tmp_path, skip_relax=True, metrics=frozenset())

        assert results["prep"] == {"error": str(failure)}
        assert results["relax"] == {"skipped": True}

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_unreadable_cyclic_hints_are_logged_and_the_run_carries_on(
        self, tmp_path, monkeypatch, caplog, failure
    ):
        def broken(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr("binding_metrics.io.structures.load_structure", broken)
        with caplog.at_level(logging.DEBUG, logger="binding_metrics.cli.run"):
            results = self._run(tmp_path, set())

        assert results["relax"] == {"skipped": True}
        assert "Cyclic bond hints unavailable" in caplog.text

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_openfold_is_recorded(self, tmp_path, monkeypatch, failure):
        from binding_metrics.metrics import openfold

        def broken(**_kwargs):
            raise failure

        monkeypatch.setattr(openfold, "run_openfold_scoring", broken)

        assert self._run(tmp_path, {"openfold"})["openfold"] == {"error": str(failure)}

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failing_evobind_steps_are_recorded_next_to_the_openfold_metrics(
        self, tmp_path, monkeypatch, failure
    ):
        from binding_metrics.metrics import evobind, openfold

        def broken(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(openfold, "run_openfold_scoring", lambda **_kw: tmp_path)
        monkeypatch.setattr(
            openfold, "compute_openfold_metrics", lambda **_kw: {"structure_path": "of3.cif"}
        )
        monkeypatch.setattr(evobind, "compute_evobind_score", broken)
        monkeypatch.setattr(evobind, "compute_evobind_adversarial_check", broken)

        of_metrics = self._run(tmp_path, {"openfold"})["openfold"]

        assert of_metrics["evobind_error"] == str(failure)
        assert of_metrics["adversarial_error"] == str(failure)
