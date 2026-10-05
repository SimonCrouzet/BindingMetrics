"""The ``Relaxer`` contract and a relaxer injected into ``run_pipeline`` (issue #32)."""

import inspect
from pathlib import Path

import pytest

from binding_metrics.cli.run import _collect_failures, run_pipeline
from binding_metrics.protocols import relaxation
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationResult
from binding_metrics.protocols.relaxer import Relaxer

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


class RecordingRelaxer(Relaxer):
    """Relaxes nothing: reports the input as its own minimised structure."""

    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def run(self, input_path, output_dir, sample_id=None):
        self.calls.append((Path(input_path), Path(output_dir), sample_id))
        if not self.success:
            return RelaxationResult(sample_id=sample_id, success=False, error_message="stub failed")
        return RelaxationResult(
            sample_id=sample_id,
            success=True,
            potential_energy_minimized=-1234.5,
            minimized_structure_path=str(input_path),
            platform="Stub",
        )


class TestContract:
    def test_relaxer_cannot_be_instantiated_without_run(self):
        with pytest.raises(TypeError, match="abstract"):
            Relaxer()

        class Incomplete(Relaxer):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_implicit_relaxation_is_a_relaxer(self):
        assert issubclass(ImplicitRelaxation, Relaxer)
        assert isinstance(ImplicitRelaxation(relaxation.RelaxationConfig()), Relaxer)

    def test_implicit_relaxation_keeps_the_contract_signature(self):
        contract = inspect.signature(Relaxer.run)
        implementation = inspect.signature(ImplicitRelaxation.run)
        assert list(implementation.parameters) == list(contract.parameters)
        for name, parameter in contract.parameters.items():
            assert implementation.parameters[name].default == parameter.default

    def test_run_is_the_only_abstract_method(self):
        assert Relaxer.__abstractmethods__ == frozenset({"run"})


class TestInjectedRelaxer:
    @pytest.fixture(autouse=True)
    def _default_relaxer_must_not_be_built(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("the default relaxer was built although one was injected")

        monkeypatch.setattr(relaxation, "ImplicitRelaxation", refuse)
        monkeypatch.setattr(relaxation, "RelaxationConfig", refuse)

    def _run(self, tmp_path, relaxer, **kwargs):
        return run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            sample_id="s1",
            skip_prep=True,
            relaxer=relaxer,
            metrics=kwargs.pop("metrics", frozenset()),
            **kwargs,
        )

    def test_the_injected_relaxer_does_the_relaxation(self, tmp_path):
        relaxer = RecordingRelaxer()
        results = self._run(tmp_path, relaxer)
        assert relaxer.calls == [(EXAMPLE_1YCR, tmp_path, "s1")]
        assert results["relax"]["success"] is True
        assert results["relax"]["potential_energy_minimized"] == -1234.5
        assert results["relax"]["elapsed_s"] >= 0.0
        assert results["provenance"]["platform"] == "Stub"
        assert _collect_failures(results) == []

    def test_downstream_metrics_read_the_relaxed_structure(self, tmp_path):
        relaxer = RecordingRelaxer()
        results = self._run(tmp_path, relaxer, metrics=frozenset({"interface"}))
        # MDM2 (A) and the p53 peptide (B) bury about 1500 A^2 at their interface.
        assert 800.0 < results["interface"]["delta_sasa"] < 3000.0

    def test_a_failed_relaxation_is_reported_and_the_pipeline_continues(self, tmp_path):
        results = self._run(
            tmp_path, RecordingRelaxer(success=False), metrics=frozenset({"interface"})
        )
        assert results["relax"]["success"] is False
        assert results["relax"]["error_message"] == "stub failed"
        assert [step for step, _ in _collect_failures(results)] == ["relax"]
        assert results["interface"]["delta_sasa"] > 0.0  # analysed the unrelaxed input

    def test_skip_relax_leaves_the_relaxer_unused(self, tmp_path):
        relaxer = RecordingRelaxer()
        results = self._run(tmp_path, relaxer, skip_relax=True)
        assert relaxer.calls == []
        assert results["relax"] == {"skipped": True}
