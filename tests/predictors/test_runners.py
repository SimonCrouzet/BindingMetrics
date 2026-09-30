"""PredictionRunner: the abstract steps and the defaults of the optional ones."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from binding_metrics.predictors.runners import PredictionRunner


class _Minimal(PredictionRunner):
    name = "minimal"

    def prepare(self, request, work_dir):
        return Path(work_dir) / "input.txt"

    def run(self, request, work_dir):
        return Path(work_dir)


class TestTheAbstractClass:
    def test_cannot_be_instantiated_without_prepare_and_run(self):
        class OnlyRun(PredictionRunner):
            name = "only_run"

            def run(self, request, work_dir):
                return Path(work_dir)

        with pytest.raises(TypeError, match="abstract"):
            OnlyRun()

    def test_a_subclass_with_the_two_steps_can_be_instantiated(self):
        assert _Minimal().name == "minimal"

    def test_no_capabilities_are_declared_by_default(self):
        """The pre-flight lane defines the class later; a runner declares None until then."""
        assert PredictionRunner.capabilities is None
        assert _Minimal.capabilities is None

    def test_a_runner_may_declare_capabilities_and_a_sibling_is_unaffected(self):
        marker = object()

        class Constrained(_Minimal):
            capabilities = marker

        assert Constrained.capabilities is marker
        assert _Minimal.capabilities is None


class TestTheOptionalSteps:
    def test_a_runner_is_available_and_has_no_version_by_default(self):
        runner = _Minimal()
        assert runner.is_available() is True
        assert runner.version() is None

    def test_no_request_can_be_batched_by_default(self):
        assert _Minimal().supports_batch(object()) is False

    def test_run_many_says_that_the_runner_has_no_batched_mode(self):
        with pytest.raises(NotImplementedError, match="minimal runner has no batched mode"):
            _Minimal().run_many([], Path("."))

    def test_the_message_survives_a_runner_without_a_name(self):
        class Nameless(PredictionRunner):
            def prepare(self, request, work_dir):
                return Path(work_dir)

            def run(self, request, work_dir):
                return Path(work_dir)

        with pytest.raises(NotImplementedError, match="Nameless runner"):
            Nameless().run_many([], Path("."))


def test_importing_the_module_imports_only_the_standard_library():
    code = textwrap.dedent(
        """
        import sys
        import binding_metrics.predictors.runners
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "openfold3", "gemmi")
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
