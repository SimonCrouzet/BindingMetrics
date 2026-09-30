"""Lazy exports of the prediction API: ``compute_prediction_metrics`` and the run-once store."""

import subprocess
import sys
import textwrap

import pytest

import binding_metrics
import binding_metrics.metrics as metrics_package
from binding_metrics import predictors

STORE_NAMES = {
    "PredictionRequest": "binding_metrics.predictors.store",
    "StoredPrediction": "binding_metrics.predictors.store",
    "PredictionStore": "binding_metrics.predictors.store",
    "PredictionFailedError": "binding_metrics.predictors.store",
    "PredictionUnavailableError": "binding_metrics.predictors.store",
    "PredictionSession": "binding_metrics.predictors.session",
    "PredictionRunner": "binding_metrics.predictors.runners",
    "OpenFold3Runner": "binding_metrics.predictors.of3_runner",
}


class TestComputePredictionMetrics:
    @pytest.mark.parametrize("package", [binding_metrics, metrics_package])
    def test_it_is_exported_by_both_packages(self, package):
        from binding_metrics.metrics.prediction import compute_prediction_metrics

        assert "compute_prediction_metrics" in package.__all__
        assert (
            package._EXPORTS["compute_prediction_metrics"] == "binding_metrics.metrics.prediction"
        )
        assert package.compute_prediction_metrics is compute_prediction_metrics

    def test_it_is_the_function_of_the_registered_metric(self):
        from binding_metrics.metrics.registry import get_metric

        spec = get_metric("prediction")
        assert spec.import_path == "binding_metrics.metrics.prediction:compute_prediction_metrics"
        assert metrics_package.compute_prediction_metrics is spec.load()


class TestStoreNames:
    @pytest.mark.parametrize("name, module", sorted(STORE_NAMES.items()))
    def test_each_name_is_exported_from_its_module(self, name, module):
        import importlib

        assert predictors._EXPORTS[name] == module
        assert name in predictors.__all__
        assert getattr(predictors, name) is getattr(importlib.import_module(module), name)

    def test_the_names_load_without_a_model_or_a_heavy_dependency(self):
        code = textwrap.dedent(
            f"""
            import sys
            import binding_metrics.predictors as p
            for name in {sorted(STORE_NAMES)!r}:
                getattr(p, name)
            heavy = [m for m in ("biotite", "scipy", "openmm", "gemmi", "torch", "openfold3")
                     if m in sys.modules]
            model_code = [m for m in ("binding_metrics.metrics.openfold",) if m in sys.modules]
            print(heavy, model_code)
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        assert out.strip() == "[] []"
