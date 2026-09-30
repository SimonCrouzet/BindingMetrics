"""compute_prediction_metrics and the ``prediction`` registry spec."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics import openfold
from binding_metrics.metrics.prediction import compute_prediction_metrics
from binding_metrics.metrics.registry import get_metric
from binding_metrics.predictors.registry import PARSERS, ParserSpec, register_parser
from tests.predictors import contract, synth, synth_of3, synth_stub
from tests.predictors.test_openfold_golden import _KEYS

NAME = contract.NAME


@pytest.fixture
def of3_run(tmp_path):
    synth_of3.write_prediction(tmp_path, NAME, synth.synthetic_complex())
    return tmp_path


@pytest.fixture
def stub_registered():
    saved = dict(PARSERS)
    register_parser(
        ParserSpec(
            name="stub",
            import_path="tests.predictors.synth_stub:StubParser",
            display_name="Stub model",
            family="af3",
        ),
        replace=True,
    )
    yield
    PARSERS.clear()
    PARSERS.update(saved)


class TestComputePredictionMetrics:
    def test_for_openfold3_it_is_the_legacy_result_plus_the_model(self, of3_run):
        kwargs = dict(binder_chain="B", receptor_chain="A")
        legacy = openfold.compute_openfold_metrics(of3_run, NAME, include_matrices=True, **kwargs)
        result = compute_prediction_metrics(of3_run, "of3", NAME, include_matrices=True, **kwargs)
        assert list(result) == ["model", *_KEYS]
        assert result.pop("model") == "of3"
        np.testing.assert_equal(result, legacy)

    def test_another_model_goes_through_its_own_adapter(self, tmp_path, stub_registered):
        synth_stub.write_prediction(tmp_path, NAME, synth.synthetic_complex())
        result = compute_prediction_metrics(
            tmp_path, "stub", NAME, binder_chain="B", receptor_chain="A"
        )
        assert result["model"] == "stub"
        assert result["avg_plddt"] == pytest.approx(82.0)  # the stub writes 0-1, converted
        assert result["sample_ranking_score"] == pytest.approx(0.82)
        assert result["mean_interface_pae"] == pytest.approx(3.4375)
        assert result["binder_avg_plddt"] == pytest.approx(76.0)
        assert result["bespoke_iptm"] == {}
        assert "reason" not in result

    def test_chain_map_lets_the_caller_use_its_own_chain_names(self, of3_run):
        result = compute_prediction_metrics(
            of3_run,
            "of3",
            NAME,
            binder_chain="P",
            receptor_chain="R",
            chain_map={"A": "R", "B": "P"},
        )
        assert result["mean_interface_pde"] == pytest.approx(1.9375)
        assert result["binder_avg_plddt"] == pytest.approx(76.0)
        # without the map the caller's names do not exist in the structure
        with pytest.warns(UserWarning, match="binder pLDDT|interface"):
            unmapped = compute_prediction_metrics(
                of3_run, "of3", NAME, binder_chain="P", receptor_chain="R"
            )
        assert np.isnan(unmapped["mean_interface_pde"])

    def test_seed_index_takes_precedence_and_target_chain_is_an_alias(self, of3_run):
        synth_of3.write_prediction(
            of3_run, NAME, synth.synthetic_complex(plddt_shift=10.0), seed_index=2
        )
        result = compute_prediction_metrics(
            of3_run, "of3", NAME, seed=1, seed_index=2, binder_chain="B", target_chain="A"
        )
        assert (result["seed"], result["avg_plddt"]) == (2, pytest.approx(72.0))
        assert result["mean_interface_pae"] == pytest.approx(3.4375)
        with pytest.raises(ValueError, match="target_chain"):
            compute_prediction_metrics(of3_run, "of3", NAME, receptor_chain="A", target_chain="C")

    def test_an_unknown_model_lists_the_known_ones(self, of3_run):
        with pytest.raises(KeyError, match=r"Unknown predictor 'nope'. Available: .*of3"):
            compute_prediction_metrics(of3_run, "nope", NAME)

    def test_an_invalid_chain_map_is_refused(self, of3_run):
        with pytest.raises(ValueError, match="same ID"):
            compute_prediction_metrics(of3_run, "of3", NAME, chain_map={"A": "X", "B": "X"})

    def test_a_missing_directory_gives_nan_and_a_reason(self, tmp_path):
        result = compute_prediction_metrics(tmp_path / "nowhere", "of3", NAME)
        assert result["model"] == "of3" and np.isnan(result["avg_plddt"])
        assert result["reason"].startswith(f"no confidence files found for query '{NAME}'")

    def test_a_warning_names_the_caller_and_the_function(self, tmp_path):
        run = tmp_path / "run"
        synth_of3.write_prediction(run, NAME, synth.synthetic_complex())
        (run / NAME / "seed_9" / f"{NAME}_seed_9_sample_1_confidences.json").write_text(
            '{"plddt": [90.0, 80.0]}', encoding="utf-8"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_prediction_metrics(run, "of3", NAME, binder_chain="B", receptor_chain="A")
        assert str(caught[0].message).startswith("compute_prediction_metrics: ")
        assert Path(caught[0].filename).name == Path(__file__).name


class TestRegistrySpec:
    def test_the_prediction_spec_describes_the_function(self):
        spec = get_metric("prediction")
        assert spec.import_path == "binding_metrics.metrics.prediction:compute_prediction_metrics"
        assert spec.input_type == "prediction_dir"
        assert spec.chain_mode == "none"
        assert spec.formats == ()
        assert spec.path_arg == "prediction_dir"
        assert (spec.binder_chain_arg, spec.target_chain_arg) == ("binder_chain", "target_chain")
        assert spec.cost_class == "model"
        assert spec.requires_extras == ("biotite",)
        assert spec.headline_key is None and spec.direction is None and spec.unit is None

    def test_declared_arguments_are_parameters_of_the_function(self):
        spec = get_metric("prediction")
        parameters = inspect.signature(spec.load()).parameters
        for argument in (spec.path_arg, spec.binder_chain_arg, spec.target_chain_arg):
            assert argument in parameters
        assert "model" in parameters and "name" in parameters

    def test_call_through_the_registry(self, of3_run):
        spec = get_metric("prediction")
        result = spec.call(
            **{spec.path_arg: of3_run, "model": "of3", "name": NAME},
            **{spec.binder_chain_arg: "B", spec.target_chain_arg: "A"},
        )
        assert result["model"] == "of3"
        assert result["mean_interface_pae"] == pytest.approx(3.4375)

    def test_the_openfold_entries_are_unchanged(self):
        assert get_metric("openfold").input_type == "openfold_json"
        assert get_metric("interface_pae").input_type == "openfold_json"
