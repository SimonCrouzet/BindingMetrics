"""The registry docstrings document the whole API, so they cannot drift from the code.

The module docstring is the consumer-facing contract (what may be relied on, how
to build a call from a spec). These tests fail when a field, an input type, a
chain mode or an allowed metadata value is added to the code without a line in
the documentation.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing

import pytest

from binding_metrics.metrics import registry
from binding_metrics.metrics.registry import MetricSpec

ORIGINAL_FIELDS = (
    "name",
    "import_path",
    "description",
    "input_type",
    "chain_mode",
    "formats",
    "path_arg",
    "secondary_path_arg",
    "chain_arg",
    "peptide_chain_arg",
    "receptor_chain_arg",
)
METADATA_FIELDS = (
    "headline_key",
    "direction",
    "unit",
    "cost_class",
    "requires_extras",
    "requires_gpu",
)


class TestModuleDocstring:
    def test_exists(self):
        assert registry.__doc__ and len(registry.__doc__.split()) > 150

    @pytest.mark.parametrize("input_type", typing.get_args(registry.InputType))
    def test_every_input_type_is_described(self, input_type):
        assert f"\n{input_type}\n" in registry.__doc__, f"input type {input_type!r} undocumented"

    @pytest.mark.parametrize("chain_mode", typing.get_args(registry.ChainMode))
    def test_every_chain_mode_is_described(self, chain_mode):
        assert chain_mode in registry.__doc__

    @pytest.mark.parametrize("field", ORIGINAL_FIELDS + METADATA_FIELDS)
    def test_every_field_is_named(self, field):
        assert f"``{field}``" in registry.__doc__, f"field {field!r} not named in module docstring"

    def test_states_the_call_contract(self):
        doc = registry.__doc__
        assert "spec.call(**kwargs)" in doc
        assert "ImportError" in doc
        assert "Building a call from a spec" in doc


class TestClassDocstring:
    def test_every_dataclass_field_is_documented(self):
        doc = inspect.getdoc(MetricSpec)
        for field in dataclasses.fields(MetricSpec):
            assert f"\n{field.name}:\n" in "\n" + doc, f"{field.name} missing from MetricSpec doc"

    def test_metadata_allowed_values_are_documented(self):
        doc = inspect.getdoc(MetricSpec)
        for cost_class in typing.get_args(registry.CostClass):
            assert f'"{cost_class}"' in doc, cost_class
        for direction in typing.get_args(registry.Direction):
            assert direction in doc, direction

    def test_allowed_sets_match_the_type_aliases(self):
        assert set(registry.DIRECTIONS) == set(typing.get_args(registry.Direction))
        assert set(registry.COST_CLASSES) == set(typing.get_args(registry.CostClass))


class TestFunctionDocstrings:
    @pytest.mark.parametrize(
        "obj",
        [
            MetricSpec.load,
            MetricSpec.call,
            registry.get_metric,
            registry.metrics_by_input_type,
        ],
        ids=lambda o: o.__qualname__,
    )
    def test_public_api_has_a_docstring(self, obj):
        assert inspect.getdoc(obj), f"{obj.__qualname__} has no docstring"

    def test_load_documents_its_failure_modes(self):
        assert "ImportError" in inspect.getdoc(MetricSpec.load)

    def test_get_metric_documents_its_failure_mode(self):
        assert "KeyError" in inspect.getdoc(registry.get_metric)

    def test_call_documents_that_it_adds_nothing(self):
        assert "unchanged" in inspect.getdoc(MetricSpec.call)
