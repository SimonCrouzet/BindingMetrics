"""The pipeline's metric names come from the registry and stay what they were (issue #30)."""

import argparse

import pytest

from binding_metrics.cli import run
from binding_metrics.cli.run import ALL_METRICS, KNOWN_METRICS, REFERENCE_METRICS, _parse_metrics
from binding_metrics.metrics.registry import get_metric

# The literals cli/run.py held before it read the registry.
OLD_ALL_METRICS = frozenset({"energy", "interface", "geometry", "electrostatics", "openfold"})
OLD_REFERENCE_METRICS = frozenset({"dockq"})


def test_sets_equal_the_former_literals():
    assert ALL_METRICS == OLD_ALL_METRICS
    assert REFERENCE_METRICS == OLD_REFERENCE_METRICS
    assert KNOWN_METRICS == OLD_ALL_METRICS | OLD_REFERENCE_METRICS


def test_accepted_metrics_option_values_are_unchanged():
    assert _parse_metrics("energy, interface") == frozenset({"energy", "interface"})
    assert _parse_metrics(",".join(sorted(KNOWN_METRICS))) == KNOWN_METRICS
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown metric.*ramachandran"):
        _parse_metrics("energy,ramachandran")
    with pytest.raises(argparse.ArgumentTypeError, match="Valid choices: dockq, electrostatics"):
        _parse_metrics("nonsense")


def test_every_step_runs_registered_metrics_that_take_a_path():
    for step, names in run._STEP_METRICS.items():
        for name in names:
            spec = get_metric(name)
            assert spec.input_type not in run._NON_PATH_INPUT_TYPES, (step, name)


def test_steps_over_in_memory_inputs_are_left_out():
    # hbonds takes a loaded AtomArray; evobind_score takes a pLDDT array.
    steps = run._steps_taking_a_path(
        {
            "interface": ("interface",),
            "hbonds": ("hbonds",),
            "evobind": ("evobind_score",),
            "mixed": ("interface", "saltbridges"),
        }
    )
    assert steps == frozenset({"interface"})


def test_a_step_naming_an_unregistered_metric_fails_loudly():
    with pytest.raises(KeyError, match="Unknown metric 'no_such_metric'"):
        run._steps_taking_a_path({"step": ("no_such_metric",)})


def test_batch_and_run_share_the_registry_derived_sets():
    from binding_metrics.cli import batch

    assert batch.ALL_METRICS is run.ALL_METRICS
