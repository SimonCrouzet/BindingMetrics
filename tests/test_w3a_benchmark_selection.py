"""``benchmarks/run.py`` keeps timing the metrics it timed before the registry grew (issue #30).

The registry went from 18 to 28 specs. The benchmark must not start timing the new ones
(``receptor_quality`` needs a GPU, ``evobind_adversarial`` is a self-comparison), or its
result files stop being comparable with earlier runs.
"""

import importlib.util
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).parent.parent / "benchmarks" / "run.py"

# Registry order at the time the 18-spec registry was current.
OLD_STATIC = [
    "interface",
    "coulomb",
    "ramachandran",
    "omega",
    "shape_complementarity",
    "void_volume",
    "structure_rmsd",
    "dockq",
]
OLD_TRAJECTORY = [
    "interaction_energy",
    "component_energies",
    "rmsd",
    "rmsf",
    "ligand_rmsd",
    "receptor_drift",
    "buried_sasa",
    "contacts",
]
# The ten specs registered since.
NEWER = [
    "delta_sasa_static",
    "receptor_quality",
    "evobind_adversarial",
    "hbonds",
    "saltbridges",
    "evobind_score",
    "interface_sasa",
    "contact_residues",
    "structure_interaction_energy",
    "interface_pae",
]


@pytest.fixture(scope="module")
def benchmark():
    if not BENCHMARK.exists():
        pytest.skip("benchmarks/run.py is not part of this checkout")
    spec = importlib.util.spec_from_file_location("bm_benchmark_run", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_and_trajectory_loops_time_the_former_metrics(benchmark):
    assert [s.name for s in benchmark.select_specs("static_structure")] == OLD_STATIC
    assert [s.name for s in benchmark.select_specs("trajectory")] == OLD_TRAJECTORY


def test_newer_registry_entries_are_not_timed(benchmark):
    timed = {
        s.name
        for input_type in ("static_structure", "trajectory")
        for s in benchmark.select_specs(input_type)
    }
    assert not timed & set(NEWER)


def test_metrics_option_offers_the_former_names(benchmark):
    names = benchmark._ALL_METRIC_NAMES
    assert set(names) == set(OLD_STATIC + OLD_TRAJECTORY + ["md_implicit", "openfold"])
    assert len(names) == 18
    assert not set(names) & set(NEWER)


def test_requested_metrics_narrow_the_selection(benchmark):
    specs = benchmark.select_specs("static_structure", ["coulomb", "hbonds", "omega"])
    # ``hbonds`` is a newer spec (and an atom_array metric), so it stays out.
    assert [s.name for s in specs] == ["coulomb", "omega"]


def test_every_listed_metric_is_still_registered(benchmark):
    from binding_metrics.metrics.registry import METRICS_BY_NAME

    assert benchmark._BENCHMARKED_METRICS <= set(METRICS_BY_NAME)
