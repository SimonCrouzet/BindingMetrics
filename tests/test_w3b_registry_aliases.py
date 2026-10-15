"""``binder_chain_arg`` and ``target_chain_arg`` on the registry specs."""

import inspect
from pathlib import Path

import pytest

from binding_metrics.metrics.registry import METRICS, MetricSpec, get_metric

P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def _has_binder_role(spec: MetricSpec) -> bool:
    """A chain-ID binder argument: not an index list, not the receptor-only ``chain_arg``."""
    if spec.peptide_chain_arg and spec.peptide_chain_arg != "ligand_indices":
        return True
    return spec.chain_arg == "chain"


def _has_target_role(spec: MetricSpec) -> bool:
    if spec.receptor_chain_arg and spec.receptor_chain_arg != "receptor_indices":
        return True
    return spec.chain_arg == "receptor_chain"


def test_new_fields_default_to_none():
    spec = MetricSpec(name="x", import_path="m:f", description="d", input_type="static_structure")
    assert spec.binder_chain_arg is None and spec.target_chain_arg is None


def test_new_fields_come_after_the_existing_ones():
    """A spec built positionally keeps its meaning."""
    import dataclasses

    names = [f.name for f in dataclasses.fields(MetricSpec)]
    assert names[-2:] == ["binder_chain_arg", "target_chain_arg"]


@pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
def test_alias_names_are_the_role_names_or_none(spec):
    assert spec.binder_chain_arg in (None, "binder_chain")
    assert spec.target_chain_arg in (None, "target_chain")


@pytest.mark.parametrize(
    "spec",
    [s for s in METRICS if s.binder_chain_arg or s.target_chain_arg],
    ids=lambda s: s.name,
)
def test_declared_aliases_are_parameters_of_the_function(spec):
    try:
        function = spec.load()
    except ImportError as exc:
        pytest.skip(f"{spec.name}: optional dependency not installed: {exc}")
    params = inspect.signature(function).parameters
    for alias in (spec.binder_chain_arg, spec.target_chain_arg):
        if alias:
            assert alias in params, f"{spec.name}: {spec.import_path} has no {alias!r} parameter"


@pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
def test_every_alias_a_function_accepts_is_declared(spec):
    """No function takes ``binder_chain``/``target_chain`` without its spec saying so."""
    try:
        function = spec.load()
    except ImportError as exc:
        pytest.skip(f"{spec.name}: optional dependency not installed: {exc}")
    if inspect.isclass(function):
        pytest.skip("class-based spec")
    params = inspect.signature(function).parameters
    if "binder_chain" in params:
        assert spec.binder_chain_arg == "binder_chain", spec.name
    if "target_chain" in params:
        assert spec.target_chain_arg == "target_chain", spec.name


@pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
def test_alias_fields_follow_the_roles_of_the_spec(spec):
    """Declared only for a role the metric has; the index-based trajectory metrics have none."""
    # ``openfold`` and ``prediction`` are registered with chain_mode "none" (they parse a
    # directory) but their functions take optional binder and receptor chains.
    takes_chains_despite_mode = spec.name in ("openfold", "prediction")
    if spec.binder_chain_arg:
        assert _has_binder_role(spec) or takes_chains_despite_mode, spec.name
    if spec.target_chain_arg:
        assert _has_target_role(spec) or takes_chains_despite_mode, spec.name


@pytest.mark.parametrize("name", ["interface", "coulomb", "delta_sasa_static"])
def test_call_through_the_registry_with_the_alias_names(name):
    if not P53.exists():
        pytest.skip("1YCR example not bundled")
    pytest.importorskip("biotite")
    spec = get_metric(name)
    via_alias = spec.call(
        **{spec.path_arg: P53, spec.binder_chain_arg: "B", spec.target_chain_arg: "A"}
    )
    via_old_names = spec.call(
        **{spec.path_arg: P53, spec.peptide_chain_arg: "B", spec.receptor_chain_arg: "A"}
    )
    for key in ("delta_sasa", "coulomb_energy_kJ", "sasa_complex"):
        if key in via_old_names:
            assert via_alias[key] == via_old_names[key]
    assert via_alias.keys() == via_old_names.keys()


def test_trajectory_metrics_declare_no_chain_alias():
    for spec in METRICS:
        if spec.input_type == "trajectory" and spec.peptide_chain_arg == "ligand_indices":
            assert spec.binder_chain_arg is None and spec.target_chain_arg is None, spec.name
