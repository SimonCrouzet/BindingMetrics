"""Tests for binding_metrics.metrics.registry."""

import importlib
import inspect
import pkgutil
import re

import pytest

from binding_metrics.metrics.registry import (
    METRICS,
    METRICS_BY_NAME,
    MetricSpec,
    get_metric,
    metrics_by_input_type,
)

_VALID_INPUT_TYPES = {
    "static_structure",
    "trajectory",
    "md_simulation",
    "openfold_json",
    "atom_array",
    "predicted_structure",
    "prediction_dir",
}
_VALID_CHAIN_MODES = {"none", "single", "interface", "interface_2paths"}


class TestRegistryStructure:
    def test_metrics_is_nonempty(self):
        assert len(METRICS) > 0

    def test_all_entries_are_metric_spec(self):
        for m in METRICS:
            assert isinstance(m, MetricSpec)

    def test_no_duplicate_names(self):
        names = [m.name for m in METRICS]
        assert len(names) == len(set(names)), f"Duplicate metric names: {names}"

    def test_metrics_by_name_covers_all(self):
        assert set(METRICS_BY_NAME.keys()) == {m.name for m in METRICS}

    def test_all_names_nonempty(self):
        for m in METRICS:
            assert m.name.strip(), f"Empty name in {m}"

    def test_all_import_paths_have_colon(self):
        for m in METRICS:
            assert ":" in m.import_path, (
                f"{m.name}: import_path must be 'module:function', got {m.import_path!r}"
            )

    def test_all_input_types_valid(self):
        for m in METRICS:
            assert m.input_type in _VALID_INPUT_TYPES, (
                f"{m.name}: unknown input_type {m.input_type!r}"
            )

    def test_all_chain_modes_valid(self):
        for m in METRICS:
            assert m.chain_mode in _VALID_CHAIN_MODES, (
                f"{m.name}: unknown chain_mode {m.chain_mode!r}"
            )

    def test_all_descriptions_nonempty(self):
        for m in METRICS:
            assert m.description.strip(), f"{m.name}: empty description"

    def test_all_formats_are_known(self):
        known = {"pdb", "cif", "mmcif"}
        for m in METRICS:
            for fmt in m.formats:
                assert fmt in known, f"{m.name}: unknown format {fmt!r}"


class TestMetricsByInputType:
    def test_static_structure_nonempty(self):
        specs = metrics_by_input_type("static_structure")
        assert len(specs) > 0

    def test_trajectory_nonempty(self):
        specs = metrics_by_input_type("trajectory")
        assert len(specs) > 0

    def test_md_simulation_nonempty(self):
        specs = metrics_by_input_type("md_simulation")
        assert len(specs) > 0

    def test_openfold_json_nonempty(self):
        specs = metrics_by_input_type("openfold_json")
        assert len(specs) > 0

    def test_atom_array_nonempty(self):
        specs = metrics_by_input_type("atom_array")
        assert {s.name for s in specs} == {"hbonds", "saltbridges"}

    def test_predicted_structure_nonempty(self):
        specs = metrics_by_input_type("predicted_structure")
        assert {s.name for s in specs} == {"evobind_score"}

    def test_prediction_dir_holds_the_model_neutral_prediction_metric(self):
        specs = metrics_by_input_type("prediction_dir")
        assert {s.name for s in specs} == {"prediction"}

    def test_all_types_partition_metrics(self):
        """Every metric appears in exactly one input_type bucket."""
        all_from_buckets = []
        for it in _VALID_INPUT_TYPES:
            all_from_buckets.extend(metrics_by_input_type(it))
        assert len(all_from_buckets) == len(METRICS)

    def test_returns_only_matching_type(self):
        for it in _VALID_INPUT_TYPES:
            for spec in metrics_by_input_type(it):
                assert spec.input_type == it


class TestGetMetric:
    def test_known_metric_returns_spec(self):
        spec = get_metric("interface")
        assert spec.name == "interface"

    def test_unknown_metric_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown_metric"):
            get_metric("unknown_metric")

    def test_error_message_lists_available(self):
        with pytest.raises(KeyError) as exc_info:
            get_metric("does_not_exist")
        assert "Available" in str(exc_info.value)


class TestTheCoreMetricsAreRegistered:
    """The metrics that every report and runner relies on keep their registry entry.

    Whether each entry's ``import_path`` resolves is checked for the whole registry by
    ``TestEveryMetricLoads.test_import_path_resolves_to_callable``.
    """

    _CORE = (
        "interface",
        "coulomb",
        "ramachandran",
        "omega",
        "shape_complementarity",
        "void_volume",
        "structure_rmsd",
        "rmsd",
        "rmsf",
        "receptor_drift",
        "buried_sasa",
        "contacts",
        "openfold",
        "dockq",
    )

    def test_each_core_metric_has_an_entry(self):
        missing = [name for name in self._CORE if name not in METRICS_BY_NAME]
        assert not missing, f"missing from the registry: {missing}"


class TestEveryMetricLoads:
    """Effective check: EVERY registered import_path must resolve to a callable.

    ``test_all_import_paths_have_colon`` only checks the *string* shape. A new
    metric with a typo'd module/function (or a renamed function) would pass it
    yet blow up the first time a runner calls ``spec.load()``. Iterating over the
    whole registry closes that gap so import_path drift can't slip through.
    """

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_import_path_resolves_to_callable(self, spec):
        try:
            fn = spec.load()
        except ImportError as e:
            pytest.skip(f"{spec.name}: optional dependency not installed — {e}")
        assert callable(fn), f"{spec.name}: import_path did not resolve to a callable"

    @pytest.mark.parametrize("spec", METRICS, ids=lambda s: s.name)
    def test_declared_names_are_parameters_of_the_function(self, spec):
        """Every argument name the spec declares is a parameter of the function it loads.

        A generic runner feeds the primary input to every metric through
        ``spec.call(**{spec.path_arg: ...})`` and adds the chain and secondary-path
        keywords the spec names. A name that the function does not accept raises
        ``TypeError`` at runtime: the declared-versus-effective gap that the registry
        claims to close. One case per metric lists every name that is wrong.

        * ``path_arg`` is checked for every spec, class-based ones included. It
          regression-guards the ``md_implicit`` bug where the spec declared
          ``path_arg="input_path"`` for ``ImplicitRelaxation``, whose ``__init__`` only
          takes ``config``.
        * ``secondary_path_arg``, ``chain_arg``, ``peptide_chain_arg``,
          ``receptor_chain_arg``, ``binder_chain_arg`` and ``target_chain_arg`` are checked
          for functions only (a class is constructed, not called by keyword), and the five
          that the runner forwards must also bind to the signature.
        * ``binder_chain_arg`` and ``target_chain_arg`` must be parameters by name, even
          for a function that takes ``**kwargs``; and a function that takes ``binder_chain``
          or ``target_chain`` must have a spec that declares it.
        """
        try:
            obj = spec.load()
        except ImportError as e:
            pytest.skip(f"{spec.name}: optional dependency not installed — {e}")

        params = inspect.signature(obj).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        problems = []
        if not (has_var_kw or spec.path_arg in params):
            problems.append(f"path_arg {spec.path_arg!r} is not a parameter")
        for role in ("binder_chain_arg", "target_chain_arg"):
            alias = getattr(spec, role)
            if alias and alias not in params:
                problems.append(f"{role} {alias!r} is not a parameter")

        if not inspect.isclass(obj):
            forwarded = {
                role: getattr(spec, role)
                for role in (
                    "path_arg",
                    "secondary_path_arg",
                    "chain_arg",
                    "peptide_chain_arg",
                    "receptor_chain_arg",
                )
                if getattr(spec, role)
            }
            for role, arg in forwarded.items():
                if role != "path_arg" and not (has_var_kw or arg in params):
                    problems.append(f"{role} {arg!r} is not a parameter")
            try:
                # bind_partial: required parameters that the spec does not declare (a pLDDT
                # array, a query name) are the caller's business; unknown names are the
                # registry's bug.
                inspect.signature(obj).bind_partial(**{arg: object() for arg in forwarded.values()})
            except TypeError as e:
                problems.append(f"the declared names do not bind to the signature: {e}")
            if "binder_chain" in params and spec.binder_chain_arg != "binder_chain":
                problems.append("the function takes binder_chain but binder_chain_arg is not set")
            if "target_chain" in params and spec.target_chain_arg != "target_chain":
                problems.append("the function takes target_chain but target_chain_arg is not set")

        assert not problems, f"{spec.name} ({spec.import_path}): " + "; ".join(problems)


class TestStaticStructureSpecs:
    """Spot-check chain mode / format metadata for static metrics."""

    def test_interface_is_cif_and_pdb(self):
        spec = get_metric("interface")
        assert "pdb" in spec.formats
        assert "cif" in spec.formats

    def test_interface_chain_mode_is_interface(self):
        assert get_metric("interface").chain_mode == "interface"

    def test_structure_rmsd_has_secondary_path(self):
        spec = get_metric("structure_rmsd")
        assert spec.secondary_path_arg is not None
        assert spec.chain_mode == "interface_2paths"

    def test_ramachandran_is_single_chain(self):
        spec = get_metric("ramachandran")
        assert spec.chain_mode == "single"
        assert spec.chain_arg is not None

    def test_omega_is_single_chain(self):
        spec = get_metric("omega")
        assert spec.chain_mode == "single"


class TestTrajectorySpecs:
    """Spot-check trajectory metric metadata."""

    def test_rmsd_chain_mode_none(self):
        # rmsd auto-detects atoms, needs no chain arg
        assert get_metric("rmsd").chain_mode == "none"

    def test_rmsf_chain_mode_none(self):
        assert get_metric("rmsf").chain_mode == "none"

    def test_receptor_drift_is_single_chain(self):
        spec = get_metric("receptor_drift")
        assert spec.chain_mode == "single"
        assert spec.chain_arg == "receptor_chain"

    def test_interface_metrics_have_both_chain_args(self):
        for name in ("interaction_energy", "buried_sasa", "contacts", "ligand_rmsd"):
            spec = get_metric(name)
            assert spec.chain_mode == "interface", f"{name}: expected interface chain_mode"
            assert spec.peptide_chain_arg is not None
            assert spec.receptor_chain_arg is not None


# ---------------------------------------------------------------------------
# Coverage: every public metric function has a registry entry
# ---------------------------------------------------------------------------

_METRIC_FUNCTION_PATTERN = re.compile(r"^(compute|calculate)_")

# Public ``compute_*`` / ``calculate_*`` functions that are deliberately not metrics.
# Map "module:function" to the reason. Add an entry only for a helper or for an
# interface that raises NotImplementedError until a backend lands (the MLFF
# interaction energy): a working metric belongs in the registry.
_NOT_METRICS: dict[str, str] = {
    "binding_metrics.metrics.mlff_energy:compute_mlff_interaction_energy": (
        "interface only; registered when a backend lands"
    ),
    "binding_metrics.metrics.evobind:compute_evobind_adversarial_from_records": (
        "takes PredictedPose objects, not paths; called by code, not by the registry consumers"
    ),
    "binding_metrics.metrics.evobind:compute_evobind_score_from_record": (
        "takes a PredictedPose object, not a path; called by code, not by the registry consumers"
    ),
}


def _metric_modules() -> list[str]:
    import binding_metrics.metrics as metrics_pkg

    return sorted(
        f"{metrics_pkg.__name__}.{info.name}"
        for info in pkgutil.iter_modules(metrics_pkg.__path__)
        if info.name != "registry"
    )


class TestRegistryCoverage:
    """A metric function that is not registered is invisible to every registry consumer."""

    @pytest.mark.parametrize("module_name", _metric_modules())
    def test_every_public_metric_function_is_registered(self, module_name):
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            pytest.skip(f"{module_name}: optional dependency not installed — {e}")

        registered = {m.import_path for m in METRICS}
        missing = [
            f"{module_name}:{name}"
            for name, fn in inspect.getmembers(module, inspect.isfunction)
            if fn.__module__ == module_name
            and _METRIC_FUNCTION_PATTERN.match(name)
            and f"{module_name}:{name}" not in registered
            and f"{module_name}:{name}" not in _NOT_METRICS
        ]
        assert not missing, (
            f"public metric functions without a registry entry: {missing}. "
            "Add a MetricSpec, or list the function in _NOT_METRICS with the reason."
        )

    def test_package_exports_are_registered(self):
        """Every metric function re-exported by ``binding_metrics.metrics`` is registered."""
        import binding_metrics.metrics as metrics_pkg

        registered_names = {m.import_path.split(":")[1] for m in METRICS}
        exported = [n for n in metrics_pkg.__all__ if _METRIC_FUNCTION_PATTERN.match(n)]
        missing = [n for n in exported if n not in registered_names]
        assert not missing, f"exported metric functions without a registry entry: {missing}"

    def test_not_metrics_entries_are_documented(self):
        for target, reason in _NOT_METRICS.items():
            assert ":" in target and reason.strip()


class TestNewlyRegisteredMetrics:
    """The ten metrics that were public but missing from the registry."""

    _EXPECTED = {
        "receptor_quality": "binding_metrics.metrics.receptor_quality:compute_receptor_quality",
        "evobind_score": "binding_metrics.metrics.evobind:compute_evobind_score",
        "evobind_adversarial": (
            "binding_metrics.metrics.evobind:compute_evobind_adversarial_check"
        ),
        "interface_pae": "binding_metrics.metrics.openfold:compute_interface_pae",
        "structure_interaction_energy": (
            "binding_metrics.metrics.energy:compute_interaction_energy"
        ),
        "hbonds": "binding_metrics.metrics.polar_contacts:compute_hbonds",
        "saltbridges": "binding_metrics.metrics.polar_contacts:compute_saltbridges",
        "delta_sasa_static": "binding_metrics.metrics.sasa:compute_delta_sasa_static",
        "interface_sasa": "binding_metrics.metrics.sasa:calculate_interface_sasa",
        "contact_residues": "binding_metrics.metrics.contacts:calculate_contact_residues",
    }

    @pytest.mark.parametrize("name", sorted(_EXPECTED))
    def test_registered_with_expected_import_path(self, name):
        assert get_metric(name).import_path == self._EXPECTED[name]

    def test_interaction_energy_entries_are_distinct(self):
        """The per-frame trajectory metric and the per-structure one both stay reachable."""
        per_frame = get_metric("interaction_energy")
        per_structure = get_metric("structure_interaction_energy")
        assert per_frame.import_path.endswith(":calculate_interaction_energy")
        assert per_structure.import_path.endswith(":compute_interaction_energy")
        assert per_frame.input_type == "trajectory"
        assert per_structure.input_type == "md_simulation"

    def test_receptor_quality_declares_only_the_receptor_role(self):
        spec = get_metric("receptor_quality")
        assert spec.receptor_chain_arg == "receptor_chain"
        assert spec.peptide_chain_arg is None

    def test_two_path_entries_declare_a_secondary_path(self):
        for name in ("evobind_adversarial", "interface_pae"):
            spec = get_metric(name)
            assert spec.chain_mode == "interface_2paths"
            assert spec.secondary_path_arg, f"{name}: no secondary_path_arg"
            assert spec.secondary_path_arg != spec.path_arg
