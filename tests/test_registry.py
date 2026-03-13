"""Tests for binding_metrics.metrics.registry."""

import pytest

from binding_metrics.metrics.registry import (
    METRICS,
    METRICS_BY_NAME,
    MetricSpec,
    get_metric,
    metrics_by_input_type,
)

_VALID_INPUT_TYPES = {"static_structure", "trajectory", "md_simulation", "openfold_json"}
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


class TestMetricSpecLoad:
    """Verify import_path strings resolve to callables when deps are present."""

    def _try_load(self, spec: MetricSpec):
        """Attempt to load the metric; skip if dependency is missing."""
        try:
            fn = spec.load()
            assert callable(fn), f"{spec.name}: load() did not return a callable"
        except ImportError as e:
            pytest.skip(f"{spec.name}: optional dependency not installed — {e}")

    def test_load_interface(self):
        self._try_load(get_metric("interface"))

    def test_load_coulomb(self):
        self._try_load(get_metric("coulomb"))

    def test_load_ramachandran(self):
        self._try_load(get_metric("ramachandran"))

    def test_load_omega(self):
        self._try_load(get_metric("omega"))

    def test_load_shape_complementarity(self):
        self._try_load(get_metric("shape_complementarity"))

    def test_load_void_volume(self):
        self._try_load(get_metric("void_volume"))

    def test_load_structure_rmsd(self):
        self._try_load(get_metric("structure_rmsd"))

    def test_load_rmsd(self):
        self._try_load(get_metric("rmsd"))

    def test_load_rmsf(self):
        self._try_load(get_metric("rmsf"))

    def test_load_receptor_drift(self):
        self._try_load(get_metric("receptor_drift"))

    def test_load_buried_sasa(self):
        self._try_load(get_metric("buried_sasa"))

    def test_load_contacts(self):
        self._try_load(get_metric("contacts"))

    def test_load_openfold(self):
        self._try_load(get_metric("openfold"))


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
