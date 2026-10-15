"""Tests for package-level imports and exports."""

import importlib

import pytest
from conftest import BEST_PLATFORM

# (module, attribute, must be callable): the names a user imports first. An attribute of None
# means the module itself. The class and submodule entries only have to import; the functions
# must also be callable.
_IMPORTS = [
    ("binding_metrics", None, False),
    ("binding_metrics", "MDSimulation", False),
    ("binding_metrics", "ForceFieldConfig", False),
    ("binding_metrics", "PeptideBindingProtocol", False),
    ("binding_metrics", "run_simulation", True),
    ("binding_metrics", "get_forcefield", True),
    ("binding_metrics", "prepare_system", True),
    ("binding_metrics.core", None, False),
    ("binding_metrics.protocols", None, False),
    ("binding_metrics.metrics", None, False),
    ("binding_metrics.io", None, False),
    ("binding_metrics.metrics", "calculate_buried_sasa", True),
    ("binding_metrics.metrics", "calculate_contacts", True),
    ("binding_metrics.metrics", "calculate_interaction_energy", True),
    ("binding_metrics.metrics", "calculate_rmsd", True),
    ("binding_metrics.io", "load_complex", True),
    ("binding_metrics.io", "get_chain_atom_indices", True),
]


class TestPackageImports:
    """Test that all public API is importable."""

    @pytest.mark.parametrize(
        ("module", "attribute", "must_be_callable"),
        _IMPORTS,
        ids=[f"{m}:{a}" if a else m for m, a, _ in _IMPORTS],
    )
    def test_the_name_imports(self, module, attribute, must_be_callable):
        imported = importlib.import_module(module)
        if attribute is not None:
            imported = getattr(imported, attribute)
        assert imported is not None
        if must_be_callable:
            assert callable(imported)

    def test_version_exists(self):
        """Package should have __version__."""
        import binding_metrics

        assert hasattr(binding_metrics, "__version__")
        # The value is the installed distribution's version, which an editable install
        # keeps until it is reinstalled: pyproject.toml and the changelog are compared
        # in tests/test_release_version.py, not here.
        assert isinstance(binding_metrics.__version__, str) and binding_metrics.__version__

    def test_import_simulation_config(self):
        """SimulationConfig should be importable from main package."""
        from binding_metrics import SimulationConfig

        config = SimulationConfig()
        assert config.temperature == 300.0

    def test_import_protocol_results(self):
        """ProtocolResults should be importable from main package."""
        from binding_metrics import ProtocolResults

        results = ProtocolResults()
        assert results.sasa_buried_mean == 0.0

    def test_all_exports(self):
        """__all__ should contain expected exports."""
        import binding_metrics

        expected = {
            # Core
            "ForceFieldConfig",
            "get_forcefield",
            "MDSimulation",
            "SimulationConfig",
            "run_simulation",
            "prepare_system",
            # I/O
            "load_structure",
            "detect_chains",
            "save_cif",
            # Protocols
            "ProtocolResults",
            "PeptideBindingProtocol",
            "ImplicitRelaxation",
            "RelaxationConfig",
            "RelaxationResult",
            "Relaxer",
            # Pipeline API
            "run_pipeline",
            "run_batch",
            # Metrics (existing)
            "compute_interaction_energy",
            "compute_structure_rmsd",
            "compute_hbonds",
            "compute_saltbridges",
            "compute_interface_metrics",
            "compute_delta_sasa_static",
            "compute_openfold_metrics",
            "run_openfold",
            "compute_prediction_metrics",
            # Metrics (new)
            "compute_coulomb_cross_chain",
            "compute_ramachandran",
            "compute_omega_planarity",
            "compute_shape_complementarity",
            "compute_buried_void_volume",
            "compute_receptor_drift",
            # EvoBind metrics
            "compute_evobind_score",
            "compute_evobind_adversarial_check",
            # DockQ reference metric
            "compute_dockq_metrics",
        }

        assert set(binding_metrics.__all__) == expected

    def test_all_exports_are_bound_attributes(self):
        """Every name in __all__ must actually resolve on the package.

        The set-equality test above guards the *literal* list against drift, but a
        name can be listed in __all__ and still not be bound (a typo, or an export
        that was removed from the submodule but left in __all__). That only surfaces
        as an AttributeError on ``from binding_metrics import *`` or ``getattr``.
        This asserts each advertised export is effectively importable.
        """
        import binding_metrics

        for name in binding_metrics.__all__:
            assert hasattr(binding_metrics, name), (
                f"__all__ lists {name!r} but it is not bound on the package"
            )
            assert getattr(binding_metrics, name) is not None


class TestTypicalUsage:
    """Test typical usage patterns work."""

    def test_create_config_and_protocol(self, sample_pdb_path):
        """Typical workflow should not raise on setup."""
        from binding_metrics import PeptideBindingProtocol, SimulationConfig

        config = SimulationConfig(
            duration_ns=1.0,
            temperature=300.0,
            platform=BEST_PLATFORM,
        )

        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
            forcefield="amber",
            simulation_config=config,
        )

        assert protocol is not None
        assert protocol.simulation_config.duration_ns == 1.0

    def test_protocol_results_workflow(self):
        """ProtocolResults should support typical analysis workflow."""
        import numpy as np

        from binding_metrics import ProtocolResults

        # Simulate analysis results
        results = ProtocolResults(
            sasa_buried=np.array([95.0, 100.0, 105.0, 98.0, 102.0]),
            sasa_buried_mean=100.0,
            sasa_buried_std=3.5,
            interface_contacts=np.array([15, 16, 14, 15, 16]),
            interface_contacts_mean=15.2,
            interaction_energy=np.array([-45.0, -50.0, -48.0, -52.0, -47.0]),
            interaction_energy_mean=-48.4,
            interaction_energy_std=2.5,
            rmsd=np.array([0.05, 0.08, 0.10, 0.09, 0.11]),
            rmsd_mean=0.086,
        )

        # Get summary
        summary = results.summary()
        assert "Buried SASA" in summary
        assert "100.0" in summary

        # Get dict for serialization
        d = results.to_dict()
        assert d["sasa_buried_mean"] == 100.0
        assert d["n_frames"] == 5
