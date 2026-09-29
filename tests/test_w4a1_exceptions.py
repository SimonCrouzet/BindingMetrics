"""Exception handling in the metrics modules: chained causes, narrowed catches, warnings.

The metrics behave as before; these tests pin what a caller can observe when an optional
dependency is missing or a step fails: the raised error names the original one as its
cause, and a failure that is caught on purpose leaves a record.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics import geometry, interface, polar_contacts, receptor_quality, sasa
from binding_metrics.metrics._common import import_biotite
from binding_metrics.metrics.comparison import compute_structure_rmsd

P53_MDM2 = Path(__file__).resolve().parents[1] / "data" / "example_linear_p53_1YCR.pdb"

# Per-structure isolation: whatever a step raises, the result carries a reason.
STEP_FAILURES = [RuntimeError("boom"), ValueError("boom"), IndexError("boom"), KeyError("boom")]


class TestCommon:
    def test_missing_biotite_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "biotite.structure", None)
        with pytest.raises(ImportError, match="biotite is required for tests") as info:
            import_biotite("tests")
        assert isinstance(info.value.__cause__, ImportError)


class TestComparison:
    def test_missing_gemmi_error_names_the_original_import_failure(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "gemmi", None)
        with pytest.raises(ImportError, match="gemmi is required") as info:
            compute_structure_rmsd(tmp_path / "a.pdb", tmp_path / "b.pdb", "A")
        assert isinstance(info.value.__cause__, ImportError)


class TestGeometry:
    def test_missing_scipy_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
        with pytest.raises(ImportError, match="scipy is required") as info:
            geometry._import_scipy()
        assert isinstance(info.value.__cause__, ImportError)


class TestReceptorQuality:
    def test_missing_scipy_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
        with pytest.raises(ImportError, match="scipy is required") as info:
            receptor_quality._import_scipy()
        assert isinstance(info.value.__cause__, ImportError)

    def test_missing_openmm_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openmm", None)
        with pytest.raises(ImportError, match="openmm is required") as info:
            receptor_quality._import_openmm()
        assert isinstance(info.value.__cause__, ImportError)


class TestSasa:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_static_sasa_failure_is_recorded_whatever_its_type(self, monkeypatch, failure):
        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(importlib.import_module("biotite.structure.sasa"), "sasa", broken)
        result = sasa.compute_delta_sasa_static(P53_MDM2, peptide_chain="B", receptor_chain="A")
        assert np.isnan(result["delta_sasa"])
        assert result["reason"].startswith(f"SASA computation failed: {type(failure).__name__}")


class TestInterface:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_sasa_failure_is_recorded_whatever_its_type(self, monkeypatch, failure):
        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(interface, "_per_atom_sasa", broken)
        result = interface.compute_interface_metrics(P53_MDM2)
        assert result["reason"].startswith(f"SASA computation failed: {type(failure).__name__}")

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_polar_contact_failure_is_recorded_whatever_its_type(self, monkeypatch, failure):
        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(polar_contacts, "compute_hbonds", broken)
        result = interface.compute_interface_metrics(P53_MDM2)
        assert result["reason"].startswith(
            f"H-bond/salt bridge computation failed: {type(failure).__name__}"
        )
        assert result["hbonds"] == 0
        # The SASA part of the result is still computed.
        assert np.isfinite(result["delta_sasa"])
