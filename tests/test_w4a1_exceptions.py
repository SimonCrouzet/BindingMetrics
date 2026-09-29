"""Exception handling in the metrics modules: chained causes, narrowed catches, warnings.

The metrics behave as before; these tests pin what a caller can observe when an optional
dependency is missing or a step fails: the raised error names the original one as its
cause, and a failure that is caught on purpose leaves a record.
"""

import sys

import pytest

from binding_metrics.metrics import geometry, receptor_quality
from binding_metrics.metrics._common import import_biotite
from binding_metrics.metrics.comparison import compute_structure_rmsd


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
