"""Exception handling in the metrics modules: chained causes, narrowed catches, warnings.

The metrics behave as before; these tests pin what a caller can observe when an optional
dependency is missing or a step fails: the raised error names the original one as its
cause, and a failure that is caught on purpose leaves a record.
"""

import sys

import pytest

from binding_metrics.metrics._common import import_biotite


class TestCommon:
    def test_missing_biotite_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "biotite.structure", None)
        with pytest.raises(ImportError, match="biotite is required for tests") as info:
            import_biotite("tests")
        assert isinstance(info.value.__cause__, ImportError)
