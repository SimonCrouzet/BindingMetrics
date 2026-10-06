"""Fixtures for tests of code built on BindingMetrics: no pytest import, plain functions.

``make_synthetic_poses`` builds a pose set from the bundled 1YCR complex whose pairwise
RMSD is known in closed form, so a regression test needs no model output. ``tests/`` is not
part of the installed package; this subpackage is.

Names load on first access (PEP 562); importing this package imports nothing heavy, and
biotite is imported only when a structure is read or written.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "make_synthetic_poses": "binding_metrics.testing.synthetic",
    "bundled_complex_path": "binding_metrics.testing.synthetic",
    "SyntheticPoses": "binding_metrics.testing.synthetic",
    "SyntheticTruth": "binding_metrics.testing.synthetic",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.testing.synthetic import (
        SyntheticPoses,
        SyntheticTruth,
        bundled_complex_path,
        make_synthetic_poses,
    )

__all__ = [
    "SyntheticPoses",
    "SyntheticTruth",
    "bundled_complex_path",
    "make_synthetic_poses",
]
