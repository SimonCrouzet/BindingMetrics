"""Poses: one structure of a complex, classified by where it came from.

``Pose`` is the abstract base class; ``DockedPose`` (a docking or design tool),
``ExperimentalPose`` (a crystal, NMR or EM structure) and ``PredictedPose`` (a co-folding
model, in ``binding_metrics.predictors.record``) are its kinds. Every pose has an ``id``, the
structure (``atoms()``), a ``content_hash()``, ``provenance`` and ``native_scores()``: what
the tool that made it reported. A computed metric is never stored on a pose.

Names load on first access (PEP 562); importing this package imports nothing heavy, and
biotite is imported only when a structure is read.
"""

from typing import TYPE_CHECKING

from binding_metrics import _lazy_exports

# Public name -> defining module; see ``binding_metrics._lazy_exports``.
_EXPORTS = {
    "Pose": "binding_metrics.poses.pose",
    "DockedPose": "binding_metrics.poses.docked",
    "ExperimentalPose": "binding_metrics.poses.experimental",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.poses.docked import DockedPose
    from binding_metrics.poses.experimental import ExperimentalPose
    from binding_metrics.poses.pose import Pose

__all__ = [
    "DockedPose",
    "ExperimentalPose",
    "Pose",
]
