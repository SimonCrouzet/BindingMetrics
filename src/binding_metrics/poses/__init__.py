"""Poses: one structure of a complex, classified by where it came from, and the analysis of a set.

``Pose`` is the abstract base class; ``DockedPose`` (a docking or design tool),
``ExperimentalPose`` (a crystal, NMR or EM structure) and ``PredictedPose`` (a co-folding
model; defined in ``binding_metrics.predictors.record`` and exported here too) are its
kinds. Every pose has an ``id``, the structure (``atoms()``), a ``content_hash()``,
``provenance`` and ``native_scores()``: what the tool that made it reported. A computed
metric is never stored on a pose.

``score_pose_set`` takes the poses of one construct and returns a ``PoseSetResult``: each pose
placed on a ``Reference`` receptor, measured with the registry metrics, compared with the other
poses (RMSD, clusters, medoids, interface overlap), summarised per group and with one
representative pose per group. The other names are what it is built from: the reference and the
frame (``Reference``, ``superpose_receptor``), the aggregation (``aggregate_poses``), the
descriptors of a pose against a reference (``residue_set_descriptors``) and the reconciliation of
two back-ends (``gap_between_backends``). The pose-agreement functions are in
``binding_metrics.poses.agreement``.

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
    "PredictedPose": "binding_metrics.predictors.record",
    "Reference": "binding_metrics.poses.reference",
    "Partner": "binding_metrics.poses.reference",
    "ResidueKey": "binding_metrics.poses.reference",
    "normalize_references": "binding_metrics.poses.reference",
    "group_by_frame": "binding_metrics.poses.reference",
    "polymer_residues": "binding_metrics.poses.reference",
    "FrameOptions": "binding_metrics.poses.frame",
    "FrameResult": "binding_metrics.poses.frame",
    "Transform": "binding_metrics.poses.frame",
    "superpose_receptor": "binding_metrics.poses.frame",
    "pair_residues": "binding_metrics.poses.frame",
    "prepare_reference": "binding_metrics.poses.frame",
    "place_in_reference_frame": "binding_metrics.poses.frame",
    "renumber_to_reference": "binding_metrics.poses.frame",
    "score_pose_set": "binding_metrics.poses.score",
    "PoseSetResult": "binding_metrics.poses.score",
    "PoseSetOptions": "binding_metrics.poses.score",
    "AgreementOptions": "binding_metrics.poses.score",
    "AggregationOptions": "binding_metrics.poses.score",
    "RepresentativeOptions": "binding_metrics.poses.score",
    "PoseScoringError": "binding_metrics.poses.score",
    "SCHEMA_VERSION": "binding_metrics.poses.score",
    "write_pose_report": "binding_metrics.poses.report",
    "aggregate_poses": "binding_metrics.poses.aggregate",
    "aggregate_values": "binding_metrics.poses.aggregate",
    "select_representatives": "binding_metrics.poses.aggregate",
    "PoseDistances": "binding_metrics.poses.aggregate",
    "rank_by": "binding_metrics.poses.aggregate",
    "group_key": "binding_metrics.poses.aggregate",
    "resolve_statistics": "binding_metrics.poses.aggregate",
    "Aggregator": "binding_metrics.poses.aggregate",
    "DEFAULT_STATISTICS": "binding_metrics.poses.aggregate",
    "interface_areas": "binding_metrics.poses.descriptors",
    "InterfaceAreas": "binding_metrics.poses.descriptors",
    "residue_set_descriptors": "binding_metrics.poses.descriptors",
    "partner_descriptors": "binding_metrics.poses.descriptors",
    "interface_residue_table": "binding_metrics.poses.descriptors",
    "sphere_overlap_volume": "binding_metrics.poses.descriptors",
    "SET_COLUMNS": "binding_metrics.poses.descriptors",
    "PARTNER_COLUMNS": "binding_metrics.poses.descriptors",
    "TABLE_COLUMNS": "binding_metrics.poses.descriptors",
    "DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2": "binding_metrics.poses.descriptors",
    "DEFAULT_MAX_UNMAPPED_FRACTION": "binding_metrics.poses.descriptors",
    "DEFAULT_PROBE_RADIUS": "binding_metrics.poses.descriptors",
    "DEFAULT_CLASH_CUTOFFS_ANGSTROM": "binding_metrics.poses.descriptors",
    "DEFAULT_SHELL_ANGSTROM": "binding_metrics.poses.descriptors",
    "gap_between_backends": "binding_metrics.poses.reconcile",
    "consensus_interface": "binding_metrics.poses.reconcile",
    "compare_consensus_interfaces": "binding_metrics.poses.reconcile",
    "ConsensusInterface": "binding_metrics.poses.reconcile",
    "rank_agreement": "binding_metrics.poses.reconcile",
    "hotspot_consistency": "binding_metrics.poses.reconcile",
    "POOLED_SCALE": "binding_metrics.poses.reconcile",
    "RANK_METHODS": "binding_metrics.poses.reconcile",
    "DEFAULT_TOP_M": "binding_metrics.poses.reconcile",
}

__getattr__, __dir__ = _lazy_exports(__name__, _EXPORTS, globals())

if TYPE_CHECKING:
    from binding_metrics.poses.aggregate import (
        DEFAULT_STATISTICS,
        Aggregator,
        PoseDistances,
        aggregate_poses,
        aggregate_values,
        group_key,
        rank_by,
        resolve_statistics,
        select_representatives,
    )
    from binding_metrics.poses.descriptors import (
        DEFAULT_CLASH_CUTOFFS_ANGSTROM,
        DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2,
        DEFAULT_MAX_UNMAPPED_FRACTION,
        DEFAULT_PROBE_RADIUS,
        DEFAULT_SHELL_ANGSTROM,
        PARTNER_COLUMNS,
        SET_COLUMNS,
        TABLE_COLUMNS,
        InterfaceAreas,
        interface_areas,
        interface_residue_table,
        partner_descriptors,
        residue_set_descriptors,
        sphere_overlap_volume,
    )
    from binding_metrics.poses.docked import DockedPose
    from binding_metrics.poses.experimental import ExperimentalPose
    from binding_metrics.poses.frame import (
        FrameOptions,
        FrameResult,
        Transform,
        pair_residues,
        place_in_reference_frame,
        prepare_reference,
        renumber_to_reference,
        superpose_receptor,
    )
    from binding_metrics.poses.pose import Pose
    from binding_metrics.poses.reconcile import (
        DEFAULT_TOP_M,
        POOLED_SCALE,
        RANK_METHODS,
        ConsensusInterface,
        compare_consensus_interfaces,
        consensus_interface,
        gap_between_backends,
        hotspot_consistency,
        rank_agreement,
    )
    from binding_metrics.poses.reference import (
        Partner,
        Reference,
        ResidueKey,
        group_by_frame,
        normalize_references,
        polymer_residues,
    )
    from binding_metrics.poses.report import write_pose_report
    from binding_metrics.poses.score import (
        SCHEMA_VERSION,
        AggregationOptions,
        AgreementOptions,
        PoseScoringError,
        PoseSetOptions,
        PoseSetResult,
        RepresentativeOptions,
        score_pose_set,
    )
    from binding_metrics.predictors.record import PredictedPose

__all__ = [
    "AggregationOptions",
    "Aggregator",
    "AgreementOptions",
    "ConsensusInterface",
    "DEFAULT_CLASH_CUTOFFS_ANGSTROM",
    "DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2",
    "DEFAULT_MAX_UNMAPPED_FRACTION",
    "DEFAULT_PROBE_RADIUS",
    "DEFAULT_SHELL_ANGSTROM",
    "DEFAULT_STATISTICS",
    "DEFAULT_TOP_M",
    "DockedPose",
    "ExperimentalPose",
    "FrameOptions",
    "FrameResult",
    "InterfaceAreas",
    "PARTNER_COLUMNS",
    "POOLED_SCALE",
    "Partner",
    "Pose",
    "PoseDistances",
    "PoseScoringError",
    "PoseSetOptions",
    "PoseSetResult",
    "PredictedPose",
    "RANK_METHODS",
    "Reference",
    "RepresentativeOptions",
    "ResidueKey",
    "SCHEMA_VERSION",
    "SET_COLUMNS",
    "TABLE_COLUMNS",
    "Transform",
    "aggregate_poses",
    "aggregate_values",
    "compare_consensus_interfaces",
    "consensus_interface",
    "gap_between_backends",
    "group_by_frame",
    "group_key",
    "hotspot_consistency",
    "interface_areas",
    "interface_residue_table",
    "normalize_references",
    "pair_residues",
    "partner_descriptors",
    "place_in_reference_frame",
    "polymer_residues",
    "prepare_reference",
    "rank_agreement",
    "rank_by",
    "renumber_to_reference",
    "residue_set_descriptors",
    "resolve_statistics",
    "score_pose_set",
    "select_representatives",
    "sphere_overlap_volume",
    "superpose_receptor",
    "write_pose_report",
]
