"""Score a set of poses of one construct: ``score_pose_set`` and its result, ``PoseSetResult``.

One call takes the poses (a pose is one structure of the complex with the record of where it
came from, see ``binding_metrics.poses.pose``), places each of them in the frame of one
reference receptor, measures each with the registry metrics that were asked for, measures how
much the poses agree with each other, summarises each group of poses (the back-end by
default) and names one representative pose per group. The result is one nested dictionary.
Nothing here says "good" or "bad": thresholds are options with documented defaults.

Stages
------
1. Every argument is checked before a structure is read: an unknown or unsupported metric, an
   unknown option or choice, a repeated pose id, a metric name that is also a reserved row name
   (``ValueError``, ``KeyError``, ``TypeError``).
2. The first pose that can be read (in id order) gives the chains (``binder_chain`` and
   ``target_chain`` are detected from it when they are not given), the implicit reference (when
   no ``reference`` is given, see below) and the profile that the pre-flight check reads.
3. Per pose, in parallel when ``n_workers > 1``: read the structure, profile the binder, place
   the pose on the receptor of each reference variant (``frame``), write the structure that the
   path-based metrics read, and run the light metrics.
4. Agreement (parent process): peptide RMSD for every pair of poses, statistics inside and
   between groups, medoids, clusters, interface and contact-map overlap, positional spread.
5. Representatives, then the heavy metrics on the representatives (serially, in the parent).
6. Aggregates per group and over all poses, assembly of the result.

The common frame. The poses are placed on the receptor of the first variant of ``reference``;
without a reference the pose with the lowest id that can be read plays that role (the result
says so under ``references``), so the agreement block is always in one frame. The registry
metrics are computed on the transformed structure, written to a temporary mmCIF file in a
per-call directory (``tmp_dir``, never the current directory), one file per pose, and removed
on exit. ``FrameOptions.metrics_frame="original"`` runs them on the pose's own file instead.
A metric of the ``atom_array`` input type receives the transformed array itself.

Which metrics run on which poses. A metric whose ``cost_class`` is ``static`` or ``model`` runs
on every pose. One that is ``structural`` or ``md``, or that declares no class, is heavy: with
``heavy="representative"`` (default) it runs on the representatives only, ``"all"`` runs it on
every pose and ``"skip"`` leaves it out. A pose that was not asked for a metric has no block
for it and no entry in ``reasons``, so the aggregate of a heavy metric counts only the poses
that were asked. The registry name ``prediction`` is served from the pose itself (a
``PredictedPose``); the metrics that read a trajectory or a prediction directory are refused.

The row of a pose
-----------------
``id``, ``kind``, ``content_hash``, ``provenance`` (the pose's own, plus ``backend``, ``name``,
``seed_index``, ``sample`` and ``seed`` for a predicted pose and ``backend`` for a docked pose
that names its program, unless the pose sets them), ``native`` (what the tool that made the pose
reported), ``status``, ``reason``, ``reasons``, ``independent_of_design``, ``profile``,
``frame``, then one block per requested registry metric exactly as the function returned it
(numpy values become lists and numbers), then ``descriptors`` and ``residue_table`` when the
reference asks for them: ``descriptors[variant] = {"sets": {set name: block}, "partners":
{partner name: block}}`` for each reference variant that has residue sets or partners
(``binding_metrics.poses.descriptors``), and with ``residue_table=True`` the interface residue
by residue. ``status`` is ``"failed"`` when the structure could
not be read, a chain is missing or the pose could not be placed on the reference (nothing else
is attempted, ``reason`` says why), ``"partial"`` when a metric raised, was skipped or reported
a ``reason``, else ``"ok"``. ``reasons`` maps the stage (``load``, ``chains``, ``frame``,
``profile``, ``agreement``) or the metric name to a sentence; a metric that raised has no block
and its sentence is ``"<ExceptionType>: <message>"``. A pose that failed counts in ``n_total`` of
the aggregates and never in ``n_ok``. No value is NaN unless a registry metric returned it; the
paths of those NaN are in ``nonfinite_fields``.

Determinism. No random numbers are used. Rows are sorted by id, so the order of ``poses`` does
not change the result, and ``n_workers`` does not change a value. ``timing`` and ``provenance``
describe the run and are not part of ``input_key``, the SHA-256 of the inputs and parameters.
``to_json(include_timing=False)`` is byte-identical for the same inputs.

Errors. ``on_error="record"`` (default) turns a pose that cannot be loaded or placed, and a
metric that raises, into a row with a reason. ``on_error="raise"`` raises ``PoseScoringError``
at the first pose (in id order) whose row would be ``failed`` or whose metric raised, chained to
the original exception when there is one, and does not process the poses after it. A metric
that returns a ``reason`` is not an exception in either mode. A refused pre-flight check
raises ``IncompatibleInputError`` before any pose is scored.

Importing this module imports numpy and the standard library and the pose modules; biotite is
imported when a structure is read.
"""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.process
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np

from binding_metrics.metrics.registry import MetricSpec, get_metric
from binding_metrics.poses.aggregate import (
    DEFAULT_STATISTICS,
    REASON_POLICIES,
    RESERVED_GROUP_KEYS,
    PoseDistances,
    aggregate_poses,
    group_key,
    native_score_direction,
    resolve_statistics,
    select_representatives,
)
from binding_metrics.poses.agreement import (
    CYCLIC_SHIFTS,
    DEFAULT_CLUSTER_CUTOFFS_ANGSTROM,
    DEFAULT_CONTACT_CUTOFF_ANGSTROM,
    DEFAULT_MIN_SHARED_ATOM_FRACTION,
    KINDS,
    RESIDUE_PAIRINGS,
    SYMMETRY_MODES,
    InterfaceResidues,
    PairMatrix,
    complete_linkage_clusters,
    contact_map_jaccard,
    heavy_atom_contacts,
    interface_jaccard,
    medoids,
    pair_statistics,
    pairwise_peptide_rmsd,
    positional_spread,
    positional_spread_by_group,
)
from binding_metrics.poses.descriptors import (
    DEFAULT_CLASH_CUTOFFS_ANGSTROM,
    DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2,
    DEFAULT_MAX_UNMAPPED_FRACTION,
    DEFAULT_PROBE_RADIUS,
    DEFAULT_SHELL_ANGSTROM,
    interface_areas,
    interface_residue_table,
    partner_descriptors,
    residue_set_descriptors,
)
from binding_metrics.poses.frame import (
    FrameOptions,
    FrameResult,
    place_in_reference_frame,
    prepare_reference,
    renumber_to_reference,
    superpose_receptor,
)
from binding_metrics.poses.pose import Pose
from binding_metrics.poses.reference import Reference, ResidueKey, normalize_references

logger = logging.getLogger(__name__)

#: Value of ``schema`` in every result.
SCHEMA = "binding_metrics.pose_set_result"

#: Integer version of the result layout. Adding a key or a statistic does not change it;
#: renaming or removing a key, or changing a meaning or a unit, does. Independent of
#: ``provenance.SCHEMA_VERSION``, which versions the provenance block only.
SCHEMA_VERSION = 1

#: The names of a row that no metric block may take.
RESERVED_ROW_NAMES = (
    "id",
    "kind",
    "content_hash",
    "provenance",
    "native",
    "status",
    "reason",
    "reasons",
    "independent_of_design",
    "profile",
    "frame",
    "descriptors",
    "residue_table",
)

#: The keys of a result, in order.
RESULT_KEYS = (
    "schema",
    "schema_version",
    "package_version",
    "construct",
    "binder_chain",
    "target_chain",
    "input_key",
    "provenance",
    "parameters",
    "references",
    "preflight",
    "poses",
    "agreement",
    "aggregates",
    "representatives",
    "warnings",
    "nonfinite_fields",
    "timing",
)

HEAVY_MODES = ("representative", "all", "skip")
PREFLIGHT_MODES = ("error", "skip", "warn", "off")
ON_ERROR_MODES = ("record", "raise")

#: Name of the reference that stands in when none is given (the first readable pose).
IMPLICIT_REFERENCE_NAME = "first_pose"

#: The keys of ``result["representatives"]`` that are not a group: the rule and the sentences.
REPRESENTATIVE_META_KEYS = ("rule", "reasons", "caveats")

#: The registry name that the layer serves from the pose itself.
SERVED_METRIC = "prediction"

#: Input types of registry metrics that read something other than one structure.
_REFUSED_INPUT_TYPES = {
    "trajectory": "reads a trajectory and a topology file, not one structure",
    "openfold_json": "reads the output directory of an OpenFold3 run",
    "prediction_dir": "reads the output directory of a prediction run",
}
_FILE_INPUT_TYPES = ("static_structure", "md_simulation", "predicted_structure")
_STRUCTURE_SUFFIXES = (".cif", ".mmcif", ".pdb")
_LIGHT_COST_CLASSES = ("static", "model")


class PoseScoringError(RuntimeError):
    """A pose could not be scored and ``on_error="raise"`` was set.

    Attributes:
        pose_id: The id of the first pose that failed, in id order.
        stage: Where it failed: ``load``, ``chains``, ``profile``, ``frame``, or the name of the
            metric that raised.
    """

    def __init__(self, message: str, pose_id: str, stage: str):
        super().__init__(message)
        self.pose_id = pose_id
        self.stage = stage


# ---------------------------------------------------------------------------- options


def _one_of(name: str, value: Any, allowed: Sequence) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _from_mapping(cls, data: Mapping[str, Any], what: str):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {what} options {unknown}; the options are {sorted(known)}")
    return cls(**dict(data))


@dataclass(frozen=True)
class AgreementOptions:
    """The choices and thresholds of the agreement stage (design section 4), with defaults.

    Attributes:
        kind: The atom set of the matrix that feeds the clusters and the medoids: ``"ca"``,
            ``"backbone"`` or ``"heavy"``. All three matrices are reported. The positional spread
            is always of the alpha carbons.
        cluster_cutoffs_angstrom: One clustering per cutoff, complete linkage.
        linkage: ``"complete"``, the only linkage (kept as a field so the result records it).
        symmetry: ``"standard"`` matches the interchangeable side-chain atoms in the labelling
            with the smallest RMSD (``kind="heavy"``), ``"off"`` compares atoms as named.
        cyclic_shift: ``"off"`` or ``"auto"``, whether a head-to-tail ring has a free origin.
        residue_pairing: ``"auto"``, ``"key"`` or ``"position"``: how the binder residues of
            two poses are paired.
        expected_closures: The ring-closure families that the poses should have (``"head_to_tail"``,
            ``"disulfide"``, ...), an empty tuple for a linear binder, or None for the families
            that most poses have (a tie goes to the smaller sorted tuple). Each row's
            ``profile.closures_as_expected`` compares with it.
        exclude_closure_mismatch: Give a pair of poses with different closure families no value.
        contact_cutoff_angstrom: Heavy-atom distance of a contact and of the interface used
            for the Jaccard overlap.
        min_shared_atom_fraction: A pair that shares a smaller fraction of the atoms of the
            larger set has no value.
        shape_rmsd: Also report the shape RMSD (the peptides refitted on the matched atoms).
    """

    kind: str = "ca"
    cluster_cutoffs_angstrom: tuple = DEFAULT_CLUSTER_CUTOFFS_ANGSTROM
    linkage: str = "complete"
    symmetry: str = "standard"
    cyclic_shift: str = "off"
    residue_pairing: str = "auto"
    expected_closures: Optional[tuple] = None
    exclude_closure_mismatch: bool = False
    contact_cutoff_angstrom: float = DEFAULT_CONTACT_CUTOFF_ANGSTROM
    min_shared_atom_fraction: float = DEFAULT_MIN_SHARED_ATOM_FRACTION
    shape_rmsd: bool = True

    def __post_init__(self):
        _one_of("kind", self.kind, KINDS)
        _one_of("linkage", self.linkage, ("complete",))
        _one_of("symmetry", self.symmetry, SYMMETRY_MODES)
        _one_of("cyclic_shift", self.cyclic_shift, CYCLIC_SHIFTS)
        _one_of("residue_pairing", self.residue_pairing, RESIDUE_PAIRINGS)
        cutoffs = self.cluster_cutoffs_angstrom
        cutoffs = (cutoffs,) if isinstance(cutoffs, (int, float)) else tuple(cutoffs)
        if not cutoffs or not all(
            isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and c > 0
            for c in cutoffs
        ):
            raise ValueError(
                "cluster_cutoffs_angstrom must be one or more positive finite numbers, got "
                f"{self.cluster_cutoffs_angstrom!r}"
            )
        object.__setattr__(self, "cluster_cutoffs_angstrom", tuple(float(c) for c in cutoffs))
        if self.expected_closures is not None:
            if isinstance(self.expected_closures, str):
                raise ValueError(
                    "expected_closures must be a list of closure family names (or None), "
                    f"got the string {self.expected_closures!r}"
                )
            families = tuple(sorted(set(self.expected_closures)))
            if not all(isinstance(f, str) and f for f in families):
                raise ValueError(
                    f"expected_closures must hold non-empty names, got {self.expected_closures!r}"
                )
            object.__setattr__(self, "expected_closures", families)
        for name in ("exclude_closure_mismatch", "shape_rmsd"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be True or False, got {getattr(self, name)!r}")
        if not (
            isinstance(self.contact_cutoff_angstrom, (int, float))
            and not isinstance(self.contact_cutoff_angstrom, bool)
            and math.isfinite(self.contact_cutoff_angstrom)
            and self.contact_cutoff_angstrom > 0
        ):
            raise ValueError(
                "contact_cutoff_angstrom must be a positive finite number, got "
                f"{self.contact_cutoff_angstrom!r}"
            )
        if not (
            isinstance(self.min_shared_atom_fraction, (int, float))
            and not isinstance(self.min_shared_atom_fraction, bool)
            and 0.0 < self.min_shared_atom_fraction <= 1.0
        ):
            raise ValueError(
                f"min_shared_atom_fraction must be in (0, 1], got {self.min_shared_atom_fraction!r}"
            )

    def to_dict(self) -> dict:
        """The options as a JSON-ready dictionary (tuples as lists)."""
        data = dataclasses.asdict(self)
        data["cluster_cutoffs_angstrom"] = list(self.cluster_cutoffs_angstrom)
        if self.expected_closures is not None:
            data["expected_closures"] = list(self.expected_closures)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgreementOptions:
        """Inverse of ``to_dict``; a missing key takes its default.

        Raises:
            ValueError: If a key is not an option.
        """
        return _from_mapping(cls, data, "agreement")


@dataclass(frozen=True)
class AggregationOptions:
    """The choices of the aggregation stage (design section 5).

    Attributes:
        min_n_ok: A key with fewer valid values yields None for every statistic and a reason.
        reason_policy: ``"exclude"`` drops every value under a dictionary that has a
            ``reason``; ``"keep"`` keeps them and counts the reasons as caveats.
    """

    min_n_ok: int = 1
    reason_policy: str = "exclude"

    def __post_init__(self):
        if (
            isinstance(self.min_n_ok, bool)
            or not isinstance(self.min_n_ok, (int, np.integer))
            or self.min_n_ok < 1
        ):
            raise ValueError(f"min_n_ok must be an integer of at least 1, got {self.min_n_ok!r}")
        _one_of("reason_policy", self.reason_policy, REASON_POLICIES)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AggregationOptions:
        return _from_mapping(cls, data, "aggregation")


@dataclass(frozen=True)
class RepresentativeOptions:
    """The native score that the ``"confidence"`` rule ranks by (design section 6).

    Attributes:
        confidence_key: The name in ``native``; default the model's own ranking score.
        confidence_direction: ``"max"`` or ``"min"`` (which end is better), or None for the
            known direction of the score (``aggregate.native_score_direction``).
    """

    confidence_key: str = "ranking_score"
    confidence_direction: Optional[str] = None

    def __post_init__(self):
        if not (isinstance(self.confidence_key, str) and self.confidence_key):
            raise ValueError(f"confidence_key must be a score name, got {self.confidence_key!r}")
        if self.confidence_direction is not None:
            _one_of("confidence_direction", self.confidence_direction, ("max", "min"))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepresentativeOptions:
        return _from_mapping(cls, data, "representative")


@dataclass(frozen=True)
class DescriptorOptions:
    """The thresholds of the reference-site descriptors (design section 8), with defaults.

    Attributes:
        interface_threshold_angstrom2: Smallest buried area of a residue that puts it in the
            interface (the threshold of ``compute_interface_metrics``).
        clash_cutoffs_angstrom: Heavy-atom distances for the clash counts of a partner; the
            keys of ``clash_atoms`` and ``clash_pairs`` are these numbers as text (``"2.2"``).
        shell_angstrom: A binder atom within this distance of a partner atom is in its shell.
        max_unmapped_fraction: A residue set of which a larger fraction is missing from the pose
            has no value and a reason.
        probe_radius: Radius of the solvent probe in angstrom.
    """

    interface_threshold_angstrom2: float = DEFAULT_INTERFACE_THRESHOLD_ANGSTROM2
    clash_cutoffs_angstrom: tuple = DEFAULT_CLASH_CUTOFFS_ANGSTROM
    shell_angstrom: float = DEFAULT_SHELL_ANGSTROM
    max_unmapped_fraction: float = DEFAULT_MAX_UNMAPPED_FRACTION
    probe_radius: float = DEFAULT_PROBE_RADIUS

    def __post_init__(self):
        cutoffs = self.clash_cutoffs_angstrom
        cutoffs = (cutoffs,) if isinstance(cutoffs, (int, float)) else tuple(cutoffs)
        if not cutoffs or not all(_is_positive_number(c) for c in cutoffs):
            raise ValueError(
                "clash_cutoffs_angstrom must be one or more positive finite numbers, got "
                f"{self.clash_cutoffs_angstrom!r}"
            )
        object.__setattr__(self, "clash_cutoffs_angstrom", tuple(float(c) for c in cutoffs))
        if not (
            isinstance(self.interface_threshold_angstrom2, (int, float))
            and not isinstance(self.interface_threshold_angstrom2, bool)
            and math.isfinite(self.interface_threshold_angstrom2)
            and self.interface_threshold_angstrom2 >= 0
        ):
            raise ValueError(
                "interface_threshold_angstrom2 must be a finite number of at least 0, got "
                f"{self.interface_threshold_angstrom2!r}"
            )
        for name in ("shell_angstrom", "probe_radius"):
            if not _is_positive_number(getattr(self, name)):
                raise ValueError(
                    f"{name} must be a positive finite number, got {getattr(self, name)!r}"
                )
        if not (
            isinstance(self.max_unmapped_fraction, (int, float))
            and not isinstance(self.max_unmapped_fraction, bool)
            and 0.0 <= self.max_unmapped_fraction <= 1.0
        ):
            raise ValueError(
                f"max_unmapped_fraction must be in [0, 1], got {self.max_unmapped_fraction!r}"
            )

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["clash_cutoffs_angstrom"] = list(self.clash_cutoffs_angstrom)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DescriptorOptions:
        return _from_mapping(cls, data, "descriptors")


def _is_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


_OPTION_GROUPS = {
    "frame": FrameOptions,
    "agreement": AgreementOptions,
    "aggregation": AggregationOptions,
    "representative": RepresentativeOptions,
    "descriptors": DescriptorOptions,
}


@dataclass(frozen=True)
class PoseSetOptions:
    """Every threshold of a pose-set call, in one frozen object with the defaults of the design.

    Attributes:
        frame: ``FrameOptions`` (pairing, fit atoms, refusal limits, ``renumber_receptor``,
            ``metrics_frame``).
        agreement: ``AgreementOptions``.
        aggregation: ``AggregationOptions``.
        representative: ``RepresentativeOptions``.
        descriptors: ``DescriptorOptions``.

    The result echoes the resolved values under ``parameters["options"]``.
    """

    frame: FrameOptions = field(default_factory=FrameOptions)
    agreement: AgreementOptions = field(default_factory=AgreementOptions)
    aggregation: AggregationOptions = field(default_factory=AggregationOptions)
    representative: RepresentativeOptions = field(default_factory=RepresentativeOptions)
    descriptors: DescriptorOptions = field(default_factory=DescriptorOptions)

    def __post_init__(self):
        for name, cls in _OPTION_GROUPS.items():
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = cls.from_dict(value)
            elif not isinstance(value, cls):
                raise TypeError(
                    f"options.{name} must be {cls.__name__} or a mapping, got "
                    f"{type(value).__name__}"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        """``{group: {option: value}}``, JSON-ready."""
        return {name: getattr(self, name).to_dict() for name in _OPTION_GROUPS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PoseSetOptions:
        """Inverse of ``to_dict``; a missing group or key takes its default.

        Raises:
            ValueError: If a group or a key is not an option.
        """
        unknown = sorted(set(data) - set(_OPTION_GROUPS))
        if unknown:
            raise ValueError(
                f"unknown option groups {unknown}; the groups are {sorted(_OPTION_GROUPS)}"
            )
        return cls(**dict(data))


def _coerce_options(options: Any) -> PoseSetOptions:
    if options is None:
        return PoseSetOptions()
    if isinstance(options, PoseSetOptions):
        return options
    if isinstance(options, Mapping):
        return PoseSetOptions.from_dict(options)
    raise TypeError(
        f"options must be PoseSetOptions, a mapping or None, got {type(options).__name__}"
    )


# ---------------------------------------------------------------------------- plain values


def _plain_key(key: Any) -> str:
    return key if isinstance(key, str) else str(key)


def _plain(value: Any) -> Any:
    """``value`` as JSON-ready Python: numpy to numbers and lists, sets sorted, keys as text.

    NaN and infinity stay as they are (``nonfinite_fields`` lists them). An object that is
    none of the known types becomes its text.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, ResidueKey):
        return value.to_list()
    if isinstance(value, Mapping):
        return {_plain_key(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain({f.name: getattr(value, f.name) for f in fields(value)})
    return str(value)


def _describe(exc: BaseException) -> str:
    """The reason text of an exception: ``"ValueError: message"``."""
    return f"{type(exc).__name__}: {exc}"


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every NaN or infinite number; list items are ``name[3]``."""
    if isinstance(value, Mapping):
        found: list[str] = []
        for key, child in value.items():
            found += _nonfinite_paths(child, f"{prefix}.{key}" if prefix else str(key))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, child in enumerate(value):
            found += _nonfinite_paths(child, f"{prefix}[{index}]")
        return found
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    return []


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------- the result


class PoseSetResult(dict):
    """The result of ``score_pose_set``: one nested dictionary, JSON-serialisable.

    It is a ``dict`` with the keys ``RESULT_KEYS`` (the layout is in the module docstring),
    so ``result["poses"]``, ``json.dumps(result)`` and ``copy.deepcopy`` work as they do for
    a dictionary. The methods read it and never change it.

    ``to_dict()`` is a plain-dictionary copy, ``to_json()`` the canonical JSON text,
    ``from_dict()`` the inverse (it rejects a ``schema_version`` it does not know) and
    ``to_flat_rows()`` one flat row of scalar columns per pose, ``summary()`` a Markdown text.
    ``write_pose_report`` in ``binding_metrics.poses.report`` writes JSON, CSV, Markdown and
    HTML.
    """

    @property
    def schema_version(self) -> int:
        """The integer version of the layout (``SCHEMA_VERSION`` when the result was made)."""
        return self["schema_version"]

    @property
    def rows(self) -> list[dict]:
        """The per-pose rows, sorted by id (the list ``self["poses"]``)."""
        return self["poses"]

    @property
    def input_key(self) -> str:
        """SHA-256 of the inputs and parameters: equal for equal inputs, whatever the run."""
        return self["input_key"]

    def row(self, pose_id: str) -> dict:
        """The row of one pose.

        Raises:
            KeyError: If no pose has that id.
        """
        for row in self["poses"]:
            if row["id"] == pose_id:
                return row
        known = [row["id"] for row in self["poses"]]
        raise KeyError(f"no pose {pose_id!r} in the result; the ids are {known}")

    def to_dict(self) -> dict:
        """A deep copy made of plain ``dict``, ``list`` and scalar values."""
        return _plain(dict(self))

    def to_json(self, *, indent: Optional[int] = 2, include_timing: bool = True) -> str:
        """The result as JSON text, keys in the order of the result.

        ``NaN`` and ``Infinity`` tokens are kept for the values that a registry metric
        returned as such (``nonfinite_fields`` lists them); Python's ``json`` reads them back.

        Args:
            indent: Spaces per level; None writes one line.
            include_timing: False leaves the ``timing`` block out, which makes the text
                identical for the same inputs whatever the machine load.
        """
        data = self.to_dict()
        if not include_timing:
            data.pop("timing", None)
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PoseSetResult:
        """Inverse of ``to_dict`` (and of ``json.loads(result.to_json())``).

        Raises:
            ValueError: If ``data`` is not a pose-set result, its ``schema_version`` is not
                the one this code writes, or a key of the layout is missing.
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"a pose-set result is a mapping, got {type(data).__name__}")
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not a pose-set result: schema is {data.get('schema')!r}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"pose-set result schema_version {data.get('schema_version')!r} is not "
                f"supported (this version reads {SCHEMA_VERSION})"
            )
        missing = [key for key in RESULT_KEYS if key not in data and key != "timing"]
        if missing:
            raise ValueError(f"the pose-set result lacks the keys {missing}")
        return cls(_plain(dict(data)))

    def summary(self) -> str:
        """A short Markdown summary: a table per group of poses, the failed poses with their
        reasons, the agreement block, the aggregates, the representatives and the warnings.

        Always a string. ``binding_metrics.poses.report`` has the rest of the writers.
        """
        from binding_metrics.poses.report import summary_markdown

        return summary_markdown(self)

    def to_flat_rows(self, *, max_list_items: int = 16) -> list[dict]:
        """One flat dictionary of scalar columns per pose, rows in id order.

        Keys of nested dictionaries are joined with ``.`` (``interface.delta_sasa``,
        ``native.iptm``, ``frame.receptor_rmsd_angstrom``). A list of scalars, or of lists of
        scalars, with at most ``max_list_items`` items in all is written one column per item,
        ``frame.transform.translation[0]``, ``frame.transform.rotation[1][2]``. A longer list
        and a list that holds dictionaries (``residue_table``, a per-atom array) is left out:
        it stays in the JSON. Every row has every column, in one fixed order (``id``,
        ``kind``, ``status``, ``reason``, ``content_hash``, then the other columns in the
        order they first occur), with None where a pose has no value.

        A key is never changed or escaped, so a key that contains a dot, such as the clash
        cutoff ``"2.2"`` of the partner descriptors, gives the column
        ``...clash_atoms.2.2``. The column names of the numeric values are exactly the key
        paths of ``aggregates``, which join keys the same way.
        """
        if not isinstance(max_list_items, int) or max_list_items < 0:
            raise ValueError(f"max_list_items must be a non-negative integer, got {max_list_items}")
        flat_rows = [_flatten_row(row, max_list_items) for row in self["poses"]]
        leading = ("id", "kind", "status", "reason", "content_hash")
        columns = [name for name in leading if any(name in row for row in flat_rows)]
        seen = set(columns)
        for row in flat_rows:
            for name in row:
                if name not in seen:
                    seen.add(name)
                    columns.append(name)
        return [{name: row.get(name) for name in columns} for row in flat_rows]


_FLAT_SKIP = ("residue_table",)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _list_leaves(value: Any) -> Optional[int]:
    """The number of scalars in a list of scalars (or of such lists), None for another list."""
    if _is_scalar(value):
        return 1
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            count = _list_leaves(item)
            if count is None:
                return None
            total += count
        return total
    return None


def _flatten_value(value: Any, name: str, out: dict, max_list_items: int) -> None:
    if _is_scalar(value):
        out[name] = value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_value(child, f"{name}.{key}" if name else str(key), out, max_list_items)
    elif isinstance(value, (list, tuple)):
        count = _list_leaves(value)
        if count is None or count > max_list_items:
            return
        for index, child in enumerate(value):
            if _is_scalar(child):
                out[f"{name}[{index}]"] = child
            else:
                _flatten_value(child, f"{name}[{index}]", out, max_list_items)


def _flatten_row(row: Mapping[str, Any], max_list_items: int) -> dict:
    out: dict = {}
    for key, value in row.items():
        if key in _FLAT_SKIP:
            continue
        _flatten_value(_plain(value), str(key), out, max_list_items)
    return out


# ---------------------------------------------------------------------------- the arguments


@dataclass(frozen=True)
class _MetricPlan:
    """One requested registry metric and where it runs.

    ``scope`` is ``"all"`` (every pose), ``"representatives"`` or ``"none"``; ``skipped`` is the
    reason the pre-flight check left the metric out, which every pose records.
    """

    name: str
    spec: MetricSpec
    served: bool
    scope: str
    skipped: Optional[str] = None


def _is_heavy(spec: MetricSpec) -> bool:
    return spec.cost_class not in _LIGHT_COST_CLASSES


def _plan_metrics(metrics: Any, *, has_reference: bool, heavy: str) -> list[_MetricPlan]:
    """Check the metric names and decide where each one runs.

    Raises:
        TypeError, ValueError, KeyError: For a name that is not a string, is repeated, is a
            reserved row name, is not in the registry, reads something other than one structure,
            or needs a reference that was not given. The registry is not asked to import a
            function here.
    """
    if isinstance(metrics, str):
        raise ValueError(
            f"metrics must be a sequence of registry names, not the string {metrics!r}; "
            f"write [{metrics!r}]"
        )
    names = list(metrics)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"metric names must be strings, got {name!r}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"metrics repeats {repeated}")
    plans = []
    for name in names:
        if name in RESERVED_ROW_NAMES:
            raise ValueError(
                f"the metric name {name!r} is a reserved row name ({list(RESERVED_ROW_NAMES)}): "
                "its block would overwrite a field of the row"
            )
        spec = get_metric(name)
        served = name == SERVED_METRIC
        if not served and spec.input_type in _REFUSED_INPUT_TYPES:
            raise ValueError(
                f"metric {name!r} cannot score a pose: it {_REFUSED_INPUT_TYPES[spec.input_type]}"
            )
        if not served and spec.secondary_path_arg and not has_reference:
            raise ValueError(
                f"metric {name!r} compares the pose with a second structure: give reference="
            )
        scope = "all"
        if not served and _is_heavy(spec):
            scope = {"representative": "representatives", "all": "all", "skip": "none"}[heavy]
        plans.append(_MetricPlan(name, spec, served, scope))
    return plans


_SERVED_RESERVED_KWARGS = ("record", "binder_chain", "receptor_chain", "target_chain")


def _reserved_kwargs(plan: _MetricPlan) -> set[str]:
    if plan.served:
        return set(_SERVED_RESERVED_KWARGS)
    spec = plan.spec
    names = {
        spec.path_arg,
        spec.secondary_path_arg,
        spec.chain_arg,
        spec.peptide_chain_arg,
        spec.receptor_chain_arg,
        spec.binder_chain_arg,
        spec.target_chain_arg,
    }
    return {name for name in names if name}


def _check_metric_kwargs(metric_kwargs: Any, plans: Sequence[_MetricPlan]) -> dict[str, dict]:
    if metric_kwargs is None:
        return {}
    if not isinstance(metric_kwargs, Mapping):
        raise TypeError(
            f"metric_kwargs must be a mapping of metric name to keyword arguments, got "
            f"{type(metric_kwargs).__name__}"
        )
    by_name = {plan.name: plan for plan in plans}
    checked: dict[str, dict] = {}
    for name, kwargs in metric_kwargs.items():
        if name not in by_name:
            raise ValueError(
                f"metric_kwargs has an entry for {name!r}, which is not one of the requested "
                f"metrics {list(by_name)}"
            )
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"metric_kwargs[{name!r}] must be a mapping, got {kwargs!r}")
        clash = sorted(set(kwargs) & _reserved_kwargs(by_name[name]))
        if clash:
            raise ValueError(
                f"metric_kwargs[{name!r}] sets {clash}, which score_pose_set passes itself "
                "(the structure, the second structure and the chains)"
            )
        checked[name] = dict(kwargs)
    return checked


def _check_group_by(group_by: Any) -> tuple[str, ...]:
    names = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"group_by names must be non-empty strings, got {name!r}")
    return names


def _check_chain_id(role: str, value: Any) -> None:
    if value is not None and not (isinstance(value, str) and value.strip()):
        raise ValueError(f"{role} must be a chain ID string or None, got {value!r}")


def _resolve_poses(poses: Any) -> list[tuple[str, Pose]]:
    """The poses as ``[(id, pose)]`` sorted by id.

    Raises:
        TypeError: If ``poses`` is not a sequence of ``Pose``.
        ValueError: If it is empty, a pose has no usable id, or an id is repeated.
    """
    if isinstance(poses, (str, bytes, os.PathLike, Pose)):
        raise TypeError(
            "poses must be a sequence of Pose objects (DockedPose, ExperimentalPose, "
            f"PredictedPose), got {type(poses).__name__}"
        )
    try:
        items = list(poses)
    except TypeError:
        raise TypeError(
            f"poses must be a sequence of Pose objects, got {type(poses).__name__}"
        ) from None
    if not items:
        raise ValueError("poses is empty: give at least one Pose")
    resolved = []
    for position, pose in enumerate(items):
        if not isinstance(pose, Pose):
            raise TypeError(
                f"poses[{position}] is a {type(pose).__name__}, not a Pose; wrap a structure in "
                "DockedPose, ExperimentalPose or PredictedPose"
            )
        try:
            resolved.append((pose.id, pose))
        except (ValueError, OSError) as exc:
            raise ValueError(f"poses[{position}] has no usable id: {_describe(exc)}") from exc
    counts = Counter(pose_id for pose_id, _ in resolved)
    repeated = sorted(pose_id for pose_id, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(f"pose ids must be unique; repeated: {repeated}")
    return sorted(resolved, key=lambda item: item[0])


def _check_residue_maps(residue_maps: Any, ids: Sequence[str]) -> dict[str, Mapping]:
    if residue_maps is None:
        return {}
    if not isinstance(residue_maps, Mapping):
        raise TypeError("residue_maps must be a mapping of pose id to residue map")
    unknown = sorted(set(residue_maps) - set(ids))
    if unknown:
        raise ValueError(f"residue_maps has entries for {unknown}, which are not pose ids")
    checked = {}
    for pose_id, mapping in residue_maps.items():
        if not isinstance(mapping, Mapping):
            raise TypeError(f"residue_maps[{pose_id!r}] must be a mapping, got {mapping!r}")
        checked[pose_id] = {ResidueKey.coerce(k): ResidueKey.coerce(v) for k, v in mapping.items()}
    return checked


# ---------------------------------------------------------------------------- one pose


class _NotApplicableError(Exception):
    """A requested block that the pose cannot have (not a failure of the pose)."""


@dataclass
class _Outcome:
    """What the per-pose stage hands back: the row, the placed atoms and how it failed."""

    pose_id: str
    index: int
    row: dict
    metrics_atoms: Any = None
    failure: Optional[tuple[str, str]] = None
    exception: Optional[BaseException] = None
    seconds: float = 0.0


@dataclass(frozen=True)
class _Context:
    """What every worker needs, built once in the parent and sent to each worker once.

    Everything in it pickles, because a worker may be started by ``spawn``.
    """

    binder_chain: str
    target_chain: str
    options: PoseSetOptions
    variants: dict
    plans: tuple
    metric_kwargs: dict
    tmp_root: Optional[str]
    reference_path: Optional[str]
    residue_maps: dict
    include_transform: bool
    raise_mode: bool
    residue_table: bool = False


def _provenance_of(pose: Pose) -> dict:
    """The row's provenance: the pose's own facts, plus the group keys of its kind.

    A predicted pose gets ``backend`` (its model), ``name``, ``seed_index``, ``sample`` and,
    when the adapter recorded it, ``seed``; a docked pose gets ``backend`` from its program.
    A key that the pose's own provenance sets is not replaced.
    """
    derived: dict[str, Any] = {}
    if pose.kind == "predicted":
        derived = {
            "backend": getattr(pose, "model", None),
            "name": getattr(pose, "name", None),
            "seed_index": getattr(pose, "seed_index", None),
            "sample": getattr(pose, "sample", None),
        }
        seed = (getattr(pose, "extras", None) or {}).get("seed_value")
        if seed is not None and str(seed).isdecimal():
            derived["seed"] = int(seed)
    elif pose.kind == "docked" and getattr(pose, "program", None):
        derived = {"backend": pose.program}
    merged = {key: value for key, value in derived.items() if value is not None}
    merged.update(pose.provenance)
    return _plain(merged)


def _new_row(pose: Pose) -> dict:
    try:
        digest = pose.content_hash()
    except (ValueError, OSError):
        digest = None  # the load stage reports why
    return {
        "id": pose.id,
        "kind": pose.kind,
        "content_hash": digest,
        "provenance": _provenance_of(pose),
        "native": {},
        "status": "ok",
        "reason": None,
        "reasons": {},
        "independent_of_design": None,
        "profile": None,
        "frame": None,
    }


def _refresh_status(row: dict) -> None:
    """``partial`` when any stage or metric left a reason, else ``ok``; ``failed`` stays."""
    if row["status"] != "failed":
        row["status"] = "partial" if row["reasons"] else "ok"


def _fail(outcome: _Outcome, stage: str, text: str, exc: Optional[BaseException] = None) -> None:
    row = outcome.row
    row["status"] = "failed"
    row["reason"] = text
    row["reasons"][stage] = text
    if outcome.failure is None:
        outcome.failure, outcome.exception = (stage, text), exc


def _note(outcome: _Outcome, stage: str, text: str, exc: Optional[BaseException] = None) -> None:
    """Record a reason for a stage or metric without failing the row."""
    outcome.row["reasons"][stage] = text
    if outcome.failure is None and exc is not None:
        outcome.failure, outcome.exception = (stage, text), exc


def _profile_block(profile) -> dict:
    data = profile.to_dict()
    block = {
        key: data[key]
        for key in (
            "n_binder_residues",
            "binder_type",
            "closures",
            "closure_bonds",
            "residue_classes",
        )
    }
    # the names of the canonical residues are the sequence; only the others carry information
    block["residue_names"] = {
        kind: names for kind, names in data["residue_names"].items() if kind != "canonical"
    }
    block["notes"] = data["notes"]
    block["closures_as_expected"] = None
    block["differs_from_first"] = []
    return block


def _frame_block(frames: Mapping[str, Any], include_transform: bool) -> dict:
    """The row's ``frame``: the first variant's block, plus all variants' when there are several."""
    names = list(frames)
    first = frames[names[0]]
    block = _frame_dict(first, include_transform)
    block["variant"] = names[0]
    if len(names) > 1:
        block["variants"] = {name: _frame_dict(frames[name], include_transform) for name in names}
    return block


def _frame_dict(frame: Any, include_transform: bool) -> dict:
    if isinstance(frame, FrameResult):
        return frame.to_dict(include_transform=include_transform)
    return dict(frame)  # a variant whose frame call raised: {"status": "failed", "reason": ...}


def _own_file(pose: Pose) -> Optional[str]:
    """The pose's own structure file when the metrics can read it as it is, else None.

    The file must exist, carry the user's chain IDs (no ``chain_map``) and have a suffix that
    ``load_structure`` reads as it is (a compressed file is not).
    """
    path = pose.structure_path
    if path is None or pose.chain_map or Path(path).suffix.lower() not in _STRUCTURE_SUFFIXES:
        return None
    return str(path)


def _write_structure(atoms: Any, path: Path) -> None:
    """Write an ``AtomArray`` as mmCIF with biotite, in UTF-8.

    The array gets a zero ``charge`` annotation when it has none, so that the loader of the
    interface metrics finds the formal-charge column and does not warn about it. biotite sets
    that charge to 0 anyway.
    """
    import io

    from binding_metrics.metrics._common import import_biotite

    _, pdbx, _ = import_biotite("writing the structures that the metrics read")
    if "charge" not in atoms.get_annotation_categories():
        atoms = atoms.copy()
        atoms.add_annotation("charge", dtype=int)
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms)
    buffer = io.StringIO()
    cif.write(buffer)
    Path(path).write_text(buffer.getvalue(), encoding="utf-8")


class _PoseFiles:
    """The structure file that the path-based metrics read for one pose, written once."""

    def __init__(self, ctx: _Context, index: int, pose: Pose, metrics_atoms: Any):
        self.ctx, self.index, self.pose, self.metrics_atoms = ctx, index, pose, metrics_atoms
        self._path: Optional[str] = None

    def path(self) -> str:
        if self._path is None:
            self._path = self._resolve()
        return self._path

    def _resolve(self) -> str:
        if self.ctx.options.frame.metrics_frame == "original":
            own = _own_file(self.pose)
            if own is not None:
                return own
            atoms = self.pose.atoms()
        else:
            atoms = self.metrics_atoms
        if self.ctx.tmp_root is None:  # pragma: no cover - the parent creates it when needed
            raise RuntimeError("no temporary directory was prepared for the structure files")
        target = Path(self.ctx.tmp_root) / f"pose_{self.index:04d}.cif"
        if not target.exists():
            _write_structure(atoms, target)
        return str(target)

    def atoms(self):
        """A copy of the array for a metric that takes an ``AtomArray`` (it may modify it)."""
        if self.ctx.options.frame.metrics_frame == "original":
            return self.pose.atoms().copy()
        return self.metrics_atoms.copy()


def _chain_kwargs(spec: MetricSpec, binder: str, target: str) -> dict[str, str]:
    """The chain arguments of a registry call, as the registry module prescribes."""
    if spec.chain_mode == "single":
        arg = spec.binder_chain_arg or spec.chain_arg
        return {arg: binder} if arg else {}
    if spec.chain_mode in ("interface", "interface_2paths"):
        kwargs = {}
        binder_arg = spec.binder_chain_arg or spec.peptide_chain_arg
        target_arg = spec.target_chain_arg or spec.receptor_chain_arg
        if binder_arg:
            kwargs[binder_arg] = binder
        if target_arg:
            kwargs[target_arg] = target
        return kwargs
    return {}


def _serve_prediction(pose: Pose, ctx: _Context, extra: Mapping[str, Any]):
    from binding_metrics.metrics.prediction import summarize_prediction
    from binding_metrics.predictors.record import PredictedPose

    if not isinstance(pose, PredictedPose):
        raise _NotApplicableError(
            "the 'prediction' block is built from the confidence outputs of a predicted pose; "
            f"this pose is {pose.kind}"
        )
    return summarize_prediction(
        pose,
        binder_chain=ctx.binder_chain,
        receptor_chain=ctx.target_chain,
        caller="score_pose_set",
        **extra,
    )


def _evaluate_metric(plan: _MetricPlan, pose: Pose, ctx: _Context, files: _PoseFiles):
    extra = ctx.metric_kwargs.get(plan.name, {})
    if plan.served:
        return _serve_prediction(pose, ctx, extra)
    spec = plan.spec
    kwargs: dict[str, Any] = {}
    if spec.input_type == "atom_array":
        kwargs[spec.path_arg] = files.atoms()
    else:
        kwargs[spec.path_arg] = files.path()
    if spec.input_type == "predicted_structure" and "plddt_per_atom" not in extra:
        plddt = getattr(pose, "plddt_per_atom", None)
        if plddt is None:
            raise _NotApplicableError(
                f"{plan.name} needs the per-atom pLDDT of a predicted pose; this pose has none"
            )
        kwargs["plddt_per_atom"] = plddt
    if spec.secondary_path_arg:
        kwargs[spec.secondary_path_arg] = ctx.reference_path
    kwargs.update(_chain_kwargs(spec, ctx.binder_chain, ctx.target_chain))
    kwargs.update(extra)
    return spec.call(**kwargs)


def _run_plans(
    outcome: _Outcome, plans: Sequence[_MetricPlan], pose: Pose, ctx: _Context, files: _PoseFiles
) -> None:
    """Run the metrics of ``plans`` on one pose and write their blocks and reasons to its row."""
    row = outcome.row
    for plan in plans:
        name = plan.name
        try:
            block = _evaluate_metric(plan, pose, ctx, files)
        except _NotApplicableError as note:
            row["reasons"][name] = str(note)
            continue
        except Exception as exc:  # noqa: BLE001 - a metric may raise anything; the sentence goes into the row's reasons and the failure is logged
            text = _describe(exc)
            logger.warning("metric %s failed on pose %s: %s", name, pose.id, text)
            _note(outcome, name, text, exc)
            if ctx.raise_mode:
                return
            continue
        plain = _plain(block)
        row[name] = plain
        reason = plain.get("reason") if isinstance(plain, dict) else None
        if isinstance(reason, str) and reason:
            row["reasons"][name] = reason


def _pose_stage(pose: Pose, index: int, ctx: _Context) -> _Outcome:
    """Stage 3 for one pose: read, profile, place on the references, run the light metrics."""
    started = time.perf_counter()
    outcome = _Outcome(pose.id, index, _new_row(pose))
    try:
        _score_pose(pose, ctx, outcome)
    finally:
        outcome.seconds = time.perf_counter() - started
    _refresh_status(outcome.row)
    return outcome


def _score_pose(pose: Pose, ctx: _Context, outcome: _Outcome) -> None:
    from binding_metrics.capabilities import profile_input

    row = outcome.row
    try:
        atoms = pose.atoms()
    except Exception as exc:  # noqa: BLE001 - a structure file can fail in any parser; the row records the reason
        _fail(outcome, "load", _describe(exc), exc)
        return
    try:
        row["native"] = _plain(pose.native_scores())
    except Exception as exc:  # noqa: BLE001 - a custom pose may raise anything; the row records the reason
        _note(outcome, "native", _describe(exc), exc)

    chains = {str(chain) for chain in np.unique(atoms.chain_id)}
    absent = [
        f"{role} chain '{chain}'"
        for role, chain in (("binder", ctx.binder_chain), ("target", ctx.target_chain))
        if chain not in chains
    ]
    if absent:
        _fail(
            outcome,
            "chains",
            f"{' and '.join(absent)} not in the structure; it has {sorted(chains)}",
        )
        return
    try:
        row["profile"] = _profile_block(profile_input(atoms, ctx.binder_chain, ctx.target_chain))
    except Exception as exc:  # noqa: BLE001 - profiling reads the whole structure; the row records the reason
        _note(outcome, "profile", _describe(exc), exc)
    if ctx.raise_mode and outcome.failure is not None:
        return

    frames: dict[str, Any] = {}
    for position, (name, reference) in enumerate(ctx.variants.items()):
        stage = "frame" if position == 0 else f"frame:{name}"
        try:
            frames[name] = superpose_receptor(
                pose,
                reference,
                receptor_chain=ctx.target_chain,
                residue_map=ctx.residue_maps.get(pose.id),
                options=ctx.options.frame,
            )
        except Exception as exc:  # noqa: BLE001 - a pose-specific input problem; the row records the reason
            text = _describe(exc)
            frames[name] = {"status": "failed", "reason": text}
            if position == 0:
                row["frame"] = _frame_block(frames, ctx.include_transform)
                _fail(outcome, stage, text, exc)
                return
            _note(outcome, stage, text, exc)
            continue
        if not frames[name].ok:
            if position == 0:
                row["frame"] = _frame_block(frames, ctx.include_transform)
                _fail(outcome, stage, frames[name].reason or "the pose could not be placed")
                return
            _note(outcome, stage, frames[name].reason or "the pose could not be placed")
    row["frame"] = _frame_block(frames, ctx.include_transform)

    first_name = next(iter(ctx.variants))
    first_frame = frames[first_name]
    first_reference = ctx.variants[first_name]
    try:
        placed_source = (
            renumber_to_reference(
                atoms, first_frame, first_reference, receptor_chain=ctx.target_chain
            )
            if ctx.options.frame.renumber_receptor
            else atoms
        )
        outcome.metrics_atoms = place_in_reference_frame(placed_source, first_frame)
    except Exception as exc:  # noqa: BLE001 - placing reads the pose's atoms; the row records the reason
        _fail(outcome, "frame", _describe(exc), exc)
        return

    light = [plan for plan in ctx.plans if plan.scope == "all" and plan.skipped is None]
    if light:
        files = _PoseFiles(ctx, outcome.index, pose, outcome.metrics_atoms)
        _run_plans(outcome, light, pose, ctx, files)
    if ctx.raise_mode and outcome.failure is not None:
        return
    _run_descriptors(outcome, atoms, frames, ctx)


def _run_descriptors(outcome: _Outcome, atoms: Any, frames: Mapping[str, Any], ctx: _Context):
    """The reference-site descriptors of one pose, per reference variant (design section 8).

    Each variant that has residue sets or partners gets ``descriptors[variant] = {"sets": ...,
    "partners": ...}``, measured on the pose placed in the frame of that variant; variants that
    share a receptor (equal ``frame_key``) share the placed atoms and the buried areas.
    ``residue_table`` is the per-residue interface table of the first variant. The descriptor
    functions take the atoms with the numbering the frame stage read, so the renumbered copy of
    the metrics is not used here. A block that has no value keeps its ``reason`` and is
    recorded in ``reasons`` as ``descriptors.<variant>.sets.<set>`` (or ``.partners.<partner>``).
    """
    options = ctx.options.descriptors
    first_name = next(iter(ctx.variants))
    wanted = [
        name
        for name, reference in ctx.variants.items()
        if reference.residue_sets or reference.partners
    ]
    if not wanted and not ctx.residue_table:
        return
    row = outcome.row
    measured: dict[tuple, tuple] = {}

    def placed_and_areas(name: str):
        frame, reference = frames[name], ctx.variants[name]
        key = (reference.frame_key(ctx.options.frame.frame_residues), frame.ok)
        if key not in measured:
            placed = place_in_reference_frame(atoms, frame) if frame.ok else atoms
            try:
                areas = (
                    interface_areas(
                        placed,
                        ctx.binder_chain,
                        ctx.target_chain,
                        probe_radius=options.probe_radius,
                    )
                    if frame.ok
                    else None
                )
            except ValueError:
                areas = None  # a chain with no heavy polymer atom: each function gives the reason
            measured[key] = (placed, areas)
        return measured[key]

    common = dict(binder_chain=ctx.binder_chain, target_chain=ctx.target_chain)
    blocks: dict[str, Any] = {}
    for name in wanted:
        frame, reference = frames.get(name), ctx.variants[name]
        if not isinstance(frame, FrameResult):  # the frame call raised: the row says so already
            _note(outcome, f"descriptors.{name}", f"no frame for this variant: {frame['reason']}")
            continue
        try:
            placed, areas = placed_and_areas(name)
            sets = (
                residue_set_descriptors(
                    placed,
                    frame,
                    reference,
                    interface_threshold_angstrom2=options.interface_threshold_angstrom2,
                    max_unmapped_fraction=options.max_unmapped_fraction,
                    probe_radius=options.probe_radius,
                    areas=areas,
                    **common,
                )
                if reference.residue_sets
                else {}
            )
            partners = (
                partner_descriptors(
                    placed,
                    frame,
                    reference,
                    clash_cutoffs_angstrom=options.clash_cutoffs_angstrom,
                    shell_angstrom=options.shell_angstrom,
                    probe_radius=options.probe_radius,
                    areas=areas,
                    **common,
                )
                if reference.partners
                else {}
            )
        except Exception as exc:  # noqa: BLE001 - the descriptors read the whole pose; the row records the reason
            _note(outcome, f"descriptors.{name}", _describe(exc), exc)
            continue
        blocks[name] = {"sets": _plain(sets), "partners": _plain(partners)}
        for kind, group in (("sets", sets), ("partners", partners)):
            for member, block in group.items():
                if block.get("reason"):
                    row["reasons"][f"descriptors.{name}.{kind}.{member}"] = block["reason"]
    if blocks:
        row["descriptors"] = blocks
    if not ctx.residue_table:
        return
    frame = frames.get(first_name)
    try:
        placed, areas = placed_and_areas(first_name)
        table = interface_residue_table(
            placed,
            frame,
            ctx.variants[first_name],
            interface_threshold_angstrom2=options.interface_threshold_angstrom2,
            probe_radius=options.probe_radius,
            areas=areas,
            **common,
        )
    except Exception as exc:  # noqa: BLE001 - the table reads the whole pose; the row records the reason
        _note(outcome, "residue_table", _describe(exc), exc)
        return
    row["residue_table"] = _plain(table)


# ---------------------------------------------------------------------------- running poses

_WORKER_CONTEXT: Optional[_Context] = None


def _init_worker(ctx: _Context) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = ctx


def _pool_task(index: int, pose: Pose) -> _Outcome:
    outcome = _pose_stage(pose, index, _WORKER_CONTEXT)
    outcome.exception = None  # the sentence in ``failure`` travels; the object may not pickle
    return outcome


def _pool_context():
    """The multiprocessing context of the worker pool: the start method of the platform.

    ``fork`` on Linux up to Python 3.13, ``spawn`` on macOS and Windows, ``forkserver`` from
    Python 3.14 on Linux. A test replaces this function to force one start method.
    """
    return multiprocessing.get_context()


class _PoolUnavailableError(Exception):
    """The worker processes could not be started; the poses are scored in this process."""


def _raise_failure(outcome: _Outcome) -> None:
    stage, text = outcome.failure
    message = f"pose {outcome.pose_id!r} failed at stage {stage!r}: {text}"
    error = PoseScoringError(message, outcome.pose_id, stage)
    if outcome.exception is not None:
        raise error from outcome.exception
    raise error


def _score_serially(
    items: Sequence[tuple[int, Pose]], ctx: _Context, outcomes: list[_Outcome]
) -> None:
    for index, pose in items:
        outcome = _pose_stage(pose, index, ctx)
        outcomes.append(outcome)
        if ctx.raise_mode and outcome.failure is not None:
            _raise_failure(outcome)


def _score_in_pool(
    items: Sequence[tuple[int, Pose]], ctx: _Context, n_workers: int, outcomes: list[_Outcome]
) -> None:
    """Score ``items`` in worker processes, appending the outcomes to ``outcomes`` in order.

    Raises:
        _PoolUnavailableError: If the pool or its first worker could not be started: the
            executor cannot be made, a process cannot be created, or the pool breaks before any
            pose came back. A pool that breaks after poses came back is a crash of a worker, not
            a start problem, and its ``BrokenProcessPool`` propagates.
        PoseScoringError: With ``on_error="raise"``, at the first failing pose.
    """
    broken = concurrent.futures.process.BrokenProcessPool
    try:
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(n_workers, len(items)),
            mp_context=_pool_context(),
            initializer=_init_worker,
            initargs=(ctx,),
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        raise _PoolUnavailableError(_describe(exc)) from exc
    with pool:
        futures: list = []
        try:
            for index, pose in items:  # a process is created at the first submit
                futures.append(pool.submit(_pool_task, index, pose))
        except (OSError, broken) as exc:
            for future in futures:
                future.cancel()
            raise _PoolUnavailableError(_describe(exc)) from exc
        try:
            for future in futures:
                outcome = future.result()
                outcomes.append(outcome)
                if ctx.raise_mode and outcome.failure is not None:
                    _raise_failure(outcome)
        except broken as exc:
            for future in futures:
                future.cancel()
            if outcomes:
                raise
            raise _PoolUnavailableError(_describe(exc)) from exc
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _score_poses(
    items: Sequence[tuple[int, Pose]], ctx: _Context, n_workers: int
) -> list[_Outcome]:
    """Stage 3 over all poses, in id order; a pool of processes when ``n_workers > 1``.

    The workers are started with the start method of the platform (``fork``, ``spawn`` or
    ``forkserver``); the context and the poses are pickled to them, and the outcomes (the rows
    and the placed atoms) are pickled back. With ``on_error="raise"`` the first failing pose (in
    id order) raises and the poses after it are dropped. When no worker can be started, a
    warning is logged and the poses that have no outcome yet are scored one after the other.
    """
    outcomes: list[_Outcome] = []
    if n_workers > 1 and len(items) > 1:
        try:
            _score_in_pool(items, ctx, n_workers, outcomes)
            return outcomes
        except _PoolUnavailableError as error:
            logger.warning(
                "n_workers=%d: the worker processes could not be used (%s); scoring the poses "
                "one after the other",
                n_workers,
                error,
            )
    _score_serially(items[len(outcomes) :], ctx, outcomes)
    return outcomes


# ---------------------------------------------------------------------------- first pose


@dataclass
class _Scan:
    """The first pose that can be read, and what it gives: chains, profile, structure."""

    pose: Optional[Pose]
    atoms: Any
    binder_chain: Optional[str]
    target_chain: Optional[str]
    profile: Any
    failures: dict[str, tuple[str, str]]
    detected: dict[str, bool]


def _scan_first_pose(
    poses: Sequence[tuple[str, Pose]],
    binder_chain: Optional[str],
    target_chain: Optional[str],
    raise_mode: bool,
) -> _Scan:
    """Read the poses in id order until one gives both chains and a profile.

    The chains that were not given are detected from that pose (the smaller protein chain is
    the binder, the larger the target). Under ``on_error="raise"`` a pose that fails before
    it raises.
    """
    from binding_metrics.capabilities import profile_input
    from binding_metrics.metrics.interface import detect_interface_chains

    failures: dict[str, tuple[str, str]] = {}

    def failed(pose_id: str, stage: str, text: str, exc: Optional[BaseException] = None):
        failures[pose_id] = (stage, text)
        if raise_mode:
            error = PoseScoringError(
                f"pose {pose_id!r} failed at stage {stage!r}: {text}", pose_id, stage
            )
            raise error from exc

    for pose_id, pose in poses:
        try:
            atoms = pose.atoms()
        except Exception as exc:  # noqa: BLE001 - a structure file can fail in any parser; recorded in the row later
            failed(pose_id, "load", _describe(exc), exc)
            continue
        binder, target = binder_chain, target_chain
        if binder is None or target is None:
            found_binder, found_target = detect_interface_chains(atoms, design_chain=binder)
            binder, target = binder or found_binder, target or found_target
        if binder is None or target is None or binder == target:
            failed(
                pose_id,
                "chains",
                f"could not tell the binder and the target chain apart (binder {binder!r}, "
                f"target {target!r}); pass binder_chain and target_chain",
            )
            continue
        chains = {str(chain) for chain in np.unique(atoms.chain_id)}
        absent = [c for c in (binder, target) if c not in chains]
        if absent:
            failed(
                pose_id, "chains", f"chains {absent} not in the structure; it has {sorted(chains)}"
            )
            continue
        try:
            profile = profile_input(atoms, binder, target)
        except Exception as exc:  # noqa: BLE001 - profiling reads the whole structure; recorded in the row later
            failed(pose_id, "profile", _describe(exc), exc)
            continue
        detected = {"binder_chain": binder_chain is None, "target_chain": target_chain is None}
        return _Scan(pose, atoms, binder, target, profile, failures, detected)
    return _Scan(None, None, binder_chain, target_chain, None, failures, {})


def _preflight(
    profile: Any,
    plans: Sequence[_MetricPlan],
    mode: str,
    *,
    has_reference: bool,
    first_kind: str,
) -> tuple[Optional[dict], list[_MetricPlan], list[str]]:
    """The pre-flight check of the requested metrics against the first pose's profile.

    Returns the report as a dictionary (None when ``mode`` is ``"off"``), the plans with the
    metrics that the check left out (policy ``"skip"``) marked as skipped, and the warnings of
    the report. A refused input raises ``IncompatibleInputError``.
    """
    from binding_metrics.capabilities import preflight

    if mode == "off":
        return None, list(plans), []
    provided = set()
    if has_reference:
        provided.add("reference_structure")
    if first_kind == "predicted":
        provided.add("predicted_structure")
    checked = [plan.name for plan in plans if plan.scope != "none"]
    report = preflight(profile, checked, provided=provided, policy=mode)
    kept = set(report.metrics_to_run)
    updated = []
    for plan in plans:
        if plan.scope != "none" and plan.name not in kept:
            facts = [
                v.fact for v in report.violations if v.kind == "metric" and v.name == plan.name
            ]
            reason = "left out by the pre-flight check: " + "; ".join(facts)
            plan = dataclasses.replace(plan, skipped=reason)
        updated.append(plan)
    warnings = [f"preflight: {text}" for text in report.warnings]
    return report.to_dict(), updated, warnings


# ---------------------------------------------------------------------------- agreement


def _matrix_entry(matrix: PairMatrix, groups: Optional[Mapping[str, str]], group_by) -> dict:
    """A matrix as a dictionary with its statistics inside and between the groups."""
    statistics = pair_statistics(matrix, groups)
    if statistics["between_reason"] is not None and tuple(group_by) == ("backend",):
        statistics["between_reason"] = "only one back-end"
    return {"matrix": matrix.to_dict(), "statistics": statistics}


def _majority_closures(families: Mapping[str, frozenset]) -> frozenset:
    """The closure families that most poses have; a tie goes to the smaller sorted tuple."""
    tally = Counter(families.values())
    ranked = sorted(tally.items(), key=lambda item: (-item[1], tuple(sorted(item[0]))))
    return ranked[0][0] if ranked else frozenset()


def _agreement_stage(
    options: AgreementOptions,
    binder_chain: str,
    target_chain: str,
    group_by: tuple[str, ...],
    outcomes: Sequence[_Outcome],
) -> tuple[dict, Optional[PoseDistances], dict[str, str]]:
    """Stage 4: how much the placed poses agree with each other.

    Returns the ``agreement`` block, the distance matrix of ``options.kind`` over every pose
    that was placed (for the medoid representatives), and ``{pose id: reason}`` for the poses
    whose binder could not be read (they stay in the distance matrix with undefined pairs).
    """
    rows = {outcome.pose_id: outcome.row for outcome in outcomes}
    excluded = {
        pose_id: row["reason"] or "failed"
        for pose_id, row in rows.items()
        if row["status"] == "failed"
    }
    atoms = {
        outcome.pose_id: outcome.metrics_atoms
        for outcome in outcomes
        if outcome.metrics_atoms is not None and rows[outcome.pose_id]["status"] != "failed"
    }
    if not atoms:
        block = {
            "status": "skipped",
            "reason": "no pose could be placed on the reference",
            "kind": options.kind,
            "pose_ids": [],
            "excluded": excluded,
        }
        return block, None, {}

    families = {
        pose_id: frozenset(f for f in row["profile"]["closures"] if f != "none")
        for pose_id, row in rows.items()
        if pose_id in atoms and row["profile"] is not None
    }
    expected = (
        frozenset(options.expected_closures)
        if options.expected_closures is not None
        else _majority_closures(families)
    )
    for pose_id, found in families.items():
        rows[pose_id]["profile"]["closures_as_expected"] = found == expected

    common = dict(
        symmetry=options.symmetry,
        residue_pairing=options.residue_pairing,
        cyclic_shift=options.cyclic_shift,
        min_shared_atom_fraction=options.min_shared_atom_fraction,
        closure_families=families,
        exclude_closure_mismatch=options.exclude_closure_mismatch,
    )
    pose_rmsd = {
        kind: pairwise_peptide_rmsd(atoms, binder_chain, kind=kind, refit=False, **common)
        for kind in KINDS
    }
    shape_rmsd = (
        {
            kind: pairwise_peptide_rmsd(atoms, binder_chain, kind=kind, refit=True, **common)
            for kind in KINDS
        }
        if options.shape_rmsd
        else None
    )
    chosen = pose_rmsd[options.kind]
    unreadable = dict(chosen.pose_reasons)
    for pose_id, reason in unreadable.items():
        excluded[pose_id] = reason
    ok_ids = [pose_id for pose_id in chosen.pose_ids if pose_id not in unreadable]
    groups = (
        {pose_id: group_key(rows[pose_id], group_by) for pose_id in ok_ids} if group_by else None
    )
    distance = chosen.subset(ok_ids)
    ok_atoms = {pose_id: atoms[pose_id] for pose_id in ok_ids}

    block: dict[str, Any] = {
        "status": "ok",
        "reason": None,
        "kind": options.kind,
        "linkage": options.linkage,
        "pose_ids": ok_ids,
        "groups": groups,
        "excluded": excluded,
        "expected_closures": sorted(expected),
        "rmsd": {
            kind: _matrix_entry(pose_rmsd[kind].subset(ok_ids), groups, group_by) for kind in KINDS
        },
        "shape_rmsd": (
            None
            if shape_rmsd is None
            else {
                kind: _matrix_entry(shape_rmsd[kind].subset(ok_ids), groups, group_by)
                for kind in KINDS
            }
        ),
        "unknown_residues": list(pose_rmsd["heavy"].unknown_residues),
        "closure_mismatch_pairs": sorted(
            [list(pair) for pair, found in chosen.pairs.items() if found.closure_mismatch]
        ),
        "medoids": medoids(distance, groups),
        "clusters": {
            str(result.cutoff_angstrom): result.to_dict()
            for result in complete_linkage_clusters(
                distance, options.cluster_cutoffs_angstrom, groups=groups
            )
        },
        "spread": {
            "all": positional_spread(
                ok_atoms, binder_chain, residue_pairing=options.residue_pairing
            ).to_dict(),
            "per_group": {
                group: spread.to_dict()
                for group, spread in (
                    positional_spread_by_group(
                        ok_atoms, binder_chain, groups, residue_pairing=options.residue_pairing
                    ).items()
                    if groups
                    else ()
                )
            },
        },
    }

    contacts: dict[str, frozenset] = {}
    not_in_contacts: dict[str, str] = {}
    for pose_id in ok_ids:
        try:
            contacts[pose_id] = heavy_atom_contacts(
                atoms[pose_id],
                binder_chain,
                target_chain=target_chain,
                cutoff_angstrom=options.contact_cutoff_angstrom,
            )
        except ValueError as exc:
            not_in_contacts[pose_id] = _describe(exc)
    block["contacts"] = {
        "definition": (
            "residue pairs with two heavy atoms closer than cutoff_angstrom; the interface is "
            "the residues in those pairs; receptor residues in the numbering of the reference "
            "when frame.renumber_receptor is true"
        ),
        "cutoff_angstrom": options.contact_cutoff_angstrom,
        "excluded": not_in_contacts,
        "interface_jaccard": {
            name: _matrix_entry(matrix, groups, group_by)
            for name, matrix in interface_jaccard(
                {
                    pose_id: InterfaceResidues(
                        binder=frozenset(position for position, _ in pairs),
                        receptor=frozenset(key for _, key in pairs),
                    )
                    for pose_id, pairs in contacts.items()
                }
            ).items()
        }
        if contacts
        else {},
        "contact_map_jaccard": _matrix_entry(contact_map_jaccard(contacts), groups, group_by)
        if contacts
        else None,
    }
    distances = PoseDistances(chosen.pose_ids, chosen.values)
    return _plain(block), distances, unreadable


# ---------------------------------------------------------------------------- the call


def score_pose_set(
    poses: Sequence[Pose],
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    metrics: Sequence[str] = (),
    reference: Reference | Mapping[str, Reference] | None = None,
    *,
    options: PoseSetOptions | Mapping | None = None,
    aggregate: Sequence[str] | Mapping[str, Callable] = DEFAULT_STATISTICS,
    group_by: Sequence[str] = ("backend",),
    representative: str | Callable = "medoid",
    heavy: Literal["representative", "all", "skip"] = "representative",
    metric_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    preflight: Literal["error", "skip", "warn", "off"] = "error",
    on_error: Literal["record", "raise"] = "record",
    n_workers: int = 1,
    tmp_dir: str | os.PathLike | None = None,
    keep_tmp: bool = False,
    residue_table: bool = False,
    include_transform: bool = True,
    construct: Optional[str] = None,
    residue_maps: Mapping[str, Mapping] | None = None,
) -> PoseSetResult:
    """Score the poses of one construct: frame, metrics, agreement, aggregates, representatives.

    The module docstring describes the stages, the row of a pose and the error semantics.

    Args:
        poses: The poses (``DockedPose``, ``ExperimentalPose``, ``PredictedPose``) with unique
            ids. The order does not matter: rows are sorted by id.
        binder_chain: The binder (peptide) chain ID in the poses (user chain IDs, after any
            ``chain_map``); None detects the smaller protein chain of the first readable pose.
        target_chain: The receptor chain ID; None detects the larger other protein chain.
        metrics: Registry names (``"interface"``, ``"hbonds"``, ``"delta_sasa_static"``, ...) run
            on each pose; ``()`` gives the frame, the native scores and the agreement only.
            Metrics that read a trajectory or a prediction directory are refused; ``"prediction"``
            is built from a ``PredictedPose``. A metric that compares with a second structure
            (``structure_rmsd``, ``dockq``, ``evobind_adversarial``) needs ``reference``.
        reference: A ``Reference``, or ``{variant name: Reference}``, that the poses are placed
            on and measured against. The first variant defines the frame of the metrics and of
            the agreement. None uses the first readable pose (lowest id) as the reference.
        options: ``PoseSetOptions`` or a mapping ``{group: {option: value}}`` with the groups
            ``frame``, ``agreement``, ``aggregation``, ``representative`` and ``descriptors``.
        aggregate: Statistic names (``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``)
            or a mapping ``{name: callable}``.
        group_by: Keys of the poses' provenance that define the groups; ``("backend",)`` by
            default. ``()`` gives the group ``"all"`` only.
        representative: ``"medoid"`` (default), ``"confidence"`` (the best native score of
            ``options.representative``) or a callable ``(rows_of_group, PoseDistances) -> id``.
        heavy: Where the heavy metrics (``cost_class`` ``structural`` and ``md``, or none
            declared) run: ``"representative"`` (default), ``"all"`` or ``"skip"``.
        metric_kwargs: ``{metric name: {keyword: value}}``, passed on as they are.
        preflight: ``"error"`` (default) refuses an input that a requested metric cannot take
            (``IncompatibleInputError``) before any pose is scored; ``"skip"`` leaves those
            metrics out and lists them; ``"warn"`` logs and runs; ``"off"`` does not check.
        on_error: ``"record"`` (default) or ``"raise"``, see the module docstring.
        n_workers: Processes for the per-pose stage; 1 runs in this process. The workers use the
            start method of the platform (``fork``, ``spawn`` or ``forkserver``); a pool that
            cannot start gives a logged warning and the poses are scored in this process. The
            metrics that need a GPU never run in a worker.
        tmp_dir: An existing directory for the temporary structure files (default the system
            temporary directory); a directory ``bm_poseset_*`` is made in it and removed on exit.
        keep_tmp: Leave that directory in place (its path is in ``provenance["temp_dir"]``).
        residue_table: Add ``residue_table`` to each row: the interface residue by residue, in
            long format (``descriptors.TABLE_COLUMNS``), with the reference numbering and the
            residue sets of the first reference variant. Off by default; it is long.
        include_transform: Put the rotation and translation of each frame in the rows.
        construct: A name for the construct, recorded in the result.
        residue_maps: ``{pose id: {pose ResidueKey: reference ResidueKey}}`` that replace the
            residue map of the reference for those poses.

    Returns:
        A ``PoseSetResult``; its type never depends on an argument.

    Raises:
        TypeError, ValueError, KeyError: For an argument that is invalid, before any pose is
            scored. An invalid ``Reference`` raises ``ValueError`` listing every problem.
        ImportError: If a requested metric needs a package that is not installed.
        IncompatibleInputError: If the pre-flight check refuses the input.
        PoseScoringError: With ``on_error="raise"``, at the first pose that failed.
    """
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    # -- 1. arguments --------------------------------------------------------------------
    options = _coerce_options(options)
    for role, value in (("binder_chain", binder_chain), ("target_chain", target_chain)):
        _check_chain_id(role, value)
    if binder_chain is not None and binder_chain == target_chain:
        raise ValueError(f"binder_chain and target_chain are both {binder_chain!r}: two chains")
    _one_of("heavy", heavy, HEAVY_MODES)
    _one_of("preflight", preflight, PREFLIGHT_MODES)
    _one_of("on_error", on_error, ON_ERROR_MODES)
    if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError(f"n_workers must be an integer of at least 1, got {n_workers!r}")
    for name, flag in (
        ("keep_tmp", keep_tmp),
        ("residue_table", residue_table),
        ("include_transform", include_transform),
    ):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be True or False, got {flag!r}")
    if construct is not None and not isinstance(construct, str):
        raise TypeError(f"construct must be a string or None, got {construct!r}")
    if tmp_dir is not None and not Path(tmp_dir).is_dir():
        raise ValueError(f"tmp_dir {str(tmp_dir)!r} is not an existing directory")
    aggregators = resolve_statistics(aggregate)
    group_by = _check_group_by(group_by)
    rule_name = _check_representative(representative, options.representative)
    variants = normalize_references(reference)
    plans = _plan_metrics(metrics, has_reference=bool(variants), heavy=heavy)
    kwargs_by_metric = _check_metric_kwargs(metric_kwargs, plans)
    sorted_poses = _resolve_poses(poses)
    ids = [pose_id for pose_id, _ in sorted_poses]
    maps_by_pose = _check_residue_maps(residue_maps, ids)
    provenance_by_id = {pose_id: _provenance_of(pose) for pose_id, pose in sorted_poses}
    reserved_groups = sorted(
        {
            group_key({"provenance": provenance_by_id[pose_id]}, group_by)
            for pose_id in ids
            if group_by
        }
        & set(RESERVED_GROUP_KEYS)
    )
    if reserved_groups:
        raise ValueError(
            f"the group keys {reserved_groups} are reserved for the result; rename the "
            "provenance value or group by another key"
        )
    for plan in plans:
        if not plan.served and plan.scope != "none":
            plan.spec.load()  # ImportError for a missing extra, before any pose is scored
    for name in variants:
        variants[name].frame_key(options.frame.frame_residues)  # an unknown frame_residues
        prepare_reference(variants[name], site_cutoff_angstrom=options.frame.site_cutoff_angstrom)
    stage_seconds["arguments"] = time.perf_counter() - started
    logger.info(
        "pose set: %d poses, metrics %s, %d reference variant(s), on_error=%s, n_workers=%d",
        len(sorted_poses),
        [plan.name for plan in plans],
        len(variants),
        on_error,
        n_workers,
    )

    # -- 2. first pose, chains, reference, pre-flight --------------------------------------
    stage_started = time.perf_counter()
    raise_mode = on_error == "raise"
    scan = _scan_first_pose(sorted_poses, binder_chain, target_chain, raise_mode)
    implicit = not variants
    preflight_report: Optional[dict] = None
    notes: list[str] = []
    if scan.pose is not None:
        if implicit:
            variants = {
                IMPLICIT_REFERENCE_NAME: Reference(
                    scan.pose,
                    scan.target_chain,
                    scan.binder_chain,
                    name=IMPLICIT_REFERENCE_NAME,
                )
            }
            variants[IMPLICIT_REFERENCE_NAME].frame_key(options.frame.frame_residues)
            prepare_reference(
                variants[IMPLICIT_REFERENCE_NAME],
                site_cutoff_angstrom=options.frame.site_cutoff_angstrom,
            )
            logger.info("no reference given: poses are placed on the receptor of %r", scan.pose.id)
        preflight_report, plans, notes = _preflight(
            scan.profile,
            plans,
            preflight,
            has_reference=not implicit,
            first_kind=scan.pose.kind,
        )
    stage_seconds["first_pose"] = time.perf_counter() - stage_started

    tmp_root: Optional[str] = None
    try:
        needs_files = any(
            plan.scope != "none"
            and plan.skipped is None
            and not plan.served
            and (plan.spec.input_type in _FILE_INPUT_TYPES or plan.spec.secondary_path_arg)
            for plan in plans
        )
        if needs_files and scan.pose is not None:
            tmp_root = tempfile.mkdtemp(
                prefix="bm_poseset_", dir=None if tmp_dir is None else str(tmp_dir)
            )
        reference_path = None
        if tmp_root is not None and any(plan.spec.secondary_path_arg for plan in plans):
            first_reference = next(iter(variants.values()))
            reference_path = _own_file(first_reference.pose)
            if reference_path is None:
                target_file = Path(tmp_root) / "reference.cif"
                _write_structure(first_reference.atoms(), target_file)
                reference_path = str(target_file)

        # -- 3. per pose ----------------------------------------------------------------
        stage_started = time.perf_counter()
        if scan.pose is None:
            outcomes = _all_failed(sorted_poses, scan)
        else:
            ctx = _Context(
                binder_chain=scan.binder_chain,
                target_chain=scan.target_chain,
                options=options,
                variants=variants,
                plans=tuple(plans),
                metric_kwargs=kwargs_by_metric,
                tmp_root=tmp_root,
                reference_path=reference_path,
                residue_maps=maps_by_pose,
                include_transform=include_transform,
                raise_mode=raise_mode,
                residue_table=residue_table,
            )
            items = list(enumerate(pose for _, pose in sorted_poses))
            outcomes = _score_poses(items, ctx, n_workers)
        stage_seconds["poses"] = time.perf_counter() - stage_started
        logger.info(
            "per-pose stage done: %d ok, %d failed",
            sum(1 for o in outcomes if o.row["status"] != "failed"),
            sum(1 for o in outcomes if o.row["status"] == "failed"),
        )
        rows = [outcome.row for outcome in outcomes]
        profile_warnings = _compare_profiles(outcomes, scan)

        # -- 4. agreement ------------------------------------------------------------------
        stage_started = time.perf_counter()
        agreement, distances, unreadable = _run_agreement(
            options.agreement, scan, group_by, outcomes, raise_mode
        )
        for pose_id, reason in unreadable.items():
            row = next(row for row in rows if row["id"] == pose_id)
            row["reasons"]["agreement"] = reason
        stage_seconds["agreement"] = time.perf_counter() - stage_started
        logger.info(
            "agreement: %s, %d pose(s) compared", agreement["status"], len(agreement["pose_ids"])
        )

        # -- 5. representatives, heavy metrics ------------------------------------------------
        stage_started = time.perf_counter()
        representatives = _select(
            rows, distances, group_by, representative, rule_name, options.representative
        )
        stage_seconds["representatives"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        if scan.pose is not None:
            _run_heavy(plans, representatives, sorted_poses, outcomes, ctx, raise_mode)
        stage_seconds["heavy_metrics"] = time.perf_counter() - stage_started
    finally:
        if tmp_root is not None and not keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)

    # -- 6. aggregates and assembly -----------------------------------------------------------
    stage_started = time.perf_counter()
    for plan in plans:
        if plan.skipped is not None:
            for row in rows:
                if row["status"] != "failed":
                    row["reasons"][plan.name] = plan.skipped
    for row in rows:
        _refresh_status(row)
        if row["status"] == "failed":
            logger.warning("pose %s failed: %s", row["id"], row["reason"])
    metric_order = [plan.name for plan in plans]
    rows = [_ordered_row(row, metric_order) for row in rows]
    aggregates = aggregate_poses(
        rows,
        group_by,
        statistics=aggregators,
        min_n_ok=options.aggregation.min_n_ok,
        reason_policy=options.aggregation.reason_policy,
    )
    stage_seconds["aggregates"] = time.perf_counter() - stage_started

    parameters = {
        "options": options.to_dict(),
        "metrics": metric_order,
        "metric_scope": {plan.name: "none" if plan.skipped else plan.scope for plan in plans},
        "metric_kwargs": _plain(kwargs_by_metric),
        "aggregate": list(aggregators),
        "group_by": list(group_by),
        "representative": rule_name,
        "heavy": heavy,
        "preflight": preflight,
        "residue_maps": {
            pose_id: sorted([k.to_list(), v.to_list()] for k, v in mapping.items())
            for pose_id, mapping in sorted(maps_by_pose.items())
        },
        "residue_table": residue_table,
        "include_transform": include_transform,
    }
    references = _references_block(variants, implicit, scan)
    warnings = _collect_warnings(rows, representatives, notes, profile_warnings, agreement)
    from binding_metrics.provenance import collect_provenance

    provenance = collect_provenance()
    provenance["pose_set_schema_version"] = SCHEMA_VERSION
    provenance["n_poses"] = len(rows)
    provenance["run"] = {
        "n_workers": n_workers,
        "on_error": on_error,
        "tmp_dir": None if tmp_dir is None else str(tmp_dir),
        "keep_tmp": keep_tmp,
    }
    if keep_tmp and tmp_root is not None:
        provenance["temp_dir"] = tmp_root
    input_key = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "poses": [
                    {
                        "id": row["id"],
                        "content_hash": row["content_hash"],
                        "provenance": row["provenance"],
                    }
                    for row in rows
                ],
                "binder_chain": scan.binder_chain,
                "target_chain": scan.target_chain,
                "references": {
                    name: {"content_hash": ref.content_hash(), "implicit": implicit}
                    for name, ref in variants.items()
                },
                "parameters": parameters,
            }
        ).encode("utf-8")
    ).hexdigest()
    stage_seconds["total"] = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "package_version": provenance.get("package_version"),
        "construct": construct,
        "binder_chain": scan.binder_chain,
        "target_chain": scan.target_chain,
        "input_key": input_key,
        "provenance": provenance,
        "parameters": parameters,
        "references": references,
        "preflight": preflight_report,
        "poses": rows,
        "agreement": agreement,
        "aggregates": _plain(aggregates),
        "representatives": _plain(representatives),
        "warnings": warnings,
        "timing": {
            "stages_seconds": stage_seconds,
            "poses_seconds": {outcome.pose_id: outcome.seconds for outcome in outcomes},
        },
    }
    ordered = {key: result[key] for key in RESULT_KEYS if key in result}
    ordered["nonfinite_fields"] = _nonfinite_paths(
        {key: value for key, value in ordered.items() if key != "timing"}
    )
    ordered = {key: ordered[key] for key in RESULT_KEYS if key in ordered}
    logger.info(
        "pose set scored in %.2f s: %d poses, %d failed",
        stage_seconds["total"],
        len(rows),
        sum(1 for row in rows if row["status"] == "failed"),
    )
    return PoseSetResult(ordered)


# ---------------------------------------------------------------------------- helpers of the call


def _check_representative(representative: Any, options: RepresentativeOptions) -> str:
    """Validate the representative rule before any pose is read; return its name."""
    if callable(representative):
        return "custom"
    if representative == "medoid":
        return "medoid"
    if representative == "confidence":
        direction = options.confidence_direction or native_score_direction(options.confidence_key)
        if direction is None:
            raise ValueError(
                f"the direction of native score {options.confidence_key!r} is not known; set "
                "options.representative.confidence_direction to 'max' or 'min'"
            )
        return "confidence"
    raise ValueError(
        f"representative must be 'medoid', 'confidence' or a callable, got {representative!r}"
    )


def _all_failed(poses: Sequence[tuple[str, Pose]], scan: _Scan) -> list[_Outcome]:
    """The outcomes of a call in which no pose could be read: every row failed."""
    outcomes = []
    for index, (pose_id, pose) in enumerate(poses):
        outcome = _Outcome(pose_id, index, _new_row(pose))
        stage, text = scan.failures.get(pose_id, ("load", "the pose could not be read"))
        _fail(outcome, stage, text)
        outcomes.append(outcome)
    return outcomes


def _compare_profiles(outcomes: Sequence[_Outcome], scan: _Scan) -> list[str]:
    """Mark the rows whose binder profile differs from the first pose's; return the warnings."""
    if scan.pose is None:
        return []
    first = next((o.row for o in outcomes if o.pose_id == scan.pose.id), None)
    if first is None or first["profile"] is None:
        return []
    base = first["profile"]
    warnings = []
    for outcome in outcomes:
        profile = outcome.row["profile"]
        if profile is None or outcome.pose_id == scan.pose.id:
            continue
        differing = []
        if profile["n_binder_residues"] != base["n_binder_residues"]:
            differing.append("n_binder_residues")
        if set(profile["closures"]) != set(base["closures"]):
            differing.append("closures")
        if set(profile["residue_classes"]) != set(base["residue_classes"]):
            differing.append("residue_classes")
        profile["differs_from_first"] = differing
        if differing:
            warnings.append(
                f"{outcome.pose_id}: the binder profile differs from that of {scan.pose.id} "
                f"in {', '.join(differing)}"
            )
    return warnings


def _run_agreement(
    options: AgreementOptions,
    scan: _Scan,
    group_by: tuple[str, ...],
    outcomes: Sequence[_Outcome],
    raise_mode: bool,
) -> tuple[dict, Optional[PoseDistances], dict[str, str]]:
    if scan.pose is None:
        block = {
            "status": "skipped",
            "reason": "no pose could be read",
            "kind": options.kind,
            "pose_ids": [],
            "excluded": {o.pose_id: o.row["reason"] for o in outcomes},
        }
        return block, None, {}
    try:
        return _agreement_stage(options, scan.binder_chain, scan.target_chain, group_by, outcomes)
    except Exception as exc:  # noqa: BLE001 - the agreement stage reads every pose; the failure is logged and recorded in the block
        if raise_mode:
            raise
        logger.warning("agreement stage failed: %s", _describe(exc))
        block = {
            "status": "failed",
            "reason": _describe(exc),
            "kind": options.kind,
            "pose_ids": [],
            "excluded": {},
        }
        return block, None, {}


def _select(
    rows: Sequence[dict],
    distances: Optional[PoseDistances],
    group_by: tuple[str, ...],
    representative: Any,
    rule_name: str,
    options: RepresentativeOptions,
) -> dict:
    """Stage 5a: one representative pose per group and overall."""
    alive = [row for row in rows if row["status"] != "failed"]
    if rule_name == "medoid" and distances is None and alive:
        keys = sorted({group_key(row, group_by) for row in rows}) if group_by else []
        reason = "the agreement stage gave no distance matrix"
        return {
            **{key: None for key in keys},
            "all": None,
            "rule": "medoid",
            "reasons": {key: reason for key in [*keys, "all"]},
        }
    if distances is None:
        distances = PoseDistances((), np.zeros((0, 0)))
    return select_representatives(
        rows,
        distances,
        group_by=group_by,
        rule=representative,
        confidence_key=options.confidence_key,
        confidence_direction=options.confidence_direction,
    )


def _run_heavy(
    plans: Sequence[_MetricPlan],
    representatives: Mapping[str, Any],
    poses: Sequence[tuple[str, Pose]],
    outcomes: Sequence[_Outcome],
    ctx: _Context,
    raise_mode: bool,
) -> None:
    """Stage 5b: the heavy metrics on the representatives, one after the other in this process."""
    heavy = [plan for plan in plans if plan.scope == "representatives" and plan.skipped is None]
    if not heavy:
        return
    chosen = sorted(
        {
            pose_id
            for key, pose_id in representatives.items()
            if key not in REPRESENTATIVE_META_KEYS and isinstance(pose_id, str)
        }
    )
    logger.info("heavy metrics %s on %d representative(s)", [p.name for p in heavy], len(chosen))
    pose_by_id = dict(poses)
    outcome_by_id = {outcome.pose_id: outcome for outcome in outcomes}
    for pose_id in chosen:
        outcome = outcome_by_id[pose_id]
        if outcome.row["status"] == "failed" or outcome.metrics_atoms is None:
            continue
        outcome.failure = None
        files = _PoseFiles(ctx, outcome.index, pose_by_id[pose_id], outcome.metrics_atoms)
        _run_plans(outcome, heavy, pose_by_id[pose_id], ctx, files)
        if raise_mode and outcome.failure is not None:
            _raise_failure(outcome)


_ROW_ORDER = (
    "id",
    "kind",
    "content_hash",
    "provenance",
    "native",
    "status",
    "reason",
    "reasons",
    "independent_of_design",
    "profile",
    "frame",
)


def _ordered_row(row: Mapping[str, Any], metric_order: Sequence[str]) -> dict:
    """The row with its keys in the fixed order: reserved fields, metric blocks, then the rest."""
    ordered = {key: row[key] for key in _ROW_ORDER if key in row}
    for name in (*metric_order, "descriptors", "residue_table"):
        if name in row:
            ordered[name] = row[name]
    for key, value in row.items():
        ordered.setdefault(key, value)
    return ordered


def _references_block(variants: Mapping[str, Reference], implicit: bool, scan: _Scan) -> dict:
    return {
        "implicit": implicit,
        "implicit_pose_id": scan.pose.id if implicit and scan.pose is not None else None,
        "metrics_frame_variant": next(iter(variants), None),
        "variants": {
            name: {"content_hash": reference.content_hash(), "reference": reference.to_dict()}
            for name, reference in variants.items()
        },
    }


def _medoid_caveats(
    agreement: Mapping[str, Any], representatives: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """``[(group, sentence)]`` of the medoids chosen with a caveat, once per group and sentence.

    Read from ``agreement["medoids"]`` (the overall medoid is the group ``"all"``) and from
    ``representatives["caveats"]``; both come from ``agreement.medoid`` and say the same thing
    when they cover the same poses.
    """
    found: list[tuple[str, str]] = []
    medoids = agreement.get("medoids") or {}
    overall = medoids.get("overall") or {}
    if overall.get("caveat"):
        found.append(("all", overall["caveat"]))
    for group, entry in (medoids.get("per_group") or {}).items():
        if entry.get("caveat"):
            found.append((group, entry["caveat"]))
    found += list((representatives.get("caveats") or {}).items())
    return list(dict.fromkeys(found))


def _collect_warnings(
    rows: Sequence[Mapping[str, Any]],
    representatives: Mapping[str, Any],
    preflight_notes: Sequence[str],
    profile_warnings: Sequence[str],
    agreement: Mapping[str, Any],
) -> list[str]:
    warnings = list(preflight_notes)
    for row in rows:
        frame = row.get("frame")
        if isinstance(frame, Mapping):
            warnings += [f"{row['id']}: {text}" for text in frame.get("warnings", ())]
    warnings += profile_warnings
    for key, reason in (representatives.get("reasons") or {}).items():
        warnings.append(f"no representative for {key!r}: {reason}")
    for group, caveat in _medoid_caveats(agreement, representatives):
        warnings.append(f"medoid of {group!r}: {caveat}")
    return warnings


__all__ = [
    "AgreementOptions",
    "AggregationOptions",
    "IMPLICIT_REFERENCE_NAME",
    "PoseScoringError",
    "PoseSetOptions",
    "PoseSetResult",
    "REPRESENTATIVE_META_KEYS",
    "RESERVED_ROW_NAMES",
    "RESULT_KEYS",
    "RepresentativeOptions",
    "SCHEMA",
    "SCHEMA_VERSION",
    "score_pose_set",
]
