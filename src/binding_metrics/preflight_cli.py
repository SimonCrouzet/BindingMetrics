"""The pre-flight check of the command-line pipeline (``binding-metrics-run`` and ``-batch``).

``check_input`` profiles one structure, lists the steps that the run would execute as registry
metrics and predictors, and hands them to ``binding_metrics.capabilities.preflight``. It runs
first, before preparation, relaxation, any model run and any touch of the prediction store.
What comes back is a ``PreflightOutcome``: the block for ``results["preflight"]``, the steps to
leave out under ``--on-incompatible skip``, and the error to raise under ``error``.

Rules the hooks follow:

* The steps checked are the ones the run will execute: ``relax`` (``md_implicit``), ``energy``,
  ``interface``, ``geometry`` (its three metrics one by one), ``electrostatics`` and the model
  step (the ``openfold`` metric for the OpenFold3 step, ``prediction`` for ``--predictor``).
  ``dockq`` is not checked: without a reference the pipeline already skips it and says so.
* A model whose output was made elsewhere (``--prediction-dir``) gets the policy ``warn``: what
  it was given is not known, so its limits never refuse an input that may have been predicted
  correctly.
* An input that cannot be profiled runs as it did before, with a warning and the status
  ``skipped`` in the block: a failed check is not an incompatibility.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from binding_metrics.capabilities import (
    BINDER_TYPES,
    POLICIES,
    IncompatibleInputError,
    PreflightReport,
    check_openfold3_residues,
    preflight,
    profile_input,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BINDER_TYPE_CHOICES",
    "MODEL_STEP",
    "PIPELINE_STEP_METRICS",
    "PreflightOutcome",
    "add_preflight_args",
    "check_input",
    "model_step_of",
    "refusal_block",
    "steps_that_run",
]

#: The choices of ``--binder-type``.
BINDER_TYPE_CHOICES: tuple[str, ...] = ("auto", *BINDER_TYPES)

#: The pipeline step that runs a structure-prediction model (its name in ``--metrics``).
MODEL_STEP = "openfold"

#: Pipeline step -> the registry metrics it runs, for the steps whose limits are checked.
PIPELINE_STEP_METRICS: dict[str, tuple[str, ...]] = {
    "relax": ("md_implicit",),
    "energy": ("structure_interaction_energy",),
    "interface": ("interface",),
    "geometry": ("ramachandran", "omega", "shape_complementarity"),
    "electrostatics": ("coulomb",),
}


def add_preflight_args(parser) -> None:
    """Add ``--binder-type``, ``--on-incompatible`` and ``--preflight-only`` to a parser."""
    group = parser.add_argument_group("Pre-flight check")
    group.add_argument(
        "--binder-type",
        choices=BINDER_TYPE_CHOICES,
        default="auto",
        help=(
            "What the binder is, for the checks that depend on it. auto (default) estimates it "
            "from the number of residues: at most 40 a peptide, at most 100 a miniprotein, "
            "longer unknown, which skips the type checks. A nanobody or an antibody chain is "
            "never guessed: name it."
        ),
    )
    group.add_argument(
        "--on-incompatible",
        choices=POLICIES,
        default="error",
        help=(
            "What to do when the input cannot go through a requested step or model, found "
            "before anything runs. error (default): refuse, listing every problem with its fix. "
            "skip: leave out the incompatible steps, record why, and run the rest. warn: log "
            "the problems and run everything. An output read with --prediction-dir only warns."
        ),
    )
    group.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Print the pre-flight plan (what would run, what is incompatible and why) and "
            "stop, without preparing, relaxing or predicting anything. Exit status 1 when "
            "--on-incompatible is error and something is refused."
        ),
    )


def steps_that_run(metrics, *, skip_relax: bool, custom_relaxer: bool = False) -> frozenset[str]:
    """The checked pipeline steps that a run with these options executes.

    ``metrics`` are the step names of ``--metrics``. A run with its own ``Relaxer`` object is not
    checked for ``relax``: its limits are its own.
    """
    steps = {name for name in metrics if name in PIPELINE_STEP_METRICS}
    if not skip_relax and not custom_relaxer:
        steps.add("relax")
    return frozenset(steps)


def model_step_of(
    predictor: Optional[str], prediction_dir: Optional[Path], metrics
) -> Optional[tuple[str, str, bool]]:
    """The model step a run executes as ``(model, metric name, output made elsewhere)``.

    None when the run has no model step. The step is ``openfold`` in ``--metrics``: without
    ``--predictor`` it runs OpenFold3 (metric ``openfold``); with it, that model's prediction
    through the store (metric ``prediction``), adopted from ``--prediction-dir`` when given.
    """
    if MODEL_STEP not in metrics:
        return None
    if predictor is None:
        return ("of3", "openfold", False)
    return (predictor, "prediction", prediction_dir is not None)


@dataclass
class PreflightOutcome:
    """The result of one pre-flight check.

    Attributes:
        block: The dict for ``results["preflight"]``: ``status`` (``ok``, ``warn``, ``skipped``,
            ``refused`` or ``not_checked``), ``reason``, the policy, the steps left out, and the
            full ``report`` (``PreflightReport.to_dict``).
        report: The ``PreflightReport``; None when the input could not be profiled.
        error: The ``IncompatibleInputError`` a caller raises (policy ``error``); None otherwise.
        skipped_steps: Pipeline step -> reason, for the steps left out under ``skip``.
        skipped_geometry: Metric -> reason, for the metrics of the ``geometry`` step left out
            under ``skip`` (the step itself only when all three are).
        model_usable: False when the model step is left out under ``skip``.
    """

    block: dict
    report: Optional[PreflightReport] = None
    error: Optional[IncompatibleInputError] = None
    skipped_steps: dict[str, str] = field(default_factory=dict)
    skipped_geometry: dict[str, str] = field(default_factory=dict)
    model_usable: bool = True

    @property
    def refused(self) -> bool:
        return self.error is not None


def _without_residue_check(model: str) -> Any:
    """The registered predictor ``model`` minus the OpenFold3 residue check, if it has one.

    ``--on-unmappable-residue x`` asks the query builder to send an ``X`` for a residue it cannot
    take and to log a warning, so the pre-flight check must not refuse that residue.
    """
    from binding_metrics.predictors.registry import PARSERS

    spec = PARSERS.get(model)
    declared = spec.load_capabilities() if spec is not None else None
    if declared is None or check_openfold3_residues not in declared.extra_checks:
        return model
    kept = tuple(c for c in declared.extra_checks if c is not check_openfold3_residues)
    return SimpleNamespace(
        name=model,
        display_name=spec.display_name,
        capabilities=dataclasses.replace(declared, extra_checks=kept),
    )


def _short(violation) -> str:
    return f"{violation.subject}: {violation.constraint}: {violation.fact}"


def _not_checked(reason: str, policy: str) -> PreflightOutcome:
    return PreflightOutcome(
        block={"status": "not_checked", "reason": reason, "policy": policy, "report": None}
    )


def refusal_block(error: IncompatibleInputError) -> dict:
    """The ``results["preflight"]`` block of a refused input (a batch error row carries it)."""
    report = error.report
    return {
        "status": "refused",
        "reason": "; ".join(_short(v) for v in report.violations),
        "policy": report.policy,
        "skipped_steps": {},
        "skipped_geometry": {},
        "report": report.to_dict(),
    }


def check_input(
    input_path: Path,
    binder_chain: Optional[str],
    receptor_chain: Optional[str],
    *,
    binder_type: str = "auto",
    on_incompatible: str = "error",
    steps: frozenset[str] = frozenset(),
    model: Optional[tuple[str, str, bool]] = None,
    reference_path: Optional[Path] = None,
    include_plan: bool = False,
    on_unmappable_residue: str = "error",
) -> PreflightOutcome:
    """Check one input against the steps and the model a run will execute.

    Never raises ``IncompatibleInputError``: under ``error`` it comes back as
    ``outcome.error`` and the caller decides (``run_pipeline`` raises it; ``--preflight-only``
    prints it). It reads the structure once and imports no model.

    Args:
        input_path: The complex structure.
        binder_chain, receptor_chain: Resolved chain IDs (the author IDs the metrics use).
        binder_type: ``auto`` or one of ``BINDER_TYPES``.
        on_incompatible: ``error``, ``skip`` or ``warn``.
        steps: The pipeline steps that run, from ``steps_that_run``.
        model: The model step, from ``model_step_of``.
        reference_path: The reference structure, when one is given.
        include_plan: Put the text of the plan into the block (``--preflight-only``).
        on_unmappable_residue: ``x`` lifts the residue check of the OpenFold3 query builder: the
            builder sends an ``X`` and logs a warning, as the option asks.
    """
    if not binder_chain:
        return _not_checked(
            "no binder chain was found, so the input was not profiled", on_incompatible
        )
    try:
        profile = profile_input(Path(input_path), binder_chain, receptor_chain, binder_type)
    except Exception as exc:  # noqa: BLE001 - a check that fails is not an incompatibility
        logger.warning("  pre-flight check skipped, the input could not be profiled: %s", exc)
        return _not_checked(f"the input could not be profiled: {exc}", on_incompatible)

    metric_names: list[str] = []
    step_of_metric: dict[str, str] = {}
    for step in PIPELINE_STEP_METRICS:
        if step in steps:
            for name in PIPELINE_STEP_METRICS[step]:
                metric_names.append(name)
                step_of_metric[name] = step
    predictor = None
    adopted = False
    provided: set[str] = set()
    if reference_path is not None:
        provided.add("reference_structure")
    if model is not None:
        predictor, model_metric, adopted = model
        if on_unmappable_residue == "x":
            predictor = _without_residue_check(predictor)
        metric_names.append(model_metric)
        step_of_metric[model_metric] = MODEL_STEP
        provided.add("predicted_structure")

    error: Optional[IncompatibleInputError] = None
    try:
        report = preflight(
            profile,
            metric_names,
            predictor,
            policy=on_incompatible,
            predictor_policy="warn" if adopted else None,
            provided=provided,
        )
    except IncompatibleInputError as refused:
        error, report = refused, refused.report

    skipped_steps: dict[str, str] = {}
    skipped_geometry: dict[str, str] = {}
    if on_incompatible == "skip":
        for name in report.skipped_metrics:
            reason = "; ".join(
                _short(v) for v in report.violations if v.kind == "metric" and v.name == name
            )
            step = step_of_metric[name]
            if step == "geometry":
                skipped_geometry[name] = reason
                if set(skipped_geometry) >= set(PIPELINE_STEP_METRICS["geometry"]):
                    skipped_steps["geometry"] = "; ".join(skipped_geometry.values())
            else:
                skipped_steps[step] = reason
    model_usable = report.predictor_usable
    if not model_usable and model is not None:
        skipped_steps[MODEL_STEP] = "; ".join(
            _short(v) for v in report.violations if v.kind == "predictor"
        )
        # the model's registry metric goes with its predictor; the plan must not list it as run
        report = dataclasses.replace(
            report,
            metrics_to_run=tuple(m for m in report.metrics_to_run if m != model[1]),
        )

    violations = [_short(v) for v in report.violations]
    if error is not None:
        status, reason = "refused", "; ".join(violations)
    elif skipped_steps or skipped_geometry:
        status, reason = "skipped", "; ".join(violations)
    elif violations:
        status, reason = "warn", "; ".join(violations)
    elif report.warnings:
        status, reason = "warn", "; ".join(report.warnings)
    else:
        status, reason = "ok", ""
    block: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "policy": on_incompatible,
        "skipped_steps": dict(skipped_steps),
        "skipped_geometry": dict(skipped_geometry),
        "report": report.to_dict(),
    }
    if include_plan:
        block["plan"] = report.format()
    return PreflightOutcome(
        block=block,
        report=report,
        error=error,
        skipped_steps=skipped_steps,
        skipped_geometry=skipped_geometry,
        model_usable=model_usable,
    )
