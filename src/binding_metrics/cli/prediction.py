"""The structure-prediction step of ``binding-metrics-run`` and ``binding-metrics-batch``.

With ``--predictor MODEL`` the step of the pipeline that ``--metrics openfold`` names does not
call OpenFold3 directly. It builds one ``PredictionSession`` per sample on a ``PredictionStore``,
so a model runs at most once for all the metrics that read its output (the confidence scalars,
the interface PAE and PDE, the EvoBind score and the adversarial check), and writes
``results["prediction"]``. Without ``--predictor`` nothing here runs and the OpenFold step keeps
its own code path and ``results["openfold"]``.

What the step does for one sample, in order::

    request = <runner>.make_request(...)      # a run: OpenFold3, ColabFold, Boltz-2, Protenix
            = PredictionRequest(...)          # --prediction-dir: outputs the user made
    session.adopt(request, prediction_dir)    # only for --prediction-dir; the model never runs
    record = session.record(request)          # runs the model on the first miss, parses once
    summarize_prediction(record, ...)         # scalars, binder pLDDT, interface PAE/PDE, RMSD
    EvoBind score of the record               # merged into the block
    EvoBind adversarial check                 # the input pose against the record

``results["prediction"]`` holds the keys of ``summarize_prediction`` (with ``model``), the EvoBind
keys merged as the OpenFold step merges them, ``mode`` (how the model was used: ``predict``,
``refold``, ``score`` or ``score-lock``; None for an output whose making is not stated),
``weights`` (the weights of the run, see ``weights_description``) and ``cache``: the counters
of ``PredictionSession.stats()`` and ``request_key``, the name of the store entry. A prediction
that failed, or that cannot be run here, gives
``{"model": ..., "mode": ..., "error": <reason>, "cache": ...}``; the sample goes on with its
other steps.

Every registered model has a runner (``RUNNERS``): ``af2`` (ColabFold), ``boltz2``, ``of3`` and
``protenix``. A model without one is refused without ``--prediction-dir`` while the command line
is checked, before anything runs. The runners differ in what they can do, and the step asks them:

* ``supported_modes`` and ``default_mode`` of the runner class say which ``--prediction-mode`` a
  run from here can have and which it has by default (``runner_modes``, ``runner_default_mode``);
  a mode the runner does not have is a usage error that names the ones it has.
* the keywords of the runner's ``make_request`` decide which settings reach it: ``binder_cyclic``,
  ``use_msa_server``, ``lock_threshold_angstrom`` and ``weights`` are passed only when the option
  was given and only to a runner that has the keyword; given for a runner that has not, they are a
  usage error that names the option and the model (``resolve_prediction_options``).
* ``output_chain_map(request)`` of the runner gives the chain IDs of its prediction when they are
  not the input's, and is passed to the reader as ``chain_map`` unless ``--prediction-binder-chain``
  or ``--prediction-target-chain`` is given.

``--prediction-cyclic``, ``--prediction-no-msa-server`` and ``--prediction-conda-env`` are the
settings of the ``--predictor`` route for every model. ``--openfold-cyclic``,
``--openfold-no-msa-server`` and ``--openfold-conda-env`` stay the settings of the OpenFold3 step
and, for ``--predictor of3``, the same setting under the older name: giving both with different
values is a usage error. For another model the ``--openfold-*`` spelling is a usage error that
names the generic one.

The store lives in ``--prediction-cache DIR``; the default is ``<output-dir>/predictions`` for a
single run and ``<output-dir>/_predictions`` for a batch (the underscore keeps it apart from a
sample called "predictions"). A batch shares one store root, so a second run over the same
samples finds every prediction and starts no model.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

from binding_metrics.capabilities import MODES
from binding_metrics.cli import (
    OPENFOLD_CYCLIC_CHOICES,
    check_openfold_cyclic,
    merge_reason,
)
from binding_metrics.predictors.registry import PARSERS

logger = logging.getLogger("binding_metrics.cli.prediction")

#: Models that have a runner: ``"module:Class"`` of a ``PredictionRunner`` whose constructor
#: takes the conda environment. Imported when a runner is made, so a test can patch the class.
RUNNERS: dict[str, str] = {
    "af2": "binding_metrics.predictors.af2_runner:ColabFoldRunner",
    "boltz2": "binding_metrics.predictors.boltz2_runner:Boltz2Runner",
    "of3": "binding_metrics.predictors.of3_runner:OpenFold3Runner",
    "protenix": "binding_metrics.predictors.protenix_runner:ProtenixRunner",
}

#: Value of ``--openfold-conda-env`` when it is not given (the option names an environment on
#: purpose): it does not count as a value that conflicts with ``--prediction-conda-env``.
OPENFOLD_DEFAULT_CONDA_ENV = "openfold3"

#: Store folder inside ``--output-dir`` for a single run and for a batch.
DEFAULT_CACHE_DIRNAME = "predictions"
BATCH_CACHE_DIRNAME = "_predictions"

# Distance below which a receptor residue is at the interface; the value of the EvoBind functions.
_INTERFACE_CUTOFF_ANGSTROM = 8.0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def add_prediction_args(parser: argparse.ArgumentParser, *, batch: bool = False) -> None:
    """Add the "Prediction" option group of ``binding-metrics-run`` and ``-batch``.

    Args:
        parser: The parser to add the group to.
        batch: True for ``binding-metrics-batch``, where ``--prediction-dir`` is the root that
            holds one output per sample.
    """
    group = parser.add_argument_group("Prediction")
    group.add_argument(
        "--predictor",
        choices=sorted(PARSERS),
        default=None,
        help=(
            "Read the structure prediction of this model in the 'openfold' step, write it to "
            "results['prediction'] and run the model at most once for all metrics that use "
            "it. Every model can be run from here (af2 through ColabFold, boltz2, of3, "
            "protenix) unless --prediction-dir gives an output you made. Default: not given, "
            "the step runs OpenFold3 and writes results['openfold']."
        ),
    )
    where = (
        "Root that holds one output per sample, found by the sample ID (the file stem)."
        if batch
        else "Directory of the output you made with the model."
    )
    group.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            f"{where} It is read, never run: the layout under it is the model's own "
            "(for OpenFold3 the folder that holds <sample>/seed_*/). Needs --predictor."
        ),
    )
    group.add_argument(
        "--prediction-mode",
        choices=MODES,
        default=None,
        help=(
            "How the model is used for the complex: predict (sequences only), refold (receptor "
            "templated, binder predicted freely), score (every chain templated on its own, the "
            "pose not given: re-docking) or score-lock (score, with the pose pinned to the input). "
            "It is checked against what the model supports and, for a run from here, against "
            "what its runner can do (of3: refold, score; boltz2: all four; af2 and protenix: "
            "predict), before anything runs, and recorded. Default: the runner's own mode, "
            "that is for --predictor of3 the value of --openfold-mode (score), score for "
            "boltz2 and predict for af2 and protenix; for an output read with --prediction-dir, "
            "not stated and not checked. Needs --predictor."
        ),
    )
    group.add_argument(
        "--prediction-weights",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Custom weights for the model, for instance a fine-tuned checkpoint: a file for a "
            "model that takes a checkpoint file (OpenFold3: --inference-ckpt-path), a directory "
            "for a model whose weights are a directory. It applies to --predictor MODEL run from "
            "here and to the OpenFold3 step without --predictor (binding-metrics-openfold "
            "names it --ckpt). The weights are identified by content (SHA-256) in the key of "
            "the prediction store and recorded in the results. A model whose runner cannot take "
            "custom weights is refused before anything runs. Cannot be combined with "
            "--prediction-dir: the weights are whatever made that output. Default: the model's "
            "own weights."
        ),
    )
    group.add_argument(
        "--prediction-cyclic",
        choices=OPENFOLD_CYCLIC_CHOICES,
        default=None,
        help=(
            "Whether the binder is given to the model as cyclic, for --predictor MODEL run from "
            "here. auto (default): when the binder has a head-to-tail bond (for OpenFold3 also "
            "only when it consists of standard residues: with a D-amino acid and N-methylated "
            "residues, 1CWA, the flag lowered ipTM from 0.91-0.92 to 0.78-0.81 in one complex, "
            "three seeds, so auto leaves such a binder linear); on: always; off: never. "
            "OpenFold3 (>= 0.4.5) and Boltz-2 get 'cyclic: true' on the binder chain, which only "
            "wraps its relative positions and does not enforce the closure bond; Protenix gets "
            "the head-to-tail and disulfide bonds as covalent_bonds; ColabFold has no such "
            "setting and refuses a value. For --predictor of3 it is the setting of "
            "--openfold-cyclic (both given with different values is an error). Needs "
            "--predictor."
        ),
    )
    group.add_argument(
        "--prediction-no-msa-server",
        action="store_true",
        help=(
            "Do not use an MSA server, for --predictor MODEL run from here: ColabFold runs "
            "single_sequence, Boltz-2 writes 'msa: empty', Protenix runs with --use_msa false, "
            "OpenFold3 as --openfold-no-msa-server. The accuracy for a natural receptor "
            "drops; no sequence leaves the machine. Needs --predictor."
        ),
    )
    group.add_argument(
        "--prediction-conda-env",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Conda environment that has the model, for --predictor MODEL run from here "
            "(conda run -n NAME). Default: the model's executable on PATH, that is the current "
            "environment; an empty string says the same. For --predictor of3 it is the "
            "setting of --openfold-conda-env (default openfold3) and wins over its default; "
            "both given with different values is an error. Needs --predictor."
        ),
    )
    group.add_argument(
        "--prediction-lock-threshold",
        type=lock_threshold_arg,
        default=None,
        metavar="ANGSTROM",
        help=(
            "Only for --prediction-mode score-lock: how far, in angstrom, a residue may move "
            "from the pinned template before the model pulls it back (the threshold of the "
            "forced template of Boltz-2). Default: 2.0, the choice of the Boltz-2 runner (Boltz-2 "
            "documents none). A runner or a mode that does not use it refuses it. Needs "
            "--predictor."
        ),
    )
    group.add_argument(
        "--prediction-binder-chain",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Chain ID of the binder inside the prediction, when it differs from the ID in "
            "the input (default: the input's ID). Needs --predictor."
        ),
    )
    group.add_argument(
        "--prediction-target-chain",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Chain ID of the receptor inside the prediction, when it differs from the ID "
            "in the input (default: the input's ID). Needs --predictor."
        ),
    )
    group.add_argument(
        "--prediction-cache",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Store of finished predictions, keyed by everything that changes the output. "
            f"Default: <output-dir>/{BATCH_CACHE_DIRNAME if batch else DEFAULT_CACHE_DIRNAME}. "
            "A run over the same input, options and model version reuses it and starts no "
            "model. Needs --predictor."
        ),
    )
    group.add_argument(
        "--rerun-predictions",
        action="store_true",
        help=(
            "Run each prediction again although the store has it (once per run). Outputs "
            "given with --prediction-dir are never replaced. Needs --predictor."
        ),
    )


#: Options that mean nothing without ``--predictor``, as (attribute, option string).
_NEEDS_PREDICTOR = (
    ("prediction_dir", "--prediction-dir"),
    ("prediction_mode", "--prediction-mode"),
    ("prediction_cyclic", "--prediction-cyclic"),
    ("prediction_no_msa_server", "--prediction-no-msa-server"),
    ("prediction_conda_env", "--prediction-conda-env"),
    ("prediction_lock_threshold", "--prediction-lock-threshold"),
    ("prediction_binder_chain", "--prediction-binder-chain"),
    ("prediction_target_chain", "--prediction-target-chain"),
    ("prediction_cache", "--prediction-cache"),
    ("rerun_predictions", "--rerun-predictions"),
)


def display_name(model: str) -> str:
    """The name of ``model`` for messages (``"Boltz-2"`` for ``"boltz2"``); the key if unknown."""
    spec = PARSERS.get(model)
    return spec.display_name if spec is not None else model


def has_runner(model: str) -> bool:
    """True when binding-metrics can run ``model`` itself (see ``RUNNERS``)."""
    return model in RUNNERS


def runner_class(model: str) -> Optional[type]:
    """The runner class registered for ``model`` (``RUNNERS``), or None when it has none.

    The module is imported here and not when this module is, so that a runner whose model is not
    installed costs nothing until it is asked for, and a test can patch the class.
    """
    target = RUNNERS.get(model)
    if target is None:
        return None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def modes_of(runner: Any) -> frozenset[str]:
    """The modes ``runner`` (a class or an object) can run for a complex structure.

    Its ``supported_modes`` (``PredictionRunner`` gives ``{"predict"}``), plus ``run_modes`` for a
    runner that names the modes it builds an input for by that older name (``Boltz2Runner``). An
    object that has neither (a stand-in that does not subclass ``PredictionRunner``) states no
    limit, so every mode counts.
    """
    declared = getattr(runner, "supported_modes", None)
    named = getattr(runner, "run_modes", None)
    if declared is None and named is None:
        return frozenset(MODES)
    return frozenset(declared or ()) | frozenset(named or ())


def default_mode_of(runner: Any) -> Optional[str]:
    """The mode ``runner`` is run in when the caller names none.

    Its ``default_mode``; when that is None (the ``PredictionRunner`` default), the default of the
    ``mode`` parameter of its ``make_request``; None when it has neither.
    """
    declared = getattr(runner, "default_mode", None)
    if declared:
        return str(declared)
    parameters = _request_parameters(runner)
    parameter = None if parameters is None else parameters.get("mode")
    if parameter is None or parameter.default in (inspect.Parameter.empty, None):
        return None
    return str(parameter.default)


def _request_parameters(runner: Any) -> Optional[Mapping[str, inspect.Parameter]]:
    """The parameters of ``runner.make_request`` by name; None when it takes ``**kwargs``."""
    try:
        parameters = inspect.signature(runner.make_request).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return parameters


def takes_setting(runner: Any, keyword: str) -> bool:
    """True when ``runner.make_request`` has the parameter ``keyword`` (or takes ``**kwargs``)."""
    parameters = _request_parameters(runner)
    return parameters is None or keyword in parameters


def runner_modes(model: str) -> frozenset[str]:
    """The modes the runner of ``model`` can run for a complex structure; empty without a runner."""
    cls = runner_class(model)
    return frozenset() if cls is None else modes_of(cls)


def runner_default_mode(model: str) -> Optional[str]:
    """The mode the runner of ``model`` is run in by default; None without a runner."""
    cls = runner_class(model)
    return None if cls is None else default_mode_of(cls)


def runner_weights_kinds() -> dict[str, str]:
    """Model to the kind of custom weights its runner takes, for the runners that take any.

    Read from the runner classes of ``RUNNERS`` (``supports_custom_weights`` and
    ``weights_kind``, class attributes: nothing is instantiated and no model started), so a
    runner added to ``RUNNERS`` appears here without a change. A runner module that cannot be
    imported is left out and logged.
    """
    kinds: dict[str, str] = {}
    for model in RUNNERS:
        try:
            cls = runner_class(model)
        except Exception as exc:  # noqa: BLE001 - one broken runner must not hide the others
            logger.warning("could not read the weights support of the %s runner: %s", model, exc)
            continue
        if getattr(cls, "supports_custom_weights", False):
            kinds[model] = getattr(cls, "weights_kind", "file")
    return kinds


def lock_threshold_arg(value: str) -> float:
    """``type`` of ``--prediction-lock-threshold``: a positive, finite number of angstrom."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of angstrom") from None
    if not (math.isfinite(number) and number > 0):
        raise argparse.ArgumentTypeError(f"must be a positive number of angstrom, got {value}")
    return number


def no_runner_message(model: str) -> str:
    """Why ``model`` cannot be run from here and what to do instead."""
    return (
        f"{display_name(model)} has no runner yet, so binding-metrics cannot start it: "
        f"run it yourself and pass its output with --prediction-dir DIR "
        f"(--predictor {model} reads it and never runs the model)"
    )


def check_prediction_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a combination of the prediction options that cannot work, before anything runs.

    Ends the program through ``parser.error`` (exit code 2) when a prediction option is given
    without ``--predictor``, when ``--predictor`` names a model that has no runner and no
    ``--prediction-dir`` is given, or when a setting is not one the model's runner has (a mode it
    does not run, ``--prediction-cyclic on`` for ColabFold, ``--prediction-lock-threshold``
    outside ``score-lock``, an ``--openfold-*`` option for a model other than OpenFold3, or the
    two spellings of one setting with different values; see ``resolve_prediction_options``). Ends
    with exit code 1 and a message when ``--prediction-dir`` is not an existing directory, or
    when ``--prediction-weights`` does not exist or is not the kind the model's runner takes.
    ``--prediction-weights`` with ``--prediction-dir`` is a usage error: the weights of an output
    you made are whatever produced it.
    """
    weights = getattr(args, "prediction_weights", None)
    if weights is not None and args.prediction_dir is not None:
        parser.error(weights_with_a_directory_message())
    if args.predictor is None:
        for attribute, option in _NEEDS_PREDICTOR:
            if getattr(args, attribute, None):
                parser.error(f"{option} needs --predictor (which model made the output?)")
    else:
        if args.prediction_dir is None and not has_runner(args.predictor):
            parser.error(no_runner_message(args.predictor))
        try:
            resolve_prediction_options(
                args.predictor,
                args.prediction_dir,
                prediction_mode=getattr(args, "prediction_mode", None),
                openfold_mode=getattr(args, "openfold_mode", "score"),
                openfold_cyclic=getattr(args, "openfold_cyclic", "auto"),
                prediction_cyclic=getattr(args, "prediction_cyclic", None),
                openfold_use_msa_server=not getattr(args, "openfold_no_msa_server", False),
                prediction_use_msa_server=not getattr(args, "prediction_no_msa_server", False),
                openfold_conda_env=getattr(args, "openfold_conda_env", None),
                prediction_conda_env=getattr(args, "prediction_conda_env", None),
                prediction_lock_threshold=getattr(args, "prediction_lock_threshold", None),
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.prediction_dir is not None and not Path(args.prediction_dir).is_dir():
            print(
                f"ERROR: --prediction-dir is not a directory: {args.prediction_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
    if weights is not None:
        try:
            check_weights_arg(weights, args.predictor)
        except ValueError as exc:
            print(f"ERROR: --prediction-weights: {exc}", file=sys.stderr)
            sys.exit(1)


def weights_with_a_directory_message() -> str:
    """Why ``--prediction-weights`` cannot go with ``--prediction-dir``."""
    return (
        "--prediction-weights cannot be combined with --prediction-dir: an output you made is "
        "read, never run, so the weights are whatever produced it (OpenFold3 records "
        "inference_ckpt_path and inference_ckpt_name in experiment_config.json, and they are shown "
        "in results['prediction']['weights'])"
    )


def model_refuses_mode(model: str, mode: str) -> bool:
    """True when the declared limits of the model (``Capabilities.modes``) do not list ``mode``.

    The pre-flight check refuses such a mode with the models that do have it, according to the
    policy ``--on-incompatible``, so the command line leaves it to that check and refuses only a
    mode that the model has and its runner does not run.
    """
    spec = PARSERS.get(model)
    declared = None if spec is None else spec.load_capabilities()
    return declared is not None and bool(declared.modes) and mode not in declared.modes


def unsupported_mode_message(model: str, mode: str) -> str:
    """Why ``mode`` cannot be run from here for ``model``, with the modes that can."""
    have = sorted(runner_modes(model))
    listed = " and ".join([", ".join(have[:-1]), have[-1]] if len(have) > 1 else have)
    return (
        f"--prediction-mode {mode} cannot be run from here for {display_name(model)}: its runner "
        f"runs {listed} on a complex structure. To use {mode}, run the model yourself "
        f"and read its output with --prediction-dir DIR --prediction-mode {mode}"
    )


#: The ``make_request`` keyword of a setting that only some runners have, and the option that
#: sets it. A runner without the keyword is given none and refuses the option.
_RUNNER_SETTING_OPTIONS = {
    "binder_cyclic": "--prediction-cyclic",
    "use_msa_server": "--prediction-no-msa-server",
    "lock_threshold_angstrom": "--prediction-lock-threshold",
    "weights": "--prediction-weights",
    "seeds": "--openfold-seeds",
    "on_unmappable_residue": "--on-unmappable-residue",
}


def unsupported_setting_message(model: str, keyword: str) -> str:
    """Why the option for ``keyword`` is refused for ``model``: its runner has no such setting."""
    option = _RUNNER_SETTING_OPTIONS.get(keyword, keyword)
    return (
        f"{option} is not available for {display_name(model)}: its runner has no '{keyword}' "
        "setting"
    )


def openfold_option_message(option: str, generic: str, model: str) -> str:
    """Why a legacy ``--openfold-*`` option is refused for a ``--predictor`` other than of3."""
    return (
        f"{option} sets how OpenFold3 is run and does not apply to --predictor {model} "
        f"({display_name(model)}); use {generic}"
    )


def _spelled(value: Any) -> Any:
    """A cyclic setting as the command line spells it (``on`` and ``off`` for True and False)."""
    return ("on" if value else "off") if isinstance(value, bool) else value


def _conflict_message(legacy: str, generic: str, legacy_value: Any, generic_value: Any) -> str:
    return (
        f"{legacy} {_spelled(legacy_value)} and {generic} {_spelled(generic_value)} set the same "
        "thing for --predictor of3 and disagree; give one of them"
    )


class PredictionOptions(NamedTuple):
    """The settings of the ``--predictor`` route once the two spellings are merged.

    ``mode`` is the mode the run has (None for an output read with ``--prediction-dir`` whose
    mode is not given); ``cyclic`` is ``"auto"``, True or False; ``use_msa_server`` a bool;
    ``conda_env`` the environment of the runner (None: the current one); ``lock_threshold`` the
    threshold of ``score-lock`` in angstrom, None for the runner's own.
    """

    mode: Optional[str]
    cyclic: bool | str
    use_msa_server: bool
    conda_env: Optional[str]
    lock_threshold: Optional[float]


def resolve_prediction_options(
    predictor: Optional[str],
    prediction_dir: Optional[Path],
    *,
    prediction_mode: Optional[str] = None,
    openfold_mode: str = "score",
    openfold_cyclic: bool | str = "auto",
    prediction_cyclic: Optional[bool | str] = None,
    openfold_use_msa_server: bool = True,
    prediction_use_msa_server: bool = True,
    openfold_conda_env: Optional[str] = None,
    prediction_conda_env: Optional[str] = None,
    prediction_lock_threshold: Optional[float] = None,
) -> PredictionOptions:
    """Merge the ``--openfold-*`` and ``--prediction-*`` spellings and check them for the runner.

    The ``--prediction-*`` options (``prediction_cyclic``, ``prediction_use_msa_server``,
    ``prediction_conda_env``, ``prediction_lock_threshold``) are the settings of the
    ``--predictor`` route for every model. The ``--openfold-*`` ones (``openfold_mode``,
    ``openfold_cyclic``, ``openfold_use_msa_server``, ``openfold_conda_env``) are those of the
    OpenFold3 step and, for ``predictor == "of3"``, the same settings under their older names.
    Nothing is checked or changed without ``predictor`` (the OpenFold3 step), and the
    ``--openfold-*`` values are returned as they are; with ``prediction_dir`` the model is not run,
    so a setting of a run is refused and the ``--openfold-*`` ones are ignored, as they always were.

    For a run from here:

    * ``--openfold-*`` options other than their defaults are refused for a model other than
      OpenFold3, naming the ``--prediction-*`` option;
    * for OpenFold3 the two spellings give one value; both given with different values is refused
      (``--openfold-conda-env openfold3`` is its default and counts as not given);
    * the mode (``--prediction-mode``, else ``--openfold-mode`` for OpenFold3, else the default
      of the runner) must be one the runner runs, unless the model does not have it at all
      (``model_refuses_mode``): the pre-flight check refuses that one, with the models that do;
    * a setting is passed on only when the keyword is in the runner's ``make_request``:
      ``binder_cyclic`` other than ``auto``, ``use_msa_server`` False and
      ``lock_threshold_angstrom`` are refused for a runner that has not; the lock threshold
      also needs the mode ``score-lock``.

    Returns:
        The merged settings; for the OpenFold3 step or an adopted output they are the
        ``--openfold-*`` values and the mode of ``effective_prediction_mode``.

    Raises:
        ValueError: A setting that the model's runner cannot take, naming the option and the
            model, or a value that is not valid.
    """
    wants_run = predictor is not None and prediction_dir is None
    generic_given = [
        option
        for option, given in (
            ("--prediction-cyclic", prediction_cyclic is not None),
            ("--prediction-no-msa-server", not prediction_use_msa_server),
            ("--prediction-conda-env", prediction_conda_env is not None),
            ("--prediction-lock-threshold", prediction_lock_threshold is not None),
        )
        if given
    ]
    if not wants_run:
        if generic_given and predictor is None:
            raise ValueError(f"{generic_given[0]} needs --predictor (which model made the output?)")
        if generic_given:
            raise ValueError(
                f"{generic_given[0]} cannot be combined with --prediction-dir: an output you made "
                "is read, never run"
            )
        return PredictionOptions(
            effective_prediction_mode(predictor, prediction_dir, prediction_mode, openfold_mode),
            check_openfold_cyclic(openfold_cyclic),
            bool(openfold_use_msa_server),
            openfold_conda_env,
            None,
        )

    runner = None if predictor is None else runner_class(predictor)
    if runner is None:
        raise ValueError(no_runner_message(str(predictor)))
    legacy_cyclic = check_openfold_cyclic(openfold_cyclic)
    generic_cyclic = None if prediction_cyclic is None else check_openfold_cyclic(prediction_cyclic)
    legacy_env_given = openfold_conda_env not in (None, OPENFOLD_DEFAULT_CONDA_ENV)
    if predictor == "of3":
        cyclic = legacy_cyclic if generic_cyclic is None else generic_cyclic
        if generic_cyclic is not None and legacy_cyclic not in ("auto", generic_cyclic):
            raise ValueError(
                _conflict_message(
                    "--openfold-cyclic", "--prediction-cyclic", legacy_cyclic, generic_cyclic
                )
            )
        if prediction_conda_env is None:
            conda_env = openfold_conda_env
        elif legacy_env_given and openfold_conda_env != prediction_conda_env:
            raise ValueError(
                _conflict_message(
                    "--openfold-conda-env",
                    "--prediction-conda-env",
                    openfold_conda_env,
                    prediction_conda_env,
                )
            )
        else:
            conda_env = prediction_conda_env
        use_msa_server = bool(openfold_use_msa_server) and bool(prediction_use_msa_server)
        mode = prediction_mode or openfold_mode
    else:
        for option, generic, given in (
            ("--openfold-cyclic", "--prediction-cyclic", legacy_cyclic != "auto"),
            ("--openfold-no-msa-server", "--prediction-no-msa-server", not openfold_use_msa_server),
            ("--openfold-conda-env", "--prediction-conda-env", legacy_env_given),
            ("--openfold-mode", "--prediction-mode", openfold_mode != "score"),
        ):
            if given:
                raise ValueError(openfold_option_message(option, generic, predictor))
        cyclic = "auto" if generic_cyclic is None else generic_cyclic
        conda_env = prediction_conda_env
        use_msa_server = bool(prediction_use_msa_server)
        mode = prediction_mode or default_mode_of(runner)

    if (
        mode is not None
        and mode not in modes_of(runner)
        and not model_refuses_mode(predictor, mode)
    ):
        raise ValueError(unsupported_mode_message(predictor, mode))
    if cyclic != "auto" and not takes_setting(runner, "binder_cyclic"):
        raise ValueError(unsupported_setting_message(predictor, "binder_cyclic"))
    if not use_msa_server and not takes_setting(runner, "use_msa_server"):
        raise ValueError(unsupported_setting_message(predictor, "use_msa_server"))
    if prediction_lock_threshold is not None:
        if not takes_setting(runner, "lock_threshold_angstrom"):
            raise ValueError(unsupported_setting_message(predictor, "lock_threshold_angstrom"))
        if mode != "score-lock":
            raise ValueError(
                f"--prediction-lock-threshold is the threshold of the mode score-lock, and the "
                f"run is in mode {mode}; add --prediction-mode score-lock"
            )
    return PredictionOptions(mode, cyclic, use_msa_server, conda_env, prediction_lock_threshold)


def check_predictor(
    predictor: Optional[str],
    prediction_dir: Optional[Path],
    prediction_mode: Optional[str] = None,
    prediction_weights: Optional[Path] = None,
    *,
    openfold_mode: str = "score",
    openfold_cyclic: bool | str = "auto",
    prediction_cyclic: Optional[bool | str] = None,
    openfold_use_msa_server: bool = True,
    prediction_use_msa_server: bool = True,
    openfold_conda_env: Optional[str] = None,
    prediction_conda_env: Optional[str] = None,
    prediction_lock_threshold: Optional[float] = None,
) -> PredictionOptions:
    """The Python-API counterpart of ``check_prediction_args``: raise before any step runs.

    Returns the merged settings of ``resolve_prediction_options``, which the pipelines use in
    place of the two spellings.

    Raises:
        ValueError: ``predictor`` is not a registered model, it has no runner and no
            ``prediction_dir`` is given, ``prediction_mode`` is not one of ``MODES`` or needs
            ``predictor``, ``prediction_weights`` is given with ``prediction_dir``, or a setting
            is not one the runner of the model has (``resolve_prediction_options``).
    """
    if prediction_weights is not None and prediction_dir is not None:
        raise ValueError(weights_with_a_directory_message())
    if prediction_mode is not None and prediction_mode not in MODES:
        raise ValueError(f"prediction_mode must be one of {MODES}, got {prediction_mode!r}")
    if predictor is None:
        if prediction_mode is not None:
            raise ValueError("prediction_mode needs a predictor (which model made the output?)")
    elif predictor not in PARSERS:
        raise ValueError(
            f"Unknown predictor {predictor!r}. Available: {', '.join(sorted(PARSERS))}"
        )
    return resolve_prediction_options(
        predictor,
        prediction_dir,
        prediction_mode=prediction_mode,
        openfold_mode=openfold_mode,
        openfold_cyclic=openfold_cyclic,
        prediction_cyclic=prediction_cyclic,
        openfold_use_msa_server=openfold_use_msa_server,
        prediction_use_msa_server=prediction_use_msa_server,
        openfold_conda_env=openfold_conda_env,
        prediction_conda_env=prediction_conda_env,
        prediction_lock_threshold=prediction_lock_threshold,
    )


def check_weights_arg(
    prediction_weights: Optional[str | Path], predictor: Optional[str] = None
) -> Optional[Path]:
    """Check ``--prediction-weights`` before any step runs; the path as an absolute ``Path``.

    The path must exist and be readable, and be the kind that the runner of the model takes (a
    file or a directory, from ``runner_weights_kinds``). A model whose runner takes no custom
    weights is not decided here: the pre-flight check refuses it with the runners that do. Without
    ``predictor`` the weights are for the OpenFold3 step.

    Raises:
        ValueError: ``capabilities.check_weights_path`` says what was wrong.
    """
    if prediction_weights is None:
        return None
    from binding_metrics.capabilities import check_weights_path

    kind = runner_weights_kinds().get(predictor or "of3")
    return check_weights_path(prediction_weights, kind)


def weights_description(extras: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The weights of a prediction, for ``results[...]["weights"]`` and the provenance.

    ``extras`` is ``record.extras`` (or a dict with the same keys). Custom weights (the
    ``weights`` entry that ``PredictionSession.record`` sets from the request) give ``custom``
    True, ``name`` (the file name), ``path``, ``kind``, ``sha256``, ``size`` and ``n_files``.
    Otherwise the checkpoint that the model recorded for its own default weights is described
    (``inference_ckpt_name`` and ``inference_ckpt_path`` of OpenFold3; ``custom`` False, ``sha256``
    and ``size`` None because the file was not read). None when nothing is known.
    """
    custom = extras.get("weights")
    if custom:
        return {"custom": True, "name": Path(str(custom.get("path", ""))).name or None, **custom}
    name, path = extras.get("inference_ckpt_name"), extras.get("inference_ckpt_path")
    if name or path:
        return {
            "custom": False,
            "name": None if name is None else str(name),
            "path": None if path is None else str(path),
            "sha256": None,
            "size": None,
        }
    return None


def output_weights(prediction_dir: str | Path) -> dict[str, Any]:
    """The checkpoint that an OpenFold3 run recorded in its output, as record extras.

    Reads ``experiment_config.json`` of the output directory through the adapter (the legacy
    OpenFold3 step has no record). Empty when the file is absent or has no checkpoint.
    """
    from binding_metrics.predictors.of3 import _run_provenance

    found = _run_provenance(Path(prediction_dir))
    return {key: value for key, value in found.items() if key.startswith("inference_ckpt")}


def effective_prediction_mode(
    predictor: Optional[str],
    prediction_dir: Optional[Path],
    prediction_mode: Optional[str],
    openfold_mode: str = "score",
) -> Optional[str]:
    """The mode a prediction step runs in, or None when it is not known.

    ``--prediction-mode`` wins. Without it, the OpenFold3 step (no ``--predictor``) and
    ``--predictor of3`` run from here use ``--openfold-mode``, so a run that did not use the new
    option behaves as before; any other model run from here uses the default mode of its runner
    (``default_mode``: ``score`` for Boltz-2, ``predict`` for ColabFold and Protenix); an output
    read with ``--prediction-dir`` was made elsewhere, and its mode is not known unless the user
    says it.
    """
    if prediction_mode is not None:
        return prediction_mode
    if predictor is None:
        return openfold_mode
    if prediction_dir is not None:
        return None
    if predictor == "of3":
        return openfold_mode
    return runner_default_mode(predictor)


# ---------------------------------------------------------------------------
# Requests, runners, sessions
# ---------------------------------------------------------------------------


def make_runner(predictor: str, conda_env: Optional[str] = None) -> Optional[Any]:
    """A runner for ``predictor``, or None when it has none.

    ``conda_env`` is the environment with the model (``--prediction-conda-env``, or
    ``--openfold-conda-env`` for OpenFold3); an empty string or None means the current
    environment, where the model's executable is looked up on PATH.
    """
    cls = runner_class(predictor)
    return None if cls is None else cls(conda_env or None)


def make_store(cache_dir: str | Path):
    """The ``PredictionStore`` rooted at ``cache_dir`` (made when the first entry is written)."""
    from binding_metrics.predictors.store import PredictionStore

    return PredictionStore(cache_dir)


def make_session(store, runner: Optional[Any], *, rerun: bool = False):
    """A ``PredictionSession`` on ``store`` that can run ``runner``'s model (none: it cannot)."""
    from binding_metrics.predictors.session import PredictionSession

    return PredictionSession(store, [runner] if runner is not None else None, rerun=rerun)


def make_request(
    predictor: str,
    sample_id: str,
    input_path: str | Path,
    *,
    binder_chain: str,
    receptor_chain: str,
    runner: Optional[Any] = None,
    adopt: bool = False,
    openfold_mode: str = "score",
    openfold_seeds=None,
    on_unmappable_residue: str = "error",
    openfold_cyclic: bool | str = "auto",
    openfold_use_msa_server: bool = True,
    prediction_mode: Optional[str] = None,
    prediction_weights=None,
    prediction_lock_threshold: Optional[float] = None,
):
    """The store request of one sample.

    With ``adopt`` False the runner builds it (every setting that changes the output written
    out, so two callers that mean one run share its key). With ``adopt`` True it describes
    outputs the user made: the model, the sample name, the content of the input file and the
    chain roles. The name is part of that key, so two samples with identical inputs adopt
    their own outputs and not one another's. ``prediction_mode`` (``--prediction-mode``) sets the
    mode of the request, and so its key; without it the mode is the one of
    ``effective_prediction_mode`` (``openfold_mode`` for OpenFold3, the default mode of the
    runner for another model) and, for an adopted output of another model, ``predict``, as before.

    The runner is called with the keywords of its ``make_request``. ``name``, the chain roles and
    ``mode`` always; ``seeds`` and ``on_unmappable_residue`` as given; ``binder_cyclic`` (when
    ``openfold_cyclic`` is not ``"auto"``), ``use_msa_server`` (when ``openfold_use_msa_server``
    is False), ``lock_threshold_angstrom``, and ``weights`` (a path, or the ``WeightsRef`` of
    ``PredictionStore.weights_reference``; by content in the key) only when they are not at their
    defaults, so a runner without the keyword is never called with it. ``openfold_cyclic`` and
    ``openfold_use_msa_server`` are the settings of the run, after ``resolve_prediction_options``
    merged them with the ``--prediction-*`` spelling. An adopted request has no weights: the
    weights of an output you made are not known.

    Raises:
        ValueError: ``runner`` is None and ``adopt`` is False, the runner has no keyword for a
            setting that is not at its default (the message names the option and the model), or
            the runner refuses a setting.
        OSError: The input file cannot be read.
    """
    if adopt:
        from binding_metrics.predictors.store import PredictionRequest

        return PredictionRequest(
            predictor,
            sample_id,
            mode=prediction_mode or (openfold_mode if predictor == "of3" else "predict"),
            input_path=input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
        )
    if runner is None:
        raise ValueError(no_runner_message(predictor))
    mode = prediction_mode or (openfold_mode if predictor == "of3" else default_mode_of(runner))
    cyclic = check_openfold_cyclic(openfold_cyclic)
    # (keyword, value, whether it is passed although it is the default)
    settings = (
        ("seeds", openfold_seeds, True),
        ("on_unmappable_residue", on_unmappable_residue, True),
        ("binder_cyclic", cyclic, cyclic != "auto"),
        ("use_msa_server", False, not openfold_use_msa_server),
        (
            "lock_threshold_angstrom",
            prediction_lock_threshold,
            prediction_lock_threshold is not None,
        ),
        ("weights", prediction_weights, prediction_weights is not None),
    )
    keywords: dict[str, Any] = {}
    for keyword, value, passed in settings:
        if not passed:
            continue
        if takes_setting(runner, keyword):
            keywords[keyword] = value
        elif keyword not in ("seeds", "on_unmappable_residue") or value not in (None, "error"):
            raise ValueError(unsupported_setting_message(predictor, keyword))
    if mode is not None:
        keywords["mode"] = mode
    return runner.make_request(
        input_path,
        name=sample_id,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        **keywords,
    )


def scored_seed_index(
    openfold_seeds: Optional[Sequence[int]],
    directory: Optional[str | Path] = None,
    query_name: Optional[str] = None,
) -> int:
    """The ``seed_index`` of the sample the pipeline scores: that of the first seed given.

    OpenFold3 writes one ``seed_<value>`` directory per seed, and the adapters count them in the
    numeric order of the values, so ``--openfold-seeds 9 3`` scores seed 9, the second directory.
    With ``directory`` and ``query_name`` the position is looked up among the directories that
    exist, which also holds when an earlier run left other seed directories in the same output
    directory; without them (the prediction store, where every request has a directory of its
    own) it is the position among the seeds given.

    Returns:
        1-based position; 1 when no seeds were given (the lowest seed, the only one by default).
    """
    if not openfold_seeds:
        return 1
    first = int(openfold_seeds[0])
    if directory is not None and query_name is not None:
        from binding_metrics.predictors.of3 import _seed_directories

        names = [d.name[len("seed_") :] for d in _seed_directories(Path(directory) / query_name)]
        if str(first) in names:
            return names.index(str(first)) + 1
    return sorted({int(seed) for seed in openfold_seeds}).index(first) + 1


def scored_seed_kwargs(
    openfold_seeds: Optional[Sequence[int]], directory: str | Path, query_name: str
) -> dict:
    """``{"seed": index}`` for ``compute_openfold_metrics`` when seeds were given, else ``{}``.

    Without ``--openfold-seeds`` the call is the one that always scored the first sample of the
    first seed directory, so a function that predates the option is called as before.
    """
    if not openfold_seeds:
        return {}
    return {"seed": scored_seed_index(openfold_seeds, directory, query_name)}


def prediction_chain_map(
    prediction_binder_chain: Optional[str],
    prediction_target_chain: Optional[str],
    binder_chain: str,
    receptor_chain: str,
) -> Optional[dict[str, str]]:
    """Model chain ID to the input's chain ID, from ``--prediction-binder-chain`` and ``-target-``.

    Returns None when neither option is given (the chain IDs already agree).
    """
    mapping: dict[str, str] = {}
    if prediction_binder_chain:
        mapping[prediction_binder_chain] = binder_chain
    if prediction_target_chain:
        mapping[prediction_target_chain] = receptor_chain
    return mapping or None


def reference_for(predictor: str, input_path: Path) -> Optional[Path]:
    """The structure the binder RMSD is measured against: the input pose, for every model.

    The binder is placed by the model in every mode: a prediction from sequences does not see the
    pose, and a template holds one chain and no inter-chain geometry (OpenFold3, Boltz-2 ``score``
    and ``refold``), so the binder RMSD in the receptor frame (``binder_ca_rmsd``) says how far the
    predicted pose is from the input pose, in ``predict``, ``refold`` and ``score`` alike. In
    ``score`` the binder also has its own fold as a template, so the number is less free of the
    input than in ``refold``, and in ``predict`` the model has seen neither chain.
    ``delta_com_angstrom`` of the EvoBind adversarial check is the displacement of the binder
    centre of mass between the same two poses. ``score-lock`` pins the pose, so the number then
    measures how well the pin held.

    When the receptor frame cannot be built (a chain missing from the prediction under the input's
    ID, or another number of C-alpha atoms) ``binder_ca_rmsd`` stays NaN and the reason of the
    block says why and names the model.

    Args:
        predictor: The model; unused, kept so that a caller can ask per model.
        input_path: The complex structure of the sample.
    """
    return input_path


def runner_chain_map(runner: Optional[Any], request) -> Optional[dict[str, str]]:
    """The chain IDs of the runner's prediction as ``{prediction ID: input ID}``, or None.

    From ``runner.output_chain_map(request)`` (``PredictionRunner`` gives None: the prediction
    keeps the input's IDs; ``ColabFoldRunner`` names the receptor ``A`` and the binder ``B``). A
    runner object without the method counts as None.
    """
    method = getattr(runner, "output_chain_map", None)
    if method is None:
        return None
    mapping = method(request)
    return dict(mapping) if mapping else None


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def record_binder_cyclic(
    block: dict,
    input_path: str | Path,
    binder_chain: str,
    openfold_cyclic: bool | str = "auto",
    conda_env: Optional[str] = None,
) -> None:
    """Add what the OpenFold3 query did with the binder's ``cyclic`` flag to a result block.

    Adds ``binder_cyclic`` (bool: the binder chain was sent as ``"cyclic": true``) and, when a
    head-to-tail binder was left linear because the OpenFold3 version is too old or unreadable,
    the reason under ``reason`` (joined to an existing one). The query builders decide the same
    way from the same input; this repeats the decision without logging it again, because the
    functions that run OpenFold3 return a path and nothing else. The block of a model that
    did not run here (an adopted output) is not given the key: nothing is known of its query.

    Nothing is added when the decision cannot be made (``True`` with an OpenFold3 that is too
    old, which the run itself refused), and a failure is logged.
    """
    from binding_metrics.metrics.openfold import decide_binder_cyclic

    try:
        decision = decide_binder_cyclic(
            input_path,
            binder_chain,
            check_openfold_cyclic(openfold_cyclic),
            conda_env=conda_env or None,
            log=False,
        )
    except ValueError as e:
        logger.warning("  [warning] binder_cyclic not recorded: %s", e)
        return
    block["binder_cyclic"] = decision.cyclic
    if decision.reason:
        merge_reason(block, {"reason": decision.reason}, "binder_cyclic")


def record_templates(block: dict, predictions_dir: Optional[str | Path], query_name: str) -> None:
    """Add what became of the templates of an OpenFold3 run to a result block.

    OpenFold3 0.5.0 goes on without a template when it cannot use one (the MSA server replaced
    its alignment, or its preprocessing failed), exits with status 0, and then a ``score`` result
    is a template-free prediction with the same keys. This adds ``templates``, ``{chain ID:
    {"requested", "source", "used", "cause", "detail", "entry_ids"}}`` read from the output
    (see ``binding_metrics.metrics._openfold_templates``), and, when a chain asked for a
    template and got none, says so in ``reason`` (joined to an existing one). The CSV row gets a
    column per entry, ``<prefix>_templates_<chain>_used`` and so on.

    Nothing is added when the output has no ``inference_query_set.json`` (an output made by
    another tool or version) or when it cannot be read.
    """
    if predictions_dir is None:
        return
    from binding_metrics.metrics._openfold_templates import (
        describe_missing,
        read_template_accounting,
    )

    try:
        chains = read_template_accounting(predictions_dir, query_name)
    except Exception as e:  # noqa: BLE001 - provenance of a finished run must not fail it
        logger.warning("  [warning] template accounting not recorded: %s", e)
        return
    if not chains:
        return
    block["templates"] = chains
    sentence = describe_missing(chains)
    if sentence:
        merge_reason(block, {"reason": sentence}, "templates")


def _cache_block(session, request, *, adopted: bool = False) -> dict[str, Any]:
    """The session's counters and the key of the store entry, for ``results["prediction"]``.

    An output adopted from ``--prediction-dir`` is stored under the key of the request with its
    name (``for_adoption``), so two samples that share an input file have two entries.
    """
    key = request.for_adoption().key() if adopted else request.key()
    return {**session.stats(), "request_key": key}


def _evobind_score_of(record, binder_chain: str, receptor_chain: str) -> dict:
    """Primary EvoBind score of a record, in the input's chain IDs (its ``chain_map`` applied)."""
    from binding_metrics.metrics.evobind import compute_evobind_score_from_record

    return compute_evobind_score_from_record(
        record,
        binder_chain,
        receptor_chain,
        interface_cutoff_angstrom=_INTERFACE_CUTOFF_ANGSTROM,
    )


def error_text(error: BaseException, model: str, runner: Optional[Any] = None) -> str:
    """The text of a failed prediction for ``results["prediction"]["error"]``.

    A model that cannot be started here gets what was looked for (``runner.unavailable_reason()``
    when the runner has it: the conda environment is named) and the ways to start it: its
    executable on PATH, the conda environment option, or an output you made. Any other error is
    its own text.
    """
    from binding_metrics.predictors.store import PredictionUnavailableError

    text = str(error)
    if isinstance(error, PredictionUnavailableError) and "cannot be started" in text:
        why = getattr(runner, "unavailable_reason", None)
        reason = why() if callable(why) else None
        if reason:
            text += f"\nWhy: {reason}"
        env = "--openfold-conda-env" if model == "of3" else "--prediction-conda-env"
        text += (
            f"\nHint: put the executable of {display_name(model)} on PATH, or name the conda "
            f"environment that has it with {env} NAME, or read an output you made with "
            "--prediction-dir DIR"
        )
    return text


def missing_chains_reason(record, binder_chain: str, receptor_chain: str) -> Optional[str]:
    """Why the metrics cannot find the chains in the prediction, or None when both are there.

    The model names the chains of its prediction itself (ColabFold: ``A`` and ``B``), or the output
    of ``--prediction-dir`` uses other IDs than the input, and every metric that needs a chain by
    its ID in the input then reports a mismatch that does not say whose chains are meant. The
    sentence names the model and the option that fixes it. It is best effort: a structure that
    cannot be read gives None (the metrics say why).
    """
    if record.structure_path is None:
        return None
    try:
        present = sorted({str(chain) for chain in record.atoms().chain_id})
    except Exception as exc:  # noqa: BLE001 - the metrics report an unreadable structure themselves
        logger.debug("chain IDs of the %s prediction not read: %s", record.model, exc)
        return None
    missing = [chain for chain in (binder_chain, receptor_chain) if chain not in present]
    if not missing:
        return None
    return (
        f"the prediction has chains {', '.join(present)} and no chain "
        f"{' or '.join(repr(chain) for chain in missing)} of the input; it names its chains "
        "itself: give --prediction-binder-chain and --prediction-target-chain "
        "(the IDs it uses)"
    )


def _analyse(
    session,
    request,
    *,
    input_path: Path,
    binder_chain: str,
    receptor_chain: str,
    chain_map: Optional[Mapping[str, str]],
    reference_path: Optional[Path],
    adopted: bool = False,
    seed_index: int = 1,
    mode: Optional[str] = None,
    runner: Optional[Any] = None,
) -> tuple[dict, dict]:
    """Every consumer of one prediction, all reading the record the session parsed once.

    ``runner`` is the runner of the session, asked why the model cannot be started when it cannot.
    """
    from binding_metrics.metrics.evobind import compute_evobind_adversarial_from_records
    from binding_metrics.metrics.prediction import summarize_prediction
    from binding_metrics.predictors.store import PredictionFailedError, PredictionUnavailableError

    try:
        record = session.record(request, seed_index=seed_index, chain_map=chain_map)
    except (PredictionFailedError, PredictionUnavailableError) as error:
        logger.warning("  [warning] %s prediction failed: %s", display_name(request.model), error)
        return {
            "model": request.model,
            "mode": mode,
            "error": error_text(error, request.model, runner),
            "cache": _cache_block(session, request, adopted=adopted),
        }, {}

    block = summarize_prediction(
        record,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        reference_structure_path=reference_path,
    )
    chains = missing_chains_reason(record, binder_chain, receptor_chain)
    if chains:
        merge_reason(block, {"reason": chains}, display_name(request.model))
    if record.structure_path is not None:
        try:
            evobind = _evobind_score_of(record, binder_chain, receptor_chain)
            merge_reason(block, evobind, "evobind")
            block.update(evobind)
        except Exception as e:  # noqa: BLE001 - per-score isolation; see evobind_error
            logger.warning("  [warning] EvoBind score failed: %s", e)
            block["evobind_error"] = str(e)

        # Does the prediction agree with the input pose? A large COM shift means the model
        # puts the binder elsewhere, so the design pose is not supported by the prediction.
        try:
            adversarial = compute_evobind_adversarial_from_records(
                input_path, record, binder_chain, receptor_chain
            )
            merge_reason(block, adversarial, "evobind adversarial")
            block.update(adversarial)
        except Exception as e:  # noqa: BLE001 - per-check isolation; see adversarial_error
            logger.warning("  [warning] EvoBind adversarial check failed: %s", e)
            block["adversarial_error"] = str(e)

    block["mode"] = mode
    if record.model == "of3":
        record_templates(block, getattr(record.files, "directory", None), record.name)
    weights = weights_description(record.extras)
    if weights is not None:
        block["weights"] = weights
    block["cache"] = _cache_block(session, request, adopted=adopted)
    provenance: dict[str, Any] = {}
    checkpoint = record.extras.get("inference_ckpt_name")
    if record.model == "of3" and checkpoint:
        provenance["openfold3_checkpoint"] = str(checkpoint)
    if weights is not None:
        provenance["prediction_weights"] = weights
    return block, provenance


def run_prediction_step(
    session,
    request,
    *,
    input_path: Path,
    binder_chain: str,
    receptor_chain: str,
    prediction_dir: Optional[Path] = None,
    prediction_binder_chain: Optional[str] = None,
    prediction_target_chain: Optional[str] = None,
    reference_path: Optional[Path] = None,
    seed_index: int = 1,
    mode: Optional[str] = None,
    default_chain_map: Optional[Mapping[str, str]] = None,
    runner: Optional[Any] = None,
) -> tuple[dict, dict]:
    """The prediction step for one sample; never raises.

    Args:
        session: The sample's ``PredictionSession``.
        request: The sample's request (``make_request``).
        input_path: The complex structure of the sample: the design pose for the adversarial
            check.
        binder_chain, receptor_chain: Chain IDs in the input.
        prediction_dir: Outputs the user made; they are adopted into the store, and the model
            never runs.
        prediction_binder_chain, prediction_target_chain: Chain IDs inside the prediction, when
            they differ from the input's.
        reference_path: Structure for ``binder_ca_rmsd`` (``reference_for``).
        seed_index: Position of the seed directory to read (``scored_seed_index``).
        mode: The mode the prediction was made in (``effective_prediction_mode``); recorded as
            ``mode`` in the block, None when it is not known.
        default_chain_map: The chain IDs of the prediction as ``{prediction ID: input ID}`` when
            the runner names them itself (``runner_chain_map``). It is used when neither
            ``prediction_binder_chain`` nor ``prediction_target_chain`` is given, which take
            precedence.
        runner: The runner of the session; when the model cannot be started, its
            ``unavailable_reason()`` (if it has one) is added to the error text.

    Returns:
        ``(block, provenance)``: the value of ``results["prediction"]`` and the provenance keys
        this prediction adds (``openfold3_checkpoint`` when the record names its checkpoint).
        The block is ``{"model", "error", "cache"}`` when the prediction failed, cannot be run
        here, or an unexpected error stopped the step.
    """
    try:
        if prediction_dir is not None:
            session.adopt(request, prediction_dir)
        return _analyse(
            session,
            request,
            input_path=input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            chain_map=prediction_chain_map(
                prediction_binder_chain, prediction_target_chain, binder_chain, receptor_chain
            )
            or (dict(default_chain_map) if default_chain_map else None),
            reference_path=reference_path,
            adopted=prediction_dir is not None,
            seed_index=seed_index,
            mode=mode,
            runner=runner,
        )
    except Exception as e:  # noqa: BLE001 - per-metric isolation; recorded in results["prediction"]
        logger.warning("  [warning] %s prediction failed: %s", display_name(request.model), e)
        traceback.print_exc()
        return {
            "model": request.model,
            "mode": mode,
            "error": str(e),
            "cache": _cache_block(session, request, adopted=prediction_dir is not None),
        }, {}


def run_single_prediction(
    predictor: str,
    input_path: Path,
    output_dir: Path,
    sample_id: str,
    *,
    binder_chain: Optional[str],
    receptor_chain: Optional[str],
    prediction_dir: Optional[Path] = None,
    prediction_binder_chain: Optional[str] = None,
    prediction_target_chain: Optional[str] = None,
    prediction_cache: Optional[Path] = None,
    rerun_predictions: bool = False,
    openfold_mode: str = "score",
    openfold_conda_env: Optional[str] = None,
    openfold_seeds=None,
    on_unmappable_residue: str = "error",
    openfold_cyclic: bool | str = "auto",
    openfold_use_msa_server: bool = True,
    prediction_mode: Optional[str] = None,
    prediction_weights: Optional[Path] = None,
    prediction_lock_threshold: Optional[float] = None,
) -> tuple[dict, dict]:
    """The whole prediction step of ``run_pipeline``: store, session, request, consumers.

    Builds one session for the sample on the store in ``prediction_cache`` (default
    ``<output_dir>/predictions``) and calls ``run_prediction_step``. Never raises.
    ``prediction_weights`` are custom weights for the model: hashed once with the cache in the
    store root (``PredictionStore.weights_reference``), part of the request key, and recorded in
    ``block["weights"]`` and ``provenance["prediction_weights"]``; ignored for an adopted output.
    ``openfold_cyclic``, ``openfold_use_msa_server`` and ``openfold_conda_env`` are the settings
    of the run: the pipelines have merged them with the ``--prediction-*`` spelling
    (``resolve_prediction_options``). ``prediction_lock_threshold`` is the threshold of
    ``score-lock`` in angstrom (None: the runner's own). The runner's ``output_chain_map`` names
    the chains of its prediction, unless ``prediction_binder_chain`` or ``prediction_target_chain``
    is given.

    Returns:
        ``(block, provenance)`` as ``run_prediction_step``; the block is ``{"skipped": True}``
        when a chain is unknown, and ``{"model", "error"}`` when the request cannot be made
        (an unreadable input, a runner that refuses a setting).
    """
    if not binder_chain or not receptor_chain:
        logger.warning(
            "  [warning] Prediction requires --peptide-chain and --receptor-chain "
            "(or auto-detect); skipping."
        )
        return {"skipped": True}, {}
    mode = effective_prediction_mode(predictor, prediction_dir, prediction_mode, openfold_mode)
    try:
        adopt = prediction_dir is not None
        runner = None if adopt else make_runner(predictor, openfold_conda_env)
        store = make_store(prediction_cache or Path(output_dir) / DEFAULT_CACHE_DIRNAME)
        session = make_session(store, runner, rerun=rerun_predictions)
        weights = None
        if prediction_weights is not None and not adopt:
            weights = store.weights_reference(
                prediction_weights, expect=runner_weights_kinds().get(predictor)
            )
        request = make_request(
            predictor,
            sample_id,
            input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            runner=runner,
            adopt=adopt,
            openfold_mode=openfold_mode,
            openfold_seeds=openfold_seeds,
            on_unmappable_residue=on_unmappable_residue,
            openfold_cyclic=openfold_cyclic,
            openfold_use_msa_server=openfold_use_msa_server,
            prediction_mode=prediction_mode,
            prediction_weights=weights,
            prediction_lock_threshold=prediction_lock_threshold,
        )
        default_chain_map = None if adopt else runner_chain_map(runner, request)
    except Exception as e:  # noqa: BLE001 - per-metric isolation; recorded in results["prediction"]
        logger.warning("  [warning] %s prediction failed: %s", display_name(predictor), e)
        traceback.print_exc()
        return {"model": predictor, "mode": mode, "error": str(e)}, {}
    block, provenance = run_prediction_step(
        session,
        request,
        input_path=input_path,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        prediction_dir=prediction_dir,
        prediction_binder_chain=prediction_binder_chain,
        prediction_target_chain=prediction_target_chain,
        reference_path=reference_for(predictor, input_path),
        # an adopted output is the user's own: its seeds are not the ones given here
        seed_index=1 if adopt else scored_seed_index(openfold_seeds),
        mode=mode,
        default_chain_map=default_chain_map,
        runner=runner,
    )
    if predictor == "of3" and not adopt and not block.get("error"):
        record_binder_cyclic(block, input_path, binder_chain, openfold_cyclic, openfold_conda_env)
    return block, provenance
