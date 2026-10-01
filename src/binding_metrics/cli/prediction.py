"""The structure-prediction step of ``binding-metrics-run`` and ``binding-metrics-batch``.

With ``--predictor MODEL`` the step of the pipeline that ``--metrics openfold`` names does not
call OpenFold3 directly. It builds one ``PredictionSession`` per sample on a ``PredictionStore``,
so a model runs at most once for all the metrics that read its output (the confidence scalars,
the interface PAE and PDE, the EvoBind score and the adversarial check), and writes
``results["prediction"]``. Without ``--predictor`` nothing here runs and the OpenFold step keeps
its own code path and ``results["openfold"]``.

What the step does for one sample, in order::

    request = <runner>.make_request(...)      # a run: OpenFold3 today
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

Only OpenFold3 has a runner (``RUNNERS``). ``check_prediction_args`` refuses any other model
without ``--prediction-dir`` while the command line is checked, before anything runs.

The store lives in ``--prediction-cache DIR``; the default is ``<output-dir>/predictions`` for a
single run and ``<output-dir>/_predictions`` for a batch (the underscore keeps it apart from a
sample called "predictions"). A batch shares one store root, so a second run over the same
samples finds every prediction and starts no model.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from binding_metrics.capabilities import MODES
from binding_metrics.cli import (
    check_openfold_cyclic,
    merge_reason,
    openfold_cyclic_kwargs,
    openfold_msa_server_kwargs,
)
from binding_metrics.predictors.registry import PARSERS

logger = logging.getLogger("binding_metrics.cli.prediction")

#: Models that have a runner: ``"module:Class"`` of a ``PredictionRunner`` whose constructor
#: takes the conda environment. Imported when a runner is made, so a test can patch the class.
RUNNERS: dict[str, str] = {"of3": "binding_metrics.predictors.of3_runner:OpenFold3Runner"}

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
            "it. Only of3 (OpenFold3) can be run from here; the others need "
            "--prediction-dir. Default: not given, the step runs OpenFold3 and writes "
            "results['openfold']."
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
            "It is checked "
            "against what the model supports before anything runs, and recorded. Default: for "
            "--predictor of3 run from here, the value of --openfold-mode; for an output read "
            "with --prediction-dir, not stated and not checked. Needs --predictor."
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


def runner_weights_kinds() -> dict[str, str]:
    """Model to the kind of custom weights its runner takes, for the runners that take any.

    Read from the runner classes of ``RUNNERS`` (``supports_custom_weights`` and
    ``weights_kind``, class attributes: nothing is instantiated and no model started), so a
    runner added to ``RUNNERS`` appears here without a change. A runner module that cannot be
    imported is left out and logged.
    """
    kinds: dict[str, str] = {}
    for model, target in RUNNERS.items():
        module_name, class_name = target.split(":")
        try:
            runner_class = getattr(importlib.import_module(module_name), class_name)
        except Exception as exc:  # noqa: BLE001 - one broken runner must not hide the others
            logger.warning("could not read the weights support of the %s runner: %s", model, exc)
            continue
        if getattr(runner_class, "supports_custom_weights", False):
            kinds[model] = getattr(runner_class, "weights_kind", "file")
    return kinds


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
    without ``--predictor``, or when ``--predictor`` names a model that has no runner and no
    ``--prediction-dir`` is given. Ends with exit code 1 and a message when ``--prediction-dir``
    is not an existing directory, or when ``--prediction-weights`` does not exist or is not the
    kind the model's runner takes. ``--prediction-weights`` with ``--prediction-dir`` is a usage
    error: the weights of an output you made are whatever produced it.
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
        if args.prediction_dir is None and getattr(args, "prediction_mode", None) == "predict":
            parser.error(predict_needs_a_directory_message(args.predictor))
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


def predict_needs_a_directory_message(model: str) -> str:
    """Why ``--prediction-mode predict`` cannot be run from here."""
    return (
        f"--prediction-mode predict cannot be run from here: a run predicts the complex with its "
        f"structure as template (score or refold). Read an output made from sequences with "
        f"--prediction-dir DIR (--predictor {model})"
    )


def check_predictor(
    predictor: Optional[str],
    prediction_dir: Optional[Path],
    prediction_mode: Optional[str] = None,
    prediction_weights: Optional[Path] = None,
) -> None:
    """The Python-API counterpart of ``check_prediction_args``: raise before any step runs.

    Raises:
        ValueError: ``predictor`` is not a registered model, it has no runner and no
            ``prediction_dir`` is given, ``prediction_mode`` is not one of ``MODES`` or needs
            ``predictor``, it is ``predict`` for a run from here, or ``prediction_weights`` is
            given with ``prediction_dir``.
    """
    if prediction_weights is not None and prediction_dir is not None:
        raise ValueError(weights_with_a_directory_message())
    if prediction_mode is not None and prediction_mode not in MODES:
        raise ValueError(f"prediction_mode must be one of {MODES}, got {prediction_mode!r}")
    if predictor is None:
        if prediction_mode is not None:
            raise ValueError("prediction_mode needs a predictor (which model made the output?)")
        return
    if predictor not in PARSERS:
        raise ValueError(
            f"Unknown predictor {predictor!r}. Available: {', '.join(sorted(PARSERS))}"
        )
    if prediction_dir is None and not has_runner(predictor):
        raise ValueError(no_runner_message(predictor))
    if prediction_dir is None and prediction_mode == "predict":
        raise ValueError(predict_needs_a_directory_message(predictor))


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
    option behaves as before; an output read with ``--prediction-dir`` was made elsewhere, and
    its mode is not known unless the user says it.
    """
    if prediction_mode is not None:
        return prediction_mode
    if predictor is None:
        return openfold_mode
    if prediction_dir is None and predictor == "of3":
        return openfold_mode
    return None


# ---------------------------------------------------------------------------
# Requests, runners, sessions
# ---------------------------------------------------------------------------


def make_runner(predictor: str, conda_env: Optional[str] = None) -> Optional[Any]:
    """A runner for ``predictor``, or None when it has none.

    ``conda_env`` is the environment with the model (``--openfold-conda-env``); an empty string
    means the current environment, as the option documents.
    """
    target = RUNNERS.get(predictor)
    if target is None:
        return None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)(conda_env or None)


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
):
    """The store request of one sample.

    With ``adopt`` False the runner builds it (every setting that changes the output written
    out, so two callers that mean one run share its key). With ``adopt`` True it describes
    outputs the user made: the model, the sample name, the content of the input file and the
    chain roles. The name is part of that key, so two samples with identical inputs adopt
    their own outputs and not one another's. ``prediction_mode`` (``--prediction-mode``) sets the
    mode of the request, and so its key; without it the mode is ``openfold_mode`` for OpenFold3
    and ``predict`` for an adopted output of another model, as before. ``prediction_weights`` (a
    path, or the ``WeightsRef`` of ``PredictionStore.weights_reference``) goes to the runner's
    ``make_request`` as ``weights`` and so into the key by content; it is left out when None, and
    an adopted request has none (the weights of an output you made are not known).

    Raises:
        ValueError: ``runner`` is None and ``adopt`` is False, or the runner refuses a setting.
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
    return runner.make_request(
        input_path,
        name=sample_id,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        mode=prediction_mode or openfold_mode,
        seeds=openfold_seeds,
        on_unmappable_residue=on_unmappable_residue,
        **openfold_cyclic_kwargs(openfold_cyclic),
        **openfold_msa_server_kwargs(openfold_use_msa_server),
        **({} if prediction_weights is None else {"weights": prediction_weights}),
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
    """The structure the binder RMSD is measured against: the input pose, for OpenFold3.

    Both modes of OpenFold3 place the binder themselves, because a template holds one chain and
    no inter-chain geometry, so the binder RMSD in the receptor frame (``binder_ca_rmsd``) says
    how far the predicted pose is from the input pose, in ``score`` as in ``refold`` mode. In
    ``score`` the binder also has its own fold as a template, so the number is less free of
    the input than in ``refold``. ``delta_com_angstrom`` of the EvoBind adversarial check is the
    displacement of the binder centre of mass between the same two poses.
    """
    return input_path if predictor == "of3" else None


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
) -> tuple[dict, dict]:
    """Every consumer of one prediction, all reading the record the session parsed once."""
    from binding_metrics.metrics.evobind import compute_evobind_adversarial_from_records
    from binding_metrics.metrics.prediction import summarize_prediction
    from binding_metrics.predictors.store import PredictionFailedError, PredictionUnavailableError

    try:
        record = session.record(request, seed_index=seed_index, chain_map=chain_map)
    except (PredictionFailedError, PredictionUnavailableError) as error:
        logger.warning("  [warning] Prediction failed: %s", error)
        return {
            "model": request.model,
            "mode": mode,
            "error": str(error),
            "cache": _cache_block(session, request, adopted=adopted),
        }, {}

    block = summarize_prediction(
        record,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        reference_structure_path=reference_path,
    )
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
            ),
            reference_path=reference_path,
            adopted=prediction_dir is not None,
            seed_index=seed_index,
            mode=mode,
        )
    except Exception as e:  # noqa: BLE001 - per-metric isolation; recorded in results["prediction"]
        logger.warning("  [warning] Prediction failed: %s", e)
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
) -> tuple[dict, dict]:
    """The whole prediction step of ``run_pipeline``: store, session, request, consumers.

    Builds one session for the sample on the store in ``prediction_cache`` (default
    ``<output_dir>/predictions``) and calls ``run_prediction_step``. Never raises.
    ``prediction_weights`` are custom weights for the model: hashed once with the cache in the
    store root (``PredictionStore.weights_reference``), part of the request key, and recorded in
    ``block["weights"]`` and ``provenance["prediction_weights"]``; ignored for an adopted output.

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
        )
    except Exception as e:  # noqa: BLE001 - per-metric isolation; recorded in results["prediction"]
        logger.warning("  [warning] Prediction failed: %s", e)
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
    )
    if predictor == "of3" and not adopt and not block.get("error"):
        record_binder_cyclic(block, input_path, binder_chain, openfold_cyclic, openfold_conda_env)
    return block, provenance
