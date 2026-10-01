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
keys merged as the OpenFold step merges them, and ``cache``: the counters of
``PredictionSession.stats()`` and ``request_key``, the name of the store entry. A prediction that
failed, or that cannot be run here, gives ``{"model": ..., "error": <reason>, "cache": ...}``; the
sample goes on with its other steps.

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
    is not an existing directory.
    """
    if args.predictor is None:
        for attribute, option in _NEEDS_PREDICTOR:
            if getattr(args, attribute, None):
                parser.error(f"{option} needs --predictor (which model made the output?)")
        return
    if args.prediction_dir is None and not has_runner(args.predictor):
        parser.error(no_runner_message(args.predictor))
    if args.prediction_dir is not None and not Path(args.prediction_dir).is_dir():
        print(f"ERROR: --prediction-dir is not a directory: {args.prediction_dir}", file=sys.stderr)
        sys.exit(1)


def check_predictor(predictor: Optional[str], prediction_dir: Optional[Path]) -> None:
    """The Python-API counterpart of ``check_prediction_args``: raise before any step runs.

    Raises:
        ValueError: ``predictor`` is not a registered model, or it has no runner and no
            ``prediction_dir`` is given.
    """
    if predictor is None:
        return
    if predictor not in PARSERS:
        raise ValueError(
            f"Unknown predictor {predictor!r}. Available: {', '.join(sorted(PARSERS))}"
        )
    if prediction_dir is None and not has_runner(predictor):
        raise ValueError(no_runner_message(predictor))


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
):
    """The store request of one sample.

    With ``adopt`` False the runner builds it (every setting that changes the output written
    out, so two callers that mean one run share its key). With ``adopt`` True it describes
    outputs the user made: the model, the sample name, the content of the input file and the
    chain roles. The name is part of that key, so two samples with identical inputs adopt
    their own outputs and not one another's.

    Raises:
        ValueError: ``runner`` is None and ``adopt`` is False, or the runner refuses a setting.
        OSError: The input file cannot be read.
    """
    if adopt:
        from binding_metrics.predictors.store import PredictionRequest

        return PredictionRequest(
            predictor,
            sample_id,
            mode=openfold_mode if predictor == "of3" else "predict",
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
        mode=openfold_mode,
        seeds=openfold_seeds,
        on_unmappable_residue=on_unmappable_residue,
        **openfold_cyclic_kwargs(openfold_cyclic),
        **openfold_msa_server_kwargs(openfold_use_msa_server),
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


def reference_for(predictor: str, openfold_mode: str, input_path: Path) -> Optional[Path]:
    """The structure the binder RMSD is measured against: the input pose in refold mode only.

    In score mode the pipeline reports the displacement of the binder centre of mass instead
    (``delta_com_angstrom``, in the EvoBind adversarial check) and leaves ``binder_ca_rmsd`` NaN.
    """
    return input_path if predictor == "of3" and openfold_mode == "refold" else None


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

    block["cache"] = _cache_block(session, request, adopted=adopted)
    provenance: dict[str, Any] = {}
    checkpoint = record.extras.get("inference_ckpt_name")
    if record.model == "of3" and checkpoint:
        provenance["openfold3_checkpoint"] = str(checkpoint)
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
        )
    except Exception as e:  # noqa: BLE001 - per-metric isolation; recorded in results["prediction"]
        logger.warning("  [warning] Prediction failed: %s", e)
        traceback.print_exc()
        return {
            "model": request.model,
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
) -> tuple[dict, dict]:
    """The whole prediction step of ``run_pipeline``: store, session, request, consumers.

    Builds one session for the sample on the store in ``prediction_cache`` (default
    ``<output_dir>/predictions``) and calls ``run_prediction_step``. Never raises.

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
    try:
        adopt = prediction_dir is not None
        runner = None if adopt else make_runner(predictor, openfold_conda_env)
        session = make_session(
            make_store(prediction_cache or Path(output_dir) / DEFAULT_CACHE_DIRNAME),
            runner,
            rerun=rerun_predictions,
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
        )
    except Exception as e:  # noqa: BLE001 - per-metric isolation; recorded in results["prediction"]
        logger.warning("  [warning] Prediction failed: %s", e)
        traceback.print_exc()
        return {"model": predictor, "error": str(e)}, {}
    block, provenance = run_prediction_step(
        session,
        request,
        input_path=input_path,
        binder_chain=binder_chain,
        receptor_chain=receptor_chain,
        prediction_dir=prediction_dir,
        prediction_binder_chain=prediction_binder_chain,
        prediction_target_chain=prediction_target_chain,
        reference_path=reference_for(predictor, openfold_mode, input_path),
        # an adopted output is the user's own: its seeds are not the ones given here
        seed_index=1 if adopt else scored_seed_index(openfold_seeds),
    )
    if predictor == "of3" and not adopt and not block.get("error"):
        record_binder_cyclic(block, input_path, binder_chain, openfold_cyclic, openfold_conda_env)
    return block, provenance
