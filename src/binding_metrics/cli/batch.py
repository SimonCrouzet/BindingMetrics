"""Run the binding-metrics pipeline on all structures in a directory.

Scans --input-dir for .cif / .pdb / .mmcif files, runs the full pipeline on
each one (optionally in parallel), and aggregates all results into a single CSV.
Per-sample JSON reports and intermediate files are written to sub-directories
inside --output-dir.

Usage:
    binding-metrics-batch \\
        --input-dir 04_analysis_inputs/refold_cif/ \\
        --output-csv custom_metrics.csv \\
        --workers 4 \\
        [all the same options as binding-metrics-run]

Every CSV row ends with ``provenance_*`` columns (package version, git sha,
seed, platform) that tie the row to the code and settings that produced it.

Configuration file:
    --config batch.toml supplies option defaults; flags on the command line
    override the file. Keys are the long option names (workers, md-duration-ps or
    md_duration_ps):

        # batch.toml
        input-dir = "designs/"
        output-csv = "metrics.csv"
        workers = 4
        skip-relax = true
        metrics = "interface,geometry"

    A flag takes true or false, an option with several values takes a list, and
    an unknown key is an error. An option the file sets need not be repeated on
    the command line, required ones included.

Sample status (CSV column ``batch_status``)
-------------------------------------------
``ok`` (all steps completed), ``partial`` (the pipeline finished but a step
failed; see ``batch_failed_steps`` and ``batch_failed_reasons``) or ``error``
(the worker raised; see ``batch_error``). The exit code is non-zero only when
no sample has status ``ok``.

Structure prediction (--predictor MODEL)
----------------------------------------
Like the batched OpenFold3 call, the prediction step runs once for all samples after the
workers, in the main process. The samples share one prediction store (``--prediction-cache``,
default ``<output-dir>/_predictions``): the model starts once for the predictions the store
lacks, each sample then reads its prediction from the store, and a second run over the same
samples starts no model. ``--prediction-dir ROOT`` reads existing outputs instead, one per
sample ID, and never runs the model. The columns are ``prediction_*`` (the only ``openfold_*``
column is ``openfold_skipped``). A sample whose prediction failed is ``partial`` with
``prediction`` in ``batch_failed_steps``; the others are not affected.

The legacy batched OpenFold3 call (``--openfold-*`` without ``--predictor``) marks a sample it
failed for ``partial`` with ``openfold`` in ``batch_failed_steps`` and the reason in
``batch_failed_reasons``; a failure of the whole call marks every sample it covered.

Concurrency model (--workers > 1)
----------------------------------
Workers are OS processes (ProcessPoolExecutor), not threads. This means:

  * No shared memory — each worker has its own address space. There are no
    shared variables and no risk of race conditions between workers.

  * No concurrent file writes — each worker is given a unique output directory
    ({output-dir}/{sample_id}/). Workers never write to the same path.

  * The aggregated CSV is written by the main process only after all workers
    have finished, so it is always written by a single writer.

  * Per-sample logs are written inside each sample's own output directory
    ({output-dir}/{sample_id}/{sample_id}.log), again unique per worker.

Note on GPU parallelism:
    Each worker process initialises its own CUDA context. Running N workers
    on a single GPU will divide VRAM by N and is likely to cause
    out-of-memory errors. Use --device cpu when --workers > 1 unless you
    have dedicated GPUs (one per worker).
"""

import argparse
import logging
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from binding_metrics._constants import (
    DEFAULT_DEVICE,
    DEFAULT_MD_DURATION_PS,
    DEFAULT_PH,
    DEFAULT_RANDOM_SEED,
)
from binding_metrics.capabilities import POLICIES, IncompatibleInputError
from binding_metrics.cli import (
    OPENFOLD_MODE_HELP,
    add_config_arg,
    add_on_unmappable_residue_arg,
    add_openfold_cyclic_arg,
    add_openfold_no_msa_server_arg,
    add_openfold_seeds_arg,
    add_random_seed_arg,
    check_on_unmappable_residue,
    check_openfold_cyclic,
    on_unmappable_residue_kwargs,
    openfold_cyclic_kwargs,
    openfold_msa_server_kwargs,
    parse_args_with_config,
)
from binding_metrics.cli.prediction import (
    BATCH_CACHE_DIRNAME,
    add_prediction_args,
    check_prediction_args,
    check_predictor,
    check_weights_arg,
    display_name,
    effective_prediction_mode,
    error_text,
    make_request,
    make_runner,
    make_session,
    make_store,
    output_weights,
    record_binder_cyclic,
    record_templates,
    reference_for,
    run_prediction_step,
    runner_chain_map,
    runner_weights_kinds,
    scored_seed_index,
    scored_seed_kwargs,
    weights_description,
)
from binding_metrics.cli.run import (
    ALL_METRICS,
    KNOWN_METRICS,
    _collect_failures,
    _merge_reason,
    _parse_metrics,
    run_pipeline,
)
from binding_metrics.metrics._common import ChainAliasAction, resolve_chain_role
from binding_metrics.preflight_cli import (
    BINDER_TYPE_CHOICES,
    add_preflight_args,
    check_input,
    model_step_of,
    refusal_block,
)
from binding_metrics.provenance import collect_provenance, conda_python_command, openfold3_version
from binding_metrics.utils import configure_logging

# Named explicitly: ``python -m binding_metrics.cli.batch`` executes this file as
# ``__main__``, and a logger called ``__main__`` would sit outside the package
# logger that ``configure_logging`` sets up, so its INFO lines would be lost.
logger = logging.getLogger("binding_metrics.cli.batch")

_STRUCTURE_SUFFIXES = {".cif", ".pdb", ".mmcif"}


def _build_reference_map(reference_dir: Path) -> dict[str, Path]:
    """Map native structures in *reference_dir* by filename stem, for DockQ.

    Each sample input is later matched to its reference by stem (e.g. an input
    ``target1.cif`` pairs with a reference ``target1.pdb``). Only files with a
    recognised structure suffix are included. If two references share a stem
    (e.g. ``target1.pdb`` and ``target1.cif``), the first in sorted order wins,
    so the mapping is deterministic.

    Args:
        reference_dir: Directory containing native/reference structures.

    Returns:
        Dict mapping filename stem → reference path.
    """
    refs: dict[str, Path] = {}
    for p in sorted(reference_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _STRUCTURE_SUFFIXES:
            refs.setdefault(p.stem, p)
    return refs


def _provenance_columns(provenance: dict) -> dict:
    """Flatten a provenance block into ``provenance_<key>`` CSV columns.

    ``report._flatten`` only knows the metric sections, so the block is
    flattened here. The columns come after every existing one.
    """
    return {f"provenance_{key}": value for key, value in provenance.items()}


def _resolve_log_path(sample_output_dir: Path, sid: str, log_file: Optional[Path]) -> Path:
    """Return the file a worker logs to.

    Without ``--log-file`` every sample gets ``{sample_dir}/{sid}.log``; this is
    what ``--per-sample-log`` asks for, so the flag needs no separate branch.
    An explicit ``--log-file`` wins and all samples share it.
    """
    return Path(log_file) if log_file else sample_output_dir / f"{sid}.log"


# ---------------------------------------------------------------------------
# Worker (must be module-level so it is picklable by multiprocessing)
# ---------------------------------------------------------------------------


def _run_one(
    input_path: Path,
    output_dir: Path,
    sample_id: Optional[str],
    skip_prep: bool,
    ph: float,
    keep_water: bool,
    canonicalize: bool,
    skip_relax: bool,
    md_duration_ps: float,
    device: str,
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    metrics: frozenset,
    energy_modes: tuple,
    openfold_mode: str,
    openfold_conda_env: Optional[str],
    log_file: Optional[Path],
    reference_path: Optional[Path] = None,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    raise_errors: bool = False,
    binder_type: str = "auto",
    on_incompatible: str = "error",
    preflight_model: Optional[tuple] = None,
    on_unmappable_residue: str = "error",
    prediction_weights: Optional[Path] = None,
) -> dict:
    """Run the pipeline for a single structure and return a flat results dict.

    ``binder_chain`` and ``target_chain`` are keyword-only aliases of
    ``peptide_chain`` and ``receptor_chain``; both spellings with different IDs
    make the sample an ``"error"`` row, like any other pipeline failure. With
    ``raise_errors`` an exception from the pipeline is re-raised instead of
    becoming an ``"error"`` row. An input that the pre-flight check refuses
    (``on_incompatible="error"``) is such an error row: ``batch_error`` and
    ``preflight_reason`` carry the reason and nothing was prepared or run.
    ``preflight_model`` is the whole-batch model step of this sample, as ``(model, metric,
    adopted)``, so that its limits are checked here, before this worker does anything.

    The row carries ``batch_status``:

    * ``"ok"``: every requested step completed.
    * ``"partial"``: the pipeline finished but at least one step reported an
      error or ``success=False`` (same rule as ``binding-metrics-run``). The
      step names go to ``batch_failed_steps`` (``;``-joined) and
      ``step: reason`` pairs to ``batch_failed_reasons`` (``|``-joined).
    * ``"error"``: the worker itself raised; the cause is in ``batch_error``.
    """
    from binding_metrics.cli import log_to_file
    from binding_metrics.protocols.report import _flatten, write_report

    sid = sample_id or input_path.stem
    sample_output_dir = output_dir / sid
    log_path = _resolve_log_path(sample_output_dir, sid, log_file)

    t0 = time.time()
    error_msg: Optional[str] = None
    results: dict = {"sample_id": sid, "input": str(input_path)}

    sample_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # A shared --log-file is truncated once by main(); each worker appends,
        # otherwise every sample would erase the previous samples' logs.
        with log_to_file(log_path, mode="a" if log_file else "w"):
            logger.info("\n%s", "#" * 60)
            logger.info("  binding-metrics-batch worker: %s", sid)
            logger.info("  Input:  %s", input_path)
            logger.info("  Output: %s", sample_output_dir)
            logger.info("  Log:    %s", log_path)
            logger.info("%s", "#" * 60)

            results = run_pipeline(
                input_path=input_path,
                output_dir=sample_output_dir,
                sample_id=sid,
                skip_prep=skip_prep,
                ph=ph,
                keep_water=keep_water,
                canonicalize=canonicalize,
                skip_relax=skip_relax,
                md_duration_ps=md_duration_ps,
                device=device,
                peptide_chain=peptide_chain,
                receptor_chain=receptor_chain,
                metrics=metrics,
                energy_modes=energy_modes,
                reference_path=reference_path,
                openfold_mode=openfold_mode,
                openfold_conda_env=openfold_conda_env,
                random_seed=random_seed,
                binder_chain=binder_chain,
                target_chain=target_chain,
                binder_type=binder_type,
                on_incompatible=on_incompatible,
                preflight_model=preflight_model,
                on_unmappable_residue=on_unmappable_residue,
                prediction_weights=prediction_weights,
            )
            results["total_elapsed_s"] = round(time.time() - t0, 1)

            write_report(results, sample_output_dir, sid, fmt="json")

    except Exception as e:
        if raise_errors:
            raise
        error_msg = f"{type(e).__name__}: {e}"
        if isinstance(e, IncompatibleInputError):
            # a decision, not a crash: the message says what to change
            logger.warning("  %s: refused by the pre-flight check\n%s", sid, e)
            results["preflight"] = refusal_block(e)
        else:
            traceback.print_exc()
        results["total_elapsed_s"] = round(time.time() - t0, 1)
        results["batch_error"] = error_msg

    flat = _flatten(results)
    failures = [] if error_msg else _collect_failures(results)
    if error_msg:
        flat["batch_status"] = "error"
        flat["batch_error"] = error_msg
    elif failures:
        flat["batch_status"] = "partial"
        flat["batch_failed_steps"] = ";".join(step for step, _ in failures)
        flat["batch_failed_reasons"] = " | ".join(f"{step}: {why}" for step, why in failures)
    else:
        flat["batch_status"] = "ok"
    # A worker that raised has no pipeline results, so fall back to a fresh block.
    flat.update(
        _provenance_columns(results.get("provenance") or collect_provenance(seed=random_seed))
    )
    return flat


# ---------------------------------------------------------------------------
# Steps that run once for all samples, after the workers
# ---------------------------------------------------------------------------


def _detect_sample_chains(
    rows: list[dict],
    sid_to_input: dict[str, Path],
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    step: str,
) -> list[tuple[int, str, Path, str, str]]:
    """The samples a whole-batch step can process, with their chains.

    A sample qualifies when its worker did not fail, its input file is known and both chains
    are given or detected (the logic of ``run_pipeline``). ``step`` names the step in the
    warning for an input whose chains cannot be detected.

    Returns:
        ``(row index, sample ID, input path, binder chain, receptor chain)`` per sample.
    """
    from binding_metrics.io.structures import detect_chains_from_file

    eligible = []
    for i, row in enumerate(rows):
        sid = row.get("sample_id")
        if not sid or row.get("batch_status") == "error":
            continue
        input_path = sid_to_input.get(sid)
        if input_path is None:
            continue

        # Detect chains from the input file (same logic as run_pipeline)
        try:
            chain_info = detect_chains_from_file(
                input_path,
                peptide_chain=peptide_chain,
                receptor_chain=receptor_chain,
            )
        except Exception as e:  # noqa: BLE001 - one unreadable input must not stop the batch
            logger.warning("  %s: skipped for %s, chain detection failed: %s", sid, step, e)
            continue

        pchain = chain_info.get("peptide_chain")
        rchain = chain_info.get("receptor_chain")
        if not pchain or not rchain:
            continue
        eligible.append((i, sid, input_path, pchain, rchain))
    return eligible


def _model_step_allowed(
    sid: str,
    input_path: Path,
    pchain: str,
    rchain: str,
    model: tuple,
    binder_type: str,
    on_incompatible: str,
    on_unmappable_residue: str = "error",
    prediction_weights: Optional[Path] = None,
) -> tuple[bool, str]:
    """Whether the whole-batch model step may run for one sample, and why not.

    The worker of the sample has already checked the model (``preflight_model``): under ``error``
    it refused the sample, so it is not here, and under ``warn`` it logged the problems. What
    is left is ``skip``, where the sample's model step is dropped and the reason recorded. The
    check runs again here, before any request is built or the store touched, so that the
    whole-batch step never depends on what the worker did.
    """
    if on_incompatible == "warn":
        return True, ""
    outcome = check_input(
        input_path,
        pchain,
        rchain,
        binder_type=binder_type,
        on_incompatible=on_incompatible,
        steps=frozenset(),
        model=model,
        on_unmappable_residue=on_unmappable_residue,
        prediction_weights=prediction_weights,
    )
    if outcome.refused or not outcome.model_usable:
        reason = outcome.block["reason"] or outcome.block["status"]
        logger.warning("  %s: left out of the model step by the pre-flight check: %s", sid, reason)
        return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Batched OpenFold (single subprocess for all samples)
# ---------------------------------------------------------------------------


def _add_provenance(row: dict, key: str, value) -> None:
    """Write a provenance value into a row as ``provenance_<key>``.

    A dict (the weights of a prediction) becomes one column per entry, ``provenance_<key>_<name>``,
    so that the CSV has ``provenance_prediction_weights_sha256`` and ``..._path`` and no cell holds
    a dictionary.
    """
    if isinstance(value, dict):
        for name, item in value.items():
            row[f"provenance_{key}_{name}"] = item
    else:
        row[f"provenance_{key}"] = value


def _run_batched_openfold(
    rows: list[dict],
    sid_to_input: dict[str, Path],
    output_dir: Path,
    openfold_mode: str,
    openfold_conda_env: Optional[str],
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    openfold_seeds: Optional[Sequence[int]] = None,
    on_unmappable_residue: str = "error",
    binder_type: str = "auto",
    on_incompatible: str = "error",
    openfold_cyclic: bool | str = "auto",
    openfold_use_msa_server: bool = True,
    prediction_weights: Optional[Path] = None,
    prediction_cache: Optional[Path] = None,
) -> None:
    """Run OpenFold3 on all successful samples in a single subprocess.

    Modifies *rows* in-place, merging OF3 and EvoBind metrics into each
    sample's flat dict.  Also updates each sample's JSON report on disk.
    ``openfold_seeds`` are the seeds OpenFold3 samples with (written to its runner YAML); ``None``
    keeps the default, 42. ``openfold_cyclic`` decides, sample by sample, whether the binder gets
    ``"cyclic": true``; each sample's block records ``binder_cyclic``. ``openfold_use_msa_server``
    False runs OpenFold3 without the ColabFold MSA server; each row records it as
    ``provenance_openfold3_use_msa_server``. ``prediction_weights`` (a checkpoint file) is passed
    to OpenFold3 as ``inference_ckpt_path`` for the whole batch, hashed once with the cache in
    ``prediction_cache`` (default ``<output_dir>/_predictions``) and recorded in each block as
    ``weights`` and in the row as ``provenance_prediction_weights_*``. ``on_unmappable_residue``
    is passed to the query preparation of every sample; a residue OpenFold3 cannot take then
    stops the whole
    batch call (the error names each such residue) before the model starts. A sample that the
    pre-flight check leaves out (``on_incompatible="skip"``) gets ``openfold_skipped`` and the
    reason in ``openfold_reason`` and is not part of the call.
    """
    from binding_metrics.metrics.openfold import (
        _BatchSample,
        compute_openfold_metrics,
        run_openfold_batched,
    )
    from binding_metrics.protocols.report import _flatten

    # Build list of eligible samples (succeeded, have chain info)
    samples: list[_BatchSample] = []
    # Map query_name → row index for merging results back
    sid_to_row_idx: dict[str, int] = {}
    # Map query_name → chain info for metrics extraction
    sid_to_chains: dict[str, dict] = {}

    for i, sid, input_path, pchain, rchain in _detect_sample_chains(
        rows, sid_to_input, peptide_chain, receptor_chain, "OpenFold"
    ):
        allowed, why_not = _model_step_allowed(
            sid,
            input_path,
            pchain,
            rchain,
            ("of3", "openfold", False, openfold_mode),
            binder_type,
            on_incompatible,
            on_unmappable_residue,
            prediction_weights,
        )
        if not allowed:
            rows[i]["openfold_skipped"] = True
            rows[i]["openfold_reason"] = why_not
            continue
        samples.append(
            _BatchSample(
                query_name=sid,
                complex_structure_path=input_path,
                receptor_chain=rchain,
                binder_chain=pchain,
            )
        )
        sid_to_row_idx[sid] = i
        sid_to_chains[sid] = {"peptide": pchain, "receptor": rchain}

    if not samples:
        logger.info("  [skip] No eligible samples for batched OpenFold.")
        return

    bar = "=" * 60
    logger.info(
        "\n%s\n  Step: Batched OpenFold3 (%d samples in one call)\n%s", bar, len(samples), bar
    )

    installed = openfold3_version(conda_python_command(openfold_conda_env))
    for idx in sid_to_row_idx.values():
        rows[idx]["provenance_openfold3_version"] = installed
        rows[idx]["provenance_openfold3_use_msa_server"] = bool(openfold_use_msa_server)

    of_dir = output_dir / "_openfold_batch"
    weights_ref = None
    try:
        extra_run_arguments = {}
        if prediction_weights is not None:
            weights_ref = make_store(
                prediction_cache or Path(output_dir) / BATCH_CACHE_DIRNAME
            ).weights_reference(prediction_weights, expect="file")
            extra_run_arguments["inference_ckpt_path"] = str(weights_ref.path)
        predictions_dir = run_openfold_batched(
            samples=samples,
            output_dir=of_dir,
            mode=openfold_mode,
            conda_env=openfold_conda_env,
            **extra_run_arguments,
            **({"seeds": tuple(openfold_seeds)} if openfold_seeds else {}),
            **on_unmappable_residue_kwargs(on_unmappable_residue),
            **openfold_cyclic_kwargs(openfold_cyclic),
            **openfold_msa_server_kwargs(openfold_use_msa_server),
        )
    except Exception as e:  # noqa: BLE001 - the batch call spawns a subprocess; see openfold_error
        # Warning level keeps the line on stdout, where it was printed before.
        logger.warning("  [ERROR] Batched OpenFold failed: %s", e)
        import traceback

        traceback.print_exc()
        for idx in sid_to_row_idx.values():
            rows[idx]["openfold_error"] = str(e)
            _mark_step_failed(rows[idx], "openfold", str(e))
        return

    # Extract per-sample metrics and merge into rows
    for s in samples:
        sid = s.query_name
        idx = sid_to_row_idx[sid]
        chains = sid_to_chains[sid]
        pchain = chains["peptide"]
        rchain = chains["receptor"]

        try:
            # the input pose is the reference in both modes: OpenFold3 places the binder itself
            of_metrics = compute_openfold_metrics(
                output_dir=predictions_dir,
                query_name=sid,
                binder_chain=pchain,
                receptor_chain=rchain,
                reference_structure_path=sid_to_input[sid],
                **scored_seed_kwargs(openfold_seeds, predictions_dir, sid),
            )

            record_binder_cyclic(
                of_metrics, sid_to_input[sid], pchain, openfold_cyclic, openfold_conda_env
            )
            record_templates(of_metrics, predictions_dir, sid)
            recorded = output_weights(predictions_dir)
            if weights_ref is not None:
                recorded["weights"] = weights_ref.to_dict()
            weights = weights_description(recorded)
            if weights is not None:
                of_metrics["weights"] = weights
                _add_provenance(rows[idx], "prediction_weights", weights)

            # EvoBind metrics — reuse OF3 outputs
            of_structure = of_metrics.get("structure_path")
            plddt = of_metrics.get("plddt_per_atom")
            if of_structure:
                try:
                    from binding_metrics.metrics.evobind import (
                        compute_evobind_adversarial_check,
                        compute_evobind_score,
                    )

                    evobind = compute_evobind_score(
                        of_structure,
                        plddt_per_atom=plddt,
                        binder_chain=pchain,
                        receptor_chain=rchain,
                    )
                    _merge_reason(of_metrics, evobind, "evobind")
                    of_metrics.update(evobind)
                    adversarial = compute_evobind_adversarial_check(
                        design_structure_path=sid_to_input[sid],
                        afm_structure_path=of_structure,
                        binder_chain=pchain,
                        receptor_chain=rchain,
                        afm_plddt_per_atom=plddt,
                    )
                    _merge_reason(of_metrics, adversarial, "evobind adversarial")
                    of_metrics.update(adversarial)
                except Exception as e:  # noqa: BLE001 - per-sample isolation; see evobind_error
                    of_metrics["evobind_error"] = str(e)

            # Flatten OF3 metrics into the row
            for k, v in _flatten({"openfold": of_metrics}).items():
                if k.startswith("openfold_"):
                    rows[idx][k] = v

            # Update per-sample JSON report on disk
            _update_sample_json(output_dir / sid, sid, of_metrics)

            logger.info(
                "  %s: ipTM=%s, pLDDT=%s",
                sid,
                of_metrics.get("iptm", "?"),
                of_metrics.get("avg_plddt", "?"),
            )

        except Exception as e:  # noqa: BLE001 - per-sample isolation; see openfold_error
            logger.warning("  %s: OpenFold metrics failed: %s", sid, e)
            rows[idx]["openfold_error"] = str(e)
            _mark_step_failed(rows[idx], "openfold", str(e))


def _update_sample_json(
    sample_dir: Path,
    sid: str,
    of_metrics: dict,
    section: str = "openfold",
    provenance: Optional[dict] = None,
) -> None:
    """Merge one step's results into an existing per-sample JSON report.

    ``section`` is the key the block is written to (``"openfold"`` or ``"prediction"``);
    ``provenance`` adds keys to the report's provenance block.
    """
    import json

    from binding_metrics.protocols.report import _json_default

    json_path = sample_dir / f"{sid}_results.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data[section] = of_metrics
        if provenance:
            data.setdefault("provenance", {}).update(provenance)
        json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - non-critical, the CSV row has the data anyway
        logger.warning("  %s: could not update %s: %s", sid, json_path.name, e)


def _mark_step_failed(row: dict, step: str, reason: str) -> None:
    """Record a failed step on a finished row, as ``_run_one`` records the steps it ran.

    An ``ok`` row becomes ``partial``; the step joins ``batch_failed_steps`` and
    ``step: reason`` joins ``batch_failed_reasons``.
    """
    if row.get("batch_status") == "ok":
        row["batch_status"] = "partial"
    steps = [name for name in str(row.get("batch_failed_steps") or "").split(";") if name]
    row["batch_failed_steps"] = ";".join([*steps, step])
    entry = f"{step}: {str(reason)[:200]}"
    previous = row.get("batch_failed_reasons")
    row["batch_failed_reasons"] = f"{previous} | {entry}" if previous else entry


# ---------------------------------------------------------------------------
# Batched prediction (one shared store, one model start for the missing samples)
# ---------------------------------------------------------------------------


def _run_batched_prediction(
    rows: list[dict],
    sid_to_input: dict[str, Path],
    output_dir: Path,
    *,
    predictor: str,
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    prediction_dir: Optional[Path] = None,
    prediction_binder_chain: Optional[str] = None,
    prediction_target_chain: Optional[str] = None,
    prediction_cache: Optional[Path] = None,
    rerun_predictions: bool = False,
    openfold_mode: str = "score",
    openfold_conda_env: Optional[str] = None,
    openfold_seeds: Optional[Sequence[int]] = None,
    on_unmappable_residue: str = "error",
    binder_type: str = "auto",
    on_incompatible: str = "error",
    openfold_cyclic: bool | str = "auto",
    openfold_use_msa_server: bool = True,
    prediction_mode: Optional[str] = None,
    prediction_weights: Optional[Path] = None,
    prediction_lock_threshold: Optional[float] = None,
) -> None:
    """The ``--predictor`` step of a batch: every sample through one shared prediction store.

    Runs after the workers, in this process. All samples share the store in
    ``prediction_cache`` (default ``<output_dir>/_predictions``). Without ``prediction_dir``
    the requests of all samples go to ``PredictionSession.prefetch``, which starts the model
    once for those the store lacks (OpenFold3 predicts them in one call); with it each
    sample's output is adopted from ``<prediction_dir>`` and the model never runs. Every
    sample then gets its own session, reads its record from the store, and computes the
    metrics of ``run_prediction_step``.

    Modifies *rows* in place: the ``prediction_*`` columns, ``provenance_openfold3_*`` when
    OpenFold3 was run, and, for a sample whose prediction failed, ``batch_status`` "partial"
    with the step named in ``batch_failed_steps``. Also writes ``prediction`` into each
    sample's JSON report. A failed prediction of one sample, or of the whole batch call,
    never stops the others. A sample that the pre-flight check leaves out
    (``on_incompatible="skip"``, the limits of the model) is dropped before its request is built,
    with ``prediction_skipped`` and the reason in the row, and does not fail the batch.
    """
    from binding_metrics.protocols.report import _flatten

    candidates = _detect_sample_chains(
        rows, sid_to_input, peptide_chain, receptor_chain, "prediction"
    )
    adopted = prediction_dir is not None
    mode = effective_prediction_mode(predictor, prediction_dir, prediction_mode, openfold_mode)
    eligible = []
    for entry in candidates:
        idx, sid, input_path, pchain, rchain = entry
        allowed, why_not = _model_step_allowed(
            sid,
            input_path,
            pchain,
            rchain,
            (predictor, "prediction", adopted, mode),
            binder_type,
            on_incompatible,
            on_unmappable_residue,
            prediction_weights,
        )
        if allowed:
            eligible.append(entry)
            continue
        block = {"model": predictor, "mode": mode, "skipped": True, "reason": why_not}
        for key, value in _flatten({"prediction": block}).items():
            if key.startswith("prediction_"):
                rows[idx][key] = value
        _update_sample_json(output_dir / sid, sid, block, section="prediction")
    if not eligible:
        logger.info("  [skip] No eligible samples for the batched prediction.")
        return

    bar = "=" * 60
    logger.info(
        "\n%s\n  Step: Structure prediction (%s), %d samples\n%s",
        bar,
        display_name(predictor),
        len(eligible),
        bar,
    )

    adopt = prediction_dir is not None
    outcomes: dict[str, tuple[dict, dict]] = {}
    requests = {}
    runner = None
    store = make_store(prediction_cache or Path(output_dir) / BATCH_CACHE_DIRNAME)
    try:
        runner = None if adopt else make_runner(predictor, openfold_conda_env)
        weights_ref = None
        if prediction_weights is not None and not adopt:
            # one reference for the whole batch: the weights are hashed (or looked up) once
            weights_ref = store.weights_reference(
                prediction_weights, expect=runner_weights_kinds().get(predictor)
            )
        for _, sid, input_path, pchain, rchain in eligible:
            try:
                requests[sid] = make_request(
                    predictor,
                    sid,
                    input_path,
                    binder_chain=pchain,
                    receptor_chain=rchain,
                    runner=runner,
                    adopt=adopt,
                    openfold_mode=openfold_mode,
                    openfold_seeds=openfold_seeds,
                    on_unmappable_residue=on_unmappable_residue,
                    openfold_cyclic=openfold_cyclic,
                    openfold_use_msa_server=openfold_use_msa_server,
                    prediction_mode=prediction_mode,
                    prediction_weights=weights_ref,
                    prediction_lock_threshold=prediction_lock_threshold,
                )
            except Exception as e:  # noqa: BLE001 - one unreadable input must not stop the batch
                logger.warning(
                    "  %s: no %s prediction request: %s", sid, display_name(predictor), e
                )
                outcomes[sid] = ({"model": predictor, "mode": mode, "error": str(e)}, {})
        if requests and not adopt:
            make_session(store, runner, rerun=rerun_predictions).prefetch(requests.values())
    except Exception as e:  # noqa: BLE001 - the batch call starts a model; see prediction_error
        # Warning level keeps the line on stdout, where the OpenFold step printed its own.
        logger.warning("  [ERROR] Batched %s prediction failed: %s", display_name(predictor), e)
        traceback.print_exc()
        for sid in requests:
            outcomes[sid] = (
                {"model": predictor, "mode": mode, "error": error_text(e, predictor)},
                {},
            )
        requests = {}

    if predictor == "of3" and not adopt:
        installed = openfold3_version(conda_python_command(openfold_conda_env))
        for idx, *_ in eligible:
            rows[idx]["provenance_openfold3_version"] = installed
            rows[idx]["provenance_openfold3_use_msa_server"] = bool(openfold_use_msa_server)

    for idx, sid, input_path, pchain, rchain in eligible:
        if sid in requests:
            outcomes[sid] = run_prediction_step(
                # a session of its own: it finds what the prefetch or an earlier run stored,
                # and a forced rerun has already happened for the whole batch
                make_session(store, runner),
                requests[sid],
                input_path=input_path,
                binder_chain=pchain,
                receptor_chain=rchain,
                prediction_dir=prediction_dir,
                prediction_binder_chain=prediction_binder_chain,
                prediction_target_chain=prediction_target_chain,
                reference_path=reference_for(predictor, input_path),
                # an adopted output is the user's own: its seeds are not the ones given here
                seed_index=1 if adopt else scored_seed_index(openfold_seeds),
                mode=mode,
                default_chain_map=None if adopt else runner_chain_map(runner, requests[sid]),
            )
            block = outcomes[sid][0]
            if predictor == "of3" and not adopt and not block.get("error"):
                record_binder_cyclic(block, input_path, pchain, openfold_cyclic, openfold_conda_env)
        block, provenance = outcomes[sid]
        for key, value in _flatten({"prediction": block}).items():
            if key.startswith("prediction_"):
                rows[idx][key] = value
        for key, value in provenance.items():
            _add_provenance(rows[idx], key, value)
        _update_sample_json(
            output_dir / sid, sid, block, section="prediction", provenance=provenance or None
        )
        if block.get("error"):
            _mark_step_failed(rows[idx], "prediction", block["error"])
        logger.info(
            "  %s: ipTM=%s, pLDDT=%s",
            sid,
            block.get("iptm", "?"),
            block.get("avg_plddt", "?"),
        )


# ---------------------------------------------------------------------------
# In-process batch API
# ---------------------------------------------------------------------------


class _LostSample(dict):
    """Row of a sample whose worker failed outside the pipeline's own error handling.

    Raised by the worker process dying, by pickling, or by the output directory
    being unwritable, so the ordinary ``"error"`` row of ``_run_one`` never
    existed. ``main`` reports these as ``FATAL``.
    """


def _lost_sample_row(sample_id: str, error: BaseException) -> _LostSample:
    return _LostSample(
        sample_id=sample_id,
        batch_status="error",
        batch_error=f"{type(error).__name__}: {error}",
    )


def run_batch(
    paths: Iterable,
    output_dir,
    *,
    skip_prep: bool = False,
    ph: float = DEFAULT_PH,
    keep_water: bool = False,
    canonicalize: bool = False,
    skip_relax: bool = False,
    md_duration_ps: float = DEFAULT_MD_DURATION_PS,
    device: str = DEFAULT_DEVICE,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    metrics: Iterable[str] = ALL_METRICS,
    energy_modes: Sequence[str] = ("relaxed",),
    references: Optional[Mapping[str, Path]] = None,
    openfold_mode: str = "score",
    openfold_conda_env: Optional[str] = None,
    openfold_seeds: Optional[Sequence[int]] = None,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    log_file: Optional[Path] = None,
    n_workers: int = 1,
    on_result: Optional[Callable[[dict], None]] = None,
    on_error: str = "record",
    on_start: Optional[Callable[[Path], None]] = None,
    on_unmappable_residue: str = "error",
    predictor: Optional[str] = None,
    prediction_dir: Optional[Path] = None,
    prediction_binder_chain: Optional[str] = None,
    prediction_target_chain: Optional[str] = None,
    prediction_cache: Optional[Path] = None,
    rerun_predictions: bool = False,
    binder_type: str = "auto",
    on_incompatible: str = "error",
    preflight_only: bool = False,
    openfold_cyclic: bool | str = "auto",
    openfold_use_msa_server: bool = True,
    prediction_mode: Optional[str] = None,
    prediction_weights: Optional[Path] = None,
    prediction_cyclic: Optional[bool | str] = None,
    prediction_use_msa_server: bool = True,
    prediction_conda_env: Optional[str] = None,
    prediction_lock_threshold: Optional[float] = None,
) -> list[dict]:
    """Run the pipeline on every structure in ``paths``; the in-process ``binding-metrics-batch``.

    One structure is one item: it gets its own directory ``output_dir/<stem>/``
    and the same per-sample JSON report and log file as the command line
    writes. The options have the meaning and defaults of the ``binding-metrics-batch``
    flags of the same name; ``binder_chain`` and ``target_chain`` are aliases of
    ``peptide_chain`` and ``receptor_chain``.

    Args:
        paths: Structure files (CIF or PDB). The sample ID of each is its file
            stem, so two paths with the same stem would share a directory.
        output_dir: Directory for the per-sample sub-directories; created when missing.
        metrics: Names from ``KNOWN_METRICS``. ``openfold`` runs once for all
            samples after the others, as in the CLI, so the ``openfold_*`` columns
            reach the returned rows but not the copies already handed to ``on_result``
            (the dicts are the same objects, updated in place).
        references: Native structures for DockQ, by sample ID (file stem). A
            non-empty mapping enables the ``dockq`` metric, as ``--reference-dir`` does.
        log_file: One file all samples log to (truncated at the start); by
            default every sample logs to ``<stem>/<stem>.log``.
        n_workers: 1 runs the items one after the other in this process; more
            uses that many worker processes, as ``--workers`` does (mind the GPU
            memory each one takes).
        on_result: Called with each item's row as soon as that item finishes,
            in the calling process. With several workers the calls follow
            completion order, not input order.
        on_error: ``"record"`` (default) turns an exception raised while
            processing one item into an ``"error"`` row and goes on with the
            others; ``"raise"`` re-raises it (other items still waiting are
            dropped). A step that reports a failure is not an exception: it
            gives a ``"partial"`` row in both modes.
        on_start: Called with the path of each item just before it starts (in
            the sequential case), or as it is submitted (with workers).
        on_unmappable_residue: ``"error"`` (default) or ``"x"``: what the OpenFold3 call does
            with a residue it cannot take (see ``--on-unmappable-residue``).
        openfold_seeds: Seed values OpenFold3 samples with, written to its runner YAML; ``None``
            keeps the default, 42. The sample scored is the first sample of the first seed
            given (see ``--openfold-seeds``).
        openfold_use_msa_server: Whether OpenFold3 uses the ColabFold MSA server (default True;
            ``--openfold-no-msa-server`` is False). With the server off OpenFold3 runs without a
            computed MSA (single-sequence unless MSAs are supplied elsewhere), which lowers
            accuracy for a natural receptor, but the template alignments written by the toolkit
            are no longer replaced by the server (issue #68). Each row that OpenFold3 was run for
            records it as ``provenance_openfold3_use_msa_server``.
        openfold_cyclic: ``"auto"`` (default), ``True`` (``"on"``) or ``False`` (``"off"``):
            whether the binder chain of each OpenFold3 query gets ``"cyclic": true`` (see
            ``--openfold-cyclic``). The ``openfold_*`` (or ``prediction_*``) columns then
            include ``binder_cyclic`` and, when a head-to-tail binder was left linear, the
            reason.
        predictor: A key of ``binding_metrics.predictors.PARSERS`` (keyword-only). The
            ``openfold`` step then runs as the prediction step of every sample through one
            shared prediction store and fills the ``prediction_*`` columns instead of the
            ``openfold_*`` ones; ``None`` (default) keeps the batched OpenFold3 call. Like
            ``openfold`` it runs once for all samples after the others. Every registered model
            has a runner and can be run from here (``cli.prediction.RUNNERS``); ``prediction_dir``
            reads outputs you made instead.
        prediction_dir: With ``predictor``, the root that holds one output per sample ID (the
            file stem); the outputs are adopted into the store and the model never runs.
        prediction_binder_chain, prediction_target_chain: With ``predictor``, the chain IDs
            inside the predictions when they differ from the inputs'.
        prediction_cache: The prediction store shared by all samples; default
            ``<output_dir>/_predictions``. A second run over the same samples, options and
            model version finds every prediction there and starts no model.
        rerun_predictions: Run every prediction again although the store has it (once).
            Outputs given as ``prediction_dir`` are never replaced.
        binder_type, on_incompatible: The pre-flight check of every sample (see
            ``--binder-type`` and ``--on-incompatible``). It runs first in each worker, before
            preparation, relaxation and the model step, which it checks as well. A refused
            sample is an ``"error"`` row with ``preflight_status`` ``refused`` and the reason;
            it does not stop the batch. Under ``"skip"`` the incompatible steps of a sample are
            recorded as skipped with their reason and the rest runs.
        prediction_mode: How the model is used for the complex (``--prediction-mode``): ``predict``,
            ``refold``, ``score`` or ``score-lock``. It is checked against what the model supports
            in the pre-flight check of every sample, for a run from here against what the runner
            of the model runs (``ValueError`` for another mode), and recorded as
            ``prediction_mode``. None takes ``openfold_mode`` for OpenFold3 run from here, the
            ``default_mode`` of the runner for another model (boltz2 ``score``, af2 and protenix
            ``predict``) and is "not known, not checked" for outputs read from
            ``prediction_dir``. Needs ``predictor``.
        prediction_cyclic, prediction_use_msa_server, prediction_conda_env,
        prediction_lock_threshold: The settings of ``predictor`` run from here, for every model
            (``--prediction-cyclic``, ``--prediction-no-msa-server``, ``--prediction-conda-env``,
            ``--prediction-lock-threshold``), with the meaning and the checks they have in
            ``run_pipeline``: for ``predictor="of3"`` they are the settings of ``openfold_cyclic``,
            ``openfold_use_msa_server`` and ``openfold_conda_env``, and a setting that the model's
            runner has no keyword for raises ``ValueError``.
        prediction_weights: Custom weights for the model, such as a fine-tuned checkpoint
            (``--prediction-weights``): a file or a directory, as the model's runner takes it. They
            apply to ``predictor`` run from here and to the batched OpenFold3 step
            (``--inference-ckpt-path``), are hashed once for the whole batch, and are part of the
            key of the prediction store. Each row gets ``prediction_weights_*`` (or
            ``openfold_weights_*``) and ``provenance_prediction_weights_sha256`` and
            ``..._path``. The path is checked before any sample runs; with ``prediction_dir`` it
            raises ``ValueError``.
        preflight_only: Check every sample and return one row each (``preflight_status``,
            ``preflight_reason``, ``preflight_plan``) without preparing, relaxing or predicting
            anything and without creating ``output_dir``.

    Returns:
        One flat row per path, in the order of ``paths`` whatever the number of
        workers. It is the row the CLI writes to its CSV: ``sample_id``,
        ``input``, the flattened results, ``batch_status`` (``ok``, ``partial``
        or ``error``; see ``_run_one``) and the ``provenance_*`` columns.

    Example:
        >>> rows = run_batch(
        ...     sorted(Path("designs").glob("*.cif")),
        ...     "results/",
        ...     metrics={"interface", "geometry"},
        ...     skip_relax=True,
        ...     n_workers=4,
        ...     on_result=lambda row: print(row["sample_id"], row["batch_status"]),
        ... )

    Raises:
        ValueError: ``n_workers`` below 1, an unknown ``on_error``, an unknown
            metric name, an ``on_unmappable_residue`` other than ``"error"`` or ``"x"``, a
            ``predictor`` that is not registered or has no runner and no ``prediction_dir``, or
            a chain given through both spellings with different IDs, or an ``openfold_cyclic``
            that is not ``"auto"``, ``"on"``, ``"off"``, ``True`` or ``False``.
        Exception: whatever an item raised, when ``on_error="raise"``.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}")
    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    selected = frozenset(metrics)
    unknown = selected - KNOWN_METRICS
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Valid choices: {', '.join(sorted(KNOWN_METRICS))}"
        )
    peptide_chain = resolve_chain_role("peptide_chain", peptide_chain, "binder_chain", binder_chain)
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    check_on_unmappable_residue(on_unmappable_residue)
    check_openfold_cyclic(openfold_cyclic)
    route = check_predictor(
        predictor,
        prediction_dir,
        prediction_mode,
        prediction_weights,
        openfold_mode=openfold_mode,
        openfold_cyclic=openfold_cyclic,
        prediction_cyclic=prediction_cyclic,
        openfold_use_msa_server=openfold_use_msa_server,
        prediction_use_msa_server=prediction_use_msa_server,
        openfold_conda_env=openfold_conda_env,
        prediction_conda_env=prediction_conda_env,
        prediction_lock_threshold=prediction_lock_threshold,
    )
    if predictor is not None and prediction_dir is None:
        # the --prediction-* spellings are merged into the settings of the run
        openfold_cyclic, openfold_use_msa_server, openfold_conda_env = (
            route.cyclic,
            route.use_msa_server,
            route.conda_env,
        )
    prediction_weights = check_weights_arg(prediction_weights, predictor)
    if binder_type not in BINDER_TYPE_CHOICES:
        raise ValueError(f"binder_type must be one of {BINDER_TYPE_CHOICES}, got {binder_type!r}")
    if on_incompatible not in POLICIES:
        raise ValueError(f"on_incompatible must be one of {POLICIES}, got {on_incompatible!r}")

    input_paths = [Path(p) for p in paths]
    output_dir = Path(output_dir)
    references = references or {}
    if references:
        selected = selected | {"dockq"}
    if preflight_only:
        return _preflight_only_rows(
            input_paths,
            output_dir,
            metrics=selected,
            skip_relax=skip_relax,
            peptide_chain=peptide_chain,
            receptor_chain=receptor_chain,
            references=references,
            predictor=predictor,
            prediction_dir=prediction_dir,
            prediction_mode=prediction_mode,
            openfold_mode=openfold_mode,
            on_unmappable_residue=on_unmappable_residue,
            prediction_weights=prediction_weights,
            binder_type=binder_type,
            on_incompatible=on_incompatible,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Strip openfold from per-worker metrics: it runs as a single batched
    # subprocess after all other metrics finish.
    want_openfold = "openfold" in selected
    # The model step runs after the workers, but each worker checks its limits first.
    preflight_model = (
        model_step_of(
            predictor,
            prediction_dir,
            selected,
            prediction_mode=prediction_mode,
            openfold_mode=openfold_mode,
        )
        if want_openfold
        else None
    )

    common_kwargs = dict(
        output_dir=output_dir,
        sample_id=None,  # derived from file stem per sample
        skip_prep=skip_prep,
        ph=ph,
        keep_water=keep_water,
        canonicalize=canonicalize,
        skip_relax=skip_relax,
        md_duration_ps=md_duration_ps,
        device=device,
        peptide_chain=peptide_chain,
        receptor_chain=receptor_chain,
        metrics=selected - {"openfold"},
        energy_modes=tuple(energy_modes),
        openfold_mode=openfold_mode,
        openfold_conda_env=openfold_conda_env,
        log_file=log_file,  # None: per-sample log inside the sample dir
        random_seed=random_seed,
        binder_type=binder_type,
        on_incompatible=on_incompatible,
        preflight_model=preflight_model,
        on_unmappable_residue=on_unmappable_residue,
        prediction_weights=prediction_weights,
    )
    if on_error == "raise":
        common_kwargs["raise_errors"] = True

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).write_text("", encoding="utf-8")  # workers append to it

    rows: list[Optional[dict]] = [None] * len(input_paths)
    sid_to_input: dict[str, Path] = {}

    def finish(index: int, row: dict) -> None:
        row.setdefault("sample_id", input_paths[index].stem)
        rows[index] = row
        if on_result is not None:
            on_result(row)

    if n_workers == 1:
        # Sequential path: simpler stack traces, easier debugging.
        for index, input_path in enumerate(input_paths):
            sid = input_path.stem
            sid_to_input[sid] = input_path
            if on_start is not None:
                on_start(input_path)
            try:
                row = _run_one(
                    input_path=input_path, reference_path=references.get(sid), **common_kwargs
                )
            except Exception as error:
                if on_error == "raise":
                    raise
                row = _lost_sample_row(sid, error)
            finish(index, row)
    else:
        # Parallel path: each worker is an independent OS process, so there are no
        # shared variables and no shared file paths (each writes to its own sample
        # directory). ``rows`` and the callbacks are only touched here, in the
        # calling process, as ``as_completed`` delivers one result at a time.
        futures = {}
        # Workers started with "spawn" or "forkserver" do not inherit the handlers
        # installed by the caller, so each configures logging itself. Without it
        # their pipeline messages would never reach the per-sample log files.
        with ProcessPoolExecutor(max_workers=n_workers, initializer=configure_logging) as pool:
            for index, input_path in enumerate(input_paths):
                sid = input_path.stem
                sid_to_input[sid] = input_path
                if on_start is not None:
                    on_start(input_path)
                future = pool.submit(
                    _run_one,
                    input_path=input_path,
                    reference_path=references.get(sid),
                    **common_kwargs,
                )
                futures[future] = index
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    if on_error == "raise":
                        for pending in futures:
                            pending.cancel()
                        raise
                    row = _lost_sample_row(input_paths[index].stem, error)
                finish(index, row)

    finished = [row for row in rows if row is not None]
    if want_openfold and predictor is not None:
        _run_batched_prediction(
            rows=finished,
            sid_to_input=sid_to_input,
            output_dir=output_dir,
            predictor=predictor,
            peptide_chain=peptide_chain,
            receptor_chain=receptor_chain,
            prediction_dir=prediction_dir,
            prediction_binder_chain=prediction_binder_chain,
            prediction_target_chain=prediction_target_chain,
            prediction_cache=prediction_cache,
            rerun_predictions=rerun_predictions,
            openfold_mode=openfold_mode,
            openfold_conda_env=openfold_conda_env,
            openfold_seeds=openfold_seeds,
            on_unmappable_residue=on_unmappable_residue,
            binder_type=binder_type,
            on_incompatible=on_incompatible,
            openfold_cyclic=openfold_cyclic,
            openfold_use_msa_server=openfold_use_msa_server,
            prediction_mode=prediction_mode,
            prediction_weights=prediction_weights,
            prediction_lock_threshold=route.lock_threshold,
        )
    elif want_openfold:
        _run_batched_openfold(
            rows=finished,
            sid_to_input=sid_to_input,
            output_dir=output_dir,
            openfold_mode=openfold_mode,
            openfold_conda_env=openfold_conda_env,
            peptide_chain=peptide_chain,
            receptor_chain=receptor_chain,
            openfold_seeds=openfold_seeds,
            on_unmappable_residue=on_unmappable_residue,
            binder_type=binder_type,
            on_incompatible=on_incompatible,
            openfold_cyclic=openfold_cyclic,
            openfold_use_msa_server=openfold_use_msa_server,
            prediction_weights=prediction_weights,
            prediction_cache=prediction_cache,
        )
    return finished


def _preflight_only_rows(
    input_paths: list[Path],
    output_dir: Path,
    *,
    metrics: frozenset,
    skip_relax: bool,
    peptide_chain: Optional[str],
    receptor_chain: Optional[str],
    references: Mapping[str, Path],
    predictor: Optional[str],
    prediction_dir: Optional[Path],
    prediction_mode: Optional[str],
    openfold_mode: str,
    on_unmappable_residue: str,
    binder_type: str,
    on_incompatible: str,
    prediction_weights: Optional[Path] = None,
) -> list[dict]:
    """One row per sample with the pre-flight decision and nothing else (``--preflight-only``).

    Nothing is prepared, relaxed or predicted and no directory is created. A refused sample
    (policy ``error``) is an ``"error"`` row; a sample that only warns or leaves steps out is
    ``"ok"``.
    """
    rows = []
    for input_path in input_paths:
        sid = input_path.stem
        row: dict = {"sample_id": sid, "input": str(input_path)}
        try:
            results = run_pipeline(
                input_path=input_path,
                output_dir=output_dir / sid,
                sample_id=sid,
                skip_relax=skip_relax,
                peptide_chain=peptide_chain,
                receptor_chain=receptor_chain,
                metrics=metrics,
                reference_path=references.get(sid),
                predictor=predictor,
                prediction_dir=prediction_dir,
                prediction_mode=prediction_mode,
                openfold_mode=openfold_mode,
                on_unmappable_residue=on_unmappable_residue,
                prediction_weights=prediction_weights,
                binder_type=binder_type,
                on_incompatible=on_incompatible,
                preflight_only=True,
            )
            block = results["preflight"]
            row["preflight_status"] = block["status"]
            row["preflight_reason"] = block["reason"]
            row["preflight_plan"] = block.get("plan", "")
            row["batch_status"] = "error" if block["status"] == "refused" else "ok"
            if block["status"] == "refused":
                row["batch_error"] = block["reason"]
        except Exception as error:  # noqa: BLE001 - one unreadable input must not stop the plan
            row["batch_status"] = "error"
            row["batch_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Run the binding-metrics pipeline on all structures in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Batch I/O
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing .cif / .pdb / .mmcif files",
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True, help="Path for the aggregated CSV results file"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for per-sample outputs (default: same directory as --output-csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). "
        "Use --device cpu when workers > 1 on a single GPU.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Optional glob pattern to filter files within --input-dir "
        "(e.g. '*.cif'). Default: all .cif/.pdb/.mmcif files.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory of native/reference structures for DockQ. "
        "Each sample is matched to a reference by filename stem "
        "(e.g. input 'target1.cif' → reference 'target1.pdb'). "
        "Supplying this auto-enables the 'dockq' metric. "
        "Requires: pip install DockQ",
    )

    # Forwarded single-run options
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help=f"Compute device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--peptide-chain",
        "--binder-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Peptide chain ID applied to all structures (auto-detect per structure if omitted)",
    )
    parser.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Receptor chain ID applied to all structures (auto-detect per structure if omitted)",
    )

    prep_group = parser.add_argument_group("Preparation")
    prep_group.add_argument("--skip-prep", action="store_true", help="Skip PDBFixer prep")
    prep_group.add_argument(
        "--ph",
        type=float,
        default=DEFAULT_PH,
        help=f"pH for hydrogen placement during prep (default: {DEFAULT_PH})",
    )
    prep_group.add_argument(
        "--keep-water",
        action="store_true",
        help="Retain crystallographic water molecules during prep",
    )
    prep_group.add_argument(
        "--canonicalize",
        action="store_true",
        help="Replace non-standard residues with standard equivalents",
    )

    relax_group = parser.add_argument_group("Relaxation")
    relax_group.add_argument("--skip-relax", action="store_true", help="Skip relaxation")
    relax_group.add_argument(
        "--md-duration-ps",
        type=float,
        default=DEFAULT_MD_DURATION_PS,
        help=f"MD duration in ps (0 = minimize only, default: {DEFAULT_MD_DURATION_PS:g})",
    )
    add_random_seed_arg(
        relax_group,
        "every stochastic step of each sample (hydrogen placement, MD velocities "
        "and thermostat); the same seed is used for all samples",
    )

    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=ALL_METRICS,
        metavar="METRICS",
        help=(
            "Comma-separated list of metrics to compute. "
            f"Valid: {', '.join(sorted(KNOWN_METRICS))}. "
            "Default: all reference-free metrics; 'dockq' is enabled by --reference-dir."
        ),
    )
    metrics_group.add_argument(
        "--energy-modes",
        nargs="+",
        choices=["raw", "relaxed", "after_md"],
        default=["relaxed"],
        help="Energy evaluation modes (default: relaxed)",
    )

    openfold_group = parser.add_argument_group("OpenFold")
    openfold_group.add_argument(
        "--openfold-mode",
        choices=["score", "refold"],
        default="score",
        help=OPENFOLD_MODE_HELP,
    )
    openfold_group.add_argument(
        "--openfold-conda-env",
        type=str,
        default="openfold3",
        help="Conda env where OpenFold3 is installed (default: openfold3)",
    )
    add_openfold_seeds_arg(openfold_group)
    add_openfold_cyclic_arg(openfold_group)
    add_openfold_no_msa_server_arg(openfold_group)
    add_on_unmappable_residue_arg(openfold_group)

    add_prediction_args(parser, batch=True)
    add_preflight_args(parser)

    from binding_metrics.cli import add_log_file_arg

    log_group = parser.add_argument_group("Logging")
    add_log_file_arg(log_group)
    log_group.add_argument(
        "--per-sample-log",
        action="store_true",
        help="Write a separate .log file for each sample inside its output "
        "directory (always on when --log-file is not set, ignored "
        "when --log-file is provided)",
    )

    add_config_arg(parser)

    args = parse_args_with_config(parser)
    check_prediction_args(parser, args)

    if args.per_sample_log and args.log_file is not None:
        print(
            "warning: --per-sample-log is ignored because --log-file was given; "
            f"all samples log to {args.log_file}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ Resolve dirs
    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir is not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output_dir or args.output_csv.parent
    if not args.preflight_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Collect inputs
    if args.glob:
        input_files = sorted(args.input_dir.glob(args.glob))
    else:
        input_files = sorted(
            p
            for p in args.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _STRUCTURE_SUFFIXES
        )

    if not input_files:
        print(f"ERROR: no structure files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'#' * 60}")
    print("  binding-metrics-batch")
    print(f"  Input dir:  {args.input_dir}  ({len(input_files)} structures)")
    print(f"  Output dir: {output_dir}")
    print(f"  Output CSV: {args.output_csv}")
    print(f"  Workers:    {args.workers}")
    print(f"{'#' * 60}\n")

    # ------------------------------------------------------------------ References
    # Map each sample (by filename stem) to a native structure for DockQ.
    # Supplying --reference-dir auto-enables the dockq metric.
    reference_map: dict[str, Path] = {}
    selected_metrics = args.metrics
    if args.reference_dir is not None:
        if not args.reference_dir.is_dir():
            print(
                f"ERROR: --reference-dir is not a directory: {args.reference_dir}", file=sys.stderr
            )
            sys.exit(1)
        reference_map = _build_reference_map(args.reference_dir)
        selected_metrics = selected_metrics | {"dockq"}
        matched = sum(1 for f in input_files if f.stem in reference_map)
        print(
            f"  References: {args.reference_dir}  "
            f"({matched}/{len(input_files)} samples matched by stem)\n"
        )

    if args.preflight_only:
        rows = run_batch(
            input_files,
            output_dir,
            skip_relax=args.skip_relax,
            peptide_chain=args.peptide_chain,
            receptor_chain=args.receptor_chain,
            metrics=selected_metrics,
            references=reference_map,
            predictor=args.predictor,
            prediction_dir=args.prediction_dir,
            prediction_mode=args.prediction_mode,
            prediction_weights=args.prediction_weights,
            openfold_mode=args.openfold_mode,
            on_unmappable_residue=args.on_unmappable_residue,
            binder_type=args.binder_type,
            on_incompatible=args.on_incompatible,
            preflight_only=True,
        )
        for row in rows:
            print(f"--- {row['sample_id']}: {row.get('preflight_status', 'error')}")
            print(row.get("preflight_plan") or row.get("batch_error", ""))
        n_refused = sum(1 for row in rows if row["batch_status"] == "error")
        print(f"\nPre-flight: {len(rows) - n_refused} of {len(rows)} samples can run")
        sys.exit(1 if n_refused else 0)

    # ------------------------------------------------------------------ Run
    # run_batch does the work; the callbacks print the progress lines. Started
    # items are announced only in the sequential case, where one item runs at a time.
    t_batch_start = time.time()
    n_inputs = len(input_files)
    started = 0
    finished = 0

    def announce_start(input_path: Path) -> None:
        nonlocal started
        started += 1
        print(f"[{started}/{n_inputs}] Processing: {input_path.stem}", flush=True)

    def report_finished(row: dict) -> None:
        nonlocal finished
        finished += 1
        status = row.get("batch_status", "ok")
        if args.workers == 1:
            if status == "ok":
                print(f"  -> ok  ({row.get('total_elapsed_s', '?')}s)", flush=True)
            elif status == "partial":
                print(f"  -> PARTIAL: failed steps: {row.get('batch_failed_steps')}", flush=True)
            else:
                print(f"  -> ERROR: {row.get('batch_error', '?')}", flush=True)
            return
        head = f"[{finished}/{n_inputs}] {row.get('sample_id')}"
        if status == "ok":
            print(f"{head} -> ok ({row.get('total_elapsed_s', '?')}s)", flush=True)
        elif status == "partial":
            print(f"{head} -> PARTIAL: failed steps: {row.get('batch_failed_steps')}", flush=True)
        elif isinstance(row, _LostSample):
            print(f"{head} -> FATAL: {row['batch_error'].partition(': ')[2]}", flush=True)
        else:
            print(f"{head} -> ERROR: {row.get('batch_error', '?')}", flush=True)

    rows = run_batch(
        input_files,
        output_dir,
        skip_prep=args.skip_prep,
        ph=args.ph,
        keep_water=args.keep_water,
        canonicalize=args.canonicalize,
        skip_relax=args.skip_relax,
        md_duration_ps=args.md_duration_ps,
        device=args.device,
        peptide_chain=args.peptide_chain,
        receptor_chain=args.receptor_chain,
        metrics=selected_metrics,
        energy_modes=args.energy_modes,
        references=reference_map,
        openfold_mode=args.openfold_mode,
        openfold_conda_env=args.openfold_conda_env,
        openfold_seeds=args.openfold_seeds,
        openfold_cyclic=args.openfold_cyclic,
        openfold_use_msa_server=not args.openfold_no_msa_server,
        on_unmappable_residue=args.on_unmappable_residue,
        predictor=args.predictor,
        prediction_dir=args.prediction_dir,
        prediction_binder_chain=args.prediction_binder_chain,
        prediction_target_chain=args.prediction_target_chain,
        prediction_cache=args.prediction_cache,
        rerun_predictions=args.rerun_predictions,
        prediction_mode=args.prediction_mode,
        prediction_weights=args.prediction_weights,
        prediction_cyclic=args.prediction_cyclic,
        prediction_use_msa_server=not args.prediction_no_msa_server,
        prediction_conda_env=args.prediction_conda_env,
        prediction_lock_threshold=args.prediction_lock_threshold,
        binder_type=args.binder_type,
        on_incompatible=args.on_incompatible,
        random_seed=args.random_seed,
        log_file=args.log_file,
        n_workers=args.workers,
        on_start=announce_start if args.workers == 1 else None,
        on_result=report_finished,
    )
    statuses = [row.get("batch_status", "ok") for row in rows]
    n_ok = statuses.count("ok")
    n_partial = statuses.count("partial")  # finished, but at least one pipeline step failed
    n_err = len(statuses) - n_ok - n_partial

    # ------------------------------------------------------------------ Write CSV
    # Written here, in the main process, after run_batch returned (i.e. all workers
    # are guaranteed to be done). There is exactly one writer and no worker can
    # race against it.
    # Collect the union of all column names (preserving insertion order via dict).
    all_keys: dict[str, None] = {}
    for row in rows:
        all_keys.update(dict.fromkeys(row.keys()))
    fieldnames = list(all_keys)

    import csv

    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    elapsed = round(time.time() - t_batch_start, 1)
    print(f"\n{'#' * 60}")
    partial_note = f", {n_partial} with failed steps" if n_partial else ""
    print(f"  DONE in {elapsed}s — {n_ok} ok, {n_err} error(s){partial_note}")
    print(f"  Results: {args.output_csv}")
    print(f"{'#' * 60}\n")

    # Non-zero only when no sample completed cleanly, as before; a sample whose
    # steps failed counts as not completed.
    sys.exit(1 if (n_err or n_partial) and n_ok == 0 else 0)


if __name__ == "__main__":
    main()
