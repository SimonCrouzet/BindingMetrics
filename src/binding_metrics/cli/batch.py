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
from binding_metrics.cli import (
    add_config_arg,
    add_on_unmappable_residue_arg,
    add_openfold_seeds_arg,
    add_random_seed_arg,
    check_on_unmappable_residue,
    on_unmappable_residue_kwargs,
    parse_args_with_config,
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
) -> dict:
    """Run the pipeline for a single structure and return a flat results dict.

    ``binder_chain`` and ``target_chain`` are keyword-only aliases of
    ``peptide_chain`` and ``receptor_chain``; both spellings with different IDs
    make the sample an ``"error"`` row, like any other pipeline failure. With
    ``raise_errors`` an exception from the pipeline is re-raised instead of
    becoming an ``"error"`` row.

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
            )
            results["total_elapsed_s"] = round(time.time() - t0, 1)

            write_report(results, sample_output_dir, sid, fmt="json")

    except Exception as e:
        if raise_errors:
            raise
        error_msg = f"{type(e).__name__}: {e}"
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


# ---------------------------------------------------------------------------
# Batched OpenFold (single subprocess for all samples)
# ---------------------------------------------------------------------------


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
) -> None:
    """Run OpenFold3 on all successful samples in a single subprocess.

    Modifies *rows* in-place, merging OF3 and EvoBind metrics into each
    sample's flat dict.  Also updates each sample's JSON report on disk.
    ``openfold_seeds`` are the seed values written to the query JSON; ``None``
    keeps the OpenFold default. ``on_unmappable_residue`` is passed to the query
    preparation of every sample; a residue OpenFold3 cannot take then stops the whole
    batch call (the error names each such residue) before the model starts.
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

    of_dir = output_dir / "_openfold_batch"
    try:
        predictions_dir = run_openfold_batched(
            samples=samples,
            output_dir=of_dir,
            mode=openfold_mode,
            conda_env=openfold_conda_env,
            **({"seeds": tuple(openfold_seeds)} if openfold_seeds else {}),
            **on_unmappable_residue_kwargs(on_unmappable_residue),
        )
    except Exception as e:  # noqa: BLE001 - the batch call spawns a subprocess; see openfold_error
        # Warning level keeps the line on stdout, where it was printed before.
        logger.warning("  [ERROR] Batched OpenFold failed: %s", e)
        import traceback

        traceback.print_exc()
        for idx in sid_to_row_idx.values():
            rows[idx]["openfold_error"] = str(e)
        return

    # Extract per-sample metrics and merge into rows
    for s in samples:
        sid = s.query_name
        idx = sid_to_row_idx[sid]
        chains = sid_to_chains[sid]
        pchain = chains["peptide"]
        rchain = chains["receptor"]

        try:
            if openfold_mode == "refold":
                of_metrics = compute_openfold_metrics(
                    output_dir=predictions_dir,
                    query_name=sid,
                    binder_chain=pchain,
                    receptor_chain=rchain,
                    reference_structure_path=sid_to_input[sid],
                )
            else:
                of_metrics = compute_openfold_metrics(
                    output_dir=predictions_dir,
                    query_name=sid,
                    binder_chain=pchain,
                    receptor_chain=rchain,
                )

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


def _update_sample_json(sample_dir: Path, sid: str, of_metrics: dict) -> None:
    """Merge OpenFold results into an existing per-sample JSON report."""
    import json

    from binding_metrics.protocols.report import _json_default

    json_path = sample_dir / f"{sid}_results.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["openfold"] = of_metrics
        json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - non-critical, the CSV row has the data anyway
        logger.warning("  %s: could not update %s: %s", sid, json_path.name, e)


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
            metric name, an ``on_unmappable_residue`` other than ``"error"`` or ``"x"``, or a
            chain given through both spellings with different IDs.
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

    input_paths = [Path(p) for p in paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = references or {}
    if references:
        selected = selected | {"dockq"}
    # Strip openfold from per-worker metrics: it runs as a single batched
    # subprocess after all other metrics finish.
    want_openfold = "openfold" in selected

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
    if want_openfold:
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
        )
    return finished


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
        help="score: both chains as templates; refold: binder predicted freely. Default: score",
    )
    openfold_group.add_argument(
        "--openfold-conda-env",
        type=str,
        default="openfold3",
        help="Conda env where OpenFold3 is installed (default: openfold3)",
    )
    add_openfold_seeds_arg(openfold_group)
    add_on_unmappable_residue_arg(openfold_group)

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
        on_unmappable_residue=args.on_unmappable_residue,
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
