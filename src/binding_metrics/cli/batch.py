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
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from binding_metrics.cli.run import ALL_METRICS, _parse_metrics, run_pipeline

_STRUCTURE_SUFFIXES = {".cif", ".pdb", ".mmcif"}


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
) -> dict:
    """Run the pipeline for a single structure and return a flat results dict."""
    from binding_metrics.cli import log_to_file
    from binding_metrics.protocols.report import write_report, _flatten

    sid = sample_id or input_path.stem
    sample_output_dir = output_dir / sid
    log_path = log_file if log_file else sample_output_dir / f"{sid}.log"

    t0 = time.time()
    error_msg: Optional[str] = None
    results: dict = {"sample_id": sid, "input": str(input_path)}

    sample_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with log_to_file(log_path):
            print(f"\n{'#'*60}")
            print(f"  binding-metrics-batch worker: {sid}")
            print(f"  Input:  {input_path}")
            print(f"  Output: {sample_output_dir}")
            print(f"  Log:    {log_path}")
            print(f"{'#'*60}")

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
                openfold_mode=openfold_mode,
                openfold_conda_env=openfold_conda_env,
            )
            results["total_elapsed_s"] = round(time.time() - t0, 1)

            write_report(results, sample_output_dir, sid, fmt="json")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        results["total_elapsed_s"] = round(time.time() - t0, 1)
        results["batch_error"] = error_msg

    flat = _flatten(results)
    flat["batch_status"] = "error" if error_msg else "ok"
    if error_msg:
        flat["batch_error"] = error_msg
    return flat


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
) -> None:
    """Run OpenFold3 on all successful samples in a single subprocess.

    Modifies *rows* in-place, merging OF3 and EvoBind metrics into each
    sample's flat dict.  Also updates each sample's JSON report on disk.
    """
    from binding_metrics.io.structures import detect_chains_from_file
    from binding_metrics.metrics.openfold import (
        _BatchSample,
        run_openfold_batched,
        compute_openfold_metrics,
    )
    from binding_metrics.protocols.report import _flatten

    # Build list of eligible samples (succeeded, have chain info)
    samples: list[_BatchSample] = []
    # Map query_name → row index for merging results back
    sid_to_row_idx: dict[str, int] = {}
    # Map query_name → chain info for metrics extraction
    sid_to_chains: dict[str, dict] = {}

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
        except Exception:
            continue

        pchain = chain_info.get("peptide_chain")
        rchain = chain_info.get("receptor_chain")
        if not pchain or not rchain:
            continue

        samples.append(_BatchSample(
            query_name=sid,
            complex_structure_path=input_path,
            receptor_chain=rchain,
            binder_chain=pchain,
        ))
        sid_to_row_idx[sid] = i
        sid_to_chains[sid] = {"peptide": pchain, "receptor": rchain}

    if not samples:
        print("  [skip] No eligible samples for batched OpenFold.", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"  Step: Batched OpenFold3 ({len(samples)} samples in one call)", flush=True)
    print(f"{'='*60}", flush=True)

    of_dir = output_dir / "_openfold_batch"
    try:
        predictions_dir = run_openfold_batched(
            samples=samples,
            output_dir=of_dir,
            mode=openfold_mode,
            conda_env=openfold_conda_env,
        )
    except Exception as e:
        print(f"  [ERROR] Batched OpenFold failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        for sid, idx in sid_to_row_idx.items():
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
                        compute_evobind_score,
                        compute_evobind_adversarial_check,
                    )
                    of_metrics.update(compute_evobind_score(
                        of_structure,
                        plddt_per_atom=plddt,
                        binder_chain=pchain,
                        receptor_chain=rchain,
                    ))
                    of_metrics.update(compute_evobind_adversarial_check(
                        design_structure_path=sid_to_input[sid],
                        afm_structure_path=of_structure,
                        binder_chain=pchain,
                        receptor_chain=rchain,
                        afm_plddt_per_atom=plddt,
                    ))
                except Exception as e:
                    of_metrics["evobind_error"] = str(e)

            # Flatten OF3 metrics into the row
            for k, v in _flatten({"openfold": of_metrics}).items():
                if k.startswith("openfold_"):
                    rows[idx][k] = v

            # Update per-sample JSON report on disk
            _update_sample_json(output_dir / sid, sid, of_metrics)

            print(f"  {sid}: ipTM={of_metrics.get('iptm', '?')}, "
                  f"pLDDT={of_metrics.get('avg_plddt', '?')}", flush=True)

        except Exception as e:
            print(f"  {sid}: OpenFold metrics failed: {e}", flush=True)
            rows[idx]["openfold_error"] = str(e)


def _update_sample_json(sample_dir: Path, sid: str, of_metrics: dict) -> None:
    """Merge OpenFold results into an existing per-sample JSON report."""
    import json
    json_path = sample_dir / f"{sid}_results.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        data["openfold"] = of_metrics
        json_path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass  # non-critical — CSV has the data anyway


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the binding-metrics pipeline on all structures in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Batch I/O
    parser.add_argument("--input-dir", "-i", type=Path, required=True,
                        help="Directory containing .cif / .pdb / .mmcif files")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Path for the aggregated CSV results file")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Directory for per-sample outputs "
                             "(default: same directory as --output-csv)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1). "
                             "Use --device cpu when workers > 1 on a single GPU.")
    parser.add_argument("--glob", type=str, default=None,
                        help="Optional glob pattern to filter files within --input-dir "
                             "(e.g. '*.cif'). Default: all .cif/.pdb/.mmcif files.")

    # Forwarded single-run options
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Compute device (default: cuda)")
    parser.add_argument("--peptide-chain", type=str, default=None,
                        help="Peptide chain ID applied to all structures "
                             "(auto-detect per structure if omitted)")
    parser.add_argument("--receptor-chain", type=str, default=None,
                        help="Receptor chain ID applied to all structures "
                             "(auto-detect per structure if omitted)")

    prep_group = parser.add_argument_group("Preparation")
    prep_group.add_argument("--skip-prep", action="store_true",
                            help="Skip PDBFixer prep")
    prep_group.add_argument("--ph", type=float, default=7.4,
                            help="pH for hydrogen placement during prep (default: 7.4)")
    prep_group.add_argument("--keep-water", action="store_true",
                            help="Retain crystallographic water molecules during prep")
    prep_group.add_argument("--canonicalize", action="store_true",
                            help="Replace non-standard residues with standard equivalents")

    relax_group = parser.add_argument_group("Relaxation")
    relax_group.add_argument("--skip-relax", action="store_true",
                             help="Skip relaxation")
    relax_group.add_argument("--md-duration-ps", type=float, default=200.0,
                             help="MD duration in ps (0 = minimize only, default: 200)")

    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=ALL_METRICS,
        metavar="METRICS",
        help=(
            "Comma-separated list of metrics to compute. "
            f"Valid: {', '.join(sorted(ALL_METRICS))}. "
            "Default: all."
        ),
    )
    metrics_group.add_argument("--energy-modes", nargs="+",
                               choices=["raw", "relaxed", "after_md"],
                               default=["relaxed"],
                               help="Energy evaluation modes (default: relaxed)")

    openfold_group = parser.add_argument_group("OpenFold")
    openfold_group.add_argument("--openfold-mode", choices=["score", "refold"], default="score",
                                help="score: both chains as templates; "
                                     "refold: binder predicted freely. Default: score")
    openfold_group.add_argument("--openfold-conda-env", type=str, default="openfold3",
                                help="Conda env where OpenFold3 is installed (default: openfold3)")

    from binding_metrics.cli import add_log_file_arg
    log_group = parser.add_argument_group("Logging")
    add_log_file_arg(log_group)
    log_group.add_argument("--per-sample-log", action="store_true",
                           help="Write a separate .log file for each sample inside its output "
                                "directory (always on when --log-file is not set, ignored "
                                "when --log-file is provided)")

    args = parser.parse_args()

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
            p for p in args.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _STRUCTURE_SUFFIXES
        )

    if not input_files:
        print(f"ERROR: no structure files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  binding-metrics-batch")
    print(f"  Input dir:  {args.input_dir}  ({len(input_files)} structures)")
    print(f"  Output dir: {output_dir}")
    print(f"  Output CSV: {args.output_csv}")
    print(f"  Workers:    {args.workers}")
    print(f"{'#'*60}\n")

    # ------------------------------------------------------------------ Build kwargs
    # Strip openfold from per-worker metrics — it will be run as a single
    # batched subprocess after all other metrics finish.
    want_openfold = "openfold" in args.metrics
    worker_metrics = args.metrics - {"openfold"}

    common_kwargs = dict(
        output_dir=output_dir,
        sample_id=None,          # derived from file stem per sample
        skip_prep=args.skip_prep,
        ph=args.ph,
        keep_water=args.keep_water,
        canonicalize=args.canonicalize,
        skip_relax=args.skip_relax,
        md_duration_ps=args.md_duration_ps,
        device=args.device,
        peptide_chain=args.peptide_chain,
        receptor_chain=args.receptor_chain,
        metrics=worker_metrics,
        energy_modes=tuple(args.energy_modes),
        openfold_mode=args.openfold_mode,
        openfold_conda_env=args.openfold_conda_env,
        log_file=args.log_file,   # None → per-sample log inside sample dir
    )

    # ------------------------------------------------------------------ Run
    t_batch_start = time.time()
    rows: list[dict] = []
    # Map sample_id → input_path for the batched OF3 call later
    sid_to_input: dict[str, Path] = {}
    n_ok = 0
    n_err = 0

    if args.workers == 1:
        # Sequential path — simpler stack traces, easier debugging.
        # `rows` is appended only here in the main process; no concurrency.
        for i, input_path in enumerate(input_files, 1):
            sid = input_path.stem
            sid_to_input[sid] = input_path
            print(f"[{i}/{len(input_files)}] Processing: {sid}", flush=True)
            flat = _run_one(input_path=input_path, **common_kwargs)
            rows.append(flat)
            status = flat.get("batch_status", "ok")
            if status == "ok":
                n_ok += 1
                print(f"  -> ok  ({flat.get('total_elapsed_s', '?')}s)", flush=True)
            else:
                n_err += 1
                print(f"  -> ERROR: {flat.get('batch_error', '?')}", flush=True)
    else:
        # Parallel path — each worker is an independent OS process:
        #   - No shared variables (separate address spaces).
        #   - No shared file paths (each worker writes to its own sample dir).
        #   - `rows` is appended only in the main process via as_completed(),
        #     which delivers results one at a time → no concurrent list mutation.
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for input_path in input_files:
                sid = input_path.stem
                sid_to_input[sid] = input_path
                fut = pool.submit(_run_one, input_path=input_path, **common_kwargs)
                futures[fut] = sid

            done = 0
            for fut in as_completed(futures):
                done += 1
                sid = futures[fut]
                try:
                    flat = fut.result()
                    rows.append(flat)
                    status = flat.get("batch_status", "ok")
                    if status == "ok":
                        n_ok += 1
                        print(f"[{done}/{len(input_files)}] {sid} -> ok "
                              f"({flat.get('total_elapsed_s', '?')}s)", flush=True)
                    else:
                        n_err += 1
                        print(f"[{done}/{len(input_files)}] {sid} -> ERROR: "
                              f"{flat.get('batch_error', '?')}", flush=True)
                except Exception as e:
                    n_err += 1
                    rows.append({"sample_id": sid, "batch_status": "error",
                                 "batch_error": f"{type(e).__name__}: {e}"})
                    print(f"[{done}/{len(input_files)}] {sid} -> FATAL: {e}", flush=True)

    # -------------------------------------------------------- Batched OpenFold
    if want_openfold:
        _run_batched_openfold(
            rows=rows,
            sid_to_input=sid_to_input,
            output_dir=output_dir,
            openfold_mode=args.openfold_mode,
            openfold_conda_env=args.openfold_conda_env,
            peptide_chain=args.peptide_chain,
            receptor_chain=args.receptor_chain,
        )

    # ------------------------------------------------------------------ Write CSV
    # Written here, in the main process, after the ProcessPoolExecutor context
    # manager exits (i.e. all workers are guaranteed to be done). There is
    # exactly one writer and no worker can race against it.
    # Collect the union of all column names (preserving insertion order via dict).
    all_keys: dict[str, None] = {}
    for row in rows:
        all_keys.update(dict.fromkeys(row.keys()))
    fieldnames = list(all_keys)

    import csv
    with open(args.output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    elapsed = round(time.time() - t_batch_start, 1)
    print(f"\n{'#'*60}")
    print(f"  DONE in {elapsed}s — {n_ok} ok, {n_err} error(s)")
    print(f"  Results: {args.output_csv}")
    print(f"{'#'*60}\n")

    sys.exit(1 if n_err and n_ok == 0 else 0)


if __name__ == "__main__":
    main()
