"""Full binding-metrics pipeline: prep → relax → energy → interface → geometry → electrostatics.

Runs every analysis step on a single structure and writes results to JSON.

Usage:
    binding-metrics-run \\
        --input complex.cif \\
        --output-dir results/ \\
        [--skip-prep] [--skip-relax] \\
        [--ph 7.4] \\
        [--metrics energy,interface,geometry,electrostatics,openfold] \\
        [--md-duration-ps 200] \\
        [--device cuda]
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from binding_metrics.core.system import DEFAULT_RANDOM_SEED

ALL_METRICS = frozenset({"energy", "interface", "geometry", "electrostatics", "openfold"})
# Reference-based metrics require a native structure (--reference) and are not
# part of the default set; they are auto-enabled when a reference is supplied.
REFERENCE_METRICS = frozenset({"dockq"})
KNOWN_METRICS = ALL_METRICS | REFERENCE_METRICS


def _seed_arg(value: str) -> Optional[int]:
    """Parse --random-seed: an integer, or 'none'/'random' for fresh randomness."""
    if value.strip().lower() in ("none", "random", "off"):
        return None
    return int(value)


def _warn(msg: str) -> None:
    print(f"  [warning] {msg}", flush=True)


def _step(name: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"  Step: {name}", flush=True)
    print(f"{'='*60}", flush=True)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    sample_id: Optional[str] = None,
    # prep
    skip_prep: bool = False,
    ph: float = 7.4,
    keep_water: bool = False,
    canonicalize: bool = False,
    # relax
    skip_relax: bool = False,
    md_duration_ps: float = 200.0,
    device: str = "cuda",
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    # metrics
    metrics: frozenset = ALL_METRICS,
    energy_modes: tuple = ("relaxed",),
    # reference-based (dockq)
    reference_path: Optional[Path] = None,
    # openfold
    openfold_mode: str = "score",
    openfold_conda_env: Optional[str] = None,
    # reproducibility
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> dict:
    """Run the full pipeline and return a results dict."""
    if sample_id is None:
        sample_id = input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"sample_id": sample_id, "input": str(input_path)}

    # ---------------------------------------------------------- Chain detection
    from binding_metrics.io.structures import detect_chains_from_file

    chain_info = detect_chains_from_file(
        input_path,
        peptide_chain=peptide_chain,
        receptor_chain=receptor_chain,
        verbose=True,
    )
    peptide_chain = chain_info["peptide_chain"]           # auth_asym_id (biotite)
    receptor_chain = chain_info["receptor_chain"]
    peptide_chain_label = chain_info["peptide_chain_label"]   # label_asym_id (OpenMM)
    receptor_chain_label = chain_info["receptor_chain_label"]
    results["chains"] = chain_info

    # ------------------------------------------------------------------- Prep
    prepped_path = input_path
    if not skip_prep:
        _step("Structure Preparation (PDBFixer)")
        try:
            from binding_metrics.core.system import HAS_PDBFIXER, prep_structure
            from binding_metrics.io.structures import load_structure, save_structure

            if not HAS_PDBFIXER:
                _warn("pdbfixer not available — skipping prep. Install with: pip install binding-metrics[structure]")
            else:
                topology, positions = load_structure(input_path)
                topology, positions = prep_structure(
                    topology, positions, ph=ph,
                    keep_water=keep_water, canonicalize=canonicalize,
                    random_seed=random_seed,
                )
                prepped_path = output_dir / f"{sample_id}_cleaned.cif"
                save_structure(topology, positions, prepped_path, source_path=input_path)
                print(f"  Prepped structure: {prepped_path}")
                results["prep"] = {"output": str(prepped_path), "ph": ph, "keep_water": keep_water}
                # save_cif preserves original auth IDs and aligns label IDs to match,
                # so downstream OpenMM steps will see the original chain IDs.
                # Re-detect from the cleaned file so peptide_chain_label is up-to-date.
                prepped_chain_info = detect_chains_from_file(
                    prepped_path,
                    peptide_chain=peptide_chain,
                    receptor_chain=receptor_chain,
                )
                peptide_chain_label  = prepped_chain_info["peptide_chain_label"]
                receptor_chain_label = prepped_chain_info["receptor_chain_label"]
        except Exception as e:
            _warn(f"Prep failed: {e} — continuing with raw input")
            traceback.print_exc()
            prepped_path = input_path
            results["prep"] = {"error": str(e)}
    else:
        print("\n  [skip] Prep skipped — using raw input.")
        results["prep"] = {"skipped": True}

    # ------------------------------------------------------------------ Relax
    # Detect cyclic bonds from the original file before PDBFixer strips STRUCT_CONN.
    # Passed as hints to relaxation so cyclization survives the prep round-trip.
    cyclic_bond_hints = []
    try:
        from binding_metrics.core.cyclic import detect_cyclization
        from binding_metrics.io.structures import load_structure
        _orig_topo, _orig_pos = load_structure(input_path)
        cyclic_bond_hints = detect_cyclization(_orig_topo, _orig_pos, peptide_chain_label)
        if cyclic_bond_hints:
            print(f"  Cyclic bond hints from original file: "
                  f"{[b.cyclic_type for b in cyclic_bond_hints]}")
    except Exception:
        pass

    relaxed_path: Optional[Path] = None
    if not skip_relax:
        _step("Relaxation (implicit MD)")
        if device == "cpu" and md_duration_ps > 0:
            print(
                "\n  *** WARNING: running MD on CPU is extremely slow and not recommended. ***\n"
                "  *** For production use, run on a CUDA-capable GPU (--device cuda).   ***\n"
                "  *** Use --md-duration-ps 0 to minimize only if GPU is unavailable.   ***\n",
                flush=True,
            )
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        config = RelaxationConfig(
            md_duration_ps=md_duration_ps,
            device=device,
            peptide_chain_id=peptide_chain_label,
            receptor_chain_id=receptor_chain_label,
            cyclic_bond_hints=cyclic_bond_hints or None,
            # Auto-parameterise any non-canonical residue (e.g. cyclosporin's
            # BMT/ABA) with GAFF2 ExternalBond templates so relaxation builds.
            small_molecules="auto",
            random_seed=random_seed,
        )
        relaxer = ImplicitRelaxation(config)
        t0 = time.time()
        relax_result = relaxer.run(prepped_path, output_dir, sample_id=sample_id)
        elapsed = time.time() - t0

        results["relax"] = relax_result.to_dict()
        results["relax"]["elapsed_s"] = round(elapsed, 1)

        if not relax_result.success:
            print(f"\n[FAILED] Relaxation failed: {relax_result.error_message}")
            print("  Continuing with prepped input for downstream steps...")
            relaxed_path = prepped_path
            working_peptide = peptide_chain_label
            working_receptor = receptor_chain_label
        else:
            # Prefer MD-final structure; fall back to minimized
            if relax_result.md_final_structure_path:
                relaxed_path = Path(relax_result.md_final_structure_path)
            else:
                relaxed_path = Path(relax_result.minimized_structure_path)
            print(f"\n  Relaxed structure: {relaxed_path}")
            # OpenMM writes the relaxed CIF using label IDs as both auth and label,
            # so all downstream steps should use the label IDs.
            working_peptide = peptide_chain_label
            working_receptor = receptor_chain_label
    else:
        # Prefer PDBFixer-prepped structure (proper termini, removed heterogens)
        # over raw input; fall back to raw only if prep was skipped or failed.
        relaxed_path = prepped_path if prepped_path != input_path else input_path
        working_peptide = peptide_chain_label
        working_receptor = receptor_chain_label
        print(f"\n  [skip] Relaxation skipped — using {'prepped' if relaxed_path != input_path else 'raw'} input for downstream steps.")
        results["relax"] = {"skipped": True}

    # ------------------------------------------------------------------ Energy
    if "energy" in metrics:
        _step("Interaction Energy")
        try:
            from binding_metrics.metrics.energy import compute_interaction_energy

            energy = compute_interaction_energy(
                relaxed_path,
                peptide_chain=peptide_chain_label,
                receptor_chain=receptor_chain_label,
                device=device,
                sample_id=sample_id,
                modes=energy_modes,
                ph=ph,
                random_seed=random_seed,
            )
            results["energy"] = energy
        except Exception as e:
            _warn(f"Energy computation failed: {e}")
            traceback.print_exc()
            results["energy"] = {"error": str(e)}
    else:
        results["energy"] = {"skipped": True}

    # --------------------------------------------------------------- Interface
    if "interface" in metrics:
        _step("Interface Metrics (SASA, H-bonds, salt bridges)")
        try:
            from binding_metrics.metrics.interface import compute_interface_metrics

            interface = compute_interface_metrics(
                relaxed_path,
                design_chain=working_peptide,
                receptor_chain=working_receptor,
            )
            results["interface"] = interface
        except Exception as e:
            _warn(f"Interface metrics failed: {e}")
            traceback.print_exc()
            results["interface"] = {"error": str(e)}
    else:
        results["interface"] = {"skipped": True}

    # --------------------------------------------------------------- Geometry
    if "geometry" in metrics:
        _step("Geometry (Ramachandran + omega planarity + shape complementarity)")
        try:
            from binding_metrics.metrics.geometry import (
                compute_omega_planarity,
                compute_ramachandran,
                compute_shape_complementarity,
            )

            rama = compute_ramachandran(relaxed_path, chain=working_peptide)
            omega = compute_omega_planarity(relaxed_path, chain=working_peptide)
            sc = compute_shape_complementarity(
                relaxed_path,
                peptide_chain=working_peptide,
                receptor_chain=working_receptor,
            )
            results["geometry"] = {"ramachandran": rama, "omega": omega, "shape_complementarity": sc}
        except Exception as e:
            _warn(f"Geometry metrics failed: {e}")
            traceback.print_exc()
            results["geometry"] = {"error": str(e)}
    else:
        results["geometry"] = {"skipped": True}

    # --------------------------------------------------------- Electrostatics
    if "electrostatics" in metrics:
        _step("Electrostatics (Coulomb cross-chain)")
        try:
            from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

            elec = compute_coulomb_cross_chain(
                relaxed_path,
                peptide_chain=working_peptide,
                receptor_chain=working_receptor,
            )
            results["electrostatics"] = elec
        except Exception as e:
            _warn(f"Electrostatics failed: {e}")
            traceback.print_exc()
            results["electrostatics"] = {"error": str(e)}
    else:
        results["electrostatics"] = {"skipped": True}

    # ----------------------------------------------- DockQ (reference-based)
    if "dockq" in metrics:
        _step("DockQ CAPRI accuracy (vs reference)")
        if reference_path is None:
            _warn("DockQ requested but no --reference supplied; skipping.")
            results["dockq"] = {"skipped": True}
        else:
            try:
                from binding_metrics.metrics.dockq import compute_dockq_metrics

                # Score the prediction *as submitted* (original input coordinates)
                # against the reference: CAPRI evaluates the predicted coordinates,
                # so we deliberately use input_path, not the prepped/relaxed pose
                # (prep + MD would move atoms and confound the accuracy measure).
                dockq = compute_dockq_metrics(input_path, reference_path)
                results["dockq"] = dockq
                score = dockq.get("dockq")
                if score is not None:
                    print(f"  DockQ: {score:.3f} ({dockq.get('capri_class')})")
            except Exception as e:
                _warn(f"DockQ failed: {e}")
                traceback.print_exc()
                results["dockq"] = {"error": str(e)}
    else:
        results["dockq"] = {"skipped": True}

    # --------------------------------------------------------- OpenFold
    if "openfold" in metrics:
        _step("OpenFold3 confidence scoring")
        try:
            from binding_metrics.metrics.openfold import (
                compute_openfold_metrics,
                run_openfold_refolding,
                run_openfold_scoring,
            )

            if not peptide_chain or not receptor_chain:
                _warn(
                    "OpenFold requires --peptide-chain and --receptor-chain (or auto-detect); skipping."
                )
                results["openfold"] = {"skipped": True}
            else:
                of_dir = output_dir / "openfold"
                if openfold_mode == "refold":
                    predictions_dir = run_openfold_refolding(
                        complex_structure_path=input_path,
                        receptor_chain=receptor_chain,
                        binder_chain=peptide_chain,
                        query_name=sample_id,
                        output_dir=of_dir,
                        conda_env=openfold_conda_env,
                    )
                    of_metrics = compute_openfold_metrics(
                        output_dir=predictions_dir,
                        query_name=sample_id,
                        binder_chain=peptide_chain,
                        receptor_chain=receptor_chain,
                        reference_structure_path=input_path,
                    )
                else:  # score (default)
                    predictions_dir = run_openfold_scoring(
                        complex_structure_path=input_path,
                        receptor_chain=receptor_chain,
                        binder_chain=peptide_chain,
                        query_name=sample_id,
                        output_dir=of_dir,
                        conda_env=openfold_conda_env,
                    )
                    of_metrics = compute_openfold_metrics(
                        output_dir=predictions_dir,
                        query_name=sample_id,
                        binder_chain=peptide_chain,
                        receptor_chain=receptor_chain,
                    )
                # EvoBind metrics — no extra model calls, reuse OF3 outputs
                of_structure = of_metrics.get("structure_path")
                plddt = of_metrics.get("plddt_per_atom")
                if of_structure:
                    from binding_metrics.metrics.evobind import (
                        compute_evobind_adversarial_check,
                        compute_evobind_score,
                    )
                    # Primary score on the OF3 prediction
                    try:
                        of_metrics.update(compute_evobind_score(
                            of_structure,
                            plddt_per_atom=plddt,
                            binder_chain=peptide_chain,
                            receptor_chain=receptor_chain,
                        ))
                    except Exception as e:
                        _warn(f"EvoBind score failed: {e}")
                        of_metrics["evobind_error"] = str(e)

                    # Adversarial check: does the OF3 prediction agree with the
                    # input design pose? Large ΔCOM = OF3 places the binder
                    # elsewhere → design pose not supported by the prediction.
                    try:
                        of_metrics.update(compute_evobind_adversarial_check(
                            design_structure_path=input_path,
                            afm_structure_path=of_structure,
                            binder_chain=peptide_chain,
                            receptor_chain=receptor_chain,
                            afm_plddt_per_atom=plddt,
                        ))
                    except Exception as e:
                        _warn(f"EvoBind adversarial check failed: {e}")
                        of_metrics["adversarial_error"] = str(e)

                results["openfold"] = of_metrics

        except Exception as e:
            _warn(f"OpenFold failed: {e}")
            traceback.print_exc()
            results["openfold"] = {"error": str(e)}
    else:
        results["openfold"] = {"skipped": True}

    return results


def _collect_failures(results: dict) -> list:
    """Return ``(step, reason)`` for every metric step that did not complete.

    A step counts as failed when its result dict carries an ``error`` key, or
    when a step that reports a ``success`` flag (relaxation, energy) has
    ``success is False``. Steps marked ``{"skipped": True}`` are not failures.
    Used to make the pipeline exit non-zero instead of silently reporting
    success when metrics could not be computed.
    """
    failures = []
    for step, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue
        if "error" in res:
            failures.append((step, str(res["error"])[:200]))
        elif res.get("success") is False:
            reason = res.get("error_message") or "step reported success=False"
            failures.append((step, str(reason)[:200]))
    return failures


def _parse_metrics(value: str) -> frozenset:
    names = {v.strip() for v in value.split(",")}
    unknown = names - KNOWN_METRICS
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Valid choices: {', '.join(sorted(KNOWN_METRICS))}"
        )
    return frozenset(names)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full binding-metrics pipeline on a single structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input CIF or PDB file")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Directory to write all outputs")
    parser.add_argument("--sample-id", type=str, default=None,
                        help="Sample identifier (defaults to input file stem)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Compute device (default: cuda)")
    parser.add_argument("--peptide-chain", type=str, default=None,
                        help="Peptide chain ID (auto-detect if omitted)")
    parser.add_argument("--receptor-chain", type=str, default=None,
                        help="Receptor chain ID (auto-detect if omitted)")
    parser.add_argument("--reference", "--native", dest="reference", type=Path, default=None,
                        help="Reference/native complex for reference-based accuracy "
                             "metrics (DockQ, fnat, i-RMSD, L-RMSD). Supplying this "
                             "auto-enables the 'dockq' metric. Requires: pip install DockQ")

    # Prep
    prep_group = parser.add_argument_group("Preparation")
    prep_group.add_argument("--skip-prep", action="store_true",
                            help="Skip PDBFixer prep; run relax on raw input")
    prep_group.add_argument("--ph", type=float, default=7.4,
                            help="pH for hydrogen placement during prep (default: 7.4)")
    prep_group.add_argument("--keep-water", action="store_true",
                            help="Retain crystallographic water molecules during prep")
    prep_group.add_argument(
        "--canonicalize", action="store_true",
        help=(
            "Replace non-standard residues with standard equivalents during prep "
            "(e.g. MSE→MET, SEP→SER). By default they are preserved for GAFF2 "
            "parameterisation in relax (--small-molecules auto)."
        ),
    )

    # Relaxation
    relax_group = parser.add_argument_group("Relaxation")
    relax_group.add_argument("--skip-relax", action="store_true",
                             help="Skip relaxation; run metrics on raw input")
    relax_group.add_argument("--md-duration-ps", type=float, default=200.0,
                             help="MD duration in ps (0 = minimize only, default: 200)")
    relax_group.add_argument(
        "--random-seed", type=_seed_arg, default=DEFAULT_RANDOM_SEED,
        metavar="INT|none",
        help=(
            "Seed for all stochastic steps (hydrogen placement, MD velocities "
            "and thermostat). A fixed integer makes the run reproducible "
            f"(default: {DEFAULT_RANDOM_SEED}); pass 'none' for fresh randomness "
            "each run, e.g. to generate independent MD replicas."
        ),
    )

    # Metrics
    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=ALL_METRICS,
        metavar="METRICS",
        help=(
            "Comma-separated list of metrics to compute. "
            f"Valid: {', '.join(sorted(KNOWN_METRICS))}. "
            "Default: all reference-free metrics. 'dockq' also needs --reference."
        ),
    )
    metrics_group.add_argument("--energy-modes", nargs="+",
                               choices=["raw", "relaxed", "after_md"],
                               default=["relaxed"],
                               help="Energy evaluation modes (default: relaxed)")

    # OpenFold
    openfold_group = parser.add_argument_group("OpenFold")
    openfold_group.add_argument("--openfold-mode", choices=["score", "refold"], default="score",
                                help="score: both chains as templates (confidence); "
                                     "refold: binder predicted freely (refolding RMSD). "
                                     "Default: score")
    openfold_group.add_argument("--openfold-conda-env", type=str, default="openfold3",
                                help="Conda environment name where OpenFold3 is installed "
                                     "(default: openfold3). Set to empty string to use "
                                     "the current environment if openfold3 is installed there.")

    # Report
    report_group = parser.add_argument_group("Report")
    report_group.add_argument("--format", choices=["json", "csv"], default="json",
                              dest="fmt", help="Results output format (default: json)")
    report_group.add_argument("--summary", action="store_true",
                              help="Also write a human-readable summary (*_report.md or *_report.html)")
    report_group.add_argument("--summary-format", choices=["md", "html"], default="md",
                              dest="summary_format", help="Summary format (default: md)")
    from binding_metrics.cli import add_log_file_arg
    add_log_file_arg(report_group)

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # A reference auto-enables DockQ; validate it exists up front.
    metrics = args.metrics
    if args.reference is not None:
        if not args.reference.exists():
            print(f"ERROR: reference file not found: {args.reference}", file=sys.stderr)
            sys.exit(1)
        metrics = metrics | {"dockq"}

    from binding_metrics.cli import log_to_file
    with log_to_file(args.log_file):
        sample_id = args.sample_id or args.input.stem
        print(f"\n{'#'*60}")
        print(f"  binding-metrics-run: {sample_id}")
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output_dir}")
        if args.log_file:
            print(f"  Log:    {args.log_file}")
        print(f"{'#'*60}")

        t_total = time.time()
        results = run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            sample_id=sample_id,
            skip_prep=args.skip_prep,
            ph=args.ph,
            keep_water=args.keep_water,
            canonicalize=args.canonicalize,
            skip_relax=args.skip_relax,
            md_duration_ps=args.md_duration_ps,
            device=args.device,
            peptide_chain=args.peptide_chain,
            receptor_chain=args.receptor_chain,
            metrics=metrics,
            energy_modes=tuple(args.energy_modes),
            reference_path=args.reference,
            openfold_mode=args.openfold_mode,
            openfold_conda_env=args.openfold_conda_env,
            random_seed=args.random_seed,
        )
        results["total_elapsed_s"] = round(time.time() - t_total, 1)

        from binding_metrics.protocols.report import write_report
        results_path = write_report(results, args.output_dir, sample_id,
                                    fmt=args.fmt, summary=args.summary,
                                    summary_format=args.summary_format)

        failures = _collect_failures(results)
        if failures:
            print(f"\n{'#'*60}")
            print(f"  FAILED in {results['total_elapsed_s']}s — "
                  f"{len(failures)} step(s) did not complete:")
            for step, reason in failures:
                print(f"    [x] {step}: {reason}")
            print(f"  Partial results: {results_path}")
            print(f"{'#'*60}\n")
            sys.exit(1)

        print(f"\n{'#'*60}")
        print(f"  DONE in {results['total_elapsed_s']}s")
        print(f"  Results: {results_path}")
        print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
