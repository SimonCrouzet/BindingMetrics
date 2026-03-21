"""Full binding-metrics pipeline: relax → energy → interface → geometry → electrostatics.

Runs every analysis step on a single structure and writes results to JSON.

Usage:
    binding-metrics-run \\
        --input complex.cif \\
        --output-dir results/ \\
        [--skip-relax] \\
        [--md-duration-ps 200] \\
        [--device cuda]
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional


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
    # relax
    skip_relax: bool = False,
    md_duration_ps: float = 200.0,
    device: str = "cuda",
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    # energy
    skip_energy: bool = False,
    energy_modes: tuple = ("relaxed",),
    # metrics
    skip_interface: bool = False,
    skip_geometry: bool = False,
    skip_electrostatics: bool = False,
    # openfold
    skip_openfold: bool = False,
    openfold_mode: str = "score",
    openfold_conda_env: Optional[str] = None,
) -> dict:
    """Run the full pipeline and return a results dict."""
    if sample_id is None:
        sample_id = input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"sample_id": sample_id, "input": str(input_path)}

    # ------------------------------------------------------------------ Relax
    relaxed_path: Optional[Path] = None
    if not skip_relax:
        _step("Relaxation (implicit MD)")
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        config = RelaxationConfig(
            md_duration_ps=md_duration_ps,
            device=device,
            peptide_chain_id=peptide_chain,
            receptor_chain_id=receptor_chain,
        )
        relaxer = ImplicitRelaxation(config)
        t0 = time.time()
        relax_result = relaxer.run(input_path, output_dir, sample_id=sample_id)
        elapsed = time.time() - t0

        results["relax"] = relax_result.to_dict()
        results["relax"]["elapsed_s"] = round(elapsed, 1)

        if not relax_result.success:
            print(f"\n[FAILED] Relaxation failed: {relax_result.error_message}")
            print("  Continuing with raw input for downstream steps...")
            relaxed_path = input_path
        else:
            # Prefer MD-final structure; fall back to minimized
            if relax_result.md_final_structure_path:
                relaxed_path = Path(relax_result.md_final_structure_path)
            else:
                relaxed_path = Path(relax_result.minimized_structure_path)
            print(f"\n  Relaxed structure: {relaxed_path}")
    else:
        print("\n  [skip] Relaxation skipped — using raw input for downstream steps.")
        relaxed_path = input_path
        results["relax"] = {"skipped": True}

    # ------------------------------------------------------------------ Energy
    if not skip_energy:
        _step("Interaction Energy")
        try:
            from binding_metrics.metrics.energy import compute_interaction_energy

            energy = compute_interaction_energy(
                relaxed_path,
                peptide_chain=peptide_chain,
                receptor_chain=receptor_chain,
                device=device,
                sample_id=sample_id,
                modes=energy_modes,
            )
            results["energy"] = energy
        except Exception as e:
            _warn(f"Energy computation failed: {e}")
            traceback.print_exc()
            results["energy"] = {"error": str(e)}
    else:
        results["energy"] = {"skipped": True}

    # --------------------------------------------------------------- Interface
    if not skip_interface:
        _step("Interface Metrics (SASA, H-bonds, salt bridges)")
        try:
            from binding_metrics.metrics.interface import compute_interface_metrics

            interface = compute_interface_metrics(
                relaxed_path,
                design_chain=peptide_chain,
                receptor_chain=receptor_chain,
            )
            results["interface"] = interface
        except Exception as e:
            _warn(f"Interface metrics failed: {e}")
            traceback.print_exc()
            results["interface"] = {"error": str(e)}
    else:
        results["interface"] = {"skipped": True}

    # --------------------------------------------------------------- Geometry
    if not skip_geometry:
        _step("Geometry (Ramachandran + omega planarity)")
        try:
            from binding_metrics.metrics.geometry import (
                compute_ramachandran,
                compute_omega_planarity,
            )

            rama = compute_ramachandran(relaxed_path, chain=peptide_chain)
            omega = compute_omega_planarity(relaxed_path, chain=peptide_chain)
            results["geometry"] = {"ramachandran": rama, "omega": omega}
        except Exception as e:
            _warn(f"Geometry metrics failed: {e}")
            traceback.print_exc()
            results["geometry"] = {"error": str(e)}
    else:
        results["geometry"] = {"skipped": True}

    # --------------------------------------------------------- Electrostatics
    if not skip_electrostatics:
        _step("Electrostatics (Coulomb cross-chain)")
        try:
            from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain

            elec = compute_coulomb_cross_chain(
                relaxed_path,
                peptide_chain=peptide_chain,
                receptor_chain=receptor_chain,
            )
            results["electrostatics"] = elec
        except Exception as e:
            _warn(f"Electrostatics failed: {e}")
            traceback.print_exc()
            results["electrostatics"] = {"error": str(e)}
    else:
        results["electrostatics"] = {"skipped": True}

    # --------------------------------------------------------- OpenFold
    if not skip_openfold:
        _step("OpenFold3 confidence scoring")
        try:
            from binding_metrics.metrics.openfold import (
                run_openfold_scoring,
                run_openfold_refolding,
                compute_openfold_metrics,
            )

            if not peptide_chain or not receptor_chain:
                _warn(
                    "OpenFold requires explicit --peptide-chain and --receptor-chain; skipping."
                )
                results["openfold"] = {"skipped": True}
            else:
                of_dir = output_dir / "openfold"
                if openfold_mode == "refold":
                    predictions_dir = run_openfold_refolding(
                        complex_structure_path=relaxed_path,
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
                        reference_structure_path=relaxed_path,
                    )
                else:  # score (default)
                    predictions_dir = run_openfold_scoring(
                        complex_structure_path=relaxed_path,
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
                results["openfold"] = of_metrics
        except Exception as e:
            _warn(f"OpenFold failed: {e}")
            traceback.print_exc()
            results["openfold"] = {"error": str(e)}
    else:
        results["openfold"] = {"skipped": True}

    return results


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

    # Relaxation
    relax_group = parser.add_argument_group("Relaxation")
    relax_group.add_argument("--skip-relax", action="store_true",
                             help="Skip relaxation; run metrics on raw input")
    relax_group.add_argument("--md-duration-ps", type=float, default=200.0,
                             help="MD duration in ps (0 = minimize only, default: 200)")

    # Energy
    energy_group = parser.add_argument_group("Energy")
    energy_group.add_argument("--skip-energy", action="store_true",
                              help="Skip interaction energy computation")
    energy_group.add_argument("--energy-modes", nargs="+",
                              choices=["raw", "relaxed", "after_md"],
                              default=["relaxed"],
                              help="Energy evaluation modes (default: relaxed)")

    # Metrics toggles
    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument("--skip-interface", action="store_true",
                               help="Skip interface metrics (SASA, H-bonds, salt bridges)")
    metrics_group.add_argument("--skip-geometry", action="store_true",
                               help="Skip geometry metrics (Ramachandran, omega)")
    metrics_group.add_argument("--skip-electrostatics", action="store_true",
                               help="Skip electrostatics (Coulomb cross-chain)")

    # OpenFold
    openfold_group = parser.add_argument_group("OpenFold")
    openfold_group.add_argument("--skip-openfold", action="store_true",
                                help="Skip OpenFold3 confidence scoring")
    openfold_group.add_argument("--openfold-mode", choices=["score", "refold"], default="score",
                                help="score: both chains as templates (confidence); "
                                     "refold: binder predicted freely (refolding RMSD). "
                                     "Default: score")
    openfold_group.add_argument("--openfold-conda-env", type=str, default=None,
                                help="Conda environment name where OpenFold3 is installed")

    # Report
    report_group = parser.add_argument_group("Report")
    report_group.add_argument("--format", choices=["json", "csv"], default="json",
                              dest="fmt", help="Results output format (default: json)")
    report_group.add_argument("--summary", action="store_true",
                              help="Also write a human-readable *_report.md")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    sample_id = args.sample_id or args.input.stem
    print(f"\n{'#'*60}")
    print(f"  binding-metrics-run: {sample_id}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*60}")

    t_total = time.time()
    results = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_id=sample_id,
        skip_relax=args.skip_relax,
        md_duration_ps=args.md_duration_ps,
        device=args.device,
        peptide_chain=args.peptide_chain,
        receptor_chain=args.receptor_chain,
        skip_energy=args.skip_energy,
        energy_modes=tuple(args.energy_modes),
        skip_interface=args.skip_interface,
        skip_geometry=args.skip_geometry,
        skip_electrostatics=args.skip_electrostatics,
        skip_openfold=args.skip_openfold,
        openfold_mode=args.openfold_mode,
        openfold_conda_env=args.openfold_conda_env,
    )
    results["total_elapsed_s"] = round(time.time() - t_total, 1)

    from binding_metrics.protocols.report import write_report
    results_path = write_report(results, args.output_dir, sample_id,
                                fmt=args.fmt, summary=args.summary)

    print(f"\n{'#'*60}")
    print(f"  DONE in {results['total_elapsed_s']}s")
    print(f"  Results: {results_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
