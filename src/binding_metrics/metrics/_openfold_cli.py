"""Command-line interface of ``binding-metrics-openfold``.

The API lives in :mod:`binding_metrics.metrics.openfold`, which re-exports
``main`` so the console script and ``python -m binding_metrics.metrics.openfold``
keep working. The commands look their functions up on that module when they run,
so patching ``binding_metrics.metrics.openfold.<name>`` still redirects them.
"""

import argparse
from pathlib import Path

import numpy as np

from binding_metrics.metrics._common import ChainAliasAction
from binding_metrics.metrics._openfold_run import _DEFAULT_MODEL_PRESETS
from binding_metrics.utils import configure_logging

_PRESETS_TEXT = " ".join(_DEFAULT_MODEL_PRESETS)


def _add_parse_args(p, include_chain_args: bool = False) -> None:
    """Add common parse/metrics arguments to a subparser."""
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="1-based index of the seed directory to parse, not a random seed value (default: 1).",
    )
    p.add_argument("--sample", type=int, default=1, help="Sample index (default: 1).")
    p.add_argument(
        "--include-matrices",
        action="store_true",
        help="Include full PDE matrix in output (large).",
    )
    if include_chain_args:
        p.add_argument(
            "--binder-chain",
            type=str,
            default=None,
            metavar="CHAIN",
            help="Chain ID of the binder. Enables per-residue pLDDT and binder RMSD.",
        )
        p.add_argument(
            "--receptor-chain",
            "--target-chain",
            action=ChainAliasAction,
            type=str,
            default=None,
            metavar="CHAIN",
            help="Chain ID of the receptor. Enables interface PAE and receptor-frame RMSD.",
        )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        metavar="CIF",
        help="Reference structure CIF/PDB for binder Cα RMSD (requires --binder-chain).",
    )


def _add_query_seeds_arg(p) -> None:
    """Add ``--seeds`` (seed values for the query JSON) to a subparser."""
    from binding_metrics.metrics import openfold as of

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(of._DEFAULT_QUERY_SEEDS),
        metavar="SEED",
        help="Seed values written to the query JSON (default: %(default)s).",
    )


def _print_metrics(metrics: dict, seed: int, sample: int) -> None:
    """Print OpenFold3 metrics to stdout."""
    print(f"\nOpenFold3 confidence metrics (seed={seed}, sample={sample}):")
    print(f"  Structure:            {metrics['structure_path'] or 'not found'}")
    print(f"  Atoms:                {metrics['n_atoms']}")

    def _fmt(label, val, unit=""):
        if isinstance(val, float) and np.isnan(val):
            print(f"  {label:<26} N/A")
        else:
            print(f"  {label:<26} {val:.4f}{unit}")

    _fmt("avg_pLDDT [0–100]:", metrics["avg_plddt"])
    _fmt("gPDE (Å):", metrics["gpde"])
    _fmt("pTM [0–1]:", metrics["ptm"])
    _fmt("ipTM [0–1]:", metrics["iptm"])
    _fmt("Disorder:", metrics["disorder"])
    _fmt("has_clash:", metrics["has_clash"])
    _fmt("Ranking score:", metrics["sample_ranking_score"])
    _fmt("Max PDE (Å):", metrics["max_pde"])

    if not np.isnan(metrics.get("binder_avg_plddt", float("nan"))):
        _fmt("Binder avg pLDDT:", metrics["binder_avg_plddt"])
    if not np.isnan(metrics.get("mean_interface_pde", float("nan"))):
        _fmt("Interface PDE mean (Å):", metrics["mean_interface_pde"])
        _fmt("Interface PDE max (Å):", metrics["max_interface_pde"])
    if not np.isnan(metrics.get("binder_ca_rmsd", float("nan"))):
        _fmt("Binder Cα RMSD (Å):", metrics["binder_ca_rmsd"])

    if metrics.get("chain_ptm"):
        print("\n  Per-chain pTM:")
        for chain, val in metrics["chain_ptm"].items():
            print(f"    chain {chain}: {float(val):.4f}")

    if metrics.get("chain_pair_iptm"):
        print("\n  Chain-pair ipTM:")
        for pair, val in metrics["chain_pair_iptm"].items():
            print(f"    {pair}: {float(val):.4f}")

    if metrics.get("binder_plddt_per_residue") is not None:
        arr = metrics["binder_plddt_per_residue"]
        print(f"\n  Binder per-residue pLDDT ({len(arr)} residues):")
        print(f"    Min: {arr.min():.1f}  Median: {np.median(arr):.1f}  Max: {arr.max():.1f}")
        print(
            f"    ≥90: {int((arr >= 90).sum())}  70–89: {int(((arr >= 70) & (arr < 90)).sum())}"
            f"  50–69: {int(((arr >= 50) & (arr < 70)).sum())}  <50: {int((arr < 50).sum())}"
        )

    if metrics.get("timing"):
        print("\nTiming:")
        for k, v in metrics["timing"].items():
            print(f"  {k}: {v:.2f}s" if isinstance(v, (int, float)) else f"  {k}: {v}")

    if metrics.get("plddt_per_atom") is not None:
        arr = metrics["plddt_per_atom"]
        print(f"\nPer-atom pLDDT summary ({len(arr)} atoms):")
        print(f"  Min: {arr.min():.1f}  Median: {np.median(arr):.1f}  Max: {arr.max():.1f}")
        print(
            f"  ≥90: {int((arr >= 90).sum())}  70–89: {int(((arr >= 70) & (arr < 90)).sum())}"
            f"  50–69: {int(((arr >= 50) & (arr < 70)).sum())}  <50: {int((arr < 50).sum())}"
        )


def main():
    configure_logging()
    from binding_metrics.metrics import openfold as of

    parser = argparse.ArgumentParser(
        description=(
            "OpenFold3 confidence metrics and structure prediction.\n\n"
            "Subcommands:\n"
            "  parse         Parse metrics from existing OF3 output.\n"
            "  run           Run OF3 inference, then parse metrics.\n"
            "  prepare-query Prepare a query JSON for binder refolding.\n"
            "  refold        Run OF3 binder refolding (receptor fixed as template)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- parse subcommand ---
    p_parse = sub.add_parser(
        "parse",
        help="Parse metrics from an existing OpenFold3 output directory.",
    )
    p_parse.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="OpenFold3 output directory.",
    )
    p_parse.add_argument(
        "--query-name",
        "-n",
        type=str,
        required=True,
        help="Query name (as specified in the input JSON).",
    )
    _add_parse_args(p_parse, include_chain_args=True)

    # --- run subcommand ---
    p_run = sub.add_parser(
        "run",
        help="Run OpenFold3 inference, then parse and print metrics.",
    )
    p_run.add_argument(
        "--query-json",
        type=Path,
        required=True,
        help="Input JSON file describing the prediction query.",
    )
    p_run.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory.",
    )
    p_run.add_argument(
        "--query-name",
        "-n",
        type=str,
        required=True,
        help="Query name to parse after inference.",
    )
    p_run.add_argument(
        "--ckpt", type=Path, default=None, help="Model checkpoint path (uses default if omitted)."
    )
    p_run.add_argument(
        "--num-samples", type=int, default=5, help="Number of diffusion samples (default: 5)."
    )
    p_run.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Passed to OpenFold3 as --num_model_seeds (default: 1).",
    )
    p_run.add_argument(
        "--no-msa-server",
        action="store_true",
        help="Disable ColabFold MSA server (use pre-computed MSAs).",
    )
    p_run.add_argument(
        "--presets",
        nargs="+",
        default=list(_DEFAULT_MODEL_PRESETS),
        metavar="PRESET",
        help=f"Model configuration presets (default: {_PRESETS_TEXT}).",
    )
    p_run.add_argument(
        "--runner-yaml",
        type=Path,
        default=None,
        help="Explicit YAML config file; overrides --presets.",
    )
    p_run.add_argument(
        "--conda-env",
        type=str,
        default=None,
        metavar="ENV",
        help="Conda env where OpenFold3 is installed (e.g. 'openfold3').",
    )
    _add_parse_args(p_run, include_chain_args=True)

    # --- prepare-query subcommand ---
    p_prep = sub.add_parser(
        "prepare-query",
        help="Prepare OF3 query JSON for binder refolding (receptor as template).",
    )
    p_prep.add_argument(
        "--complex",
        type=Path,
        required=True,
        metavar="CIF",
        help="Complex CIF/PDB file (receptor + binder).",
    )
    p_prep.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        required=True,
        metavar="CHAIN",
        help="Chain ID of the receptor/target (fixed as template).",
    )
    p_prep.add_argument(
        "--binder-chain",
        type=str,
        required=True,
        metavar="CHAIN",
        help="Chain ID of the binder (refolded from sequence only).",
    )
    p_prep.add_argument(
        "--query-name",
        "-n",
        type=str,
        required=True,
        help="Prediction query name.",
    )
    p_prep.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for query JSON and template files.",
    )
    p_prep.add_argument(
        "--template-cif",
        type=Path,
        default=None,
        metavar="CIF",
        help="Pre-prepared receptor CIF (e.g., after MD relaxation). "
        "If omitted, receptor chain is extracted from --complex.",
    )
    _add_query_seeds_arg(p_prep)

    # --- refold subcommand ---
    p_refold = sub.add_parser(
        "refold",
        help="Run OF3 binder refolding: receptor fixed as template, binder predicted freely.",
    )
    p_refold.add_argument(
        "--complex",
        type=Path,
        required=True,
        metavar="CIF",
        help="Complex CIF/PDB file (receptor + binder).",
    )
    p_refold.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        required=True,
        metavar="CHAIN",
        help="Chain ID of the receptor/target (fixed as template).",
    )
    p_refold.add_argument(
        "--binder-chain",
        type=str,
        required=True,
        metavar="CHAIN",
        help="Chain ID of the binder (refolded from sequence only).",
    )
    p_refold.add_argument(
        "--query-name",
        "-n",
        type=str,
        required=True,
        help="Prediction query name.",
    )
    p_refold.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Top-level output directory (query/ and predictions/ written here).",
    )
    p_refold.add_argument(
        "--template-cif",
        type=Path,
        default=None,
        metavar="CIF",
        help="Pre-prepared receptor CIF (e.g., after MD relaxation).",
    )
    p_refold.add_argument("--ckpt", type=Path, default=None, help="Model checkpoint path.")
    p_refold.add_argument(
        "--num-samples", type=int, default=5, help="Number of diffusion samples (default: 5)."
    )
    p_refold.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Passed to OpenFold3 as --num_model_seeds (default: 1).",
    )
    p_refold.add_argument(
        "--no-msa-server", action="store_true", help="Disable ColabFold MSA server."
    )
    p_refold.add_argument(
        "--presets",
        nargs="+",
        default=list(_DEFAULT_MODEL_PRESETS),
        metavar="PRESET",
        help=f"Model configuration presets (default: {_PRESETS_TEXT}).",
    )
    p_refold.add_argument(
        "--runner-yaml", type=Path, default=None, help="Explicit YAML config; overrides --presets."
    )
    p_refold.add_argument(
        "--conda-env",
        type=str,
        default=None,
        metavar="ENV",
        help="Conda env where OpenFold3 is installed (e.g. 'openfold3').",
    )
    _add_query_seeds_arg(p_refold)
    _add_parse_args(p_refold, include_chain_args=False)

    # --- prepare-scoring-query subcommand ---
    p_prep_score = sub.add_parser(
        "prepare-scoring-query",
        help="Prepare OF3 query JSON to score an existing complex (both chains as templates).",
    )
    p_prep_score.add_argument(
        "--complex",
        type=Path,
        required=True,
        metavar="CIF",
        help="Complex CIF/PDB file (receptor + binder).",
    )
    p_prep_score.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        required=True,
        metavar="CHAIN",
        help="Chain ID of the receptor.",
    )
    p_prep_score.add_argument(
        "--binder-chain", type=str, required=True, metavar="CHAIN", help="Chain ID of the binder."
    )
    p_prep_score.add_argument(
        "--query-name", "-n", type=str, required=True, help="Prediction query name."
    )
    p_prep_score.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for query JSON and template files.",
    )
    p_prep_score.add_argument(
        "--template-cif",
        type=Path,
        default=None,
        metavar="CIF",
        help="Pre-prepared complex CIF (e.g., MD-relaxed). Both chains extracted from it.",
    )
    _add_query_seeds_arg(p_prep_score)

    # --- score subcommand ---
    p_score = sub.add_parser(
        "score",
        help="Run OF3 scoring of an existing complex (both chains as templates).",
    )
    p_score.add_argument(
        "--complex",
        type=Path,
        required=True,
        metavar="CIF",
        help="Complex CIF/PDB file (receptor + binder).",
    )
    p_score.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        required=True,
        metavar="CHAIN",
        help="Chain ID of the receptor.",
    )
    p_score.add_argument(
        "--binder-chain", type=str, required=True, metavar="CHAIN", help="Chain ID of the binder."
    )
    p_score.add_argument(
        "--query-name", "-n", type=str, required=True, help="Prediction query name."
    )
    p_score.add_argument(
        "--output-dir", "-o", type=Path, required=True, help="Top-level output directory."
    )
    p_score.add_argument(
        "--template-cif",
        type=Path,
        default=None,
        metavar="CIF",
        help="Pre-prepared complex CIF (e.g., MD-relaxed). Both chains extracted from it.",
    )
    p_score.add_argument("--ckpt", type=Path, default=None, help="Model checkpoint path.")
    p_score.add_argument(
        "--num-samples", type=int, default=5, help="Number of diffusion samples (default: 5)."
    )
    p_score.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Passed to OpenFold3 as --num_model_seeds (default: 1).",
    )
    p_score.add_argument(
        "--no-msa-server", action="store_true", help="Disable ColabFold MSA server."
    )
    p_score.add_argument(
        "--presets",
        nargs="+",
        default=list(_DEFAULT_MODEL_PRESETS),
        metavar="PRESET",
        help=f"Model configuration presets (default: {_PRESETS_TEXT}).",
    )
    p_score.add_argument(
        "--runner-yaml", type=Path, default=None, help="Explicit YAML config; overrides --presets."
    )
    p_score.add_argument(
        "--conda-env",
        type=str,
        default=None,
        metavar="ENV",
        help="Conda env where OpenFold3 is installed (e.g. 'openfold3').",
    )
    _add_query_seeds_arg(p_score)
    _add_parse_args(p_score, include_chain_args=False)

    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import _apply_log_redirect

    _apply_log_redirect(args.log_file)

    # --- prepare-scoring-query ---
    if args.command == "prepare-scoring-query":
        path = of.prepare_scoring_query(
            complex_structure_path=args.complex,
            receptor_chain=args.receptor_chain,
            binder_chain=args.binder_chain,
            query_name=args.query_name,
            output_dir=args.output_dir,
            template_cif_path=args.template_cif,
            seeds=args.seeds,
        )
        print(f"Scoring query JSON written to: {path}")
        return

    # --- score ---
    if args.command == "score":
        print(f"Running OF3 structure scoring: {args.complex}")
        print(f"  Receptor chain (template): {args.receptor_chain}")
        print(f"  Binder chain  (template): {args.binder_chain}")
        predictions_dir = of.run_openfold_scoring(
            complex_structure_path=args.complex,
            receptor_chain=args.receptor_chain,
            binder_chain=args.binder_chain,
            query_name=args.query_name,
            output_dir=args.output_dir,
            template_cif_path=args.template_cif,
            inference_ckpt_path=args.ckpt,
            num_diffusion_samples=args.num_samples,
            num_model_seeds=args.num_seeds,
            use_msa_server=not args.no_msa_server,
            model_presets=args.presets,
            runner_yaml=args.runner_yaml,
            conda_env=args.conda_env,
            seeds=args.seeds,
        )
        print(f"\nParsing scoring metrics from: {predictions_dir}")
        metrics = of.compute_openfold_metrics(
            output_dir=predictions_dir,
            query_name=args.query_name,
            seed=args.seed,
            sample=args.sample,
            include_matrices=args.include_matrices,
            reference_structure_path=args.reference,
            binder_chain=args.binder_chain,
            receptor_chain=args.receptor_chain,
        )
        _print_metrics(metrics, args.seed, args.sample)
        return

    # --- prepare-query ---
    if args.command == "prepare-query":
        path = of.prepare_refolding_query(
            complex_structure_path=args.complex,
            receptor_chain=args.receptor_chain,
            binder_chain=args.binder_chain,
            query_name=args.query_name,
            output_dir=args.output_dir,
            template_cif_path=args.template_cif,
            seeds=args.seeds,
        )
        print(f"Query JSON written to: {path}")
        return

    # --- refold ---
    if args.command == "refold":
        print(f"Running OF3 binder refolding: {args.complex}")
        print(f"  Receptor chain (template): {args.receptor_chain}")
        print(f"  Binder chain (free):       {args.binder_chain}")
        predictions_dir = of.run_openfold_refolding(
            complex_structure_path=args.complex,
            receptor_chain=args.receptor_chain,
            binder_chain=args.binder_chain,
            query_name=args.query_name,
            output_dir=args.output_dir,
            template_cif_path=args.template_cif,
            inference_ckpt_path=args.ckpt,
            num_diffusion_samples=args.num_samples,
            num_model_seeds=args.num_seeds,
            use_msa_server=not args.no_msa_server,
            model_presets=args.presets,
            runner_yaml=args.runner_yaml,
            conda_env=args.conda_env,
            seeds=args.seeds,
        )
        print(f"\nParsing refolding metrics from: {predictions_dir}")
        metrics = of.compute_openfold_metrics(
            output_dir=predictions_dir,
            query_name=args.query_name,
            seed=args.seed,
            sample=args.sample,
            include_matrices=args.include_matrices,
            reference_structure_path=args.reference,
            binder_chain=args.binder_chain,
            receptor_chain=args.receptor_chain,
        )
        _print_metrics(metrics, args.seed, args.sample)
        return

    # --- run ---
    if args.command == "run":
        print(f"Running OpenFold3 inference: {args.query_json}")
        print(f"  Presets: {args.presets}")
        of.run_openfold(
            query_json=args.query_json,
            output_dir=args.output_dir,
            inference_ckpt_path=args.ckpt,
            num_diffusion_samples=args.num_samples,
            num_model_seeds=args.num_seeds,
            use_msa_server=not args.no_msa_server,
            model_presets=args.presets,
            runner_yaml=args.runner_yaml,
            conda_env=args.conda_env,
        )

    # --- parse (and fallthrough from run) ---
    print(f"\nParsing OpenFold3 metrics for: {args.query_name}")
    metrics = of.compute_openfold_metrics(
        output_dir=args.output_dir,
        query_name=args.query_name,
        seed=args.seed,
        sample=args.sample,
        include_matrices=args.include_matrices,
        reference_structure_path=args.reference,
        binder_chain=args.binder_chain,
        receptor_chain=args.receptor_chain,
    )
    _print_metrics(metrics, args.seed, args.sample)
