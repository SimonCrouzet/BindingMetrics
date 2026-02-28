"""OpenFold3 structure prediction metrics.

Provides utilities to run OpenFold3 inference and extract confidence metrics
from its output files.

OpenFold3 is a fully open-source (Apache 2.0) biomolecular structure prediction
model based on AlphaFold3, developed by the OpenFold Consortium. It predicts
structures of proteins, RNA, DNA, and small-molecule complexes.

Output files per prediction (seed S, sample M):
  {output_dir}/{query_name}/seed_{S}/{prefix}_confidences_aggregated.json
      Scalar confidence metrics: avg_plddt, gpde, ptm, iptm, chain_ptm,
      chain_pair_iptm, bespoke_iptm, disorder, has_clash, sample_ranking_score
  {output_dir}/{query_name}/seed_{S}/{prefix}_confidences.json (or .npz)
      Per-atom arrays: plddt[n_atoms], pde[n_tokens, n_tokens],
      pae[n_tokens, n_tokens] (if PAE head enabled)
  {output_dir}/{query_name}/seed_{S}/{prefix}_model.cif
      3D structure (pLDDT in B-factor column)
  {output_dir}/{query_name}/seed_{S}/timing.json
      Runtime (excluding MSA computation)

References:
  Ahdritz et al. (2024) OpenFold3: An open-source, trainable implementation
  of AlphaFold3. GitHub: github.com/aqlaboratory/openfold-3

Usage (Python API):
    from binding_metrics import compute_openfold_metrics

    metrics = compute_openfold_metrics(
        output_dir="./openfold_out",
        query_name="my_complex",
        seed=1,
        sample=1,
    )
    print(metrics["avg_plddt"], metrics["ptm"], metrics["iptm"])

Usage (CLI):
    # Parse existing output:
    binding-metrics-openfold parse --output-dir ./openfold_out --query-name my_complex

    # Run inference then parse:
    binding-metrics-openfold run \\
        --query-json query.json \\
        --output-dir ./openfold_out \\
        --query-name my_complex
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Output file discovery
# ---------------------------------------------------------------------------


def _find_prediction_files(
    output_dir: Path,
    query_name: str,
    seed: int = 1,
    sample: int = 1,
) -> dict[str, Optional[Path]]:
    """Locate OpenFold3 output files for a given seed/sample.

    Args:
        output_dir: Top-level OpenFold3 output directory.
        query_name: Query name as specified in the input JSON.
        seed: Seed index (default 1).
        sample: Sample index (default 1).

    Returns:
        Dict with keys: ``structure``, ``confidences``, ``confidences_aggregated``,
        ``timing``. Values are resolved Paths or None if not found.
    """
    seed_dir = output_dir / query_name / f"seed_{seed}"
    prefix = f"{query_name}_seed_{seed}_sample_{sample}"

    structure = None
    for ext in (".cif", ".pdb"):
        p = seed_dir / f"{prefix}_model{ext}"
        if p.exists():
            structure = p
            break

    confidences = None
    for ext in (".json", ".npz"):
        p = seed_dir / f"{prefix}_confidences{ext}"
        if p.exists():
            confidences = p
            break

    agg = seed_dir / f"{prefix}_confidences_aggregated.json"
    timing = seed_dir / "timing.json"

    return {
        "structure": structure,
        "confidences": confidences,
        "confidences_aggregated": agg if agg.exists() else None,
        "timing": timing if timing.exists() else None,
    }


# ---------------------------------------------------------------------------
# Confidence file parsers
# ---------------------------------------------------------------------------


def _parse_confidences_aggregated(path: Path) -> dict:
    """Parse ``*_confidences_aggregated.json``.

    Contains scalar metrics computed over the full complex.

    Args:
        path: Path to the aggregated confidence JSON file.

    Returns:
        Dict with all scalar confidence metrics. Missing keys are NaN.
        Keys: avg_plddt, gpde, ptm, iptm, disorder, has_clash,
        sample_ranking_score, chain_ptm (dict), chain_pair_iptm (dict),
        bespoke_iptm (dict).
    """
    with open(path) as fh:
        raw = json.load(fh)

    def _f(key):
        val = raw.get(key)
        return float(val) if val is not None else float("nan")

    return {
        "avg_plddt": _f("avg_plddt"),
        "gpde": _f("gpde"),
        "ptm": _f("ptm"),
        "iptm": _f("iptm"),
        "disorder": _f("disorder"),
        "has_clash": _f("has_clash"),
        "sample_ranking_score": _f("sample_ranking_score"),
        "chain_ptm": raw.get("chain_ptm", {}),
        "chain_pair_iptm": raw.get("chain_pair_iptm", {}),
        "bespoke_iptm": raw.get("bespoke_iptm", {}),
    }


def _parse_confidences(path: Path) -> dict:
    """Parse ``*_confidences.json`` or ``*_confidences.npz``.

    Contains per-atom pLDDT, per-token PDE/PAE matrices, and scalar
    aggregates. The pLDDT array has one value per heavy atom.

    Args:
        path: Path to the per-atom confidence file (.json or .npz).

    Returns:
        Dict with keys:
            plddt_per_atom (np.ndarray): per-atom pLDDT [0–100], shape (n_atoms,)
            pde (np.ndarray | None): predicted distance error matrix (n_tokens, n_tokens)
            pae (np.ndarray | None): predicted aligned error matrix (n_tokens, n_tokens)
            gpde (float): global PDE scalar
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        raw = {k: data[k] for k in data.files}
    else:
        with open(path) as fh:
            raw = json.load(fh)

    def _arr(key):
        val = raw.get(key)
        if val is None:
            return None
        return np.array(val, dtype=float)

    def _scalar(key):
        val = raw.get(key)
        if val is None:
            return float("nan")
        arr = np.asarray(val, dtype=float)
        return float(arr.ravel()[0]) if arr.size > 0 else float("nan")

    return {
        "plddt_per_atom": _arr("plddt"),
        "pde": _arr("pde"),
        "pae": _arr("pae"),
        "gpde": _scalar("gpde"),
    }


def _parse_timing(path: Path) -> dict:
    """Parse ``timing.json``."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_openfold_metrics(
    output_dir: str | Path,
    query_name: str,
    seed: int = 1,
    sample: int = 1,
    include_matrices: bool = False,
) -> dict:
    """Extract confidence metrics from OpenFold3 output files.

    Parses the aggregated confidence JSON (scalar metrics) and the per-atom
    confidence file (pLDDT array, PDE and PAE matrices) for the specified
    seed/sample prediction.

    Args:
        output_dir: Top-level OpenFold3 output directory.
        query_name: Query name as specified in the input JSON (used to locate
            the ``{output_dir}/{query_name}/`` subdirectory).
        seed: Seed index to parse (default 1).
        sample: Sample index to parse (default 1).
        include_matrices: If True, include the full PDE and PAE matrices in
            the result (can be large). Default False.

    Returns:
        Dictionary with keys:

        Structure:
            structure_path (str | None): path to the predicted .cif/.pdb file
            query_name (str), seed (int), sample (int)

        Scalar confidence metrics [from confidences_aggregated.json]:
            avg_plddt (float): mean pLDDT across all atoms [0–100]
            gpde (float): global predicted distance error (Å)
            ptm (float): predicted TM-score [0–1]; NaN if PAE head disabled
            iptm (float): interface pTM [0–1]; NaN if single chain or PAE disabled
            disorder (float): average relative SASA [0–1]
            has_clash (float): 1.0 if steric clashes detected, 0.0 otherwise
            sample_ranking_score (float): weighted composite score for ranking
            chain_ptm (dict): per-chain pTM scores {chain_id: float}
            chain_pair_iptm (dict): pairwise interface pTM {(A,B): float}
            bespoke_iptm (dict): bespoke interface score {(A,B): float}

        Per-atom / per-token data [from confidences.json]:
            plddt_per_atom (np.ndarray | None): per-atom pLDDT, shape (n_atoms,)
            n_atoms (int): number of atoms
            pde (np.ndarray | None): PDE matrix (n_tokens×n_tokens); only if
                include_matrices=True
            pae (np.ndarray | None): PAE matrix (n_tokens×n_tokens); only if
                include_matrices=True
            max_pae (float): max PAE value (Å); NaN if PAE unavailable

        Timing:
            timing (dict): runtime entries from timing.json, empty if absent
    """
    output_dir = Path(output_dir)
    files = _find_prediction_files(output_dir, query_name, seed=seed, sample=sample)

    result: dict = {
        "query_name": query_name,
        "seed": seed,
        "sample": sample,
        "structure_path": str(files["structure"]) if files["structure"] else None,
        # Scalar confidence metrics (NaN = not available)
        "avg_plddt": float("nan"),
        "gpde": float("nan"),
        "ptm": float("nan"),
        "iptm": float("nan"),
        "disorder": float("nan"),
        "has_clash": float("nan"),
        "sample_ranking_score": float("nan"),
        "chain_ptm": {},
        "chain_pair_iptm": {},
        "bespoke_iptm": {},
        # Per-atom data
        "plddt_per_atom": None,
        "n_atoms": 0,
        "pde": None,
        "pae": None,
        "max_pae": float("nan"),
        # Timing
        "timing": {},
    }

    # --- Aggregated confidence (scalar metrics) ---
    if files["confidences_aggregated"] is not None:
        agg = _parse_confidences_aggregated(files["confidences_aggregated"])
        result.update(agg)

    # --- Per-atom confidence file ---
    if files["confidences"] is not None:
        conf = _parse_confidences(files["confidences"])
        plddt_arr = conf["plddt_per_atom"]
        result["plddt_per_atom"] = plddt_arr
        result["n_atoms"] = int(len(plddt_arr)) if plddt_arr is not None else 0

        # avg_plddt from per-atom data as fallback
        if plddt_arr is not None and np.isnan(result["avg_plddt"]):
            result["avg_plddt"] = float(np.mean(plddt_arr))

        if include_matrices:
            result["pde"] = conf["pde"]
            result["pae"] = conf["pae"]

        # max PAE regardless of include_matrices flag
        pae = conf["pae"]
        if pae is not None:
            result["max_pae"] = float(pae.max())

    # --- Timing ---
    if files["timing"] is not None:
        result["timing"] = _parse_timing(files["timing"])

    return result


# ---------------------------------------------------------------------------
# Runner: invoke OpenFold3 as a subprocess
# ---------------------------------------------------------------------------


def _write_runner_yaml(output_dir: Path, presets: list[str]) -> Path:
    """Write a minimal runner YAML with model presets.

    Args:
        output_dir: Directory in which to write the file.
        presets: List of model preset names, e.g.
            ``["predict", "pae_enabled", "low_mem"]``.

    Returns:
        Path to the written YAML file.
    """
    try:
        import yaml
        content = yaml.dump({"model_update": {"presets": presets}}, default_flow_style=False)
    except ImportError:
        # Fallback: write YAML manually (presets are simple strings)
        lines = ["model_update:\n", "  presets:\n"]
        lines += [f"    - {p}\n" for p in presets]
        content = "".join(lines)

    yaml_path = output_dir / "runner_config.yaml"
    yaml_path.write_text(content)
    return yaml_path


def run_openfold(
    query_json: str | Path,
    output_dir: str | Path,
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
) -> Path:
    """Run OpenFold3 inference as a subprocess.

    Invokes ``run_openfold predict`` from the OpenFold3 package.
    OpenFold3 must be installed (``pip install openfold3`` and ``setup_openfold``).

    Args:
        query_json: Path to the input JSON file describing the prediction query.
            See https://openfold-3.readthedocs.io/en/latest/input_format.html
        output_dir: Directory where OpenFold3 writes predictions.
        inference_ckpt_path: Optional path to a model checkpoint (.pt file).
            Uses the default downloaded checkpoint if None.
        num_diffusion_samples: Number of structure samples per query (default 5).
        num_model_seeds: Number of random seeds per query (default 1).
        use_msa_server: Use the ColabFold MSA server for alignment generation
            (default True). Set False if MSAs are pre-computed.
        model_presets: List of model configuration presets. These are written to
            a runner YAML and passed via ``--runner_yaml``. The ``"predict"``
            preset is always prepended if not already present. Available presets:

            - ``"predict"`` — required base preset for inference
            - ``"pae_enabled"`` — enable the PAE head (required for pTM, ipTM,
              disorder, chain scores; required by official OpenFold3 weights)
            - ``"low_mem"`` — memory-efficient mode; pairformer embeddings are
              computed sequentially. Recommended for large complexes or limited
              GPU memory. Significant slowdown with many diffusion samples.

            Defaults to ``["predict", "pae_enabled", "low_mem"]``.
            Ignored if ``runner_yaml`` is also provided.
        runner_yaml: Explicit path to a runner YAML configuration file. Overrides
            ``model_presets`` when both are provided. CLI flags always take
            precedence over YAML values.
        extra_args: Additional CLI arguments passed verbatim.

    Returns:
        Path to the output directory.

    Raises:
        FileNotFoundError: If ``run_openfold`` is not on PATH.
        subprocess.CalledProcessError: If OpenFold3 exits non-zero.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("run_openfold") is None:
        raise FileNotFoundError(
            "run_openfold not found on PATH. "
            "Install OpenFold3 with: pip install openfold3 && setup_openfold"
        )

    # Resolve runner YAML: explicit path takes precedence over model_presets
    effective_yaml: Optional[Path] = None
    if runner_yaml is not None:
        effective_yaml = Path(runner_yaml)
    else:
        presets = list(model_presets) if model_presets is not None else ["predict", "pae_enabled", "low_mem"]
        if "predict" not in presets:
            presets.insert(0, "predict")
        effective_yaml = _write_runner_yaml(output_dir, presets)

    cmd = [
        "run_openfold", "predict",
        f"--query_json={query_json}",
        f"--output_dir={output_dir}",
        f"--num_diffusion_samples={num_diffusion_samples}",
        f"--num_model_seeds={num_model_seeds}",
        f"--use_msa_server={str(use_msa_server).lower()}",
        f"--runner_yaml={effective_yaml}",
    ]

    if inference_ckpt_path is not None:
        cmd.append(f"--inference_ckpt_path={inference_ckpt_path}")

    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True)
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract OpenFold3 confidence metrics (pLDDT, gPDE, pTM, ipTM, PAE) "
            "from prediction output files, or run OpenFold3 inference."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- parse subcommand ---
    p_parse = sub.add_parser(
        "parse",
        help="Parse metrics from an existing OpenFold3 output directory.",
    )
    p_parse.add_argument(
        "--output-dir", "-o", type=Path, required=True,
        help="OpenFold3 output directory.",
    )
    p_parse.add_argument(
        "--query-name", "-n", type=str, required=True,
        help="Query name (as specified in the input JSON).",
    )
    p_parse.add_argument(
        "--seed", type=int, default=1,
        help="Seed index to parse (default: 1).",
    )
    p_parse.add_argument(
        "--sample", type=int, default=1,
        help="Sample index to parse (default: 1).",
    )
    p_parse.add_argument(
        "--include-matrices", action="store_true",
        help="Include full PDE/PAE matrices in output (large).",
    )

    # --- run subcommand ---
    p_run = sub.add_parser(
        "run",
        help="Run OpenFold3 inference, then parse and print metrics.",
    )
    p_run.add_argument(
        "--query-json", type=Path, required=True,
        help="Input JSON file describing the prediction query.",
    )
    p_run.add_argument(
        "--output-dir", "-o", type=Path, required=True,
        help="Output directory.",
    )
    p_run.add_argument(
        "--query-name", "-n", type=str, required=True,
        help="Query name to parse after inference.",
    )
    p_run.add_argument(
        "--ckpt", type=Path, default=None,
        help="Model checkpoint path (uses default if omitted).",
    )
    p_run.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of diffusion samples (default: 5).",
    )
    p_run.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of random seeds (default: 1).",
    )
    p_run.add_argument(
        "--no-msa-server", action="store_true",
        help="Disable ColabFold MSA server (use pre-computed MSAs).",
    )
    p_run.add_argument(
        "--presets", nargs="+",
        default=["predict", "pae_enabled", "low_mem"],
        metavar="PRESET",
        help=(
            "Model configuration presets (default: predict pae_enabled low_mem). "
            "Available: predict, pae_enabled, low_mem."
        ),
    )
    p_run.add_argument(
        "--runner-yaml", type=Path, default=None,
        help="Explicit YAML config file; overrides --presets.",
    )
    p_run.add_argument(
        "--seed", type=int, default=1,
        help="Seed index to parse after inference (default: 1).",
    )
    p_run.add_argument(
        "--sample", type=int, default=1,
        help="Sample index to parse after inference (default: 1).",
    )

    args = parser.parse_args()

    if args.command == "run":
        print(f"Running OpenFold3 inference: {args.query_json}")
        print(f"  Presets: {args.presets}")
        run_openfold(
            query_json=args.query_json,
            output_dir=args.output_dir,
            inference_ckpt_path=args.ckpt,
            num_diffusion_samples=args.num_samples,
            num_model_seeds=args.num_seeds,
            use_msa_server=not args.no_msa_server,
            model_presets=args.presets,
            runner_yaml=args.runner_yaml,
        )

    print(f"\nParsing OpenFold3 metrics for: {args.query_name}")
    metrics = compute_openfold_metrics(
        output_dir=args.output_dir,
        query_name=args.query_name,
        seed=args.seed,
        sample=args.sample,
    )

    # --- Print results ---
    print(f"\nOpenFold3 confidence metrics (seed={args.seed}, sample={args.sample}):")
    print(f"  Structure:            {metrics['structure_path'] or 'not found'}")
    print(f"  Atoms:                {metrics['n_atoms']}")

    def _fmt(label, val, unit=""):
        if np.isnan(val):
            print(f"  {label:<22} N/A")
        else:
            print(f"  {label:<22} {val:.4f}{unit}")

    _fmt("avg_pLDDT [0–100]:", metrics["avg_plddt"])
    _fmt("gPDE (Å):", metrics["gpde"])
    _fmt("pTM [0–1]:", metrics["ptm"])
    _fmt("ipTM [0–1]:", metrics["iptm"])
    _fmt("Disorder:", metrics["disorder"])
    _fmt("has_clash:", metrics["has_clash"])
    _fmt("Ranking score:", metrics["sample_ranking_score"])
    _fmt("Max PAE (Å):", metrics["max_pae"])

    if metrics["chain_ptm"]:
        print(f"\n  Per-chain pTM:")
        for chain, val in metrics["chain_ptm"].items():
            print(f"    chain {chain}: {float(val):.4f}")

    if metrics["chain_pair_iptm"]:
        print(f"\n  Chain-pair ipTM:")
        for pair, val in metrics["chain_pair_iptm"].items():
            print(f"    {pair}: {float(val):.4f}")

    if metrics["timing"]:
        print("\nTiming:")
        for k, v in metrics["timing"].items():
            print(f"  {k}: {v:.2f}s" if isinstance(v, (int, float)) else f"  {k}: {v}")

    if metrics["plddt_per_atom"] is not None:
        arr = metrics["plddt_per_atom"]
        print(f"\nPer-atom pLDDT summary:")
        print(f"  Min:    {arr.min():.1f}")
        print(f"  Median: {np.median(arr):.1f}")
        print(f"  Max:    {arr.max():.1f}")
        print(f"  Very high (≥90): {int((arr >= 90).sum())} atoms")
        print(f"  High    (70–90): {int(((arr >= 70) & (arr < 90)).sum())} atoms")
        print(f"  Low     (50–70): {int(((arr >= 50) & (arr < 70)).sum())} atoms")
        print(f"  Very low  (<50): {int((arr < 50).sum())} atoms")


if __name__ == "__main__":
    main()
