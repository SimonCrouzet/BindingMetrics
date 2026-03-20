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
import warnings
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
# Structural analysis helpers (per-chain pLDDT, PAE slice, RMSD)
# ---------------------------------------------------------------------------


def _import_biotite_struc():
    """Lazy import of biotite structure modules."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        return struc, pdbx
    except ImportError:
        raise ImportError(
            "biotite is required for per-chain structural analysis. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _load_atoms(path: Path):
    """Load an AtomArray from a CIF or PDB file using biotite (model 1)."""
    _, pdbx = _import_biotite_struc()
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(f, model=1)
    import biotite.structure.io.pdb as pdb_io
    f = pdb_io.PDBFile.read(str(path))
    return pdb_io.get_structure(f, model=1)


def _chain_token_offsets(atoms) -> dict[str, tuple[int, int]]:
    """Map chain IDs to [start, end) PAE token ranges (one token per residue).

    Preserves the order chains first appear in the structure, which matches
    the PAE matrix token ordering (same order as the query JSON chains).

    Returns:
        Dict ``{chain_id: (start, end)}`` where ``end = start + n_residues``.
    """
    seen: list[str] = []
    for cid in atoms.chain_id:
        if cid not in seen:
            seen.append(cid)
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for chain_id in seen:
        n_res = int(np.unique(atoms.res_id[atoms.chain_id == chain_id]).size)
        offsets[chain_id] = (offset, offset + n_res)
        offset += n_res
    return offsets


def _binder_plddt_per_residue(
    plddt_per_atom: np.ndarray,
    atoms,
    binder_chain: str,
) -> np.ndarray:
    """Mean pLDDT per residue for one chain.

    Args:
        plddt_per_atom: Per-atom pLDDT array from OpenFold3, shape (n_atoms,).
            Must be in the same atom order as ``atoms``.
        atoms: Biotite AtomArray from the predicted model CIF.
        binder_chain: Chain ID to extract.

    Returns:
        Array of shape ``(n_residues,)`` with mean pLDDT per residue [0–100].

    Raises:
        ValueError: If ``len(plddt_per_atom) != atoms.array_length()``.
    """
    if len(plddt_per_atom) != atoms.array_length():
        raise ValueError(
            f"plddt_per_atom length ({len(plddt_per_atom)}) != "
            f"atom count in structure ({atoms.array_length()}). "
            "The pLDDT array and structure file must be from the same prediction."
        )
    mask = atoms.chain_id == binder_chain
    chain_plddt = plddt_per_atom[mask]
    chain_res_ids = atoms.res_id[mask]
    if chain_res_ids.size == 0:
        return np.array([], dtype=float)
    unique_res = np.unique(chain_res_ids)
    return np.array(
        [chain_plddt[chain_res_ids == r].mean() for r in unique_res], dtype=float
    )


def _binder_ca_rmsd(
    pred_atoms,
    ref_atoms,
    binder_chain: str,
    receptor_chain: Optional[str] = None,
) -> float:
    """Binder Cα RMSD (Å) between predicted and reference structures.

    If ``receptor_chain`` is supplied, the predicted structure is first
    superposed onto the reference receptor Cα atoms before measuring the
    binder RMSD. This gives the physically meaningful "receptor-frame" RMSD:
    how much the predicted binder deviates from the reference binder pose
    relative to the receptor.

    Args:
        pred_atoms: Biotite AtomArray of the predicted structure.
        ref_atoms: Biotite AtomArray of the reference structure.
        binder_chain: Chain ID of the binder.
        receptor_chain: Chain ID of the receptor (used for superposition).
            If None, superpose directly on binder Cα.

    Returns:
        Binder Cα RMSD in Å, or ``nan`` if there are no matching Cα atoms.

    Raises:
        ValueError: If binder Cα counts differ between prediction and reference.
    """
    struc, _ = _import_biotite_struc()

    def _ca(atoms, chain):
        return atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]

    pred_binder_ca = _ca(pred_atoms, binder_chain)
    ref_binder_ca = _ca(ref_atoms, binder_chain)

    n_pred = pred_binder_ca.array_length()
    n_ref = ref_binder_ca.array_length()
    if n_pred != n_ref:
        raise ValueError(
            f"Binder Cα count mismatch (chain '{binder_chain}'): "
            f"predicted {n_pred}, reference {n_ref}. "
            "Structures may have different sequence lengths."
        )
    if n_pred == 0:
        return float("nan")

    if receptor_chain is not None:
        pred_rec_ca = _ca(pred_atoms, receptor_chain)
        ref_rec_ca = _ca(ref_atoms, receptor_chain)
        if (
            pred_rec_ca.array_length() == ref_rec_ca.array_length()
            and pred_rec_ca.array_length() >= 3
        ):
            _, transform = struc.superimpose(ref_rec_ca, pred_rec_ca)
            pred_binder_ca = struc.superimpose_apply(pred_binder_ca, transform)

    return float(struc.rmsd(ref_binder_ca, pred_binder_ca))


def _interface_pae_stats(
    pae: np.ndarray,
    atoms,
    binder_chain: str,
    receptor_chain: str,
) -> dict:
    """PAE statistics for the binder–receptor interface region.

    Slices the full PAE matrix to the binder×receptor token sub-matrix and
    returns summary statistics and (optionally) the raw slice.

    Args:
        pae: Full PAE matrix, shape ``(n_tokens, n_tokens)``.
        atoms: Biotite AtomArray of the predicted structure.
        binder_chain: Chain ID of the binder.
        receptor_chain: Chain ID of the receptor.

    Returns:
        Dict with:
            ``pae_interface`` (np.ndarray): Sub-matrix ``(n_binder_res, n_receptor_res)``.
            ``mean_interface_pae`` (float): Mean PAE over the interface slice (Å).
            ``max_interface_pae`` (float): Max PAE over the interface slice (Å).
            ``n_binder_tokens`` (int): Binder residue token count.
            ``n_receptor_tokens`` (int): Receptor residue token count.
    """
    offsets = _chain_token_offsets(atoms)
    missing = [c for c in (binder_chain, receptor_chain) if c not in offsets]
    if missing:
        raise ValueError(f"Chains not found in structure: {missing}")
    b0, b1 = offsets[binder_chain]
    r0, r1 = offsets[receptor_chain]
    sub = pae[b0:b1, r0:r1]
    return {
        "pae_interface": sub,
        "mean_interface_pae": float(sub.mean()),
        "max_interface_pae": float(sub.max()),
        "n_binder_tokens": b1 - b0,
        "n_receptor_tokens": r1 - r0,
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_openfold_metrics(
    output_dir: str | Path,
    query_name: str,
    seed: int = 1,
    sample: int = 1,
    include_matrices: bool = False,
    reference_structure_path: Optional[str | Path] = None,
    binder_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
) -> dict:
    """Extract confidence metrics from OpenFold3 output files.

    **Mode 1 — confidence metrics only (no reference):**
    Parse the aggregated and per-atom confidence files to get scalar metrics
    (avg_plddt, ptm, iptm, chain scores) and optionally per-residue binder
    pLDDT and interface PAE statistics. Pass ``binder_chain`` (and optionally
    ``receptor_chain``) to enable per-chain analysis.

    **Mode 1 with reference — refolding RMSD:**
    Pass ``reference_structure_path`` together with ``binder_chain`` (and
    ``receptor_chain`` for receptor-frame alignment) to additionally compute
    the binder Cα RMSD between the OF3 prediction and a known reference
    structure (e.g., crystal or MD-relaxed). This measures how faithfully OF3
    recovers the bound binder conformation.

    Args:
        output_dir: Top-level OpenFold3 output directory.
        query_name: Query name as specified in the input JSON (used to locate
            the ``{output_dir}/{query_name}/`` subdirectory).
        seed: Seed index to parse (default 1).
        sample: Sample index to parse (default 1).
        include_matrices: If True, include the full PDE and PAE matrices in
            the result (can be large). Default False.
        reference_structure_path: Optional path to a reference CIF/PDB
            (e.g., crystal structure). When supplied together with
            ``binder_chain``, the binder Cα RMSD between the OF3 prediction
            and this reference is computed and stored as ``binder_ca_rmsd``.
        binder_chain: Chain ID of the binder/ligand in the predicted structure.
            Enables per-residue pLDDT for the binder, interface PAE stats
            (when ``receptor_chain`` is also given), and binder RMSD
            (when ``reference_structure_path`` is also given).
        receptor_chain: Chain ID of the receptor/target. When provided
            alongside ``binder_chain``:
              - Interface PAE statistics are computed (mean/max PAE for the
                binder×receptor token block).
              - Receptor Cα atoms are used as the superposition reference
                when computing ``binder_ca_rmsd``, giving the
                receptor-frame binder RMSD.

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

        Per-chain structural analysis [requires binder_chain]:
            binder_plddt_per_residue (np.ndarray | None): mean pLDDT per
                residue for the binder chain, shape (n_binder_res,)
            binder_avg_plddt (float): mean pLDDT over all binder residues

        Interface PAE [requires binder_chain + receptor_chain + PAE available]:
            mean_interface_pae (float): mean PAE over binder×receptor tokens (Å)
            max_interface_pae (float): max PAE over binder×receptor tokens (Å)
            pae_interface (np.ndarray | None): raw PAE slice, shape
                (n_binder_res, n_receptor_res); only if include_matrices=True

        Refolding RMSD [requires binder_chain + reference_structure_path]:
            binder_ca_rmsd (float): binder Cα RMSD vs. reference (Å).
                Computed in the receptor frame if receptor_chain is given
                (predicted structure superposed on receptor Cα first).

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
        # Per-chain structural analysis (populated when binder_chain is given)
        "binder_plddt_per_residue": None,
        "binder_avg_plddt": float("nan"),
        # Interface PAE (populated when binder_chain + receptor_chain are given)
        "mean_interface_pae": float("nan"),
        "max_interface_pae": float("nan"),
        "pae_interface": None,
        # Refolding RMSD (populated when binder_chain + reference_structure_path)
        "binder_ca_rmsd": float("nan"),
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

    # --- Per-chain structural analysis ---
    # Requires binder_chain; uses the predicted model CIF.
    if binder_chain is not None and files["structure"] is not None:
        try:
            pred_atoms = _load_atoms(files["structure"])

            # Per-residue binder pLDDT
            if result["plddt_per_atom"] is not None:
                try:
                    per_res = _binder_plddt_per_residue(
                        result["plddt_per_atom"], pred_atoms, binder_chain
                    )
                    result["binder_plddt_per_residue"] = per_res
                    result["binder_avg_plddt"] = (
                        float(per_res.mean()) if per_res.size > 0 else float("nan")
                    )
                except Exception as exc:
                    warnings.warn(f"compute_openfold_metrics: per-residue binder pLDDT skipped: {exc}")

            # Interface PAE statistics (binder × receptor token block)
            if receptor_chain is not None:
                pae_src = result.get("pae")
                if pae_src is None and files["confidences"] is not None:
                    # Load PAE even when include_matrices=False for stats only
                    pae_src = _parse_confidences(files["confidences"]).get("pae")
                if pae_src is not None:
                    try:
                        pae_stats = _interface_pae_stats(
                            pae_src, pred_atoms, binder_chain, receptor_chain
                        )
                        result["mean_interface_pae"] = pae_stats["mean_interface_pae"]
                        result["max_interface_pae"] = pae_stats["max_interface_pae"]
                        if include_matrices:
                            result["pae_interface"] = pae_stats["pae_interface"]
                    except Exception as exc:
                        warnings.warn(f"compute_openfold_metrics: interface PAE skipped: {exc}")

            # Binder Cα RMSD vs. reference structure
            if reference_structure_path is not None:
                try:
                    ref_atoms = _load_atoms(Path(reference_structure_path))
                    result["binder_ca_rmsd"] = _binder_ca_rmsd(
                        pred_atoms, ref_atoms, binder_chain, receptor_chain
                    )
                except Exception as exc:
                    warnings.warn(f"compute_openfold_metrics: binder RMSD skipped: {exc}")

        except Exception as exc:
            warnings.warn(f"compute_openfold_metrics: structural analysis failed: {exc}")

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
# Mode 2: binder refolding with receptor fixed as template
# ---------------------------------------------------------------------------


def _extract_sequence_from_structure(structure, chain_id: str) -> str:
    """Extract one-letter amino acid sequence for a chain using gemmi.

    Skips non-amino-acid residues (waters, ligands, etc.).

    Args:
        structure: A ``gemmi.Structure`` object.
        chain_id: Chain ID to extract.

    Returns:
        One-letter sequence string (non-standard residues become 'X').

    Raises:
        ValueError: If the chain is not found or contains no amino acids.
    """
    import gemmi

    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            seq = []
            for res in chain:
                tbl = gemmi.find_tabulated_residue(res.name)
                if tbl is None or not tbl.is_amino_acid():
                    continue
                seq.append(tbl.one_letter_code or "X")
            if not seq:
                raise ValueError(
                    f"Chain '{chain_id}' found but contains no amino acid residues"
                )
            return "".join(seq)
    raise ValueError(f"Chain '{chain_id}' not found in structure")


def _extract_chain_to_cif(structure, chain_id: str, output_path: Path) -> None:
    """Write a single chain from a gemmi Structure to a CIF file.

    Args:
        structure: Source ``gemmi.Structure``.
        chain_id: Chain ID to extract (taken from the first model).
        output_path: Destination CIF file path.
    """
    import gemmi

    new_st = gemmi.Structure()
    new_st.cell = structure.cell
    new_st.spacegroup_hm = structure.spacegroup_hm
    new_model = gemmi.Model("1")
    for chain in structure[0]:
        if chain.name == chain_id:
            new_model.add_chain(chain.clone())
            break
    new_st.add_model(new_model)
    new_st.make_mmcif_document().write_file(str(output_path))


def _write_a3m_self_alignment(
    sequence: str,
    query_id: str,
    template_id: str,
    output_path: Path,
) -> None:
    """Write a minimal A3M alignment for a self-template (100% identity).

    The A3M file contains two sequences: the query and the template
    (identical sequences, meaning the template perfectly matches the query).
    The template header uses the ColabFold/OF3 convention::

        >{template_id} /1-{N}

    where the template ID is what OF3 uses to look up the structure CIF in
    the template database directory (``--template_mmcif_dir``).

    Args:
        sequence: One-letter amino acid sequence.
        query_id: Identifier for the query sequence.
        template_id: Entry ID for the template. OF3 looks for
            ``{template_id}.cif`` in the template directory.
        output_path: Destination A3M file path.
    """
    n = len(sequence)
    output_path.write_text(
        f">{query_id}\n{sequence}\n"
        f">{template_id} /1-{n}\n{sequence}\n"
    )


def prepare_refolding_query(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
) -> Path:
    """Prepare an OpenFold3 query JSON for binder refolding with receptor as template.

    **Mode 2 — target-fixed refolding:**
    The receptor chain is provided as a structural template so that OF3 is
    conditioned on the known receptor geometry. The binder chain is predicted
    from sequence only (no template). This answers: "given the known receptor,
    can OF3 recover the bound binder conformation?"

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json            — OF3 input JSON
          {query_name}_receptor_{chain}.a3m  — self-alignment for receptor
          templates/
            {query_name}_receptor_{chain}.cif — receptor template structure

    The template CIF must be discoverable by OF3 at inference time. Pass::

        --template_mmcif_dir {output_dir}/templates

    to ``run_openfold`` (via ``extra_args``) if OF3 does not automatically
    locate templates relative to the alignment file.

    .. note::
        OF3's template pipeline currently supports **monomeric templates**
        (protein chains only). The extracted receptor CIF is a single-chain
        structure; multi-chain receptors should be merged into one chain
        before calling this function, or handled with separate templates per
        chain.

    Args:
        complex_structure_path: CIF or PDB file of the full complex.
        receptor_chain: Chain ID of the receptor/target to fix as template.
        binder_chain: Chain ID of the binder to refold (no template).
        query_name: Name for the prediction query (used in file names and the
            OF3 ``name`` field).
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional path to a pre-prepared receptor CIF
            (e.g., after MD relaxation). Must be a monomer (one chain).
            If None, the receptor chain is extracted from
            ``complex_structure_path``.

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If a specified chain is not found or has no amino acids.
    """
    import gemmi

    complex_structure_path = Path(complex_structure_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Extract sequences
    st = gemmi.read_structure(str(complex_structure_path))
    receptor_seq = _extract_sequence_from_structure(st, receptor_chain)
    binder_seq = _extract_sequence_from_structure(st, binder_chain)

    # Template CIF (receptor chain only — OF3 templates are monomeric)
    template_id = f"{query_name}_receptor_{receptor_chain}"
    template_dest = templates_dir / f"{template_id}.cif"
    if template_cif_path is not None:
        import shutil
        shutil.copy2(str(template_cif_path), str(template_dest))
    else:
        _extract_chain_to_cif(st, receptor_chain, template_dest)

    # A3M self-alignment for receptor template
    a3m_path = output_dir / f"{template_id}.a3m"
    _write_a3m_self_alignment(receptor_seq, f"query_{receptor_chain}", template_id, a3m_path)

    # Query JSON
    query = {
        "name": query_name,
        "sequences": [
            {
                "protein": {
                    "id": [receptor_chain],
                    "sequence": receptor_seq,
                    "template_alignment_file_path": str(a3m_path),
                }
            },
            {
                "protein": {
                    "id": [binder_chain],
                    "sequence": binder_seq,
                }
            },
        ],
    }
    query_json_path = output_dir / f"{query_name}_query.json"
    query_json_path.write_text(json.dumps(query, indent=2))
    return query_json_path


def prepare_scoring_query(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
) -> Path:
    """Prepare an OpenFold3 query JSON to score an existing complex structure.

    **Mode 1 — structure scoring:**
    Both receptor and binder chains are provided as structural templates so
    that OF3 evaluates the known conformation rather than predicting de-novo.
    OF3 outputs confidence scores (pLDDT, pTM, ipTM, etc.) reflecting how
    self-consistent it finds that specific structure.

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json
          {query_name}_receptor_{chain}.a3m
          {query_name}_binder_{chain}.a3m
          templates/
            {query_name}_receptor_{chain}.cif
            {query_name}_binder_{chain}.cif

    Args:
        complex_structure_path: CIF or PDB file of the full complex.
        receptor_chain: Chain ID of the receptor.
        binder_chain: Chain ID of the binder.
        query_name: Name for the prediction query.
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional pre-prepared complex or receptor CIF
            (e.g., after MD relaxation). When provided, both chain templates
            are extracted from this file instead of ``complex_structure_path``.

    Returns:
        Path to the written query JSON file.
    """
    import gemmi

    complex_structure_path = Path(complex_structure_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Source structure for sequences (always the original complex)
    st = gemmi.read_structure(str(complex_structure_path))
    receptor_seq = _extract_sequence_from_structure(st, receptor_chain)
    binder_seq = _extract_sequence_from_structure(st, binder_chain)

    # Template source: use template_cif_path if provided, else the complex
    template_src = (
        gemmi.read_structure(str(template_cif_path))
        if template_cif_path is not None
        else st
    )

    # Extract each chain as a monomer CIF template
    receptor_template_id = f"{query_name}_receptor_{receptor_chain}"
    binder_template_id = f"{query_name}_binder_{binder_chain}"

    _extract_chain_to_cif(template_src, receptor_chain, templates_dir / f"{receptor_template_id}.cif")
    _extract_chain_to_cif(template_src, binder_chain, templates_dir / f"{binder_template_id}.cif")

    # A3M self-alignments for both chains
    receptor_a3m = output_dir / f"{receptor_template_id}.a3m"
    binder_a3m = output_dir / f"{binder_template_id}.a3m"
    _write_a3m_self_alignment(receptor_seq, f"query_{receptor_chain}", receptor_template_id, receptor_a3m)
    _write_a3m_self_alignment(binder_seq, f"query_{binder_chain}", binder_template_id, binder_a3m)

    # Query JSON — both chains have template_alignment_file_path
    query = {
        "name": query_name,
        "sequences": [
            {
                "protein": {
                    "id": [receptor_chain],
                    "sequence": receptor_seq,
                    "template_alignment_file_path": str(receptor_a3m),
                }
            },
            {
                "protein": {
                    "id": [binder_chain],
                    "sequence": binder_seq,
                    "template_alignment_file_path": str(binder_a3m),
                }
            },
        ],
    }
    query_json_path = output_dir / f"{query_name}_query.json"
    query_json_path.write_text(json.dumps(query, indent=2))
    return query_json_path


def run_openfold_scoring(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
) -> Path:
    """Run OpenFold3 scoring of an existing complex structure (Mode 1).

    Both receptor and binder chains are provided as structural templates.
    OF3 scores the known conformation and outputs confidence metrics.
    Use :func:`compute_openfold_metrics` with ``binder_chain`` and
    ``receptor_chain`` to extract per-chain scores after inference.

    Args:
        complex_structure_path: CIF/PDB of the full complex.
        receptor_chain: Chain ID of the receptor.
        binder_chain: Chain ID of the binder.
        query_name: Prediction query name.
        output_dir: Top-level output directory.
        template_cif_path: Optional pre-prepared complex CIF (e.g., MD-relaxed).
        inference_ckpt_path: Optional model checkpoint path.
        num_diffusion_samples: Structure samples per query (default 5).
        num_model_seeds: Random seeds per query (default 1).
        use_msa_server: Use ColabFold MSA server (default True).
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3.

    Returns:
        Path to the OF3 predictions output directory
        (``{output_dir}/predictions/``).
    """
    output_dir = Path(output_dir)
    query_dir = output_dir / "query"
    predictions_dir = output_dir / "predictions"

    query_json = prepare_scoring_query(
        complex_structure_path=complex_structure_path,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        query_name=query_name,
        output_dir=query_dir,
        template_cif_path=template_cif_path,
    )

    effective_extra: list[str] = list(extra_args or [])
    template_dir_arg = f"--template_mmcif_dir={query_dir / 'templates'}"
    if not any("template_mmcif_dir" in a for a in effective_extra):
        effective_extra.append(template_dir_arg)

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=effective_extra,
    )
    return predictions_dir


def run_openfold_refolding(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
) -> Path:
    """Run OpenFold3 refolding: binder predicted freely, receptor fixed as template.

    Convenience wrapper for Mode 2. Calls :func:`prepare_refolding_query` to
    build the input JSON, then invokes :func:`run_openfold`. After inference,
    call :func:`compute_openfold_metrics` with ``binder_chain``,
    ``receptor_chain``, and ``reference_structure_path`` to get refolding
    RMSD and per-chain confidence metrics.

    Directory layout::

        {output_dir}/
          query/            — query JSON, A3M, and template CIF
          predictions/      — OF3 output (structures, confidence files)

    Args:
        complex_structure_path: CIF/PDB of the full complex.
        receptor_chain: Chain ID of the receptor (fixed as template).
        binder_chain: Chain ID of the binder (refolded from sequence only).
        query_name: Prediction query name.
        output_dir: Top-level output directory.
        template_cif_path: Optional pre-prepared receptor template CIF.
        inference_ckpt_path: Optional model checkpoint path.
        num_diffusion_samples: Structure samples per query (default 5).
        num_model_seeds: Random seeds per query (default 1).
        use_msa_server: Use ColabFold MSA server (default True).
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3. To use the template CIF,
            pass ``["--template_mmcif_dir=<path>"]`` if OF3 requires it.
            By default ``--template_mmcif_dir`` is automatically appended
            pointing to ``{output_dir}/query/templates/``.

    Returns:
        Path to the OF3 predictions output directory
        (``{output_dir}/predictions/``).
    """
    output_dir = Path(output_dir)
    query_dir = output_dir / "query"
    predictions_dir = output_dir / "predictions"

    query_json = prepare_refolding_query(
        complex_structure_path=complex_structure_path,
        receptor_chain=receptor_chain,
        binder_chain=binder_chain,
        query_name=query_name,
        output_dir=query_dir,
        template_cif_path=template_cif_path,
    )

    # Automatically include the local template directory
    effective_extra: list[str] = list(extra_args or [])
    template_dir_arg = f"--template_mmcif_dir={query_dir / 'templates'}"
    if not any("template_mmcif_dir" in a for a in effective_extra):
        effective_extra.append(template_dir_arg)

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=effective_extra,
    )
    return predictions_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_parse_args(p, include_chain_args: bool = False) -> None:
    """Add common parse/metrics arguments to a subparser."""
    p.add_argument("--seed", type=int, default=1, help="Seed index (default: 1).")
    p.add_argument("--sample", type=int, default=1, help="Sample index (default: 1).")
    p.add_argument(
        "--include-matrices", action="store_true",
        help="Include full PDE/PAE matrices in output (large).",
    )
    if include_chain_args:
        p.add_argument(
            "--binder-chain", type=str, default=None, metavar="CHAIN",
            help="Chain ID of the binder. Enables per-residue pLDDT and binder RMSD.",
        )
        p.add_argument(
            "--receptor-chain", type=str, default=None, metavar="CHAIN",
            help="Chain ID of the receptor. Enables interface PAE and receptor-frame RMSD.",
        )
    p.add_argument(
        "--reference", type=Path, default=None, metavar="CIF",
        help="Reference structure CIF/PDB for binder Cα RMSD (requires --binder-chain).",
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
    _fmt("Max PAE (Å):", metrics["max_pae"])

    if not np.isnan(metrics.get("binder_avg_plddt", float("nan"))):
        _fmt("Binder avg pLDDT:", metrics["binder_avg_plddt"])
    if not np.isnan(metrics.get("mean_interface_pae", float("nan"))):
        _fmt("Interface PAE mean (Å):", metrics["mean_interface_pae"])
        _fmt("Interface PAE max (Å):", metrics["max_interface_pae"])
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
        print(f"    ≥90: {int((arr >= 90).sum())}  70–89: {int(((arr >= 70) & (arr < 90)).sum())}"
              f"  50–69: {int(((arr >= 50) & (arr < 70)).sum())}  <50: {int((arr < 50).sum())}")

    if metrics.get("timing"):
        print("\nTiming:")
        for k, v in metrics["timing"].items():
            print(f"  {k}: {v:.2f}s" if isinstance(v, (int, float)) else f"  {k}: {v}")

    if metrics.get("plddt_per_atom") is not None:
        arr = metrics["plddt_per_atom"]
        print(f"\nPer-atom pLDDT summary ({len(arr)} atoms):")
        print(f"  Min: {arr.min():.1f}  Median: {np.median(arr):.1f}  Max: {arr.max():.1f}")
        print(f"  ≥90: {int((arr >= 90).sum())}  70–89: {int(((arr >= 70) & (arr < 90)).sum())}"
              f"  50–69: {int(((arr >= 50) & (arr < 70)).sum())}  <50: {int((arr < 50).sum())}")


def main():
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
        "--output-dir", "-o", type=Path, required=True,
        help="OpenFold3 output directory.",
    )
    p_parse.add_argument(
        "--query-name", "-n", type=str, required=True,
        help="Query name (as specified in the input JSON).",
    )
    _add_parse_args(p_parse, include_chain_args=True)

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
    p_run.add_argument("--ckpt", type=Path, default=None,
                       help="Model checkpoint path (uses default if omitted).")
    p_run.add_argument("--num-samples", type=int, default=5,
                       help="Number of diffusion samples (default: 5).")
    p_run.add_argument("--num-seeds", type=int, default=1,
                       help="Number of random seeds (default: 1).")
    p_run.add_argument("--no-msa-server", action="store_true",
                       help="Disable ColabFold MSA server (use pre-computed MSAs).")
    p_run.add_argument(
        "--presets", nargs="+", default=["predict", "pae_enabled", "low_mem"],
        metavar="PRESET",
        help="Model configuration presets (default: predict pae_enabled low_mem).",
    )
    p_run.add_argument("--runner-yaml", type=Path, default=None,
                       help="Explicit YAML config file; overrides --presets.")
    _add_parse_args(p_run, include_chain_args=True)

    # --- prepare-query subcommand ---
    p_prep = sub.add_parser(
        "prepare-query",
        help="Prepare OF3 query JSON for binder refolding (receptor as template).",
    )
    p_prep.add_argument(
        "--complex", type=Path, required=True, metavar="CIF",
        help="Complex CIF/PDB file (receptor + binder).",
    )
    p_prep.add_argument(
        "--receptor-chain", type=str, required=True, metavar="CHAIN",
        help="Chain ID of the receptor/target (fixed as template).",
    )
    p_prep.add_argument(
        "--binder-chain", type=str, required=True, metavar="CHAIN",
        help="Chain ID of the binder (refolded from sequence only).",
    )
    p_prep.add_argument(
        "--query-name", "-n", type=str, required=True,
        help="Prediction query name.",
    )
    p_prep.add_argument(
        "--output-dir", "-o", type=Path, required=True,
        help="Directory for query JSON and template files.",
    )
    p_prep.add_argument(
        "--template-cif", type=Path, default=None, metavar="CIF",
        help="Pre-prepared receptor CIF (e.g., after MD relaxation). "
             "If omitted, receptor chain is extracted from --complex.",
    )

    # --- refold subcommand ---
    p_refold = sub.add_parser(
        "refold",
        help="Run OF3 binder refolding: receptor fixed as template, binder predicted freely.",
    )
    p_refold.add_argument(
        "--complex", type=Path, required=True, metavar="CIF",
        help="Complex CIF/PDB file (receptor + binder).",
    )
    p_refold.add_argument(
        "--receptor-chain", type=str, required=True, metavar="CHAIN",
        help="Chain ID of the receptor/target (fixed as template).",
    )
    p_refold.add_argument(
        "--binder-chain", type=str, required=True, metavar="CHAIN",
        help="Chain ID of the binder (refolded from sequence only).",
    )
    p_refold.add_argument(
        "--query-name", "-n", type=str, required=True,
        help="Prediction query name.",
    )
    p_refold.add_argument(
        "--output-dir", "-o", type=Path, required=True,
        help="Top-level output directory (query/ and predictions/ written here).",
    )
    p_refold.add_argument(
        "--template-cif", type=Path, default=None, metavar="CIF",
        help="Pre-prepared receptor CIF (e.g., after MD relaxation).",
    )
    p_refold.add_argument("--ckpt", type=Path, default=None,
                          help="Model checkpoint path.")
    p_refold.add_argument("--num-samples", type=int, default=5,
                          help="Number of diffusion samples (default: 5).")
    p_refold.add_argument("--num-seeds", type=int, default=1,
                          help="Number of random seeds (default: 1).")
    p_refold.add_argument("--no-msa-server", action="store_true",
                          help="Disable ColabFold MSA server.")
    p_refold.add_argument(
        "--presets", nargs="+", default=["predict", "pae_enabled", "low_mem"],
        metavar="PRESET", help="Model configuration presets.",
    )
    p_refold.add_argument("--runner-yaml", type=Path, default=None,
                          help="Explicit YAML config; overrides --presets.")
    _add_parse_args(p_refold, include_chain_args=False)

    # --- prepare-scoring-query subcommand ---
    p_prep_score = sub.add_parser(
        "prepare-scoring-query",
        help="Prepare OF3 query JSON to score an existing complex (both chains as templates).",
    )
    p_prep_score.add_argument("--complex", type=Path, required=True, metavar="CIF",
                              help="Complex CIF/PDB file (receptor + binder).")
    p_prep_score.add_argument("--receptor-chain", type=str, required=True, metavar="CHAIN",
                              help="Chain ID of the receptor.")
    p_prep_score.add_argument("--binder-chain", type=str, required=True, metavar="CHAIN",
                              help="Chain ID of the binder.")
    p_prep_score.add_argument("--query-name", "-n", type=str, required=True,
                              help="Prediction query name.")
    p_prep_score.add_argument("--output-dir", "-o", type=Path, required=True,
                              help="Directory for query JSON and template files.")
    p_prep_score.add_argument(
        "--template-cif", type=Path, default=None, metavar="CIF",
        help="Pre-prepared complex CIF (e.g., MD-relaxed). Both chains extracted from it.",
    )

    # --- score subcommand ---
    p_score = sub.add_parser(
        "score",
        help="Run OF3 scoring of an existing complex (both chains as templates).",
    )
    p_score.add_argument("--complex", type=Path, required=True, metavar="CIF",
                         help="Complex CIF/PDB file (receptor + binder).")
    p_score.add_argument("--receptor-chain", type=str, required=True, metavar="CHAIN",
                         help="Chain ID of the receptor.")
    p_score.add_argument("--binder-chain", type=str, required=True, metavar="CHAIN",
                         help="Chain ID of the binder.")
    p_score.add_argument("--query-name", "-n", type=str, required=True,
                         help="Prediction query name.")
    p_score.add_argument("--output-dir", "-o", type=Path, required=True,
                         help="Top-level output directory.")
    p_score.add_argument(
        "--template-cif", type=Path, default=None, metavar="CIF",
        help="Pre-prepared complex CIF (e.g., MD-relaxed). Both chains extracted from it.",
    )
    p_score.add_argument("--ckpt", type=Path, default=None, help="Model checkpoint path.")
    p_score.add_argument("--num-samples", type=int, default=5,
                         help="Number of diffusion samples (default: 5).")
    p_score.add_argument("--num-seeds", type=int, default=1,
                         help="Number of random seeds (default: 1).")
    p_score.add_argument("--no-msa-server", action="store_true",
                         help="Disable ColabFold MSA server.")
    p_score.add_argument(
        "--presets", nargs="+", default=["predict", "pae_enabled", "low_mem"],
        metavar="PRESET", help="Model configuration presets.",
    )
    p_score.add_argument("--runner-yaml", type=Path, default=None,
                         help="Explicit YAML config; overrides --presets.")
    _add_parse_args(p_score, include_chain_args=False)

    args = parser.parse_args()

    # --- prepare-scoring-query ---
    if args.command == "prepare-scoring-query":
        path = prepare_scoring_query(
            complex_structure_path=args.complex,
            receptor_chain=args.receptor_chain,
            binder_chain=args.binder_chain,
            query_name=args.query_name,
            output_dir=args.output_dir,
            template_cif_path=args.template_cif,
        )
        print(f"Scoring query JSON written to: {path}")
        return

    # --- score ---
    if args.command == "score":
        print(f"Running OF3 structure scoring: {args.complex}")
        print(f"  Receptor chain (template): {args.receptor_chain}")
        print(f"  Binder chain  (template): {args.binder_chain}")
        predictions_dir = run_openfold_scoring(
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
        )
        print(f"\nParsing scoring metrics from: {predictions_dir}")
        metrics = compute_openfold_metrics(
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
        path = prepare_refolding_query(
            complex_structure_path=args.complex,
            receptor_chain=args.receptor_chain,
            binder_chain=args.binder_chain,
            query_name=args.query_name,
            output_dir=args.output_dir,
            template_cif_path=args.template_cif,
        )
        print(f"Query JSON written to: {path}")
        return

    # --- refold ---
    if args.command == "refold":
        print(f"Running OF3 binder refolding: {args.complex}")
        print(f"  Receptor chain (template): {args.receptor_chain}")
        print(f"  Binder chain (free):       {args.binder_chain}")
        predictions_dir = run_openfold_refolding(
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
        )
        print(f"\nParsing refolding metrics from: {predictions_dir}")
        metrics = compute_openfold_metrics(
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

    # --- parse (and fallthrough from run) ---
    print(f"\nParsing OpenFold3 metrics for: {args.query_name}")
    metrics = compute_openfold_metrics(
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


if __name__ == "__main__":
    main()
