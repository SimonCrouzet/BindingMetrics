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

Layout: this module parses outputs (``compute_openfold_metrics``) and runs
inference (``run_openfold`` and the ``run_openfold_*`` wrappers). The runner
YAML and query preparation live in ``_openfold_run.py`` and the command line in
``_openfold_cli.py``; every name they define is re-exported here.

References:
  The OpenFold3 Team (2025) OpenFold3-preview. Software, doi:10.5281/zenodo.19001000,
  github.com/aqlaboratory/openfold-3. This is the citation the repository asks
  for; it also asks that work citing OpenFold3 cite AlphaFold 3, below. (The
  "Ahdritz et al. 2024" OpenFold paper, Nat. Methods 21:1514, describes the
  AlphaFold2 reimplementation, not OpenFold3.)
  Abramson et al. (2024) Accurate structure prediction of biomolecular
  interactions with AlphaFold 3. Nature 630:493-500. Defines the outputs parsed
  here (pLDDT, PAE, PDE, pTM, ipTM) and the tokenisation that the interface
  PDE/PAE slicing relies on: one token per standard residue, one per heavy atom
  for ligands and modified residues.

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

import json
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from binding_metrics.metrics._common import import_biotite, load_structure, resolve_chain_role
from binding_metrics.metrics._openfold_cli import (  # noqa: F401  (re-exported)
    _add_parse_args,
    _add_query_seeds_arg,
    _print_metrics,
    main,
)
from binding_metrics.metrics._openfold_run import (  # noqa: F401  (re-exported)
    _DEFAULT_QUERY_SEEDS,
    _BatchSample,
    _extract_chain_to_cif,
    _extract_sequence_from_structure,
    _query_seeds,
    _safe_entry_id,
    _write_a3m_self_alignment,
    _write_runner_yaml,
    prepare_batched_refolding_queries,
    prepare_batched_scoring_queries,
    prepare_refolding_query,
    prepare_scoring_query,
)

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
        seed: Seed index (1-based index into the sorted list of seed directories),
              not a seed value. OF3 transforms input seeds, so this selects by
              position rather than value.
        sample: Sample index (default 1).

    Returns:
        Dict with keys: ``structure``, ``confidences``, ``confidences_aggregated``,
        ``timing``. Values are resolved Paths or None if not found.
    """
    query_dir = output_dir / query_name
    seed_dirs = sorted(query_dir.glob("seed_*")) if query_dir.exists() else []
    if seed_dirs and 1 <= seed <= len(seed_dirs):
        seed_dir = seed_dirs[seed - 1]
        actual_seed = seed_dir.name[len("seed_") :]
    else:
        actual_seed = str(seed)
        seed_dir = query_dir / f"seed_{seed}"
    prefix = f"{query_name}_seed_{actual_seed}_sample_{sample}"

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
    struc, pdbx, _ = import_biotite("per-chain structural analysis")
    return struc, pdbx


def _load_atoms(path: Path):
    """Load an AtomArray from a CIF or PDB file using biotite (model 1)."""
    return load_structure(path, purpose="per-chain structural analysis")


def _chain_token_offsets(atoms) -> dict[str, tuple[int, int]]:
    """Map chain IDs to [start, end) PAE token ranges (one token per residue).

    Preserves the order chains first appear in the structure, which matches
    the PAE matrix token ordering (same order as the query JSON chains).
    The one-token-per-residue assumption fails for ligands and modified
    residues; :func:`_check_token_offsets` compares the result with the matrix.

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


def _check_token_offsets(
    offsets: dict[str, tuple[int, int]],
    matrix: np.ndarray,
    matrix_name: str,
) -> None:
    """Raise ``ValueError`` unless residue-based token offsets fit ``matrix``.

    ``_chain_token_offsets`` assumes one token per residue. AlphaFold3-style
    models use one token per standard residue but one per heavy atom for
    ligands and modified residues, so a structure with such components has
    fewer residues than the matrix has tokens and every offset after the first
    such component would slice the wrong block. Only an exact match is trusted.

    Args:
        offsets: Result of :func:`_chain_token_offsets`.
        matrix: PDE or PAE matrix.
        matrix_name: ``"PDE"`` or ``"PAE"``, for the error message.

    Raises:
        ValueError: If the matrix is not square or its size differs from the
            total residue count.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_name} matrix must be square, got shape {matrix.shape}.")
    n_tokens = matrix.shape[0]
    n_residues = max((end for _, end in offsets.values()), default=0)
    if n_tokens != n_residues:
        per_chain = ", ".join(f"{cid}: {end - start}" for cid, (start, end) in offsets.items())
        raise ValueError(
            f"{matrix_name} matrix has {n_tokens} tokens but the structure has {n_residues} "
            f"residues ({per_chain}). Residue-based chain offsets do not apply, probably "
            "because a ligand, ion or modified residue is tokenised per atom."
        )


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
    return np.array([chain_plddt[chain_res_ids == r].mean() for r in unique_res], dtype=float)


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
            pred_binder_ca = transform.apply(pred_binder_ca)

    return float(struc.rmsd(ref_binder_ca, pred_binder_ca))


def _interface_pde_stats(
    pde: np.ndarray,
    atoms,
    binder_chain: str,
    receptor_chain: str,
) -> dict:
    """PDE statistics for the binder–receptor interface region.

    Slices the full PDE matrix to the binder×receptor token sub-matrix and
    returns summary statistics and (optionally) the raw slice.

    Args:
        pde: Full PDE matrix, shape ``(n_tokens, n_tokens)``.
        atoms: Biotite AtomArray of the predicted structure.
        binder_chain: Chain ID of the binder.
        receptor_chain: Chain ID of the receptor.

    Returns:
        Dict with:
            ``pde_interface`` (np.ndarray): Sub-matrix ``(n_binder_res, n_receptor_res)``.
            ``mean_interface_pde`` (float): Mean PDE over the interface slice (Å).
            ``max_interface_pde`` (float): Max PDE over the interface slice (Å).
            ``n_binder_tokens`` (int): Binder residue token count.
            ``n_receptor_tokens`` (int): Receptor residue token count.

    Raises:
        ValueError: If a chain is not in the structure, or the matrix size does
            not equal the residue count (see :func:`_check_token_offsets`).
    """
    offsets = _chain_token_offsets(atoms)
    missing = [c for c in (binder_chain, receptor_chain) if c not in offsets]
    if missing:
        raise ValueError(f"Chains not found in structure: {missing}")
    _check_token_offsets(offsets, pde, "PDE")
    b0, b1 = offsets[binder_chain]
    r0, r1 = offsets[receptor_chain]
    sub = pde[b0:b1, r0:r1]
    return {
        "pde_interface": sub,
        "mean_interface_pde": float(sub.mean()),
        "max_interface_pde": float(sub.max()),
        "n_binder_tokens": b1 - b0,
        "n_receptor_tokens": r1 - r0,
    }


def _interface_pae_stats(
    pae: np.ndarray,
    atoms,
    binder_chain: str,
    receptor_chain: str,
) -> dict:
    """PAE statistics for the binder–receptor interface region.

    Slices the full PAE (predicted aligned error) matrix to the
    binder×receptor token sub-matrix and returns summary statistics and the
    raw slice. PAE at token (i, j) is the expected position error of token i
    when the prediction is aligned on token j; the interface block therefore
    reports how confidently the binder is placed relative to the receptor.

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

    Raises:
        ValueError: If a chain is not in the structure, or the matrix size does
            not equal the residue count (see :func:`_check_token_offsets`).
    """
    offsets = _chain_token_offsets(atoms)
    missing = [c for c in (binder_chain, receptor_chain) if c not in offsets]
    if missing:
        raise ValueError(f"Chains not found in structure: {missing}")
    _check_token_offsets(offsets, pae, "PAE")
    b0, b1 = offsets[binder_chain]
    r0, r1 = offsets[receptor_chain]
    # PAE is asymmetric; average the binder→receptor and receptor→binder blocks
    # so the reported interface confidence does not depend on slice direction.
    sub_br = pae[b0:b1, r0:r1]
    sub_rb = pae[r0:r1, b0:b1]
    return {
        "pae_interface": sub_br,
        "mean_interface_pae": float((sub_br.mean() + sub_rb.mean()) / 2.0),
        "max_interface_pae": float(max(sub_br.max(), sub_rb.max())),
        "n_binder_tokens": b1 - b0,
        "n_receptor_tokens": r1 - r0,
    }


def compute_interface_pae(
    confidences_path,
    structure_path,
    binder_chain: str,
    receptor_chain: Optional[str] = None,
    *,
    target_chain: Optional[str] = None,
) -> dict:
    """Interface PAE slice from OpenFold3 output.

    OpenFold3 (>= v0.4.1) writes the full ``pae`` matrix to
    ``*_confidences.json/.npz`` alongside ``plddt`` and ``pde`` when the full
    confidence scores are requested with the ``pae_enabled`` model preset. This
    loads that matrix and the predicted structure and returns the interface
    (binder×receptor) PAE statistics.

    Args:
        confidences_path: Path to ``*_confidences.json`` or ``.npz``.
        structure_path: Path to the predicted model (.cif/.pdb) — used to map
            chains to PAE token ranges.
        binder_chain: Chain ID of the binder.
        receptor_chain: Chain ID of the receptor. Required, through this
            parameter or ``target_chain``.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dict from :func:`_interface_pae_stats` (``pae_interface``,
        ``mean_interface_pae``, ``max_interface_pae``, token counts).

    Raises:
        ValueError: If the confidences file has no PAE matrix (the run did not
            enable the PAE head / persist full confidences), or if the matrix
            size does not equal the structure's residue count (a ligand or
            modified residue makes the token count differ).
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    conf = _parse_confidences(Path(confidences_path))
    pae = conf.get("pae")
    if pae is None:
        raise ValueError(
            f"No PAE matrix in {confidences_path}. Re-run OpenFold3 with the "
            "'pae_enabled' preset and full confidence output so the 'pae' array "
            "is written to the confidences file."
        )
    atoms = _load_atoms(Path(structure_path))
    return _interface_pae_stats(pae, atoms, binder_chain, receptor_chain)


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
    *,
    seed_index: Optional[int] = None,
    target_chain: Optional[str] = None,
) -> dict:
    """Extract confidence metrics from OpenFold3 output files.

    **Mode 1 — confidence metrics only (no reference):**
    Parse the aggregated and per-atom confidence files to get scalar metrics
    (avg_plddt, ptm, iptm, chain scores) and optionally per-residue binder
    pLDDT and interface PDE statistics. Pass ``binder_chain`` (and optionally
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
        seed: 1-based index into the sorted ``seed_*`` directories of the query,
            not a random seed value: OpenFold3 names the directories after its
            own transformed seeds, so they are selected by position (default 1).
        sample: Sample index to parse (default 1).
        include_matrices: If True, include the full PDE matrix in the result
            (can be large). Default False.
        reference_structure_path: Optional path to a reference CIF/PDB
            (e.g., crystal structure). When supplied together with
            ``binder_chain``, the binder Cα RMSD between the OF3 prediction
            and this reference is computed and stored as ``binder_ca_rmsd``.
        binder_chain: Chain ID of the binder/ligand in the predicted structure.
            Enables per-residue pLDDT for the binder, interface PDE stats
            (when ``receptor_chain`` is also given), and binder RMSD
            (when ``reference_structure_path`` is also given).
        receptor_chain: Chain ID of the receptor/target. When provided
            alongside ``binder_chain``:
              - Interface PDE statistics are computed (mean/max PDE for the
                binder×receptor token block).
              - Receptor Cα atoms are used as the superposition reference
                when computing ``binder_ca_rmsd``, giving the
                receptor-frame binder RMSD.
        seed_index: Clearer name for ``seed``; when given it takes precedence.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dictionary with keys:

        Structure:
            structure_path (str | None): path to the predicted .cif/.pdb file
            query_name (str), seed (int, the seed index used), sample (int)

        Scalar confidence metrics [from confidences_aggregated.json]:
            avg_plddt (float): mean pLDDT across all atoms [0–100]
            gpde (float): global predicted distance error (Å)
            ptm (float): predicted TM-score [0–1]; NaN if pae_enabled preset off
            iptm (float): interface pTM [0–1]; NaN if single chain or pae_enabled off
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
            max_pde (float): max PDE value (Å)
            pae (np.ndarray | None): PAE matrix (n_tokens×n_tokens); only if
                include_matrices=True and the run persisted the PAE head
            max_pae (float): max PAE value (Å); NaN if no PAE matrix present

        Per-chain structural analysis [requires binder_chain]:
            binder_plddt_per_residue (np.ndarray | None): mean pLDDT per
                residue for the binder chain, shape (n_binder_res,)
            binder_avg_plddt (float): mean pLDDT over all binder residues

        Interface PDE / PAE [requires binder_chain + receptor_chain]:
            The block is located with one token per residue. When the matrix
            size differs from the structure's residue count (a ligand, ion or
            modified residue is tokenised per atom), the interface values stay
            NaN, a warning is issued and ``reason`` says why.
            mean_interface_pde (float): mean PDE over binder×receptor tokens (Å)
            max_interface_pde (float): max PDE over binder×receptor tokens (Å)
            pde_interface (np.ndarray | None): raw PDE slice, shape
                (n_binder_res, n_receptor_res); only if include_matrices=True
            mean_interface_pae (float): mean PAE over the interface tokens (Å),
                averaged over both slice directions; NaN if no PAE persisted
            max_interface_pae (float): max PAE over the interface tokens (Å)
            pae_interface (np.ndarray | None): raw PAE slice (binder→receptor);
                only if include_matrices=True

        Refolding RMSD [requires binder_chain + reference_structure_path]:
            binder_ca_rmsd (float): binder Cα RMSD vs. reference (Å).
                Computed in the receptor frame if receptor_chain is given
                (predicted structure superposed on receptor Cα first).

        Timing:
            timing (dict): runtime entries from timing.json, empty if absent

        Failures:
            reason (str): present only when a value could not be computed (it
                keeps its NaN or None sentinel). Names each affected analysis
                and its cause, separated by "; ": missing output files, a
                missing model structure, a pLDDT or PDE/PAE array that does not
                fit the structure, a binder RMSD mismatch, or a structure that
                could not be parsed.
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    if seed_index is not None:
        seed = seed_index
    output_dir = Path(output_dir)
    files = _find_prediction_files(output_dir, query_name, seed=seed, sample=sample)
    reasons: list[str] = []

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
        "max_pde": float("nan"),
        "pae": None,
        "max_pae": float("nan"),
        # Per-chain structural analysis (populated when binder_chain is given)
        "binder_plddt_per_residue": None,
        "binder_avg_plddt": float("nan"),
        # Interface PDE / PAE (populated when binder_chain + receptor_chain are given)
        "mean_interface_pde": float("nan"),
        "max_interface_pde": float("nan"),
        "pde_interface": None,
        "mean_interface_pae": float("nan"),
        "max_interface_pae": float("nan"),
        "pae_interface": None,
        # Refolding RMSD (populated when binder_chain + reference_structure_path)
        "binder_ca_rmsd": float("nan"),
        # Timing
        "timing": {},
    }

    if files["confidences_aggregated"] is None and files["confidences"] is None:
        reasons.append(
            f"no confidence files found for query '{query_name}' "
            f"(seed index {seed}, sample {sample}) in {output_dir}"
        )
    elif files["confidences_aggregated"] is None:
        reasons.append("aggregated confidences file not found")
    elif files["confidences"] is None:
        reasons.append("per-atom confidences file not found")

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

        pde = conf["pde"]
        if pde is not None:
            result["max_pde"] = float(pde.max())

        pae = conf["pae"]
        if pae is not None:
            result["max_pae"] = float(pae.max())

    # --- Timing ---
    if files["timing"] is not None:
        result["timing"] = _parse_timing(files["timing"])

    # --- Per-chain structural analysis ---
    # Requires binder_chain; uses the predicted model CIF.
    if binder_chain is not None and files["structure"] is None:
        reasons.append("structure file not found; per-chain values not computed")
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
                except ValueError as exc:  # pLDDT length differs from the atom count
                    warnings.warn(
                        f"compute_openfold_metrics: per-residue binder pLDDT skipped: {exc}"
                    )
                    reasons.append(f"binder pLDDT: {exc}")
            elif files["confidences"] is not None:
                reasons.append("binder pLDDT: no per-atom pLDDT in the confidences file")

            # Interface PDE statistics (binder × receptor token block)
            if receptor_chain is not None:
                pde_src = result.get("pde")
                if pde_src is None and files["confidences"] is not None:
                    # Load PDE even when include_matrices=False for stats only
                    pde_src = _parse_confidences(files["confidences"]).get("pde")
                if pde_src is not None:
                    try:
                        pde_stats = _interface_pde_stats(
                            pde_src, pred_atoms, binder_chain, receptor_chain
                        )
                        result["mean_interface_pde"] = pde_stats["mean_interface_pde"]
                        result["max_interface_pde"] = pde_stats["max_interface_pde"]
                        if include_matrices:
                            result["pde_interface"] = pde_stats["pde_interface"]
                    except ValueError as exc:  # missing chain or token/residue mismatch
                        warnings.warn(f"compute_openfold_metrics: interface PDE skipped: {exc}")
                        reasons.append(f"interface PDE: {exc}")
                elif files["confidences"] is not None:
                    reasons.append("interface PDE: no PDE matrix in the confidences file")

                # Interface PAE statistics (binder × receptor token block)
                pae_src = result.get("pae")
                if pae_src is None and files["confidences"] is not None:
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
                    except ValueError as exc:  # missing chain or token/residue mismatch
                        warnings.warn(f"compute_openfold_metrics: interface PAE skipped: {exc}")
                        reasons.append(f"interface PAE: {exc}")
                elif files["confidences"] is not None:
                    reasons.append("interface PAE: no PAE matrix in the confidences file")

            # Binder Cα RMSD vs. reference structure
            if reference_structure_path is not None:
                try:
                    ref_atoms = _load_atoms(Path(reference_structure_path))
                    result["binder_ca_rmsd"] = _binder_ca_rmsd(
                        pred_atoms, ref_atoms, binder_chain, receptor_chain
                    )
                except (ValueError, OSError) as exc:  # Cα count mismatch or unreadable file
                    warnings.warn(f"compute_openfold_metrics: binder RMSD skipped: {exc}")
                    reasons.append(f"binder RMSD: {exc}")

        except Exception as exc:
            # Broad on purpose: the model file is external input and structure parsers
            # raise several exception types. Values computed so far are kept.
            warnings.warn(f"compute_openfold_metrics: structural analysis failed: {exc}")
            reasons.append(f"structural analysis failed: {type(exc).__name__}: {exc}")

    if reasons:
        result["reason"] = "; ".join(reasons)
    return result


# ---------------------------------------------------------------------------
# Runner: invoke OpenFold3 as a subprocess
# ---------------------------------------------------------------------------


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
    conda_env: Optional[str] = None,
    template_dir: Optional[str | Path] = None,
) -> Path:
    """Run OpenFold3 inference as a subprocess.

    Invokes ``run_openfold predict`` from the OpenFold3 package.
    OpenFold3 must be installed in either the current environment or a named
    conda environment (see ``conda_env``).

    Args:
        query_json: Path to the input JSON file describing the prediction query.
        output_dir: Directory where OpenFold3 writes predictions.
        inference_ckpt_path: Optional path to a model checkpoint (.pt file).
            Uses the default downloaded checkpoint if None.
        num_diffusion_samples: Number of structure samples per query (default 5).
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values of a query are the ``"seeds"`` list in ``query_json``, which
            the ``prepare_*`` functions set from their ``seeds`` argument
            (default ``[42]``).
        use_msa_server: Use the ColabFold MSA server for alignment generation
            (default True). MSAs then come from a remote service, so results can
            change over time, and the sequences leave the machine. Set False if
            MSAs are pre-computed.
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
        conda_env: Name of the conda environment where OpenFold3 is installed
            (e.g. ``"openfold3"``). When given, the command is wrapped as
            ``conda run -n {conda_env} --no-capture-output run_openfold ...``.
            If None, ``run_openfold`` must be on the current PATH.

    Returns:
        Path to the output directory.

    Raises:
        FileNotFoundError: If ``run_openfold`` is not on PATH and no
            ``conda_env`` is specified.
        subprocess.CalledProcessError: If OpenFold3 exits non-zero.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if conda_env is None and shutil.which("run_openfold") is None:
        raise FileNotFoundError(
            "run_openfold not found on PATH. "
            "Pass conda_env='openfold3' (or whichever env has OF3 installed)."
        )

    # Resolve runner YAML: explicit path takes precedence over model_presets
    effective_yaml: Optional[Path] = None
    if runner_yaml is not None:
        effective_yaml = Path(runner_yaml)
    else:
        presets = (
            list(model_presets)
            if model_presets is not None
            else ["predict", "pae_enabled", "low_mem"]
        )
        if "predict" not in presets:
            presets.insert(0, "predict")
        effective_yaml = _write_runner_yaml(
            output_dir,
            presets,
            template_dir=Path(template_dir) if template_dir is not None else None,
        )

    of3_cmd = [
        "run_openfold",
        "predict",
        f"--query_json={query_json}",
        f"--output_dir={output_dir}",
        f"--num_diffusion_samples={num_diffusion_samples}",
        f"--num_model_seeds={num_model_seeds}",
        f"--use_msa_server={str(use_msa_server).lower()}",
        f"--runner_yaml={effective_yaml}",
    ]

    if inference_ckpt_path is not None:
        of3_cmd.append(f"--inference_ckpt_path={inference_ckpt_path}")

    if extra_args:
        of3_cmd.extend(extra_args)

    if conda_env is not None:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output"] + of3_cmd
    else:
        cmd = of3_cmd

    subprocess.run(cmd, check=True)
    return output_dir


# ---------------------------------------------------------------------------
# Wrappers: prepare the query, then run OpenFold3 (scoring and refolding)
# ---------------------------------------------------------------------------


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
    conda_env: Optional[str] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
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
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values themselves are set by ``seeds``.
        use_msa_server: Use the ColabFold MSA server (default True). MSAs then
            come from a remote service, so results can change over time, and
            the sequences leave the machine. Pass False with pre-computed MSAs.
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3.
        conda_env: Conda environment where OpenFold3 is installed.
        seeds: Seed values written to the query JSON (default ``(42,)``).

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
        seeds=seeds,
    )

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=extra_args,
        conda_env=conda_env,
        template_dir=query_dir / "templates",
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
    conda_env: Optional[str] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
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
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values themselves are set by ``seeds``.
        use_msa_server: Use the ColabFold MSA server (default True). MSAs then
            come from a remote service, so results can change over time, and
            the sequences leave the machine. Pass False with pre-computed MSAs.
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3. To use the template CIF,
            pass ``["--template_mmcif_dir=<path>"]`` if OF3 requires it.
            By default ``--template_mmcif_dir`` is automatically appended
            pointing to ``{output_dir}/query/templates/``.
        conda_env: Conda environment where OpenFold3 is installed.
        seeds: Seed values written to the query JSON (default ``(42,)``).

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
        seeds=seeds,
    )

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=extra_args,
        conda_env=conda_env,
        template_dir=query_dir / "templates",
    )
    return predictions_dir


# ---------------------------------------------------------------------------
# Batched inference (multiple queries in one OF3 subprocess)
# ---------------------------------------------------------------------------


def run_openfold_batched(
    samples: list[_BatchSample],
    output_dir: str | Path,
    mode: str = "score",
    inference_ckpt_path: Optional[str | Path] = None,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    use_msa_server: bool = True,
    model_presets: Optional[list[str]] = None,
    runner_yaml: Optional[str | Path] = None,
    extra_args: Optional[list[str]] = None,
    conda_env: Optional[str] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
) -> Path:
    """Run OpenFold3 inference on multiple samples in a single subprocess.

    Prepares all queries into one combined JSON, calls ``run_openfold``
    once (amortising model loading and CUDA initialisation), and returns
    the predictions directory.  Use :func:`compute_openfold_metrics` with
    each sample's ``query_name`` to extract per-sample results.

    Args:
        samples: Per-sample descriptors.
        output_dir: Top-level output directory.
        mode: ``"score"`` (both chains as templates) or ``"refold"``
            (binder predicted from sequence only).
        inference_ckpt_path: Optional model checkpoint path.
        num_diffusion_samples: Structure samples per query (default 5).
        num_model_seeds: Passed to OpenFold3 as ``--num_model_seeds``. The seed
            values themselves are set by ``seeds``.
        use_msa_server: Use the ColabFold MSA server (default True). MSAs then
            come from a remote service, so results can change over time, and
            the sequences leave the machine. Pass False with pre-computed MSAs.
        model_presets: Model configuration presets.
        runner_yaml: Explicit runner YAML; overrides ``model_presets``.
        extra_args: Additional CLI args for OF3.
        conda_env: Conda environment name (default None).
        seeds: Seed values written to the query JSON (default ``(42,)``).

    Returns:
        Path to the OF3 predictions output directory.
    """
    output_dir = Path(output_dir)
    query_dir = output_dir / "query"
    predictions_dir = output_dir / "predictions"

    if mode == "refold":
        query_json = prepare_batched_refolding_queries(samples, query_dir, seeds=seeds)
    else:
        query_json = prepare_batched_scoring_queries(samples, query_dir, seeds=seeds)

    run_openfold(
        query_json=query_json,
        output_dir=predictions_dir,
        inference_ckpt_path=inference_ckpt_path,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        model_presets=model_presets,
        runner_yaml=runner_yaml,
        extra_args=extra_args,
        conda_env=conda_env,
        template_dir=query_dir / "templates",
    )
    return predictions_dir


if __name__ == "__main__":
    main()
