"""Confidence metrics of a structure prediction, from any model with an adapter.

``summarize_prediction`` turns a ``binding_metrics.predictors.PredictionRecord`` into the
result dictionary that ``compute_openfold_metrics`` has always returned, plus a ``model`` key:
the scalar scores, per-residue pLDDT of the binder, interface PDE and PAE statistics and the
binder C-alpha RMSD against a reference. It reads only the record, so the analysis is the same
for every model; a model-specific value stays in ``record.extras``.

Chains: ``binder_chain`` and ``receptor_chain`` are the chain IDs of the user's input, that is
the IDs of ``record.atoms()`` after ``record.chain_map`` has renamed the model's chains.

Not computed values keep their sentinel (NaN, None, empty dict) and a sentence in ``reason``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from binding_metrics.metrics._common import resolve_chain_role
from binding_metrics.predictors._confidence import (
    _binder_ca_rmsd,
    _binder_plddt_per_residue,
    _interface_pae_stats,
    _interface_pde_stats,
    _load_atoms,
)
from binding_metrics.predictors.record import PredictionRecord

_NAN = float("nan")


def _token_ranges(record: PredictionRecord) -> Optional[dict[str, tuple[int, int]]]:
    """Token ranges by the user's chain IDs, or None for one token per residue.

    A record with a ``TokenLayout`` says which tokens belong to which chain, so the
    interface block is cut with it and not with residue counts. The layout keeps the
    model's chain IDs; ``record.chain_map`` renames them like the atoms.

    Raises:
        ValueError: If the tokens of a chain are not contiguous.
    """
    if record.tokens is None:
        return None
    return {
        record.chain_map.get(chain, chain): span
        for chain, span in record.tokens.token_ranges().items()
    }


def summarize_prediction(
    record: PredictionRecord,
    *,
    include_matrices: bool = False,
    reference_structure_path: Optional[str | Path] = None,
    binder_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    caller: str = "summarize_prediction",
    stacklevel: int = 2,
) -> dict:
    """Summarise one prediction record as a result dictionary.

    Args:
        record: The parsed prediction sample.
        include_matrices: Include the full PDE and PAE matrices and the interface slices
            (large). The scalar statistics are computed either way.
        reference_structure_path: Reference CIF or PDB. With ``binder_chain`` it gives
            ``binder_ca_rmsd``: superposed on the receptor C-alpha when ``receptor_chain`` is
            given, so the number is the binder displacement in the receptor frame.
        binder_chain: Binder chain ID in the user's naming. Enables the per-residue binder
            pLDDT, the interface statistics (with ``receptor_chain``) and the RMSD.
        receptor_chain: Receptor chain ID in the user's naming.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise ``ValueError``.
        caller: Name used to prefix the warnings, so that a wrapper can name itself.
        stacklevel: ``stacklevel`` of the warnings; the default names the caller of this
            function, a wrapper passes one more.

    Returns:
        A dict with the keys below, in this order.

        Identity:
            model (str), query_name (str, the prediction name), seed (int, the seed index),
            sample (int), structure_path (str | None)

        Scalars (NaN when the model does not provide them):
            avg_plddt [0-100], gpde (A), ptm, iptm [0-1], disorder, has_clash,
            sample_ranking_score (the model's own ranking score), chain_ptm (dict),
            chain_pair_iptm (dict; keys as the model writes them, ``"(A, B)"`` for
            OpenFold3), bespoke_iptm (dict; OpenFold3's own, ``{}`` for other models)

        Arrays:
            plddt_per_atom (ndarray | None), n_atoms (int), pde and pae (ndarray | None; only
            with ``include_matrices``), max_pde and max_pae (A, NaN when absent)

        Binder (needs ``binder_chain``):
            binder_plddt_per_residue (ndarray | None), binder_avg_plddt

        Interface (needs both chains). The block is cut with ``record.tokens`` when the record
        has a token layout, and otherwise located with one token per residue; it is left NaN,
        with a warning and a reason, when the matrix size differs from the residue count (a
        ligand or modified residue tokenised per atom):
            mean_interface_pde, max_interface_pde, pde_interface (only with
            ``include_matrices``), mean_interface_pae (average of both slice directions),
            max_interface_pae, pae_interface (binder rows, receptor columns; only with
            ``include_matrices``)

        Reference (needs ``binder_chain`` and ``reference_structure_path``):
            binder_ca_rmsd (A)

        timing (dict), and ``reason`` (str) only when a value could not be computed: the
        parser's reasons followed by one clause for each affected analysis, joined by "; ".
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    reasons: list[str] = list(record.reasons)
    # A missing full-confidence file is already explained by the parser; the clauses below
    # apply when the file was read but does not hold the value.
    arrays_read = record.files is not None and record.files.arrays is not None
    plddt_arr = record.plddt_per_atom

    result: dict = {
        "model": record.model,
        "query_name": record.name,
        "seed": record.seed_index,
        "sample": record.sample,
        "structure_path": str(record.structure_path) if record.structure_path else None,
        # Scalar confidence metrics (NaN = not available)
        "avg_plddt": record.avg_plddt,
        "gpde": record.gpde,
        "ptm": record.ptm,
        "iptm": record.iptm,
        "disorder": record.disorder,
        "has_clash": record.has_clash,
        "sample_ranking_score": record.ranking_score,
        "chain_ptm": record.chain_ptm,
        "chain_pair_iptm": record.chain_pair_iptm,
        "bespoke_iptm": record.extras.get("bespoke_iptm", {}),
        # Per-atom data
        "plddt_per_atom": plddt_arr,
        "n_atoms": record.n_atoms,
        "pde": record.pde if include_matrices else None,
        "max_pde": float(record.pde.max()) if record.pde is not None else _NAN,
        "pae": record.pae if include_matrices else None,
        "max_pae": float(record.pae.max()) if record.pae is not None else _NAN,
        # Per-chain structural analysis (populated when binder_chain is given)
        "binder_plddt_per_residue": None,
        "binder_avg_plddt": _NAN,
        # Interface PDE / PAE (populated when binder_chain + receptor_chain are given)
        "mean_interface_pde": _NAN,
        "max_interface_pde": _NAN,
        "pde_interface": None,
        "mean_interface_pae": _NAN,
        "max_interface_pae": _NAN,
        "pae_interface": None,
        # Refolding RMSD (populated when binder_chain + reference_structure_path)
        "binder_ca_rmsd": _NAN,
        # Timing
        "timing": record.timing,
    }

    # --- Per-chain structural analysis ---
    # Requires binder_chain; uses the predicted structure.
    if binder_chain is not None and record.structure_path is None:
        reasons.append("structure file not found; per-chain values not computed")
    if binder_chain is not None and record.structure_path is not None:
        try:
            pred_atoms = record.atoms()

            # Per-residue binder pLDDT
            if plddt_arr is not None:
                try:
                    per_res = _binder_plddt_per_residue(plddt_arr, pred_atoms, binder_chain)
                    result["binder_plddt_per_residue"] = per_res
                    result["binder_avg_plddt"] = float(per_res.mean()) if per_res.size > 0 else _NAN
                except ValueError as exc:  # pLDDT length differs from the atom count
                    warnings.warn(
                        f"{caller}: per-residue binder pLDDT skipped: {exc}",
                        stacklevel=stacklevel,
                    )
                    reasons.append(f"binder pLDDT: {exc}")
            elif arrays_read:
                reasons.append("binder pLDDT: no per-atom pLDDT in the confidences file")

            if receptor_chain is not None:
                # Interface PDE statistics (binder x receptor token block)
                if record.pde is not None:
                    try:
                        pde_stats = _interface_pde_stats(
                            record.pde,
                            pred_atoms,
                            binder_chain,
                            receptor_chain,
                            token_ranges=_token_ranges(record),
                        )
                        result["mean_interface_pde"] = pde_stats["mean_interface_pde"]
                        result["max_interface_pde"] = pde_stats["max_interface_pde"]
                        if include_matrices:
                            result["pde_interface"] = pde_stats["pde_interface"]
                    except ValueError as exc:  # missing chain or token/residue mismatch
                        warnings.warn(
                            f"{caller}: interface PDE skipped: {exc}", stacklevel=stacklevel
                        )
                        reasons.append(f"interface PDE: {exc}")
                elif arrays_read:
                    reasons.append("interface PDE: no PDE matrix in the confidences file")

                # Interface PAE statistics (binder x receptor token block)
                if record.pae is not None:
                    try:
                        pae_stats = _interface_pae_stats(
                            record.pae,
                            pred_atoms,
                            binder_chain,
                            receptor_chain,
                            token_ranges=_token_ranges(record),
                        )
                        result["mean_interface_pae"] = pae_stats["mean_interface_pae"]
                        result["max_interface_pae"] = pae_stats["max_interface_pae"]
                        if include_matrices:
                            result["pae_interface"] = pae_stats["pae_interface"]
                    except ValueError as exc:  # missing chain or token/residue mismatch
                        warnings.warn(
                            f"{caller}: interface PAE skipped: {exc}", stacklevel=stacklevel
                        )
                        reasons.append(f"interface PAE: {exc}")
                elif arrays_read:
                    reasons.append("interface PAE: no PAE matrix in the confidences file")

            # Binder C-alpha RMSD vs. reference structure
            if reference_structure_path is not None:
                try:
                    ref_atoms = _load_atoms(Path(reference_structure_path))
                    result["binder_ca_rmsd"] = _binder_ca_rmsd(
                        pred_atoms, ref_atoms, binder_chain, receptor_chain
                    )
                except (ValueError, OSError) as exc:  # C-alpha count mismatch or unreadable file
                    warnings.warn(f"{caller}: binder RMSD skipped: {exc}", stacklevel=stacklevel)
                    reasons.append(f"binder RMSD: {exc}")

        except Exception as exc:  # noqa: BLE001 - external model file; recorded in "reason"
            # Broad on purpose: the model file is external input and structure parsers
            # raise several exception types. Values computed so far are kept.
            warnings.warn(f"{caller}: structural analysis failed: {exc}", stacklevel=stacklevel)
            reasons.append(f"structural analysis failed: {type(exc).__name__}: {exc}")

    if reasons:
        result["reason"] = "; ".join(reasons)
    return result
