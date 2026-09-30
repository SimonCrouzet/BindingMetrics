"""Model-agnostic helpers for the confidence outputs of a structure predictor.

Every function takes arrays plus a biotite ``AtomArray`` and chain IDs; none of them knows
which model wrote the arrays. They were written for OpenFold3 and live here so that every
predictor adapter shares them. ``binding_metrics.metrics.openfold`` re-exports the old
private names, so code and tests that patch them there keep working.

What the arrays mean (the rules every adapter converts to)
----------------------------------------------------------
pLDDT
    Per atom, in the atom order of the structure file, on a 0-100 scale. A model that
    reports 0-1, or one value per residue or token, is rescaled and expanded by its adapter.
    A residue value is the mean over the residue's atoms.
PAE
    ``(n_tokens, n_tokens)`` in angstrom. ``pae[i, j]`` is the expected position error of
    token ``j`` when the predicted structure is aligned on token ``i``: the row is the
    alignment frame, the column the scored token (the AlphaFold2 definition, Jumper et al.
    2021, doi:10.1038/s41586-021-03819-2). The pTM code of AlphaFold2, Boltz, Protenix,
    Chai-1 and OpenFold3 sums over the column index of each row and takes the maximum over
    the rows, which fixes the orientation.
PDE
    ``(n_tokens, n_tokens)`` in angstrom, the expected error of the distance between two
    tokens (AlphaFold3 confidence head); no alignment frame is involved.
Tokens
    One token per standard residue in chain order holds for AlphaFold2-style models.
    AlphaFold3-style models tokenise a ligand or a modified residue per heavy atom, so
    residue-count offsets fit a matrix only when the sizes match exactly
    (:func:`_check_token_offsets`); otherwise the interface statistics are refused, never
    guessed.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics.metrics._common import import_biotite, load_structure


def _import_biotite_struc():
    """Lazy import of biotite structure modules."""
    struc, pdbx, _ = import_biotite("per-chain structural analysis")
    return struc, pdbx


def _load_atoms(path: Path):
    """Load an AtomArray from a CIF or PDB file using biotite (model 1)."""
    return load_structure(path, purpose="per-chain structural analysis")


def _residue_index(atoms, mask) -> tuple[np.ndarray, int]:
    """Number the residues of the atoms selected by ``mask``, one index per selected atom.

    A residue is identified by ``(res_id, ins_code)``, so residues 52 and 52A are two.
    Residues are numbered in ascending order of that pair, which for a chain without
    insertion codes is ascending residue number.

    Returns:
        ``(index, n_residues)``: ``index[k]`` is the residue number of the k-th selected atom.
    """
    res_id = np.asarray(atoms.res_id)[mask]
    if "ins_code" in atoms.get_annotation_categories():
        ins_code = np.asarray(atoms.ins_code)[mask]
    else:
        ins_code = np.full(res_id.shape, "", dtype="U1")
    if res_id.size == 0:
        return np.zeros(0, dtype=int), 0
    order = np.lexsort((ins_code, res_id))
    new_residue = np.ones(res_id.size, dtype=bool)
    new_residue[1:] = (res_id[order][1:] != res_id[order][:-1]) | (
        ins_code[order][1:] != ins_code[order][:-1]
    )
    numbered = np.cumsum(new_residue) - 1
    index = np.empty(res_id.size, dtype=int)
    index[order] = numbered
    return index, int(numbered[-1]) + 1


def _chain_token_offsets(atoms) -> dict[str, tuple[int, int]]:
    """Map chain IDs to [start, end) PAE token ranges (one token per residue).

    Preserves the order chains first appear in the structure, which matches
    the PAE matrix token ordering (the order of the chains in the prediction input).
    The one-token-per-residue assumption fails for ligands and modified
    residues; :func:`_check_token_offsets` compares the result with the matrix.

    Residues are counted by ``(res_id, ins_code)``, so an insertion code adds a token.

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
        _, n_res = _residue_index(atoms, atoms.chain_id == chain_id)
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

    Residues are told apart by ``(res_id, ins_code)``, in ascending order of that pair.

    Args:
        plddt_per_atom: Per-atom pLDDT array of the prediction, shape (n_atoms,).
            Must be in the same atom order as ``atoms``.
        atoms: Biotite AtomArray of the predicted model file.
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
    residue, n_residues = _residue_index(atoms, mask)
    return np.array([chain_plddt[residue == r].mean() for r in range(n_residues)], dtype=float)


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
    relative to the receptor. Without a receptor chain the binder Cα atoms are
    superposed on each other, which measures the binder's shape difference
    alone.

    Args:
        pred_atoms: Biotite AtomArray of the predicted structure.
        ref_atoms: Biotite AtomArray of the reference structure.
        binder_chain: Chain ID of the binder.
        receptor_chain: Chain ID of the receptor (used for superposition).
            If None, superpose directly on binder Cα.

    Returns:
        Binder Cα RMSD in Å, or ``nan`` if there are no binder Cα atoms.

    Raises:
        ValueError: If binder Cα counts differ between prediction and reference;
            if the receptor Cα counts differ, or the receptor has fewer than 3
            Cα atoms (a receptor-frame superposition is then not defined); or,
            without a receptor chain, if the binder has fewer than 3 Cα atoms.
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
        n_pred_rec = pred_rec_ca.array_length()
        n_ref_rec = ref_rec_ca.array_length()
        if n_pred_rec != n_ref_rec:
            raise ValueError(
                f"Receptor Cα count mismatch (chain '{receptor_chain}'): "
                f"predicted {n_pred_rec}, reference {n_ref_rec}. "
                "The receptor-frame superposition needs the same residues in both."
            )
        if n_pred_rec < 3:
            raise ValueError(
                f"Receptor chain '{receptor_chain}' has {n_pred_rec} Cα atoms; "
                "at least 3 are needed to superpose on the receptor."
            )
        _, transform = struc.superimpose(ref_rec_ca, pred_rec_ca)
    else:
        if n_pred < 3:
            raise ValueError(
                f"Binder chain '{binder_chain}' has {n_pred} Cα atoms; without a receptor "
                "chain at least 3 are needed to superpose the binder on the reference."
            )
        _, transform = struc.superimpose(ref_binder_ca, pred_binder_ca)
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
    raw slice. PAE at token (i, j) is the expected position error of token j
    when the prediction is aligned on token i (the row is the alignment frame,
    the column the scored token, as in AlphaFold's per-alignment sums). The raw
    binder-rows by receptor-columns slice therefore describes how well the
    receptor tokens are placed when the structure is aligned on the binder; the
    mean and maximum are taken over both blocks and do not depend on the
    orientation.

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
