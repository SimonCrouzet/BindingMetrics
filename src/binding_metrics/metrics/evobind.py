"""EvoBind-style interface distance and adversarial check metrics.

Implements the loss functions from:
    Bryant, P. et al. (2025). EvoBind: In-silico directed evolution of
    peptide binders with AlphaFold2. Communications Chemistry.
    https://doi.org/10.1038/s42004-025-01601-3

Two main metrics:

1. Primary EvoBind score (single structure):
       score = mean_closest_if_dist(peptide→interface) / (mean_pLDDT_peptide / 100)
   Lower is better (closer binding, higher confidence).

2. Adversarial check (two predictions, e.g. OpenFold3 vs AlphaFold-Multimer):
       adversarial_score = mean_if_dist × (100 / pLDDT_pep_afm) × ΔCOM
   where ΔCOM is the peptide centre-of-mass displacement (Å) after receptor
   Cα superposition.  Low score → two models agree on binding pose → reliable.

Usage:
    from binding_metrics.metrics.evobind import (
        compute_evobind_score,
        compute_evobind_adversarial_check,
    )

    # Single-structure primary score
    score = compute_evobind_score(
        "complex.cif",
        plddt_per_atom=metrics["plddt_per_atom"],
        binder_chain="B",
        receptor_chain="A",
    )

    # Adversarial check between two predictions
    check = compute_evobind_adversarial_check(
        design_structure_path="design_pred.cif",
        afm_structure_path="afm_pred.cif",
        binder_chain="B",
        receptor_chain="A",
        afm_plddt_per_atom=afm_metrics["plddt_per_atom"],
    )

    # The same check between two predictions read by the predictor adapters, whatever
    # model made them. Chain IDs are the user's IDs after each record's chain_map.
    from binding_metrics.metrics.evobind import compute_evobind_adversarial_from_records
    from binding_metrics.predictors import get_parser

    of3 = get_parser("of3")
    design = of3.load("design_scoring_out", "cmplx_007")
    adversary = of3.load("sequence_only_out", "cmplx_007")
    check = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from binding_metrics.core.residues import VARIANT_TO_PARENT_RESIDUE
from binding_metrics.metrics._common import import_biotite, load_structure, resolve_chain_role
from binding_metrics.predictors._confidence import _binder_plddt_per_residue, _residue_index
from binding_metrics.predictors.record import PredictionRecord

# ---------------------------------------------------------------------------
# Internal helpers: coordinate extraction
# ---------------------------------------------------------------------------


def _import_biotite():
    struc, pdbx, _ = import_biotite("EvoBind metrics")
    return struc, pdbx


def _load_atoms(path: Path):
    """Load an AtomArray from a CIF or PDB file (model 1)."""
    return load_structure(path, purpose="EvoBind metrics")


def _cb_atoms(atoms, chain_id: str):
    """Return a biotite AtomArray of Cβ atoms for each residue in a chain.

    Glycine and any residue lacking CB fall back to Cα. A residue is identified by
    residue number and insertion code, so residues 52 and 52A are two, and the result
    lists them in ascending order of that pair.
    """
    chain_mask = atoms.chain_id == chain_id
    all_indices = np.where(chain_mask)[0]
    residue, n_residues = _residue_index(atoms, chain_mask)
    chain_atom_names = atoms.atom_name[all_indices]

    selected_indices = []
    for r in range(n_residues):
        res_mask = residue == r
        res_names = chain_atom_names[res_mask]
        res_global_idx = all_indices[res_mask]

        cb_sel = res_global_idx[res_names == "CB"]
        if cb_sel.size > 0:
            selected_indices.append(cb_sel[0])
        else:
            ca_sel = res_global_idx[res_names == "CA"]
            if ca_sel.size > 0:
                selected_indices.append(ca_sel[0])

    if not selected_indices:
        return atoms[:0]  # empty AtomArray
    return atoms[np.array(selected_indices)]


def _ca_atoms(atoms, chain_id: str):
    """Return a biotite AtomArray of Cα atoms for a chain."""
    return atoms[(atoms.chain_id == chain_id) & (atoms.atom_name == "CA")]


# ---------------------------------------------------------------------------
# Internal helpers: distance calculations
# ---------------------------------------------------------------------------


def _pairwise_min_dists(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """For each point in A, the minimum Euclidean distance to any point in B.

    Args:
        coords_a: shape (N, 3)
        coords_b: shape (M, 3)

    Returns:
        Array of shape (N,) with min distance from each point in A to B.
    """
    # (N, 1, 3) - (1, M, 3) → (N, M, 3) → (N, M)
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    return dists.min(axis=1)


def _auto_interface_mask(
    rec_cb_coords: np.ndarray,
    pep_cb_coords: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Boolean mask over receptor Cβ positions within ``cutoff`` Å of any peptide Cβ."""
    diff = rec_cb_coords[:, np.newaxis, :] - pep_cb_coords[np.newaxis, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    return dists.min(axis=1) < cutoff


def _resname_mismatch_fraction(atoms_a, atoms_b) -> float:
    """Fraction of paired atoms whose residue names differ (pair k = atom k of each array).

    Protonation-state variants of one amino acid (HID/HIE/HIP, CYX, ASH, ...)
    count as the same residue. Returns 0.0 for empty input.
    """
    if atoms_a.array_length() == 0:
        return 0.0

    def _canonical(atoms):
        return [
            VARIANT_TO_PARENT_RESIDUE.get(str(r).strip(), str(r).strip()) for r in atoms.res_name
        ]

    return float(np.mean([a != b for a, b in zip(_canonical(atoms_a), _canonical(atoms_b))]))


# The per-residue mean lives with the other model-agnostic confidence helpers. It is
# kept under its old name here because callers and tests import it from this module.
_per_residue_plddt = _binder_plddt_per_residue


def _require_chains(atoms, chains: dict[str, str], which: str, note: str) -> None:
    """Raise ``ValueError`` if a role's chain ID is not a chain of ``atoms``.

    ``chains`` maps a role (``"binder"``, ``"receptor"``) to the chain ID asked for; the
    message lists the chains that exist, since a chain ID of the model's own file instead
    of the user's is the usual mistake.
    """
    present = sorted({str(chain) for chain in atoms.chain_id})
    for role, chain in chains.items():
        if chain not in present:
            raise ValueError(
                f"{role} chain '{chain}' is not in the {which} structure ({note}): it has {present}"
            )


def _residue_keys(atoms) -> list[tuple[int, str]]:
    """``(res_id, ins_code)`` of each atom: the identity of its residue."""
    if "ins_code" in atoms.get_annotation_categories():
        ins_code = atoms.ins_code
    else:
        ins_code = [""] * atoms.array_length()
    return [(int(res_id), str(ins)) for res_id, ins in zip(atoms.res_id, ins_code)]


def _pair_by_residue_id(design_ca, adversary_ca):
    """Cα atoms of the residues that both structures have, by ``(res_id, ins_code)``.

    Returns ``(design_matched, adversary_matched, n_shared_residues)``; each array keeps the
    atom order of its structure. The two arrays differ in length when a residue number
    repeats within a chain of one structure; see :func:`_require_same_count`.
    """
    design_keys = _residue_keys(design_ca)
    adversary_keys = _residue_keys(adversary_ca)
    shared = set(design_keys) & set(adversary_keys)
    design_matched = design_ca[np.array([key in shared for key in design_keys], dtype=bool)]
    adversary_matched = adversary_ca[
        np.array([key in shared for key in adversary_keys], dtype=bool)
    ]
    return design_matched, adversary_matched, len(shared)


def _require_same_count(
    design_matched, adversary_matched, n_shared: int, region: str, adversary_label: str
) -> None:
    """Raise ``ValueError`` when residue-ID pairing gave arrays of different length."""
    n_design = design_matched.array_length()
    n_adversary = adversary_matched.array_length()
    if n_design != n_adversary:
        raise ValueError(
            f"The {n_shared} {region} residues that the design and the {adversary_label} "
            f"structure share (by residue number and insertion code) select {n_design} Cα "
            f"atoms in the design and {n_adversary} in the {adversary_label} structure, so "
            "they cannot be paired; a residue number and insertion code probably occur twice "
            "in a chain."
        )


# ---------------------------------------------------------------------------
# Core computations: structures already loaded, chain IDs already the caller's
# ---------------------------------------------------------------------------


def _score_from_atoms(
    atoms,
    plddt_per_atom: Optional[np.ndarray],
    binder_chain: str,
    receptor_chain: str,
    receptor_interface_residues: Optional[list[int]],
    interface_cutoff_angstrom: float,
) -> dict:
    """Primary EvoBind score of a structure that is already loaded.

    ``atoms`` is the biotite AtomArray with the chain IDs of the caller and
    ``plddt_per_atom`` is None or aligned with ``atoms``. The arguments and the
    keys of the result are those of :func:`compute_evobind_score`, which loads the
    file and calls this function.
    """
    pep_cb = _cb_atoms(atoms, binder_chain)
    rec_cb = _cb_atoms(atoms, receptor_chain)

    if pep_cb.array_length() == 0:
        raise ValueError(f"No Cβ/Cα atoms found for binder chain '{binder_chain}'")
    if rec_cb.array_length() == 0:
        raise ValueError(f"No Cβ/Cα atoms found for receptor chain '{receptor_chain}'")

    pep_cb_coords = pep_cb.coord
    rec_cb_coords = rec_cb.coord
    rec_res_ids = rec_cb.res_id  # one per residue by construction of _cb_atoms

    # Determine receptor interface residues
    interface_fallback_used = False
    if receptor_interface_residues is not None:
        if_mask = np.isin(rec_res_ids, receptor_interface_residues)
        if not if_mask.any():
            raise ValueError(
                f"None of receptor_interface_residues {list(receptor_interface_residues)} "
                f"is a residue number of receptor chain '{receptor_chain}'."
            )
    else:
        if_mask = _auto_interface_mask(rec_cb_coords, pep_cb_coords, interface_cutoff_angstrom)
        if not if_mask.any():
            # Nothing within the cutoff: the whole receptor stands in for the
            # interface, which turns the score into a binder-to-receptor distance.
            if_mask = np.ones(len(rec_res_ids), dtype=bool)
            interface_fallback_used = True

    rec_if_coords = rec_cb_coords[if_mask]

    # Asymmetric mean-closest distances
    pep_min_dists = _pairwise_min_dists(pep_cb_coords, rec_if_coords)
    rec_min_dists = _pairwise_min_dists(rec_if_coords, pep_cb_coords)

    if_dist_pep_to_rec = float(pep_min_dists.mean())
    if_dist_rec_to_pep = float(rec_min_dists.mean())
    if_dist_symmetric = (if_dist_pep_to_rec + if_dist_rec_to_pep) / 2.0

    result: dict = {
        "if_dist_pep_to_rec": if_dist_pep_to_rec,
        "if_dist_rec_to_pep": if_dist_rec_to_pep,
        "if_dist_symmetric": if_dist_symmetric,
        "n_interface_receptor_residues": int(if_mask.sum()),
        "interface_fallback_used": interface_fallback_used,
        "mean_plddt_binder": None,
        "evobind_score": None,
    }

    if plddt_per_atom is not None:
        per_res = _binder_plddt_per_residue(plddt_per_atom, atoms, binder_chain)
        mean_plddt = float(per_res.mean()) if per_res.size > 0 else float("nan")
        result["mean_plddt_binder"] = mean_plddt
        if mean_plddt > 0 and np.isfinite(mean_plddt):
            result["evobind_score"] = if_dist_pep_to_rec / (mean_plddt / 100.0)
        else:
            result["reason"] = "mean binder pLDDT is zero or not finite"

    return result


def _adversarial_from_atoms(
    design_atoms,
    adversary_atoms,
    adversary_plddt: Optional[np.ndarray],
    binder_chain: str,
    receptor_chain: str,
    interface_cutoff_angstrom: float,
    max_resname_mismatch_fraction: float,
    *,
    adversary_label: str = "AFM",
) -> dict:
    """Adversarial check between two structures that are already loaded.

    Both AtomArrays carry the chain IDs of the caller and ``adversary_plddt`` is
    None or aligned with ``adversary_atoms``. The arguments and the keys of the
    result are those of :func:`compute_evobind_adversarial_check`, which loads the
    files and calls this function; the ``afm_`` keys describe the second structure
    whatever model made it. ``adversary_label`` is how error messages and the
    ``reason`` name that second structure.
    """
    struc, _ = _import_biotite()

    # --- Receptor Cα superposition: place design into the AFM coordinate frame ---
    # Match by residue number and insertion code — the two structures may cover different
    # sequence extents (e.g. a cropped design vs a full-length OF3 prediction).
    design_rec_ca = _ca_atoms(design_atoms, receptor_chain)
    afm_rec_ca = _ca_atoms(adversary_atoms, receptor_chain)

    design_rec_ca_matched, afm_rec_ca_matched, n_shared_rec = _pair_by_residue_id(
        design_rec_ca, afm_rec_ca
    )
    if n_shared_rec >= 3:
        receptor_pairing = "residue_number"
        _require_same_count(
            design_rec_ca_matched, afm_rec_ca_matched, n_shared_rec, "receptor", adversary_label
        )
    else:
        # Fall back to positional pairing (e.g. OF3 renumbered from 1)
        receptor_pairing = "position"
        n = min(design_rec_ca.array_length(), afm_rec_ca.array_length())
        if n < 3:
            raise ValueError(
                f"Fewer than 3 receptor Cα atoms for superposition (design "
                f"{design_rec_ca.array_length()}, {adversary_label} {afm_rec_ca.array_length()})."
            )
        design_rec_ca_matched = design_rec_ca[:n]
        afm_rec_ca_matched = afm_rec_ca[:n]

    # Numbers alone can pair unrelated residues (a design numbered 100-200
    # against a prediction renumbered from 1); residue names catch that.
    receptor_mismatch = _resname_mismatch_fraction(design_rec_ca_matched, afm_rec_ca_matched)
    if receptor_mismatch > max_resname_mismatch_fraction:
        raise ValueError(
            f"{receptor_mismatch:.0%} of the receptor residues paired for superposition "
            f"(by {receptor_pairing}) have different residue names in the two structures "
            f"(limit {max_resname_mismatch_fraction:.0%}); the residue numbering or the "
            "chain contents probably do not correspond."
        )

    # superimpose(reference, mobile) → (superimposed_mobile, transform)
    _, transform = struc.superimpose(afm_rec_ca_matched, design_rec_ca_matched)

    design_pep_ca = _ca_atoms(design_atoms, binder_chain)
    afm_pep_ca = _ca_atoms(adversary_atoms, binder_chain)

    if design_pep_ca.array_length() == 0:
        raise ValueError(
            f"No Cα atoms found for binder chain '{binder_chain}' in design structure."
        )
    if afm_pep_ca.array_length() == 0:
        raise ValueError(
            f"No Cα atoms found for binder chain '{binder_chain}' in {adversary_label} structure."
        )

    # For CoM: match binder residues by number; fall back to positional
    # pairing when numbering schemes differ (e.g. OF3 renumbers from 1).
    design_pep_ca_matched, afm_pep_ca_matched, n_shared_pep = _pair_by_residue_id(
        design_pep_ca, afm_pep_ca
    )
    if n_shared_pep >= 1:
        binder_pairing = "residue_number"
        _require_same_count(
            design_pep_ca_matched, afm_pep_ca_matched, n_shared_pep, "binder", adversary_label
        )
    else:
        # No residue-number overlap — pair positionally up to the shorter length
        binder_pairing = "position"
        n = min(design_pep_ca.array_length(), afm_pep_ca.array_length())
        if n == 0:
            raise ValueError(
                f"No binder Cα atoms for chain '{binder_chain}' in one of the structures."
            )
        design_pep_ca_matched = design_pep_ca[:n]
        afm_pep_ca_matched = afm_pep_ca[:n]

    binder_mismatch = _resname_mismatch_fraction(design_pep_ca_matched, afm_pep_ca_matched)
    if binder_mismatch > max_resname_mismatch_fraction:
        raise ValueError(
            f"{binder_mismatch:.0%} of the binder residues paired for the centre of mass "
            f"(by {binder_pairing}) have different residue names in the two structures "
            f"(limit {max_resname_mismatch_fraction:.0%}); the residue numbering or the "
            "chain contents probably do not correspond."
        )

    # Apply receptor superposition transform to design binder Cα
    design_pep_ca_in_afm_frame = transform.apply(design_pep_ca_matched).coord

    # Centre-of-mass displacement
    design_com = design_pep_ca_in_afm_frame.mean(axis=0)
    afm_com = afm_pep_ca_matched.coord.mean(axis=0)
    delta_com = float(np.linalg.norm(design_com - afm_com))

    # --- Interface residues from the design structure ---
    design_pep_cb = _cb_atoms(design_atoms, binder_chain)
    design_rec_cb = _cb_atoms(design_atoms, receptor_chain)

    if design_pep_cb.array_length() == 0:
        raise ValueError(
            f"No Cβ/Cα atoms found for binder chain '{binder_chain}' in design structure."
        )
    if design_rec_cb.array_length() == 0:
        raise ValueError(
            f"No Cβ/Cα atoms found for receptor chain '{receptor_chain}' in design structure."
        )

    interface_fallback_used = False
    if_mask = _auto_interface_mask(
        design_rec_cb.coord, design_pep_cb.coord, interface_cutoff_angstrom
    )
    if not if_mask.any():
        if_mask = np.ones(design_rec_cb.array_length(), dtype=bool)
        interface_fallback_used = True
    if_keys = set(_residue_keys(design_rec_cb[if_mask]))

    # Map those interface residues (number and insertion code) to the AFM structure
    afm_rec_cb = _cb_atoms(adversary_atoms, receptor_chain)
    if afm_rec_cb.array_length() == 0:
        raise ValueError(
            f"No Cβ/Cα atoms found for receptor chain '{receptor_chain}' "
            f"in {adversary_label} structure."
        )

    afm_if_mask = np.array([key in if_keys for key in _residue_keys(afm_rec_cb)], dtype=bool)
    if not afm_if_mask.any():
        # None of the interface residues exists in the AFM model.
        afm_if_mask = np.ones(afm_rec_cb.array_length(), dtype=bool)
        interface_fallback_used = True
    afm_rec_if_coords = afm_rec_cb.coord[afm_if_mask]

    afm_pep_cb = _cb_atoms(adversary_atoms, binder_chain)
    if afm_pep_cb.array_length() == 0:
        raise ValueError(
            f"No Cβ/Cα atoms found for binder chain '{binder_chain}' "
            f"in {adversary_label} structure."
        )
    afm_pep_cb_coords = afm_pep_cb.coord

    # AFM interface distances
    afm_pep_min_dists = _pairwise_min_dists(afm_pep_cb_coords, afm_rec_if_coords)
    afm_rec_min_dists = _pairwise_min_dists(afm_rec_if_coords, afm_pep_cb_coords)

    afm_if_dist_pep_to_rec = float(afm_pep_min_dists.mean())
    afm_if_dist_rec_to_pep = float(afm_rec_min_dists.mean())
    afm_mean_if_dist = (afm_if_dist_pep_to_rec + afm_if_dist_rec_to_pep) / 2.0

    result: dict = {
        "delta_com_angstrom": delta_com,
        "n_superposition_residues": int(n_shared_rec),
        "n_superposition_atoms": int(design_rec_ca_matched.array_length()),
        "receptor_pairing": receptor_pairing,
        "binder_pairing": binder_pairing,
        "receptor_resname_mismatch_fraction": receptor_mismatch,
        "binder_resname_mismatch_fraction": binder_mismatch,
        "afm_if_dist_pep_to_rec": afm_if_dist_pep_to_rec,
        "afm_if_dist_rec_to_pep": afm_if_dist_rec_to_pep,
        "afm_mean_if_dist": afm_mean_if_dist,
        "interface_fallback_used": interface_fallback_used,
        "afm_mean_plddt_binder": None,
        "evobind_adversarial_score": None,
    }

    if adversary_plddt is not None:
        per_res = _binder_plddt_per_residue(adversary_plddt, adversary_atoms, binder_chain)
        afm_mean_plddt = float(per_res.mean()) if per_res.size > 0 else float("nan")
        result["afm_mean_plddt_binder"] = afm_mean_plddt
        if afm_mean_plddt > 0 and np.isfinite(afm_mean_plddt):
            result["evobind_adversarial_score"] = (
                afm_mean_if_dist * (100.0 / afm_mean_plddt) * delta_com
            )
        else:
            result["reason"] = (
                f"mean binder pLDDT in the {adversary_label} model is zero or not finite"
            )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_evobind_score(
    structure_path: str | Path,
    plddt_per_atom: Optional[np.ndarray],
    binder_chain: str,
    receptor_chain: Optional[str] = None,
    receptor_interface_residues: Optional[list[int]] = None,
    interface_cutoff_angstrom: float = 8.0,
    *,
    target_chain: Optional[str] = None,
) -> dict:
    """Compute the primary EvoBind design score for a single predicted structure.

    The primary EvoBind loss is:
        score = if_dist_pep / (mean_pLDDT_pep / 100)

    where ``if_dist_pep`` is the mean over binder Cβ atoms of the minimum
    distance to the receptor interface Cβ atoms (Å), and ``mean_pLDDT_pep``
    is the mean per-residue pLDDT of the binder chain [0–100].

    Dividing by pLDDT/100 (dimensionless confidence) keeps the score in Å
    and penalises low-confidence predictions.  Lower score → better.

    Args:
        structure_path: Path to the predicted CIF or PDB file.
        plddt_per_atom: Per-atom pLDDT [0–100], shape (n_atoms,), from the
            same prediction. If None only distance metrics are returned.
        binder_chain: Chain ID of the designed binder/peptide.
        receptor_chain: Chain ID of the receptor/target. Required, through
            this parameter or ``target_chain``.
        receptor_interface_residues: Explicit list of receptor residue numbers
            that define the binding interface. If None the interface is
            auto-detected as receptor residues whose Cβ lies within
            ``interface_cutoff_angstrom`` Å of any binder Cβ.
        interface_cutoff_angstrom: Distance cutoff (Å) for auto-detection
            (default 8.0, matching the EvoBind CB-contact threshold).
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dictionary with keys:

        if_dist_pep_to_rec (float):
            Mean closest-approach distance from each binder Cβ to the
            nearest receptor interface Cβ (Å).  This is the ``if_dist``
            term in EvoBind's primary loss.
        if_dist_rec_to_pep (float):
            Mean closest-approach distance from each receptor interface Cβ
            to the nearest binder Cβ (Å).
        if_dist_symmetric (float):
            Average of the two asymmetric distances above.
        n_interface_receptor_residues (int):
            Number of receptor residues used as the interface.
        interface_fallback_used (bool):
            True when no receptor Cβ lay within ``interface_cutoff_angstrom`` of
            a binder Cβ and the whole receptor was used as the interface. The
            distances then describe the binder against the entire receptor.
        mean_plddt_binder (float | None):
            Mean per-residue pLDDT of the binder chain [0–100].
            None when ``plddt_per_atom`` is not provided.
        evobind_score (float | None):
            Primary EvoBind score: ``if_dist_pep_to_rec / (mean_plddt / 100)``.
            None when ``plddt_per_atom`` is not provided or mean_plddt is zero.
        reason (str):
            Present only when ``plddt_per_atom`` was given but the score could
            not be computed (mean pLDDT zero or not finite).

    Raises:
        ValueError: If the binder or receptor chain has no Cβ/Cα atoms, or
            ``receptor_interface_residues`` matches no receptor residue.

    Reference:
        Bryant et al. 2025, Commun. Chem. (doi:10.1038/s42004-025-01601-3).
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    atoms = _load_atoms(Path(structure_path))
    return _score_from_atoms(
        atoms,
        plddt_per_atom,
        binder_chain,
        receptor_chain,
        receptor_interface_residues,
        interface_cutoff_angstrom,
    )


def compute_evobind_adversarial_check(
    design_structure_path: str | Path,
    afm_structure_path: str | Path,
    binder_chain: str,
    receptor_chain: Optional[str] = None,
    afm_plddt_per_atom: Optional[np.ndarray] = None,
    interface_cutoff_angstrom: float = 8.0,
    max_resname_mismatch_fraction: float = 0.5,
    *,
    target_chain: Optional[str] = None,
) -> dict:
    """EvoBind adversarial check: consistency between two structure predictions.

    Compares a design-model prediction (e.g., OpenFold3 in design-scoring mode)
    against an independent multimer prediction (e.g., AlphaFold-Multimer or
    a second OpenFold3 run).  Agreement between the two indicates the predicted
    binding pose is robust and not a model artefact.

    The combined adversarial score from EvoBind is:
        adversarial_score = mean_if_dist × (100 / pLDDT_pep_afm) × ΔCOM

    where ``mean_if_dist`` is the symmetric interface distance in the AFM
    prediction, ``pLDDT_pep_afm`` is the binder mean pLDDT from the AFM
    model, and ``ΔCOM`` is the Euclidean distance (Å) between the peptide
    centres of mass after superposing the receptor Cα atoms.

    Low adversarial score → consistent predictions → reliable design.
    High ΔCOM with high mean_if_dist → the second model places the peptide
    far from the interface, indicating a hallucinated binding pose.

    Args:
        design_structure_path: Path to the first prediction (design/reference),
            e.g. the AF2-monomer EvoBind result or OpenFold3 design output.
        afm_structure_path: Path to the second (adversarial) prediction, e.g.
            AlphaFold-Multimer or OpenFold3 standard complex scoring.
        binder_chain: Chain ID of the binder in **both** structures.
        receptor_chain: Chain ID of the receptor in **both** structures.
            Required, through this parameter or ``target_chain``.
        afm_plddt_per_atom: Per-atom pLDDT [0–100] from the AFM prediction,
            shape (n_atoms,). Used to compute the pLDDT-weighted adversarial
            score.  If None, only geometric metrics are returned.
        interface_cutoff_angstrom: Distance cutoff (Å) used to identify
            receptor interface residues from the design structure (default 8.0).
        max_resname_mismatch_fraction: Residues are paired between the two
            structures by residue number and insertion code, or by position when
            the numberings do not overlap. If more than this fraction of the paired receptor (or
            binder) residues have different residue names (histidine and
            cysteine protonation variants count as equal), the pairing is
            rejected with a ValueError (default 0.5).
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        Dictionary with keys:

        delta_com_angstrom (float):
            Peptide CoM displacement (Å) between the two predictions after
            receptor Cα superposition.  Large values indicate disagreement.
        n_superposition_residues (int):
            Receptor residues matched by residue number and insertion code. 0 to 2 when the
            structures share no numbering and positional pairing was used
            instead; see ``n_superposition_atoms`` for the real count.
        n_superposition_atoms (int):
            Receptor Cα atoms used for the superposition, whichever pairing
            was applied.
        receptor_pairing, binder_pairing (str):
            ``"residue_number"`` or ``"position"``: how residues were paired
            for the superposition and for the binder centre of mass.
        receptor_resname_mismatch_fraction, binder_resname_mismatch_fraction (float):
            Fraction of paired residues with different residue names.
        afm_if_dist_pep_to_rec (float):
            Mean closest-approach distance from binder Cβ to receptor
            interface Cβ in the AFM structure (Å).
        afm_if_dist_rec_to_pep (float):
            Mean closest-approach distance from receptor interface Cβ to
            binder Cβ in the AFM structure (Å).
        afm_mean_if_dist (float):
            Symmetric interface distance in the AFM prediction (Å).
        interface_fallback_used (bool):
            True when the design structure has no receptor Cβ within
            ``interface_cutoff_angstrom`` of a binder Cβ, or none of its
            interface residue numbers exist in the AFM structure; the whole
            receptor was then used as the interface.
        afm_mean_plddt_binder (float | None):
            Mean per-residue pLDDT of the binder in the AFM prediction.
            None if ``afm_plddt_per_atom`` is not provided.
        evobind_adversarial_score (float | None):
            Combined adversarial score.  None if ``afm_plddt_per_atom`` is
            not provided or the mean pLDDT is zero.
        reason (str):
            Present only when ``afm_plddt_per_atom`` was given but the score
            could not be computed (mean pLDDT zero or not finite).

    Raises:
        ValueError: If a chain has no Cα/Cβ atoms in either structure, fewer
            than three receptor Cα atoms can be paired, a residue number and
            insertion code that occurs twice in a chain makes the paired Cα arrays
            differ in length, or the residue names of the paired residues disagree
            beyond ``max_resname_mismatch_fraction``.

    Reference:
        Bryant et al. 2025, Commun. Chem. (doi:10.1038/s42004-025-01601-3).
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    design_atoms = _load_atoms(Path(design_structure_path))
    afm_atoms = _load_atoms(Path(afm_structure_path))
    return _adversarial_from_atoms(
        design_atoms,
        afm_atoms,
        afm_plddt_per_atom,
        binder_chain,
        receptor_chain,
        interface_cutoff_angstrom,
        max_resname_mismatch_fraction,
    )


def compute_evobind_adversarial_from_records(
    design: Union[PredictionRecord, str, Path],
    adversary: PredictionRecord,
    binder_chain: str,
    receptor_chain: Optional[str] = None,
    *,
    interface_cutoff_angstrom: float = 8.0,
    max_resname_mismatch_fraction: float = 0.5,
    target_chain: Optional[str] = None,
) -> dict:
    """EvoBind adversarial check between two predictions read by the predictor adapters.

    Same computation as :func:`compute_evobind_adversarial_check`, but the two structures
    come from ``PredictionRecord`` objects, so the second prediction can be made by any
    model with an adapter and its per-atom pLDDT travels with it. Chain IDs are the
    USER's IDs: each record renames the chains of its file through its ``chain_map``
    (``get_parser(model).load(directory, name, chain_map={"B": "P"})``), so a receptor
    that one model calls ``A`` and another ``B`` needs no argument here.

    The score divides by the adversary's pLDDT, and pLDDT is calibrated per model, so
    compare ``evobind_adversarial_score`` between designs only when the same adversary
    model made the second prediction. When the adversary was run in OpenFold3 score mode
    it was templated on the design pose, so agreement with the design is partly by
    construction; an independent adversary is a sequence-only prediction.

    Args:
        design: The first prediction, or the input pose. A ``PredictionRecord`` (its
            ``chain_map`` applies) or the path of a structure file (chain IDs as in
            the file).
        adversary: The second prediction, a ``PredictionRecord``. Its
            ``plddt_per_atom`` (0 to 100, in the atom order of its structure file)
            gives the pLDDT-weighted score.
        binder_chain: Chain ID of the binder in both structures, after ``chain_map``.
        receptor_chain: Chain ID of the receptor in both structures, after
            ``chain_map``. Required, through this parameter or ``target_chain``.
        interface_cutoff_angstrom: Distance cutoff (Å) that picks the receptor interface
            residues from the design structure (default 8.0).
        max_resname_mismatch_fraction: Largest accepted fraction of paired residues with
            different residue names (default 0.5); see
            :func:`compute_evobind_adversarial_check`.
        target_chain: Alias of ``receptor_chain``; different IDs in both raise
            ``ValueError``.

    Returns:
        The dictionary of :func:`compute_evobind_adversarial_check`, whose keys keep their
        ``afm_`` prefix and here describe the adversary whatever model it is, plus:

        design_model (str | None):
            ``design.model`` for a record, None when ``design`` is a path.
        adversary_model (str):
            ``adversary.model``.
        reason (str):
            Present when the score could not be computed: the adversary has no per-atom
            pLDDT (the geometric keys are still returned, ``evobind_adversarial_score``
            and ``afm_mean_plddt_binder`` are None; the adapter's own reasons are
            appended), or its mean binder pLDDT is zero or not finite.

    Raises:
        TypeError: If ``adversary`` is not a ``PredictionRecord``, or ``design`` is neither
            a record nor a path.
        ValueError: If a record has no structure file, a chain is not in a structure, the
            pLDDT array does not have one value per atom of the adversary structure, or
            for any of the reasons listed for :func:`compute_evobind_adversarial_check`.

    Reference:
        Bryant et al. 2025, Commun. Chem. (doi:10.1038/s42004-025-01601-3).
    """
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain, required=True
    )
    if not isinstance(adversary, PredictionRecord):
        raise TypeError(
            "adversary must be a PredictionRecord (get_parser(model).load(...)); "
            f"got {type(adversary).__name__}. For two structure files use "
            "compute_evobind_adversarial_check."
        )
    if isinstance(design, PredictionRecord):
        design_atoms = design.atoms()
        design_model: Optional[str] = design.model
        design_note = f"chains after the chain_map of the {design.model} record"
    elif isinstance(design, (str, os.PathLike)):
        design_atoms = _load_atoms(Path(design))
        design_model = None
        design_note = "chain IDs as in the file"
    else:
        raise TypeError(
            f"design must be a PredictionRecord or a structure path, got {type(design).__name__}"
        )
    adversary_atoms = adversary.atoms()

    chains = {"binder": binder_chain, "receptor": receptor_chain}
    _require_chains(design_atoms, chains, "design", design_note)
    _require_chains(
        adversary_atoms,
        chains,
        "adversary",
        f"chains after the chain_map of the {adversary.model} record",
    )

    plddt = adversary.plddt_per_atom
    result = _adversarial_from_atoms(
        design_atoms,
        adversary_atoms,
        plddt,
        binder_chain,
        receptor_chain,
        interface_cutoff_angstrom,
        max_resname_mismatch_fraction,
        adversary_label="adversary",
    )
    if plddt is None:
        reason = "adversary has no per-atom pLDDT"
        if adversary.reasons:
            reason += ": " + "; ".join(adversary.reasons)
        result["reason"] = reason
    return {"design_model": design_model, "adversary_model": adversary.model, **result}
