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
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers: coordinate extraction
# ---------------------------------------------------------------------------


def _import_biotite():
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        return struc, pdbx
    except ImportError:
        raise ImportError(
            "biotite is required for EvoBind metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _load_atoms(path: Path):
    """Load an AtomArray from a CIF or PDB file (model 1)."""
    struc, pdbx = _import_biotite()
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(f, model=1)
    import biotite.structure.io.pdb as pdb_io
    f = pdb_io.PDBFile.read(str(path))
    return pdb_io.get_structure(f, model=1)


def _cb_atoms(atoms, chain_id: str):
    """Return a biotite AtomArray of Cβ atoms for each residue in a chain.

    Glycine and any residue lacking CB fall back to Cα.
    """
    chain_atoms = atoms[atoms.chain_id == chain_id]
    unique_res_ids = np.unique(chain_atoms.res_id)

    selected_indices = []
    all_indices = np.where(atoms.chain_id == chain_id)[0]
    chain_res_ids = atoms.res_id[all_indices]
    chain_atom_names = atoms.atom_name[all_indices]

    for rid in unique_res_ids:
        res_mask = chain_res_ids == rid
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
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    return dists.min(axis=1)


def _auto_interface_mask(
    rec_cb_coords: np.ndarray,
    pep_cb_coords: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Boolean mask over receptor Cβ positions within ``cutoff`` Å of any peptide Cβ."""
    diff = rec_cb_coords[:, np.newaxis, :] - pep_cb_coords[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    return dists.min(axis=1) < cutoff


def _per_residue_plddt(
    plddt_per_atom: np.ndarray,
    atoms,
    chain_id: str,
) -> np.ndarray:
    """Mean pLDDT per residue for a chain.

    Args:
        plddt_per_atom: Per-atom pLDDT [0–100], shape (n_atoms,). Must be
            aligned with ``atoms``.
        atoms: Biotite AtomArray from the same prediction.
        chain_id: Chain to extract.

    Returns:
        Array of shape (n_residues,).
    """
    mask = atoms.chain_id == chain_id
    chain_plddt = plddt_per_atom[mask]
    chain_res_ids = atoms.res_id[mask]
    unique_res = np.unique(chain_res_ids)
    return np.array(
        [chain_plddt[chain_res_ids == r].mean() for r in unique_res], dtype=float
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_evobind_score(
    structure_path: str | Path,
    plddt_per_atom: Optional[np.ndarray],
    binder_chain: str,
    receptor_chain: str,
    receptor_interface_residues: Optional[list[int]] = None,
    interface_cutoff_angstrom: float = 8.0,
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
        receptor_chain: Chain ID of the receptor/target.
        receptor_interface_residues: Explicit list of receptor residue numbers
            that define the binding interface. If None the interface is
            auto-detected as receptor residues whose Cβ lies within
            ``interface_cutoff_angstrom`` Å of any binder Cβ.
        interface_cutoff_angstrom: Distance cutoff (Å) for auto-detection
            (default 8.0, matching the EvoBind CB-contact threshold).

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
        mean_plddt_binder (float | None):
            Mean per-residue pLDDT of the binder chain [0–100].
            None when ``plddt_per_atom`` is not provided.
        evobind_score (float | None):
            Primary EvoBind score: ``if_dist_pep_to_rec / (mean_plddt / 100)``.
            None when ``plddt_per_atom`` is not provided or mean_plddt is zero.
    """
    atoms = _load_atoms(Path(structure_path))

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
    if receptor_interface_residues is not None:
        if_mask = np.isin(rec_res_ids, receptor_interface_residues)
    else:
        if_mask = _auto_interface_mask(rec_cb_coords, pep_cb_coords, interface_cutoff_angstrom)
        if not if_mask.any():
            # No residues within cutoff — use the full receptor as fallback
            if_mask = np.ones(len(rec_res_ids), dtype=bool)

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
        "mean_plddt_binder": None,
        "evobind_score": None,
    }

    if plddt_per_atom is not None:
        per_res = _per_residue_plddt(plddt_per_atom, atoms, binder_chain)
        mean_plddt = float(per_res.mean()) if per_res.size > 0 else float("nan")
        result["mean_plddt_binder"] = mean_plddt
        if mean_plddt > 0 and np.isfinite(mean_plddt):
            result["evobind_score"] = if_dist_pep_to_rec / (mean_plddt / 100.0)

    return result


def compute_evobind_adversarial_check(
    design_structure_path: str | Path,
    afm_structure_path: str | Path,
    binder_chain: str,
    receptor_chain: str,
    afm_plddt_per_atom: Optional[np.ndarray] = None,
    interface_cutoff_angstrom: float = 8.0,
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
        afm_plddt_per_atom: Per-atom pLDDT [0–100] from the AFM prediction,
            shape (n_atoms,). Used to compute the pLDDT-weighted adversarial
            score.  If None, only geometric metrics are returned.
        interface_cutoff_angstrom: Distance cutoff (Å) used to identify
            receptor interface residues from the design structure (default 8.0).

    Returns:
        Dictionary with keys:

        delta_com_angstrom (float):
            Peptide CoM displacement (Å) between the two predictions after
            receptor Cα superposition.  Large values indicate disagreement.
        afm_if_dist_pep_to_rec (float):
            Mean closest-approach distance from binder Cβ to receptor
            interface Cβ in the AFM structure (Å).
        afm_if_dist_rec_to_pep (float):
            Mean closest-approach distance from receptor interface Cβ to
            binder Cβ in the AFM structure (Å).
        afm_mean_if_dist (float):
            Symmetric interface distance in the AFM prediction (Å).
        afm_mean_plddt_binder (float | None):
            Mean per-residue pLDDT of the binder in the AFM prediction.
            None if ``afm_plddt_per_atom`` is not provided.
        evobind_adversarial_score (float | None):
            Combined adversarial score.  None if ``afm_plddt_per_atom`` is
            not provided or the mean pLDDT is zero.
    """
    struc, _ = _import_biotite()

    design_atoms = _load_atoms(Path(design_structure_path))
    afm_atoms = _load_atoms(Path(afm_structure_path))

    # --- Receptor Cα superposition: place design into the AFM coordinate frame ---
    # Match by residue number — the two structures may cover different sequence
    # extents (e.g. a cropped design vs a full-length OF3 prediction).
    design_rec_ca = _ca_atoms(design_atoms, receptor_chain)
    afm_rec_ca = _ca_atoms(afm_atoms, receptor_chain)

    common_rec_res = np.intersect1d(design_rec_ca.res_id, afm_rec_ca.res_id)
    if len(common_rec_res) >= 3:
        design_rec_ca_matched = design_rec_ca[np.isin(design_rec_ca.res_id, common_rec_res)]
        afm_rec_ca_matched = afm_rec_ca[np.isin(afm_rec_ca.res_id, common_rec_res)]
    else:
        # Fall back to positional pairing (e.g. OF3 renumbered from 1)
        n = min(design_rec_ca.array_length(), afm_rec_ca.array_length())
        if n < 3:
            raise ValueError(
                f"Fewer than 3 receptor Cα atoms for superposition "
                f"(design {design_rec_ca.array_length()}, AFM {afm_rec_ca.array_length()})."
            )
        design_rec_ca_matched = design_rec_ca[:n]
        afm_rec_ca_matched = afm_rec_ca[:n]

    # superimpose(reference, mobile) → (superimposed_mobile, transform)
    _, transform = struc.superimpose(afm_rec_ca_matched, design_rec_ca_matched)

    design_pep_ca = _ca_atoms(design_atoms, binder_chain)
    afm_pep_ca = _ca_atoms(afm_atoms, binder_chain)

    if design_pep_ca.array_length() == 0:
        raise ValueError(
            f"No Cα atoms found for binder chain '{binder_chain}' in design structure."
        )
    if afm_pep_ca.array_length() == 0:
        raise ValueError(
            f"No Cα atoms found for binder chain '{binder_chain}' in AFM structure."
        )

    # For CoM: match binder residues by number; fall back to positional
    # pairing when numbering schemes differ (e.g. OF3 renumbers from 1).
    common_pep_res = np.intersect1d(design_pep_ca.res_id, afm_pep_ca.res_id)
    if len(common_pep_res) >= 1:
        design_pep_ca_matched = design_pep_ca[np.isin(design_pep_ca.res_id, common_pep_res)]
        afm_pep_ca_matched = afm_pep_ca[np.isin(afm_pep_ca.res_id, common_pep_res)]
    else:
        # No residue-number overlap — pair positionally up to the shorter length
        n = min(design_pep_ca.array_length(), afm_pep_ca.array_length())
        if n == 0:
            raise ValueError(
                f"No binder Cα atoms for chain '{binder_chain}' in one of the structures."
            )
        design_pep_ca_matched = design_pep_ca[:n]
        afm_pep_ca_matched = afm_pep_ca[:n]

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

    if_mask = _auto_interface_mask(
        design_rec_cb.coord, design_pep_cb.coord, interface_cutoff_angstrom
    )
    if not if_mask.any():
        if_mask = np.ones(design_rec_cb.array_length(), dtype=bool)
    if_res_ids = design_rec_cb.res_id[if_mask]

    # Map those interface residue numbers to the AFM structure
    afm_rec_cb = _cb_atoms(afm_atoms, receptor_chain)
    if afm_rec_cb.array_length() == 0:
        raise ValueError(
            f"No Cβ/Cα atoms found for receptor chain '{receptor_chain}' in AFM structure."
        )

    afm_if_mask = np.isin(afm_rec_cb.res_id, if_res_ids)
    if not afm_if_mask.any():
        afm_if_mask = np.ones(afm_rec_cb.array_length(), dtype=bool)
    afm_rec_if_coords = afm_rec_cb.coord[afm_if_mask]

    afm_pep_cb = _cb_atoms(afm_atoms, binder_chain)
    if afm_pep_cb.array_length() == 0:
        raise ValueError(
            f"No Cβ/Cα atoms found for binder chain '{binder_chain}' in AFM structure."
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
        "n_superposition_residues": int(len(common_rec_res)),
        "afm_if_dist_pep_to_rec": afm_if_dist_pep_to_rec,
        "afm_if_dist_rec_to_pep": afm_if_dist_rec_to_pep,
        "afm_mean_if_dist": afm_mean_if_dist,
        "afm_mean_plddt_binder": None,
        "evobind_adversarial_score": None,
    }

    if afm_plddt_per_atom is not None:
        per_res = _per_residue_plddt(afm_plddt_per_atom, afm_atoms, binder_chain)
        afm_mean_plddt = float(per_res.mean()) if per_res.size > 0 else float("nan")
        result["afm_mean_plddt_binder"] = afm_mean_plddt
        if afm_mean_plddt > 0 and np.isfinite(afm_mean_plddt):
            result["evobind_adversarial_score"] = (
                afm_mean_if_dist * (100.0 / afm_mean_plddt) * delta_com
            )

    return result
