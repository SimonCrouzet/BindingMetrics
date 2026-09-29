"""Hydrogen-bond and salt-bridge detection for protein complexes.

Both metrics return a small dict containing an energy estimate (kcal/mol,
primary signal) plus interpretable count(s) (supplementary). See
``docs/metrics.md`` for the full description and rationale.
"""

import warnings
from functools import lru_cache
from typing import Literal, Optional

import numpy as np

# Coulomb constant in kcal/mol when distance is in Å and charges in e
_COULOMB_K = 332.0637133

# Effective interior dielectric used by the salt-bridge energy.
# Matches ``compute_coulomb_cross_chain`` so the two metrics are comparable.
_DIELECTRIC = 4.0

# H-bond energy scale: chosen so an ideal H-bond (d_HA = 2.0 Å, θ_DHA = 180°)
# evaluates to ~ -2.5 kcal/mol — the conventional protein H-bond strength.
_HBOND_K = 5.0


def _import_biotite():
    try:
        import biotite.structure as structure
        import biotite.structure.io.pdbx as pdbx
        from biotite.structure.info import vdw_radius_single
        from biotite.structure.sasa import sasa

        return structure, pdbx, sasa, vdw_radius_single
    except ImportError:
        raise ImportError(
            "biotite is required for H-bond/salt bridge metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _import_hydride():
    try:
        import hydride

        return hydride
    except ImportError:
        return None


def _prepare_for_hydride(atoms, structure_mod):
    """Ensure the AtomArray has the BondList and charge field hydride needs.

    Hydride requires (a) an associated BondList and (b) a ``charge`` annotation.
    Raw AlphaFold/OpenFold-style CIFs typically have neither, which used to make
    the call fail silently and yield zero H-bonds. We infer bonds from residue
    names (CCD-based) and zero-fill missing charges as needed.
    """
    if atoms.bonds is None:
        atoms.bonds = structure_mod.connect_via_residue_names(atoms)
    if "charge" not in atoms.get_annotation_categories():
        atoms.set_annotation("charge", np.zeros(len(atoms), dtype=int))
    return atoms


def _add_hydrogens_if_needed(atoms, structure_mod):
    """Add explicit hydrogens via hydride iff the structure has none yet.

    Returns ``(atoms, reason)``. ``reason`` is None unless hydride failed on a
    structure that needed hydrogens; the H-bond count is then 0 for want of
    hydrogens, not because no bond exists.
    """
    n_h = int(np.sum(atoms.element == "H"))
    if n_h > 0:
        return atoms, None

    hydride = _import_hydride()
    if hydride is None:
        warnings.warn(
            "hydride is not installed and the structure has no explicit hydrogens; "
            "H-bonds will be undercounted. Install with: pip install binding-metrics[biotite]",
            stacklevel=2,
        )
        return atoms, None

    atoms = _prepare_for_hydride(atoms, structure_mod)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms_h, _ = hydride.add_hydrogen(atoms)
        return atoms_h, None
    except (ValueError, KeyError, structure_mod.BadStructureError) as e:
        # ValueError: non-finite coordinates; BadStructureError: residues hydride cannot
        # complete. Either way the structure has no hydrogens to build H-bonds from.
        message = f"hydride.add_hydrogen failed ({type(e).__name__}: {e})"
        warnings.warn(f"{message}; H-bonds may be undercounted", stacklevel=2)
        return atoms, message


def _angle_deg(a, b, c):
    """D-H...A angle in degrees for vector triplets (a=D, b=H, c=A)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = np.maximum(n1 * n2, 1e-12)
    cos_t = np.clip(np.sum(v1 * v2, axis=-1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def _no_hbonds(reason: Optional[str]) -> dict:
    result = {"hbond_energy": 0.0, "hbonds": 0}
    if reason is not None:
        result["reason"] = reason
    return result


def compute_hbonds(
    atoms,
    peptide_chain: str,
    receptor_chain: str,
    *,
    hetero: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Detect cross-chain hydrogen bonds and score them.

    Uses biotite's Baker-Hubbard detector (default: H-acceptor distance ≤ 2.5 Å,
    D-H···A angle ≥ 120°). If the input has no explicit hydrogens, hydride is
    used to add them after building a BondList.

    Triplets returned by biotite are deduplicated to unique cross-chain
    ``(donor_heavy, acceptor_heavy)`` pairs so that e.g. ARG NH1's two
    equivalent hydrogens contacting the same acceptor count as one H-bond
    rather than two. The shortest-H–A triplet is kept per pair.

    Energy per kept pair:

        E = -k_hb * cos²(180° - θ_DHA) / d_HA
        k_hb = 5.0 kcal·Å/mol  (ideal d=2.0, θ=180° → -2.5 kcal/mol)

    Parameters
    ----------
    hetero : {"ignore", "keep"}
        "ignore" (default) drops waters, ions, ligands and glycans before the
        chain selection, so a water that carries a receptor chain ID is not
        counted as a receptor atom. "keep" uses every atom as before.

    Returns
    -------
    dict with keys:
        hbond_energy : float  — sum of pair energies, kcal/mol (≤ 0)
        hbonds       : int    — number of unique cross-chain heavy-atom pairs
        reason       : str    — only when hydrogens could not be built or the
                                detector failed; the count is then 0 for that
                                reason rather than a measured absence of H-bonds
    """
    from binding_metrics.metrics.interface import filter_hetero_atoms

    structure, _, _, _ = _import_biotite()
    atoms = filter_hetero_atoms(atoms, hetero)
    atoms, hydrogen_reason = _add_hydrogens_if_needed(atoms, structure)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            triplets = structure.hbond(atoms)
        except (ValueError, IndexError, structure.BadStructureError) as e:
            message = f"biotite.hbond failed ({type(e).__name__}: {e})"
            warnings.warn(message, stacklevel=2)
            return {"hbond_energy": 0.0, "hbonds": 0, "reason": message}

    if len(triplets) == 0:
        return _no_hbonds(hydrogen_reason)

    chain_id = atoms.chain_id
    coords = atoms.coord

    donor_chains = chain_id[triplets[:, 0]]
    acceptor_chains = chain_id[triplets[:, 2]]
    cross = ((donor_chains == peptide_chain) & (acceptor_chains == receptor_chain)) | (
        (donor_chains == receptor_chain) & (acceptor_chains == peptide_chain)
    )
    triplets = triplets[cross]
    if len(triplets) == 0:
        return _no_hbonds(hydrogen_reason)

    d_idx = triplets[:, 0]
    h_idx = triplets[:, 1]
    a_idx = triplets[:, 2]

    d_HA = np.linalg.norm(coords[h_idx] - coords[a_idx], axis=-1)

    # Dedupe: keep the shortest-distance triplet per unique (donor_heavy, acceptor_heavy) pair.
    best: dict[tuple[int, int], int] = {}
    for k, (di, ai) in enumerate(zip(d_idx, a_idx)):
        key = (int(di), int(ai))
        if key not in best or d_HA[k] < d_HA[best[key]]:
            best[key] = k
    keep = np.fromiter(best.values(), dtype=int)

    d_idx_k = d_idx[keep]
    h_idx_k = h_idx[keep]
    a_idx_k = a_idx[keep]
    d_HA_k = d_HA[keep]
    theta = _angle_deg(coords[d_idx_k], coords[h_idx_k], coords[a_idx_k])

    angular = np.cos(np.radians(180.0 - theta)) ** 2
    energies = -_HBOND_K * angular / np.maximum(d_HA_k, 1e-3)

    return {
        "hbond_energy": float(np.sum(energies)),
        "hbonds": int(len(keep)),
    }


# Side-chain charged-atom allowlists (L-residue names; D-residues are mapped to their
# L counterpart first, see ``l_equivalent_residue_names``).
# HIS / HID / HIE are intentionally excluded — see docs/metrics.md.
# SEP/TPO/PTR are the phosphorylated residues of the AMBER phosaa set used by the
# relaxation step (core/phosaa.py): a phosphate monoester carrying net -2 at pH 7,
# shared by the three non-bridging oxygens.
_POSITIVE_ATOMS: set[tuple[str, str]] = {
    ("LYS", "NZ"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("ARG", "NE"),
    ("HIP", "ND1"),
    ("HIP", "NE2"),
}
_NEGATIVE_ATOMS: set[tuple[str, str]] = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
    ("SEP", "O1P"),
    ("SEP", "O2P"),
    ("SEP", "O3P"),
    ("TPO", "O1P"),
    ("TPO", "O2P"),
    ("TPO", "O3P"),
    ("PTR", "O1P"),
    ("PTR", "O2P"),
    ("PTR", "O3P"),
}


@lru_cache(maxsize=1)
def _d_to_l_residue_names() -> dict[str, str]:
    # Imported on first use so that importing this module does not pull in the
    # simulation stack that ``binding_metrics.core`` may load.
    from binding_metrics.core.nonstandard import D_AA_MAP

    return dict(D_AA_MAP)


def l_equivalent_residue_names(res_names) -> np.ndarray:
    """Upper-case, stripped residue names with D-amino-acid codes mapped to L.

    D-amino acids keep the side-chain atom names of their L counterpart (DLY has
    NZ, DAR has NH1/NH2, DAS has OD1/OD2), so charge tables written for L names
    apply after this mapping. The D-code registry is
    ``binding_metrics.core.nonstandard.D_AA_MAP``.
    """
    names = np.char.upper(np.char.strip(np.asarray(res_names).astype(str)))
    if names.size == 0:
        return names
    mapping = _d_to_l_residue_names()
    unique, inverse = np.unique(names, return_inverse=True)
    mapped = np.array([mapping.get(name, name) for name in unique])
    return mapped[inverse]


def compute_saltbridges(
    atoms,
    peptide_chain: str,
    receptor_chain: str,
    distance_min: float = 0.5,
    distance_max: float = 5.5,
    *,
    hetero: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Detect cross-chain salt bridges and score them.

    A cross-chain residue pair (one positive side chain, one negative side
    chain) is considered a salt bridge if at least one positive-atom /
    negative-atom contact falls in ``(distance_min, distance_max)`` Å.

    Charged-atom allowlist (D-amino acids are matched through their L
    counterpart, e.g. DLY as LYS):
        positive: LYS NZ, ARG NH1/NH2/NE, HIP ND1/NE2
        negative: ASP OD1/OD2, GLU OE1/OE2, and the non-bridging phosphate
                  oxygens O1P/O2P/O3P of SEP, TPO and PTR

    Plain HIS, HID and HIE are treated as neutral (no pKa lookup performed).
    HIP is the AMBER name for the doubly-protonated, +1 form. After
    pdbfixer/OpenMM relaxation, charged histidines are typically renamed to
    HIP; raw predicted CIFs only carry HIS and so contribute no histidine
    salt bridges.

    Energy per residue pair (Coulomb at ε=4, closest atom-pair distance):

        E_pair = (q_pos * q_neg) * 332.06 / (ε * r_min)
               = -83.02 / r_min  kcal/mol  (unit charges)

    A bidentate bridge naturally scores stronger because r_min is the
    shorter of the two contacts.

    Parameters
    ----------
    hetero : {"ignore", "keep"}
        "ignore" (default) drops waters, ions, ligands and glycans before the
        chain selection; "keep" uses every atom as before.

    Returns
    -------
    dict with keys:
        saltbridge_energy     : float — sum of pair energies, kcal/mol (≤ 0)
        saltbridges           : int   — residue-pair count
        saltbridges_bidentate : int   — pairs with ≥ 2 atom-pair contacts
    """
    from binding_metrics.metrics.interface import filter_hetero_atoms

    atoms = filter_hetero_atoms(atoms, hetero)
    pos_mask = np.zeros(len(atoms), dtype=bool)
    neg_mask = np.zeros(len(atoms), dtype=bool)

    res_names = l_equivalent_residue_names(atoms.res_name)
    atom_names = np.char.strip(atoms.atom_name.astype(str))

    for i in range(len(atoms)):
        key = (res_names[i], atom_names[i])
        if key in _POSITIVE_ATOMS:
            pos_mask[i] = True
        elif key in _NEGATIVE_ATOMS:
            neg_mask[i] = True

    pos_atoms = atoms[pos_mask]
    neg_atoms = atoms[neg_mask]

    empty = {"saltbridge_energy": 0.0, "saltbridges": 0, "saltbridges_bidentate": 0}
    if len(pos_atoms) == 0 or len(neg_atoms) == 0:
        return empty

    diff = pos_atoms.coord[:, None, :] - neg_atoms.coord[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    in_window = (distances > distance_min) & (distances < distance_max)
    pos_chains = pos_atoms.chain_id
    neg_chains = neg_atoms.chain_id

    cross = ((pos_chains[:, None] == peptide_chain) & (neg_chains[None, :] == receptor_chain)) | (
        (pos_chains[:, None] == receptor_chain) & (neg_chains[None, :] == peptide_chain)
    )
    valid = in_window & cross

    if not np.any(valid):
        return empty

    # Aggregate atom-pair contacts to residue-pair level.
    pos_res_keys = list(
        zip(
            pos_atoms.chain_id.tolist(),
            pos_atoms.res_id.tolist(),
            pos_atoms.res_name.tolist(),
        )
    )
    neg_res_keys = list(
        zip(
            neg_atoms.chain_id.tolist(),
            neg_atoms.res_id.tolist(),
            neg_atoms.res_name.tolist(),
        )
    )

    # For each (pos_res, neg_res) that has any qualifying contact:
    #   r_min = closest qualifying atom-pair distance
    #   n_contacts = number of qualifying atom-pair contacts
    pair_rmin: dict[tuple, float] = {}
    pair_count: dict[tuple, int] = {}

    pi_idx, ni_idx = np.where(valid)
    for k in range(len(pi_idx)):
        pi, ni = int(pi_idx[k]), int(ni_idx[k])
        key = (pos_res_keys[pi], neg_res_keys[ni])
        d = float(distances[pi, ni])
        if key not in pair_rmin or d < pair_rmin[key]:
            pair_rmin[key] = d
        pair_count[key] = pair_count.get(key, 0) + 1

    n_pairs = len(pair_rmin)
    n_bidentate = sum(1 for c in pair_count.values() if c >= 2)

    # E_pair = q_pos * q_neg * COULOMB_K / (eps * r_min); unit charges of opposite sign.
    energy = -_COULOMB_K / _DIELECTRIC * sum(1.0 / r for r in pair_rmin.values())

    return {
        "saltbridge_energy": float(energy),
        "saltbridges": int(n_pairs),
        "saltbridges_bidentate": int(n_bidentate),
    }
