"""Hydrogen bond and salt bridge detection for protein complexes.

Uses biotite for H-bond detection and distance-based salt bridge
identification. Hydride is used optionally to add explicit hydrogen
positions before detection.
"""

import warnings

import numpy as np


def _import_biotite():
    """Import biotite structure modules (lazy)."""
    try:
        import biotite.structure as structure
        import biotite.structure.io.pdbx as pdbx
        from biotite.structure.sasa import sasa
        from biotite.structure.info import vdw_radius_single

        return structure, pdbx, sasa, vdw_radius_single
    except ImportError:
        raise ImportError(
            "biotite is required for H-bond/salt bridge metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _import_hydride():
    """Import hydride for hydrogen addition (optional)."""
    try:
        import hydride

        return hydride
    except ImportError:
        return None


def compute_hbonds(atoms, peptide_chain: str, receptor_chain: str) -> int:
    """Count hydrogen bonds between peptide and receptor chains.

    Uses biotite's hbond() detector. If the hydride package is available,
    explicit hydrogen positions are added first for more accurate detection.

    Args:
        atoms: biotite AtomArray (ideally with explicit hydrogens)
        peptide_chain: Chain ID of the peptide
        receptor_chain: Chain ID of the receptor

    Returns:
        Number of cross-chain H-bonds
    """
    structure, _, _, _ = _import_biotite()
    hydride = _import_hydride()

    if hydride is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms, _ = hydride.add_hydrogen(atoms)
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            hbonds = structure.hbond(atoms)
        except Exception as e:
            print(f"  Warning: H-bond detection failed: {e}")
            return 0

    if len(hbonds) == 0:
        return 0

    donor_chains = atoms.chain_id[hbonds[:, 0]]
    acceptor_chains = atoms.chain_id[hbonds[:, 2]]

    pep_to_rec = int(np.sum((donor_chains == peptide_chain) & (acceptor_chains == receptor_chain)))
    rec_to_pep = int(np.sum((donor_chains == receptor_chain) & (acceptor_chains == peptide_chain)))

    return pep_to_rec + rec_to_pep


def compute_saltbridges(
    atoms,
    peptide_chain: str,
    receptor_chain: str,
    distance_min: float = 0.5,
    distance_max: float = 5.5,
) -> int:
    """Count salt bridges between peptide and receptor.

    Salt bridges are detected as contacts between positively charged atoms
    (LYS NZ, ARG NH1/NH2/NE, HIS ND1/NE2) and negatively charged atoms
    (ASP OD1/OD2, GLU OE1/OE2) from opposite chains.

    Args:
        atoms: biotite AtomArray
        peptide_chain: Chain ID of the peptide
        receptor_chain: Chain ID of the receptor
        distance_min: Minimum distance in Å (to exclude covalent bonds)
        distance_max: Maximum distance in Å for a salt bridge

    Returns:
        Number of cross-chain salt bridges
    """
    positive_atoms = {
        ("LYS", "NZ"),
        ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
        ("HIS", "ND1"), ("HIS", "NE2"),
    }
    negative_atoms = {
        ("ASP", "OD1"), ("ASP", "OD2"),
        ("GLU", "OE1"), ("GLU", "OE2"),
    }

    pos_mask = np.zeros(len(atoms), dtype=bool)
    neg_mask = np.zeros(len(atoms), dtype=bool)

    for i, atom in enumerate(atoms):
        res_name = str(atom.res_name).strip()
        atom_name = str(atom.atom_name).strip()
        if (res_name, atom_name) in positive_atoms:
            pos_mask[i] = True
        elif (res_name, atom_name) in negative_atoms:
            neg_mask[i] = True

    pos_atoms = atoms[pos_mask]
    neg_atoms = atoms[neg_mask]

    if len(pos_atoms) == 0 or len(neg_atoms) == 0:
        return 0

    diff = pos_atoms.coord[:, np.newaxis, :] - neg_atoms.coord[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    pos_idxs, neg_idxs = np.where((distances > distance_min) & (distances < distance_max))

    if len(pos_idxs) == 0:
        return 0

    pos_chains = pos_atoms.chain_id
    neg_chains = neg_atoms.chain_id

    pep_pos_rec_neg = int(np.sum(
        (pos_chains[pos_idxs] == peptide_chain) & (neg_chains[neg_idxs] == receptor_chain)
    ))
    rec_pos_pep_neg = int(np.sum(
        (pos_chains[pos_idxs] == receptor_chain) & (neg_chains[neg_idxs] == peptide_chain)
    ))

    return pep_pos_rec_neg + rec_pos_pep_neg
