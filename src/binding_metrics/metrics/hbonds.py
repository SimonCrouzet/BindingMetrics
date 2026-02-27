"""Hydrogen bond and salt bridge metrics for protein complexes.

Uses the biotite library for H-bond detection and distance-based salt bridge
identification. Hydride is used optionally to improve H-bond accuracy by
adding explicit hydrogen positions before detection.

Usage:
    binding-metrics-hbonds --input complex.cif --design-chain A
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

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


def load_biotite_structure(cif_path: str | Path):
    """Load a CIF file as a biotite AtomArray.

    Args:
        cif_path: Path to CIF file

    Returns:
        biotite AtomArray
    """
    structure, pdbx, _, _ = _import_biotite()
    pdbx_file = pdbx.CIFFile.read(str(cif_path))
    return pdbx.get_structure(pdbx_file, model=1, extra_fields=["charge"])


def _detect_biotite_chains(atoms, design_chain: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Identify peptide and receptor chains from a biotite AtomArray.

    Standard amino acid residues are used to identify protein chains.
    The smallest protein chain is taken as the peptide, the largest as receptor.

    Args:
        atoms: biotite AtomArray
        design_chain: Optional explicit chain ID for the peptide

    Returns:
        Tuple of (peptide_chain_id, receptor_chain_id)
    """
    amino_acids = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "MSE", "SEC", "PYL", "HYP", "MLY", "SEP", "TPO", "PTR",
    }

    chain_ids = np.unique(atoms.chain_id)
    protein_chains = []

    for chain_id in chain_ids:
        chain_atoms = atoms[atoms.chain_id == chain_id]
        res_names = {str(r).strip().upper() for r in np.unique(chain_atoms.res_name)}
        if res_names & amino_acids:
            n_res = len(np.unique(chain_atoms.res_id))
            if n_res > 0:
                protein_chains.append((chain_id, n_res))

    if not protein_chains:
        return None, None

    if design_chain:
        peptide_chain = design_chain
        others = [(c, n) for c, n in protein_chains if c != design_chain]
        receptor_chain = max(others, key=lambda x: x[1])[0] if others else None
    else:
        protein_chains.sort(key=lambda x: x[1])
        if len(protein_chains) >= 2:
            peptide_chain = protein_chains[0][0]
            receptor_chain = protein_chains[-1][0]
        elif len(protein_chains) == 1:
            peptide_chain = protein_chains[0][0]
            receptor_chain = None
        else:
            peptide_chain = None
            receptor_chain = None

    return peptide_chain, receptor_chain


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

    # Add hydrogens if hydride is available
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
        distance_min: Minimum distance in Ångström (to exclude covalent bonds)
        distance_max: Maximum distance in Ångström for a salt bridge

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
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    pos_idxs, neg_idxs = np.where(
        (distances > distance_min) & (distances < distance_max)
    )

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


def compute_structure_interactions(
    cif_path: str | Path,
    design_chain: Optional[str] = None,
) -> dict:
    """Compute H-bonds, salt bridges, and delta SASA for a protein complex.

    Loads the structure with biotite and computes interaction metrics between
    the designed (peptide) chain and the receptor chain.

    Args:
        cif_path: Path to CIF structure file
        design_chain: Peptide chain ID. Auto-detected if None.

    Returns:
        Dictionary with keys:
            - hbonds (int): Number of cross-chain hydrogen bonds
            - saltbridges (int): Number of cross-chain salt bridges
            - delta_sasa (float, Å²): Buried surface area upon binding
            - sasa_peptide (float, Å²): SASA of peptide alone
            - sasa_receptor (float, Å²): SASA of receptor alone
            - sasa_complex (float, Å²): SASA of complex
            - peptide_chain (str): Peptide chain ID used
            - receptor_chain (str): Receptor chain ID used
    """
    from binding_metrics.metrics.sasa import compute_delta_sasa_static

    atoms = load_biotite_structure(cif_path)
    peptide_chain, receptor_chain = _detect_biotite_chains(atoms, design_chain)

    result: dict = {
        "hbonds": 0,
        "saltbridges": 0,
        "delta_sasa": np.nan,
        "sasa_peptide": np.nan,
        "sasa_receptor": np.nan,
        "sasa_complex": np.nan,
        "peptide_chain": peptide_chain,
        "receptor_chain": receptor_chain,
    }

    if peptide_chain is None or receptor_chain is None:
        print(f"  Warning: Could not identify peptide/receptor chains in {cif_path}")
        return result

    result["hbonds"] = compute_hbonds(atoms, peptide_chain, receptor_chain)
    result["saltbridges"] = compute_saltbridges(atoms, peptide_chain, receptor_chain)

    sasa_metrics = compute_delta_sasa_static(cif_path, peptide_chain, receptor_chain)
    result.update(sasa_metrics)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute H-bonds, salt bridges, and delta SASA for a protein complex"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF file")
    parser.add_argument("--design-chain", type=str, default=None, help="Peptide chain ID (auto-detect if omitted)")
    args = parser.parse_args()

    print(f"Computing structure interactions for: {args.input}")
    metrics = compute_structure_interactions(args.input, args.design_chain)

    print("\nResults:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
