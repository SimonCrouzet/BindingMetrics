"""Electrostatic cross-chain Coulomb energy for protein complexes.

Computes a simplified Coulomb interaction energy between formal charges
on ionisable residues across the peptide/receptor interface at pH 7.

Usage:
    binding-metrics-electrostatics --input complex.cif --design-chain A
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# Formal partial charges assigned to ionisable atoms at pH 7.
# Split charges on ARG to reflect resonance delocalization.
_FORMAL_CHARGES: dict[tuple[str, str], float] = {
    ("LYS", "NZ"):  +1.0,
    ("ARG", "NH1"): +0.5,
    ("ARG", "NH2"): +0.5,
    ("ASP", "OD1"): -0.5,
    ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5,
    ("GLU", "OE2"): -0.5,
}

# Coulomb constant: e²/(4πε₀) × N_A in kJ·Å/mol
_COULOMB_KJ_ANG_MOL: float = 1389.35
_KJ_TO_KCAL: float = 1.0 / 4.184


def _import_biotite():
    """Lazy import of required biotite modules."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb_io

        return struc, pdbx, pdb_io
    except ImportError:
        raise ImportError(
            "biotite is required for electrostatics metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def _load_structure(path: Path):
    """Load a PDB or CIF file as a biotite AtomArray."""
    struc, pdbx, pdb_io = _import_biotite()
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        pdbx_file = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(pdbx_file, model=1)
    else:
        pdb_file = pdb_io.PDBFile.read(str(path))
        return pdb_io.get_structure(pdb_file, model=1)


def compute_coulomb_cross_chain(
    cif_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    dielectric: float = 4.0,
    cutoff_ang: float = 12.0,
) -> dict:
    """Compute simplified Coulomb cross-chain interaction energy.

    Assigns formal charges to ionisable residue atoms at pH 7 and computes
    a pairwise Coulomb sum between peptide and receptor charged atoms within
    a distance cutoff. Uses a uniform dielectric constant.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        peptide_chain: Chain ID of peptide (auto-detected if None)
        receptor_chain: Chain ID of receptor (auto-detected if None)
        dielectric: Effective dielectric constant (default 4.0)
        cutoff_ang: Distance cutoff in Å (default 12.0)

    Returns:
        Dictionary with keys:

        Scores:
            coulomb_energy_kJ (float): Total cross-chain Coulomb energy in kJ/mol;
                negative = attractive
            coulomb_energy_kcal (float): Same in kcal/mol
            n_charged_pairs (int): Number of charged atom pairs within cutoff
            n_attractive (int): Number of opposite-sign pairs within cutoff
            n_repulsive (int): Number of same-sign pairs within cutoff

        Features:
            charged_atoms_peptide (list[dict]): Per charged atom info for peptide;
                each dict has residue, atom, charge, coords
            charged_atoms_receptor (list[dict]): Per charged atom info for receptor
    """
    from binding_metrics.metrics.interface import detect_interface_chains

    cif_path = Path(cif_path)
    atoms = _load_structure(cif_path)

    if peptide_chain is None or receptor_chain is None:
        auto_pep, auto_rec = detect_interface_chains(atoms, peptide_chain)
        peptide_chain = peptide_chain or auto_pep
        receptor_chain = receptor_chain or auto_rec

    _default = {
        "coulomb_energy_kJ": np.nan,
        "coulomb_energy_kcal": np.nan,
        "n_charged_pairs": 0,
        "n_attractive": 0,
        "n_repulsive": 0,
        "charged_atoms_peptide": [],
        "charged_atoms_receptor": [],
    }

    if peptide_chain is None or receptor_chain is None:
        return _default

    pep_mask = atoms.chain_id == peptide_chain
    rec_mask = atoms.chain_id == receptor_chain

    # Collect charged atoms for each chain
    def _collect_charged(chain_atoms) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Return pos (n,3), charges (n,), and info list for charged atoms."""
        positions = []
        charges = []
        info = []
        for atom in chain_atoms:
            key = (str(atom.res_name).strip(), str(atom.atom_name).strip())
            q = _FORMAL_CHARGES.get(key)
            if q is not None:
                positions.append(atom.coord.tolist())
                charges.append(q)
                info.append({
                    "residue": f"{str(atom.res_name).strip()}:{str(atom.chain_id)}:{atom.res_id}",
                    "atom": str(atom.atom_name).strip(),
                    "charge": q,
                    "coords": atom.coord.tolist(),
                })
        if positions:
            return np.array(positions), np.array(charges), info
        return np.zeros((0, 3)), np.zeros(0), []

    pep_atoms = atoms[pep_mask]
    rec_atoms = atoms[rec_mask]

    pos_pep, q_pep, info_pep = _collect_charged(pep_atoms)
    pos_rec, q_rec, info_rec = _collect_charged(rec_atoms)

    if len(pos_pep) == 0 or len(pos_rec) == 0:
        result = dict(_default)
        result["charged_atoms_peptide"] = info_pep
        result["charged_atoms_receptor"] = info_rec
        return result

    # Vectorised pairwise distances: (n_pep, n_rec)
    diff = pos_pep[:, np.newaxis, :] - pos_rec[np.newaxis, :, :]  # (n_pep, n_rec, 3)
    r = np.linalg.norm(diff, axis=-1)  # (n_pep, n_rec)

    within_cutoff = r < cutoff_ang
    # Avoid division by zero (shouldn't happen cross-chain but be safe)
    r_safe = np.where(within_cutoff & (r > 0), r, np.inf)

    # Charge product matrix
    qq = q_pep[:, np.newaxis] * q_rec[np.newaxis, :]  # (n_pep, n_rec)

    # Energy sum over pairs within cutoff
    e_matrix = np.where(within_cutoff, qq / r_safe, 0.0)
    coulomb_kJ = float(np.sum(e_matrix) * _COULOMB_KJ_ANG_MOL / dielectric)

    # Count pairs
    within_mask = within_cutoff & (r > 0)
    n_charged_pairs = int(np.sum(within_mask))
    n_attractive = int(np.sum(within_mask & (qq < 0)))
    n_repulsive = int(np.sum(within_mask & (qq > 0)))

    return {
        "coulomb_energy_kJ": coulomb_kJ,
        "coulomb_energy_kcal": coulomb_kJ * _KJ_TO_KCAL,
        "n_charged_pairs": n_charged_pairs,
        "n_attractive": n_attractive,
        "n_repulsive": n_repulsive,
        "charged_atoms_peptide": info_pep,
        "charged_atoms_receptor": info_rec,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-chain Coulomb electrostatic energy"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF/PDB file")
    parser.add_argument(
        "--design-chain", type=str, default=None,
        help="Peptide chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain", type=str, default=None,
        help="Receptor chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--dielectric", type=float, default=4.0,
        help="Effective dielectric constant (default 4.0)",
    )
    parser.add_argument(
        "--cutoff", type=float, default=12.0,
        help="Distance cutoff in Å (default 12.0)",
    )
    args = parser.parse_args()

    print(f"Computing Coulomb cross-chain energy for: {args.input}")
    metrics = compute_coulomb_cross_chain(
        args.input,
        peptide_chain=args.design_chain,
        receptor_chain=args.receptor_chain,
        dielectric=args.dielectric,
        cutoff_ang=args.cutoff,
    )

    print("\nElectrostatics summary:")
    scalar_keys = [
        "coulomb_energy_kJ", "coulomb_energy_kcal",
        "n_charged_pairs", "n_attractive", "n_repulsive",
    ]
    for key in scalar_keys:
        val = metrics[key]
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\n  Charged atoms peptide: {len(metrics['charged_atoms_peptide'])}")
    print(f"  Charged atoms receptor: {len(metrics['charged_atoms_receptor'])}")


if __name__ == "__main__":
    main()
