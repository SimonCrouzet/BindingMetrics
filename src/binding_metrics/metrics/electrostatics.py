"""Electrostatic cross-chain Coulomb energy for protein complexes.

Computes a simplified Coulomb interaction energy between formal charges
on ionisable residues across the peptide/receptor interface at pH 7:

    E = (k / ε) Σ q_i q_j / r_ij   over cross-chain charged-atom pairs with r_ij < cutoff

with one uniform dielectric ε = 4 (the interior dielectric of a macromolecule,
Krissinel & Henrick, J. Mol. Biol. 372:774-797, 2007). It is a heuristic ranking
score: formal charges from residue names, no pKa shifts, no chain termini, no
solvent screening, and a package-chosen 12 Å cutoff. Do not read it as a
binding free energy.

Usage:
    binding-metrics-electrostatics --input complex.cif --design-chain A
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics.metrics._common import (
    KJ_TO_KCAL,
    ChainAliasAction,
    import_biotite,
    load_structure,
    resolve_chain_role,
)
from binding_metrics.metrics.polar_contacts import l_equivalent_residue_names
from binding_metrics.utils import configure_logging

# Formal partial charges assigned to ionisable atoms at pH 7, keyed by L-residue name
# (D-amino acids are mapped to their L counterpart before the lookup).
# Charges are split over resonance-equivalent atoms: the ARG guanidinium and the
# HIP imidazolium (+1 each), the ASP/GLU carboxylate (-1), and the three non-bridging
# oxygens of a phosphorylated residue, which carries net -2 in the AMBER phosaa
# templates the relaxation step uses (core/phosaa.py). Plain HIS is neutral.
_FORMAL_CHARGES: dict[tuple[str, str], float] = {
    ("LYS", "NZ"): +1.0,
    ("ARG", "NH1"): +0.5,
    ("ARG", "NH2"): +0.5,
    ("HIP", "ND1"): +0.5,
    ("HIP", "NE2"): +0.5,
    ("ASP", "OD1"): -0.5,
    ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5,
    ("GLU", "OE2"): -0.5,
    **{(res, atom): -2.0 / 3.0 for res in ("SEP", "TPO", "PTR") for atom in ("O1P", "O2P", "O3P")},
}

# Residues whose ionisation state is modelled or known to be neutral. An amino-acid
# residue outside this set (an ncAA such as MLE or BMT) may carry a charge that
# _FORMAL_CHARGES does not know; it is counted in ``n_residues_unrecognised``.
_RECOGNISED_RESIDUES = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "HID HIE HIN HIP CYX CYM ASH GLH LYN MSE SEP TPO PTR"
    ).split()
)

# Coulomb constant: e²/(4πε₀) × N_A = 1389.35 kJ·Å/mol·e²
# Assumes distances in Ångströms; returns energy in kJ/mol.
_COULOMB_KJ_ANG_MOL: float = 1389.35
_KJ_TO_KCAL: float = KJ_TO_KCAL


def _import_biotite():
    """Lazy import of required biotite modules."""
    return import_biotite("electrostatics metrics")


def _load_structure(path: Path):
    """Load a PDB or CIF file as a biotite AtomArray."""
    return load_structure(path, purpose="electrostatics metrics")


def _collect_charged_atoms(chain_atoms) -> tuple[np.ndarray, np.ndarray, list[dict], int, int]:
    """Charged atoms of one chain and residue counts for the result schema.

    Returns:
        positions (n, 3) in Å, charges (n,), a per-atom info list, the number of
        residues that carry at least one charged atom, and the number of
        amino-acid residues whose name is not in ``_RECOGNISED_RESIDUES``.
    """
    struc, _, _ = _import_biotite()
    from binding_metrics.metrics.interface import _amino_acid_mask

    res_l = l_equivalent_residue_names(chain_atoms.res_name)
    atom_names = np.char.strip(chain_atoms.atom_name.astype(str))
    all_charges = np.array([_FORMAL_CHARGES.get(key, 0.0) for key in zip(res_l, atom_names)])
    charged = all_charges != 0.0

    charged_atoms = chain_atoms[charged]
    info = [
        {
            "residue": f"{str(atom.res_name).strip()}:{str(atom.chain_id)}:{atom.res_id}",
            "atom": str(atom.atom_name).strip(),
            "charge": float(q),
            "coords": atom.coord.tolist(),
        }
        for atom, q in zip(charged_atoms, all_charges[charged])
    ]
    n_ionisable = len(struc.get_residue_starts(charged_atoms)) if len(charged_atoms) else 0

    amino_acids = chain_atoms[_amino_acid_mask(chain_atoms)]
    if len(amino_acids):
        starts = struc.get_residue_starts(amino_acids)
        residue_names = l_equivalent_residue_names(amino_acids.res_name[starts])
        n_unrecognised = int(np.sum(~np.isin(residue_names, list(_RECOGNISED_RESIDUES))))
    else:
        n_unrecognised = 0

    if len(charged_atoms) == 0:
        return np.zeros((0, 3)), np.zeros(0), [], n_ionisable, n_unrecognised
    return (
        charged_atoms.coord.astype(float),
        all_charges[charged],
        info,
        n_ionisable,
        n_unrecognised,
    )


def compute_coulomb_cross_chain(
    cif_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    dielectric: float = 4.0,
    cutoff_ang: float = 12.0,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
) -> dict:
    """Compute simplified Coulomb cross-chain interaction energy.

    Assigns formal charges to ionisable residue atoms at pH 7 and computes
    a pairwise Coulomb sum between peptide and receptor charged atoms within
    a distance cutoff. Uses a uniform dielectric constant. D-amino acids use
    the charges of their L counterpart; phosphorylated SEP, TPO and PTR carry
    -2 spread over their three non-bridging oxygens; histidine is neutral
    unless it is named HIP (+1 over ND1 and NE2). Chain termini are not
    charged.

    Type: score

    Args:
        cif_path: Path to structure file (CIF or PDB)
        peptide_chain: Chain ID of peptide (auto-detected if None)
        receptor_chain: Chain ID of receptor (auto-detected if None)
        dielectric: Effective dielectric constant (default 4.0)
        cutoff_ang: Distance cutoff in Å (default 12.0)
        binder_chain: Alias of ``peptide_chain``; different IDs in both raise
            ``ValueError``.
        target_chain: Alias of ``receptor_chain``, same rule.

    Returns:
        Dictionary with keys:

        Scores:
            coulomb_energy_kJ (float): Total cross-chain Coulomb energy in kJ/mol;
                negative = attractive
            coulomb_energy_kcal (float): Same in kcal/mol
            n_charged_pairs (int): Number of charged atom pairs within cutoff
            n_attractive (int): Number of opposite-sign pairs within cutoff
            n_repulsive (int): Number of same-sign pairs within cutoff

        Coverage of the charge table (peptide and receptor chains together):
            n_ionisable_residues_seen (int): residues with at least one charged
                atom in the table (Lys, Arg, Asp, Glu, HIP, SEP/TPO/PTR and their
                D-forms)
            n_residues_unrecognised (int): amino-acid residues whose name is
                neither a standard residue, a protonation variant, a D-form nor a
                phospho residue. Their charge, if any, is not modelled; a
                non-zero count means the energy may be incomplete.

        reason (str): present only when the energy could not be evaluated
            (a chain was not found); the scores are then the zeros above.

        Features:
            charged_atoms_peptide (list[dict]): Per charged atom info for peptide;
                each dict has residue, atom, charge, coords
            charged_atoms_receptor (list[dict]): Per charged atom info for receptor
    """
    from binding_metrics.metrics.interface import detect_interface_chains

    peptide_chain = resolve_chain_role("peptide_chain", peptide_chain, "binder_chain", binder_chain)
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    cif_path = Path(cif_path)
    atoms = _load_structure(cif_path)

    if peptide_chain is None or receptor_chain is None:
        auto_pep, auto_rec = detect_interface_chains(atoms, peptide_chain)
        peptide_chain = peptide_chain or auto_pep
        receptor_chain = receptor_chain or auto_rec

    _default = {
        "coulomb_energy_kJ": 0.0,
        "coulomb_energy_kcal": 0.0,
        "n_charged_pairs": 0,
        "n_attractive": 0,
        "n_repulsive": 0,
        "n_ionisable_residues_seen": 0,
        "n_residues_unrecognised": 0,
        "charged_atoms_peptide": [],
        "charged_atoms_receptor": [],
    }

    if peptide_chain is None or receptor_chain is None:
        # The all-zero default is kept for callers that sum or rank on it; ``reason``
        # tells a 0.0 that means "not evaluated" from a genuine 0.0.
        return {
            **_default,
            "reason": (
                f"chain detection found peptide_chain={peptide_chain!r}, "
                f"receptor_chain={receptor_chain!r}; pass both chain IDs explicitly"
            ),
        }

    pep_mask = atoms.chain_id == peptide_chain
    rec_mask = atoms.chain_id == receptor_chain

    pos_pep, q_pep, info_pep, n_ion_pep, n_unrec_pep = _collect_charged_atoms(atoms[pep_mask])
    pos_rec, q_rec, info_rec, n_ion_rec, n_unrec_rec = _collect_charged_atoms(atoms[rec_mask])
    residue_counts = {
        "n_ionisable_residues_seen": n_ion_pep + n_ion_rec,
        "n_residues_unrecognised": n_unrec_pep + n_unrec_rec,
    }

    if len(pos_pep) == 0 or len(pos_rec) == 0:
        result = dict(_default)
        result.update(residue_counts)
        result["charged_atoms_peptide"] = info_pep
        result["charged_atoms_receptor"] = info_rec
        return result

    diff = pos_pep[:, np.newaxis, :] - pos_rec[np.newaxis, :, :]  # (n_pep, n_rec, 3)
    r = np.linalg.norm(diff, axis=-1)  # (n_pep, n_rec)

    within_cutoff = r < cutoff_ang
    # Coincident atoms would divide by zero; treat them as outside the cutoff.
    r_safe = np.where(within_cutoff & (r > 0), r, np.inf)

    qq = q_pep[:, np.newaxis] * q_rec[np.newaxis, :]  # (n_pep, n_rec)

    e_matrix = np.where(within_cutoff, qq / r_safe, 0.0)
    coulomb_kJ = float(np.sum(e_matrix) * _COULOMB_KJ_ANG_MOL / dielectric)

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
        **residue_counts,
        "charged_atoms_peptide": info_pep,
        "charged_atoms_receptor": info_rec,
    }


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Compute cross-chain Coulomb electrostatic energy")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF/PDB file")
    parser.add_argument(
        "--design-chain",
        "--binder-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Peptide chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Receptor chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--dielectric",
        type=float,
        default=4.0,
        help="Effective dielectric constant (default 4.0)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=12.0,
        help="Distance cutoff in Å (default 12.0)",
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
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
            "coulomb_energy_kJ",
            "coulomb_energy_kcal",
            "n_charged_pairs",
            "n_attractive",
            "n_repulsive",
        ]
        for key in scalar_keys:
            val = metrics[key]
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

        print(f"\n  Charged atoms peptide: {len(metrics['charged_atoms_peptide'])}")
        print(f"  Charged atoms receptor: {len(metrics['charged_atoms_receptor'])}")


if __name__ == "__main__":
    main()
