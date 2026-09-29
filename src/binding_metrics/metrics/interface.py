"""Binding interface analysis for protein complexes.

Aggregates SASA, hydrogen bonds, salt bridges, and solvation energy into a
single interface characterisation. The solvation binding energy (ΔG_int)
follows the PISA approach of Krissinel & Henrick (J. Mol. Biol. 372:774-797,
2007) using Eisenberg-McLachlan atomic solvation parameters (Nature
319:199-203, 1986).

Areas come from the Shrake-Rupley algorithm (J. Mol. Biol. 79:351-371, 1973)
as implemented in biotite, with a 1.4 Å water probe and biotite's
single-radius van der Waals table by element (documented there as Mantina et
al., J. Phys. Chem. A 113:5806-5812, 2009; 1.8 Å for an element without an
entry). Water and monoatomic ions are not part of the surface.

Usage:
    binding-metrics-interface --input complex.cif --design-chain A
"""

import argparse
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from binding_metrics.metrics._common import resolve_chain_role
from binding_metrics.metrics.polar_contacts import (
    _NEGATIVE_ATOMS,
    _POSITIVE_ATOMS,
    l_equivalent_residue_names,
)
from binding_metrics.utils import backfill_auth_columns, configure_logging

logger = logging.getLogger(__name__)

_HETERO_MODES = ("ignore", "keep")

# Polymer residues that biotite's CCD-based amino-acid filter does not know:
# AMBER/OpenMM protonation variants and the terminal capping groups. Structures
# that come from MD or from other modelling tools carry them, and dropping them
# would remove residues of the chain instead of the surrounding solvent.
_AMBER_VARIANT_NAMES = frozenset({"HID", "HIE", "HIN", "CYX", "ASH"})
_CAP_NAMES = frozenset({"ACE", "NME", "NH2"})

# Atomic solvation parameters of Eisenberg & McLachlan, Nature 319:199-203 (1986).
# The published values (cal/mol/Å² of accessible area; positive = exposing the atom
# to water is unfavorable) are, with their standard errors:
#     C +16 (2), S +21 (10), neutral O/N -6 (4), charged O(-) -24 (10), charged N(+) -50 (9)
# as tabulated for the same reference in Table 1 of Krissinel & Henrick, J. Mol. Biol.
# 372:774-797 (2007). ΔG_int below is the solvation part of binding, so each value enters
# with the opposite sign and per Å² of BURIED area (kcal/mol/Å²; negative = burial is
# favorable, positive = burial is unfavorable).
_SOLVATION_PARAMS: dict[str, float] = {
    "C": -0.016,
    "S": -0.021,
    "N/O": +0.006,
    "O-": +0.024,
    "N+": +0.050,
}

_KCAL_TO_KJ = 4.184

# Points sampled on each atom sphere by the Shrake-Rupley algorithm (Shrake &
# Rupley, J. Mol. Biol. 79:351-371, 1973). More points reduce the sampling noise
# of the per-atom areas at linear cost.
SASA_POINT_NUMBER = 960


def _import_biotite():
    """Lazy import of required biotite modules."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx
        from biotite.structure.info import vdw_radius_single
        from biotite.structure.sasa import sasa as biotite_sasa

        return struc, pdbx, pdb_io, biotite_sasa, vdw_radius_single
    except ImportError:
        raise ImportError(
            "biotite is required for interface metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )


def load_biotite_structure(cif_path: str | Path):
    """Load a PDB or CIF file as a biotite AtomArray.

    Args:
        cif_path: Path to CIF or PDB file

    Returns:
        biotite AtomArray
    """
    _, pdbx, pdb_io, _, _ = _import_biotite()
    path = Path(cif_path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        pdbx_file = pdbx.CIFFile.read(str(path))
        backfill_auth_columns(pdbx_file)
        try:
            return pdbx.get_structure(pdbx_file, model=1, extra_fields=["charge"])
        except (KeyError, ValueError):
            # Missing or malformed pdbx_formal_charge column: the charge annotation is optional.
            return pdbx.get_structure(pdbx_file, model=1)
    else:
        pdb_file = pdb_io.PDBFile.read(str(path))
        return pdb_io.get_structure(pdb_file, model=1)


def _amino_acid_mask(atoms) -> np.ndarray:
    """Atoms of amino-acid residues: CCD peptide-linking residues plus AMBER variants."""
    struc, _, _, _, _ = _import_biotite()
    res_names = np.char.upper(np.char.strip(atoms.res_name.astype(str)))
    return struc.filter_amino_acids(atoms) | np.isin(res_names, list(_AMBER_VARIANT_NAMES))


def filter_hetero_atoms(atoms, hetero: Literal["ignore", "keep"] = "ignore"):
    """Apply the heteroatom policy of the structure-based metrics.

    Waters, ions, ligands and glycans often carry the chain ID of the protein
    chain they sit next to. Selecting atoms by chain ID alone then counts them
    as protein atoms, and biotite reports NaN SASA for waters and monoatomic
    ions, which turned every area sum into NaN.

    Args:
        atoms: biotite AtomArray.
        hetero: "ignore" keeps the polymer: amino-acid atoms selected with
            ``biotite.structure.filter_amino_acids`` (this includes D- and
            other non-canonical peptide-linking residues), the AMBER
            protonation variants (HID, HIE, HIN, CYX, ASH) and the ACE, NME
            and NH2 capping groups. "keep" returns the input unchanged.

    Returns:
        The AtomArray to use for chain selection.

    Raises:
        ValueError: If ``hetero`` is not "ignore" or "keep".
    """
    if hetero not in _HETERO_MODES:
        raise ValueError(f"hetero must be one of {_HETERO_MODES}, got {hetero!r}")
    if hetero == "keep":
        return atoms
    res_names = np.char.upper(np.char.strip(atoms.res_name.astype(str)))
    polymer = _amino_acid_mask(atoms) | np.isin(res_names, list(_CAP_NAMES))
    return atoms[polymer]


def detect_interface_chains(
    atoms, design_chain: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Identify peptide and receptor chains from a biotite AtomArray.

    A chain is a protein chain if it has amino-acid atoms: standard, D- and
    other non-canonical peptide-linking residues (biotite's
    ``filter_amino_acids``) and the AMBER protonation variants. Waters, ions and
    ligands that share a chain ID are not counted towards its size. The smallest
    protein chain is taken as the peptide, the largest as receptor.

    Args:
        atoms: biotite AtomArray
        design_chain: Optional explicit chain ID for the peptide

    Returns:
        Tuple of (peptide_chain_id, receptor_chain_id)
    """
    amino_acid_atoms = atoms[_amino_acid_mask(atoms)]
    protein_chains = []
    for chain_id in np.unique(amino_acid_atoms.chain_id):
        chain_atoms = amino_acid_atoms[amino_acid_atoms.chain_id == chain_id]
        protein_chains.append((chain_id, len(np.unique(chain_atoms.res_id))))

    if not protein_chains:
        return None, None

    if design_chain:
        others = [(c, n) for c, n in protein_chains if c != design_chain]
        receptor_chain = max(others, key=lambda x: x[1])[0] if others else None
        return design_chain, receptor_chain

    protein_chains.sort(key=lambda x: x[1])
    if len(protein_chains) >= 2:
        return protein_chains[0][0], protein_chains[-1][0]
    return protein_chains[0][0], None


# ---------------------------------------------------------------------------
# Internal SASA helpers
# ---------------------------------------------------------------------------


def _get_vdw_radii(atoms, vdw_fn) -> np.ndarray:
    radii = []
    for atom in atoms:
        element = str(atom.element).strip()
        r = vdw_fn(element)
        radii.append(r if r is not None else 1.8)
    return np.array(radii, dtype=float)


def _per_atom_sasa(atoms, probe_radius, sasa_fn, vdw_fn) -> np.ndarray:
    """Per-atom Shrake-Rupley SASA (Å²), SASA_POINT_NUMBER points per atom.

    biotite returns NaN for atoms it does not sample: water, monoatomic ions
    and atoms with non-finite coordinates. Those atoms neither occlude nor
    expose surface in biotite's model, so they count as 0.0 here and cannot
    poison the sums built from this array.
    """
    radii = _get_vdw_radii(atoms, vdw_fn)
    per_atom = sasa_fn(
        atoms, probe_radius=probe_radius, vdw_radii=radii, point_number=SASA_POINT_NUMBER
    )
    return np.where(np.isfinite(per_atom), per_atom, 0.0)


def _solvation_types(atoms) -> np.ndarray:
    """Eisenberg-McLachlan atom type per atom.

    Returns an array of "C", "S", "N/O" (neutral), "O-" (carboxylate oxygen),
    "N+" (Lys NZ, Arg NE/NH1/NH2, HIP ring nitrogens) or "" for atoms that carry
    no parameter (hydrogen, phosphorus, selenium, halogens). The charged atoms are
    those of the salt-bridge allowlists in ``polar_contacts`` (D-residues through
    their L counterpart, phosphate oxygens of SEP/TPO/PTR as O(-)), so both metrics
    agree on which groups are charged. Termini are typed as neutral N/O.
    """
    elements = np.char.upper(np.char.strip(atoms.element.astype(str)))
    res_names = l_equivalent_residue_names(atoms.res_name)
    atom_names = np.char.strip(atoms.atom_name.astype(str))

    types = np.full(len(atoms), "", dtype="<U3")
    types[elements == "C"] = "C"
    types[elements == "S"] = "S"
    polar = np.isin(elements, ["N", "O"])
    types[polar] = "N/O"
    for i in np.flatnonzero(polar):
        key = (res_names[i], atom_names[i])
        if key in _POSITIVE_ATOMS:
            types[i] = "N+"
        elif key in _NEGATIVE_ATOMS:
            types[i] = "O-"
    return types


def _gamma_array(atoms) -> np.ndarray:
    """Per-atom solvation parameter array (kcal/mol/Å² of buried area)."""
    types = _solvation_types(atoms)
    gamma = np.zeros(len(atoms))
    for atom_type, value in _SOLVATION_PARAMS.items():
        gamma[types == atom_type] = value
    return gamma


def _polar_mask(atoms) -> np.ndarray:
    """Boolean mask: True for N and O atoms."""
    return np.array([str(a.element).strip().upper() in ("N", "O") for a in atoms])


def _apolar_mask(atoms) -> np.ndarray:
    """Boolean mask: True for C and S atoms."""
    return np.array([str(a.element).strip().upper() in ("C", "S") for a in atoms])


def _collect_per_residue(chain_atoms, buried_sasa: np.ndarray, threshold: float) -> list[dict]:
    """Aggregate per-atom buried SASA into per-residue metrics.

    Args:
        chain_atoms: biotite AtomArray for one chain
        buried_sasa: per-atom buried SASA array (Å²)
        threshold: minimum residue total buried SASA (Å²) to include

    Returns:
        List of dicts with buried_sasa, delta_g_res, polar_area, apolar_area
        for each residue whose buried SASA meets the threshold.
    """
    res_data: dict[tuple, dict] = {}
    gamma_per_atom = _gamma_array(chain_atoms)

    for atom, bsasa, gamma in zip(chain_atoms, buried_sasa, gamma_per_atom):
        key = (str(atom.chain_id), int(atom.res_id), str(atom.ins_code).strip())
        if key not in res_data:
            res_data[key] = {
                "residue": f"{str(atom.res_name).strip()}:{str(atom.chain_id)}:{atom.res_id}",
                "chain": str(atom.chain_id),
                "res_name": str(atom.res_name).strip(),
                "res_id": int(atom.res_id),
                "buried_sasa": 0.0,
                "delta_g_res": 0.0,
                "polar_area": 0.0,
                "apolar_area": 0.0,
            }
        element = str(atom.element).strip().upper()
        res_data[key]["buried_sasa"] += float(bsasa)
        res_data[key]["delta_g_res"] += gamma * float(bsasa)
        if element in ("N", "O"):
            res_data[key]["polar_area"] += float(bsasa)
        elif element in ("C", "S"):
            res_data[key]["apolar_area"] += float(bsasa)

    return [v for v in res_data.values() if v["buried_sasa"] >= threshold]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def compute_interface_metrics(
    cif_path: str | Path,
    design_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    probe_radius: float = 1.4,
    interface_threshold: float = 0.5,
    *,
    binder_chain: Optional[str] = None,
    target_chain: Optional[str] = None,
    hetero: Literal["ignore", "keep"] = "ignore",
) -> dict:
    """Compute binding interface metrics for a protein complex.

    Combines per-atom SASA analysis with hydrogen bond and salt bridge
    counting. The solvation binding energy (ΔG_int) is the sum over atoms of
    an atomic solvation parameter times the atom's buried area, with the five
    Eisenberg & McLachlan (1986) atom types (C, S, neutral N/O, charged O(-),
    charged N(+)) as used in PISA (Krissinel & Henrick, 2007). ΔG_int is only
    the solvation term: PISA adds explicit hydrogen-bond and salt-bridge terms,
    which are reported here as separate keys. The parameters are not fitted to
    this package's data, so read ΔG_int as a relative score between designs of
    one target, not as an absolute affinity.

    Args:
        cif_path: Path to CIF structure file
        design_chain: Chain ID of the peptide/designed chain. Auto-detected
            (smallest protein chain) if None.
        receptor_chain: Chain ID of the receptor. Auto-detected (largest
            protein chain) if None.
        probe_radius: Solvent probe radius in Å (default 1.4 Å = water)
        interface_threshold: Minimum residue buried SASA (Å²) to classify
            a residue as an interface residue (default 0.5 Å²)
        hetero: "ignore" (default) keeps only the polymer before the chain
            selection (see :func:`filter_hetero_atoms`), so waters, ions,
            ligands and glycans that carry a protein chain ID are dropped.
            "keep" uses every atom with the chain ID as before; atoms without
            a defined SASA (water, ions) then count as zero area instead of
            turning the sums into NaN.
        binder_chain: Alias of ``design_chain``; different IDs in both raise
            ``ValueError``.
        target_chain: Alias of ``receptor_chain``, same rule.

    Returns:
        Dictionary with keys:

        Chains:
            peptide_chain (str), receptor_chain (str)

        SASA (Å²):
            delta_sasa: total buried SASA = SASA(pep) + SASA(rec) - SASA(complex)
            sasa_peptide: peptide SASA in isolation
            sasa_receptor: receptor SASA in isolation
            sasa_complex: complex SASA

        Solvation energy (Eisenberg-McLachlan / PISA):
            delta_g_int (kcal/mol): ΔG_int = Σ_i γ_i × ΔA_i, with ΔA_i the
                buried area of atom i and γ_i in kcal/mol/Å² of buried area:
                C -0.016, S -0.021, neutral N/O +0.006, O(-) +0.024,
                N(+) +0.050. Negative = burial is favorable.
            delta_g_int_kJ (kJ/mol)
            polar_area (Å²): buried area from N and O atoms
            apolar_area (Å²): buried area from C and S atoms
            fraction_polar: polar_area / delta_sasa (NaN when nothing is buried)

        Interface residues:
            n_interface_residues_peptide (int)
            n_interface_residues_receptor (int)
            interface_residues_peptide (list[str]): "RES:CHAIN:NUM" labels
            interface_residues_receptor (list[str])
            per_residue (list[dict]): per interface residue —
                residue, chain, res_name, res_id, buried_sasa,
                delta_g_res, polar_area, apolar_area

        Interactions (heuristic ranking scores, see ``polar_contacts``):
            hbonds (int): cross-chain hydrogen bonds
            hbond_energy (float, kcal/mol, ≤ 0): sum of H-bond pair scores
            saltbridges (int): cross-chain salt-bridge residue pairs
            saltbridges_bidentate (int): pairs with at least two atom contacts
            saltbridge_energy (float, kcal/mol, ≤ 0): sum of pair Coulomb
                scores at ε = 4

        reason (str): present only when a value could not be computed. An empty
            chain after the ``hetero`` filter or a failed SASA leaves the areas
            and ΔG_int NaN; a failed H-bond step leaves hbonds at 0 (a 0 that
            then does not mean "no H-bonds").
    """
    from binding_metrics.metrics.polar_contacts import compute_hbonds, compute_saltbridges

    design_chain = resolve_chain_role("design_chain", design_chain, "binder_chain", binder_chain)
    receptor_chain = resolve_chain_role(
        "receptor_chain", receptor_chain, "target_chain", target_chain
    )
    _, _, _, sasa_fn, vdw_fn = _import_biotite()

    cif_path = Path(cif_path)
    atoms = load_biotite_structure(cif_path)
    chain_source = filter_hetero_atoms(atoms, hetero)

    if design_chain is None or receptor_chain is None:
        auto_pep, auto_rec = detect_interface_chains(chain_source, design_chain)
        design_chain = design_chain or auto_pep
        receptor_chain = receptor_chain or auto_rec

    result: dict = {
        "peptide_chain": design_chain,
        "receptor_chain": receptor_chain,
        "delta_sasa": np.nan,
        "sasa_peptide": np.nan,
        "sasa_receptor": np.nan,
        "sasa_complex": np.nan,
        "delta_g_int": np.nan,
        "delta_g_int_kJ": np.nan,
        "polar_area": np.nan,
        "apolar_area": np.nan,
        "fraction_polar": np.nan,
        "n_interface_residues_peptide": 0,
        "n_interface_residues_receptor": 0,
        "interface_residues_peptide": [],
        "interface_residues_receptor": [],
        "per_residue": [],
        "hbonds": 0,
        "hbond_energy": 0.0,
        "saltbridges": 0,
        "saltbridges_bidentate": 0,
        "saltbridge_energy": 0.0,
    }

    if design_chain is None or receptor_chain is None:
        raise ValueError(
            f"Chain auto-detection failed for {cif_path}: "
            f"peptide_chain={design_chain!r}, receptor_chain={receptor_chain!r}. "
            "Pass --design-chain / --receptor-chain explicitly."
        )

    peptide_mask = chain_source.chain_id == design_chain
    receptor_mask = chain_source.chain_id == receptor_chain
    complex_mask = peptide_mask | receptor_mask

    peptide_atoms = chain_source[peptide_mask]
    receptor_atoms = chain_source[receptor_mask]
    complex_atoms = chain_source[complex_mask]

    if len(peptide_atoms) == 0 or len(receptor_atoms) == 0:
        logger.warning(f"  Warning: Empty chain(s) in {cif_path}")
        empty_chains = [
            chain
            for chain, chain_atoms in (
                (design_chain, peptide_atoms),
                (receptor_chain, receptor_atoms),
            )
            if len(chain_atoms) == 0
        ]
        result["reason"] = (
            f"no atoms left for chain(s) {empty_chains} (hetero={hetero!r}); "
            "interface values are undefined"
        )
        return result

    try:
        sasa_pep = _per_atom_sasa(peptide_atoms, probe_radius, sasa_fn, vdw_fn)
        sasa_rec = _per_atom_sasa(receptor_atoms, probe_radius, sasa_fn, vdw_fn)
        sasa_cpx = _per_atom_sasa(complex_atoms, probe_radius, sasa_fn, vdw_fn)
    except Exception as e:  # kept broad: one bad structure must not abort a batch (see reason)
        logger.warning(f"  Warning: SASA computation failed: {e}")
        result["reason"] = f"SASA computation failed: {type(e).__name__}: {e}"
        return result

    # Split complex SASA back by chain — complex_atoms preserves the original
    # atom ordering so chain submasks correctly select the respective atoms.
    sasa_pep_in_cpx = sasa_cpx[complex_atoms.chain_id == design_chain]
    sasa_rec_in_cpx = sasa_cpx[complex_atoms.chain_id == receptor_chain]

    # Clamp at 0: the isolated and complex areas are sampled independently, so a
    # buried atom can come out marginally negative.
    buried_pep = np.maximum(sasa_pep - sasa_pep_in_cpx, 0.0)
    buried_rec = np.maximum(sasa_rec - sasa_rec_in_cpx, 0.0)

    delta_sasa = float(buried_pep.sum() + buried_rec.sum())

    delta_g_int = float(
        np.dot(_gamma_array(peptide_atoms), buried_pep)
        + np.dot(_gamma_array(receptor_atoms), buried_rec)
    )

    polar_area = float(
        np.dot(_polar_mask(peptide_atoms), buried_pep)
        + np.dot(_polar_mask(receptor_atoms), buried_rec)
    )
    apolar_area = float(
        np.dot(_apolar_mask(peptide_atoms), buried_pep)
        + np.dot(_apolar_mask(receptor_atoms), buried_rec)
    )

    per_res_pep = _collect_per_residue(peptide_atoms, buried_pep, interface_threshold)
    per_res_rec = _collect_per_residue(receptor_atoms, buried_rec, interface_threshold)

    try:
        hbond_result = compute_hbonds(atoms, design_chain, receptor_chain, hetero=hetero)
        saltbridge_result = compute_saltbridges(atoms, design_chain, receptor_chain, hetero=hetero)
    except Exception as e:  # kept broad: one bad structure must not abort a batch (see reason)
        logger.warning(f"  Warning: H-bond/salt bridge computation failed: {e}")
        hbond_result = {"hbonds": 0, "hbond_energy": 0.0}
        saltbridge_result = {"saltbridges": 0, "saltbridges_bidentate": 0, "saltbridge_energy": 0.0}
        polar_reason = f"H-bond/salt bridge computation failed: {type(e).__name__}: {e}"
    else:
        polar_reason = hbond_result.get("reason")
    if polar_reason is not None:
        result["reason"] = polar_reason

    result.update(
        {
            "delta_sasa": delta_sasa,
            "sasa_peptide": float(sasa_pep.sum()),
            "sasa_receptor": float(sasa_rec.sum()),
            "sasa_complex": float(sasa_cpx.sum()),
            "delta_g_int": delta_g_int,
            "delta_g_int_kJ": delta_g_int * _KCAL_TO_KJ,
            "polar_area": polar_area,
            "apolar_area": apolar_area,
            "fraction_polar": polar_area / delta_sasa if delta_sasa > 0 else np.nan,
            "n_interface_residues_peptide": len(per_res_pep),
            "n_interface_residues_receptor": len(per_res_rec),
            "interface_residues_peptide": [r["residue"] for r in per_res_pep],
            "interface_residues_receptor": [r["residue"] for r in per_res_rec],
            "per_residue": per_res_pep + per_res_rec,
            "hbonds": hbond_result.get("hbonds", 0),
            "hbond_energy": hbond_result.get("hbond_energy", 0.0),
            "saltbridges": saltbridge_result.get("saltbridges", 0),
            "saltbridges_bidentate": saltbridge_result.get("saltbridges_bidentate", 0),
            "saltbridge_energy": saltbridge_result.get("saltbridge_energy", 0.0),
        }
    )

    return result


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Compute binding interface metrics (SASA, ΔG_int, H-bonds, salt bridges)"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF file")
    parser.add_argument(
        "--design-chain",
        type=str,
        default=None,
        help="Peptide chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain",
        type=str,
        default=None,
        help="Receptor chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=1.4,
        help="Solvent probe radius in Å (default 1.4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Min buried SASA per residue to count as interface residue (Å², default 0.5)",
    )
    parser.add_argument(
        "--hetero",
        choices=_HETERO_MODES,
        default="ignore",
        help=(
            "Heteroatoms (waters, ions, ligands, glycans): 'ignore' keeps only the polymer "
            "(amino acids and terminal caps, default), 'keep' uses every atom carrying the "
            "chain ID"
        ),
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file

    with log_to_file(args.log_file):
        print(f"Computing interface metrics for: {args.input}")
        metrics = compute_interface_metrics(
            args.input,
            design_chain=args.design_chain,
            receptor_chain=args.receptor_chain,
            probe_radius=args.probe_radius,
            interface_threshold=args.threshold,
            hetero=args.hetero,
        )

        print("\nInterface summary:")
        scalar_keys = [
            "peptide_chain",
            "receptor_chain",
            "delta_sasa",
            "sasa_peptide",
            "sasa_receptor",
            "sasa_complex",
            "delta_g_int",
            "delta_g_int_kJ",
            "polar_area",
            "apolar_area",
            "fraction_polar",
            "n_interface_residues_peptide",
            "n_interface_residues_receptor",
            "hbonds",
            "hbond_energy",
            "saltbridges",
            "saltbridges_bidentate",
            "saltbridge_energy",
        ]
        for key in scalar_keys:
            val = metrics[key]
            print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")

        if "reason" in metrics:
            print(f"\n  Reason: {metrics['reason']}")

        print(f"\n  Interface residues (peptide): {metrics['interface_residues_peptide']}")
        print(f"  Interface residues (receptor): {metrics['interface_residues_receptor']}")

        if metrics["per_residue"]:
            print("\nPer-residue contributions (sorted by buried SASA):")
            sorted_res = sorted(
                metrics["per_residue"], key=lambda r: r["buried_sasa"], reverse=True
            )
            print(f"  {'Residue':<20} {'BuriedSASA':>10} {'ΔG_res':>10} {'Polar':>8} {'Apolar':>8}")
            for r in sorted_res:
                print(
                    f"  {r['residue']:<20} {r['buried_sasa']:>10.2f} "
                    f"{r['delta_g_res']:>10.4f} {r['polar_area']:>8.2f} {r['apolar_area']:>8.2f}"
                )


if __name__ == "__main__":
    main()
