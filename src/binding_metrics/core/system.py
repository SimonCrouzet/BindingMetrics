"""System preparation utilities for MD simulations."""

import os
import tempfile
from typing import Literal

import openmm.unit as unit
from openmm.app import Modeller, PDBFile, ForceField

from binding_metrics.core.forcefields import get_forcefield

# Optional pdbfixer for structure repair
try:
    from pdbfixer import PDBFixer
    HAS_PDBFIXER = True
except ImportError:
    HAS_PDBFIXER = False


def _extract_custom_bonds(topology) -> list:
    """Capture non-sequential intra-chain bonds that PDBxFile.writeFile drops.

    Returns a list of (chain_id, local_res_idx, atom_name,
                        chain_id, local_res_idx, atom_name) tuples.
    Disulfides (SG-SG) are excluded — PDBFixer handles those via STRUCT_CONN.
    """
    res_local: dict[int, tuple] = {}
    for chain in topology.chains():
        for local_idx, res in enumerate(chain.residues()):
            res_local[res.index] = (chain.id, local_idx)

    bonds = []
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        r1, r2 = a1.residue, a2.residue
        if r1.chain.id != r2.chain.id:
            continue
        if abs(r1.index - r2.index) <= 1:
            continue
        if a1.name == "SG" and a2.name == "SG":
            continue  # disulfide — preserved by PDBFixer
        bonds.append((
            res_local[r1.index][0], res_local[r1.index][1], a1.name,
            res_local[r2.index][0], res_local[r2.index][1], a2.name,
        ))
    return bonds


def _readd_custom_bonds(topology, custom_bonds: list):
    """Re-add custom bonds to a post-PDBFixer topology by position."""
    if not custom_bonds:
        return topology
    lookup: dict[tuple, object] = {}
    for chain in topology.chains():
        for local_idx, res in enumerate(chain.residues()):
            for atom in res.atoms():
                lookup[(chain.id, local_idx, atom.name)] = atom

    for (ch1, li1, an1, ch2, li2, an2) in custom_bonds:
        a1 = lookup.get((ch1, li1, an1))
        a2 = lookup.get((ch2, li2, an2))
        if a1 and a2:
            topology.addBond(a1, a2)
        else:
            print(f"  Warning: could not restore custom bond "
                  f"{ch1}[{li1}].{an1} – {ch2}[{li2}].{an2} after prep")
    return topology


def _topology_to_fixer(topology, positions) -> "PDBFixer":
    """Write topology+positions to a temp CIF and load with PDBFixer.

    CIF preserves residue numbers and chain IDs through the roundtrip;
    PDB format truncates residue numbers and may lose chain information.
    """
    if not HAS_PDBFIXER:
        raise ImportError(
            "pdbfixer is required. Install with: pip install binding-metrics[structure]"
        )
    from openmm.app import PDBxFile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
        PDBxFile.writeFile(topology, positions, tmp)
        tmp_path = tmp.name
    try:
        fixer = PDBFixer(filename=tmp_path)
    finally:
        os.unlink(tmp_path)
    return fixer


# Standard amino acids and nucleotides recognised by AMBER ff14SB.
_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
    # protonation variants
    "HIE", "HID", "HIP", "CYX", "ASH", "GLH", "LYN",
    # nucleotides
    "DA", "DC", "DG", "DT", "A", "C", "G", "T", "U",
}

# Common metal ions parameterised in standard force fields (no GAFF2 needed).
_METAL_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Mg", "Ca", "Sr", "Ba",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

_WATER_NAMES = {"HOH", "WAT", "SOL", "TIP", "TIP3", "H2O"}


def _add_hydrogens_cyclic(topology, positions, custom_bonds: list, ph: float) -> tuple:
    """Add hydrogens to a topology that contains non-sequential cyclic bonds.

    PDBFixer's addMissingHydrogens cannot handle cyclic peptides — it applies
    standard N-terminal templates that try to place H2/H3 on the N atom, which
    fails when the N is already bonded to the C-terminus carbon.  This function
    uses the cyclic-aware ForceField templates instead.
    """
    from openmm.app import ForceField, Modeller
    from binding_metrics.core.cyclic import (
        patch_cyclic_topology,
        get_addh_variants,
        load_extra_xmls,
        rename_disulfide_cys_to_cyx,
    )

    # custom bonds are intra-chain, so both ends share the same chain ID.
    cyclic_chain = custom_bonds[0][0]

    # patch_cyclic_topology detects the cyclic bond by distance (works even
    # without the bond already in the topology), removes C-terminal OXT and
    # terminal H atoms that PDBFixer added, and adds the N-C closure bond.
    topology, positions, bond_info = patch_cyclic_topology(topology, positions, cyclic_chain)

    if not bond_info:
        # Shouldn't happen if custom_bonds is non-empty, but fall back gracefully.
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(pH=ph)
        return modeller.topology, modeller.positions

    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    load_extra_xmls(ff, bond_info)

    # Rename SS-bonded CYS → CYX so addHydrogens uses the correct template.
    topology, positions = rename_disulfide_cys_to_cyx(topology, positions)

    modeller = Modeller(topology, positions)
    addh_variants = get_addh_variants(modeller.topology, bond_info, cyclic_chain)
    modeller.addHydrogens(ff, pH=ph, variants=addh_variants)
    return modeller.topology, modeller.positions


def prep_structure(
    topology,
    positions,
    ph: float = 7.4,
    keep_water: bool = False,
    canonicalize: bool = False,
) -> tuple:
    """Fix missing residues/atoms and add hydrogens in one PDBFixer pass.

    Args:
        topology: OpenMM Topology
        positions: OpenMM positions
        ph: pH for hydrogen placement (default 7.4)
        keep_water: If True, retain crystallographic water molecules
        canonicalize: If True, replace non-standard residues with their nearest
            standard equivalents (e.g. MSE→MET, SEP→SER, acetyl-LYS→LYS).
            If False (default), non-standard residues and non-canonical amino
            acids are preserved so they can be parameterised downstream with
            GAFF2 (``--small-molecules auto`` in binding-metrics-relax).

    Returns:
        Tuple of (topology, positions) with repaired and protonated structure
    """
    # Capture non-sequential intra-chain bonds (e.g. head-to-tail N→C) before
    # the PDBFixer round-trip drops them (PDBxFile.writeFile only writes SS bonds).
    custom_bonds = _extract_custom_bonds(topology)

    fixer = _topology_to_fixer(topology, positions)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()

    if canonicalize:
        fixer.replaceNonstandardResidues()
        print("  --canonicalize: non-standard residues replaced with standard equivalents.")
    else:
        nonstandard = getattr(fixer, "nonstandardResidues", [])
        if nonstandard:
            names = ", ".join(f"{r.name}" for r, _ in nonstandard)
            print(f"  Non-standard residues detected: {names}")
            print("  These will be preserved. Use --small-molecules auto in relax to")
            print("  parameterise them with GAFF2 (small organic molecules / modified AAs).")
            print("  Use --canonicalize to replace them with standard equivalents instead.")

    # Chain-aware heterogen filter — replaces PDBFixer's removeHeterogens():
    #   • Standard AA / nucleotide          → always keep
    #   • Metal ion (by element)            → always keep (ff14SB has parameters)
    #   • Non-standard residue in a protein chain → keep (non-canonical AA)
    #   • Water                             → keep if keep_water, else remove
    #   • Everything else (free ligands, crystallographic additives, glycans
    #     in their own chain …)             → remove
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    protein_chains: set = set()
    for chain in fixer.topology.chains():
        for res in chain.residues():
            if res.name in _STANDARD_RESIDUES:
                protein_chains.add(chain.id)
                break

    residues_to_remove = []
    kept_nonstandard: list = []
    removed_heterogens: list = []

    for chain in fixer.topology.chains():
        for res in chain.residues():
            if res.name in _STANDARD_RESIDUES:
                continue

            elements = {
                atom.element.symbol
                for atom in res.atoms()
                if atom.element is not None
            }
            if elements & _METAL_ELEMENTS:
                kept_nonstandard.append(f"{res.name} (metal, chain {chain.id})")
                continue

            if res.name in _WATER_NAMES:
                if not keep_water:
                    residues_to_remove.append(res)
                continue

            if chain.id in protein_chains:
                kept_nonstandard.append(f"{res.name} (chain {chain.id})")
                continue

            removed_heterogens.append(f"{res.name} (chain {chain.id})")
            residues_to_remove.append(res)

    if kept_nonstandard:
        print(f"  Kept non-standard residues: {', '.join(kept_nonstandard)}")
    if removed_heterogens:
        print(f"  Removed heterogens: {', '.join(removed_heterogens)}")

    if residues_to_remove:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(residues_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    if custom_bonds:
        # Use cyclic-aware H placement: patch_cyclic_topology (called inside)
        # removes C-terminal OXT that PDBFixer added, adds the N-C closure bond,
        # then uses cyclic FF templates for addHydrogens.  Do NOT restore bonds
        # here — patch_cyclic_topology detects and adds the bond itself.
        result_topo, result_pos = _add_hydrogens_cyclic(
            fixer.topology, fixer.positions, custom_bonds, ph
        )
    else:
        fixer.addMissingHydrogens(ph)
        result_topo, result_pos = fixer.topology, fixer.positions

    return result_topo, result_pos


def solvate(
    topology,
    positions,
    forcefield: ForceField | None = None,
    forcefield_name: Literal["amber", "charmm"] = "amber",
    padding: float = 1.0,
    ionic_strength: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
) -> Modeller:
    """Add explicit solvent and ions with periodic boundary conditions.

    Args:
        topology: OpenMM Topology
        positions: OpenMM positions
        forcefield: Pre-configured ForceField. If None, uses forcefield_name.
        forcefield_name: Force field to use if forcefield is None
        padding: Distance in nm between solute and box edge
        ionic_strength: Salt concentration in M
        positive_ion: Positive ion type
        negative_ion: Negative ion type

    Returns:
        Modeller with solvated and ionized system
    """
    if forcefield is None:
        forcefield = get_forcefield(forcefield_name)
    modeller = Modeller(topology, positions)
    modeller.addSolvent(
        forcefield,
        padding=padding * unit.nanometer,
        ionicStrength=ionic_strength * unit.molar,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
    )
    return modeller


def prepare_system(
    pdb: PDBFile,
    forcefield: ForceField | None = None,
    forcefield_name: Literal["amber", "charmm"] = "amber",
    padding: float = 1.0,
    ionic_strength: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    fix: bool = True,
    ph: float = 7.4,
) -> Modeller:
    """Prepare a molecular system for simulation: fix+protonate → solvate.

    Args:
        pdb: Loaded PDB file with the molecular structure
        forcefield: Pre-configured ForceField object. If None, uses forcefield_name.
        forcefield_name: Name of force field to use if forcefield is None
        padding: Distance in nm between solute and box edge
        ionic_strength: Salt concentration in M (molar)
        positive_ion: Positive ion type for neutralization
        negative_ion: Negative ion type for neutralization
        fix: If True and pdbfixer is available, fix missing atoms and protonate
        ph: pH for hydrogen placement when fix=True (default 7.4)

    Returns:
        Modeller object with solvated and ionized system
    """
    topology, positions = pdb.topology, pdb.positions

    if fix and HAS_PDBFIXER:
        topology, positions = prep_structure(topology, positions, ph=ph)
    else:
        ff = forcefield if forcefield is not None else get_forcefield(forcefield_name)
        tmp_modeller = Modeller(topology, positions)
        tmp_modeller.addHydrogens(ff)
        topology, positions = tmp_modeller.topology, tmp_modeller.positions

    return solvate(
        topology, positions,
        forcefield=forcefield,
        forcefield_name=forcefield_name,
        padding=padding,
        ionic_strength=ionic_strength,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
    )


def get_system_info(modeller: Modeller) -> dict:
    """Get information about the prepared system.

    Args:
        modeller: Prepared Modeller object

    Returns:
        Dictionary with system information
    """
    topology = modeller.topology
    n_atoms = topology.getNumAtoms()
    n_residues = topology.getNumResidues()
    n_chains = topology.getNumChains()

    # Count water molecules and ions
    n_waters = 0
    n_ions = 0
    ion_names = {"NA", "CL", "K", "MG", "CA", "ZN"}

    for residue in topology.residues():
        if residue.name == "HOH" or residue.name == "WAT":
            n_waters += 1
        elif residue.name in ion_names:
            n_ions += 1

    # Get box vectors
    box_vectors = modeller.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        box_size = [v[i].value_in_unit(unit.nanometer) for i, v in enumerate(box_vectors)]
    else:
        box_size = None

    return {
        "n_atoms": n_atoms,
        "n_residues": n_residues,
        "n_chains": n_chains,
        "n_waters": n_waters,
        "n_ions": n_ions,
        "box_size_nm": box_size,
    }
