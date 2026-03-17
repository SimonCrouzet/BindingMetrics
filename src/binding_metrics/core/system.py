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


def _topology_to_fixer(topology, positions) -> "PDBFixer":
    """Write topology+positions to a temp PDB and load with PDBFixer."""
    if not HAS_PDBFIXER:
        raise ImportError(
            "pdbfixer is required. Install with: pip install binding-metrics[structure]"
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        PDBFile.writeFile(topology, positions, tmp)
        tmp_path = tmp.name
    try:
        fixer = PDBFixer(filename=tmp_path)
    finally:
        os.unlink(tmp_path)
    return fixer


def prep_structure(topology, positions, ph: float = 7.4, keep_water: bool = False) -> tuple:
    """Fix missing residues/atoms and add hydrogens in one PDBFixer pass.

    Args:
        topology: OpenMM Topology
        positions: OpenMM positions
        ph: pH for hydrogen placement (default 7.4)
        keep_water: If True, retain crystallographic water molecules

    Returns:
        Tuple of (topology, positions) with repaired and protonated structure
    """
    fixer = _topology_to_fixer(topology, positions)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    return fixer.topology, fixer.positions


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
