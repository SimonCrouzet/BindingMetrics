"""System preparation utilities for MD simulations."""

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


def prepare_system(
    pdb: PDBFile,
    forcefield: ForceField | None = None,
    forcefield_name: Literal["amber", "charmm"] = "amber",
    padding: float = 1.0,
    ionic_strength: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    fix_structure: bool = True,
) -> Modeller:
    """Prepare a molecular system for simulation by adding solvent and ions.

    Args:
        pdb: Loaded PDB file with the molecular structure
        forcefield: Pre-configured ForceField object. If None, uses forcefield_name.
        forcefield_name: Name of force field to use if forcefield is None
        padding: Distance in nm between solute and box edge
        ionic_strength: Salt concentration in M (molar)
        positive_ion: Positive ion type for neutralization
        negative_ion: Negative ion type for neutralization
        fix_structure: If True and pdbfixer is available, fix missing atoms

    Returns:
        Modeller object with solvated and ionized system
    """
    if forcefield is None:
        forcefield = get_forcefield(forcefield_name)

    topology = pdb.topology
    positions = pdb.positions

    # Use pdbfixer to repair structure if available and requested
    if fix_structure and HAS_PDBFIXER:
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            PDBFile.writeFile(topology, positions, tmp)
            tmp_path = tmp.name
        try:
            fixer = PDBFixer(filename=tmp_path)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)  # pH 7.0
            topology = fixer.topology
            positions = fixer.positions
        finally:
            os.unlink(tmp_path)

    modeller = Modeller(topology, positions)

    # Add hydrogen atoms if missing (pdbfixer may have already added them)
    if not (fix_structure and HAS_PDBFIXER):
        modeller.addHydrogens(forcefield)

    # Add solvent with periodic box
    modeller.addSolvent(
        forcefield,
        padding=padding * unit.nanometer,
        ionicStrength=ionic_strength * unit.molar,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
    )

    return modeller


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
