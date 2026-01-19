"""Interaction energy calculations."""

from pathlib import Path
from typing import Literal

import numpy as np

try:
    import mdtraj as md
except ImportError:
    md = None

import openmm
import openmm.unit as unit
from openmm.app import ForceField, PDBFile

from binding_metrics.core.forcefields import get_forcefield


def calculate_interaction_energy(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    forcefield_name: Literal["amber", "charmm"] = "amber",
) -> np.ndarray:
    """Calculate ligand-receptor interaction energy for each frame.

    The interaction energy is computed as:
        E_interaction = E_complex - E_ligand - E_receptor

    This captures the non-bonded (electrostatic + vdW) interaction
    between ligand and receptor.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        forcefield_name: Force field for energy calculation

    Returns:
        Array of interaction energies (kJ/mol) for each frame
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for energy calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))

    # Load topology for OpenMM
    pdb = PDBFile(str(topology_path))
    forcefield = get_forcefield(forcefield_name)

    # Create system for energy evaluation
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=None,
    )

    # Get the NonbondedForce
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("No NonbondedForce found in system")

    # Create sets for quick lookup
    ligand_set = set(ligand_indices)
    receptor_set = set(receptor_indices)

    # Calculate interaction energy for each frame
    interaction_energies = []

    for frame_idx in range(traj.n_frames):
        positions = traj.xyz[frame_idx]  # nm

        # Calculate pairwise interaction energy between ligand and receptor
        energy = 0.0

        for lig_idx in ligand_indices:
            # Get ligand atom parameters
            q1, sig1, eps1 = nonbonded_force.getParticleParameters(lig_idx)
            q1 = q1.value_in_unit(unit.elementary_charge)
            sig1 = sig1.value_in_unit(unit.nanometer)
            eps1 = eps1.value_in_unit(unit.kilojoule_per_mole)

            for rec_idx in receptor_indices:
                # Get receptor atom parameters
                q2, sig2, eps2 = nonbonded_force.getParticleParameters(rec_idx)
                q2 = q2.value_in_unit(unit.elementary_charge)
                sig2 = sig2.value_in_unit(unit.nanometer)
                eps2 = eps2.value_in_unit(unit.kilojoule_per_mole)

                # Distance
                r = np.linalg.norm(positions[lig_idx] - positions[rec_idx])

                if r < 0.01:  # Avoid division by zero
                    continue

                # Coulomb energy (in kJ/mol)
                # E = k * q1 * q2 / r, k = 138.935... kJ*nm/(mol*e^2)
                coulomb_k = 138.935456
                e_coulomb = coulomb_k * q1 * q2 / r

                # Lennard-Jones energy
                sigma = (sig1 + sig2) / 2
                epsilon = np.sqrt(eps1 * eps2)
                if epsilon > 0 and sigma > 0:
                    sr6 = (sigma / r) ** 6
                    e_lj = 4 * epsilon * (sr6 ** 2 - sr6)
                else:
                    e_lj = 0.0

                energy += e_coulomb + e_lj

        interaction_energies.append(energy)

    return np.array(interaction_energies)


def calculate_component_energies(
    trajectory_path: str | Path,
    topology_path: str | Path,
    ligand_indices: list[int],
    receptor_indices: list[int],
    forcefield_name: Literal["amber", "charmm"] = "amber",
) -> dict[str, np.ndarray]:
    """Calculate separated electrostatic and vdW interaction energies.

    Args:
        trajectory_path: Path to trajectory file
        topology_path: Path to topology file
        ligand_indices: Atom indices of the ligand
        receptor_indices: Atom indices of the receptor
        forcefield_name: Force field for energy calculation

    Returns:
        Dictionary with 'electrostatic', 'vdw', and 'total' energy arrays
    """
    if md is None:
        raise ImportError(
            "mdtraj is required for energy calculations. "
            "Install with: pip install binding-metrics[analysis]"
        )

    traj = md.load(str(trajectory_path), top=str(topology_path))
    pdb = PDBFile(str(topology_path))
    forcefield = get_forcefield(forcefield_name)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=None,
    )

    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("No NonbondedForce found in system")

    coulomb_energies = []
    lj_energies = []

    coulomb_k = 138.935456

    for frame_idx in range(traj.n_frames):
        positions = traj.xyz[frame_idx]
        e_coulomb_total = 0.0
        e_lj_total = 0.0

        for lig_idx in ligand_indices:
            q1, sig1, eps1 = nonbonded_force.getParticleParameters(lig_idx)
            q1 = q1.value_in_unit(unit.elementary_charge)
            sig1 = sig1.value_in_unit(unit.nanometer)
            eps1 = eps1.value_in_unit(unit.kilojoule_per_mole)

            for rec_idx in receptor_indices:
                q2, sig2, eps2 = nonbonded_force.getParticleParameters(rec_idx)
                q2 = q2.value_in_unit(unit.elementary_charge)
                sig2 = sig2.value_in_unit(unit.nanometer)
                eps2 = eps2.value_in_unit(unit.kilojoule_per_mole)

                r = np.linalg.norm(positions[lig_idx] - positions[rec_idx])
                if r < 0.01:
                    continue

                e_coulomb_total += coulomb_k * q1 * q2 / r

                sigma = (sig1 + sig2) / 2
                epsilon = np.sqrt(eps1 * eps2)
                if epsilon > 0 and sigma > 0:
                    sr6 = (sigma / r) ** 6
                    e_lj_total += 4 * epsilon * (sr6 ** 2 - sr6)

        coulomb_energies.append(e_coulomb_total)
        lj_energies.append(e_lj_total)

    elec = np.array(coulomb_energies)
    vdw = np.array(lj_energies)

    return {
        "electrostatic": elec,
        "vdw": vdw,
        "total": elec + vdw,
    }
