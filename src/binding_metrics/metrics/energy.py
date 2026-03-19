"""Interaction energy calculations."""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Literal, Optional

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


# ---------------------------------------------------------------------------
# Subsystem decomposition: E_complex - E_peptide - E_receptor
# ---------------------------------------------------------------------------


def _repair_structure(topology, positions):
    """Repair a structure using PDBFixer if available.

    Removes origin-placeholder atoms and rebuilds missing heavy atoms.
    Falls back to the original topology/positions if PDBFixer is not installed.

    Returns:
        (topology, positions) — repaired if PDBFixer available, original otherwise.
    """
    try:
        import tempfile
        import os
        from pdbfixer import PDBFixer
        from openmm.app import Modeller

        # Write to temp PDB so PDBFixer can read it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            PDBFile.writeFile(topology, positions, tmp)
            tmp_path = tmp.name
        try:
            fixer = PDBFixer(filename=tmp_path)
        finally:
            os.unlink(tmp_path)

        # Remove origin-placeholder atoms (structure prediction artefacts)
        origin_indices = [
            i for i, pos in enumerate(fixer.positions)
            if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6
        ]
        if origin_indices:
            all_atoms = list(fixer.topology.atoms())
            modeller = Modeller(fixer.topology, fixer.positions)
            modeller.delete([all_atoms[i] for i in origin_indices])
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
                PDBFile.writeFile(modeller.topology, modeller.positions, tmp)
                tmp_path = tmp.name
            try:
                fixer = PDBFixer(filename=tmp_path)
            finally:
                os.unlink(tmp_path)

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        return fixer.topology, fixer.positions

    except ImportError:
        return topology, positions


def _create_implicit_system(topology, positions, solvent_model: str = "obc2"):
    """Create an OpenMM system with implicit solvent after adding hydrogens.

    Returns:
        Tuple of (system, topology_with_h, positions_with_h)
    """
    gb_file = "implicit/gbn2.xml" if solvent_model == "gbn2" else "implicit/obc2.xml"
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", gb_file)

    from openmm.app import Modeller
    modeller = Modeller(topology, positions)
    try:
        modeller.addHydrogens(ff, pH=7.0)
    except Exception:
        modeller.addHydrogens(ff)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=openmm.app.HBonds,
    )
    return system, modeller.topology, modeller.positions


def _build_subsystem(topology, solvent_model: str = "obc2"):
    """Build an OpenMM system for a topology that already contains hydrogens."""
    gb_file = "implicit/gbn2.xml" if solvent_model == "gbn2" else "implicit/obc2.xml"
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml", gb_file)
    return ff.createSystem(
        topology,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=openmm.app.HBonds,
    )


def _get_platform(device: str = "cuda"):
    """Get the best available OpenMM platform."""
    if device == "cuda":
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            return platform, {"CudaPrecision": "mixed"}
        except Exception:
            pass
    return openmm.Platform.getPlatformByName("CPU"), {}


def _evaluate_potential_energy(
    system, topology, positions, device: str = "cuda", min_iterations: int = 0
) -> float:
    """Evaluate potential energy for a system at given positions.

    Args:
        min_iterations: Optional brief minimization before evaluation (0 = none).
    """
    from openmm.app import Simulation
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = _get_platform(device)
    sim = Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    if min_iterations > 0:
        sim.minimizeEnergy(maxIterations=min_iterations)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def _extract_chain(topology, positions, chain_id: str):
    """Extract a single chain as a new (topology, positions) pair."""
    from openmm.app import Topology
    new_topology = Topology()
    new_positions = []
    old_to_new: dict[int, object] = {}

    for chain in topology.chains():
        if chain.id == chain_id:
            new_chain = new_topology.addChain(chain.id)
            for residue in chain.residues():
                new_residue = new_topology.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    new_atom = new_topology.addAtom(atom.name, atom.element, new_residue)
                    old_to_new[atom.index] = new_atom
                    new_positions.append(positions[atom.index])

    # Copy bonds that are entirely within the extracted chain
    for bond in topology.bonds():
        a1, a2 = bond.atom1, bond.atom2
        if a1.index in old_to_new and a2.index in old_to_new:
            new_topology.addBond(old_to_new[a1.index], old_to_new[a2.index])

    new_positions = unit.Quantity(
        np.array([[p.x, p.y, p.z] for p in new_positions]),
        unit.nanometers,
    )
    return new_topology, new_positions


def _evaluate_subsystem_energies(
    simulation,
    topo_h,
    positions,
    peptide_chain: str,
    receptor_chain: str,
    solvent_model: str,
    device: str,
) -> tuple:
    """Evaluate E_complex, E_peptide, E_receptor at given positions.

    Uses the existing simulation context for complex energy (avoids rebuilding
    the system). Creates fresh subsystem simulations for peptide and receptor.

    Returns:
        (e_complex, e_peptide, e_receptor) with None values on failure.
    """
    try:
        simulation.context.setPositions(positions)
        e_c = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        if not np.isfinite(e_c):
            return None, None, None

        pep_topo, pep_pos = _extract_chain(topo_h, positions, peptide_chain)
        sys_p = _build_subsystem(pep_topo, solvent_model)
        e_p = _evaluate_potential_energy(sys_p, pep_topo, pep_pos, device)

        rec_topo, rec_pos = _extract_chain(topo_h, positions, receptor_chain)
        sys_r = _build_subsystem(rec_topo, solvent_model)
        e_r = _evaluate_potential_energy(sys_r, rec_topo, rec_pos, device)

        if not (np.isfinite(e_p) and np.isfinite(e_r)):
            return e_c, None, None

        return e_c, e_p, e_r

    except Exception:
        return None, None, None


def compute_interaction_energy(
    input_path: str | Path,
    peptide_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    solvent_model: str = "obc2",
    device: str = "cuda",
    sample_id: Optional[str] = None,
    modes: tuple = ("raw", "relaxed", "after_md"),
    relaxed_min_steps_restrained: int = 500,
    relaxed_min_steps_full: int = 2000,
    after_md_duration_ps: float = 10.0,
    after_md_timestep_fs: float = 2.0,
    after_md_temperature_k: float = 300.0,
) -> dict:
    """Compute interaction energies via subsystem decomposition for multiple modes.

    For each mode, computes E_interaction = E_complex - E_peptide - E_receptor
    using AMBER ff14SB + implicit solvent (OBC2 or GBn2). System preparation
    (hydrogen addition) is performed once and shared across modes.

    Modes:
        raw:      Evaluate at the H-added input geometry. May return None for
                  structures with severe clashes (useful as a clash indicator).
        relaxed:  Backbone-restrained minimization → full unrestrained minimization,
                  then evaluate. Resolves clashes while preserving backbone geometry.
        after_md: From the relaxed geometry, run short MD and evaluate at final frame.

    Args:
        input_path: Path to CIF or PDB structure file
        peptide_chain: Peptide chain ID. Auto-detected if None (smallest chain).
        receptor_chain: Receptor chain ID. Auto-detected if None (largest chain).
        solvent_model: Implicit solvent model ('obc2' or 'gbn2')
        device: Compute device ('cuda' or 'cpu')
        sample_id: Identifier for this computation (defaults to file stem)
        modes: Tuple of modes to compute. Subset of ('raw', 'relaxed', 'after_md').
        relaxed_min_steps_restrained: Backbone-restrained minimization steps.
        relaxed_min_steps_full: Unrestrained minimization steps.
        after_md_duration_ps: Short MD duration in picoseconds.
        after_md_timestep_fs: MD timestep in femtoseconds.
        after_md_temperature_k: MD temperature in Kelvin.

    Returns:
        Flat dictionary with keys:
            sample_id, success, error_message, num_contacts, num_close_contacts,
            {mode}_interaction_energy, {mode}_e_complex, {mode}_e_peptide,
            {mode}_e_receptor  —  for each requested mode.
    """
    from binding_metrics.io.structures import detect_chains, load_structure

    input_path = Path(input_path)
    if sample_id is None:
        sample_id = input_path.stem

    result: dict = {
        "sample_id": sample_id,
        "success": False,
        "error_message": None,
        "num_contacts": None,
        "num_close_contacts": None,
    }
    for mode in modes:
        result[f"{mode}_interaction_energy"] = None
        result[f"{mode}_e_complex"] = None
        result[f"{mode}_e_peptide"] = None
        result[f"{mode}_e_receptor"] = None

    try:
        topology, positions = load_structure(input_path)
        topology, positions = _repair_structure(topology, positions)

        if peptide_chain is None or receptor_chain is None:
            auto_pep, auto_rec = detect_chains(topology)
            peptide_chain = peptide_chain or auto_pep
            receptor_chain = receptor_chain or auto_rec

        # Strip heterogens (non-protein residues) before system creation
        from binding_metrics.io.structures import strip_heterogens
        topology, positions = strip_heterogens(topology, positions, peptide_chain, receptor_chain)

        if peptide_chain is None or receptor_chain is None:
            raise ValueError("Could not identify two protein chains in structure")

        print(f"[{sample_id}] Chains: peptide={peptide_chain}, receptor={receptor_chain}")

        # Contact counts from original heavy-atom positions (before H addition)
        pos_array = np.array([[p.x, p.y, p.z] for p in positions]) * 10  # nm -> Å
        pep_indices = [a.index for a in topology.atoms() if a.residue.chain.id == peptide_chain]
        rec_indices = [a.index for a in topology.atoms() if a.residue.chain.id == receptor_chain]
        if pep_indices and rec_indices:
            distances = np.linalg.norm(
                pos_array[pep_indices][:, np.newaxis, :] - pos_array[rec_indices][np.newaxis, :, :],
                axis=-1,
            )
            result["num_contacts"] = int(np.sum(distances < 8.0))
            result["num_close_contacts"] = int(np.sum(distances < 4.0))

        # Build complex system — adds hydrogens once, shared across all modes
        sys_complex, topo_h, pos_h = _create_implicit_system(topology, positions, solvent_model)
        platform, props = _get_platform(device)

        # Single simulation object used for all modes (sequential: raw → relaxed → after_md)
        integrator = openmm.LangevinMiddleIntegrator(
            after_md_temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            after_md_timestep_fs * unit.femtosecond,
        )
        simulation = openmm.app.Simulation(topo_h, sys_complex, integrator, platform, props)
        simulation.context.setPositions(pos_h)

        any_success = False

        # --- RAW mode ---
        if "raw" in modes:
            print(f"[{sample_id}] Raw mode...")
            e_c, e_p, e_r = _evaluate_subsystem_energies(
                simulation, topo_h, pos_h, peptide_chain, receptor_chain, solvent_model, device
            )
            if e_c is not None and e_p is not None and e_r is not None:
                result["raw_e_complex"] = e_c
                result["raw_e_peptide"] = e_p
                result["raw_e_receptor"] = e_r
                result["raw_interaction_energy"] = e_c - e_p - e_r
                any_success = True
                print(f"[{sample_id}]   E_int(raw) = {result['raw_interaction_energy']:.1f} kJ/mol")
            else:
                print(f"[{sample_id}]   Raw evaluation returned NaN (likely clashes)")

        # --- RELAXED mode (also prepares state for after_md) ---
        pos_relaxed = pos_h
        if "relaxed" in modes or "after_md" in modes:
            print(f"[{sample_id}] Minimizing (backbone-restrained + unrestrained)...")
            try:
                backbone_names = {"N", "CA", "C", "O"}
                restraint = openmm.CustomExternalForce(
                    "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
                )
                restraint.addGlobalParameter(
                    "k", 100.0 * unit.kilojoules_per_mole / unit.nanometer**2
                )
                restraint.addPerParticleParameter("x0")
                restraint.addPerParticleParameter("y0")
                restraint.addPerParticleParameter("z0")
                for atom in topo_h.atoms():
                    if atom.name in backbone_names:
                        pos = pos_h[atom.index]
                        restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                sys_complex.addForce(restraint)
                simulation.context.reinitialize(preserveState=True)

                simulation.minimizeEnergy(maxIterations=relaxed_min_steps_restrained)
                simulation.context.setParameter("k", 0.0)
                simulation.minimizeEnergy(maxIterations=relaxed_min_steps_full)

                state = simulation.context.getState(getPositions=True)
                pos_relaxed = state.getPositions()

                if "relaxed" in modes:
                    e_c, e_p, e_r = _evaluate_subsystem_energies(
                        simulation, topo_h, pos_relaxed, peptide_chain, receptor_chain,
                        solvent_model, device,
                    )
                    if e_c is not None and e_p is not None and e_r is not None:
                        result["relaxed_e_complex"] = e_c
                        result["relaxed_e_peptide"] = e_p
                        result["relaxed_e_receptor"] = e_r
                        result["relaxed_interaction_energy"] = e_c - e_p - e_r
                        any_success = True
                        print(
                            f"[{sample_id}]   E_int(relaxed) = "
                            f"{result['relaxed_interaction_energy']:.1f} kJ/mol"
                        )
            except Exception as e:
                print(f"[{sample_id}] Warning: relaxed/minimization failed: {e}")

        # --- AFTER_MD mode ---
        if "after_md" in modes:
            print(f"[{sample_id}] Running MD ({after_md_duration_ps} ps)...")
            try:
                simulation.context.setPositions(pos_relaxed)
                simulation.context.setVelocitiesToTemperature(
                    after_md_temperature_k * unit.kelvin
                )
                n_steps = int(after_md_duration_ps * 1000 / after_md_timestep_fs)
                simulation.step(n_steps)

                state = simulation.context.getState(getPositions=True)
                pos_md = state.getPositions()

                e_c, e_p, e_r = _evaluate_subsystem_energies(
                    simulation, topo_h, pos_md, peptide_chain, receptor_chain,
                    solvent_model, device,
                )
                if e_c is not None and e_p is not None and e_r is not None:
                    result["after_md_e_complex"] = e_c
                    result["after_md_e_peptide"] = e_p
                    result["after_md_e_receptor"] = e_r
                    result["after_md_interaction_energy"] = e_c - e_p - e_r
                    any_success = True
                    print(
                        f"[{sample_id}]   E_int(after_md) = "
                        f"{result['after_md_interaction_energy']:.1f} kJ/mol"
                    )
            except Exception as e:
                print(f"[{sample_id}] Warning: after_md failed: {e}")

        result["success"] = any_success

    except Exception as e:
        result["error_message"] = f"{type(e).__name__}: {e}"
        print(f"[{sample_id}] ERROR: {result['error_message']}")
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute interaction energy via subsystem decomposition (implicit solvent)"
    )
    parser.add_argument("--input", "-i", type=Path, help="Single input structure file")
    parser.add_argument("--input-dir", type=Path, help="Directory of structure files")
    parser.add_argument("--glob-pattern", default="*.cif", help="Glob pattern for --input-dir")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV path")
    parser.add_argument("--peptide-chain", type=str, default=None)
    parser.add_argument("--receptor-chain", type=str, default=None)
    parser.add_argument("--solvent-model", choices=["obc2", "gbn2"], default="obc2")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--modes", nargs="+", default=["raw", "relaxed", "after_md"],
        choices=["raw", "relaxed", "after_md"],
        help="Which modes to compute (default: all three)",
    )
    parser.add_argument("--relaxed-min-steps-restrained", type=int, default=500)
    parser.add_argument("--relaxed-min-steps-full", type=int, default=2000)
    parser.add_argument("--after-md-duration-ps", type=float, default=10.0)
    parser.add_argument("--after-md-timestep-fs", type=float, default=2.0)
    parser.add_argument("--after-md-temperature-k", type=float, default=300.0)
    args = parser.parse_args()

    import pandas as pd

    if args.input:
        input_files = [Path(args.input)]
    elif args.input_dir:
        input_files = sorted(Path(args.input_dir).glob(args.glob_pattern))
    else:
        parser.error("Must specify --input or --input-dir")

    results = []
    for path in input_files:
        r = compute_interaction_energy(
            path,
            peptide_chain=args.peptide_chain,
            receptor_chain=args.receptor_chain,
            solvent_model=args.solvent_model,
            device=args.device,
            modes=tuple(args.modes),
            relaxed_min_steps_restrained=args.relaxed_min_steps_restrained,
            relaxed_min_steps_full=args.relaxed_min_steps_full,
            after_md_duration_ps=args.after_md_duration_ps,
            after_md_timestep_fs=args.after_md_timestep_fs,
            after_md_temperature_k=args.after_md_temperature_k,
        )
        results.append(r)

    df = pd.DataFrame(results)
    if args.output:
        df.to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nSaved to: {args.output}")
    else:
        print("\nResults:")
        print(df.to_string())


if __name__ == "__main__":
    main()
