"""Implicit solvent MD relaxation protocol for protein complexes.

Performs multi-stage energy minimization followed by an optional short MD
simulation using OpenMM with implicit solvent (OBC2 or GBn2). Designed for
fast GPU-accelerated evaluation of protein-peptide complexes.

Minimization stages:
    Stage 1: Initial global relaxation (resolves clashes)
    Stage 2: Backbone-restrained optimization (side chains optimize)
    Stage 3: Final unrestrained refinement

Usage:
    python -m binding_metrics.protocols.relaxation \\
        --input complex.cif \\
        --output-dir results/ \\
        --md-duration-ps 200
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class RelaxationConfig:
    """Configuration for implicit solvent MD relaxation.

    Attributes:
        min_steps_initial: Steps for initial global minimization (stage 1)
        min_steps_restrained: Steps for backbone-restrained minimization (stage 2)
        min_steps_final: Steps for final unrestrained minimization (stage 3)
        min_tolerance: Energy tolerance in kJ/mol/nm for final stage
        restraint_strength: Backbone restraint force constant in kJ/mol/nm²
        md_duration_ps: MD simulation duration in picoseconds (0 to skip)
        md_timestep_fs: MD integration timestep in femtoseconds
        md_temperature_k: Simulation temperature in Kelvin
        md_friction: Langevin friction coefficient in 1/ps
        md_save_interval_ps: Interval between saved trajectory frames in ps
        solvent_model: Implicit solvent model ('obc2', 'gbn2')
        device: Compute device ('cuda', 'cpu')
        peptide_chain_id: Peptide chain ID (auto-detect smallest chain if None)
        receptor_chain_id: Receptor chain ID (auto-detect largest chain if None)
        custom_bond_handler: Optional callable invoked after hydrogen addition.
            Signature: (topology, positions, peptide_chain) -> (topology, positions, bond_info)
            where bond_info is a list of tuples passed back to the caller for
            post-processing (e.g. harmonic restraints for custom bonds).
    """
    min_steps_initial: int = 1000
    min_steps_restrained: int = 500
    min_steps_final: int = 2000
    min_tolerance: float = 1.0
    restraint_strength: float = 100.0

    md_duration_ps: float = 200.0
    md_timestep_fs: float = 2.0
    md_temperature_k: float = 300.0
    md_friction: float = 1.0
    md_save_interval_ps: float = 10.0

    solvent_model: str = "obc2"
    device: str = "cuda"

    peptide_chain_id: Optional[str] = None
    receptor_chain_id: Optional[str] = None

    custom_bond_handler: Optional[Callable] = None


@dataclass
class RelaxationResult:
    """Results from an implicit solvent MD relaxation run.

    Attributes:
        sample_id: Identifier for the structure
        success: Whether the run completed without errors
        error_message: Error description if success is False
        potential_energy_minimized: Potential energy after minimization (kJ/mol)
        potential_energy_md_avg: Mean potential energy over MD trajectory (kJ/mol)
        potential_energy_md_std: Std of potential energy over MD trajectory (kJ/mol)
        rmsd_md_final: RMSD of final MD frame vs minimized structure (Angstroms)
        peptide_rmsf_mean: Mean per-residue RMSF of peptide over MD (Angstroms)
        peptide_rmsf_max: Max per-residue RMSF of peptide over MD (Angstroms)
        peptide_rmsf_per_residue: Per-residue RMSF list (Angstroms)
        minimization_time_s: Wall time for minimization in seconds
        md_time_s: Wall time for MD simulation in seconds
        minimized_structure_path: Path to saved minimized structure CIF
        md_final_structure_path: Path to saved final MD frame CIF
    """
    sample_id: str
    success: bool
    error_message: Optional[str] = None

    potential_energy_minimized: Optional[float] = None
    potential_energy_md_avg: Optional[float] = None
    potential_energy_md_std: Optional[float] = None

    rmsd_md_final: Optional[float] = None
    peptide_rmsf_mean: Optional[float] = None
    peptide_rmsf_max: Optional[float] = None
    peptide_rmsf_per_residue: Optional[list] = None

    minimization_time_s: Optional[float] = None
    md_time_s: Optional[float] = None

    minimized_structure_path: Optional[str] = None
    md_final_structure_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert result to a flat dictionary for CSV export."""
        d = {
            "sample_id": self.sample_id,
            "success": self.success,
            "error_message": self.error_message,
            "potential_energy_minimized": self.potential_energy_minimized,
            "potential_energy_md_avg": self.potential_energy_md_avg,
            "potential_energy_md_std": self.potential_energy_md_std,
            "rmsd_md_final": self.rmsd_md_final,
            "peptide_rmsf_mean": self.peptide_rmsf_mean,
            "peptide_rmsf_max": self.peptide_rmsf_max,
            "minimization_time_s": self.minimization_time_s,
            "md_time_s": self.md_time_s,
            "minimized_structure_path": self.minimized_structure_path,
            "md_final_structure_path": self.md_final_structure_path,
        }
        if self.peptide_rmsf_per_residue is not None:
            d["peptide_rmsf_per_residue"] = json.dumps(self.peptide_rmsf_per_residue)
        return d


class ImplicitRelaxation:
    """Implicit solvent MD relaxation for protein complexes.

    Runs multi-stage energy minimization followed by an optional short MD
    simulation using AMBER ff14SB with OBC2 or GBn2 implicit solvent.

    Structure preparation:
        - Removes atoms placed at the origin (0,0,0), which some structure
          prediction tools use as placeholders for unresolved side chains.
        - Rebuilds missing heavy atoms and adds hydrogens using PDBFixer.

    Minimization protocol (3 stages):
        Stage 1: Global relaxation (resolves clashes from side-chain rebuilding)
        Stage 2: Backbone-restrained (side chains optimize, backbone preserved)
        Stage 3: Final unrestrained refinement

    Example:
        >>> config = RelaxationConfig(md_duration_ps=200, device="cuda")
        >>> relaxer = ImplicitRelaxation(config)
        >>> result = relaxer.run(Path("complex.cif"), Path("output/"))
        >>> print(result.potential_energy_minimized)
    """

    def __init__(self, config: RelaxationConfig):
        self.config = config
        self._openmm_imported = False

    def _import_openmm(self):
        if self._openmm_imported:
            return
        global openmm, app, unit, PDBxFile
        try:
            import openmm as _openmm
            import openmm.unit as _unit
            from openmm import app as _app
            from openmm.app import PDBxFile as _PDBxFile
            openmm = _openmm
            app = _app
            unit = _unit
            PDBxFile = _PDBxFile
            self._openmm_imported = True
        except ImportError as e:
            raise ImportError(
                "OpenMM is required. Install with: conda install -c conda-forge openmm"
            ) from e

    def _identify_chains(self, topology) -> tuple[str, Optional[str]]:
        """Identify peptide (smallest) and receptor (largest) protein chains."""
        self._import_openmm()
        standard_residues = set(app.PDBFile._standardResidues)

        chain_sizes = []
        for chain in topology.chains():
            n_protein = sum(1 for r in chain.residues() if r.name in standard_residues)
            if n_protein > 0:
                chain_sizes.append((chain.id, n_protein))

        if not chain_sizes:
            raise ValueError("No protein chains found in structure")

        chain_sizes.sort(key=lambda x: x[1])

        peptide_chain = self.config.peptide_chain_id or chain_sizes[0][0]
        receptor_chain = (
            self.config.receptor_chain_id or (chain_sizes[-1][0] if len(chain_sizes) > 1 else None)
        )
        return peptide_chain, receptor_chain

    def _setup_system(self, input_path: Path):
        """Load structure, prepare topology, and create OpenMM system.

        Steps:
            1. Remove atoms at the origin (placeholder atoms)
            2. Rebuild missing heavy atoms with PDBFixer
            3. Add hydrogens at pH 7.0
            4. Apply custom_bond_handler if configured
            5. Create OpenMM system with implicit solvent

        Returns:
            Tuple of (system, topology, positions, bond_info)
        """
        import tempfile

        self._import_openmm()

        # --- Structure loading and repair ---
        try:
            from pdbfixer import PDBFixer

            fixer = PDBFixer(filename=str(input_path))

            # Remove origin placeholder atoms
            origin_indices = [
                i for i, pos in enumerate(fixer.positions)
                if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6
            ]
            if origin_indices:
                print(f"  Removing {len(origin_indices)} origin-placeholder atoms...")
                all_atoms = list(fixer.topology.atoms())
                atoms_to_delete = [all_atoms[i] for i in origin_indices]
                modeller = app.Modeller(fixer.topology, fixer.positions)
                modeller.delete(atoms_to_delete)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
                    app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp)
                    tmp_path = tmp.name
                fixer = PDBFixer(filename=tmp_path)
                Path(tmp_path).unlink(missing_ok=True)

            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            topology, positions = fixer.topology, fixer.positions

        except ImportError:
            print("  Note: PDBFixer not available, using basic loader")
            if input_path.suffix.lower() in (".cif", ".mmcif"):
                struct = PDBxFile(str(input_path))
            else:
                struct = app.PDBFile(str(input_path))
            topology, positions = struct.topology, struct.positions

            # Still remove origin atoms without PDBFixer
            modeller = app.Modeller(topology, positions)
            origin_atoms = [
                a for a, pos in zip(topology.atoms(), positions)
                if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6
            ]
            if origin_atoms:
                print(f"  Removing {len(origin_atoms)} origin-placeholder atoms...")
                modeller.delete(origin_atoms)
                topology, positions = modeller.topology, modeller.positions

        # --- Identify chains ---
        peptide_chain, receptor_chain = self._identify_chains(topology)

        # --- Force field setup ---
        gb_file = "implicit/gbn2.xml" if self.config.solvent_model == "gbn2" else "implicit/obc2.xml"
        base_xmls = ["amber14-all.xml", "amber14/tip3pfb.xml", gb_file]
        ff = app.ForceField(*base_xmls)

        # --- Add hydrogens ---
        print("  Adding hydrogens...")
        modeller = app.Modeller(topology, positions)
        try:
            modeller.addHydrogens(ff, pH=7.0)
        except Exception as e:
            print(f"  Warning: addHydrogens(pH=7.0) failed ({e}), retrying without pH...")
            try:
                modeller.addHydrogens(ff)
            except Exception as e2:
                print(f"  Warning: addHydrogens failed: {e2}")
        topology, positions = modeller.topology, modeller.positions

        # --- Custom bond handler (plugin hook) ---
        bond_info = []
        if self.config.custom_bond_handler is not None:
            topology, positions, bond_info = self.config.custom_bond_handler(
                topology, positions, peptide_chain
            )
            if bond_info:
                # Allow handler to supply additional XML files via bond_info metadata
                extra_xmls = getattr(bond_info, "extra_xmls", [])
                if extra_xmls:
                    ff = app.ForceField(*base_xmls, *extra_xmls)

        # --- Create system ---
        system = ff.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        return system, topology, positions, bond_info

    def _add_restraints(self, system, topology, positions, backbone_only: bool = True) -> int:
        """Add harmonic position restraints to the system.

        Args:
            system: OpenMM System
            topology: OpenMM Topology
            positions: Reference positions for restraints
            backbone_only: If True, only restrain backbone atoms (N, CA, C, O)

        Returns:
            Force index in the system
        """
        backbone_names = {"N", "CA", "C", "O"}
        restraint = openmm.CustomExternalForce(
            "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k",
            self.config.restraint_strength * unit.kilojoules_per_mole / unit.nanometer**2,
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        for atom in topology.atoms():
            if backbone_only and atom.name not in backbone_names:
                continue
            pos = positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])

        return system.addForce(restraint)

    def _compute_rmsd(self, positions1, positions2, atom_indices=None) -> float:
        """Compute Kabsch-aligned RMSD between two position sets.

        Args:
            positions1: Reference positions (OpenMM Quantity or list of Vec3)
            positions2: Target positions
            atom_indices: Subset of atoms to use (all if None)

        Returns:
            RMSD in Angstroms
        """
        pos1 = np.array([[p.x, p.y, p.z] for p in positions1])
        pos2 = np.array([[p.x, p.y, p.z] for p in positions2])

        if atom_indices is not None:
            pos1 = pos1[atom_indices]
            pos2 = pos2[atom_indices]

        pos1 -= pos1.mean(axis=0)
        pos2 -= pos2.mean(axis=0)

        H = pos1.T @ pos2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return float(np.sqrt(np.mean(np.sum((pos1 @ R - pos2) ** 2, axis=1))) * 10)

    def _compute_rmsf(self, trajectory_positions, atom_indices) -> np.ndarray:
        """Compute per-atom RMSF from a list of trajectory frame positions.

        Args:
            trajectory_positions: List of OpenMM position sets (one per frame)
            atom_indices: Atom indices to include

        Returns:
            Per-atom RMSF array in Angstroms
        """
        all_pos = (
            np.array([
                np.array([[p.x, p.y, p.z] for p in frame])[atom_indices]
                for frame in trajectory_positions
            ])
            * 10  # nm -> Angstroms
        )
        mean_pos = all_pos.mean(axis=0)
        return np.sqrt(np.mean((all_pos - mean_pos) ** 2, axis=0).sum(axis=1))

    def _get_platform(self):
        """Get the OpenMM compute platform."""
        self._import_openmm()
        if self.config.device == "cuda":
            try:
                platform = openmm.Platform.getPlatformByName("CUDA")
                properties = {"CudaPrecision": "mixed"}
                return platform, properties
            except Exception:
                pass
        return openmm.Platform.getPlatformByName("CPU"), {}

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        sample_id: Optional[str] = None,
    ) -> RelaxationResult:
        """Run implicit solvent MD relaxation on a single structure.

        Args:
            input_path: Path to input CIF or PDB file
            output_dir: Directory to write output structures
            sample_id: Identifier for this run (defaults to input file stem)

        Returns:
            RelaxationResult with energies, RMSD/RMSF, and output paths
        """
        from binding_metrics.io.structures import save_cif

        self._import_openmm()

        if sample_id is None:
            sample_id = input_path.stem

        result = RelaxationResult(sample_id=sample_id, success=False)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[{sample_id}] Preparing system...")
            system, topology, positions, bond_info = self._setup_system(input_path)

            peptide_chain, receptor_chain = self._identify_chains(topology)

            # Integrator
            integrator = openmm.LangevinMiddleIntegrator(
                self.config.md_temperature_k * unit.kelvin,
                self.config.md_friction / unit.picosecond,
                self.config.md_timestep_fs * unit.femtosecond,
            )

            platform, properties = self._get_platform()
            simulation = app.Simulation(topology, system, integrator, platform, properties)
            simulation.context.setPositions(positions)

            # --- Multi-stage minimization ---
            print(f"[{sample_id}] Minimizing (3 stages)...")
            min_start = time.time()

            print(f"[{sample_id}]   Stage 1: Global relaxation")
            simulation.minimizeEnergy(
                maxIterations=self.config.min_steps_initial,
                tolerance=self.config.min_tolerance * 10 * unit.kilojoules_per_mole / unit.nanometer,
            )

            print(f"[{sample_id}]   Stage 2: Backbone-restrained optimization")
            self._add_restraints(system, topology, positions, backbone_only=True)
            simulation.context.reinitialize(preserveState=True)
            simulation.minimizeEnergy(
                maxIterations=self.config.min_steps_restrained,
                tolerance=self.config.min_tolerance * 5 * unit.kilojoules_per_mole / unit.nanometer,
            )

            print(f"[{sample_id}]   Stage 3: Final unrestrained refinement")
            simulation.context.setParameter("k", 0.0)
            simulation.minimizeEnergy(
                maxIterations=self.config.min_steps_final,
                tolerance=self.config.min_tolerance * unit.kilojoules_per_mole / unit.nanometer,
            )

            state = simulation.context.getState(getEnergy=True, getPositions=True)
            result.potential_energy_minimized = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            minimized_positions = state.getPositions()
            result.minimization_time_s = time.time() - min_start

            # Save minimized structure
            min_path = output_dir / f"{sample_id}_minimized.cif"
            save_cif(topology, minimized_positions, min_path, source_cif_path=input_path)
            result.minimized_structure_path = str(min_path)
            print(f"[{sample_id}] Minimized: {result.potential_energy_minimized:.1f} kJ/mol")

            # --- MD simulation ---
            if self.config.md_duration_ps > 0:
                print(f"[{sample_id}] Running MD ({self.config.md_duration_ps} ps)...")
                md_start = time.time()
                simulation.context.setVelocitiesToTemperature(
                    self.config.md_temperature_k * unit.kelvin
                )

                steps_per_save = int(
                    self.config.md_save_interval_ps * 1000 / self.config.md_timestep_fs
                )
                total_saves = int(self.config.md_duration_ps / self.config.md_save_interval_ps)

                trajectory_positions = []
                md_energies = []
                for i in range(total_saves):
                    simulation.step(steps_per_save)
                    frame_state = simulation.context.getState(getPositions=True, getEnergy=True)
                    trajectory_positions.append(frame_state.getPositions())
                    md_energies.append(
                        frame_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    )

                result.md_time_s = time.time() - md_start
                result.potential_energy_md_avg = float(np.mean(md_energies))
                result.potential_energy_md_std = float(np.std(md_energies))

                final_positions = trajectory_positions[-1]
                result.rmsd_md_final = self._compute_rmsd(final_positions, minimized_positions)

                # RMSF for peptide CA atoms
                peptide_ca_indices = [
                    a.index
                    for a in topology.atoms()
                    if a.residue.chain.id == peptide_chain and a.name == "CA"
                ]
                if peptide_ca_indices:
                    rmsf = self._compute_rmsf(trajectory_positions, peptide_ca_indices)
                    result.peptide_rmsf_mean = float(rmsf.mean())
                    result.peptide_rmsf_max = float(rmsf.max())
                    result.peptide_rmsf_per_residue = rmsf.tolist()

                # Save final MD structure
                final_path = output_dir / f"{sample_id}_md_final.cif"
                save_cif(topology, final_positions, final_path, source_cif_path=input_path)
                result.md_final_structure_path = str(final_path)

            result.success = True

        except Exception as e:
            result.error_message = f"{type(e).__name__}: {e}"
            print(f"[{sample_id}] ERROR: {result.error_message}")
            traceback.print_exc()

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Implicit solvent MD relaxation for protein complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF or PDB file")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--md-duration-ps", type=float, default=200.0, help="MD duration in ps (0 to minimize only)")
    parser.add_argument("--md-save-interval-ps", type=float, default=10.0, help="Frame save interval in ps")
    parser.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature in K")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Compute device")
    parser.add_argument("--solvent-model", choices=["obc2", "gbn2"], default="obc2", help="Implicit solvent model")
    parser.add_argument("--peptide-chain", type=str, default=None, help="Peptide chain ID (auto-detect if omitted)")
    parser.add_argument("--receptor-chain", type=str, default=None, help="Receptor chain ID (auto-detect if omitted)")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample identifier (defaults to input file stem)")
    args = parser.parse_args()

    config = RelaxationConfig(
        md_duration_ps=args.md_duration_ps,
        md_save_interval_ps=args.md_save_interval_ps,
        md_temperature_k=args.temperature,
        device=args.device,
        solvent_model=args.solvent_model,
        peptide_chain_id=args.peptide_chain,
        receptor_chain_id=args.receptor_chain,
    )

    relaxer = ImplicitRelaxation(config)
    result = relaxer.run(args.input, args.output_dir, sample_id=args.sample_id)

    if result.success:
        print(f"\nSUCCESS")
        if result.minimized_structure_path:
            print(f"  Minimized: {result.minimized_structure_path}")
        if result.md_final_structure_path:
            print(f"  MD final:  {result.md_final_structure_path}")
    else:
        print(f"\nFAILED: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
