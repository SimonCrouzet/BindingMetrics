"""OpenMM simulation engine for binding metrics evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import openmm
import openmm.unit as unit
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, System
from openmm.app import (
    DCDReporter,
    ForceField,
    Modeller,
    PDBFile,
    Simulation,
    StateDataReporter,
)

from binding_metrics.core.forcefields import get_forcefield
from binding_metrics.core.system import prepare_system


@dataclass
class SimulationConfig:
    """Configuration for MD simulation.

    Attributes:
        temperature: Simulation temperature in Kelvin
        pressure: Simulation pressure in bar (None for NVT)
        timestep: Integration timestep in femtoseconds
        duration_ns: Total simulation duration in nanoseconds
        equilibration_ns: Equilibration duration in nanoseconds
        save_interval_ps: Interval for saving trajectory frames in picoseconds
        friction: Langevin friction coefficient in 1/ps
        nonbonded_cutoff: Nonbonded interaction cutoff in nm
        constraints: Constraint type for bonds
        platform: Compute platform (CUDA, OpenCL, CPU, or auto)
    """

    temperature: float = 300.0
    pressure: float | None = 1.0
    timestep: float = 2.0
    duration_ns: float = 10.0
    equilibration_ns: float = 0.1
    save_interval_ps: float = 10.0
    friction: float = 1.0
    nonbonded_cutoff: float = 1.0
    constraints: Literal["none", "hbonds", "allbonds"] = "hbonds"
    platform: Literal["CUDA", "OpenCL", "CPU", "auto"] = "auto"


class MDSimulation:
    """Molecular dynamics simulation runner using OpenMM.

    Attributes:
        config: Simulation configuration
        modeller: Prepared system (topology + positions)
        forcefield: Force field for the simulation
        simulation: OpenMM Simulation object (after setup)
    """

    def __init__(
        self,
        modeller: Modeller,
        forcefield: ForceField,
        config: SimulationConfig | None = None,
    ):
        """Initialize the MD simulation.

        Args:
            modeller: Prepared Modeller object with solvated system
            forcefield: OpenMM ForceField object
            config: Simulation configuration (uses defaults if None)
        """
        self.config = config or SimulationConfig()
        self.modeller = modeller
        self.forcefield = forcefield
        self.simulation: Simulation | None = None
        self._system: System | None = None

    def setup(self) -> None:
        """Set up the simulation system and integrator."""
        config = self.config

        # Map constraint type
        constraint_map = {
            "none": None,
            "hbonds": openmm.app.HBonds,
            "allbonds": openmm.app.AllBonds,
        }

        # Create system
        self._system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=config.nonbonded_cutoff * unit.nanometer,
            constraints=constraint_map[config.constraints],
        )

        # Add barostat for NPT
        if config.pressure is not None:
            barostat = MonteCarloBarostat(
                config.pressure * unit.bar,
                config.temperature * unit.kelvin,
            )
            self._system.addForce(barostat)

        # Create integrator
        integrator = LangevinMiddleIntegrator(
            config.temperature * unit.kelvin,
            config.friction / unit.picosecond,
            config.timestep * unit.femtosecond,
        )

        # Select platform
        platform = self._get_platform()

        # Create simulation
        self.simulation = Simulation(
            self.modeller.topology,
            self._system,
            integrator,
            platform,
        )
        self.simulation.context.setPositions(self.modeller.positions)

    def _get_platform(self) -> Platform:
        """Get the compute platform based on configuration."""
        if self.config.platform == "auto":
            # Try platforms in order of preference
            for name in ["CUDA", "OpenCL", "CPU"]:
                try:
                    return Platform.getPlatformByName(name)
                except Exception:
                    continue
            return Platform.getPlatformByName("Reference")
        return Platform.getPlatformByName(self.config.platform)

    def minimize(self, max_iterations: int = 1000, tolerance: float = 10.0) -> None:
        """Perform energy minimization.

        Args:
            max_iterations: Maximum minimization steps (0 for unlimited)
            tolerance: Energy tolerance in kJ/mol/nm
        """
        if self.simulation is None:
            raise RuntimeError("Call setup() before minimize()")

        self.simulation.minimizeEnergy(
            maxIterations=max_iterations,
            tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
        )

    def equilibrate(self) -> None:
        """Run equilibration phase."""
        if self.simulation is None:
            raise RuntimeError("Call setup() before equilibrate()")

        config = self.config
        n_steps = int(config.equilibration_ns * 1e6 / config.timestep)
        self.simulation.step(n_steps)

    def run(
        self,
        output_dir: str | Path,
        trajectory_file: str = "trajectory.dcd",
        state_file: str = "state.csv",
    ) -> Path:
        """Run the production simulation.

        Args:
            output_dir: Directory to save output files
            trajectory_file: Name of trajectory file
            state_file: Name of state data file

        Returns:
            Path to the trajectory file
        """
        if self.simulation is None:
            raise RuntimeError("Call setup() before run()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        traj_path = output_dir / trajectory_file
        state_path = output_dir / state_file

        # Calculate steps
        n_steps = int(config.duration_ns * 1e6 / config.timestep)
        save_interval = int(config.save_interval_ps * 1000 / config.timestep)

        # Add reporters
        self.simulation.reporters.append(
            DCDReporter(str(traj_path), save_interval)
        )
        self.simulation.reporters.append(
            StateDataReporter(
                str(state_path),
                save_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            )
        )

        # Run simulation
        self.simulation.step(n_steps)

        return traj_path

    def get_positions(self) -> np.ndarray:
        """Get current positions as numpy array in nm."""
        if self.simulation is None:
            raise RuntimeError("Call setup() before get_positions()")

        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        return positions.value_in_unit(unit.nanometer)

    def get_potential_energy(self) -> float:
        """Get current potential energy in kJ/mol."""
        if self.simulation is None:
            raise RuntimeError("Call setup() before get_potential_energy()")

        state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoule_per_mole)

    @property
    def system(self) -> System | None:
        """Get the OpenMM System object."""
        return self._system


def run_simulation(
    pdb_path: str | Path,
    output_dir: str | Path,
    forcefield_name: Literal["amber", "charmm"] = "amber",
    config: SimulationConfig | None = None,
) -> Path:
    """Convenience function to run a complete simulation.

    Args:
        pdb_path: Path to input PDB file
        output_dir: Directory to save output files
        forcefield_name: Force field to use
        config: Simulation configuration

    Returns:
        Path to the trajectory file
    """
    # Load structure
    pdb = PDBFile(str(pdb_path))
    forcefield = get_forcefield(forcefield_name)

    # Prepare system
    modeller = prepare_system(pdb, forcefield=forcefield)

    # Run simulation
    sim = MDSimulation(modeller, forcefield, config)
    sim.setup()
    sim.minimize()
    sim.equilibrate()

    return sim.run(output_dir)
