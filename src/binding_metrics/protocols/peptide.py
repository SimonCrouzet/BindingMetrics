"""Peptide binding protocol for evaluating peptide-protein interactions."""

from pathlib import Path
from typing import Literal

import numpy as np
from openmm.app import PDBFile

from binding_metrics.core.forcefields import get_forcefield
from binding_metrics.core.simulation import MDSimulation, SimulationConfig
from binding_metrics.core.system import prepare_system
from binding_metrics.io.structures import load_complex, get_chain_atom_indices
from binding_metrics.metrics.contacts import calculate_contacts
from binding_metrics.metrics.energy import calculate_interaction_energy
from binding_metrics.metrics.rmsd import calculate_rmsd
from binding_metrics.metrics.sasa import calculate_buried_sasa
from binding_metrics.protocols.base import BaseProtocol, ProtocolResults


class PeptideBindingProtocol(BaseProtocol):
    """Protocol for evaluating peptide binding to a receptor.

    This protocol runs an MD simulation of a pre-docked peptide-receptor
    complex and computes key binding metrics including buried SASA,
    interface contacts, interaction energy, and structural stability.

    Example:
        >>> protocol = PeptideBindingProtocol(
        ...     pdb_path="complex.pdb",
        ...     ligand_chain="B",
        ...     receptor_chains=["A"],
        ...     forcefield="amber",
        ... )
        >>> protocol.run(output_dir="./results")
        >>> results = protocol.analyze()
        >>> print(results.summary())
    """

    def __init__(
        self,
        pdb_path: str | Path,
        ligand_chain: str,
        receptor_chains: list[str],
        forcefield: Literal["amber", "charmm"] = "amber",
        simulation_config: SimulationConfig | None = None,
    ):
        """Initialize the peptide binding protocol.

        Args:
            pdb_path: Path to the input PDB file (pre-docked complex)
            ligand_chain: Chain ID of the peptide ligand
            receptor_chains: List of chain IDs for the receptor
            forcefield: Force field to use ('amber' or 'charmm')
            simulation_config: Custom simulation configuration. If None,
                uses default 10ns quick evaluation settings.
        """
        super().__init__(pdb_path, ligand_chain, receptor_chains)
        self.forcefield_name = forcefield
        self.simulation_config = simulation_config or SimulationConfig()
        self._pdb: PDBFile | None = None
        self._simulation: MDSimulation | None = None
        self._topology_path: Path | None = None

    def run(
        self,
        output_dir: str | Path,
        minimize: bool = True,
        equilibrate: bool = True,
    ) -> Path:
        """Run the MD simulation.

        Args:
            output_dir: Directory to save output files
            minimize: Whether to perform energy minimization
            equilibrate: Whether to run equilibration

        Returns:
            Path to the trajectory file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare system
        self._pdb = load_complex(self.pdb_path)
        forcefield = get_forcefield(self.forcefield_name)
        modeller = prepare_system(self._pdb, forcefield=forcefield)

        # Save solvated topology for analysis
        self._topology_path = output_dir / "solvated.pdb"
        PDBFile.writeFile(
            modeller.topology,
            modeller.positions,
            open(self._topology_path, "w"),
        )

        # Set up simulation
        self._simulation = MDSimulation(modeller, forcefield, self.simulation_config)
        self._simulation.setup()

        if minimize:
            self._simulation.minimize()

        if equilibrate:
            self._simulation.equilibrate()

        # Run production
        self._trajectory_path = self._simulation.run(output_dir)

        return self._trajectory_path

    def analyze(self, trajectory_path: Path | None = None) -> ProtocolResults:
        """Analyze the trajectory and compute binding metrics.

        Args:
            trajectory_path: Path to trajectory file. If None, uses the
                trajectory from the most recent run().

        Returns:
            ProtocolResults with computed metrics

        Raises:
            RuntimeError: If no trajectory is available
        """
        traj_path = trajectory_path or self._trajectory_path
        if traj_path is None:
            raise RuntimeError("No trajectory available. Call run() first or provide trajectory_path.")

        topology_path = self._topology_path
        if topology_path is None:
            # Use original PDB if no solvated topology
            topology_path = self.pdb_path

        # Get atom indices for ligand and receptor
        ligand_indices = get_chain_atom_indices(topology_path, [self.ligand_chain])
        receptor_indices = get_chain_atom_indices(topology_path, self.receptor_chains)

        # Calculate metrics
        buried_sasa = calculate_buried_sasa(
            traj_path,
            topology_path,
            ligand_indices,
            receptor_indices,
        )

        contacts = calculate_contacts(
            traj_path,
            topology_path,
            ligand_indices,
            receptor_indices,
        )

        interaction_energy = calculate_interaction_energy(
            traj_path,
            topology_path,
            ligand_indices,
            receptor_indices,
            forcefield_name=self.forcefield_name,
        )

        rmsd = calculate_rmsd(
            traj_path,
            topology_path,
            atom_indices=ligand_indices + receptor_indices,
        )

        # Compile results
        self._results = ProtocolResults(
            sasa_buried=buried_sasa,
            sasa_buried_mean=float(np.mean(buried_sasa)),
            sasa_buried_std=float(np.std(buried_sasa)),
            interface_contacts=contacts,
            interface_contacts_mean=float(np.mean(contacts)),
            interaction_energy=interaction_energy,
            interaction_energy_mean=float(np.mean(interaction_energy)),
            interaction_energy_std=float(np.std(interaction_energy)),
            rmsd=rmsd,
            rmsd_mean=float(np.mean(rmsd)),
            raw_data={
                "ligand_chain": self.ligand_chain,
                "receptor_chains": self.receptor_chains,
                "forcefield": self.forcefield_name,
                "trajectory_path": str(traj_path),
            },
        )

        return self._results
