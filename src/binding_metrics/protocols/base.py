"""Base protocol class for binding metrics evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ProtocolResults:
    """Results from a binding protocol analysis.

    Attributes:
        sasa_buried: Buried solvent accessible surface area (A^2) per frame
        sasa_buried_mean: Mean buried SASA over trajectory
        sasa_buried_std: Standard deviation of buried SASA
        interface_contacts: Number of interface contacts per frame
        interface_contacts_mean: Mean interface contacts
        interaction_energy: Interaction energy (kJ/mol) per frame
        interaction_energy_mean: Mean interaction energy
        interaction_energy_std: Standard deviation of interaction energy
        rmsd: RMSD (nm) per frame relative to initial structure
        rmsd_mean: Mean RMSD over trajectory
        raw_data: Dictionary of additional raw analysis data
    """

    sasa_buried: np.ndarray = field(default_factory=lambda: np.array([]))
    sasa_buried_mean: float = 0.0
    sasa_buried_std: float = 0.0
    interface_contacts: np.ndarray = field(default_factory=lambda: np.array([]))
    interface_contacts_mean: float = 0.0
    interaction_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    interaction_energy_mean: float = 0.0
    interaction_energy_std: float = 0.0
    rmsd: np.ndarray = field(default_factory=lambda: np.array([]))
    rmsd_mean: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            "sasa_buried_mean": self.sasa_buried_mean,
            "sasa_buried_std": self.sasa_buried_std,
            "interface_contacts_mean": self.interface_contacts_mean,
            "interaction_energy_mean": self.interaction_energy_mean,
            "interaction_energy_std": self.interaction_energy_std,
            "rmsd_mean": self.rmsd_mean,
            "n_frames": len(self.sasa_buried) if len(self.sasa_buried) > 0 else 0,
        }

    def summary(self) -> str:
        """Return a summary string of the results."""
        return (
            f"Binding Metrics Summary:\n"
            f"  Buried SASA: {self.sasa_buried_mean:.1f} +/- {self.sasa_buried_std:.1f} A^2\n"
            f"  Interface contacts: {self.interface_contacts_mean:.1f}\n"
            f"  Interaction energy: {self.interaction_energy_mean:.1f} +/- "
            f"{self.interaction_energy_std:.1f} kJ/mol\n"
            f"  RMSD: {self.rmsd_mean:.3f} nm"
        )


class BaseProtocol(ABC):
    """Abstract base class for binding evaluation protocols.

    Protocols define complete workflows for evaluating binding metrics,
    including system setup, simulation, and analysis.
    """

    def __init__(
        self,
        pdb_path: str | Path,
        ligand_chain: str,
        receptor_chains: list[str],
    ):
        """Initialize the protocol.

        Args:
            pdb_path: Path to the input PDB file (pre-docked complex)
            ligand_chain: Chain ID of the ligand
            receptor_chains: List of chain IDs for the receptor
        """
        self.pdb_path = Path(pdb_path)
        self.ligand_chain = ligand_chain
        self.receptor_chains = receptor_chains
        self._trajectory_path: Path | None = None
        self._results: ProtocolResults | None = None

    @property
    def trajectory_path(self) -> Path | None:
        """Path to the trajectory file after running simulation."""
        return self._trajectory_path

    @property
    def results(self) -> ProtocolResults | None:
        """Results from analysis, if available."""
        return self._results

    @abstractmethod
    def run(self, output_dir: str | Path, **kwargs) -> Path:
        """Run the simulation protocol.

        Args:
            output_dir: Directory to save output files
            **kwargs: Protocol-specific parameters

        Returns:
            Path to the trajectory file
        """
        pass

    @abstractmethod
    def analyze(self, trajectory_path: Path | None = None) -> ProtocolResults:
        """Analyze the trajectory and compute binding metrics.

        Args:
            trajectory_path: Path to trajectory file. If None, uses the
                trajectory from the most recent run().

        Returns:
            ProtocolResults with computed metrics
        """
        pass

    def run_and_analyze(
        self,
        output_dir: str | Path,
        **kwargs,
    ) -> ProtocolResults:
        """Run simulation and analyze in one step.

        Args:
            output_dir: Directory to save output files
            **kwargs: Protocol-specific parameters

        Returns:
            ProtocolResults with computed metrics
        """
        self.run(output_dir, **kwargs)
        return self.analyze()
