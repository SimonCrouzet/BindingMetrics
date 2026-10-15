"""Peptide binding protocol for evaluating peptide-protein interactions."""

import string
from pathlib import Path
from typing import Literal

import numpy as np
from openmm.app import PDBFile

from binding_metrics.core.forcefields import get_forcefield
from binding_metrics.core.residues import AMBER_STANDARD_RESIDUES, WATER_NAMES_ALL
from binding_metrics.core.simulation import MDSimulation, SimulationConfig
from binding_metrics.core.system import prepare_system
from binding_metrics.io.structures import get_chain_atom_indices, load_complex
from binding_metrics.metrics.contacts import calculate_contacts
from binding_metrics.metrics.energy import calculate_interaction_energy
from binding_metrics.metrics.rmsd import calculate_rmsd
from binding_metrics.metrics.sasa import calculate_buried_sasa
from binding_metrics.protocols.base import BaseProtocol, ProtocolResults

#: ``calculate_buried_sasa`` returns nm^2 (mdtraj units) and ``ProtocolResults`` reports A^2:
#: 1 nm = 10 A, so 1 nm^2 = 100 A^2.
_NM2_TO_ANGSTROM2 = 100.0


#: Characters a PDB file takes as a chain ID: the record has one column for it.
_PDB_CHAIN_ID_CHARACTERS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _is_solvent_chain(chain) -> bool:
    """True for a chain of water and ions, such as those ``Modeller`` adds.

    An ion is a residue of one atom that is not a standard residue, so a chain of alpha
    carbons only is a solute chain.
    """

    def is_solvent(residue) -> bool:
        if residue.name in WATER_NAMES_ALL:
            return True
        return residue.name not in AMBER_STANDARD_RESIDUES and len(list(residue.atoms())) == 1

    residues = list(chain.residues())
    return bool(residues) and all(is_solvent(residue) for residue in residues)


def _pdb_chain_ids(topology) -> tuple[list[str], dict[str, str]]:
    """Chain ID to write for each chain of ``topology``, and the ID of each solute chain.

    ``Modeller`` names the water and ion chains by their number (``"3"``, ``"4"``), which
    can be the ID of a solute chain. A chain keeps its own ID when it is one character of
    a PDB chain ID and no earlier chain, solute first, has it; any other chain takes the
    first unused character. Chains of the same ID and kind share one written ID, so the
    chains the caller names by one ID stay together.

    Returns:
        The IDs in chain order, and a mapping from the ID of each solute chain in
        ``topology`` to the ID it is written under.

    Raises:
        ValueError: If the chains have more distinct IDs than a PDB file has characters.
    """
    chains = list(topology.chains())
    kinds = [(chain.id, _is_solvent_chain(chain)) for chain in chains]
    ordered = sorted(dict.fromkeys(kinds), key=lambda kind: kind[1])
    if len(ordered) > len(_PDB_CHAIN_ID_CHARACTERS):
        raise ValueError(
            f"the system has {len(ordered)} distinct chain IDs and a PDB file has "
            f"{len(_PDB_CHAIN_ID_CHARACTERS)}: solvated.pdb could not tell its chains apart"
        )
    written: dict[tuple[str, bool], str] = {}
    for kind in ordered:
        chain_id = kind[0]
        if (
            len(chain_id) == 1
            and chain_id in _PDB_CHAIN_ID_CHARACTERS
            and chain_id not in written.values()
        ):
            written[kind] = chain_id
    unused = (c for c in _PDB_CHAIN_ID_CHARACTERS if c not in written.values())
    for kind in ordered:
        if kind not in written:
            written[kind] = next(unused)
    solute = {
        chain_id: new_id for (chain_id, is_solvent), new_id in written.items() if not is_solvent
    }
    return [written[kind] for kind in kinds], solute


def _chain_ids_present(topology) -> list[str]:
    """IDs of the chains of ``topology`` that are not water or ions, without repeats."""
    return list(dict.fromkeys(c.id for c in topology.chains() if not _is_solvent_chain(c)))


def _quoted(chain_ids) -> str:
    return ", ".join(repr(chain_id) for chain_id in chain_ids)


def _require_chains_present(role: str, asked: list[str], present: list[str], where: str) -> None:
    """Raise ``ValueError`` naming the chain IDs asked for that are not in ``present``."""
    missing = [chain_id for chain_id in asked if chain_id not in present]
    if missing:
        raise ValueError(
            f"{role} {_quoted(missing)} not in {where}; chain IDs present: {_quoted(present)}"
        )


def _require_frames(series) -> None:
    """Raise if a per-frame series is empty: ``np.mean`` of it would be NaN."""
    if len(series) == 0:
        raise RuntimeError("trajectory has 0 frames")


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
        self._written_chain_ids: dict[str, str] = {}

    def run(
        self,
        output_dir: str | Path,
        minimize: bool = True,
        equilibrate: bool = True,
    ) -> Path:
        """Run the MD simulation.

        ``solvated.pdb`` keeps the chain and residue IDs of the system. A chain ID that a
        PDB file cannot hold (more than one character, a blank, or the ID of a water chain)
        is written as another character, and ``analyze`` looks the caller's IDs up under
        the written ones.

        Args:
            output_dir: Directory to save output files
            minimize: Whether to perform energy minimization
            equilibrate: Whether to run equilibration

        Returns:
            Path to the trajectory file

        Raises:
            ValueError: If the ligand or a receptor chain is not in the input, or the
                system has more chain IDs than a PDB file can hold.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._pdb = load_complex(self.pdb_path)
        present = _chain_ids_present(self._pdb.topology)
        name = self.pdb_path.name
        _require_chains_present("ligand_chain", [self.ligand_chain], present, name)
        _require_chains_present("receptor_chains", list(self.receptor_chains), present, name)
        forcefield = get_forcefield(self.forcefield_name)
        modeller = prepare_system(self._pdb, forcefield=forcefield)

        chain_ids, self._written_chain_ids = _pdb_chain_ids(modeller.topology)
        self._topology_path = output_dir / "solvated.pdb"
        self._write_topology(modeller, chain_ids)

        self._simulation = MDSimulation(modeller, forcefield, self.simulation_config)
        self._simulation.setup()

        if minimize:
            self._simulation.minimize()

        if equilibrate:
            self._simulation.equilibrate()

        self._trajectory_path = self._simulation.run(output_dir)

        return self._trajectory_path

    def _write_topology(self, modeller, chain_ids: list[str]) -> None:
        """Write ``solvated.pdb`` under ``chain_ids``, keeping the residue IDs of the topology."""
        chains = list(modeller.topology.chains())
        own_ids = [chain.id for chain in chains]
        for chain, chain_id in zip(chains, chain_ids):
            chain.id = chain_id
        try:
            with open(self._topology_path, "w", encoding="utf-8") as topology_file:
                PDBFile.writeFile(
                    modeller.topology, modeller.positions, topology_file, keepIds=True
                )
        finally:
            for chain, chain_id in zip(chains, own_ids):
                chain.id = chain_id

    def _partner_atom_indices(self, topology_path: Path) -> tuple[list[int], list[int]]:
        """Atom indices of the ligand and the receptor in ``topology_path``.

        The caller's chain IDs are looked up under the IDs ``run`` wrote them with.

        Raises:
            ValueError: If either partner selects no atom.
        """
        ligand_id = self._written_chain_ids.get(self.ligand_chain, self.ligand_chain)
        receptor_ids = [self._written_chain_ids.get(c, c) for c in self.receptor_chains]
        ligand_indices = get_chain_atom_indices(topology_path, [ligand_id])
        receptor_indices = get_chain_atom_indices(topology_path, receptor_ids)
        for role, asked, looked_up, indices in (
            ("ligand_chain", [self.ligand_chain], [ligand_id], ligand_indices),
            ("receptor_chains", list(self.receptor_chains), receptor_ids, receptor_indices),
        ):
            if not indices:
                present = list(
                    dict.fromkeys(c.id for c in PDBFile(str(topology_path)).topology.chains())
                )
                found_as = "" if looked_up == asked else f" (written as {_quoted(looked_up)})"
                raise ValueError(
                    f"{role} {_quoted(asked)}{found_as} selects no atoms in {topology_path.name}; "
                    f"chain IDs present: {_quoted(present)}"
                )
        return ligand_indices, receptor_indices

    def analyze(self, trajectory_path: Path | None = None) -> ProtocolResults:
        """Analyze the trajectory and compute binding metrics.

        Args:
            trajectory_path: Path to trajectory file. If None, uses the
                trajectory from the most recent run().

        Returns:
            ProtocolResults with computed metrics

        Raises:
            RuntimeError: If no trajectory is available, or if it has no frames
                (the means would otherwise be NaN and look like a result).
            ValueError: If the ligand or the receptor selects no atom of the topology.
        """
        traj_path = trajectory_path or self._trajectory_path
        if traj_path is None:
            raise RuntimeError(
                "No trajectory available. Call run() first or provide trajectory_path."
            )

        topology_path = self._topology_path
        if topology_path is None:
            # Use original PDB if no solvated topology
            topology_path = self.pdb_path

        ligand_indices, receptor_indices = self._partner_atom_indices(topology_path)

        buried_sasa_nm2 = calculate_buried_sasa(
            traj_path,
            topology_path,
            ligand_indices,
            receptor_indices,
        )
        _require_frames(buried_sasa_nm2)
        buried_sasa = buried_sasa_nm2 * _NM2_TO_ANGSTROM2

        contacts = calculate_contacts(
            traj_path,
            topology_path,
            ligand_indices,
            receptor_indices,
        )
        _require_frames(contacts)

        interaction_energy = calculate_interaction_energy(
            traj_path,
            topology_path,
            ligand_indices,
            receptor_indices,
            forcefield_name=self.forcefield_name,
        )
        _require_frames(interaction_energy)

        # mdtraj keeps the atoms of a slice in file order and warns about a descending list, which
        # the concatenation is when the ligand chain follows the receptor chains in the file.
        rmsd = calculate_rmsd(
            traj_path,
            topology_path,
            atom_indices=sorted(ligand_indices + receptor_indices),
        )
        _require_frames(rmsd)

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
