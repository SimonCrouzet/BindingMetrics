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
        ph: pH for hydrogen addition (default 7.4)
        solvent_model: Implicit solvent model ('obc2', 'gbn2')
        device: Compute device ('cuda', 'cpu')
        peptide_chain_id: Peptide chain ID (auto-detect smallest chain if None)
        receptor_chain_id: Receptor chain ID (auto-detect largest chain if None)
        custom_bond_handler: Optional callable invoked after hydrogen addition.
            Signature: (topology, positions, peptide_chain) -> (topology, positions, bond_info)
            where bond_info is a list of tuples passed back to the caller for
            post-processing (e.g. harmonic restraints for custom bonds).
        small_molecules: List of non-standard residues to parameterize with GAFF2
            (SMILES strings, openff.toolkit.Molecule, or RDKit Mol). Requires
            openmmforcefields. See field docstring for details.
        small_molecule_ff: GAFF2 version string (default 'gaff-2.2.20').
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

    ph: float = 7.4

    solvent_model: str = "obc2"
    device: str = "cuda"

    peptide_chain_id: Optional[str] = None
    receptor_chain_id: Optional[str] = None

    cyclic_bond_hints: Optional[list] = None
    """CyclicBondInfo objects detected from the original structure file (before
    PDBFixer prep). Used as fallback in patch_cyclic_topology when the prepped
    file has lost STRUCT_CONN records and geometry is too strained for distance
    detection."""

    custom_bond_handler: Optional[Callable] = None

    small_molecules: Optional[list] = None
    """Non-standard residues / small-molecule co-factors to parameterize with GAFF2.

    Two usage modes:

    ``"auto"`` (recommended):
        Automatically discovers all residues not covered by AMBER ff14SB and
        builds GAFF2 parameters for them from the topology geometry. No SMILES
        needed — the molecule graph is constructed directly from the atom
        connectivity after hydrogen addition.

        >>> config = RelaxationConfig(small_molecules="auto")

    Explicit list:
        Provide a list whose elements can be any of:
            • SMILES strings (e.g. ``"CC(=O)Nc1ccc(O)cc1"``)
            • ``openff.toolkit.Molecule`` objects
            • RDKit ``Chem.Mol`` objects

        >>> config = RelaxationConfig(small_molecules=["NC(CS)C(=O)O"])

    In both cases, ``openmmforcefields`` must be installed:
    ``conda install -c conda-forge openmmforcefields openff-toolkit``.

    Residues not covered by ff14SB **and** not matched by GAFF2 will still
    raise a ``ValueError`` from ``createSystem``.
    """

    small_molecule_ff: str = "gaff-2.2.20"
    """GAFF2 force-field version used by :attr:`small_molecules`.

    Passed as the ``forcefield`` argument to
    ``GAFFTemplateGenerator``.  Run
    ``GAFFTemplateGenerator.INSTALLED_FORCEFIELDS`` for available versions.
    Default is ``"gaff-2.2.20"`` (latest stable at package release time).
    """

    # Cyclization is always auto-detected; see patch_cyclic_topology.
    # Supported types (detected by inter-atom distance):
    #   • head_to_tail  — backbone C(last)–N(first) amide
    #   • disulfide     — CYS SG–SG (residues renamed CYX)
    #   • lactam_n_asp  — ASP CG–N-terminus amide  (residue renamed ASPL)
    #   • lactam_n_glu  — GLU CD–N-terminus amide  (residue renamed GLUL)
    #   • lactam_c_lys  — LYS NZ–C-terminus amide  (residue renamed LYSL)
    #
    # Unsupported types (hydrocarbon staples, thioethers, macrolactones, …)
    # raise a CyclizationError with guidance on using custom_bond_handler
    # with GAFF2/SMIRNOFF.
    #
    # Ring-aware restraint protocol (active when cyclization is detected):
    #   Stage 0 — closure bond distance restraint (strong, before Stage 1)
    #   Warmup MD — backbone φ/ψ dihedral restraints (10 ps, before production)


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

    # Cyclic bond metadata — populated when cyclization is detected.
    # Each entry: {"type": str, "atom1": "chain:res_idx:atom", "atom2": ...}
    cyclic_bonds: Optional[list] = None

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
            "cyclic_bonds": self.cyclic_bonds,
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

    @staticmethod
    def _coerce_molecules(molecules: list) -> list:
        """Convert SMILES strings or RDKit mols to openff.toolkit.Molecule objects.

        Accepts any mix of:
            • ``str`` — interpreted as a SMILES string
            • ``openff.toolkit.Molecule`` — passed through unchanged
            • ``rdkit.Chem.Mol`` — converted via openff.toolkit

        Returns a list of ``openff.toolkit.Molecule`` objects.
        """
        from openff.toolkit import Molecule

        result = []
        for m in molecules:
            if isinstance(m, str):
                result.append(Molecule.from_smiles(m, allow_undefined_stereo=True))
            elif isinstance(m, Molecule):
                result.append(m)
            else:
                result.append(Molecule.from_rdkit(m, allow_undefined_stereo=True))
        return result

    # Standard AMBER ff14SB residue names — these are handled by the base FF.
    _AMBER_STANDARD = frozenset({
        # Canonical amino acids + protonation variants
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
        "CYX", "HID", "HIE", "HIP", "HIN", "LYN", "ASH", "GLH",
        # Our custom lactam residues
        "ASPL", "GLUL", "LYSL",
        # Common capping groups and ions
        "ACE", "NME", "NMA", "FOR",
        # Water / ions
        "HOH", "WAT", "H2O", "NA", "CL", "K", "MG", "CA", "ZN",
    })

    @classmethod
    def _discover_heterogens(cls, topology) -> list:
        """Auto-discover non-standard residues and build GAFF-ready Molecule objects.

        Any residue whose name is not in the AMBER ff14SB standard set is
        treated as a heterogen. Each such residue is converted to an
        ``openff.toolkit.Molecule`` by building its heavy-atom graph from the
        OpenMM topology bonds, letting RDKit perceive bond orders via
        sanitization (matching what ``GAFFTemplateGenerator`` does internally),
        and then adding implicit H to satisfy valence.

        Args:
            topology: OpenMM Topology (after PDBFixer, before addHydrogens).
                Molecules are built as heavy-atom-only here; GAFF/antechamber
                adds H internally when generating parameters.

        Returns:
            List of unique ``openff.toolkit.Molecule`` objects (heavy atoms only),
            one per unknown residue name.
        """
        from rdkit import Chem
        from openff.toolkit import Molecule

        seen: set = set()
        result = []
        for res in topology.residues():
            if res.name in cls._AMBER_STANDARD or res.name in seen:
                continue
            seen.add(res.name)

            # Build RDKit heavy-atom molecule from topology bonds (all single).
            # We do NOT call Chem.AddHs: the molecule must match the topology
            # residue at this stage (before addHydrogens), which has no H.
            rwmol = Chem.RWMol()
            idx_map: dict = {}
            for atom in res.atoms():
                if atom.element is None or atom.element.atomic_number == 1:
                    continue
                idx_map[atom.index] = rwmol.AddAtom(Chem.Atom(atom.element.atomic_number))
            for bond in topology.bonds():
                i1, i2 = bond.atom1.index, bond.atom2.index
                if i1 in idx_map and i2 in idx_map:
                    rwmol.AddBond(idx_map[i1], idx_map[i2], Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rwmol)   # perceives double bonds / aromaticity
                # hydrogens_are_explicit=True prevents openff from adding
                # implicit H — the molecule must match the topology at this
                # stage (before addHydrogens), which has heavy atoms only.
                mol = Molecule.from_rdkit(
                    rwmol,
                    allow_undefined_stereo=True,
                    hydrogens_are_explicit=True,
                )
                result.append(mol)
                print(f"  Auto-GAFF2: '{res.name}' ({mol.n_atoms} heavy atoms)")
            except Exception as exc:
                print(f"  Warning: could not build GAFF2 molecule for '{res.name}': "
                      f"{exc}. Skipping (residue will be excluded from the system).")

        return result

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
        # Amino acids only — exclude water (HOH), nucleic acids (A/C/G/T/U/I/DA/…)
        amino_acids = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL",
            # common non-standard variants also treated as protein
            "CYX", "HID", "HIE", "HIP", "ASPL", "GLUL", "LYSL",
            "NMG", "NMA", "MVA", "MLE",
        }

        chain_sizes = []
        for chain in topology.chains():
            n_protein = sum(1 for r in chain.residues() if r.name in amino_acids)
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

    def _strip_heterogens(self, topology, positions, peptide_chain: str, receptor_chain: Optional[str],
                          warn_cutoff_ang: float = 8.0):
        from binding_metrics.io.structures import strip_heterogens
        return strip_heterogens(topology, positions, peptide_chain, receptor_chain, warn_cutoff_ang)

    def _setup_system(self, input_path: Path):
        """Load structure, prepare topology, and create OpenMM system.

        Steps:
            1. Remove atoms at the origin (placeholder atoms)
            2. Rebuild missing heavy atoms with PDBFixer
            3. Add hydrogens at pH 7.4
            4. Apply custom_bond_handler if configured
            5. Create OpenMM system with implicit solvent

        Returns:
            Tuple of (system, topology, positions, bond_info)
        """
        import tempfile

        self._import_openmm()

        # --- Structure loading ---
        # The input is expected to already be prepared (via binding-metrics-prep).
        # No PDBFixer repair here — just load and strip any origin placeholders.
        if input_path.suffix.lower() in (".cif", ".mmcif"):
            struct = PDBxFile(str(input_path))
        else:
            struct = app.PDBFile(str(input_path))
        topology, positions = struct.topology, struct.positions

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

        # --- Strip heterogens (non-protein residues outside the two chains) ---
        topology, positions = self._strip_heterogens(
            topology, positions, peptide_chain, receptor_chain
        )

        # --- Force field setup ---
        gb_file = "implicit/gbn2.xml" if self.config.solvent_model == "gbn2" else "implicit/obc2.xml"
        base_xmls = ["amber14-all.xml", "amber14/tip3pfb.xml", gb_file]
        ff = app.ForceField(*base_xmls)

        # --- Non-standard residue patching (D-AAs and NMe-AAs, before H addition) ---
        from binding_metrics.core.nonstandard import (
            detect_nonstandard,
            patch_nonstandard,
            load_nonstandard_xmls,
        )
        ns_info = detect_nonstandard(topology, peptide_chain)
        if not ns_info.is_empty:
            if ns_info.has_d_residues:
                names = [e["original_name"] for e in ns_info.d_residues]
                print(f"  D-amino acids: {names} → renamed to L counterparts for FF")
            if ns_info.has_nmethyl:
                names = [e["original_name"] for e in ns_info.nmethyl_residues]
                print(f"  N-methylated residues: {names}")
            topology, positions = patch_nonstandard(topology, positions, peptide_chain, ns_info)
            load_nonstandard_xmls(ff, ns_info)

        # --- Cyclic peptide topology patching (before hydrogen addition) ---
        # Always auto-detect: linear peptides pass through unchanged.
        # Must run here so addHydrogens sees the correct internal-residue topology.
        bond_info = []
        from binding_metrics.core.cyclic import (
            CyclizationError,
            patch_cyclic_topology,
            rename_disulfide_cys_to_cyx,
            load_extra_xmls,
        )
        topology, positions, bond_info = patch_cyclic_topology(
            topology, positions, peptide_chain,
            hints=self.config.cyclic_bond_hints,
        )
        topology, positions = rename_disulfide_cys_to_cyx(topology, positions)
        if bond_info:
            print(f"  Cyclic peptide detected — {len(bond_info)} bond(s):")
            # Build a residue-name lookup: (chain_id, res_idx_in_chain) → res_name
            res_name_map: dict = {}
            for chain in topology.chains():
                for i, res in enumerate(chain.residues()):
                    res_name_map[(chain.id, i)] = res.name
            for b in bond_info:
                c1, r1, a1 = b.atom1_id
                c2, r2, a2 = b.atom2_id
                rname1 = res_name_map.get((c1, r1), "???")
                rname2 = res_name_map.get((c2, r2), "???")
                print(f"    {b.cyclic_type:<14}: {rname1}[{r1}].{a1} → {rname2}[{r2}].{a2}")
            load_extra_xmls(ff, bond_info)
        else:
            print("  Linear peptide (no cyclization detected)")

        # --- GAFF2 for non-standard residues / small-molecule co-factors ---
        # Must be registered BEFORE addHydrogens so that addHydrogens can use
        # the GAFF template to add H to non-standard residues.
        # Molecules are built as heavy-atom-only here (matching the topology
        # state before H addition). GAFF/antechamber adds H internally during
        # parameterization.
        #
        # Note: GAFF2 is a general small-molecule force field. It works for
        # small organic co-factors and modified amino acids (as a pragmatic
        # approximation), but purpose-built parameters (e.g. CGENFF, RESP-fitted
        # charges) give higher accuracy for MD production runs.
        if not self.config.small_molecules:
            nonstandard_names = [
                res.name for res in topology.residues()
                if res.name not in self._AMBER_STANDARD
            ]
            if nonstandard_names:
                unique = sorted(set(nonstandard_names))
                print(f"  [warning] Non-standard residues found: {', '.join(unique)}")
                print("  These will likely cause 'No template found' errors.")
                print("  → Run with --small-molecules auto to parameterise via GAFF2.")
                print("  → Or run binding-metrics-prep --canonicalize to replace them first.")

        if self.config.small_molecules:
            try:
                from openmmforcefields.generators import GAFFTemplateGenerator
            except ImportError as exc:
                raise ImportError(
                    "openmmforcefields is required for small_molecules support. "
                    "Install with: conda install -c conda-forge openmmforcefields openff-toolkit"
                ) from exc
            if self.config.small_molecules == "auto":
                mols = self._discover_heterogens(topology)
            else:
                mols = self._coerce_molecules(self.config.small_molecules)
            if mols:
                gaff = GAFFTemplateGenerator(molecules=mols, forcefield=self.config.small_molecule_ff)
                ff.registerTemplateGenerator(gaff.generator)
                print(f"  Registered GAFF2 ({self.config.small_molecule_ff}) "
                      f"for {len(mols)} non-standard residue(s).")
            else:
                print("  Note: small_molecules='auto' found no non-standard residues.")

        # --- Add hydrogens ---
        print("  Adding hydrogens...")
        modeller = app.Modeller(topology, positions)

        # For cyclic peptides, pass explicit variants so addHydrogens uses
        # internal residue templates at the closure sites (not N/C-terminal).
        addh_variants = None
        if bond_info:
            from binding_metrics.core.cyclic import get_addh_variants
            addh_variants = get_addh_variants(modeller.topology, bond_info, peptide_chain)

        try:
            modeller.addHydrogens(ff, pH=self.config.ph, variants=addh_variants)
        except Exception as e:
            print(f"  Warning: addHydrogens(ff, pH={self.config.ph}) failed ({e}), "
                  "retrying without ForceField (approximate H positions)...")
            try:
                modeller.addHydrogens(pH=self.config.ph, variants=addh_variants)
            except Exception as e2:
                print(f"  Warning: addHydrogens failed: {e2}")
        topology, positions = modeller.topology, modeller.positions

        # --- Custom bond handler (plugin hook, called after H addition) ---
        if self.config.custom_bond_handler is not None:
            topology, positions, user_bond_info = self.config.custom_bond_handler(
                topology, positions, peptide_chain
            )
            if user_bond_info:
                extra_xmls = getattr(user_bond_info, "extra_xmls", [])
                if extra_xmls:
                    ff = app.ForceField(*base_xmls, *extra_xmls)
                if not bond_info:
                    bond_info = user_bond_info

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

    def _run_cyclic_warmup(
        self,
        system,
        simulation,
        topology,
        ref_positions,
        peptide_chain: str,
        omega_indices,
        warmup_ps: float = 10.0,
    ) -> None:
        """Run short restrained MD to preserve ring conformation on velocity init.

        Adds backbone φ/ψ dihedral restraints (cosine form) centred on the
        minimised structure, runs ``warmup_ps`` picoseconds with a progressive
        three-phase release, then removes all restraint forces before returning.

        Args:
            system: OpenMM System (modified in-place; forces are removed after warmup).
            simulation: Active Simulation context.
            topology: OpenMM Topology.
            ref_positions: Reference positions (minimised) for measuring reference angles.
            peptide_chain: Peptide chain ID.
            omega_indices: 4-tuple of atom indices for the closure ω dihedral,
                or None (ω restraint is skipped when None).
            warmup_ps: Total warmup MD duration in picoseconds (default 10).
        """
        import math

        # Collect backbone φ/ψ atom quads for each residue in the peptide chain
        phi_quads = []
        psi_quads = []

        residues = []
        for chain in topology.chains():
            if chain.id == peptide_chain:
                residues = list(chain.residues())
                break

        def _idx(res, name):
            for atom in res.atoms():
                if atom.name == name:
                    return atom.index
            return None

        n = len(residues)
        for i, res in enumerate(residues):
            n_i  = _idx(res, "N")
            ca_i = _idx(res, "CA")
            c_i  = _idx(res, "C")
            if n_i is None or ca_i is None or c_i is None:
                continue
            # φ(i): C(i-1)–N(i)–CA(i)–C(i)   [uses cyclic wrap for residue 0]
            prev = residues[(i - 1) % n]
            c_prev = _idx(prev, "C")
            if c_prev is not None:
                phi_quads.append((c_prev, n_i, ca_i, c_i))
            # ψ(i): N(i)–CA(i)–C(i)–N(i+1)   [uses cyclic wrap for last residue]
            nxt = residues[(i + 1) % n]
            n_next = _idx(nxt, "N")
            if n_next is not None:
                psi_quads.append((n_i, ca_i, c_i, n_next))

        if not phi_quads and not psi_quads:
            return   # nothing to restrain

        # Measure reference dihedrals from minimised positions
        ref_pos = np.array([[p.x, p.y, p.z] for p in ref_positions])

        def _dihedral_rad(p, i1, i2, i3, i4):
            b1 = p[i2] - p[i1]
            b2 = p[i3] - p[i2]
            b3 = p[i4] - p[i3]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            x  = np.dot(n1, n2)
            y  = np.dot(m1, n2)
            return math.atan2(y, x)

        # Build torsion force: V = k * (1 - cos(θ - θ0))  →  harmonic near θ0
        torsion_force = openmm.CustomTorsionForce(
            "k_phi * (1 - cos(theta - theta0))"
        )
        torsion_force.addGlobalParameter(
            "k_phi",
            50.0 * unit.kilojoules_per_mole,
        )
        torsion_force.addPerTorsionParameter("theta0")

        for quad in phi_quads + psi_quads:
            theta0 = _dihedral_rad(ref_pos, *quad)
            torsion_force.addTorsion(*quad, [theta0])

        # Add ω restraint for the closure bond (if available)
        omega_force = None
        omega_idx = None
        if omega_indices is not None:
            omega_force = openmm.CustomTorsionForce(
                "k_omega * (1 - cos(theta - theta0_omega))"
            )
            omega_force.addGlobalParameter(
                "k_omega",
                100.0 * unit.kilojoules_per_mole,
            )
            omega_force.addPerTorsionParameter("theta0_omega")
            theta0_omega = _dihedral_rad(ref_pos, *omega_indices)
            omega_force.addTorsion(*omega_indices, [theta0_omega])
            omega_idx = system.addForce(omega_force)

        torsion_idx = system.addForce(torsion_force)
        simulation.context.reinitialize(preserveState=True)

        steps_per_ps = int(1000 / self.config.md_timestep_fs)
        warmup_steps = int(warmup_ps * steps_per_ps)

        # Phase 1: full restraint (first half)
        simulation.step(warmup_steps // 2)

        # Phase 2: reduce to 20 % (second quarter)
        simulation.context.setParameter("k_phi", 10.0 * unit.kilojoules_per_mole)
        if omega_force is not None:
            simulation.context.setParameter(
                "k_omega", 20.0 * unit.kilojoules_per_mole
            )
        simulation.step(warmup_steps // 4)

        # Phase 3: near-zero (last quarter)
        simulation.context.setParameter("k_phi", 1.0 * unit.kilojoules_per_mole)
        if omega_force is not None:
            simulation.context.setParameter(
                "k_omega", 2.0 * unit.kilojoules_per_mole
            )
        simulation.step(warmup_steps - warmup_steps // 2 - warmup_steps // 4)

        # Remove restraint forces so production MD is unrestrained.
        # Remove in descending index order to avoid renumbering the lower one.
        indices_to_remove = sorted(
            [i for i in [torsion_idx, omega_idx] if i is not None], reverse=True
        )
        for i in indices_to_remove:
            system.removeForce(i)
        simulation.context.reinitialize(preserveState=True)

    def _get_platform(self):
        """Get the OpenMM compute platform, falling back to CPU if CUDA fails."""
        self._import_openmm()
        if self.config.device == "cuda":
            try:
                platform = openmm.Platform.getPlatformByName("CUDA")
                properties = {"CudaPrecision": "mixed"}
                # Probe with a minimal context — catches driver/PTX version mismatches
                # that only surface at context creation, not at platform lookup.
                _sys = openmm.System()
                _sys.addParticle(1.0)
                _ctx = openmm.Context(_sys, openmm.VerletIntegrator(0.001), platform)
                del _ctx, _sys
                print(f"  Platform: CUDA (mixed precision)")
                return platform, properties
            except Exception as e:
                print(f"  Warning: CUDA unavailable ({e}), falling back to CPU.")
        print(f"  Platform: CPU")
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

            if bond_info:
                # atom1_id / atom2_id store (chain_id, res_idx_in_chain, atom_name)
                # where res_idx_in_chain is a 0-based list index used internally.
                # For display we want the actual residue number (auth_seq_id).
                # Build a per-chain index→res_id lookup from the topology.
                _chain_res_ids: dict[str, list[str]] = {}
                for _chain in topology.chains():
                    _chain_res_ids[_chain.id] = [r.id for r in _chain.residues()]

                def _fmt_atom(aid: tuple) -> str:
                    ch, idx, name = aid
                    res_id = _chain_res_ids.get(ch, [None] * (idx + 1))[idx] if idx < len(_chain_res_ids.get(ch, [])) else idx
                    return f"{ch}:{res_id}:{name}"

                result.cyclic_bonds = [
                    {
                        "type": b.cyclic_type,
                        "atom1": _fmt_atom(b.atom1_id),
                        "atom2": _fmt_atom(b.atom2_id),
                    }
                    for b in bond_info
                ]

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

            # --- Resolve cyclic closure atom indices (post-addHydrogens) ---
            # closure_indices_list: list of (idx1, idx2) tuples, one per bond
            # omega_indices: first non-None omega from the list (for warmup dihedral)
            closure_indices_list = []
            omega_indices = None
            if bond_info:
                from binding_metrics.core.cyclic import (
                    resolve_closure_atoms,
                    resolve_omega_atoms,
                )
                for bi in bond_info:
                    try:
                        ci = resolve_closure_atoms(topology, bi, peptide_chain)
                        closure_indices_list.append(ci)
                        if omega_indices is None:
                            omega_indices = resolve_omega_atoms(topology, bi, peptide_chain)
                    except Exception as e:
                        print(f"[{sample_id}]   Warning: could not resolve closure atoms: {e}")

            # --- Multi-stage minimization ---
            n_stages = "4" if closure_indices_list else "3"
            print(f"[{sample_id}] Minimizing ({n_stages} stages)...")
            min_start = time.time()

            # Stage 0 (cyclic only): relax all closure bond geometries before Stage 1.
            # One CustomBondForce covers all closure bonds (monocyclic or bicyclic+).
            if closure_indices_list:
                print(f"[{sample_id}]   Stage 0: Closure bond geometry relaxation "
                      f"({len(closure_indices_list)} bond(s))")
                closure_force = openmm.CustomBondForce(
                    "0.5 * k_closure * (r - r0_closure)^2"
                )
                closure_force.addGlobalParameter(
                    "k_closure",
                    1000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
                )
                closure_force.addGlobalParameter(
                    "r0_closure",
                    0.1325 * unit.nanometers,  # ideal amide bond; S-S is 0.205 nm but
                )                              # the force field enforces the correct length
                for ci in closure_indices_list:
                    closure_force.addBond(ci[0], ci[1], [])
                system.addForce(closure_force)
                simulation.context.reinitialize(preserveState=True)
                simulation.minimizeEnergy(maxIterations=200)
                simulation.context.setParameter("k_closure", 0.0)

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

            # Save minimized structure — pass input as source so auth chain IDs
            # and residue numbers from the (already-prepped) input are preserved.
            min_path = output_dir / f"{sample_id}_minimized.cif"
            src = input_path if input_path.suffix.lower() in (".cif", ".mmcif") else None
            save_cif(topology, minimized_positions, min_path, source_cif_path=src)
            result.minimized_structure_path = str(min_path)
            print(f"[{sample_id}] Minimized: {result.potential_energy_minimized:.1f} kJ/mol")

            # --- MD simulation ---
            if self.config.md_duration_ps > 0:
                print(f"[{sample_id}] Running MD ({self.config.md_duration_ps} ps)...")
                md_start = time.time()
                simulation.context.setVelocitiesToTemperature(
                    self.config.md_temperature_k * unit.kelvin
                )

                # Cyclic warmup: 10 ps backbone φ/ψ dihedral restraints to
                # preserve ring conformation during velocity initialisation.
                if closure_indices_list:
                    print(f"[{sample_id}]   Cyclic warmup: 10 ps restrained MD "
                          "(backbone φ/ψ restraints)...")
                    self._run_cyclic_warmup(
                        system, simulation, topology, minimized_positions,
                        peptide_chain, omega_indices,
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

                # Save final MD structure — preserve auth IDs from input
                final_path = output_dir / f"{sample_id}_md_final.cif"
                save_cif(topology, final_positions, final_path, source_cif_path=src)
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
    parser.add_argument("--ph", type=float, default=7.4, help="pH for hydrogen addition (default 7.4)")
    parser.add_argument("--solvent-model", choices=["obc2", "gbn2"], default="obc2", help="Implicit solvent model")
    parser.add_argument("--peptide-chain", type=str, default=None, help="Peptide chain ID (auto-detect if omitted)")
    parser.add_argument("--receptor-chain", type=str, default=None, help="Receptor chain ID (auto-detect if omitted)")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample identifier (defaults to input file stem)")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="Path to write relax results JSON "
                             "(default: <output-dir>/<sample-id>_relax_results.json)")
    from binding_metrics.cli import add_log_file_arg
    add_log_file_arg(parser)
    args = parser.parse_args()

    from binding_metrics.cli import log_to_file
    with log_to_file(args.log_file):
        config = RelaxationConfig(
            md_duration_ps=args.md_duration_ps,
            md_save_interval_ps=args.md_save_interval_ps,
            md_temperature_k=args.temperature,
            ph=args.ph,
            device=args.device,
            solvent_model=args.solvent_model,
            peptide_chain_id=args.peptide_chain,
            receptor_chain_id=args.receptor_chain,
        )

        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(args.input, args.output_dir, sample_id=args.sample_id)

        results_path: Path = args.results_json or (
            args.output_dir / f"{result.sample_id}_relax_results.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as _fh:
            json.dump(result.to_dict(), _fh, indent=2, default=str)
        print(f"  Results:   {results_path}")

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
