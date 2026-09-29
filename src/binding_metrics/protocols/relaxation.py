"""Implicit solvent MD relaxation protocol for protein complexes.

Performs multi-stage energy minimization followed by an optional short MD
simulation using OpenMM with implicit solvent (OBC2 or GBn2). Designed for
fast GPU-accelerated evaluation of protein-peptide complexes.

Minimization stages:
    Stage 1: Initial global relaxation (resolves clashes)
    Stage 2: Backbone-restrained optimization (side chains optimize)
    Stage 3: Final unrestrained refinement

Model and method references:
    Force field     AMBER ff14SB (Maier et al., J. Chem. Theory Comput. 11, 3696, 2015),
                    loaded through OpenMM's ``amber14-all.xml``.
    Implicit water  OBC2 (Onufriev, Bashford and Case, Proteins 55, 383, 2004) or
                    GBn2 (Nguyen, Roe and Simmerling, J. Chem. Theory Comput. 9,
                    2020, 2013) generalized Born, no cutoff.
    Constraints     Bonds to hydrogen are constrained, which is what allows the
                    default 2 fs time step.
    MD integrator   Langevin "middle" scheme (Zhang, Liu, Yan, Tuckerman and Liu,
                    J. Phys. Chem. A 123, 6056, 2019), as implemented by OpenMM's
                    ``LangevinMiddleIntegrator``.
    RMSD            Optimal superposition by the Kabsch algorithm (Acta Cryst. A32,
                    922, 1976).

Usage:
    python -m binding_metrics.protocols.relaxation \\
        --input complex.cif \\
        --output-dir results/ \\
        --md-duration-ps 200

Configuration file:
    --config relax.toml supplies option defaults; flags on the command line
    override the file. Keys are the long option names (md-duration-ps or
    md_duration_ps):

        # relax.toml
        md-duration-ps = 100
        solvent-model = "gbn2"
        temperature = 310
        small-molecules = "none"

    A flag takes true or false, and an unknown key is an error.
"""

import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from binding_metrics._constants import (
    DEFAULT_DEVICE,
    DEFAULT_MD_DURATION_PS,
    DEFAULT_MD_SAVE_INTERVAL_PS,
    DEFAULT_PH,
    DEFAULT_RANDOM_SEED,
)
from binding_metrics.core.residues import (
    AMBER_STANDARD_VARIANTS,
    BACKBONE_HEAVY_ATOM_NAMES,
    FORCE_FIELD_CAP_NAMES,
    ION_NAMES_COMMON,
    LACTAM_TEMPLATE_RESIDUES,
    PROTEIN_RESIDUES,
    STANDARD_AMINO_ACIDS,
    WATER_NAMES_WITH_H2O,
)
from binding_metrics.protocols.relaxer import Relaxer

logger = logging.getLogger(__name__)

# --- Minimization schedule -------------------------------------------------
#
# The three stages use a tolerance that tightens from coarse to fine: the first
# stages only need to remove clashes and relieve strain, so stopping early saves
# time, while the last stage converges to ``RelaxationConfig.min_tolerance``.
# OpenMM measures the tolerance as the RMS force in kJ/mol/nm.

#: Stage 1 (global relaxation) tolerance, as a multiple of ``min_tolerance``.
STAGE1_TOLERANCE_FACTOR = 10
#: Stage 2 (backbone-restrained) tolerance, as a multiple of ``min_tolerance``.
STAGE2_TOLERANCE_FACTOR = 5

# --- Cyclic closure (Stage 0) ----------------------------------------------
#
# A closure bond taken from a predicted or crystal structure can start far from
# its equilibrium length. Stage 0 first pulls each closure bond to a peptide-bond
# length with a strong harmonic restraint, then switches the restraint off so the
# force field alone decides the final geometry in Stages 1-3.

#: Restraint force constant of the closure bond in kJ/mol/nm^2.
CLOSURE_RESTRAINT_K_KJ_MOL_NM2 = 1000.0
#: Target length of the closure bond in nm: a peptide C-N bond is about 0.133 nm
#: (Engh and Huber, Acta Cryst. A47, 392, 1991). A disulfide S-S bond is longer
#: (0.205 nm), but the restraint is switched off before Stage 1 and the force
#: field then relaxes the bond to its own length.
CLOSURE_BOND_TARGET_NM = 0.1325
#: Iteration cap of the Stage 0 minimization.
CLOSURE_MINIMIZATION_MAX_ITERATIONS = 200

# --- Cyclic warm-up MD -----------------------------------------------------
#
# Assigning Maxwell-Boltzmann velocities to a minimized macrocycle can kick it
# out of the ring conformation the minimization found. The warm-up runs the
# first picoseconds of MD with cosine restraints on the backbone phi/psi
# dihedrals (and on the closure omega), centred on the minimized angles, and
# releases them in three steps: 100 %, 20 % and 2 % of the initial force
# constant over the first half, next quarter and last quarter of the warm-up.
# The energy is k * (1 - cos(theta - theta0)), which is harmonic with force
# constant k near theta0.

#: Length of the restrained warm-up in ps.
CYCLIC_WARMUP_PS = 10.0
#: Force constant k of the phi/psi restraints in kJ/mol for the three phases.
WARMUP_PHI_PSI_K_KJ_MOL = (50.0, 10.0, 1.0)
#: Force constant k of the closure omega restraint in kJ/mol for the three
#: phases. It is twice the phi/psi value: the single closure omega is held more
#: tightly than the phi/psi torsions.
WARMUP_OMEGA_K_KJ_MOL = (100.0, 20.0, 2.0)

#: Slack when deciding whether ``md_duration_ps`` is a whole number of save
#: intervals; absorbs floating-point error such as 0.3 / 0.1 = 2.9999999999999996.
_FRAME_COUNT_TOLERANCE = 1e-9


def _md_frame_count(duration_ps: float, save_interval_ps: float) -> int:
    """Number of trajectory frames an MD run of ``duration_ps`` produces."""
    return int(duration_ps / save_interval_ps + _FRAME_COUNT_TOLERANCE)


@dataclass
class RelaxationConfig:
    """Configuration for implicit solvent MD relaxation.

    Attributes:
        min_steps_initial: Steps for initial global minimization (stage 1)
        min_steps_restrained: Steps for backbone-restrained minimization (stage 2)
        min_steps_final: Steps for final unrestrained minimization (stage 3)
        min_tolerance: Energy tolerance in kJ/mol/nm for the final stage; Stages
            1 and 2 use ``STAGE1_TOLERANCE_FACTOR`` and ``STAGE2_TOLERANCE_FACTOR``
            times this value
        restraint_strength: Backbone restraint force constant in kJ/mol/nm² for
            Stage 2 (100 kJ/mol/nm² is 1 kJ/mol/Å²), centred on the input backbone
        md_duration_ps: MD simulation duration in picoseconds (0 to skip)
        md_timestep_fs: MD integration timestep in femtoseconds
        md_temperature_k: Simulation temperature in Kelvin
        md_friction: Langevin friction coefficient in 1/ps
        md_save_interval_ps: Interval between saved trajectory frames in ps.
            Must not exceed ``md_duration_ps`` when MD runs; a duration that is
            not a whole number of intervals is cut to the last full one, with a
            warning.
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

    md_duration_ps: float = DEFAULT_MD_DURATION_PS
    md_timestep_fs: float = 2.0
    md_temperature_k: float = 300.0
    md_friction: float = 1.0
    md_save_interval_ps: float = DEFAULT_MD_SAVE_INTERVAL_PS

    ph: float = DEFAULT_PH

    solvent_model: str = "obc2"
    device: str = DEFAULT_DEVICE

    random_seed: Optional[int] = DEFAULT_RANDOM_SEED
    """Seed for every stochastic step (hydrogen placement, MD initial velocities
    and the Langevin thermostat). A fixed int (the default) makes a run
    reproducible; ``None`` opts into fresh randomness, e.g. to generate
    independent MD replicas. Note: minimization is reproducible regardless, but
    GPU MD may still differ in the last digits across runs even with a fixed seed
    because CUDA force reduction order is not deterministic."""

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

    def __post_init__(self) -> None:
        """Reject MD settings that would produce no frames or a shorter run.

        Raises:
            ValueError: ``md_duration_ps`` is positive but ``md_save_interval_ps``
                is not, or the duration is shorter than one save interval (the
                run would save 0 frames and fail after the MD finished).
        """
        if self.md_duration_ps <= 0:
            return
        if self.md_save_interval_ps <= 0:
            raise ValueError(
                f"md_save_interval_ps must be positive when MD runs, got {self.md_save_interval_ps}"
            )
        n_frames = _md_frame_count(self.md_duration_ps, self.md_save_interval_ps)
        if n_frames < 1:
            raise ValueError(
                f"md_duration_ps={self.md_duration_ps} is shorter than "
                f"md_save_interval_ps={self.md_save_interval_ps}: the run would save no "
                "frames. Lower md_save_interval_ps or lengthen md_duration_ps."
            )
        simulated_ps = n_frames * self.md_save_interval_ps
        if abs(simulated_ps - self.md_duration_ps) > _FRAME_COUNT_TOLERANCE * max(
            1.0, self.md_duration_ps
        ):
            warnings.warn(
                f"md_duration_ps={self.md_duration_ps} is not a multiple of "
                f"md_save_interval_ps={self.md_save_interval_ps}: MD stops after "
                f"{simulated_ps:g} ps ({n_frames} frames).",
                UserWarning,
                stacklevel=3,
            )


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
        receptor_rmsd_md_final: Receptor Cα RMSD of final MD frame vs minimized (Angstroms)
        receptor_drift_mean: Mean receptor Cα RMSD across all MD frames vs minimized (Angstroms)
        pep_rec_com_distance_delta: Change in peptide–receptor Cα COM distance from minimized
            to final MD frame (Angstroms). Positive = separating.
        minimization_time_s: Wall time for minimization in seconds
        md_time_s: Wall time for MD simulation in seconds
        minimized_structure_path: Path to saved minimized structure CIF
        md_final_structure_path: Path to saved final MD frame CIF
        peptide_cyclic_bonds: Detected closure bonds of a cyclic peptide
        platform: OpenMM platform the run used ("CUDA" or "CPU")
        precision: Numeric precision on that platform ("mixed" on CUDA, None
            where OpenMM reports none)
        platform_fallback_reason: Why CUDA was not used, when CUDA was requested
            and the run fell back to CPU
        qc: Structural QC of the relaxed structure (``{"passed", "failed",
            "checks"}`` from ``protocols.qc.check_relaxed_structure``), with the
            same schema under ``"md_final"`` when MD ran. Advisory.
        qc_passed: True when every QC check passed, False when one failed, None
            when QC did not run
        ncaa_bond_order_source: Where the bond orders of each auto-parameterised
            non-canonical residue came from, ``{residue name: "ccd" or
            "single_bonds"}``. ``"single_bonds"`` marks a residue that is not in
            the Chemical Component Dictionary (or disagrees with its entry); its
            double bonds, aromatic rings and hydrogen count are unreliable.
            Empty when no residue was parameterised this way.
        dropped_protein_chains: IDs of the protein chains that are neither the peptide
            nor the receptor and were removed before the system was built (see
            ``io.structures.drop_other_protein_chains``). Empty when there were none.
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
    receptor_rmsd_md_final: Optional[float] = None
    receptor_drift_mean: Optional[float] = None
    pep_rec_com_distance_delta: Optional[float] = None

    minimization_time_s: Optional[float] = None
    md_time_s: Optional[float] = None

    minimized_structure_path: Optional[str] = None
    md_final_structure_path: Optional[str] = None

    # Cyclic bond metadata — populated when cyclization is detected in the peptide.
    # Each entry: {"type": str, "atom1": "chain:res_idx:atom", "atom2": ...}
    peptide_cyclic_bonds: Optional[list] = None

    # OpenMM platform the run actually used ("CUDA" or "CPU") and its numeric
    # precision ("mixed" on CUDA; None where OpenMM reports no precision, as on
    # the CPU platform). ``platform_fallback_reason`` is set only when CUDA was
    # requested and the run fell back to CPU.
    platform: Optional[str] = None
    precision: Optional[str] = None
    platform_fallback_reason: Optional[str] = None

    # Structural QC (see protocols/qc.py). ``qc`` is the dict returned by
    # ``check_relaxed_structure`` for the minimized structure, with the same
    # schema under ``qc["md_final"]`` when MD ran. It is advisory: a failed
    # check never flips ``success``. ``qc_passed`` is None when QC did not run.
    qc: Optional[dict] = None
    qc_passed: Optional[bool] = None

    ncaa_bond_order_source: dict = field(default_factory=dict)
    dropped_protein_chains: list = field(default_factory=list)

    def _qc_failed_checks(self) -> list:
        """Names of failed QC checks, ``md_final:`` prefixed for the MD frame."""
        if not self.qc:
            return []
        failed = list(self.qc.get("failed", []))
        failed += [f"md_final:{name}" for name in (self.qc.get("md_final") or {}).get("failed", [])]
        return failed

    def _qc_check_rows(self) -> list:
        """One row per QC check (a list, so the CSV flattening leaves it out).

        Empty, not None, when QC did not run: the flattening turns a None into a
        column of its own, which would then exist only for failed runs.
        """
        if not self.qc:
            return []
        rows = []
        for stage, block in (("minimized", self.qc), ("md_final", self.qc.get("md_final"))):
            for name, check in ((block or {}).get("checks") or {}).items():
                rows.append({"stage": stage, "check": name, **check})
        return rows

    def to_dict(self) -> dict:
        """Convert result to a flat dictionary for CSV export.

        ``qc_passed`` and ``qc_failed_checks`` (comma-separated names) are scalar
        columns; ``qc_checks`` lists every check with its value and is kept for
        the JSON output.
        """
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
            "receptor_rmsd_md_final": self.receptor_rmsd_md_final,
            "receptor_drift_mean": self.receptor_drift_mean,
            "pep_rec_com_distance_delta": self.pep_rec_com_distance_delta,
            "minimization_time_s": self.minimization_time_s,
            "md_time_s": self.md_time_s,
            "minimized_structure_path": self.minimized_structure_path,
            "md_final_structure_path": self.md_final_structure_path,
            "peptide_cyclic_bonds": self.peptide_cyclic_bonds,
            "platform": self.platform,
            "precision": self.precision,
            "platform_fallback_reason": self.platform_fallback_reason,
            "qc_passed": self.qc_passed,
            "qc_failed_checks": ",".join(self._qc_failed_checks()),
            "qc_checks": self._qc_check_rows(),
            "ncaa_bond_order_source": dict(self.ncaa_bond_order_source),
            "dropped_protein_chains": list(self.dropped_protein_chains),
        }
        if self.peptide_rmsf_per_residue is not None:
            d["peptide_rmsf_per_residue"] = json.dumps(self.peptide_rmsf_per_residue)
        return d


class ImplicitRelaxation(Relaxer):
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
        # Set by _setup_system: detection result for D-amino acids and N-methyl
        # residues, so run() can restore their names before saving.
        self._ns_info = None
        # Set by _get_platform, copied into the result by run().
        self._platform_used: Optional[str] = None
        self._precision_used: Optional[str] = None
        self._platform_fallback_reason: Optional[str] = None
        # Set by _setup_system from the GAFF template step, copied into the result.
        self._ncaa_bond_order_source: dict = {}

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

    # Residue names the base force field or a curated template already covers, so
    # none needs GAFF2. Residues, waters and ions share this set on purpose.
    _AMBER_STANDARD = frozenset(
        # Canonical amino acids + protonation variants (CYM included)
        STANDARD_AMINO_ACIDS
        | AMBER_STANDARD_VARIANTS
        # Our custom lactam residues
        | LACTAM_TEMPLATE_RESIDUES
        # Common capping groups and ions; NMA is the only N-methylated
        # residue listed
        | FORCE_FIELD_CAP_NAMES
        | {"NMA"}
        | WATER_NAMES_WITH_H2O
        | ION_NAMES_COMMON
    )

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
        from openff.toolkit import Molecule
        from rdkit import Chem

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
                Chem.SanitizeMol(rwmol)  # perceives double bonds / aromaticity
                # hydrogens_are_explicit=True prevents openff from adding
                # implicit H — the molecule must match the topology at this
                # stage (before addHydrogens), which has heavy atoms only.
                mol = Molecule.from_rdkit(
                    rwmol,
                    allow_undefined_stereo=True,
                    hydrogens_are_explicit=True,
                )
                result.append(mol)
                logger.info("  Auto-GAFF2: '%s' (%s heavy atoms)", res.name, mol.n_atoms)
            except Exception as exc:  # noqa: BLE001 - one residue; RDKit and openff raise many types
                logger.warning(
                    "  Warning: could not build GAFF2 molecule for '%s': "
                    "%s. Skipping (residue will be excluded from the system).",
                    res.name,
                    exc,
                )

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
        chain_sizes = []
        for chain in topology.chains():
            n_protein = sum(1 for r in chain.residues() if r.name in PROTEIN_RESIDUES)
            if n_protein > 0:
                chain_sizes.append((chain.id, n_protein))

        if not chain_sizes:
            raise ValueError("No protein chains found in structure")

        chain_sizes.sort(key=lambda x: x[1])

        peptide_chain = self.config.peptide_chain_id or chain_sizes[0][0]
        receptor_chain = self.config.receptor_chain_id or (
            chain_sizes[-1][0] if len(chain_sizes) > 1 else None
        )
        return peptide_chain, receptor_chain

    def _strip_heterogens(
        self,
        topology,
        positions,
        peptide_chain: str,
        receptor_chain: Optional[str],
        warn_cutoff_ang: float = 8.0,
        report: Optional[dict] = None,
    ):
        """Strip heterogens and the protein chains outside the pair; fill ``report``.

        ``report`` is filled as in ``io.structures.strip_heterogens`` and
        ``io.structures.drop_other_protein_chains`` (``dropped_protein_chains``).
        """
        from binding_metrics.io.structures import drop_other_protein_chains, strip_heterogens

        topology, positions = strip_heterogens(
            topology, positions, peptide_chain, receptor_chain, warn_cutoff_ang, report=report
        )
        return drop_other_protein_chains(
            topology, positions, peptide_chain, receptor_chain, report=report
        )

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

        self._import_openmm()
        self._ns_info = None
        self._ncaa_bond_order_source = {}
        self._dropped_protein_chains = []

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
            a
            for a, pos in zip(topology.atoms(), positions)
            if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6
        ]
        if origin_atoms:
            logger.info("  Removing %d origin-placeholder atoms...", len(origin_atoms))
            modeller.delete(origin_atoms)
            topology, positions = modeller.topology, modeller.positions

        # OpenMM bonds each residue to the next by name, whatever the distance, so a
        # chain break is closed by the minimisation. Say so; the run is unchanged.
        from binding_metrics.core.system import find_chain_breaks

        for gap in find_chain_breaks(topology, positions):
            logger.warning(
                "  Chain break in chain %s: residues %s and %s are %.2f A apart (C to N); "
                "the two are bonded and the minimisation pulls them together.",
                gap["chain"],
                gap["residue_before"],
                gap["residue_after"],
                gap["c_n_distance_angstrom"],
            )

        # --- Identify chains ---
        peptide_chain, receptor_chain = self._identify_chains(topology)

        # --- Strip heterogens (non-protein residues outside the two chains) and
        # any third protein chain: E_complex would include it, the isolated
        # components would not, and its termini and patches are not handled ---
        strip_report: dict = {}
        topology, positions = self._strip_heterogens(
            topology, positions, peptide_chain, receptor_chain, report=strip_report
        )
        self._dropped_protein_chains = list(strip_report.get("dropped_protein_chains", []))

        # --- Force field setup ---
        gb_file = (
            "implicit/gbn2.xml" if self.config.solvent_model == "gbn2" else "implicit/obc2.xml"
        )
        base_xmls = ["amber14-all.xml", "amber14/tip3pfb.xml", gb_file]
        ff = app.ForceField(*base_xmls)

        # Phosphorylated residues use AMBER phosaa params (net −2), not GAFF —
        # GAFF would perceive the phosphate as neutral and protonate it away.
        from binding_metrics.core import phosaa

        phosaa.register(ff)
        phosaa.ensure_hydrogen_definitions()

        # --- Non-standard residue patching (D-AAs and NMe-AAs, before H addition) ---
        from binding_metrics.core.nonstandard import (
            detect_nonstandard,
            load_nonstandard_xmls,
            patch_nonstandard,
        )

        ns_info = detect_nonstandard(topology, peptide_chain)
        self._ns_info = ns_info
        if not ns_info.is_empty:
            if ns_info.has_d_residues:
                names = [e["original_name"] for e in ns_info.d_residues]
                logger.info("  D-amino acids: %s → renamed to L counterparts for FF", names)
            if ns_info.has_nmethyl:
                names = [e["original_name"] for e in ns_info.nmethyl_residues]
                logger.info("  N-methylated residues: %s", names)
            topology, positions = patch_nonstandard(topology, positions, peptide_chain, ns_info)
            load_nonstandard_xmls(ff, ns_info)

        # --- Cyclic peptide topology patching (before hydrogen addition) ---
        # Always auto-detect: linear peptides pass through unchanged.
        # Must run here so addHydrogens sees the correct internal-residue topology.
        bond_info = []
        from binding_metrics.core.cyclic import (
            load_extra_xmls,
            patch_cyclic_topology,
            rename_disulfide_cys_to_cyx,
        )

        topology, positions, bond_info = patch_cyclic_topology(
            topology,
            positions,
            peptide_chain,
            hints=self.config.cyclic_bond_hints,
        )
        topology, positions = rename_disulfide_cys_to_cyx(topology, positions)
        if bond_info:
            logger.info("  Cyclic peptide detected — %d bond(s):", len(bond_info))
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
                logger.info(
                    "    %-14s: %s[%s].%s → %s[%s].%s",
                    b.cyclic_type,
                    rname1,
                    r1,
                    a1,
                    rname2,
                    r2,
                    a2,
                )
            load_extra_xmls(ff, bond_info)
        else:
            logger.info("  Linear peptide (no cyclization detected)")

        # --- GAFF2 for non-standard residues / small-molecule co-factors ---
        # Must run BEFORE addHydrogens so the generated residue templates (with
        # <ExternalBond> tags) match the backbone-embedded NCAA residues and so
        # their hydrogens are injected before createSystem.
        #
        # Note: GAFF2 is a general small-molecule force field. It works for
        # small organic co-factors and modified amino acids (as a pragmatic
        # approximation), but purpose-built parameters (e.g. CGENFF, RESP-fitted
        # charges) give higher accuracy for MD production runs.
        if self.config.small_molecules == "auto":
            # Auto path: generate ExternalBond residue templates for every exotic
            # NCAA (BMT, ABA, …) directly from the topology geometry. Residues
            # covered by ff14SB or curated templates (NMG/MVA/MLE, lactams, CYX)
            # are skipped. This rebuilds the topology to inject the NCAA hydrogens.
            from binding_metrics.core.gaff_ncaa import parameterize_ncaa_residues

            topology, positions, ncaa_xmls = parameterize_ncaa_residues(
                topology,
                positions,
                ff,
                gaff_version=self.config.small_molecule_ff,
                random_seed=self.config.random_seed,
            )
            self._ncaa_bond_order_source = dict(
                getattr(ncaa_xmls, "bond_order_source_by_residue", {})
            )
            if ncaa_xmls:
                logger.info(
                    "  Registered GAFF2 (%s) ExternalBond templates for %d NCAA residue(s).",
                    self.config.small_molecule_ff,
                    len(ncaa_xmls),
                )
        elif self.config.small_molecules:
            # Explicit list: free small-molecule co-factors (no backbone bonds) via
            # openmmforcefields' template generator.
            try:
                from openmmforcefields.generators import GAFFTemplateGenerator
            except ImportError as exc:
                raise ImportError(
                    "openmmforcefields is required for small_molecules support. "
                    "Install with: conda install -c conda-forge openmmforcefields openff-toolkit"
                ) from exc
            mols = self._coerce_molecules(self.config.small_molecules)
            if mols:
                # The generator's own AM1-BCC call lets sqm pick its diagonaliser by timing,
                # so the charges would change from run to run; set them here instead.
                from binding_metrics.core.gaff_ncaa import _assign_am1bcc_charges

                _assign_am1bcc_charges(mols, self.config.random_seed)
                gaff = GAFFTemplateGenerator(
                    molecules=mols, forcefield=self.config.small_molecule_ff
                )
                ff.registerTemplateGenerator(gaff.generator)
                logger.info(
                    "  Registered GAFF2 (%s) for %d small-molecule(s).",
                    self.config.small_molecule_ff,
                    len(mols),
                )
        else:
            nonstandard_names = [
                res.name for res in topology.residues() if res.name not in self._AMBER_STANDARD
            ]
            if nonstandard_names:
                unique = sorted(set(nonstandard_names))
                logger.warning("  [warning] Non-standard residues found: %s", ", ".join(unique))
                logger.warning("  These will likely cause 'No template found' errors.")
                logger.warning("  → Run with --small-molecules auto to parameterise via GAFF2.")
                logger.warning(
                    "  → Or run binding-metrics-prep --canonicalize to replace them first."
                )

        # --- Add hydrogens ---
        logger.info("  Adding hydrogens...")
        modeller = app.Modeller(topology, positions)

        # For cyclic peptides, pass explicit variants so addHydrogens uses
        # internal residue templates at the closure sites (not N/C-terminal).
        addh_variants = None
        if bond_info:
            from binding_metrics.core.cyclic import get_addh_variants

            addh_variants = get_addh_variants(modeller.topology, bond_info, peptide_chain)

        from binding_metrics.core.system import deterministic_hydrogen_placement

        seed = self.config.random_seed
        try:
            with deterministic_hydrogen_placement(seed):
                modeller.addHydrogens(ff, pH=self.config.ph, variants=addh_variants)
        except Exception as e:  # noqa: BLE001 - OpenMM raises many types; the retry below is logged
            logger.warning(
                "  Warning: addHydrogens(ff, pH=%s) failed (%s), "
                "retrying without ForceField (approximate H positions)...",
                self.config.ph,
                e,
            )
            try:
                with deterministic_hydrogen_placement(seed):
                    modeller.addHydrogens(pH=self.config.ph, variants=addh_variants)
            except Exception as e2:
                logger.error("addHydrogens failed with and without the force field (%s; %s)", e, e2)
                # Continuing would hand createSystem an un-protonated topology
                # and surface as an unrelated template error.
                raise RuntimeError(
                    f"addHydrogens failed with and without the force field: {e2}"
                ) from e2
        topology, positions = modeller.topology, modeller.positions

        # addHydrogens can strand a Cα H on the wrong face (random jitter +
        # frozen heavy atoms); left in place, minimization inverts the
        # stereocenter. No-op for inputs already repaired by prep_structure.
        from binding_metrics.core.system import repair_ca_hydrogen_chirality

        positions = repair_ca_hydrogen_chirality(topology, positions)

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

        Each restrained atom feels ``0.5 * k * |x - x0|^2`` with ``k`` from
        ``config.restraint_strength``. ``k`` is a global parameter named ``"k"``,
        so setting it to 0 in the context switches the restraint off without
        removing the force (Stage 3 does this).

        Args:
            system: OpenMM System
            topology: OpenMM Topology
            positions: Reference positions for restraints
            backbone_only: If True, only restrain backbone atoms (N, CA, C, O)

        Returns:
            Force index in the system
        """
        restraint = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        restraint.addGlobalParameter(
            "k",
            self.config.restraint_strength * unit.kilojoules_per_mole / unit.nanometer**2,
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        for atom in topology.atoms():
            if backbone_only and atom.name not in BACKBONE_HEAVY_ATOM_NAMES:
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

        # Kabsch (1976): H = P^T Q = U S V^T gives R = V U^T, the rotation for
        # COLUMN vectors (R p_i ~ q_i). The coordinates here are rows, so the
        # rotated set is pos1 @ R.T; applying pos1 @ R rotates by the inverse
        # and inflates the RMSD of any pair that is not already superposed.
        H = pos1.T @ pos2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return float(np.sqrt(np.mean(np.sum((pos1 @ R.T - pos2) ** 2, axis=1))) * 10)

    def _compute_rmsf(self, trajectory_positions, atom_indices) -> np.ndarray:
        """Compute per-atom RMSF from a list of trajectory frame positions.

        RMSF_i = sqrt(mean over frames of |x_i(t) - <x_i>|^2), where <x_i> is the
        mean position over the saved frames. Frames are not superposed first, so
        the value includes any overall drift of the complex.

        Args:
            trajectory_positions: List of OpenMM position sets (one per frame)
            atom_indices: Atom indices to include

        Returns:
            Per-atom RMSF array in Angstroms
        """
        all_pos = (
            np.array(
                [
                    np.array([[p.x, p.y, p.z] for p in frame])[atom_indices]
                    for frame in trajectory_positions
                ]
            )
            * 10  # nm -> Angstroms
        )
        mean_pos = all_pos.mean(axis=0)
        return np.sqrt(np.mean((all_pos - mean_pos) ** 2, axis=0).sum(axis=1))

    @staticmethod
    def _com_distance_delta(
        positions_a,
        positions_b,
        peptide_indices: list[int],
        receptor_indices: list[int],
    ) -> float:
        """Change in peptide–receptor Cα COM distance between two frames.

        Returns:
            Delta in Angstroms (positive = chains moved apart).
        """

        def _com_dist(positions):
            pos = np.array([[p.x, p.y, p.z] for p in positions]) * 10  # nm → Å
            pep_com = pos[peptide_indices].mean(axis=0)
            rec_com = pos[receptor_indices].mean(axis=0)
            return float(np.linalg.norm(pep_com - rec_com))

        return _com_dist(positions_b) - _com_dist(positions_a)

    def _run_cyclic_warmup(
        self,
        system,
        simulation,
        topology,
        ref_positions,
        peptide_chain: str,
        omega_indices,
        warmup_ps: float = CYCLIC_WARMUP_PS,
    ) -> None:
        """Run short restrained MD to preserve ring conformation on velocity init.

        Adds backbone φ/ψ dihedral restraints (cosine form) centred on the
        minimised structure, runs ``warmup_ps`` picoseconds with a progressive
        three-phase release, then removes all restraint forces before returning.
        The force constants and the reason for the release schedule are with
        ``WARMUP_PHI_PSI_K_KJ_MOL`` and ``WARMUP_OMEGA_K_KJ_MOL``.

        Args:
            system: OpenMM System (modified in-place; forces are removed after warmup).
            simulation: Active Simulation context.
            topology: OpenMM Topology.
            ref_positions: Reference positions (minimised) for measuring reference angles.
            peptide_chain: Peptide chain ID.
            omega_indices: 4-tuple of atom indices for the closure ω dihedral,
                or None (ω restraint is skipped when None).
            warmup_ps: Total warmup MD duration in picoseconds
                (default ``CYCLIC_WARMUP_PS``).
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
            n_i = _idx(res, "N")
            ca_i = _idx(res, "CA")
            c_i = _idx(res, "C")
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
            return  # nothing to restrain

        # Measure reference dihedrals from minimised positions
        ref_pos = np.array([[p.x, p.y, p.z] for p in ref_positions])

        def _dihedral_rad(p, i1, i2, i3, i4):
            b1 = p[i2] - p[i1]
            b2 = p[i3] - p[i2]
            b3 = p[i4] - p[i3]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            m1 = np.cross(n1, b2 / np.linalg.norm(b2))
            x = np.dot(n1, n2)
            y = np.dot(m1, n2)
            return math.atan2(y, x)

        # Build torsion force: V = k * (1 - cos(θ - θ0))  →  harmonic near θ0
        phi_psi_k_kj_mol = WARMUP_PHI_PSI_K_KJ_MOL
        omega_k_kj_mol = WARMUP_OMEGA_K_KJ_MOL
        torsion_force = openmm.CustomTorsionForce("k_phi * (1 - cos(theta - theta0))")
        torsion_force.addGlobalParameter(
            "k_phi",
            phi_psi_k_kj_mol[0] * unit.kilojoules_per_mole,
        )
        torsion_force.addPerTorsionParameter("theta0")

        for quad in phi_quads + psi_quads:
            theta0 = _dihedral_rad(ref_pos, *quad)
            torsion_force.addTorsion(*quad, [theta0])

        # Add ω restraint for the closure bond (if available)
        omega_force = None
        omega_idx = None
        if omega_indices is not None:
            omega_force = openmm.CustomTorsionForce("k_omega * (1 - cos(theta - theta0_omega))")
            omega_force.addGlobalParameter(
                "k_omega",
                omega_k_kj_mol[0] * unit.kilojoules_per_mole,
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

        # Phase 2: 20 % of the initial force constant (second quarter)
        simulation.context.setParameter("k_phi", phi_psi_k_kj_mol[1] * unit.kilojoules_per_mole)
        if omega_force is not None:
            simulation.context.setParameter("k_omega", omega_k_kj_mol[1] * unit.kilojoules_per_mole)
        simulation.step(warmup_steps // 4)

        # Phase 3: 2 % of the initial force constant (last quarter)
        simulation.context.setParameter("k_phi", phi_psi_k_kj_mol[2] * unit.kilojoules_per_mole)
        if omega_force is not None:
            simulation.context.setParameter("k_omega", omega_k_kj_mol[2] * unit.kilojoules_per_mole)
        simulation.step(warmup_steps - warmup_steps // 2 - warmup_steps // 4)

        # Remove restraint forces so production MD is unrestrained.
        # Remove in descending index order to avoid renumbering the lower one.
        indices_to_remove = sorted(
            [i for i in [torsion_idx, omega_idx] if i is not None], reverse=True
        )
        for i in indices_to_remove:
            system.removeForce(i)
        simulation.context.reinitialize(preserveState=True)

    @staticmethod
    def _qc_snapshot(sample_id: str, topology, positions):
        """Snapshot for the structural QC, or None if it cannot be built."""
        from binding_metrics.protocols.qc import AtomSnapshot

        try:
            return AtomSnapshot.from_topology(topology, positions)
        except (ValueError, ImportError) as exc:
            logger.warning("[%s] structural QC snapshot failed: %s", sample_id, exc)
            return None

    def _structural_qc(self, sample_id: str, reference, topology, positions, **kwargs) -> dict:
        """Run the structural QC of ``positions`` against ``reference``.

        The QC only annotates the result, so a failure to run it is recorded as
        ``{"passed": None, "reason": ...}`` and never fails the relaxation.
        """
        from binding_metrics.protocols.qc import AtomSnapshot, check_relaxed_structure

        if reference is None:
            return {"passed": None, "failed": [], "checks": {}, "reason": "no reference snapshot"}
        try:
            return check_relaxed_structure(
                reference, AtomSnapshot.from_topology(topology, positions), **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - advisory: a QC bug must not fail a relaxation
            logger.warning("[%s] structural QC could not run: %s", sample_id, exc, exc_info=True)
            return {
                "passed": None,
                "failed": [],
                "checks": {},
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _get_platform(self):
        """Get the OpenMM compute platform, falling back to CPU if CUDA fails.

        CUDA runs in "mixed" precision: forces are computed in single precision
        and the integration is done in double precision, the usual OpenMM
        setting for MD on GPUs (Eastman et al., PLoS Comput. Biol. 13,
        e1005659, 2017). A context is created once as a probe, because a driver or
        PTX mismatch only shows up at context creation, not at platform lookup.

        Besides returning ``(platform, properties)``, records what was chosen in
        ``self._platform_used``, ``self._precision_used`` and
        ``self._platform_fallback_reason`` so :meth:`run` can put them in the
        result.
        """
        self._import_openmm()
        self._platform_fallback_reason = None
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
                logger.info("  Platform: CUDA (mixed precision)")
                self._platform_used = "CUDA"
                self._precision_used = properties["CudaPrecision"]
                return platform, properties
            except Exception as e:  # noqa: BLE001 - any CUDA failure falls back to CPU, recorded
                logger.warning("CUDA requested but unavailable, falling back to CPU: %s", e)
                self._platform_fallback_reason = f"{type(e).__name__}: {e}"
                logger.warning("  Warning: CUDA unavailable (%s), falling back to CPU.", e)
        logger.info("  Platform: CPU")
        self._platform_used = "CPU"
        self._precision_used = None
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
        from binding_metrics.core.nonstandard import restore_nonstandard_names
        from binding_metrics.io.structures import save_cif

        self._import_openmm()

        if sample_id is None:
            sample_id = input_path.stem

        result = RelaxationResult(sample_id=sample_id, success=False)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("[%s] Preparing system...", sample_id)
            system, topology, positions, bond_info = self._setup_system(input_path)
            result.ncaa_bond_order_source = dict(self._ncaa_bond_order_source)
            result.dropped_protein_chains = list(self._dropped_protein_chains)

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
                    res_id = (
                        _chain_res_ids.get(ch, [None] * (idx + 1))[idx]
                        if idx < len(_chain_res_ids.get(ch, []))
                        else idx
                    )
                    return f"{ch}:{res_id}:{name}"

                result.peptide_cyclic_bonds = [
                    {
                        "type": b.cyclic_type,
                        "atom1": _fmt_atom(b.atom1_id),
                        "atom2": _fmt_atom(b.atom2_id),
                    }
                    for b in bond_info
                ]

            peptide_chain, receptor_chain = self._identify_chains(topology)

            # Integrator. Seed the Langevin noise stream for reproducibility
            # (config.random_seed None => leave unseeded for fresh randomness).
            integrator = openmm.LangevinMiddleIntegrator(
                self.config.md_temperature_k * unit.kelvin,
                self.config.md_friction / unit.picosecond,
                self.config.md_timestep_fs * unit.femtosecond,
            )
            if self.config.random_seed is not None:
                integrator.setRandomNumberSeed(self.config.random_seed)

            platform, properties = self._get_platform()
            result.platform = self._platform_used
            result.precision = self._precision_used
            result.platform_fallback_reason = self._platform_fallback_reason
            simulation = app.Simulation(topology, system, integrator, platform, properties)
            simulation.context.setPositions(positions)

            # Reference for the structural QC: geometry and energy of the system
            # exactly as it enters minimization.
            qc_reference = self._qc_snapshot(sample_id, topology, positions)
            energy_reference_kj_mol = (
                simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )

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
                    except Exception as e:  # noqa: BLE001 - per-bond isolation, logged
                        logger.warning(
                            "[%s]   Warning: could not resolve closure atoms: %s", sample_id, e
                        )

            # --- Multi-stage minimization ---
            n_stages = "4" if closure_indices_list else "3"
            logger.info("[%s] Minimizing (%s stages)...", sample_id, n_stages)
            min_start = time.time()

            # Stage 0 (cyclic only): relax all closure bond geometries before Stage 1.
            # One CustomBondForce covers all closure bonds (monocyclic or bicyclic+).
            if closure_indices_list:
                logger.info(
                    "[%s]   Stage 0: Closure bond geometry relaxation (%d bond(s))",
                    sample_id,
                    len(closure_indices_list),
                )
                closure_force = openmm.CustomBondForce("0.5 * k_closure * (r - r0_closure)^2")
                closure_force.addGlobalParameter(
                    "k_closure",
                    CLOSURE_RESTRAINT_K_KJ_MOL_NM2 * unit.kilojoules_per_mole / unit.nanometer**2,
                )
                closure_force.addGlobalParameter(
                    "r0_closure", CLOSURE_BOND_TARGET_NM * unit.nanometers
                )
                for ci in closure_indices_list:
                    closure_force.addBond(ci[0], ci[1], [])
                system.addForce(closure_force)
                simulation.context.reinitialize(preserveState=True)
                simulation.minimizeEnergy(maxIterations=CLOSURE_MINIMIZATION_MAX_ITERATIONS)
                simulation.context.setParameter("k_closure", 0.0)

            logger.info("[%s]   Stage 1: Global relaxation", sample_id)
            simulation.minimizeEnergy(
                maxIterations=self.config.min_steps_initial,
                tolerance=self.config.min_tolerance
                * STAGE1_TOLERANCE_FACTOR
                * unit.kilojoules_per_mole
                / unit.nanometer,
            )

            logger.info("[%s]   Stage 2: Backbone-restrained optimization", sample_id)
            # The restraints are centred on the input backbone (``positions``),
            # so side chains settle while the backbone stays near the input.
            self._add_restraints(system, topology, positions, backbone_only=True)
            simulation.context.reinitialize(preserveState=True)
            simulation.minimizeEnergy(
                maxIterations=self.config.min_steps_restrained,
                tolerance=self.config.min_tolerance
                * STAGE2_TOLERANCE_FACTOR
                * unit.kilojoules_per_mole
                / unit.nanometer,
            )

            logger.info("[%s]   Stage 3: Final unrestrained refinement", sample_id)
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

            result.qc = self._structural_qc(
                sample_id,
                qc_reference,
                topology,
                minimized_positions,
                energy_kj_mol=result.potential_energy_minimized,
                energy_before_kj_mol=energy_reference_kj_mol,
            )

            # The force field needed L / template names for D-amino acids and
            # N-methyl residues (DAL -> ALA, SAR -> NMG). Put the input names
            # back before anything is written, or downstream Ramachandran
            # scoring treats every D-residue as L. Residue names are not used by
            # the MD code below, so the topology stays restored for the rest of
            # the run.
            if self._ns_info is not None:
                restore_nonstandard_names(topology, self._ns_info)

            # Save minimized structure — pass input as source so auth chain IDs
            # and residue numbers from the (already-prepped) input are preserved.
            min_path = output_dir / f"{sample_id}_minimized.cif"
            src = input_path if input_path.suffix.lower() in (".cif", ".mmcif") else None
            save_cif(topology, minimized_positions, min_path, source_cif_path=src)
            result.minimized_structure_path = str(min_path)
            logger.info("[%s] Minimized: %.1f kJ/mol", sample_id, result.potential_energy_minimized)

            # --- MD simulation ---
            if self.config.md_duration_ps > 0:
                logger.info("[%s] Running MD (%s ps)...", sample_id, self.config.md_duration_ps)
                md_start = time.time()
                # Seed the initial Maxwell-Boltzmann velocities too, else MD is
                # nondeterministic even with a seeded integrator.
                if self.config.random_seed is not None:
                    simulation.context.setVelocitiesToTemperature(
                        self.config.md_temperature_k * unit.kelvin,
                        self.config.random_seed,
                    )
                else:
                    simulation.context.setVelocitiesToTemperature(
                        self.config.md_temperature_k * unit.kelvin
                    )

                # Cyclic warmup: 10 ps backbone φ/ψ dihedral restraints to
                # preserve ring conformation during velocity initialisation.
                if closure_indices_list:
                    logger.info(
                        "[%s]   Cyclic warmup: 10 ps restrained MD (backbone φ/ψ restraints)...",
                        sample_id,
                    )
                    self._run_cyclic_warmup(
                        system,
                        simulation,
                        topology,
                        minimized_positions,
                        peptide_chain,
                        omega_indices,
                    )

                steps_per_save = int(
                    self.config.md_save_interval_ps * 1000 / self.config.md_timestep_fs
                )
                total_saves = _md_frame_count(
                    self.config.md_duration_ps, self.config.md_save_interval_ps
                )

                trajectory_positions = []
                md_energies = []
                for _ in range(total_saves):
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

                # Receptor Cα RMSD / drift
                receptor_ca_indices = [
                    a.index
                    for a in topology.atoms()
                    if a.residue.chain.id == receptor_chain and a.name == "CA"
                ]
                if receptor_ca_indices:
                    result.receptor_rmsd_md_final = self._compute_rmsd(
                        final_positions, minimized_positions, receptor_ca_indices
                    )
                    per_frame_rmsd = [
                        self._compute_rmsd(frame, minimized_positions, receptor_ca_indices)
                        for frame in trajectory_positions
                    ]
                    result.receptor_drift_mean = float(np.mean(per_frame_rmsd))

                # Peptide–receptor Cα COM distance delta (minimized → final)
                if peptide_ca_indices and receptor_ca_indices:
                    result.pep_rec_com_distance_delta = self._com_distance_delta(
                        minimized_positions,
                        final_positions,
                        peptide_ca_indices,
                        receptor_ca_indices,
                    )

                # Save final MD structure — preserve auth IDs from input
                final_path = output_dir / f"{sample_id}_md_final.cif"
                save_cif(topology, final_positions, final_path, source_cif_path=src)
                result.md_final_structure_path = str(final_path)

                # MD legitimately moves away from the minimum, so the frame is
                # compared with the minimized structure and has no RMSD bound.
                result.qc["md_final"] = self._structural_qc(
                    sample_id,
                    self._qc_snapshot(sample_id, topology, minimized_positions),
                    topology,
                    final_positions,
                    energy_kj_mol=result.potential_energy_md_avg,
                    max_rmsd_angstrom=None,
                )

            outcomes = [result.qc["passed"], (result.qc.get("md_final") or {}).get("passed", True)]
            result.qc_passed = None if None in outcomes else all(outcomes)
            result.success = True

        except Exception as e:  # noqa: BLE001 - one failed sample is a result, see error_message
            result.error_message = f"{type(e).__name__}: {e}"
            logger.warning("[%s] ERROR: %s", sample_id, result.error_message)
            traceback.print_exc()

        return result


def run_implicit_relaxation(
    input_path: Path,
    output_dir: Optional[Path] = None,
    *,
    config: Optional["RelaxationConfig"] = None,
    sample_id: Optional[str] = None,
    **config_kwargs,
) -> "RelaxationResult":
    """Function wrapper around :class:`ImplicitRelaxation` for generic invocation.

    ``ImplicitRelaxation`` takes its ``RelaxationConfig`` in ``__init__`` and the
    structure path in ``.run()``, so it cannot be driven by a generic runner that
    forwards ``input_path`` as a single keyword. This thin wrapper exposes the
    interface the metric registry advertises: ``input_path`` (and ``output_dir``)
    as call kwargs, with the config either passed explicitly or built from keyword
    overrides taken from the manifest ``md`` block.

    Args:
        input_path: Path to input CIF or PDB structure file.
        output_dir: Directory for output structures. Defaults to a temporary
            directory when omitted (e.g. pure benchmark timing runs).
        config: Optional pre-built RelaxationConfig. If omitted, one is created
            from ``config_kwargs``.
        sample_id: Identifier for this run (defaults to the input file stem).
        **config_kwargs: Forwarded to ``RelaxationConfig`` when ``config`` is None.

    Returns:
        RelaxationResult with energies, RMSD/RMSF, timing, and output paths.
    """
    import tempfile

    if config is None:
        config = RelaxationConfig(**config_kwargs)
    elif config_kwargs:
        raise TypeError(
            "run_implicit_relaxation: pass either `config` or config keyword overrides, not both"
        )

    relaxer = ImplicitRelaxation(config)

    if output_dir is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            return relaxer.run(Path(input_path), Path(tmp_dir), sample_id=sample_id)
    return relaxer.run(Path(input_path), Path(output_dir), sample_id=sample_id)


def _run_one(
    relaxer: "ImplicitRelaxation",
    input_path: Path,
    output_dir: Path,
    sample_id: Optional[str],
    results_json: Optional[Path],
    model_num: Optional[int],
) -> "RelaxationResult":
    """Extract one model (if needed), run relaxation, write JSON, return result."""
    from binding_metrics.io.structures import extract_model_to_tempfile
    from binding_metrics.protocols.report import _json_default

    tmp_path: Optional[Path] = None
    if model_num is not None:
        tmp_path = extract_model_to_tempfile(input_path, model_num)
        if tmp_path != input_path:
            print(f"  Extracted model {model_num} → {tmp_path}")
            if sample_id is None:
                sample_id = f"{input_path.stem}_model{model_num}"

    try:
        effective_input = tmp_path if tmp_path is not None else input_path
        result = relaxer.run(effective_input, output_dir, sample_id=sample_id)
    finally:
        if tmp_path is not None and tmp_path != input_path:
            tmp_path.unlink(missing_ok=True)

    if results_json is not None:
        rp = Path(results_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as _fh:
            json.dump(result.to_dict(), _fh, indent=2, default=_json_default)
        print(f"  Results:   {rp}")

    if result.success:
        print("\nSUCCESS")
        if result.minimized_structure_path:
            print(f"  Minimized: {result.minimized_structure_path}")
        if result.md_final_structure_path:
            print(f"  MD final:  {result.md_final_structure_path}")
    else:
        print(f"\nFAILED: {result.error_message}")

    return result


def main():
    from binding_metrics.utils import configure_logging

    configure_logging()

    from binding_metrics.cli import add_config_arg, parse_args_with_config, small_molecules_arg
    from binding_metrics.metrics._common import ChainAliasAction

    parser = argparse.ArgumentParser(
        description="Implicit solvent MD relaxation for protein complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CIF or PDB file")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--md-duration-ps",
        type=float,
        default=DEFAULT_MD_DURATION_PS,
        help="MD duration in ps (0 to minimize only)",
    )
    parser.add_argument(
        "--md-save-interval-ps",
        type=float,
        default=DEFAULT_MD_SAVE_INTERVAL_PS,
        help="Frame save interval in ps",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0, help="Simulation temperature in K"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default=DEFAULT_DEVICE, help="Compute device"
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=DEFAULT_PH,
        help=f"pH for hydrogen addition (default {DEFAULT_PH})",
    )
    parser.add_argument(
        "--solvent-model", choices=["obc2", "gbn2"], default="obc2", help="Implicit solvent model"
    )
    parser.add_argument(
        "--peptide-chain",
        "--binder-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Peptide chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--receptor-chain",
        "--target-chain",
        action=ChainAliasAction,
        type=str,
        default=None,
        help="Receptor chain ID (auto-detect if omitted)",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample identifier (defaults to input file stem)",
    )
    parser.add_argument(
        "--small-molecules",
        type=small_molecules_arg,
        default="auto",
        help="Non-standard residue parameterisation. 'auto' (default) "
        "builds GAFF2 ExternalBond templates for every exotic NCAA; "
        "'none' disables it.",
    )
    parser.add_argument(
        "--random-seed",
        type=str,
        default=str(DEFAULT_RANDOM_SEED),
        metavar="INT|none",
        help="Seed for stochastic steps (hydrogen placement, MD "
        f"velocities/thermostat). Default {DEFAULT_RANDOM_SEED} "
        "(reproducible); 'none' for fresh randomness each run.",
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=int,
        default=None,
        help="Extract and relax a single model from a multi-model CIF (1-based)",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Relax every model in a multi-model CIF; "
        "errors on single-model files if given explicitly. "
        "sample-id is auto-set to <stem>_model<N> for each.",
    )

    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Path to write relax results as JSON. "
        "Omit to skip JSON output (ignored with --all-models).",
    )
    from binding_metrics.cli import add_log_file_arg

    add_log_file_arg(parser)
    add_config_arg(parser)
    args = parse_args_with_config(parser)

    if args.all_models and args.sample_id is not None:
        parser.error("--sample-id cannot be used with --all-models (IDs are auto-generated)")

    if args.random_seed.strip().lower() in ("none", "random", "off"):
        seed = None
    else:
        seed = int(args.random_seed)

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
            small_molecules=args.small_molecules,
            random_seed=seed,
        )
        relaxer = ImplicitRelaxation(config)

        if args.all_models:
            from binding_metrics.io.structures import detect_models, merge_cif_models

            models = detect_models(args.input)
            if len(models) == 1:
                print(
                    f"  [warning] --all-models: only one model found ({models[0]}); "
                    f"proceeding with model {models[0]}."
                )
            failed: list[int] = []
            minimized: list[tuple[int, Path]] = []
            for m in models:
                print(f"\n{'=' * 60}\n  Model {m}\n{'=' * 60}")
                result = _run_one(
                    relaxer,
                    args.input,
                    args.output_dir,
                    sample_id=None,
                    results_json=None,
                    model_num=m,
                )
                if not result.success:
                    failed.append(m)
                else:
                    out_cif = result.md_final_structure_path or result.minimized_structure_path
                    if out_cif:
                        minimized.append((m, Path(out_cif)))

            # Merge all minimized models into a single multi-model CIF
            if minimized:
                merged_path = args.output_dir / f"{args.input.stem}_minimized.cif"
                merge_cif_models(minimized, merged_path)
                for _, p in minimized:
                    p.unlink(missing_ok=True)
                print(f"\n  Merged output ({len(minimized)} model(s)): {merged_path}")

            if failed:
                print(f"\nFAILED models: {failed}", file=sys.stderr)
                sys.exit(1)
        else:
            result = _run_one(
                relaxer,
                args.input,
                args.output_dir,
                sample_id=args.sample_id,
                results_json=args.results_json,
                model_num=args.model,
            )
            if not result.success:
                sys.exit(1)


if __name__ == "__main__":
    main()
