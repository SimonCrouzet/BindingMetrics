"""Central registry of all implemented metrics.

Each MetricSpec describes one metric: how to import it, what kind of input
it expects, which chain arguments it accepts, and which file formats it supports.

This module is the single source of truth for metric discovery. Tools such as
the benchmark runner iterate METRICS rather than hardcoding function names.

Input types
-----------
static_structure
    Accepts a single structure file (PDB or CIF as noted in ``formats``).
    Chain assignment is optional — metrics auto-detect if not provided.
trajectory
    Requires a trajectory file AND a topology file (e.g. MDTraj-based metrics).
openfold_json
    Reads an OpenFold3 output directory / JSON file; no structure file needed
    (except ``interface_pae``, which also takes the predicted structure).
atom_array
    Takes an already loaded ``biotite.structure.AtomArray`` (``path_arg`` names
    the kwarg that receives it). These are the building blocks that the
    ``interface`` metric calls internally.
predicted_structure
    Takes the path of a predicted structure plus per-atom confidence arrays
    (for example ``plddt_per_atom`` from the ``openfold`` metric) that the
    caller passes as extra keyword arguments.

Chain modes
-----------
none        No chain arguments.
single      One chain (``chain_arg`` kwarg, defaults to peptide/designed chain).
interface   Two chains (``peptide_chain_arg`` + ``receptor_chain_arg``). One of
            the two may be None when the function needs only one role
            (``receptor_quality`` takes only the receptor).
interface_2paths
    Two chains AND two structure paths (``path_arg`` + ``secondary_path_arg``).
    The benchmark passes the same path for both to measure pure compute cost.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

InputType = Literal[
    "static_structure",
    "trajectory",
    "md_simulation",
    "openfold_json",
    "atom_array",
    "predicted_structure",
]
ChainMode = Literal["none", "single", "interface", "interface_2paths"]


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric function.

    Attributes
    ----------
    name:
        Short identifier used as a key (e.g. in benchmark output).
    import_path:
        ``"module.path:function_name"`` — resolved lazily so optional
        dependencies (openmm, mdtraj, …) are never imported just by loading
        this registry.
    description:
        One-line human-readable description.
    input_type:
        Category of input this metric expects.
    chain_mode:
        How chain identifiers are passed to the function.
    formats:
        File formats accepted by this metric (relevant for static_structure).
    path_arg:
        Name of the kwarg that receives the primary file path.
    secondary_path_arg:
        Name of the kwarg for a second path (``interface_2paths`` only).
    chain_arg:
        Kwarg name for single-chain metrics.
    peptide_chain_arg:
        Kwarg name for the peptide / designed chain.
    receptor_chain_arg:
        Kwarg name for the receptor chain.
    """

    name: str
    import_path: str
    description: str
    input_type: InputType
    chain_mode: ChainMode = "none"
    formats: tuple[str, ...] = ("pdb", "cif")
    path_arg: str = "cif_path"
    secondary_path_arg: Optional[str] = None
    chain_arg: Optional[str] = None
    peptide_chain_arg: Optional[str] = None
    receptor_chain_arg: Optional[str] = None

    def load(self) -> Callable:
        """Import and return the metric function (lazy)."""
        module_path, fn_name = self.import_path.split(":")
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)

    def call(self, **kwargs: Any) -> Any:
        """Call the metric function with the given keyword arguments."""
        return self.load()(**kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS: list[MetricSpec] = [
    # --- Static structure metrics -------------------------------------------
    MetricSpec(
        name="interface",
        import_path="binding_metrics.metrics.interface:compute_interface_metrics",
        description="Binding interface: ΔSASA, ΔG_int, H-bonds, salt bridges (PISA approach)",
        input_type="static_structure",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        peptide_chain_arg="design_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="coulomb",
        import_path="binding_metrics.metrics.electrostatics:compute_coulomb_cross_chain",
        description="Coulomb cross-chain interaction energy (formal charges, pH 7)",
        input_type="static_structure",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="ramachandran",
        import_path="binding_metrics.metrics.geometry:compute_ramachandran",
        description="Ramachandran φ/ψ dihedral quality: favoured / allowed / outlier %",
        input_type="static_structure",
        chain_mode="single",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        chain_arg="chain",
    ),
    MetricSpec(
        name="omega",
        import_path="binding_metrics.metrics.geometry:compute_omega_planarity",
        description="Peptide bond ω planarity: mean deviation from 180°, outlier fraction",
        input_type="static_structure",
        chain_mode="single",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        chain_arg="chain",
    ),
    MetricSpec(
        name="shape_complementarity",
        import_path="binding_metrics.metrics.geometry:compute_shape_complementarity",
        description="Shape complementarity Sc (Lawrence & Colman 1993) via surface dots",
        input_type="static_structure",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="void_volume",
        import_path="binding_metrics.metrics.geometry:compute_buried_void_volume",
        description="Buried void volume at the interface (grid flood-fill)",
        input_type="static_structure",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="structure_rmsd",
        import_path="binding_metrics.metrics.comparison:compute_structure_rmsd",
        description="Kabsch-aligned RMSD between two structures (all-atom and backbone)",
        input_type="static_structure",
        chain_mode="interface_2paths",
        formats=("pdb", "cif"),
        path_arg="initial_path",
        secondary_path_arg="processed_path",
        peptide_chain_arg="design_chain",
    ),
    MetricSpec(
        name="delta_sasa_static",
        import_path="binding_metrics.metrics.sasa:compute_delta_sasa_static",
        description="Buried SASA on binding for one static structure (biotite, probe 1.4 Å)",
        input_type="static_structure",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="cif_path",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="receptor_quality",
        import_path="binding_metrics.metrics.receptor_quality:compute_receptor_quality",
        description=(
            "MolProbity-style receptor quality: Ramachandran, clashscore, rotamers, "
            "Cβ deviation, bond geometry, force-field energy"
        ),
        input_type="static_structure",
        chain_mode="interface",  # receptor only: the function has no peptide-chain argument
        formats=("pdb", "cif"),
        path_arg="path",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="evobind_adversarial",
        import_path="binding_metrics.metrics.evobind:compute_evobind_adversarial_check",
        description=(
            "EvoBind adversarial check: binder centre-of-mass shift between two "
            "predictions after receptor Cα superposition (Bryant et al. 2025)"
        ),
        input_type="static_structure",
        chain_mode="interface_2paths",
        formats=("pdb", "cif"),
        path_arg="design_structure_path",
        secondary_path_arg="afm_structure_path",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
    ),
    # --- In-memory structure and prediction inputs --------------------------
    # These do not read a file path. ``path_arg`` names the kwarg that receives
    # the AtomArray, resp. the predicted-structure path, and the caller supplies
    # the extra arrays the function needs.
    MetricSpec(
        name="hbonds",
        import_path="binding_metrics.metrics.polar_contacts:compute_hbonds",
        description="Cross-chain H-bonds (Baker-Hubbard): count and distance/angle-weighted energy",
        input_type="atom_array",
        chain_mode="interface",
        formats=(),
        path_arg="atoms",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="saltbridges",
        import_path="binding_metrics.metrics.polar_contacts:compute_saltbridges",
        description="Cross-chain salt bridges: residue-pair count, bidentate count, Coulomb energy",
        input_type="atom_array",
        chain_mode="interface",
        formats=(),
        path_arg="atoms",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="evobind_score",
        import_path="binding_metrics.metrics.evobind:compute_evobind_score",
        description=(
            "EvoBind primary score: binder-to-interface distance divided by binder "
            "pLDDT/100 (Bryant et al. 2025)"
        ),
        input_type="predicted_structure",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="structure_path",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
    ),
    # --- Reference-based accuracy metrics -----------------------------------
    # These require a *reference* (native) structure and only make sense for
    # benchmarking / retrospective validation, not for scoring a design in
    # isolation. path_arg = predicted model, secondary_path_arg = reference.
    # DockQ performs its own automatic optimal chain-mapping search.
    MetricSpec(
        name="dockq",
        import_path="binding_metrics.metrics.dockq:compute_dockq_metrics",
        description="Reference-based CAPRI accuracy: DockQ, fnat, fnonnat, i-RMSD, L-RMSD",
        input_type="static_structure",
        chain_mode="interface_2paths",
        formats=("pdb", "cif"),
        path_arg="model_path",
        secondary_path_arg="reference_path",
    ),
    # --- Trajectory metrics -------------------------------------------------
    # All trajectory metrics receive topology_path from the manifest.
    # Chain arguments map to manifest fields resolved by the benchmark runner:
    #   "ligand_indices"   — atom indices for the ligand/peptide chain
    #   "receptor_indices" — atom indices for the receptor chain
    #   "receptor_chain"   — chain ID string (compute_receptor_drift only)
    # ligand_indices / receptor_indices are auto-computed from ligand_chain /
    # receptor_chain in the manifest, so users never have to supply raw indices.
    MetricSpec(
        name="interaction_energy",
        import_path="binding_metrics.metrics.energy:calculate_interaction_energy",
        description="Pairwise Coulomb + LJ interaction energy per frame (OpenMM)",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb",),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",  # resolved to indices by runner
        receptor_chain_arg="receptor_indices",
    ),
    MetricSpec(
        name="component_energies",
        import_path="binding_metrics.metrics.energy:calculate_component_energies",
        description="Separated electrostatic and vdW interaction energies per frame (OpenMM)",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb",),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",
        receptor_chain_arg="receptor_indices",
    ),
    MetricSpec(
        name="rmsd",
        import_path="binding_metrics.metrics.rmsd:calculate_rmsd",
        description="Per-frame RMSD relative to reference frame; auto-selects protein heavy atoms",
        input_type="trajectory",
        chain_mode="none",  # atom_indices optional, auto-detected
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
    ),
    MetricSpec(
        name="rmsf",
        import_path="binding_metrics.metrics.rmsd:calculate_rmsf",
        description="Per-atom RMSF; auto-selects protein heavy atoms",
        input_type="trajectory",
        chain_mode="none",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
    ),
    MetricSpec(
        name="ligand_rmsd",
        import_path="binding_metrics.metrics.rmsd:calculate_ligand_rmsd",
        description="Ligand RMSD after receptor alignment per frame",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",
        receptor_chain_arg="receptor_indices",
    ),
    MetricSpec(
        name="receptor_drift",
        import_path="binding_metrics.metrics.rmsd:compute_receptor_drift",
        description="Receptor backbone drift over trajectory: aligned and raw RMSD",
        input_type="trajectory",
        chain_mode="single",  # takes receptor_chain (chain ID string)
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        chain_arg="receptor_chain",
    ),
    MetricSpec(
        name="buried_sasa",
        import_path="binding_metrics.metrics.sasa:calculate_buried_sasa",
        description="Buried SASA upon binding per frame (MDTraj Shrake-Rupley)",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",
        receptor_chain_arg="receptor_indices",
    ),
    MetricSpec(
        name="contacts",
        import_path="binding_metrics.metrics.contacts:calculate_contacts",
        description="Interface heavy-atom contact count per frame",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",
        receptor_chain_arg="receptor_indices",
    ),
    MetricSpec(
        name="interface_sasa",
        import_path="binding_metrics.metrics.sasa:calculate_interface_sasa",
        description="Ligand, receptor, complex and buried SASA per frame (MDTraj Shrake-Rupley)",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",
        receptor_chain_arg="receptor_indices",
    ),
    MetricSpec(
        name="contact_residues",
        import_path="binding_metrics.metrics.contacts:calculate_contact_residues",
        description="Residues in interface contact over the trajectory",
        input_type="trajectory",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        peptide_chain_arg="ligand_indices",
        receptor_chain_arg="receptor_indices",
    ),
    # --- MD simulation ------------------------------------------------------
    # input_type="md_simulation": takes a single structure file (CIF or PDB),
    # runs the full relaxation pipeline (minimization + MD), and returns timing
    # from RelaxationResult.minimization_time_s / .md_time_s.
    # MD parameters (md_duration_ps, md_timestep_fs, device, …) are specified
    # per-entry in the manifest under the "md" key and forwarded to RelaxationConfig.
    MetricSpec(
        name="md_implicit",
        import_path="binding_metrics.protocols.relaxation:run_implicit_relaxation",
        description=(
            "Implicit solvent MD relaxation (AMBER ff14SB + OBC2/GBn2): "
            "3-stage minimization + Langevin MD"
        ),
        input_type="md_simulation",
        chain_mode="none",  # chains auto-detected; override via manifest
        formats=("pdb", "cif"),
        path_arg="input_path",
    ),
    # The per-structure counterpart of "interaction_energy": E_complex - E_peptide -
    # E_receptor by subsystem decomposition, after optional minimisation and MD
    # (the default ``modes`` include a short MD run, hence md_simulation).
    MetricSpec(
        name="structure_interaction_energy",
        import_path="binding_metrics.metrics.energy:compute_interaction_energy",
        description=(
            "Peptide-receptor interaction energy of one structure by subsystem "
            "decomposition (AMBER ff14SB + implicit solvent): raw, relaxed, after MD"
        ),
        input_type="md_simulation",
        chain_mode="interface",
        formats=("pdb", "cif"),
        path_arg="input_path",
        peptide_chain_arg="peptide_chain",
        receptor_chain_arg="receptor_chain",
    ),
    # --- OpenFold metrics ---------------------------------------------------
    MetricSpec(
        name="openfold",
        import_path="binding_metrics.metrics.openfold:compute_openfold_metrics",
        description="Parse OpenFold3 output: pLDDT, pAE, pTM, ipTM, GPDE, has_clash",
        input_type="openfold_json",
        chain_mode="none",
        formats=(),
        path_arg="output_dir",
    ),
    MetricSpec(
        name="interface_pae",
        import_path="binding_metrics.metrics.openfold:compute_interface_pae",
        description="Binder x receptor PAE slice from OpenFold3 confidences: mean and max",
        input_type="openfold_json",
        chain_mode="interface_2paths",
        formats=(),
        path_arg="confidences_path",
        secondary_path_arg="structure_path",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
    ),
]

# Fast lookup by name
METRICS_BY_NAME: dict[str, MetricSpec] = {m.name: m for m in METRICS}


def get_metric(name: str) -> MetricSpec:
    """Return the MetricSpec for *name*, raising KeyError if not found."""
    try:
        return METRICS_BY_NAME[name]
    except KeyError:
        available = ", ".join(METRICS_BY_NAME)
        raise KeyError(f"Unknown metric {name!r}. Available: {available}") from None


def metrics_by_input_type(input_type: InputType) -> list[MetricSpec]:
    """Return all metrics of a given input type."""
    return [m for m in METRICS if m.input_type == input_type]
