"""Registry of the metric functions, and what a consumer may rely on.

Each ``MetricSpec`` describes one metric function without importing it: where it
lives, what input it takes, how chains are passed, and how to read and schedule
the result. ``METRICS`` lists the specs, ``get_metric`` looks one up by name and
``metrics_by_input_type`` filters by input type. The functions are the API of
the package; the registry is the adapter that lets a generic caller drive them
without knowing their signatures, so metrics have no common base class.

What a consumer may rely on
---------------------------
* The field names ``name``, ``import_path``, ``description``, ``input_type``,
  ``chain_mode``, ``formats``, ``path_arg``, ``secondary_path_arg``,
  ``chain_arg``, ``peptide_chain_arg`` and ``receptor_chain_arg`` keep their
  meaning, and the values of the existing entries are pinned by a contract test.
  The registry grows by adding entries, input types and optional fields, so a
  consumer skips an input type it does not know instead of failing on it.
* ``spec.call(**kwargs)`` imports the function on first use and calls it with
  exactly those keyword arguments. It adds, renames and validates nothing, and
  returns what the function returns.
* ``binder_chain_arg`` and ``target_chain_arg`` are optional. When set, the
  function also accepts that keyword as an alias of its binder / target chain
  argument.
* Loading is lazy. This module imports no metric module; a spec imports its
  function when ``load`` or ``call`` runs. Both raise ``ImportError`` when an
  optional dependency is missing, and ``requires_extras`` names the extras to
  install.
* The metadata fields ``headline_key``, ``direction``, ``unit``, ``cost_class``,
  ``requires_extras`` and ``requires_gpu`` are optional. None (or an empty
  tuple) means "not declared", never a guess.

Building a call from a spec
---------------------------
Start with ``{spec.path_arg: <primary input>}``. Add
``{spec.secondary_path_arg: <second path>}`` when it is set. Then, by
``chain_mode``: ``single`` puts the chain in ``spec.chain_arg``; ``interface``
and ``interface_2paths`` put the binder in ``spec.peptide_chain_arg`` and the
target in ``spec.receptor_chain_arg``, each only when it is not None. Every
other parameter keeps the function's own default. Inputs the spec does not
declare are the caller's to supply: trajectory metrics also take a
``topology_path``, and the interface ones receive atom-index lists
(``ligand_indices``, ``receptor_indices``) where static metrics take chain IDs;
``evobind_score`` takes a ``plddt_per_atom`` array, and ``prediction`` takes the
``model`` and ``name`` of the prediction.

Not registered: the ``run_openfold*`` and ``prepare_*`` functions (they start or
prepare a model job and return no metric), the command-line ``main`` functions,
and helpers such as ``capri_class`` and ``detect_interface_chains``.

Input types
-----------
static_structure
    Accepts a single structure file (PDB or CIF as noted in ``formats``).
    Chain assignment is optional — metrics auto-detect if not provided.
trajectory
    Requires a trajectory file AND a topology file (e.g. MDTraj-based metrics).
md_simulation
    Takes a single structure file and runs a relaxation or MD protocol on it.
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

prediction_dir
    Reads the output directory of a structure-prediction model that has an adapter in
    ``binding_metrics.predictors`` (``path_arg`` names the kwarg that receives the
    directory). The caller also supplies ``model`` (a key of ``predictors.PARSERS``) and
    ``name`` (the prediction name); ``interface_pae`` and ``openfold`` remain the OpenFold3
    entries of ``openfold_json``.

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
    "prediction_dir",
]
ChainMode = Literal["none", "single", "interface", "interface_2paths"]
Direction = Literal["higher_is_better", "lower_is_better"]
CostClass = Literal["static", "structural", "md", "model"]

#: Allowed values of ``MetricSpec.direction``.
DIRECTIONS: tuple[str, ...] = ("higher_is_better", "lower_is_better")

#: Allowed values of ``MetricSpec.cost_class``, in rough order of cost.
COST_CLASSES: tuple[str, ...] = ("static", "structural", "md", "model")

#: Allowed values of ``MetricSpec.unit``. Spelled in ASCII so the strings survive
#: any CSV, JSON or terminal. ``nm`` and ``nm^2`` are the MDTraj units of the
#: trajectory metrics; the static metrics report angstrom.
KNOWN_UNITS: frozenset[str] = frozenset(
    {
        "kJ/mol",
        "kcal/mol",
        "angstrom",
        "angstrom^2",
        "angstrom^3",
        "nm",
        "nm^2",
        "degree",
        "percent",
        "fraction",
        "count",
        "dimensionless",
    }
)


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric function.

    A spec is immutable and holds names only: the function is imported when
    ``load`` or ``call`` runs. The kwarg names (``path_arg`` and the ``*_arg``
    fields) are the names of real parameters of the function; a test checks
    them against ``inspect.signature``.

    Attributes
    ----------
    name:
        Short identifier used as a key (e.g. in benchmark output). Unique in
        ``METRICS``.
    import_path:
        ``"module.path:function_name"`` — resolved lazily so optional
        dependencies (openmm, mdtraj, …) are never imported just by loading
        this registry.
    description:
        One-line human-readable description.
    input_type:
        Category of input this metric expects (see the module docstring).
    chain_mode:
        How chain identifiers are passed to the function.
    formats:
        File formats accepted by this metric (relevant for static_structure);
        empty when the metric reads no structure file.
    path_arg:
        Name of the kwarg that receives the primary input: a file path, a
        directory for ``openfold``, or the ``AtomArray`` for ``atom_array``
        metrics.
    secondary_path_arg:
        Name of the kwarg for a second path (``interface_2paths`` only).
    chain_arg:
        Kwarg name for single-chain metrics.
    peptide_chain_arg:
        Kwarg name for the peptide / designed chain (the binder). For
        trajectory metrics it receives the ligand atom indices.
    receptor_chain_arg:
        Kwarg name for the receptor chain (the target). For trajectory metrics
        it receives the receptor atom indices.
    headline_key:
        For a metric that returns a dict, the key that ``direction`` and
        ``unit`` describe; a dotted path (``"summary.molprobity_score"``)
        reaches into a nested dict. None when the function returns the
        quantity itself (an array or a scalar) or when the result is a bundle
        of descriptors with no single headline.
    direction:
        ``"higher_is_better"`` or ``"lower_is_better"`` for the headline value,
        None when there is no single accepted reading (bundles, counts whose
        preferred direction depends on the question).
    unit:
        Unit of the headline value, one of ``KNOWN_UNITS``; None when the value
        has no documented unit.
    cost_class:
        What has to run to obtain the result from a bare structure:
        ``"static"`` reads one structure and does geometry, no force field, no
        simulation, no learned model (seconds or less);
        ``"structural"`` builds a force-field system on one structure (adds
        hydrogens, evaluates or minimises the energy);
        ``"md"`` needs a molecular-dynamics trajectory, or runs a simulation
        (the analysis of an existing trajectory is cheap, the simulation that
        produces it is not);
        ``"model"`` needs a structure-prediction run (OpenFold3, AlphaFold),
        whose outputs the metric parses or compares.
    requires_extras:
        Names of the ``pip install binding-metrics[<extra>]`` extras (keys of
        ``[project.optional-dependencies]`` in ``pyproject.toml``) that the
        default use of the metric needs.
    requires_gpu:
        True when the metric runs its heavy computation on a CUDA device by
        default. A scheduling hint, read before the function is imported.
    binder_chain_arg:
        Name of the role-alias kwarg for the binder chain, ``"binder_chain"``,
        on a metric whose function accepts it; None where it does not (the
        trajectory metrics take atom indices, not chains). It means what
        ``peptide_chain_arg`` (or ``chain_arg`` for a single-chain metric)
        means, and passing both with different IDs is a ``ValueError``.
    target_chain_arg:
        The same for the target chain: ``"target_chain"`` as the alias of
        ``receptor_chain_arg``.
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
    headline_key: Optional[str] = None
    direction: Optional[Direction] = None
    unit: Optional[str] = None
    cost_class: Optional[CostClass] = None
    requires_extras: tuple[str, ...] = ()
    requires_gpu: bool = False
    binder_chain_arg: Optional[str] = None
    target_chain_arg: Optional[str] = None

    def load(self) -> Callable:
        """Import the module and return the metric function.

        Raises:
            ImportError: The module, or an optional dependency it needs, is not
                installed (``requires_extras`` names the extras).
            AttributeError: The module has no function of that name.
        """
        module_path, fn_name = self.import_path.split(":")
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)

    def call(self, **kwargs: Any) -> Any:
        """Call the metric function with exactly the given keyword arguments.

        Nothing is added, renamed or checked here: the function's own defaults
        apply and its result (or its exception) comes back unchanged. See the
        module docstring for how to build ``kwargs`` from the spec fields.
        """
        return self.load()(**kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Adding a metric: append a MetricSpec, fill direction/unit/cost_class/
# requires_extras (None where the reading is unclear) and run tests/test_registry.py
# and tests/test_l9_registry_*.py. They fail when a public compute_*/calculate_*
# function has no entry, when a declared kwarg is not a parameter of the function,
# and when a metadata value is outside the allowed sets.

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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        # A bundle of descriptors with different directions and units: no headline.
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="coulomb_energy_kJ",
        direction="lower_is_better",
        unit="kJ/mol",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        headline_key="ramachandran_favoured_pct",
        direction="higher_is_better",
        unit="percent",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        headline_key="omega_outlier_fraction",
        direction="lower_is_better",
        unit="fraction",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="sc",
        direction="higher_is_better",
        unit="dimensionless",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="void_volume_A3",
        direction="lower_is_better",
        unit="angstrom^3",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        headline_key="rmsd",
        direction="lower_is_better",
        unit="angstrom",
        cost_class="static",
        requires_extras=("structure",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="delta_sasa",
        direction="higher_is_better",
        unit="angstrom^2",
        cost_class="static",
        requires_extras=("biotite",),
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
        target_chain_arg="target_chain",
        # The composite MolProbity-style score of the per-model summary; it has no unit.
        headline_key="summary.molprobity_score",
        direction="lower_is_better",
        cost_class="structural",
        requires_extras=("biotite", "simulation", "structure"),
        requires_gpu=True,
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        # No unit is documented: a product of two distances and a confidence ratio.
        headline_key="evobind_adversarial_score",
        direction="lower_is_better",
        cost_class="model",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="hbond_energy",
        direction="lower_is_better",
        unit="kcal/mol",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="saltbridge_energy",
        direction="lower_is_better",
        unit="kcal/mol",
        cost_class="static",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="evobind_score",
        direction="lower_is_better",
        unit="angstrom",
        cost_class="model",
        requires_extras=("biotite",),
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
        headline_key="dockq",
        direction="higher_is_better",
        unit="dimensionless",
        cost_class="static",
        requires_extras=("dockq",),
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
        # Returns the per-frame array itself, in kJ/mol.
        direction="lower_is_better",
        unit="kJ/mol",
        cost_class="md",
        requires_extras=("simulation", "analysis"),
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
        headline_key="total",
        direction="lower_is_better",
        unit="kJ/mol",
        cost_class="md",
        requires_extras=("simulation", "analysis"),
    ),
    MetricSpec(
        name="rmsd",
        import_path="binding_metrics.metrics.rmsd:calculate_rmsd",
        description="Per-frame RMSD relative to reference frame; auto-selects protein heavy atoms",
        input_type="trajectory",
        chain_mode="none",  # atom_indices optional, auto-detected
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        # Returns the per-frame array itself, in nm (MDTraj units).
        direction="lower_is_better",
        unit="nm",
        cost_class="md",
        requires_extras=("analysis",),
    ),
    MetricSpec(
        name="rmsf",
        import_path="binding_metrics.metrics.rmsd:calculate_rmsf",
        description="Per-atom RMSF; auto-selects protein heavy atoms",
        input_type="trajectory",
        chain_mode="none",
        formats=("pdb", "cif"),
        path_arg="trajectory_path",
        # Returns the per-atom array itself, in nm (MDTraj units).
        direction="lower_is_better",
        unit="nm",
        cost_class="md",
        requires_extras=("analysis",),
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
        headline_key="ligand_rmsd",
        direction="lower_is_better",
        unit="nm",
        cost_class="md",
        requires_extras=("analysis",),
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
        target_chain_arg="target_chain",
        headline_key="drift_aligned_mean",
        direction="lower_is_better",
        unit="angstrom",
        cost_class="md",
        requires_extras=("analysis",),
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
        # Returns the per-frame array itself, in nm^2 (MDTraj units).
        direction="higher_is_better",
        unit="nm^2",
        cost_class="md",
        requires_extras=("analysis",),
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
        # Returns the per-frame count array itself. Whether more contacts is better
        # depends on the question, so no direction is declared.
        unit="count",
        cost_class="md",
        requires_extras=("analysis",),
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
        headline_key="buried",
        direction="higher_is_better",
        unit="nm^2",
        cost_class="md",
        requires_extras=("analysis",),
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
        # Residue lists: no direction, no unit.
        cost_class="md",
        requires_extras=("analysis",),
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
        # Returns a RelaxationResult, not a scored quantity: no direction, no unit.
        cost_class="md",
        requires_extras=("simulation", "structure"),
        requires_gpu=True,
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        # The default modes include "relaxed"; docs/report_thresholds.md scores E_int on
        # the minimised structure.
        headline_key="relaxed_interaction_energy",
        direction="lower_is_better",
        unit="kJ/mol",
        cost_class="md",
        requires_extras=("simulation",),
        requires_gpu=True,
    ),
    # --- OpenFold metrics ---------------------------------------------------
    MetricSpec(
        name="openfold",
        import_path="binding_metrics.metrics.openfold:compute_openfold_metrics",
        description="Parse OpenFold3 output: pLDDT, pAE, pTM, ipTM, GPDE, has_clash",
        input_type="openfold_json",
        chain_mode="none",
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        formats=(),
        path_arg="output_dir",
        # Bundle: pLDDT and ipTM are higher-is-better, pDE is lower-is-better.
        cost_class="model",
        requires_extras=("biotite",),
    ),
    MetricSpec(
        name="prediction",
        import_path="binding_metrics.metrics.prediction:compute_prediction_metrics",
        description=(
            "Confidence metrics of a structure prediction from a registered model: "
            "pLDDT, pTM, ipTM, PAE, PDE, binder pLDDT, interface PAE and PDE"
        ),
        input_type="prediction_dir",
        chain_mode="none",
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        formats=(),
        path_arg="prediction_dir",
        # Bundle: pLDDT and ipTM are higher-is-better, PDE and PAE lower-is-better; the
        # scales of pLDDT and ipTM differ between models.
        cost_class="model",
        requires_extras=("biotite",),
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
        binder_chain_arg="binder_chain",
        target_chain_arg="target_chain",
        headline_key="mean_interface_pae",
        direction="lower_is_better",
        unit="angstrom",
        cost_class="model",
        requires_extras=("biotite",),
    ),
]

# Fast lookup by name. Built from METRICS at import time, so it is a snapshot:
# append to METRICS only in this module, above this line.
METRICS_BY_NAME: dict[str, MetricSpec] = {m.name: m for m in METRICS}


def get_metric(name: str) -> MetricSpec:
    """Return the ``MetricSpec`` registered under *name*.

    Raises:
        KeyError: No metric has that name; the message lists the available
            names.
    """
    try:
        return METRICS_BY_NAME[name]
    except KeyError:
        available = ", ".join(METRICS_BY_NAME)
        raise KeyError(f"Unknown metric {name!r}. Available: {available}") from None


def metrics_by_input_type(input_type: InputType) -> list[MetricSpec]:
    """Return the specs of one input type, in registry order.

    The list is empty for an input type no metric uses; nothing is raised.
    """
    return [m for m in METRICS if m.input_type == input_type]
