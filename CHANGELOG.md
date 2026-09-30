# Changelog

All notable changes to BindingMetrics are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Issue numbers refer to the GitHub issue
tracker of the project. Example values come from the structures in `data/`: 1YCR (p53 peptide with
MDM2), 3P8F (SFTI-1 with matriptase), 1CWA (cyclosporin A with cyclophilin A), 1XY4 (somatostatin
analogue) and 3V3B (stapled peptide).

## [Unreleased]

### Results that change

Values from earlier versions differ in the cases below. A change reads "before -> after".

- **RMSD superposition (#13).** The Kabsch rotation was applied in the wrong direction, so any pair
  that was not already superposed came out too high.
  - `compute_structure_rmsd`: 1YCR rotated by 30 degrees and translated, `rmsd` 11.23 -> 0.00 A and
    `rmsd_design` 6.80 -> 0.00 A. It also reads only the first model of a multi-model file.
  - Relaxation: `rmsd_md_final`, `receptor_rmsd_md_final` and `receptor_drift_mean` use the same routine.
- **Waters, ions and ligands are dropped by default (#14).** `hetero="ignore"` keeps amino-acid atoms,
  AMBER protonation variants and ACE/NME/NH2 caps. `hetero="keep"` selects every atom of the chain as before;
  water and ion atoms have no SASA and count as zero area, where they used to make the sum NaN.
  - ΔSASA: 1CWA `delta_sasa` NaN -> 985.4 A^2, 3P8F NaN -> 1508.3 A^2. 1YCR has no waters and is unchanged.
  - H-bonds: 1CWA 6 -> 5 (`hbond_energy` -10.56 -> -8.22 kcal/mol), 3P8F 11 -> 9 (-20.42 -> -16.67).
  - Shape complementarity: `sc` 1CWA 0.722 -> 0.750, 3P8F 0.732 -> 0.738.
  - Void volume: interface atoms 1CWA 153 -> 123, 3P8F 227 -> 207.
- **Solvation parameters of `delta_g_int` (#18).** The old table gave every N +0.063 and every O
  +0.024 kcal/mol/A^2. Five atom types after Eisenberg and McLachlan (1986) replace it: C -0.016,
  S -0.021, neutral N/O +0.006, O(-) +0.024, N(+) +0.050 (values as tabulated by Krissinel and Henrick, 2007).
  - `delta_g_int`: 1YCR -0.77 -> -11.05 kcal/mol, 1CWA +1.26 -> -6.11 (water-free input), 3P8F NaN -> -4.97.
- **Buried area and `delta_g_int` on heavy atoms (#39).** With explicit hydrogens the surface of a heavy atom
  was partly taken by its hydrogens, which carry no solvation parameter, and polar + apolar did not add up to
  `delta_sasa`. Hydrogens are dropped before the areas (`hydrogens="keep"` restores the old atom set).
  - 1YCR relaxed by `binding-metrics-run` (851 H of 1670 atoms): `delta_sasa` 1552.4 -> 1488.4 A^2, polar + apolar
    651.4 -> 1488.4 A^2, `delta_g_int` -0.95 -> -10.50 kcal/mol. The bundled files without hydrogens are unchanged.
- **Void volume honours `probe_radius` (#17).** A void must be closed to the probe in the complex and
  open for each chain alone; before, the probe was ignored and a cavity walled by one chain counted.
  - `void_volume_A3` at the default probe: 1YCR 0.25 -> 55.875, 3P8F 0.38 -> 31.875, 1CWA 0.50 -> 16.25 A^3.
- **Clashscore of `compute_receptor_quality` (#16).** Covalent links (disulfides, ring closures, lactams,
  staples) and N...O hydrogen-bond pairs are no longer counted; `exclude_bonded=False` restores the count.
  - Clashscore: 3P8F chain I 66.67 -> 0.00 (`molprobity_score` 3.863 -> 2.068), 3P8F chain A 18.24 -> 0.54.
  - Clashscore: 1XY4 52.17 -> 0.00, 1YCR chain A 11.35 -> 7.09, 1CWA chain A 2.37 -> 0.00, 3V3B chain C 31.25 -> 0.00.
- **GAFF bond orders of non-canonical residues (#38).** Bond orders were perceived on a graph without
  hydrogens and came out single. They now come from the wwPDB Chemical Component Dictionary in biotite.
  - 1CWA: MeBmt has 17 hydrogens, not 19, and its CE=CZ bond relaxes to 1.340 A, not 1.544 A (crystal 1.336 A).
  - 1CWA `raw_interaction_energy` -268.4 -> -299.7 kJ/mol, `relaxed_interaction_energy` -280.9 -> -299.7
    kJ/mol (`--md-duration-ps 0`, seed 1, CUDA; the values after the GAFF2 entries below).
  - The backbone C=O of 0EH and MK8 (3V3B) was read as C-OH.
  - IAM (1XY4): PDBFixer lists the IAM-THR peptide bond twice, so the capped molecule got two caps on the backbone C
    and IAM fell back to single bonds. A bond that the topology lists twice is capped once, and IAM takes the
    dictionary bond orders: `ncaa_bond_order_source` `single_bonds` -> `ccd`, hydrogens 24 -> 18, an aromatic ring
    instead of cyclohexane.
- **GAFF2 charges repeat from run to run.** `sqm` picked its diagonalisation routine by timing seven of
  them at start-up, and one of them ended the AM1 minimisation of MeBmt in another geometry. The BMT template
  charges changed by up to 9e-4 e between builds (0.013 e in the AM1-BCC output), prep, relaxation and the
  energy step each built their own template, and the relaxed 1CWA complex changed between runs. `sqm` now runs
  with its own diagonaliser on one thread, and the conformer that the charges start from is seeded
  (`random_seed` of `parameterize_ncaa_residues`, default 1, set by `--random-seed`). The three steps build
  byte-identical templates.
  - 1CWA, four runs of one command (`--md-duration-ps 0`, seed 1, CUDA) gave three results: minimised complex
    energy -21555.1, -21562.7 or -21563.2 kJ/mol; `relaxed_interaction_energy` -297.1, -298.6 or -298.9 kJ/mol;
    buried area 1006.4, 1010.3 or 1007.9 A^2. After the change every run gives the same files.
- **AM1-BCC charges of GAFF2 residues use the stereochemistry of the structure.** The residue graph had none,
  so the conformer that `sqm` minimises was a random stereoisomer: MeBmt (BMT, 1CWA) was embedded with CA R,
  CB S, CG2 R and a Z double bond, against S, R, R and E in the structure. Stereocentres and E/Z bonds are read
  from the geometry now. AM1-BCC charges depend on the conformer, so a residue's charges also change with
  `random_seed`: MeBmt differs by up to 0.065 e between seeds 1 and 2.
  - 1CWA: BMT template charges change by up to 0.047 e and ABA by up to 0.017 e. Minimised complex energy
    -21562.7 -> -21553.4 kJ/mol, `raw_interaction_energy` -280.2 -> -280.4, `relaxed_interaction_energy`
    -298.6 -> -298.5, buried area 1010.3 -> 996.8 A^2, void volume 9.9 -> 4.9 A^3, `sc` 0.685 -> 0.684.
- **GAFF2 residues keep their relaxed hydrogens.** `parameterize_ncaa_residues` dropped the hydrogens of every
  GAFF2 residue and injected the ones RDKit places for the heavy atoms, in prep, relaxation and the energy step.
  The energy step therefore evaluated a relaxed structure with unrelaxed hydrogens on such a residue: the complex
  energy of 1CWA was 124 kJ/mol above the relaxation minimum, against 0.02 kJ/mol for complexes without a GAFF2
  residue. A residue that holds exactly its template's hydrogens (same names on the same parent atoms) now keeps
  them and their positions; a residue without hydrogens, or with others, is built as before.
  - 1CWA (`--md-duration-ps 0`, seed 1, CUDA), before -> after: minimised complex energy -21553.4 -> -21552.6
    kJ/mol, energy-step `raw_e_complex` -21429.4 -> -21552.6 (gap to the minimum 124.0 -> 0.02),
    `raw_interaction_energy` -280.4 -> -299.7, `relaxed_interaction_energy` -298.5 -> -299.7, buried area
    996.8 -> 996.3 A^2, void volume 4.9 -> 5.6 A^3, `sc` 0.684 -> 0.685. Prep is unchanged.
- **Charges of `small_molecules` given as a list repeat from run to run.** The listed molecules went to
  `GAFFTemplateGenerator`, whose own AM1-BCC call had the `sqm` timing problem above. They now get the same
  seeded, single-diagonaliser charges before the generator is built; charges set on a molecule are kept.
- **D-amino-acid and N-methyl names survive prep and relax (#15).** The relaxed CIF of 1CWA had
  `ALA ... NMG` where the input has `DAL ... SAR`; it now keeps the input names.
- **Ligand RMSD (#37).** `calculate_ligand_rmsd` fitted the ligand a second time and returned 0 for a
  rigid-body shift. 1YCR chain B moved 5 A against chain A: `ligand_rmsd` [0, 0] -> [0, 0.5] nm.
- **Ring-closing phi, psi and omega (#15).** For a head-to-tail peptide, `compute_ramachandran` and
  `compute_omega_planarity` score the closing amide bond (C(last) to N(first) within 2 A).
  - 1CWA chain C: residues 9 -> 11, `n_d_residues` 0 -> 1, favoured/allowed/outlier 66.7/22.2/11.1 ->
    72.7/18.2/9.1 %, omega bonds 10 -> 11.
  - 3P8F chain I: residues 12 -> 14, 91.7/0.0/8.3 -> 85.7/7.1/7.1 %, omega outliers 1 -> 2.
- **Coulomb energy and salt bridges for D, HIP and phospho residues (#29).** The charge tables knew only
  L-residue names. D codes now map to their L counterpart, HIP carries +1 and SEP/TPO/PTR carry -2.
  - An all-D peptide (DLY) against ASP at 4 A: `coulomb_energy_kJ` 0.0 -> -101.3, equal to its L twin.
  - 1YCR (-91.59 kJ/mol) and 3P8F (-299.59 kJ/mol) do not change.
- **Chain detection (#29).** `detect_interface_chains` counts amino-acid atoms instead of matching 28
  residue names. A 20-residue ALA receptor with a 5-residue DAL peptide: `('A', None)` -> `('B', 'A')`.
- **Seeding (#23).** `solvate` and `prepare_system` used unseeded randomness for ions and hydrogens and
  now use seed 1 (`random_seed=None` restores fresh randomness).
  - `compute_receptor_quality`, 1YCR chain A energy: 4828.7, 4834.2 and 2337.5 kJ/mol in three runs
    -> 4371.8 kJ/mol every time.
- **`MED` for D-methionine (#15).** `D_AA_MAP` mapped `DME` to MET, but `DME` is decamethonium in the
  Chemical Component Dictionary. `MED` is now the D-methionine code and `DME` is no longer a D residue.
- **`HIN` in EvoBind residue matching.** `HIN` pairs with `HIS` as a histidine variant, so
  `*_resname_mismatch_fraction` does not count it as a mismatch.
- **OpenFold3 query sequences (#75, #76).** D-amino acids and modified residues became lower-case letters
  in the query sequence (`DAL` -> `a`, `SEP` -> `s`), which OpenFold3 turns into unknown residues, and
  CYX, HID, HIE, HIP and names gemmi does not know were dropped from it, so the chain came out shorter
  without a message. D-amino acids, N-methylated and other modified residues now keep their parent
  letter in upper case and go to `non_canonical_residues` with their CCD code (the toolkit's NMG and NMA
  as SAR and MAA); protonation variants take the parent letter.
  - 1CWA chain C: `allvtaglVlA` -> `ALLVTAGLVLA`, with nine residues (`DAL`, `MLE`, `MVA`, `BMT`, `ABA`,
    `SAR`) in `non_canonical_residues`. 1QJB chain Q: `ARSHsYPA` -> `ARSHSYPA` with `{5: SEP}`. 3V3B chain C:
    `TFNLWRLLl` (nine letters, `0EH` left out) -> `TFXNLWRLLL` with `{3: 0EH, 10: MK8}`.
  - **Deliberate behaviour change:** a residue with a backbone that OpenFold3 cannot take (not a standard,
    variant, D- or peptide-linking CCD residue) raises `UnmappableResidueError` before any file is
    written or process started, where it was left out or turned into a wrong letter. The message names
    the chain and the residues. `on_unmappable_residue="x"` (`--on-unmappable-residue x`) restores a
    query with `X` at their positions and a logged warning.
- **Interface PDE and PAE of OpenFold3 output (#33).** A ligand or modified residue tokenised per atom
  shifted the residue-based slice onto the wrong block. The values are NaN with a `reason` now.
- **Batch status and order (#19, #32).** A sample whose steps failed had `batch_status` `ok`; it is
  `partial` now, with `batch_failed_steps` and `batch_failed_reasons`.
  - The exit code is non-zero when no sample is `ok`. With `--workers > 1` the CSV rows follow the input order.
- **`--md-duration-ps` from 1 to 9 (#27).** A duration below the 10 ps save interval failed with an
  IndexError after the MD had run. The run saves one frame at the end and completes.
- **Console streams (#22).** Library warnings that reached stderr through the logging last-resort handler
  appear on stdout, for example in `binding-metrics-relax` and `-report`; the `core.gaff_ncaa` messages
  appear in relax. `binding-metrics-prep` and `-solvate` keep warnings on stderr.
- **Warnings of `compute_openfold_metrics`** name the line that called it.
- **Structural QC bond lengths (#44).** The `bond_lengths` check took its bond list from input distances, so
  atoms that PDBFixer had rebuilt inside a clash counted as bonds, and a relaxation that resolved the clash
  failed the check. Bonds now come from the topology and are measured in the relaxed structure only; the
  limits are unchanged, and a bond already over 2.5 A in the input is named in `detail`.
  - `bond_lengths` on the example runs: 1QJB fail (3 of 1932 bonds) -> pass; 4KRL_relaxed fail (29 of 2613)
    -> pass; `qc_passed` False -> True for both.
  - The longest real bonds are 1.82 A (Met C-S) and 2.05 A (disulfide); the old maxima (2.39 A for 1YCR,
    2.25 A for 3P8F) were non-bonded pairs.
- **The relaxation and the interaction energy use the peptide-receptor pair only (#42).** Every other
  protein chain is removed after `strip_heterogens`, with a warning that names it. Before, it stayed in
  E_complex but not in the isolated components, and a chain that had lost its caps could not be built.
  - 1QJB (peptide Q, receptor A; chains B and S removed): `relaxed_interaction_energy` -41065.6 -> -546.0
    kJ/mol and `potential_energy_minimized` -80432.4 -> -39984.3 kJ/mol (`--md-duration-ps 0`). The .cif and
    the .pdb file agree to all digits.
  - 5WGD (peptide E, receptor A; chains B and F removed): the relaxation failed with "No template found for
    residue 473 (SER)"; it now completes, `relaxed_interaction_energy` -235.7 kJ/mol.
  - 1YCR with a copy of the p53 chain 6 nm away as a third chain: `raw_interaction_energy` +139.6 -> -16.0
    kJ/mol, the value of the two chains alone.
- **Chain IDs of the input survive prep (#41).** `prep_structure` names the chains as the caller did,
  and `save_cif` without a source CIF writes the topology's own chain IDs and residue numbers. A PDB input
  used to come out as A, B, C, D and `--peptide-chain Q` failed after prep.
  - 1QJB.pdb prepped: chains A B C D -> A B Q S. The relaxed CIF of a PDB input carries the input's chain
    IDs and residue numbers, no longer the letters A, B, ... and 1, 2, ....
  - `peptide_chain_label` of `detect_chains_from_file` is the chain ID OpenMM gives the file: the label ID
    only when the file has more label IDs than author IDs. A protein-only mmCIF whose label and author
    letters are swapped (4KRL without waters and ligands) returned the other chain.
- **Raw mmCIF with different label and author numbering (#40).** OpenMM matches the `_struct_conn` rows
  by label numbering and the atoms by author numbering, so a covalent link was dropped, and a residue
  such as phosphoserine loaded without any bond. 1QJB.cif: HIS6 C to SEP7 N absent -> present, and SEP
  has 0 -> 9 internal bonds, as from the PDB file. The relaxation of the four-chain 1QJB failed with "bonds
  are different" and now runs.
- **Peptide bonds next to a non-standard residue (#43).** `patch_cyclic_topology` rebuilds the bonds of a
  residue that OpenMM loaded without any, and the peptide bonds beside it, in every protein chain, the
  receptor included. 6SBA as a PDB file without CONECT records for P1L (S-palmitoyl-cysteine): "No template
  found for residue 144 (LEU)" -> minimised -32300.7 kJ/mol, as from the mmCIF. P1L goes through the
  GAFF2 route, whose charge calculation runs at every system build and takes minutes.
- **`CYM` is a standard residue (#34).** A deprotonated cysteine was listed under `kept_nonstandard`, and the
  GAFF2 route and the heterogen scan of the relaxation treated it as a new residue. 1YCR with one CYS renamed
  CYM: `kept_nonstandard` `['CYM (chain A)']` -> `[]`.
- **`detect_chains` finds all-D chains (#41).** `io.structures.detect_chains`, which
  `compute_interaction_energy` uses when no chain is named, counts every amino-acid residue. A 20-residue ALA
  chain with a 5-residue DAL chain: `('A', None)` -> `('B', 'A')`.

### Added

- `binding_metrics.predictors`, the readers of structure-prediction output, starts with `PredictionRecord`, `PredictionFiles`, `TokenLayout` and `SampleRef`: one neutral form for a prediction sample (pLDDT per atom on 0-100, PAE and PDE in angstrom with the row as the alignment frame, NaN for a value the model lacks, and a `chain_map` that renames chains). `PredictionRecord.validate()` rejects a wrong scale or shape.
- `PredictionParser`, the base class of a predictor adapter (`find_files`, `parse`, `load`, `list_samples`), and its registry in `binding_metrics.predictors`: `ParserSpec`, `PARSERS`, `get_parser` and `register_parser`. An adapter reads a model's output files and never runs the model.
- The `of3` adapter, `get_parser("of3")`, reads an OpenFold3 output directory into a `PredictionRecord`: structure files `.cif`, `.cif.gz` and `.pdb`, seed directories in numeric order of the seed value, `.npz` confidences read without pickle, and the OpenFold3-only `bespoke_iptm` in `record.extras` (#78, #79, #80).
- `binding_metrics.metrics.mlff_energy`, a reserved interface for an interaction energy from a machine-learned force field (Ryczko et al., ChemRxiv 10.26434/chemrxiv.15008810): `MLFFBackend`, `PocketSpec`, `register_backend`, `get_backend`, `available_backends` and `compute_mlff_interaction_energy`, which validates its arguments and raises `NotImplementedError`. It is not a registered metric.
- `random_seed` on `parameterize_ncaa_residues` (default 1, keyword-only): the seed of the conformer that the
  AM1-BCC charges of GAFF2 residues start from. `None` draws a new one.
- `--config PATH` for `binding-metrics-run`, `-batch` and `-relax`: a TOML file supplies option
  defaults and command-line flags override it (#32).
- `run_batch` runs a batch in-process with an `on_result` callback, and `binding-metrics-batch` is built
  on it. `Relaxer` is the base class that `run_pipeline(relaxer=...)` accepts. All three are exported (#32).
- `--binder-chain` and `--target-chain` on the metric CLIs, the keywords `binder_chain` and
  `target_chain` on the metric functions, and the registry fields `binder_chain_arg` and `target_chain_arg` (#31).
- Registry: 10 new specs (18 to 28), the input types `atom_array` and `predicted_structure`, and the fields
  `headline_key`, `direction`, `unit`, `cost_class`, `requires_extras` and `requires_gpu` (#30).
- `results["provenance"]` (package version, git sha, Python, OS, OpenMM version, platform, seed),
  `provenance_*` columns in the batch CSV, and `collect_provenance` (#24).
- A release workflow that attaches the sdist and wheel to a GitHub Release for each `v*` tag (#24).
- `--random-seed` on `binding-metrics-batch`, `-prep`, `-solvate`, `-energy` and `-receptor-quality`,
  and `--ph` on `-energy`. The `random_seed` keyword of `solvate`, `prepare_system` and
  `compute_receptor_quality` (#23).
- `results["prep"]` reports `removed_heterogens`, `n_removed_waters`, `kept_nonstandard`,
  `n_missing_atoms_rebuilt` and `n_missing_residue_gaps` (#28).
- `results["prep"]["chain_breaks"]` lists the consecutive residues of a chain whose C and N atoms are more
  than 2.0 A apart, and prep and relaxation log a warning. OpenMM bonds the two residues by name and the
  relaxation closes the gap; nothing else changes (#45).
  - 1QJB chain A, residues 68 and 73: 7.45 A. In 5WGD the relaxation inverted the C-alpha of residue A:460,
    next to a gap of 11.4 A, and the structural QC `chirality` check flags it.
- `results["relax"]["dropped_protein_chains"]` and `RelaxationResult.dropped_protein_chains`: the protein
  chains removed because they are neither the peptide nor the receptor (#42).
- `find_chain_breaks`, `drop_other_protein_chains` and `reconstruct_nonstandard_residue_bonds`, and the
  keyword-only `residues` argument of `reconstruct_intraresidue_bonds`.
- `ncaa_bond_order_source`, `{residue: "ccd" or "single_bonds"}`, in `results["prep"]` and
  `results["relax"]` (#38).
- `parameterize_ncaa_residues` warns for each residue with an acid, phosphate, sulfate, primary amine or
  guanidine group, which the GAFF2 route builds neutral. Its third return value is a `NcaaTemplateList`, a
  list with `net_charge_by_residue`, `neutral_ionizable_groups` and `bond_order_source_by_residue` (#28, #38).
- Structural QC of the relaxed structure: `qc_passed`, `qc_failed_checks` and `qc_checks` in
  `results["relax"]`, and a warning line when a check fails (#26).
- `platform`, `precision` and `platform_fallback_reason` in `results["relax"]` (#27).
- A `reason` string next to each NaN, None or zero that means "could not compute" (#25), and
  `results["nonfinite_fields"]`, the paths of NaN and infinite values in the JSON (#25).
- `hetero` keyword and `--hetero {ignore,keep}` flag for the interface, static SASA, H-bond,
  salt-bridge, shape-complementarity and void-volume metrics (#14).
- `hydrogens` keyword of `compute_interface_metrics` and `compute_delta_sasa_static`, and
  `--hydrogens {ignore,keep}` on `binding-metrics-interface` (#39).
- Result keys `omega_cis_count`, `cyclic_closure_detected`, `cyclic_closure_evaluated`,
  `n_ionisable_residues_seen` and `n_residues_unrecognised`.
- EvoBind keys `interface_fallback_used`, `n_superposition_atoms`, `receptor_pairing`,
  `binder_pairing` and two `*_resname_mismatch_fraction` (#33).
- `--openfold-seeds` for `binding-metrics-run` and `-batch`, a `seeds` argument for the OpenFold3
  query JSON, and the `seed_index` alias of `seed` in `compute_openfold_metrics` (#33).
- `on_empty="raise"` for `calculate_rmsd` and `calculate_contacts` (#25), and `exclude_bonded` for
  `compute_receptor_quality` (#16).
- The `static` extra, which lists what the single-structure metrics import. The `all` extra gains
  scipy and DockQ (#21, #36).
- `binding_metrics.utils.configure_logging` for entry points and scripts (#22).
- `environment.lock.yml` (exact versions of the development environment), a pre-commit configuration
  with ruff pinned to the CI version, and monthly Dependabot updates of the GitHub Actions (#36).
- A CI job that runs the static metrics with OpenMM blocked, and a `ruff` configuration that also
  enforces flake8-bugbear and the blind-except rule (`B905` stays off) (#21, #36).

### Changed

- The default OpenFold3 model presets of `binding-metrics-openfold run`, `refold` and `score` are
  `predict low_mem`. OpenFold3 0.4.1 removed `pae_enabled` (the PAE head is on by default; pTM, ipTM and PAE
  are always written) and 0.5.0 only logs a warning for it. An explicit `pae_enabled` is left out of the runner
  YAML with a `DeprecationWarning`, and the run continues (#49).
- `binding-metrics-check-env` reports the installed `openfold3` version and whether the default checkpoint
  (`of3-ob-2025-06-30-174k.pt`, found through `$OPENFOLD_CACHE` or `~/.openfold3` and its `ckpt_root` file)
  is on disk. openfold3 0.5.0 or later without that file fails the check, since every run would stop there;
  Preview2 weights alone do not count. A version below 0.5.0, or one that cannot be read, is a warning
  (#74). The check was a bare `import openfold3`.
- The `:full` image and its README section describe the default Triton path of openfold3 0.5: the DeepSpeed
  `evoformer_attn` JIT cache mount, the CUTLASS and ninja layer and `CUTLASS_PATH` are gone (DeepSpeed is
  an opt-in extra that the environment file does not install), a mount for `TRITON_CACHE_DIR` takes their
  place, and the README names `openbind-2025-06-30-174k`, not `openfold3-p2-155k`, as the default
  checkpoint (#72, #81). The image was not built.
- `environment_openfold3.yml` pins `openfold3>=0.5.0,<0.6` (it was unpinned) and names OpenBind-0, the default
  weights of the 0.5 series, which the 0.4.x releases and Preview2 weights cannot use (#82). The pin follows
  the release notes; the resolved torch and CUDA wheels were not tested.
- `import binding_metrics` no longer imports OpenMM. Names load on first access, and a name whose
  optional dependency is missing raises an error that names the extra to install (#21).
- Library code logs through `logging`. The command-line tools call `configure_logging`, which sends
  records up to WARNING to stdout and ERROR and above to stderr (#22).
- `__version__` is read from the package metadata and is `0.0.0+unknown` in a tree that is not installed (#24).
- `compute_openfold_metrics` documents `seed` as the 1-based index of a seed directory. The OpenFold3
  reference is OpenFold3-preview (The OpenFold3 Team, 2025) together with AlphaFold 3.
- `METRICS.md` and `docs/metrics.md` are one document, `docs/metrics.md`. It now covers DockQ,
  interface PAE, provenance and the registry metadata (#36).
- `openfold.py` is split into `_openfold_run.py` and `_openfold_cli.py`, and the residue and water name
  sets live in `core/residues.py`. Public names and import paths are unchanged (#34).
- The `report` and `all` extras no longer list matplotlib, which nothing imports. The `openfold` extra
  installs `openfold3` (it named a distribution `openfold`), and `openfold3` is an alias (#36).
- Broad `except Exception` blocks catch the errors they expect, or say in a comment why they stay
  broad. Those that passed silently log a warning or debug record. `compute_hbonds`, the platform probe
  of `MDSimulation` and the scorecard of the report no longer hide an unexpected error (#25).

### Fixed

- `core/gaff_ncaa.py` writes force-field files and reads antechamber output as UTF-8 whatever the locale.
- Batch: `--per-sample-log` was never read, and workers overwrote a shared `--log-file` (#19). An
  exception outside the pipeline's own handling is an error row in the sequential case too.
- A chain ID that is not in the structure raises `ChainNotFoundError` in `run_pipeline`, and
  `compute_ramachandran` and `compute_omega_planarity` raise a `ValueError` that lists the chains (#20).
- Relaxation: a duration shorter than the save interval raises `ValueError` when the config is built,
  and two failed `addHydrogens` attempts raise `RuntimeError` (#27).
- A failed minimisation no longer leaves the backbone restraint in the system, where its energy was
  added to a later `after_md` evaluation (#27).
- `compute_interaction_energy` lists each failed mode in `error_message` as `"<mode>: <reason>"`; it stayed
  None when `success` was False (#25). Without OpenMM it names the `simulation` extra (#21).
- The help of `binding-metrics-batch --metrics` lists `dockq`, which the option already accepted (#30).
- `--small-molecules` accepts `auto` or `none`. A typo used to register each character as a SMILES (#27).
- Without gemmi, `save_cif` logs a warning and `extract_model_to_tempfile` raises `ImportError` for a CIF;
  both used to degrade without a message (#28).
- The `:full` Docker image entrypoint fetches the OpenBind-0 checkpoint (`of3-ob-2025-06-30-174k.pt`, the
  default of openfold3 0.5.0) with `setup_openfold --non-interactive`. It fed `setup_openfold` a canned
  answer sequence that raises `EOFError` from openfold3 0.4.2 on when pytest is installed (#70), counted any
  `*.pt` file as the weights, so a volume with Preview2 weights skipped the download and the first run
  stopped (#71), and named `openfold3-p2-155k` as the default (#72). Statically checked and run under bash
  with a stub `conda`; the image was not built.
- OpenFold3 removes the parent of its template `structure_directory` when a run with the MSA server and
  templates ends (0.3.1 to 0.5.0), and the toolkit's is `<output>/query`. The runner YAML now sets
  `msa_computation_settings.cleanup_msa_dir: false` when it sets `structure_directory`, so the query JSON,
  the A3M files and the template CIFs stay next to the predictions (#69). Read from the OpenFold3 source, not run.
- Batched OpenFold3 queries: sample IDs that differ only by `_` versus `-` (or by case) shared one template
  file, so the second sample's CIF replaced the first's and that query was predicted from the wrong
  template. `a_b` and `a-b` both became `templates/a-brec.cif`; each now gets a short hash of its own ID
  (`a-b-648fa9b3rec` and `a-b-d44362d6rec`). IDs that do not collide keep their names (#86).
- OpenFold3 and EvoBind: token offsets are checked against the matrix size, residue names are checked
  when two structures are paired, and the batch OpenFold JSON holds numbers, not strings (#25, #33).
- The report writes numpy scalars as numbers and shows NaN as N/A in the scorecard, where NaN used to
  fall through to red (#25).
- `python -m binding_metrics.<module>` logs through the package handlers, and `prep` and `solvate` keep
  their JSON summary alone on stdout (#22).
- `PeptideBindingProtocol.analyze` raises on a trajectory without frames, `calculate_rmsd` and
  `calculate_contacts` warn on an empty selection, and `compute_receptor_quality` says why a term is NaN (#25).
- `.dockerignore` mirrors the private paths of `.gitignore` (#36).
- The summary read `cyclic_bonds` where the results carry `peptide_cyclic_bonds`, so its "Cyclic topology"
  section never appeared; it reads both keys (#45).
- `strip_heterogens` keeps SEP, TPO and PTR in the chains that are not selected; it deleted the
  phosphoserine of an unselected chain as a distant heterogen and cut the chain in two (#40).
- `save_cif` writes the restored residue number into `label_seq_id` and the `_struct_conn` rows, so a
  prepped file reloads with the links of its non-standard residues (#40).
- `binding-metrics-run --skip-prep --skip-relax` on 3P8F: the metrics that read the raw file with biotite
  got the label ID of the peptide (B) and failed with "chain 'B' not found"; they get the author ID (I).
  The cyclic-bond hints of the pipeline looked the chain up in the raw topology by the ID of the prepped
  file (#41).
- `prep_structure` on the stapled peptide 3V3B failed with "Chain 'C' not found in topology", because the
  cyclic-bond hints named the input chain and PDBFixer had renamed it (#41).
