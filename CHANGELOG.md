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
  AMBER protonation variants and ACE/NME/NH2 caps. `hetero="keep"` restores the old inputs.
  - ΔSASA: 1CWA `delta_sasa` NaN -> 985.4 A^2, 3P8F NaN -> 1508.3 A^2. 1YCR has no waters and is unchanged.
  - H-bonds: 1CWA 6 -> 5 (`hbond_energy` -10.56 -> -8.22 kcal/mol), 3P8F 11 -> 9 (-20.42 -> -16.67).
  - Shape complementarity: `sc` 1CWA 0.722 -> 0.750, 3P8F 0.732 -> 0.738.
  - Void volume: interface atoms 1CWA 153 -> 123, 3P8F 227 -> 207.
- **Solvation parameters of `delta_g_int` (#18).** The old table gave every N +0.063 and every O
  +0.024 kcal/mol/A^2. Five atom types after Eisenberg and McLachlan (1986) replace it: C -0.016,
  S -0.021, neutral N/O +0.006, O(-) +0.024, N(+) +0.050 (values as tabulated by Krissinel and Henrick, 2007).
  - `delta_g_int`: 1YCR -0.77 -> -11.05 kcal/mol, 1CWA +1.26 -> -6.11 (water-free input), 3P8F NaN -> -4.97.
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
  - 1CWA `raw_interaction_energy` -268.4 -> -280.2 kJ/mol, `relaxed_interaction_energy` -280.9 -> -297.4
    kJ/mol (`--md-duration-ps 0`).
  - The ring of IAM (1XY4) was cyclohexane and is aromatic now. The C=O of 0EH and MK8 (3V3B) was read as C-OH.
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

### Added

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
- `ncaa_bond_order_source`, `{residue: "ccd" or "single_bonds"}`, in `results["prep"]` and
  `results["relax"]` (#38).
- Structural QC of the relaxed structure: `qc_passed`, `qc_failed_checks` and `qc_checks` in
  `results["relax"]`, and a warning line when a check fails (#26).
- `platform`, `precision` and `platform_fallback_reason` in `results["relax"]` (#27).
- A `reason` string next to each NaN, None or zero that means "could not compute" (#25), and
  `results["nonfinite_fields"]`, the paths of NaN and infinite values in the JSON (#25).
- `hetero` keyword and `--hetero {ignore,keep}` flag for the interface, static SASA, H-bond,
  salt-bridge, shape-complementarity and void-volume metrics (#14).
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

### Changed

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

### Fixed

- Batch: `--per-sample-log` was never read, and workers overwrote a shared `--log-file` (#19). An
  exception outside the pipeline's own handling is an error row in the sequential case too.
- A chain ID that is not in the structure raises `ChainNotFoundError` in `run_pipeline`, and
  `compute_ramachandran` and `compute_omega_planarity` raise a `ValueError` that lists the chains (#20).
- Relaxation: a duration shorter than the save interval raises `ValueError` when the config is built,
  and two failed `addHydrogens` attempts raise `RuntimeError` (#27).
- A failed minimisation no longer leaves the backbone restraint in the system, where its energy was
  added to a later `after_md` evaluation (#27).
- `--small-molecules` accepts `auto` or `none`. A typo used to register each character as a SMILES (#27).
- Without gemmi, `save_cif` logs a warning and `extract_model_to_tempfile` raises `ImportError` for a CIF;
  both used to degrade without a message (#28).
- OpenFold3 and EvoBind: token offsets are checked against the matrix size, residue names are checked
  when two structures are paired, and the batch OpenFold JSON holds numbers, not strings (#25, #33).
- The report writes numpy scalars as numbers and shows NaN as N/A in the scorecard, where NaN used to
  fall through to red (#25).
- `python -m binding_metrics.<module>` logs through the package handlers, and `prep` and `solvate` keep
  their JSON summary alone on stdout (#22).
- `PeptideBindingProtocol.analyze` raises on a trajectory without frames, `calculate_rmsd` and
  `calculate_contacts` warn on an empty selection, and `compute_receptor_quality` says why a term is NaN (#25).
- `.dockerignore` mirrors the private paths of `.gitignore` (#36).
