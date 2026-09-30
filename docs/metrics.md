# metrics reference

Every metric in BindingMetrics: function signature, result keys, units and algorithm notes. Installation and the command line are described in [`README.md`](../README.md). This file is the only metrics reference; the root-level `METRICS.md` points here.

Run every metric on one structure:

```bash
binding-metrics-run --input complex.cif --output-dir results/
```

---

## table of contents

1. [conventions](#1-conventions)
2. [interface geometry](#2-interface-geometry)
3. [hydrogen bonds & salt bridges](#3-hydrogen-bonds--salt-bridges)
4. [electrostatics](#4-electrostatics)
5. [force-field interaction energy](#5-force-field-interaction-energy)
6. [ramachandran & omega planarity](#6-ramachandran--omega-planarity)
7. [shape complementarity](#7-shape-complementarity)
8. [buried void volume](#8-buried-void-volume)
9. [structure comparison (RMSD)](#9-structure-comparison-rmsd)
10. [reference-based accuracy (DockQ)](#10-reference-based-accuracy-dockq)
11. [MD trajectory metrics](#11-md-trajectory-metrics)
12. [OpenFold3 confidence scores](#12-openfold3-confidence-scores)
13. [EvoBind scoring](#13-evobind-scoring)
14. [receptor quality](#14-receptor-quality)
15. [pipeline results and provenance](#15-pipeline-results-and-provenance)
16. [metric registry](#16-metric-registry)
17. [scorecard thresholds](#17-scorecard-thresholds)
18. [score versus feature](#18-score-versus-feature)
19. [I/O utilities](#19-io-utilities)
20. [unit summary](#20-unit-summary)
21. [references](#21-references)

---

## 1. conventions

**binder and target.** Every structure metric works on two single chains. The binder is the peptide or design chain, the target is the receptor. Functions name the binder `peptide_chain`, `design_chain` or `chain` and the target `receptor_chain`; each also takes the keyword-only aliases `binder_chain` and `target_chain`. Giving both spellings with different chain IDs raises `ValueError`. The command-line tools accept `--binder-chain` and `--target-chain` next to the older flags (`--peptide-chain`, `--design-chain`, `--chain`, `--receptor-chain`). `binding-metrics-geometry` reads `--binder-chain` as `--chain` for Ramachandran and omega, and as `--peptide-chain` for shape complementarity and void volume.

**chain auto-detection.** Where a chain is optional, the metric functions take the smallest protein chain as binder and the largest as target. A protein chain is a chain with amino-acid atoms; D-amino acids and other non-canonical peptide-linking residues count, and waters or ligands that share the chain ID do not. `binding-metrics-run` and `-batch` resolve the chains once with `detect_chains_from_file` ([§19](#19-io-utilities)), which picks the target by Cα contacts when a file has more than two protein chains, and pass them to every step. `compute_interaction_energy` detects chains from the OpenMM topology with `io.structures.detect_chains`, which counts every amino-acid residue: the standard residues and the AMBER variants, the D-amino acids, the phospho residues and, with biotite installed, the peptide-linking components of the Chemical Component Dictionary. An all-D peptide is found as well.

**heteroatoms.** Waters, ions, ligands and glycans often carry the chain ID of a neighbouring protein chain. The keyword `hetero` (default `"ignore"`) decides what a chain ID selects. `"ignore"` keeps polymer atoms only: amino-acid residues as biotite's `filter_amino_acids` defines them (D-amino acids and other peptide-linking residues included), the AMBER variants `HID`, `HIE`, `HIN`, `CYX` and `ASH`, and the caps `ACE`, `NME` and `NH2`. `"keep"` uses every atom of the chain; atoms without a defined SASA (water, ions) then count as zero area. `hetero` exists on `compute_interface_metrics`, `compute_delta_sasa_static`, `compute_hbonds`, `compute_saltbridges`, `compute_shape_complementarity` and `compute_buried_void_volume`, and as `--hetero {ignore,keep}` on `binding-metrics-interface` and `binding-metrics-geometry`. The other metrics do not take it.

**values that could not be computed.** A function that cannot compute a value keeps its sentinel (NaN, 0 or None) and adds a string under the key `reason`. The key is absent when everything was computed, so a 0.0 without `reason` is a computed 0.0. Functions that already report `success`, `error_message` or `error` keep those keys.

**dependencies.** `import binding_metrics` does not import OpenMM, so the static metrics work on an install without it. A name whose dependency is missing raises an error that names the extra to install.

| extra | needed by |
|-------|-----------|
| `static` | interface, H-bonds, salt bridges, Coulomb, Ramachandran, omega, shape complementarity, void volume, static ΔSASA, structure comparison, EvoBind, parsing of OpenFold3 output |
| `simulation` | force-field interaction energy, relaxation, and the energy term of receptor quality |
| `structure` | PDBFixer preparation (`binding-metrics-prep`, the pipeline's prep step) and gemmi |
| `analysis` | trajectory metrics (§11) |
| `dockq` | DockQ (§10) |
| `report` | HTML summary (`markdown`) and the `binding-metrics-energy` CSV output (`pandas`) |

GAFF2 parameters for non-canonical residues need openmmforcefields, openff-toolkit, RDKit and AmberTools, which are conda-forge only; `environment.yml` installs them. OpenFold3 runs in its own environment ([`README.md`](../README.md#openfold3-optional)).

**units.** Static structure metrics report Å, Å², Å³ and degrees. Energies are in kJ/mol, except `delta_g_int` and the H-bond and salt-bridge scores (kcal/mol). Trajectory metrics use the MDTraj units (nm, nm²). See [§20](#20-unit-summary).

---

## 2. interface geometry

`compute_interface_metrics(cif_path, design_chain=None, receptor_chain=None, probe_radius=1.4, interface_threshold=0.5, *, binder_chain=None, target_chain=None, hetero="ignore", hydrogens="ignore")` — `binding_metrics.metrics.interface`

PISA-inspired interface descriptors on a static structure. Per-atom areas come from the Shrake-Rupley algorithm (Shrake & Rupley 1973) as implemented in biotite (960 points per atom, probe 1.4 Å, biotite's single van der Waals radius per element, 1.8 Å for an element without an entry). Water and monoatomic ions are not part of the surface. The buried area of an atom is its area alone minus its area in the complex, clamped at 0.

Hydrogen and deuterium atoms are dropped before the areas are computed (`hydrogens="ignore"`, the default). The solvation parameters below describe heavy atoms, and the hydrogens attached to a carbon or nitrogen would otherwise take part of its surface, so the areas, the polar/apolar partition and `delta_g_int` refer to heavy atoms whatever the protonation of the input. `hydrogens="keep"` uses the atoms left by the `hetero` filter; the total then includes the hydrogens, which are in neither the polar nor the apolar mask. H-bond and salt-bridge counts do not depend on the setting.

| key | type | unit | description |
|-----|------|------|-------------|
| `peptide_chain` | str | — | resolved binder chain ID |
| `receptor_chain` | str | — | resolved target chain ID |
| `delta_sasa` | float | Å² | buried SASA, summed over both partners: SASA(pep) + SASA(rec) − SASA(complex) |
| `sasa_peptide` | float | Å² | SASA of the isolated binder |
| `sasa_receptor` | float | Å² | SASA of the isolated target |
| `sasa_complex` | float | Å² | SASA of the complex |
| `delta_g_int` | float | kcal/mol | solvation term of binding (below); more negative = burial more favourable |
| `delta_g_int_kJ` | float | kJ/mol | same × 4.184 |
| `polar_area` | float | Å² | buried area of N and O atoms |
| `apolar_area` | float | Å² | buried area of C and S atoms |
| `fraction_polar` | float | — | `polar_area / delta_sasa`; NaN when nothing is buried |
| `n_interface_residues_peptide` | int | — | binder residues with at least `interface_threshold` Å² buried |
| `n_interface_residues_receptor` | int | — | same for the target |
| `interface_residues_peptide` | list[str] | — | `"RES:CHAIN:NUM"` labels |
| `interface_residues_receptor` | list[str] | — | `"RES:CHAIN:NUM"` labels |
| `per_residue` | list[dict] | — | per interface residue: `residue`, `chain`, `res_name`, `res_id`, `buried_sasa` (Å²), `delta_g_res` (kcal/mol), `polar_area` (Å²), `apolar_area` (Å²) |
| `hbonds` | int | — | cross-chain heavy-atom H-bond pairs (§3) |
| `hbond_energy` | float | kcal/mol | H-bond score, ≤ 0 |
| `saltbridges` | int | — | cross-chain salt-bridge residue pairs (§3) |
| `saltbridges_bidentate` | int | — | pairs with at least two atom-pair contacts |
| `saltbridge_energy` | float | kcal/mol | salt-bridge score, ≤ 0 |
| `reason` | str | — | only when a value could not be computed |

`polar_area` and `apolar_area` cover N, O, C and S. With the default they add up to `delta_sasa` unless the structure has atoms of other elements (phosphorus in a phosphorylated residue, for example). `delta_sasa` counts both partners, so the area buried on one side is about half of it.

**solvation energy.** ΔG_int = Σᵢ γᵢ · ΔAᵢ over the atoms of both chains, with ΔAᵢ the buried area of atom i. The five atom types are those of Eisenberg & McLachlan (1986), as tabulated in Table 1 of Krissinel & Henrick (2007) for the PISA method.

| atom type | γ (kcal/mol/Å² of buried area) | atoms |
|-----------|--------------------------------|-------|
| C | −0.016 | all carbon |
| S | −0.021 | sulfur |
| N/O (neutral) | +0.006 | backbone N and O, hydroxyl, amide, neutral His, termini |
| O(−) | +0.024 | Asp OD1/OD2, Glu OE1/OE2, phosphate oxygens of SEP/TPO/PTR |
| N(+) | +0.050 | Lys NZ, Arg NE/NH1/NH2, HIP ND1/NE2 |

The published values are given per Å² of accessible area (C +16, S +21, neutral N/O −6, O(−) −24, N(+) −50 cal/mol/Å²). ΔG_int is the solvation part of binding, so each value enters with the opposite sign and per Å² of buried area. Hydrogen, phosphorus, selenium and halogens carry no parameter. Charged atoms are those of the salt-bridge list in §3, so D-amino acids count through their L counterpart.

`delta_g_int` is uncalibrated. PISA adds explicit hydrogen-bond and salt-bridge terms, which this package reports as separate keys, and the parameters are not fitted to any data set used here. No threshold or affinity relation has been established for it, so read it as a relative score between designs of one target. On the deposited files of the bundled complexes (heavy atoms, waters removed) it is −11.05 (1YCR), −6.11 (1CWA) and −4.97 (3P8F) kcal/mol; on their relaxed structures it is −10.50, −6.79 and −5.43 kcal/mol.

### `compute_delta_sasa_static(cif_path, peptide_chain, receptor_chain, probe_radius=1.4, *, binder_chain=None, target_chain=None, hetero="ignore", hydrogens="ignore")`

Buried area only, without the per-atom decomposition. Both chains are required. Returns `delta_sasa`, `sasa_peptide`, `sasa_receptor` and `sasa_complex` (Å²), plus `reason` when a chain has no atoms or the SASA calculation failed. Atoms with an undefined area (water, ions under `hetero="keep"`) count as zero, and hydrogens are dropped unless `hydrogens="keep"`.

### CLI: `binding-metrics-interface`

```bash
binding-metrics-interface --input complex.cif [--binder-chain B] [--target-chain A] \
    [--probe-radius 1.4] [--threshold 0.5] [--hetero ignore] [--hydrogens ignore]
```

---

## 3. hydrogen bonds & salt bridges

`compute_hbonds(atoms, peptide_chain, receptor_chain, *, binder_chain=None, target_chain=None, hetero="ignore")`, `compute_saltbridges(atoms, peptide_chain, receptor_chain, distance_min=0.5, distance_max=5.5, *, binder_chain=None, target_chain=None, hetero="ignore")` — `binding_metrics.metrics.polar_contacts`

Both take a loaded `biotite.structure.AtomArray` and require both chains; `compute_interface_metrics` calls them on the structure it loaded. Each reports a score (kcal/mol) and a count. Both scores are heuristic ranking terms with package-chosen constants; they are not fitted to measured energies and do not compare with force-field or experimental free energies.

**hydrogen bonds** — returns `hbonds` (int), `hbond_energy` (kcal/mol, ≤ 0) and, when hydrogens could not be built or the detector failed, `reason`. In that case the count is 0 for that reason and does not mean that no H-bond exists.
- detector: biotite `struc.hbond()`, the Baker-Hubbard (1984) criterion with biotite's defaults: H···acceptor distance ≤ 2.5 Å, D–H···A angle 120° (donors and acceptors N, O, S)
- a structure without hydrogens (raw AlphaFold or OpenFold predictions, for example) gets a BondList from `connect_via_residue_names` and hydrogens from `hydride.add_hydrogen`; a failure warns and is reported in `reason`
- biotite returns `(donor, H, acceptor)` triplets; they are reduced to unique cross-chain `(donor_heavy, acceptor_heavy)` pairs, keeping the triplet with the shortest H···A distance, so an ARG NH1 with two equivalent hydrogens on one acceptor counts once
- score per pair: `E = −5.0 · cos²(180° − θ) / d_HA` with d_HA in Å, so an ideal geometry (2.0 Å, 180°) gives −2.5 kcal/mol and bent contacts are down-weighted smoothly

**salt bridges** — returns `saltbridges` (residue pairs), `saltbridges_bidentate` (pairs with at least two atom-pair contacts) and `saltbridge_energy` (kcal/mol, ≤ 0).

| residue | positive atoms | residue | negative atoms |
|---------|----------------|---------|----------------|
| LYS | NZ | ASP | OD1, OD2 |
| ARG | NH1, NH2, NE | GLU | OE1, OE2 |
| HIP | ND1, NE2 | SEP, TPO, PTR | O1P, O2P, O3P |

- D-amino acids are matched through their L counterpart (`DLY` as LYS)
- distance window: `0.5 Å < r < 5.5 Å` between a positive and a negative atom on opposite chains, no angle filter; the usual heavy-atom criterion for an ion pair is 4 Å (Barlow & Thornton 1983), so 5.5 Å favours sensitivity
- `HIS`, `HID` and `HIE` are treated as neutral; only `HIP` (AMBER's doubly protonated form) contributes. Raw predicted CIFs carry only `HIS`, so they have no histidine salt bridges; structures written by OpenMM may carry `HIP`
- atom-pair contacts are aggregated to residue pairs
- score per pair: Coulomb energy at ε = 4 with unit charges over the closest atom-pair distance, `E_pair = −83.02 / r_min` kcal/mol. A bidentate bridge scores stronger because `r_min` is the shorter contact

Neither score includes solvent screening, cooperativity or burial. For an electrostatic score over all charged atoms use `compute_coulomb_cross_chain` (§4).

---

## 4. electrostatics

`compute_coulomb_cross_chain(cif_path, peptide_chain=None, receptor_chain=None, dielectric=4.0, cutoff_ang=12.0, *, binder_chain=None, target_chain=None)` — `binding_metrics.metrics.electrostatics`

| key | type | unit | description |
|-----|------|------|-------------|
| `coulomb_energy_kJ` | float | kJ/mol | total cross-chain Coulomb energy; negative = net attractive |
| `coulomb_energy_kcal` | float | kcal/mol | same / 4.184 |
| `n_charged_pairs` | int | — | charged atom pairs within the cutoff |
| `n_attractive` | int | — | opposite-sign pairs within the cutoff |
| `n_repulsive` | int | — | same-sign pairs within the cutoff |
| `n_ionisable_residues_seen` | int | — | residues, on both chains, with at least one charged atom in the table |
| `n_residues_unrecognised` | int | — | amino-acid residues whose charge state the table does not model |
| `charged_atoms_peptide` | list[dict] | — | per charged atom: `residue`, `atom`, `charge`, `coords` |
| `charged_atoms_receptor` | list[dict] | — | same for the target |
| `reason` | str | — | only when a chain was not found; the scores are then 0 |

formula: E = (k / ε) · Σ q_i q_j / r_ij over cross-chain pairs with r_ij below the cutoff, with k = 1389.35 kJ·Å/(mol·e²), ε = 4.0 and a 12 Å cutoff (both settable). One uniform dielectric, formal charges taken from residue names, no pKa shifts, no chain termini and no solvent screening: a heuristic ranking score, not a binding free energy.

formal charges (D-amino acids use the entry of their L counterpart):

| residue | atoms | charge (e) |
|---------|-------|------------|
| LYS | NZ | +1 |
| ARG | NH1, NH2 | +0.5 each |
| HIP | ND1, NE2 | +0.5 each |
| ASP | OD1, OD2 | −0.5 each |
| GLU | OE1, OE2 | −0.5 each |
| SEP, TPO, PTR | O1P, O2P, O3P | −2/3 each |

Plain `HIS`, `HID` and `HIE` are neutral. `n_residues_unrecognised` counts amino-acid residues outside the standard, protonation-variant, D and phospho residues: non-canonical residues such as MLE, BMT or ABA in 1CWA (8 residues). Their charge, if any, is not modelled, so a non-zero count means the energy may be incomplete. A 0.0 with `n_residues_unrecognised` above zero is a computed 0.0 for the charges the table knows.

### CLI: `binding-metrics-electrostatics`

```bash
binding-metrics-electrostatics --input complex.cif [--binder-chain B] [--target-chain A] \
    [--dielectric 4.0] [--cutoff 12.0]
```

---

## 5. force-field interaction energy

`compute_interaction_energy(input_path, peptide_chain=None, receptor_chain=None, solvent_model="obc2", device="cuda", sample_id=None, modes=("raw", "relaxed", "after_md"), ph=7.4, relaxed_min_steps_restrained=500, relaxed_min_steps_full=2000, after_md_duration_ps=10.0, after_md_timestep_fs=2.0, after_md_temperature_k=300.0, random_seed=1, *, binder_chain=None, target_chain=None)` — `binding_metrics.metrics.energy`

Needs OpenMM (Eastman et al. 2017; `pip install "binding-metrics[simulation]"`, or `environment.yml` for a CUDA build). Registered as `structure_interaction_energy`.

E_int = E_complex − E_peptide − E_receptor, with AMBER ff14SB and implicit solvent. It is an end-state estimate in the spirit of MM-GBSA (Kollman et al. 2000): the isolated binder and target are evaluated at their geometry in the complex. E_int therefore contains the force-field interaction and the change in generalized Born polar solvation and nonpolar surface term, and it leaves out conformational reorganisation and entropy. Compare values between structures of the same system, not with measured binding free energies.

| key | type | unit | description |
|-----|------|------|-------------|
| `sample_id` | str | — | identifier (file stem by default) |
| `success` | bool | — | True when at least one mode produced an energy |
| `error_message` | str \| None | — | one `"<mode>: <reason>"` entry per failed mode, joined by `"; "` |
| `num_contacts` | int | — | binder–target atom pairs closer than 8 Å in the input file |
| `num_close_contacts` | int | — | pairs closer than 4 Å |
| `raw_interaction_energy` | float | kJ/mol | E_int at the input geometry with hydrogens added |
| `raw_e_complex`, `raw_e_peptide`, `raw_e_receptor` | float | kJ/mol | the three terms (raw) |
| `relaxed_interaction_energy` | float | kJ/mol | E_int after minimisation |
| `relaxed_e_complex`, `relaxed_e_peptide`, `relaxed_e_receptor` | float | kJ/mol | the three terms (relaxed) |
| `after_md_interaction_energy` | float | kJ/mol | E_int at the last frame of a short MD run |
| `after_md_e_complex`, `after_md_e_peptide`, `after_md_e_receptor` | float | kJ/mol | the three terms (after MD) |

Keys of modes that were not requested are absent. A mode that failed is `None`. `num_contacts` and `num_close_contacts` count the atoms present in the file, hydrogens included when the file has them, and do not depend on the modes.

**modes**

| mode | protocol |
|------|----------|
| `raw` | add hydrogens, single-point energy; `None` for severe clashes, which makes it a clash indicator |
| `relaxed` | add hydrogens, 500 steps of backbone-restrained minimisation, then 2000 steps unrestrained |
| `after_md` | its own restrained and unrestrained minimisation, then Langevin MD (default 10 ps, 2 fs step, 300 K, friction 1/ps); needs no `relaxed` |

The `after_md` duration is the argument `after_md_duration_ps` (10 ps). The 200 ps default of `binding-metrics-run --md-duration-ps` belongs to the relaxation step, which runs before the energy step ([§15](#15-pipeline-results-and-provenance)).

**implementation notes**
- force field: ff14SB (Maier et al. 2015) through `amber14-all.xml`; implicit solvent: OBC2 (Onufriev et al. 2004) or GBn2 (Nguyen et al. 2013) from OpenMM's `implicit/obc2.xml` and `implicit/gbn2.xml` (solute dielectric 1, solvent 78.5, OpenMM's nonpolar surface term included); no cutoff; bonds to hydrogen constrained
- hydrogens are added once, at `ph`, and shared by the modes. If the pH-aware call fails, hydrogens are added again without a pH (OpenMM's default 7.0)
- residues outside the binder and target chains that are not amino acids (waters, ions, ligands) are stripped, with a warning when one lies within 8 Å of the protein. Waters or ligands that carry the ID of the binder or target chain are not touched here, and `binding-metrics-run` removes them in its prep step
- protein chains other than the binder and the target are removed as well, and a warning names each of them. E_complex would otherwise contain a chain that the isolated terms leave out. The relaxation does the same and lists the IDs under `dropped_protein_chains` ([§15](#15-pipeline-results-and-provenance)); [`nonstandard.md`](nonstandard.md#other-protein-chains) gives the reason. A receptor of several chains has to be reduced to one before it goes in
- cyclic peptides: the closure bond is patched from custom templates; CYS–CYS disulfides are renamed CYX before `addHydrogens`, and a CYX whose partner lies on the other chain is converted back to CYS for the per-chain terms
- non-canonical residues: D-amino acids and N-methylated residues use the templates of [`nonstandard.md`](nonstandard.md); other residues get GAFF2 templates; phosphorylated residues use the AMBER phosaa parameters
- the seed drives hydrogen placement, the conformer of the AM1-BCC charges of GAFF2 residues, the Langevin noise and the initial velocities; `random_seed=None` draws fresh randomness. CUDA runs in mixed precision, which is not bit-reproducible, so GPU energies from one seed can differ in the last digits

### CLI: `binding-metrics-energy`

```bash
binding-metrics-energy --input complex.cif --modes raw relaxed --device cuda --output scores.csv
binding-metrics-energy --input-dir designs/ --glob-pattern "*.cif" --modes relaxed after_md --output scores.csv
```

The CLI computes all three modes when `--modes` is omitted. Further flags: `--solvent-model {obc2,gbn2}`, `--relaxed-min-steps-restrained`, `--relaxed-min-steps-full`, `--after-md-duration-ps`, `--after-md-timestep-fs`, `--after-md-temperature-k`, `--ph`, `--random-seed INT|none`, `--binder-chain`, `--target-chain`.

### Reserved interface: MLFF interaction energy

`compute_mlff_interaction_energy(structure_path, binder_chain=None, receptor_chain=None, *, backend="uma", pocket=None, unit="kcal_mol", hetero="ignore", hydrogens="keep", target_chain=None)` — `binding_metrics.metrics.mlff_energy`

The function checks its arguments and raises `NotImplementedError`: no backend exists yet. It is not in the metric registry ([§16](#16-metric-registry)), so code that runs the registered metrics never calls it; it is registered when a backend lands. Importing the module needs no MLFF package, OpenMM or torch.

It will compute a static-pocket interaction energy from a pretrained machine-learned force field, after Ryczko et al. (2026, ChemRxiv preprint): the pocket around the binder is cut out and capped, and E(pocket complex) − E(binder) − E(receptor pocket) is evaluated at the geometry of the complex, with no relaxation. The result keys will be `mlff_interaction_energy_<unit>` (`_kcal_mol` by default; `unit` also takes `"kj_mol"` and `"ev"`), `backend`, `weights_licence`, `n_atoms_complex`, `n_atoms_binder`, `n_atoms_receptor`, `pocket_cutoff_angstrom` and, only when the energy is NaN, `reason`.

- It complements E_int above and does not replace it. E_int includes generalised Born solvation and the solvent treatment of the reference protocol is not known, so the two values may not be comparable.
- The reference benchmarks congeneric small-molecule series. Peptides, D-amino acids, N-methylated and phosphorylated residues and macrocycles are unvalidated, and a pocket cropper with capping, which does not exist yet, is needed before any of them can be scored.
- The weights of a model are not part of this package and are never bundled. UMA weights are gated under the FAIR Chemistry License v1 (acceptable-use policy and acknowledgement duty). The licence of each backend is `MLFFBackend.weights_licence` and appears in the result.
- Backends: `MLFFBackend` (abstract `energy_ev(atoms, *, charge, spin)` in eV), `PocketSpec`, `register_backend`, `get_backend` and `available_backends`. The names `uma`, `mace`, `orb` and `aimnet2` are placeholders: they are known, are never listed as available, and raise `NotImplementedError` that names the backend and the reference.

---

## 6. ramachandran & omega planarity

`compute_ramachandran(cif_path, chain=None, *, binder_chain=None)`, `compute_omega_planarity(cif_path, chain=None, *, binder_chain=None)` — `binding_metrics.metrics.geometry`

Without `chain`, the smallest protein chain is evaluated. A chain ID that is not in the file raises `ValueError` and lists the chains present. A chain that exists but holds no amino acids returns NaN with a `reason`.

### ramachandran

| key | type | unit | description |
|-----|------|------|-------------|
| `ramachandran_favoured_pct` | float | % | residues in favoured φ/ψ regions |
| `ramachandran_allowed_pct` | float | % | residues in allowed regions |
| `ramachandran_outlier_pct` | float | % | residues outside both |
| `ramachandran_outlier_count` | int | — | number of outliers |
| `n_residues_evaluated` | int | — | residues with a complete φ/ψ pair |
| `n_d_residues` | int | — | evaluated residues that are D-amino acids |
| `cyclic_closure_detected` | bool | — | last C is bonded to first N (C–N distance below 2 Å) |
| `cyclic_closure_evaluated` | bool | — | the ring-closing φ and ψ were computed and are included |
| `per_residue` | list[dict] | — | `res_id`, `res_name`, `chain`, `phi` (°), `psi` (°), `is_d_aa`, `region` |
| `reason` | str | — | only when no residue could be evaluated |

The regions are hand-drawn rectangles in the (φ, ψ) plane, not the density contours of MolProbity (Lovell et al. 2003; Williams et al. 2018). They approximate the general L-residue case: glycine, proline and pre-proline residues have no classes of their own, so glycine at φ > 0 is judged against the general regions. The percentages are a screen and do not match MolProbity's.

favoured regions:

| region | φ (°) | ψ (°) |
|--------|-------|-------|
| α-helix | −90 to −30 | −80 to +10 |
| β-sheet | −180 to −45 | 90 to 180 or −180 to −160 |
| PPII | −90 to −50 | 120 to 180 |
| left-handed helix | +20 to +90 | 0 to +85 |

allowed regions: φ −125 to 0 with ψ −100 to +30; φ −180 to −30 with ψ 60 to 180 or −180 to −100; φ 0 to +110 with ψ −30 to +100. Everything else is an outlier.

D-amino acids: φ and ψ are negated before the lookup, so a D-α-helix (φ ≈ +57°, ψ ≈ +47°) falls in the α-helix region.

Terminal residues of a linear chain have no complete φ/ψ pair and are skipped. For a head-to-tail cyclic peptide, φ of the first residue and ψ of the last residue are computed across the closing amide bond, so every residue is evaluated. N-methylation and ring constraints move φ/ψ outside the general regions: the favoured fraction of the bundled macrocycles is 72.7 % (1CWA, chain C) and 85.7 % (3P8F, chain I), against 90.9 % for the linear peptide of 1YCR.

### omega planarity

| key | type | unit | description |
|-----|------|------|-------------|
| `omega_mean_dev` | float | ° | mean deviation of ω from 180° |
| `omega_max_dev` | float | ° | maximum deviation |
| `omega_outlier_fraction` | float | — | fraction of bonds with a deviation above 15° |
| `omega_outlier_count` | int | — | number of such bonds |
| `n_bonds_evaluated` | int | — | peptide bonds with a defined ω |
| `omega_cis_count` | int | — | evaluated bonds with \|ω\| < 30° |
| `cyclic_closure_detected` | bool | — | as above |
| `cyclic_closure_evaluated` | bool | — | the closing peptide bond was evaluated; it is listed under the last residue |
| `per_residue` | list[dict] | — | `res_id`, `res_name`, `chain`, `omega` (°), `deviation` (°), `is_outlier` |
| `reason` | str | — | only when no bond could be evaluated |

deviation = min(\|ω − 180°\|, \|ω + 180°\|). The 15° cut-off is a package heuristic. A cis bond has a deviation near 180° and counts as an outlier, cis-proline and N-methylated amides included; `omega_cis_count` tells cis bonds from twisted trans bonds.

### CLI: `binding-metrics-geometry`

```bash
binding-metrics-geometry --metric ramachandran --input complex.cif [--binder-chain B]
binding-metrics-geometry --metric omega --input complex.cif [--binder-chain B]
```

---

## 7. shape complementarity

`compute_shape_complementarity(cif_path, peptide_chain=None, receptor_chain=None, n_dots=150, interface_cutoff=6.0, buried_cutoff=2.4, normal_radius=6.0, weight=0.5, *, binder_chain=None, target_chain=None, hetero="ignore")` — `binding_metrics.metrics.geometry`

A dot-and-normal approximation of the Sc score of Lawrence & Colman (1993). It is not a port of the CCP4 `sc` program, so do not compare its values numerically with `sc` output; compare values computed here with each other. The surface is built from every atom that is present after the `hetero` filter, hydrogens included, so compare complexes with the same protonation state: MDM2-p53 (1YCR) gives 0.632 on the deposited file, 0.628 after relaxation with hydrogens and 0.664 with the hydrogens of the relaxed structure removed.

| key | type | unit | description |
|-----|------|------|-------------|
| `sc` | float | [−1, 1] | mean of the two directional medians; NaN when there is no interface |
| `sc_A_to_B` | float | [−1, 1] | median score of binder surface dots against the target |
| `sc_B_to_A` | float | [−1, 1] | median score of target surface dots against the binder |
| `n_surface_dots_A` | int | — | interface surface dots on the binder |
| `n_surface_dots_B` | int | — | interface surface dots on the target |
| `per_dot_scores_A`, `per_dot_scores_B` | np.ndarray | — | the S values behind the medians |
| `reason` | str | — | only when Sc could not be computed |

algorithm:
1. Atoms of each chain that lie within `interface_cutoff` (6.0 Å) of the other chain are the candidates; this pre-selection only decides which atoms get dots.
2. Each candidate atom gets `n_dots` (150) points from a Fibonacci lattice on its van der Waals sphere (Bondi radii: C 1.70, N 1.55, O 1.52, S 1.80, H 1.20, P 1.80 Å; 1.80 Å for other elements). Points inside another same-chain sphere are dropped.
3. A point stays only when the nearest atom of the other chain is within `buried_cutoff` (2.4 Å). This keeps the buried contact patch.
4. The outward normal at a point is the direction from the centroid of the same-chain atoms within `normal_radius` (6.0 Å) to the point.
5. For each dot on one surface, S = (n · −n′) · exp(−w d²), with n′ the normal at the nearest dot of the other surface, d their distance and w = `weight` (0.5 Å⁻²). Anti-parallel normals give a positive product.
6. Each direction takes the median of S over its dots; `sc` is the mean of the two medians.

Typical values for native interfaces are about 0.65–0.75 for protein–protein and protease–inhibitor packing, somewhat lower for peptide interfaces; values below about 0.4 indicate flat, non-complementary surfaces. On the bundled complexes: 0.632 (1YCR), 0.738 (3P8F), 0.750 (1CWA).

### CLI

```bash
binding-metrics-geometry --metric sc --input complex.cif [--binder-chain B] [--target-chain A] \
    [--n-dots 150] [--interface-cutoff 6.0] [--buried-cutoff 2.4] [--normal-radius 6.0] \
    [--weight 0.5] [--hetero ignore]
```

---

## 8. buried void volume

`compute_buried_void_volume(cif_path, peptide_chain=None, receptor_chain=None, grid_spacing=0.5, probe_radius=1.4, interface_cutoff=5.0, padding=3.0, *, binder_chain=None, target_chain=None, hetero="ignore")` — `binding_metrics.metrics.geometry`

| key | type | unit | description |
|-----|------|------|-------------|
| `void_volume_A3` | float | Å³ | volume of the interface void; lower = tighter packing |
| `void_grid_fraction` | float | — | void voxels / voxels of the interface box |
| `interface_box_volume_A3` | float | Å³ | volume of the interface box |
| `n_interface_atoms` | int | — | atoms that define the interface box |
| `reason` | str | — | only when the volume could not be computed |

An interface void is a cavity that a probe of radius `probe_radius` cannot reach from the bulk in the complex, that would be open if the binder were removed, and that would be open if the target were removed. It is space that only the two partners together close off. A cavity walled by one chain is not counted, and neither are gaps between atoms that are too narrow for the probe. `probe_radius` must be ≥ 0.

algorithm, on a regular grid:
1. The grid covers the interface box (atoms within `interface_cutoff` of the other chain, padded by `padding`) and extends a few Å beyond it, so that cavities cut by the box face are classified whole; only voxels inside the box are counted.
2. All atoms of both chains that reach into the grid are occluders. A probe centre is allowed where it stays at least van der Waals radius + `probe_radius` from every atom centre (Lee & Richards 1971).
3. Allowed centres that are not connected to the outside in the complex, but are connected to it for the binder alone and for the target alone, seed the void.
4. The void is the region swept by the probe body from those seeds (Richards 1977; Connolly 1983), minus anything the bulk-solvent probe can touch. `probe_radius=0` reduces to the spaces closed to a point probe. Grid cavity detection in this spirit is used by VOIDOO (Kleywegt & Jones 1994).

The volume depends on `probe_radius` and `grid_spacing`; compare only values computed with the same settings. It is a voxel count, so its discretisation error grows with the cavity surface and the voxel size. On the bundled 1YCR, 3P8F and 1CWA examples at the default probe the void is 55.9, 31.9 and 16.3 Å³ (55.875, 31.875 and 16.25); it is 0 for tightly packed interfaces. On 1YCR the value is 55.1, 55.9 and 16.5 Å³ at probe radii 0.5, 1.4 and 3.0 Å.

SASA and void volume answer different questions: SASA measures surface reachable by a probe, the void measures enclosed empty space that no probe reaches from outside.

### CLI

```bash
binding-metrics-geometry --metric void --input complex.cif [--binder-chain B] [--target-chain A] \
    [--grid-spacing 0.5] [--probe-radius 1.4] [--interface-cutoff 6.0] [--hetero ignore]
```

`--interface-cutoff` defaults to 6.0 on the command line, where it also sets the Sc pre-selection, while the function default for the void is 5.0. The command line therefore reports the void with a 6.0 Å cutoff (56.0 Å³ for 1YCR instead of 55.875).

---

## 9. structure comparison (RMSD)

`compute_structure_rmsd(initial_path, processed_path, design_chain=None, *, binder_chain=None)` — `binding_metrics.metrics.comparison`

Needs gemmi (`pip install "binding-metrics[structure]"` or `[static]`). Optimal superposition by the Kabsch algorithm (Kabsch 1976); a rigidly rotated and translated copy gives 0.

| key | type | unit | description |
|-----|------|------|-------------|
| `rmsd` | float \| None | Å | all-atom RMSD of the full complex |
| `bb_rmsd` | float \| None | Å | backbone (N, CA, C, O) RMSD of the full complex |
| `rmsd_design` | float \| None | Å | all-atom RMSD of the design chain |
| `bb_rmsd_design` | float \| None | Å | backbone RMSD of the design chain |
| `reason` | str | — | only when a value is None; names each variant and why |

- only the first model of each file is read
- `HOH` and `WAT` residues are skipped; ions and ligands are included
- when the two selections hold the same number of atoms, atoms are paired in file order; otherwise atoms are matched by (chain, residue number, atom name) and only the shared ones are used
- without `design_chain`, the smallest chain with a non-water residue is the design chain; a one-residue ligand chain would win, so pass the chain then

### CLI: `binding-metrics-compare`

```bash
binding-metrics-compare --initial initial.cif --processed relaxed.cif [--binder-chain B]
```

The reason, when there is one, goes to stderr.

---

## 10. reference-based accuracy (DockQ)

`compute_dockq_metrics(model_path, reference_path, mapping=None)` — `binding_metrics.metrics.dockq`

Scores a predicted complex against a known native with the DockQ tool (Basu & Wallner 2016), which reports the CAPRI quantities. It needs the DockQ package (`pip install "binding-metrics[dockq]"`), and the `DockQ` executable must be on `PATH`. Unlike the other structure metrics it compares two structures: use it for benchmarking against a native and not to score a design alone.

The function runs `DockQ <model> <reference> --json ...` and parses the JSON. DockQ searches for the best chain mapping itself, so chains that are named or ordered differently in prediction and reference are matched (antibody–antigen complexes among them). `mapping` fixes part or all of the mapping in DockQ's `model:native` convention (`"AB:CD"`).

| key | type | unit | description |
|-----|------|------|-------------|
| `dockq` | float | [0, 1] | global DockQ, the mean over interfaces |
| `capri_class` | str | — | class of the global score |
| `n_interfaces` | int | — | number of interfaces scored |
| `best_mapping` | str \| None | — | chain mapping DockQ chose |
| `interfaces` | list[dict] | — | per interface: `chains`, `DockQ`, `fnat`, `fnonnat`, `iRMSD` (Å), `LRMSD` (Å), `clashes`, `capri_class` |

CAPRI classes from the DockQ score: below 0.23 `Incorrect`, from 0.23 `Acceptable`, from 0.49 `Medium`, from 0.80 `High`. `fnat` is the fraction of native residue contacts recovered, `fnonnat` the fraction of predicted contacts that are not native, `iRMSD` the interface backbone RMSD and `LRMSD` the ligand RMSD after receptor superposition. The function raises `ImportError` without DockQ, `FileNotFoundError` for a missing file and `RuntimeError` when DockQ fails.

In the pipeline, `--reference` (single) or `--reference-dir` (batch, files matched by stem) enables the `dockq` metric. The prediction is scored as submitted, not the prepped or relaxed pose, because prep and MD move atoms and would confound the accuracy measure.

```bash
binding-metrics-dockq --model predicted.cif --reference native.cif [--mapping AB:CD]
```

---

## 11. MD trajectory metrics

`binding_metrics.metrics.rmsd`, `.sasa`, `.contacts`, `.energy` — need MDTraj (`pip install "binding-metrics[analysis]"`); the energy functions also need OpenMM. Trajectory functions take `trajectory_path` and `topology_path`, and atom-index lists (`ligand_indices`, `receptor_indices`) where the static metrics take chain IDs. Distances are in nm.

| function | returns | unit |
|----------|---------|------|
| `calculate_rmsd(..., atom_indices=None, reference_frame=0, *, on_empty="warn")` | per-frame RMSD after superposition (default selection `protein and not type H`) | nm |
| `calculate_rmsf(..., atom_indices=None)` | per-atom fluctuation about the mean after a fit on frame 0 | nm |
| `calculate_ligand_rmsd(..., ligand_indices, receptor_indices, reference_frame=0)` | dict `ligand_rmsd`, `receptor_rmsd` (per frame) | nm |
| `calculate_buried_sasa(..., ligand_indices, receptor_indices, probe_radius=0.14)` | per-frame buried SASA | nm² |
| `calculate_interface_sasa(...)` | dict `ligand`, `receptor`, `complex`, `buried` (per frame) | nm² |
| `calculate_contacts(..., cutoff=0.45, *, on_empty="warn")` | atom-pair contacts per frame | count |
| `calculate_contact_residues(..., cutoff=0.45)` | dict `ligand_residues`, `receptor_residues`, `contact_frequency` | — |
| `calculate_interaction_energy(..., forcefield_name="amber")` | per-frame Coulomb + Lennard-Jones sum over ligand–receptor pairs (vacuum, no cutoff) | kJ/mol |
| `calculate_component_energies(...)` | dict `electrostatic`, `vdw`, `total` (per frame) | kJ/mol |

- `calculate_ligand_rmsd` superposes each frame on the receptor atoms and reports the plain root-mean-square displacement of the ligand atoms in that frame, with no second fit (the CAPRI ligand RMSD; Mendez et al. 2003). It contains the rigid-body motion of the ligand relative to the receptor, so a ligand moved 5 Å without any internal change gives 0.5 nm. Earlier versions fitted the ligand a second time and returned 0 for such a shift.
- With an empty selection, `calculate_rmsd` and `calculate_contacts` return zeros and emit a `RuntimeWarning`, which is not an RMSD or a contact count; `on_empty="raise"` raises `ValueError` instead. The default selection of `calculate_rmsd` knows standard residue names only, so a chain made of unrecognised non-canonical residues matches nothing.
- `calculate_contacts` and `calculate_contact_residues` count the atoms whose indices are passed, hydrogens included, so the count depends on the protonation state of the input unless heavy-atom indices are given.
- `calculate_interaction_energy` (trajectory) is registered as `interaction_energy`; the per-structure decomposition of §5 is `structure_interaction_energy`.

### receptor drift

`compute_receptor_drift(trajectory_path, topology_path, receptor_chain=None, reference_frame=0, *, target_chain=None)` — `binding_metrics.metrics.rmsd`

Stability of the receptor backbone across the frames. `receptor_chain` (or `target_chain`) is required.

| key | type | unit | description |
|-----|------|------|-------------|
| `drift_aligned_mean` | float | Å | mean per-frame receptor Cα RMSD after superposition (conformational drift) |
| `drift_aligned_max` | float | Å | maximum aligned drift |
| `drift_raw_mean` | float | Å | mean Cα displacement without superposition; NaN if PBC is detected |
| `drift_raw_max` | float | Å | maximum raw drift; NaN if PBC is detected |
| `pbc_detected` | bool | — | the trajectory stores a unit cell |
| `drift_aligned_per_frame` | np.ndarray | Å | per-frame aligned drift (n_frames) |
| `drift_raw_per_frame` | np.ndarray | Å | per-frame raw drift; NaN if PBC is detected |
| `n_receptor_ca` | int | — | receptor Cα atoms used |
| `n_frames` | int | — | frames in the trajectory |

algorithm: select the receptor Cα atoms by chain ID; aligned drift superposes every frame on the Cα atoms of `reference_frame` and takes the RMSD, which isolates conformational change from global motion and is still computed when a unit cell is stored; raw drift takes the RMSD of the positions without superposition, which also contains global translation and rotation, and is set to NaN when a unit cell is stored because wrapped coordinates make raw displacements meaningless. Use `drift_aligned_*` to ask whether the receptor changes conformation, and `drift_raw_*` only for non-periodic runs, to ask whether the complex diffuses or tumbles.

---

## 12. OpenFold3 confidence scores

`compute_openfold_metrics(output_dir, query_name, seed=1, sample=1, include_matrices=False, reference_structure_path=None, binder_chain=None, receptor_chain=None, *, seed_index=None, target_chain=None)` — `binding_metrics.metrics.openfold`

Parses the output of an OpenFold3 run. It needs no OpenFold3 install, only the output files. `seed` is the 1-based position of a `seed_*` directory of the query in the numeric order of the seed values (`seed_9` before `seed_10`), not a random seed value (OpenFold3 names the directories after its own seeds); `seed_index` is a clearer name for the same argument and takes precedence. The function reads the output through the `of3` adapter (`get_parser("of3")`, see "Prediction records" below) and analyses the record with `summarize_prediction`.

expected layout:

```
{output_dir}/{query_name}/seed_{S}/
    {prefix}_model.cif (.cif.gz or .pdb)   predicted structure, pLDDT in the B-factor column
    {prefix}_confidences_aggregated.json   scalar scores
    {prefix}_confidences.json (or .npz)    per-atom and per-token arrays
    timing.json
```

| key | type | unit | description |
|-----|------|------|-------------|
| `query_name`, `seed`, `sample` | str, int, int | — | what was parsed (`seed` is the index) |
| `structure_path` | str \| None | — | path to the predicted structure |
| `avg_plddt` | float | [0–100] | mean per-atom pLDDT |
| `gpde` | float | Å | global predicted distance error |
| `ptm` | float | [0–1] | predicted TM-score; NaN without the PAE head |
| `iptm` | float | [0–1] | interface pTM; NaN for a single chain or without the PAE head |
| `disorder` | float | [0–1] | mean relative SASA |
| `has_clash` | float | 0 or 1 | steric clash in the prediction |
| `sample_ranking_score` | float | — | OpenFold3 composite ranking score |
| `chain_ptm` | dict | — | per-chain pTM: `{chain_id: float}` |
| `chain_pair_iptm` | dict | — | pairwise ipTM |
| `bespoke_iptm` | dict | — | OpenFold3's bespoke interface score, per chain pair |
| `plddt_per_atom` | np.ndarray \| None | [0–100] | per-atom pLDDT, shape (n_atoms) |
| `n_atoms` | int | — | number of atoms |
| `max_pde`, `max_pae` | float | Å | largest PDE and PAE values; NaN if the matrix is absent |
| `pde`, `pae` | np.ndarray \| None | Å | full matrices (n_tokens × n_tokens); only with `include_matrices=True` |
| `timing` | dict | — | runtime entries of `timing.json`, empty if absent |
| `binder_avg_plddt` | float | [0–100] | mean pLDDT over the binder residues; needs `binder_chain` |
| `binder_plddt_per_residue` | np.ndarray \| None | [0–100] | per-residue mean pLDDT of the binder |
| `mean_interface_pde`, `max_interface_pde` | float | Å | PDE over binder × target tokens; needs both chains |
| `mean_interface_pae`, `max_interface_pae` | float | Å | PAE over binder × target tokens (below); needs both chains |
| `pde_interface`, `pae_interface` | np.ndarray \| None | Å | the slices; only with `include_matrices=True` |
| `binder_ca_rmsd` | float | Å | binder Cα RMSD against `reference_structure_path` (refolding mode): in the receptor frame when `receptor_chain` is given, otherwise after superposing the binder Cα; NaN with a `reason` when the receptor Cα counts differ or are fewer than 3 |
| `reason` | str | — | only when a value could not be computed; names each affected analysis |

**interface PAE.** PAE at token (i, j) is the expected position error of token j when the prediction is aligned on token i (the row is the alignment frame, the column the scored token). The interface block is the binder × target slice of the PAE matrix, that is the error of the target tokens when the structure is aligned on the binder. `mean_interface_pae` averages the binder→target and target→binder blocks, so it does not depend on the slice direction, and `max_interface_pae` is the larger of the two maxima. `compute_interface_pae(confidences_path, structure_path, binder_chain, receptor_chain=None, *, target_chain=None)` returns the slice and its statistics (`pae_interface`, `mean_interface_pae`, `max_interface_pae`, `n_binder_tokens`, `n_receptor_tokens`) from a confidences file and the predicted structure; it is registered as `interface_pae` and raises `ValueError` when the file has no PAE matrix. The PAE head is written by the `pae_enabled` model preset.

The interface blocks are located with one token per residue. OpenFold3 uses one token per standard residue but one per heavy atom for ligands and modified residues, so a prediction that holds such components makes the matrix larger than the residue count. The interface values then stay NaN, a warning is issued and `reason` says why; the earlier code sliced the wrong block silently.

Higher pLDDT, pTM and ipTM and lower PDE and PAE are better. The Markdown summary lists binder residues with a pLDDT below 70. The package sets no other limit for OpenFold3 scores, and none has been calibrated here.

### `run_openfold(query_json, output_dir, inference_ckpt_path=None, num_diffusion_samples=5, num_model_seeds=1, use_msa_server=True, model_presets=None, runner_yaml=None, extra_args=None, conda_env=None, template_dir=None) → Path`

Runs `run_openfold predict` as a subprocess, in the current environment or through `conda run -n <conda_env>`, and returns the output directory. `run_openfold_scoring` (both chains as templates, confidence of an existing complex) and `run_openfold_refolding` (binder predicted freely, receptor as template) prepare the query and call it; both take `seeds` (default `(42,)`), the seed values written to the query JSON.

| argument | default | description |
|----------|---------|-------------|
| `num_diffusion_samples` | 5 | structures sampled per query |
| `num_model_seeds` | 1 | passed as `--num_model_seeds` |
| `use_msa_server` | True | ColabFold MSA server; the sequences leave the machine, and results can change over time because the alignments come from a remote service |
| `model_presets` | `["predict", "low_mem"]` | presets written to a runner YAML; `predict` is always included |
| `runner_yaml` | None | explicit YAML; overrides `model_presets` |

presets: `predict` is the required base preset and `low_mem` computes the pairformer embeddings sequentially, which suits large complexes or limited GPU memory. There is no preset for the PAE head: OpenFold3 0.4.1 removed `pae_enabled`, and pTM, ipTM and PAE are always written. A `pae_enabled` entry in `model_presets` (or `--presets`) is left out of the runner YAML with a `DeprecationWarning`, and the run continues.

**inputs stay on disk.** With templates and the MSA server (both on by default) OpenFold3 removes the parent of `template_preprocessor_settings.structure_directory` when a run ends. `run_openfold_scoring`, `run_openfold_refolding` and `run_openfold_batched` put the template CIFs in `<output_dir>/query/templates`, so the runner YAML they write sets `msa_computation_settings.cleanup_msa_dir: false`, and `<output_dir>/query` (query JSON, A3M files, template CIFs) survives. The setting is not written when no `template_dir` is given, or into a `runner_yaml` you supply.

**query sequences.** The sequence of each chain comes from the residue names of the input structure. The 20 standard residues are plain letters, and protonation and cross-link variants (HID, HIE, HIP, HIN, CYX, CYM, ASH, GLH, LYN and the lactam templates ASPL, GLUL, LYSL) take the letter of their parent residue; the variant or link itself is not sent. D-amino acids, N-methylated residues and other peptide-linking components of the Chemical Component Dictionary keep their chemistry through `non_canonical_residues` in the query JSON (`{"1": "DAL"}`, 1-based positions, CCD codes; the toolkit's template names NMG and NMA are sent as SAR and MAA), and their sequence letter is the parent residue's, upper case; OpenFold3 reads only upper-case letters and would turn the lower-case letters that gemmi uses for modified residues into unknown residues. Waters, ions, ligands and terminal caps are left out. A residue with a backbone that is none of these raises `UnmappableResidueError` (a `ValueError`) before any file is written or process started; the message names the chain, the residues and their numbers, and a batch lists every affected sample. `on_unmappable_residue="x"` on the `prepare_*` functions, or `--on-unmappable-residue x` on the command line, sends an `X` in its place instead and logs a warning.

### CLI: `binding-metrics-openfold`

```bash
# parse an existing output directory
binding-metrics-openfold parse --output-dir ./openfold_out --query-name my_complex \
    --seed 1 --sample 1 [--binder-chain B --target-chain A] [--include-matrices]

# run inference, then parse
binding-metrics-openfold run --query-json query.json --output-dir ./openfold_out \
    --query-name my_complex --num-samples 5 --num-seeds 1 \
    [--presets predict low_mem] [--no-msa-server]
```

Further subcommands: `prepare-query` and `refold` (binder refolding), `prepare-scoring-query` and `score` (scoring an existing complex). `--seed` is the seed-directory index, as in the function. In `binding-metrics-run` and `-batch`, `--openfold-seeds SEED [SEED ...]` sets the seed values of the query JSON and the first seed's first sample is scored.

### Prediction records (`binding_metrics.predictors`)

`PredictionRecord` is the form of one prediction sample that every predictor adapter converts a model's files to, and the only thing the confidence metrics read. `PredictionFiles` says where an adapter found the files of a sample, `TokenLayout` says what each PAE and PDE token is, and `SampleRef` names one sample of a directory. The rules are the same for every model:

| field | type | unit | meaning |
|-------|------|------|---------|
| `model`, `name` | str | — | adapter name (`"of3"`) and prediction name; the only positional arguments |
| `seed_index`, `sample` | int | — | 1-based positions in the model's natural order, not seed values and not a ranking |
| `structure_path` | Path \| None | — | predicted structure; `atoms()` reads model 1 with the author IDs, gzip-compressed files included |
| `chain_map` | dict | — | model chain ID to user chain ID, applied by `atoms()`; empty means no renaming |
| `avg_plddt` | float | [0–100] | mean pLDDT |
| `ptm`, `iptm` | float | [0–1] | definitions differ by model |
| `gpde` | float | Å | global predicted distance error |
| `ranking_score`, `ranking_score_name` | float, str | — | the model's own score and its name there; never compared across models |
| `has_clash`, `disorder` | float | — | clash flag (0 or 1) and disorder fraction |
| `chain_ptm`, `chain_pair_iptm` | dict | [0–1] | keys as the model writes them |
| `plddt_per_atom` | ndarray \| None | [0–100] | one value per atom, in the atom order of `structure_path` |
| `pae`, `pde` | ndarray \| None | Å | `(n_tokens, n_tokens)`; `pae[i, j]` is the error of token j when the structure is aligned on token i |
| `tokens` | TokenLayout \| None | — | chain, residue and atom of each token; None means one token per residue |
| `extras`, `timing`, `reasons` | dict, dict, list | — | model-specific values, reported run times, one sentence per value that could not be provided |
| `files` | PredictionFiles \| None | — | the files the record was parsed from; `load` fills it |

A scalar the model does not provide is NaN, never None. A file that is absent leaves the fields it feeds at NaN (None for an array) and adds a sentence to `reasons`; a file that is present but corrupt raises. An adapter converts a 0–1 pLDDT to 0–100 and expands a per-residue or per-token pLDDT to atoms. `record.validate()` raises `ValueError` listing every violation of these rules (a scale, a shape, a `chain_map` that renames two chains to one ID), and reports a pLDDT array whose largest value is at most 1 as a probable unconverted 0–1 scale; `validate(check_structure=True)` also reads the structure and checks that the pLDDT array has one value per atom and that the `chain_map` fits the file.

A model-specific scalar or dictionary goes in `extras` under the key the model uses (`bespoke_iptm` for OpenFold3), a per-token array in `TokenLayout.extras`, and an extra file of a sample in `PredictionFiles.extra`. Generic code never reads `extras`.

**Adapters.** An adapter subclasses `PredictionParser` (`binding_metrics.predictors.base`) and reads the output of one model without running it. It implements `find_files(prediction_dir, name, *, seed_index=1, sample=1) -> PredictionFiles` and `parse(files, *, name, seed_index=1, sample=1) -> PredictionRecord`; the base class supplies `load(prediction_dir, name, *, seed_index=1, sample=1, chain_map=None)`, which chains the two and applies the chain map, and `list_samples(prediction_dir, name)`, which returns `SampleRef(seed_index, sample, ranking_score)` for each sample in the model's natural order. The class attributes are `name`, `display_name`, `family` (`"af2"`: one token per residue; `"af3"`: one token per standard residue and one per heavy atom of a ligand or modified residue) and `capabilities` (None until an adapter declares which inputs its model can take). `sample=1` is the first output in the model's own order, not the best-ranked one. Parsing the scalars imports no biotite and does not open the structure file.

`binding_metrics.predictors.PARSERS` maps a model name to a `ParserSpec` that names the adapter class and imports it only when it is needed; `get_parser(name)` returns an adapter instance (`KeyError` listing the known names for an unknown model) and `register_parser(spec)` adds one. A contract test in `tests/predictors` runs every registered adapter against a synthetic complex; an adapter brings `tests/predictors/synth_<name>.py` with a `write_prediction` function that writes that complex in the model's layout, and needs no other test edit to be covered.

The `of3` adapter reads the layout above: `.cif`, `.cif.gz` or `.pdb` structures, JSON or NPZ confidences (NPZ without pickle), `seed_index` as the position in the numeric order of the seed values, `sample` counted from 1. `bespoke_iptm` is in `record.extras`, `chain_pair_iptm` keys are the strings OpenFold3 writes (`"(A, B)"`), and `record.tokens` is None because the files carry no token layout. Its layout was checked against the OpenFold3 v0.5.0 source, not against a 0.5.0 run.

---

## 13. EvoBind scoring

`compute_evobind_score(structure_path, plddt_per_atom, binder_chain, receptor_chain=None, receptor_interface_residues=None, interface_cutoff_angstrom=8.0, *, target_chain=None)`, `compute_evobind_adversarial_check(design_structure_path, afm_structure_path, binder_chain, receptor_chain=None, afm_plddt_per_atom=None, interface_cutoff_angstrom=8.0, max_resname_mismatch_fraction=0.5, *, target_chain=None)` — `binding_metrics.metrics.evobind`

The interface-distance and confidence losses of Bryant et al. (2025). Gly and residues without Cβ use Cα. When `binding-metrics-run` runs the `openfold` metric, both functions run on the OpenFold3 output and their keys are merged into the OpenFold3 result; no extra model call is needed. Registered as `evobind_score` and `evobind_adversarial`.

### primary score

| key | type | unit | description |
|-----|------|------|-------------|
| `if_dist_pep_to_rec` | float | Å | mean over binder Cβ atoms of the distance to the nearest receptor interface Cβ |
| `if_dist_rec_to_pep` | float | Å | mean over receptor interface Cβ atoms of the distance to the nearest binder Cβ |
| `if_dist_symmetric` | float | Å | mean of the two |
| `n_interface_receptor_residues` | int | — | receptor residues taken as the interface (Cβ within `interface_cutoff_angstrom` of a binder Cβ, or `receptor_interface_residues`) |
| `interface_fallback_used` | bool | — | no receptor residue was in contact, so the whole receptor stood in for the interface |
| `mean_plddt_binder` | float \| None | [0–100] | mean per-residue binder pLDDT |
| `evobind_score` | float \| None | Å | `if_dist_pep_to_rec / (mean_plddt / 100)`; lower is better |
| `reason` | str | — | pLDDT was given but its mean is zero or not finite |

### adversarial check

Compares a design pose with a second prediction, for example an OpenFold3 prediction of the same complex: the two structures are superposed on the receptor Cα atoms and the binder centres of mass are compared. Residues are paired by residue number, or by position when the numberings do not overlap; a `ValueError` is raised when more than `max_resname_mismatch_fraction` of the pairs have different residue names (histidine and cysteine protonation variants, `HIN` included, count as equal).

| key | type | unit | description |
|-----|------|------|-------------|
| `delta_com_angstrom` | float | Å | binder centre-of-mass displacement after receptor Cα superposition |
| `n_superposition_residues` | int | — | receptor residues matched by residue number (0 to 2 when position pairing was used) |
| `n_superposition_atoms` | int | — | receptor Cα atoms used for the superposition, whichever pairing applied |
| `receptor_pairing`, `binder_pairing` | str | — | `"residue_number"` or `"position"` |
| `receptor_resname_mismatch_fraction`, `binder_resname_mismatch_fraction` | float | — | fraction of paired residues with different names |
| `afm_if_dist_pep_to_rec`, `afm_if_dist_rec_to_pep`, `afm_mean_if_dist` | float | Å | interface distances in the second structure |
| `interface_fallback_used` | bool | — | as above |
| `afm_mean_plddt_binder` | float \| None | [0–100] | binder pLDDT in the second structure |
| `evobind_adversarial_score` | float \| None | — | `mean_if_dist × (100 / pLDDT) × ΔCOM`; lower = the second prediction agrees with the design pose |
| `reason` | str | — | pLDDT was given but its mean is zero or not finite |

A high pLDDT and a small interface distance with a large ΔCOM mean that the second prediction places the binder elsewhere on the receptor surface, which suggests that the design pose is not supported.

---

## 14. receptor quality

`compute_receptor_quality(path, receptor_chain=None, clash_cutoff=0.4, solvent_model="obc2", device="cuda", *, exclude_bonded=True, random_seed=1, target_chain=None)` — `binding_metrics.metrics.receptor_quality` (also importable from `binding_metrics.metrics`)

MolProbity-style quality of one receptor chain, independent of a binder. It accepts receptor-only files and complexes (other chains are ignored) and scores every model of a multi-model file. Without `receptor_chain` it takes the largest protein chain. The terms follow MolProbity (Chen et al. 2010; Williams et al. 2018) but are lighter approximations, so the composite score is indicative and does not compare with published MolProbity values. The energy term needs OpenMM.

| term | definition |
|------|------------|
| Ramachandran | the box regions of §6 |
| rotamers | χ1 only: an outlier deviates more than 40° from the nearest of g−, g+, trans |
| Cβ deviation | count of residues whose Cβ is more than 0.25 Å from its ideal position |
| backbone geometry | bonds and angles more than 4σ from the Engh & Huber (1991) values; inter-residue C–N bonds longer than 2.5 Å are skipped |
| clashscore | heavy-atom overlaps per 1000 heavy atoms: two atoms clash when their van der Waals radii overlap by at least `clash_cutoff` (0.4 Å; Word et al. 1999). Hydrogens are ignored, so it is usually higher than MolProbity's value. With `exclude_bonded=True` pairs within three covalent bonds (peptide bonds, disulfides, ring closures, lactams, staples) and N/O pairs at 2.5 Å or more (hydrogen bonds) are not scored |
| MolProbity score | `0.426·ln(1 + clashscore) + 0.33·ln(1 + max(0, Rama outlier % − 0.2)/0.2) + 0.25·ln(1 + max(0, rotamer outlier % − 2)/2) + 0.5`; lower is better; NaN when a term is undefined |
| B-factors | mean, max, min, std, and the number of residues above 60 Å² |
| energy | absolute ff14SB + implicit-solvent potential energy at the input geometry with hydrogens added; hydrogen placement and PDBFixer's rebuild are seeded, so the value is reproducible for one seed |

The result holds `receptor_chain`, `n_models`, a list `models` with one dict per model (terms as sub-dicts, each with a `reason` when it could not be computed) and `summary`, the mean over models with `best_model_index`, the model with the lowest MolProbity score. `summary` keys: `ramachandran_favoured_pct`, `ramachandran_outlier_pct`, `ramachandran_outlier_count`, `clashscore`, `rotamer_outlier_pct`, `rotamer_outlier_count`, `cb_deviation_count`, `bad_bonds_pct`, `bad_angles_pct`, `molprobity_score`, `energy_kJ_mol`, `energy_per_residue_kJ_mol`, `best_model_index`. A file without a protein chain returns `error` instead.

```bash
binding-metrics-receptor-quality --input receptor.pdb --output quality.csv    # or .json
```

---

## 15. pipeline results and provenance

`binding_metrics.cli.run.run_pipeline(...)` (command line: `binding-metrics-run`) returns one dict per structure; `binding-metrics-run` writes it as `<sample>_results.json`. Keys:

| key | content |
|-----|---------|
| `sample_id`, `input`, `total_elapsed_s` | identifiers and wall time |
| `provenance` | which code, seed and machine produced the result (below) |
| `chains` | result of `detect_chains_from_file`: chain IDs, label IDs, residue counts |
| `prep` | what preparation changed, or `{"skipped": True}` / `{"error": ...}` |
| `relax` | relaxation result (below), or `{"skipped": True}` |
| `energy` | result of §5 |
| `interface`, `geometry`, `electrostatics` | results of §2 to §8; `geometry` holds `ramachandran`, `omega` and `shape_complementarity` |
| `dockq`, `openfold` | results of §10 and §12 (with the EvoBind keys merged into `openfold`) |
| `nonfinite_fields` | paths of every NaN or infinite value in the file, for example `relax.rmsd_md_final` |

A metric that did not run is `{"skipped": True}`; one that failed is `{"error": message}`, and the command exits with 1 when any step failed.

**prep.** `output`, `ph`, `keep_water`, plus the report of preparation: `removed_heterogens` (list of `"NAME (chain X)"`), `n_removed_waters`, `kept_nonstandard` (non-standard residues and metal ions that were kept), `n_missing_atoms_rebuilt`, `n_missing_residue_gaps` and `chain_breaks`. Each entry of `chain_breaks` is `{"chain", "residue_before", "residue_after", "c_n_distance_angstrom"}`: two consecutive residues of a chain whose C and N atoms are more than 2.0 Å apart in the input. Prep logs a warning for each and leaves them as they are; the relaxation bonds the two residues by name and closes the gap. The chain IDs in `removed_heterogens`, `kept_nonstandard` and `chain_breaks` are those of the OpenMM topology, which for a CIF with more label IDs than author IDs are the label IDs (the peptide of 1CWA is author chain C and appears as chain B). For a cyclic peptide with residues that needed a GAFF template, `ncaa_bond_order_source` says where each residue's bond orders came from: `"ccd"` (Chemical Component Dictionary) or `"single_bonds"` (fallback, see [`nonstandard.md`](nonstandard.md)).

**relax.** `RelaxationResult.to_dict()`, plus `elapsed_s`:

| key | unit | description |
|-----|------|-------------|
| `success`, `error_message` | — | whether the run completed |
| `potential_energy_minimized` | kJ/mol | potential energy after minimisation |
| `potential_energy_md_avg`, `potential_energy_md_std` | kJ/mol | mean and standard deviation over the saved MD frames |
| `rmsd_md_final` | Å | Kabsch RMSD of the last MD frame against the minimised structure, over all atoms of the system (hydrogens included) |
| `peptide_rmsf_mean`, `peptide_rmsf_max` | Å | fluctuation of the binder Cα atoms about their mean position; frames are not superposed first, so overall drift is included |
| `receptor_rmsd_md_final`, `receptor_drift_mean` | Å | receptor Cα RMSD of the last frame, and its mean over all saved frames, against the minimised structure |
| `pep_rec_com_distance_delta` | Å | change of the binder–receptor Cα centre-of-mass distance, minimised to last frame; positive = separating |
| `minimized_structure_path`, `md_final_structure_path` | — | saved CIF files |
| `peptide_cyclic_bonds` | — | closure bonds detected in the binder |
| `dropped_protein_chains` | — | IDs of the protein chains, other than the binder and the target, that the relaxation removed (empty list when none) |
| `platform`, `precision`, `platform_fallback_reason` | — | OpenMM platform used (`CUDA` or `CPU`), its precision, and why CUDA was not used when it was requested |
| `qc_passed`, `qc_failed_checks`, `qc_checks` | — | structural QC (below) |
| `ncaa_bond_order_source` | — | as under `prep` |

The MD-based keys are `None` when `--md-duration-ps 0`.

**structural QC.** After minimisation, and again for the last MD frame, seven checks compare the result with the input structure:

1. `energy`: finite and between −1e8 and 1e6 kJ/mol; for the minimised structure also not above the energy before minimisation (1 kJ/mol tolerance). The MD frame is checked with the mean MD potential energy and against the range only
2. `rmsd`: heavy-atom RMSD to the input below 5 Å (minimised structure only)
3. `coordinates_finite`: no NaN or infinite coordinate
4. `min_heavy_distance`: no two heavy atoms of different residues closer than 0.8 Å (fused atoms; this is not a clash score)
5. `bond_lengths`: every heavy-atom bond of the topology (of the residue templates of the Chemical Component Dictionary when QC reads a file) stays within 0.5 to 2.5 Å in the relaxed structure; a bond that was already longer than 2.5 Å in the input is named in `detail` and does not fail the check
6. `chirality`: no Cα stereocentre changed sign
7. `composition`: no heavy atom added, dropped or renamed

The limits are wide on purpose: they separate a relaxed structure from an exploded one and do not measure quality. QC is advisory. `qc_passed` is True, False, or None when QC did not run; `qc_failed_checks` is a comma-separated string of the failed check names (empty when none failed; MD-frame checks are prefixed `md_final:`); `qc_checks` has one row per check with its value, limit and detail and appears in the JSON only. A failed check logs a warning line and changes neither `success` nor the exit code.

**provenance.**

```
{
    "schema_version": 1,          # bumped when a key changes meaning
    "package_version": "0.1.0",   # installed distribution version
    "git_sha": "...",             # HEAD of the source checkout, else None
    "python": "3.12.3",
    "os": "Linux-...",
    "openmm_version": "8.2",      # None when OpenMM is not importable
    "platform": "CUDA",           # platform the relaxation reported, else None
    "seed": 1,                    # random seed of the run; None = fresh randomness
}
```

`git_sha` is set only when the package is imported from its own git checkout. Collection is best effort and never raises; a field that cannot be determined is `None`. `binding-metrics-batch` writes the same block as `provenance_<key>` columns at the end of every CSV row. `binding_metrics.provenance.collect_provenance(seed, platform)` builds the block for your own result files.

**batch rows.** A row holds `sample_id`, `input`, the flattened results, `batch_status` and the `provenance_*` columns. `batch_status` is `ok` (every step completed), `partial` (the pipeline finished but a step failed; see `batch_failed_steps` and `batch_failed_reasons`) or `error` (the worker raised; see `batch_error`). `run_batch` returns the rows in the order of its input paths.

---

## 16. metric registry

`binding_metrics.metrics.registry` describes each metric function without importing it, so that a generic caller can drive them: `METRICS` lists the `MetricSpec` entries, `get_metric(name)` returns one and `metrics_by_input_type(input_type)` filters. The functions are the API; the registry is the adapter. `spec.call(**kwargs)` imports the function on first use and calls it with exactly the given keywords.

fields of a `MetricSpec`:

| field | meaning |
|-------|---------|
| `name`, `import_path`, `description` | identifier, `"module:function"`, one line |
| `input_type` | `static_structure` (one PDB or CIF file), `trajectory` (trajectory and topology), `md_simulation` (one structure, runs a relaxation or MD protocol), `openfold_json` (OpenFold3 output), `atom_array` (a loaded `biotite.structure.AtomArray`), `predicted_structure` (a predicted structure path plus per-atom confidence arrays) |
| `chain_mode` | `none`, `single`, `interface` or `interface_2paths` |
| `formats`, `path_arg`, `secondary_path_arg` | accepted file formats, and the keyword that receives the primary and the second input |
| `chain_arg`, `peptide_chain_arg`, `receptor_chain_arg` | keywords that receive the chains |
| `binder_chain_arg`, `target_chain_arg` | keywords of the role aliases, on functions that accept them |
| `headline_key` | result key that `direction` and `unit` describe; a dotted path (`summary.molprobity_score`) reaches into a nested dict |
| `direction` | `higher_is_better`, `lower_is_better` or None |
| `unit` | one of `kJ/mol`, `kcal/mol`, `angstrom`, `angstrom^2`, `angstrom^3`, `nm`, `nm^2`, `degree`, `percent`, `fraction`, `count`, `dimensionless`, or None |
| `cost_class` | `static` (one structure, geometry only, seconds), `structural` (builds a force-field system), `md` (needs or runs a molecular-dynamics trajectory), `model` (needs a structure-prediction run) |
| `requires_extras` | extras of `pip install binding-metrics[...]` that the default use needs |
| `requires_gpu` | the heavy computation runs on CUDA by default |

`direction`, `unit`, `headline_key`, `cost_class` and `requires_extras` are optional; None or an empty tuple means "not declared", never a guess. A consumer skips an input type it does not know: the registry grows by adding entries, input types and optional fields. The pipeline's `--metrics` names (`interface`, `geometry`, `electrostatics`, `energy`, `openfold`, `dockq`) are derived from it; the `atom_array` and `predicted_structure` metrics need an in-memory object and are not pipeline steps.

| name | function | input type | headline key | direction | unit | cost class | GPU |
|------|----------|------------|--------------|-----------|------|------------|-----|
| `interface` | `compute_interface_metrics` | static_structure | — | — | — | static |  |
| `coulomb` | `compute_coulomb_cross_chain` | static_structure | `coulomb_energy_kJ` | lower | kJ/mol | static |  |
| `ramachandran` | `compute_ramachandran` | static_structure | `ramachandran_favoured_pct` | higher | percent | static |  |
| `omega` | `compute_omega_planarity` | static_structure | `omega_outlier_fraction` | lower | fraction | static |  |
| `shape_complementarity` | `compute_shape_complementarity` | static_structure | `sc` | higher | dimensionless | static |  |
| `void_volume` | `compute_buried_void_volume` | static_structure | `void_volume_A3` | lower | angstrom^3 | static |  |
| `structure_rmsd` | `compute_structure_rmsd` | static_structure | `rmsd` | lower | angstrom | static |  |
| `delta_sasa_static` | `compute_delta_sasa_static` | static_structure | `delta_sasa` | higher | angstrom^2 | static |  |
| `receptor_quality` | `compute_receptor_quality` | static_structure | `summary.molprobity_score` | lower | — | structural | yes |
| `evobind_adversarial` | `compute_evobind_adversarial_check` | static_structure | `evobind_adversarial_score` | lower | — | model |  |
| `dockq` | `compute_dockq_metrics` | static_structure | `dockq` | higher | dimensionless | static |  |
| `hbonds` | `compute_hbonds` | atom_array | `hbond_energy` | lower | kcal/mol | static |  |
| `saltbridges` | `compute_saltbridges` | atom_array | `saltbridge_energy` | lower | kcal/mol | static |  |
| `evobind_score` | `compute_evobind_score` | predicted_structure | `evobind_score` | lower | angstrom | model |  |
| `interaction_energy` | `calculate_interaction_energy` | trajectory | — | lower | kJ/mol | md |  |
| `component_energies` | `calculate_component_energies` | trajectory | `total` | lower | kJ/mol | md |  |
| `rmsd` | `calculate_rmsd` | trajectory | — | lower | nm | md |  |
| `rmsf` | `calculate_rmsf` | trajectory | — | lower | nm | md |  |
| `ligand_rmsd` | `calculate_ligand_rmsd` | trajectory | `ligand_rmsd` | lower | nm | md |  |
| `receptor_drift` | `compute_receptor_drift` | trajectory | `drift_aligned_mean` | lower | angstrom | md |  |
| `buried_sasa` | `calculate_buried_sasa` | trajectory | — | higher | nm^2 | md |  |
| `contacts` | `calculate_contacts` | trajectory | — | — | count | md |  |
| `interface_sasa` | `calculate_interface_sasa` | trajectory | `buried` | higher | nm^2 | md |  |
| `contact_residues` | `calculate_contact_residues` | trajectory | — | — | — | md |  |
| `md_implicit` | `run_implicit_relaxation` | md_simulation | — | — | — | md | yes |
| `structure_interaction_energy` | `compute_interaction_energy` | md_simulation | `relaxed_interaction_energy` | lower | kJ/mol | md | yes |
| `openfold` | `compute_openfold_metrics` | openfold_json | — | — | — | model |  |
| `interface_pae` | `compute_interface_pae` | openfold_json | `mean_interface_pae` | lower | angstrom | model |  |

---

## 17. scorecard thresholds

The `--summary` flag produces a RAG (🟢/🟡/🔴/⬜) scorecard. The thresholds are in `src/binding_metrics/protocols/report.py`; the rationale and the caveats are in [`report_thresholds.md`](report_thresholds.md).

| metric | 🟢 | 🟡 | 🔴 |
|--------|----|----|-----|
| MD RMSD (Å) | < 2 | 2 to 5 | > 5 |
| RMSF mean (Å) | < 1 | 1 to 2 | > 2 |
| E_int (kJ/mol) | < −40 | −40 to 0 | > 0 |
| ΔSASA (Å²) | > 1000 | 500 to 1000 | < 500 |
| H-bonds | ≥ 5 | 2 to 4 | < 2 |
| H-bond energy (kcal/mol) | ≤ −10 | −10 to −2 | > −2 |
| salt bridges | ≥ 2 | 1 | 0 |
| salt-bridge energy (kcal/mol) | ≤ −40 | −40 to −10 | > −10 |
| Ramachandran favoured (%) | > 95 | 80 to 95 | < 80 |
| ω outlier fraction | < 0.05 | 0.05 to 0.20 | > 0.20 |
| Sc | > 0.7 | 0.5 to 0.7 | < 0.5 |
| Coulomb (kJ/mol) | < −100 | −100 to 0 | > 0 |
| DockQ | ≥ 0.80 | 0.23 to 0.80 | < 0.23 |

A missing or non-finite value is shown as ⬜ (N/A) and not rated.

---

## 18. score versus feature

Every value a metric returns is a score or a feature.

| type | definition | examples |
|------|------------|----------|
| **score** | has a direction: higher or lower is clearly better | `delta_sasa`, `sc`, `ramachandran_outlier_count`, `coulomb_energy_kJ`, `iptm` |
| **feature** | a descriptor without an intrinsic quality direction, for analysis, clustering or as model input | `fraction_polar`, `per_residue`, `plddt_per_atom`, `pbc_detected`, `n_charged_pairs` |

The registry's `headline_key` and `direction` name the score of a metric that has a single one. When building composite scorers or learned models, use scores as labels or objectives and features as inputs or auxiliary descriptors.

---

## 19. I/O utilities

`binding_metrics.io.structures`. The loaders below use OpenMM; `detect_chains_from_file` needs only biotite.

- `load_structure(path) → (topology, positions)`: reads a `.pdb`, `.cif` or `.mmcif` file into an OpenMM topology and positions.
- `detect_chains(topology) → (peptide_chain, receptor_chain)`: smallest and largest protein chain of an OpenMM topology, by the number of amino-acid residues (the residue set is listed under chain auto-detection in [§1](#1-conventions)); a single chain gives `(chain, None)`.
- `detect_chains_from_file(path, peptide_chain=None, receptor_chain=None, verbose=True) → dict`: biotite-based detection and the function the pipeline uses. For two protein chains the smaller is the peptide and the larger the receptor. For more than two, the receptor is the chain with the most Cα atoms within 8 Å of the peptide, not the largest. Explicit chain IDs are returned as given, even when they are not in the file (the pipeline checks them and raises `ChainNotFoundError`).
- `save_cif(topology, positions, output_path, source_cif_path=None)`: writes a CIF and, given a source CIF, restores the author chain IDs and residue numbers. Without gemmi it logs a warning and keeps OpenMM's sequential IDs. Without a source CIF it writes the topology's own chain IDs and residue numbers when the chain IDs are unique and alphanumeric and the residue numbers are integers, so relaxing a PDB input writes the chain IDs of the input; otherwise it uses OpenMM's letters and sequential numbers.
- `drop_other_protein_chains(topology, positions, peptide_chain, receptor_chain, report=None)`: deletes every protein chain other than the two named, after `strip_heterogens`; nothing is removed unless both chains are given and present. `report["dropped_protein_chains"]` receives the removed IDs.

`detect_chains_from_file` returns:

| key | description |
|-----|-------------|
| `peptide_chain`, `receptor_chain` | author chain IDs (`auth_asym_id` in a CIF); `receptor_chain` is None for a file with one protein chain |
| `peptide_chain_label`, `receptor_chain_label` | the IDs as OpenMM names the chains, used by the OpenMM-based steps: the label ID (`label_asym_id`) when the file has more label IDs than author IDs (waters and ligands each get a label ID), the author ID otherwise; equal to the author IDs for PDB files |
| `peptide_n_residues`, `receptor_n_residues` | residue counts |
| `all_chains` | every protein chain as `{"id", "n_residues"}`, smallest first |

---

## 20. unit summary

| quantity | unit |
|----------|------|
| SASA, distances, RMSD of static structures | Å, Å² |
| void volume | Å³ |
| `delta_g_int`, `delta_g_res` | **kcal/mol** |
| `hbond_energy`, `saltbridge_energy` | **kcal/mol** |
| `coulomb_energy_kcal` | kcal/mol (convenience alias of `coulomb_energy_kJ`) |
| all other energies (OpenMM, Coulomb, receptor energy) | **kJ/mol** |
| angles (φ, ψ, ω) | degrees |
| pLDDT | [0–100] |
| pTM, ipTM | [0–1] |
| PAE, PDE, gPDE | Å |
| trajectory metrics (RMSD, RMSF, ligand RMSD, SASA, contacts) | nm, nm² (MDTraj units) |
| receptor drift | Å |

---

## 21. references

- Eisenberg & McLachlan (1986). *Nature* 319, 199–203.
- Krissinel & Henrick (2007). *J. Mol. Biol.* 372, 774–797.
- Shrake & Rupley (1973). *J. Mol. Biol.* 79, 351–371.
- Baker & Hubbard (1984). *Prog. Biophys. Mol. Biol.* 44, 97–179.
- Barlow & Thornton (1983). *J. Mol. Biol.* 168, 867–885.
- Lawrence & Colman (1993). *J. Mol. Biol.* 234, 946–950.
- Lee & Richards (1971). *J. Mol. Biol.* 55, 379–400.
- Richards (1977). *Annu. Rev. Biophys. Bioeng.* 6, 151–176.
- Connolly (1983). *Science* 221, 709–713.
- Kleywegt & Jones (1994). *Acta Cryst.* D50, 178–185.
- Maier et al. (2015). *J. Chem. Theory Comput.* 11, 3696–3713.
- Onufriev, Bashford & Case (2004). *Proteins* 55, 383–394.
- Nguyen, Roe & Simmerling (2013). *J. Chem. Theory Comput.* 9, 2020–2034.
- Kollman et al. (2000). *Acc. Chem. Res.* 33, 889–897.
- Eastman et al. (2017). OpenMM 7. *PLOS Comput. Biol.* 13, e1005659.
- Kabsch (1976). *Acta Cryst.* A32, 922.
- Mendez et al. (2003). *Proteins* 52, 51.
- Basu & Wallner (2016). DockQ. *PLoS ONE* 11, e0161879.
- Chen et al. (2010). MolProbity. *Acta Cryst.* D66, 12–21.
- Williams et al. (2018). MolProbity. *Protein Sci.* 27, 293–315.
- Lovell et al. (2003). *Proteins* 50, 437.
- Word et al. (1999). *J. Mol. Biol.* 285, 1735.
- Engh & Huber (1991). *Acta Cryst.* A47, 392–400.
- The OpenFold3 Team (2026). OpenFold3, v0.5.0 (OpenBind-0 weights). https://github.com/aqlaboratory/openfold-3, doi:10.5281/zenodo.22042719 (doi:10.5281/zenodo.17485509 stands for all versions; cite the release that produced the results).
- OpenBind Consortium (2026). OpenBind-0 announcement, 21 August 2026. https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/
- Abramson et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500.
- Bryant et al. (2025). EvoBind. *Communications Chemistry*. https://doi.org/10.1038/s42004-025-01601-3
- Ryczko et al. (2026). Machine-learning force-field scoring rivals free-energy perturbation for congeneric ligand ranking across public benchmarks. ChemRxiv preprint, doi:10.26434/chemrxiv.15008810 (v2, 20 Sep 2026).
