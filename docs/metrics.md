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
| `static` | interface, H-bonds, salt bridges, Coulomb, Ramachandran, omega, shape complementarity, void volume, static ΔSASA, structure comparison, EvoBind, parsing of prediction output (§12) |
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

The chain IDs (`peptide_chain`, `receptor_chain`, and the `binding-metrics-energy` flags) are author IDs for an mmCIF, as for every other metric. OpenMM names the chains of a CIF with more label IDs than author IDs by their label IDs, so the peptide of 1CWA is author chain C and chain B of the topology, and the function finds it from either ID. 1CWA, raw file, `peptide_chain="C"`: 2288 contacts and −260.4 kJ/mol, the values of `"B"`.

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

Parses the output of an OpenFold3 run. It needs no OpenFold3 install, only the output files. `seed` is the 1-based position of a `seed_*` directory of the query in the numeric order of the seed values (`seed_9` before `seed_10`), not a random seed value (the directories are named after the seeds OpenFold3 sampled with: the `seeds` of the run, 42 by default; `seed_value` of the result is that number; with `num_model_seeds` OpenFold3 generates the seed and `experiment_config.json` does not record it (`seeds: [42]`, `num_seeds: null` next to the directory `seed_2746317213`), so the directory name is the only source); `seed_index` is a clearer name for the same argument and takes precedence. The function reads the output through the `of3` adapter (`get_parser("of3")`, see "Prediction records" below) and analyses the record with `summarize_prediction`.

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
| `seed_value` | int \| None | — | the seed in the name of the `seed_<value>` directory that was parsed, which is the seed OpenFold3 sampled with; None when no file was found or the name is not a number. It is the last key before `reason`, so the order of the others is unchanged |
| `structure_path` | str \| None | — | path to the predicted structure |
| `avg_plddt` | float | [0–100] | mean per-atom pLDDT |
| `gpde` | float | Å | global predicted distance error |
| `ptm` | float | [0–1] | predicted TM-score; NaN when the aggregated file lacks it |
| `iptm` | float | [0–1] | interface pTM; NaN for a single chain or when the aggregated file lacks it |
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
| `binder_ca_rmsd` | float | Å | binder Cα RMSD against `reference_structure_path` (the refolding RMSD of a `refold` run; in a `score` run, how far OpenFold3's own pose is from the reference pose): in the receptor frame when `receptor_chain` is given, otherwise after superposing the binder Cα; NaN with a `reason` when the receptor Cα counts differ or are fewer than 3 |
| `reason` | str | — | only when a value could not be computed; names each affected analysis |

**interface PAE.** PAE at token (i, j) is the expected position error of token j when the prediction is aligned on token i (the row is the alignment frame, the column the scored token). The interface block is the binder × target slice of the PAE matrix, that is the error of the target tokens when the structure is aligned on the binder. `mean_interface_pae` averages the binder→target and target→binder blocks, so it does not depend on the slice direction, and `max_interface_pae` is the larger of the two maxima. `compute_interface_pae(confidences_path, structure_path, binder_chain, receptor_chain=None, *, target_chain=None)` returns the slice and its statistics (`pae_interface`, `mean_interface_pae`, `max_interface_pae`, `n_binder_tokens`, `n_receptor_tokens`) from a confidences file and the predicted structure; it is registered as `interface_pae` and raises `ValueError` when the file has no PAE matrix or when the matrix fits neither the token layout of the structure nor its residue count. It cuts the blocks with the token layout that `OpenFold3Parser.complete` builds from the structure (one token per residue of the standard set, one per heavy atom of any other residue), like `compute_openfold_metrics`, so a binder with modified residues gives its value: the matrix of 1CWA is 240 x 240 for 176 residues, and on the real output of a refold run (seed 42) the function went from `ValueError: PAE matrix has 240 tokens but the structure has 176 residues` to `mean_interface_pae` 8.76 A (75 binder tokens by 165 receptor tokens). OpenFold3 0.4.1 and later always compute PAE, pTM and ipTM; the `pae` array is in the full confidences file, which is missing when the run used `write_full_confidence_scores: false`, and `reason` says so.

The interface blocks are located with one token per residue. OpenFold3 uses one token per standard residue but one per heavy atom for ligands and modified residues, so a prediction that holds such components makes the matrix larger than the residue count (1CWA: 240 x 240 for 176 residues). For OpenFold3 the adapter builds the token layout from the structure (`OpenFold3Parser.complete`, called by `compute_openfold_metrics`, `compute_prediction_metrics` and `PredictionSession.record`), so a binder with modified residues gets its interface values. They stay NaN, with a warning and a `reason`, only when the layout cannot be shown to fit the files (see "Model adapters", OpenFold3) and for a record that only went through `load`; the earlier code sliced the wrong block silently.

Higher pLDDT, pTM and ipTM and lower PDE and PAE are better. The Markdown summary lists binder residues with a pLDDT below 70. The package sets no other limit for OpenFold3 scores, and none has been calibrated here.

### `run_openfold(query_json, output_dir, inference_ckpt_path=None, num_diffusion_samples=5, num_model_seeds=None, use_msa_server=True, model_presets=None, runner_yaml=None, extra_args=None, conda_env=None, template_dir=None, seeds=None) → Path`

Runs `run_openfold predict` as a subprocess, in the current environment or through `conda run -n <conda_env>`, and returns the output directory. `run_openfold_scoring` (each chain templated on its own structure; OpenFold3 places the binder itself) and `run_openfold_refolding` (receptor templated, binder predicted from its sequence) prepare the query and call it; both take `seeds` (default `(42,)`), the seed values OpenFold3 samples with, and `binder_cyclic` (see "cyclic binder" below). The seeds go to `experiment_settings.seeds` of the runner YAML, because OpenFold3 0.5.0 does not read a `seeds` field of the query JSON; the `seeds` argument of the `prepare_*` functions is accepted and ignored (a value other than `(42,)` raises a `DeprecationWarning`). Seen in real runs of 0.5.0 (1YCR): the default writes `seed_42/`, `seeds=(7, 11)` writes `seed_7/` and `seed_11/` (the runner YAML holds `seeds: [7, 11]` and the command has no `--num_model_seeds`), and `num_model_seeds=1` writes `seed_2746317213/`, as documented here. The samples of different seeds differ: with a template (1YCR, `score`, server off) four seeds gave ipTM 0.852, 0.861, 0.839 and 0.858 and binder Cα RMSD against the input 1.12, 1.53, 0.58 and 1.77 A (seeds 42, 7, 11 and 2746317213), while a repeat of seed 42 agreed to 0.01; without a template and without an MSA the seeds differ much more (ipTM 0.17 to 0.63, binder 21 to 24 A, three seeds).

**the two modes.** In `score` mode the query gives each chain its own structure from the complex as a template (an A3M self-alignment and a single-chain template CIF per chain): the binder its conformation, the receptor its conformation. A template carries the fold of the chain it is made from and no inter-chain geometry: each query chain gets its own template structures (the template CIFs here hold one chain each), so no cross-chain geometry is ever supplied, and `_embed_feats` of the template embedder (`openfold3/core/model/feature_embedders/template_embedders.py` at v0.5.0) applies the same-chain mask (`asym_id[i] == asym_id[j]`) to the validity indicators of the template pair features, so cross-chain pairs are marked invalid (the distogram and unit-vector tensors are not multiplied by that mask in that function). OpenFold3 therefore places the binder against the receptor itself. Its confidences (pLDDT, pTM, ipTM, PAE) describe that pose, not the pose of the input. How far the predicted pose is from the input pose is a separate result, given in both modes by `binder_ca_rmsd` (binder Cα RMSD against the input in the receptor frame, which `binding-metrics-run` and `-batch` measure with the input as the reference) and by `delta_com_angstrom` of the EvoBind adversarial check (the displacement of the binder centre of mass between the input and the prediction after superposing the receptor Cα atoms). In `score` mode the binder also has its own fold as a template, so `binder_ca_rmsd` is less free of the input than in `refold` mode, where only the receptor is templated and the binder is predicted from its sequence, so OpenFold3 predicts the binder conformation and its pose; there `binder_ca_rmsd` is the refolding RMSD. The summary labels the row "Refolding RMSD" in both modes. The statements about the template embedder come from reading the OpenFold3 source, not from a run.

**the MSA server and the templates (issue #68, measured on one complex).** With the ColabFold MSA server on, the default, OpenFold3 0.5.0 overwrites the `template_alignment_file_path` of every chain with the alignment that the server returns (a `UserWarning` of `colabfold_msa_server.py` on stderr). Measured on 1YCR (`score`, OpenFold3 0.5.0, one seed, one sample): the server returned 135 template hits for the receptor (1rv1_C, 6t2e_A, ...) and 70 for the 13-residue binder (3dac_B, 1ycr_B, 2mwy_B, ...), and because the toolkit runs OpenFold3 with `fetch_missing_structures: false` and those structures are not in the template directory, OpenFold3 ended with `template_entry_chain_ids: []` for both chains: **the default `score` run has no template at all**. It is a template-free prediction with the keys of a scored one, and the server's MSA alone brings the complex to a binder Cα RMSD against the input of 1.62 A (ipTM 0.847), where the server off with the template read gives 1.12 A (ipTM 0.852; receptor 0.51 A). Not tried: `fetch_missing_structures: true`, with which the hits for 1YCR would include 1YCR itself. The toolkit now says what happened to each template (see "what became of the templates" below): a warning names the chain and the cause, and `results[...]["templates"]` records it. Three ways keep the template: switch the server off (`use_msa_server=False` of the `run_openfold*` functions, `--no-msa-server` of `binding-metrics-openfold`, `--openfold-no-msa-server` of `binding-metrics-run` and `-batch`, `openfold_use_msa_server=False` of `run_pipeline` and `run_batch`, for both the `openfold` step and `--predictor of3`; the run then has a dummy MSA, which lowers accuracy for a natural receptor); give the template as a structure (`template_mode="structure"`, `--openfold-templates structure`: the server does not overwrite `template_cif_paths`, and with the server on both chains kept their template, binder 1.57 A, ipTM 0.865); or both. The server setting and the template mode are part of the request key of the prediction store, so each gets its own entry.

| argument | default | description |
|----------|---------|-------------|
| `num_diffusion_samples` | 5 | structures sampled per query |
| `num_model_seeds` | None | not passed; a number is passed as `--num_model_seeds N`, which makes OpenFold3 generate N seeds (from `random.seed(42)`) and use them in place of the seeds of the runner YAML, so it cannot be combined with `seeds` (`ValueError`, also when the option is in `extra_args`) |
| `seeds` | None | seed values, one set of `num_diffusion_samples` structures each, in `seed_<value>` directories. None writes `[42]` to the generated YAML, keeps the seeds of a `runner_yaml`, and writes none when `num_model_seeds` is given. Explicit seeds replace those of a `runner_yaml` in the copy that OpenFold3 reads (below) |
| `use_msa_server` | True | ColabFold MSA server; the sequences leave the machine, and results can change over time because the alignments come from a remote service |
| `model_presets` | `["predict", "low_mem"]` | presets written to a runner YAML; `predict` is always included |
| `runner_yaml` | None | explicit YAML; overrides `model_presets`. With templates (`template_dir`, which the scoring, refolding and batched wrappers always set) or `seeds`, OpenFold3 gets a copy of it, `<output_dir>/runner_config_merged.yaml`, that adds `template_preprocessor_settings.structure_directory` and `experiment_settings.seeds`; the file itself is not modified |
| `template_dir` | None | folder with the template CIFs that the query's A3M files point to; written to the runner YAML as `template_preprocessor_settings.structure_directory` |

presets: `predict` is the required base preset and `low_mem` computes the pairformer embeddings sequentially, which suits large complexes or limited GPU memory. There is no preset for the PAE head: OpenFold3 0.4.1 removed `pae_enabled`, and pTM, ipTM and PAE are always written. A `pae_enabled` entry in `model_presets` (or `--presets`) is left out of the runner YAML with a `DeprecationWarning`, and the run continues. It is kept when the OpenFold3 that will run is older than 0.4.0, where PAE is off without it: the installation in `conda_env` when one is named (asked through `conda run -n <conda_env> python`), the current interpreter otherwise; a version that cannot be read counts as a current one.

**failures.** stderr of `run_openfold` is shown as before and its last lines are kept. A non-zero exit raises `OpenFoldRunError` (a `subprocess.CalledProcessError`); its message starts with the line that states the reason, which for a default checkpoint that is not on disk is `Value error, Default checkpoint ... cowardly refusing to perform inference` and not the pydantic `1 validation error` line above it (otherwise the last exception line), and adds the fix for missing or incompatible weights, GPU memory and `/dev/shm`. OpenFold3 also exits with status 0 when a query fails inside it, so the run's `summary.txt` and `logs/predict_err_rank<N>.log` are read: a run in which every query failed raises `OpenFoldQueryError`, and a partial failure is logged with the reason of each query. `predict_err_rank<N>.log` exists only for a failure in the model's forward pass. A failure while the features of a query are built, before the model runs (a template CIF that OpenFold3 cannot read, for one), is only a warning on stderr, `Failed to process <query> with preferredException type: ...` with a traceback, and the log directory is removed as empty; `run_openfold` reads that warning while the stream goes by and puts its exception line in the reason (`OpenFold3 failed while it built the features of this query: ValueError: invalid literal for int() with base 10: np.str_('.')` for the template CIF that earlier versions wrote). No reason names the work directory of the run (the store renames it): a log is named `logs/predict_err_rank0.log in the output of the run`. A model that cannot be started here records nothing; the error names what was looked for (`OpenFold3Runner.unavailable_reason()`: the conda environment `no_such_env`, `conda is not on PATH`, or `run_openfold is not on PATH`): after `Why:` in the `PredictionUnavailableError` of the store, so a caller of `PredictionSession` gets it too, and in the error text of the pipeline, once. A `predict` request whose name differs from the key of the query in the query file fails with `OpenFold3 wrote no output for query '<name>'` and now names the keys that have output in the folder (OpenFold3 names the output after the key of the query file). Parsing the output of a query that failed gives NaN values and the same reason in `reason`.

**inputs stay on disk.** With templates and the MSA server (both on by default) OpenFold3 removes the parent of `template_preprocessor_settings.structure_directory` when a run ends. `run_openfold_scoring`, `run_openfold_refolding` and `run_openfold_batched` put the template CIFs in `<output_dir>/query/templates`, so the runner YAML they write sets `msa_computation_settings.cleanup_msa_dir: false`, and `<output_dir>/query` (query JSON, A3M files, template CIFs) survives. The setting is not written when no `template_dir` is given. A `runner_yaml` you supply is copied to `<output_dir>/runner_config_merged.yaml` with `structure_directory` and, unless your file sets it, `cleanup_msa_dir: false` added, so the templates are found and the query folder survives; your file is not modified, comments are not carried into the copy, and without PyYAML the settings are appended as text (a file that already sets `template_preprocessor_settings`, `msa_computation_settings` or, with `seeds`, `experiment_settings` then raises `ValueError`). If your file sets another `structure_directory`, it is kept and a warning names both directories: OpenFold3 finds the templates of the run only if that directory holds them.

**template files.** Each templated chain gets `templates/<entry>.cif` (`receptor`, `binder`; `<sample>rec` and `<sample>bnd` in a batch) and an A3M self-alignment that points to it. OpenFold3 0.5.0 reads the CIF with the parsers it uses for PDB entries, so the file has what those need: one polymer entity with the integer id 1 (`_entity`, `_entity_poly` with `pdbx_seq_one_letter_code_can`, `_entity_poly_seq`, `_struct_asym`, `_pdbx_poly_seq_scheme`), the chain under its own ID, `label_seq_id` counting the residues 1 to N (the A3M indexes them that way; the author numbers would not match), a `_chem_comp.type` for every component and a release date of 1900-01-01. Only the residues of the query sequence are written, so waters, ligands and terminal caps of the chain are not, and a protonation variant or a lactam template is named after its parent residue (`HID` as `HIS`, `NMG` as `SAR`). A template file whose chain has another number of residues than the query sequence is refused with a `ValueError` that names both counts. Before this was fixed, OpenFold3 rejected the file (`label_entity_id` was `.`) and every query of `score` mode and of the receptor of `refold` mode failed with exit status 0 when the MSA server was off; with the server on the file was never read (see the limitation above). OpenFold3 keeps the result of its template preprocessing in `$TMPDIR/of3-of-<user>/template_data/template_cache`, keyed on the sequence of the chain and the content of its A3M file only, so a cache entry made from an older template CIF would be used for a new one. The query row of the A3M (`>query-b3_A/1-85`) therefore carries the version of the query builders, `QUERY_BUILDER_VERSION` (`binding_metrics.metrics.openfold`, 3), which changes the content and with it the cache key whenever the builders change. Entries that older versions of this toolkit left in that cache are not reached any more and can be deleted. The same version is in the request key of `OpenFold3Runner` (`options["query_builder_version"]`; None for `predict`, whose query file is the caller's), so a prediction that the store kept from an older builder is not reused for a request that the newer builder writes differently: the key of every `score` and `refold` request of OpenFold3 changes once, with this version, and the store runs the model again for them. The version is raised when the template CIF, the query JSON or the MSA files change in a way that can change a prediction: 1 before the version existed (a template CIF that OpenFold3 0.5.0 rejected), 2 for the repaired CIF, 3 for the cyclic rule, `template_mode` and the dummy MSA.

**MSA-free runs (`use_msa_server=False`, `--openfold-no-msa-server`).** The query gives each protein chain a dummy MSA that holds only its sequence: `main_msa_file_paths` points to `<query dir>/msas/<query name>_<chain>/colabfold_main.a3m` (`>query_<chain>` and the sequence; `dummy_msa=True` of the `prepare_*` functions, which `run_openfold_*` set when the server is off, `dummy_msa=False` leaves the MSA input out). OpenFold3's input reference (`input_format_reference.md`, `use_msas`) suggests MSA-free inference through such a dummy MSA and discourages leaving the MSA input out; OpenFold3 0.5.0 builds the same dummy itself for a chain without MSA files and warns ("Expected MSA file for chain ... A dummy MSA with only the query sequence will be used"), so the prediction does not change: three seeds of one complex without a template gave, with against without the file, ipTM 0.171 / 0.220 / 0.514 against 0.171 / 0.221 / 0.483 and binder Cα RMSD against the crystal pose 22.06 / 24.10 / 21.12 A against 22.09 / 24.18 / 21.11 A (seeds 11, 42, 7); with a template (1YCR, `score`, seed 42) pLDDT 91.39, ipTM 0.852 and binder 1.12 A with the file, 91.39, 0.853 and 1.12 A without. The file is named `colabfold_main.a3m` because OpenFold3 parses an MSA file only when its name is a key of `MSASettings.max_seq_counts` (`uniref90_hits`, ..., `colabfold_main`) and takes the folder name as the id of the chain's alignment: a query-only file with another name (`my_msa.a3m`) is skipped and the query fails while its features are built (`IndexError: list index out of range`, seen in a real run). With the server on the dummy is not written, because OpenFold3 replaces `main_msa_file_paths` with the server's MSA. What an MSA-free run is worth, binder Cα RMSD against the crystal pose in one complex (1YCR, OpenFold3 0.5.0, one seed): 1.6 A with the server's MSA and no template (the default settings lose the template, see below), 21.6 A with no MSA and no template (21 to 24 A over three seeds, ipTM 0.17 to 0.63), 1.1 A with a working template and no MSA. A natural receptor without an MSA is predicted worse; the numbers are those of one complex whose structure the weights have very probably seen.

**template mode (`template_mode`, `--openfold-templates {alignment,structure}`).** `alignment` (the default) gives each templated chain an A3M self-alignment that points to the template CIF (`template_alignment_file_path`). `structure` gives the CIF itself (`template_cif_paths` and `template_cif_chain_ids` of the chain), OpenFold3's CIF Direct Template Mode: no A3M file is written, OpenFold3 aligns the chains of the file to the query itself with Kalign and keeps the best-matching chain of each file (protein chains only; the files here hold one chain each). The two keys are mutually exclusive in OpenFold3, and the ColabFold MSA-server step overwrites the first and leaves the second alone (`colabfold_msa_server.py`: "already has template_cif_paths set ... not overwritten"), so a `structure` template survives the default server settings. `template_mode` is an argument of `prepare_refolding_query`, `prepare_scoring_query`, the two batched `prepare_*` functions, `run_openfold_scoring`, `run_openfold_refolding`, `run_openfold_batched` (keyword-only, the query builders get it only when it is not the default), `OpenFold3Runner.make_request` (part of the request key, `options["template_mode"]`; None for `predict`, whose query file names its own templates) and `run_pipeline` / `run_batch` (`openfold_templates`); the commands have `--openfold-templates` (also on the `prepare-query`, `prepare-scoring-query`, `refold` and `score` subcommands of `binding-metrics-openfold`). It is OpenFold3's setting: there is no `--prediction-templates`, and `--openfold-templates structure` with another `--predictor` is a usage error. A CIF-direct entry is keyed in OpenFold3's template cache by the content of the CIF and the chain it is paired with, so it needs no builder-version marker. One complex, one seed, one sample (1YCR, `score`, OpenFold3 0.5.0): with the server on, `alignment` has no template (binder Cα RMSD against the input 1.62 A, ipTM 0.847) and `structure` has both (1.57 A, ipTM 0.865); with the server off both have both templates and give the same prediction (1.12 A, ipTM 0.852; receptor 0.51 A). 1CWA (server off, so `alignment` is not overwritten): in `score` both chains have their template in both modes, and `structure` gives ipTM 0.970 and binder 0.40 A, the same as `alignment` (0.970, 0.40 A); in `refold` only the receptor has one (the binder is free) in both modes, ipTM 0.9215 with `structure` and 0.9214 with `alignment`, binder 0.70 A with both; with no template and no MSA (`predict`, the query written by hand) the same complex gives ipTM 0.54, binder 12.2 A and receptor 16.3 A from the input. The default is `alignment` and has not been changed; what the server's MSA and a template each add on other complexes is not measured.

**what became of the templates.** OpenFold3 0.5.0 goes on without a template when it cannot use one, exits with status 0 and reports "Successful Queries", so a `score` result can be a template-free prediction with the keys of a scored one. After each run the toolkit reads `<predictions>/inference_query_set.json`, which OpenFold3 rewrites after the template preprocessing (`template_entry_chain_ids` lists the entries a chain kept, `[]` when none, `null` when the chain declared no source), and the two messages that explain a loss: the `UserWarning` of `colabfold_msa_server.py` about an alignment that the server overwrote (stderr) and `Failed to preprocess template alignment ...` (a `print`, so stdout; `run_openfold` therefore reads both streams while it echoes them). For each chain it records whether the query asked for a template, whether OpenFold3 kept one, and the cause: the server replaced the alignment (default settings), the preprocessing raised (the exception is the `detail`), or nothing is known beyond "no template kept". The result is logged as a warning naming the chain and the cause, written as `<predictions>/template_accounting.json`, returned in `OpenFoldRunInfo.templates` (the object `_run_openfold_command` returns) and put in `results["prediction"]["templates"]` and `results["openfold"]["templates"]` (`{chain ID: {"requested", "source", "used", "cause", "detail", "entry_ids"}}`, the CSV row has `prediction_templates_<chain>_used`, `..._cause` and so on). `cause` is None for a used template, else `replaced_by_msa_server`, `preprocessing_failed` (`detail` is the exception OpenFold3 printed), `no_template_kept`, `not_requested` (the binder of a `refold` query) or `not_recorded`; a chain that asked for a template and got none also adds a sentence to `reason` (`templates: ...`), and the Markdown report has a row "Templates" (`A: used, B: not used (replaced_by_msa_server)`; `none asked for` for a chain that did not ask). `read_template_accounting(predictions_dir, query_name)` reads it back from a stored or adopted output (an output without the file is read from `inference_query_set.json` alone, with `requested` None). One complex, two runs of the same score query: with the template CIF unreadable and the server off, the binder is 21.6 A from the input pose (as in a run with no template), and with it read, 1.12 A.

**query sequences.** The sequence of each chain comes from the residue names of the input structure. The 20 standard residues are plain letters, and protonation and cross-link variants (HID, HIE, HIP, HIN, CYX, CYM, ASH, GLH, LYN and the lactam templates ASPL, GLUL, LYSL) take the letter of their parent residue; the variant or link itself is not sent. D-amino acids, N-methylated residues and other peptide-linking components of the Chemical Component Dictionary keep their chemistry through `non_canonical_residues` in the query JSON (`{"1": "DAL"}`, 1-based positions, CCD codes; the toolkit's template names NMG and NMA are sent as SAR and MAA), and their sequence letter is the parent residue's, upper case; OpenFold3 reads only upper-case letters and would turn the lower-case letters that gemmi uses for modified residues into unknown residues. Waters, ions, ligands and terminal caps are left out. A residue with a backbone that is none of these raises `UnmappableResidueError` (a `ValueError`) before any file is written or process started; the message names the chain, the residues and their numbers, and a batch lists every affected sample. `on_unmappable_residue="x"` on the `prepare_*` functions, or `--on-unmappable-residue x` on the command line, sends an `X` in its place instead and logs a warning.

**cyclic binder.** `binder_cyclic` of the `prepare_*` and `run_openfold_*` functions decides whether the binder chain of the query carries `"cyclic": true` (OpenFold3 0.4.5 or later; older versions reject the field). `"auto"` (the default) writes it when `capabilities.detect_closures` finds a head-to-tail bond in the binder of the input structure, the binder is made of standard residues only, and the installed OpenFold3 (read in `conda_env`, once per process) is 0.4.5 or later. "Standard" is the classification of the query: a protonation or cross-link variant (HID, CYX, ...) is sent as its parent letter and counts as standard, while a residue sent through `non_canonical_residues` (a D-amino acid, an N-methylated residue, any other CCD component) and selenocysteine do not. A head-to-tail binder that is left linear, because it has modified residues or because the version is older or unreadable, is logged as a warning and the decision says why (`BinderCyclicDecision.reason`). `True` writes the flag on the binder whatever the structure says (with a warning when the binder has modified residues) and raises `ValueError`, before anything is written, when OpenFold3 is known to be older than 0.4.5; `False` never writes it. Only the binder chain gets the field, and a disulfide, a lactam or a staple is never written, because OpenFold3 has no input for them. OpenFold3 uses the field only to wrap the relative positions of the chain: it does not enforce the closure bond, documents the field only in an example query (`examples/example_inference_inputs/query_multimer_cyclic.json`), and has published no accuracy benchmark for cyclic peptides. `decide_binder_cyclic(structure_path, binder_chain, binder_cyclic="auto", *, conda_env=None)` returns the decision as a `BinderCyclicDecision(cyclic, reason)`. In `binding-metrics-run` and `-batch` the choice is `--openfold-cyclic {auto,on,off}` (`openfold_cyclic` of `run_pipeline` and `run_batch`), and the block of the OpenFold3 step (`results["openfold"]`, or `results["prediction"]` with `--predictor of3`) gets `binder_cyclic` (bool: the binder was sent as cyclic) and, when a head-to-tail binder was left linear, a `reason`; they are the columns `openfold_binder_cyclic` and `prediction_binder_cyclic` and a row of the Markdown summary. `compute_openfold_metrics` does not know the query and does not return the key; an output adopted with `--prediction-dir` has no such key either. **Why a binder with modified residues is left linear.** `cyclic_offset` (`openfold3/core/utils/relpos.py`, v0.5.0) builds the wrapped offsets from the number of tokens of the chain and ignores the residue indices, and OpenFold3 makes one token of each heavy atom of a modified residue (1CWA chain C: 75 tokens for 11 residues), so for such a binder the wrap does not follow the ring (on a toy chain of seven tokens, residue index `[0, 1, 1, 1, 2, 2, 3]`, `apply_cyclic_offsets` turns the offsets between the three tokens of residue 1 from all 0 into `[[0, -1, -2], [1, 0, -1], [2, 1, 0]]`). The runs agree, on one complex per case (OpenFold3 0.5.0, OpenBind-0 weights, no MSA, one diffusion sample per seed; both complexes are old PDB entries that the weights have very probably seen, so these are observations about the flag, not a benchmark): for 1CWA (D-Ala and N-methylated residues, `refold`, receptor template read) the flag lowered ipTM from 0.921 / 0.922 / 0.912 to 0.784 / 0.776 / 0.809 and raised the binder Cα RMSD against the input from 0.70 / 0.47 / 0.51 A to 3.01 / 4.66 / 4.75 A (seeds 42, 7, 11; binder pLDDT 82.3 to 61.9 for seed 42; mean interface PAE 8.7-9.0 A to 10.9-11.4 A; the ring was closed without the flag, C-N distance 1.24, 1.01 and 1.28 A), and for SFTI-1 (3P8F chain I, standard residues, `refold`, one seed) it closed the ring: C-N distance 7.40 A without the flag, 1.38 A with it (1.44 A in the input), binder Cα RMSD 0.65 to 0.38 A. The mechanism is read from the source (the offsets above, run on OpenFold3's function on CPU); no ablation of the model was made, so the cause is not established. In the default `auto`, 1CWA chain C is not sent as cyclic (the log and the `reason` of the result block say why), 3P8F chain I is, and 1YCR chain B, linear, is not.

**Known OpenFold3 0.5.0 behaviours the toolkit works around.** Observed in real runs of 0.5.0 and in its source; each is handled as the paragraph named says, and none is a defect of the toolkit.

- A template that cannot be preprocessed is dropped silently. The process exits with status 0 and reports "Successful Queries", `inference_query_set.json` is rewritten with `template_entry_chain_ids: []` for the chain, and the explanation (`Failed to preprocess template alignment ...`) is a `print` on stdout and leaves no log file ("what became of the templates").
- With the ColabFold MSA server on, the server step overwrites `template_alignment_file_path` (a `UserWarning` on stderr) and `main_msa_file_paths`, and not `template_cif_paths` (the MSA server and the templates; "template mode").
- The A3M parser splits every header row on `_` into exactly two parts, so the query row of a self-alignment is `query-b<V>_<chain>`, and an MSA file is parsed only when its file name stem is a key of `MSASettings.max_seq_counts` (`colabfold_main`; another name raises an `IndexError`), the folder it is in being the representation ID ("template files"; "MSA-free runs").
- The cache of the alignment mode is keyed by the sequence hash and the hash of the A3M content only, so a template CIF that was repaired while its A3M stayed the same is served stale from the cache; the `<V>` of the query row (`QUERY_BUILDER_VERSION`) changes the key when the builder changes ("template files"). The CIF direct mode hashes the CIF content.
- `experiment_config.json` does not record a seed that OpenFold3 generated (`--num_model_seeds=1` gave the directory `seed_2746317213` and `seeds: [42]`, `num_seeds: null`), so the seed is the name of the directory (`compute_openfold_metrics`).
- A query that fails while its features are built (an unreadable template, for one) is listed as failed in `summary.txt` but leaves no `logs/predict_err_rank<N>.log`: the exception is a logger warning on stderr (`Failed to process <query> with preferredException type: ...`) and the exit status is 0 ("failures").

### CLI: `binding-metrics-openfold`

```bash
# parse an existing output directory
binding-metrics-openfold parse --output-dir ./openfold_out --query-name my_complex \
    --seed 1 --sample 1 [--binder-chain B --target-chain A] [--include-matrices]

# run inference, then parse
binding-metrics-openfold run --query-json query.json --output-dir ./openfold_out \
    --query-name my_complex --num-samples 5 [--seeds 1 2 3] \
    [--presets predict low_mem] [--no-msa-server]
```

Further subcommands: `prepare-query` and `refold` (binder refolding), `prepare-scoring-query` and `score` (scoring a complex, each chain templated on its own). `--seed` is the seed-directory index, as in the function. `run`, `score` and `refold` take `--seeds` (alias `--openfold-seeds`), the seeds OpenFold3 samples with (default 42), and `--num-seeds N`, which asks OpenFold3 to generate N seeds and cannot be combined with `--seeds`; `--num-seeds 1` is passed as `--num_model_seeds=1`, which samples with seed 2746317213 (the first seed generated from 42), not with 42. `prepare-query` and `prepare-scoring-query` still accept `--seeds` and ignore it, because a query file carries no seeds that OpenFold3 reads. `score`, `refold`, `prepare-query` and `prepare-scoring-query` take `--openfold-cyclic {auto,on,off}`, and the two prepare commands `--conda-env`, the environment asked for the OpenFold3 version. In `binding-metrics-run` and `-batch`, `--openfold-seeds SEED [SEED ...]` sets the seeds OpenFold3 samples with (default 42) and the first seed given, first sample, is scored: `--openfold-seeds 9 3` scores seed 9, the second `seed_*` directory.

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
| `chain_ptm`, `chain_pair_iptm` | dict | [0–1] | keyed by chain ID of the model's structure file (`chain_map` does not rename the keys); the pair key form is the adapter's: OpenFold3 `"(A, B)"` (as written), ColabFold and Boltz-2 `"A-B"`. Boltz-2 writes chain indices, so its adapter names them by chain ID and keeps the raw index-keyed values in `extras`. Protenix writes lists indexed by chain position, so its keys are the positions in the file, `"0"` and `"0-1"` |
| `plddt_per_atom` | ndarray \| None | [0–100] | one value per atom, in the atom order of `structure_path` |
| `pae`, `pde` | ndarray \| None | Å | `(n_tokens, n_tokens)`; `pae[i, j]` is the error of token j when the structure is aligned on token i |
| `tokens` | TokenLayout \| None | — | chain, residue and atom of each token; None means one token per residue |
| `extras`, `timing`, `reasons` | dict, dict, list | — | model-specific values, reported run times, one sentence per value that could not be provided |
| `files` | PredictionFiles \| None | — | the files the record was parsed from; `load` fills it |
| `not_provided` | frozenset[str] | — | the fields the model never writes; `load` copies them from the adapter, `validate` expects them to stay empty, and `summarize_prediction` gives no reason for them |

A scalar the model does not provide is NaN, never None. A file that is absent leaves the fields it feeds at NaN (None for an array) and adds a sentence to `reasons`; a file that is present but corrupt raises. A field that the model never writes at all is named in `not_provided` (AlphaFold2 writes no PDE) and gets no reason: a `reason` means that a value the model writes was missing. An adapter converts a 0–1 pLDDT to 0–100 and expands a per-residue or per-token pLDDT to atoms. `record.validate()` raises `ValueError` listing every violation of these rules (a scale, a shape, a `chain_map` that renames two chains to one ID), and reports a pLDDT array whose largest value is at most 1 as a probable unconverted 0–1 scale; `validate(check_structure=True)` also reads the structure and checks that the pLDDT array has one value per atom and that the `chain_map` fits the file.

A model-specific scalar or dictionary goes in `extras` under the key the model uses (`bespoke_iptm` for OpenFold3), a per-token array in `TokenLayout.extras`, and an extra file of a sample in `PredictionFiles.extra`. Generic code never reads `extras`.

**Adapters.** An adapter subclasses `PredictionParser` (`binding_metrics.predictors.base`) and reads the output of one model without running it. It implements `find_files(prediction_dir, name, *, seed_index=1, sample=1) -> PredictionFiles` and `parse(files, *, name, seed_index=1, sample=1) -> PredictionRecord`; the base class supplies `load(prediction_dir, name, *, seed_index=1, sample=1, chain_map=None)`, which chains the two and applies the chain map, and `list_samples(prediction_dir, name)`, which returns `SampleRef(seed_index, sample, ranking_score)` for each sample in the model's natural order. `complete(record)` finishes what needs the structure file, which `load` does not open (the default returns the record unchanged; it never raises for a problem of the data, it adds a `reason`): `compute_prediction_metrics`, `compute_openfold_metrics` and `PredictionSession.record` call it, and the last is the one place the pipeline and `binding-metrics-prediction` read records from. The class attributes are `name`, `display_name`, `family` (`"af2"`: one token per residue; `"af3"`: one token per standard residue and one per heavy atom of a ligand or modified residue), `capabilities` (a `Capabilities` that declares which inputs the model cannot take, or None for no declared limit; see [`preflight.md`](preflight.md)) and `not_provided` (the record fields the model never writes). `sample=1` is the first output in the model's own order; whether that is also the best-ranked one depends on the adapter (it is for Boltz-2, Protenix and ColabFold, and it is not for OpenFold3 and AlphaFold2). Parsing the scalars imports no biotite and does not open the structure file. An adapter whose confidence arrays are per residue or per token (AlphaFold2 and ColabFold, Boltz-2) opens the structure to expand them to atoms: a PDB, and for Boltz-2 a CIF, is read as text, an AlphaFold2 or ColabFold mmCIF through biotite. When the structure cannot be read, `plddt_per_atom` stays empty and `reasons` says why.

`binding_metrics.predictors.PARSERS` maps a model name to a `ParserSpec` that names the adapter class and imports it only when it is needed; `get_parser(name)` returns an adapter instance (`KeyError` listing the known names for an unknown model) and `register_parser(spec)` adds one. A contract test in `tests/predictors` runs every registered adapter against a synthetic complex; an adapter brings `tests/predictors/synth_<name>.py` with a `write_prediction` function that writes that complex in the model's layout, and needs no other test edit to be covered.

`compute_prediction_metrics(prediction_dir, model, name, seed=1, sample=1, include_matrices=False, reference_structure_path=None, binder_chain=None, receptor_chain=None, chain_map=None, *, seed_index=None, target_chain=None)` — `binding_metrics.metrics.prediction`, registered as `prediction` — loads one sample with the adapter of `model` and returns the result dictionary of `compute_openfold_metrics` (same keys, same order) with a `model` key first; `summarize_prediction(record, ...)` does the second step for a record that is already loaded. `binder_chain` and `receptor_chain` are the chain IDs of your input: give `chain_map` (`{"A": "R", "B": "P"}`, model chain to your chain) when the model named its chains differently. `KeyError` lists the registered models for an unknown one. pLDDT and ipTM are calibrated per model, so compare them within one model. The interface PDE and PAE blocks are cut with the record's `TokenLayout` when it has one, and otherwise with one token per residue.

The `of3` adapter reads the layout above: `.cif`, `.cif.gz` or `.pdb` structures, JSON or NPZ confidences (NPZ without pickle), `seed_index` as the position in the numeric order of the seed values, `sample` counted from 1. `bespoke_iptm` is in `record.extras`, `chain_pair_iptm` keys are the strings OpenFold3 writes (`"(A, B)"`), and `record.tokens` is None after `load` and is set by `complete`, from the structure and the rule of the tokenizer (see "Model adapters"). The layout of the files was checked against the OpenFold3 v0.5.0 source (released 2026-08-21), the 0.3 and 0.4 outputs the earlier parser read, and real 0.5.0 output of three complexes (see "Model adapters"). Where each adapter stands is listed under "Model adapters".

### Model adapters

Four adapters are registered (`sorted(binding_metrics.predictors.PARSERS)`). Each was written from the source and the documentation of its model. The table gives the version and the date of the source that the layout was read from and the real output that was read. Each subsection below ends with what no real run has confirmed. An adapter never starts a model. Each of the four models has a runner that does (see "Run-once prediction store"); the OpenFold3 and Boltz-2 runners have been run on the real models (see their subsections: OpenFold3 0.5.0 with the OpenBind-0 weights on 1YCR, 1CWA and 3P8F, one to three seeds, no held-out complex; Boltz-2 2.2.1 once), and the runners of Protenix and ColabFold were tested with a stand-in process only.

| adapter | model | layout read from | real output seen |
|---------|-------|------------------|------------------|
| `af2` | AlphaFold2, AlphaFold-Multimer, ColabFold | ColabFold 1.6.3 (2026-09-14) and 1.5.4, AlphaFold2 2.3.2 (2023-04-05) and main (c77e5d2); read on 2026-09-29 and 2026-09-30 | one ColabFold multimer-v3 scores file; no AlphaFold2 result pickle |
| `boltz2` | Boltz-2 | Boltz 2.2.1 (2025-09-08), main b1ebfc4; read on 2026-09-29 | Boltz-2 2.2.1 output of the runner (single-sequence, 2026-10-01); see its subsection |
| `of3` | OpenFold3 | OpenFold3 0.5.0 (2026-08-21); the layout did not change between 0.4.0 and 0.5.0 | 0.3 and 0.4 outputs; 0.5.0 output of the runner (OpenBind-0 weights; 1YCR, 1CWA and 3P8F, 2026-10-01); see its subsection |
| `protenix` | Protenix | commit 85767b8 (2026-09-21, version 2.0.0); read on 2026-09-30 | none |

#### `af2`: AlphaFold2, AlphaFold-Multimer and ColabFold

`AlphaFold2Parser` (`family = "af2"`, display name "AlphaFold2 / ColabFold") reads four layouts, told apart by their file names. `prediction_dir` is searched, and `prediction_dir/name` when it holds none of them. `name` is the ColabFold job name, the AlphaFold2 FASTA name (its files sit in `{output}/{fasta_name}/`) or the file stem of a bare structure.

| layout | files of one sample | what they give |
|--------|---------------------|----------------|
| ColabFold (`colabfold_batch`) | `{name}_{unrelaxed\|relaxed}_rank_{RRR}_{model_type}_model_{k}_seed_{SSS}.pdb`, `{name}_scores_rank_{RRR}_{model_type}_model_{k}_seed_{SSS}.json` | scores JSON: `plddt` (per residue, 0-100), `pae`, `ptm`, `iptm` (2 decimals); with `--calc-extra-ptm` `per_chain_ptm` (read as `chain_ptm`), `pairwise_iptm` (`"A-B"` keys, read as `chain_pair_iptm`), `pairwise_actifptm` and `actifptm`; 1.6.3 adds `ipsae`, `pdockq` and `pdockq2` (in `record.extras`) |
| AlphaFold2 v2.3.2 (`run_alphafold.py`) | `{unrelaxed\|relaxed}_{model}_pred_{i}.pdb`, `result_{model}_pred_{i}.pkl`, `ranking_debug.json`, `timings.json` | pickle: `plddt`, `predicted_aligned_error`, `ptm`, `iptm` (multimer only), `ranking_confidence`; the ranking score is named `iptm+ptm` or `plddts` as AlphaFold2 names it; `timings.json` is copied to `record.timing` |
| AlphaFold2 main JSON | `confidence_{model}_pred_{i}.json`, `pae_{model}_pred_{i}.json` with the PDB | per-residue pLDDT (`confidenceScore`) and PAE; no pTM or ipTM; read only when the sample has no pickle |
| structure alone | `{name}.pdb` (or `.cif`, `.mmcif`, `.pdb.gz`, `.cif.gz`, `.ent`), or any layout above without its confidence file | pLDDT from the B-factor column; everything else NaN or None, with a reason |

Tokens are residues, so `pae` is `(n_residues, n_residues)`, stored as written and never transposed (`pae[i, j]` is the error of residue j when the structures are aligned on residue i), and `record.tokens` is None. The models give one pLDDT per residue; the adapter repeats it over the atoms of each residue by reading the structure file. The residue counts of the two files must agree or `ValueError` is raised. `avg_plddt` is the mean over residues, as ColabFold and AlphaFold2 report it, and the per-residue array is `record.extras["plddt_per_residue"]`. `gpde`, `disorder`, `has_clash` and `pde` are in `not_provided` and carry no reason. Also empty: the ranking score of ColabFold (its rank is in the file name and in `extras["colabfold_rank"]`), and `ptm`, `iptm` and PAE for a monomer model without the pTM head, for the JSON files of AlphaFold2 main and for a bare structure.

Chains are those of the structure file: A, B, ... in input order, except that ColabFold places identical sequences next to each other, so the order can differ from the FASTA order; use `chain_map`. `ColabFoldRunner` (see the run-once prediction store) starts ColabFold from the receptor and the binder in that order, so a prediction it made has the receptor as chain `A` and the binder as chain `B`, and its `output_chain_map(request)` gives the `chain_map` that the pipeline passes to the reader. `seed_index` is the position of the seed in the numeric order of its value (`seed_{SSS}` for ColabFold, `pred_{i}` for AlphaFold2) and `sample` the position inside the seed: by rank for ColabFold (`rank_001` first, so with one seed `sample=1` is the best model) and by model number for AlphaFold2. The structure is the relaxed one when it exists, and the unrelaxed one is then `files.extra["unrelaxed_structure"]`. Relaxation writes hydrogens, renumbers the residues from 1 in each chain and sets every B-factor to the pLDDT of its residue (read from the ColabFold 1.5.4 source).

AlphaFold2 result pickles run code when they are loaded with `pickle`. `read_result_pickle` uses an unpickler that builds numpy arrays, numpy scalars and dictionaries and refuses every other global, naming it, before it is imported. The whole pickle is read, distogram and logits included, which takes memory in proportion to the square of the number of residues.

`load_bfactor_record(structure_path, *, name=None, chain_map=None, scale="auto")` returns the record of a bare complex, for example a BindCraft or AlphaFold2 model, and `read_bfactor_plddt(structure_path, *, scale="auto")` returns the per-atom array (`ValueError` when the column cannot be a pLDDT). `scale` is `"auto"` (a column whose largest value is at most 1 is multiplied by 100), `"percent"` or `"fraction"`. A column that is all zero or outside 0-100 gives no pLDDT and a reason. Nothing distinguishes an experimental B-factor from a pLDDT, so give it predicted structures only. The record can be the second prediction of `compute_evobind_adversarial_from_records`.

To verify. No AlphaFold2 or ColabFold run was made and no real AlphaFold2 result pickle was seen. Unconfirmed: that ColabFold 1.6.3 does what 1.5.4 does for the score arrays, the rank tags and the chain names (the 1.6.3 source was read only for file names and keys); the file names of ColabFold before 1.5.4; the names `confidence_*.json` and `pae_*.json` and the mmCIF names of AlphaFold2 main; that a real result pickle holds numpy data only, written with pickle protocol 3 to 5, and the keys of `timings.json`; that a relaxed file from a real run looks as the code says; the B-factor scale that BindCraft and ColabDesign write.

#### `boltz2`: Boltz-2

`Boltz2Parser` reads the output tree of `boltz predict`, `{out_dir}/boltz_results_{stem}/predictions/{stem}/`. `prediction_dir` can be the output directory of the run, its `boltz_results_*` folder, the `predictions` folder or the sample folder itself; `name` is the stem of the input file. `sample` counts from 1 in the rank order of Boltz-2: `sample=1` is file `model_0`, the model with the highest `confidence_score`. A run has one seed, so `seed_index` must be 1; another seed is another output directory.

| file (`{stem}_model_{r}`) | read for |
|---------------------------|----------|
| `.cif` (or `.pdb`) | structure; its atom records tell the atoms of each token |
| `confidence_....json` | `confidence_score` (ranking score), `ptm`, `iptm`, `complex_plddt` (mean pLDDT over tokens), `complex_pde` (`gpde`), `chains_ptm`, `pair_chains_iptm`; `ligand_iptm`, `protein_iptm`, `complex_iplddt` and `complex_ipde` go to `extras` |
| `plddt_....npz` | key `plddt`, one value per token, 0-1 |
| `pae_....npz`, `pde_....npz` | keys `pae` and `pde`, `(n_tokens, n_tokens)` in angstrom |

- Boltz-2 makes one token of a polymer residue, standard or modified, and one token per atom of a ligand. Nothing in the output lists the tokens, so the adapter applies that rule to the atoms of the structure file, builds `record.tokens` from it (`is_atom_token` is True for a ligand atom) and refuses a file whose token count differs from the arrays.
- pLDDT is one value per token; `plddt_per_atom` repeats it over the atoms of the token. `avg_plddt` is `complex_plddt`, the mean over tokens, which differs from the mean over atoms when the complex has a ligand or a modified residue.
- `chains_ptm` and `pair_chains_iptm` are keyed by chain index in the file. The record names them by the chain IDs of the structure file (index 0 is the first chain of the file). `chain_pair_iptm["A-B"]` is the ipTM of the tokens of chain A with the structure aligned on chain B, so `"A-B"` and `"B-A"` differ.
- `iptm` is NaN for a single chain, where Boltz-2 writes 0. `has_clash` and `disorder` are NaN and `timing` is empty: Boltz-2 writes none of them.
- The confidence score is `0.8 complex_plddt + 0.2 ipTM` (pTM for a single chain), so it ranks mostly by pLDDT; compare it only within Boltz-2.

To verify. The first real run of Boltz-2 was made on 2026-10-01 (Boltz-2 v2.2.1, single-sequence mode, `--no_kernels`, two samples) through `Boltz2Runner`, the store and this adapter, on 1YCR in the four modes and on 1CWA in `score-lock`; the adapter read the outputs cleanly (`bfactor_matches_plddt` was true and `reasons` empty for all five runs). Confirmed on those outputs: the layout `boltz_results_<name>/predictions/<name>/` and the file names; the mmCIF columns of the adapter's fixture, with `auth_asym_id` and `auth_seq_id`; the token rule (98 tokens for 1YCR, 176 for 1CWA with nine modified residues, each one token), PAE and PDE of that size, and the pLDDT in the B-factor column; and the chain indices `"0"` and `"1"` of the confidence file as chains A and B of the structure file (1YCR) and A and C (1CWA). Not confirmed: complexes of three or more chains and identical chains listed apart; that a standard residue never appears in a ligand chain and a modified residue never in a `HETATM` record of a polymer chain; that PAE and PDE are written without `--write_full_pae` and `--write_full_pde` (the runner always passes them, and the 2.2.1 source reads them in Boltz-1 only); that `complex_pde` is the quantity that OpenFold3 and Protenix write as `gpde`; versions after 2.2.1. The runner did not exercise the paths in which Boltz-2 exits with status 0 without output, `use_msa_server=True` (network) and a conda environment, and the effect of `score-lock` was measured once, on one complex: with the binder of 1YCR moved rigidly, `score` returned the same structure for every pose and `score-lock` kept the prediction within about the threshold of the supplied pose and not on it (the table is under `Boltz2Runner`, "Observed behaviour").

#### `of3`: OpenFold3

`OpenFold3Parser` (family `af3`) reads the output of `run_openfold predict`, the `--output_dir` of the run. A real 0.5.0 run (OpenBind-0 weights) writes

```
<pred>/<name>/seed_<S>/<name>_seed_<S>_sample_<k>_model.cif           (.cif.gz or .pdb with structure_format)
<pred>/<name>/seed_<S>/<name>_seed_<S>_sample_<k>_confidences_aggregated.json
<pred>/<name>/seed_<S>/<name>_seed_<S>_sample_<k>_confidences.json    (.npz with full_confidence_output_format)
<pred>/<name>/seed_<S>/timing.json                                    {"runtime_s": seconds}, one per seed
<pred>/{experiment_config.json, summary.txt, runner_config.yaml, inference_query_set.json, model_config.json, msas/}
```

`.cif.gz` with `.npz` and `.pdb` with `.json` outputs (`output_writer_settings` of the runner YAML) were read without change, once each, for 1YCR.

- `<S>` is the seed value OpenFold3 sampled with: `seeds=(7, 11)` gives `seed_7` and `seed_11`, the default `seed_42`, and `--num_model_seeds=1` (`num_model_seeds=1`) gives `seed_2746317213`, which `experiment_config.json` does not record (`seeds: [42]`, `num_seeds: null`). `seed_index` is the position of the directory in the numeric order of the seed values (`seed_7` is index 1, `seed_11` index 2), `seed_value` of the result is `<S>`, and `k` counts the samples from 1 and is not a ranking (in a five-sample run the ranking scores were 0.930, 0.919, 0.938, 0.933 and 0.939).
- The aggregated file has `avg_plddt`, `gpde`, `iptm`, `ptm`, `disorder`, `has_clash`, `sample_ranking_score`, `chain_ptm`, `chain_pair_iptm` (keys `"(A, B)"`) and `bespoke_iptm`. The full file has `plddt` (817 values for the 98 residues of 1YCR) and `pae` and `pde` (98 x 98 for 1YCR, one row per token).
- OpenFold3 makes one token of each residue whose name is in the standard set (the 20 amino acids and UNK, RNA A G C U N, DNA DA DG DC DT DN) and one token per heavy atom of every other residue, a modified residue given through `non_canonical_residues` or a ligand (`tokenize_atom_array`, `openfold3/core/data/primitives/structure/tokenization.py`, v0.5.0; the standard names are `STANDARD_RESIDUES_3` in `core/data/resources/residues.py`). The structure file is written from the atom array the tokenizer ran on, so its atoms are in token order, and `OpenFold3Parser.complete(record)` builds `record.tokens` from it with `binding_metrics.predictors.of3.token_layout(record)`. The `hetero` flag of the file is not used (UNK carries it and is one token). The rule can only count too few tokens (a free ligand named like a standard residue looks like one residue), never too many, so a layout whose size equals that of `pae` and `pde` is the model's. A record is left without a layout, and a `reason` says why, when the token count differs from the matrix size, `chain_ptm` or `chain_pair_iptm` name other chains than the structure has (chain IDs, or the numbers 1, 2, ... that the model gives them in sorted order), or the tokens of a chain are not one run. Matrices with one row per residue although the rule gives more tokens leave `record.tokens` None without a reason (the sizes agree, so the interface blocks are cut by residue, as before), and a structure that cannot be read is reported by the analysis that needs it.
- `compute_interface_pae`, `compute_openfold_metrics`, `compute_prediction_metrics` and `--predictor of3` agree on the interface values: for the real output of 1CWA (refold, seed 42) mean interface PAE 8.76 A, where the size error of the residue-count reading stood before.
- The same adapter reads the stored output of the runner (`PredictionStore`), and the run writes two things of its own next to the output of OpenFold3: `template_accounting.json` (see "what became of the templates" above) and, in the query folder, the dummy MSAs. `OpenFold3Parser.parse` puts what the run says about the templates in `record.extras["templates"]` (`{chain ID: {"requested", "source", "used", "cause", "detail", "entry_ids"}}`), read from `template_accounting.json` or, without it, from `inference_query_set.json` alone (`used` is then known and `requested` is None); an output with neither file has no such key.

To verify. The OpenFold3 runner and this adapter have been run for real: OpenFold3 0.5.0, OpenBind-0 weights (`of3-ob-2025-06-30-174k.pt`), one RTX 3070 Laptop GPU of 8 GB (about 3 GB were in use by the display), presets `["predict", "low_mem"]`, on three complexes, 1YCR (MDM2 with a p53 peptide), 1CWA (cyclophilin A with cyclosporin A, nine modified residues) and 3P8F (matriptase with SFTI-1), with one to three seeds and one to five samples; none of them is a held-out complex (they are old PDB entries that the weights have very probably seen, so the accuracy of the model is not measured here). A run of one sample took 50 to 61 s for 98 to 255 tokens (OpenFold3's own `runtime_s` 22 to 28 s, the rest is the start and the weights; 135 s for the first process after a restart), used 1.8 to 2.9 GiB of GPU memory over the baseline (sampled once a second, so a short peak can be missed) and never ran out of memory. Confirmed on that output: the layout above and the file names; the token rule (98 tokens for 1YCR and 255 for 3P8F, 241 + 14, one per residue; 240 for 1CWA, 165 for the receptor and 75 for the binder, whose 11 residues include nine modified ones; in 58 sample files the rule gave the size of `pae` and `pde`, and the interface values of the 44 files of standard residues are the same with and without the layout); the aggregated and full files, in JSON and NPZ; the seed directories above; `experiment_config.json` naming the checkpoint (`openbind-2025-06-30-174k`). Not confirmed: complexes of three or more chains, a ligand, a nucleic acid, complexes above about 260 tokens, the memory options of the OpenFold3 documentation, the batched path (`run_openfold_batched`, `prefetch`) and a merged user `runner.yml`, other presets and `low_mem` off, other releases than 0.5.0.

#### `protenix`: Protenix

`ProtenixParser` reads the output of `protenix pred`, the `-o` directory.

| file (under `{out}/{name}/seed_{S}/predictions/`) | content |
|---------------------------------------------------|---------|
| `{name}_sample_{r}.cif` | structure; the B-factor is the per-atom pLDDT, 0-100 |
| `{name}_summary_confidence_sample_{r}.json` | `plddt` (mean, 0-100), `gpde`, `ptm`, `iptm`, `has_clash`, `ranking_score`, `num_recycles`, per-chain lists |
| `{name}_full_data_sample_{r}.json` | only with `--need_atom_confidence true`: `atom_plddt` (0-1, 2 decimals), `token_pair_pae` and `token_pair_pde` (angstrom, 2 decimals), `token_asym_id`, `token_has_frame`, `atom_to_token_idx` |
| `{out}/ERR/{name}.txt` | written when a sample fails; its first line goes to `record.reasons` |

- `seed_index` is the position of the seed directory in the numeric order of the seed values. `sample` is 1-based and `r = sample - 1` is the rank by `ranking_score` inside the seed, so `sample=1` is the best sample of the seed.
- Without `--need_atom_confidence true` the record has the summary scalars, no per-atom pLDDT, PAE or PDE, and a `reason` that names the flag. The B-factor of the CIF still holds the per-atom pLDDT, finer than `atom_plddt` (a resolution of 1 on 0-100); the adapter does not read it.
- The summary lists chains by position, so `load` keys `record.chain_ptm` `"0"`, `"1"`, ... and `record.chain_pair_iptm` `"0-1"`, `"1-0"`, ...: the position of the chain in the structure file, in order of first appearance. `complete` renames them to the chain IDs of the structure file (`"A"`, `"A-B"`), the model's own IDs before `chain_map`, and keeps the position-keyed dictionaries in `extras["chain_ptm_by_position"]` and `["chain_pair_iptm_by_position"]`. If the structure cannot be read or the layout cannot be built, the keys stay positions and a `reason` says so. The diagonal of the model's pair matrix is 0 and is dropped. The other per-chain lists (`chain_iptm`, `chain_pair_iptm_global`, `chain_plddt`, `chain_gpde`, ...) are in `record.extras` as written.
- Protenix writes `disorder` as 0 for every sample, so an exact 0 reads as NaN and the written value is in `extras["disorder_written"]`. `bespoke_iptm` does not exist and no timing file is written.
- A modified residue, a ligand or an ion is one token per atom, so `pae` and `pde` are larger than the residue count and `record.tokens` is None after `load`. `ProtenixParser.complete(record)` builds the layout with `binding_metrics.predictors.protenix.token_layout(record)` from the structure and the `atom_to_token_idx` of the full-data file, so `compute_prediction_metrics`, `--predictor protenix` and the pipeline cut the interface PAE and PDE at the chain boundaries. A record that only went through `load` has them only when the matrix size equals the residue count. When the layout cannot be built (the files are not from one sample, the chain numbering disagrees) `record.tokens` stays None and a `reason` says why.

To verify. No Protenix run was made and no real output was parsed; `ProtenixRunner` was tested with a stand-in process only. Unconfirmed: that the tag v2.0.0 writes the layout of the commit above; one real run with `--need_atom_confidence true` against the adapter; that the atoms of the CIF follow the order of the arrays (`validate(check_structure=True)` and `token_layout` check the atom count and the chains); the 0-1 scale of `atom_plddt` and the shape of every array (a value or a shape that differs raises).

### Run-once prediction store (`binding_metrics.predictors`)

One prediction feeds several metrics: the confidence scalars, the interface PAE and PDE, the EvoBind score and the adversarial check. The store makes each model run happen at most once per request, across metrics, batch workers, processes and restarts. This subsection is the Python interface; the options of the command line and the layout of a stored entry are under "Pipeline" below.

```python
from binding_metrics.predictors import OpenFold3Runner, PredictionSession, PredictionStore

runner = OpenFold3Runner(conda_env="openfold3")
session = PredictionSession(PredictionStore("predictions"), [runner])
request = runner.make_request("design.cif", name="design", binder_chain="B", receptor_chain="A")
record = session.record(request)      # starts OpenFold3 on the first call only
print(session.stats()["runs"])        # 1 here; a new session or process on the same store gives 0
```

`PredictionRequest(model, name, *, mode="score", input_path=None, binder_chain=None, receptor_chain=None, sequences=None, extra_files=None, seeds=(42,), num_samples=5, model_version="", options=None)` describes a prediction. `mode` is `predict` (the sequences only), `refold` (the binder from its sequence beside a receptor given as template), `score` (each chain given its own structure as a template) or `score-lock` (`score` with the pose pinned to the input); a runner raises for a mode it cannot run (see "Runners"). `request.key()` is the SHA-256 of the canonical JSON of the model, its version, the mode, the seeds, the number of samples, the options, the chain roles, the sequences, the content hash of the input file and of each extra file, and, for a request with custom weights, the content of the weights. A moved or renamed identical file gives the same key; another seed, option, model version or file content gives another. `weights` (keyword-only, a path or the `WeightsRef` of `PredictionStore.weights_reference`) names custom weights, such as a fine-tuned checkpoint: the key holds their kind, SHA-256 and size and never their path, so a copy under another name shares the entry and changed weights get their own, and a request without weights has the key it always had (existing stores stay valid). `request.json` records the path next to the hash. A request without `model_version` takes the version that the runner reports, so predictions of two versions never share an entry. `name` labels the output files and is not part of the key of a run, so two samples with the same structure share one run.

Outputs that the user made are the exception: they belong to a name, because a directory holds one output per name. `adopt` stores the entry under `request.for_adoption().key()`, the key with the name added, so two samples with one input file each keep their own outputs, and `lookup` tries the adopted entry first. Output adopted without a model version is found first even when the runner reports one. `PredictionStore.get_or_run(..., rerun=True)` runs the model again and drops the adoption of that request; `PredictionSession(..., rerun=True)` leaves the outputs adopted through it alone.

`PredictionStore(root)` keeps `<root>/<model>/<key[:2]>/<key>/` with `request.json`, `STATUS.json` (`done`, `failed` with the reason, or `adopted`; timestamps; runner and version) and `outputs/`, the files of the model, untouched.

| method | effect |
|--------|--------|
| `lookup(request)` | the stored entry of a request (`StoredPrediction`), or None |
| `get_or_run(request, runner, *, rerun=False)` | the entry; the model runs only when there is none |
| `adopt(request, directory, *, copy_outputs=False)` | registers outputs the user produced; they stay where they are unless `copy_outputs` is True |
| `run_missing(requests, runner, *, rerun=False, max_batch=256)` | runs the requests that have no entry, in one batched call (of at most `max_batch` requests) when the runner supports it; a failed run is recorded, not raised |

A run is written to a temporary directory and renamed into place, so a killed run leaves no finished entry, and an advisory `fcntl.flock` per key makes a second process wait and reuse the result. That needs a POSIX file system on which `flock` works, on one host at a time; it is not available on Windows. A run that raises is recorded as `failed` with the exception text as its reason, and asking again raises `PredictionFailedError` with that reason until a run is forced. A model that cannot be started (`PredictionRunner.is_available()` is False) raises `PredictionUnavailableError` and records nothing, so a missing installation is not remembered as a failed prediction.

`PredictionSession(store, runners=None, parsers=None, *, rerun=False)` is what a pipeline passes around. `runners` maps a model name to its runner, or is a list of runners; a model without a runner is served from what the store holds. `parsers` defaults to the registry.

| method | effect |
|--------|--------|
| `record(request, *, seed_index=1, sample=1, chain_map=None)` | the parsed `PredictionRecord`; the model runs on the first miss only and the files are parsed once |
| `prefetch(requests)` | runs all missing requests of a batch of samples in one model process |
| `adopt(request, directory, *, copy_outputs=False, check=True)` | registers a directory for parse-only use; with `check`, a directory in which the adapter finds no file of the sample raises `ValueError` |
| `stats()` | the counters below |

`stats()` returns `requests`, `memo_hits` (answered from the session's memory), `hits` (a finished run was in the store), `adopted`, `misses` (the model was run, or a run was forced or attempted), `runs` (predictions the model computed for this session; a failed run counts), `failed` and `parsed`, with `requests == memo_hits + hits + adopted + misses`.

`PredictionRunner` (`binding_metrics.predictors.runners`) is the base class of a model runner: `prepare(request, work_dir)` writes the input files of the model, `run(request, work_dir)` runs the model and returns the directory its adapter loads, and the optional `supports_batch`, `run_many`, `is_available` and `version` say what it can do. A runner never reads the output; the adapter of the same model does. Two class attributes say whether it can start its model with custom weights: `supports_custom_weights` (default `False`) and `weights_kind` (`"file"`, the default, or `"directory"`). A runner that sets `supports_custom_weights = True` passes `request.weights.path` to its model; for any other runner the store refuses a request that has weights, before anything runs and before anything is stored, with a `ValueError` that names the model (`require_weights_support`; a runner's `prepare` and `run` call `self.check_weights(request)` for a direct call). A runner that ignored the weights would run the default model and store the result under the key of the custom one. Three more class attributes and methods describe what a runner can do with a complex structure: `supported_modes` (a frozenset of `predict`, `refold`, `score` and `score-lock`; `{"predict"}` by default), `default_mode` (the mode used when the caller names none; `None` takes the default of the `mode` parameter of its `make_request`) and `output_chain_map(request)` (`{chain ID in the prediction: chain ID in the input}` when the model names the chains itself, `None`, the default, when the prediction keeps the input's IDs). The command line reads them to refuse a mode the runner does not run and to pass the chain map to the reader; a runner that names the modes it builds an input for `run_modes` (`Boltz2Runner`) is read through that name. Every registered model has a runner, listed under "Runners" below. `OpenFold3Runner(conda_env=None)` starts OpenFold3 through `run_openfold_scoring`, `run_openfold_refolding`, `run_openfold_batched` and `run_openfold`. Its `make_request` writes every setting that changes the output into the request, so two callers who mean one run get one key: the presets, the MSA server switch, the seeds (explicit, `[42]` by default, or none when `num_model_seeds` asks OpenFold3 to generate them) and samples, `binder_cyclic`, `on_unmappable_residue`, the checkpoint and its size (`inference_ckpt_path`) or, for custom weights (`weights`, a checkpoint file passed as `--inference-ckpt-path`), the content of the file, the content of a template or runner YAML, and the content of OpenFold3's user-default `runner.yml`. The alignments that an MSA server returns are not in the key, because the answer of a remote server can change over time.

#### Runners

One runner per registered model (`cli.prediction.RUNNERS`, the model key to `"module:Class"`). Each takes the name of a conda environment as its only constructor argument (`conda run -n NAME`); None starts the model's executable from `PATH`. Each `make_request(input_path, *, name, binder_chain, receptor_chain, mode, seeds, ...)` writes every setting that changes the output into the request, so two callers that mean one run share a key, and the conda environment is never part of it (the model version is).

| model | runner (module in `binding_metrics.predictors`) | executable | modes (default) | seeds (default) | custom weights | `use_msa_server=False` | `binder_cyclic` | chains of the prediction |
|-------|-------------------|------------|-----------------|-----------------|----------------|------------------------|-----------------|---------------------------|
| `of3` | `OpenFold3Runner` (`of3_runner`) | `run_openfold` | `refold`, `score` (`score`) | any list (`[42]`) | one file, `--inference-ckpt-path` | OpenFold3 with a dummy MSA of the query sequence and without the ColabFold MSA server | `cyclic: true` on the binder chain when it has a head-to-tail bond and standard residues only (OpenFold3 0.4.5 or later) | the input's |
| `boltz2` | `Boltz2Runner` (`boltz2_runner`) | `boltz predict` | `predict`, `refold`, `score`, `score-lock` (`score`) | exactly one (`[42]`) | one file, `--checkpoint` | `msa: "empty"` on every entity | `cyclic: true` on the binder entity | the input's |
| `af2` | `ColabFoldRunner` (`af2_runner`) | `colabfold_batch` | `predict` (`predict`) | consecutive integers from 0 (`[0]`) | a directory, `--data` | `--msa-mode single_sequence` | no setting: the runner has no `binder_cyclic` | receptor `A`, binder `B` |
| `protenix` | `ProtenixRunner` (`protenix_runner`) | `protenix pred` | `predict` (`predict`) | distinct non-negative integers (`[101]`) | a directory, through a temporary `PROTENIX_ROOT_DIR` | `--use_msa false` | head-to-tail and disulfide bonds as `covalent_bonds` | the input's |

The default of a mode is the one the runner's `make_request` documents. Boltz-2 is the only runner with all four modes, and its default is `score`, as for OpenFold3: every chain is given its own structure as a template, so the two models answer the same question and their results can be set side by side; `predict` is its sequence-only mode. A mode a runner does not run raises `ValueError` from `make_request`, and the command line refuses it before anything runs (see "Pipeline").

**`OpenFold3Runner`** runs `score` and `refold` on a complex structure. Its `predict` mode takes a query file that the caller wrote and is not reachable from the command line, and `score-lock` cannot be run: the templates carry the fold of each chain and no pose between chains. See the runner paragraph above for its key. It has been run on a real OpenFold3 0.5.0 (OpenBind-0 weights; 1YCR in `score`, `refold` and `predict`, 1CWA in `score` and `refold`, 3P8F in `refold`, and the pipeline end to end for 1YCR): one process per request through `conda run -n openfold3`, a missing conda environment, a missing default checkpoint, a query that fails while its features are built and a `predict` request named differently from its query key were exercised and give a reason, and the batched path (`run_many`) was not run.

**`ColabFoldRunner`** (AlphaFold2 and AlphaFold-Multimer through `colabfold_batch`) runs the mode `predict` only. The input is one FASTA record, the receptor and the binder joined by `:`, written from the two sequences of the input structure; the request holds the sequences and no input file, so two poses of one receptor and binder (and two samples with the same sequences) share one run. ColabFold names the chains in that order, so the prediction has the receptor as chain `A` and the binder as chain `B` whatever their IDs in the input, and `ColabFoldRunner.output_chain_map(request)` gives `{"A": receptor, "B": binder}`, which the pipeline passes as `chain_map` unless `--prediction-binder-chain` or `--prediction-target-chain` is given; the adapter renames the atoms only, so the keys of `chain_ptm` stay `A` and `B`. A chain with a residue other than the 20 amino acids and `X` (a D-amino acid, an N-methylated or phosphorylated residue, any component with its own Chemical Component Dictionary code, selenocysteine) is refused before anything is written, naming the chain and the residues; `--on-unmappable-residue x` is not accepted. The seeds must be consecutive integers from 0 (`--random-seed` and `--num-seeds`), `num_samples` is `--num-models`, and the job name is a file label: a sample ID with a space, a slash or another character that ColabFold rewrites is refused. The key holds the sequences, the seeds, `num_samples`, `use_msa_server` (`mmseqs2_uniref_env` or `single_sequence`), `model_type` (`alphafold2_multimer_v3`), `num_recycles`, `num_relax` (0: no relaxation), `extra_args` and the ColabFold version. Custom weights are a `--data` directory identified by content; one without the marker file of the model type in `params/` is refused, because `colabfold_batch` would download the default parameters over it. ColabFold exits with status 0 when it skips a job (a failed MSA request, GPU out of memory), so the runner checks that the structures exist and records the line ColabFold logged as the reason. `refold`, `score` and `score-lock` raise: the source shows no way to give ColabFold a template per chain. What the runner writes was read in the source of ColabFold v1.6.3 (2026-10-01); it was not run against ColabFold.

**`ProtenixRunner`** (`protenix pred`) runs the mode `predict` only: Protenix takes templates as alignment files (`templatesPath`), so a structure cannot be given as a template, and `refold`, `score` and `score-lock` raise. The input JSON has the receptor as entity 1 and the binder as entity 2, each with `id` set to its chain ID in the input, so the prediction keeps the input's chain IDs (to verify with a real run: the CIF writer was not read). A D-amino acid, an N-methylated residue or another peptide-linking component is its parent letter plus a `CCD_<code>` modification, and a residue with no such code is refused before anything is written (`--on-unmappable-residue x` sends an `X`). `--need_atom_confidence true` is always given: it makes Protenix write the per-atom pLDDT, PAE and PDE that the interface PAE and PDE need. The key holds the model name, dtype, seeds, samples, `use_msa_server`, `msa_server_mode`, the optional `constraints` (the `constraint` section, read by `protenix_base_constraint_v0.5.0` only) and `covalent_bonds`, `extra_args`, the closure setting, the size of the default weights file and the Protenix version. With `use_msa_server` on, Protenix sends the sequences to its MSA server (`https://protenix-server.com/api/msa` by default); off, it runs without an MSA, which Protenix warns degrades the result. `binder_cyclic="auto"` writes the head-to-tail and disulfide closures that `detect_closures` finds in the binder as `covalent_bonds` (C of the last residue to N of the first; SG to SG); a closure of another kind is not written, and one that cannot be placed raises instead of predicting a ring as a linear chain. That this makes the amide bond was read in the source and not run. Custom weights are a directory with `<model_name>.pt`: `protenix pred` has no option for it, so the run points `PROTENIX_ROOT_DIR` at a temporary root whose `checkpoint` links to that directory and whose other entries link to the real root, which must already hold the CCD data (run Protenix once with its default weights). Protenix exits with status 0 when a sample fails or when the MSA search fails (it then continues with the sequence as its own MSA), so the runner checks the output and records the error that Protenix wrote to `ERR/`, a missing seed or sample, or the failed MSA search as the reason. What the runner writes was read in the source of Protenix 2.0.0 (commit 85767b8, 2026-10-01); it was not run against Protenix.

**`Boltz2Runner`** (`boltz predict`) builds a YAML from the complex structure, with one protein entity per chain under the chain IDs of the input, the sequence and the modified residues read as for OpenFold3 (CCD codes in `modifications`; selenocysteine is sent as `C` with `SEC`; a residue that cannot be expressed raises `ValueError` naming the chain and the residues before any file is written), `cyclic: true` on a head-to-tail binder (`binder_cyclic="auto"`; Boltz-2 turns the flag into a cyclic period of the chain length for the relative positions and does not enforce the closure bond), and `msa: "empty"` without the MSA server. `templates` of the YAML depend on the mode:

| Mode | `templates` of the YAML |
|---|---|
| `predict` | none |
| `refold` | `receptor.cif` for the receptor chain; the binder is free |
| `score` | `receptor.cif` and `binder.cif`, one chain each, unforced |
| `score-lock` | `lock.cif` with both chains in the frame of the input, listed for both chains with `force: true` and `threshold` |

The template CIFs name every residue after its parent (`MLE` becomes `LEU`), because Boltz-2 reads a modified residue of a template as `X`. The threshold of `score-lock` is `lock_threshold_angstrom` (`--prediction-lock-threshold`), 2.0 angstrom by default: Boltz-2 requires a threshold with `force` and documents no default, so 2.0 is this package's choice, and it is part of the key. It is the distance, after a rigid alignment of the templated residues, that a residue's representative atom (CB, CA for glycine) may move from the template before a guidance term of weight 0.1 pulls it back; it is not a hard constraint. A request takes exactly one seed (default 42), because a Boltz-2 run has no seed dimension: `--openfold-seeds` with more than one seed is recorded as the error of the step, and a second seed is a second request. `num_samples` (default 5) is `--diffusion_samples`. Custom weights go as `--checkpoint`. The Boltz-2 version in the key is read from the installation that `run` starts: by the interpreter named on the first line of the `boltz` script that `PATH` finds (what `pip`, `pipx` and `conda` write), or by `conda run -n ENV python` with a conda environment, from the package metadata. It is empty when `boltz` is not on `PATH`, when its script is not a Python script (a shell wrapper) or when that interpreter has no `boltz` distribution. `extra_args` go to `boltz predict` as they are and may not repeat a flag the runner sets; `--no_kernels` is the one to know for a GPU on which the cuequivariance kernels do not run. Chains with one sequence become one Boltz-2 entity that keeps the modifications and the `cyclic` flag of the first, so such chains that differ raise `ValueError`. A failed `boltz predict` raises `Boltz2RunError` with the key line of its output and the command; a run that exits with status 0 without output (a failed input, a batch out of memory) raises `RuntimeError` with Boltz-2's message. This is the one runner that has been run for real: on 2026-10-01 through the store and the adapter on a real Boltz-2 v2.2.1, in single-sequence mode with `--no_kernels`, in all four modes on 1YCR and in `score-lock` on 1CWA; the adapter read the outputs cleanly. The exit-0-without-output paths, `use_msa_server=True` and a conda environment were not exercised, and the effect of `score-lock` was measured on one complex only (see below).

**Observed behaviour (Boltz-2, one complex).** The modes were run on 1YCR (MDM2 as chain A, the 13-residue p53 peptide as chain B) with Boltz-2 v2.2.1 in single-sequence mode (`use_msa_server=False`), one diffusion sample and the seeds 42, 43 and 44. The binder was moved as a rigid body with the receptor fixed, and each moved pose was the input of a run: 3, 6 and 10 angstrom away from the pocket, a 30 degree tilt of the helix with 4 angstrom of translation, and a 30 degree roll about the helix axis with 4 angstrom of translation. The `score-lock` rows use the default threshold, 2.0 angstrom. Each row is the mean over the three seeds; the two RMSD columns are the binder C-alpha RMSD of the prediction after superposing the receptor, to the pose that was the input and to the crystal pose.

| Mode | Input pose | ipTM | Interface PAE (A) | RMSD to input pose (A) | RMSD to crystal pose (A) |
|---|---|---|---|---|---|
| `predict` | none (sequences only) | 0.948 | 2.25 | - | 1.99 |
| `score` | crystal | 0.954 | 2.12 | 1.56 | 1.56 |
| `score` | any decoy | as for the crystal pose | as for the crystal pose | 2.9 to 9.6 | as for the crystal pose |
| `score-lock` | crystal | 0.953 | 2.15 | 1.26 | 1.26 |
| `score-lock` | 3 A decoy | 0.953 | 2.00 | 2.02 | 1.61 |
| `score-lock` | 6 A decoy | 0.829 | 3.52 | 2.27 | 3.93 |
| `score-lock` | 10 A decoy | 0.579 | 6.80 | 2.31 | 7.82 |
| `score-lock` | 30 degree tilt, 4 A | 0.862 | 3.08 | 2.04 | 3.64 |
| `score-lock` | 30 degree roll, 4 A | 0.910 | 2.52 | 1.76 | 2.64 |

Reading it. Under `score` the predicted structure did not depend on the pose that was supplied: for five decoys its coordinates were identical to those of the crystal-pose run with the same seed, so a `score` run cannot tell a right pose from a wrong one. Under `score-lock` the prediction stays within about the threshold of the supplied pose and not on it: the representative atom of every templated residue ended within 2.06 angstrom of the input after a rigid alignment in all 18 runs, so the RMSD to the input pose is 1.8 to 2.3 angstrom for decoys 3 to 10 angstrom out, and the RMSD to the crystal pose grows with the displacement. The confidence falls as the supplied pose moves away from what the model prefers: ipTM and the interface PAE are worse than for the crystal pose by more than the spread of the seeds from 6 angstrom or 30 degrees on (by little for the roll), and not at 3 angstrom, where ipTM is 0.953 as for the crystal pose. Read it together with the RMSD to the input pose, which says how far the model pulled the pose: a low ipTM next to a prediction 2 angstrom from the supplied pose suggests that Boltz-2 finds that pose less plausible, and does not mean that the pose was reproduced. A smaller `--prediction-lock-threshold` held the pose tighter in checks with one seed each (the 6 angstrom decoy: RMSD to the input pose 1.18 angstrom at a threshold of 1.0 and 0.63 at 0.5, against 2.27 at 2.0). The same two-chain template without `force: true` did not carry the pose: for three decoys (one seed) the prediction was near the crystal pose, as under `score`.

Caveats. This is one complex, which Boltz-2 very likely saw in training (not checked), so the model already finds the pocket from the sequences (`predict` gives ipTM 0.95), and a template shows in the numbers only where it conflicts with that. The sample is one crystal pose, five decoys and three seeds of one diffusion sample, and the ranges are minima and maxima, not confidence intervals. A binder 10 angstrom away is not bound (no atom pair within 4 angstrom), so its low ipTM is partly that. The lock gave the crystal pose no more confidence than `predict` or `score` did. Nothing here says how `score-lock` behaves for a designed binder, another complex, another model or another Boltz-2 version, with the MSA server, or with more than one diffusion sample.

### Pipeline: `--predictor`, the prediction store and `results["prediction"]`

`binding-metrics-run` and `binding-metrics-batch` read a prediction in their `openfold` step. Without `--predictor` that step is what it always was: it runs OpenFold3, parses it with `compute_openfold_metrics` and writes `results["openfold"]` (`openfold_*` columns). With `--predictor MODEL` it works for any model with an adapter, runs the model itself unless `--prediction-dir` gives an output you made, and writes `results["prediction"]` (`prediction_*` columns); `results["openfold"]` is then `{"skipped": true}`. The step is selected by `openfold` in `--metrics` whichever model it runs: `--metrics` accepts `dockq`, `electrostatics`, `energy`, `geometry`, `interface` and `openfold`, and `prediction` is not a metric name (`--metrics prediction` is a usage error that lists the valid ones). A run of `binding-metrics-run --predictor of3 --metrics interface,geometry,openfold` on 1YCR (OpenFold3 0.5.0, the default settings: five samples, the MSA server) exited with 0 in 66 s and wrote `results["prediction"]`, the report and the CSV row; the same command again, from the stored entry, took 10 s and started no model (`prediction_cache_hits` 1, `prediction_cache_runs` 0) and read back the same `templates`.

| option | default | effect |
|--------|---------|--------|
| `--predictor {af2,boltz2,of3,protenix}` | none | the model whose prediction the step reads; the choices are `sorted(binding_metrics.predictors.PARSERS)`. Without `--prediction-dir` the model is run (every one has a runner, see "Runners") |
| `--prediction-dir DIR` | none | read an output you made; the model never runs. `DIR` is the directory the adapter loads with the sample ID (`--sample-id`, default the file stem) as the name; for OpenFold3 the folder that holds `<sample>/seed_*/`. The layout under it is the adapter's own, described in the docstring of `binding_metrics.predictors.<model>`. In `binding-metrics-batch` it is the root that holds one output per sample ID |
| `--prediction-mode {predict,refold,score,score-lock}` | the default mode of the runner: `--openfold-mode` (`score`) for `of3`, `score` for `boltz2`, `predict` for `af2` and `protenix`; not stated for `--prediction-dir` | how the model is used: `predict` (sequences only), `refold` (receptor templated, binder free), `score` (every chain templated, the pose not given), `score-lock` (`score` with the pose pinned to the input). Checked against `Capabilities.modes` of the model (pre-flight) and, for a run from here, against the `supported_modes` of its runner (`of3`: `refold`, `score`; `boltz2`: all four; `af2` and `protenix`: `predict`); recorded as `mode`. Needs `--predictor` |
| `--prediction-weights PATH` | the model's own weights | custom weights for the model, such as a fine-tuned checkpoint: a file for a model that takes a checkpoint file (OpenFold3: `--inference-ckpt-path`), a directory for a model whose weights are a directory. It applies to `--predictor` run from here and to the `openfold` step without `--predictor` (`binding-metrics-openfold` names it `--ckpt`). A usage error with `--prediction-dir`. See "custom weights" below |
| `--prediction-binder-chain`, `--prediction-target-chain` | the input's IDs | chain IDs inside the prediction when they differ from the input's; they build the `chain_map` of the record (model chain to input chain) |
| `--prediction-cache DIR` | `<output-dir>/predictions` (`-batch`: `<output-dir>/_predictions`) | the store of finished predictions, shared by all samples of a batch |
| `--rerun-predictions` | off | run each prediction again although the store has it, once; outputs given with `--prediction-dir` are never replaced |
| `--prediction-cyclic {auto,on,off}` | `auto` | whether the binder is given to the model as cyclic, for a run from here: `auto` when the binder has a head-to-tail bond, `on` always, `off` never. OpenFold3 and Boltz-2 get `cyclic: true` on the binder chain, which wraps its relative positions and does not enforce the closure bond; Protenix gets the head-to-tail and disulfide bonds as `covalent_bonds`; ColabFold has no such setting and refuses `on` and `off` |
| `--prediction-no-msa-server` | the server is on | run without an MSA server (`af2`: `--msa-mode single_sequence`; `boltz2`: `msa: "empty"`; `protenix`: `--use_msa false`; `of3`: as `--openfold-no-msa-server`). The accuracy for a natural receptor drops; no sequence leaves the machine |
| `--prediction-conda-env NAME` | the executable on `PATH` | the conda environment that has the model (`conda run -n NAME`); an empty string says the current environment |
| `--prediction-lock-threshold ANGSTROM` | 2.0, the Boltz-2 runner's own | the threshold of the forced template of `score-lock`; accepted only for a runner that has the setting (Boltz-2) and with `--prediction-mode score-lock` |
| `--openfold-mode`, `--openfold-seeds`, `--openfold-cyclic`, `--openfold-no-msa-server`, `--openfold-conda-env`, `--on-unmappable-residue` | as before; `auto` for `--openfold-cyclic`; the server is on; `openfold3` for `--openfold-conda-env` | configure the OpenFold3 step, and `--predictor of3` run from here (`binder_ca_rmsd` against the input is measured in both modes). `--openfold-seeds` and `--on-unmappable-residue` go to the runner of any `--predictor` (see "Runners" for the seeds each takes); `--openfold-mode`, `--openfold-cyclic`, `--openfold-no-msa-server` and `--openfold-conda-env` are the OpenFold3 spelling of `--prediction-mode`, `--prediction-cyclic`, `--prediction-no-msa-server` and `--prediction-conda-env`: for `--predictor of3` both spellings are one setting (different values are a usage error), for another model the `--openfold-*` spelling other than its default is a usage error that names the `--prediction-*` one; ignored for a prediction that is read |

Every registered model has a runner, so `--predictor MODEL` without `--prediction-dir` runs the model for all four. The command line is checked before anything runs, and a setting that the model's runner cannot take is a usage error (exit status 2) that names the option and the model: a mode outside the runner's `supported_modes` (the message lists the modes it runs and says to run the model yourself and read its output with `--prediction-dir DIR --prediction-mode MODE`; a mode that the model itself does not have, such as `score-lock` for OpenFold3, is left to the pre-flight check, which also lists the models that have it), `--prediction-cyclic` for ColabFold, `--prediction-lock-threshold` for a runner without it or outside `score-lock`, an `--openfold-*` option other than its default for a model other than OpenFold3, and the two spellings of one setting with different values for OpenFold3. The settings of a run are refused with `--prediction-dir`, because the model is not run; a prediction option without `--predictor` is refused too. A seed list that the runner cannot take (`1 3` for ColabFold, two seeds for Boltz-2) is not a usage error: the runner raises when the request is built and the message is the `error` of the step. In Python, `run_pipeline` takes the keyword-only `predictor`, `prediction_dir`, `prediction_binder_chain`, `prediction_target_chain`, `prediction_cache`, `rerun_predictions`, `prediction_mode`, `prediction_weights`, `prediction_cyclic`, `prediction_use_msa_server`, `prediction_conda_env` and `prediction_lock_threshold`, and `run_batch` takes the same, and both raise `ValueError` for the same combinations before any step runs. The runner gets only the settings it has a keyword for (`cli.prediction.make_request` reads the signature of its `make_request`), so an option that a runner lacks is a `ValueError` that names it, never a `TypeError`. TOML config files set the options under their long names (`predictor = "boltz2"`, `prediction-dir = "outputs/"`).

**the chains of the prediction.** `binder_ca_rmsd` and the EvoBind check read the prediction by the chain IDs of the input. A runner whose model names the chains itself gives the map (`output_chain_map(request)`: ColabFold, receptor `A` and binder `B`), which is passed as `chain_map` to the reader unless `--prediction-binder-chain` or `--prediction-target-chain` is given. For an output read with `--prediction-dir` that uses other IDs, give those options; when a chain of the input is not in the prediction the `reason` of the block says which chains it has, names the model and names the options. `binder_ca_rmsd` is measured against the input pose for every model, in the receptor frame, in every mode: the binder is placed by the model itself in `predict`, `refold` and `score` alike, because a template holds one chain and no pose between chains.

**one run per prediction.** A sample gets one `PredictionSession` on the store. The session asks the store for the sample's request; the model runs only when the store has no finished entry for it, and the record is parsed once. The confidence scalars, the interface PAE and PDE, the primary EvoBind score and the adversarial check all read that one record, so a model runs at most once for all of them, and a second run of the same input, model version, seeds and options starts no model. In a batch the requests of all samples go to the session's `prefetch` first, which starts the model once for those the store lacks (OpenFold3 predicts them in one call, as the batched OpenFold3 step always did; the other runners have no batched mode and start the model once for each prediction the store lacks, and ColabFold, whose request holds the sequences and no file, once for each distinct pair of sequences); every sample then reads its record with a session of its own, so its `cache` shows `hits: 1`. The step runs after the workers, in the main process, like the batched OpenFold3 call.

**the store.** One entry per request, in `<cache>/<model>/<key[:2]>/<key>/`:

```
request.json    what was asked: the model, its version, the mode, seeds, samples, options, chain roles, input hashes
STATUS.json     status (done, failed with the reason, or adopted), timestamps, runner and version
outputs/        the model's own files, untouched
```

`<key>` is the SHA-256 of a canonical description of the request: everything that changes the output, and the content hash of the input file, never its path. The same input, seeds, options and model version give the same key on every machine and after a move or rename of the file; another seed, option, model version or file content gives another. A run is written to a temporary directory and renamed into place under a file lock per key, so a run that is killed leaves no finished entry and two processes that ask for one request start the model once. A run that failed is recorded with its reason and is not retried unless `--rerun-predictions` is given; a model that cannot be started here (no `run_openfold` on the PATH, no `openfold3` in the conda environment) records nothing, so a missing installation is not remembered as a failure. Output given with `--prediction-dir` is adopted into the store by reference (`adopted`, the files stay where they are); its key includes the sample name, so two samples with identical input files keep their own outputs.

**custom weights.** `--prediction-weights PATH` (`prediction_weights` of `run_pipeline` and `run_batch`) runs the model with weights you choose, for instance a fine-tuned checkpoint. The path must exist, be readable and be the kind the model's runner takes (OpenFold3: one checkpoint file, given to it as `--inference-ckpt-path`); that is checked before anything runs, and a bad path raises `ValueError` (the command exits with status 1) naming what was found. The pre-flight check refuses a model whose runner cannot take custom weights, in the usual form (found, requires, why, fix), and the fix names the runners that can, read from the runner classes; it never lets such a run go on with the default weights. When the weights are accepted the plan has a note: the limits declared for the model come from its input format and architecture, and the caveats about accuracy and published benchmarks refer to the standard weights; the declared hard limits stay in force. With `--prediction-dir` the option is a usage error, because the weights of an output you made are whatever produced it.

The weights are identified by content. A file is hashed by SHA-256 and a directory by a manifest, the sorted list of (relative path, size, SHA-256) of its files, so a moved or renamed copy gives the same store key and any change in content gives another. A checkpoint of 2 GB is not read on every run: the SHA-256 of each file is kept in `<store root>/.weights-sha256.json` under its absolute path, size and modification time (`mtime_ns`), written atomically under a lock and rebuilt when it is corrupt. A file edited in place with an unchanged size and `mtime_ns` is not detected; delete the cache file to force a re-read.

`results["prediction"]["weights"]` (`results["openfold"]["weights"]` without `--predictor`) holds `custom` (bool), `name`, `path`, `sha256`, `size` and, for custom weights, `kind` (`file` or `directory`) and `n_files`; for the model's own weights `custom` is False, `name` and `path` are the checkpoint that OpenFold3 recorded in `experiment_config.json` (the same for an output read with `--prediction-dir`) and `sha256` and `size` are None because the file was not read. The key is absent when nothing is known. `provenance["prediction_weights"]` repeats it, the CSV has `prediction_weights_sha256`, `prediction_weights_path` and the other entries (`openfold_weights_*` without `--predictor`), and the summary has a "Weights" row. `record.extras["weights"]` is set from the request. One real run (OpenFold3 0.5.0, 1YCR, `score`, seed 42) passed the default checkpoint file as custom weights: another request key, the SHA-256 of the 2.29 GB file in `record.extras["weights"]` (hashing took about 50 s once), and the same numbers as the default run (pLDDT 91.383 against 91.392, ipTM 0.852 both, interface PAE 4.232 against 4.233, binder Cα RMSD 1.122 against 1.123 A). A custom checkpoint has no published benchmark here: the limits and caveats of the model were written for its standard weights.

**`results["prediction"]`.** The keys of the table in section 12, with `model` first, plus:

| key | description |
|-----|-------------|
| `model` | the adapter name (`"of3"`, `"boltz2"`, ...) |
| `if_dist_pep_to_rec`, `if_dist_rec_to_pep`, `if_dist_symmetric`, `n_interface_receptor_residues`, `interface_fallback_used`, `mean_plddt_binder`, `evobind_score` | the primary EvoBind score of section 13 on the predicted structure, with the binder pLDDT of the same record |
| `delta_com_angstrom`, `n_superposition_residues`, `n_superposition_atoms`, `receptor_pairing`, `binder_pairing`, `*_resname_mismatch_fraction`, `afm_*`, `evobind_adversarial_score` | the adversarial check of section 13 between the input pose and the prediction; `afm_*` describe the prediction whatever its model |
| `design_model`, `adversary_model` | `None` (the input pose is a file) and the adapter name |
| `binder_cyclic` | bool: the query sent the binder chain as `"cyclic": true` (`--openfold-cyclic`, see "cyclic binder" above). Present when OpenFold3 was run here (not for an output adopted with `--prediction-dir`, and not for the other models: Boltz-2 and Protenix decide inside their runners and the setting, not the decision, is in the `request.json` of the store entry); a `reason` joins the key when a head-to-tail binder was left linear because it has modified residues or the OpenFold3 version is older than 0.4.5 or unreadable |
| `templates` | `{chain ID: {"requested", "source", "used", "cause", "detail", "entry_ids"}}`: whether the query asked for a template for the chain, whether OpenFold3 kept one (`used`) and, when it did not, the `cause` (see "what became of the templates" above). Present for OpenFold3 when its output has `template_accounting.json` or `inference_query_set.json`, also for an output adopted with `--prediction-dir` (`requested` is then None); absent for the other models and for an output with neither file. A chain that asked for a template and got none adds a sentence to `reason` (`templates: ...`) |
| `seed_value` | the seed behind the sample (`int`, from the name of the seed directory; None for an adapter that names none), as in `compute_openfold_metrics` |
| `evobind_error`, `adversarial_error` | why the score or the check could not be computed; the rest of the block is unaffected |
| `reason` | the `reason` strings of the parts, labelled and joined with `; ` (`evobind: ...`, `evobind adversarial: ...`) |
| `mode` | how the model was used: `predict`, `refold`, `score` or `score-lock`; `null` for an output whose making is not stated |
| `cache` | `requests`, `memo_hits`, `hits`, `adopted`, `misses`, `runs`, `failed`, `parsed` (the counters of `PredictionSession.stats()`; `runs` is 1 when the model was computed for this sample, 0 when the store or `--prediction-dir` supplied it) and `request_key`, the name of the store entry |

A prediction that failed, or that cannot be started here, gives `{"model": ..., "error": <reason>, "cache": {...}}`; the step counts as failed (exit code 1, `partial` in a batch, with `prediction` in `batch_failed_steps`) and the other steps of the sample go on. The step is `{"skipped": true}` when the chains are unknown, or when `openfold` is not among `--metrics`. `provenance` gains `openfold3_checkpoint` (the checkpoint file name the of3 record names) and, when the pipeline started OpenFold3 itself, `openfold3_version` (below).

The adversarial check compares the input pose with the prediction. pLDDT and ipTM are calibrated per model, and `evobind_adversarial_score` divides by the pLDDT of the prediction, so compare scores between designs only when they come from one model. With `--predictor of3 --openfold-mode score` each chain of the input is a template, so the folds of the two chains agree with the input in part by construction; the pose is not templated, so `delta_com_angstrom` tests it. A prediction from another model, or `--openfold-mode refold`, is more independent: the binder comes from its sequence.

### Pipeline: the pre-flight check and `results["preflight"]`

`binding-metrics-run` and `binding-metrics-batch` check the input against what the requested steps and models can take before anything runs: before the output directory, preparation, relaxation, any model run and the prediction store. The limits are the `capabilities` of the registry entries and of the predictor adapters; see [`docs/preflight.md`](preflight.md) for what is declared and why.

| option | default | effect |
|--------|---------|--------|
| `--binder-type {auto,peptide,miniprotein,nanobody,antibody}` | `auto` | what the binder is, for the checks that depend on it; `auto` estimates it from the size (at most 40 residues a peptide, at most 100 a miniprotein, longer unknown, which skips the type checks) |
| `--on-incompatible {error,skip,warn}` | `error` | `error` refuses the sample and lists every problem with its fix; `skip` leaves out the incompatible steps, records why and runs the rest; `warn` logs the problems and runs everything. An output read with `--prediction-dir` only warns |
| `--preflight-only` | off | print the plan (what would run, what is incompatible and why) and stop; exit status 1 when `--on-incompatible` is `error` and something is refused |

The steps checked are the ones the run executes: the relaxation (`md_implicit`), `energy`, `interface`, the three metrics of `geometry` one by one, `electrostatics`, and the model step (`openfold` for the OpenFold3 step, `prediction` for `--predictor`, with the limits of the model). `dockq` is left alone: without a reference the pipeline already skips it and says so.

**`results["preflight"]`.**

| key | description |
|-----|-------------|
| `status` | `ok`, `warn` (a warning or, with `--on-incompatible warn`, a problem that was logged), `skipped` (steps left out), `refused` (batch error rows only: a refused `binding-metrics-run` raises before it writes anything) or `not_checked` (the input could not be profiled; the run goes on as before) |
| `reason` | the problems or warnings in one line, `subject: constraint: fact` joined with `; ` |
| `policy` | the `--on-incompatible` value |
| `mode` | the mode the model step ran in (`predict`, `refold`, `score`, `score-lock`), `null` when it is not stated (an output read with `--prediction-dir`) |
| `skipped_steps`, `skipped_geometry` | the steps (and the metrics of `geometry`) that were left out under `skip`, each with its reason; a left-out step is `{"skipped": true, "reason": ...}` in the results |
| `report` | the full report: the input profile (binder type, closures, residue classes), every violation with its fact, requirement, reason and fix, the warnings and the notes |

A batch row carries `preflight_status` and `preflight_reason`. A refused sample is an `error` row (`batch_error` holds the whole message) and does not stop the batch; a step left out under `skip` has `<step>_skipped` and, for a model step, `openfold_reason` or `prediction_reason`.

---

## 13. EvoBind scoring

`compute_evobind_score(structure_path, plddt_per_atom, binder_chain, receptor_chain=None, receptor_interface_residues=None, interface_cutoff_angstrom=8.0, *, target_chain=None)`, `compute_evobind_adversarial_check(design_structure_path, afm_structure_path, binder_chain, receptor_chain=None, afm_plddt_per_atom=None, interface_cutoff_angstrom=8.0, max_resname_mismatch_fraction=0.5, *, target_chain=None)` — `binding_metrics.metrics.evobind`

`compute_evobind_score_from_record(record, binder_chain, receptor_chain=None, *, receptor_interface_residues=None, interface_cutoff_angstrom=8.0, target_chain=None)`, `compute_evobind_adversarial_from_records(design, adversary, binder_chain, receptor_chain=None, *, interface_cutoff_angstrom=8.0, max_resname_mismatch_fraction=0.5, target_chain=None)` — same module; they take `PredictionRecord` objects (§12) and are not registry entries.

The interface-distance and confidence losses of Bryant et al. (2025). Gly and residues without Cβ use Cα. When `binding-metrics-run` runs the `openfold` metric, both functions run on the OpenFold3 output and their keys are merged into the OpenFold3 result; no extra model call is needed. With `--predictor` the same two computations run on the record of that model and their keys are merged into `results["prediction"]` (see the end of section 12). Registered as `evobind_score` and `evobind_adversarial`.

**What the scores divide by.** Both divide by the mean per-residue pLDDT of the binder in the structure that they score: `if_dist_pep_to_rec / (mean_plddt_binder / 100)` for the primary score, and `afm_mean_if_dist × (100 / afm_mean_plddt_binder) × ΔCOM` for the adversarial check, where the pLDDT is that of the second structure. Each model calibrates pLDDT differently, so compare a score between designs only when one model made every structure whose pLDDT it divides by. The interface distances and ΔCOM are geometry and compare across models.

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

A `plddt_per_atom` array whose length differs from the number of atoms in the structure raises a `ValueError` (#92); the same holds for `afm_plddt_per_atom`. A list of values is accepted. `receptor_interface_residues` names residue numbers: number 52 selects residues 52 and 52A.

### adversarial check

Compares a design pose with a second prediction, for example an OpenFold3 prediction of the same complex: the two structures are superposed on the receptor Cα atoms and the binder centres of mass are compared. Residues are paired by residue number and insertion code. When the two structures share too few of them (fewer than three receptor residues, no binder residue), the residues are paired by position up to the shorter chain. When the shared numbers pair mostly different residues (a second model that numbers every chain from 1; 1YCR has its receptor at 25-109), or a number and insertion code occurs twice in a chain, they are paired by position too, which needs the same number of Cα atoms in both chains, and the receptor interface residues then follow the positional pairing. A `ValueError` says why when neither pairing agrees in residue name (at most `max_resname_mismatch_fraction` of the pairs may differ; histidine and cysteine protonation variants, `HIN` included, count as equal) or when the chains differ in length.

| key | type | unit | description |
|-----|------|------|-------------|
| `delta_com_angstrom` | float | Å | binder centre-of-mass displacement after receptor Cα superposition |
| `n_superposition_residues` | int | — | receptor residues matched by residue number and insertion code (0 to 2 when the numberings barely overlap; 0 when position pairing replaced a pairing by number that named other residues or repeated a number) |
| `n_superposition_atoms` | int | — | receptor Cα atoms used for the superposition, whichever pairing applied |
| `receptor_pairing`, `binder_pairing` | str | — | `"residue_number"` or `"position"` |
| `receptor_resname_mismatch_fraction`, `binder_resname_mismatch_fraction` | float | — | fraction of paired residues with different names |
| `afm_if_dist_pep_to_rec`, `afm_if_dist_rec_to_pep`, `afm_mean_if_dist` | float | Å | interface distances in the second structure |
| `interface_fallback_used` | bool | — | as above |
| `afm_mean_plddt_binder` | float \| None | [0–100] | binder pLDDT in the second structure |
| `evobind_adversarial_score` | float \| None | — | `mean_if_dist × (100 / pLDDT) × ΔCOM`; lower = the second prediction agrees with the design pose |
| `reason` | str | — | pLDDT was given but its mean is zero or not finite |

A high pLDDT and a small interface distance with a large ΔCOM mean that the second prediction places the binder elsewhere on the receptor surface, which suggests that the design pose is not supported.

### scores from prediction records

The same computations with the structures read by a predictor adapter (§12), so the second prediction can come from any model that has one and its per-atom pLDDT travels in the record. `design` is a `PredictionRecord` or the path of the input pose; `adversary` is a `PredictionRecord`.

```python
from binding_metrics.metrics.evobind import (
    compute_evobind_adversarial_from_records,
    compute_evobind_score_from_record,
)
from binding_metrics.predictors import get_parser

of3 = get_parser("of3")
design = of3.load("design_scoring_out", "cmplx_007")
adversary = of3.load("sequence_only_out", "cmplx_007", chain_map={"R": "A"})
check = compute_evobind_adversarial_from_records(design, adversary, binder_chain="B", receptor_chain="A")
score = compute_evobind_score_from_record(adversary, binder_chain="B", receptor_chain="A")
```

Chain IDs are the user's IDs, after each record's `chain_map` (model chain ID to user chain ID). A model that calls the receptor `A` and another that calls it `R` therefore need no argument here, only the `chain_map` of their record. A chain that is not in a structure raises a `ValueError` that lists the chains it has.

`compute_evobind_score_from_record` returns the keys of `compute_evobind_score` plus `model`; a record without per-atom pLDDT gives the distances, `mean_plddt_binder` and `evobind_score` as None, and a `reason`. `compute_evobind_adversarial_from_records` returns the keys of the adversarial check, where the `afm_` prefix now means "the second prediction", plus two more:

| key | type | unit | description |
|-----|------|------|-------------|
| `design_model` | str \| None | — | `design.model`; None when `design` is a path |
| `adversary_model` | str | — | `adversary.model` |
| `reason` | str | — | the second prediction has no per-atom pLDDT (the geometric keys are returned, `afm_mean_plddt_binder` and `evobind_adversarial_score` are None, and the adapter's own reasons are appended), or its mean binder pLDDT is zero or not finite |

Two cautions apply to the second prediction:

- `evobind_adversarial_score` divides by the binder pLDDT of the second prediction, so compare scores between designs only when one adversary model made all the second predictions. `delta_com_angstrom` and the interface distances are geometry and compare across models.
- When the second prediction is an OpenFold3 run in score mode, each chain was given its own structure from the design as a template, so the folds of the chains agree with the design in part by construction. The relative pose of binder and receptor is not templated (OpenFold3 places the binder itself), so `delta_com_angstrom` tests the pose. A second prediction that is independent of the design is a sequence-only run (refold or predict).

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
| `prediction` | only with `--predictor` / `predictor=`: the prediction of any model with an adapter, its EvoBind keys and the store counters (§12, "Pipeline"); `openfold` is then `{"skipped": True}` |
| `preflight` | the decision of the pre-flight check: `status`, `reason`, `policy`, the steps left out and the full report (§12, "Pipeline: the pre-flight check"; [`preflight.md`](preflight.md)) |
| `nonfinite_fields` | paths of every NaN or infinite value in the file, for example `relax.rmsd_md_final` |

A metric that did not run is `{"skipped": True}`; one that failed is `{"error": message}`, and the command exits with 1 when any step failed.

**prep.** `output`, `ph`, `keep_water`, plus the report of preparation: `removed_heterogens` (list of `"NAME (chain X)"`), `n_removed_waters`, `kept_nonstandard` (non-standard residues and metal ions that were kept), `n_missing_atoms_rebuilt`, `n_missing_residue_gaps` and `chain_breaks`. Each entry of `chain_breaks` is `{"chain", "residue_before", "residue_after", "c_n_distance_angstrom"}`: two consecutive residues of a chain whose C and N atoms are more than 2.0 Å apart in the input. Prep logs a warning for each and leaves them as they are; the relaxation bonds the two residues by name and closes the gap. The chain IDs in `removed_heterogens`, `kept_nonstandard` and `chain_breaks` are author IDs (`auth_asym_id` in a CIF), as everywhere else a chain is named. OpenMM names the chains of a CIF by their label IDs when the file has more label IDs than author IDs, so the peptide of 1CWA is chain B in its topology and author chain C; `load_structure` records the author ID of each chain on the topology (see [§19](#19-io-utilities)) and prep reports chain C. 3P8F: the ligand GSH has label ID C and author ID A, and is listed as `GSH (chain A)`. A PDB input has one set of IDs. For a cyclic peptide with residues that needed a GAFF template, `ncaa_bond_order_source` says where each residue's bond orders came from: `"ccd"` (Chemical Component Dictionary) or `"single_bonds"` (fallback, see [`nonstandard.md`](nonstandard.md)).

**relax.** `RelaxationResult.to_dict()`, plus `elapsed_s`. The relaxation expects a prepared structure: with `--skip-prep` on a raw file, a chain that ends in a standard residue without its terminal oxygen (OXT), with no cap and no closure bond, fails before the force field with `success` False and an `error_message` such as `chain A ends in VAL244 without its terminal oxygen (OXT): the structure is not prepared ...` (3P8F, 1YCR, 1QJB and 1XY4 raw; 1CWA and 3V3B raw have no such chain). The interaction energy checks the same way:

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
| `peptide_cyclic_bonds` | — | closure bonds detected in the binder, each as `{"type", "atom1", "atom2"}` with atoms written `author chain:residue number:atom name` (`C:11:C` for the head-to-tail amide of 1CWA) |
| `dropped_protein_chains` | — | author IDs of the protein chains, other than the binder and the target, that the relaxation removed (empty list when none) |
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

Four optional keys describe a run that used OpenFold3 and are absent otherwise (adding them needs no schema bump). `openfold3_version` is the installed `openfold3` version (None when it is not installed or unreadable), added when the pipeline starts OpenFold3: the `openfold` step without `--predictor`, or `--predictor of3` without `--prediction-dir`; the environment of `--openfold-conda-env` is asked through `conda run -n <env> python`, and `collect_provenance(openfold3=True, openfold3_python_cmd=...)` adds it in your own code. It is not recorded for an output that was read from disk, because the installed version is not the one that made it. `openfold3_checkpoint` is the checkpoint file name that the record of the prediction names (`inference_ckpt_name` of OpenFold3's `experiment_config.json`), added by `--predictor` when the record has it. `openfold3_use_msa_server` (bool) is whether the run used the ColabFold MSA server (`--openfold-no-msa-server` makes it False), added together with `openfold3_version`. `prediction_weights` is the `weights` dictionary of the prediction step (see "custom weights" under the prediction store), added when the step knows the weights: custom ones, or the checkpoint that OpenFold3 recorded for its default weights. A batch writes `provenance_openfold3_version` and `provenance_openfold3_use_msa_server` for the samples OpenFold3 predicted, and `provenance_prediction_weights_sha256`, `..._path` and the other entries of the weights.

**batch rows.** A row holds `sample_id`, `input`, the flattened results, `batch_status` and the `provenance_*` columns. `batch_status` is `ok` (every step completed), `partial` (the pipeline finished but a step failed; see `batch_failed_steps` and `batch_failed_reasons`) or `error` (the worker raised; see `batch_error`). `run_batch` returns the rows in the order of its input paths. List and array fields (per-atom pLDDT, the PAE and PDE matrices, per-residue pLDDT) are not columns: they stay in the per-sample JSON. With `--predictor` the row holds the `prediction_*` columns of `results["prediction"]` (for example `prediction_avg_plddt`, `prediction_iptm`, `prediction_evobind_score`, `prediction_cache_runs`, `prediction_cache_request_key`).

---

## 16. metric registry

`binding_metrics.metrics.registry` describes each metric function without importing it, so that a generic caller can drive them: `METRICS` lists the `MetricSpec` entries, `get_metric(name)` returns one and `metrics_by_input_type(input_type)` filters. The functions are the API; the registry is the adapter. `spec.call(**kwargs)` imports the function on first use and calls it with exactly the given keywords.

fields of a `MetricSpec`:

| field | meaning |
|-------|---------|
| `name`, `import_path`, `description` | identifier, `"module:function"`, one line |
| `input_type` | `static_structure` (one PDB or CIF file), `trajectory` (trajectory and topology), `md_simulation` (one structure, runs a relaxation or MD protocol), `openfold_json` (OpenFold3 output), `atom_array` (a loaded `biotite.structure.AtomArray`), `predicted_structure` (a predicted structure path plus per-atom confidence arrays), `prediction_dir` (the output directory of a model that has an adapter; the caller also gives `model` and `name`) |
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
| `prediction` | `compute_prediction_metrics` | prediction_dir | — | — | — | model |  |
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

- `load_structure(path) → (topology, positions)`: reads a `.pdb`, `.cif` or `.mmcif` file into an OpenMM topology and positions. OpenMM names the chains of a CIF by `label_asym_id` when the file has more label IDs than author IDs, and the topology keeps those IDs (the waters of an author chain would merge with its protein if they were renamed). `load_structure` records the author ID (`auth_asym_id`) of every chain on the topology, by matching `_atom_site.id` with gemmi; without gemmi it logs a message and the author IDs are taken to be the topology's own. 1CWA: topology chains A (protein), B (peptide), C and D (waters) are author chains A, C, A, C; 3P8F: A, B, C (GSH), D, E are author chains A, I, A, A, I.
- `author_chain_ids(topology) → list[str]`: the author ID of every chain, in chain order; the chain's own ID for a topology without them (a PDB file).
- `openmm_chain_id(topology, author_id) → str | None`: the ID, in the topology, of the amino-acid chain that the author ID names. A chain that holds only waters, ions or ligands is never the answer while a protein chain of that author ID exists (1CWA: `"C"` is chain B, not the water chain C; `"A"` is chain A). None when no amino-acid chain has the author ID; `ValueError` when its amino-acid residues sit in chains with different IDs in the topology.
- `attach_author_chain_ids(topology, source_path) → bool` and `copy_author_chain_ids(source, target)`: record the author IDs on a topology read from `source_path` (what `load_structure` does), and carry them to a topology built from another one with `Modeller` (`strip_heterogens`, `drop_other_protein_chains` and `prep_structure` do). A record that no longer fits the chains is ignored.
- Chain arguments of `strip_heterogens`, `drop_other_protein_chains`, `patch_cyclic_topology` and `detect_chains` are IDs in the topology: the relaxation, the energy and prep hold such IDs, and an author ID of the same letter would shadow them in a file whose label and author letters are swapped. `detect_cyclization`, `detect_nonstandard`, `patch_nonstandard`, `restore_nonstandard_names` (its `chain_id` argument), `compute_interaction_energy` and `RelaxationConfig.peptide_chain_id` and `receptor_chain_id` receive the caller's ID and read it as an author ID first and as an ID in the topology otherwise (`topology_chain_id(topology, chain_id)`). A result that names a chain of a topology, such as `NonstandardInfo.chain_id` or the entries of `detect_cyclization`, carries the ID in the topology.
- `detect_chains(topology) → (peptide_chain, receptor_chain)`: smallest and largest protein chain of an OpenMM topology, by the number of amino-acid residues (the residue set is listed under chain auto-detection in [§1](#1-conventions)); a single chain gives `(chain, None)`.
- `detect_chains_from_file(path, peptide_chain=None, receptor_chain=None, verbose=True) → dict`: biotite-based detection and the function the pipeline uses. For two protein chains the smaller is the peptide and the larger the receptor. For more than two, the receptor is the chain with the most Cα atoms within 8 Å of the peptide, not the largest. Explicit chain IDs are returned as given, even when they are not in the file (the pipeline checks them and raises `ChainNotFoundError`).
- `save_cif(topology, positions, output_path, source_cif_path=None)`: writes a CIF and, given a source CIF, restores the author chain IDs and residue numbers. Without gemmi it logs a warning and keeps OpenMM's sequential IDs. Without a source CIF it writes the topology's own chain IDs and residue numbers when the chain IDs are unique and alphanumeric and the residue numbers are integers, so relaxing a PDB input writes the chain IDs of the input; otherwise it uses OpenMM's letters and sequential numbers.
- `drop_other_protein_chains(topology, positions, peptide_chain, receptor_chain, report=None)`: deletes every protein chain other than the two named, after `strip_heterogens`; nothing is removed unless both chains are given and present. `report["dropped_protein_chains"]` receives the removed IDs, as author IDs.

`detect_chains_from_file` returns:

| key | description |
|-----|-------------|
| `peptide_chain`, `receptor_chain` | author chain IDs (`auth_asym_id` in a CIF); `receptor_chain` is None for a file with one protein chain |
| `peptide_chain_label`, `receptor_chain_label` | the IDs as OpenMM names the chains: the label ID (`label_asym_id`) when the file has more label IDs than author IDs (waters and ligands each get a label ID), the author ID otherwise; equal to the author IDs for PDB files. The pipeline hands every step the author IDs (the energy step and the relaxation find the chains in the OpenMM topology from them); these keys serve code that reads the file with OpenMM itself |
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
