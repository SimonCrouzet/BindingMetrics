# metrics reference

Details on every metric computed by BindingMetrics: function signatures, output keys, units, and algorithm notes.

Run everything in one shot:

```bash
binding-metrics-run --input complex.cif --output-dir results/
```

---

## table of contents

1. [interface geometry](#1-interface-geometry)
2. [hydrogen bonds & salt bridges](#2-hydrogen-bonds--salt-bridges)
3. [electrostatics](#3-electrostatics)
4. [force-field interaction energy](#4-force-field-interaction-energy)
5. [ramachandran & omega planarity](#5-ramachandran--omega-planarity)
6. [shape complementarity](#6-shape-complementarity)
7. [buried void volume](#7-buried-void-volume)
8. [structure comparison (RMSD)](#8-structure-comparison-rmsd)
9. [MD receptor drift](#9-md-receptor-drift)
10. [OpenFold3 confidence scores](#10-openfold3-confidence-scores)
11. [EvoBind scoring](#11-evobind-scoring)
12. [scorecard thresholds](#12-scorecard-thresholds)

---

## 1. interface geometry

`compute_interface_metrics(cif_path, design_chain=None, receptor_chain=None)` — `binding_metrics.metrics.interface`

PISA-inspired interface descriptors on a static structure. Uses biotite's Lee-Richards SASA calculator (probe 1.4 Å, 960 points/atom).

| key | type | unit | description |
|-----|------|------|-------------|
| `peptide_chain` | str | — | resolved peptide chain ID |
| `receptor_chain` | str | — | resolved receptor chain ID |
| `delta_sasa` | float | Å² | buried SASA = SASA(pep) + SASA(rec) − SASA(complex) |
| `sasa_peptide` | float | Å² | SASA of isolated peptide |
| `sasa_receptor` | float | Å² | SASA of isolated receptor |
| `sasa_complex` | float | Å² | SASA of complex |
| `delta_g_int` | float | kcal/mol | solvation binding energy (Eisenberg-McLachlan); more negative = more favorable |
| `delta_g_int_kJ` | float | kJ/mol | same × 4.184 |
| `polar_area` | float | Å² | buried area from N/O atoms |
| `apolar_area` | float | Å² | buried area from C/S atoms |
| `fraction_polar` | float | — | `polar_area / delta_sasa` |
| `n_interface_residues_peptide` | int | — | residues contributing ≥ 0.5 Å² buried SASA |
| `n_interface_residues_receptor` | int | — | same for receptor |
| `interface_residues_peptide` | list[str] | — | `"RES:CHAIN:NUM"` labels |
| `interface_residues_receptor` | list[str] | — | `"RES:CHAIN:NUM"` labels |
| `per_residue` | list[dict] | — | per interface residue: `residue`, `chain`, `res_name`, `res_id`, `buried_sasa` (Å²), `delta_g_res` (kcal/mol), `polar_area` (Å²), `apolar_area` (Å²) |
| `hbonds` | int | — | cross-chain hydrogen bonds |
| `saltbridges` | int | — | cross-chain salt bridges |

**solvation energy (Eisenberg-McLachlan 1986):** ΔG_int = Σᵢ γᵢ · ΔAᵢ

| atom | γ (kcal/mol/Å²) |
|------|-----------------|
| C | −0.016 |
| N | +0.063 |
| O | +0.024 |
| S | −0.021 |

typical tight binders: ΔG_int −5 to −20 kcal/mol; ΔSASA > 1000 Å².

---

## 2. hydrogen bonds & salt bridges

Called internally by `compute_interface_metrics`; also in `binding_metrics.metrics.hbonds`.

**hydrogen bonds**
- detector: biotite `struc.hbond()`, optional explicit H via `hydride`
- criteria: donor–acceptor distance < 3.5 Å, D–H···A angle > 120°
- cross-chain pairs only

**salt bridges**

| atom | charge |
|------|--------|
| LYS NZ | +1.0 |
| ARG NH1 | +0.5 |
| ARG NH2 | +0.5 |
| ASP OD1 | −0.5 |
| ASP OD2 | −0.5 |
| GLU OE1 | −0.5 |
| GLU OE2 | −0.5 |

distance criterion: 0.5 Å < r < 5.5 Å between opposite-charge atoms across chains.

---

## 3. electrostatics

`compute_coulomb_cross_chain(cif_path, peptide_chain=None, receptor_chain=None, dielectric=4.0, cutoff_ang=12.0)` — `binding_metrics.metrics.electrostatics`

| key | type | unit | description |
|-----|------|------|-------------|
| `coulomb_energy_kJ` | float | kJ/mol | total cross-chain Coulomb energy; negative = net attractive |
| `coulomb_energy_kcal` | float | kcal/mol | same / 4.184 |
| `n_charged_pairs` | int | — | charged atom pairs within cutoff |
| `n_attractive` | int | — | opposite-sign pairs within cutoff |
| `n_repulsive` | int | — | same-sign pairs within cutoff |
| `charged_atoms_peptide` | list[dict] | — | per atom: `residue`, `atom`, `charge`, `coords` |
| `charged_atoms_receptor` | list[dict] | — | same for receptor |

formula: E = (k / ε) · Σ q_i q_j / r_ij, with k = 1389.35 kJ·Å/mol·e², ε = 4.0 (default), cutoff = 12 Å.

charges follow the same table as §2; ARG split across NH1/NH2 to reflect resonance (sum = +1.0).

---

## 4. force-field interaction energy

`compute_interaction_energy(cif_path, peptide_chain=None, receptor_chain=None, modes=("relaxed",), device="cuda", ph=7.4, solvent_model="obc2")` — `binding_metrics.metrics.energy`

E_int = E_complex − E_peptide − E_receptor with AMBER ff14SB + implicit solvent.

| key | type | unit | description |
|-----|------|------|-------------|
| `sample_id` | str | — | identifier |
| `num_contacts` | int | — | heavy-atom pairs < 8.0 Å across chains |
| `num_close_contacts` | int | — | heavy-atom pairs < 4.0 Å across chains |
| `raw_interaction_energy` | float | kJ/mol | E_int at H-added input geometry |
| `raw_e_complex` | float | kJ/mol | E_complex (raw) |
| `raw_e_peptide` | float | kJ/mol | E_peptide (raw) |
| `raw_e_receptor` | float | kJ/mol | E_receptor (raw) |
| `relaxed_interaction_energy` | float | kJ/mol | E_int after minimization |
| `relaxed_e_complex` | float | kJ/mol | E_complex (relaxed) |
| `relaxed_e_peptide` | float | kJ/mol | E_peptide (relaxed) |
| `relaxed_e_receptor` | float | kJ/mol | E_receptor (relaxed) |
| `after_md_interaction_energy` | float | kJ/mol | E_int after minimization + MD |
| `after_md_e_complex` | float | kJ/mol | E_complex (after MD) |
| `after_md_e_peptide` | float | kJ/mol | E_peptide (after MD) |
| `after_md_e_receptor` | float | kJ/mol | E_receptor (after MD) |

keys for modes not requested are absent from the dict.

**modes**

| mode | protocol |
|------|----------|
| `raw` | add H → single-point energy; may return `None` for severe clashes |
| `relaxed` | add H → 500-step backbone-restrained + 2000-step unrestrained minimization |
| `after_md` | same as `relaxed` + NVT MD at 300 K (default 200 ps, 2 fs timestep) |

**implementation notes**
- force field: AMBER ff14SB (`amber14-all.xml`)
- implicit solvent: OBC2 (`implicit/obc2.xml`) or GBn2
- non-bonded cutoff: NoCutoff (all pairwise interactions)
- cyclic peptides: head-to-tail N→C bond patched via custom XML templates; CYS→CYX rename in-memory for SS-bonded residues before `addHydrogens`
- subsystem decomposition: each chain extracted separately; orphaned CYX (cross-chain SS severed) converted back to CYS+HG before template matching

E_int is a gas-phase quantity — solvation and entropy are excluded. Use for within-campaign ranking.

---

## 5. ramachandran & omega planarity

`compute_ramachandran(cif_path, chain=None)`, `compute_omega_planarity(cif_path, chain=None)` — `binding_metrics.metrics.geometry`

### ramachandran

| key | type | unit | description |
|-----|------|------|-------------|
| `ramachandran_favoured_pct` | float | % | residues in favoured φ/ψ regions |
| `ramachandran_allowed_pct` | float | % | residues in allowed regions |
| `ramachandran_outlier_pct` | float | % | residues in disallowed regions |
| `ramachandran_outlier_count` | int | — | absolute count of outliers |
| `n_residues_evaluated` | int | — | residues with complete backbone (excl. termini) |
| `n_d_residues` | int | — | D-amino acid count |
| `per_residue` | list[dict] | — | `res_id`, `res_name`, `chain`, `phi` (°), `psi` (°), `is_d_aa`, `region` |

favoured regions (L-amino acids):

| region | φ (°) | ψ (°) |
|--------|-------|-------|
| α-helix | −90 to −30 | −80 to +10 |
| β-sheet | −180 to −45 | 90–180 or −180 to −160 |
| PPII | −90 to −50 | 120–180 |
| left-handed helix | +20 to +90 | 0 to +85 |

D-amino acids: φ/ψ are negated before lookup. Cyclic and D-peptides naturally accumulate more outliers — a 🟡 here should not be penalised for macrocyclic designs.

### omega planarity

| key | type | unit | description |
|-----|------|------|-------------|
| `omega_mean_dev` | float | ° | mean \|ω − 180°\| across all peptide bonds |
| `omega_max_dev` | float | ° | maximum deviation |
| `omega_outlier_fraction` | float | — | fraction with \|dev\| > 15° |
| `omega_outlier_count` | int | — | absolute count |
| `n_bonds_evaluated` | int | — | total bonds assessed |
| `per_residue` | list[dict] | — | `res_id`, `res_name`, `chain`, `omega` (°), `deviation` (°), `is_outlier` |

cis-Pro bonds (ω ≈ 0°) are handled; deviation is measured from the nearest of 0° or 180°.

---

## 6. shape complementarity

`compute_shape_complementarity(cif_path, design_chain=None, receptor_chain=None)` — `binding_metrics.metrics.geometry`

Lawrence & Colman (1993) Sc score.

| key | type | unit | description |
|-----|------|------|-------------|
| `sc` | float | [−1, 1] | overall Sc (mean of both directions) |
| `sc_A_to_B` | float | [−1, 1] | median score from peptide surface dots → receptor |
| `sc_B_to_A` | float | [−1, 1] | median score from receptor surface dots → peptide |
| `n_surface_dots_A` | int | — | interface surface dots on peptide |
| `n_surface_dots_B` | int | — | interface surface dots on receptor |

algorithm: 50 Fibonacci-sphere dots/atom, trimmed by VDW; interface = within 5 Å of opposite chain; score S_i = Σ_j exp(−d²_ij / 49) · (n̂_A,i · (−n̂_B,j)); chain score = median over dots.

typical values: 0.6–0.9 well-packed; 0.0–0.4 poor/flat interface.

---

## 7. buried void volume

`compute_buried_void_volume(cif_path, design_chain=None, receptor_chain=None, grid_spacing=0.5, probe_radius=1.4)` — `binding_metrics.metrics.geometry`

| key | type | unit | description |
|-----|------|------|-------------|
| `void_volume_A3` | float | Å³ | total void volume at interface; lower = better packing |
| `void_grid_fraction` | float | — | void voxels / bounding-box voxels |
| `interface_box_volume_A3` | float | Å³ | bounding-box volume of interface region |
| `n_interface_atoms` | int | — | atoms used to define the interface |

algorithm: 0.5 Å voxel grid over interface (atoms within 5 Å of opposite chain); occupied = within atom VDW + 1.4 Å probe; exterior from boundary flood-fill; void = unoccupied and not exterior.

---

## 8. structure comparison (RMSD)

`compute_structure_rmsd(ref_path, query_path, design_chain=None)` — `binding_metrics.metrics.comparison`

Kabsch-aligned RMSD, atoms matched by (chain, residue number, atom name).

| key | type | unit | description |
|-----|------|------|-------------|
| `rmsd` | float | Å | all-atom RMSD of full complex |
| `bb_rmsd` | float | Å | backbone RMSD (N, CA, C, O) of full complex |
| `rmsd_design` | float | Å | all-atom RMSD of design chain only |
| `bb_rmsd_design` | float | Å | backbone RMSD of design chain only |

water excluded; unmatched atoms silently skipped.

---

## 9. MD receptor drift

`compute_receptor_drift(traj_path, top_path, receptor_chain=None)` — `binding_metrics.metrics.rmsd`

Receptor backbone stability over an MD trajectory (MDTraj).

| key | type | unit | description |
|-----|------|------|-------------|
| `drift_aligned_mean` | float | Å | mean per-frame receptor Cα RMSD after superposition (conformational drift) |
| `drift_aligned_max` | float | Å | maximum aligned drift |
| `drift_raw_mean` | float | Å | mean absolute Cα displacement; NaN if PBC detected |
| `drift_raw_max` | float | Å | maximum raw drift; NaN if PBC detected |
| `pbc_detected` | bool | — | periodic boundary conditions present |
| `drift_aligned_per_frame` | np.ndarray | Å | per-frame aligned drift (shape: n_frames) |
| `drift_raw_per_frame` | np.ndarray | Å | per-frame raw drift; NaN if PBC |
| `n_receptor_ca` | int | — | receptor Cα atoms used |
| `n_frames` | int | — | trajectory frames |

aligned drift superimposes receptor Cα onto frame 0 before computing RMSD, isolating conformational change from rigid-body motion.

---

## 10. OpenFold3 confidence scores

`compute_openfold_metrics(output_dir, query_name, binder_chain=None, receptor_chain=None, seed=1, sample=1, reference_structure_path=None, include_matrices=False)` — `binding_metrics.metrics.openfold`

Parses OF3 JSON + CIF output files.

| key | type | unit | description |
|-----|------|------|-------------|
| `structure_path` | str \| None | — | path to predicted structure |
| `avg_plddt` | float | [0–100] | mean per-atom pLDDT |
| `binder_avg_plddt` | float \| None | [0–100] | mean pLDDT of binder chain |
| `binder_plddt_per_residue` | np.ndarray \| None | [0–100] | per-residue mean pLDDT of binder |
| `gpde` | float | Å | global predicted distance error |
| `ptm` | float | [0–1] | predicted TM-score |
| `iptm` | float | [0–1] | interface pTM; NaN if single-chain |
| `disorder` | float | [0–1] | mean relative SASA (disorder proxy) |
| `has_clash` | float | 0 or 1 | steric clash in prediction |
| `sample_ranking_score` | float | — | OF3 composite ranking score |
| `chain_ptm` | dict | — | per-chain pTM: `{chain_id: float}` |
| `chain_pair_iptm` | dict | — | pairwise ipTM: `{(A, B): float}` |
| `mean_interface_pde` | float | Å | mean PDE over binder × receptor token pairs |
| `max_interface_pde` | float | Å | max interface PDE |
| `plddt_per_atom` | np.ndarray \| None | [0–100] | per-atom pLDDT (shape: n_atoms) |
| `binder_ca_rmsd` | float \| None | Å | binder Cα RMSD vs reference (refolding mode) |
| `pde` | np.ndarray \| None | Å | full PDE matrix (n_tokens × n_tokens); only if `include_matrices=True` |
| `pde_interface` | np.ndarray \| None | Å | binder × receptor PDE slice; only if `include_matrices=True` |
| `timing` | dict | — | runtime from `timing.json` |

interpretation: pLDDT > 80 high confidence; ipTM > 0.75 reliable complex; interface PDE < 2 Å good; binder_ca_rmsd < 2 Å good pose reproduction.

---

## 11. EvoBind scoring

`compute_evobind_score(structure_path, plddt_per_atom=None, binder_chain=None, receptor_chain=None)` — `binding_metrics.metrics.evobind`

`compute_evobind_adversarial_check(design_structure_path, afm_structure_path, binder_chain=None, receptor_chain=None, afm_plddt_per_atom=None)`

Bryant et al. (2025) interface-distance + confidence losses.

### primary score

| key | type | unit | description |
|-----|------|------|-------------|
| `if_dist_pep_to_rec` | float | Å | mean min-distance from each binder Cβ to receptor interface Cβ |
| `if_dist_rec_to_pep` | float | Å | mean min-distance from each receptor interface Cβ to binder Cβ |
| `if_dist_symmetric` | float | Å | mean of both above |
| `n_interface_receptor_residues` | int | — | receptor residues within 8 Å of binder |
| `mean_plddt_binder` | float \| None | [0–100] | mean per-residue pLDDT of binder |
| `evobind_score` | float \| None | Å | `if_dist_pep_to_rec / (mean_plddt / 100)`; **lower is better** |

### adversarial check

| key | type | unit | description |
|-----|------|------|-------------|
| `delta_com_angstrom` | float | Å | binder CoM displacement after receptor Cα superposition |
| `n_superposition_residues` | int | — | receptor residues used for Kabsch alignment |
| `afm_if_dist_pep_to_rec` | float | Å | EvoBind if_dist on OF3 prediction |
| `afm_if_dist_rec_to_pep` | float | Å | same, reverse direction |
| `afm_mean_if_dist` | float | Å | mean of both |
| `afm_mean_plddt_binder` | float \| None | [0–100] | binder pLDDT on OF3 prediction |
| `evobind_adversarial_score` | float \| None | — | `mean_if_dist × (100 / pLDDT) × ΔCOM`; **lower = OF3 agrees with design pose** |

Gly and residues without Cβ use Cα. The adversarial score detects hallucinations: a high pLDDT + small if_dist but large ΔCOM means OF3 places the binder elsewhere on the receptor surface.

---

## 12. scorecard thresholds

The `--summary` flag produces a RAG (🟢/🟡/🔴) scorecard. Thresholds are in `src/binding_metrics/protocols/report.py`; full rationale in [`report_thresholds.md`](report_thresholds.md).

| metric | 🟢 | 🟡 | 🔴 |
|--------|----|----|-----|
| E_int (kJ/mol) | < −40 | −40 to 0 | > 0 |
| ΔSASA (Å²) | > 1000 | 500–1000 | < 500 |
| H-bonds | ≥ 5 | 2–4 | < 2 |
| salt bridges | ≥ 2 | 1 | 0 |
| Coulomb (kJ/mol) | < −100 | −100 to 0 | > 0 |
| Ramachandran favoured (%) | > 95 | 80–95 | < 80 |
| ω outlier fraction | < 0.05 | 0.05–0.20 | > 0.20 |
| MD RMSD (Å) | < 2 | 2–5 | > 5 |
| RMSF mean (Å) | < 1 | 1–2 | > 2 |

---

## unit summary

| quantity | unit |
|----------|------|
| SASA, RMSD, distances | Å |
| `delta_g_int` (Eisenberg-McLachlan) | **kcal/mol** |
| all other energies (OpenMM, Coulomb) | **kJ/mol** |
| `coulomb_energy_kcal` | kcal/mol (convenience alias) |
| angles (φ, ψ, ω) | degrees |
| pLDDT | [0–100] |
| pTM, ipTM | [0–1] |
| void volume | Å³ |

---

## references

- Eisenberg & McLachlan (1986). *Nature* 319, 199–203.
- Krissinel & Henrick (2007). *J. Mol. Biol.* 372, 774–797.
- Lawrence & Colman (1993). *J. Mol. Biol.* 234, 946–950.
- Eastman et al. (2017). OpenMM 7. *PLOS Comput. Biol.* 13, e1005659.
- Ahdritz et al. (2024). OpenFold3. https://github.com/aqlaboratory/openfold-3
- Bryant et al. (2025). EvoBind. *Communications Chemistry*. https://doi.org/10.1038/s42004-025-01601-3
