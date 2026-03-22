# BindingMetrics — Detailed Metrics Reference

Full API documentation, return value schemas, algorithm details, and CLI options for all metrics. For installation and quick start, see [`README.md`](README.md).

---

## Table of Contents

1. [Interface analysis](#1-interface-analysis)
2. [Coulomb electrostatics](#2-coulomb-electrostatics)
3. [Backbone geometry](#3-backbone-geometry)
4. [Interface geometry](#4-interface-geometry)
5. [Force-field energy](#5-force-field-energy)
6. [Structure comparison (RMSD)](#6-structure-comparison-rmsd)
7. [Trajectory analysis](#7-trajectory-analysis)
8. [OpenFold3 confidence metrics](#8-openfold3-confidence-metrics)
9. [Score vs. feature taxonomy](#9-score-vs-feature-taxonomy)

---

## 1. Interface analysis

**Backend:** biotite + hydride (optional) + scipy
**Install:** `pip install "binding-metrics[biotite]"`

### `compute_interface_metrics(cif_path, design_chain=None, receptor_chain=None, probe_radius=1.4, interface_threshold=0.5) → dict`

Full interface characterisation for a CIF structure. Computes per-atom buried SASA and derives thermodynamic and structural metrics.

Chain auto-detection: smallest protein chain = peptide, largest = receptor.

**Solvation energy** uses Eisenberg-McLachlan atomic solvation parameters:

| Element | γ (kcal/mol/Å²) | Character |
|---|---|---|
| C | −0.016 | hydrophobic (burial favorable) |
| N | +0.063 | polar (burial unfavorable) |
| O | +0.024 | polar (burial unfavorable) |
| S | −0.021 | weakly hydrophobic |

```
ΔG_int = Σ_i  γ_i × ΔA_i
```

**Returns:**

```python
{
    "peptide_chain": str,
    "receptor_chain": str,

    # SASA (Å²)
    "delta_sasa": float,           # total buried SASA — score (larger = more buried)
    "sasa_peptide": float,
    "sasa_receptor": float,
    "sasa_complex": float,

    # Solvation energy
    "delta_g_int": float,          # kcal/mol — score (negative = hydrophobic-driven, favorable)
    "delta_g_int_kJ": float,

    # Interface composition
    "polar_area": float,           # Å²
    "apolar_area": float,          # Å²
    "fraction_polar": float,       # feature

    # Interface residues
    "n_interface_residues_peptide": int,
    "n_interface_residues_receptor": int,
    "interface_residues_peptide": list[str],   # "RES:CHAIN:NUM"
    "interface_residues_receptor": list[str],

    # Per-residue details
    "per_residue": list[dict],     # feature: buried_sasa, delta_g_res, polar_area, apolar_area

    # Interactions
    "hbonds": int,                 # score (more = better)
    "saltbridges": int,            # score (more = better)
}
```

**Interpretation:**
- Δ*G*_int < −5 kcal/mol: likely stable, hydrophobically-driven interface.
- Δ*G*_int > 0: polar burial dominates — may indicate a weak or crystal-packing contact.

### `compute_hbonds(atoms, peptide_chain, receptor_chain) → int`

Count cross-chain hydrogen bonds using biotite's H-bond detector. Adds explicit hydrogens with hydride if available.

### `compute_saltbridges(atoms, peptide_chain, receptor_chain, distance_min=0.5, distance_max=5.5) → int`

Count cross-chain salt bridges by distance between oppositely-charged atoms:
- Positive: LYS NZ, ARG NH1/NH2, HIS ND1/NE2
- Negative: ASP OD1/OD2, GLU OE1/OE2

### `compute_delta_sasa_static(cif_path, peptide_chain, receptor_chain, probe_radius=1.4) → dict`

Lightweight SASA-only calculation (totals, no per-atom decomposition).

### CLI: `binding-metrics-interface`

```bash
binding-metrics-interface --input complex.cif [--design-chain A] [--receptor-chain B]
    [--probe-radius 1.4] [--threshold 0.5]
```

---

## 2. Coulomb electrostatics

**Backend:** biotite + scipy
**Install:** `pip install "binding-metrics[biotite]"`

### `compute_coulomb_cross_chain(cif_path, peptide_chain=None, receptor_chain=None, dielectric=4.0, cutoff_ang=12.0) → dict`

Pairwise Coulomb interaction energy between formally-charged atoms on the peptide and receptor within `cutoff_ang`. Protein interior dielectric ε_r = 4 (default).

**Formal partial charges:**

| Residue | Atom(s) | Charge |
|---|---|---|
| LYS | NZ | +1 |
| ARG | NH1, NH2 | +0.5 each |
| ASP | OD1, OD2 | −0.5 each |
| GLU | OE1, OE2 | −0.5 each |

```
E = Σ_{i∈peptide, j∈receptor}  q_i · q_j / (ε_r · r_ij)  ×  1389.35 kJ·Å/mol
```

**Returns:**

```python
{
    "coulomb_energy_kJ": float,     # score (negative = net attractive)
    "coulomb_energy_kcal": float,

    "n_charged_pairs": int,         # feature
    "n_attractive": int,            # feature
    "n_repulsive": int,             # feature
    "charged_atoms_peptide": int,   # feature
    "charged_atoms_receptor": int,  # feature
}
```

**Interpretation:** Negative energy = net attractive electrostatics. Useful alongside ΔG_int (which captures hydrophobic burial) to characterise the electrostatic component of binding.

### CLI: `binding-metrics-electrostatics`

```bash
binding-metrics-electrostatics --input complex.cif [--design-chain A] [--receptor-chain B]
    [--dielectric 4.0] [--cutoff 12.0]
```

---

## 3. Backbone geometry

**Backend:** biotite
**Install:** `pip install "binding-metrics[biotite]"`

### `compute_ramachandran(cif_path, chain=None) → dict`

Classify backbone (φ, ψ) dihedrals using simplified rectangular Ramachandran regions. Glycine and proline are excluded (non-standard φ/ψ landscape).

**Region definitions (approximate):**

| Region | φ range | ψ range |
|---|---|---|
| Favoured α-helix | −80° to −40° | −60° to −20° |
| Favoured β-sheet | −160° to −60° | 100° to 180° |
| Favoured left-handed | 40° to 80° | 20° to 60° |
| Allowed | expanded ±20° around favoured | |
| Outlier | everything else | |

> **Note:** These are rectangular approximations. For publication-quality validation, compare against MolProbity or PROCHECK.

**Returns:**

```python
{
    "ramachandran_favoured_pct": float,   # score (higher = better)
    "ramachandran_allowed_pct": float,
    "ramachandran_outlier_pct": float,    # score (lower = better)
    "ramachandran_outlier_count": int,    # score (lower = better)
    "n_residues_evaluated": int,

    "per_residue": list[dict],            # feature: chain, res_num, res_name, phi_deg, psi_deg, region
}
```

### `compute_omega_planarity(cif_path, chain=None) → dict`

Measure peptide bond ω-angle deviation from ideal planarity (180°). Outlier threshold: 15°.

```
dev = min(|ω − 180°|, |ω + 180°|)
```

**Returns:**

```python
{
    "omega_mean_dev": float,          # score (lower = better planarity)
    "omega_max_dev": float,           # score
    "omega_outlier_fraction": float,  # score (lower = better)
    "omega_outlier_count": int,       # score
    "n_bonds_evaluated": int,

    "per_residue": list[dict],        # feature: chain, res_num, res_name, omega_deg, deviation_deg, is_outlier
}
```

### CLI: `binding-metrics-geometry`

```bash
binding-metrics-geometry --metric ramachandran --input complex.cif [--chain A]
binding-metrics-geometry --metric omega --input complex.cif [--chain A]
```

---

## 4. Interface geometry

**Backend:** biotite + scipy
**Install:** `pip install "binding-metrics[biotite]"`

### `compute_shape_complementarity(cif_path, peptide_chain=None, receptor_chain=None, n_dots=50, interface_cutoff=5.0, sigma=7.0) → dict`

Shape complementarity *S*c as defined by Lawrence & Colman (1993).

**Algorithm:**
1. For each interface atom (within `interface_cutoff` of the other chain), place `n_dots` surface dots on its VdW sphere using a Fibonacci lattice.
2. Occlude dots buried by same-chain neighbours.
3. For each surface dot on chain A, find the nearest dot on chain B via KDTree.
4. Compute Gaussian-weighted dot-product of outward normals: `S_i = exp(−d²/σ²) × (n̂_A · −n̂_B)`.
5. *S*c = mean of median(A→B) and median(B→A).

VdW radii: C 1.7 Å, N 1.55 Å, O 1.52 Å, S 1.8 Å, default 1.5 Å.

> **Note:** This is a Fibonacci-sphere approximation, not the MSMS-based original. Values are directionally correct but may differ slightly from published *S*c computed by CCP4/SC.

**Returns:**

```python
{
    "sc": float,          # score: 0 = flat, 1 = perfect lock-and-key (>0.7 = good)
    "sc_A_to_B": float,   # directional score peptide→receptor
    "sc_B_to_A": float,   # directional score receptor→peptide

    "n_surface_dots_A": int,          # feature
    "n_surface_dots_B": int,          # feature
    "per_dot_scores_A": list[float],  # feature
    "per_dot_scores_B": list[float],  # feature
}
```

**Interpretation:** *S*c > 0.7 is typical of tight protein–protein interfaces. Antibody–antigen complexes typically score 0.64–0.68; crystal contacts ~0.57.

### `compute_buried_void_volume(cif_path, peptide_chain=None, receptor_chain=None, grid_spacing=0.5, probe_radius=1.4, interface_cutoff=5.0, padding=3.0) → dict`

Detect empty voxels enclosed at the interface using a 3D grid flood-fill.

**Algorithm:**
1. Build a 3D grid over the interface region (atoms within `interface_cutoff`, padded by `padding` Å).
2. Mark voxels as solid if within the VdW radius of any atom — separately for complex, chain A alone, and chain B alone.
3. Flood-fill from the grid boundary to identify solvent-accessible exterior in the complex.
4. Void voxels = not solid in complex AND not reachable from exterior AND (accessible in A alone OR B alone).

**Returns:**

```python
{
    "void_volume_A3": float,           # score (lower = tighter packing)
    "void_grid_fraction": float,       # void voxels / interface-box voxels
    "interface_box_volume_A3": float,  # feature
    "n_interface_atoms": int,          # feature
}
```

**Interpretation:** Well-packed interfaces typically have void_volume_A3 close to 0. Large values indicate cavities that may destabilise binding.

**Distinction from SASA:** SASA detects concave surface pockets accessible to a probe; void volume detects enclosed empty space not reachable from the exterior — geometrically distinct from surface accessibility.

### CLI: `binding-metrics-geometry`

```bash
binding-metrics-geometry --metric sc --input complex.cif [--design-chain A] [--receptor-chain B]
    [--n-dots 50] [--interface-cutoff 5.0]
binding-metrics-geometry --metric void --input complex.cif [--design-chain A] [--receptor-chain B]
    [--grid-spacing 0.5] [--interface-cutoff 5.0]
```

---

## 5. Force-field energy

**Backend:** OpenMM + AMBER ff14SB + PDBFixer
**Install:** `pip install "binding-metrics[simulation,structure]"`

### `compute_interaction_energy(input_path, peptide_chain=None, receptor_chain=None, solvent_model="obc2", device="cuda", modes=("raw", "relaxed", "after_md"), ...) → dict`

Subsystem decomposition:

```
E_int = E_complex − E_peptide − E_receptor
```

Uses AMBER ff14SB force field with implicit solvent (OBC2 or GBn2). Hydrogens and missing heavy atoms are added via PDBFixer automatically.

**Modes:**

| Mode | Description |
|---|---|
| `raw` | Energy at the H-added input geometry. Returns `None` for severe clashes (clash indicator). |
| `relaxed` | Backbone-restrained → unrestrained minimization, then evaluate. Resolves clashes while preserving backbone geometry. |
| `after_md` | From relaxed geometry, run short MD, evaluate at final frame. |

**Returns** a flat dict with `{mode}_interaction_energy`, `{mode}_e_complex`, `{mode}_e_peptide`, `{mode}_e_receptor` for each requested mode (all in kJ/mol). Values are `None` if the mode failed (e.g., severe clash in `raw`).

**Key options:**

| Argument | Default | Description |
|---|---|---|
| `solvent_model` | `"obc2"` | Implicit solvent: `"obc2"` or `"gbn2"` |
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `modes` | `("raw", "relaxed", "after_md")` | Which modes to compute |
| `relaxed_min_steps_restrained` | 500 | Backbone-restrained minimization steps |
| `relaxed_min_steps_full` | 2000 | Unrestrained minimization steps |
| `after_md_duration_ps` | 10.0 | Short MD duration (ps) |
| `after_md_temperature_k` | 300.0 | MD temperature (K) |

### CLI: `binding-metrics-energy`

```bash
binding-metrics-energy --input complex.cif --modes raw relaxed --device cuda --output scores.csv
binding-metrics-energy --input-dir designs/ --glob-pattern "*.cif" --modes relaxed after_md --output scores.csv
```

---

## 6. Structure comparison (RMSD)

**Backend:** gemmi
**Install:** `pip install "binding-metrics[structure]"`

### `compute_structure_rmsd(initial_path, processed_path, design_chain=None) → dict`

Kabsch-aligned RMSD between two structures. Atoms are matched by `(chain, residue_number, atom_name)` so structures that differ in hydrogen count are handled correctly (only common atoms are used).

**Returns:**

```python
{
    "rmsd": float,           # Å — all-atom RMSD of full complex
    "bb_rmsd": float,        # Å — backbone-only RMSD (N, CA, C, O)
    "rmsd_design": float,    # Å — all-atom RMSD of design chain only
    "bb_rmsd_design": float, # Å — backbone-only RMSD of design chain
    # Values are None if computation failed
}
```

### CLI: `binding-metrics-compare`

```bash
binding-metrics-compare --initial initial.cif --processed relaxed.cif [--design-chain A]
```

---

## 7. Trajectory analysis

**Backend:** MDTraj
**Install:** `pip install "binding-metrics[analysis]"`

### `compute_receptor_drift(trajectory_path, topology_path, receptor_chain, reference_frame=0) → dict`

Measures receptor backbone stability across MD frames.

**Algorithm:**
1. Select receptor Cα atoms by chain ID.
2. **Aligned drift**: superpose each frame on the receptor Cα of `reference_frame` (removes global translation/rotation), then measure receptor Cα RMSD — captures conformational drift only.
3. **Raw drift**: Euclidean RMSD without superposition — captures total displacement including global motion. Set to NaN if PBC is detected (`trajectory.unitcell_lengths is not None`).

**Returns:**

```python
{
    "drift_aligned_mean": float,          # Å — score (lower = more stable receptor)
    "drift_aligned_max": float,           # Å — score
    "drift_raw_mean": float,              # Å — NaN if PBC detected
    "drift_raw_max": float,               # Å — NaN if PBC detected

    "pbc_detected": bool,                 # feature
    "drift_aligned_per_frame": list[float],  # feature
    "drift_raw_per_frame": list[float],      # feature (NaN entries if PBC)
    "n_receptor_ca": int,                    # feature
    "n_frames": int,                         # feature
}
```

**When to use which drift:**
- Use `drift_aligned_*` to ask: "is the receptor conformationally changing during MD?" (PBC-safe).
- Use `drift_raw_*` only for non-PBC simulations to ask: "is the complex diffusing / tumbling?"

---

## 8. OpenFold3 confidence metrics

**Backend:** OpenFold3 (`aqlaboratory/openfold-3`)
**Install:** `pip install openfold3 && setup_openfold`

### `compute_openfold_metrics(output_dir, query_name, seed=1, sample=1, include_matrices=False) → dict`

Parse confidence metrics from an existing OpenFold3 output directory.

**Expected output directory structure:**

```
{output_dir}/{query_name}/seed_{seed}/
    {prefix}_model.cif
    {prefix}_confidences_aggregated.json   ← scalar metrics
    {prefix}_confidences.json              ← per-atom arrays
    timing.json
```

**Returns:**

```python
{
    "query_name": str,
    "seed": int,
    "sample": int,
    "structure_path": str | None,

    # Scalar scores from confidences_aggregated.json
    "avg_plddt": float,              # [0–100] — score (higher = better)
    "gpde": float,                   # global predicted distance error, Å — score (lower = better)
    "ptm": float,                    # predicted TM-score [0–1] — score (higher = better)
    "iptm": float,                   # interface pTM [0–1] — score (NaN for monomers)
    "disorder": float,               # avg relative SASA [0–1] — feature
    "has_clash": float,              # 1.0 if steric clashes — score (lower = better)
    "sample_ranking_score": float,   # composite ranking — score (higher = better)
    "chain_ptm": dict,               # {chain_id: float} — per-chain score
    "chain_pair_iptm": dict,         # {"(A, B)": float} — per-pair score
    "bespoke_iptm": dict,            # {"(A, B)": float}

    # Per-atom data from confidences.json / .npz
    "plddt_per_atom": np.ndarray,    # shape (n_atoms,) — feature
    "n_atoms": int,
    "max_pae": float,                # Å — feature (always computed)
    "pde": np.ndarray | None,        # (n_tokens, n_tokens) — only if include_matrices=True
    "pae": np.ndarray | None,        # (n_tokens, n_tokens) — only if include_matrices=True

    "timing": dict,                  # from timing.json; empty if absent
}
```

**Model presets:**

| Preset | Effect |
|---|---|
| `"predict"` | Required base preset for inference |
| `"pae_enabled"` | Enable PAE head — required for pTM, ipTM, disorder, chain scores (and by the official weights) |
| `"low_mem"` | Memory-efficient sequential pairformer — recommended for large complexes or limited GPU memory |

### `run_openfold(query_json, output_dir, ...) → Path`

Run OpenFold3 inference as a subprocess (`run_openfold predict`). Returns the output directory path.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `num_diffusion_samples` | 5 | Structures sampled per query |
| `num_model_seeds` | 1 | Random seeds per query |
| `use_msa_server` | True | Use ColabFold server for MSAs |
| `model_presets` | `["predict", "pae_enabled", "low_mem"]` | Config presets |
| `runner_yaml` | None | Explicit YAML config; overrides `model_presets` |

### CLI: `binding-metrics-openfold`

```bash
# Parse existing output
binding-metrics-openfold parse --output-dir ./openfold_out --query-name my_complex \
    --seed 1 --sample 1 [--include-matrices]

# Run then parse
binding-metrics-openfold run --query-json query.json --output-dir ./openfold_out \
    --query-name my_complex --num-samples 5 --num-seeds 1 \
    [--presets predict pae_enabled low_mem] [--no-msa-server]
```

---

## 9. Score vs. feature taxonomy

Every metric return value is classified as a **score** or a **feature**:

| Type | Definition | Examples |
|---|---|---|
| **Score** | Has a reference or ground truth; directional interpretation (higher or lower is clearly better) | `delta_sasa`, `iptm`, `ramachandran_outlier_count`, `sc`, `coulomb_energy_kJ` |
| **Feature** | Descriptor without an intrinsic quality direction; useful for analysis, clustering, or as model input | `fraction_polar`, `per_residue`, `plddt_per_atom`, `pbc_detected`, `n_charged_pairs` |

When building composite scorers or ML models, use scores as labels or primary objectives and features as inputs or auxiliary descriptors.

---

## I/O utilities

### `load_structure(path) → (topology, positions)`

Load a PDB or CIF file into OpenMM topology and positions.

### `detect_chains(topology) → (peptide_chain, receptor_chain)`

Auto-detect smallest and largest protein chain from an OpenMM topology.

### `detect_chains_from_file(path, peptide_chain=None, receptor_chain=None, verbose=False) → dict`

Biotite-based chain detection returning both `auth_asym_id` (biotite) and `label_asym_id` (OpenMM) for the peptide and receptor. For two protein chains: smaller = peptide, larger = receptor. For more than two chains: receptor is picked by Cα proximity (most contacts within 8 Å of the peptide) rather than by size.

### `save_cif(topology, positions, output_path, source_cif_path=None)`

Save structure as CIF, optionally merging coordinates into the source CIF to preserve metadata.
