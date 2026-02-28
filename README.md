# BindingMetrics

A Python toolkit for evaluating biologics binding quality through physics-based metrics. Designed for scoring peptide–protein complexes produced by structure prediction or computational design pipelines.

Metrics span from fast static-structure analysis (SASA, H-bonds, PISA solvation energy) to force-field-based interaction energies computed with optional energy minimization and short MD equilibration.

---

## Features

| Category | Metric | Backend |
|---|---|---|
| Interface geometry | Buried SASA (Δ SASA) | biotite |
| Interface energetics | Solvation binding energy Δ*G*_int (kcal/mol) | biotite + Eisenberg-McLachlan |
| Interface composition | Polar / apolar buried area, fraction polar | biotite |
| Interface inventory | Interface residues per chain, per-residue breakdown | biotite |
| Interactions | Cross-chain H-bonds | biotite + hydride |
| Interactions | Cross-chain salt bridges | biotite |
| Force-field energy | Interaction energy *E*_complex − *E*_peptide − *E*_receptor | OpenMM + AMBER ff14SB |
| Force-field energy | Raw / after minimization / after short MD modes | OpenMM |
| Structure comparison | All-atom and backbone RMSD (Kabsch-aligned) | gemmi |
| Trajectory analysis | Buried SASA, contacts, RMSD, RMSF per frame | MDTraj |
| Structure prediction | avg_pLDDT, pTM, ipTM, gPDE, PAE, per-chain scores | OpenFold3 |

---

## Installation

```bash
pip install binding-metrics
```

Install optional dependency groups as needed:

```bash
# OpenMM for energy calculations and MD simulations
pip install "binding-metrics[simulation]"

# MDTraj for trajectory-based metrics
pip install "binding-metrics[analysis]"

# PDBFixer + gemmi for structure repair and RMSD comparison
pip install "binding-metrics[structure]"

# biotite + hydride for PISA interface analysis and H-bonds
pip install "binding-metrics[biotite]"

# Everything
pip install "binding-metrics[all]"
```

OpenFold3 must be installed separately (GPU required, large model weights):

```bash
pip install openfold3
setup_openfold   # downloads model weights
```

Requires Python ≥ 3.11.

---

## Quick Start

### Interface analysis (PISA-inspired)

```python
from binding_metrics import compute_interface_metrics

metrics = compute_interface_metrics("complex.cif")

print(f"Buried SASA:          {metrics['delta_sasa']:.1f} Å²")
print(f"ΔG_int:               {metrics['delta_g_int']:.2f} kcal/mol")
print(f"Fraction polar:       {metrics['fraction_polar']:.2f}")
print(f"H-bonds:              {metrics['hbonds']}")
print(f"Salt bridges:         {metrics['saltbridges']}")
print(f"Interface residues:   {metrics['interface_residues_peptide']}")
```

### Interaction energy

```python
from binding_metrics import compute_interaction_energy

result = compute_interaction_energy(
    "complex.cif",
    modes=("raw", "relaxed"),   # skip after_md for speed
    device="cuda",              # or "cpu"
)

print(f"Raw E_int:     {result['raw_interaction_energy']:.1f} kJ/mol")
print(f"Relaxed E_int: {result['relaxed_interaction_energy']:.1f} kJ/mol")
```

### Structure comparison

```python
from binding_metrics import compute_structure_rmsd

result = compute_structure_rmsd("initial.cif", "relaxed.cif", design_chain="A")

print(f"RMSD (full):    {result['rmsd']:.3f} Å")
print(f"BB RMSD:        {result['bb_rmsd']:.3f} Å")
print(f"RMSD (design):  {result['rmsd_design']:.3f} Å")
```

### OpenFold3 confidence metrics

Parse confidence scores from an existing OpenFold3 output directory:

```python
from binding_metrics import compute_openfold_metrics

metrics = compute_openfold_metrics(
    output_dir="./openfold_out",
    query_name="my_complex",   # as specified in the input JSON
    seed=1,
    sample=1,
)

print(f"avg_pLDDT:        {metrics['avg_plddt']:.1f}")
print(f"pTM:              {metrics['ptm']:.3f}")
print(f"ipTM:             {metrics['iptm']:.3f}")
print(f"gPDE:             {metrics['gpde']:.3f} Å")
print(f"Max PAE:          {metrics['max_pae']:.1f} Å")
print(f"Ranking score:    {metrics['sample_ranking_score']:.3f}")
print(f"Per-chain pTM:    {metrics['chain_ptm']}")
```

To also run inference:

```python
from binding_metrics import run_openfold, compute_openfold_metrics

run_openfold(
    query_json="query.json",
    output_dir="./openfold_out",
    num_diffusion_samples=5,
    num_model_seeds=1,
    # Defaults: predict + pae_enabled + low_mem
    model_presets=["predict", "pae_enabled", "low_mem"],
)
metrics = compute_openfold_metrics("./openfold_out", "my_complex")
```

### Batch scoring

```python
from pathlib import Path
from binding_metrics import compute_interface_metrics, compute_interaction_energy
import pandas as pd

rows = []
for cif in sorted(Path("designs/").glob("*.cif")):
    iface = compute_interface_metrics(cif)
    energy = compute_interaction_energy(cif, modes=("relaxed",), device="cuda")
    rows.append({
        "sample": cif.stem,
        "delta_sasa": iface["delta_sasa"],
        "delta_g_int": iface["delta_g_int"],
        "fraction_polar": iface["fraction_polar"],
        "hbonds": iface["hbonds"],
        "saltbridges": iface["saltbridges"],
        "relaxed_e_int": energy["relaxed_interaction_energy"],
    })

df = pd.DataFrame(rows).sort_values("relaxed_e_int")
df.to_csv("scores.csv", index=False)
```

---

## CLI Tools

All tools accept a CIF file as input and auto-detect peptide / receptor chains (smallest and largest protein chain respectively). Pass `--design-chain` / `--receptor-chain` to override.

### `binding-metrics-interface`

Compute PISA-inspired interface metrics for a static structure.

```bash
binding-metrics-interface --input complex.cif --design-chain A
```

```
Interface summary:
  delta_sasa:                    892.456 Å²
  delta_g_int:                   -8.412 kcal/mol
  delta_g_int_kJ:                -35.196 kJ/mol
  polar_area:                    312.100 Å²
  apolar_area:                   501.230 Å²
  fraction_polar:                0.350
  n_interface_residues_peptide:  11
  n_interface_residues_receptor: 23
  hbonds:                        7
  saltbridges:                   2

Per-residue contributions (sorted by buried SASA):
  Residue              BuriedSASA     ΔG_res    Polar   Apolar
  PHE:A:12                 98.34    -1.5734    12.10    80.12
  ...
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--design-chain` | auto | Peptide chain ID |
| `--receptor-chain` | auto | Receptor chain ID |
| `--probe-radius` | 1.4 | Solvent probe radius (Å) |
| `--threshold` | 0.5 | Min buried SASA per residue to count as interface (Å²) |

### `binding-metrics-energy`

Compute interaction energy via subsystem decomposition (AMBER ff14SB + implicit solvent).

```bash
binding-metrics-energy --input complex.cif --modes raw relaxed --device cuda --output scores.csv
```

```bash
# Process a directory of CIF files
binding-metrics-energy --input-dir designs/ --glob-pattern "*.cif" \
    --modes relaxed after_md --output scores.csv
```

**Modes:**

| Mode | Description |
|---|---|
| `raw` | Energy at the H-added input geometry. Returns `None` for structures with severe clashes (useful as a clash indicator). |
| `relaxed` | Backbone-restrained → full unrestrained minimization, then evaluate. Resolves clashes while preserving backbone geometry. |
| `after_md` | From the relaxed geometry, run short MD and evaluate at the final frame. |

Options:

| Flag | Default | Description |
|---|---|---|
| `--modes` | `raw relaxed after_md` | Which modes to compute |
| `--solvent-model` | `obc2` | Implicit solvent (`obc2` or `gbn2`) |
| `--device` | `cuda` | Compute device (`cuda` or `cpu`) |
| `--relaxed-min-steps-restrained` | 500 | Backbone-restrained minimization steps |
| `--relaxed-min-steps-full` | 2000 | Unrestrained minimization steps |
| `--after-md-duration-ps` | 10.0 | Short MD duration (ps) |
| `--after-md-temperature-k` | 300.0 | MD temperature (K) |

### `binding-metrics-compare`

Compute RMSD between two structures (e.g. initial vs. relaxed).

```bash
binding-metrics-compare --initial initial.cif --processed relaxed.cif --design-chain A
```

### `binding-metrics-openfold`

Parse OpenFold3 confidence metrics from an output directory, or run inference then parse.

**Parse existing output:**

```bash
binding-metrics-openfold parse \
    --output-dir ./openfold_out \
    --query-name my_complex \
    --seed 1 --sample 1
```

```
OpenFold3 confidence metrics (seed=1, sample=1):
  Structure:             ./openfold_out/my_complex/seed_1/my_complex_seed_1_sample_1_model.cif
  Atoms:                 1247
  avg_pLDDT [0–100]:     87.4200
  gPDE (Å):              1.2300
  pTM [0–1]:             0.8800
  ipTM [0–1]:            0.7600
  Disorder:              0.1200
  has_clash:             0.0000
  Ranking score:         0.8200
  Max PAE (Å):           4.1000

  Per-chain pTM:
    chain 1: 0.8800
    chain 2: 0.8000

  Chain-pair ipTM:
    (1, 2): 0.7600

Per-atom pLDDT summary:
  Min:    61.2
  Median: 89.5
  Max:    98.3
  Very high (≥90): 823 atoms
  High    (70–90): 381 atoms
  Low     (50–70):  43 atoms
  Very low  (<50):   0 atoms
```

**Run inference then parse:**

```bash
binding-metrics-openfold run \
    --query-json query.json \
    --output-dir ./openfold_out \
    --query-name my_complex \
    --num-samples 5 --num-seeds 1
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--seed` | 1 | Seed index to parse |
| `--sample` | 1 | Sample index to parse |
| `--include-matrices` | off | Include full PDE/PAE matrices in output |
| `--num-samples` | 5 | Diffusion samples per query (run only) |
| `--num-seeds` | 1 | Random seeds per query (run only) |
| `--no-msa-server` | off | Disable ColabFold MSA server (run only) |
| `--presets` | `predict pae_enabled low_mem` | Model config presets (run only) |
| `--ckpt` | auto | Model checkpoint path (run only) |
| `--runner-yaml` | none | Explicit YAML config; overrides `--presets` (run only) |

### `binding-metrics-relax`

Run implicit-solvent energy minimization on a structure.

```bash
binding-metrics-relax --input complex.cif --output relaxed.cif --device cuda
```

---

## API Reference

### Interface analysis

#### `compute_interface_metrics(cif_path, design_chain=None, receptor_chain=None, probe_radius=1.4, interface_threshold=0.5) → dict`

Full interface characterisation for a CIF structure. Computes per-atom buried SASA and derives thermodynamic and structural metrics.

The solvation binding energy uses Eisenberg-McLachlan atomic solvation parameters:

| Element | γ (kcal/mol/Å²) | Character |
|---|---|---|
| C | −0.016 | hydrophobic (burial favorable) |
| N | +0.063 | polar (burial unfavorable) |
| O | +0.024 | polar (burial unfavorable) |
| S | −0.021 | weakly hydrophobic |

**Returns:**

```python
{
    # Chains
    "peptide_chain": str,
    "receptor_chain": str,

    # SASA (Å²)
    "delta_sasa": float,          # total buried SASA
    "sasa_peptide": float,        # peptide SASA in isolation
    "sasa_receptor": float,       # receptor SASA in isolation
    "sasa_complex": float,        # complex SASA

    # Solvation energy
    "delta_g_int": float,         # kcal/mol — negative = hydrophobic-driven
    "delta_g_int_kJ": float,      # kJ/mol

    # Interface composition
    "polar_area": float,          # Å² from N and O atoms
    "apolar_area": float,         # Å² from C and S atoms
    "fraction_polar": float,      # polar_area / delta_sasa

    # Interface residues
    "n_interface_residues_peptide": int,
    "n_interface_residues_receptor": int,
    "interface_residues_peptide": list[str],   # "RES:CHAIN:NUM"
    "interface_residues_receptor": list[str],

    # Per-residue details
    "per_residue": list[dict],    # buried_sasa, delta_g_res, polar_area, apolar_area

    # Interactions
    "hbonds": int,
    "saltbridges": int,
}
```

#### `compute_hbonds(atoms, peptide_chain, receptor_chain) → int`

Count cross-chain hydrogen bonds using biotite's detector. Adds explicit hydrogens with hydride if available.

#### `compute_saltbridges(atoms, peptide_chain, receptor_chain, distance_min=0.5, distance_max=5.5) → int`

Count cross-chain salt bridges by distance between charged atoms (LYS/ARG/HIS ↔ ASP/GLU).

#### `compute_delta_sasa_static(cif_path, peptide_chain, receptor_chain, probe_radius=1.4) → dict`

Lightweight SASA-only calculation (totals, no per-atom decomposition).

### OpenFold3

#### `compute_openfold_metrics(output_dir, query_name, seed=1, sample=1, include_matrices=False) → dict`

Parse confidence metrics from an OpenFold3 output directory. Reads `*_confidences_aggregated.json` for scalar scores and `*_confidences.json` (or `.npz`) for per-atom arrays.

**Returns:**

```python
{
    "query_name": str,
    "seed": int,
    "sample": int,
    "structure_path": str | None,   # path to the .cif/.pdb file

    # Scalar confidence metrics (NaN if PAE head disabled or file absent)
    "avg_plddt": float,             # mean pLDDT across all atoms [0–100]
    "gpde": float,                  # global predicted distance error (Å)
    "ptm": float,                   # predicted TM-score [0–1]
    "iptm": float,                  # interface pTM [0–1] (NaN for monomers)
    "disorder": float,              # avg relative SASA [0–1]
    "has_clash": float,             # 1.0 if steric clashes detected
    "sample_ranking_score": float,  # composite ranking score
    "chain_ptm": dict,              # {chain_id: float}
    "chain_pair_iptm": dict,        # {"(A, B)": float}
    "bespoke_iptm": dict,           # {"(A, B)": float}

    # Per-atom data
    "plddt_per_atom": np.ndarray,   # shape (n_atoms,)
    "n_atoms": int,
    "pde": np.ndarray | None,       # (n_tokens, n_tokens) — only if include_matrices=True
    "pae": np.ndarray | None,       # (n_tokens, n_tokens) — only if include_matrices=True
    "max_pae": float,               # max PAE value (Å); computed regardless of include_matrices

    "timing": dict,                 # from timing.json; empty if absent
}
```

#### `run_openfold(query_json, output_dir, ...) → Path`

Run OpenFold3 inference as a subprocess (`run_openfold predict`). Requires OpenFold3 installed and `setup_openfold` completed. Returns the output directory path.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `num_diffusion_samples` | 5 | Structures sampled per query |
| `num_model_seeds` | 1 | Random seeds per query |
| `use_msa_server` | True | Use ColabFold server for MSAs |
| `model_presets` | `["predict", "pae_enabled", "low_mem"]` | Model configuration presets (see below) |
| `runner_yaml` | None | Explicit YAML config; overrides `model_presets` |

**Available model presets** (passed via auto-generated `runner_config.yaml`):

| Preset | Effect |
|---|---|
| `"predict"` | Required base preset for inference |
| `"pae_enabled"` | Enable PAE head — required for pTM, ipTM, disorder, chain scores, and by the official OpenFold3 weights |
| `"low_mem"` | Memory-efficient mode: pairformer embeddings computed sequentially. Recommended for large complexes or limited GPU memory; note significant slowdown with many diffusion samples |

The input JSON format is described in the [OpenFold3 documentation](https://openfold-3.readthedocs.io/en/latest/input_format.html).

### Force-field energy

#### `compute_interaction_energy(input_path, peptide_chain=None, receptor_chain=None, solvent_model="obc2", device="cuda", modes=("raw", "relaxed", "after_md"), ...) → dict`

Subsystem decomposition: *E*_int = *E*_complex − *E*_peptide − *E*_receptor. Uses AMBER ff14SB + implicit solvent (OBC2 or GBn2). Hydrogen addition and structure repair (via PDBFixer) are performed automatically.

Returns a flat dict with `{mode}_interaction_energy`, `{mode}_e_complex`, `{mode}_e_peptide`, `{mode}_e_receptor` for each requested mode.

### Structure comparison

#### `compute_structure_rmsd(initial_path, processed_path, design_chain=None) → dict`

Kabsch-aligned RMSD between two structures. Atoms are matched by `(chain, residue, atom_name)` so structures that differ in hydrogen count are handled correctly.

Returns `rmsd`, `bb_rmsd`, `rmsd_design`, `bb_rmsd_design`.

### I/O utilities

#### `load_structure(path) → (topology, positions)`

Load a PDB or CIF file into OpenMM topology and positions.

#### `detect_chains(topology) → (peptide_chain, receptor_chain)`

Auto-detect smallest and largest protein chain from an OpenMM topology.

#### `save_cif(topology, positions, output_path, source_cif_path=None)`

Save structure as CIF, optionally merging coordinates into the source CIF to preserve metadata.

---

## Background: PISA Solvation Energy

The solvation binding energy Δ*G*_int implemented here follows the approach of [Krissinel & Henrick (2007)](https://doi.org/10.1016/j.jmb.2007.05.022) as implemented in the PDBePISA service, using per-atom solvation parameters from Eisenberg & McLachlan (1986):

```
ΔG_int = Σ_i  γ_i × ΔA_i
```

where ΔA_i is the buried SASA of atom *i* (SASA in isolation minus SASA in complex) and γ_i is the atomic solvation parameter for its element.

- **Negative Δ*G*_int**: hydrophobic burial dominates — typical of stable, biologically relevant interfaces (< −5 kcal/mol is a common threshold).
- **Positive Δ*G*_int**: polar burial dominates — may indicate a less stable or crystal-packing contact.

Note that Δ*G*_int captures the solvation component of binding only. For a full binding free energy estimate, combine with the force-field interaction energy from `compute_interaction_energy`.

---

## Development

```bash
git clone https://github.com/binding-metrics/binding-metrics
cd binding-metrics
pip install -e ".[all,dev]"
pytest
```

Run only fast tests (no GPU required):

```bash
pytest -m "not gpu and not slow"
```

---

## References

- Krissinel, E. & Henrick, K. (2007). Inference of macromolecular assemblies from crystalline state. *J. Mol. Biol.* 372, 774–797. https://doi.org/10.1016/j.jmb.2007.05.022
- Eisenberg, D. & McLachlan, A.D. (1986). Solvation energy in protein folding and binding. *Nature* 319, 199–203.
- Eastman, P. et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLOS Comput. Biol.* 13, e1005659.
- Ahdritz, G. et al. (2024). OpenFold3: An open-source, trainable implementation of AlphaFold3. GitHub: https://github.com/aqlaboratory/openfold-3
