# BindingMetrics

**BindingMetrics** is a Python toolkit for evaluating biologics binding quality through physics-based metrics. It is designed for scoring peptide–protein complexes produced by structure prediction or computational design pipelines.

Metrics span from fast static-structure analysis (buried SASA, hydrogen bonds, salt bridges, Ramachandran validation) to force-field interaction energies computed with optional energy minimization and short MD equilibration, to trajectory-level receptor drift and structure prediction confidence scores from OpenFold3.

---

## Cyclic peptide support

BindingMetrics has first-class support for **cyclic peptides** — both head-to-tail (N→C amide) and sidechain-anchored (lactam, disulfide) ring closures.

Cyclic connectivity is read from `_struct_conn` in the input CIF file (written by BoltzGen and other structure prediction pipelines that support cyclic designs). The pipeline:

- **Auto-detects** the cyclic bond type (`head_to_tail`, `sidechain_to_sidechain`, etc.) and the atoms involved
- **Propagates** the bond hint through PDBFixer prep — the N→C bond is written back into `_struct_conn` of the saved `_cleaned.cif`, so downstream runs on that file also detect the cyclic topology
- **Applies cyclic AMBER templates** (NCYS/NCYX, C-terminal patches) during hydrogen placement and system creation so force-field template matching succeeds
- **Handles cross-chain disulfides** (e.g. peptide CYS ↔ receptor CYS) alongside the cyclic closure: PDBFixer-detected SS bonds trigger a CYS→CYX rename in-memory before `addHydrogens`, and orphaned CYX residues (whose SS partner is on the other chain, severed during per-chain energy decomposition) are automatically converted back to CYS+HG
- **Minimizes cyclic geometry** with a dedicated closure-bond relaxation stage before global minimization

No flags needed — cyclic topology is detected and applied automatically whenever the input CIF contains the relevant `_struct_conn` entries.

---

## Receptor quality assessment

**`binding-metrics-receptor-quality`** is a standalone tool for evaluating receptor structural quality, independent of the main peptide-binding pipeline. It works on receptor-only files or complex structures (non-receptor chains are silently ignored), and scores all models in multi-model PDB/CIF files independently.

Metrics follow the MolProbity convention (Chen et al. 2010):

| Metric | Goal |
|---|---|
| Ramachandran favoured % | > 98% |
| Ramachandran outliers % | < 0.05% |
| Rotamer outliers % (χ1) | < 1% |
| Cβ deviations > 0.25 Å | 0 |
| Bad backbone bonds | 0% |
| Bad backbone angles | < 0.1% |
| Clashscore | < 1 (high-resolution crystal) |
| MolProbity score | lower = better (resolution-like scale) |
| Absolute AMBER ff14SB energy | lower = less strained |

```python
from binding_metrics import compute_receptor_quality

# Works on receptor-only or complex structures; auto-detects largest chain
result = compute_receptor_quality("receptor.pdb", device="cuda")
s = result["summary"]
print(f"MolProbity score : {s['molprobity_score']:.2f}")
print(f"Ramachandran     : {s['ramachandran_favoured_pct']:.1f}% favoured, "
      f"{s['ramachandran_outlier_pct']:.1f}% outliers")
print(f"Clashscore       : {s['clashscore']:.2f}")
print(f"Rotamer outliers : {s['rotamer_outlier_pct']:.1f}%")
print(f"Cβ deviations    : {s['cb_deviation_count']:.0f}")
print(f"Bad bonds/angles : {s['bad_bonds_pct']:.2f}% / {s['bad_angles_pct']:.2f}%")
print(f"Energy           : {s['energy_kJ_mol']:.1f} kJ/mol")
print(f"Best model       : {s['best_model_index']}")
```

```bash
# CLI — output format auto-detected from extension (.csv or .json)
binding-metrics-receptor-quality --input receptor.pdb --output quality.csv
binding-metrics-receptor-quality --input ensemble.cif --receptor-chain A --device cpu
```

---

## Metrics at a glance

**Scores** have a clear direction (higher or lower is better). **Features** are descriptors without an intrinsic quality direction, useful for analysis or as model inputs.

| Category | Metric | Type | Backend |
|---|---|---|---|
| Interface geometry | Buried SASA Δ*A*, polar/apolar breakdown | Score | biotite |
| Interface energetics | Solvation energy Δ*G*_int — negative = hydrophobic-driven | Score | biotite |
| Interactions | Cross-chain H-bonds, salt bridges | Score | biotite + hydride |
| Electrostatics | Coulomb cross-chain energy — negative = net attractive | Score | biotite + scipy |
| Backbone geometry | Ramachandran outlier %, ω-angle deviation | Score | biotite |
| Interface shape | Shape complementarity *S*c — 0 = flat, 1 = lock-and-key | Score | biotite + scipy |
| Interface packing | Buried void volume — large = loose packing | Score | biotite + scipy |
| Force-field energy | *E*_int = *E*_cpx − *E*_pep − *E*_rec (AMBER ff14SB); raw / relaxed / after MD | Score | OpenMM |
| Structure comparison | All-atom and backbone RMSD (Kabsch-aligned) | Score | gemmi |
| MD trajectory | Receptor backbone drift — aligned (conformational) and raw | Score | MDTraj |
| Structure prediction | avg_pLDDT, pTM, ipTM, gPDE — OpenFold3 confidence | Score | OpenFold3 |
| EvoBind scoring | Interface distance / pLDDT — confidence-weighted binding score (Å) | Score | biotite |
| EvoBind adversarial check | Δ COM between design pose and OF3 prediction after receptor superposition — flags hallucinated poses | Score | biotite |
| All of the above | Per-residue breakdowns, per-atom arrays, per-frame series | Feature | — |

---

## Installation

### Recommended: conda (GPU-accelerated)

MD simulations are computationally prohibitive on CPU. **A CUDA-capable GPU and
the conda-forge OpenMM build are strongly recommended** for any workflow that
involves energy minimization or MD (`binding-metrics-relax`, `compute_interaction_energy`
with `mode="relaxed"` or `mode="md"`).

```bash
conda env create -f environment.yml   # creates the binding-metrics conda env
conda activate binding-metrics
```

This installs:
- GPU-ready OpenMM from conda-forge (CUDA/OpenCL binaries)
- openmmforcefields + openff-toolkit for GAFF2 small-molecule parameterization
- All other dependencies via pip

### Alternative: pip (CPU only)

> **Warning — MD on CPU is extremely slow.** Only use this path for static
> metrics (interface geometry, electrostatics, RMSD) or for CI/testing with
> `device="cpu"` and minimal minimization steps.

```bash
pip install binding-metrics            # core only
```

Install optional dependency groups based on the metrics you need:

```bash
pip install "binding-metrics[simulation]"   # OpenMM (CPU-only via PyPI)
pip install "binding-metrics[analysis]"     # MDTraj — trajectory metrics
pip install "binding-metrics[structure]"    # PDBFixer + gemmi — structure repair, RMSD
pip install "binding-metrics[biotite]"      # biotite + hydride + scipy — interface, geometry
pip install "binding-metrics[gaff]"         # openmmforcefields + openff-toolkit — GAFF2 for non-standard residues
pip install "binding-metrics[report]"       # no additional dependencies — JSON/CSV/Markdown output
pip install "binding-metrics[all]"          # everything above (OpenMM via PyPI = CPU only)
```

> The PyPI `openmm` wheel has no CUDA support. `pip install binding-metrics[all]`
> gives a functional install for testing, but MD runs will be orders of magnitude
> slower than on GPU. For production use, always install OpenMM via conda-forge.

### Docker (GPU, recommended for production)

A pre-built image is available on Docker Hub. It includes GPU-ready OpenMM (CUDA 12.2), all conda-forge dependencies, the full `[all]` extras, and a ready-to-use `openfold3` conda env.

**One-time OpenFold3 setup** — run this once after pulling the image to download model weights into the container (see [OpenFold3 on GitHub](https://github.com/aqlaboratory/openfold-3/) for details):

```bash
docker run -it --gpus all simoncrouzet/binding-metrics:latest \
    conda run -n openfold3 setup_openfold
```

```bash
docker pull simoncrouzet/binding-metrics:latest
```

Run with GPU access (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker run --gpus all --rm \
    -v /path/to/your/structures:/data \
    simoncrouzet/binding-metrics:latest \
    binding-metrics-relax --input /data/complex.cif --output /data/relaxed.cif
```

To build the image locally from source:

```bash
docker build -t simoncrouzet/binding-metrics:latest .
docker push simoncrouzet/binding-metrics:latest   # optional
```

The image is rebuilt and pushed to Docker Hub automatically on every push to `main` and on version tags (`v*`) via GitHub Actions.

### OpenFold3 (optional)

OpenFold3 confidence scoring is optional — all other metrics work without it.
It requires a GPU, model weights, and a compatible Python version, so we recommend
installing it in a **dedicated conda environment** named `openfold3`:

```bash
conda create -n openfold3 python=3.10
conda activate openfold3
pip install openfold3
setup_openfold   # downloads model weights
```

`binding-metrics-run` will automatically use the `openfold3` env via
`--openfold-conda-env openfold3` (the default). You can point it at a
different environment name if needed, or leave OpenFold out entirely:

```bash
# Uses openfold3 env by default — no flag needed if installed there
binding-metrics-run --input complex.cif --output-dir results/

# Explicit env name if you used a different name
binding-metrics-run --input complex.cif --output-dir results/ \
    --openfold-conda-env my_openfold_env

# Skip OpenFold entirely
binding-metrics-run --input complex.cif --output-dir results/ \
    --metrics energy,interface,geometry,electrostatics
```

Run `binding-metrics-check-env` to verify whether OpenFold3 is correctly installed.

---

## Quick Start

### Interface analysis

```python
from binding_metrics import compute_interface_metrics

metrics = compute_interface_metrics("complex.cif")
print(f"Buried SASA:    {metrics['delta_sasa']:.1f} Å²")
print(f"ΔG_int:         {metrics['delta_g_int']:.2f} kcal/mol")
print(f"H-bonds:        {metrics['hbonds']}")
print(f"Salt bridges:   {metrics['saltbridges']}")
```

### Backbone geometry

```python
from binding_metrics import compute_ramachandran, compute_omega_planarity

rama = compute_ramachandran("complex.cif", chain="A")
print(f"Favoured: {rama['ramachandran_favoured_pct']:.1f}%  Outliers: {rama['ramachandran_outlier_count']}")

omega = compute_omega_planarity("complex.cif", chain="A")
print(f"Mean ω deviation: {omega['omega_mean_dev']:.1f}°  Outliers: {omega['omega_outlier_count']}")
```

### Shape complementarity & void volume

```python
from binding_metrics import compute_shape_complementarity, compute_buried_void_volume

sc = compute_shape_complementarity("complex.cif")
print(f"Sc: {sc['sc']:.3f}")           # 0 = flat, 1 = perfect lock-and-key

void = compute_buried_void_volume("complex.cif")
print(f"Void volume: {void['void_volume_A3']:.1f} Å³")
```

### Coulomb electrostatics

```python
from binding_metrics import compute_coulomb_cross_chain

result = compute_coulomb_cross_chain("complex.cif")
print(f"Coulomb energy: {result['coulomb_energy_kJ']:.1f} kJ/mol  ({result['n_charged_pairs']} pairs)")
```

### Force-field interaction energy

```python
from binding_metrics import compute_interaction_energy

result = compute_interaction_energy("complex.cif", modes=("raw", "relaxed"), device="cuda")
print(f"Raw E_int:     {result['raw_interaction_energy']:.1f} kJ/mol")
print(f"Relaxed E_int: {result['relaxed_interaction_energy']:.1f} kJ/mol")
```

### Receptor backbone drift (MD trajectory)

```python
from binding_metrics import compute_receptor_drift

result = compute_receptor_drift("traj.dcd", "complex.pdb", receptor_chain="A")
print(f"Aligned drift — mean: {result['drift_aligned_mean']:.3f} Å  max: {result['drift_aligned_max']:.3f} Å")
```

### OpenFold3 confidence scores

```python
from binding_metrics import compute_openfold_metrics

metrics = compute_openfold_metrics("./openfold_out", query_name="my_complex", seed=1, sample=1)
print(f"pLDDT: {metrics['avg_plddt']:.1f}  ipTM: {metrics['iptm']:.3f}  gPDE: {metrics['gpde']:.3f} Å")
```

When the `openfold` metric is enabled in `binding-metrics-run`, EvoBind scores are automatically computed and merged into the OpenFold3 result dict — no additional model calls required.

### EvoBind scoring

```python
from binding_metrics.metrics.evobind import (
    compute_evobind_score,
    compute_evobind_adversarial_check,
)

# Primary score on the OF3 prediction: interface distance / pLDDT (Å)
# Lower is better — penalises both poor contact and low confidence.
score = compute_evobind_score(
    "of3_prediction.cif",
    plddt_per_atom=metrics["plddt_per_atom"],
    binder_chain="B",
    receptor_chain="A",
)
print(f"EvoBind score: {score['evobind_score']:.2f} Å  (if_dist: {score['if_dist_pep_to_rec']:.2f} Å)")

# Adversarial check: does the OF3 prediction agree with the input design pose?
# Large Δ COM means OF3 places the binder elsewhere → hallucinated pose.
check = compute_evobind_adversarial_check(
    design_structure_path="input_design.cif",
    afm_structure_path="of3_prediction.cif",
    binder_chain="B",
    receptor_chain="A",
    afm_plddt_per_atom=metrics["plddt_per_atom"],
)
print(f"Δ COM: {check['delta_com_angstrom']:.2f} Å  adversarial score: {check['evobind_adversarial_score']:.1f}")
```

Both functions implement the losses from [Bryant et al. (2025) *EvoBind*, Communications Chemistry](https://doi.org/10.1038/s42004-025-01601-3). The primary score is `if_dist / (pLDDT/100)`; the adversarial score is `mean_if_dist × (100/pLDDT) × ΔCOM`.

### Batch scoring

```python
from pathlib import Path
import pandas as pd
from binding_metrics import compute_interface_metrics, compute_interaction_energy

rows = []
for cif in sorted(Path("designs/").glob("*.cif")):
    iface = compute_interface_metrics(cif)
    energy = compute_interaction_energy(cif, modes=("relaxed",), device="cuda")
    rows.append({
        "sample": cif.stem,
        "delta_sasa": iface["delta_sasa"],
        "delta_g_int": iface["delta_g_int"],
        "hbonds": iface["hbonds"],
        "relaxed_e_int": energy["relaxed_interaction_energy"],
    })

pd.DataFrame(rows).sort_values("relaxed_e_int").to_csv("scores.csv", index=False)
```

---

## CLI Tools

**Structure preparation**

| Command | Description |
|---|---|
| `binding-metrics-prep` | Fix missing atoms/residues, add hydrogens (`--ph 7.4`), optionally canonicalize non-standard residues (`--canonicalize`) |
| `binding-metrics-solvate` | Add explicit water box and ions for MD |

These two commands are composable pipeline steps:

```bash
binding-metrics-prep    --input complex.cif --output cleaned.cif --ph 7.4
binding-metrics-solvate --input cleaned.cif --output solvated.pdb
```

**Full pipeline**

| Command | Description |
|---|---|
| `binding-metrics-run` | Run the complete pipeline (prep → relax → energy → interface → geometry → electrostatics → OpenFold3) on a single structure |

```bash
binding-metrics-run \
    --input complex.cif \
    --output-dir results/ \
    --summary                      # also write a human-readable *_report.md
```

Peptide and receptor chains are **auto-detected** (smallest chain = peptide; when more than two chains are present, the one with the most Cα contacts to the peptide is the receptor). Override with `--peptide-chain` / `--receptor-chain`.

**Cyclic peptides** are handled automatically — cyclic connectivity is read from `_struct_conn` in the input CIF and propagated through prep, relaxation, and energy decomposition with no extra flags required. See [Cyclic peptide support](#cyclic-peptide-support) above.

The pipeline always starts with a **prep step** (equivalent to `binding-metrics-prep`) that fixes missing atoms, adds hydrogens, and removes waters. Key prep flags:

```bash
binding-metrics-run --input complex.cif --output-dir results/ \
    --ph 7.4           # protonation pH (default: 7.4)
    --keep-water       # keep crystallographic waters
    --canonicalize     # rename non-standard residues to their canonical equivalents
    --skip-prep        # skip prep — use if the structure is already protonated
```

All metric steps are enabled by default. Skip relaxation with `--skip-relax`; select a subset of metrics with `--metrics`:

```bash
# MD + energy only:
binding-metrics-run --input complex.cif --output-dir results/ \
    --metrics energy
# Everything except OpenFold:
binding-metrics-run --input complex.cif --output-dir results/ \
    --metrics energy,interface,geometry,electrostatics
```

OpenFold3 runs in the `openfold3` conda env by default (see [OpenFold3 install](#openfold3-optional) above). Use `--openfold-mode refold` to measure refolding RMSD (binder predicted freely, receptor fixed as template).

**Scoring (individual steps)**

| Command | Description |
|---|---|
| `binding-metrics-interface` | PISA-inspired interface metrics |
| `binding-metrics-energy` | Force-field interaction energy (raw / relaxed / after MD) |
| `binding-metrics-electrostatics` | Coulomb cross-chain interaction energy |
| `binding-metrics-geometry` | Ramachandran, ω planarity, shape complementarity, void volume |
| `binding-metrics-compare` | RMSD between two structures |
| `binding-metrics-openfold` | Parse / run OpenFold3 confidence metrics |
| `binding-metrics-relax` | Implicit-solvent energy minimization; supports multi-model CIFs via `--model N` or `--all-models` |

`binding-metrics-relax` can operate on **multi-model CIFs** (e.g. outputs from homology modelling pipelines):

```bash
# Minimize a single model from a multi-model CIF
binding-metrics-relax --input models.cif --output-dir results/ --md-duration-ps 0 --model 3

# Minimize every model — output is a single multi-model CIF <stem>_minimized.cif
binding-metrics-relax --input models.cif --output-dir results/ --md-duration-ps 0 --all-models
```

**Receptor quality (standalone)**

| Command | Description |
|---|---|
| `binding-metrics-receptor-quality` | MolProbity-style quality assessment for receptor chains — Ramachandran, rotamer outliers, Cβ deviations, bad bonds/angles, clashscore, MolProbity score, B-factors, absolute AMBER energy. Works on receptor-only files or complexes; scores all models in multi-model files. Output to `.csv` or `.json`. |

**Utilities**

| Command | Description |
|---|---|
| `binding-metrics-check-env` | Verify that all runtime dependencies (GPU, OpenMM, …) are working |

Run it after installation to confirm everything is set up correctly:

```bash
binding-metrics-check-env
```

Inside Docker (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker run --rm --gpus all binding-metrics binding-metrics-check-env
```

**Reporting**

| Command | Description |
|---|---|
| `binding-metrics-report` | Re-export a `*_results.json` to JSON or CSV, with optional Markdown summary |

```bash
# Re-export to CSV and regenerate the Markdown summary:
binding-metrics-report --results results/my_run/sample_results.json \
    --format csv --summary

# Generate an HTML report instead:
binding-metrics-report --results results/my_run/sample_results.json \
    --summary --summary-format html
```

The `--summary` flag (available on both `binding-metrics-run` and `binding-metrics-report`) writes a human-readable summary alongside the JSON/CSV output. Use `--summary-format md` (default) for Markdown or `--summary-format html` for a self-contained HTML page. It includes a RAG scorecard (🟢/🟡/🔴) for the key metrics, cyclic topology metadata when present, and per-residue breakdowns for the interface and geometry sections. See [`docs/report_thresholds.md`](docs/report_thresholds.md) for the scorecard thresholds and their scientific rationale.

All scoring tools auto-detect peptide and receptor chains. Pass `--peptide-chain` / `--receptor-chain` to override. See `--help` on each command for full options, or `METRICS.md` for detailed documentation.

---

## Documentation

Full API reference, return value schemas, algorithm notes, and implementation details are in [`docs/metrics.md`](docs/metrics.md).

---

## License

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**.

You may freely use, modify, and distribute this software — including for commercial purposes — provided that you preserve the copyright notice and license text in any distribution. See [`LICENSE`](LICENSE) for the full terms.

If you use BindingMetrics in published work or a commercial product, crediting the original project is appreciated.

---

## About

I'm Simon Crouzet, an independent researcher and consultant in AI/ML for molecular design and drug discovery. BindingMetrics grew out of my own need for principled, reproducible binding quality metrics in peptide design pipelines.

If you find this useful, have ideas, or are working on something in the same space and want to exchange — feel free to reach out. I'm also available for project-based work in computational molecular design and ML workflow development.

- **GitHub:** [@simoncrouzet](https://github.com/simoncrouzet)

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue to discuss significant changes before submitting a pull request. All pull requests should include tests and pass the existing test suite (`pytest`).

---

## Credit & Citation

BindingMetrics is open source under the Apache 2.0 License. You are free to use it in research and commercial work — please credit the original project and respect the license terms.

If you use BindingMetrics in your work, please acknowledge it and feel free to get in touch.

---

## References

- Krissinel, E. & Henrick, K. (2007). Inference of macromolecular assemblies from crystalline state. *J. Mol. Biol.* 372, 774–797.
- Eisenberg, D. & McLachlan, A.D. (1986). Solvation energy in protein folding and binding. *Nature* 319, 199–203.
- Lawrence, M.C. & Colman, P.M. (1993). Shape complementarity at protein/protein interfaces. *J. Mol. Biol.* 234, 946–950.
- Eastman, P. et al. (2017). OpenMM 7. *PLOS Comput. Biol.* 13, e1005659.
- Chen, V.B. et al. (2010). MolProbity: all-atom structure validation for macromolecular crystallography. *Acta Cryst.* D66, 12–21.
- Engh, R.A. & Huber, R. (1991). Accurate bond and angle parameters for X-ray protein structure refinement. *Acta Cryst.* A47, 392–400.
- Ahdritz, G. et al. (2024). OpenFold3. https://github.com/aqlaboratory/openfold-3
