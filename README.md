# BindingMetrics

**BindingMetrics** is a Python toolkit for evaluating designed peptide–protein complexes with physics-based metrics. It sits downstream of a design or prediction run and upstream of experimental validation. A confident model is not the same as a physically reasonable interface, so the package reports interface geometry and energetics, backbone quality and force-field interaction energies next to the confidence scores it parses from OpenFold3 (pLDDT, pTM, ipTM, PAE).

The metrics range from static single-structure analysis (buried SASA, hydrogen bonds, salt bridges, Ramachandran and ω validation, shape complementarity, void volume) to force-field interaction energies after minimization and an optional short MD run. The relaxed output goes through a structural QC of seven checks: finite energy that did not rise, heavy-atom RMSD to the input, finite coordinates, no fused atoms, no stretched or broken bonds, no inverted Cα stereocentre, and an unchanged heavy-atom composition. It flags an exploded or corrupted structure; it is not a clash score. The package also reports receptor drift over an MD trajectory and, when a native structure is available, **reference-based CAPRI accuracy** (DockQ, fnat, i-RMSD, L-RMSD), for instance to benchmark predictions of antibody–antigen complexes. The OpenMM-based steps are seeded, so results are reproducible by default (`--random-seed none` opts into fresh randomness); see [Reproducibility](#reproducibility).

---

## Scope and limits

The package targets peptide–protein complexes first. The cyclic-peptide and non-canonical-residue handling, the examples and the scorecard bands are built around peptide binders. The chain roles, the heteroatom handling and the static metrics do not depend on the binder being a peptide, and they also run on larger binders. On the raw file of the nanobody 7D12 (122 residues) bound to EGFR domain III (205 residues; PDB 4KRL, with its glycans, waters, MES and iodide left in), the interface, shape complementarity, void volume, Ramachandran, ω and Coulomb functions took 5 to 7 s in total in two runs on a laptop CPU.

Limits:
- **Single-chain roles.** The binder and the target are one chain each. A binder of several chains (a Fab) is handled by DockQ only, which maps chains itself; the other metrics score one binder chain at a time.
- **No antibody-specific analysis.** There is no CDR numbering (no ANARCI), and an antibody is scored like any other chain.
- **No calibration for non-peptides.** No calibration against binder and non-binder data ships with the package, for peptides or for other binders. The scorecard bands, the shape-complementarity ranges and `delta_g_int` are heuristics.
- **Chemistry handling is for the binder chain.** Cyclic closures and non-canonical residues are looked for in the binder chain.
- **OpenFold3 interface PDE and PAE** need one token per residue. A prediction with ligands or modified residues gives NaN for them, with a `reason`.

---

## Cyclic peptide support

BindingMetrics supports **cyclic peptides**: head-to-tail (N→C amide) rings, disulfides, lactam bridges and hydrocarbon staples. The closure types it recognises are `head_to_tail`, `disulfide`, `lactam_n_asp`, `lactam_n_glu`, `lactam_c_lys`, `lactam_sc_lys_asp`, `lactam_sc_lys_glu` and `hydrocarbon_staple`, and a peptide can have several (SFTI-1 in 3P8F has a head-to-tail bond and a disulfide). Cyclisation is looked for in the binder (peptide) chain.

The closure is read from the bonds recorded in the input (`_struct_conn` records of a CIF, as written by BoltzGen and other pipelines that support cyclic designs, or CONECT records of a PDB) and from heavy-atom distances for the amide, disulfide and lactam types. The pipeline:

- **Propagates** the bond hint through PDBFixer prep: the N→C bond is written back into `_struct_conn` of the saved `_cleaned.cif`, so downstream runs on that file also detect the cyclic topology
- **Patches the topology** before hydrogens are placed (closure bond, terminal atoms PDBFixer added, lactam templates) so that force-field template matching succeeds
- **Handles cross-chain disulfides** (for example peptide CYS with receptor CYS) alongside the cyclic closure: PDBFixer-detected SS bonds trigger a CYS→CYX rename in memory before `addHydrogens`, and orphaned CYX residues (whose SS partner is on the other chain, severed during per-chain energy decomposition) are converted back to CYS+HG
- **Minimizes the ring** with a closure-bond relaxation stage before the global minimization, and applies backbone φ/ψ restraints that are released in steps during the first 10 ps of MD
- **Scores the closing bond**: for a head-to-tail ring, `compute_ramachandran` and `compute_omega_planarity` include the ring-closing φ, ψ and ω

No flags are needed; the topology is detected and applied automatically. Details are in [`docs/nonstandard.md`](docs/nonstandard.md).

---

## Non-canonical and D-amino-acid residues

Therapeutic peptides are often rich in non-canonical chemistry: cyclosporin A, for instance, is a head-to-tail macrocycle with a D-alanine and seven N-methylated residues. BindingMetrics handles these without user flags:

- **D-amino acids**: the 19 codes of `D_AA_MAP` (the D form of every standard amino acid except glycine) reuse the ff14SB parameters of their L counterparts, and Ramachandran validation mirrors φ/ψ for them. The prepped and the relaxed files keep the input residue names (`DAL`, not `ALA`).
- **N-methylated residues**: sarcosine (SAR, N-Me-Gly), N-Me-Ala, N-Me-Val and N-Me-Leu use curated templates with ForceField_NCAA RESP charges.
- **Phosphorylated residues**: SEP, TPO and PTR use the AMBER phosaa parameters, with their net charge of −2.
- **Other non-canonical residues** (cyclosporin's MeBmt and 2-aminobutyrate, the residues of hydrocarbon staples) are parameterised on the fly with GAFF2. Such a residue carries backbone (external) bonds that general small-molecule parameterisation rejects, so BindingMetrics generates a residue template that records them. Bond orders come from the wwPDB Chemical Component Dictionary that ships with biotite; a residue that is not in it, or does not match its entry, is built with single bonds, a warning says so, and `results["prep"]` and `results["relax"]` list it under `ncaa_bond_order_source` as `"single_bonds"`. Controlled by `--small-molecules auto` (the default in `binding-metrics-run`). The charge model has limits, in particular non-amide backbone N and H charges on these residues; see [`docs/nonstandard.md`](docs/nonstandard.md).

The bundled examples in `data/`:

| File | Content | Interface metrics |
|---|---|---|
| `example_linear_p53_1YCR.pdb` | MDM2 with the p53 peptide, linear | yes |
| `example_bicyclic_sfti1_3P8F.cif` | SFTI-1 with matriptase: head-to-tail ring plus disulfide | yes |
| `example_ncaa_cyclosporin_1CWA.cif` | cyclosporin A with cyclophilin A: D-Ala, N-methylation, GAFF2 residues | yes |
| `example_lactam_somatostatin_1XY4.cif` | somatostatin analogue alone: Lys–Glu lactam, disulfide, D-Trp, IAM | no receptor |
| `example_phospho_1QJB.pdb` | phosphopeptide alone (chain Q, SEP) | no receptor |
| `example_staple_3V3B.pdb` | hydrocarbon-stapled p53 peptide alone (chain C, residues MK8 and 0EH) | no receptor |

---

## Receptor quality assessment

**`binding-metrics-receptor-quality`** is a standalone tool for evaluating receptor structural quality, independent of the main peptide-binding pipeline. It works on receptor-only files or complex structures (non-receptor chains are silently ignored), and scores all models in multi-model PDB/CIF files independently.

The terms follow MolProbity (Chen et al. 2010) as lighter approximations: the clashscore counts heavy-atom overlaps only (hydrogens are ignored, and covalent links and hydrogen-bond pairs are not scored), the rotamer check uses χ1 only, and the Ramachandran regions are boxes. The composite score is therefore indicative and does not compare with published MolProbity values (details in [`docs/metrics.md`](docs/metrics.md#14-receptor-quality)). The goals below are those of MolProbity.

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
from binding_metrics.metrics import compute_receptor_quality

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
binding-metrics-receptor-quality --input ensemble.cif --receptor-chain A --output quality.json
```

---

## Metrics at a glance

**Scores** have a clear direction (higher or lower is better). **Features** are descriptors without an intrinsic quality direction, useful for analysis or as model inputs.

| Category | Metric | Type | Backend |
|---|---|---|---|
| Interface geometry | Buried SASA Δ*A* of heavy atoms (both partners), polar/apolar breakdown | Score | biotite |
| Interface energetics | Solvation term Δ*G*_int (negative = burial favourable; uncalibrated) | Score | biotite |
| Interactions | Cross-chain H-bonds, salt bridges, each with a heuristic energy score | Score | biotite + hydride |
| Electrostatics | Coulomb cross-chain energy of formal charges — negative = net attractive | Score | biotite |
| Backbone geometry | Ramachandran outlier %, ω-angle deviation (ring-closing bond included for head-to-tail rings) | Score | biotite |
| Interface shape | Shape complementarity *S*c (dot-and-normal approximation) | Score | biotite + scipy |
| Interface packing | Buried void volume — large = loose packing; depends on probe and grid | Score | biotite + scipy |
| Force-field energy | *E*_int = *E*_cpx − *E*_pep − *E*_rec (AMBER ff14SB, implicit solvent); raw / relaxed / after MD | Score | OpenMM |
| Structural QC | Seven pass/fail checks on the relaxed output (advisory) | Flag | OpenMM |
| Structure comparison | All-atom and backbone RMSD (Kabsch-aligned) | Score | gemmi |
| Reference accuracy | DockQ, fnat, fnonnat, i-RMSD, L-RMSD + CAPRI class — requires a native reference | Score | DockQ |
| MD trajectory | Receptor backbone drift — aligned (conformational) and raw; ligand RMSD, RMSF, contacts | Score | MDTraj |
| Receptor quality | MolProbity-style terms for a receptor chain (approximate) | Score | biotite + OpenMM |
| Structure prediction | avg_pLDDT, pTM, ipTM, gPDE, interface PDE and PAE — OpenFold3 confidence | Score | OpenFold3 output |
| EvoBind scoring | Interface distance / pLDDT — confidence-weighted binding score (Å) | Score | biotite |
| EvoBind adversarial check | Δ COM between design pose and OF3 prediction after receptor superposition — a large value means the prediction places the binder elsewhere | Score | biotite |
| All of the above | Per-residue breakdowns, per-atom arrays, per-frame series | Feature | — |

Every result value is a score or a feature; the metric registry (`binding_metrics.metrics.registry`) declares the direction, unit and cost class of each metric's headline value. The full list of keys, units and algorithms is in [`docs/metrics.md`](docs/metrics.md).

---

## Installation

### Recommended: conda (GPU-accelerated)

The force-field energies, the relaxation and the MD run on OpenMM, and a CUDA-capable GPU makes them practical: the pipeline warns when MD is requested on the CPU. The conda environment installs OpenMM from conda-forge with CUDA 12.4.

```bash
conda env create -f environment.yml   # creates the binding-metrics conda env
conda activate binding-metrics
binding-metrics-check-env             # verifies OpenMM, the GPU and MDTraj
```

The environment contains:
- OpenMM (CUDA 12.4 build), MDTraj, PDBFixer, gemmi, biotite, hydride, scipy, pandas, matplotlib and markdown, all from conda-forge
- openmmforcefields, openff-toolkit, RDKit and AmberTools (`antechamber` and `sqm` must be on `PATH`) for the GAFF2 parameters of non-canonical residues
- DockQ, installed with pip, and this package in editable mode

`environment.lock.yml` records the exact versions of the development environment; see [Reproducibility](#reproducibility).

### Alternative: pip

Nothing is published to PyPI: the release workflow attaches the sdist and the wheel to a GitHub Release. Install from a checkout of the repository, choosing the extras you need:

```bash
pip install .                       # numpy only
pip install ".[static]"             # every single-structure metric, no OpenMM needed
pip install ".[static,simulation]"  # plus force-field energies and relaxation
```

| Extra | Installs | For |
|---|---|---|
| `static` | biotite, hydride, gemmi, scipy | interface, H-bonds, salt bridges, Coulomb, Ramachandran, ω, shape complementarity, void volume, structure comparison, EvoBind, parsing of OpenFold3 output |
| `simulation` | openmm | force-field energies and relaxation; the plain package, a CPU build (use `environment.yml` for a GPU) |
| `structure` | pdbfixer, gemmi | structure preparation (`binding-metrics-prep`, the pipeline's prep step) |
| `analysis` | mdtraj | trajectory metrics |
| `biotite` | biotite, hydride, scipy | the `static` extra without gemmi; structure comparison needs gemmi |
| `dockq` | DockQ | reference-based CAPRI accuracy |
| `report` | pandas, matplotlib, markdown | the HTML summary (`markdown`) and the CSV output of `binding-metrics-energy` (`pandas`); JSON, CSV and Markdown output of the pipeline needs none of them |
| `gaff` | nothing | placeholder: openmmforcefields, openff-toolkit, RDKit and AmberTools are conda-forge only, so use `environment.yml` |
| `all` | openmm, mdtraj, pdbfixer, gemmi, biotite, hydride, scipy, DockQ, pandas, matplotlib, markdown | everything above except the GAFF2 stack |

A residue that needs GAFF2 parameters (the MeBmt of cyclosporin A, hydrocarbon-staple residues) requires the conda-forge packages, so a pip install alone cannot parameterise it. The extras are also listed in `pyproject.toml`. A name whose dependency is missing raises an error that names the extra to install.

### Docker (GPU, recommended for production)

Pre-built images are available on Docker Hub (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

| Tag | Contents |
|---|---|
| `latest` / `main` | the `environment.yml` environment: OpenMM with CUDA 12.4, the dependencies of the `all` extra and the GAFF2 stack |
| `full` | Everything in `latest` + OpenFold3 conda env |
| `<version>`, `<version>-full` | the same two images, built when a `v*` tag is pushed |

```bash
# Base image (no OpenFold3)
docker pull simoncrouzet/binding-metrics:latest

# Full image (with OpenFold3)
docker pull simoncrouzet/binding-metrics:full
```

**OpenFold3 weights and JIT cache** — model weights (~2.3 GB) are not included in the image, and DeepSpeed's `evoformer_attn` kernel JIT-compiles on first use (~1–3 min via nvcc). Bind-mount both to host directories so they persist across container runs and host reboots:

```bash
mkdir -p ~/.openfold-weights ~/.of3-jit-cache

docker run -it --gpus all --shm-size=8g \
    -v ~/.openfold-weights:/root/.openfold3 \
    -v ~/.of3-jit-cache:/tmp/.cache/torch_extensions \
    -v /path/to/your/structures:/data \
    simoncrouzet/binding-metrics:full bash
```

First run downloads the default checkpoint (`openfold3-p2-155k`) and JIT-builds the kernel. Subsequent runs skip the download and load the cached `.so` in milliseconds. The `binding-metrics` conda env is activated automatically on shell start.

> **Bind mount vs. named volume.** Docker named volumes (`-v openfold3-weights:/root/.openfold3`) work too, but they live under `/var/lib/docker/volumes/` which is often on ephemeral storage in cloud / studio environments (RunPod, Lambda, etc.). Bind-mounting to `~/...` keeps the data in your persistent user home.

**`--shm-size=8g` is required for OpenFold3.** Docker's default `/dev/shm` is 64 MB, which is too small for PyTorch DataLoader workers — OF3 will crash with `RuntimeError: unable to allocate shared memory`. Use `--ipc=host` instead if you prefer the host's IPC namespace (single-tenant only).

**Advanced:**

- `-e BINDING_METRICS_SKIP_WEIGHTS_CHECK=1` — skip the weights check and auto-download. Useful when you know the weights aren't needed (e.g. `binding-metrics-check-env` only verifies that the `openfold3` package is importable, not that weights are present).
- To download a non-default checkpoint or run OpenFold3's integration tests, bypass the entrypoint and run `setup_openfold` interactively:
  ```bash
  docker run -it --rm --gpus all --entrypoint bash \
      -v ~/.openfold-weights:/root/.openfold3 \
      simoncrouzet/binding-metrics:full \
      -c 'conda run -n openfold3 --no-capture-output setup_openfold'
  ```

**Running the base image:**

```bash
docker run --gpus all --rm \
    -v /path/to/your/structures:/data \
    simoncrouzet/binding-metrics:latest \
    binding-metrics-relax --input /data/complex.cif --output /data/relaxed.cif
```

To build the images locally from source:

```bash
docker build --target base -t simoncrouzet/binding-metrics:latest .
docker build --target full -t simoncrouzet/binding-metrics:full .
```

Images are rebuilt and pushed to Docker Hub automatically on every push to `main` and on version tags (`v*`) via GitHub Actions.

### OpenFold3 (optional)

OpenFold3 confidence scoring is optional; every other metric works without it. It requires a GPU, model weights and a compatible Python version, so it belongs in a **dedicated conda environment** named `openfold3`. No extra installs it:

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

The default `--metrics` includes `openfold`, so pass a list without it when OpenFold3 is not installed. The pipeline's OpenFold3 queries use the ColabFold MSA server: the sequences leave the machine, and the alignments, and so the predictions, can change over time. The query JSON carries the seed 42 unless `--openfold-seeds` is given. Run `binding-metrics-check-env` to verify whether OpenFold3 is correctly installed.

---

## Quick Start

Every structure metric works on one binder chain and one target chain. Name them with `peptide_chain` (or `design_chain`, `chain`) and `receptor_chain`, or with the aliases `binder_chain` and `target_chain`; the command-line tools take `--binder-chain` and `--target-chain`. Without them, the metric functions take the smallest protein chain as the binder and the largest as the target. Waters, ions, ligands and glycans that carry a protein chain ID are dropped by default (`hetero="ignore"`); `hetero="keep"`, or `--hetero keep` on `binding-metrics-interface` and `-geometry`, uses every atom of the chain. The buried areas and `delta_g_int` are computed on heavy atoms whatever the protonation of the input (`hydrogens="keep"` or `--hydrogens keep` includes the hydrogens).

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

### DockQ — reference-based CAPRI accuracy

Score a predicted complex against a known native. Requires `pip install "binding-metrics[dockq]"`.
DockQ performs its own optimal chain-mapping search, so antibody–antigen chains named or ordered
differently between prediction and reference are matched automatically.

```python
from binding_metrics import compute_dockq_metrics

result = compute_dockq_metrics("predicted.cif", "native.cif")   # (model, reference)
print(f"DockQ: {result['dockq']:.3f}  ({result['capri_class']})")
for iface in result["interfaces"]:
    print(f"  {iface['chains']}: fnat={iface['fnat']:.2f}  "
          f"i-RMSD={iface['iRMSD']:.2f} Å  L-RMSD={iface['LRMSD']:.2f} Å")
```

In the pipeline, pass `--reference` (single) or `--reference-dir` (batch) to auto-enable it:

```bash
# Single prediction against its native — auto-enables the dockq metric
binding-metrics-run --input predicted.cif --output-dir results/ --reference native.cif

# Batch: each input is matched to a native by filename stem (target1.cif → target1.pdb)
binding-metrics-batch --input-dir preds/ --output-csv metrics.csv --reference-dir natives/
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
# A large Δ COM means OF3 places the binder elsewhere, so the prediction
# does not support the design pose.
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

### Pipeline and batch in Python

`binding-metrics-run` and `binding-metrics-batch` are thin wrappers over two functions that run in-process:

```python
from pathlib import Path
from binding_metrics import run_pipeline, run_batch

results = run_pipeline(
    Path("complex.cif"), Path("results/"),
    skip_prep=True, skip_relax=True,                 # static metrics only
    metrics=frozenset({"interface", "geometry"}),
)
print(results["interface"]["delta_sasa"], results["provenance"]["seed"])

rows = run_batch(
    sorted(Path("designs/").glob("*.cif")), "results/",
    metrics={"interface", "geometry"}, skip_prep=True, skip_relax=True,
    n_workers=4,
    on_result=lambda row: print(row["sample_id"], row["batch_status"]),
)
```

`run_pipeline` takes the options of the `binding-metrics-run` flags as keyword arguments and returns the results dict described under [Results](#results-and-provenance). `binder_chain`, `target_chain` and `openfold_seeds` are keyword arguments too. `run_batch` returns one flat row per path in the order of the paths, whatever the number of workers, and writes the per-sample JSON and log; `on_result` is called with each row as it finishes (in completion order when `n_workers > 1`), and `on_error="raise"` re-raises an exception instead of recording an error row. The CSV of `binding-metrics-batch` is these rows.

The relaxation step is a `Relaxer`. `ImplicitRelaxation` is the one the package ships; pass another implementation, for a different force field or a stub in a test, with `run_pipeline(..., relaxer=...)`. The pipeline reads `success`, `error_message` and the structure path from the returned `RelaxationResult` and records its `to_dict()` under `results["relax"]`:

```python
from binding_metrics import Relaxer, run_pipeline
from binding_metrics.protocols.relaxation import RelaxationResult

class NoRelaxation(Relaxer):
    """Hand the input structure on unchanged."""

    def run(self, input_path, output_dir, sample_id=None):
        return RelaxationResult(
            sample_id=sample_id or input_path.stem,
            success=True,
            minimized_structure_path=str(input_path),
        )

results = run_pipeline(
    Path("complex.cif"), Path("results/"),
    skip_prep=True, relaxer=NoRelaxation(), metrics=frozenset({"interface"}),
)
```

Names that need an optional dependency raise an error that names the extra, and `import binding_metrics` does not import OpenMM, so the static metrics, `run_pipeline` and `run_batch` import on an install without it.

---

## CLI Tools

**Structure preparation**

| Command | Description |
|---|---|
| `binding-metrics-prep` | Fix missing atoms/residues, add hydrogens (`--ph 7.4`), optionally canonicalize non-standard residues (`--canonicalize`); `--random-seed` |
| `binding-metrics-solvate` | Add explicit water box and ions for MD; `--random-seed` seeds the ion placement |

These two commands are composable pipeline steps:

```bash
binding-metrics-prep    --input complex.cif --output cleaned.cif --ph 7.4
binding-metrics-solvate --input cleaned.cif --output solvated.pdb
```

Both print a JSON summary on stdout; library warnings go to stderr so that the JSON stays alone on stdout.

**Full pipeline**

| Command | Description |
|---|---|
| `binding-metrics-run` | Run the pipeline (prep → relax → energy → interface → geometry → electrostatics → OpenFold3, and DockQ with a reference) on a single structure |
| `binding-metrics-batch` | Run it on every structure of a directory and write one CSV row per structure |

```bash
binding-metrics-run \
    --input complex.cif \
    --output-dir results/ \
    --summary                      # also write a human-readable *_report.md
```

Peptide and receptor chains are **auto-detected** (smallest chain = peptide; when more than two chains are present, the one with the most Cα contacts to the peptide is the receptor). Override with `--peptide-chain` / `--receptor-chain`, or their aliases `--binder-chain` / `--target-chain`. An ID that is not in the structure stops the run with an error that lists the chains present.

**Cyclic peptides** are handled automatically: the closure is detected from the input and propagated through prep, relaxation and energy decomposition with no extra flags. See [Cyclic peptide support](#cyclic-peptide-support) above.

Unless `--skip-prep` is given, the pipeline starts with a **prep step** (equivalent to `binding-metrics-prep`) that fixes missing atoms, adds hydrogens and removes waters and other heterogens. The options of `binding-metrics-run`, all of which `binding-metrics-batch` accepts too (batch names the reference option `--reference-dir`):

| Option | Default | Effect |
|---|---|---|
| `--metrics LIST` | `energy,interface,geometry,electrostatics,openfold` | comma-separated subset; `dockq` is added by `--reference` |
| `--skip-prep`, `--skip-relax` | off | skip preparation, or relaxation and MD |
| `--ph` | 7.4 | protonation pH |
| `--keep-water`, `--canonicalize` | off | keep crystallographic waters; rename non-standard residues to their canonical equivalents |
| `--md-duration-ps` | 200 | MD after minimization; 0 minimizes only; a value below 10 saves one frame at the end |
| `--energy-modes` | `relaxed` | any of `raw`, `relaxed`, `after_md` (the `after_md` run lasts 10 ps, independent of `--md-duration-ps`) |
| `--random-seed INT\|none` | 1 | seed of the stochastic steps; `none` for fresh randomness |
| `--reference PATH` | none | native structure; enables DockQ |
| `--openfold-mode`, `--openfold-conda-env`, `--openfold-seeds` | `score`, `openfold3`, seed 42 | OpenFold3 step |
| `--config PATH` | none | TOML file with option defaults (below) |
| `--summary`, `--summary-format`, `--format` | off, `md`, `json` | write a summary with the scorecard; results as JSON or CSV (`binding-metrics-run` only) |
| `--log-file PATH` | none | send all output to a file |

```bash
# Relaxation and energy only:
binding-metrics-run --input complex.cif --output-dir results/ --metrics energy
# Everything except OpenFold:
binding-metrics-run --input complex.cif --output-dir results/ \
    --metrics energy,interface,geometry,electrostatics
```

OpenFold3 runs in the `openfold3` conda env by default (see [OpenFold3 install](#openfold3-optional) above). Use `--openfold-mode refold` to measure refolding RMSD (binder predicted freely, receptor fixed as template).

**Configuration files.** `--config` (on `binding-metrics-run`, `-batch` and `-relax`) reads option defaults from a flat TOML file. Keys are long option names with dashes or underscores, a flag takes `true` or `false`, an option with several values takes a list, and an unknown key is an error. Precedence is the built-in default, then the file, then the command line.

```toml
# run.toml
md-duration-ps = 100
ph = 7.0
metrics = "interface,geometry"
energy-modes = ["relaxed", "raw"]
skip-prep = true
```

```bash
binding-metrics-run --config run.toml --input complex.cif --output-dir results/
```

**Batch runs.**

```bash
binding-metrics-batch --input-dir designs/ --output-csv metrics.csv --workers 4
```

Each structure gets its own directory, JSON report and log under `--output-dir` (by default the directory of the CSV). The CSV has one row per structure, in input order, with a `batch_status` column: `ok` (every step completed), `partial` (the pipeline finished but a step failed; see `batch_failed_steps` and `batch_failed_reasons`) or `error` (the worker raised; see `batch_error`). The last columns, `provenance_*`, record the package version, git sha, Python, OS, OpenMM version, platform and seed. The exit code is non-zero only when no sample is `ok`. Each worker process opens its own CUDA context and takes its share of GPU memory, so several workers on one GPU can run out of memory. Structures are matched to natives for DockQ by file stem (`--reference-dir`, `target1.cif` with `target1.pdb`).

**Scoring (individual steps)**

| Command | Description |
|---|---|
| `binding-metrics-interface` | PISA-inspired interface metrics; `--hetero {ignore,keep}`, `--hydrogens {ignore,keep}` |
| `binding-metrics-energy` | Force-field interaction energy (raw / relaxed / after MD); `--ph`, `--random-seed` |
| `binding-metrics-electrostatics` | Coulomb cross-chain interaction energy |
| `binding-metrics-geometry` | Ramachandran, ω planarity, shape complementarity, void volume; `--hetero {ignore,keep}` |
| `binding-metrics-compare` | RMSD between two structures |
| `binding-metrics-dockq` | DockQ, fnat, fnonnat, i-RMSD, L-RMSD of a prediction against a native |
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

The `--summary` flag (available on both `binding-metrics-run` and `binding-metrics-report`) writes a human-readable summary alongside the JSON/CSV output. Use `--summary-format md` (default) for Markdown or `--summary-format html` for a self-contained HTML page (needs the `report` extra). It has a section for each step (marked skipped when the step did not run), per-residue buried SASA of the peptide in the interface section, and a RAG scorecard (🟢/🟡/🔴, ⬜ for a value that was not computed). The scorecard thresholds are heuristic; see [`docs/report_thresholds.md`](docs/report_thresholds.md).

The individual scoring tools take a peptide and a receptor chain through `--peptide-chain` (or `--design-chain`, `--binder-chain`) and `--receptor-chain` (or `--target-chain`); without them the smallest protein chain is the peptide and the largest the receptor. See `--help` on each command for the options.

---

## Results and provenance

`binding-metrics-run` writes `<sample>_results.json` (`--format csv` writes the flattened CSV instead), and `run_pipeline` returns the same dict. Top-level keys:

| Key | Content |
|---|---|
| `sample_id`, `input`, `total_elapsed_s` | identifiers and wall time |
| `provenance` | package version, git sha (when the package runs from its own checkout), Python, OS, OpenMM version, platform, seed |
| `chains` | resolved chain IDs and residue counts |
| `prep` | what preparation changed: `removed_heterogens`, `n_removed_waters`, `kept_nonstandard`, `n_missing_atoms_rebuilt`, `n_missing_residue_gaps`, and `ncaa_bond_order_source` for GAFF2 residues |
| `relax` | energies, RMSD and RMSF, the OpenMM `platform`, and the structural QC: `qc_passed`, `qc_failed_checks` and (in the JSON) `qc_checks` |
| `energy`, `interface`, `geometry`, `electrostatics`, `dockq`, `openfold` | one dict per metric: `{"skipped": True}` when it did not run, `{"error": message}` when it failed |
| `nonfinite_fields` | the JSON paths of every NaN or infinite value |

A value that could not be computed keeps its NaN, 0 or None, and its dict gains a string under `reason` that says why; a dict without `reason` was computed in full. The QC of the relaxed structure is advisory: a failed check logs a warning and changes neither the results nor the exit code. `binding-metrics-run` exits with 1 when a step failed, after writing the partial results. The schemas are in [`docs/metrics.md`](docs/metrics.md#15-pipeline-results-and-provenance).

---

## Logging

Library code logs through `logging` and never configures it on import. The command-line tools call `binding_metrics.utils.configure_logging()`, which sends records up to WARNING to stdout and ERROR and above to stderr, as the bare message. `binding-metrics-prep` and `-solvate` send warnings to stderr as well, so that their JSON summary is alone on stdout. `--log-file PATH` redirects both streams of a command to a file. `binding-metrics-batch` writes one log per sample, `<output-dir>/<sample>/<sample>.log`, unless `--log-file` names one file shared by all samples.

In a script, call `configure_logging()` for the same behaviour, or attach handlers to the `binding_metrics` logger yourself:

```python
import logging
from binding_metrics.utils import configure_logging

configure_logging(logging.INFO)
```

---

## Reproducibility

- **Seeds.** Hydrogen placement, PDBFixer's rebuilding of missing atoms, the MD initial velocities and Langevin noise, the ion placement of `binding-metrics-solvate` and the hydrogen placement in the receptor energy term of `binding-metrics-receptor-quality` are seeded. The default seed is 1. Set it with `--random-seed INT` on `binding-metrics-run`, `-batch`, `-relax`, `-energy`, `-prep`, `-solvate` and `-receptor-quality`, or with `random_seed=` in the API; `--random-seed none` draws fresh randomness, for instance to generate independent MD replicas. The static metrics have no random step.
- **OpenFold3.** The seed above does not drive it. The query JSON carries the seed 42 unless `--openfold-seeds` is given, and the MSA server can return different alignments over time.
- **GPU precision.** CUDA runs in mixed precision and its force reduction order is not deterministic, so energies and MD from one seed can differ in the last digits between GPU runs.
- **Provenance.** Every results file carries the `provenance` block (package version, git sha, Python, OS, OpenMM version, platform, seed), and batch CSV rows carry it as `provenance_*` columns, so a result can be tied to the code and settings that produced it.
- **Environments.** `environment.yml` is the specification that CI and the Dockerfile build from. `environment.lock.yml` is a snapshot of the exact versions of the development environment (`conda env create -n binding-metrics -f environment.lock.yml`, then `pip install --no-deps -e .`); neither CI nor the Dockerfile reads it, and its header says how to regenerate it. The Docker images are built from `environment.yml` (see [Docker](#docker-gpu-recommended-for-production)).
- **Releases.** Pushing a `v*` tag builds the sdist and the wheel and attaches them to a GitHub Release. Nothing is published to PyPI.

---

## Documentation

- [`docs/metrics.md`](docs/metrics.md): every metric with its signature, result keys, units and algorithm notes; the pipeline results; the metric registry
- [`docs/nonstandard.md`](docs/nonstandard.md): D-amino acids, N-methylated and phosphorylated residues, the GAFF2 route, cyclic closures and their limits
- [`docs/report_thresholds.md`](docs/report_thresholds.md): the scorecard thresholds
- [`CHANGELOG.md`](CHANGELOG.md): what changed, with the results that differ from earlier versions
- [`METRICS.md`](METRICS.md) points to `docs/metrics.md`

---

## License

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**.

You may freely use, modify, and distribute this software — including for commercial purposes — provided that you preserve the copyright notice and license text in any distribution. See [`LICENSE`](LICENSE) for the full terms.

If you use BindingMetrics in published work or a commercial product, crediting the original project is appreciated.

---

## About

I'm Simon Crouzet, an independent researcher and consultant in AI/ML for molecular design and drug discovery. BindingMetrics grew out of my own need for reproducible quality metrics in peptide design pipelines.

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
- The OpenFold3 Team (2025). OpenFold3-preview. https://github.com/aqlaboratory/openfold-3, doi:10.5281/zenodo.19001000
- Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500.
