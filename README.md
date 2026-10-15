# BindingMetrics

**BindingMetrics** is a Python toolkit for evaluating designed peptide–protein complexes with physics-based metrics. It reports interface geometry and energetics, backbone quality and force-field interaction energies next to the confidence scores (pLDDT, pTM, ipTM, PAE) that it parses from the output of OpenFold3, AlphaFold2 and ColabFold, Boltz-2 and Protenix.

The metrics range from static single-structure analysis (buried SASA, hydrogen bonds, salt bridges, Ramachandran and ω validation, shape complementarity, void volume) to force-field interaction energies after minimization and an optional short MD run. The relaxed output goes through a structural QC of seven pass/fail checks that flags an exploded or corrupted structure; it is not a clash score ([Results and provenance](#results-and-provenance)). The package also reports receptor drift over an MD trajectory and, when a native structure is available, **reference-based CAPRI accuracy** (DockQ, fnat, i-RMSD, L-RMSD). The OpenMM-based steps are seeded, so results are reproducible by default (`--random-seed none` opts into fresh randomness; see [Reproducibility](#reproducibility)).

---

## Scope and limits

The package targets peptide–protein complexes. The cyclic-peptide and non-canonical-residue handling, the examples and the scorecard bands are built around peptide binders. The chain roles, the heteroatom handling and the static metrics do not depend on the binder being a peptide, and they also run on larger binders (nanobody–antigen complexes, for instance).

Limits:
- **Single-chain roles.** The binder and the target are one chain each. A binder of several chains (a Fab) is handled by DockQ only, which maps chains itself. The relaxation and the energy step remove every other protein chain, with a warning that names it, so a receptor of several chains has to be reduced to one before it goes in.
- **No antibody-specific analysis.** There is no CDR numbering (no ANARCI); an antibody is scored like any other chain.
- **No calibration.** No calibration against binder and non-binder data ships with the package. The scorecard bands, the shape-complementarity ranges and `delta_g_int` are heuristics.
- **Chemistry handling is for the binder chain.** Cyclic closures and non-canonical residues are looked for in the binder chain.
- **Interface PDE and PAE** are cut by chain from one token per residue, or from the token layout of the record when the parser builds one (the OpenFold3, Boltz-2 and Protenix adapters do; Protenix needs its full-data file). A prediction with ligands or modified residues has them when the layout fits the matrices; otherwise they are NaN, with a `reason`.

---

## Cyclic peptide support

Supported closures: head-to-tail (N→C amide) rings, disulfides, lactam bridges and hydrocarbon staples (`head_to_tail`, `disulfide`, `lactam_n_asp`, `lactam_n_glu`, `lactam_c_lys`, `lactam_sc_lys_asp`, `lactam_sc_lys_glu`, `hydrocarbon_staple`); a peptide can have several (SFTI-1 in 3P8F has a head-to-tail bond and a disulfide). The closure is looked for in the binder chain. It is read from the bonds recorded in the input (`_struct_conn` records of a CIF, as written by BoltzGen and other pipelines that support cyclic designs, or CONECT records of a PDB) and from heavy-atom distances. At the prep step a head-to-tail ring and a lactam (a lysine NZ with an aspartate CG, a glutamate CD or the C-terminus) that the input does not list are found from the distance (below 0.20 nm), a disulfide from the S–S distance, and the other closures are taken from the links of the input.

The pipeline writes the N→C bond back to `_struct_conn` of the saved `_cleaned.cif`, patches the topology before hydrogens are placed, handles cross-chain disulfides, relaxes the ring with a closure-bond stage before the global minimization (backbone φ/ψ restraints are released in steps during the first 10 ps of MD), and includes the ring-closing φ, ψ and ω in `compute_ramachandran` and `compute_omega_planarity`. No flags are needed; see [`docs/nonstandard.md`](docs/nonstandard.md).

---

## Non-canonical and D-amino-acid residues

Handled without user flags:

- **D-amino acids**: the 19 codes of `D_AA_MAP` (the D form of every standard amino acid except glycine) reuse the ff14SB parameters of their L counterparts, and Ramachandran validation mirrors φ/ψ for them. The prepped and the relaxed files keep the input residue names (`DAL`, not `ALA`).
- **N-methylated residues**: sarcosine (SAR, N-Me-Gly), N-Me-Ala, N-Me-Val and N-Me-Leu use curated templates with ForceField_NCAA RESP charges.
- **Phosphorylated residues**: SEP, TPO and PTR use the AMBER phosaa parameters, with their net charge of −2.
- **Other non-canonical residues** (the MeBmt and 2-aminobutyrate of cyclosporin A, the residues of hydrocarbon staples) are parameterised on the fly with GAFF2, with a residue template that records their backbone (external) bonds. Bond orders come from the wwPDB Chemical Component Dictionary that ships with biotite; a residue that is not in it, or does not match its entry, is built with single bonds, a warning says so, and `results["prep"]` and `results["relax"]` list it under `ncaa_bond_order_source` as `"single_bonds"`. Controlled by `--small-molecules auto` (the default in `binding-metrics-run`); see [`docs/nonstandard.md`](docs/nonstandard.md).

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

**`binding-metrics-receptor-quality`** is a standalone tool for the structural quality of a receptor, independent of the peptide-binding pipeline. It works on receptor-only files or complex structures (non-receptor chains are ignored) and scores every model of a multi-model PDB/CIF file.

The terms follow MolProbity (Chen et al. 2010) as lighter approximations (heavy-atom clashscore, χ1-only rotamers, box Ramachandran regions), so the composite score does not compare with published MolProbity values ([`docs/metrics.md`](docs/metrics.md#14-receptor-quality)). The goals below are those of MolProbity.

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

In Python, `compute_receptor_quality(path, receptor_chain=None, device="cuda")` returns `models` (one dict per model) and a `summary` (`molprobity_score`, `ramachandran_favoured_pct`, `clashscore`, `rotamer_outlier_pct`, `energy_kJ_mol`, `best_model_index`, ...; keys in [`docs/metrics.md`](docs/metrics.md#14-receptor-quality)).

```bash
# CLI — output format auto-detected from extension (.csv or .json)
binding-metrics-receptor-quality --input receptor.pdb --output quality.csv
binding-metrics-receptor-quality --input ensemble.cif --receptor-chain A --output quality.json
```

---

## Metrics at a glance

**Scores** have a direction (higher or lower is better). **Features** are descriptors without an intrinsic quality direction, for analysis or as model inputs.

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
| Structure prediction | avg_pLDDT, pTM, ipTM, gPDE, interface PDE and PAE — confidence of OpenFold3, AlphaFold2 (ColabFold), Boltz-2 or Protenix output | Score | model output |
| EvoBind scoring | Interface distance / pLDDT — confidence-weighted binding score (Å) | Score | biotite |
| EvoBind adversarial check | Δ COM between design pose and the prediction after receptor superposition — a large value means the prediction places the binder elsewhere | Score | biotite |
| All of the above | Per-residue breakdowns, per-atom arrays, per-frame series | Feature | — |

The metric registry (`binding_metrics.metrics.registry`) declares the direction, unit and cost class of each metric's headline value. The full list of keys, units and algorithms is in [`docs/metrics.md`](docs/metrics.md).

---

## Installation

### Recommended: conda (GPU-accelerated)

The force-field energies, the relaxation and the MD run on OpenMM; a CUDA-capable GPU makes them practical, and the pipeline warns when MD is requested on the CPU. `--device cuda` and `platform="auto"` use the CUDA platform only when a context can be created on it, and the CPU platform otherwise. The conda environment installs OpenMM from conda-forge with CUDA 12.4.

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
| `static` | biotite, hydride, gemmi, scipy | interface, H-bonds, salt bridges, Coulomb, Ramachandran, ω, shape complementarity, void volume, structure comparison, EvoBind, parsing of prediction output (OpenFold3, AlphaFold2 and ColabFold, Boltz-2, Protenix) |
| `simulation` | openmm | force-field energies and relaxation; the plain package, a CPU build (use `environment.yml` for a GPU) |
| `structure` | pdbfixer, gemmi | structure preparation (`binding-metrics-prep`, the pipeline's prep step) |
| `analysis` | mdtraj | trajectory metrics |
| `biotite` | biotite, hydride, scipy | the `static` extra without gemmi; structure comparison needs gemmi |
| `dockq` | DockQ | reference-based CAPRI accuracy |
| `report` | pandas, markdown | the HTML summary (`markdown`) and the CSV output of `binding-metrics-energy` (`pandas`); JSON, CSV and Markdown output of the pipeline needs none of them |
| `openfold`, `openfold3` | openfold3 | the OpenFold3 distribution; two names for one extra (see [OpenFold3](#openfold3-optional)) |
| `gaff` | nothing | placeholder: openmmforcefields, openff-toolkit, RDKit and AmberTools are conda-forge only, so use `environment.yml` |
| `all` | openmm, mdtraj, pdbfixer, gemmi, biotite, hydride, scipy, DockQ, pandas, markdown | everything above except OpenFold3 and the GAFF2 stack |

A residue that needs GAFF2 parameters (the MeBmt of cyclosporin A, hydrocarbon-staple residues) requires the conda-forge packages, so a pip install alone cannot parameterise it. A name whose dependency is missing raises an error that names the extra to install.

### Docker (GPU, recommended for production)

Pre-built images are available on Docker Hub (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

| Tag | Contents |
|---|---|
| `latest` | the full image of the newest stable release: the `environment.yml` environment (OpenMM with CUDA 12.4, the dependencies of the `all` extra and the GAFF2 stack) and the OpenFold3 conda env |
| `latest-base` | the base image of the newest stable release: the `environment.yml` environment without OpenFold3 |
| `<major>.<minor>`, `<major>.<minor>-base` | the full and the base image of the newest release of that minor version, built when a `v*` tag is pushed |
| `main`, `main-base` | the full and the base image as of the last push to `main` |

```bash
# Full image (with OpenFold3)
docker pull simoncrouzet/binding-metrics:latest

# Base image (no OpenFold3)
docker pull simoncrouzet/binding-metrics:latest-base
```

**OpenFold3 weights and kernel cache.** The model weights (~2.3 GB) are not included in the image, and Triton compiles the kernels of OpenFold3 on first use. Bind-mount both to host directories so they persist across container runs:

```bash
mkdir -p ~/.openfold-weights ~/.of3-triton-cache

docker run -it --gpus all --shm-size=8g \
    -v ~/.openfold-weights:/root/.openfold3 \
    -v ~/.of3-triton-cache:/tmp/triton_cache \
    -v /path/to/your/structures:/data \
    simoncrouzet/binding-metrics:latest bash
```

The first run downloads the default checkpoint (`openbind-2025-06-30-174k`, the OpenBind-0 weights of openfold3 0.5) and compiles the kernels. The second mount is the `TRITON_CACHE_DIR` that the image sets. A volume that holds Preview2 weights (`of3-p2-*.pt`) keeps them, but they do not load into openfold3 0.5: the entrypoint downloads the OpenBind-0 file next to them. The `binding-metrics` conda env is activated on shell start.

**`--shm-size=8g` is required for OpenFold3.** The default `/dev/shm` of Docker is 64 MB, too small for PyTorch DataLoader workers: OpenFold3 stops with `RuntimeError: unable to allocate shared memory`. `--ipc=host` works as well (single-tenant only).

**Advanced:**

- `-e BINDING_METRICS_SKIP_WEIGHTS_CHECK=1` skips the weights check and the auto-download. `binding-metrics-check-env` reports the `openfold3` version and whether the default checkpoint is on disk.
- To download to another folder, bypass the entrypoint (`--entrypoint bash`) and run `setup_openfold` interactively; `setup_openfold --non-interactive` is what the entrypoint runs.

Running the base image:

```bash
docker run --gpus all --rm \
    -v /path/to/your/structures:/data \
    simoncrouzet/binding-metrics:latest-base \
    binding-metrics-relax --input /data/complex.cif --output /data/relaxed.cif
```

Build locally with `docker build --target full -t simoncrouzet/binding-metrics:latest .` or `docker build --target base -t simoncrouzet/binding-metrics:latest-base .`. GitHub Actions rebuilds and pushes both images on every push to `main` (tags `main` and `main-base`) and on version tags (`v*`), which push `<major>.<minor>` and `<major>.<minor>-base` and, for a stable version, `latest` and `latest-base`. A version tag is published only when it is `v` plus the `pyproject.toml` version and `ci.yml` has passed on the tagged commit; `latest` and `latest-base` follow the last release and `main` and `main-base` follow the main branch.

### OpenFold3 (optional)

OpenFold3 confidence scoring is optional; every other metric works without it. It needs a GPU, model weights and a compatible Python version, so it belongs in a **dedicated conda environment** named `openfold3`. The `openfold` extra (alias `openfold3`) lists the `openfold3` distribution. openfold3 0.5 supports Python 3.10 to 3.13 and BindingMetrics needs 3.11 or later; installing both into one environment has not been tested (openfold3 0.5.0 requires `rdkit<2026`), so a separate environment is the route below:

```bash
conda create -n openfold3 python=3.10   # 3.10 to 3.13 work
conda activate openfold3
pip install "openfold3>=0.5.0,<0.6"
setup_openfold --non-interactive   # downloads the OpenBind-0 weights (about 2.3 GB)
```

`environment_openfold3.yml` does the same in one step. openfold3 0.5 stops without its default checkpoint (`of3-ob-2025-06-30-174k.pt`) instead of downloading it at first use, and Preview2 weights from an older install do not load into it, so run `setup_openfold --non-interactive` once.

`binding-metrics-run` uses the `openfold3` environment by default (`--openfold-conda-env openfold3`):

```bash
# Uses the openfold3 env by default
binding-metrics-run --input complex.cif --output-dir results/

# Explicit env name if you used a different name
binding-metrics-run --input complex.cif --output-dir results/ \
    --openfold-conda-env my_openfold_env

# Skip OpenFold entirely
binding-metrics-run --input complex.cif --output-dir results/ \
    --metrics energy,interface,geometry,electrostatics
```

The default `--metrics` includes `openfold`, so pass a list without it when OpenFold3 is not installed. The OpenFold3 step uses the ColabFold MSA server unless `--openfold-no-msa-server` is given (the sequences leave the machine, and the alignments can change over time), samples with seed 42 unless `--openfold-seeds` is given (the first seed given, first sample, is scored), gives the template of each chain as a structure (`--openfold-templates structure`, the default; `alignment` gives an A3M that the server replaces, so a run with the server on has no template), and sends a head-to-tail binder of standard residues as cyclic (`--openfold-cyclic auto`, the default; `on` always, `off` never). The result records what became of each template (`results["openfold"]["templates"]`, or `results["prediction"]["templates"]` with `--predictor of3`). Run `binding-metrics-check-env` to verify the installation; the details are in [`docs/metrics.md`](docs/metrics.md#12-openfold3-confidence-scores).

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

### Structure-prediction confidence scores

Parsers read the output of OpenFold3, AlphaFold2 and ColabFold, Boltz-2 and Protenix into one record, a `PredictedPose` (`of3`, `af2`, `boltz2`, `protenix`). Each model can also be run from here, in its own conda environment or from `PATH` (`--predictor MODEL` without `--prediction-dir`; [OpenFold3](#openfold3-optional) has its own setup), or you run the model yourself and pass its output directory. pLDDT and ipTM are calibrated per model: compare them within one model.

Tested with OpenFold3 0.5.0 (OpenBind-0 weights; 1YCR, 1CWA, 3P8F) and Boltz-2 2.2.1 (1YCR, 1CWA, 3P8F, 1XY4). Not run: Protenix and ColabFold (stand-in process only), AlphaFold2. See [`docs/metrics.md`](docs/metrics.md#model-adapters).

```python
from binding_metrics import compute_openfold_metrics, compute_prediction_metrics

metrics = compute_openfold_metrics("./openfold_out", query_name="my_complex", seed=1, sample=1)
print(f"pLDDT: {metrics['avg_plddt']:.1f}  ipTM: {metrics['iptm']:.3f}  gPDE: {metrics['gpde']:.3f} Å")

# The same summary for the output of any model with an adapter (af2, boltz2, of3, protenix)
metrics = compute_prediction_metrics(
    "./boltz_out", model="boltz2", name="my_complex", binder_chain="B", receptor_chain="A"
)
```

`compute_openfold_metrics` also returns `seed_value`, the seed in the name of the seed directory that was parsed; `compute_prediction_metrics` returns the same dictionary with a `model` key.

### DockQ — reference-based CAPRI accuracy

Scores a predicted complex against a known native. Requires `pip install "binding-metrics[dockq]"`. DockQ searches the chain mapping itself, so chains named or ordered differently in prediction and reference are matched.

```python
from binding_metrics import compute_dockq_metrics

result = compute_dockq_metrics("predicted.cif", "native.cif")   # (model, reference)
print(f"DockQ: {result['dockq']:.3f}  ({result['capri_class']})")
for iface in result["interfaces"]:
    print(f"  {iface['chains']}: fnat={iface['fnat']:.2f}  "
          f"i-RMSD={iface['iRMSD']:.2f} Å  L-RMSD={iface['LRMSD']:.2f} Å")
```

In the pipeline, `--reference` (single) or `--reference-dir` (batch) enables it:

```bash
# Single prediction against its native — enables the dockq metric
binding-metrics-run --input predicted.cif --output-dir results/ --reference native.cif

# Batch: each input is matched to a native by filename stem (target1.cif → target1.pdb)
binding-metrics-batch --input-dir preds/ --output-csv metrics.csv --reference-dir natives/
```

When the `openfold` metric is enabled in `binding-metrics-run`, the EvoBind scores are computed and merged into the OpenFold3 result without another model call. With `--predictor` they are merged into `results["prediction"]`, computed on the prediction of that model.

### EvoBind scoring

```python
from binding_metrics.metrics.evobind import (
    compute_evobind_score,
    compute_evobind_adversarial_check,
)

# Primary score on the OF3 prediction: interface distance / pLDDT (Å); lower is better
score = compute_evobind_score(
    "of3_prediction.cif",
    plddt_per_atom=metrics["plddt_per_atom"],
    binder_chain="B",
    receptor_chain="A",
)
print(f"EvoBind score: {score['evobind_score']:.2f} Å  (if_dist: {score['if_dist_pep_to_rec']:.2f} Å)")

# Adversarial check: Δ COM between the input design pose and the prediction
check = compute_evobind_adversarial_check(
    design_structure_path="input_design.cif",
    afm_structure_path="of3_prediction.cif",
    binder_chain="B",
    receptor_chain="A",
    afm_plddt_per_atom=metrics["plddt_per_atom"],
)
print(f"Δ COM: {check['delta_com_angstrom']:.2f} Å  adversarial score: {check['evobind_adversarial_score']:.1f}")
```

Both functions implement the losses from [Bryant et al. (2025) *EvoBind*, Communications Chemistry](https://doi.org/10.1038/s42004-025-01601-3). The primary score is `if_dist / (pLDDT/100)`; the adversarial score is `mean_if_dist × (100/pLDDT) × ΔCOM`; a large ΔCOM means the prediction places the binder elsewhere than the design.

Both scores also take the records that the parsers read (`compute_evobind_score_from_record` and `compute_evobind_adversarial_from_records`, with `get_parser(model).load(...)` from `binding_metrics.predictors`), so the second prediction can come from any model with a parser. Chain IDs are yours, after the `chain_map` of each record. pLDDT is calibrated per model: compare a score between designs only when one model made the predictions it divides by.

### Pose sets: several seeds and models of one construct

`score_pose_set` places the poses of one construct (several seeds, several models) on the receptor of a reference structure, measures each with the registry metrics you ask for, compares the poses with one another (pairwise peptide RMSD, clusters, medoids, interface overlap) and summarises them per back-end, with `n_ok` next to `n_total` so that a failed pose is counted. The result is one JSON-ready dictionary, which `write_pose_report` writes as JSON, CSV, Markdown or HTML.

```python
from binding_metrics.poses import Reference, score_pose_set, write_pose_report
from binding_metrics.predictors import get_parser

poses = []
for model, out_dir in (("boltz2", "./boltz_out"), ("of3", "./openfold_out")):
    parser = get_parser(model)
    poses += [
        parser.load(out_dir, "my_complex", seed_index=s.seed_index, sample=s.sample)
        for s in parser.list_samples(out_dir, "my_complex")
    ]

result = score_pose_set(
    poses, binder_chain="B", target_chain="A", metrics=["interface", "hbonds"],
    reference=Reference("crystal.cif", receptor_chain="A", binder_chain="B"),
)
entry = result["aggregates"]["boltz2"]["interface.delta_sasa"]
print(f"{entry['n_ok']} of {entry['n_total']} poses, mean {entry['mean']:.0f} Å²")
print(result["representatives"])
write_pose_report(result, "md", "my_complex_poses.md")
```

The layer applies no threshold. Tested on synthetic pose sets (`binding_metrics.testing.make_synthetic_poses`), the bundled complexes and the fixtures of the adapters; not run on real co-folding output. See [`docs/pose_sets.md`](docs/pose_sets.md).

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

`run_pipeline` takes the options of the `binding-metrics-run` flags as keyword arguments (`binder_chain`, `target_chain`, `openfold_seeds`, `openfold_cyclic`, `openfold_use_msa_server`, `on_unmappable_residue`, `binder_type`, `on_incompatible`, `preflight_only`, `predictor`, `prediction_dir`, `prediction_binder_chain`, `prediction_target_chain`, `prediction_cache`, `rerun_predictions`, ...; `run_batch` takes them too) and returns the results dict described under [Results and provenance](#results-and-provenance). `run_batch` returns one flat row per path in the order of the paths and writes the per-sample JSON and log; `on_result` is called with each row as it finishes (in completion order when `n_workers > 1`), and `on_error="raise"` re-raises an exception instead of recording an error row. The keyword-only `nonfinite` (default `"keep"`) is the mode of the per-sample JSON reports, including the blocks that the OpenFold3 and prediction steps add afterwards; the CSV rows do not depend on it. The CSV of `binding-metrics-batch` is these rows.

The relaxation step is a `Relaxer`; `ImplicitRelaxation` is the one the package ships. Pass another implementation, for a different force field or a stub in a test, with `run_pipeline(..., relaxer=...)`: subclass `binding_metrics.Relaxer`, implement `run(input_path, output_dir, sample_id=None)` and return a `binding_metrics.protocols.relaxation.RelaxationResult`. The pipeline reads `success`, `error_message` and the structure path from it and records its `to_dict()` under `results["relax"]`.

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

Peptide and receptor chains are auto-detected (smallest chain = peptide; with more than two chains, the one with the most Cα contacts to the peptide is the receptor). Override with `--peptide-chain` / `--receptor-chain`, or their aliases `--binder-chain` / `--target-chain`. An ID that is not in the structure stops the run with an error that lists the chains present. Cyclic closures are detected from the input and carried through prep, relaxation and energy decomposition with no extra flags ([Cyclic peptide support](#cyclic-peptide-support)).

Unless `--skip-prep` is given, the pipeline starts with a prep step (equivalent to `binding-metrics-prep`) that fixes missing atoms, adds hydrogens and removes waters and other heterogens. The options of `binding-metrics-run`, all of which `binding-metrics-batch` accepts too (batch names the reference option `--reference-dir`):

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
| `--openfold-mode`, `--openfold-conda-env`, `--openfold-seeds`, `--openfold-cyclic`, `--openfold-no-msa-server`, `--openfold-templates` | `score`, `openfold3`, seed 42, `auto`, server on, `structure` | OpenFold3 step; `--openfold-cyclic {auto,on,off}`; `--openfold-no-msa-server` runs it with a dummy MSA of the query sequence; `--openfold-templates {structure,alignment}` |
| `--on-unmappable-residue {error,x}` | `error` | a residue OpenFold3 cannot take stops the run before the model starts; `x` sends an `X` in its place |
| `--binder-type {auto,peptide,miniprotein,nanobody,antibody}` | `auto` | what the binder is, for the pre-flight checks that depend on it ([Pre-flight check](#pre-flight-check)) |
| `--on-incompatible {error,skip,warn}` | `error` | an input that a requested step or model cannot take is refused before anything runs (`error`), left out with its reason (`skip`) or logged (`warn`) |
| `--designed-with {boltzgen,bindcraft,colabdesign}` | none | the pipeline that made the designed binder (BoltzGen ran Boltz-2; BindCraft and ColabDesign ran AlphaFold2). Without it nothing is decided and a run with a model step logs once at INFO level that independence was not checked |
| `--design-policy {error,skip,warn}` | `skip` | the EvoBind adversarial check when the model of the run is one that `--designed-with` ran: `skip` leaves it out with a warning, `warn` runs it and marks the result `independent_of_design: false`, `error` refuses before anything runs |
| `--preflight-only` | off | print the pre-flight plan and stop; exit status 1 when something is refused |
| `--predictor {af2,boltz2,of3,protenix}` | none | the `openfold` step reads the prediction of this model into `results["prediction"]` and runs the model at most once ([Other prediction models](#other-prediction-models-and-the-run-once-store)) |
| `--prediction-weights PATH` | the model's own weights | custom weights (a file for OpenFold3, `--inference-ckpt-path`; a directory for other runners), identified by content in the store key; a usage error with `--prediction-dir` |
| `--prediction-mode {predict,refold,score,score-lock,refold-lock}` | the mode of the runner: `--openfold-mode` (`score`) for `of3`, `score` for `boltz2`, `predict` for `af2` and `protenix`; not stated for `--prediction-dir` | how the model is used; checked against the model and its runner (`of3`: `refold`, `score`; `boltz2`: all five; `af2`, `protenix`: `predict`) before anything runs; recorded as `mode` |
| `--prediction-cyclic {auto,on,off}`, `--prediction-no-msa-server`, `--prediction-conda-env NAME` | `auto`, server on, the executable on `PATH` | for `--predictor` run from here: whether the binder is given as cyclic (ColabFold refuses a value), whether an MSA server is used, the conda environment of the model. For `of3` they are `--openfold-cyclic`, `--openfold-no-msa-server` and `--openfold-conda-env`; for another model the `--openfold-*` spelling is refused |
| `--prediction-msa-policy {default,forbid-server}`, `--prediction-receptor-msa PATH` | `default`, none | `forbid-server` forbids sending any sequence to an MSA server; it needs `--prediction-no-msa-server` and a receptor MSA, and a run that would use the server is refused under every `--on-incompatible` value. The receptor MSA is an a3m (`.csv` for Boltz-2) or the word `single`; ColabFold takes `single` only |
| `--prediction-lock-threshold ANGSTROM` | 2.0 (Boltz-2's runner) | threshold of the forced template of `score-lock` and `refold-lock` (Boltz-2 only) |
| `--prediction-dir DIR` | none | read the output you made with the model (never run); in `-batch` the root with one output per sample ID |
| `--prediction-binder-chain`, `--prediction-target-chain` | the input's IDs | chain IDs inside the prediction when they differ |
| `--prediction-cache DIR`, `--rerun-predictions` | `<output-dir>/predictions` (`-batch`: `_predictions`), off | the store of finished predictions; run again although it has one |
| `--config PATH` | none | TOML file with option defaults (below) |
| `--summary`, `--summary-format`, `--format` | off, `md`, `json` | write a summary with the scorecard; results as JSON or CSV (`binding-metrics-run` only) |
| `--nonfinite {keep,null}` | `keep` | JSON results: `keep` writes NaN and infinity as the tokens `NaN`, `Infinity` and `-Infinity`; `null` writes `null` and lists the paths in `nonfinite_fields` (`-run`, `-batch`, `-report`, `-relax`, `-receptor-quality`; not the CSV) |
| `--log-file PATH` | none | send all output to a file |

```bash
# Relaxation and energy only:
binding-metrics-run --input complex.cif --output-dir results/ --metrics energy
# Everything except OpenFold:
binding-metrics-run --input complex.cif --output-dir results/ \
    --metrics energy,interface,geometry,electrostatics
```

With the OpenFold3 step, `--openfold-mode score` (the default) gives each chain its own structure from the input as a template and OpenFold3 places the binder itself: its confidences refer to its own pose, and `binder_ca_rmsd` (binder Cα RMSD against the input, in the receptor frame) and `delta_com_angstrom` of the EvoBind adversarial check give the distance from the input pose. `--openfold-mode refold` templates only the receptor and predicts the binder from its sequence; `binder_ca_rmsd` is then the refolding RMSD. `--openfold-no-msa-server` (`use_msa_server=False` in the Python functions, `--no-msa-server` of `binding-metrics-openfold`) switches the MSA server off. With `--prediction-receptor-msa`, the toolkit checks that the first sequence of the file is the receptor's before anything is written.

#### Pre-flight check

Some inputs cannot go through some steps: a binder with a disulfide sent to OpenFold3, which reads only a head-to-tail closure; a binder residue that its query builder cannot express; `interface` on a structure with no receptor chain. Before anything runs (preparation, relaxation, any model run and the prediction store), the pipeline compares the input with the declared limits of every requested step and model and refuses it with all the problems at once, each with the fact found, the requirement, the reason and a fix:

```
Pre-flight check failed: 1 incompatibility between the input and what was requested (policy: error).
Input: binder chain I: 14 residues; type peptide (estimated from size); closures head_to_tail, disulfide; ...

predictor OpenFold3 0.5.0: closures
    found:    the binder has a disulfide bond (CYS 3.SG - CYS 11.SG)
    requires: closures limited to: none, head_to_tail
    why:      OpenFold3 0.5.0 takes one kind of ring closure: ...
    fix:      use a predictor whose declared limits accept this input: ...; or use policy='skip' ...
```

`--on-incompatible skip` computes the steps that apply and records why the others were left out, `--on-incompatible warn` runs everything, and `--preflight-only` prints the plan without running anything (`binding-metrics-batch` prints one per sample, writes no CSV and exits 1 when a sample is refused). In a batch a refused sample is an error row with `preflight_status` and `preflight_reason`, and the others go on. A prediction read with `--prediction-dir` was made elsewhere, so the limits of its model only warn. A run under `--prediction-msa-policy forbid-server` that would still call an MSA server is refused whatever `--on-incompatible` says. Pre-flight warns when the target has neither a template nor an MSA (mode `predict`, MSA server off, no receptor MSA); the warning never refuses the run.

With `--designed-with`, the EvoBind adversarial check is left out of a run whose model is one that the design pipeline ran (`--design-policy skip`): the block has `adversarial_skipped` and `independent_of_design: false`, and every other step runs. `--prediction-mode` is checked against the modes the model supports (OpenFold3 `predict`, `refold`, `score`; Boltz-2 also `score-lock` and `refold-lock`, with a warning) and the modes its runner runs (ColabFold and Protenix `predict` only, OpenFold3 `refold` and `score`, Boltz-2 all five); the mode is recorded in `results["prediction"]["mode"]` and `results["preflight"]["mode"]`. What is declared, the design-origin mapping and the keys are in [`docs/preflight.md`](docs/preflight.md) and [`docs/pose_sets.md`](docs/pose_sets.md).

#### Other prediction models and the run-once store

`--predictor MODEL` makes the `openfold` step model-agnostic. The prediction of AlphaFold2 (ColabFold), Boltz-2, Protenix or OpenFold3 is read through the adapter of that model, and every metric that needs it reads one record, so a model runs at most once. The results go to `results["prediction"]` (`prediction_*` CSV columns) and `results["openfold"]` is skipped.

```bash
# Read a Boltz-2 output you made yourself (the model never runs); the chains are named P and R in it
binding-metrics-run --input design.cif --output-dir results/ --metrics interface,openfold \
    --predictor boltz2 --prediction-dir boltz_out/ \
    --prediction-binder-chain P --prediction-target-chain R

# Run a model through the store: a second run of this input starts no model
binding-metrics-run --input design.cif --output-dir results/ --predictor of3
binding-metrics-run --input design.cif --output-dir results/ --predictor boltz2 --prediction-conda-env boltz
binding-metrics-run --input design.cif --output-dir results/ --predictor af2 --prediction-no-msa-server
binding-metrics-run --input design.cif --output-dir results/ --predictor boltz2 \
    --prediction-mode score-lock --prediction-lock-threshold 3.0
binding-metrics-run --input design.cif --output-dir results/ --predictor of3 --rerun-predictions  # runs again

# A batch: one model start for the samples the store lacks, one shared store
binding-metrics-batch --input-dir designs/ --output-csv metrics.csv --predictor of3
```

`--predictor af2` starts ColabFold (`colabfold_batch`), `boltz2` starts `boltz predict`, `protenix` starts `protenix pred` and `of3` starts OpenFold3, each from `PATH` or from the conda environment of `--prediction-conda-env` (OpenFold3: `--openfold-conda-env`, default `openfold3`). A setting that the runner of a model has no keyword for (`--prediction-cyclic` for ColabFold, `--prediction-lock-threshold` outside the lock modes of Boltz-2) is a usage error. ColabFold names the chains of its prediction `A` (receptor) and `B` (binder), and the pipeline maps them back to the IDs of the input. The store (`--prediction-cache`, default `<output-dir>/predictions`) keeps one directory per request, named by a hash of the content of the input file, the model version, the seeds and the options; a failed run is recorded and not retried until `--rerun-predictions`. `results["prediction"]["cache"]` holds the counters (`runs`, `hits`, `adopted`, ...) that show whether the model ran for the sample. The keys, the store layout and the batch behaviour are in [`docs/metrics.md`](docs/metrics.md#pipeline---predictor-the-prediction-store-and-resultsprediction).

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

Each structure gets its own directory, JSON report and log under `--output-dir` (by default the directory of the CSV). The file stem is the sample ID, so two inputs with the same stem, case ignored (`a/sample.pdb` and `b/sample.pdb` found by `--glob`, `sample.pdb` and `sample.cif`, or `Sample.pdb` and `sample.pdb`, which a case-insensitive file system writes to one directory) end the run with a usage error (exit code 2) before anything is created; `run_batch` raises `ValueError`. The CSV has one row per structure, in input order, with a `batch_status` column: `ok` (every step completed), `partial` (the pipeline finished but a step failed, or failed in part, as an energy mode that failed beside one that succeeded, whose other values stay in the row; see `batch_failed_steps` and `batch_failed_reasons`) or `error` (the worker raised; see `batch_error`). The last columns, `provenance_*`, record the package version, git sha, Python, OS, OpenMM version, platform and seed. The exit code is non-zero only when no sample is `ok`. Each worker process opens its own CUDA context and takes its share of GPU memory, so several workers on one GPU can run out of memory. Structures are matched to natives for DockQ by file stem (`--reference-dir`, `target1.cif` with `target1.pdb`).

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

`binding-metrics-relax` can operate on multi-model CIFs (e.g. outputs from homology modelling pipelines):

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

```bash
binding-metrics-check-env

# Inside Docker (requires NVIDIA Container Toolkit)
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

The `--summary` flag (available on both `binding-metrics-run` and `binding-metrics-report`) writes a human-readable summary alongside the JSON/CSV output. Use `--summary-format md` (default) for Markdown or `--summary-format html` for a self-contained HTML page (needs the `report` extra); result text (errors, reasons, names) is escaped, so it appears as written. It has a section for each step (marked skipped when the step did not run), per-residue buried SASA of the peptide in the interface section, and a RAG scorecard (🟢/🟡/🔴, ⬜ for a value that was not computed). When one mode of the interaction energy failed, a line `A mode failed: <error_message>` follows the energy table. `--nonfinite null` writes `null` for every NaN and infinity of the JSON; a report that was written that way is read back and written again unchanged. The scorecard thresholds are heuristic; see [`docs/report_thresholds.md`](docs/report_thresholds.md).

The individual scoring tools take the binder chain as `--binder-chain` and the target chain as `--target-chain`, each also under its older name (`--peptide-chain` or `--design-chain`, depending on the tool, and `--receptor-chain`). A tool accepts only the chains it uses: `binding-metrics-receptor-quality` takes the target and `binding-metrics-compare` the binder. Without them the smallest protein chain is the binder and the largest the target. See `--help` on each command for the options.

---

## Results and provenance

`binding-metrics-run` writes `<sample>_results.json` (`--format csv` writes the flattened CSV instead), and `run_pipeline` returns the same dict. Top-level keys:

| Key | Content |
|---|---|
| `sample_id`, `input`, `total_elapsed_s` | identifiers and wall time |
| `provenance` | package version, git sha (when the package runs from its own checkout), Python, OS, OpenMM version, platform, seed; the installed OpenFold3 version, the checkpoint name and whether the MSA server was used when OpenFold3 ran |
| `chains` | resolved chain IDs and residue counts |
| `prep` | what preparation changed: `removed_heterogens`, `n_removed_waters`, `kept_nonstandard`, `n_missing_atoms_rebuilt`, `n_missing_residue_gaps`, `chain_breaks` (consecutive residues whose C and N atoms are more than 2 Å apart), `missing_residues` (gaps and truncated chain ends, from SEQRES or `_entity_poly_seq` when the file has them), and `ncaa_bond_order_source` and `ncaa_charge_scheme` for GAFF2 residues |
| `relax` | energies, RMSD and RMSF, the OpenMM `platform`, `dropped_protein_chains` (protein chains other than the peptide and the receptor, which the relaxation removes) and the structural QC: `qc_passed`, `qc_failed_checks`, `qc_warnings` and (in the JSON) `qc_checks` |
| `energy`, `interface`, `geometry`, `electrostatics`, `dockq`, `openfold` | one dict per metric: `{"skipped": True}` when it did not run, `{"error": message}` when it failed |
| `prediction` | with `--predictor`: the confidence scores of the prediction, its EvoBind keys and `cache` (how the store served it) |
| `preflight` | the decision of the [pre-flight check](#pre-flight-check): `status`, `reason`, `policy`, the steps left out and the full report |
| `nonfinite_fields` | the JSON paths of every NaN or infinite value. With `--nonfinite null` the listed values are `null` in the file. A step that adds a block after the report was written (`binding-metrics-batch`: `openfold`, `prediction`) updates the list: a block that it replaces loses its entries and gets the ones of its new values |

A value that could not be computed keeps its NaN, 0 or None, and its dict gains a string under `reason` that says why; a dict without `reason` was computed in full. The seven QC checks of the relaxed structure (finite energy that did not rise, heavy-atom RMSD to the input, finite coordinates, no fused atoms, no stretched or broken bonds, no inverted Cα stereocentre, unchanged heavy-atom composition) are advisory: a failed check logs a warning and changes neither the results nor the exit code. `binding-metrics-run` exits with 1 when a step failed, after writing the partial results. A requested EvoBind score or adversarial check that could not be computed counts as a failed step. A step that failed in part counts as failed: an energy with a mode that failed beside one that succeeded (`success` true, `error_message` naming the mode, for example `relaxed: minimization failed: ...`), or the energy term of `receptor_quality`; the values that were computed stay in the results. The schemas are in [`docs/metrics.md`](docs/metrics.md#15-pipeline-results-and-provenance).

---

## Logging

Library code logs through `logging` and never configures it on import. The command-line tools call `binding_metrics.utils.configure_logging()`, which sends records up to WARNING to stdout and ERROR and above to stderr, as the bare message (`binding-metrics-prep` and `-solvate` send warnings to stderr as well, so that their JSON summary is alone on stdout). `--log-file PATH` redirects both streams of a command to a file (always UTF-8); `binding-metrics-batch` writes one log per sample, `<output-dir>/<sample>/<sample>.log`, unless `--log-file` names one shared file. A console that cannot encode a character the tools print (`LC_ALL=C`) shows a `?` in its place and raises no `UnicodeEncodeError`.

In a script, call `configure_logging()` for the same behaviour, or attach handlers to the `binding_metrics` logger yourself:

```python
import logging
from binding_metrics.utils import configure_logging

configure_logging(logging.INFO)
```

---

## Reproducibility

- **Seeds.** Hydrogen placement, PDBFixer's rebuilding of missing atoms, the conformer behind the AM1-BCC charges of non-canonical residues, the MD initial velocities and Langevin noise, the ion placement of `binding-metrics-solvate` and the hydrogen placement in the receptor energy term of `binding-metrics-receptor-quality` are seeded. The default seed is 1. Set it with `--random-seed INT` on `binding-metrics-run`, `-batch`, `-relax`, `-energy`, `-prep`, `-solvate` and `-receptor-quality`, or with `random_seed=` in the API; `--random-seed none` draws fresh randomness, for instance to generate independent MD replicas. The static metrics have no random step.
- **OpenFold3.** The seed above does not drive it. OpenFold3 samples with seed 42 (written to its runner YAML) unless `--openfold-seeds` is given, and the MSA server can return different alignments over time. With `--predictor` a finished prediction is stored under a key of the content of the input, the model version, the seeds and the options, and a repeated run reuses it; `--rerun-predictions` forces a new one. The alignments a remote MSA server returns are not part of the key.
- **GPU precision.** CUDA runs in mixed precision and its force reduction order is not deterministic, so energies and MD from one seed can differ in the last digits between GPU runs.
- **Provenance.** Every results file carries the `provenance` block (package version, git sha, Python, OS, OpenMM version, platform, seed), and batch CSV rows carry it as `provenance_*` columns.
- **Environments.** `environment.yml` is the specification that CI and the Dockerfile build from. `environment.lock.yml` is a snapshot of the exact versions of the development environment (`conda env create -n binding-metrics -f environment.lock.yml`, then `pip install --no-deps -e .`); neither CI nor the Dockerfile reads it, and its header says how to regenerate it.
- **Releases.** Pushing a `v*` tag builds the sdist and the wheel and attaches them to a GitHub Release, and builds the Docker images, once the tag is `v` plus the `pyproject.toml` version and `ci.yml` has passed on the tagged commit (the push to `main`); a tag pushed before that run ends fails and is retried with "Re-run failed jobs". The wheel is installed in a clean virtualenv on Python 3.11 and 3.12 and every console script is run with `--help` before the release is created. Nothing is published to PyPI.

---

## Documentation

- [`docs/metrics.md`](docs/metrics.md): every metric with its signature, result keys, units and algorithm notes; the pipeline results; the metric registry
- [`docs/preflight.md`](docs/preflight.md): the inputs that a step or a prediction model cannot take, how they are declared, and the pre-flight check
- [`docs/poses.md`](docs/poses.md): the pose kinds (`PredictedPose`, `DockedPose`, `ExperimentalPose`), what every pose has (`id`, `atoms()`, `content_hash()`, `native_scores()`), and the deprecated name `PredictionRecord`
- [`docs/pose_sets.md`](docs/pose_sets.md): scoring the poses of one construct together (`score_pose_set`): the common frame, agreement between poses, aggregation, representative poses, reference-site descriptors, reconciliation of two back-ends, the result schema and the writers
- [`docs/nonstandard.md`](docs/nonstandard.md): D-amino acids, N-methylated and phosphorylated residues, the GAFF2 route, cyclic closures and their limits
- [`docs/report_thresholds.md`](docs/report_thresholds.md): the scorecard thresholds
- [`docs/testing.md`](docs/testing.md): the test tiers (`pytest`, `--full`, `--gpu`), the area markers of the tests (`pytest -m poses_frame`), the helper that runs the tests for a diff, and `pytest --full` as the gate before a pull request
- [`CHANGELOG.md`](CHANGELOG.md): what changed
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

Contributions, bug reports, and feature requests are welcome. Please open an issue to discuss significant changes before submitting a pull request. All pull requests should include tests and pass the existing test suite (`pytest --full`; a plain `pytest` runs the fast tier, see [`docs/testing.md`](docs/testing.md)).

**Warnings as errors.** `pyproject.toml` turns a `DeprecationWarning` attributed to a `binding_metrics` module into an error; a test that provokes the warning on purpose says so with `pytest.warns`. To make every `DeprecationWarning` an error, run `python -W error::DeprecationWarning -m pytest --full`. OpenMM 8.2 crashes the interpreter (exit code 139) when its own import-time warning (`builtin type SwigPyPacked has no __module__ attribute`) is an error, as in `python -W error::DeprecationWarning -c "import openmm"`; the test configuration ignores that warning by message, so the flag is safe for the suite and not for a bare `import openmm` of your own.

**Running part of the suite.** Every test carries markers for the area of the code that its file is about, so `pytest -m poses_frame` or `pytest -m "poses or preflight"` runs a part of it, and `python scripts/pytest_for_diff.py` runs the tests that the changed files can affect ([`docs/testing.md`](docs/testing.md) lists the areas). A plain `pytest` runs the fast tier, without the slow, integration, OpenFold3 and GPU tests; `pytest --full` runs every test but the GPU ones and is the gate before a pull request and before a signed release.

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
- The OpenFold3 Team (2026). OpenFold3, v0.5.0 (OpenBind-0 weights). https://github.com/aqlaboratory/openfold-3, doi:10.5281/zenodo.22042719 (doi:10.5281/zenodo.17485509 stands for all versions; cite the release that produced the results).
- OpenBind Consortium (2026). OpenBind-0 announcement, 21 August 2026. https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/
- Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500.
