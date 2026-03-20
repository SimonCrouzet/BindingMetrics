# Non-Standard Residue Support

This document describes how BindingMetrics handles residues outside the 20 canonical L-amino acids, including the force field strategy, current status, and known limitations.

---

## Overview

Non-standard residues are processed in `_setup_system` (relaxation protocol) through two sequential steps — both running **after PDBFixer and before `addHydrogens`**:

1. **`detect_nonstandard(topology, chain_id)`** — scans residue names against the registries.
2. **`patch_nonstandard(topology, positions, chain_id, info)`** — renames residues and removes spurious atoms so `addHydrogens` sees the correct topology.

Detection is purely name-based (no coordinate analysis). The original residue names are preserved in `NonstandardInfo` for metric reporting; only the OpenMM topology is renamed.

---

## D-Amino Acids

**Strategy:** rename to L counterpart in the topology; preserve coordinates.

AMBER ff14SB has no chirality-sensitive energy terms — bond lengths, angles, and torsion parameters are identical for D and L forms. The chirality is encoded entirely in the 3D coordinates, which are untouched.

**Force field:** standard ff14SB templates used after rename. No custom XML needed.

**Ramachandran:** φ/ψ angles are negated before region classification, so a D-α-helix (φ ≈ +57°, ψ ≈ +47°) maps correctly to the L-α-helix basin (φ ≈ −57°, ψ ≈ −47°). The `per_residue` output includes an `is_d_aa` flag.

**Limitation:** PDBFixer's `addMissingAtoms` rebuilds missing atoms using L-amino acid geometry. If a D-residue has missing heavy atoms in the input structure, PDBFixer will introduce L-configuration atoms. **The input structure must be complete** (all heavy atoms present) for D-amino acids to be handled correctly.

| PDB code | Full name | L counterpart | Status |
|----------|-----------|---------------|--------|
| DAL | D-alanine | ALA | ✅ handled |
| DAS | D-aspartic acid | ASP | ✅ handled |
| DSG | D-asparagine | ASN | ✅ handled |
| DCY | D-cysteine | CYS | ✅ handled |
| DGL | D-glutamic acid | GLU | ✅ handled |
| DGN | D-glutamine | GLN | ✅ handled |
| DHI | D-histidine | HIS | ✅ handled |
| DIL | D-isoleucine | ILE | ✅ handled |
| DLE | D-leucine | LEU | ✅ handled |
| DLY | D-lysine | LYS | ✅ handled |
| DME | D-methionine | MET | ✅ handled |
| DPN | D-phenylalanine | PHE | ✅ handled |
| DPR | D-proline | PRO | ✅ handled |
| DSN | D-serine | SER | ✅ handled |
| DTH | D-threonine | THR | ✅ handled |
| DTR | D-tryptophan | TRP | ✅ handled |
| DTY | D-tyrosine | TYR | ✅ handled |
| DVA | D-valine | VAL | ✅ handled |
| DAR | D-arginine | ARG | ✅ handled |

> Glycine is achiral — no D form exists.

---

## N-Methylated Amino Acids

**Strategy:** rename to canonical template name, load custom AMBER XML, remove any spurious backbone H added by PDBFixer, then let `addHydrogens` add the N-methyl H atoms from the template.

**Force field:** custom AMBER-format XML residue templates. Partial charges are RESP-fitted values from **ForceField_NCAA** (Khoury et al., *ACS Synth. Biol.* 2014, [PMC4277759](https://pmc.ncbi.nlm.nih.gov/articles/PMC4277759/)), derived at the HF/6-31G* level using the ff03 RESP protocol. Atom types follow ff14SB conventions: `CX` for Cα (preserving ff14SB backbone torsion parameters), `CT` for sp3 carbons, `H1` for H on C adjacent to N, `HC` for aliphatic methyl H.

**Charge caveat:** ForceField_NCAA charges use the ff03 condensed-phase dielectric protocol rather than ff14SB's gas-phase RESP. This is a minor inconsistency and matches what published cyclosporin A MD studies use (e.g. [JACS 2022](https://pubs.acs.org/doi/10.1021/jacs.2c01743)). For production free-energy calculations, recompute RESP charges using antechamber/R.E.D. Server with the ACE-NMeAA-NME dipeptide at HF/6-31G* in vacuo.

**Backbone N atom type:** `N` (same as standard amide N and proline N in ff14SB). No new atom types are introduced.

**Limitation:** The N-methyl heavy atoms (CN and its three H) must already be present in the input structure. `patch_nonstandard` only removes spurious atoms — it does not add missing heavy atoms.

### Supported templates

| Input code(s) | Template name | Based on | Charge source | Status |
|---------------|---------------|----------|---------------|--------|
| SAR, NMG | NMG | Glycine | ForceField_NCAA RESP | ✅ handled |
| NMA, MAA | NMA | Alanine | ForceField_NCAA RESP | ✅ handled |
| MVA | MVA | Valine | ForceField_NCAA RESP | ✅ handled |
| MLE | MLE | Leucine | ForceField_NCAA RESP | ✅ handled |

### Not yet supported

The following N-methylated residues appear in natural products and designed macrocycles but do not yet have templates. Adding one requires: (1) a custom XML with RESP charges, (2) adding the input code → template name mapping to `NME_AA_MAP`, and (3) a test.

| PDB code | Full name | Notes |
|----------|-----------|-------|
| NMI | N-methyl-isoleucine | Two stereocentres; common in cyclosporin analogues |
| NMC | N-methyl-cysteine | May participate in disulfide — interact with cyclic patching |
| NMS | N-methyl-serine | Hydroxyl sidechain |
| NMT | N-methyl-threonine (BMT) | bMeBmt in cyclosporin A; unique C-4 substituent |
| NMK | N-methyl-lysine (backbone) | Distinct from side-chain trimethyl-lysine |
| NMR | N-methyl-arginine | Guanidinium sidechain |
| NMQ | N-methyl-glutamine | |
| NMH | N-methyl-histidine | Protonation state handling needed |
| NMW | N-methyl-tryptophan | Bulky indole; charges likely needed from quantum calc |
| NMY | N-methyl-tyrosine | |
| MME | N-methyl-methionine | |

ForceField_NCAA RESP charges for all of the above are available in the ffncaa.zip supplementary data ([Wayback Machine](https://web.archive.org/web/20160322042443/http://selene.princeton.edu/FFNCAA/files/ffncaa.zip)).

---

## Cyclic Peptide-Specific Non-Standard Residues

These are created dynamically by `patch_cyclic_topology` (see `core/cyclic.py`) and are not input residue names — they appear in the OpenMM topology only after patching.

| Internal name | Origin | Description | Status |
|---------------|--------|-------------|--------|
| CYX | CYS | Disulfide-bonded cysteine | ✅ handled (ff14SB built-in) |
| ASPL | ASP | ASP sidechain CG acting as lactam carbonyl | ✅ handled (custom XML) |
| GLUL | GLU | GLU sidechain CD acting as lactam carbonyl | ✅ handled (custom XML) |
| LYSL | LYS | LYS sidechain NZ acting as lactam amide N | ✅ handled (custom XML) |

---

## Unsupported Cyclization / Crosslink Types

The following crosslink types raise `CyclizationError` with guidance when detected by `detect_cyclization`. They require a `custom_bond_handler` with GAFF2/SMIRNOFF parameters.

| Type | Example | Notes |
|------|---------|-------|
| Hydrocarbon staple | Aileron-type all-carbon bridge | Use GAFF2 via `openmmforcefields` |
| Thioether bridge | Lanthipeptide S–C | Use GAFF2 |
| Macrolactone | Ser/Thr/Tyr O → carbonyl C | Use GAFF2 |
| Biaryl ether | Vancomycin-type | Use GAFF2 or SMIRNOFF |

---

## Adding a New Non-Standard Residue

### New D-amino acid code
Add one entry to `D_AA_MAP` in `core/nonstandard.py`:
```python
"DXX": "STD",   # D-name → L counterpart
```
No XML template needed.

### New N-methylated amino acid
1. Add an entry to `NME_AA_MAP`:
   ```python
   "NMX": "NMX",   # input code → template name
   ```
2. Write an XML template `_XML_NMX` with ForceField_NCAA RESP charges (or compute with antechamber). Ensure Σq = 0, `ExternalBond` on `N` and `C`, no `H` on `N`.
3. Register it in `_NME_XMLS`:
   ```python
   "NMX": _XML_NMX,
   ```
4. Add tests to `tests/test_nonstandard.py`.

### Entirely new residue type (e.g. Cα-methyl, β-amino acid)
Use `custom_bond_handler` in `RelaxationConfig`. See the docstring in `relaxation.py` and the `CyclizationError` message for a GAFF2 example.
