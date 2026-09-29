# Non-Standard Residue Support

This document describes how BindingMetrics handles residues outside the 20 canonical L-amino acids: the force-field strategy for each kind, which residues are covered, and the known limitations.

---

## Overview

The relaxation step (`ImplicitRelaxation._setup_system`) reads the prepped structure and handles non-standard residues in four steps, all **before `addHydrogens`**:

1. **`detect_nonstandard(topology, chain_id)`** scans the peptide chain for D-amino-acid and N-methylated residue names. Detection is name-based; no coordinates are analysed.
2. **`patch_nonstandard(topology, positions, chain_id, info)`** renames those residues to the names a force-field template exists for and removes spurious atoms, so `addHydrogens` sees the correct topology.
3. **`patch_cyclic_topology`** adds the closure bonds of a cyclic peptide (`core/cyclic.py`).
4. With `--small-molecules auto` (the default of `binding-metrics-relax` and `binding-metrics-run`), **`parameterize_ncaa_residues`** builds a GAFF2 template for every remaining non-canonical residue ([GAFF2 route](#gaff2-route-for-other-non-canonical-residues)).

For cyclic peptides, `prep_structure` applies the same renaming, cyclic patching and GAFF2 templates while it adds hydrogens.

The force field needs the L and template names, but the original names are kept in `NonstandardInfo`, and `restore_nonstandard_names(topology, info)` writes them back before a structure is saved. The prepped and the relaxed CIF files therefore carry the input names: the peptide of 1CWA reads `DAL ... SAR ...` in the input, the prepped file and the relaxed file, not `ALA ... NMG ...`.

---

## Bonds Around a Non-Standard Residue

OpenMM builds the bonds of a residue from a table of standard residue names. A residue it does not know, phosphoserine or S-palmitoyl-cysteine (P1L in 6SBA) for instance, loads as a group of atoms without a single bond, and the peptide bond from the residue before it is missing too. The force field then rejects the standard neighbour ("No template found for residue ... the bonds are different"), not the residue that lacks the bonds. Three steps restore them:

1. **`load_structure`** (CIF input). The link rows of `_struct_conn` (`covale`, `disulf`, `modres`) are read through the author numbering, because OpenMM looks the partners up by label numbering while it keys the atoms by author numbering, and drops a row when the two differ. This needs gemmi. A residue that the file leaves without any bond is bonded by covalent radii, as the CONECT records of a PDB file would do. `save_cif` writes the restored residue number into `label_seq_id` and the `_struct_conn` rows, so a prepped file keeps its links on reload. 1QJB: HIS6 C to SEP7 N and the nine bonds inside SEP are present after loading the mmCIF, as after loading the PDB file.
2. **`patch_cyclic_topology`** calls `reconstruct_nonstandard_residue_bonds` for every protein chain, the receptor included. It restores the bonds inside each residue that has none and the C(i) to N(i+1) peptide bond next to it when the two atoms are within 0.20 nm; a chain break is not bridged. A PDB file without CONECT records for the residue relies on this step.
3. **`strip_heterogens`** keeps SEP, TPO and PTR in every chain, so an unselected chain is not cut at its phosphoserine.

A residue of the receptor that the force fields do not cover goes through the [GAFF2 route](#gaff2-route-for-other-non-canonical-residues) like one of the peptide. The charge calculation for P1L of 6SBA is repeated each time a system is built and takes minutes.

`CYM`, the AMBER deprotonated cysteine, is a standard residue, as `HID`, `HIE`, `HIP`, `HIN`, `CYX`, `ASH`, `GLH` and `LYN` are; amber14 has its template, so no GAFF2 parameters are built for it.

---

## Other Protein Chains

The relaxation and the interaction energy describe the peptide-receptor pair. A third protein chain, such as the second copy of the complex in the asymmetric unit (5WGD holds two: with peptide E and receptor A, chains B and F are the second copy), would enter the energy of the complex but not that of the isolated peptide and receptor, and its caps and patches are not handled. `drop_other_protein_chains` removes every protein chain other than the peptide and the receptor, after `strip_heterogens`, and a warning names each removed chain. `RelaxationResult.dropped_protein_chains` (`results["relax"]["dropped_protein_chains"]` in the pipeline) lists them. Nothing is removed unless both chains are named and present. There is no option to keep the chains: a receptor made of several chains, a Fab for example, has to be reduced to the chain that carries the interface before it goes in.

---

## D-Amino Acids

**Strategy:** rename to L counterpart in the topology; preserve coordinates.

AMBER ff14SB has no chirality-sensitive energy terms — bond lengths, angles, and torsion parameters are identical for D and L forms. The chirality is encoded entirely in the 3D coordinates, which are untouched.

**Force field:** standard ff14SB templates used after rename. No custom XML needed.

**Ramachandran:** φ/ψ angles are negated before region classification, so a D-α-helix (φ ≈ +57°, ψ ≈ +47°) maps correctly to the L-α-helix basin (φ ≈ −57°, ψ ≈ −47°). The `per_residue` output includes an `is_d_aa` flag, and `n_d_residues` counts the D residues evaluated.

**Charges in the metrics:** the salt-bridge, Coulomb and solvation tables are keyed by L-residue names, and D codes are mapped to their L counterpart before the lookup (`DLY` counts as LYS).

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
| MED | D-methionine | MET | ✅ handled |
| DPN | D-phenylalanine | PHE | ✅ handled |
| DPR | D-proline | PRO | ✅ handled |
| DSN | D-serine | SER | ✅ handled |
| DTH | D-threonine | THR | ✅ handled |
| DTR | D-tryptophan | TRP | ✅ handled |
| DTY | D-tyrosine | TYR | ✅ handled |
| DVA | D-valine | VAL | ✅ handled |
| DAR | D-arginine | ARG | ✅ handled |

The table lists the 19 entries of `D_AA_MAP` in `core/nonstandard.py`: the D form of every standard amino acid except glycine, which is achiral. The CCD code of D-methionine is `MED`; `DME` is decamethonium and is not treated as a D residue.

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

### Without a curated template

The following N-methylated residues appear in natural products and designed macrocycles but have no curated template. With `--small-molecules auto` they go through the [GAFF2 route](#gaff2-route-for-other-non-canonical-residues), whose charge model has the limits described there; the bundled examples exercise that route for BMT, ABA, IAM, 0EH and MK8 only. A curated template needs: (1) a custom XML with RESP charges, (2) the input code → template name mapping in `NME_AA_MAP`, and (3) a test.

| PDB code | Full name | Notes |
|----------|-----------|-------|
| NMI | N-methyl-isoleucine | Two stereocentres; common in cyclosporin analogues |
| NMC | N-methyl-cysteine | May participate in disulfide — interact with cyclic patching |
| NMS | N-methyl-serine | Hydroxyl sidechain |
| NMT | N-methyl-threonine | Not BMT: the MeBmt of cyclosporin A has the code BMT, which the GAFF2 route parameterises |
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

## Recognised Closure Types

`detect_cyclization` reports these types for the peptide chain (`CyclicBondInfo.cyclic_type`):

| Type | Bond |
|------|------|
| `head_to_tail` | backbone C(last) to N(first) amide |
| `disulfide` | CYS SG to SG |
| `lactam_n_asp`, `lactam_n_glu` | ASP CG or GLU CD to the N-terminal amide N |
| `lactam_c_lys` | LYS NZ to the C-terminal carbonyl C |
| `lactam_sc_lys_asp`, `lactam_sc_lys_glu` | LYS NZ to ASP CG or GLU CD, both residues internal (a side-chain staple) |
| `hydrocarbon_staple` | all-carbon side-chain cross-link between two residues, for example MK8 and 0EH (3V3B) |

A peptide can have several closures (SFTI-1 in 3P8F has `head_to_tail` and `disulfide`). Detection reads the bonds recorded in the file (`_struct_conn` records of a CIF, CONECT records of a PDB) and, for the amide, disulfide and lactam types, also atom distances. The staple residues are parameterised by the GAFF2 route, and their cross-link is covered by the GAFF2 terms, so it needs no template of its own.

---

## Unsupported Cyclization / Crosslink Types

The following crosslink types raise `CyclizationError` with guidance when detected by `detect_cyclization`. They require a `custom_bond_handler` with GAFF2/SMIRNOFF parameters.

| Type | Example | Notes |
|------|---------|-------|
| Thioether bridge | Lanthipeptide S–C | Use GAFF2 |
| Macrolactone | Ser/Thr/Tyr O → carbonyl C | Use GAFF2 |
| Biaryl ether | Vancomycin-type | Use GAFF2 or SMIRNOFF |

---

## Phosphorylated Residues

**Strategy:** the AMBER phosaa parameters, not GAFF2.

Phosphoserine (SEP), phosphothreonine (TPO) and phosphotyrosine (PTR), together with the phosaa protonation variants (S1P, T1P, Y1P, H1D, H2D, H1E, H2E), use the RESP-fitted `phosaa14SB.xml` that ships with openmmforcefields (Raguette et al., *J. Chem. Theory Comput.* 2024, 20, 7199). `core/phosaa.py` adapts its atom-type names to `amber14-all.xml` and changes no charge or bonded term. A phosphate monoester carries a net charge of −2. The GAFF2 route skips these residues on purpose: it builds residues in their neutral form and would protonate the phosphate to net 0.

`core/phosaa.py` also registers hydrogen definitions, so `addHydrogens` places the hydrogens of a heavy-atom-only phospho residue. The relaxation, the interaction energy and prep use these parameters. In the metrics, the non-bridging phosphate oxygens O1P, O2P and O3P count as negative atoms in the salt-bridge and Coulomb tables (−2/3 e each) and as O(−) in the solvation parameters. The bundled example is the phosphopeptide of 1QJB (chain Q, SEP).

---

## GAFF2 Route for Other Non-Canonical Residues

**Strategy:** for every residue that ff14SB and the curated templates do not cover, `core/gaff_ncaa.py` builds an OpenMM residue template with `<ExternalBond>` tags from the residue's own geometry. It runs with `--small-molecules auto` and covers, for example, MeBmt (BMT) and 2-aminobutyrate (ABA) in cyclosporin A, IAM in 1XY4, and the staple residues 0EH and MK8 in 3V3B.

Openmmforcefields' GAFF template generator cannot match a backbone residue, because the backbone N and C carry peptide bonds and the generator only matches residues without external bonds. The route works around this for each residue:

1. It builds an RDKit molecule of the heavy atoms and adds a carbon cap at each external bond (backbone N and C, closure atoms).
2. **Bond orders and formal charges come from the wwPDB Chemical Component Dictionary (CCD)** that ships with biotite, so no network access is needed. The entry is used only if it describes the same molecule: every atom name must exist with the same element and the bonds between named atoms must be the same set. The molecule has no hydrogens, so bond orders cannot be recovered from its connectivity; valence-based perception would leave every bond single and turn an alkene into an alkane. Ionisable groups are neutralised.
3. RDKit adds hydrogens, and openmmforcefields' `GAFFTemplateGenerator` supplies GAFF2 atom types, bonded parameters and AM1-BCC charges. This needs AmberTools (`antechamber` and `sqm`) on `PATH`.
4. The template drops the caps, adds an `<ExternalBond>` at every capped atom, retypes the backbone atoms to the ff14SB types, and adds explicit GAFF terms for the backbone–side-chain junctions. The charge of the removed cap atoms is spread evenly over the remaining atoms, so the template has an integer net charge, normally 0.
5. The hydrogens of the residue are injected into the topology, which is rebuilt so that the residue matches its template.

**Fallback to single bonds.** A residue that is not in the CCD, whose atoms disagree with its entry, or whose CCD bond orders do not sanitise in RDKit is built with every bond single. The double bonds, aromatic rings and hydrogen count of such a residue are unreliable, and a warning says so. The pipeline records the outcome for each residue under `ncaa_bond_order_source` in `results["prep"]` (cyclic peptides) and `results["relax"]`: `"ccd"` or `"single_bonds"`. For the somatostatin analogue 1XY4, prep reports `{"IAM": "ccd"}`. Use the CCD residue code and atom names to get the chemistry right.

Effect of the CCD bond orders on the cyclosporin A residues of 1CWA: MeBmt has 17 hydrogens instead of 19, and its CE=CZ bond relaxes to 1.340 Å instead of 1.544 Å (crystal 1.336 Å).

### Charge-model limits

- **Neutral form.** Bond-order perception runs at total charge 0, so carboxylic acid, phosphate, sulfate, primary amine and guanidine side chains come out uncharged. A warning names the residue, the group and the expected net charge; electrostatics and energies involving such a residue are unreliable unless its charge is set by hand.
- **Backbone charges.** The template keeps the AM1-BCC charges of the capped molecule for every atom, backbone atoms included, and only the atom types of the backbone are set to ff14SB. The backbone N and H charges are not amide-like. In 1CWA the template of ABA has N −0.787 and HN +0.389, and BMT (N-methylated, no amide H) has N −0.716, against N −0.416 and H +0.272 for ALA in ff14SB. The carbon caps that stand in for the neighbouring residues during the charge calculation are the likely cause; capping with ACE and NME groups is not implemented. Treat energies that involve such residues as approximate.
- **General force field.** GAFF2 is a general small-molecule force field, a pragmatic approximation for exotic building blocks and not a substitute for purpose-built RESP-fitted parameters.

---

## Charge-Model Caveats

| Residues | Charge source | Caveat |
|----------|---------------|--------|
| D-amino acids | ff14SB, as the L counterpart | none beyond the missing-atom limitation above |
| N-methylated (NMG, NMA, MVA, MLE) | ForceField_NCAA RESP (ff03 protocol) | condensed-phase RESP used with ff14SB (see above) |
| Lactam residues (ASPL, GLUL, LYSL) | approximate, from the ff14SB ASN/GLN analogues | recompute RESP charges for production or free-energy work and pass them through `custom_bond_handler` |
| Phosphorylated (SEP, TPO, PTR, ...) | phosaa14SB (RESP) | none known here |
| Everything else | AM1-BCC on a capped, neutral molecule | neutral form and non-amide backbone charges (above) |

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
With `--small-molecules auto`, the [GAFF2 route](#gaff2-route-for-other-non-canonical-residues) parameterises it from its geometry; give it the CCD residue code and atom names so the bond orders are right. Use `custom_bond_handler` in `RelaxationConfig` when you need parameters of your own (RESP charges, for instance) or a crosslink type listed as unsupported above. See the docstring in `relaxation.py` and the `CyclizationError` message for a GAFF2 example.
