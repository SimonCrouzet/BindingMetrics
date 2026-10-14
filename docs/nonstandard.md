# Non-Standard Residue Support

How BindingMetrics handles residues outside the 20 canonical L-amino acids: the force-field route for each kind, the residues covered and the known limits.

---

## Overview

The relaxation step (`ImplicitRelaxation._setup_system`) reads the prepped structure and handles non-standard residues in four steps, all before `addHydrogens`:

1. **`detect_nonstandard(topology, chain_id)`** scans the peptide chain for D-amino-acid and N-methylated residue names. Detection is name-based; no coordinates are analysed. `chain_id` is the author ID for a topology read by `load_structure` from an mmCIF, or the ID of the chain in the topology: the peptide of 1CWA is author chain C and chain B of the topology, whose chain C holds waters, and `"C"` and `"B"` find the same chain. `NonstandardInfo.chain_id` is the ID in the topology. `patch_nonstandard` and the `chain_id` argument of `restore_nonstandard_names` read the ID the same way. The relaxation, the energy and prep call internal versions that take the ID in the topology.
2. **`patch_nonstandard(topology, positions, chain_id, info)`** renames those residues to the names a force-field template exists for and removes spurious atoms, so `addHydrogens` sees the correct topology.
3. **`patch_cyclic_topology`** adds the closure bonds of a cyclic peptide (`core/cyclic.py`).
4. With `--small-molecules auto` (the default of `binding-metrics-relax` and `binding-metrics-run`), **`parameterize_ncaa_residues`** builds a GAFF2 template for every remaining non-canonical residue ([GAFF2 route](#gaff2-route-for-other-non-canonical-residues)).

For cyclic peptides, `prep_structure` applies the same renaming, cyclic patching and GAFF2 templates while it adds hydrogens.

The force field needs the L and template names, and the original names are kept in `NonstandardInfo`. `restore_nonstandard_names(topology, info)` writes them back before a structure is saved, so the prepped and the relaxed CIF files carry the input names: the peptide of 1CWA reads `DAL ... SAR ...` in the input, the prepped file and the relaxed file.

---

## Bonds Around a Non-Standard Residue

OpenMM builds the bonds of a residue from a table of standard residue names. A residue it does not know, phosphoserine or S-palmitoyl-cysteine (P1L in 6SBA) for instance, loads as a group of atoms without a bond, and the peptide bond from the residue before it is missing too. The force field then rejects the standard neighbour ("No template found for residue ... the bonds are different"), not the residue that lacks the bonds. Three steps restore the bonds:

1. **`load_structure`** (CIF input). The link rows of `_struct_conn` (`covale`, `disulf`, `modres`) are read through the author numbering, because OpenMM looks the partners up by label numbering while it keys the atoms by author numbering, and drops a row when the two differ. This needs gemmi. A residue that the file leaves without any bond is bonded by covalent radii, as the CONECT records of a PDB file would do. `save_cif` writes the restored residue number into `label_seq_id` and the `_struct_conn` rows, so a prepped file keeps its links on reload; where residues share a number (60 and 60A) the rows also carry the author partner columns and the insertion codes, and `label_seq_id` is `.` for those residues.
2. **`patch_cyclic_topology`** calls `reconstruct_nonstandard_residue_bonds` for every protein chain, the receptor included. It restores the bonds inside each residue that has none and the C(i) to N(i+1) peptide bond next to it when the two atoms are within 0.20 nm; a chain break is not bridged. A PDB file without CONECT records for the residue relies on this step.
3. **`strip_heterogens`** keeps what `detect_chains` calls an amino acid in every chain (SEP, TPO and PTR, the D-amino acids and, with biotite installed, the peptide-linking components of the Chemical Component Dictionary), so an unselected chain is not cut at its phosphoserine or its D residues, and a third chain of D residues is reported as a dropped protein chain.

A residue of the receptor that the force fields do not cover goes through the [GAFF2 route](#gaff2-route-for-other-non-canonical-residues) like one of the peptide. The charge calculation is repeated each time a system is built (minutes for P1L of 6SBA).

`CYM`, the AMBER deprotonated cysteine, is a standard residue, as `HID`, `HIE`, `HIP`, `HIN`, `CYX`, `ASH`, `GLH` and `LYN` are; amber14 has its template, so no GAFF2 parameters are built for it.

---

## Other Protein Chains

The relaxation and the interaction energy describe the peptide-receptor pair. A third protein chain, such as the second copy of the complex in the asymmetric unit (5WGD holds two: with peptide E and receptor A, chains B and F are the second copy), would enter the energy of the complex but not that of the isolated peptide and receptor, and its caps and patches are not handled. `drop_other_protein_chains` removes every protein chain other than the peptide and the receptor, after `strip_heterogens`, and a warning names each removed chain by its author ID. `RelaxationResult.dropped_protein_chains` (`results["relax"]["dropped_protein_chains"]` in the pipeline) lists them. Nothing is removed unless both chains are named and present. There is no option to keep the chains: a receptor made of several chains, a Fab for example, has to be reduced to the chain that carries the interface before it goes in.

---

## D-Amino Acids

**Strategy:** rename to the L counterpart in the topology; keep the coordinates. ff14SB parameters are the same for D and L, and the chirality is in the coordinates. The standard ff14SB templates are used after the rename; no custom XML is needed.

**Hydrogens in prep:** `Modeller.addHydrogens` has no definition for a D residue name. Both routes of `prep_structure` rename the D residues to their L templates while the hydrogens are added and give them their D names back. The hydrogens come from the heavy-atom geometry, which is the D geometry whatever the residue is called, so the HA of a D residue lies on the D side of the alpha carbon. A peptide made only of D residues is a protein chain for prep and for the heterogen strip, like a mixed one.

**Ramachandran:** φ/ψ angles are negated before region classification, so a D-α-helix (φ ≈ +57°, ψ ≈ +47°) maps to the L-α-helix basin. The `per_residue` output includes an `is_d_aa` flag, and `n_d_residues` counts the D residues evaluated.

**Charges in the metrics:** the salt-bridge, Coulomb and solvation tables are keyed by L-residue names, and D codes are mapped to their L counterpart before the lookup (`DLY` counts as LYS).

**Limit:** PDBFixer's `addMissingAtoms` rebuilds missing atoms with L-amino acid geometry, so a D residue with missing heavy atoms gets L-configuration atoms. The input structure must have all heavy atoms of its D residues.

| PDB code | L counterpart |
|----------|---------------|
| DAL | ALA |
| DAS | ASP |
| DSG | ASN |
| DCY | CYS |
| DGL | GLU |
| DGN | GLN |
| DHI | HIS |
| DIL | ILE |
| DLE | LEU |
| DLY | LYS |
| MED | MET |
| DPN | PHE |
| DPR | PRO |
| DSN | SER |
| DTH | THR |
| DTR | TRP |
| DTY | TYR |
| DVA | VAL |
| DAR | ARG |

The table lists the 19 entries of `D_AA_MAP` in `core/nonstandard.py`: the D form of every standard amino acid except glycine, which is achiral. The CCD code of D-methionine is `MED`; `DME` is decamethonium and is not treated as a D residue.

---

## N-Methylated Amino Acids

**Strategy:** rename to the canonical template name, load the custom AMBER XML, remove any spurious backbone H added by PDBFixer, then let `addHydrogens` add the N-methyl H atoms from the template.

**Force field:** custom AMBER-format XML residue templates. The partial charges are RESP-fitted values from ForceField_NCAA (Khoury et al., *ACS Synth. Biol.* 2014, [PMC4277759](https://pmc.ncbi.nlm.nih.gov/articles/PMC4277759/)), derived at the HF/6-31G* level with the ff03 RESP protocol. Atom types follow ff14SB: `CX` for Cα (keeping the ff14SB backbone torsion parameters), `CT` for sp3 carbons, `H1` for H on C adjacent to N, `HC` for aliphatic methyl H. The backbone N has the type `N`, as the standard amide N and proline N in ff14SB; no new atom type is introduced. For free-energy work, recompute the RESP charges with antechamber or the R.E.D. Server on the ACE-NMeAA-NME dipeptide at HF/6-31G* in vacuo.

**Limit:** the N-methyl heavy atoms (CN and its three H) must be present in the input structure. `patch_nonstandard` removes spurious atoms and does not add missing heavy atoms.

### Supported templates

| Input code(s) | Template name | Based on | Charge source |
|---------------|---------------|----------|---------------|
| SAR, NMG | NMG | Glycine | ForceField_NCAA RESP |
| NMA, MAA | NMA | Alanine | ForceField_NCAA RESP |
| MVA | MVA | Valine | ForceField_NCAA RESP |
| MLE | MLE | Leucine | ForceField_NCAA RESP |

### Without a curated template

These N-methylated residues have no curated template. With `--small-molecules auto` they go through the [GAFF2 route](#gaff2-route-for-other-non-canonical-residues); the bundled examples exercise that route for BMT, ABA, IAM, 0EH and MK8 only. A curated template needs a custom XML with RESP charges, the input code to template name mapping in `NME_AA_MAP`, and a test.

| PDB code | Full name | Notes |
|----------|-----------|-------|
| NMI | N-methyl-isoleucine | |
| NMC | N-methyl-cysteine | May form a disulfide, which interacts with cyclic patching |
| NMS | N-methyl-serine | |
| NMT | N-methyl-threonine | Not BMT: the MeBmt of cyclosporin A has the code BMT, which the GAFF2 route parameterises |
| NMK | N-methyl-lysine (backbone) | Distinct from side-chain trimethyl-lysine |
| NMR | N-methyl-arginine | |
| NMQ | N-methyl-glutamine | |
| NMH | N-methyl-histidine | Needs protonation-state handling |
| NMW | N-methyl-tryptophan | |
| NMY | N-methyl-tyrosine | |
| MME | N-methyl-methionine | |

The ForceField_NCAA RESP charges for all of these are in the ffncaa.zip supplementary data ([Wayback Machine](https://web.archive.org/web/20160322042443/http://selene.princeton.edu/FFNCAA/files/ffncaa.zip)).

---

## Cyclic Peptide-Specific Non-Standard Residues

`patch_cyclic_topology` (`core/cyclic.py`) creates these. They are not input residue names and appear in the OpenMM topology only after patching.

| Internal name | Origin | Description | Parameters |
|---------------|--------|-------------|------------|
| CYX | CYS | Disulfide-bonded cysteine | ff14SB built-in |
| ASPL | ASP | ASP sidechain CG acting as lactam carbonyl | custom XML |
| GLUL | GLU | GLU sidechain CD acting as lactam carbonyl | custom XML |
| LYSL | LYS | LYS sidechain NZ acting as lactam amide N | custom XML |

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

`detect_cyclization(topology, positions, chain_id)` takes the chain ID as `detect_nonstandard` does (the author ID that `--peptide-chain` takes, or the ID in the topology), and its entries name the chain as the topology does. A peptide can have several closures (SFTI-1 in 3P8F has `head_to_tail` and `disulfide`). Detection reads the bonds recorded in the file (`_struct_conn` records of a CIF, CONECT records of a PDB) and, for the amide, disulfide and lactam types, also atom distances. The staple residues are parameterised by the GAFF2 route, and their cross-link is covered by the GAFF2 terms, so it needs no template of its own.

---

## Unsupported Cyclization / Crosslink Types

`detect_cyclization` raises `CyclizationError` with guidance for these crosslink types. They require a `custom_bond_handler` with GAFF2/SMIRNOFF parameters.

| Type | Example | Notes |
|------|---------|-------|
| Thioether bridge | Lanthipeptide S–C | Use GAFF2 |
| Macrolactone | Ser/Thr/Tyr O → carbonyl C | Use GAFF2 |
| Biaryl ether | Vancomycin-type | Use GAFF2 or SMIRNOFF |

---

## Phosphorylated Residues

**Strategy:** the AMBER phosaa parameters, not GAFF2.

Phosphoserine (SEP), phosphothreonine (TPO) and phosphotyrosine (PTR), together with the phosaa protonation variants (S1P, T1P, Y1P, H1D, H2D, H1E, H2E), use the RESP-fitted `phosaa14SB.xml` that ships with openmmforcefields (Raguette et al., *J. Chem. Theory Comput.* 2024, 20, 7199). `core/phosaa.py` adapts its atom-type names to `amber14-all.xml` and changes no charge or bonded term. A phosphate monoester carries a net charge of −2. The GAFF2 route skips these residues on purpose: it builds residues in their neutral form and would protonate the phosphate to net 0.

`core/phosaa.py` also supplies hydrogen definitions, so `addHydrogens` places the hydrogens of a heavy-atom-only phospho residue: SEP gets H, HA, HB2 and HB3 whatever the pH, because the residue name sets the protonation of the phosphate (SEP, TPO and PTR are the dianions, S1P, T1P and Y1P the monoanions). The definitions are loaded for the duration of one `addHydrogens` call (`phosaa.hydrogen_definitions()`) and the earlier state is restored, so a structure prepared in a fresh process and one prepared after a cyclic peptide come out the same. The relaxation, the interaction energy and prep use these parameters. In the metrics, the non-bridging phosphate oxygens O1P, O2P and O3P count as negative atoms in the salt-bridge and Coulomb tables (−2/3 e each) and as O(−) in the solvation parameters. The bundled example is the phosphopeptide of 1QJB (chain Q, SEP).

---

## GAFF2 Route for Other Non-Canonical Residues

**Strategy:** for every residue that ff14SB and the curated templates do not cover, `core/gaff_ncaa.py` builds an OpenMM residue template with `<ExternalBond>` tags from the residue's own geometry. It runs with `--small-molecules auto` and covers, for example, MeBmt (BMT) and 2-aminobutyrate (ABA) in cyclosporin A, IAM in 1XY4, and the staple residues 0EH and MK8 in 3V3B.

The GAFF template generator of openmmforcefields cannot match a backbone residue, because the backbone N and C carry peptide bonds and the generator only matches residues without external bonds. The route works around this for each residue:

1. It builds an RDKit molecule of the heavy atoms and adds a carbon cap at each external bond (backbone N and C, closure atoms). For a residue with an N-CA-C=O backbone, the cap on a peptide bond is completed to an acetyl group (ACE, on the N) or an N-methylamide (NME, on the C), so that the backbone is an amide when the charges are computed. The cap of a side-chain link (disulfide, staple, lactam) stays a methyl carbon, and so do all the caps of a residue without that backbone, such as glutathione; the log says which of the two a template got.
2. **Bond orders and formal charges come from the wwPDB Chemical Component Dictionary (CCD)** that ships with biotite, so no network access is needed. The entry is used only if it describes the same molecule: every atom name must exist with the same element and the bonds between named atoms must be the same set. The molecule has no hydrogens, so valence-based perception would leave every bond single. Ionisable groups are neutralised.
3. RDKit adds hydrogens and reads the stereocentres and E/Z double bonds from the geometry. `antechamber` and `sqm` (AmberTools, on `PATH`) compute AM1-BCC charges on a conformer that RDKit embeds with `random_seed` (default 1), and openmmforcefields' `GAFFTemplateGenerator` supplies the GAFF2 atom types and bonded parameters.
4. The template drops the caps, adds an `<ExternalBond>` at every capped atom, retypes the backbone atoms to the ff14SB types, and adds explicit GAFF terms for the backbone–side-chain junctions. The charge of the removed cap atoms is spread evenly over the remaining atoms, so the template has an integer net charge, normally 0; the charges add up to the integer charge of the residue to 1e-9 e.
5. The hydrogens of the residue are injected into the topology, which is rebuilt so that the residue matches its template. They sit where RDKit puts them for the heavy atoms. A residue that already holds exactly its template's hydrogens, the same names on the same parent atoms, keeps them and their positions: prep injects them, and the relaxation and the energy step then read them back from the relaxed file, so the `raw` energy of a relaxed complex uses the relaxed hydrogens. A residue with other hydrogens loses them and gets the ones RDKit places.

**Fallback to single bonds.** A residue that is not in the CCD, whose atoms disagree with its entry, or whose CCD bond orders do not sanitise in RDKit is built with every bond single. Its double bonds, aromatic rings and hydrogen count are then unreliable, and a warning says so. The pipeline records the outcome for each residue under `ncaa_bond_order_source` in `results["prep"]` (cyclic peptides) and `results["relax"]`: `"ccd"` or `"single_bonds"`. For the somatostatin analogue 1XY4, prep reports `{"IAM": "ccd"}`. Use the CCD residue code and atom names to get the chemistry right.

**Charge scheme.** `NcaaTemplateList.charge_scheme_by_residue` names, for each template, how the capped molecule ended at its external bonds, with the version of the scheme (`CHARGE_SCHEME_VERSION`, 2): `"ace-nme-caps-v2"` (the peptide bonds were capped by ACE and NME), `"methyl-caps-v2"` (every external bond by a methyl carbon) or `"uncapped-v2"` (the residue has no external bond). The pipeline records it for each residue under `ncaa_charge_scheme`, next to `ncaa_bond_order_source`. Version 1 capped every external bond with a methyl carbon; a template or a charge that is stored somewhere has to carry the version, because the same residue and seed give other charges in another version.

### Reproducibility

The template of a residue is the same in every run and at every step: prep, relaxation and the energy step each build it, and they give byte-identical XML for the same residue and seed. `sqm` normally picks its diagonalisation routine by timing seven of them at start-up, and the routine chosen can change the charges between builds. The route runs `sqm` with its own diagonaliser (`diag_routine=1`) on one thread, and so do the molecules of the `small_molecules` list of `RelaxationConfig`. `random_seed` (default 1, `--random-seed` in the pipeline CLIs) seeds the conformer that `sqm` starts from, and AM1-BCC charges depend on that conformer (those of MeBmt differ by up to 0.037 e between seeds 1 and 2). A run with `--random-seed none` does not reproduce, and results from different seeds are not comparable in the last digits.

### Charge-model limits

- **Neutral form.** Bond-order perception runs at total charge 0, so carboxylic acid, phosphate, sulfate, primary amine and guanidine side chains come out uncharged. A warning names the residue, the group and the expected net charge; electrostatics and energies involving such a residue are unreliable unless its charge is set by hand.
- **Backbone charges.** The template keeps the AM1-BCC charges of the capped molecule for every atom, backbone atoms included, and only the atom types of the backbone are set to ff14SB. Energies that involve such residues are approximate.
- **General force field.** GAFF2 is a general small-molecule force field; its parameters for these residues are approximate.

---

## Charge-Model Caveats

| Residues | Charge source | Caveat |
|----------|---------------|--------|
| D-amino acids | ff14SB, as the L counterpart | none beyond the missing-atom limit above |
| N-methylated (NMG, NMA, MVA, MLE) | ForceField_NCAA RESP (ff03 protocol) | condensed-phase RESP used with ff14SB |
| Lactam residues (ASPL, GLUL, LYSL) | approximate, from the ff14SB ASN/GLN analogues | recompute RESP charges for production or free-energy work and pass them through `custom_bond_handler` |
| Phosphorylated (SEP, TPO, PTR, ...) | phosaa14SB (RESP) | none known here |
| Everything else | AM1-BCC on a capped, neutral molecule | neutral form; AM1-BCC backbone charges (above) |

---

## Adding a New Non-Standard Residue

### New D-amino acid code
Add one entry to `D_AA_MAP` in `core/nonstandard.py`:
```python
"DXX": "STD",   # D-name → L counterpart
```
No XML template is needed.

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
