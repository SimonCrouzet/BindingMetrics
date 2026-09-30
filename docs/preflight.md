# Pre-flight Checks

Some steps cannot take some inputs. A structure-prediction model may read a cyclic binder as a linear chain, a metric may only make sense for a peptide, and another may need a receptor chain. Without a check the run fails late, or returns numbers for the wrong molecule. `binding_metrics.capabilities` lets a step declare what it accepts, describes the input once, and compares the two before anything expensive starts.

Importing the module needs the standard library and the two residue tables of `binding_metrics.core`. numpy and biotite are imported when a structure is profiled. OpenMM is never imported.

Three rules hold everywhere:

- A step that declares nothing is never refused: every field of `Capabilities` defaults to "no constraint".
- A constraint is declared only when code or the model's documentation backs it, and it carries a sentence that says why it holds and what to do instead.
- An input the check cannot classify never blocks. A binder type that cannot be estimated skips the checks that depend on it, and the profile says so.

---

## Capabilities

`Capabilities` is a frozen dataclass. A set field lists what is accepted, an empty set accepts anything, and an input is refused when it has a member that the set does not list.

| Field | Meaning |
|---|---|
| `binder_types` | Accepted binder types: `peptide`, `miniprotein`, `nanobody`, `antibody`. |
| `closures` | Accepted ring closures: `none` (a linear binder), `head_to_tail`, `disulfide`, `lactam`, `staple`, `other`. |
| `residue_classes` | Accepted residue classes in the binder: `canonical`, `d_amino`, `n_methyl`, `phospho`, `other_ncaa`, `cap`, `ligand`. |
| `min_binder_residues`, `max_binder_residues` | Bounds on the number of amino-acid residues of the binder, ends included. |
| `multi_chain_binder` | `False` refuses a binder that spans several chains. |
| `needs` | What the step cannot run without: `receptor_chain`, `reference_structure`, `predicted_structure`, `gpu`. |
| `reasons` | One sentence per constraint. Keys are the field name (`"closures"`) or field and value (`"closures:disulfide"`); the second is looked up first. |
| `caveats` | One sentence per accepted input value that was never validated, keyed `"<field>:<value>"`. |
| `version` | The version of the model or metric the limits were checked against. |

A constraint without a `reasons` entry under its field name is rejected when the object is built, so no limit can be declared without saying why. `closures={"none", "head_to_tail"}` refuses a binder with a disulfide. A step that needs a ring lists every family except `none`, and a linear binder is refused.

`Capabilities.check(profile, provided=None)` returns every violation, not the first one. Each `Violation` carries the constraint, the fact found in the input, the requirement, and the step's own sentence. `receptor_chain` is checked against the profile. The other needs are facts only the caller knows: they are checked when `provided` lists what the caller makes available, and skipped when it is `None`. `caveats_for(profile)` returns the caveats that apply to the input, and `accepts(profile)` is the boolean form.

A predictor adapter declares its limits in the class attribute `PredictionParser.capabilities`, read without creating the adapter through `ParserSpec.load_capabilities()`. The value is `None` or a `Capabilities`; the contract tests of `tests/predictors` check it.

---

## Input profile

`profile_input(structure, binder_chain, receptor_chain=None, binder_type="auto")` reads the structure once and returns an `InputProfile`. `structure` is a PDB or mmCIF path (first model, author chain IDs, bonds from CONECT records or `_struct_conn`) or a biotite `AtomArray`. `binder_chain` may be a list of IDs for a binder that spans several chains. A chain ID that is not in the structure raises a `ValueError` that lists the chains.

| Field | Content |
|---|---|
| `binder_chains`, `receptor_chain` | The chain roles. |
| `n_binder_residues` | Amino-acid residues of the binder. Waters, ions, ligands and capping groups are not counted. |
| `binder_type`, `binder_type_source` | The type and whether it was `given` or `estimated`. |
| `closures`, `closure_bonds` | The closure families present (`none` alone for a linear binder) and the links behind them. |
| `residue_classes`, `residue_names` | The classes present in the binder and the distinct residue codes of each. |
| `chain_ids`, `notes` | Every chain of the structure; what makes the profile less certain. |

### Ring closures

`detect_closures(atoms, chain_id)` finds the closures on the biotite structure without OpenMM. It follows `core.cyclic.detect_cyclization`, and `tests/test_pre_closure_parity.py` checks that both return the same links (same type, same two atoms) for every chain of the bundled examples 1YCR, 1CWA, 3P8F, 1XY4, 3V3B and 1QJB.

| Type | Link |
|---|---|
| `head_to_tail` | C of the last residue to N of the first |
| `disulfide` | SG to SG |
| `lactam_n_asp`, `lactam_n_glu` | Asp CG or Glu CD to the N-terminus |
| `lactam_c_lys` | Lys NZ to the C-terminus |
| `lactam_sc_lys_asp`, `lactam_sc_lys_glu` | Lys NZ to an Asp CG or Glu CD side chain |
| `hydrocarbon_staple` | an all-carbon link between two side chains, from the bond table |
| `unsupported_crosslink` | any other link between residues that are not neighbours, from the bond table |

A pair counts as linked when the bond table lists it or the atoms are closer than 2.0 A (2.6 A for SG to SG), the same cut-offs as `core.cyclic`. A strained model without a bond record is therefore still found. An `AtomArray` without a bond table finds no staples or other cross-links, and the profile notes it. Two differences from `detect_cyclization`: a cysteine named CYX or DCY counts for a disulfide, and only the amino-acid residues of the chain are read, so waters and ligands that share the author chain ID cannot shift the first or last residue.

The closure families are `head_to_tail`, `disulfide`, `lactam` (the four lactam types), `staple` and `other`.

### Residue classes

| Class | Residues |
|---|---|
| `canonical` | The 20 amino acids, their AMBER and CHARMM protonation variants (HID, HIE, HIP, CYX, HSD, ...) and the lactam template names. |
| `d_amino` | The D-amino-acid codes of `core.nonstandard.D_AA_MAP` (DAL, DTR, ...). |
| `n_methyl` | The N-methylated codes of `NME_AA_MAP` (SAR, NMA, MVA, MLE, ...) and the template names. |
| `phospho` | SEP, TPO, PTR. |
| `other_ncaa` | Any other amino acid: listed as peptide-linking in the Chemical Component Dictionary, or with the backbone atoms N, CA and C (BMT, ABA, IAM, MSE, MK8, ...). |
| `cap` | ACE, NME, FOR, NH2. |
| `ligand` | A group that is not an amino acid (a glycan, a small molecule). |

### Binder type

`binder_type="auto"` estimates the type from the number of residues: at most 40 a `peptide`, at most 100 a `miniprotein`, longer `unknown`. The 40-residue line is the one the US FDA uses to separate peptides from proteins (21 CFR 600.3(h)(6)); the 100-residue line is a convention of this package that keeps miniproteins apart from nanobodies, which are about 110 to 130 residues long. Neither is a physical boundary. A nanobody or an antibody chain cannot be told from any other domain of that length by size, so they are only ever set by the caller: `binder_type` accepts `peptide`, `miniprotein`, `nanobody` and `antibody`. `unknown` never refuses an input: the checks that depend on the type are skipped.
