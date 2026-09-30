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
| `needs` | What the step cannot run without: `receptor_chain`, `reference_structure`, `predicted_structure`, `gpu`. `receptor_chain` is met by a receptor given or by any other protein chain of the structure, because the metrics take the largest other protein chain when none is given. |
| `extra_checks` | Functions `profile -> violations` for a limit that the sets above cannot state, such as the residue names that a model's query builder can express. A check reuses the code that enforces the limit at run time and imports it when called, so declaring it costs nothing at import. |
| `reasons` | One sentence per constraint. Keys are the field name (`"closures"`) or field and value (`"closures:disulfide"`); the second is looked up first. |
| `caveats` | One sentence per accepted input value that was never validated, keyed `"<field>:<value>"`. |
| `version` | The version of the model or metric the limits were checked against. |

A constraint without a `reasons` entry under its field name is rejected when the object is built, so no limit can be declared without saying why. The violations of an `extra_checks` function carry their own reason. `closures={"none", "head_to_tail"}` refuses a binder with a disulfide. A step that needs a ring lists every family except `none`, and a linear binder is refused.

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
| `residue_labels` | For each class, the residues in chain order as `NAME number` (with the insertion code), for messages that point at a residue. |
| `chain_ids`, `other_protein_chains`, `notes` | Every chain of the structure; the protein chains that are not the binder; what makes the profile less certain. |

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

A pair counts as linked when the bond table lists it or the atoms are closer than 2.0 A (2.6 A for SG to SG), the same cut-offs as `core.cyclic`. A strained model without a bond record is therefore still found. An `AtomArray` without a bond table finds no staples or other cross-links, and the profile notes it. One difference from `detect_cyclization`: only the amino-acid residues of the chain are read, so waters and ligands that share the author chain ID cannot shift the first or last residue. Both take a disulfide between cysteines named CYS, CYX or DCY.

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

---

## The check

`preflight(profile, metrics, predictor=None, *, policy="error", provided=None)` compares the profile with the limits of every requested metric and predictor, before anything runs. It reads declarations only: it never runs, prepares or instantiates a metric, a predictor or a runner.

- `metrics` are registry names (`"omega"`) or objects with a `name` and a `capabilities` attribute (a `MetricSpec`). A name the registry does not know has no declared limit.
- `predictor` is a registered name (`"of3"`), an adapter or runner (class or instance) with a `capabilities` attribute, a `Capabilities`, or a list of these.
- `provided` lists what the caller makes available among the needs `reference_structure`, `predicted_structure` and `gpu`. `None` leaves them unchecked and the report says so.

All violations are collected and reported at once. Each has the fact found in the input, the requirement, the step's own reason and a fix. For a refused predictor the fix lists the other registered predictors in two lists: those whose declared limits accept the input, and those that declare no limits (which is not the same as validated). The refused predictor is never offered, even when it is passed under another name than its registry entry: the entry is matched by registry name, by display name or by adapter class. When no other predictor declares support for the input, or none is registered, the fix says so.

| Policy | Effect of an incompatibility |
|---|---|
| `error` (default) | Raises `IncompatibleInputError`, a `ValueError` that carries `.report`. Nothing runs. |
| `skip` | The metrics with a violation are left out (`report.metrics_to_run` holds the rest). A refused predictor sets `report.predictor_usable` to `False`; the caller leaves it out with what depends on it, because `preflight` does not know which metrics read a prediction. |
| `warn` | Every violation is logged and everything runs. |

Inputs a step accepts but never validated give a warning in `report.warnings` under every policy, from `Capabilities.caveats`. `report.to_dict()` is JSON-ready and `report.format()` prints the plan.

An example, with a made-up model that reads head-to-tail closures only and the bicyclic peptide of 3P8F:

```
Pre-flight check failed: 1 incompatibility between the input and what was requested (policy: error).
Input: binder chain I: 14 residues; type peptide (estimated from size); closures head_to_tail, disulfide; residue classes canonical; receptor chain A

predictor NarrowFold 9.9: closures
    found:    the binder has a disulfide bond (CYS 3.SG - CYS 11.SG)
    requires: closures limited to: none, head_to_tail
    why:      A disulfide has no field in the query.
    fix:      use a predictor whose declared limits accept this input: WideFold (wide); or use policy='skip' to leave the predictor out and run the rest
```

---

## Declared limits

A limit is declared only where code or the documentation of the model shows it, and its `reasons` sentence names that source. The tests in `tests/test_pre_openfold3_limits.py` and `tests/test_pre_metric_limits.py` pin the behaviour each sentence describes.

### OpenFold3 0.5.0

Declared on `OpenFold3Parser.capabilities`.

| Limit | Basis |
|---|---|
| Closures: `none` and `head_to_tail` | `cyclic: true` on a protein chain wraps the whole chain, so it is head-to-tail (`openfold3/core/utils/relpos.py`). The query schema has `covalent_bonds` and nothing reads it (`openfold3/projects/of3_all_atom/config/inference_query_format.py`), so a disulfide, a lactam, a staple or another cross-link cannot be given. |
| Residues that the query builder cannot express | `check_openfold3_residues` calls `metrics._openfold_run._residue_letter_and_ccd`, the rule of `_extract_query_chain`, and the reason is the text of the `UnmappableResidueError` that the builder raises for the same chain. D-amino acids, N-methylated and other peptide-linking Chemical Component Dictionary residues are expressible and pass. |

Warnings, not refusals: a head-to-tail binder is folded as a linear chain, because the query builders of this package do not write `cyclic: true`; terminal capping groups and non-amino-acid groups (ligands, glycans) are left out of the query by `_extract_query_chain`, so the prediction is of the uncapped peptide without them.

### Protenix 2.0.0

Declared on `ProtenixParser.capabilities`, from `docs/infer_json_format.md` of Protenix (commit 85767b8, 2026-09-21).

| Limit | Basis |
|---|---|
| Closures: `none`, `head_to_tail` and `disulfide` | "Covalent bonds between two polymer residues ... are generally not supported. Exceptions are made for cyclic peptides", for a head-to-tail amide bond or a disulfide between cysteines; other polymer-polymer bonds "are not reliably handled by the current model". The source takes any atom pair (`json_to_feature.py`), so a lactam, a staple or another link can be written; the limit is about what the model does with it. |

### Boltz-2 2.2.1

Declared on `Boltz2Parser.capabilities`: no limit, one warning. The `cyclic: true` flag wraps a chain head-to-tail, and the `bond` constraint takes any two atoms of the input (`atom_idx_map` in `boltz/data/parse/schema.py`), which the featuriser turns into a cyclic period when it joins the first and last residue of a chain, so a disulfide or a lactam between canonical residues can be given. The documentation lists the `bond` constraint as supported for "CCD ligands and canonical residues" only (`docs/prediction.md`), which is why a hydrocarbon staple, whose residues are not canonical, gets a warning and not a refusal.

### Metrics

Declared on `MetricSpec.capabilities`, a keyword-only optional field.

| Metric | Limit | Evidence that the tests pin |
|---|---|---|
| `interface` | needs a receptor chain | raises `ValueError` ("Chain auto-detection failed") for one protein chain |
| `coulomb` | needs a receptor chain | returns 0.0 kJ/mol and no charged pair, which reads as "no interaction" |
| `shape_complementarity` | needs a receptor chain | returns NaN with no surface dots |
| `void_volume` | needs a receptor chain | returns NaN with the reason "fewer than two protein chains" |
| `delta_sasa_static`, `hbonds`, `saltbridges` | need a receptor chain | `receptor_chain` is a required argument (`TypeError`) |
| `structure_interaction_energy` | needs a receptor chain; closures limited | returns `success=False` ("Could not identify two protein chains"); see the closure row below |
| `evobind_score` | needs a receptor chain and a predicted structure | receptor chain required; `evobind_score` is None without per-atom pLDDT |
| `evobind_adversarial` | needs a receptor chain and a predicted structure | `afm_structure_path` and the receptor chain are required |
| `interface_pae` | needs a receptor chain and a predicted structure | `confidences_path` and the receptor chain are required |
| `openfold`, `prediction` | need a predicted structure | `output_dir` / `prediction_dir` are required; they run no model |
| `dockq` | needs a reference structure | `reference_path` is required |
| `md_implicit`, `structure_interaction_energy` | closures `none`, `head_to_tail`, `disulfide`, `lactam`, `staple` | both call `core.cyclic.patch_cyclic_topology` without a switch, and it raises `CyclizationError` for another link (a thioether, a macrolactone, a biaryl ether) |

`preflight` checks the receptor need from the profile, and the reference and prediction needs only when the caller passes `provided`.

### Considered and not declared

| Candidate | Why it is not declared |
|---|---|
| OpenFold3 binder size (memory) | No source gives a number. |
| OpenFold3 binder made of several chains, or of a given binder type | Nothing shows a limit. |
| OpenFold3 on D-amino acids, N-methyl and phospho residues | The builder expresses them through CCD codes; no benchmark or documentation says how well the model predicts them, so no caveat is written. |
| `requires_gpu` metrics (`receptor_quality`, `md_implicit`, `structure_interaction_energy`) | The field is a scheduling hint ("on CUDA by default"), not a limit of the input. |
| `receptor_quality` | Runs on a single chain; the receptor argument is optional. |
| `ramachandran`, `omega` | They score the closing bond of a head-to-tail peptide and work on a linear one; no metric of the registry is cyclic-only. |
| Antibody or nanobody metrics | None exists in the registry. |
| `structure_rmsd` | Both structures come from the same run; it is not a reference supplied by the user. |
| Trajectory metrics | Their input is a trajectory and a topology, which the needs vocabulary does not describe. |
| Residue limits of `md_implicit` and `structure_interaction_energy` | Unknown residues go through GAFF2 templates whose success depends on the residue and on `antechamber`. |
| Closure limits of `openfold` and `interface_pae` | They read an existing output that another model may have written; the limit belongs to the predictor that writes it. |
| `compute_evobind_adversarial_from_records` | It is not a registry entry, so there is no `MetricSpec` to declare on; its needs are those of `evobind_adversarial`. |
