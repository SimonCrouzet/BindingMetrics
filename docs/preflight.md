# Pre-flight Checks

`binding_metrics.capabilities` lets a step declare the inputs it accepts, describes the input once, and compares the two before anything expensive starts. A prediction model may read a cyclic binder as a linear chain, and a metric may need a receptor chain; without the check such a run fails late or returns numbers for a different molecule than the input.

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
| `modes` | The ways a model can be used that it supports: `predict`, `refold`, `score`, `score-lock`, `refold-lock` (see [Modes](#modes)). Empty declares nothing and refuses nothing; a set that leaves a mode out refuses a request for it. |
| `extra_checks` | Functions `profile -> violations` for a limit that the sets above cannot state, such as the residue names that a model's query builder can express. A check reuses the code that enforces the limit at run time and imports it when called, so declaring it costs nothing at import. |
| `reasons` | One sentence per constraint. Keys are the field name (`"closures"`) or field and value (`"closures:disulfide"`); the second is looked up first. |
| `caveats` | One sentence per accepted input value that was never validated, keyed `"<field>:<value>"`. |
| `version` | The version of the model or metric the limits were checked against. |

A constraint without a `reasons` entry under its field name is rejected when the object is built. The violations of an `extra_checks` function carry their own reason. `closures={"none", "head_to_tail"}` refuses a binder with a disulfide. A step that needs a ring lists every family except `none`, and a linear binder is refused.

`Capabilities.check(profile, provided=None, mode=None)` returns every violation, not the first one. Each `Violation` carries the constraint, the fact found in the input, the requirement, and the step's own sentence. `receptor_chain` is checked against the profile. The other needs are facts only the caller knows: they are checked when `provided` lists what the caller makes available, and skipped when it is `None`. `caveats_for(profile, mode=None)` returns the caveats that apply to the input and the mode, and `accepts(profile, mode=None)` is the boolean form.

### Modes

A mode is how a model is used for a complex. The words are the constant `capabilities.MODES` and the values of `PredictionRequest.mode` and of `--prediction-mode`; `score-lock` and `refold-lock` are written with the hyphen everywhere.

| Mode | Meaning |
|---|---|
| `predict` | From sequences only. |
| `refold` | The receptor is given as a template and the binder is predicted freely. |
| `score` | Every chain is given its own structure as a template and the relative pose of the chains is not given: the model re-docks them. |
| `score-lock` | `score` with the relative pose of the chains pinned to the input, by a forced template, constraints or both. |
| `refold-lock` | `refold` with the receptor pinned to the input by a forced template and the binder left free. |

`modes` lists what a model supports. A set that leaves a mode out refuses a request for it and needs `reasons["modes"]` (or `"modes:<mode>"`) saying what is not supported and why; the full set declares full support and refuses nothing. Partial or unreliable support of a mode is a caveat, `caveats["modes:<mode>"]`, a warning that applies to a request for that mode whether or not `modes` is declared. A model that declares nothing is never refused for a mode.

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

A pair counts as linked when the bond table lists it or the atoms are closer than 2.0 A (2.6 A for SG to SG), the same cut-offs as `core.cyclic`, so a strained model without a bond record is still found. An `AtomArray` without a bond table finds no staples or other cross-links, and the profile notes it. `detect_closures` differs from `detect_cyclization` in one point: only the amino-acid residues of the chain are read, so waters and ligands that share the author chain ID cannot shift the first or last residue. Both take a disulfide between cysteines named CYS, CYX or DCY.

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

`binder_type="auto"` estimates the type from the number of residues: at most 40 a `peptide`, at most 100 a `miniprotein`, longer `unknown`. The 40-residue line is the definition of the US FDA (21 CFR 600.3(h)(6)); the 100-residue line is a convention of this package. A nanobody or an antibody chain cannot be told from another domain of that length by size, so only the caller sets them: `binder_type` accepts `peptide`, `miniprotein`, `nanobody` and `antibody`. `unknown` never refuses an input: the checks that depend on the type are skipped.

---

## The check

`preflight(profile, metrics, predictor=None, *, policy="error", predictor_policy=None, provided=None, mode=None, runnable=None, weights=None, weights_kinds=None, design_source=None, design_policy="skip", prediction_models=None, msa_policy="default", msa_server=None, receptor_msa=None)` compares the profile with the limits of every requested metric and predictor, before anything runs. It reads declarations only: it never runs, prepares or instantiates a metric, a predictor or a runner.

- `metrics` are registry names (`"omega"`) or objects with a `name` and a `capabilities` attribute (a `MetricSpec`). A name the registry does not know has no declared limit.
- `predictor` is a registered name (`"of3"`), an adapter or runner (class or instance) with a `capabilities` attribute, a `Capabilities`, or a list of these.
- `provided` lists what the caller makes available among the needs `reference_structure`, `predicted_structure` and `gpu`. `None` leaves them unchecked and the report says so.
- `predictor_policy` is the policy for the predictors when it is not the one of the metrics; a caller that only reads a prediction made elsewhere passes `warn`.
- `mode` is how the predictors are asked to run, one of `MODES`; `None` does not say and the mode is not checked. A predictor that declares `modes` without it is refused (constraint `modes`; fact "mode 'score-lock' was requested", requirement "supported modes: ..."), and the fix lists the models that declare the mode, then the models with no declaration. A caveat for the mode is a warning.
- `runnable` names the predictors that can be run from here (the command-line tools pass the keys of `cli.prediction.RUNNERS`); a model outside it that the fix offers is marked `[no runner here: give its output with --prediction-dir]`. All four registered models have a runner, so the mark shows only for a model that a caller adds to the registry without one.
- `weights` is the path of custom weights (`--prediction-weights`). It must exist and be readable (`capabilities.check_weights_path`; a plain `ValueError` that lists what was found, not a violation). A predictor whose runner takes no custom weights is refused (constraint `weights`, kind `predictor`; fact "custom weights were given (path, a file)", requirement "a runner that starts the model with weights you choose"), and the fix names the models whose runner does, with the kind each takes, then offers `--prediction-dir` and leaving the weights out. A predictor whose runner takes them gets a note, not a violation.
- `weights_kinds` maps a model name to the kind of weights its runner takes (`"file"` or `"directory"`). The command-line tools read it from the runner classes of `cli.prediction.RUNNERS` (`runner_weights_kinds`: the class attributes `supports_custom_weights` and `weights_kind`; nothing is instantiated). `None` says the caller does not know, and the report says the weights were not checked against the runners.
- `design_source` is the pipeline that made the designed binder, and `design_policy` says what to do with a check that rests on independence from it; see [The design origin](#the-design-origin). `prediction_models` names the models that make the second structures of those checks, by default the names of `predictor`.
- `msa_policy` is `default`, which checks nothing, or `forbid-server`, which says that no sequence may be sent to an MSA server; `msa_server` says whether the run of the predictors would use one (`True`, `False`, or `None` when the caller does not know or the model is not run here). Under `forbid-server`, `msa_server=True` gives one violation per predictor (constraint `msa_policy`, kind `msa`, fact "the run would use an MSA server"), and `None` adds a note that the policy was not checked. See [The MSA policy](#the-msa-policy). `report.msa_policy` holds the policy, `report.format()` prints "MSA policy: forbid-server (no MSA server may be called)" when it is not `default`, and `to_dict()["msa_policy"]` is present only then.
- `receptor_msa` is the receptor MSA of the request: `None` (no source), the word `single` or the path of an alignment. It is read only by the caveat for a target with neither a template nor an MSA (see [A target with neither a template nor an MSA](#a-target-with-neither-a-template-nor-an-msa)); the file is not opened.

A receptor chain that is also a binder chain gives one violation of kind `input` (constraint `chain_roles`, subject "input chains") whatever the steps: found "chain 'B' is named as the binder and as the receptor", required two different chains, and the fix names the options. It follows `policy`: `error` raises `IncompatibleInputError`; `skip` and `warn` report it and leave nothing out, because every step is affected. `run_pipeline`, `run_batch` and both commands refuse the same chain for the two roles earlier (`ValueError`; on the command line see below).

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

## The design origin

`evobind_adversarial` compares the design pose with a second prediction of the same complex. A design pipeline selects its binders with the confidence of the models it runs, so a second structure from such a model is not independent of the design. `preflight` takes the pipeline that made the design and leaves the check out when its second structure would come from a model that the pipeline ran itself.

| `design_source` | Models the pipeline ran itself |
|---|---|
| `boltzgen` | `boltz2` (Boltz-2) |
| `bindcraft`, `colabdesign` | `af2` (AlphaFold2, run through ColabFold) |

The table is `capabilities.DESIGN_SOURCE_PREDICTORS`; a row added there supports another pipeline. The name is read in any case. A list of registry names, `design_source=("boltz2",)`, gives the models directly, for a pipeline that is not in the table. A name that is not in the table is never refused: the report warns that independence was not checked and everything runs. Without a source nothing is decided and the report is unchanged: `preflight` logs, at INFO level, that independence from the design was not checked for a check that rests on it.

The checks that rest on independence are `capabilities.INDEPENDENCE_DEPENDENT_CHECKS`, today `evobind_adversarial`. The confidence metrics (`prediction`, `openfold`, `interface_pae`, `evobind_score`) report one model's confidence in its own structure and run whoever made the design.

`design_policy` is separate from `policy`, so `--on-incompatible` does not change it. It applies to a check on a model that the design ran, and only to that check:

| `design_policy` | Effect |
|---|---|
| `skip` (default) | The check is left out of `report.metrics_to_run` and put in `report.skipped_metrics` when every model that makes its second structure is one the design ran (with several models, as in a pose set, it stays for the others). One sentence goes to `report.warnings` and to the log at WARNING level. Everything else runs. |
| `warn` | The check runs. The same finding is a warning that says the result is not independent of the design. |
| `error` | Raises `IncompatibleInputError`; the message shows the finding with its reason and fix. |

The sentence names the check, the reason and the way out:

```
evobind_adversarial is skipped because the design was made with boltzgen, which ran Boltz-2 itself, so a Boltz-2 structure agrees with the design in part because the design was selected for that model; predict with AlphaFold2 / ColabFold (af2), OpenFold3 (of3) or Protenix (protenix) instead, or pass design_policy='warn' (--design-policy warn) to run it anyway
```

The report keeps the same finding in machine-readable form. A violation of kind `design` (constraint `design_origin`) is in `violations`, the skipped check is in `metrics_skipped` with the fact as its reason, and `to_dict()["design"]` is present when a source was given:

| Key | Content |
|---|---|
| `source`, `known` | The pipeline name in lower case (None when models were given), and whether its models are known. |
| `predictors` | The models the pipeline ran itself. |
| `policy` | The `design_policy`. |
| `models`, `not_independent` | The models that make the second structures, and those among them that the design ran. |
| `checks` | One entry per check and circular model: `check`, `model`, `action` (`skipped`, `run` or `refused`) and `reason`. |

`capabilities.check_design_origin(design_source, checks, models, policy=...)` makes the same decision without a structure, for a caller that has none (a step that runs after the workers of a batch, a pose set); `preflight` calls it for its metrics and predictors.

## The MSA policy

Every runner sends the sequences of the complex to a public MSA server unless it is told not to (`use_msa_server` is true by default for all four). `msa_policy="forbid-server"` forbids any server call. `receptor_msa` gives the receptor's alignment: the path of an alignment file, or the word `single` for a receptor predicted from its own sequence. The binder is single-sequence under the policy, and each runner is told so in its own way.

Three rules hold under `forbid-server`, and `binding_metrics.predictors._msa` applies them for all four runners:

1. `use_msa_server` is false. A request with `use_msa_server=True` is refused with "the sequences would be sent to an MSA server and the policy forbids it".
2. The receptor has an MSA source, `receptor_msa=PATH` or `receptor_msa="single"`. A request with none is refused.
3. The runner can take what was asked. ColabFold accepts `single` only, so the path of an alignment is refused for it with the reason, under any policy.

Rule 1 is also a pre-flight violation, and it does not depend on `--on-incompatible`: a forbidden server stops the run under `error`, `skip` and `warn` alike, also when `predictor_policy` is `warn` (exit status 1 for `binding-metrics-run --preflight-only` and for `binding-metrics-batch --preflight-only`, which reports "0 of 1 samples can run"). It is not checked for an output read with `--prediction-dir`, because the model does not run there. `make_request` applies the three rules; `prepare` and `run` apply them again to a request built by hand, then read the command line the runner has built, `extra_args` included, for a flag that switches a server on (`assert_no_server_flag`), before a file is written or a process starts.

The guard reads the command as the tools do: the hyphen and the underscore spelling of an option are one option (`--use-msa-server` is `--use_msa_server`), a value is `--name=value` or the next word, an option given twice is read from its last occurrence, a bare flag is on, a prefix of a watched name that `argparse` would accept (`--msa-m`) counts as the name, and a value that is not a boolean word is refused. It watches `--use_msa_server` (Boltz-2 and OpenFold3), `--use_msa` (Protenix) and `--msa-mode` (ColabFold, which calls the server when the flag is absent, so the command must carry `--msa-mode single_sequence`). For Boltz-2, `extra_args` may not repeat any flag the runner sets, in either spelling, under every policy (#153); the guard is the second line. `extra_args` may not carry the ColabFold template search (`--templates`), which asks the server too, under any policy.

| Runner | `receptor_msa` as a file | `receptor_msa="single"` | What keeps the server out |
|---|---|---|---|
| Boltz-2 | `.a3m` or `.csv`; the first sequence must be the receptor's; the receptor entity gets `msa: <copy of the file>` and the binder `msa: "empty"` | yes: `msa: "empty"` on both entities | `--use_msa_server` is never on the command |
| OpenFold3 | A3M; staged as `colabfold_main.a3m` for the receptor chain, whose first sequence must be the receptor's; the binder keeps the dummy MSA; needs `use_msa_server=False` under any policy | yes: the dummy MSA for the receptor as well | the command carries `--use_msa_server=false` and no other value |
| Protenix | an a3m file, written as the `unpairedMsaPath` of the receptor; the binder's file holds its own sequence; the format is not confirmed by a run | yes: a one-sequence a3m for both chains | every protein chain of the input JSON names a file before `--use_msa true` is accepted |
| ColabFold | refused, with the reason | yes: `--msa-mode single_sequence`, which applies to both chains | `--msa-mode single_sequence` must be on the command |

The key of the request holds `options["msa_policy"]` only when it is `forbid-server`, `options["receptor_msa"] = "single"` only when that was asked, and `extra_files["receptor_msa"]` only when a file is given. The store hashes the file by its content, so the same alignment at another path is the same request and another alignment is another one. A request without these fields keeps the key it had. `PredictionSession.record` writes the facts into the pose (`docs/metrics.md`, "How the prediction was requested").

On the command line the policy is `--prediction-msa-policy forbid-server`, which needs `--prediction-no-msa-server`, `--prediction-receptor-msa PATH` (or `single`) and `--predictor`. `--prediction-dir` refuses the policy, the receptor MSA and `--prediction-no-msa-server`, because the model does not run there. A model whose runner lacks the setting refuses the option as a usage error that names the model. A batch gives the same file to every sample. For OpenFold3 the file also needs the server off under any policy, which `--openfold-no-msa-server` and `--prediction-no-msa-server` do; the error of `make_request` says so.

**Status.** Tested: Boltz-2 2.2.1 and OpenFold3 0.5.0 on 1YCR with a given receptor alignment, offline (`unshare -Urn`, a user and network namespace without an interface, with CUDA available inside; `unshare -n` is refused on the development machine). Not run: Protenix (not installed; the runner is tested with a stand-in process, and the alignment format was read in the source of commit 85767b8) and ColabFold (the installation here cannot build features; `forbid-server` was checked up to the argument parser of ColabFold 1.5.4 and its `--msa-only` stage). Ligands, nucleic acids, three or more chains and complexes above 255 tokens were not run with a given alignment.

### A target with neither a template nor an MSA

The check adds a caveat to `report.warnings`, for each predictor and under every `--on-incompatible` value, when the target has neither a template nor an MSA: the mode `predict` (no template on the target), the MSA server off, and a receptor MSA that is absent or `single`. The run goes on, and `--preflight-only` exits with status 0. A given alignment (`--prediction-receptor-msa PATH`) and a search on the server both count as an MSA, so the default run (server on) and every templated or pinned mode (`refold`, `score`, `refold-lock`, `score-lock`) never get it. The rule is the same for a peptide, a miniprotein, a nanobody and an antibody. It also holds under `--prediction-msa-policy forbid-server` with `--prediction-receptor-msa single`.

A pose set (`score_pose_set`) reads predictions made elsewhere, names no predictor, mode or MSA and does not record whether the server was used, so its pre-flight block does not carry the caveat. The same holds for an output read with `--prediction-dir`.

---

## On the command line

`binding-metrics-run` and `binding-metrics-batch` call the check first, through `binding_metrics.preflight_cli.check_input`. It runs before the output directory is created, before the provenance probe, preparation, relaxation, any model run and any use of the prediction store; tests replace all of those with stubs that must not be called when the input is refused.

`--peptide-chain` and `--receptor-chain` with the same ID print `ERROR: peptide chain and receptor chain are both '<ID>'; name two different chains` to stderr and exit with status 1 before the pre-flight check, for `binding-metrics-run` and `binding-metrics-batch`.

| Option | Default | Effect |
|---|---|---|
| `--binder-type {auto,peptide,miniprotein,nanobody,antibody}` | `auto` | The binder type of the profile. |
| `--on-incompatible {error,skip,warn}` | `error` | The policy of `preflight`. |
| `--designed-with {boltzgen,bindcraft,colabdesign}` | none | The pipeline that made the designed binder (`design_source`; any case). See [The design origin](#the-design-origin). |
| `--design-policy {error,skip,warn}` | `skip` | The `design_policy` of `preflight`. `--on-incompatible` does not change it. |
| `--preflight-only` | off | Print the plan and stop. Exit status 1 when the policy is `error` and something is refused. |
| `--prediction-weights PATH` | none | Custom weights for the model (see "Custom weights" below). Checked with the rest: the path, the kind the runner takes, and that the runner takes custom weights at all. A usage error with `--prediction-dir`. |
| `--prediction-mode {predict,refold,score,score-lock,refold-lock}` | none | How the model is used (see [Modes](#modes)). The default is the mode of the runner when the model is run from here: `--openfold-mode` (`score`) for `--predictor of3`, `score` for Boltz-2 and `predict` for ColabFold and Protenix; for an output read with `--prediction-dir` the mode is not known and not checked unless it is given. Needs `--predictor`. A run from here is also limited to what its runner runs on a complex structure (OpenFold3: `refold` and `score`; Boltz-2: all five; ColabFold and Protenix: `predict`), and another mode is a usage error before anything runs. |
| `--prediction-msa-policy {default,forbid-server}` | `default` | Whether the run may call an MSA server (see [The MSA policy](#the-msa-policy)). `forbid-server` needs `--prediction-no-msa-server` and `--prediction-receptor-msa`, and the pre-flight check refuses a run that would use a server under every `--on-incompatible` value; the request is refused before anything starts. A model whose runner has no such setting refuses the option as a usage error. Needs `--predictor`; a usage error with `--prediction-dir`. |
| `--prediction-receptor-msa PATH` | none | The receptor's alignment, as the path of a file, or the word `single` for a receptor predicted from its own sequence. The binder is single-sequence. The file is identified by its content in the key of the prediction store. Boltz-2, OpenFold3 and Protenix take a file or `single`; ColabFold takes `single` only and refuses a file in the argument check, before anything runs. For OpenFold3 it also needs `--prediction-no-msa-server` (or `--openfold-no-msa-server`). Needs `--predictor`; a usage error with `--prediction-dir`. In a batch the same file serves every sample. |

**What is checked.** The steps the run executes, as registry metrics: the relaxation (`md_implicit`), `energy` (`structure_interaction_energy`), `interface`, the metrics of `geometry` one by one (`ramachandran`, `omega`, `shape_complementarity`), `electrostatics` (`coulomb`), and the model step. The model step is `openfold` in `--metrics`: without `--predictor` it is OpenFold3 (metric `openfold`, predictor `of3`); with `--predictor MODEL` it is the metric `prediction` with the limits of that model. The EvoBind adversarial check runs inside the model step. With `--designed-with` it is listed with it as the metric `evobind_adversarial`, without limits of its own, because only the design origin can leave it out; without the option it is not listed. `dockq` is not checked: without a reference the pipeline already skips it with a warning. A run with its own `Relaxer` object is not checked for the relaxation.

**The design origin on the command line.** `--designed-with` and `--design-policy` apply the rule of [The design origin](#the-design-origin) to the model of the run (`--predictor`, `of3` for the `openfold` step without one). Under `skip` (the default) the block of the model step (`results["prediction"]`, `results["openfold"]`) has `adversarial_skipped` with the reason and none of the keys of the check; `warn` runs the check and the block says `independent_of_design: false`; `error` refuses the run before anything is written. The block has `independent_of_design` whenever `--designed-with` is given (False for a model the pipeline ran, True for another one). The rule holds whatever `--on-incompatible` says, also for an output read with `--prediction-dir`, where the model that made it is the one named. Without `--designed-with` nothing is decided, and a run with a model step logs one INFO line that independence from the design was not checked (a batch logs it once). In a batch each worker applies the rule to its sample, and the model step that runs once for all samples applies it again, with the WARNING logged once for the batch and once in each sample's log; the rows get `prediction_adversarial_skipped` and `prediction_independent_of_design` (`openfold_*` without `--predictor`). `run_pipeline` and `run_batch` take the same options as `designed_with` and `design_policy`; `designed_with` also takes a list of the registry names of the models the design ran.

**What the run provides.** The needs `predicted_structure` (the model step supplies it) and `reference_structure` (when `--reference` is given) come from the arguments. The receptor need is met by a receptor given or by any other protein chain, found as the pipeline finds it.

**`--on-unmappable-residue x`.** The option asks the OpenFold3 query builder to send an `X` for a residue it cannot take and to log a warning, so the residue check of OpenFold3 is lifted and the closure limit stays.

**The mode.** The model step is checked in the mode it runs in: `--prediction-mode`, else `--openfold-mode` for OpenFold3 run from here (and for the legacy `openfold` step, which keeps its own option). A request for a mode the model does not list is refused before anything runs, with the models that declare the mode in the fix; for `score-lock` and `refold-lock` that is Boltz-2 (`--predictor boltz2 --prediction-mode score-lock`, with `--prediction-lock-threshold` for the threshold of its forced template, 2.0 angstrom unless given, for both lock modes). The mode is recorded as `mode` in `results["prediction"]` (the column `prediction_mode`) and in `results["preflight"]`; it is `null` for an output whose making is not stated. `OpenFold3Runner` also refuses `score-lock` and `refold-lock` with the declared reason, for a caller that did not run the check.

The runner is the other limit on the mode. The command line refuses a mode that the model has and its runner does not run (for instance `--prediction-mode score` with `--predictor protenix`) as a usage error that names the modes the runner runs; a mode that the model itself does not have is left to this check. `ProtenixRunner` and `ColabFoldRunner` run `predict` only, because neither can be given a structure as a template per chain, so `refold-lock`, like the three other modes, is a usage error for them that points to `--prediction-dir DIR --prediction-mode refold-lock`. With `--prediction-dir` the runner is not involved and the mode is the one you state.

**Custom weights.** `--prediction-weights PATH` gives the model weights you choose, such as a fine-tuned checkpoint. The path is checked first, before the input is profiled: it must exist, be readable, and be the kind that the runner takes, a file for OpenFold3 (one checkpoint, passed as `--inference-ckpt-path`) and a directory for a model whose weights are one. The error says what was found (the entries of the directory that holds a missing file, the content of a directory given for a file, the size of a file given for a directory). A model whose runner takes no custom weights is refused as described under `weights` above. When the weights are accepted the plan shows the note *the limits declared for the model come from its input format and architecture; the caveats about accuracy and published benchmarks refer to the standard weights*, and the declared hard limits (closures, residue classes, modes, sizes) stay in force. Under `--on-incompatible skip` the refused model step is left out and recorded, as for any limit. With `--prediction-dir` the option is a usage error, because an output made elsewhere has whatever weights produced it; `results["prediction"]["weights"]` shows the checkpoint that OpenFold3 recorded. The store keys the weights by content (see `docs/metrics.md`, "custom weights").

**A prediction made elsewhere.** With `--prediction-dir` the model does not run here and what it was given is not known (it may have been a patched model), so the limits of the model use the policy `warn` whatever `--on-incompatible` says; the metrics keep the policy.

**Policies.**

| Policy | `binding-metrics-run` | `binding-metrics-batch` |
|---|---|---|
| `error` | Raises `IncompatibleInputError` before anything is written; the command prints `ERROR:` and the message and exits 1. | The sample is an `error` row: `batch_error` holds the message, `preflight_status` is `refused`, `preflight_reason` the problems. The other samples run. |
| `skip` | The incompatible steps are `{"skipped": true, "reason": ...}`; a refused metric of `geometry` is left out alone; the rest runs. A left-out step is not a failure. | The same per sample. A left-out model step gives `openfold_skipped` and `openfold_reason` (or `prediction_skipped` and `prediction_reason`), and the sample is not part of the model call. |
| `warn` | The problems are logged and everything runs. | The same. |

In a batch each worker checks its sample first, the model step included, so a refused sample costs nothing. The whole-batch model step checks again under `skip`, before a request is built or the store is touched, and drops the samples that were left out.

**Decision.** `results["preflight"]` holds `status`, `reason`, `policy`, the steps left out and the full report; see `docs/metrics.md`. The CSV row has `preflight_status` and `preflight_reason`, and the summary (`--summary`) a short block.

**`--preflight-only`.** `binding-metrics-run` prints the plan of one input; `binding-metrics-batch` prints one plan per sample and a count, writes no CSV and creates no directory. `run_pipeline(preflight_only=True)` and `run_batch(preflight_only=True)` return the same without printing. The plan shows "MSA policy: forbid-server (no MSA server may be called)" and "Receptor MSA: PATH or single" when they are given, and a run that would use a server is refused with exit status 1.

**An input that cannot be profiled** (an unreadable structure, no binder chain) is not an incompatibility: the run goes on and the block says `not_checked` with the reason.

---

## In a pose set

`score_pose_set` runs the same check, on the profile of the first readable pose of the set and the metrics that it was asked for. The policy is its `preflight` argument (`"error"` by default, `"skip"`, `"warn"` or `"off"`), `provided` holds `reference_structure` when a reference is given and `predicted_structure` when the first pose is a predicted one, and `result["preflight"]` carries `report.to_dict()`. A metric that `skip` leaves out has the reason in the `reasons` of every pose. The design origin is part of it: `design_source` and `design_policy` of `score_pose_set` apply the rule above to the back-ends of the poses (`prediction_models`), `result["preflight"]` gets the `design` block, and the check is left out of the poses of a back-end that the design ran while the other back-ends keep it; `preflight="off"` does not switch this rule off. In a pose set `evobind_adversarial` takes the reference as the design and each pose as the second structure, weighted by its own per-atom pLDDT, so a set whose poses are not predicted poses is refused with a message that says so. [`pose_sets.md`](pose_sets.md) describes the layer.

---

## Declared limits

A limit is declared only where code or the documentation of the model shows it, and its `reasons` sentence names that source. The tests in `tests/test_pre_openfold3_limits.py` and `tests/test_pre_metric_limits.py` pin the behaviour each sentence describes.

### OpenFold3 0.5.0

Declared on `OpenFold3Parser.capabilities`.

| Limit | Basis |
|---|---|
| Closures: `none` and `head_to_tail` | `cyclic: true` on a protein chain wraps the whole chain, so it is head-to-tail (`openfold3/core/utils/relpos.py`). The query schema has `covalent_bonds` and nothing reads it (`openfold3/projects/of3_all_atom/config/inference_query_format.py`), so a disulfide, a lactam, a staple or another cross-link cannot be given. |
| Modes: `predict`, `refold` and `score`, not `score-lock` and not `refold-lock` | A template gives the fold of one chain, not the pose between chains: the template pair features are masked to the same chain (`multichain_pair_mask` in `openfold3/core/data/primitives/featurization/template.py`; `_embed_feats` in `openfold3/core/model/feature_embedders/template_embedders.py`), and a multi-chain CIF template gives one chain (`docs/source/template_how_to.md`). No setting forces a template, so `refold-lock` is refused as well. The pocket constraint is documented for small-molecule ligands only (`docs/source/input_format_reference.md`, section 4); its use for a peptide binder was not confirmed. |
| Residues that the query builder cannot express | `check_openfold3_residues` calls `metrics._openfold_run._residue_letter_and_ccd`, the rule of `_extract_query_chain`, and the reason is the text of the `UnmappableResidueError` that the builder raises for the same chain. D-amino acids, N-methylated and other peptide-linking Chemical Component Dictionary residues are expressible and pass. |

Warnings, not refusals:

- A head-to-tail binder of standard residues is sent with `cyclic: true` by default (`binder_cyclic="auto"`, `--openfold-cyclic`; OpenFold3 0.4.5 or later). The flag wraps the relative positions of the chain; OpenFold3 does not enforce the closure bond and has published no accuracy benchmark for cyclic peptides.
- A head-to-tail binder with modified residues (D-amino acids, N-methylated residues) is left linear by `auto`; `--openfold-cyclic on` forces the flag.
- A `score` or `refold` query whose template is given as an alignment (`--openfold-templates alignment`, not the default) is left without its template when the MSA server is on. The pre-flight does not check this because it depends on what the server returns; the run reports it per chain in `results[...]["templates"]`. The default `--openfold-templates structure` (OpenFold3 0.4.2 or later) keeps the template (see `docs/metrics.md`).
- Terminal capping groups and non-amino-acid groups (ligands, glycans) are left out of the query by `_extract_query_chain`, so the prediction is of the uncapped peptide without them.

MSA policy. A receptor alignment (`--prediction-receptor-msa PATH`, or `receptor_msa` of `OpenFold3Runner.make_request`) is given to the receptor chain as `colabfold_main.a3m` in `main_msa_file_paths`, and the binder keeps the dummy MSA that a run without the server writes. The run needs the server off, because with the server on OpenFold3 replaces the MSA files of every chain (with a warning). OpenFold3 parses an MSA file only when its name without the suffix is a key of `MSASettings.max_seq_counts` (`colabfold_main` is one, `.a3m` or `.sto`), so the runner stages the file under that name, one folder per chain. OpenFold3 0.5.0 takes the first row of a chain's alignment as the query row and places its columns on the residues of the chain by position, without comparing it with the chain (`MsaSampleParserInference.parse_msas` in `core/data/io/sequence/msa.py`, `map_msas_to_tokens` in `core/data/primitives/featurization/msa.py`). The toolkit compares the first sequence of the file (insertions, the lower-case letters, left out) with the receptor sequence before anything is written. An alignment whose only row is shorter than the chain fails in the feature step with `IndexError`.

### Protenix 2.0.0

Declared on `ProtenixParser.capabilities`: no limit, and warnings for a lactam, a staple and another cross-link, and for the mode `score-lock`. The `contact` and `pocket` constraints are soft ("the model is encouraged, but not strictly required, to satisfy it", `docs/infer_json_format.md`, section constraint), so they do not pin the complete pose. No other mode is declared: templates enter only through `templatesPath` (a3m or hhr alignments), so whether `refold` and `score` exist is not shown.

`docs/infer_json_format.md` (commit 85767b8, section `covalent_bonds`) supports a covalent bond between two polymer residues for a head-to-tail amide bond and for a disulfide between cysteines, which stay silent. For other types it says they "can still be specified in the input, but they are not reliably handled by the current model". The source takes any atom pair (`json_to_feature.py`), so these are a caveat and the input runs under the default policy.

MSA policy. `ProtenixRunner.make_request` takes `msa_policy` and `receptor_msa` (a file, or `single`). Protenix has no switch for the server, only `--use_msa`: with `--use_msa true`, a chain without an alignment path in the input JSON is searched on the server, and one such chain sends all the chains of the task (`need_msa_search` and `update_seq_msa` in `runner/msa_search.py`); `--use_msa false` skips the alignment features altogether. The runner therefore gives every protein chain an `unpairedMsaPath` (a copy of the file for the receptor; for the receptor under `single` and for the binder, an a3m with its own sequence; no `pairedMsaPath`) and runs with `--use_msa true`. Under `forbid-server` it reads the JSON it wrote back before it starts the process and accepts `--use_msa true` only when every protein chain names a file that exists. A request with a `receptor_msa` records `use_msa_server=False` whatever was passed, because no server is called. The first record of the file must have as many aligned columns (uppercase letters and `-`) as the receptor has residues (`RawMsa._verify_query`). A first sequence that differs from the receptor at some positions is allowed, for the alignment of a wild type given for a point mutant, and logged as a warning.

Not confirmed by a run (no Protenix installation or weights on the development machine): that a job whose chains all have an `unpairedMsaPath` and no `pairedMsaPath` runs to the end and calls no server, that a one-sequence a3m with an `X` is accepted, and how the species of a header is parsed. The formats were read in the source of commit 85767b8 (`docs/infer_json_format.md`, `runner/msa_search.py`, `protenix/data/msa/msa_featurizer.py`, `msa_utils.py`, `protenix/data/tools/common.py`). All the writing is in `_write_msa_files` of `predictors/protenix_runner.py`, the one function to correct if a run disagrees.

### Boltz-2 2.2.1

Declared on `Boltz2Parser.capabilities`: all five modes, no limit, and three warnings.

- `predict` is the plain input.
- `refold` and `score` need a template per chain, which `templates` takes with `chain_id` (`docs/prediction.md`, Templates). An unforced template carries no pose: the template module lets features attend within the same chain only (`src/boltz/model/modules/trunkv2.py`, "Compute asym mask").
- `score-lock` is a template with `force: true` and a `threshold`. `process_template_features` (`src/boltz/data/feature/featurizerv2.py`) puts all the chains that a template file maps into one row, and `TemplateReferencePotential` (`src/boltz/model/potentials/potentials.py`) aligns that row rigidly over its templated tokens and penalises a deviation larger than the threshold. It is a guidance term with weight 0.1, not a hard constraint, and templates are for protein chains only, so the mode carries a warning.
- `refold-lock` is the same template for the receptor chain alone, with the binder free, and carries the same warning.

No closure is refused: `cyclic: true` wraps a chain head-to-tail, and the `bond` constraint takes any two atoms of the input (`atom_idx_map` in `boltz/data/parse/schema.py`), so a disulfide or a lactam between canonical residues can be given. `Boltz2Runner.make_request(..., closure_bonds="auto")` writes one `constraints: - bond:` entry for each head-to-tail, disulfide and lactam closure that `detect_closures` finds on the binder, at its position in the sequence sent (a modified residue does not shift it). A hydrocarbon staple or another cross-link is not written and logs a warning. A list of bonds is written as given, and the default, `"none"`, writes none. Boltz-2 applies a head-to-tail `bond` as a cyclic period only for the first chain of the input (the featuriser compares the residue indices of the whole structure with 0), and the runner writes the receptor first, so a binder needs `cyclic: true` (#150). The documentation lists the `bond` constraint as supported for "CCD ligands and canonical residues" only (`docs/prediction.md`), which is why a hydrocarbon staple gets a warning and not a refusal.

MSA policy. Boltz-2 takes the receptor alignment as `msa: <file>` on the receptor entity (`.a3m` or `.csv`) and `msa: "empty"` on the peptide, because it refuses a custom MSA next to an automatic one in one input ("Cannot mix custom and auto-generated MSAs in the same input!"). It raises "Missing MSA's in input and --use_msa_server flag not set" for an entity without an `msa` when the flag is absent (`main.py`), and the runner never writes `--use_msa_server` under `forbid-server`. The first sequence of the file is the query of the entity and must be the receptor's: the runner compares them (an `X` or `U` on either side matches anything; lower-case letters are insertions). A receptor and a binder with one sequence would be one Boltz-2 entity with one MSA, so a file is refused for such an input.

### AlphaFold2 and ColabFold

Declared on `AlphaFold2Parser.capabilities`, from the model code (AlphaFold commit c77e5d2, ColabFold 1.6.3). No mode is declared: no verified statement says which modes a sequence-and-template model supports.

| Limit | Basis |
|---|---|
| Closures: `none` | ColabFold builds its features from `make_sequence_features`, `make_msa_features` and the template features (`colabfold/batch.py`, `build_monomer_feature`, one such dictionary per chain for a multimer); none holds a bond, so a ring closure cannot be given and the chain is folded as a linear one. |
| Residue classes: `canonical`, plus `cap` and `ligand` | The residue types are the 20 amino acids and X (`restypes_with_x` in `alphafold/common/residue_constants.py`), and `make_sequence_features` maps any other letter to X (`alphafold/data/pipeline.py`). A D-amino acid, an N-methylated, phosphorylated or other modified residue cannot be given. A capping group or a ligand is not part of a sequence and gives a warning. |

MSA policy. `ColabFoldRunner.make_request` takes `msa_policy` and `receptor_msa`. ColabFold leaves its MSA server only with `--msa-mode single_sequence`, which applies to both chains, so the runner accepts `receptor_msa="single"` and refuses the path of an alignment: the runner gives `colabfold_batch` a FASTA file, whose alignments ColabFold builds itself. ColabFold also reads an a3m file as input (the file it writes for a complex is a block-diagonal a3m headed `#<length A>,<length B>`); the runner does not build that route. Under `forbid-server` the runner checks the command line it built, and any other `--msa-mode`, or none (the default of ColabFold calls the server), is refused before the process starts; `extra_args` may not carry `--msa-mode` or a prefix of it. A request with `receptor_msa="single"` records `use_msa_server=False`. No prediction was run: the installation used for the tests has no AlphaFold2 parameters and no CUDA jaxlib, and its numpy breaks `alphafold/data/msa_pairing.py`.

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

### The rule between a limit and a caveat

A limit refuses the input (policy `error`) and is declared only where the source says the input cannot be given or is not read: OpenFold3 reads `covalent_bonds` nowhere, AlphaFold2 and ColabFold take a sequence of 20 amino acids and X, a metric raises or returns nothing without its receptor. A source that says "not reliably", "may" or "experimental" gives a caveat, a warning under every policy: the Protenix bonds above, the Boltz-2 staple, the OpenFold3 head-to-tail closure sent as a linear chain.

### Considered and not declared

| Candidate | Reason |
|---|---|
| OpenFold3, Boltz-2, Protenix, AlphaFold2 and ColabFold binder size | No source gives a maximum for the binder. The memory figures of the ColabFold FAQ depend on the GPU. The 256-token and 2048-atom cropping limits of Boltz-2 belong to its affinity data set. The 2560 tokens of `protenix-v2` (`runner/inference.py`) are a limit of the whole complex and of one model name, and the profile holds the binder only. |
| OpenFold3 binder of several chains, or of a given binder type | Nothing shows a limit. |
| D-amino acids, N-methyl and phospho residues in OpenFold3, Boltz-2 and Protenix | A modified residue is a CCD code in the query (`modifications` for Boltz-2 and Protenix), and no documentation or benchmark states how these residues are predicted, so no caveat is written. The OpenFold3 cyclic flag is the exception (see the warnings above). |
| Protenix `refold` and `score` | Structures cannot be given as templates (`templatesPath` takes alignments); nothing shows the modes exist. |
| AlphaFold2 and ColabFold modes | A template route exists in ColabFold; whether it gives `refold` or `score` as defined here was not checked. |
| Cyclic-offset forks of AlphaFold2 (ColabDesign, BindCraft) | The same adapter reads their outputs. The closure limit describes AlphaFold2 and ColabFold as released, and `policy="warn"` lets such an output through. |
| `requires_gpu` metrics (`receptor_quality`, `md_implicit`, `structure_interaction_energy`) | The field is a scheduling hint ("on CUDA by default"), not a limit of the input. |
| `receptor_quality` | Runs on a single chain; the receptor argument is optional. |
| `ramachandran`, `omega` | They score the closing bond of a head-to-tail peptide and work on a linear one. No metric of the registry is cyclic-only. |
| `structure_rmsd` | Both structures come from the same run; it is not a reference supplied by the user. |
| Antibody and nanobody metrics, trajectory metrics | None exists in the registry for antibodies; a trajectory metric takes a trajectory and a topology, which the needs vocabulary does not describe. |
| Residue limits of `md_implicit` and `structure_interaction_energy` | Unknown residues go through GAFF2 templates whose success depends on the residue and on `antechamber`. |
| Closure limits of `openfold` and `interface_pae` | They read an existing output that another model may have written; the limit belongs to the predictor that writes it. |
| `compute_evobind_adversarial_from_records` | Not a registry entry, so there is no `MetricSpec` to declare on; its needs are those of `evobind_adversarial`. |
| A pre-flight violation for a receptor MSA that a runner cannot take (a file for ColabFold) | `Capabilities` has no field for what a runner accepts as an MSA. The runner states it (`supports_given_msa`, `supports_single`) and the argument check refuses the option before the plan, as it refuses a mode the runner does not run. |
