# Pose sets

A co-folding model run with five seeds returns five structures of one peptide-receptor complex, and two models return ten. Each structure can be measured on its own, but the questions that decide which design to take sit across the set: how far apart the poses lie, whether two models put the peptide in the same place, which pose can speak for a group, and how many poses stand behind an average. A mean over the poses that happened to succeed answers none of them and hides the failures. The pose-set layer answers them in one call. `score_pose_set` takes the poses of one construct, places each on the receptor of one reference, measures each with the registry metrics that were asked for, compares the poses with one another, summarises them per group (the back-end by default) with the number of poses behind every value, and names a representative pose. The result is one JSON-ready dictionary with a versioned schema.

The layer reports numbers and judges none of them. Every threshold is an option with a documented default, and the code that reads the result decides what a gap or an overlap means.

The names load from `binding_metrics.poses` on first access. Importing the package imports nothing heavy, and biotite is imported when a structure is read. A call with the static metrics imports no simulation package (`openmm`, `simtk`, `mdtraj`, `pdbfixer`) and does not look for one, so it runs on the `static` extra alone. The pose kinds are described in [`poses.md`](poses.md) and the metrics that the layer runs in [`metrics.md`](metrics.md).

## A worked example

`binding_metrics.testing.make_synthetic_poses` builds a pose set without a model. The receptor of the bundled 1YCR complex (MDM2, chain A) stays where it is, the p53 peptide (chain B) is moved as a rigid body, and the pairwise RMSD of the peptides is known in closed form. The default set has two back-ends, `backend_a` and `backend_b`, with three seeds each, and every pose is stored in its own random frame.

```python
from binding_metrics.poses import score_pose_set
from binding_metrics.testing import make_synthetic_poses

synthetic = make_synthetic_poses()
print([pose.id for pose in synthetic.poses])

result = score_pose_set(synthetic.poses, metrics=["interface"])
print(result["binder_chain"], result["target_chain"], result["references"]["implicit_pose_id"])

statistics = result["agreement"]["rmsd"]["ca"]["statistics"]
scopes = {**statistics["within"], "between": statistics["between"]["backend_a|backend_b"]}
for name, entry in scopes.items():
    print(f"{name:10s} {entry['n_pairs']} pairs  min {entry['min']:.2f}  mean {entry['mean']:.2f}  max {entry['max']:.2f}")

print(result["representatives"])
print(f"{synthetic.truth.pair('backend_a-s1', 'backend_a-s2'):.2f}")
```

```text
['backend_a-s1', 'backend_a-s2', 'backend_a-s3', 'backend_b-s1', 'backend_b-s2', 'backend_b-s3']
B A backend_a-s1
backend_a  3 pairs  min 0.40  mean 0.53  max 0.80
backend_b  3 pairs  min 0.30  mean 0.60  max 0.90
between    9 pairs  min 1.20  mean 2.00  max 2.90
{'backend_a': 'backend_a-s2', 'backend_b': 'backend_b-s2', 'all': 'backend_a-s3', 'rule': 'medoid', 'reasons': {}}
0.40
```

No chain and no reference was given. The binder is the smaller protein chain of the first readable pose and the target the larger one, and the pose with the lowest id, `backend_a-s1`, plays the reference: every other pose was superposed on its receptor, so the peptides are compared in one frame. The 15 pairs of peptide RMSD split into 3 inside each back-end and 9 between them. The generator moved the peptide of `backend_a-s2` by 0.40 Å relative to `backend_a-s1`, and the layer measured 0.40 Å from the coordinates alone, after undoing the random frames. Each back-end is represented by its middle pose. The overall representative, `backend_a-s3`, ties with `backend_b-s1` at a mean distance of 1.2 Å to the other five poses, and the lowest id wins a tie.

A pose that fails stays in the count. In the next example one pose has lost its peptide:

```python
from binding_metrics.poses import DockedPose, score_pose_set
from binding_metrics.testing import make_synthetic_poses

synthetic = make_synthetic_poses()
receptor = synthetic.reference.atoms()
receptor = receptor[receptor.chain_id == "A"]  # a pose that lost its peptide
lost = DockedPose.from_atoms(receptor, id="backend_b-s4", program="backend_b")

result = score_pose_set([*synthetic.poses, lost], metrics=["interface"])
print(result.row("backend_b-s4")["status"], "-", result.row("backend_b-s4")["reason"])

entry = result["aggregates"]["backend_b"]["interface.delta_sasa"]
print(entry["n_ok"], entry["n_total"], entry["complete"], entry["reasons"])
print(result["agreement"]["pose_ids"][-1], result["agreement"]["excluded"])
```

```text
failed - binder chain 'B' not in the structure; it has ['A']
3 4 False {"binder chain 'B' not in the structure; it has ['A']": 1}
backend_b-s3 {'backend_b-s4': "binder chain 'B' not in the structure; it has ['A']"}
```

The mean buried area of `backend_b` is taken over three poses, and the entry says so: `n_ok` is 3 and `n_total` is 4. The failed pose is left out of the agreement block with its reason, and the call logs a warning.

## Poses, groups and the pose kinds

The call takes poses of any kind (`PredictedPose`, `DockedPose`, `ExperimentalPose`) in a list. The ids must be unique and the order is irrelevant: rows are sorted by id.

| Kind | `backend` in the row's provenance | Also added |
|---|---|---|
| `predicted` | the model of the adapter (`of3`, `boltz2`, `af2`, `protenix`) | `name`, `seed_index`, `sample` and, when the adapter recorded it, `seed` |
| `docked` | `program`, when the pose names one | |
| `experimental` | none | |

A key that the pose's own provenance already sets is not replaced. A pose read through `PredictionSession.record` sets `backend`, `seed_index`, `sample` and `seed` itself (and `mode`, `receptor_forced`, `receptor_templated`, `request_key`, `model_version` and `request_name`), so `group_by=("backend", "mode")` or `("backend", "receptor_forced")` groups a mixed set of predictions without further work; `seed` there is the value the model sampled with, or None when it cannot be told (the `reasons` of the pose say why). The groups of the result are the values of the provenance keys named in `group_by`, `("backend",)` by default: a group is written `boltz2` for the back-end alone and `boltz2|seed=42` for `group_by=("backend", "seed")`, and a pose without the key falls in `backend=None`. Give a pose its back-end through its provenance (`DockedPose.from_file(path, program="haddock")` or `DockedPose.from_file(path, backend="haddock")`). `group_by=()` leaves the group `all` alone. The names `all`, `rule`, `reasons` and `caveats` are used by the result and cannot be a group.

A pose set holds the poses of one construct: one binder sequence against one target. Two binders that differ in sequence (after the protonation variants and the D-amino-acid codes are mapped to their L parent) are not compared, and the pair has no value with a reason.

## The call

```python
score_pose_set(poses, binder_chain=None, target_chain=None, metrics=(), reference=None, *,
               options=None, aggregate=DEFAULT_STATISTICS, group_by=("backend",),
               representative="medoid", heavy="representative", metric_kwargs=None,
               preflight="error", on_error="record", n_workers=1, tmp_dir=None,
               keep_tmp=False, residue_table=False, include_transform=True,
               construct=None, residue_maps=None, start_method="spawn",
               design_source=None, design_policy="skip")
```

| Argument | Meaning |
|---|---|
| `poses` | The poses, with unique ids. |
| `binder_chain`, `target_chain` | The chain IDs as the user names them, after any `chain_map`. None detects them on the first readable pose in id order: the smaller protein chain is the binder and the larger the target. |
| `metrics` | Registry names (`"interface"`, `"hbonds"`, `"saltbridges"`, ...) run on each pose. The default `()` gives the frame, the native scores and the agreement. A metric that reads a trajectory or a prediction directory is refused with `ValueError`. `"prediction"` is built from a `PredictedPose`. `structure_rmsd`, `dockq` and `evobind_adversarial` compare with a second structure and need `reference`. For `evobind_adversarial` the reference is the design and each pose is the second structure, weighted by its own per-atom pLDDT, so the poses must be predicted poses (`PredictedPose`): the pre-flight check refuses other poses, and with `preflight="warn"` their `evobind_adversarial_score` is None with a `reason` while `delta_com_angstrom` is still given. |
| `reference` | A `Reference`, or `{variant name: Reference}`. None takes the first readable pose as the reference. See [Reference](#reference). |
| `options` | `PoseSetOptions` or a mapping `{group: {option: value}}` with the groups `frame`, `agreement`, `aggregation`, `representative` and `descriptors`; the tables below give every option and its default. |
| `aggregate` | Statistic names (`"mean"`, `"std"`, `"min"`, `"max"`, `"median"`) or a mapping `{name: callable}`. |
| `group_by` | The provenance keys that define the groups. |
| `representative` | `"medoid"`, `"confidence"` or a callable `(rows of the group, PoseDistances) -> id`. See [The representative pose](#the-representative-pose). |
| `heavy` | Where the heavy metrics run: `"representative"`, `"all"` or `"skip"`. |
| `metric_kwargs` | `{metric name: {keyword: value}}`, passed to the metric as it is. It may not set the structure or the chain arguments. |
| `preflight` | `"error"` refuses an input that a requested metric cannot take (`IncompatibleInputError`) before any pose is scored. `"skip"` leaves those metrics out and lists them, `"warn"` logs and runs, `"off"` does not check. The check reads the profile of the first readable pose; see [`preflight.md`](preflight.md). |
| `on_error` | `"record"` or `"raise"`. See [Errors](#errors-and-on_error). |
| `n_workers` | Processes for the per-pose stage; 1 runs in this process. See [Parallelism](#parallelism). |
| `tmp_dir`, `keep_tmp` | An existing directory for the temporary structure files (the system directory by default), and whether to keep the `bm_poseset_*` directory the call makes in it; the kept path is in `provenance["temp_dir"]`. |
| `residue_table` | Adds the per-residue interface table to each row. Off by default because it is long. |
| `include_transform` | Puts the rotation and translation of each frame in the rows. |
| `construct` | A name for the construct, recorded in the result. |
| `residue_maps` | `{pose id: {pose ResidueKey: reference ResidueKey}}`, replacing the residue map of the reference for those poses. |
| `start_method` | How the worker processes of `n_workers` above 1 are started: `"spawn"` (the default), `"forkserver"`, `"fork"`, or None for the platform default. See [Parallelism](#parallelism). |
| `design_source` | The pipeline that made the designed binder: `"boltzgen"`, `"bindcraft"` or `"colabdesign"` in any case, or the registry names of the models it ran itself, `("boltz2",)`. It sets `independent_of_design` in every row and drives `design_policy`; see [The design origin](#the-design-origin). None decides nothing. |
| `design_policy` | `"skip"` (the default), `"warn"` or `"error"`: what to do with a check that rests on independence (`evobind_adversarial`) on the poses of a back-end that the design pipeline ran itself. `preflight` does not change it. |

Which metrics run on which poses depends on the `cost_class` that the registry declares for them. A metric with the class `static` or `model` runs on every pose. A metric with the class `structural` or `md`, and a metric that declares none, is heavy: with `heavy="representative"` it runs on the representative poses only, in this process and one after the other, so that a metric that needs the GPU never runs in a worker. `heavy="all"` runs it on every pose and `"skip"` leaves it out. A pose that was not asked for a metric has no block for it and no entry in `reasons`.

## Reference

A `Reference` is the structure that the poses are placed on and measured against. It is a role that any structure can play: a crystal structure, a receptor that a model was forced to keep, a docked complex.

```python
Reference(structure, receptor_chain, binder_chain=None, residue_map=None,
          residue_sets=None, partners=None, name="reference")
```

| Field | Meaning |
|---|---|
| `structure` | A path (mmCIF or PDB, gzip allowed), a biotite `AtomArray` of one model, or a pose. A bare path or array becomes an `ExperimentalPose`; pass a `DockedPose` or `PredictedPose` to say that the reference is a docked or a predicted complex. |
| `receptor_chain` | The receptor chain of the structure, as the user names it. |
| `binder_chain` | The peptide of the reference complex, or None. With it the frame stage reports the binding site: the receptor residues with a heavy atom within `site_cutoff_angstrom` (4.5) of the peptide. |
| `residue_map` | `{pose ResidueKey: reference ResidueKey}`: the pairing of pose residues with reference residues, for every pose. It is one-to-one, its reference side lies on the receptor chain, and it wins over every automatic pairing. |
| `residue_sets` | `{name: residues}`: named receptor residue sets in the numbering of the reference (an epitope, hotspots, a pocket). A set is not empty and lies on the receptor chain, and its name is not `all` or `site`, which are region names of the frame stage. |
| `partners` | `{name: Partner(structure, chains=None)}`: structures that are already placed in the frame of the reference, such as a known binder on the same receptor. A partner is read as it is and never moved. `chains` names the partner's chains, and None takes every polymer chain. |
| `name` | The name of this variant, the key of every block that is reported per variant. |

A residue is a `ResidueKey(chain, number, icode)`, so 52 and 52A are two residues. Everywhere a key is expected, a tuple `("A", 52)` or `("A", 52, "B")` and a label `"A:52B"` are read as well. A reference is frozen and compares by identity, like a pose; `content_hash()` tells two references with the same content apart from others. `validate()` checks the reference against its structure and lists every problem in one `ValueError`. `to_dict()` and `from_dict()` write and read it: a structure that is a file is recorded by path and content hash and checked when it is read back, and a structure that was in memory has to be passed again. `Reference`, `Partner`, `ResidueKey`, `FrameOptions`, `FrameResult` and `Transform` pickle and deep-copy, so they can go to worker processes; a `Reference` does not carry the cache that `prepare_reference` builds, and the worker rebuilds it. That cache is checked against the content hash of the reference structure, so a reference file rewritten at the same path is prepared again.

```python
from binding_metrics.poses import Partner, Reference

crystal = Reference(
    "data/example_linear_p53_1YCR.pdb", receptor_chain="A", binder_chain="B",
    residue_sets={"core": [("A", n) for n in (51, 54, 62, 72, 93, 96, 100)]},
    partners={"crystal_peptide": Partner("data/example_linear_p53_1YCR.pdb", chains=["B"])},
    name="crystal",
)
```

The residues of `core` are the seven receptor residues of the crystal complex that bury at least 40 Å² (Lys51, Leu54, Met62, Gln72, Val93, His96, Tyr100).

**Several variants.** `reference={"strict": Reference(...), "loose": Reference(...)}` places each pose once per variant, and the first variant defines the frame of the registry metrics and of the agreement. The row's `frame` block describes the first variant and `frame["variants"]` holds one block per variant. A variant with residue sets or partners adds `descriptors[variant]`. Variants may differ in their residue sets, their partners, their residue map or the residues the fit uses. Two variants have the same frame when `Reference.frame_key()` is equal: the structure and its chain renaming, the receptor chain, the residue map and, when the fit is restricted to a set, that set's residues.

**No reference.** The pose with the lowest id that can be read plays the reference, under the variant name `first_pose`, with the receptor chain and binder chain of the call. The result says so: `references["implicit"]` is true and `references["implicit_pose_id"]` names the pose. Without a reference there are no residue sets or partners, and the descriptors are not computed.

## The common frame

Every pose is placed in the frame of the reference receptor before anything is compared or measured. Poses of one construct come from different programs, which store a complex in different frames, and the peptide RMSD between two poses means nothing until their receptors coincide.

For each pose and each reference variant, the stage pairs the receptor residues of the pose with those of the reference, fits a rigid transform on the paired C-alpha atoms, and reports how good the frame is.

**Pairing.** The first method that succeeds is used and named in `frame["pairing"]`.

| Method | Pairs |
|---|---|
| `map` | The residues that a `residue_map` names and that both structures have. With `pairing="auto"` a map wins over the other methods. |
| `number` | Residues with the same number and insertion code and compatible names, when at least `min_pair_fraction` (0.9) of the shorter chain pairs this way. Two names are compatible when they are equal after the protonation variants (HID to HIS), the D-amino-acid codes (DAL to ALA) and the modified residues MSE, SEP, TPO and PTR are replaced by the amino acid they stand for. |
| `sequence` | A global alignment of the one-letter sequences of the two receptor chains (BLOSUM62, affine gap penalty -10 and -1, no penalty for end gaps). The aligned columns with compatible names are paired, which handles trimmed ends, internal gaps, renumbering and insertion codes. When several alignments score the same, the first in biotite's order is used and `alignment_ambiguous` is true, with a warning. |
| `position` | The i-th residue to the i-th, for chains of equal length. It is used only when asked for (`pairing="position"`). |

**Fit.** A Kabsch fit on the C-alpha atoms of the paired residues: a proper rotation, never a reflection, in float64, with no random numbers. `frame_atoms="backbone"` fits the N, CA, C and O atoms that both residues have. `frame_residues` fits only the paired residues of a named residue set of the reference, or of the binding site (`"site"`, which needs a `binder_chain`). `outliers="iqr"` rejects residues, not atoms, in rounds of at most `iqr_max_iterations`: a residue is an outlier when its mean squared deviation exceeds the upper quartile by `iqr_threshold` interquartile ranges. A residue that deviates by less than `iqr_min_deviation_angstrom` is never rejected, and the rejection stops before `min_frame_residues` would be violated. The transform maps a coordinate `x` of the pose to `rotation @ x + translation`, which is `coords @ rotation.T + translation` for an array of row vectors.

**Refusal.** A pose fails, with `status` `failed`, a `reason` and no transform, when its receptor chain is missing, when the sequence identity of the pairing is below `min_sequence_identity`, when fewer than `min_frame_fraction` of the reference receptor residues are paired, when fewer than `min_frame_residues` residues are left to fit, or when the fit atoms of the pose or of the reference are collinear. A pose that fails is a failed row and nothing else is attempted for it. The identity limit does not apply to a `residue_map`, which states the correspondence itself.

Two consequences follow. The paired fraction is taken over the residues of the reference receptor, so a pose that holds one domain of a larger reference fails at the default 0.5 until `min_frame_fraction` is lowered. And the free end gaps let the alignment of two unrelated chains keep a short overlap with a high identity over its few columns, so it is the paired fraction that refuses an unrelated receptor: the trypsin of 3P8F against the MDM2 of 1YCR pairs 2 of 85 residues ("only 2 of the 85 reference receptor residues are paired (0.02); min_frame_fraction is 0.50"), and the identity limit refuses a related chain that has many other residues.

**The `frame` block of a row.** `status`, `reason`, `pairing`, `n_receptor_residues_pose`, `n_receptor_residues_reference`, `n_paired`, `paired_fraction`, `sequence_identity`, `alignment_ambiguous`, `n_fit_residues`, `n_fit_atoms`, `n_rejected`, `receptor_rmsd_angstrom`, `region_rmsd_angstrom`, `region_n_residues`, `region_n_paired`, `region_reasons`, `warnings`, then `transform` (`{"rotation": 3 x 3, "translation": 3}`, when `include_transform` is true), `variant` and, with several variants, `variants`. The region RMSD is the C-alpha RMSD of the paired residues of a region in the final frame, with no refit, so it measures how far the region moved. The regions are `all` (every paired residue), `site` (present when the reference has a binder chain) and every residue set of the reference. A region with no paired residue is None, with its reason in `region_reasons`.

**Options of the frame stage** (`options["frame"]`, class `FrameOptions`).

| Option | Default | Meaning |
|---|---|---|
| `pairing` | `"auto"` | `"auto"` (map, then number, then sequence), or one method. |
| `frame_atoms` | `"ca"` | `"ca"` or `"backbone"`. |
| `frame_residues` | `None` | None fits every paired residue, otherwise a residue set name or `"site"`. |
| `outliers` | `None` | None or `"iqr"`. |
| `min_frame_residues` | `10` | Fewest residues to fit, at least 3. |
| `min_frame_fraction` | `0.5` | Smallest fraction of the reference receptor residues that must be paired. |
| `min_sequence_identity` | `0.8` | Smallest fraction of paired residues with compatible names. |
| `min_pair_fraction` | `0.9` | Share of the shorter chain that must pair by number for the `number` method. |
| `site_cutoff_angstrom` | `4.5` | Heavy-atom distance to the reference binder that puts a receptor residue in the site. |
| `iqr_threshold` | `1.5` | Interquartile ranges above the upper quartile that make an outlier. |
| `iqr_max_iterations` | `10` | Most rounds of rejection. |
| `iqr_min_deviation_angstrom` | `0.01` | A residue that deviates by less is never rejected. |
| `alignment_matrix` | `"BLOSUM62"` | The biotite substitution matrix of the sequence alignment. |
| `alignment_gap_penalty` | `(-10, -1)` | Open and extend. |
| `renumber_receptor` | `True` | Number the paired receptor residues of the structure that the metrics read as the reference does. |
| `metrics_frame` | `"reference"` | `"reference"` runs the registry metrics on the transformed pose, `"original"` on the pose's own file. |

**Metrics in the frame of the reference.** A metric that reads a file gets the transformed structure, written as mmCIF to a temporary directory (`tmp_dir`), one file per pose, removed when the call ends; a metric of the `atom_array` input type receives the transformed array. With `renumber_receptor` the paired receptor residues carry the numbers of the reference, so the labels of the metrics (`interface_residues_receptor` of `interface`) are in one numbering for all poses, and the peptide keeps the numbering of its file. Surface areas depend slightly on the orientation of a structure, and measuring every pose in one frame removes that difference between poses. On the synthetic set above, `delta_sasa` of `backend_a-s2` is 1481.7 Å² in the frame of the reference and 1475.8 Å² in its own random frame (`metrics_frame="original"`), a difference of 0.4 %.

## Agreement between poses

The poses are compared through the peptide, in the frame of the reference receptor. The peptide is not refitted, so the RMSD measures displacement and conformation together, which is what says whether two models put the peptide in the same place. The functions are in `binding_metrics.poses.agreement` and take biotite `AtomArray` objects that are already in the common frame, with the binder chain ID. `score_pose_set` calls them and puts the output in `result["agreement"]`.

**All-pairs RMSD.** For every pair of poses the RMSD of the matched atoms, `sqrt(sum |x_a,i - x_b,i|^2 / m)` in angstrom, an average over atoms. It is computed for three atom sets, and all three are always reported.

| `kind` | Atoms |
|---|---|
| `ca` | The alpha carbon of each residue. |
| `backbone` | N, CA, C and O of each residue (OXT is not part of it). |
| `heavy` | Every non-hydrogen atom of the binder residues, matched by residue and atom name. |

`result["agreement"]["rmsd"][kind]` holds the matrix and its statistics. `shape_rmsd` holds the same for the peptides refitted on the matched atoms (at least 3), which separates the conformation from the position; it is switched off with `shape_rmsd=False`.

**Residue pairing.** Residues of two poses are paired by `ResidueKey`, or by position when the numbering differs (`residue_pairing="auto"`: by key when the keys of one pose are all in the other pose, otherwise by position for equal lengths; `"key"` and `"position"` force one). Two paired residues must be the same amino acid after the protonation variants and the D-amino-acid codes are mapped to their L parent, so a model that flips a D residue to L is measured and a change of sequence is refused. Residues and atoms present in one pose only are dropped from that pair. A pair that shares fewer than `min_shared_atom_fraction` (0.8) of the atoms of the larger set has no value and a reason. Waters and hydrogens are never part of the binder.

**Symmetric atoms.** Atoms that the chemistry cannot tell apart carry names that a model assigns arbitrarily. With `symmetry="standard"` the heavy-atom RMSD compares each residue in the labelling that gives the smallest sum of squared distances. The groups are Asp OD1/OD2, Glu OE1/OE2, Leu CD1/CD2, Val CG1/CG2, Arg NH1/NH2, the ring flip of Phe and Tyr (CD1/CD2 and CE1/CE2 together), and the three phosphate oxygens of Sep, Tpo and Ptr. D residues, protonation variants and N-methylated residues (MLE, MVA, SAR, NMA) use the entry of their L parent. Asn, Gln and His are not swapped. A residue that has no entry and is not known to lack symmetric atoms (BMT, ABA, the lactam templates, a stapling residue) is compared as named, which can only overestimate the RMSD, and its name is listed in `agreement["unknown_residues"]`. `symmetry="off"` compares atoms as named. `ca` and `backbone` do not depend on the labels.

**The free C-terminal carboxylate.** O and OXT of the last residue of a linear binder are equivalent by resonance whatever the residue, and a model that writes them the other way round raises the heavy-atom RMSD by the 2.2 Å between the two oxygens, spread over the atoms that are compared. On the 1YCR peptide (113 heavy atoms) with O and OXT of the last residue exchanged in one pose, the heavy-atom RMSD is 0.29 Å with the names as written and 0.00 Å with the rule. With `symmetry="standard"` and `kind="heavy"` they are a swappable pair when both binders are linear (neither is closed head to tail) and both poses carry O and OXT on the last residue. A pose without OXT, or a ring, leaves O as named. The `backbone` set holds O and not OXT, so the rule does not apply there, and the backbone RMSD of the same pair is 0.30 Å.

**The heavy-atom RMSD is not a metric.** The symmetric-atom labelling is chosen per pair of poses, so the heavy-atom matrix is a minimum over labellings and need not obey the triangle inequality. The clusters and the medoid are still defined on it, and an argument that needs a metric space does not apply to them.

**Cyclic peptides.** No terminus is assumed and no residue is dropped by position. The atoms that close a ring (C of the last residue and N of the first for a head-to-tail amide, the SG atoms of a disulfide, the side-chain atoms of a lactam or a staple) are ordinary atoms of the sets. A head-to-tail ring has no chain break: the pairing never looks at which residue follows which. With `cyclic_shift="auto"` (`"off"` by default), two poses that are both closed head to tail (C of the last residue within 2.0 Å of N of the first), have the same number of residues and the same residue names under the shift, take the smallest RMSD over the cyclic shifts of the pairing; a tie goes to the smallest shift. The ring-closure families of each pose are listed in its `profile.closures`. `expected_closures` (by default the families that most poses have) is compared with each pose in `profile.closures_as_expected`, a pair with different families is listed in `agreement["closure_mismatch_pairs"]`, and `exclude_closure_mismatch=True` gives such a pair no value. The code does not detect closures beyond that distance test, does not try the reversed direction of a ring, and does not apply the shift to the spread or the contact maps.

**Statistics of the matrix.**

| Statistic | Definition | Needs |
|---|---|---|
| `statistics` (`pair_statistics`) | Minimum, mean and maximum over the pairs of the whole set, inside each group and between each two groups, with the pairs that give the extremes. `between_reason` says why there is no between-group entry (`"only one back-end"`). | one pair |
| `clusters` | Complete linkage, written here: start with one cluster per pose, merge the two clusters with the smallest linkage distance (the largest distance between a member of one and a member of the other) and stop when it exceeds the cutoff. Ties go to the pair with the smallest rank of its lowest members, the rank being the position in the sorted ids. Labels follow (size descending, lowest member). One clustering per cutoff, keyed by the cutoff as text (`"1.0"`), with `n_clusters`, `sizes`, `largest_cluster_fraction`, `labels`, `clusters` and, with groups, `mixed_clusters`. | two poses |
| `medoids` | The pose with the smallest mean distance to the other poses, overall and per group, and the RMSD between the medoids of two groups. See [The representative pose](#the-representative-pose). | one pose (it is its own medoid) |
| `contacts.interface_jaccard` | The Jaccard index `\|A ∩ B\| / \|A ∪ B\|` of the interfaces of two poses, for all interface residues, the receptor residues and the binder positions. Two empty sets have no value. | two poses |
| `contacts.contact_map_jaccard` | The same index over the residue pairs in contact. | two poses |
| `spread` | For each binder position, `sqrt(mean_i \|x_i - mean(x)\|^2)` of the alpha carbons over the poses (divisor n), the mean of that over the positions, for all poses and per group. It is always computed on the alpha carbons, whatever `kind` is. | two poses per position |

Two distances closer than 1e-9 Å are equal in every tie rule. An undefined distance counts as infinite in the clustering. The interface of the Jaccard overlap follows a distance rule: a residue is in the interface, and a residue pair is in contact, when two of their heavy atoms are closer than `contact_cutoff_angstrom` (5.0). It is not the buried-area interface of the [descriptors](#reference-site-descriptors). Binder residues are named by their 1-based position in the binder chain and receptor residues by their key in the numbering of the reference. A quantity that cannot be computed is None with a reason, never NaN.

**Options of the agreement stage** (`options["agreement"]`, class `AgreementOptions`).

| Option | Default | Meaning |
|---|---|---|
| `kind` | `"ca"` | The atom set of the matrix that feeds the clusters and the medoids. |
| `cluster_cutoffs_angstrom` | `(1.0, 2.0)` | One clustering per cutoff. |
| `linkage` | `"complete"` | The only linkage; kept so that the result records it. |
| `symmetry` | `"standard"` | `"standard"` or `"off"`. |
| `cyclic_shift` | `"off"` | `"off"` or `"auto"`. |
| `residue_pairing` | `"auto"` | `"auto"`, `"key"` or `"position"`. |
| `expected_closures` | `None` | The closure families that the poses should have, `()` for a linear binder; None takes the families of most poses. |
| `exclude_closure_mismatch` | `False` | Give a pair with different closure families no value. |
| `contact_cutoff_angstrom` | `5.0` | Heavy-atom distance of a contact and of the interface for the Jaccard overlap. |
| `min_shared_atom_fraction` | `0.8` | A pair that shares less has no value. |
| `shape_rmsd` | `True` | Also report the shape RMSD. |

**The block.** `status` (`ok`, `skipped` when no pose could be placed, `failed` when the stage itself raised), `reason`, `kind`, `linkage`, `pose_ids` (the poses that were compared), `groups`, `excluded` (`{id: reason}` for the failed poses and the poses whose binder cannot be read), `expected_closures`, `rmsd` and `shape_rmsd` (`{ca, backbone, heavy: {matrix, statistics}}`), `unknown_residues`, `closure_mismatch_pairs`, `medoids`, `clusters`, `spread` (`all`, `per_group`) and `contacts` (`definition`, `cutoff_angstrom`, `excluded`, `interface_jaccard`, `contact_map_jaccard`).

## Aggregation

Every numeric value in the rows is summarised per group and over all poses, and every summary says how many poses gave a value: `n_ok` next to `n_total`. A pose that failed counts in `n_total` and never in `n_ok`, so a mean over the survivors cannot pass for a mean over the set.

A value of a key is valid when the pose did not fail, no dictionary on the path to the value carries a `reason`, and the value is a finite number (not None, NaN, infinity, a bool or a string). The second condition is `reason_policy="exclude"`, the default: the convention of the registry metrics is a sentinel plus a `reason`, and some of them set it when one step failed, so the keys that the reason covers are not known and every value under that dictionary is dropped. The row keeps its values. `reason_policy="keep"` keeps them and lists the reasons under `caveats`.

A pose counts in `n_total` of a key when it failed, when its row has the block of the key, or when its `reasons` names the block (a metric that raised). A pose that has none of these was not asked for that metric (a heavy metric run on the representatives only) and is not counted.

An entry holds the statistics in the order asked (`mean`, `std`, `min`, `max`, `median`), then:

| Key | Meaning |
|---|---|
| `n_ok`, `n_total` | The valid values and the poses that count. |
| `complete` | `n_ok == n_total`. |
| `reasons` | `{text: count}` of the poses without a valid value, most frequent first. |
| `caveats` | The same for the values that were kept under `reason_policy="keep"`. |
| `statistic_reasons` | `{statistic: text}` for a statistic that is None. |

`std` is the sample standard deviation (`ddof=1`) and needs two values. With fewer than `min_n_ok` valid values every statistic is None and `statistic_reasons` says why. A custom statistic is a function of the sorted valid values that returns a finite number; an exception it raises becomes a None with its text as the reason.

The result is `aggregates[group][key path]`, with the groups in sorted order and `all` last. A key path joins the keys of the row with dots: `interface.delta_sasa`, `native.iptm`, `frame.receptor_rmsd_angstrom`, `descriptors.crystal.partners.crystal_peptide.clash_atoms.2.2` (the dot of the cutoff key stays in the path). The blocks that are not aggregated are `id`, `kind`, `content_hash`, `provenance`, `status`, `reason`, `reasons`, `independent_of_design`, `profile` and `residue_table`. Values are sorted before any statistic is computed, so permuting the rows changes no digit.

**Options of the aggregation stage** (`options["aggregation"]`, class `AggregationOptions`).

| Option | Default | Meaning |
|---|---|---|
| `min_n_ok` | `1` | A key with fewer valid values gives None for every statistic. |
| `reason_policy` | `"exclude"` | `"exclude"` or `"keep"`. |

## The representative pose

One pose speaks for each group and one for the whole set. `result["representatives"]` is `{group: id or None, ..., "all": id or None, "rule": ..., "reasons": {key: text}}`, with `reasons` holding a sentence for every None, and a `caveats` entry for a group where the medoid had to be chosen by the fallback below. A pose with `status` `failed` is never a candidate.

**`"medoid"`** (the default) is the pose with the smallest mean distance to the other poses of its group in the matrix of `agreement.kind`. A pose is a candidate only when its row has no undefined distance to another pose of the group; then the smallest mean is also the smallest sum. A tie within 1e-9 Å goes to the lowest id. When no pose has a complete row, the mean runs over the distances that each pose has, a pose with none is left out, and the result carries a `caveat` that says so. A group of one pose is represented by it. `agreement["medoids"]` and `representatives` use one rule, so the two cannot name different poses.

**`"confidence"`** is the pose with the best native score named by `options["representative"]`: `confidence_key` (default `ranking_score`) and `confidence_direction` (`"max"` or `"min"`, None for the known direction). A larger value is better for `ranking_score`, `iptm`, `ptm`, `avg_plddt`, `chain_ptm.*` and `chain_pair_iptm.*`, and a smaller one for `gpde`, `has_clash` and `disorder`. Another score needs `confidence_direction`. Scores within 1e-9 (relative) tie and the lowest id wins. With several groups the overall pose under `confidence` is None with a reason, because the scores of two models are not on one scale.

**A callable** takes the rows of a group that did not fail, sorted by id, and the `PoseDistances` (or None), and returns the id of one of them. `binding_metrics.poses.rank_by(key, largest=False)` builds one from a function of a row, for instance `rank_by(lambda row: row["interface"]["delta_sasa"], largest=True)`.

The representatives are the poses that the heavy metrics run on.

**Options** (`options["representative"]`, class `RepresentativeOptions`): `confidence_key` (`"ranking_score"`) and `confidence_direction` (`None`).

## Reference-site descriptors

When the reference has residue sets or partners, each pose gets descriptors of where it sits against what is known about the target. They are computed on the pose placed in the frame of that variant, with the numbering that the frame stage read, and they are stored under `descriptors[variant]["sets"][set name]` and `descriptors[variant]["partners"][partner name]`. A block that has no value keeps its keys as None and has a `reason`, which is also recorded in the row's `reasons` as `descriptors.<variant>.sets.<set>` or `descriptors.<variant>.partners.<partner>`. A single column that cannot be computed alone is None and is named in the block's `column_reasons`, and the other columns stay valid.

A residue is in the interface when its buried area is at least `interface_threshold_angstrom2` (0.5). The buried area of an atom is its Shrake-Rupley solvent-accessible area with the chain alone minus its area in the complex (960 points per atom, 1.4 Å probe, clamped at zero), as in `compute_interface_metrics`, and the buried area of a residue is the sum over its heavy atoms. Waters, ligands and hydrogens are not part of the surface. The area of a residue near the threshold can fall on either side of it after a rotation, which is another reason for measuring all poses in one frame.

**Residue-set overlap.** Let S be a named set in the numbering of the reference, P the receptor residues of the pose that the frame pairs with reference residues (a trimmed construct lacks some), and I the residues of P that are in the interface.

| Column | Definition |
|---|---|
| `n_set` | `\|S\|` |
| `n_set_mapped` | `\|S ∩ P\|`: the set residues that the pose has. |
| `n_interface` | `\|I\|` |
| `n_overlap` | `\|I ∩ S\|` |
| `covered_fraction` | `n_overlap / n_set_mapped` |
| `interface_in_set_fraction` | `n_overlap / n_interface` |
| `jaccard` | `n_overlap / \|I ∪ (S ∩ P)\|` |
| `receptor_buried_area_in_set_angstrom2` | The buried area of the receptor residues in `S ∩ P` (the receptor side only). |
| `receptor_buried_area_in_set_fraction` | That area over the total buried area of the receptor. |

A set of which more than `max_unmapped_fraction` (0.5) of the residues are not in the pose has every column but `n_set` and `n_set_mapped` set to None and a `reason` such as "22 set residues, 14 not in the pose (max_unmapped_fraction is 0.5)". The same holds for every set when the frame failed.

**Steric exclusion against a partner.** A partner is a known binder that is already placed on the reference receptor. Let B be the heavy atoms of the pose's binder and Q those of the partner (polymer atoms only, both in the reference frame), `d_ij` the distance of binder atom i to partner atom j, and `r_i` the van der Waals radius of the surface-area code (biotite's table by element, 1.8 Å for an element without an entry).

| Column | Definition |
|---|---|
| `min_distance_angstrom` | The smallest `d_ij`. |
| `clash_atoms` | `{cutoff: number of binder atoms with some d_ij below the cutoff}`, for each cutoff of `clash_cutoffs_angstrom` (2.2 and 3.0), keyed `"2.2"` and `"3.0"`. |
| `clash_pairs` | `{cutoff: number of pairs (i, j) with d_ij below the cutoff}`. |
| `shell_fraction` | The fraction of binder atoms within `shell_angstrom` (4.0) of a partner atom. |
| `shared_buried_area_angstrom2` | `sum_a min(b_pose(a), b_partner(a))` over the receptor atoms a. |
| `partner_buried_area_angstrom2` | `sum_a b_partner(a)` |
| `pose_buried_area_angstrom2` | `sum_a b_pose(a)` |
| `shared_fraction_of_partner` | The shared area over the partner's area. |
| `overlap_volume_angstrom3` | The sum, over the atom pairs with `d_ij < r_i + r_j`, of the volume that the two spheres share. |

`b_X(a)` is the area of receptor atom a that X buries: the area of the receptor alone minus its area with X added, clamped at zero. The sums run over the receptor atoms that the pose and the reference both have (residues paired by the frame, same atom name). `b_pose` is measured on the pose's receptor with the pose's binder, and `b_partner` on the reference's receptor with the partner, so the shared area is the receptor surface that both cover. When the partner is the pose's own binder in its own receptor, the shared area is the whole area of the partner. The overlap volume is a proxy: a triple overlap is counted once per pair, so a crowded site is overestimated, atoms at van der Waals distance contribute nothing, and compression adds volume. A partner that is not a polymer (a ligand) has no atom left after the polymer filter and gets a block without value.

With the crystal peptide of 1YCR as the partner (the `crystal` reference above) and the synthetic poses shifted along a fixed direction, the shared area falls as the peptide leaves the groove:

```python
synthetic = make_synthetic_poses(
    offsets_angstrom={"backend_a": (0.0, 3.0, 6.0), "backend_b": (-3.0, 10.0, 12.0)}
)
result = score_pose_set(synthetic.poses, reference=crystal)
for row in result["poses"]:
    block = row["descriptors"]["crystal"]["partners"]["crystal_peptide"]
```

| Pose | Shift (Å) | `min_distance_angstrom` | `shell_fraction` | `shared_buried_area_angstrom2` | `shared_fraction_of_partner` |
|---|---:|---:|---:|---:|---:|
| `backend_b-s1` | -3 | 0.37 | 1.00 | 378.6 | 0.57 |
| `backend_a-s1` | 0 | 0.00 | 1.00 | 662.3 | 1.00 |
| `backend_a-s2` | 3 | 0.37 | 1.00 | 513.0 | 0.77 |
| `backend_a-s3` | 6 | 0.66 | 0.66 | 371.2 | 0.56 |
| `backend_b-s2` | 10 | 0.80 | 0.20 | 164.9 | 0.25 |
| `backend_b-s3` | 12 | 2.18 | 0.05 | 80.6 | 0.12 |

**The per-residue interface table.** With `residue_table=True` each row gets `residue_table`: the interface residue by residue, in a long format with fixed columns, for the first reference variant. It has one line for every binder residue (a zero area is allowed) and for every receptor residue with a buried area above zero, sorted by role, chain, number and insertion code, so the binder lines come first.

| Column | Meaning |
|---|---|
| `chain`, `role`, `number`, `icode`, `name` | The residue as the pose labels it; `role` is `"binder"` or `"receptor"`. |
| `binder_index` | The 1-based position in the binder chain (None for a receptor residue). |
| `reference_chain`, `reference_number`, `reference_icode` | The reference residue that the frame pairs with a receptor residue (None for a binder residue and for a receptor residue without a pair). |
| `buried_area_angstrom2` | The buried area of the residue. |
| `polar_area_angstrom2`, `apolar_area_angstrom2` | The part of it on N and O atoms, and on C and S atoms. |
| `solvation_dg_kcal_mol` | The solvation term of the residue (Eisenberg and McLachlan, 1986), negative when burial is favourable. |
| `in_interface` | Whether the buried area is at least `interface_threshold_angstrom2`. It does not remove lines. |
| `in_sets` | For a receptor residue, the sorted names of the residue sets that contain its reference residue (an empty list when there are none); None for a binder residue. |

There is no per-residue interaction energy. The only energy-like column is the solvation term, which is a surface-area model.

**Options of the descriptors** (`options["descriptors"]`, class `DescriptorOptions`).

| Option | Default | Meaning |
|---|---|---|
| `interface_threshold_angstrom2` | `0.5` | The smallest buried area that puts a residue in the interface. |
| `clash_cutoffs_angstrom` | `(2.2, 3.0)` | The distances of the clash counts. |
| `shell_angstrom` | `4.0` | The distance of the shell. |
| `max_unmapped_fraction` | `0.5` | The largest fraction of a set that the pose may lack. |
| `probe_radius` | `1.4` | The radius of the solvent probe in angstrom. |

## Reconciling two back-ends

When two back-ends predict the same constructs, the next questions are how far their means differ, whether they hold the same interface, whether they rank a population of constructs in the same order, and whether the residues that an alanine scan finds sensitive are the residues a pose buries. The functions of `binding_metrics.poses.reconcile` (also importable from `binding_metrics.poses`) answer them from what the layer already produced: the aggregates of `aggregate_poses`, the per-pose rows with a `residue_table`, or a table with one row per construct and back-end. Nothing is read from disk and nothing is judged: thresholds are arguments, and a value that cannot be computed is None with a reason.

**Gap and standardised gap.** `gap_between_backends(aggregates, "boltz2", "of3")` returns, per key path, `gap = mean_a - mean_b` (positive when the first back-end has the larger mean: the order of the call is the sign) and `standardised_gap = gap / s_pooled`, with `s_pooled = sqrt(((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2))`, where `s` is the sample standard deviation over the valid values of each back-end (the denominator of Cohen's d). The standardised gap is None with a reason unless both back-ends have at least two valid values and `s_pooled` is above 0. With three seeds per back-end the pooled value has 4 degrees of freedom and is unstable, so `scale={key: value}` replaces it with a scale of the caller's choosing, for example the spread of the same key over constructs, and then needs neither two seeds nor a spread. The gap is None with a reason when either back-end has no valid value, or when only one back-end is in the aggregates (a set with a single model). An entry holds `gap`, `standardised_gap`, `mean_a`, `mean_b`, `n_ok_a`, `n_ok_b`, `scale`, `scale_kind` (`"pooled"` or `"given"`) and `reasons` (`{field: sentence}`, empty when everything was computed). On made-up values, three per back-end (1260.2, 1290.5 and 1320.8 against 1197.9, 1284.0 and 1370.1, with sample standard deviations 30.3 and 86.1), the gap is +6.5, the pooled deviation 64.5 and the standardised gap 0.10.

**Consensus interface.** `consensus_interface(rows, "boltz2", min_fraction=0.5)` collects, over the valid poses of a back-end (not failed, with a `residue_table`), the receptor residues (reference keys; a residue that the pose cannot map to the reference is left out and counted in `n_unmapped_receptor`) and the binder positions that are in the interface, and keeps those that at least `min_fraction` of the poses hold: `count / n_poses >= min_fraction`, so exactly half is in at 0.5. `compare_consensus_interfaces(a, b, limit=None)` returns the Jaccard index of the two consensus sets (`jaccard`, `jaccard_binder`, `jaccard_receptor`), `only_a` and `only_b` (`{"binder": [...], "receptor": [ResidueKey, ...]}`) and, when `limit` is given, `below_limit` (true when `jaccard < limit`). It never says "differs" without a limit. Two empty consensus sets have no Jaccard index. A back-end with no valid pose gives None everywhere with the reason, so that an empty consensus is not read as "no interface".

**Rank agreement over a population.** `rank_agreement(table, "boltz2", "of3", metrics=None, method="spearman")` takes rows `{"construct": ..., "backend": ..., "<metric>": value}`, one per construct and back-end, and returns per metric `method`, `correlation`, `n` (the constructs with a valid value for both back-ends), `n_constructs`, `n_dropped`, `n_missing_a`, `n_missing_b` and, when the correlation is None, a `reason` (fewer than 3 constructs, or all the values of one back-end tie). Spearman's rho is the Pearson correlation of the midranks, and `method="kendall"` gives tau-b (ties in either list corrected). Only exact ties count as ties. A construct with a missing value for either back-end is dropped and counted. Both agree with `scipy.stats` to 1e-15 on seeded data with ties; scipy is not imported by the library.

**Hotspot consistency.** `hotspot_consistency(sensitivities, residue_table, top_m=3)` takes an alanine scan `{binder position: sensitivity}` (larger means more important; pass the negated scale otherwise) and the per-residue table of one pose, and returns `spearman_area_vs_sensitivity` (the buried area against the sensitivity over the positions that have both; it needs 3 positions and a spread in both), `top_in_interface_fraction` (the fraction of the `top_m` most sensitive positions that are in the interface; it needs at least `top_m` positions), `top_positions`, `n_top`, `n_top_in_interface`, `n_positions`, `n_missing_sensitivity`, `n_unmatched_positions` and `reasons`. Every position whose sensitivity is at least that of the `top_m`-th most sensitive one is in the top set, so the set has more than `top_m` members when a tie straddles the cut, and the fraction is taken over the whole set. The default `top_m` of 3 is a choice, not a benchmark. The caller runs it on one pose's `residue_table`; `score_pose_set` has no argument for the sensitivities.

## The design origin

A model that took part in the design agrees with the design partly because the design was selected for that model. When BoltzGen made a construct, the Boltz-2 poses of the construct are therefore not an independent witness, and an OpenFold3 pose of the same construct is. `design_source` names the pipeline (`capabilities.DESIGN_SOURCE_PREDICTORS` maps BoltzGen to `boltz2` and BindCraft and ColabDesign to `af2`; the models can also be named directly) and the layer compares it with the `backend` of each pose. It reports the label and never drops a pose for it.

- `independent_of_design` in each row is False for a pose whose back-end the pipeline ran, True for a pose with another back-end, and None when no usable source was given or the pose names no back-end.
- The checks that rest on independence (`capabilities.INDEPENDENCE_DEPENDENT_CHECKS`, today `evobind_adversarial`) follow `design_policy`. With `"skip"`, the default, the check is not run on the poses of such a back-end: their rows hold the reason under the metric's name in `reasons` (status `partial`), their group counts them in `n_total` and not in `n_ok` of the keys of that metric, and the other back-ends are measured as asked. When every back-end is one the design ran, the check is left out of the whole set and `preflight["metrics_skipped"]` lists it. `"warn"` runs the check on all the poses. `"error"` raises `IncompatibleInputError` before a pose is scored.
- `warnings` carries the sentence of the check (which check, why, and how to run it anyway) and one sentence that names the back-ends whose poses are not independent, with the number of poses; `summary()` prints both. `preflight` has a `design` block (see [`preflight.md`](preflight.md#the-design-origin)) and `parameters["design"]` holds `source`, `predictors` and `policy`, so that a result with another source has another `input_key`.

The rule holds whatever `preflight` says: `preflight="off"` skips the limits of the metrics and not the design origin. The aggregates have no column for the label; a group of a back-end that is not independent shows it in the `reasons` of the keys of a skipped check, and in the label of its rows.

## The result

`score_pose_set` returns a `PoseSetResult`, a `dict` subclass: `result["poses"]`, `json.dumps(result)` and `copy.deepcopy` work as they do for a dictionary, and the methods read it and never change it.

| Member | Meaning |
|---|---|
| `schema_version`, `rows`, `input_key` | Properties for `result["schema_version"]`, `result["poses"]` and `result["input_key"]`. |
| `row(id)` | The row of one pose (`KeyError` listing the ids for an unknown one). |
| `to_dict()` | A deep copy made of plain `dict`, `list` and scalar values. |
| `to_json(indent=2, include_timing=True)` | The canonical JSON text, keys in the order of the result. `NaN` and `Infinity` are kept as tokens for the values that a registry metric returned as such; `nonfinite_fields` lists them. |
| `from_dict(data)` | The inverse of `to_dict` and of `json.loads(result.to_json())`. It refuses a `schema_version` it does not know. |
| `summary()` | A Markdown summary. |
| `to_flat_rows(max_list_items=16)` | One flat row of scalar columns per pose. |

**Top-level keys**, in this order.

| Key | Content |
|---|---|
| `schema` | `"binding_metrics.pose_set_result"`. |
| `schema_version` | `1` |
| `package_version` | The version of binding-metrics that made the result. |
| `construct` | The label given to the call, or None. |
| `binder_chain`, `target_chain` | The chains that were used, given or detected. |
| `input_key` | The SHA-256 of the pose ids, content hashes and provenance, the chains, the content hash of each reference and the parameters. Equal for equal inputs, whatever the run: it does not depend on the package version, `n_workers` or the temporary directory. A callable is not in it. |
| `provenance` | `collect_provenance()` (package version, git sha, Python, OS, OpenMM version read from the installed distribution without importing it, platform, seed) plus `pose_set_schema_version`, `n_poses` and `run` (`n_workers`, `start_method`, `on_error`, `tmp_dir`, `keep_tmp`), and `temp_dir` when the directory was kept. |
| `parameters` | `options` (every option group), `metrics`, `metric_scope` (`all`, `representatives` or `none` for each metric), `metric_kwargs`, `aggregate`, `group_by`, `representative`, `heavy`, `preflight`, `residue_maps`, `residue_table`, `include_transform`, and `design` (`source`, `predictors`, `policy`) when `design_source` is given. The resolved values, so that a result says what produced it. |
| `references` | `implicit`, `implicit_pose_id`, `metrics_frame_variant` and `variants`: for each variant the `content_hash` and the `Reference.to_dict()` record. |
| `preflight` | The `PreflightReport` as a dictionary, or None with `preflight="off"`. It has one more block, `design`, when `design_source` is given. |
| `poses` | The rows, sorted by id. |
| `agreement` | The block described in [Agreement between poses](#agreement-between-poses). |
| `aggregates` | `{group: {key path: entry}}`, see [Aggregation](#aggregation). |
| `representatives` | See [The representative pose](#the-representative-pose). |
| `warnings` | Strings: the warnings of the pre-flight check (with a design source, the check that was left out and the back-ends whose poses are not independent), of each frame (prefixed by the pose id), a binder profile that differs from the first pose's, a group without representative, and a medoid that was chosen with a caveat. |
| `nonfinite_fields` | The key paths of every NaN or infinite number in the result. |
| `timing` | `stages_seconds` and `poses_seconds`. Not part of `input_key`. |

**`schema_version`** changes when a key is renamed or removed, or when a meaning or a unit changes. Adding a key or a statistic does not change it, so a reader should ignore keys it does not know. It is independent of the `schema_version` inside `provenance`, which versions the provenance block alone.

**The row of a pose**, keys in this order.

| Key | Content |
|---|---|
| `id`, `kind` | The pose's id and kind. |
| `content_hash` | The pose's content hash, None when the structure could not be hashed. |
| `provenance` | The pose's own provenance plus the group keys of its kind (see [Poses, groups and the pose kinds](#poses-groups-and-the-pose-kinds)). |
| `native` | `native_scores()` of the pose: what the tool that made it reported. |
| `status` | `ok`, `partial` or `failed`. |
| `reason` | The sentence that explains a `failed` status, else None. |
| `reasons` | `{stage or metric: sentence}` for everything that is missing. |
| `independent_of_design` | False for a pose whose back-end is a model that the design pipeline ran itself, True for a pose with another back-end, None when `design_source` says nothing usable (absent, or a name outside the table) or the pose names no back-end. See [The design origin](#the-design-origin). |
| `profile` | The binder profile read by the pre-flight machinery: `n_binder_residues`, `binder_type`, `closures`, `closure_bonds`, `residue_classes`, `residue_names` (the non-canonical classes), `notes`, then `closures_as_expected` and `differs_from_first` (the fields in which the profile differs from the first pose's). |
| `frame` | The block described in [The common frame](#the-common-frame). |
| one block per metric | In the order requested, under the registry name, exactly as the function returned it (numpy values become lists and numbers). |
| `descriptors` | `{variant: {"sets": {...}, "partners": {...}}}`, for the variants that have residue sets or partners. |
| `residue_table` | The per-residue table, with `residue_table=True`. |

`status` is `failed` when the structure could not be read, a chain is missing, or the pose could not be placed on the reference: the stage is the key of `reasons` (`load`, `chains`, `frame`) and nothing else is attempted. It is `partial` when `reasons` has an entry for anything else: `profile`, `agreement` (a binder that cannot be read), `frame:<variant>` (a variant other than the first), a metric name (the metric raised, was left out by the pre-flight check, does not apply to the kind of pose, or returned a `reason`), `descriptors.<variant>` and `descriptors.<variant>.sets.<set>`, `residue_table`. Otherwise it is `ok`. A metric that raised has no block and the sentence `"<ExceptionType>: <message>"`.

**Keys of the nested blocks**, each in the order the result writes them.

| Block | Keys |
|---|---|
| `preflight` | `policy`, `predictor_policy`, `mode`, `compatible`, `profile`, `metrics_requested`, `metrics_to_run`, `metrics_skipped`, `predictors`, `predictor_usable`, `violations`, `warnings`, `notes` |
| `references.variants[name]` | `content_hash`, `reference` |
| `references.variants[name].reference` | `schema`, `schema_version`, `name`, `structure`, `receptor_chain`, `binder_chain`, `residue_map`, `residue_sets`, `partners` |
| `agreement.rmsd[kind]` | `matrix`, `statistics` |
| `agreement.rmsd[kind].matrix` | `name`, `unit`, `direction`, `pose_ids`, `matrix`, `undefined_pairs` |
| `agreement.rmsd[kind].statistics` | `n_poses`, `overall`, `within`, `between`, `between_reason` |
| `overall`, `within[group]`, `between[a\|b]` | `n_pairs`, `n_defined`, `min`, `mean`, `max`, `min_pair`, `max_pair` |
| `agreement.medoids` | `overall`, `per_group`, `between_groups` |
| a medoid (`overall`, `per_group[group]`) | `pose_id`, `sum_distance_angstrom`, `mean_distance_angstrom`, `tied_with`, `n_poses`, `excluded` |
| `agreement.medoids.between_groups[a\|b]` | `medoid_a`, `medoid_b`, `distance_angstrom` |
| `agreement.clusters[cutoff]` | `cutoff_angstrom`, `linkage`, `n_clusters`, `sizes`, `largest_cluster_fraction`, `labels`, `clusters`, `mixed_clusters`, `n_undefined_pairs` |
| `agreement.spread` | `all`, `per_group` |
| `agreement.spread.all`, `agreement.spread.per_group[group]` | `pose_ids`, `reference_id`, `positions`, `mean_spread_angstrom`, `excluded` |
| a position of the spread | `binder_index`, `chain`, `number`, `icode`, `residue`, `n_poses`, `spread_angstrom` |
| `agreement.contacts` | `definition`, `cutoff_angstrom`, `excluded`, `interface_jaccard`, `contact_map_jaccard` |
| `agreement.contacts.interface_jaccard` | `all`, `receptor`, `binder` |

A matrix is `{"name": "rmsd_ca", "unit": "angstrom", "direction": "distance", ...}` with `matrix` as rows in the order of `pose_ids`; the Jaccard matrices have the direction `similarity`. A medoid also carries a `reason` when it has no distance to report (one pose, or no candidate) and a `caveat` when it was chosen by the fallback of the [representative rule](#the-representative-pose). Each entry of `agreement.contacts.interface_jaccard` and `agreement.contacts.contact_map_jaccard` has the keys `matrix` and `statistics` of a peptide RMSD.

**A metric block next to the reserved names.** A metric is requested by its registry name and its block sits under that name in the row, so a metric whose name is one of the row's own keys is refused (`ValueError`, before any pose is read). The reserved names are `id`, `kind`, `content_hash`, `provenance`, `native`, `status`, `reason`, `reasons`, `independent_of_design`, `profile`, `frame`, `descriptors` and `residue_table`. A reader of the result finds a metric at `row[metric_name]` and a structured value at `row[metric_name][key]`, as `compute_interface_metrics` returns it.

## Writers

`binding_metrics.poses.write_pose_report(result, fmt="json", path=None, *, include_timing=True, max_list_items=16)` writes one file from a `PoseSetResult` (or the dictionary that `json.loads` gives back) and returns its path. Missing parent directories are created and an existing file is replaced. Every writer is a view of the result: none of them changes it, and the same result gives the same bytes. `fmt` is one of:

| `fmt` | Content |
|---|---|
| `json` | The canonical dictionary, UTF-8, `to_json(include_timing=...)`. A NaN or infinity that a registry metric returned is written as the token `NaN` or `Infinity`, and `nonfinite_fields` lists where they are. `include_timing=False` leaves the `timing` block out. |
| `csv` | One row per pose with scalar columns only, from `to_flat_rows`. See the rule below. |
| `md` | `summary()`: a table per group of poses with the status, the receptor RMSD of the frame and the headline value of each metric; the failed poses and the poses with a missing value, each with its reason; the agreement block; the aggregates of the headline values; the representatives; the warnings. |
| `html` | The same content as one page with inline styles, no script, no link, and no font or image from a network. Every text is escaped. |

```text
# Pose set

6 poses in 2 groups: 6 ok, 0 partial, 0 failed. Binder chain B, target chain A. Reference: the first readable pose (backend_a-s1).
```

These are the first lines of `result.summary()` for the synthetic set of the worked example.

**The flattening rule of the CSV.** The columns are the keys of the row joined with dots (`interface.delta_sasa`, `native.iptm`, `frame.receptor_rmsd_angstrom`). A key is never changed or escaped, so a key that contains a dot, such as the clash cutoffs `"2.2"` and `"3.0"` of the partner descriptors, gives a column with that dot: `descriptors.crystal.partners.crystal_peptide.clash_atoms.2.2`. The column names of the numeric values are the key paths of `aggregates`, which are joined in the same way, so a column and an aggregate have one name. A column name does not tell a nested key from a key with a dot; the JSON does. A list of scalars, or of lists of scalars, with at most `max_list_items` items in all is written as one column per item (`frame.transform.translation[0]`, `frame.transform.rotation[1][2]`). A longer list and a list of dictionaries (the `residue_table`, a per-atom array) is left out and stays in the JSON. The first columns are `id`, `kind`, `status`, `reason` and `content_hash`, and the others follow in the order of their first occurrence over the poses in id order. Every row has every column. A missing value is an empty cell, NaN is `nan`, a float is written with `repr` so that it reads back to the same number, and the file is UTF-8 with one `\n` after each line.

## Errors and `on_error`

Failures come in three kinds.

- **A wrong argument raises before any pose is read**: `ValueError`, `TypeError` or `KeyError` for an unknown metric or option, a repeated pose id, a reserved metric name or an invalid `Reference` (one error that lists every problem); `ImportError` for a metric whose extra is not installed; `IncompatibleInputError` when the pre-flight check refuses the input.
- **A pose that cannot be scored is a row with a reason** under `on_error="record"` (the default): a structure that cannot be read, a missing chain, a pose that cannot be placed on the reference. It counts in `n_total` and never in `n_ok`, and a warning is logged. A metric that raises gives a `partial` row with the sentence in `reasons`.
- **`on_error="raise"` stops at the first failure.** The first pose in id order whose row would be `failed`, or whose metric raised, raises `PoseScoringError` (with `pose_id` and `stage`, chained to the original exception when there is one), and the poses after it are not processed. When the poses ran in worker processes the exception object stays in the worker: the error has no chained cause and its message holds the text.

A metric that returns a `reason` is not an exception in either mode. When no pose can be read, every row is `failed`, the agreement block has the status `skipped`, the representatives are None with a reason each, and the aggregates are empty.

## Determinism

No random numbers are used. The rows are sorted by id, so the order of `poses` changes nothing, and `n_workers` does not change a value. `to_json(include_timing=False)` is byte-identical for the same inputs on one installation; `provenance` records the versions and `timing` the run, and neither is part of `input_key`. A call changes none of its inputs (the poses, the reference, the options) and writes only the temporary structure files, in `tmp_dir` or the system temporary directory, which it removes at the end unless `keep_tmp` is set. The synthetic poses are deterministic as well: the only randomness of `make_synthetic_poses` is the frame noise, seeded by `rng_seed`.

## Checking a change against the golden results

A source checkout has two golden results in `tests/golden/pose_sets/`: `synthetic_1ycr.json`, the `score_pose_set` result for `make_synthetic_poses()` against the unmoved 1YCR complex, and `descriptors_1ycr.json`, the crystal peptide of 1YCR and the same peptide shifted 6 Å. Both use an epitope of 22 receptor residues (the residues within 4.5 Å of the peptide) and a buried-area interface of 25 residues as residue sets, and the crystal peptide as partner. `tests/poses/test_golden_pose_set.py` rebuilds both results and compares them with the files. The files hold no `timing`, no top-level `provenance`, no `package_version` and no file path. A float agrees when `|live - golden| <= 1e-6 + 1e-6 * |golden|`; integers, booleans, strings, `None`, the keys and the length of every list must be equal, and a count that became a float is a failure. A failure names the path, the two values and the tolerance: `<path>: golden <value>, result <value> (difference <d>, relative <r>; tolerance rtol=1e-06 atol=1e-06)`.

The values come from the code, and `tests/poses/test_golden_descriptors.py` checks the ones that matter most against a calculation from the PDB file that does not use the layer: the set overlap, the buried area shared with the partner, the minimum distance, the clash and shell counts and the overlap volume, for the crystal peptide and for the shifted one. The pairwise RMSD in `synthetic_1ycr.json` is compared with its closed form (`SyntheticTruth.rmsd`) to 1e-4 Å, the accuracy of float32 coordinates.

`python scripts/regen_pose_golden.py --write` rewrites the files and prints what moved. It refuses to run without `--write`, and it leaves a file alone when every value agrees within the tolerance, so that regenerating on another machine does not rewrite rounding noise. A regeneration is legitimate when the new numbers are intended and understood: a deliberate change of the pose-set code or of a registry metric that the cases call, of the synthetic generator, or a dependency whose algorithm changed (biotite surface areas, for example), after the new values were checked against an independent calculation. The commit that carries the new files states the reason as a regression note: one named value, before and after.

The golden files are not installed with the package. An engine that wants the same check writes `make_synthetic_poses()` into its own fixture, calls its own `score_pose_set`, and compares with the rule above; the 1YCR complex that the generator reads is part of the installed package (`binding_metrics/testing/data/`).

## Parallelism

`n_workers` above 1 scores the poses in worker processes, up to one per pose. The agreement, the representatives and the aggregates run in the calling process, and so do the heavy metrics, one after the other. The result does not depend on `n_workers`.

The workers are started with `start_method`. The default is `"spawn"`: each worker is a fresh interpreter, which is safe in a process that has threads or has loaded GPU libraries, and costs a few seconds of start-up per worker. `"forkserver"` starts the workers from a server process, `"fork"` copies the calling process (the quickest start, but forking a process that has threads is deprecated since Python 3.12 and can deadlock), and None takes the start method of the platform: `fork` on Linux up to Python 3.13, `forkserver` on Linux from Python 3.14, `spawn` on macOS and Windows. An unknown name is a `ValueError`, and so is a method that the platform does not have when `n_workers` is above 1. The method that was used is in `provenance["run"]["start_method"]`; it does not change a value and is not part of `input_key`. The call's context and the poses are pickled to each worker and the rows and the placed atoms are pickled back, so a pose class of your own has to pickle; `Reference`, `Partner`, `FrameResult`, the declared limits of the registry metrics (`Capabilities`, `InputProfile`, `PreflightReport` and every registered `MetricSpec`) and the pose kinds of the package do. With `spawn` (the default) or `forkserver` the script that makes the call needs the usual `if __name__ == "__main__":` guard. When no worker process can be started (the executor cannot be made, a process cannot be created, or the pool breaks before the first pose came back), the call logs a warning and scores the poses one after the other. A pool that breaks after poses came back raises `BrokenProcessPool`. One pose, or `n_workers=1`, never starts a pool.

## What the layer does not cover

- **No per-pose hotspot call.** `hotspot_consistency` takes the sensitivities and one pose's residue table, and the caller runs it. The result has no `hotspots` key.
- **The design origin rests on a table.** `DESIGN_SOURCE_PREDICTORS` maps three pipelines to the models they ran, and `evobind_adversarial` is the only check that follows the rule. A pipeline outside the table needs `design_source=("model",)`. The confidence metrics of a model that made the design are reported with the label and are not left out. The layer compares back-end names and cannot tell that a pose really came from the model it names.
- **No per-residue energy.** The residue table has a solvation term, which is a surface-area model.
- **No real model output was used to test the layer.** The repository holds no output of a real co-folding model. The layer was tested on the synthetic set, on the bundled complexes of `data/` and on the fixtures of the adapters (the Boltz-2 case is the adapter fixture with the 1YCR atoms). The heavy metrics that need OpenMM were tested with stand-ins that are registered under test names, not run. The worker pool was tested on Linux with the `fork`, `spawn` and `forkserver` start methods; macOS and Windows were not tested.
- **One binder chain and one target chain.** A binder of several chains is not supported.
- **Ring closures are read from the structure.** The agreement does not detect closures beyond the distance test of `cyclic_shift="auto"`, does not try the reversed direction of a ring, and does not apply the cyclic shift to the spread or the contact maps.
- **No significance.** The correlations of `rank_agreement` come without p-values or confidence intervals, and the gap is expressed in units of a spread, not tested.
- **Ligand partners.** A partner that is a ligand has no polymer atom and gets a block without value.

## Using it from the engine side

A design engine that scores each construct with two models reads the output directories of the models through the adapters, builds one `Reference` for the target, and keeps the result of each construct as a JSON file. The example runs on the synthetic poses; with real output the lines that build `poses` are the adapters' `load`.

```python
from pathlib import Path

from binding_metrics.poses import (
    Reference,
    compare_consensus_interfaces,
    consensus_interface,
    gap_between_backends,
    score_pose_set,
    write_pose_report,
)
from binding_metrics.testing import bundled_complex_path, make_synthetic_poses

# With real output: parser = get_parser("boltz2"); poses += [parser.load(out_dir, name,
#     seed_index=s.seed_index, sample=s.sample) for s in parser.list_samples(out_dir, name)]
poses = make_synthetic_poses().poses
reference = Reference(bundled_complex_path(), receptor_chain="A", binder_chain="B")

result = score_pose_set(
    poses, "B", "A", metrics=["interface", "hbonds"], reference=reference,
    residue_table=True, construct="design_0042", n_workers=2,
)
write_pose_report(result, "json", Path("results") / "design_0042.json")

gap = gap_between_backends(result["aggregates"], "backend_a", "backend_b", keys=["interface.delta_sasa"])
a, b = (consensus_interface(result["poses"], name) for name in ("backend_a", "backend_b"))
overlap = compare_consensus_interfaces(a, b, limit=0.8)

entry = result["aggregates"]["all"]["interface.delta_sasa"]
print(result["construct"], entry["n_ok"], entry["complete"], overlap["below_limit"])
```

```text
design_0042 6 True False
```

The engine decides what a standardised gap or a Jaccard index of the consensus interfaces means. It reads `complete` before it trusts a mean, `representatives` for the pose to take forward, and `result["poses"]` for the reason of every pose that dropped out.
