# Poses

A pose is one structure of a complex, classified by where it came from. A metric measures a structure, and the origin decides what else is known about it: a co-folding model reports confidences, a docking program reports its own scores, a deposition reports a resolution. The three kinds share one interface, so code that needs a structure and a name for it takes any of them. The poses of one construct, from several models, seeds or programs, are scored together by `score_pose_set`, which [`pose_sets.md`](pose_sets.md) describes.

```
Pose                    abstract base class           binding_metrics.poses
 +-- PredictedPose      output of a co-folding model  binding_metrics.predictors.record
 +-- DockedPose         output of a docking or design tool
 +-- ExperimentalPose   crystal, NMR or EM structure
```

`Pose`, `DockedPose`, `ExperimentalPose` and `PredictedPose` load from `binding_metrics.poses` on first access. Importing the package imports nothing heavy (numpy and the standard library when a class is used), and biotite is imported when a structure is read. `PredictedPose` is the record of a prediction sample that the predictor adapters return, the `PredictionRecord` of earlier versions (see [Renamed](#renamed-predictionrecord)); its fields are in [`metrics.md`](metrics.md#prediction-records-binding_metricspredictors).

## What every pose has

| Member | Meaning |
|---|---|
| `kind` | Class attribute: `"predicted"`, `"docked"` or `"experimental"`. |
| `id` | A stable string. The id the caller gave wins. Otherwise a predicted pose is `<model>:<name>:s<seed_index>:<sample>` (`of3:1ycr:s1:2`), and a docked or experimental pose is the file stem (the PDB ID, else `"docked"` or `"experimental"`, for a structure in memory) and the first 8 hex digits of `content_hash()`: `decoy_01-3f9a1c2e`. The default of a docked or experimental pose reads the file for its hash. |
| `structure_path` | The structure file, or None for a structure in memory. |
| `atoms()` | The structure as a biotite `AtomArray` with the chain IDs of `chain_map` applied; model 1 with author IDs, gzip-compressed files included, cached with the size and modification time of the file and read again when either changes (a pose is a view of the file as it is when read). Treat it as read-only. A structure in memory is returned as it is when `chain_map` is empty and as a renamed copy otherwise; the array that was given is never modified. |
| `chain_map` | Chain ID of the structure to the user's chain ID; empty means no renaming. A swap such as `{"A": "B", "B": "A"}` is applied at once. A map that names a chain the structure lacks, or that would merge two chains, raises `ValueError` when the structure is read. |
| `provenance` | Free facts about the pose (a protocol, a run ID, the request `mode`): a dictionary of JSON scalars (`str`, `int`, `float`, `bool`, `None`; a NaN or infinite float is refused). |
| `content_hash()` | SHA-256 of the bytes of the file for a path (a compressed file as stored), and of a canonical serialisation of the atoms for a structure in memory: chain ID, residue number, insertion code, residue name, atom name, element and the coordinates rounded to 0.001 Å as int64, in array order. The chain IDs are those of the source, before `chain_map`. A path and the same structure loaded into memory have different hashes. The hash of a path is cached with the size and modification time of the file and computed again when either changes. |
| `native_scores()` | What the tool that produced the pose reported: a flat dictionary of names to `float` or None, values only. None is a missing value (NaN becomes None), and the reason, where a kind keeps one, stays in `reasons`. A new dictionary each time. |

## The kinds

**`PredictedPose`** is a co-folding model's output for one sample, with every field, method and rule described with the adapters. Its `native_scores()` holds `avg_plddt`, `ptm`, `iptm`, `gpde`, `ranking_score`, `has_clash` and `disorder`, then the per-chain values as `chain_ptm.<chain>` and `chain_pair_iptm.<pair key>`, with the keys as the adapter wrote them (`chain_pair_iptm.(A, B)` for OpenFold3). The arrays (`plddt_per_atom`, `pae`, `pde`) are not scalars and stay fields of the record. Two optional keyword fields serve the pose interface: `provenance` and `given_id`.

**`DockedPose(source, *, chain_map, program, scores, decoy, provenance, given_id)`** is the output of a docking or design tool, which reports no model confidences. `source` is a path or an `AtomArray`, `program` names the tool, and `scores` holds the program's own scores under the names it uses (`native_scores()` returns them). `decoy=True` marks a structure that was made wrong on purpose. A bare structure that is neither predicted nor experimental (a decoy displaced by hand, a fixture) is a `DockedPose` with `decoy=True` and no `program`. Constructors: `DockedPose.from_file(path, *, program=None, scores=None, decoy=False, chain_map=None, id=None, **provenance)` and `DockedPose.from_atoms(atoms, *, program=None, scores=None, decoy=False, chain_map=None, id=None, **provenance)`; the keyword arguments left over are the provenance.

**`ExperimentalPose(source, *, chain_map, method, resolution_angstrom, pdb_id, quality, provenance, given_id)`** is a crystal, NMR or cryo-EM structure with what the deposition reports: `method` as the deposition states it, `resolution_angstrom` (positive, None for NMR), `pdb_id`, and `quality`, a dictionary of other values such as `{"r_free": 0.25}`. `native_scores()` returns `resolution_angstrom` (None when unknown), then the keys of `quality`; `quality` may not use the key `resolution_angstrom`. Constructors: `ExperimentalPose.from_file(path, *, method=None, resolution_angstrom=None, pdb_id=None, quality=None, chain_map=None, id=None, **provenance)` and `ExperimentalPose.from_atoms(atoms, *, ...)` with the same keywords.

`DockedPose` and `ExperimentalPose` are frozen and, like `PredictedPose`, compare by identity (`content_hash()` tells whether two poses hold the same structure). The constructors copy the dictionaries they are given and check them: `TypeError` for a provenance value that is not a JSON scalar, a score that is neither a number nor None, or a `source` that is neither a path nor a single-model `AtomArray`; `ValueError` for an invalid `chain_map`, an infinite score (use None for a value that is missing) or a resolution that is not positive and finite. `from_file` raises `FileNotFoundError` for a path that is not a file; the file is read when `atoms()`, `content_hash()` or a default `id` asks.

## Computed metrics are not stored on a pose

A pose holds what came with the structure and what the producing tool reported. A metric computed from it (buried area, an RMSD, an interface PAE) is a function of the pose, and its value lives in the result dictionary of that function. The same pose can then be measured again with other settings, and a computed value is never mistaken for one that the tool reported. In a pose-set result the two sit side by side in the row of a pose, the reported values under `native` and each computed block under the name of its metric ([`pose_sets.md`](pose_sets.md#the-result)).

```python
from binding_metrics.poses import DockedPose, ExperimentalPose
from binding_metrics.predictors import get_parser

crystal = ExperimentalPose.from_file(
    "1ycr.cif", method="X-RAY DIFFRACTION", resolution_angstrom=2.2, pdb_id="1YCR"
)
docked = DockedPose.from_file("rank_1.pdb", program="haddock", scores={"haddock_score": -112.4})
predicted = get_parser("of3").load("out/", "1ycr")  # a PredictedPose

for pose in (crystal, docked, predicted):
    print(pose.kind, pose.id, pose.native_scores())
```

## Renamed: PredictionRecord

`PredictedPose` is the former `PredictionRecord`; nothing was removed from it. The old name still imports from `binding_metrics.predictors`, from `binding_metrics.predictors.record` and from the modules that imported it (the adapters, `binding_metrics.metrics.evobind`, `.openfold` and `.prediction`), and every use emits a `DeprecationWarning`: "PredictionRecord is deprecated and renamed PredictedPose; the old name will be removed in a future release (not before 0.4.0)". It is the same class, so `isinstance` and subclassing work, and it is not in `__all__`. The `TypeError` messages of `compute_evobind_adversarial_from_records` and `compute_evobind_score_from_record` name `PredictedPose`.
