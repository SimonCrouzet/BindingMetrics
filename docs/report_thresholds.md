# Report scorecard thresholds

The `--summary` flag of `binding-metrics-run` and `binding-metrics-report` writes a summary with a RAG
scorecard (🟢 OK / 🟡 AMBER / 🔴 RED / ⬜ N/A). Thresholds are defined in
`src/binding_metrics/protocols/report.py` (`_THRESHOLDS` list) and are easy to adjust per project. A
missing or non-finite value is shown as ⬜ and is not rated.

The bands are heuristic starting points for ranking designs. No calibration against binder and non-binder
data ships with the package, so read a colour as a prompt to look at the number and not as a verdict. Where
a rationale below cites a bundled example, the value comes from running the metric on that structure.

The metrics themselves are described in [`metrics.md`](metrics.md).

---

## MD RMSD — final frame (Å)

RMSD between the last MD frame and the energy-minimized structure, computed over all atoms of the system
(hydrogens included) after Kabsch alignment.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | < 2 Å  | The complex stays close to the minimized pose |
| 🟡 AMBER | 2–5 Å  | Moderate drift; may indicate flexibility or force-field strain |
| 🔴 RED   | > 5 Å  | Large conformational change during MD; pose reliability uncertain |

**Note:** thresholds depend on MD duration (default 200 ps) and system size. For longer runs or larger
systems consider relaxing to 5/10 Å. The row is ⬜ when the run has no MD (`--md-duration-ps 0`).

---

## RMSF mean (Å)

Mean per-residue RMSF of peptide Cα atoms over the saved MD frames. The frames are not superposed first, so
the value includes any overall drift of the complex.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | < 1 Å  | Little motion of the peptide |
| 🟡 AMBER | 1–2 Å  | Moderate flexibility |
| 🔴 RED   | > 2 Å  | Large motion; the binding pose may not be representative |

---

## E_int — interaction energy (kJ/mol)

`compute_interaction_energy` on the relaxed structure: E_complex − E_peptide − E_receptor with AMBER ff14SB
and implicit solvent (generalized Born), a single-structure end-state estimate. The scorecard takes the
`relaxed` value, else `after_md`, else `raw`. Solvation is included as described in
[`metrics.md`](metrics.md#5-force-field-interaction-energy); conformational reorganisation and entropy are
not. Useful for ranking within a campaign, not directly comparable to experimental ΔG or Kd.

| Band  | Value      | Rationale |
|-------|------------|-----------|
| 🟢 OK    | < −40 kJ/mol  | Net attraction of more than about 10 kcal/mol |
| 🟡 AMBER | −40–0 kJ/mol  | Weak or marginal attraction |
| 🔴 RED   | > 0 kJ/mol    | Net repulsion |

---

## ΔSASA (Å²)

Solvent-accessible surface area buried at the interface upon complex formation:
SASA_peptide + SASA_receptor − SASA_complex, computed on heavy atoms. Both partners are counted, so about
half of it lies on each side.

| Band  | Value     | Rationale |
|-------|-----------|-----------|
| 🟢 OK    | > 1000 Å² | Large buried area |
| 🟡 AMBER | 500–1000 Å² | Partial burial |
| 🔴 RED   | < 500 Å²  | Small buried area |

The bundled complexes give 1466 Å² (1YCR), 1508 Å² (3P8F) and 985 Å² (1CWA), so the cyclosporin A complex
falls in the amber band.

---

## H-bonds

Cross-chain H-bond count (Baker-Hubbard with biotite's defaults: H···acceptor ≤ 2.5 Å, D–H···A angle 120°),
deduplicated to heavy-atom donor/acceptor pairs.

| Band  | Value | Rationale |
|-------|-------|-----------|
| 🟢 OK    | ≥ 5   | Many polar contacts |
| 🟡 AMBER | 2–4   | A few polar contacts |
| 🔴 RED   | < 2   | Very few polar contacts |

`hbond_energy` (kcal/mol, ≤ 0): 🟢 ≤ −10, 🟡 ≤ −2, 🔴 > −2. The energy is a heuristic ranking score, not a
force-field energy.

---

## Salt bridges

Cross-chain residue-pair count (positive: LYS/ARG/HIP, negative: ASP/GLU and the phosphate oxygens of
SEP/TPO/PTR; 0.5–5.5 Å). Plain HIS is treated as neutral.

| Band  | Value | Rationale |
|-------|-------|-----------|
| 🟢 OK    | ≥ 2   | Several ionic contacts |
| 🟡 AMBER | 1     | A single salt bridge |
| 🔴 RED   | 0     | No ionic contacts |

`saltbridge_energy` (kcal/mol at ε=4, ≤ 0): 🟢 ≤ −40, 🟡 ≤ −10, 🔴 > −10. `saltbridges_bidentate` is the
subset with ≥ 2 atom-pair contacts. A binder without ionisable side chains, such as the macrocycle of 1CWA,
has 0 salt bridges and is rated 🔴 by construction.

---

## Ramachandran favoured (%)

Percentage of peptide backbone residues in the favoured φ/ψ regions, evaluated on the MD-final structure (or
the minimized one when there is no MD). The regions are the box approximations described in
[`metrics.md`](metrics.md#6-ramachandran--omega-planarity), not MolProbity's density contours.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | > 95 % | Nearly all residues in the favoured boxes |
| 🟡 AMBER | 80–95 % | Some residues outside the favoured boxes |
| 🔴 RED   | < 80 % | Many residues outside the favoured boxes |

**Note:** N-methylation and ring constraints move φ/ψ outside the general regions. On the bundled
structures the favoured fraction is 90.9 % for the linear peptide of 1YCR, 85.7 % for the bicyclic SFTI-1
peptide of 3P8F and 72.7 % for cyclosporin A (1CWA), which is rated 🔴.

---

## ω outlier fraction

Fraction of peptide bonds whose ω deviates by more than 15° from 180°. Cis bonds are not treated separately:
a cis-proline or an N-methylated cis amide deviates by about 180° and counts as an outlier.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | < 0.05 | Fewer than 5 % of the bonds are non-planar |
| 🟡 AMBER | 0.05–0.20 | 5–20 % of the bonds; warrants inspection |
| 🔴 RED   | > 0.20 | More than 20 % of the bonds |

---

## Shape complementarity Sc

Sc of the interface (`sc`), a dot-and-normal approximation of Lawrence & Colman (1993) that is not
numerically comparable with CCP4 `sc` ([`metrics.md`](metrics.md#7-shape-complementarity)).

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | > 0.7  | Complementary surfaces |
| 🟡 AMBER | 0.5–0.7 | Moderate complementarity |
| 🔴 RED   | < 0.5  | Flat or poorly matched surfaces |

The docstring of the metric puts well-formed native protein–protein interfaces at about 0.65–0.75 and peptide
interfaces a little lower. The bundled complexes give 0.632 (1YCR, 🟡), 0.738 (3P8F, 🟢) and 0.750 (1CWA, 🟢).

---

## Coulomb energy (kJ/mol)

Sum of pairwise Coulomb interactions between the formally charged atoms of the peptide and the receptor
(ε = 4, 12 Å cutoff; charges taken from residue names, no termini, no solvent screening). It complements the
force-field interaction energy with an explicit electrostatic view; see
[`metrics.md`](metrics.md#4-electrostatics).

| Band  | Value       | Rationale |
|-------|-------------|-----------|
| 🟢 OK    | < −100 kJ/mol | Strong net electrostatic attraction |
| 🟡 AMBER | −100–0 kJ/mol | Weak or mixed electrostatics |
| 🔴 RED   | > 0 kJ/mol    | Net electrostatic repulsion |

---

## DockQ

Reference-based score of a prediction against a native structure. The row is ⬜ unless `--reference` (or
`--reference-dir` in batch runs) supplied one. The bands follow the CAPRI classes of the DockQ score
([`metrics.md`](metrics.md#10-reference-based-accuracy-dockq)).

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | ≥ 0.80 | CAPRI class High |
| 🟡 AMBER | 0.23–0.80 | CAPRI class Acceptable or Medium |
| 🔴 RED   | < 0.23 | CAPRI class Incorrect |
