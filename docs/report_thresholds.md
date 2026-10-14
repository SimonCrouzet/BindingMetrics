# Report scorecard thresholds

The `--summary` flag of `binding-metrics-run` and `binding-metrics-report` writes a summary with a RAG
scorecard (🟢 OK / 🟡 AMBER / 🔴 RED / ⬜ N/A). The thresholds are the `_THRESHOLDS` list in
`src/binding_metrics/protocols/report.py`. A missing or non-finite value is shown as ⬜ and is not rated.
The bands are heuristic; no calibration against binder and non-binder data ships with the package.

The metrics themselves are described in [`metrics.md`](metrics.md).

---

## MD RMSD — final frame (Å)

RMSD between the last MD frame and the energy-minimized structure, computed over all atoms of the system
(hydrogens included) after Kabsch alignment. The bands were set for the default MD duration (200 ps); for
longer runs or larger systems adjust `_THRESHOLDS`. The row is ⬜ when the run has no MD
(`--md-duration-ps 0`).

| Band     | Value |
|----------|-------|
| 🟢 OK    | < 2 Å |
| 🟡 AMBER | 2–5 Å |
| 🔴 RED   | > 5 Å |

---

## RMSF mean (Å)

Mean per-residue RMSF of peptide Cα atoms over the saved MD frames. The frames are not superposed first, so
the value includes any overall drift of the complex.

| Band     | Value |
|----------|-------|
| 🟢 OK    | < 1 Å |
| 🟡 AMBER | 1–2 Å |
| 🔴 RED   | > 2 Å |

---

## E_int — interaction energy (kJ/mol)

`compute_interaction_energy` on the relaxed structure: E_complex − E_peptide − E_receptor with AMBER ff14SB
and implicit solvent (generalized Born), a single-structure end-state estimate. The scorecard takes the
`relaxed` value, else `after_md`, else `raw`. Solvation is included as described in
[`metrics.md`](metrics.md#5-force-field-interaction-energy); conformational reorganisation and entropy are
not. The value is not comparable to an experimental ΔG or Kd.

| Band     | Value                            |
|----------|----------------------------------|
| 🟢 OK    | < −40 kJ/mol (about 10 kcal/mol) |
| 🟡 AMBER | −40–0 kJ/mol                     |
| 🔴 RED   | > 0 kJ/mol                       |

---

## ΔSASA (Å²)

Solvent-accessible surface area buried at the interface upon complex formation:
SASA_peptide + SASA_receptor − SASA_complex, computed on heavy atoms. Both partners are counted, so about
half of it lies on each side.

| Band     | Value       |
|----------|-------------|
| 🟢 OK    | > 1000 Å²   |
| 🟡 AMBER | 500–1000 Å² |
| 🔴 RED   | < 500 Å²    |

---

## H-bonds

Cross-chain H-bond count (Baker-Hubbard with biotite's defaults: H···acceptor ≤ 2.5 Å, D–H···A angle 120°),
deduplicated to heavy-atom donor/acceptor pairs.

| Band     | Value |
|----------|-------|
| 🟢 OK    | ≥ 5   |
| 🟡 AMBER | 2–4   |
| 🔴 RED   | < 2   |

`hbond_energy` (kcal/mol, ≤ 0): 🟢 ≤ −10, 🟡 ≤ −2, 🔴 > −2. The energy is a heuristic ranking score, not a
force-field energy.

---

## Salt bridges

Cross-chain residue-pair count (positive: LYS/ARG/HIP, negative: ASP/GLU and the phosphate oxygens of
SEP/TPO/PTR; 0.5–5.5 Å). Plain HIS is treated as neutral.

| Band     | Value |
|----------|-------|
| 🟢 OK    | ≥ 2   |
| 🟡 AMBER | 1     |
| 🔴 RED   | 0     |

`saltbridge_energy` (kcal/mol at ε=4, ≤ 0): 🟢 ≤ −40, 🟡 ≤ −10, 🔴 > −10. `saltbridges_bidentate` is the
subset with ≥ 2 atom-pair contacts. A binder without ionisable side chains, such as the macrocycle of 1CWA,
has 0 salt bridges and is rated 🔴.

---

## Ramachandran favoured (%)

Percentage of peptide backbone residues in the favoured φ/ψ regions, evaluated on the MD-final structure (or
the minimized one when there is no MD). The regions are the box approximations described in
[`metrics.md`](metrics.md#6-ramachandran--omega-planarity), not MolProbity's density contours.

| Band     | Value   |
|----------|---------|
| 🟢 OK    | > 95 %  |
| 🟡 AMBER | 80–95 % |
| 🔴 RED   | < 80 %  |

N-methylation and ring constraints move φ/ψ outside the general regions, so an N-methylated or cyclic
peptide can be rated 🔴.

---

## ω outlier fraction

Fraction of peptide bonds whose ω deviates by more than 15° from 180°. Cis bonds are not treated separately:
a cis-proline or an N-methylated cis amide deviates by about 180° and counts as an outlier.

| Band     | Value     |
|----------|-----------|
| 🟢 OK    | < 0.05    |
| 🟡 AMBER | 0.05–0.20 |
| 🔴 RED   | > 0.20    |

---

## Shape complementarity Sc

Sc of the interface (`sc`), a dot-and-normal approximation of Lawrence & Colman (1993) that is not
numerically comparable with CCP4 `sc` ([`metrics.md`](metrics.md#7-shape-complementarity)).

| Band     | Value   |
|----------|---------|
| 🟢 OK    | > 0.7   |
| 🟡 AMBER | 0.5–0.7 |
| 🔴 RED   | < 0.5   |

---

## Coulomb energy (kJ/mol)

Sum of pairwise Coulomb interactions between the formally charged atoms of the peptide and the receptor
(ε = 4, 12 Å cutoff; charges taken from residue names, no termini, no solvent screening); see
[`metrics.md`](metrics.md#4-electrostatics).

| Band     | Value         |
|----------|---------------|
| 🟢 OK    | < −100 kJ/mol |
| 🟡 AMBER | −100–0 kJ/mol |
| 🔴 RED   | > 0 kJ/mol    |

---

## DockQ

Reference-based score of a prediction against a native structure. The row is ⬜ unless `--reference` (or
`--reference-dir` in batch runs) supplied one. The bands are the CAPRI classes of the DockQ score
([`metrics.md`](metrics.md#10-reference-based-accuracy-dockq)).

| Band     | Value     | CAPRI class          |
|----------|-----------|----------------------|
| 🟢 OK    | ≥ 0.80    | High                 |
| 🟡 AMBER | 0.23–0.80 | Acceptable or Medium |
| 🔴 RED   | < 0.23    | Incorrect            |
