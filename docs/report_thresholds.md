# Report scorecard thresholds

The `--summary` flag in `binding-metrics-report` appends a RAG scorecard
(🟢 OK / 🟡 AMBER / 🔴 RED) to the Markdown report.  Thresholds are defined
in `src/binding_metrics/protocols/report.py` (`_THRESHOLDS` list) and are
easy to adjust per project.

---

## MD RMSD — final frame (Å)

RMSD between the last MD frame and the energy-minimized structure, computed
over all heavy atoms after Kabsch alignment.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | < 2 Å  | Peptide stays close to minimized pose; conformation is stable |
| 🟡 AMBER | 2–5 Å  | Moderate drift; may indicate flexibility or force-field strain |
| 🔴 RED   | > 5 Å  | Large conformational change during MD; pose reliability uncertain |

**Note:** thresholds depend on MD duration (default 200 ps) and peptide
size.  For longer runs or larger peptides consider relaxing to 5/10 Å.

---

## RMSF mean (Å)

Mean per-residue RMSF of peptide Cα atoms over the MD trajectory.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | < 1 Å  | Rigid, well-anchored peptide |
| 🟡 AMBER | 1–2 Å  | Moderate flexibility; still plausible binder |
| 🔴 RED   | > 2 Å  | Highly flexible; binding pose may not be representative |

---

## E_int — interaction energy (kJ/mol)

OpenMM non-bonded interaction energy: E_complex − E_peptide − E_receptor,
evaluated on the energy-minimized structure.  **Gas-phase only** — solvation
and entropy are not included.  Useful for ranking within a campaign but not
directly comparable to experimental ΔG or Kd.

| Band  | Value      | Rationale |
|-------|------------|-----------|
| 🟢 OK    | < −40 kJ/mol  | Strong non-bonded attraction (~−10 kcal/mol) |
| 🟡 AMBER | −40–0 kJ/mol  | Weak or marginal attraction |
| 🔴 RED   | > 0 kJ/mol    | Net repulsion in gas phase |

---

## ΔSASA (Å²)

Solvent-accessible surface area buried at the interface upon complex
formation: SASA_peptide + SASA_receptor − SASA_complex.

| Band  | Value     | Rationale |
|-------|-----------|-----------|
| 🟢 OK    | > 1000 Å² | Well-buried interface; typical of tight peptide binders |
| 🟡 AMBER | 500–1000 Å² | Partial burial; may be sufficient for moderate affinity |
| 🔴 RED   | < 500 Å²  | Poorly buried; likely a weak or non-specific interaction |

---

## H-bonds

Number of intermolecular hydrogen bonds at the interface (donor–acceptor
distance < 3.5 Å, angle > 120°).

| Band  | Value | Rationale |
|-------|-------|-----------|
| 🟢 OK    | ≥ 5   | Good polar complementarity |
| 🟡 AMBER | 2–4   | Acceptable for hydrophobic-dominant binders |
| 🔴 RED   | < 2   | Very few polar contacts; binding likely non-specific |

---

## Salt bridges

Number of charge–charge pairs at the interface (oppositely charged heavy
atoms within 4 Å).

| Band  | Value | Rationale |
|-------|-------|-----------|
| 🟢 OK    | ≥ 2   | Multiple ionic interactions; strong electrostatic contribution |
| 🟡 AMBER | 1     | One salt bridge; minor electrostatic contribution |
| 🔴 RED   | 0     | No ionic contacts |

---

## Ramachandran favoured (%)

Percentage of peptide backbone residues in favoured φ/ψ regions
(MolProbity definition).  Evaluated on the MD-final structure.

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | > 95 % | Excellent backbone geometry |
| 🟡 AMBER | 80–95 % | Acceptable; expected for constrained or cyclic peptides |
| 🔴 RED   | < 80 % | Poor backbone geometry; possible force-field artefact |

**Note:** cyclic and D-amino-acid-containing peptides inherently have more
Ramachandran outliers than linear L-peptides.  A 🟡 rating here is expected
and should not be penalised for macrocyclic designs.

---

## ω outlier fraction

Fraction of peptide bonds with |ω deviation| > 15° from planarity (180° or 0°
for cis-Pro).

| Band  | Value  | Rationale |
|-------|--------|-----------|
| 🟢 OK    | < 0.05 | < 5 % of bonds non-planar; normal |
| 🟡 AMBER | 0.05–0.20 | 5–20 %; warrants inspection |
| 🔴 RED   | > 0.20 | > 20 %; likely a simulation or modelling artefact |

---

## Coulomb energy (kJ/mol)

Sum of pairwise Coulomb interactions between charged atoms of the peptide and
receptor (ε = 1, no cutoff).  Complements the OpenMM interaction energy with
an explicit electrostatic view.

| Band  | Value       | Rationale |
|-------|-------------|-----------|
| 🟢 OK    | < −100 kJ/mol | Strong net electrostatic attraction |
| 🟡 AMBER | −100–0 kJ/mol | Weak or mixed electrostatics |
| 🔴 RED   | > 0 kJ/mol    | Net electrostatic repulsion |
