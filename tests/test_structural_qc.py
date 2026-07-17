"""Structural-integrity / QC checks on relaxation output.

binding-metrics is a structure-QC tool: a minimization can report
``success=True`` with a finite energy while having blown up the geometry
(exploded coordinates, fused atoms, NaNs). The existing relaxation tests only
assert ``result.success`` and ``energy is not None`` — they would *not* catch a
structurally-broken result. This module adds a QC pass that verifies the
relaxed structure is physically sound.

For each bundled public example (1YCR linear peptide, 3P8F cyclic peptide, and
cyclosporin — the cyclophilin A–CsA complex, an NCAA / head-to-tail macrocycle
with D-amino acids, N-methylation and GAFF-auto-parameterized residues) we run a
short minimize-only relaxation on CUDA and then assert:

1. Energy is finite, within a sane range, and did not *increase* during
   minimization (minimization can only lower the energy — a higher final energy
   would signal a broken run). The pre-minimization energy is captured
   best-effort; the finite/range check always runs.
2. The minimized structure did not explode: heavy-atom RMSD to the relaxation
   input is finite and bounded (< 5 Å).
3. All coordinates in the minimized structure are finite (no NaN/inf).
4. No egregious clashes: the closest pair of heavy atoms belonging to
   *different residues* is farther apart than 0.8 Å.
5. No covalent bond was stretched/broken: every heavy–heavy bond perceived by
   distance in the (good) input geometry stays within a physical covalent range
   [0.5, 2.5] Å in the minimized structure.
6. Chirality preserved: the signed tetrahedral volume at each Cα does not change
   sign between input and minimized (no stereocenter inversion — critical for
   D-amino acids).
7. No missing/extra heavy atoms: the minimized structure has the same total
   heavy-atom count and the same per-residue heavy-atom composition as the
   input.

The measured values on current code (short 50/20/50-step minimizations) are::

                 energy(min)   energy(pre)   heavy RMSD   min inter-res dist
    1YCR        -13999 kJ/mol  +5130         0.327 Å      1.328 Å
    3P8F        -33074 kJ/mol -27673         0.281 Å      1.329 Å
    cyclosporin -21513 kJ/mol   n/a          0.313 Å      1.327 Å

                min bond   max bond   Cα centers   heavy atoms
    1YCR        1.219 Å    2.398 Å    94           819
    3P8F        1.218 Å    2.250 Å    215          1970
    cyclosporin 1.220 Å    2.098 Å    152          1351

so every bound below is set with wide margin and passes on a genuinely-relaxed
structure while still failing on an exploded one.

Known issue — check 6 (chirality) currently fails on 3P8F in roughly one run in
three, and this is a *real relaxation defect, not a test artifact*: the
minimizer inverts the Cα stereocenter of A140 (PHE), whose signed volume goes
from +2.54 Å³ in the input to about -1.77 Å³ in the minimized structure. Both
values are clean, large-magnitude tetrahedra (so this is not sign noise on a
near-planar centre), and A140's input geometry is entirely typical (improper
52.2° against a 52.3° median over the 215 centres), so there is no principled
basis for excluding it. The outcome is deterministic within a process but varies
across processes (CUDA non-determinism tips which side of the improper barrier
the minimizer lands on), which is what makes it intermittent. The check is
deliberately left strict: weakening it to go green would defeat the very
inversion it exists to catch. 1YCR and cyclosporin — including cyclosporin's
D-alanine, whose Cα carries the opposite sign (-1.0) to every L-residue — show
0 flips.

Runs are guarded by ``requires_cuda`` and skip gracefully without a GPU. Systems
are kept small (minimize-only, tiny step counts) to share an 8 GB GPU.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from conftest import requires_cuda

from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig
from binding_metrics.metrics.comparison import compute_structure_rmsd

DATA_DIR = Path(__file__).parent.parent / "data"

# --- QC thresholds (see module docstring for the measured values behind them) ---
ENERGY_MAX_KJ = 1.0e6          # blown-up structures show |E| >> 1e6 (or inf)
ENERGY_MIN_KJ = -1.0e8         # sane lower bound for a small implicit-solvent system
ENERGY_DECREASE_TOL = 1.0      # kJ/mol slack when asserting min <= pre-min
RMSD_MAX_ANG = 5.0             # heavy-atom RMSD above this = exploded
MIN_HEAVY_DIST_ANG = 0.8       # closest non-same-residue heavy-atom pair must exceed this

# Heavy-heavy covalent window used to *perceive* bonds from the (good) input
# geometry: a C–C/C–N/C–O single bond is ~1.5 Å, an aromatic bond ~1.39 Å, and
# an S–S disulfide ~2.05 Å, so [0.9, 2.1] Å captures real heavy-heavy bonds
# without picking up non-bonded first-shell contacts (~2.8 Å+).
BOND_PERCEIVE_MIN_ANG = 0.9
BOND_PERCEIVE_MAX_ANG = 2.1
# Allowed length of a perceived bond in the *minimized* structure: wide enough
# to cover any real heavy-heavy bond (up to S–S ~2.05 Å) with margin, tight
# enough that a stretched/broken multi-Å bond fails.
BOND_LENGTH_MIN_ANG = 0.5
BOND_LENGTH_MAX_ANG = 2.5

# A real Cα stereocenter has |signed tetrahedral volume| ~2.2–2.6 Å³. A genuine
# inversion swings the sign between two such large-magnitude values; only a
# near-planar (|V| ~0) center has a numerically ambiguous sign that mixed-
# precision minimization noise can tip. We therefore only count a sign change as
# an inversion when BOTH the input and minimized volumes exceed this magnitude,
# which is far below any real center yet well above the noise floor.
CHIRALITY_MIN_VOLUME_A3 = 0.5


@dataclass
class RelaxedExample:
    """Bundle of everything the QC assertions need for one relaxed example."""
    name: str
    input_path: Path          # the (prepped) structure handed to the relaxer
    minimized_path: Path      # the minimized CIF the relaxer wrote
    energy_min: float         # potential_energy_minimized (kJ/mol)
    energy_pre: Optional[float]  # pre-minimization energy, or None if not captured


# ---------------------------------------------------------------------------
# Helpers (kept local to this file to avoid colliding with concurrent edits)
# ---------------------------------------------------------------------------

def _prep_on_the_fly(raw: Path, out: Path) -> Path:
    """Run the same PDBFixer prep the pipeline uses, writing a FF-ready CIF."""
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_structure

    topology, positions = load_structure(raw)
    topology, positions = prep_structure(
        topology, positions, ph=7.4, keep_water=False, canonicalize=False
    )
    save_structure(topology, positions, out, source_path=raw)
    return out


def _small_config() -> RelaxationConfig:
    """Minimize-only config with tiny step counts (GPU-friendly)."""
    return RelaxationConfig(
        md_duration_ps=0.0,
        min_steps_initial=50,
        min_steps_restrained=20,
        min_steps_final=50,
        device="cuda",
        small_molecules=None,
    )


def _small_config_gaff() -> RelaxationConfig:
    """Same tiny minimize-only config, but GAFF auto-parameterizes NCAAs.

    Required for structures carrying exotic residues (e.g. cyclosporin's
    BMT/ABA): with ``small_molecules=None`` the system setup cannot build a
    template for them and fails.
    """
    config = _small_config()
    config.small_molecules = "auto"
    return config


def _capture_premin_energy(
    input_path: Path, config: Optional[RelaxationConfig] = None
) -> Optional[float]:
    """Single-point potential energy of the relaxation input, before minimizing.

    Best-effort: builds the same OpenMM system the relaxer would and evaluates
    the energy at the input coordinates on the CPU platform (so it never
    competes with the CUDA minimization for GPU memory). Returns None on any
    failure so the core QC checks still run if internals change.
    """
    try:
        import openmm
        import openmm.unit as unit

        relaxer = ImplicitRelaxation(config or _small_config())
        system, _topology, positions, _bond_info = relaxer._setup_system(input_path)
        context = openmm.Context(
            system,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("CPU"),
        )
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        return float(energy.value_in_unit(unit.kilojoules_per_mole))
    except Exception:
        return None


def _relax(
    input_path: Path,
    output_dir: Path,
    name: str,
    config: Optional[RelaxationConfig] = None,
    capture_premin: bool = True,
) -> RelaxedExample:
    """Capture pre-min energy, run a short minimize-only relaxation, bundle it.

    ``config`` defaults to :func:`_small_config`; pass :func:`_small_config_gaff`
    for NCAA structures. ``capture_premin`` can be disabled for slow GAFF cases
    where the single-point setup would trigger a second full parameterization —
    ``energy_pre=None`` is handled gracefully by the energy-decrease check.
    """
    config = config or _small_config()
    energy_pre = _capture_premin_energy(input_path, config) if capture_premin else None

    relaxer = ImplicitRelaxation(config)
    result = relaxer.run(input_path, output_dir, sample_id=name)

    assert result.success, f"{name} relaxation failed: {result.error_message}"
    assert result.minimized_structure_path is not None
    assert result.potential_energy_minimized is not None

    return RelaxedExample(
        name=name,
        input_path=input_path,
        minimized_path=Path(result.minimized_structure_path),
        energy_min=float(result.potential_energy_minimized),
        energy_pre=energy_pre,
    )


def _all_coords_finite(cif_path: Path) -> bool:
    """True if every atom coordinate in the structure is finite (no NaN/inf)."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    for chain in model:
        for residue in chain:
            for atom in residue:
                p = atom.pos
                if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
                    return False
    return True


def _iter_residues(model):
    """Yield ``(chain, residue, key)`` with a key that is unique per residue.

    ``(chain, seqid)`` is NOT unique in our own examples: the prepped 3P8F
    receptor chain genuinely repeats seqids 228-239, and their insertion codes
    are blank, so the insertion code does not disambiguate them either. Keying
    checks on ``(chain, seqid)`` therefore made residues collide *silently* —
    ~11 Cα centres simply vanished from the chirality check (215 reported for a
    226-residue topology), and the bond-length map could pair atoms across two
    different residues that happened to share a number.

    Both the input and the minimized CIF enumerate residues in the same order
    (the minimized file is written from the same topology), so appending an
    occurrence counter restores a stable 1:1 correspondence between the files.
    """
    seen: dict[tuple[str, int, str], int] = {}
    for chain in model:
        for residue in chain:
            base = (chain.name, residue.seqid.num, residue.seqid.icode)
            occ = seen.get(base, 0)
            seen[base] = occ + 1
            yield chain, residue, base + (occ,)


def _min_interresidue_heavy_distance(cif_path: Path) -> tuple[float, int]:
    """Closest heavy-atom pair whose atoms live in *different* residues.

    Simplification (documented): we exclude only same-residue pairs, not
    covalently-bonded inter-residue pairs. The tightest legitimate inter-residue
    contacts are therefore the bonded ones — the backbone peptide bond
    C(i)–N(i+1) at ~1.33 Å and disulfide S–S at ~2.05 Å. The 0.8 Å clash
    threshold sits safely *below* those, so genuine bonded contacts pass while a
    real steric clash (fused/overlapping atoms, << 0.8 Å) is still caught.

    Returns (min_distance_Å, n_heavy_atoms).
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    coords: list[list[float]] = []
    res_keys: list[int] = []
    model = structure[0]
    for _chain, residue, key in _iter_residues(model):
        if residue.name in {"HOH", "WAT"}:
            continue
        for atom in residue:
            if atom.element.name == "H":
                continue
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            res_keys.append(hash(key))

    xyz = np.asarray(coords, dtype=float)
    keys = np.asarray(res_keys)
    n = len(xyz)

    # Row-by-row min distance to atoms in other residues (memory-light for ~2k atoms).
    min_dist = math.inf
    for i in range(n):
        d = np.sqrt(((xyz[i] - xyz) ** 2).sum(axis=1))
        other = keys != keys[i]
        if other.any():
            min_dist = min(min_dist, float(d[other].min()))
    return min_dist, n


def _heavy_atom_positions(cif_path: Path) -> dict[tuple[str, int, str], np.ndarray]:
    """Map ``(chain, resseq, atomname) -> xyz`` for every heavy atom.

    Waters and hydrogens are excluded. Keying by atom name (rather than file
    order) lets us line up the *same* atom between the input and minimized
    structures without trusting that both files enumerate atoms identically.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    positions: dict[tuple, np.ndarray] = {}
    for _chain, residue, key in _iter_residues(model):
        if residue.name in {"HOH", "WAT"}:
            continue
        for atom in residue:
            if atom.element.name == "H":
                continue
            positions[key + (atom.name,)] = np.array(
                [atom.pos.x, atom.pos.y, atom.pos.z]
            )
    return positions


def _perceive_heavy_bonds(
    positions: dict[tuple[str, int, str], np.ndarray],
) -> list[tuple[tuple[str, int, str], tuple[str, int, str]]]:
    """Perceive heavy–heavy covalent bonds by distance from a good geometry.

    Simplification (documented): connectivity is *distance-perceived* from the
    input structure, whose geometry is trusted (un-exploded). Any heavy-atom
    pair whose input separation falls in the covalent window
    ``[BOND_PERCEIVE_MIN_ANG, BOND_PERCEIVE_MAX_ANG]`` is treated as a bond.
    This avoids the fragility of trusting a fixed atom order or an external
    topology; the perceived pairs (keyed by atom name) are then looked up in the
    minimized structure to check they were not stretched or broken.

    Returns a list of ``(key_a, key_b)`` atom-key pairs.
    """
    keys = list(positions.keys())
    xyz = np.array([positions[k] for k in keys])
    n = len(keys)
    bonds: list[tuple[tuple[str, int, str], tuple[str, int, str]]] = []
    for i in range(n):
        if i + 1 >= n:
            break
        d = np.sqrt(((xyz[i] - xyz[i + 1 :]) ** 2).sum(axis=1))
        for offset, dist in enumerate(d):
            if BOND_PERCEIVE_MIN_ANG <= dist <= BOND_PERCEIVE_MAX_ANG:
                bonds.append((keys[i], keys[i + 1 + offset]))
    return bonds


def _ca_signed_volumes(cif_path: Path) -> dict[tuple[str, int], float]:
    """Signed Cα tetrahedral volume per residue, keyed by ``(chain, resseq)``.

    For each residue carrying all four of N, CA, C, CB, returns the signed
    volume ``dot(N-CA, cross(C-CA, CB-CA))`` (Å³). Its *sign* encodes the Cα
    handedness; we only ever compare the sign between the input and minimized
    structures to detect stereocenter inversion, never the absolute L/D
    configuration — so it works for D-amino acids without knowing they are D.
    The magnitude is returned too so callers can ignore near-planar centers
    whose sign is numerically ambiguous (see ``CHIRALITY_MIN_VOLUME_A3``).
    Residues missing any of the four atoms (e.g. glycine / sarcosine, which
    have no CB) are skipped.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    volumes: dict[tuple, float] = {}
    for _chain, residue, key in _iter_residues(model):
        atoms: dict[str, np.ndarray] = {}
        for atom in residue:
            if atom.name in {"N", "CA", "C", "CB"}:
                atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        if len(atoms) < 4:
            continue
        ca = atoms["CA"]
        volumes[key] = float(
            np.dot(atoms["N"] - ca, np.cross(atoms["C"] - ca, atoms["CB"] - ca))
        )
    return volumes


def _heavy_atom_composition(
    cif_path: Path,
) -> tuple[int, dict[tuple[str, int], "Counter[str]"]]:
    """Total heavy-atom count and per-residue heavy-atom-name multiset.

    Returns ``(n_heavy_total, {(chain, resseq): Counter(atomname -> count)})``.
    Waters and hydrogens are excluded. The per-residue Counter lets us assert
    that minimization dropped/added/renamed no heavy atom in any residue.
    """
    import gemmi
    from collections import Counter

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    total = 0
    composition: dict[tuple, "Counter[str]"] = {}
    for _chain, residue, key in _iter_residues(model):
        if residue.name in {"HOH", "WAT"}:
            continue
        counter: "Counter[str]" = Counter()
        for atom in residue:
            if atom.element.name == "H":
                continue
            counter[atom.name] += 1
            total += 1
        if counter:
            composition[key] = counter
    return total, composition


# ---------------------------------------------------------------------------
# Fixtures: relax each example once per session, then share across QC checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _relaxed_1ycr(prepped_example_cif, tmp_path_factory) -> RelaxedExample:
    """1YCR (linear peptide), prepped by the session conftest fixture."""
    out = tmp_path_factory.mktemp("qc_relax_1ycr")
    return _relax(Path(prepped_example_cif), out, "1YCR")


@pytest.fixture(scope="session")
def _relaxed_3p8f(tmp_path_factory) -> RelaxedExample:
    """3P8F (cyclic peptide), prepped on the fly (needs H / capped termini)."""
    raw = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")
    prep_dir = tmp_path_factory.mktemp("qc_prep_3p8f")
    prepped = _prep_on_the_fly(raw, prep_dir / "example_bicyclic_sfti1_3P8F_prepped.cif")
    out = tmp_path_factory.mktemp("qc_relax_3p8f")
    return _relax(prepped, out, "3P8F")


@pytest.fixture(scope="session")
def _relaxed_cyclosporin(tmp_path_factory) -> RelaxedExample:
    """Cyclosporin (cyclophilin A–CsA, NCAA / head-to-tail macrocycle).

    Exercises D-amino acids (DAL), N-methylation (MLE/MVA/SAR), head-to-tail
    cyclization and GAFF auto-parameterization of exotic residues (BMT/ABA) in
    one structure. Prepped on the fly (same keep_water=False PDBFixer prep as
    3P8F) and relaxed with ``small_molecules="auto"``. Pre-min energy capture is
    skipped: it would trigger a second (slow) GAFF parameterization.
    """
    raw = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")
    prep_dir = tmp_path_factory.mktemp("qc_prep_cyclosporin")
    prepped = _prep_on_the_fly(
        raw, prep_dir / "example_ncaa_cyclosporin_1CWA_prepped.cif"
    )
    out = tmp_path_factory.mktemp("qc_relax_cyclosporin")
    return _relax(
        prepped, out, "cyclosporin",
        config=_small_config_gaff(),
        capture_premin=False,
    )


@pytest.fixture(params=["1YCR", "3P8F", "cyclosporin"])
def relaxed(request) -> RelaxedExample:
    """Parametrized access to each relaxed example."""
    return request.getfixturevalue(f"_relaxed_{request.param.lower()}")


# ---------------------------------------------------------------------------
# QC tests
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_energy_finite_and_did_not_increase(relaxed: RelaxedExample):
    """Check 1: minimized energy is finite, sane, and not higher than pre-min."""
    e = relaxed.energy_min
    assert math.isfinite(e), f"{relaxed.name}: non-finite minimized energy {e}"
    assert ENERGY_MIN_KJ < e < ENERGY_MAX_KJ, (
        f"{relaxed.name}: minimized energy {e:.1f} kJ/mol out of sane range "
        f"({ENERGY_MIN_KJ}, {ENERGY_MAX_KJ}) — structure likely exploded"
    )
    # Minimization can only lower the energy; a higher final value = broken run.
    if relaxed.energy_pre is not None:
        assert math.isfinite(relaxed.energy_pre)
        assert e <= relaxed.energy_pre + ENERGY_DECREASE_TOL, (
            f"{relaxed.name}: minimized energy {e:.1f} exceeds pre-min "
            f"{relaxed.energy_pre:.1f} kJ/mol"
        )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_structure_did_not_explode(relaxed: RelaxedExample):
    """Check 2: heavy-atom RMSD to the input is finite and bounded (< 5 Å)."""
    rmsd = compute_structure_rmsd(str(relaxed.input_path), str(relaxed.minimized_path))
    value = rmsd["rmsd"]
    assert value is not None, f"{relaxed.name}: RMSD could not be computed"
    assert math.isfinite(value), f"{relaxed.name}: non-finite RMSD {value}"
    assert value < RMSD_MAX_ANG, (
        f"{relaxed.name}: heavy-atom RMSD {value:.3f} Å >= {RMSD_MAX_ANG} Å "
        f"— structure moved far more than a minimization should"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_coordinates_finite(relaxed: RelaxedExample):
    """Check 3: no NaN/inf coordinates in the minimized structure."""
    assert _all_coords_finite(relaxed.minimized_path), (
        f"{relaxed.name}: minimized structure contains non-finite coordinates"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_no_egregious_clashes(relaxed: RelaxedExample):
    """Check 4: no heavy-atom pair in different residues is closer than 0.8 Å."""
    min_dist, n_heavy = _min_interresidue_heavy_distance(relaxed.minimized_path)
    assert n_heavy > 0, f"{relaxed.name}: no heavy atoms found"
    assert math.isfinite(min_dist), f"{relaxed.name}: non-finite min distance"
    assert min_dist > MIN_HEAVY_DIST_ANG, (
        f"{relaxed.name}: closest inter-residue heavy-atom pair is {min_dist:.3f} Å "
        f"(<= {MIN_HEAVY_DIST_ANG} Å) — fused/overlapping atoms indicate an explosion"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_bond_lengths_preserved(relaxed: RelaxedExample):
    """Check 5: no covalent bond was stretched/broken by minimization.

    Bonds are distance-perceived from the input geometry (heavy–heavy pairs in
    the covalent window), keyed by atom name, then re-measured in the minimized
    structure and required to stay in a physical range.
    """
    input_pos = _heavy_atom_positions(relaxed.input_path)
    min_pos = _heavy_atom_positions(relaxed.minimized_path)
    bonds = _perceive_heavy_bonds(input_pos)
    assert bonds, f"{relaxed.name}: no heavy–heavy bonds perceived in input"

    dmin, dmax = math.inf, 0.0
    checked = 0
    for key_a, key_b in bonds:
        if key_a not in min_pos or key_b not in min_pos:
            continue
        dist = float(np.linalg.norm(min_pos[key_a] - min_pos[key_b]))
        dmin, dmax = min(dmin, dist), max(dmax, dist)
        checked += 1
        assert BOND_LENGTH_MIN_ANG <= dist <= BOND_LENGTH_MAX_ANG, (
            f"{relaxed.name}: bond {key_a}–{key_b} is {dist:.3f} Å in the "
            f"minimized structure, outside the physical range "
            f"[{BOND_LENGTH_MIN_ANG}, {BOND_LENGTH_MAX_ANG}] Å — stretched/broken"
        )
    assert checked > 0, f"{relaxed.name}: no perceived bonds could be matched by key"


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_chirality_preserved(relaxed: RelaxedExample):
    """Check 6: minimization did not invert any Cα stereocenter.

    Critical for D-amino acids: we compare the sign of the signed tetrahedral
    volume at each Cα between input and minimized (handedness-agnostic) and
    require it never to flip. Near-planar centers whose sign is numerically
    ambiguous (|V| < CHIRALITY_MIN_VOLUME_A3, far below any real tetrahedral
    center) are ignored so mixed-precision minimization noise cannot spuriously
    tip a sign — a genuine inversion swings between two large-magnitude values
    and is still caught.
    """
    input_vols = _ca_signed_volumes(relaxed.input_path)
    min_vols = _ca_signed_volumes(relaxed.minimized_path)
    shared = set(input_vols) & set(min_vols)
    assert shared, f"{relaxed.name}: no residues with N/CA/C/CB to check chirality"

    flipped = [
        k
        for k in shared
        if (input_vols[k] > 0) != (min_vols[k] > 0)
        and abs(input_vols[k]) >= CHIRALITY_MIN_VOLUME_A3
        and abs(min_vols[k]) >= CHIRALITY_MIN_VOLUME_A3
    ]
    assert not flipped, (
        f"{relaxed.name}: {len(flipped)} Cα stereocenter(s) inverted during "
        f"minimization (chirality flip): "
        f"{sorted((k, round(input_vols[k], 2), round(min_vols[k], 2)) for k in flipped)}"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_no_missing_heavy_atoms(relaxed: RelaxedExample):
    """Check 7: minimization dropped/added no heavy atom.

    Asserts the total heavy-atom count matches and, more strongly, that the
    per-residue heavy-atom-name multiset is identical between input and
    minimized (the input is already H-prepped, so counts must match exactly).
    """
    in_total, in_comp = _heavy_atom_composition(relaxed.input_path)
    min_total, min_comp = _heavy_atom_composition(relaxed.minimized_path)

    assert in_total > 0, f"{relaxed.name}: no heavy atoms found in input"
    assert min_total == in_total, (
        f"{relaxed.name}: heavy-atom count changed {in_total} -> {min_total} "
        f"during minimization"
    )
    assert set(in_comp) == set(min_comp), (
        f"{relaxed.name}: residue set changed during minimization "
        f"(added {sorted(set(min_comp) - set(in_comp))}, "
        f"dropped {sorted(set(in_comp) - set(min_comp))})"
    )
    mismatched = [k for k in in_comp if in_comp[k] != min_comp[k]]
    assert not mismatched, (
        f"{relaxed.name}: per-residue heavy-atom composition changed for "
        f"{len(mismatched)} residue(s): {sorted(mismatched)}"
    )
