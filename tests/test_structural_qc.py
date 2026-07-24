"""Structural-integrity / QC checks on relaxation output.

binding-metrics is a structure-QC tool: a minimization can report
``success=True`` with a finite energy while having blown up the geometry
(exploded coordinates, fused atoms, NaNs). The existing relaxation tests only
assert ``result.success`` and ``energy is not None`` — they would *not* catch a
structurally-broken result. This module adds a QC pass that verifies the
relaxed structure is physically sound.

The three bundled examples are chosen to span the peptide feature space rather
than to repeat it:

    1YCR         p53 / MDM2            linear, all-standard residues
    3P8F         SFTI-1 / matriptase   bicyclic: head-to-tail *and* disulfide
    cyclosporin  CsA / cyclophilin A   head-to-tail macrocycle + D-alanine +
                                       N-methylation (MLE/MVA/SAR) + exotic
                                       residues auto-parameterized by GAFF
                                       (BMT/ABA)

For each we run a short minimize-only relaxation on CUDA and then assert:

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

Measured values (short 50/20/50-step minimizations)::

                 energy(min)   energy(pre)   heavy RMSD   min inter-res dist
    1YCR        -14460 kJ/mol  +5082         0.409 Å      1.330 Å
    3P8F        -33108 kJ/mol -27312         0.293 Å      1.329 Å
    cyclosporin -21515 kJ/mol   n/a          0.297 Å      1.327 Å

                min bond   max bond   Cα centers   heavy atoms
    1YCR        1.218 Å    2.391 Å     94          819
    3P8F        1.217 Å    2.247 Å    225          1970
    cyclosporin 1.219 Å    2.087 Å    152          1351

These are now reproducible run-to-run on a given machine: prep seeds hydrogen
placement and pins its minimization to the deterministic Reference platform,
and the CUDA main minimization is bit-deterministic from a fixed input (see
``test_relaxation_energy_is_reproducible``). The absolute values can still shift
across machines/GPU models and force-field versions, so the bounds below are
kept wide with intent — each passes on any genuinely-relaxed structure while
still failing on an exploded one. Do not tighten them to the numbers above.

The chirality check (6) earned its place immediately: it caught a real prep bug
in which ``addHydrogens`` stranded a Cα hydrogen on the wrong face, which then
forced the minimizer to invert that stereocenter. See
``core.system.repair_ca_hydrogen_chirality`` for the mechanism. Keep this check
strict — weakening it would defeat the very inversion it exists to catch.
Cyclosporin's D-alanine Cα carries the opposite sign to every L-residue, and
the check reads it correctly without being told it is D.

Runs are guarded by ``requires_cuda`` and skip gracefully without a GPU. Systems
are kept small (minimize-only, tiny step counts) to share an 8 GB GPU.
"""

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from conftest import requires_cuda

from binding_metrics.metrics.comparison import compute_structure_rmsd
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

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
) -> tuple[int, dict[tuple[str, int], Counter]]:
    """Total heavy-atom count and per-residue heavy-atom-name multiset.

    Returns ``(n_heavy_total, {(chain, resseq): Counter(atomname -> count)})``.
    Waters and hydrogens are excluded. The per-residue Counter lets us assert
    that minimization dropped/added/renamed no heavy atom in any residue.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    total = 0
    composition: dict[tuple, Counter] = {}
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


@pytest.fixture(scope="session")
def _relaxed_somatostatin(tmp_path_factory) -> RelaxedExample:
    """1XY4 somatostatin analog — a peptide-only LACTAM example.

    The one bundled structure that exercises the side-chain Lys–Glu lactam
    closure (``lactam_sc_lys_glu``), together with a disulfide, a D-amino acid
    (D-Trp), and GAFF auto-parameterization of a non-canonical residue (IAM). It
    is peptide-only (no receptor), so it covers the relaxation and
    structural-QC path — not interface metrics.

    It also guards the lactam residue-name round-trip: prep renames the closing
    residues to the lactam templates GLUL/LYSL, and ``save_cif`` must rename them
    back to GLU/LYS on output. Without that rename-back the prepped file's
    closure is not re-detected and relaxation raises a spurious CyclizationError,
    so this fixture reaching ``success`` is itself the regression check.
    """
    raw = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")
    prep_dir = tmp_path_factory.mktemp("qc_prep_somatostatin")
    prepped = _prep_on_the_fly(
        raw, prep_dir / "example_lactam_somatostatin_1XY4_prepped.cif"
    )
    out = tmp_path_factory.mktemp("qc_relax_somatostatin")
    return _relax(
        prepped, out, "somatostatin",
        config=_small_config_gaff(),
        capture_premin=False,
    )


@pytest.fixture(params=["1YCR", "3P8F", "cyclosporin", "somatostatin"])
def relaxed(request) -> RelaxedExample:
    """Parametrized access to each relaxed example."""
    return request.getfixturevalue(f"_relaxed_{request.param.lower()}")


# ---------------------------------------------------------------------------
# QC tests
# ---------------------------------------------------------------------------

# Two identical prep+relax runs must agree far more tightly than this. The
# defect it guards against (unseeded hydrogen placement) produced a spread of
# hundreds of kJ/mol, so a 0.1 kJ/mol bound catches any regression by a wide
# margin while tolerating any last-bit platform float noise.
ENERGY_REPRODUCIBILITY_TOL_KJ = 0.1


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_relaxation_energy_is_reproducible(tmp_path_factory):
    """The same input must give the same minimized energy on every run.

    Regression guard for the whole prep+relax chain. Hydrogen placement used to
    draw from an unseeded RNG and minimize hydrogens on a non-deterministic GPU
    platform, so identical input yielded a different structure — and a different
    minimized energy (1YCR was seen spanning ~600 kJ/mol) — on each run. That is
    a defect in a tool whose output is a QC measurement. Prep is now seeded and
    its hydrogen minimization pinned to the deterministic Reference platform, and
    the CUDA main minimization is itself bit-deterministic from a fixed input, so
    the end-to-end energy must now be stable.

    Uses 1YCR, whose PDBFixer prep path was the one that drifted (the cyclic
    path was already reproducible), and preps *independently* each iteration so
    the hydrogen-placement RNG and platform are genuinely re-exercised.
    """
    raw = DATA_DIR / "example_linear_p53_1YCR.pdb"
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")

    energies = []
    for i in range(2):
        work = tmp_path_factory.mktemp(f"qc_repro_{i}")
        prepped = _prep_on_the_fly(raw, work / "prepped.cif")
        result = ImplicitRelaxation(_small_config()).run(
            prepped, work, sample_id=f"repro{i}"
        )
        assert result.success, f"run {i} failed: {result.error_message}"
        assert result.potential_energy_minimized is not None
        energies.append(float(result.potential_energy_minimized))

    spread = abs(energies[0] - energies[1])
    assert spread <= ENERGY_REPRODUCIBILITY_TOL_KJ, (
        f"minimized energy is not reproducible: {energies[0]:.4f} vs "
        f"{energies[1]:.4f} kJ/mol (spread {spread:.4f} > "
        f"{ENERGY_REPRODUCIBILITY_TOL_KJ} kJ/mol) — hydrogen placement "
        f"determinism has regressed"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_is_reproducible_by_default_and_random_when_opted_out(prepped_example_cif,
                                                                 tmp_path_factory):
    """MD is deterministic with the default seed and stochastic with seed=None.

    Guards the seeding of the two MD randomness sources the minimize-only path
    never touches: the Langevin thermostat (``integrator.setRandomNumberSeed``)
    and the initial Maxwell–Boltzmann velocities (``setVelocitiesToTemperature``).
    Both were unseeded, so the default (MD-on) pipeline was nondeterministic.

    With ``random_seed`` fixed, two short MD runs from the same input must land
    in the same place; with ``random_seed=None`` they must be free to diverge
    (that is the whole point of the opt-out). The divergence assertion uses a
    loose floor so it is not flaky — independent 2 ps Langevin trajectories from
    freshly drawn velocities separate by far more than this.
    """
    def md_run(seed):
        cfg = RelaxationConfig(
            min_steps_initial=50, min_steps_restrained=20, min_steps_final=50,
            md_duration_ps=2.0, md_save_interval_ps=2.0,
            device="cuda", small_molecules=None, random_seed=seed,
        )
        out = tmp_path_factory.mktemp("qc_md")
        result = ImplicitRelaxation(cfg).run(Path(prepped_example_cif), out, sample_id="md")
        assert result.success, f"MD run failed: {result.error_message}"
        assert result.rmsd_md_final is not None
        return float(result.rmsd_md_final)

    seeded = [md_run(1), md_run(1)]
    assert abs(seeded[0] - seeded[1]) <= 1e-4, (
        f"MD is not reproducible with a fixed seed: {seeded[0]:.6f} vs "
        f"{seeded[1]:.6f} Å — Langevin/velocity seeding has regressed"
    )

    unseeded = [md_run(None), md_run(None)]
    assert abs(unseeded[0] - unseeded[1]) > 1e-3, (
        f"random_seed=None did not restore stochastic MD: {unseeded[0]:.6f} vs "
        f"{unseeded[1]:.6f} Å — the opt-out is not wired through"
    )


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


# ---------------------------------------------------------------------------
# MD-path structural QC
#
# The checks above are all minimize-only (md_duration_ps=0). MD is a distinct
# code path — a Langevin integrator, initial velocities, and for cyclic peptides
# a dihedral-restrained warmup — none of which minimization exercises. These
# tests run a short MD on each example and assert the final frame is physically
# sound. Unlike minimization, MD legitimately samples away from the input, so we
# do NOT bound RMSD-to-input here (a free peptide can drift several Å in a few
# ps); instead "did it blow up" is caught by finite energy/coords, no fused
# atoms, no broken bonds, and no stereocenter inversion.
# ---------------------------------------------------------------------------

#: Short MD used for the QC pass. Long enough to exercise the integrator, the
#: velocity initialisation and (for cyclic peptides) the dihedral warmup; short
#: enough to keep the GPU cost bounded.
MD_DURATION_PS = 5.0

#: name -> (bundled filename, small_molecules mode) for the MD-path examples.
_MD_EXAMPLES = {
    "1YCR": ("example_linear_p53_1YCR.pdb", None),
    "3P8F": ("example_bicyclic_sfti1_3P8F.cif", None),
    "cyclosporin": ("example_ncaa_cyclosporin_1CWA.cif", "auto"),
    "somatostatin": ("example_lactam_somatostatin_1XY4.cif", "auto"),
}


@dataclass
class MDRelaxedExample:
    """Everything the MD-path assertions need for one example."""
    name: str
    minimized_path: Path      # pre-MD (minimized) frame — the sanity baseline
    md_final_path: Path       # final MD frame
    energy_md_avg: float      # mean potential energy over the MD trajectory


def _md_config(small_molecules: Optional[str]) -> RelaxationConfig:
    """Minimize + short-MD config (GPU-friendly step counts)."""
    return RelaxationConfig(
        md_duration_ps=MD_DURATION_PS,
        md_save_interval_ps=MD_DURATION_PS,
        md_temperature_k=300.0,
        min_steps_initial=50,
        min_steps_restrained=20,
        min_steps_final=50,
        device="cuda",
        small_molecules=small_molecules,
    )


@pytest.fixture(scope="session", params=list(_MD_EXAMPLES))
def md_relaxed(request, tmp_path_factory) -> MDRelaxedExample:
    """Prep + minimize + short MD for each example (once per session)."""
    name = request.param
    filename, small_molecules = _MD_EXAMPLES[name]
    raw = DATA_DIR / filename
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")

    prep_dir = tmp_path_factory.mktemp(f"qc_md_prep_{name}")
    prepped = _prep_on_the_fly(raw, prep_dir / f"{name}_prepped.cif")
    out = tmp_path_factory.mktemp(f"qc_md_relax_{name}")

    result = ImplicitRelaxation(_md_config(small_molecules)).run(prepped, out, sample_id=name)
    assert result.success, f"{name} MD relaxation failed: {result.error_message}"
    assert result.md_final_structure_path is not None, f"{name}: no MD frame written"
    assert result.minimized_structure_path is not None
    assert result.potential_energy_md_avg is not None, f"{name}: no MD energy"
    return MDRelaxedExample(
        name=name,
        minimized_path=Path(result.minimized_structure_path),
        md_final_path=Path(result.md_final_structure_path),
        energy_md_avg=float(result.potential_energy_md_avg),
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_energy_finite_and_sane(md_relaxed: MDRelaxedExample):
    """MD check 1: mean trajectory energy is finite and in the sane range."""
    e = md_relaxed.energy_md_avg
    assert math.isfinite(e), f"{md_relaxed.name}: non-finite MD energy {e}"
    assert ENERGY_MIN_KJ < e < ENERGY_MAX_KJ, (
        f"{md_relaxed.name}: MD mean energy {e:.1f} kJ/mol out of sane range "
        f"— trajectory likely blew up"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_coordinates_finite(md_relaxed: MDRelaxedExample):
    """MD check 2: no NaN/inf coordinates in the final MD frame."""
    assert _all_coords_finite(md_relaxed.md_final_path), (
        f"{md_relaxed.name}: MD final frame contains non-finite coordinates"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_no_egregious_clashes(md_relaxed: MDRelaxedExample):
    """MD check 3: no fused atoms (closest inter-residue heavy pair > 0.8 Å)."""
    min_dist, n_heavy = _min_interresidue_heavy_distance(md_relaxed.md_final_path)
    assert n_heavy > 0, f"{md_relaxed.name}: no heavy atoms in MD frame"
    assert math.isfinite(min_dist), f"{md_relaxed.name}: non-finite MD min distance"
    assert min_dist > MIN_HEAVY_DIST_ANG, (
        f"{md_relaxed.name}: MD final frame has fused atoms — closest "
        f"inter-residue heavy pair is {min_dist:.3f} Å (<= {MIN_HEAVY_DIST_ANG})"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_bonds_not_broken(md_relaxed: MDRelaxedExample):
    """MD check 4: covalent bonds stay intact (perceived from the minimized frame)."""
    pin = _heavy_atom_positions(md_relaxed.minimized_path)
    pmd = _heavy_atom_positions(md_relaxed.md_final_path)
    stretched = []
    for a, b in _perceive_heavy_bonds(pin):
        if a in pmd and b in pmd:
            d = float(np.linalg.norm(pmd[a] - pmd[b]))
            if not (BOND_LENGTH_MIN_ANG <= d <= BOND_LENGTH_MAX_ANG):
                stretched.append((a, b, d))
    assert not stretched, (
        f"{md_relaxed.name}: {len(stretched)} bond(s) stretched/broken during MD, "
        f"e.g. {stretched[0][0]}–{stretched[0][1]} = {stretched[0][2]:.2f} Å"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_no_chirality_inversion(md_relaxed: MDRelaxedExample):
    """MD check 5: no Cα stereocenter inverts during MD (critical for D-residues)."""
    pre = _ca_signed_volumes(md_relaxed.minimized_path)
    post = _ca_signed_volumes(md_relaxed.md_final_path)
    flipped = [
        k for k in set(pre) & set(post)
        if (pre[k] > 0) != (post[k] > 0)
        and abs(pre[k]) >= CHIRALITY_MIN_VOLUME_A3
        and abs(post[k]) >= CHIRALITY_MIN_VOLUME_A3
    ]
    assert not flipped, (
        f"{md_relaxed.name}: {len(flipped)} Cα stereocenter(s) inverted during MD: "
        f"{sorted(flipped)}"
    )


@requires_cuda
@pytest.mark.integration
@pytest.mark.gpu
def test_md_no_missing_heavy_atoms(md_relaxed: MDRelaxedExample):
    """MD check 6: MD neither drops nor adds atoms."""
    pre_total, _ = _heavy_atom_composition(md_relaxed.minimized_path)
    post_total, _ = _heavy_atom_composition(md_relaxed.md_final_path)
    assert pre_total == post_total, (
        f"{md_relaxed.name}: heavy-atom count changed during MD "
        f"({pre_total} → {post_total})"
    )
