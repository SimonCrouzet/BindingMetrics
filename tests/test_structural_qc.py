"""Structural-integrity / QC checks on relaxation output.

binding-metrics is a structure-QC tool: a minimization can report
``success=True`` with a finite energy while having blown up the geometry
(exploded coordinates, fused atoms, NaNs). The plain relaxation tests only
assert ``result.success`` and ``energy is not None``; they would not catch a
structurally broken result. This module runs the QC of
``binding_metrics.protocols.qc`` on real relaxed structures. The checks, their
limits and the values measured on the examples are documented there; the
relaxer runs the same code and stores the outcome in ``result.qc``.

The bundled examples are chosen to span the peptide feature space rather than
to repeat it:

    1YCR         p53 / MDM2            linear, all-standard residues
    3P8F         SFTI-1 / matriptase   bicyclic: head-to-tail *and* disulfide
    cyclosporin  CsA / cyclophilin A   head-to-tail macrocycle + D-alanine +
                                       N-methylation (MLE/MVA/SAR) + exotic
                                       residues auto-parameterized by GAFF
                                       (BMT/ABA)
    somatostatin 1XY4                  lactam, disulfide, D-Trp, GAFF residue

For each we run a short minimize-only relaxation on CUDA, then check the
minimized CIF file against the relaxation input (the file-level comparison also
guards ``save_cif``), and check that the in-run ``result.qc`` passed. A second
group of tests does the same for the final frame of a short MD run.

The relaxation results are reproducible run-to-run on a given machine: prep
seeds hydrogen placement and pins its minimization to the deterministic
Reference platform, and the CUDA main minimization is bit-deterministic from a
fixed input (see ``test_relaxation_energy_is_reproducible``). Absolute values
can still shift across GPU models and force-field versions, so the limits in
``qc`` stay wide with intent. Do not tighten them.

The chirality check earned its place immediately: it caught a real prep bug in
which ``addHydrogens`` stranded a C-alpha hydrogen on the wrong face, which then
forced the minimizer to invert that stereocenter. See
``core.system.repair_ca_hydrogen_chirality`` for the mechanism. Keep the check
strict; weakening it would defeat the inversion it exists to catch.
Cyclosporin's D-alanine C-alpha carries the opposite sign to every L-residue,
and the check reads it correctly without being told it is D.

Runs are guarded by ``requires_cuda`` and skip gracefully without a GPU. Systems
are kept small (minimize-only, tiny step counts) to share an 8 GB GPU.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
from conftest import requires_cuda

from binding_metrics.protocols import qc
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class RelaxedExample:
    """Bundle of everything the QC assertions need for one relaxed example."""

    name: str
    input_path: Path  # the (prepped) structure handed to the relaxer
    minimized_path: Path  # the minimized CIF the relaxer wrote
    energy_min: float  # potential_energy_minimized (kJ/mol)
    energy_pre: Optional[float]  # pre-minimization energy, or None if not captured
    qc: Optional[dict]  # result.qc of the relaxation run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_check(name: str, check: dict) -> None:
    """Fail with the check's own message; a check with nothing to test also fails."""
    assert check["evaluated"], f"{name}: check could not be evaluated ({check['reason']})"
    assert check["passed"], f"{name}: {check['detail']} (rule: {check['limit']})"


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
    except Exception:  # noqa: BLE001 - best effort by design; None tells the caller it failed
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
        qc=result.qc,
    )


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
    prepped = _prep_on_the_fly(raw, prep_dir / "example_ncaa_cyclosporin_1CWA_prepped.cif")
    out = tmp_path_factory.mktemp("qc_relax_cyclosporin")
    return _relax(
        prepped,
        out,
        "cyclosporin",
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
    prepped = _prep_on_the_fly(raw, prep_dir / "example_lactam_somatostatin_1XY4_prepped.cif")
    out = tmp_path_factory.mktemp("qc_relax_somatostatin")
    return _relax(
        prepped,
        out,
        "somatostatin",
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
        result = ImplicitRelaxation(_small_config()).run(prepped, work, sample_id=f"repro{i}")
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
def test_md_is_reproducible_by_default_and_random_when_opted_out(
    prepped_example_cif, tmp_path_factory
):
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
            min_steps_initial=50,
            min_steps_restrained=20,
            min_steps_final=50,
            md_duration_ps=2.0,
            md_save_interval_ps=2.0,
            device="cuda",
            small_molecules=None,
            random_seed=seed,
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
def test_energy_finite_and_did_not_increase(relaxed: RelaxedExample):
    """Check 1: minimized energy is finite, sane, and not higher than pre-min."""
    _assert_check(relaxed.name, qc.check_energy(relaxed.energy_min, relaxed.energy_pre))


@requires_cuda
@pytest.mark.integration
def test_structure_did_not_explode(relaxed: RelaxedExample):
    """Check 2: heavy-atom RMSD to the input is finite and bounded."""
    _assert_check(relaxed.name, qc.check_rmsd(relaxed.input_path, relaxed.minimized_path))


@requires_cuda
@pytest.mark.integration
def test_coordinates_finite(relaxed: RelaxedExample):
    """Check 3: no NaN/inf coordinates in the minimized structure."""
    _assert_check(relaxed.name, qc.check_coordinates_finite(relaxed.minimized_path))


@requires_cuda
@pytest.mark.integration
def test_no_egregious_clashes(relaxed: RelaxedExample):
    """Check 4: no heavy-atom pair in different residues is fused."""
    _assert_check(relaxed.name, qc.check_min_heavy_distance(relaxed.minimized_path))


@requires_cuda
@pytest.mark.integration
def test_bond_lengths_preserved(relaxed: RelaxedExample):
    """Check 5: no covalent bond was stretched/broken by minimization."""
    _assert_check(relaxed.name, qc.check_bond_lengths(relaxed.input_path, relaxed.minimized_path))


@requires_cuda
@pytest.mark.integration
def test_chirality_preserved(relaxed: RelaxedExample):
    """Check 6: minimization did not invert any C-alpha stereocenter (D-residues included)."""
    _assert_check(relaxed.name, qc.check_chirality(relaxed.input_path, relaxed.minimized_path))


@requires_cuda
@pytest.mark.integration
def test_no_missing_heavy_atoms(relaxed: RelaxedExample):
    """Check 7: minimization dropped/added/renamed no heavy atom."""
    _assert_check(relaxed.name, qc.check_composition(relaxed.input_path, relaxed.minimized_path))


@requires_cuda
@pytest.mark.integration
def test_run_qc_block_passes(relaxed: RelaxedExample):
    """The QC the relaxer attaches to its own result agrees: nothing was flagged."""
    assert relaxed.qc is not None
    assert relaxed.qc["passed"] is True, relaxed.qc["failed"]
    for name, check in relaxed.qc["checks"].items():
        assert check["passed"], f"{name}: {check['detail']}"


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
    minimized_path: Path  # pre-MD (minimized) frame — the sanity baseline
    md_final_path: Path  # final MD frame
    energy_md_avg: float  # mean potential energy over the MD trajectory
    qc: Optional[dict]  # result.qc of the relaxation run (has an "md_final" block)


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
        qc=result.qc,
    )


@requires_cuda
@pytest.mark.integration
def test_md_energy_finite_and_sane(md_relaxed: MDRelaxedExample):
    """MD check 1: mean trajectory energy is finite and in the sane range."""
    _assert_check(md_relaxed.name, qc.check_energy(md_relaxed.energy_md_avg))


@requires_cuda
@pytest.mark.integration
def test_md_coordinates_finite(md_relaxed: MDRelaxedExample):
    """MD check 2: no NaN/inf coordinates in the final MD frame."""
    _assert_check(md_relaxed.name, qc.check_coordinates_finite(md_relaxed.md_final_path))


@requires_cuda
@pytest.mark.integration
def test_md_no_egregious_clashes(md_relaxed: MDRelaxedExample):
    """MD check 3: no fused atoms in the final MD frame."""
    _assert_check(md_relaxed.name, qc.check_min_heavy_distance(md_relaxed.md_final_path))


@requires_cuda
@pytest.mark.integration
def test_md_bonds_not_broken(md_relaxed: MDRelaxedExample):
    """MD check 4: covalent bonds stay intact (perceived from the minimized frame)."""
    check = qc.check_bond_lengths(md_relaxed.minimized_path, md_relaxed.md_final_path)
    _assert_check(md_relaxed.name, check)


@requires_cuda
@pytest.mark.integration
def test_md_no_chirality_inversion(md_relaxed: MDRelaxedExample):
    """MD check 5: no C-alpha stereocenter inverts during MD (critical for D-residues)."""
    check = qc.check_chirality(md_relaxed.minimized_path, md_relaxed.md_final_path)
    _assert_check(md_relaxed.name, check)


@requires_cuda
@pytest.mark.integration
def test_md_no_missing_heavy_atoms(md_relaxed: MDRelaxedExample):
    """MD check 6: MD neither drops, adds nor renames heavy atoms."""
    check = qc.check_composition(md_relaxed.minimized_path, md_relaxed.md_final_path)
    _assert_check(md_relaxed.name, check)


@requires_cuda
@pytest.mark.integration
def test_md_run_qc_block_passes(md_relaxed: MDRelaxedExample):
    """The in-run QC of the MD frame agrees, and there is no RMSD bound for MD."""
    assert md_relaxed.qc is not None and md_relaxed.qc["passed"] is True
    md_block = md_relaxed.qc["md_final"]
    assert md_block["passed"] is True, md_block["failed"]
    assert "rmsd" not in md_block["checks"]
