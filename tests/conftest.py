"""Pytest configuration and shared fixtures."""

import sys
import warnings
from pathlib import Path

import pytest

# Allow `from conftest import ...` in test files
sys.path.insert(0, str(Path(__file__).parent))

from _tiers import (  # noqa: E402 - needs the line above
    GPU,
    HEAVY,
    TIER_MARKERS,
    TierCounts,
    TierRequest,
    describe,
    is_named,
    kind_of,
    named_targets,
)

# ---------------------------------------------------------------------------
# GPU / CUDA availability
# ---------------------------------------------------------------------------

try:
    # OpenMM 8.2 emits DeprecationWarning ("builtin type SwigPyPacked has no __module__
    # attribute") while its SWIG extension is created. Where that warning is an error
    # (python -W error::DeprecationWarning) the interpreter segfaults inside the import, so
    # the import is done with the warning ignored and the suite can run with warnings as errors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import openmm as _openmm

    _openmm.Platform.getPlatformByName("CUDA")
    HAS_CUDA = True
except Exception:  # noqa: BLE001 - probe: whatever the failure, there is no usable CUDA here
    HAS_CUDA = False


def requires_cuda(item):
    """Mark a test (or class) as needing a real CUDA device.

    Applies both halves deliberately:

    - the ``gpu`` marker, so that neither the default tier nor ``--full`` runs it
      and ``--gpu`` (or ``-m gpu``) selects it. This is what makes the GPU tier
      addressable as a set — CI cannot run these, but it can still collect them
      to prove they import.
    - ``skipif``, so that selecting the gpu tier on a machine without a device
      skips cleanly instead of erroring.

    Using the marker alone would fail on a GPU-less machine that asked for the
    tier; using skipif alone (the previous behaviour) left these tests selected
    by every "not gpu" run, reporting as skips that looked like passes and
    escaping the collect-only guard entirely.
    """
    item = pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")(item)
    return pytest.mark.gpu(item)


# Best available OpenMM platform: CUDA if available, otherwise CPU
BEST_PLATFORM = "CUDA" if HAS_CUDA else "CPU"


# ---------------------------------------------------------------------------
# Options: the test tiers and the guard against untagged slow tests
# ---------------------------------------------------------------------------

# The fast tier leaves out whatever carries one of these markers (docs/testing.md), so a test
# that grows slow and has none of them silently lengthens the everyday run. The guard lists such
# tests at the end of a session. It only reports unless --strict-slow is given, because a duration
# depends on the load of the machine (a run with -n inflates the times of the heavy tests).
SPEED_MARKERS = TIER_MARKERS
SLOW_CALL_SECONDS = 2.0


def pytest_addoption(parser):
    tiers = parser.getgroup("tiers", "test tiers (rules in tests/_tiers.py)")
    tiers.addoption(
        "--full",
        action="store_true",
        default=False,
        help="also run the slow, integration and openfold tests: every test but the GPU ones. "
        "Without it only the tests that carry none of slow, integration, openfold and gpu run",
    )
    tiers.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="also run the tests that carry the gpu marker (they skip without a CUDA device); "
        "--full --gpu runs every test",
    )
    tiers.addoption(
        "--tier-named",
        action="store_true",
        default=False,
        help="apply the tier to the test files and node ids named on the command line too, "
        "which otherwise run whatever their tier (for a script that names the files to run)",
    )
    guard = parser.getgroup("slow-guard", "untagged slow tests")
    guard.addoption(
        "--strict-slow",
        action="store_true",
        default=False,
        help="fail the run when a test without a speed marker (slow, integration, openfold, gpu) "
        "takes --slow-seconds or more",
    )
    guard.addoption(
        "--slow-seconds",
        type=float,
        default=SLOW_CALL_SECONDS,
        metavar="SECONDS",
        help="duration of the call of a test from which it should carry a speed marker "
        f"(default {SLOW_CALL_SECONDS})",
    )


# ---------------------------------------------------------------------------
# Test tiers: the fast tier by default, --full and --gpu for the rest
# ---------------------------------------------------------------------------


class _TierSelection:
    """Leaves out of the run the tests that its tier does not include (tests/_tiers.py).

    The hook wraps the others, so it sees the tests that ``-k`` and ``-m`` leave and counts what
    the tier holds back from those. A pytest-xdist worker collects and applies the same rule to
    the same command line, so every worker ends with the same list; the controller never collects,
    so it prints the line about the tier from the counts that the workers hand over at the end.
    """

    def __init__(self, config):
        self.request = TierRequest(
            full=config.getoption("--full"),
            gpu=config.getoption("--gpu"),
            tier_named=config.getoption("--tier-named"),
            markexpr=config.option.markexpr,
            named=named_targets(config.args, config.invocation_params.dir),
        )
        self.counts: TierCounts | None = None
        self.printed = False

    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection_modifyitems(self, config, items):
        collected = len(items)
        named = set()
        if self.request.named and not self.request.chosen_by_hand:
            named = {id(i) for i in items if is_named(str(i.path), i.nodeid, self.request.named)}
        outcome = yield
        if outcome.excinfo is not None:
            return
        kept, held = [], {HEAVY: [], GPU: []}
        for item in items:
            kind = kind_of(mark.name for mark in item.iter_markers())
            if self.request.holds_back(kind, id(item) in named):
                held[kind].append(item)
            else:
                kept.append(item)
        deselected = held[HEAVY] + held[GPU]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = kept
        self.counts = TierCounts(collected, len(items), len(held[HEAVY]), len(held[GPU]))
        if hasattr(config, "workeroutput"):
            config.workeroutput["tier_counts"] = self.counts.as_dict()

    def pytest_report_collectionfinish(self, config):
        if self.counts is None or hasattr(config, "workerinput"):
            return None
        self.printed = True
        return describe(self.request, self.counts)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        counts = (getattr(node, "workeroutput", None) or {}).get("tier_counts")
        if counts and self.counts is None:
            self.counts = TierCounts(**counts)

    def pytest_terminal_summary(self, terminalreporter, config):
        if self.counts is not None and not self.printed and not hasattr(config, "workerinput"):
            self.printed = True
            terminalreporter.write_line(describe(self.request, self.counts))


# ---------------------------------------------------------------------------
# Guard against untagged slow tests
# ---------------------------------------------------------------------------


class _SlowGuard:
    """Collects the tests that took long and carry no speed marker, from their reports.

    The reports are what a pytest-xdist worker sends to the controller, which is the process
    that runs this plugin, so the list is complete with ``-n``. Only the call of a test counts:
    a fixture shared by a module is billed to the first test that uses it, and tagging that test
    would only move the cost to the next one. The markers are read from ``report.keywords``,
    which has the name of every marker of the test, its class and its module.
    """

    def __init__(self, config):
        self.strict = config.getoption("--strict-slow")
        self.threshold_seconds = config.getoption("--slow-seconds")
        self.untagged: dict[str, float] = {}

    def pytest_runtest_logreport(self, report):
        if report.when != "call" or report.outcome not in ("passed", "failed"):
            return
        if report.duration < self.threshold_seconds:
            return
        if any(marker in report.keywords for marker in SPEED_MARKERS):
            return
        self.untagged[report.nodeid] = report.duration

    def pytest_terminal_summary(self, terminalreporter):
        if not self.untagged:
            return
        terminalreporter.section("untagged slow tests")
        count = len(self.untagged)
        noun, pronoun = ("test", "it") if count == 1 else ("tests", "them")
        terminalreporter.write_line(
            f"{count} {noun} took {self.threshold_seconds:g} s or more without any of "
            f"{', '.join(SPEED_MARKERS)}, so the fast tier does not leave {pronoun} out. "
            "scripts/tag_slow_tests.py tags a list of node ids."
        )
        for nodeid, seconds in sorted(self.untagged.items(), key=lambda item: -item[1]):
            terminalreporter.write_line(f"{seconds:8.2f} s  {nodeid}")
        if self.strict:
            terminalreporter.write_line("--strict-slow: the run fails.", red=True)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.strict and self.untagged and exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    config.pluginmanager.register(_TierSelection(config), "test-tiers")
    # A pytest-xdist worker has no terminal; its reports reach the controller's guard.
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(_SlowGuard(config), "slow-guard")


# ---------------------------------------------------------------------------
# Area tags
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Tag every test with the areas of the code its file exercises.

    ``pytest -m poses`` or ``-m "poses or preflight"`` then runs a part of the suite. The areas
    come from the imports of the test file (tests/_areas.py), so nothing is written by hand and a
    new test file is tagged when it is collected. ``tryfirst`` puts the tags on before pytest
    applies ``-m``. Each area is registered in pyproject.toml, which ``--strict-markers`` needs.
    """
    from _areas import areas_of_test_file

    for item in items:
        for area in sorted(areas_of_test_file(Path(str(item.path)))):
            item.add_marker(area)


@pytest.fixture(autouse=True)
def _restore_package_logging():
    """Undo what a CLI ``main()`` run inside a test leaves on the package loggers.

    ``configure_logging`` attaches console handlers and sets a level; without this
    they would survive into later tests and add lines to their captured output.
    """
    import logging

    tracked = [logging.getLogger(name) for name in ("binding_metrics", "__main__")]
    saved = [(lg, lg.level, list(lg.handlers)) for lg in tracked]
    yield
    for lg, level, handlers in saved:
        lg.setLevel(level)
        for handler in list(lg.handlers):
            if handler not in handlers:
                lg.removeHandler(handler)


# Path to real example PDB for integration tests
EXAMPLE_PDB_PATH = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


@pytest.fixture
def sample_pdb_content() -> str:
    """Minimal valid PDB content with two chains for testing.

    Chain A: Alanine dipeptide (receptor) - with hydrogens
    Chain B: Glycine (ligand) - with hydrogens

    Note: Coordinates are approximate but chemically reasonable.
    """
    return """\
HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ALA A   1      -0.350   0.840   0.420  1.00  0.00           H
ATOM      3  H2  ALA A   1      -0.350  -0.840   0.420  1.00  0.00           H
ATOM      4  H3  ALA A   1      -0.350   0.000  -0.950  1.00  0.00           H
ATOM      5  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      6  HA  ALA A   1       1.780   0.000   1.040  1.00  0.00           H
ATOM      7  C   ALA A   1       2.009   1.240  -0.700  1.00  0.00           C
ATOM      8  O   ALA A   1       1.400   1.700  -1.670  1.00  0.00           O
ATOM      9  CB  ALA A   1       1.986  -1.280  -0.680  1.00  0.00           C
ATOM     10  HB1 ALA A   1       1.660  -1.280  -1.720  1.00  0.00           H
ATOM     11  HB2 ALA A   1       3.070  -1.280  -0.640  1.00  0.00           H
ATOM     12  HB3 ALA A   1       1.610  -2.150  -0.150  1.00  0.00           H
ATOM     13  N   ALA A   2       3.150   1.770  -0.240  1.00  0.00           N
ATOM     14  H   ALA A   2       3.620   1.350   0.550  1.00  0.00           H
ATOM     15  CA  ALA A   2       3.800   2.950  -0.820  1.00  0.00           C
ATOM     16  HA  ALA A   2       3.080   3.450  -1.470  1.00  0.00           H
ATOM     17  C   ALA A   2       4.300   3.900   0.270  1.00  0.00           C
ATOM     18  O   ALA A   2       4.100   3.620   1.450  1.00  0.00           O
ATOM     19  OXT ALA A   2       4.850   4.940  -0.080  1.00  0.00           O
ATOM     20  CB  ALA A   2       4.980   2.540  -1.700  1.00  0.00           C
ATOM     21  HB1 ALA A   2       5.700   3.350  -1.780  1.00  0.00           H
ATOM     22  HB2 ALA A   2       5.470   1.680  -1.240  1.00  0.00           H
ATOM     23  HB3 ALA A   2       4.640   2.280  -2.700  1.00  0.00           H
TER
ATOM     24  N   GLY B   1      10.000  10.000  10.000  1.00  0.00           N
ATOM     25  H1  GLY B   1       9.650  10.840  10.420  1.00  0.00           H
ATOM     26  H2  GLY B   1       9.650   9.160  10.420  1.00  0.00           H
ATOM     27  H3  GLY B   1       9.650  10.000   9.050  1.00  0.00           H
ATOM     28  CA  GLY B   1      11.458  10.000  10.000  1.00  0.00           C
ATOM     29  HA2 GLY B   1      11.780  10.000  11.040  1.00  0.00           H
ATOM     30  HA3 GLY B   1      11.780  10.900   9.480  1.00  0.00           H
ATOM     31  C   GLY B   1      12.009  11.240   9.300  1.00  0.00           C
ATOM     32  O   GLY B   1      11.400  11.700   8.330  1.00  0.00           O
ATOM     33  OXT GLY B   1      13.050  11.800   9.700  1.00  0.00           O
TER
END
"""


@pytest.fixture
def sample_pdb_path(sample_pdb_content: str, tmp_path: Path) -> Path:
    """Create a temporary PDB file for testing."""
    pdb_path = tmp_path / "test_complex.pdb"
    pdb_path.write_text(sample_pdb_content, encoding="utf-8")
    return pdb_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def example_pdb_path() -> Path:
    """Path to real example PDB for integration tests.

    Skips test if the example PDB is not available.
    """
    if not EXAMPLE_PDB_PATH.exists():
        pytest.skip(f"Example PDB not found: {EXAMPLE_PDB_PATH}")
    return EXAMPLE_PDB_PATH


@pytest.fixture
def example_pdb_chains() -> dict:
    """Chain IDs for the bundled example PDB structure (public PDB 1YCR).

    data/example_linear_p53_1YCR.pdb is the MDM2-p53 complex: chain A = receptor (MDM2),
    chain B = peptide (p53). Returns dict with 'ligand' and 'receptor' chain
    IDs. Tests should use these rather than hard-coding chain IDs.
    """
    return {
        "ligand": "B",
        "receptor": ["A"],
    }


@pytest.fixture(scope="session")
def prepped_example_cif(tmp_path_factory) -> Path:
    """Force-field-ready 1YCR, prepped from the bundled raw example on the fly.

    The raw crystal (data/example_linear_p53_1YCR.pdb) has no hydrogens and an
    uncapped terminus, so it can't build an OpenMM system directly. Rather than
    ship a separate pre-prepped file, we run the same PDBFixer prep the pipeline
    uses (once per session) and hand the result to the force-field tests.
    """
    pytest.importorskip("openmm", reason="OpenMM required")
    pytest.importorskip("pdbfixer", reason="pdbfixer required")
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_structure

    raw = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")
    out = tmp_path_factory.mktemp("prep") / "example_linear_p53_1YCR_prepped.cif"
    topology, positions = load_structure(raw)
    topology, positions = prep_structure(
        topology, positions, ph=7.4, keep_water=False, canonicalize=False
    )
    save_structure(topology, positions, out, source_path=raw)
    return out


@pytest.fixture(scope="session")
def prepped_example_pdb(tmp_path_factory) -> Path:
    """Same prepped 1YCR as :func:`prepped_example_cif`, written as PDB.

    The trajectory-based energy functions read their topology with OpenMM's
    ``PDBFile``, which does not accept mmCIF, so they need this variant rather
    than the CIF one. 1YCR is linear and all-standard, so the PDB round-trip
    has nothing to lose here — that is not true of the cyclic examples, which
    is why this fixture is deliberately tied to 1YCR.
    """
    pytest.importorskip("openmm", reason="OpenMM required")
    pytest.importorskip("pdbfixer", reason="pdbfixer required")
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_structure

    raw = EXAMPLE_PDB_PATH
    if not raw.exists():
        pytest.skip(f"bundled example not found: {raw}")
    out = tmp_path_factory.mktemp("prep_pdb") / "example_linear_p53_1YCR_prepped.pdb"
    topology, positions = load_structure(raw)
    topology, positions = prep_structure(
        topology, positions, ph=7.4, keep_water=False, canonicalize=False
    )
    save_structure(topology, positions, out, source_path=raw)
    return out
