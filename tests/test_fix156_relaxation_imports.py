"""Every method of ``ImplicitRelaxation`` that uses OpenMM binds it first, in any order.

``protocols/relaxation.py`` does not import OpenMM at module level: ``_import_openmm`` binds the
module globals ``openmm``, ``app``, ``unit`` and ``PDBxFile`` on first use. A method that used
them without calling it passed whenever an earlier test in the same process had bound them and
raised ``NameError`` when it ran alone (#156). The behavioural tests run each such method in a
fresh interpreter; the source test names every method that has to call ``_import_openmm``.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("openmm")

REPO = Path(__file__).resolve().parents[1]
RELAXATION = REPO / "src" / "binding_metrics" / "protocols" / "relaxation.py"

#: The module globals that only ``_import_openmm`` binds.
LAZY_NAMES = frozenset({"openmm", "app", "unit", "PDBxFile"})

#: Builds a two-residue backbone (N, CA, C) as a Topology, a System of free particles and the
#: positions, with OpenMM imported by the script itself: the module under test has bound nothing.
TINY_SYSTEM = """
import openmm
from openmm import app

topology = app.Topology()
chain = topology.addChain(id="A")
for number in (1, 2):
    residue = topology.addResidue("ALA", chain)
    for name in ("N", "CA", "C"):
        topology.addAtom(name, app.element.carbon, residue)
coordinates_nm = [
    (0.00, 0.00, 0.00), (0.15, 0.00, 0.00), (0.20, 0.14, 0.00),
    (0.33, 0.13, 0.05), (0.40, 0.25, 0.12), (0.55, 0.24, 0.20),
]
positions = [openmm.Vec3(*xyz) for xyz in coordinates_nm]
system = openmm.System()
for _ in positions:
    system.addParticle(12.0)



def platform_for_a_tiny_system():
    # CUDA when a context can be made on it, as the suite does (conftest BEST_PLATFORM)
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        openmm.Context(openmm.System(), openmm.VerletIntegrator(0.001), platform)
        return platform
    except Exception:
        return openmm.Platform.getPlatformByName("CPU")


import binding_metrics.protocols.relaxation as relaxation
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

assert not any(hasattr(relaxation, name) for name in ("openmm", "app", "unit", "PDBxFile")), (
    "the module bound OpenMM at import time: the test would prove nothing"
)
"""


def run_fresh(body: str) -> subprocess.CompletedProcess:
    script = TINY_SYSTEM + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )


def assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok"), result.stdout + result.stderr


@pytest.mark.slow
class TestAMethodRunsAloneInAFreshInterpreter:
    def test_add_restraints(self):
        assert_ok(
            run_fresh(
                """
                relaxer = ImplicitRelaxation(RelaxationConfig())
                index = relaxer._add_restraints(system, topology, positions, backbone_only=True)
                force = system.getForce(index)
                assert force.getNumParticles() == 6, force.getNumParticles()
                print("ok")
                """
            )
        )

    def test_run_cyclic_warmup(self):
        assert_ok(
            run_fresh(
                """
                platform = platform_for_a_tiny_system()
                integrator = openmm.LangevinMiddleIntegrator(300, 1.0, 0.002)
                simulation = app.Simulation(topology, system, integrator, platform)
                simulation.context.setPositions(positions)
                relaxer = ImplicitRelaxation(RelaxationConfig())
                relaxer._run_cyclic_warmup(
                    system, simulation, topology, positions, "A", (1, 2, 3, 4), warmup_ps=0.01
                )
                assert system.getNumForces() == 0, "the warm-up restraints were not removed"
                print("ok")
                """
            )
        )

    def test_get_platform(self):
        assert_ok(
            run_fresh(
                """
                relaxer = ImplicitRelaxation(RelaxationConfig(device="cpu"))
                platform, _ = relaxer._get_platform()
                assert platform.getName() == "CPU", platform.getName()
                print("ok")
                """
            )
        )

    def test_identify_chains(self):
        assert_ok(
            run_fresh(
                """
                relaxer = ImplicitRelaxation(RelaxationConfig())
                peptide, receptor = relaxer._identify_chains(topology)
                assert (peptide, receptor) == ("A", None), (peptide, receptor)
                print("ok")
                """
            )
        )

    def test_a_method_binds_the_module_names_once_for_the_later_calls(self):
        assert_ok(
            run_fresh(
                """
                relaxer = ImplicitRelaxation(RelaxationConfig())
                relaxer._add_restraints(system, topology, positions)
                assert relaxation.openmm is openmm and relaxation.unit is not None
                assert relaxation.app is app and relaxation.PDBxFile is not None
                print("ok")
                """
            )
        )

    def test_an_instance_unpickled_after_use_still_binds_the_names(self):
        """A relaxer sent to a worker process carries its flag but not the module globals."""
        assert_ok(
            run_fresh(
                """
                import pickle

                used = ImplicitRelaxation(RelaxationConfig())
                used._openmm_imported = True  # as after a run in the parent process
                clone = pickle.loads(pickle.dumps(used))
                assert clone._openmm_imported
                index = clone._add_restraints(system, topology, positions)
                assert system.getForce(index).getNumParticles() == 6
                print("ok")
                """
            )
        )


def _functions_and_methods(tree):
    """The functions of the module and the methods of its classes (not the nested functions)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )


def _methods_that_use_the_lazy_names():
    """``(name, line, line of the first _import_openmm call, line of the first use)`` of each
    function that loads one of the lazy names without binding it itself."""
    tree = ast.parse(RELAXATION.read_text(encoding="utf-8"))
    rows = []
    for function in _functions_and_methods(tree):
        if function.name == "_import_openmm":
            continue  # the function that binds them
        bound_here = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        uses, import_calls = [], []
        for node in ast.walk(function):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bound_here.update(
                    (alias.asname or alias.name).split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound_here.add(node.id)
            elif isinstance(node, ast.Name) and node.id in LAZY_NAMES:
                uses.append(node.lineno)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_import_openmm"
            ):
                import_calls.append(node.lineno)
        if uses and not LAZY_NAMES & bound_here:
            rows.append(
                (function.name, function.lineno, min(import_calls, default=None), min(uses))
            )
    return rows


AUDIT_ROWS = _methods_that_use_the_lazy_names()


class TestEveryUserOfTheLazyNamesBindsThem:
    def test_the_audit_finds_the_known_users(self):
        names = {name for name, *_ in AUDIT_ROWS}
        assert {"_setup_system", "_add_restraints", "_run_cyclic_warmup", "run"} <= names

    @pytest.mark.parametrize(
        "name, line, import_line, first_use",
        AUDIT_ROWS,
        ids=[row[0] for row in AUDIT_ROWS],
    )
    def test_the_method_calls_import_openmm_before_its_first_use(
        self, name, line, import_line, first_use
    ):
        assert import_line is not None, (
            f"{RELAXATION.name}:{line} {name} uses {sorted(LAZY_NAMES)} without calling "
            "self._import_openmm()"
        )
        assert import_line < first_use, (
            f"{RELAXATION.name}:{line} {name} calls self._import_openmm() at line {import_line}, "
            f"after its first use at line {first_use}"
        )
