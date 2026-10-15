"""Failed energy modes are recorded in ``error_message`` instead of only being logged.

The OpenMM simulation is replaced by a stub so each failure can be injected at a
chosen step (minimisation, MD, subsystem evaluation) without a GPU or a
force-field build.
"""

import openmm
import openmm.app
import pytest
from conftest import EXAMPLE_PDB_PATH

from binding_metrics.metrics import energy


class _StubContext:
    def setPositions(self, positions):
        self.positions = positions

    def reinitialize(self, preserveState=False):
        pass

    def setParameter(self, name, value):
        pass

    def setVelocitiesToTemperature(self, *args):
        pass

    def getState(self, **kwargs):
        return self

    def getPositions(self):
        return self.positions


class _StubSystem:
    def __init__(self, remove_error=None):
        self._forces = []
        self._remove_error = remove_error

    def getNumForces(self):
        return len(self._forces)

    def addForce(self, force):
        self._forces.append(force)

    def removeForce(self, index):
        if self._remove_error is not None:
            raise self._remove_error
        del self._forces[index]


def _stub_simulation(minimize_error=None, step_error=None):
    class _StubSimulation:
        def __init__(self, topology, system, integrator, platform, properties):
            self.context = _StubContext()

        def minimizeEnergy(self, maxIterations=0):
            if minimize_error is not None:
                raise minimize_error

        def step(self, n_steps):
            if step_error is not None:
                raise step_error

    return _StubSimulation


@pytest.fixture
def stubbed_energy(monkeypatch):
    """Patch everything below ``compute_interaction_energy`` that needs OpenMM."""

    systems: list = []
    remove_error: list = [None]

    def fake_create_implicit_system(topology, positions, *args, **kwargs):
        system = _StubSystem(remove_error[0])
        systems.append(system)
        return system, topology, positions, None, []

    monkeypatch.setattr(energy, "_create_implicit_system", fake_create_implicit_system)
    monkeypatch.setattr(energy, "_get_platform", lambda device="cuda": (None, {}))

    def install(minimize_error=None, step_error=None, evaluate=None, remove_restraint_error=None):
        remove_error[0] = remove_restraint_error
        monkeypatch.setattr(openmm.app, "Simulation", _stub_simulation(minimize_error, step_error))
        monkeypatch.setattr(
            energy,
            "_evaluate_subsystem_energies",
            evaluate or (lambda *args, **kwargs: (-10.0, -3.0, -2.0)),
        )
        return systems

    return install


def _run(modes, **kwargs):
    return energy.compute_interaction_energy(
        EXAMPLE_PDB_PATH, peptide_chain="B", receptor_chain="A", modes=modes, **kwargs
    )


class TestModeFailuresAreRecorded:
    def test_success_leaves_error_message_none(self, stubbed_energy):
        stubbed_energy()
        result = _run(("raw", "relaxed", "after_md"))
        assert result["success"] is True
        assert result["error_message"] is None
        for mode in ("raw", "relaxed", "after_md"):
            assert result[f"{mode}_interaction_energy"] == pytest.approx(-5.0)

    def test_minimization_failure_recorded_and_success_kept(self, stubbed_energy):
        stubbed_energy(minimize_error=RuntimeError("minimizer diverged"))
        result = _run(("raw", "relaxed"))
        assert result["success"] is True  # raw energies are still valid
        assert (
            "relaxed: minimization failed: RuntimeError: minimizer diverged"
            in (result["error_message"])
        )
        assert result["raw_interaction_energy"] == pytest.approx(-5.0)
        assert result["relaxed_interaction_energy"] is None

    def test_only_mode_failing_gives_a_reason(self, stubbed_energy):
        stubbed_energy(minimize_error=RuntimeError("minimizer diverged"))
        result = _run(("relaxed",))
        assert result["success"] is False
        assert "minimizer diverged" in result["error_message"]

    def test_after_md_step_failure_recorded(self, stubbed_energy):
        stubbed_energy(step_error=RuntimeError("MD blew up"))
        result = _run(("raw", "after_md"))
        assert result["success"] is True
        assert result["error_message"] == "after_md: RuntimeError: MD blew up"
        assert result["after_md_interaction_energy"] is None

    def test_messages_from_several_steps_are_joined(self, stubbed_energy):
        stubbed_energy(
            minimize_error=RuntimeError("minimizer diverged"),
            step_error=RuntimeError("MD blew up"),
        )
        result = _run(("raw", "relaxed", "after_md"))
        message = result["error_message"]
        assert message.index("relaxed:") < message.index("after_md:")
        assert "minimizer diverged" in message and "MD blew up" in message

    def test_non_finite_evaluation_reason_reaches_error_message(self, stubbed_energy):
        def evaluate(*args, failures=None, **kwargs):
            failures.append("complex energy is not finite")
            return None, None, None

        stubbed_energy(evaluate=evaluate)
        result = _run(("relaxed",))
        assert result["success"] is False
        assert result["error_message"] == "relaxed: complex energy is not finite"

    def test_raw_evaluation_failure_reason_recorded(self, stubbed_energy):
        def evaluate(*args, failures=None, **kwargs):
            failures.append("peptide or receptor energy is not finite")
            return -1.0, None, None

        stubbed_energy(evaluate=evaluate)
        result = _run(("raw",))
        assert result["success"] is False
        assert result["error_message"] == "raw: peptide or receptor energy is not finite"

    def test_no_new_keys(self, stubbed_energy):
        stubbed_energy(minimize_error=RuntimeError("x"))
        result = _run(("raw", "relaxed"))
        assert set(result) == {
            "sample_id",
            "success",
            "error_message",
            "num_contacts",
            "num_close_contacts",
            "raw_interaction_energy",
            "raw_e_complex",
            "raw_e_peptide",
            "raw_e_receptor",
            "relaxed_interaction_energy",
            "relaxed_e_complex",
            "relaxed_e_peptide",
            "relaxed_e_receptor",
        }

    def test_top_level_failure_message_format_is_unchanged(self, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("no template for residue XYZ")

        monkeypatch.setattr(energy, "_create_implicit_system", boom)
        result = _run(("raw",))
        assert result["success"] is False
        assert result["error_message"] == "ValueError: no template for residue XYZ"


class TestBackboneRestraintIsDetached:
    """A restraint left in the system would be counted in E_complex of later modes."""

    def test_removed_after_successful_minimization(self, stubbed_energy):
        systems = stubbed_energy()
        _run(("relaxed",))
        assert systems[0].getNumForces() == 0

    def test_removed_when_minimization_fails(self, stubbed_energy):
        systems = stubbed_energy(minimize_error=RuntimeError("minimizer diverged"))
        _run(("raw", "after_md"))
        assert systems[0].getNumForces() == 0

    def test_failed_removal_is_reported_without_crashing(self, stubbed_energy):
        stubbed_energy(
            minimize_error=RuntimeError("minimizer diverged"),
            remove_restraint_error=openmm.OpenMMException("force index out of range"),
        )
        result = _run(("raw", "relaxed"))
        assert result["success"] is True
        assert "could not remove the backbone restraint" in result["error_message"]
        assert "minimizer diverged" in result["error_message"]


class TestEvaluateSubsystemEnergiesFailures:
    @staticmethod
    def _simulation(energy_kj_mol):
        class _State:
            def getPotentialEnergy(self):
                return energy_kj_mol * openmm.unit.kilojoules_per_mole

        class _Context:
            def setPositions(self, positions):
                pass

            def getState(self, getEnergy=False):
                return _State()

        class _Sim:
            context = _Context()

        return _Sim()

    def test_non_finite_complex_energy(self):
        failures: list[str] = []
        out = energy._evaluate_subsystem_energies(
            self._simulation(float("nan")), None, None, "B", "A", "obc2", "cpu", failures=failures
        )
        assert out == (None, None, None)
        assert failures == ["complex energy is not finite"]

    def test_exception_is_recorded_with_its_type(self, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("No template found for residue 3 (XYZ)")

        monkeypatch.setattr(energy, "_extract_chain", boom)
        failures: list[str] = []
        out = energy._evaluate_subsystem_energies(
            self._simulation(-100.0), None, None, "B", "A", "obc2", "cpu", failures=failures
        )
        assert out == (None, None, None)
        assert failures == ["ValueError: No template found for residue 3 (XYZ)"]

    def test_failures_argument_is_optional(self):
        out = energy._evaluate_subsystem_energies(
            self._simulation(float("nan")), None, None, "B", "A", "obc2", "cpu"
        )
        assert out == (None, None, None)


class TestGetPlatformFallback:
    def test_missing_cuda_plugin_falls_back_to_cpu(self, monkeypatch):
        real = openmm.Platform.getPlatformByName

        def fake(name):
            if name == "CUDA":
                raise openmm.OpenMMException('There is no registered Platform called "CUDA"')
            return real(name)

        monkeypatch.setattr(openmm.Platform, "getPlatformByName", staticmethod(fake))
        platform, properties = energy._get_platform("cuda")
        assert platform.getName() == "CPU"
        assert properties == {}

    def test_unrelated_errors_are_not_swallowed(self, monkeypatch):
        def fake(name):
            raise RuntimeError("driver crashed")

        monkeypatch.setattr(openmm.Platform, "getPlatformByName", staticmethod(fake))
        with pytest.raises(RuntimeError, match="driver crashed"):
            energy._get_platform("cuda")
