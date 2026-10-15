"""A platform is usable when a context can be created on it (#173).

``core/_platform.py`` decides which OpenMM platform a run uses. The unit tests replace the lookup
of a platform and the creation of the probe context, so they do not depend on the plugins of the
machine; the last class runs the real selection on a machine with a CUDA device and checks that
each call site still gets the platform it got before the probe.
"""

import types

import pytest
from conftest import HAS_CUDA, requires_cuda

pytest.importorskip("openmm")

import openmm

from binding_metrics.core import _platform
from binding_metrics.core.simulation import MDSimulation, SimulationConfig
from binding_metrics.metrics import energy

NO_DEVICE = "Error initializing CUDA: CUDA_ERROR_NO_DEVICE (100) at CudaContext.cpp:92"


class FakePlatforms:
    """Stands in for the platforms OpenMM registers and for the probe context.

    ``registered`` names the platforms the lookup finds; ``broken`` maps a name to the error text
    that the creation of a context on it raises. ``probed`` lists the names a context was created
    for, in order.
    """

    def __init__(self, monkeypatch, registered, broken=None):
        self.registered = set(registered)
        self.broken = dict(broken or {})
        self.probed = []
        monkeypatch.setattr(_platform, "_probe_results", {})
        monkeypatch.setattr(_platform, "_registered_platform", self.lookup)
        monkeypatch.setattr(_platform, "_create_probe_context", self.create_context)

    def lookup(self, name):
        if name not in self.registered:
            raise openmm.OpenMMException(f'There is no registered Platform called "{name}"')
        return types.SimpleNamespace(getName=lambda: name)

    def create_context(self, platform):
        name = platform.getName()
        self.probed.append(name)
        if name in self.broken:
            raise openmm.OpenMMException(self.broken[name])


ALL = ("CUDA", "OpenCL", "CPU", "Reference")


class TestProbe:
    def test_a_platform_with_a_context_is_usable(self, monkeypatch):
        FakePlatforms(monkeypatch, ALL)

        assert _platform.usable_platform("CPU") is True
        assert _platform.unusable_reason("CPU") is None

    def test_a_registered_platform_without_a_context_is_not_usable(self, monkeypatch):
        FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})

        assert _platform.usable_platform("CUDA") is False
        assert _platform.unusable_reason("CUDA") == NO_DEVICE

    def test_a_registered_platform_without_a_context_is_reported_once(self, monkeypatch, caplog):
        FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})

        with caplog.at_level("WARNING", logger=_platform.logger.name):
            _platform.usable_platform("CUDA")
            _platform.usable_platform("CUDA")

        reports = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(reports) == 1
        assert "CUDA" in reports[0].getMessage() and NO_DEVICE in reports[0].getMessage()

    def test_a_platform_that_is_not_registered_is_not_usable(self, monkeypatch):
        FakePlatforms(monkeypatch, ("CPU", "Reference"))

        assert _platform.usable_platform("OpenCL") is False
        assert "OpenCL" in _platform.unusable_reason("OpenCL")

    def test_a_lookup_error_that_is_not_an_openmm_one_is_not_swallowed(self, monkeypatch):
        FakePlatforms(monkeypatch, ALL)

        def crash(name):
            raise RuntimeError("driver crashed")

        monkeypatch.setattr(_platform, "_registered_platform", crash)

        with pytest.raises(RuntimeError, match="driver crashed"):
            _platform.usable_platform("CUDA")

    def test_a_probe_error_that_is_not_an_openmm_one_is_not_swallowed(self, monkeypatch):
        FakePlatforms(monkeypatch, ALL)

        def crash(platform):
            raise MemoryError("out of memory")

        monkeypatch.setattr(_platform, "_create_probe_context", crash)

        with pytest.raises(MemoryError):
            _platform.usable_platform("CUDA")


class TestCache:
    def test_a_usable_platform_is_probed_once(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL)

        results = [_platform.usable_platform("CPU") for _ in range(3)]

        assert results == [True, True, True]
        assert fake.probed == ["CPU"]

    def test_an_unusable_platform_is_probed_once(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})

        results = [_platform.usable_platform("CUDA") for _ in range(3)]
        reasons = {_platform.unusable_reason("CUDA") for _ in range(2)}

        assert results == [False, False, False]
        assert reasons == {NO_DEVICE}
        assert fake.probed == ["CUDA"]

    def test_each_platform_has_its_own_result(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})

        assert _platform.usable_platform("CUDA") is False
        assert _platform.usable_platform("CPU") is True
        assert fake.probed == ["CUDA", "CPU"]

    def test_the_result_is_kept_when_the_device_state_changes_later(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})
        assert _platform.usable_platform("CUDA") is False

        fake.broken.clear()

        assert _platform.usable_platform("CUDA") is False
        assert fake.probed == ["CUDA"]

    def test_the_lookup_is_not_cached(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL)
        assert _platform.usable_platform("CUDA") is True

        fake.registered.discard("CUDA")

        assert _platform.usable_platform("CUDA") is False


class TestAutomaticSelection:
    def test_the_first_usable_platform_of_the_order_is_taken(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL)

        assert _platform.select_platform("auto").getName() == "CUDA"
        assert fake.probed == ["CUDA"]

    def test_a_cuda_platform_without_a_device_is_skipped(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ("CUDA", "CPU", "Reference"), broken={"CUDA": NO_DEVICE})

        assert _platform.select_platform("auto").getName() == "CPU"
        assert fake.probed == ["CUDA", "CPU"]

    def test_the_order_is_cuda_opencl_cpu_reference(self, monkeypatch):
        fake = FakePlatforms(
            monkeypatch,
            ALL,
            broken={"CUDA": NO_DEVICE, "OpenCL": "no device", "CPU": "no threads"},
        )

        assert _platform.select_platform("auto").getName() == "Reference"
        assert fake.probed == ["CUDA", "OpenCL", "CPU", "Reference"]

    def test_unregistered_platforms_are_skipped_without_a_probe(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ("CPU", "Reference"))

        assert _platform.select_platform("auto").getName() == "CPU"
        assert fake.probed == ["CPU"]

    def test_nothing_usable_raises_and_lists_every_reason(self, monkeypatch):
        FakePlatforms(
            monkeypatch, ("CUDA", "Reference"), broken={"CUDA": NO_DEVICE, "Reference": "x"}
        )

        with pytest.raises(openmm.OpenMMException) as raised:
            _platform.select_platform("auto")

        message = str(raised.value)
        assert "CUDA" in message and NO_DEVICE in message
        assert "OpenCL" in message and "CPU" in message and "Reference" in message

    def test_auto_is_the_default(self, monkeypatch):
        FakePlatforms(monkeypatch, ("CPU", "Reference"))

        assert _platform.select_platform().getName() == "CPU"


class TestExplicitSelection:
    def test_a_usable_platform_is_returned(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL)

        assert _platform.select_platform("CPU").getName() == "CPU"
        assert fake.probed == ["CPU"]

    def test_an_unusable_platform_fails_with_its_name_and_the_openmm_message(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})

        with pytest.raises(openmm.OpenMMException) as raised:
            _platform.select_platform("CUDA")

        assert str(raised.value) == f"OpenMM platform CUDA is not usable: {NO_DEVICE}"
        assert fake.probed == ["CUDA"]

    def test_an_unusable_platform_does_not_fall_back(self, monkeypatch):
        fake = FakePlatforms(monkeypatch, ALL, broken={"OpenCL": "no device"})

        with pytest.raises(openmm.OpenMMException):
            _platform.select_platform("OpenCL")

        assert "CPU" not in fake.probed

    def test_a_platform_that_is_not_registered_fails_with_the_openmm_message(self, monkeypatch):
        FakePlatforms(monkeypatch, ("CPU", "Reference"))

        with pytest.raises(openmm.OpenMMException, match='no registered Platform called "CUDA"'):
            _platform.select_platform("CUDA")

    def test_the_failure_comes_again_on_the_next_request(self, monkeypatch):
        FakePlatforms(monkeypatch, ALL, broken={"CUDA": NO_DEVICE})

        for _ in range(2):
            with pytest.raises(openmm.OpenMMException, match="CUDA is not usable"):
                _platform.select_platform("CUDA")


class TestCallSites:
    """Each site that chose a platform asks the same probe."""

    def test_simulation_auto_takes_the_cpu_platform_when_cuda_has_no_device(self, monkeypatch):
        FakePlatforms(monkeypatch, ("CUDA", "CPU", "Reference"), broken={"CUDA": NO_DEVICE})
        simulation = MDSimulation(None, None, SimulationConfig(platform="auto"))

        assert simulation._get_platform().getName() == "CPU"

    def test_simulation_explicit_cuda_fails_when_cuda_has_no_device(self, monkeypatch):
        FakePlatforms(monkeypatch, ("CUDA", "CPU", "Reference"), broken={"CUDA": NO_DEVICE})
        simulation = MDSimulation(None, None, SimulationConfig(platform="CUDA"))

        with pytest.raises(openmm.OpenMMException, match="OpenMM platform CUDA is not usable"):
            simulation._get_platform()

    def test_the_energy_metrics_fall_back_to_the_cpu_platform(self, monkeypatch):
        FakePlatforms(monkeypatch, ("CUDA", "CPU", "Reference"), broken={"CUDA": NO_DEVICE})

        platform, properties = energy._get_platform("cuda")

        assert platform.getName() == "CPU"
        assert properties == {}


class TestRealPlatforms:
    """No replacement: the platforms of this install."""

    def test_the_reference_platform_is_usable(self):
        assert _platform.usable_platform("Reference") is True
        assert _platform.select_platform("Reference").getName() == "Reference"

    def test_a_name_openmm_does_not_know_is_not_usable(self):
        assert _platform.usable_platform("NoSuchPlatform") is False
        with pytest.raises(openmm.OpenMMException, match="NoSuchPlatform"):
            _platform.select_platform("NoSuchPlatform")

    def test_auto_gives_a_platform_a_context_can_be_made_on(self):
        platform = _platform.select_platform("auto")
        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)

        context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)

        assert context.getPlatform().getName() == platform.getName()

    def test_the_suite_sees_a_cuda_device_exactly_when_the_probe_does(self):
        assert HAS_CUDA is _platform.usable_platform("CUDA")

    def test_the_energy_metrics_take_the_cpu_platform_for_the_cpu_device(self):
        platform, properties = energy._get_platform("cpu")

        assert platform.getName() == "CPU"
        assert properties == {}


class TestHealthyGpu:
    """On a machine with a working CUDA device the choice is the one made before the probe."""

    @requires_cuda
    def test_auto_still_takes_cuda(self):
        assert _platform.select_platform("auto").getName() == "CUDA"

    @requires_cuda
    def test_simulation_auto_still_takes_cuda(self):
        simulation = MDSimulation(None, None, SimulationConfig(platform="auto"))

        assert simulation._get_platform().getName() == "CUDA"

    @requires_cuda
    def test_simulation_explicit_cuda_still_takes_cuda(self):
        simulation = MDSimulation(None, None, SimulationConfig(platform="CUDA"))

        assert simulation._get_platform().getName() == "CUDA"

    @requires_cuda
    def test_the_energy_metrics_still_take_cuda_in_mixed_precision(self):
        platform, properties = energy._get_platform("cuda")

        assert platform.getName() == "CUDA"
        assert properties == {"CudaPrecision": "mixed"}

    @requires_cuda
    def test_the_receptor_energy_still_runs_on_cuda(self, monkeypatch):
        from conftest import DATA_DIR
        from openmm import app

        from binding_metrics.metrics.receptor_quality import compute_receptor_quality

        used = []
        real_simulation = app.Simulation

        class Recording(real_simulation):
            def __init__(self, topology, system, integrator, platform=None, *args, **kwargs):
                used.append(platform.getName())
                super().__init__(topology, system, integrator, platform, *args, **kwargs)

        monkeypatch.setattr(app, "Simulation", Recording)

        result = compute_receptor_quality(DATA_DIR / "example_linear_p53_1YCR.pdb", "A")

        assert result["models"][0]["energy"]["error"] is None
        assert used == ["CUDA"]
