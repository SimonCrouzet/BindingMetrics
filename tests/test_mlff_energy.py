"""The MLFF interaction-energy interface: backends, pocket description and placeholders.

No backend exists yet, so these tests pin the contract that an adapter will fill:
what a placeholder says when it is asked for, what a registered backend must
provide, and that importing the module needs no MLFF package, OpenMM or torch.
"""

import subprocess
import sys
import textwrap

import pytest

from binding_metrics.metrics import mlff_energy
from binding_metrics.metrics.mlff_energy import (
    MLFFBackend,
    PocketSpec,
    available_backends,
    get_backend,
    register_backend,
)

PLACEHOLDER_NAMES = ("uma", "mace", "orb", "aimnet2")
REFERENCE_DOI = "10.26434/chemrxiv.15008810"


@pytest.fixture
def isolated_registry(monkeypatch):
    """Let a test register backends without changing the registry other tests see."""
    monkeypatch.setattr(mlff_energy, "_BACKENDS", dict(mlff_energy._BACKENDS))


def _make_backend(backend_name="fake", licence="test licence", available=True):
    """A concrete backend that returns a fixed energy, for registry tests."""

    class FakeBackend(MLFFBackend):
        name = backend_name
        weights_licence = licence

        @classmethod
        def is_available(cls):
            return available

        def energy_ev(self, atoms, *, charge, spin):
            return -1.5

    return FakeBackend


class TestMLFFBackendBase:
    def test_the_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            MLFFBackend()

    def test_a_subclass_without_energy_ev_cannot_be_instantiated(self):
        class NoEnergy(MLFFBackend):
            name = "no_energy"
            weights_licence = "test licence"

        with pytest.raises(TypeError, match="energy_ev"):
            NoEnergy()

    def test_a_subclass_with_energy_ev_works_and_is_unavailable_by_default(self):
        class Minimal(MLFFBackend):
            name = "minimal"
            weights_licence = "test licence"

            def energy_ev(self, atoms, *, charge, spin):
                return 0.25

        assert Minimal().energy_ev(None, charge=0, spin=1) == 0.25
        assert Minimal.is_available() is False


class TestPocketSpec:
    def test_defaults_are_valid_and_the_spec_is_frozen(self):
        spec = PocketSpec()
        assert spec.cutoff_angstrom > 0
        assert spec.cap in ("none", "hydrogen")
        assert spec.include_waters is False
        assert spec.protonation in ("as_given", "reprotonate")
        with pytest.raises(AttributeError):
            spec.cutoff_angstrom = 3.0

    def test_accepts_every_documented_value(self):
        spec = PocketSpec(
            cutoff_angstrom=8, cap="none", include_waters=True, protonation="reprotonate"
        )
        assert spec.cutoff_angstrom == 8
        assert spec.cap == "none"

    @pytest.mark.parametrize("cutoff", [0, -1.0, float("nan"), float("inf"), True, "6"])
    def test_rejects_a_cutoff_that_is_not_a_positive_finite_number(self, cutoff):
        with pytest.raises(ValueError, match="cutoff_angstrom"):
            PocketSpec(cutoff_angstrom=cutoff)

    def test_rejects_an_unknown_cap(self):
        with pytest.raises(ValueError, match="cap"):
            PocketSpec(cap="methyl")

    def test_rejects_an_unknown_protonation(self):
        with pytest.raises(ValueError, match="protonation"):
            PocketSpec(protonation="pH7")

    def test_rejects_a_waters_flag_that_is_not_a_bool(self):
        with pytest.raises(ValueError, match="include_waters"):
            PocketSpec(include_waters="yes")


class TestPlaceholderBackends:
    @pytest.mark.parametrize("name", PLACEHOLDER_NAMES)
    def test_asking_for_a_placeholder_names_the_backend_and_the_reference(self, name):
        with pytest.raises(NotImplementedError) as excinfo:
            get_backend(name)
        message = str(excinfo.value)
        assert repr(name) in message
        assert "Ryczko et al." in message
        assert REFERENCE_DOI in message

    @pytest.mark.parametrize("name", PLACEHOLDER_NAMES)
    def test_a_placeholder_states_its_weights_licence(self, name):
        with pytest.raises(NotImplementedError, match="Weights licence: .+"):
            get_backend(name)

    def test_uma_message_carries_the_gated_licence_caveat(self):
        with pytest.raises(NotImplementedError) as excinfo:
            get_backend("uma")
        message = str(excinfo.value)
        assert "FAIR Chemistry License" in message
        assert "must not be bundled" in message

    def test_no_placeholder_is_available(self):
        assert available_backends() == []
        for name in PLACEHOLDER_NAMES:
            assert mlff_energy._BACKENDS[name].is_available() is False

    def test_the_four_placeholder_names_are_registered(self):
        assert set(PLACEHOLDER_NAMES) <= set(mlff_energy._BACKENDS)
        for name in PLACEHOLDER_NAMES:
            cls = mlff_energy._BACKENDS[name]
            assert cls.name == name
            assert cls.weights_licence.strip()


class TestBackendRegistry:
    def test_an_unknown_name_is_a_value_error_that_lists_the_known_names(self):
        with pytest.raises(ValueError, match="unknown MLFF backend 'nope'") as excinfo:
            get_backend("nope")
        for name in PLACEHOLDER_NAMES:
            assert name in str(excinfo.value)

    @pytest.mark.parametrize("name", [None, 3, ["uma"]])
    def test_a_name_that_is_not_a_string_is_a_value_error(self, name):
        with pytest.raises(ValueError, match="unknown MLFF backend"):
            get_backend(name)

    def test_register_backend_returns_the_class_and_get_backend_instantiates_it(
        self, isolated_registry
    ):
        fake = _make_backend()
        assert register_backend(fake) is fake
        backend = get_backend("fake")
        assert isinstance(backend, fake)
        assert backend.energy_ev(None, charge=0, spin=1) == -1.5

    def test_register_backend_works_as_a_decorator(self, isolated_registry):
        @register_backend
        class Decorated(MLFFBackend):
            name = "decorated"
            weights_licence = "test licence"

            def energy_ev(self, atoms, *, charge, spin):
                return 0.0

        assert isinstance(get_backend("decorated"), Decorated)

    def test_available_backends_lists_only_the_available_ones_sorted(self, isolated_registry):
        register_backend(_make_backend("zeta"))
        register_backend(_make_backend("alpha"))
        register_backend(_make_backend("offline", available=False))
        assert available_backends() == ["alpha", "zeta"]

    def test_a_real_backend_replaces_a_placeholder(self, isolated_registry):
        register_backend(_make_backend("uma"))
        assert isinstance(get_backend("uma"), MLFFBackend)
        assert available_backends() == ["uma"]

    def test_a_registered_name_cannot_be_taken_twice(self, isolated_registry):
        register_backend(_make_backend("fake"))
        with pytest.raises(ValueError, match="already registered"):
            register_backend(_make_backend("fake"))

    def test_only_backend_subclasses_can_be_registered(self, isolated_registry):
        class NotABackend:
            name = "impostor"
            weights_licence = "test licence"

        with pytest.raises(TypeError, match="MLFFBackend"):
            register_backend(NotABackend)
        with pytest.raises(TypeError, match="MLFFBackend"):
            register_backend(_make_backend()())

    @pytest.mark.parametrize("bad_name", ["", "Fake", " fake", "fa ke", None, 3])
    def test_a_malformed_name_is_rejected(self, isolated_registry, bad_name):
        with pytest.raises(ValueError, match="name"):
            register_backend(_make_backend(bad_name))

    @pytest.mark.parametrize("bad_licence", ["", "   ", None])
    def test_a_missing_licence_is_rejected(self, isolated_registry, bad_licence):
        with pytest.raises(ValueError, match="weights_licence"):
            register_backend(_make_backend(licence=bad_licence))

    def test_a_backend_without_name_or_licence_attributes_is_rejected(self, isolated_registry):
        class Bare(MLFFBackend):
            def energy_ev(self, atoms, *, charge, spin):
                return 0.0

        with pytest.raises(ValueError, match="name"):
            register_backend(Bare)


_BLOCKED_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = ("openmm", "simtk", "torch", "ase", "fairchem", "mace", "orb_models", "aimnet2calc")


    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"{name} is blocked for this test")
            return None


    sys.meta_path.insert(0, _Block())

    from binding_metrics.metrics import mlff_energy

    loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not loaded, f"mlff_energy pulled in {loaded}"
    assert mlff_energy.available_backends() == []
    print("IMPORT_OK")
    """
)


def test_module_imports_without_openmm_torch_or_an_mlff_package(tmp_path):
    """Run in a subprocess so the blocked imports cannot leak into the test session."""
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "IMPORT_OK" in completed.stdout
