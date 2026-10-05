"""The parser contract, run over every registered adapter, and over the harness itself.

``TestRegisteredAdapters`` is the contract test: every model in ``sorted(PARSERS)`` meets every
check of ``contract.py``. A new adapter needs a ``tests/predictors/synth_<model>.py`` writer and
nothing else here. ``TestTheHarness`` checks the checks: the stub adapter passes all of them,
and each adapter in ``stub_defects.py`` fails the check that is meant to catch its defect.
"""

import sys
import types

import pytest

from binding_metrics.predictors.registry import PARSERS, ParserSpec, register_parser
from tests.predictors import contract, stub_defects
from tests.predictors.contract import CHECKS
from tests.predictors.synth_stub import StubParser

_MODELS = sorted(PARSERS)
_PAIRS = [
    pytest.param(model, check, id=f"{model}-{check.__name__}")
    for model in _MODELS
    for check in CHECKS
]
if not _PAIRS:
    _PAIRS = [
        pytest.param(
            None,
            None,
            id="no-adapter",
            marks=pytest.mark.skip(reason="no predictor adapter is registered"),
        )
    ]


@pytest.mark.parametrize("model, check", _PAIRS)
def test_registered_adapter_meets_the_contract(model, check, tmp_path):
    check(model, tmp_path)


# ---------------------------------------------------------------------------
# The harness on the stub adapter
# ---------------------------------------------------------------------------


def _spec(cls):
    return ParserSpec(
        name="stub",
        import_path=f"{cls.__module__}:{cls.__name__}",
        display_name=cls.display_name,
        family=cls.family,
    )


@pytest.fixture
def register_stub():
    """Register an adapter class as the model 'stub' for one test, then restore the registry."""
    saved = dict(PARSERS)

    def _register(cls=StubParser):
        register_parser(_spec(cls), replace=True)

    yield _register
    PARSERS.clear()
    PARSERS.update(saved)


class TestTheHarness:
    @pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
    def test_the_stub_adapter_passes_every_check(self, register_stub, check, tmp_path):
        register_stub()
        check("stub", tmp_path)

    @pytest.mark.parametrize("defect", stub_defects.DEFECTS, ids=lambda cls: cls.__name__)
    def test_each_defective_adapter_fails_the_check_that_targets_it(
        self, register_stub, defect, tmp_path
    ):
        register_stub(defect)
        check = getattr(contract, defect.FAILS_CHECK)
        with pytest.raises(AssertionError):
            check("stub", tmp_path)

    def test_every_check_is_targeted_by_at_least_one_defect(self):
        targeted = {defect.FAILS_CHECK for defect in stub_defects.DEFECTS}
        untargeted = {check.__name__ for check in CHECKS} - targeted
        # the capabilities check is exercised separately below
        assert untargeted == {"check_capabilities_declaration"}

    def test_a_model_without_a_writer_module_fails_with_an_instruction(self):
        with pytest.raises(AssertionError, match="synth_nowriter does not exist.*write_prediction"):
            contract.writer_module("nowriter")


class TestCapabilitiesCheck:
    """``capabilities`` is None or an instance of ``binding_metrics.capabilities.Capabilities``."""

    def _adapter(self, value):
        return type(
            "Declared", (StubParser,), {"capabilities": value, "__module__": StubParser.__module__}
        )

    def _register(self, register_stub, monkeypatch, value):
        cls = self._adapter(value)
        monkeypatch.setattr(sys.modules[StubParser.__module__], "Declared", cls, raising=False)
        register_stub(cls)

    def _fake_capabilities_module(self, monkeypatch):
        module = types.ModuleType("binding_metrics.capabilities")
        module.Capabilities = type("Capabilities", (), {})
        monkeypatch.setitem(sys.modules, "binding_metrics.capabilities", module)
        return module.Capabilities

    def test_none_needs_no_class(self, register_stub, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "binding_metrics.capabilities", None)  # cannot be imported
        register_stub()
        contract.check_capabilities_declaration("stub", tmp_path)

    def test_an_instance_of_the_class_passes(self, register_stub, monkeypatch, tmp_path):
        capabilities = self._fake_capabilities_module(monkeypatch)
        self._register(register_stub, monkeypatch, capabilities())
        contract.check_capabilities_declaration("stub", tmp_path)

    def test_anything_else_fails_once_the_class_exists(self, register_stub, monkeypatch, tmp_path):
        self._fake_capabilities_module(monkeypatch)
        self._register(register_stub, monkeypatch, "not capabilities")
        with pytest.raises(AssertionError, match="must be None or a Capabilities"):
            contract.check_capabilities_declaration("stub", tmp_path)

    def test_the_type_is_not_checked_while_the_class_does_not_exist(
        self, register_stub, monkeypatch, tmp_path
    ):
        monkeypatch.setitem(sys.modules, "binding_metrics.capabilities", None)
        self._register(register_stub, monkeypatch, "anything")
        with pytest.raises(pytest.skip.Exception, match="does not exist yet"):
            contract.check_capabilities_declaration("stub", tmp_path)
