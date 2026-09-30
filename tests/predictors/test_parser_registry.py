"""ParserSpec, PARSERS, register_parser and get_parser; and the lazy exports of the package."""

import ast
import dataclasses
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import binding_metrics.predictors as predictors
from binding_metrics.predictors import registry
from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.registry import (
    PARSERS,
    ParserSpec,
    get_parser,
    register_parser,
)
from tests.predictors.synth_stub import StubParser

STUB_PATH = "tests.predictors.synth_stub:StubParser"


def _stub_spec(**overrides):
    fields = dict(
        name="stub",
        import_path=STUB_PATH,
        display_name="Stub model",
        family="af3",
        description="a made-up model for the tests",
    )
    fields.update(overrides)
    return ParserSpec(**fields)


class _CapableParser(StubParser):
    """An adapter that declares capabilities (the value is a stand-in object)."""

    capabilities = ("declared",)


@pytest.fixture
def empty_registry():
    """Run a test against an empty ``PARSERS`` and restore the real one afterwards."""
    saved = dict(PARSERS)
    PARSERS.clear()
    yield PARSERS
    PARSERS.clear()
    PARSERS.update(saved)


class TestParserSpec:
    def test_fields(self):
        spec = _stub_spec()
        assert (spec.name, spec.display_name, spec.family) == ("stub", "Stub model", "af3")
        assert spec.import_path == STUB_PATH

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _stub_spec().name = "other"

    @pytest.mark.parametrize("name", ["", "Stub", "2fold", "with space", "a-b"])
    def test_the_name_is_lower_case_letters_digits_and_underscores(self, name):
        with pytest.raises(ValueError, match="parser name"):
            _stub_spec(name=name)

    @pytest.mark.parametrize("path", ["no_colon", "a:b:c"])
    def test_the_import_path_names_a_module_and_a_class(self, path):
        with pytest.raises(ValueError, match="module.path:Class"):
            _stub_spec(import_path=path)

    def test_the_family_must_be_known(self):
        with pytest.raises(ValueError, match="family must be one of"):
            _stub_spec(family="af4")

    def test_load_returns_the_class_and_create_an_instance(self):
        assert _stub_spec().load() is StubParser
        first, second = _stub_spec().create(), _stub_spec().create()
        assert isinstance(first, StubParser) and first is not second

    def test_nothing_is_imported_until_load_runs(self):
        spec = _stub_spec(import_path="tests.predictors.no_such_module:Missing")
        assert spec.name == "stub"  # building the spec did not import the module
        with pytest.raises(ModuleNotFoundError, match="no_such_module"):
            spec.load()

    def test_a_class_that_is_not_a_parser_is_refused(self):
        spec = _stub_spec(import_path="pathlib:Path")
        with pytest.raises(TypeError, match="not a PredictionParser subclass"):
            spec.load()

    def test_a_missing_class_is_an_attribute_error(self):
        with pytest.raises(AttributeError):
            _stub_spec(import_path="tests.predictors.synth_stub:Nope").load()

    def test_capabilities_are_read_without_creating_the_adapter(self, monkeypatch):
        assert _stub_spec().load_capabilities() is None

        def _no_instances(*args, **kwargs):
            raise AssertionError("the adapter must not be instantiated")

        monkeypatch.setattr(PredictionParser, "__init__", _no_instances, raising=False)
        spec = _stub_spec(import_path="tests.predictors.test_parser_registry:_CapableParser")
        assert spec.load_capabilities() == ("declared",)


class TestRegistry:
    def test_get_parser_returns_a_new_adapter_instance(self, empty_registry):
        register_parser(_stub_spec())
        first, second = get_parser("stub"), get_parser("stub")
        assert isinstance(first, StubParser)
        assert first is not second

    def test_sorted_parsers_lists_the_models(self, empty_registry):
        register_parser(_stub_spec(name="zeta"))
        register_parser(_stub_spec(name="alpha"))
        assert sorted(PARSERS) == ["alpha", "zeta"]

    def test_an_unknown_name_is_a_key_error_listing_the_known_ones(self, empty_registry):
        register_parser(_stub_spec())
        with pytest.raises(KeyError, match=r"Unknown predictor 'nope'. Available: stub"):
            get_parser("nope")

    def test_an_empty_registry_says_so(self, empty_registry):
        with pytest.raises(KeyError, match="none registered"):
            get_parser("stub")

    def test_registering_the_same_spec_twice_is_harmless(self, empty_registry):
        register_parser(_stub_spec())
        register_parser(_stub_spec())
        assert list(PARSERS) == ["stub"]

    def test_a_different_spec_under_a_taken_name_needs_replace(self, empty_registry):
        register_parser(_stub_spec())
        other = _stub_spec(description="another")
        with pytest.raises(ValueError, match="already registered"):
            register_parser(other)
        register_parser(other, replace=True)
        assert PARSERS["stub"] is other

    def test_the_registry_module_uses_the_same_dict_as_the_package(self):
        assert predictors.PARSERS is registry.PARSERS


class TestPackageExports:
    """The lazy table, the TYPE_CHECKING block and ``__all__`` are three copies of one list."""

    def _type_checking_imports(self):
        tree = ast.parse(Path(predictors.__file__).read_text(encoding="utf-8"))
        imports = {}
        for node in tree.body:
            if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
                for statement in node.body:
                    for alias in statement.names:
                        imports[alias.asname or alias.name] = statement.module
        return imports

    def test_the_three_copies_agree(self):
        assert set(predictors._EXPORTS) == set(predictors.__all__)
        assert predictors._EXPORTS == self._type_checking_imports()

    def test_every_export_is_the_object_its_module_defines(self):
        import importlib

        for name, module_name in predictors._EXPORTS.items():
            assert getattr(predictors, name) is getattr(importlib.import_module(module_name), name)

    def test_the_documented_names_are_exported(self):
        assert set(predictors.__all__) == {
            "PARSERS",
            "ParserSpec",
            "PredictionFiles",
            "PredictionParser",
            "PredictionRecord",
            "SampleRef",
            "TokenLayout",
            "get_parser",
            "register_parser",
        }

    def test_importing_the_package_imports_none_of_its_modules_and_no_heavy_dependency(self):
        code = textwrap.dedent(
            """
            import sys
            import binding_metrics.predictors as p
            loaded = sorted(m for m in sys.modules if m.startswith("binding_metrics.predictors."))
            heavy = [m for m in ("biotite", "scipy", "openmm", "gemmi") if m in sys.modules]
            p.get_parser
            heavy_after = [m for m in ("biotite", "scipy", "openmm", "gemmi") if m in sys.modules]
            print(loaded, heavy, heavy_after)
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        assert out.strip() == "[] [] []"

    def test_an_unknown_name_is_an_attribute_error(self):
        with pytest.raises(AttributeError, match="no_such_name"):
            predictors.no_such_name  # noqa: B018
