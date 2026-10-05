"""Lazy exports of binding_metrics.poses and what importing it loads."""

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import binding_metrics.poses as poses


def _type_checking_imports(package):
    tree = ast.parse(Path(package.__file__).read_text(encoding="utf-8"))
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for statement in node.body:
                for alias in statement.names:
                    imports[alias.asname or alias.name] = statement.module
    return imports


class TestExports:
    def test_the_export_table_matches_all_and_the_type_checking_block(self):
        assert set(poses._EXPORTS) == set(poses.__all__)
        assert poses._EXPORTS == _type_checking_imports(poses)

    def test_every_export_is_the_object_its_module_defines(self):
        for name, module_name in poses._EXPORTS.items():
            assert getattr(poses, name) is getattr(importlib.import_module(module_name), name)

    def test_the_three_classes_are_exported(self):
        assert {"Pose", "DockedPose", "ExperimentalPose"} <= set(poses.__all__)

    def test_dir_lists_every_export(self):
        assert set(poses.__all__) <= set(dir(poses))

    def test_an_unknown_name_is_an_attribute_error(self):
        with pytest.raises(AttributeError, match="no_such_name"):
            poses.no_such_name  # noqa: B018


class TestImports:
    @staticmethod
    def _run(code: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    def test_importing_the_package_loads_none_of_its_modules(self):
        out = self._run(
            """
            import sys
            import binding_metrics.poses
            print(sorted(m for m in sys.modules if m.startswith("binding_metrics.poses.")))
            """
        )
        assert out == "[]"

    def test_importing_the_kinds_loads_no_structure_library(self):
        out = self._run(
            """
            import sys
            from binding_metrics.poses import DockedPose, ExperimentalPose, Pose
            libraries = ("biotite", "scipy", "openmm", "gemmi", "mdtraj")
            print([m for m in libraries if m in sys.modules])
            """
        )
        assert out == "[]"

    def test_reading_a_structure_imports_biotite_and_nothing_before(self, tmp_path):
        from tests.predictors import synth

        path = synth.write_structure(synth.synthetic_complex().atoms, tmp_path / "m.cif")
        out = self._run(
            f"""
            import sys
            from binding_metrics.poses import DockedPose
            pose = DockedPose.from_file({str(path)!r})
            before = "biotite" in sys.modules
            pose.atoms()
            print(before, "biotite" in sys.modules)
            """
        )
        assert out == "False True"
