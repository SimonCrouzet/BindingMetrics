"""Shared biotite import and structure loader of the metric modules."""

import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics import (  # noqa: E402
    electrostatics,
    evobind,
    geometry,
    interface,
    openfold,
    polar_contacts,
    receptor_quality,
    sasa,
)
from binding_metrics.metrics._common import import_biotite, load_structure  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"

BLOCKED = {
    "biotite": None,
    "biotite.structure": None,
    "biotite.structure.io": None,
    "biotite.structure.io.pdb": None,
    "biotite.structure.io.pdbx": None,
}

# Each module keeps its own import function and its own install-hint wording.
HINTS = [
    (electrostatics._import_biotite, "electrostatics metrics"),
    (evobind._import_biotite, "EvoBind metrics"),
    (geometry._import_biotite, "geometry metrics"),
    (interface._import_biotite, "interface metrics"),
    (openfold._import_biotite_struc, "per-chain structural analysis"),
    (polar_contacts._import_biotite, "H-bond/salt bridge metrics"),
    (receptor_quality._import_biotite, "receptor quality metrics"),
]

LOADERS = [
    electrostatics._load_structure,
    evobind._load_atoms,
    geometry._load_structure,
    openfold._load_atoms,
    interface.load_biotite_structure,
]


def _same_atoms(a, b):
    assert a.array_length() == b.array_length()
    np.testing.assert_array_equal(a.coord, b.coord)
    for annotation in ("chain_id", "res_id", "res_name", "atom_name", "element"):
        np.testing.assert_array_equal(getattr(a, annotation), getattr(b, annotation))


class TestImportBiotite:
    def test_returns_the_three_modules(self):
        structure, pdbx, pdb = import_biotite("tests")
        assert structure.__name__ == "biotite.structure"
        assert pdbx.__name__ == "biotite.structure.io.pdbx"
        assert pdb.__name__ == "biotite.structure.io.pdb"

    def test_hint_names_the_purpose_and_the_extra(self):
        with mock.patch.dict(sys.modules, BLOCKED):
            with pytest.raises(ImportError) as excinfo:
                import_biotite("some metrics")
        assert str(excinfo.value) == (
            "biotite is required for some metrics. "
            "Install with: pip install binding-metrics[biotite]"
        )

    @pytest.mark.parametrize("function, purpose", HINTS)
    def test_each_module_keeps_its_own_function_and_wording(self, function, purpose):
        with mock.patch.dict(sys.modules, BLOCKED):
            with pytest.raises(ImportError) as excinfo:
                function()
        assert str(excinfo.value) == (
            f"biotite is required for {purpose}. Install with: pip install binding-metrics[biotite]"
        )

    @pytest.mark.parametrize("function, purpose", HINTS)
    def test_each_module_function_returns_biotite_structure_first(self, function, purpose):
        assert function()[0] is struc

    def test_static_sasa_reports_the_missing_extra_before_reading_anything(self):
        with mock.patch.dict(sys.modules, BLOCKED):
            with pytest.raises(ImportError, match=r"static SASA.*binding-metrics\[biotite\]"):
                sasa.compute_delta_sasa_static("missing.pdb", "B", "A")


@pytest.mark.skipif(not (P53.exists() and CYCLOSPORIN.exists()), reason="examples not bundled")
class TestLoader:
    @pytest.mark.parametrize("loader", LOADERS)
    @pytest.mark.parametrize("path", [P53, CYCLOSPORIN], ids=["pdb", "cif"])
    def test_every_module_loader_gives_the_shared_result(self, loader, path):
        _same_atoms(loader(path), load_structure(path))

    def test_pdb_matches_biotite_directly(self):
        direct = pdb_io.get_structure(pdb_io.PDBFile.read(str(P53)), model=1)
        _same_atoms(load_structure(P53), direct)

    def test_string_paths_are_accepted(self):
        _same_atoms(load_structure(str(P53)), load_structure(P53))

    def test_mmcif_suffix_is_read_as_mmcif(self, tmp_path):
        copy = tmp_path / "structure.mmcif"
        shutil.copy(CYCLOSPORIN, copy)
        _same_atoms(load_structure(copy), load_structure(CYCLOSPORIN))

    def test_formal_charge_is_read_only_on_request(self):
        assert "charge" not in load_structure(CYCLOSPORIN).get_annotation_categories()
        with_charge = load_structure(CYCLOSPORIN, charge=True)
        assert "charge" in with_charge.get_annotation_categories()
        assert "charge" in interface.load_biotite_structure(CYCLOSPORIN).get_annotation_categories()
        assert (
            "charge" not in electrostatics._load_structure(CYCLOSPORIN).get_annotation_categories()
        )

    def test_all_models_come_back_as_a_stack(self, tmp_path):
        atoms = load_structure(P53)
        two_models = struc.stack([atoms, atoms])
        path = tmp_path / "two.pdb"
        file = pdb_io.PDBFile()
        pdb_io.set_structure(file, two_models)
        file.write(str(path))

        assert isinstance(load_structure(path), struc.AtomArray)
        stack = load_structure(path, model=None)
        assert isinstance(stack, struc.AtomArrayStack) and stack.stack_depth() == 2
        assert len(receptor_quality._load_all_models(path)) == 2
        assert len(receptor_quality._load_all_models(P53)) == 1

    def test_missing_file_is_not_swallowed(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_structure(tmp_path / "absent.cif")
        with pytest.raises(OSError):
            load_structure(tmp_path / "absent.pdb")


class TestEnergyUnitFactors:
    """The kcal/kJ factor is defined once; the modules keep their private names."""

    def test_values_are_bitwise_the_old_literals(self):
        from binding_metrics.metrics import _common

        assert _common.KCAL_TO_KJ == 4.184
        assert _common.KJ_TO_KCAL == 1.0 / 4.184

    def test_modules_use_the_shared_factor(self):
        from binding_metrics.metrics import _common

        assert interface._KCAL_TO_KJ is _common.KCAL_TO_KJ
        assert electrostatics._KJ_TO_KCAL is _common.KJ_TO_KCAL
        assert interface._KCAL_TO_KJ * electrostatics._KJ_TO_KCAL == pytest.approx(1.0, rel=1e-15)
