"""A value that could not be computed keeps its sentinel and carries a `reason` string."""

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("hydride")

import biotite.structure.io.pdbx as pdbx  # noqa: E402

from binding_metrics.metrics import polar_contacts  # noqa: E402
from binding_metrics.metrics.interface import (  # noqa: E402
    compute_interface_metrics,
    load_biotite_structure,
)
from binding_metrics.metrics.polar_contacts import compute_hbonds  # noqa: E402
from binding_metrics.metrics.sasa import compute_delta_sasa_static  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"


@pytest.fixture
def failing_sasa(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sasa kernel failed")

    monkeypatch.setattr(importlib.import_module("biotite.structure.sasa"), "sasa", boom)


class TestSuccessHasNoReason:
    def test_interface(self):
        assert "reason" not in compute_interface_metrics(P53_MDM2)

    def test_static_sasa(self):
        assert "reason" not in compute_delta_sasa_static(P53_MDM2, "B", "A")

    def test_hbonds(self):
        atoms = load_biotite_structure(P53_MDM2)
        assert "reason" not in compute_hbonds(atoms, "B", "A")


class TestSasaFailure:
    def test_interface_keeps_nan_and_explains(self, failing_sasa):
        result = compute_interface_metrics(P53_MDM2, "B", "A")
        assert np.isnan(result["delta_sasa"])
        assert np.isnan(result["delta_g_int"])
        assert "RuntimeError" in result["reason"]
        assert "sasa kernel failed" in result["reason"]

    def test_static_sasa_keeps_nan_and_explains(self, failing_sasa):
        result = compute_delta_sasa_static(P53_MDM2, "B", "A")
        assert np.isnan(result["delta_sasa"])
        assert "sasa kernel failed" in result["reason"]


class TestPolarContactFailure:
    def test_interface_keeps_areas_and_explains_the_zero_hbonds(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("hbond detector failed")

        monkeypatch.setattr(polar_contacts, "compute_hbonds", boom)
        result = compute_interface_metrics(P53_MDM2, "B", "A")
        assert result["hbonds"] == 0
        assert np.isfinite(result["delta_sasa"])
        assert "hbond detector failed" in result["reason"]

    def test_hydrogen_placement_failure_is_reported_as_a_reason(self):
        """Non-finite coordinates make hydride fail, so there are no hydrogens to bond with."""
        atoms = load_biotite_structure(P53_MDM2)
        atoms.coord[np.where(atoms.element == "O")[0][:30]] = np.nan
        with pytest.warns(UserWarning, match="hydride.add_hydrogen failed"):
            result = compute_hbonds(atoms, "B", "A")
        assert result["hbonds"] == 0
        assert result["hbond_energy"] == 0.0
        assert "hydride.add_hydrogen failed" in result["reason"]

    def test_unexpected_hbond_errors_are_not_swallowed(self, monkeypatch):
        import biotite.structure as struc

        def boom(*args, **kwargs):
            raise RuntimeError("not a known failure mode")

        monkeypatch.setattr(struc, "hbond", boom)
        atoms = load_biotite_structure(P53_MDM2)
        with pytest.raises(RuntimeError, match="not a known failure mode"):
            compute_hbonds(atoms, "B", "A")

    def test_known_hbond_error_becomes_a_reason(self, monkeypatch):
        import biotite.structure as struc

        def boom(*args, **kwargs):
            raise struc.BadStructureError("no bonds")

        monkeypatch.setattr(struc, "hbond", boom)
        atoms = load_biotite_structure(P53_MDM2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_hbonds(atoms, "B", "A")
        assert result["hbonds"] == 0
        assert "BadStructureError" in result["reason"]


class TestLoaderFallback:
    def test_malformed_formal_charge_column_falls_back_to_no_charge_field(self, tmp_path):
        cif = pdbx.CIFFile.read(str(CYCLOSPORIN))
        atom_site = cif.block["atom_site"]
        n_atoms = len(atom_site["pdbx_formal_charge"])
        atom_site["pdbx_formal_charge"] = pdbx.CIFColumn(pdbx.CIFData(np.array(["x"] * n_atoms)))
        path = tmp_path / "bad_charge.cif"
        cif.write(str(path))

        atoms = load_biotite_structure(path)
        assert len(atoms) == n_atoms
        assert "charge" not in atoms.get_annotation_categories()

    def test_missing_file_is_not_swallowed(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_biotite_structure(tmp_path / "absent.cif")
