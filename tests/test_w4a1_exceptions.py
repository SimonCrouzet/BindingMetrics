"""Exception handling in the metrics modules: chained causes, deliberate broad catches, warnings.

The metrics behave as before; these tests pin what a caller can observe when an optional
dependency is missing or a step fails: the raised error names the original one as its
cause, and a failure that is caught on purpose (per-structure or per-mode isolation)
leaves a record whatever the exception type.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics import (
    energy,
    geometry,
    interface,
    openfold,
    polar_contacts,
    receptor_quality,
    sasa,
)
from binding_metrics.metrics._common import import_biotite
from binding_metrics.metrics.comparison import compute_structure_rmsd

P53_MDM2 = Path(__file__).resolve().parents[1] / "data" / "example_linear_p53_1YCR.pdb"

# Per-structure isolation: whatever a step raises, the result carries a reason.
STEP_FAILURES = [RuntimeError("boom"), ValueError("boom"), IndexError("boom"), KeyError("boom")]


class TestCommon:
    def test_missing_biotite_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "biotite.structure", None)
        with pytest.raises(ImportError, match="biotite is required for tests") as info:
            import_biotite("tests")
        assert isinstance(info.value.__cause__, ImportError)


class TestComparison:
    def test_missing_gemmi_error_names_the_original_import_failure(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "gemmi", None)
        with pytest.raises(ImportError, match="gemmi is required") as info:
            compute_structure_rmsd(tmp_path / "a.pdb", tmp_path / "b.pdb", "A")
        assert isinstance(info.value.__cause__, ImportError)


class TestGeometry:
    def test_missing_scipy_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
        with pytest.raises(ImportError, match="scipy is required") as info:
            geometry._import_scipy()
        assert isinstance(info.value.__cause__, ImportError)


class TestReceptorQuality:
    def test_missing_scipy_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
        with pytest.raises(ImportError, match="scipy is required") as info:
            receptor_quality._import_scipy()
        assert isinstance(info.value.__cause__, ImportError)

    def test_missing_openmm_error_names_the_original_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openmm", None)
        with pytest.raises(ImportError, match="openmm is required") as info:
            receptor_quality._import_openmm()
        assert isinstance(info.value.__cause__, ImportError)


class TestSasa:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_static_sasa_failure_is_recorded_whatever_its_type(self, monkeypatch, failure):
        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(importlib.import_module("biotite.structure.sasa"), "sasa", broken)
        result = sasa.compute_delta_sasa_static(P53_MDM2, peptide_chain="B", receptor_chain="A")
        assert np.isnan(result["delta_sasa"])
        assert result["reason"].startswith(f"SASA computation failed: {type(failure).__name__}")


class TestInterface:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_sasa_failure_is_recorded_whatever_its_type(self, monkeypatch, failure):
        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(interface, "_per_atom_sasa", broken)
        result = interface.compute_interface_metrics(P53_MDM2)
        assert result["reason"].startswith(f"SASA computation failed: {type(failure).__name__}")

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_polar_contact_failure_is_recorded_whatever_its_type(self, monkeypatch, failure):
        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(polar_contacts, "compute_hbonds", broken)
        result = interface.compute_interface_metrics(P53_MDM2)
        assert result["reason"].startswith(
            f"H-bond/salt bridge computation failed: {type(failure).__name__}"
        )
        assert result["hbonds"] == 0
        # The SASA part of the result is still computed.
        assert np.isfinite(result["delta_sasa"])


class TestEnergy:
    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_subsystem_evaluation_records_any_exception_type(self, failure):
        pytest.importorskip("openmm")

        class _Context:
            def setPositions(self, positions):
                raise failure

        class _Simulation:
            context = _Context()

        failures: list[str] = []
        out = energy._evaluate_subsystem_energies(
            _Simulation(), None, None, "B", "A", "obc2", "cpu", failures=failures
        )
        assert out == (None, None, None)
        assert failures == [f"{type(failure).__name__}: {failure}"]

    @pytest.mark.parametrize("failure", STEP_FAILURES, ids=lambda e: type(e).__name__)
    def test_failed_setup_is_recorded_in_error_message(self, monkeypatch, failure):
        pytest.importorskip("openmm")

        def broken(*args, **kwargs):
            raise failure

        monkeypatch.setattr(energy, "_create_implicit_system", broken)
        result = energy.compute_interaction_energy(
            P53_MDM2, peptide_chain="B", receptor_chain="A", modes=("raw",)
        )
        assert result["success"] is False
        assert result["error_message"] == f"{type(failure).__name__}: {failure}"


class TestOpenfold:
    @staticmethod
    def _write_run(tmp_path, *, readable_structure):
        """An OpenFold3 output directory for a receptor A (3 residues) and binder B (2)."""
        seed_dir = tmp_path / "run" / "seed_1"
        seed_dir.mkdir(parents=True)
        prefix = "run_seed_1_sample_1"
        aggregated = {"avg_plddt": 87.5, "ptm": 0.88, "iptm": 0.76}
        (seed_dir / f"{prefix}_confidences_aggregated.json").write_text(
            json.dumps(aggregated), encoding="utf-8"
        )
        # Wrong sizes on purpose: 10 pLDDT values for 5 atoms, 12-token matrices for 5 tokens.
        confidences = {
            "plddt": [90.0] * 10,
            "pde": np.ones((12, 12)).tolist(),
            "pae": np.ones((12, 12)).tolist(),
        }
        (seed_dir / f"{prefix}_confidences.json").write_text(
            json.dumps(confidences), encoding="utf-8"
        )
        model = seed_dir / f"{prefix}_model.cif"
        if readable_structure:
            pdbx = pytest.importorskip("biotite.structure.io.pdbx")
            import biotite.structure as struc

            atoms = struc.array(
                [
                    struc.Atom(
                        [float(index), 0.0, 0.0],
                        chain_id=chain,
                        res_id=res_id,
                        res_name="ALA",
                        atom_name="CA",
                        element="C",
                    )
                    for index, (chain, res_id) in enumerate(
                        [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2)]
                    )
                ]
            )
            cif = pdbx.CIFFile()
            pdbx.set_structure(cif, atoms)
            cif.write(str(model))
        else:
            model.write_text("# stub CIF\n", encoding="utf-8")
        return tmp_path

    def test_structural_failure_warning_points_at_the_caller(self, tmp_path):
        root = self._write_run(tmp_path, readable_structure=False)
        with pytest.warns(UserWarning, match="structural analysis failed") as record:
            metrics = openfold.compute_openfold_metrics(
                root, "run", binder_chain="B", receptor_chain="A"
            )
        assert Path(record[0].filename) == Path(__file__)
        assert metrics["reason"].startswith("structural analysis failed:")
        assert metrics["avg_plddt"] == pytest.approx(87.5)

    def test_per_step_warnings_point_at_the_caller(self, tmp_path):
        root = self._write_run(tmp_path, readable_structure=True)
        with pytest.warns(UserWarning) as record:
            metrics = openfold.compute_openfold_metrics(
                root,
                "run",
                binder_chain="B",
                receptor_chain="A",
                reference_structure_path=tmp_path / "absent.cif",
            )
        messages = [str(w.message) for w in record]
        for step in ("binder pLDDT", "interface PDE", "interface PAE", "binder RMSD"):
            assert any(f"{step} skipped" in message for message in messages), step
        assert all(Path(w.filename) == Path(__file__) for w in record)
        # the four skipped steps, and first the token layout that the 12-token matrices do not fit
        assert metrics["reason"].startswith("token layout not built")
        assert metrics["reason"].count(";") == 4
