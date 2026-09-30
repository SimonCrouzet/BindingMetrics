"""Array-valued result fields stay out of the CSV row (issue #91).

``_flatten`` used to keep a numpy array as ``str(array)``, which numpy cuts to
``[50. 50.03 ... 89.97 90.]`` once it holds more than 1000 values: a cell that carries
neither the data nor a marker that data is missing. Arrays are now skipped like list
fields and stay in the JSON.
"""

import csv
import json
from pathlib import Path

import numpy as np

from binding_metrics.cli import batch
from binding_metrics.protocols.report import _flatten, write_report

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

N_ATOMS = 1500  # numpy summarises with "..." above 1000 elements
ARRAY_FIELDS = (
    "plddt_per_atom",
    "binder_plddt_per_residue",
    "pde",
    "pae",
    "pde_interface",
    "pae_interface",
)


def _openfold_metrics() -> dict:
    """The dictionary of ``compute_openfold_metrics`` with every array field filled."""
    plddt = np.linspace(50.0, 90.0, N_ATOMS)
    matrix = np.linspace(0.5, 20.0, 40 * 40).reshape(40, 40)
    return {
        "structure_path": "of3/model.cif",
        "avg_plddt": float(plddt.mean()),
        "iptm": 0.81,
        "n_atoms": N_ATOMS,
        "chain_ptm": {"A": 0.9, "B": 0.7},
        "plddt_per_atom": plddt,
        "binder_plddt_per_residue": np.linspace(60.0, 95.0, 1500),
        "pde": matrix,
        "pae": matrix.T,
        "pde_interface": matrix[:10, 10:],
        "pae_interface": matrix[10:, :10],
    }


class TestFlatten:
    def test_the_six_array_columns_are_absent(self):
        flat = _flatten({"openfold": _openfold_metrics()})
        for field in ARRAY_FIELDS:
            assert f"openfold_{field}" not in flat

    def test_scalars_and_nested_scalars_are_kept(self):
        flat = _flatten({"openfold": _openfold_metrics()})
        assert flat["openfold_iptm"] == 0.81
        assert flat["openfold_n_atoms"] == N_ATOMS
        assert flat["openfold_chain_ptm_A"] == 0.9

    def test_no_cell_is_a_truncated_array(self):
        flat = _flatten({"openfold": _openfold_metrics()})
        assert not [key for key, value in flat.items() if "..." in str(value)]

    def test_a_list_field_is_still_skipped(self):
        flat = _flatten({"openfold": {"iptm": 0.5, "flags": [1, 2, 3]}})
        assert "openfold_flags" not in flat


class TestWriteReport:
    def test_csv_has_no_array_column_and_json_keeps_every_value(self, tmp_path):
        results = {"sample_id": "s1", "input": "s1.cif", "openfold": _openfold_metrics()}
        csv_path = write_report(results, tmp_path / "csv", "s1", fmt="csv")
        json_path = write_report(results, tmp_path / "json", "s1", fmt="json")

        with open(csv_path, newline="", encoding="utf-8") as handle:
            (row,) = list(csv.DictReader(handle))
        assert row["openfold_iptm"] == "0.81"
        assert not [name for name in row if name.removeprefix("openfold_") in ARRAY_FIELDS]
        assert "..." not in csv_path.read_text(encoding="utf-8")

        stored = json.loads(json_path.read_text(encoding="utf-8"))["openfold"]
        assert len(stored["plddt_per_atom"]) == N_ATOMS
        assert len(stored["pae"]) == 40 and len(stored["pae"][0]) == 40


class TestBatchRow:
    """The batch row is built from the same ``_flatten``; the OpenFold merge shares it."""

    def test_batched_openfold_row_has_no_array_column(self, tmp_path, monkeypatch):
        from binding_metrics.metrics import evobind as evobind_module
        from binding_metrics.metrics import openfold

        metrics = _openfold_metrics()
        monkeypatch.setattr(openfold, "run_openfold_batched", lambda **kw: tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: dict(metrics))
        monkeypatch.setattr(evobind_module, "compute_evobind_score", lambda *a, **kw: {})
        monkeypatch.setattr(evobind_module, "compute_evobind_adversarial_check", lambda **kw: {})
        rows = [{"sample_id": "s1", "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={"s1": EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain="B",
            receptor_chain="A",
        )
        (row,) = rows
        assert row["openfold_iptm"] == 0.81
        assert not [name for name in row if name.removeprefix("openfold_") in ARRAY_FIELDS]
        assert not [name for name, value in row.items() if "..." in str(value)]
