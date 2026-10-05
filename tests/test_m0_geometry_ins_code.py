"""Insertion codes in the per-residue entries of the Ramachandran and omega checks."""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics.geometry import (  # noqa: E402
    compute_omega_planarity,
    compute_ramachandran,
)
from binding_metrics.protocols.report import _md_geometry  # noqa: E402
from tests.test_receptor_quality import _helix  # noqa: E402


@pytest.fixture
def helix_with_insertions(tmp_path):
    """Six residues of an alpha helix numbered 1, 2, 3, 3A, 3B, 4."""
    atoms = _helix(6, res_ids=[1, 2, 3, 3, 3, 4])
    atoms.ins_code = np.repeat(np.array(["", "", "", "A", "B", ""]), 5)  # five atoms a residue
    path = tmp_path / "helix.pdb"
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, atoms)
    pdb_file.write(str(path))
    return path


def test_ramachandran_entries_carry_the_insertion_code(helix_with_insertions):
    """Residues 3, 3A and 3B were three entries with the same chain and res_id 3."""
    result = compute_ramachandran(helix_with_insertions, chain="A")
    keys = [(e["res_id"], e["ins_code"]) for e in result["per_residue"]]
    assert keys == [(2, ""), (3, ""), (3, "A"), (3, "B")]
    assert len(set(keys)) == len(keys)


def test_omega_entries_carry_the_insertion_code(helix_with_insertions):
    result = compute_omega_planarity(helix_with_insertions, chain="A")
    keys = [(e["res_id"], e["ins_code"]) for e in result["per_residue"]]
    assert keys == [(1, ""), (2, ""), (3, ""), (3, "A"), (3, "B")]


def test_entries_without_insertion_codes_have_an_empty_one(tmp_path):
    atoms = _helix(5)
    path = tmp_path / "plain.pdb"
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, atoms)
    pdb_file.write(str(path))
    for result in (
        compute_ramachandran(path, chain="A"),
        compute_omega_planarity(path, chain="A"),
    ):
        assert result["per_residue"]
        assert all(e["ins_code"] == "" for e in result["per_residue"])


def test_report_names_an_outlier_with_its_insertion_code():
    outlier = {"res_id": 52, "ins_code": "A", "res_name": "ALA", "region": "outlier"}
    outlier.update({"phi": 60.0, "psi": -120.0, "omega": 150.0, "deviation": 30.0})
    outlier["is_outlier"] = True
    text = _md_geometry(
        {"ramachandran": {"per_residue": [outlier]}, "omega": {"per_residue": [outlier]}}
    )
    assert "Rama outlier:** ALA52A " in text
    assert "ω outlier:** ALA52A " in text
