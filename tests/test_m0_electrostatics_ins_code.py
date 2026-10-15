"""Insertion codes in the charged-atom entries of the Coulomb energy (#119)."""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain  # noqa: E402


def _write_pdb(tmp_path, records):
    """Write (chain, res_id, ins_code, res_name, atom_name, element, xyz) records as a PDB file."""
    arr = struc.AtomArray(len(records))
    arr.chain_id = np.array([r[0] for r in records])
    arr.res_id = np.array([r[1] for r in records], dtype=int)
    arr.ins_code = np.array([r[2] for r in records])
    arr.res_name = np.array([r[3] for r in records])
    arr.atom_name = np.array([r[4] for r in records])
    arr.element = np.array([r[5] for r in records])
    arr.coord = np.array([r[6] for r in records], dtype=float)
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, arr)
    path = tmp_path / "complex.pdb"
    pdb_file.write(str(path))
    return path


def _complex(peptide_ids):
    """Chain A: one aspartate. Chain B: a lysine and an arginine numbered by ``peptide_ids``."""
    (lys_id, lys_ins), (arg_id, arg_ins) = peptide_ids
    return [
        ("A", 1, "", "ASP", "OD1", "O", (0.0, 0.0, 0.0)),
        ("A", 1, "", "ASP", "OD2", "O", (1.0, 0.0, 0.0)),
        ("B", lys_id, lys_ins, "LYS", "NZ", "N", (4.0, 0.0, 0.0)),
        ("B", arg_id, arg_ins, "ARG", "NH1", "N", (4.0, 3.0, 0.0)),
    ]


def test_residues_52_and_52A_have_different_labels(tmp_path):
    """Charged atoms of residue 52 and of residue 52A carry the insertion code.

    Before the fix the label had the number only, so an atom of 52A could not
    be told from an atom of 52 of the same name.
    """
    path = _write_pdb(tmp_path, _complex([(52, ""), (52, "A")]))
    result = compute_coulomb_cross_chain(path, "B", "A")
    entries = result["charged_atoms_peptide"]
    assert [e["residue"] for e in entries] == ["LYS:B:52", "ARG:B:52A"]
    assert [e["ins_code"] for e in entries] == ["", "A"]
    assert [e["atom"] for e in entries] == ["NZ", "NH1"]
    assert result["n_ionisable_residues_seen"] == 3  # ASP, LYS, ARG: three residues
    assert result["n_charged_pairs"] > 0


def test_same_residue_name_and_number_with_two_insertion_codes(tmp_path):
    """Two lysines 52 and 52A: the labels differ although name and number agree."""
    records = _complex([(52, ""), (52, "A")])
    records[3] = ("B", 52, "A", "LYS", "NZ", "N", (4.0, 3.0, 0.0))
    result = compute_coulomb_cross_chain(_write_pdb(tmp_path, records), "B", "A")
    labels = [e["residue"] for e in result["charged_atoms_peptide"]]
    assert labels == ["LYS:B:52", "LYS:B:52A"]


def test_labels_without_insertion_codes_are_unchanged(tmp_path):
    path = _write_pdb(tmp_path, _complex([(1, ""), (2, "")]))
    result = compute_coulomb_cross_chain(path, "B", "A")
    assert [e["residue"] for e in result["charged_atoms_peptide"]] == ["LYS:B:1", "ARG:B:2"]
    assert [e["residue"] for e in result["charged_atoms_receptor"]] == ["ASP:A:1", "ASP:A:1"]
    assert all(e["ins_code"] == "" for e in result["charged_atoms_peptide"])
    assert all(e["ins_code"] == "" for e in result["charged_atoms_receptor"])


def test_the_energy_does_not_depend_on_the_insertion_code(tmp_path):
    """The code only labels: the same atoms numbered 1, 2 give the same energy."""
    with_codes = compute_coulomb_cross_chain(
        _write_pdb(tmp_path, _complex([(52, ""), (52, "A")])), "B", "A"
    )
    without = compute_coulomb_cross_chain(
        _write_pdb(tmp_path, _complex([(1, ""), (2, "")])), "B", "A"
    )
    assert with_codes["coulomb_energy_kJ"] == pytest.approx(without["coulomb_energy_kJ"])
