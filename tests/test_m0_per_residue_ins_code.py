"""Insertion codes in the per-residue interface entries and their labels (#117).

Residues 52 and 52A are two residues; ``per_residue`` used to carry no
insertion code, so both came out as "ALA:A:52" with ``res_id`` 52.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.protocols.report import _md_interface

HAS_BIOTITE = importlib.util.find_spec("biotite") is not None
requires_biotite = pytest.mark.skipif(not HAS_BIOTITE, reason="biotite not installed")

EXAMPLE_PDB = Path("data/example_linear_p53_1YCR.pdb")

# One alanine: (atom name, element, offset from the CA in Angstrom).
_ALANINE = [
    ("N", "N", (-1.2, 0.8, 0.0)),
    ("CA", "C", (0.0, 0.0, 0.0)),
    ("C", "C", (1.3, 0.6, 0.0)),
    ("O", "O", (1.4, 1.8, 0.0)),
    ("CB", "C", (-0.5, -1.0, 1.2)),
]


def _stacked_pdb(receptor_ids, peptide_ids) -> str:
    """Two facing rows of alanines, receptor chain A below peptide chain B.

    Each id is a ``(residue number, insertion code)`` pair. Residues sit 3.8 A
    apart along x and the rows 4.2 A apart along z, so every residue buries
    area against the other chain.
    """
    lines = []
    serial = 1
    for chain, ids, z in (("A", receptor_ids, 0.0), ("B", peptide_ids, 4.2)):
        for position, (number, ins_code) in enumerate(ids):
            for atom_name, element, (dx, dy, dz) in _ALANINE:
                x = 3.8 * position + dx
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:<4s} ALA {chain}{number:4d}{ins_code:1s}   "
                    f"{x:8.3f}{dy:8.3f}{z + dz:8.3f}  1.00  0.00          {element:>2s}"
                )
                serial += 1
    return "\n".join(lines) + "\nEND\n"


def _interface(tmp_path: Path, receptor_ids, peptide_ids) -> dict:
    from binding_metrics.metrics.interface import compute_interface_metrics

    path = tmp_path / "stacked.pdb"
    path.write_text(_stacked_pdb(receptor_ids, peptide_ids), encoding="utf-8")
    return compute_interface_metrics(path, design_chain="B", receptor_chain="A")


@requires_biotite
class TestInsertionCodes:
    def test_residues_52_and_52A_are_two_distinguishable_entries(self, tmp_path):
        result = _interface(tmp_path, [(1, ""), (2, ""), (3, "")], [(52, ""), (52, "A"), (53, "")])
        assert "reason" not in result
        peptide = [e for e in result["per_residue"] if e["chain"] == "B"]
        assert [e["residue"] for e in peptide] == ["ALA:B:52", "ALA:B:52A", "ALA:B:53"]
        assert [e["res_id"] for e in peptide] == [52, 52, 53]
        assert [e["ins_code"] for e in peptide] == ["", "A", ""]
        assert len({e["residue"] for e in peptide}) == 3
        # the two residues are separate rows with their own areas
        assert peptide[0]["buried_sasa"] != peptide[1]["buried_sasa"]
        assert result["n_interface_residues_peptide"] == 3
        assert result["interface_residues_peptide"] == ["ALA:B:52", "ALA:B:52A", "ALA:B:53"]

    def test_insertion_codes_on_the_receptor_chain(self, tmp_path):
        result = _interface(tmp_path, [(52, ""), (52, "A"), (52, "B")], [(1, ""), (2, ""), (3, "")])
        assert result["interface_residues_receptor"] == ["ALA:A:52", "ALA:A:52A", "ALA:A:52B"]
        receptor = [e for e in result["per_residue"] if e["chain"] == "A"]
        assert [e["ins_code"] for e in receptor] == ["", "A", "B"]

    def test_structure_without_insertion_codes_keeps_its_labels(self, tmp_path):
        result = _interface(tmp_path, [(1, ""), (2, ""), (3, "")], [(52, ""), (53, ""), (54, "")])
        assert result["interface_residues_peptide"] == ["ALA:B:52", "ALA:B:53", "ALA:B:54"]
        assert result["interface_residues_receptor"] == ["ALA:A:1", "ALA:A:2", "ALA:A:3"]
        assert all(e["ins_code"] == "" for e in result["per_residue"])

    @pytest.mark.integration
    def test_real_structure_labels_are_unchanged(self):
        """1YCR has no insertion codes: ``residue`` is NAME:CHAIN:NUM for every entry."""
        from binding_metrics.metrics.interface import compute_interface_metrics

        if not EXAMPLE_PDB.exists():
            pytest.skip("Test PDB not available")
        result = compute_interface_metrics(EXAMPLE_PDB)
        assert result["per_residue"]
        for entry in result["per_residue"]:
            assert entry["ins_code"] == ""
            assert entry["residue"] == f"{entry['res_name']}:{entry['chain']}:{entry['res_id']}"

    def test_residue_below_the_threshold_is_dropped_on_its_own(self):
        """52 and 52A are summed separately: only the one above the threshold is listed."""
        import biotite.structure as struc

        from binding_metrics.metrics.interface import _collect_per_residue

        atoms = struc.AtomArray(4)
        atoms.chain_id[:] = "B"
        atoms.res_name[:] = "ALA"
        atoms.res_id[:] = 52
        atoms.ins_code[:] = ["", "", "A", "A"]
        atoms.atom_name[:] = ["CA", "O", "CA", "O"]
        atoms.element[:] = ["C", "O", "C", "O"]
        buried = np.array([1.0, 0.5, 0.2, 0.1])
        entries = _collect_per_residue(atoms, buried, threshold=0.5)
        assert [e["residue"] for e in entries] == ["ALA:B:52"]
        assert entries[0]["ins_code"] == ""
        assert entries[0]["buried_sasa"] == pytest.approx(1.5)


def _row(res_id, ins_code, name="ALA"):
    row = {
        "chain": "B",
        "res_id": res_id,
        "res_name": name,
        "buried_sasa": 10.0,
        "polar_area": 4.0,
        "apolar_area": 6.0,
        "delta_g_res": -0.5,
    }
    if ins_code is not None:
        row["ins_code"] = ins_code
    return row


class TestReportTable:
    def test_the_table_tells_52_and_52A_apart_and_keeps_their_order(self):
        rows = [_row(53, ""), _row(52, "A"), _row(52, "")]
        text = _md_interface({"peptide_chain": "B", "per_residue": rows})
        assert text.index("| ALA:52 ") < text.index("| ALA:52A ") < text.index("| ALA:53 ")

    def test_entries_without_an_insertion_code_keep_the_old_label(self):
        """A result written before the key existed, or with a null code, prints NAME:NUM."""
        rows = [_row(2, None, "SER"), _row(1, None, "GLY"), _row(3, "", "VAL")]
        text = _md_interface({"peptide_chain": "B", "per_residue": rows})
        assert text.index("| GLY:1 ") < text.index("| SER:2 ") < text.index("| VAL:3 ")
        assert "None" not in text
