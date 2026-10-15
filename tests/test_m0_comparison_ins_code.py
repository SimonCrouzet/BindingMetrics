"""Insertion codes in the atom key of the structure RMSD (#118).

The key was (chain, residue number, atom name), so residues 52 and 52A of one
chain shared every key.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics.comparison import (
    _get_coords,
    _kabsch_rmsd,
    _matched_rmsd,
    compute_structure_rmsd,
)

requires_gemmi = pytest.mark.skipif(
    importlib.util.find_spec("gemmi") is None, reason="gemmi not installed"
)

_BACKBONE = [
    ("N", "N", (-1.2, 0.8, 0.0)),
    ("CA", "C", (0.0, 0.0, 0.0)),
    ("C", "C", (1.3, 0.6, 0.0)),
]

# (residue number, insertion code, x, y): 52A is 5 A away from 52 in y, so
# pairing one with the other cannot superpose.
R52 = (52, "", 0.0, 0.0)
R52A = (52, "A", 3.8, 5.0)
R53 = (53, "", 7.6, 0.0)


def _write(path: Path, residues, chain: str = "A") -> Path:
    """PDB file of chain ``chain`` with three backbone atoms per residue, in the order given."""
    lines = []
    serial = 1
    for number, ins_code, x, y in residues:
        for name, element, (dx, dy, dz) in _BACKBONE:
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} ALA {chain}{number:4d}{ins_code:1s}   "
                f"{x + dx:8.3f}{y + dy:8.3f}{dz:8.3f}  1.00  0.00          {element:>2s}"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


@requires_gemmi
class TestInsertionCodeKey:
    def test_keys_carry_the_insertion_code(self, tmp_path):
        import gemmi

        path = _write(tmp_path / "a.pdb", [R52, R52A, R53])
        _, keys = _get_coords(gemmi.read_structure(str(path)))
        assert len(set(keys)) == len(keys)
        assert ("A", 52, "CA", "") in keys
        assert ("A", 52, "CA", "A") in keys
        assert all(key[3] in ("", "A") for key in keys)

    def test_same_residues_in_another_order_give_zero(self, tmp_path):
        """52A written before 52: the key lists agree in number and atom only.

        Regression: with the insertion code missing from the key the two lists
        compared equal, the atoms were paired by position (52 with 52A) and
        every value came back at 1.38 A instead of 0.
        """
        first = _write(tmp_path / "a.pdb", [R52, R52A, R53])
        second = _write(tmp_path / "b.pdb", [R52A, R52, R53])
        result = compute_structure_rmsd(first, second, design_chain="A")
        for name in ("rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"):
            assert result[name] is not None
            assert result[name] < 1e-2, name  # PDB coordinates have 3 decimals
        assert "reason" not in result

    def test_residue_missing_from_one_file_is_paired_by_insertion_code(self, tmp_path):
        """52 is absent from the second file; 52A and 53 must pair with 52A and 53.

        Before the fix the two 52A/52 atoms of the first file shared a key with
        the single residue 52 of the second one, which then paired with 52.
        """
        first = _write(tmp_path / "a.pdb", [R52, R52A, R53])
        second = _write(tmp_path / "b.pdb", [R52A, R53])
        result = compute_structure_rmsd(first, second, design_chain="A")
        assert result["rmsd"] is not None
        assert result["rmsd"] < 1e-2

    def test_a_number_with_another_insertion_code_is_another_residue(self, tmp_path):
        first = _write(tmp_path / "a.pdb", [R52, R53])
        second = _write(tmp_path / "b.pdb", [R52A])
        result = compute_structure_rmsd(first, second, design_chain="A")
        assert result["rmsd"] is None
        assert result["reason"].startswith("not computed:")

    def test_files_without_insertion_codes_give_the_direct_kabsch_value(self, tmp_path):
        import gemmi

        rng = np.random.default_rng(8)
        residues = [(n, "", 3.8 * n, float(rng.normal())) for n in range(1, 9)]
        moved = [(n, "", x + float(rng.normal()) * 0.3, y) for n, _, x, y in residues]
        first = _write(tmp_path / "a.pdb", residues)
        second = _write(tmp_path / "b.pdb", moved)
        coords1, keys1 = _get_coords(gemmi.read_structure(str(first)))
        coords2, keys2 = _get_coords(gemmi.read_structure(str(second)))
        assert all(key[3] == "" for key in keys1)
        assert keys1 == keys2
        assert _matched_rmsd(coords1, keys1, coords2, keys2) == _kabsch_rmsd(coords1, coords2)
        result = compute_structure_rmsd(first, second, design_chain="A")
        assert result["rmsd"] == pytest.approx(_kabsch_rmsd(coords1, coords2))
