"""Empty atom selections in the trajectory RMSD and contact metrics.

A selection that matches nothing used to return zeros, which reads as a
perfect fit or as "no contacts". The value is kept, but a RuntimeWarning names
the selection, and ``on_empty="raise"`` turns it into a ValueError.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from binding_metrics.metrics.contacts import calculate_contacts  # noqa: E402
from binding_metrics.metrics.rmsd import calculate_rmsd  # noqa: E402


def _write_pdb(path: Path, frames: list[list[tuple]]) -> Path:
    """Multi-model PDB; every atom is (res_name, atom_name, element, (x, y, z)) in Angstrom."""
    lines = []
    for model, atoms in enumerate(frames, start=1):
        lines.append(f"MODEL     {model:4d}")
        for serial, (res_name, atom_name, element, xyz) in enumerate(atoms, start=1):
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {res_name:>3s} A{serial:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
            )
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _chain(res_name: str, frame_shift: float = 0.0) -> list[tuple]:
    """Four heavy atoms of one residue, each on a different site so RMSD can be non-zero."""
    sites = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.0, 1.3, 0.0), (1.5, 2.3, 0.5)]
    names = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
    return [
        (res_name, name, element, (x + frame_shift * i, y, z))
        for i, ((name, element), (x, y, z)) in enumerate(zip(names, sites))
    ]


@pytest.fixture
def unrecognised_residues(tmp_path):
    """Two frames of residues whose name mdtraj's ``protein`` selection does not know."""
    frames = [_chain("XYZ"), _chain("XYZ", frame_shift=0.3)]
    return _write_pdb(tmp_path / "xyz.pdb", frames)


@pytest.fixture
def standard_residues(tmp_path):
    frames = [_chain("ALA"), _chain("ALA", frame_shift=0.3)]
    return _write_pdb(tmp_path / "ala.pdb", frames)


class TestRmsdEmptySelection:
    def test_default_selection_matches_nothing_for_unrecognised_residues(
        self, unrecognised_residues
    ):
        """Guards the premise of the other tests."""
        top = md.load(str(unrecognised_residues)).topology
        assert len(top.select("protein and not type H")) == 0

    def test_warns_naming_the_default_selection_and_keeps_zeros(self, unrecognised_residues):
        with pytest.warns(RuntimeWarning, match="protein and not type H"):
            result = calculate_rmsd(unrecognised_residues, unrecognised_residues)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_warns_naming_explicit_empty_indices(self, standard_residues):
        with pytest.warns(RuntimeWarning, match="atom_indices"):
            result = calculate_rmsd(standard_residues, standard_residues, atom_indices=[])
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_raise_mode(self, unrecognised_residues):
        with pytest.raises(ValueError, match="protein and not type H"):
            calculate_rmsd(unrecognised_residues, unrecognised_residues, on_empty="raise")

    def test_warning_points_at_the_caller(self, unrecognised_residues):
        with pytest.warns(RuntimeWarning) as record:
            calculate_rmsd(unrecognised_residues, unrecognised_residues)
        assert Path(record[0].filename) == Path(__file__)

    def test_non_empty_selection_is_silent_and_real(self, standard_residues):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = calculate_rmsd(standard_residues, standard_residues)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] > 0.005  # nm; frame 2 is deformed, so the fit is not perfect

    def test_invalid_mode_is_rejected(self, standard_residues):
        with pytest.raises(ValueError, match="on_empty"):
            calculate_rmsd(standard_residues, standard_residues, on_empty="ignore")


@pytest.fixture
def contact_system(tmp_path):
    """Ligand: heavy atom 0 and hydrogen 1; receptor: atoms 2 and 3. Two frames.

    Frame 1: ligand C-receptor C 3.0 A, ligand H-receptor C 3.16 A, atom 3 far away.
    Frame 2: receptor atom 2 moved 2 A further out, so nothing is in contact.
    """
    ligand = [("ALA", "C1", "C", (0.0, 0.0, 0.0)), ("ALA", "H1", "H", (0.0, 0.0, 1.0))]
    near = ("GLY", "C2", "C", (3.0, 0.0, 0.0))
    far = ("GLY", "C3", "C", (10.0, 0.0, 0.0))
    moved = ("GLY", "C2", "C", (5.0, 0.0, 0.0))
    path = _write_pdb(tmp_path / "contacts.pdb", [ligand + [near, far], ligand + [moved, far]])
    return path


class TestContactsEmptySelection:
    def test_empty_ligand_warns_and_names_it(self, contact_system):
        with pytest.warns(RuntimeWarning, match="ligand_indices"):
            result = calculate_contacts(contact_system, contact_system, [], [2, 3])
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_empty_receptor_warns_and_names_it(self, contact_system):
        with pytest.warns(RuntimeWarning, match="receptor_indices"):
            result = calculate_contacts(contact_system, contact_system, [0, 1], [])
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_both_empty_names_both(self, contact_system):
        with pytest.warns(RuntimeWarning, match="ligand_indices and receptor_indices"):
            calculate_contacts(contact_system, contact_system, [], [])

    def test_raise_mode(self, contact_system):
        with pytest.raises(ValueError, match="ligand_indices"):
            calculate_contacts(contact_system, contact_system, [], [2, 3], on_empty="raise")

    def test_invalid_mode_is_rejected(self, contact_system):
        with pytest.raises(ValueError, match="on_empty"):
            calculate_contacts(contact_system, contact_system, [0], [2], on_empty="ignore")

    def test_non_empty_selection_is_silent(self, contact_system):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            calculate_contacts(contact_system, contact_system, [0], [2])

    def test_counts_exactly_the_atoms_given(self, contact_system):
        """Hydrogens are counted when their indices are passed, and only then."""
        heavy_only = calculate_contacts(contact_system, contact_system, [0], [2, 3])
        with_hydrogen = calculate_contacts(contact_system, contact_system, [0, 1], [2, 3])
        np.testing.assert_array_equal(heavy_only, [1.0, 0.0])
        np.testing.assert_array_equal(with_hydrogen, [2.0, 0.0])
