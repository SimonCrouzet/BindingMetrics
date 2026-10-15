"""A receptor chain with a hole has no energy (#166).

With PDBFixer, ``removeHeterogens`` deletes a modified residue that PDBFixer does not
replace (IAM of 1XY4, 0EH of 3V3B), and the energy was that of a chain with a hole: the two
residues around it bonded by name, 3 A apart. The energy is now NaN, with a ``reason`` that
names the removed residues. A residue removed at the end of the chain leaves no hole and
changes nothing.
"""

import numpy as np
import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")
pytest.importorskip("biotite")

import biotite.structure as struc
import biotite.structure.io as sio
from conftest import DATA_DIR

from binding_metrics.metrics import receptor_quality

pytestmark = pytest.mark.slow


def _chain_atoms(path, chain: str):
    atoms = sio.load_structure(str(path))
    atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
    return atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == chain)]


class TestRemovedFromInside:
    @staticmethod
    def _residues(*names):
        return [("A", str(number), "", name) for number, name in enumerate(names, start=1)]

    def test_a_residue_between_kept_ones_is_a_hole(self):
        before = self._residues("ALA", "IAM", "GLY")
        after = [before[0], before[2]]
        assert receptor_quality._removed_from_inside_chain(before, after) == [before[1]]

    def test_the_ends_of_the_chain_are_not(self):
        before = self._residues("NME", "ALA", "GLY", "ACE")
        after = [before[1], before[2]]
        assert receptor_quality._removed_from_inside_chain(before, after) == []

    def test_each_chain_is_judged_alone(self):
        a = [("A", "1", "", "ALA"), ("A", "2", "", "GLY")]
        b = [("B", "1", "", "ALA"), ("B", "2", "", "IAM"), ("B", "3", "", "GLY")]
        assert receptor_quality._removed_from_inside_chain(a + b, a + [b[0], b[2]]) == [b[1]]

    def test_nothing_removed_nothing_found(self):
        before = self._residues("ALA", "GLY")
        assert receptor_quality._removed_from_inside_chain(before, before) == []


class TestAHoleGivesNoEnergy:
    @pytest.mark.parametrize(
        ("name", "path", "chain", "residue"),
        [
            ("1XY4", DATA_DIR / "example_lactam_somatostatin_1XY4.cif", "A", "IAM 7"),
            ("3V3B", DATA_DIR / "example_staple_3V3B.pdb", "C", "0EH 20"),
        ],
    )
    def test_the_removed_residue_is_named(self, name, path, chain, residue):
        result = receptor_quality._receptor_energy(_chain_atoms(path, chain), device="cpu")

        assert np.isnan(result["energy_kJ_mol"])
        assert np.isnan(result["energy_per_residue_kJ_mol"])
        assert result["n_atoms_with_h"] is None
        assert residue in result["reason"]
        assert "removed" in result["reason"] and "hole" in result["reason"]
        assert result["error"] == result["reason"]

    @pytest.mark.parametrize(
        ("path", "chain"),
        [
            (DATA_DIR / "example_linear_p53_1YCR.pdb", "A"),
            (DATA_DIR / "example_bicyclic_sfti1_3P8F.cif", "A"),
            (DATA_DIR / "example_phospho_1QJB.pdb", "Q"),
        ],
        ids=["1YCR", "3P8F", "1QJB"],
    )
    def test_the_receptor_chains_of_the_examples_keep_their_energy(self, path, chain):
        """A chain without a hole gets an energy (its value depends on the pdbfixer version)."""
        result = receptor_quality._receptor_energy(_chain_atoms(path, chain), device="cpu")

        assert result["error"] is None
        assert "reason" not in result or result["reason"] is None
        assert np.isfinite(result["energy_kJ_mol"])
        assert result["n_atoms_with_h"] > 0

    def test_the_receptor_chain_of_cyclosporin_complex_keeps_its_energy(self):
        """Cyclophilin A of 1CWA has no modified residue; it needs the CIF's author chain A."""
        result = receptor_quality._receptor_energy(
            _chain_atoms(DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif", "A"), device="cpu"
        )
        assert result["error"] is None
        assert np.isfinite(result["energy_kJ_mol"])
