"""Chain detection is built on amino-acid membership, not on a fixed residue-name list."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402

from binding_metrics.metrics.interface import (  # noqa: E402
    detect_interface_chains,
    load_biotite_structure,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def _chain(chain_id, res_name, n_residues, first_res_id=1, atoms_per_residue=1, hetero=False):
    """One-atom-per-residue chain; CA for amino acids, O for water."""
    n = n_residues * atoms_per_residue
    arr = struc.AtomArray(n)
    arr.chain_id[:] = chain_id
    arr.res_id = np.repeat(np.arange(first_res_id, first_res_id + n_residues), atoms_per_residue)
    arr.res_name[:] = res_name
    arr.atom_name[:] = "O" if res_name == "HOH" else "CA"
    arr.element[:] = "O" if res_name == "HOH" else "C"
    arr.hetero[:] = hetero
    return arr


def _complex(*chains):
    return struc.concatenate(list(chains))


def test_all_d_peptide_chain_is_recognised():
    atoms = _complex(_chain("A", "ALA", 20), _chain("B", "DAL", 5))
    assert detect_interface_chains(atoms) == ("B", "A")


@pytest.mark.parametrize(
    "res_name", ["MLE", "SAR", "BMT", "ABA", "MSE", "SEP", "TPO", "PTR", "HYP", "MLY", "DAR", "DLY"]
)
def test_non_canonical_peptide_residues_count_as_protein(res_name):
    atoms = _complex(_chain("A", "ALA", 20), _chain("B", res_name, 4))
    assert detect_interface_chains(atoms) == ("B", "A")


def test_amber_histidine_and_cystine_names_count_as_protein():
    atoms = _complex(_chain("A", "ALA", 20), _chain("B", "HIE", 3), _chain("C", "CYX", 2))
    assert detect_interface_chains(atoms, design_chain="C") == ("C", "A")
    assert detect_interface_chains(atoms)[0] == "C"


def test_waters_sharing_the_peptide_chain_id_do_not_inflate_its_size():
    peptide = _chain("B", "ALA", 5)
    waters = _chain("B", "HOH", 40, first_res_id=100, hetero=True)
    atoms = _complex(_chain("A", "ALA", 20), peptide, waters)
    assert detect_interface_chains(atoms) == ("B", "A")


def test_chain_with_only_water_or_ligand_is_not_a_protein_chain():
    atoms = _complex(_chain("A", "ALA", 20), _chain("B", "ALA", 5), _chain("W", "HOH", 60))
    assert detect_interface_chains(atoms) == ("B", "A")


def test_single_protein_chain_has_no_receptor():
    assert detect_interface_chains(_chain("A", "DAL", 6)) == ("A", None)


def test_explicit_design_chain_picks_the_largest_other_protein_chain():
    atoms = _complex(_chain("A", "ALA", 20), _chain("B", "ALA", 5), _chain("C", "ALA", 12))
    assert detect_interface_chains(atoms, design_chain="B") == ("B", "A")
    assert detect_interface_chains(atoms, design_chain="A") == ("A", "C")


def test_bundled_cyclosporin_complex_is_unchanged():
    atoms = load_biotite_structure(DATA / "example_ncaa_cyclosporin_1CWA.cif")
    assert detect_interface_chains(atoms) == ("C", "A")
