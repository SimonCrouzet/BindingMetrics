"""Salt-bridge residue pairs are told apart by number and insertion code."""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402

from binding_metrics.metrics.polar_contacts import compute_saltbridges  # noqa: E402

COULOMB_KCAL = 83.0159  # unit charges at epsilon 4, per 1/Angstrom


def _atoms(records):
    """AtomArray from (chain, res_id, ins_code, res_name, atom_name, element, xyz) tuples."""
    arr = struc.AtomArray(len(records))
    arr.chain_id = np.array([r[0] for r in records])
    arr.res_id = np.array([r[1] for r in records], dtype=int)
    arr.ins_code = np.array([r[2] for r in records])
    arr.res_name = np.array([r[3] for r in records])
    arr.atom_name = np.array([r[4] for r in records])
    arr.element = np.array([r[5] for r in records])
    arr.coord = np.array([r[6] for r in records], dtype=float)
    return arr


def _two_lysines(second_ins_code, second_number):
    """Two lysines of chain B in contact with one aspartate of chain A, at 3.0 and 3.5 A."""
    return _atoms(
        [
            ("A", 1, "", "ASP", "OD1", "O", (0.0, 0.0, 0.0)),
            ("B", 52, "", "LYS", "NZ", "N", (3.0, 0.0, 0.0)),
            ("B", second_number, second_ins_code, "LYS", "NZ", "N", (0.0, 3.5, 0.0)),
        ]
    )


def test_lysines_52_and_52A_are_two_salt_bridges():
    """Both lysines bridge the aspartate; they were counted as one pair of residues.

    Regression: the pair key was (chain, number, name), the same for 52 and
    52A, so the two contacts made one bidentate pair: 1 salt bridge (1
    bidentate) at -27.7 kcal/mol instead of 2 (0 bidentate) at -27.7 - 23.7 =
    -51.4 kcal/mol.
    """
    result = compute_saltbridges(_two_lysines("A", 52), "B", "A")
    assert result["saltbridges"] == 2
    assert result["saltbridges_bidentate"] == 0
    expected = -COULOMB_KCAL * (1.0 / 3.0 + 1.0 / 3.5)
    assert result["saltbridge_energy"] == pytest.approx(expected, rel=1e-4)


def test_numbers_alone_give_the_same_result():
    """The same atoms numbered 52 and 53 are two salt bridges, as before."""
    with_codes = compute_saltbridges(_two_lysines("A", 52), "B", "A")
    numbers_only = compute_saltbridges(_two_lysines("", 53), "B", "A")
    assert numbers_only["saltbridges"] == with_codes["saltbridges"] == 2
    assert numbers_only["saltbridge_energy"] == pytest.approx(with_codes["saltbridge_energy"])


def test_two_atoms_of_one_residue_with_an_insertion_code_stay_one_pair():
    atoms = _atoms(
        [
            ("A", 1, "", "ASP", "OD1", "O", (0.0, 0.0, 0.0)),
            ("A", 1, "", "ASP", "OD2", "O", (0.0, 1.0, 0.0)),
            ("B", 52, "A", "ARG", "NH1", "N", (3.0, 0.0, 0.0)),
            ("B", 52, "A", "ARG", "NH2", "N", (3.0, 1.0, 0.0)),
        ]
    )
    result = compute_saltbridges(atoms, "B", "A")
    assert result["saltbridges"] == 1
    assert result["saltbridges_bidentate"] == 1
