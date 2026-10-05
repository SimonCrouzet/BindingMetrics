"""Residue adjacency in the clashscore follows the residue sequence (#121).

It used to follow the difference of residue numbers, which ignores insertion codes.
"""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402

from binding_metrics.metrics import receptor_quality as rq  # noqa: E402


def _carbons(*records):
    """One carbon per record: (chain, residue number, insertion code, x)."""
    return struc.array(
        [
            struc.Atom(
                [x, 0.0, 0.0],
                chain_id=chain,
                res_id=number,
                ins_code=ins_code,
                res_name="ALA",
                atom_name="C1",
                element="C",
            )
            for chain, number, ins_code, x in records
        ]
    )


# 52 and 53 are two residues apart when 52A lies between them. The carbons at
# 0.0 and 2.8 A overlap by 0.6 A (radii sum 3.4 A).
def _three_residues(first, third, between_x=20.0):
    return _carbons(("A", *first, 0.0), ("A", 52, "A", between_x), ("A", *third, 2.8))


def test_residues_52_and_53_are_not_neighbours_when_52A_lies_between():
    """The contact between 52 and 53 is counted.

    Regression: the numbers 52 and 53 differ by one, so the pair was exempt as
    adjacent residues and a real overlap went unreported (0 clashes, now 1).
    """
    atoms = _three_residues((52, ""), (53, ""))
    result = rq._clashscore(atoms)
    assert result["n_clashes"] == 1
    assert result["clashscore"] == pytest.approx(1000.0 / 3)


def test_residues_52_and_52B_are_not_neighbours_when_52A_lies_between():
    atoms = _carbons(("A", 52, "", 0.0), ("A", 52, "A", 20.0), ("A", 52, "B", 2.8))
    assert rq._clashscore(atoms)["n_clashes"] == 1


def test_52_and_52A_are_neighbours_and_so_are_52A_and_53():
    assert rq._clashscore(_carbons(("A", 52, "", 0.0), ("A", 52, "A", 2.0)))["n_clashes"] == 0
    assert rq._clashscore(_carbons(("A", 52, "A", 0.0), ("A", 53, "", 2.0)))["n_clashes"] == 0


def test_52_and_53_without_an_insertion_between_are_neighbours():
    atoms = _carbons(("A", 52, "", 0.0), ("A", 53, "", 2.0), ("A", 60, "", 20.0))
    assert rq._clashscore(atoms)["n_clashes"] == 0


def test_residues_two_numbers_apart_are_still_counted():
    atoms = _carbons(("A", 52, "", 0.0), ("A", 54, "", 2.8))
    assert rq._clashscore(atoms)["n_clashes"] == 1


def test_neighbours_without_insertion_codes_follow_the_difference_of_numbers():
    """For numbers alone the new rule is the old one: same chain and |difference| <= 1."""
    rng = np.random.default_rng(10)
    chain = rng.choice(["A", "B"], size=60)
    number = rng.integers(1, 12, size=60)
    atoms = struc.array(
        [
            struc.Atom(
                [float(k), 0.0, 0.0],
                chain_id=str(c),
                res_id=int(n),
                res_name="ALA",
                atom_name="C1",
                element="C",
            )
            for k, (c, n) in enumerate(zip(chain, number))
        ]
    )
    i, j = np.triu_indices(len(atoms), k=1)
    expected = (atoms.chain_id[i] == atoms.chain_id[j]) & (
        np.abs(atoms.res_id[i] - atoms.res_id[j]) <= 1
    )
    np.testing.assert_array_equal(rq._neighbouring_residues(atoms, i, j), expected)
