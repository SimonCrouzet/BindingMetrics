"""Residues are counted by (number, insertion code) when the chains are sized (#120).

Chain A below has six residues but only four residue numbers (3, 3A and 3B
share a number); chain B has five. Counting numbers made B the larger chain,
counting residues makes A the larger one.
"""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402

from binding_metrics.metrics import receptor_quality as rq  # noqa: E402
from binding_metrics.metrics.interface import detect_interface_chains  # noqa: E402
from tests.test_receptor_quality import _fake_energy, _helix  # noqa: E402

SIX_RESIDUES_FOUR_NUMBERS = [(1, ""), (2, ""), (3, ""), (3, "A"), (3, "B"), (4, "")]
FIVE_RESIDUES = [(n, "") for n in range(1, 6)]


def _chain(chain_id, ids):
    """One CA per residue, from (residue number, insertion code) pairs."""
    arr = struc.AtomArray(len(ids))
    arr.chain_id[:] = chain_id
    arr.res_id = np.array([number for number, _ in ids])
    arr.ins_code = np.array([code for _, code in ids])
    arr.res_name[:] = "ALA"
    arr.atom_name[:] = "CA"
    arr.element[:] = "C"
    return arr


def _two_chains():
    return struc.concatenate([_chain("A", SIX_RESIDUES_FOUR_NUMBERS), _chain("B", FIVE_RESIDUES)])


def test_the_chain_with_more_residues_is_the_receptor_whatever_its_numbering():
    """Before the fix chain A counted four residues, so it was taken as the peptide."""
    assert detect_interface_chains(_two_chains()) == ("B", "A")


def test_an_explicit_design_chain_still_leaves_the_other_chain_as_receptor():
    assert detect_interface_chains(_two_chains(), design_chain="A") == ("A", "B")


def test_chains_without_insertion_codes_are_sized_as_before():
    atoms = struc.concatenate(
        [_chain("A", [(n, "") for n in range(1, 7)]), _chain("B", FIVE_RESIDUES)]
    )
    assert detect_interface_chains(atoms) == ("B", "A")


def test_receptor_quality_picks_the_chain_with_more_residues():
    assert rq._detect_receptor_chain(_two_chains()) == "A"


def test_receptor_quality_picks_the_same_chain_without_insertion_codes():
    atoms = struc.concatenate(
        [_chain("A", [(n, "") for n in range(1, 7)]), _chain("B", FIVE_RESIDUES)]
    )
    assert rq._detect_receptor_chain(atoms) == "A"


def test_n_residues_of_a_model_counts_insertion_codes(monkeypatch):
    """A helix of six residues numbered 1, 2, 3, 3A, 3B, 4 has six residues, not four."""
    _fake_energy(monkeypatch)
    helix = _helix(6, res_ids=[1, 2, 3, 3, 3, 4])
    helix.ins_code = np.repeat(np.array(["", "", "", "A", "B", ""]), 5)  # five atoms per residue
    model = rq._score_model(helix, "A", 0.4, "obc2", "cuda", 1)
    assert model["n_residues"] == 6


def test_n_residues_without_insertion_codes_is_the_number_of_residues(monkeypatch):
    _fake_energy(monkeypatch)
    model = rq._score_model(_helix(6), "A", 0.4, "obc2", "cuda", 1)
    assert model["n_residues"] == 6
