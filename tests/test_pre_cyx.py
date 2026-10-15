"""``detect_cyclization`` finds a disulfide whatever the name of its cysteines (CYS, CYX, DCY).

AMBER-prepared inputs name a disulfide cysteine CYX, and a D-cysteine is DCY in the raw file. The
detector used to look for the name CYS only, so it missed the SG-SG pair in step 2 and raised
``CyclizationError`` for it in the topology scan of step 5.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from binding_metrics.core import cyclic  # noqa: E402
from binding_metrics.core.nonstandard import D_AA_MAP  # noqa: E402
from binding_metrics.io.structures import load_structure  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
# 3P8F: the peptide is chain B for OpenMM and I for biotite (label and author chain IDs); the
# trypsin chain A holds three disulfides of its own.
PEPTIDE = ("B", "I")  # (OpenMM chain, biotite chain)
TRYPSIN = ("A", "A")


def _rename_cysteines(topology, chain_id, new_name, *, only=None):
    """Rename the CYS residues of a chain in place (``only``: indices among the cysteines)."""
    chain = next(c for c in topology.chains() if c.id == chain_id)
    cysteines = [r for r in chain.residues() if r.name == "CYS"]
    for k, residue in enumerate(cysteines):
        if only is None or k in only:
            residue.name = new_name


def _links(topology, positions, chain_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = cyclic.detect_cyclization(topology, positions, chain_id)
    return sorted((i.cyclic_type, tuple(sorted([i.atom1_id[1:], i.atom2_id[1:]]))) for i in found)


def test_the_examples_names_are_found_as_loaded():
    topology, positions = load_structure(SFTI1)
    kinds = [kind for kind, _ in _links(topology, positions, PEPTIDE[0])]
    assert kinds == ["disulfide", "head_to_tail"]


@pytest.mark.parametrize("name", ["CYX", "DCY"])
def test_renamed_disulfide_cysteines_give_the_same_closures(name):
    topology, positions = load_structure(SFTI1)
    expected = _links(topology, positions, PEPTIDE[0])
    _rename_cysteines(topology, PEPTIDE[0], name)
    assert _links(topology, positions, PEPTIDE[0]) == expected


def test_a_cys_cyx_pair_is_a_disulfide_too():
    topology, positions = load_structure(SFTI1)
    expected = _links(topology, positions, PEPTIDE[0])
    _rename_cysteines(topology, PEPTIDE[0], "CYX", only={0})
    assert _links(topology, positions, PEPTIDE[0]) == expected


def test_a_protein_with_three_disulfides_keeps_them_under_cyx_names():
    topology, positions = load_structure(SFTI1)
    expected = _links(topology, positions, TRYPSIN[0])
    assert [kind for kind, _ in expected] == ["disulfide"] * 3
    _rename_cysteines(topology, TRYPSIN[0], "CYX")
    assert _links(topology, positions, TRYPSIN[0]) == expected


def test_the_d_cysteine_codes_are_those_of_d_aa_map():
    assert {name for name, parent in D_AA_MAP.items() if parent == "CYS"} == {"DCY"}
    assert cyclic._DISULFIDE_RESIDUE_NAMES == frozenset({"CYS", "CYX", "DCY"})
