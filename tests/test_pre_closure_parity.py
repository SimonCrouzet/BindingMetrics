"""The biotite closure detector agrees with ``core.cyclic.detect_cyclization`` on the examples.

``detect_closures`` exists so that the pre-flight check needs no OpenMM. This file is the proof
that it finds the same rings: on every bundled example, for every chain that has more than one
residue, the two detectors return the same links (same type, same two atoms).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("biotite")

from binding_metrics.capabilities import (  # noqa: E402
    _AMIDE_BOND_THRESHOLD_ANGSTROM,
    _DISULFIDE_RESIDUE_NAMES,
    _DISULFIDE_THRESHOLD_ANGSTROM,
    detect_closures,
    profile_input,
)
from binding_metrics.core import cyclic  # noqa: E402
from binding_metrics.io.structures import load_structure  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

# (file, [(biotite chain, OpenMM chain)]). OpenMM numbers the chains of an mmCIF by label_asym_id
# and biotite by the author chain ID, so the peptide of 1CWA is chain C to one and B to the other.
EXAMPLES = [
    ("example_linear_p53_1YCR.pdb", [("A", "A"), ("B", "B")]),
    ("example_ncaa_cyclosporin_1CWA.cif", [("A", "A"), ("C", "B")]),
    ("example_bicyclic_sfti1_3P8F.cif", [("A", "A"), ("I", "B")]),
    ("example_lactam_somatostatin_1XY4.cif", [("A", "A")]),
    ("example_staple_3V3B.pdb", [("C", "C")]),
    ("example_phospho_1QJB.pdb", [("Q", "Q")]),
]

# What each example is known to hold, from the structures themselves (PDB entries 1YCR, 1CWA, 3P8F,
# 1XY4, 3V3B): the peptide chain and its closure families.
EXPECTED_FAMILIES = {
    ("example_linear_p53_1YCR.pdb", "B"): {"none"},
    ("example_ncaa_cyclosporin_1CWA.cif", "C"): {"head_to_tail"},
    ("example_bicyclic_sfti1_3P8F.cif", "I"): {"head_to_tail", "disulfide"},
    ("example_lactam_somatostatin_1XY4.cif", "A"): {"disulfide", "lactam"},
    ("example_staple_3V3B.pdb", "C"): {"staple"},
    ("example_phospho_1QJB.pdb", "Q"): {"none"},
}


def _openmm_links(topology, positions, chain_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = cyclic.detect_cyclization(topology, positions, chain_id)
    return sorted(
        (info.cyclic_type, tuple(sorted([info.atom1_id[1:], info.atom2_id[1:]]))) for info in found
    )


def _biotite_links(atoms, chain_id):
    return sorted(
        (
            closure.kind,
            tuple(
                sorted(
                    [
                        (closure.end1.residue_index, closure.end1.atom_name),
                        (closure.end2.residue_index, closure.end2.atom_name),
                    ]
                )
            ),
        )
        for closure in detect_closures(atoms, chain_id)
    )


def _biotite_atoms(name):
    from binding_metrics.capabilities import _read_atoms

    return _read_atoms(DATA / name)


@pytest.mark.parametrize("name, chain_pairs", EXAMPLES, ids=[example[0] for example in EXAMPLES])
def test_same_closures_as_detect_cyclization(name, chain_pairs):
    topology, positions = load_structure(DATA / name)
    atoms = _biotite_atoms(name)
    for biotite_chain, openmm_chain in chain_pairs:
        chain = next(c for c in topology.chains() if c.id == openmm_chain)
        n_openmm = sum(1 for _ in chain.residues())
        expected = _openmm_links(topology, positions, openmm_chain)
        found = _biotite_links(atoms, biotite_chain)
        assert found == expected, f"{name} chain {biotite_chain}/{openmm_chain}"
        # both read the same polymer: the residue counts match (waters may share the chain ID)
        assert profile_input(DATA / name, biotite_chain).n_binder_residues == n_openmm


@pytest.mark.parametrize(
    "name, chain, families", [(n, c, f) for (n, c), f in EXPECTED_FAMILIES.items()]
)
def test_the_families_of_the_examples(name, chain, families):
    profile = profile_input(DATA / name, chain)
    assert set(profile.closures) == families


def test_the_examples_give_the_residue_classes_of_their_peptides():
    cyclosporin = profile_input(DATA / "example_ncaa_cyclosporin_1CWA.cif", "C", "A")
    assert cyclosporin.n_binder_residues == 11
    assert cyclosporin.residue_names["d_amino"] == ("DAL",)
    assert cyclosporin.residue_names["n_methyl"] == ("MLE", "MVA", "SAR")
    assert cyclosporin.residue_names["other_ncaa"] == ("ABA", "BMT")
    somatostatin = profile_input(DATA / "example_lactam_somatostatin_1XY4.cif", "A")
    assert somatostatin.residue_names["d_amino"] == ("DTR",)
    assert somatostatin.residue_names["other_ncaa"] == ("IAM",)
    phospho = profile_input(DATA / "example_phospho_1QJB.pdb", "Q")
    assert phospho.residue_names["phospho"] == ("SEP",)
    linear = profile_input(DATA / "example_linear_p53_1YCR.pdb", "B", "A")
    assert (linear.n_binder_residues, linear.binder_type) == (13, "peptide")
    assert linear.residue_classes == frozenset({"canonical"})


def test_the_receptor_of_3p8f_has_its_three_disulfides():
    profile = profile_input(DATA / "example_bicyclic_sfti1_3P8F.cif", "A")
    assert [c.family for c in profile.closure_bonds] == ["disulfide"] * 3
    assert profile.n_binder_residues == 241
    assert profile.binder_type == "unknown"


def test_the_distance_cut_offs_equal_those_of_core_cyclic():
    assert _AMIDE_BOND_THRESHOLD_ANGSTROM == pytest.approx(cyclic._AMIDE_BOND_THRESH * 10)
    assert _DISULFIDE_THRESHOLD_ANGSTROM == pytest.approx(cyclic._DISULFIDE_THRESH * 10)


# ---------------------------------------------------------------------------
# The cysteine names of a disulfide: CYS, CYX (AMBER) and DCY (D-cysteine)
# ---------------------------------------------------------------------------

SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"


def test_the_light_detector_uses_the_cysteine_names_of_core_cyclic():
    assert _DISULFIDE_RESIDUE_NAMES == cyclic._DISULFIDE_RESIDUE_NAMES == {"CYS", "CYX", "DCY"}


@pytest.mark.parametrize("name", ["CYS", "CYX", "DCY"])
def test_the_two_detectors_agree_under_every_cysteine_name(name):
    topology, positions = load_structure(SFTI1)
    atoms = _biotite_atoms(SFTI1.name)
    chain = next(c for c in topology.chains() if c.id == "B")
    for residue in chain.residues():
        if residue.name == "CYS":
            residue.name = name
    atoms.res_name[(atoms.chain_id == "I") & (atoms.res_name == "CYS")] = name
    found = _biotite_links(atoms, "I")
    assert found == _openmm_links(topology, positions, "B")
    assert [kind for kind, _ in found] == ["disulfide", "head_to_tail"]
