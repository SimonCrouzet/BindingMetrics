"""The closure bonds of the placed file sit on the atoms that the detector found (#218).

``score_pose_set`` writes one structure file per pose for the path-based metrics, with the
closure bonds of the binder in its bond table. The bundled SFTI-1 complex (3P8F) closes a
head-to-tail ring and a disulfide between Cys 3 and Cys 11 of chain I. The structures below give
the two cysteines one residue number, with distinct insertion codes (3A and 3B) or with none, and
the bond must still join their two sulfur atoms. An end of a closure that is absent from the
chain, or present more than once, raises ``ValueError`` and names the end.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR

from binding_metrics.capabilities import Closure, ClosureEnd
from binding_metrics.poses import ExperimentalPose, Reference, score_pose_set
from binding_metrics.poses.score import _with_covalent_links
from binding_metrics.testing.synthetic import write_cif

SFTI = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
BINDER, TARGET = "I", "A"
DISULFIDE = frozenset({("CYS", "A", "SG"), ("CYS", "B", "SG")})
HEAD_TO_TAIL = frozenset({("ASP", ".", "C"), ("GLY", ".", "N")})


def _model(codes):
    """3P8F without waters, with Cys 3 and Cys 11 of the binder numbered 3 and given ``codes``."""
    from binding_metrics.metrics._common import load_structure

    atoms = load_structure(SFTI)
    atoms = atoms[atoms.res_name != "HOH"]
    binder = atoms.chain_id == BINDER
    first, second = codes
    atoms.ins_code[binder & (atoms.res_id == 3)] = first
    second_cys = binder & (atoms.res_id == 11)
    atoms.res_id[second_cys] = 3
    atoms.ins_code[second_cys] = second
    return atoms


def _with_second_sulfur(atoms):
    """A copy of ``atoms`` in which the first SG of the binder occurs twice in its residue."""
    sulfur = int(_sulfur_atoms(atoms)[0])
    duplicate = atoms[sulfur : sulfur + 1].copy()
    duplicate.coord = duplicate.coord + 0.5
    return atoms[: sulfur + 1] + duplicate + atoms[sulfur + 1 :]


def _sulfur_atoms(atoms):
    return np.flatnonzero((atoms.chain_id == BINDER) & (atoms.atom_name == "SG"))


def _bonded_pairs(atoms):
    return {frozenset((int(i), int(j))) for i, j, _ in atoms.bonds.as_array()}


def _binder_atom(atoms, residue_index, name):
    """The atom ``name`` of the ``residue_index``-th residue of the binder."""
    import biotite.structure as struc

    rows = np.flatnonzero(atoms.chain_id == BINDER)
    chain = atoms[rows]
    starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    start, stop = starts[residue_index], starts[residue_index + 1]
    return int(rows[start + np.flatnonzero(chain.atom_name[start:stop] == name)[0]])


def _score(atoms, tmp_path, metrics=("interface",)):
    path = tmp_path / "model.cif"
    write_cif(atoms, path)
    return score_pose_set(
        [ExperimentalPose.from_file(path, id="model")],
        BINDER,
        TARGET,
        metrics=list(metrics),
        reference=Reference(path, receptor_chain=TARGET, binder_chain=BINDER),
        keep_tmp=True,
        tmp_dir=tmp_path,
    )


def _written_links(result):
    """The links of ``_struct_conn`` in the file of the first pose: sets of (name, code, atom)."""
    import biotite.structure.io.pdbx as pdbx

    block = pdbx.CIFFile.read(Path(result["provenance"]["temp_dir"]) / "pose_0000.cif").block
    conn = block["struct_conn"]

    def end(number):
        names = [f"ptnr{number}_label_comp_id", f"pdbx_ptnr{number}_PDB_ins_code"]
        names.append(f"ptnr{number}_label_atom_id")
        return list(zip(*(conn[name].as_array(str) for name in names)))

    return {frozenset(pair) for pair in zip(end(1), end(2))}


@pytest.mark.parametrize("codes", [("A", "B"), ("", "")], ids=["icodes", "no_icodes"])
def test_the_disulfide_joins_the_two_cysteines_that_share_a_number(codes):
    atoms = _model(codes)
    sulfur = _sulfur_atoms(atoms)
    assert set(atoms.res_id[sulfur]) == {3}

    linked = _with_covalent_links(atoms, BINDER)

    assert frozenset(map(int, sulfur)) in _bonded_pairs(linked)


def test_the_head_to_tail_bond_is_kept_beside_the_disulfide():
    atoms = _model(("A", "B"))

    pairs = _bonded_pairs(_with_covalent_links(atoms, BINDER))

    assert frozenset((_binder_atom(atoms, 13, "C"), _binder_atom(atoms, 0, "N"))) in pairs


@pytest.mark.slow
def test_a_pose_set_writes_the_disulfide_of_residues_that_share_a_number(tmp_path):
    result = _score(_model(("A", "B")), tmp_path)

    row = result["poses"][0]
    assert row["status"] == "ok", row["reasons"]
    assert "interface" in row
    assert _written_links(result) == {DISULFIDE, HEAD_TO_TAIL}


def _closure(end1, end2):
    return Closure("disulfide", "disulfide", ClosureEnd(*end1), ClosureEnd(*end2))


@pytest.mark.parametrize(
    ("end", "message"),
    [
        ((40, "CYS", 3, "SG"), r"CYS 3\.SG \(position 40 of chain I\).*out of range"),
        ((2, "CYS", 11, "SG"), r"CYS 11\.SG \(position 2 of chain I\).*CYS 3A"),
        ((2, "SER", 3, "SG"), r"SER 3\.SG \(position 2 of chain I\).*CYS 3A"),
        ((2, "CYS", 3, "SX"), r"CYS 3\.SX \(position 2 of chain I\).*CYS 3A has no atom SX"),
    ],
    ids=["out_of_range", "other_number", "other_name", "no_atom"],
)
def test_an_end_that_the_chain_does_not_hold_is_named_in_the_error(monkeypatch, end, message):
    monkeypatch.setattr(
        "binding_metrics.capabilities.detect_closures",
        lambda atoms, chain_id: [_closure(end, (10, "CYS", 3, "SG"))],
    )

    with pytest.raises(ValueError, match=message):
        _with_covalent_links(_model(("A", "B")), BINDER)


_TWICE = r"CYS 3\.SG \(position 2 of chain I\).*CYS 3A has 2 atoms named SG"


def test_an_atom_that_occurs_twice_in_a_residue_is_an_error():
    with pytest.raises(ValueError, match=_TWICE):
        _with_covalent_links(_with_second_sulfur(_model(("A", "B"))), BINDER)


@pytest.mark.slow
def test_a_pose_set_reports_the_end_that_cannot_be_placed(tmp_path):
    result = _score(_with_second_sulfur(_model(("A", "B"))), tmp_path)

    row = result["poses"][0]
    assert row["status"] == "partial"
    assert "interface" not in row
    assert row["reasons"].keys() == {"interface"}
    assert "CYS 3A has 2 atoms named SG" in row["reasons"]["interface"]
