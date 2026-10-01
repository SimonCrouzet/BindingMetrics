"""Synthetic peptides for the pre-flight tests, and a check of the builder itself.

``build_chain`` places every residue 30 A from the last one and every atom of a residue on a short
line, so no two atoms of different residues are ever in contact. A test then puts exactly the
pairs it wants within reach of each other (``close=``) or declares a bond (``bonds=``), which
leaves the detector nothing else to find.
"""

from __future__ import annotations

import numpy as np
import pytest

struc = pytest.importorskip("biotite.structure")

# Side-chain atoms of the residues the tests use; any other name gets a CB only.
_SIDE_CHAIN_ATOMS = {
    "GLY": (),
    "CYS": ("CB", "SG"),
    "CYX": ("CB", "SG"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
}
_BACKBONE = ("N", "CA", "C", "O")
_RESIDUE_SPACING_ANGSTROM = 30.0
_ATOM_SPACING_ANGSTROM = 1.6


def _element(atom_name: str) -> str:
    return atom_name[0]


def build_chain(
    names,
    chain_id: str = "B",
    *,
    first_number: int = 1,
    side_chain_atoms: dict | None = None,
    residue_atoms: dict | None = None,
    close=(),
    bonds=(),
    bond_table: bool = True,
    x_offset: float = 0.0,
):
    """An AtomArray of one chain.

    Args:
        names: Residue names in chain order.
        chain_id: Chain ID of every atom.
        first_number: Residue number of the first residue (then consecutive).
        side_chain_atoms: Side-chain atom names per residue name, overriding the built-in table.
        residue_atoms: The complete atom list per residue name (no backbone added), for a cap
            or a ligand.
        close: ``(residue_index, atom_name, other_residue_index, other_atom_name, distance)``
            rows; the first atom is moved to ``distance`` angstrom from the second.
        bonds: ``(residue_index, atom_name, other_residue_index, other_atom_name)`` rows added to
            the bond table.
        bond_table: False leaves ``atoms.bonds`` as None.
        x_offset: Shift of the whole chain along x (to keep two chains apart).
    """
    table = {**_SIDE_CHAIN_ATOMS, **(side_chain_atoms or {})}
    rows = []
    for k, name in enumerate(names):
        if residue_atoms and name in residue_atoms:
            atom_names = tuple(residue_atoms[name])
        else:
            atom_names = _BACKBONE + tuple(table.get(name, ("CB",)))
        for j, atom_name in enumerate(atom_names):
            rows.append((k, name, atom_name, j))
    atoms = struc.AtomArray(len(rows))
    atoms.chain_id[:] = chain_id
    atoms.hetero[:] = False
    for i, (k, name, atom_name, j) in enumerate(rows):
        atoms.res_id[i] = first_number + k
        atoms.res_name[i] = name
        atoms.atom_name[i] = atom_name
        atoms.element[i] = _element(atom_name)
        atoms.coord[i] = (
            x_offset + _RESIDUE_SPACING_ANGSTROM * k + _ATOM_SPACING_ANGSTROM * j,
            0.0,
            0.0,
        )
    lookup = {(k, atom_name): i for i, (k, _, atom_name, _) in enumerate(rows)}
    for res_a, atom_a, res_b, atom_b, distance in close:
        atoms.coord[lookup[(res_a, atom_a)]] = atoms.coord[lookup[(res_b, atom_b)]] + (
            distance,
            0.0,
            0.0,
        )
    if bond_table:
        table_rows = [
            (lookup[(a, an)], lookup[(b, bn)], struc.BondType.SINGLE) for a, an, b, bn in bonds
        ]
        atoms.bonds = struc.BondList(
            atoms.array_length(), np.array(table_rows, dtype=int).reshape(-1, 3)
        )
    return atoms


def add_waters_and_ion(atoms, n_waters: int = 3):
    """The same array with waters and one zinc ion appended under the chain ID of its last atom."""
    chain_id = str(atoms.chain_id[-1])
    extra = struc.AtomArray(n_waters + 1)
    extra.chain_id[:] = chain_id
    extra.hetero[:] = True
    top = float(atoms.coord[:, 0].max()) + 50.0
    last_number = int(atoms.res_id.max())
    for i in range(n_waters):
        extra.res_id[i] = last_number + 1 + i
        extra.res_name[i] = "HOH"
        extra.atom_name[i] = "O"
        extra.element[i] = "O"
        extra.coord[i] = (top + 5.0 * i, 10.0, 0.0)
    extra.res_id[n_waters] = last_number + 1 + n_waters
    extra.res_name[n_waters] = "ZN"
    extra.atom_name[n_waters] = "ZN"
    extra.element[n_waters] = "ZN"
    extra.coord[n_waters] = (top, 20.0, 0.0)
    if atoms.bonds is not None:
        extra.bonds = struc.BondList(extra.array_length())
    return atoms + extra


def head_to_tail_ring(names, chain_id: str = "B", **kwargs):
    """A chain whose last C is 1.33 A from its first N."""
    last = len(names) - 1
    close = [(last, "C", 0, "N", 1.33), *kwargs.pop("close", ())]
    return build_chain(names, chain_id, close=close, **kwargs)


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def _distance(atoms, res_a, atom_a, res_b, atom_b, first_number=1):
    def find(res, atom):
        mask = (atoms.res_id == first_number + res) & (atoms.atom_name == atom)
        return atoms.coord[mask][0]

    return float(np.linalg.norm(find(res_a, atom_a) - find(res_b, atom_b)))


def test_builder_keeps_unrelated_atoms_apart():
    atoms = build_chain(["ALA", "CYS", "GLY", "LYS"])
    assert _distance(atoms, 0, "N", 3, "C") == pytest.approx(90 + 1.6 * 2)
    residues = struc.get_residue_starts(atoms)
    assert len(residues) == 4
    assert atoms.bonds is not None and atoms.bonds.as_array().shape == (0, 3)


def test_builder_moves_the_requested_atom_and_declares_the_requested_bond():
    atoms = head_to_tail_ring(["ALA", "GLY", "CYS"], bonds=[(0, "CA", 0, "N")])
    assert _distance(atoms, 2, "C", 0, "N") == pytest.approx(1.33)
    assert atoms.bonds.as_array().shape == (1, 3)


def test_builder_without_a_bond_table():
    assert build_chain(["ALA", "GLY"], bond_table=False).bonds is None


# ---------------------------------------------------------------------------
# Sources of the models, for the tests that re-read what a declared limit rests on
# ---------------------------------------------------------------------------

#: Directory that holds a clone of each model repository (``alphafold``, ``ColabFold``, ``boltz``,
#: ``Protenix``, ...). The evidence tests of ``test_pre_*_limits.py`` read a documentation or
#: source line from it and skip when it is not set, so CI needs no clone.
MODEL_SOURCES_ENVIRONMENT_VARIABLE = "BINDING_METRICS_MODEL_SOURCES"


def model_source_text(repository: str, relative_path: str) -> str:
    """The text of a file of a model clone, or skip the test when the clone is not there."""
    import os
    from pathlib import Path

    root = os.environ.get(MODEL_SOURCES_ENVIRONMENT_VARIABLE)
    if not root:
        pytest.skip(f"{MODEL_SOURCES_ENVIRONMENT_VARIABLE} is not set")
    path = Path(root) / repository / relative_path
    if not path.is_file():
        pytest.skip(f"{path} is not there")
    return path.read_text(encoding="utf-8")
