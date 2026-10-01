"""Synthetic gemmi structures shared by the OpenFold3 runner tests (no test in this file).

Residues get backbone atoms N, CA, C, O (or the atoms given), so that a name gemmi and the
CCD know as an amino acid behaves like one, and a name that they know only as a ligand is
told apart by its backbone. Nothing here needs OpenFold3.
"""

from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

_BACKBONE = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))


def _residue(name: str, number: int, atoms=_BACKBONE, icode: str = " "):
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(number, icode)
    for k, (atom_name, element) in enumerate(atoms):
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(3.8 * number + k, 0.5 * k, 0.0)
        res.add_atom(atom)
    return res


def _structure(chains: dict) -> "gemmi.Structure":
    """Structure from ``{chain_id: [name or (name, atoms), ...]}``; residues are numbered 1..n."""
    st = gemmi.Structure()
    model = gemmi.Model("1")
    for chain_id, names in chains.items():
        chain = gemmi.Chain(chain_id)
        for number, entry in enumerate(names, start=1):
            name, atoms = (entry, _BACKBONE) if isinstance(entry, str) else entry
            chain.add_residue(_residue(name, number, atoms))
        model.add_chain(chain)
    st.add_model(model)
    return st


def _write(tmp_path: Path, chains: dict, name: str = "complex.pdb") -> Path:
    path = tmp_path / name
    _structure(chains).write_pdb(str(path))
    return path
