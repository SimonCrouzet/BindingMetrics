"""save_cif reads hydrogens from the element, not from the first letter of the atom name (#142).

An ion named HG (mercury), HO (holmium) or HF (hafnium) has one atom, and that atom was taken for
a hydrogen when the numbering of the source was read, so the residue did not count and every
later residue of the author chain took the number of the next one.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
gemmi = pytest.importorskip("gemmi")

from binding_metrics.io.structures import load_structure, save_cif  # noqa: E402

_HEADER = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""
_GLY = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))


def _write_source(path: Path, ion: tuple[str, str]) -> Path:
    """Chain A: GLY 1 and 2, then (each in its own label chain) the ion, a sodium and two waters.

    ``ion`` is the residue and atom name of the first ion and its element symbol.
    """
    name, element = ion
    blocks = [
        ("A", "GLY", [1, 2], _GLY),
        ("B", name, [500], ((name, element),)),
        ("C", "NA", [501], (("NA", "NA"),)),
        ("D", "HOH", [601, 602], (("O", "O"),)),
    ]
    rows = []
    for label, comp, numbers, atoms in blocks:
        for number in numbers:
            for atom, symbol in atoms:
                serial = len(rows) + 1
                group = "ATOM" if comp == "GLY" else "HETATM"
                rows.append(
                    f"{group} {serial} {symbol} {atom} . {comp} {label} 1 {number} ? "
                    f"{1.6 * serial} 0.0 0.0 1.00 10.0 {number} {comp} A {atom} 1"
                )
    path.write_text(_HEADER + "\n".join(rows) + "\n#\n", encoding="utf-8")
    return path


def _numbers(path: Path) -> list[str]:
    block = gemmi.cif.read(str(path)).sole_block()
    table = block.find("_atom_site.", ["auth_seq_id", "auth_comp_id"])
    residues, previous = [], None
    for row in table:
        key = (row[0], row[1])
        if key != previous:
            residues.append(row[0])
            previous = key
    return residues


@pytest.mark.parametrize("ion", [("HG", "HG"), ("HO", "HO"), ("HF", "HF")])
def test_an_ion_whose_name_starts_with_h_counts_as_a_residue(tmp_path, ion):
    """Regression: the residues after the mercury ion were written 1, 2, 501, 601, 602, 603."""
    source = _write_source(tmp_path / "source.cif", ion)
    topology, positions = load_structure(source)
    out = tmp_path / "out.cif"
    save_cif(topology, positions, out, source_cif_path=source)
    assert _numbers(source) == ["1", "2", "500", "501", "601", "602"]
    assert _numbers(out) == _numbers(source)
