"""save_cif restores residue numbers and insertion codes of the source mmCIF (#122).

The source numbering was read by counting changes of ``auth_seq_id``, so the residues 60, 60A,
60B, ... of 3P8F (trypsin) were one residue, and every later residue of the chain took the
number of a residue that lies further along.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
gemmi = pytest.importorskip("gemmi")

from binding_metrics.io.structures import load_structure, save_cif  # noqa: E402
from binding_metrics.metrics.comparison import compute_structure_rmsd  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"

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
_GLYCINE = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))


def _write_source(path: Path, residues: list[tuple[int, str]]) -> Path:
    """Chain A of glycines, one per ``(author number, insertion code)``, as a minimal mmCIF."""
    rows = []
    for label_seq, (number, ins_code) in enumerate(residues, start=1):
        for name, element in _GLYCINE:
            serial = len(rows) + 1
            x = 1.5 * serial
            rows.append(
                f"ATOM {serial} {element} {name} . GLY A 1 {label_seq} {ins_code or '?'} "
                f"{x} 0.0 0.0 1.00 10.0 {number} GLY A {name} 1"
            )
    path.write_text(_HEADER + "\n".join(rows) + "\n#\n", encoding="utf-8")
    return path


def _residues(path: Path, polymer_only: bool = True) -> list[tuple]:
    """``(auth chain, number, insertion code, residue name)`` of each residue, in file order."""
    block = gemmi.cif.read(str(path)).sole_block()
    table = block.find(
        "_atom_site.",
        ["group_PDB", "auth_asym_id", "auth_seq_id", "?pdbx_PDB_ins_code", "auth_comp_id"],
    )
    residues, previous = [], None
    for row in table:
        if polymer_only and row[0] != "ATOM":
            continue
        ins_code = row[3] if table.has_column(3) and row[3] not in ("?", ".") else ""
        key = (row[1], row[2], ins_code, row[4])
        if key != previous:
            residues.append(key)
            previous = key
    return residues


def _round_trip(source: Path, out: Path) -> Path:
    topology, positions = load_structure(source)
    save_cif(topology, positions, out, source_cif_path=source)
    return out


def test_minimal_file_with_insertion_codes_keeps_every_number_and_code(tmp_path):
    """58, 59, 60, 60A, 60B, 61, 62 come back as they went in.

    Regression: 60, 60A and 60B counted as one source residue, so the output was numbered
    58, 59, 60, 61, 62, 63, 64 and then ran out of source numbers.
    """
    numbering = [(58, ""), (59, ""), (60, ""), (60, "A"), (60, "B"), (61, ""), (62, "")]
    source = _write_source(tmp_path / "source.cif", numbering)
    out = _round_trip(source, tmp_path / "out.cif")
    assert [(r[1], r[2]) for r in _residues(out)] == [(str(n), c) for n, c in numbering]


def test_insertion_codes_are_read_back_by_openmm(tmp_path):
    source = _write_source(tmp_path / "source.cif", [(1, ""), (1, "A"), (2, "")])
    topology, _ = load_structure(_round_trip(source, tmp_path / "out.cif"))
    assert [(r.id, r.insertionCode) for r in topology.residues()] == [
        ("1", ""),
        ("1", "A"),
        ("2", ""),
    ]


@pytest.fixture(scope="module")
def saved_trypsin(tmp_path_factory):
    if not SFTI.exists():
        pytest.skip("3P8F is not bundled")
    return _round_trip(SFTI, tmp_path_factory.mktemp("save_cif") / "3p8f_saved.cif")


@pytest.mark.skipif(not SFTI.exists(), reason="3P8F is not bundled")
class TestTrypsinWithInsertionCodes:
    def test_polymer_residues_keep_number_and_code(self, saved_trypsin):
        """Residues 60, 60A, 60B, ... and every residue after them keep their number."""
        before, after = _residues(SFTI), _residues(saved_trypsin)
        assert len(before) == 255
        assert [r[:3] for r in after] == [r[:3] for r in before]
        assert any(r[2] for r in after)  # the insertion codes are written back

    def test_structure_rmsd_against_the_saved_copy_is_zero(self, saved_trypsin):
        """Before the fix: 16.8 A for the full complex, 17.3 A for the backbone."""
        result = compute_structure_rmsd(SFTI, saved_trypsin, design_chain="I")
        for name in ("rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"):
            assert result[name] is not None
            assert result[name] < 1e-2, name  # the saved coordinates have 4 decimals
        assert "reason" not in result

    def test_cross_links_follow_their_residues(self, saved_trypsin):
        """The disulfides of trypsin and of SFTI-1 are bonded on reload, at the source numbers."""

        def long_range_bonds(path):
            topology, _ = load_structure(path)
            return {
                frozenset(((b.atom1.residue.id, b.atom1.name), (b.atom2.residue.id, b.atom2.name)))
                for b in topology.bonds()
                if abs(b.atom1.residue.index - b.atom2.residue.index) > 1
                and "GSH" not in (b.atom1.residue.name, b.atom2.residue.name)
            }

        # 3 disulfides of trypsin, 1 of SFTI-1 and its head-to-tail closure
        assert len(long_range_bonds(SFTI)) == 5
        assert long_range_bonds(saved_trypsin) == long_range_bonds(SFTI)


@pytest.mark.parametrize("source", [SOMATOSTATIN, CYCLOSPORIN], ids=["1XY4", "1CWA"])
def test_files_without_insertion_codes_keep_their_numbers(source, tmp_path):
    if not source.exists():
        pytest.skip(f"{source.name} is not bundled")
    out = _round_trip(source, tmp_path / "out.cif")
    before, after = _residues(source), _residues(out)
    assert [r[:3] for r in after] == [r[:3] for r in before]
    assert not any(r[2] for r in after)
