"""save_cif and load_structure keep a link at residues that share a number (#132).

Residues 60, 60A and 60B have one number. OpenMM's ``PDBxFile`` keys the atoms of a CIF by number,
chain and atom name, and reads the two partners of a ``_struct_conn`` row by the same three
values, so a row at such a number is resolved to the last residue that has it. The label columns
that ``save_cif`` wrote could not tell the residues apart either, and a link between SG of 2 and
SG of 3 (when 3A has an SG too) came back as a link to 3A.

The structure of these tests is built here: chain A of glycines and of the residues that close a
disulfide (cysteine) or a lactam (lysine, aspartate), where 3 and 3A are both cysteines and 7 and
7A are both aspartates. The coordinates are placeholders; the tests read the bond table.
"""

import logging
from pathlib import Path

import pytest

pytest.importorskip("openmm")
gemmi = pytest.importorskip("gemmi")

from openmm.app import PDBxFile  # noqa: E402

from binding_metrics.io.structures import (  # noqa: E402
    _patch_nonstd_bonds_in_cif,
    _restore_struct_conn_bonds,
    load_structure,
    save_cif,
)

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
_BACKBONE = ("N", "CA", "C", "O")
_SIDE_CHAIN = {
    "GLY": (),
    "CYS": ("CB", "SG"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
}
#: (name, number, insertion code) along chain A. 3 and 3A are cysteines, 7 and 7A aspartates.
_RESIDUES = [
    ("GLY", 1, ""),
    ("CYS", 2, ""),
    ("CYS", 3, ""),
    ("CYS", 3, "A"),
    ("GLY", 4, ""),
    ("LYS", 5, ""),
    ("GLY", 6, ""),
    ("ASP", 7, ""),
    ("ASP", 7, "A"),
    ("GLY", 8, ""),
    ("CYS", 9, ""),
]


def _write_source(path: Path, residues=_RESIDUES) -> Path:
    """Chain A as a minimal mmCIF; the residues follow one another on a staircase."""
    rows = []
    for position, (name, number, ins_code) in enumerate(residues, start=1):
        for atom in (*_BACKBONE, *_SIDE_CHAIN[name]):
            serial = len(rows) + 1
            rows.append(
                f"ATOM {serial} {atom[0]} {atom} . {name} A 1 {position} {ins_code or '?'} "
                f"{1.5 * serial:.3f} {3.0 * position:.3f} 0.0 1.00 10.0 {number} {name} A {atom} 1"
            )
    path.write_text(_HEADER + "\n".join(rows) + "\n#\n", encoding="utf-8")
    return path


def _atom(topology, number: int, code: str, name: str):
    for residue in topology.residues():
        if residue.id == str(number) and residue.insertionCode == code:
            return next(atom for atom in residue.atoms() if atom.name == name)
    raise KeyError((number, code, name))


def _load_with_links(source: Path, links, reverse: bool = False):
    """The topology of ``source`` with one extra bond per ``((number, code, atom), ...)`` pair."""
    topology, positions = load_structure(source)
    for first, second in links:
        a, b = _atom(topology, *first), _atom(topology, *second)
        topology.addBond(*((b, a) if reverse else (a, b)))
    return topology, positions


def _label(residue) -> str:
    return f"{residue.id}{residue.insertionCode}"


def _cross_links(topology) -> set:
    """The disulfides (SG-SG) and lactams (NZ-CG) as sets of two ``(residue label, atom)``."""
    return {
        frozenset(
            ((_label(b.atom1.residue), b.atom1.name), (_label(b.atom2.residue), b.atom2.name))
        )
        for b in topology.bonds()
        if {b.atom1.name, b.atom2.name} in ({"SG"}, {"NZ", "CG"})
    }


def _expected(links) -> set:
    return {
        frozenset(((f"{n1}{c1}", a1), (f"{n2}{c2}", a2))) for (n1, c1, a1), (n2, c2, a2) in links
    }


def _bond_pairs(topology) -> set:
    return {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()}


#: Links whose partner is the first or the second, in file order, of the residues that share a
#: number.
_LINKS = {
    "disulfide at 3, before 3A": [((2, "", "SG"), (3, "", "SG"))],
    "disulfide at 3A": [((2, "", "SG"), (3, "A", "SG"))],
    "two disulfides, at 3 and at 3A": [
        ((2, "", "SG"), (3, "", "SG")),
        ((3, "A", "SG"), (9, "", "SG")),
    ],
    "lactam at 7, before 7A": [((5, "", "NZ"), (7, "", "CG"))],
    "lactam at 7A": [((5, "", "NZ"), (7, "A", "CG"))],
}


@pytest.mark.parametrize("with_source", [True, False], ids=["source", "no_source"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
@pytest.mark.parametrize("links", list(_LINKS.values()), ids=list(_LINKS))
def test_link_reloads_between_the_atoms_it_joined(tmp_path, links, reverse, with_source):
    """Neither dropped nor bonded to the neighbour that shares the number.

    Regression: the disulfide between 2 and 3 came back as a disulfide between 2 and 3A, and the
    lactam between 5 and 7 as a lactam between 5 and 7A, with and without a source CIF; the
    links at 3A and 7A were the ones that survived.
    """
    source = _write_source(tmp_path / "source.cif")
    topology, positions = _load_with_links(source, links, reverse)
    out = tmp_path / "out.cif"
    save_cif(topology, positions, out, source_cif_path=source if with_source else None)
    reloaded, _ = load_structure(out)
    assert _cross_links(reloaded) == _expected(links)
    assert _bond_pairs(reloaded) == _bond_pairs(topology)


def _struct_conn_rows(path: Path) -> list[dict]:
    block = gemmi.cif.read(str(path)).sole_block()
    loop = block.find_loop("_struct_conn.id").get_loop()
    tags = [tag.split(".", 1)[1] for tag in loop.tags]
    values = list(loop.values)
    return [dict(zip(tags, values[i : i + len(tags)])) for i in range(0, len(values), len(tags))]


def _partners(row: dict) -> tuple:
    """``(label_seq_id, auth_seq_id, insertion code)`` of the two partners of a row."""
    return tuple(
        (
            row[f"ptnr{n}_label_seq_id"],
            row[f"ptnr{n}_auth_seq_id"],
            row[f"pdbx_ptnr{n}_PDB_ins_code"],
        )
        for n in "12"
    )


@pytest.fixture
def two_disulfides(tmp_path):
    source = _write_source(tmp_path / "source.cif")
    topology, positions = _load_with_links(source, _LINKS["two disulfides, at 3 and at 3A"])
    out = tmp_path / "out.cif"
    save_cif(topology, positions, out, source_cif_path=source)
    return out


class TestRowsOfTheSavedFile:
    def test_partners_carry_author_number_and_insertion_code(self, two_disulfides):
        rows = _struct_conn_rows(two_disulfides)
        assert [(r["ptnr1_auth_asym_id"], r["ptnr2_auth_asym_id"]) for r in rows] == [
            ("A", "A")
        ] * 2
        assert [(r["ptnr1_auth_comp_id"], r["ptnr2_auth_comp_id"]) for r in rows] == [
            ("CYS", "CYS")
        ] * 2
        # 2 - 3, then 3A - 9; "." is the label number of a partner at the shared number 3
        assert [_partners(r) for r in rows] == [
            (("2", "2", "?"), (".", "3", "?")),
            ((".", "3", "A"), ("9", "9", "?")),
        ]

    def test_a_residue_at_a_shared_number_has_no_label_number(self, two_disulfides):
        """label_seq_id is "." where the number is shared, so that PDBxFile finds no atom for it."""
        block = gemmi.cif.read(str(two_disulfides)).sole_block()
        table = block.find(
            "_atom_site.", ["auth_seq_id", "pdbx_PDB_ins_code", "label_seq_id", "label_atom_id"]
        )
        label_numbers = {(row[0], row[1]): row[2] for row in table if row[3] == "CA"}
        assert label_numbers == {
            ("1", "."): "1",
            ("2", "."): "2",
            ("3", "."): ".",
            ("3", "A"): ".",
            ("4", "."): "4",
            ("5", "."): "5",
            ("6", "."): "6",
            ("7", "."): ".",
            ("7", "A"): ".",
            ("8", "."): "8",
            ("9", "."): "9",
        }

    def test_biotite_finds_both_disulfides(self, two_disulfides):
        """biotite matches the rows on every label and author column: it raised on the label-only
        rows ("cannot be unambiguously assigned") and now finds one atom for each partner."""
        pdbx = pytest.importorskip("biotite.structure.io.pdbx")
        atoms = pdbx.get_structure(
            pdbx.CIFFile.read(str(two_disulfides)), model=1, include_bonds=True
        )
        links = {
            frozenset(
                (
                    (int(atoms.res_id[i]), str(atoms.ins_code[i])),
                    (int(atoms.res_id[j]), str(atoms.ins_code[j])),
                )
            )
            for i, j, _ in atoms.bonds.as_array()
            if atoms.atom_name[i] == "SG" and atoms.atom_name[j] == "SG"
        }
        assert links == {
            frozenset(((2, ""), (3, ""))),
            frozenset(((3, "A"), (9, ""))),
        }

    def test_gemmi_reads_the_partners_by_insertion_code(self, two_disulfides):
        structure = gemmi.read_structure(str(two_disulfides))
        ends = [(str(c.partner1.res_id), str(c.partner2.res_id)) for c in structure.connections]
        assert ends == [("2(CYS)", "3(CYS)"), ("3A(CYS)", "9(CYS)")]


class TestFilesWithoutSharedNumbers:
    """No residue shares a number: the rows and the label numbers are as they were."""

    def test_synthetic_structure_keeps_ten_label_columns(self, tmp_path):
        residues = [(name, number, "") for number, (name, _, _) in enumerate(_RESIDUES, start=1)]
        source = _write_source(tmp_path / "source.cif", residues)
        topology, positions = _load_with_links(source, [((2, "", "SG"), (3, "", "SG"))])
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source)
        rows = _struct_conn_rows(out)
        assert len(rows) == 1
        assert not any("auth" in tag or "ins_code" in tag for tag in rows[0])
        assert (rows[0]["ptnr1_label_seq_id"], rows[0]["ptnr2_label_seq_id"]) == ("2", "3")
        reloaded, _ = load_structure(out)
        assert _cross_links(reloaded) == {frozenset((("2", "SG"), ("3", "SG")))}

    @pytest.mark.parametrize("source", [SOMATOSTATIN, CYCLOSPORIN], ids=["1XY4", "1CWA"])
    def test_examples_get_no_author_partner_columns(self, source, tmp_path):
        if not source.exists():
            pytest.skip(f"{source.name} is not bundled")
        topology, positions = load_structure(source)
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source)
        tags = _struct_conn_rows(out)[0]
        assert not any("auth" in tag or "ins_code" in tag for tag in tags)
        assert _bond_pairs(load_structure(out)[0]) == _bond_pairs(topology)


@pytest.fixture(scope="module")
def saved_trypsin(tmp_path_factory):
    if not SFTI.exists():
        pytest.skip("3P8F is not bundled")
    topology, positions = load_structure(SFTI)
    out = tmp_path_factory.mktemp("struct_conn") / "3p8f_saved.cif"
    save_cif(topology, positions, out, source_cif_path=SFTI)
    return topology, out


class TestTrypsinWithInsertionCodes:
    """Residues 221 and 221A of 3P8F share a number, so its rows name their partners by author
    numbering."""

    def test_every_partner_names_an_atom_of_the_file(self, saved_trypsin):
        _, out = saved_trypsin
        block = gemmi.cif.read(str(out)).sole_block()
        site = block.find(
            "_atom_site.", ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "label_atom_id"]
        )
        atoms = {tuple(row[i] for i in range(4)) for row in site}
        rows = _struct_conn_rows(out)
        assert len(rows) == 25  # 4 disulfides, the link to GSH, its 19 bonds and the ring closure
        for row in rows:
            for n in "12":
                key = (
                    row[f"ptnr{n}_auth_asym_id"],
                    row[f"ptnr{n}_auth_seq_id"],
                    row[f"pdbx_ptnr{n}_PDB_ins_code"].replace("?", "."),
                    row[f"ptnr{n}_label_atom_id"],
                )
                assert key in atoms

    def test_reload_gives_exactly_the_bonds_of_the_source(self, saved_trypsin):
        topology, out = saved_trypsin
        assert _bond_pairs(load_structure(out)[0]) == _bond_pairs(topology)


class TestRowsOfTheLactam:
    """The rows that ``_patch_nonstd_bonds_in_cif`` appends, with and without a source CIF."""

    @pytest.mark.parametrize("with_source", [True, False], ids=["source", "no_source"])
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("lactam at 7, before 7A", (("5", "5", "?"), (".", "7", "?"))),
            ("lactam at 7A", (("5", "5", "?"), (".", "7", "A"))),
        ],
    )
    def test_partner_at_a_shared_number_is_named_by_its_insertion_code(
        self, tmp_path, name, expected, with_source
    ):
        source = _write_source(tmp_path / "source.cif")
        topology, positions = _load_with_links(source, _LINKS[name])
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source if with_source else None)
        (row,) = _struct_conn_rows(out)
        assert row["conn_type_id"] == "covale"
        assert (row["ptnr1_label_atom_id"], row["ptnr2_label_atom_id"]) == ("NZ", "CG")
        assert _partners(row) == expected


class TestReaderAndWriterFunctions:
    """The functions that read and write the rows, on files made for the purpose."""

    _TAGS = (
        "id",
        "conn_type_id",
        "ptnr1_label_asym_id",
        "ptnr1_label_seq_id",
        "ptnr1_label_atom_id",
        "ptnr2_label_asym_id",
        "ptnr2_label_seq_id",
        "ptnr2_label_atom_id",
    )
    _AUTHOR_TAGS = (
        "ptnr1_auth_asym_id",
        "ptnr1_auth_seq_id",
        "pdbx_ptnr1_PDB_ins_code",
        "ptnr2_auth_asym_id",
        "ptnr2_auth_seq_id",
        "pdbx_ptnr2_PDB_ins_code",
    )

    def _file_with_rows(self, tmp_path: Path, rows: list[str], author: bool) -> Path:
        """``_RESIDUES`` with a ``_struct_conn`` loop of the given rows."""
        path = _write_source(tmp_path / "rows.cif")
        tags = (*self._TAGS, *self._AUTHOR_TAGS) if author else self._TAGS
        loop = "loop_\n" + "".join(f"_struct_conn.{tag}\n" for tag in tags) + "\n".join(rows)
        text = path.read_text(encoding="utf-8")
        marker = "loop_\n_atom_site.group_PDB"
        path.write_text(text.replace(marker, loop + "\n#\n" + marker, 1), encoding="utf-8")
        return path

    def test_restore_bonds_the_residue_with_the_insertion_code(self, tmp_path):
        """Rows with author numbering and ``.`` as the label number of a shared number: PDBxFile
        finds no atom for them, and the reader bonds 2 to 3 and 3A to 9."""
        path = self._file_with_rows(
            tmp_path,
            [
                "d1 disulf A 2 SG A . SG A 2 ? A 3 ?",
                "d2 disulf A . SG A 9 SG A 3 A A 9 ?",
            ],
            author=True,
        )
        topology = PDBxFile(str(path)).topology
        assert _cross_links(topology) == set()
        assert _restore_struct_conn_bonds(path, topology) == 2
        assert _cross_links(topology) == {
            frozenset((("2", "SG"), ("3", "SG"))),
            frozenset((("3A", "SG"), ("9", "SG"))),
        }

    def test_pdbx_resolves_a_number_that_two_residues_share_to_the_last_one(self, tmp_path):
        """The premise of the change: label columns at a shared number name the wrong residue."""
        path = self._file_with_rows(tmp_path, ["d1 disulf A 2 SG A 3 SG"], author=False)
        topology = PDBxFile(str(path)).topology
        assert _cross_links(topology) == {frozenset((("2", "SG"), ("3A", "SG")))}

    def test_rows_that_do_not_follow_the_topology_are_left_alone(self, tmp_path, caplog):
        """The bonds listed for a row are told from the order of the topology; when that order
        does not match the file, the rows keep the label columns and a warning says why."""
        source = _write_source(tmp_path / "source.cif")
        topology, positions = _load_with_links(source, _LINKS["disulfide at 3, before 3A"])
        out = tmp_path / "out.cif"
        with open(out, "w", encoding="utf-8") as handle:
            PDBxFile.writeFile(topology, positions, handle, keepIds=True)
        topology.addBond(_atom(topology, 3, "A", "SG"), _atom(topology, 9, "", "SG"))  # not listed
        with caplog.at_level(logging.WARNING):
            _patch_nonstd_bonds_in_cif(out, topology)
        assert "cannot be told apart" in caplog.text
        assert not any("auth" in tag for tag in _struct_conn_rows(out)[0])
