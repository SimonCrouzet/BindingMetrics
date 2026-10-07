"""save_cif numbers the ligands and waters of a returning author chain like its source (#131).

mmCIF lists the polymer chains first, then the ligands, then the waters, each grouped by label
chain, so author chain A comes back after chain B. The source reader restarted its residue
counter whenever the author chain changed, and the keys of the later block collided with the
keys of the polymer, so those residues found no source number and kept the one OpenMM gave.
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
_ATOMS = {
    "GLY": (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")),
    "LIG": (("C1", "C"), ("O1", "O")),
    "HOH": (("O", "O"),),
}


def _write_source(path: Path, blocks: list[tuple]) -> Path:
    """A minimal mmCIF from ``(label chain, author chain, residue name, [numbers])`` blocks.

    The blocks keep their order, so an author chain can come back after another one, as in a
    wwPDB file where the ligands and the waters follow the polymers.
    """
    rows = []
    for label, auth, name, numbers in blocks:
        for number in numbers:
            for atom, element in _ATOMS[name]:
                serial = len(rows) + 1
                group = "ATOM" if name == "GLY" else "HETATM"
                rows.append(
                    f"{group} {serial} {element} {atom} . {name} {label} 1 {number} ? "
                    f"{1.6 * serial} 0.0 0.0 1.00 10.0 {number} {name} {auth} {atom} 1"
                )
    path.write_text(_HEADER + "\n".join(rows) + "\n#\n", encoding="utf-8")
    return path


def _residues(path: Path) -> list[tuple]:
    """``(group, author chain, number, insertion code, residue name)`` of each residue, in order."""
    block = gemmi.cif.read(str(path)).sole_block()
    table = block.find(
        "_atom_site.",
        ["group_PDB", "auth_asym_id", "auth_seq_id", "?pdbx_PDB_ins_code", "auth_comp_id"],
    )
    residues, previous = [], None
    for row in table:
        ins_code = row[3] if table.has_column(3) and row[3] not in ("?", ".") else ""
        key = (row[0], row[1], row[2], ins_code, row[4])
        if key != previous:
            residues.append(key)
            previous = key
    return residues


def _round_trip(source: Path, out: Path) -> Path:
    topology, positions = load_structure(source)
    save_cif(topology, positions, out, source_cif_path=source)
    return out


class TestChainThatReturns:
    def test_ligands_and_waters_keep_their_numbers_when_chains_return_twice(self, tmp_path):
        """Chains A and B each come back twice, after the other one.

        Regression: the ligand of A was written 1, the waters of B 1 and 2, those of A 13, 2 and
        3, and the ion of B 22.
        """
        source = _write_source(
            tmp_path / "source.cif",
            [
                ("A", "A", "GLY", [10, 11, 12]),
                ("B", "B", "GLY", [20, 21]),
                ("C", "A", "LIG", [1001]),
                ("D", "B", "HOH", [301, 302]),
                ("E", "A", "HOH", [2001, 2002, 2003]),
                ("F", "B", "LIG", [401]),
            ],
        )
        out = _round_trip(source, tmp_path / "out.cif")
        assert _residues(out) == _residues(source)

    def test_reload_gives_the_numbers_of_the_source(self, tmp_path):
        source = _write_source(
            tmp_path / "source.cif",
            [
                ("A", "A", "GLY", [1, 2]),
                ("B", "B", "GLY", [1]),
                ("C", "A", "LIG", [500]),
                ("D", "A", "HOH", [601, 602]),
                ("E", "B", "HOH", [701]),
            ],
        )
        topology, _ = load_structure(_round_trip(source, tmp_path / "out.cif"))
        assert [r.id for r in topology.residues()] == ["1", "2", "1", "500", "601", "602", "701"]

    def test_a_returning_block_that_starts_at_the_last_number_is_a_new_residue(self, tmp_path):
        """The ligand has the number of the last glycine (3) in another label chain.

        PDBxFile starts a new residue at a change of label chain, so the ligand counts as one
        and the waters after it keep their numbers. The ligand's own number is taken by the
        glycine, so it moves to the next free one.
        """
        source = _write_source(
            tmp_path / "source.cif",
            [
                ("A", "A", "GLY", [1, 2, 3]),
                ("B", "B", "GLY", [1, 2]),
                ("C", "A", "LIG", [3]),
                ("D", "A", "HOH", [2001, 2002]),
            ],
        )
        out = _round_trip(source, tmp_path / "out.cif")
        chain_a = [(r[2], r[4]) for r in _residues(out) if r[1] == "A"]
        assert chain_a == [("1", "GLY"), ("2", "GLY"), ("3", "GLY"), ("4", "LIG")] + [
            ("2001", "HOH"),
            ("2002", "HOH"),
        ]

    def test_numbers_stay_unique_per_chain_when_a_returning_block_reuses_them(self, tmp_path):
        """Waters numbered from 1 collide with the glycines of their chain; uniqueness wins."""
        source = _write_source(
            tmp_path / "source.cif",
            [
                ("A", "A", "GLY", [1, 2, 3]),
                ("B", "B", "GLY", [1, 2]),
                ("C", "A", "HOH", [1, 2]),
                ("D", "B", "HOH", [1]),
            ],
        )
        out = _round_trip(source, tmp_path / "out.cif")
        for chain in ("A", "B"):
            numbers = [(r[2], r[3]) for r in _residues(out) if r[1] == chain]
            assert len(numbers) == len(set(numbers))
        assert [(r[1], r[2]) for r in _residues(out) if r[4] == "GLY"] == [
            ("A", "1"),
            ("A", "2"),
            ("A", "3"),
            ("B", "1"),
            ("B", "2"),
        ]


@pytest.fixture(scope="module")
def saved_trypsin(tmp_path_factory):
    if not SFTI.exists():
        pytest.skip("3P8F is not bundled")
    return _round_trip(SFTI, tmp_path_factory.mktemp("returning") / "3p8f_saved.cif")


@pytest.mark.skipif(not SFTI.exists(), reason="3P8F is not bundled")
class TestTrypsin:
    """3P8F lists chain A (trypsin), chain I (SFTI-1), then the glutathione and the waters of A,
    then the waters of I."""

    def test_ligand_and_waters_keep_their_numbers(self, saved_trypsin):
        """Regression: GSH was written A 1, the waters of A 245, 2, 3, ... and those of I 15-20."""
        before, after = _residues(SFTI), _residues(saved_trypsin)
        assert ("HETATM", "A", "1001", "", "GSH") in after
        assert [r for r in after if r[4] == "HOH" and r[1] == "I"] == [
            ("HETATM", "I", str(n), "", "HOH") for n in range(101, 107)
        ]
        assert after == before
        assert sum(1 for r in after if r[4] == "HOH") == 101

    def test_polymer_numbers_and_codes_are_unchanged(self, saved_trypsin):
        def polymer(path):
            return [r for r in _residues(path) if r[0] == "ATOM"]

        assert len(polymer(SFTI)) == 255
        assert polymer(saved_trypsin) == polymer(SFTI)

    def test_structure_rmsd_against_the_saved_copy_is_zero(self, saved_trypsin):
        result = compute_structure_rmsd(SFTI, saved_trypsin, design_chain="I")
        for name in ("rmsd", "bb_rmsd", "rmsd_design", "bb_rmsd_design"):
            assert result[name] is not None
            assert result[name] < 1e-2, name  # the saved coordinates have 4 decimals
        assert "reason" not in result

    def test_reload_gives_the_same_residue_ids(self, saved_trypsin):
        def ids(path):
            topology, _ = load_structure(path)
            return [(r.name, r.id, r.insertionCode) for r in topology.residues()]

        assert ids(saved_trypsin) == ids(SFTI)


@pytest.mark.parametrize("source", [SOMATOSTATIN, CYCLOSPORIN], ids=["1XY4", "1CWA"])
def test_examples_keep_the_residue_columns_of_their_source(source, tmp_path):
    """Every residue column of 1XY4 and 1CWA, waters included, is the one of the source.

    1CWA lists its macrocycle (chain C) and the waters of chains A and C after the polymer of A;
    its 144 waters were written 166, 167, ... instead of 2001, 2002, ... .
    """
    if not source.exists():
        pytest.skip(f"{source.name} is not bundled")
    out = _round_trip(source, tmp_path / "out.cif")
    assert _residues(out) == _residues(source)
