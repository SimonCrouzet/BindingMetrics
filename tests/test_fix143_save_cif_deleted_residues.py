"""save_cif never writes the number of a neighbour when residues were dropped or added (#143).

The numbers of the source were given to the output residues by position in the author chain, so
a residue that prep or the caller had removed from the middle of a chain moved every number after
it: the waters of 3P8F with the glutathione dropped were written 1001, 1101, ..., 1194 instead of
1101, ..., 1195. The residues of source and output are now paired by name along each author chain.
A residue without a partner gets the number OpenMM gave it.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
gemmi = pytest.importorskip("gemmi")

from openmm.app import Modeller  # noqa: E402

from binding_metrics.io.structures import (  # noqa: E402
    _pair_residues,
    load_structure,
    save_cif,
)

DATA = Path(__file__).parent.parent / "data"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"

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
_ATOMS = {"LIG": (("C1", "C"), ("O1", "O")), "HOH": (("O", "O"),)}
_BACKBONE = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))
#: Twelve different names, so that the name of a residue says which source residue it is.
_NAMES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS"]


def _write_source(path: Path, blocks: list[tuple], models: int = 1) -> Path:
    """A minimal mmCIF from ``(label chain, author chain, [(name, number), ...])`` blocks.

    The blocks keep their order. Names other than LIG and HOH have four backbone atoms.
    """
    rows = []
    for model in range(1, models + 1):
        for label, auth, residues in blocks:
            for name, number in residues:
                for atom, element in _ATOMS.get(name, _BACKBONE):
                    serial = len(rows) + 1
                    group = "HETATM" if name in _ATOMS else "ATOM"
                    rows.append(
                        f"{group} {serial} {element} {atom} . {name} {label} 1 {number} ? "
                        f"{1.6 * serial + model} 0.0 0.0 1.00 10.0 {number} {name} {auth} {atom} "
                        f"{model}"
                    )
    path.write_text(_HEADER + "\n".join(rows) + "\n#\n", encoding="utf-8")
    return path


def _residues(path: Path) -> list[tuple[str, str, str]]:
    """``(author chain, number, name)`` of each residue, in order."""
    block = gemmi.cif.read(str(path)).sole_block()
    table = block.find("_atom_site.", ["auth_asym_id", "auth_seq_id", "auth_comp_id"])
    residues, previous = [], None
    for row in table:
        key = (row[0], row[1], row[2])
        if key != previous:
            residues.append(key)
            previous = key
    return residues


def _save_without(source: Path, out: Path, dropped) -> Path:
    """Save the topology of ``source`` minus the residues ``dropped(residue)`` selects."""
    topology, positions = load_structure(source)
    modeller = Modeller(topology, positions)
    modeller.delete([res for res in topology.residues() if dropped(res)])
    save_cif(modeller.topology, modeller.positions, out, source_cif_path=source)
    return out


#: Source numbers of the polymer: 100, 101, ... so that no number is also an OpenMM number.
_POLYMER = [(name, 100 + i) for i, name in enumerate(_NAMES)]


class TestResiduesDropped:
    @pytest.mark.parametrize(
        "dropped",
        [
            pytest.param({100}, id="first"),
            pytest.param({105}, id="middle"),
            pytest.param({111}, id="last"),
            pytest.param({104, 105}, id="two consecutive"),
            pytest.param({100, 101, 102}, id="start, three"),
            pytest.param({109, 110, 111}, id="end, three"),
            pytest.param({100, 105, 111}, id="start, middle and end"),
            pytest.param({101, 102, 106, 107, 108}, id="two runs"),
            pytest.param(set(range(100, 111)), id="all but the last"),
        ],
    )
    def test_the_residues_that_remain_keep_their_own_numbers(self, tmp_path, dropped):
        """Regression: with 105 dropped, 106 was written 105, 107 106, and so on."""
        source = _write_source(tmp_path / "source.cif", [("A", "A", _POLYMER)])
        out = _save_without(source, tmp_path / "out.cif", lambda r: int(r.id) in dropped)
        expected = [("A", str(n), name) for name, n in _POLYMER if n not in dropped]
        assert _residues(out) == expected

    def test_a_chain_removed_entirely_leaves_the_others_alone(self, tmp_path):
        source = _write_source(
            tmp_path / "source.cif",
            [("A", "A", _POLYMER[:6]), ("B", "B", _POLYMER[6:]), ("C", "A", [("LIG", 900)])],
        )
        out = _save_without(source, tmp_path / "out.cif", lambda r: r.chain.id == "B")
        assert _residues(out) == [("A", str(n), name) for name, n in _POLYMER[:6]] + [
            ("A", "900", "LIG")
        ]

    def test_waters_keep_their_numbers_when_the_ligand_before_them_is_dropped(self, tmp_path):
        source = _write_source(
            tmp_path / "source.cif",
            [
                ("A", "A", _POLYMER),
                ("B", "A", [("LIG", 1001)]),
                ("C", "A", [("HOH", n) for n in (1101, 1102, 1103, 1104)]),
            ],
        )
        out = _save_without(source, tmp_path / "out.cif", lambda r: r.name == "LIG")
        assert [r[1] for r in _residues(out) if r[2] == "HOH"] == ["1101", "1102", "1103", "1104"]

    @pytest.mark.skipif(not SFTI.exists(), reason="3P8F is not bundled")
    def test_3p8f_waters_keep_1101_to_1195_without_the_glutathione(self, tmp_path):
        """Regression: the 95 waters of chain A were written 1001, 1101, ..., 1194."""
        out = _save_without(SFTI, tmp_path / "out.cif", lambda r: r.name == "GSH")
        before, after = _residues(SFTI), _residues(out)
        assert after == [r for r in before if r[2] != "GSH"]
        waters = [int(r[1]) for r in after if r[0] == "A" and r[2] == "HOH"]
        assert waters == list(range(1101, 1196))

    def test_residues_that_the_names_cannot_tell_apart_get_openmm_numbers(self, tmp_path):
        """One of five glycines in a row is gone, and nothing says which one.

        The glycines cannot be paired with the source, so they take the numbers OpenMM gave them
        (2 to 5 in the chain) and not the number of a neighbour; the residues on both sides are
        unambiguous and keep theirs.
        """
        run = [("ALA", 100)] + [("GLY", 101 + i) for i in range(5)] + [("SER", 106)]
        source = _write_source(tmp_path / "source.cif", [("A", "A", run)])
        out = _save_without(source, tmp_path / "out.cif", lambda r: r.id == "103")
        assert _residues(out) == [
            ("A", "100", "ALA"),
            ("A", "2", "GLY"),
            ("A", "3", "GLY"),
            ("A", "4", "GLY"),
            ("A", "5", "GLY"),
            ("A", "106", "SER"),
        ]


class TestResiduesAdded:
    def test_capping_groups_get_numbers_of_their_own(self, tmp_path):
        """ACE and NME are not in the source: the residues between them keep their numbers."""
        source = _write_source(tmp_path / "source.cif", [("A", "A", _POLYMER[:5])])
        capped = _write_source(
            tmp_path / "capped.cif",
            [
                (
                    "A",
                    "A",
                    [
                        ("ACE", 1),
                        *[(n, 1 + i + 1) for i, (n, _) in enumerate(_POLYMER[:5])],
                        ("NME", 7),
                    ],
                )
            ],
        )
        topology, positions = load_structure(capped)
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source)
        written = _residues(out)
        assert [r[1] for r in written if r[2] not in ("ACE", "NME")] == [
            "100",
            "101",
            "102",
            "103",
            "104",
        ]
        numbers = [r[1] for r in written]
        assert len(set(numbers)) == len(numbers)
        assert [r[2] for r in written] == ["ACE", *_NAMES[:5], "NME"]

    def test_a_residue_added_in_the_middle_does_not_shift_the_others(self, tmp_path):
        source = _write_source(tmp_path / "source.cif", [("A", "A", _POLYMER[:3] + _POLYMER[4:6])])
        filled = _write_source(tmp_path / "filled.cif", [("A", "A", _POLYMER[:6])])
        topology, positions = load_structure(filled)
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source)
        numbers = {name: number for _, number, name in _residues(out)}
        assert [numbers[name] for name in ("ALA", "ARG", "ASN", "CYS", "GLN")] == [
            "100",
            "101",
            "102",
            "104",
            "105",
        ]
        assert numbers["ASP"] not in {"100", "101", "102", "104", "105"}


class TestNamesThatPrepChanges:
    def test_protonation_variants_and_waters_pair_with_their_source_names(self, tmp_path):
        residues = [("HIS", 100), ("CYS", 101), ("ASP", 102), ("LYS", 103), ("GLY", 104)]
        source = _write_source(
            tmp_path / "source.cif",
            [("A", "A", residues), ("B", "A", [("HOH", 700), ("HOH", 701)])],
        )
        topology, positions = load_structure(source)
        renamed = {"HIS": "HID", "CYS": "CYX", "ASP": "ASH", "LYS": "LYN", "HOH": "WAT"}
        for residue in topology.residues():
            residue.name = renamed.get(residue.name, residue.name)
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source)
        assert [r[1] for r in _residues(out)] == ["100", "101", "102", "103", "104", "700", "701"]

    def test_a_residue_renamed_one_for_one_keeps_its_number(self, tmp_path):
        """MSE becoming MET: the same count of residues between two anchors pairs by position."""
        source = _write_source(tmp_path / "source.cif", [("A", "A", _POLYMER[:6])])
        topology, positions = load_structure(source)
        next(r for r in topology.residues() if r.id == "103").name = "MET"
        out = tmp_path / "out.cif"
        save_cif(topology, positions, out, source_cif_path=source)
        assert [r[1] for r in _residues(out)] == [str(n) for _, n in _POLYMER[:6]]

    def test_a_dropped_residue_next_to_a_renamed_one_is_not_paired_with_either(self, tmp_path):
        """101 is dropped and 102 is renamed: its source neighbours would give it 101 or 102."""
        source = _write_source(tmp_path / "source.cif", [("A", "A", _POLYMER[:5])])
        topology, positions = load_structure(source)
        modeller = Modeller(topology, positions)
        modeller.delete([r for r in topology.residues() if r.id == "101"])
        next(r for r in modeller.topology.residues() if r.id == "102").name = "MET"
        out = tmp_path / "out.cif"
        save_cif(modeller.topology, modeller.positions, out, source_cif_path=source)
        numbers = [r[1] for r in _residues(out)]
        assert numbers[0] == "100"
        assert numbers[2:] == ["103", "104"]
        assert numbers[1] == "2"  # the number OpenMM gave it, neither 101 nor 102


def test_models_after_the_first_do_not_count(tmp_path):
    """An NMR-style source lists every residue once per model; only the first model is read."""
    source = _write_source(tmp_path / "source.cif", [("A", "A", _POLYMER[:4])], models=3)
    topology, positions = load_structure(source)
    out = tmp_path / "out.cif"
    save_cif(topology, positions, out, source_cif_path=source)
    assert [r[1] for r in _residues(out)] == ["100", "101", "102", "103"]


class TestPairResidues:
    """The pairing function on name lists: ``{output index: source index}``."""

    def test_identical_lists_pair_by_position(self):
        assert _pair_residues(["ALA", "GLY"], ["ALA", "GLY"]) == {0: 0, 1: 1}

    def test_variant_names_equal_their_parent(self):
        assert _pair_residues(["HIS", "CYS", "HOH"], ["HID", "CYX", "WAT"]) == {0: 0, 1: 1, 2: 2}

    def test_a_missing_residue_is_skipped(self):
        assert _pair_residues(["ALA", "GLY", "SER", "THR"], ["ALA", "SER", "THR"]) == {
            0: 0,
            1: 2,
            2: 3,
        }

    def test_added_residues_have_no_partner(self):
        assert _pair_residues(["ALA", "GLY"], ["ACE", "ALA", "GLY", "NME"]) == {1: 0, 2: 1}

    def test_a_run_that_lost_a_member_is_not_paired(self):
        assert _pair_residues(
            ["ALA", "GLY", "GLY", "GLY", "SER"], ["ALA", "GLY", "GLY", "SER"]
        ) == {
            0: 0,
            3: 4,
        }

    def test_an_alignment_that_depends_on_the_direction_is_not_paired(self):
        """ALA GLY SER GLY with one GLY left: the first or the second?"""
        pairs = _pair_residues(["ALA", "GLY", "SER", "GLY"], ["ALA", "GLY"])
        assert pairs == {0: 0}
