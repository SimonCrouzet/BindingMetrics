"""The Boltz-2 adapter: reading the structure and its tokens, then the confidence files.

Nothing here is real Boltz-2 output. The files are written at test time by
``tests/predictors/synth_boltz2.py`` in the layout that Boltz v2.2.1's writers produce, and a
few structure files are typed out by hand from the Boltz mmCIF example of ``ipsae.py`` and from
the column widths of ``boltz/data/write/pdb.py``.
"""

import numpy as np
import pytest

from binding_metrics.metrics._common import load_structure
from binding_metrics.predictors.boltz2 import _boltz_tokens, _read_atom_sites
from tests.predictors import synth, synth_boltz2

struc = pytest.importorskip("biotite.structure")


def _atoms(*residues):
    """AtomArray from ``(chain, res_id, res_name, atom_names, hetero[, ins_code])`` tuples."""
    atoms = []
    for chain, res_id, res_name, names, hetero, *rest in residues:
        for k, name in enumerate(names):
            atoms.append(
                struc.Atom(
                    [3.8 * res_id, 2.0 * k, 10.0 * (ord(chain) - ord("A"))],
                    chain_id=chain,
                    res_id=res_id,
                    ins_code=rest[0] if rest else "",
                    res_name=res_name,
                    atom_name=name,
                    element=name[0],
                    hetero=hetero,
                )
            )
    return struc.array(atoms)


#: A receptor residue, a modified residue (10 atoms, one token in Boltz-2), then an ATP ligand
#: of 4 atoms (four tokens) after the peptide chain B: 3 + 1 + 4 tokens.
def _mixed_complex():
    return _atoms(
        ("A", 1, "ASN", ["N", "CA", "C", "CB"], False),
        ("A", 2, "SEP", ["N", "CA", "C", "O", "CB", "OG", "P", "O1P", "O2P", "O3P"], False),
        ("A", 3, "GLY", ["N", "CA", "C"], False),
        ("B", 1, "SER", ["N", "CA", "C"], False),
        ("C", 1, "ATP", ["PG", "O1G", "O5'", "C1'"], True),
    )


def _bfactor(atoms):
    return np.arange(atoms.array_length(), dtype=float) + 50.0


class TestReadingTheStructure:
    """The reader sees what biotite sees, atom for atom, without importing biotite."""

    @pytest.mark.parametrize("suffix", [".cif", ".pdb"])
    def test_the_atom_records_match_biotite_atom_for_atom(self, tmp_path, suffix):
        atoms = _mixed_complex()
        path = synth_boltz2.write_boltz_structure(
            atoms, tmp_path / f"model{suffix}", _bfactor(atoms)
        )
        sites = _read_atom_sites(path)
        reference = load_structure(path)
        assert len(sites) == reference.array_length() == atoms.array_length()
        assert sites.chain == list(reference.chain_id)
        assert sites.res_id == [int(r) for r in reference.res_id]
        assert sites.atom_name == list(reference.atom_name)
        assert sites.hetero == [bool(h) for h in reference.hetero]

    def test_the_b_factor_column_is_read(self, tmp_path):
        atoms = _mixed_complex()
        path = synth_boltz2.write_boltz_structure(atoms, tmp_path / "m.cif", _bfactor(atoms))
        np.testing.assert_allclose(_read_atom_sites(path).b_factor, _bfactor(atoms))

    def test_boltz_numbers_the_residues_of_each_chain_from_one(self, tmp_path):
        atoms = _atoms(
            ("A", 101, "ALA", ["N", "CA"], False),
            ("A", 102, "GLY", ["N", "CA"], False),
            ("B", 7, "SER", ["N", "CA"], False),
        )
        path = synth_boltz2.write_boltz_structure(atoms, tmp_path / "m.cif", _bfactor(atoms))
        assert _read_atom_sites(path).res_id == [1, 1, 2, 2, 1, 1]

    def test_a_ligand_atom_written_with_a_quote_in_its_name_is_one_value(self, tmp_path):
        atoms = _mixed_complex()
        path = synth_boltz2.write_boltz_structure(atoms, tmp_path / "m.cif", _bfactor(atoms))
        assert "O5'" in _read_atom_sites(path).atom_name
        assert '"O5\'"' in path.read_text(encoding="utf-8")  # it was quoted in the file

    # A Boltz mmCIF typed out by hand from the example of ipsae.py: the same column order, ``.``
    # as the sequence number of the ligand rows, ``?`` as the insertion code.
    LITERAL_CIF = """data_boltz
#
loop_
_struct_asym.id
_struct_asym.entity_id
B 1
C 2
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.label_entity_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ASN 1 1 ? B 10.83538 6.06359 18.45139 1 1 B ASN 91.5 1
ATOM 2 C CA . ASN 1 1 ? B 10.76295 5.07366 19.53232 1 1 B ASN 91.5 1
ATOM 3 N N . SEP 2 2 ? B 11.21770 5.64437 20.88774 1 1 B SEP 80.25 1
ATOM 4 C CA . SEP 2 2 ? B 12.06730 6.51688 20.91168 1 1 B SEP 80.25 1
ATOM 5 P P . SEP 2 2 ? B 11.60137 3.84778 19.19481 1 1 B SEP 80.25 1
HETATM 6 P PG . ATP . 1 ? C -8.79525 6.04621 -4.99212 1 2 C ATP 70.0 1
HETATM 7 O "O5'" . ATP . 1 ? C -10.01901 6.83468 -5.24825 1 2 C ATP 71.0 1
ATOM 8 N N . ASN 1 1 ? B 0.0 0.0 0.0 1 1 B ASN 10.0 2
#
"""

    def test_a_hand_written_boltz_mmcif_is_read_by_column_name(self, tmp_path):
        path = tmp_path / "hand.cif"
        path.write_text(self.LITERAL_CIF, encoding="utf-8")
        sites = _read_atom_sites(path)
        assert sites.chain == ["B"] * 5 + ["C"] * 2  # the second model is not read
        assert sites.res_id == [1, 1, 2, 2, 2, 1, 1]
        assert sites.res_name == ["ASN"] * 2 + ["SEP"] * 3 + ["ATP"] * 2
        assert sites.atom_name == ["N", "CA", "N", "CA", "P", "PG", "O5'"]
        assert sites.hetero == [False] * 5 + [True] * 2
        assert sites.ins_code == [""] * 7
        np.testing.assert_allclose(sites.b_factor, [91.5, 91.5, 80.25, 80.25, 80.25, 70.0, 71.0])

    def test_the_label_columns_are_used_when_the_author_columns_are_absent(self, tmp_path):
        text = self.LITERAL_CIF.replace("_atom_site.auth_seq_id\n", "").replace(
            "_atom_site.auth_asym_id\n", ""
        )
        # drop the matching values: auth_seq_id (8th) and auth_asym_id (16th) of every row
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] in ("ATOM", "HETATM"):
                parts = [v for k, v in enumerate(parts) if k not in (7, 15)]
                line = " ".join(parts)
            rows.append(line)
        path = tmp_path / "label.cif"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        sites = _read_atom_sites(path)
        assert sites.chain[:2] == ["B", "B"] and sites.chain[-1] == "C"
        assert sites.res_id == [1, 1, 2, 2, 2, 0, 0]  # a ligand has no label_seq_id: 0

    def test_a_wrapped_row_is_read_like_one_line(self, tmp_path):
        text = self.LITERAL_CIF.replace("ATOM 1 N N . ASN 1 1 ? B", "ATOM 1 N N .\nASN 1 1 ? B")
        assert text != self.LITERAL_CIF
        path = tmp_path / "wrapped.cif"
        path.write_text(text, encoding="utf-8")
        sites = _read_atom_sites(path)
        assert len(sites) == 7 and sites.res_name[0] == "ASN" and sites.chain[0] == "B"

    # Two residues of chain B, one ligand atom, in the fixed columns of pdb.py.
    LITERAL_PDB = (
        "ATOM      1  N   ASN B   1      10.835   6.064  18.451  1.00 91.50           N\n"
        "ATOM      2  CA  ASN B   1      10.763   5.074  19.532  1.00 91.50           C\n"
        "ATOM      3  N   SER B   2      11.218   5.644  20.888  1.00 80.25           N\n"
        "TER       4      SER B   2\n"
        "HETATM    5  PG  LIG C   1      -8.795   6.046  -4.992  1.00 70.00           P\n"
        "END\n"
    )

    def test_a_hand_written_boltz_pdb_is_read_by_column(self, tmp_path):
        path = tmp_path / "hand.pdb"
        path.write_text(self.LITERAL_PDB, encoding="utf-8")
        sites = _read_atom_sites(path)
        assert sites.chain == ["B", "B", "B", "C"]
        assert sites.res_id == [1, 1, 2, 1]
        assert sites.res_name == ["ASN", "ASN", "SER", "LIG"]
        assert sites.atom_name == ["N", "CA", "N", "PG"]
        assert sites.hetero == [False, False, False, True]
        np.testing.assert_allclose(sites.b_factor, [91.5, 91.5, 80.25, 70.0])

    def test_only_the_first_pdb_model_is_read(self, tmp_path):
        path = tmp_path / "two.pdb"
        path.write_text(
            self.LITERAL_PDB.replace("END\n", "ENDMDL\n") + self.LITERAL_PDB, encoding="utf-8"
        )
        assert len(_read_atom_sites(path)) == 4

    def test_an_insertion_code_separates_two_residues(self, tmp_path):
        line = (
            "ATOM      1  CA  ALA A  52{ins}     10.000   6.000  18.000  1.00 90.00           C\n"
        )
        path = tmp_path / "ins.pdb"
        path.write_text(line.format(ins=" ") + line.format(ins="A") + "END\n", encoding="utf-8")
        assert _read_atom_sites(path).ins_code == ["", "A"]

    @pytest.mark.parametrize("suffix", [".cif", ".pdb"])
    def test_a_file_without_atom_records_gives_none(self, tmp_path, suffix):
        for content in ("# stub CIF\n", "", "data_x\n#\nloop_\n_entity.id\n1\n"):
            path = tmp_path / f"none{suffix}"
            path.write_text(content, encoding="utf-8")
            assert _read_atom_sites(path) is None

    def test_a_binary_file_is_refused_with_its_name(self, tmp_path):
        path = tmp_path / "bad.cif"
        path.write_bytes(b"\x00\x01\xff\xfe binary")
        with pytest.raises(ValueError, match="bad.cif.*not a text structure file"):
            _read_atom_sites(path)

    def test_a_truncated_loop_is_refused(self, tmp_path):
        path = tmp_path / "cut.cif"
        text = self.LITERAL_CIF.split("ATOM 1 N")[0] + "ATOM 1 N N . ASN 1 1\n"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="values for 19 columns.*truncated"):
            _read_atom_sites(path)

    def test_a_loop_without_the_chain_column_is_refused(self, tmp_path):
        text = self.LITERAL_CIF.replace("_atom_site.auth_asym_id", "_atom_site.x_other")
        text = text.replace("_atom_site.label_asym_id", "_atom_site.y_other")
        path = tmp_path / "nochain.cif"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="no auth_asym_id or label_asym_id column"):
            _read_atom_sites(path)

    def test_a_pdb_line_with_a_bad_column_is_refused_with_its_number(self, tmp_path):
        path = tmp_path / "bad.pdb"
        path.write_text(
            "REMARK\n" + self.LITERAL_PDB.replace("B   1      10.835", "B   x      10.835"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"bad.pdb, line 2: not a PDB atom record"):
            _read_atom_sites(path)


class TestTokens:
    """One token per polymer residue, modified or not, one per ligand atom (Boltz-2's rule)."""

    def _tokens(self, tmp_path, atoms=None, suffix=".cif"):
        atoms = _mixed_complex() if atoms is None else atoms
        path = synth_boltz2.write_boltz_structure(atoms, tmp_path / f"m{suffix}", _bfactor(atoms))
        return _boltz_tokens(_read_atom_sites(path))

    @pytest.mark.parametrize("suffix", [".cif", ".pdb"])
    def test_a_modified_residue_is_one_token_and_a_ligand_atom_is_one_token(self, tmp_path, suffix):
        tokens = self._tokens(tmp_path, suffix=suffix)
        # A: ASN, SEP (10 atoms), GLY; B: SER; C: four ligand atoms
        assert tokens.n_tokens == 3 + 1 + 4
        assert list(tokens.layout.chain_id) == ["A"] * 3 + ["B"] + ["C"] * 4
        counts = np.bincount(tokens.token_of_atom)
        assert list(counts) == [4, 10, 3, 3, 1, 1, 1, 1]
        assert tokens.token_of_atom[4:14].tolist() == [1] * 10

    def test_the_layout_flags_the_ligand_atoms_and_names_the_residues(self, tmp_path):
        layout = self._tokens(tmp_path).layout
        assert layout.is_atom_token.tolist() == [False] * 4 + [True] * 4
        assert list(layout.extras["res_name"]) == ["ASN", "SEP", "GLY", "SER"] + ["ATP"] * 4
        assert layout.res_id.tolist() == [1, 2, 3, 1, 1, 1, 1, 1]
        assert layout.problems() == []

    def test_a_residue_token_is_represented_by_its_c_alpha_and_a_ligand_token_by_its_atom(
        self, tmp_path
    ):
        # atoms: ASN 0-3 (CA at 1), SEP 4-13 (CA at 5), GLY 14-16 (CA at 15), SER 17-19 (CA at 18)
        layout = self._tokens(tmp_path).layout
        assert layout.atom_index.tolist() == [1, 5, 15, 18, 20, 21, 22, 23]

    def test_a_residue_without_a_c_alpha_is_represented_by_its_c1_prime_or_its_first_atom(
        self, tmp_path
    ):
        atoms = _atoms(
            ("A", 1, "DA", ["P", "O5'", "C1'"], False),
            ("A", 2, "XXX", ["N", "O"], False),
        )
        assert self._tokens(tmp_path, atoms).layout.atom_index.tolist() == [2, 3]

    def test_the_chain_order_is_the_order_of_appearance_not_alphabetical(self, tmp_path):
        atoms = _atoms(
            ("B", 1, "ALA", ["N", "CA"], False),
            ("A", 1, "GLY", ["N", "CA"], False),
        )
        tokens = self._tokens(tmp_path, atoms)
        assert tokens.chain_order == ["B", "A"]
        assert tokens.layout.token_ranges() == {"B": (0, 1), "A": (1, 2)}

    def test_a_chain_written_in_two_blocks_is_refused(self, tmp_path):
        path = tmp_path / "split.pdb"
        atoms = _atoms(("A", 1, "ALA", ["CA"], False), ("B", 1, "ALA", ["CA"], False))
        synth_boltz2.write_boltz_structure(atoms, path, np.full(2, 90.0))
        lines = path.read_text(encoding="utf-8").splitlines()
        atom_lines = [line for line in lines if line.startswith("ATOM")]
        path.write_text(
            "\n".join(atom_lines + [atom_lines[0].replace("   1 ", "   2 ")]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="chain 'A' is written in two separate blocks"):
            _boltz_tokens(_read_atom_sites(path))

    def test_the_tokens_of_the_default_synthetic_complex_are_its_residues(self, tmp_path):
        truth = synth.synthetic_complex()
        tokens = self._tokens(tmp_path, truth.atoms)
        assert tokens.n_tokens == truth.n_tokens == 7
        assert tokens.token_of_atom.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        assert tokens.layout.token_ranges() == {"A": (0, 4), "B": (4, 7)}

    def test_the_independent_token_rule_of_the_writer_agrees(self, tmp_path):
        atoms = _mixed_complex()
        tokens = self._tokens(tmp_path, atoms)
        np.testing.assert_array_equal(
            tokens.token_of_atom, synth_boltz2.token_index_of_atoms(atoms)
        )


def test_the_reader_and_the_token_rule_need_no_biotite(tmp_path):
    import subprocess
    import sys

    atoms = _mixed_complex()
    path = synth_boltz2.write_boltz_structure(atoms, tmp_path / "m.cif", _bfactor(atoms))
    code = (
        "import sys; sys.modules['biotite'] = None\n"
        "from binding_metrics.predictors.boltz2 import _boltz_tokens, _read_atom_sites\n"
        f"tokens = _boltz_tokens(_read_atom_sites({str(path)!r}))\n"
        "print(tokens.n_tokens)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0 and done.stdout.strip() == "8", done.stderr
