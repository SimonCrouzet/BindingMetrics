"""The Boltz-2 adapter: reading the structure and its tokens, then the confidence files.

Nothing here is real Boltz-2 output. The files are written at test time by
``tests/predictors/synth_boltz2.py`` in the layout that Boltz v2.2.1's writers produce, and a
few structure files are typed out by hand from the Boltz mmCIF example of ``ipsae.py`` and from
the column widths of ``boltz/data/write/pdb.py``.
"""

import dataclasses
import json
import logging

import numpy as np
import pytest

from binding_metrics.metrics._common import load_structure
from binding_metrics.metrics.prediction import summarize_prediction
from binding_metrics.predictors.boltz2 import (
    Boltz2Parser,
    _boltz_tokens,
    _read_atom_sites,
    parse_token_array,
)
from binding_metrics.predictors.registry import PARSERS, get_parser
from tests.predictors import contract, synth, synth_boltz2

NAME = contract.NAME

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

    def test_a_hetero_atom_is_a_token_even_next_to_atoms_of_the_same_residue(self, tmp_path):
        atoms = _atoms(
            ("A", 1, "ALA", ["N", "CA"], False),
            ("A", 1, "ALA", ["OXT"], True),
        )
        assert self._tokens(tmp_path, atoms).token_of_atom.tolist() == [0, 0, 1]

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


# ---------------------------------------------------------------------- the confidence files


def _write(tmp_path, complex_=None, **kwargs):
    synth_boltz2.write_prediction(
        tmp_path, NAME, synth.synthetic_complex() if complex_ is None else complex_, **kwargs
    )
    return tmp_path


def _leaf(tmp_path):
    return synth_boltz2.prediction_directory(tmp_path, NAME)


def _mixed_synthetic():
    """Chain A (ASN, a modified SEP of 10 atoms, GLY), chain B (SER) and a 4-atom ligand C.

    Eight tokens, 24 atoms. Every token has a pLDDT of its own (50, 55, ... 85), the same on all
    the atoms of the token, so the truth is exactly what Boltz-2 can write.
    """
    atoms = _mixed_complex()
    token_of_atom = synth_boltz2.token_index_of_atoms(atoms)
    per_token = 50.0 + 5.0 * np.arange(8)
    i, j = np.arange(8)[:, None], np.arange(8)[None, :]
    base = synth.synthetic_complex()
    return synth.SyntheticComplex(
        atoms=atoms,
        plddt_per_atom=per_token[token_of_atom],
        pae=1.0 + 0.5 * i + 0.25 * j,
        pde=0.5 + 0.25 * i + 0.125 * j,
        scalars=dict(base.scalars),
        chain_ptm={"A": 0.9, "B": 0.8, "C": 0.7},
        chain_pair_iptm={
            "A-B": 0.61,
            "B-A": 0.62,
            "A-C": 0.63,
            "C-A": 0.64,
            "B-C": 0.65,
            "C-B": 0.66,
        },
    )


class TestFindFiles:
    def test_locates_the_files_of_one_sample(self, tmp_path):
        _write(tmp_path)
        files = Boltz2Parser().find_files(tmp_path, NAME)
        leaf = _leaf(tmp_path)
        assert files.structure == leaf / f"{NAME}_model_0.cif"
        assert files.scores == leaf / f"confidence_{NAME}_model_0.json"
        assert files.extra == {
            "plddt": leaf / f"plddt_{NAME}_model_0.npz",
            "pae": leaf / f"pae_{NAME}_model_0.npz",
            "pde": leaf / f"pde_{NAME}_model_0.npz",
        }
        assert files.arrays is None and files.timing is None
        assert files.directory == tmp_path

    def test_sample_k_is_the_file_of_rank_k_minus_one(self, tmp_path):
        _write(tmp_path, sample=1)
        _write(tmp_path, sample=2)
        parser = Boltz2Parser()
        assert parser.find_files(tmp_path, NAME, sample=2).structure.name == f"{NAME}_model_1.cif"
        assert not parser.find_files(tmp_path, NAME, sample=0).has_output()
        assert not parser.find_files(tmp_path, NAME, sample=3).has_output()

    @pytest.mark.parametrize(
        "where",
        ["", f"boltz_results_{NAME}", f"boltz_results_{NAME}/predictions", "leaf"],
        ids=["run root", "results folder", "predictions folder", "sample folder"],
    )
    def test_any_level_of_the_output_tree_can_be_given(self, tmp_path, where):
        _write(tmp_path)
        directory = _leaf(tmp_path) if where == "leaf" else tmp_path / where
        found = Boltz2Parser().find_files(directory, NAME)
        assert found.structure == _leaf(tmp_path) / f"{NAME}_model_0.cif"

    def test_a_directory_of_inputs_names_the_results_folder_after_the_directory(self, tmp_path):
        _write(tmp_path)
        (tmp_path / f"boltz_results_{NAME}").rename(tmp_path / "boltz_results_my_inputs")
        found = Boltz2Parser().find_files(tmp_path, NAME)
        assert found.structure is not None
        assert found.structure.parts[-4] == "boltz_results_my_inputs"

    def test_another_name_is_not_found(self, tmp_path):
        _write(tmp_path)
        assert not Boltz2Parser().find_files(tmp_path, "other").has_output()

    def test_a_second_seed_finds_nothing_because_a_run_has_one_seed(self, tmp_path):
        _write(tmp_path)
        assert not Boltz2Parser().find_files(tmp_path, NAME, seed_index=2).has_output()

    def test_nothing_is_found_in_an_empty_or_absent_directory(self, tmp_path):
        parser = Boltz2Parser()
        assert not parser.find_files(tmp_path, NAME).any_found()
        assert not parser.find_files(tmp_path / "nowhere", NAME).any_found()

    def test_a_pdb_structure_is_found_and_the_cif_is_preferred(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_boltz2, "STRUCTURE_SUFFIX", ".pdb")
        _write(tmp_path)
        assert Boltz2Parser().find_files(tmp_path, NAME).structure.suffix == ".pdb"
        monkeypatch.setattr(synth_boltz2, "STRUCTURE_SUFFIX", ".cif")
        _write(tmp_path)
        assert Boltz2Parser().find_files(tmp_path, NAME).structure.suffix == ".cif"

    def test_the_affinity_and_embedding_files_are_not_part_of_a_sample(self, tmp_path):
        _write(tmp_path)
        leaf = _leaf(tmp_path)
        (leaf / f"affinity_{NAME}.json").write_text("{}", encoding="utf-8")
        np.savez(leaf / f"pre_affinity_{NAME}.npz", coords=np.zeros(3))
        np.savez(leaf / f"embeddings_{NAME}.npz", s=np.zeros(3))
        assert set(Boltz2Parser().find_files(tmp_path, NAME).found()) == {
            "structure",
            "scores",
            "plddt",
            "pae",
            "pde",
        }

    def test_list_samples_gives_the_confidence_scores_without_reading_the_arrays(self, tmp_path):
        for sample, score in ((1, 0.91), (2, 0.85)):
            _write(tmp_path, sample=sample)
            path = _leaf(tmp_path) / f"confidence_{NAME}_model_{sample - 1}.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["confidence_score"] = score
            path.write_text(json.dumps(summary), encoding="utf-8")
        (_leaf(tmp_path) / f"pae_{NAME}_model_0.npz").write_bytes(b"not read")
        refs = Boltz2Parser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample, r.ranking_score) for r in refs] == [
            (1, 1, 0.91),
            (1, 2, 0.85),
        ]


class TestParseTheDefaultComplex:
    def test_the_scalars_go_to_the_record_fields(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path), NAME)
        assert (record.model, record.name) == ("boltz2", NAME)
        assert record.avg_plddt == pytest.approx(82.0)  # complex_plddt 0.82 on 0-100
        assert (record.ptm, record.iptm) == (0.88, 0.76)
        assert record.gpde == 1.23  # complex_pde
        assert record.ranking_score == 0.82
        assert record.ranking_score_name == "confidence_score"
        assert np.isnan(record.has_clash) and np.isnan(record.disorder)  # Boltz-2 has neither
        assert record.timing == {} and record.reasons == []
        record.validate(check_structure=True)

    def test_the_ranking_score_is_read_from_the_file_not_recomputed(self, tmp_path):
        root = _write(tmp_path)
        path = _leaf(root) / f"confidence_{NAME}_model_0.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["confidence_score"] = 0.5  # Boltz's own formula would give 0.808
        path.write_text(json.dumps(summary), encoding="utf-8")
        assert Boltz2Parser().load(root, NAME).ranking_score == 0.5

    def test_the_model_specific_values_are_kept_in_extras(self, tmp_path):
        extras = Boltz2Parser().load(_write(tmp_path), NAME).extras
        assert extras["ligand_iptm"] == 0.0 and extras["protein_iptm"] == 0.76
        assert extras["complex_iplddt"] == pytest.approx(0.82)
        assert extras["complex_ipde"] == 1.23
        assert (extras["model_rank"], extras["n_tokens"]) == (0, 7)
        assert extras["chain_ids_by_index"] == {"0": "A", "1": "B"}
        assert extras["bfactor_matches_plddt"] is True

    def test_plddt_is_rescaled_and_expanded_from_the_token_to_its_atoms(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path), NAME)
        # one value per token (the mean of its two atoms), written on 0-1 and read on 0-100
        token_means = [93, 89, 85, 79, 62, 89, 77]
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(token_means, 2), atol=1e-4)
        assert record.plddt_per_atom.dtype == np.float64

    def test_the_arrays_are_the_files_arrays_without_a_transposition(self, tmp_path):
        root = _write(tmp_path)
        record = Boltz2Parser().load(root, NAME)
        for key in ("pae", "pde"):
            with np.load(_leaf(root) / f"{key}_{NAME}_model_0.npz") as data:
                written = data[key]
            np.testing.assert_array_equal(getattr(record, key), written)
        # pae[i, j] = 1 + 0.5 i + 0.25 j in the file: row 2, column 5 is 3.25, not 4.5
        assert record.pae[2, 5] == pytest.approx(3.25)
        assert record.pde[2, 5] == pytest.approx(0.5 + 0.5 + 0.625)

    def test_the_token_layout_says_which_tokens_belong_to_which_chain(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path), NAME)
        assert record.tokens.token_ranges() == {"A": (0, 4), "B": (4, 7)}
        assert record.tokens.atom_index.tolist() == [0, 2, 4, 6, 8, 10, 12]  # the C-alphas

    def test_a_pdb_structure_gives_the_same_record(self, tmp_path, monkeypatch):
        reference = Boltz2Parser().load(_write(tmp_path / "cif"), NAME)
        monkeypatch.setattr(synth_boltz2, "STRUCTURE_SUFFIX", ".pdb")
        record = Boltz2Parser().load(_write(tmp_path / "pdb"), NAME)
        np.testing.assert_allclose(record.plddt_per_atom, reference.plddt_per_atom, atol=1e-4)
        assert record.tokens.token_ranges() == reference.tokens.token_ranges()
        assert record.chain_pair_iptm == reference.chain_pair_iptm
        assert record.extras["bfactor_matches_plddt"] is True


class TestPaeOrientation:
    """``pae[i, j]``: error of token j aligned on token i, rows are the alignment frame."""

    def _asymmetric_complex(self):
        """Aligned on A, chain B is badly placed (20 A); aligned on B, chain A is fine (2 A)."""
        truth = synth.synthetic_complex()
        pae = np.full((7, 7), 5.0)
        pae[0:4, 4:7] = 20.0  # frame on chain A (rows), scored tokens of chain B (columns)
        pae[4:7, 0:4] = 2.0  # frame on chain B (rows), scored tokens of chain A (columns)
        return dataclasses.replace(truth, pae=pae)

    def test_the_interface_block_of_the_binder_is_cut_from_its_rows(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path, self._asymmetric_complex()), NAME)
        summary = summarize_prediction(
            record, binder_chain="B", receptor_chain="A", include_matrices=True
        )
        # binder rows, receptor columns: how well A is placed when the structure is aligned on B
        np.testing.assert_array_equal(summary["pae_interface"], np.full((3, 4), 2.0))
        # the mean and the maximum use both blocks, so they do not depend on the orientation
        assert summary["mean_interface_pae"] == pytest.approx(11.0)
        assert summary["max_interface_pae"] == pytest.approx(20.0)

    def test_the_default_complex_gives_the_documented_interface_numbers(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path), NAME)
        summary = summarize_prediction(record, binder_chain="B", receptor_chain="A")
        assert summary["mean_interface_pde"] == pytest.approx(1.9375)
        assert summary["mean_interface_pae"] == pytest.approx(3.4375)
        assert summary["max_interface_pae"] == pytest.approx(4.75)
        assert "reason" not in summary


class TestChainKeys:
    def test_the_chain_indices_of_boltz_become_chain_ids(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path), NAME)
        assert record.chain_ptm == {"A": 0.88, "B": 0.80}
        assert record.chain_pair_iptm == {"A-B": 0.76, "B-A": 0.74}
        assert record.extras["chains_ptm_by_index"] == {"0": 0.88, "1": 0.80}
        assert record.extras["pair_chains_iptm_by_index"] == {
            "0": {"0": 0.88, "1": 0.76},
            "1": {"0": 0.74, "1": 0.80},
        }

    def test_index_0_is_the_first_chain_of_the_file_not_the_first_in_the_alphabet(self, tmp_path):
        truth = synth.synthetic_complex()
        swapped = dataclasses.replace(truth, atoms=synth.renamed_atoms(truth, {"A": "B", "B": "A"}))
        record = Boltz2Parser().load(_write(tmp_path, swapped), NAME)
        # the first block of atoms is now called B, so index 0 is B
        assert record.extras["chain_ids_by_index"] == {"0": "B", "1": "A"}
        assert record.chain_ptm == {"B": 0.80, "A": 0.88}
        assert record.extras["chains_ptm_by_index"] == {"0": 0.80, "1": 0.88}
        assert record.chain_pair_iptm == {"B-A": 0.74, "A-B": 0.76}
        assert record.tokens.token_ranges() == {"B": (0, 4), "A": (4, 7)}

    def test_a_pair_key_names_the_scored_chain_first(self, tmp_path):
        root = _write(tmp_path)
        path = _leaf(root) / f"confidence_{NAME}_model_0.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["pair_chains_iptm"]["0"]["1"] = 0.11  # tokens of chain 0 aligned on chain 1
        summary["pair_chains_iptm"]["1"]["0"] = 0.22
        path.write_text(json.dumps(summary), encoding="utf-8")
        record = Boltz2Parser().load(root, NAME)
        assert record.chain_pair_iptm == {"A-B": 0.11, "B-A": 0.22}

    def test_a_single_chain_has_no_iptm_and_no_pair_values(self, tmp_path):
        truth = synth.synthetic_complex()
        one_chain = dataclasses.replace(
            truth,
            atoms=truth.atoms[truth.atoms.chain_id == "A"],
            plddt_per_atom=truth.plddt_per_atom[:8],
            pae=truth.pae[:4, :4],
            pde=truth.pde[:4, :4],
            chain_ptm={"A": 0.88},
            chain_pair_iptm={},
        )
        root = _write(tmp_path, one_chain)
        summary = json.loads(
            (_leaf(root) / f"confidence_{NAME}_model_0.json").read_text(encoding="utf-8")
        )
        assert summary["iptm"] == 0.0  # what Boltz-2 writes when there is no second chain
        record = Boltz2Parser().load(root, NAME)
        assert np.isnan(record.iptm)
        assert record.reasons == [
            "ipTM is not defined for a single chain (Boltz-2 writes 0); left NaN"
        ]
        assert record.chain_ptm == {"A": 0.88} and record.chain_pair_iptm == {}
        assert record.ptm == 0.88
        record.validate(check_structure=True)

    def test_a_chain_index_the_structure_does_not_have_is_refused(self, tmp_path):
        root = _write(tmp_path)
        path = _leaf(root) / f"confidence_{NAME}_model_0.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["chains_ptm"]["2"] = 0.5
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(
            ValueError, match=r"chain indices \['2'\] but the structure has 2 chains"
        ):
            Boltz2Parser().load(root, NAME)


class TestModifiedResiduesAndLigands:
    """The reason the adapter reads the structure: the tokens are not the residues."""

    @pytest.fixture(params=[".cif", ".pdb"])
    def record(self, tmp_path, monkeypatch, request):
        monkeypatch.setattr(synth_boltz2, "STRUCTURE_SUFFIX", request.param)
        return Boltz2Parser().load(_write(tmp_path, _mixed_synthetic()), NAME)

    def test_a_modified_residue_is_one_token_and_a_ligand_atom_one_token(self, record):
        assert record.extras["n_tokens"] == record.pae.shape[0] == 8  # not 24 atoms, not 3 + 1 + 1
        assert record.tokens.token_ranges() == {"A": (0, 3), "B": (3, 4), "C": (4, 8)}
        assert record.tokens.is_atom_token.tolist() == [False] * 4 + [True] * 4
        assert list(record.tokens.extras["res_name"])[:3] == ["ASN", "SEP", "GLY"]

    def test_the_pLDDT_of_the_modified_residue_is_on_all_its_ten_atoms(self, record):
        expected = 50.0 + 5.0 * synth_boltz2.token_index_of_atoms(_mixed_complex())
        np.testing.assert_allclose(record.plddt_per_atom, expected, atol=1e-4)
        assert record.plddt_per_atom[4:14].tolist() == pytest.approx([55.0] * 10, abs=1e-4)
        assert record.extras["bfactor_matches_plddt"] is True

    def test_the_complex_plddt_is_the_mean_over_tokens_not_over_atoms(self, record):
        assert record.avg_plddt == pytest.approx(67.5)  # mean of 50, 55, ..., 85
        assert record.plddt_per_atom.mean() != pytest.approx(67.5, abs=1.0)  # 24 atoms, 10 in SEP

    def test_the_record_is_valid_against_the_structure(self, record):
        record.validate(check_structure=True)

    def test_the_interface_block_is_not_shifted_by_the_modified_residue_or_the_ligand(self, record):
        truth = _mixed_synthetic()
        summary = summarize_prediction(
            record, binder_chain="B", receptor_chain="A", include_matrices=True
        )
        np.testing.assert_allclose(summary["pae_interface"], truth.pae[3:4, 0:3])
        np.testing.assert_allclose(summary["pde_interface"], truth.pde[3:4, 0:3])
        assert "reason" not in summary
        # the ligand chain is cut too: its four atom tokens against the receptor
        ligand = summarize_prediction(
            record, binder_chain="C", receptor_chain="A", include_matrices=True
        )
        np.testing.assert_allclose(ligand["pae_interface"], truth.pae[4:8, 0:3])

    def test_the_chain_values_cover_the_ligand_chain(self, record):
        assert record.chain_ptm == {"A": 0.9, "B": 0.8, "C": 0.7}
        assert record.chain_pair_iptm["A-C"] == 0.63 and record.chain_pair_iptm["C-A"] == 0.64
        assert len(record.chain_pair_iptm) == 6

    def test_the_residue_count_rule_would_have_been_wrong(self, record):
        # 5 residues in the structure but 8 tokens: without the layout the block is refused
        record.tokens = None
        with pytest.warns(UserWarning) as caught:
            summary = summarize_prediction(record, binder_chain="B", receptor_chain="A")
        messages = " | ".join(str(w.message) for w in caught)
        assert "interface PDE skipped" in messages and "interface PAE skipped" in messages
        assert "PAE matrix has 8 tokens" in summary["reason"]


class TestWhenTheStructureCannotTellTheTokens:
    def test_a_structure_file_with_no_atom_records_leaves_the_atom_bound_values_empty(
        self, tmp_path
    ):
        root = _write(tmp_path)
        structure = Boltz2Parser().find_files(root, NAME).structure
        structure.write_text("# stub CIF\n", encoding="utf-8")
        record = Boltz2Parser().load(root, NAME)
        assert record.plddt_per_atom is None  # per token cannot become per atom without atoms
        assert record.tokens is None and record.chain_ptm == {} and record.chain_pair_iptm == {}
        assert record.avg_plddt == pytest.approx(82.0)
        assert record.extras["chains_ptm_by_index"] == {"0": 0.88, "1": 0.80}
        assert len(record.reasons) == 1
        assert record.reasons == [
            f"{NAME}_model_0.cif holds no atom records that could be read, so the tokens are "
            "unknown and PAE and PDE have no layout; plddt_per_atom and chain_ptm and "
            "chain_pair_iptm need them and are left empty"
        ]
        with pytest.raises(Exception):  # noqa: B017 - the stub is no structure for biotite either
            record.atoms()

    def test_a_missing_structure_leaves_the_atom_bound_values_empty_and_says_why(self, tmp_path):
        root = _write(tmp_path)
        Boltz2Parser().find_files(root, NAME).structure.unlink()
        record = Boltz2Parser().load(root, NAME)
        assert record.structure_path is None and record.plddt_per_atom is None
        assert record.chain_ptm == {} and record.tokens is None
        assert record.avg_plddt == pytest.approx(82.0) and record.ptm == 0.88
        np.testing.assert_allclose(record.pae, synth.synthetic_complex().pae)
        assert record.reasons == [
            f"structure file {NAME}_model_0.cif (or .pdb) not found; Boltz-2 values are per "
            "token, so plddt_per_atom and chain_ptm and chain_pair_iptm need it and are left empty"
        ]

    def test_a_token_count_that_differs_from_the_arrays_is_refused(self, tmp_path):
        root = _write(tmp_path)
        leaf = _leaf(root)
        np.savez(leaf / f"plddt_{NAME}_model_0.npz", plddt=np.full(8, 0.9))
        np.savez(leaf / f"pae_{NAME}_model_0.npz", pae=np.ones((8, 8)))
        np.savez(leaf / f"pde_{NAME}_model_0.npz", pde=np.ones((8, 8)))
        with pytest.raises(ValueError, match="7 tokens by the Boltz-2 rule.*have 8"):
            Boltz2Parser().load(root, NAME)

    def test_a_per_atom_plddt_is_refused_rather_than_guessed(self, tmp_path):
        # a confidence head without token_level_confidence would write 14 values for 7 tokens
        root = _write(tmp_path)
        np.savez(_leaf(root) / f"plddt_{NAME}_model_0.npz", plddt=np.full(14, 0.9))
        with pytest.raises(ValueError, match="disagree on the number of tokens"):
            Boltz2Parser().load(root, NAME)

    def test_a_b_factor_column_that_disagrees_is_flagged_but_the_record_is_kept(
        self, tmp_path, caplog
    ):
        root = _write(tmp_path)
        structure = Boltz2Parser().find_files(root, NAME).structure
        truth = synth.synthetic_complex()
        synth_boltz2.write_boltz_structure(truth.atoms, structure, np.zeros(14))
        with caplog.at_level(logging.WARNING, logger="binding_metrics.predictors.boltz2"):
            record = Boltz2Parser().load(root, NAME)
        assert record.extras["bfactor_matches_plddt"] is False
        assert "B-factor column differs" in caplog.text
        np.testing.assert_allclose(
            record.plddt_per_atom, np.repeat([93, 89, 85, 79, 62, 89, 77], 2)
        )


class TestMissingFiles:
    def test_no_output_at_all(self, tmp_path):
        record = Boltz2Parser().load(tmp_path, NAME, sample=3)
        assert record.reasons == [
            f"no Boltz-2 output found for '{NAME}' (sample 3: files {NAME}_model_2.*) in "
            f"{tmp_path}; looked in that directory, in ./{NAME}, in ./predictions/{NAME} and in "
            f"./boltz_results_*/predictions/{NAME}"
        ]
        assert "model_rank" not in record.extras

    def test_a_second_seed_says_that_a_run_has_one_seed(self, tmp_path):
        record = Boltz2Parser().load(_write(tmp_path), NAME, seed_index=2)
        assert record.reasons == [
            "Boltz-2 writes one seed per run, so seed_index must be 1 (got 2); another seed is "
            "another output directory"
        ]

    def test_without_the_summary_the_mean_plddt_comes_from_the_tokens(self, tmp_path):
        root = _write(tmp_path)
        (_leaf(root) / f"confidence_{NAME}_model_0.json").unlink()
        record = Boltz2Parser().load(root, NAME)
        assert record.avg_plddt == pytest.approx(82.0)
        assert np.isnan(record.ptm) and np.isnan(record.gpde) and np.isnan(record.ranking_score)
        assert record.reasons == [f"confidence summary confidence_{NAME}_model_0.json not found"]
        assert record.chain_ptm == {}  # the chain values live in the summary

    @pytest.mark.parametrize("role", ["plddt", "pae", "pde"])
    def test_a_missing_array_leaves_its_field_empty(self, tmp_path, role):
        root = _write(tmp_path)
        (_leaf(root) / f"{role}_{NAME}_model_0.npz").unlink()
        record = Boltz2Parser().load(root, NAME)
        assert record.reasons == [f"{role}_{NAME}_model_0.npz not found"]
        field = {"plddt": record.plddt_per_atom, "pae": record.pae, "pde": record.pde}[role]
        assert field is None
        assert record.avg_plddt == pytest.approx(82.0)  # from complex_plddt
        record.validate(check_structure=True)


class TestCorruptFiles:
    def _summary_path(self, root):
        return _leaf(root) / f"confidence_{NAME}_model_0.json"

    def test_a_json_that_is_cut_off_raises(self, tmp_path):
        root = _write(tmp_path)
        self._summary_path(root).write_text('{"ptm": 0.5,', encoding="utf-8")
        with pytest.raises(ValueError):
            Boltz2Parser().load(root, NAME)

    def test_a_json_that_is_not_an_object_raises(self, tmp_path):
        root = _write(tmp_path)
        self._summary_path(root).write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object, got list"):
            Boltz2Parser().load(root, NAME)

    def test_a_score_that_is_not_a_number_raises(self, tmp_path):
        root = _write(tmp_path)
        path = self._summary_path(root)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["ptm"] = "high"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match="'ptm' is 'high', not a number"):
            Boltz2Parser().load(root, NAME)

    def test_a_chain_dictionary_of_the_wrong_shape_raises(self, tmp_path):
        root = _write(tmp_path)
        path = self._summary_path(root)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["chains_ptm"] = [0.1, 0.2]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match="'chains_ptm' must be a JSON object"):
            Boltz2Parser().load(root, NAME)

    def test_a_pair_dictionary_of_the_wrong_shape_raises(self, tmp_path):
        root = _write(tmp_path)
        path = self._summary_path(root)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["pair_chains_iptm"]["0"] = [0.1]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match=r"'pair_chains_iptm\['0'\]' must be a JSON object"):
            Boltz2Parser().load(root, NAME)

    def test_a_complex_plddt_on_0_100_raises(self, tmp_path):
        root = _write(tmp_path)
        path = self._summary_path(root)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["complex_plddt"] = 82.0
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match="'complex_plddt' is 82, not on the 0-1 scale"):
            Boltz2Parser().load(root, NAME)

    @pytest.mark.parametrize("role", ["plddt", "pae", "pde"])
    def test_an_npz_that_is_not_an_archive_raises_and_names_the_file(self, tmp_path, role):
        root = _write(tmp_path)
        path = _leaf(root) / f"{role}_{NAME}_model_0.npz"
        path.write_bytes(b"not an archive")
        with pytest.raises(ValueError, match=f"{role}_{NAME}_model_0.npz: not a readable .npz"):
            Boltz2Parser().load(root, NAME)

    def test_an_npz_without_the_key_raises(self, tmp_path):
        root = _write(tmp_path)
        np.savez(_leaf(root) / f"pae_{NAME}_model_0.npz", other=np.ones((7, 7)))
        with pytest.raises(ValueError, match=r"no array named 'pae' \(the file has \['other'\]\)"):
            Boltz2Parser().load(root, NAME)

    def test_an_object_array_is_never_unpickled(self, tmp_path):
        marker = tmp_path / "executed"

        class Payload:
            def __reduce__(self):
                return (marker.write_text, ("x",))

        evil = np.empty(1, dtype=object)
        evil[0] = Payload()
        root = _write(tmp_path)
        np.savez(_leaf(root) / f"pde_{NAME}_model_0.npz", pde=evil)
        with pytest.raises(ValueError, match="not a readable .npz"):
            Boltz2Parser().load(root, NAME)
        assert not marker.exists()

    @pytest.mark.parametrize(
        "key, array, message",
        [
            ("plddt", np.full(7, 82.0), "'plddt' runs from 82 to 82; Boltz-2 writes it on 0-1"),
            ("plddt", np.full((7, 2), 0.5), "must have one value per token"),
            ("pae", np.ones((7, 6)), "'pae' must be square"),
            ("pae", -np.ones((7, 7)), "'pae' has a negative value"),
            ("pde", np.full((7, 7), np.nan), "'pde' has a value that is not finite"),
        ],
    )
    def test_arrays_on_the_wrong_scale_or_shape_raise(self, tmp_path, key, array, message):
        root = _write(tmp_path)
        np.savez(_leaf(root) / f"{key}_{NAME}_model_0.npz", **{key: array})
        with pytest.raises(ValueError, match=message):
            Boltz2Parser().load(root, NAME)

    def test_the_array_reader_accepts_the_float16_and_float32_that_boltz_may_write(self, tmp_path):
        for dtype in (np.float16, np.float32):
            path = tmp_path / f"p_{dtype.__name__}.npz"
            np.savez_compressed(path, plddt=np.array([0.5, 0.75], dtype=dtype))
            assert parse_token_array(path, "plddt").tolist() == [0.5, 0.75]


# ---------------------------------------------------------------------- registration and contract


class TestRegistration:
    def test_boltz2_is_registered_and_lazy(self):
        spec = PARSERS["boltz2"]
        assert spec.import_path == "binding_metrics.predictors.boltz2:Boltz2Parser"
        assert (spec.display_name, spec.family) == ("Boltz-2", "af3")
        assert isinstance(get_parser("boltz2"), Boltz2Parser)

    def test_the_attributes_of_the_class(self):
        parser = Boltz2Parser()
        assert (parser.name, parser.display_name, parser.family) == ("boltz2", "Boltz-2", "af3")

    def test_no_capabilities_are_declared(self):
        assert Boltz2Parser.capabilities is None

    def test_a_record_is_loaded_through_the_registry(self, tmp_path):
        record = get_parser("boltz2").load(_write(tmp_path), NAME, chain_map={"A": "R", "B": "P"})
        assert record.model == "boltz2" and set(record.atoms().chain_id) == {"R", "P"}


class TestContractVariants:
    """The PDB layout through the checks of ``contract.py``.

    ``test_contract.py`` runs every check on the mmCIF layout for each registered model.
    """

    def test_the_pdb_variant_of_the_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_boltz2, "STRUCTURE_SUFFIX", ".pdb")
        for check in (
            contract.check_load_valid_record,
            contract.check_sample_and_seed_selection,
            contract.check_missing_files,
            contract.check_corrupt_files,
            contract.check_chain_map,
        ):
            workdir = tmp_path / check.__name__
            workdir.mkdir()
            check("boltz2", workdir)
