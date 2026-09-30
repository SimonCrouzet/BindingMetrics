"""The AlphaFold2 / AlphaFold-Multimer / ColabFold adapter: layouts, arrays and the B-factor reader.

Every file is written at test time by ``tests/predictors/synth_af2.py`` from the synthetic complex
of ``synth.py`` (asymmetric PAE, so a transposed matrix differs from the truth) or by hand, in the
layouts of ``predictors/af2.py``. Nothing here is real model output.
"""

import gzip

import numpy as np
import pytest

from binding_metrics.metrics.prediction import summarize_prediction
from binding_metrics.predictors import af2
from binding_metrics.predictors.af2 import (
    load_bfactor_record,
    read_bfactor_plddt,
)
from tests.predictors import contract, synth, synth_af2

NAME = contract.NAME

#: The default complex has two chains of 4 and 3 residues; these are their pLDDT per residue.
RESIDUE_PLDDT = np.array([93.0, 89.0, 85.0, 79.0, 62.0, 89.0, 77.0])


def _truth(**kwargs):
    return synth.synthetic_complex(**kwargs)


def _atom_line(
    serial, name, residue, chain, number, b_factor, *, icode=" ", altloc=" ", kind="ATOM"
):
    """A PDB atom record on the fixed columns (element and coordinates are filler)."""
    return (
        f"{kind:<6}{serial:>5} {name:<4}{altloc}{residue:>3} {chain}{number:>4}{icode}   "
        f"{1.0 * serial:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{b_factor:>6.2f}          "
        f"{name.strip()[0]:>2}"
    )


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _biotite_reference(path):
    """The atoms ``load_structure`` reads, with the B-factor column, and their residue numbers."""
    atoms = synth.pdb_io.get_structure(
        synth.pdb_io.PDBFile.read(str(path)), model=1, extra_fields=["b_factor"]
    )
    starts = synth.struc.get_residue_starts(atoms)
    marks = np.zeros(atoms.array_length(), dtype=int)
    marks[starts] = 1
    return np.asarray(atoms.b_factor), np.cumsum(marks) - 1


# ---------------------------------------------------------------------------
# Structure files: B-factors and residues, in biotite's atom order
# ---------------------------------------------------------------------------


class TestStructureScan:
    def _assert_matches_biotite(self, path):
        scan = af2._scan_structure(path)
        b_factor, residues = _biotite_reference(path)
        np.testing.assert_allclose(scan.b_factor, b_factor)
        np.testing.assert_array_equal(scan.residue_of_atom, residues)
        assert scan.n_residues == residues.max() + 1

    def test_a_written_complex_agrees_with_biotite(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth())
        self._assert_matches_biotite(path)
        scan = af2._scan_structure(path)
        assert scan.n_residues == 7
        np.testing.assert_array_equal(scan.residue_of_atom, np.repeat(np.arange(7), 2))

    def test_alphafold_style_text_with_insertion_codes_hetero_and_terminators(self, tmp_path):
        lines = [
            "HEADER    PREDICTED",
            _atom_line(1, " N  ", "GLY", "A", 1, 91.5),
            _atom_line(2, " CA ", "GLY", "A", 1, 91.5),
            _atom_line(3, " CA ", "ALA", "A", 2, 88.25),
            _atom_line(4, " CA ", "SER", "A", 2, 70.0, icode="A"),
            _atom_line(5, " CB ", "SER", "A", 2, 70.0, icode="A"),
            "TER",
            _atom_line(6, " CA ", "LYS", "B", 1, 60.0),
            _atom_line(7, " ZN ", " ZN", "B", 2, 10.0, kind="HETATM"),
            "END",
        ]
        path = _write_lines(tmp_path / "a.pdb", lines)
        self._assert_matches_biotite(path)
        scan = af2._scan_structure(path)
        assert scan.n_residues == 5  # 1, 2, 2A, B1, B2: the insertion code starts a residue

    def test_only_the_first_model_is_read(self, tmp_path):
        lines = ["MODEL        1"]
        lines += [
            _atom_line(1, " CA ", "ALA", "A", 1, 80.0),
            _atom_line(2, " CA ", "ALA", "A", 2, 82.0),
        ]
        lines += ["ENDMDL", "MODEL        2"]
        lines += [
            _atom_line(1, " CA ", "ALA", "A", 1, 10.0),
            _atom_line(2, " CA ", "ALA", "A", 2, 12.0),
        ]
        lines += ["ENDMDL"]
        path = _write_lines(tmp_path / "m.pdb", lines)
        self._assert_matches_biotite(path)
        assert af2._scan_structure(path).b_factor.tolist() == [80.0, 82.0]

    def test_alternate_locations_keep_the_first_one_as_biotite_does(self, tmp_path):
        lines = [
            _atom_line(1, " N  ", "SER", "A", 1, 80.0),
            _atom_line(2, " CA ", "SER", "A", 1, 81.0, altloc="B"),
            _atom_line(3, " CA ", "SER", "A", 1, 82.0, altloc="A"),
            _atom_line(4, " CB ", "SER", "A", 1, 83.0, altloc="B"),
            _atom_line(5, " CB ", "SER", "A", 1, 84.0, altloc="A"),
            _atom_line(6, " CA ", "ALA", "A", 2, 70.0),
        ]
        path = _write_lines(tmp_path / "alt.pdb", lines)
        self._assert_matches_biotite(path)
        # the first letter met in the residue is B, so the B atoms and the plain atoms stay
        assert af2._scan_structure(path).b_factor.tolist() == [80.0, 81.0, 83.0, 70.0]

    def test_windows_line_endings(self, tmp_path):
        path = tmp_path / "crlf.pdb"
        path.write_bytes(
            (_atom_line(1, " CA ", "ALA", "A", 1, 80.0) + "\r\n").encode("ascii")
            + (_atom_line(2, " CA ", "ALA", "A", 2, 82.0) + "\r\n").encode("ascii")
        )
        assert af2._scan_structure(path).b_factor.tolist() == [80.0, 82.0]

    def test_a_gzipped_pdb(self, tmp_path):
        plain = synth_af2.write_bare(tmp_path, "x", _truth())
        packed = tmp_path / "x.pdb.gz"
        packed.write_bytes(gzip.compress(plain.read_bytes()))
        np.testing.assert_array_equal(
            af2._scan_structure(packed).b_factor, af2._scan_structure(plain).b_factor
        )

    @pytest.mark.parametrize("suffix", [".cif", ".cif.gz"])
    def test_mmcif_agrees_with_the_pdb_of_the_same_structure(self, tmp_path, suffix):
        pdb = synth_af2.write_bare(tmp_path, "x", _truth())
        cif = synth_af2.write_bare(tmp_path, "y", _truth(), suffix=suffix)
        expected, from_cif = af2._scan_structure(pdb), af2._scan_structure(cif)
        np.testing.assert_allclose(from_cif.b_factor, expected.b_factor)
        np.testing.assert_array_equal(from_cif.residue_of_atom, expected.residue_of_atom)

    def test_a_short_atom_record_is_refused_with_its_line_number(self, tmp_path):
        path = _write_lines(
            tmp_path / "s.pdb",
            [_atom_line(1, " CA ", "ALA", "A", 1, 80.0), "ATOM      2  CA  ALA A   2       1.000"],
        )
        with pytest.raises(ValueError, match=r"line 2 is an atom record of 38 columns"):
            af2._scan_structure(path)

    def test_a_b_factor_field_that_is_not_a_number_is_refused(self, tmp_path):
        line = _atom_line(1, " CA ", "ALA", "A", 1, 80.0)
        path = _write_lines(tmp_path / "n.pdb", [line[:60] + "  abcd" + line[66:]])
        with pytest.raises(ValueError, match=r"line 1 has '  abcd' in the B-factor field"):
            af2._scan_structure(path)

    def test_a_file_without_atoms_is_refused(self, tmp_path):
        path = _write_lines(tmp_path / "e.pdb", ["# stub CIF"])
        with pytest.raises(ValueError, match="has no atom records"):
            af2._scan_structure(path)


# ---------------------------------------------------------------------------
# pLDDT from the B-factor column, and the record of a bare structure
# ---------------------------------------------------------------------------


class TestPlddtFromBfactor:
    def test_percent_values_are_taken_as_they_are(self):
        values, rescaled, problem = af2._plddt_from_bfactor(np.array([12.5, 97.8]))
        assert values.tolist() == [12.5, 97.8] and not rescaled and problem == ""

    def test_a_column_that_stays_below_one_is_taken_as_a_fraction(self):
        values, rescaled, _ = af2._plddt_from_bfactor(np.array([0.5, 0.97]))
        np.testing.assert_allclose(values, [50.0, 97.0])
        assert rescaled

    def test_the_scale_can_be_forced(self):
        forced, rescaled, _ = af2._plddt_from_bfactor(np.array([0.5, 0.9]), "percent")
        assert forced.tolist() == [0.5, 0.9] and not rescaled
        values, rescaled, _ = af2._plddt_from_bfactor(np.array([0.5, 0.9]), "fraction")
        np.testing.assert_allclose(values, [50.0, 90.0])
        assert rescaled

    @pytest.mark.parametrize(
        "column, expected",
        [
            (np.zeros(4), "all zero"),
            (np.array([10.0, 250.0]), "from 10 to 250, outside 0-100"),
            (np.array([-3.0, 50.0]), "from -3 to 50, outside 0-100"),
            (np.array([10.0, np.nan]), "not a finite number"),
        ],
    )
    def test_a_column_that_cannot_be_plddt_is_reported(self, column, expected):
        values, _, problem = af2._plddt_from_bfactor(column)
        assert values is None
        assert expected in problem

    def test_an_unknown_scale_is_refused(self):
        with pytest.raises(ValueError, match="scale must be one of"):
            af2._plddt_from_bfactor(np.array([1.0]), "percentage")


class TestReadBfactorPlddt:
    def test_the_residue_value_is_repeated_over_the_atoms(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth())
        np.testing.assert_allclose(read_bfactor_plddt(path), np.repeat(RESIDUE_PLDDT, 2))

    def test_a_fraction_column_is_multiplied_by_100(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth(), bfactor_scale=0.01)
        np.testing.assert_allclose(read_bfactor_plddt(path), np.repeat(RESIDUE_PLDDT, 2), atol=1e-6)

    def test_the_atoms_are_in_the_order_biotite_reads(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth())
        b_factor, _ = _biotite_reference(path)
        np.testing.assert_allclose(read_bfactor_plddt(path), b_factor)

    def test_a_column_without_plddt_raises_with_the_file_name(self, tmp_path):
        atoms = synth_af2.af2_atoms(_truth())
        atoms.set_annotation("b_factor", np.zeros(atoms.array_length()))
        path = synth.write_structure(atoms, tmp_path / "zero.pdb")
        with pytest.raises(ValueError, match=r"zero\.pdb is not a pLDDT: the column is all zero"):
            read_bfactor_plddt(path)


class TestLoadBfactorRecord:
    """A bare AlphaFold2 or BindCraft complex, with pLDDT only in its B-factor column."""

    def test_the_record_has_plddt_and_nothing_else(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "design_l80_s1_model1", _truth())
        record = load_bfactor_record(path)
        record.validate(check_structure=True)
        assert record.model == "af2"
        assert record.name == "design_l80_s1_model1"
        assert record.structure_path == path
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        assert record.avg_plddt == pytest.approx(np.repeat(RESIDUE_PLDDT, 2).mean())
        for value in (record.ptm, record.iptm, record.gpde, record.ranking_score):
            assert np.isnan(value)
        assert record.pae is None and record.pde is None and record.tokens is None
        assert record.chain_ptm == {} and record.chain_pair_iptm == {}
        assert record.extras["plddt_source"] == "b_factor"
        assert record.extras["layout"] == "bare"
        assert len(record.reasons) == 1
        assert "B-factor column" in record.reasons[0] and "PAE" in record.reasons[0]

    @pytest.mark.parametrize("suffix", [".cif", ".pdb.gz", ".cif.gz"])
    def test_the_other_structure_formats(self, tmp_path, suffix):
        path = synth_af2.write_bare(tmp_path, "x", _truth(), suffix=suffix)
        record = load_bfactor_record(path)
        record.validate(check_structure=True)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        assert record.name == "x"

    def test_a_fraction_column_is_rescaled_and_the_rescaling_recorded(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth(), bfactor_scale=0.01)
        record = load_bfactor_record(path)
        record.validate()
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2), atol=1e-6)
        assert record.extras["bfactor_scale"] == "0-1, multiplied by 100"

    def test_no_plddt_in_the_column_gives_a_reason_not_an_error(self, tmp_path):
        atoms = synth_af2.af2_atoms(_truth())
        atoms.set_annotation("b_factor", np.zeros(atoms.array_length()))
        path = synth.write_structure(atoms, tmp_path / "zero.pdb")
        record = load_bfactor_record(path)
        assert record.plddt_per_atom is None and np.isnan(record.avg_plddt)
        assert "all zero" in "; ".join(record.reasons)
        record.validate()

    def test_a_missing_file_gives_a_reason(self, tmp_path):
        record = load_bfactor_record(tmp_path / "absent.pdb")
        assert record.structure_path is None and record.plddt_per_atom is None
        assert "absent.pdb not found" in record.reasons[0]

    def test_a_file_without_atoms_raises(self, tmp_path):
        path = _write_lines(tmp_path / "e.pdb", ["REMARK nothing"])
        with pytest.raises(ValueError, match="no atom records"):
            load_bfactor_record(path)

    def test_the_chain_map_renames_the_chains_of_the_atoms(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth())
        record = load_bfactor_record(path, chain_map={"A": "R", "B": "P"})
        assert set(record.atoms().chain_id) == {"R", "P"}
        with pytest.raises(ValueError, match="same ID"):
            load_bfactor_record(path, chain_map={"A": "Z", "B": "Z"})

    def test_an_unknown_scale_is_refused(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, "x", _truth())
        with pytest.raises(ValueError, match="scale must be one of"):
            load_bfactor_record(path, scale="0-100")

    def test_the_binder_plddt_of_a_bare_complex_reaches_the_summary(self, tmp_path):
        # chain B is the binder: its residues (pLDDT 89, 77 and 62) average to 76
        record = load_bfactor_record(synth_af2.write_bare(tmp_path, "x", _truth()))
        summary = summarize_prediction(record, binder_chain="B", receptor_chain="A")
        np.testing.assert_allclose(summary["binder_plddt_per_residue"], [62.0, 89.0, 77.0])
        assert summary["binder_avg_plddt"] == pytest.approx(76.0)
        assert np.isnan(summary["ptm"]) and np.isnan(summary["mean_interface_pae"])
        assert "B-factor column" in summary["reason"]
