"""The AlphaFold2 / AlphaFold-Multimer / ColabFold adapter: layouts, arrays and the B-factor reader.

Every file is written at test time by ``tests/predictors/synth_af2.py`` from the synthetic complex
of ``synth.py`` (asymmetric PAE, so a transposed matrix differs from the truth) or by hand, in the
layouts of ``predictors/af2.py``. Nothing here is real model output.
"""

import gzip
import json
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from binding_metrics.metrics.evobind import compute_evobind_adversarial_from_records
from binding_metrics.metrics.prediction import compute_prediction_metrics, summarize_prediction
from binding_metrics.predictors import af2
from binding_metrics.predictors.af2 import (
    AlphaFold2Parser,
    load_bfactor_record,
    parse_colabfold_scores,
    parse_confidence_json,
    parse_pae_json,
    parse_ranking_debug,
    parse_result_pickle,
    read_bfactor_plddt,
    read_result_pickle,
)
from binding_metrics.predictors.registry import PARSERS, get_parser
from tests.predictors import contract, synth, synth_af2
from tests.predictors.test_adversarial import _as_synthetic_complex, _complex_atoms

NAME = contract.NAME
REPO_ROOT = contract.REPO_ROOT
GARBAGE = contract.GARBAGE

#: The default complex has two chains of 4 and 3 residues; these are their pLDDT per residue.
RESIDUE_PLDDT = np.array([93.0, 89.0, 85.0, 79.0, 62.0, 89.0, 77.0])


def _truth(**kwargs):
    return synth.synthetic_complex(**kwargs)


def _colabfold(tmp_path, **kwargs):
    synth_af2.write_colabfold(tmp_path, NAME, _truth(), **kwargs)
    return tmp_path


def _alphafold(tmp_path, complex_=None, **kwargs):
    return synth_af2.write_alphafold(tmp_path, complex_ or _truth(), **kwargs)


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


def _complex_with_atoms_per_residue(counts, *, plddt=None):
    """A complex with ``counts[chain][i]`` atoms in each residue and one pLDDT per residue."""
    names = ["N", "CA", "C", "O", "CB", "CG", "CD"]
    atoms, values, n_residues = [], [], 0
    for chain, chain_counts in counts.items():
        for i, n_atoms in enumerate(chain_counts):
            value = 50.0 + 5.0 * n_residues if plddt is None else plddt[n_residues]
            n_residues += 1
            for k in range(n_atoms):
                atoms.append(
                    synth.struc.Atom(
                        [3.8 * i, 1.5 * k, 0.0],
                        chain_id=chain,
                        res_id=i + 1,
                        res_name="ALA",
                        atom_name=names[k],
                        element="C",
                    )
                )
                values.append(value)
    array = synth.struc.array(atoms)
    index = np.arange(n_residues)[:, None], np.arange(n_residues)[None, :]
    return synth.SyntheticComplex(
        atoms=array,
        plddt_per_atom=np.array(values),
        pae=1.0 + 0.5 * index[0] + 0.25 * index[1],
        pde=0.5 + 0.25 * index[0] + 0.125 * index[1],
        scalars=_truth().scalars,
        chain_ptm={},
        chain_pair_iptm={},
    )


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


# ---------------------------------------------------------------------------
# The ColabFold scores JSON
# ---------------------------------------------------------------------------


def _scores_file(tmp_path, payload):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestColabFoldScores:
    def test_a_multimer_file_with_the_extra_ptm_keys(self, tmp_path):
        path = _scores_file(tmp_path, synth_af2.colabfold_scores(_truth()))
        parsed = parse_colabfold_scores(path)
        np.testing.assert_allclose(parsed["plddt_per_residue"], RESIDUE_PLDDT)
        assert parsed["pae"].shape == (7, 7)
        assert parsed["ptm"] == 0.88 and parsed["iptm"] == 0.76
        assert parsed["chain_ptm"] == {"A": 0.88, "B": 0.8}
        assert parsed["chain_pair_iptm"] == {"A-B": 0.76}
        assert set(parsed["extras"]) == {"pairwise_actifptm", "actifptm"}

    def test_the_ipsae_family_of_colabfold_1_6_3_goes_to_the_extras(self, tmp_path):
        payload = synth_af2.colabfold_scores(_truth())
        payload.update(ipsae={"A-B": 0.61}, pdockq={"A-B": 0.4}, pdockq2={"A-B": 0.3})
        extras = parse_colabfold_scores(_scores_file(tmp_path, payload))["extras"]
        assert extras["ipsae"] == {"A-B": 0.61}
        assert extras["pdockq"] == {"A-B": 0.4} and extras["pdockq2"] == {"A-B": 0.3}

    def test_the_pae_matrix_is_not_transposed(self, tmp_path):
        # pae[i][j] = 1 + 0.5 i + 0.25 j: row 0 runs 1.0 .. 2.5, column 0 runs 1.0 .. 4.0
        pae = parse_colabfold_scores(_scores_file(tmp_path, synth_af2.colabfold_scores(_truth())))[
            "pae"
        ]
        assert pae[0, 6] == pytest.approx(2.5) and pae[6, 0] == pytest.approx(4.0)
        np.testing.assert_allclose(pae, _truth().pae)

    def test_a_monomer_file_has_only_plddt(self, tmp_path):
        parsed = parse_colabfold_scores(_scores_file(tmp_path, {"plddt": [90.0, 80.0, 70.0]}))
        assert parsed["pae"] is None and np.isnan(parsed["ptm"]) and np.isnan(parsed["iptm"])
        assert parsed["chain_ptm"] == {} and parsed["extras"] == {}

    @pytest.mark.parametrize(
        "payload, message",
        [
            ({"plddt": [0.5, 0.9, 0.7]}, "0-1 scale"),
            ({"plddt": [50.0, 120.0]}, "outside 0-100"),
            ({"plddt": []}, "non-empty"),
            ({"plddt": ["a", "b"]}, "not a list of numbers"),
            ({"plddt": [50.0, 60.0], "pae": [[1.0, 2.0, 3.0]] * 3}, "3 by 3 but pLDDT has 2"),
            ({"plddt": [50.0, 60.0], "pae": [1.0, 2.0]}, "square matrix"),
            ({"plddt": [50.0, 60.0], "ptm": "high"}, "'ptm' is 'high'"),
            ({"plddt": [50.0, 60.0], "per_chain_ptm": [0.5]}, "must be an object"),
            ({"pae": [[1.0]]}, "not a ColabFold scores file"),
            ([1.0, 2.0], "not a ColabFold scores file"),
        ],
    )
    def test_a_file_that_is_not_a_scores_file_raises_with_the_reason(
        self, tmp_path, payload, message
    ):
        with pytest.raises(ValueError, match=message):
            parse_colabfold_scores(_scores_file(tmp_path, payload))

    def test_garbage_raises_a_value_error(self, tmp_path):
        path = tmp_path / "scores.json"
        path.write_bytes(GARBAGE)
        with pytest.raises(ValueError, match="scores.json is not a valid JSON file"):
            parse_colabfold_scores(path)


# ---------------------------------------------------------------------------
# ColabFold layout: finding and reading the files of a sample
# ---------------------------------------------------------------------------


def _touch_colabfold(directory, tag, *, name=NAME, kinds=("unrelaxed", "scores")):
    """Empty files of one ColabFold sample, for the tests of file discovery."""
    directory.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        suffix = ".json" if kind == "scores" else ".pdb"
        (directory / f"{name}_{kind}_{tag}{suffix}").write_text("", encoding="utf-8")


class TestColabFoldFindFiles:
    def test_the_files_of_a_sample_are_located(self, tmp_path):
        tag = synth_af2.write_colabfold(tmp_path, NAME, _truth())
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.structure == tmp_path / f"{NAME}_unrelaxed_{tag}.pdb"
        assert files.scores == files.arrays == tmp_path / f"{NAME}_scores_{tag}.json"
        assert files.timing is None and files.extra == {}
        assert files.directory == tmp_path

    def test_the_relaxed_structure_is_preferred_and_the_unrelaxed_one_kept(self, tmp_path):
        tag = synth_af2.write_colabfold(tmp_path, NAME, _truth(), relaxed=True)
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.structure.name == f"{NAME}_relaxed_{tag}.pdb"
        assert files.extra["unrelaxed_structure"].name == f"{NAME}_unrelaxed_{tag}.pdb"

    def test_a_rank_that_was_not_relaxed_keeps_its_unrelaxed_structure(self, tmp_path):
        synth_af2.write_colabfold(tmp_path, NAME, _truth(), relaxed=True)  # rank 1 (--num-relax 1)
        synth_af2.write_colabfold(tmp_path, NAME, _truth(), sample=2)
        parser = AlphaFold2Parser()
        assert parser.find_files(tmp_path, NAME, sample=1).structure.name.startswith(
            f"{NAME}_relaxed_rank_001"
        )
        assert parser.find_files(tmp_path, NAME, sample=2).structure.name.startswith(
            f"{NAME}_unrelaxed_rank_002"
        )

    def test_nothing_is_found_in_an_empty_or_absent_directory(self, tmp_path):
        parser = AlphaFold2Parser()
        assert not parser.find_files(tmp_path, NAME).any_found()
        assert not parser.find_files(tmp_path / "nowhere", NAME).any_found()

    def test_the_job_name_must_match_exactly(self, tmp_path):
        _touch_colabfold(
            tmp_path, "rank_001_alphafold2_multimer_v3_model_1_seed_000", name="cmplx_2"
        )
        parser = AlphaFold2Parser()
        assert not parser.find_files(tmp_path, "cmplx").has_output()
        assert parser.find_files(tmp_path, "cmplx_2").has_output()

    def test_the_job_may_sit_in_a_directory_named_like_it(self, tmp_path):
        tag = synth_af2.write_colabfold(tmp_path / NAME, NAME, _truth())
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.structure == tmp_path / NAME / f"{NAME}_unrelaxed_{tag}.pdb"

    def test_a_stray_json_of_another_kind_is_not_a_scores_file(self, tmp_path):
        _colabfold(tmp_path)
        (tmp_path / f"{NAME}_predicted_aligned_error_v1.json").write_text("{}", encoding="utf-8")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.scores.name.startswith(f"{NAME}_scores_rank_001")

    def test_a_sample_position_beyond_the_last_finds_nothing(self, tmp_path):
        _colabfold(tmp_path)
        parser = AlphaFold2Parser()
        assert not parser.find_files(tmp_path, NAME, sample=2).has_output()
        assert not parser.find_files(tmp_path, NAME, seed_index=2).has_output()
        assert not parser.find_files(tmp_path, NAME, sample=0).has_output()


class TestColabFoldSampleOrder:
    """``seed_index`` is a position in the numeric order of the seeds, ``sample`` one by rank."""

    def _job(self, tmp_path, seeds, ranks):
        for seed in seeds:
            for rank in ranks:
                _touch_colabfold(
                    tmp_path, f"rank_{rank}_alphafold2_multimer_v3_model_1_seed_{seed}"
                )
        return tmp_path

    def test_seeds_are_ordered_by_value_not_as_text(self, tmp_path):
        self._job(tmp_path, seeds=(100, 9, 10), ranks=(1,))
        found = [
            AlphaFold2Parser().find_files(tmp_path, NAME, seed_index=i).structure.name
            for i in (1, 2, 3)
        ]
        assert [name.rsplit("_seed_", 1)[1] for name in found] == ["9.pdb", "10.pdb", "100.pdb"]

    def test_samples_are_ordered_by_rank_value_not_as_text(self, tmp_path):
        self._job(tmp_path, seeds=(0,), ranks=(10, 2, 1))
        parser = AlphaFold2Parser()
        ranks = [
            parser.find_files(tmp_path, NAME, sample=i)
            .structure.name.split("_rank_")[1]
            .split("_")[0]
            for i in (1, 2, 3)
        ]
        assert ranks == ["1", "2", "10"]

    def test_the_sample_is_counted_inside_its_seed(self, tmp_path):
        # global ranks 1, 2 belong to seed 5 and 3, 4 to seed 6
        for rank, seed in ((1, 5), (2, 5), (3, 6), (4, 6)):
            _touch_colabfold(
                tmp_path, f"rank_{rank:03d}_alphafold2_multimer_v3_model_1_seed_{seed:03d}"
            )
        parser = AlphaFold2Parser()
        picked = [
            parser.find_files(tmp_path, NAME, seed_index=s, sample=k).structure.name
            for s, k in ((1, 1), (1, 2), (2, 1), (2, 2))
        ]
        assert [name.split("_rank_")[1][:3] for name in picked] == ["001", "002", "003", "004"]

    def test_a_tag_without_a_seed_counts_as_one_seed(self, tmp_path):
        for rank in (1, 2):
            _touch_colabfold(tmp_path, f"rank_{rank}_model_{rank}")  # an older style of names
        parser = AlphaFold2Parser()
        assert parser.find_files(tmp_path, NAME, sample=2).structure.name.endswith(
            "rank_2_model_2.pdb"
        )
        assert not parser.find_files(tmp_path, NAME, seed_index=2).has_output()

    def test_list_samples_follows_the_order_without_reading_the_files(self, tmp_path):
        self._job(tmp_path, seeds=(100, 9), ranks=(2, 1))
        refs = AlphaFold2Parser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1), (1, 2), (2, 1), (2, 2)]
        assert all(np.isnan(r.ranking_score) for r in refs)

    def test_the_contract_writer_gives_a_model_number_that_is_not_the_rank_order(self, tmp_path):
        # the model numbers 3 and 1 of samples 1 and 2 would swap them if sorted by model
        synth_af2.write_prediction(tmp_path, NAME, _truth(), sample=1)
        synth_af2.write_prediction(tmp_path, NAME, _truth(), sample=2)
        names = [
            AlphaFold2Parser().find_files(tmp_path, NAME, sample=s).structure.name for s in (1, 2)
        ]
        assert "model_3" in names[0] and "model_1" in names[1]


class TestColabFoldParse:
    def test_scalars_arrays_and_chain_values(self, tmp_path):
        truth = _truth()
        record = AlphaFold2Parser().load(_colabfold(tmp_path), NAME)
        record.validate(check_structure=True)
        assert record.model == "af2" and record.name == NAME
        assert (record.ptm, record.iptm) == (0.88, 0.76)
        assert record.avg_plddt == pytest.approx(RESIDUE_PLDDT.mean())
        np.testing.assert_allclose(record.pae, truth.pae)
        assert record.pde is None and record.tokens is None
        assert record.chain_ptm == {"A": 0.88, "B": 0.8}
        assert record.chain_pair_iptm == {"A-B": 0.76}
        assert np.isnan(record.gpde) and np.isnan(record.disorder) and np.isnan(record.has_clash)
        assert np.isnan(record.ranking_score) and record.ranking_score_name == ""
        assert record.timing == {}
        assert record.reasons == []
        assert record.extras["layout"] == "colabfold"
        assert record.extras["colabfold_rank"] == 1
        assert record.extras["plddt_source"] == "colabfold_scores"
        assert record.extras["actifptm"] == 0.74

    def test_plddt_is_one_value_per_residue_repeated_over_its_atoms(self, tmp_path):
        record = AlphaFold2Parser().load(_colabfold(tmp_path), NAME)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        np.testing.assert_allclose(record.extras["plddt_per_residue"], RESIDUE_PLDDT)

    def test_the_relaxed_structure_is_the_one_of_the_record(self, tmp_path):
        tag = synth_af2.write_colabfold(tmp_path, NAME, _truth(), relaxed=True)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.structure_path.name == f"{NAME}_relaxed_{tag}.pdb"
        record.validate(check_structure=True)

    def test_a_relaxed_structure_with_hydrogens_gets_the_residue_value_on_every_atom(
        self, tmp_path
    ):
        # AlphaFold2 relaxation writes hydrogens, and gives each atom, hydrogens included, the
        # pLDDT of its residue in the B-factor; the residues are numbered from 1 in each chain
        complex_ = _truth()
        tag = synth_af2.write_colabfold(tmp_path, NAME, complex_)
        atoms = synth_af2.af2_atoms(complex_)
        pieces = []
        starts = synth_af2.residue_starts(complex_)
        for start, stop in zip(starts, [*starts[1:], complex_.n_atoms]):
            residue = atoms[start:stop]
            hydrogen = synth.struc.array(
                [
                    synth.struc.Atom(
                        residue.coord[0] + 0.9,
                        chain_id=residue.chain_id[0],
                        res_id=residue.res_id[0],
                        res_name=residue.res_name[0],
                        atom_name="H",
                        element="H",
                    )
                ]
            )
            hydrogen.set_annotation("b_factor", residue.b_factor[:1])
            pieces += [residue, hydrogen]
        relaxed = synth.struc.concatenate(pieces)
        synth.write_structure(relaxed, tmp_path / f"{NAME}_relaxed_{tag}.pdb")
        record = AlphaFold2Parser().load(tmp_path, NAME)
        record.validate(check_structure=True)
        assert record.structure_path.name.startswith(f"{NAME}_relaxed_")
        assert record.plddt_per_atom.shape == (21,)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 3))
        assert record.avg_plddt == pytest.approx(RESIDUE_PLDDT.mean())

    def test_the_average_is_over_residues_and_the_expansion_follows_the_atom_counts(self, tmp_path):
        # residues of 2, 1 and 5 atoms in A and 3 and 4 in B, pLDDT 50, 55, 60, 65, 70 per residue
        complex_ = _complex_with_atoms_per_residue({"A": [2, 1, 5], "B": [3, 4]})
        synth_af2.write_colabfold(tmp_path, NAME, complex_)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        record.validate(check_structure=True)
        expected = np.repeat([50.0, 55.0, 60.0, 65.0, 70.0], [2, 1, 5, 3, 4])
        np.testing.assert_allclose(record.plddt_per_atom, expected)
        assert record.avg_plddt == pytest.approx(60.0)  # residue mean; the atom mean is 60.4

    def test_a_monomer_model_gives_reasons_for_what_it_does_not_compute(self, tmp_path):
        synth_af2.write_colabfold(
            tmp_path, NAME, _truth(), scores={"plddt": RESIDUE_PLDDT.tolist()}
        )
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.pae is None and np.isnan(record.ptm) and np.isnan(record.iptm)
        joined = "; ".join(record.reasons)
        assert "PAE and pTM not in" in joined and "ipTM not in" in joined
        assert "only the multimer models" in joined
        record.validate(check_structure=True)

    def test_a_ptm_model_without_iptm_only_lacks_the_ipTM(self, tmp_path):
        scores = {"plddt": RESIDUE_PLDDT.tolist(), "pae": _truth().pae.tolist(), "ptm": 0.7}
        synth_af2.write_colabfold(tmp_path, NAME, _truth(), scores=scores)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.ptm == 0.7 and np.isnan(record.iptm) and record.pae is not None
        assert len(record.reasons) == 1 and record.reasons[0].startswith("ipTM not in")

    def test_the_pae_orientation_is_that_of_alphafold(self, tmp_path):
        record = AlphaFold2Parser().load(_colabfold(tmp_path), NAME)
        assert record.pae[0, 6] == pytest.approx(2.5)  # the transpose has 4.0 here
        assert record.pae[6, 0] == pytest.approx(4.0)

    def test_the_interface_block_is_binder_rows_by_receptor_columns(self, tmp_path):
        # chain B (tokens 4-6) is the binder, chain A (tokens 0-3) the receptor
        record = AlphaFold2Parser().load(_colabfold(tmp_path), NAME)
        summary = summarize_prediction(
            record, binder_chain="B", receptor_chain="A", include_matrices=True
        )
        truth = _truth().pae
        np.testing.assert_allclose(summary["pae_interface"], truth[4:7, 0:4])
        assert not np.allclose(summary["pae_interface"], truth[0:4, 4:7].T)
        expected_mean = (truth[4:7, 0:4].mean() + truth[0:4, 4:7].mean()) / 2
        assert summary["mean_interface_pae"] == pytest.approx(expected_mean)
        # AlphaFold2 has no PDE by design, so nothing is missing and there is no reason
        assert "reason" not in summary

    def test_a_directory_without_output_gives_a_reason(self, tmp_path):
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.structure_path is None and record.plddt_per_atom is None
        assert record.reasons and "no AlphaFold2 or ColabFold output found" in record.reasons[0]

    def test_a_missing_scores_file_falls_back_to_the_b_factor_column(self, tmp_path):
        _colabfold(tmp_path)
        next(tmp_path.glob("*_scores_*.json")).unlink()
        record = AlphaFold2Parser().load(tmp_path, NAME)
        record.validate(check_structure=True)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        assert record.extras["plddt_source"] == "b_factor"
        assert np.isnan(record.ptm) and record.pae is None
        joined = "; ".join(record.reasons)
        assert "no ColabFold scores file" in joined and "B-factor column" in joined

    def test_a_missing_structure_keeps_the_residue_values_and_says_why(self, tmp_path):
        _colabfold(tmp_path)
        next(tmp_path.glob("*_unrelaxed_*.pdb")).unlink()
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.structure_path is None and record.plddt_per_atom is None
        assert record.avg_plddt == pytest.approx(RESIDUE_PLDDT.mean())
        np.testing.assert_allclose(record.extras["plddt_per_residue"], RESIDUE_PLDDT)
        assert "cannot be expanded to atoms" in record.reasons[0]
        record.validate()

    def test_files_of_different_predictions_are_refused(self, tmp_path):
        # the scores describe 7 residues, the structure has 5
        synth_af2.write_colabfold(tmp_path, NAME, _truth())
        short = _complex_with_atoms_per_residue({"A": [2, 2, 2], "B": [2, 2]})
        synth.write_structure(synth_af2.af2_atoms(short), next(tmp_path.glob("*_unrelaxed_*.pdb")))
        with pytest.raises(ValueError, match=r"has 7 pLDDT values but .* has 5 residues"):
            AlphaFold2Parser().load(tmp_path, NAME)

    def test_a_structure_with_a_ligand_is_refused_not_misaligned(self, tmp_path):
        synth_af2.write_colabfold(tmp_path, NAME, _truth())
        path = next(tmp_path.glob("*_unrelaxed_*.pdb"))
        lines = path.read_text(encoding="utf-8").splitlines()
        last_atom = max(i for i, line in enumerate(lines) if line.startswith("ATOM"))
        lines.insert(last_atom + 1, _atom_line(99, " ZN ", " ZN", "C", 1, 50.0, kind="HETATM"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="8 residues"):
            AlphaFold2Parser().load(tmp_path, NAME)

    def test_a_corrupt_scores_file_raises(self, tmp_path):
        _colabfold(tmp_path)
        next(tmp_path.glob("*_scores_*.json")).write_bytes(GARBAGE)
        with pytest.raises(ValueError, match="not a valid JSON file"):
            AlphaFold2Parser().load(tmp_path, NAME)

    def test_a_structure_that_cannot_be_read_keeps_the_scores_and_says_why(self, tmp_path):
        _colabfold(tmp_path)
        next(tmp_path.glob("*_unrelaxed_*.pdb")).write_text("# stub CIF\n", encoding="utf-8")
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.plddt_per_atom is None
        assert record.avg_plddt == pytest.approx(RESIDUE_PLDDT.mean())
        assert (record.ptm, record.iptm) == (0.88, 0.76)
        assert "structure file cannot be read" in record.reasons[0]
        assert "has no atom records" in record.reasons[0]
        assert "cannot be expanded to atoms" in record.reasons[0]
        record.validate()

    def test_a_malformed_atom_record_is_reported_the_same_way(self, tmp_path):
        _colabfold(tmp_path)
        next(tmp_path.glob("*_unrelaxed_*.pdb")).write_text("ATOM      1  CA\n", encoding="utf-8")
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.plddt_per_atom is None and record.pae is not None
        assert "is an atom record of 15 columns" in record.reasons[0]

    def test_a_structure_that_cannot_be_read_gives_no_pLDDT_from_its_b_factors(self, tmp_path):
        _colabfold(tmp_path)
        next(tmp_path.glob("*_scores_*.json")).unlink()
        next(tmp_path.glob("*_unrelaxed_*.pdb")).write_text("# stub CIF\n", encoding="utf-8")
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.plddt_per_atom is None and np.isnan(record.avg_plddt)
        assert any("so there is no pLDDT either" in reason for reason in record.reasons)
        record.validate()

    def test_the_chain_map_renames_the_chains_of_the_atoms_only(self, tmp_path):
        record = AlphaFold2Parser().load(_colabfold(tmp_path), NAME, chain_map={"A": "R", "B": "P"})
        assert set(record.atoms().chain_id) == {"R", "P"}
        assert record.chain_ptm == {"A": 0.88, "B": 0.8}  # keys stay as ColabFold wrote them
        summary = summarize_prediction(record, binder_chain="P", receptor_chain="R")
        assert summary["binder_avg_plddt"] == pytest.approx(76.0)


class TestParsingNeedsNoBiotite:
    """The scores, the PAE and the per-atom pLDDT of a PDB structure are read without biotite."""

    SCRIPT = textwrap.dedent(
        """
        import sys
        sys.modules["biotite"] = None  # any import of biotite now fails
        from binding_metrics.predictors.af2 import AlphaFold2Parser

        record = AlphaFold2Parser().load(sys.argv[1], sys.argv[2])
        assert record.plddt_per_atom is not None and len(record.plddt_per_atom) == 14
        assert record.pae is not None and record.avg_plddt > 0
        assert sys.modules["biotite"] is None
        print("ok")
        """
    )

    def _run(self, directory):
        done = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, str(directory), NAME],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=REPO_ROOT,
            env={**contract.os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-1500:]

    def test_a_colabfold_job(self, tmp_path):
        self._run(_colabfold(tmp_path))


# ---------------------------------------------------------------------------
# The AlphaFold2 result pickle, read without running code from it
# ---------------------------------------------------------------------------


class _Gadget:
    """A pickle that, when loaded by the plain ``pickle``, creates a directory named ``marker``."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        import os

        return (os.mkdir, (self.marker,))


def _pickle_file(tmp_path, payload, *, protocol=4):
    path = tmp_path / "result.pkl"
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=protocol)
    return path


class TestResultPickle:
    @pytest.mark.parametrize("protocol", [3, 4, 5])
    def test_a_result_pickle_is_read_whatever_the_protocol(self, tmp_path, protocol):
        path = _pickle_file(tmp_path, synth_af2.result_pickle(_truth()), protocol=protocol)
        result = read_result_pickle(path)
        assert result["plddt"].dtype == np.float32 and result["plddt"].shape == (7,)
        assert result["distogram"]["logits"].shape == (7, 7, 4)
        assert float(result["ptm"]) == pytest.approx(0.88)

    def test_the_keys_of_the_report_are_parsed(self, tmp_path):
        parsed = parse_result_pickle(_pickle_file(tmp_path, synth_af2.result_pickle(_truth())))
        np.testing.assert_allclose(parsed["plddt_per_residue"], RESIDUE_PLDDT)
        np.testing.assert_allclose(parsed["pae"], _truth().pae)
        assert parsed["pae"].dtype == np.float64
        assert parsed["ptm"] == pytest.approx(0.88) and parsed["iptm"] == pytest.approx(0.76)
        assert parsed["ranking_confidence"] == pytest.approx(0.8 * 0.76 + 0.2 * 0.88)
        assert parsed["chain_ptm"] == {} and parsed["chain_pair_iptm"] == {}

    def test_the_pae_matrix_is_not_transposed(self, tmp_path):
        pae = parse_result_pickle(_pickle_file(tmp_path, synth_af2.result_pickle(_truth())))["pae"]
        assert pae[0, 6] == pytest.approx(2.5) and pae[6, 0] == pytest.approx(4.0)

    def test_a_monomer_result_has_no_pae_or_tm_scores(self, tmp_path):
        payload = {
            "plddt": RESIDUE_PLDDT.astype(np.float32),
            "ranking_confidence": np.float32(80.0),
        }
        parsed = parse_result_pickle(_pickle_file(tmp_path, payload))
        assert parsed["pae"] is None and np.isnan(parsed["ptm"]) and np.isnan(parsed["iptm"])
        assert parsed["ranking_confidence"] == pytest.approx(80.0)

    def test_a_pickle_written_by_numpy_2_loads_in_any_numpy(self, tmp_path):
        # numpy 2 writes numpy._core.multiarray where numpy 1 writes numpy.core.multiarray;
        # protocol 3 names a global in plain text, so the module can be renamed inside the bytes
        path = _pickle_file(tmp_path, synth_af2.result_pickle(_truth()), protocol=3)
        data = path.read_bytes()
        renamed = data.replace(b"cnumpy.core.multiarray\n", b"cnumpy._core.multiarray\n")
        assert renamed != data and b"numpy.core.multiarray" not in renamed
        path.write_bytes(renamed)
        np.testing.assert_allclose(
            parse_result_pickle(path)["plddt_per_residue"], RESIDUE_PLDDT, atol=1e-4
        )

    def test_a_pickle_that_asks_for_anything_but_numpy_is_refused_and_not_run(self, tmp_path):
        marker = tmp_path / "ran"
        path = _pickle_file(tmp_path, {"plddt": np.ones(3), "trap": _Gadget(marker)})
        with pytest.raises(
            ValueError, match=r"asks for \w+\.mkdir, which is not a numpy array type"
        ):
            read_result_pickle(path)
        assert not marker.exists()

    def test_an_object_array_cannot_smuggle_a_callable(self, tmp_path):
        marker = tmp_path / "ran"
        trap = np.empty(1, dtype=object)
        trap[0] = _Gadget(marker)
        with pytest.raises(ValueError, match="not a numpy array type"):
            read_result_pickle(_pickle_file(tmp_path, {"plddt": trap}))
        assert not marker.exists()

    def test_the_message_says_what_to_do_with_a_genuine_file(self, tmp_path):
        with pytest.raises(ValueError, match="layout is TO VERIFY"):
            read_result_pickle(
                _pickle_file(tmp_path, {"when": __import__("datetime").date(2020, 1, 1)})
            )

    @pytest.mark.parametrize("damage", ["garbage", "truncated", "empty"])
    def test_a_damaged_file_raises_a_value_error(self, tmp_path, damage):
        path = _pickle_file(tmp_path, synth_af2.result_pickle(_truth()))
        data = {"garbage": GARBAGE, "truncated": path.read_bytes()[:40], "empty": b""}[damage]
        path.write_bytes(data)
        with pytest.raises(ValueError, match="cannot be read as an AlphaFold2 result pickle"):
            read_result_pickle(path)

    def test_a_pickle_of_something_else_than_a_dictionary_raises(self, tmp_path):
        with pytest.raises(ValueError, match="holds a dictionary, got list"):
            read_result_pickle(_pickle_file(tmp_path, [1, 2, 3]))

    def test_a_result_without_plddt_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no 'plddt' entry"):
            parse_result_pickle(_pickle_file(tmp_path, {"ptm": np.float32(0.5)}))

    def test_a_pae_that_does_not_match_the_plddt_raises(self, tmp_path):
        payload = {"plddt": RESIDUE_PLDDT, "predicted_aligned_error": np.ones((5, 5))}
        with pytest.raises(ValueError, match="5 by 5 but pLDDT has 7 residues"):
            parse_result_pickle(_pickle_file(tmp_path, payload))

    def test_a_scalar_that_is_not_a_number_raises_a_value_error(self, tmp_path):
        payload = {"plddt": RESIDUE_PLDDT, "ptm": {"a": 1}}
        with pytest.raises(ValueError, match="'ptm' is not a number"):
            parse_result_pickle(_pickle_file(tmp_path, payload))

    def test_a_scalar_with_several_values_raises(self, tmp_path):
        payload = {"plddt": RESIDUE_PLDDT, "ptm": np.array([0.5, 0.6])}
        with pytest.raises(ValueError, match="'ptm' has 2 values"):
            parse_result_pickle(_pickle_file(tmp_path, payload))


class TestRankingDebug:
    def test_a_multimer_ranking(self, tmp_path):
        path = tmp_path / "ranking_debug.json"
        path.write_text(
            json.dumps(
                {"iptm+ptm": {"m_pred_0": 0.7, "m_pred_1": 0.8}, "order": ["m_pred_1", "m_pred_0"]}
            ),
            encoding="utf-8",
        )
        parsed = parse_ranking_debug(path)
        assert parsed == {
            "name": "iptm+ptm",
            "scores": {"m_pred_0": 0.7, "m_pred_1": 0.8},
            "order": ["m_pred_1", "m_pred_0"],
        }

    def test_a_monomer_ranking_by_plddt(self, tmp_path):
        path = tmp_path / "ranking_debug.json"
        path.write_text(
            json.dumps({"plddts": {"model_1": 88.5}, "order": ["model_1"]}), encoding="utf-8"
        )
        assert parse_ranking_debug(path)["name"] == "plddts"

    def test_an_order_that_is_not_a_list_raises_a_value_error(self, tmp_path):
        path = tmp_path / "ranking_debug.json"
        path.write_text(json.dumps({"iptm+ptm": {"m_pred_0": 0.7}, "order": 3.5}), encoding="utf-8")
        with pytest.raises(ValueError, match="'order' must be a list, got float"):
            parse_ranking_debug(path)

    @pytest.mark.parametrize("content", ['{"order": []}', "[1]", '{"iptm+ptm": [1]}'])
    def test_anything_else_raises(self, tmp_path, content):
        path = tmp_path / "ranking_debug.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="not an AlphaFold2 ranking_debug.json"):
            parse_ranking_debug(path)


# ---------------------------------------------------------------------------
# The AlphaFold2 v2.3.2 output directory
# ---------------------------------------------------------------------------


class TestAlphaFoldFindFiles:
    def test_the_files_of_a_prediction_are_located(self, tmp_path):
        prediction_id = _alphafold(tmp_path)
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert prediction_id == "model_1_multimer_v3_pred_0"
        assert files.structure == tmp_path / f"unrelaxed_{prediction_id}.pdb"
        assert files.arrays == tmp_path / f"result_{prediction_id}.pkl"
        assert files.scores == tmp_path / "ranking_debug.json"
        assert files.timing == tmp_path / "timings.json"
        assert files.extra == {}

    def test_the_relaxed_structure_is_preferred(self, tmp_path):
        prediction_id = _alphafold(tmp_path, relaxed=True)
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.structure.name == f"relaxed_{prediction_id}.pdb"
        assert files.extra["unrelaxed_structure"].name == f"unrelaxed_{prediction_id}.pdb"

    def test_the_ranked_copies_are_not_used(self, tmp_path):
        _alphafold(tmp_path)
        (tmp_path / "ranked_0.pdb").write_text("", encoding="utf-8")
        assert AlphaFold2Parser().find_files(tmp_path, NAME).structure.name.startswith("unrelaxed_")

    def test_the_directory_of_the_fasta_name_is_searched(self, tmp_path):
        _alphafold(tmp_path / NAME)
        assert AlphaFold2Parser().find_files(tmp_path, NAME).arrays.parent == tmp_path / NAME

    def test_the_main_json_files_of_a_prediction_are_found_by_its_id(self, tmp_path):
        prediction_id = _alphafold(tmp_path, main_json=True)
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.extra["pae"].name == f"pae_{prediction_id}.json"
        assert files.extra["confidence"].name == f"confidence_{prediction_id}.json"

    def test_nothing_is_found_where_no_prediction_was_written(self, tmp_path):
        (tmp_path / "ranking_debug.json").write_text("{}", encoding="utf-8")
        assert not AlphaFold2Parser().find_files(tmp_path, NAME).has_output()


class TestAlphaFoldSampleOrder:
    """``seed_index`` is the position of ``pred_{i}`` by value, ``sample`` the model number."""

    def _grid(self, tmp_path):
        # pred 9 and 10 (text order would put 10 first), models 2 and 1 written out of order
        for pred in (10, 9):
            for model in ("model_2_multimer_v3", "model_1_multimer_v3"):
                _alphafold(tmp_path, pred=pred, model=model)
        return tmp_path

    def test_seeds_by_value_and_samples_by_model_number(self, tmp_path):
        self._grid(tmp_path)
        parser = AlphaFold2Parser()
        picked = [
            parser.find_files(tmp_path, NAME, seed_index=s, sample=k).structure.name
            for s, k in ((1, 1), (1, 2), (2, 1), (2, 2))
        ]
        assert picked == [
            "unrelaxed_model_1_multimer_v3_pred_9.pdb",
            "unrelaxed_model_2_multimer_v3_pred_9.pdb",
            "unrelaxed_model_1_multimer_v3_pred_10.pdb",
            "unrelaxed_model_2_multimer_v3_pred_10.pdb",
        ]

    def test_list_samples_reads_the_ranking_scores_and_no_pickle(self, tmp_path):
        self._grid(tmp_path)
        for path in tmp_path.glob("result_*.pkl"):
            path.write_bytes(GARBAGE)  # listing must not open them
        refs = AlphaFold2Parser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1), (1, 2), (2, 1), (2, 2)]
        assert all(r.ranking_score == pytest.approx(0.8 * 0.76 + 0.2 * 0.88) for r in refs)

    def test_a_position_beyond_the_last_finds_nothing(self, tmp_path):
        _alphafold(tmp_path)
        parser = AlphaFold2Parser()
        assert not parser.find_files(tmp_path, NAME, sample=2).has_output()
        assert not parser.find_files(tmp_path, NAME, seed_index=2).has_output()


class TestAlphaFoldParse:
    def test_scalars_arrays_ranking_and_timing(self, tmp_path):
        truth = _truth()
        _alphafold(tmp_path)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        record.validate(check_structure=True)
        assert record.model == "af2"
        assert (record.ptm, record.iptm) == (pytest.approx(0.88), pytest.approx(0.76))
        np.testing.assert_allclose(record.pae, truth.pae, atol=1e-6)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2), atol=1e-4)
        assert record.avg_plddt == pytest.approx(RESIDUE_PLDDT.mean())
        assert record.pde is None and record.tokens is None
        assert record.ranking_score == pytest.approx(0.8 * 0.76 + 0.2 * 0.88)
        assert record.ranking_score_name == "iptm+ptm"
        assert record.timing == {"features": 1.5, "predict_model_1_multimer_v3_pred_0": 7.25}
        assert record.extras["layout"] == "alphafold"
        assert record.extras["prediction_id"] == "model_1_multimer_v3_pred_0"
        assert record.extras["plddt_source"] == "result_pickle"
        assert record.reasons == []

    def test_the_pae_orientation_is_that_of_alphafold(self, tmp_path):
        _alphafold(tmp_path)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.pae[0, 6] == pytest.approx(2.5) and record.pae[6, 0] == pytest.approx(4.0)
        summary = summarize_prediction(
            record, binder_chain="B", receptor_chain="A", include_matrices=True
        )
        np.testing.assert_allclose(summary["pae_interface"], _truth().pae[4:7, 0:4], atol=1e-6)

    def test_the_ranking_of_the_right_prediction_is_used(self, tmp_path):
        _alphafold(tmp_path, pred=0, ranking=0.55)
        _alphafold(tmp_path, pred=1, ranking=0.91)
        parser = AlphaFold2Parser()
        assert parser.load(tmp_path, NAME, seed_index=1).ranking_score == pytest.approx(0.55)
        assert parser.load(tmp_path, NAME, seed_index=2).ranking_score == pytest.approx(0.91)

    def test_a_monomer_ranking_by_plddt_is_named_as_alphafold_names_it(self, tmp_path):
        prediction_id = _alphafold(tmp_path, model="model_1")
        (tmp_path / "ranking_debug.json").write_text(
            json.dumps({"plddts": {prediction_id: 84.5}, "order": [prediction_id]}),
            encoding="utf-8",
        )
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.ranking_score == 84.5 and record.ranking_score_name == "plddts"

    def test_the_ranking_of_the_pickle_is_the_fallback(self, tmp_path):
        _alphafold(tmp_path)
        (tmp_path / "ranking_debug.json").unlink()
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.ranking_score == pytest.approx(0.8 * 0.76 + 0.2 * 0.88)
        assert record.ranking_score_name == "ranking_confidence"

    def test_a_prediction_missing_from_the_ranking_file_has_no_ranking_score(self, tmp_path):
        _alphafold(tmp_path)
        (tmp_path / "ranking_debug.json").write_text(
            json.dumps({"iptm+ptm": {"other_pred_0": 0.5}, "order": ["other_pred_0"]}),
            encoding="utf-8",
        )
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.ranking_score == pytest.approx(0.8 * 0.76 + 0.2 * 0.88)  # from the pickle

    def test_a_monomer_pickle_gives_reasons_for_the_missing_scores(self, tmp_path):
        payload = {"plddt": RESIDUE_PLDDT.astype(np.float32)}
        _alphafold(tmp_path, result=payload)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        assert record.pae is None and np.isnan(record.ptm)
        assert any("PAE and pTM not in result_" in reason for reason in record.reasons)
        record.validate(check_structure=True)

    def test_without_the_pickle_the_main_json_files_give_plddt_and_pae(self, tmp_path):
        _alphafold(tmp_path, main_json=True, write_result=False)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        record.validate(check_structure=True)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        np.testing.assert_allclose(record.pae, np.round(_truth().pae, 1))
        assert record.pae[0, 6] == pytest.approx(2.5) and record.pae[6, 0] == pytest.approx(4.0)
        assert record.extras["plddt_source"] == "confidence_json"
        assert np.isnan(record.ptm) and "no pae_*.json" not in "; ".join(record.reasons)
        assert any(
            reason.startswith("pTM and ipTM are not in confidence_") for reason in record.reasons
        )

    def test_the_pickle_wins_over_the_main_json_files(self, tmp_path):
        _alphafold(tmp_path, main_json=True)
        assert AlphaFold2Parser().load(tmp_path, NAME).extras["plddt_source"] == "result_pickle"

    def test_without_any_confidence_file_the_b_factor_column_is_read(self, tmp_path):
        _alphafold(tmp_path, write_result=False)
        record = AlphaFold2Parser().load(tmp_path, NAME)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        assert record.extras["plddt_source"] == "b_factor" and record.pae is None
        joined = "; ".join(record.reasons)
        assert "no AlphaFold2 result pickle" in joined and "B-factor column" in joined
        record.validate(check_structure=True)

    def test_a_pickle_of_another_prediction_is_refused(self, tmp_path):
        short = _complex_with_atoms_per_residue({"A": [2, 2, 2], "B": [2, 2]})
        _alphafold(tmp_path)
        with open(next(tmp_path.glob("result_*.pkl")), "wb") as handle:
            pickle.dump(synth_af2.result_pickle(short), handle, protocol=4)
        with pytest.raises(ValueError, match=r"has 5 pLDDT values but .* has 7 residues"):
            AlphaFold2Parser().load(tmp_path, NAME)

    def test_a_corrupt_pickle_raises(self, tmp_path):
        _alphafold(tmp_path)
        next(tmp_path.glob("result_*.pkl")).write_bytes(GARBAGE)
        with pytest.raises(ValueError, match="cannot be read as an AlphaFold2 result pickle"):
            AlphaFold2Parser().load(tmp_path, NAME)

    def test_a_corrupt_ranking_or_timing_file_raises(self, tmp_path):
        _alphafold(tmp_path)
        (tmp_path / "ranking_debug.json").write_bytes(GARBAGE)
        with pytest.raises(ValueError, match="not a valid JSON file"):
            AlphaFold2Parser().load(tmp_path, NAME)
        (tmp_path / "ranking_debug.json").unlink()
        _alphafold(tmp_path)  # rewrites a valid ranking file
        (tmp_path / "timings.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="must hold a JSON object of run times"):
            AlphaFold2Parser().load(tmp_path, NAME)

    def test_the_pickle_layout_is_read_without_biotite(self, tmp_path):
        _alphafold(tmp_path)
        TestParsingNeedsNoBiotite()._run(tmp_path)


class TestMainJsonFiles:
    def test_the_confidence_file_is_a_per_residue_plddt(self, tmp_path):
        path = tmp_path / "confidence_x.json"
        path.write_text(
            json.dumps(
                {
                    "residueNumber": [1, 2],
                    "confidenceScore": [91.5, 40.25],
                    "confidenceCategory": ["V", "L"],
                }
            ),
            encoding="utf-8",
        )
        np.testing.assert_allclose(parse_confidence_json(path), [91.5, 40.25])

    def test_the_pae_file_is_a_list_holding_one_object(self, tmp_path):
        path = tmp_path / "pae_x.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "predicted_aligned_error": [[0.0, 3.5], [2.0, 0.0]],
                        "max_predicted_aligned_error": 31.75,
                    }
                ]
            ),
            encoding="utf-8",
        )
        pae = parse_pae_json(path, 2)
        assert pae[0, 1] == 3.5 and pae[1, 0] == 2.0

    def test_the_colabfold_form_of_a_dict_is_accepted(self, tmp_path):
        path = tmp_path / "pae_x.json"
        path.write_text(
            json.dumps({"predicted_aligned_error": [[0.0, 3.5], [2.0, 0.0]]}), encoding="utf-8"
        )
        assert parse_pae_json(path, 2).shape == (2, 2)

    @pytest.mark.parametrize("content", ["{}", "[[1.0]]", '{"distance": [1.0]}'])
    def test_a_file_of_another_form_raises(self, tmp_path, content):
        path = tmp_path / "pae_x.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="not an AlphaFold2 PAE file"):
            parse_pae_json(path, 1)
        path.write_text('{"residueNumber": [1]}', encoding="utf-8")
        with pytest.raises(ValueError, match="not an AlphaFold2 confidence file"):
            parse_confidence_json(path)

    def test_a_pae_of_the_wrong_size_raises(self, tmp_path):
        path = tmp_path / "pae_x.json"
        path.write_text(json.dumps([{"predicted_aligned_error": [[0.0]]}]), encoding="utf-8")
        with pytest.raises(ValueError, match="1 by 1 but pLDDT has 3 residues"):
            parse_pae_json(path, 3)


# ---------------------------------------------------------------------------
# A structure alone in a directory
# ---------------------------------------------------------------------------


class TestBareStructureInADirectory:
    def test_a_structure_named_like_the_prediction_is_read_from_its_b_factors(self, tmp_path):
        path = synth_af2.write_bare(tmp_path, NAME, _truth())
        files = AlphaFold2Parser().find_files(tmp_path, NAME)
        assert files.structure == path and files.scores is None and files.arrays is None
        record = AlphaFold2Parser().load(tmp_path, NAME)
        record.validate(check_structure=True)
        np.testing.assert_allclose(record.plddt_per_atom, np.repeat(RESIDUE_PLDDT, 2))
        assert record.extras["layout"] == "bare"

    def test_there_is_one_sample(self, tmp_path):
        synth_af2.write_bare(tmp_path, NAME, _truth())
        parser = AlphaFold2Parser()
        assert [(r.seed_index, r.sample) for r in parser.list_samples(tmp_path, NAME)] == [(1, 1)]
        assert not parser.find_files(tmp_path, NAME, sample=2).has_output()

    def test_a_colabfold_job_wins_over_a_stray_structure_of_the_same_name(self, tmp_path):
        _colabfold(tmp_path)
        synth_af2.write_bare(tmp_path, NAME, _truth())
        assert AlphaFold2Parser().load(tmp_path, NAME).extras["layout"] == "colabfold"

    @pytest.mark.parametrize("suffix", [".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".ent"])
    def test_the_structure_suffixes(self, tmp_path, suffix):
        path = tmp_path / f"{NAME}{suffix}"
        path.write_text("", encoding="utf-8")
        assert AlphaFold2Parser().find_files(tmp_path, NAME).structure == path


# ---------------------------------------------------------------------------
# The registered adapter, and the metrics that read its records
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_af2_is_registered_and_lazy(self):
        spec = PARSERS["af2"]
        assert spec.import_path == "binding_metrics.predictors.af2:AlphaFold2Parser"
        assert isinstance(get_parser("af2"), AlphaFold2Parser)
        assert (spec.display_name, spec.family) == ("AlphaFold2 / ColabFold", "af2")
        assert AlphaFold2Parser.name == "af2"

    def test_no_capabilities_are_declared(self):
        assert AlphaFold2Parser.capabilities is None

    def test_a_writer_module_exists_for_the_contract_tests(self):
        assert callable(contract.writer_module("af2").write_prediction)


class TestPredictionMetricsOnAlphaFoldOutput:
    def test_compute_prediction_metrics_reads_a_colabfold_job(self, tmp_path):
        truth = _truth()
        result = compute_prediction_metrics(
            _colabfold(tmp_path), "af2", NAME, binder_chain="B", receptor_chain="A"
        )
        assert result["model"] == "af2"
        assert (result["ptm"], result["iptm"]) == (0.88, 0.76)
        assert result["avg_plddt"] == pytest.approx(RESIDUE_PLDDT.mean())
        assert result["binder_avg_plddt"] == pytest.approx(76.0)
        expected = (truth.pae[4:7, 0:4].mean() + truth.pae[0:4, 4:7].mean()) / 2
        assert result["mean_interface_pae"] == pytest.approx(expected)
        assert result["max_pae"] == pytest.approx(truth.pae.max())
        assert np.isnan(result["gpde"]) and result["pde"] is None
        assert result["chain_ptm"] == {"A": 0.88, "B": 0.8}
        assert "reason" not in result

    def test_the_chains_of_the_user_are_reached_through_the_chain_map(self, tmp_path):
        result = compute_prediction_metrics(
            _colabfold(tmp_path),
            "af2",
            NAME,
            binder_chain="P",
            receptor_chain="R",
            chain_map={"A": "R", "B": "P"},
        )
        assert result["binder_avg_plddt"] == pytest.approx(76.0)
        assert np.isfinite(result["mean_interface_pae"])

    def test_the_official_layout_gives_the_same_scores(self, tmp_path):
        _alphafold(tmp_path)
        result = compute_prediction_metrics(
            tmp_path, "af2", NAME, binder_chain="B", receptor_chain="A"
        )
        assert result["iptm"] == pytest.approx(0.76)
        assert result["binder_avg_plddt"] == pytest.approx(76.0, abs=1e-3)
        assert result["sample_ranking_score"] == pytest.approx(0.8 * 0.76 + 0.2 * 0.88)


class TestBareComplexAsAdversary:
    """A complex with pLDDT only in its B-factor column can be the EvoBind adversary."""

    def test_the_binder_plddt_of_the_b_factor_column_divides_the_score(self, tmp_path):
        shifted = (0.0, 3.0, 4.0)  # 5 angstrom
        design_atoms, _ = _complex_atoms()
        design = synth.write_structure(design_atoms, tmp_path / "design.pdb")
        atoms, plddt = _complex_atoms(binder_shift=shifted)
        bare = synth_af2.write_bare(tmp_path, "adversary", _as_synthetic_complex(atoms, plddt))
        record = load_bfactor_record(bare)
        result = compute_evobind_adversarial_from_records(design, record, "B", "A")
        assert result["adversary_model"] == "af2"
        assert result["afm_mean_plddt_binder"] == pytest.approx(80.0)
        assert result["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)
        expected = result["afm_mean_if_dist"] * (100.0 / 80.0) * 5.0
        assert result["evobind_adversarial_score"] == pytest.approx(expected, rel=1e-2)
        assert "reason" not in result

    def test_a_file_without_plddt_gives_the_geometry_and_a_reason(self, tmp_path):
        design_atoms, _ = _complex_atoms()
        design = synth.write_structure(design_atoms, tmp_path / "design.pdb")
        atoms, _ = _complex_atoms()
        atoms.set_annotation("b_factor", np.zeros(atoms.array_length()))
        record = load_bfactor_record(synth.write_structure(atoms, tmp_path / "zero.pdb"))
        result = compute_evobind_adversarial_from_records(design, record, "B", "A")
        assert result["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
        assert result["evobind_adversarial_score"] is None
        assert result["reason"].startswith("adversary has no per-atom pLDDT")
        assert "all zero" in result["reason"]
