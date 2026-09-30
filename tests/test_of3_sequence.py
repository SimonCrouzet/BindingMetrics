"""Sequence and ``non_canonical_residues`` of the OpenFold3 query (issues #75, #76).

A residue that OpenFold3 cannot take must stop the run before anything is written or
started; D-amino acids and modified residues keep their chemistry through their CCD code.
The structures are built here with gemmi; OpenFold3 is not needed, and what OpenFold3 does
with the resulting query is not tested.
"""

import json
import logging
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from binding_metrics.metrics import _openfold_cli, _openfold_run, openfold  # noqa: E402
from binding_metrics.metrics._openfold_run import (  # noqa: E402
    UnmappableResidueError,
    _BatchSample,
    _extract_query_chain,
    _extract_sequence_from_structure,
)
from tests.test_of3_synth import _residue, _structure, _write  # noqa: E402

_HOH = ("HOH", (("O", "O"),))
_ZN = ("ZN", (("ZN", "Zn"),))
_ACE = ("ACE", (("C", "C"), ("O", "O"), ("CH3", "C")))
_NME = ("NME", (("N", "N"), ("C", "C")))
_RECEPTOR = ["ALA", "GLY", "SER", "LYS"]


class TestMapping:
    def test_standard_residues_are_plain_letters(self):
        st = _structure({"B": ["ALA", "ARG", "TRP", "TYR", "VAL", "GLY", "PRO"]})
        assert _extract_query_chain(st, "B") == ("ARWYVGP", {})

    def test_d_amino_acids_keep_their_ccd_code(self):
        st = _structure({"B": ["DAL", "GLY", "DPN", "MED", "DAR", "ALA"]})
        seq, non_canonical = _extract_query_chain(st, "B")
        assert seq == "AGFMRA"
        assert non_canonical == {1: "DAL", 3: "DPN", 4: "MED", 5: "DAR"}

    def test_the_sequence_never_holds_lowercase_letters(self):
        """gemmi writes 'a' for DAL and 's' for SEP; OpenFold3 turns such letters into UNK."""
        st = _structure({"B": ["DAL", "SEP", "MLE", "HYP", "ALA"]})
        seq, _ = _extract_query_chain(st, "B")
        assert seq == seq.upper()

    def test_modified_residues_go_to_non_canonical_residues(self):
        st = _structure({"B": ["ALA", "SEP", "MLE", "HYP", "MSE"]})
        seq, non_canonical = _extract_query_chain(st, "B")
        assert seq == "ASLPM"
        assert non_canonical == {2: "SEP", 3: "MLE", 4: "HYP", 5: "MSE"}

    def test_a_component_missing_from_gemmi_but_in_the_ccd_is_expressed(self):
        """0EH (D-peptide linking) and IAM are in the CCD, not in gemmi's small table."""
        pytest.importorskip("biotite")
        seq, non_canonical = _extract_query_chain(_structure({"B": ["ALA", "0EH", "IAM"]}), "B")
        assert len(seq) == 3
        assert non_canonical == {2: "0EH", 3: "IAM"}
        assert seq[0] == "A"

    @pytest.mark.parametrize(
        "name, letter",
        [
            ("HID", "H"), ("HIE", "H"), ("HIP", "H"), ("HIN", "H"), ("CYX", "C"),
            ("CYM", "C"), ("ASH", "D"), ("GLH", "E"), ("LYN", "K"),
        ],
    )  # fmt: skip
    def test_protonation_variants_take_their_parent_letter(self, name, letter):
        """HIP is doubly protonated histidine here; the CCD uses HIP for phosphohistidine."""
        seq, non_canonical = _extract_query_chain(_structure({"B": ["ALA", name, "ALA"]}), "B")
        assert seq == f"A{letter}A"
        assert non_canonical == {}

    @pytest.mark.parametrize("name, letter", [("ASPL", "D"), ("GLUL", "E"), ("LYSL", "K")])
    def test_lactam_templates_take_their_parent_letter(self, name, letter):
        seq, non_canonical = _extract_query_chain(_structure({"B": [name, "ALA"]}), "B")
        assert (seq, non_canonical) == (f"{letter}A", {})

    def test_toolkit_n_methyl_names_map_to_their_ccd_components(self):
        """NMG and NMA are template names of core.nonstandard; the CCD codes are SAR and MAA."""
        st = _structure({"B": ["NMG", "NMA", "MVA", "MLE", "SAR", "MAA"]})
        seq, non_canonical = _extract_query_chain(st, "B")
        assert seq == "GAVLGA"
        assert non_canonical == {1: "SAR", 2: "MAA", 3: "MVA", 4: "MLE", 5: "SAR", 6: "MAA"}

    def test_unknown_placeholder_and_selenocysteine_are_sequence_letters(self):
        seq, non_canonical = _extract_query_chain(_structure({"B": ["ALA", "UNK", "SEC"]}), "B")
        assert (seq, non_canonical) == ("AXU", {})

    def test_positions_count_residues_of_the_sequence_only(self):
        """Waters, ions and caps are left out, so they do not shift the positions."""
        st = _structure({"B": [_ACE, "ALA", _HOH, "DAL", _ZN, "GLY", _NME]})
        seq, non_canonical = _extract_query_chain(st, "B")
        assert seq == "AAG"
        assert non_canonical == {2: "DAL"}

    def test_caps_are_reported_at_info_level(self, caplog):
        with caplog.at_level(logging.INFO, logger=_openfold_run.logger.name):
            _extract_query_chain(_structure({"B": [_ACE, "ALA", _NME]}), "B")
        assert "ACE" in caplog.text and "NME" in caplog.text

    def test_the_existing_helper_still_returns_a_string(self):
        st = _structure({"B": ["ALA", "DAL", "CYX"]})
        assert _extract_sequence_from_structure(st, "B") == "AAC"

    def test_chain_errors_are_unchanged(self):
        st = _structure({"B": ["ALA"], "W": [_HOH]})
        with pytest.raises(ValueError, match="'Z' not found"):
            _extract_query_chain(st, "Z")
        with pytest.raises(ValueError, match="no amino acid"):
            _extract_query_chain(st, "W")


class TestGemmiFallback:
    """Without biotite's CCD the smaller table of gemmi decides."""

    @pytest.fixture(autouse=True)
    def _no_ccd(self, monkeypatch):
        pytest.importorskip("biotite")
        from biotite.structure import info

        def _unavailable(*args, **kwargs):
            raise OSError("no CCD")

        monkeypatch.setattr(info, "get_from_ccd", _unavailable)

    def test_components_of_gemmis_table_are_still_expressed(self):
        seq, non_canonical = _extract_query_chain(_structure({"B": ["ALA", "SEP", "MLE"]}), "B")
        assert (seq, non_canonical) == ("ASL", {2: "SEP", 3: "MLE"})

    def test_other_components_are_refused_rather_than_guessed(self):
        with pytest.raises(UnmappableResidueError, match="0EH 2"):
            _extract_query_chain(_structure({"B": ["ALA", "0EH"]}), "B")


class TestBundledExamples:
    """The corrected sequences of the bundled structures (the old ones are in the changelog)."""

    _DATA = Path(__file__).parent.parent / "data"

    @pytest.mark.parametrize(
        "file_name, chain, sequence, non_canonical",
        [
            (
                "example_ncaa_cyclosporin_1CWA.cif",
                "C",
                "ALLVTAGLVLA",
                {1: "DAL", 2: "MLE", 3: "MLE", 4: "MVA", 5: "BMT", 6: "ABA", 7: "SAR",
                 8: "MLE", 10: "MLE"},
            ),
            ("example_phospho_1QJB.pdb", "Q", "ARSHSYPA", {5: "SEP"}),
            ("example_staple_3V3B.pdb", "C", "TFXNLWRLLL", {3: "0EH", 10: "MK8"}),
        ],
    )  # fmt: skip
    def test_sequence_and_non_canonical_residues(self, file_name, chain, sequence, non_canonical):
        pytest.importorskip("biotite")
        st = gemmi.read_structure(str(self._DATA / file_name))
        assert _extract_query_chain(st, chain) == (sequence, non_canonical)

    def test_standard_chains_of_the_examples_have_no_entries(self):
        st = gemmi.read_structure(str(self._DATA / "example_ncaa_cyclosporin_1CWA.cif"))
        seq, non_canonical = _extract_query_chain(st, "A")
        assert non_canonical == {} and len(seq) == 165


class TestUnmappableResidues:
    def test_a_ligand_like_name_with_a_backbone_is_refused(self):
        """ZZZ is a CCD non-polymer component: no amino acid, so OpenFold3 cannot place it."""
        st = _structure({"B": ["ALA", "GLY", "ZZZ", "ALA"]})
        with pytest.raises(UnmappableResidueError) as info:
            _extract_query_chain(st, "B")
        message = str(info.value)
        assert "chain 'B'" in message
        assert "ZZZ 3" in message
        assert "OpenFold3 cannot take" in message
        assert "Remove or replace" in message
        assert "--metrics" in message

    def test_the_error_is_a_value_error_and_lists_every_residue(self):
        st = _structure({"B": ["QQQ", "ALA", "XYZ9", "GLY"]})
        with pytest.raises(ValueError) as info:
            _extract_query_chain(st, "B")
        assert isinstance(info.value, UnmappableResidueError)
        assert info.value.details == [("", "B", ["QQQ 1", "XYZ9 3"])]

    def test_an_insertion_code_is_part_of_the_residue_number(self):
        st = gemmi.Structure()
        model = gemmi.Model("1")
        chain = gemmi.Chain("B")
        chain.add_residue(_residue("ALA", 1))
        chain.add_residue(_residue("XYZ9", 2, icode="A"))
        model.add_chain(chain)
        st.add_model(model)
        with pytest.raises(UnmappableResidueError, match="XYZ9 2A"):
            _extract_query_chain(st, "B")

    def test_hetero_groups_without_a_backbone_are_not_refused(self):
        """A ligand or ion in the chain is left out, as before, and does not raise."""
        st = _structure({"B": ["ALA", ("SO4", (("S", "S"), ("O1", "O"))), _ZN, "GLY"]})
        assert _extract_query_chain(st, "B") == ("AG", {})

    def test_x_opt_out_keeps_the_length_and_logs_a_warning(self, caplog):
        st = _structure({"B": ["ALA", "ZZZ", "DAL", "XYZ9", "GLY"]})
        with caplog.at_level(logging.WARNING, logger=_openfold_run.logger.name):
            seq, non_canonical = _extract_query_chain(st, "B", on_unmappable_residue="x")
        assert seq == "AXAXG"
        assert non_canonical == {3: "DAL"}
        assert "ZZZ 2" in caplog.text and "XYZ9 4" in caplog.text

    def test_the_option_is_validated(self):
        with pytest.raises(ValueError, match="on_unmappable_residue"):
            _extract_query_chain(_structure({"B": ["ALA"]}), "B", on_unmappable_residue="skip")

    def test_the_sequence_helper_forwards_the_option(self):
        st = _structure({"B": ["ALA", "ZZZ"]})
        assert _extract_sequence_from_structure(st, "B", on_unmappable_residue="x") == "AX"
        with pytest.raises(UnmappableResidueError):
            _extract_sequence_from_structure(st, "B")


def _chain_dicts(query_json: Path, name: str = "q") -> list[dict]:
    return json.loads(query_json.read_text(encoding="utf-8"))["queries"][name]["chains"]


class TestQueryJson:
    @pytest.fixture
    def d_peptide(self, tmp_path):
        return _write(tmp_path, {"A": _RECEPTOR, "B": ["DAL", "ALA", "SEP", "GLY"]})

    def test_scoring_query_lists_non_canonical_residues_with_string_keys(self, tmp_path, d_peptide):
        path = openfold.prepare_scoring_query(d_peptide, "A", "B", "q", tmp_path / "out")
        receptor, binder = _chain_dicts(path)
        assert binder["sequence"] == "AASG"
        assert binder["non_canonical_residues"] == {"1": "DAL", "3": "SEP"}
        assert "non_canonical_residues" not in receptor

    def test_refolding_query_lists_them_too(self, tmp_path, d_peptide):
        path = openfold.prepare_refolding_query(d_peptide, "A", "B", "q", tmp_path / "out")
        _, binder = _chain_dicts(path)
        assert binder["non_canonical_residues"] == {"1": "DAL", "3": "SEP"}

    def test_a_receptor_with_a_modified_residue_is_expressed_as_well(self, tmp_path):
        complex_path = _write(tmp_path, {"A": ["ALA", "SEP", "GLY"], "B": ["ALA", "GLY"]})
        path = openfold.prepare_scoring_query(complex_path, "A", "B", "q", tmp_path / "out")
        receptor, binder = _chain_dicts(path)
        assert receptor["non_canonical_residues"] == {"2": "SEP"}
        assert "non_canonical_residues" not in binder

    def test_batched_queries_list_them_per_chain(self, tmp_path, d_peptide):
        samples = [_BatchSample("q", d_peptide, "A", "B")]
        scoring = openfold.prepare_batched_scoring_queries(samples, tmp_path / "s")
        refolding = openfold.prepare_batched_refolding_queries(samples, tmp_path / "r")
        for path in (scoring, refolding):
            assert _chain_dicts(path)[1]["non_canonical_residues"] == {"1": "DAL", "3": "SEP"}

    def test_the_template_cif_holds_the_parent_sequence(self, tmp_path):
        complex_path = _write(tmp_path, {"A": ["ALA", "DAL", "GLY"], "B": ["ALA", "GLY"]})
        openfold.prepare_scoring_query(complex_path, "A", "B", "q", tmp_path / "out")
        doc = gemmi.cif.read(str(tmp_path / "out" / "templates" / "receptor.cif"))
        assert doc.sole_block().find_value("_entity_poly.pdbx_seq_one_letter_code_can") == "AAG"

    def test_queries_of_standard_residues_have_no_new_key(self):
        """The bundled p53-MDM2 complex gives the same chain dicts as before."""
        complex_path = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
        import tempfile

        with tempfile.TemporaryDirectory() as out:
            path = openfold.prepare_scoring_query(complex_path, "A", "B", "q", out)
            receptor, binder = _chain_dicts(path)
        assert set(receptor) == {
            "molecule_type", "chain_ids", "sequence", "template_alignment_file_path",
        }  # fmt: skip
        assert set(binder) == set(receptor)
        assert binder["sequence"] == "ETFSDLWKLLPEN"


class TestFailFast:
    """An unmappable residue must stop everything before a file is written or a process starts."""

    @pytest.fixture
    def bad_complex(self, tmp_path):
        return _write(tmp_path, {"A": _RECEPTOR, "B": ["ALA", "ZZZ", "GLY"]}, "bad.pdb")

    @pytest.fixture
    def stub_subprocess(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            openfold.subprocess, "run", lambda *a, **k: calls.append((a, k)) or pytest.fail("ran")
        )
        monkeypatch.setattr(openfold.subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
        return calls

    @pytest.mark.parametrize("function", ["prepare_scoring_query", "prepare_refolding_query"])
    def test_prepare_raises_before_writing_anything(self, tmp_path, bad_complex, function):
        out = tmp_path / "out"
        with pytest.raises(UnmappableResidueError, match="ZZZ 2"):
            getattr(openfold, function)(bad_complex, "A", "B", "q", out)
        assert not out.exists()

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_the_run_wrappers_do_not_start_a_process(
        self, tmp_path, bad_complex, stub_subprocess, runner
    ):
        out = tmp_path / "out"
        with pytest.raises(UnmappableResidueError, match="ZZZ 2"):
            getattr(openfold, runner)(bad_complex, "A", "B", "q", out, conda_env="of3")
        assert stub_subprocess == []
        assert not out.exists()

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_batched_wrapper_does_not_start_a_process(
        self, tmp_path, bad_complex, stub_subprocess, mode
    ):
        out = tmp_path / "out"
        samples = [_BatchSample("bad", bad_complex, "A", "B")]
        with pytest.raises(UnmappableResidueError, match="bad: chain 'B': ZZZ 2"):
            openfold.run_openfold_batched(samples, out, mode=mode, conda_env="of3")
        assert stub_subprocess == []
        assert not out.exists()

    def test_a_batch_names_every_affected_sample_and_writes_nothing(self, tmp_path, bad_complex):
        good = _write(tmp_path, {"A": _RECEPTOR, "B": ["DAL", "GLY"]}, "good.pdb")
        # PDB files hold three-character residue names, so these are real CCD ligand codes
        worse = _write(tmp_path, {"A": ["QQQ", "ALA"], "B": ["XYZ", "GLY"]}, "worse.pdb")
        samples = [
            _BatchSample("bad", bad_complex, "A", "B"),
            _BatchSample("ok", good, "A", "B"),
            _BatchSample("worse", worse, "A", "B"),
        ]
        out = tmp_path / "out"
        with pytest.raises(UnmappableResidueError) as info:
            openfold.prepare_batched_scoring_queries(samples, out)
        assert info.value.details == [
            ("bad", "B", ["ZZZ 2"]),
            ("worse", "A", ["QQQ 1"]),
            ("worse", "B", ["XYZ 1"]),
        ]
        assert "ok:" not in str(info.value)
        assert not out.exists()

    def test_the_opt_out_writes_the_query_with_an_x(self, tmp_path, bad_complex, caplog):
        with caplog.at_level(logging.WARNING, logger=_openfold_run.logger.name):
            path = openfold.prepare_scoring_query(
                bad_complex, "A", "B", "q", tmp_path / "out", on_unmappable_residue="x"
            )
        assert _chain_dicts(path)[1]["sequence"] == "AXG"
        assert "ZZZ 2" in caplog.text

    def test_the_batched_opt_out_writes_the_queries(self, tmp_path, bad_complex):
        path = openfold.prepare_batched_refolding_queries(
            [_BatchSample("q", bad_complex, "A", "B")], tmp_path / "out", on_unmappable_residue="x"
        )
        assert _chain_dicts(path)[1]["sequence"] == "AXG"


class TestCommandLine:
    def _argv(self, command, complex_path, out, *extra):
        return [
            "prog", command, "--complex", str(complex_path), "--receptor-chain", "A",
            "--binder-chain", "B", "--query-name", "q", "--output-dir", str(out), *extra,
        ]  # fmt: skip

    @pytest.mark.parametrize("command", ["prepare-query", "prepare-scoring-query"])
    def test_prepare_commands_fail_fast_and_accept_the_opt_out(
        self, tmp_path, monkeypatch, command
    ):
        complex_path = _write(tmp_path, {"A": _RECEPTOR, "B": ["ALA", "ZZZ", "GLY"]})
        out = tmp_path / "out"
        monkeypatch.setattr("sys.argv", self._argv(command, complex_path, out))
        with pytest.raises(UnmappableResidueError, match="ZZZ 2"):
            openfold.main()
        assert not out.exists()

        monkeypatch.setattr(
            "sys.argv", self._argv(command, complex_path, out, "--on-unmappable-residue", "x")
        )
        openfold.main()
        assert (out / "q_query.json").exists()
        assert _chain_dicts(out / "q_query.json")[1]["sequence"] == "AXG"

    @pytest.mark.parametrize(
        "command, target",
        [("refold", "run_openfold_refolding"), ("score", "run_openfold_scoring")],
    )
    def test_run_commands_pass_the_option_only_when_it_is_not_the_default(
        self, tmp_path, monkeypatch, command, target
    ):
        seen = []
        monkeypatch.setattr(openfold, target, lambda **kw: seen.append(kw) or tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(_openfold_cli, "_print_metrics", lambda *a, **kw: None)
        for extra in ([], ["--on-unmappable-residue", "x"]):
            monkeypatch.setattr("sys.argv", self._argv(command, "c.cif", tmp_path, *extra))
            openfold.main()
        assert "on_unmappable_residue" not in seen[0]
        assert seen[1]["on_unmappable_residue"] == "x"

    def test_the_flag_only_accepts_the_two_values(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            self._argv("prepare-query", "c.cif", tmp_path, "--on-unmappable-residue", "skip"),
        )
        with pytest.raises(SystemExit):
            openfold.main()
        assert "invalid choice" in capsys.readouterr().err
