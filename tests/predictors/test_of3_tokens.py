"""The OpenFold3 token layout: one token per standard residue, one per atom of any other residue.

OpenFold3 tokenises a residue of the standard set as one token and every other residue (a
modified residue, a ligand) as one token per heavy atom, so the matrices of a binder with a
modified residue are larger than its residue count (1CWA: 240 tokens for 176 residues). The
adapter derives the layout from the predicted structure in ``OpenFold3Parser.complete``; these
tests build synthetic outputs with a modified residue and check the interface values against
numbers worked out by hand, the three paths that call ``complete``, and every case in which the
layout cannot be shown to fit the files and is refused with a reason.
"""

import dataclasses
import sys
import warnings
from unittest import mock

import numpy as np
import pytest

from binding_metrics.metrics import openfold
from binding_metrics.metrics.prediction import compute_prediction_metrics, summarize_prediction
from binding_metrics.predictors.of3 import (
    STANDARD_RESIDUE_NAMES,
    OpenFold3Parser,
    token_layout,
)
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import PredictionRequest, PredictionStore
from tests.predictors import contract, synth, synth_of3

NAME = contract.NAME
CHAINS = {"binder_chain": "B", "receptor_chain": "A"}


def _write(tmp_path, complex_):
    synth_of3.write_prediction(tmp_path, NAME, complex_)
    return complex_


def _load(tmp_path, **kwargs):
    return OpenFold3Parser().load(tmp_path, NAME, **kwargs)


@pytest.fixture
def truth(tmp_path):
    return _write(tmp_path, synth_of3.complex_with_a_modified_residue())


# The interface values of complex_with_a_modified_residue, by hand (see its docstring).
# Binder tokens 3-9, receptor tokens 0-2; pae[i, j] = 1 + 0.5 i + 0.25 j.
#   binder rows x receptor columns: mean i = 6, mean j = 1  ->  1 + 3 + 0.25  = 4.25
#   receptor rows x binder columns: mean i = 1, mean j = 6  ->  1 + 0.5 + 1.5 = 3.0
#   mean of the two blocks 3.625; largest value 1 + 0.5 * 9 + 0.25 * 2 = 6.0
# pde[i, j] = 0.5 + 0.25 i + 0.125 j, binder rows x receptor columns only:
#   mean 0.5 + 1.5 + 0.125 = 2.125; largest 0.5 + 0.25 * 9 + 0.125 * 2 = 3.0
HAND_MEAN_PAE, HAND_MAX_PAE = 3.625, 6.0
HAND_MEAN_PDE, HAND_MAX_PDE = 2.125, 3.0


class TestTheRule:
    def test_the_standard_set_is_the_31_names_of_the_tokenizer(self):
        assert len(STANDARD_RESIDUE_NAMES) == 31
        amino_acids = (
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"
        )
        assert set(amino_acids.split()) | {"UNK"} <= STANDARD_RESIDUE_NAMES
        assert {"A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN"} <= STANDARD_RESIDUE_NAMES
        for modified in ("MLE", "DAL", "SEP", "MSE", "SAR", "LIG"):
            assert modified not in STANDARD_RESIDUE_NAMES

    def test_a_modified_residue_is_one_token_per_atom_and_the_rest_one_per_residue(
        self, truth, tmp_path
    ):
        layout = token_layout(_load(tmp_path))
        assert len(layout) == 10  # 3 + (1 + 5 + 1) tokens for 6 residues
        assert list(layout.chain_id) == ["A"] * 3 + ["B"] * 7
        np.testing.assert_array_equal(layout.res_id, [1, 2, 3, 1, 2, 2, 2, 2, 2, 3])
        np.testing.assert_array_equal(layout.is_atom_token, [False] * 4 + [True] * 5 + [False])
        assert layout.token_ranges() == {"A": (0, 3), "B": (3, 10)}
        assert layout.problems() == []

    def test_a_token_is_its_c_alpha_or_the_atom_itself(self, truth, tmp_path):
        record = _load(tmp_path)
        layout = token_layout(record)
        # atoms: A1 CA CB, A2 CA CB, A3 CA CB, B1 CA CB, MLE N CA C O CB, B3 CA CB
        np.testing.assert_array_equal(layout.atom_index, [0, 2, 4, 6, 8, 9, 10, 11, 12, 13])
        atoms = record.atoms()
        assert list(atoms.atom_name[layout.atom_index]) == (
            ["CA"] * 4 + ["N", "CA", "C", "O", "CB"] + ["CA"]
        )

    def test_a_standard_complex_has_one_token_per_residue(self, tmp_path):
        _write(tmp_path, synth.synthetic_complex())
        layout = token_layout(_load(tmp_path))
        assert len(layout) == synth.N_TOKENS
        assert not layout.is_atom_token.any()
        assert layout.token_ranges() == {"A": (0, 4), "B": (4, 7)}
        np.testing.assert_array_equal(layout.atom_index, np.arange(0, 14, 2))  # the CA atoms

    def test_the_hetero_flag_of_the_file_is_not_the_rule(self, tmp_path):
        # a real prediction has UNK with the flag set, and it is one token
        standard = synth.synthetic_complex()
        atoms = standard.atoms.copy()
        atoms.hetero[:] = True
        _write(tmp_path, dataclasses.replace(standard, atoms=atoms))
        assert len(token_layout(_load(tmp_path))) == synth.N_TOKENS

    @pytest.mark.parametrize("name", ["UNK", "A", "DA", "GLY"])
    def test_every_name_of_the_standard_set_is_one_token(self, tmp_path, name):
        standard = synth.synthetic_complex()
        atoms = standard.atoms.copy()
        atoms.res_name[:] = name
        _write(tmp_path, dataclasses.replace(standard, atoms=atoms))
        layout = token_layout(_load(tmp_path))
        assert len(layout) == synth.N_TOKENS and not layout.is_atom_token.any()

    def test_a_record_of_another_model_is_refused(self, truth, tmp_path):
        record = _load(tmp_path)
        record.model = "protenix"
        with pytest.raises(ValueError, match="needs an OpenFold3 record"):
            token_layout(record)

    def test_the_layout_keeps_the_model_chain_ids_when_the_user_renames_them(self, truth, tmp_path):
        record = _load(tmp_path, chain_map={"A": "R", "B": "P"})
        layout = token_layout(record)
        assert list(layout.chain_id) == ["A"] * 3 + ["B"] * 7
        assert set(record.atoms().chain_id) == {"R", "P"}

    @pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".pdb"])
    @pytest.mark.parametrize("fmt", ["json", "npz"])
    def test_every_file_format_gives_the_same_layout(self, tmp_path, monkeypatch, suffix, fmt):
        monkeypatch.setattr(synth_of3, "STRUCTURE_SUFFIX", suffix)
        monkeypatch.setattr(synth_of3, "CONFIDENCE_FORMAT", fmt)
        _write(tmp_path, synth_of3.complex_with_a_modified_residue())
        record = OpenFold3Parser().complete(_load(tmp_path))
        assert record.tokens.token_ranges() == {"A": (0, 3), "B": (3, 10)}
        assert int(record.tokens.is_atom_token.sum()) == 5


def _cwa_shaped_complex():
    """A receptor of 165 residues and the binder of 1CWA with its real atom counts.

    The binder is DAL MLE MLE MVA BMT ABA SAR MLE VAL MLE ALA: nine modified residues with
    5, 9, 9, 8, 13, 6, 5, 9 and 9 heavy atoms (73 in all) and VAL and ALA as one token each.
    That is 165 + 73 + 2 = 240 tokens for 176 residues, the shape of the real 0.5.0 output.
    The receptor residues have a CA only and the coordinates are placeholders.
    """
    binder = [
        ("DAL", 5),
        ("MLE", 9),
        ("MLE", 9),
        ("MVA", 8),
        ("BMT", 13),
        ("ABA", 6),
        ("SAR", 5),
        ("MLE", 9),
        ("VAL", 7),
        ("MLE", 9),
        ("ALA", 5),
    ]
    atoms = []
    for number in range(1, 166):
        atoms.append(synth_of3._atom("A", number, "ALA", "CA", 3.8 * number, 0.0))
    for number, (name, n_atoms) in enumerate(binder, start=1):
        atoms.append(synth_of3._atom("C", number, name, "CA", 3.8 * number, 6.0))
        for k in range(n_atoms - 1):
            atoms.append(synth_of3._atom("C", number, name, f"X{k}", 3.8 * number, 6.0 + k))
    array = synth.struc.array(atoms)
    plddt = np.full(len(atoms), 90.0)
    array.set_annotation("b_factor", plddt.copy())
    n_tokens = 240
    i = np.arange(n_tokens)[:, None]
    j = np.arange(n_tokens)[None, :]
    base = synth.synthetic_complex()
    return dataclasses.replace(
        base,
        atoms=array,
        plddt_per_atom=plddt,
        pae=1.0 + 0.5 * i + 0.25 * j,
        pde=0.5 + 0.25 * i + 0.125 * j,
        chain_ptm={"A": 0.97, "C": 0.75},
        chain_pair_iptm={"A-C": 0.92, "C-A": 0.92},
    )


class TestTheShapeOfARealRun:
    """1CWA: the matrices have 240 rows and the structure 176 residues."""

    def test_the_tokens_are_165_for_the_receptor_and_75_for_the_binder(self, tmp_path):
        _write(tmp_path, _cwa_shaped_complex())
        record = OpenFold3Parser().complete(_load(tmp_path))
        assert record.reasons == []
        assert record.tokens.token_ranges() == {"A": (0, 165), "C": (165, 240)}
        assert int(record.tokens.is_atom_token.sum()) == 73
        assert record.n_atoms == 165 + 85

    def test_the_interface_was_refused_before_and_is_cut_from_the_layout_now(self, tmp_path):
        _write(tmp_path, _cwa_shaped_complex())
        chains = {"binder_chain": "C", "receptor_chain": "A"}
        refused = _load(tmp_path)
        with pytest.warns(UserWarning, match="interface PAE skipped"):
            before = summarize_prediction(refused, **chains)
        assert np.isnan(before["mean_interface_pae"])
        assert "240 tokens" in before["reason"] and "176 residues" in before["reason"]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            after = compute_prediction_metrics(tmp_path, "of3", NAME, **chains)
        # binder rows 165-239 (mean 202) x receptor columns 0-164 (mean 82):
        #   1 + 0.5 * 202 + 0.25 * 82 = 122.5; the other block 1 + 0.5 * 82 + 0.25 * 202 = 92.5
        assert after["mean_interface_pae"] == pytest.approx((122.5 + 92.5) / 2)
        # the largest value is binder row 239, receptor column 164: 1 + 119.5 + 41 = 161.5
        assert after["max_interface_pae"] == pytest.approx(161.5)
        assert "reason" not in after


class TestCompletion:
    def test_load_alone_leaves_the_interface_refused_with_the_sizes(self, truth, tmp_path):
        record = _load(tmp_path)
        assert record.tokens is None
        with pytest.warns(UserWarning, match="interface PAE skipped"):
            result = summarize_prediction(record, **CHAINS)
        assert np.isnan(result["mean_interface_pae"])
        assert "10 tokens" in result["reason"] and "6 residues" in result["reason"]

    def test_complete_attaches_the_layout_and_returns_the_same_record(self, truth, tmp_path):
        record = _load(tmp_path)
        assert OpenFold3Parser().complete(record) is record
        assert len(record.tokens) == 10 and record.reasons == []
        record.validate(check_structure=True)

    def test_complete_is_idempotent(self, truth, tmp_path):
        record = _load(tmp_path)
        parser = OpenFold3Parser()
        parser.complete(record)
        tokens, reasons = record.tokens, list(record.reasons)
        parser.complete(record)
        assert record.tokens is tokens and record.reasons == reasons

    def test_the_interface_values_are_the_hand_computed_ones(self, truth, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compute_prediction_metrics(
                tmp_path, "of3", NAME, include_matrices=True, **CHAINS
            )
        assert "reason" not in result
        assert np.isfinite(result["mean_interface_pae"])
        assert result["mean_interface_pae"] == pytest.approx(HAND_MEAN_PAE)
        assert result["max_interface_pae"] == pytest.approx(HAND_MAX_PAE)
        assert result["mean_interface_pde"] == pytest.approx(HAND_MEAN_PDE)
        assert result["max_interface_pde"] == pytest.approx(HAND_MAX_PDE)
        assert result["pae_interface"].shape == (7, 3)
        np.testing.assert_allclose(result["pae_interface"], truth.pae[3:10, 0:3])
        np.testing.assert_allclose(result["pde_interface"], truth.pde[3:10, 0:3], atol=0.01)

    def test_the_chains_of_the_user_reach_the_layout(self, truth, tmp_path):
        result = compute_prediction_metrics(
            tmp_path,
            "of3",
            NAME,
            binder_chain="P",
            receptor_chain="R",
            chain_map={"A": "R", "B": "P"},
        )
        assert "reason" not in result
        assert result["mean_interface_pae"] == pytest.approx(HAND_MEAN_PAE)

    def test_compute_openfold_metrics_gets_the_layout_too(self, truth, tmp_path):
        result = openfold.compute_openfold_metrics(tmp_path, NAME, **CHAINS)
        assert "reason" not in result
        assert result["mean_interface_pae"] == pytest.approx(HAND_MEAN_PAE)
        assert result["mean_interface_pde"] == pytest.approx(HAND_MEAN_PDE)

    def test_a_standard_complex_keeps_its_numbers(self, tmp_path):
        _write(tmp_path, synth.synthetic_complex())
        record = OpenFold3Parser().complete(_load(tmp_path))
        assert record.reasons == [] and len(record.tokens) == synth.N_TOKENS
        legacy = openfold.compute_openfold_metrics(tmp_path, NAME, **CHAINS)
        assert legacy["mean_interface_pae"] == pytest.approx(3.4375)
        assert "reason" not in legacy


class TestThePipelinePath:
    """``PredictionSession.record`` is the one place the pipeline and the CLI read from."""

    @pytest.fixture
    def session(self, tmp_path):
        return PredictionSession(
            PredictionStore(tmp_path / "store"), {}, {"of3": OpenFold3Parser()}
        )

    def _request(self, session, tmp_path):
        request = PredictionRequest("of3", NAME, sequences={"A": "AAA"})
        session.store.adopt(request, tmp_path)
        return request

    def test_the_session_hands_out_a_record_with_the_layout(self, truth, tmp_path, session):
        record = session.record(self._request(session, tmp_path))
        assert len(record.tokens) == 10
        result = summarize_prediction(record, include_matrices=True, **CHAINS)
        assert "reason" not in result
        np.testing.assert_allclose(result["pae_interface"], truth.pae[3:10, 0:3])

    def test_the_command_line_step_reports_the_interface_values(self, truth, tmp_path, session):
        from binding_metrics.cli.prediction import _analyse

        request = self._request(session, tmp_path)
        model_file = next(tmp_path.rglob("*_model.cif"))
        block, _ = _analyse(
            session,
            request,
            input_path=model_file,
            chain_map=None,
            reference_path=None,
            adopted=True,
            **CHAINS,
        )
        assert block["mean_interface_pae"] == pytest.approx(HAND_MEAN_PAE)
        assert "PAE matrix has" not in block.get("reason", "")
        assert block["model"] == "of3"


def _refused(tmp_path, complex_, **load_kwargs):
    """The record after ``complete`` for a fixture that the layout must refuse."""
    _write(tmp_path, complex_)
    record = _load(tmp_path, **load_kwargs)
    OpenFold3Parser().complete(record)
    assert record.tokens is None
    assert record.reasons[-1].startswith("token layout not built")
    return record


class TestWhenTheLayoutCannotBeShown:
    """A reason instead of a guess: ``tokens`` stays None and the old refusal applies."""

    def test_a_matrix_of_another_size_than_the_rule_gives(self, tmp_path):
        base = synth_of3.complex_with_a_modified_residue()
        i = np.arange(11)[:, None]
        j = np.arange(11)[None, :]
        wrong = dataclasses.replace(base, pae=1.0 + 0.5 * i + 0.25 * j, pde=0.5 + 0.0 * i + j)
        record = _refused(tmp_path, wrong)
        assert "gives 10 tokens" in record.reasons[-1]
        assert "(5 residues of the standard set" in record.reasons[-1]
        assert "5 atoms of other residues" in record.reasons[-1]
        assert "PAE matrix has 11 rows" in record.reasons[-1]
        with pytest.warns(UserWarning, match="interface PAE skipped"):
            result = summarize_prediction(record, **CHAINS)
        assert np.isnan(result["mean_interface_pae"])

    def test_matrices_of_one_row_per_residue_are_left_to_the_residue_count(self, tmp_path):
        # a model (or a stub) that gives a modified residue one row, as the older code assumed:
        # 6 rows for 6 residues. The sizes agree, so nothing is refused and nothing is added
        base = synth_of3.complex_with_a_modified_residue()
        i = np.arange(6)[:, None]
        j = np.arange(6)[None, :]
        per_residue = dataclasses.replace(
            base, pae=1.0 + 0.5 * i + 0.25 * j, pde=0.5 + 0.25 * i + 0.125 * j
        )
        _write(tmp_path, per_residue)
        record = OpenFold3Parser().complete(_load(tmp_path))
        assert record.tokens is None and record.reasons == []
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compute_prediction_metrics(tmp_path, "of3", NAME, **CHAINS)
        # binder rows 3-5 (mean 4) x receptor columns 0-2 (mean 1): 1 + 2 + 0.25 = 3.25;
        # receptor rows (mean 1) x binder columns (mean 4): 1 + 0.5 + 1 = 2.5; mean 2.875
        assert result["mean_interface_pae"] == pytest.approx(2.875)
        assert "reason" not in result
        with pytest.raises(ValueError, match="one row per residue"):
            token_layout(record)

    def test_a_ligand_named_like_a_standard_residue_is_refused_not_miscounted(self, tmp_path):
        # chain C is a free amino acid of 4 atoms, one token each in OpenFold3 (14 tokens) but
        # named ALA, so the rule reads it as one residue token: it counts too few and refuses
        base = synth.synthetic_complex()
        free_amino_acid = synth.struc.array(
            [
                synth_of3._atom("C", 1, "ALA", name, 20.0, k, hetero=True)
                for k, name in enumerate(["N", "CA", "C", "O"])
            ]
        )
        free_amino_acid.set_annotation("b_factor", np.full(4, 80.0))
        array = base.atoms + free_amino_acid
        plddt = np.asarray(array.b_factor)
        i = np.arange(11)[:, None]
        j = np.arange(11)[None, :]
        ligand = dataclasses.replace(
            base,
            atoms=array,
            plddt_per_atom=plddt,
            pae=1.0 + 0.5 * i + 0.25 * j,
            pde=0.5 + 0.25 * i + 0.125 * j,
            chain_ptm={"A": 0.9, "B": 0.8, "C": 0.7},
            chain_pair_iptm={"A-B": 0.76},
        )
        record = _refused(tmp_path, ligand)
        assert "gives 8 tokens" in record.reasons[-1] and "11 rows" in record.reasons[-1]

    def test_a_pLDDT_of_another_length_does_not_stop_the_layout_and_is_reported_by_the_summary(
        self, tmp_path
    ):
        # the layout rests on the structure and the matrices; the pLDDT is checked where it is used
        base = synth_of3.complex_with_a_modified_residue()
        _write(tmp_path, dataclasses.replace(base, plddt_per_atom=base.plddt_per_atom[:-1]))
        record = OpenFold3Parser().complete(_load(tmp_path))
        assert record.tokens is not None and record.reasons == []
        with pytest.warns(UserWarning, match="per-residue binder pLDDT skipped"):
            result = summarize_prediction(record, **CHAINS)
        assert (
            "binder pLDDT: plddt_per_atom length (14) != atom count in structure (15)"
            in (result["reason"])
        )
        assert result["mean_interface_pae"] == pytest.approx(HAND_MEAN_PAE)

    def test_a_chain_that_the_confidences_do_not_have(self, tmp_path):
        base = synth_of3.complex_with_a_modified_residue()
        record = _refused(tmp_path, dataclasses.replace(base, chain_ptm={"A": 0.9, "Z": 0.8}))
        assert "chain_ptm names the chains ['A', 'Z']" in record.reasons[-1]
        assert "the structure has ['A', 'B']" in record.reasons[-1]

    def test_a_chain_pair_that_the_structure_does_not_have(self, tmp_path):
        base = synth_of3.complex_with_a_modified_residue()
        record = _refused(tmp_path, dataclasses.replace(base, chain_pair_iptm={"A-Z": 0.7}))
        assert "chain_pair_iptm names the chains ['A', 'Z']" in record.reasons[-1]

    def test_a_chain_split_into_two_runs_cannot_be_sliced(self, tmp_path):
        base = synth.synthetic_complex()
        # the last residue of chain A written after chain B: one token per residue, 7 tokens
        reordered = base.atoms[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 6, 7]]
        plddt = np.asarray(reordered.b_factor)
        record = _refused(
            tmp_path, dataclasses.replace(base, atoms=reordered, plddt_per_atom=plddt)
        )
        assert "the tokens of chain 'A' are not contiguous" in record.reasons[-1]

    def test_a_one_token_residue_without_a_c_alpha(self, tmp_path):
        base = synth.synthetic_complex()
        atoms = base.atoms.copy()
        atoms.atom_name[2] = "CG"  # residue 2 of chain A is left with CG and CB
        record = _refused(tmp_path, dataclasses.replace(base, atoms=atoms))
        assert "ALA 2 of chain A has none of the atoms ['CA', \"C1'\"]" in record.reasons[-1]

    def test_the_reason_is_given_once(self, tmp_path):
        base = synth_of3.complex_with_a_modified_residue()
        record = _refused(tmp_path, dataclasses.replace(base, chain_ptm={"A": 0.9, "Z": 0.8}))
        reasons = list(record.reasons)
        OpenFold3Parser().complete(record)
        assert record.reasons == reasons and record.tokens is None

    def test_an_unreadable_structure_is_left_to_the_analysis_that_needs_it(self, truth, tmp_path):
        record = _load(tmp_path)
        record.structure_path.write_text("# stub CIF\n", encoding="utf-8")
        assert OpenFold3Parser().complete(record) is record
        assert record.tokens is None and record.reasons == []
        with pytest.warns(UserWarning, match="structural analysis failed"):
            result = summarize_prediction(record, **CHAINS)
        assert result["reason"].startswith("structural analysis failed")

    def test_a_chain_map_for_a_chain_that_is_not_there_is_left_to_the_analysis(
        self, truth, tmp_path
    ):
        record = _load(tmp_path, chain_map={"Q": "R"})
        OpenFold3Parser().complete(record)
        assert record.tokens is None and record.reasons == []

    def test_the_numbers_that_the_model_gives_its_chains_are_accepted_as_their_names(
        self, tmp_path
    ):
        # the model numbers the chains 1, 2 in sorted order; 0.5.0 maps them to chain IDs
        base = synth_of3.complex_with_a_modified_residue()
        numbered = dataclasses.replace(
            base, chain_ptm={"1": 0.9, "2": 0.8}, chain_pair_iptm={"1-2": 0.7}
        )
        _write(tmp_path, numbered)
        record = OpenFold3Parser().complete(_load(tmp_path))
        assert record.tokens is not None and record.reasons == []


class TestWhatIsLeftAlone:
    def test_a_missing_biotite_leaves_the_record_alone(self, truth, tmp_path):
        record = _load(tmp_path)
        blocked = {
            name: None
            for name in (
                "biotite",
                "biotite.structure",
                "biotite.structure.io",
                "biotite.structure.io.pdb",
                "biotite.structure.io.pdbx",
            )
        }
        with mock.patch.dict(sys.modules, blocked):
            OpenFold3Parser().complete(record)
        assert record.tokens is None and record.reasons == []

    def test_a_record_without_a_structure_is_returned_as_it_is(self, truth, tmp_path):
        record = _load(tmp_path)
        record.structure_path = None
        assert OpenFold3Parser().complete(record).tokens is None
        assert record.reasons == []

    def test_a_record_without_matrices_is_returned_as_it_is(self, truth, tmp_path):
        record = _load(tmp_path)
        record.pae = record.pde = None
        assert OpenFold3Parser().complete(record).tokens is None
        assert record.reasons == []

    def test_a_record_with_a_layout_is_returned_as_it_is(self, truth, tmp_path):
        record = _load(tmp_path)
        layout = token_layout(record)
        record.tokens = layout
        assert OpenFold3Parser().complete(record).tokens is layout

    def test_a_record_of_another_model_is_returned_as_it_is(self, truth, tmp_path):
        record = _load(tmp_path)
        record.model = "protenix"
        assert OpenFold3Parser().complete(record).tokens is None

    def test_a_directory_with_no_output_gives_a_record_that_is_returned_as_it_is(self, tmp_path):
        record = _load(tmp_path / "nowhere")
        assert OpenFold3Parser().complete(record) is record and record.tokens is None
