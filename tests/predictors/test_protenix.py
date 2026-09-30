"""The Protenix adapter: file discovery, parsing, and the conventions of its output.

Every fixture is written at test time by ``tests/predictors/synth_protenix.py`` in the layout
that ``predictors/protenix.py`` documents; nothing here is real Protenix output.
"""

import json

import numpy as np
import pytest

from binding_metrics.metrics.prediction import summarize_prediction
from binding_metrics.predictors.protenix import ProtenixParser
from binding_metrics.predictors.registry import PARSERS, ParserSpec, register_parser
from tests.predictors import contract, synth, synth_protenix

NAME = contract.NAME

SPEC = ParserSpec(
    name="protenix",
    import_path="binding_metrics.predictors.protenix:ProtenixParser",
    display_name="Protenix",
    family="af3",
)


@pytest.fixture
def registered():
    """The adapter is in the registry for one test (the contract checks look it up there)."""
    saved = dict(PARSERS)
    register_parser(SPEC, replace=True)
    yield
    PARSERS.clear()
    PARSERS.update(saved)


def _write(tmp_path, complex_=None, **kwargs):
    truth = complex_ if complex_ is not None else synth.synthetic_complex()
    synth_protenix.write_prediction(tmp_path, NAME, truth, **kwargs)
    return truth


def _predictions(tmp_path, seed=9):
    return tmp_path / NAME / f"seed_{seed}" / "predictions"


def _load(tmp_path, **kwargs):
    return ProtenixParser().load(tmp_path, NAME, **kwargs)


def _edit(path, edit):
    """Load a JSON file, apply ``edit`` to its object, and write it back."""
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _summary(tmp_path, rank=0, seed=9):
    return _predictions(tmp_path, seed) / f"{NAME}_summary_confidence_sample_{rank}.json"


def _full_data(tmp_path, rank=0, seed=9):
    return _predictions(tmp_path, seed) / f"{NAME}_full_data_sample_{rank}.json"


# ---------------------------------------------------------------------------
# The contract every adapter meets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("check", contract.CHECKS, ids=lambda check: check.__name__)
def test_the_adapter_meets_the_contract(registered, check, tmp_path):
    check("protenix", tmp_path)


# ---------------------------------------------------------------------------
# Finding the files
# ---------------------------------------------------------------------------


class TestFindFiles:
    def test_locates_the_files_of_one_sample(self, tmp_path):
        _write(tmp_path)
        files = ProtenixParser().find_files(tmp_path, NAME)
        predictions = _predictions(tmp_path)
        assert files.structure == predictions / f"{NAME}_sample_0.cif"
        assert files.scores == predictions / f"{NAME}_summary_confidence_sample_0.json"
        assert files.arrays == predictions / f"{NAME}_full_data_sample_0.json"
        assert files.timing is None  # Protenix writes no timing file
        assert files.directory == tmp_path

    def test_nothing_is_found_in_an_empty_or_absent_directory(self, tmp_path):
        parser = ProtenixParser()
        assert not parser.find_files(tmp_path, NAME).any_found()
        assert not parser.find_files(tmp_path / "nowhere", NAME).any_found()

    def test_the_sample_number_is_one_based_and_the_file_rank_starts_at_zero(self, tmp_path):
        _write(tmp_path, sample=1)
        _write(tmp_path, sample=2)
        parser = ProtenixParser()
        assert parser.find_files(tmp_path, NAME, sample=1).structure.name == f"{NAME}_sample_0.cif"
        assert parser.find_files(tmp_path, NAME, sample=2).structure.name == f"{NAME}_sample_1.cif"
        assert not parser.find_files(tmp_path, NAME, sample=0).has_output()
        assert not parser.find_files(tmp_path, NAME, sample=3).has_output()

    def test_seed_directories_are_ordered_by_seed_value_not_by_name(self, tmp_path):
        for seed_index in (1, 2, 3):  # seed values 9, 10 and 11
            _write(tmp_path, seed_index=seed_index)
        parser = ProtenixParser()
        seeds = [
            parser.find_files(tmp_path, NAME, seed_index=k).structure.parent.parent.name
            for k in (1, 2, 3)
        ]
        assert seeds == ["seed_9", "seed_10", "seed_11"]
        assert not parser.find_files(tmp_path, NAME, seed_index=4).has_output()
        assert not parser.find_files(tmp_path, NAME, seed_index=0).has_output()

    def test_a_seed_index_is_a_position_and_not_a_seed_value(self, tmp_path):
        _write(tmp_path)  # only seed_9 exists
        assert not ProtenixParser().find_files(tmp_path, NAME, seed_index=9).has_output()

    def test_the_layout_of_the_documentation_is_not_read(self, tmp_path):
        # docs/infer_json_format.md shows <name>/<seed>/<name>_<seed>_sample_0.cif, which the
        # code does not write
        wrong = tmp_path / NAME / "seed_9"
        wrong.mkdir(parents=True)
        (wrong / f"{NAME}_seed_9_sample_0.cif").write_text("data_x\n", encoding="utf-8")
        assert not ProtenixParser().find_files(tmp_path, NAME).has_output()

    def test_a_structure_without_unresolved_atoms_is_not_taken_for_the_sample(self, tmp_path):
        _write(tmp_path)
        other = _predictions(tmp_path) / f"{NAME}_sample_0_wounresol.cif"
        other.write_text("data_x\n", encoding="utf-8")
        files = ProtenixParser().find_files(tmp_path, NAME)
        assert files.structure.name == f"{NAME}_sample_0.cif"


# ---------------------------------------------------------------------------
# Scales, orientation and the values of the summary
# ---------------------------------------------------------------------------


class TestScales:
    def test_the_file_holds_0_1_and_the_record_0_100(self, tmp_path):
        truth = _write(tmp_path)
        raw = json.loads(_full_data(tmp_path).read_text(encoding="utf-8"))
        assert max(raw["atom_plddt"]) <= 1.0  # the fixture is on the model's scale
        record = _load(tmp_path)
        np.testing.assert_allclose(record.plddt_per_atom, truth.plddt_per_atom, atol=1e-9)
        assert record.plddt_per_atom.max() > 50.0

    def test_the_summary_plddt_is_already_on_0_100(self, tmp_path):
        truth = _write(tmp_path)
        raw = json.loads(_summary(tmp_path).read_text(encoding="utf-8"))
        assert raw["plddt"] > 1.0
        assert _load(tmp_path).avg_plddt == pytest.approx(truth.scalars["avg_plddt"])

    def test_the_precision_of_the_file_is_kept(self, tmp_path):
        _write(tmp_path)
        _edit(_full_data(tmp_path), lambda d: d.update(atom_plddt=[0.93, 0.5, 0.07] + [0.9] * 11))
        record = _load(tmp_path)
        np.testing.assert_allclose(record.plddt_per_atom[:3], [93.0, 50.0, 7.0])

    def test_a_pLDDT_that_is_not_on_0_1_is_refused_and_not_converted(self, tmp_path):
        _write(tmp_path)
        _edit(_full_data(tmp_path), lambda d: d.update(atom_plddt=[92.0] * 14))
        with pytest.raises(ValueError, match="not 0-1"):
            _load(tmp_path)


class TestOrientation:
    """``pae[i, j]`` is the error of token j when the structure is aligned on token i."""

    def test_the_pae_of_the_file_is_read_as_it_is(self, tmp_path):
        truth = _write(tmp_path)
        record = _load(tmp_path)
        # the fixture is pae[i, j] = 1 + 0.5 i + 0.25 j: token 6 aligned on token 0 differs
        # from token 0 aligned on token 6
        assert record.pae[0, 6] == pytest.approx(1.0 + 0.0 + 1.5)
        assert record.pae[6, 0] == pytest.approx(1.0 + 3.0 + 0.0)
        np.testing.assert_allclose(record.pae, truth.pae)
        assert not np.allclose(record.pae, record.pae.T)

    def test_a_matrix_written_transposed_comes_back_transposed(self, tmp_path):
        # the adapter must not transpose: a file that holds the transpose of the truth reads as
        # that transpose, so a parser that transposed would return the truth here and fail
        truth = _write(tmp_path)
        _edit(_full_data(tmp_path), lambda d: d.update(token_pair_pae=truth.pae.T.tolist()))
        np.testing.assert_allclose(_load(tmp_path).pae, truth.pae.T)

    def test_the_pde_of_the_file_is_read_as_it_is(self, tmp_path):
        truth = _write(tmp_path)
        record = _load(tmp_path)
        assert record.pde[0, 6] == pytest.approx(0.5 + 0.0 + 0.75)
        assert record.pde[6, 0] == pytest.approx(0.5 + 1.5 + 0.0)
        np.testing.assert_allclose(record.pde, truth.pde, atol=0.01)  # the file has 2 decimals

    def test_the_binder_rows_of_the_interface_block_are_the_frames_on_the_binder(self, tmp_path):
        # chain A is tokens 0-3 and chain B tokens 4-6; the binder-rows block is the error of the
        # receptor tokens when the structure is aligned on the binder
        truth = _write(tmp_path)
        record = _load(tmp_path)
        result = summarize_prediction(
            record, include_matrices=True, binder_chain="B", receptor_chain="A"
        )
        np.testing.assert_allclose(result["pae_interface"], truth.pae[4:7, 0:4])
        np.testing.assert_allclose(result["pde_interface"], truth.pde[4:7, 0:4], atol=0.01)
        both_blocks = (truth.pae[4:7, 0:4].mean() + truth.pae[0:4, 4:7].mean()) / 2
        assert result["mean_interface_pae"] == pytest.approx(both_blocks)


class TestSummary:
    def test_the_scalars_of_the_summary(self, tmp_path):
        truth = _write(tmp_path)
        record = _load(tmp_path)
        assert (record.model, record.name) == ("protenix", NAME)
        assert record.ptm == pytest.approx(truth.scalars["ptm"])
        assert record.iptm == pytest.approx(truth.scalars["iptm"])
        assert record.gpde == pytest.approx(truth.scalars["gpde"])
        assert record.has_clash == 0.0
        assert record.ranking_score == pytest.approx(truth.scalars["ranking_score"])
        assert record.ranking_score_name == "ranking_score"
        assert record.timing == {}
        assert record.reasons == []

    def test_the_chain_entries_are_keyed_by_chain_position(self, tmp_path):
        _write(tmp_path)
        record = _load(tmp_path)
        assert record.chain_ptm == {"0": pytest.approx(0.88), "1": pytest.approx(0.80)}
        # row i, column j of the matrix is "i-j"; the fixture is not symmetric, the model's is
        assert record.chain_pair_iptm == {"0-1": pytest.approx(0.76), "1-0": pytest.approx(0.74)}

    def test_the_diagonal_of_the_chain_pair_matrix_is_left_out(self, tmp_path):
        _write(tmp_path)

        def _diagonal(data):
            data["chain_pair_iptm"][0][0] = 0.99
            data["chain_pair_iptm"][1][1] = 0.99

        _edit(_summary(tmp_path), _diagonal)
        assert set(_load(tmp_path).chain_pair_iptm) == {"0-1", "1-0"}

    def test_the_model_specific_entries_go_to_extras_as_written(self, tmp_path):
        _write(tmp_path)
        extras = _load(tmp_path).extras
        assert extras["num_recycles"] == 10
        assert set(extras) >= {
            "chain_iptm",
            "chain_pair_iptm_global",
            "chain_plddt",
            "chain_pair_plddt",
            "chain_gpde",
            "chain_pair_gpde",
        }
        assert extras["rank_index"] == 0
        assert extras["seed_value"] == "9"
        assert "bespoke_iptm" not in extras

    def test_the_pair_pae_entries_of_the_newer_source_are_read_when_present(self, tmp_path):
        _write(tmp_path)
        _edit(
            _summary(tmp_path),
            lambda d: d.update(
                chain_pair_pae_mean=[[0, 4.5], [4.5, 0]], chain_pair_pae_min=[[0, 2]]
            ),
        )
        extras = _load(tmp_path).extras
        assert extras["chain_pair_pae_mean"] == [[0, 4.5], [4.5, 0]]
        assert extras["chain_pair_pae_min"] == [[0, 2]]

    def test_the_rank_and_seed_of_a_later_sample_are_recorded(self, tmp_path):
        _write(tmp_path, seed_index=1)  # a seed index is a position: seed 10 is the second
        _write(tmp_path, seed_index=2, sample=3)
        record = _load(tmp_path, seed_index=2, sample=3)
        assert (record.extras["rank_index"], record.extras["seed_value"]) == (2, "10")
        assert record.structure_path.name == f"{NAME}_sample_2.cif"

    def test_a_disorder_of_zero_is_a_placeholder_and_not_a_measurement(self, tmp_path):
        _write(tmp_path)
        record = _load(tmp_path)
        assert np.isnan(record.disorder)
        assert record.extras["disorder_written"] == 0.0

    def test_a_non_zero_disorder_is_kept(self, tmp_path):
        _write(tmp_path)
        _edit(_summary(tmp_path), lambda d: d.update(disorder=0.25))
        assert _load(tmp_path).disorder == pytest.approx(0.25)

    def test_a_clash_written_as_a_boolean_is_read_as_a_number(self, tmp_path):
        _write(tmp_path)
        _edit(_summary(tmp_path), lambda d: d.update(has_clash=True, ranking_score=-99.2))
        record = _load(tmp_path)
        assert record.has_clash == 1.0
        assert record.ranking_score == pytest.approx(-99.2)

    def test_a_chain_entry_of_another_shape_is_left_out_with_a_reason(self, tmp_path):
        _write(tmp_path)
        _edit(_summary(tmp_path), lambda d: d.update(chain_ptm={"A": 0.9}, chain_pair_iptm=[0.1]))
        record = _load(tmp_path)
        assert record.chain_ptm == {} and record.chain_pair_iptm == {}
        assert any("chain_ptm" in r for r in record.reasons)
        assert any("chain_pair_iptm" in r for r in record.reasons)
        assert record.ptm == pytest.approx(0.88)  # the scalars are still read
        record.validate()

    def test_a_missing_scalar_is_nan(self, tmp_path):
        _write(tmp_path)

        def _drop(data):
            del data["gpde"]
            data["iptm"] = None

        _edit(_summary(tmp_path), _drop)
        record = _load(tmp_path)
        assert np.isnan(record.gpde) and np.isnan(record.iptm)
        assert record.ptm == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# A run without --need_atom_confidence, and other missing files
# ---------------------------------------------------------------------------


class TestWithoutTheFullDataFile:
    @pytest.fixture
    def record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_protenix, "WRITE_FULL_DATA", False)
        _write(tmp_path)
        return _load(tmp_path)

    def test_the_scalars_are_kept_and_the_arrays_are_empty(self, record):
        assert record.avg_plddt == pytest.approx(synth.synthetic_complex().scalars["avg_plddt"])
        assert record.ptm == pytest.approx(0.88)
        assert record.plddt_per_atom is None and record.pae is None and record.pde is None
        assert record.tokens is None

    def test_the_reason_names_the_flag(self, record):
        assert len(record.reasons) == 1
        assert "--need_atom_confidence true" in record.reasons[0]

    def test_the_record_is_valid(self, record):
        record.validate(check_structure=True)

    def test_the_summary_says_why_the_arrays_are_missing(self, record):
        result = summarize_prediction(record, binder_chain="B", receptor_chain="A")
        assert np.isnan(result["mean_interface_pae"])
        assert result["plddt_per_atom"] is None
        assert "--need_atom_confidence true" in result["reason"]
        assert result["avg_plddt"] == pytest.approx(record.avg_plddt)


class TestOtherMissingFiles:
    def test_without_the_summary_the_mean_pLDDT_comes_from_the_atoms(self, tmp_path):
        truth = _write(tmp_path)
        _summary(tmp_path).unlink()
        record = _load(tmp_path)
        assert record.avg_plddt == pytest.approx(truth.plddt_per_atom.mean())
        assert np.isnan(record.ptm) and np.isnan(record.ranking_score)
        assert record.pae is not None
        assert record.reasons == ["summary confidence file not found"]

    def test_a_sample_that_failed_names_its_error(self, tmp_path):
        err = tmp_path / "ERR"
        err.mkdir()
        (err / f"{NAME}.txt").write_text(
            f"[Rank 0] {NAME} failed: CUDA out of memory\nTraceback (most recent call last):\n",
            encoding="utf-8",
        )
        record = _load(tmp_path)
        assert record.structure_path is None
        assert len(record.reasons) == 2
        assert "CUDA out of memory" in record.reasons[1]
        assert "Traceback" not in record.reasons[1]

    def test_a_missing_structure_is_named(self, tmp_path):
        _write(tmp_path)
        (_predictions(tmp_path) / f"{NAME}_sample_0.cif").unlink()
        record = _load(tmp_path)
        assert record.structure_path is None
        assert record.reasons == ["structure file not found"]
        assert record.pae is not None


# ---------------------------------------------------------------------------
# Files that are not what the adapter was written for
# ---------------------------------------------------------------------------


class TestUnexpectedContent:
    def test_a_summary_that_is_not_an_object_raises(self, tmp_path):
        _write(tmp_path)
        _summary(tmp_path).write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="must hold a JSON object"):
            _load(tmp_path)

    def test_a_summary_without_any_known_key_raises(self, tmp_path):
        _write(tmp_path)
        _summary(tmp_path).write_text(
            '{"avg_plddt": 90.0, "sample_ranking_score": 0.8}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="does not look like a Protenix summary_confidence"):
            _load(tmp_path)

    def test_a_scalar_that_is_not_a_number_raises(self, tmp_path):
        _write(tmp_path)
        _edit(_summary(tmp_path), lambda d: d.update(ptm="high"))
        with pytest.raises(ValueError, match="'ptm' must be a number"):
            _load(tmp_path)

    def test_a_full_data_file_without_any_known_key_raises(self, tmp_path):
        _write(tmp_path)
        _full_data(tmp_path).write_text('{"plddt": [1, 2]}', encoding="utf-8")
        with pytest.raises(ValueError, match="does not look like a Protenix full_data"):
            _load(tmp_path)

    @pytest.mark.parametrize(
        "edit, message",
        [
            (lambda d: d.update(token_pair_pae=d["token_pair_pae"][:-1]), "must be square"),
            (lambda d: d.update(token_pair_pae=[[1.0], [2.0, 3.0]]), "not a numeric array"),
            (lambda d: d.update(token_pair_pae=[1.0, 2.0]), "2 dimension"),
            (lambda d: d.update(atom_plddt=[[0.9, 0.8]]), "1 dimension"),
            (
                lambda d: d.update(token_pair_pde=[[0.5] * 6] * 6),
                "disagree on the number of tokens",
            ),
            (
                lambda d: d.update(token_asym_id=[0, 0, 1]),
                "disagree on the number of tokens",
            ),
            (
                lambda d: d.update(atom_to_token_idx=[0, 1, 2]),
                "disagree on the number of atoms",
            ),
            (
                lambda d: d.update(atom_to_token_idx=[9] * 14),
                "points to tokens",
            ),
            (lambda d: d.update(token_asym_id=[0.5] * 7), "must hold integers"),
        ],
        ids=[
            "non-square-pae",
            "ragged-pae",
            "one-dimensional-pae",
            "two-dimensional-plddt",
            "pae-and-pde-differ",
            "token-vector-length",
            "atom-vector-length",
            "token-index-out-of-range",
            "non-integer-asym-id",
        ],
    )
    def test_arrays_of_another_shape_are_refused_with_a_message(self, tmp_path, edit, message):
        _write(tmp_path)
        _edit(_full_data(tmp_path), edit)
        with pytest.raises(ValueError, match=message):
            _load(tmp_path)


# ---------------------------------------------------------------------------
# Listing the samples
# ---------------------------------------------------------------------------


class TestListSamples:
    def test_samples_come_in_seed_order_then_rank_order_with_their_scores(self, tmp_path):
        for seed_index, sample, score in ((1, 1, 0.9), (1, 2, 0.8), (2, 1, 0.7)):
            _write(tmp_path, seed_index=seed_index, sample=sample)
            summary = _summary(tmp_path, rank=sample - 1, seed=8 + seed_index)
            _edit(summary, lambda d, s=score: d.update(ranking_score=s))
        refs = ProtenixParser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample, r.ranking_score) for r in refs] == [
            (1, 1, 0.9),
            (1, 2, 0.8),
            (2, 1, 0.7),
        ]

    def test_the_full_data_files_are_not_opened(self, tmp_path):
        _write(tmp_path)
        _full_data(tmp_path).write_bytes(contract.GARBAGE)
        refs = ProtenixParser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1)]

    def test_an_empty_directory_has_no_samples(self, tmp_path):
        assert ProtenixParser().list_samples(tmp_path, NAME) == []

    def test_a_corrupt_summary_raises(self, tmp_path):
        _write(tmp_path)
        _summary(tmp_path).write_bytes(contract.GARBAGE)
        with pytest.raises(ValueError):
            ProtenixParser().list_samples(tmp_path, NAME)
