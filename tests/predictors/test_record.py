"""PredictionRecord, PredictionFiles, SampleRef and TokenLayout."""

import dataclasses

import numpy as np
import pytest

from binding_metrics.predictors.record import (
    PredictionFiles,
    PredictionRecord,
    SampleRef,
    TokenLayout,
    check_chain_map,
)
from tests.predictors import synth


def _good_record(**overrides):
    fields = dict(
        seed_index=1,
        sample=1,
        avg_plddt=82.0,
        ptm=0.88,
        iptm=0.76,
        gpde=1.23,
        ranking_score=0.82,
        ranking_score_name="sample_ranking_score",
        has_clash=0.0,
        disorder=0.12,
        chain_ptm={"A": 0.88},
        chain_pair_iptm={"(A, B)": 0.76},
        plddt_per_atom=synth.synthetic_complex().plddt_per_atom,
        pae=synth.synthetic_complex().pae,
        pde=synth.synthetic_complex().pde,
    )
    fields.update(overrides)
    return PredictionRecord("of3", "cmplx", **fields)


def _record_with_structure(tmp_path, suffix=".cif", **overrides):
    complex_ = synth.synthetic_complex()
    path = synth.write_structure(complex_.atoms, tmp_path / f"model{suffix}")
    return _good_record(structure_path=path, **overrides)


# ---------------------------------------------------------------------------
# PredictionFiles and SampleRef
# ---------------------------------------------------------------------------


class TestPredictionFiles:
    def test_nothing_found(self, tmp_path):
        files = PredictionFiles(directory=tmp_path)
        assert files.found() == {}
        assert not files.any_found()

    def test_found_lists_the_named_files_then_the_extra_ones(self, tmp_path):
        files = PredictionFiles(
            directory=tmp_path,
            structure=tmp_path / "m.cif",
            scores=tmp_path / "s.json",
            extra={"pae": tmp_path / "pae.npz"},
        )
        assert files.found() == {
            "structure": tmp_path / "m.cif",
            "scores": tmp_path / "s.json",
            "pae": tmp_path / "pae.npz",
        }
        assert files.any_found()

    def test_a_shared_timing_file_is_found_but_is_not_sample_output(self, tmp_path):
        files = PredictionFiles(directory=tmp_path, timing=tmp_path / "timing.json")
        assert files.any_found()
        assert not files.has_output()

    @pytest.mark.parametrize("role", ["structure", "scores", "arrays", "extra"])
    def test_any_file_of_the_sample_is_output(self, tmp_path, role):
        kwargs = (
            {"extra": {"pae": tmp_path / "pae.npz"}} if role == "extra" else {role: tmp_path / "x"}
        )
        assert PredictionFiles(directory=tmp_path, **kwargs).has_output()

    @pytest.mark.parametrize("role", ["structure", "scores", "arrays", "timing"])
    def test_an_extra_role_may_not_reuse_a_named_role(self, tmp_path, role):
        with pytest.raises(ValueError, match="reserved"):
            PredictionFiles(directory=tmp_path, extra={role: tmp_path / "x"})

    def test_is_immutable(self, tmp_path):
        files = PredictionFiles(directory=tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            files.structure = tmp_path / "m.cif"


def test_sample_ref_defaults_to_no_ranking_score():
    ref = SampleRef(seed_index=2, sample=3)
    assert (ref.seed_index, ref.sample) == (2, 3)
    assert np.isnan(ref.ranking_score)


# ---------------------------------------------------------------------------
# TokenLayout
# ---------------------------------------------------------------------------


def _layout(chains, **kwargs):
    n = len(chains)
    return TokenLayout(
        chain_id=np.array(chains),
        res_id=np.arange(1, n + 1),
        atom_index=np.arange(n) * 2,
        **kwargs,
    )


class TestTokenLayout:
    def test_length_and_ranges_in_order_of_first_appearance(self):
        layout = _layout(["B", "B", "A", "A", "A", "L"])
        assert len(layout) == 6
        assert layout.token_ranges() == {"B": (0, 2), "A": (2, 5), "L": (5, 6)}

    def test_a_chain_that_is_not_one_run_cannot_be_sliced(self):
        with pytest.raises(ValueError, match="chain 'A' are not contiguous"):
            _layout(["A", "B", "A"]).token_ranges()

    def test_a_sound_layout_has_no_problems(self):
        layout = _layout(
            ["A", "A", "B"],
            is_atom_token=np.array([False, False, True]),
            extras={"contact_probability": np.zeros(3)},
        )
        assert layout.problems() == []

    def test_arrays_of_different_length_are_reported(self):
        layout = TokenLayout(
            chain_id=np.array(["A", "A"]), res_id=np.array([1, 2, 3]), atom_index=np.array([0, 1])
        )
        assert "differ in length" in layout.problems()[0]

    def test_an_extra_array_of_the_wrong_length_is_reported(self):
        layout = _layout(["A", "A"], extras={"per_token": np.zeros(5)})
        assert "extras['per_token']" in layout.problems()[0]

    def test_a_negative_atom_index_is_reported(self):
        layout = TokenLayout(
            chain_id=np.array(["A"]), res_id=np.array([1]), atom_index=np.array([-1])
        )
        assert "negative" in layout.problems()[0]


# ---------------------------------------------------------------------------
# PredictionRecord: construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_scalars_are_nan_and_never_none(self):
        record = PredictionRecord("boltz2", "q")
        for name in ("avg_plddt", "ptm", "iptm", "gpde", "ranking_score", "has_clash", "disorder"):
            assert np.isnan(getattr(record, name)), name
        assert record.plddt_per_atom is None
        assert record.pae is None and record.pde is None and record.tokens is None
        assert record.structure_path is None
        assert (record.seed_index, record.sample) == (1, 1)
        assert record.n_atoms == 0

    def test_containers_are_not_shared_between_records(self):
        first, second = PredictionRecord("of3", "a"), PredictionRecord("of3", "b")
        first.reasons.append("x")
        first.extras["k"] = 1
        first.chain_map["A"] = "R"
        assert second.reasons == [] and second.extras == {} and second.chain_map == {}

    def test_only_model_and_name_are_positional(self):
        with pytest.raises(TypeError, match="positional"):
            PredictionRecord("of3", "q", 1)

    def test_structure_path_is_coerced_to_a_path(self, tmp_path):
        record = PredictionRecord("of3", "q", structure_path=str(tmp_path / "m.cif"))
        assert record.structure_path == tmp_path / "m.cif"

    def test_n_atoms_counts_the_plddt_values(self):
        assert PredictionRecord("of3", "q", plddt_per_atom=np.full(9, 80.0)).n_atoms == 9

    def test_files_default_to_none_and_hold_what_the_adapter_located(self, tmp_path):
        assert PredictionRecord("of3", "q").files is None
        files = PredictionFiles(directory=tmp_path, arrays=tmp_path / "a.json")
        assert PredictionRecord("of3", "q", files=files).files is files

    def test_equality_is_identity(self):
        record = _good_record()
        assert record == record
        assert record != _good_record()  # array fields make value equality ambiguous


# ---------------------------------------------------------------------------
# PredictionRecord.validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_a_sound_record_passes(self):
        _good_record().validate()

    def test_a_record_of_missing_values_passes(self):
        PredictionRecord("of3", "q").validate()

    def test_nan_scalars_pass(self):
        _good_record(ptm=float("nan"), iptm=float("nan"), gpde=float("nan")).validate()

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"ptm": None}, "ptm is None; use NaN"),
            ({"ptm": 1.5}, "ptm 1.5 is outside 0 to 1"),
            ({"iptm": -0.1}, "iptm -0.1 is outside 0 to 1"),
            ({"gpde": -2.0}, "gpde -2 is outside 0 to inf"),
            ({"avg_plddt": 101.0}, "avg_plddt 101 is outside 0 to 100"),
            ({"avg_plddt": 0.85}, "avg_plddt 0.85 reads as a 0-1 value"),
            ({"has_clash": 2.0}, "has_clash 2 is outside 0 to 1"),
            ({"avg_plddt": float("inf")}, "avg_plddt is infinite"),
            ({"avg_plddt": "high"}, "avg_plddt must be a number"),
            ({"ranking_score_name": ""}, "ranking_score is set but ranking_score_name is empty"),
            ({"chain_ptm": {"A": 3.0}}, "chain_ptm['A'] 3 is outside 0 to 1"),
            ({"chain_pair_iptm": {"(A, B)": None}}, "chain_pair_iptm['(A, B)'] is None"),
            ({"seed_index": 0}, "1-based"),
        ],
    )
    def test_scalar_violations_are_named(self, overrides, message):
        with pytest.raises(ValueError, match="invalid of3 prediction record 'cmplx'") as info:
            _good_record(**overrides).validate()
        assert message in str(info.value)

    def test_every_violation_is_listed_not_only_the_first(self):
        with pytest.raises(ValueError) as info:
            _good_record(ptm=2.0, iptm=None, avg_plddt=150.0).validate()
        message = str(info.value)
        assert "ptm 2 is outside" in message
        assert "iptm is None" in message
        assert "avg_plddt 150 is outside" in message

    @pytest.mark.parametrize(
        "plddt, message",
        [
            (np.full(14, 0.9), "looks like a 0-1 scale"),
            (np.array([50.0, 120.0]), "outside 0-100"),
            (np.array([50.0, -1.0]), "outside 0-100"),
            (np.array([50.0, np.nan]), "non-finite"),
            (np.full((2, 2), 80.0), "one-dimensional"),
            ([80.0, 90.0], "one-dimensional numpy array"),
        ],
    )
    def test_plddt_array_violations(self, plddt, message):
        with pytest.raises(ValueError, match=message):
            _good_record(plddt_per_atom=plddt).validate()

    @pytest.mark.parametrize("name", ["pae", "pde"])
    @pytest.mark.parametrize(
        "matrix, message",
        [
            (np.zeros((7, 5)), "square two-dimensional"),
            (np.zeros(7), "square two-dimensional"),
            (-np.ones((7, 7)), "finite and not negative"),
            (np.full((7, 7), np.nan), "finite and not negative"),
        ],
    )
    def test_matrix_violations(self, name, matrix, message):
        with pytest.raises(ValueError, match=message):
            _good_record(**{name: matrix}).validate()

    def test_a_matrix_must_be_as_large_as_the_token_layout(self):
        layout = _layout(["A"] * 4 + ["B"] * 3)
        _good_record(tokens=layout).validate()
        with pytest.raises(ValueError, match="pae has 5 rows but tokens has 7"):
            _good_record(tokens=layout, pae=np.zeros((5, 5))).validate()

    def test_an_inconsistent_token_layout_is_reported(self):
        layout = TokenLayout(
            chain_id=np.array(["A", "A"]), res_id=np.array([1]), atom_index=np.array([0, 1])
        )
        with pytest.raises(ValueError, match="differ in length"):
            _good_record(tokens=layout, pae=None, pde=None).validate()

    def test_two_chains_renamed_to_one_id_are_reported(self):
        with pytest.raises(ValueError, match="renamed to the same ID"):
            _good_record(chain_map={"A": "X", "B": "X"}).validate()


class TestValidateAgainstTheStructure:
    def test_a_sound_record_passes(self, tmp_path):
        _record_with_structure(tmp_path).validate(check_structure=True)

    def test_plddt_that_does_not_fit_the_atom_count(self, tmp_path):
        record = _record_with_structure(tmp_path, plddt_per_atom=np.full(10, 80.0))
        with pytest.raises(ValueError, match=r"plddt_per_atom has 10 values but .* 14 atoms"):
            record.validate(check_structure=True)

    def test_a_chain_map_that_names_an_absent_chain(self, tmp_path):
        record = _record_with_structure(tmp_path, chain_map={"Z": "R"})
        with pytest.raises(ValueError, match="structure could not be read: ValueError"):
            record.validate(check_structure=True)

    def test_token_atom_indices_must_point_at_atoms(self, tmp_path):
        layout = TokenLayout(
            chain_id=np.array(["A"] * 7),
            res_id=np.arange(1, 8),
            atom_index=np.arange(7) * 3,  # the last is 18, beyond the 14 atoms
        )
        record = _record_with_structure(tmp_path, tokens=layout)
        with pytest.raises(ValueError, match="beyond the last atom"):
            record.validate(check_structure=True)

    def test_a_missing_structure_is_a_problem_only_when_asked(self):
        record = _good_record()
        record.validate()
        with pytest.raises(ValueError, match="structure_path is None"):
            record.validate(check_structure=True)

    def test_an_unreadable_structure_is_reported_not_raised_raw(self, tmp_path):
        stub = tmp_path / "model.cif"
        stub.write_text("# stub CIF\n", encoding="utf-8")
        with pytest.raises(ValueError, match="structure could not be read"):
            _good_record(structure_path=stub).validate(check_structure=True)


# ---------------------------------------------------------------------------
# PredictionRecord.atoms
# ---------------------------------------------------------------------------


class TestAtoms:
    def test_reads_the_structure_with_its_chain_ids(self, tmp_path):
        atoms = _record_with_structure(tmp_path).atoms()
        assert atoms.array_length() == synth.N_ATOMS
        assert list(dict.fromkeys(atoms.chain_id)) == ["A", "B"]

    def test_is_read_once(self, tmp_path):
        record = _record_with_structure(tmp_path)
        assert record.atoms() is record.atoms()

    def test_reads_pdb_and_gzip_compressed_files(self, tmp_path):
        for suffix in (".pdb", ".cif.gz", ".pdb.gz"):
            record = _record_with_structure(tmp_path, suffix=suffix)
            assert record.atoms().array_length() == synth.N_ATOMS, suffix

    def test_chain_map_renames_chains(self, tmp_path):
        record = _record_with_structure(tmp_path, chain_map={"A": "R", "B": "P"})
        atoms = record.atoms()
        assert set(atoms.chain_id) == {"R", "P"}
        assert (atoms.chain_id == "R").sum() == 8  # 4 residues x 2 atoms

    def test_a_partial_chain_map_leaves_the_other_chains_alone(self, tmp_path):
        atoms = _record_with_structure(tmp_path, chain_map={"A": "R"}).atoms()
        assert set(atoms.chain_id) == {"R", "B"}

    def test_a_swap_is_applied_at_once(self, tmp_path):
        record = _record_with_structure(tmp_path, chain_map={"A": "B", "B": "A"})
        atoms = record.atoms()
        # chain A had 4 residues; after the swap the 8 atoms of the old A are called B
        assert (atoms.chain_id == "B").sum() == 8
        assert (atoms.chain_id == "A").sum() == 6

    def test_chain_ids_longer_than_the_file_allows_are_kept(self, tmp_path):
        atoms = _record_with_structure(tmp_path, chain_map={"A": "RECEPTOR"}).atoms()
        assert "RECEPTOR" in set(atoms.chain_id)

    def test_a_chain_the_structure_does_not_have_is_an_error(self, tmp_path):
        record = _record_with_structure(tmp_path, chain_map={"Z": "R"})
        with pytest.raises(ValueError, match=r"names chains \['Z'\].*has \['A', 'B'\]"):
            record.atoms()

    def test_renaming_into_an_existing_chain_would_merge_and_is_refused(self, tmp_path):
        record = _record_with_structure(tmp_path, chain_map={"A": "B"})
        with pytest.raises(ValueError, match="would merge chains"):
            record.atoms()

    def test_the_cache_follows_a_change_of_chain_map(self, tmp_path):
        record = _record_with_structure(tmp_path)
        assert set(record.atoms().chain_id) == {"A", "B"}
        record.chain_map = {"A": "R"}
        assert set(record.atoms().chain_id) == {"R", "B"}
        record.chain_map = {}
        assert set(record.atoms().chain_id) == {"A", "B"}

    def test_the_structure_file_is_not_modified(self, tmp_path):
        record = _record_with_structure(tmp_path, chain_map={"A": "R"})
        before = record.structure_path.read_bytes()
        record.atoms()
        assert record.structure_path.read_bytes() == before

    def test_no_structure_file_gives_a_value_error_with_the_reasons(self):
        record = PredictionRecord("of3", "q", reasons=["structure file not found"])
        with pytest.raises(ValueError, match="no structure file; structure file not found"):
            record.atoms()


# ---------------------------------------------------------------------------
# check_chain_map
# ---------------------------------------------------------------------------


class TestCheckChainMap:
    def test_returns_a_plain_dict(self):
        assert check_chain_map({"A": "R"}) == {"A": "R"}
        assert type(check_chain_map({"A": "R"})) is dict

    @pytest.mark.parametrize(
        "chain_map, message",
        [
            ({"A": 1}, "non-empty strings"),
            ({"": "R"}, "non-empty strings"),
            ({"A": ""}, "non-empty strings"),
            ({"A": "X", "B": "X"}, "same ID"),
            (["A", "R"], "must be a mapping"),
        ],
    )
    def test_invalid_maps(self, chain_map, message):
        with pytest.raises(ValueError, match=message):
            check_chain_map(chain_map)


# ---------------------------------------------------------------------------
# The synthetic complex used by every adapter test
# ---------------------------------------------------------------------------


class TestSyntheticComplex:
    def test_truth_is_consistent(self):
        truth = synth.synthetic_complex()
        assert truth.n_atoms == synth.N_ATOMS == truth.plddt_per_atom.size
        assert truth.n_tokens == synth.N_TOKENS
        assert truth.scalars["avg_plddt"] == pytest.approx(82.0)
        assert truth.chain_ids == ("A", "B")
        assert not np.allclose(truth.pae, truth.pae.T)  # a transposed matrix is detectable
        assert not np.allclose(truth.pde, truth.pde.T)

    def test_a_shifted_sample_differs_only_in_plddt(self):
        base, shifted = synth.synthetic_complex(), synth.synthetic_complex(plddt_shift=10.0)
        assert shifted.scalars["avg_plddt"] == pytest.approx(72.0)
        np.testing.assert_allclose(shifted.plddt_per_atom, base.plddt_per_atom - 10.0)
        np.testing.assert_allclose(shifted.pae, base.pae)

    def test_the_written_structure_carries_the_plddt_in_the_b_factor_column(self, tmp_path):
        import biotite.structure.io.pdbx as pdbx

        truth = synth.synthetic_complex()
        for suffix in (".cif", ".pdb"):
            path = synth.write_structure(truth.atoms, tmp_path / f"m{suffix}")
            if suffix == ".cif":
                atoms = pdbx.get_structure(
                    pdbx.CIFFile.read(str(path)), model=1, extra_fields=["b_factor"]
                )
            else:
                import biotite.structure.io.pdb as pdb_io

                atoms = pdb_io.get_structure(
                    pdb_io.PDBFile.read(str(path)), model=1, extra_fields=["b_factor"]
                )
            np.testing.assert_allclose(atoms.b_factor, truth.plddt_per_atom, atol=0.01)

    def test_bfactor_scale_mimics_a_model_that_writes_zero_to_one(self, tmp_path):
        import biotite.structure.io.pdbx as pdbx

        truth = synth.synthetic_complex()
        path = synth.write_structure(truth.atoms, tmp_path / "m.cif", bfactor_scale=0.01)
        atoms = pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1, extra_fields=["b_factor"])
        np.testing.assert_allclose(atoms.b_factor, truth.plddt_per_atom / 100.0, atol=0.005)

    def test_renamed_atoms_swap_chains_at_once(self):
        truth = synth.synthetic_complex()
        swapped = synth.renamed_atoms(truth, {"A": "B", "B": "A"})
        assert (swapped.chain_id == "B").sum() == 8
        assert (truth.atoms.chain_id == "A").sum() == 8  # the truth is not modified

    def test_json_and_npz_helpers_round_trip(self, tmp_path):
        payload = {"pae": np.arange(4.0).reshape(2, 2), "ptm": np.float32(0.5)}
        path = synth.write_json(tmp_path / "s.json", payload)
        import json

        assert json.loads(path.read_text(encoding="utf-8")) == {
            "pae": [[0.0, 1.0], [2.0, 3.0]],
            "ptm": 0.5,
        }
        npz = synth.write_npz(tmp_path / "a.npz", pae=np.arange(4.0).reshape(2, 2))
        with np.load(npz) as data:
            np.testing.assert_allclose(data["pae"], [[0.0, 1.0], [2.0, 3.0]])
