"""PredictionParser: the abstract steps, ``load`` and the default ``list_samples``."""

from pathlib import Path

import numpy as np
import pytest

from binding_metrics.predictors import base
from binding_metrics.predictors.base import FAMILIES, PredictionParser
from binding_metrics.predictors.record import PredictionFiles, PredictionRecord
from tests.predictors import synth
from tests.predictors.synth_stub import StubParser, write_prediction


@pytest.fixture
def written(tmp_path):
    """A stub prediction 'cmplx', sample 1 (pLDDT 82 on average) and sample 2 (72)."""
    write_prediction(tmp_path, "cmplx", synth.synthetic_complex())
    write_prediction(tmp_path, "cmplx", synth.synthetic_complex(plddt_shift=10.0), sample=2)
    return tmp_path


class TestTheAbstractClass:
    def test_cannot_be_instantiated_without_the_two_steps(self):
        class OnlyFind(PredictionParser):
            name, display_name, family = "x", "X", "af2"

            def find_files(self, prediction_dir, name, *, seed_index=1, sample=1):
                return PredictionFiles(directory=Path(prediction_dir))

        with pytest.raises(TypeError, match="abstract"):
            OnlyFind()

    def test_a_family_must_be_one_of_the_two_known(self):
        assert FAMILIES == ("af2", "af3")
        with pytest.raises(TypeError, match=r"family must be one of \('af2', 'af3'\)"):

            class Bad(PredictionParser):
                name, display_name, family = "bad", "Bad", "af4"

                def find_files(self, *args, **kwargs): ...

                def parse(self, *args, **kwargs): ...

    def test_no_capabilities_are_declared_by_default(self):
        assert PredictionParser.capabilities is None
        assert StubParser.capabilities is None

    def test_an_adapter_may_declare_capabilities_and_a_sibling_is_unaffected(self):
        marker = object()

        class Constrained(StubParser):
            capabilities = marker

        assert Constrained.capabilities is marker
        assert StubParser.capabilities is None

    def test_the_stub_declares_the_class_attributes(self):
        assert (StubParser.name, StubParser.display_name, StubParser.family) == (
            "stub",
            "Stub model",
            "af3",
        )


class TestLoad:
    def test_returns_the_record_of_the_requested_sample(self, written):
        first = StubParser().load(written, "cmplx")
        second = StubParser().load(written, "cmplx", sample=2)
        assert (first.seed_index, first.sample) == (1, 1)
        assert (second.seed_index, second.sample) == (1, 2)
        assert first.avg_plddt == pytest.approx(82.0)
        assert second.avg_plddt == pytest.approx(72.0)
        first.validate(check_structure=True)

    def test_a_string_path_is_accepted(self, written):
        assert StubParser().load(str(written), "cmplx").model == "stub"

    def test_the_values_are_converted_to_the_record_rules(self, written):
        truth = synth.synthetic_complex()
        record = StubParser().load(written, "cmplx")
        np.testing.assert_allclose(record.plddt_per_atom, truth.plddt_per_atom, atol=1e-9)
        np.testing.assert_allclose(record.pae, truth.pae)
        assert record.extras == {"stub_only": 42}
        assert record.ranking_score_name == "stub_score"
        assert record.timing == {"inference": 1.5}

    def test_no_chain_map_means_no_renaming(self, written):
        record = StubParser().load(written, "cmplx")
        assert record.chain_map == {}
        assert set(record.atoms().chain_id) == {"A", "B"}

    def test_a_chain_map_is_stored_and_applied_to_the_atoms(self, written):
        record = StubParser().load(written, "cmplx", chain_map={"A": "R", "B": "P"})
        assert record.chain_map == {"A": "R", "B": "P"}
        assert set(record.atoms().chain_id) == {"R", "P"}
        # the keys of the model's own dictionaries are left as the model wrote them
        assert set(record.chain_ptm) == {"A", "B"}

    def test_the_callers_chain_map_is_copied(self, written):
        mapping = {"A": "R"}
        record = StubParser().load(written, "cmplx", chain_map=mapping)
        mapping["B"] = "P"
        assert record.chain_map == {"A": "R"}

    def test_an_invalid_chain_map_is_refused_before_any_file_is_read(self, written, monkeypatch):
        def _never(*args, **kwargs):
            raise AssertionError("find_files must not run")

        monkeypatch.setattr(StubParser, "find_files", _never)
        with pytest.raises(ValueError, match="same ID"):
            StubParser().load(written, "cmplx", chain_map={"A": "X", "B": "X"})


class TestMissingAndCorruptFiles:
    def test_an_empty_directory_gives_a_record_with_a_reason(self, tmp_path):
        record = StubParser().load(tmp_path, "cmplx")
        assert np.isnan(record.avg_plddt) and np.isnan(record.ptm)
        assert record.plddt_per_atom is None and record.pae is None
        assert record.structure_path is None
        assert record.reasons == [
            f"no stub output found for 'cmplx' (seed index 1, sample 1) in {tmp_path}"
        ]

    def test_a_directory_that_does_not_exist_is_not_an_error(self, tmp_path):
        record = StubParser().load(tmp_path / "nowhere", "cmplx")
        assert record.reasons

    def test_a_missing_arrays_file_keeps_the_scalars(self, written):
        (written / "cmplx_s1_m1.arrays.npz").unlink()
        record = StubParser().load(written, "cmplx")
        assert record.reasons == ["arrays file not found"]
        assert record.avg_plddt == pytest.approx(82.0)
        assert record.pae is None

    def test_a_missing_scores_file_keeps_the_arrays(self, written):
        (written / "cmplx_s1_m1.scores.json").unlink()
        record = StubParser().load(written, "cmplx")
        assert record.reasons == ["scores file not found"]
        assert np.isnan(record.ptm)
        assert record.avg_plddt == pytest.approx(82.0)  # from the per-atom values

    def test_a_missing_sample_is_not_found(self, written):
        assert StubParser().load(written, "cmplx", sample=3).reasons
        assert StubParser().load(written, "cmplx", seed_index=2).reasons

    @pytest.mark.parametrize("suffix", [".scores.json", ".arrays.npz"])
    def test_a_corrupt_file_raises(self, written, suffix):
        (written / f"cmplx_s1_m1{suffix}").write_bytes(b"\x00 not a valid file \xff")
        with pytest.raises((ValueError, OSError, EOFError)):
            StubParser().load(written, "cmplx")

    def test_a_missing_structure_is_reported_when_the_atoms_are_asked_for(self, written):
        (written / "cmplx_s1_m1.cif").unlink()
        record = StubParser().load(written, "cmplx")
        assert record.structure_path is None
        with pytest.raises(ValueError, match="has no structure file"):
            record.atoms()


class TestListSamples:
    def test_lists_seeds_and_samples_in_natural_order_with_their_ranking_scores(self, written):
        write_prediction(written, "cmplx", synth.synthetic_complex(), seed_index=2)
        refs = StubParser().list_samples(written, "cmplx")
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1), (1, 2), (2, 1)]
        assert [r.ranking_score for r in refs] == [pytest.approx(0.82)] * 3

    def test_nothing_found_gives_an_empty_list(self, tmp_path):
        assert StubParser().list_samples(tmp_path, "cmplx") == []

    def test_a_timing_file_shared_by_the_samples_does_not_make_absent_samples_appear(
        self, tmp_path
    ):
        class SharedTiming(StubParser):
            def find_files(self, prediction_dir, name, *, seed_index=1, sample=1):
                files = super().find_files(
                    prediction_dir, name, seed_index=seed_index, sample=sample
                )
                shared = Path(prediction_dir) / "timing.json"
                return PredictionFiles(
                    directory=files.directory,
                    structure=files.structure,
                    scores=files.scores,
                    arrays=files.arrays,
                    timing=shared if shared.exists() else None,
                )

        write_prediction(tmp_path, "cmplx", synth.synthetic_complex())
        (tmp_path / "timing.json").write_text("{}", encoding="utf-8")
        refs = SharedTiming().list_samples(tmp_path, "cmplx")
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1)]

    def test_the_list_is_capped_for_an_adapter_that_always_finds_files(self, monkeypatch):
        class Endless(StubParser):
            def find_files(self, prediction_dir, name, *, seed_index=1, sample=1):
                return PredictionFiles(directory=Path(prediction_dir), scores=Path("x"))

            def parse(self, files, *, name, seed_index=1, sample=1):
                return PredictionRecord(self.name, name, seed_index=seed_index, sample=sample)

        monkeypatch.setattr(base, "_PROBE_LIMIT", 4)
        refs = Endless().list_samples(Path("."), "q")
        assert len(refs) == 4
