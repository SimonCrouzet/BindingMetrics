"""PredictedPose as a Pose: kind, id, content hash, provenance and native_scores."""

import dataclasses
import hashlib
import math

import numpy as np
import pytest

from binding_metrics.poses import DockedPose, ExperimentalPose, Pose
from binding_metrics.predictors.record import PredictedPose
from tests.predictors import synth


def _pose(**overrides):
    fields = dict(
        seed_index=2,
        sample=3,
        avg_plddt=82.0,
        ptm=0.88,
        iptm=0.76,
        gpde=1.23,
        ranking_score=0.82,
        ranking_score_name="sample_ranking_score",
        has_clash=0.0,
        disorder=0.12,
        chain_ptm={"A": 0.88, "B": 0.8},
        chain_pair_iptm={"(A, B)": 0.76},
        plddt_per_atom=synth.synthetic_complex().plddt_per_atom,
        pae=synth.synthetic_complex().pae,
        pde=synth.synthetic_complex().pde,
    )
    fields.update(overrides)
    return PredictedPose("of3", "cmplx", **fields)


@pytest.fixture
def structure_file(tmp_path):
    return synth.write_structure(synth.synthetic_complex().atoms, tmp_path / "model.cif")


class TestKind:
    def test_a_predicted_pose_is_a_pose_of_the_predicted_kind(self):
        pose = _pose()
        assert isinstance(pose, Pose)
        assert PredictedPose.kind == "predicted" == pose.kind

    def test_the_three_kinds_are_distinct_subclasses_of_pose(self):
        kinds = {cls.kind: cls for cls in (PredictedPose, DockedPose, ExperimentalPose)}
        assert set(kinds) == {"predicted", "docked", "experimental"}
        assert all(issubclass(cls, Pose) for cls in kinds.values())

    def test_kind_is_not_a_dataclass_field(self):
        assert "kind" not in {f.name for f in dataclasses.fields(PredictedPose)}


class TestNativeScores:
    def test_the_scalars_and_the_per_chain_values_are_flat(self):
        assert _pose().native_scores() == {
            "avg_plddt": 82.0,
            "ptm": 0.88,
            "iptm": 0.76,
            "gpde": 1.23,
            "ranking_score": 0.82,
            "has_clash": 0.0,
            "disorder": 0.12,
            "chain_ptm.A": 0.88,
            "chain_ptm.B": 0.8,
            "chain_pair_iptm.(A, B)": 0.76,
        }

    def test_keys_come_in_a_fixed_order(self):
        keys = list(_pose().native_scores())
        assert keys[:7] == [
            "avg_plddt",
            "ptm",
            "iptm",
            "gpde",
            "ranking_score",
            "has_clash",
            "disorder",
        ]

    def test_nan_becomes_none(self):
        scores = PredictedPose("af2", "q", avg_plddt=70.0).native_scores()
        assert scores["avg_plddt"] == 70.0
        assert all(scores[k] is None for k in scores if k != "avg_plddt")
        assert len(scores) == 7

    def test_a_nan_per_chain_value_is_none_too(self):
        scores = _pose(chain_ptm={"A": float("nan")}).native_scores()
        assert scores["chain_ptm.A"] is None

    def test_values_are_plain_floats_and_none_only(self):
        scores = _pose(ptm=np.float32(0.5)).native_scores()
        assert all(v is None or type(v) is float for v in scores.values())
        assert not any(isinstance(v, float) and math.isnan(v) for v in scores.values())

    def test_the_arrays_are_not_scores(self):
        scores = _pose().native_scores()
        assert not {"plddt_per_atom", "pae", "pde"} & set(scores)
        assert all(v is None or np.isscalar(v) for v in scores.values())

    def test_the_names_of_the_ranking_score_are_not_values(self):
        assert "ranking_score_name" not in _pose().native_scores()

    def test_a_new_dictionary_each_time(self):
        pose = _pose()
        pose.native_scores()["ptm"] = 0.0
        assert pose.native_scores()["ptm"] == 0.88

    def test_it_is_json_serialisable(self):
        import json

        json.dumps(_pose().native_scores(), allow_nan=False)
        json.dumps(PredictedPose("af2", "q").native_scores(), allow_nan=False)


class TestId:
    def test_the_default_is_model_name_seed_index_and_sample(self):
        assert _pose().id == "of3:cmplx:s2:3"

    def test_a_given_id_wins(self):
        assert _pose(given_id="engine-key-7").id == "engine-key-7"

    def test_the_default_needs_no_structure_file(self):
        assert PredictedPose("boltz2", "q").id == "boltz2:q:s1:1"

    def test_it_follows_the_fields(self):
        pose = _pose()
        pose.sample = 5
        assert pose.id == "of3:cmplx:s2:5"

    def test_given_id_must_be_a_non_empty_string(self):
        for bad in ("", 3):
            with pytest.raises(ValueError, match="given_id"):
                _pose(given_id=bad).validate()


class TestContentHash:
    def test_a_pose_without_a_structure_file_has_no_hash(self):
        pose = PredictedPose("of3", "q", reasons=["structure file not found"])
        with pytest.raises(ValueError, match="no structure file; structure file not found"):
            pose.content_hash()

    def test_it_follows_a_change_of_the_structure_path(self, tmp_path, structure_file):
        other = tmp_path / "other.cif"
        other.write_text("data_other\n#\n", encoding="utf-8")
        pose = _pose(structure_path=structure_file)
        first = pose.content_hash()
        pose.structure_path = other
        assert pose.content_hash() != first
        assert pose.content_hash() == hashlib.sha256(other.read_bytes()).hexdigest()


class TestProvenance:
    def test_it_is_empty_by_default_and_never_shared(self):
        first, second = _pose(), _pose()
        first.provenance["mode"] = "score"
        assert second.provenance == {}

    def test_it_holds_json_scalars(self):
        pose = _pose(provenance={"mode": "score", "receptor_forced": True, "n": 3, "x": None})
        pose.validate()

    def test_a_value_that_is_not_a_json_scalar_is_reported(self):
        with pytest.raises(ValueError, match=r"provenance\['box'\] must be a JSON scalar"):
            _pose(provenance={"box": [1, 2]}).validate()

    def test_a_non_finite_float_is_reported(self):
        with pytest.raises(ValueError, match="JSON cannot hold"):
            _pose(provenance={"t": float("nan")}).validate()


class TestStructureThroughThePoseBase:
    def test_atoms_come_from_the_base_class(self, structure_file):
        pose = _pose(structure_path=structure_file, chain_map={"A": "R"})
        assert set(pose.atoms().chain_id) == {"R", "B"}
        assert pose.atoms() is pose.atoms()
        assert "atoms" not in PredictedPose.__dict__
