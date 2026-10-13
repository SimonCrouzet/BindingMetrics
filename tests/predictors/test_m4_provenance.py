"""The facts of a request on the poses of a session: ``provenance``, ``receptor_forced``.

Every sample is parsed from the synthetic output of the four models (``synth_<model>``) by the
real adapter, through a stub runner that writes it: no model, GPU or network is involved.
``SynthRunner`` is also the runner of ``test_m4_manifest``.
"""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from binding_metrics import predictors
from binding_metrics.predictors import receptor_forced, receptor_templated
from binding_metrics.predictors.of3 import OpenFold3Parser
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import (
    KEY_FORMAT,
    MODES,
    PredictionRequest,
    PredictionStore,
    file_sha256,
)
from tests.predictors import (
    synth,
    synth_af2,
    synth_boltz2,
    synth_of3,
    synth_protenix,
    synth_stub,
)
from tests.predictors.synth_stub import StubParser

SEQUENCES = {"A": "ACDEFGH", "B": "KLM"}


def truth_for(seed_index: int = 1, sample: int = 1, ranking_score: float = 0.5):
    """The synthetic complex of one sample; its pLDDT and ranking score tell samples apart."""
    base = synth.synthetic_complex(plddt_shift=float(sample - 1))
    return dataclasses.replace(base, scalars={**base.scalars, "ranking_score": ranking_score})


class SynthRunner(PredictionRunner):
    """A runner named after a real model that writes that model's synthetic output.

    It writes ``positions`` seed positions (default ``len(request.seeds)``; one for a model
    without a seed dimension) of ``request.num_samples`` samples each. ``score(seed_index,
    sample)`` gives the ranking score of a sample. ``runs`` lists the names it was asked to run.
    """

    def __init__(self, model, writer, *, version="1.0", score=None, positions=None):
        self.name = model
        self.writer = writer
        self._version = version
        self.score = score or (lambda seed_index, sample: 0.5)
        self.positions = positions
        self.runs = []

    def prepare(self, request, work_dir):
        return Path(work_dir)

    def run(self, request, work_dir):
        self.runs.append(request.name)
        positions = len(request.seeds) if self.positions is None else self.positions
        if not getattr(self.writer, "SUPPORTS_SEED_INDEX", True):
            positions = 1
        for seed_index in range(1, max(positions, 1) + 1):
            for sample in range(1, request.num_samples + 1):
                self.writer.write_prediction(
                    work_dir,
                    request.name,
                    truth_for(seed_index, sample, self.score(seed_index, sample)),
                    seed_index=seed_index,
                    sample=sample,
                )
        return Path(work_dir)

    def version(self):
        return self._version


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "store")


def make_request(model, name="design", **fields):
    fields.setdefault("sequences", SEQUENCES)
    fields.setdefault("mode", "predict")
    fields.setdefault("num_samples", 2)
    return PredictionRequest(model, name, **fields)


# ---------------------------------------------------------------------------- the two functions

#: model, mode, receptor_forced, receptor_templated
TRUTH_TABLE = [
    (model, mode, forced and model == "boltz2", mode != "predict")
    for model in ("af2", "boltz2", "of3", "protenix")
    for mode, forced in (
        ("predict", False),
        ("refold", False),
        ("score", False),
        ("refold-lock", True),
        ("score-lock", True),
    )
]


class TestReceptorForcedAndTemplated:
    @pytest.mark.parametrize("model, mode, forced, templated", TRUTH_TABLE)
    def test_the_modes_of_the_four_models(self, model, mode, forced, templated):
        request = SimpleNamespace(model=model, mode=mode)  # modes compared as strings
        assert receptor_forced(request) is forced
        assert receptor_templated(request) is templated

    def test_only_boltz2_forces(self):
        forced = {row[0] for row in TRUTH_TABLE if row[2]}
        assert forced == {"boltz2"}

    def test_an_openfold3_refold_pose_reads_templated_but_not_forced(self):
        request = SimpleNamespace(model="of3", mode="refold")
        assert (receptor_forced(request), receptor_templated(request)) == (False, True)

    def test_an_unknown_model_gives_none(self):
        for mode in ("predict", "refold", "score-lock"):
            request = SimpleNamespace(model="chai1", mode=mode)
            assert receptor_forced(request) is None
            assert receptor_templated(request) is None

    def test_an_unknown_mode_gives_none(self):
        request = SimpleNamespace(model="boltz2", mode="dock")
        assert receptor_forced(request) is None
        assert receptor_templated(request) is None

    @pytest.mark.parametrize("mode", [m for m in MODES if m != "refold-lock"])
    def test_a_real_request_is_read_the_same_way(self, mode):
        request = make_request("boltz2", mode=mode)
        assert receptor_forced(request) is (mode == "score-lock")
        assert receptor_templated(request) is (mode != "predict")

    def test_the_functions_are_exported_from_the_package(self):
        from binding_metrics.predictors import session

        assert predictors.receptor_forced is session.receptor_forced
        assert predictors.receptor_templated is session.receptor_templated
        assert {"receptor_forced", "receptor_templated"} <= set(predictors.__all__)
        assert predictors._EXPORTS["receptor_forced"] == "binding_metrics.predictors.session"


# ---------------------------------------------------------------------------- the provenance

#: model, writer module, mode, seeds of the request (the values the writer names the seeds)
MODELS = [
    ("of3", synth_of3, "refold", (9, 10, 11)),
    ("boltz2", synth_boltz2, "score-lock", (42,)),
    ("protenix", synth_protenix, "predict", (9, 10)),
    ("af2", synth_af2, "predict", (99, 100)),
]


def make_session(store, runner):
    return PredictionSession(store, [runner])


class TestProvenanceOfTheFourModels:
    @pytest.mark.parametrize("model, writer, mode, seeds", MODELS)
    def test_every_sample_carries_the_facts_of_its_request(self, store, model, writer, mode, seeds):
        runner = SynthRunner(model, writer)
        session = make_session(store, runner)
        request = make_request(model, mode=mode, seeds=seeds, model_version="2.0")
        for seed_index, seed in enumerate(seeds, start=1):
            for sample in (1, 2):
                pose = session.record(request, seed_index=seed_index, sample=sample)
                assert pose.provenance == {
                    "backend": model,
                    "mode": mode,
                    "receptor_forced": model == "boltz2" and mode == "score-lock",
                    "receptor_templated": mode != "predict",
                    "seed": seed,
                    "seed_index": seed_index,
                    "sample": sample,
                    "request_key": request.key(),
                    "model_version": "2.0",
                    "request_name": "design",
                }
        assert runner.runs == ["design"]

    @pytest.mark.parametrize("model, writer, mode, seeds", MODELS)
    def test_the_record_still_validates_and_the_facts_are_json(
        self, store, model, writer, mode, seeds
    ):
        pose = make_session(store, SynthRunner(model, writer)).record(
            make_request(model, mode=mode, seeds=seeds)
        )
        pose.validate()
        assert json.loads(json.dumps(pose.provenance)) == pose.provenance

    def test_a_lock_mode_of_boltz2_is_forced_and_one_of_openfold3_is_not(self, store):
        boltz = make_session(store, SynthRunner("boltz2", synth_boltz2)).record(
            make_request("boltz2", mode="score-lock", seeds=(7,))
        )
        assert boltz.provenance["receptor_forced"] is True
        assert boltz.provenance["receptor_templated"] is True
        assert boltz.provenance["seed"] == 7
        refold = make_session(store, SynthRunner("of3", synth_of3)).record(
            make_request("of3", mode="refold", seeds=(9,))
        )
        assert refold.provenance["receptor_forced"] is False
        assert refold.provenance["receptor_templated"] is True

    def test_a_predict_request_is_not_templated(self, store):
        pose = make_session(store, SynthRunner("of3", synth_of3)).record(
            make_request("of3", mode="predict", seeds=(9,))
        )
        assert pose.provenance["receptor_forced"] is False
        assert pose.provenance["receptor_templated"] is False


def stub_session(store, runner):
    """A session on the made-up ``stub`` model, whose adapter reports no seed value."""
    return PredictionSession(store, [runner], {"stub": StubParser()})


UNSORTED = "the adapter does not report it and the request seeds are not sorted"


class TestTheSeedValue:
    """The seed is the output's when the adapter reports it, else from sorted request seeds."""

    def test_openfold3_seeds_9_10_11_are_values_not_strings_to_sort(self, store):
        session = make_session(store, SynthRunner("of3", synth_of3))
        request = make_request("of3", seeds=(9, 10, 11))
        seeds = [session.record(request, seed_index=i).provenance["seed"] for i in (1, 2, 3)]
        assert seeds == [9, 10, 11]
        assert all(isinstance(seed, int) for seed in seeds)

    def test_a_request_with_its_seeds_out_of_order_still_names_the_right_seed(self, store):
        """The adapters count seed directories in numeric order, whatever order was asked."""
        session = make_session(store, SynthRunner("of3", synth_of3))
        request = make_request("of3", seeds=(11, 9, 10))
        seeds = [session.record(request, seed_index=i).provenance["seed"] for i in (1, 2, 3)]
        assert seeds == [9, 10, 11]

    def test_openfold3_seeds_42_7_11_are_read_from_the_directories(self, store, monkeypatch):
        """seed_7, seed_11 and seed_42 are positions 1, 2 and 3; the request says 42, 7, 11."""
        monkeypatch.setattr(synth_of3, "seed_value", lambda seed_index: (7, 11, 42)[seed_index - 1])
        session = make_session(store, SynthRunner("of3", synth_of3))
        request = make_request("of3", seeds=(42, 7, 11), num_samples=1)
        poses = [session.record(request, seed_index=i) for i in (1, 2, 3)]
        assert [pose.provenance["seed"] for pose in poses] == [7, 11, 42]
        assert [pose.extras["seed_value"] for pose in poses] == ["7", "11", "42"]
        assert all("seed value unknown" not in " ".join(pose.reasons) for pose in poses)

    def test_protenix_reports_its_seed_directories(self, store):
        session = make_session(store, SynthRunner("protenix", synth_protenix))
        request = make_request("protenix", seeds=(10, 9))
        assert [session.record(request, seed_index=i).provenance["seed"] for i in (1, 2)] == [9, 10]

    def test_colabfold_reports_the_seed_of_its_file_names(self, store):
        session = make_session(store, SynthRunner("af2", synth_af2))
        request = make_request("af2", seeds=(100, 99))
        poses = [session.record(request, seed_index=i) for i in (1, 2)]
        assert [pose.provenance["seed"] for pose in poses] == [99, 100]
        assert [pose.extras["seed_value"] for pose in poses] == ["99", "100"]

    def test_a_request_with_no_seed_takes_the_value_from_the_output(self, store):
        """OpenFold3 generates the seeds when ``num_model_seeds`` is asked: the request has none."""
        session = make_session(store, SynthRunner("of3", synth_of3, positions=2))
        request = make_request("of3", seeds=(), num_samples=1)
        assert session.record(request, seed_index=2).provenance["seed"] == 10

    def test_the_seed_of_the_output_wins_over_the_request(self, store):
        session = make_session(store, SynthRunner("of3", synth_of3))
        pose = session.record(make_request("of3", seeds=(1,), num_samples=1))
        assert pose.provenance["seed"] == 9

    def test_the_single_seed_of_a_boltz2_request(self, store):
        session = make_session(store, SynthRunner("boltz2", synth_boltz2))
        request = make_request("boltz2", mode="score", seeds=(44,))
        pose = session.record(request)
        assert pose.provenance["seed"] == 44
        assert not any("seed value unknown" in reason for reason in pose.reasons)

    def test_an_adapter_without_a_seed_value_uses_sorted_request_seeds(self, store):
        session = stub_session(store, SynthRunner("stub", synth_stub))
        request = make_request("stub", seeds=(3, 8, 20), num_samples=1)
        poses = [session.record(request, seed_index=i) for i in (1, 2, 3)]
        assert [pose.provenance["seed"] for pose in poses] == [3, 8, 20]
        assert not any("seed value unknown" in reason for pose in poses for reason in pose.reasons)

    @pytest.mark.parametrize("seeds", [(20, 3, 8), (3, 20, 8), (3, 3, 8)])
    def test_unsorted_request_seeds_give_no_seed_and_say_why(self, store, seeds):
        """Position 1 of the output is not the first seed asked for: do not guess."""
        session = stub_session(store, SynthRunner("stub", synth_stub))
        request = make_request("stub", seeds=seeds, num_samples=1)
        for seed_index in (1, 2, 3):
            pose = session.record(request, seed_index=seed_index)
            assert pose.provenance["seed"] is None
            assert f"seed value unknown: {UNSORTED}" in pose.reasons
            assert pose.provenance["seed_index"] == seed_index  # the rest is still recorded

    def test_a_position_beyond_the_request_seeds_gives_no_seed_and_says_why(self, store):
        session = stub_session(store, SynthRunner("stub", synth_stub, positions=2))
        request = make_request("stub", seeds=(3,), num_samples=1)
        assert session.record(request, seed_index=1).provenance["seed"] == 3
        pose = session.record(request, seed_index=2)
        assert pose.provenance["seed"] is None
        assert (
            "seed value unknown: the adapter does not report it and the request has no seed "
            "at position 2"
        ) in pose.reasons

    def test_a_request_without_seeds_and_an_adapter_without_seed_value_gives_none(self, store):
        session = stub_session(store, SynthRunner("stub", synth_stub))
        pose = session.record(make_request("stub", seeds=(), num_samples=1))
        assert pose.provenance["seed"] is None
        assert any(reason.startswith("seed value unknown") for reason in pose.reasons)

    def test_the_reason_is_added_once_per_record(self, store):
        session = stub_session(store, SynthRunner("stub", synth_stub))
        request = make_request("stub", seeds=(8, 3), num_samples=1)
        first = session.record(request)
        again = session.record(request)
        assert again is first
        assert sum(r.startswith("seed value unknown") for r in first.reasons) == 1

    def test_the_other_facts_do_not_depend_on_the_seed(self, store):
        session = stub_session(store, SynthRunner("stub", synth_stub))
        request = make_request("stub", seeds=(8, 3), num_samples=1)
        pose = session.record(request)
        assert pose.provenance["backend"] == "stub"
        assert pose.provenance["request_key"] == request.key()
        assert pose.provenance["receptor_forced"] is None  # an unknown model
        pose.validate()


class TestTheAdaptersThatReportTheSeed:
    def test_openfold3_protenix_and_colabfold_set_seed_value_and_the_others_do_not(self, tmp_path):
        from binding_metrics.predictors.registry import get_parser

        truth = truth_for()
        found = {}
        for model, writer in (
            ("of3", synth_of3),
            ("protenix", synth_protenix),
            ("af2", synth_af2),
            ("boltz2", synth_boltz2),
        ):
            folder = tmp_path / model
            folder.mkdir()
            # the second seed position alone on disk, so it is the first one the adapter finds
            writer.write_prediction(
                folder, "design", truth, seed_index=1 if model == "boltz2" else 2
            )
            found[model] = get_parser(model).load(folder, "design").extras.get("seed_value")
        assert found == {"of3": "10", "protenix": "10", "af2": "100", "boltz2": None}

    def test_an_alphafold2_layout_names_no_seed(self, tmp_path):
        from binding_metrics.predictors.registry import get_parser

        synth_af2.write_alphafold(tmp_path, truth_for())
        record = get_parser("af2").load(tmp_path, "design")
        assert record.extras["layout"] == "alphafold"
        assert "seed_value" not in record.extras


class TestTheModelVersion:
    def test_a_request_without_a_version_takes_the_one_the_runner_reported(self, store):
        session = make_session(store, SynthRunner("of3", synth_of3, version="0.5.0"))
        pose = session.record(make_request("of3", seeds=(9,)))
        assert pose.provenance["model_version"] == "0.5.0"

    def test_the_version_of_the_request_wins(self, store):
        session = make_session(store, SynthRunner("of3", synth_of3, version="0.5.0"))
        pose = session.record(make_request("of3", seeds=(9,), model_version="0.4.1"))
        assert pose.provenance["model_version"] == "0.4.1"

    def test_no_version_anywhere_is_an_empty_string(self, store):
        runner = SynthRunner("of3", synth_of3, version=None)
        pose = make_session(store, runner).record(make_request("of3", seeds=(9,)))
        assert pose.provenance["model_version"] == ""


class TestTheMsaFacts:
    def _facts(self, store, **fields):
        session = make_session(store, SynthRunner("of3", synth_of3))
        provenance = session.record(make_request("of3", seeds=(9,), **fields)).provenance
        return {key: value for key, value in provenance.items() if key.startswith("msa")}

    def test_a_request_without_an_msa_request_has_no_msa_keys(self, store):
        assert self._facts(store, options={"use_msa_server": False}) == {}

    def test_a_policy_alone(self, store):
        assert self._facts(store, options={"msa_policy": "forbid-server"}) == {
            "msa_policy": "forbid-server",
            "msa_receptor": None,
            "msa_binder": "single",
        }

    def test_a_single_sequence_receptor(self, store):
        facts = self._facts(
            store, options={"msa_policy": "forbid-server", "receptor_msa": "single"}
        )
        assert facts == {
            "msa_policy": "forbid-server",
            "msa_receptor": "single",
            "msa_binder": "single",
        }

    def test_a_receptor_msa_file_is_named_by_the_start_of_its_content_hash(self, store, tmp_path):
        a3m = tmp_path / "receptor.a3m"
        a3m.write_text(">query\nACDEFGH\n", encoding="utf-8")
        facts = self._facts(store, extra_files={"receptor_msa": a3m})
        assert facts == {
            "msa_policy": "default",
            "msa_receptor": file_sha256(a3m)[:12],
            "msa_binder": "single",
        }
        assert len(facts["msa_receptor"]) == 12

    def test_the_same_content_under_another_path_gives_the_same_name(self, store, tmp_path):
        first, second = tmp_path / "one.a3m", tmp_path / "sub" / "two.a3m"
        second.parent.mkdir()
        for path in (first, second):
            path.write_text(">query\nACDEFGH\n", encoding="utf-8")
        assert (
            self._facts(store, extra_files={"receptor_msa": first})["msa_receptor"]
            == self._facts(store, extra_files={"receptor_msa": second})["msa_receptor"]
        )


class TestWhatTheSessionLeavesAlone:
    def test_a_provenance_key_that_the_adapter_set_is_kept(self, store):
        class Annotating(OpenFold3Parser):
            def complete(self, record):
                record = super().complete(record)
                record.provenance["note"] = "set by the adapter"
                return record

        session = PredictionSession(store, [SynthRunner("of3", synth_of3)], {"of3": Annotating()})
        pose = session.record(make_request("of3", seeds=(9,)))
        assert pose.provenance["note"] == "set by the adapter"
        assert pose.provenance["backend"] == "of3"

    def test_a_second_ask_returns_the_same_record_with_the_same_facts(self, store):
        session = make_session(store, SynthRunner("of3", synth_of3))
        request = make_request("of3", seeds=(9,))
        first = session.record(request)
        facts = dict(first.provenance)
        second = session.record(request)
        assert second is first and second.provenance == facts

    def test_adopted_outputs_carry_the_facts_too(self, store, tmp_path):
        outputs = tmp_path / "mine"
        outputs.mkdir()
        synth_of3.write_prediction(outputs, "design", truth_for(1, 1), seed_index=2)  # seed_10
        session = PredictionSession(store)  # no runner: parse-only usage
        request = make_request("of3", seeds=(10,), num_samples=1)
        session.adopt(request, outputs)
        pose = session.record(request)
        assert pose.provenance["backend"] == "of3"
        assert pose.provenance["seed"] == 10
        assert pose.provenance["model_version"] == ""
        assert pose.provenance["request_key"] == request.key()


class TestTheKeyIsUnchanged:
    """Nothing in the provenance work may change the key of an existing request."""

    def test_the_format_and_the_canonical_fields_are_the_ones_of_before(self):
        request = make_request("of3", seeds=(1,))
        assert KEY_FORMAT == 1
        assert set(request.canonical()) == {
            "format",
            "model",
            "model_version",
            "mode",
            "file_sha256",
            "binder_chain",
            "receptor_chain",
            "sequences",
            "seeds",
            "num_samples",
            "options",
        }

    def test_a_request_built_the_old_way_keeps_its_key(self):
        request = PredictionRequest(
            "of3",
            "pin",
            mode="predict",
            sequences={"A": "ACDEFGHIK", "B": "LMNPQ"},
            seeds=(1, 2),
            num_samples=3,
            model_version="0.5.0",
            options={"use_msa_server": False, "presets": ["a"]},
        )
        assert request.key() == "f2f95e7b430f98b19494f44ec07382aa56401d20d8a313d1f9d5f3c10945f0f9"

    def test_recording_does_not_change_the_key(self, store):
        session = make_session(store, SynthRunner("of3", synth_of3))
        request = make_request("of3", seeds=(9,), model_version="0.5.0")
        before = request.key()
        session.record(request)
        assert request.key() == before
        assert request.model_version == "0.5.0"
