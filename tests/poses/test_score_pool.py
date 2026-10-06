"""score_pose_set with n_workers > 1: every start method, pickling, and a pool that cannot start."""

import concurrent.futures
import json
import logging
import multiprocessing
import pickle
from pathlib import Path

import pytest

from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses import score as score_module
from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.frame import superpose_receptor
from binding_metrics.poses.reference import Partner, Reference, ResidueKey
from binding_metrics.poses.score import PoseScoringError, PoseSetResult, score_pose_set
from binding_metrics.testing import make_synthetic_poses
from tests.poses.test_score_frame import build_predicted_poses

pytest.importorskip("biotite.structure")

YCR = Path(__file__).resolve().parents[2] / "data" / "example_linear_p53_1YCR.pdb"
EPITOPE = [25, 26, 50, 51, 54, 57, 58, 61, 62, 67, 70, 71, 72, 73, 75, 91, 93, 94, 96, 99, 100, 104]
START_METHODS = multiprocessing.get_all_start_methods()


def _stub_fails_on_the_second_pose(cif_path, binder_chain=None, receptor_chain=None):
    if Path(cif_path).name == "pose_0001.cif":
        raise ValueError("stub failure")
    return {"value": 1.0}


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setitem(
        METRICS_BY_NAME,
        "stub_fails",
        MetricSpec(
            name="stub_fails",
            import_path=f"{__name__}:_stub_fails_on_the_second_pose",
            description="stub metric of the pool tests",
            input_type="static_structure",
            chain_mode="interface",
            path_arg="cif_path",
            peptide_chain_arg="binder_chain",
            receptor_chain_arg="receptor_chain",
            cost_class="static",
        ),
    )


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_poses()


@pytest.fixture(scope="module")
def reference():
    return Reference(
        YCR,
        "A",
        "B",
        residue_sets={"epitope": [ResidueKey("A", n) for n in EPITOPE]},
        partners={"crystal_peptide": Partner(YCR, chains=["B"])},
    )


def comparable(result):
    """The result without what describes the run: timing and provenance."""
    data = json.loads(result.to_json(include_timing=False))
    data.pop("provenance")
    return data


def use_start_method(monkeypatch, method):
    monkeypatch.setattr(score_module, "_pool_context", lambda: multiprocessing.get_context(method))


CALL = dict(metrics=["delta_sasa_static", "hbonds"], residue_table=True)


@pytest.fixture(scope="module")
def serial(synthetic, reference):
    return score_pose_set(synthetic.poses, "B", "A", reference=reference, **CALL)


class TestStartMethods:
    def test_the_platform_default_gives_the_serial_result(self, synthetic, reference, serial):
        pooled = score_pose_set(synthetic.poses, "B", "A", reference=reference, n_workers=2, **CALL)
        assert comparable(pooled) == comparable(serial)
        assert pooled["provenance"]["run"]["n_workers"] == 2

    @pytest.mark.parametrize("method", START_METHODS)
    def test_each_start_method_gives_the_serial_result(
        self, monkeypatch, synthetic, reference, serial, method
    ):
        use_start_method(monkeypatch, method)
        pooled = score_pose_set(synthetic.poses, "B", "A", reference=reference, n_workers=3, **CALL)
        assert comparable(pooled) == comparable(serial)
        assert pooled.to_json(include_timing=False).count('"status": "failed"') == 0

    def test_spawn_gives_the_serial_result_of_a_set_with_failures(
        self, monkeypatch, synthetic, reference, tmp_path
    ):
        use_start_method(monkeypatch, "spawn")
        corrupt = tmp_path / "corrupt.pdb"
        corrupt.write_text("this is not a structure\n", encoding="utf-8")
        poses = list(synthetic.poses[:3]) + [DockedPose.from_file(corrupt, id="zz_corrupt")]
        pooled = score_pose_set(poses, "B", "A", reference=reference, n_workers=2, **CALL)
        monkeypatch.undo()
        again = score_pose_set(poses, "B", "A", reference=reference, **CALL)
        assert pooled.row("zz_corrupt")["status"] == "failed"
        assert comparable(pooled) == comparable(again)

    def test_spawn_runs_predicted_poses(self, monkeypatch, tmp_path):
        use_start_method(monkeypatch, "spawn")
        poses = build_predicted_poses(tmp_path)
        reference = Reference(YCR, "A", "B")
        pooled = score_pose_set(
            poses,
            "B",
            "A",
            metrics=["delta_sasa_static", "prediction"],
            reference=reference,
            n_workers=2,
        )
        monkeypatch.undo()
        serial = score_pose_set(
            poses, "B", "A", metrics=["delta_sasa_static", "prediction"], reference=reference
        )
        assert comparable(pooled) == comparable(serial)
        assert pooled["poses"][0]["prediction"]["binder_avg_plddt"] > 60

    def test_spawn_raises_the_first_failure_in_id_order(self, monkeypatch, synthetic, stub):
        use_start_method(monkeypatch, "spawn")
        with pytest.raises(PoseScoringError, match="'backend_a-s2'.*stub failure") as info:
            score_pose_set(
                synthetic.poses,
                "B",
                "A",
                metrics=["stub_fails"],
                on_error="raise",
                n_workers=2,
            )
        assert (info.value.pose_id, info.value.stage) == ("backend_a-s2", "stub_fails")

    def test_a_pool_of_one_worker_or_one_pose_runs_in_this_process(
        self, monkeypatch, synthetic, serial
    ):
        def refuse():
            raise AssertionError("no pool is needed for one worker or one pose")

        monkeypatch.setattr(score_module, "_pool_context", refuse)
        score_pose_set(synthetic.poses, "B", "A", n_workers=1)
        score_pose_set(synthetic.poses[:1], "B", "A", n_workers=4)


class TestPickling:
    """Everything that crosses the process boundary survives pickle."""

    @pytest.fixture
    def captured(self, monkeypatch, synthetic, reference):
        seen = {}

        def capture(items, ctx, n_workers, outcomes):
            seen.update(items=list(items), ctx=ctx, n_workers=n_workers)
            raise score_module._PoolUnavailableError("captured")

        monkeypatch.setattr(score_module, "_score_in_pool", capture)
        result = score_pose_set(
            synthetic.poses,
            "B",
            "A",
            reference=reference,
            metrics=["delta_sasa_static", "hbonds", "receptor_quality", "md_implicit"],
            heavy="skip",
            n_workers=2,
            preflight="off",
            residue_table=True,
        )
        return seen, result

    def test_the_context_and_the_poses_the_workers_receive_pickle(self, captured):
        seen, _ = captured
        ctx = pickle.loads(pickle.dumps(seen["ctx"]))
        assert ctx.binder_chain == "B" and ctx.target_chain == "A"
        assert ctx.options == seen["ctx"].options
        assert list(ctx.variants) == ["reference"]
        assert [p.name for p in ctx.plans] == [
            "delta_sasa_static",
            "hbonds",
            "receptor_quality",
            "md_implicit",
        ]
        assert all(plan.spec.capabilities is None for plan in ctx.plans)
        poses = pickle.loads(pickle.dumps(seen["items"]))
        assert [pose.id for _, pose in poses] == [pose.id for _, pose in seen["items"]]

    def test_the_specs_in_the_context_still_name_the_function_to_call(self, captured):
        seen, _ = captured
        spec = pickle.loads(pickle.dumps(seen["ctx"])).plans[0].spec
        assert spec.import_path == METRICS_BY_NAME["delta_sasa_static"].import_path
        assert spec.path_arg == "cif_path" and spec.cost_class == "static"
        # the registry itself is untouched: only the copy sent to a worker lost its limits
        assert METRICS_BY_NAME["interface"].capabilities is not None

    def test_an_unavailable_pool_falls_back_to_the_serial_result(self, captured, synthetic):
        _, result = captured
        assert [row["status"] for row in result["poses"]] == ["ok"] * 6

    def test_a_pose_with_its_structure_read_pickles_and_reads_the_same_atoms(self, synthetic):
        pose = synthetic.poses[0]
        pose.atoms()
        clone = pickle.loads(pickle.dumps(pose))
        assert clone.id == pose.id and clone.content_hash() == pose.content_hash()
        assert (clone.atoms().coord == pose.atoms().coord).all()

    def test_a_pose_from_a_file_and_a_predicted_pose_pickle(self, tmp_path):
        from_file = DockedPose.from_file(YCR, id="crystal", backend="x")
        clone = pickle.loads(pickle.dumps(from_file))
        assert clone.id == "crystal" and clone.atoms().array_length() == 818
        for pose in build_predicted_poses(tmp_path):
            pose.atoms()
            twin = pickle.loads(pickle.dumps(pose))
            assert twin.id == pose.id and twin.native_scores() == pose.native_scores()
            assert (twin.plddt_per_atom == pose.plddt_per_atom).all()

    def test_a_reference_a_frame_and_the_options_pickle(self, synthetic, reference):
        clone = pickle.loads(pickle.dumps(reference))
        assert clone.content_hash() == reference.content_hash()
        frame = superpose_receptor(synthetic.poses[1], reference, receptor_chain="A")
        again = pickle.loads(pickle.dumps(frame))
        assert again.to_dict() == frame.to_dict()
        options = score_module.PoseSetOptions(agreement={"cluster_cutoffs_angstrom": [0.5]})
        assert pickle.loads(pickle.dumps(options)) == options

    def test_an_outcome_and_a_result_pickle(self, synthetic, reference, serial):
        ctx_items = []

        def capture(items, ctx, n_workers, outcomes):
            ctx_items.append((items, ctx))
            raise score_module._PoolUnavailableError("captured")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(score_module, "_score_in_pool", capture)
            score_pose_set(synthetic.poses[:2], "B", "A", reference=reference, n_workers=2)
        items, ctx = ctx_items[0]
        outcome = score_module._pose_stage(items[1][1], items[1][0], ctx)
        clone = pickle.loads(pickle.dumps(outcome))
        assert clone.row == outcome.row and clone.pose_id == outcome.pose_id
        assert (clone.metrics_atoms.coord == outcome.metrics_atoms.coord).all()
        copy = pickle.loads(pickle.dumps(serial))
        assert isinstance(copy, PoseSetResult)
        assert copy.to_json(include_timing=False) == serial.to_json(include_timing=False)


def _broken_initializer(ctx):
    raise RuntimeError("this worker cannot start")


class TestPoolThatCannotStart:
    def test_an_executor_that_cannot_be_made_gives_a_warning_and_the_serial_result(
        self, monkeypatch, caplog, synthetic, serial, reference
    ):
        def refuse(*args, **kwargs):
            raise OSError("no more processes")

        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", refuse)
        with caplog.at_level(logging.WARNING, logger=score_module.logger.name):
            result = score_pose_set(
                synthetic.poses, "B", "A", reference=reference, n_workers=2, **CALL
            )
        assert comparable(result) == comparable(serial)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("could not be used" in w and "no more processes" in w for w in warnings)

    def test_a_worker_that_fails_at_start_gives_a_warning_and_the_serial_result(
        self, monkeypatch, caplog, synthetic, serial, reference
    ):
        monkeypatch.setattr(score_module, "_init_worker", _broken_initializer)
        with caplog.at_level(logging.WARNING, logger=score_module.logger.name):
            result = score_pose_set(
                synthetic.poses, "B", "A", reference=reference, n_workers=2, **CALL
            )
        assert comparable(result) == comparable(serial)
        assert any("could not be used" in r.getMessage() for r in caplog.records)

    def test_a_pool_that_works_logs_no_such_warning(self, caplog, synthetic, reference):
        with caplog.at_level(logging.WARNING, logger=score_module.logger.name):
            score_pose_set(synthetic.poses[:3], "B", "A", reference=reference, n_workers=2)
        assert not any("could not be used" in r.getMessage() for r in caplog.records)

    def test_a_poses_failure_is_not_mistaken_for_a_start_problem(
        self, monkeypatch, caplog, synthetic, stub
    ):
        with caplog.at_level(logging.WARNING, logger=score_module.logger.name):
            with pytest.raises(PoseScoringError):
                score_pose_set(
                    synthetic.poses,
                    "B",
                    "A",
                    metrics=["stub_fails"],
                    on_error="raise",
                    n_workers=2,
                )
        assert not any("could not be used" in r.getMessage() for r in caplog.records)
