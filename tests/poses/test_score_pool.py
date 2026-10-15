"""score_pose_set with n_workers > 1: start methods, pickling, and a pool that cannot start."""

import concurrent.futures
import inspect
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


CALL = dict(metrics=["delta_sasa_static", "hbonds"], residue_table=True)


@pytest.fixture(scope="module")
def serial(synthetic, reference):
    return score_pose_set(synthetic.poses, "B", "A", reference=reference, **CALL)


class TestStartMethods:
    @pytest.mark.slow
    def test_the_default_is_spawn_and_it_is_recorded_with_the_run(
        self, monkeypatch, synthetic, reference, serial
    ):
        assert inspect.signature(score_pose_set).parameters["start_method"].default == "spawn"
        asked = []
        real = score_module._pool_context
        monkeypatch.setattr(
            score_module, "_pool_context", lambda method=None: asked.append(method) or real(method)
        )
        pooled = score_pose_set(synthetic.poses, "B", "A", reference=reference, n_workers=2, **CALL)
        assert asked == ["spawn"]
        assert pooled["provenance"]["run"]["start_method"] == "spawn"
        assert pooled["provenance"]["run"]["n_workers"] == 2
        assert comparable(pooled) == comparable(serial)

    # the platform default is fork on Linux, and a threaded parent makes it warn on Python 3.12
    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_none_is_the_platform_default_and_the_recorded_name_is_that_method(
        self, monkeypatch, synthetic, reference, serial
    ):
        asked = []
        real = score_module._pool_context
        monkeypatch.setattr(
            score_module, "_pool_context", lambda method=None: asked.append(method) or real(method)
        )
        pooled = score_pose_set(
            synthetic.poses, "B", "A", reference=reference, n_workers=2, start_method=None, **CALL
        )
        assert asked == [None]
        default = multiprocessing.get_context().get_start_method()
        assert pooled["provenance"]["run"]["start_method"] == default
        assert comparable(pooled) == comparable(serial)

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["spawn", "forkserver"])
    def test_spawn_and_forkserver_give_the_serial_result(
        self, synthetic, reference, serial, method
    ):
        if method not in START_METHODS:
            pytest.skip(f"{method} is not available on this platform")
        pooled = score_pose_set(
            synthetic.poses,
            "B",
            "A",
            reference=reference,
            n_workers=3,
            start_method=method,
            **CALL,
        )
        assert comparable(pooled) == comparable(serial)
        assert pooled["provenance"]["run"]["start_method"] == method
        assert pooled.to_json(include_timing=False).count('"status": "failed"') == 0

    # a threaded parent (pytest has threads) makes the fork of Python 3.12 warn
    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fork_runs_where_the_platform_has_it_and_gives_the_serial_result(
        self, synthetic, reference, serial
    ):
        if "fork" not in START_METHODS:
            pytest.skip("fork is not available on this platform")
        pooled = score_pose_set(
            synthetic.poses,
            "B",
            "A",
            reference=reference,
            n_workers=2,
            start_method="fork",
            **CALL,
        )
        assert comparable(pooled) == comparable(serial)
        assert pooled["provenance"]["run"]["start_method"] == "fork"

    @pytest.mark.slow
    def test_the_method_is_not_part_of_the_inputs_or_of_the_parameters(
        self, synthetic, reference, serial
    ):
        other = score_pose_set(
            synthetic.poses[:3], "B", "A", reference=reference, start_method="forkserver", **CALL
        )
        again = score_pose_set(synthetic.poses[:3], "B", "A", reference=reference, **CALL)
        assert other.input_key == again.input_key
        assert "start_method" not in other["parameters"]
        assert other["provenance"]["run"]["start_method"] == "forkserver"
        assert again["provenance"]["run"]["start_method"] == "spawn"
        assert comparable(other) == comparable(again)

    @pytest.mark.parametrize("name", ["Spawn", "threads", "", "fork ", 3, True, ["spawn"]])
    def test_an_invalid_name_raises_before_any_structure_is_read(self, tmp_path, name):
        corrupt = tmp_path / "corrupt.pdb"
        corrupt.write_text("not a structure\n", encoding="utf-8")

        class Unreadable(DockedPose):
            def atoms(self):  # pragma: no cover - reached only when the check is missing
                raise AssertionError("a structure was read before start_method was checked")

        poses = [Unreadable(corrupt, given_id="p1"), Unreadable(corrupt, given_id="p2")]
        with pytest.raises(ValueError, match="start_method must be one of"):
            score_pose_set(poses, "B", "A", n_workers=2, start_method=name)
        with pytest.raises(ValueError, match="start_method must be one of"):
            score_pose_set(poses, "B", "A", start_method=name)  # also without a pool

    def test_a_method_the_platform_lacks_is_a_clear_error_when_a_pool_is_asked_for(
        self, monkeypatch, synthetic
    ):
        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
        with pytest.raises(ValueError, match=r"'fork' is not available on this platform.*spawn"):
            score_pose_set(synthetic.poses[:2], "B", "A", n_workers=2, start_method="fork")
        # one worker never starts a pool, so the method is not needed
        score_pose_set(synthetic.poses[:2], "B", "A", n_workers=1, start_method="fork")

    @pytest.mark.slow
    def test_spawn_gives_the_serial_result_of_a_set_with_failures(
        self, synthetic, reference, tmp_path
    ):
        corrupt = tmp_path / "corrupt.pdb"
        corrupt.write_text("this is not a structure\n", encoding="utf-8")
        poses = list(synthetic.poses[:3]) + [DockedPose.from_file(corrupt, id="zz_corrupt")]
        pooled = score_pose_set(poses, "B", "A", reference=reference, n_workers=2, **CALL)
        again = score_pose_set(poses, "B", "A", reference=reference, **CALL)
        assert pooled.row("zz_corrupt")["status"] == "failed"
        assert comparable(pooled) == comparable(again)

    @pytest.mark.slow
    def test_spawn_runs_predicted_poses(self, tmp_path):
        poses = build_predicted_poses(tmp_path)
        reference = Reference(YCR, "A", "B")
        call = dict(metrics=["delta_sasa_static", "prediction"], reference=reference)
        pooled = score_pose_set(poses, "B", "A", n_workers=2, **call)
        serial = score_pose_set(poses, "B", "A", **call)
        assert comparable(pooled) == comparable(serial)
        assert pooled["poses"][0]["prediction"]["binder_avg_plddt"] > 60

    @pytest.mark.slow
    def test_spawn_raises_the_first_failure_in_id_order(self, synthetic, stub):
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

    @pytest.mark.slow
    def test_a_pool_of_one_worker_or_one_pose_runs_in_this_process(
        self, monkeypatch, synthetic, serial
    ):
        def refuse(*args):
            raise AssertionError("no pool is needed for one worker or one pose")

        monkeypatch.setattr(score_module, "_pool_context", refuse)
        score_pose_set(synthetic.poses, "B", "A", n_workers=1)
        score_pose_set(synthetic.poses[:1], "B", "A", n_workers=4)


class TestPickling:
    """Everything that crosses the process boundary survives pickle."""

    @pytest.fixture
    def captured(self, monkeypatch, synthetic, reference):
        seen = {}

        def capture(items, ctx, n_workers, start_method, outcomes):
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

    @pytest.mark.slow
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
        assert ctx.plans == seen["ctx"].plans
        poses = pickle.loads(pickle.dumps(seen["items"]))
        assert [pose.id for _, pose in poses] == [pose.id for _, pose in seen["items"]]

    @pytest.mark.slow
    def test_the_specs_in_the_context_keep_their_declared_limits(self, captured):
        seen, _ = captured
        plans = pickle.loads(pickle.dumps(seen["ctx"])).plans
        spec = plans[0].spec
        assert spec.import_path == METRICS_BY_NAME["delta_sasa_static"].import_path
        assert spec.path_arg == "cif_path" and spec.cost_class == "static"
        assert spec.capabilities == METRICS_BY_NAME["delta_sasa_static"].capabilities
        assert spec.capabilities is not None and spec.capabilities.needs == {"receptor_chain"}

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_an_outcome_and_a_result_pickle(self, synthetic, reference, serial):
        ctx_items = []

        def capture(items, ctx, n_workers, start_method, outcomes):
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


@pytest.mark.slow
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
