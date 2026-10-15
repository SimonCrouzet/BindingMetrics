"""Strict JSON export of the per-sample reports of ``run_batch`` and the batch command (#220).

Every JSON report that a batch writes or extends takes the ``nonfinite`` mode of the run: the
worker that writes it, and the OpenFold3 and prediction steps that add their block afterwards.
"""

import json
import math
import sys

import numpy as np
import pytest

from binding_metrics.cli import batch
from binding_metrics.protocols.report import _json_default, write_report
from tests.test_feat_c_support import EXAMPLE_1YCR, StubOpenFold
from tests.test_t3_pipeline_json import LISTED, pipeline_results
from tests.test_t3_support import (  # noqa: F401  (the fixture is used by name)
    blank,
    read,
    restore_package_logger,
    strict_loads,
)

pytestmark = pytest.mark.usefixtures("restore_package_logger")


class TestBatch:
    @staticmethod
    def _worker(tmp_path, **overrides):
        kwargs = dict(
            input_path=EXAMPLE_1YCR,
            output_dir=tmp_path,
            sample_id="s1",
            skip_prep=True,
            ph=7.4,
            keep_water=False,
            canonicalize=False,
            skip_relax=True,
            md_duration_ps=0.0,
            device="cuda",
            peptide_chain=None,
            receptor_chain=None,
            metrics=frozenset(),
            energy_modes=("relaxed",),
            openfold_mode="score",
            openfold_conda_env=None,
            log_file=None,
        )
        kwargs.update(overrides)
        return kwargs

    def test_the_worker_writes_the_report_in_the_mode_it_is_given(self, tmp_path, monkeypatch):
        kept_results, nulled_results = pipeline_results("s1"), pipeline_results("s1")
        monkeypatch.setattr(batch, "run_pipeline", lambda **_: kept_results)
        batch._run_one(**self._worker(tmp_path / "keep"))
        monkeypatch.setattr(batch, "run_pipeline", lambda **_: nulled_results)
        batch._run_one(**self._worker(tmp_path / "null", nonfinite="null"))
        kept = read(tmp_path / "keep" / "s1" / "s1_results.json")
        assert kept == json.dumps(kept_results, indent=2, default=_json_default)
        assert '"rmsd_md_final": NaN' in kept
        nulled = strict_loads(read(tmp_path / "null" / "s1" / "s1_results.json"))
        assert nulled["nonfinite_fields"] == LISTED and nulled == blank(json.loads(kept))

    def test_run_batch_passes_the_mode_to_every_worker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "run_pipeline", lambda **_: pipeline_results("s1"))
        rows = batch.run_batch(
            [EXAMPLE_1YCR], tmp_path / "out", metrics=[], skip_relax=True, nonfinite="null"
        )
        assert rows[0]["batch_status"] == "ok"
        report = tmp_path / "out" / EXAMPLE_1YCR.stem / f"{EXAMPLE_1YCR.stem}_results.json"
        assert strict_loads(read(report))["nonfinite_fields"] == LISTED

    def test_run_batch_refuses_another_value_before_it_creates_anything(self, tmp_path):
        with pytest.raises(ValueError, match=r"keep.*null"):
            batch.run_batch([EXAMPLE_1YCR], tmp_path / "out", nonfinite="drop")
        assert not (tmp_path / "out").exists()

    def test_main_forwards_the_flag(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.cif").write_text("data_x\n", encoding="utf-8")
        seen = []

        def fake_run_one(input_path, **kwargs):
            seen.append(kwargs["nonfinite"])
            return {"batch_status": "ok", "sample_id": input_path.stem}

        monkeypatch.setattr(batch, "_run_one", fake_run_one)
        argv = ["binding-metrics-batch", "-i", str(input_dir), "--output-csv", str(tmp_path / "m")]
        for extra in ([], ["--nonfinite", "null"], ["--nonfinite", "keep"]):
            monkeypatch.setattr(sys, "argv", [*argv, "--metrics", "energy", *extra])
            with pytest.raises(SystemExit):
                batch.main()
        assert seen == ["keep", "null", "keep"]

    def test_the_help_names_the_flag_and_its_default(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-batch", "--help"])
        with pytest.raises(SystemExit):
            batch.main()
        text = " ".join(capsys.readouterr().out.split())
        assert "--nonfinite {keep,null}" in text and "(default: keep)" in text


class TestSampleJsonUpdate:
    """A step that runs after the workers writes its block into the report of the sample."""

    METRICS = {"iptm": float("nan"), "plddt": np.array([90.0, np.nan]), "n": np.int64(4)}

    def test_a_finite_block_gives_the_text_json_dumps_gives(self, tmp_path):
        write_report(pipeline_results(), tmp_path, "sample")
        path = tmp_path / "sample_results.json"
        block = {"iptm": 0.5, "plddt": np.array([90.0, 80.0]), "n": np.int64(4)}
        data = json.loads(read(path))
        data["openfold"] = block
        expected = json.dumps(data, indent=2, default=_json_default)
        batch._update_sample_json(tmp_path, "sample", block)
        assert read(path) == expected

    def test_null_keeps_the_report_strict_and_extends_the_list(self, tmp_path):
        write_report(pipeline_results(), tmp_path, "sample", nonfinite="null")
        batch._update_sample_json(tmp_path, "sample", self.METRICS, nonfinite="null")
        data = strict_loads(read(tmp_path / "sample_results.json"))
        assert data["nonfinite_fields"] == [*LISTED, "openfold.iptm", "openfold.plddt[1]"]
        assert data["openfold"] == {"iptm": None, "plddt": [90.0, None], "n": 4}
        assert data["relax"]["rmsd_md_final"] is None

    def test_null_adds_the_list_to_a_report_that_has_none(self, tmp_path):
        (tmp_path / "s1_results.json").write_text("{}", encoding="utf-8")
        batch._update_sample_json(tmp_path, "s1", {"x": float("inf")}, nonfinite="null")
        assert strict_loads(read(tmp_path / "s1_results.json")) == {
            "openfold": {"x": None},
            "nonfinite_fields": ["openfold.x"],
        }

    def test_the_list_names_the_values_of_the_new_block_by_default(self, tmp_path):
        write_report(pipeline_results(), tmp_path, "sample")
        batch._update_sample_json(tmp_path, "sample", self.METRICS)
        data = json.loads(read(tmp_path / "sample_results.json"))
        assert data["nonfinite_fields"] == [*LISTED, "openfold.iptm", "openfold.plddt[1]"]

    def test_only_the_list_differs_from_the_text_json_dumps_gives(self, tmp_path):
        write_report(pipeline_results(), tmp_path, "sample")
        path = tmp_path / "sample_results.json"
        data = json.loads(read(path))
        data["openfold"] = self.METRICS
        data["nonfinite_fields"] = [*LISTED, "openfold.iptm", "openfold.plddt[1]"]
        batch._update_sample_json(tmp_path, "sample", self.METRICS)
        assert read(path) == json.dumps(data, indent=2, default=_json_default)

    @pytest.mark.parametrize("mode", ["keep", "null"])
    def test_a_block_that_is_finite_now_leaves_the_list(self, tmp_path, mode):
        write_report(pipeline_results(), tmp_path, "sample", nonfinite=mode)
        batch._update_sample_json(tmp_path, "sample", self.METRICS, nonfinite=mode)
        batch._update_sample_json(tmp_path, "sample", {"iptm": 0.8}, nonfinite=mode)
        data = json.loads(read(tmp_path / "sample_results.json"))
        assert data["nonfinite_fields"] == LISTED
        assert data["openfold"] == {"iptm": 0.8}

    @pytest.mark.parametrize("mode", ["keep", "null"])
    def test_two_blocks_are_listed_under_their_own_section(self, tmp_path, mode):
        write_report({"sample_id": "s"}, tmp_path, "s", nonfinite=mode)
        batch._update_sample_json(tmp_path, "s", {"x": float("nan")}, nonfinite=mode)
        batch._update_sample_json(
            tmp_path, "s", {"y": [1.0, float("inf")]}, section="prediction", nonfinite=mode
        )
        batch._update_sample_json(tmp_path, "s", {"x": float("-inf")}, nonfinite=mode)
        data = json.loads(read(tmp_path / "s_results.json"))
        assert sorted(data["nonfinite_fields"]) == ["openfold.x", "prediction.y[1]"]

    def test_the_list_is_added_to_a_report_that_has_none_by_default(self, tmp_path):
        (tmp_path / "s1_results.json").write_text("{}", encoding="utf-8")
        batch._update_sample_json(tmp_path, "s1", {"x": float("inf")})
        data = json.loads(read(tmp_path / "s1_results.json"))
        assert data["nonfinite_fields"] == ["openfold.x"]

    @staticmethod
    def _openfold_step(tmp_path, monkeypatch, **kwargs):
        from binding_metrics.metrics import evobind as evobind_module
        from binding_metrics.metrics import openfold

        (tmp_path / "s1").mkdir()
        (tmp_path / "s1" / "s1_results.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(openfold, "run_openfold_batched", lambda **kw: tmp_path)
        monkeypatch.setattr(
            openfold,
            "compute_openfold_metrics",
            lambda **kw: {"structure_path": "of3.cif", "iptm": float("nan")},
        )
        monkeypatch.setattr(evobind_module, "compute_evobind_score", lambda *a, **kw: {})
        monkeypatch.setattr(evobind_module, "compute_evobind_adversarial_check", lambda **kw: {})
        batch._run_batched_openfold(
            rows=[{"sample_id": "s1", "batch_status": "ok"}],
            sid_to_input={"s1": EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain="B",
            receptor_chain="A",
            **kwargs,
        )
        return json.loads(read(tmp_path / "s1" / "s1_results.json"))

    def test_the_batched_openfold_step_writes_in_the_mode_of_the_run(self, tmp_path, monkeypatch):
        self._openfold_step(tmp_path, monkeypatch, nonfinite="null")
        data = strict_loads(read(tmp_path / "s1" / "s1_results.json"))
        assert data["openfold"]["iptm"] is None
        assert data["nonfinite_fields"] == ["openfold.iptm"]

    def test_the_batched_openfold_step_lists_its_nonfinite_values_by_default(
        self, tmp_path, monkeypatch
    ):
        data = self._openfold_step(tmp_path, monkeypatch)
        assert math.isnan(data["openfold"]["iptm"])
        assert data["nonfinite_fields"] == ["openfold.iptm"]

    def test_the_batched_prediction_step_writes_in_the_mode_of_the_run(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        modes = []
        original = batch._update_sample_json

        def spy(*args, **kwargs):
            modes.append(kwargs.get("nonfinite", "keep"))
            return original(*args, **kwargs)

        monkeypatch.setattr(batch, "_update_sample_json", spy)
        sample = tmp_path / "in" / "a.pdb"
        sample.parent.mkdir()
        sample.write_text(read(EXAMPLE_1YCR), encoding="utf-8")
        batch.run_batch(
            [sample],
            tmp_path / "out",
            skip_prep=True,
            skip_relax=True,
            metrics={"openfold"},
            openfold_conda_env=None,
            predictor="of3",
            nonfinite="null",
        )
        assert modes == ["null"]
        report = strict_loads(read(tmp_path / "out" / "a" / "a_results.json"))
        assert report["prediction"]["model"] == "of3" and "nonfinite_fields" in report

    @pytest.mark.parametrize("mode", ["keep", "null"])
    def test_the_batched_prediction_step_lists_its_nonfinite_values(
        self, tmp_path, monkeypatch, mode
    ):
        StubOpenFold(monkeypatch)
        original = batch.run_prediction_step

        def with_nan(*args, **kwargs):
            block, provenance = original(*args, **kwargs)
            block["iptm"] = float("nan")
            return block, provenance

        monkeypatch.setattr(batch, "run_prediction_step", with_nan)
        sample = tmp_path / "in" / "a.pdb"
        sample.parent.mkdir()
        sample.write_text(read(EXAMPLE_1YCR), encoding="utf-8")
        batch.run_batch(
            [sample],
            tmp_path / "out",
            skip_prep=True,
            skip_relax=True,
            metrics={"openfold"},
            openfold_conda_env=None,
            predictor="of3",
            nonfinite=mode,
        )
        data = json.loads(read(tmp_path / "out" / "a" / "a_results.json"))
        assert data["nonfinite_fields"] == ["prediction.iptm"]
        assert (data["prediction"]["iptm"] is None) == (mode == "null")
