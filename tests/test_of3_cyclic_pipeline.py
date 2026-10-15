"""``--openfold-cyclic`` in ``binding-metrics-run`` and ``-batch`` (#77).

The model is a stub; the query builders, the decision (``decide_binder_cyclic``), the store, the
session and the report are the real code. The OpenFold3 version is a stub too: what OpenFold3
does with a ``cyclic`` field rests on its source, not on a run.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import add_openfold_cyclic_arg, batch, check_openfold_cyclic, run
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import _openfold_run, openfold
from binding_metrics.protocols.report import _flatten, write_report
from tests.test_feat_c_support import StubOpenFold

pytest.importorskip("biotite")

DATA = Path(__file__).parent.parent / "data"
# chain C is closed head to tail and has D-Ala and N-methylated residues: "auto" leaves it linear
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
# chain I (SFTI-1) is closed head to tail, has standard residues only, and a disulfide, which the
# pre-flight refuses unless it only warns
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"  # chain B is linear


@pytest.fixture(autouse=True)
def _openfold3_version(monkeypatch):
    """OpenFold3 0.5.0 in this interpreter, and an empty cache of conda versions."""
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )
    monkeypatch.setattr(_openfold_run, "_VERSION_BY_PYTHON", {})


@pytest.fixture
def set_version(monkeypatch):
    def _set(version):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: version
        )

    return _set


def pipeline(tmp_path, structure=CYCLOSPORIN, binder="C", receptor="A", **kwargs):
    kwargs.setdefault("openfold_conda_env", None)
    return run_pipeline(
        structure,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        peptide_chain=binder,
        receptor_chain=receptor,
        **kwargs,
    )


def standard_pipeline(tmp_path, **kwargs):
    """The pipeline on a head-to-tail binder of standard residues, which "auto" sends as cyclic."""
    kwargs.setdefault("on_incompatible", "warn")  # the disulfide of SFTI-1 has no OpenFold3 input
    return pipeline(tmp_path, structure=SFTI1, binder="I", **kwargs)


class TestChoices:
    @pytest.mark.parametrize(
        "value, expected",
        [("auto", "auto"), ("on", True), ("off", False), (True, True), (False, False)],
    )
    def test_the_command_line_choices_and_the_api_values(self, value, expected):
        resolved = check_openfold_cyclic(value)
        assert resolved == expected and type(resolved) is type(expected)

    @pytest.mark.parametrize("value", ["yes", None, 1, "ON"])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(ValueError, match="openfold_cyclic must be one of"):
            check_openfold_cyclic(value)

    def test_the_option_is_in_both_commands_with_the_default_auto(self, monkeypatch):
        import argparse

        for module in ("run", "batch"):
            holder = {}

            def stop(self, *args, holder=holder, **kwargs):
                holder["parser"] = self
                raise SystemExit

            monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
            monkeypatch.setattr(sys, "argv", ["prog"])
            with pytest.raises(SystemExit):
                {"run": run, "batch": batch}[module].main()
            action = next(
                a for a in holder["parser"]._actions if "--openfold-cyclic" in a.option_strings
            )
            assert action.default == "auto" and list(action.choices) == ["auto", "on", "off"]

    def test_the_help_says_what_the_flag_does_and_does_not(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_openfold_cyclic_arg(parser)
        text = " ".join(parser.format_help().split())
        assert "--openfold-cyclic {auto,on,off}" in text
        for stated in (
            "head-to-tail",
            "wrap the relative positions",
            "does not enforce the closure bond",
            "only in an example query",
            "no accuracy benchmark for cyclic peptides",
        ):
            assert stated in text

    @pytest.mark.parametrize("value", ["yes", None])
    def test_the_pipeline_refuses_a_bad_value_before_it_runs(self, tmp_path, value):
        with pytest.raises(ValueError, match="openfold_cyclic"):
            pipeline(tmp_path, openfold_cyclic=value)
        assert not (tmp_path / "openfold").exists()

    def test_the_batch_refuses_a_bad_value_before_it_runs(self, tmp_path):
        with pytest.raises(ValueError, match="openfold_cyclic"):
            batch.run_batch([P53], tmp_path / "out", openfold_cyclic="yes")
        assert not (tmp_path / "out").exists()


class TestFlagsOfTheCommands:
    @staticmethod
    def _parsed(monkeypatch, tmp_path, module, extra):
        captured = {}
        target = {"run": "run_pipeline", "batch": "run_batch"}[module]
        monkeypatch.setattr(
            {"run": run, "batch": batch}[module],
            target,
            lambda *a, **kwargs: captured.update(kwargs) or ({} if module == "run" else []),
        )
        input_args = ["-i", str(P53), "-o", str(tmp_path / "o")]
        if module == "batch":
            input_args = ["-i", str(P53.parent), "--output-csv", str(tmp_path / "m.csv")]
        monkeypatch.setattr(sys, "argv", ["prog", *input_args, *extra])
        try:
            {"run": run, "batch": batch}[module].main()
        except SystemExit:
            pass
        return captured

    @pytest.mark.parametrize("module", ["run", "batch"])
    @pytest.mark.parametrize(
        "extra, expected",
        [([], "auto"), (["--openfold-cyclic", "on"], "on"), (["--openfold-cyclic", "off"], "off")],
    )
    def test_the_choice_reaches_the_api(self, monkeypatch, tmp_path, module, extra, expected):
        captured = self._parsed(monkeypatch, tmp_path, module, extra)
        assert captured["openfold_cyclic"] == expected


class TestLegacyOpenFoldStep:
    """``results["openfold"]`` without ``--predictor``."""

    @staticmethod
    def _stub(monkeypatch, tmp_path):
        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            return tmp_path

        monkeypatch.setattr(openfold, "run_openfold_scoring", record)
        monkeypatch.setattr(openfold, "run_openfold_refolding", record)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        return seen

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_default_passes_no_argument_and_records_a_cyclic_binder(
        self, tmp_path, monkeypatch, mode
    ):
        seen = self._stub(monkeypatch, tmp_path)
        block = standard_pipeline(tmp_path, openfold_mode=mode)["openfold"]
        assert "binder_cyclic" not in seen  # the run function's own default is "auto"
        assert block["binder_cyclic"] is True and "reason" not in block

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_modified_residues_leave_the_binder_linear_and_the_block_says_why(
        self, tmp_path, monkeypatch, mode
    ):
        seen = self._stub(monkeypatch, tmp_path)
        block = pipeline(tmp_path, openfold_mode=mode)["openfold"]  # 1CWA chain C
        assert "binder_cyclic" not in seen  # the choice is still "auto"
        assert block["binder_cyclic"] is False
        assert block["reason"].startswith(
            "binder_cyclic: chain C is closed head to tail but has modified residues "
            "(ABA, BMT, DAL, MLE, MVA, SAR)"
        )
        assert "one complex" in block["reason"] and "--openfold-cyclic on" in block["reason"]

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_a_linear_binder_is_recorded_as_not_cyclic(self, tmp_path, monkeypatch, mode):
        self._stub(monkeypatch, tmp_path)
        block = pipeline(tmp_path, structure=P53, binder="B", receptor="A", openfold_mode=mode)[
            "openfold"
        ]
        assert block["binder_cyclic"] is False and "reason" not in block

    @pytest.mark.parametrize("choice, forwarded", [("on", True), ("off", False)])
    def test_on_and_off_are_forwarded(self, tmp_path, monkeypatch, choice, forwarded):
        seen = self._stub(monkeypatch, tmp_path)
        block = pipeline(tmp_path, openfold_cyclic=choice)["openfold"]
        assert seen["binder_cyclic"] is forwarded
        assert block["binder_cyclic"] is forwarded

    def test_the_api_takes_true_and_false(self, tmp_path, monkeypatch):
        seen = self._stub(monkeypatch, tmp_path)
        assert pipeline(tmp_path / "a", openfold_cyclic=False)["openfold"]["binder_cyclic"] is False
        assert seen["binder_cyclic"] is False

    def test_off_does_not_even_read_the_structure(self, tmp_path, monkeypatch):
        self._stub(monkeypatch, tmp_path)

        def broken(*args, **kwargs):
            raise AssertionError("the structure must not be read")

        monkeypatch.setattr(_openfold_run, "_binder_is_head_to_tail", broken)
        assert pipeline(tmp_path, openfold_cyclic="off")["openfold"]["binder_cyclic"] is False

    def test_an_old_openfold3_leaves_the_binder_linear_and_the_block_says_why(
        self, tmp_path, monkeypatch, set_version
    ):
        self._stub(monkeypatch, tmp_path)
        set_version("0.4.4")
        block = pipeline(tmp_path)["openfold"]
        assert block["binder_cyclic"] is False
        assert block["reason"].startswith("binder_cyclic: chain C is closed head to tail")
        assert "0.4.4" in block["reason"] and "0.4.5" in block["reason"]

    def test_an_unreadable_version_says_how_to_force_the_flag(
        self, tmp_path, monkeypatch, set_version
    ):
        self._stub(monkeypatch, tmp_path)
        set_version(None)
        block = pipeline(tmp_path)["openfold"]
        assert block["binder_cyclic"] is False
        assert "--openfold-cyclic on" in block["reason"]

    def test_the_reason_is_joined_to_the_reasons_of_the_model_results(
        self, tmp_path, monkeypatch, set_version
    ):
        self._stub(monkeypatch, tmp_path)
        monkeypatch.setattr(
            openfold, "compute_openfold_metrics", lambda **kw: {"reason": "interface PDE: size"}
        )
        set_version("0.4.4")
        reason = pipeline(tmp_path)["openfold"]["reason"]
        assert reason.startswith("interface PDE: size; binder_cyclic: chain C")

    def test_a_forced_flag_with_an_old_openfold3_is_a_failed_step_not_a_silent_one(
        self, tmp_path, monkeypatch, set_version
    ):
        set_version("0.4.4")
        monkeypatch.setattr(
            openfold, "run_openfold", lambda **kw: pytest.fail("OpenFold3 must not start")
        )
        block = pipeline(tmp_path, openfold_cyclic="on")["openfold"]
        assert "0.4.5" in block["error"] and "0.4.4" in block["error"]


class TestPredictorPath:
    """``results["prediction"]`` with ``--predictor of3``."""

    def test_the_default_records_a_cyclic_binder(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        block = standard_pipeline(tmp_path, predictor="of3")["prediction"]
        # (the stub's PAE has one token per residue number, so the interface blocks of 3P8F, whose
        # numbering has insertion codes, may carry a reason of their own)
        assert block["binder_cyclic"] is True and "binder_cyclic" not in block.get("reason", "")
        assert "binder_cyclic" not in stub.calls[0]["kwargs"]  # the default is left out

    def test_modified_residues_leave_the_binder_linear_and_the_block_says_why(
        self, tmp_path, monkeypatch
    ):
        StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3")["prediction"]  # 1CWA chain C
        assert block["binder_cyclic"] is False
        assert (
            "binder_cyclic: chain C is closed head to tail but has modified residues"
            in (block["reason"])
        )

    def test_the_request_key_follows_the_choice(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        keys = {
            choice: pipeline(
                tmp_path / choice, predictor="of3", prediction_cache=store, openfold_cyclic=choice
            )["prediction"]["cache"]["request_key"]
            for choice in ("auto", "on", "off")
        }
        assert len(set(keys.values())) == 3

    @pytest.mark.parametrize("choice, forwarded", [("on", True), ("off", False)])
    def test_on_and_off_reach_the_runner_function(self, tmp_path, monkeypatch, choice, forwarded):
        stub = StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, predictor="of3", openfold_cyclic=choice)["prediction"]
        assert stub.calls[0]["kwargs"]["binder_cyclic"] is forwarded
        assert block["binder_cyclic"] is forwarded

    def test_a_linear_binder_is_recorded_as_not_cyclic(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        block = pipeline(tmp_path, structure=P53, binder="B", receptor="A", predictor="of3")[
            "prediction"
        ]
        assert block["binder_cyclic"] is False

    def test_an_old_openfold3_gives_the_reason(self, tmp_path, monkeypatch, set_version):
        StubOpenFold(monkeypatch)
        set_version("0.4.4")
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        assert block["binder_cyclic"] is False and "0.4.4" in block["reason"]

    def test_an_adopted_output_says_nothing_of_its_query(self, tmp_path, monkeypatch):
        from tests.test_feat_c_support import write_of3_output

        output = tmp_path / "mine"
        write_of3_output(output, CYCLOSPORIN.stem, CYCLOSPORIN)
        block = pipeline(tmp_path / "o", predictor="of3", prediction_dir=output)["prediction"]
        assert "binder_cyclic" not in block

    def test_a_failed_prediction_has_no_decision(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch, error=RuntimeError("boom"))
        block = pipeline(tmp_path, predictor="of3")["prediction"]
        assert "error" in block and "binder_cyclic" not in block

    def test_the_decision_is_not_logged_a_second_time(
        self, tmp_path, monkeypatch, caplog, set_version
    ):
        StubOpenFold(monkeypatch)
        set_version("0.4.4")
        with caplog.at_level(logging.INFO, logger=_openfold_run.logger.name):
            pipeline(tmp_path, predictor="of3")
        # the stub replaces the query builder, so the pipeline's own record logs nothing
        assert "closed head to tail" not in caplog.text


class TestReportAndRows:
    def test_the_csv_row_and_the_json_carry_the_key_and_the_reason(
        self, tmp_path, monkeypatch, set_version
    ):
        StubOpenFold(monkeypatch)
        set_version("0.4.4")
        results = pipeline(tmp_path, predictor="of3")
        flat = _flatten(results)
        assert flat["prediction_binder_cyclic"] is False
        assert "0.4.4" in flat["prediction_reason"]
        path = write_report(results, tmp_path, "s", fmt="json", summary=True)
        assert json.loads(path.read_text(encoding="utf-8"))["prediction"]["binder_cyclic"] is False

    def test_the_markdown_report_shows_the_flag(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        results = pipeline(tmp_path, predictor="of3")
        write_report(results, tmp_path, "s", fmt="json", summary=True)
        summary = (tmp_path / "s_report.md").read_text(encoding="utf-8")
        assert "Binder sent as cyclic" in summary

    def test_the_legacy_block_flattens_to_an_openfold_column(self, tmp_path, monkeypatch):
        monkeypatch.setattr(openfold, "run_openfold_scoring", lambda **kw: tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        flat = _flatten(standard_pipeline(tmp_path))
        assert flat["openfold_binder_cyclic"] is True


class TestBatch:
    def test_the_batched_openfold_call_gets_the_choice_and_each_row_the_decision(
        self, tmp_path, monkeypatch
    ):
        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            return tmp_path

        monkeypatch.setattr(openfold, "run_openfold_batched", record)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(
            batch,
            "_detect_sample_chains",
            lambda rows, sid_to_input, p, r, label: [
                (0, "example_bicyclic_sfti1_3P8F", SFTI1, "I", "A"),
                (1, "example_ncaa_cyclosporin_1CWA", CYCLOSPORIN, "C", "A"),
                (2, "example_linear_p53_1YCR", P53, "B", "A"),
            ],
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [
            {"sample_id": "example_bicyclic_sfti1_3P8F", "batch_status": "ok"},
            {"sample_id": "example_ncaa_cyclosporin_1CWA", "batch_status": "ok"},
            {"sample_id": "example_linear_p53_1YCR", "batch_status": "ok"},
        ]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={
                "example_bicyclic_sfti1_3P8F": SFTI1,
                "example_ncaa_cyclosporin_1CWA": CYCLOSPORIN,
                "example_linear_p53_1YCR": P53,
            },
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
        )
        assert "binder_cyclic" not in seen
        assert rows[0]["openfold_binder_cyclic"] is True
        assert rows[1]["openfold_binder_cyclic"] is False  # modified residues
        assert "modified residues" in rows[1]["openfold_reason"]
        assert rows[2]["openfold_binder_cyclic"] is False

    @pytest.mark.parametrize("choice, forwarded", [("on", True), ("off", False)])
    def test_on_and_off_reach_the_batched_call(self, tmp_path, monkeypatch, choice, forwarded):
        seen = {}
        monkeypatch.setattr(
            openfold, "run_openfold_batched", lambda **kw: seen.update(kw) or tmp_path
        )
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(
            batch, "_detect_sample_chains", lambda *a, **k: [(0, "s", P53, "B", "A")]
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [{"sample_id": "s", "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={"s": P53},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
            openfold_cyclic=choice,
        )
        assert seen["binder_cyclic"] is forwarded
        assert rows[0]["openfold_binder_cyclic"] is forwarded

    def test_the_batched_prediction_records_each_sample(self, tmp_path, monkeypatch, set_version):
        stub = StubOpenFold(monkeypatch)
        set_version("0.5.0")
        sid_to_input = {
            "example_bicyclic_sfti1_3P8F": SFTI1,
            "example_ncaa_cyclosporin_1CWA": CYCLOSPORIN,
            "example_linear_p53_1YCR": P53,
        }
        monkeypatch.setattr(
            batch,
            "_detect_sample_chains",
            lambda *a, **k: [
                (0, "example_bicyclic_sfti1_3P8F", SFTI1, "I", "A"),
                (1, "example_ncaa_cyclosporin_1CWA", CYCLOSPORIN, "C", "A"),
                (2, "example_linear_p53_1YCR", P53, "B", "A"),
            ],
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [
            {"sample_id": "example_bicyclic_sfti1_3P8F", "batch_status": "ok"},
            {"sample_id": "example_ncaa_cyclosporin_1CWA", "batch_status": "ok"},
            {"sample_id": "example_linear_p53_1YCR", "batch_status": "ok"},
        ]
        batch._run_batched_prediction(
            rows=rows,
            sid_to_input=sid_to_input,
            output_dir=tmp_path,
            predictor="of3",
            peptide_chain=None,
            receptor_chain=None,
        )
        assert stub.starts == 1  # the samples share one request signature, so one model start
        assert rows[0]["prediction_binder_cyclic"] is True
        assert rows[1]["prediction_binder_cyclic"] is False  # modified residues
        assert "modified residues" in rows[1]["prediction_reason"]
        assert rows[2]["prediction_binder_cyclic"] is False

    def test_the_batched_prediction_gives_the_reason_of_an_old_openfold3(
        self, tmp_path, monkeypatch, set_version
    ):
        StubOpenFold(monkeypatch)
        set_version("0.4.4")
        monkeypatch.setattr(
            batch, "_detect_sample_chains", lambda *a, **k: [(0, "c", CYCLOSPORIN, "C", "A")]
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [{"sample_id": "c", "batch_status": "ok"}]
        batch._run_batched_prediction(
            rows=rows,
            sid_to_input={"c": CYCLOSPORIN},
            output_dir=tmp_path,
            predictor="of3",
            peptide_chain=None,
            receptor_chain=None,
        )
        assert rows[0]["prediction_binder_cyclic"] is False
        assert "0.4.4" in rows[0]["prediction_reason"]
