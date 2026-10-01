"""``--openfold-no-msa-server`` in ``binding-metrics-run`` and ``-batch`` (#68 workaround).

With the ColabFold server on (the default) OpenFold3 can replace the template alignment of a
chain for which it finds hits; with it off, OpenFold3 runs without a computed MSA and the
template alignments that the toolkit writes stay. The model is a stub. What OpenFold3 does with
the option rests on its source, not on a run.
"""

import argparse
import importlib
import sys

import pytest

from binding_metrics.cli import batch, run
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import openfold
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
)


def pipeline(tmp_path, **kwargs):
    return run_pipeline(
        EXAMPLE_1YCR,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        peptide_chain=PEPTIDE_CHAIN,
        receptor_chain=RECEPTOR_CHAIN,
        openfold_conda_env=None,
        **kwargs,
    )


class TestTheOption:
    @pytest.mark.parametrize("module", ["binding_metrics.cli.run", "binding_metrics.cli.batch"])
    def test_it_is_a_flag_that_is_off_by_default(self, module, monkeypatch):
        holder = {}

        def stop(self, *args, **kwargs):
            holder["parser"] = self
            raise SystemExit

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            importlib.import_module(module).main()
        action = next(
            a for a in holder["parser"]._actions if "--openfold-no-msa-server" in a.option_strings
        )
        assert action.default is False and action.nargs == 0

    def test_the_help_says_what_the_run_does_and_what_one_complex_showed(self):
        parser = argparse.ArgumentParser()
        from binding_metrics.cli import add_openfold_no_msa_server_arg

        add_openfold_no_msa_server_arg(parser)
        help_text = " ".join(parser.format_help().split())
        for stated in (
            "dummy MSA that holds only the query sequence of each chain",
            "input reference suggests this for MSA-free runs",
            "lowers accuracy for a natural receptor",
            "no longer replaced by the server (issue #68)",
        ):
            assert stated in help_text
        # the measurement is stated as one complex, one seed, with its numbers
        for stated in (
            "One complex (1YCR, OpenFold3 0.5.0, one seed)",
            "1.6 A with the server and no template",
            "21.6 A with no MSA and no template",
            "1.1 A with a working template and no MSA",
        ):
            assert stated in help_text

    @staticmethod
    def _parsed(monkeypatch, tmp_path, module, extra):
        captured = {}
        target = {"run": "run_pipeline", "batch": "run_batch"}[module]
        monkeypatch.setattr(
            {"run": run, "batch": batch}[module],
            target,
            lambda *a, **kwargs: captured.update(kwargs) or ({} if module == "run" else []),
        )
        input_args = ["-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o")]
        if module == "batch":
            input_args = ["-i", str(EXAMPLE_1YCR.parent), "--output-csv", str(tmp_path / "m.csv")]
        monkeypatch.setattr(sys, "argv", ["prog", *input_args, *extra])
        try:
            {"run": run, "batch": batch}[module].main()
        except SystemExit:
            pass
        return captured

    @pytest.mark.parametrize("module", ["run", "batch"])
    @pytest.mark.parametrize("extra, expected", [([], True), (["--openfold-no-msa-server"], False)])
    def test_the_flag_reaches_the_api(self, monkeypatch, tmp_path, module, extra, expected):
        assert self._parsed(monkeypatch, tmp_path, module, extra)["openfold_use_msa_server"] is (
            expected
        )


class TestLegacyStep:
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
    def test_the_default_passes_no_argument(self, tmp_path, monkeypatch, mode):
        seen = self._stub(monkeypatch, tmp_path)
        pipeline(tmp_path, openfold_mode=mode)
        assert "use_msa_server" not in seen  # the run function's own default is True

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_off_reaches_the_run_function(self, tmp_path, monkeypatch, mode):
        seen = self._stub(monkeypatch, tmp_path)
        pipeline(tmp_path, openfold_mode=mode, openfold_use_msa_server=False)
        assert seen["use_msa_server"] is False

    @pytest.mark.parametrize("use, recorded", [(True, True), (False, False)])
    def test_the_provenance_records_the_setting(self, tmp_path, monkeypatch, use, recorded):
        self._stub(monkeypatch, tmp_path)
        provenance = pipeline(tmp_path, openfold_use_msa_server=use)["provenance"]
        assert provenance["openfold3_use_msa_server"] is recorded

    def test_the_provenance_has_no_key_when_openfold3_does_not_run_here(self, tmp_path):
        results = run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset(),
            peptide_chain=PEPTIDE_CHAIN,
            receptor_chain=RECEPTOR_CHAIN,
            openfold_use_msa_server=False,
        )
        assert "openfold3_use_msa_server" not in results["provenance"]


class TestPredictorStep:
    def test_the_default_passes_no_argument(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        results = pipeline(tmp_path, predictor="of3")
        assert "use_msa_server" not in stub.calls[0]["kwargs"]
        assert results["provenance"]["openfold3_use_msa_server"] is True

    def test_off_reaches_the_runner_function_and_the_provenance(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        results = pipeline(tmp_path, predictor="of3", openfold_use_msa_server=False)
        assert stub.calls[0]["kwargs"]["use_msa_server"] is False
        assert results["provenance"]["openfold3_use_msa_server"] is False

    def test_the_setting_is_part_of_the_request_key_and_gets_its_own_store_entry(
        self, tmp_path, monkeypatch
    ):
        stub = StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        on = pipeline(tmp_path / "a", predictor="of3", prediction_cache=store)["prediction"]
        off = pipeline(
            tmp_path / "b",
            predictor="of3",
            prediction_cache=store,
            openfold_use_msa_server=False,
        )["prediction"]
        again = pipeline(
            tmp_path / "c",
            predictor="of3",
            prediction_cache=store,
            openfold_use_msa_server=False,
        )["prediction"]
        assert on["cache"]["request_key"] != off["cache"]["request_key"]
        assert off["cache"]["request_key"] == again["cache"]["request_key"]
        assert stub.starts == 2  # the second setting is a new prediction; the third is the store's
        assert again["cache"]["hits"] == 1 and again["cache"]["runs"] == 0

    def test_the_runner_option_that_carries_it(self, tmp_path):
        runner = OpenFold3Runner()
        runner.version = lambda: "0.5.0"
        on = runner.make_request(
            EXAMPLE_1YCR, name="q", binder_chain="B", receptor_chain="A", use_msa_server=True
        )
        off = runner.make_request(
            EXAMPLE_1YCR, name="q", binder_chain="B", receptor_chain="A", use_msa_server=False
        )
        assert on.options["use_msa_server"] is True and off.options["use_msa_server"] is False
        assert on.key() != off.key()


class TestBatch:
    @staticmethod
    def _eligible(monkeypatch):
        monkeypatch.setattr(
            batch,
            "_detect_sample_chains",
            lambda *a, **k: [(0, EXAMPLE_1YCR.stem, EXAMPLE_1YCR, PEPTIDE_CHAIN, RECEPTOR_CHAIN)],
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))

    @pytest.mark.parametrize("use", [True, False])
    def test_the_batched_openfold_call_and_the_row(self, tmp_path, monkeypatch, use):
        self._eligible(monkeypatch)
        seen = {}
        monkeypatch.setattr(
            openfold, "run_openfold_batched", lambda **kw: seen.update(kw) or tmp_path
        )
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        rows = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
            openfold_use_msa_server=use,
        )
        assert ("use_msa_server" in seen) is (not use)
        assert seen.get("use_msa_server", True) is use
        assert rows[0]["provenance_openfold3_use_msa_server"] is use

    def test_the_batched_prediction_has_its_own_key_and_row(self, tmp_path, monkeypatch):
        self._eligible(monkeypatch)
        stub = StubOpenFold(monkeypatch)
        rows = {}
        for use in (True, False):
            row = [{"sample_id": EXAMPLE_1YCR.stem, "batch_status": "ok"}]
            batch._run_batched_prediction(
                rows=row,
                sid_to_input={EXAMPLE_1YCR.stem: EXAMPLE_1YCR},
                output_dir=tmp_path,
                predictor="of3",
                peptide_chain=None,
                receptor_chain=None,
                openfold_use_msa_server=use,
            )
            rows[use] = row[0]
        assert stub.starts == 2
        assert (
            rows[True]["prediction_cache_request_key"]
            != rows[False]["prediction_cache_request_key"]
        )
        assert rows[True]["provenance_openfold3_use_msa_server"] is True
        assert rows[False]["provenance_openfold3_use_msa_server"] is False
        assert "use_msa_server" not in stub.calls[0]["kwargs"]
        assert stub.calls[1]["kwargs"]["use_msa_server"] is False


class TestDocs:
    def test_the_limitation_paragraph_names_the_flag_of_the_pipeline(self):
        from pathlib import Path

        text = (Path(__file__).parent.parent / "docs" / "metrics.md").read_text(encoding="utf-8")
        start = text.index("**known limitation: the MSA server and the templates")
        section = text[start : text.index("\n\n", start)]
        assert "`--openfold-no-msa-server`" in section and "`--no-msa-server`" in section
        assert "lowers accuracy for a natural receptor" in section
