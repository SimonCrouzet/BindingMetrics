"""``--on-unmappable-residue`` reaches the OpenFold3 run functions of run and batch.

The default (``error``) is left out of the call, so a function that predates the option, or a
test double that replaces it, is called exactly as before. Only ``x`` is passed on.
"""

import sys
from pathlib import Path

import pytest

from binding_metrics.cli import batch, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import run_pipeline

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def _pipeline_kwargs_seen(tmp_path, monkeypatch, mode, **pipeline_kwargs):
    from binding_metrics.metrics import openfold

    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return tmp_path

    monkeypatch.setattr(openfold, "run_openfold_scoring", record)
    monkeypatch.setattr(openfold, "run_openfold_refolding", record)
    monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
    run_pipeline(
        EXAMPLE_1YCR,
        tmp_path,
        skip_prep=True,
        skip_relax=True,
        metrics=frozenset({"openfold"}),
        openfold_mode=mode,
        **pipeline_kwargs,
    )
    return seen


class TestRunPipeline:
    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_x_is_forwarded(self, tmp_path, monkeypatch, mode):
        seen = _pipeline_kwargs_seen(tmp_path, monkeypatch, mode, on_unmappable_residue="x")
        assert seen["on_unmappable_residue"] == "x"

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_default_is_not_passed(self, tmp_path, monkeypatch, mode):
        assert "on_unmappable_residue" not in _pipeline_kwargs_seen(tmp_path, monkeypatch, mode)

    def test_an_unknown_value_raises_before_anything_runs(self, tmp_path):
        with pytest.raises(ValueError, match="on_unmappable_residue must be one of"):
            run_pipeline(EXAMPLE_1YCR, tmp_path, metrics=frozenset(), on_unmappable_residue="skip")
        assert not any(tmp_path.iterdir())


class TestRunMain:
    @staticmethod
    def _parsed(monkeypatch, tmp_path, extra):
        captured = {}

        def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return {"sample_id": "x", "provenance": {}}

        monkeypatch.setattr(run, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o"), *extra],
        )
        run.main()
        return captured

    def test_default_is_error(self, monkeypatch, tmp_path):
        assert self._parsed(monkeypatch, tmp_path, [])["on_unmappable_residue"] == "error"

    def test_x_reaches_the_pipeline(self, monkeypatch, tmp_path):
        parsed = self._parsed(monkeypatch, tmp_path, ["--on-unmappable-residue", "x"])
        assert parsed["on_unmappable_residue"] == "x"

    def test_an_unknown_choice_is_refused_by_the_parser(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as stop:
            self._parsed(monkeypatch, tmp_path, ["--on-unmappable-residue", "skip"])
        assert stop.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_the_config_file_can_set_it(self, monkeypatch, tmp_path):
        config = tmp_path / "run.toml"
        config.write_text('on-unmappable-residue = "x"\n', encoding="utf-8")
        parsed = self._parsed(monkeypatch, tmp_path, ["--config", str(config)])
        assert parsed["on_unmappable_residue"] == "x"


class TestBatchedOpenFold:
    @staticmethod
    def _seen(tmp_path, monkeypatch, **kwargs):
        from binding_metrics.metrics import openfold

        seen = {}

        def record(**kw):
            seen.update(kw)
            return tmp_path

        monkeypatch.setattr(openfold, "run_openfold_batched", record)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
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
        return seen

    def test_x_reaches_the_batched_call(self, tmp_path, monkeypatch):
        assert (
            self._seen(tmp_path, monkeypatch, on_unmappable_residue="x")["on_unmappable_residue"]
            == "x"
        )

    def test_the_default_is_not_passed(self, tmp_path, monkeypatch):
        assert "on_unmappable_residue" not in self._seen(tmp_path, monkeypatch)


class TestRunBatch:
    def test_the_option_reaches_the_batched_step(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(batch, "_run_one", lambda **kw: {"batch_status": "ok"})
        monkeypatch.setattr(
            batch, "_run_batched_openfold", lambda **kw: seen.append(kw["on_unmappable_residue"])
        )
        for value in ("error", "x"):
            run_batch([Path("a.cif")], tmp_path, metrics={"openfold"}, on_unmappable_residue=value)
        assert seen == ["error", "x"]

    def test_an_unknown_value_raises(self, tmp_path):
        with pytest.raises(ValueError, match="on_unmappable_residue must be one of"):
            run_batch([Path("a.cif")], tmp_path, metrics=set(), on_unmappable_residue="skip")

    def test_main_forwards_the_flag(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.cif").write_text("data_x\n", encoding="utf-8")
        seen = []

        def fake_run_batch(paths, output_dir, **kw):
            seen.append(kw["on_unmappable_residue"])
            return []

        monkeypatch.setattr(batch, "run_batch", fake_run_batch)
        argv = [
            "binding-metrics-batch",
            "-i",
            str(input_dir),
            "--output-csv",
            str(tmp_path / "m.csv"),
        ]
        for extra, expected in (([], "error"), (["--on-unmappable-residue", "x"], "x")):
            monkeypatch.setattr(sys, "argv", argv + extra)
            with pytest.raises(SystemExit):
                batch.main()
            assert seen[-1] == expected
