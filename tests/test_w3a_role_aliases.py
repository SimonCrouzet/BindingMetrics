"""``--binder-chain`` and ``--target-chain`` in the pipeline layer (issue #31).

They spell the same options as ``--peptide-chain`` and ``--receptor-chain``: same
destination, old flags unchanged, and two spellings with different IDs are an error.
"""

import json
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import batch, run
from binding_metrics.protocols import relaxation

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


@pytest.fixture
def parsed(monkeypatch, tmp_path):
    """Run ``main`` of one CLI with its worker stubbed; return the chain values it received."""

    def _parse(cli, extra):
        seen = {}
        if cli == "run":

            def fake_pipeline(**kwargs):
                seen.update(kwargs)
                return {"sample_id": "x", "provenance": {}}

            monkeypatch.setattr(run, "run_pipeline", fake_pipeline)
            argv = ["-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o")]
            entry = run.main
        elif cli == "batch":
            (tmp_path / "in").mkdir(exist_ok=True)
            (tmp_path / "in" / "a.cif").write_text("data_x\n")

            def fake_run_one(input_path, **kwargs):
                seen.update(kwargs)
                return {"sample_id": input_path.stem, "batch_status": "ok"}

            monkeypatch.setattr(batch, "_run_one", fake_run_one)
            argv = ["-i", str(tmp_path / "in"), "--output-csv", str(tmp_path / "m.csv")]
            entry = batch.main
        else:

            class FakeRelaxer:
                def __init__(self, config):
                    seen["peptide_chain"] = config.peptide_chain_id
                    seen["receptor_chain"] = config.receptor_chain_id

            def fake_run_one(relaxer, *args, **kwargs):
                return type(
                    "Result",
                    (),
                    {
                        "success": True,
                        "minimized_structure_path": None,
                        "md_final_structure_path": None,
                    },
                )()

            monkeypatch.setattr(relaxation, "ImplicitRelaxation", FakeRelaxer)
            monkeypatch.setattr(relaxation, "_run_one", fake_run_one)
            argv = ["-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "o")]
            entry = relaxation.main
        monkeypatch.setattr(sys, "argv", ["prog", *argv, *extra])
        try:
            entry()
        except SystemExit as exit_request:
            if exit_request.code not in (0, None):
                raise
        return seen

    return _parse


@pytest.mark.parametrize("cli", ["run", "batch", "relax"])
class TestPipelineCliAliases:
    def test_aliases_fill_the_old_destinations(self, parsed, cli):
        seen = parsed(cli, ["--binder-chain", "B", "--target-chain", "A"])
        assert (seen["peptide_chain"], seen["receptor_chain"]) == ("B", "A")

    def test_old_flags_still_work(self, parsed, cli):
        seen = parsed(cli, ["--peptide-chain", "B", "--receptor-chain", "A"])
        assert (seen["peptide_chain"], seen["receptor_chain"]) == ("B", "A")

    def test_nothing_given_means_auto_detect(self, parsed, cli):
        seen = parsed(cli, [])
        assert (seen["peptide_chain"], seen["receptor_chain"]) == (None, None)

    def test_the_same_id_through_both_spellings_is_fine(self, parsed, cli):
        seen = parsed(cli, ["--peptide-chain", "B", "--binder-chain", "B"])
        assert seen["peptide_chain"] == "B"

    @pytest.mark.parametrize(
        "flags",
        [
            ["--peptide-chain", "B", "--binder-chain", "C"],
            ["--target-chain", "X", "--receptor-chain", "A"],
        ],
    )
    def test_different_ids_are_a_usage_error(self, parsed, cli, flags, capsys):
        with pytest.raises(SystemExit) as exit_request:
            parsed(cli, flags)
        assert exit_request.value.code == 2
        assert "conflicts with" in capsys.readouterr().err

    def test_help_lists_the_alias_spellings(self, cli, monkeypatch, capsys):
        entry = {"run": run.main, "batch": batch.main, "relax": relaxation.main}[cli]
        monkeypatch.setattr(sys, "argv", ["prog", "--help"])
        with pytest.raises(SystemExit):
            entry()
        out = capsys.readouterr().out
        assert "--peptide-chain PEPTIDE_CHAIN, --binder-chain PEPTIDE_CHAIN" in out
        assert "--receptor-chain RECEPTOR_CHAIN, --target-chain RECEPTOR_CHAIN" in out


class TestRunPipelineAliases:
    """1YCR: chain B is the 13-residue p53 peptide, chain A is MDM2."""

    def _run(self, tmp_path, **kwargs):
        return run.run_pipeline(
            EXAMPLE_1YCR, tmp_path, skip_prep=True, skip_relax=True, metrics=frozenset(), **kwargs
        )

    def test_aliases_select_the_same_chains_as_the_old_names(self, tmp_path):
        old = self._run(tmp_path / "old", peptide_chain="B", receptor_chain="A")
        new = self._run(tmp_path / "new", binder_chain="B", target_chain="A")
        assert new["chains"] == old["chains"]
        assert new["chains"]["peptide_n_residues"] == 13

    def test_alias_can_be_the_only_one_given(self, tmp_path):
        results = self._run(tmp_path, binder_chain="B")
        assert results["chains"]["peptide_chain"] == "B"

    def test_same_id_through_both_spellings_is_accepted(self, tmp_path):
        results = self._run(tmp_path, peptide_chain="B", binder_chain="B")
        assert results["chains"]["peptide_chain"] == "B"

    def test_different_ids_raise_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="peptide_chain='B' and binder_chain='A'"):
            self._run(tmp_path, peptide_chain="B", binder_chain="A")
        with pytest.raises(ValueError, match="receptor_chain='A' and target_chain='B'"):
            self._run(tmp_path, receptor_chain="A", target_chain="B")

    def test_unknown_alias_chain_is_reported_like_the_old_name(self, tmp_path):
        with pytest.raises(run.ChainNotFoundError, match="chain 'Z' not found"):
            self._run(tmp_path, binder_chain="Z")

    def test_aliases_are_keyword_only(self):
        import inspect

        parameters = inspect.signature(run.run_pipeline).parameters
        assert parameters["binder_chain"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["target_chain"].kind is inspect.Parameter.KEYWORD_ONLY


class TestBatchWorkerAliases:
    def _row(self, tmp_path, **kwargs):
        return batch._run_one(
            input_path=EXAMPLE_1YCR,
            output_dir=tmp_path,
            sample_id=None,
            skip_prep=True,
            ph=7.4,
            keep_water=False,
            canonicalize=False,
            skip_relax=True,
            md_duration_ps=0.0,
            device="cuda",
            peptide_chain=kwargs.pop("peptide_chain", None),
            receptor_chain=kwargs.pop("receptor_chain", None),
            metrics=frozenset(),
            energy_modes=("relaxed",),
            openfold_mode="score",
            openfold_conda_env=None,
            log_file=None,
            **kwargs,
        )

    def test_aliases_run_the_sample(self, tmp_path):
        row = self._row(tmp_path, binder_chain="B", target_chain="A")
        assert row["batch_status"] == "ok"
        report = json.loads(
            (tmp_path / row["sample_id"] / f"{row['sample_id']}_results.json").read_text()
        )
        assert (report["chains"]["peptide_chain"], report["chains"]["receptor_chain"]) == ("B", "A")

    def test_conflicting_spellings_make_an_error_row(self, tmp_path):
        row = self._row(tmp_path, peptide_chain="B", binder_chain="A")
        assert row["batch_status"] == "error"
        assert "peptide_chain='B' and binder_chain='A'" in row["batch_error"]
