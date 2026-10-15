"""The optional ``openfold3_version`` key of the provenance block (issue #82).

The block keeps its eight keys unless a caller asks for the OpenFold3 version, which needs a
look at an installation (a process, when that installation is a conda environment).
"""

import json
from pathlib import Path

import pytest

from binding_metrics import provenance
from binding_metrics.cli import batch
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import _openfold_run
from binding_metrics.provenance import collect_provenance, conda_python_command

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


@pytest.fixture
def asked(monkeypatch):
    """Replace the version probe by one that records what it was asked and answers 0.5.0."""
    calls = []

    def probe(python_cmd=None):
        calls.append(None if python_cmd is None else list(python_cmd))
        return "0.5.0"

    monkeypatch.setattr(_openfold_run, "installed_openfold3_version", probe)
    return calls


class TestCollectProvenance:
    def test_the_key_is_absent_unless_asked(self, asked):
        assert "openfold3_version" not in collect_provenance(seed=1)
        assert asked == []

    def test_asking_adds_the_installed_version(self, asked):
        block = collect_provenance(seed=1, openfold3=True)
        assert block["openfold3_version"] == "0.5.0"
        assert asked == [None]  # the current interpreter

    def test_a_conda_environment_is_asked_through_conda_run(self, asked):
        command = conda_python_command("of3env")
        assert command == ["conda", "run", "-n", "of3env", "python"]
        collect_provenance(openfold3=True, openfold3_python_cmd=command)
        assert asked == [command]

    def test_not_installed_is_none(self, monkeypatch):
        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", lambda cmd=None: None)
        assert collect_provenance(openfold3=True)["openfold3_version"] is None

    def test_a_probe_that_raises_gives_none(self, monkeypatch):
        def broken(cmd=None):
            raise RuntimeError("no such interpreter")

        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", broken)
        assert collect_provenance(openfold3=True)["openfold3_version"] is None

    def test_the_block_stays_json_safe(self, asked):
        block = collect_provenance(seed=3, openfold3=True)
        assert json.loads(json.dumps(block))["openfold3_version"] == "0.5.0"

    @pytest.mark.parametrize("env", [None, ""])
    def test_no_environment_means_the_current_interpreter(self, env):
        assert conda_python_command(env) is None

    def test_the_real_probe_reads_the_current_interpreter(self):
        expected = provenance.openfold3_version(None)
        assert expected is None or isinstance(expected, str)


class TestRunPipeline:
    @staticmethod
    def _run(tmp_path, metrics, **kwargs):
        return run_pipeline(
            EXAMPLE_1YCR, tmp_path, skip_prep=True, skip_relax=True, metrics=metrics, **kwargs
        )["provenance"]

    def test_absent_when_the_openfold_step_is_not_selected(self, tmp_path, asked):
        assert "openfold3_version" not in self._run(tmp_path, frozenset())
        assert asked == []

    def test_present_when_the_openfold_step_runs(self, tmp_path, asked, monkeypatch):
        from binding_metrics.metrics import openfold

        monkeypatch.setattr(openfold, "run_openfold_scoring", lambda **kw: tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        block = self._run(tmp_path, frozenset({"openfold"}))
        assert block["openfold3_version"] == "0.5.0"
        assert asked == [None]

    def test_the_conda_environment_of_the_flag_is_asked(self, tmp_path, asked, monkeypatch):
        from binding_metrics.metrics import openfold

        monkeypatch.setattr(openfold, "run_openfold_scoring", lambda **kw: tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        self._run(tmp_path, frozenset({"openfold"}), openfold_conda_env="of3env")
        assert asked == [["conda", "run", "-n", "of3env", "python"]]


class TestBatchRows:
    def test_batched_openfold_rows_carry_the_version(self, tmp_path, asked, monkeypatch):
        from binding_metrics.metrics import openfold

        monkeypatch.setattr(openfold, "run_openfold_batched", lambda **kw: tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        rows = [
            {"sample_id": "s1", "batch_status": "ok"},
            {"sample_id": "s2", "batch_status": "error"},
        ]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={"s1": EXAMPLE_1YCR, "s2": EXAMPLE_1YCR},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env="of3env",
            peptide_chain="B",
            receptor_chain="A",
        )
        assert rows[0]["provenance_openfold3_version"] == "0.5.0"
        assert "provenance_openfold3_version" not in rows[1]  # its worker failed: no OpenFold3 call
        assert asked == [["conda", "run", "-n", "of3env", "python"]]  # one probe for the batch
