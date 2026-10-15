"""An energy mode that failed beside one that succeeded is a failed step (#217).

``compute_interaction_energy`` returns ``success=True`` with a non-empty ``error_message`` when
one requested mode succeeds and another fails (``relaxed: minimization failed: ...``). The pose
set layer reads that shape as partial; the single run and the batch read it as complete, so
``binding-metrics-run`` exited 0 and the batch row was ``ok`` without a failed step. The values of
the modes that succeeded stay in the row and in the JSON.
"""

import json
import sys

import pytest
from conftest import DATA_DIR
from openmm.app import Simulation

from binding_metrics.cli import run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import _collect_failures

EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
RELAXED_FAILURE = "relaxed: minimization failed: RuntimeError: the minimizer stopped"


class TestCollectFailures:
    def test_a_failed_mode_beside_a_successful_one_fails_the_step(self):
        energy = {
            "success": True,
            "error_message": RELAXED_FAILURE,
            "raw_interaction_energy": -16.0,
            "relaxed_interaction_energy": None,
        }
        assert _collect_failures({"energy": energy}) == [("energy", RELAXED_FAILURE)]

    @pytest.mark.parametrize("message", [None, ""])
    def test_a_block_without_a_message_is_complete(self, message):
        energy = {"success": True, "error_message": message, "raw_interaction_energy": -16.0}
        assert _collect_failures({"energy": energy}) == []

    def test_a_reason_alone_is_not_a_failure(self):
        energy = {"success": True, "error_message": None, "reason": "no contacts at the interface"}
        assert _collect_failures({"energy": energy}) == []

    def test_a_block_that_failed_in_part_without_a_success_flag_fails_the_step(self):
        receptor = {"models": [], "error_message": "energy of model 1: ValueError: no template"}
        assert _collect_failures({"receptor_quality": receptor}) == [
            ("receptor_quality", "energy of model 1: ValueError: no template")
        ]

    def test_a_skipped_step_with_a_message_is_not_a_failure(self):
        assert _collect_failures({"energy": {"skipped": True, "error_message": "x"}}) == []

    def test_the_message_of_a_failed_step_is_named_once(self):
        energy = {"success": False, "error_message": RELAXED_FAILURE}
        assert _collect_failures({"energy": energy}) == [("energy", RELAXED_FAILURE)]

    def test_the_reason_is_cut_like_the_other_failures(self):
        ((_, reason),) = _collect_failures(
            {"energy": {"success": True, "error_message": "relaxed: " + "x" * 500}}
        )
        assert len(reason) == 200


def fail_minimization(monkeypatch):
    """Make the minimization of every OpenMM simulation raise; the raw mode does not minimize."""

    def minimize_energy(self, *args, **kwargs):
        raise RuntimeError("the minimizer stopped")

    monkeypatch.setattr(Simulation, "minimizeEnergy", minimize_energy)


def results_json(folder):
    (path,) = folder.rglob(f"{EXAMPLE_1YCR.stem}_results.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.slow
@pytest.mark.integration
class TestRealEnergyModes:
    """Real preparation and a real raw energy; only the minimization is made to fail."""

    def run_batch_of_two_modes(self, tmp_path, modes=("raw", "relaxed")):
        return run_batch(
            [EXAMPLE_1YCR],
            tmp_path / "out",
            skip_relax=True,
            metrics=frozenset({"energy"}),
            energy_modes=modes,
            peptide_chain="B",
            receptor_chain="A",
            device="cpu",
        )

    def test_the_batch_row_is_partial_and_keeps_the_raw_energy(self, tmp_path, monkeypatch):
        fail_minimization(monkeypatch)
        (row,) = self.run_batch_of_two_modes(tmp_path)
        assert row["batch_status"] == "partial"
        assert row["batch_failed_steps"] == "energy"
        assert row["batch_failed_reasons"].startswith("energy: relaxed: minimization failed: ")
        assert "the minimizer stopped" in row["batch_failed_reasons"]
        assert row["energy_success"] is True
        assert row["energy_raw_interaction_energy"] < 0.0
        assert row["energy_relaxed_interaction_energy"] is None

    def test_the_json_keeps_the_raw_energy_and_names_the_failed_mode(self, tmp_path, monkeypatch):
        fail_minimization(monkeypatch)
        (row,) = self.run_batch_of_two_modes(tmp_path)
        energy = results_json(tmp_path / "out")["energy"]
        assert energy["raw_interaction_energy"] == pytest.approx(
            row["energy_raw_interaction_energy"], abs=1.5
        )
        assert energy["error_message"].startswith("relaxed: minimization failed: ")

    def test_a_batch_whose_modes_all_succeed_stays_ok(self, tmp_path):
        (row,) = self.run_batch_of_two_modes(tmp_path, modes=("raw",))
        assert row["batch_status"] == "ok"
        assert "batch_failed_steps" not in row
        assert row["energy_raw_interaction_energy"] < 0.0

    def test_the_run_exits_1_and_names_the_failed_mode(self, tmp_path, monkeypatch, capsys):
        fail_minimization(monkeypatch)
        argv = [
            "binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(tmp_path / "run"),
            "--skip-relax", "--metrics", "energy", "--energy-modes", "raw", "relaxed",
            "--peptide-chain", "B", "--receptor-chain", "A", "--device", "cpu",
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 1
        shown = capsys.readouterr().out
        assert "1 step(s) did not complete" in shown
        assert "[x] energy: relaxed: minimization failed: " in shown
        energy = results_json(tmp_path / "run")["energy"]
        assert energy["raw_interaction_energy"] < 0.0, "the Partial results file keeps it"
