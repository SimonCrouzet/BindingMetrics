"""The pre-flight check of custom weights (``--prediction-weights``), before anything runs.

The path must exist, be readable and be the kind the model's runner takes; a model whose runner
cannot take custom weights is refused in the pre-flight style (found, requires, why, fix), the fix
naming the runners that can, read from the runner registry; and when the weights are accepted the
plan carries a note that the declared limits come from the input format and architecture while the
caveats about accuracy refer to the standard weights. The declared hard limits stay in force.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from binding_metrics import preflight_cli
from binding_metrics.capabilities import (
    IncompatibleInputError,
    check_weights_path,
    preflight,
    profile_input,
)
from binding_metrics.cli import prediction as cli_prediction
from binding_metrics.cli.prediction import runner_weights_kinds
from binding_metrics.predictors.runners import PredictionRunner

DATA = Path(__file__).resolve().parent.parent / "data"

NOTE = (
    "custom weights: the limits declared for OpenFold3 come from its input format and "
    "architecture; the caveats about accuracy and published benchmarks refer to the standard "
    "weights"
)


class FileRunner(PredictionRunner):
    name = "boltz2"
    supports_custom_weights = True
    weights_kind = "file"

    def prepare(self, request, work_dir):  # pragma: no cover - never run here
        raise NotImplementedError

    def run(self, request, work_dir):  # pragma: no cover - never run here
        raise NotImplementedError


class DirectoryRunner(FileRunner):
    name = "protenix"
    weights_kind = "directory"


class PlainRunner(FileRunner):
    name = "af2"
    supports_custom_weights = False


@pytest.fixture
def p53():
    return profile_input(DATA / "example_linear_p53_1YCR.pdb", "B", "A")


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "w" / "of3-finetuned.pt"
    path.parent.mkdir()
    path.write_bytes(b"weights")
    return path


@pytest.fixture
def weights_directory(tmp_path):
    root = tmp_path / "wd"
    (root / "shards").mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "shards" / "a.bin").write_bytes(b"a")
    return root


class TestCheckWeightsPath:
    def test_an_existing_file_passes_and_comes_back_absolute(self, checkpoint):
        assert check_weights_path(checkpoint) == checkpoint
        assert check_weights_path(checkpoint, "file") == checkpoint

    def test_an_existing_directory_passes(self, weights_directory):
        assert check_weights_path(weights_directory, "directory") == weights_directory

    def test_a_missing_file_lists_what_its_directory_holds(self, checkpoint):
        with pytest.raises(ValueError, match="do not exist") as caught:
            check_weights_path(checkpoint.parent / "other.pt")
        assert "of3-finetuned.pt" in str(caught.value)

    def test_a_missing_directory_says_so(self, tmp_path):
        with pytest.raises(ValueError, match="do not exist: .*does not exist"):
            check_weights_path(tmp_path / "nowhere" / "w.pt")

    def test_a_directory_for_a_file_runner_lists_its_content(self, weights_directory):
        with pytest.raises(ValueError, match="are a directory .*config.json.*one checkpoint file"):
            check_weights_path(weights_directory, "file")

    def test_a_file_for_a_directory_runner_says_its_size(self, checkpoint):
        with pytest.raises(ValueError, match=r"are a file \(7 bytes\).*directory of weights"):
            check_weights_path(checkpoint, "directory")

    def test_an_empty_directory_is_refused(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="holds no files"):
            check_weights_path(tmp_path / "empty")

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read everything")
    def test_an_unreadable_file_is_refused(self, checkpoint):
        checkpoint.chmod(0)
        try:
            with pytest.raises(ValueError, match="cannot be read"):
                check_weights_path(checkpoint)
        finally:
            checkpoint.chmod(0o644)

    def test_an_unknown_kind_is_a_programming_error(self, checkpoint):
        with pytest.raises(ValueError, match="kind must be"):
            check_weights_path(checkpoint, "folder")

    def test_a_home_directory_is_expanded(self, checkpoint, monkeypatch):
        monkeypatch.setenv("HOME", str(checkpoint.parent))
        assert check_weights_path("~/of3-finetuned.pt") == checkpoint


class TestPreflightAcceptsWeights:
    def test_a_runner_that_takes_them_gets_the_note_and_no_violation(self, p53, checkpoint):
        report = preflight(p53, [], "of3", weights=checkpoint, weights_kinds={"of3": "file"})
        assert report.compatible and NOTE in report.notes
        assert NOTE in report.format()

    def test_the_note_is_a_note_and_not_a_warning_or_violation(self, p53, checkpoint):
        report = preflight(p53, [], "of3", weights=checkpoint, weights_kinds={"of3": "file"})
        assert NOTE not in report.warnings and not report.violations

    def test_the_caveats_of_the_standard_weights_are_still_listed(self, checkpoint):
        cyclic = profile_input(DATA / "example_ncaa_cyclosporin_1CWA.cif", "C", "A")
        report = preflight(cyclic, [], "of3", weights=checkpoint, weights_kinds={"of3": "file"})
        assert any("cyclic: true" in w for w in report.warnings)  # the note says what they mean

    def test_the_hard_limits_stay_in_force_with_custom_weights(self, checkpoint):
        bicyclic = profile_input(DATA / "example_bicyclic_sfti1_3P8F.cif", "I", "A")
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(bicyclic, [], "of3", weights=checkpoint, weights_kinds={"of3": "file"})
        assert [v.constraint for v in caught.value.violations] == ["closures"]

    def test_a_directory_of_weights_for_a_directory_runner(self, p53, weights_directory):
        report = preflight(
            p53, [], "boltz2", weights=weights_directory, weights_kinds={"boltz2": "directory"}
        )
        assert report.compatible and any(n.startswith("custom weights:") for n in report.notes)

    def test_the_wrong_kind_is_a_plain_error_listing_what_was_found(self, p53, weights_directory):
        with pytest.raises(ValueError, match="are a directory") as caught:
            preflight(p53, [], "of3", weights=weights_directory, weights_kinds={"of3": "file"})
        assert not isinstance(caught.value, IncompatibleInputError)

    def test_a_missing_path_is_a_plain_error_before_the_models_are_looked_at(self, p53, tmp_path):
        with pytest.raises(ValueError, match="do not exist"):
            preflight(p53, [], "of3", weights=tmp_path / "no.pt", weights_kinds={"of3": "file"})

    def test_without_weights_nothing_changes(self, p53):
        report = preflight(p53, [], "of3", weights_kinds={"of3": "file"})
        assert not any("custom weights" in n for n in report.notes)

    def test_a_caller_that_does_not_know_the_runners_gets_a_note_that_it_was_not_checked(
        self, p53, checkpoint
    ):
        report = preflight(p53, [], "of3", weights=checkpoint)
        assert report.compatible
        assert any("not checked against predictor OpenFold3" in n for n in report.notes)

    def test_weights_without_a_predictor_are_noted_as_unused(self, p53, checkpoint):
        report = preflight(p53, ["interface"], None, weights=checkpoint, weights_kinds={})
        assert any("no predictor is checked, so they are not used" in n for n in report.notes)


class TestPreflightRefusesARunnerWithoutSupport:
    KINDS = {"of3": "file", "protenix": "directory"}

    def _refusal(self, profile, checkpoint, predictor="boltz2", **kwargs):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(
                profile, [], predictor, weights=checkpoint, weights_kinds=self.KINDS, **kwargs
            )
        return caught.value

    def test_it_is_a_violation_in_the_pre_flight_style(self, p53, checkpoint):
        error = self._refusal(p53, checkpoint)
        (violation,) = error.violations
        assert violation.constraint == "weights" and violation.kind == "predictor"
        assert violation.name == "boltz2"
        assert str(checkpoint) in violation.fact and "a file" in violation.fact
        assert "starts Boltz-2 with weights you choose" in violation.requirement
        assert "does not pass weights to the model" in violation.reason
        text = str(error)
        for label in ("found:", "requires:", "why:", "fix:"):
            assert label in text

    def test_the_fix_names_the_runners_that_do_from_the_registry_data(self, p53, checkpoint):
        (violation,) = self._refusal(p53, checkpoint).violations
        assert "OpenFold3 (of3, weights as a file)" in violation.fix
        assert "Protenix (protenix, weights as a directory)" in violation.fix
        assert "--prediction-dir" in violation.fix

    def test_the_fix_says_so_when_no_runner_takes_weights(self, p53, checkpoint):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(p53, [], "boltz2", weights=checkpoint, weights_kinds={})
        assert (
            "no runner of this package takes custom weights yet" in caught.value.violations[0].fix
        )

    def test_under_skip_the_predictor_is_left_out(self, p53, checkpoint):
        report = preflight(
            p53, [], "boltz2", policy="skip", weights=checkpoint, weights_kinds=self.KINDS
        )
        assert not report.predictor_usable and report.violations[0].constraint == "weights"

    def test_a_predictor_that_only_warns_runs_with_the_problem_logged(
        self, p53, checkpoint, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            report = preflight(
                p53,
                [],
                "boltz2",
                predictor_policy="warn",
                weights=checkpoint,
                weights_kinds=self.KINDS,
            )
        assert report.violations and report.predictor_usable
        assert "weights" in caplog.text

    def test_a_supported_model_next_to_an_unsupported_one(self, p53, checkpoint):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(p53, [], ["of3", "boltz2"], weights=checkpoint, weights_kinds={"of3": "file"})
        assert [v.name for v in caught.value.violations] == ["boltz2"]


class TestRunnerRegistryData:
    def test_the_openfold3_runner_is_the_only_one_registered_today(self, monkeypatch):
        monkeypatch.setattr(cli_prediction, "RUNNERS", {"of3": cli_prediction.RUNNERS["of3"]})
        assert runner_weights_kinds() == {"of3": "file"}

    def test_a_runner_added_to_the_registry_is_read_from_its_class(self, monkeypatch):
        monkeypatch.setattr(
            cli_prediction,
            "RUNNERS",
            {
                "of3": cli_prediction.RUNNERS["of3"],
                "boltz2": f"{__name__}:FileRunner",
                "protenix": f"{__name__}:DirectoryRunner",
                "af2": f"{__name__}:PlainRunner",
            },
        )
        assert runner_weights_kinds() == {"of3": "file", "boltz2": "file", "protenix": "directory"}

    def test_a_runner_module_that_cannot_be_imported_is_skipped_and_logged(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            cli_prediction,
            "RUNNERS",
            {"of3": cli_prediction.RUNNERS["of3"], "boltz2": "no_such_module_anywhere:Runner"},
        )
        with caplog.at_level(logging.WARNING, logger=cli_prediction.logger.name):
            assert runner_weights_kinds() == {"of3": "file"}
        assert "boltz2" in caplog.text

    def test_the_check_input_fix_comes_from_the_registry(self, monkeypatch, checkpoint):
        monkeypatch.setattr(
            cli_prediction,
            "RUNNERS",
            {"of3": cli_prediction.RUNNERS["of3"], "protenix": f"{__name__}:DirectoryRunner"},
        )
        outcome = preflight_cli.check_input(
            DATA / "example_linear_p53_1YCR.pdb",
            "B",
            "A",
            model=("boltz2", "prediction", False, "score"),
            prediction_weights=checkpoint,
        )
        assert outcome.refused
        assert "Protenix (protenix, weights as a directory)" in outcome.error.violations[0].fix


class TestCheckInput:
    P53 = DATA / "example_linear_p53_1YCR.pdb"

    def test_the_plan_of_an_accepted_run_has_the_note(self, checkpoint):
        outcome = preflight_cli.check_input(
            self.P53,
            "B",
            "A",
            model=("of3", "openfold", False, "score"),
            prediction_weights=checkpoint,
            include_plan=True,
        )
        assert not outcome.refused and outcome.block["status"] == "ok"
        assert NOTE in outcome.block["plan"]
        assert NOTE in outcome.block["report"]["notes"]

    def test_the_predictor_route_is_checked_the_same_way(self, checkpoint):
        outcome = preflight_cli.check_input(
            self.P53,
            "B",
            "A",
            model=("of3", "prediction", False, "score"),
            prediction_weights=checkpoint,
        )
        assert NOTE in outcome.block["report"]["notes"]

    def test_a_directory_is_refused_for_the_checkpoint_file_of_openfold3(self, weights_directory):
        with pytest.raises(ValueError, match="one checkpoint file"):
            preflight_cli.check_input(
                self.P53,
                "B",
                "A",
                model=("of3", "openfold", False, "score"),
                prediction_weights=weights_directory,
            )

    def test_a_missing_path_is_refused_before_the_input_is_profiled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            preflight_cli, "profile_input", lambda *a, **k: pytest.fail("profiled first")
        )
        with pytest.raises(ValueError, match="do not exist"):
            preflight_cli.check_input(
                self.P53,
                "B",
                "A",
                model=("of3", "openfold", False, "score"),
                prediction_weights=tmp_path / "missing.pt",
            )

    def test_weights_without_a_model_step_are_dropped_with_a_warning(self, checkpoint, caplog):
        with caplog.at_level(logging.WARNING, logger=preflight_cli.logger.name):
            outcome = preflight_cli.check_input(
                self.P53, "B", "A", model=None, prediction_weights=checkpoint
            )
        assert "no model step" in caplog.text
        assert not any("custom weights" in n for n in outcome.block["report"]["notes"])

    def test_without_weights_the_report_is_as_before(self):
        outcome = preflight_cli.check_input(
            self.P53, "B", "A", model=("of3", "openfold", False, "score")
        )
        assert not any("custom weights" in n for n in outcome.block["report"]["notes"])
