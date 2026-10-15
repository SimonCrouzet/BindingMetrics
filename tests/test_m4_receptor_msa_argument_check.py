"""The argument check asks the runner what receptor MSA it accepts, before anything runs.

``ColabFoldRunner`` takes ``receptor_msa="single"`` only, so ``--prediction-receptor-msa PATH`` for
ColabFold used to pass the argument check and the pre-flight plan and fail when the request was
built, after preparation and relaxation. The runners now state what they accept in the class
attributes ``supports_given_msa`` and ``supports_single``, and ``check_receptor_msa_arg`` gives
them to ``predictors._msa.check_msa_request``, the check that ``make_request`` makes. The refusal
is a usage error (exit status 2) in the argument check and in ``--preflight-only`` of both
commands, and a ``ValueError`` from the Python functions, before any output directory exists.
"""

from __future__ import annotations

import shutil
import sys
from types import SimpleNamespace

import pytest

from binding_metrics.cli import batch, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.prediction import (
    check_predictor,
    check_receptor_msa_arg,
    resolve_prediction_options,
    runner_class,
)
from binding_metrics.cli.run import run_pipeline
from binding_metrics.predictors.af2_runner import ColabFoldRunner
from binding_metrics.predictors.protenix_runner import ProtenixRunner
from binding_metrics.predictors.runners import PredictionRunner
from tests.test_pre_cli_run import LINEAR

MDM2 = "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"


@pytest.fixture
def alignment(tmp_path):
    path = tmp_path / "mdm2.a3m"
    path.write_text(f">query\n{MDM2}\n", encoding="utf-8")
    return path


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "in"
    path.mkdir()
    shutil.copy(LINEAR, path / "a.pdb")
    return path


def exit_of(main, argv, monkeypatch, capsys):
    """Run ``main`` with ``argv``; the exit status and the usage error, if any."""
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        main()
    streams = capsys.readouterr()
    error = streams.err.partition("error: ")[2].strip() if caught.value.code == 2 else ""
    return caught.value.code, error


def both_commands(tmp_path, folder, words):
    run_argv = [
        "binding-metrics-run",
        "--input", str(LINEAR),
        "--output-dir", str(tmp_path / "out"),
        "--metrics", "openfold",
        *words,
        "--preflight-only",
    ]  # fmt: skip
    batch_argv = [
        "binding-metrics-batch",
        "--input-dir", str(folder),
        "--output-csv", str(tmp_path / "res" / "out.csv"),
        "--metrics", "openfold",
        *words,
        "--preflight-only",
    ]  # fmt: skip
    return run_argv, batch_argv


# ---------------------------------------------------------------------------- the declarations


class TestTheRunnersSayWhatTheyAccept:
    def test_the_base_class_accepts_both(self):
        assert PredictionRunner.supports_given_msa is True
        assert PredictionRunner.supports_single is True

    def test_colabfold_accepts_the_word_single_only(self):
        assert (ColabFoldRunner.supports_given_msa, ColabFoldRunner.supports_single) == (
            False,
            True,
        )
        assert "FASTA" in ColabFoldRunner.msa_unsupported_reason

    def test_protenix_accepts_a_file_and_the_word_single(self):
        assert (ProtenixRunner.supports_given_msa, ProtenixRunner.supports_single) == (True, True)

    @pytest.mark.parametrize("model", ["boltz2", "of3"])
    def test_the_other_runners_accept_both_until_their_lanes_say_otherwise(self, model):
        cls = runner_class(model)
        assert (cls.supports_given_msa, cls.supports_single) == (True, True)

    def test_make_request_uses_the_attributes_the_command_line_reads(self, alignment):
        """One declaration: a runner that flips an attribute changes both checks."""

        class Takes(ColabFoldRunner):
            supports_given_msa = True

        with pytest.raises(ValueError, match="cannot be given the alignment"):
            ColabFoldRunner().make_request(
                LINEAR, name="p", binder_chain="B", receptor_chain="A", receptor_msa=alignment
            )
        check_receptor_msa_arg(alignment, Takes, "af2")  # accepted by the argument check
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            check_receptor_msa_arg(alignment, ColabFoldRunner, "af2")


# ---------------------------------------------------------------------------- the function


class TestCheckReceptorMsaArg:
    def test_a_file_for_colabfold_is_refused_with_the_option_the_reason_and_the_way_out(
        self, alignment
    ):
        with pytest.raises(ValueError) as caught:
            check_receptor_msa_arg(alignment, ColabFoldRunner, "af2")
        message = str(caught.value)
        assert message.startswith(f"--prediction-receptor-msa {alignment}: af2 cannot be given")
        assert "FASTA file" in message
        assert message.endswith("(on the command line: --prediction-receptor-msa single)")

    def test_the_word_single_is_accepted_by_colabfold(self):
        check_receptor_msa_arg("single", ColabFoldRunner, "af2")

    def test_protenix_takes_both(self, alignment):
        check_receptor_msa_arg(alignment, ProtenixRunner, "protenix")
        check_receptor_msa_arg("single", ProtenixRunner, "protenix")

    def test_a_file_that_is_named_single_is_a_file(self, tmp_path):
        named = tmp_path / "single"
        named.write_text(f">q\n{MDM2}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            check_receptor_msa_arg(named, ColabFoldRunner, "af2")

    def test_a_path_that_is_not_a_file_is_refused_first_whatever_the_runner(self, tmp_path):
        for runner in (None, ColabFoldRunner, ProtenixRunner):
            with pytest.raises(ValueError, match="is not a file"):
                check_receptor_msa_arg(tmp_path / "none.a3m", runner, "x")

    def test_a_stand_in_without_the_attributes_states_no_limit(self, alignment):
        check_receptor_msa_arg(alignment, SimpleNamespace(), "stub")

    def test_without_a_runner_only_the_file_is_checked(self, alignment):
        check_receptor_msa_arg(alignment)
        check_receptor_msa_arg("single")

    def test_a_runner_object_is_read_like_its_class(self, alignment):
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            check_receptor_msa_arg(alignment, ColabFoldRunner(), "af2")

    def test_openfold3_refuses_a_file_or_the_word_with_the_server_on(self, alignment):
        for value in (alignment, "single"):
            with pytest.raises(ValueError) as caught:
                check_receptor_msa_arg(value, runner_class("of3"), "of3", use_msa_server=True)
            message = str(caught.value)
            assert message.startswith(f"--prediction-receptor-msa {value} needs ")
            assert "--prediction-no-msa-server for OpenFold3" in message
            assert "would not be used" in message
            check_receptor_msa_arg(value, runner_class("of3"), "of3", use_msa_server=False)

    @pytest.mark.parametrize("model", ["af2", "protenix", "boltz2"])
    def test_the_other_runners_accept_the_value_with_the_server_on(self, model, alignment):
        """ColabFold and Protenix record the server off, Boltz-2 keeps the flag moot."""
        value = "single" if model == "af2" else alignment
        check_receptor_msa_arg(value, runner_class(model), model, use_msa_server=True)

    def test_a_stand_in_without_the_attribute_states_no_conflict(self, alignment):
        check_receptor_msa_arg(alignment, SimpleNamespace(), "stub", use_msa_server=True)

    def test_the_flag_is_not_checked_when_the_caller_does_not_say(self, alignment):
        check_receptor_msa_arg(alignment, runner_class("of3"), "of3")

    def test_the_policy_and_the_server_are_not_this_check(self, alignment):
        """``forbid-server`` with the server on is the pre-flight's refusal, not a usage error."""
        options = resolve_prediction_options(
            "protenix",
            None,
            prediction_receptor_msa=alignment,
            prediction_msa_policy="forbid-server",
        )
        assert options.use_msa_server is True


class TestOpenFold3WithTheServerOn:
    """Issue 155: the pair is refused by the request, and now by the argument check too."""

    def test_the_declarations(self):
        assert PredictionRunner.msa_conflicts_with_server is False
        assert runner_class("of3").msa_conflicts_with_server is True
        for model in ("af2", "protenix", "boltz2"):
            assert runner_class(model).msa_conflicts_with_server is False

    def test_the_request_and_the_argument_check_agree(self, alignment):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        with pytest.raises(ValueError, match="receptor_msa needs use_msa_server=False"):
            OpenFold3Runner().make_request(
                LINEAR, name="p", binder_chain="B", receptor_chain="A", mode="score",
                receptor_msa=alignment,
            )  # fmt: skip
        with pytest.raises(ValueError, match="needs --prediction-no-msa-server"):
            resolve_prediction_options("of3", None, prediction_receptor_msa=alignment)

    def test_the_server_off_through_either_spelling_passes(self, alignment):
        for keywords in (
            dict(prediction_use_msa_server=False),
            dict(openfold_use_msa_server=False),
        ):
            options = resolve_prediction_options(
                "of3", None, prediction_receptor_msa=alignment, **keywords
            )
            assert options.use_msa_server is False

    def test_both_commands_exit_2_with_and_without_the_plan_flag(
        self, tmp_path, folder, alignment, monkeypatch, capsys
    ):
        words = ["--predictor", "of3", "--prediction-receptor-msa", str(alignment)]
        errors = []
        for plan_only in (True, False):
            run_argv, batch_argv = both_commands(tmp_path, folder, words)
            if not plan_only:
                run_argv.remove("--preflight-only")
                batch_argv.remove("--preflight-only")
            for main, argv in ((run.main, run_argv), (batch.main, batch_argv)):
                status, error = exit_of(main, argv, monkeypatch, capsys)
                assert status == 2
                errors.append(error)
        assert len(set(errors)) == 1
        assert "--prediction-no-msa-server for OpenFold3" in errors[0]
        assert not (tmp_path / "out").exists() and not (tmp_path / "res").exists()


class TestResolvePredictionOptions:
    def test_colabfold_with_a_file_is_refused(self, alignment):
        with pytest.raises(ValueError, match="af2 cannot be given the alignment"):
            resolve_prediction_options(
                "af2", None, prediction_use_msa_server=False, prediction_receptor_msa=alignment
            )

    def test_the_python_api_check_refuses_it_too(self, alignment):
        with pytest.raises(ValueError, match="af2 cannot be given the alignment"):
            check_predictor(
                "af2", None, prediction_use_msa_server=False, prediction_receptor_msa=alignment
            )

    @pytest.mark.parametrize(
        "model, value",
        [("af2", "single"), ("protenix", "single"), ("protenix", "FILE")],
    )
    def test_what_the_runner_takes_is_accepted(self, model, value, alignment):
        receptor_msa = alignment if value == "FILE" else value
        options = check_predictor(
            model, None, prediction_use_msa_server=False, prediction_receptor_msa=receptor_msa
        )
        assert options.use_msa_server is False


# ---------------------------------------------------------------------------- the commands


class TestTheCommandLines:
    def test_colabfold_with_a_file_is_a_usage_error_in_both_commands(
        self, tmp_path, folder, alignment, monkeypatch, capsys
    ):
        words = [
            "--predictor", "af2", "--prediction-no-msa-server",
            "--prediction-receptor-msa", str(alignment),
        ]  # fmt: skip
        run_argv, batch_argv = both_commands(tmp_path, folder, words)
        run_status, run_error = exit_of(run.main, run_argv, monkeypatch, capsys)
        batch_status, batch_error = exit_of(batch.main, batch_argv, monkeypatch, capsys)
        assert (run_status, batch_status) == (2, 2)
        assert run_error == batch_error
        assert "af2 cannot be given the alignment of the receptor" in run_error
        assert "--prediction-receptor-msa single" in run_error
        assert not (tmp_path / "out").exists() and not (tmp_path / "res").exists()

    def test_without_the_plan_flag_nothing_runs_either(
        self, tmp_path, alignment, monkeypatch, capsys
    ):
        """The argument check is the first thing ``main`` does; the pipeline is not entered."""
        called = []
        monkeypatch.setattr(run, "run_pipeline", lambda *a, **k: called.append(1))
        argv = [
            "binding-metrics-run",
            "--input", str(LINEAR),
            "--output-dir", str(tmp_path / "out"),
            "--metrics", "openfold",
            "--predictor", "af2", "--prediction-no-msa-server",
            "--prediction-receptor-msa", str(alignment),
        ]  # fmt: skip
        status, error = exit_of(run.main, argv, monkeypatch, capsys)
        assert status == 2 and "af2 cannot be given the alignment" in error
        assert called == [] and not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        "predictor, value",
        [("af2", "single"), ("protenix", "single"), ("protenix", "FILE")],
    )
    def test_what_the_runner_takes_gives_a_plan(
        self, tmp_path, folder, alignment, monkeypatch, capsys, predictor, value
    ):
        words = [
            "--predictor", predictor, "--prediction-no-msa-server",
            "--prediction-receptor-msa", str(alignment) if value == "FILE" else value,
        ]  # fmt: skip
        run_argv, batch_argv = both_commands(tmp_path, folder, words)
        statuses = []
        for main, argv in ((run.main, run_argv), (batch.main, batch_argv)):
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(SystemExit) as caught:
                main()
            statuses.append(caught.value.code)
            assert "Receptor MSA" in capsys.readouterr().out
        assert statuses == [0, 0]

    def test_the_forbidden_server_with_a_file_for_colabfold_is_still_the_runner_s_refusal(
        self, tmp_path, folder, alignment, monkeypatch, capsys
    ):
        """The runner's limit comes first: the policy cannot be met with a file for ColabFold."""
        words = [
            "--predictor", "af2", "--prediction-msa-policy", "forbid-server",
            "--prediction-no-msa-server", "--prediction-receptor-msa", str(alignment),
        ]  # fmt: skip
        run_argv, _ = both_commands(tmp_path, folder, words)
        status, error = exit_of(run.main, run_argv, monkeypatch, capsys)
        assert status == 2 and "takes receptor_msa='single' only" in error


class TestThePythonFunctions:
    def test_run_pipeline_and_run_batch_refuse_alike_before_an_output_exists(
        self, tmp_path, folder, alignment
    ):
        common = dict(
            metrics={"openfold"},
            skip_prep=True,
            skip_relax=True,
            openfold_conda_env=None,
            peptide_chain="B",
            receptor_chain="A",
            preflight_only=True,
            predictor="af2",
            prediction_use_msa_server=False,
            prediction_receptor_msa=alignment,
        )
        with pytest.raises(ValueError, match="af2 cannot be given the alignment") as from_batch:
            run_batch([folder / "a.pdb"], tmp_path / "out", **common)
        with pytest.raises(ValueError, match="af2 cannot be given the alignment") as from_run:
            run_pipeline(LINEAR, tmp_path / "out2", **common)
        assert str(from_batch.value) == str(from_run.value)
        assert not (tmp_path / "out").exists() and not (tmp_path / "out2").exists()
