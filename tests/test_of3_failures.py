"""Why an OpenFold3 run failed, and when it failed while exiting with status 0 (issues #83-#85).

The "OpenFold3" here is a tiny Python process that writes what the real one writes: the text
of the messages comes from openfold3 v0.5.0 (``validator.py`` and ``experiment_runner.py``
for the exceptions, ``core/runners/writer.py`` for ``summary.txt``, ``projects/of3_all_atom/
runner.py`` for ``logs/predict_err_rank<N>.log``). Nothing here starts a real OpenFold3, so
what is not shown is that those files look like this in a real run.
"""

import logging
import os
import subprocess
import sys
import textwrap

import pytest

from binding_metrics.metrics import _openfold_run
from binding_metrics.metrics._openfold_run import (
    OpenFoldQueryError,
    OpenFoldRunError,
    _error_log_reasons,
    _failed_query_reasons,
    _read_run_summary,
    _run_openfold_command,
    _user_default_runner_yaml,
)
from binding_metrics.metrics._openfold_templates import StreamNotes


def _fake_openfold(code: str) -> list[str]:
    """Command line of a Python process that runs ``code`` in place of ``run_openfold``."""
    return [sys.executable, "-c", textwrap.dedent(code)]


def _summary_text(total: int, failed: list[str]) -> str:
    """``summary.txt`` as ``writer.py#_write_summary`` of openfold3 v0.5.0 builds it."""
    lines = [
        "\n" + "=" * 50,
        "    PREDICTION SUMMARY (COMPLETE)    ",
        "=" * 50,
        f"Total Queries Processed: {total}",
        f"  - Successful Queries:  {total - len(failed)}",
        f"  - Failed Queries:      {len(failed)}",
    ]
    if failed:
        lines.append(f"\nFailed Queries: {', '.join(sorted(set(failed)))}")
    lines.append("=" * 50 + "\n")
    return "\n".join(lines)


def _error_log_entry(query_ids: list[str], kind: str, message: str) -> str:
    """One entry of a ``predict_err_rank<N>.log`` file, as openfold3's runner writes it."""
    return "\n".join(
        [
            "=" * 50,
            "Timestamp: 2026-09-30 12:00:00",
            f"Query ID(s): {', '.join(query_ids)}",
            f"Error Type: {kind}",
            f"Error Message: {message}",
            "-" * 50,
            "Traceback:Traceback (most recent call last):\n  File x.py, line 1\n" + kind,
            "=" * 50,
        ]
    )


class TestNonZeroExit:
    def test_the_reason_is_in_the_error_and_the_error_is_still_a_called_process_error(
        self, tmp_path
    ):
        cmd = _fake_openfold(
            """
            import sys
            print("Traceback (most recent call last):", file=sys.stderr)
            print("ValueError: Default checkpoint openbind-2025-06-30-174k not found in /w,"
                  " cowardly refusing to perform inference.", file=sys.stderr)
            sys.exit(1)
            """
        )
        with pytest.raises(subprocess.CalledProcessError) as info:
            _run_openfold_command(cmd, tmp_path)
        error = info.value
        assert isinstance(error, OpenFoldRunError)
        assert error.returncode == 1
        text = str(error)
        assert text.splitlines()[0].startswith("OpenFold3 exited with status 1: ValueError:")
        assert "cowardly refusing" in text
        assert "setup_openfold --non-interactive" in text
        assert "Last lines of stderr" in text
        assert "Command:" in text

    def test_the_reason_survives_the_200_character_cut_of_the_pipeline(self, tmp_path):
        """``cli.run._collect_failures`` keeps the first 200 characters of ``str(error)``."""
        cmd = _fake_openfold("import sys; sys.stderr.write('RuntimeError: boom\\n'); sys.exit(2)")
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd + ["--query_json=" + "x" * 400], tmp_path)
        assert "RuntimeError: boom" in str(info.value)[:200]

    def test_stderr_still_reaches_the_console_and_stdout_is_untouched(self, tmp_path, capfd):
        cmd = _fake_openfold(
            """
            import sys
            print("to stdout")
            print("to stderr", file=sys.stderr)
            sys.exit(1)
            """
        )
        with pytest.raises(OpenFoldRunError):
            _run_openfold_command(cmd, tmp_path)
        captured = capfd.readouterr()
        assert "to stdout" in captured.out
        assert "to stderr" in captured.err
        assert "to stderr" not in captured.out

    def test_progress_bars_do_not_bury_the_reason(self, tmp_path):
        cmd = _fake_openfold(
            r"""
            import sys
            sys.stderr.write("Predicting:  10%\rPredicting:  60%\rPredicting: 100%\n")
            sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory (2 GiB)\n")
            sys.exit(1)
            """
        )
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd, tmp_path)
        text = str(info.value).split("\nCommand:")[0]
        assert text.splitlines()[0].endswith("CUDA out of memory (2 GiB)")
        assert "Predicting: 100%" in text
        assert "Predicting:  10%" not in text
        assert "num_diffusion_samples" in text

    @pytest.mark.parametrize(
        "message, fragment",
        [
            ("ValueError: Checkpoint state_dict keys do not match model state_dict keys.",
             "OpenBind-0"),
            ("ValueError: Selected checkpoint x is not compatible with the currently installed "
             "OpenFold3 version 0.5.0.", "Preview2"),
            ("RuntimeError: unable to allocate shared memory", "--shm-size=8g"),
        ],
    )  # fmt: skip
    def test_known_failures_carry_their_fix(self, tmp_path, message, fragment):
        cmd = _fake_openfold(f"import sys; sys.stderr.write({message!r} + '\\n'); sys.exit(1)")
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd, tmp_path)
        assert "Hint:" in str(info.value)
        assert fragment in str(info.value)

    def test_an_unknown_failure_has_no_hint(self, tmp_path):
        cmd = _fake_openfold(
            "import sys; sys.stderr.write('KeyError: nothing known\\n'); sys.exit(1)"
        )
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd, tmp_path)
        assert "Hint:" not in str(info.value)

    def test_silence_still_gives_a_message(self, tmp_path):
        with pytest.raises(OpenFoldRunError, match="exited with status 3") as info:
            _run_openfold_command(_fake_openfold("import sys; sys.exit(3)"), tmp_path)
        assert info.value.stderr == ""

    def test_only_the_tail_of_a_long_stderr_is_kept(self, tmp_path):
        cmd = _fake_openfold(
            """
            import sys
            for i in range(5000):
                sys.stderr.write(f"noise line {i}\\n")
            sys.stderr.write("ValueError: the end\\n")
            sys.exit(1)
            """
        )
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd, tmp_path)
        assert len(info.value.stderr) <= _openfold_run._STDERR_TAIL_CHARS
        assert "ValueError: the end" in str(info.value)
        assert "noise line 0\n" not in info.value.stderr

    def test_bytes_that_are_not_utf8_do_not_stop_the_run(self, tmp_path):
        cmd = _fake_openfold(
            "import sys; sys.stderr.buffer.write(b'ValueError: bad \\xff\\xfe\\n'); sys.exit(1)"
        )
        with pytest.raises(OpenFoldRunError, match="ValueError: bad"):
            _run_openfold_command(cmd, tmp_path)

    def test_a_missing_executable_is_still_a_file_not_found_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _run_openfold_command(["definitely-not-a-run-openfold-binary"], tmp_path)


class TestSummary:
    def test_counts_and_failed_names(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(3, ["q3", "q1"]), encoding="utf-8")
        summary = _read_run_summary(tmp_path)
        assert (summary.total, summary.succeeded, summary.failed) == (3, 1, 2)
        assert summary.failed_queries == ("q1", "q3")

    def test_a_run_without_failures_lists_none(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(2, []), encoding="utf-8")
        summary = _read_run_summary(tmp_path)
        assert summary.failed_queries == () and summary.failed == 0

    def test_a_query_named_like_a_number_is_not_a_count(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(9, ["7"]), encoding="utf-8")
        summary = _read_run_summary(tmp_path)
        assert summary.failed == 1
        assert summary.failed_queries == ("7",)

    def test_no_summary_gives_none(self, tmp_path):
        assert _read_run_summary(tmp_path) is None
        assert _failed_query_reasons(tmp_path) == {}

    def test_a_summary_from_an_earlier_run_is_ignored(self, tmp_path):
        path = tmp_path / "summary.txt"
        path.write_text(_summary_text(1, ["q"]), encoding="utf-8")
        os.utime(path, (1_000_000, 1_000_000))
        assert _read_run_summary(tmp_path, not_before=2_000_000) is None
        assert _read_run_summary(tmp_path, not_before=0.0) is not None


class TestErrorLog:
    def _write_log(self, out, *entries, rank=0):
        logs = out / "logs"
        logs.mkdir(exist_ok=True)
        # OpenFold3 appends entries without a separating newline
        (logs / f"predict_err_rank{rank}.log").write_text("".join(entries), encoding="utf-8")
        return logs / f"predict_err_rank{rank}.log"

    def test_each_query_gets_the_error_of_its_entry(self, tmp_path):
        log = self._write_log(
            tmp_path,
            _error_log_entry(
                ["q1"], "OutOfMemoryError", "CUDA out of memory.\nTried to allocate 2 GiB"
            ),
            _error_log_entry(["q2", "q3"], "ValueError", "bad shape"),
        )
        reasons = _error_log_reasons(tmp_path)
        assert reasons["q1"] == (
            "OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
            log,
        )
        assert reasons["q2"][0] == reasons["q3"][0] == "ValueError: bad shape"

    def test_a_later_entry_replaces_an_earlier_one_for_the_same_query(self, tmp_path):
        self._write_log(
            tmp_path,
            _error_log_entry(["q"], "ValueError", "old run"),
            _error_log_entry(["q"], "OutOfMemoryError", "this run"),
        )
        assert _error_log_reasons(tmp_path)["q"][0] == "OutOfMemoryError: this run"

    def test_every_rank_file_is_read(self, tmp_path):
        self._write_log(tmp_path, _error_log_entry(["a"], "E0", "m0"), rank=0)
        self._write_log(tmp_path, _error_log_entry(["b"], "E1", "m1"), rank=1)
        assert set(_error_log_reasons(tmp_path)) == {"a", "b"}

    def test_no_logs_directory_is_no_reason(self, tmp_path):
        assert _error_log_reasons(tmp_path) == {}

    def test_a_failed_query_without_a_log_entry_still_gets_a_reason(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(2, ["q2"]), encoding="utf-8")
        reasons = _failed_query_reasons(tmp_path)
        assert list(reasons) == ["q2"]
        assert "summary.txt" in reasons["q2"]

    def test_the_reason_names_the_error_and_the_log(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(2, ["q2"]), encoding="utf-8")
        log = self._write_log(
            tmp_path, _error_log_entry(["q2"], "OutOfMemoryError", "CUDA out of memory")
        )
        reasons = _failed_query_reasons(tmp_path)
        assert reasons == {
            "q2": "OpenFold3 failed on this query: OutOfMemoryError: CUDA out of memory "
            "(logs/predict_err_rank0.log in the output of the run)"
        }
        assert str(tmp_path) not in reasons["q2"] and log.parent.name == "logs"


_WRITE_RUN = """
import pathlib, sys
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "summary.txt").write_text(sys.argv[2], encoding="utf-8")
if len(sys.argv) > 3:
    logs = out / "logs"
    logs.mkdir()
    (logs / "predict_err_rank0.log").write_text(sys.argv[3], encoding="utf-8")
"""


class TestExitZeroWithFailedQueries:
    def _run(self, out, summary, log=None):
        cmd = _fake_openfold(_WRITE_RUN) + [str(out), summary] + ([log] if log else [])
        return _run_openfold_command(cmd, out)

    def test_a_clean_run_reports_nothing(self, tmp_path):
        info = self._run(tmp_path / "o", _summary_text(2, []))
        assert info.failed_queries == {}

    def test_a_partial_failure_is_returned_and_logged_but_does_not_raise(self, tmp_path, caplog):
        entry = _error_log_entry(["q2"], "OutOfMemoryError", "CUDA out of memory")
        with caplog.at_level(logging.WARNING, logger=_openfold_run.logger.name):
            info = self._run(tmp_path / "o", _summary_text(3, ["q2"]), entry)
        assert list(info.failed_queries) == ["q2"]
        assert "OutOfMemoryError" in info.failed_queries["q2"]
        assert "failed on 1 query" in caplog.text and "q2" in caplog.text

    def test_a_run_in_which_every_query_failed_raises(self, tmp_path):
        entry = _error_log_entry(["only"], "OutOfMemoryError", "CUDA out of memory")
        out = tmp_path / "o"
        with pytest.raises(OpenFoldQueryError) as info:
            self._run(out, _summary_text(1, ["only"]), entry)
        error = info.value
        assert not isinstance(error, subprocess.CalledProcessError)
        assert list(error.failures) == ["only"]
        assert "failed on every query" in str(error)
        assert "CUDA out of memory" in str(error)
        assert "logs/predict_err_rank0.log" in str(error)
        # the store renames the folder of a run, so no recorded text may name it
        assert str(out) not in str(error) and error.output_dir == out

    def test_a_stale_summary_does_not_make_a_new_run_fail(self, tmp_path):
        out = tmp_path / "o"
        out.mkdir()
        stale = out / "summary.txt"
        stale.write_text(_summary_text(1, ["q"]), encoding="utf-8")
        os.utime(stale, (1_000_000, 1_000_000))
        info = _run_openfold_command(_fake_openfold("pass"), out)
        assert info.failed_queries == {}


class TestUserDefaultRunnerYaml:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENFOLD_CACHE", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def test_none_when_there_is_no_file(self):
        assert _user_default_runner_yaml() is None

    def test_the_home_cache_is_looked_at_by_default(self, tmp_path):
        default = tmp_path / "home" / ".openfold3" / "runner.yml"
        default.parent.mkdir(parents=True)
        default.write_text("experiment_settings: {}\n", encoding="utf-8")
        assert _user_default_runner_yaml() == default

    def test_openfold_cache_takes_precedence(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "runner.yml").write_text("{}\n", encoding="utf-8")
        home_default = tmp_path / "home" / ".openfold3" / "runner.yml"
        home_default.parent.mkdir(parents=True)
        home_default.write_text("{}\n", encoding="utf-8")
        monkeypatch.setenv("OPENFOLD_CACHE", str(cache))
        assert _user_default_runner_yaml() == cache / "runner.yml"

    def test_a_run_says_that_the_file_is_merged(self, tmp_path, monkeypatch, caplog):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "runner.yml").write_text("{}\n", encoding="utf-8")
        monkeypatch.setenv("OPENFOLD_CACHE", str(cache))
        with caplog.at_level(logging.WARNING, logger=_openfold_run.logger.name):
            info = _run_openfold_command(_fake_openfold("pass"), tmp_path / "o")
        assert info.user_default_runner_yaml == cache / "runner.yml"
        assert str(cache / "runner.yml") in caplog.text
        assert "user-default runner YAML" in caplog.text

    def test_a_run_without_the_file_is_quiet(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=_openfold_run.logger.name):
            info = _run_openfold_command(_fake_openfold("pass"), tmp_path / "o")
        assert info.user_default_runner_yaml is None
        assert "runner YAML" not in caplog.text


#: The warning that openfold3 0.5.0 logs when the features of a query cannot be built
#: (``single_datasets/inference.py``), as a real run printed it (1YCR, score, template CIF
#: rejected). No ``logs/predict_err_rank<N>.log`` exists for it.
_FEATURE_FAILURE = """\
----------------------------------------
Failed to process {query} with preferredException type: ValueError
Traceback: Traceback (most recent call last):
  File "openfold3/core/data/framework/single_datasets/inference.py", line 362, in __getitem__
    features = self.create_all_features(query)
  File "openfold3/core/data/primitives/structure/labels.py", line 459, in assign_entity_ids
    atom_array.set_annotation("entity_id", atom_array.label_entity_id.astype(int))
ValueError: invalid literal for int() with base 10: np.str_('.')
----------------------------------------
"""

_WRITE_FEATURE_FAILURE = """
import pathlib, sys
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "summary.txt").write_text(sys.argv[2], encoding="utf-8")
sys.stderr.write(sys.argv[3])
"""


class TestFailureWhileTheFeaturesAreBuilt:
    """OpenFold3 writes no log for it: the exception is only in what it printed."""

    def _run(self, out, names, failed, stderr_for):
        stderr = "".join(stderr_for(name) for name in failed)
        cmd = _fake_openfold(_WRITE_FEATURE_FAILURE) + [
            str(out),
            _summary_text(len(names), failed),
            "x" * 12000 + "\n" + stderr + "progress bar noise\n" * 800,  # far from the end
        ]
        return _run_openfold_command(cmd, out)

    def test_the_exception_line_is_the_reason(self, tmp_path):
        out = tmp_path / "o"
        with pytest.raises(OpenFoldQueryError) as info:
            self._run(out, ["q"], ["q"], lambda name: _FEATURE_FAILURE.format(query=name))
        reason = info.value.failures["q"]
        assert reason == (
            "OpenFold3 failed while it built the features of this query: "
            "ValueError: invalid literal for int() with base 10: np.str_('.')"
        )
        assert "invalid literal for int()" in str(info.value)
        assert "see " not in reason and "logs directory" not in reason

    def test_no_reason_names_the_work_directory(self, tmp_path):
        out = tmp_path / "work" / "predictions"
        with pytest.raises(OpenFoldQueryError) as info:
            self._run(out, ["q"], ["q"], lambda name: _FEATURE_FAILURE.format(query=name))
        assert str(tmp_path) not in str(info.value) + info.value.failures["q"]

    def test_each_failed_query_gets_its_own_exception(self, tmp_path):
        def stderr_for(name):
            kind = "KeyError: 'x'" if name == "b" else "ValueError: bad shape"
            text = _FEATURE_FAILURE.format(query=name)
            return text.replace(
                "ValueError: invalid literal for int() with base 10: np.str_('.')", kind
            ).replace("Exception type: ValueError", f"Exception type: {kind.split(':')[0]}")

        info = self._run(tmp_path / "o", ["a", "b", "c"], ["a", "b"], stderr_for)
        assert info.failed_queries["a"].endswith("ValueError: bad shape")
        assert info.failed_queries["b"].endswith("KeyError: 'x'")
        assert "c" not in info.failed_queries

    def test_a_forward_pass_error_log_still_wins(self, tmp_path):
        """A query with a log file is explained by it, not by the warning."""
        out = tmp_path / "o"
        out.mkdir()
        (out / "summary.txt").write_text(_summary_text(2, ["q"]), encoding="utf-8")
        logs = out / "logs"
        logs.mkdir()
        (logs / "predict_err_rank0.log").write_text(
            _error_log_entry(["q"], "OutOfMemoryError", "CUDA out of memory"), encoding="utf-8"
        )
        notes = StreamNotes()
        notes.feed(_FEATURE_FAILURE.format(query="q"))
        notes.finish()
        assert "OutOfMemoryError" in _failed_query_reasons(out, notes=notes)["q"]

    def test_without_the_warning_the_reason_says_where_to_look_without_a_path(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(2, ["q2"]), encoding="utf-8")
        reason = _failed_query_reasons(tmp_path)["q2"]
        assert "summary.txt" in reason and "stderr" in reason and str(tmp_path) not in reason


class TestNotesOfFailedQueries:
    def test_the_exception_is_the_last_line_that_starts_with_its_type(self):
        notes = StreamNotes()
        notes.feed(_FEATURE_FAILURE.format(query="q"))
        notes.finish()
        assert notes.failed_queries == {
            "q": "ValueError: invalid literal for int() with base 10: np.str_('.')"
        }

    def test_a_block_cut_by_the_end_of_the_stream_is_kept(self):
        notes = StreamNotes()
        notes.feed(_FEATURE_FAILURE.format(query="q").rsplit("----", 2)[0])
        notes.finish()
        assert "invalid literal" in notes.failed_queries["q"]

    def test_without_a_traceback_line_the_type_is_the_reason(self):
        notes = StreamNotes()
        notes.feed(
            "Failed to process q with preferredException type: MemoryError\n" + "-" * 40 + "\n"
        )
        notes.finish()
        assert notes.failed_queries == {"q": "MemoryError"}

    def test_the_text_of_two_queries_is_kept_apart(self):
        notes = StreamNotes()
        notes.feed(_FEATURE_FAILURE.format(query="q1") + _FEATURE_FAILURE.format(query="q2"))
        notes.finish()
        assert set(notes.failed_queries) == {"q1", "q2"}


class TestTheHeadlineOfAMissingDefaultCheckpoint:
    """openfold3 0.5.0: the pydantic line says nothing; the reason is a later line."""

    _STDERR = (
        "Traceback (most recent call last):\n"
        '  File "run_openfold.py", line 223, in predict\n'
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for "
        "InferenceExperimentConfig\n"
        "  Value error, Default checkpoint openbind-2025-06-30-174k not found in "
        "/home/u/.openfold3, cowardly refusing to perform inference.Please run `setup_openfold` "
        "to download the current default\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/value_error\n"
        "ERROR conda.cli.main_run:execute(125): `conda run run_openfold predict` failed\n"
    )

    def test_the_headline_is_the_reason_and_not_the_validation_error(self, tmp_path):
        cmd = _fake_openfold(f"import sys; sys.stderr.write({self._STDERR!r}); sys.exit(1)")
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd, tmp_path)
        headline = str(info.value).splitlines()[0]
        assert headline.startswith(
            "OpenFold3 exited with status 1: Value error, Default checkpoint"
        )
        assert "cowardly refusing" in headline
        assert "1 validation error" not in headline
        assert "setup_openfold --non-interactive" in str(info.value)  # the hint stays

    def test_another_failure_keeps_its_exception_line(self, tmp_path):
        cmd = _fake_openfold("import sys; sys.stderr.write('RuntimeError: boom\\n'); sys.exit(2)")
        with pytest.raises(OpenFoldRunError) as info:
            _run_openfold_command(cmd, tmp_path)
        assert str(info.value).splitlines()[0].endswith("RuntimeError: boom")


class TestAMissingCondaEnvironmentIsNamed:
    """``conda run -n <env>`` of a typo gives "cannot be started on this machine" and no name."""

    @pytest.fixture
    def no_openfold3(self, monkeypatch):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: None
        )

    def test_the_runner_says_which_environment_it_looked_in(self, no_openfold3, monkeypatch):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/conda")
        reason = OpenFold3Runner(conda_env="no_such_env_xyz").unavailable_reason()
        assert "'no_such_env_xyz'" in reason
        assert "does not exist or has no openfold3" in reason
        assert "conda run -n no_such_env_xyz python" in reason

    def test_without_conda_the_reason_says_so(self, no_openfold3, monkeypatch):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        monkeypatch.setattr("shutil.which", lambda name: None)
        reason = OpenFold3Runner(conda_env="of3").unavailable_reason()
        assert "conda is not on PATH" in reason and "'of3'" in reason

    def test_without_an_environment_the_reason_is_the_executable(self, monkeypatch):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        monkeypatch.setattr("shutil.which", lambda name: None)
        assert "run_openfold is not on PATH" in OpenFold3Runner().unavailable_reason()

    def test_an_available_runner_has_no_reason(self, monkeypatch):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
        )
        assert OpenFold3Runner(conda_env="openfold3").unavailable_reason() is None

    def test_the_text_of_the_pipeline_names_the_environment(self, no_openfold3, tmp_path):
        from pathlib import Path

        from binding_metrics.cli.prediction import run_single_prediction

        p53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
        block, _ = run_single_prediction(
            "of3",
            p53,
            tmp_path,
            "s1",
            binder_chain="B",
            receptor_chain="A",
            openfold_conda_env="no_such_env_xyz",
        )
        assert "the of3 model cannot be started on this machine" in block["error"]
        assert "Why: " in block["error"] and "'no_such_env_xyz'" in block["error"]
        assert "--openfold-conda-env" in block["error"]  # the hint stays

    def test_a_caller_of_the_session_gets_the_reason_from_the_store_text(
        self, no_openfold3, tmp_path, monkeypatch
    ):
        from pathlib import Path

        from binding_metrics.predictors import OpenFold3Runner, PredictionSession, PredictionStore
        from binding_metrics.predictors.store import PredictionUnavailableError

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/conda")
        p53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
        runner = OpenFold3Runner(conda_env="no_such_env_xyz")
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        request = runner.make_request(p53, name="s1", binder_chain="B", receptor_chain="A")
        with pytest.raises(PredictionUnavailableError) as info:
            session.entry(request)
        assert "\nWhy: " in str(info.value) and "'no_such_env_xyz'" in str(info.value)

    def test_the_text_of_the_pipeline_has_the_reason_once(self, no_openfold3, tmp_path):
        from pathlib import Path

        from binding_metrics.cli.prediction import run_single_prediction

        p53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
        block, _ = run_single_prediction(
            "of3",
            p53,
            tmp_path,
            "s1",
            binder_chain="B",
            receptor_chain="A",
            openfold_conda_env="no_such_env_xyz",
        )
        assert block["error"].count("Why: ") == 1

    def test_a_runner_without_the_method_gives_the_old_text(self):
        from binding_metrics.cli.prediction import error_text
        from binding_metrics.predictors.store import PredictionUnavailableError

        error = PredictionUnavailableError("the x model cannot be started on this machine")
        assert "Why:" not in error_text(error, "of3", runner=object())
        assert "Why:" not in error_text(error, "of3")


class TestOutputUnderAnotherKey:
    """Mode ``predict``: the output is named after the key of the query file."""

    def _folder(self, tmp_path, *keys):
        from pathlib import Path

        from tests.test_feat_c_support import write_of3_output

        p53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
        predictions = tmp_path / "predictions"
        predictions.mkdir()
        for key in keys:
            write_of3_output(predictions, key, p53)
        (predictions / "msas").mkdir(exist_ok=True)  # not a query
        (predictions / "logs").mkdir(exist_ok=True)
        return predictions

    def _request(self, tmp_path, name="requested"):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        query = tmp_path / "query.json"
        query.write_text('{"queries": {"in_the_file": {"chains": []}}}', encoding="utf-8")
        return OpenFold3Runner().make_request(query, name=name, mode="predict")

    def test_the_keys_that_have_output_are_named(self, tmp_path):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        predictions = self._folder(tmp_path, "in_the_file")
        with pytest.raises(RuntimeError) as info:
            OpenFold3Runner._require_output(self._request(tmp_path), predictions)
        text = str(info.value)
        assert text.startswith("OpenFold3 wrote no output for query 'requested'")
        assert "holds output for 'in_the_file'" in text
        assert "key of the query in the query file" in text
        assert "msas" not in text and "logs" not in text and str(tmp_path) not in text

    def test_two_keys_are_both_named(self, tmp_path):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        predictions = self._folder(tmp_path, "b_query", "a_query")
        with pytest.raises(RuntimeError, match="holds output for 'a_query', 'b_query'"):
            OpenFold3Runner._require_output(self._request(tmp_path), predictions)

    def test_a_folder_without_output_names_no_key(self, tmp_path):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        predictions = self._folder(tmp_path)
        with pytest.raises(RuntimeError) as info:
            OpenFold3Runner._require_output(self._request(tmp_path), predictions)
        assert "holds output" not in str(info.value)

    def test_a_query_that_failed_keeps_its_own_reason(self, tmp_path):
        from binding_metrics.predictors.of3_runner import _no_output_message

        request = self._request(tmp_path, name="q")
        text = _no_output_message(request, {"q": "OpenFold3 failed: boom"}, ["other"])
        assert text.endswith(": OpenFold3 failed: boom") and "holds output" not in text

    def test_a_request_with_the_right_key_finds_its_output(self, tmp_path):
        from binding_metrics.predictors.of3_runner import OpenFold3Runner

        predictions = self._folder(tmp_path, "in_the_file")
        OpenFold3Runner._require_output(self._request(tmp_path, name="in_the_file"), predictions)
