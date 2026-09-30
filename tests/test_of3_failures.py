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
            f"(see {log})"
        }


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
        assert str(out / "logs" / "predict_err_rank0.log") in str(error)

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
