"""The publishing workflows run only for a tag whose commit passed CI (issue #174).

The workflow files are parsed, and the scripts of their `run:` steps are executed here with
the GitHub-side calls replaced by a stub of `gh`, so the decisions they make are exercised
without a runner.
"""

import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
VERIFY_CALL = "./.github/workflows/verify-ci.yml"
PUBLISHING_FILES = ["release.yml", "docker-publish.yml"]
ALL_FILES = [*PUBLISHING_FILES, "verify-ci.yml"]

pytestmark = pytest.mark.skipif(
    not WORKFLOWS.is_dir(), reason="no .github/workflows in this checkout"
)


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    """The `on:` table; YAML 1.1 reads the bare key `on` as the boolean True."""
    return workflow["on"] if "on" in workflow else workflow[True]


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _publishes(job: dict) -> bool:
    for step in job.get("steps", []):
        uses = step.get("uses", "")
        if uses.startswith("docker/login-action"):
            return True
        if uses.startswith("docker/build-push-action") and step["with"].get("push") is True:
            return True
        if "gh release create" in step.get("run", ""):
            return True
    return False


def _step_with_run(job: dict, needle: str) -> dict:
    (step,) = [s for s in job["steps"] if needle in s.get("run", "")]
    return step


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]


def _run_script(step: dict, tmp_path: Path, env: dict, cwd: Path) -> subprocess.CompletedProcess:
    """Run the `run:` text of a step the way a runner does (`bash -eo pipefail`).

    The constants of the step's `env:` are applied; `env` supplies what the runner would
    expand from an expression.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "python"
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    constants = {k: str(v) for k, v in step.get("env", {}).items() if "${{" not in str(v)}
    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        **constants,
        **env,
    }
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )


class TestTriggers:
    def test_release_runs_on_version_tags_only(self):
        assert _triggers(_workflow("release.yml")) == {"push": {"tags": ["v*"]}}

    def test_docker_publish_runs_on_main_and_version_tags(self):
        assert _triggers(_workflow("docker-publish.yml")) == {
            "push": {"branches": ["main"], "tags": ["v*"]}
        }


class TestPermissions:
    @pytest.mark.parametrize("name", ALL_FILES)
    def test_the_file_and_every_job_declare_permissions(self, name):
        workflow = _workflow(name)
        assert "permissions" in workflow, f"{name} has no top-level permissions"
        for job_id, job in workflow["jobs"].items():
            assert "permissions" in job, f"{name}: job {job_id} has no permissions"

    def test_only_the_release_job_can_write(self):
        writers = []
        for name in ALL_FILES:
            workflow = _workflow(name)
            scopes = [("<top>", workflow["permissions"])]
            scopes += [(job_id, job["permissions"]) for job_id, job in workflow["jobs"].items()]
            for job_id, permissions in scopes:
                writers += [
                    (name, job_id, scope)
                    for scope, level in (permissions or {}).items()
                    if level == "write"
                ]
        assert writers == [("release.yml", "release", "contents")]

    def test_the_verification_workflow_reads_actions_and_contents_only(self):
        workflow = _workflow("verify-ci.yml")
        for permissions in [workflow["permissions"]] + [
            job["permissions"] for job in workflow["jobs"].values()
        ]:
            assert permissions == {"actions": "read", "contents": "read"}

    @pytest.mark.parametrize("name", PUBLISHING_FILES)
    def test_the_verification_call_is_granted_what_the_workflow_reads(self, name):
        calls = [job for job in _workflow(name)["jobs"].values() if job.get("uses") == VERIFY_CALL]
        assert len(calls) == 1
        assert calls[0]["permissions"] == {"actions": "read", "contents": "read"}


class TestPublishingNeedsVerification:
    def test_verify_ci_is_a_reusable_workflow(self):
        assert "workflow_call" in _triggers(_workflow("verify-ci.yml"))

    @pytest.mark.parametrize("name", PUBLISHING_FILES)
    def test_every_publishing_job_needs_the_verification_job(self, name):
        jobs = _workflow(name)["jobs"]
        verify_ids = [job_id for job_id, job in jobs.items() if job.get("uses") == VERIFY_CALL]
        publishing = [job_id for job_id, job in jobs.items() if _publishes(job)]
        assert len(verify_ids) == 1
        assert publishing, f"{name}: no job recognised as publishing, so this test checks nothing"
        for job_id in publishing:
            assert verify_ids[0] in _needs(jobs[job_id]), f"{name}: {job_id} publishes unverified"

    @pytest.mark.parametrize("name", PUBLISHING_FILES)
    def test_no_job_lets_a_failed_verification_through(self, name):
        for job_id, job in _workflow(name)["jobs"].items():
            condition = str(job.get("if", ""))
            assert "always()" not in condition, f"{name}: {job_id} runs after a failed verify"

    def test_verification_steps_do_nothing_on_a_branch_push(self):
        # docker-publish.yml also runs for pushes to main, where there is no tag to verify.
        (job,) = _workflow("verify-ci.yml")["jobs"].values()
        for step in job["steps"]:
            assert "github.ref_type" in str(step.get("if", "")), step


class TestMainPushImages:
    def test_a_push_to_main_publishes_no_latest_tag(self):
        # ci.yml runs on the same commit at the same time, so its result is unknown here.
        steps = _workflow("docker-publish.yml")["jobs"]["docker"]["steps"]
        for step in steps:
            if step.get("uses", "").startswith("docker/metadata-action"):
                for line in step["with"]["tags"].splitlines():
                    assert not ("latest" in line and "is_default_branch" in line), line


class TestTagMatchesProjectVersion:
    @staticmethod
    def _step():
        (job,) = _workflow("verify-ci.yml")["jobs"].values()
        return _step_with_run(job, "pyproject.toml")

    def _check(self, tmp_path, tag, version):
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8"
        )
        return _run_script(self._step(), tmp_path, {"TAG": tag}, tmp_path)

    def test_no_expression_is_interpolated_into_the_script(self):
        assert "${{" not in self._step()["run"]

    @pytest.mark.parametrize(
        ("tag", "version"),
        [("v0.2.0", "0.2.0"), ("v0.2.0-rc1", "0.2.0rc1"), ("v1.0.0", "1.0.0")],
    )
    def test_the_tag_of_the_project_version_passes(self, tmp_path, tag, version):
        result = self._check(tmp_path, tag, version)
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize(
        ("tag", "version"),
        [("v0.2.1", "0.2.0"), ("0.2.0", "0.2.0"), ("v0.2", "0.2.1"), ("vnext", "0.2.0")],
    )
    def test_any_other_tag_fails(self, tmp_path, tag, version):
        result = self._check(tmp_path, tag, version)
        assert result.returncode != 0
        assert tag in result.stdout + result.stderr

    def test_the_tag_of_the_version_in_this_checkout_passes(self, tmp_path):
        version = _pyproject()["version"]
        result = self._check(tmp_path, f"v{version}", version)
        assert result.returncode == 0, result.stdout + result.stderr


SHA = "a" * 40
OTHER_SHA = "b" * 40


def _run(event="push", branch="main", status="completed", conclusion="success", sha=SHA, number=1):
    return {
        "name": "CI",
        "run_number": number,
        "event": event,
        "head_branch": branch,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.example/runs/{number}",
    }


class TestCiRunOfTheTaggedCommit:
    @staticmethod
    def _step():
        (job,) = _workflow("verify-ci.yml")["jobs"].values()
        return _step_with_run(job, "gh api")

    def _check(self, tmp_path, runs, *, gh_exit=0):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        reply = tmp_path / "reply.json"
        reply.write_text(
            json.dumps({"total_count": len(runs), "workflow_runs": runs}), encoding="utf-8"
        )
        log = tmp_path / "gh_args.txt"
        gh = bin_dir / "gh"
        gh.write_text(
            '#!/bin/sh\necho "$@" >> "$GH_ARGS_LOG"\n'
            'if [ "$GH_EXIT" != 0 ]; then echo "HTTP 403" >&2; exit "$GH_EXIT"; fi\n'
            'cat "$GH_REPLY"\n',
            encoding="utf-8",
        )
        gh.chmod(0o755)
        env = {
            "GITHUB_SHA": SHA,
            "GITHUB_REPOSITORY": "owner/repo",
            "DEFAULT_BRANCH": "main",
            "GH_TOKEN": "unused",
            "RUNNER_TEMP": str(tmp_path),
            "GH_REPLY": str(reply),
            "GH_ARGS_LOG": str(log),
            "GH_EXIT": str(gh_exit),
        }
        result = _run_script(self._step(), tmp_path, env, tmp_path)
        return result, log

    def test_no_expression_is_interpolated_into_the_script(self):
        assert "${{" not in self._step()["run"]

    def test_a_successful_run_of_the_push_to_main_passes(self, tmp_path):
        result, log = self._check(tmp_path, [_run()])
        assert result.returncode == 0, result.stdout + result.stderr
        request = log.read_text(encoding="utf-8")
        assert "repos/owner/repo/actions/workflows/ci.yml/runs" in request
        assert f"head_sha={SHA}" in request

    def test_a_failed_retry_after_a_success_does_not_hide_the_success(self, tmp_path):
        runs = [_run(conclusion="failure", number=1), _run(number=2)]
        assert self._check(tmp_path, runs)[0].returncode == 0

    @pytest.mark.parametrize(
        "runs",
        [
            pytest.param([], id="no-run"),
            pytest.param([_run(status="in_progress", conclusion=None)], id="still-running"),
            pytest.param([_run(status="queued", conclusion=None)], id="queued"),
            pytest.param([_run(conclusion="failure")], id="failed"),
            pytest.param([_run(conclusion="cancelled")], id="cancelled"),
            pytest.param([_run(conclusion="skipped")], id="skipped"),
            pytest.param([_run(event="pull_request", branch="feature")], id="pull-request-only"),
            pytest.param([_run(event="pull_request")], id="pull-request-from-a-fork-main"),
            pytest.param([_run(event="workflow_dispatch")], id="manual-run"),
            pytest.param([_run(status="in_progress")], id="rerun-in-progress"),
            pytest.param([_run(branch="feature")], id="push-to-another-branch"),
            pytest.param([_run(sha=OTHER_SHA)], id="another-commit"),
        ],
    )
    def test_anything_but_a_completed_success_on_main_fails(self, tmp_path, runs):
        result, _ = self._check(tmp_path, runs)
        assert result.returncode != 0
        assert SHA in result.stdout + result.stderr

    def test_a_failed_listing_fails_the_job(self, tmp_path):
        result, _ = self._check(tmp_path, [_run()], gh_exit=1)
        assert result.returncode != 0
        assert "403" in result.stdout + result.stderr


def _build_job_id(jobs: dict) -> str:
    (job_id,) = [
        job_id
        for job_id, job in jobs.items()
        if any("python -m build" in step.get("run", "") for step in job.get("steps", []))
    ]
    return job_id


def _smoke_job(jobs: dict) -> dict:
    (job,) = [job for job in jobs.values() if "strategy" in job]
    return job


class TestCleanInstallSmoke:
    @staticmethod
    def _jobs():
        return _workflow("release.yml")["jobs"]

    def test_the_release_publishes_the_files_the_smoke_job_installed(self):
        jobs = self._jobs()
        build_id = _build_job_id(jobs)
        (release_id,) = [job_id for job_id, job in jobs.items() if _publishes(job)]
        release = jobs[release_id]
        assert release_id != build_id
        assert not any("python -m build" in step.get("run", "") for step in release["steps"])
        assert any(
            step.get("uses", "").startswith("actions/download-artifact")
            for step in release["steps"]
        )
        assert any(
            step.get("uses", "").startswith("actions/upload-artifact")
            for step in jobs[build_id]["steps"]
        )

    def test_the_order_is_verify_then_build_then_smoke_then_release(self):
        jobs = self._jobs()
        build_id = _build_job_id(jobs)
        verify_id = next(job_id for job_id, job in jobs.items() if job.get("uses") == VERIFY_CALL)
        smoke = _smoke_job(jobs)
        (release_id,) = [job_id for job_id, job in jobs.items() if _publishes(job)]
        assert verify_id in _needs(jobs[build_id])
        assert {verify_id, build_id} <= set(_needs(smoke))
        smoke_id = next(job_id for job_id, job in jobs.items() if job is smoke)
        assert {verify_id, smoke_id} <= set(_needs(jobs[release_id]))

    def test_the_matrix_is_every_python_version_the_project_declares(self):
        project = _pyproject()
        declared = {
            c.rsplit("::", 1)[1].strip()
            for c in project["classifiers"]
            if c.startswith("Programming Language :: Python :: 3.")
        }
        assert declared, "pyproject.toml lists no Python 3.x classifier"
        matrix = _smoke_job(self._jobs())["strategy"]["matrix"]["python-version"]
        assert all(isinstance(version, str) for version in matrix), "unquoted 3.10 reads as 3.1"
        assert set(matrix) == declared
        lowest = project["requires-python"].removeprefix(">=").strip()
        assert min(matrix, key=lambda v: tuple(map(int, v.split(".")))) == lowest

    def test_one_failing_version_does_not_hide_the_others(self):
        assert _smoke_job(self._jobs())["strategy"]["fail-fast"] is False

    def test_the_virtualenv_and_every_command_on_it_are_outside_the_source_tree(self):
        steps = _smoke_job(self._jobs())["steps"]
        on_the_install = [s for s in steps if "smoke-venv" in s.get("run", "")]
        assert len(on_the_install) == 3  # install, import, console scripts
        for step in on_the_install:
            assert "${{ runner.temp }}" in step["working-directory"], step
            assert "GITHUB_WORKSPACE/smoke-venv" not in step["run"]
        (create,) = [s for s in steps if "python -m venv" in s.get("run", "")]
        assert '"$RUNNER_TEMP/smoke-venv"' in create["run"]

    def test_the_wheel_is_installed_and_not_the_source_tree(self):
        steps = _smoke_job(self._jobs())["steps"]
        (install,) = [s for s in steps if " pip install " in s.get("run", "")]
        assert "/dist/*.whl" in install["run"]
        assert "pip install ." not in install["run"] and " -e " not in install["run"]

    def test_the_installed_package_is_imported_and_its_version_printed(self, tmp_path):
        pytest.importorskip("binding_metrics")
        step = _step_with_run(_smoke_job(self._jobs()), "import binding_metrics")
        venv_bin = tmp_path / "smoke-venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8"
        )
        (venv_bin / "python").chmod(0o755)
        result = _run_script(step, tmp_path, {"RUNNER_TEMP": str(tmp_path)}, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        import binding_metrics

        assert result.stdout.strip() == binding_metrics.__version__

    def _run_scripts_step(self, tmp_path, workspace, present, failing=()):
        """Run the console-script step against stubs; return the process and the log."""
        venv_bin = tmp_path / "temp" / "smoke-venv" / "bin"
        venv_bin.mkdir(parents=True)
        log = tmp_path / "calls.txt"
        for name in present:
            stub = venv_bin / name
            code = 1 if name in failing else 0
            stub.write_text(
                f'#!/bin/sh\necho "{name} $@" >> {shlex.quote(str(log))}\nexit {code}\n',
                encoding="utf-8",
            )
            stub.chmod(0o755)
        step = _step_with_run(_smoke_job(self._jobs()), "--help")
        env = {"RUNNER_TEMP": str(tmp_path / "temp"), "GITHUB_WORKSPACE": str(workspace)}
        result = _run_script(step, tmp_path, env, tmp_path)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls

    def test_every_console_script_of_pyproject_is_run_with_help(self, tmp_path):
        names = list(_pyproject()["scripts"])
        assert len(names) > 10
        result, calls = self._run_scripts_step(tmp_path, REPO_ROOT, names)
        assert result.returncode == 0, result.stdout + result.stderr
        assert sorted(calls) == sorted(f"{name} --help" for name in names)

    def test_the_scripts_come_from_the_pyproject_of_the_checkout(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text(
            '[project]\nname = "x"\nversion = "1"\n[project.scripts]\nx-one = "x:main"\n',
            encoding="utf-8",
        )
        result, calls = self._run_scripts_step(tmp_path, workspace, ["x-one", "stray"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert calls == ["x-one --help"]

    def test_a_script_that_is_missing_from_the_wheel_fails_the_job(self, tmp_path):
        names = list(_pyproject()["scripts"])
        result, calls = self._run_scripts_step(tmp_path, REPO_ROOT, names[1:])
        assert result.returncode != 0
        assert names[0] in result.stdout + result.stderr
        assert len(calls) == len(names) - 1  # the others still ran

    def test_a_script_whose_help_fails_fails_the_job(self, tmp_path):
        names = list(_pyproject()["scripts"])
        result, _ = self._run_scripts_step(tmp_path, REPO_ROOT, names, failing=[names[2]])
        assert result.returncode != 0
        assert names[2] in result.stdout + result.stderr

    def test_a_pyproject_without_console_scripts_fails_the_job(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text(
            '[project]\nname = "x"\nversion = "1"\n[project.scripts]\n', encoding="utf-8"
        )
        result, _ = self._run_scripts_step(tmp_path, workspace, [])
        assert result.returncode != 0
