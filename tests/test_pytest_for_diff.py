"""scripts/pytest_for_diff.py picks the areas and test files for a diff.

Each case builds a small git repository in a temporary directory (a package with a few
modules, tests that import them, a README and a Dockerfile), changes some files and checks the
selection and the pytest command lines. Pytest itself is never run: the commands are only
printed, or handed to a fake ``subprocess.run``.
"""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "pytest_for_diff.py"
GIT_ENVIRONMENT = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("pytest_for_diff", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pytest_for_diff"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("pytest_for_diff", None)


def git(repo, *args):
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false"]
        + ["-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=GIT_ENVIRONMENT,
        check=True,
    )
    return done.stdout.strip()


def write(repo, files):
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


def commit(repo, message="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


BASE_FILES = {
    "src/binding_metrics/__init__.py": "",
    "src/binding_metrics/core/__init__.py": "",
    "src/binding_metrics/core/system.py": "def prep(): ...\n",
    "src/binding_metrics/poses/__init__.py": "",
    "src/binding_metrics/poses/frame.py": "def frame(): ...\n",
    "src/binding_metrics/poses/report.py": "from binding_metrics.poses import frame\n",
    "src/binding_metrics/cli/__init__.py": "def add_flag(): ...\n",
    "src/binding_metrics/cli/run.py": "from binding_metrics.core import system\n",
    "src/binding_metrics/chain/__init__.py": "",
    "src/binding_metrics/chain/a.py": "def a(): ...\n",
    "src/binding_metrics/chain/b.py": "def b():\n    from binding_metrics.chain import a\n",
    "src/binding_metrics/chain/c.py": "def c():\n    from binding_metrics.chain import b\n",
    "src/binding_metrics/metrics/__init__.py": "",
    "src/binding_metrics/metrics/geometry.py": "def ramachandran(): ...\n",
    "src/binding_metrics/metrics/sasa.py": "def sasa(): ...\n",
    "src/binding_metrics/metrics/registry.py": """
        import importlib
        PATHS = ["binding_metrics.metrics.geometry:ramachandran"]
        def load(path):
            module, name = path.split(":")
            return getattr(importlib.import_module(module), name)
        """,
    "tests/__init__.py": "",
    "tests/conftest.py": "",
    "tests/support.py": "from binding_metrics.poses.frame import frame\n",
    "tests/test_frame.py": "from binding_metrics.poses.frame import frame\n",
    "tests/test_run.py": "from binding_metrics.cli.run import main\n",
    "tests/test_prep.py": "from binding_metrics.core.system import prep\n",
    "tests/test_uses_support.py": "from tests.support import frame\n",
    "tests/test_chain_b.py": "from binding_metrics.chain import b\n",
    "tests/test_chain_c.py": "from binding_metrics.chain import c\n",
    "tests/test_flags.py": "from binding_metrics.cli import add_flag\n",
    "tests/test_registry_use.py": "from binding_metrics.metrics.registry import load\n",
    "tests/test_sasa.py": "from binding_metrics.metrics.sasa import sasa\n",
    "tests/test_guide.py": 'from pathlib import Path\nGUIDE = Path("docs") / "guide.md"\n',
    "tests/test_docker.py": 'TEXT = open("Dockerfile").read()\n',
    "docs/guide.md": "Guide\n",
    "README.md": "Readme\n",
    "Dockerfile": "FROM scratch\n",
    "environment.yml": "name: x\n",
    "pyproject.toml": "[project]\nname = 'x'\n",
}


@pytest.fixture
def repo(tmp_path, script):
    """A repository on branch ``feature`` that left ``main`` (and ``origin/main``) at one commit."""
    root = tmp_path / "project"
    root.mkdir()
    write(root, BASE_FILES)
    write(root, {guard: "def test_ok():\n    pass\n" for guard in script.GUARD_TESTS})
    git(root, "init", "-q", "-b", "main")
    commit(root, "base")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(root, "switch", "-q", "-c", "feature")
    return root


def selection(script, repo, *args, include_slow=False, pytest_args=()):
    """Plan and commands for the changes of ``repo``, as ``main`` computes them."""
    parsed, extra = script.parse_arguments(["--repo", str(repo), *args])
    mode = "staged" if parsed.staged else "working-tree" if parsed.working_tree else "base"
    changed, _ = script.changed_files(repo, mode, parsed.base)
    project = script.load_areas().Project(repo)
    plan = script.plan_for(project, changed, max_lazy=parsed.max_lazy)
    return changed, plan, script.build_commands(plan, include_slow, [*extra, *pytest_args])


def test_the_guard_tests_exist_in_this_repository(script):
    assert len(set(script.GUARD_TESTS)) == len(script.GUARD_TESTS)
    for guard in script.GUARD_TESTS:
        assert (REPO / guard).is_file(), f"guard test {guard} is gone: update GUARD_TESTS"


def test_the_script_shares_its_table_and_scan_with_the_markers(script):
    areas = script.load_areas()
    assert areas is sys.modules["_areas"]
    assert areas.AREA_OF_MODULE and areas.Project


def test_the_default_base_is_the_merge_base_with_origin_main(script, repo):
    initial = git(repo, "rev-parse", "HEAD")
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})
    commit(repo)  # committed on the branch
    write(repo, {"src/binding_metrics/core/system.py": "def prep(): return 1\n"})  # not committed
    write(repo, {"docs/new.md": "New\n"})  # untracked

    base, how = script.default_base(repo)
    assert (base, how) == (initial, "merge-base with origin/main")
    changed, _ = script.changed_files(repo, "base", None)
    assert changed == [
        "docs/new.md",
        "src/binding_metrics/core/system.py",
        "src/binding_metrics/poses/frame.py",
    ]


def test_without_a_main_branch_the_base_is_the_previous_commit(script, tmp_path):
    root = tmp_path / "lone"
    root.mkdir()
    write(root, {"a.txt": "1\n", "b.txt": "1\n"})
    git(root, "init", "-q", "-b", "work")
    commit(root, "first")
    write(root, {"a.txt": "2\n"})
    commit(root, "second")
    write(root, {"b.txt": "2\n"})

    assert script.default_base(root) == ("HEAD~1", "previous commit")
    assert script.changed_files(root, "base", None)[0] == ["a.txt", "b.txt"]


def test_in_a_repository_with_one_commit_every_file_is_changed(script, tmp_path):
    root = tmp_path / "single"
    root.mkdir()
    write(root, {"a.txt": "1\n", "b/c.txt": "1\n"})
    git(root, "init", "-q", "-b", "work")
    commit(root, "only")

    assert script.default_base(root)[1] == "empty tree"
    assert script.changed_files(root, "base", None)[0] == ["a.txt", "b/c.txt"]


def test_staged_and_working_tree_and_explicit_base(script, repo):
    write(repo, {"README.md": "Readme 2\n", "Dockerfile": "FROM scratch\nRUN true\n"})
    git(repo, "add", "README.md")
    write(repo, {"docs/untracked.md": "x\n"})

    assert script.changed_files(repo, "staged", None)[0] == ["README.md"]
    assert script.changed_files(repo, "working-tree", None)[0] == [
        "Dockerfile",
        "README.md",
        "docs/untracked.md",
    ]
    commit(repo)
    assert script.changed_files(repo, "working-tree", None)[0] == []
    assert script.changed_files(repo, "base", "HEAD~1")[0] == [
        "Dockerfile",
        "README.md",
        "docs/untracked.md",
    ]


def test_a_source_file_runs_the_tests_of_the_modules_that_import_it(script, repo):
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})
    changed, plan, commands = selection(script, repo)

    assert changed == ["src/binding_metrics/poses/frame.py"]
    assert not plan.everything
    chosen = set(plan.files) - set(script.GUARD_TESTS)
    # test_frame imports the module, test_uses_support imports a helper that does
    assert chosen == {"tests/test_frame.py", "tests/test_uses_support.py"}
    assert len(commands) == 1
    assert commands[0][:7] == [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-m",
        "not gpu and not slow",
    ]
    assert commands[0][7:] == list(plan.files)


def test_a_change_in_a_base_module_reaches_the_modules_that_import_it(script, repo):
    write(repo, {"src/binding_metrics/core/system.py": "def prep(): return 1\n"})
    _, plan, _ = selection(script, repo)

    chosen = set(plan.files) - set(script.GUARD_TESTS)
    assert chosen == {"tests/test_prep.py", "tests/test_run.py"}  # test_run goes through cli.run
    assert plan.files["tests/test_run.py"] == ["imports cli.run -> core.system"]


def test_the_guard_tests_always_run(script, repo):
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})
    _, plan, _ = selection(script, repo)
    assert set(script.GUARD_TESTS) <= set(plan.files)
    assert all(plan.files[guard] == ["guard test"] for guard in script.GUARD_TESTS)


def test_a_module_loaded_by_name_selects_the_tests_that_import_the_loader(script, repo):
    write(repo, {"src/binding_metrics/metrics/geometry.py": "def ramachandran(): return 1\n"})
    _, plan, _ = selection(script, repo)
    chosen = set(plan.files) - set(script.GUARD_TESTS)
    assert chosen == {"tests/test_registry_use.py"}
    assert "loads metrics.geometry by name" in plan.files["tests/test_registry_use.py"][0]

    # a module that no table names is not reached through the loader
    commit(repo)
    write(repo, {"src/binding_metrics/metrics/sasa.py": "def sasa(): return 1\n"})
    _, plan, _ = selection(script, repo, "--working-tree")
    assert set(plan.files) - set(script.GUARD_TESTS) == {"tests/test_sasa.py"}


def test_a_package_init_selects_the_importers_of_its_names_only(script, repo):
    write(repo, {"src/binding_metrics/cli/__init__.py": "def add_flag(): return 1\n"})
    _, plan, _ = selection(script, repo)
    chosen = set(plan.files) - set(script.GUARD_TESTS)
    assert chosen == {"tests/test_flags.py"}  # test_run imports cli.run, not the package


def test_max_lazy_sets_how_many_imports_inside_functions_a_chain_may_contain(script, repo):
    write(repo, {"src/binding_metrics/chain/a.py": "def a(): return 1\n"})

    def chosen(*args):
        plan = selection(script, repo, *args)[1]
        return set(plan.files) - set(script.GUARD_TESTS)

    # b imports a inside a function, c imports b inside a function: two such links
    assert chosen() == {"tests/test_chain_b.py"}
    assert chosen("--max-lazy", "0") == set()
    assert chosen("--max-lazy", "2") == {"tests/test_chain_b.py", "tests/test_chain_c.py"}


def test_a_changed_test_file_runs_itself_and_a_changed_helper_runs_its_users(script, repo):
    write(repo, {"tests/test_run.py": "from binding_metrics.cli.run import main\nX = 1\n"})
    _, plan, _ = selection(script, repo, "--working-tree")
    assert set(plan.files) - set(script.GUARD_TESTS) == {"tests/test_run.py"}

    commit(repo)
    write(repo, {"tests/support.py": "from binding_metrics.poses.frame import frame\nY = 1\n"})
    _, plan, _ = selection(script, repo, "--working-tree")
    assert set(plan.files) - set(script.GUARD_TESTS) == {"tests/test_uses_support.py"}
    assert "tests/support.py" not in plan.files  # a helper has no tests of its own


def test_a_markdown_file_runs_the_tests_that_read_documentation(script, repo):
    write(repo, {"README.md": "Readme 2\n", "docs/guide.md": "Guide 2\n"})
    _, plan, _ = selection(script, repo)
    assert set(plan.files) - set(script.GUARD_TESTS) == {"tests/test_guide.py"}


def test_pyproject_or_conftest_runs_the_fast_tier(script, repo):
    write(repo, {"pyproject.toml": "[project]\nname = 'y'\n"})
    _, plan, commands = selection(script, repo)
    assert plan.everything
    assert len(commands) == 1
    assert commands[0][5:] == ["-m", "not gpu and not slow", "tests"]

    commit(repo)
    write(repo, {"tests/conftest.py": "X = 1\n"})
    assert selection(script, repo, "--working-tree")[1].everything


def test_include_slow_keeps_the_slow_tests_and_gpu_stays_out(script, repo):
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})
    _, _, commands = selection(script, repo, include_slow=True)
    assert commands[0][5:7] == ["-m", "not gpu"]


def test_another_file_runs_the_tests_that_name_it(script, repo):
    write(repo, {"Dockerfile": "FROM scratch\nRUN true\n"})
    _, plan, _ = selection(script, repo)
    assert "tests/test_docker.py" in plan.files

    commit(repo)
    write(repo, {"environment.yml": "name: y\n"})
    _, plan, _ = selection(script, repo, "--working-tree")
    assert "tests/test_docker.py" not in plan.files
    assert any("environment.yml: no test refers to it" in note for note in plan.notes)


def test_pytest_arguments_after_the_double_dash_reach_the_command(script, repo):
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})
    _, _, commands = selection(script, repo, "--", "-x", "-q")
    assert commands[0][-2:] == ["-x", "-q"]


def test_explain_says_why_each_test_file_runs(script, repo, capsys):
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})
    assert script.main(["--repo", str(repo), "--dry-run", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "tests/test_frame.py: imports poses.frame" in out
    assert "tests/test_registry.py: guard test" in out
    printed = [shlex.split(line) for line in out.splitlines() if line.startswith(sys.executable)]
    assert printed and "tests/test_frame.py" in printed[0]


def test_a_dry_run_prints_the_command_and_runs_nothing(script, repo, capsys, monkeypatch):
    calls = []
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return real_run(command, *args, **kwargs) if command[0] == "git" else None

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})

    assert script.main(["--repo", str(repo), "--dry-run", "--", "-q"]) == 0
    out = capsys.readouterr().out
    printed = [shlex.split(line) for line in out.splitlines() if line.startswith(sys.executable)]
    assert len(printed) == 1 and printed[0][-1] == "-q"
    assert "src/binding_metrics/poses/frame.py" in out
    assert not [c for c in calls if c[0] != "git"]


@pytest.mark.parametrize(("pytest_code", "expected"), [(5, 0), (1, 1), (0, 0)])
def test_the_exit_code_is_pytests_and_no_tests_selected_is_not_a_failure(
    script, repo, monkeypatch, pytest_code, expected
):
    commands = []
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[0] == "git":
            return real_run(command, *args, **kwargs)
        commands.append((command, kwargs.get("cwd")))
        return subprocess.CompletedProcess(command, pytest_code)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    write(repo, {"src/binding_metrics/poses/frame.py": "def frame(): return 1\n"})

    assert script.main(["--repo", str(repo)]) == expected
    assert len(commands) == 1
    assert Path(commands[0][1]).resolve() == repo.resolve()
