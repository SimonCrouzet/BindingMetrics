#!/usr/bin/env python3
"""Run the tests that a diff can affect, instead of the whole suite.

The full non-GPU suite takes about twenty minutes, and most changes reach a small part of it.
This script turns a list of changed files into the test files to run:

- a file of ``src/binding_metrics`` stands for its module. The tests chosen are those whose
  file, test helpers and fixtures import that module or a module that imports it, directly or
  not (the chain is printed by ``--explain``). A module that a table loads by name (a metric
  of the registry, a model adapter) also selects the tests that import the code doing the
  loading, since no import statement shows that they can run it. A package ``__init__`` selects
  only the importers of its names, not the importers of its children;
- a changed test file runs itself, and so do the test files that import it as a helper;
- a Markdown file (README, CHANGELOG, docs/) runs the tests that read documentation;
- ``pyproject.toml``, ``tests/conftest.py``, ``tests/_tiers.py`` and ``tests/_areas.py`` can
  change what every test does, so the whole of ``tests`` runs;
- any other file runs the test files whose text names it (``Dockerfile``, ``environment.yml``);
- the guard tests in ``GUARD_TESTS`` always run, because they catch breakage that crosses areas.

The scan of imports is the one that tags the tests (``tests/_areas.py``), imported from there so
that the script and ``tests/conftest.py`` cannot disagree; nothing from the package is imported.
Measured against the lines that each test executed (docs/testing.md gives the figures), the
selection finds about 99 % of the test files that run any line of a module. It cannot be narrow
for the modules that most of the code depends on (``io/structures.py``, ``metrics/_common.py``,
``utils.py``): a change there selects most of the suite, which is correct.

Usage::

    python scripts/pytest_for_diff.py                  # branch vs the merge-base with origin/main
    python scripts/pytest_for_diff.py --working-tree   # uncommitted work vs HEAD
    python scripts/pytest_for_diff.py --staged         # what is staged
    python scripts/pytest_for_diff.py --base HEAD~3    # work since a commit or branch
    python scripts/pytest_for_diff.py --explain        # say why each test file was chosen
    python scripts/pytest_for_diff.py --dry-run        # print the command and run nothing
    python scripts/pytest_for_diff.py -- -x -q         # arguments after -- go to pytest

The chosen tests run in the fast tier (``tests/_tiers.py``) and ``--full`` adds the ``slow``,
``integration`` and ``openfold`` tests; ``gpu`` tests never run here. The guard files are named
in a command without a tier, so every test of them runs and a guard test is never left out. The
full suite (``pytest --full``) is the gate before a pull request and a signed release.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_REPO = Path(__file__).resolve().parent.parent

# Cheap tests that run on every diff, whatever changed. Each one fails on a breakage that a
# change in one area can cause in another (about half a minute together, without GPU).
GUARD_REASON = "guard test"
GUARD_TESTS = (
    "tests/test_s1_encoding.py",  # every source file is UTF-8 and free of mojibake
    "tests/test_registry.py",  # the metric registry: names, paths, metadata of each metric
    "tests/test_w2a_lazy_exports.py",  # the lazy exports match _EXPORTS, __all__ and the stubs
    "tests/test_w2a_import_without_openmm.py",  # the static metrics import without OpenMM
    "tests/test_w3a_cli_help.py",  # the options and help of every command-line entry point
    "tests/test_area_tags.py",  # every test has an area, every module has a row in the table
)

# A change here can alter how every test is collected or run: every test file runs.
WHOLE_SUITE_TRIGGERS = ("pyproject.toml", "tests/conftest.py", "tests/_tiers.py", "tests/_areas.py")


def load_areas():
    """Import ``tests/_areas.py`` of this script's checkout (standard library only)."""
    tests_dir = str(SCRIPT_REPO / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import _areas

    return _areas


def git(repo: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if check and done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout if done.returncode == 0 else ""


def default_base(repo: Path) -> tuple[str, str]:
    """Commit to diff against, and how it was found.

    The merge-base of HEAD with ``origin/main`` (then ``main``, ``master``) is the point where the
    branch left the main line, so the diff is everything a pull request would contain plus the
    work in progress. Without such a branch the base is the previous commit, and in a repository
    with a single commit the empty tree, which makes every file count as changed.
    """
    for branch in ("origin/main", "main", "origin/master", "master"):
        if git(repo, "rev-parse", "--verify", "--quiet", branch, check=False).strip():
            base = git(repo, "merge-base", "HEAD", branch, check=False).strip()
            if base:
                return base, f"merge-base with {branch}"
    if git(repo, "rev-parse", "--verify", "--quiet", "HEAD~1", check=False).strip():
        return "HEAD~1", "previous commit"
    return git(repo, "hash-object", "-t", "tree", "/dev/null").strip(), "empty tree"


def split_names(output: str) -> list[str]:
    return [name for name in output.split("\0") if name]


def changed_files(repo: Path, mode: str, base: str | None) -> tuple[list[str], str]:
    """Paths relative to the repository root that differ, and a description of the comparison."""
    untracked = split_names(git(repo, "ls-files", "-z", "--others", "--exclude-standard"))
    if mode == "staged":
        names = split_names(git(repo, "diff", "-z", "--name-only", "--no-renames", "--cached"))
        return sorted(set(names)), "staged files against HEAD"
    if mode == "working-tree":
        reference, how = "HEAD", "uncommitted work against HEAD"
        if not git(repo, "rev-parse", "--verify", "--quiet", "HEAD", check=False).strip():
            reference = git(repo, "hash-object", "-t", "tree", "/dev/null").strip()
    elif base is not None:
        reference, how = base, f"work since {base}"
    else:
        reference, how = default_base(repo)
        how = f"work since the {how} ({reference[:10]})"
    names = split_names(git(repo, "diff", "-z", "--name-only", "--no-renames", reference))
    return sorted(set(names) | set(untracked)), how


@dataclass
class Plan:
    """What to run for a set of changed files."""

    everything: bool = False
    files: dict[str, list[str]] = field(default_factory=dict)  # test file -> why it runs
    notes: list[str] = field(default_factory=list)  # one line per changed file

    def add(self, test_file: str, reason: str) -> None:
        self.files.setdefault(test_file, []).append(reason)


def plan_for(
    project, changed: list[str], guards: tuple[str, ...] = GUARD_TESTS, max_lazy: int | None = None
) -> Plan:
    """Map changed files to test files, with the project's own import scan."""
    plan = Plan()
    triggers = [name for name in changed if name in WHOLE_SUITE_TRIGGERS]
    if triggers:  # the whole suite includes every other selection
        plan.everything = True
        plan.notes = [f"{name}: can change every test, so all test files run" for name in triggers]
        add_guards(plan, project, guards)
        return plan

    def relative(path: Path) -> str:
        return path.relative_to(project.root).as_posix()

    for name in changed:
        if name.startswith("tests/") and name.endswith(".py"):
            path = project.root / name
            if not path.exists():
                plan.notes.append(f"{name}: deleted")
                continue
            is_test = Path(name).name.startswith("test_")
            if is_test:
                plan.add(name, f"{name} changed")
            users = [relative(t) for t in project.tests_using(path)]
            for user in users:
                plan.add(user, f"imports the changed {name}")
            kind = "a test file" if is_test else "a test helper"
            plan.notes.append(f"{name}: {kind}; {len(users)} other test file(s) import it")
        elif name.startswith("src/binding_metrics/"):
            seeds = project.modules_of_source_file(name)
            limit = {} if max_lazy is None else {"max_lazy": max_lazy}
            affected = project.affected_modules(seeds, **limit)
            chosen = project.select_tests(seeds, **limit)
            for test, reason in chosen.items():
                plan.add(relative(test), reason)
            plan.notes.append(
                f"{name}: {', '.join(sorted(m or 'binding_metrics' for m in seeds)[:3])}"
                f"{' ...' if len(seeds) > 3 else ''}; {len(affected) - len(seeds)} module(s) "
                f"depend on it; {len(chosen)} test file(s)"
            )
        elif name.endswith(".md"):
            readers = [
                relative(t) for t in project.test_files() if "docs" in project.areas_of_test_file(t)
            ]
            for reader in readers:
                plan.add(reader, f"reads documentation ({name} changed)")
            plan.notes.append(f"{name}: documentation; {len(readers)} test file(s) read it")
        else:
            mentioning = [relative(t) for t in project.tests_mentioning(Path(name).name)]
            for test in mentioning:
                plan.add(test, f"names {Path(name).name}")
            if mentioning:
                plan.notes.append(f"{name}: named in {len(mentioning)} test file(s)")
            else:
                plan.notes.append(f"{name}: no test refers to it, nothing to run")

    add_guards(plan, project, guards)
    plan.files = dict(sorted(plan.files.items()))
    return plan


def add_guards(plan: Plan, project, guards: tuple[str, ...]) -> None:
    """Put the guard tests that exist in this checkout into the plan."""
    for guard in guards:
        if (project.root / guard).exists():
            plan.add(guard, GUARD_REASON)
        else:
            plan.notes.append(f"{guard}: guard test is missing from this checkout")


def build_commands(plan: Plan, full: bool, pytest_args: list[str]) -> list[list[str]]:
    """The pytest command lines for a plan.

    The chosen test files run in the fast tier, or with ``full`` in every tier but GPU. Pytest
    runs a file named on its command line whatever its tier, so ``--tier-named`` puts the tier back
    on them. The guard files are the exception, since a guard test has to run whatever the tier
    leaves out: they are named in a command without a tier, and are left out of the first so that
    none of their tests runs twice. The plan for the whole suite names the guard files in front of
    ``tests``: the files run whole and the directory obeys the tier, in one command because pytest
    collects a file once.
    """
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    tier = ["--full"] if full else []
    guards = [name for name, why in plan.files.items() if GUARD_REASON in why]
    if plan.everything:
        return [[*base, *tier, *guards, "tests", *pytest_args]]
    commands = []
    chosen = [name for name in plan.files if name not in guards]
    if chosen:
        commands.append([*base, *tier, "--tier-named", *chosen, *pytest_args])
    if guards:
        commands.append([*base, *guards, *pytest_args])
    return commands


def run(commands: list[list[str]], repo: Path) -> int:
    """Run the commands one after the other; the exit code is the first failure's."""
    exit_code = 0
    for command in commands:
        print(f"\n$ {shlex.join(command)}", flush=True)
        code = subprocess.run(command, cwd=repo, check=False).returncode
        if code == 5:  # pytest: nothing selected, which is not a failure of the diff
            code = 0
        exit_code = exit_code or code
    return exit_code


def parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    own, pytest_args = (
        (argv[: argv.index("--")], argv[argv.index("--") + 1 :]) if "--" in argv else (argv, [])
    )
    parser = argparse.ArgumentParser(
        description="Run the tests that the changed files can affect.",
        epilog="Arguments after -- are passed to pytest.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--base", metavar="REF", help="compare with this commit or branch")
    mode.add_argument("--staged", action="store_true", help="only the staged changes")
    mode.add_argument(
        "--working-tree",
        action="store_true",
        help="uncommitted work (staged, unstaged, untracked) against HEAD",
    )
    parser.add_argument(
        "--max-lazy",
        type=int,
        metavar="N",
        help="follow chains with at most N imports inside functions (default 1; 0 is narrower "
        "and misses tests that reach the module through a function, 2 or more is wider)",
    )
    parser.add_argument(
        "--full",
        "--include-slow",
        dest="full",
        action="store_true",
        help="run the tests that the fast tier leaves out too (slow, integration, openfold); "
        "gpu tests never run. --include-slow is the earlier name",
    )
    parser.add_argument("--explain", action="store_true", help="say why each test file was chosen")
    parser.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    parser.add_argument("--repo", type=Path, default=SCRIPT_REPO, help=argparse.SUPPRESS)
    return parser.parse_args(own), pytest_args


def main(argv: list[str] | None = None) -> int:
    args, pytest_args = parse_arguments(sys.argv[1:] if argv is None else argv)
    repo = Path(git(args.repo, "rev-parse", "--show-toplevel").strip())
    mode = "staged" if args.staged else "working-tree" if args.working_tree else "base"
    changed, how = changed_files(repo, mode, args.base)

    project = load_areas().Project(repo)
    plan = plan_for(project, changed, max_lazy=args.max_lazy)
    commands = build_commands(plan, args.full, pytest_args)

    print(f"{len(changed)} changed file(s): {how}")
    for note in plan.notes:
        print(f"  {note}")
    tier = "the full tier" if args.full else "the fast tier"
    print(
        f"selection: all test files, {tier}"
        if plan.everything
        else f"selection: {len(plan.files)} test file(s), {tier}"
    )
    if args.explain:
        for test_file, reasons in plan.files.items():
            print(
                f"  {test_file}: {reasons[0]}"
                + (f" (+{len(reasons) - 1} more)" if reasons[1:] else "")
            )
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    return run(commands, repo)


if __name__ == "__main__":
    sys.exit(main())
