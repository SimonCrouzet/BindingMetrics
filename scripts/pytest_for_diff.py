#!/usr/bin/env python3
"""Run the tests that a diff can affect, instead of the whole suite.

The full non-GPU suite takes about twenty minutes, and most changes reach a small part of it.
Every test carries area markers (``pytest -m poses``; see ``tests/_areas.py``), and this script
turns a list of changed files into the areas to run:

- a file of ``src/binding_metrics`` stands for its module and for every module that imports it,
  directly or not; the areas of these modules are run;
- a changed test file runs itself, and so do the test files that import it as a helper;
- a Markdown file (README, CHANGELOG, docs/) runs the ``docs`` area;
- ``pyproject.toml``, ``tests/conftest.py`` and ``tests/_areas.py`` can change what every test
  does, so the fast tier (everything except ``gpu`` and ``slow``) runs;
- any other file runs the test files whose text names it (``Dockerfile``, ``environment.yml``);
- the guard tests in ``GUARD_TESTS`` always run, because they catch breakage that crosses areas.

The table of areas and the scan of imports are those of the markers, imported from
``tests/_areas.py``, so the script and ``tests/conftest.py`` cannot disagree. Nothing from the
package is imported. The selection follows ``import`` statements and ``import_module("...")``
calls; a module that another one only names in data (the ``import_path`` strings of the metric
registry) is not a dependency, so a change in how a metric is called deserves ``-m cli`` as well.

Usage::

    python scripts/pytest_for_diff.py                  # branch vs the merge-base with origin/main
    python scripts/pytest_for_diff.py --working-tree   # uncommitted work vs HEAD
    python scripts/pytest_for_diff.py --staged         # what is staged
    python scripts/pytest_for_diff.py --base HEAD~3    # work since a commit or branch
    python scripts/pytest_for_diff.py --dry-run        # print the commands and run nothing
    python scripts/pytest_for_diff.py -- -x -q         # arguments after -- go to pytest

``gpu`` tests never run here and ``slow`` ones only with ``--include-slow``. The full suite
remains the gate before a pull request and a signed release.
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
GUARD_TESTS = (
    "tests/test_s1_encoding.py",  # every source file is UTF-8 and free of mojibake
    "tests/test_registry.py",  # the metric registry: names, paths, metadata of each metric
    "tests/test_w2a_lazy_exports.py",  # the lazy exports match _EXPORTS, __all__ and the stubs
    "tests/test_w2a_import_without_openmm.py",  # the static metrics import without OpenMM
    "tests/test_w3a_cli_help.py",  # the options and help of every command-line entry point
    "tests/test_area_tags.py",  # every test has an area, every module has a row in the table
)

# A change here can alter how every test is collected or run: run the fast tier.
FAST_TIER_TRIGGERS = ("pyproject.toml", "tests/conftest.py", "tests/_areas.py")


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

    areas: set[str] = field(default_factory=set)
    everything: bool = False
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def plan_for(project, changed: list[str], guards: tuple[str, ...] = GUARD_TESTS) -> Plan:
    """Map changed files to areas and test files, with the project's own table and scan."""
    plan = Plan()
    triggers = [name for name in changed if name in FAST_TIER_TRIGGERS]
    if triggers:  # the fast tier includes every other selection
        plan.everything = True
        plan.notes = [f"{name}: can change every test, so the fast tier runs" for name in triggers]
        return plan

    files: set[str] = set()

    def relative(path: Path) -> str:
        return path.relative_to(project.root).as_posix()

    for name in changed:
        if name.startswith("tests/") and name.endswith(".py"):
            path = project.root / name
            if not path.exists():
                plan.notes.append(f"{name}: deleted")
                continue
            if Path(name).name.startswith("test_"):
                files.add(name)
            users = [relative(t) for t in project.tests_using(path)]
            files.update(users)
            kind = "a test file" if Path(name).name.startswith("test_") else "a test helper"
            plan.notes.append(f"{name}: {kind}; {len(users)} other test file(s) import it")
        elif name.startswith("src/binding_metrics/"):
            seeds = project.modules_of_source_file(name)
            affected = project.dependents_closure(seeds)
            areas = project.areas_of_modules(affected)
            plan.areas |= areas
            plan.notes.append(
                f"{name}: {len(seeds)} module(s) and {len(affected) - len(seeds)} that import "
                f"them -> {', '.join(sorted(areas))}"
            )
        elif name.endswith(".md"):
            plan.areas.add("docs")
            plan.notes.append(f"{name}: documentation -> docs")
        else:
            mentioning = [relative(t) for t in project.tests_mentioning(Path(name).name)]
            if mentioning:
                files.update(mentioning)
                plan.notes.append(f"{name}: named in {len(mentioning)} test file(s)")
            else:
                plan.notes.append(f"{name}: no test refers to it, nothing to run")

    for guard in guards:
        if (project.root / guard).exists():
            files.add(guard)
        else:
            plan.notes.append(f"{guard}: guard test is missing from this checkout")

    # A file whose tests already carry a selected area runs with that area.
    plan.files = sorted(
        f for f in files if not project.areas_of_test_file(project.root / f) & plan.areas
    )
    return plan


def build_commands(plan: Plan, include_slow: bool, pytest_args: list[str]) -> list[list[str]]:
    """The pytest command lines: one for the areas (or the fast tier), one for loose files.

    A marker expression filters file arguments too, so the files that are not covered by a
    selected area cannot share a command with it.
    """
    tier = "not gpu" if include_slow else "not gpu and not slow"
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    commands = []
    if plan.everything:
        commands.append([*base, "-m", tier, "tests", *pytest_args])
    elif plan.areas:
        expression = f"{tier} and ({' or '.join(sorted(plan.areas))})"
        commands.append([*base, "-m", expression, "tests", *pytest_args])
    if plan.files:
        commands.append([*base, "-m", tier, *plan.files, *pytest_args])
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
    parser.add_argument("--include-slow", action="store_true", help="also run tests marked slow")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--repo", type=Path, default=SCRIPT_REPO, help=argparse.SUPPRESS)
    return parser.parse_args(own), pytest_args


def main(argv: list[str] | None = None) -> int:
    args, pytest_args = parse_arguments(sys.argv[1:] if argv is None else argv)
    repo = Path(git(args.repo, "rev-parse", "--show-toplevel").strip())
    mode = "staged" if args.staged else "working-tree" if args.working_tree else "base"
    changed, how = changed_files(repo, mode, args.base)

    project = load_areas().Project(repo)
    plan = plan_for(project, changed)
    commands = build_commands(plan, args.include_slow, pytest_args)

    print(f"{len(changed)} changed file(s): {how}")
    for note in plan.notes:
        print(f"  {note}")
    if plan.everything:
        print("selection: the fast tier")
    else:
        print(f"selection: areas {', '.join(sorted(plan.areas)) or '(none)'}")
    if plan.files:
        print(f"           and {len(plan.files)} test file(s) outside them")
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    return run(commands, repo)


if __name__ == "__main__":
    sys.exit(main())
