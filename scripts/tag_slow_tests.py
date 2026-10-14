#!/usr/bin/env python3
"""Mark the tests named in a list of node ids with ``@pytest.mark.slow``.

The fast tier of the suite (a plain ``pytest``) leaves out the tests that take a second or
more, so a test that grows past that line has to be tagged. This script does the tagging from a
list of pytest node ids instead of by hand, one id per line (the id of a case can hold blanks, so
a line ends at a tab: a column of seconds after a tab is ignored, and so are blank lines and
``#`` comments)::

    python scripts/tag_slow_tests.py slow_tests.txt --dry-run
    python scripts/tag_slow_tests.py slow_tests.txt --collected collected.txt

``--collected`` is the output of ``pytest --collect-only -q --full --gpu``, which lists every
case. It is only read for a parametrised test of which some cases, not all, are in the list: the
position of a case among the collected ones of its function is the position of its entry in the
literal list of ``@pytest.mark.parametrize``, and that entry gets ``marks=pytest.mark.slow``.

What it does to a file, by editing the text and leaving the rest of it as it was (line
endings, comments, formatting):

- a test function or method gets ``@pytest.mark.slow`` on a line of its own above its first
  decorator, with the indentation of the ``def``;
- a class gets the decorator instead of its methods when at least ``--class-share`` of its
  test methods are in the list (and at least two), none of its methods carries a speed marker
  and none is a partial case of a parametrisation: the few fast methods of such a class leave
  the fast tier with the others;
- a parametrised test of which only some cases are slow gets ``marks=pytest.mark.slow`` on
  those entries of its list (a plain value is wrapped in ``pytest.param``); when the
  parametrisation is not one literal list that the script can follow (a name, a comprehension,
  stacked decorators, a fixture with params, a call as an entry), the whole function is tagged
  and the report says so;
- ``import pytest`` is added when the file has no module-level one. Its place in the import
  block is not sorted: run ``ruff check --select I --fix`` and ``ruff format`` on the files.

A test that already carries a speed marker (``slow``, ``integration``, ``openfold``, ``gpu``,
``requires_cuda``, on the function, its class or the module ``pytestmark``) is never touched, so
a second run changes nothing. Nothing is written with ``--dry-run``. The exit status is 2 when
an id could not be resolved or a file could not be edited, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SPEED_MARKERS = frozenset({"slow", "integration", "openfold", "gpu"})
CLASS_SHARE_DEFAULT = 0.8
MIN_CLASS_TARGETS = 2
_LINE_END = re.compile(rb"\r\n|\n|\r")

FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Target:
    """One node id, split into the file, the names that lead to the function and the case."""

    path: str
    names: tuple[str, ...]
    case: str | None


@dataclass
class FileReport:
    """What happened to one file: the lines of the report, the counts and the new text."""

    path: str
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)
    whole_function_fallbacks: list[tuple[str, str]] = field(default_factory=list)
    already_marked: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    import_added: bool = False
    error: str | None = None
    new_bytes: bytes | None = None

    @property
    def changed(self) -> bool:
        return self.new_bytes is not None


def parse_node_id(line: str) -> Target | None:
    """Split ``path::Class::test[case]``; None for a blank line or a comment.

    The id is the whole line up to a tab, because the id of a case can hold blanks.
    """
    token = line.split("\t")[0].strip()
    if not token or token.startswith("#"):
        return None
    bracket = token.find("[")
    case = None
    head = token
    if bracket != -1:
        head = token[:bracket]
        case = token[bracket + 1 :]
        if case.endswith("]"):
            case = case[:-1]
    path, *names = head.split("::")
    if not names:
        raise ValueError(f"not a test node id (no '::'): {token}")
    return Target(path=path, names=tuple(names), case=case)


def dotted(node: ast.AST) -> str:
    """``pytest.mark.slow`` for an attribute chain, ``slow`` for a name, '' for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def speed_marker_named(node: ast.AST) -> str | None:
    """The speed marker that an attribute chain or a name stands for, if it is one."""
    parts = dotted(node).split(".")
    if len(parts) >= 3 and parts[-2] == "mark" and parts[-1] in SPEED_MARKERS:
        return parts[-1]
    if parts[-1] == "requires_cuda":
        return "gpu"
    return None


def speed_markers_in(expression: ast.AST) -> set[str]:
    """The speed markers named anywhere inside an expression (a ``marks=`` or ``pytestmark``)."""
    found = {speed_marker_named(node) for node in ast.walk(expression)}
    found.discard(None)
    return found


def speed_markers_of(node: ast.AST) -> set[str]:
    """The speed markers among the decorators of a function or class (not those nested in the
    arguments of a decorator, such as the ``marks=`` of a ``pytest.param``)."""
    found = set()
    for decorator in getattr(node, "decorator_list", []):
        head = decorator.func if isinstance(decorator, ast.Call) else decorator
        found.add(speed_marker_named(head))
    found.discard(None)
    return found


def module_speed_markers(tree: ast.Module) -> set[str]:
    """Markers of a module-level ``pytestmark = ...`` assignment."""
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            found |= speed_markers_in(node.value)
    return found


def definitions(body: list[ast.stmt], prefix: tuple[str, ...] = ()):
    """Yield ``(qualified name, node)`` for the classes and functions of a body.

    Looks through ``if``, ``try`` and ``with`` blocks, into classes, and never into functions.
    """
    for node in body:
        if isinstance(node, (ast.ClassDef, *FUNCTION_TYPES)):
            name = (*prefix, node.name)
            yield name, node
            if isinstance(node, ast.ClassDef):
                yield from definitions(node.body, name)
        elif isinstance(node, ast.If):
            yield from definitions(node.body, prefix)
            yield from definitions(node.orelse, prefix)
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
                yield from definitions(block, prefix)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            yield from definitions(node.body, prefix)


class SourceText:
    """The bytes of a file with the offsets of its lines, so that AST columns (UTF-8 bytes) map
    to positions in the text and the file is edited without being decoded and re-encoded."""

    def __init__(self, data: bytes):
        self.data = data
        self.line_starts = [0] + [m.end() for m in _LINE_END.finditer(data)]
        self.eol = b"\r\n" if b"\r\n" in data else b"\n"

    def offset(self, line: int, column: int) -> int:
        return self.line_starts[line - 1] + column

    def start(self, node: ast.AST) -> int:
        return self.offset(node.lineno, node.col_offset)

    def end(self, node: ast.AST) -> int:
        return self.offset(node.end_lineno, node.end_col_offset)

    def line_text(self, line: int) -> bytes:
        stop = self.line_starts[line] if line < len(self.line_starts) else len(self.data)
        return self.data[self.line_starts[line - 1] : stop]

    def indent_of(self, line: int) -> bytes:
        text = self.line_text(line)
        return text[: len(text) - len(text.lstrip(b" \t"))]


def first_line_of(source: SourceText, node: ast.AST) -> int:
    """The line of the first decorator's ``@``, or of the ``def``/``class`` when it has none."""
    decorators = getattr(node, "decorator_list", [])
    if not decorators:
        return node.lineno
    line = decorators[0].lineno
    while line > 1 and not source.line_text(line).lstrip().startswith(b"@"):
        line -= 1
    return line


@dataclass
class Edit:
    offset: int
    remove: int
    text: bytes


def decorator_edit(source: SourceText, node: ast.AST, marker: str) -> Edit:
    line = first_line_of(source, node)
    text = source.indent_of(line) + f"@pytest.mark.{marker}".encode() + source.eol
    return Edit(source.line_starts[line - 1], 0, text)


def has_pytest_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            (alias.asname or alias.name) == "pytest" for alias in node.names
        ):
            return True
    return False


def import_edit(source: SourceText, tree: ast.Module) -> Edit:
    """An ``import pytest`` line after the first run of imports (or the docstring)."""
    run_end = None
    seen = False
    for index, node in enumerate(tree.body):
        is_import = isinstance(node, (ast.Import, ast.ImportFrom))
        if is_import:
            seen = True
            run_end = node.end_lineno
        elif seen:
            break
        elif index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            run_end = node.end_lineno  # the module docstring
    if run_end is None:
        return Edit(0, 0, b"import pytest" + source.eol)
    prefix = source.eol if not seen else b""
    offset = source.line_starts[run_end] if run_end < len(source.line_starts) else len(source.data)
    text = prefix + b"import pytest" + source.eol
    if offset == len(source.data) and not source.data.endswith((b"\n", b"\r")):
        text = source.eol + text
    return Edit(offset, 0, text)


# ---------------------------------------------------------------------------------------------
# Cases of one parametrisation
# ---------------------------------------------------------------------------------------------


class NotEditableError(Exception):
    """The parametrisation of a function is not one literal list that can be followed."""


def parametrize_calls(function: ast.AST) -> list[ast.Call]:
    return [
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call) and dotted(decorator.func).endswith("mark.parametrize")
    ]


def argument_count(call: ast.Call) -> int:
    names = call.args[0] if call.args else None
    if isinstance(names, ast.Constant) and isinstance(names.value, str):
        return len([n for n in names.value.split(",") if n.strip()])
    if isinstance(names, (ast.List, ast.Tuple)):
        return len(names.elts)
    raise NotEditableError("the argument names of the parametrisation are not literal")


def argument_values(call: ast.Call) -> list[ast.expr]:
    if len(call.args) >= 2:
        values = call.args[1]
    else:
        keyword = next((k for k in call.keywords if k.arg == "argvalues"), None)
        if keyword is None:
            raise NotEditableError("the argvalues of the parametrisation are not given")
        values = keyword.value
    if not isinstance(values, (ast.List, ast.Tuple)):
        raise NotEditableError("the values of the parametrisation are not a literal list")
    if any(isinstance(entry, ast.Starred) for entry in values.elts):
        raise NotEditableError("the list of values unpacks another list")
    return list(values.elts)


def is_pytest_param(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and dotted(node.func) in {"pytest.param", "param"}


def last_argument(call: ast.Call) -> ast.AST | None:
    arguments = [*call.args, *[k.value for k in call.keywords]]
    if not arguments:
        return None
    return max(arguments, key=lambda a: (a.end_lineno, a.end_col_offset))


def case_edit(source: SourceText, entry: ast.expr, name_count: int, marker: str) -> Edit | None:
    """The edit that gives one entry of a ``parametrize`` list the mark; None if it has one."""
    mark = f"pytest.mark.{marker}".encode()
    if is_pytest_param(entry):
        existing = next((k for k in entry.keywords if k.arg == "marks"), None)
        if existing is not None:
            if speed_markers_in(existing.value):
                return None
            value = existing.value
            if isinstance(value, (ast.List, ast.Tuple)) and value.elts:
                last = value.elts[-1]
                return Edit(source.end(last), 0, b", " + mark)
            span = source.data[source.start(value) : source.end(value)]
            return Edit(
                source.start(value),
                source.end(value) - source.start(value),
                b"[" + span + b", " + mark + b"]",
            )
        last = last_argument(entry)
        if last is None:
            return Edit(source.end(entry) - 1, 0, b"marks=" + mark)
        return Edit(source.end(last), 0, b", marks=" + mark)
    if isinstance(entry, ast.Call):
        raise NotEditableError("an entry of the list is a call that may return a pytest.param")
    start, end = source.start(entry), source.end(entry)
    text = source.data[start:end]
    if name_count > 1:
        if isinstance(entry, (ast.Tuple, ast.List)) and text[:1] in (b"(", b"["):
            inner = text[1:-1].strip().rstrip(b",")
            return Edit(start, end - start, b"pytest.param(" + inner + b", marks=" + mark + b")")
        return Edit(start, end - start, b"pytest.param(*" + text + b", marks=" + mark + b")")
    return Edit(start, end - start, b"pytest.param(" + text + b", marks=" + mark + b")")


def case_edits(
    source: SourceText,
    function: ast.AST,
    slow_cases: list[str],
    collected_cases: list[str] | None,
    marker: str,
) -> list[tuple[str, Edit | None]]:
    """The edit for each slow case on the one literal ``parametrize`` list of a function.

    ``collected_cases`` are the ids of every collected case of the function, in order. The edit
    of a case whose entry already carries a speed marker is None. Raises NotEditableError when
    the entries cannot be told apart from the list.
    """
    calls = parametrize_calls(function)
    if len(calls) != 1:
        raise NotEditableError(f"the function has {len(calls)} parametrize decorators")
    if collected_cases is None:
        raise NotEditableError("no list of the collected cases was given (--collected)")
    entries = argument_values(calls[0])
    if len(entries) != len(collected_cases):
        raise NotEditableError(
            f"the list has {len(entries)} entries and {len(collected_cases)} cases are collected"
        )
    name_count = argument_count(calls[0])
    plan = []
    for case in slow_cases:
        if case not in collected_cases:
            raise NotEditableError(f"the case {case!r} is not among the collected ones")
        entry = entries[collected_cases.index(case)]
        plan.append((case, case_edit(source, entry, name_count, marker)))
    return plan


# ---------------------------------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------------------------------


def apply_edits(data: bytes, edits: list[Edit]) -> bytes:
    for edit in sorted(edits, key=lambda e: e.offset, reverse=True):
        data = data[: edit.offset] + edit.text + data[edit.offset + edit.remove :]
    return data


def is_test_method(node: ast.AST) -> bool:
    return isinstance(node, FUNCTION_TYPES) and node.name.startswith("test")


def tag_file(
    path: Path,
    label: str,
    targets: list[Target],
    collected: dict[tuple[str, tuple[str, ...]], list[str]],
    marker: str = "slow",
    class_share: float = CLASS_SHARE_DEFAULT,
) -> FileReport:
    """Work out the edits for the targets of one file; the new bytes are in the report."""
    report = FileReport(path=label)
    data = path.read_bytes()
    try:
        tree = ast.parse(data)
    except SyntaxError as error:
        report.error = f"cannot parse: {error}"
        return report
    source = SourceText(data)
    table = dict(definitions(tree.body))
    module_marked = bool(module_speed_markers(tree))

    # name -> list of cases named in the list, None for "the whole function"
    wanted: dict[tuple[str, ...], list[str] | None] = {}
    for target in targets:
        node = table.get(target.names)
        if node is None or not isinstance(node, FUNCTION_TYPES):
            report.unresolved.append(
                "::".join(target.names) + (f"[{target.case}]" if target.case else "")
            )
            continue
        if target.case is None:
            wanted[target.names] = None
        elif target.names not in wanted:
            wanted[target.names] = [target.case]
        elif wanted[target.names] is not None:
            wanted[target.names].append(target.case)

    def inherited_marked(names: tuple[str, ...]) -> bool:
        if module_marked:
            return True
        return any(
            speed_markers_of(table[names[:k]]) for k in range(1, len(names)) if names[:k] in table
        )

    edits: list[Edit] = []
    decided: dict[tuple[str, ...], str] = {}  # function name -> "function" or "cases"
    for names, cases in wanted.items():
        function = table[names]
        qualified = "::".join(names)
        if inherited_marked(names) or speed_markers_of(function):
            report.already_marked.append(qualified)
            continue
        if cases is None:
            decided[names] = "function"
            continue
        every_case = collected.get((label, names))
        if every_case is not None and sorted(set(every_case)) == sorted(set(cases)):
            decided[names] = "function"
            continue
        try:
            plan = case_edits(source, function, cases, every_case, marker)
        except NotEditableError as reason:
            report.whole_function_fallbacks.append((qualified, str(reason)))
            decided[names] = "function"
            continue
        decided[names] = "cases"
        for case, edit in plan:
            if edit is None:
                report.already_marked.append(f"{qualified}[{case}]")
            else:
                report.cases.append(f"{qualified}[{case}]")
                edits.append(edit)

    # A class takes the decorator when most of its test methods are tagged as a whole.
    by_class: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for names, how in decided.items():
        if how == "function" and len(names) > 1:
            by_class[names[:-1]].append(names)
    partial_in = {names[:-1] for names, how in decided.items() if how == "cases"}
    class_marked: set[tuple[str, ...]] = set()
    for class_names, methods in by_class.items():
        node = table.get(class_names)
        if not isinstance(node, ast.ClassDef) or class_names in partial_in:
            continue
        # the class itself and the classes around it
        if inherited_marked((*class_names, "x")):
            continue
        test_methods = [m for m in node.body if is_test_method(m)]
        if len(methods) < MIN_CLASS_TARGETS or not test_methods:
            continue
        if any(speed_markers_of(m) for m in test_methods):
            continue
        if len(methods) / len(test_methods) < class_share:
            continue
        class_marked.add(class_names)
        edits.append(decorator_edit(source, node, marker))
        report.classes.append("::".join(class_names))
    for names, how in decided.items():
        if how == "function" and names[:-1] not in class_marked:
            edits.append(decorator_edit(source, table[names], marker))
            report.functions.append("::".join(names))

    if not edits:
        return report
    if not has_pytest_import(tree):
        edits.append(import_edit(source, tree))
        report.import_added = True
    new = apply_edits(data, edits)
    try:
        ast.parse(new)
    except SyntaxError as error:
        report.error = f"the edited file does not parse, left unchanged: {error}"
        report.import_added = False
        return report
    report.new_bytes = new
    return report


def read_ids(stream) -> list[Target]:
    targets = []
    for line in stream:
        target = parse_node_id(line)
        if target is not None:
            targets.append(target)
    return targets


def read_collected(text: str) -> dict[tuple[str, tuple[str, ...]], list[str]]:
    """Case ids of each parametrised function, in collection order, from ``--collect-only -q``."""
    cases: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for line in text.splitlines():
        if "::" not in line or "[" not in line:
            continue
        target = parse_node_id(line)
        if target is not None and target.case is not None:
            cases[(target.path, target.names)].append(target.case)
    return dict(cases)


def tag_tests(
    targets: list[Target],
    root: Path,
    collected: dict[tuple[str, tuple[str, ...]], list[str]],
    marker: str = "slow",
    class_share: float = CLASS_SHARE_DEFAULT,
    write: bool = True,
) -> list[FileReport]:
    """Tag the targets under ``root``; with ``write`` False nothing is written."""
    by_file: dict[str, list[Target]] = defaultdict(list)
    for target in targets:
        by_file[target.path].append(target)
    reports = []
    for label in sorted(by_file):
        path = root / label
        if not path.is_file():
            report = FileReport(path=label, error="no such file")
            reports.append(report)
            continue
        report = tag_file(path, label, by_file[label], collected, marker, class_share)
        if write and report.changed:
            path.write_bytes(report.new_bytes)
        reports.append(report)
    return reports


def format_report(reports: list[FileReport], dry_run: bool) -> str:
    verb = "would tag" if dry_run else "tagged"
    lines = []
    totals = defaultdict(int)
    for report in reports:
        if report.error:
            lines.append(f"ERROR {report.path}: {report.error}")
            totals["errors"] += 1
        for name in report.unresolved:
            lines.append(f"UNRESOLVED {report.path}::{name}")
            totals["unresolved"] += 1
        for name, reason in report.whole_function_fallbacks:
            lines.append(f"WHOLE FUNCTION {report.path}::{name}: {reason}")
            totals["whole_function_fallbacks"] += 1
        if report.changed:
            parts = []
            if report.functions:
                parts.append(f"{len(report.functions)} functions")
            if report.classes:
                parts.append(f"{len(report.classes)} classes ({', '.join(report.classes)})")
            if report.cases:
                parts.append(f"{len(report.cases)} parametrised cases")
            if report.import_added:
                parts.append("import pytest added")
            lines.append(f"{verb} {report.path}: " + ", ".join(parts))
            totals["files"] += 1
            totals["functions"] += len(report.functions)
            totals["classes"] += len(report.classes)
            totals["cases"] += len(report.cases)
        totals["already"] += len(report.already_marked)
    lines.append(
        f"{verb} {totals['functions']} functions, {totals['classes']} classes and "
        f"{totals['cases']} parametrised cases in {totals['files']} files; "
        f"{totals['already']} already marked, {totals['unresolved']} unresolved, "
        f"{totals['errors']} errors, {totals['whole_function_fallbacks']} whole-function fallbacks"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add @pytest.mark.slow to the tests named by a list of pytest node ids."
    )
    parser.add_argument("ids", help="file with one node id per line, or - for the standard input")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="directory of the node ids")
    parser.add_argument("--marker", default="slow", help="marker to add (default: slow)")
    parser.add_argument(
        "--collected",
        type=Path,
        help="output of `pytest --collect-only -q --full --gpu` (every case), needed to mark "
        "single cases of a test",
    )
    parser.add_argument(
        "--class-share",
        type=float,
        default=CLASS_SHARE_DEFAULT,
        help="share of the test methods of a class in the list that makes the class itself marked",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the changes and write nothing"
    )
    args = parser.parse_args(argv)

    if args.ids == "-":
        targets = read_ids(sys.stdin)
    else:
        with open(args.ids, encoding="utf-8") as stream:
            targets = read_ids(stream)
    collected = {}
    if args.collected is not None:
        collected = read_collected(args.collected.read_text(encoding="utf-8"))
    reports = tag_tests(
        targets,
        args.root,
        collected,
        marker=args.marker,
        class_share=args.class_share,
        write=not args.dry_run,
    )
    print(format_report(reports, args.dry_run))
    failed = any(r.error or r.unresolved for r in reports)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
