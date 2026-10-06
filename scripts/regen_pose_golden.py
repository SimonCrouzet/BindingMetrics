#!/usr/bin/env python3
"""Rewrite the golden pose-set results in tests/golden/pose_sets/.

Usage:
    python scripts/regen_pose_golden.py --write                      # every case
    python scripts/regen_pose_golden.py --write --case synthetic_1ycr
    python scripts/regen_pose_golden.py --write --golden-dir DIR     # another directory

The golden files are reference data: the tests compare the live ``score_pose_set`` result with
them, so rewriting a file changes what the tests accept. The script therefore refuses to run
without ``--write``. For each case it rebuilds the result, compares it with the file that is
there (with the tolerance of ``tests/golden/pose_set_cases.py``), prints what changed, and
rewrites the file only when a value moved beyond the tolerance or the case description changed,
so that regenerating on another machine does not rewrite rounding noise.

A regeneration is legitimate when the new numbers are intended and understood:

  * a deliberate change of the pose-set code (frame, agreement, aggregation, descriptors,
    result layout) or of a registry metric that the cases call;
  * a deliberate change of the synthetic generator (``binding_metrics.testing.synthetic``);
  * a dependency whose algorithm changed (biotite surface areas, for instance), after the new
    values were checked against an independent calculation.

It is not legitimate to regenerate to make a failing test pass without knowing why the value
moved. The commit that carries the new files states the reason in a regression note: one named
example and the value before and after (the script prints a line to start from).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.golden import pose_set_cases as golden  # noqa: E402

_LEGITIMATE = """\
A regeneration is legitimate when the new numbers are intended and understood: a deliberate
change of the pose-set code or of a registry metric the cases call, of the synthetic
generator, or a dependency whose algorithm changed (after an independent check of the new
values). It is not legitimate to regenerate to make a failing test pass without knowing why a
value moved. Put the reason in the commit as a regression note: one named example, before ->
after."""


def _block(path: str) -> str:
    """The part of a leaf path that names its block: list indexes folded, two components."""
    folded = re.sub(r"\[\d+\]", "[*]", path)
    return ".".join(folded.split(".")[:2])


def _relative_change(mismatch: golden.Mismatch) -> float:
    if not (isinstance(mismatch.golden, float) and isinstance(mismatch.live, float)):
        return math.inf
    scale = abs(mismatch.golden) or 1.0
    return abs(mismatch.live - mismatch.golden) / scale


def summarise(case_name: str, mismatches: list[golden.Mismatch], out) -> None:
    """Print what changed in one case: counts, blocks, the largest moves, a note to start from."""
    kinds = Counter(mismatch.kind for mismatch in mismatches)
    changed = kinds["changed"] + kinds["type"]
    print(
        f"{case_name}: {changed} value(s) changed beyond the tolerance, "
        f"{kinds['extra']} added, {kinds['missing']} removed",
        file=out,
    )
    blocks = Counter(_block(mismatch.path) for mismatch in mismatches)
    for block, count in blocks.most_common(8):
        print(f"  {count:5d}  {block}", file=out)
    largest = sorted(mismatches, key=_relative_change, reverse=True)[:5]
    print("  largest changes:", file=out)
    for mismatch in largest:
        print(f"    {mismatch.describe()}", file=out)
    first = next((m for m in largest if m.kind in ("changed", "type")), None)
    if first is not None:
        print(
            f"  regression note to start from: {case_name}, {first.path}: "
            f"{first.golden!r} -> {first.live!r}",
            file=out,
        )


def regenerate(case: golden.GoldenCase, golden_dir: Path, out) -> bool:
    """Rebuild one case and write its file when it changed; True when a file was written."""
    path = Path(golden_dir) / f"{case.name}.json"
    document = golden.golden_document(case, case.run())
    if not path.is_file():
        golden.write_golden(path, document)
        print(f"{case.name}: new file {path}", file=out)
        return True
    old = golden.read_golden(path)
    mismatches = golden.compare(old["result"], document["result"])
    old_meta = {key: value for key, value in old.items() if key != "result"}
    new_meta = {key: value for key, value in document.items() if key != "result"}
    if not mismatches and old_meta == new_meta:
        print(f"{case.name}: unchanged within the tolerance, file left as it is", file=out)
        return False
    if mismatches:
        summarise(case.name, mismatches, out)
    else:
        print(f"{case.name}: values unchanged, description or tolerance record changed", file=out)
    golden.write_golden(path, document)
    print(f"{case.name}: rewrote {path}", file=out)
    return True


def main(argv: list[str] | None = None, out=None) -> int:
    out = sys.stdout if out is None else out
    parser = argparse.ArgumentParser(
        description="Rewrite the golden pose-set results (refuses to run without --write).",
        epilog=_LEGITIMATE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--write", action="store_true", help="really rewrite the golden files (required)"
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(golden.CASES),
        help="case to rewrite; repeat for several (default: all)",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=golden.GOLDEN_DIR,
        help=f"directory of the golden files (default: {golden.GOLDEN_DIR})",
    )
    args = parser.parse_args(argv)
    if not args.write:
        print(
            "refusing to run without --write: the golden files are reference data and "
            "rewriting one changes what the tests accept.\n" + _LEGITIMATE,
            file=sys.stderr,
        )
        return 2
    selected = args.case or sorted(golden.CASES)
    written = [regenerate(golden.CASES[name], args.golden_dir, out) for name in selected]
    if any(written):
        print(
            "\nRead `git diff tests/golden` before committing.\n" + _LEGITIMATE,
            file=out,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
