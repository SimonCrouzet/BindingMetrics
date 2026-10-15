"""The test tiers: which tests a run selects, chosen on the command line.

Four markers say that a test is costly: ``slow`` (a second or more), ``integration`` (the real
OpenMM and biotite stack), ``openfold`` (the parsers of an installed OpenFold3) and ``gpu`` (a
CUDA device). A run that selects tests by naming the four in a negative expression is long to
type and easy to get wrong, so the tiers are options instead, with the cheap tier as the default
(the ``--runslow`` pattern of the pytest documentation, extended to four markers):

- ``pytest`` is the fast tier: the tests that carry none of the four markers;
- ``pytest --full`` adds the ``slow``, ``integration`` and ``openfold`` tests, which is every
  test except the GPU ones: the gate before a pull request or a release, and the run of CI;
- ``pytest --gpu`` adds the ``gpu`` tests (they keep their ``skipif`` when no CUDA device
  exists), and ``pytest --full --gpu`` selects every test. A test that carries ``gpu`` follows
  ``--gpu`` alone, whatever other speed marker it has;
- a test file or a node id named on the command line runs whatever its tier, since it was asked
  for. A directory (``tests``, ``tests/poses``) and no argument at all (``testpaths``) obey the
  tier. ``--tier-named`` makes the named files obey it as well, for a script that names the
  files to run (``scripts/pytest_for_diff.py``);
- ``-m EXPR`` that mentions any of the four markers is a tier chosen by hand, so no tier
  filtering is applied and ``-m "not gpu"`` keeps its meaning of "every test but the GPU ones",
  ``-m slow`` runs the slow tests, and a command that was written before the options existed
  selects what it selected. An expression about anything else (``-m poses``, ``-m "poses or
  preflight"``) leaves the tier on, applied to the tests the expression selects. ``-k`` does the
  same.

The run prints one line about its tier, with the number of tests selected and the number that
each option would add (``describe``). With pytest-xdist the workers collect and the controller
prints that line, at the end of the session, from the counts that the workers hand over.

The module imports only the standard library, so ``tests/conftest.py`` (which holds the hooks)
and ``tests/test_tiers.py`` (which checks the rules without running pytest) share one
implementation. The decision for a test depends only on its markers, on its place in the
command line and on the options, so a pytest-xdist controller and every worker agree on it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

HEAVY_MARKERS = ("slow", "integration", "openfold")
GPU_MARKER = "gpu"
TIER_MARKERS = (*HEAVY_MARKERS, GPU_MARKER)

FAST, HEAVY, GPU = "fast", "heavy", "gpu"

# The characters that pytest allows in an identifier of a -m expression (_pytest/mark/expression);
# the keywords and, or and not come out of the scan as well and are no marker.
_EXPRESSION_WORD = re.compile(r"(?:\w|:|\+|-|\.|\[|\]|\\|/)+")


def markers_in_expression(expression: str) -> set[str]:
    """The words of a ``-m`` expression: whole identifiers, so ``slowish`` is not ``slow``, and
    the keywords and, or and not are among them."""
    return set(_EXPRESSION_WORD.findall(expression))


def kind_of(marker_names: Iterable[str]) -> str:
    """``gpu`` for a test with the gpu marker, ``heavy`` for another speed marker, else ``fast``."""
    names = set(marker_names)
    if GPU_MARKER in names:
        return GPU
    return HEAVY if names.intersection(HEAVY_MARKERS) else FAST


@lru_cache(maxsize=None)
def _real_path(path: str) -> str:
    """Cached, since a run asks it once per test and a file holds many tests."""
    return os.path.normcase(os.path.realpath(path))


@dataclass(frozen=True)
class Named:
    """A test file named on the command line, whole (``node`` empty) or by a node id."""

    path: str  # real path of the file
    node: str  # what follows the first "::", for example "TestA::test_b" or "test_c[1]"


def named_targets(arguments: Iterable[str], invocation_dir: str | Path) -> tuple[Named, ...]:
    """The files among the positional arguments of the command line, with their node ids.

    An argument is relative to the directory pytest was started in. A directory, a missing path
    and anything else is left out: the tests below a directory obey the tier.
    """
    targets = []
    for argument in arguments:
        path_text, _, node = argument.partition("::")
        path = _real_path(os.path.join(invocation_dir, path_text))
        if os.path.isfile(path):
            targets.append(Named(path, node))
    return tuple(targets)


def is_named(item_path: str, nodeid: str, targets: Iterable[Named]) -> bool:
    """Whether a test was asked for by file or by node id.

    A node id names the tests it starts: ``test_a.py::test_b`` is also ``test_b[1]`` and
    ``test_a.py::TestC`` is every test of the class, as pytest selects them.
    """
    rest = nodeid.partition("::")[2]
    path = _real_path(item_path)
    for target in targets:
        if target.path != path:
            continue
        if not target.node or rest == target.node:
            return True
        if rest.startswith((target.node + "::", target.node + "[")):
            return True
    return False


@dataclass(frozen=True)
class TierRequest:
    """What the command line says about the tier."""

    full: bool = False
    gpu: bool = False
    tier_named: bool = False
    markexpr: str = ""
    named: tuple[Named, ...] = ()

    @property
    def chosen_by_hand(self) -> bool:
        """``-m`` mentions a tier marker: the person chose the tier, nothing is held back."""
        return bool(markers_in_expression(self.markexpr).intersection(TIER_MARKERS))

    @property
    def label(self) -> str:
        if self.chosen_by_hand:
            return "none, -m names a tier marker"
        return ("full" if self.full else "fast") + ("+gpu" if self.gpu else "")

    def holds_back(self, kind: str, named: bool) -> bool:
        """Whether the tier leaves out a test of this kind, named on the command line or not."""
        if self.chosen_by_hand or (named and not self.tier_named):
            return False
        if kind == GPU:
            return not self.gpu
        return kind == HEAVY and not self.full


@dataclass(frozen=True)
class TierCounts:
    """How many tests a run collected and what the tier held back from those left by -m and -k."""

    collected: int  # all the tests collected, before -m, -k and the tier
    selected: int
    held_heavy: int  # slow, integration or openfold, without the gpu marker: --full adds them
    held_gpu: int  # with the gpu marker, whatever else they carry: --gpu adds them

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _tests(count: int, noun: str = "") -> str:
    return f"{count} {noun + ' ' if noun else ''}test{'' if count == 1 else 's'}"


def describe(request: TierRequest, counts: TierCounts) -> str:
    """The line the run prints about its tier, for example::

        tier: fast - 8809 of 9304 tests selected; --full adds 407 slow/integration/openfold tests,
        --gpu adds 88 GPU tests

    (one line). An option that is already given has no clause, and neither has a count of zero.
    """
    line = f"tier: {request.label} - {counts.selected} of {_tests(counts.collected)} selected"
    adds = []
    if counts.held_heavy:
        adds.append(f"--full adds {_tests(counts.held_heavy, 'slow/integration/openfold')}")
    if counts.held_gpu:
        adds.append(f"--gpu adds {_tests(counts.held_gpu, 'GPU')}")
    return "; ".join([line, ", ".join(adds)]) if adds else line
