"""CLI utilities shared across all binding-metrics entry points."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


@contextmanager
def log_to_file(log_file, mode: str = "w"):
    """Context manager: redirect stdout+stderr to *log_file* when provided.

    Usage::

        with log_to_file(args.log_file):
            # all print() calls go to the file (or stdout if log_file is None)
            ...

    ``mode`` is the ``open`` mode: ``"w"`` (default) starts the file afresh,
    ``"a"`` appends, for callers that enter the context several times on the
    same file (one batch run logging every sample to a shared ``--log-file``).
    """
    if log_file is None:
        yield
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, mode, encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = fh
    try:
        yield
    finally:
        fh.flush()
        sys.stdout = old_out
        sys.stderr = old_err
        fh.close()


def _apply_log_redirect(log_file) -> None:
    """Redirect stdout+stderr to *log_file* for the rest of the process.

    Unlike ``log_to_file``, this is a fire-and-forget helper for CLIs whose
    body is too large to wrap in a context manager.  Streams are restored when
    the process exits normally (via ``atexit``).
    """
    import atexit

    if log_file is None:
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w", encoding="utf-8", buffering=1)
    _old_out, _old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = fh

    def _restore():
        fh.flush()
        sys.stdout = _old_out
        sys.stderr = _old_err
        fh.close()

    atexit.register(_restore)


def add_log_file_arg(parser) -> None:
    """Add the --log-file argument to an argparse parser."""
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Redirect all output (stdout + stderr) to this file",
    )


def seed_arg(value: str) -> Optional[int]:
    """Parse ``--random-seed``: an integer, or ``none``/``random``/``off`` for fresh randomness."""
    if value.strip().lower() in ("none", "random", "off"):
        return None
    return int(value)


def add_random_seed_arg(parser, what: str) -> None:
    """Add ``--random-seed INT|none`` to an argparse parser.

    The default is the library-wide ``DEFAULT_RANDOM_SEED``, so a CLI run is
    reproducible unless the user asks for fresh randomness with ``none``.

    Args:
        parser: Parser or argument group to add the flag to.
        what: Which stochastic steps the seed drives, worded for the help text
            (for example ``"ion placement"``).
    """
    from binding_metrics._constants import DEFAULT_RANDOM_SEED

    parser.add_argument(
        "--random-seed",
        type=seed_arg,
        default=DEFAULT_RANDOM_SEED,
        metavar="INT|none",
        help=(
            f"Seed for {what}. A fixed integer makes the run reproducible "
            "(default: %(default)s); pass 'none' for fresh randomness each run."
        ),
    )


_SMALL_MOLECULES_CHOICES = ("auto", "none")


def small_molecules_arg(value: str):
    """Parse ``--small-molecules``: ``"auto"``, or ``"none"`` (returned as ``None``).

    Any other string is rejected. ``RelaxationConfig.small_molecules`` iterates
    a string it does not recognise character by character and treats each
    character as a SMILES, so a typo such as ``aut`` would register methane
    and other fragments as "small molecules" instead of failing. Explicit
    SMILES lists are a Python-API feature (``RelaxationConfig(small_molecules=[...])``)
    and are not accepted on the command line.

    Use as ``type=small_molecules_arg`` in an argparse argument; the value is
    matched case-insensitively.

    Raises:
        argparse.ArgumentTypeError: for anything but ``auto`` / ``none``.
    """
    choice = str(value).strip().lower()
    if choice not in _SMALL_MOLECULES_CHOICES:
        raise argparse.ArgumentTypeError(
            f"invalid value {value!r}: expected one of {', '.join(_SMALL_MOLECULES_CHOICES)}"
        )
    return None if choice == "none" else choice
