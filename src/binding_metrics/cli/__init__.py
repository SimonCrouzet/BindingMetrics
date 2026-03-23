"""CLI utilities shared across all binding-metrics entry points."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def log_to_file(log_file):
    """Context manager: redirect stdout+stderr to *log_file* when provided.

    Usage::

        with log_to_file(args.log_file):
            # all print() calls go to the file (or stdout if log_file is None)
            ...
    """
    if log_file is None:
        yield
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w", encoding="utf-8", buffering=1)
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
