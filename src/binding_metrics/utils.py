"""Shared utility helpers (no heavy top-level imports)."""

import logging
import sys

logger = logging.getLogger(__name__)

_PACKAGE_LOGGER = "binding_metrics"


class _CurrentStreamHandler(logging.StreamHandler):
    """Stream handler that writes to whatever ``sys.<stream_name>`` is at emit time.

    ``cli.log_to_file`` replaces ``sys.stdout`` and ``sys.stderr`` for the duration
    of a run; a handler that kept the stream it was created with would keep
    writing to the console after that swap.
    """

    def __init__(self, stream_name: str, level: int = logging.NOTSET):
        self._stream_name = stream_name
        super().__init__()
        self.setLevel(level)

    @property
    def stream(self):
        return getattr(sys, self._stream_name)

    @stream.setter
    def stream(self, value) -> None:
        # StreamHandler.__init__ assigns a stream; the live sys attribute wins.
        pass


def configure_logging(level: int = logging.INFO) -> None:
    """Send ``binding_metrics`` log records to the console, once, from a CLI entry point.

    Library code logs through ``logging.getLogger(__name__)`` and never prints;
    a command-line ``main()`` calls this first so those records keep appearing
    exactly where the former ``print`` calls put them: records up to WARNING on
    stdout, ERROR and above on stderr, with the bare message and no prefix.
    Calling it again only updates the level, so nested entry points are safe.

    ``python -m binding_metrics.<module>`` runs the module as ``__main__``, so its
    ``getLogger(__name__)`` logger is named ``__main__`` and sits outside the
    package hierarchy; in that case the same handlers are attached to it. A user's
    own ``__main__`` script is not touched.

    Args:
        level: Threshold for the package logger (default INFO).
    """
    loggers = [logging.getLogger(_PACKAGE_LOGGER)]
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if (getattr(main_spec, "name", None) or "").startswith(f"{_PACKAGE_LOGGER}."):
        loggers.append(logging.getLogger("__main__"))
    for target in loggers:
        target.setLevel(level)
        _attach_console_handlers(target)


def _attach_console_handlers(target: logging.Logger) -> None:
    """Add the stdout/stderr handlers to ``target`` unless they are already there."""
    if any(isinstance(h, _CurrentStreamHandler) for h in target.handlers):
        return
    formatter = logging.Formatter("%(message)s")
    to_stdout = _CurrentStreamHandler("stdout")
    to_stdout.addFilter(lambda record: record.levelno < logging.ERROR)
    to_stderr = _CurrentStreamHandler("stderr", level=logging.ERROR)
    for handler in (to_stdout, to_stderr):
        handler.setFormatter(formatter)
        target.addHandler(handler)


def extend_report(report: dict, key: str, values: list) -> None:
    """Append ``values`` to the list at ``report[key]``, creating it if absent.

    Used by the prep functions that fill an optional ``report`` dict: lists
    accumulate when one dict is passed through several steps.
    """
    report.setdefault(key, []).extend(values)


def add_to_report(report: dict, key: str, count: int) -> None:
    """Add ``count`` to the integer at ``report[key]``, creating it if absent."""
    report[key] = report.get(key, 0) + count


def backfill_auth_columns(cif_file) -> None:
    """Backfill auth_atom_id/auth_comp_id from label_* equivalents if absent.

    BoltzGen CIFs (produced by gemmi.make_mmcif_document) omit these auth_*
    columns.  biotite.pdbx.get_structure falls back correctly to label_atom_id
    and label_comp_id, but emits a noisy UserWarning for every atom.  Copying
    the label columns under the auth names before calling get_structure silences
    the warnings without hiding any real issue.

    A file without an ``atom_site`` category, without the ``label_*`` source
    columns, or with several data blocks is left unchanged (logged at debug level):
    the caller's own read of the file reports whatever is actually wrong.
    """
    try:
        atom_site = cif_file.block["atom_site"]
        if "auth_atom_id" not in atom_site:
            atom_site["auth_atom_id"] = atom_site["label_atom_id"]
        if "auth_comp_id" not in atom_site:
            atom_site["auth_comp_id"] = atom_site["label_comp_id"]
    except (KeyError, ValueError) as exc:
        logger.debug("auth_* column backfill skipped: %s: %s", type(exc).__name__, exc)
