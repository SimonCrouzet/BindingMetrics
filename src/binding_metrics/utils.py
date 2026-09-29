"""Shared utility helpers (no heavy top-level imports)."""

import logging

logger = logging.getLogger(__name__)


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
