"""Shared utility helpers (no heavy top-level imports)."""


def backfill_auth_columns(cif_file) -> None:
    """Backfill auth_atom_id/auth_comp_id from label_* equivalents if absent.

    BoltzGen CIFs (produced by gemmi.make_mmcif_document) omit these auth_*
    columns.  biotite.pdbx.get_structure falls back correctly to label_atom_id
    and label_comp_id, but emits a noisy UserWarning for every atom.  Copying
    the label columns under the auth names before calling get_structure silences
    the warnings without hiding any real issue.
    """
    try:
        atom_site = cif_file.block["atom_site"]
        if "auth_atom_id" not in atom_site:
            atom_site["auth_atom_id"] = atom_site["label_atom_id"]
        if "auth_comp_id" not in atom_site:
            atom_site["auth_comp_id"] = atom_site["label_comp_id"]
    except Exception:
        pass
