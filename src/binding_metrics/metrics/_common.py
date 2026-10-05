"""Private helpers of the metric modules: biotite import and structure loader,
chain-role aliases, OpenMM check and energy unit factors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from binding_metrics.utils import backfill_auth_columns

_MMCIF_SUFFIXES = (".cif", ".mmcif")

# Thermochemical calorie: 1 kcal = 4.184 kJ (exact by definition).
KCAL_TO_KJ = 4.184
KJ_TO_KCAL = 1.0 / KCAL_TO_KJ


def import_biotite(purpose: str):
    """Import the biotite modules the metrics use for reading structures.

    Args:
        purpose: What biotite is needed for, worded as the end of "biotite is
            required for <purpose>" in the install hint (for example
            ``"interface metrics"``).

    Returns:
        ``(struc, pdbx, pdb_io)``: ``biotite.structure``,
        ``biotite.structure.io.pdbx`` and ``biotite.structure.io.pdb``.

    Raises:
        ImportError: If biotite is not installed; the message names the extra.
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx
    except ImportError as exc:
        raise ImportError(
            f"biotite is required for {purpose}. Install with: pip install binding-metrics[biotite]"
        ) from exc
    return struc, pdbx, pdb_io


def require_openmm(feature: str) -> None:
    """Raise a ModuleNotFoundError that names the extra when OpenMM is missing.

    Args:
        feature: The function that needs OpenMM, for the message.

    Raises:
        ModuleNotFoundError: If OpenMM cannot be imported (an ImportError
            subclass, ``name="openmm"``), as ``get_forcefield`` does.
    """
    try:
        import openmm  # noqa: F401
    except ImportError as exc:
        raise ModuleNotFoundError(
            f"{feature} needs OpenMM, which could not be imported. "
            "Install it with `pip install binding-metrics[simulation]`, "
            "or use environment.yml for a GPU build.",
            name="openmm",
        ) from exc


def load_structure(
    path: str | Path,
    *,
    model: Optional[int] = 1,
    charge: bool = False,
    purpose: str = "structure loading",
):
    """Load a PDB or mmCIF file as a biotite AtomArray (or AtomArrayStack).

    mmCIF files (suffix ``.cif`` or ``.mmcif``) go through
    ``backfill_auth_columns`` first, so the author chain and residue IDs are
    used even when a file only has the label columns. Every other suffix is
    read as PDB.

    Args:
        path: Structure file.
        model: 1-based model number to read (default 1, the first model).
            None reads every model: an ``AtomArrayStack`` when the file has
            more than one.
        charge: Also read the mmCIF formal-charge column into the ``charge``
            annotation. A missing or malformed column is skipped, since the
            annotation is optional. PDB files are read without it.
        purpose: Wording for the install hint, see :func:`import_biotite`.

    Raises:
        ImportError: If biotite is not installed.
    """
    _, pdbx, pdb_io = import_biotite(purpose)
    path = Path(path)
    if path.suffix.lower() in _MMCIF_SUFFIXES:
        pdbx_file = pdbx.CIFFile.read(str(path))
        backfill_auth_columns(pdbx_file)
        if charge:
            try:
                return pdbx.get_structure(pdbx_file, model=model, extra_fields=["charge"])
            except (KeyError, ValueError):
                # Missing or malformed pdbx_formal_charge column: the charge annotation is optional.
                pass
        return pdbx.get_structure(pdbx_file, model=model)
    pdb_file = pdb_io.PDBFile.read(str(path))
    return pdb_io.get_structure(pdb_file, model=model)


def resolve_chain_role(
    legacy_name: str,
    legacy_value: Optional[str],
    alias_name: str,
    alias_value: Optional[str],
    *,
    required: bool = False,
) -> Optional[str]:
    """Return the chain ID given through a legacy parameter or its role alias.

    The aliases ``binder_chain`` and ``target_chain`` fill the parameters
    ``peptide_chain``, ``design_chain``, ``chain`` and ``receptor_chain``.

    Args:
        legacy_name: Name of the existing parameter (used in messages).
        legacy_value: Its value.
        alias_name: Name of the alias parameter (used in messages).
        alias_value: Its value.
        required: True when the chain is mandatory; a missing value then
            raises ``TypeError``.

    Returns:
        The chain ID given through either spelling, or None if neither was given.

    Raises:
        ValueError: If both spellings are given with different chain IDs.
        TypeError: If ``required`` and neither spelling is given.
    """
    if alias_value is None:
        chain = legacy_value
    elif legacy_value is None or legacy_value == alias_value:
        chain = alias_value
    else:
        raise ValueError(
            f"{legacy_name}={legacy_value!r} and {alias_name}={alias_value!r} name "
            "different chains; give only one of them"
        )
    if chain is None and required:
        raise TypeError(f"missing required argument: {legacy_name!r} (or its alias {alias_name!r})")
    return chain


class ChainAliasAction(argparse.Action):
    """Store a chain ID for an option that also has a role-alias spelling.

    Use it on an ``add_argument`` call that lists the old flag and its alias
    (``"--design-chain", "--binder-chain"``): both fill the same destination.
    Giving the two spellings with different IDs ends the program through
    ``parser.error``. Repeating one spelling keeps argparse's rule that the
    last value wins.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        spellings = namespace.__dict__.setdefault("_chain_spellings", {})
        previous = spellings.get(self.dest)
        if previous is not None:
            previous_option, previous_value = previous
            if previous_option != option_string and previous_value != values:
                parser.error(
                    f"argument {option_string}: {values!r} conflicts with "
                    f"{previous_option} {previous_value!r}; give only one of the two spellings"
                )
        spellings[self.dest] = (option_string, values)
        setattr(namespace, self.dest, values)


def resolve_cli_chain_alias(
    parser: argparse.ArgumentParser, args: argparse.Namespace, dest: str, alias_dest: str
) -> None:
    """Fold a separately parsed alias flag into the old option's attribute.

    For a CLI where the alias cannot share the old flag's destination (the
    geometry CLI reads the binder from ``--chain`` or ``--peptide-chain``
    depending on ``--metric``). Afterwards ``args.<dest>`` holds the chain ID
    from whichever flag was given; different IDs end the program through
    ``parser.error``.
    """
    try:
        resolved = resolve_chain_role(
            f"--{dest.replace('_', '-')}",
            getattr(args, dest),
            f"--{alias_dest.replace('_', '-')}",
            getattr(args, alias_dest),
        )
    except ValueError as exc:
        parser.error(str(exc))
    setattr(args, dest, resolved)
