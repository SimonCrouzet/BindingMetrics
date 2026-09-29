"""Private helpers shared by the metric modules.

Nothing here is public API: the names may change without notice. The metric
modules keep their own private names (``_import_biotite`` and the like) as thin
wrappers, because tests patch those.
"""

from __future__ import annotations

import argparse
from typing import Optional


def resolve_chain_role(
    legacy_name: str,
    legacy_value: Optional[str],
    alias_name: str,
    alias_value: Optional[str],
    *,
    required: bool = False,
) -> Optional[str]:
    """Merge a legacy chain parameter with its role alias.

    The metric functions name the binder chain ``peptide_chain``,
    ``design_chain``, ``binder_chain`` or ``chain`` and the target chain
    ``receptor_chain``. The role aliases ``binder_chain`` and ``target_chain``
    fill the legacy parameter, so a caller can use one spelling everywhere.

    Args:
        legacy_name: Name of the existing parameter (used in messages).
        legacy_value: Its value.
        alias_name: Name of the alias parameter (used in messages).
        alias_value: Its value.
        required: True when the legacy parameter had no default before the
            alias existed; a missing value then raises ``TypeError``, as the
            missing positional argument did.

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
