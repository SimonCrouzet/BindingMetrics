"""Private helpers shared by the metric modules.

Nothing here is public API: the names may change without notice. The metric
modules keep their own private names (``_import_biotite`` and the like) as thin
wrappers, because tests patch those.
"""

from __future__ import annotations

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
