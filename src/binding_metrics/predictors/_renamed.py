"""The deprecated name ``PredictionRecord``, which ``PredictedPose`` replaced.

The class kept its fields and methods when it was renamed. Each module that defined or
imported ``PredictionRecord`` hands the old name out through a module-level ``__getattr__``
(PEP 562) built by ``deprecated_name_getattr``, so ``from binding_metrics.predictors.record
import PredictionRecord`` and ``binding_metrics.predictors.PredictionRecord`` still work. The
old name is not in any ``__all__`` or ``dir()``, so a star import and a listing never trigger
the warning, and it is never cached on a module, so every use warns.

Each use warns once, attributed to the line that asked for the name. For ``from package import
Name`` the import machinery first checks that the package has the name (``_handle_fromlist``);
that probe returns the class without a warning, so the statement warns once.

Importing this module imports the standard library only.
"""

from __future__ import annotations

import sys
import warnings
from typing import Callable, Optional

#: The name that was replaced and its replacement.
OLD_NAME = "PredictionRecord"
NEW_NAME = "PredictedPose"

#: The text of the ``DeprecationWarning``.
MESSAGE = (
    f"{OLD_NAME} is deprecated and renamed {NEW_NAME}; the old name will be removed in a "
    "future release (not before 0.4.0)"
)


def old_name_class() -> type:
    """Warn that ``PredictionRecord`` is deprecated and return ``PredictedPose``.

    Call it from a module-level ``__getattr__`` only: the warning is attributed to the caller
    of that ``__getattr__``.
    """
    from binding_metrics.predictors.record import PredictedPose

    if not _is_import_probe(sys._getframe(2) if hasattr(sys, "_getframe") else None):
        warnings.warn(MESSAGE, DeprecationWarning, stacklevel=3)
    return PredictedPose


def _is_import_probe(frame) -> bool:
    """True when ``frame`` is the ``hasattr`` check of ``from package import Name``."""
    return (
        frame is not None
        and frame.f_code.co_name == "_handle_fromlist"
        and "importlib" in frame.f_code.co_filename
    )


def deprecated_name_getattr(
    module_name: str, fallback: Optional[Callable[[str], object]] = None
) -> Callable[[str], object]:
    """Build the module-level ``__getattr__`` that gives out ``PredictionRecord`` and warns.

    Args:
        module_name: ``__name__`` of the module the function is for, for the error message.
        fallback: What to ask for any other name; without it, an ``AttributeError`` as a
            module without ``__getattr__`` raises.
    """

    def module_getattr(name: str):
        if name == OLD_NAME:
            return old_name_class()
        if fallback is not None:
            return fallback(name)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return module_getattr
