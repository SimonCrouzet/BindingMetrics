"""Registry of the predictor adapters.

Each ``ParserSpec`` names one adapter without importing it, as ``MetricSpec`` does for a
metric: ``PARSERS`` maps a model name to its spec, ``get_parser(name)`` returns an adapter
instance, and the adapter module is imported when ``ParserSpec.load`` runs. Loading the
registry imports no adapter, so an adapter that needs a heavier dependency later does not
slow the others down.

What a consumer may rely on:

* the field names of ``ParserSpec`` and the names ``PARSERS``, ``get_parser`` and
  ``register_parser`` keep their meaning; the registry grows by adding entries;
* ``sorted(PARSERS)`` lists every registered model, and the contract tests in
  ``tests/predictors`` run over that list;
* ``get_parser(name)`` raises ``KeyError`` (with the list of names) for an unknown model;
* ``ParserSpec.load_capabilities()`` returns the ``capabilities`` class attribute of the
  adapter (None when it declares no constraint) and imports the adapter module only when it
  is called; adapters import numpy only, so this needs no installed model.

Adding a model: append one ``ParserSpec`` to ``PARSERS`` below (one entry per line block, in
alphabetical order) and add ``tests/predictors/synth_<name>.py``. Code outside the package,
such as a test with a stub model, adds an entry with ``register_parser``.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from binding_metrics.predictors.base import PredictionParser

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ParserSpec:
    """Names one adapter and where to import it from.

    Attributes:
        name: Registry key, lower case letters, digits and underscores; equal to the
            adapter's ``name`` class attribute.
        import_path: ``"module.path:ClassName"`` of a ``PredictionParser`` subclass whose
            constructor takes no argument. Resolved lazily.
        display_name: Name for reports; equal to the adapter's ``display_name``.
        family: ``"af2"`` or ``"af3"``; equal to the adapter's ``family``.
        description: One line for humans (the model and the layout the adapter reads).
    """

    name: str
    import_path: str
    display_name: str
    family: str
    description: str = ""

    def __post_init__(self):
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"parser name {self.name!r} must be lower case letters, digits and underscores"
            )
        if self.import_path.count(":") != 1:
            raise ValueError(f"import_path {self.import_path!r} must look like 'module.path:Class'")
        from binding_metrics.predictors.base import FAMILIES

        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")

    def load(self) -> type[PredictionParser]:
        """Import the adapter module and return the adapter class.

        Raises:
            ImportError: The module cannot be imported.
            AttributeError: The module has no class of that name.
            TypeError: The name is not a ``PredictionParser`` subclass.
        """
        from binding_metrics.predictors.base import PredictionParser

        module_path, class_name = self.import_path.split(":")
        cls = getattr(importlib.import_module(module_path), class_name)
        if not (isinstance(cls, type) and issubclass(cls, PredictionParser)):
            raise TypeError(f"{self.import_path} is not a PredictionParser subclass")
        return cls

    def create(self) -> PredictionParser:
        """Return a new adapter instance."""
        return self.load()()

    def load_capabilities(self) -> Optional[Any]:
        """The adapter's ``capabilities`` (None when it declares no constraint).

        Imports the adapter module, which is cheap, and does not instantiate the adapter.
        """
        return self.load().capabilities


# One ParserSpec per adapter, in alphabetical order of the name. The dict is keyed by name so
# that ``sorted(PARSERS)`` lists the models.
PARSERS: dict[str, ParserSpec] = {}


def register_parser(spec: ParserSpec, *, replace: bool = False) -> None:
    """Add ``spec`` to ``PARSERS``.

    Args:
        spec: The adapter to register.
        replace: Overwrite an existing entry of the same name; without it a repeated name
            raises.

    Raises:
        ValueError: A different adapter is already registered under that name.
    """
    if spec.name in PARSERS and not replace and PARSERS[spec.name] != spec:
        raise ValueError(f"a parser named {spec.name!r} is already registered")
    PARSERS[spec.name] = spec


def get_parser(name: str) -> PredictionParser:
    """Return a new adapter instance for the model ``name``.

    Raises:
        KeyError: No adapter has that name; the message lists the available names.
    """
    try:
        spec = PARSERS[name]
    except KeyError:
        available = ", ".join(sorted(PARSERS)) or "none registered"
        raise KeyError(f"Unknown predictor {name!r}. Available: {available}") from None
    return spec.create()
