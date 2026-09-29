"""Backends and pocket description for an interaction energy from a machine-learned force field.

No energy is computed yet. The module fixes the names of the parts of a static-pocket
interaction energy scored with a pretrained machine-learned force field (MLFF), the
protocol of Ryczko et al., ChemRxiv 10.26434/chemrxiv.15008810 (v2, 20 Sep 2026; a
preprint whose full text was not read when this interface was written; no public
code was found). A backend adapter and a pocket cropper are the missing pieces, and
both are added behind these names:

- :class:`MLFFBackend` is the base class of a backend: one pretrained model that
  returns the energy of an atom set in eV.
- :class:`PocketSpec` describes how the pocket around the binder is cut out.
- :func:`register_backend`, :func:`get_backend` and :func:`available_backends`
  manage the backends. The names ``uma``, ``mace``, ``orb`` and ``aimnet2`` are
  placeholders: they are known, they are never available and they raise
  ``NotImplementedError`` when asked for.

Importing the module needs no MLFF package, no OpenMM and no torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, ClassVar, Literal

__all__ = [
    "MLFFBackend",
    "PocketSpec",
    "available_backends",
    "get_backend",
    "register_backend",
]

_REFERENCE = "Ryczko et al., ChemRxiv 10.26434/chemrxiv.15008810"

_POCKET_CAPS = ("none", "hydrogen")
_POCKET_PROTONATIONS = ("as_given", "reprotonate")


def _require_choice(argument: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{argument} must be one of {allowed}, got {value!r}")


@dataclass(frozen=True)
class PocketSpec:
    """How the pocket around the binder is cut out of the complex.

    Documented and validated, and not used until a backend and a pocket cropper
    exist. The pocket is the set of receptor residues that come within
    ``cutoff_angstrom`` of the binder, plus the binder. The reference protocol's
    own radius and capping are TO VERIFY against its full text.

    Attributes:
        cutoff_angstrom: Distance in Å from any binder atom within which a
            receptor residue joins the pocket. The default is a placeholder.
        cap: How the bonds cut at the pocket boundary are closed. ``"hydrogen"``
            saturates each cut bond with a hydrogen atom along the bond; ``"none"``
            leaves the valence open, which an MLFF energy does not describe well.
        include_waters: Keep waters that lie within the cutoff.
        protonation: ``"as_given"`` keeps the hydrogens of the input file;
            ``"reprotonate"`` assigns protonation states before the pocket is cut.
            The total charge of the pocket depends on this choice.
    """

    cutoff_angstrom: float = 6.0
    cap: Literal["none", "hydrogen"] = "hydrogen"
    include_waters: bool = False
    protonation: Literal["as_given", "reprotonate"] = "as_given"

    def __post_init__(self) -> None:
        if (
            isinstance(self.cutoff_angstrom, bool)
            or not isinstance(self.cutoff_angstrom, Real)
            or not isfinite(self.cutoff_angstrom)
            or self.cutoff_angstrom <= 0
        ):
            raise ValueError(
                f"cutoff_angstrom must be a positive finite number, got {self.cutoff_angstrom!r}"
            )
        _require_choice("cap", self.cap, _POCKET_CAPS)
        _require_choice("protonation", self.protonation, _POCKET_PROTONATIONS)
        if not isinstance(self.include_waters, bool):
            raise ValueError(f"include_waters must be a bool, got {self.include_waters!r}")


class MLFFBackend(ABC):
    """One pretrained machine-learned force field that returns the energy of an atom set.

    A backend adapter subclasses this, sets ``name`` and ``weights_licence`` and is
    passed to :func:`register_backend`. Loading a model and importing its package
    belong in ``__init__``, so importing the adapter's module stays cheap, and
    :meth:`is_available` says whether that would succeed without doing it.

    Attributes:
        name: Lower-case registry name, for example ``"uma"``.
        weights_licence: The licence of the model weights, in one line. It is shown
            in error messages and in the result of the energy function. The weights
            are never bundled with this package; the user obtains them.
    """

    name: ClassVar[str]
    weights_licence: ClassVar[str]

    @classmethod
    def is_available(cls) -> bool:
        """Whether the backend can run here: its package importable and its weights reachable.

        The base class reports False, so a backend that does not override this is
        never listed by :func:`available_backends`.
        """
        return False

    @abstractmethod
    def energy_ev(self, atoms: Any, *, charge: int, spin: int) -> float:
        """Energy of one atom set, in eV.

        Args:
            atoms: The atoms as an ASE ``Atoms`` object (positions in Å). ASE is
                imported by the adapter, never by this module.
            charge: Total charge of the atom set, in units of e.
            spin: Spin multiplicity handed to the model, 1 for a closed-shell atom
                set. The convention of each model is TO VERIFY when its adapter is
                written.
        """


class _PlaceholderBackend(MLFFBackend):
    """A backend name that is reserved and has no adapter: asking for it raises."""

    def __init__(self) -> None:
        raise NotImplementedError(
            f"MLFF backend {self.name!r} is a placeholder: there is no adapter for it yet, "
            f"so it cannot compute an energy. Reference protocol: {_REFERENCE}. "
            f"Weights licence: {self.weights_licence}"
        )

    def energy_ev(self, atoms: Any, *, charge: int, spin: int) -> float:
        raise NotImplementedError(f"MLFF backend {self.name!r} is a placeholder")


_UNVERIFIED_LICENCE = (
    "not checked; read the licence of the weights you download before you use them"
)

_BACKENDS: dict[str, type[MLFFBackend]] = {}


def register_backend(cls: type[MLFFBackend]) -> type[MLFFBackend]:
    """Add a backend class to the registry under ``cls.name``; usable as a decorator.

    A real backend may take the name of a placeholder and replaces it. Any other
    name that is already registered is an error.

    Args:
        cls: A subclass of :class:`MLFFBackend` with ``name`` (a non-empty,
            lower-case string) and ``weights_licence`` (a non-empty string).

    Returns:
        ``cls``, unchanged.

    Raises:
        TypeError: If ``cls`` is not a subclass of :class:`MLFFBackend`.
        ValueError: If ``name`` or ``weights_licence`` is missing or malformed, or
            the name belongs to a backend that is not a placeholder.
    """
    if not (isinstance(cls, type) and issubclass(cls, MLFFBackend)):
        raise TypeError(f"register_backend needs a subclass of MLFFBackend, got {cls!r}")
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name or name != name.strip().lower() or " " in name:
        raise ValueError(f"{cls.__name__}.name must be a non-empty lower-case string, got {name!r}")
    licence = getattr(cls, "weights_licence", None)
    if not isinstance(licence, str) or not licence.strip():
        raise ValueError(f"{cls.__name__}.weights_licence must be a non-empty string")
    existing = _BACKENDS.get(name)
    if existing is not None and not issubclass(existing, _PlaceholderBackend):
        raise ValueError(f"an MLFF backend named {name!r} is already registered")
    _BACKENDS[name] = cls
    return cls


def _backend_class(name: str) -> type[MLFFBackend]:
    try:
        return _BACKENDS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown MLFF backend {name!r}; known backends: {sorted(_BACKENDS)}"
        ) from None


def get_backend(name: str) -> MLFFBackend:
    """Return an instance of the backend registered as ``name``.

    Args:
        name: A registry name, such as ``"uma"``.

    Raises:
        ValueError: If no backend has that name.
        NotImplementedError: If the name is a placeholder; the message names the
            backend, the reference and the weights licence.
    """
    return _backend_class(name)()


def available_backends() -> list[str]:
    """Names of the registered backends that can run here, sorted.

    A placeholder is never available, so the list is empty until an adapter is
    registered and finds its package and weights.
    """
    return sorted(name for name, cls in _BACKENDS.items() if cls.is_available())


@register_backend
class _UMABackend(_PlaceholderBackend):
    name = "uma"
    # Terms as stated on the UMA model card, huggingface.co/facebook/UMA (checked 2026-09-29).
    weights_licence = (
        "FAIR Chemistry License v1 (gated on Hugging Face, with an acceptable-use policy "
        "and an acknowledgement duty); the weights must not be bundled"
    )


@register_backend
class _MACEBackend(_PlaceholderBackend):
    name = "mace"
    weights_licence = _UNVERIFIED_LICENCE


@register_backend
class _OrbBackend(_PlaceholderBackend):
    name = "orb"
    weights_licence = _UNVERIFIED_LICENCE


@register_backend
class _AIMNet2Backend(_PlaceholderBackend):
    name = "aimnet2"
    weights_licence = _UNVERIFIED_LICENCE
