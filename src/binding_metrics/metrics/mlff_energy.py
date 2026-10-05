"""Reserved interface for an interaction energy from a machine-learned force field.

No energy is computed yet. The module fixes the names and the argument checks of a
static-pocket interaction energy scored with a pretrained machine-learned force
field (MLFF), the protocol of Ryczko et al., ChemRxiv 10.26434/chemrxiv.15008810
(v2, 20 Sep 2026; a preprint whose full text was not read when this interface was
written; no public code was found). A backend adapter and a pocket cropper are
the missing pieces, and both are added behind these names:

- :class:`MLFFBackend` is the base class of a backend: one pretrained model that
  returns the energy of an atom set in eV.
- :class:`PocketSpec` describes how the pocket around the binder is cut out.
- :func:`register_backend`, :func:`get_backend` and :func:`available_backends`
  manage the backends. The names ``uma``, ``mace``, ``orb`` and ``aimnet2`` are
  placeholders: they are known, they are never available and they raise
  ``NotImplementedError`` when asked for.
- :func:`compute_mlff_interaction_energy` validates its arguments and raises
  ``NotImplementedError``.

Importing the module needs no MLFF package, no OpenMM and no torch. The function is
deliberately absent from the metric registry (``binding_metrics.metrics.registry``),
so code that runs every registered metric never reaches it; it is registered when
a backend lands.

Usage:
    from binding_metrics.metrics.mlff_energy import compute_mlff_interaction_energy

    compute_mlff_interaction_energy("complex.cif", "B", "A", backend="uma")
    # NotImplementedError: names the backend and the reference
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from binding_metrics.metrics._common import KJ_TO_KCAL, resolve_chain_role

__all__ = [
    "MLFFBackend",
    "PocketSpec",
    "available_backends",
    "compute_mlff_interaction_energy",
    "get_backend",
    "register_backend",
]

_REFERENCE = "Ryczko et al., ChemRxiv 10.26434/chemrxiv.15008810"

# One eV per particle in kJ/mol: e * N_A / 1000, with the exact SI values of e and N_A.
_KJ_MOL_PER_EV = 96.48533212331002

# Energy unit -> factor applied to a backend energy in eV. The keys are the accepted
# values of ``unit`` and the suffix of the result key.
_ENERGY_UNIT_PER_EV: dict[str, float] = {
    "ev": 1.0,
    "kj_mol": _KJ_MOL_PER_EV,
    "kcal_mol": _KJ_MOL_PER_EV * KJ_TO_KCAL,
}
_HETERO_MODES = ("ignore", "keep")
_HYDROGEN_MODES = ("ignore", "keep")
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

    # TODO(mlff): a real adapter subclasses MLFFBackend (not this class), sets name and
    #   weights_licence, and calls register_backend, which then replaces the placeholder.
    #   Each adapter has to:
    #   - import its package inside __init__ (never at module level) and report the missing
    #     package or weights in is_available(), so available_backends() stays cheap and
    #     never raises;
    #   - fetch the weights on first use into the user's cache, after the user has accepted
    #     the licence; never ship or vendor them (see weights_licence);
    #   - take the device from the caller (cuda by default, as the rest of the package) and
    #     load the model once per instance;
    #   - implement energy_ev(atoms, *, charge, spin) by putting charge and spin on the ASE
    #     atoms and returning the potential energy in eV, with no unit conversion (the
    #     conversion to kcal/mol and kJ/mol happens once, in compute_mlff_interaction_energy);
    #   - record the model name and version it ran (checkpoint id) so the result can carry it.

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
    # TODO(mlff): implement with fairchem-core (MIT package): load a UMA predictor and wrap it
    #   in FAIRChemCalculator with the molecular task (the OMol25 task, whose energies come
    #   with a charge and a spin); the checkpoint names seen so far are uma-s-1.2.1 and
    #   uma-m-1.1 (TO VERIFY the current names, the task name and how charge and spin are
    #   passed, atoms.info["charge"] and atoms.info["spin"]). The weights are gated on
    #   Hugging Face: the user must accept the FAIR Chemistry License and be logged in, and
    #   the acknowledgement duty of the licence must be honoured in the docs and the output.
    name = "uma"
    # Terms as stated on the UMA model card, huggingface.co/facebook/UMA (checked 2026-09-29).
    weights_licence = (
        "FAIR Chemistry License v1 (gated on Hugging Face, with an acceptable-use policy "
        "and an acknowledgement duty); the weights must not be bundled"
    )


@register_backend
class _MACEBackend(_PlaceholderBackend):
    # TODO(mlff): implement with mace-torch (MIT package). Pick a molecular model, not a
    #   materials one (MACE-MP): TO VERIFY which released MACE checkpoint was trained on
    #   OMol25-like data and how it takes charge and spin. TO VERIFY the licence of the chosen
    #   weights before setting weights_licence; no weight licence was checked for MACE.
    name = "mace"
    weights_licence = _UNVERIFIED_LICENCE


@register_backend
class _OrbBackend(_PlaceholderBackend):
    # TODO(mlff): implement with orb-models (Apache-2.0 package): load a molecular Orb
    #   checkpoint and its ASE calculator. TO VERIFY which checkpoint handles charged and
    #   open-shell systems and how it takes charge and spin, and the licence of the weights;
    #   no weight licence was checked for Orb.
    name = "orb"
    weights_licence = _UNVERIFIED_LICENCE


@register_backend
class _AIMNet2Backend(_PlaceholderBackend):
    # TODO(mlff): implement with the AIMNet2 ASE calculator. TO VERIFY the package name, the
    #   licence of the code and the weights, and the element and charge range: AIMNet2 covers
    #   a limited set of elements and total charges, so is_available() or energy_ev() must
    #   refuse an atom set outside it with a clear message instead of returning a number.
    name = "aimnet2"
    weights_licence = _UNVERIFIED_LICENCE


def _require_chain_id(argument: str, value: Optional[str]) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{argument} must be a non-empty chain ID or None, got {value!r}")


# TODO(mlff): when a backend lands and this function computes an energy:
#   - register it in metrics/registry.py as a MetricSpec: input_type "static_structure",
#     chain_mode "interface", direction "lower", requires_gpu True, cost_class "model"
#     (the registry docs describe it as a structure-prediction run: TO VERIFY it fits an
#     MLFF energy, or extend that description), unit "kcal/mol", headline_key
#     "mlff_interaction_energy_kcal_mol", and requires_extras
#     set to a new "mlff" extra declared in pyproject.toml (fairchem-core, mace-torch,
#     orb-models, ase; check the extra against the registry contract tests);
#   - remove the function from _NOT_METRICS in tests/test_registry.py;
#   - add the lazy export in metrics/__init__.py and a pointer in README.md;
#   - add tests with a tiny real backend or a mocked one (a fake MLFFBackend returning fixed
#     energies, as tests/test_mlff_energy.py does) that check the three-term arithmetic, the
#     unit conversion, the NaN-plus-reason path and the returned keys;
#   - update the docs/metrics.md subsection and add a CHANGELOG entry;
#   - turn the "reserved interface" wording of this docstring and of the module into a
#     description of what the function does.
def compute_mlff_interaction_energy(
    structure_path: str | Path,
    binder_chain: Optional[str] = None,
    receptor_chain: Optional[str] = None,
    *,
    backend: str = "uma",
    pocket: Optional[PocketSpec] = None,
    unit: Literal["kcal_mol", "kj_mol", "ev"] = "kcal_mol",
    hetero: Literal["ignore", "keep"] = "ignore",
    hydrogens: Literal["ignore", "keep"] = "keep",
    target_chain: Optional[str] = None,
) -> dict[str, Any]:
    """Interaction energy of a binder and its target from a machine-learned force field.

    Reserved interface: the arguments are validated and the function then raises
    ``NotImplementedError``. It is not in the metric registry.

    Planned method, after Ryczko et al., ChemRxiv 10.26434/chemrxiv.15008810 (a
    static-pocket score: no relaxation and no sampling, so nothing is stochastic).
    The pocket around the binder is cut out and capped, and the energy is
    E(pocket complex) - E(binder in the pocket) - E(receptor pocket), each term
    evaluated by the backend at the geometry of the complex. The three-term form is
    inferred from the abstract of the reference and is TO VERIFY against its full text.

    How it relates to the rest of the package. The energy complements
    :func:`binding_metrics.metrics.energy.compute_interaction_energy` (ff14SB with
    implicit solvent, kJ/mol) as a second, independent column and does not replace
    it. E_int includes generalised Born solvation and the solvent treatment of the
    reference protocol is not known, so the two values may not be comparable.

    Limits of validation. The reference benchmarks congeneric small-molecule series.
    Peptides, D-amino acids, N-methylated and phosphorylated residues and macrocycles
    are unvalidated. A pocket cropper with capping is the missing piece; each of
    those residue types would also need its total charge set correctly, and the
    charge handling of the model changes the result.

    Licence of the weights. The weights of a backend are not part of this package
    and are never downloaded by it. UMA weights are gated under the FAIR Chemistry
    License v1 (acceptable-use policy and acknowledgement duty) and must not be
    bundled or redistributed. The licence of each backend is on
    ``MLFFBackend.weights_licence`` and appears in the result.

    Args:
        structure_path: PDB or mmCIF file of the complex. It is read once a backend
            exists; the current version does not open it.
        binder_chain: Chain ID of the binder. None will take the smallest protein
            chain, as the other metrics do.
        receptor_chain: Chain ID of the target. None will take the largest protein
            chain.
        backend: Registry name of the model: ``"uma"``, ``"mace"``, ``"orb"``,
            ``"aimnet2"`` or the name of a registered backend.
        pocket: A :class:`PocketSpec`. None means ``PocketSpec()``.
        unit: Unit of the returned energy: ``"kcal_mol"``, ``"kj_mol"`` or ``"ev"``.
        hetero: As in the other structure metrics. ``"ignore"`` keeps polymer atoms
            (amino acids, AMBER variants, ACE/NME/NH2 caps) and drops waters, ions
            and ligands; ``"keep"`` uses every atom of the chain.
        hydrogens: ``"keep"`` (default) hands the hydrogens of the file to the model;
            ``"ignore"`` drops them first. A model energy needs a complete set of
            hydrogens, so ``"ignore"`` is for inputs that will be reprotonated.
        target_chain: Alias of ``receptor_chain``.

    Returns:
        Once a backend exists, a dict with the keys below. It is never returned now.

        - ``mlff_interaction_energy_<unit>`` (float): the energy, with ``<unit>`` as
          given, so ``mlff_interaction_energy_kcal_mol`` by default; NaN when it
          could not be computed.
        - ``backend`` (str): the backend name.
        - ``weights_licence`` (str): the licence of its weights.
        - ``n_atoms_complex``, ``n_atoms_binder``, ``n_atoms_receptor`` (int): atoms
          in the three evaluated systems.
        - ``pocket_cutoff_angstrom`` (float): the cutoff of the pocket.
        - ``reason`` (str): present only when the energy is NaN, and says why.

    Raises:
        ValueError: If a chain ID is empty, the two chain IDs are the same chain or
            the receptor is given twice with different IDs, or ``unit``, ``hetero``,
            ``hydrogens`` or ``backend`` is not an allowed value. Every check runs
            before the ``NotImplementedError``.
        TypeError: If ``pocket`` is neither None nor a :class:`PocketSpec`.
        NotImplementedError: Always, after the checks. The message names the backend
            and the reference.
    """
    receptor = resolve_chain_role("receptor_chain", receptor_chain, "target_chain", target_chain)
    _require_chain_id("binder_chain", binder_chain)
    _require_chain_id("receptor_chain", receptor)
    if binder_chain is not None and binder_chain == receptor:
        raise ValueError(f"binder_chain and receptor_chain are both {binder_chain!r}")
    _require_choice("unit", unit, tuple(_ENERGY_UNIT_PER_EV))
    _require_choice("hetero", hetero, _HETERO_MODES)
    _require_choice("hydrogens", hydrogens, _HYDROGEN_MODES)
    _backend_class(backend)
    if pocket is not None and not isinstance(pocket, PocketSpec):
        raise TypeError(f"pocket must be a PocketSpec or None, got {type(pocket).__name__}")

    # TODO(mlff): pocket cropper with capping, the missing piece. Load the structure with
    #   load_structure, apply the hetero and hydrogens filters (filter_hetero_atoms and
    #   filter_hydrogens in metrics/interface.py), then select the binder atoms plus every
    #   receptor residue with an atom within pocket.cutoff_angstrom of any binder atom, as
    #   whole residues (add waters only when pocket.include_waters). Where a peptide bond is
    #   cut at the pocket boundary, close it with a hydrogen along the cut bond when
    #   pocket.cap == "hydrogen" (a fixed N-H or C-H bond length: TO VERIFY the value and
    #   whether the reference caps with hydrogens or with ACE/NME groups). The three
    #   energies must come from ONE pocket: E(complex) is the whole pocket, E(binder) and
    #   E(receptor) are its two parts at the same coordinates with the same caps, so the
    #   cap atoms of a cut receptor residue belong to the receptor term only. Count the atoms
    #   of each part for n_atoms_complex, n_atoms_binder and n_atoms_receptor.
    #
    # TODO(mlff): energies. Call backend.energy_ev three times (complex, binder, receptor;
    #   the calls are independent, so batch them where the backend allows it), form
    #   E(complex) - E(binder) - E(receptor) in eV and multiply by _ENERGY_UNIT_PER_EV[unit].
    #   On any failure return NaN under mlff_interaction_energy_<unit> with a `reason`.
    #
    # TODO(mlff): charge and spin per part. An MLFF energy needs the total charge of every
    #   evaluated part. Sum the formal charges at the pH implied by pocket.protonation:
    #   ionisable residues (Asp, Glu, Lys, Arg, His by its protonation state, N- and
    #   C-terminus), phospho residues (-2 for a dianionic phosphate; TO VERIFY the state at
    #   the chosen pH), the cap atoms (neutral), and any ligand or ion kept by hetero="keep".
    #   Pass spin 1 (closed shell) unless the input says otherwise. The energy depends on the
    #   charge assignment, and MLFFs trained on OMol25 are reported, by the ChemRxiv paper
    #   and by independent tests, to over-bind. Calibrate and validate against the
    #   implicit-solvent compute_interaction_energy (registered as
    #   structure_interaction_energy) on the bundled complexes in data/ (1YCR, 1CWA, 3P8F,
    #   1XY4, 3V3B) before the values are shown next to E_int.
    #
    # TODO(mlff): solvent. The treatment in the reference protocol (gas phase, an implicit
    #   correction, explicit waters in the pocket) is unknown: the full text of the ChemRxiv
    #   paper has not been read. Read it, decide between a gas-phase energy and an implicit
    #   correction, and state the choice in this docstring and in docs/metrics.md, since
    #   E_int carries generalised Born solvation and the two values are otherwise not
    #   comparable.
    #
    # TODO(mlff): validation. None of the sources validates the method on peptide-protein
    #   complexes, D-amino acids, N-methylated residues, phospho residues or macrocycles
    #   (the reference uses congeneric small-molecule series). Build a small benchmark from
    #   the bundled complexes and public affinity data, check that the D-amino acid and
    #   N-methyl cases give the same energy as their L and NH counterparts up to the expected
    #   difference, and record the outcome in docs/metrics.md before any claim is made.
    get_backend(backend)  # a placeholder raises NotImplementedError here
    raise NotImplementedError(
        f"compute_mlff_interaction_energy has no pocket cropper with capping yet, so backend "
        f"{backend!r} cannot be used. Reference protocol: {_REFERENCE}. "
        "compute_interaction_energy gives the force-field interaction energy."
    )
