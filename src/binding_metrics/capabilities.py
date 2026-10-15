"""What a model or a metric can take as input, and a check of an input against it.

Some steps cannot handle some inputs: a structure-prediction model that reads a cyclic binder as
a linear chain, a metric that only makes sense for a peptide, a metric that needs a receptor
chain. Without a check the run either fails late or returns numbers for the wrong molecule. This
module lets a step declare its limits (``Capabilities``), describes the input once
(``InputProfile``, made by ``profile_input``) and compares the two before anything expensive
starts.

Three rules keep it safe to add to the existing pipeline:

* every field of ``Capabilities`` defaults to "no constraint", so a step that declares nothing is
  never refused;
* a constraint carries one human sentence in ``reasons`` that says why it holds and what to do
  instead, and it must be backed by code or by the documentation of the model it describes;
* an input the check cannot classify (a binder type that cannot be estimated) never blocks: the
  checks that depend on it are skipped and the profile says so.

Importing the module needs the standard library and the two pure residue tables of
``binding_metrics.core``. numpy and biotite are imported when an input is profiled, and OpenMM
is never imported: the ring closures are found on the biotite structure (``detect_closures``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, Mapping, Optional

from binding_metrics.core.nonstandard import D_AA_MAP, NME_AA_MAP
from binding_metrics.core.residues import (
    AMBER_VARIANTS_OUTSIDE_CCD,
    BACKBONE_HEAVY_ATOM_NAMES,
    CYSTEINE_NAMES,
    FORCE_FIELD_CAP_NAMES,
    LACTAM_TEMPLATE_RESIDUES,
    N_METHYLATED_RESIDUES,
    PHOSPHO_RESIDUES,
    STANDARD_AMINO_ACIDS,
    TERMINAL_CAP_NAMES,
    VARIANT_TO_PARENT_RESIDUE,
    WATER_NAMES_ALL,
)

if TYPE_CHECKING:
    from biotite.structure import AtomArray

logger = logging.getLogger(__name__)

__all__ = [
    "BINDER_TYPES",
    "CLOSURE_FAMILIES",
    "MODES",
    "NEEDS",
    "POLICIES",
    "RESIDUE_CLASSES",
    "Capabilities",
    "Closure",
    "ClosureEnd",
    "IncompatibleInputError",
    "InputProfile",
    "PreflightReport",
    "Violation",
    "check_openfold3_residues",
    "check_weights_path",
    "classify_residue",
    "detect_closures",
    "estimate_binder_type",
    "preflight",
    "profile_input",
]

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: The binder types a constraint or a ``--binder-type`` option can name.
BINDER_TYPES: tuple[str, ...] = ("peptide", "miniprotein", "nanobody", "antibody")

#: Ring closures of a binder. ``none`` is a linear binder. ``lactam`` covers the terminal and the
#: side-chain amide bridges, ``staple`` an all-carbon side-chain cross-link, ``other`` any other
#: covalent link between two residues of the binder (a thioether, an ester, a biaryl ether).
CLOSURE_FAMILIES: tuple[str, ...] = (
    "none",
    "head_to_tail",
    "disulfide",
    "lactam",
    "staple",
    "other",
)

#: Classes of residue in a binder, see ``classify_residue``.
RESIDUE_CLASSES: tuple[str, ...] = (
    "canonical",
    "d_amino",
    "n_methyl",
    "phospho",
    "other_ncaa",
    "cap",
    "ligand",
)

#: Things a step can need besides the binder itself. ``receptor_chain`` is read off the profile;
#: the others are facts only the caller knows and are checked against ``provided`` of
#: ``preflight``.
NEEDS: tuple[str, ...] = ("receptor_chain", "reference_structure", "predicted_structure", "gpu")

#: How a structure-prediction model is used for a complex, the vocabulary of
#: ``Capabilities.modes``. ``predict``: from sequences only. ``refold``: the receptor is given as
#: a template and the binder is predicted freely. ``score``: every chain is given its own
#: structure as a template and the relative pose of the chains is NOT given, so the model
#: re-docks them. ``score-lock``: ``score`` with the relative pose of the chains pinned to the
#: input, by a forced template, constraints or both.
MODES: tuple[str, ...] = ("predict", "refold", "score", "score-lock")

_MODE_TEXT = {
    "predict": "predict (sequences only)",
    "refold": "refold (receptor templated, binder predicted freely)",
    "score": "score (every chain templated on its own, the pose not given)",
    "score-lock": "score-lock (score, with the pose of the chains pinned to the input)",
}

_FAMILY_TEXT = {
    "none": "no ring closure (a linear chain)",
    "head_to_tail": "a head-to-tail amide closure",
    "disulfide": "a disulfide bond",
    "lactam": "a lactam bridge",
    "staple": "a hydrocarbon staple",
    "other": "another covalent cross-link",
}
_CLASS_TEXT = {
    "canonical": "canonical amino acids",
    "d_amino": "D-amino acids",
    "n_methyl": "N-methylated residues",
    "phospho": "phosphorylated residues",
    "other_ncaa": "other non-canonical amino acids",
    "cap": "terminal capping groups",
    "ligand": "non-amino-acid groups (ligand, glycan)",
}
_NEED_TEXT = {
    "receptor_chain": "a receptor chain",
    "reference_structure": "a reference structure",
    "predicted_structure": "a predicted structure",
    "gpu": "a GPU",
}
_NEED_FACT = {
    "receptor_chain": "no receptor chain was given and the structure has no other protein chain",
    "reference_structure": "no reference structure was provided",
    "predicted_structure": "no predicted structure was provided",
    "gpu": "no GPU was provided",
}

# Fields of Capabilities that hold a set of vocabulary words, with their vocabulary.
_SET_FIELDS: dict[str, tuple[str, ...]] = {
    "binder_types": BINDER_TYPES,
    "closures": CLOSURE_FAMILIES,
    "residue_classes": RESIDUE_CLASSES,
    "needs": NEEDS,
    "modes": MODES,
}
# Fields whose values a caveat can name (needs is a requirement, not a validation state).
_CAVEAT_FIELDS = ("binder_types", "closures", "residue_classes", "modes")
# Constraints without a vocabulary; their reasons are keyed by the field name alone.
_REASON_KEYS = frozenset({"min_binder_residues", "max_binder_residues", "multi_chain_binder"})


def _join(words: Iterable[str]) -> str:
    words = list(words)
    return ", ".join(words) if words else "none"


def _is_field_value_key(key: str, fields: tuple[str, ...], *, value_required: bool = False) -> bool:
    """True for ``"<field>"`` (unless a value is required) or ``"<field>:<value>"``."""
    name, colon, value = key.partition(":")
    if name not in fields:
        return False
    if not colon:
        return not value_required
    return value in _SET_FIELDS[name]


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One way in which an input breaks the limits of one step.

    Attributes:
        constraint: The field of ``Capabilities`` that was broken (``closures``,
            ``residue_classes``, ``binder_types``, ``min_binder_residues``, ...).
        fact: What the input has, as a sentence.
        requirement: What the step accepts, as a sentence.
        reason: The step's own sentence for the limit (``Capabilities.reasons``), may be empty.
        fix: What to do instead; ``preflight`` fills it in.
        subject: The step, for example ``predictor OpenFold3 0.5.0`` or ``metric 'omega'``;
            ``preflight`` fills it in.
        kind: ``metric`` or ``predictor``, filled in by ``preflight``.
        name: The registry name of the step, filled in by ``preflight``.
    """

    constraint: str
    fact: str
    requirement: str
    reason: str = ""
    fix: str = ""
    subject: str = ""
    kind: str = ""
    name: str = ""

    def format(self) -> str:
        """A short block: the step and constraint, then what was found, required, why, the fix."""
        head = f"{self.subject}: {self.constraint}" if self.subject else self.constraint

        def indented(text: str) -> str:
            return text.replace("\n", "\n" + " " * 14)

        lines = [
            head,
            f"    found:    {indented(self.fact)}",
            f"    requires: {indented(self.requirement)}",
        ]
        if self.reason:
            lines.append(f"    why:      {indented(self.reason)}")
        if self.fix:
            lines.append(f"    fix:      {indented(self.fix)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-ready form, for ``results["preflight"]``."""
        return {
            "kind": self.kind,
            "name": self.name,
            "subject": self.subject,
            "constraint": self.constraint,
            "fact": self.fact,
            "requirement": self.requirement,
            "reason": self.reason,
            "fix": self.fix,
        }


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """The inputs a model or a metric accepts. Every field defaults to "no constraint".

    A set field (``binder_types``, ``closures``, ``residue_classes``) lists what is accepted, an
    empty set accepts anything. An input is refused when it has a member that the set does not
    list: with ``closures={"none", "head_to_tail"}`` a binder with a disulfide is refused, and
    with ``closures={"head_to_tail"}`` a linear binder is refused as well.

    Attributes:
        binder_types: Accepted binder types, a subset of ``BINDER_TYPES``.
        closures: Accepted ring closures, a subset of ``CLOSURE_FAMILIES`` (``none`` stands for a
            linear binder).
        residue_classes: Accepted residue classes in the binder, a subset of ``RESIDUE_CLASSES``.
        min_binder_residues, max_binder_residues: Bounds on the number of amino-acid residues of
            the binder; None leaves a side open.
        multi_chain_binder: False refuses a binder that spans several chains.
        needs: What the step cannot run without, a subset of ``NEEDS``. ``receptor_chain`` is met
            by a receptor given or by any other protein chain in the structure.
        modes: The ways a model can be used that it supports, a subset of ``MODES`` (``predict``,
            ``refold``, ``score``, ``score-lock``; see ``MODES``). Empty declares nothing and
            refuses nothing. A non-empty set that leaves a mode out refuses a request for it and
            needs ``reasons["modes"]`` (or ``"modes:<mode>"``) naming what is not supported and
            why; the full set declares full support and refuses nothing. Partial or unreliable
            support of a listed mode is a caveat, ``caveats["modes:<mode>"]``, which applies to a
            request for that mode whether or not ``modes`` is declared.
        extra_checks: Functions ``profile -> violations`` for a limit that the sets above cannot
            state, such as the residue names a model's query builder can express. Each returns
            ``Violation`` objects with ``fact``, ``requirement`` and a ``reason`` that says why and
            what to do instead. A function reuses the code that enforces the limit at run time
            rather than copying its rules, and imports it when called, so declaring the check
            costs nothing at import.
        reasons: One sentence per constraint saying why it holds and what to do instead. Keys are
            the field name (``"closures"``), or field and value (``"closures:disulfide"``) for a
            sentence about one value; the second form is looked up first. A constraint without a
            ``reasons`` entry under its field name is rejected when the object is built.
        caveats: One sentence per input value that is accepted but was never validated, keyed
            ``"<field>:<value>"`` (fields ``binder_types``, ``closures``, ``residue_classes``).
            ``preflight`` reports it as a warning when the input has that value.
        version: The version of the model or metric the limits were checked against (shown in the
            messages, for example ``"0.5.0"``); empty when not applicable.
    """

    binder_types: frozenset[str] = frozenset()
    closures: frozenset[str] = frozenset()
    residue_classes: frozenset[str] = frozenset()
    min_binder_residues: Optional[int] = None
    max_binder_residues: Optional[int] = None
    multi_chain_binder: bool = True
    needs: frozenset[str] = frozenset()
    modes: frozenset[str] = frozenset()
    extra_checks: tuple[Callable[[InputProfile], Iterable[Violation]], ...] = ()
    # The two mappings are unhashable and long; equality still compares them.
    reasons: Mapping[str, str] = field(default_factory=dict, hash=False, repr=False)
    caveats: Mapping[str, str] = field(default_factory=dict, hash=False, repr=False)
    version: str = ""

    def __post_init__(self):
        for name, vocabulary in _SET_FIELDS.items():
            raw = getattr(self, name)
            if isinstance(raw, str):
                raise ValueError(f"{name} must be a collection of words, not the string {raw!r}")
            values = frozenset(raw)
            unknown = sorted(values - set(vocabulary))
            if unknown:
                raise ValueError(f"{name} has unknown value(s) {unknown}; choose from {vocabulary}")
            object.__setattr__(self, name, values)
        for name in ("min_binder_residues", "max_binder_residues"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}")
        low, high = self.min_binder_residues, self.max_binder_residues
        if low is not None and high is not None and low > high:
            raise ValueError(f"min_binder_residues {low} exceeds max_binder_residues {high}")
        checks = tuple(self.extra_checks)
        if not all(callable(check) for check in checks):
            raise ValueError("extra_checks must be functions taking an InputProfile")
        object.__setattr__(self, "extra_checks", checks)
        reasons, caveats = dict(self.reasons), dict(self.caveats)
        for key in reasons:
            if key not in _REASON_KEYS and not _is_field_value_key(key, tuple(_SET_FIELDS)):
                raise ValueError(f"reasons has an unknown key {key!r}")
        for key in caveats:
            if not _is_field_value_key(key, _CAVEAT_FIELDS, value_required=True):
                raise ValueError(f"caveats has an unknown key {key!r}; use '<field>:<value>'")
        for mapping_name, mapping in (("reasons", reasons), ("caveats", caveats)):
            for key, sentence in mapping.items():
                if not isinstance(sentence, str) or not sentence.strip():
                    raise ValueError(f"{mapping_name}[{key!r}] must be a non-empty sentence")
        for name in self.constrained_fields():
            if name != "extra_checks" and name not in reasons:
                raise ValueError(
                    f"{name} is constrained but reasons has no sentence under {name!r}; say why "
                    "the limit holds and what to do instead"
                )
        object.__setattr__(self, "reasons", MappingProxyType(reasons))
        object.__setattr__(self, "caveats", MappingProxyType(caveats))

    def constrained_fields(self) -> tuple[str, ...]:
        """Names of the fields that differ from "no constraint"."""
        names = [
            name
            for name in _SET_FIELDS
            if getattr(self, name) and not (name == "modes" and set(self.modes) == set(MODES))
        ]
        if self.min_binder_residues is not None:
            names.append("min_binder_residues")
        if self.max_binder_residues is not None:
            names.append("max_binder_residues")
        if not self.multi_chain_binder:
            names.append("multi_chain_binder")
        if self.extra_checks:
            names.append("extra_checks")
        return tuple(names)

    @property
    def is_unconstrained(self) -> bool:
        """True when nothing is constrained (caveats aside)."""
        return not self.constrained_fields()

    def declares_mode(self, mode: str) -> bool:
        """True when ``modes`` lists ``mode``: the model declares that it supports it."""
        return mode in self.modes

    def reason_for(self, name: str, value: Optional[str] = None) -> str:
        """The sentence for a constraint: ``name:value`` if there is one, else ``name``, else ''."""
        if value is not None and f"{name}:{value}" in self.reasons:
            return self.reasons[f"{name}:{value}"]
        return self.reasons.get(name, "")

    def check(
        self,
        profile: InputProfile,
        *,
        provided: Optional[Collection[str]] = None,
        mode: Optional[str] = None,
    ) -> list[Violation]:
        """Every way in which ``profile`` breaks a constraint (an empty list when it fits).

        Args:
            profile: The input, from ``profile_input``.
            provided: What the caller makes available among ``NEEDS`` besides the receptor
                chain (which the profile knows): ``{"reference_structure", "gpu"}``. None means
                the caller does not say, and those needs are not checked.
            mode: The mode the model is asked to run in, one of ``MODES``; None does not say, and
                ``modes`` is not checked.

        The returned violations have no subject yet; ``preflight`` fills it in.
        """
        found: list[Violation] = []

        def add(constraint: str, fact: str, requirement: str, value: Optional[str] = None):
            found.append(
                Violation(
                    constraint=constraint,
                    fact=fact,
                    requirement=requirement,
                    reason=self.reason_for(constraint, value),
                )
            )

        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode is not None and self.modes and mode not in self.modes:
            add(
                "modes",
                f"mode '{mode}' was requested",
                "supported modes: " + _join(_sorted_by(MODES, self.modes)),
                mode,
            )
        if self.binder_types and profile.binder_type != "unknown":
            if profile.binder_type not in self.binder_types:
                origin = "given" if profile.binder_type_source == "given" else "estimated from size"
                add(
                    "binder_types",
                    f"the binder type is {profile.binder_type} ({origin})",
                    f"binder type one of: {_join(sorted(self.binder_types))}",
                    profile.binder_type,
                )
        for family in _sorted_by(CLOSURE_FAMILIES, profile.closures):
            if self.closures and family not in self.closures:
                add(
                    "closures",
                    _closure_fact(profile, family),
                    _closure_requirement(self.closures),
                    family,
                )
        for name in _sorted_by(RESIDUE_CLASSES, profile.residue_classes):
            if self.residue_classes and name not in self.residue_classes:
                add(
                    "residue_classes",
                    f"the binder has {_CLASS_TEXT[name]} "
                    f"({_join(profile.residue_names.get(name, ()))})",
                    "residue classes limited to: "
                    + _join(_sorted_by(RESIDUE_CLASSES, self.residue_classes)),
                    name,
                )
        n_residues = profile.n_binder_residues
        if self.min_binder_residues is not None and n_residues < self.min_binder_residues:
            add(
                "min_binder_residues",
                f"the binder has {n_residues} residues",
                f"at least {self.min_binder_residues} residues",
            )
        if self.max_binder_residues is not None and n_residues > self.max_binder_residues:
            add(
                "max_binder_residues",
                f"the binder has {n_residues} residues",
                f"at most {self.max_binder_residues} residues",
            )
        if not self.multi_chain_binder and len(profile.binder_chains) > 1:
            add(
                "multi_chain_binder",
                f"the binder spans {len(profile.binder_chains)} chains "
                f"({_join(profile.binder_chains)})",
                "a binder made of one chain",
            )
        for need in _sorted_by(NEEDS, self.needs):
            if need == "receptor_chain":
                # The metrics pick the largest other protein chain when none is given
                # (metrics.interface.detect_interface_chains), so one in the file is enough.
                missing = profile.receptor_chain is None and not profile.other_protein_chains
            else:
                missing = provided is not None and need not in provided
            if missing:
                add("needs", _NEED_FACT[need], f"needs {_NEED_TEXT[need]}", need)
        for extra in self.extra_checks:
            for violation in extra(profile):
                if not isinstance(violation, Violation):
                    raise TypeError(
                        f"an extra check returned {type(violation).__name__}, not Violation"
                    )
                found.append(violation)
        return found

    def caveats_for(self, profile: InputProfile, mode: Optional[str] = None) -> list[str]:
        """The caveats that apply to ``profile`` and ``mode``: what the step never validated."""
        present = {
            "binder_types": {profile.binder_type},
            "closures": set(profile.closures),
            "residue_classes": set(profile.residue_classes),
            "modes": {mode} if mode is not None else set(),
        }
        return [
            sentence
            for key, sentence in self.caveats.items()
            if key.partition(":")[2] in present[key.partition(":")[0]]
        ]

    def accepts(
        self,
        profile: InputProfile,
        *,
        provided: Optional[Collection[str]] = None,
        mode: Optional[str] = None,
    ) -> bool:
        """True when ``check`` finds nothing."""
        return not self.check(profile, provided=provided, mode=mode)


def _sorted_by(order: tuple[str, ...], values: Iterable[str]) -> list[str]:
    """``values`` in the order of the vocabulary ``order``."""
    members = set(values)
    return [word for word in order if word in members]


def _closure_fact(profile: InputProfile, family: str) -> str:
    if family == "none":
        return "the binder is linear (no ring closure found)"
    links = "; ".join(f"{c.end1} - {c.end2}" for c in profile.closure_bonds if c.family == family)
    return f"the binder has {_FAMILY_TEXT[family]} ({links})"


def _closure_requirement(allowed: frozenset[str]) -> str:
    words = _sorted_by(CLOSURE_FAMILIES, allowed)
    if "none" not in allowed:
        return f"needs a cyclic binder with a closure among: {_join(words)}"
    return f"closures limited to: {_join(words)}"


# ---------------------------------------------------------------------------
# Residue classes
# ---------------------------------------------------------------------------

_CANONICAL_NAMES = (
    STANDARD_AMINO_ACIDS | frozenset(VARIANT_TO_PARENT_RESIDUE) | LACTAM_TEMPLATE_RESIDUES
)
_N_METHYL_NAMES = frozenset(NME_AA_MAP) | N_METHYLATED_RESIDUES
_CAP_NAMES = TERMINAL_CAP_NAMES | FORCE_FIELD_CAP_NAMES
# Amino acids whose name the CCD-based peptide test may not list.
_KNOWN_AMINO_ACID_NAMES = (
    _CANONICAL_NAMES
    | AMBER_VARIANTS_OUTSIDE_CCD
    | frozenset(D_AA_MAP)
    | _N_METHYL_NAMES
    | PHOSPHO_RESIDUES
)


def classify_residue(res_name: str, *, is_amino_acid: bool = True) -> Optional[str]:
    """The class of a residue of the binder, one of ``RESIDUE_CLASSES``; None for a water.

    ``canonical`` covers the 20 amino acids, their AMBER and CHARMM protonation variants (HID,
    HIE, HIP, CYX, ...) and the lactam template names; ``d_amino`` the D-amino-acid codes of
    ``core.nonstandard.D_AA_MAP``; ``n_methyl`` the N-methylated codes of ``NME_AA_MAP`` and the
    template names; ``phospho`` SEP, TPO and PTR; ``cap`` ACE, NME, FOR and NH2. Any other amino
    acid (BMT, ABA, MSE, ...) is ``other_ncaa`` and anything that is not an amino acid ``ligand``.

    Args:
        res_name: Residue name (three-letter code).
        is_amino_acid: Whether the residue is an amino acid (the CCD lists it as peptide-linking
            or it has the N, CA and C backbone atoms); decides between ``other_ncaa`` and
            ``ligand`` for a name this module does not list.
    """
    name = res_name.strip().upper()
    if name in WATER_NAMES_ALL:
        return None
    if name in _CANONICAL_NAMES:
        return "canonical"
    if name in D_AA_MAP:
        return "d_amino"
    if name in _N_METHYL_NAMES:
        return "n_methyl"
    if name in PHOSPHO_RESIDUES:
        return "phospho"
    if name in _CAP_NAMES:
        return "cap"
    return "other_ncaa" if is_amino_acid else "ligand"


# ---------------------------------------------------------------------------
# Ring closures of a binder (biotite, no OpenMM)
# ---------------------------------------------------------------------------

# Cut-offs in angstrom. They equal the nanometre values of ``core.cyclic`` (``_AMIDE_BOND_THRESH``
# 0.20 nm and ``_DISULFIDE_THRESH`` 0.26 nm; a test checks the equality), so that this detector and
# ``core.cyclic.detect_cyclization`` agree on the same coordinates. Amide C-N is 1.33 A (Engh and
# Huber, Acta Cryst. A47, 392, 1991) and a disulfide S-S 2.03 A; the margin absorbs the stretched
# closure bonds of predicted models.
_AMIDE_BOND_THRESHOLD_ANGSTROM = 2.0
_DISULFIDE_THRESHOLD_ANGSTROM = 2.6

# Names that hold the sulfur of a disulfide, the same rule as ``core.cyclic``: CYS, its AMBER
# disulfide form CYX, and the D-cysteine codes of D_AA_MAP (a test compares the two sets).
_DISULFIDE_RESIDUE_NAMES = CYSTEINE_NAMES | frozenset(
    name for name, parent in D_AA_MAP.items() if parent == "CYS"
)
# The atoms that make a residue a peptide-chain member even when no table knows its name.
_BACKBONE_CORE = frozenset({"N", "CA", "C"})
# Backbone atoms; a cross-link that touches one is not a side-chain staple.
_BACKBONE_ATOM_NAMES = BACKBONE_HEAVY_ATOM_NAMES | {"OXT"}

_FAMILY_OF_KIND = {
    "head_to_tail": "head_to_tail",
    "disulfide": "disulfide",
    "lactam_n_asp": "lactam",
    "lactam_n_glu": "lactam",
    "lactam_c_lys": "lactam",
    "lactam_sc_lys_asp": "lactam",
    "lactam_sc_lys_glu": "lactam",
    "hydrocarbon_staple": "staple",
    "unsupported_crosslink": "other",
}

# (acid residue, closure atom, kind) of the lactams that close on the N-terminus.
_N_TERMINAL_LACTAMS = (("ASP", "CG", "lactam_n_asp"), ("GLU", "CD", "lactam_n_glu"))
# (acid residue, closure atom, kind) of the lactams between a Lys side chain and an acid side chain.
_SIDE_CHAIN_LACTAMS = (("ASP", "CG", "lactam_sc_lys_asp"), ("GLU", "CD", "lactam_sc_lys_glu"))


@dataclass(frozen=True)
class ClosureEnd:
    """One end of a closure bond.

    Attributes:
        residue_index: 0-based position among the amino-acid residues of the chain (the index
            that ``core.cyclic.CyclicBondInfo.atom1_id`` carries).
        residue_name: Three-letter code.
        residue_number: The residue number of the file.
        atom_name: Atom name.
    """

    residue_index: int
    residue_name: str
    residue_number: int
    atom_name: str

    def __str__(self) -> str:
        # A space keeps a name that ends in a digit (MK8, 0EH) apart from the number.
        return f"{self.residue_name} {self.residue_number}.{self.atom_name}"


@dataclass(frozen=True)
class Closure:
    """A covalent link that closes a ring inside the binder.

    Attributes:
        kind: The type as ``core.cyclic`` names it (``head_to_tail``, ``disulfide``,
            ``lactam_n_asp``, ``lactam_n_glu``, ``lactam_c_lys``, ``lactam_sc_lys_asp``,
            ``lactam_sc_lys_glu``, ``hydrocarbon_staple``), or ``unsupported_crosslink`` for a link
            between two non-adjacent residues that fits none of them (a thioether, an ester).
        family: One of ``CLOSURE_FAMILIES`` (never ``none``).
        end1, end2: The two atoms.
    """

    kind: str
    family: str
    end1: ClosureEnd
    end2: ClosureEnd

    def describe(self) -> str:
        return f"{self.kind.replace('_', ' ')} {self.end1} - {self.end2}"


@dataclass
class _Residue:
    """One residue of the chain as ``detect_closures`` reads it."""

    index: int  # position among the amino-acid residues (-1 for a group that is not one)
    name: str
    number: int
    label: str  # "NAME number" plus the insertion code, for messages
    start: int  # atom range [start, stop) in the chain's atom array
    stop: int
    is_amino_acid: bool
    atoms: dict[str, int]  # first atom of each name


@dataclass
class _ChainView:
    atoms: Any  # the chain's atoms, a biotite AtomArray
    groups: list[_Residue]  # every residue group of the chain, waters and ions included
    amino_acids: list[_Residue]  # the amino-acid residues, in file order


def _amino_acid_mask(atoms) -> Any:
    """Boolean mask of the atoms that belong to an amino-acid residue."""
    import biotite.structure as struc
    import numpy as np

    return struc.filter_amino_acids(atoms) | np.isin(
        atoms.res_name, sorted(_KNOWN_AMINO_ACID_NAMES)
    )


def _chain_view(atoms, chain_id: str) -> _ChainView:
    """Split the atoms of ``chain_id`` into residues and mark the amino-acid ones.

    A chain ID can hold waters and ions after the polymer (author chain IDs do); they are kept as
    groups but never counted as residues of the binder.

    Raises:
        ValueError: The structure has no chain of that ID.
    """
    import biotite.structure as struc

    chain_atoms = atoms[atoms.chain_id == chain_id]
    if chain_atoms.array_length() == 0:
        available = sorted(set(map(str, atoms.chain_id)))
        raise ValueError(f"chain {chain_id!r} not found in the structure; chains: {available}")
    amino = _amino_acid_mask(chain_atoms)
    starts = list(struc.get_residue_starts(chain_atoms)) + [chain_atoms.array_length()]
    groups: list[_Residue] = []
    amino_acids: list[_Residue] = []
    for start, stop in zip(starts[:-1], starts[1:]):
        # A residue is an amino acid when the CCD or the name tables say so, or when it has the
        # N, CA and C atoms of a backbone (a custom residue name that no table lists).
        is_amino_acid = bool(amino[start]) or _BACKBONE_CORE <= set(
            chain_atoms.atom_name[start:stop]
        )
        group = _Residue(
            index=len(amino_acids) if is_amino_acid else -1,
            name=str(chain_atoms.res_name[start]),
            number=int(chain_atoms.res_id[start]),
            label=(
                f"{chain_atoms.res_name[start]} {int(chain_atoms.res_id[start])}"
                f"{str(chain_atoms.ins_code[start]).strip()}"
            ),
            start=int(start),
            stop=int(stop),
            is_amino_acid=is_amino_acid,
            atoms={},
        )
        if is_amino_acid:
            for offset, atom_name in enumerate(chain_atoms.atom_name[start:stop]):
                group.atoms.setdefault(str(atom_name), int(start) + offset)
            amino_acids.append(group)
        groups.append(group)
    return _ChainView(chain_atoms, groups, amino_acids)


def _bonded_pairs(atoms) -> tuple[set, list]:
    """The atom-index pairs the structure declares as bonded, and the bond rows.

    Coordination bonds (metal sites) are not covalent and are left out. Both are empty when the
    structure has no bond table.
    """
    if atoms.bonds is None:
        return set(), []
    import biotite.structure as struc

    rows = [
        (int(i), int(j))
        for i, j, bond_type in atoms.bonds.as_array()
        if bond_type != struc.BondType.COORDINATION
    ]
    return {frozenset(pair) for pair in rows}, rows


def _closures_of(view: _ChainView, notes: Optional[list[str]] = None) -> list[Closure]:
    """The closures of one chain; the logic follows ``core.cyclic.detect_cyclization``."""
    import numpy as np

    residues = view.amino_acids
    if len(residues) < 2:
        return []
    notes = notes if notes is not None else []
    coord = view.atoms.coord
    declared, bond_rows = _bonded_pairs(view.atoms)

    def linked(i: int, j: int, threshold: float) -> bool:
        """Bonded in the file, or closer than ``threshold`` (strained models stay detected)."""
        return frozenset((i, j)) in declared or (
            float(np.linalg.norm(coord[i] - coord[j])) < threshold
        )

    def end(residue: _Residue, atom_name: str) -> ClosureEnd:
        return ClosureEnd(residue.index, residue.name, residue.number, atom_name)

    found: list[Closure] = []
    detected: set = set()  # atom-index pairs claimed by a named pattern

    def record(kind: str, res1: _Residue, atom1: str, res2: _Residue, atom2: str) -> None:
        found.append(Closure(kind, _FAMILY_OF_KIND[kind], end(res1, atom1), end(res2, atom2)))
        detected.add(frozenset((res1.atoms[atom1], res2.atoms[atom2])))

    first, last = residues[0], residues[-1]
    n_first, c_last = first.atoms.get("N"), last.atoms.get("C")
    if n_first is None:
        notes.append(
            f"the first residue {first.name}{first.number} has no atom N, so a head-to-tail or "
            "N-terminal lactam closure could not be tested"
        )
    if c_last is None:
        notes.append(
            f"the last residue {last.name}{last.number} has no atom C, so a head-to-tail or "
            "C-terminal lactam closure could not be tested"
        )

    if n_first is not None and c_last is not None:
        if linked(n_first, c_last, _AMIDE_BOND_THRESHOLD_ANGSTROM):
            record("head_to_tail", last, "C", first, "N")

    cysteines = [r for r in residues if r.name in _DISULFIDE_RESIDUE_NAMES]
    for k, res_i in enumerate(cysteines):
        for res_j in cysteines[k + 1 :]:
            sg_i, sg_j = res_i.atoms.get("SG"), res_j.atoms.get("SG")
            if sg_i is not None and sg_j is not None:
                if linked(sg_i, sg_j, _DISULFIDE_THRESHOLD_ANGSTROM):
                    record("disulfide", res_i, "SG", res_j, "SG")

    if n_first is not None:
        for acid, atom_name, kind in _N_TERMINAL_LACTAMS:
            for res in residues:
                partner = res.atoms.get(atom_name) if res.name == acid else None
                if partner is not None and linked(partner, n_first, _AMIDE_BOND_THRESHOLD_ANGSTROM):
                    record(kind, res, atom_name, first, "N")

    if c_last is not None:
        for res in residues:
            nz = res.atoms.get("NZ") if res.name == "LYS" else None
            if nz is not None and linked(nz, c_last, _AMIDE_BOND_THRESHOLD_ANGSTROM):
                record("lactam_c_lys", last, "C", res, "NZ")

    for lysine in (r for r in residues if r.name == "LYS"):
        nz = lysine.atoms.get("NZ")
        if nz is None:
            continue
        for acid, atom_name, kind in _SIDE_CHAIN_LACTAMS:
            for res in residues:
                partner = res.atoms.get(atom_name) if res.name == acid else None
                if res is lysine or partner is None:
                    continue
                if frozenset((nz, partner)) in detected:
                    continue  # already claimed by a terminal lactam
                if linked(nz, partner, _AMIDE_BOND_THRESHOLD_ANGSTROM):
                    record(kind, lysine, "NZ", res, atom_name)

    # Links the file declares between residues that are not neighbours and that no pattern above
    # named: an all-carbon side-chain link is a hydrocarbon staple, anything else is reported as
    # an unsupported cross-link (core.cyclic raises CyclizationError for it).
    residue_of_atom = np.full(view.atoms.array_length(), -1, dtype=int)
    for res in residues:
        residue_of_atom[res.start : res.stop] = res.index
    for i, j in bond_rows:
        index_i, index_j = int(residue_of_atom[i]), int(residue_of_atom[j])
        if index_i < 0 or index_j < 0 or abs(index_i - index_j) <= 1:
            continue
        if frozenset((i, j)) in detected:
            continue
        res_i, res_j = residues[index_i], residues[index_j]
        name_i, name_j = str(view.atoms.atom_name[i]), str(view.atoms.atom_name[j])
        carbon_side_chains = all(
            str(view.atoms.element[atom]).upper() == "C" and name not in _BACKBONE_ATOM_NAMES
            for atom, name in ((i, name_i), (j, name_j))
        )
        kind = "hydrocarbon_staple" if carbon_side_chains else "unsupported_crosslink"
        found.append(Closure(kind, _FAMILY_OF_KIND[kind], end(res_i, name_i), end(res_j, name_j)))
        detected.add(frozenset((i, j)))
    return found


def detect_closures(atoms: AtomArray, chain_id: str) -> list[Closure]:
    """The covalent ring closures of a chain, found on a biotite structure without OpenMM.

    It is the light counterpart of ``core.cyclic.detect_cyclization`` and finds the same
    closures on the same coordinates: head-to-tail amide (C of the last residue to N of the
    first), disulfide (SG to SG), the lactams (Asp CG or Glu CD to the N-terminus, Lys NZ to the
    C-terminus, Lys NZ to an Asp or Glu side chain), and, from the bond table of the structure,
    hydrocarbon staples and other cross-links. A pair counts as linked when the bond table lists
    it or the atoms are closer than 2.0 A (2.6 A for SG-SG), so a strained model that has no
    bond record is still found. A test compares the two detectors on the bundled examples.

    It differs on one point: only the amino-acid residues of the chain are looked at, so the
    waters and the ligands that share the chain ID of an author-numbered file cannot shift the
    first or last residue. Both detectors take a disulfide between cysteines named CYS, CYX or
    DCY.

    Args:
        atoms: A biotite ``AtomArray``. With a bond table (``include_bonds=True`` when read) the
            staples and other cross-links are found; without one only the distance patterns are.
        chain_id: The chain to examine.

    Returns:
        One ``Closure`` per link, empty for a linear chain.

    Raises:
        ValueError: The structure has no chain of that ID.
    """
    return _closures_of(_chain_view(atoms, chain_id))


# ---------------------------------------------------------------------------
# The input profile
# ---------------------------------------------------------------------------

#: A binder of at most this many residues is called a peptide. The line is a length convention,
#: the one the US FDA uses to separate peptides from proteins (21 CFR 600.3(h)(6): a protein is
#: an amino-acid polymer of more than 40 residues), not a physical boundary.
PEPTIDE_MAX_RESIDUES = 40
#: A binder of more than ``PEPTIDE_MAX_RESIDUES`` and at most this many residues is called a
#: miniprotein. 100 is a convention of this package: it keeps miniproteins apart from the single
#: domain antibodies (nanobodies, about 110 to 130 residues), which size alone cannot tell from
#: any other domain of that length.
MINIPROTEIN_MAX_RESIDUES = 100


def estimate_binder_type(n_residues: int) -> str:
    """The binder type a size class suggests: peptide, miniprotein, or ``unknown``.

    Up to ``PEPTIDE_MAX_RESIDUES`` residues is a ``peptide``, up to ``MINIPROTEIN_MAX_RESIDUES``
    a ``miniprotein``. A longer chain could be a nanobody, an antibody chain or any other
    protein, and size cannot tell them apart, so the answer is ``unknown``. A nanobody or an
    antibody is only ever set by the caller. ``unknown`` never refuses an input: the checks that
    depend on the binder type are skipped and the report says so.
    """
    if n_residues <= PEPTIDE_MAX_RESIDUES:
        return "peptide"
    if n_residues <= MINIPROTEIN_MAX_RESIDUES:
        return "miniprotein"
    return "unknown"


@dataclass(frozen=True)
class InputProfile:
    """What a sample looks like, computed once by ``profile_input`` and read by ``preflight``.

    Every field but ``binder_chains`` has a default, so a test can build a profile by hand.

    Attributes:
        binder_chains: Chain IDs of the binder (one for the usual case).
        receptor_chain: Chain ID of the receptor, None when none was given.
        n_binder_residues: Amino-acid residues in the binder chains (caps, ligands, waters and
            ions are not counted).
        binder_type: ``peptide``, ``miniprotein``, ``nanobody``, ``antibody`` or ``unknown``.
        binder_type_source: ``given`` when the caller named the type, ``estimated`` when it comes
            from the size class (see ``estimate_binder_type``).
        closures: The closure families present (``none`` alone for a linear binder).
        closure_bonds: The links behind ``closures``.
        residue_classes: The classes of residue present in the binder (see ``classify_residue``).
        residue_names: For each class present, the distinct residue names, sorted.
        residue_labels: For each class present, the residues in chain order as ``"NAME number"``
            (the insertion code follows the number), for messages that point at a residue.
        chain_ids: Every chain ID of the structure.
        other_protein_chains: The protein chains that are not the binder, in file order: the
            receptor that a metric can find by itself when none is given.
        notes: Things that make the profile less certain (an atom missing, no bond table, a binder
            type that could not be estimated).
    """

    binder_chains: tuple[str, ...]
    receptor_chain: Optional[str] = None
    n_binder_residues: int = 0
    binder_type: str = "unknown"
    binder_type_source: str = "estimated"
    closures: frozenset[str] = frozenset({"none"})
    closure_bonds: tuple[Closure, ...] = ()
    residue_classes: frozenset[str] = frozenset({"canonical"})
    residue_names: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict, hash=False, repr=False
    )
    chain_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    residue_labels: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict, hash=False, repr=False
    )
    other_protein_chains: tuple[str, ...] = ()

    def __post_init__(self):
        chains = (
            (self.binder_chains,) if isinstance(self.binder_chains, str) else self.binder_chains
        )
        object.__setattr__(self, "binder_chains", tuple(chains))
        object.__setattr__(self, "closures", frozenset(self.closures))
        object.__setattr__(self, "residue_classes", frozenset(self.residue_classes))
        object.__setattr__(
            self,
            "residue_names",
            MappingProxyType({k: tuple(v) for k, v in dict(self.residue_names).items()}),
        )
        object.__setattr__(
            self,
            "residue_labels",
            MappingProxyType({k: tuple(v) for k, v in dict(self.residue_labels).items()}),
        )
        object.__setattr__(self, "other_protein_chains", tuple(self.other_protein_chains))
        if self.binder_type != "unknown" and self.binder_type not in BINDER_TYPES:
            raise ValueError(f"binder_type {self.binder_type!r} is not one of {BINDER_TYPES}")

    @property
    def binder_chain(self) -> str:
        """The first binder chain."""
        return self.binder_chains[0]

    @property
    def n_chains(self) -> int:
        return len(self.chain_ids)

    def describe(self) -> str:
        """One line for messages: chain, size, type, closures and residue classes."""
        origin = "given" if self.binder_type_source == "given" else "estimated from size"
        classes = _join(_sorted_by(RESIDUE_CLASSES, self.residue_classes))
        parts = [
            f"binder chain {_join(self.binder_chains)}: {self.n_binder_residues} residues",
            f"type {self.binder_type} ({origin})",
            f"closures {_join(_sorted_by(CLOSURE_FAMILIES, self.closures))}",
            f"residue classes {classes}",
        ]
        receptor = self.receptor_chain if self.receptor_chain is not None else "not given"
        return "; ".join(parts) + f"; receptor chain {receptor}"

    def to_dict(self) -> dict:
        """JSON-ready form, for ``results["preflight"]``."""
        return {
            "binder_chains": list(self.binder_chains),
            "receptor_chain": self.receptor_chain,
            "n_binder_residues": self.n_binder_residues,
            "binder_type": self.binder_type,
            "binder_type_source": self.binder_type_source,
            "closures": _sorted_by(CLOSURE_FAMILIES, self.closures),
            "closure_bonds": [c.describe() for c in self.closure_bonds],
            "residue_classes": _sorted_by(RESIDUE_CLASSES, self.residue_classes),
            "residue_names": {k: list(v) for k, v in self.residue_names.items()},
            "residue_labels": {k: list(v) for k, v in self.residue_labels.items()},
            "chain_ids": list(self.chain_ids),
            "other_protein_chains": list(self.other_protein_chains),
            "notes": list(self.notes),
        }


def _read_atoms(structure) -> Any:
    """A biotite AtomArray with its bond table from a path, an AtomArray or an AtomArrayStack.

    A path ends in ``.cif`` or ``.mmcif`` (read as mmCIF) or anything else (read as PDB), and may
    carry a ``.gz`` after that (``.cif.gz``, ``.mmcif.gz``, ``.pdb.gz``), which is decompressed;
    these are the files that gemmi, which the OpenFold3 query builders use, reads as well.
    """
    if isinstance(structure, (str, os.PathLike)):
        import gzip

        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx

        from binding_metrics.utils import backfill_auth_columns

        path = Path(structure)
        if not path.is_file():
            raise FileNotFoundError(f"Structure file not found: {path}")
        suffixes = [suffix.lower() for suffix in path.suffixes]
        gzipped = suffixes[-1:] == [".gz"]
        extension = suffixes[-2] if gzipped and len(suffixes) > 1 else (suffixes or [""])[-1]
        # a plain file is opened by biotite as before; a gzip is decompressed into a text stream
        stream = gzip.open(path, "rt", encoding="utf-8", errors="replace") if gzipped else None
        try:
            source = stream if gzipped else str(path)
            if extension in (".cif", ".mmcif"):
                cif = pdbx.CIFFile.read(source)
                backfill_auth_columns(cif)
                return pdbx.get_structure(cif, model=1, include_bonds=True)
            return pdb_io.get_structure(pdb_io.PDBFile.read(source), model=1, include_bonds=True)
        finally:
            if stream is not None:
                stream.close()
    if hasattr(structure, "stack_depth"):  # AtomArrayStack: the first model
        return structure[0]
    return structure


def profile_input(
    structure,
    binder_chain,
    receptor_chain: Optional[str] = None,
    binder_type: str = "auto",
) -> InputProfile:
    """Describe a sample once: size, type, ring closures and residue classes of the binder.

    Args:
        structure: A PDB or mmCIF path (first model, author chain IDs, bonds from CONECT or
            ``struct_conn``), or a biotite ``AtomArray`` / ``AtomArrayStack`` (first model).
        binder_chain: Chain ID of the binder; a list or tuple of IDs for a binder that spans
            several chains.
        receptor_chain: Chain ID of the receptor, None when there is none or it is not known.
            A step that needs a receptor is satisfied by any other protein chain of the structure
            (``InputProfile.other_protein_chains``), because the metrics pick one when none is
            given.
        binder_type: ``auto`` estimates the type from the number of residues (see
            ``estimate_binder_type``: at most 40 residues a peptide, at most 100 a miniprotein,
            longer ``unknown``); or ``peptide``, ``miniprotein``, ``nanobody``, ``antibody``.
            ``unknown`` never refuses an input; the checks that depend on the type are skipped.

    Returns:
        The ``InputProfile``. Closures come from ``detect_closures`` on each binder chain.

    Raises:
        ValueError: ``binder_type`` is not one of the choices, or a chain ID is not in the
            structure.
        FileNotFoundError: ``structure`` is a path that does not exist.
    """
    if binder_type != "auto" and binder_type not in BINDER_TYPES:
        raise ValueError(
            f"binder_type must be 'auto' or one of {BINDER_TYPES}, got {binder_type!r}"
        )
    binder_chains = (binder_chain,) if isinstance(binder_chain, str) else tuple(binder_chain)
    if not binder_chains or len(set(binder_chains)) != len(binder_chains):
        raise ValueError(
            f"binder_chain must name one or more distinct chains, got {binder_chain!r}"
        )

    atoms = _read_atoms(structure)
    chain_ids = tuple(dict.fromkeys(map(str, atoms.chain_id)))
    if receptor_chain is not None and receptor_chain not in chain_ids:
        raise ValueError(f"receptor chain {receptor_chain!r} not found; chains: {list(chain_ids)}")
    notes: list[str] = []
    if atoms.bonds is None:
        notes.append(
            "the structure has no bond table, so hydrocarbon staples and other side-chain "
            "cross-links were not looked for"
        )

    closures: list[Closure] = []
    names_by_class: dict[str, set[str]] = {}
    labels_by_class: dict[str, list[str]] = {}
    n_residues = 0
    for chain in binder_chains:
        view = _chain_view(atoms, chain)
        closures.extend(_closures_of(view, notes))
        for group in view.groups:
            if not group.is_amino_acid and group.stop - group.start == 1:
                continue  # an ion or a single-atom heterogen
            residue_class = classify_residue(group.name, is_amino_acid=group.is_amino_acid)
            if residue_class is not None:
                names_by_class.setdefault(residue_class, set()).add(group.name)
                labels_by_class.setdefault(residue_class, []).append(group.label)
        n_residues += len(view.amino_acids)

    if binder_type == "auto":
        resolved, source = estimate_binder_type(n_residues), "estimated"
        if resolved == "unknown":
            notes.append(
                f"the binder type is unknown: {n_residues} residues is too long to tell a "
                "miniprotein, a nanobody or an antibody chain apart by size, so the checks that "
                "depend on the binder type are skipped (pass binder_type to enable them)"
            )
    else:
        resolved, source = binder_type, "given"

    protein_chain_ids = set(map(str, atoms.chain_id[_amino_acid_mask(atoms)]))
    families = frozenset(c.family for c in closures) or frozenset({"none"})
    return InputProfile(
        binder_chains=binder_chains,
        receptor_chain=receptor_chain,
        n_binder_residues=n_residues,
        binder_type=resolved,
        binder_type_source=source,
        closures=families,
        closure_bonds=tuple(closures),
        residue_classes=frozenset(names_by_class),
        residue_names={k: sorted(v) for k, v in names_by_class.items()},
        chain_ids=chain_ids,
        notes=tuple(dict.fromkeys(notes)),
        residue_labels=labels_by_class,
        other_protein_chains=tuple(
            c for c in chain_ids if c in protein_chain_ids and c not in binder_chains
        ),
    )


# ---------------------------------------------------------------------------
# Checks that reuse the rule of the code they guard
# ---------------------------------------------------------------------------

#: The classes whose residues are amino acids, that is, the ones an OpenFold3 query names.
_AMINO_ACID_CLASSES = ("canonical", "d_amino", "n_methyl", "phospho", "other_ncaa")


def check_openfold3_residues(profile: InputProfile) -> list[Violation]:
    """The binder residues that the OpenFold3 query builder cannot express.

    It asks the builder itself: ``metrics._openfold_run._residue_letter_and_ccd`` maps a residue
    name to a one-letter code and, for a D-amino acid or a modified residue, to the Chemical
    Component Dictionary code written into ``non_canonical_residues``. A name it cannot map is
    what ``_extract_query_chain`` reports as an ``UnmappableResidueError``, and the ``reason`` of
    the violation is the text of that error, so a run that gets past the check and one that does
    not say the same thing. Caps, ligands, waters and ions are left out of the query by the
    builder and are not checked.

    The builder is imported when the check runs, which needs biotite (or gemmi) for the CCD
    lookup. This is the ``extra_checks`` entry of the OpenFold3 adapter.

    Args:
        profile: The input; ``residue_labels`` gives the residues, ``residue_names`` is used when
            a hand-built profile has no labels.

    Returns:
        At most one ``Violation``, listing every residue that cannot be expressed.
    """
    from binding_metrics.metrics._openfold_run import (
        UnmappableResidueError,
        _residue_letter_and_ccd,
    )

    mappable: dict[str, bool] = {}
    unmappable: list[str] = []
    for class_name in _AMINO_ACID_CLASSES:
        labels = profile.residue_labels.get(class_name)
        if labels:
            residues = [(label, label.rsplit(" ", 1)[0]) for label in labels]
        else:
            residues = [(name, name) for name in profile.residue_names.get(class_name, ())]
        for label, residue_name in residues:
            if residue_name not in mappable:
                mappable[residue_name] = _residue_letter_and_ccd(residue_name) is not None
            if not mappable[residue_name]:
                unmappable.append(label)
    if not unmappable:
        return []
    chains = ", ".join(profile.binder_chains)
    error = UnmappableResidueError([("", chains, unmappable)])
    return [
        Violation(
            constraint="residue_classes",
            fact=f"the binder has residues that OpenFold3 cannot take: {_join(unmappable)}",
            requirement=(
                "amino acids that are one of the 20 standard residues, X, or a peptide-linking "
                "Chemical Component Dictionary code"
            ),
            reason=str(error),
        )
    ]


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

#: What ``preflight`` does with an incompatibility: raise, leave the step out, or log and go on.
POLICIES: tuple[str, ...] = ("error", "skip", "warn")


@dataclass(frozen=True)
class PreflightReport:
    """The outcome of ``preflight``: what was found and what will run.

    Attributes:
        policy: ``error``, ``skip`` or ``warn``.
        profile: The input that was checked.
        violations: Every incompatibility found, for every metric and predictor.
        warnings: Accepted inputs a step never validated (``Capabilities.caveats``).
        notes: Checks that could not be made (a binder type that is unknown, needs the caller did
            not describe) and the notes of the profile.
        metrics_requested: The metric names, in the order given.
        metrics_to_run: The metrics that go on: all of them, except under ``skip`` where a metric
            with a violation is left out.
        predictors: The predictors that were checked, as written in the messages.
        predictor_usable: False when the policy of the predictor is ``skip`` and it has a
            violation. The caller then leaves out the predictor and what depends on it:
            ``preflight`` does not know which metrics read a prediction.
        predictor_policy: The policy applied to the predictors when it differs from ``policy``;
            empty when it is the same.
        mode: The mode the predictors were asked to run in, one of ``MODES``; None when the
            caller did not say (an output whose making is not known).
    """

    policy: str
    profile: InputProfile
    violations: tuple[Violation, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metrics_requested: tuple[str, ...] = ()
    metrics_to_run: tuple[str, ...] = ()
    predictors: tuple[str, ...] = ()
    predictor_usable: bool = True
    predictor_policy: str = ""
    mode: Optional[str] = None

    def policy_of(self, kind: str) -> str:
        """The policy that applies to a violation of ``kind`` (``metric`` or ``predictor``)."""
        return (self.predictor_policy or self.policy) if kind == "predictor" else self.policy

    @property
    def compatible(self) -> bool:
        """True when nothing was found."""
        return not self.violations

    @property
    def refused(self) -> bool:
        """True when a violation falls under the policy ``error``: ``preflight`` raised for it."""
        return any(self.policy_of(v.kind) == "error" for v in self.violations)

    @property
    def skipped_metrics(self) -> tuple[str, ...]:
        """The requested metrics that will not run."""
        kept = set(self.metrics_to_run)
        return tuple(name for name in self.metrics_requested if name not in kept)

    def format(self) -> str:
        """The plan as text: the input, what runs, every problem with its fix, warnings, notes."""
        n = len(self.violations)
        policy = self.policy
        if self.predictor_policy and self.predictor_policy != self.policy:
            policy = f"{self.policy}, predictor: {self.predictor_policy}"
        if not n:
            heading = f"Pre-flight check: the input fits everything requested (policy: {policy})."
        elif self.refused:
            heading = (
                f"Pre-flight check failed: {n} incompatibilit{'y' if n == 1 else 'ies'} between "
                f"the input and what was requested (policy: {policy})."
            )
        else:
            heading = (
                f"Pre-flight check found {n} incompatibilit{'y' if n == 1 else 'ies'} "
                f"(policy: {policy})."
            )
        lines = [heading, f"Input: {self.profile.describe()}"]
        if self.mode is not None:
            lines.append(f"Mode: {_MODE_TEXT[self.mode]}")
        if not self.refused:
            runs = [f"metrics {_join(self.metrics_to_run)}"] if self.metrics_requested else []
            runs += [f"predictor {label}" for label in self.predictors if self.predictor_usable]
            lines.append(f"Runs: {'; '.join(runs) if runs else 'nothing was requested'}")
            if self.skipped_metrics:
                lines.append(f"Left out: metrics {_join(self.skipped_metrics)}")
            if not self.predictor_usable:
                lines.append(f"Left out: predictor {_join(self.predictors)}")
        subjects_shown: set[str] = set()
        for violation in self.violations:
            # the fix belongs to the step, so it is printed once for all its violations
            repeated = violation.subject in subjects_shown
            subjects_shown.add(violation.subject)
            lines += ["", (replace(violation, fix="") if repeated else violation).format()]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  - {text}" for text in self.warnings]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {text}" for text in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-ready form, for ``results["preflight"]``."""
        return {
            "policy": self.policy,
            "predictor_policy": self.predictor_policy or self.policy,
            "mode": self.mode,
            "compatible": self.compatible,
            "profile": self.profile.to_dict(),
            "metrics_requested": list(self.metrics_requested),
            "metrics_to_run": list(self.metrics_to_run),
            "metrics_skipped": [
                {
                    "name": name,
                    "reasons": [
                        v.fact for v in self.violations if v.kind == "metric" and v.name == name
                    ],
                }
                for name in self.skipped_metrics
            ],
            "predictors": list(self.predictors),
            "predictor_usable": self.predictor_usable,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


class IncompatibleInputError(ValueError):
    """The input cannot go through a requested metric or predictor (policy ``error``).

    The message lists every incompatibility with the fact found, the requirement and a fix.

    Attributes:
        report: The ``PreflightReport``; ``report.violations`` holds the details.
    """

    def __init__(self, report: PreflightReport):
        self.report = report
        super().__init__(report.format())

    @property
    def violations(self) -> tuple[Violation, ...]:
        return self.report.violations


@dataclass(frozen=True)
class _Step:
    """A metric or a predictor that ``preflight`` checks."""

    kind: str  # "metric" or "predictor"
    name: str
    subject: str  # how the messages call it
    capabilities: Optional[Capabilities]
    display: str = ""  # the display name of a predictor
    item: Any = None  # what the caller passed for a predictor, if it was an adapter or a runner


def _declared_capabilities(owner: str, value: Any) -> Optional[Capabilities]:
    if value is not None and not isinstance(value, Capabilities):
        raise TypeError(
            f"capabilities of {owner} must be None or a Capabilities, got {type(value).__name__}"
        )
    return value


def _metric_steps(metrics) -> list[_Step]:
    """The metrics as steps: registry names, or objects with ``name`` and ``capabilities``."""
    if isinstance(metrics, str):
        metrics = [metrics]
    known = None
    steps = []
    for item in metrics:
        if isinstance(item, str):
            if known is None:
                from binding_metrics.metrics.registry import METRICS

                known = {spec.name: spec for spec in METRICS}
            name, declared = item, getattr(known.get(item), "capabilities", None)
        else:
            name = getattr(item, "name", None)
            if not isinstance(name, str):
                raise TypeError(
                    f"a metric must be a registry name or have a str name, got {item!r}"
                )
            declared = getattr(item, "capabilities", None)
        steps.append(
            _Step(
                "metric",
                name,
                f"metric {name!r}",
                _declared_capabilities(f"metric {name!r}", declared),
            )
        )
    return steps


def _text_attribute(obj: Any, attribute: str) -> Optional[str]:
    value = getattr(obj, attribute, None)
    return value if isinstance(value, str) and value else None


def _predictor_steps(predictor) -> list[_Step]:
    """The predictors as steps: registry names, adapters or runners (classes or instances)."""
    if predictor is None:
        return []
    items = predictor if isinstance(predictor, (list, tuple)) else [predictor]
    steps = []
    for item in items:
        given = None
        if isinstance(item, str):
            from binding_metrics.predictors.registry import PARSERS

            spec = PARSERS.get(item)
            if spec is None:
                available = ", ".join(sorted(PARSERS)) or "none registered"
                raise KeyError(f"Unknown predictor {item!r}. Available: {available}")
            name, display, declared = item, spec.display_name, spec.load_capabilities()
        elif isinstance(item, Capabilities):
            name, display, declared = "predictor", "(unnamed)", item
        else:
            display = (
                _text_attribute(item, "display_name")
                or _text_attribute(item, "name")
                or type(item).__name__
            )
            name = _text_attribute(item, "name") or display
            declared = getattr(item, "capabilities", None)
            given = item
        declared = _declared_capabilities(f"predictor {display}", declared)
        label = f"{display} {declared.version}".strip() if declared is not None else display
        steps.append(_Step("predictor", name, f"predictor {label}", declared, display, given))
    return steps


def _is_registry_entry_of(spec, step: _Step) -> bool:
    """True when the registry entry ``spec`` is the predictor of ``step``.

    A predictor passed as an object need not use the registry name, so the entry is matched by
    name, by display name, or by being the same adapter class.
    """
    if spec.name == step.name or spec.display_name.casefold() == step.display.casefold():
        return True
    if step.item is None:
        return False
    try:
        adapter = spec.load()
    except Exception as exc:  # noqa: BLE001 - a broken adapter must not stop the check of another
        logger.warning("could not load predictor %r: %s", spec.name, exc)
        return False
    return adapter is step.item or adapter is type(step.item)


def _other_predictors(
    profile: InputProfile,
    rejected: _Step,
    provided: Optional[Collection[str]],
    mode: Optional[str] = None,
    runnable: Optional[Collection[str]] = None,
) -> tuple[list[str], list[str], int]:
    """The registered predictors other than ``rejected``, as (declared compatible, no declared
    limits, number of other predictors).

    A predictor that declares limits and refuses the input is in neither list. When a ``mode``
    is requested, a predictor is declared compatible only if it lists that mode in ``modes``; one
    that declares no modes counts as having no declaration. ``runnable`` names the models that
    can be run from here; a listed model outside it is marked as readable only.
    """
    from binding_metrics.predictors.registry import PARSERS

    accepting, undeclared, others = [], [], 0
    for name in sorted(PARSERS):
        if _is_registry_entry_of(PARSERS[name], rejected):
            continue
        others += 1
        try:
            declared = PARSERS[name].load_capabilities()
        except Exception as exc:  # noqa: BLE001 - a broken adapter must not stop the check of another
            logger.warning("could not read the capabilities of predictor %r: %s", name, exc)
            continue
        label = f"{PARSERS[name].display_name} ({name})"
        if runnable is not None and name not in runnable:
            label += " [no runner here: give its output with --prediction-dir]"
        if declared is None:
            undeclared.append(label)
        elif declared.accepts(profile, provided=provided, mode=mode):
            if mode is not None and not declared.declares_mode(mode):
                undeclared.append(label)
            else:
                accepting.append(label)
    return accepting, undeclared, others


def _listing(path: Path, limit: int = 8) -> str:
    """The first entries of the directory ``path``, for a message ("none" when it is empty)."""
    try:
        names = sorted(entry.name + ("/" if entry.is_dir() else "") for entry in path.iterdir())
    except OSError as exc:
        return f"it cannot be listed ({exc})"
    if not names:
        return "it is empty"
    more = f" and {len(names) - limit} more" if len(names) > limit else ""
    return f"it holds {_join(names[:limit])}{more}"


def check_weights_path(path: str | os.PathLike, kind: Optional[str] = None) -> Path:
    """Check that custom weights exist, can be read, and are the kind a runner takes.

    Args:
        path: The weights file or directory (``~`` is expanded).
        kind: ``"file"`` (a checkpoint file), ``"directory"``, or None for either.

    Returns:
        The absolute path.

    Raises:
        ValueError: The path does not exist (the message lists what its directory holds), cannot
            be read, is a directory where a file is needed or the other way round (the message
            says what was found), or is a directory without files.
    """
    if kind not in (None, "file", "directory"):
        raise ValueError(f"kind must be 'file', 'directory' or None, got {kind!r}")
    given = Path(path).expanduser()
    resolved = given.absolute()
    if not resolved.exists():
        parent = resolved.parent
        where = (
            f"{parent} exists and {_listing(parent)}"
            if parent.is_dir()
            else f"{parent} does not exist"
        )
        raise ValueError(f"the custom weights {path} do not exist: {where}")
    if resolved.is_dir():
        if not os.access(resolved, os.R_OK | os.X_OK):
            raise ValueError(f"the custom weights directory {path} cannot be read")
        if kind == "file":
            raise ValueError(
                f"the custom weights {path} are a directory ({_listing(resolved)}), and the "
                "model takes one checkpoint file: give the file"
            )
        if not any(member.is_file() for member in resolved.rglob("*")):
            raise ValueError(f"the custom weights directory {path} holds no files")
        return resolved
    if not os.access(resolved, os.R_OK):
        raise ValueError(f"the custom weights file {path} cannot be read")
    if kind == "directory":
        raise ValueError(
            f"the custom weights {path} are a file ({resolved.stat().st_size} bytes), and the "
            "model takes a directory of weights: give the directory"
        )
    return resolved


def _weights_fix(weights_kinds: Mapping[str, str]) -> str:
    """What to do when the model's runner cannot take custom weights, from the runner registry."""
    from binding_metrics.predictors.registry import PARSERS

    takers = [
        f"{PARSERS[name].display_name if name in PARSERS else name} ({name}, "
        f"weights as a {'file' if kind == 'file' else 'directory'})"
        for name, kind in sorted(weights_kinds.items())
    ]
    if takers:
        use = f"use a model whose runner takes custom weights: {_join(takers)}"
    else:
        use = "no runner of this package takes custom weights yet"
    return (
        f"{use}; or run the model yourself with your weights and read its output with "
        "--prediction-dir (without --prediction-weights); or leave the custom weights out"
    )


def _fix_for(
    step: _Step,
    profile: InputProfile,
    provided: Optional[Collection[str]],
    mode: Optional[str] = None,
    runnable: Optional[Collection[str]] = None,
) -> str:
    """What to do about a violation of ``step``.

    For a predictor the fix names the other registered predictors in two lists: those whose
    declared limits accept the input (and, when a mode is requested, that declare the mode), and
    those that declare no limits, which is not the same as validated. It says so plainly when a
    list is empty, and never offers the rejected predictor.
    """
    if step.kind == "metric":
        return (
            f"leave {step.name!r} out of the metric list, or use policy='skip' "
            "(--on-incompatible skip) to compute only the metrics that apply"
        )
    accepting, undeclared, others = _other_predictors(profile, step, provided, mode, runnable)
    declares = f" that declare the mode '{mode}'" if mode is not None else ""
    parts = []
    if not others:
        parts.append("no other predictor is registered")
    else:
        if accepting:
            parts.append(
                f"predictors{declares} whose declared limits accept this input: {_join(accepting)}"
            )
        else:
            parts.append(
                "no other registered predictor declares support for this input"
                + (f" in the mode '{mode}'" if mode is not None else "")
            )
        if undeclared:
            parts.append(
                f"predictors that declare no limits (not validated for this input): "
                f"{_join(undeclared)}"
            )
    parts.append(
        "or use policy='skip' (--on-incompatible skip) to leave the predictor out and run the rest"
    )
    return "; ".join(parts)


def _check_weights(
    steps: list[_Step],
    weights: str | os.PathLike,
    weights_kinds: Optional[Mapping[str, str]],
    violations: list[Violation],
    notes: list[str],
    refused: set[tuple[str, str]],
) -> None:
    """The custom-weights checks of ``preflight``: adds violations and notes in place."""
    check_weights_path(weights)  # exists and can be read, whatever the model
    predictors = [s for s in steps if s.kind == "predictor"]
    if not predictors:
        notes.append(
            f"custom weights were given ({weights}) and no predictor is checked, so they are "
            "not used"
        )
        return
    for step in predictors:
        display = step.display or step.name
        if weights_kinds is None:
            notes.append(
                f"the custom weights were not checked against {step.subject}: the caller did not "
                "say which runners take them"
            )
            continue
        kind = weights_kinds.get(step.name)
        if kind is None:
            found = "a directory" if Path(weights).expanduser().is_dir() else "a file"
            violations.append(
                Violation(
                    constraint="weights",
                    fact=f"custom weights were given ({weights}, {found})",
                    requirement=f"a runner that starts {display} with weights you choose",
                    reason=(
                        f"the {display} runner of this package does not pass weights to the model, "
                        "so a run would use the default weights and the result would not be the "
                        "model you asked for"
                    ),
                    fix=_weights_fix(weights_kinds),
                    subject=step.subject,
                    kind="predictor",
                    name=step.name,
                )
            )
            refused.add(("predictor", step.name))
            continue
        check_weights_path(weights, kind)
        notes.append(
            f"custom weights: the limits declared for {display} come from its input format and "
            "architecture; the caveats about accuracy and published benchmarks refer to the "
            "standard weights"
        )


def preflight(
    profile: InputProfile,
    metrics: Iterable[Any] = (),
    predictor: Any = None,
    *,
    policy: str = "error",
    predictor_policy: Optional[str] = None,
    provided: Optional[Collection[str]] = None,
    mode: Optional[str] = None,
    runnable: Optional[Collection[str]] = None,
    weights: Optional[str | os.PathLike] = None,
    weights_kinds: Optional[Mapping[str, str]] = None,
) -> PreflightReport:
    """Check an input against the limits of every requested metric and predictor, before any run.

    Call it first: it reads declarations only and never runs, prepares or instantiates a metric,
    a predictor or a runner. It collects every incompatibility, not the first one, each with the
    fact found in the input, the requirement, the step's own reason and a fix.

    Args:
        profile: The input, from ``profile_input``.
        metrics: Registry names (``"omega"``) or objects with a ``name`` and a ``capabilities``
            attribute (a ``MetricSpec``). A name the registry does not know has no declared limit.
        predictor: None, a registered predictor name (``"of3"``), an adapter or runner (class or
            instance) with a ``capabilities`` attribute, a ``Capabilities``, or a list of these.
            When it is refused, the message lists the other registered predictors that accept the
            input.
        policy: ``error`` (default) raises ``IncompatibleInputError`` when anything is
            incompatible, so nothing runs. ``skip`` leaves out each metric that has a violation
            (``report.metrics_to_run`` holds the rest) and marks a refused predictor
            ``report.predictor_usable=False``. ``warn`` logs every violation and lets everything
            run.
        predictor_policy: The policy for the violations of the predictors, when it is not the
            one of the metrics; None uses ``policy``. A caller that only reads a prediction made
            elsewhere passes ``warn``, because what the model was given is not known.
        provided: What the caller makes available among ``NEEDS`` besides the receptor chain,
            for example ``{"reference_structure", "gpu"}``; None leaves those needs unchecked and
            the report says so.
        mode: The mode the predictors are asked to run in, one of ``MODES``. A predictor that
            declares ``modes`` without it is refused (constraint ``modes``) and its fix lists the
            models that declare the mode, then those with no declaration; a caveat for the mode
            is a warning. None does not say: the mode is not checked.
        runnable: The names of the predictors that can be run from here; a model outside it
            that the fix offers is marked as readable only. None leaves the fix unmarked.
        weights: Custom weights (``--prediction-weights``) that the run will give the model. The
            path must exist and be readable (``check_weights_path``). A predictor whose runner
            does not take custom weights is refused (constraint ``weights``, with a fix that
            names the runners that do), and one that does gets a note: the limits it declares
            come from its input format and architecture, and its caveats about accuracy and
            published benchmarks refer to the standard weights. The declared hard limits stay in
            force.
        weights_kinds: Model name to the kind of weights its runner takes (``"file"`` or
            ``"directory"``), for the models whose runner supports custom weights
            (``cli.prediction.runner_weights_kinds`` reads it from the runner registry). None
            says the caller does not know: the weights are not checked against the runners and the
            report says so.

    Returns:
        The ``PreflightReport``. Soft warnings for inputs a step accepts but never validated are
        in ``report.warnings`` whatever the policy.

    Raises:
        IncompatibleInputError: Policy ``error`` and at least one violation. Subclass of
            ``ValueError``.
        ValueError: ``policy`` or ``predictor_policy`` is not one of ``POLICIES``, ``mode`` is
            not one of ``MODES``, or ``weights`` does not exist, cannot be read or is not the kind
            a runner takes (``check_weights_path``).
        KeyError: A predictor name that is not registered.
        TypeError: A ``capabilities`` attribute that is neither None nor a ``Capabilities``.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if predictor_policy is not None and predictor_policy not in POLICIES:
        raise ValueError(f"predictor_policy must be one of {POLICIES}, got {predictor_policy!r}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    effective_predictor_policy = predictor_policy or policy
    steps = _metric_steps(metrics) + _predictor_steps(predictor)

    violations: list[Violation] = []
    warnings: list[str] = []
    notes: list[str] = list(profile.notes)
    refused: set[tuple[str, str]] = set()
    for step in steps:
        declared = step.capabilities
        if declared is None:
            continue
        fix = None  # built once per step: it may list the other registered predictors
        step_mode = mode if step.kind == "predictor" else None
        for found in declared.check(profile, provided=provided, mode=step_mode):
            if fix is None:
                fix = _fix_for(step, profile, provided, step_mode, runnable)
            violations.append(
                replace(found, subject=step.subject, kind=step.kind, name=step.name, fix=fix)
            )
            refused.add((step.kind, step.name))
        warnings += [f"{step.subject}: {text}" for text in declared.caveats_for(profile, step_mode)]

    if weights is not None:
        _check_weights(steps, weights, weights_kinds, violations, notes, refused)

    typed = [s.subject for s in steps if s.capabilities and s.capabilities.binder_types]
    if profile.binder_type == "unknown" and typed:
        notes.append(
            f"the binder type is unknown, so the binder-type check of {_join(typed)} was skipped"
        )
        logger.info("binder type unknown: binder-type checks of %s skipped", _join(typed))
    if provided is None:
        unchecked = [
            s.subject for s in steps if s.capabilities and s.capabilities.needs - {"receptor_chain"}
        ]
        if unchecked:
            notes.append(
                f"the needs of {_join(unchecked)} other than the receptor chain were not checked: "
                "the caller did not say what it provides"
            )

    requested = tuple(s.name for s in steps if s.kind == "metric")
    if policy == "skip":
        to_run = tuple(name for name in requested if ("metric", name) not in refused)
    else:
        to_run = requested
    usable = not (
        effective_predictor_policy == "skip" and any(kind == "predictor" for kind, _ in refused)
    )
    report = PreflightReport(
        policy=policy,
        profile=profile,
        violations=tuple(violations),
        warnings=tuple(warnings),
        notes=tuple(dict.fromkeys(notes)),
        metrics_requested=requested,
        metrics_to_run=to_run,
        predictors=tuple(
            s.subject.removeprefix("predictor ") for s in steps if s.kind == "predictor"
        ),
        predictor_usable=usable,
        predictor_policy=predictor_policy or "",
        mode=mode,
    )
    if report.refused:
        raise IncompatibleInputError(report)
    for violation in violations:
        action = (
            "leaving it out" if report.policy_of(violation.kind) == "skip" else "running anyway"
        )
        logger.warning(
            "%s is incompatible with the input (%s: %s); %s",
            violation.subject,
            violation.constraint,
            violation.fact,
            action,
        )
    return report
