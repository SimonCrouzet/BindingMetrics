"""``Reference``: the structure that a set of poses is placed on and measured against.

A reference is an analysis role, not a kind of pose: a crystal structure, a receptor that a
model was forced to keep, or a docked complex can all play it. It names

* the receptor chain, and optionally the binder chain (the peptide of the reference complex),
* an optional ``residue_map`` that pairs the residues of a pose with the residues of the
  reference,
* NAMED residue sets in the numbering of the reference (an epitope, hotspots, a pocket),
* NAMED partners: other structures that are already placed in the frame of the reference
  (a known binder on the same receptor).

The reference structure is given as a path, a biotite ``AtomArray`` or a ``Pose``; it is
stored as a pose (a bare path or array becomes an ``ExperimentalPose``), so its content hash
comes from the pose machinery. Nothing is read until ``atoms()`` or ``validate()`` is called.

A residue is identified by a ``ResidueKey``, ``(chain, number, insertion code)``, as
``load_structure`` reads the author numbering: ``52`` and ``52A`` are two residues.

Several variants of the reference can be given to one call as a mapping
``{variant_name: Reference}``; ``normalize_references`` turns either form into that mapping.

Importing this module imports numpy and the standard library (and the pose modules, which add
nothing else); biotite is imported when a structure is read.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Iterable, Mapping, NamedTuple, Optional

import numpy as np

from binding_metrics.poses.docked import DockedPose
from binding_metrics.poses.experimental import ExperimentalPose
from binding_metrics.poses.pose import Pose, check_source

REFERENCE_SCHEMA = "binding_metrics.reference"
REFERENCE_SCHEMA_VERSION = 1

#: Region names the frame stage reports itself: the whole paired receptor and the binding site
#: (receptor residues near the reference binder). A residue set may not take them.
RESERVED_REGION_NAMES = ("all", "site")

_LABEL_PATTERN = re.compile(r"^(?P<chain>[^:\s]+):(?P<number>-?\d+)(?P<icode>[A-Za-z]?)$")


@dataclass(frozen=True, order=True)
class ResidueKey:
    """The identity of one residue: chain, residue number and insertion code.

    Author numbering, as ``load_structure`` reads it, so ``ResidueKey("A", 52, "")`` and
    ``ResidueKey("A", 52, "A")`` are different residues. Keys are hashable and sort by chain,
    number and insertion code. ``ResidueKey.coerce`` also reads ``("A", 52)``,
    ``("A", 52, "B")`` and the labels ``"A:52"`` and ``"A:52B"``.

    Raises:
        TypeError: If the chain is not a string, the number is not an integer (a ``bool`` is
            not), or the insertion code is not a string.
        ValueError: If the chain is empty or the insertion code has more than one character.
    """

    chain: str
    number: int
    icode: str = ""

    def __post_init__(self):
        if not isinstance(self.chain, str):
            raise TypeError(f"chain must be a string, got {self.chain!r}")
        if not self.chain.strip():
            raise ValueError("chain must not be empty")
        if isinstance(self.number, bool) or not isinstance(self.number, (int, np.integer)):
            raise TypeError(f"residue number must be an integer, got {self.number!r}")
        if not isinstance(self.icode, str):
            raise TypeError(f"insertion code must be a string, got {self.icode!r}")
        icode = self.icode.strip()
        if len(icode) > 1:
            raise ValueError(f"insertion code must be one character or empty, got {self.icode!r}")
        object.__setattr__(self, "number", int(self.number))
        object.__setattr__(self, "icode", icode)

    @property
    def label(self) -> str:
        """``"A:52"`` or ``"A:52B"``: chain, number and insertion code."""
        return f"{self.chain}:{self.number}{self.icode}"

    def __str__(self) -> str:
        return self.label

    @classmethod
    def parse(cls, label: str) -> ResidueKey:
        """Read ``"A:52"`` or ``"A:52B"`` (a letter after the number is the insertion code).

        Raises:
            ValueError: If ``label`` has another form.
        """
        match = _LABEL_PATTERN.match(label.strip()) if isinstance(label, str) else None
        if match is None:
            raise ValueError(
                f"cannot read {label!r} as a residue key: write 'A:52' or 'A:52B' "
                "(chain, colon, number, optional insertion code letter)"
            )
        return cls(match["chain"], int(match["number"]), match["icode"])

    @classmethod
    def coerce(cls, value: Any) -> ResidueKey:
        """``value`` as a key: a key, a label string, or a ``(chain, number[, icode])`` pair."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        if isinstance(value, (tuple, list)) and len(value) in (2, 3):
            return cls(*value)
        raise TypeError(
            f"cannot read {value!r} as a residue key: give a ResidueKey, a label such as "
            "'A:52B', or a (chain, number) or (chain, number, icode) tuple"
        )

    def as_tuple(self) -> tuple:
        """``(chain, number, icode)``."""
        return (self.chain, self.number, self.icode)

    def to_list(self) -> list:
        """``[chain, number, icode]``, the JSON form."""
        return [self.chain, self.number, self.icode]

    @classmethod
    def from_list(cls, values: Iterable) -> ResidueKey:
        """Inverse of ``to_list``."""
        values = list(values)
        if len(values) != 3:
            raise ValueError(f"a residue key is [chain, number, icode], got {values!r}")
        return cls(*values)


class ChainResidue(NamedTuple):
    """One polymer residue of one chain of an ``AtomArray``.

    Attributes:
        key: Chain, number and insertion code.
        name: Residue name as the structure writes it.
        atom_indices: Indices of the residue's atoms in the array, in array order.
        ca_index: Index of its CA atom in the array.
    """

    key: ResidueKey
    name: str
    atom_indices: np.ndarray
    ca_index: int


def polymer_residues(atoms, chain: str) -> list[ChainResidue]:
    """The polymer residues of ``chain``, in the order of the array.

    A residue is a run of consecutive atoms of the chain with one residue number, insertion
    code and name; it is a polymer residue when it has a CA atom of carbon (so a calcium ion,
    named ``CA``, and a water are not). A number and insertion code that occurs twice in a
    chain gives two entries with equal keys; callers that need unique keys check for that.

    Args:
        atoms: A biotite ``AtomArray`` (one model).
        chain: The chain ID, as the array writes it.
    """
    chain_atoms = np.flatnonzero(np.asarray(atoms.chain_id) == chain)
    if chain_atoms.size == 0:
        return []
    numbers = np.asarray(atoms.res_id)[chain_atoms]
    icodes = _annotation(atoms, "ins_code")[chain_atoms]
    names = np.asarray(atoms.res_name)[chain_atoms]
    boundary = np.ones(chain_atoms.size, dtype=bool)
    boundary[1:] = (
        (numbers[1:] != numbers[:-1]) | (icodes[1:] != icodes[:-1]) | (names[1:] != names[:-1])
    )
    starts = np.flatnonzero(boundary)
    stops = np.append(starts[1:], chain_atoms.size)
    atom_names = np.asarray(atoms.atom_name)[chain_atoms]
    elements = _annotation(atoms, "element")[chain_atoms]
    residues = []
    for start, stop in zip(starts, stops):
        is_alpha_carbon = atom_names[start:stop] == "CA"
        is_alpha_carbon &= np.isin(elements[start:stop], ("C", ""))
        found = np.flatnonzero(is_alpha_carbon)
        if found.size == 0:
            continue
        residues.append(
            ChainResidue(
                key=ResidueKey(chain, int(numbers[start]), str(icodes[start])),
                name=str(names[start]),
                atom_indices=chain_atoms[start:stop],
                ca_index=int(chain_atoms[start + found[0]]),
            )
        )
    return residues


def _annotation(atoms, name: str) -> np.ndarray:
    """An annotation as an array; a category the array lacks reads as empty strings."""
    if name in atoms.get_annotation_categories():
        return np.asarray(atoms.get_annotation(name))
    return np.full(atoms.array_length(), "", dtype="U1")


def _chains_of(atoms) -> list[str]:
    return sorted({str(chain) for chain in np.unique(np.asarray(atoms.chain_id))})


def _as_pose(structure: Any, what: str) -> Pose:
    """A pose for a path, an ``AtomArray`` or a pose; a path must be a file."""
    pose = structure if isinstance(structure, Pose) else ExperimentalPose(check_source(structure))
    path = pose.structure_path
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"{what} structure file not found: {path}")
    return pose


def _describe(pose: Pose) -> str:
    path = pose.structure_path
    return f"'{path}'" if path is not None else "the in-memory structure"


def _structure_to_dict(pose: Pose) -> dict:
    path = pose.structure_path
    return {
        "path": None if path is None else str(path),
        "kind": pose.kind,
        "chain_map": dict(sorted(pose.chain_map.items())),
        "content_hash": pose.content_hash(),
    }


def _structure_from_dict(data: Mapping, given: Any, what: str) -> Pose:
    """Rebuild a structure from its dictionary; ``given`` replaces the recorded path."""
    unknown = sorted(set(data) - {"path", "kind", "chain_map", "content_hash"})
    if unknown:
        raise ValueError(f"unknown keys {unknown} in the {what} structure")
    chain_map = dict(data.get("chain_map") or {})
    if given is not None:
        pose = _as_pose(given, what)
    else:
        path = data.get("path")
        if path is None:
            raise ValueError(
                f"the {what} structure was in memory when the dictionary was written: pass it "
                "as an argument"
            )
        kind = DockedPose if data.get("kind") in ("docked", "predicted") else ExperimentalPose
        pose = _as_pose(kind(Path(path), chain_map=chain_map), what)
    recorded = data.get("content_hash")
    if recorded is not None and pose.content_hash() != recorded:
        raise ValueError(
            f"the {what} structure {_describe(pose)} does not match the recorded content hash "
            "(the file changed, or another structure was given)"
        )
    return pose


def _sha256_of(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chain_list(chains: Any) -> Optional[tuple[str, ...]]:
    """``chains`` as a tuple of distinct chain IDs, or None (every polymer chain)."""
    if chains is None:
        return None
    if isinstance(chains, str):
        chains = (chains,)
    try:
        chains = tuple(chains)
    except TypeError as error:
        raise TypeError(
            f"chains must be a chain ID or a sequence of chain IDs, got {chains!r}"
        ) from error
    if not chains or not all(isinstance(c, str) and c.strip() for c in chains):
        raise ValueError(f"chains must be non-empty chain ID strings, got {chains!r}")
    if len(set(chains)) != len(chains):
        raise ValueError(f"chains repeats a chain ID: {chains!r}")
    return chains


@dataclass(frozen=True, eq=False, repr=False)
class Partner:
    """A named partner that is already placed in the frame of the reference.

    A known binder on the same receptor, for instance. It is not moved: the descriptors that
    use it read its coordinates as they are. ``eq=False``: like a pose, a partner compares by
    identity; ``content_hash()`` tells two partners with the same structure apart from others.

    Attributes:
        structure: A path (mmCIF or PDB), a biotite ``AtomArray`` of one model, or a ``Pose``.
            A bare path or array becomes an ``ExperimentalPose``. The array is not copied: do
            not modify it afterwards.
        chains: The chains of the structure that make up the partner (a chain ID or a sequence
            of them), or None for every polymer chain, that is, every chain with at least one
            residue that has a carbon CA atom.

    Raises:
        TypeError: If ``structure`` is not a path, an ``AtomArray`` or a pose.
        FileNotFoundError: If a path is not a file.
        ValueError: If ``chains`` is empty or repeats a chain ID.
    """

    structure: Any
    chains: Optional[Any] = None

    def __post_init__(self):
        object.__setattr__(self, "structure", _as_pose(self.structure, "partner"))
        object.__setattr__(self, "chains", _chain_list(self.chains))

    @property
    def pose(self) -> Pose:
        """The structure as a pose."""
        return self.structure

    def atoms(self):
        """The atoms of the partner chains, a new ``AtomArray`` in the order of the structure.

        Raises:
            ValueError: If a chain in ``chains`` is not in the structure, or the structure has
                no polymer chain and ``chains`` is None.
        """
        atoms = self.structure.atoms()
        present = _chains_of(atoms)
        if self.chains is None:
            chains = [chain for chain in present if polymer_residues(atoms, chain)]
            if not chains:
                raise ValueError(f"partner {_describe(self.structure)} has no polymer chain")
        else:
            missing = [chain for chain in self.chains if chain not in present]
            if missing:
                raise ValueError(
                    f"partner chains {missing} are not in {_describe(self.structure)}: "
                    f"it has chains {present}"
                )
            chains = list(self.chains)
        return atoms[np.isin(np.asarray(atoms.chain_id), chains)]

    def content_hash(self) -> str:
        """SHA-256 hex digest of the structure (the pose's hash), its chain renaming and the
        chains used."""
        return _sha256_of(
            {
                "schema": "binding_metrics.partner.v1",
                "structure": self.structure.content_hash(),
                "chain_map": sorted(self.structure.chain_map.items()),
                "chains": None if self.chains is None else list(self.chains),
            }
        )

    def to_dict(self) -> dict:
        """JSON-ready form; an in-memory structure is recorded by its hash only."""
        return {
            "structure": _structure_to_dict(self.structure),
            "chains": None if self.chains is None else list(self.chains),
        }

    @classmethod
    def from_dict(cls, data: Mapping, *, structure: Any = None) -> Partner:
        """Inverse of ``to_dict``.

        Args:
            data: A dictionary written by ``to_dict``.
            structure: The structure to use in place of the recorded path (required for a
                partner that was in memory). Its content hash must equal the recorded one.

        Raises:
            ValueError: If a key is unknown, the structure is missing, or its hash differs.
        """
        unknown = sorted(set(data) - {"structure", "chains"})
        if unknown:
            raise ValueError(f"unknown keys {unknown} in a partner dictionary")
        pose = _structure_from_dict(data["structure"], structure, "partner")
        return cls(pose, data.get("chains"))

    def __repr__(self) -> str:
        return f"Partner(structure={_describe(self.structure)}, chains={self.chains!r})"


def _as_partner(value: Any) -> Partner:
    return value if isinstance(value, Partner) else Partner(value)


def _frozen_map(mapping: dict) -> Mapping:
    return MappingProxyType(mapping)


#: The fields of ``Reference`` that hold read-only mapping views.
_MAPPING_FIELDS = ("residue_map", "residue_sets", "partners")


@dataclass(frozen=True, eq=False, repr=False)
class Reference:
    """The structure that poses are placed on and measured against (see the module docstring).

    The reference compares by identity (as poses do); ``content_hash()`` tells two references
    with the same content apart from others. Its mappings are read-only views.

    Attributes:
        structure: The reference receptor, and the reference peptide when ``binder_chain`` is
            given: a path (mmCIF or PDB, possibly gzip-compressed), a biotite ``AtomArray`` of
            one model (not copied: do not modify it afterwards) or a ``Pose``. A bare path or
            array becomes an ``ExperimentalPose``; pass a pose of another kind to say that the
            reference is a docked complex or a predicted one. Stored as a pose.
        receptor_chain: The receptor chain of the structure (user chain ID: the pose's
            ``chain_map`` is applied).
        binder_chain: The peptide chain of the reference complex, or None. With it the frame
            stage reports the binding-site region (receptor residues within
            ``site_cutoff_angstrom`` of the peptide).
        residue_map: ``{pose ResidueKey: reference ResidueKey}``, the pairing of pose residues
            with reference residues for every pose; it wins over every automatic pairing. The
            reference side is on ``receptor_chain``, and no two pose residues map to the same
            reference residue. Keys may be given as tuples or labels.
        residue_sets: ``{name: collection of ResidueKey}``: named receptor residue sets in the
            numbering of the reference (an epitope, hotspots). A set is not empty and lies on
            ``receptor_chain``; its name is not ``"all"`` or ``"site"``, which are the regions
            that the frame stage reports itself.
        partners: ``{name: Partner}``, structures already placed in the reference frame. A
            path, array or pose is wrapped as ``Partner(value)``.
        name: The variant name of this reference; the key of every block that is reported per
            variant.

    Raises:
        TypeError: For an argument of the wrong type.
        ValueError: For an empty or repeated name, a binder chain equal to the receptor chain,
            a residue set that is empty, reserved-named or off the receptor chain, or a residue
            map that is not one-to-one or has a reference side off the receptor chain.
        FileNotFoundError: If a structure path is not a file.
    """

    structure: Any
    receptor_chain: str
    binder_chain: Optional[str] = None
    residue_map: Optional[Mapping] = None
    residue_sets: Optional[Mapping] = None
    partners: Optional[Mapping] = None
    name: str = "reference"

    def __post_init__(self):
        object.__setattr__(self, "structure", _as_pose(self.structure, "reference"))
        if not isinstance(self.receptor_chain, str):
            raise TypeError(
                f"receptor_chain must be a chain ID string, got {self.receptor_chain!r}"
            )
        if not self.receptor_chain.strip():
            raise ValueError("receptor_chain must not be empty")
        binder = self.binder_chain
        if binder is not None:
            if not isinstance(binder, str):
                raise TypeError(f"binder_chain must be a chain ID string or None, got {binder!r}")
            if not binder.strip():
                raise ValueError("binder_chain must not be empty (use None for no binder)")
            if binder == self.receptor_chain:
                raise ValueError(
                    f"binder_chain and receptor_chain are both '{binder}': they are two chains"
                )
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a string, got {self.name!r}")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        object.__setattr__(self, "residue_map", self._checked_map(self.residue_map))
        object.__setattr__(self, "residue_sets", self._checked_sets(self.residue_sets))
        object.__setattr__(self, "partners", self._checked_partners(self.partners))

    # ------------------------------------------------------------------ construction checks

    def _checked_map(self, residue_map: Optional[Mapping]) -> Mapping:
        if residue_map is None:
            return _frozen_map({})
        if not isinstance(residue_map, Mapping):
            raise TypeError("residue_map must be a mapping of pose residue to reference residue")
        pairs = {ResidueKey.coerce(k): ResidueKey.coerce(v) for k, v in residue_map.items()}
        off_chain = sorted(v.label for v in pairs.values() if v.chain != self.receptor_chain)
        if off_chain:
            raise ValueError(
                f"residue_map values are reference residues and must be on the receptor chain "
                f"'{self.receptor_chain}'; found {off_chain[:5]}"
            )
        seen: dict[ResidueKey, ResidueKey] = {}
        for pose_key, reference_key in pairs.items():
            if reference_key in seen:
                raise ValueError(
                    f"residue_map is not one-to-one: {seen[reference_key].label} and "
                    f"{pose_key.label} both map to the reference residue {reference_key.label}"
                )
            seen[reference_key] = pose_key
        return _frozen_map(dict(sorted(pairs.items())))

    def _checked_sets(self, residue_sets: Optional[Mapping]) -> Mapping:
        if residue_sets is None:
            return _frozen_map({})
        if not isinstance(residue_sets, Mapping):
            raise TypeError("residue_sets must be a mapping of set name to a collection of keys")
        checked: dict[str, frozenset] = {}
        for name, members in residue_sets.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"residue set names must be non-empty strings, got {name!r}")
            if name in RESERVED_REGION_NAMES:
                raise ValueError(
                    f"residue set name '{name}' is reserved: the frame stage reports the "
                    f"regions {list(RESERVED_REGION_NAMES)} itself"
                )
            if isinstance(members, (str, bytes)) or not isinstance(members, Collection):
                raise TypeError(
                    f"residue set '{name}' must be a collection of ResidueKey, got {members!r}"
                )
            keys = frozenset(ResidueKey.coerce(member) for member in members)
            if not keys:
                raise ValueError(f"residue set '{name}' is empty")
            off_chain = sorted(k.label for k in keys if k.chain != self.receptor_chain)
            if off_chain:
                raise ValueError(
                    f"residue set '{name}' has residues off the receptor chain "
                    f"'{self.receptor_chain}': {off_chain[:5]}; a residue set names receptor "
                    "residues in reference numbering"
                )
            checked[name] = keys
        return _frozen_map(dict(sorted(checked.items())))

    def _checked_partners(self, partners: Optional[Mapping]) -> Mapping:
        if partners is None:
            return _frozen_map({})
        if not isinstance(partners, Mapping):
            raise TypeError("partners must be a mapping of partner name to Partner")
        checked: dict[str, Partner] = {}
        for name, value in partners.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"partner names must be non-empty strings, got {name!r}")
            checked[name] = _as_partner(value)
        return _frozen_map(dict(sorted(checked.items())))

    # ------------------------------------------------------------------ structure

    @property
    def pose(self) -> Pose:
        """The reference structure as a pose."""
        return self.structure

    def atoms(self):
        """The reference structure as an ``AtomArray`` (the pose's cached, read-only array)."""
        return self.structure.atoms()

    def validate(self) -> None:
        """Check the reference against its structure, reading it.

        Every problem is listed in one error: a chain that the structure does not have, a
        receptor chain without polymer residues, residue-set or residue-map residues that the
        receptor chain does not have, a partner chain that is missing.

        Raises:
            ValueError: With one line per problem.
            ImportError: If biotite is needed and not installed.
        """
        atoms = self.atoms()
        present = _chains_of(atoms)
        where = _describe(self.structure)
        problems = []
        for role, chain in (("receptor", self.receptor_chain), ("binder", self.binder_chain)):
            if chain is not None and chain not in present:
                problems.append(
                    f"{role} chain '{chain}' is not in {where}: it has chains {present}"
                )
        if self.receptor_chain in present:
            receptor_keys = {r.key for r in polymer_residues(atoms, self.receptor_chain)}
            if not receptor_keys:
                problems.append(f"receptor chain '{self.receptor_chain}' has no polymer residues")
            else:
                for set_name, members in self.residue_sets.items():
                    missing = sorted(members - receptor_keys)
                    if missing:
                        problems.append(
                            _missing_line(f"residue set '{set_name}'", missing, members)
                        )
                missing = sorted(set(self.residue_map.values()) - receptor_keys)
                if missing:
                    problems.append(
                        _missing_line("residue_map", missing, set(self.residue_map.values()))
                    )
        for partner_name, partner in self.partners.items():
            try:
                partner.atoms()
            except ValueError as error:
                problems.append(f"partner '{partner_name}': {error}")
        if problems:
            raise ValueError(f"reference '{self.name}' is not valid:\n  " + "\n  ".join(problems))

    # ------------------------------------------------------------------ identity

    def content_hash(self) -> str:
        """SHA-256 hex digest of everything that changes a number computed against the reference.

        The structure (the pose's content hash and its chain renaming), the chains, the residue
        map, the residue sets and the partners (each by its own hash). The ``name`` is not in
        it: a variant keeps its content under another name. Independent of the order in which
        the sets, partners and map entries were given.
        """
        return _sha256_of(
            {
                "schema": f"{REFERENCE_SCHEMA}.v{REFERENCE_SCHEMA_VERSION}",
                "structure": self.structure.content_hash(),
                "chain_map": sorted(self.structure.chain_map.items()),
                "receptor_chain": self.receptor_chain,
                "binder_chain": self.binder_chain,
                "residue_map": [[k.to_list(), v.to_list()] for k, v in self.residue_map.items()],
                "residue_sets": {
                    name: [key.to_list() for key in sorted(members)]
                    for name, members in self.residue_sets.items()
                },
                "partners": {name: p.content_hash() for name, p in self.partners.items()},
            }
        )

    def frame_key(self, frame_residues: Optional[str] = None) -> tuple:
        """What decides the superposition: variants with equal keys share one superposition.

        The structure (hash and chain renaming), the receptor chain, the residue map (a
        different pairing is a different frame), and, when the fit is restricted to a named
        residue set, that set's residues. The binder chain, the other sets and the partners do
        not change the frame. ``frame_residues="site"`` adds the binder chain, which defines
        the site.

        Raises:
            ValueError: If ``frame_residues`` is neither ``"site"`` (which needs a binder
                chain) nor a residue set of this reference.
        """
        fit_set: tuple = ()
        if frame_residues == "site":
            if self.binder_chain is None:
                raise ValueError(
                    f"frame_residues='site' needs a binder_chain on reference '{self.name}'"
                )
            fit_set = ("site", self.binder_chain)
        elif frame_residues is not None:
            members = self.residue_sets.get(frame_residues)
            if members is None:
                raise ValueError(
                    f"frame_residues='{frame_residues}' is not a residue set of reference "
                    f"'{self.name}' (it has {list(self.residue_sets)}) and not 'site'"
                )
            fit_set = (frame_residues, tuple(key.as_tuple() for key in sorted(members)))
        return (
            self.structure.content_hash(),
            tuple(sorted(self.structure.chain_map.items())),
            self.receptor_chain,
            tuple((k.as_tuple(), v.as_tuple()) for k, v in self.residue_map.items()),
            fit_set,
        )

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> dict:
        """JSON-ready form: a structure that is a file is recorded by path and hash, an
        in-memory one by hash only; keys are ``[chain, number, icode]`` lists.

        The result is deterministic: names and keys are sorted.
        """
        return {
            "schema": REFERENCE_SCHEMA,
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "name": self.name,
            "structure": _structure_to_dict(self.structure),
            "receptor_chain": self.receptor_chain,
            "binder_chain": self.binder_chain,
            "residue_map": [[k.to_list(), v.to_list()] for k, v in self.residue_map.items()],
            "residue_sets": {
                name: [key.to_list() for key in sorted(members)]
                for name, members in self.residue_sets.items()
            },
            "partners": {name: partner.to_dict() for name, partner in self.partners.items()},
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping,
        *,
        structure: Any = None,
        partner_structures: Optional[Mapping[str, Any]] = None,
    ) -> Reference:
        """Inverse of ``to_dict``.

        Args:
            data: A dictionary written by ``to_dict``.
            structure: The reference structure to use in place of the recorded path; needed
                when the reference was in memory.
            partner_structures: The same for partners, by partner name.

        Raises:
            ValueError: If a key or the schema version is unknown, a structure is missing
                (an in-memory structure has no path), or a structure does not match its
                recorded content hash.
        """
        expected = {
            "schema",
            "schema_version",
            "name",
            "structure",
            "receptor_chain",
            "binder_chain",
            "residue_map",
            "residue_sets",
            "partners",
        }
        unknown = sorted(set(data) - expected)
        if unknown:
            raise ValueError(f"unknown keys {unknown} in a reference dictionary")
        if data.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(f"not a reference dictionary: schema is {data.get('schema')!r}")
        if data.get("schema_version") != REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"reference schema_version {data.get('schema_version')!r} is not supported "
                f"(this version reads {REFERENCE_SCHEMA_VERSION})"
            )
        given = dict(partner_structures or {})
        stray = sorted(set(given) - set(data.get("partners", {})))
        if stray:
            raise ValueError(f"partner_structures names partners {stray} that the data lacks")
        return cls(
            _structure_from_dict(data["structure"], structure, "reference"),
            data["receptor_chain"],
            data.get("binder_chain"),
            {
                ResidueKey.from_list(k): ResidueKey.from_list(v)
                for k, v in data.get("residue_map", [])
            },
            {
                name: [ResidueKey.from_list(key) for key in members]
                for name, members in data.get("residue_sets", {}).items()
            },
            {
                name: Partner.from_dict(entry, structure=given.get(name))
                for name, entry in data.get("partners", {}).items()
            },
            name=data["name"],
        )

    # ------------------------------------------------------------------ pickling

    def __getstate__(self) -> dict:
        """The state for pickle and deepcopy: the read-only mapping views as plain dicts.

        ``MappingProxyType`` cannot be pickled, and a worker process needs the reference. The
        cache that ``prepare_reference`` keeps is left out: it is rebuilt from the structure.
        """
        state = {key: value for key, value in self.__dict__.items() if key != "_prepared_cache"}
        for name in _MAPPING_FIELDS:
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state: dict) -> None:
        state = dict(state)
        for name in _MAPPING_FIELDS:
            state[name] = _frozen_map(state[name])
        self.__dict__.update(state)

    def __repr__(self) -> str:
        sets = {name: len(members) for name, members in self.residue_sets.items()}
        parts = [
            f"name={self.name!r}",
            f"structure={_describe(self.structure)}",
            f"receptor_chain={self.receptor_chain!r}",
        ]
        if self.binder_chain is not None:
            parts.append(f"binder_chain={self.binder_chain!r}")
        if self.residue_map:
            parts.append(f"residue_map=<{len(self.residue_map)} residues>")
        if sets:
            parts.append(f"residue_sets={sets}")
        if self.partners:
            parts.append(f"partners={list(self.partners)}")
        return f"Reference({', '.join(parts)})"


def _missing_line(what: str, missing: list, members: Collection) -> str:
    shown = ", ".join(key.label for key in missing[:8])
    more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
    return (
        f"{what} names {len(missing)} of {len(members)} residues that the reference receptor "
        f"chain does not have: {shown}{more}"
    )


def normalize_references(reference: Any) -> dict[str, Reference]:
    """The references of a call as ``{variant_name: Reference}``.

    A call takes ``None`` (no reference, an empty dict), one ``Reference`` (the variant name is
    its ``name``, ``"reference"`` by default) or a mapping ``{variant_name: Reference}``. In a
    mapping the key is the variant name: a reference with another ``name`` is replaced by a
    copy that has the key as its name.

    Raises:
        TypeError: If ``reference`` is none of these, or a mapping value is not a ``Reference``.
        ValueError: If the mapping is empty or a key is not a non-empty string.
    """
    if reference is None:
        return {}
    if isinstance(reference, Reference):
        return {reference.name: reference}
    if isinstance(reference, Mapping):
        if not reference:
            raise ValueError("reference is an empty mapping: give a Reference, variants or None")
        variants: dict[str, Reference] = {}
        for key, value in reference.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"variant names must be non-empty strings, got {key!r}")
            if not isinstance(value, Reference):
                raise TypeError(
                    f"reference['{key}'] must be a Reference, got {type(value).__name__}"
                )
            variants[key] = value if value.name == key else dataclasses.replace(value, name=key)
        return variants
    raise TypeError(
        "reference must be a Reference, a mapping of variant name to Reference, or None; "
        f"got {type(reference).__name__}"
    )


def group_by_frame(
    references: Mapping[str, Reference], *, frame_residues: Optional[str] = None
) -> list[tuple[str, ...]]:
    """Group variants that share one superposition (equal ``Reference.frame_key``).

    The groups are in the order of their first variant, and each group lists its variants in
    the order of ``references``; the first group's first variant is the first variant of the
    call, which defines the frame in which the registry metrics run.
    """
    groups: dict[tuple, list[str]] = {}
    for name, reference in references.items():
        groups.setdefault(reference.frame_key(frame_residues), []).append(name)
    return [tuple(names) for names in groups.values()]


__all__ = [
    "ChainResidue",
    "Partner",
    "RESERVED_REGION_NAMES",
    "Reference",
    "ResidueKey",
    "group_by_frame",
    "normalize_references",
    "polymer_residues",
]
