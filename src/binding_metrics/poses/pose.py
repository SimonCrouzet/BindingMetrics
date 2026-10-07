"""``Pose``: one structure of a complex, whatever produced it, and what every kind shares.

A pose is classified by its origin, and each kind has its own class::

    Pose                    abstract base class (this module)
     +-- PredictedPose      the output of a co-folding model (``binding_metrics.predictors.record``)
     +-- DockedPose         the output of a docking or design tool: no model confidences
     +-- ExperimentalPose   a crystal, NMR or EM structure

What the kinds have in common, and what the metrics and a downstream pipeline may rely on:

* ``kind``: ``"predicted"``, ``"docked"`` or ``"experimental"`` (class attribute).
* ``id``: a stable string. A caller-given id wins; otherwise a predicted pose uses
  ``<model>:<name>:s<seed_index>:<sample>`` and a docked or experimental one uses the file
  stem (the name of the kind for an in-memory structure) and the first 8 hex digits of the
  content hash, ``decoy_01-3f9a1c2e``.
* ``structure_path``: the structure file, or None for an in-memory structure.
* ``atoms()``: the structure as a biotite ``AtomArray`` with the chain IDs of ``chain_map``
  applied. A gzip-compressed file is decompressed first.
* ``chain_map``: renames chains, structure-file ID to the user's ID; empty means no renaming.
* ``provenance``: free facts about the pose (program, method, a decoy flag, the run mode), a
  dictionary of JSON scalars.
* ``content_hash()``: SHA-256 of the file bytes for a path (``file_sha256``), and of a
  canonical serialisation of the atoms for an in-memory structure (``atoms_sha256``). A path
  and the same structure loaded into memory therefore have different hashes.
* ``native_scores()``: what the tool that produced the pose reported, flat, values only
  (``float`` or None for a value that is missing). A computed metric is never stored on a
  pose: metrics are functions of a pose, and their results live in the result dictionaries.

``atoms()`` and ``content_hash()`` of a pose with a path are a view of the file as it is when
they are called: the result is cached with the file's size and modification time, and the
file is read again when either has changed (a rewrite at the same path, a relaxed structure
written back over its input). A call on an unchanged file costs one ``os.stat``. A rewrite
that keeps both the size and the modification time, which a file system with a coarse clock
allows within one tick, is not seen. An in-memory structure is never stale.

Importing this module imports numpy and the standard library only (and
``binding_metrics.predictors.store`` for ``file_sha256``, which imports the standard library
only); biotite is imported when a structure is read.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import math
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from binding_metrics.predictors.store import file_sha256

#: The kinds of pose, by origin.
POSE_KINDS = ("predicted", "docked", "experimental")


class Pose(ABC):
    """One structure of a complex, classified by its origin (see the module docstring).

    A subclass sets ``kind``, provides ``chain_map``, ``provenance`` and ``structure_path``
    (dataclass fields or properties) and implements ``native_scores``. Equality is identity;
    two poses with the same structure are told apart from each other by ``content_hash()``.
    """

    #: One of ``POSE_KINDS``.
    kind: ClassVar[str]

    chain_map: dict[str, str]
    provenance: dict[str, Any]
    structure_path: Optional[Path]

    # ------------------------------------------------------------------ identity

    @property
    def id(self) -> str:
        """The stable id: the id the caller gave, or the default of the kind."""
        given = getattr(self, "given_id", None)
        return given if given else self._default_id()

    def _default_id(self) -> str:
        return f"{self._id_label()}-{self.content_hash()[:8]}"

    def _id_label(self) -> str:
        """The part of the default id before the hash: the file stem, else the kind."""
        path = self.structure_path
        return _stem(path) if path is not None else self.kind

    def content_hash(self) -> str:
        """SHA-256 hex digest that identifies the structure.

        The bytes of the file for a pose with a ``structure_path`` (a compressed file is
        hashed as it is stored), the canonical serialisation of the atoms for an in-memory
        structure (``atoms_sha256``; the chain IDs of the source, before ``chain_map``).
        The digest of a path is cached with the file's size and modification time and
        computed again when either has changed, so it is always that of the file as it is
        now (one ``os.stat`` per call when the file is unchanged).

        Raises:
            ValueError: If the pose has no structure.
            FileNotFoundError: If the structure file does not exist.
        """
        path = self.structure_path
        inline = self._inline_atoms()
        if path is None and inline is None:
            raise ValueError(self._no_structure_message())
        # the stamp is taken before the read: a rewrite during the read leaves an older stamp
        # than the file, so the next call reads again instead of pairing new stamp and old bytes
        key = (str(path), _file_stamp(path)) if path is not None else None
        cached = self.__dict__.get("_content_hash_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        digest = file_sha256(path) if path is not None else atoms_sha256(inline)
        object.__setattr__(self, "_content_hash_cache", (key, digest))
        return digest

    # ------------------------------------------------------------------ structure

    def atoms(self):
        """The structure as a biotite ``AtomArray``, chain IDs renamed by ``chain_map``.

        The array is cached and returned again until ``structure_path``, ``chain_map`` or the
        file's size or modification time changes, when the file is read again (one
        ``os.stat`` per call on an unchanged file); treat the result as read-only and copy it
        before modifying. A call that follows a rewrite returns a new array, and an array
        returned before it keeps the old coordinates. Model 1 is read, with the author chain
        and residue IDs. A gzip-compressed file is decompressed first. An in-memory structure
        is returned as it is when ``chain_map`` is empty and as a renamed copy otherwise; the
        array that was given is never modified.

        Raises:
            ValueError: If there is no structure, or ``chain_map`` names a chain the
                structure does not have or would merge two chains into one.
            FileNotFoundError: If the structure file does not exist (also when an earlier
                call read it).
            ImportError: If biotite is not installed.
        """
        path = self.structure_path
        inline = self._inline_atoms()
        if path is None and inline is None:
            raise ValueError(self._no_structure_message())
        key = (
            str(path) if path is not None else None,
            _file_stamp(path) if path is not None else None,  # before the read, as in content_hash
            tuple(sorted(self.chain_map.items())),
        )
        cached = self.__dict__.get("_atoms_cache")
        if cached is None or cached[0] != key:
            if path is not None:
                atoms = _rename_chains(_read_structure(path), self.chain_map)
            elif self.chain_map:
                atoms = _rename_chains(inline.copy(), self.chain_map)
            else:
                atoms = inline
            object.__setattr__(self, "_atoms_cache", (key, atoms))
            cached = (key, atoms)
        return cached[1]

    def _inline_atoms(self):
        """The in-memory ``AtomArray`` the pose was built from, or None (a file, or nothing)."""
        return None

    def _no_structure_message(self) -> str:
        return f"{self.kind} pose has no structure"

    # ------------------------------------------------------------------ scores

    @abstractmethod
    def native_scores(self) -> dict[str, Optional[float]]:
        """What the tool that produced the pose reported: flat, values only.

        Keys are stable names, values are floats, and a value that is missing is None. A
        reason for a missing value stays in ``reasons`` where the kind has one. The result is
        a new dictionary the caller may change.
        """


class SourcedPose(Pose):
    """Shared part of the kinds that wrap a ``source``: a structure path or an ``AtomArray``.

    ``DockedPose`` and ``ExperimentalPose`` are frozen dataclasses with a ``source`` field and
    call ``_check_common`` from ``__post_init__``; they differ in what they report about the
    structure. Not for direct use.
    """

    source: Any

    @property
    def structure_path(self) -> Optional[Path]:
        """The structure file, or None when ``source`` is an in-memory ``AtomArray``."""
        return self.source if isinstance(self.source, Path) else None

    def _inline_atoms(self):
        return None if isinstance(self.source, Path) else self.source

    def _check_common(self) -> None:
        """Validate and normalise the shared fields of a frozen dataclass kind, in place."""
        fields = {
            "source": check_source(self.source),
            "chain_map": check_chain_map(self.chain_map),
            "provenance": check_provenance(self.provenance),
        }
        if self.given_id is not None and not (isinstance(self.given_id, str) and self.given_id):
            raise TypeError(f"id must be a non-empty string, got {self.given_id!r}")
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        source = repr(str(self.source)) if self.structure_path else _describe_atoms(self.source)
        shown = [
            f"{field.name}={getattr(self, field.name)!r}"
            for field in dataclasses.fields(self)
            if field.name != "source" and getattr(self, field.name) not in (None, {}, False)
        ]
        return f"{type(self).__name__}({', '.join([f'source={source}', *shown])})"


def _describe_atoms(atoms) -> str:
    return f"<in-memory structure, {atoms.array_length()} atoms>"


def check_source(source: Any):
    """Return ``source`` as a ``Path`` (for a path) or unchanged (for an ``AtomArray``).

    Raises:
        TypeError: If ``source`` is neither a path nor a single-model ``AtomArray``.
    """
    if isinstance(source, (str, os.PathLike)):
        return Path(source)
    if hasattr(source, "coord") and hasattr(source, "array_length"):
        if getattr(source.coord, "ndim", 2) != 2:
            raise TypeError(
                "source must be one model: got an AtomArrayStack; take one model with stack[i]"
            )
        return source
    raise TypeError(
        f"source must be a structure path or a biotite AtomArray, got {type(source).__name__}"
    )


# ---------------------------------------------------------------------- structure files


def _stem(path: Path) -> str:
    """File name without ``.gz`` and the structure suffix: ``decoy_01.pdb.gz`` -> ``decoy_01``."""
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return Path(name).stem or name


def _file_stamp(path: Path) -> tuple[int, int]:
    """Size in bytes and modification time in nanoseconds of ``path``: what the caches check."""
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def _read_structure(path: Path):
    """Read model 1 of a structure file; a ``.gz`` file is decompressed to a temporary copy."""
    from binding_metrics.metrics._common import load_structure

    if path.suffix.lower() != ".gz":
        return load_structure(path, purpose="per-chain structural analysis")
    with tempfile.TemporaryDirectory(prefix="bm_prediction_") as scratch:
        plain = Path(scratch) / path.stem  # keeps the inner suffix (.cif or .pdb)
        with gzip.open(path, "rb") as source, open(plain, "wb") as target:
            target.write(source.read())
        return load_structure(plain, purpose="per-chain structural analysis")


def _rename_chains(atoms, chain_map: Mapping[str, str]):
    """Rename chains all at once, so a swap ``{"A": "B", "B": "A"}`` works."""
    if not chain_map:
        return atoms
    unique, inverse = np.unique(atoms.chain_id, return_inverse=True)
    present = [str(chain) for chain in unique]
    unknown = sorted(set(chain_map) - set(present))
    if unknown:
        raise ValueError(
            f"chain_map names chains {unknown} that the structure does not have (it has {present})"
        )
    renamed = [chain_map.get(chain, chain) for chain in present]
    if len(set(renamed)) != len(renamed):
        raise ValueError(
            f"chain_map would merge chains: {dict(zip(present, renamed))} has a repeated ID"
        )
    atoms.chain_id = np.array(renamed)[inverse]
    return atoms


def atoms_sha256(atoms) -> str:
    """SHA-256 hex digest of the canonical serialisation of an ``AtomArray``.

    Per atom, in array order: chain ID, residue number, insertion code, residue name, atom
    name, element, and the coordinates rounded to 0.001 angstrom as int64. Nothing else is
    read (no B-factor, occupancy or charge), so two arrays that differ only in annotations
    outside this list have one digest, and a shift of a coordinate by less than about
    0.0005 angstrom does not change it unless the value sits on a rounding boundary.
    """
    n_atoms = atoms.array_length()
    digest = hashlib.sha256()
    digest.update(b"binding_metrics.atoms.v1\n")
    digest.update(n_atoms.to_bytes(8, "little"))
    for name in ("chain_id", "res_id", "ins_code", "res_name", "atom_name", "element"):
        column = _annotation(atoms, name, n_atoms)
        if name == "res_id":
            block = np.asarray(column, dtype="<i8").tobytes()
        else:
            block = "\x1f".join(str(value) for value in column).encode("utf-8")
        digest.update(len(block).to_bytes(8, "little"))
        digest.update(block)
    scaled = np.asarray(atoms.coord, dtype=np.float64) * 1000.0
    finite = np.isfinite(scaled)
    # a non-finite coordinate has no integer; it hashes as 0 plus a mask, never as a platform cast
    digest.update(np.rint(np.where(finite, scaled, 0.0)).astype("<i8").tobytes())
    digest.update(np.packbits(~finite).tobytes())
    return digest.hexdigest()


def _annotation(atoms, name: str, n_atoms: int):
    """An annotation of the array; a category the array lacks reads as empty strings."""
    if name in atoms.get_annotation_categories():
        return atoms.get_annotation(name)
    return np.full(n_atoms, "", dtype="U1")


# ---------------------------------------------------------------------- validated inputs


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _chain_map_problems(chain_map: Mapping[str, str]) -> list[str]:
    if not isinstance(chain_map, Mapping):
        return ["chain_map must be a mapping of model chain ID to user chain ID"]
    problems: list[str] = []
    for old, new in chain_map.items():
        if not (isinstance(old, str) and isinstance(new, str) and old and new):
            problems.append(f"chain IDs must be non-empty strings, got {old!r} -> {new!r}")
    targets = [new for new in chain_map.values() if isinstance(new, str)]
    duplicated = sorted({t for t in targets if targets.count(t) > 1})
    if duplicated:
        problems.append(f"several chains are renamed to the same ID {duplicated}")
    return problems


def check_chain_map(chain_map: Mapping[str, str]) -> dict[str, str]:
    """Return ``chain_map`` as a plain dict, or raise ``ValueError`` if it cannot be one.

    Every key and value must be a non-empty string, and no two chains may be renamed to the
    same ID (that would merge them).
    """
    problems = _chain_map_problems(chain_map)
    if problems:
        raise ValueError("invalid chain_map: " + "; ".join(problems))
    return dict(chain_map)


def _provenance_problems(provenance: Any) -> list[str]:
    """Problems of a provenance mapping: keys must be strings, values JSON scalars."""
    if not isinstance(provenance, Mapping):
        return ["provenance must be a mapping"]
    problems = []
    for key, value in provenance.items():
        if not isinstance(key, str) or not key:
            problems.append(f"provenance keys must be non-empty strings, got {key!r}")
        elif isinstance(value, float) and not math.isfinite(value):
            problems.append(f"provenance['{key}'] is {value}, which JSON cannot hold")
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            problems.append(
                f"provenance['{key}'] must be a JSON scalar (str, int, float, bool or None), "
                f"got {type(value).__name__}"
            )
    return problems


def check_provenance(provenance: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``provenance`` made of JSON scalars, or raise ``TypeError``."""
    if not provenance:
        return {}
    problems = _provenance_problems(provenance)
    if problems:
        raise TypeError("invalid provenance: " + "; ".join(problems))
    return dict(provenance)


def check_scores(scores: Optional[Mapping[str, Any]], label: str) -> dict[str, Optional[float]]:
    """Return ``scores`` as ``{name: float or None}``; NaN becomes None, other values raise.

    Args:
        scores: Score names to numbers (``int``, ``float``, numpy scalars) or None.
        label: What the mapping is called in the error message.

    Raises:
        TypeError: If a name is not a string or a value is neither a number nor None
            (a ``bool`` is not a number here).
        ValueError: If a value is infinite; use None for a value that is missing.
    """
    if not scores:
        return {}
    if not isinstance(scores, Mapping):
        raise TypeError(f"{label} must be a mapping of name to number or None")
    checked: dict[str, Optional[float]] = {}
    for name, value in scores.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"{label} names must be non-empty strings, got {name!r}")
        if value is None:
            checked[name] = None
        elif not _is_number(value):
            raise TypeError(f"{label}['{name}'] must be a number or None, got {value!r}")
        elif math.isnan(value):
            checked[name] = None
        elif math.isinf(value):
            raise ValueError(f"{label}['{name}'] is infinite; use None for a missing value")
        else:
            checked[name] = float(value)
    return checked
