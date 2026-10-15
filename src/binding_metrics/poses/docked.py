"""``DockedPose``: a structure from a docking or design tool, with no model confidences.

A docking program (HADDOCK, Rosetta, a design pipeline) writes a structure and its own
scores. ``DockedPose`` holds both, the scores as reported; nothing is computed from them. A
bare structure that is neither predicted nor experimental (a hand-displaced decoy, a test
fixture) is a ``DockedPose`` with ``decoy=True`` and no ``program``::

    pose = DockedPose.from_file("rank_1.pdb", program="haddock", scores={"haddock_score": -112.4})
    decoy = DockedPose.from_atoms(shifted_atoms, decoy=True)

Importing this module imports numpy and the standard library only; biotite is imported when
the structure is read.
"""

from __future__ import annotations

import os
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from binding_metrics.poses.pose import SourcedPose, check_scores


@dataclass(frozen=True, eq=False, repr=False)
class DockedPose(SourcedPose):
    """A structure produced by a docking or design tool.

    ``source`` is the only positional argument; every other field is keyword-only. The pose is
    frozen and compares by identity.

    Attributes:
        source: The structure: a path (mmCIF or PDB, possibly gzip-compressed) or a biotite
            ``AtomArray`` of one model. A path is not read until ``atoms()`` is called.
        chain_map: Structure-file chain ID to the user's chain ID; empty means no renaming.
        program: The tool that wrote the structure (``"haddock"``), or None for a bare
            structure.
        scores: The program's own scores, free-form: name to ``float`` or None. NaN becomes
            None. ``native_scores()`` returns a copy of it.
        decoy: True for a structure made wrong on purpose (a displaced peptide, a shuffled
            sequence), to serve as a negative.
        provenance: Free facts, JSON scalars only (a protocol, a run ID, a parameter).
        given_id: The ``id`` the caller chose; None means the default, the file stem (or
            ``"docked"``) and the first 8 hex digits of ``content_hash()``.

    Raises:
        TypeError: If ``source`` is neither a path nor an ``AtomArray``, a score is not a
            number or None, or ``provenance`` holds something other than a JSON scalar.
        ValueError: If ``chain_map`` is not a valid map, or a score is infinite.
    """

    kind = "docked"

    source: Union[str, os.PathLike, Any]
    _: KW_ONLY
    chain_map: dict[str, str] = field(default_factory=dict)
    program: Optional[str] = None
    scores: dict[str, Optional[float]] = field(default_factory=dict)
    decoy: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    given_id: Optional[str] = None

    def __post_init__(self):
        self._check_common()
        if self.program is not None and not (isinstance(self.program, str) and self.program):
            raise TypeError(f"program must be a non-empty string or None, got {self.program!r}")
        if not isinstance(self.decoy, bool):
            raise TypeError(f"decoy must be True or False, got {self.decoy!r}")
        object.__setattr__(self, "scores", check_scores(self.scores, "scores"))

    @classmethod
    def from_file(
        cls,
        path: Union[str, os.PathLike],
        *,
        program: Optional[str] = None,
        scores: Optional[Mapping[str, Optional[float]]] = None,
        decoy: bool = False,
        chain_map: Optional[Mapping[str, str]] = None,
        id: Optional[str] = None,
        **provenance: Any,
    ) -> DockedPose:
        """A docked pose from a structure file; other keyword arguments are provenance.

        Args:
            path: The structure file.
            program, scores, decoy, chain_map: See the class.
            id: The id of the pose; None derives it from the file stem and the content hash.
            **provenance: Free facts about the pose, JSON scalars.

        Raises:
            FileNotFoundError: If ``path`` is not a file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"docked pose file not found: {path}")
        return cls(
            path,
            chain_map=dict(chain_map or {}),
            program=program,
            scores=dict(scores or {}),
            decoy=decoy,
            provenance=provenance,
            given_id=id,
        )

    @classmethod
    def from_atoms(
        cls,
        atoms: Any,
        *,
        program: Optional[str] = None,
        scores: Optional[Mapping[str, Optional[float]]] = None,
        decoy: bool = False,
        chain_map: Optional[Mapping[str, str]] = None,
        id: Optional[str] = None,
        **provenance: Any,
    ) -> DockedPose:
        """A docked pose from an in-memory biotite ``AtomArray``; see ``from_file``.

        The array is not copied: do not modify it afterwards. ``atoms()`` returns it as it
        is, or a renamed copy when ``chain_map`` is not empty.
        """
        return cls(
            atoms,
            chain_map=dict(chain_map or {}),
            program=program,
            scores=dict(scores or {}),
            decoy=decoy,
            provenance=provenance,
            given_id=id,
        )

    def native_scores(self) -> dict[str, Optional[float]]:
        """The program's own scores (``scores``), as a new dictionary."""
        return dict(self.scores)
