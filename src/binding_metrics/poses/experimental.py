"""``ExperimentalPose``: a crystal, NMR or cryo-EM structure.

The pose holds the facts the deposition reports (method, resolution, R-free) as reported,
never recomputed::

    pose = ExperimentalPose.from_file(
        "1ycr.cif", method="X-RAY DIFFRACTION", resolution_angstrom=2.2, pdb_id="1YCR",
        quality={"r_free": 0.25},
    )

Importing this module imports numpy and the standard library only; biotite is imported when
the structure is read.
"""

from __future__ import annotations

import os
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from binding_metrics.poses.pose import SourcedPose, _is_number, check_scores

#: The key of the resolution in ``native_scores()``; ``quality`` may not use it.
RESOLUTION_KEY = "resolution_angstrom"


@dataclass(frozen=True, eq=False, repr=False)
class ExperimentalPose(SourcedPose):
    """An experimentally determined structure.

    ``source`` is the only positional argument; every other field is keyword-only. The pose is
    frozen and compares by identity.

    Attributes:
        source: The structure: a path (mmCIF or PDB, possibly gzip-compressed) or a biotite
            ``AtomArray`` of one model.
        chain_map: Structure-file chain ID to the user's chain ID; empty means no renaming.
        method: The experimental method as the deposition states it (``"X-RAY DIFFRACTION"``,
            ``"SOLUTION NMR"``, ``"ELECTRON MICROSCOPY"``), or None.
        resolution_angstrom: The resolution in angstrom, positive, or None (NMR has none).
        pdb_id: The PDB entry ID, or None.
        quality: Other quality values the deposition reports, name to ``float`` or None
            (``{"r_free": 0.25, "r_work": 0.20}``). NaN becomes None.
        provenance: Free facts, JSON scalars only.
        given_id: The ``id`` the caller chose; None means the default, the file stem (or the
            PDB ID, or ``"experimental"``) and the first 8 hex digits of ``content_hash()``.

    Raises:
        TypeError: If ``source`` is neither a path nor an ``AtomArray``, a quality value is
            not a number or None, or ``provenance`` holds something other than a JSON scalar.
        ValueError: If ``chain_map`` is not a valid map, the resolution is not positive and
            finite, or ``quality`` uses the key ``resolution_angstrom``.
    """

    kind = "experimental"

    source: Union[str, os.PathLike, Any]
    _: KW_ONLY
    chain_map: dict[str, str] = field(default_factory=dict)
    method: Optional[str] = None
    resolution_angstrom: Optional[float] = None
    pdb_id: Optional[str] = None
    quality: dict[str, Optional[float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    given_id: Optional[str] = None

    def __post_init__(self):
        self._check_common()
        for label, value in (("method", self.method), ("pdb_id", self.pdb_id)):
            if value is not None and not (isinstance(value, str) and value):
                raise TypeError(f"{label} must be a non-empty string or None, got {value!r}")
        resolution = self.resolution_angstrom
        if resolution is not None:
            if not _is_number(resolution):
                raise TypeError(f"resolution_angstrom must be a number or None, got {resolution!r}")
            if not 0.0 < float(resolution) < float("inf"):
                raise ValueError(
                    f"resolution_angstrom must be positive and finite, got {resolution}"
                )
            object.__setattr__(self, "resolution_angstrom", float(resolution))
        quality = check_scores(self.quality, "quality")
        if RESOLUTION_KEY in quality:
            raise ValueError(
                f"quality may not use the key '{RESOLUTION_KEY}': it is the field of that name"
            )
        object.__setattr__(self, "quality", quality)

    def _id_label(self) -> str:
        if self.structure_path is not None:
            return super()._id_label()
        return self.pdb_id or self.kind

    @classmethod
    def from_file(
        cls,
        path: Union[str, os.PathLike],
        *,
        method: Optional[str] = None,
        resolution_angstrom: Optional[float] = None,
        pdb_id: Optional[str] = None,
        quality: Optional[Mapping[str, Optional[float]]] = None,
        chain_map: Optional[Mapping[str, str]] = None,
        id: Optional[str] = None,
        **provenance: Any,
    ) -> ExperimentalPose:
        """An experimental pose from a structure file; other keyword arguments are provenance.

        Args:
            path: The structure file.
            method, resolution_angstrom, pdb_id, quality, chain_map: See the class.
            id: The id of the pose; None derives it from the file stem and the content hash.
            **provenance: Free facts about the pose, JSON scalars.

        Raises:
            FileNotFoundError: If ``path`` is not a file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"experimental pose file not found: {path}")
        return cls(
            path,
            chain_map=dict(chain_map or {}),
            method=method,
            resolution_angstrom=resolution_angstrom,
            pdb_id=pdb_id,
            quality=dict(quality or {}),
            provenance=provenance,
            given_id=id,
        )

    @classmethod
    def from_atoms(
        cls,
        atoms: Any,
        *,
        method: Optional[str] = None,
        resolution_angstrom: Optional[float] = None,
        pdb_id: Optional[str] = None,
        quality: Optional[Mapping[str, Optional[float]]] = None,
        chain_map: Optional[Mapping[str, str]] = None,
        id: Optional[str] = None,
        **provenance: Any,
    ) -> ExperimentalPose:
        """An experimental pose from an in-memory biotite ``AtomArray``; see ``from_file``.

        The array is not copied: do not modify it afterwards.
        """
        return cls(
            atoms,
            chain_map=dict(chain_map or {}),
            method=method,
            resolution_angstrom=resolution_angstrom,
            pdb_id=pdb_id,
            quality=dict(quality or {}),
            provenance=provenance,
            given_id=id,
        )

    def native_scores(self) -> dict[str, Optional[float]]:
        """The resolution under ``resolution_angstrom`` (None when unknown) and ``quality``.

        A new dictionary; the resolution comes first, then the keys of ``quality`` in order.
        """
        return {RESOLUTION_KEY: self.resolution_angstrom, **self.quality}
