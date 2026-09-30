"""Adapter for the output of AlphaFold2, AlphaFold-Multimer and ColabFold.

This part reads pLDDT from the B-factor column of a predicted structure, where AlphaFold2 and
ColabFold write it (0-100, one value per residue, repeated over the atoms of the residue).
``read_bfactor_plddt`` returns the per-atom array. ``load_bfactor_record`` returns a
``PredictionRecord`` for a bare structure, for example a BindCraft complex: ``plddt_per_atom`` and
``avg_plddt`` are set, ``ptm``, ``iptm`` and the other scalars are NaN, ``pae`` and ``pde`` are
None, and ``reasons`` says so. Chains are those of the file, in its order.

* The scale comes from the values: a column whose largest value is at most 1 is multiplied by 100
  (noted in ``extras["bfactor_scale"]``), and a column that is all zero, not finite or outside
  0-100 gives no pLDDT and a reason. Nothing distinguishes an experimental B-factor from a pLDDT,
  so give these readers predicted structures only. The scale that BindCraft and ColabDesign
  write is TO VERIFY.
* A PDB file is read as text, without biotite, and gives the atoms ``load_structure`` returns:
  model 1, the first alternate location of each residue, hetero atoms included. An mmCIF file is
  read with biotite. Either may be gzip-compressed.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Mapping, NamedTuple, Optional

import numpy as np

from binding_metrics.predictors.record import (
    PredictionFiles,
    PredictionRecord,
    check_chain_map,
)

#: Registry key of the adapter and ``PredictionRecord.model`` of its records.
_MODEL = "af2"

#: Values of the ``scale`` argument of the B-factor reader.
_BFACTOR_SCALES = ("auto", "percent", "fraction")

# --------------------------------------------------------------------------- structure files


class _StructureScan(NamedTuple):
    """What the adapter needs from a predicted structure, in the atom order of the file.

    Attributes:
        b_factor: ``(n_atoms,)`` B-factor column.
        residue_of_atom: ``(n_atoms,)`` 0-based residue number of each atom, counting a new
            residue whenever the chain, residue number, insertion code or residue name changes.
        n_residues: Number of residues.
    """

    b_factor: np.ndarray
    residue_of_atom: np.ndarray
    n_residues: int


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _first_altloc_mask(altlocs: list[str], residues: np.ndarray) -> np.ndarray:
    """Keep the atoms without an alternate location and, per residue, those of the first one.

    This is biotite's ``altloc="first"`` (``filter_first_altloc``), so the atoms kept here are
    the atoms ``load_structure`` returns.
    """
    first: dict[int, str] = {}
    for altloc, residue in zip(altlocs, residues):
        if altloc.isalnum() and residue not in first:
            first[int(residue)] = altloc
    return np.array(
        [
            not altloc.isalnum() or altloc == first[int(residue)]
            for altloc, residue in zip(altlocs, residues)
        ],
        dtype=bool,
    )


def _scan_pdb(path: Path) -> _StructureScan:
    """Read the B-factors and the residue of each atom from the ATOM and HETATM records.

    Only model 1 is read (up to the first ``ENDMDL``), as ``load_structure`` does. The parse is
    plain text on the fixed PDB columns, so it needs no biotite and gives the same atoms.
    """
    b_factors: list[float] = []
    altlocs: list[str] = []
    residues: list[int] = []
    residue, previous = -1, None
    with _open_text(path) as handle:
        for number, line in enumerate(handle, start=1):
            kind = line[:6]
            if kind == "ENDMDL":
                break
            if kind not in ("ATOM  ", "HETATM"):
                continue
            line = line.rstrip("\r\n")
            if len(line) < 66:
                raise ValueError(
                    f"{path}: line {number} is an atom record of {len(line)} columns; "
                    "the B-factor field ends at column 66"
                )
            # residue name, chain, residue number with its insertion code
            key = (line[17:20], line[21], line[22:27])
            if key != previous:
                residue += 1
                previous = key
            try:
                b_factors.append(float(line[60:66]))
            except ValueError:
                raise ValueError(
                    f"{path}: line {number} has {line[60:66]!r} in the B-factor field"
                ) from None
            altlocs.append(line[16])
            residues.append(residue)
    b_factor = np.array(b_factors, dtype=float)
    residue_of_atom = np.array(residues, dtype=int)
    if any(altloc != " " for altloc in altlocs):
        keep = _first_altloc_mask(altlocs, residue_of_atom)
        b_factor = b_factor[keep]
        _, residue_of_atom = np.unique(residue_of_atom[keep], return_inverse=True)
    n_residues = int(residue_of_atom.max()) + 1 if residue_of_atom.size else 0
    return _StructureScan(b_factor, residue_of_atom, n_residues)


def _scan_cif(path: Path) -> _StructureScan:
    """The same as ``_scan_pdb`` for an mmCIF file, read with biotite (imported here)."""
    from binding_metrics.metrics._common import import_biotite
    from binding_metrics.utils import backfill_auth_columns

    struc, pdbx, _ = import_biotite("reading the B-factors of an mmCIF structure")
    if path.suffix.lower() == ".gz":
        with _open_text(path) as handle:
            cif = pdbx.CIFFile.read(io.StringIO(handle.read()))
    else:
        cif = pdbx.CIFFile.read(str(path))
    backfill_auth_columns(cif)
    atoms = pdbx.get_structure(cif, model=1, extra_fields=["b_factor"])
    residue_of_atom = np.zeros(atoms.array_length(), dtype=int)
    if residue_of_atom.size:
        residue_of_atom[struc.get_residue_starts(atoms)] = 1
        residue_of_atom = np.cumsum(residue_of_atom) - 1
    n_residues = int(residue_of_atom.max()) + 1 if residue_of_atom.size else 0
    return _StructureScan(np.asarray(atoms.b_factor, dtype=float), residue_of_atom, n_residues)


def _scan_structure(path: Path) -> _StructureScan:
    """B-factors and residues of a PDB or mmCIF file (either may be gzip-compressed).

    Raises:
        ValueError: The file has no atom record, or a record is malformed.
    """
    name = path.name.lower()
    name = name[: -len(".gz")] if name.endswith(".gz") else name
    scan = _scan_cif(path) if name.endswith((".cif", ".mmcif")) else _scan_pdb(path)
    if scan.b_factor.size == 0:
        raise ValueError(f"{path} has no atom records; it is not a PDB or mmCIF structure")
    return scan


def _plddt_from_bfactor(
    b_factor: np.ndarray, scale: str = "auto"
) -> tuple[Optional[np.ndarray], bool, str]:
    """pLDDT (0-100) from a B-factor column.

    Args:
        b_factor: The B-factor of every atom.
        scale: ``"percent"`` (the values are on 0-100), ``"fraction"`` (0-1, multiplied by
            100) or ``"auto"`` (a column whose largest value is at most 1 is 0-1).

    Returns:
        ``(plddt, rescaled, problem)``: ``plddt`` is None when the column cannot be a pLDDT,
        and ``problem`` then says why; ``rescaled`` is True when the values were multiplied.
    """
    if scale not in _BFACTOR_SCALES:
        raise ValueError(f"scale must be one of {_BFACTOR_SCALES}, got {scale!r}")
    values = np.asarray(b_factor, dtype=float)
    if not np.all(np.isfinite(values)):
        return None, False, "the column has a value that is not a finite number"
    if not values.any():
        return None, False, "the column is all zero, so the file carries no pLDDT"
    rescaled = scale == "fraction" or (scale == "auto" and values.max() <= 1.0)
    if rescaled:
        values = values * 100.0
    if values.min() < 0.0 or values.max() > 100.0:
        return (
            None,
            rescaled,
            f"the values run from {values.min():g} to {values.max():g}, outside 0-100, so they are "
            "not pLDDT",
        )
    return values, rescaled, ""


def read_bfactor_plddt(structure_path: str | Path, *, scale: str = "auto") -> np.ndarray:
    """Per-atom pLDDT (0-100) from the B-factor column of a predicted structure.

    AlphaFold2, ColabFold and the models built on them write pLDDT there (one value per
    residue, repeated over its atoms). Give only predicted structures: an experimental B-factor
    column is read as pLDDT without complaint when its values are between 0 and 100.

    Args:
        structure_path: PDB or mmCIF file, possibly gzip-compressed (mmCIF needs biotite).
        scale: ``"auto"`` (a column whose largest value is at most 1 is taken as 0-1 and
            multiplied by 100), ``"percent"`` or ``"fraction"``.

    Returns:
        ``(n_atoms,)`` array in the atom order of the file, as ``load_structure`` reads it.

    Raises:
        ValueError: The file has no atoms, or its B-factor column cannot be a pLDDT (all zero,
            not finite, or outside 0-100).
    """
    path = Path(structure_path)
    plddt, _, problem = _plddt_from_bfactor(_scan_structure(path).b_factor, scale)
    if plddt is None:
        raise ValueError(f"the B-factor column of {path} is not a pLDDT: {problem}")
    return plddt


def _apply_bfactor(
    record: PredictionRecord, scan: _StructureScan, structure: Path, scale: str
) -> None:
    """Fill the pLDDT of ``record`` from the B-factor column; every other value stays unset."""
    plddt, rescaled, problem = _plddt_from_bfactor(scan.b_factor, scale)
    if plddt is None:
        record.reasons.append(f"no pLDDT in the B-factor column of {structure.name}: {problem}")
        return
    record.extras["plddt_source"] = "b_factor"
    record.plddt_per_atom = plddt
    record.avg_plddt = float(plddt.mean())
    if rescaled:
        record.extras["bfactor_scale"] = "0-1, multiplied by 100"
    record.reasons.append(
        f"no confidence file found next to {structure.name}: pLDDT is read from its B-factor "
        "column; pTM, ipTM, PAE and the per-chain values are not available"
    )


def load_bfactor_record(
    structure_path: str | Path,
    *,
    name: Optional[str] = None,
    chain_map: Optional[Mapping[str, str]] = None,
    scale: str = "auto",
) -> PredictionRecord:
    """A record with the pLDDT of one predicted structure, read from its B-factor column.

    For a bare AlphaFold2 or BindCraft complex (PDB or mmCIF) that has no confidence file.
    ``plddt_per_atom`` and ``avg_plddt`` are set; ``ptm``, ``iptm``, ``gpde`` and the other
    scalars are NaN, ``pae`` and ``pde`` are None, and ``reasons`` says so. Chains are those of
    the file, in its order. See the module docstring for the scale rule and its limit.

    Args:
        structure_path: The structure; a missing file gives a record without structure and
            a reason, like every adapter.
        name: The record's name (default: the file name without its suffixes).
        chain_map: Model chain ID to user chain ID, as for ``PredictionParser.load``.
        scale: ``"auto"``, ``"percent"`` or ``"fraction"`` (see ``read_bfactor_plddt``).

    Raises:
        ValueError: The file has no atoms or a malformed atom record, or ``chain_map`` or
            ``scale`` is invalid.
    """
    path = Path(structure_path)
    if scale not in _BFACTOR_SCALES:
        raise ValueError(f"scale must be one of {_BFACTOR_SCALES}, got {scale!r}")
    checked_map = check_chain_map(chain_map) if chain_map else {}
    if name is None:
        name = path.name
        for suffix in (".gz", ".pdb", ".cif", ".mmcif", ".ent"):
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
    if path.is_file():
        files = PredictionFiles(directory=path.parent, structure=path)
        record = _parse_bare(files, name, scale)
    else:
        record = PredictionRecord(_MODEL, name, files=PredictionFiles(directory=path.parent))
        record.reasons.append(f"structure file {path} not found")
    record.chain_map = checked_map
    return record


def _parse_bare(files: PredictionFiles, name: str, scale: str) -> PredictionRecord:
    record = PredictionRecord(_MODEL, name, structure_path=files.structure, files=files)
    record.extras["layout"] = "bare"
    _apply_bfactor(record, _scan_structure(files.structure), files.structure, scale)
    return record
