"""Adapter for the output of AlphaFold2, AlphaFold-Multimer and ColabFold.

Layouts checked on 2026-09-29 against ColabFold v1.6.3 (2026-09-14) and AlphaFold2 v2.3.2
(2023-04-05) and main (c77e5d2), by reading their source (``colabfold/batch.py``,
``colabfold/alphafold/extra_ptm.py``, ``run_alphafold.py``, ``alphafold/model/model.py``,
``alphafold/model/confidence.py``, ``alphafold/common/protein.py``) and one real ColabFold
multimer-v3 scores file, and on 2026-09-30 against the source of ColabFold 1.5.4 and
alphafold-colabfold 2.3.6 (a fork of AlphaFold2 2.3) and of OpenMM 8.4. What the text marks
"(1.5.4)" comes from that second reading. No model was run and no real AlphaFold2 result pickle
was seen. What the source does not settle is listed under TO VERIFY at the end; the parser then
refuses a file it does not know, with a message, and does not guess.

Files
-----
``prediction_dir`` is searched, and ``prediction_dir/name`` when it holds none of them. The four
layouts are told apart by their file names.

ColabFold (``colabfold_batch``; ``name`` is the job name)::

    {name}_{unrelaxed|relaxed}_rank_{RRR}_{model_type}_model_{k}_seed_{SSS}.pdb
    {name}_scores_rank_{RRR}_{model_type}_model_{k}_seed_{SSS}.json

AlphaFold2 v2.3.2 (``run_alphafold.py``; its files sit in ``{output_dir}/{fasta_name}/``)::

    {unrelaxed|relaxed}_{model}_pred_{i}.pdb        result_{model}_pred_{i}.pkl
    ranking_debug.json                               timings.json

AlphaFold2 main adds ``confidence_{model}_pred_{i}.json`` and ``pae_{model}_pred_{i}.json``; they
are read when the sample has no pickle. A bare structure ``{name}.pdb`` (or ``.cif``, ``.mmcif``,
``.pdb.gz``, ``.cif.gz``, ``.ent``) is the fourth layout, and so is any of the first three with its
confidence file removed.

Values
------
* ColabFold scores JSON: ``plddt`` (one value per residue, 0-100, two decimals), ``pae`` (residues
  by residues, two decimals), ``max_pae``, ``ptm`` and ``iptm`` (two decimals). A monomer model
  without the pTM head writes no ``pae``, ``ptm`` or ``iptm``, and only the multimer models write
  ``iptm``. ``--calc-extra-ptm`` adds ``per_chain_ptm`` (``{"A": x}``, read as ``chain_ptm``),
  ``pairwise_iptm`` (``{"A-B": x}``, upper triangle only, read as ``chain_pair_iptm``),
  ``pairwise_actifptm`` and ``actifptm``; ColabFold 1.6.3 adds ``ipsae``, ``pdockq`` and
  ``pdockq2`` dictionaries keyed ``"A-B"`` for a complex. The last five go to ``record.extras``
  under their own names.
* AlphaFold2 result pickle: ``plddt`` (per residue, 0-100), ``predicted_aligned_error`` (residues by
  residues, angstrom), ``max_predicted_aligned_error``, ``ptm``, ``iptm`` (multimer only) and
  ``ranking_confidence``. A pickle runs code when it is loaded, so the file is read by an
  unpickler that builds numpy arrays, numpy scalars and dictionaries and refuses any other
  global, naming it, before it is imported. The whole pickle is read, distogram and logits
  included, which takes memory in proportion to the square of the number of residues.
  ``ranking_debug.json`` gives the ranking score under the name AlphaFold2 uses (``iptm+ptm`` for
  the multimer models, ``plddts`` for the others); ``timings.json`` goes to ``record.timing`` as
  it is.
* AlphaFold2 main JSON files: ``confidenceScore`` (per-residue pLDDT) and
  ``predicted_aligned_error`` (list-wrapped, one decimal). pTM and ipTM are not in them.
* Bare structure, or a sample without its confidence file: pLDDT is read from the B-factor column,
  where AlphaFold2 and ColabFold write it (0-100, one value per residue repeated over its atoms),
  and every other value is NaN or None with a reason. ``load_bfactor_record`` does this for one
  file, for example a BindCraft complex. The scale comes from the values: a column whose largest
  value is at most 1 is multiplied by 100 (noted in ``extras["bfactor_scale"]``), and a column that
  is all zero or outside 0-100 gives no pLDDT and a reason. Nothing distinguishes an experimental
  B-factor from a pLDDT, so give this reader predicted structures only.

Not provided, so NaN, None or empty: ``gpde``, ``disorder``, ``has_clash``, PDE (``pde`` is None),
the ranking score of ColabFold (its rank is in the file name, and in ``extras["colabfold_rank"]``),
and ``ptm``, ``iptm`` and PAE for a monomer model without the pTM head, for AlphaFold2 main JSON
files and for a bare structure.

Conventions
-----------
* Tokens are residues: ``pae`` is ``(n_residues, n_residues)`` and ``record.tokens`` is None.
  ``pae[i, j]`` is the error of residue ``j`` when the structures are aligned on residue ``i``
  (AlphaFold2 ``confidence.py:243-244``). The arrays are stored as written, never transposed.
* pLDDT is per atom in the record. The models give one value per residue, which is repeated over
  the atoms of the structure file; ``avg_plddt`` is the mean over residues, as AlphaFold2 and
  ColabFold report it, and the per-residue array stays in ``record.extras["plddt_per_residue"]``.
  The structure file is read as text for a PDB (no biotite) and with biotite for an mmCIF. The
  residue counts of the two files must agree, or ``ValueError`` is raised; a structure file that
  cannot be read leaves ``plddt_per_atom`` None, with a reason.
* Chains are those of the structure file: A, B, ... in input order for AlphaFold2 and, for
  ColabFold, in input order except that identical sequences are placed next to each other, so the
  order can differ from the FASTA order. ``chain_ptm`` and ``chain_pair_iptm`` use the same letters.
  Rename chains with ``chain_map``. A bare structure keeps the chains of its file in the order of
  the file. (1.5.4) A complex predicted with a monomer model is named by the same rule as with a
  multimer model (``protein.from_prediction`` takes the chain from ``asym_id`` in both cases).
* (1.5.4) Relaxed files are written by OpenMM with hydrogens, the chains renamed A, B, ... in order
  and the residues numbered from 1 in each chain (``openmm/app/pdbfile.py``, ``keepIds=False``);
  ``overwrite_b_factors`` then sets the B-factor of every atom, hydrogens included, to the pLDDT of
  its residue. A relaxed file therefore has more atoms and can number its residues differently from
  the unrelaxed one, and the expansion of the per-residue pLDDT over its atoms is unchanged.
* (1.5.4) ColabFold writes ``plddt`` and ``pae`` of the scores file from the model's ``plddt`` and
  ``predicted_aligned_error`` without reordering (``batch.py:478-487``), and renames every file of
  a sample with the same ``rank_{RRR}_{tag}``, the rank counted over all seeds and models
  (``batch.py:508-535``).
* ``seed_index`` is the 1-based position of the seed in the numeric order of its value (ColabFold
  ``seed_{SSS}``, AlphaFold2 ``pred_{i}``), and ``sample`` the 1-based position inside that seed:
  by rank for ColabFold (``rank_001`` first, so with one seed ``sample=1`` is the best model) and
  by model number for AlphaFold2. A bare structure is one sample. The structure of a sample is
  the relaxed one when it exists; the unrelaxed file is then ``files.extra["unrelaxed_structure"]``.

TO VERIFY
---------
1. That ColabFold 1.6.3 does what 1.5.4 does where the text says (1.5.4): the score arrays, the
   rank tags and the chain names. The 1.6.3 source was read only for the file names and keys.
2. The file names of ColabFold versions before 1.5.4 (only ``rank_{RRR}`` and the
   ``_seed_{SSS}`` ending are used).
3. The names ``confidence_*.json`` and ``pae_*.json`` of AlphaFold2 main (with or without
   ``_pred_{i}``), and the names of its mmCIF files (only PDB files are read).
4. That a real result pickle holds numpy data only and was written with pickle protocol 3 to 5;
   the keys of ``timings.json``.
5. That a relaxed file from a real run looks as the code says (the relaxation was not run).
6. The scale of the B-factor column that BindCraft and ColabDesign write.
"""

from __future__ import annotations

import gzip
import importlib
import io
import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional

import numpy as np

from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.record import (
    PredictionFiles,
    PredictionRecord,
    SampleRef,
    check_chain_map,
)

_NAN = float("nan")

#: Registry key of the adapter and ``PredictionRecord.model`` of its records.
_MODEL = "af2"

#: Values of the ``scale`` argument of the B-factor reader.
_BFACTOR_SCALES = ("auto", "percent", "fraction")

#: Structure suffixes of a bare prediction, in order of preference.
_BARE_SUFFIXES = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".ent")

#: Keys that ColabFold adds with ``--calc-extra-ptm`` and 1.6.3 adds for complexes; copied to
#: ``record.extras`` as they are.
_COLABFOLD_EXTRA_KEYS = ("pairwise_actifptm", "actifptm", "ipsae", "pdockq", "pdockq2")

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


# --------------------------------------------------------------------------- confidence files


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise ValueError(f"{path} is not a valid JSON file ({exc})") from exc


def _number(value: Any, label: str, source: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: '{label}' is {value!r}, not a number") from None


def _optional_number(raw: dict, key: str, source: Path) -> float:
    value = raw.get(key)
    return _NAN if value is None else _number(value, key, source)


def _number_dict(value: Any, label: str, source: Path) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{source}: '{label}' must be an object of numbers, got {type(value).__name__}"
        )
    return {str(key): _number(item, f"{label}[{key}]", source) for key, item in value.items()}


def _plddt_per_residue(values: Any, source: Path) -> np.ndarray:
    """A pLDDT list as a float array on 0-100, or ``ValueError``."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: pLDDT is not a list of numbers") from None
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{source}: pLDDT must be a non-empty list, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{source}: pLDDT has a value that is not a finite number")
    if array.min() < 0.0 or array.max() > 100.0:
        raise ValueError(
            f"{source}: pLDDT runs from {array.min():g} to {array.max():g}, outside 0-100"
        )
    if array.max() <= 1.0:
        raise ValueError(
            f"{source}: pLDDT runs from {array.min():g} to {array.max():g}, which is a 0-1 scale; "
            "AlphaFold2 and ColabFold write 0-100, so this file is not one this adapter knows "
            "(TO VERIFY), and the scale is not guessed"
        )
    return array


def _pae_matrix(values: Any, n_residues: int, source: Path) -> np.ndarray:
    """A PAE matrix as a float array, checked against the number of residues."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: PAE is not a matrix of numbers") from None
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{source}: PAE must be a square matrix, got shape {array.shape}")
    if array.shape[0] != n_residues:
        raise ValueError(
            f"{source}: PAE is {array.shape[0]} by {array.shape[0]} but pLDDT has "
            f"{n_residues} residues"
        )
    return array


def parse_colabfold_scores(path: Path) -> dict:
    """Parse ``{job}_scores_rank_*.json`` of ColabFold.

    Returns:
        ``plddt_per_residue`` (``(n_residues,)``, 0-100), ``pae`` (``(n_residues, n_residues)``
        angstrom, or None), ``ptm`` and ``iptm`` (floats, NaN when absent), ``chain_ptm`` and
        ``chain_pair_iptm`` (dicts as written, ``{}`` when absent) and ``extras`` (the keys of
        ``_COLABFOLD_EXTRA_KEYS`` that the file has).

    Raises:
        ValueError: The file is not JSON, is not an object with a ``plddt`` list, has pLDDT
            outside 0-100 or on a 0-1 scale, or a PAE that does not match the pLDDT.
    """
    path = Path(path)
    raw = _read_json(path)
    if not isinstance(raw, dict) or "plddt" not in raw:
        raise ValueError(
            f"{path} is not a ColabFold scores file: it must be a JSON object with a 'plddt' list "
            "(the layout of older ColabFold versions is TO VERIFY)"
        )
    plddt = _plddt_per_residue(raw["plddt"], path)
    return {
        "plddt_per_residue": plddt,
        "pae": None if raw.get("pae") is None else _pae_matrix(raw["pae"], plddt.size, path),
        "ptm": _optional_number(raw, "ptm", path),
        "iptm": _optional_number(raw, "iptm", path),
        "chain_ptm": _number_dict(raw.get("per_chain_ptm"), "per_chain_ptm", path),
        "chain_pair_iptm": _number_dict(raw.get("pairwise_iptm"), "pairwise_iptm", path),
        "extras": {key: raw[key] for key in _COLABFOLD_EXTRA_KEYS if key in raw},
    }


#: The classes a numpy array or scalar needs to be unpickled; nothing else is accepted.
_PICKLE_ALLOWED = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        ("collections", "OrderedDict"),
    }
)


class _ArrayOnlyUnpickler(pickle.Unpickler):
    """An unpickler that builds numpy arrays, numpy scalars and containers, and nothing else.

    ``pickle`` calls any importable callable named in the file, which is how a malicious pickle
    runs code. Here every global other than the numpy array types is refused before it is
    imported, so the load cannot run anything (Python documentation, "Restricting Globals").
    """

    def find_class(self, module: str, name: str):
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(
                f"the pickle asks for {module}.{name}, which is not a numpy array type, and it "
                "was not run. AlphaFold2 result pickles hold numpy arrays only; if this file "
                "is genuine, its layout is TO VERIFY"
            )
        if module.startswith("numpy.") and module.split(".")[-1] in ("multiarray", "numeric"):
            # numpy 2 renamed numpy.core to numpy._core; a pickle from either loads in either
            for package in ("numpy._core", "numpy.core"):
                try:
                    return getattr(
                        importlib.import_module(f"{package}.{module.split('.')[-1]}"), name
                    )
                except ImportError:
                    continue
        return super().find_class(module, name)


def read_result_pickle(path: Path) -> dict:
    """Load an AlphaFold2 ``result_*.pkl`` without running any code from it.

    The whole file is read, including the distogram and confidence logits, which take memory
    in proportion to the square of the number of residues.

    Raises:
        ValueError: The file is not a pickle of a dictionary of numpy data, or it asks for
            anything but numpy arrays (the message names what it asked for).
    """
    path = Path(path)
    with open(path, "rb") as handle:
        try:
            result = _ArrayOnlyUnpickler(handle).load()
        except Exception as exc:  # noqa: BLE001 - unpickling bytes can raise nearly any type; re-raised as ValueError
            raise ValueError(
                f"{path} cannot be read as an AlphaFold2 result pickle: {type(exc).__name__}: {exc}"
            ) from exc
    if not isinstance(result, dict):
        raise ValueError(
            f"{path}: an AlphaFold2 result pickle holds a dictionary, got {type(result).__name__}"
        )
    return result


def _pickle_scalar(raw: dict, key: str, source: Path) -> float:
    value = raw.get(key)
    if value is None:
        return _NAN
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: '{key}' is not a number") from None
    if array.size != 1:
        raise ValueError(f"{source}: '{key}' has {array.size} values, expected one")
    return float(array[0])


def parse_result_pickle(path: Path) -> dict:
    """Parse an AlphaFold2 ``result_{model}_pred_{i}.pkl`` (v2.3.2 layout).

    Returns:
        The keys of ``parse_colabfold_scores``, with ``chain_ptm``, ``chain_pair_iptm`` and
        ``extras`` empty (AlphaFold2 writes none of them), and ``ranking_confidence`` (float,
        NaN when absent).

    Raises:
        ValueError: As ``read_result_pickle``, or the pickle has no ``plddt``.
    """
    path = Path(path)
    raw = read_result_pickle(path)
    if "plddt" not in raw:
        raise ValueError(f"{path} has no 'plddt' entry; it is not an AlphaFold2 result pickle")
    plddt = _plddt_per_residue(raw["plddt"], path)
    pae = raw.get("predicted_aligned_error")
    return {
        "plddt_per_residue": plddt,
        "pae": None if pae is None else _pae_matrix(pae, plddt.size, path),
        "ptm": _pickle_scalar(raw, "ptm", path),
        "iptm": _pickle_scalar(raw, "iptm", path),
        "chain_ptm": {},
        "chain_pair_iptm": {},
        "extras": {},
        "ranking_confidence": _pickle_scalar(raw, "ranking_confidence", path),
    }


def parse_ranking_debug(path: Path) -> dict:
    """Parse ``ranking_debug.json`` of AlphaFold2.

    Returns:
        ``name`` (``"iptm+ptm"`` for the multimer models, ``"plddts"`` for the others, as
        AlphaFold2 names it), ``scores`` (``{prediction id: score}``) and ``order`` (the
        prediction ids, best first).

    Raises:
        ValueError: The file is not JSON or holds neither ranking key.
    """
    path = Path(path)
    raw = _read_json(path)
    for name in ("iptm+ptm", "plddts"):
        if isinstance(raw, dict) and isinstance(raw.get(name), dict):
            order = raw.get("order", [])
            if not isinstance(order, list):
                raise ValueError(f"{path}: 'order' must be a list, got {type(order).__name__}")
            return {
                "name": name,
                "scores": _number_dict(raw[name], name, path),
                "order": [str(item) for item in order],
            }
    raise ValueError(
        f"{path} is not an AlphaFold2 ranking_debug.json: it needs an 'iptm+ptm' or 'plddts' object"
    )


def parse_confidence_json(path: Path) -> np.ndarray:
    """Per-residue pLDDT of ``confidence_{model}.json`` (AlphaFold2 main): ``confidenceScore``."""
    path = Path(path)
    raw = _read_json(path)
    if not isinstance(raw, dict) or "confidenceScore" not in raw:
        raise ValueError(
            f"{path} has no 'confidenceScore' list; it is not an AlphaFold2 confidence file"
        )
    return _plddt_per_residue(raw["confidenceScore"], path)


def parse_pae_json(path: Path, n_residues: int) -> np.ndarray:
    """The PAE matrix of ``pae_{model}.json`` (AlphaFold2 main).

    The file is a list holding one object with ``predicted_aligned_error``, or that object
    itself (ColabFold's ``predicted_aligned_error_v1.json``).
    """
    path = Path(path)
    raw = _read_json(path)
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    if not isinstance(raw, dict) or "predicted_aligned_error" not in raw:
        raise ValueError(
            f"{path} has no 'predicted_aligned_error' matrix; it is not an AlphaFold2 PAE file"
        )
    return _pae_matrix(raw["predicted_aligned_error"], n_residues, path)


# --------------------------------------------------------------------------- finding the files


class _Prediction(NamedTuple):
    """One predicted structure of a job, with the place it takes in the model's order."""

    seed_value: int
    order_key: tuple
    tag: str
    roles: dict[str, Path]


_COLABFOLD_TAG = r"(?P<tag>rank_(?P<rank>\d+)(?:_.*)?)"
_SEED_IN_TAG = re.compile(r"_seed_(\d+)$")
_ALPHAFOLD_ID = r"(?P<id>(?P<model>.+)_pred_(?P<pred>\d+))"
_ALPHAFOLD_STRUCTURE = re.compile(rf"^(?P<kind>unrelaxed|relaxed)_{_ALPHAFOLD_ID}\.pdb$")
_ALPHAFOLD_RESULT = re.compile(rf"^result_{_ALPHAFOLD_ID}\.pkl$")
_MODEL_NUMBER = re.compile(r"^model_(\d+)")

#: File-name patterns that tell the layout of the files of a sample.
_COLABFOLD_NAME = re.compile(r"_(?:unrelaxed|relaxed|scores)_rank_\d+")
_ALPHAFOLD_NAME = re.compile(
    r"^(?:(?:unrelaxed|relaxed|result)_.+_pred_\d+\.(?:pdb|pkl)|ranking_debug\.json)$"
)


def _files_of(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file())


def _colabfold_predictions(directory: Path, name: str) -> list[_Prediction]:
    prefix = re.escape(name)
    structure = re.compile(rf"^{prefix}_(?P<kind>unrelaxed|relaxed)_{_COLABFOLD_TAG}\.pdb$")
    scores = re.compile(rf"^{prefix}_scores_{_COLABFOLD_TAG}\.json$")
    by_tag: dict[str, dict[str, Path]] = {}
    ranks: dict[str, int] = {}
    for path in _files_of(directory):
        match = structure.match(path.name)
        if match:
            role = "structure" if match["kind"] == "relaxed" else "unrelaxed_structure"
        else:
            match = scores.match(path.name)
            role = "scores"
        if match:
            by_tag.setdefault(match["tag"], {})[role] = path
            ranks[match["tag"]] = int(match["rank"])
    predictions = []
    for tag, roles in by_tag.items():
        if "structure" not in roles and "unrelaxed_structure" in roles:
            roles["structure"] = roles.pop("unrelaxed_structure")
        if "scores" in roles:
            roles["arrays"] = roles["scores"]  # one file holds the scalars and the arrays
        seed = _SEED_IN_TAG.search(tag)
        predictions.append(_Prediction(int(seed[1]) if seed else 0, (ranks[tag], tag), tag, roles))
    return predictions


def _alphafold_predictions(directory: Path, name: str) -> list[_Prediction]:
    by_id: dict[str, dict[str, Path]] = {}
    preds: dict[str, int] = {}
    models: dict[str, str] = {}
    files = _files_of(directory)
    for path in files:
        match = _ALPHAFOLD_STRUCTURE.match(path.name)
        if match:
            role = "structure" if match["kind"] == "relaxed" else "unrelaxed_structure"
        else:
            match = _ALPHAFOLD_RESULT.match(path.name)
            role = "arrays"
        if match:
            by_id.setdefault(match["id"], {})[role] = path
            preds[match["id"]] = int(match["pred"])
            models[match["id"]] = match["model"]
    shared = {
        role: directory / filename
        for role, filename in (("scores", "ranking_debug.json"), ("timing", "timings.json"))
        if (directory / filename).is_file()
    }
    predictions = []
    for prediction_id, roles in by_id.items():
        if "structure" not in roles and "unrelaxed_structure" in roles:
            roles["structure"] = roles.pop("unrelaxed_structure")
        roles.update(shared)
        for role, stem in (("pae", "pae_"), ("confidence", "confidence_")):
            candidate = directory / f"{stem}{prediction_id}.json"
            if candidate.is_file():
                roles[role] = candidate
        number = _MODEL_NUMBER.match(models[prediction_id])
        order = (int(number[1]) if number else 10**9, prediction_id)
        predictions.append(_Prediction(preds[prediction_id], order, prediction_id, roles))
    return predictions


def _bare_predictions(directory: Path, name: str) -> list[_Prediction]:
    for suffix in _BARE_SUFFIXES:
        path = directory / f"{name}{suffix}"
        if path.is_file():
            return [_Prediction(0, (1, name), name, {"structure": path})]
    return []


def _discover(directory: Path, name: str) -> list[_Prediction]:
    """The predictions of ``name`` under ``directory`` (or ``directory/name``), one layout.

    ColabFold files are looked for first, then AlphaFold2 files, then a bare structure, so a
    directory that has both a job and a stray ``{name}.pdb`` is read as the job.
    """
    searched = [path for path in (directory, directory / name) if path.is_dir()]
    for enumerate_layout in (_colabfold_predictions, _alphafold_predictions, _bare_predictions):
        for path in searched:
            found = enumerate_layout(path, name)
            if found:
                return found
    return []


def _by_seed(predictions: list[_Prediction]) -> list[list[_Prediction]]:
    """The predictions grouped by seed value (numeric order) and sorted inside a seed."""
    seeds: dict[int, list[_Prediction]] = {}
    for prediction in predictions:
        seeds.setdefault(prediction.seed_value, []).append(prediction)
    return [sorted(seeds[seed], key=lambda p: p.order_key) for seed in sorted(seeds)]


def _select(predictions: list[_Prediction], seed_index: int, sample: int) -> Optional[_Prediction]:
    groups = _by_seed(predictions)
    if not 1 <= seed_index <= len(groups):
        return None
    group = groups[seed_index - 1]
    return group[sample - 1] if 1 <= sample <= len(group) else None


def _files_from(directory: Path, roles: dict[str, Path]) -> PredictionFiles:
    named = ("structure", "scores", "arrays", "timing")
    return PredictionFiles(
        directory=directory,
        extra={role: path for role, path in roles.items() if role not in named},
        **{role: roles[role] for role in named if role in roles},
    )


def _layout_of(files: PredictionFiles) -> str:
    """``"colabfold"``, ``"alphafold"`` or ``"bare"``, from the names of the files found."""
    names = [path.name for path in files.found().values()]
    if any(_COLABFOLD_NAME.search(file_name) for file_name in names):
        return "colabfold"
    if any(_ALPHAFOLD_NAME.match(file_name) for file_name in names):
        return "alphafold"
    return "bare"


# --------------------------------------------------------------------------- the record


def _prediction_id(files: PredictionFiles) -> Optional[str]:
    """The AlphaFold2 prediction id (``model_1_multimer_v3_pred_0``) of the files, if any."""
    for path in (files.arrays, files.structure):
        if path is not None:
            for pattern in (_ALPHAFOLD_RESULT, _ALPHAFOLD_STRUCTURE):
                match = pattern.match(path.name)
                if match:
                    return match["id"]
    return None


def _read_confidences(files: PredictionFiles, record: PredictionRecord) -> Optional[dict]:
    """Read the file that holds the per-residue pLDDT; None (and a reason) when there is none."""
    arrays = files.arrays
    if arrays is not None and arrays.suffix.lower() == ".json":
        confidences = parse_colabfold_scores(arrays)
        confidences["source"] = "colabfold_scores"
        return confidences
    if arrays is not None and arrays.suffix.lower() == ".pkl":
        confidences = parse_result_pickle(arrays)
        confidences["source"] = "result_pickle"
        return confidences
    confidence = files.extra.get("confidence")
    if confidence is not None:
        plddt = parse_confidence_json(confidence)
        pae_file = files.extra.get("pae")
        return {
            "plddt_per_residue": plddt,
            "pae": None if pae_file is None else parse_pae_json(pae_file, plddt.size),
            "ptm": _NAN,
            "iptm": _NAN,
            "chain_ptm": {},
            "chain_pair_iptm": {},
            "extras": {},
            "source": "confidence_json",
        }
    layout = _layout_of(files)
    if layout == "colabfold":
        record.reasons.append(
            "no ColabFold scores file (*_scores_rank_*.json) found for this sample"
        )
    elif layout == "alphafold":
        record.reasons.append(
            "no AlphaFold2 result pickle (result_*_pred_*.pkl) or confidence_*.json found for this "
            "sample; pTM, ipTM and PAE are only written there"
        )
    return None


def _note_missing(record: PredictionRecord, confidences: dict, source_name: str) -> None:
    """One sentence for each value the confidence file does not hold."""
    if confidences["source"] == "confidence_json":
        record.reasons.append(
            f"pTM and ipTM are not in {source_name}; AlphaFold2 writes them only in the result "
            "pickle, which was not found"
        )
        if confidences["pae"] is None:
            record.reasons.append("no pae_*.json found: PAE is not available")
        return
    absent = [
        label
        for label, missing in (
            ("PAE", confidences["pae"] is None),
            ("pTM", np.isnan(confidences["ptm"])),
        )
        if missing
    ]
    if absent:
        record.reasons.append(
            f"{' and '.join(absent)} not in {source_name}: a monomer model without the pTM head "
            "does not compute them"
        )
    if np.isnan(confidences["iptm"]):
        record.reasons.append(f"ipTM not in {source_name}: only the multimer models compute it")


def _apply_confidences(record: PredictionRecord, confidences: dict, source_name: str) -> None:
    plddt = confidences["plddt_per_residue"]
    record.avg_plddt = float(plddt.mean())
    record.ptm = confidences["ptm"]
    record.iptm = confidences["iptm"]
    record.pae = confidences["pae"]
    record.chain_ptm = confidences["chain_ptm"]
    record.chain_pair_iptm = confidences["chain_pair_iptm"]
    record.extras.update(confidences["extras"])
    record.extras["plddt_per_residue"] = plddt
    record.extras["plddt_source"] = confidences["source"]
    _note_missing(record, confidences, source_name)


def _expand_per_residue(plddt: np.ndarray, scan: _StructureScan, structure: Path) -> np.ndarray:
    """One pLDDT per atom from one per residue; the residue counts must agree."""
    if plddt.size != scan.n_residues:
        raise ValueError(
            f"the confidence file has {plddt.size} pLDDT values but {structure} has "
            f"{scan.n_residues} residues: the files are not from the same prediction (or the "
            "structure holds ligands or waters, which AlphaFold2 does not predict)"
        )
    return plddt[scan.residue_of_atom]


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


def _parse_files(
    files: PredictionFiles, name: str, seed_index: int, sample: int, scale: str = "auto"
) -> PredictionRecord:
    record = PredictionRecord(
        _MODEL,
        name,
        seed_index=seed_index,
        sample=sample,
        structure_path=files.structure,
        files=files,
    )
    if not files.has_output():
        record.reasons.append(
            f"no AlphaFold2 or ColabFold output found for '{name}' (seed index {seed_index}, "
            f"sample {sample}) in {files.directory}"
        )
        return record
    record.extras["layout"] = _layout_of(files)

    confidences = _read_confidences(files, record)
    scan, structure_problem = None, "no structure file found"
    if files.structure is not None:
        try:
            scan = _scan_structure(files.structure)
        except ValueError as exc:
            structure_problem = f"the structure file cannot be read ({exc})"
    if confidences is not None:
        source_file = files.arrays or files.extra["confidence"]
        _apply_confidences(record, confidences, source_file.name)
        if scan is not None:
            record.plddt_per_atom = _expand_per_residue(
                confidences["plddt_per_residue"], scan, files.structure
            )
        else:
            record.reasons.append(
                f"{structure_problem}: the per-residue pLDDT (record.extras"
                "['plddt_per_residue']) cannot be expanded to atoms"
            )
    elif scan is not None:
        _apply_bfactor(record, scan, files.structure, scale)
    else:
        record.reasons.append(f"{structure_problem}, so there is no pLDDT either")

    _apply_layout_details(record, files, confidences)
    return record


def _apply_layout_details(
    record: PredictionRecord, files: PredictionFiles, confidences: Optional[dict]
) -> None:
    """Ranking score, timing and file names that belong to one layout."""
    layout = record.extras["layout"]
    if layout == "colabfold" and files.scores is not None:
        rank = re.search(r"_rank_(\d+)", files.scores.name)
        if rank:
            record.extras["colabfold_rank"] = int(rank[1])
    if layout == "alphafold":
        prediction_id = _prediction_id(files)
        record.extras["prediction_id"] = prediction_id
        ranking = None if files.scores is None else parse_ranking_debug(files.scores)
        if ranking is not None and prediction_id in ranking["scores"]:
            record.ranking_score = ranking["scores"][prediction_id]
            record.ranking_score_name = ranking["name"]
        elif confidences is not None and "ranking_confidence" in confidences:
            if np.isfinite(confidences["ranking_confidence"]):
                record.ranking_score = confidences["ranking_confidence"]
                record.ranking_score_name = "ranking_confidence"
        if files.timing is not None:
            timing = _read_json(files.timing)
            if not isinstance(timing, dict):
                raise ValueError(f"{files.timing} must hold a JSON object of run times")
            record.timing = timing


class AlphaFold2Parser(PredictionParser):
    """Reads AlphaFold2, AlphaFold-Multimer and ColabFold output (see the module docstring)."""

    name = _MODEL
    display_name = "AlphaFold2 / ColabFold"
    family = "af2"
    not_provided = frozenset({"pde", "gpde", "disorder", "has_clash"})

    def find_files(
        self,
        prediction_dir: Path,
        name: str,
        *,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionFiles:
        """Locate the files of one sample of job or prediction ``name``.

        ``seed_index`` and ``sample`` are 1-based positions (see the module docstring); a
        position beyond the last one finds nothing.
        """
        directory = Path(prediction_dir)
        chosen = _select(_discover(directory, name), seed_index, sample)
        if chosen is None:
            return PredictionFiles(directory=directory)
        return _files_from(directory, chosen.roles)

    def parse(
        self,
        files: PredictionFiles,
        *,
        name: str,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionRecord:
        """Read the located files into a record.

        A missing confidence file leaves its values at NaN (None for an array), adds a reason
        and, when the structure is there, reads pLDDT from its B-factor column. A structure file
        that cannot be read leaves ``plddt_per_atom`` None with a reason. A corrupt confidence
        file raises ``ValueError``, and so do confidence and structure files that disagree on the
        number of residues.
        """
        return _parse_files(files, name, seed_index, sample)

    def list_samples(self, prediction_dir: str | Path, name: str) -> list[SampleRef]:
        """The samples present, by seed position then sample position, without parsing them.

        The ranking score is AlphaFold2's from ``ranking_debug.json``, and NaN for ColabFold
        (its rank is in the file name).
        """
        directory = Path(prediction_dir)
        predictions = _discover(directory, name)
        ranking = None
        for prediction in predictions:
            if (
                "scores" in prediction.roles
                and prediction.roles["scores"].name == "ranking_debug.json"
            ):
                ranking = parse_ranking_debug(prediction.roles["scores"])["scores"]
                break
        refs = []
        for seed_index, group in enumerate(_by_seed(predictions), start=1):
            for sample, prediction in enumerate(group, start=1):
                score = _NAN if ranking is None else ranking.get(prediction.tag, _NAN)
                refs.append(SampleRef(seed_index, sample, score))
        return refs


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
