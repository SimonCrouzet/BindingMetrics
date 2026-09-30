"""Adapter for the output of Protenix (``protenix pred``, ByteDance).

Layout checked on 2026-09-30 by reading the Protenix source at commit 85767b8 (2026-09-21;
``protenix/version.py`` says 2.0.0) and the format report of 2026-09-29. Nothing was run and no
real output was parsed: every statement below comes from the writer code, cited as file:line at
that commit. TO VERIFY: that the tag v2.0.0 writes the same layout, and one real run with
``--need_atom_confidence true`` against this adapter.

    {prediction_dir}/{name}/seed_{S}/predictions/{name}_sample_{r}.cif
    {prediction_dir}/{name}/seed_{S}/predictions/{name}_summary_confidence_sample_{r}.json
    {prediction_dir}/{name}/seed_{S}/predictions/{name}_full_data_sample_{r}.json
    {prediction_dir}/ERR/{name}.txt                       written when a sample fails

``{prediction_dir}`` is the ``-o`` directory of ``protenix pred`` (``runner/dumper.py:105``; the
dataset name is empty, ``runner/inference.py:604``). ``docs/infer_json_format.md`` shows
``<name>/<seed>/<name>_<seed>_sample_0.cif``, which the code does not write and this adapter
does not read.

* ``S`` is the seed value; ``seed_index`` is the 1-based position of the seed directory in the
  numeric order of ``S`` (``seed_9`` before ``seed_10``), so a position beyond the last
  directory finds nothing.
* ``r`` is the rank of the sample by ``ranking_score`` within the seed, from 0 (``dumper.py:
  196-200,231-233``; the command line always ranks, ``runner/inference.py:82-84,189-201``).
  ``sample`` is 1-based, so ``sample=1`` reads ``r=0``: the first sample of a Protenix seed is
  also its best. ``record.extras["rank_index"]`` holds ``r``.
* The summary file is always written. It holds ``plddt`` (mean pLDDT, 0-100), ``gpde``
  (angstrom), ``ptm``, ``iptm``, ``has_clash``, ``ranking_score`` (0.8 ipTM + 0.2 pTM + 0.5
  disorder - 100 clash), ``num_recycles`` and per-chain lists (``sample_confidence.py:94-176``).
  Its values are not rounded.
* Chain lists are indexed by chain position: ``chain_ptm`` and ``chain_iptm`` are
  ``[n_chains]``, ``chain_pair_iptm`` and ``chain_pair_iptm_global`` are ``[n_chains,
  n_chains]``, and ``chain_plddt``, ``chain_gpde`` and their pair versions likewise. Position
  ``i`` is the ``i``-th chain of the structure file in order of first appearance
  (``token_asym_id`` numbers chains that way, ``data/core/parser.py:3199-3226``; chain IDs are
  the ``id`` list of the input JSON, or A, B, ... in entity and copy order,
  ``data/inference/json_to_feature.py:94-160``). ``chain_pair_iptm`` is symmetric with a zero
  diagonal (``sample_confidence.py:531-545``); the adapter drops the diagonal. The keys of
  ``record.chain_ptm`` are the positions as strings (``"0"``), those of
  ``record.chain_pair_iptm`` are ``"0-1"`` in both directions; ``chain_map`` renames the chains
  of the structure, not these keys. ``chain_pair_pae_mean`` and ``chain_pair_pae_min`` exist at
  that commit and, by the report, not in 2.0.0: they are read when present.
* The full-data file is written only with ``--need_atom_confidence true``
  (``dumper.py:258-275``). It holds ``atom_plddt`` (per atom, 0-1, rounded to 2 decimals: a
  resolution of 1 on 0-100), ``token_pair_pae`` and ``token_pair_pde`` (per token pair,
  angstrom, 0-32, rounded to 2 decimals), ``contact_probs`` (not read), ``token_has_frame``,
  ``token_asym_id`` and ``atom_to_token_idx`` (``dumper.py:28-45``,
  ``sample_confidence.py:94-216``). ``pae[i, j]`` is the error of token ``j`` aligned on token
  ``i`` (the row is the alignment frame: the pTM code sums over ``j`` for each row,
  ``sample_confidence.py:423-471``); the file is read as it is, never transposed.
* Without that flag the adapter returns the scalars of the summary, leaves the fields the
  full-data file feeds at NaN or None, and adds a ``reason`` that names the flag. The structure
  file still has the per-atom pLDDT as its B-factor, 0-100 with 2 decimals (``dumper.py:
  134-147,203``; the model always returns the full data, ``protenix/model/protenix.py:648``).
  That is the finer source, but parsing does not open the structure file.
* Tokens: a residue named in the standard set that is not a ligand is one token holding all its
  atoms; every other residue (a modified residue, a ligand, an ion) is one token per atom
  (``data/tokenizer.py:112-154``). ``pae`` and ``pde`` have one row per token, so they are
  larger than the residue count when such a residue is present. The chain and residue of a
  token need the structure, which parsing does not open: ``record.tokens`` is None and the
  interface statistics apply only when the matrix size equals the residue count.

Not provided by Protenix, so NaN or empty: ``bespoke_iptm`` (OpenFold3's), the run time (no
timing file), and ``disorder``, which the model writes as 0 for every sample
(``sample_confidence.py:171``): the written value is kept in
``record.extras["disorder_written"]`` and only a non-zero value reaches ``record.disorder``.
Model-specific values are in ``record.extras`` under the model's own key: ``chain_iptm``,
``chain_pair_iptm_global``, ``chain_plddt``, ``chain_pair_plddt``, ``chain_gpde``,
``chain_pair_gpde``, ``chain_pair_pae_mean``, ``chain_pair_pae_min``, ``num_recycles`` and,
from the full-data file, ``token_asym_id``, ``token_has_frame`` and ``atom_to_token_idx``.

Where the source does not settle a point the adapter refuses instead of guessing. TO VERIFY
against real output: (1) that the atoms of the CIF follow the order of the arrays: the writer
implies it, and ``record.validate(check_structure=True)`` checks the atom count;
(2) the 0-1 scale of ``atom_plddt``, where a value above 1 raises; (3) the shape of every
array, where another shape raises; (4) a chain list of another shape, which is left out and
explained in ``record.reasons``.

The module imports numpy only; the structure file is not opened while parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.record import (
    PredictionFiles,
    PredictionRecord,
)

logger = logging.getLogger(__name__)

_NAN = float("nan")

#: Keys of the summary file that identify it; at least one must be present.
_SUMMARY_CORE_KEYS = ("plddt", "ptm", "iptm", "ranking_score")

#: Keys of the full-data file that identify it; at least one must be present.
_FULL_DATA_CORE_KEYS = ("atom_plddt", "token_pair_pae", "token_pair_pde")

#: Summary entries kept in ``record.extras`` as the model writes them.
_SUMMARY_EXTRAS = (
    "chain_iptm",
    "chain_pair_iptm_global",
    "chain_plddt",
    "chain_pair_plddt",
    "chain_gpde",
    "chain_pair_gpde",
    "chain_pair_pae_mean",
    "chain_pair_pae_min",
    "num_recycles",
)


# ---------------------------------------------------------------------- file helpers


def _seed_key(directory: Path) -> tuple[int, int, str]:
    """Sort key of a ``seed_*`` directory: numeric seeds by value, then any other name."""
    suffix = directory.name[len("seed_") :]
    return (0, int(suffix), "") if suffix.isdecimal() else (1, 0, suffix)


def _seed_directories(query_dir: Path) -> list[Path]:
    if not query_dir.is_dir():
        return []
    return sorted((d for d in query_dir.glob("seed_*") if d.is_dir()), key=_seed_key)


def _failed_sample_reason(prediction_dir: Path, name: str) -> Optional[str]:
    """First line of ``ERR/{name}.txt``, where ``protenix pred`` records why a sample failed.

    The run writes ``{out}/ERR/{name}.txt`` on an input or a model error
    (``runner/inference.py:137,578,630``). Returns None when there is no such file. The file is
    a message with a traceback; only its first non-empty line is kept.
    """
    path = Path(prediction_dir) / "ERR" / f"{name}.txt"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("%s could not be read: %s", path, exc)
        return None
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"Protenix recorded an error for '{name}': {first[:300]}" if first else None


def _read_json_object(path: Path, what: str) -> dict:
    """Read a JSON file that must hold an object; a corrupt file raises ``ValueError``."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: the Protenix {what} file must hold a JSON object, got {type(raw).__name__}"
        )
    return raw


def _number(raw: dict, key: str, path: Path) -> float:
    """A scalar of the summary file: NaN when absent or null, ``ValueError`` when not a number."""
    value = raw.get(key)
    if value is None:
        return _NAN
    if isinstance(value, (int, float)):  # bool is an int: has_clash may be written as true
        return float(value)
    raise ValueError(f"{path}: '{key}' must be a number, got {type(value).__name__}: {value!r}")


def _array(raw: dict, key: str, path: Path, *, ndim: int) -> Optional[np.ndarray]:
    """A numeric array of the full-data file, or None when absent; ``ValueError`` if malformed."""
    value = raw.get(key)
    if value is None:
        return None
    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: '{key}' is not a numeric array of {ndim} dimension(s): {exc}"
        ) from exc
    if array.ndim != ndim:
        raise ValueError(f"{path}: '{key}' must have {ndim} dimension(s), got shape {array.shape}")
    return array


def _square(array: Optional[np.ndarray], key: str, path: Path) -> Optional[np.ndarray]:
    if array is not None and array.shape[0] != array.shape[1]:
        raise ValueError(f"{path}: '{key}' must be square, got shape {array.shape}")
    return array


def _index_vector(raw: dict, key: str, path: Path) -> Optional[np.ndarray]:
    """A one-dimensional integer array (a token or atom index) of the full-data file."""
    array = _array(raw, key, path, ndim=1)
    if array is None:
        return None
    if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
        raise ValueError(f"{path}: '{key}' must hold integers")
    return array.astype(int)


# ---------------------------------------------------------------------- the two JSON files


def parse_summary_confidence(path: Path) -> dict:
    """Parse ``{name}_summary_confidence_sample_{r}.json``.

    Returns:
        ``avg_plddt`` (0-100), ``gpde``, ``ptm``, ``iptm``, ``has_clash``, ``disorder_written``,
        ``ranking_score`` (floats, NaN when absent), ``chain_ptm`` (``{"0": x, ...}``),
        ``chain_pair_iptm`` (``{"0-1": x, ...}``, off-diagonal), ``extras`` (the other chain
        entries and ``num_recycles`` as written) and ``reasons`` (one sentence for each
        chain entry that was left out because its shape was not the expected one).

    Raises:
        ValueError: The file is not a JSON object, has none of the keys of a summary file, or
            holds a scalar that is not a number.
    """
    path = Path(path)
    raw = _read_json_object(path, "summary confidence")
    if not any(key in raw for key in _SUMMARY_CORE_KEYS):
        raise ValueError(
            f"{path}: none of the keys {list(_SUMMARY_CORE_KEYS)} is present, so this does not "
            "look like a Protenix summary_confidence file (layout checked against Protenix "
            "commit 85767b8)"
        )
    reasons: list[str] = []
    chain_ptm: dict[str, float] = {}
    chain_pair_iptm: dict[str, float] = {}

    value = raw.get("chain_ptm")
    if value is not None:
        vector = _try_array(value, ndim=1)
        if vector is None:
            reasons.append("chain_ptm is not a list of numbers; per-chain pTM left out")
        else:
            chain_ptm = {str(i): float(x) for i, x in enumerate(vector)}
    value = raw.get("chain_pair_iptm")
    if value is not None:
        matrix = _try_array(value, ndim=2)
        if matrix is None or matrix.shape[0] != matrix.shape[1]:
            reasons.append("chain_pair_iptm is not a square list of lists; pair ipTM left out")
        else:
            chain_pair_iptm = {
                f"{i}-{j}": float(matrix[i, j])
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if i != j
            }
    return {
        "avg_plddt": _number(raw, "plddt", path),
        "gpde": _number(raw, "gpde", path),
        "ptm": _number(raw, "ptm", path),
        "iptm": _number(raw, "iptm", path),
        "has_clash": _number(raw, "has_clash", path),
        "disorder_written": _number(raw, "disorder", path),
        "ranking_score": _number(raw, "ranking_score", path),
        "chain_ptm": chain_ptm,
        "chain_pair_iptm": chain_pair_iptm,
        "extras": {key: raw[key] for key in _SUMMARY_EXTRAS if key in raw},
        "reasons": reasons,
    }


def _try_array(value: Any, *, ndim: int) -> Optional[np.ndarray]:
    """``value`` as a float array of ``ndim`` dimensions, or None when it is not one."""
    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.ndim == ndim else None


def parse_full_data(path: Path) -> dict:
    """Parse ``{name}_full_data_sample_{r}.json`` (written with ``--need_atom_confidence true``).

    Returns:
        ``plddt_per_atom`` (``(n_atoms,)``, rescaled from 0-1 to 0-100, or None), ``pae`` and
        ``pde`` (``(n_tokens, n_tokens)`` angstrom, or None) and the integer vectors
        ``token_asym_id``, ``token_has_frame`` (``(n_tokens,)``) and ``atom_to_token_idx``
        (``(n_atoms,)``), each None when absent.

    Raises:
        ValueError: The file is not a JSON object, has none of the keys of a full-data file,
            an array has the wrong number of dimensions or is not square, ``atom_plddt`` has a
            value above 1 (the scale is not the 0-1 one this adapter converts), or the arrays
            disagree on the number of tokens or atoms.
    """
    path = Path(path)
    raw = _read_json_object(path, "full data")
    if not any(key in raw for key in _FULL_DATA_CORE_KEYS):
        raise ValueError(
            f"{path}: none of the keys {list(_FULL_DATA_CORE_KEYS)} is present, so this does "
            "not look like a Protenix full_data file (layout checked against Protenix "
            "commit 85767b8)"
        )
    atom_plddt = _array(raw, "atom_plddt", path, ndim=1)
    pae = _square(_array(raw, "token_pair_pae", path, ndim=2), "token_pair_pae", path)
    pde = _square(_array(raw, "token_pair_pde", path, ndim=2), "token_pair_pde", path)
    token_asym_id = _index_vector(raw, "token_asym_id", path)
    token_has_frame = _index_vector(raw, "token_has_frame", path)
    atom_to_token_idx = _index_vector(raw, "atom_to_token_idx", path)

    if atom_plddt is not None:
        finite = atom_plddt[np.isfinite(atom_plddt)]
        if finite.size and (finite.max() > 1.0 + 1e-6 or finite.min() < -1e-6):
            raise ValueError(
                f"{path}: 'atom_plddt' spans {finite.min():g} to {finite.max():g}, not 0-1. "
                "This adapter rescales the 0-1 pLDDT of Protenix commit 85767b8 to 0-100 and "
                "will not guess another scale"
            )
        atom_plddt = 100.0 * atom_plddt

    n_tokens = _agree(
        path,
        "tokens",
        {
            "token_pair_pae": None if pae is None else pae.shape[0],
            "token_pair_pde": None if pde is None else pde.shape[0],
            "token_asym_id": None if token_asym_id is None else len(token_asym_id),
            "token_has_frame": None if token_has_frame is None else len(token_has_frame),
        },
    )
    _agree(
        path,
        "atoms",
        {
            "atom_plddt": None if atom_plddt is None else len(atom_plddt),
            "atom_to_token_idx": None if atom_to_token_idx is None else len(atom_to_token_idx),
        },
    )
    if atom_to_token_idx is not None and atom_to_token_idx.size and n_tokens is not None:
        if atom_to_token_idx.min() < 0 or atom_to_token_idx.max() >= n_tokens:
            raise ValueError(
                f"{path}: 'atom_to_token_idx' points to tokens "
                f"{atom_to_token_idx.min()}..{atom_to_token_idx.max()} but there are "
                f"{n_tokens} tokens"
            )
    return {
        "plddt_per_atom": atom_plddt,
        "pae": pae,
        "pde": pde,
        "token_asym_id": token_asym_id,
        "token_has_frame": token_has_frame,
        "atom_to_token_idx": atom_to_token_idx,
    }


def _agree(path: Path, what: str, sizes: dict[str, Optional[int]]) -> Optional[int]:
    """The common size of the arrays that are present; ``ValueError`` when they differ."""
    present = {key: size for key, size in sizes.items() if size is not None}
    if len(set(present.values())) > 1:
        raise ValueError(f"{path}: the arrays disagree on the number of {what}: {present}")
    return next(iter(present.values()), None)


# ---------------------------------------------------------------------- the adapter


class ProtenixParser(PredictionParser):
    """Reads the output directory of ``protenix pred`` (see the module docstring)."""

    name = "protenix"
    display_name = "Protenix"
    family = "af3"

    def find_files(
        self,
        prediction_dir: Path,
        name: str,
        *,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionFiles:
        """Locate the files of one sample of the input named ``name``.

        ``seed_index`` is the 1-based position of a ``seed_*`` directory in the numeric order
        of the seed values and ``sample`` is 1-based, so ``sample=1`` is the file with
        ``r=0``. A position that does not exist gives a ``PredictionFiles`` with no file.
        """
        directory = Path(prediction_dir)
        seed_dirs = _seed_directories(directory / name)
        if not (1 <= seed_index <= len(seed_dirs)) or sample < 1:
            return PredictionFiles(directory=directory)
        predictions = seed_dirs[seed_index - 1] / "predictions"
        rank = sample - 1

        def _existing(file_name: str) -> Optional[Path]:
            path = predictions / file_name
            return path if path.exists() else None

        return PredictionFiles(
            directory=directory,
            structure=_existing(f"{name}_sample_{rank}.cif"),
            scores=_existing(f"{name}_summary_confidence_sample_{rank}.json"),
            arrays=_existing(f"{name}_full_data_sample_{rank}.json"),
        )

    def parse(
        self,
        files: PredictionFiles,
        *,
        name: str,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionRecord:
        """Read the summary and full-data files into a record.

        A missing file leaves its values at NaN (None for an array) and adds a reason; the
        missing full-data file names ``--need_atom_confidence``. A corrupt file, an array of
        another shape and a pLDDT that is not on 0-1 raise ``ValueError``. ``avg_plddt`` falls
        back to the mean of the per-atom pLDDT when the summary lacks it.
        """
        record = PredictionRecord(
            self.name,
            name,
            seed_index=seed_index,
            sample=sample,
            structure_path=files.structure,
            ranking_score_name="ranking_score",
            files=files,
        )
        record.extras["rank_index"] = sample - 1

        if files.scores is None and files.arrays is None:
            record.reasons.append(
                f"no confidence files found for '{name}' "
                f"(seed index {seed_index}, sample {sample}) in {files.directory}"
            )
            failure = _failed_sample_reason(Path(files.directory), name)
            if failure:
                record.reasons.append(failure)
        elif files.scores is None:
            record.reasons.append("summary confidence file not found")
        if files.arrays is None and files.scores is not None:
            record.reasons.append(
                "full-data file not found; Protenix writes it (per-atom pLDDT, PAE, PDE) only "
                "when the run used '--need_atom_confidence true'. The structure file still has "
                "the per-atom pLDDT as its B-factor"
            )
        if files.structure is None and files.any_found():
            record.reasons.append("structure file not found")

        if files.scores is not None:
            summary = parse_summary_confidence(files.scores)
            record.avg_plddt = summary["avg_plddt"]
            record.gpde = summary["gpde"]
            record.ptm = summary["ptm"]
            record.iptm = summary["iptm"]
            record.has_clash = summary["has_clash"]
            record.ranking_score = summary["ranking_score"]
            record.chain_ptm = summary["chain_ptm"]
            record.chain_pair_iptm = summary["chain_pair_iptm"]
            record.extras.update(summary["extras"])
            record.reasons.extend(summary["reasons"])
            written = summary["disorder_written"]
            record.extras["disorder_written"] = written
            # The model writes an exact 0 for every sample (sample_confidence.py:171): that is
            # a placeholder, not a measured disorder.
            record.disorder = _NAN if not written else written

        if files.arrays is not None:
            full = parse_full_data(files.arrays)
            record.plddt_per_atom = full["plddt_per_atom"]
            record.pae = full["pae"]
            record.pde = full["pde"]
            for key in ("token_asym_id", "token_has_frame", "atom_to_token_idx"):
                if full[key] is not None:
                    record.extras[key] = full[key]
            if record.plddt_per_atom is not None:
                record.extras["plddt_per_atom_source"] = "atom_plddt of the full-data file"
                if np.isnan(record.avg_plddt):
                    record.avg_plddt = float(np.mean(record.plddt_per_atom))

        located = next(iter(files.found().values()), None)
        if located is not None and located.parent.parent.name.startswith("seed_"):
            record.extras["seed_value"] = located.parent.parent.name[len("seed_") :]
        return record
