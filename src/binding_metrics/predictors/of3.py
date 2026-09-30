"""Adapter for the output of OpenFold3 (``run_openfold predict``).

Layout checked against the released source of OpenFold3 v0.5.0 (2026-08-21) and against the
0.3 and 0.4 outputs this toolkit parsed before; the layout did not change between 0.4.0 and
0.5.0. No 0.5.0 run was available: what follows comes from reading the writer
(``openfold3/core/runners/writer.py``) and the option validator, not from running the model.

    {prediction_dir}/{query}/seed_{S}/{query}_seed_{S}_sample_{k}_model.{cif|cif.gz|pdb}
    {prediction_dir}/{query}/seed_{S}/{query}_seed_{S}_sample_{k}_confidences_aggregated.json
    {prediction_dir}/{query}/seed_{S}/{query}_seed_{S}_sample_{k}_confidences.{json|npz}
    {prediction_dir}/{query}/seed_{S}/timing.json                    one per seed directory

* ``S`` is a seed value chosen by OpenFold3, not a position; ``seed_index`` is the 1-based
  position of the seed directory in numeric order of ``S`` (``seed_9`` before ``seed_10``).
  ``k`` counts samples from 1 and is not a ranking.
* The aggregated file holds ``avg_plddt``, ``gpde``, ``ptm``, ``iptm``, ``disorder``,
  ``has_clash``, ``sample_ranking_score`` and the dictionaries ``chain_ptm``,
  ``chain_pair_iptm`` and ``bespoke_iptm``. The chain-pair keys are strings such as
  ``"(A, B)"``, not tuples. ``bespoke_iptm`` is OpenFold3's own and goes to
  ``record.extras``.
* The full confidence file holds ``plddt`` (per atom, 0-100), ``pde`` and ``pae`` (per token,
  angstrom, ``pae[i, j]`` the error of token ``j`` aligned on token ``i``). It exists only
  when the run used ``write_full_confidence_scores`` (the default). ``.npz`` files hold plain
  numeric arrays (float16 by default) and are read without pickle.
* pLDDT is also the B-factor of the structure file. The tokens are one per standard residue
  and one per heavy atom of a ligand or modified residue; the files carry no token layout,
  so ``record.tokens`` is None and the interface statistics apply only when the matrix size
  equals the residue count.
* pTM, ipTM and PAE are always written by 0.4.1 and later (the ``pae_enabled`` preset was
  removed); a missing value means a missing file.
* A query that fails inside OpenFold3 leaves no confidence files and the process still exits
  with status 0; the run's ``summary.txt`` and ``logs/predict_err_rank<N>.log`` say why, and
  the reason is added to ``record.reasons``.

The module imports numpy only (the runner module is imported when a failed query has to be
explained); the structure file is not opened while parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.record import PredictionFiles, PredictionRecord

_NAN = float("nan")

#: Structure file extensions in order of preference.
_STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".pdb")

#: The arrays OpenFold3 writes to the full confidence file; nothing else is read from a
#: ``.npz`` (numpy 1.26 adds a stray ``allow_pickle`` array to files written by OpenFold3).
_FULL_CONFIDENCE_KEYS = ("plddt", "pde", "pae", "gpde")


def _seed_key(directory: Path) -> tuple[int, int, str]:
    """Sort key of a ``seed_*`` directory: numeric seeds by value, then any other name."""
    suffix = directory.name[len("seed_") :]
    return (0, int(suffix), "") if suffix.isdecimal() else (1, 0, suffix)


def _seed_directories(query_dir: Path) -> list[Path]:
    if not query_dir.is_dir():
        return []
    return sorted((d for d in query_dir.glob("seed_*") if d.is_dir()), key=_seed_key)


def _failed_query_reason(output_dir: Path, query_name: str) -> Optional[str]:
    """Why OpenFold3 reported ``query_name`` as failed, from ``summary.txt`` and ``logs/``.

    OpenFold3 exits with status 0 when a query fails inside it, so a query without
    confidence files may have failed rather than never run. Returns None when the run's
    summary does not list the query. The readers live with the runner code.
    """
    from binding_metrics.metrics._openfold_run import _failed_query_reasons

    return _failed_query_reasons(output_dir).get(query_name)


def parse_aggregated_confidences(path: Path) -> dict:
    """Parse ``*_confidences_aggregated.json``: the scalars of the whole complex.

    Returns:
        ``avg_plddt``, ``gpde``, ``ptm``, ``iptm``, ``disorder``, ``has_clash``,
        ``sample_ranking_score`` (floats, NaN when the key is absent) and ``chain_ptm``,
        ``chain_pair_iptm``, ``bespoke_iptm`` (dicts as written: chain IDs, and strings such
        as ``"(A, B)"`` for pairs; ``{}`` when absent).
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    def _f(key):
        val = raw.get(key)
        return float(val) if val is not None else _NAN

    return {
        "avg_plddt": _f("avg_plddt"),
        "gpde": _f("gpde"),
        "ptm": _f("ptm"),
        "iptm": _f("iptm"),
        "disorder": _f("disorder"),
        "has_clash": _f("has_clash"),
        "sample_ranking_score": _f("sample_ranking_score"),
        "chain_ptm": raw.get("chain_ptm", {}),
        "chain_pair_iptm": raw.get("chain_pair_iptm", {}),
        "bespoke_iptm": raw.get("bespoke_iptm", {}),
    }


def parse_full_confidences(path: Path) -> dict:
    """Parse ``*_confidences.json`` or ``*_confidences.npz``.

    Only ``plddt``, ``pde``, ``pae`` and ``gpde`` are read. An ``.npz`` is opened without
    pickle, so a file that holds object arrays raises ``ValueError``.

    Returns:
        ``plddt_per_atom`` (``(n_atoms,)``, 0-100, or None), ``pde`` and ``pae``
        (``(n_tokens, n_tokens)`` angstrom, or None) and ``gpde`` (float, NaN when absent).
    """
    path = Path(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            raw = {}
            for key in _FULL_CONFIDENCE_KEYS:
                if key not in data.files:
                    continue
                try:
                    raw[key] = data[key]
                except ValueError as exc:  # numpy refuses an object array without pickle
                    raise ValueError(
                        f"{path}: array '{key}' cannot be read without pickle. OpenFold3 writes "
                        "plain numeric arrays, so this file is corrupt or was not written by "
                        f"OpenFold3 ({exc})"
                    ) from exc
    else:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    def _arr(key):
        val = raw.get(key)
        if val is None:
            return None
        return np.array(val, dtype=float)

    def _scalar(key):
        val = raw.get(key)
        if val is None:
            return _NAN
        arr = np.asarray(val, dtype=float)
        return float(arr.ravel()[0]) if arr.size > 0 else _NAN

    return {
        "plddt_per_atom": _arr("plddt"),
        "pde": _arr("pde"),
        "pae": _arr("pae"),
        "gpde": _scalar("gpde"),
    }


def parse_timing(path: Path) -> dict:
    """Parse ``timing.json`` (OpenFold3 0.5.0 writes ``{"runtime_s": seconds}``)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class OpenFold3Parser(PredictionParser):
    """Reads the output directory of one OpenFold3 query (see the module docstring)."""

    name = "of3"
    display_name = "OpenFold3"
    family = "af3"

    def find_files(
        self,
        prediction_dir: Path,
        name: str,
        *,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionFiles:
        """Locate the files of one sample of query ``name``.

        ``seed_index`` is the 1-based position of a ``seed_*`` directory in the numeric
        order of the seed values. A position beyond the last directory (or a query with no
        seed directory) is taken as the seed value itself, so ``seed_index=2`` finds
        ``seed_2`` when that is the only directory. ``sample`` is the sample number in the
        file names, counted from 1.
        """
        query_dir = Path(prediction_dir) / name
        seed_dirs = _seed_directories(query_dir)
        if seed_dirs and 1 <= seed_index <= len(seed_dirs):
            seed_dir = seed_dirs[seed_index - 1]
            actual_seed = seed_dir.name[len("seed_") :]
        else:
            actual_seed = str(seed_index)
            seed_dir = query_dir / f"seed_{seed_index}"
        prefix = f"{name}_seed_{actual_seed}_sample_{sample}"

        def _first_existing(stem: str, suffixes) -> Optional[Path]:
            for suffix in suffixes:
                path = seed_dir / f"{stem}{suffix}"
                if path.exists():
                    return path
            return None

        return PredictionFiles(
            directory=Path(prediction_dir),
            structure=_first_existing(f"{prefix}_model", _STRUCTURE_SUFFIXES),
            scores=_first_existing(f"{prefix}_confidences_aggregated", (".json",)),
            arrays=_first_existing(f"{prefix}_confidences", (".json", ".npz")),
            timing=_first_existing("timing", (".json",)),
        )

    def parse(
        self,
        files: PredictionFiles,
        *,
        name: str,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionRecord:
        """Read the aggregated and full confidence files into a record.

        A missing file leaves its values at NaN (None for an array) and adds a reason; a
        corrupt file raises. ``avg_plddt`` falls back to the mean of the per-atom pLDDT when
        the aggregated file does not give it. ``gpde`` comes from the aggregated file only.
        """
        record = PredictionRecord(
            self.name,
            name,
            seed_index=seed_index,
            sample=sample,
            structure_path=files.structure,
            ranking_score_name="sample_ranking_score",
            files=files,
        )

        if files.scores is None and files.arrays is None:
            record.reasons.append(
                f"no confidence files found for query '{name}' "
                f"(seed index {seed_index}, sample {sample}) in {files.directory}"
            )
            failure = _failed_query_reason(Path(files.directory), name)
            if failure:
                record.reasons.append(failure)
        elif files.scores is None:
            record.reasons.append("aggregated confidences file not found")
        elif files.arrays is None:
            record.reasons.append(
                "per-atom confidences file not found; OpenFold3 writes it only when "
                "write_full_confidence_scores is true"
            )

        if files.scores is not None:
            aggregated = parse_aggregated_confidences(files.scores)
            record.avg_plddt = aggregated["avg_plddt"]
            record.gpde = aggregated["gpde"]
            record.ptm = aggregated["ptm"]
            record.iptm = aggregated["iptm"]
            record.disorder = aggregated["disorder"]
            record.has_clash = aggregated["has_clash"]
            record.ranking_score = aggregated["sample_ranking_score"]
            record.chain_ptm = aggregated["chain_ptm"]
            record.chain_pair_iptm = aggregated["chain_pair_iptm"]
            record.extras["bespoke_iptm"] = aggregated["bespoke_iptm"]

        if files.arrays is not None:
            full = parse_full_confidences(files.arrays)
            record.plddt_per_atom = full["plddt_per_atom"]
            record.pde = full["pde"]
            record.pae = full["pae"]
            if record.plddt_per_atom is not None and np.isnan(record.avg_plddt):
                record.avg_plddt = float(np.mean(record.plddt_per_atom))

        if files.timing is not None:
            record.timing = parse_timing(files.timing)

        located = next(iter(files.found().values()), None)
        if located is not None and located.parent.name.startswith("seed_"):
            record.extras["seed_value"] = located.parent.name[len("seed_") :]
        return record
