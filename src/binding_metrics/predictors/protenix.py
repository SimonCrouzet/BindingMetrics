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
  diagonal (``sample_confidence.py:531-545``); the adapter drops the diagonal. ``parse`` keys
  ``record.chain_ptm`` by the positions as strings (``"0"``) and ``record.chain_pair_iptm`` by
  ``"0-1"`` in both directions; :meth:`ProtenixParser.complete` renames them to the chain IDs
  of the structure (``"A"``, ``"A-B"``) and keeps the position-keyed dictionaries in
  ``extras["chain_ptm_by_position"]`` and ``["chain_pair_iptm_by_position"]``. ``chain_map``
  renames the atoms, and these keys stay the model's own chain IDs, as in the other adapters.
  ``chain_pair_pae_mean`` and ``chain_pair_pae_min`` exist at that commit and, by the report,
  not in 2.0.0: they are read when present.
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
* Tokens: a residue named in ``STANDARD_RESIDUE_NAMES`` that is not a ligand is one token
  holding all its atoms; every other residue (a modified residue, a ligand, an ion) is one
  token per atom (``data/tokenizer.py:112-154``). ``pae`` and ``pde`` have one row per token, so
  they are larger than the residue count when such a residue is present. The chain and residue
  of a token need the structure, which parsing does not open: ``load`` leaves ``record.tokens``
  None. :meth:`ProtenixParser.complete` builds the layout with :func:`token_layout`, and is
  called by ``compute_prediction_metrics`` and ``PredictionSession.record``; a record that
  only went through ``load`` has the interface statistics only when the matrix size equals the
  residue count.

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
implies it, and ``record.validate(check_structure=True)`` checks the atom count,
and :func:`token_layout` checks the chains;
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

from binding_metrics.capabilities import Capabilities
from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.record import (
    PredictionFiles,
    PredictionRecord,
    SampleRef,
    TokenLayout,
)

logger = logging.getLogger(__name__)

_NAN = float("nan")

#: Bound on the samples ``list_samples`` returns, as in the base class.
_MAX_SAMPLES = 1000

#: Residue names tokenised as one token per residue (``protenix/data/constants.py:270-314``:
#: the 20 amino acids and UNK, RNA A G C U N, DNA DA DG DC DT DN). Any other residue is
#: tokenised per atom.
STANDARD_RESIDUE_NAMES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK "
    "A G C U N DA DG DC DT DN".split()
)

#: Atoms that stand for a residue token: the C-alpha of a protein residue and the C1' of a
#: nucleotide (the centre atom of ``data/core/featurizer.py:150-175``).
_CENTRE_ATOM_NAMES = ("CA", "C1'")

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

    # The inputs Protenix 2.0.0 can be given, from its documentation (commit 85767b8, checked
    # against the clone on 2026-09-30; tests/test_pre_protenix_limits.py re-reads it when the
    # clone is at hand). No closure is refused: the documentation gives a head-to-tail amide bond
    # and a disulfide as the supported cases and calls other polymer-polymer bonds "not reliably
    # handled", a reliability statement, so they are warnings and the source takes any atom pair.
    # Not declared because nothing shows it: a limit on the residue classes (modified residues go
    # through a CCD code, D-amino acids and N-methyl are not mentioned), and the 2560-token limit
    # of the model protenix-v2, which is a limit of the whole complex.
    capabilities = Capabilities(
        caveats={
            family: (
                "Protenix documents a covalent bond between two polymer residues only for a "
                "head-to-tail amide bond and a disulfide between cysteines; other types "
                '"can still be specified in the input, but they are not reliably handled by the '
                "current model. In such cases, the specified residues may tend to be positioned "
                'in close proximity, though typically not close enough to form a covalent bond" '
                "(docs/infer_json_format.md, section covalent_bonds)."
            )
            for family in ("closures:lactam", "closures:staple", "closures:other")
        },
        version="2.0.0",
    )

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

    def complete(self, record: PredictionRecord) -> PredictionRecord:
        """Attach the token layout and name the chains, both of which need the structure file.

        * ``record.tokens`` becomes :func:`token_layout` of the record. Without it the interface
          PAE and PDE of a prediction with a modified residue, ligand or ion are refused,
          because the matrices then have more rows than the structure has residues.
        * ``record.chain_ptm`` and ``record.chain_pair_iptm``, which parsing keys by chain
          position (``"0"``, ``"0-1"``), are keyed by chain ID like those of the other
          adapters: position ``i`` is the ``i``-th chain of the structure file in order of first
          appearance, by the model's own chain IDs (before ``record.chain_map``, which renames
          the atoms and not these keys). The dictionaries as parsed are kept in
          ``record.extras["chain_ptm_by_position"]`` and ``["chain_pair_iptm_by_position"]``.

        A record without a structure is returned as it is. When the structure cannot be read, or
        the layout cannot be built (the files are not from one sample, the chain numbering
        disagrees), ``tokens`` stays None, the keys stay positions, and a reason says so.
        A structure cannot be read without biotite, which leaves the record as it is: the
        analysis that needs the structure reports it. The call is idempotent.
        """
        if (
            record.model != self.name
            or record.structure_path is None
            or "chain_ptm_by_position" in record.extras
            or (record.tokens is not None and not (record.chain_ptm or record.chain_pair_iptm))
        ):
            return record
        try:
            atoms = record.atoms()
        except ImportError:
            return record
        except Exception as exc:  # noqa: BLE001 - external structure file; recorded as a reason
            self._say(
                record,
                "structure could not be read, so the token layout is not built and the chain "
                f"keys stay positions: {exc}",
            )
            return record

        layout_built = record.tokens is not None
        if not layout_built and "atom_to_token_idx" in record.extras:
            try:
                record.tokens = token_layout(record)
                layout_built = True
            except (ValueError, OSError) as exc:
                self._say(
                    record,
                    "token layout not built, so the interface blocks are cut by residue count "
                    f"and the chain keys stay positions: {exc}",
                )
        # The chain order is checked against token_asym_id inside token_layout; without the
        # full-data file there is nothing to check it against and the documented rule applies.
        if layout_built or "atom_to_token_idx" not in record.extras:
            self._name_chains(record, atoms)
        return record

    @staticmethod
    def _say(record: PredictionRecord, reason: str) -> None:
        if reason not in record.reasons:  # idempotent
            record.reasons.append(reason)

    def _name_chains(self, record: PredictionRecord, atoms) -> None:
        """Key ``chain_ptm`` and ``chain_pair_iptm`` by the chain IDs of the structure file."""
        if not (record.chain_ptm or record.chain_pair_iptm):
            return
        to_model = {user: model for model, user in record.chain_map.items()}
        order = list(dict.fromkeys(to_model.get(str(c), str(c)) for c in atoms.chain_id))
        try:
            named_ptm = {order[int(k)]: v for k, v in record.chain_ptm.items()}
            named_pairs = {}
            for key, value in record.chain_pair_iptm.items():
                first, second = key.split("-")
                named_pairs[f"{order[int(first)]}-{order[int(second)]}"] = value
        except (IndexError, ValueError):
            positions = sorted(
                {int(p) for k in (*record.chain_ptm, *record.chain_pair_iptm) for p in k.split("-")}
            )
            self._say(
                record,
                f"chain_ptm and chain_pair_iptm name chain positions {positions} but the structure "
                f"has {len(order)} chains {order}; the keys stay positions",
            )
            return
        record.extras["chain_ptm_by_position"] = record.chain_ptm
        record.extras["chain_pair_iptm_by_position"] = record.chain_pair_iptm
        record.chain_ptm = named_ptm
        record.chain_pair_iptm = named_pairs

    def list_samples(self, prediction_dir: str | Path, name: str) -> list[SampleRef]:
        """The samples present, in seed order and rank order, with their ranking scores.

        Reads only the small summary files: the full-data files hold three token-by-token
        matrices as text, which the default implementation would parse for every sample.
        """
        directory = Path(prediction_dir)
        refs: list[SampleRef] = []
        seed_index = 0
        while len(refs) < _MAX_SAMPLES:
            seed_index += 1
            sample = 0
            found_in_seed = False
            while len(refs) < _MAX_SAMPLES:
                sample += 1
                files = self.find_files(directory, name, seed_index=seed_index, sample=sample)
                if not files.has_output():
                    break
                found_in_seed = True
                score = _NAN
                if files.scores is not None:
                    raw = _read_json_object(files.scores, "summary confidence")
                    score = _number(raw, "ranking_score", files.scores)
                refs.append(SampleRef(seed_index, sample, score))
            if not found_in_seed:
                break
        return refs


# ---------------------------------------------------------------------- token layout


def token_layout(record: PredictionRecord) -> TokenLayout:
    """The token layout of a Protenix record, from its full data and its structure.

    ``pae`` and ``pde`` have one row per token, and a modified residue or ligand is one token
    per atom, so the chain blocks of the matrices are found from ``atom_to_token_idx`` (which
    tokens the atoms of the structure belong to) and the structure (which chain and residue
    each atom has). Assign the result to ``record.tokens`` before ``summarize_prediction``::

        record = get_parser("protenix").load(directory, name)
        record.tokens = token_layout(record)

    Each token is represented by its first atom, except a token of several atoms (a standard
    residue) that is represented by its C-alpha (C1' for a nucleotide). ``is_atom_token`` is True
    for a token of a residue that is not in ``STANDARD_RESIDUE_NAMES`` or is a hetero residue.
    ``chain_id`` is the chain ID of the structure file, before ``record.chain_map``.
    ``extras`` holds ``token_asym_id`` and ``token_has_frame`` when the file has them.

    Raises:
        ValueError: The record is not a Protenix record, was parsed without the full-data file
            (which holds ``atom_to_token_idx``), or the structure does not fit the arrays: a
            different number of atoms, atoms of a token that are not consecutive, or a chain
            order that disagrees with ``token_asym_id``.
        ImportError: biotite is not installed.
    """
    if record.model != ProtenixParser.name:
        raise ValueError(f"token_layout needs a Protenix record, got model '{record.model}'")
    atom_to_token = record.extras.get("atom_to_token_idx")
    if atom_to_token is None:
        raise ValueError(
            "the record has no atom_to_token_idx: it was parsed without the full-data file, "
            "which Protenix writes only when run with '--need_atom_confidence true'"
        )
    atoms = record.atoms()
    n_atoms = atoms.array_length()
    if len(atom_to_token) != n_atoms:
        raise ValueError(
            f"atom_to_token_idx has {len(atom_to_token)} entries but the structure has "
            f"{n_atoms} atoms; the files are not from the same sample"
        )
    if record.pae is not None:
        n_tokens = int(record.pae.shape[0])
    else:
        n_tokens = int(atom_to_token.max()) + 1 if len(atom_to_token) else 0
    if len(atom_to_token) and atom_to_token.max() >= n_tokens:
        raise ValueError(
            f"atom_to_token_idx points to token {int(atom_to_token.max())} but the matrices "
            f"have {n_tokens} tokens"
        )
    if np.any(np.diff(atom_to_token) < 0):
        raise ValueError("the atoms of a token are not consecutive in the structure file")
    tokens = np.arange(n_tokens)
    first_atom = np.searchsorted(atom_to_token, tokens, side="left")
    in_range = first_atom < n_atoms
    has_atoms = np.zeros(n_tokens, dtype=bool)
    has_atoms[in_range] = atom_to_token[first_atom[in_range]] == tokens[in_range]
    if not np.all(has_atoms):
        raise ValueError(
            f"{int((~has_atoms).sum())} of the {n_tokens} tokens have no atom in the structure"
        )
    atoms_per_token = np.bincount(atom_to_token, minlength=n_tokens)

    to_model_chain = {user: model for model, user in record.chain_map.items()}
    chain_ids = np.array(
        [to_model_chain.get(str(c), str(c)) for c in atoms.chain_id[first_atom]], dtype=object
    )
    _check_chain_order(record, chain_ids)

    atom_index = first_atom.copy()
    centre = np.isin(np.asarray(atoms.atom_name), _CENTRE_ATOM_NAMES)
    for token in np.flatnonzero(atoms_per_token > 1):
        span = slice(first_atom[token], first_atom[token] + atoms_per_token[token])
        hits = np.flatnonzero(centre[span])
        if hits.size:
            atom_index[token] = first_atom[token] + hits[0]

    hetero = (
        np.asarray(atoms.hetero, dtype=bool)
        if "hetero" in atoms.get_annotation_categories()
        else np.zeros(n_atoms, dtype=bool)
    )
    residue_names = np.asarray(atoms.res_name)[first_atom]
    is_atom_token = ~np.isin(residue_names, list(STANDARD_RESIDUE_NAMES)) | hetero[first_atom]

    extras = {
        key: record.extras[key]
        for key in ("token_asym_id", "token_has_frame")
        if key in record.extras
    }
    return TokenLayout(
        chain_id=chain_ids.astype(str),
        res_id=np.asarray(atoms.res_id)[first_atom],
        atom_index=atom_index,
        is_atom_token=is_atom_token,
        extras=extras,
    )


def _check_chain_order(record: PredictionRecord, token_chain_ids: np.ndarray) -> None:
    """Refuse a structure whose chain order is not the one ``token_asym_id`` numbers.

    ``token_asym_id`` is the position of the chain in order of first appearance in the atoms
    (``data/core/parser.py:3199-3226``). Guessing which chain a token belongs to from the
    structure alone is safe only when that holds, so it is checked instead of assumed.
    """
    asym = record.extras.get("token_asym_id")
    if asym is None:
        return
    order = list(dict.fromkeys(token_chain_ids.tolist()))
    expected = np.array([order.index(chain) for chain in token_chain_ids.tolist()])
    if not np.array_equal(expected, asym):
        raise ValueError(
            "token_asym_id does not match the chain order of the structure file (chains "
            f"{order} in order of first appearance); the structure and the full-data file are "
            "not from the same sample, or Protenix numbers chains differently from the "
            "layout this adapter was checked against (TO VERIFY)"
        )
