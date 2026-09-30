"""The neutral record of one structure-prediction sample, and the objects that locate it.

A predictor adapter (``binding_metrics.predictors.base.PredictionParser``) turns the files
one model wrote for one sample into a ``PredictionRecord``; the confidence metrics and the
EvoBind check read only the record. The rules that every adapter converts to:

* pLDDT is per atom, in the atom order of ``structure_path``, on a 0-100 scale. A model that
  writes 0-1 values, or one value per residue or token, is rescaled and expanded by its
  adapter.
* PAE and PDE are ``(n_tokens, n_tokens)`` arrays in angstrom. ``pae[i, j]`` is the expected
  position error of token ``j`` when the structure is aligned on token ``i`` (the row is the
  alignment frame, the column the scored token). PDE has no alignment frame.
* ``ptm`` and ``iptm`` are on 0-1, ``gpde`` in angstrom. Their definitions differ between
  models, so a value is compared only within one model.
* A scalar the model does not provide is NaN, never None. A file that is absent leaves the
  fields it feeds at NaN (or None for an array) and adds a sentence to ``reasons``; a file
  that is present but corrupt raises.
* ``chain_map`` renames chains: it maps a chain ID of the model's structure file to the ID
  the user's input uses. An empty map means no renaming. ``chain_ptm`` and
  ``chain_pair_iptm`` keep the keys as the model writes them.

Extension points, for a model whose output has something the record has no field for:

* a scalar or dictionary of that model goes in ``PredictionRecord.extras`` under the key the
  model itself uses (``bespoke_iptm`` for OpenFold3, ``ligand_iptm`` for Boltz-2). Generic
  code never reads ``extras``; a metric that needs a value from it names the model.
* a per-token array of that model goes in ``TokenLayout.extras`` (same length as the token
  list), and an additional file of a sample in ``PredictionFiles.extra``.
* a field that every model could fill is added to ``PredictionRecord`` as the last keyword
  argument (``files`` is the latest), with NaN, None or an empty container as the default, so
  no adapter has to change.
* how a value was obtained (for example that the pLDDT came from the B-factor column, or the
  model version the layout was checked against) goes in ``extras``; a reason that a value is
  missing goes in ``reasons``.
* ``structure_path`` may point to a gzip-compressed file (``.cif.gz``, ``.pdb.gz``);
  ``atoms()`` reads it.

Importing this module imports numpy and the (lazy) structure loader only; biotite is
imported when ``atoms()`` reads a structure.
"""

from __future__ import annotations

import gzip
import tempfile
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from binding_metrics.metrics._common import load_structure

_NAN = float("nan")

#: Roles of the files of a sample that ``PredictionFiles`` names itself.
_FILE_ROLES = ("structure", "scores", "arrays", "timing")


@dataclass(frozen=True)
class PredictionFiles:
    """Where an adapter found the files of one sample. A file that does not exist is None.

    Attributes:
        directory: The prediction directory that was searched.
        structure: The predicted structure (mmCIF or PDB, possibly gzip-compressed).
        scores: The file with the scalar confidence summary (pTM, ipTM, ranking score, ...).
        arrays: The file with the per-atom and per-token arrays (pLDDT, PAE, PDE) when the
            model writes them in one file; a model that writes one file per array lists them
            in ``extra`` instead.
        timing: A run-time file, when the model writes one; it may be shared by several samples.
        extra: Any other file of the sample, by a role name the adapter chooses (for
            example ``"pae"``). A role must not be one of ``structure``, ``scores``,
            ``arrays`` and ``timing``.
    """

    directory: Path
    structure: Optional[Path] = None
    scores: Optional[Path] = None
    arrays: Optional[Path] = None
    timing: Optional[Path] = None
    extra: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        clash = sorted(set(self.extra) & set(_FILE_ROLES))
        if clash:
            raise ValueError(f"extra file roles {clash} are reserved names of PredictionFiles")

    def found(self) -> dict[str, Path]:
        """The files that were found, by role (``structure``, ``scores``, ... then ``extra``)."""
        named = {role: getattr(self, role) for role in _FILE_ROLES}
        named.update(self.extra)
        return {role: Path(path) for role, path in named.items() if path is not None}

    def any_found(self) -> bool:
        """True when at least one file exists, the run-time file included."""
        return bool(self.found())

    def has_output(self) -> bool:
        """True when a file that belongs to this sample exists.

        Unlike ``any_found`` it ignores ``timing``, which a model may write once for all
        the samples of a seed (OpenFold3 does), so a sample number that was never written
        does not count as present.
        """
        return any(role != "timing" for role in self.found())


@dataclass(frozen=True)
class SampleRef:
    """One sample of a prediction directory, in the natural order of the model's outputs.

    ``ranking_score`` is the model's own score (NaN when it has none) and is never compared
    across models.
    """

    seed_index: int
    sample: int
    ranking_score: float = _NAN


@dataclass(frozen=True, eq=False)
class TokenLayout:
    """What each token of the PAE and PDE matrices is, in the matrices' order.

    Attributes:
        chain_id: ``(n_tokens,)`` chain ID of each token, as in the model's structure file
            (before ``chain_map``).
        res_id: ``(n_tokens,)`` residue number of each token, as in the structure file.
        atom_index: ``(n_tokens,)`` index, in the atoms of the structure file, of the atom
            that represents the token: the C-alpha of a residue token, the atom itself of a
            per-atom token.
        is_atom_token: ``(n_tokens,)`` bool, True for a token that is one atom of a ligand or
            of a modified residue (AlphaFold3-style tokenisation); None when the model has
            one token per residue.
        extras: Model-specific per-token arrays, each of length ``n_tokens``.

    A record without a layout means one token per residue in chain order, which the
    interface statistics accept only when the matrix size equals the residue count.
    """

    chain_id: np.ndarray
    res_id: np.ndarray
    atom_index: np.ndarray
    is_atom_token: Optional[np.ndarray] = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(np.shape(self.chain_id)[0])

    def token_ranges(self) -> dict[str, tuple[int, int]]:
        """``{chain_id: (start, end)}`` token ranges, chains in order of first appearance.

        Raises:
            ValueError: If the tokens of a chain are not one contiguous run, in which case
                a chain block cannot be cut as a slice of the matrix.
        """
        ranges: dict[str, tuple[int, int]] = {}
        chain_ids = [str(c) for c in self.chain_id]
        start = 0
        while start < len(chain_ids):
            chain = chain_ids[start]
            end = start
            while end < len(chain_ids) and chain_ids[end] == chain:
                end += 1
            if chain in ranges:
                raise ValueError(
                    f"the tokens of chain '{chain}' are not contiguous; "
                    "a chain block cannot be sliced from the PAE matrix"
                )
            ranges[chain] = (start, end)
            start = end
        return ranges

    def problems(self) -> list[str]:
        """What is inconsistent in the layout, one sentence each (empty when it is sound)."""
        problems: list[str] = []
        arrays = {"chain_id": self.chain_id, "res_id": self.res_id, "atom_index": self.atom_index}
        if self.is_atom_token is not None:
            arrays["is_atom_token"] = self.is_atom_token
        arrays.update({f"extras['{key}']": value for key, value in self.extras.items()})
        for label, array in arrays.items():
            if np.ndim(array) != 1:
                problems.append(f"tokens.{label} must be one-dimensional")
        lengths = {
            label: np.shape(array)[0] for label, array in arrays.items() if np.ndim(array) == 1
        }
        if len(set(lengths.values())) > 1:
            problems.append(f"token arrays differ in length: {lengths}")
        elif np.ndim(self.atom_index) == 1 and len(self) and np.min(self.atom_index) < 0:
            problems.append("tokens.atom_index has a negative entry")
        return problems


@dataclass(eq=False)
class PredictionRecord:
    """The parsed confidence outputs and the structure path of one prediction sample.

    ``model`` and ``name`` are the only positional arguments; every other field is
    keyword-only and defaults to "not provided" (NaN, None, or empty). Equality is identity,
    because the array fields make ``==`` ambiguous.

    Attributes:
        model: Adapter name (``"of3"``, ``"af2"``, ...).
        name: Prediction name: the query name (OpenFold3), the job name (ColabFold), the
            input stem (Boltz-2).
        seed_index, sample: 1-based positions in the model's natural order of seeds and
            samples (not seed values, and not a ranking).
        structure_path: The predicted structure, read with the author chain and residue IDs
            and model 1; None when the file is absent.
        chain_map: Model chain ID to user chain ID; empty means no renaming.
        avg_plddt: Mean pLDDT, 0-100.
        ptm, iptm: 0-1.
        gpde: Global predicted distance error, angstrom.
        ranking_score, ranking_score_name: The model's own ranking score and what it is
            called there; never compared across models.
        has_clash, disorder: The model's clash flag (0 or 1) and disorder fraction (0-1).
        chain_ptm, chain_pair_iptm: Per-chain pTM and per-chain-pair ipTM, keys as the model
            writes them.
        plddt_per_atom: ``(n_atoms,)`` pLDDT, 0-100, in the atom order of ``structure_path``.
        pae, pde: ``(n_tokens, n_tokens)`` in angstrom; see the module docstring for the
            orientation of ``pae``.
        tokens: The token layout of ``pae`` and ``pde``; None means one token per residue.
        extras: Model-specific values; see the module docstring.
        timing: Run times the model reports, seconds.
        reasons: One sentence for each value that could not be provided.
        files: The files the record was parsed from; ``PredictionParser.load`` fills it when
            the adapter did not. ``summarize_prediction`` reads it to tell a full-confidence
            file that is absent (already explained in ``reasons``) from one that lacks a value.
    """

    model: str
    name: str
    _: KW_ONLY
    seed_index: int = 1
    sample: int = 1
    structure_path: Optional[Path] = None
    chain_map: dict[str, str] = field(default_factory=dict)
    avg_plddt: float = _NAN
    ptm: float = _NAN
    iptm: float = _NAN
    gpde: float = _NAN
    ranking_score: float = _NAN
    ranking_score_name: str = ""
    has_clash: float = _NAN
    disorder: float = _NAN
    chain_ptm: dict[str, float] = field(default_factory=dict)
    chain_pair_iptm: dict[str, float] = field(default_factory=dict)
    plddt_per_atom: Optional[np.ndarray] = None
    pae: Optional[np.ndarray] = None
    pde: Optional[np.ndarray] = None
    tokens: Optional[TokenLayout] = None
    extras: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    files: Optional[PredictionFiles] = None
    _atoms_cache: Optional[tuple[Any, Any]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.structure_path is not None:
            self.structure_path = Path(self.structure_path)

    @property
    def n_atoms(self) -> int:
        """Number of atoms with a pLDDT value; 0 when there is no per-atom pLDDT."""
        return 0 if self.plddt_per_atom is None else int(len(self.plddt_per_atom))

    def atoms(self):
        """The predicted structure as a biotite ``AtomArray``, chain IDs renamed by ``chain_map``.

        Read once and cached (until ``structure_path`` or ``chain_map`` changes); treat the
        result as read-only and copy it before modifying. Model 1 is read, with the author
        chain and residue IDs. A gzip-compressed file is decompressed first.

        Raises:
            ValueError: If there is no structure file, or ``chain_map`` names a chain the
                structure does not have or would merge two chains into one.
            ImportError: If biotite is not installed.
        """
        if self.structure_path is None:
            raise ValueError(
                f"{self.model} prediction '{self.name}' has no structure file; "
                + ("; ".join(self.reasons) or "the adapter found none")
            )
        key = (str(self.structure_path), tuple(sorted(self.chain_map.items())))
        if self._atoms_cache is None or self._atoms_cache[0] != key:
            atoms = _read_structure(self.structure_path)
            self._atoms_cache = (key, _rename_chains(atoms, self.chain_map))
        return self._atoms_cache[1]

    def validate(self, *, check_structure: bool = False) -> None:
        """Raise ``ValueError`` listing every violation of the record rules.

        Checks the scale of each value (pLDDT 0-100, pTM and ipTM 0-1, PAE and PDE finite
        and not negative), the shape of the arrays (per-atom pLDDT one-dimensional, PAE and
        PDE square and as large as the token layout), the ``chain_map``, and that no scalar
        is None. A pLDDT array whose largest value is at most 1 is reported as a probable
        0-1 scale that the adapter forgot to convert.

        Args:
            check_structure: Also read the structure and check that the pLDDT array has one
                value per atom, that the ``chain_map`` fits the chains of the file and that
                the token layout points at existing atoms. Needs biotite and the file.
        """
        problems = self._problems(check_structure)
        if problems:
            raise ValueError(
                f"invalid {self.model} prediction record '{self.name}': " + "; ".join(problems)
            )

    # ------------------------------------------------------------------ checks

    def _problems(self, check_structure: bool) -> list[str]:
        problems: list[str] = []
        if not self.model or not self.name:
            problems.append("model and name must not be empty")
        if self.seed_index < 1 or self.sample < 1:
            problems.append("seed_index and sample are 1-based positions")

        for label, value, low, high in (
            ("avg_plddt", self.avg_plddt, 0.0, 100.0),
            ("ptm", self.ptm, 0.0, 1.0),
            ("iptm", self.iptm, 0.0, 1.0),
            ("gpde", self.gpde, 0.0, None),
            ("has_clash", self.has_clash, 0.0, 1.0),
            ("disorder", self.disorder, 0.0, 1.0),
        ):
            problems += _scalar_problems(label, value, low, high)
        if _is_number(self.avg_plddt) and 0.0 < self.avg_plddt <= 1.0:
            problems.append(f"avg_plddt {self.avg_plddt:g} reads as a 0-1 value, not 0-100")
        problems += _scalar_problems("ranking_score", self.ranking_score, None, None)
        if _is_number(self.ranking_score) and np.isfinite(self.ranking_score):
            if not self.ranking_score_name:
                problems.append("ranking_score is set but ranking_score_name is empty")

        for label, mapping in (
            ("chain_ptm", self.chain_ptm),
            ("chain_pair_iptm", self.chain_pair_iptm),
        ):
            for key, value in mapping.items():
                problems += _scalar_problems(f"{label}['{key}']", value, 0.0, 1.0)

        problems += self._array_problems()
        problems += _chain_map_problems(self.chain_map)
        if check_structure:
            problems += self._structure_problems()
        return problems

    def _array_problems(self) -> list[str]:
        problems: list[str] = []
        plddt = self.plddt_per_atom
        if plddt is not None:
            if not isinstance(plddt, np.ndarray) or plddt.ndim != 1:
                problems.append("plddt_per_atom must be a one-dimensional numpy array")
            elif plddt.size:
                if not np.all(np.isfinite(plddt)):
                    problems.append("plddt_per_atom has a non-finite value")
                elif plddt.min() < 0.0 or plddt.max() > 100.0:
                    span = f"{plddt.min():g} to {plddt.max():g}"
                    problems.append(f"plddt_per_atom is outside 0-100 (range {span})")
                elif plddt.max() <= 1.0:
                    problems.append(
                        f"plddt_per_atom looks like a 0-1 scale (largest value {plddt.max():g}); "
                        "the adapter must rescale to 0-100"
                    )
        for label, matrix in (("pae", self.pae), ("pde", self.pde)):
            if matrix is None:
                continue
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.ndim != 2
                or matrix.shape[0] != matrix.shape[1]
            ):
                shape = getattr(matrix, "shape", type(matrix).__name__)
                problems.append(
                    f"{label} must be a square two-dimensional numpy array, got {shape}"
                )
            elif not np.all(np.isfinite(matrix)) or matrix.min() < 0.0:
                problems.append(f"{label} must be finite and not negative (angstrom)")
            elif self.tokens is not None and matrix.shape[0] != len(self.tokens):
                problems.append(
                    f"{label} has {matrix.shape[0]} rows but tokens has {len(self.tokens)}"
                )
        if self.tokens is not None:
            problems += self.tokens.problems()
        return problems

    def _structure_problems(self) -> list[str]:
        if self.structure_path is None:
            return ["structure_path is None"]
        try:
            atoms = self.atoms()
        except Exception as exc:  # noqa: BLE001 - reported as a problem of the record
            return [f"structure could not be read: {type(exc).__name__}: {exc}"]
        problems: list[str] = []
        n_atoms = atoms.array_length()
        if self.plddt_per_atom is not None and len(self.plddt_per_atom) != n_atoms:
            problems.append(
                f"plddt_per_atom has {len(self.plddt_per_atom)} values but the structure "
                f"has {n_atoms} atoms"
            )
        if (
            self.tokens is not None
            and len(self.tokens)
            and np.max(self.tokens.atom_index) >= n_atoms
        ):
            problems.append("tokens.atom_index points beyond the last atom of the structure")
        return problems


# ---------------------------------------------------------------------- helpers


def check_chain_map(chain_map: Mapping[str, str]) -> dict[str, str]:
    """Return ``chain_map`` as a plain dict, or raise ``ValueError`` if it cannot be one.

    Every key and value must be a non-empty string, and no two chains may be renamed to the
    same ID (that would merge them).
    """
    problems = _chain_map_problems(chain_map)
    if problems:
        raise ValueError("invalid chain_map: " + "; ".join(problems))
    return dict(chain_map)


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _scalar_problems(
    label: str, value: Any, low: Optional[float], high: Optional[float]
) -> list[str]:
    """Problems of one scalar: None or a non-number, infinity, or outside ``[low, high]``."""
    if value is None:
        return [f"{label} is None; use NaN for a value the model does not provide"]
    if not _is_number(value):
        return [f"{label} must be a number, got {type(value).__name__}"]
    if np.isnan(value):
        return []
    if np.isinf(value):
        return [f"{label} is infinite"]
    if low is not None and value < low or high is not None and value > high:
        bounds = (
            f"{'-inf' if low is None else f'{low:g}'} to {'inf' if high is None else f'{high:g}'}"
        )
        return [f"{label} {value:g} is outside {bounds}"]
    return []


def _read_structure(path: Path):
    """Read model 1 of a structure file; a ``.gz`` file is decompressed to a temporary copy."""
    if path.suffix.lower() != ".gz":
        return load_structure(path, purpose="prediction structure")
    with tempfile.TemporaryDirectory(prefix="bm_prediction_") as scratch:
        plain = Path(scratch) / path.stem  # keeps the inner suffix (.cif or .pdb)
        with gzip.open(path, "rb") as source, open(plain, "wb") as target:
            target.write(source.read())
        return load_structure(plain, purpose="prediction structure")


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
