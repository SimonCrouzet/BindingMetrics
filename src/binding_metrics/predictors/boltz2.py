"""Adapter for the output of Boltz-2 (``boltz predict``).

Layout checked against the source of Boltz v2.2.1 (released 2025-09-08; main b1ebfc4, read on
2026-09-29). No Boltz-2 run and no real output file was available: everything below comes from
reading the writers (``boltz/data/write/writer.py``, ``mmcif.py``, ``pdb.py``), the confidence
heads (``boltz/model/modules/confidencev2.py``, ``confidence_utils.py``) and the tokenizer
(``boltz/data/tokenize/boltz2.py``), and from one Boltz mmCIF excerpt in the ipSAE script
(``ipsae.py:186-207`` of https://github.com/DunbrackLab/IPSAE). Line numbers refer to that source.

    {out}/boltz_results_{stem}/predictions/{stem}/          (main.py:1134, 752)
        {stem}_model_{r}.cif | .pdb                          the structure (--output_format)
        confidence_{stem}_model_{r}.json                     the scalars
        plddt_{stem}_model_{r}.npz    key plddt  (n_tokens,)             0-1
        pae_{stem}_model_{r}.npz      key pae    (n_tokens, n_tokens)    angstrom
        pde_{stem}_model_{r}.npz      key pde    (n_tokens, n_tokens)    angstrom
        affinity_{stem}.json                                 not read

* ``r`` counts from 0 and is the rank by ``confidence_score`` (``writer.py:73-79,159``): sample
  ``k`` of this adapter is file ``r = k - 1``, so ``sample=1`` is the best-ranked model, not just
  the first written. ``{stem}`` is the name of the input file without its suffix. When ``boltz
  predict`` is given a directory of inputs, ``boltz_results_`` carries the name of that
  directory, so the adapter also looks in ``boltz_results_*/predictions/{name}``. It looks in
  ``prediction_dir`` itself, in ``prediction_dir/{name}``, in ``prediction_dir/predictions/{name}``
  and then in those ``boltz_results_*`` folders, and takes the first that holds the sample.
* One ``boltz predict`` run has one seed (``--seed``, ``main.py:919``), so there is no seed
  dimension: ``seed_index`` must be 1, another seed is another output directory.
* PAE and PDE are written whenever the confidence head ran: ``predict_step`` adds ``pde``
  unconditionally and ``pae`` when ``alpha_pae > 0`` (``boltz2.py:1084-1102``), which is 1 in
  the released checkpoint (read from its ``data.pkl`` by the format report). The
  ``--write_full_pae`` and ``--write_full_pde`` flags are read only by Boltz-1
  (``boltz1.py:1192-1194``), although ``docs/prediction.md`` lists them as switches for these
  files (TO VERIFY on a run without the flags; a file that is absent gives a reason). The
  weights are not in the output, so nothing records which checkpoint produced a file.
* The confidence summary has ``confidence_score``, ``ptm``, ``iptm``, ``ligand_iptm``,
  ``protein_iptm``, ``complex_plddt``, ``complex_iplddt``, ``complex_pde``, ``complex_ipde``
  (``writer.py:190-201``) and the dictionaries ``chains_ptm`` and ``pair_chains_iptm``
  (``writer.py:202-214``), which are keyed by the INDEX of the chain (``"0"``, ``"1"``), not its
  name. The index is ``asym_id``, the position of the chain after the input has been grouped by
  entity (``schema.py:1014-1037,1345``), and the structure file lists the chains in that same
  order (``mmcif.py:142``); the adapter names them by the chain IDs of the structure file.
* Scales. ``plddt``, ``complex_plddt`` and ``complex_iplddt`` are 0-1 (``confidence_utils.py:8``),
  scaled to 0-100 here; PAE and PDE are angstrom, 0.25 to 31.75 (64 bins of 0.5,
  ``confidencev2.py:433,478``). ``complex_plddt`` is the mean over tokens
  (``confidencev2.py:325-330``), not over atoms, so with a ligand or a modified residue it differs
  from the mean of ``plddt_per_atom``.
* ``ptm`` and ``iptm`` come from the PAE head as in AlphaFold2 (Jumper et al. 2021,
  doi:10.1038/s41586-021-03819-2). ``iptm`` is exactly 0 when the complex has one chain
  (``confidence_utils.py:91-96,119-123``): the record then gets NaN and a reason.
  ``confidence_score`` is ``(4 complex_plddt + iptm) / 5`` (``ptm`` in place of an ``iptm`` of 0;
  ``boltz2.py:1086-1095``), so it is dominated by pLDDT and is the ranking score of the record.
  ``complex_pde`` is the PDE averaged over token pairs, weighted by the probability of contact
  (the first 20 distogram bins, about 8 angstrom; ``confidencev2.py:433-458``); it fills the
  record's ``gpde``.
* PAE orientation: ``pae[i, j]`` is the error of token j when the structures are aligned on token
  i. Boltz sums the TM term over the column index j for each row i and takes the maximum over
  the rows i (``confidence_utils.py:108-123``), so the row is the alignment frame, and the
  chain-pair value ``pair_chains_iptm[a][b]`` scores the tokens of chain ``a`` with the alignment
  on chain ``b`` (``confidence_utils.py:160-179``); the record's key ``"A-B"`` is the value of
  ``pair_chains_iptm[A][B]``. PDE is symmetric (``z + z.T``, ``confidencev2.py:313``).
* Tokens (``tokenize/boltz2.py:181-340``): a residue of a polymer chain is one token, standard
  or modified (all the atoms of a modified residue are in that token; OpenFold3, Protenix and
  Chai-1 make one token per atom of it), a ligand is one token per atom. The token order is the
  chain order of the structure file, then the atom order. Nothing in the output lists the tokens,
  so the adapter reads the atom records of the structure file without biotite
  (:func:`_read_atom_sites`) and applies that rule (:func:`_boltz_tokens`) to build
  ``record.tokens``; a token count that differs from the array size raises. The pLDDT of a
  token is written to every atom of it (``mmcif.py:192-214``); the adapter expands the pLDDT
  array through the token map and compares the result with the B-factor column of the file.
* Structure files (see :func:`_read_atom_sites`): atoms chain by chain, residues numbered from 1
  in each chain (``mmcif.py:187``), chain names as given in the input, a polymer chain as ``ATOM``
  records and a ligand chain as ``HETATM`` records with ``label_seq_id`` ``.``.

Fields this model does not provide, left NaN or empty: ``has_clash`` and ``disorder`` (Boltz-2
computes neither) and ``timing`` (it writes no timing file). ``chain_pair_iptm`` holds the
off-diagonal ``pair_chains_iptm`` entries and ``chain_ptm`` the diagonal, keyed by chain ID and
by ``"A-B"``; the values as Boltz writes them, keyed by index, are in
``record.extras["chains_ptm_by_index"]`` and ``["pair_chains_iptm_by_index"]``. Also in extras:
``ligand_iptm``, ``protein_iptm`` (0 when the complex has no ligand or a single protein chain),
``complex_iplddt`` (0-1), ``complex_ipde``, ``chain_ids_by_index``, ``model_rank``, ``n_tokens``
and ``bfactor_matches_plddt``.

TO VERIFY against a real Boltz-2 run (each is marked where it is used):

1. The whole layout: no real output file was read. Versions after 2.2.1 are not checked. Boltz-1
   writes the same file names with other semantics (it tokenises modified residues per atom), which
   this adapter does not implement: a token count that differs raises.
2. That the mmCIF written through the ihm library has ``auth_asym_id`` and ``auth_seq_id`` (the
   Boltz-1 excerpt in ``ipsae.py`` does) and that they equal the chain name and the residue number
   ``res_idx + 1``. Without them the label columns are read.
3. That the chain index of the confidence file is the position of the chain in the structure
   file for complexes of three or more chains and for identical chains listed apart (the input
   is grouped by entity, so ``A: x, B: y, C: x`` gives the chains A, C, B; ``schema.py:1014-1037``).
   The adapter uses the order of the file, which the writer takes from the same chain table.
4. A checkpoint whose confidence head is per atom (``token_level_confidence=False``,
   ``confidencev2.py:398``) would write a longer ``plddt``; the adapter refuses it (the array
   length differs from the token count) instead of guessing.
5. That a standard residue never appears in a ligand chain and a modified residue never in a
   ``HETATM`` record of a polymer chain, which the token rule assumes (``mmcif.py:149-153``).
6. That ``complex_pde`` (contact-weighted mean PDE) is the quantity that OpenFold3 and Protenix
   write as ``gpde``; the record's ``gpde`` is compared only within one model either way.

The module imports numpy only. The structure file is read as text when the confidence files are
parsed, never with biotite, and biotite is imported only when ``record.atoms()`` is called.
"""

from __future__ import annotations

import glob
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

_MMCIF_SUFFIXES = (".cif", ".mmcif")

#: The per-token arrays of a sample, each in its own ``.npz`` under the key of the same name
#: (``writer.py:229,238,247``).
_ARRAY_ROLES = ("plddt", "pae", "pde")

#: Scalars of ``confidence_{stem}_model_{r}.json`` (``writer.py:190-201``).
_SUMMARY_KEYS = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)

#: How far a 0-1 value may exceed 1 before it is taken for another scale.
_SCALE_TOLERANCE = 1e-3

#: The B-factor column has the pLDDT rounded to 3 decimals in mmCIF and 2 in PDB.
_BFACTOR_TOLERANCE = 0.01

#: Upper bound of the samples ``list_samples`` returns.
_MAX_SAMPLES = 1000

#: One value of a CIF row: a quoted string (a quote closes only before white space, so
#: ``"O5'"`` is one value) or a run of non-blank characters.
_CIF_VALUE = re.compile(r"""'[^']*'(?=\s|$)|"[^"]*"(?=\s|$)|\S+""")
_CIF_UNKNOWN = (".", "?")

#: The atom that stands for a residue token, by preference: the C-alpha of an amino acid, the
#: C1' of a nucleotide. Boltz-2 itself centres a modified residue on its first atom
#: (``schema.py:784``); the C-alpha is the atom a peptide toolkit expects to find.
_TOKEN_ATOMS = ("CA", "C1'")


@dataclass(frozen=True, eq=False)
class _AtomSites:
    """The atom records of a structure file in file order (model 1).

    Attributes:
        chain: Chain ID of each atom (author ID in mmCIF).
        res_id: Residue number of each atom (author number in mmCIF); 0 when the file gives none.
        ins_code: Insertion code of each atom, ``""`` when there is none.
        res_name: Residue name of each atom.
        atom_name: Atom name.
        hetero: True for a ``HETATM`` record (a ligand in a Boltz-2 file).
        b_factor: ``(n_atoms,)`` B-factor column; NaN where the file has none.
    """

    chain: list[str]
    res_id: list[int]
    ins_code: list[str]
    res_name: list[str]
    atom_name: list[str]
    hetero: list[bool]
    b_factor: np.ndarray

    def __len__(self) -> int:
        return len(self.chain)


def _cif_values(line: str) -> list[str]:
    """The values of one CIF line, unquoted."""
    values = []
    for match in _CIF_VALUE.finditer(line):
        token = match.group()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
            token = token[1:-1]
        values.append(token)
    return values


def _atom_site_table(lines: list[str]) -> Optional[tuple[list[str], list[str]]]:
    """The tags and the flat list of values of the first ``_atom_site`` loop, or None.

    Values are collected across lines, so a row that is wrapped over several lines is read
    like one that is not.
    """
    tags: list[str] = []
    values: list[str] = []
    in_header = False
    in_atom_loop = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "loop_" or line.startswith("data_"):
            if in_atom_loop:
                break
            in_header = line == "loop_"
            tags = []
            continue
        if in_header and line.startswith("_"):
            tags.append(line.split()[0])
            continue
        if in_header:  # the first data line of a loop
            in_header = False
            in_atom_loop = bool(tags) and tags[0].startswith("_atom_site.")
        if in_atom_loop:
            if line.startswith("_"):  # a tag after the loop: the loop is over
                break
            values.extend(_cif_values(line))
    if not in_atom_loop:
        return None
    return tags, values


def _cif_int(value: str) -> Optional[int]:
    if value in _CIF_UNKNOWN:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cif_float(value: str) -> float:
    if value in _CIF_UNKNOWN:
        return _NAN
    return float(value)


def _read_mmcif_sites(lines: list[str], path: Path) -> Optional[_AtomSites]:
    table = _atom_site_table(lines)
    if table is None:
        return None
    tags, values = table
    columns = {tag.split(".", 1)[1]: index for index, tag in enumerate(tags)}
    if len(values) % len(tags):
        raise ValueError(
            f"{path}: the _atom_site loop has {len(values)} values for {len(tags)} columns; "
            "the file is truncated or not an mmCIF written by Boltz"
        )

    def _first(*names: str) -> Optional[int]:
        return next((columns[name] for name in names if name in columns), None)

    group = _first("group_PDB")
    atom_name = _first("label_atom_id", "auth_atom_id")
    res_name = _first("label_comp_id", "auth_comp_id")
    chain = _first("auth_asym_id", "label_asym_id")
    b_column = _first("B_iso_or_equiv")
    auth_seq = _first("auth_seq_id")
    label_seq = _first("label_seq_id")
    for label, column in (
        ("group_PDB", group),
        ("label_atom_id", atom_name),
        ("label_comp_id", res_name),
        ("auth_asym_id or label_asym_id", chain),
        ("B_iso_or_equiv", b_column),
    ):
        if column is None:
            raise ValueError(f"{path}: the _atom_site loop has no {label} column")
    if auth_seq is None and label_seq is None:
        raise ValueError(f"{path}: the _atom_site loop has no auth_seq_id or label_seq_id column")
    ins = _first("pdbx_PDB_ins_code")
    model = _first("pdbx_PDB_model_num")

    n_columns = len(tags)
    rows = [values[start : start + n_columns] for start in range(0, len(values), n_columns)]
    if model is not None and rows:
        rows = [row for row in rows if row[model] == rows[0][model]]  # model 1 only

    def _res_id(row: list[str]) -> int:
        for column in (auth_seq, label_seq):
            if column is not None:
                number = _cif_int(row[column])
                if number is not None:
                    return number
        return 0

    if not rows:
        return None
    return _AtomSites(
        chain=[row[chain] for row in rows],
        res_id=[_res_id(row) for row in rows],
        ins_code=["" if ins is None or row[ins] in _CIF_UNKNOWN else row[ins] for row in rows],
        res_name=[row[res_name] for row in rows],
        atom_name=[row[atom_name] for row in rows],
        hetero=[row[group] == "HETATM" for row in rows],
        b_factor=np.array([_cif_float(row[b_column]) for row in rows], dtype=float),
    )


def _read_pdb_sites(lines: list[str], path: Path) -> Optional[_AtomSites]:
    chain, res_id, ins_code, res_name, atom_name, hetero, b_factor = ([] for _ in range(7))
    for number, line in enumerate(lines, start=1):
        record = line[:6]
        if record == "ENDMDL":
            break  # model 1 only
        if record not in ("ATOM  ", "HETATM"):
            continue
        try:
            residue_number = int(line[22:26])
            b_value = float(line[60:66])
        except ValueError as exc:
            raise ValueError(
                f"{path}, line {number}: not a PDB atom record (residue number or B-factor "
                f"column unreadable): {exc}"
            ) from exc
        chain.append(line[21].strip())
        res_id.append(residue_number)
        ins_code.append(line[26].strip())
        res_name.append(line[17:20].strip())
        atom_name.append(line[12:16].strip())
        hetero.append(record == "HETATM")
        b_factor.append(b_value)
    if not chain:
        return None
    return _AtomSites(
        chain, res_id, ins_code, res_name, atom_name, hetero, np.array(b_factor, dtype=float)
    )


def _read_atom_sites(path: Path) -> Optional[_AtomSites]:
    """Read the atom records of a Boltz-2 structure file (mmCIF or PDB), or None if it has none.

    A file with the suffix ``.cif`` or ``.mmcif`` is read as mmCIF, any other as PDB, as
    ``binding_metrics.metrics._common.load_structure`` does. Only the first model is read.

    mmCIF: the first ``_atom_site`` loop. Columns are picked by name, so their order does not
    matter (the order of a Boltz mmCIF is ``group_PDB, id, type_symbol, label_atom_id,
    label_alt_id, label_comp_id, label_seq_id, auth_seq_id, pdbx_PDB_ins_code, label_asym_id,
    Cartn_x/y/z, occupancy, label_entity_id, auth_asym_id, auth_comp_id, B_iso_or_equiv,
    pdbx_PDB_model_num`` in the Boltz example of ``ipsae.py:186-207``). The chain is
    ``auth_asym_id`` and the residue number ``auth_seq_id``, each falling back to its ``label_``
    column, the fields that biotite reads. A ligand row has ``.`` as ``label_seq_id`` and its
    residue number in ``auth_seq_id``. TO VERIFY: that Boltz-2 files have the author columns.

    PDB: the fixed columns of ``pdb.py:127-134`` (atom name 13-16, residue name 18-20, chain
    22, residue number 23-26, insertion code 27, B-factor 61-66); ligand residues are named LIG.

    Raises:
        ValueError: The file is not text, or an atom record of it is malformed.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not a text structure file ({exc})") from exc
    if path.suffix.lower() in _MMCIF_SUFFIXES:
        return _read_mmcif_sites(lines, path)
    return _read_pdb_sites(lines, path)


@dataclass(frozen=True, eq=False)
class _BoltzTokens:
    """The tokens of a Boltz-2 structure.

    Attributes:
        token_of_atom: ``(n_atoms,)`` index of the token of each atom.
        layout: The tokens as a ``TokenLayout``, in the order of the PAE and PDE matrices.
        chain_order: The chain IDs in the order of their first appearance, which is the order
            of the chain indices in ``confidence_*.json``.
        b_factor: ``(n_atoms,)`` the B-factor column of the file, for a check of the token map.
    """

    token_of_atom: np.ndarray
    layout: TokenLayout
    chain_order: list[str]
    b_factor: np.ndarray

    @property
    def n_tokens(self) -> int:
        return len(self.layout)


def _boltz_tokens(sites: _AtomSites) -> _BoltzTokens:
    """Tokenise the atoms the way Boltz-2 does (``tokenize/boltz2.py``).

    Every ``HETATM`` atom (a ligand) is a token. The ``ATOM`` atoms of one residue, told
    apart by chain, residue number and insertion code, are one token, whether the residue is
    standard or modified. The token of a residue is represented by its C-alpha (its C1' for a
    nucleotide, its first atom otherwise), the token of a ligand atom by that atom.

    TO VERIFY: the rule assumes that a standard residue never appears in a ligand chain and a
    modified residue never as ``HETATM`` in a polymer chain (module docstring, item 5). Where it
    fails the token count differs from the arrays and ``Boltz2Parser.parse`` refuses the file.

    Raises:
        ValueError: A chain is written in two separate blocks; Boltz-2 writes each chain in
            one block, and the confidence files index chains by their order.
    """
    n_atoms = len(sites)
    token_of_atom = np.empty(n_atoms, dtype=int)
    first_atoms: list[int] = []
    for i in range(n_atoms):
        if sites.hetero[i]:
            starts_token = True
        elif i == 0 or sites.hetero[i - 1]:
            starts_token = True
        else:
            starts_token = (sites.chain[i], sites.res_id[i], sites.ins_code[i]) != (
                sites.chain[i - 1],
                sites.res_id[i - 1],
                sites.ins_code[i - 1],
            )
        if starts_token:
            first_atoms.append(i)
        token_of_atom[i] = len(first_atoms) - 1

    ends = first_atoms[1:] + [n_atoms]
    representative = []
    for start, end in zip(first_atoms, ends):
        chosen = start
        if not sites.hetero[start]:
            names = sites.atom_name[start:end]
            for preferred in _TOKEN_ATOMS:
                if preferred in names:
                    chosen = start + names.index(preferred)
                    break
        representative.append(chosen)

    token_chains = [sites.chain[i] for i in first_atoms]
    chain_order: list[str] = []
    for position, chain in enumerate(token_chains):
        if position and chain == token_chains[position - 1]:
            continue
        if chain in chain_order:
            raise ValueError(
                f"chain '{chain}' is written in two separate blocks of atoms; Boltz-2 writes "
                "each chain in one block, so this file is not a Boltz-2 prediction as read here"
            )
        chain_order.append(chain)

    layout = TokenLayout(
        chain_id=np.array(token_chains, dtype=str),
        res_id=np.array([sites.res_id[i] for i in first_atoms], dtype=int),
        atom_index=np.array(representative, dtype=int),
        is_atom_token=np.array([sites.hetero[i] for i in first_atoms], dtype=bool),
        extras={"res_name": np.array([sites.res_name[i] for i in first_atoms], dtype=str)},
    )
    return _BoltzTokens(token_of_atom, layout, chain_order, sites.b_factor)


# ---------------------------------------------------------------------- the confidence files


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _summary_number(raw: dict, key: str, path: Path) -> float:
    """``raw[key]`` as a float; NaN when the key is absent or null."""
    value = raw.get(key)
    if value is None:
        return _NAN
    if not _is_number(value):
        raise ValueError(f"{path}: '{key}' is {value!r}, not a number")
    return float(value)


def _number_mapping(value, label: str, path: Path) -> dict[str, float]:
    """A JSON object of numbers as ``{str: float}``; None gives ``{}``."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: '{label}' must be a JSON object, got {type(value).__name__}")
    if not all(_is_number(v) for v in value.values()):
        raise ValueError(f"{path}: '{label}' holds a value that is not a number")
    return {str(k): float(v) for k, v in value.items()}


def parse_confidence_summary(path: Path) -> dict:
    """Parse ``confidence_{stem}_model_{r}.json``.

    Returns:
        The nine scalars of ``_SUMMARY_KEYS`` as floats (NaN when a key is absent) and
        ``chains_ptm`` (``{"0": x}``) and ``pair_chains_iptm`` (``{"0": {"1": x}}``) as written,
        keyed by chain index (``{}`` when absent).

    Raises:
        ValueError: The file is not a JSON object, or a value has the wrong type.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    summary = {key: _summary_number(raw, key, path) for key in _SUMMARY_KEYS}
    summary["chains_ptm"] = _number_mapping(raw.get("chains_ptm"), "chains_ptm", path)
    pairs = raw.get("pair_chains_iptm")
    if pairs is not None and not isinstance(pairs, dict):
        raise ValueError(
            f"{path}: 'pair_chains_iptm' must be a JSON object, got {type(pairs).__name__}"
        )
    summary["pair_chains_iptm"] = {
        str(a): _number_mapping(row, f"pair_chains_iptm['{a}']", path)
        for a, row in (pairs or {}).items()
    }
    return summary


def parse_token_array(path: Path, key: str) -> np.ndarray:
    """Read one of ``plddt``, ``pae`` or ``pde`` from its ``.npz`` file, as floats.

    ``plddt`` must be one-dimensional on 0-1, ``pae`` and ``pde`` square, finite and not
    negative. The file is opened without pickle.

    Raises:
        ValueError: The file is not a readable ``.npz`` (a message that names the file), the key
            is absent, or the array has the wrong shape, is not finite, or is not on the scale
            Boltz-2 writes.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            available = list(data.files)
            array = np.asarray(data[key], dtype=float) if key in available else None
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path}: not a readable .npz file ({type(exc).__name__}: {exc})") from exc
    if array is None:
        raise ValueError(f"{path}: no array named '{key}' (the file has {available})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: '{key}' has a value that is not finite")
    if key == "plddt":
        if array.ndim != 1:
            raise ValueError(f"{path}: 'plddt' must have one value per token, got {array.shape}")
        if array.size and (array.min() < 0.0 or array.max() > 1.0 + _SCALE_TOLERANCE):
            raise ValueError(
                f"{path}: 'plddt' runs from {array.min():g} to {array.max():g}; Boltz-2 writes "
                "it on 0-1, so this file was not written as this adapter reads it"
            )
    else:
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"{path}: '{key}' must be square, got {array.shape}")
        if array.size and array.min() < 0.0:
            raise ValueError(f"{path}: '{key}' has a negative value (angstrom expected)")
    return array


def _percent(value: float, key: str, path: Path) -> float:
    """A 0-1 value of the summary on 0-100; NaN stays NaN and another scale raises."""
    if np.isnan(value):
        return value
    if value < 0.0 or value > 1.0 + _SCALE_TOLERANCE:
        raise ValueError(f"{path}: '{key}' is {value:g}, not on the 0-1 scale of Boltz-2")
    return 100.0 * value


def _sample_files(directory: Path, name: str, rank: int) -> dict:
    """The files of ``{name}_model_{rank}`` in ``directory``, by role (absent ones left out)."""
    stem = f"{name}_model_{rank}"
    candidates = {
        "structure": (f"{stem}.cif", f"{stem}.pdb"),
        "scores": (f"confidence_{stem}.json",),
        **{role: (f"{role}_{stem}.npz",) for role in _ARRAY_ROLES},
    }
    found = {}
    for role, filenames in candidates.items():
        for filename in filenames:
            if (directory / filename).is_file():
                found[role] = directory / filename
                break
    return found


def _candidate_directories(root: Path, name: str) -> list[Path]:
    """Where the sample files of ``name`` may be, given the directory the user pointed at."""
    candidates = [
        root,
        root / name,
        root / "predictions" / name,
        root / f"boltz_results_{name}" / "predictions" / name,
    ]
    if root.is_dir():
        candidates += sorted(root.glob(f"boltz_results_*/predictions/{glob.escape(name)}"))
    return candidates


class Boltz2Parser(PredictionParser):
    """Reads the output of one Boltz-2 input (see the module docstring)."""

    name = "boltz2"
    display_name = "Boltz-2"
    family = "af3"

    # The inputs Boltz-2 2.2.1 can be given, from its source and documentation (checked against
    # the clone on 2026-09-30; tests/test_pre_boltz2_limits.py re-reads it when the clone is at
    # hand). No closure is refused: `cyclic: true` wraps a chain head-to-tail
    # (boltz/data/parse/schema.py) and the `bond` constraint takes any two atoms of the input, which
    # the featuriser reads as a cyclic period when it joins the ends of a chain
    # (boltz/data/feature/featurizerv2.py), so a disulfide or a lactam between canonical residues
    # can be given. Not declared because nothing shows it: a limit on the residue classes (a
    # modified residue is a CCD code in `modifications`; D-amino acids and N-methyl are not
    # mentioned), and on the binder size (no maximum is stated).
    # Modes (v2.2.1 source and documentation, checked against the clone on 2026-10-01): all four
    # are declared. predict is the plain input; refold and score need a template per chain, which
    # `templates` takes with `chain_id` (docs/prediction.md, Templates); an unforced template
    # carries no pose, because the template module lets features attend within the same chain only
    # (src/boltz/model/modules/trunkv2.py, "Compute asym mask"). lock is the template with
    # `force: true` and a `threshold`: process_template_features
    # (src/boltz/data/feature/featurizerv2.py) puts all the chains a template file maps into one
    # row, and TemplateReferencePotential (src/boltz/model/potentials/potentials.py) aligns that
    # row rigidly over its templated tokens (weighted_rigid_align) and penalises a deviation
    # larger than the threshold. It is a guidance term, so lock is a caveat and not a promise.
    capabilities = Capabilities(
        modes={"predict", "refold", "score", "lock"},
        caveats={
            "modes:lock": (
                "Boltz-2 pins the pose through a template with `force: true` and a `threshold` "
                "(docs/prediction.md, Templates): TemplateReferencePotential "
                "(src/boltz/model/potentials/potentials.py) aligns the template rigidly over its "
                "templated tokens and pulls the prediction back when it deviates by more than "
                "the threshold. That is a guidance term with weight 0.1, not a hard constraint; "
                "templates are for protein chains only; and this was read from the source and "
                "never run."
            ),
            "closures:staple": (
                "Boltz-2 takes a covalent link between residues only as a `bond` constraint, and "
                "its documentation lists that constraint as supported for CCD ligands and "
                "canonical residues only (docs/prediction.md, section Constraints); the residues "
                "of a hydrocarbon staple are not canonical, so the staple is outside what the "
                "documentation says is supported."
            ),
        },
        version="2.2.1",
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

        ``sample`` is the 1-based position in the rank order of Boltz-2 (file ``model_{sample-1}``).
        ``seed_index`` other than 1 finds nothing: a run has one seed. The directory is searched
        as the module docstring says.
        """
        root = Path(prediction_dir)
        found: dict = {}
        if seed_index == 1 and sample >= 1:
            for directory in _candidate_directories(root, name):
                found = _sample_files(directory, name, sample - 1) if directory.is_dir() else {}
                if found:
                    break
        return PredictionFiles(
            directory=root,
            structure=found.get("structure"),
            scores=found.get("scores"),
            extra={role: found[role] for role in _ARRAY_ROLES if role in found},
        )

    def list_samples(self, prediction_dir: str | Path, name: str) -> list[SampleRef]:
        """The samples present, ranked as Boltz-2 wrote them, with ``confidence_score``.

        Only the confidence summaries are read, not the arrays or the structures.
        """
        refs: list[SampleRef] = []
        while len(refs) < _MAX_SAMPLES:
            sample = len(refs) + 1
            files = self.find_files(Path(prediction_dir), name, sample=sample)
            if not files.has_output():
                break
            score = _NAN
            if files.scores is not None:
                score = parse_confidence_summary(files.scores)["confidence_score"]
            refs.append(SampleRef(1, sample, score))
        return refs

    def parse(
        self,
        files: PredictionFiles,
        *,
        name: str,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionRecord:
        """Read the confidence files and the structure of one sample into a record.

        A missing file leaves its values at NaN (None for an array) and adds a reason; a
        corrupt or inconsistent file raises ``ValueError``. The structure file is read as text
        to tell the atoms of each token; when it is absent or holds no atom records the
        atom-bound values (``plddt_per_atom``, the chain-keyed dictionaries, the token layout)
        are left empty and a reason says so, while ``avg_plddt`` still comes from the tokens.
        """
        record = PredictionRecord(
            self.name,
            name,
            seed_index=seed_index,
            sample=sample,
            structure_path=files.structure,
            ranking_score_name="confidence_score",
            files=files,
        )
        if not files.has_output():
            record.reasons.append(_no_output_reason(files, name, seed_index, sample))
            return record

        stem = f"{name}_model_{sample - 1}"
        record.extras["model_rank"] = sample - 1
        if files.scores is None:
            record.reasons.append(f"confidence summary confidence_{stem}.json not found")
        for role in _ARRAY_ROLES:
            if role not in files.extra:
                record.reasons.append(f"{role}_{stem}.npz not found")

        summary = None if files.scores is None else parse_confidence_summary(files.scores)
        arrays = {
            role: parse_token_array(files.extra[role], role)
            for role in _ARRAY_ROLES
            if role in files.extra
        }
        sizes = {role: array.shape[0] for role, array in arrays.items()}
        # TO VERIFY: a per-atom pLDDT (a head without token_level_confidence, module docstring
        # item 4) has more values than PAE has rows and is refused here.
        if len(set(sizes.values())) > 1:
            raise ValueError(
                f"the arrays of {stem} disagree on the number of tokens: {sizes}; "
                "they were not written by one run"
            )
        n_tokens = next(iter(sizes.values()), None)
        if n_tokens is not None:
            record.extras["n_tokens"] = n_tokens

        # What can only be filled through the atoms of the structure file
        needs = []
        if "plddt" in arrays:
            needs.append("plddt_per_atom")
        if summary is not None and (summary["chains_ptm"] or summary["pair_chains_iptm"]):
            needs.append("chain_ptm and chain_pair_iptm")

        tokens = None
        if files.structure is None:
            reason = f"structure file {stem}.cif (or .pdb) not found"
            if needs:
                reason += (
                    f"; Boltz-2 values are per token, so {' and '.join(needs)} need it and are "
                    "left empty"
                )
            record.reasons.append(reason)
        elif arrays or needs:
            tokens = self._tokens_of(record, files, stem, n_tokens, needs)
        if tokens is not None and "plddt" in arrays:
            self._expand_plddt(record, files, arrays["plddt"], tokens)

        if summary is not None:
            self._set_scalars(record, files, summary, tokens)
        elif "plddt" in arrays:
            record.avg_plddt = 100.0 * float(arrays["plddt"].mean())
        record.pae = arrays.get("pae")
        record.pde = arrays.get("pde")
        return record

    # ------------------------------------------------------------------ steps of parse

    @staticmethod
    def _tokens_of(record, files, stem, n_tokens, needs) -> Optional[_BoltzTokens]:
        """The tokens of the structure file, set on the record; None if the file has no atoms.

        Raises:
            ValueError: The token count of the structure differs from the array size.
        """
        sites = _read_atom_sites(files.structure)
        if sites is None:
            record.reasons.append(
                f"{files.structure.name} holds no atom records that could be read, so the tokens "
                "are unknown and PAE and PDE have no layout"
                + (f"; {' and '.join(needs)} need them and are left empty" if needs else "")
            )
            return None
        tokens = _boltz_tokens(sites)
        if n_tokens is not None and tokens.n_tokens != n_tokens:
            n_ligand = int(tokens.layout.is_atom_token.sum())
            raise ValueError(
                f"{files.structure.name} has {tokens.n_tokens} tokens by the Boltz-2 rule "
                f"({tokens.n_tokens - n_ligand} polymer residues and {n_ligand} ligand atoms) "
                f"but the arrays of {stem} have {n_tokens}. The files are not from one Boltz-2 "
                "run, or the run used another tokenisation (this adapter was checked against "
                "Boltz-2 v2.2.1; Boltz-1 tokenises modified residues per atom)"
            )
        record.tokens = tokens.layout
        record.extras["n_tokens"] = tokens.n_tokens
        return tokens

    @staticmethod
    def _expand_plddt(record, files, plddt_token, tokens) -> None:
        """Expand the per-token pLDDT to atoms and compare it with the B-factor column."""
        record.plddt_per_atom = 100.0 * plddt_token[tokens.token_of_atom]
        gap = np.abs(tokens.b_factor - record.plddt_per_atom)
        agrees = bool(np.all(gap <= _BFACTOR_TOLERANCE))
        record.extras["bfactor_matches_plddt"] = agrees
        if not agrees:
            logger.warning(
                "%s: the B-factor column differs from the expanded pLDDT (largest gap %s); the "
                "B-factors were rewritten, or the token-to-atom map does not fit this file",
                files.structure,
                "unknown" if np.isnan(gap).any() else f"{gap.max():.3f}",
            )

    @staticmethod
    def _set_scalars(record, files, summary, tokens) -> None:
        """Scalars, the extras and the chain-keyed dictionaries of the confidence summary."""
        path = files.scores
        record.avg_plddt = _percent(summary["complex_plddt"], "complex_plddt", path)
        record.ptm = summary["ptm"]
        record.iptm = summary["iptm"]
        record.gpde = summary["complex_pde"]
        record.ranking_score = summary["confidence_score"]
        for key in ("ligand_iptm", "protein_iptm", "complex_iplddt", "complex_ipde"):
            if not np.isnan(summary[key]):
                record.extras[key] = summary[key]
        record.extras["chains_ptm_by_index"] = summary["chains_ptm"]
        record.extras["pair_chains_iptm_by_index"] = summary["pair_chains_iptm"]

        if len(summary["chains_ptm"]) == 1:
            # no pair of different chains exists, and Boltz-2 writes 0 for that case
            record.iptm = _NAN
            record.reasons.append(
                "ipTM is not defined for a single chain (Boltz-2 writes 0); left NaN"
            )
        if tokens is None:
            return
        # TO VERIFY (module docstring, item 3): chain index k is the k-th chain of the file.
        order = tokens.chain_order
        by_index = {str(k): chain for k, chain in enumerate(order)}
        record.extras["chain_ids_by_index"] = by_index
        unknown = sorted(
            (set(summary["chains_ptm"]) | set(summary["pair_chains_iptm"])) - set(by_index)
        )
        if unknown:
            raise ValueError(
                f"{path.name} has chain indices {unknown} but the structure has "
                f"{len(order)} chains {order}; the files are not from one Boltz-2 run"
            )
        record.chain_ptm = {by_index[k]: v for k, v in summary["chains_ptm"].items()}
        record.chain_pair_iptm = {
            f"{by_index[a]}-{by_index[b]}": value
            for a, row in summary["pair_chains_iptm"].items()
            for b, value in row.items()
            if a != b
        }


def _no_output_reason(files: PredictionFiles, name: str, seed_index: int, sample: int) -> str:
    if seed_index != 1:
        return (
            f"Boltz-2 writes one seed per run, so seed_index must be 1 (got {seed_index}); "
            "another seed is another output directory"
        )
    return (
        f"no Boltz-2 output found for '{name}' (sample {sample}: files "
        f"{name}_model_{sample - 1}.*) in {files.directory}; looked in that directory, in "
        f"./{name}, in ./predictions/{name} and in ./boltz_results_*/predictions/{name}"
    )
