"""Adapter for the output of Boltz-2 (``boltz predict``): reading the structure and its tokens.

The confidence files of Boltz-2 are per token, and the atoms of a token are known only from the
structure file, so this module first reads the atom records of that file (without biotite, so
that parsing the scalars needs no structure library) and applies the tokenisation rule of
Boltz-2 to them.

Layout of the structure files, checked against the source of Boltz v2.2.1 (2025-09-08; main
b1ebfc4, read on 2026-09-29): no Boltz-2 run and no real output file was available, so what
follows comes from reading the writers (``boltz/data/write/mmcif.py``, ``pdb.py``) and the
tokenizer (``boltz/data/tokenize/boltz2.py``), not from running the model.

* ``{stem}_model_{r}.cif`` (default) or ``.pdb`` (``--output_format pdb``). Atoms are written
  chain by chain in the order of the input, residue by residue; a chain of a protein, DNA or
  RNA is written as ``ATOM`` records, a ligand chain as ``HETATM`` records (``mmcif.py:149-153``,
  ``pdb.py:66-70``).
* mmCIF: one ``_atom_site`` loop (the column order of a Boltz mmCIF is shown in
  ``ipsae.py:186-207`` of https://github.com/DunbrackLab/IPSAE, which reads Boltz files). The
  reader picks columns by name: chain from ``auth_asym_id`` (else ``label_asym_id``), residue
  number from ``auth_seq_id`` (else ``label_seq_id``), the same fields that biotite reads.
  Ligand rows have ``label_seq_id`` ``.``. TO VERIFY: that ``auth_asym_id`` and ``auth_seq_id``
  are present in Boltz-2 files (they are in the Boltz-1 example of ``ipsae.py``, which the same
  writer produces) and that the ihm library writes them as the chain name and ``res_idx + 1``.
* PDB: fixed columns (``pdb.py:127-134``); ligand residues are all named ``LIG``.
* Residue numbers run 1 to N in each chain whatever the numbers of the input
  (``mmcif.py:187``, ``pdb.py:97``); chain names are the ``id`` values of the input.

Tokens (``tokenize/boltz2.py:181-340``): a residue of a polymer chain is one token, standard or
modified (a modified residue keeps all its atoms in that one token, unlike the AlphaFold3-style
tokenisation of OpenFold3, Protenix and Chai-1); a ligand is one token per atom. The token order
is the order of the atoms in the file. The B-factor of every atom is the pLDDT (0-100) of its
token, rounded to 3 decimals in mmCIF and 2 in PDB (``mmcif.py:192-214``, ``pdb.py:105-124``),
which is an independent check of the token map (:func:`_boltz_tokens`).

The reader does not open the file with biotite. It returns None for a file that holds no atom
records, and raises ``ValueError`` for a file it cannot read as text or whose atom records are
malformed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics.predictors.record import TokenLayout

logger = logging.getLogger(__name__)

_NAN = float("nan")

_MMCIF_SUFFIXES = (".cif", ".mmcif")

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
    """

    token_of_atom: np.ndarray
    layout: TokenLayout
    chain_order: list[str]

    @property
    def n_tokens(self) -> int:
        return len(self.layout)


def _boltz_tokens(sites: _AtomSites) -> _BoltzTokens:
    """Tokenise the atoms the way Boltz-2 does (``tokenize/boltz2.py``).

    Every ``HETATM`` atom (a ligand) is a token. The ``ATOM`` atoms of one residue, told
    apart by chain, residue number and insertion code, are one token, whether the residue is
    standard or modified. The token of a residue is represented by its C-alpha (its C1' for a
    nucleotide, its first atom otherwise), the token of a ligand atom by that atom.

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
    return _BoltzTokens(token_of_atom, layout, chain_order)
