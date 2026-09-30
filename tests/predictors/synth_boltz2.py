"""Writer of a synthetic Boltz-2 output directory (layout in ``predictors/boltz2.py``).

    {directory}/boltz_results_{name}/predictions/{name}/
        {name}_model_{r}.cif            the structure
        confidence_{name}_model_{r}.json
        plddt_{name}_model_{r}.npz      key plddt, (n_tokens,), 0-1
        pae_{name}_model_{r}.npz        key pae, (n_tokens, n_tokens)
        pde_{name}_model_{r}.npz        key pde, (n_tokens, n_tokens)

``r`` is the rank by ``confidence_score`` counted from 0, so sample ``k`` of the contract is
file ``r = k - 1``. The structure is written the way Boltz writes it, not the way biotite does
(``mmcif.py`` and ``pdb.py`` of Boltz v2.2.1): chain by chain, residues renumbered from 1 in
each chain, a column order of the mmCIF as in the Boltz example of ``ipsae.py``, ``.`` as the
sequence number of ligand rows, and the pLDDT of the token (0-100) in the B-factor column.

Boltz-2 has one pLDDT per token, so the per-atom truth of the contract cannot be written
exactly: the writer stores the mean of the atoms of each token, and the module sets
``PLDDT_ATOL`` to cover the spread inside a token of the default synthetic complex (2 points).
"""

from pathlib import Path

import numpy as np

#: The tokens of the default complex hold two atoms whose pLDDT differ by up to 4 points, so
#: the per-token mean is up to 2 points from the per-atom truth.
PLDDT_ATOL = 2.5
#: Boltz-2 writes one seed per run: a second seed is a second output directory.
SUPPORTS_SEED_INDEX = False
#: Suffix of the structure file (``.cif`` or ``.pdb``); tests change it.
STRUCTURE_SUFFIX = ".cif"

_CIF_TAGS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_seq_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "label_asym_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "label_entity_id",
    "auth_asym_id",
    "auth_comp_id",
    "B_iso_or_equiv",
    "pdbx_PDB_model_num",
)


def token_index_of_atoms(atoms) -> np.ndarray:
    """Index of the Boltz-2 token of each atom, from the tokenizer's rule.

    A hetero atom is a token of its own; the atoms of one residue (same chain, residue
    number and insertion code, not hetero) are one token, standard or modified.
    """
    n_atoms = atoms.array_length()
    index = np.empty(n_atoms, dtype=int)
    token = -1
    previous = None
    for i in range(n_atoms):
        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.ins_code[i]))
        hetero = bool(atoms.hetero[i])
        if hetero or previous is None or previous[1] or previous[0] != key:
            token += 1
        index[i] = token
        previous = (key, hetero)
    return index


def _boltz_numbering(atoms):
    """Chains in order of first appearance, and the residue number 1..N of each atom in its chain.

    Boltz numbers the residues of a chain from 1 whatever the numbers of the input; the atoms
    of a ligand residue share one number.

    Raises:
        ValueError: The atoms of a chain are not in one block, which Boltz never writes.
    """
    chains: list[str] = []
    numbers = np.empty(atoms.array_length(), dtype=int)
    last_residue = None
    number = 0
    for i in range(atoms.array_length()):
        chain = str(atoms.chain_id[i])
        if not chains or chains[-1] != chain:
            if chain in chains:
                raise ValueError(f"chain {chain} is not in one block; Boltz writes chain by chain")
            chains.append(chain)
            number = 0
            last_residue = None
        residue = (int(atoms.res_id[i]), str(atoms.ins_code[i]))
        if residue != last_residue:
            number += 1
            last_residue = residue
        numbers[i] = number
    return chains, numbers


def _quoted(value: str) -> str:
    return f'"{value}"' if "'" in value else value


def mmcif_text(atoms, bfactor: np.ndarray, data_name: str = "boltz") -> str:
    """An mmCIF the way Boltz writes one (see the module docstring)."""
    chains, numbers = _boltz_numbering(atoms)
    lines = [f"data_{data_name}", "#", "loop_", "_entity.id", "_entity.type"]
    lines += [f"{k + 1} polymer" for k in range(len(chains))] + ["#", "loop_"]
    lines += [f"_atom_site.{tag}" for tag in _CIF_TAGS]
    for i in range(atoms.array_length()):
        chain = str(atoms.chain_id[i])
        hetero = bool(atoms.hetero[i])
        res_name = str(atoms.res_name[i])
        x, y, z = (f"{c:.5f}" for c in atoms.coord[i])
        fields = [
            "HETATM" if hetero else "ATOM",
            str(i + 1),
            str(atoms.element[i]).upper(),
            _quoted(str(atoms.atom_name[i])),
            ".",
            res_name,
            "." if hetero else str(numbers[i]),
            str(numbers[i]),
            "?",
            chain,
            x,
            y,
            z,
            "1",
            str(chains.index(chain) + 1),
            chain,
            res_name,
            f"{round(float(bfactor[i]), 3)}",
            "1",
        ]
        lines.append(" ".join(fields))
    lines.append("#")
    return "\n".join(lines) + "\n"


def pdb_text(atoms, bfactor: np.ndarray) -> str:
    """A PDB file the way Boltz writes one (``pdb.py:127-134``): ligand residues are ``LIG``."""
    chains, numbers = _boltz_numbering(atoms)
    lines = []
    serial = 1
    for i in range(atoms.array_length()):
        hetero = bool(atoms.hetero[i])
        name = str(atoms.atom_name[i])
        name = name if len(name) == 4 else f" {name}"
        res_name = "LIG" if hetero else str(atoms.res_name[i])[:3]
        x, y, z = atoms.coord[i]
        lines.append(
            f"{'HETATM' if hetero else 'ATOM':<6}{serial:>5} {name:<4}{'':>1}"
            f"{res_name:>3} {str(atoms.chain_id[i]):>1}"
            f"{numbers[i]:>4}{'':>1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{1.0:>6.2f}{float(bfactor[i]):>6.2f}          "
            f"{str(atoms.element[i]).upper():>2}{'':>2}"
        )
        serial += 1
        ends_chain = i + 1 == atoms.array_length() or atoms.chain_id[i + 1] != atoms.chain_id[i]
        if ends_chain and str(atoms.chain_id[i]) != chains[-1]:
            lines.append(
                f"{'TER':<6}{serial:>5}      {res_name:>3} {str(atoms.chain_id[i]):>1}"
                f"{numbers[i]:>4}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(line.ljust(80) for line in lines) + "\n"


def write_boltz_structure(atoms, path: Path, bfactor: np.ndarray) -> Path:
    """Write ``atoms`` as Boltz would, mmCIF or PDB by the suffix of ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = mmcif_text(atoms, bfactor) if path.suffix == ".cif" else pdb_text(atoms, bfactor)
    path.write_text(text, encoding="utf-8")
    return path
