"""Writer of a synthetic Protenix output directory (layout in ``predictors/protenix.py``).

    {directory}/{name}/seed_{S}/predictions/{name}_sample_{r}.cif
                                            {name}_summary_confidence_sample_{r}.json
                                            {name}_full_data_sample_{r}.json

``sample`` is 1-based and ``r = sample - 1``. ``seed_index`` 1, 2, 3 are written as the seed
values 9, 10, 11: their numeric order is the position order while their string order is not,
so a parser that sorts seed directories as strings fails the seed check of the contract tests.

The writer follows the writer code of Protenix (commit 85767b8) where the format report and the
source say what a file holds: the summary values are not rounded, the full-data arrays are
rounded to 2 decimals, ``atom_plddt`` is on 0-1, ``disorder`` is always 0, ``chain_pair_iptm``
is a matrix with a zero diagonal, and the tokens of a residue outside the standard set are one
per atom. It accepts any ``SyntheticComplex``:

* the chains are numbered by order of first appearance in the atoms, and the per-chain lists
  take their values from ``chain_ptm`` and ``chain_pair_iptm`` of the complex in the order of
  the keys of ``chain_ptm`` (the keys need not be the chain IDs of the atoms);
* the token of each atom is one per residue when ``pae`` has one row per residue, one per atom
  for a residue outside the standard set when that gives ``pae`` its size, and otherwise the
  token arrays are left out of the file (a real file always has them).
"""

from pathlib import Path
from typing import Optional

import numpy as np

from tests.predictors import synth

#: ``atom_plddt`` is rounded to 2 decimals on 0-1, so a value is off by up to 0.5 on 0-100.
PLDDT_ATOL = 0.51

#: Tests set this to False to mimic a run without ``--need_atom_confidence true``.
WRITE_FULL_DATA = True

#: Residue names that Protenix tokenises as one token (constants.py:270-314 of the source).
_STANDARD = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK "
    "A G C U N DA DG DC DT DN".split()
)

_NUM_RECYCLES = 10


def seed_value(seed_index: int) -> int:
    return 8 + seed_index


def _residues(atoms) -> list[tuple[int, int]]:
    """``(start, stop)`` atom ranges of the residues, a residue being a run of equal IDs."""
    ins_code = (
        np.asarray(atoms.ins_code)
        if "ins_code" in atoms.get_annotation_categories()
        else np.full(atoms.array_length(), "")
    )
    keys = list(zip(map(str, atoms.chain_id), map(int, atoms.res_id), map(str, ins_code)))
    bounds = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    return list(zip(bounds[:-1], bounds[1:]))


def token_indices(atoms, n_tokens: int) -> Optional[np.ndarray]:
    """``atom_to_token_idx`` for a structure whose PAE has ``n_tokens`` rows, or None.

    One token per residue when that gives ``n_tokens``; otherwise the Protenix rule (a residue
    outside the standard set is one token per atom); None when neither gives ``n_tokens``.
    """
    residues = _residues(atoms)
    names = np.asarray(atoms.res_name)
    per_residue = np.zeros(atoms.array_length(), dtype=int)
    for token, (start, stop) in enumerate(residues):
        per_residue[start:stop] = token
    if len(residues) == n_tokens:
        return per_residue
    token = 0
    per_rule = np.zeros(atoms.array_length(), dtype=int)
    for start, stop in residues:
        if str(names[start]) in _STANDARD:
            per_rule[start:stop] = token
            token += 1
        else:
            per_rule[start:stop] = token + np.arange(stop - start)
            token += stop - start
    return per_rule if token == n_tokens else None


def _chain_positions(atoms) -> dict[str, int]:
    return {chain: i for i, chain in enumerate(dict.fromkeys(map(str, atoms.chain_id)))}


def _per_chain(values: list[float], n_chains: int) -> list[float]:
    return (list(values) + [0.0] * n_chains)[:n_chains]


def _pair_matrix(complex_: synth.SyntheticComplex, n_chains: int, scale: float = 1.0):
    """``chain_pair_iptm`` as a matrix: ``"P-Q"`` goes to the row of ``P`` and column of ``Q``.

    ``P`` and ``Q`` are looked up in the keys of ``chain_ptm`` (the order of the chains); the
    diagonal is 0, as in Protenix.
    """
    order = list(complex_.chain_ptm)
    matrix = np.zeros((n_chains, n_chains))
    for key, value in complex_.chain_pair_iptm.items():
        first, second = key.split("-")
        if first in order and second in order:
            i, j = order.index(first), order.index(second)
            if i < n_chains and j < n_chains and i != j:
                matrix[i, j] = value * scale
    return matrix


def write_prediction(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    seed_index: int = 1,
    sample: int = 1,
) -> None:
    """Write ``complex_`` as Protenix sample ``sample`` (rank ``sample - 1``) of one seed."""
    predictions = Path(directory) / name / f"seed_{seed_value(seed_index)}" / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    rank = sample - 1
    scalars = complex_.scalars
    atoms = complex_.atoms
    n_chains = len(_chain_positions(atoms))

    synth.write_structure(atoms, predictions / f"{name}_sample_{rank}.cif")

    chain_ptm = _per_chain(list(complex_.chain_ptm.values()), n_chains)
    synth.write_json(
        predictions / f"{name}_summary_confidence_sample_{rank}.json",
        {
            "plddt": scalars["avg_plddt"],
            "gpde": scalars["gpde"],
            "ptm": scalars["ptm"],
            "iptm": scalars["iptm"],
            "chain_ptm": chain_ptm,
            "chain_iptm": _per_chain([0.5] * n_chains, n_chains),
            "chain_pair_iptm": _pair_matrix(complex_, n_chains),
            "chain_pair_iptm_global": _pair_matrix(complex_, n_chains, 0.5),
            "chain_plddt": _per_chain([scalars["avg_plddt"]] * n_chains, n_chains),
            "chain_pair_plddt": np.full((n_chains, n_chains), scalars["avg_plddt"]),
            "chain_gpde": _per_chain([scalars["gpde"]] * n_chains, n_chains),
            "chain_pair_gpde": np.full((n_chains, n_chains), scalars["gpde"]),
            "has_clash": float(scalars["has_clash"]),
            "disorder": 0.0,
            "ranking_score": scalars["ranking_score"],
            "num_recycles": _NUM_RECYCLES,
        },
    )
    if not WRITE_FULL_DATA:
        return

    n_tokens = complex_.n_tokens
    i = np.arange(n_tokens)[:, None]
    j = np.arange(n_tokens)[None, :]
    full = {
        "atom_plddt": np.round(complex_.plddt_per_atom / 100.0, 2),
        "token_pair_pde": np.round(complex_.pde, 2),
        "contact_probs": np.round(1.0 / (1.0 + np.abs(i - j)), 2),
        "token_pair_pae": np.round(complex_.pae, 2),
    }
    atom_to_token = token_indices(atoms, n_tokens)
    if atom_to_token is not None:
        positions = _chain_positions(atoms)
        chain_of_atom = np.array([positions[str(c)] for c in atoms.chain_id])
        first_atom = np.searchsorted(atom_to_token, np.arange(n_tokens), side="left")
        full["token_asym_id"] = chain_of_atom[first_atom]
        full["token_has_frame"] = np.ones(n_tokens, dtype=int)
        full["atom_to_token_idx"] = atom_to_token
    synth.write_json(predictions / f"{name}_full_data_sample_{rank}.json", full)
