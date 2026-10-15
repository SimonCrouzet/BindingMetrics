"""One tiny synthetic complex and the helpers that write it to disk, for the adapter tests.

Every model's writer (``tests/predictors/synth_<model>.py``) serialises the same
``SyntheticComplex`` in that model's layout, so the contract tests and the cross-model tests
compare records with known truth. Nothing here is a real model output: the values are round
numbers picked so that a wrong scale, a transposed matrix or a shifted chain shows up.

The complex has two chains, A (4 residues) and B (3 residues), each residue with a CA and a
CB atom: 14 atoms and 7 tokens (one token per residue).

    pde[i, j] = 0.5 + 0.25 i + 0.125 j        pae[i, j] = 1 + 0.5 i + 0.25 j

Both are asymmetric, so a transposed matrix differs from the truth. pLDDT is on 0-100.

Writer protocol (what ``tests/predictors/synth_<model>.py`` provides, and what the contract
tests in ``contract.py`` call):

    write_prediction(directory, name, complex_, *, seed_index=1, sample=1) -> None

writes the smallest output of the model that describes ``complex_`` under ``directory``, in
the model's own layout and scales, so that ``get_parser("<model>").load(directory, name,
seed_index=seed_index, sample=sample)`` returns a record equal to the truth within the
writer's rounding. The directory exists when the writer is called. A module may set

    PLDDT_ATOL, SCALAR_ATOL, MATRIX_ATOL    tolerances for a model that rounds its values
                                            (default 0.05, 0.01 and 0.01)
    SUPPORTS_SEED_INDEX                     False for a model with no seed dimension (default
                                            True); the seed check then uses samples only

and it writes only what its model writes: a model without PDE leaves ``pde`` out, and the
contract compares only the values the parser returns. Files are written at test time; nothing
is committed.
"""

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

struc = pytest.importorskip("biotite.structure")
pdbx = pytest.importorskip("biotite.structure.io.pdbx")
pdb_io = pytest.importorskip("biotite.structure.io.pdb")

CHAIN_RESIDUES = {"A": 4, "B": 3}
N_ATOMS = 14
N_TOKENS = 7

_PLDDT = np.array([92, 94, 90, 88, 86, 84, 80, 78, 60, 64, 88, 90, 75, 79], dtype=float)


@dataclass(frozen=True, eq=False)
class SyntheticComplex:
    """The truth that a writer serialises and a parser must return.

    Attributes:
        atoms: The structure (chains A and B, CA and CB per residue, ``b_factor`` = pLDDT).
        plddt_per_atom: ``(14,)``, 0-100, in the atom order of ``atoms``.
        pae, pde: ``(7, 7)`` in angstrom, ``pae[i, j]`` the error of token j aligned on i.
        scalars: ``avg_plddt``, ``ptm``, ``iptm``, ``gpde``, ``ranking_score``, ``has_clash``,
            ``disorder``; a writer leaves out those its model does not have.
        chain_ptm, chain_pair_iptm: per chain (``"A"``) and per ordered pair (``"A-B"``).
    """

    atoms: Any
    plddt_per_atom: np.ndarray
    pae: np.ndarray
    pde: np.ndarray
    scalars: dict[str, float]
    chain_ptm: dict[str, float] = field(default_factory=dict)
    chain_pair_iptm: dict[str, float] = field(default_factory=dict)

    @property
    def chain_ids(self) -> tuple[str, ...]:
        return tuple(CHAIN_RESIDUES)

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.array_length())

    @property
    def n_tokens(self) -> int:
        return int(self.pae.shape[0])


def _build_atoms() -> Any:
    atoms = []
    for chain_id, n_residues in CHAIN_RESIDUES.items():
        y = 0.0 if chain_id == "A" else 6.0
        for i in range(n_residues):
            for name, dy in (("CA", 0.0), ("CB", 1.5)):
                atoms.append(
                    struc.Atom(
                        [3.8 * i, y + dy, 0.0],
                        chain_id=chain_id,
                        res_id=i + 1,
                        res_name="ALA",
                        atom_name=name,
                        element="C",
                    )
                )
    return struc.array(atoms)


def synthetic_complex(plddt_shift: float = 0.0) -> SyntheticComplex:
    """The synthetic complex; ``plddt_shift`` lowers every pLDDT (a different sample).

    Args:
        plddt_shift: Subtracted from every per-atom pLDDT (and so from ``avg_plddt``).
    """
    plddt = _PLDDT - plddt_shift
    atoms = _build_atoms()
    atoms.set_annotation("b_factor", plddt.copy())
    i = np.arange(N_TOKENS)[:, None]
    j = np.arange(N_TOKENS)[None, :]
    return SyntheticComplex(
        atoms=atoms,
        plddt_per_atom=plddt,
        pae=1.0 + 0.5 * i + 0.25 * j,
        pde=0.5 + 0.25 * i + 0.125 * j,
        scalars={
            "avg_plddt": float(plddt.mean()),
            "ptm": 0.88,
            "iptm": 0.76,
            "gpde": 1.23,
            "ranking_score": 0.82,
            "has_clash": 0.0,
            "disorder": 0.12,
        },
        chain_ptm={"A": 0.88, "B": 0.80},
        chain_pair_iptm={"A-B": 0.76, "B-A": 0.74},
    )


def renamed_atoms(complex_: SyntheticComplex, chain_ids: dict[str, str]):
    """A copy of the structure with chains renamed all at once (a swap works)."""
    atoms = complex_.atoms.copy()
    old = atoms.chain_id.copy()
    new = np.array([chain_ids.get(str(c), str(c)) for c in old])
    atoms.chain_id = new
    return atoms


# ---------------------------------------------------------------------- writers' tools


def write_structure(atoms, path: Path, *, bfactor_scale: float = 1.0) -> Path:
    """Write ``atoms`` as mmCIF or PDB (by the suffix, ``.gz`` compresses); return the path.

    The B-factor column holds the ``b_factor`` annotation of ``atoms`` times
    ``bfactor_scale`` (use 0.01 to mimic a model that writes pLDDT on 0-1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = atoms.copy()
    if "b_factor" in atoms.get_annotation_categories():
        atoms.set_annotation("b_factor", np.asarray(atoms.b_factor) * bfactor_scale)
    plain = path.with_suffix("") if path.suffix == ".gz" else path
    if plain.suffix in (".cif", ".mmcif"):
        cif = pdbx.CIFFile()
        pdbx.set_structure(cif, atoms)
        cif.write(str(plain))
    else:
        pdb = pdb_io.PDBFile()
        pdb_io.set_structure(pdb, atoms)
        pdb.write(str(plain))
    if path.suffix == ".gz":
        with open(plain, "rb") as source, gzip.open(path, "wb") as target:
            target.write(source.read())
        plain.unlink()
    return path


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as JSON; numpy arrays and scalars become lists and numbers."""

    def _default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot serialise {type(value).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_default), encoding="utf-8")
    return path


def write_npz(path: Path, **arrays: np.ndarray) -> Path:
    """Write an uncompressed ``.npz`` (built with ``np.savez``, never committed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)
    return path
