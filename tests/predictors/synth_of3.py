"""Writer of a synthetic OpenFold3 output directory (layout in ``predictors/of3.py``).

    {directory}/{name}/seed_{S}/{name}_seed_{S}_sample_{k}_model.cif
                              /{name}_seed_{S}_sample_{k}_confidences_aggregated.json
                              /{name}_seed_{S}_sample_{k}_confidences.json
                              /timing.json

``seed_index`` 1, 2, 3 are written as the seed values 9, 10, 11: their numeric order is the
position order while their string order is not, so a parser that sorts seed directories as
strings fails the seed check of the contract tests.

``complex_with_a_modified_residue`` builds a ``SyntheticComplex`` that has the tokenisation of a
real run with a modified residue (one token per heavy atom of that residue, one token per other
residue): the writer serialises it like any other complex.
"""

import dataclasses
from pathlib import Path

import numpy as np

from tests.predictors import synth

#: Suffix of the structure file (``.cif``, ``.cif.gz`` or ``.pdb``); tests change it.
STRUCTURE_SUFFIX = ".cif"
#: ``full_confidence_output_format``: ``json`` or ``npz``; tests change it.
CONFIDENCE_FORMAT = "json"


def seed_value(seed_index: int) -> int:
    return 8 + seed_index


def _pair_key(key: str) -> str:
    first, second = key.split("-")
    return f"({first}, {second})"


def write_prediction(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    seed_index: int = 1,
    sample: int = 1,
) -> None:
    """Write ``complex_`` as OpenFold3 sample ``sample`` of seed directory ``seed_index``."""
    seed = seed_value(seed_index)
    seed_dir = Path(directory) / name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prefix = seed_dir / f"{name}_seed_{seed}_sample_{sample}"
    scalars = complex_.scalars

    synth.write_structure(complex_.atoms, seed_dir / f"{prefix.name}_model{STRUCTURE_SUFFIX}")
    synth.write_json(
        seed_dir / f"{prefix.name}_confidences_aggregated.json",
        {
            "avg_plddt": scalars["avg_plddt"],
            "gpde": scalars["gpde"],
            "ptm": scalars["ptm"],
            "iptm": scalars["iptm"],
            "disorder": scalars["disorder"],
            "has_clash": scalars["has_clash"],
            "sample_ranking_score": scalars["ranking_score"],
            "chain_ptm": complex_.chain_ptm,
            "chain_pair_iptm": {_pair_key(k): v for k, v in complex_.chain_pair_iptm.items()},
            "bespoke_iptm": {"(A, B)": 0.74},
        },
    )
    arrays = {"plddt": complex_.plddt_per_atom, "pde": complex_.pde, "pae": complex_.pae}
    if CONFIDENCE_FORMAT == "npz":
        synth.write_npz(seed_dir / f"{prefix.name}_confidences.npz", **arrays)
    else:
        synth.write_json(seed_dir / f"{prefix.name}_confidences.json", arrays)
    synth.write_json(seed_dir / "timing.json", {"runtime_s": 12.5})


def _atom(chain, res_id, res_name, name, x, y, hetero=False):
    return synth.struc.Atom(
        [x, y, 0.0],
        chain_id=chain,
        res_id=res_id,
        res_name=res_name,
        atom_name=name,
        element=name[0],
        hetero=hetero,
    )


def complex_with_a_modified_residue() -> synth.SyntheticComplex:
    """Receptor A of three alanines; binder B of ALA, MLE (N-methyl-Leu), ALA.

    OpenFold3 makes one token of each residue in the standard set and one token of each heavy
    atom of the modified residue: chain A has 3 tokens, chain B has 1 + 5 + 1 = 7, so ``pae`` and
    ``pde`` are 10 x 10 for 6 residues (the shape of 1CWA, 240 tokens for 176 residues). MLE
    has the atoms N, CA, C, O and CB here, each its own token; the other residues have CA and CB.

        pae[i, j] = 1 + 0.5 i + 0.25 j        pde[i, j] = 0.5 + 0.25 i + 0.125 j

    as in ``synth.synthetic_complex``, so the interface values can be written down by hand:
    the binder rows (tokens 3-9) by receptor columns (0-2) of ``pae`` average 4.25, the other
    block 3.0 and the interface PAE is their mean, 3.625.
    """
    atoms = [
        _atom("A", 1, "ALA", "CA", 0.0, 0.0),
        _atom("A", 1, "ALA", "CB", 0.0, 1.5),
        _atom("A", 2, "ALA", "CA", 3.8, 0.0),
        _atom("A", 2, "ALA", "CB", 3.8, 1.5),
        _atom("A", 3, "ALA", "CA", 7.6, 0.0),
        _atom("A", 3, "ALA", "CB", 7.6, 1.5),
        _atom("B", 1, "ALA", "CA", 0.0, 6.0),
        _atom("B", 1, "ALA", "CB", 0.0, 7.5),
        _atom("B", 2, "MLE", "N", 2.4, 6.0, hetero=True),
        _atom("B", 2, "MLE", "CA", 3.8, 6.0, hetero=True),
        _atom("B", 2, "MLE", "C", 5.2, 6.0, hetero=True),
        _atom("B", 2, "MLE", "O", 5.2, 7.2, hetero=True),
        _atom("B", 2, "MLE", "CB", 3.8, 7.5, hetero=True),
        _atom("B", 3, "ALA", "CA", 7.6, 6.0),
        _atom("B", 3, "ALA", "CB", 7.6, 7.5),
    ]
    array = synth.struc.array(atoms)
    plddt = np.linspace(95.0, 65.0, len(atoms))
    array.set_annotation("b_factor", plddt.copy())
    n_tokens = 10
    i = np.arange(n_tokens)[:, None]
    j = np.arange(n_tokens)[None, :]
    base = synth.synthetic_complex()
    return dataclasses.replace(
        base,
        atoms=array,
        plddt_per_atom=plddt,
        pae=1.0 + 0.5 * i + 0.25 * j,
        pde=0.5 + 0.25 * i + 0.125 * j,
        scalars={**base.scalars, "avg_plddt": float(plddt.mean())},
    )
