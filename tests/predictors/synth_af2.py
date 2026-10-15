"""Writers of synthetic AlphaFold2 and ColabFold output (layouts in ``predictors/af2.py``).

``write_prediction`` is the writer of the contract tests: a ColabFold job with the multimer-v3
model type and ``--calc-extra-ptm``,

    {directory}/{name}_unrelaxed_rank_{RRR}_alphafold2_multimer_v3_model_{k}_seed_{SSS}.pdb
    {directory}/{name}_scores_rank_{RRR}_alphafold2_multimer_v3_model_{k}_seed_{SSS}.json

The scores JSON has the keys of a real ColabFold multimer-v3 file, in its order: ``plddt``,
``max_pae``, ``pae``, ``pairwise_actifptm``, ``pairwise_iptm``, ``per_chain_ptm``, ``actifptm``,
``ptm``, ``iptm``, with two decimals. The seed position maps to the seed values 99, 100, 101 and
the sample position to a rank, and the model number is not in rank order, so a parser that sorts
by model number or by file name fails the seed and sample check of the contract tests.

The other writers serve ``test_af2.py``: ``write_colabfold`` (relaxed file, no extra pTM, another
directory), ``write_alphafold`` (the v2.3.2 layout with a result pickle, ``ranking_debug.json``
and ``timings.json``, or the AlphaFold2 main JSON files) and ``write_bare`` (a structure alone).

Every writer takes the structure from a ``SyntheticComplex`` and writes what the model writes:
one pLDDT per residue, the mean over the atoms of the residue in the truth, and the same value in
the B-factor column of each atom of the residue. The truth has one pLDDT per atom, so
``PLDDT_ATOL`` is the largest difference between an atom and its residue mean in the default
complex.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from tests.predictors import synth

#: The default complex has residues whose two atoms differ by up to 4, so a residue mean is
#: within 2 of each atom.
PLDDT_ATOL = 2.01

#: The pLDDT of a residue is repeated over its atoms by reading the structure file, so a test
#: cannot replace that file by a stub (``contract.check_scalars_parse_without_biotite`` does).
PLDDT_FROM_STRUCTURE = True

MODEL_TYPE = "alphafold2_multimer_v3"

#: Model number of the sample at each position: not in rank order on purpose.
_MODEL_OF_SAMPLE = {1: 3, 2: 1, 3: 4, 4: 2, 5: 5}


def seed_value(seed_index: int) -> int:
    return 98 + seed_index


def rank_number(seed_index: int, sample: int) -> int:
    """ColabFold ranks are global to a job; here they grow with the seed position."""
    return 10 * (seed_index - 1) + sample


def colabfold_tag(seed_index: int = 1, sample: int = 1) -> str:
    return (
        f"rank_{rank_number(seed_index, sample):03d}_{MODEL_TYPE}_model_"
        f"{_MODEL_OF_SAMPLE[sample]}_seed_{seed_value(seed_index):03d}"
    )


# ---------------------------------------------------------------------- what the model writes


def residue_starts(complex_: synth.SyntheticComplex) -> np.ndarray:
    return synth.struc.get_residue_starts(complex_.atoms)


def plddt_per_residue(complex_: synth.SyntheticComplex) -> np.ndarray:
    """The mean pLDDT of the atoms of each residue, in the order of the structure."""
    starts = residue_starts(complex_)
    counts = np.diff(np.append(starts, complex_.n_atoms))
    return np.add.reduceat(complex_.plddt_per_atom, starts) / counts


def af2_atoms(complex_: synth.SyntheticComplex):
    """The structure with the pLDDT of its residue in the B-factor of each atom."""
    starts = residue_starts(complex_)
    counts = np.diff(np.append(starts, complex_.n_atoms))
    atoms = complex_.atoms.copy()
    atoms.set_annotation("b_factor", np.repeat(plddt_per_residue(complex_), counts))
    return atoms


def _check_tokens_are_residues(complex_: synth.SyntheticComplex) -> None:
    n_residues = len(residue_starts(complex_))
    if complex_.n_tokens != n_residues:
        raise ValueError(
            f"AlphaFold2 has one token per residue: the complex has {n_residues} residues "
            f"and {complex_.n_tokens} tokens"
        )


def _upper_triangle(pairs: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in pairs.items() if key.split("-")[0] < key.split("-")[1]}


def colabfold_scores(complex_: synth.SyntheticComplex, *, extra_ptm: bool = True) -> dict:
    """The scores JSON of ColabFold for ``complex_``, values rounded to two decimals."""
    _check_tokens_are_residues(complex_)
    scalars = complex_.scalars
    payload = {
        "plddt": np.round(plddt_per_residue(complex_), 2).tolist(),
        "max_pae": round(float(complex_.pae.max()), 2),
        "pae": np.round(complex_.pae, 2).tolist(),
    }
    if extra_ptm:
        payload["pairwise_actifptm"] = _upper_triangle(complex_.chain_pair_iptm)
        payload["pairwise_iptm"] = _upper_triangle(complex_.chain_pair_iptm)
        payload["per_chain_ptm"] = dict(complex_.chain_ptm)
        payload["actifptm"] = round(scalars["iptm"] - 0.02, 2)
    payload["ptm"] = round(scalars["ptm"], 2)
    payload["iptm"] = round(scalars["iptm"], 2)
    return payload


# ---------------------------------------------------------------------- ColabFold


def write_colabfold(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    seed_index: int = 1,
    sample: int = 1,
    relaxed: bool = False,
    unrelaxed: bool = True,
    extra_ptm: bool = True,
    scores: Optional[dict] = None,
) -> str:
    """Write one ColabFold sample and return its tag (``rank_001_..._seed_099``).

    Args:
        relaxed, unrelaxed: Which structure files to write; ``--num-relax`` writes both.
        extra_ptm: Write the ``--calc-extra-ptm`` keys.
        scores: Replace the scores payload (for a file with another set of keys).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tag = colabfold_tag(seed_index, sample)
    atoms = af2_atoms(complex_)
    for written, kind in ((unrelaxed, "unrelaxed"), (relaxed, "relaxed")):
        if written:
            synth.write_structure(atoms, directory / f"{name}_{kind}_{tag}.pdb")
    payload = colabfold_scores(complex_, extra_ptm=extra_ptm) if scores is None else scores
    synth.write_json(directory / f"{name}_scores_{tag}.json", payload)
    return tag


def write_prediction(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    seed_index: int = 1,
    sample: int = 1,
) -> None:
    """The writer of the contract tests: a ColabFold job, ``--calc-extra-ptm``, no relaxation."""
    write_colabfold(directory, name, complex_, seed_index=seed_index, sample=sample)


# ---------------------------------------------------------------------- bare structure


def write_bare(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    suffix: str = ".pdb",
    bfactor_scale: float = 1.0,
) -> Path:
    """A structure alone, with the pLDDT of each residue in the B-factor column."""
    return synth.write_structure(
        af2_atoms(complex_), Path(directory) / f"{name}{suffix}", bfactor_scale=bfactor_scale
    )


# ---------------------------------------------------------------------- AlphaFold2 layouts


def alphafold_id(model: str = "model_1_multimer_v3", pred: int = 0) -> str:
    return f"{model}_pred_{pred}"


def result_pickle(complex_: synth.SyntheticComplex) -> dict:
    """What ``run_alphafold.py`` pickles for a multimer prediction (the keys of the report)."""
    _check_tokens_are_residues(complex_)
    n = complex_.n_tokens
    scalars = complex_.scalars
    return {
        "plddt": plddt_per_residue(complex_).astype(np.float32),
        "predicted_aligned_error": complex_.pae.astype(np.float32),
        "max_predicted_aligned_error": np.float32(31.75),
        "aligned_confidence_probs": np.zeros((n, n, 4), dtype=np.float32),
        "distogram": {"logits": np.zeros((n, n, 4), dtype=np.float32), "bin_edges": np.arange(3.0)},
        "ptm": np.float32(scalars["ptm"]),
        "iptm": np.float32(scalars["iptm"]),
        "ranking_confidence": np.float32(0.8 * scalars["iptm"] + 0.2 * scalars["ptm"]),
    }


def write_alphafold(
    directory: Path,
    complex_: synth.SyntheticComplex,
    *,
    pred: int = 0,
    model: str = "model_1_multimer_v3",
    relaxed: bool = False,
    unrelaxed: bool = True,
    result: Optional[dict] = None,
    write_result: bool = True,
    protocol: int = 4,
    main_json: bool = False,
    ranking: Optional[float] = None,
    timings: bool = True,
) -> str:
    """Write one AlphaFold2 v2.3.2 prediction into ``directory`` and return its id.

    Args:
        pred: The prediction number (``pred_{i}``), the seed position minus one.
        model: The model name, whose number gives the sample position.
        result: Replace the pickled dictionary.
        write_result: Write the result pickle (False leaves only the structure).
        protocol: Pickle protocol; AlphaFold2 uses 4.
        main_json: Also write ``confidence_{id}.json`` and ``pae_{id}.json`` of AlphaFold2 main
            (list-wrapped, one decimal), and no pickle when ``write_result`` is False.
        ranking: The ranking score to add to ``ranking_debug.json`` (None: the ipTM+pTM of the
            complex); the file lists every prediction written so far in its ``order``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prediction_id = alphafold_id(model, pred)
    atoms = af2_atoms(complex_)
    for written, kind in ((unrelaxed, "unrelaxed"), (relaxed, "relaxed")):
        if written:
            synth.write_structure(atoms, directory / f"{kind}_{prediction_id}.pdb")
    if write_result:
        payload = result_pickle(complex_) if result is None else result
        with open(directory / f"result_{prediction_id}.pkl", "wb") as handle:
            pickle.dump(payload, handle, protocol=protocol)
    if main_json:
        residues = plddt_per_residue(complex_)
        synth.write_json(
            directory / f"confidence_{prediction_id}.json",
            {
                "residueNumber": list(range(1, len(residues) + 1)),
                "confidenceScore": np.round(residues, 2).tolist(),
                "confidenceCategory": ["D"] * len(residues),
            },
        )
        synth.write_json(
            directory / f"pae_{prediction_id}.json",
            [
                {
                    "predicted_aligned_error": np.round(complex_.pae, 1).tolist(),
                    "max_predicted_aligned_error": 31.75,
                }
            ],
        )
    _update_ranking_debug(directory, prediction_id, complex_, ranking)
    if timings:
        synth.write_json(
            directory / "timings.json", {"features": 1.5, f"predict_{prediction_id}": 7.25}
        )
    return prediction_id


def _update_ranking_debug(directory: Path, prediction_id: str, complex_, ranking) -> None:
    path = directory / "ranking_debug.json"
    debug = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"iptm+ptm": {}, "order": []}
    )
    scalars = complex_.scalars
    score = 0.8 * scalars["iptm"] + 0.2 * scalars["ptm"] if ranking is None else ranking
    debug["iptm+ptm"][prediction_id] = score
    debug["order"] = sorted(debug["iptm+ptm"], key=lambda key: -debug["iptm+ptm"][key])
    path.write_text(json.dumps(debug), encoding="utf-8")
