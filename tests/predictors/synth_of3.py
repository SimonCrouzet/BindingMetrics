"""Writer of a synthetic OpenFold3 output directory (layout in ``predictors/of3.py``).

    {directory}/{name}/seed_{S}/{name}_seed_{S}_sample_{k}_model.cif
                              /{name}_seed_{S}_sample_{k}_confidences_aggregated.json
                              /{name}_seed_{S}_sample_{k}_confidences.json
                              /timing.json

``seed_index`` 1, 2, 3 are written as the seed values 9, 10, 11: their numeric order is the
position order while their string order is not, so a parser that sorts seed directories as
strings fails the seed check of the contract tests.
"""

from pathlib import Path

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
