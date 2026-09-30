"""A made-up model, "stub", that exercises the adapter machinery without any real format.

It is not a prediction model: it exists so that the base class, the registry and the contract
harness can be tested before (and independently of) a real adapter. It also shows what an
adapter and its ``synth_<model>.py`` writer look like, so a new model's files can be modelled
on it.

Layout (all files of one sample share a stem, ``{name}_s{seed_index}_m{sample}``):

    {stem}.cif             structure, pLDDT on 0-100 in the B-factor column
    {stem}.scores.json     plddt_mean (0-1), ptm, iptm, gpde, score, has_clash, disorder,
                           chain_ptm {"A": x}, chain_pair_iptm {"A-B": x}, stub_only
    {stem}.arrays.npz      plddt [n_atoms] (0-1), pae and pde [n_tokens, n_tokens]
    {stem}.timing.json     {"inference": seconds}

The stub writes pLDDT on 0-1, so its parser has a scale conversion to do, and it names its
ranking score ``stub_score``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.record import PredictionFiles, PredictionRecord
from tests.predictors import synth

_NAN = float("nan")


def _stem(directory: Path, name: str, seed_index: int, sample: int) -> Path:
    return Path(directory) / f"{name}_s{seed_index}_m{sample}"


class StubParser(PredictionParser):
    """Adapter of the stub model; follows the parser contract."""

    name = "stub"
    display_name = "Stub model"
    family = "af3"

    def find_files(
        self, prediction_dir: Path, name: str, *, seed_index: int = 1, sample: int = 1
    ) -> PredictionFiles:
        stem = _stem(prediction_dir, name, seed_index, sample)

        def _existing(suffix: str):
            path = stem.with_name(stem.name + suffix)
            return path if path.exists() else None

        return PredictionFiles(
            directory=Path(prediction_dir),
            structure=_existing(".cif"),
            scores=_existing(".scores.json"),
            arrays=_existing(".arrays.npz"),
            timing=_existing(".timing.json"),
        )

    def parse(
        self, files: PredictionFiles, *, name: str, seed_index: int = 1, sample: int = 1
    ) -> PredictionRecord:
        record = PredictionRecord(
            self.name,
            name,
            seed_index=seed_index,
            sample=sample,
            structure_path=files.structure,
        )
        if not files.any_found():
            record.reasons.append(
                f"no stub output found for '{name}' (seed index {seed_index}, sample {sample}) "
                f"in {files.directory}"
            )
            return record

        if files.scores is None:
            record.reasons.append("scores file not found")
        else:
            with open(files.scores, encoding="utf-8") as fh:
                raw = json.load(fh)

            def _scalar(key: str) -> float:
                value = raw.get(key)
                return _NAN if value is None else float(value)

            record.avg_plddt = 100.0 * _scalar("plddt_mean")  # the stub writes 0-1
            record.ptm = _scalar("ptm")
            record.iptm = _scalar("iptm")
            record.gpde = _scalar("gpde")
            record.ranking_score = _scalar("score")
            record.ranking_score_name = "stub_score"
            record.has_clash = _scalar("has_clash")
            record.disorder = _scalar("disorder")
            record.chain_ptm = dict(raw.get("chain_ptm", {}))
            record.chain_pair_iptm = dict(raw.get("chain_pair_iptm", {}))
            if "stub_only" in raw:
                record.extras["stub_only"] = raw["stub_only"]

        if files.arrays is None:
            record.reasons.append("arrays file not found")
        else:
            with np.load(files.arrays) as data:
                record.plddt_per_atom = 100.0 * np.asarray(data["plddt"], dtype=float)
                record.pae = np.asarray(data["pae"], dtype=float)
                record.pde = np.asarray(data["pde"], dtype=float)
            if np.isnan(record.avg_plddt):
                record.avg_plddt = float(record.plddt_per_atom.mean())

        if files.timing is not None:
            with open(files.timing, encoding="utf-8") as fh:
                record.timing = json.load(fh)
        return record


def write_prediction(
    directory: Path,
    name: str,
    complex_: synth.SyntheticComplex,
    *,
    seed_index: int = 1,
    sample: int = 1,
) -> None:
    """Write ``complex_`` as the stub model's files (see the module docstring)."""
    stem = _stem(directory, name, seed_index, sample)
    Path(directory).mkdir(parents=True, exist_ok=True)
    scalars = complex_.scalars
    synth.write_structure(complex_.atoms, stem.with_name(stem.name + ".cif"))
    payload: dict[str, Any] = {
        "plddt_mean": scalars["avg_plddt"] / 100.0,
        "ptm": scalars["ptm"],
        "iptm": scalars["iptm"],
        "gpde": scalars["gpde"],
        "score": scalars["ranking_score"],
        "has_clash": scalars["has_clash"],
        "disorder": scalars["disorder"],
        "chain_ptm": complex_.chain_ptm,
        "chain_pair_iptm": complex_.chain_pair_iptm,
        "stub_only": 42,
    }
    synth.write_json(stem.with_name(stem.name + ".scores.json"), payload)
    synth.write_npz(
        stem.with_name(stem.name + ".arrays.npz"),
        plddt=complex_.plddt_per_atom / 100.0,
        pae=complex_.pae,
        pde=complex_.pde,
    )
    synth.write_json(stem.with_name(stem.name + ".timing.json"), {"inference": 1.5})
