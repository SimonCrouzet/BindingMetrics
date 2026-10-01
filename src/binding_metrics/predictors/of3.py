"""Adapter for the output of OpenFold3 (``run_openfold predict``).

Layout checked against the released source of OpenFold3 v0.5.0 (2026-08-21) and against the
0.3 and 0.4 outputs this toolkit parsed before; the layout did not change between 0.4.0 and
0.5.0. No 0.5.0 run was available: what follows comes from reading the writer
(``openfold3/core/runners/writer.py``) and the option validator, not from running the model.

    {prediction_dir}/{query}/seed_{S}/{query}_seed_{S}_sample_{k}_model.{cif|cif.gz|pdb}
    {prediction_dir}/{query}/seed_{S}/{query}_seed_{S}_sample_{k}_confidences_aggregated.json
    {prediction_dir}/{query}/seed_{S}/{query}_seed_{S}_sample_{k}_confidences.{json|npz}
    {prediction_dir}/{query}/seed_{S}/timing.json                    one per seed directory

* ``S`` is a seed value chosen by OpenFold3, not a position; ``seed_index`` is the 1-based
  position of the seed directory in numeric order of ``S`` (``seed_9`` before ``seed_10``).
  ``k`` counts samples from 1 and is not a ranking.
* The aggregated file holds ``avg_plddt``, ``gpde``, ``ptm``, ``iptm``, ``disorder``,
  ``has_clash``, ``sample_ranking_score`` and the dictionaries ``chain_ptm``,
  ``chain_pair_iptm`` and ``bespoke_iptm``. The chain-pair keys are strings such as
  ``"(A, B)"``, not tuples. ``bespoke_iptm`` is OpenFold3's own and goes to
  ``record.extras``.
* The full confidence file holds ``plddt`` (per atom, 0-100), ``pde`` and ``pae`` (per token,
  angstrom, ``pae[i, j]`` the error of token ``j`` aligned on token ``i``). It exists only
  when the run used ``write_full_confidence_scores`` (the default). ``.npz`` files hold plain
  numeric arrays (float16 by default) and are read without pickle.
* pLDDT is also the B-factor of the structure file. The tokens are one per standard residue
  and one per heavy atom of a ligand or modified residue; the files carry no token layout,
  so ``record.tokens`` is None and the interface statistics apply only when the matrix size
  equals the residue count.
* pTM, ipTM and PAE are always written by 0.4.1 and later (the ``pae_enabled`` preset was
  removed); a missing value means a missing file.
* ``<prediction_dir>/experiment_config.json`` records the checkpoint of the run
  (``inference_ckpt_path``, ``inference_ckpt_name``) and the user-default ``runner.yml`` that
  was merged; they go to ``record.extras`` as ``inference_ckpt_path``, ``inference_ckpt_name``
  and ``user_default_runner_yaml``. Which weights produced a prediction is otherwise not
  visible in the output files, and Preview2 and OpenBind-0 outputs share one layout.
* A query that fails inside OpenFold3 leaves no confidence files and the process still exits
  with status 0; the run's ``summary.txt`` and ``logs/predict_err_rank<N>.log`` say why, and
  the reason is added to ``record.reasons``.

The module imports numpy only (the runner module is imported when a failed query has to be
explained); the structure file is not opened while parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from binding_metrics.capabilities import Capabilities, check_openfold3_residues
from binding_metrics.predictors.base import PredictionParser
from binding_metrics.predictors.record import PredictionFiles, PredictionRecord

logger = logging.getLogger(__name__)

_NAN = float("nan")

#: Structure file extensions in order of preference.
_STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".pdb")

#: The arrays OpenFold3 writes to the full confidence file; nothing else is read from a
#: ``.npz`` (numpy 1.26 adds a stray ``allow_pickle`` array to files written by OpenFold3).
_FULL_CONFIDENCE_KEYS = ("plddt", "pde", "pae", "gpde")


def _seed_key(directory: Path) -> tuple[int, int, str]:
    """Sort key of a ``seed_*`` directory: numeric seeds by value, then any other name."""
    suffix = directory.name[len("seed_") :]
    return (0, int(suffix), "") if suffix.isdecimal() else (1, 0, suffix)


def _seed_directories(query_dir: Path) -> list[Path]:
    if not query_dir.is_dir():
        return []
    return sorted((d for d in query_dir.glob("seed_*") if d.is_dir()), key=_seed_key)


def _failed_query_reason(output_dir: Path, query_name: str) -> Optional[str]:
    """Why OpenFold3 reported ``query_name`` as failed, from ``summary.txt`` and ``logs/``.

    OpenFold3 exits with status 0 when a query fails inside it, so a query without
    confidence files may have failed rather than never run. Returns None when the run's
    summary does not list the query. The readers live with the runner code.
    """
    from binding_metrics.metrics._openfold_run import _failed_query_reasons

    return _failed_query_reasons(output_dir).get(query_name)


def _run_provenance(output_dir: Path) -> dict:
    """The checkpoint and the user-default runner YAML of the run that wrote ``output_dir``.

    OpenFold3 writes its resolved settings to ``experiment_config.json`` in the output
    directory (``InferenceExperimentConfig.model_dump_json``, v0.5.0), with the top-level keys
    ``inference_ckpt_path``, ``inference_ckpt_name`` and ``user_default_runner_yaml_path``.
    Without that file the user-default ``runner.yml`` that this machine would merge is probed
    instead (``$OPENFOLD_CACHE/runner.yml``, default ``~/.openfold3``), and nothing is known
    about the checkpoint. A file that cannot be read is ignored: this is provenance, not data.
    """
    path = Path(output_dir) / "experiment_config.json"
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.debug("%s could not be read for provenance: %s", path, exc)
        else:
            if isinstance(config, dict):
                return {
                    "inference_ckpt_path": config.get("inference_ckpt_path"),
                    "inference_ckpt_name": config.get("inference_ckpt_name"),
                    "user_default_runner_yaml": config.get("user_default_runner_yaml_path"),
                }
    from binding_metrics.metrics._openfold_run import _user_default_runner_yaml

    probed = _user_default_runner_yaml()
    return {} if probed is None else {"user_default_runner_yaml": str(probed)}


def parse_aggregated_confidences(path: Path) -> dict:
    """Parse ``*_confidences_aggregated.json``: the scalars of the whole complex.

    Returns:
        ``avg_plddt``, ``gpde``, ``ptm``, ``iptm``, ``disorder``, ``has_clash``,
        ``sample_ranking_score`` (floats, NaN when the key is absent) and ``chain_ptm``,
        ``chain_pair_iptm``, ``bespoke_iptm`` (dicts as written: chain IDs, and strings such
        as ``"(A, B)"`` for pairs; ``{}`` when absent).
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    def _f(key):
        val = raw.get(key)
        return float(val) if val is not None else _NAN

    return {
        "avg_plddt": _f("avg_plddt"),
        "gpde": _f("gpde"),
        "ptm": _f("ptm"),
        "iptm": _f("iptm"),
        "disorder": _f("disorder"),
        "has_clash": _f("has_clash"),
        "sample_ranking_score": _f("sample_ranking_score"),
        "chain_ptm": raw.get("chain_ptm", {}),
        "chain_pair_iptm": raw.get("chain_pair_iptm", {}),
        "bespoke_iptm": raw.get("bespoke_iptm", {}),
    }


def parse_full_confidences(path: Path) -> dict:
    """Parse ``*_confidences.json`` or ``*_confidences.npz``.

    Only ``plddt``, ``pde``, ``pae`` and ``gpde`` are read. An ``.npz`` is opened without
    pickle, so a file that holds object arrays raises ``ValueError``.

    Returns:
        ``plddt_per_atom`` (``(n_atoms,)``, 0-100, or None), ``pde`` and ``pae``
        (``(n_tokens, n_tokens)`` angstrom, or None) and ``gpde`` (float, NaN when absent).
    """
    path = Path(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            raw = {}
            for key in _FULL_CONFIDENCE_KEYS:
                if key not in data.files:
                    continue
                try:
                    raw[key] = data[key]
                except ValueError as exc:  # numpy refuses an object array without pickle
                    raise ValueError(
                        f"{path}: array '{key}' cannot be read without pickle. OpenFold3 writes "
                        "plain numeric arrays, so this file is corrupt or was not written by "
                        f"OpenFold3 ({exc})"
                    ) from exc
    else:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    def _arr(key):
        val = raw.get(key)
        if val is None:
            return None
        return np.array(val, dtype=float)

    def _scalar(key):
        val = raw.get(key)
        if val is None:
            return _NAN
        arr = np.asarray(val, dtype=float)
        return float(arr.ravel()[0]) if arr.size > 0 else _NAN

    return {
        "plddt_per_atom": _arr("plddt"),
        "pde": _arr("pde"),
        "pae": _arr("pae"),
        "gpde": _scalar("gpde"),
    }


def parse_timing(path: Path) -> dict:
    """Parse ``timing.json`` (OpenFold3 0.5.0 writes ``{"runtime_s": seconds}``)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class OpenFold3Parser(PredictionParser):
    """Reads the output directory of one OpenFold3 query (see the module docstring)."""

    name = "of3"
    display_name = "OpenFold3"
    family = "af3"

    # The inputs OpenFold3 0.5.0 can be given, from its source (tag v0.5.0): the closure limit
    # from openfold3/core/utils/relpos.py and the query schema, the residue limit from the rule of
    # this package's own query builder (binding_metrics.metrics._openfold_run), imported when the
    # check runs. Not declared because nothing shows it: a limit on the binder size or on a binder
    # of several chains.
    # Modes (v0.5.0 source, checked against the clone on 2026-10-01): predict, refold and score
    # are supported, lock is not. The template pair features are multiplied by a same-chain mask
    # when they are built (openfold3/core/data/primitives/featurization/template.py,
    # create_template_distogram and create_template_unit_vector; the call is in
    # openfold3/core/data/pipelines/featurization/template.py) and again in the embedder
    # (openfold3/core/model/feature_embedders/template_embedders.py, _embed_feats), so a template
    # gives the fold of each chain and never the pose between chains; a multi-chain CIF template
    # gives one chain (docs/source/template_how_to.md, "CIF Direct Mode"); the only constraint is
    # the pocket constraint, documented for small-molecule ligands (docs/source/
    # input_format_reference.md, section 4).
    capabilities = Capabilities(
        closures={"none", "head_to_tail"},
        modes={"predict", "refold", "score"},
        extra_checks=(check_openfold3_residues,),
        reasons={
            "modes": (
                "OpenFold3 0.5.0 cannot be given the relative pose of the chains. A template "
                "carries the fold of one chain: the template pair features are zeroed between "
                "chains (create_template_distogram and create_template_unit_vector take a "
                "same-chain mask in openfold3/core/data/primitives/featurization/template.py, "
                "and the embedder applies it again in openfold3/core/model/feature_embedders/"
                "template_embedders.py), and a multi-chain CIF template gives one chain "
                "(docs/source/template_how_to.md). The pocket constraint is documented for "
                "small-molecule ligands only (docs/source/input_format_reference.md, section 4), "
                "and its use for a peptide binder was not confirmed. Use 'score' (the model "
                "re-docks the chains) or 'refold', or a model that pins the pose."
            ),
            "closures": (
                "OpenFold3 0.5.0 takes one kind of ring closure: `cyclic: true` on a protein chain "
                "wraps the whole chain, which makes it head-to-tail (openfold3/core/utils/"
                "relpos.py). The query schema has a `covalent_bonds` field but nothing reads it "
                "(openfold3/projects/of3_all_atom/config/inference_query_format.py), so a "
                "disulfide, a lactam, a staple or another cross-link cannot be given and the model "
                "would fold the chain without it. Use a model that takes the link as input, or "
                "leave the OpenFold3 step out."
            ),
        },
        caveats={
            "closures:head_to_tail": (
                "OpenFold3 takes a head-to-tail closure only through `cyclic: true`, which the "
                "query builders of this package write on the binder when binder_cyclic is "
                '"auto" (the default; `--openfold-cyclic off` or binder_cyclic=False turns it '
                "off) and OpenFold3 is 0.4.5 or later. The flag only wraps the relative "
                "positions of the chain: OpenFold3 does not enforce the closure bond, documents "
                "the flag in an example query and not in its documentation, and has published "
                "no accuracy benchmark for cyclic peptides, so there is no published check of "
                "its confidence values for a cyclic binder."
            ),
            "residue_classes:ligand": (
                "The OpenFold3 query builder (_extract_query_chain) leaves groups that are not "
                "amino acids out of a chain, so a ligand or glycan bonded to the binder is not "
                "part of the prediction."
            ),
            "residue_classes:cap": (
                "The OpenFold3 query builder (_extract_query_chain) leaves terminal capping "
                "groups out of the query, so the prediction is of the uncapped peptide."
            ),
        },
        version="0.5.0",
    )

    def find_files(
        self,
        prediction_dir: Path,
        name: str,
        *,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionFiles:
        """Locate the files of one sample of query ``name``.

        ``seed_index`` is the 1-based position of a ``seed_*`` directory in the numeric
        order of the seed values. A position beyond the last directory (or a query with no
        seed directory) is taken as the seed value itself, so ``seed_index=2`` finds
        ``seed_2`` when that is the only directory. ``sample`` is the sample number in the
        file names, counted from 1.
        """
        query_dir = Path(prediction_dir) / name
        seed_dirs = _seed_directories(query_dir)
        if seed_dirs and 1 <= seed_index <= len(seed_dirs):
            seed_dir = seed_dirs[seed_index - 1]
            actual_seed = seed_dir.name[len("seed_") :]
        else:
            actual_seed = str(seed_index)
            seed_dir = query_dir / f"seed_{seed_index}"
        prefix = f"{name}_seed_{actual_seed}_sample_{sample}"

        def _first_existing(stem: str, suffixes) -> Optional[Path]:
            for suffix in suffixes:
                path = seed_dir / f"{stem}{suffix}"
                if path.exists():
                    return path
            return None

        return PredictionFiles(
            directory=Path(prediction_dir),
            structure=_first_existing(f"{prefix}_model", _STRUCTURE_SUFFIXES),
            scores=_first_existing(f"{prefix}_confidences_aggregated", (".json",)),
            arrays=_first_existing(f"{prefix}_confidences", (".json", ".npz")),
            timing=_first_existing("timing", (".json",)),
        )

    def parse(
        self,
        files: PredictionFiles,
        *,
        name: str,
        seed_index: int = 1,
        sample: int = 1,
    ) -> PredictionRecord:
        """Read the aggregated and full confidence files into a record.

        A missing file leaves its values at NaN (None for an array) and adds a reason; a
        corrupt file raises. ``avg_plddt`` falls back to the mean of the per-atom pLDDT when
        the aggregated file does not give it. ``gpde`` comes from the aggregated file only.
        """
        record = PredictionRecord(
            self.name,
            name,
            seed_index=seed_index,
            sample=sample,
            structure_path=files.structure,
            ranking_score_name="sample_ranking_score",
            files=files,
        )

        if files.scores is None and files.arrays is None:
            record.reasons.append(
                f"no confidence files found for query '{name}' "
                f"(seed index {seed_index}, sample {sample}) in {files.directory}"
            )
            failure = _failed_query_reason(Path(files.directory), name)
            if failure:
                record.reasons.append(failure)
        elif files.scores is None:
            record.reasons.append("aggregated confidences file not found")
        elif files.arrays is None:
            record.reasons.append(
                "per-atom confidences file not found; OpenFold3 writes it only when "
                "write_full_confidence_scores is true"
            )

        if files.scores is not None:
            aggregated = parse_aggregated_confidences(files.scores)
            record.avg_plddt = aggregated["avg_plddt"]
            record.gpde = aggregated["gpde"]
            record.ptm = aggregated["ptm"]
            record.iptm = aggregated["iptm"]
            record.disorder = aggregated["disorder"]
            record.has_clash = aggregated["has_clash"]
            record.ranking_score = aggregated["sample_ranking_score"]
            record.chain_ptm = aggregated["chain_ptm"]
            record.chain_pair_iptm = aggregated["chain_pair_iptm"]
            record.extras["bespoke_iptm"] = aggregated["bespoke_iptm"]

        if files.arrays is not None:
            full = parse_full_confidences(files.arrays)
            record.plddt_per_atom = full["plddt_per_atom"]
            record.pde = full["pde"]
            record.pae = full["pae"]
            if record.plddt_per_atom is not None and np.isnan(record.avg_plddt):
                record.avg_plddt = float(np.mean(record.plddt_per_atom))

        if files.timing is not None:
            record.timing = parse_timing(files.timing)

        record.extras.update(_run_provenance(Path(files.directory)))

        located = next(iter(files.found().values()), None)
        if located is not None and located.parent.name.startswith("seed_"):
            record.extras["seed_value"] = located.parent.name[len("seed_") :]
        return record
