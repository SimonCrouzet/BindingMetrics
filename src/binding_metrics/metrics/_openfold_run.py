"""OpenFold3 runner configuration and query preparation.

Builds the runner YAML and the query JSON (with template CIFs and A3M
self-alignments) that ``run_openfold predict`` reads. The subprocess call and the
``run_openfold_*`` wrappers stay in :mod:`binding_metrics.metrics.openfold`, which
re-exports everything defined here.
"""

import codecs
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

from binding_metrics.core.nonstandard import D_AA_MAP
from binding_metrics.core.residues import (
    FORCE_FIELD_CAP_NAMES,
    LACTAM_TEMPLATE_RESIDUES,
    TERMINAL_CAP_NAMES,
    VARIANT_TO_PARENT_RESIDUE,
)
from binding_metrics.metrics._openfold_templates import (
    StreamNotes,
    account_for_templates,
    warn_about_missing_templates,
    write_template_accounting,
)

logger = logging.getLogger(__name__)

#: Seed values written to ``experiment_settings.seeds`` of the runner YAML that the toolkit
#: generates, when the caller gives neither ``seeds`` nor ``num_model_seeds``. OpenFold3 samples
#: from these, so the same query gives the same prediction (up to GPU non-determinism). The
#: value carries no meaning; pass ``seeds=`` to the ``run_openfold*`` functions to use others.
#: OpenFold3 0.5.0 does not read seeds from the query JSON (its input reference says so), which
#: is why the ``prepare_*`` functions no longer write them.
_DEFAULT_QUERY_SEEDS: tuple[int, ...] = (42,)


#: Version of what the query builders write: the query JSON, the template CIFs, the A3M
#: self-alignments and the dummy MSAs. It is raised whenever a change to them can change a
#: prediction, and it has two uses:
#:
#: * it is part of the request key of ``OpenFold3Runner`` (``options["query_builder_version"]``),
#:   so a prediction that the store kept from an older builder is not reused for a request that the
#:   newer builder would write differently;
#: * it is part of the A3M query row (see :func:`_write_a3m_self_alignment`): OpenFold3 keeps the
#:   result of its template preprocessing in a cache keyed on the chain sequence and the content of
#:   the A3M file only, so an A3M that does not change when the template CIF writer does would be
#:   answered with an entry made from the older CIF.
#:
#: History: 1, the builders before the version existed (their template CIF was rejected by
#: OpenFold3 0.5.0); 2, a template CIF that OpenFold3 0.5.0 reads; 3, the cyclic flag only for a
#: binder of standard residues, ``template_mode``, and the dummy MSA of a run without the MSA
#: server.
QUERY_BUILDER_VERSION = 3


def _query_seeds(seeds: Sequence[int]) -> list[int]:
    """Validate ``seeds`` and return them as a list of ints."""
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of integers, not a string.")
    values = [int(s) for s in seeds]
    if not values:
        raise ValueError("seeds must contain at least one integer.")
    return values


_SEEDS_CONFLICT = (
    "seeds and num_model_seeds cannot be combined. seeds are the values to sample with "
    "(written to experiment_settings.seeds of the runner YAML); num_model_seeds asks "
    "OpenFold3 to generate that many seeds, and OpenFold3 lets --num_model_seeds replace the "
    "seeds of the runner YAML, so the explicit seeds would never be used. Give seeds, or "
    "num_model_seeds, not both."
)

#: Spellings of the OpenFold3 option that generates seeds (both are accepted by its CLI).
_NUM_MODEL_SEEDS_FLAGS = ("--num_model_seeds", "--num-model-seeds")


def _resolve_run_seeds(
    seeds: Optional[Sequence[int]],
    num_model_seeds: Optional[int],
    extra_args: Optional[Sequence[str]] = None,
) -> tuple[Optional[list[int]], Optional[int]]:
    """Check the two ways of choosing seeds and return ``(seed values, generated count)``.

    OpenFold3 0.5.0 takes seeds from ``experiment_settings.seeds`` of the runner YAML, or
    generates them from ``--num_model_seeds`` (seeds from ``random.seed(42)``), and the option
    replaces the YAML seeds whenever it is given. Both being set therefore loses the explicit
    values silently, and is refused. ``extra_args`` is read for the option too, because a
    caller can pass it there.

    Returns:
        ``(None, None)`` when the caller chose neither (the toolkit's default seeds then apply
        to the YAML it writes), the validated seed list, or the generated-seed count.

    Raises:
        ValueError: Both are given, ``seeds`` is empty, or ``num_model_seeds`` is below 1.
        TypeError: ``seeds`` is a string.
    """
    values = None if seeds is None else _query_seeds(seeds)
    count = None
    if num_model_seeds is not None:
        count = int(num_model_seeds)
        if count < 1:
            raise ValueError(f"num_model_seeds must be at least 1, got {num_model_seeds!r}.")
    overridden_in_extra_args = any(
        str(argument).split("=", 1)[0] in _NUM_MODEL_SEEDS_FLAGS for argument in extra_args or ()
    )
    if values is not None and (count is not None or overridden_in_extra_args):
        raise ValueError(_SEEDS_CONFLICT)
    return values, count


def _warn_query_seeds_ignored(seeds: Sequence[int]) -> None:
    """Validate the ``seeds`` of a ``prepare_*`` function and warn when they are not the default.

    The argument is kept so that existing calls work, but a query JSON carries no seeds that
    OpenFold3 reads. A value other than the default would have changed the run before; the
    warning says where it goes now.
    """
    values = _query_seeds(seeds)
    if tuple(values) != _DEFAULT_QUERY_SEEDS:
        warnings.warn(
            "The seeds argument of the prepare_* functions has no effect: OpenFold3 does not "
            "read seeds from the query JSON. Pass seeds to run_openfold, run_openfold_scoring, "
            "run_openfold_refolding or run_openfold_batched (or experiment_settings.seeds in the "
            "runner YAML).",
            DeprecationWarning,
            stacklevel=3,
        )


#: Model presets used when the caller names none. ``predict`` is the inference base and
#: ``low_mem`` computes the pairformer embeddings sequentially, which suits large
#: complexes. OpenFold3 0.4.1 removed ``pae_enabled``: the PAE head is on by default
#: and pTM, ipTM and PAE are always written.
_DEFAULT_MODEL_PRESETS: tuple[str, ...] = ("predict", "low_mem")

#: First OpenFold3 release in which the PAE head is on by default. Before it (0.3.x),
#: ``pae_enabled`` is the only way to get PAE, pTM and ipTM.
_PAE_ON_BY_DEFAULT_SINCE = (0, 4, 0)

_VERSION_PROBE = "from importlib.metadata import version; print(version('openfold3'))"


def _version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric release fields of ``version`` (``"0.5.0.dev3"`` gives ``(0, 5, 0)``)."""
    match = re.match(r"\d+(?:\.\d+)*", version.strip())
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def installed_openfold3_version(python_cmd: Optional[Sequence[str]] = None) -> Optional[str]:
    """Return the installed ``openfold3`` version, or None when it is not installed.

    Args:
        python_cmd: Command that starts the interpreter to ask, for example
            ``["conda", "run", "-n", "openfold3", "python"]``. None asks the current
            interpreter without starting a process.

    Returns:
        The distribution version string, or None if the package is absent, the
        interpreter cannot be started or does not answer within a minute.
    """
    if python_cmd is None:
        try:
            return importlib.metadata.version("openfold3")
        except importlib.metadata.PackageNotFoundError:
            return None
    try:
        probe = subprocess.run(
            [*python_cmd, "-c", _VERSION_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("could not ask %s for the openfold3 version: %s", list(python_cmd), exc)
        return None
    if probe.returncode != 0:
        return None
    return probe.stdout.strip() or None


#: Versions already read from a conda environment, by interpreter command. Starting ``conda run``
#: costs seconds, and one batch asks once per sample. Only a version that was read is kept, so an
#: environment that is fixed later is looked at again. The interpreter that runs this code is not
#: cached: asking it is a metadata lookup.
_VERSION_BY_PYTHON: dict[tuple[str, ...], str] = {}


def _python_command(conda_env: Optional[str]) -> Optional[list[str]]:
    """The command that starts the interpreter of ``conda_env``; None for the current one."""
    if not conda_env:
        return None
    return [shutil.which("conda") or "conda", "run", "-n", conda_env, "python"]


def _installed_version_once(conda_env: Optional[str]) -> Optional[str]:
    """``installed_openfold3_version`` for the environment ``conda_env``, asked once per process."""
    python_cmd = _python_command(conda_env)
    if python_cmd is None:
        return installed_openfold3_version(None)
    key = tuple(python_cmd)
    if key not in _VERSION_BY_PYTHON:
        version = installed_openfold3_version(python_cmd)
        if version is None:
            return None
        _VERSION_BY_PYTHON[key] = version
    return _VERSION_BY_PYTHON[key]


#: First OpenFold3 release that has the per-chain ``cyclic`` field of the query schema (0.4.5,
#: ``REL/0.4.5`` of aqlaboratory/openfold-3). Older versions reject the field: the ``Chain`` model
#: of the query forbids extra keys.
_CYCLIC_FIELD_SINCE = (0, 4, 5)


def _check_binder_cyclic(binder_cyclic) -> None:
    """Raise ``ValueError`` unless ``binder_cyclic`` is ``True``, ``False`` or ``"auto"``."""
    if not (isinstance(binder_cyclic, bool) or binder_cyclic == "auto"):
        raise ValueError(f"binder_cyclic must be True, False or 'auto', got {binder_cyclic!r}.")


@dataclasses.dataclass(frozen=True)
class BinderCyclicDecision:
    """Whether the binder chain of a query gets ``"cyclic": true``, and why not when it does not.

    ``cyclic`` is the value to write. ``reason`` is set when "auto" has a head-to-tail binder and
    leaves it as a linear chain: the installed OpenFold3 is too old for the field or its version
    could not be read, the binder has modified residues (D-amino acids, N-methylated or other
    non-standard residues, with which the flag made the one complex tried worse), or the structure
    could not be searched. It is None otherwise, including when the binder has no such bond.
    """

    cyclic: bool
    reason: Optional[str] = None


def _binder_is_head_to_tail(structure_path: str | Path, binder_chain: str) -> bool:
    """True when ``binder_chain`` of the structure file has a head-to-tail amide bond.

    Reads the file with biotite and looks the closures up with ``capabilities.detect_closures``
    (C of the last residue to N of the first, from the bond table or within 2.0 A). Only the
    ``head_to_tail`` family counts: a disulfide, a lactam or a staple is not what the ``cyclic``
    field of OpenFold3 describes.
    """
    from binding_metrics.capabilities import _read_atoms, detect_closures

    atoms = _read_atoms(structure_path)
    return any(c.family == "head_to_tail" for c in detect_closures(atoms, binder_chain))


#: What was measured with the flag on a binder of modified residues, stated wherever the rule that
#: leaves such a binder linear is explained. One complex, OpenFold3 0.5.0, no MSA, one diffusion
#: sample per seed, a prediction with the receptor template and the binder from its sequence.
_MODIFIED_RESIDUE_OBSERVATION = (
    "OpenFold3 wraps the relative positions of the chain by its token count and gives every atom "
    "of a modified residue a token of its own, so the wrap does not follow the ring. On one "
    "complex (1CWA, cyclosporin A, OpenFold3 0.5.0, no MSA, three seeds) the flag lowered ipTM "
    "from 0.91-0.92 to 0.78-0.81 and raised the binder C-alpha RMSD from 0.5-0.7 A to 3.0-4.8 A; "
    "on a binder of standard residues (SFTI-1, 3P8F, one seed) it closed the ring (C-N 7.40 A "
    "without the flag, 1.38 A with it, 1.44 A in the input)"
)


def _modified_binder_residues(structure_path: str | Path, binder_chain: str) -> Optional[list[str]]:
    """The residues of ``binder_chain`` that OpenFold3 would tokenise one token per atom.

    OpenFold3 makes one token of each standard residue and one token of each heavy atom of any
    other residue, and what it builds follows the query, not the structure file: the standard
    letters (a protonation variant or a cross-link variant is sent as its parent letter) stay
    standard, while a residue sent through ``non_canonical_residues`` (a D-amino acid, an
    N-methylated residue, any other CCD component) and selenocysteine (``U``) do not. The
    classification is that of :func:`_extract_query_chain`, so it is the one the query uses.

    Returns:
        The sorted, distinct CCD codes of those residues (``SEC`` for ``U``); an empty list when
        the binder is all standard; None when the structure cannot be read or the chain has no
        residue that the query can express.
    """
    import gemmi

    try:
        structure = gemmi.read_structure(str(structure_path))
        letters, non_canonical = _extract_query_chain(
            structure, binder_chain, on_unmappable_residue="x"
        )
    except Exception as exc:  # noqa: BLE001 - the caller decides what an unreadable file means
        logger.debug(
            "residues of chain %s of %s not classified: %s", binder_chain, structure_path, exc
        )
        return None
    modified = set(non_canonical.values())
    if "U" in letters:
        modified.add("SEC")
    return sorted(modified)


def decide_binder_cyclic(
    structure_path: str | Path,
    binder_chain: str,
    binder_cyclic: bool | str = "auto",
    *,
    conda_env: Optional[str] = None,
    log: bool = True,
) -> BinderCyclicDecision:
    """Decide whether the query writes ``"cyclic": true`` for ``binder_chain``.

    OpenFold3 uses the field only to wrap the relative-position offsets of the chain, which
    describes a head-to-tail closure. It does not enforce the closure bond, the field appears
    in its documentation only through an example query, and no accuracy figure for cyclic
    peptides has been published (source: ``input_format_reference.md`` and
    ``examples/example_inference_inputs/query_multimer_cyclic.json`` at tag v0.5.0, and the
    release notes of 0.4.5).

    The offsets are built from the number of tokens of the chain (``cyclic_offset`` in
    ``openfold3/core/utils/relpos.py``) and ignore the residue indices, while OpenFold3 makes a
    token of each heavy atom of a modified residue, so for a binder with modified residues the
    wrap does not follow the ring. Measured once (one complex per case, OpenFold3 0.5.0, no MSA;
    see ``_MODIFIED_RESIDUE_OBSERVATION``): with the flag, 1CWA (D-Ala and N-methylated residues)
    lost 0.10 to 0.15 ipTM and gained 2.3 to 4.2 A of binder RMSD over three seeds, and SFTI-1
    (standard residues) closed its ring (C-N 7.40 A without the flag, 1.38 A with it, one seed).

    Args:
        structure_path: The structure the query is built from (the complex file).
        binder_chain: Chain ID of the binder, the only chain that can get the field.
        binder_cyclic: ``False`` never writes it. ``True`` always writes it (with a warning when
            the binder has modified residues). ``"auto"`` (the default) writes it when the binder
            has a head-to-tail bond, is made of standard residues only (a protonation or
            cross-link variant counts as standard: it is sent as its parent residue) and the
            installed OpenFold3 is 0.4.5 or later; a head-to-tail binder that is left linear is
            logged as a warning and the decision says why in ``reason`` (version too old or
            unreadable, or modified residues).
        conda_env: Conda environment that runs OpenFold3, asked for its version; None asks the
            current interpreter.
        log: False keeps the decision out of the log. The query builders log it when they
            write a query; a caller that only wants to record the decision (the pipeline, for its
            result) passes False, so a warning is not given twice.

    Returns:
        The decision. Nothing is logged for a binder without a head-to-tail bond.

    Raises:
        ValueError: ``binder_cyclic`` is not ``True``, ``False`` or ``"auto"``, or it is
            ``True`` and the installed OpenFold3 is known to be older than 0.4.5.
    """
    _check_binder_cyclic(binder_cyclic)
    if binder_cyclic is False:
        return BinderCyclicDecision(False)

    if binder_cyclic is True:
        installed = _installed_version_once(conda_env)
        if installed is None:
            if log:
                logger.warning(
                    "Could not read the installed OpenFold3 version. binder_cyclic=True writes "
                    "'cyclic: true' on chain %s anyway; OpenFold3 older than 0.4.5 rejects that "
                    "field.",
                    binder_chain,
                )
        elif _version_tuple(installed) < _CYCLIC_FIELD_SINCE:
            raise ValueError(
                f"binder_cyclic=True needs OpenFold3 0.4.5 or later, which added the 'cyclic' "
                f"chain field; the installed version is {installed}, and older versions reject "
                "the field. Upgrade OpenFold3, or pass binder_cyclic=False."
            )
        if log:
            modified = _modified_binder_residues(structure_path, binder_chain)
            if modified:
                logger.warning(
                    "%s: binder_cyclic=True writes 'cyclic: true' on chain %s, which has modified "
                    "residues (%s). %s.",
                    structure_path,
                    binder_chain,
                    ", ".join(modified),
                    _MODIFIED_RESIDUE_OBSERVATION,
                )
        return BinderCyclicDecision(True)

    try:
        head_to_tail = _binder_is_head_to_tail(structure_path, binder_chain)
    except Exception as exc:  # noqa: BLE001 - "auto" must not fail a run that works without it
        reason = (
            f"could not look for a head-to-tail bond in chain {binder_chain} ({exc}), so "
            "'cyclic: true' is not written. binder_cyclic=True (--openfold-cyclic on) writes "
            "it regardless."
        )
        if log:
            logger.warning("%s: %s", structure_path, reason)
        return BinderCyclicDecision(False, reason)
    if not head_to_tail:
        return BinderCyclicDecision(False)

    installed = _installed_version_once(conda_env)
    if installed is None:
        reason = (
            f"chain {binder_chain} is closed head to tail, but the installed OpenFold3 version "
            "could not be read, so 'cyclic: true' is not written and the binder is predicted as "
            "a linear chain. binder_cyclic=True (--openfold-cyclic on) writes it regardless."
        )
    elif _version_tuple(installed) < _CYCLIC_FIELD_SINCE:
        reason = (
            f"chain {binder_chain} is closed head to tail, but the installed OpenFold3 "
            f"({installed}) predates the 'cyclic' chain field (0.4.5), so the binder is "
            "predicted as a linear chain. Upgrade OpenFold3 to 0.4.5 or later."
        )
    else:
        modified = _modified_binder_residues(structure_path, binder_chain)
        if modified is None:
            reason = (
                f"chain {binder_chain} is closed head to tail, but its residues could not be "
                "classified, so 'cyclic: true' is not written and the binder is predicted as a "
                "linear chain. binder_cyclic=True (--openfold-cyclic on) writes it regardless."
            )
        elif modified:
            reason = (
                f"chain {binder_chain} is closed head to tail but has modified residues "
                f"({', '.join(modified)}), so 'cyclic: true' is not written and the binder is "
                f"predicted as a linear chain. {_MODIFIED_RESIDUE_OBSERVATION}. "
                "binder_cyclic=True (--openfold-cyclic on) writes it regardless."
            )
        else:
            if log:
                logger.info(
                    "Chain %s of %s has a head-to-tail bond and standard residues only: the query "
                    "sets 'cyclic: true' on it (OpenFold3 %s).",
                    binder_chain,
                    structure_path,
                    installed,
                )
            return BinderCyclicDecision(True)
    if log:
        logger.warning("%s: %s", structure_path, reason)
    return BinderCyclicDecision(False, reason)


def _drop_removed_presets(presets: Sequence[str], conda_env: Optional[str] = None) -> list[str]:
    """Return ``presets`` without ``pae_enabled``, warning when it was given.

    OpenFold3 0.4.1 removed the preset and 0.5.0 still only logs a deprecation warning
    for it, because the PAE head has been on by default since 0.4.0. The name stays
    in the list only when the installation that will run has an openfold3 older than 0.4.0,
    where PAE is off unless the preset asks for it. That installation is the conda
    environment ``conda_env`` when one is used, else the current interpreter; a version that
    cannot be read counts as a current one.
    """
    kept = list(presets)
    if "pae_enabled" not in kept:
        return kept
    python_cmd = None
    if conda_env is not None:
        python_cmd = [shutil.which("conda") or "conda", "run", "-n", conda_env, "python"]
    installed = installed_openfold3_version(python_cmd)
    if installed is not None and _version_tuple(installed) < _PAE_ON_BY_DEFAULT_SINCE:
        return kept
    message = (
        "The 'pae_enabled' model preset is deprecated: OpenFold3 >= 0.4 computes the PAE "
        "head by default, so the preset is left out of the runner YAML."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(message)
    return [p for p in kept if p != "pae_enabled"]


def _write_runner_yaml(
    output_dir: Path,
    presets: list[str],
    template_dir: Optional[Path] = None,
    *,
    conda_env: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Path:
    """Write a runner YAML with model presets, optional template settings and optional seeds.

    Args:
        output_dir: Directory in which to write the file.
        presets: List of model preset names, e.g. ``["predict", "low_mem"]``. A
            ``"pae_enabled"`` entry is dropped with a ``DeprecationWarning`` (the
            preset was removed in OpenFold3 0.4.1 and the PAE head is always on).
            The OpenFold3 version that decides is the one in ``conda_env`` when given.
        template_dir: If given, adds ``template_preprocessor_settings`` with
            ``structure_directory`` pointing here and
            ``fetch_missing_structures: false`` so OF3 uses local CIFs only. It also adds
            ``msa_computation_settings.cleanup_msa_dir: false``: with the MSA server and
            templates on (the defaults), OpenFold3 deletes ``structure_directory.parent``
            at the end of a run (0.3.1 to 0.5.0; unreleased ``main`` no longer deletes
            user-chosen directories), which here is the folder that holds the query JSON,
            the A3M files and the template CIFs. In 0.5.0 ``cleanup_msa_dir`` guards only
            that deletion; 0.4.0 also removed the MSA output directory with it.
        conda_env: Conda environment that will run OpenFold3, asked for its version when
            ``presets`` names ``pae_enabled``; None asks the current interpreter.
        seeds: If given, written as ``experiment_settings.seeds``, the seeds OpenFold3 samples
            with. OpenFold3 0.5.0 reads seeds from this key or from ``--num_model_seeds`` and
            ignores the query JSON; the command-line option replaces the YAML value, so it must
            not be passed alongside. None writes no seeds (OpenFold3's own default is ``[42]``).

    Returns:
        Path to the written YAML file.
    """
    presets = _drop_removed_presets(presets, conda_env)
    cfg: dict = {"model_update": {"presets": presets}}
    if seeds is not None:
        cfg["experiment_settings"] = {"seeds": _query_seeds(seeds)}
    if template_dir is not None:
        cfg["template_preprocessor_settings"] = {
            "structure_directory": str(template_dir),
            "structure_file_format": "cif",
            "fetch_missing_structures": False,
        }
        cfg["msa_computation_settings"] = {"cleanup_msa_dir": False}

    try:
        import yaml

        content = yaml.dump(cfg, default_flow_style=False)
    except ImportError:
        # Fallback: write YAML manually
        lines = ["model_update:\n", "  presets:\n"]
        lines += [f"    - {p}\n" for p in presets]
        if seeds is not None:
            lines += ["experiment_settings:\n", f"  seeds: {json.dumps(_query_seeds(seeds))}\n"]
        if template_dir is not None:
            lines += [
                "template_preprocessor_settings:\n",
                # a JSON string is valid YAML and survives ': ', ' #' and quotes in the path
                f"  structure_directory: {json.dumps(str(template_dir))}\n",
                "  structure_file_format: cif\n",
                "  fetch_missing_structures: false\n",
                "msa_computation_settings:\n",
                "  cleanup_msa_dir: false\n",
            ]
        content = "".join(lines)

    yaml_path = output_dir / "runner_config.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _mentions_top_level_key(text: str, keys: Sequence[str]) -> bool:
    """True when a line of ``text`` starts one of the top-level ``keys`` of a YAML mapping.

    The text fallback cannot parse the file, so it only appends a section that the file does
    not mention yet.
    """
    pattern = r"^[\"']?(?:" + "|".join(re.escape(k) for k in keys) + r")[\"']?\s*:"
    return re.search(pattern, text, re.MULTILINE) is not None


def _section(cfg: dict, key: str, runner_yaml: Path) -> dict:
    """Return ``cfg[key]`` as a dict, creating it when absent; raise if it is not a mapping."""
    section = cfg.get(key)
    if section is None:
        section = cfg[key] = {}
    if not isinstance(section, dict):
        raise ValueError(f"{runner_yaml}: '{key}' must be a mapping, got {type(section).__name__}.")
    return section


def _merge_runner_settings_with_yaml(
    text: str,
    runner_yaml: Path,
    template_dir: Optional[Path],
    seeds: Optional[list[int]],
) -> tuple[Optional[str], list[str]]:
    """Parse the user's YAML and add the template directory and the seeds.

    Returns the new text and a description of each setting added. The text is None when nothing
    is to be added: the file already says everything the merge would add, or it names another
    ``structure_directory`` (which is logged and kept) and no seeds are to be set.
    """
    import yaml

    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{runner_yaml} is not valid YAML: {exc}") from exc
    if cfg is None:  # an empty file
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{runner_yaml} must hold a mapping at the top level, got {type(cfg).__name__}."
        )
    original = copy.deepcopy(cfg)
    added: list[str] = []

    if template_dir is not None:
        settings = _section(cfg, "template_preprocessor_settings", runner_yaml)
        users_directory = settings.get("structure_directory")
        if users_directory is None:
            settings["structure_directory"] = str(template_dir)
        elif Path(str(users_directory)).expanduser().resolve() != template_dir.resolve():
            logger.warning(
                "The runner YAML %s sets template_preprocessor_settings.structure_directory to "
                "%s, which is kept, but the templates of this run are in %s. OpenFold3 finds them "
                "only if the first directory holds them too.",
                runner_yaml,
                users_directory,
                template_dir,
            )
            cfg = copy.deepcopy(original)  # leave the template sections as the user wrote them
            template_dir = None
        if template_dir is not None:
            # With structure_directory set, OpenFold3 deletes its parent at the end of a run
            # unless cleanup_msa_dir is false (see _write_runner_yaml); that parent holds the
            # query files.
            _section(cfg, "msa_computation_settings", runner_yaml).setdefault(
                "cleanup_msa_dir", False
            )
            if cfg != original:
                added.append(f"the template directory {template_dir}")
    if seeds is not None:
        before_seeds = copy.deepcopy(cfg)
        _section(cfg, "experiment_settings", runner_yaml)["seeds"] = list(seeds)
        if cfg != before_seeds:
            added.append(f"the seeds {list(seeds)}")
    if cfg == original:
        return None, []
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True), added


def _merge_runner_settings_as_text(
    text: str,
    runner_yaml: Path,
    template_dir: Optional[Path],
    seeds: Optional[list[int]],
) -> tuple[str, list[str]]:
    """Append the template and seed settings to the user's YAML text, without a YAML parser.

    Works only when the file is a plain block mapping that mentions none of the sections that
    are appended (``template_preprocessor_settings`` and ``msa_computation_settings`` for the
    template directory, ``experiment_settings`` for the seeds); anything else needs PyYAML to be
    merged safely.
    """
    sections = []
    if template_dir is not None:
        sections += ["template_preprocessor_settings", "msa_computation_settings"]
    if seeds is not None:
        sections.append("experiment_settings")
    first_line = next(
        (line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")),
        "",
    )
    if (
        _mentions_top_level_key(text, sections)
        or first_line.lstrip().startswith("{")
        or re.search(r"^\.\.\.\s*$", text, re.MULTILINE)
    ):
        wanted = " and ".join(
            what
            for what, needed in (
                (f"the template directory {template_dir}", template_dir is not None),
                (f"the seeds {seeds}", seeds is not None),
            )
            if needed
        )
        raise ValueError(
            f"PyYAML is not installed and {runner_yaml} sets one of {', '.join(sections)}, or is "
            f"not a plain block mapping, so {wanted} cannot be added to it. Install PyYAML, or "
            "set the same keys in the file yourself (template_preprocessor_settings."
            "structure_directory, experiment_settings.seeds)."
        )
    lines = [text if text.endswith("\n") or not text else text + "\n"]
    added = []
    if template_dir is not None:
        added.append(f"the template directory {template_dir}")
        lines += [
            "# added by binding_metrics: where the templates of this run are\n",
            "template_preprocessor_settings:\n",
            # a JSON string is valid YAML and survives ': ', ' #' and quotes in the path
            f"  structure_directory: {json.dumps(str(template_dir))}\n",
            "msa_computation_settings:\n",
            "  cleanup_msa_dir: false\n",
        ]
    if seeds is not None:
        added.append(f"the seeds {list(seeds)}")
        lines += [
            "# added by binding_metrics: the seeds of this run\n",
            "experiment_settings:\n",
            f"  seeds: {json.dumps(list(seeds))}\n",
        ]
    return "".join(lines), added


def _merge_runner_settings(
    runner_yaml: Path,
    output_dir: Path,
    template_dir: Optional[Path] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Path:
    """Return the runner YAML to pass to OpenFold3 once the toolkit's settings are added.

    A runner YAML given by the user says nothing about where the toolkit put the template CIFs
    that the query's A3M files point to, and OpenFold3 looks for them only in
    ``template_preprocessor_settings.structure_directory``. With ``template_dir`` this writes a
    copy of the user's file to ``output_dir / "runner_config_merged.yaml"`` with that key set to
    ``template_dir``, and ``msa_computation_settings.cleanup_msa_dir: false`` unless the user
    set it (otherwise OpenFold3 deletes the folder that holds the query files at the end of a
    run; see :func:`_write_runner_yaml`).

    With ``seeds`` the copy also sets ``experiment_settings.seeds``, replacing the user's value:
    seeds the caller asked for explicitly win over the file. Without ``seeds`` the file's own
    seeds (or OpenFold3's default of ``[42]``) apply, and nothing about seeds is touched. Nothing
    else is added and the user's file is never modified.

    The copy is made by loading and dumping the YAML, so comments and anchors of the original
    are not kept. Without PyYAML the settings are appended to the text instead, which works only
    for a plain block mapping that sets none of the sections that are appended.

    A ``structure_directory`` that the user's file already sets is not overwritten. The
    differing path is logged as a warning that names both directories: the toolkit's templates
    are found only if the user's directory holds them. When the file already says everything
    that would be added, the user's own path is returned and no copy is written; so it is when
    there is nothing to add (neither ``template_dir`` nor ``seeds``), and the file is not read.

    Args:
        runner_yaml: The user's runner YAML.
        output_dir: Directory in which to write the copy.
        template_dir: Directory that holds the template CIFs of the query, or None.
        seeds: Seeds that replace the file's, or None to leave them.

    Returns:
        Path of the merged copy, or ``runner_yaml`` when there is nothing to add.

    Raises:
        FileNotFoundError: ``runner_yaml`` does not exist (only when something is to be added).
        ValueError: The file is not valid YAML or not a mapping (the message names it), or
            PyYAML is missing and the file cannot be extended as text.
    """
    runner_yaml = Path(runner_yaml)
    if template_dir is None and seeds is None:
        return runner_yaml
    seed_values = None if seeds is None else _query_seeds(seeds)
    text = runner_yaml.read_text(encoding="utf-8")
    try:
        merged, added = _merge_runner_settings_with_yaml(
            text, runner_yaml, template_dir, seed_values
        )
    except ImportError:  # no PyYAML
        merged, added = _merge_runner_settings_as_text(text, runner_yaml, template_dir, seed_values)
    if merged is None:
        return runner_yaml
    merged_path = output_dir / "runner_config_merged.yaml"
    merged_path.write_text(merged, encoding="utf-8")
    logger.info(
        "Wrote %s: a copy of the runner YAML %s that adds %s. %s is not modified.",
        merged_path,
        runner_yaml,
        " and ".join(added),
        runner_yaml,
    )
    return merged_path


# ---------------------------------------------------------------------------
# Running OpenFold3 and reporting why it failed
# ---------------------------------------------------------------------------

#: How much of OpenFold3's stderr is kept for the error message (characters).
_STDERR_TAIL_CHARS = 8000

#: Lines of that tail shown in the error message.
_STDERR_LINES_IN_MESSAGE = 8

#: Longest reason text of one query (characters); the error log holds the full traceback.
_REASON_CHARS = 300

_KNOWN_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("cowardly refusing", "Default checkpoint"),
        "The default checkpoint (OpenBind-0, of3-ob-2025-06-30-174k.pt) is not on disk. "
        "Run `setup_openfold --non-interactive` in the OpenFold3 environment, or pass "
        "inference_ckpt_path. openfold3 >= 0.5.0 does not download it at first use.",
    ),
    (
        ("state_dict keys do not match", "is not compatible with the currently installed"),
        "This checkpoint does not belong to the installed openfold3: Preview2 weights need "
        "openfold3 < 0.5, the OpenBind-0 weights need openfold3 >= 0.5.0.",
    ),
    (
        ("out of memory", "OutOfMemoryError"),
        "The GPU ran out of memory: lower num_diffusion_samples, keep the low_mem preset or "
        "use a GPU with more memory.",
    ),
    (
        ("unable to allocate shared memory",),
        "Docker's default /dev/shm is too small: start the container with --shm-size=8g "
        "(or --ipc=host).",
    ),
)


def _stderr_hint(text: str) -> str:
    """Return the advice for a known OpenFold3 failure message, or an empty string."""
    lowered = text.lower()
    for needles, hint in _KNOWN_FAILURE_HINTS:
        if any(needle.lower() in lowered for needle in needles):
            return hint
    return ""


def _stderr_lines(text: str) -> list[str]:
    """Non-empty lines of ``text``; a progress bar's carriage returns keep only the last state."""
    lines = []
    for raw in text.split("\n"):
        line = raw.split("\r")[-1].strip()
        if line:
            lines.append(line)
    return lines


#: Phrases of a line that gives the reason of a failure better than the exception that carries it.
#: openfold3 0.5.0 without its default checkpoint stops with a pydantic ``ValidationError`` whose
#: first line says only "1 validation error for InferenceExperimentConfig"; the reason is a later
#: line, "Value error, Default checkpoint ... cowardly refusing to perform inference".
_REASON_PHRASES = ("cowardly refusing",)


def _key_line(lines: Sequence[str]) -> str:
    """The line that names the failure.

    A line that states the reason (``_REASON_PHRASES``) wins; else the last exception line; else
    the last line.
    """
    for line in reversed(lines):
        if any(phrase in line for phrase in _REASON_PHRASES):
            return line
    for line in reversed(lines):
        if re.match(r"^[\w.]*(Error|Exception|Exit|Interrupt)\b", line):
            return line
    return lines[-1] if lines else ""


class OpenFoldRunError(subprocess.CalledProcessError):
    """``run_openfold`` exited with a non-zero status.

    A ``subprocess.CalledProcessError`` whose message starts with the reason: the line of
    OpenFold3's stderr that names the failure, advice for the failures that have a known fix
    (missing or incompatible weights, GPU memory, shared memory) and the last lines of stderr.
    ``stderr`` holds the last few kilobytes of it and ``hint`` the advice.
    """

    def __init__(self, returncode: int, cmd, stderr_tail: str = ""):
        super().__init__(returncode, cmd, stderr=stderr_tail)
        self.hint = _stderr_hint(stderr_tail)

    def __str__(self) -> str:
        lines = _stderr_lines(self.stderr or "")
        head = f"OpenFold3 exited with status {self.returncode}"
        if lines:
            head += f": {_key_line(lines)[:_REASON_CHARS]}"
        parts = [head]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if lines:
            shown = "\n".join(
                f"  {line[:_REASON_CHARS]}" for line in lines[-_STDERR_LINES_IN_MESSAGE:]
            )
            parts.append(f"Last lines of stderr:\n{shown}")
        parts.append(f"Command: {self.cmd}")
        return "\n".join(parts)


class OpenFoldQueryError(RuntimeError):
    """``run_openfold`` exited with status 0 but every query of the run failed.

    OpenFold3 logs an out-of-memory error or any other failure inside a query, skips the query
    and still exits normally. ``failures`` maps each failed query to the reason text.
    """

    def __init__(self, output_dir: Path, failures: dict[str, str]):
        self.output_dir = Path(output_dir)
        self.failures = failures
        lines = "\n".join(f"  {name}: {why}" for name, why in failures.items())
        # No path: the store renames the folder of a run when it keeps it, so a reason that
        # names it would point at nothing. ``output_dir`` is an attribute.
        super().__init__(f"OpenFold3 exited normally but failed on every query:\n{lines}")


@dataclasses.dataclass(frozen=True)
class _RunSummary:
    """The counts and failed queries that OpenFold3 writes to ``<output>/summary.txt``."""

    total: Optional[int]
    succeeded: Optional[int]
    failed: Optional[int]
    failed_queries: tuple[str, ...]


def _read_run_summary(output_dir: Path, not_before: float = 0.0) -> Optional[_RunSummary]:
    """Parse ``<output_dir>/summary.txt`` (format of OpenFold3 0.5.0, ``writer.py``).

    Returns None when there is no summary, or when it is older than ``not_before`` (a
    modification time in seconds), which means an earlier run wrote it.
    """
    path = Path(output_dir) / "summary.txt"
    try:
        if path.stat().st_mtime < not_before:
            logger.debug("%s predates this run and is ignored", path)
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    def _count(label: str, bullet: str = "") -> Optional[int]:
        # The bullet keeps a one-query list "Failed Queries: 7" from being read as a count.
        match = re.search(rf"^\s*{bullet}{label}:\s*(\d+)\s*$", text, re.MULTILINE)
        return int(match.group(1)) if match else None

    listed = re.search(r"^Failed Queries:\s*(.+?)\s*$", text, re.MULTILINE)
    names = tuple(n.strip() for n in listed.group(1).split(",") if n.strip()) if listed else ()
    return _RunSummary(
        total=_count("Total Queries Processed"),
        succeeded=_count("Successful Queries", bullet="-\\s*"),
        failed=_count("Failed Queries", bullet="-\\s*"),
        failed_queries=names,
    )


_ERROR_LOG_ENTRY = re.compile(
    r"Query ID\(s\): (?P<ids>[^\n]*)\nError Type: (?P<kind>[^\n]*)\n"
    r"Error Message: (?P<message>.*?)\n-{20,}\nTraceback:",
    re.DOTALL,
)


def _error_log_reasons(output_dir: Path) -> dict[str, tuple[str, Path]]:
    """Map each query in OpenFold3's ``logs/predict_err_rank<N>.log`` files to its last error.

    The value is the one-line reason and the log file. A later entry for the same query
    replaces an earlier one, because OpenFold3 appends to these files across runs.
    """
    reasons: dict[str, tuple[str, Path]] = {}
    for log in sorted((Path(output_dir) / "logs").glob("predict_err_rank*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for entry in _ERROR_LOG_ENTRY.finditer(text):
            message = " ".join(entry.group("message").split())
            reason = f"{entry.group('kind').strip()}: {message}"[:_REASON_CHARS]
            for name in entry.group("ids").split(","):
                if name.strip():
                    reasons[name.strip()] = (reason, log)
    return reasons


def _failed_query_reasons(
    output_dir: Path, not_before: float = 0.0, notes: Optional[StreamNotes] = None
) -> dict[str, str]:
    """Return ``{query name: reason}`` for the queries OpenFold3 reports as failed.

    Reads ``<output_dir>/summary.txt`` for the names and ``logs/predict_err_rank<N>.log``
    for the error of each. OpenFold3 exits with status 0 when a query fails (out of memory
    or any other exception inside the forward pass), so this is the only trace of it. An empty
    dict means no failure was reported, or that there is no summary.

    A query that fails while its features are built (before the model runs: an unreadable
    template, for one) leaves no log file: OpenFold3 only prints a warning, and removes the empty
    log directory. ``notes`` (what the run printed, see :class:`StreamNotes`) then gives the
    exception. No reason names ``output_dir``: the store renames the folder of a run when it
    keeps it, so a path in a recorded reason would point at nothing.
    """
    summary = _read_run_summary(output_dir, not_before)
    if summary is None:
        return {}
    logged = _error_log_reasons(output_dir)
    printed = {} if notes is None else notes.failed_queries
    reasons = {}
    for name in summary.failed_queries:
        if name in logged:
            why, log = logged[name]
            reasons[name] = (
                f"OpenFold3 failed on this query: {why} (logs/{log.name} in the output of the run)"
            )
        elif name in printed:
            reasons[name] = (
                f"OpenFold3 failed while it built the features of this query: {printed[name]}"
            )
        else:
            reasons[name] = (
                "OpenFold3 reported this query as failed (summary.txt of the run lists it) and "
                "wrote no log of the error; the reason is in what it printed to stderr"
            )
    return reasons


#: Key and file name of the checkpoint that openfold3 >= 0.5.0 loads by default (OpenBind-0,
#: ``openfold3/entry_points/parameters.py`` of v0.5.0). openfold3 0.5.0 does not download it
#: at first use: it stops with "cowardly refusing to perform inference" when the file is missing.
_OPENFOLD_DEFAULT_CHECKPOINT_NAME = "openbind-2025-06-30-174k"
_OPENFOLD_DEFAULT_CHECKPOINT_FILE = "of3-ob-2025-06-30-174k.pt"

#: Files of the Preview and Preview2 checkpoints, which openfold3 >= 0.5 cannot load.
_OPENFOLD_PREVIEW_CHECKPOINT_FILES = ("of3-p2-145k.pt", "of3-p2-155k.pt", "of3_ft3_v1.pt")


def _openfold_cache_dir() -> Path:
    """OpenFold3's cache directory: ``$OPENFOLD_CACHE``, else ``~/.openfold3``."""
    return Path(os.environ.get("OPENFOLD_CACHE") or (Path.home() / ".openfold3"))


def _openfold_checkpoint_dir() -> Path:
    """Directory in which openfold3 looks for checkpoint files.

    The file ``<cache>/ckpt_root`` holds the path when the weights were put elsewhere; without
    it the cache directory itself is used (``get_default_checkpoint_dir`` in openfold3).
    """
    cache = _openfold_cache_dir()
    try:
        pointer = (cache / "ckpt_root").read_text(encoding="utf-8").strip()
    except OSError:
        return cache
    return Path(pointer) if pointer else cache


def _user_default_runner_yaml() -> Optional[Path]:
    """Path of the user-default ``runner.yml`` that OpenFold3 >= 0.5 merges into every run.

    ``run_openfold predict`` loads ``$OPENFOLD_CACHE/runner.yml`` (default
    ``~/.openfold3/runner.yml``) first and layers the runner YAML it is given over it, so
    settings the toolkit does not write (structure format, MSA server URL, ...) come from that
    file. The seeds of the YAML that the toolkit generates replace the file's; a runner YAML
    of the user's own keeps whatever seeds it sets, and without any the file's seeds apply.
    Returns None when there is none.
    """
    candidate = _openfold_cache_dir() / "runner.yml"
    return candidate if candidate.is_file() else None


@dataclasses.dataclass(frozen=True)
class OpenFoldRunInfo:
    """What a finished ``run_openfold`` call reports besides its output files.

    ``failed_queries`` maps the queries that OpenFold3 skipped (it still exits with status 0)
    to their reasons; ``user_default_runner_yaml`` is the file it merged under the toolkit's
    runner YAML, if any. ``templates`` is what became of the templates of the queries that ran:
    ``{query: {chain ID: record}}`` as :func:`~binding_metrics.metrics._openfold_templates.
    account_for_templates` describes it (whether OpenFold3 used the template that the query asked
    for, and if not, why); it is empty when OpenFold3 wrote no ``inference_query_set.json``.
    """

    failed_queries: dict[str, str]
    user_default_runner_yaml: Optional[Path]
    templates: dict[str, dict[str, dict]] = dataclasses.field(default_factory=dict)


def _option_value(cmd: Sequence[str], name: str) -> Optional[str]:
    """The value of the option ``name`` in ``cmd`` (``--name=value`` or ``--name value``).

    OpenFold3's click options take the underscore and the hyphen spelling alike; the last
    occurrence wins, as it does for click. None when the option is absent.
    """
    value = None
    arguments = [str(argument) for argument in cmd]
    for index, argument in enumerate(arguments):
        key, separator, given = argument.partition("=")
        if key.startswith("--") and key[2:].replace("-", "_") == name:
            if separator:
                value = given
            elif index + 1 < len(arguments):
                value = arguments[index + 1]
    return value


def _pump_stream(source, destination: str, notes: StreamNotes) -> None:
    """Copy ``source`` (a binary pipe) to ``sys.<destination>`` as it arrives and note it.

    The destination is looked up for every write, so a test or a caller that replaces
    ``sys.stdout`` is still honoured. A console that cannot be written to does not stop the run.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while chunk := source.read1(4096):
        text = decoder.decode(chunk)
        notes.feed(text)
        try:
            stream = getattr(sys, destination)
            stream.write(text)
            stream.flush()
        except (OSError, ValueError):
            pass


def _run_openfold_command(cmd: Sequence[str], output_dir: Path) -> OpenFoldRunInfo:
    """Run an OpenFold3 command line and explain how it failed.

    stdout and stderr are both echoed as they arrive. The last few kilobytes of stderr are kept,
    so a non-zero exit raises :class:`OpenFoldRunError` with the reason instead of only the exit
    status. Both streams are also read for the lines that say why a template was lost or a
    query failed (:class:`StreamNotes`): OpenFold3 prints a failed template preprocessing on
    stdout, and the replacement of an alignment by the MSA server and a failure while the
    features of a query are built on stderr, well before the tail that is kept. After an exit
    with status 0:

    * the run's ``summary.txt`` is read: queries that failed inside OpenFold3 are logged with
      their reasons (the log file of the forward pass, or the warning of the data processing),
      and when every query failed :class:`OpenFoldQueryError` is raised;
    * ``inference_query_set.json`` is read to see which chains kept the template that the query
      asked for (``--query_json`` of the command says which asked). The accounting is logged as
      a warning for each chain that has none, kept as ``template_accounting.json`` in
      ``output_dir``, and returned.

    Args:
        cmd: Full command line (``run_openfold predict ...``, possibly behind ``conda run``).
        output_dir: The ``--output_dir`` of the command; ``summary.txt``, ``logs/`` and
            ``inference_query_set.json`` are read from there.

    Returns:
        The queries that failed in a run that otherwise succeeded, the user-default runner YAML
        that was merged in, and the template accounting.

    Raises:
        OpenFoldRunError: The process exited non-zero.
        OpenFoldQueryError: The process exited with status 0 and every query failed.
    """
    default_yaml = _user_default_runner_yaml()
    if default_yaml is not None:
        logger.warning(
            "OpenFold3 merges the user-default runner YAML %s under the toolkit's runner YAML "
            "(openfold3 >= 0.5); settings it holds that the toolkit does not write, such as "
            "structure_format or the MSA server URL, apply to this run.",
            default_yaml,
        )
    started = time.time()
    process = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_notes, stderr_notes = StreamNotes(), StreamNotes()
    stdout_pump = threading.Thread(
        target=_pump_stream, args=(process.stdout, "stdout", stdout_notes), daemon=True
    )
    stdout_pump.start()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    try:
        while chunk := process.stderr.read1(4096):
            text = decoder.decode(chunk)
            stderr_notes.feed(text)
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except (OSError, ValueError):  # a closed or unwritable console must not stop the run
                pass
            tail = (tail + text)[-_STDERR_TAIL_CHARS:]
        process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        stdout_pump.join(timeout=30)
        process.stderr.close()
        process.stdout.close()
    if process.returncode != 0:
        raise OpenFoldRunError(process.returncode, list(cmd), tail)
    stdout_notes.finish()
    stderr_notes.finish()
    notes = stdout_notes.merge(stderr_notes)

    # A file written a moment before the run started still counts as an earlier run's.
    not_before = started - 2.0
    failures = _failed_query_reasons(output_dir, not_before=not_before, notes=notes)
    if failures:
        logger.warning(
            "OpenFold3 exited normally but failed on %d quer%s: %s",
            len(failures),
            "y" if len(failures) == 1 else "ies",
            "; ".join(f"{name}: {why}" for name, why in failures.items()),
        )
        summary = _read_run_summary(output_dir, not_before=not_before)
        if summary is not None and summary.total and len(failures) >= summary.total:
            raise OpenFoldQueryError(output_dir, failures)
    use_msa_server = _option_value(cmd, "use_msa_server")
    templates = account_for_templates(
        output_dir,
        query_json=_option_value(cmd, "query_json"),
        use_msa_server=None if use_msa_server is None else use_msa_server.lower() == "true",
        notes=notes,
        not_before=not_before,
    )
    templates = {query: chains for query, chains in templates.items() if query not in failures}
    if templates:
        write_template_accounting(output_dir, templates)
        warn_about_missing_templates(templates)
    return OpenFoldRunInfo(
        failed_queries=failures, user_default_runner_yaml=default_yaml, templates=templates
    )


#: One-letter codes of the 20 standard amino acids.
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip

#: Residue names that ``core.nonstandard`` uses for its N-methyl templates but that are
#: not the Chemical Component Dictionary entries of those residues (the CCD code NMG is
#: an unrelated non-polymer component). The CCD codes of sarcosine and N-methylalanine are
#: SAR and MAA. MVA and MLE are CCD codes already.
_TEMPLATE_NAME_TO_CCD: dict[str, str] = {"NMG": "SAR", "NMA": "MAA"}

#: Values of ``on_unmappable_residue``.
_ON_UNMAPPABLE_CHOICES = ("error", "x")

_STANDARD_LETTERS = frozenset(_THREE_TO_ONE.values())


class UnmappableResidueError(ValueError):
    """A chain holds residues that OpenFold3 cannot take in a query.

    ``details`` lists ``(source, chain_id, residue_labels)`` for every offending chain,
    where ``source`` names the sample in a batch and is empty otherwise.
    """

    def __init__(self, details: list[tuple[str, str, list[str]]]):
        self.details = details
        lines = [
            f"  - {source + ': ' if source else ''}chain '{chain}': {', '.join(labels)}"
            for source, chain, labels in details
        ]
        super().__init__(
            "OpenFold3 cannot take these residues:\n" + "\n".join(lines) + "\n"
            "OpenFold3 accepts the 20 standard amino acids, X, and other amino acids only as "
            "Chemical Component Dictionary codes; the names above are none of these, so the "
            "prediction would model something else than the input. Remove or replace the "
            "residues, or run without the OpenFold3 metrics (leave 'openfold' out of "
            "--metrics). on_unmappable_residue='x' (--on-unmappable-residue x) sends an X "
            "instead and logs a warning."
        )


def _peptide_linking_parent_letter(ccd_code: str) -> Optional[str]:
    """Parent one-letter code of a peptide-linking CCD component, or None if it is not one.

    The Chemical Component Dictionary bundled with biotite decides; without it (or with a
    biotite that lacks the lookup) gemmi's built-in table of amino-acid components is used,
    which is smaller. The letter is ``X`` when the component has no standard parent.
    """
    try:
        from biotite.structure import info as ccd_info

        types = ccd_info.get_from_ccd("chem_comp", ccd_code, "type")
        if types is None:
            return None
        values = types.as_array()
        if len(values) == 0 or "PEPTIDE LINKING" not in str(values[0]).upper():
            return None
        letter = ccd_info.one_letter_code(ccd_code)
    except (ImportError, AttributeError, KeyError, ValueError, OSError) as exc:
        logger.debug("CCD lookup of %s unavailable (%s); using gemmi's table", ccd_code, exc)
        import gemmi

        table_entry = gemmi.find_tabulated_residue(ccd_code)
        if table_entry is None or not table_entry.is_amino_acid():
            return None
        letter = table_entry.one_letter_code
    letter = (letter or "X").upper()
    return letter if letter in _STANDARD_LETTERS else "X"


def _residue_letter_and_ccd(name: str) -> Optional[tuple[str, Optional[str]]]:
    """Map a residue name to ``(one-letter code, CCD code or None)`` for an OpenFold3 chain.

    The CCD code is set when the residue must be listed in ``non_canonical_residues``
    (D-amino acids, N-methylated and other modified residues). Protonation and
    cross-link variants (HID, HIE, HIP, CYX, ...) and the lactam-bridge templates take the
    letter of their parent residue and no CCD entry: the state or the link is not sent, and
    HIP names doubly protonated histidine here although the CCD uses it for
    phosphohistidine. Returns None when the name is no amino acid OpenFold3 can express.
    """
    if name in _THREE_TO_ONE:
        return _THREE_TO_ONE[name], None
    if name in VARIANT_TO_PARENT_RESIDUE:
        return _THREE_TO_ONE[VARIANT_TO_PARENT_RESIDUE[name]], None
    if name in LACTAM_TEMPLATE_RESIDUES:
        return _THREE_TO_ONE[name[:3]], None
    if name in D_AA_MAP:
        return _THREE_TO_ONE[D_AA_MAP[name]], name
    if name == "UNK":
        return "X", None
    if name == "SEC":  # selenocysteine is a sequence letter of OpenFold3
        return "U", None
    ccd_code = _TEMPLATE_NAME_TO_CCD.get(name, name)
    letter = _peptide_linking_parent_letter(ccd_code)
    return None if letter is None else (letter, ccd_code)


def _extract_query_chain(
    structure, chain_id: str, *, on_unmappable_residue: str = "error"
) -> tuple[str, dict[int, str]]:
    """Read one chain of a ``gemmi.Structure`` as an OpenFold3 query chain.

    Returns the one-letter sequence and ``non_canonical_residues``, a map from the 1-based
    position in that sequence to the CCD code of the residue. The 20 standard residues are
    plain letters; protonation and cross-link variants take their parent letter; D-amino
    acids, N-methylated and other peptide-linking CCD components keep their chemistry
    through the CCD entry (the letter is the parent, which the MSA search uses). Waters,
    ions, ligands and terminal caps are left out, as OpenFold3 takes them as separate
    chains.

    A residue that has a backbone (N, CA, C) but is none of the above cannot be expressed.

    Args:
        structure: A ``gemmi.Structure``.
        chain_id: Chain ID to read (first model).
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` naming the residues; ``"x"`` sends an ``X`` at
            their positions and logs a warning.

    Raises:
        ValueError: If the chain is not found, has no amino acids, or
            ``on_unmappable_residue`` is not one of ``"error"``, ``"x"``.
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    if on_unmappable_residue not in _ON_UNMAPPABLE_CHOICES:
        raise ValueError(
            f"on_unmappable_residue must be one of {_ON_UNMAPPABLE_CHOICES}, "
            f"got {on_unmappable_residue!r}."
        )
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            letters: list[str] = []
            non_canonical: dict[int, str] = {}
            unmappable: list[str] = []
            left_out: list[str] = []
            for res in chain:
                mapped = _residue_letter_and_ccd(res.name)
                if mapped is not None:
                    letters.append(mapped[0])
                    if mapped[1] is not None:
                        non_canonical[len(letters)] = mapped[1]
                    continue
                atom_names = {atom.name for atom in res}
                if {"N", "CA", "C"} <= atom_names:
                    unmappable.append(f"{res.name} {res.seqid.num}{res.seqid.icode.strip()}")
                    if on_unmappable_residue == "x":
                        letters.append("X")
                else:
                    left_out.append(res.name)
            if unmappable and on_unmappable_residue == "error":
                raise UnmappableResidueError([("", chain_id, unmappable)])
            if unmappable:
                logger.warning(
                    "Chain '%s': sent X for %s, which OpenFold3 cannot take; the prediction "
                    "does not model these residues.",
                    chain_id,
                    ", ".join(unmappable),
                )
            caps = sorted({n for n in left_out if n in TERMINAL_CAP_NAMES | FORCE_FIELD_CAP_NAMES})
            if caps:
                logger.info(
                    "Chain '%s': terminal cap(s) %s left out of the OpenFold3 query.",
                    chain_id,
                    ", ".join(caps),
                )
            if not letters:
                raise ValueError(f"Chain '{chain_id}' found but contains no amino acid residues")
            return "".join(letters), non_canonical
    raise ValueError(f"Chain '{chain_id}' not found in structure")


def _extract_sequence_from_structure(
    structure, chain_id: str, *, on_unmappable_residue: str = "error"
) -> str:
    """Extract the one-letter sequence of a chain for an OpenFold3 query using gemmi.

    Skips waters, ligands and caps. D-amino acids and modified residues give the letter of
    their parent residue (upper case); the residue itself travels in
    ``non_canonical_residues``, which :func:`_extract_query_chain` returns together with
    this sequence. Protonation variants (CYX, HID, HIE, HIP, ...) give their parent letter.

    Args:
        structure: A ``gemmi.Structure`` object.
        chain_id: Chain ID to extract.
        on_unmappable_residue: ``"error"`` (default) or ``"x"``; see
            :func:`_extract_query_chain`.

    Returns:
        One-letter sequence string.

    Raises:
        ValueError: If the chain is not found or contains no amino acids.
        UnmappableResidueError: If a residue cannot be expressed to OpenFold3 (and
            ``on_unmappable_residue`` is ``"error"``).
    """
    return _extract_query_chain(structure, chain_id, on_unmappable_residue=on_unmappable_residue)[0]


def _require_chain(structure, chain_id: str, source: str = "") -> None:
    """Raise ``ValueError`` unless the first model of ``structure`` has a chain ``chain_id``.

    ``source`` names where the structure came from (a file path) so that the message says
    which template file lacks the chain; without this check a missing chain gives a template
    CIF with no atoms and no error.
    """
    chain_ids = [chain.name for chain in structure[0]]
    if chain_id not in chain_ids:
        where = f" {source}" if source else ""
        raise ValueError(
            f"Chain '{chain_id}' not found in template structure{where} "
            f"(chains in its first model: {', '.join(chain_ids) or 'none'}). The template "
            "must use the chain IDs of the complex; rename the chain in the template file "
            "or pass a template that has it."
        )


#: Types of the amino-acid components that are not plain L-amino acids in the Chemical Component
#: Dictionary: glycine and sarcosine have no chirality; the D-amino acids that the toolkit knows
#: have their own type. OpenFold3 maps all three to the protein molecule type.
_ACHIRAL_PEPTIDE_COMPONENTS = frozenset({"GLY", "SAR"})


def _template_component_name(residue_name: str) -> str:
    """The Chemical Component Dictionary name that a template residue is written under.

    OpenFold3 looks every template residue up in the CCD. Protonation and cross-link variants
    (HID, CYX, ...) and the lactam-bridge templates are written as their parent residue, the
    way the query sends them, and the N-methyl names of ``core.nonstandard`` as the CCD codes
    of the same residues (see ``_TEMPLATE_NAME_TO_CCD``). Anything else keeps its name.
    """
    if residue_name in VARIANT_TO_PARENT_RESIDUE:
        return VARIANT_TO_PARENT_RESIDUE[residue_name]
    if residue_name in LACTAM_TEMPLATE_RESIDUES:
        return residue_name[:3]
    return _TEMPLATE_NAME_TO_CCD.get(residue_name, residue_name)


def _template_chem_comp_type(component_name: str) -> str:
    """The ``_chem_comp.type`` of a peptide component (one that OpenFold3 maps to protein)."""
    if component_name in D_AA_MAP:
        return "D-PEPTIDE LINKING"
    if component_name in _ACHIRAL_PEPTIDE_COMPONENTS:
        return "PEPTIDE LINKING"
    return "L-PEPTIDE LINKING"


def _template_residues(chain) -> list[tuple]:
    """The residues of ``chain`` that make up its template, as ``(residue, component name)``.

    These are the residues that :func:`_extract_query_chain` turns into letters of the query
    sequence: amino acids and the residues that are sent as ``X`` (an ``UNK`` component).
    Waters, ligands and terminal caps are left out, as in the query, so that residue ``i`` of
    the template is letter ``i`` of the sequence that the alignment indexes.
    """
    residues = []
    for residue in chain:
        if _residue_letter_and_ccd(residue.name) is not None:
            residues.append((residue, _template_component_name(residue.name)))
        elif {"N", "CA", "C"} <= {atom.name for atom in residue}:
            residues.append((residue, "UNK"))
    return residues


def _extract_chain_to_cif(
    structure, chain_id: str, output_path: Path, sequence: str = "", *, source: str = ""
) -> None:
    """Write one chain of a gemmi Structure as the template CIF that OpenFold3 reads.

    OpenFold3 0.5.0 reads the file twice, with the parsers it uses for PDB entries: the
    template preprocessor takes the canonical sequence of each chain and the release date, and
    the data loader reads the atoms and indexes the residues by ``label_seq_id``. A file as
    gemmi writes a bare chain is rejected (``label_entity_id`` is ``.``, there is no
    ``_entity_poly_seq``, ``label_seq_id`` is ``.`` and ``_chem_comp.type`` is ``.``), and an
    ``_entity_poly`` without ``pdbx_seq_one_letter_code_can`` makes the preprocessing fail with
    a message that OpenFold3 only prints, after which the chain has no template. The file written
    here has:

    * one polymer entity with the integer id 1, in ``_entity``, ``_entity_poly`` (with
      ``pdbx_seq_one_letter_code_can``), ``_entity_poly_seq`` and ``_struct_asym``, and
      ``_pdbx_poly_seq_scheme`` for the chain;
    * the chain under its own ID as ``label_asym_id``, and ``label_seq_id`` counting the residues
      1..N, which is how the A3M self-alignment of :func:`_write_a3m_self_alignment` numbers
      them (OpenFold3 would take the author numbers when ``label_seq_id`` is ``.``);
    * ``_chem_comp.type`` set for every component (``L-PEPTIDE LINKING``, ``D-PEPTIDE LINKING``
      for the D-amino acids, ``PEPTIDE LINKING`` for glycine and sarcosine), and a release date
      far in the past, so that no release-date filter discards the template.

    Only the residues of the query sequence are written (:func:`_template_residues`), under the
    component names that the CCD knows (:func:`_template_component_name`).

    Args:
        structure: Source ``gemmi.Structure``.
        chain_id: Chain ID to extract (taken from the first model).
        output_path: Destination CIF file path.
        sequence: One-letter amino acid sequence for this chain, written as the canonical
            sequence of the entity and the one the alignment carries. If empty, it is read from
            the residues.
        source: File the structure was read from, named in the error message.

    Raises:
        ValueError: If the chain is not in the structure, has no residue that the query would
            send, or has another number of them than ``sequence`` has letters; nothing is
            written then.
    """
    import gemmi

    _require_chain(structure, chain_id, source)

    source_chain = next(c for c in structure[0] if c.name == chain_id)
    template = _template_residues(source_chain)
    if not template:
        raise ValueError(
            f"Chain '{chain_id}' of the template structure"
            f"{' ' + source if source else ''} has no amino-acid residue to write as a template."
        )
    if sequence and len(sequence) != len(template):
        raise ValueError(
            f"Chain '{chain_id}' of the template structure{' ' + source if source else ''} has "
            f"{len(template)} amino-acid residues but the query sequence has {len(sequence)}; the "
            "template must hold the residues of the query, one for one. A template file that "
            "adds or removes residues (caps excepted) cannot be used."
        )
    if not sequence:
        sequence = "".join(
            (_residue_letter_and_ccd(residue.name) or ("X",))[0] for residue, _ in template
        )

    chain = gemmi.Chain(chain_id)
    components = [component for _, component in template]
    for number, (residue, component) in enumerate(template, start=1):
        written = chain.add_residue(residue)  # a copy; the source structure is not changed
        written.name = component
        written.subchain = chain_id
        written.entity_id = "1"
        written.entity_type = gemmi.EntityType.Polymer
        written.label_seq = number

    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.subchains = [chain_id]
    entity.full_sequence = components

    new_st = gemmi.Structure()
    new_st.cell = structure.cell
    new_st.spacegroup_hm = structure.spacegroup_hm
    new_model = gemmi.Model("1")
    new_model.add_chain(chain)
    new_st.add_model(new_model)
    new_st.entities.append(entity)

    doc = new_st.make_mmcif_document()
    block = doc.sole_block()

    # gemmi leaves the component types open and writes no canonical sequence.
    chem_comp_ids = [str(value).strip("'\"") for value in block.find_values("_chem_comp.id")]
    chem_comp_types = block.find_values("_chem_comp.type")
    for index, component in enumerate(chem_comp_ids):
        chem_comp_types[index] = gemmi.cif.quote(_template_chem_comp_type(component))
    entity_poly = block.find_loop("_entity_poly.entity_id").get_loop()
    entity_poly.add_columns(["_entity_poly.pdbx_seq_one_letter_code_can"], sequence)

    # A far-past release date: the preprocessor filters templates by it.
    revisions = block.init_loop("_pdbx_audit_revision_history.", ["ordinal", "revision_date"])
    revisions.add_row(["1", "1900-01-01"])
    # asym_id to entity_id: how OpenFold3 finds the canonical sequence of a chain.
    scheme = block.init_loop("_pdbx_poly_seq_scheme.", ["asym_id", "entity_id"])
    scheme.add_row([chain_id, "1"])

    doc.write_file(str(output_path))


def _check_template_chain_id(chain_id: str, source: str = "") -> None:
    """Raise ``ValueError`` if ``chain_id`` cannot be the chain of a template header.

    OpenFold3 reads a template header as ``<entry>_<chain>`` and splits it on the underscore
    into exactly two parts (``A3mParser`` in v0.5.0), so a chain that carries a template must
    have no underscore in its ID. The toolkit rejects such an ID rather than renaming the
    chain, which would change the chain IDs of the result. Chains without a template
    (the binder of a refolding query) are not affected.
    """
    if "_" in chain_id:
        where = f" in {source}" if source else ""
        raise ValueError(
            f"Chain ID '{chain_id}'{where} contains an underscore. OpenFold3 reads a template "
            "header as <entry>_<chain> and splits it on one underscore, so a chain that "
            "carries a template cannot have one in its ID. Rename the chain (for example to "
            "a letter) before scoring or refolding."
        )


def _write_a3m_self_alignment(
    sequence: str,
    query_id: str,
    entry_id: str,
    chain_id: str,
    output_path: Path,
) -> None:
    """Write a minimal A3M alignment for a self-template (100% identity).

    The A3M contains two sequences: the query and the template (identical).
    Header format follows OF3's A3M parser::

        >{entry_id}_{chain_id}/{start}-{end}

    OF3 splits on ``_`` to get (entry_id, chain_id), then looks for
    ``{entry_id}.cif`` in the ``template_preprocessor_settings.structure_directory``. It splits
    the header of the query row the same way, so ``query_id`` has the form ``<name>_<chain>``.

    The header of the query row carries ``-b<QUERY_BUILDER_VERSION>`` after the name, which has
    no effect on the alignment and changes the content hash that OpenFold3 keys its template
    cache on whenever the builders change (see :data:`QUERY_BUILDER_VERSION`).

    Args:
        sequence: One-letter amino acid sequence.
        query_id: Identifier for the query (first) sequence, ``<name>_<chain>``.
        entry_id: Template entry identifier — must contain no underscores.
            OF3 looks for ``{entry_id}.cif`` in the template directory.
        chain_id: Chain identifier within the template CIF (e.g. ``"A"``).
        output_path: Destination A3M file path.
    """
    _check_template_chain_id(chain_id)
    n = len(sequence)
    name, separator, chain = query_id.rpartition("_")
    query_header = f"{name}-b{QUERY_BUILDER_VERSION}{separator}{chain}"
    template_header = f"{entry_id}_{chain_id}/{1}-{n}"
    output_path.write_text(
        f">{query_header}/1-{n}\n{sequence}\n>{template_header}\n{sequence}\n", encoding="utf-8"
    )


#: How a template reaches OpenFold3 (``template_mode`` of the ``prepare_*`` and ``run_openfold_*``
#: functions). ``"structure"``: the CIF itself in ``template_cif_paths`` (OpenFold3's CIF Direct
#: Template Mode, which needs no alignment and which the ColabFold MSA-server step does not
#: overwrite; the default). ``"alignment"``: an A3M self-alignment per chain that points to the
#: template CIF through ``template_alignment_file_path`` (the first way, and the default until
#: the toolkit changed it; the server overwrites that path, so a run with the server on has no
#: template).
TEMPLATE_MODES = ("structure", "alignment")

#: The ``template_mode`` that the functions and commands use when none is given.
DEFAULT_TEMPLATE_MODE = "structure"

#: First OpenFold3 release that has the CIF Direct Template Mode (``template_cif_paths``, 0.4.2,
#: ``REL/0.4.2`` of aqlaboratory/openfold-3: "Cifs direct template structure").
_TEMPLATE_CIF_SINCE = (0, 4, 2)


def _check_template_mode(template_mode) -> None:
    """Raise ``ValueError`` unless ``template_mode`` is ``"structure"`` or ``"alignment"``."""
    if template_mode not in TEMPLATE_MODES:
        raise ValueError(f"template_mode must be one of {TEMPLATE_MODES}, got {template_mode!r}.")


def _check_template_mode_supported(template_mode, conda_env: Optional[str] = None) -> None:
    """Raise ``ValueError`` when ``"structure"`` is asked of an OpenFold3 older than 0.4.2.

    Older releases have no ``template_cif_paths`` and reject the query. A version that cannot be
    read does not stop anything (the run says why if it fails); ``"alignment"`` needs no check.
    """
    if template_mode != "structure":
        return
    installed = _installed_version_once(conda_env)
    if installed is not None and _version_tuple(installed) < _TEMPLATE_CIF_SINCE:
        raise ValueError(
            f"template_mode='structure' needs OpenFold3 0.4.2 or later, which added the CIF "
            f"Direct Template Mode; the installed version is {installed}. Upgrade OpenFold3, or "
            "pass template_mode='alignment' (--openfold-templates alignment)."
        )


def _query_chain(
    chain_id: str,
    sequence: str,
    non_canonical_residues: dict[int, str],
    template_alignment_file_path: Optional[str] = None,
    cyclic: bool = False,
    template_cif_paths: Optional[Sequence[str]] = None,
    template_cif_chain_ids: Optional[Sequence[Optional[str]]] = None,
    main_msa_file_paths: Optional[Sequence[str]] = None,
) -> dict:
    """Build the query JSON dict of one protein chain.

    ``non_canonical_residues`` is written only when it is not empty, so queries of chains
    made of standard residues are the same as before it existed. OpenFold3 reads its keys
    as 1-based residue positions; JSON needs them as strings. ``cyclic`` writes
    ``"cyclic": true`` (OpenFold3 >= 0.4.5) and is left out when False; only a binder chain
    is passed ``cyclic=True`` (see :func:`decide_binder_cyclic`).

    A template is either an alignment (``template_alignment_file_path``) or structures
    (``template_cif_paths``, with ``template_cif_chain_ids`` naming the chain of each file);
    OpenFold3 refuses a chain that has both. The ColabFold MSA-server step overwrites an
    alignment path and leaves CIF paths alone (``colabfold_msa_server.py``). ``main_msa_file_paths``
    (the MSA of the chain, see :func:`_msa_fields`) is overwritten by the server too.
    """
    chain: dict = {"molecule_type": "protein", "chain_ids": [chain_id], "sequence": sequence}
    if cyclic:
        chain["cyclic"] = True
    if non_canonical_residues:
        chain["non_canonical_residues"] = {str(i): c for i, c in non_canonical_residues.items()}
    if template_alignment_file_path is not None and template_cif_paths:
        raise ValueError("a chain takes a template alignment or template CIF files, not both")
    if template_alignment_file_path is not None:
        chain["template_alignment_file_path"] = template_alignment_file_path
    if template_cif_paths:
        chain["template_cif_paths"] = [str(path) for path in template_cif_paths]
        if template_cif_chain_ids is not None:
            chain["template_cif_chain_ids"] = list(template_cif_chain_ids)
    if main_msa_file_paths:
        chain["main_msa_file_paths"] = [str(path) for path in main_msa_file_paths]
    return chain


#: File name of a dummy MSA. OpenFold3 0.5.0 parses an MSA file only when its name (without the
#: suffix) is a key of ``MSASettings.max_seq_counts`` (``colabfold_main`` is one, and the name of
#: its own dummy MSA), and skips any other file; a chain with no parsed MSA then fails with
#: ``IndexError`` while its features are built.
_DUMMY_MSA_FILE = "colabfold_main.a3m"


def _write_dummy_msa(sequence: str, chain_id: str, directory: Path) -> Path:
    """Write an MSA that holds only the query sequence, and return the path of the A3M file.

    OpenFold3 takes the name of the folder that holds an A3M file as the identifier of the
    chain's alignment, so each chain gets a folder of its own (``directory``).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _DUMMY_MSA_FILE
    path.write_text(f">query_{chain_id}\n{sequence}\n", encoding="utf-8")
    return path


def _msa_fields(dummy_msa: bool, *, sequence: str, chain_id: str, directory: Path) -> dict:
    """The arguments of :func:`_query_chain` that give a chain a dummy MSA, or none.

    OpenFold3's input reference suggests MSA-free inference through a dummy MSA that holds only
    the query sequence and discourages leaving the MSA input out. 0.5.0 builds the same dummy
    for a chain without MSA files (with a warning), so the prediction is the same either way
    (measured: three seeds of one complex, within 0.03 ipTM and 0.1 A of binder RMSD); the
    file is written so that the query says what it uses. Only a run without the ColabFold MSA
    server needs it: with the server on, OpenFold3 replaces ``main_msa_file_paths`` (with a
    warning).
    """
    if not dummy_msa:
        return {}
    return {"main_msa_file_paths": [str(_write_dummy_msa(sequence, chain_id, directory))]}


def _template_fields(
    template_mode: str,
    *,
    sequence: str,
    chain_id: str,
    entry_id: str,
    cif_path: Path,
    a3m_path: Path,
) -> dict:
    """The arguments of :func:`_query_chain` that give a chain its template.

    ``"alignment"`` writes the A3M self-alignment to ``a3m_path`` (see
    :func:`_write_a3m_self_alignment`) and gives its path; ``"structure"`` writes nothing more and
    gives the template CIF and its chain. The chain ID is checked in both: OpenFold3 reads the
    template as ``<entry>_<chain>`` and splits it on the underscore either way.
    """
    if template_mode == "structure":
        _check_template_chain_id(chain_id)
        return {"template_cif_paths": [str(cif_path)], "template_cif_chain_ids": [chain_id]}
    _write_a3m_self_alignment(sequence, f"query_{chain_id}", entry_id, chain_id, a3m_path)
    return {"template_alignment_file_path": str(a3m_path)}


def prepare_refolding_query(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
    binder_cyclic: bool | str = "auto",
    conda_env: Optional[str] = None,
    template_mode: str = DEFAULT_TEMPLATE_MODE,
    dummy_msa: bool = False,
) -> Path:
    """Prepare an OpenFold3 query JSON for binder refolding with receptor as template.

    **Mode 2 — refolding:**
    The receptor chain is given its own structure from the complex as a template, so that OF3
    is conditioned on the receptor fold. The binder chain is predicted from sequence only (no
    template), so OF3 predicts both the binder conformation and its pose against the receptor.
    This answers: "next to a receptor of known fold, can OF3 recover the bound binder
    conformation and pose from its sequence?" ``binder_ca_rmsd`` against the input, in the
    receptor frame, measures it (the refolding RMSD). The receptor template carries the
    receptor fold only, not the position of the binder (see the note under Mode 1).

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json     — OF3 input JSON
          {query_name}_receptor.a3m   — self-alignment for receptor (``template_mode`` "alignment"
                                        only; with "structure" the CIF is in the query)
          templates/
            receptor.cif              — receptor template structure

    OpenFold3 finds the template CIF through
    ``template_preprocessor_settings.structure_directory`` of the runner YAML, which
    :func:`run_openfold` writes from its ``template_dir`` argument (the
    ``run_openfold_*`` wrappers pass ``{output_dir}/query/templates``). ``run_openfold
    predict`` has no ``--template_mmcif_dir`` option.

    .. note::
        OF3's template pipeline currently supports **monomeric templates**
        (protein chains only). The extracted receptor CIF is a single-chain
        structure; multi-chain receptors should be merged into one chain
        before calling this function, or handled with separate templates per
        chain.

    Args:
        complex_structure_path: CIF or PDB file of the full complex.
        receptor_chain: Chain ID of the receptor/target to give as template.
        binder_chain: Chain ID of the binder to refold (no template).
        query_name: Name for the prediction query (used in file names and the
            OF3 ``name`` field).
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional path to a pre-prepared receptor CIF
            (e.g., after MD relaxation). Must be a monomer (one chain).
            If None, the receptor chain is extracted from
            ``complex_structure_path``.
        seeds: Ignored. OpenFold3 does not read seeds from the query JSON, so none are
            written; give ``seeds`` to :func:`run_openfold` or a ``run_openfold_*`` wrapper,
            which write them to the runner YAML. A value other than the default ``(42,)``
            raises a ``DeprecationWarning``.
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written when a chain
            holds a residue that OpenFold3 cannot take; ``"x"`` sends an ``X`` for it
            and logs a warning. D-amino acids and modified residues are not affected:
            they go to ``non_canonical_residues`` with their CCD code.
        binder_cyclic: ``"auto"`` (default) writes ``"cyclic": true`` on the binder chain when
            it has a head-to-tail bond, is made of standard residues only and the installed
            OpenFold3 is 0.4.5 or later; a head-to-tail binder that is left linear because it
            has modified residues (D-amino acids, N-methylated residues; the flag made the one
            complex tried worse) or because the version is too old or unreadable is logged as a
            warning. ``True`` writes it on the binder whatever the structure says (with a warning
            when the binder has modified residues), ``False`` never. OpenFold3 uses the field
            only to wrap the relative positions of the chain; it does not enforce the closure
            bond, documents the field only in an example query, and has published no accuracy
            benchmark for cyclic peptides. Disulfide, lactam and staple closures have no
            OpenFold3 input and are not written. See :func:`decide_binder_cyclic`.
        conda_env: Conda environment that runs OpenFold3, asked for its version when
            ``binder_cyclic`` is not ``False``; None asks the current interpreter.
        template_mode: How the receptor template reaches OpenFold3. ``"structure"`` (default)
            gives the template CIF as ``template_cif_paths`` (OpenFold3's CIF Direct Template
            Mode, OpenFold3 0.4.2 or later: protein chains only, the best-matching chain of each
            file, an alignment made by OpenFold3 itself), which the ColabFold MSA server does
            not overwrite, and writes no A3M file. ``"alignment"`` writes an A3M self-alignment
            and gives its path as ``template_alignment_file_path``; the server overwrites that
            path (see ``the MSA server and the templates`` of the metrics documentation), so a
            run with the server on has no template. ``"alignment"`` is what the default was
            before the toolkit changed it.
        dummy_msa: Give every chain a dummy MSA that holds only its sequence
            (``main_msa_file_paths``, one file ``colabfold_main.a3m`` in a folder per chain below
            ``{output_dir}/msas``). It is for a run without the ColabFold MSA server, which
            overwrites it: ``run_openfold_*`` sets it when ``use_msa_server`` is False. Default
            False, which leaves the MSA input out (OpenFold3 then builds the same dummy itself,
            with a warning; its input reference discourages leaving the MSA input out).

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If a specified chain is not found or has no amino acids, ``seeds`` is
            empty, ``binder_cyclic`` is not ``True``, ``False`` or ``"auto"``, it is ``True``
            and the installed OpenFold3 is older than 0.4.5, or ``template_mode`` is not
            ``"structure"`` or ``"alignment"``, or it is ``"structure"`` and the installed
            OpenFold3 is older than 0.4.2. Nothing is written then.
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    import gemmi

    _check_template_mode(template_mode)
    _warn_query_seeds_ignored(seeds)
    complex_structure_path = Path(complex_structure_path)

    # Read the sequences first: a residue OpenFold3 cannot take must stop the run before
    # any file is written.
    st = gemmi.read_structure(str(complex_structure_path))
    receptor_seq, receptor_nc = _extract_query_chain(
        st, receptor_chain, on_unmappable_residue=on_unmappable_residue
    )
    binder_seq, binder_nc = _extract_query_chain(
        st, binder_chain, on_unmappable_residue=on_unmappable_residue
    )
    _check_template_mode_supported(template_mode, conda_env)
    cyclic = decide_binder_cyclic(
        complex_structure_path, binder_chain, binder_cyclic, conda_env=conda_env
    ).cyclic

    # Template source: the provided CIF (e.g. after MD relaxation) or the complex. Its chain is
    # checked before anything is written, because the relaxed file may name chains differently.
    if template_cif_path is not None:
        template_src = gemmi.read_structure(str(template_cif_path))
        template_source = str(template_cif_path)
    else:
        template_src, template_source = st, str(complex_structure_path)
    _check_template_chain_id(receptor_chain, str(complex_structure_path))
    _require_chain(template_src, receptor_chain, template_source)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Template CIF — named receptor.cif so OF3 finds entry_id="receptor"
    receptor_entry_id = "receptor"
    template_dest = templates_dir / f"{receptor_entry_id}.cif"
    # Only the receptor chain is extracted from the template source
    _extract_chain_to_cif(
        template_src, receptor_chain, template_dest, sequence=receptor_seq, source=template_source
    )

    # The receptor's template: an A3M self-alignment (header receptor_{chain}/{1}-{N}) that
    # points to the CIF, or the CIF itself
    receptor_template = _template_fields(
        template_mode,
        sequence=receptor_seq,
        chain_id=receptor_chain,
        entry_id=receptor_entry_id,
        cif_path=template_dest,
        a3m_path=output_dir / f"{query_name}_receptor.a3m",
    )

    # Query JSON — receptor has template, binder is free (sequence only)
    query = {
        "queries": {
            query_name: {
                "chains": [
                    _query_chain(
                        receptor_chain,
                        receptor_seq,
                        receptor_nc,
                        **receptor_template,
                        **_msa_fields(
                            dummy_msa,
                            sequence=receptor_seq,
                            chain_id=receptor_chain,
                            directory=output_dir / "msas" / f"{query_name}_{receptor_chain}",
                        ),
                    ),
                    _query_chain(
                        binder_chain,
                        binder_seq,
                        binder_nc,
                        cyclic=cyclic,
                        **_msa_fields(
                            dummy_msa,
                            sequence=binder_seq,
                            chain_id=binder_chain,
                            directory=output_dir / "msas" / f"{query_name}_{binder_chain}",
                        ),
                    ),
                ],
            }
        },
    }
    query_json_path = output_dir / f"{query_name}_query.json"
    query_json_path.write_text(json.dumps(query, indent=2), encoding="utf-8")
    return query_json_path


def prepare_scoring_query(
    complex_structure_path: str | Path,
    receptor_chain: str,
    binder_chain: str,
    query_name: str,
    output_dir: str | Path,
    template_cif_path: Optional[str | Path] = None,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
    binder_cyclic: bool | str = "auto",
    conda_env: Optional[str] = None,
    template_mode: str = DEFAULT_TEMPLATE_MODE,
    dummy_msa: bool = False,
) -> Path:
    """Prepare an OpenFold3 query JSON to score an existing complex structure.

    **Mode 1 — structure scoring:**
    Each chain is given its own structure from the complex as a template: the binder its
    conformation (its scaffold) and the receptor its conformation. OF3 outputs confidence
    scores (pLDDT, pTM, ipTM, etc.) for the complex that it predicts from them.

    A template carries the fold of the chain it is made from and no inter-chain geometry. Each
    query chain gets its own template structures (the template files here hold one chain each),
    so no cross-chain geometry is ever supplied; and ``_embed_feats`` of the template embedder
    of OpenFold3 (v0.5.0, ``openfold3/core/model/feature_embedders/template_embedders.py``)
    applies the same-chain mask (``asym_id[i] == asym_id[j]``) to the validity indicators of the
    template pair features, so cross-chain pairs are marked invalid (the distogram and
    unit-vector tensors are not multiplied by that mask there). OpenFold3 therefore places the
    binder against the receptor itself, and its confidences (pLDDT, pTM, ipTM, PAE) describe
    that pose, not the pose of the input. How far the predicted pose is from the input pose is
    a separate result: ``binder_ca_rmsd`` (binder Cα RMSD against the input, in the receptor
    frame) and ``delta_com_angstrom`` of the EvoBind adversarial check (the displacement of the
    binder centre of mass after superposing the receptor). In this mode the binder also has its
    own fold as a template, so ``binder_ca_rmsd`` is less free of the input than in refold mode.

    Issue #68, measured on one complex (1YCR, OpenFold3 0.5.0, one seed): with the ColabFold MSA
    server on OpenFold3 replaces the template alignment of both chains with the server's, finds
    none of the structures it lists (the toolkit runs it with ``fetch_missing_structures:
    false``) and goes on without any template, so a run with ``template_mode="alignment"`` is a
    template-free prediction (binder Cα RMSD against the input 1.62 A, against 1.57 A with the
    template given as a structure and 1.12 A with the template read and the server off). The
    default ``template_mode="structure"`` is not affected, because the server does not overwrite
    ``template_cif_paths``; ``run_openfold`` records what became of each template (see
    :mod:`binding_metrics.metrics._openfold_templates`).

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json
          {query_name}_receptor.a3m   (``template_mode`` "alignment" only)
          {query_name}_binder.a3m     (``template_mode`` "alignment" only)
          templates/
            receptor.cif
            binder.cif

    Args:
        complex_structure_path: CIF or PDB file of the full complex.
        receptor_chain: Chain ID of the receptor.
        binder_chain: Chain ID of the binder.
        query_name: Name for the prediction query.
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional pre-prepared complex or receptor CIF
            (e.g., after MD relaxation). When provided, both chain templates
            are extracted from this file instead of ``complex_structure_path``.
        seeds: Ignored, as for :func:`prepare_refolding_query`: no seeds are written to the
            query JSON, and a value other than the default ``(42,)`` raises a
            ``DeprecationWarning``.
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written when a chain
            holds a residue that OpenFold3 cannot take; ``"x"`` sends an ``X`` for it
            and logs a warning. D-amino acids and modified residues are not affected:
            they go to ``non_canonical_residues`` with their CCD code.
        binder_cyclic: ``"auto"`` (default), ``True`` or ``False``: whether the binder chain
            gets ``"cyclic": true``; see :func:`prepare_refolding_query`.
        conda_env: Conda environment that runs OpenFold3, asked for its version; see
            :func:`prepare_refolding_query`.
        template_mode: ``"structure"`` (default) or ``"alignment"``: how each chain's template
            reaches OpenFold3; see :func:`prepare_refolding_query`.
        dummy_msa: Give every chain a dummy MSA (query sequence only); see
            :func:`prepare_refolding_query`.

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty, ``binder_cyclic`` is ``True`` and the installed
            OpenFold3 is older than 0.4.5, or ``template_mode`` is not ``"structure"`` or
            ``"alignment"`` (or it is ``"structure"`` and OpenFold3 is older than 0.4.2).
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    import gemmi

    _check_template_mode(template_mode)
    _warn_query_seeds_ignored(seeds)
    complex_structure_path = Path(complex_structure_path)

    # Source structure for sequences (always the original complex). Read first: a residue
    # OpenFold3 cannot take must stop the run before any file is written.
    st = gemmi.read_structure(str(complex_structure_path))
    receptor_seq, receptor_nc = _extract_query_chain(
        st, receptor_chain, on_unmappable_residue=on_unmappable_residue
    )
    binder_seq, binder_nc = _extract_query_chain(
        st, binder_chain, on_unmappable_residue=on_unmappable_residue
    )
    _check_template_mode_supported(template_mode, conda_env)
    cyclic = decide_binder_cyclic(
        complex_structure_path, binder_chain, binder_cyclic, conda_env=conda_env
    ).cyclic

    # Template source: use template_cif_path if provided, else the complex. Both chains are
    # checked before anything is written, because a relaxed file may name chains differently.
    if template_cif_path is not None:
        template_src = gemmi.read_structure(str(template_cif_path))
        template_source = str(template_cif_path)
    else:
        template_src, template_source = st, str(complex_structure_path)
    _check_template_chain_id(receptor_chain, str(complex_structure_path))
    _check_template_chain_id(binder_chain, str(complex_structure_path))
    _require_chain(template_src, receptor_chain, template_source)
    _require_chain(template_src, binder_chain, template_source)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    # Template CIF files — named {entry_id}.cif so OF3 can find them.
    # Entry IDs must contain no underscores; chain_id is the suffix after "_".
    receptor_entry_id = "receptor"
    binder_entry_id = "binder"

    receptor_cif = templates_dir / f"{receptor_entry_id}.cif"
    binder_cif = templates_dir / f"{binder_entry_id}.cif"
    _extract_chain_to_cif(
        template_src,
        receptor_chain,
        receptor_cif,
        sequence=receptor_seq,
        source=template_source,
    )
    _extract_chain_to_cif(
        template_src,
        binder_chain,
        binder_cif,
        sequence=binder_seq,
        source=template_source,
    )

    # The templates: A3M self-alignments (header {entry_id}_{chain_id}/{1}-{N}) that point to
    # the CIFs, or the CIFs themselves
    receptor_template = _template_fields(
        template_mode,
        sequence=receptor_seq,
        chain_id=receptor_chain,
        entry_id=receptor_entry_id,
        cif_path=receptor_cif,
        a3m_path=output_dir / f"{query_name}_receptor.a3m",
    )
    binder_template = _template_fields(
        template_mode,
        sequence=binder_seq,
        chain_id=binder_chain,
        entry_id=binder_entry_id,
        cif_path=binder_cif,
        a3m_path=output_dir / f"{query_name}_binder.a3m",
    )

    # Query JSON — OF3 format: {"queries": {"name": {"chains": [...]}}}; the seeds are not part
    # of it (OpenFold3 reads them from the runner YAML or --num_model_seeds).
    query = {
        "queries": {
            query_name: {
                "chains": [
                    _query_chain(
                        receptor_chain,
                        receptor_seq,
                        receptor_nc,
                        **receptor_template,
                        **_msa_fields(
                            dummy_msa,
                            sequence=receptor_seq,
                            chain_id=receptor_chain,
                            directory=output_dir / "msas" / f"{query_name}_{receptor_chain}",
                        ),
                    ),
                    _query_chain(
                        binder_chain,
                        binder_seq,
                        binder_nc,
                        cyclic=cyclic,
                        **binder_template,
                        **_msa_fields(
                            dummy_msa,
                            sequence=binder_seq,
                            chain_id=binder_chain,
                            directory=output_dir / "msas" / f"{query_name}_{binder_chain}",
                        ),
                    ),
                ],
            }
        },
    }
    query_json_path = output_dir / f"{query_name}_query.json"
    query_json_path.write_text(json.dumps(query, indent=2), encoding="utf-8")
    return query_json_path


@dataclasses.dataclass
class _BatchSample:
    """Descriptor for one sample in a batched OF3 run."""

    query_name: str
    complex_structure_path: Path
    receptor_chain: str
    binder_chain: str


def _safe_entry_id(sample_id: str, suffix: str) -> str:
    """Return an OF3-safe entry ID (no underscores).

    OF3 splits A3M template headers on ``_`` to get (entry_id, chain_id),
    so entry_id must not contain underscores.
    """
    return sample_id.replace("_", "-") + suffix


def _unique_entry_ids(sample_ids: Sequence[str], suffix: str) -> dict[str, str]:
    """Return the entry ID of every sample, unique across the batch.

    An ID is :func:`_safe_entry_id` of the sample ID, so samples whose IDs contain no
    underscore and do not collide keep exactly that. ``a_b`` and ``a-b`` (or ``Ab`` and
    ``aB``, which collide on a case-insensitive file system) would both give
    ``templates/a-b<suffix>.cif`` and the second CIF would replace the first, so one query
    would be predicted from the other's template. Every sample of such a group gets a short
    hash of its own sample ID before the suffix (``a-b-1f2e3d4c<suffix>``), which does not
    depend on the other samples or their order.

    Args:
        sample_ids: Sample IDs of the batch; a repeated ID is one sample.
        suffix: Role suffix (``"rec"`` or ``"bnd"``).

    Returns:
        Map from each distinct sample ID to its entry ID.
    """
    unique_ids = list(dict.fromkeys(sample_ids))
    plain = {sid: _safe_entry_id(sid, suffix) for sid in unique_ids}
    groups: dict[str, list[str]] = {}
    for sid, entry in plain.items():
        groups.setdefault(entry.casefold(), []).append(sid)
    entry_ids = dict(plain)
    for members in groups.values():
        if len(members) < 2:
            continue
        for length in range(8, 41, 4):  # widen the hash in the (unlikely) event of a clash
            hashed = {
                sid: _safe_entry_id(
                    sid, f"-{hashlib.sha256(sid.encode('utf-8')).hexdigest()[:length]}{suffix}"
                )
                for sid in members
            }
            if len({entry.casefold() for entry in hashed.values()}) == len(members):
                entry_ids.update(hashed)
                break
        else:
            raise ValueError(f"Cannot build distinct template entry IDs for {members}.")
    return entry_ids


def _read_batch_chains(samples: list[_BatchSample], on_unmappable_residue: str) -> list[tuple]:
    """Read receptor and binder of every sample before anything is written.

    Returns ``(sample, structure, receptor_seq, receptor_nc, binder_seq, binder_nc)`` per
    sample. With ``on_unmappable_residue="error"`` the residues that OpenFold3 cannot take
    are collected over all samples and raised together as one
    :class:`UnmappableResidueError`, so one run names every affected sample.

    Raises:
        ValueError: If two samples share a query name. The name is the key of the query
            JSON, the stem of the A3M files and the source of the template entry ID, so
            the second sample would replace the first and OpenFold3 would never predict it.
    """
    import gemmi

    names = [s.query_name for s in samples]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"Query names must be unique within a batch; repeated: {', '.join(repeated)}. "
            "A repeated name would make the later sample replace the earlier one."
        )

    rows = []
    problems: list[tuple[str, str, list[str]]] = []
    for s in samples:
        st = gemmi.read_structure(str(s.complex_structure_path))
        chains = {}
        for role, chain_id in (("receptor", s.receptor_chain), ("binder", s.binder_chain)):
            try:
                chains[role] = _extract_query_chain(
                    st, chain_id, on_unmappable_residue=on_unmappable_residue
                )
            except UnmappableResidueError as exc:
                problems.extend((s.query_name, chain, labels) for _, chain, labels in exc.details)
        if len(chains) == 2:
            rows.append((s, st, *chains["receptor"], *chains["binder"]))
    if problems:
        raise UnmappableResidueError(problems)
    return rows


def _check_batch_template_chain_ids(rows: list[tuple], binder_has_template: bool) -> None:
    """Check the template chain IDs of every sample before anything is written.

    Raises one ``ValueError`` that names each sample whose template chain ID has an
    underscore (see :func:`_check_template_chain_id`).
    """
    problems = []
    for sample, *_ in rows:
        chains = [sample.receptor_chain] + ([sample.binder_chain] if binder_has_template else [])
        for chain_id in chains:
            try:
                _check_template_chain_id(chain_id, f"sample '{sample.query_name}'")
            except ValueError as exc:
                problems.append(str(exc))
    if problems:
        raise ValueError("\n".join(dict.fromkeys(problems)))


def _batch_cyclic_flags(
    rows: list[tuple], binder_cyclic: bool | str, conda_env: Optional[str]
) -> dict[str, bool]:
    """Whether the binder of each sample gets ``"cyclic": true``, by query name.

    Decided for every sample before anything is written, so that a ``ValueError`` (``True``
    with an OpenFold3 that is too old) leaves no files behind. The OpenFold3 version is asked
    once per environment.
    """
    _check_binder_cyclic(binder_cyclic)
    return {
        sample.query_name: decide_binder_cyclic(
            sample.complex_structure_path, sample.binder_chain, binder_cyclic, conda_env=conda_env
        ).cyclic
        for sample, *_ in rows
    }


def prepare_batched_scoring_queries(
    samples: list[_BatchSample],
    output_dir: str | Path,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
    binder_cyclic: bool | str = "auto",
    conda_env: Optional[str] = None,
    template_mode: str = DEFAULT_TEMPLATE_MODE,
    dummy_msa: bool = False,
) -> Path:
    """Prepare a single OF3 query JSON that scores multiple complexes.

    All template CIFs and A3M files are written into a shared directory
    structure so that one ``run_openfold predict`` call processes every
    sample.

    Args:
        samples: Per-sample descriptors.
        output_dir: Directory to write query JSON and supporting files.
        seeds: Ignored, as for :func:`prepare_scoring_query`.
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError`, listing every affected sample, before
            anything is written; ``"x"`` sends an ``X`` and logs a warning. See
            :func:`prepare_scoring_query`.
        binder_cyclic: ``"auto"`` (default), ``True`` or ``False``: whether the binder chain of
            each sample gets ``"cyclic": true``; see :func:`prepare_refolding_query`. Decided
            sample by sample, before anything is written.
        conda_env: Conda environment that runs OpenFold3, asked for its version once.
        template_mode: ``"structure"`` (default) or ``"alignment"``: how each chain's template
            reaches OpenFold3; see :func:`prepare_refolding_query`.
        dummy_msa: Give every chain a dummy MSA (query sequence only); see
            :func:`prepare_refolding_query`.

    Returns:
        Path to the combined query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty, two samples share a query name, ``binder_cyclic``
            is ``True`` and the installed OpenFold3 is older than 0.4.5, or ``template_mode``
            is not ``"structure"`` or ``"alignment"`` (or it is ``"structure"`` and OpenFold3
            is older than 0.4.2).
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    _check_template_mode(template_mode)
    _warn_query_seeds_ignored(seeds)
    rows = _read_batch_chains(samples, on_unmappable_residue)
    _check_template_mode_supported(template_mode, conda_env)
    _check_batch_template_chain_ids(rows, binder_has_template=True)
    cyclic_flags = _batch_cyclic_flags(rows, binder_cyclic, conda_env)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    names = [s.query_name for s, *_ in rows]
    rec_entries = _unique_entry_ids(names, "rec")
    bnd_entries = _unique_entry_ids(names, "bnd")

    queries: dict = {}
    for s, st, receptor_seq, receptor_nc, binder_seq, binder_nc in rows:
        rec_entry = rec_entries[s.query_name]
        bnd_entry = bnd_entries[s.query_name]

        rec_cif = templates_dir / f"{rec_entry}.cif"
        bnd_cif = templates_dir / f"{bnd_entry}.cif"
        _extract_chain_to_cif(st, s.receptor_chain, rec_cif, sequence=receptor_seq)
        _extract_chain_to_cif(st, s.binder_chain, bnd_cif, sequence=binder_seq)

        receptor_template = _template_fields(
            template_mode,
            sequence=receptor_seq,
            chain_id=s.receptor_chain,
            entry_id=rec_entry,
            cif_path=rec_cif,
            a3m_path=output_dir / f"{s.query_name}_receptor.a3m",
        )
        binder_template = _template_fields(
            template_mode,
            sequence=binder_seq,
            chain_id=s.binder_chain,
            entry_id=bnd_entry,
            cif_path=bnd_cif,
            a3m_path=output_dir / f"{s.query_name}_binder.a3m",
        )

        queries[s.query_name] = {
            "chains": [
                _query_chain(
                    s.receptor_chain,
                    receptor_seq,
                    receptor_nc,
                    **receptor_template,
                    **_msa_fields(
                        dummy_msa,
                        sequence=receptor_seq,
                        chain_id=s.receptor_chain,
                        directory=output_dir / "msas" / f"{s.query_name}_{s.receptor_chain}",
                    ),
                ),
                _query_chain(
                    s.binder_chain,
                    binder_seq,
                    binder_nc,
                    cyclic=cyclic_flags[s.query_name],
                    **binder_template,
                    **_msa_fields(
                        dummy_msa,
                        sequence=binder_seq,
                        chain_id=s.binder_chain,
                        directory=output_dir / "msas" / f"{s.query_name}_{s.binder_chain}",
                    ),
                ),
            ],
        }

    query_json_path = output_dir / "batch_query.json"
    query_json_path.write_text(json.dumps({"queries": queries}, indent=2), encoding="utf-8")
    return query_json_path


def prepare_batched_refolding_queries(
    samples: list[_BatchSample],
    output_dir: str | Path,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
    binder_cyclic: bool | str = "auto",
    conda_env: Optional[str] = None,
    template_mode: str = DEFAULT_TEMPLATE_MODE,
    dummy_msa: bool = False,
) -> Path:
    """Prepare a single OF3 query JSON that refolds binders for multiple complexes.

    Each receptor chain is given its own structure as a template; binder chains are
    predicted from sequence only, so OF3 predicts their conformation and pose (see
    :func:`prepare_refolding_query`).

    Args:
        samples: Per-sample descriptors.
        output_dir: Directory to write query JSON and supporting files.
        seeds: Ignored, as for :func:`prepare_refolding_query`.
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError`, listing every affected sample, before
            anything is written; ``"x"`` sends an ``X`` and logs a warning. See
            :func:`prepare_refolding_query`.
        binder_cyclic: ``"auto"`` (default), ``True`` or ``False``: whether the binder chain of
            each sample gets ``"cyclic": true``; see :func:`prepare_refolding_query`. Decided
            sample by sample, before anything is written.
        conda_env: Conda environment that runs OpenFold3, asked for its version once.
        template_mode: ``"structure"`` (default) or ``"alignment"``: how each receptor's template
            reaches OpenFold3; see :func:`prepare_refolding_query`.
        dummy_msa: Give every chain a dummy MSA (query sequence only); see
            :func:`prepare_refolding_query`.

    Returns:
        Path to the combined query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty, two samples share a query name, ``binder_cyclic``
            is ``True`` and the installed OpenFold3 is older than 0.4.5, or ``template_mode``
            is not ``"structure"`` or ``"alignment"`` (or it is ``"structure"`` and OpenFold3
            is older than 0.4.2).
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    _check_template_mode(template_mode)
    _warn_query_seeds_ignored(seeds)
    rows = _read_batch_chains(samples, on_unmappable_residue)
    _check_template_mode_supported(template_mode, conda_env)
    _check_batch_template_chain_ids(rows, binder_has_template=False)
    cyclic_flags = _batch_cyclic_flags(rows, binder_cyclic, conda_env)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    rec_entries = _unique_entry_ids([s.query_name for s, *_ in rows], "rec")

    queries: dict = {}
    for s, st, receptor_seq, receptor_nc, binder_seq, binder_nc in rows:
        rec_entry = rec_entries[s.query_name]

        rec_cif = templates_dir / f"{rec_entry}.cif"
        _extract_chain_to_cif(st, s.receptor_chain, rec_cif, sequence=receptor_seq)

        receptor_template = _template_fields(
            template_mode,
            sequence=receptor_seq,
            chain_id=s.receptor_chain,
            entry_id=rec_entry,
            cif_path=rec_cif,
            a3m_path=output_dir / f"{s.query_name}_receptor.a3m",
        )

        queries[s.query_name] = {
            "chains": [
                _query_chain(
                    s.receptor_chain,
                    receptor_seq,
                    receptor_nc,
                    **receptor_template,
                    **_msa_fields(
                        dummy_msa,
                        sequence=receptor_seq,
                        chain_id=s.receptor_chain,
                        directory=output_dir / "msas" / f"{s.query_name}_{s.receptor_chain}",
                    ),
                ),
                _query_chain(
                    s.binder_chain,
                    binder_seq,
                    binder_nc,
                    cyclic=cyclic_flags[s.query_name],
                    **_msa_fields(
                        dummy_msa,
                        sequence=binder_seq,
                        chain_id=s.binder_chain,
                        directory=output_dir / "msas" / f"{s.query_name}_{s.binder_chain}",
                    ),
                ),
            ],
        }

    query_json_path = output_dir / "batch_query.json"
    query_json_path.write_text(json.dumps({"queries": queries}, indent=2), encoding="utf-8")
    return query_json_path
