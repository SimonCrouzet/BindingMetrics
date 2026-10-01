"""Protenix as a prediction runner for the prediction store.

``ProtenixRunner`` writes a Protenix input JSON from the complex structure and starts
``protenix pred`` in the current environment or in a conda environment. It builds on the same
scheme as ``OpenFold3Runner`` (a request that holds every setting that changes the output, a
``prepare`` that fails on a bad input before anything is written, a ``run`` that returns the
directory the parser loads and raises when the run left no output). The parser of the same model
is ``binding_metrics.predictors.protenix.ProtenixParser``.

Provenance. Everything below about Protenix was read from its source and documentation at
commit 85767b8 (2026-09-21; ``protenix/version.py`` says 2.0.0) on 2026-10-01, cited as file:line
at that commit. Nothing was run: no Protenix installation, no weights and no real output were
used, so no statement here has been checked against a real run. TO VERIFY with one run on a GPU
machine: (1) that the command line and the JSON below run end to end; (2) that the output CIF
carries the ``id`` values of the JSON as its chain IDs (the JSON reader assigns them,
``protenix/data/inference/json_to_feature.py:101-147``; the CIF writer was not read); (3) that a
modified residue given as a ``ptmType`` is built as intended for D-amino acids and N-methylated
residues (``protenix/data/inference/json_parser.py:316-364`` replaces the residue by the CCD
component, and the polymer bond is made by ``ccd._connect_inter_residue``, not read here).

Public names and signatures (the module imports the standard library and the runner ABC; gemmi
and the openfold run module are imported when an input is read, never before)::

    class ProtenixRunError(RuntimeError)       # .returncode (None for an exit with status 0),
                                               # .cmd, .output_tail
    class ProtenixRunner(PredictionRunner):
        ProtenixRunner(conda_env: Optional[str] = None)
        name = "protenix"; capabilities = None
        supports_custom_weights = True; weights_kind = "directory"
        .conda_env
        .make_request(input_path, *, name: str, binder_chain: str, receptor_chain: str,
            mode: str = "predict", seeds: Optional[Sequence[int]] = None, num_samples: int = 5,
            use_msa_server: bool = True, msa_server_mode: str = "protenix",
            model_name: str = "protenix_base_default_v1.0.0", dtype: str = "bf16",
            weights: Optional[str | Path | WeightsRef] = None,
            constraints: Optional[Mapping] = None,
            covalent_bonds: Optional[Sequence[Mapping]] = None,
            on_unmappable_residue: str = "error", extra_args: Sequence[str] = ())
            -> PredictionRequest
        .check_weights(request)                           # raises for any request with weights
        .prepare(request, work_dir) -> Path               # <work_dir>/input/<name>.json
        .run(request, work_dir) -> Path                   # <work_dir>/predictions
        .is_available() -> bool
        .version() -> Optional[str]

The input. ``input_path`` is the complex structure (PDB or mmCIF, first model). The receptor and
the binder chain become two ``proteinChain`` entities, the receptor first, so that a
``constraints`` or ``covalent_bonds`` entry refers to ``entity`` 1 (receptor) and 2 (binder)
(``docs/infer_json_format.md:233-234``; ``position`` is the 1-based position in the sequence
sent, which counts the residues of the chain that are amino acids)::

    [{"name": <name>,
      "sequences": [
        {"proteinChain": {"sequence": "<receptor>", "count": 1, "id": ["A"]}},
        {"proteinChain": {"sequence": "<binder>", "count": 1, "id": ["B"],
                          "modifications": [{"ptmType": "CCD_DAL", "ptmPosition": 1}, ...]}}],
      "covalent_bonds": [...],      # only with covalent_bonds
      "constraint": {...}}]         # only with constraints

Keys and their meaning: ``docs/infer_json_format.md:38-68`` (``proteinChain``), ``206-246``
(``covalent_bonds``), ``252-366`` (``constraint``). ``id`` keeps the chain IDs of the input
(``json_to_feature.py:101-125`` requires a list of strings, as many as ``count``, none repeated),
which makes the output chain IDs equal the input's. The sequence and the modifications come from
``_extract_query_chain`` of ``metrics/_openfold_run.py``, the reader of the OpenFold3 query: the
20 standard residues are letters, protonation and cross-link variants take the letter of their
parent, and a D-amino acid, an N-methylated residue or another peptide-linking CCD component is
its parent letter plus a modification ``CCD_<code>`` at that position
(``json_parser.py:333-341`` replaces the residue at ``ptmPosition`` by the component).
Selenocysteine, which that reader writes as the letter ``U`` that Protenix does not know
(``PROTEIN_1to3``, ``json_parser.py:53-75``), is an ``X`` with ``CCD_SEC``. A residue with a
backbone that is none of these raises ``ValueError`` naming the chain and the residues before any
file is written (``on_unmappable_residue="x"`` sends an ``X`` instead). Terminal caps, waters,
ions and ligands are left out, as in the OpenFold3 query.

No ring closure is written. Protenix documents a head-to-tail amide bond and a disulfide as
``covalent_bonds`` (``docs/infer_json_format.md:222-229``) and this runner does not derive them
from the structure: a cyclic binder is predicted as a linear chain unless ``covalent_bonds``
gives the bond. The option is a passthrough: the list is written under ``covalent_bonds`` as
given and is not interpreted.

Modes. Only ``predict`` (from sequences) is supported. A template enters Protenix as an alignment
file, ``templatesPath`` in .a3m or .hhr format (``docs/infer_json_format.md:68``;
``--use_template``, ``runner/batch_inference.py:667-672``), so a structure cannot be given as a
template, and building a self-template alignment is not verified: ``refold``, ``score`` and
``score-lock`` raise ``ValueError`` (``make_request``, ``prepare`` and ``run``). The documented
way to steer a prediction is the ``constraint`` section (``contact`` and ``pocket``,
``docs/infer_json_format.md:252-254``), "a soft constraint: the model is encouraged, but not
strictly required, to satisfy it" (line 256): it guides the interface and does not pin the pose.
``constraints`` is written under ``constraint`` as given. Only the model
``protenix_base_constraint_v0.5.0`` has constraint embedders switched on
(``configs/configs_model_type.py:123-136``; every other model has them off,
``configs/configs_base.py:282-307``, and ``protenix/model/protenix.py:218-228`` adds nothing when
the embedder returns None), so a request for ``constraints`` with another model raises
``ValueError`` instead of running with the section ignored.

The command line (``runner/batch_inference.py:599-762``; the console script ``protenix`` is
``runner.batch_inference:protenix_cli`` and the command is registered as ``pred``, line 1352)::

    protenix pred --input <work_dir>/input/<name>.json --out_dir <work_dir>/predictions
        --seeds 101,102 --sample 5 --dtype bf16 --model_name protenix_base_default_v1.0.0
        --use_msa true --msa_server_mode protenix --need_atom_confidence true [extra_args]

behind ``conda run -n <env> --no-capture-output`` when ``conda_env`` is set, with ``cwd`` set to
``work_dir``. The long names are the ones of the click options (``-i -o -s -e -d -n`` are their
short forms); ``--sample`` is the number of samples per seed (line 607, ``N_sample``) and
``--out_dir`` is the output directory (the ``--dump_dir`` and ``--sample_diffusion.N_sample`` of
the documentation belong to ``runner/inference.py``, a second entry point). ``--need_atom_confidence
true`` is always given: it is what makes Protenix write the per-atom pLDDT, PAE and PDE file the
adapter needs for the interface PAE and PDE (``runner/dumper.py:258-275``). ``--seeds`` takes
the values joined by commas, non-negative and distinct (the adapter orders seed directories by
their numeric value and reads only decimal ones). ``--dtype`` is ``bf16``, ``fp32`` or ``fp16``
(``runner/inference.py:217-221``; the documentation names the first two); a GPU of compute
capability 7.x is forced to fp32 by Protenix (``inference.py:665-694``), which the request does
not record. ``extra_args`` is appended as given (for example ``["--cycle", "4", "--step", "5"]``
for the mini models, or ``["--trimul_kernel", "torch", "--triatt_kernel", "torch"]`` without
cuequivariance, ``batch_inference.py:628-642``); an argument that sets what the runner sets itself,
or ``--use_template``, ``--use_rna_msa`` or ``--use_seeds_in_json``, raises ``ValueError``.

MSA. ``use_msa_server`` (default True) is ``--use_msa``. Protenix has no separate switch for the
server: with ``--use_msa true`` and no ``pairedMsaPath``/``unpairedMsaPath`` in the JSON, which
this runner never writes, ``protenix pred`` searches the MSA over the network, and the sequences
leave the machine (``runner/msa_search.py:194-253``). ``msa_server_mode`` is ``--msa_server_mode``:
``protenix`` asks ``https://protenix-server.com/api/msa``, and the code asserts that URL in this
mode (``protenix/web_service/colab_request_utils.py:58-59``); ``colabfold`` asks the host in
``$MMSEQS_SERVICE_HOST_URL``, which defaults to the same URL
(``protenix/web_service/colab_request_parser.py:38-39,279-300``). ``False`` runs without an MSA,
which Protenix warns "might degrade significantly" (``msa_search.py:243-249``). A failed search
does not stop Protenix: it prints "MMSEQS2 failed with the following error message" and
continues with the sequence itself as the MSA, exit status 0 (``colab_request_parser.py:263-278``,
``451-457``). The runner reads the output for those lines and raises, because that prediction is
not the one the request asks for. What the server returns, and ``MMSEQS_SERVICE_HOST_URL``, are
not in the key (``use_msa_server`` and ``msa_server_mode`` are).

Weights. ``model_name`` selects the model; ``protenix pred`` loads
``<load_checkpoint_dir>/<model_name>.pt`` and downloads the file, and the CCD data files, from
ByteDance's servers when they are missing (``runner/inference.py:398-454``). Custom weights are a
directory plus the model name: ``request.weights`` is that directory (``weights_kind`` is
``"directory"``) and the file is ``<directory>/<model_name>.pt``. The command has no option for
``load_checkpoint_dir``: it is ``$PROTENIX_ROOT_DIR/checkpoint``, ``~/checkpoint`` without the
variable, fixed when the package is imported (``configs/configs_inference.py:21,29``). Only
``runner/inference.py``, which does not search the MSA, has ``--load_checkpoint_dir``. So a request
with ``weights`` cannot be run: ``make_request``, ``check_weights``, ``prepare`` and ``run`` raise
``ValueError`` that names the route that exists (put or link the file in that directory, or set
``PROTENIX_ROOT_DIR`` in the environment that runs Protenix; the variable also moves the CCD data
directory). ``supports_custom_weights`` is True so that the declaration matches the weights of
Protenix (a directory and a name) and the refusal comes with its reason; the store passes such a
request to the runner, which records the reason as the failure. The size of the default file at
``$PROTENIX_ROOT_DIR/checkpoint/<model_name>.pt`` is recorded in the key
(``checkpoint_size_bytes``; None while it is not there).

The request. ``make_request`` writes every setting that changes the output into the request, the
defaults included, so two callers that mean one run share its key. The key holds: the Protenix
version (``version()``; empty when it cannot be told), the mode, the seeds (``[101]``, the
command-line default, when the caller gives none), the samples per seed, the chain roles, the
content hash of the structure and ``options``: ``model_name``, ``dtype``, ``use_msa_server``,
``msa_server_mode``, ``on_unmappable_residue``, ``constraints``, ``covalent_bonds``,
``extra_args``, ``need_atom_confidence`` (always true) and ``checkpoint_size_bytes``. Not in the
key: the conda environment and where the input lives.

Output and failures. The output directory ``<work_dir>/predictions`` holds
``<name>/seed_<S>/predictions/<name>_sample_<r>.cif``, ``..._summary_confidence_sample_<r>.json``
and ``..._full_data_sample_<r>.json`` for every seed and every ``r`` below ``num_samples``
(layout and evidence: ``predictors/protenix.py``), and ``ERR/<name>.txt`` when a sample failed.
``protenix pred`` exits with status 0 when a sample fails: it writes the error to
``ERR/<name>.txt`` (``runner/inference.py:577-582,629-634``), ``ERR/error.txt`` when it cannot
build the dataloader (``inference.py:559-560``), or logs "Run inference failed" for an error
before that, such as the MSA search (``batch_inference.py:538-562``). ``run`` therefore checks the
output and raises ``ProtenixRunError`` when the process exits with another status, when it
reports an MSA search that failed, or when any requested seed or sample lacks its structure, its
summary or its full-data file (a missing seed would also shift the seed positions that readers
use). The message gives the exit status, what Protenix recorded in ``ERR/``, the "Run inference
failed" line, advice for the failures that have a known fix, the last lines of output and the
command, with the work directory written as ``<work_dir>`` because the store renames it; the
store records it as the reason. ``FileNotFoundError`` is raised when ``protenix`` is
not on PATH. ``supports_batch`` stays False.

Protenix licence: Apache 2.0; no weights or model code are read or shipped here.
"""

from __future__ import annotations

import codecs
import importlib.metadata
import json
import logging
import operator
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence

from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import MODES, PredictionRequest
from binding_metrics.predictors.weights import WeightsRef

logger = logging.getLogger(__name__)

#: Defaults of ``protenix pred`` (``runner/batch_inference.py:604,607,608,613,616-621,661-666``),
#: written into every request so that the key does not depend on the installed defaults.
DEFAULT_MODEL_NAME = "protenix_base_default_v1.0.0"
_DEFAULT_SEEDS = (101,)
_DEFAULT_NUM_SAMPLES = 5
_DEFAULT_DTYPE = "bf16"
_DEFAULT_MSA_SERVER_MODE = "protenix"
_DEFAULT_ON_UNMAPPABLE = "error"

_DTYPES = ("bf16", "fp32", "fp16")
_MSA_SERVER_MODES = ("protenix", "colabfold")
_ON_UNMAPPABLE_CHOICES = ("error", "x")

#: The only model with constraint embedders switched on (``configs/configs_model_type.py:123-136``).
CONSTRAINT_MODEL_NAME = "protenix_base_constraint_v0.5.0"

#: Letters of a ``proteinChain`` sequence: the 20 amino acids and X (``json_parser.py:53-75``).
_PROTEIN_LETTERS = frozenset("ARNDCQEGHILKMFPSTWYVX")

#: Flags the runner sets itself, or whose feature it does not offer, with what to use instead.
_RESERVED_FLAGS = {
    "-i": "the input is written by the runner",
    "--input": "the input is written by the runner",
    "-o": "the output directory is set by the runner",
    "--out_dir": "the output directory is set by the runner",
    "-s": "pass seeds",
    "--seeds": "pass seeds",
    "-e": "pass num_samples",
    "--sample": "pass num_samples",
    "-d": "pass dtype",
    "--dtype": "pass dtype",
    "-n": "pass model_name",
    "--model_name": "pass model_name",
    "--use_msa": "pass use_msa_server",
    "--msa_server_mode": "pass msa_server_mode",
    "--need_atom_confidence": "the runner always asks for the per-atom confidence file",
    "--use_template": "templates are not supported by this runner (see the modes)",
    "--use_rna_msa": "RNA is not supported by this runner",
    "--use_seeds_in_json": "the seeds are passed with --seeds",
}

#: Lines of the output that mean the MSA search failed and Protenix went on without an MSA.
_MSA_FAILURE_MARKERS = (
    "MMSEQS2 failed with the following error message",
    "using the sequence itself as MSA",
)

#: The line ``protenix pred`` logs for an error it caught (``runner/batch_inference.py:562``).
_RUN_FAILED_MARKER = "Run inference failed"

#: How much of the output is kept, and how many of its lines go into an error message.
_OUTPUT_TAIL_CHARS = 8000
_LINES_IN_MESSAGE = 8
_LINE_CHARS = 300

_VERSION_PROBE = "from importlib.metadata import version; print(version('protenix'))"

_KNOWN_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("Given checkpoint path not exist", "Download model checkpoint failed"),
        "Protenix loads <PROTENIX_ROOT_DIR or ~>/checkpoint/<model_name>.pt and downloads it from "
        "its own server when it is missing. Put the file there, or set PROTENIX_ROOT_DIR in the "
        "environment that runs Protenix (runner/inference.py:398-454).",
    ),
    (
        ("out of memory", "OutOfMemoryError"),
        "The GPU ran out of memory: lower num_samples, or use a smaller model_name.",
    ),
    (
        ("No module named 'cuequivariance", "No module named 'cuequivariance_ops"),
        "The default triangle kernels need cuequivariance. Pass extra_args=['--trimul_kernel', "
        "'torch', '--triatt_kernel', 'torch'] to use the PyTorch kernels "
        "(runner/batch_inference.py:628-642).",
    ),
)


class ProtenixRunError(RuntimeError):
    """``protenix pred`` failed, or exited normally and left no usable output.

    The message starts with the reason. ``returncode`` is the exit status (None when the process
    exited with status 0 and the output was refused), ``cmd`` the command line and
    ``output_tail`` the last few kilobytes of what it wrote.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: Optional[int] = None,
        cmd: Sequence[str] = (),
        output_tail: str = "",
    ):
        super().__init__(message)
        self.returncode = returncode
        self.cmd = list(cmd)
        self.output_tail = output_tail


# ---------------------------------------------------------------------- messages


def _unsupported_mode_message(mode: str) -> str:
    """Why the runner cannot run ``mode``, and the documented routes that exist."""
    if mode not in MODES:
        return f"mode must be one of {MODES}, got {mode!r}; the Protenix runner runs 'predict'"
    needs = {
        "refold": "gives the receptor as a template and predicts the binder freely",
        "score": "gives every chain its own structure as a template",
        "score-lock": (
            "gives every chain its own structure as a template and pins the relative pose of "
            "the chains"
        ),
    }[mode]
    return (
        f"The Protenix runner supports mode 'predict' only, not '{mode}'. Mode '{mode}' {needs}. "
        "Protenix takes templates as alignment files, `templatesPath` in .a3m or .hhr format "
        "(docs/infer_json_format.md:68; --use_template, runner/batch_inference.py:667-672), so a "
        "structure cannot be given as a template, and building a template alignment from the "
        "input structure is not verified. The documented way to steer a prediction is the "
        "`constraint` section with `contact` and `pocket` constraints "
        "(docs/infer_json_format.md:252-254), which the documentation calls "
        '"a soft constraint: the model is encouraged, but not strictly required, to satisfy it" '
        "(docs/infer_json_format.md:256): they guide the interface and do not pin the complete "
        "pose of the chains. Pass them with constraints= in mode 'predict' (only the model "
        f"{CONSTRAINT_MODEL_NAME} reads them), or give an output made elsewhere with "
        "--prediction-dir."
    )


def _constraint_model_message(model_name: str) -> str:
    return (
        f"constraints are read by the model {CONSTRAINT_MODEL_NAME} only, not by {model_name!r}: "
        "every other model has its constraint embedders switched off "
        "(configs/configs_base.py:282-307, configs/configs_model_type.py:123-136), so the "
        "section would be ignored and the prediction would look constrained without being "
        f"(protenix/model/protenix.py:218-228). Use model_name={CONSTRAINT_MODEL_NAME!r}."
    )


def _weights_message(weights: Any, model_name: str) -> str:
    """Why ``weights`` cannot be used, and the route that exists."""
    path = Path(getattr(weights, "path", weights))
    kind = getattr(weights, "kind", "directory" if path.is_dir() else "file")
    if kind != "directory":
        return (
            f"the Protenix runner takes its weights as a directory that holds {model_name}.pt, "
            f"and {path} is a file"
        )
    present = "is there" if (path / f"{model_name}.pt").is_file() else "is not there"
    return (
        f"weights={str(path)!r} cannot be passed to `protenix pred`: the command has no option "
        f"for the weights directory. Protenix would load {path}/{model_name}.pt ({model_name}.pt "
        f"{present}) as <load_checkpoint_dir>/{model_name}.pt, where load_checkpoint_dir is "
        "$PROTENIX_ROOT_DIR/checkpoint (~/checkpoint without the variable), fixed when the "
        "package is imported (configs/configs_inference.py:21,29; runner/inference.py:154-160). "
        "Only runner/inference.py has --load_checkpoint_dir, and it does not search the MSA. "
        f"Put or link {model_name}.pt in that directory, or set PROTENIX_ROOT_DIR in the "
        "environment that runs Protenix (the variable also moves the CCD data directory)."
    )


def _unmappable_message(details: Sequence[tuple[str, str, Sequence[str]]]) -> str:
    lines = [f"  - chain '{chain}': {', '.join(labels)}" for _, chain, labels in details]
    return (
        "Protenix cannot take these residues:\n" + "\n".join(lines) + "\n"
        "A proteinChain sequence takes the 20 standard amino acids and X, and any other amino "
        "acid only as the CCD code of a modification (docs/infer_json_format.md:60-65; "
        "protenix/data/inference/json_parser.py:333-341); the names above are none of these, so "
        "the prediction would model something else than the input. Remove or replace the "
        "residues, or pass on_unmappable_residue='x' (--on-unmappable-residue x) to send an X in "
        "their place."
    )


# ---------------------------------------------------------------------- validation


def _check_name(name: str) -> None:
    """The name is a directory and a file name of the output: refuse what could leave it."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if name in (".", "..") or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError(
            f"name {name!r} becomes a directory and a file name of the output and cannot hold a "
            "path separator or be '.' or '..'"
        )


def _resolve_seeds(seeds: Optional[Sequence[int]]) -> tuple[int, ...]:
    """The seeds as integers: the command-line default for None; non-negative and distinct."""
    if seeds is None:
        return _DEFAULT_SEEDS
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a sequence of integers, not a string")
    values: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool):
            raise ValueError(f"seeds must be integers, got {seed!r}")
        try:
            values.append(operator.index(seed))
        except TypeError:
            raise ValueError(f"seeds must be integers, got {seed!r}") from None
    if not values:
        raise ValueError("give at least one seed (the default is 101)")
    if min(values) < 0:
        raise ValueError(
            f"seeds must not be negative, got {values}: the output directory seed_<S> is read "
            "back in the numeric order of decimal seeds only"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"seeds must be distinct, got {values}")
    return tuple(values)


def _check_extra_args(extra_args: Sequence[str]) -> list[str]:
    if isinstance(extra_args, (str, bytes)):
        raise ValueError("extra_args is a list of arguments, not a string")
    arguments = [str(argument) for argument in extra_args]
    for argument in arguments:
        if argument.startswith("--"):
            flag = argument.split("=", 1)[0]
        elif argument.startswith("-") and len(argument) >= 2:
            flag = argument[:2]
        else:
            continue
        if flag in _RESERVED_FLAGS:
            raise ValueError(
                f"extra_args cannot hold {argument!r}: {_RESERVED_FLAGS[flag]}, so that the "
                "request describes the run"
            )
    return arguments


def _default_checkpoint(model_name: str) -> Optional[Path]:
    """The file ``protenix pred`` loads for ``model_name`` (``configs_inference.py:21,29``)."""
    try:
        root = os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))
    except (RuntimeError, OSError):  # no home directory
        return None
    if not root:
        return None
    return Path(root) / "checkpoint" / f"{model_name}.pt"


def _resolve_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The options of a run with every default written out, checked.

    ``raw`` may lack any key (a request made without ``make_request``); the result holds the
    keys that the key of the request hashes, except ``checkpoint_size_bytes`` and
    ``need_atom_confidence``, which ``make_request`` adds.
    """
    model_name = raw.get("model_name", DEFAULT_MODEL_NAME)
    if not isinstance(model_name, str) or not model_name or "/" in model_name:
        raise ValueError(f"model_name must be a model name such as {DEFAULT_MODEL_NAME!r}")
    dtype = raw.get("dtype", _DEFAULT_DTYPE)
    if dtype not in _DTYPES:
        raise ValueError(
            f"dtype must be one of {_DTYPES} (runner/inference.py:217-221), got {dtype!r}"
        )
    msa_server_mode = raw.get("msa_server_mode", _DEFAULT_MSA_SERVER_MODE)
    if msa_server_mode not in _MSA_SERVER_MODES:
        raise ValueError(
            f"msa_server_mode must be one of {_MSA_SERVER_MODES}, got {msa_server_mode!r}"
        )
    on_unmappable = raw.get("on_unmappable_residue", _DEFAULT_ON_UNMAPPABLE)
    if on_unmappable not in _ON_UNMAPPABLE_CHOICES:
        raise ValueError(
            f"on_unmappable_residue must be one of {_ON_UNMAPPABLE_CHOICES}, got {on_unmappable!r}"
        )
    constraints = raw.get("constraints") or None
    if constraints is not None:
        if not isinstance(constraints, Mapping):
            raise ValueError("constraints must be a dict, the content of the `constraint` section")
        if model_name != CONSTRAINT_MODEL_NAME:
            raise ValueError(_constraint_model_message(model_name))
        constraints = dict(constraints)
    covalent_bonds = raw.get("covalent_bonds") or None
    if covalent_bonds is not None:
        if isinstance(covalent_bonds, (str, bytes, Mapping)) or not all(
            isinstance(bond, Mapping) for bond in covalent_bonds
        ):
            raise ValueError(
                "covalent_bonds must be a list of dicts (docs/infer_json_format.md:206)"
            )
        covalent_bonds = [dict(bond) for bond in covalent_bonds]
    return {
        "model_name": model_name,
        "dtype": dtype,
        "use_msa_server": bool(raw.get("use_msa_server", True)),
        "msa_server_mode": msa_server_mode,
        "on_unmappable_residue": on_unmappable,
        "constraints": constraints,
        "covalent_bonds": covalent_bonds,
        "extra_args": _check_extra_args(raw.get("extra_args") or ()),
    }


# ---------------------------------------------------------------------- the input


def _protenix_chain(
    sequence: str, non_canonical: Mapping[int, str]
) -> tuple[str, list[dict[str, Any]]]:
    """The ``sequence`` and ``modifications`` of a ``proteinChain`` from an OpenFold3-style chain.

    Args:
        sequence: One letter per residue; the parent letter for a modified residue, and ``U``
            for selenocysteine (the OpenFold3 query letter).
        non_canonical: 1-based position to CCD code, for the residues that are modified.

    Raises:
        ValueError: A letter that Protenix does not take and no modification explains.
    """
    letters = list(sequence)
    modifications = {position: code for position, code in non_canonical.items()}
    for index, letter in enumerate(letters):
        position = index + 1
        if letter == "U" and position not in modifications:
            letters[index] = "X"
            modifications[position] = "SEC"
        elif letter not in _PROTEIN_LETTERS:
            raise ValueError(
                f"residue letter {letter!r} at position {position} is not one of the 20 amino "
                "acids or X, which is all a Protenix proteinChain sequence takes "
                "(docs/infer_json_format.md:60)"
            )
    return "".join(letters), [
        {"ptmType": f"CCD_{code}", "ptmPosition": position}
        for position, code in sorted(modifications.items())
    ]


def _read_entities(
    structure_path: Path,
    chain_ids: Sequence[str],
    on_unmappable_residue: str,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """``(chain ID, sequence, modifications)`` for each of ``chain_ids``, in order.

    Raises:
        ValueError: A chain is not in the structure, has no amino acid, or holds a residue that
            Protenix cannot take (every offending chain is named, none is written).
    """
    import gemmi

    from binding_metrics.metrics._openfold_run import UnmappableResidueError, _extract_query_chain

    structure = gemmi.read_structure(str(structure_path))
    if len(structure) == 0:
        raise ValueError(f"{structure_path} has no model")
    present = [chain.name for chain in structure[0]]
    for chain_id in chain_ids:
        if chain_id not in present:
            raise ValueError(
                f"chain {chain_id!r} is not in {structure_path} (chains in its first model: "
                f"{', '.join(present) or 'none'})"
            )
    entities: list[tuple[str, str, list[dict[str, Any]]]] = []
    unmappable: list[tuple[str, str, Sequence[str]]] = []
    for chain_id in chain_ids:
        try:
            sequence, non_canonical = _extract_query_chain(
                structure, chain_id, on_unmappable_residue=on_unmappable_residue
            )
        except UnmappableResidueError as exc:
            unmappable.extend(exc.details)
            continue
        sequence, modifications = _protenix_chain(sequence, non_canonical)
        entities.append((chain_id, sequence, modifications))
    if unmappable:
        raise ValueError(_unmappable_message(unmappable))
    return entities


def _job_dict(request: PredictionRequest, options: Mapping[str, Any]) -> dict[str, Any]:
    """The Protenix job (one element of the input JSON list) for ``request``."""
    chain_ids = [request.receptor_chain, request.binder_chain]
    entities = _read_entities(Path(request.input_path), chain_ids, options["on_unmappable_residue"])
    sequences = []
    for chain_id, sequence, modifications in entities:
        protein_chain: dict[str, Any] = {"sequence": sequence, "count": 1, "id": [chain_id]}
        if modifications:
            protein_chain["modifications"] = modifications
        sequences.append({"proteinChain": protein_chain})
    job: dict[str, Any] = {"name": request.name, "sequences": sequences}
    if options["covalent_bonds"] is not None:
        job["covalent_bonds"] = options["covalent_bonds"]
    if options["constraints"] is not None:
        job["constraint"] = options["constraints"]
    return job


# ---------------------------------------------------------------------- the process


def _installed_version(python_cmd: Optional[Sequence[str]] = None) -> Optional[str]:
    """The installed ``protenix`` version, or None when it cannot be told.

    Args:
        python_cmd: The command that starts the interpreter to ask (``["conda", "run", "-n",
            "protenix", "python"]``); None asks the current interpreter without a process.
    """
    if python_cmd is None:
        try:
            return importlib.metadata.version("protenix")
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
        logger.debug("could not ask %s for the protenix version: %s", list(python_cmd), exc)
        return None
    lines = probe.stdout.strip().splitlines() if probe.returncode == 0 else []
    return lines[-1].strip() or None if lines else None


def _lines(text: str) -> list[str]:
    """Non-empty lines of ``text``; a progress bar's carriage returns keep only the last state."""
    return [line for raw in text.split("\n") if (line := raw.split("\r")[-1].strip())]


def _key_line(lines: Sequence[str]) -> str:
    """The line that names a failure: the last exception line, else the last line."""
    for line in reversed(lines):
        if re.match(r"^[\w.]*(Error|Exception|Exit|Interrupt)\b", line):
            return line
    return lines[-1] if lines else ""


def _run_process(cmd: Sequence[str], cwd: Path) -> tuple[int, str, list[str]]:
    """Run ``cmd``, echo its output to ``sys.stderr`` and keep what the error message needs.

    stdout and stderr are read as one stream, because Protenix prints the MSA failure to stdout
    and logs the rest to stderr.

    Returns:
        The exit status, the last ``_OUTPUT_TAIL_CHARS`` characters of the output and the lines
        that report a failed MSA search.

    Raises:
        FileNotFoundError: The executable is not found.
    """
    try:
        process = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(cwd)
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"cannot start {cmd[0]!r} ({exc}): install Protenix (pip install protenix) in the "
            "environment that runs it, or pass conda_env"
        ) from exc
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    pending = ""
    msa_failures: list[str] = []

    def scan(text: str) -> None:
        for line in re.split(r"[\r\n]", text):
            if any(marker in line for marker in _MSA_FAILURE_MARKERS):
                msa_failures.append(line.strip()[:_LINE_CHARS])

    try:
        while chunk := process.stdout.read1(4096):
            text = decoder.decode(chunk)
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except (OSError, ValueError):  # a closed or unwritable console must not stop the run
                pass
            tail = (tail + text)[-_OUTPUT_TAIL_CHARS:]
            *complete, pending = re.split(r"[\r\n]", pending + text)
            scan("\n".join(complete))
        pending += decoder.decode(b"", final=True)
        scan(pending)
        process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
    return process.returncode, tail, msa_failures


def _recorded_errors(output_dir: Path, name: str) -> list[str]:
    """What ``protenix pred`` wrote to ``ERR/<name>.txt`` and ``ERR/error.txt``, one line each.

    Each file is a message followed by a traceback (``runner/inference.py:577-582,629-634``); the
    first line and the last line (the exception) are kept.
    """
    recorded = []
    for file_name in dict.fromkeys((f"{name}.txt", "error.txt")):
        path = Path(output_dir) / "ERR" / file_name
        if not path.is_file():
            continue
        try:
            lines = _lines(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            logger.debug("%s could not be read: %s", path, exc)
            continue
        if not lines:
            continue
        shown = lines[0][:_LINE_CHARS]
        if lines[-1] != lines[0]:
            shown += f" ... {lines[-1][:_LINE_CHARS]}"
        recorded.append(f"ERR/{file_name}: {shown}")
    return recorded


def _failure_message(
    headline: str,
    *,
    recorded: Sequence[str],
    output_tail: str,
    cmd: Sequence[str],
) -> str:
    """The text of a ``ProtenixRunError``: reason first, then the evidence."""
    lines = _lines(output_tail)
    parts = [headline]
    parts.extend(f"Recorded by Protenix: {entry}" for entry in recorded)
    failed = next((line for line in reversed(lines) if _RUN_FAILED_MARKER in line), None)
    if failed is not None:
        parts.append(f"Protenix log: {failed[: 2 * _LINE_CHARS]}")
    lowered = output_tail.lower() + " ".join(recorded).lower()
    for needles, hint in _KNOWN_FAILURE_HINTS:
        if any(needle.lower() in lowered for needle in needles):
            parts.append(f"Hint: {hint}")
            break
    if lines:
        shown = "\n".join(f"  {line[:_LINE_CHARS]}" for line in lines[-_LINES_IN_MESSAGE:])
        parts.append(f"Last lines of output:\n{shown}")
    parts.append(f"Command: {' '.join(cmd)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------- the runner


class ProtenixRunner(PredictionRunner):
    """Runs ``protenix pred`` in the current environment or in a conda environment.

    Args:
        conda_env: Name of the conda environment that has Protenix (``conda run -n <env>``);
            None uses the ``protenix`` on PATH.
    """

    name = "protenix"
    #: Protenix takes its weights as a directory plus a model name (``<directory>/<model>.pt``).
    #: The command has no option for the directory, so a request with weights is refused with
    #: the reason (see the module docstring).
    supports_custom_weights = True
    weights_kind = "directory"

    def __init__(self, conda_env: Optional[str] = None):
        self.conda_env = conda_env
        self._version: Optional[str] = None
        self._version_probed = False

    # ------------------------------------------------------------------ the machine

    def version(self) -> Optional[str]:
        """The installed ``protenix`` version, or None when it cannot be told.

        Read from the package metadata (in the conda environment when one is set, which starts a
        process); asked once per runner.
        """
        if not self._version_probed:
            python_cmd = (
                None if self.conda_env is None else ["conda", "run", "-n", self.conda_env, "python"]
            )
            self._version = _installed_version(python_cmd)
            self._version_probed = True
        return self._version

    def is_available(self) -> bool:
        """True when ``protenix`` is on PATH, or the conda environment has the package."""
        if self.conda_env is None:
            return shutil.which("protenix") is not None
        return self.version() is not None

    # ------------------------------------------------------------------ the request

    def make_request(
        self,
        input_path: str | Path,
        *,
        name: str,
        binder_chain: Optional[str] = None,
        receptor_chain: Optional[str] = None,
        mode: str = "predict",
        seeds: Optional[Sequence[int]] = None,
        num_samples: int = _DEFAULT_NUM_SAMPLES,
        use_msa_server: bool = True,
        msa_server_mode: str = _DEFAULT_MSA_SERVER_MODE,
        model_name: str = DEFAULT_MODEL_NAME,
        dtype: str = _DEFAULT_DTYPE,
        weights: Optional[str | Path | WeightsRef] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        covalent_bonds: Optional[Sequence[Mapping[str, Any]]] = None,
        on_unmappable_residue: str = _DEFAULT_ON_UNMAPPABLE,
        extra_args: Sequence[str] = (),
    ) -> PredictionRequest:
        """The store request of one Protenix run (see the module docstring for what it holds).

        Args:
            input_path: The complex structure (PDB or mmCIF).
            name: Job name; the output files are named after it.
            binder_chain, receptor_chain: Chain roles; both are required. The receptor is
                entity 1 and the binder entity 2 of the input JSON.
            mode: ``"predict"``; the other modes raise (see the module docstring).
            seeds: Seed values; None is ``[101]``, the default of ``protenix pred``.
            num_samples: Structures per seed (``--sample``).
            use_msa_server: Search the MSA over the network (``--use_msa``); False runs without.
            msa_server_mode: ``"protenix"`` or ``"colabfold"`` (``--msa_server_mode``).
            model_name: Model variant (``--model_name``).
            dtype: ``"bf16"``, ``"fp32"`` or ``"fp16"`` (``--dtype``).
            weights: A directory that holds ``<model_name>.pt``, or a ``WeightsRef`` of one. Not
                supported: ``protenix pred`` has no option for the directory, so any value raises
                ``ValueError`` with the route that exists.
            constraints: The content of the ``constraint`` section, written as given; only the
                model ``protenix_base_constraint_v0.5.0`` reads it.
            covalent_bonds: The ``covalent_bonds`` list, written as given (a ring closure).
            on_unmappable_residue: ``"error"`` or ``"x"`` (see the module docstring).
            extra_args: Extra command-line arguments, passed verbatim after the runner's own.

        Raises:
            ValueError: A mode other than ``predict``, ``weights``, a missing or equal
                chain role, an unusable seed, a choice that is not offered, ``constraints`` with
                a model that ignores them, or an ``extra_args`` entry that sets what the runner
                sets.
        """
        if mode != "predict":
            raise ValueError(_unsupported_mode_message(mode))
        if weights is not None:
            raise ValueError(_weights_message(weights, model_name))
        _check_name(name)
        if not (binder_chain and receptor_chain):
            raise ValueError("a Protenix request needs binder_chain and receptor_chain")
        if binder_chain == receptor_chain:
            raise ValueError(
                f"binder_chain and receptor_chain are both {binder_chain!r}: they are two "
                "entities of the input"
            )
        options = _resolve_options(
            {
                "model_name": model_name,
                "dtype": dtype,
                "use_msa_server": use_msa_server,
                "msa_server_mode": msa_server_mode,
                "on_unmappable_residue": on_unmappable_residue,
                "constraints": constraints,
                "covalent_bonds": covalent_bonds,
                "extra_args": extra_args,
            }
        )
        checkpoint = _default_checkpoint(options["model_name"])
        checkpoint_size = checkpoint.stat().st_size if checkpoint and checkpoint.is_file() else None
        return PredictionRequest(
            self.name,
            name,
            mode=mode,
            input_path=input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            seeds=_resolve_seeds(seeds),
            num_samples=num_samples,
            model_version=self.version() or "",
            options={
                **options,
                "need_atom_confidence": True,
                "checkpoint_size_bytes": checkpoint_size,
            },
        )

    # ------------------------------------------------------------------ running

    def prepare(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Write the input JSON to ``<work_dir>/input/<name>.json`` and return its path.

        The structure is read first: a residue Protenix cannot take, or a chain the structure
        lacks, raises before any file is written. Protenix is not started.
        """
        options = self._check_request(request)
        job = _job_dict(request, options)
        path = Path(work_dir).resolve() / "input" / f"{request.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([job], indent=2) + "\n", encoding="utf-8")
        return path

    def run(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Run ``protenix pred`` for ``request`` and return ``<work_dir>/predictions``.

        Raises:
            ValueError: As for ``prepare``, or a mode other than ``predict``.
            FileNotFoundError: ``protenix`` is not on PATH and no conda environment is set.
            ProtenixRunError: Protenix exited non-zero, reported a failed MSA search, or left no
                complete output for every seed and sample (see the module docstring).
        """
        work_dir = Path(work_dir).resolve()
        job_path = self.prepare(request, work_dir)
        output_dir = work_dir / "predictions"
        cmd = self._command(request, job_path, output_dir)
        logger.info("Running Protenix: %s", " ".join(cmd))
        returncode, tail, msa_failures = _run_process(cmd, work_dir)
        recorded = _recorded_errors(output_dir, request.name)

        def fail(headline: str, code: Optional[int] = None) -> ProtenixRunError:
            # The store renames the work directory when the run is stored, so a reason that
            # names it would point at nothing.
            message = _failure_message(headline, recorded=recorded, output_tail=tail, cmd=cmd)
            return ProtenixRunError(
                message.replace(str(work_dir), "<work_dir>"),
                returncode=code,
                cmd=cmd,
                output_tail=tail,
            )

        if returncode != 0:
            lines = _lines(tail)
            reason = f": {_key_line(lines)[:_LINE_CHARS]}" if lines else ""
            raise fail(f"Protenix exited with status {returncode}{reason}", returncode)
        if msa_failures:
            raise fail(
                "The MSA search failed and Protenix continued with the sequence itself as the "
                "MSA, so this is not the prediction the request asks for "
                f"(use_msa_server=True): {msa_failures[0]}. Run again when the MSA server is "
                "reachable, or pass use_msa_server=False to run without an MSA."
            )
        missing = self._missing_output(request, output_dir)
        if missing:
            raise fail(
                f"Protenix exited normally but wrote no complete output for '{request.name}' "
                f"(below predictions/): {'; '.join(missing)}"
            )
        return output_dir

    # ------------------------------------------------------------------ arguments

    def check_weights(self, request: PredictionRequest) -> None:
        """Raise ``ValueError`` for a request with weights: the command cannot take them.

        The ABC check comes first (weights that are a file, not a directory); then every request
        with weights is refused with the route that exists (see the module docstring).
        """
        super().check_weights(request)
        if request.weights is not None:
            model_name = _resolve_options(request.options)["model_name"]
            raise ValueError(_weights_message(request.weights, model_name))

    def _check_request(self, request: PredictionRequest) -> dict[str, Any]:
        """Refuse a request this runner cannot run; return its options with defaults written."""
        self.check_weights(request)
        if request.mode != "predict":
            raise ValueError(_unsupported_mode_message(request.mode))
        if request.model != self.name:
            raise ValueError(f"the {self.name} runner cannot run a '{request.model}' request")
        if request.input_path is None:
            raise ValueError("a Protenix request needs a structure file as input_path")
        _check_name(request.name)
        if not (request.binder_chain and request.receptor_chain):
            raise ValueError("a Protenix request needs binder_chain and receptor_chain")
        if request.binder_chain == request.receptor_chain:
            raise ValueError("binder_chain and receptor_chain must be different chains")
        _resolve_seeds(request.seeds)
        return _resolve_options(request.options)

    def _command(self, request: PredictionRequest, job_path: Path, output_dir: Path) -> list[str]:
        """The ``protenix pred`` command line (see the module docstring)."""
        options = _resolve_options(request.options)
        executable = "protenix" if self.conda_env else shutil.which("protenix") or "protenix"
        cmd = [
            executable,
            "pred",
            "--input",
            str(job_path),
            "--out_dir",
            str(output_dir),
            "--seeds",
            ",".join(str(seed) for seed in _resolve_seeds(request.seeds)),
            "--sample",
            str(request.num_samples),
            "--dtype",
            options["dtype"],
            "--model_name",
            options["model_name"],
            "--use_msa",
            "true" if options["use_msa_server"] else "false",
            "--msa_server_mode",
            options["msa_server_mode"],
            "--need_atom_confidence",
            "true",
            *options["extra_args"],
        ]
        if self.conda_env:
            cmd = ["conda", "run", "-n", self.conda_env, "--no-capture-output", *cmd]
        return cmd

    @staticmethod
    def _missing_output(request: PredictionRequest, output_dir: Path) -> list[str]:
        """What is absent from ``output_dir``: one entry per seed or sample that lacks a file.

        Every requested seed must have every sample with its structure, summary and full-data
        file. The adapter finds the files of a seed by its position among the seed directories,
        so a seed that is missing would shift every later one.
        """
        from binding_metrics.predictors.protenix import ProtenixParser

        parser = ProtenixParser()
        seeds = sorted(_resolve_seeds(request.seeds))
        missing: list[str] = []
        for position, seed in enumerate(seeds, start=1):
            first = parser.find_files(output_dir, request.name, seed_index=position, sample=1)
            located = first.structure or first.scores or first.arrays
            if located is None or located.parent.parent.name != f"seed_{seed}":
                missing.append(f"no output for seed {seed}")
                continue
            for rank in range(request.num_samples):
                files = parser.find_files(
                    output_dir, request.name, seed_index=position, sample=rank + 1
                )
                absent = [
                    label
                    for label, path in (
                        ("structure", files.structure),
                        ("summary confidence", files.scores),
                        ("full-data", files.arrays),
                    )
                    if path is None
                ]
                if absent:
                    missing.append(f"seed {seed} sample {rank} lacks its {', '.join(absent)} file")
        return missing
