"""AlphaFold2 and AlphaFold-Multimer, through ColabFold, as a prediction runner.

``ColabFoldRunner`` starts ``colabfold_batch`` (ColabFold, LocalColabFold) on one complex of two
chains, a receptor and a binder, folded from their sequences (mode ``predict``), and returns the
directory that ``binding_metrics.predictors.af2.AlphaFold2Parser`` loads. There is no technical
reason that only OpenFold3 can be started from here; this runner follows the scheme of
``OpenFold3Runner`` (a request with every setting that changes the output, a run in the work
directory the store gives, a check that the model wrote its output).

Public names and signatures (the module imports the standard library and the runner ABC; gemmi
is imported when a request is made, never before)::

    class ColabFoldRunner(PredictionRunner):
        ColabFoldRunner(conda_env: Optional[str] = None)
        name = "af2"; capabilities = None
        .conda_env
        .make_request(input_path, *, name: str, binder_chain: str, receptor_chain: str,
            mode: str = "predict", seeds: Optional[Sequence[int]] = None, num_samples: int = 5,
            num_recycles: Optional[int] = None, use_msa_server: bool = True,
            model_type: str = "alphafold2_multimer_v3", num_relax: int = 0,
            data_dir: Optional[str | Path] = None, on_unmappable_residue: str = "error",
            extra_args: Sequence[str] = ()) -> PredictionRequest
        .fasta_text(request) -> str
        .output_chain_map(request) -> dict[str, str]          # {"A": receptor, "B": binder}
        .prepare(request, work_dir) -> Path                   # <work_dir>/query/<name>.fasta
        .run(request, work_dir) -> Path                       # <work_dir>/predictions
        .supports_batch(request) -> bool                      # False
        .is_available() -> bool
        .version() -> Optional[str]
    class ColabFoldRunError(subprocess.CalledProcessError)

What rests on reading the source, and what was run
--------------------------------------------------
Nothing in this module was run against ColabFold: no ``colabfold_batch`` is installed where it
was written, and the tests use a stub executable that writes synthetic output. Every statement
about ColabFold below was read on 2026-10-01 in the source of ColabFold v1.6.3 (tag ``v1.6.3``,
commit 84c27d9, 2026-09-14; the head of ``main`` at that date, efbf31c, holds one more notebook
and the same ``colabfold`` package). Flags and defaults are those of ``colabfold/batch.py``
(``main``, line numbers below are of that file unless another is named); the tests re-read them
when ``BINDING_METRICS_MODEL_SOURCES`` points at a clone. Whether another release has the same
flags was not checked. What the runner writes was read against the adapter (``af2.py``), not
against a real output.

Input and chains
----------------
``input_path`` is the complex structure, a PDB or mmCIF file, and gives only the two sequences.
A prediction from sequences does not depend on the coordinates, so the request holds the
sequences (``PredictionRequest.sequences``) and no input file: two poses of the same receptor
and binder share one run and one stored entry. The sequences are read by ``_extract_query_chain``
of the OpenFold3 query builder, which maps protonation variants (HID, HIE, CYX, ...) to their
parent residue and an ``UNK`` to ``X``.

``prepare`` writes one FASTA record whose header is the job name and whose sequence is the
receptor and the binder joined by ``:`` in that order (``input.py:get_queries``, a ``:`` in a
FASTA sequence makes a complex; the header is the job name). ColabFold names the chains ``A``,
``B``, ... in the order of the distinct sequences of the record (``get_msa_and_templates`` keeps
the first appearance of each sequence and counts its copies, ``generate_input_feature`` assigns
``protein.PDB_CHAIN_IDS[chain_cnt]``, line 979), so the receptor is ``A`` and the binder ``B``
whatever their chain IDs in the input. ColabFold also puts identical sequences next to each
other; that changes nothing for a receptor and a binder that differ, and for two identical
chains the labels are interchangeable. The prediction therefore has chains ``A`` and ``B`` and a
reader that wants the input's IDs on the atoms passes
``chain_map=ColabFoldRunner.output_chain_map(request)`` (``--prediction-target-chain A
--prediction-binder-chain B`` on the command line); for 1YCR, receptor ``A`` and binder ``B``, the
map is the identity.

AlphaFold2 takes the 20 amino acids and ``X`` (``AlphaFold2Parser.capabilities`` gives the
reason). ``make_request`` raises ``ValueError`` before anything is written when a chain holds
any other residue, naming the chain and the residues: a D-amino acid, an N-methylated residue,
a phosphorylated one or any other component with its own Chemical Component Dictionary code
(the OpenFold3 builder would send it as a CCD code, ColabFold cannot), and selenocysteine. The
prediction would otherwise be of the parent residue and not of the input.
``on_unmappable_residue`` exists so that the pipeline can pass the option it has for OpenFold3;
only ``"error"`` is accepted.

Job name. ColabFold writes its files under ``safe_filename(header)`` (``input.py:8``,
``batch.py:1458``: any character that is not alphanumeric or one of ``_ . -`` becomes ``_``),
and the parser looks the files up under the request name. A name that ColabFold would change is
refused, so a request never points at files that are not there.

Modes
-----
Only ``predict`` runs. ``refold``, ``score`` and ``score-lock`` raise ``ValueError`` (from
``make_request`` and from ``prepare`` and ``run`` of a request built by hand), because the source
shows no way to give ColabFold a template per chain: ``--custom-template-path DIR`` (line 1869,
used only with ``--templates``, line 1861) names one directory of PDB or mmCIF files for the whole
job. ``mk_hhsearch_db`` (line 297) indexes every chain of every file in it as an HHsearch
database entry ``<file stem>_<chain id>``, ``get_msa_and_templates`` gives that same directory to
every distinct query sequence (line 742), and ``mk_template`` (line 162) lets HHsearch choose the
hits for that sequence's alignment. Nothing in the input says which structure is the template of
which chain, and nothing fixes the relative pose of the chains. ``--initial-guess`` (line 1972;
``predict_structure`` stores the file's coordinates as ``all_atom_positions``, line 491) is the
nearest option; whether the model uses those coordinates to start from a pose was not verified,
and ``--templates`` without a custom directory searches the whole PDB, which can return the
structure that is being scored. The documented route for these modes is to run ``colabfold_batch``
yourself and read its output with ``--prediction-dir``. A model with templates in complexes also
needs a multimer model type: ``generate_input_feature`` ignores templates for the other types
(line 951).

Settings that change the output
-------------------------------
* ``seeds``: ColabFold runs ``range(random_seed, random_seed + num_seeds)`` (line 451), so the
  seeds are consecutive integers from 0 up and the command line is ``--random-seed first
  --num-seeds count``. ``None`` gives ``(0,)``, the defaults of both flags (lines 1932, 1939).
  Another list raises ``ValueError``.
* ``num_samples`` is the number of structures per seed: ``--num-models`` (1 to 5, default 5,
  line 1945). The early-stop threshold ``--stop-at-score`` keeps its default of 100, which no
  ranking confidence exceeds, so every model runs.
* ``use_msa_server``: ``True`` writes ``--msa-mode mmseqs2_uniref_env`` (the default of the
  flag, line 1831; the sequences are sent to the ColabFold MSA server, ``--host-url``), ``False``
  writes ``--msa-mode single_sequence``, ColabFold's MSA-free mode, where
  ``get_msa_and_templates`` builds the alignment from the query itself and makes no server
  request. The alignments a server returns are not in the key.
* ``model_type``: ``--model-type``, ``alphafold2_multimer_v3`` by default, the model the adapter
  was written against. ``auto`` is not offered: for a complex it means the same
  (``set_model_type``, line 1731).
* ``num_recycles``: ``--num-recycle``; ``None`` leaves the flag out and ColabFold uses the
  model's own number.
* ``num_relax``: ``--num-relax``, the number of top-ranked structures relaxed with OpenMM; 0 (the
  default of ColabFold, line 2036) writes only ``unrelaxed`` files. ``--amber`` is not written:
  it only sets ``num_relax`` when that is 0 (line 2282), so ``--num-relax`` says it alone.
* ``data_dir``: ``--data`` (line 2025). It is the directory that holds ``params/``, the
  parameter files (``download.py:39``, ``alphafold/models.py:59``); ColabFold downloads the
  parameters of the model type into it when they are missing, which needs the network. ``None``
  leaves the flag out and ColabFold uses its cache directory. The path is part of the key, as
  the checkpoint path is for OpenFold3; a path that is not a directory raises ``ValueError``,
  since ColabFold would create it and download into it.
* ``extra_args``: more command-line arguments, passed after the others and part of the key. A
  flag that the runner writes itself, one that changes what the adapter reads (``--zip``,
  ``--jobname-prefix``) and one that turns ``predict`` into a templated run (``--templates``,
  ``--initial-guess``, ...) raises ``ValueError``.

Not written, so ColabFold's defaults hold: templates (``--templates`` is off, line 1861; the
templates of a multimer are then mock templates), ``--zip`` (the adapter reads loose files),
``--save-all``, ``--calc-extra-ptm`` and the others. ``--calc-extra-ptm`` can be given in
``extra_args``.

Running
-------
``colabfold_batch`` on PATH, or ``conda run -n <conda_env> --no-capture-output colabfold_batch``.
The command is ``colabfold_batch <fasta> <results> <flags>`` with the flags above always
written out (the defaults included), so a change of ColabFold's own defaults cannot change what a
stored request means. ``results`` is ``<work_dir>/predictions``, which ``run`` returns; ColabFold
writes ``<job>_<unrelaxed|relaxed>_rank_*_seed_*.pdb`` and ``<job>_scores_rank_*.json`` there
(``predict_structure``, lines 583, 627, 667 and 671; the layout the adapter reads), with
``log.txt`` and ``config.json`` beside them. Output of the process goes to stderr as it arrives.

``is_available()`` is true when ``colabfold_batch`` is on PATH, or when the conda environment has
the ``colabfold`` package. ``version()`` is the package version (``colabfold_batch --help``
prints none). For an executable on PATH it asks the interpreter named in the script's first
line, so a LocalColabFold installation that is not the environment of this process is found; for
a conda environment it asks ``conda run -n <env> python``. It is part of the request key and is
the empty string when it cannot be told. ColabFold is installed with ``pip`` (LocalColabFold
creates its environment by path, ``conda run -n`` takes names: put its ``bin`` directory on PATH).

Failures. ColabFold catches most errors and exits with status 0: a failed MSA request
(line 1525), a failed feature build (line 1541) and a ``RuntimeError`` of the prediction, "Not
Enough GPU memory?" (line 1667), are logged and the job is skipped. ``run`` therefore checks
that the parser finds the structures that were asked for, and when it does not, raises
``RuntimeError`` with the line of ``log.txt`` that names the failure. A non-zero exit raises
``ColabFoldRunError`` with the last lines of the output. Both texts become the reason that the
store records.

Licence: ColabFold is MIT and the AlphaFold2 parameters CC BY 4.0; nothing of either is read or
shipped here.
"""

from __future__ import annotations

import codecs
import importlib
import importlib.metadata
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import PredictionRequest

logger = logging.getLogger(__name__)

_MODEL = "af2"
_EXECUTABLE = "colabfold_batch"

#: Defaults that are written out in every request. A test compares them with ``batch.py``.
_DEFAULT_NUM_SAMPLES = 5  # --num-models
_MAX_NUM_SAMPLES = 5  # its choices are 1 to 5
_DEFAULT_SEED = 0  # --random-seed
_DEFAULT_MODEL_TYPE = "alphafold2_multimer_v3"
_MODEL_TYPES = (
    "alphafold2",
    "alphafold2_ptm",
    "alphafold2_multimer_v1",
    "alphafold2_multimer_v2",
    "alphafold2_multimer_v3",
    "deepfold_v1",
)
_MSA_SERVER_MODE = "mmseqs2_uniref_env"
_NO_MSA_MODE = "single_sequence"

#: The 20 amino acids and X, the residue types of AlphaFold2 (``AlphaFold2Parser.capabilities``).
_SEQUENCE_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWY") | {"X"}

#: Flags that the runner writes from the request; given again in ``extra_args`` they would
#: override a setting that the key records.
_OWNED_FLAGS = (
    "--msa-mode",
    "--model-type",
    "--num-models",
    "--num-seeds",
    "--random-seed",
    "--num-recycle",
    "--num-relax",
    "--amber",
    "--data",
)
#: Flags that move the output away from the layout the adapter reads, or write none.
_LAYOUT_FLAGS = ("--zip", "--jobname-prefix", "--msa-only", "--af3-json")
#: Flags that turn a prediction from sequences into a templated or seeded one.
_TEMPLATE_FLAGS = (
    "--templates",
    "--custom-template-path",
    "--custom-template-cache-path",
    "--pdb-hit-file",
    "--local-pdb-path",
    "--initial-guess",
)

_QUERY_DIRNAME = "query"
_RESULT_DIRNAME = "predictions"

#: How much of the process output is kept for the error message (characters), and how many
#: lines of it the message shows.
_TAIL_CHARS = 8000
_LINES_IN_MESSAGE = 8
_LINE_CHARS = 300

_VERSION_PROBE = "from importlib.metadata import version; print(version('colabfold'))"

_MODE_ROUTE = (
    "ColabFold's only way to give it structures, --templates with --custom-template-path DIR "
    "(colabfold/batch.py, v1.6.3), takes one directory for the whole job and offers it to every "
    "chain alike; each chain's templates are then chosen by an HHsearch of its alignment, and "
    "the source shows no way to say which structure is the template of which chain. "
)


def _mode_message(mode: str) -> str:
    """Why the runner cannot start ``mode``, and the route that exists."""
    if mode == "score-lock":
        why = (
            "no input of ColabFold fixes the relative pose of the chains. "
            + "The nearest, --initial-guess, was not verified to do it. "
        )
    else:
        why = _MODE_ROUTE
    return (
        f"ColabFoldRunner runs mode 'predict' only, not {mode!r}: {why}"
        "Run colabfold_batch yourself and give its output with --prediction-dir."
    )


def _run_module() -> Any:
    """The module with the query builder; imported here so that the package stays light."""
    return importlib.import_module("binding_metrics.metrics._openfold_run")


def _safe_job_name(name: str) -> str:
    """The name ColabFold gives its files for the FASTA header ``name`` (``input.py:8``)."""
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)


# ---------------------------------------------------------------------------- the sequences


def _non_standard_residues(structure: Any, chain_id: str) -> list[str]:
    """The residues of a chain that AlphaFold2 cannot take, as ``NAME number`` labels.

    The classification is that of ``_extract_query_chain`` (through ``_residue_letter_and_ccd``),
    which sends such a residue as a CCD code or refuses it; here every one is a problem. Waters,
    ions, caps and ligands have no backbone and are left out, as the builder leaves them out.
    """
    run_module = _run_module()
    labels: list[str] = []
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            for residue in chain:
                mapped = run_module._residue_letter_and_ccd(residue.name)
                if mapped is None:
                    backbone = {"N", "CA", "C"} <= {atom.name for atom in residue}
                    bad = backbone
                else:
                    letter, ccd_code = mapped
                    bad = ccd_code is not None or letter not in _SEQUENCE_LETTERS
                if bad:
                    number = f"{residue.seqid.num}{residue.seqid.icode.strip()}"
                    labels.append(f"{residue.name} {number}")
            return labels
    return labels


def _complex_sequences(
    input_path: str | Path, receptor_chain: str, binder_chain: str
) -> dict[str, str]:
    """The sequences of the two chains of a structure file, receptor first.

    Raises:
        ValueError: A chain is missing or holds a residue AlphaFold2 cannot take.
    """
    import gemmi

    structure = gemmi.read_structure(str(input_path))
    problems = []
    for chain_id in (receptor_chain, binder_chain):
        labels = _non_standard_residues(structure, chain_id)
        if labels:
            problems.append(f"  - chain '{chain_id}': {', '.join(labels)}")
    if problems:
        raise ValueError(
            "AlphaFold2 / ColabFold cannot take these residues:\n" + "\n".join(problems) + "\n"
            "Its input is a sequence of the 20 amino acids and X, so a D-amino acid, an "
            "N-methylated, phosphorylated or other modified residue would be folded as its parent "
            "residue or as an unknown one, and the prediction would model something else than "
            "the input. Use a model that takes modified residues (OpenFold3 sends them as "
            "Chemical Component Dictionary codes), or replace the residues yourself."
        )
    extract = _run_module()._extract_query_chain
    return {
        chain_id: extract(structure, chain_id)[0] for chain_id in (receptor_chain, binder_chain)
    }


# ---------------------------------------------------------------------------- the process


class ColabFoldRunError(subprocess.CalledProcessError):
    """``colabfold_batch`` exited with a non-zero status.

    A ``subprocess.CalledProcessError`` whose message starts with the reason: the last line of
    the output that looks like an exception, advice for the failures that have a known fix, and
    the last lines of the output. ``output`` holds the last few kilobytes of it and ``hint`` the
    advice.
    """

    def __init__(self, returncode: int, cmd, output_tail: str = ""):
        super().__init__(returncode, cmd, output=output_tail)
        self.hint = _hint(output_tail)

    def __str__(self) -> str:
        lines = _output_lines(self.output or "")
        head = f"colabfold_batch exited with status {self.returncode}"
        if lines:
            head += f": {_key_line(lines)[:_LINE_CHARS]}"
        parts = [head]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if lines:
            shown = "\n".join(f"  {line[:_LINE_CHARS]}" for line in lines[-_LINES_IN_MESSAGE:])
            parts.append(f"Last lines of output:\n{shown}")
        return "\n".join(parts)


#: Advice for failures with a known fix: (phrases of the output, advice).
_KNOWN_FAILURES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("Could not get MSA/templates", "MMseqs2 API"),
        "ColabFold could not build the alignment. With use_msa_server=True it asks the MSA "
        "server for it, which needs the network and a server that answers; "
        "use_msa_server=False folds from the single sequence, with no server, and is much "
        "less accurate.",
    ),
    (
        ("Error downloading files", "Downloading alphafold2", "storage.googleapis.com"),
        "ColabFold downloads the AlphaFold2 parameters when its data directory has none. Give "
        "data_dir a directory that holds params/ with them, or allow the download.",
    ),
)


def _hint(text: str) -> str:
    for phrases, advice in _KNOWN_FAILURES:
        if any(phrase in text for phrase in phrases):
            return advice
    return ""


def _output_lines(text: str) -> list[str]:
    """Non-empty lines of ``text``; a progress bar's carriage returns split its updates."""
    return [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]


def _key_line(lines: Sequence[str]) -> str:
    """The last line that names an exception, else the last line."""
    for line in reversed(lines):
        if re.search(r"\b\w*(Error|Exception)\b", line):
            return line
    return lines[-1]


def _run_command(command: Sequence[str]) -> None:
    """Run ``command``, echo its output to stderr and keep the tail for the error.

    ColabFold logs through ``tqdm.write`` (``colabfold/utils.py``, ``TqdmHandler``), whose
    default stream is stdout, so both streams are read together.

    Raises:
        ColabFoldRunError: The process exited non-zero.
    """
    process = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    try:
        while chunk := process.stdout.read1(4096):
            text = decoder.decode(chunk)
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except (OSError, ValueError):  # a closed or unwritable console must not stop the run
                pass
            tail = (tail + text)[-_TAIL_CHARS:]
        process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if process.returncode != 0:
        raise ColabFoldRunError(process.returncode, list(command), tail)


def _logged_failure(results: Path) -> str:
    """The line of ColabFold's ``log.txt`` that says why a job was skipped; '' when none does."""
    try:
        text = (results / "log.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if "Could not" in line or "Failed to" in line:
            return re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ ", "", line.strip())
    return ""


def _ask_python(python_command: Sequence[str]) -> Optional[str]:
    """The ``colabfold`` package version in the interpreter ``python_command``, or None."""
    try:
        probe = subprocess.run(
            [*python_command, "-c", _VERSION_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("could not ask %s for the colabfold version: %s", list(python_command), exc)
        return None
    if probe.returncode != 0:
        return None
    lines = _output_lines(probe.stdout)
    return lines[-1] if lines else None


def _script_interpreter(script: str | Path) -> Optional[list[str]]:
    """The interpreter that a console script starts, from its first lines; None if not told.

    ``pip`` writes ``#!<python>``; for a path that is too long for a shebang it writes
    ``#!/bin/sh`` and a second line ``'''exec' <python> "$0" "$@"``.
    """
    try:
        with open(script, encoding="utf-8") as handle:
            first = handle.readline().strip()
            second = handle.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not first.startswith("#!"):
        return None
    try:
        tokens = shlex.split(first[2:])
        if tokens and os.path.basename(tokens[0]) == "sh" and second.startswith("'''exec'"):
            tokens = shlex.split(second)[1:]
        elif tokens and os.path.basename(tokens[0]) == "env":
            tokens = [token for token in tokens[1:] if not token.startswith("-")]
    except ValueError:
        return None
    if not tokens or "python" not in os.path.basename(tokens[0]):
        return None
    interpreter = tokens[0] if os.sep in tokens[0] else shutil.which(tokens[0])
    return None if interpreter is None else [interpreter]


# ---------------------------------------------------------------------------- the runner


class ColabFoldRunner(PredictionRunner):
    """Runs ``colabfold_batch`` in the current environment or in a conda environment.

    Args:
        conda_env: Name of the conda environment that has ColabFold (``conda run -n <env>``);
            None uses the ``colabfold_batch`` on PATH.
    """

    name = _MODEL

    def __init__(self, conda_env: Optional[str] = None):
        self.conda_env = conda_env
        self._version: Optional[str] = None
        self._version_probed = False

    # ------------------------------------------------------------------ the machine

    def _probe_version(self) -> Optional[str]:
        if self.conda_env is not None:
            return _ask_python(
                [shutil.which("conda") or "conda", "run", "-n", self.conda_env, "python"]
            )
        script = shutil.which(_EXECUTABLE)
        interpreter = None if script is None else _script_interpreter(script)
        if interpreter is not None:
            return _ask_python(interpreter)
        try:
            return importlib.metadata.version("colabfold")
        except importlib.metadata.PackageNotFoundError:
            return None

    def version(self) -> Optional[str]:
        """The installed ``colabfold`` version, or None when it cannot be told.

        Asked once per runner (see the module docstring for which interpreter is asked).
        """
        if not self._version_probed:
            self._version = self._probe_version()
            self._version_probed = True
        return self._version

    def is_available(self) -> bool:
        """True when ``colabfold_batch`` is on PATH, or the conda environment has ``colabfold``."""
        if self.conda_env is None:
            return shutil.which(_EXECUTABLE) is not None
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
        num_recycles: Optional[int] = None,
        use_msa_server: bool = True,
        model_type: str = _DEFAULT_MODEL_TYPE,
        num_relax: int = 0,
        data_dir: Optional[str | Path] = None,
        on_unmappable_residue: str = "error",
        extra_args: Sequence[str] = (),
    ) -> PredictionRequest:
        """The store request of one ColabFold run (see the module docstring for what it holds).

        Args:
            input_path: The complex structure (PDB or mmCIF); only the sequences of the two
                chains are read.
            name: Job name; the output files are named after it. It must be made of letters,
                digits and ``_ . -``.
            binder_chain, receptor_chain: The two chains of the structure. The receptor is
                written first, so it is chain ``A`` of the prediction and the binder ``B``.
            mode: ``"predict"``; the other modes raise.
            seeds: Consecutive integers from 0 up (``--random-seed``, ``--num-seeds``); None
                gives ``[0]``.
            num_samples: Structures per seed (``--num-models``, 1 to 5).
            num_recycles: ``--num-recycle``; None leaves it to ColabFold.
            use_msa_server: Send the sequences to the ColabFold MSA server (the default); False
                folds from the single sequence (``--msa-mode single_sequence``).
            model_type: ``--model-type``.
            num_relax: ``--num-relax``, the number of top-ranked structures to relax.
            data_dir: ``--data``, the directory that holds ``params/``.
            on_unmappable_residue: Only ``"error"`` (see the module docstring).
            extra_args: Extra command-line arguments, passed verbatim after the others.

        Raises:
            ValueError: A mode other than ``predict``, a missing or equal chain role, a chain the
                structure lacks, a residue AlphaFold2 cannot take, a job name ColabFold would
                change, seeds that are not consecutive from 0 up, an unknown model type or a
                setting out of range, a ``data_dir`` that is not a directory, or an argument in
                ``extra_args`` that the runner owns.
            OSError: The structure file cannot be read.
        """
        if mode != "predict":
            raise ValueError(_mode_message(mode))
        if not (binder_chain and receptor_chain):
            raise ValueError("a ColabFold request needs binder_chain and receptor_chain")
        if binder_chain == receptor_chain:
            raise ValueError("binder_chain and receptor_chain must be two different chains")
        if on_unmappable_residue != "error":
            raise ValueError(
                "on_unmappable_residue must be 'error' for ColabFold, got "
                f"{on_unmappable_residue!r}: the only substitute for a residue AlphaFold2 has no "
                "letter for is X, and the runner does not send it"
            )
        sequences = _complex_sequences(input_path, receptor_chain, binder_chain)
        request = PredictionRequest(
            self.name,
            name,
            mode="predict",
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            sequences=sequences,
            seeds=(_DEFAULT_SEED,) if seeds is None else tuple(sorted(int(s) for s in seeds)),
            num_samples=num_samples,
            model_version=self.version() or "",
            options={
                "use_msa_server": bool(use_msa_server),
                "model_type": model_type,
                "num_recycles": num_recycles,
                "num_relax": num_relax,
                "data_dir": None if data_dir is None else str(data_dir),
                "extra_args": [str(argument) for argument in extra_args],
            },
        )
        self._check_request(request)
        return request

    @staticmethod
    def fasta_text(request: PredictionRequest) -> str:
        """The FASTA record ``colabfold_batch`` reads: ``>name`` and ``receptor:binder``."""
        return (
            f">{request.name}\n"
            f"{request.sequences[request.receptor_chain]}:{request.sequences[request.binder_chain]}\n"
        )

    @staticmethod
    def output_chain_map(request: PredictionRequest) -> dict[str, str]:
        """Chain ID in the prediction to chain ID in the input: ``{"A": receptor, "B": binder}``.

        Pass it as ``chain_map`` when a record is read, so that the atoms of its structure carry
        the IDs of the input; ColabFold names the chains in the order of the FASTA record. The
        adapter renames the atoms only: the keys of ``chain_ptm`` and ``chain_pair_iptm`` stay
        ``A`` and ``B``.
        """
        if not (request.receptor_chain and request.binder_chain):
            raise ValueError("the request has no receptor_chain and binder_chain")
        return {"A": request.receptor_chain, "B": request.binder_chain}

    # ------------------------------------------------------------------ running

    def prepare(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Write the FASTA record to ``<work_dir>/query/<name>.fasta`` and return it.

        Raises what ``run`` would raise about the request, without starting ColabFold.
        """
        self._check_request(request)
        path = Path(work_dir) / _QUERY_DIRNAME / f"{request.name}.fasta"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.fasta_text(request), encoding="utf-8")
        return path

    def run(self, request: PredictionRequest, work_dir: Path) -> Path:
        """Run ``colabfold_batch`` for ``request`` and return ``<work_dir>/predictions``.

        Raises:
            ValueError: The request is not one this runner can run (before anything is written).
            FileNotFoundError: ``colabfold_batch`` is not on PATH and no conda environment is set.
            ColabFoldRunError: ColabFold exited non-zero.
            RuntimeError: ColabFold exited normally but wrote fewer structures than asked; the
                text holds the reason it logged, when it logged one.
        """
        self._check_request(request)
        if self.conda_env is None and shutil.which(_EXECUTABLE) is None:
            raise FileNotFoundError(
                f"{_EXECUTABLE} not found on PATH. Put the bin directory of the ColabFold or "
                "LocalColabFold installation on PATH, or pass conda_env (the environment that "
                "has ColabFold)."
            )
        fasta = self.prepare(request, work_dir)
        results = Path(work_dir) / _RESULT_DIRNAME
        results.mkdir(parents=True, exist_ok=True)
        _run_command(self._command(request, fasta, results))
        self._require_output(request, results)
        return results

    # ------------------------------------------------------------------ arguments

    def _check_request(self, request: PredictionRequest) -> None:
        """Raise ``ValueError`` unless this runner can run ``request`` as it stands."""
        if request.model != self.name:
            raise ValueError(f"the af2 runner cannot run a '{request.model}' request")
        if request.mode != "predict":
            raise ValueError(_mode_message(request.mode))
        receptor, binder = request.receptor_chain, request.binder_chain
        if not (receptor and binder) or receptor == binder:
            raise ValueError("a ColabFold request needs two different chains, binder and receptor")
        if set(request.sequences) != {receptor, binder}:
            raise ValueError(
                f"a ColabFold request holds the sequences of the receptor '{receptor}' and the "
                f"binder '{binder}' and nothing else, got chains {sorted(request.sequences)}"
            )
        for chain_id, sequence in request.sequences.items():
            unknown = sorted(set(sequence) - _SEQUENCE_LETTERS)
            if not sequence or unknown:
                raise ValueError(
                    f"the sequence of chain '{chain_id}' must be written with the 20 amino acid "
                    f"letters and X in capitals, got {unknown or 'an empty sequence'}"
                )
        if _safe_job_name(request.name) != request.name:
            raise ValueError(
                f"ColabFold would write the job '{request.name}' as "
                f"'{_safe_job_name(request.name)}' (anything but letters, digits and _ . - "
                "becomes _, colabfold/input.py), and the output could not be found under the "
                "name; give the request a name made of those characters"
            )
        seeds = tuple(request.seeds)
        if not seeds or seeds != tuple(range(seeds[0], seeds[0] + len(seeds))) or seeds[0] < 0:
            raise ValueError(
                "ColabFold runs range(random_seed, random_seed + num_seeds), so the seeds must be "
                f"consecutive integers from 0 up (for example 0, 1, 2), got {list(seeds)}"
            )
        if not 1 <= request.num_samples <= _MAX_NUM_SAMPLES:
            raise ValueError(
                "num_samples is --num-models of ColabFold and must be 1 to "
                f"{_MAX_NUM_SAMPLES}, got {request.num_samples}"
            )
        options = request.options
        model_type = options.get("model_type", _DEFAULT_MODEL_TYPE)
        if model_type not in _MODEL_TYPES:
            raise ValueError(f"model_type must be one of {_MODEL_TYPES}, got {model_type!r}")
        for key in ("num_recycles", "num_relax"):
            value = options.get(key)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{key} must be a non-negative integer or None, got {value!r}")
        data_dir = options.get("data_dir")
        if data_dir and not Path(data_dir).is_dir():
            raise ValueError(
                f"data_dir '{data_dir}' is not a directory. It must hold the parameter files in "
                "params/; ColabFold would otherwise create it and download the parameters into "
                "it. Create the directory first if that is what you want."
            )
        _check_extra_args(options.get("extra_args") or [])

    def _command(self, request: PredictionRequest, fasta: Path, results: Path) -> list[str]:
        """The command line of the run; every setting is written, the defaults included."""
        options = request.options
        seeds = tuple(request.seeds)
        msa_mode = _MSA_SERVER_MODE if options.get("use_msa_server", True) else _NO_MSA_MODE
        arguments = [
            str(fasta),
            str(results),
            "--msa-mode",
            msa_mode,
            "--model-type",
            options.get("model_type", _DEFAULT_MODEL_TYPE),
            "--num-models",
            str(request.num_samples),
            "--random-seed",
            str(seeds[0]),
            "--num-seeds",
            str(len(seeds)),
        ]
        if options.get("num_recycles") is not None:
            arguments += ["--num-recycle", str(options["num_recycles"])]
        if options.get("num_relax"):
            arguments += ["--num-relax", str(options["num_relax"])]
        if options.get("data_dir"):
            arguments += ["--data", str(options["data_dir"])]
        arguments += [str(argument) for argument in options.get("extra_args") or []]
        if self.conda_env is None:
            return [_EXECUTABLE, *arguments]
        conda = shutil.which("conda") or "conda"
        return [conda, "run", "-n", self.conda_env, "--no-capture-output", _EXECUTABLE, *arguments]

    @staticmethod
    def _require_output(request: PredictionRequest, results: Path) -> None:
        """Raise unless ``results`` holds every structure that was asked for."""
        from binding_metrics.predictors.af2 import AlphaFold2Parser

        expected = request.num_samples * len(request.seeds)
        found = len(AlphaFold2Parser().list_samples(results, request.name))
        if found >= expected:
            return
        # The work directory is renamed when the store keeps the run, so a path would go stale.
        message = (
            f"ColabFold wrote {found} of {expected} structures for job '{request.name}'; its log "
            f"is {_RESULT_DIRNAME}/log.txt below the work directory (outputs/ of a stored entry)"
        )
        reason = _logged_failure(results)
        if reason:
            message += f". It logged: {reason}"
            advice = _hint(reason)
            if advice:
                message += f"\nHint: {advice}"
        raise RuntimeError(message)


def _check_extra_args(extra_args: Sequence[str]) -> None:
    """Raise ``ValueError`` for an argument that the runner owns or that changes the mode.

    ``argparse`` accepts an unambiguous prefix of a flag, so a prefix of a refused flag counts.
    """
    refused = (
        (_OWNED_FLAGS, "the runner writes it from the request, whose key records the setting"),
        (_LAYOUT_FLAGS, "it changes the files the adapter reads or leaves none"),
        (_TEMPLATE_FLAGS, "mode 'predict' folds from sequences only, and it adds templates"),
    )
    for argument in extra_args:
        flag = str(argument).split("=", 1)[0]
        if not flag.startswith("--") or len(flag) < 4:
            continue
        for flags, reason in refused:
            for refused_flag in flags:
                if flag == refused_flag or refused_flag.startswith(flag):
                    raise ValueError(f"extra_args may not hold '{argument}': {reason}")
