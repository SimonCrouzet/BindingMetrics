"""OpenFold3 runner configuration and query preparation.

Builds the runner YAML and the query JSON (with template CIFs and A3M
self-alignments) that ``run_openfold predict`` reads. The subprocess call and the
``run_openfold_*`` wrappers stay in :mod:`binding_metrics.metrics.openfold`, which
re-exports everything defined here.
"""

import codecs
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import subprocess
import sys
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

logger = logging.getLogger(__name__)

#: Seed values written into the ``"seeds"`` field of the query JSON. OpenFold3
#: derives its sampling from these, so the same query gives the same prediction
#: (up to GPU non-determinism). The value carries no meaning; pass ``seeds=`` to
#: the ``prepare_*`` and ``run_openfold_*`` functions to use others.
_DEFAULT_QUERY_SEEDS: tuple[int, ...] = (42,)


def _query_seeds(seeds: Sequence[int]) -> list[int]:
    """Validate ``seeds`` and return them as a list of ints for the query JSON."""
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of integers, not a string.")
    values = [int(s) for s in seeds]
    if not values:
        raise ValueError("seeds must contain at least one integer.")
    return values


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


def _drop_removed_presets(presets: Sequence[str]) -> list[str]:
    """Return ``presets`` without ``pae_enabled``, warning when it was given.

    OpenFold3 0.4.1 removed the preset and 0.5.0 still only logs a deprecation warning
    for it, because the PAE head has been on by default since 0.4.0. The name stays
    in the list only when the current interpreter has an openfold3 older than 0.4.0,
    where PAE is off unless the preset asks for it.
    """
    kept = list(presets)
    if "pae_enabled" not in kept:
        return kept
    installed = installed_openfold3_version()
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
) -> Path:
    """Write a runner YAML with model presets and optional template settings.

    Args:
        output_dir: Directory in which to write the file.
        presets: List of model preset names, e.g. ``["predict", "low_mem"]``. A
            ``"pae_enabled"`` entry is dropped with a ``DeprecationWarning`` (the
            preset was removed in OpenFold3 0.4.1 and the PAE head is always on).
        template_dir: If given, adds ``template_preprocessor_settings`` with
            ``structure_directory`` pointing here and
            ``fetch_missing_structures: false`` so OF3 uses local CIFs only. It also adds
            ``msa_computation_settings.cleanup_msa_dir: false``: with the MSA server and
            templates on (the defaults), OpenFold3 deletes ``structure_directory.parent``
            at the end of a run (0.3.1 to 0.5.0; unreleased ``main`` no longer deletes
            user-chosen directories), which here is the folder that holds the query JSON,
            the A3M files and the template CIFs. In 0.5.0 ``cleanup_msa_dir`` guards only
            that deletion; 0.4.0 also removed the MSA output directory with it.

    Returns:
        Path to the written YAML file.
    """
    presets = _drop_removed_presets(presets)
    # TODO(#67): seeds go here as experiment_settings.seeds; OpenFold3 ignores the query "seeds".
    cfg: dict = {"model_update": {"presets": presets}}
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


def _key_line(lines: Sequence[str]) -> str:
    """The line that names the failure: the last exception line, else the last line."""
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
        super().__init__(
            f"OpenFold3 exited normally but failed on every query in {output_dir}:\n{lines}"
        )


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


def _failed_query_reasons(output_dir: Path, not_before: float = 0.0) -> dict[str, str]:
    """Return ``{query name: reason}`` for the queries OpenFold3 reports as failed.

    Reads ``<output_dir>/summary.txt`` for the names and ``logs/predict_err_rank<N>.log``
    for the error of each. OpenFold3 exits with status 0 when a query fails (out of memory
    or any other exception inside the forward pass), so this is the only trace of it. An empty
    dict means no failure was reported, or that there is no summary.
    """
    summary = _read_run_summary(output_dir, not_before)
    if summary is None:
        return {}
    logged = _error_log_reasons(output_dir)
    reasons = {}
    for name in summary.failed_queries:
        if name in logged:
            why, log = logged[name]
            reasons[name] = f"OpenFold3 failed on this query: {why} (see {log})"
        else:
            reasons[name] = (
                "OpenFold3 reported this query as failed (see "
                f"{Path(output_dir) / 'summary.txt'} and the logs directory)"
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
    settings the toolkit does not write (structure format, MSA server URL, seeds, ...) come
    from that file. Returns None when there is none.
    """
    candidate = _openfold_cache_dir() / "runner.yml"
    return candidate if candidate.is_file() else None


@dataclasses.dataclass(frozen=True)
class OpenFoldRunInfo:
    """What a finished ``run_openfold`` call reports besides its output files.

    ``failed_queries`` maps the queries that OpenFold3 skipped (it still exits with status 0)
    to their reasons; ``user_default_runner_yaml`` is the file it merged under the toolkit's
    runner YAML, if any.
    """

    failed_queries: dict[str, str]
    user_default_runner_yaml: Optional[Path]


def _run_openfold_command(cmd: Sequence[str], output_dir: Path) -> OpenFoldRunInfo:
    """Run an OpenFold3 command line and explain how it failed.

    stdout goes to the parent's stdout untouched. stderr is echoed to ``sys.stderr`` as it
    arrives and its last few kilobytes are kept, so a non-zero exit raises
    :class:`OpenFoldRunError` with the reason instead of only the exit status. After an exit
    with status 0 the run's ``summary.txt`` is read: queries that failed inside OpenFold3 are
    logged, and when every query failed :class:`OpenFoldQueryError` is raised.

    Args:
        cmd: Full command line (``run_openfold predict ...``, possibly behind ``conda run``).
        output_dir: The ``--output_dir`` of the command; ``summary.txt`` and ``logs/`` are
            read from there.

    Returns:
        The queries that failed in a run that otherwise succeeded, and the user-default
        runner YAML that was merged in.

    Raises:
        OpenFoldRunError: The process exited non-zero.
        OpenFoldQueryError: The process exited with status 0 and every query failed.
    """
    default_yaml = _user_default_runner_yaml()
    if default_yaml is not None:
        logger.warning(
            "OpenFold3 merges the user-default runner YAML %s under the toolkit's runner YAML "
            "(openfold3 >= 0.5); settings it holds that the toolkit does not write, such as "
            "seeds, structure_format or the MSA server URL, apply to this run.",
            default_yaml,
        )
    started = time.time()
    process = subprocess.Popen(list(cmd), stderr=subprocess.PIPE)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    try:
        while chunk := process.stderr.read1(4096):
            text = decoder.decode(chunk)
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
        process.stderr.close()
    if process.returncode != 0:
        raise OpenFoldRunError(process.returncode, list(cmd), tail)

    # A file written a moment before the run started still counts as an earlier run's.
    failures = _failed_query_reasons(output_dir, not_before=started - 2.0)
    if failures:
        logger.warning(
            "OpenFold3 exited normally but failed on %d quer%s: %s",
            len(failures),
            "y" if len(failures) == 1 else "ies",
            "; ".join(f"{name}: {why}" for name, why in failures.items()),
        )
        summary = _read_run_summary(output_dir, not_before=started - 2.0)
        if summary is not None and summary.total and len(failures) >= summary.total:
            raise OpenFoldQueryError(output_dir, failures)
    return OpenFoldRunInfo(failed_queries=failures, user_default_runner_yaml=default_yaml)


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


def _extract_chain_to_cif(
    structure, chain_id: str, output_path: Path, sequence: str = "", *, source: str = ""
) -> None:
    """Write a single chain from a gemmi Structure to a CIF file.

    Also patches the missing mmCIF metadata tables required by OF3's template
    preprocessor (``_pdbx_audit_revision_history``, ``_entity_poly``,
    ``_pdbx_poly_seq_scheme``).

    Args:
        structure: Source ``gemmi.Structure``.
        chain_id: Chain ID to extract (taken from the first model).
        output_path: Destination CIF file path.
        sequence: One-letter amino acid sequence for this chain (used to
            populate ``_entity_poly``). If empty, extracted from structure atoms.
        source: File the structure was read from, named in the error message.

    Raises:
        ValueError: If the chain is not in the structure; nothing is written then.
    """
    import gemmi

    _require_chain(structure, chain_id, source)

    new_st = gemmi.Structure()
    new_st.cell = structure.cell
    new_st.spacegroup_hm = structure.spacegroup_hm
    new_model = gemmi.Model("1")
    for chain in structure[0]:
        if chain.name == chain_id:
            new_model.add_chain(chain.clone())
            break
    new_st.add_model(new_model)
    new_st.make_mmcif_document().write_file(str(output_path))

    # OF3 template preprocessor requires metadata tables that OpenMM/gemmi
    # do not write. Patch them in using gemmi's CIF API.
    if not sequence:
        sequence = _extract_sequence_from_structure(new_st, chain_id)

    doc = gemmi.cif.read(str(output_path))
    block = doc.sole_block()

    # 1. Release date — OF3 requires this; use a sentinel far-past date so
    #    no template release-date filter will discard it.
    rdh = block.init_loop("_pdbx_audit_revision_history.", ["ordinal", "revision_date"])
    rdh.add_row(["1", "1900-01-01"])

    # 2. entity_poly — entity_id → canonical 1-letter sequence.
    #    gemmi's make_mmcif_document() writes entity_id as "?" in struct_asym,
    #    so we always use "1" (single-chain, single-entity template).
    entity_id = "1"
    ep = block.init_loop("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"])
    ep.add_row([entity_id, sequence])

    # 3. pdbx_poly_seq_scheme — asym_id → entity_id.
    #    Materialise struct_asym rows eagerly before init_loop (block.find()
    #    returns a lazy view that is invalidated by subsequent init_loop calls).
    sa = block.find(["_struct_asym.id"])
    asym_ids = [row[0] for row in sa] if sa else [chain_id]
    pss = block.init_loop("_pdbx_poly_seq_scheme.", ["asym_id", "entity_id"])
    for asym_id in asym_ids:
        pss.add_row([asym_id, entity_id])

    doc.write_file(str(output_path))


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
    ``{entry_id}.cif`` in the ``template_preprocessor_settings.structure_directory``.

    Args:
        sequence: One-letter amino acid sequence.
        query_id: Identifier for the query (first) sequence.
        entry_id: Template entry identifier — must contain no underscores.
            OF3 looks for ``{entry_id}.cif`` in the template directory.
        chain_id: Chain identifier within the template CIF (e.g. ``"A"``).
        output_path: Destination A3M file path.
    """
    n = len(sequence)
    template_header = f"{entry_id}_{chain_id}/{1}-{n}"
    output_path.write_text(
        f">{query_id}/1-{n}\n{sequence}\n>{template_header}\n{sequence}\n", encoding="utf-8"
    )


def _query_chain(
    chain_id: str,
    sequence: str,
    non_canonical_residues: dict[int, str],
    template_alignment_file_path: Optional[str] = None,
) -> dict:
    """Build the query JSON dict of one protein chain.

    ``non_canonical_residues`` is written only when it is not empty, so queries of chains
    made of standard residues are the same as before it existed. OpenFold3 reads its keys
    as 1-based residue positions; JSON needs them as strings.
    """
    chain: dict = {"molecule_type": "protein", "chain_ids": [chain_id], "sequence": sequence}
    # TODO(#77): "cyclic": True for a binder closed head to tail.
    if non_canonical_residues:
        chain["non_canonical_residues"] = {str(i): c for i, c in non_canonical_residues.items()}
    if template_alignment_file_path is not None:
        # TODO(#68): template_cif_paths instead; the MSA-server step overwrites this A3M path.
        chain["template_alignment_file_path"] = template_alignment_file_path
    return chain


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
) -> Path:
    """Prepare an OpenFold3 query JSON for binder refolding with receptor as template.

    **Mode 2 — target-fixed refolding:**
    The receptor chain is provided as a structural template so that OF3 is
    conditioned on the known receptor geometry. The binder chain is predicted
    from sequence only (no template). This answers: "given the known receptor,
    can OF3 recover the bound binder conformation?"

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json     — OF3 input JSON
          {query_name}_receptor.a3m   — self-alignment for receptor
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
        receptor_chain: Chain ID of the receptor/target to fix as template.
        binder_chain: Chain ID of the binder to refold (no template).
        query_name: Name for the prediction query (used in file names and the
            OF3 ``name`` field).
        output_dir: Directory to write query JSON and supporting files.
        template_cif_path: Optional path to a pre-prepared receptor CIF
            (e.g., after MD relaxation). Must be a monomer (one chain).
            If None, the receptor chain is extracted from
            ``complex_structure_path``.
        seeds: Seed values written to the query JSON's ``"seeds"`` field
            (default ``(42,)``). One prediction is made per seed.
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written when a chain
            holds a residue that OpenFold3 cannot take; ``"x"`` sends an ``X`` for it
            and logs a warning. D-amino acids and modified residues are not affected:
            they go to ``non_canonical_residues`` with their CCD code.

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If a specified chain is not found or has no amino acids,
            or ``seeds`` is empty.
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    import gemmi

    seed_values = _query_seeds(seeds)
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

    # Template source: the provided CIF (e.g. after MD relaxation) or the complex. Its chain is
    # checked before anything is written, because the relaxed file may name chains differently.
    if template_cif_path is not None:
        template_src = gemmi.read_structure(str(template_cif_path))
        template_source = str(template_cif_path)
    else:
        template_src, template_source = st, str(complex_structure_path)
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

    # A3M self-alignment for receptor — header: receptor_{chain}/{1}-{N}
    a3m_path = output_dir / f"{query_name}_receptor.a3m"
    _write_a3m_self_alignment(
        receptor_seq,
        f"query_{receptor_chain}",
        receptor_entry_id,
        receptor_chain,
        a3m_path,
    )

    # Query JSON — receptor has template, binder is free (sequence only)
    query = {
        "seeds": seed_values,
        "queries": {
            query_name: {
                "chains": [
                    _query_chain(
                        receptor_chain,
                        receptor_seq,
                        receptor_nc,
                        template_alignment_file_path=str(a3m_path),
                    ),
                    _query_chain(binder_chain, binder_seq, binder_nc),
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
) -> Path:
    """Prepare an OpenFold3 query JSON to score an existing complex structure.

    **Mode 1 — structure scoring:**
    Both receptor and binder chains are provided as structural templates so
    that OF3 evaluates the known conformation rather than predicting de-novo.
    OF3 outputs confidence scores (pLDDT, pTM, ipTM, etc.) reflecting how
    self-consistent it finds that specific structure.

    Files written under ``output_dir``:

    .. code-block:: text

        {output_dir}/
          {query_name}_query.json
          {query_name}_receptor.a3m
          {query_name}_binder.a3m
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
        seeds: Seed values written to the query JSON's ``"seeds"`` field
            (default ``(42,)``). One prediction is made per seed.
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError` before anything is written when a chain
            holds a residue that OpenFold3 cannot take; ``"x"`` sends an ``X`` for it
            and logs a warning. D-amino acids and modified residues are not affected:
            they go to ``non_canonical_residues`` with their CCD code.

    Returns:
        Path to the written query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty.
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    import gemmi

    seed_values = _query_seeds(seeds)
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

    # Template source: use template_cif_path if provided, else the complex. Both chains are
    # checked before anything is written, because a relaxed file may name chains differently.
    if template_cif_path is not None:
        template_src = gemmi.read_structure(str(template_cif_path))
        template_source = str(template_cif_path)
    else:
        template_src, template_source = st, str(complex_structure_path)
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

    _extract_chain_to_cif(
        template_src,
        receptor_chain,
        templates_dir / f"{receptor_entry_id}.cif",
        sequence=receptor_seq,
        source=template_source,
    )
    _extract_chain_to_cif(
        template_src,
        binder_chain,
        templates_dir / f"{binder_entry_id}.cif",
        sequence=binder_seq,
        source=template_source,
    )

    # A3M self-alignments — header: {entry_id}_{chain_id}/{1}-{N}
    receptor_a3m = output_dir / f"{query_name}_receptor.a3m"
    binder_a3m = output_dir / f"{query_name}_binder.a3m"
    _write_a3m_self_alignment(
        receptor_seq,
        f"query_{receptor_chain}",
        receptor_entry_id,
        receptor_chain,
        receptor_a3m,
    )
    _write_a3m_self_alignment(
        binder_seq,
        f"query_{binder_chain}",
        binder_entry_id,
        binder_chain,
        binder_a3m,
    )

    # Query JSON — OF3 format: {"seeds": [...], "queries": {"name": {"chains": [...]}}}
    query = {
        "seeds": seed_values,
        "queries": {
            query_name: {
                "chains": [
                    _query_chain(
                        receptor_chain,
                        receptor_seq,
                        receptor_nc,
                        template_alignment_file_path=str(receptor_a3m),
                    ),
                    _query_chain(
                        binder_chain,
                        binder_seq,
                        binder_nc,
                        template_alignment_file_path=str(binder_a3m),
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
    """
    import gemmi

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


def prepare_batched_scoring_queries(
    samples: list[_BatchSample],
    output_dir: str | Path,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
) -> Path:
    """Prepare a single OF3 query JSON that scores multiple complexes.

    All template CIFs and A3M files are written into a shared directory
    structure so that one ``run_openfold predict`` call processes every
    sample.

    Args:
        samples: Per-sample descriptors.
        output_dir: Directory to write query JSON and supporting files.
        seeds: Seed values written to the query JSON (default ``(42,)``).
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError`, listing every affected sample, before
            anything is written; ``"x"`` sends an ``X`` and logs a warning. See
            :func:`prepare_scoring_query`.

    Returns:
        Path to the combined query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty.
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    seed_values = _query_seeds(seeds)
    rows = _read_batch_chains(samples, on_unmappable_residue)

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

        _extract_chain_to_cif(
            st, s.receptor_chain, templates_dir / f"{rec_entry}.cif", sequence=receptor_seq
        )
        _extract_chain_to_cif(
            st, s.binder_chain, templates_dir / f"{bnd_entry}.cif", sequence=binder_seq
        )

        rec_a3m = output_dir / f"{s.query_name}_receptor.a3m"
        bnd_a3m = output_dir / f"{s.query_name}_binder.a3m"
        _write_a3m_self_alignment(
            receptor_seq, f"query_{s.receptor_chain}", rec_entry, s.receptor_chain, rec_a3m
        )
        _write_a3m_self_alignment(
            binder_seq, f"query_{s.binder_chain}", bnd_entry, s.binder_chain, bnd_a3m
        )

        queries[s.query_name] = {
            "chains": [
                _query_chain(
                    s.receptor_chain,
                    receptor_seq,
                    receptor_nc,
                    template_alignment_file_path=str(rec_a3m),
                ),
                _query_chain(
                    s.binder_chain,
                    binder_seq,
                    binder_nc,
                    template_alignment_file_path=str(bnd_a3m),
                ),
            ],
        }

    query_json_path = output_dir / "batch_query.json"
    query_json_path.write_text(
        json.dumps({"seeds": seed_values, "queries": queries}, indent=2), encoding="utf-8"
    )
    return query_json_path


def prepare_batched_refolding_queries(
    samples: list[_BatchSample],
    output_dir: str | Path,
    seeds: Sequence[int] = _DEFAULT_QUERY_SEEDS,
    *,
    on_unmappable_residue: str = "error",
) -> Path:
    """Prepare a single OF3 query JSON that refolds binders for multiple complexes.

    Receptor chains are provided as structural templates; binder chains are
    predicted from sequence only.

    Args:
        samples: Per-sample descriptors.
        output_dir: Directory to write query JSON and supporting files.
        seeds: Seed values written to the query JSON (default ``(42,)``).
        on_unmappable_residue: ``"error"`` (default) raises
            :class:`UnmappableResidueError`, listing every affected sample, before
            anything is written; ``"x"`` sends an ``X`` and logs a warning. See
            :func:`prepare_refolding_query`.

    Returns:
        Path to the combined query JSON file.

    Raises:
        ValueError: If ``seeds`` is empty.
        UnmappableResidueError: See ``on_unmappable_residue``.
    """
    seed_values = _query_seeds(seeds)
    rows = _read_batch_chains(samples, on_unmappable_residue)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    rec_entries = _unique_entry_ids([s.query_name for s, *_ in rows], "rec")

    queries: dict = {}
    for s, st, receptor_seq, receptor_nc, binder_seq, binder_nc in rows:
        rec_entry = rec_entries[s.query_name]

        _extract_chain_to_cif(
            st, s.receptor_chain, templates_dir / f"{rec_entry}.cif", sequence=receptor_seq
        )

        rec_a3m = output_dir / f"{s.query_name}_receptor.a3m"
        _write_a3m_self_alignment(
            receptor_seq, f"query_{s.receptor_chain}", rec_entry, s.receptor_chain, rec_a3m
        )

        queries[s.query_name] = {
            "chains": [
                _query_chain(
                    s.receptor_chain,
                    receptor_seq,
                    receptor_nc,
                    template_alignment_file_path=str(rec_a3m),
                ),
                _query_chain(s.binder_chain, binder_seq, binder_nc),
            ],
        }

    query_json_path = output_dir / "batch_query.json"
    query_json_path.write_text(
        json.dumps({"seeds": seed_values, "queries": queries}, indent=2), encoding="utf-8"
    )
    return query_json_path
