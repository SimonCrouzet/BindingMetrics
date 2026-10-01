"""CLI utilities shared across all binding-metrics entry points."""

from __future__ import annotations

import argparse
import difflib
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

from binding_metrics._constants import DEFAULT_MD_SAVE_INTERVAL_PS


@contextmanager
def log_to_file(log_file, mode: str = "w"):
    """Context manager: redirect stdout+stderr to *log_file* when provided.

    Usage::

        with log_to_file(args.log_file):
            # all print() calls go to the file (or stdout if log_file is None)
            ...

    ``mode`` is the ``open`` mode: ``"w"`` (default) starts the file afresh,
    ``"a"`` appends, for callers that enter the context several times on the
    same file (one batch run logging every sample to a shared ``--log-file``).
    """
    if log_file is None:
        yield
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, mode, encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = fh
    try:
        yield
    finally:
        fh.flush()
        sys.stdout = old_out
        sys.stderr = old_err
        fh.close()


def _apply_log_redirect(log_file) -> None:
    """Redirect stdout+stderr to *log_file* for the rest of the process.

    Unlike ``log_to_file``, this is a fire-and-forget helper for CLIs whose
    body is too large to wrap in a context manager.  Streams are restored when
    the process exits normally (via ``atexit``).
    """
    import atexit

    if log_file is None:
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w", encoding="utf-8", buffering=1)
    _old_out, _old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = fh

    def _restore():
        fh.flush()
        sys.stdout = _old_out
        sys.stderr = _old_err
        fh.close()

    atexit.register(_restore)


def add_log_file_arg(parser) -> None:
    """Add the --log-file argument to an argparse parser."""
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Redirect all output (stdout + stderr) to this file",
    )


def seed_arg(value: str) -> Optional[int]:
    """Parse ``--random-seed``: an integer, or ``none``/``random``/``off`` for fresh randomness."""
    if value.strip().lower() in ("none", "random", "off"):
        return None
    return int(value)


def add_random_seed_arg(parser, what: str) -> None:
    """Add ``--random-seed INT|none`` to an argparse parser.

    The default is the library-wide ``DEFAULT_RANDOM_SEED``, so a CLI run is
    reproducible unless the user asks for fresh randomness with ``none``.

    Args:
        parser: Parser or argument group to add the flag to.
        what: Which stochastic steps the seed drives, worded for the help text
            (for example ``"ion placement"``).
    """
    from binding_metrics._constants import DEFAULT_RANDOM_SEED

    parser.add_argument(
        "--random-seed",
        type=seed_arg,
        default=DEFAULT_RANDOM_SEED,
        metavar="INT|none",
        help=(
            f"Seed for {what}. A fixed integer makes the run reproducible "
            "(default: %(default)s); pass 'none' for fresh randomness each run."
        ),
    )


def md_save_interval_for(md_duration_ps: float) -> float:
    """Frame interval in ps for a pipeline relaxation whose MD lasts ``md_duration_ps``.

    ``RelaxationConfig`` refuses an MD run shorter than one save interval. The
    pipeline CLIs have no interval flag, so a short ``--md-duration-ps`` (below
    the default interval) saves one frame at the end of the run instead of
    failing. ``0`` (minimise only) and longer runs keep the default interval.
    """
    if md_duration_ps > 0:
        return min(DEFAULT_MD_SAVE_INTERVAL_PS, md_duration_ps)
    return DEFAULT_MD_SAVE_INTERVAL_PS


def add_openfold_seeds_arg(parser) -> None:
    """Add ``--openfold-seeds SEED [SEED ...]`` to an argparse parser or group.

    The default is ``None``: the OpenFold functions then keep their own default seed, 42.
    """
    parser.add_argument(
        "--openfold-seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "Seed values OpenFold3 samples with, written to its runner YAML (default: 42). "
            "It makes one seed_<value> directory per seed; the first seed given, first sample, "
            "is scored. With --predictor MODEL the seeds go to that model's runner (ColabFold: "
            "consecutive integers from 0, default 0; Boltz-2: exactly one, default 42; Protenix: "
            "default 101)."
        ),
    )


#: Help of ``--openfold-mode`` in ``binding-metrics-run`` and ``-batch``. A template carries the
#: fold of one chain and no cross-chain geometry, so ``score`` does not hand OpenFold3 the pose.
OPENFOLD_MODE_HELP = (
    "score: each chain is given its own structure from the input as a template and OpenFold3 "
    "places the binder itself, so its confidences refer to its own pose (binder_ca_rmsd and "
    "delta_com_angstrom show how far it is from the input pose); refold: only the receptor "
    "is templated and the binder is predicted from its sequence (binder_ca_rmsd is the "
    "refolding RMSD). Default: score"
)


def add_openfold_no_msa_server_arg(parser) -> None:
    """Add ``--openfold-no-msa-server`` to an argparse parser or group.

    The OpenFold3 step uses the ColabFold MSA server unless this is given. Without it OpenFold3
    has no computed MSA (a dummy MSA that holds only the query sequence is written for each chain,
    as OpenFold3's input reference suggests). The template is kept either way with the default
    ``--openfold-templates structure``; with ``alignment`` only the run without the server keeps
    it, because the server replaces the template alignments that the toolkit writes (issue #68).
    """
    parser.add_argument(
        "--openfold-no-msa-server",
        action="store_true",
        help=(
            "Do not use the ColabFold MSA server for OpenFold3: it then runs with a dummy MSA "
            "that holds only the query sequence of each chain (OpenFold3's input reference "
            "suggests this for MSA-free runs), which lowers accuracy for a natural receptor. "
            "With --openfold-templates alignment the template alignments written by the toolkit "
            "are no longer replaced by the server (issue #68); the default, structure, keeps the "
            "template with the server on too. One complex (1YCR, OpenFold3 0.5.0, one seed), "
            "binder C-alpha RMSD against the crystal pose: 1.57 A with the server and the "
            "template as a structure (the default), 1.62 A with the server and the template as "
            "an alignment (the server replaces it: no template), 21.6 A with no MSA and no "
            "template, 1.12 A with a working template and no MSA."
        ),
    )


def openfold_msa_server_kwargs(use_msa_server: bool) -> dict:
    """The keyword argument for an OpenFold3 run function, only when the server is off.

    The default (server on) is left out so that a function that predates the option, or a test
    double that replaces it, is called exactly as before.
    """
    return {} if use_msa_server else {"use_msa_server": False}


#: Values of ``--openfold-templates``, and its default (the ``template_mode`` of the run
#: functions: ``binding_metrics.metrics._openfold_run.TEMPLATE_MODES`` and
#: ``DEFAULT_TEMPLATE_MODE``, which a test keeps equal; this module does not import the metrics).
OPENFOLD_TEMPLATE_CHOICES = ("structure", "alignment")
DEFAULT_OPENFOLD_TEMPLATES = "structure"


def add_openfold_templates_arg(parser) -> None:
    """Add ``--openfold-templates {structure,alignment}`` to an argparse parser or group.

    How the template of each chain reaches OpenFold3 in ``score`` and ``refold`` mode. The
    default, ``structure``, gives the template CIF itself (OpenFold3's CIF Direct Template Mode,
    OpenFold3 0.4.2 or later), which the ColabFold MSA server does not overwrite. ``alignment``
    writes an A3M self-alignment per chain, the way earlier versions did it; the server
    overwrites it, so with the server on (also the default) the run has no template. The setting
    is OpenFold3's: there is no ``--prediction-templates``.
    """
    parser.add_argument(
        "--openfold-templates",
        choices=OPENFOLD_TEMPLATE_CHOICES,
        default=DEFAULT_OPENFOLD_TEMPLATES,
        help=(
            "How the template of each chain reaches OpenFold3 (modes score and refold; OpenFold3 "
            "only, also for --predictor of3). structure (default): the template CIF itself "
            "(OpenFold3's CIF Direct Template Mode, OpenFold3 0.4.2 or later: protein chains "
            "only, the best-matching chain of each file, the alignment made by OpenFold3), which "
            "the ColabFold MSA server does not overwrite. alignment: an A3M self-alignment that "
            "points to the template CIF, the way earlier versions did it; the server overwrites "
            "it, so with the server on (the default) the run has no template, which "
            "results['openfold'] or results['prediction'] reports under 'templates' (use "
            "--openfold-no-msa-server to keep it)."
        ),
    )


def check_openfold_templates(value) -> str:
    """The ``template_mode`` argument for ``value``: ``"structure"`` or ``"alignment"``.

    Raises:
        ValueError: ``value`` is neither.
    """
    if value in OPENFOLD_TEMPLATE_CHOICES:
        return value
    raise ValueError(
        f"openfold_templates must be one of {OPENFOLD_TEMPLATE_CHOICES}, got {value!r}"
    )


def openfold_template_kwargs(value) -> dict:
    """The keyword argument for an OpenFold3 run function, only when it is not the default.

    The default is left out so that a function that predates the option, or a test double that
    replaces it, is called exactly as before.
    """
    resolved = check_openfold_templates(value)
    return {} if resolved == DEFAULT_OPENFOLD_TEMPLATES else {"template_mode": resolved}


#: Values of ``--openfold-cyclic``; the first is the default.
OPENFOLD_CYCLIC_CHOICES = ("auto", "on", "off")


def add_openfold_cyclic_arg(parser) -> None:
    """Add ``--openfold-cyclic {auto,on,off}`` to an argparse parser or group.

    Whether the binder chain of the OpenFold3 query gets ``"cyclic": true``. The default,
    ``auto``, writes it for a binder with a head-to-tail bond and standard residues only when the
    installed OpenFold3 can read it (0.4.5 or later); with modified residues it leaves the binder
    linear (see ``decide_binder_cyclic``).
    """
    parser.add_argument(
        "--openfold-cyclic",
        choices=OPENFOLD_CYCLIC_CHOICES,
        default="auto",
        help=(
            "Whether the binder chain of the OpenFold3 query gets 'cyclic: true' "
            "(OpenFold3 >= 0.4.5). auto (default): when the binder has a head-to-tail bond, "
            "consists of standard residues only and the installed OpenFold3 is new enough; on: "
            "always; off: never. OpenFold3 uses the flag only to wrap the relative positions of "
            "the chain: it does not enforce the closure bond, documents the flag only in an "
            "example query, and has published no accuracy benchmark for cyclic peptides. It "
            "builds the wrap from the token count of the chain and gives every atom of a "
            "modified residue its own token: for 1CWA (D-amino acid, N-methylated residues; one "
            "complex, three seeds) the flag lowered ipTM from 0.91-0.92 to 0.78-0.81 and raised "
            "the binder C-alpha RMSD from 0.5-0.7 A to 3.0-4.8 A, so auto leaves such a binder "
            "linear (on forces the flag); for SFTI-1 (standard residues; one seed) the flag "
            "closed the ring (C-N 7.40 A without it, 1.38 A with it). Disulfide, lactam and "
            "staple closures cannot be given to OpenFold3 and are not written."
        ),
    )


def check_openfold_cyclic(value) -> bool | str:
    """The ``binder_cyclic`` argument for ``value``: ``"auto"``, ``True`` or ``False``.

    Takes the command-line choices (``"auto"``, ``"on"``, ``"off"``) and the API values
    (``True``, ``False``, ``"auto"``).

    Raises:
        ValueError: ``value`` is none of these.
    """
    if value is True or value == "on":
        return True
    if value is False or value == "off":
        return False
    if value == "auto":
        return "auto"
    raise ValueError(
        f"openfold_cyclic must be one of {OPENFOLD_CYCLIC_CHOICES}, True or False, got {value!r}"
    )


def openfold_cyclic_kwargs(value) -> dict:
    """The keyword argument for an OpenFold3 run function, only when it is not ``"auto"``.

    The default is left out so that a function that predates the option, or a test double
    that replaces it, is called exactly as before.
    """
    resolved = check_openfold_cyclic(value)
    return {} if resolved == "auto" else {"binder_cyclic": resolved}


def merge_reason(target: dict, extra: dict, label: str) -> None:
    """Move ``extra["reason"]`` into ``target["reason"]`` as ``"<label>: <reason>"``.

    Metric dicts merged into one flat namespace (OpenFold, then the EvoBind
    metrics that reuse its output) each carry an optional ``reason``; a plain
    ``dict.update`` would let the last one erase the diagnosis of the first.
    Reasons are joined with ``"; "``, and nothing is added when ``extra`` has none.
    """
    reason = extra.pop("reason", None)
    if reason:
        target["reason"] = "; ".join(filter(None, [target.get("reason"), f"{label}: {reason}"]))


#: Values of ``--on-unmappable-residue``; the first is the default.
ON_UNMAPPABLE_RESIDUE_CHOICES = ("error", "x")


def add_on_unmappable_residue_arg(parser) -> None:
    """Add ``--on-unmappable-residue {error,x}`` to an argparse parser or group.

    What OpenFold3 does with a residue it cannot take. The default stops the run before
    anything is written or started, so no model time is spent on a query that would fail.
    """
    parser.add_argument(
        "--on-unmappable-residue",
        choices=ON_UNMAPPABLE_RESIDUE_CHOICES,
        default="error",
        help=(
            "A residue that OpenFold3 cannot take (not a standard, D-, modified or "
            "protonation-variant amino acid) stops the run before the model starts "
            "(default: %(default)s). 'x' sends an X in its place and logs a warning."
        ),
    )


def on_unmappable_residue_kwargs(value: str) -> dict:
    """The keyword argument for an OpenFold3 run function, only when it is not the default.

    The default is left out so that a function that predates the option, or a test double
    that replaces it, is called exactly as before.
    """
    return {} if value == "error" else {"on_unmappable_residue": value}


def check_on_unmappable_residue(value: str) -> str:
    """Return ``value`` when it is one of ``ON_UNMAPPABLE_RESIDUE_CHOICES``, else raise.

    Raises:
        ValueError: naming the accepted values.
    """
    if value not in ON_UNMAPPABLE_RESIDUE_CHOICES:
        raise ValueError(
            f"on_unmappable_residue must be one of {ON_UNMAPPABLE_RESIDUE_CHOICES}, got {value!r}"
        )
    return value


_SMALL_MOLECULES_CHOICES = ("auto", "none")


def small_molecules_arg(value: str):
    """Parse ``--small-molecules``: ``"auto"``, or ``"none"`` (returned as ``None``).

    Any other string is rejected. ``RelaxationConfig.small_molecules`` iterates
    a string it does not recognise character by character and treats each
    character as a SMILES, so a typo such as ``aut`` would register methane
    and other fragments as "small molecules" instead of failing. Explicit
    SMILES lists are a Python-API feature (``RelaxationConfig(small_molecules=[...])``)
    and are not accepted on the command line.

    Use as ``type=small_molecules_arg`` in an argparse argument; the value is
    matched case-insensitively.

    Raises:
        argparse.ArgumentTypeError: for anything but ``auto`` / ``none``.
    """
    choice = str(value).strip().lower()
    if choice not in _SMALL_MOLECULES_CHOICES:
        raise argparse.ArgumentTypeError(
            f"invalid value {value!r}: expected one of {', '.join(_SMALL_MOLECULES_CHOICES)}"
        )
    return None if choice == "none" else choice


# ---------------------------------------------------------------------------
# --config: option defaults from a TOML file
# ---------------------------------------------------------------------------

_CONFIG_HELP = (
    "TOML file that supplies option defaults. Keys are long option names "
    "without the leading dashes (md-duration-ps or md_duration_ps). Flags given "
    "on the command line override the file."
)

# argparse action classes that cannot take a value from a file.
_UNSETTABLE_ACTIONS = (
    argparse._HelpAction,
    argparse._VersionAction,
    argparse._CountAction,
    argparse._AppendAction,
    argparse._AppendConstAction,
)


def add_config_arg(parser) -> None:
    """Add ``--config PATH`` to an argparse parser; use ``parse_args_with_config`` to read it."""
    parser.add_argument("--config", type=Path, default=None, metavar="PATH", help=_CONFIG_HELP)


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None):
    """Parse ``argv`` (default ``sys.argv[1:]``); a ``--config`` TOML file supplies defaults.

    The file is flat: each key is the long name of an option of ``parser``,
    with dashes or underscores (``md-duration-ps`` or ``md_duration_ps``). An
    option's alias spelling works too (``binder-chain``). Values are
    converted exactly as the same text on the command line would be, so ``ph = 7``,
    ``ph = "7.0"`` and ``--ph 7`` agree, and a ``choices`` list is enforced:

    * a flag such as ``skip-prep`` takes ``true`` or ``false``;
    * an option that takes several values (``energy-modes``) takes a list;
    * a comma-separated option (``metrics``) takes its string (``"interface,geometry"``);
    * paths are used as written, relative to the working directory.

    Precedence is built-in default, then the file, then the command line. A
    file value also satisfies a ``required`` option. Where the file sets one
    member of a mutually exclusive group and the command line names another,
    the command line wins.

    Ends the program through ``parser.error`` (exit code 2) for an unreadable
    or invalid file, an unknown key (the message names it and suggests the
    closest option), a value of the wrong type or outside the choices, and
    two keys that set the same option.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    finder = argparse.ArgumentParser(prog=parser.prog, add_help=False)
    finder.add_argument("--config", type=Path, default=None)
    config_path = finder.parse_known_args(argv)[0].config
    if config_path is not None:
        _apply_config_file(parser, Path(config_path), argv)
    return parser.parse_args(argv)


def _config_options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Map each long option name (underscored, without dashes) to its action."""
    options: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if isinstance(action, _UNSETTABLE_ACTIONS) or action.dest == "config":
            continue
        for spelling in action.option_strings:
            if spelling.startswith("--"):
                options[spelling[2:].replace("-", "_")] = action
    return options


def _config_value(parser, path: Path, key: str, action: argparse.Action, raw):
    """Convert one TOML value the way argparse would convert the same text."""

    def fail(message: str):
        parser.error(f"--config {path}: key {key!r}: {message}")

    if isinstance(raw, dict):
        fail("expected a value, not a table")
    if action.nargs == 0:
        if not isinstance(raw, bool):
            fail(f"expected true or false, got {raw!r}")
        return raw
    takes_many = action.nargs in ("+", "*") or (isinstance(action.nargs, int) and action.nargs > 1)
    if isinstance(raw, list) and not takes_many:
        fail("expected a single value, not a list")
    converted = []
    for item in raw if isinstance(raw, list) else [raw]:
        if isinstance(item, (bool, dict, list)):
            fail(f"invalid value {item!r}")
        try:
            value = action.type(str(item)) if action.type is not None else str(item)
        except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
            fail(f"invalid value {item!r}: {error}")
        if action.choices is not None and value not in action.choices:
            fail(f"{value!r} is not one of {', '.join(str(c) for c in action.choices)}")
        converted.append(value)
    return converted if takes_many else converted[0]


def _apply_config_file(parser: argparse.ArgumentParser, path: Path, argv: list[str]) -> None:
    """Load ``path`` and install its values as the parser's defaults."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        parser.error(f"--config: cannot read {path}: {error.strerror or error}")
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        parser.error(f"--config {path}: not valid TOML: {error}")

    options = _config_options(parser)
    values: dict[str, tuple[str, object]] = {}
    for key, raw in data.items():
        action = options.get(key.replace("-", "_"))
        if action is None:
            close = difflib.get_close_matches(key.replace("-", "_"), options, n=1)
            hint = f"; did you mean {close[0].replace('_', '-')!r}?" if close else ""
            parser.error(f"--config {path}: unknown key {key!r}{hint}")
        if action.dest in values:
            parser.error(
                f"--config {path}: keys {values[action.dest][0]!r} and {key!r} set the same option"
            )
        values[action.dest] = (key, _config_value(parser, path, key, action, raw))

    # A command-line flag beats a file value in the same mutually exclusive group.
    for group in parser._mutually_exclusive_groups:
        named_on_command_line = [
            action
            for action in group._group_actions
            if any(token.split("=", 1)[0] in action.option_strings for token in argv)
        ]
        if named_on_command_line:
            for action in group._group_actions:
                if action not in named_on_command_line:
                    values.pop(action.dest, None)

    for action in parser._actions:
        if action.dest in values:
            action.required = False  # the file provides it
    parser.set_defaults(**{dest: value for dest, (_, value) in values.items()})
