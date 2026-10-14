"""The MSA policy of a prediction request: a receptor MSA that is given, and no MSA server.

Every runner sends the sequences to a public MSA server unless it is told not to
(``use_msa_server=True`` is the default of all four). The policy ``forbid-server`` keeps the
sequences on the machine, and the receptor's alignment is given as a file. This module holds the
pieces the runners share:

* ``check_msa_request`` applies the rules to the arguments of a runner's ``make_request`` and
  returns the pieces to add to the request. A runner calls it in ``make_request`` and passes on
  what it returns.
* ``check_msa_of_request`` applies the same rules to a request that was built by hand, for
  ``prepare`` and ``run``.
* ``assert_no_server_flag`` looks at the command line a runner has built and refuses it when it
  holds a flag that makes the model call a server. It is the last check before a process starts.

The rules under ``msa_policy="forbid-server"``:

1. ``use_msa_server`` must be false: the sequences would be sent to the server otherwise.
2. The receptor needs an MSA source. ``receptor_msa`` is a path to the receptor's alignment, or
   the word ``"single"`` for a receptor that is to be predicted from its own sequence. None is
   refused: some models would search the server for a receptor without a source, and others
   would run it without an alignment.
3. The runner must be able to take what was asked: a runner that cannot take an alignment file
   (ColabFold) accepts ``"single"`` only.

The binder (a short peptide) is single-sequence under the policy: no alignment is offered for
it, and every runner is told so in its own way.

What goes into the key of the request, so that a request without these fields keeps the key it
had: ``options["msa_policy"]`` only when the policy is not ``"default"``,
``options["receptor_msa"] = "single"`` only when that was asked, and
``extra_files["receptor_msa"]`` only when a file is given. The store hashes an extra file by its
content (``PredictionRequest.content_hashes``), so the same alignment at another path is the same
request and another alignment is another one. ``use_msa_server`` stays an option of the request
as before.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Optional, Sequence, Union

from binding_metrics.capabilities import DEFAULT_MSA_POLICY, MSA_POLICIES

if TYPE_CHECKING:
    from binding_metrics.predictors.store import PredictionRequest

__all__ = [
    "DEFAULT_MSA_POLICY",
    "FORBID_SERVER",
    "MSA_POLICIES",
    "RECEPTOR_MSA_ROLE",
    "SINGLE_SEQUENCE",
    "MsaFragments",
    "assert_no_server_flag",
    "check_msa_of_request",
    "check_msa_request",
]

#: The policy that forbids the server.
FORBID_SERVER = "forbid-server"

#: ``receptor_msa`` value for a receptor predicted from its own sequence, with no alignment.
SINGLE_SEQUENCE = "single"

#: The role of the receptor's alignment in ``PredictionRequest.extra_files``, and the name of its
#: option in ``PredictionRequest.options`` when the receptor is single-sequence.
RECEPTOR_MSA_ROLE = "receptor_msa"

#: Models whose default, with the flag left out, is to call the server: the flag must be on the
#: command line (ColabFold's ``--msa-mode`` defaults to ``mmseqs2_uniref_env``).
_SERVER_BY_DEFAULT = {"af2": "--msa_mode"}

#: The options that decide whether a model calls an MSA server, spelled with underscores.
_WATCHED_FLAGS = ("--use_msa_server", "--use_msa", "--msa_mode")

#: Shortest option name (dashes included) read as the abbreviation of a watched flag.
_SHORTEST_ABBREVIATION = 5

_TRUE_WORDS = frozenset({"true", "1", "yes", "on", "t", "y"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off", "f", "n"})


class MsaFragments(NamedTuple):
    """What the MSA arguments add to a request: merge both into the ones the runner builds.

    Attributes:
        options: ``{"msa_policy": ...}`` when the policy is not the default and
            ``{"receptor_msa": "single"}`` when that was asked; empty for a request that asks for
            nothing, so its key does not change.
        extra_files: ``{"receptor_msa": path}`` when an alignment file is given; empty otherwise.
    """

    options: dict[str, Any]
    extra_files: dict[str, Path]


def _file_of(receptor_msa: Union[str, os.PathLike]) -> Path:
    path = Path(receptor_msa).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"receptor_msa is not a file: {path}")
    return path


def _is_single(receptor_msa: Any) -> bool:
    """True for the word ``"single"``; a ``Path`` named that is a file, whatever its name."""
    return isinstance(receptor_msa, str) and receptor_msa == SINGLE_SEQUENCE


def check_msa_request(
    runner_name: str,
    *,
    use_msa_server: bool,
    msa_policy: str = DEFAULT_MSA_POLICY,
    receptor_msa: Optional[Union[str, os.PathLike]] = None,
    supports_given_msa: bool,
    supports_single: bool,
    unsupported_reason: str = "",
) -> MsaFragments:
    """Apply the MSA rules to the arguments of ``make_request`` and return the request pieces.

    Args:
        runner_name: The model, for the messages (``"boltz2"``).
        use_msa_server: Whether the run is to call an MSA server.
        msa_policy: One of ``MSA_POLICIES``.
        receptor_msa: None (no source), ``"single"`` (the receptor is predicted from its own
            sequence) or the path of the receptor's alignment. A ``str`` other than ``"single"``
            is a path; pass a ``Path`` for a file that is called ``single``.
        supports_given_msa: True when the runner can give an alignment file to its model.
        supports_single: True when the runner can run a receptor without an alignment.
        unsupported_reason: Why the runner cannot take what was asked, for the message of rule 3
            (what the model offers instead); empty gives a general sentence.

    Returns:
        The ``MsaFragments`` to merge into the options and extra files of the request.

    Raises:
        ValueError: ``msa_policy`` is unknown; under ``forbid-server``, ``use_msa_server`` is
            true or ``receptor_msa`` is None; or the runner cannot take the alignment or the
            single-sequence receptor that was asked for (under any policy).
        FileNotFoundError: ``receptor_msa`` is a path that is not a file.
        TypeError: ``receptor_msa`` is neither None, a string nor a path.
    """
    if msa_policy not in MSA_POLICIES:
        raise ValueError(f"msa_policy must be one of {MSA_POLICIES}, got {msa_policy!r}")
    if receptor_msa is not None and not isinstance(receptor_msa, (str, os.PathLike)):
        raise TypeError(
            f"receptor_msa must be None, {SINGLE_SEQUENCE!r} or a path, got "
            f"{type(receptor_msa).__name__}"
        )
    forbidding = msa_policy == FORBID_SERVER
    if forbidding and use_msa_server:
        raise ValueError(
            f"{runner_name}: the sequences would be sent to an MSA server and the policy "
            f"forbids it; pass use_msa_server=False and receptor_msa=PATH (the alignment of the "
            f"receptor) or receptor_msa={SINGLE_SEQUENCE!r}"
        )
    if forbidding and receptor_msa is None:
        raise ValueError(
            f"{runner_name}: msa_policy={FORBID_SERVER!r} needs a source for the MSA of the "
            f"receptor: pass receptor_msa=PATH, or receptor_msa={SINGLE_SEQUENCE!r} for a "
            "receptor predicted from its own sequence (the server would otherwise be asked "
            "for it, or the model left without one)"
        )

    options: dict[str, Any] = {}
    extra_files: dict[str, Path] = {}
    if forbidding:
        options["msa_policy"] = msa_policy
    if receptor_msa is None:
        return MsaFragments(options, extra_files)

    because = f" ({unsupported_reason})" if unsupported_reason else ""
    if _is_single(receptor_msa):
        if not supports_single:
            raise ValueError(
                f"{runner_name} cannot run a receptor without an alignment{because}, so "
                f"receptor_msa={SINGLE_SEQUENCE!r} is not available"
            )
        options[RECEPTOR_MSA_ROLE] = SINGLE_SEQUENCE
    else:
        if not supports_given_msa:
            only = f"; it takes receptor_msa={SINGLE_SEQUENCE!r} only" if supports_single else ""
            raise ValueError(
                f"{runner_name} cannot be given the alignment of the receptor{because}{only}"
            )
        extra_files[RECEPTOR_MSA_ROLE] = _file_of(receptor_msa)
    return MsaFragments(options, extra_files)


def check_msa_of_request(
    runner_name: str,
    request: "PredictionRequest",
    *,
    supports_given_msa: bool,
    supports_single: bool,
    unsupported_reason: str = "",
) -> None:
    """Apply the rules of ``check_msa_request`` to a request that was already built.

    For ``prepare`` and ``run``: a request made by hand, or by an older version, skips
    ``make_request``, and a runner must not start a process on a request that breaks the policy
    it states. Only the facts in the request are read: ``options["msa_policy"]``,
    ``options["use_msa_server"]`` (true when absent, the default of every runner),
    ``options["receptor_msa"]`` and ``extra_files["receptor_msa"]``.

    Raises:
        ValueError: As ``check_msa_request``.
        FileNotFoundError: The alignment file of the request is gone.
    """
    options: Mapping[str, Any] = request.options
    given = request.extra_files.get(RECEPTOR_MSA_ROLE)
    asked: Optional[Union[str, os.PathLike]] = None
    if given is not None:
        asked = given
    elif options.get(RECEPTOR_MSA_ROLE) == SINGLE_SEQUENCE:
        asked = SINGLE_SEQUENCE
    check_msa_request(
        runner_name,
        use_msa_server=bool(options.get("use_msa_server", True)),
        msa_policy=options.get("msa_policy", DEFAULT_MSA_POLICY),
        receptor_msa=asked,
        supports_given_msa=supports_given_msa,
        supports_single=supports_single,
        unsupported_reason=unsupported_reason,
    )


def _boolean(word: str) -> Optional[bool]:
    lowered = word.strip().lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    return None


def _value_of(command: Sequence[str], index: int) -> Optional[str]:
    """The value of the option at ``command[index]``: ``--name=value`` or the next word.

    None for an option without one: the next word is another option, or there is none.
    """
    name, equals, value = command[index].partition("=")
    if equals:
        return value
    if index + 1 < len(command) and not command[index + 1].startswith("--"):
        return command[index + 1]
    return None


def _spelled_with_underscores(name: str) -> str:
    """``--use-msa-server`` as ``--use_msa_server``: the tools take the two spellings alike."""
    return "--" + name[2:].replace("-", "_")


def _flags_denoted_by(name: str) -> tuple[str, ...]:
    """The watched flags that the option ``name`` can stand for, in the underscore spelling.

    The name itself when it is one of them. Otherwise the flags it abbreviates: ``argparse``
    accepts an unambiguous prefix of an option (``--msa-m`` for ``--msa-mode``), so a prefix of
    a watched flag counts as that flag, and an ambiguous prefix as all of them (the guard errs
    towards the refusal). Case is kept: option names are case sensitive.
    """
    spelled = _spelled_with_underscores(name)
    if spelled in _WATCHED_FLAGS:
        return (spelled,)
    if len(spelled) < _SHORTEST_ABBREVIATION:
        return ()
    return tuple(flag for flag in _WATCHED_FLAGS if flag.startswith(spelled))


def assert_no_server_flag(
    command: Union[str, Sequence[Union[str, os.PathLike]]],
    runner_name: str,
    *,
    chains_have_msa_paths: bool = False,
) -> None:
    """Refuse a command line that would make the model call an MSA server.

    Call it on the final command a runner is about to start (``extra_args`` included) when the
    request's policy is ``forbid-server``. It reads the flags of the four models:

    * ``--use_msa_server`` (Boltz-2, OpenFold3): on when bare or ``true``; ``false`` is fine
      (OpenFold3 is started with ``--use_msa_server=false``);
    * ``--use_msa`` (Protenix): ``true`` searches the server for every chain that has no path
      in the input JSON, which a command line cannot show, so it is refused unless
      ``chains_have_msa_paths`` says that the runner gave every chain a path;
    * ``--msa-mode`` (ColabFold): only ``single_sequence`` does not call the server
      (``mmseqs2_uniref_env`` and ``mmseqs2_uniref`` do). The ``af2`` runner must have the flag on
      the command line, because the model's default calls the server.

    How the options are read, as the tools read them:

    * The hyphen and the underscore spelling of a name are the same option
      (``--use-msa-server`` is ``--use_msa_server``; OpenFold3 takes both). Case is kept.
    * The value is ``--name=value`` or the next word. A bare ``--use_msa_server`` or one followed
      by a word that is not a boolean (an argument of the command) is on.
    * An option can be given again and the last occurrence wins, so
      ``--use_msa_server=false --use-msa-server=true`` is a server call and the reverse is not.
    * A prefix of a watched name (``--msa-m``) counts as the option, for the tools that take an
      unambiguous prefix.

    A value that is not a boolean word is refused too: the check errs towards the refusal,
    because the cost of a wrong guess is a sequence sent out.

    Args:
        command: The words of the command, or one string (split like a shell).
        runner_name: The model, for the message.
        chains_have_msa_paths: True when every protein chain of the Protenix input has an
            alignment path, so that ``--use_msa true`` reads files and calls no server.

    Raises:
        ValueError: The command holds a server flag; the message names the flag.
    """
    words = shlex.split(command) if isinstance(command, str) else [os.fspath(w) for w in command]

    # the last occurrence of each watched flag: (the name as written, its value or None)
    last: dict[str, tuple[str, Optional[str]]] = {}
    for index, word in enumerate(words):
        if not word.startswith("--"):
            continue
        name = word.partition("=")[0]
        flags = _flags_denoted_by(name)
        if not flags:
            continue
        value = _value_of(words, index)
        if value is not None and "=" not in word and _boolean(value) is None:
            if "--use_msa_server" in flags:
                value = None  # a bare flag followed by an argument of the command
        for flag in flags:
            last[flag] = (name, value)

    def refuse(name: str, value: Optional[str]) -> None:
        shown = name if value is None else f"{name} {value}"
        raise ValueError(
            f"{runner_name}: msa_policy={FORBID_SERVER!r} but the command line would make the "
            f"model use an MSA server ({shown}); nothing was started"
        )

    for flag in _WATCHED_FLAGS:
        if flag not in last:
            continue
        name, value = last[flag]
        state = None if value is None else _boolean(value)
        if flag == "--use_msa_server":
            if state is not False:
                refuse(name, value)
        elif flag == "--use_msa":
            if state is None or (state and not chains_have_msa_paths):
                refuse(name, value)
        elif value != "single_sequence":
            refuse(name, value)

    if _SERVER_BY_DEFAULT.get(runner_name) and "--msa_mode" not in last:
        raise ValueError(
            f"{runner_name}: msa_policy={FORBID_SERVER!r} but the command line has no "
            f"--msa-mode, and the default of the model calls an MSA server; nothing was started"
        )
