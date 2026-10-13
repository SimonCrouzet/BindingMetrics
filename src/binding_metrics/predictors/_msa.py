"""The MSA policy of a prediction request: a receptor MSA that is given, and no MSA server.

Every runner sends the sequences to a public MSA server unless it is told not to
(``use_msa_server=True`` is the default of all four). A caller who cannot let a sequence leave the
machine needs two things: a way to say so that no setting can undo, and a way to give the
receptor's alignment instead. This module holds the pieces the runners share, so that they do not
each decide what the policy means:

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
   refused: a receptor without a source would be searched on the server by some models, and
   silently left without an alignment by others.
3. The runner must be able to take what was asked: a runner that cannot take an alignment file
   (ColabFold) accepts ``"single"`` only.

The peptide is single-sequence under the policy: no alignment is offered for the binder (an
alignment of 4 to 12 residues is of no use), and every runner is told so in its own way.

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
_SERVER_BY_DEFAULT = {"af2": ("--msa-mode", "--msa_mode")}

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


def _value_of(command: Sequence[str], index: int, flag: str) -> tuple[Optional[str], int]:
    """The value of ``flag`` at ``command[index]``: ``--flag=value`` or the next word.

    Returns ``(value, number of words used)``; ``value`` is None for a flag without one.
    """
    word = command[index]
    if word.startswith(flag + "="):
        return word[len(flag) + 1 :], 1
    if index + 1 < len(command) and not command[index + 1].startswith("--"):
        return command[index + 1], 2
    return None, 1


def assert_no_server_flag(
    command: Union[str, Sequence[Union[str, os.PathLike]]],
    runner_name: str,
    *,
    chains_have_msa_paths: bool = False,
) -> None:
    """Refuse a command line that would make the model call an MSA server.

    Call it on the command a runner is about to start when the request's policy is
    ``forbid-server``. It reads the flags of the four models:

    * ``--use_msa_server`` (Boltz-2, OpenFold3): on when bare or ``true``; ``false`` is fine
      (OpenFold3 is started with ``--use_msa_server=false``);
    * ``--use_msa`` (Protenix): ``true`` searches the server for every chain that has no path
      in the input JSON, which a command line cannot show, so it is refused unless
      ``chains_have_msa_paths`` says that the runner gave every chain a path;
    * ``--msa-mode`` (ColabFold): only ``single_sequence`` does not call the server
      (``mmseqs2_uniref_env`` and ``mmseqs2_uniref`` do). The ``af2`` runner must have the flag on
      the command line, because the model's default calls the server.

    A flag whose value is not a boolean word is refused too: the check errs towards the refusal,
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

    def refuse(flag: str, value: Optional[str]) -> None:
        shown = flag if value is None else f"{flag} {value}"
        raise ValueError(
            f"{runner_name}: msa_policy={FORBID_SERVER!r} but the command line would make the "
            f"model use an MSA server ({shown}); nothing was started"
        )

    for index, word in enumerate(words):
        flag = word.partition("=")[0]
        if flag == "--use_msa_server":
            value, _ = _value_of(words, index, flag)
            if "=" not in word and value is not None and _boolean(value) is None:
                value = None  # a bare flag followed by an argument of the command
            if value is None or _boolean(value) is not False:
                refuse(flag, value)
        elif flag == "--use_msa":
            value, _ = _value_of(words, index, flag)
            if value is None or _boolean(value) is None or _boolean(value):
                if not (chains_have_msa_paths and value is not None and _boolean(value)):
                    refuse(flag, value)
        elif flag in ("--msa-mode", "--msa_mode"):
            value, _ = _value_of(words, index, flag)
            if value != "single_sequence":
                refuse(flag, value)

    spellings = _SERVER_BY_DEFAULT.get(runner_name)
    if spellings is not None and not any(w.partition("=")[0] in spellings for w in words):
        raise ValueError(
            f"{runner_name}: msa_policy={FORBID_SERVER!r} but the command line has no "
            f"{spellings[0]}, and the default of the model calls an MSA server; nothing was started"
        )
