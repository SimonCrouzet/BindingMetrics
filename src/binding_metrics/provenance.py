"""Provenance for result files: which code, seed and machine produced them.

``collect_provenance`` returns a small JSON-serialisable dict that the
pipeline attaches to its results (``results["provenance"]``), so a result file
can be tied back to the code and settings that produced it.

Schema (``schema_version`` 1)::

    {
        "schema_version": 1,            # int, bumped when a key changes meaning
        "package_version": "0.1.0",     # installed distribution version
        "git_sha": "eff961f...",        # HEAD of the source checkout, else None
        "python": "3.12.3",
        "os": "Linux-6.18-x86_64-...",  # platform.platform()
        "openmm_version": "8.2",        # None when OpenMM is not importable
        "platform": "CUDA",             # compute platform reported by the caller, else None
        "seed": 1,                      # random seed of the run; None = fresh randomness
    }

Two optional keys describe a run that used OpenFold3. They are absent from a block that
was not asked for them, and adding them needs no schema bump:

* ``openfold3_version``: the installed ``openfold3`` distribution version (None when it is
  not installed or cannot be read), added by ``collect_provenance(openfold3=True)``. It
  describes the installation that the run would use, so a pipeline asks for it only when
  it starts OpenFold3 itself and not when it reads the output of an earlier run.
* ``openfold3_checkpoint``: the file name of the checkpoint that produced the prediction
  (``inference_ckpt_name`` in the record extras of the ``of3`` adapter). The pipeline adds
  it after it has read the prediction; nothing here can know it earlier.

Collection is best effort: a field that cannot be determined is ``None`` and
nothing here raises.
"""

from __future__ import annotations

import functools
import logging
import platform as _platform
import subprocess
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

#: Version of the provenance block itself. Bump when a key is renamed, removed
#: or changes meaning; adding a key does not need a bump.
SCHEMA_VERSION = 1

_DISTRIBUTION_NAME = "binding-metrics"
_PACKAGE_DIR = Path(__file__).resolve().parent
_GIT_TIMEOUT_S = 5


@functools.lru_cache(maxsize=1)
def _package_version() -> Optional[str]:
    """Installed distribution version, else the in-source ``__version__``."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(_DISTRIBUTION_NAME)
        except PackageNotFoundError:
            pass
        from binding_metrics import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - provenance must never break a run; None is the record
        logger.debug("package version unavailable", exc_info=True)
        return None


@functools.lru_cache(maxsize=1)
def _git_sha() -> Optional[str]:
    """HEAD commit of the checkout this package was imported from.

    Returns ``None`` when git is missing, the package is not in a git
    checkout, or the checkout that contains it is some other project (for
    example an installed copy under a ``.venv`` that lives inside the user's
    own repository): the sha must describe this package's source tree.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            return None
        lines = proc.stdout.split("\n")
        toplevel, sha = Path(lines[0].strip()), lines[1].strip()
        if (toplevel / "src" / _PACKAGE_DIR.name).resolve() != _PACKAGE_DIR:
            return None
        return sha or None
    except Exception:  # noqa: BLE001 - git missing, timeout or odd output; None is the record
        logger.debug("git sha unavailable", exc_info=True)
        return None


def _openmm_version() -> Optional[str]:
    try:
        import openmm

        return str(openmm.__version__)
    except Exception:  # noqa: BLE001 - OpenMM missing or broken; None is the record
        logger.debug("OpenMM version unavailable", exc_info=True)
        return None


def conda_python_command(conda_env: Optional[str]) -> Optional[list[str]]:
    """The command that starts the interpreter of the conda environment ``conda_env``.

    Returns ``["conda", "run", "-n", conda_env, "python"]``, or None for None or an empty
    name, which means the current interpreter (the meaning of ``--openfold-conda-env ""``).
    """
    if not conda_env:
        return None
    return ["conda", "run", "-n", conda_env, "python"]


def openfold3_version(python_cmd: Optional[Sequence[str]] = None) -> Optional[str]:
    """The installed ``openfold3`` version, or None when it is not installed or unreadable.

    Args:
        python_cmd: Command that starts the interpreter to ask (see ``conda_python_command``);
            None asks the current interpreter without starting a process.

    Never raises.
    """
    try:
        from binding_metrics.metrics import _openfold_run

        return _openfold_run.installed_openfold3_version(python_cmd)
    except Exception:  # noqa: BLE001 - provenance must never break a run; None is the record
        logger.debug("openfold3 version unavailable", exc_info=True)
        return None


def collect_provenance(
    seed: Optional[int] = None,
    platform: Optional[str] = None,
    *,
    openfold3: bool = False,
    openfold3_python_cmd: Optional[Sequence[str]] = None,
) -> dict:
    """Describe the code and environment that produce a result.

    Args:
        seed: Random seed used by the run (``None`` when the run used fresh
            randomness or the seed is unknown).
        platform: Compute platform the run used, for example ``"CUDA"``, when
            the caller knows it; ``None`` otherwise.
        openfold3: Add the key ``openfold3_version`` (keyword-only). Asking starts a process
            when ``openfold3_python_cmd`` names an interpreter, which takes seconds, so it is
            left off by default.
        openfold3_python_cmd: Command that starts the interpreter that has OpenFold3, for
            example ``["conda", "run", "-n", "openfold3", "python"]``; None asks the
            current interpreter. Used only with ``openfold3``.

    Returns:
        Dict with the keys listed in the module docstring. Never raises; keys
        that cannot be determined are ``None``.
    """
    try:
        os_name: Optional[str] = _platform.platform()
    except Exception:  # noqa: BLE001 - platform probing failed; None is the record
        logger.debug("OS name unavailable", exc_info=True)
        os_name = None
    block = {
        "schema_version": SCHEMA_VERSION,
        "package_version": _package_version(),
        "git_sha": _git_sha(),
        "python": _platform.python_version(),
        "os": os_name,
        "openmm_version": _openmm_version(),
        "platform": platform,
        "seed": seed,
    }
    if openfold3:
        block["openfold3_version"] = openfold3_version(openfold3_python_cmd)
    return block
