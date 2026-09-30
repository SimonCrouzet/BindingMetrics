"""
binding-metrics-check-env — verify that all required runtime dependencies are working.

Checks are run in sequence and results are printed in a pytest-like style with
coloured pass/fail indicators and detailed diagnostics on failure.

Exit code is 0 if all checks pass, 1 otherwise.
"""

import subprocess
import sys

from binding_metrics.utils import configure_logging

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def _fail(title: str, cause: str, steps: list[str]) -> None:
    print(f"  {RED}✘{RESET}  {BOLD}{title}{RESET}")
    print()
    print(f"     {BOLD}What happened:{RESET}")
    print(f"       {cause}")
    print()
    print(f"     {BOLD}How to fix it:{RESET}")
    for step in steps:
        if step == "":
            print()
        else:
            print(f"       {step}")
    print()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

_OPENFOLD_CONDA_ENV = "openfold3"


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET}  {msg}")


def _report_openfold_readiness(python_cmd: list[str], where: str) -> bool:
    """Say which openfold3 version is installed and whether its default checkpoint is on disk.

    Importing ``openfold3`` says little: since 0.5.0 it refuses to run without the OpenBind-0
    checkpoint (``of3-ob-2025-06-30-174k.pt``), and Preview2 weights of a 0.4.x install do not
    load into it. ``python_cmd`` starts the interpreter of the installation, for the version;
    the checkpoint folder is read from ``$OPENFOLD_CACHE`` (default ``~/.openfold3``), which a
    ``conda run`` of the same user shares. Returns False when openfold3 >= 0.5 has no default
    checkpoint to load, since every run would stop there.
    """
    from binding_metrics.metrics import _openfold_run as run

    version = run.installed_openfold3_version(python_cmd)
    checkpoint_dir = run._openfold_checkpoint_dir()
    default_file = run._OPENFOLD_DEFAULT_CHECKPOINT_FILE
    found = checkpoint_dir / default_file
    preview = [f for f in run._OPENFOLD_PREVIEW_CHECKPOINT_FILES if (checkpoint_dir / f).is_file()]
    version_tuple = run._version_tuple(version) if version else ()
    label = f"openfold3 {version}" if version else "openfold3 (version not readable)"
    _ok(f"{label} available in {where}")

    if not version_tuple:
        _warn("The installed version could not be read, so the checkpoint cannot be judged.")
        return True

    if version_tuple < (0, 5, 0):
        _warn(
            f"openfold3 {version} predates 0.5.0, the release this toolkit is written against "
            "(default checkpoint OpenBind-0, cyclic peptides from 0.4.5). Upgrade with "
            "'pip install \"openfold3>=0.5.0,<0.6\"' and 'setup_openfold --non-interactive'."
        )
        return True

    if found.is_file():
        _ok(f"default checkpoint {default_file} found in {checkpoint_dir}")
        return True

    cause = (
        f"openfold3 {version} loads {default_file} by default and it is not in {checkpoint_dir}."
    )
    if preview:
        cause += (
            f" Only Preview weights are there ({', '.join(preview)}); they do not load into "
            "openfold3 >= 0.5."
        )
    _fail(
        title=f"Default checkpoint {default_file} not found",
        cause=cause + " Every run would stop with 'cowardly refusing to perform inference'.",
        steps=[
            f"{BOLD}Download it (about 2.3 GB):{RESET}",
            "          setup_openfold --non-interactive   # inside the OpenFold3 environment",
            "",
            "Or point a run at a checkpoint file with the inference_ckpt_path argument.",
        ],
    )
    return False


def _check_openfold() -> bool:
    print(f"\n{BOLD}[ OpenFold3 ]{RESET}")

    def _run_openfold_available(python: str) -> bool:
        """Return True if run_openfold binary is importable / on PATH."""
        r = subprocess.run(
            [
                python,
                "-c",
                "import shutil, sys; sys.exit(0 if shutil.which('run_openfold') else 1)",
            ],
            capture_output=True,
        )
        return r.returncode == 0

    def _of3_importable(python: str) -> bool:
        r = subprocess.run(
            [python, "-c", "import openfold3"],
            capture_output=True,
        )
        return r.returncode == 0

    # 1. Check current environment first
    if _of3_importable(sys.executable) or _run_openfold_available(sys.executable):
        return _report_openfold_readiness([sys.executable], "current environment")

    # 2. Check dedicated conda env — look for run_openfold binary directly
    import shutil

    conda = shutil.which("conda") or "conda"
    result = subprocess.run(
        [
            conda,
            "run",
            "-n",
            _OPENFOLD_CONDA_ENV,
            "python",
            "-c",
            "import openfold3; import shutil; print(shutil.which('run_openfold') or 'ok')",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode == 0:
        ready = _report_openfold_readiness(
            [conda, "run", "-n", _OPENFOLD_CONDA_ENV, "python"],
            f"conda env '{_OPENFOLD_CONDA_ENV}'",
        )
        print(
            f"     {DIM}(used automatically — default for --openfold-conda-env){RESET}\n"
            f"     {DIM}Run integration tests with:{RESET}\n"
            f"     {DIM}  conda run -n {_OPENFOLD_CONDA_ENV} pytest "
            f"$(conda run -n {_OPENFOLD_CONDA_ENV} python -c "
            f'"import openfold3; print(openfold3.__path__[0])")/tests/{RESET}'
        )
        return ready

    # 3. Check whether the conda env exists at all
    env_check = subprocess.run(
        [conda, "env", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    env_exists = _OPENFOLD_CONDA_ENV in (env_check.stdout + env_check.stderr)

    if env_exists:
        _fail(
            title=f"Conda env '{_OPENFOLD_CONDA_ENV}' exists but openfold3 is not importable",
            cause="The env was found but 'import openfold3' failed — the package may not be "
            "installed or its dependencies are broken.",
            steps=[
                f"{BOLD}Step 1{RESET} — Check what's installed:",
                f"          conda run -n {_OPENFOLD_CONDA_ENV} pip show openfold3",
                "",
                f"{BOLD}Step 2{RESET} — Reinstall if needed:",
                f"          conda activate {_OPENFOLD_CONDA_ENV}",
                "          pip install openfold3",
                "          setup_openfold --non-interactive   # downloads model weights",
            ],
        )
    else:
        _fail(
            title=(
                f"OpenFold3 not found (checked current env and conda env '{_OPENFOLD_CONDA_ENV}')"
            ),
            cause="OpenFold3 is an optional dependency used for confidence scoring. "
            "Binding metrics will still run without it.",
            steps=[
                f"{BOLD}To install OpenFold3:{RESET}",
                f"  conda create -n {_OPENFOLD_CONDA_ENV} python=3.10   # 3.10 to 3.13 work",
                f"  conda activate {_OPENFOLD_CONDA_ENV}",
                "  pip install openfold3",
                "  setup_openfold --non-interactive   # downloads model weights",
                "",
                f"{BOLD}Then pass to binding-metrics-run:{RESET}",
                f"  --openfold-conda-env {_OPENFOLD_CONDA_ENV}",
                "",
                f"{DIM}(OpenFold is optional — all other metrics work without it){RESET}",
            ],
        )
    return False


def _check_mdtraj() -> bool:
    print(f"\n{BOLD}[ MDTraj ]{RESET}")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mdtraj as md, numpy as np; "
            "xyz = np.random.randn(10, 5, 3).astype(np.float32); "
            "top = md.Topology(); chain = top.add_chain(); "
            "res = top.add_residue('ALA', chain); "
            "[top.add_atom(n, md.element.carbon, res) for n in ['N','CA','C','O','CB']]; "
            "traj = md.Trajectory(xyz, top); "
            "md.rmsd(traj, traj, 0); "
            "md.compute_distances(traj, [[0,1]]); "
            "md.shrake_rupley(traj); "
            "print(md._version)",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        _ok(f"mdtraj {version} — trajectory ops OK")
        return True

    stderr = result.stderr
    if "numpy" in stderr.lower():
        _fail(
            title="mdtraj failed — NumPy version mismatch",
            cause="mdtraj was compiled against a different NumPy major version "
            "than what is currently installed.",
            steps=[
                f"{BOLD}Step 1{RESET} — Check installed versions:",
                '          python -c "import numpy; print(numpy.__version__)"',
                "          pip show mdtraj",
                "",
                f"{BOLD}Step 2{RESET} — Fix the mismatch:",
                "          pip install --force-reinstall mdtraj",
            ],
        )
    else:
        _fail(
            title="mdtraj import or computation failed",
            cause="mdtraj may not be installed or has broken dependencies.",
            steps=[
                f"{BOLD}Step 1{RESET} — Check the raw error:",
                '          python -c "import mdtraj"',
                "",
                f"{BOLD}Step 2{RESET} — Reinstall if needed:",
                "          pip install mdtraj",
            ],
        )
    return False


def _check_openmm() -> bool:
    print(f"\n{BOLD}[ OpenMM ]{RESET}")
    print(f"  {DIM}Running openmm.testInstallation…{RESET}")

    result = subprocess.run(
        [sys.executable, "-m", "openmm.testInstallation"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout + result.stderr

    version_line = next(
        (line for line in output.splitlines() if line.startswith("OpenMM Version:")),
        None,
    )
    version = version_line.split()[-1] if version_line else "unknown"

    if "CUDA - Successfully computed forces" in output:
        _ok(f"CUDA platform active  (OpenMM {version})")
        return True

    if "CUDA" in output and result.returncode != 0:
        _fail(
            title=f"OpenMM {version}: CUDA platform found but force computation failed",
            cause="The CUDA libraries are visible but something failed at runtime — "
            "most likely a driver / CUDA toolkit version mismatch.",
            steps=[
                f"{BOLD}Step 1{RESET} — Check your driver and the CUDA version it supports:",
                "          nvidia-smi",
                "        The 'CUDA Version' shown top-right is the maximum "
                "supported by your driver.",
                "        It must be ≥ the CUDA version OpenMM was compiled against.",
                "",
                f"{BOLD}Step 2{RESET} — Read the full error from OpenMM:",
                "          python -m openmm.testInstallation",
                f"        {YELLOW}PTX version error{RESET} → your driver is too old, update it.",
                f"        {YELLOW}library not found{RESET} → CUDA runtime "
                "missing or not on LD_LIBRARY_PATH.",
                "",
                f"{BOLD}Step 3{RESET} — Update your NVIDIA driver if needed:",
                "        → Linux:        install the latest nvidia-driver package for your distro.",
                "        → Windows/WSL2: update the NVIDIA driver on the Windows host.",
            ],
        )
        return False

    if "CPU - Successfully computed forces" in output:
        _fail(
            title=f"OpenMM {version}: works on CPU only — no GPU detected",
            cause="OpenMM cannot see any GPU. Simulations will run on CPU and be very slow.",
            steps=[
                f"{BOLD}Step 1{RESET} — Check that the NVIDIA driver is loaded:",
                "          nvidia-smi",
                "        If this command fails, your driver is missing or not running.",
                "        → Linux:        install the nvidia-driver package for your distro.",
                "        → Windows/WSL2: install the NVIDIA driver on the "
                "Windows host (not inside WSL).",
                "",
                f"{BOLD}Step 2{RESET} — Check that CUDA runtime libraries are on the system:",
                "          ldconfig -p | grep libcuda",
                "        If nothing appears, the CUDA runtime is missing.",
                "        → Install the cuda-runtime package, or set "
                "LD_LIBRARY_PATH to its location.",
                "",
                f"{BOLD}Step 3{RESET} — If you are running inside a container:",
                "        Make sure it was started with GPU access and that the",
                "        NVIDIA Container Toolkit is installed on the host:",
                "          docker run --gpus all ...",
                "          # to install the toolkit on Ubuntu:",
                "          sudo apt-get install nvidia-container-toolkit",
                "          sudo nvidia-ctk runtime configure --runtime=docker",
                "          sudo systemctl restart docker",
            ],
        )
        return False

    _fail(
        title="OpenMM failed to run at all",
        cause="OpenMM may not be installed, or the Python environment is broken.",
        steps=[
            f"{BOLD}Step 1{RESET} — Run the test manually to see the raw error:",
            "          python -m openmm.testInstallation",
            "",
            f"{BOLD}Step 2{RESET} — Check that OpenMM is installed:",
            "          conda list openmm",
            "        If missing, reinstall:",
            "          conda install -c conda-forge openmm",
        ],
    )
    return False


# ---------------------------------------------------------------------------
# Registry — add new dependency checks here
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, object]] = [
    ("OpenMM", _check_openmm),
    ("MDTraj", _check_mdtraj),
    ("OpenFold3", _check_openfold),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    print(f"\n{BOLD}{'=' * 56}{RESET}")
    print(f"{BOLD}  BindingMetrics — environment check{RESET}")
    print(f"{BOLD}{'=' * 56}{RESET}")

    passed, failed = 0, 0
    for _name, check in CHECKS:
        if check():
            passed += 1
        else:
            failed += 1

    print(f"{BOLD}{'=' * 56}{RESET}")
    if failed == 0:
        print(f"{GREEN}{BOLD}  All {passed} check(s) passed — good, ready to go.{RESET}")
    else:
        print(f"{RED}{BOLD}  {passed} passed, {failed} failed.{RESET}")
        print("  Please fix the issues above before running BindingMetrics.")
    print(f"{BOLD}{'=' * 56}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
