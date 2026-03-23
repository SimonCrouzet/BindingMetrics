"""
binding-metrics-check-env — verify that all required runtime dependencies are working.

Checks are run in sequence and results are printed in a pytest-like style with
coloured pass/fail indicators and detailed diagnostics on failure.

Exit code is 0 if all checks pass, 1 otherwise.
"""

import subprocess
import sys

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


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


def _check_openfold() -> bool:
    print(f"\n{BOLD}[ OpenFold3 ]{RESET}")

    def _run_openfold_available(python: str) -> bool:
        """Return True if run_openfold binary is importable / on PATH."""
        r = subprocess.run(
            [python, "-c",
             "import shutil, sys; sys.exit(0 if shutil.which('run_openfold') else 1)"],
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
        _ok("openfold3 available in current environment")
        return True

    # 2. Check dedicated conda env — look for run_openfold binary directly
    import shutil
    conda = shutil.which("conda") or "conda"
    result = subprocess.run(
        [conda, "run", "-n", _OPENFOLD_CONDA_ENV,
         "python", "-c", "import openfold3; import shutil; print(shutil.which('run_openfold') or 'ok')"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        _ok(
            f"openfold3 available in conda env '{_OPENFOLD_CONDA_ENV}'\n"
            f"     {DIM}(used automatically — default for --openfold-conda-env){RESET}"
        )
        return True

    # 3. Check whether the conda env exists at all
    env_check = subprocess.run(
        [conda, "env", "list"],
        capture_output=True, text=True,
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
                "          setup_openfold   # downloads model weights",
            ],
        )
    else:
        _fail(
            title=f"OpenFold3 not found (checked current env and conda env '{_OPENFOLD_CONDA_ENV}')",
            cause="OpenFold3 is an optional dependency used for confidence scoring. "
                  "Binding metrics will still run without it.",
            steps=[
                f"{BOLD}To install OpenFold3:{RESET}",
                f"  conda create -n {_OPENFOLD_CONDA_ENV} python=3.10",
                f"  conda activate {_OPENFOLD_CONDA_ENV}",
                "  pip install openfold3",
                "  setup_openfold   # downloads model weights",
                "",
                f"{BOLD}Then pass to binding-metrics-run:{RESET}",
                f"  --openfold-conda-env {_OPENFOLD_CONDA_ENV}",
                "",
                f"{DIM}(OpenFold is optional — all other metrics work without it){RESET}",
            ],
        )
    return False


def _check_openmm() -> bool:
    print(f"\n{BOLD}[ OpenMM ]{RESET}")
    print(f"  {DIM}Running openmm.testInstallation…{RESET}")

    result = subprocess.run(
        [sys.executable, "-m", "openmm.testInstallation"],
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr

    version_line = next((l for l in output.splitlines() if l.startswith("OpenMM Version:")), None)
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
                "        The 'CUDA Version' shown top-right is the maximum supported by your driver.",
                "        It must be ≥ the CUDA version OpenMM was compiled against.",
                "",
                f"{BOLD}Step 2{RESET} — Read the full error from OpenMM:",
                "          python -m openmm.testInstallation",
                f"        {YELLOW}PTX version error{RESET} → your driver is too old, update it.",
                f"        {YELLOW}library not found{RESET} → CUDA runtime missing or not on LD_LIBRARY_PATH.",
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
                "        → Windows/WSL2: install the NVIDIA driver on the Windows host (not inside WSL).",
                "",
                f"{BOLD}Step 2{RESET} — Check that CUDA runtime libraries are on the system:",
                "          ldconfig -p | grep libcuda",
                "        If nothing appears, the CUDA runtime is missing.",
                "        → Install the cuda-runtime package, or set LD_LIBRARY_PATH to its location.",
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
    ("OpenMM",    _check_openmm),
    ("OpenFold3", _check_openfold),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
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
        print(f"  Please fix the issues above before running BindingMetrics.")
    print(f"{BOLD}{'=' * 56}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
