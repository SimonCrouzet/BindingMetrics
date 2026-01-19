#!/usr/bin/env python3
"""
BindingMetrics Test Runner

Usage:
    python scripts/run_tests.py              # Environment check + unit tests
    python scripts/run_tests.py --all        # Run all tests
    python scripts/run_tests.py --check      # Environment check only
    python scripts/run_tests.py -v           # Verbose output
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(text: str) -> str:
    return f"{GREEN}✓{RESET} {text}"


def fail(text: str) -> str:
    return f"{RED}✗{RESET} {text}"


def warn(text: str) -> str:
    return f"{YELLOW}○{RESET} {text}"


def info(text: str) -> str:
    return f"{CYAN}→{RESET} {text}"


def header(text: str) -> None:
    print(f"\n{BOLD}{BLUE}▸ {text}{RESET}")


def check_import(module: str) -> tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "ok")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_environment() -> bool:
    """Check environment setup."""
    header("Environment")

    all_ok = True

    # Python
    v = sys.version_info
    if v >= (3, 11):
        print(ok(f"Python {v.major}.{v.minor}"))
    else:
        print(fail(f"Python {v.major}.{v.minor} {DIM}(need 3.11+){RESET}"))
        all_ok = False

    # Required
    for pkg in ["numpy", "openmm", "mdtraj"]:
        success, ver = check_import(pkg)
        if success:
            print(ok(f"{pkg} {DIM}{ver}{RESET}"))
        else:
            print(fail(f"{pkg} {DIM}not installed{RESET}"))
            all_ok = False

    # binding_metrics
    success, ver = check_import("binding_metrics")
    if success:
        print(ok(f"binding_metrics {DIM}{ver}{RESET}"))
    else:
        print(fail(f"binding_metrics {DIM}not installed{RESET}"))
        print(f"   {DIM}pip install -e .[dev]{RESET}")
        all_ok = False

    # Optional
    for pkg in ["pdbfixer", "pytest"]:
        success, ver = check_import(pkg)
        if success:
            print(ok(f"{pkg} {DIM}{ver}{RESET}"))
        else:
            print(warn(f"{pkg} {DIM}optional{RESET}"))

    # GPU
    try:
        import openmm
        platforms = [openmm.Platform.getPlatform(i).getName()
                     for i in range(openmm.Platform.getNumPlatforms())]
        has_gpu = "CUDA" in platforms or "OpenCL" in platforms
        if has_gpu:
            gpu_platform = "CUDA" if "CUDA" in platforms else "OpenCL"
            print(ok(f"GPU {DIM}{gpu_platform}{RESET}"))
        else:
            print(warn(f"GPU {DIM}not available, using CPU{RESET}"))
    except:
        pass

    return all_ok


def run_tests(test_type: str = "unit", verbose: bool = False) -> int:
    """Run tests and return exit code."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    cmd = [
        sys.executable, "-m", "pytest",
        str(tests_dir),
        "-W", "ignore::UserWarning",
        "--tb=short" if verbose else "--tb=no",
    ]

    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--no-header"])

    # Markers
    if test_type == "unit":
        cmd.extend(["-m", "not integration and not slow"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration and not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])

    if verbose:
        # Show output directly
        return subprocess.call(cmd, cwd=project_root)
    else:
        # Capture and parse
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        output = result.stdout + result.stderr

        # Parse counts
        import re
        passed = failed = skipped = 0
        for match in re.finditer(r"(\d+) (passed|failed|skipped)", output):
            count, status = int(match.group(1)), match.group(2)
            if status == "passed": passed = count
            elif status == "failed": failed = count
            elif status == "skipped": skipped = count

        # Show results
        total = passed + failed + skipped
        if total > 0:
            # Progress bar style
            bar_width = 30
            passed_width = int(bar_width * passed / total)
            failed_width = int(bar_width * failed / total)
            skipped_width = bar_width - passed_width - failed_width

            bar = f"{GREEN}{'█' * passed_width}{RESET}"
            bar += f"{RED}{'█' * failed_width}{RESET}"
            bar += f"{YELLOW}{'░' * skipped_width}{RESET}"

            print(f"  {bar} {passed}/{total}")
            print()

            if passed > 0:
                print(f"  {GREEN}{passed:3d} passed{RESET}")
            if failed > 0:
                print(f"  {RED}{failed:3d} failed{RESET}")
            if skipped > 0:
                print(f"  {DIM}{skipped:3d} skipped{RESET}")

        return result.returncode


def main():
    parser = argparse.ArgumentParser(description="BindingMetrics test runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--integration", action="store_true", help="Integration tests only")
    parser.add_argument("--slow", action="store_true", help="Slow tests only")
    parser.add_argument("--check", action="store_true", help="Check environment only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"\n{BOLD}BindingMetrics{RESET} {DIM}Test Suite{RESET}")
    print("─" * 40)

    env_ok = check_environment()

    if args.check:
        print()
        return 0 if env_ok else 1

    if not env_ok:
        print(f"\n{RED}Fix environment issues first.{RESET}")
        return 1

    # Test type
    if args.all:
        test_type, label = "all", "all tests"
    elif args.integration:
        test_type, label = "integration", "integration tests"
    elif args.slow:
        test_type, label = "slow", "slow tests"
    else:
        test_type, label = "unit", "unit tests"

    header(f"Running {label}")
    print()

    exit_code = run_tests(test_type, verbose=args.verbose)

    # Final message
    print()
    if exit_code == 0:
        print(f"{GREEN}✓ All tests passed{RESET}")
    else:
        print(f"{RED}✗ Some tests failed{RESET}")
        if not args.verbose:
            print(f"  {DIM}Run with -v for details{RESET}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
