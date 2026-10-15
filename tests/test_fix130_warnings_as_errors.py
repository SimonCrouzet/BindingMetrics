"""The suite can run with warnings as errors, and a package DeprecationWarning fails a test.

``[tool.pytest.ini_options]`` turns a DeprecationWarning attributed to a ``binding_metrics``
module into an error and ignores the import-time SWIG warning of OpenMM 8.2, which as an
error crashes the interpreter. ``tests/conftest.py`` imports OpenMM with that warning ignored.
Each check runs a separate pytest session, so the filters of this session stay untouched.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALIAS_MESSAGE = "PredictionRecord is deprecated and renamed PredictedPose"

#: A DeprecationWarning attributed to a module of the package: the old name is used by a frame
#: whose module is ``binding_metrics.<something>``, as when the package's own code uses it.
FROM_THE_PACKAGE = (
    "exec(compile('from binding_metrics.predictors import PredictionRecord', '<package>', "
    "'exec'), {'__name__': 'binding_metrics.some_module'})"
)
FROM_A_TEST_MODULE = "from binding_metrics.predictors import PredictionRecord"


def _run_pytest(*args, interpreter_flags=()):
    """Run ``python [flags] -m pytest args`` with this repository's configuration file."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    command = [
        sys.executable,
        *interpreter_flags,
        "-m",
        "pytest",
        "-c",
        str(REPO / "pyproject.toml"),
        "--rootdir",
        str(REPO),
        "-p",
        "no:cacheprovider",
        "-q",
        *map(str, args),
    ]
    return subprocess.run(
        command, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300
    )


def _write_case(tmp_path, body: str) -> Path:
    path = tmp_path / "test_case.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


@pytest.mark.slow
class TestAPackageDeprecationFailsTheTest:
    def test_the_old_name_used_from_a_package_module_is_an_error(self, tmp_path):
        case = _write_case(
            tmp_path,
            f"""
            def test_uses_the_old_name():
                {FROM_THE_PACKAGE}
            """,
        )
        result = _run_pytest(case)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "1 failed" in result.stdout
        assert f"DeprecationWarning: {ALIAS_MESSAGE}" in result.stdout

    def test_without_the_ini_filter_the_same_use_is_only_a_warning(self, tmp_path):
        """Control: the failure above comes from the filter in pyproject.toml."""
        case = _write_case(
            tmp_path,
            f"""
            def test_uses_the_old_name():
                {FROM_THE_PACKAGE}
            """,
        )
        result = _run_pytest(case, "-o", "filterwarnings=")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout

    def test_the_same_use_from_a_test_module_stays_a_warning(self, tmp_path):
        case = _write_case(
            tmp_path,
            f"""
            def test_uses_the_old_name():
                {FROM_A_TEST_MODULE}
            """,
        )
        result = _run_pytest(case)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout
        assert ALIAS_MESSAGE in result.stdout  # reported in the warnings summary

    def test_pytest_warns_catches_the_package_warning(self, tmp_path):
        case = _write_case(
            tmp_path,
            f"""
            import pytest

            def test_expects_the_warning():
                with pytest.warns(DeprecationWarning, match="PredictionRecord is deprecated"):
                    {FROM_THE_PACKAGE}
            """,
        )
        result = _run_pytest(case)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout

    def test_a_deprecation_from_another_library_is_not_turned_into_an_error(self, tmp_path):
        case = _write_case(
            tmp_path,
            """
            import warnings

            def test_third_party_warning():
                warnings.warn("old api", DeprecationWarning, stacklevel=1)
            """,
        )
        result = _run_pytest(case)
        assert result.returncode == 0, result.stdout + result.stderr


class TestOpenMMImportUnderWarningsAsErrors:
    """``python -W error::DeprecationWarning -c "import openmm"`` segfaults (exit 139)."""

    @pytest.fixture(autouse=True)
    def _needs_openmm(self):
        pytest.importorskip("openmm")

    @pytest.mark.slow
    def test_the_ini_filter_lets_a_test_module_import_openmm(self, tmp_path):
        case = _write_case(
            tmp_path,
            """
            import openmm

            def test_imported():
                assert openmm.Platform.getNumPlatforms() >= 1
            """,
        )
        result = _run_pytest(case, interpreter_flags=["-W", "error::DeprecationWarning"])
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "1 passed" in result.stdout

    def test_the_conftest_imports_openmm_when_the_ini_filters_are_off(self):
        """``-p no:warnings`` drops the ini filters; the import in conftest must hold alone."""
        result = _run_pytest(
            "tests/predictors/test_deprecated_alias.py::TestTheOldNameIsTheSameClass",
            "-p",
            "no:warnings",
            interpreter_flags=["-W", "error::DeprecationWarning"],
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "3 passed" in result.stdout

    def test_the_package_tests_run_under_the_flag_with_the_ini_filters(self):
        result = _run_pytest(
            "tests/predictors/test_deprecated_alias.py::TestTheOldNameIsTheSameClass",
            interpreter_flags=["-W", "error::DeprecationWarning"],
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "3 passed" in result.stdout
