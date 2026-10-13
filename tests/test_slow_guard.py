"""The guard of tests/conftest.py lists the slow tests that carry no speed marker.

Each case runs pytest on a small project written in a temporary directory, whose conftest loads
the hooks of tests/conftest.py. The duration of a test is set by a ``seconds`` marker in a hook
of the project, so no case sleeps (the case with pytest-xdist sleeps for a third of a second, to
see a real duration reach the controller).
"""

import re
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

REPO_CONFTEST = Path(__file__).resolve().parent / "conftest.py"

PROJECT_CONFTEST = f'''
import importlib.util

import pytest

_spec = importlib.util.spec_from_file_location("repo_tests_conftest", r"{REPO_CONFTEST}")
_repo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repo)
pytest_addoption = _repo.pytest_addoption
pytest_configure = _repo.pytest_configure


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    marker = item.get_closest_marker("seconds")
    if report.when == "call" and marker is not None:
        report.duration = marker.args[0]
    return report
'''

INI = """
[pytest]
markers =
    seconds(n): the duration reported for the call of the test
    slow: slow
    integration: integration
    openfold: openfold
    gpu: gpu
"""

TESTS = """
import pytest


@pytest.mark.seconds(3.0)
def test_untagged_and_slow():
    pass


@pytest.mark.seconds(1.0)
def test_untagged_and_fast_enough():
    pass


@pytest.mark.seconds(9.0)
@pytest.mark.slow
def test_tagged_slow():
    pass


@pytest.mark.seconds(9.0)
@pytest.mark.integration
def test_tagged_integration():
    pass


@pytest.mark.seconds(9.0)
@pytest.mark.openfold
def test_tagged_openfold():
    pass


@pytest.mark.seconds(9.0)
@pytest.mark.gpu
def test_tagged_gpu():
    pass


@pytest.mark.slow
class TestTaggedClass:
    @pytest.mark.seconds(9.0)
    def test_inherits_the_mark_of_its_class(self):
        pass


@pytest.mark.seconds(2.0)
def test_exactly_at_the_threshold():
    pass
"""


@pytest.fixture
def project(pytester):
    pytester.makeini(INI)
    pytester.makeconftest(PROJECT_CONFTEST)
    pytester.makepyfile(test_durations=TESTS)
    return pytester


def listed(result):
    """The node ids under the 'untagged slow tests' section of a run, longest first."""
    lines = result.outlines
    header = [i for i, line in enumerate(lines) if "untagged slow tests" in line]
    if not header:
        return []
    found = (re.fullmatch(r"\s*\d+\.\d\d s  (\S+)", line) for line in lines[header[0] + 1 :])
    return [match.group(1) for match in found if match]


class TestTheReport:
    def test_untagged_slow_tests_are_listed_and_the_run_still_passes(self, project):
        result = project.runpytest("-q", "-p", "no:cacheprovider")
        assert result.ret == 0
        assert listed(result) == [
            "test_durations.py::test_untagged_and_slow",
            "test_durations.py::test_exactly_at_the_threshold",
        ]
        result.stdout.fnmatch_lines(
            [
                "*untagged slow tests*",
                "*2 tests took 2 s or more without any of slow, integration, openfold, gpu*",
            ]
        )
        assert not any("--strict-slow" in line for line in result.outlines)

    def test_a_tagged_test_or_a_fast_one_is_not_listed(self, project):
        names = " ".join(listed(project.runpytest("-q", "-p", "no:cacheprovider")))
        for name in ("::test_tagged_", "::TestTaggedClass", "fast_enough"):
            assert name not in names

    def test_the_threshold_is_an_option(self, project):
        result = project.runpytest("-q", "-p", "no:cacheprovider", "--slow-seconds", "1.0")
        assert listed(result) == [
            "test_durations.py::test_untagged_and_slow",
            "test_durations.py::test_exactly_at_the_threshold",
            "test_durations.py::test_untagged_and_fast_enough",
        ]
        result = project.runpytest("-q", "-p", "no:cacheprovider", "--slow-seconds", "5")
        assert listed(result) == []
        assert not any("untagged slow tests" in line for line in result.outlines)


class TestStrict:
    def test_strict_slow_fails_the_run_that_has_an_untagged_slow_test(self, project):
        result = project.runpytest("-q", "-p", "no:cacheprovider", "--strict-slow")
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.stdout.fnmatch_lines(["*--strict-slow: the run fails.*"])
        assert len(listed(result)) == 2

    def test_strict_slow_passes_when_every_slow_test_is_tagged(self, project):
        result = project.runpytest(
            "-q", "-p", "no:cacheprovider", "--strict-slow", "-m", "slow or integration or gpu"
        )
        assert result.ret == 0
        assert listed(result) == []


class TestWithXdist:
    @pytest.mark.slow
    def test_the_durations_of_the_workers_reach_the_guard(self, pytester):
        pytest.importorskip("xdist")
        pytester.makeini(INI)
        pytester.makeconftest(PROJECT_CONFTEST)
        pytester.makepyfile(
            test_sleep="""
            import time

            import pytest


            def test_a_slow_call_without_a_marker():
                time.sleep(0.3)


            @pytest.mark.slow
            def test_a_slow_call_with_a_marker():
                time.sleep(0.3)


            def test_a_quick_call():
                pass
            """
        )
        result = pytester.runpytest_subprocess(
            "-q", "-p", "no:cacheprovider", "-n", "2", "--slow-seconds", "0.2", "--strict-slow"
        )
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert listed(result) == ["test_sleep.py::test_a_slow_call_without_a_marker"]
        result.assert_outcomes(passed=3)
