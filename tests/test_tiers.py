"""The tiers of tests/_tiers.py: which tests a run selects, from the command line.

The rules are checked twice. The functions of ``tests/_tiers.py`` are called directly (no pytest
session), and each rule of its docstring is run through pytester on a small project written to a
temporary directory, whose conftest loads the hooks of ``tests/conftest.py``. Those cases list
what a run collects with ``--collect-only -q``, which applies the tier like a run does, so
nothing executes; the cases with pytest-xdist run the tests for real in a subprocess.

The small project has 20 tests: 9 in the fast tier, 8 that carry slow, integration or openfold
(the ones ``--full`` adds) and 3 that carry gpu (the ones ``--gpu`` adds).
"""

import re
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _tiers
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
    slow: slow
    integration: integration
    openfold: openfold
    gpu: gpu
    poses: an area
    preflight: an area
    seconds(n): the duration reported for the call of the test
"""

PROJECT = {
    "test_plain.py": """
        import pytest

        def test_fast_one(): pass
        def test_fast_two(): pass

        @pytest.mark.parametrize("n", [1, 2])
        def test_fast_param(n): pass
    """,
    "test_costly.py": """
        import pytest

        @pytest.mark.slow
        def test_slow(): pass

        @pytest.mark.integration
        def test_integration(): pass

        @pytest.mark.openfold
        def test_openfold(): pass

        def test_unmarked_in_a_costly_file(): pass

        @pytest.mark.parametrize("n", [pytest.param(1, marks=pytest.mark.slow), 2])
        def test_one_slow_case(n): pass

        @pytest.mark.slow
        class TestSlowClass:
            def test_a(self): pass
            def test_b(self): pass
    """,
    "test_device.py": """
        import pytest

        @pytest.mark.gpu
        def test_gpu(): pass

        @pytest.mark.gpu
        @pytest.mark.slow
        def test_gpu_and_slow(): pass
    """,
    "inner/test_inner.py": """
        import pytest

        def test_inner_fast(): pass

        @pytest.mark.slow
        def test_inner_slow(): pass
    """,
    "poses_area/test_poses.py": """
        import pytest

        pytestmark = pytest.mark.poses

        def test_poses_fast(): pass

        @pytest.mark.slow
        def test_poses_slow(): pass

        @pytest.mark.gpu
        def test_poses_gpu(): pass
    """,
    "other_area/test_other.py": """
        import pytest

        pytestmark = pytest.mark.preflight

        def test_preflight_fast(): pass
    """,
}

FAST = {
    "test_plain.py::test_fast_one",
    "test_plain.py::test_fast_two",
    "test_plain.py::test_fast_param[1]",
    "test_plain.py::test_fast_param[2]",
    "test_costly.py::test_unmarked_in_a_costly_file",
    "test_costly.py::test_one_slow_case[2]",
    "inner/test_inner.py::test_inner_fast",
    "poses_area/test_poses.py::test_poses_fast",
    "other_area/test_other.py::test_preflight_fast",
}
HEAVY = {
    "test_costly.py::test_slow",
    "test_costly.py::test_integration",
    "test_costly.py::test_openfold",
    "test_costly.py::test_one_slow_case[1]",
    "test_costly.py::TestSlowClass::test_a",
    "test_costly.py::TestSlowClass::test_b",
    "inner/test_inner.py::test_inner_slow",
    "poses_area/test_poses.py::test_poses_slow",
}
GPU_TESTS = {
    "test_device.py::test_gpu",
    "test_device.py::test_gpu_and_slow",
    "poses_area/test_poses.py::test_poses_gpu",
}
EVERYTHING = FAST | HEAVY | GPU_TESTS
THE_OLD_EVERYDAY_EXPRESSION = "not gpu and not integration and not slow and not openfold"


@pytest.fixture
def project(pytester):
    pytester.makeini(INI)
    pytester.makeconftest(PROJECT_CONFTEST)
    for name, text in PROJECT.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return pytester


def run_collection(project, *args):
    return project.runpytest("--collect-only", "-q", "-p", "no:cacheprovider", *args)


def collected(project, *args):
    """The node ids that ``pytest --collect-only -q ARGS`` lists."""
    result = run_collection(project, *args)
    assert result.ret in (0, 5), result.stdout.str()
    return {line for line in result.outlines if "::" in line}


def tier_lines(result):
    return [line for line in result.outlines if line.startswith("tier: ")]


class TestTheTiers:
    def test_the_fast_tier_is_the_default_and_leaves_out_the_four_markers(self, project):
        assert collected(project) == FAST

    def test_the_default_tier_is_what_the_negative_expression_selected(self, project):
        assert collected(project) == collected(project, "-m", THE_OLD_EVERYDAY_EXPRESSION)

    def test_full_adds_slow_integration_and_openfold_but_no_gpu_test(self, project):
        assert collected(project, "--full") == FAST | HEAVY
        assert collected(project, "--full") == collected(project, "-m", "not gpu")

    def test_gpu_adds_the_gpu_tests_whatever_else_they_carry(self, project):
        found = collected(project, "--gpu")
        assert found == FAST | GPU_TESTS
        assert "test_device.py::test_gpu_and_slow" in found  # gpu and slow: gpu decides

    def test_full_and_gpu_together_select_every_test(self, project):
        assert collected(project, "--full", "--gpu") == EVERYTHING

    def test_the_tier_applies_with_no_argument_and_to_a_directory(self, project):
        assert collected(project, "inner") == {"inner/test_inner.py::test_inner_fast"}
        assert collected(project, ".") == FAST
        assert collected(project, "inner", "--full") == {
            "inner/test_inner.py::test_inner_fast",
            "inner/test_inner.py::test_inner_slow",
        }


class TestFilesAndNodeIdsNamedOnTheCommandLine:
    def test_a_named_file_runs_every_test_of_it(self, project):
        assert collected(project, "test_device.py") == {
            "test_device.py::test_gpu",
            "test_device.py::test_gpu_and_slow",
        }
        assert {i for i in collected(project, "test_costly.py")} == {
            i for i in EVERYTHING if i.startswith("test_costly.py")
        }

    def test_a_named_node_id_runs_the_tests_it_names(self, project):
        assert collected(project, "test_costly.py::test_slow") == {"test_costly.py::test_slow"}
        assert collected(project, "test_device.py::test_gpu_and_slow") == {
            "test_device.py::test_gpu_and_slow"
        }
        assert collected(project, "test_costly.py::TestSlowClass") == {
            "test_costly.py::TestSlowClass::test_a",
            "test_costly.py::TestSlowClass::test_b",
        }
        assert collected(project, "test_costly.py::TestSlowClass::test_b") == {
            "test_costly.py::TestSlowClass::test_b"
        }

    def test_a_node_id_without_brackets_names_every_case_of_the_test(self, project):
        assert collected(project, "test_costly.py::test_one_slow_case") == {
            "test_costly.py::test_one_slow_case[1]",
            "test_costly.py::test_one_slow_case[2]",
        }
        assert collected(project, "test_costly.py::test_one_slow_case[1]") == {
            "test_costly.py::test_one_slow_case[1]"
        }

    def test_a_named_file_and_a_directory_each_follow_their_own_rule(self, project):
        assert collected(project, "test_device.py", "inner") == {
            "test_device.py::test_gpu",
            "test_device.py::test_gpu_and_slow",
            "inner/test_inner.py::test_inner_fast",
        }

    def test_the_directory_that_holds_a_named_file_still_obeys_the_tier(self, project):
        found = collected(project, "inner/test_inner.py::test_inner_slow", "inner")
        # the node id asks for the slow test; the directory brings the fast one
        assert found == {
            "inner/test_inner.py::test_inner_slow",
            "inner/test_inner.py::test_inner_fast",
        }
        assert collected(project, "inner/test_inner.py::test_inner_fast", "inner") == {
            "inner/test_inner.py::test_inner_fast"
        }

    def test_a_file_is_found_from_a_path_that_climbs_out_of_the_directory(
        self, project, monkeypatch
    ):
        monkeypatch.chdir(project.path / "inner")
        found = collected(project, "../test_device.py", "../test_costly.py::test_slow")
        assert found == {
            "test_device.py::test_gpu",
            "test_device.py::test_gpu_and_slow",
            "test_costly.py::test_slow",
        }

    def test_a_named_file_follows_the_marker_expression_but_not_the_tier(self, project):
        found = collected(project, "-m", "poses", "poses_area/test_poses.py")
        assert found == {
            "poses_area/test_poses.py::test_poses_fast",
            "poses_area/test_poses.py::test_poses_slow",
            "poses_area/test_poses.py::test_poses_gpu",
        }

    def test_tier_named_makes_the_named_files_obey_the_tier(self, project):
        assert collected(project, "--tier-named", "test_device.py") == set()
        assert collected(project, "--tier-named", "test_device.py", "--gpu") == {
            "test_device.py::test_gpu",
            "test_device.py::test_gpu_and_slow",
        }
        assert collected(project, "--tier-named", "inner/test_inner.py") == {
            "inner/test_inner.py::test_inner_fast"
        }
        assert collected(project, "--tier-named", "inner/test_inner.py", "--full") == {
            "inner/test_inner.py::test_inner_fast",
            "inner/test_inner.py::test_inner_slow",
        }
        assert collected(project, "--tier-named", "test_costly.py::TestSlowClass") == set()


class TestTheMarkerExpression:
    def test_not_gpu_keeps_its_meaning_of_every_test_but_the_gpu_ones(self, project):
        assert collected(project, "-m", "not gpu") == FAST | HEAVY
        assert collected(project, "-m", "not gpu", "--gpu") == FAST | HEAVY

    def test_slow_runs_the_slow_tests(self, project):
        found = collected(project, "-m", "slow")
        assert found == {
            "test_costly.py::test_slow",
            "test_costly.py::test_one_slow_case[1]",
            "test_costly.py::TestSlowClass::test_a",
            "test_costly.py::TestSlowClass::test_b",
            "test_device.py::test_gpu_and_slow",
            "inner/test_inner.py::test_inner_slow",
            "poses_area/test_poses.py::test_poses_slow",
        }

    @pytest.mark.parametrize(
        "expression",
        [
            "gpu",
            "integration",
            "openfold or integration",
            "poses and not slow",
            "(poses or preflight) and not gpu",
            "not (slow or gpu or integration or openfold)",
        ],
    )
    def test_an_expression_that_names_a_tier_marker_leaves_the_tier_off(self, project, expression):
        by_expression = collected(project, "-m", expression)
        assert by_expression  # the expression selects something
        # the same set as before the options existed: pytest alone decides
        only_the_marker = project.runpytest(
            "--collect-only", "-q", "-p", "no:cacheprovider", "-m", expression, "--full", "--gpu"
        )
        assert by_expression == {line for line in only_the_marker.outlines if "::" in line}

    def test_gpu_expression_selects_the_gpu_tests_without_the_option(self, project):
        assert collected(project, "-m", "gpu") == GPU_TESTS

    def test_an_area_marker_leaves_the_tier_on(self, project):
        assert collected(project, "-m", "poses") == {"poses_area/test_poses.py::test_poses_fast"}
        assert collected(project, "-m", "poses", "--full") == {
            "poses_area/test_poses.py::test_poses_fast",
            "poses_area/test_poses.py::test_poses_slow",
        }
        assert collected(project, "-m", "poses", "--full", "--gpu") == {
            "poses_area/test_poses.py::test_poses_fast",
            "poses_area/test_poses.py::test_poses_slow",
            "poses_area/test_poses.py::test_poses_gpu",
        }

    def test_two_areas_obey_the_tier(self, project):
        assert collected(project, "-m", "poses or preflight") == {
            "poses_area/test_poses.py::test_poses_fast",
            "other_area/test_other.py::test_preflight_fast",
        }

    def test_the_selection_by_name_obeys_the_tier(self, project):
        assert collected(project, "-k", "inner") == {"inner/test_inner.py::test_inner_fast"}
        assert collected(project, "-k", "inner", "--full") == {
            "inner/test_inner.py::test_inner_fast",
            "inner/test_inner.py::test_inner_slow",
        }


class TestTheLineAboutTheTier:
    @pytest.mark.parametrize(
        ("arguments", "line"),
        [
            (
                [],
                "tier: fast - 9 of 20 tests selected; "
                "--full adds 8 slow/integration/openfold tests, --gpu adds 3 GPU tests",
            ),
            (["--full"], "tier: full - 17 of 20 tests selected; --gpu adds 3 GPU tests"),
            (
                ["--gpu"],
                "tier: fast+gpu - 12 of 20 tests selected; "
                "--full adds 8 slow/integration/openfold tests",
            ),
            (["--full", "--gpu"], "tier: full+gpu - 20 of 20 tests selected"),
            (
                ["-m", "not gpu"],
                "tier: none, -m names a tier marker - 17 of 20 tests selected",
            ),
            (
                ["-m", "poses"],
                "tier: fast - 1 of 20 tests selected; "
                "--full adds 1 slow/integration/openfold test, --gpu adds 1 GPU test",
            ),
            (
                ["-k", "inner"],
                "tier: fast - 1 of 20 tests selected; --full adds 1 slow/integration/openfold test",
            ),
            (
                ["test_device.py", "inner"],
                "tier: fast - 3 of 4 tests selected; --full adds 1 slow/integration/openfold test",
            ),
        ],
    )
    def test_the_line_counts_what_the_collection_selected_and_what_each_option_adds(
        self, project, arguments, line
    ):
        assert tier_lines(run_collection(project, *arguments)) == [line]

    def test_the_counts_agree_with_the_totals_that_pytest_prints(self, project):
        result = run_collection(project)
        assert tier_lines(result)[0].startswith("tier: fast - 9 of 20 tests selected")
        assert any("9/20 tests collected (11 deselected)" in line for line in result.outlines)
        result = run_collection(project, "--full", "--gpu")
        assert any("20 tests collected" in line for line in result.outlines)

    def test_the_line_is_printed_once_when_the_tests_run(self, project):
        for arguments in ([], ["-q"], ["-v"]):
            result = project.runpytest("-p", "no:cacheprovider", *arguments)
            assert len(tier_lines(result)) == 1, arguments
            result.assert_outcomes(passed=9, deselected=11)
        # -qq has no summary line, only the one about the tier
        assert len(tier_lines(project.runpytest("-p", "no:cacheprovider", "-qq"))) == 1

    def test_the_line_is_in_the_header_of_a_plain_collection_and_of_a_run(self, project):
        result = project.runpytest("--collect-only", "-p", "no:cacheprovider")
        assert len(tier_lines(result)) == 1
        index = [i for i, line in enumerate(result.outlines) if line.startswith("collected ")]
        assert index and result.outlines[index[0] + 1].startswith("tier: fast")

    def test_what_a_run_leaves_out_is_reported_as_deselected(self, project):
        result = project.runpytest("-p", "no:cacheprovider", "--full")
        result.assert_outcomes(passed=17, deselected=3)
        result = project.runpytest("-p", "no:cacheprovider", "--full", "--gpu")
        result.assert_outcomes(passed=20)


class TestAlongsideTheOtherHooks:
    def test_the_area_tags_are_on_the_tests_when_the_tier_is_applied(self, project):
        (project.path / "conftest.py").write_text(
            PROJECT_CONFTEST
            + "\npytest_collection_modifyitems = _repo.pytest_collection_modifyitems\n",
            encoding="utf-8",
        )
        # a test file outside the repository imports nothing from the package: area misc
        (project.path / "pytest.ini").write_text(INI + "    misc: an area\n", encoding="utf-8")
        assert collected(project, "-m", "misc") == FAST
        assert collected(project, "-m", "misc", "--full") == FAST | HEAVY

    def test_the_slow_guard_sees_only_the_tests_that_run(self, project):
        (project.path / "test_seconds.py").write_text(
            textwrap.dedent(
                """
                import pytest

                @pytest.mark.seconds(3.0)
                def test_untagged_and_slow(): pass

                @pytest.mark.slow
                @pytest.mark.seconds(9.0)
                def test_tagged_slow(): pass
                """
            ),
            encoding="utf-8",
        )
        result = project.runpytest("-p", "no:cacheprovider", "--strict-slow", "test_seconds.py")
        # the file is named, so both tests run; the tagged one is not listed
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.stdout.fnmatch_lines(["*1 test took 2 s or more without any of slow*"])
        result = project.runpytest(
            "-p", "no:cacheprovider", "--strict-slow", "--tier-named", "test_seconds.py"
        )
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.assert_outcomes(passed=1, failed=0, deselected=1)
        result = project.runpytest(
            "-p", "no:cacheprovider", "--strict-slow", "--slow-seconds", "20", "--full"
        )
        assert result.ret == 0


class TestWithXdist:
    """The controller and the workers agree, and the controller prints the line once."""

    @staticmethod
    def passed_ids(result):
        found = (re.search(r"\[gw\d\] \[ *\d+%\] PASSED (\S+)", line) for line in result.outlines)
        return {match.group(1) for match in found if match}

    @pytest.mark.slow
    def test_the_workers_run_the_tests_of_the_tier_and_one_line_is_printed(self, project):
        pytest.importorskip("xdist")
        result = project.runpytest_subprocess("-p", "no:cacheprovider", "-n", "2", "-v")
        assert result.ret == 0, result.stdout.str()
        assert self.passed_ids(result) == FAST
        assert tier_lines(result) == [
            "tier: fast - 9 of 20 tests selected; "
            "--full adds 8 slow/integration/openfold tests, --gpu adds 3 GPU tests"
        ]

        result = project.runpytest_subprocess(
            "-p", "no:cacheprovider", "-n", "2", "-v", "--full", "-m", "not poses"
        )
        assert result.ret == 0, result.stdout.str()
        assert self.passed_ids(result) == {i for i in FAST | HEAVY if not i.startswith("poses")}
        assert len(tier_lines(result)) == 1

    @pytest.mark.slow
    def test_a_named_file_and_a_directory_follow_their_rules_in_every_worker(self, project):
        pytest.importorskip("xdist")
        result = project.runpytest_subprocess(
            "-p", "no:cacheprovider", "-n", "2", "-v", "test_device.py", "inner"
        )
        assert self.passed_ids(result) == {
            "test_device.py::test_gpu",
            "test_device.py::test_gpu_and_slow",
            "inner/test_inner.py::test_inner_fast",
        }
        assert len(tier_lines(result)) == 1

    @pytest.mark.slow
    def test_a_collection_with_workers_prints_the_line_once(self, project):
        pytest.importorskip("xdist")
        result = project.runpytest_subprocess(
            "-p", "no:cacheprovider", "-n", "2", "--collect-only", "-q"
        )
        assert len(tier_lines(result)) == 1
        assert {line for line in result.outlines if "::" in line} == FAST


class TestTheRulesApartFromPytest:
    @pytest.mark.parametrize(
        ("expression", "words"),
        [
            ("", set()),
            ("slow", {"slow"}),
            ("not gpu and not slow", {"not", "gpu", "and", "slow"}),
            ("(poses or preflight)", {"poses", "or", "preflight"}),
            ("slowish", {"slowish"}),
            ("not_gpu", {"not_gpu"}),
            ("a.b-c and d[x]", {"a.b-c", "and", "d[x]"}),
        ],
    )
    def test_the_words_of_an_expression_are_whole_identifiers(self, expression, words):
        assert _tiers.markers_in_expression(expression) == words

    @pytest.mark.parametrize(
        ("expression", "by_hand"),
        [
            ("", False),
            ("poses", False),
            ("poses or preflight", False),
            ("slowish", False),
            ("not_gpu", False),
            ("slow", True),
            ("not gpu", True),
            ("poses and (openfold or integration)", True),
            ("(gpu)", True),
        ],
    )
    def test_an_expression_is_a_tier_by_hand_when_it_names_one_of_the_four_markers(
        self, expression, by_hand
    ):
        assert _tiers.TierRequest(markexpr=expression).chosen_by_hand is by_hand

    @pytest.mark.parametrize(
        ("names", "kind"),
        [
            ([], "fast"),
            (["poses", "parametrize"], "fast"),
            (["slow"], "heavy"),
            (["integration", "poses"], "heavy"),
            (["openfold"], "heavy"),
            (["gpu"], "gpu"),
            (["gpu", "slow"], "gpu"),
            (["slow", "integration", "gpu", "openfold"], "gpu"),
        ],
    )
    def test_a_test_is_fast_heavy_or_gpu_by_its_markers(self, names, kind):
        assert _tiers.kind_of(names) == kind

    def test_the_four_markers_are_the_ones_of_the_slow_guard(self):
        assert _tiers.TIER_MARKERS == ("slow", "integration", "openfold", "gpu")

    @pytest.mark.parametrize(
        ("options", "kind", "named", "held"),
        [
            ({}, "fast", False, False),
            ({}, "heavy", False, True),
            ({}, "gpu", False, True),
            ({"full": True}, "heavy", False, False),
            ({"full": True}, "gpu", False, True),
            ({"gpu": True}, "heavy", False, True),
            ({"gpu": True}, "gpu", False, False),
            ({"full": True, "gpu": True}, "gpu", False, False),
            ({}, "heavy", True, False),
            ({}, "gpu", True, False),
            ({"tier_named": True}, "heavy", True, True),
            ({"tier_named": True}, "gpu", True, True),
            ({"tier_named": True, "full": True}, "heavy", True, False),
            ({"markexpr": "not gpu"}, "gpu", False, False),
            ({"markexpr": "slow"}, "heavy", False, False),
            ({"markexpr": "poses"}, "heavy", False, True),
            ({"markexpr": "poses"}, "heavy", True, False),
        ],
    )
    def test_what_the_tier_holds_back(self, options, kind, named, held):
        assert _tiers.TierRequest(**options).holds_back(kind, named) is held

    def test_a_node_id_names_the_tests_it_starts(self, tmp_path):
        (tmp_path / "t").mkdir()
        file = tmp_path / "t" / "test_a.py"
        file.write_text("", encoding="utf-8")
        other = tmp_path / "test_b.py"
        other.write_text("", encoding="utf-8")

        def named(*arguments, nodeid, path=file):
            targets = _tiers.named_targets(arguments, tmp_path)
            return _tiers.is_named(str(path), nodeid, targets)

        assert named("t/test_a.py", nodeid="t/test_a.py::test_x")
        assert not named("t/test_a.py", nodeid="t/test_b.py::test_x", path=other)
        assert named("t/test_a.py::test_x", nodeid="t/test_a.py::test_x")
        assert named("t/test_a.py::test_x", nodeid="t/test_a.py::test_x[1-a::b]")
        assert not named("t/test_a.py::test_x", nodeid="t/test_a.py::test_xy")
        assert named("t/test_a.py::TestC", nodeid="t/test_a.py::TestC::test_m[2]")
        assert not named("t/test_a.py::TestC", nodeid="t/test_a.py::TestCD::test_m")
        assert named("t/test_a.py::TestC::test_m", nodeid="t/test_a.py::TestC::test_m[2]")
        assert not named("t/test_a.py::TestC::test_m", nodeid="t/test_a.py::TestC::test_n")
        assert named("t/test_a.py::test_x[1]", nodeid="t/test_a.py::test_x[1]")
        assert not named("t/test_a.py::test_x[1]", nodeid="t/test_a.py::test_x[2]")
        assert named("t/../t/test_a.py", nodeid="t/test_a.py::test_x")
        assert not named("t", nodeid="t/test_a.py::test_x")  # a directory names nothing
        assert not named("t/missing.py", nodeid="t/test_a.py::test_x")

    def test_a_directory_and_a_missing_path_are_not_named_files(self, tmp_path):
        (tmp_path / "d").mkdir()
        (tmp_path / "f.py").write_text("", encoding="utf-8")
        targets = _tiers.named_targets(
            ["d", "f.py", "missing.py::test_x", "f.py::test_y"], tmp_path
        )
        assert [(Path(t.path).name, t.node) for t in targets] == [("f.py", ""), ("f.py", "test_y")]

    @pytest.mark.parametrize(
        ("options", "counts", "line"),
        [
            (
                {},
                (9304, 8809, 407, 88),
                "tier: fast - 8809 of 9304 tests selected; "
                "--full adds 407 slow/integration/openfold tests, --gpu adds 88 GPU tests",
            ),
            (
                {"full": True},
                (9304, 9216, 0, 88),
                "tier: full - 9216 of 9304 tests selected; --gpu adds 88 GPU tests",
            ),
            (
                {"gpu": True},
                (9304, 8897, 407, 0),
                "tier: fast+gpu - 8897 of 9304 tests selected; "
                "--full adds 407 slow/integration/openfold tests",
            ),
            (
                {"full": True, "gpu": True},
                (9304, 9304, 0, 0),
                "tier: full+gpu - 9304 of 9304 tests selected",
            ),
            (
                {},
                (1, 1, 0, 0),
                "tier: fast - 1 of 1 test selected",
            ),
            (
                {},
                (3, 0, 1, 2),
                "tier: fast - 0 of 3 tests selected; "
                "--full adds 1 slow/integration/openfold test, --gpu adds 2 GPU tests",
            ),
            (
                {"markexpr": "slow"},
                (10, 4, 0, 0),
                "tier: none, -m names a tier marker - 4 of 10 tests selected",
            ),
        ],
    )
    def test_the_line_names_the_tier_and_what_each_option_adds(self, options, counts, line):
        assert _tiers.describe(_tiers.TierRequest(**options), _tiers.TierCounts(*counts)) == line

    def test_the_counts_travel_as_a_plain_dictionary(self):
        counts = _tiers.TierCounts(10, 6, 3, 1)
        assert _tiers.TierCounts(**counts.as_dict()) == counts


class TestTheRealSuite:
    def test_the_options_are_registered(self, request):
        for option in ("--full", "--gpu", "--tier-named"):
            assert isinstance(request.config.getoption(option), bool)

    def test_the_speed_markers_of_the_guard_are_the_tier_markers(self):
        import conftest

        assert conftest.SPEED_MARKERS == _tiers.TIER_MARKERS

    @pytest.mark.slow
    def test_the_tiers_of_a_directory_are_what_the_negative_expressions_selected(self):
        """The real conftest on tests/poses, which holds 64 slow or integration cases."""
        repo = Path(__file__).resolve().parent.parent
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]

        def listing(*arguments):
            done = subprocess.run(
                [*command, *arguments, "tests/poses"],
                cwd=repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
                check=False,
            )
            return {line for line in done.stdout.splitlines() if "::" in line}

        runs = (
            (),
            ("-m", THE_OLD_EVERYDAY_EXPRESSION),
            ("--full",),
            ("-m", "not gpu"),
        )
        with ThreadPoolExecutor(max_workers=len(runs)) as pool:  # the collections are alike
            default, former_default, full, former_full = pool.map(lambda a: listing(*a), runs)
        assert default and default == former_default
        assert full == former_full
        assert default < full  # the directory holds cases that only --full adds
