"""scripts/tag_slow_tests.py adds ``@pytest.mark.slow`` to the tests named in a list of node ids.

Each case writes a small test file in a temporary directory, tags some of its tests and reads
the file back; the last cases ask pytest itself which tests carry the marker. The real test
suite is never touched.
"""

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tag_slow_tests.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("tag_slow_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tag_slow_tests"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("tag_slow_tests", None)


def write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8", newline="")
    return path


def tag(script, root, name, ids, collected=None, **options):
    """Tag the ids in ``root/name`` and return (new text, the report of that file)."""
    targets = [script.parse_node_id(i) for i in ids]
    cases = script.read_collected("\n".join(collected)) if collected else {}
    reports = script.tag_tests(targets, root, cases, **options)
    assert len(reports) == 1
    return (root / name).read_text(encoding="utf-8"), reports[0]


PLAIN = """
import pytest


def test_fast():
    assert True


@pytest.fixture
def unused():
    return 1


# explains the next test
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.filterwarnings("ignore")
def test_heavy(n):
    assert n


class TestA:
    def test_one(self):
        assert True

    @pytest.mark.usefixtures("unused")
    def test_two(self):
        assert True
"""


class TestNodeIds:
    def test_a_node_id_is_split_into_file_names_and_case(self, script):
        target = script.parse_node_id("tests/poses/test_x.py::TestA::test_b[a-b::c d]")
        assert target.path == "tests/poses/test_x.py"
        assert target.names == ("TestA", "test_b")
        assert target.case == "a-b::c d"

    def test_a_text_after_a_tab_and_blank_or_comment_lines_are_ignored(self, script):
        assert script.parse_node_id("tests/t.py::test_a\t12.5 s").names == ("test_a",)
        assert script.parse_node_id("   ") is None
        assert script.parse_node_id("# a comment") is None

    def test_an_id_without_a_test_is_refused(self, script):
        with pytest.raises(ValueError, match="no '::'"):
            script.parse_node_id("tests/test_x.py")


class TestPlacement:
    def test_a_function_gets_the_mark_and_nothing_else_changes(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PLAIN)
        text, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_fast"])
        assert report.functions == ["test_fast"]
        assert "@pytest.mark.slow\ndef test_fast():" in text
        assert text.replace("@pytest.mark.slow\n", "", 1) == textwrap.dedent(PLAIN).lstrip("\n")

    def test_decorators_keep_their_order_and_the_mark_goes_under_the_comment(
        self, script, tmp_path
    ):
        write(tmp_path, "tests/test_a.py", PLAIN)
        ids = ["tests/test_a.py::test_heavy[1]", "tests/test_a.py::test_heavy[2]"]
        text, report = tag(script, tmp_path, "tests/test_a.py", ids, ids)
        assert report.functions == ["test_heavy"] and not report.whole_function_fallbacks
        assert (
            "# explains the next test\n@pytest.mark.slow\n"
            '@pytest.mark.parametrize("n", [1, 2])\n'
            '@pytest.mark.filterwarnings("ignore")\ndef test_heavy(n):'
        ) in text

    def test_a_method_takes_the_indentation_of_its_def(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PLAIN)
        text, report = tag(
            script, tmp_path, "tests/test_a.py", ["tests/test_a.py::TestA::test_two"]
        )
        assert report.functions == ["TestA::test_two"]
        assert (
            '    @pytest.mark.slow\n    @pytest.mark.usefixtures("unused")\n    def test_two(self):'
        ) in text
        assert "    def test_one" in text and "slow\n    def test_one" not in text

    def test_the_file_keeps_its_line_endings_and_non_ascii_text(self, script, tmp_path):
        source = (
            '"""Étude: température."""\r\nimport pytest\r\n\r\n\r\n'
            'def test_a():\r\n    assert "é"\r\n'
        )
        path = tmp_path / "tests" / "test_a.py"
        path.parent.mkdir()
        path.write_bytes(source.encode("utf-8"))
        script.tag_tests([script.parse_node_id("tests/test_a.py::test_a")], tmp_path, {})
        data = path.read_bytes()
        assert data == source.replace("def test_a", "@pytest.mark.slow\r\ndef test_a").encode()


class TestImport:
    def test_import_pytest_is_added_when_the_file_lacks_it(self, script, tmp_path):
        write(
            tmp_path,
            "tests/test_a.py",
            '''
            """Doc."""

            import os


            def test_a():
                assert os
            ''',
        )
        text, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_a"])
        assert report.import_added
        assert text.count("import pytest") == 1
        assert text.index("import os") < text.index("import pytest") < text.index("@pytest")
        compile(text, "test_a.py", "exec")

    def test_import_pytest_is_added_after_a_docstring_when_there_is_no_import(
        self, script, tmp_path
    ):
        write(tmp_path, "tests/test_a.py", '"""Doc."""\n\n\ndef test_a():\n    pass\n')
        text, _ = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_a"])
        assert text.startswith('"""Doc."""\n\nimport pytest\n')
        compile(text, "test_a.py", "exec")

    def test_a_pytest_import_inside_a_function_does_not_count(self, script, tmp_path):
        write(
            tmp_path,
            "tests/test_a.py",
            """
            def test_a():
                import pytest

                assert pytest
            """,
        )
        text, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_a"])
        assert report.import_added
        assert text.startswith("import pytest\n")

    def test_an_existing_import_is_not_repeated(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PLAIN)
        text, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_fast"])
        assert not report.import_added
        assert text.count("import pytest") == 1


class TestSpeedMarkersAreRespected:
    @pytest.mark.parametrize(
        "decorator",
        [
            "@pytest.mark.slow",
            "@pytest.mark.integration",
            "@pytest.mark.openfold",
            "@pytest.mark.gpu",
            "@requires_cuda",
            "@pytest.mark.skipif(False, reason='x')\n@pytest.mark.integration",
        ],
    )
    def test_a_test_with_a_speed_marker_is_left_alone(self, script, tmp_path, decorator):
        source = (
            "import pytest\nfrom conftest import requires_cuda\n\n\n"
            f"{decorator}\ndef test_a():\n    pass\n"
        )
        path = write(tmp_path, "tests/test_a.py", source)
        _, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_a"])
        assert not report.changed and report.already_marked == ["test_a"]
        assert path.read_text(encoding="utf-8") == source

    def test_a_marked_class_and_a_marked_module_cover_their_tests(self, script, tmp_path):
        source = (
            "import pytest\n\npytestmark = [pytest.mark.skipif(False, reason='x')]\n\n\n"
            "@pytest.mark.integration\nclass TestA:\n    def test_a(self):\n        pass\n"
        )
        write(tmp_path, "tests/test_a.py", source)
        _, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::TestA::test_a"])
        assert not report.changed and report.already_marked == ["TestA::test_a"]
        write(
            tmp_path,
            "tests/test_b.py",
            "import pytest\n\npytestmark = pytest.mark.slow\n\n\ndef test_b():\n    pass\n",
        )
        reports = script.tag_tests([script.parse_node_id("tests/test_b.py::test_b")], tmp_path, {})
        assert not reports[0].changed

    def test_a_second_run_changes_nothing(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PLAIN)
        ids = ["tests/test_a.py::test_fast", "tests/test_a.py::TestA::test_two"]
        first, report = tag(script, tmp_path, "tests/test_a.py", ids)
        assert report.changed
        second, report = tag(script, tmp_path, "tests/test_a.py", ids)
        assert second == first and not report.changed
        assert sorted(report.already_marked) == ["TestA::test_two", "test_fast"]


CLASSES = """
import pytest


class TestMostlySlow:
    def test_a(self):
        pass

    def test_b(self):
        pass

    def test_c(self):
        pass

    def test_d(self):
        pass

    def test_e(self):
        pass


class TestHalf:
    def test_a(self):
        pass

    def test_b(self):
        pass


class TestWithAnIntegrationTest:
    def test_a(self):
        pass

    def test_b(self):
        pass

    @pytest.mark.integration
    def test_c(self):
        pass
"""


class TestClasses:
    def test_a_class_is_marked_when_most_of_its_methods_are_listed(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", CLASSES)
        ids = [f"tests/test_a.py::TestMostlySlow::test_{n}" for n in "abcd"]
        text, report = tag(script, tmp_path, "tests/test_a.py", ids)
        assert report.classes == ["TestMostlySlow"] and report.functions == []
        assert "@pytest.mark.slow\nclass TestMostlySlow:" in text
        assert text.count("@pytest.mark.slow") == 1

    def test_the_methods_are_marked_one_by_one_below_the_share(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", CLASSES)
        ids = [f"tests/test_a.py::TestMostlySlow::test_{n}" for n in "ab"]
        text, report = tag(script, tmp_path, "tests/test_a.py", ids)
        assert report.classes == []
        assert report.functions == ["TestMostlySlow::test_a", "TestMostlySlow::test_b"]
        assert text.count("@pytest.mark.slow") == 2

    def test_the_share_is_an_option(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", CLASSES)
        ids = [f"tests/test_a.py::TestMostlySlow::test_{n}" for n in "ab"]
        _, report = tag(script, tmp_path, "tests/test_a.py", ids, class_share=0.4)
        assert report.classes == ["TestMostlySlow"]

    def test_one_listed_method_never_marks_a_class(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", CLASSES)
        _, report = tag(
            script,
            tmp_path,
            "tests/test_a.py",
            ["tests/test_a.py::TestHalf::test_a"],
            class_share=0.0,
        )
        assert report.classes == [] and report.functions == ["TestHalf::test_a"]

    def test_a_class_with_a_method_of_another_speed_marker_is_not_marked(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", CLASSES)
        ids = [f"tests/test_a.py::TestWithAnIntegrationTest::test_{n}" for n in "ab"]
        text, report = tag(script, tmp_path, "tests/test_a.py", ids)
        assert report.classes == [] and len(report.functions) == 2
        assert "@pytest.mark.slow\nclass" not in text


PARAMS = """
import pytest


@pytest.mark.parametrize("n", [1, 2, 3])
def test_numbers(n):
    pass


@pytest.mark.parametrize(
    "a, b",
    [
        (1, 2),
        pytest.param(3, 4, id="named"),
        pytest.param(
            5,
            6,
            id="skipped",
            marks=pytest.mark.skipif(False, reason="x"),
        ),
        pytest.param(7, 8, id="marked", marks=pytest.mark.integration),
        (9, 10),
    ],
)
def test_pairs(a, b):
    pass


NAMES = ["x", "y"]


@pytest.mark.parametrize("name", NAMES)
def test_from_a_name(name):
    pass


@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [3, 4])
def test_stacked(a, b):
    pass


@pytest.fixture(params=["p", "q"])
def thing(request):
    return request.param


def test_with_a_fixture(thing):
    pass


@pytest.mark.parametrize("value", [(1, 2), (3, 4)])
def test_tuples_as_one_value(value):
    pass
"""

PARAM_COLLECTED = [
    "tests/test_a.py::test_numbers[1]",
    "tests/test_a.py::test_numbers[2]",
    "tests/test_a.py::test_numbers[3]",
    "tests/test_a.py::test_pairs[1-2]",
    "tests/test_a.py::test_pairs[named]",
    "tests/test_a.py::test_pairs[skipped]",
    "tests/test_a.py::test_pairs[marked]",
    "tests/test_a.py::test_pairs[9-10]",
    "tests/test_a.py::test_from_a_name[x]",
    "tests/test_a.py::test_from_a_name[y]",
    "tests/test_a.py::test_stacked[3-1]",
    "tests/test_a.py::test_stacked[3-2]",
    "tests/test_a.py::test_stacked[4-1]",
    "tests/test_a.py::test_stacked[4-2]",
    "tests/test_a.py::test_with_a_fixture[p]",
    "tests/test_a.py::test_with_a_fixture[q]",
    "tests/test_a.py::test_tuples_as_one_value[value0]",
    "tests/test_a.py::test_tuples_as_one_value[value1]",
    "",
    "1 test collected in 0.01s",
]


class TestParametrisedCases:
    def test_only_the_slow_entries_of_a_literal_list_are_marked(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PARAMS)
        text, report = tag(
            script,
            tmp_path,
            "tests/test_a.py",
            ["tests/test_a.py::test_numbers[2]"],
            PARAM_COLLECTED,
        )
        assert report.cases == ["test_numbers[2]"] and report.functions == []
        assert (
            '@pytest.mark.parametrize("n", [1, pytest.param(2, marks=pytest.mark.slow), 3])' in text
        )

    def test_entries_that_are_tuples_are_unpacked_and_pytest_params_keep_their_arguments(
        self, script, tmp_path
    ):
        write(tmp_path, "tests/test_a.py", PARAMS)
        ids = [
            "tests/test_a.py::test_pairs[1-2]",
            "tests/test_a.py::test_pairs[named]",
            "tests/test_a.py::test_pairs[skipped]",
            "tests/test_a.py::test_pairs[marked]",
        ]
        text, report = tag(script, tmp_path, "tests/test_a.py", ids, PARAM_COLLECTED)
        assert report.cases == [f"test_pairs[{c}]" for c in ("1-2", "named", "skipped")]
        assert report.already_marked == ["test_pairs[marked]"]
        assert "pytest.param(1, 2, marks=pytest.mark.slow)," in text
        assert 'pytest.param(3, 4, id="named", marks=pytest.mark.slow),' in text
        assert 'marks=[pytest.mark.skipif(False, reason="x"), pytest.mark.slow],' in text
        assert 'pytest.param(7, 8, id="marked", marks=pytest.mark.integration),' in text
        assert "(9, 10)," in text
        compile(text, "test_a.py", "exec")

    def test_a_tuple_given_as_one_value_is_wrapped_whole(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PARAMS)
        text, _ = tag(
            script,
            tmp_path,
            "tests/test_a.py",
            ["tests/test_a.py::test_tuples_as_one_value[value1]"],
            PARAM_COLLECTED,
        )
        assert "[(1, 2), pytest.param((3, 4), marks=pytest.mark.slow)]" in text

    def test_every_case_listed_marks_the_function(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PARAMS)
        ids = [f"tests/test_a.py::test_numbers[{n}]" for n in (1, 2, 3)]
        text, report = tag(script, tmp_path, "tests/test_a.py", ids, PARAM_COLLECTED)
        assert report.functions == ["test_numbers"] and report.cases == []
        assert "@pytest.mark.slow\n@pytest.mark.parametrize" in text

    @pytest.mark.parametrize(
        ("test", "case", "why"),
        [
            ("test_from_a_name", "x", "not a literal list"),
            ("test_stacked", "3-1", "2 parametrize decorators"),
            ("test_with_a_fixture", "p", "0 parametrize decorators"),
        ],
    )
    def test_a_parametrisation_that_cannot_be_followed_marks_the_whole_function(
        self, script, tmp_path, test, case, why
    ):
        write(tmp_path, "tests/test_a.py", PARAMS)
        text, report = tag(
            script,
            tmp_path,
            "tests/test_a.py",
            [f"tests/test_a.py::{test}[{case}]"],
            PARAM_COLLECTED,
        )
        assert report.functions == [test] and report.cases == []
        assert report.whole_function_fallbacks[0][0] == test
        assert why in report.whole_function_fallbacks[0][1]
        assert text.count("@pytest.mark.slow\n") == 1

    def test_without_the_collected_list_the_whole_function_is_marked(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PARAMS)
        _, report = tag(script, tmp_path, "tests/test_a.py", ["tests/test_a.py::test_numbers[2]"])
        assert report.functions == ["test_numbers"]
        assert "--collected" in report.whole_function_fallbacks[0][1]

    def test_a_second_run_changes_nothing(self, script, tmp_path):
        write(tmp_path, "tests/test_a.py", PARAMS)
        ids = ["tests/test_a.py::test_numbers[2]", "tests/test_a.py::test_pairs[1-2]"]
        first, _ = tag(script, tmp_path, "tests/test_a.py", ids, PARAM_COLLECTED)
        second, report = tag(script, tmp_path, "tests/test_a.py", ids, PARAM_COLLECTED)
        assert second == first and not report.changed


class TestCommandLine:
    def test_dry_run_writes_nothing_and_says_what_it_would_do(self, script, tmp_path, capsys):
        path = write(tmp_path, "tests/test_a.py", PLAIN)
        ids = write(tmp_path, "slow.txt", "tests/test_a.py::test_fast\t1.5\n# note\n")
        status = script.main([str(ids), "--root", str(tmp_path), "--dry-run"])
        assert status == 0
        assert path.read_text(encoding="utf-8") == textwrap.dedent(PLAIN).lstrip("\n")
        out = capsys.readouterr().out
        assert "would tag tests/test_a.py: 1 functions" in out
        assert "would tag 1 functions, 0 classes and 0 parametrised cases in 1 files" in out

    def test_a_run_writes_the_file_and_an_unknown_test_fails_the_status(
        self, script, tmp_path, capsys
    ):
        path = write(tmp_path, "tests/test_a.py", PLAIN)
        ids = write(
            tmp_path, "slow.txt", "tests/test_a.py::test_fast\ntests/test_a.py::test_gone\n"
        )
        status = script.main([str(ids), "--root", str(tmp_path)])
        assert status == 2
        assert "@pytest.mark.slow\ndef test_fast" in path.read_text(encoding="utf-8")
        assert "UNRESOLVED tests/test_a.py::test_gone" in capsys.readouterr().out

    def test_a_missing_file_and_a_syntax_error_are_reported_and_left_alone(
        self, script, tmp_path, capsys
    ):
        broken = write(tmp_path, "tests/test_b.py", "def test_b(:\n")
        ids = write(tmp_path, "slow.txt", "tests/test_b.py::test_b\ntests/test_none.py::test_c\n")
        assert script.main([str(ids), "--root", str(tmp_path)]) == 2
        out = capsys.readouterr().out
        assert "ERROR tests/test_b.py: cannot parse" in out
        assert "ERROR tests/test_none.py: no such file" in out
        assert broken.read_text(encoding="utf-8") == "def test_b(:\n"

    def test_the_collected_list_comes_from_a_file(self, script, tmp_path, capsys):
        path = write(tmp_path, "tests/test_a.py", PARAMS)
        collected = write(tmp_path, "collected.txt", "\n".join(PARAM_COLLECTED))
        ids = write(tmp_path, "slow.txt", "tests/test_a.py::test_numbers[3]\n")
        status = script.main([str(ids), "--root", str(tmp_path), "--collected", str(collected)])
        assert status == 0
        assert "[1, 2, pytest.param(3, marks=pytest.mark.slow)]" in path.read_text(encoding="utf-8")


class TestPytestSelectsWhatWasMarked:
    def test_the_marked_tests_are_exactly_the_slow_ones(self, script, pytester):
        pytester.makeini("[pytest]\nmarkers =\n    slow: slow\n    integration: integration\n")
        path = pytester.makepyfile(
            test_mix=textwrap.dedent(PLAIN).replace("def test_fast", "def test_slow_one")
            + textwrap.dedent(
                """

                @pytest.mark.parametrize("n", [1, 2, 3])
                def test_numbers(n):
                    pass
                """
            )
        )
        listing = pytester.runpytest("--collect-only", "-q")
        collected = [line for line in listing.outlines if "::" in line]
        assert len(collected) == 8
        ids = [
            "test_mix.py::test_slow_one",
            "test_mix.py::TestA::test_two",
            "test_mix.py::test_numbers[3]",
        ]
        reports = script.tag_tests(
            [script.parse_node_id(i) for i in ids],
            Path(path.parent),
            script.read_collected("\n".join(collected)),
        )
        assert sum(r.changed for r in reports) == 1
        everything = pytester.runpytest("--collect-only", "-q", "--strict-markers")
        assert len([line for line in everything.outlines if "::" in line]) == 8
        slow = pytester.runpytest("--collect-only", "-q", "-m", "slow", "--strict-markers")
        assert sorted(line for line in slow.outlines if "::" in line) == [
            "test_mix.py::TestA::test_two",
            "test_mix.py::test_numbers[3]",
            "test_mix.py::test_slow_one",
        ]
        everyday = pytester.runpytest("-q", "-m", "not slow", "--strict-markers")
        everyday.assert_outcomes(passed=5)
