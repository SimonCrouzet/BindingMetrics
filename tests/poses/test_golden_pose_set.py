"""Golden pose-set results: the live result against the committed files, and the machinery.

The golden files in ``tests/golden/pose_sets/`` are the ``score_pose_set`` result of two fixed
calls (see ``tests/golden/pose_set_cases.py``). A failure here means a number or a key of the
result moved: the message names the key, the old and the new value and the tolerance. If the
change is intended, ``scripts/regen_pose_golden.py --write`` rewrites the files and the commit
carries a regression note.
"""

import copy
import importlib.util
import io
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.poses.score import SCHEMA, SCHEMA_VERSION
from tests.golden import pose_set_cases as golden

pytest.importorskip("biotite.structure")

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "regen_pose_golden.py"

#: Coordinates are stored as float32, so a closed-form RMSD agrees with the poses to this much.
FLOAT32_RMSD_TOLERANCE = 1e-4


def load_script():
    spec = importlib.util.spec_from_file_location("regen_pose_golden", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=sorted(golden.CASES))
def case(request):
    return golden.CASES[request.param]


@pytest.fixture(scope="module")
def live(case):
    return case.run()


@pytest.fixture(scope="module")
def document(case):
    return golden.read_golden(case.path)


# ---------------------------------------------------------------------------
# the live result against the committed file
# ---------------------------------------------------------------------------


class TestAgainstTheCommittedFile:
    def test_the_live_result_equals_the_golden_result_within_the_tolerance(
        self, case, live, document
    ):
        mismatches = golden.compare(document["result"], live.to_dict())
        assert not mismatches, golden.failure_message(case.name, mismatches)

    def test_the_file_records_the_case_and_the_tolerance_that_the_test_uses(self, case, document):
        assert document["golden_format"] == golden.FORMAT_VERSION
        assert document["name"] == case.name
        assert document["description"] == case.description
        assert document["tolerance"] == {
            "rtol": golden.RTOL,
            "atol": golden.ATOL,
            "overrides": golden.TOLERANCE_OVERRIDES,
        }
        assert document["ignored"] == list(golden.IGNORED)
        assert "regen_pose_golden.py --write" in document["regenerate"]

    def test_the_description_names_the_metrics_that_run_and_no_other(self, case, document):
        run = document["result"]["parameters"]["metrics"]
        ending = re.search(r"the metrics? (.+)\.$", case.description)
        assert ending, case.description
        named = re.split(r", | and ", ending.group(1))
        assert named == run

    def test_the_default_tolerance_is_one_in_a_million_of_the_value_and_of_one(self):
        assert (golden.RTOL, golden.ATOL) == (1e-6, 1e-6)

    def test_the_result_is_the_current_schema(self, document):
        result = document["result"]
        assert (result["schema"], result["schema_version"]) == (SCHEMA, SCHEMA_VERSION)

    def test_the_file_holds_nothing_about_the_run_or_the_machine(self, document):
        result = document["result"]
        for key in ("timing", "provenance", "package_version"):
            assert key not in result
        leaves = golden.flatten(result)
        assert not any(golden.is_ignored(path) for path in leaves)
        strings = [value for value in leaves.values() if isinstance(value, str)]
        assert not any(value.startswith("/") or ":\\" in value for value in strings)

    def test_the_file_is_exactly_what_the_script_writes(self, case, document):
        text = case.path.read_text(encoding="utf-8")
        assert text == golden.dumps(document) + "\n"
        assert "\r" not in text

    def test_the_golden_values_are_finite_and_nothing_is_flagged_nonfinite(self, document):
        result = document["result"]
        assert result["nonfinite_fields"] == []
        assert all(
            np.isfinite(value)
            for value in golden.flatten(result).values()
            if isinstance(value, float)
        )


# ---------------------------------------------------------------------------
# the golden values are right, not only what the code gave
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    return golden.read_golden(golden.CASES["synthetic_1ycr"].path)["result"]


@pytest.fixture(scope="module")
def truth():
    return golden.synthetic_poses().truth


class TestSyntheticGoldenAgainstTheConstruction:
    @pytest.mark.parametrize("kind", ["ca", "backbone", "heavy"])
    def test_the_pairwise_rmsd_is_the_closed_form_of_the_construction(self, result, truth, kind):
        block = result["agreement"]["rmsd"][kind]["matrix"]
        assert tuple(block["pose_ids"]) == truth.ids
        matrix = np.array(block["matrix"], dtype=float)
        assert np.abs(matrix - truth.rmsd[kind]).max() < FLOAT32_RMSD_TOLERANCE

    def test_the_offsets_of_the_design_are_in_the_matrix(self, result):
        matrix = result["agreement"]["rmsd"]["ca"]["matrix"]["matrix"]
        within_a = [matrix[0][1], matrix[1][2], matrix[0][2]]
        within_b = [matrix[3][4], matrix[4][5], matrix[3][5]]
        assert within_a == pytest.approx([0.4, 0.4, 0.8], abs=1e-4)
        assert within_b == pytest.approx([0.3, 0.6, 0.9], abs=1e-4)
        assert matrix[2][3] == pytest.approx(1.2, abs=1e-4)  # the nearest pair across back-ends
        assert matrix[0][5] == pytest.approx(2.9, abs=1e-4)

    def test_the_medoids_and_the_clusters_follow_from_the_offsets(self, result):
        assert result["representatives"]["backend_a"] == "backend_a-s2"
        assert result["representatives"]["backend_b"] == "backend_b-s2"
        clusters = result["agreement"]["clusters"]["1.0"]
        assert clusters["clusters"] == [
            ["backend_a-s1", "backend_a-s2", "backend_a-s3"],
            ["backend_b-s1", "backend_b-s2", "backend_b-s3"],
        ]

    def test_every_pose_is_placed_on_the_receptor_of_the_reference(self, result):
        for row in result["poses"]:
            frame = row["frame"]
            assert row["status"] == "ok" and frame["n_paired"] == 85
            assert frame["receptor_rmsd_angstrom"] < 1e-5  # the receptor is not perturbed
            assert frame["region_rmsd_angstrom"]["epitope"] < 1e-5

    def test_the_peptide_shift_sets_the_shared_area_with_the_crystal_peptide(self, result):
        by_id = {row["id"]: row["descriptors"]["reference"] for row in result["poses"]}
        shared = [
            by_id[pose_id]["partners"]["crystal_peptide"]["shared_buried_area_angstrom2"]
            for pose_id in sorted(by_id)
        ]
        # a1 sits on the crystal peptide, and the area they share falls as the offset grows
        assert shared[0] == pytest.approx(662.335, abs=0.5)
        assert shared[0] > shared[1] > shared[2] > shared[3]
        assert shared[3] > shared[4] > shared[5]

    def test_an_interface_residue_set_equal_to_the_interface_is_covered_completely(self, result):
        block = result["poses"][0]["descriptors"]["reference"]["sets"]["crystal_interface"]
        assert (block["n_set"], block["n_overlap"], block["jaccard"]) == (25, 25, 1.0)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


class TestComparison:
    base = {
        "count": 3,
        "flag": True,
        "name": "a",
        "nothing": None,
        "value": 100.0,
        "tiny": 0.0,
        "empty": {},
        "none_here": [],
        "block": {"matrix": [[1.0, 2.0], [3.0, 4.0]], "ids": ["x", "y"]},
        "timing": {"total": 1.0},
        "provenance": {"git_sha": "abc"},
        "package_version": "0.1.0",
        "references": {"variants": {"r": {"structure": {"path": "/somewhere/a.pdb"}}}},
    }

    def changed(self, **edits):
        live = copy.deepcopy(self.base)
        live.update(edits)
        return live

    def test_an_identical_result_has_no_mismatch(self):
        assert golden.compare(self.base, copy.deepcopy(self.base)) == []

    def test_a_float_inside_the_tolerance_agrees(self):
        live = self.changed(value=100.0 * (1 + 0.9e-6), tiny=0.9e-6)
        assert golden.compare(self.base, live) == []

    def test_a_float_beyond_the_tolerance_is_a_mismatch_that_names_the_key(self):
        live = self.changed(value=100.0 * (1 + 1.2e-6) + 1e-5)
        (mismatch,) = golden.compare(self.base, live)
        assert (mismatch.path, mismatch.kind) == ("value", "changed")
        text = mismatch.describe()
        assert text.startswith("value:") and "rtol=1e-06" in text and "atol=1e-06" in text

    def test_a_small_absolute_change_of_a_zero_is_judged_by_the_absolute_tolerance(self):
        assert golden.compare(self.base, self.changed(tiny=2e-6))[0].path == "tiny"

    def test_an_integer_a_string_a_boolean_and_none_must_be_equal(self):
        live = self.changed(count=4, name="b", flag=False, nothing=0)
        assert [m.path for m in golden.compare(self.base, live)] == [
            "count",
            "flag",
            "name",
            "nothing",
        ]

    def test_a_count_that_became_a_float_is_a_type_change(self):
        (mismatch,) = golden.compare(self.base, self.changed(count=3.0))
        assert mismatch.kind == "type" and "type changed" in mismatch.describe()

    def test_a_list_item_is_named_by_its_position(self):
        block = {"matrix": [[1.0, 2.0], [3.0, 4.5]], "ids": ["x", "y"]}
        (mismatch,) = golden.compare(self.base, self.changed(block=block))
        assert mismatch.path == "block.matrix[1][1]"

    def test_a_list_of_another_length_has_a_missing_or_extra_item(self):
        shorter = golden.compare(self.base, self.changed(block={"matrix": [], "ids": ["x"]}))
        assert {(m.path, m.kind) for m in shorter} >= {("block.ids[1]", "missing")}
        longer = golden.compare(
            self.base,
            self.changed(block={"matrix": [[1.0, 2.0], [3.0, 4.0]], "ids": ["x", "y", "z"]}),
        )
        assert [(m.path, m.kind) for m in longer] == [("block.ids[2]", "extra")]

    def test_a_missing_and_an_extra_key_are_both_reported(self):
        live = self.changed()
        del live["name"]
        live["new_key"] = 1
        kinds = {(m.path, m.kind) for m in golden.compare(self.base, live)}
        assert kinds == {("name", "missing"), ("new_key", "extra")}

    def test_an_empty_container_that_gained_content_is_a_mismatch(self):
        kinds = {(m.path, m.kind) for m in golden.compare(self.base, self.changed(empty={"a": 1}))}
        assert kinds == {("empty", "missing"), ("empty.a", "extra")}

    def test_what_describes_the_run_is_ignored(self):
        live = self.changed(
            timing={"total": 99.0},
            provenance={"git_sha": "other", "python": "9"},
            package_version="9.9",
            references={"variants": {"r": {"structure": {"path": "/elsewhere/b.pdb"}}}},
        )
        assert golden.compare(self.base, live) == []

    def test_nan_equals_nan_and_nothing_else(self):
        with_nan = {"v": float("nan")}
        assert golden.compare(with_nan, {"v": float("nan")}) == []
        assert golden.compare(with_nan, {"v": 1.0})[0].kind == "changed"
        assert golden.compare({"v": 1.0}, with_nan)[0].kind == "changed"

    def test_an_override_widens_the_tolerance_of_the_paths_it_names(self, monkeypatch):
        live = self.changed(value=100.5)
        assert golden.compare(self.base, live)[0].path == "value"
        monkeypatch.setitem(golden.TOLERANCE_OVERRIDES, "val*", (1e-2, 0.0))
        assert golden.compare(self.base, live) == []
        assert golden.compare(self.base, self.changed(tiny=1e-3))[0].path == "tiny"

    def test_the_failure_message_names_the_case_the_keys_and_the_way_out(self):
        live = self.changed(value=101.0, count=4)
        mismatches = golden.compare(self.base, live)
        message = golden.failure_message("demo", mismatches)
        assert "golden case 'demo'" in message and "demo.json" in message
        assert "value: golden 100.0, result 101.0" in message
        assert "count: golden 3, result 4" in message
        assert "regen_pose_golden.py --write" in message and "regression note" in message

    def test_a_long_list_of_mismatches_is_cut_and_counted(self):
        many = [golden.Mismatch(f"k{i}", "changed", 1, 2) for i in range(30)]
        message = golden.failure_message("demo", many, limit=5)
        assert "30 value(s)" in message and "and 25 more" in message
        assert message.count("\n  k") == 5


class TestFileFormat:
    def test_floats_are_rounded_to_ten_digits_and_noise_is_written_as_zero(self):
        assert golden.canonical({"a": 1.23456789012345, "b": 3e-15, "c": -2e-13, "d": 5})[
            "a"
        ] == pytest.approx(1.234567890, abs=0)
        out = golden.canonical([3e-15, -2e-13, 5, True, None, "x", float("inf")])
        assert out == [0.0, 0.0, 5, True, None, "x", float("inf")]

    def test_the_text_is_json_and_reads_back_as_the_same_data(self):
        data = {
            "matrix": [[1.0, 0.5], [0.5, 1.0]],
            "ids": ["a", "b"],
            "block": {"n": 1, "reasons": {}, "nested": {"x": [1, 2]}},
            "rows": [{"id": "a", "v": 1.5}, {"id": "b", "v": None}],
            "empty": [],
            "text": "café",
        }
        assert json.loads(golden.dumps(data)) == data

    def test_vectors_and_short_flat_dictionaries_take_one_line(self):
        data = {"v": [1, 2, 3], "m": [[1, 2], [3, 4]], "d": {"a": 1, "b": None}, "rows": [{"i": 1}]}
        assert golden.dumps(data).splitlines() == [
            "{",
            '  "v": [1, 2, 3],',
            '  "m": [[1, 2], [3, 4]],',
            '  "d": {"a": 1, "b": null},',
            '  "rows": [',
            '    {"i": 1}',
            "  ]",
            "}",
        ]
        assert golden.dumps({"v": [1, 2], "d": {"a": 1}}) == '{"v": [1, 2], "d": {"a": 1}}'

    def test_a_long_dictionary_and_a_list_of_dictionaries_take_several_lines(self):
        long = {f"key_{i}": float(i) for i in range(30)}
        assert len(golden.dumps(long).splitlines()) == 32
        rows = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
        assert golden.dumps(rows) == '[\n  {"id": "a", "v": 1},\n  {"id": "b", "v": 2}\n]'

    def test_the_file_is_utf8_with_unix_newlines(self, tmp_path):
        path = tmp_path / "x" / "case.json"
        golden.write_golden(path, {"text": "caf\u00e9", "rows": [{"a": 1}]})
        raw = path.read_bytes()
        expected = '{\n  "text": "caf\u00e9",\n  "rows": [\n    {"a": 1}\n  ]\n}\n'
        assert raw == expected.encode("utf-8") and b"\r" not in raw

    def test_stripping_removes_the_ignored_keys_at_any_depth(self):
        stripped = golden.strip_ignored(TestComparison.base)
        assert set(stripped) == set(TestComparison.base) - {
            "timing",
            "provenance",
            "package_version",
        }
        assert stripped["references"] == {"variants": {"r": {"structure": {}}}}


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------


class TestRegenerationScript:
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=REPO,
        )

    def test_it_refuses_to_run_without_write_and_writes_nothing(self, tmp_path):
        for arguments in ([], ["--golden-dir", str(tmp_path)], ["--case", "synthetic_1ycr"]):
            completed = self.run_cli(*arguments)
            assert completed.returncode == 2
            assert "refusing to run without --write" in completed.stderr
            assert "regression note" in completed.stderr
            assert completed.stdout == ""
        assert list(tmp_path.iterdir()) == []

    def test_help_documents_when_a_regeneration_is_legitimate(self):
        completed = self.run_cli("--help")
        assert completed.returncode == 0
        assert "--write" in completed.stdout and "legitimate" in completed.stdout

    def test_an_unknown_case_is_refused(self):
        completed = self.run_cli("--write", "--case", "nope")
        assert completed.returncode == 2 and "invalid choice" in completed.stderr

    def test_write_builds_a_file_that_agrees_with_the_committed_one(self, tmp_path):
        completed = self.run_cli(
            "--write", "--golden-dir", str(tmp_path), "--case", "descriptors_1ycr"
        )
        assert completed.returncode == 0, completed.stderr
        assert "descriptors_1ycr: new file" in completed.stdout
        assert "legitimate" in completed.stdout
        written = golden.read_golden(tmp_path / "descriptors_1ycr.json")
        committed = golden.read_golden(golden.CASES["descriptors_1ycr"].path)
        assert golden.compare(committed["result"], written["result"]) == []
        assert {path.name for path in tmp_path.iterdir()} == {"descriptors_1ycr.json"}

    def test_a_changed_value_is_summarised_with_its_key_and_the_file_is_rewritten(
        self, case, live, document, tmp_path
    ):
        script = load_script()
        stand_in = replace(case, run=lambda: live)
        edited = copy.deepcopy(document)
        edited["result"]["poses"][0]["frame"]["n_paired"] = 84
        by_id = edited["result"]["poses"][0]
        by_id["frame"]["receptor_rmsd_angstrom"] = 0.5
        golden.write_golden(tmp_path / f"{case.name}.json", edited)

        out = io.StringIO()
        assert script.regenerate(stand_in, tmp_path, out) is True
        text = out.getvalue()
        assert "2 value(s) changed beyond the tolerance, 0 added, 0 removed" in text
        assert "poses[0].frame.n_paired: golden 84, result 85" in text
        assert "poses[*].frame" in text
        assert f"regression note to start from: {case.name}, poses[0].frame" in text
        assert f"rewrote {tmp_path / (case.name + '.json')}" in text
        rewritten = golden.read_golden(tmp_path / f"{case.name}.json")
        assert golden.compare(document["result"], rewritten["result"]) == []

    def test_a_file_that_agrees_within_the_tolerance_is_left_untouched(
        self, case, live, document, tmp_path
    ):
        script = load_script()
        stand_in = replace(case, run=lambda: live)
        nudged = copy.deepcopy(document)
        nudged["result"]["poses"][0]["frame"]["receptor_rmsd_angstrom"] += 4e-7
        path = tmp_path / f"{case.name}.json"
        golden.write_golden(path, nudged)
        before = path.read_bytes()

        out = io.StringIO()
        assert script.regenerate(stand_in, tmp_path, out) is False
        assert "unchanged within the tolerance, file left as it is" in out.getvalue()
        assert path.read_bytes() == before

    def test_a_new_description_rewrites_the_file_even_when_every_value_agrees(
        self, case, live, document, tmp_path
    ):
        script = load_script()
        stand_in = replace(case, run=lambda: live, description="another description")
        path = tmp_path / f"{case.name}.json"
        golden.write_golden(path, document)

        out = io.StringIO()
        assert script.regenerate(stand_in, tmp_path, out) is True
        assert "description or tolerance record changed" in out.getvalue()
        assert golden.read_golden(path)["description"] == "another description"
