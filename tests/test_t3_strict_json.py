"""Strict JSON export of a pose-set result: ``nonfinite="null"`` (#220).

The default ``nonfinite="keep"`` writes NaN and infinity as the tokens ``NaN``, ``Infinity`` and
``-Infinity``; the text is the one the writers always produced. ``"null"`` writes ``null`` in
their place, lists the paths in ``nonfinite_fields`` and gives text that a strict parser
accepts. Every test of the null text parses it with a ``parse_constant`` that raises.
"""

import json
import math

import numpy as np
import pytest

from binding_metrics.metrics.registry import METRICS_BY_NAME, MetricSpec
from binding_metrics.poses.report import write_pose_report
from binding_metrics.poses.score import PoseSetResult, score_pose_set
from binding_metrics.testing import make_synthetic_poses
from tests.test_t3_support import blank, read, strict_loads, value_at

pytest.importorskip("biotite.structure")


# ---------------------------------------------------------------------------
# the shared helper
# ---------------------------------------------------------------------------


class TestHelper:
    def test_the_copy_has_none_where_a_value_is_not_finite(self):
        from binding_metrics._jsonutil import null_nonfinite

        original = {
            "a": {"b": [1.0, float("nan"), {"c": np.float32("inf")}]},
            "d": (float("-inf"), 2),
            "e": np.array([[1.0, np.nan]]),
            "f": np.float64("nan"),
            "g": "NaN",
            "h": None,
            "i": 3,
        }
        copy, paths = null_nonfinite(original)
        assert paths == ["a.b[1]", "a.b[2].c", "d[0]", "e[0][1]", "f"]
        assert copy == {
            "a": {"b": [1.0, None, {"c": None}]},
            "d": [None, 2],
            "e": [[1.0, None]],
            "f": None,
            "g": "NaN",
            "h": None,
            "i": 3,
        }
        assert math.isnan(original["a"]["b"][1]) and isinstance(original["d"], tuple)

    def test_the_paths_of_the_copy_are_the_paths_of_the_original(self):
        from binding_metrics._jsonutil import nonfinite_paths, null_nonfinite

        value = {"x": [float("inf"), {"y": np.array([np.nan, 1.0])}], "z": 1.5}
        assert nonfinite_paths(value) == null_nonfinite(value)[1] == ["x[0]", "x[1].y[0]"]
        assert nonfinite_paths([float("nan")]) == ["[0]"]
        assert nonfinite_paths({"a": 1, "b": "x"}) == []

    def test_the_two_modes_and_an_error_that_names_them(self):
        from binding_metrics._jsonutil import check_nonfinite

        assert check_nonfinite("keep") == "keep" and check_nonfinite("null") == "null"
        with pytest.raises(ValueError, match=r"keep.*null.*'drop'"):
            check_nonfinite("drop")

    def test_the_text_lists_the_fields_and_adds_the_list_when_it_is_absent(self):
        from binding_metrics._jsonutil import strict_json_text

        assert strict_loads(strict_json_text({"x": float("nan"), "y": 1.0})) == {
            "x": None,
            "y": 1.0,
            "nonfinite_fields": ["x"],
        }
        assert strict_loads(strict_json_text({"y": 1.0}))["nonfinite_fields"] == []

    def test_the_listed_fields_stay_and_no_field_is_listed_twice(self):
        from binding_metrics._jsonutil import strict_json_text

        data = {"old": None, "x": float("inf"), "nonfinite_fields": ["old", "x"]}
        assert strict_loads(strict_json_text(data))["nonfinite_fields"] == ["old", "x"]
        data = {"old": None, "x": float("inf"), "nonfinite_fields": ["old"]}
        assert strict_loads(strict_json_text(data))["nonfinite_fields"] == ["old", "x"]

    def test_a_value_made_by_default_cannot_bring_a_token_back(self):
        from binding_metrics._jsonutil import strict_json_text

        with pytest.raises(ValueError):
            strict_json_text({"x": object()}, default=lambda obj: float("nan"))

    def test_the_keywords_of_json_dumps_pass_through(self):
        from binding_metrics._jsonutil import strict_json_text

        text = strict_json_text({"name": "é", "x": float("nan")}, indent=None, ensure_ascii=False)
        assert text == '{"name": "é", "x": null, "nonfinite_fields": ["x"]}'


# ---------------------------------------------------------------------------
# the pose-set result and its report writer
# ---------------------------------------------------------------------------


def _stub_mixed(cif_path, binder_chain=None, receptor_chain=None):
    return {
        "nan": float("nan"),
        "up": float("inf"),
        "down": float("-inf"),
        "ok": 2.5,
        "series": [1.0, float("nan"), 3.0],
    }


@pytest.fixture(scope="module")
def nan_result():
    spec = MetricSpec(
        name="stub_mixed",
        import_path=f"{__name__}:_stub_mixed",
        description="stub metric of the strict JSON tests",
        input_type="static_structure",
        chain_mode="interface",
        peptide_chain_arg="binder_chain",
        receptor_chain_arg="receptor_chain",
        cost_class="static",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(METRICS_BY_NAME, "stub_mixed", spec)
        return score_pose_set(make_synthetic_poses().poses[:2], "B", "A", metrics=["stub_mixed"])


class TestPoseSetJson:
    def test_the_result_has_the_nonfinite_values_to_export(self, nan_result):
        listed = nan_result["nonfinite_fields"]
        for name in ("nan", "up", "down", "series[1]"):
            assert f"poses[0].stub_mixed.{name}" in listed
        assert len(listed) >= 8

    @pytest.mark.parametrize("kwargs", [{}, {"indent": None}, {"include_timing": False}])
    def test_keep_is_the_text_json_dumps_gives(self, nan_result, kwargs):
        data = nan_result.to_dict()
        if kwargs.get("include_timing") is False:
            data.pop("timing")
        expected = json.dumps(data, indent=kwargs.get("indent", 2), ensure_ascii=False)
        assert nan_result.to_json(**kwargs) == expected
        assert nan_result.to_json(nonfinite="keep", **kwargs) == expected
        assert "NaN" in expected and "Infinity" in expected and "-Infinity" in expected

    def test_null_gives_text_that_a_strict_parser_reads(self, nan_result):
        text = nan_result.to_json(nonfinite="null")
        document = strict_loads(text)
        listed = nan_result["nonfinite_fields"]
        assert document["nonfinite_fields"] == listed
        for path in listed:
            assert value_at(document, path) is None
        assert value_at(document, "poses[0].stub_mixed.ok") == 2.5

    def test_null_changes_no_other_value(self, nan_result):
        kept = json.loads(nan_result.to_json())
        assert strict_loads(nan_result.to_json(nonfinite="null")) == blank(kept)

    def test_the_options_of_to_json_work_in_null_mode(self, nan_result):
        one_line = nan_result.to_json(indent=None, include_timing=False, nonfinite="null")
        assert "\n" not in one_line
        document = strict_loads(one_line)
        assert "timing" not in document and document["nonfinite_fields"]

    def test_the_result_itself_is_not_changed(self, nan_result):
        before = nan_result.to_json()
        nan_result.to_json(nonfinite="null")
        assert nan_result.to_json() == before
        assert math.isnan(nan_result["poses"][0]["stub_mixed"]["nan"])

    def test_the_text_reads_back_as_a_result(self, nan_result):
        text = nan_result.to_json(nonfinite="null")
        loaded = PoseSetResult.from_dict(strict_loads(text))
        assert loaded["nonfinite_fields"] == nan_result["nonfinite_fields"]
        assert loaded["poses"][0]["stub_mixed"]["nan"] is None
        assert loaded.to_json(nonfinite="null") == text

    def test_a_clean_result_has_the_same_text_in_both_modes(self, nan_result):
        clean = PoseSetResult.from_dict(strict_loads(nan_result.to_json(nonfinite="null")))
        assert clean.to_json(nonfinite="null") == clean.to_json()

    def test_another_value_is_an_error_that_names_the_two(self, nan_result):
        with pytest.raises(ValueError, match=r"keep.*null"):
            nan_result.to_json(nonfinite="drop")


class TestPoseReport:
    def test_the_default_json_file_is_unchanged(self, nan_result, tmp_path):
        path = write_pose_report(nan_result, "json", tmp_path / "r.json")
        expected = json.dumps(nan_result.to_dict(), indent=2, ensure_ascii=False) + "\n"
        assert read(path) == expected
        explicit = write_pose_report(nan_result, "json", tmp_path / "k.json", nonfinite="keep")
        assert read(explicit) == expected

    def test_null_writes_strict_json(self, nan_result, tmp_path):
        path = write_pose_report(nan_result, "json", tmp_path / "r.json", nonfinite="null")
        text = read(path)
        assert text == nan_result.to_json(nonfinite="null") + "\n"
        document = strict_loads(text)
        assert document["nonfinite_fields"] == nan_result["nonfinite_fields"]
        assert value_at(document, "poses[1].stub_mixed.up") is None

    def test_a_dictionary_read_back_from_the_json_is_accepted(self, nan_result, tmp_path):
        path = write_pose_report(
            json.loads(nan_result.to_json()), "json", tmp_path / "r.json", nonfinite="null"
        )
        assert strict_loads(read(path))["nonfinite_fields"] == nan_result["nonfinite_fields"]

    @pytest.mark.parametrize("fmt", ["csv", "md", "html"])
    def test_the_other_formats_do_not_depend_on_the_option(self, nan_result, tmp_path, fmt):
        kept = write_pose_report(nan_result, fmt, tmp_path / f"keep.{fmt}")
        nulled = write_pose_report(nan_result, fmt, tmp_path / f"null.{fmt}", nonfinite="null")
        assert read(kept) == read(nulled)

    @pytest.mark.parametrize("fmt", ["json", "csv"])
    def test_another_value_writes_nothing(self, nan_result, tmp_path, fmt):
        with pytest.raises(ValueError, match=r"keep.*null"):
            write_pose_report(nan_result, fmt, tmp_path / "r.out", nonfinite="drop")
        assert not (tmp_path / "r.out").exists()
