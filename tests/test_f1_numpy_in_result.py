"""``PoseSetResult.to_dict`` holds plain Python values only (issue #188).

``numpy.float64`` is a ``float`` subclass, so the branch of ``_plain`` that keeps ``int`` and
``float`` unchanged also kept the numpy scalars that a metric returned (the per-residue table of
``interface``), although ``to_dict`` documents plain ``dict``, ``list`` and scalar values.
"""

import numpy as np
import pytest

from binding_metrics.poses.score import PoseSetResult, score_pose_set
from binding_metrics.testing import make_synthetic_poses

pytest.importorskip("biotite.structure")

PLAIN_SCALARS = (str, int, float, bool, type(None))


def numpy_objects(value, path="result"):
    """Every path under ``value`` that holds a numpy scalar or array."""
    if isinstance(value, (np.generic, np.ndarray)):
        return [f"{path}: {type(value).__name__}"]
    if isinstance(value, dict):
        return [
            found for key, child in value.items() for found in numpy_objects(child, f"{path}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            found
            for index, child in enumerate(value)
            for found in numpy_objects(child, f"{path}[{index}]")
        ]
    return []


def exact_types(value):
    """The set of the types of the leaves under ``value``, containers excluded."""
    if isinstance(value, dict):
        return set().union(*(exact_types(child) for child in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(exact_types(child) for child in value)) if value else set()
    return {type(value)}


@pytest.fixture(scope="module")
def result():
    return score_pose_set(make_synthetic_poses().poses[:2], "B", "A", metrics=["interface"])


class TestToDict:
    def test_the_interface_table_has_no_numpy_scalar(self, result):
        assert numpy_objects(result.to_dict()) == []

    def test_every_leaf_is_an_exact_python_scalar(self, result):
        assert exact_types(result.to_dict()) <= set(PLAIN_SCALARS)

    def test_the_rows_of_the_result_are_plain_too(self, result):
        assert numpy_objects(dict(result)) == []

    def test_the_json_text_is_unchanged_by_the_conversion(self, result):
        assert PoseSetResult.from_dict(result.to_dict()).to_json() == result.to_json()


class TestNumpyValuesInAResult:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (np.float64(0.25), float),
            (np.float32(0.5), float),
            (np.int64(3), int),
            (np.int8(3), int),
            (np.bool_(True), bool),
            (np.float64("nan"), float),
        ],
    )
    def test_a_numpy_scalar_becomes_the_python_scalar(self, result, value, expected):
        data = PoseSetResult({**result, "extra": {"value": value}}).to_dict()
        assert type(data["extra"]["value"]) is expected

    def test_numpy_values_inside_containers_are_converted(self, result):
        nested = {"a": [np.float64(1.5), {"b": (np.int64(2),)}], "c": np.arange(3.0)}
        data = PoseSetResult({**result, "extra": nested}).to_dict()
        assert numpy_objects(data["extra"]) == []
        assert exact_types(data["extra"]) == {float, int}

    def test_python_values_are_kept(self, result):
        extra = {"x": 1, "y": 2.5, "z": True, "w": None, "s": "t"}
        assert PoseSetResult({**result, "extra": extra}).to_dict()["extra"] == extra
