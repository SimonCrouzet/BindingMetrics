"""What docs/poses.md, docs/metrics.md and the README say about poses is true."""

import inspect
from pathlib import Path

import pytest

from binding_metrics.poses import DockedPose, ExperimentalPose, Pose, PredictedPose
from binding_metrics.poses.pose import POSE_KINDS
from binding_metrics.predictors._renamed import MESSAGE
from binding_metrics.predictors.record import _NATIVE_SCALARS

ROOT = Path(__file__).resolve().parents[2]
POSES = (ROOT / "docs" / "poses.md").read_text(encoding="utf-8")
METRICS = (ROOT / "docs" / "metrics.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _signature_text(function, name: str) -> str:
    """``name(path, *, a=None, ..., **provenance)`` as the docs write it."""
    parts, keyword_only = [], False
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{parameter.name}")
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_only:
            parts.append("*")
            keyword_only = True
        default = "" if parameter.default is inspect.Parameter.empty else f"={parameter.default!r}"
        parts.append(f"{parameter.name}{default}")
    return f"{name}({', '.join(parts)})"


@pytest.mark.parametrize("cls", [DockedPose, ExperimentalPose])
def test_the_constructor_signatures_in_the_docs_are_the_real_ones(cls):
    text = _signature_text(cls.from_file, f"{cls.__name__}.from_file")
    assert text in POSES, text


@pytest.mark.parametrize("cls", [DockedPose, ExperimentalPose])
def test_from_atoms_takes_the_keywords_of_from_file(cls):
    keywords = [p for p in inspect.signature(cls.from_file).parameters][1:]
    assert [p for p in inspect.signature(cls.from_atoms).parameters][1:] == keywords


def test_the_kinds_the_docs_name_are_the_kinds_of_the_classes():
    assert set(POSE_KINDS) == {"predicted", "docked", "experimental"}
    for cls in (PredictedPose, DockedPose, ExperimentalPose):
        assert issubclass(cls, Pose)
        assert cls.kind in POSE_KINDS
        assert f'"{cls.kind}"' in POSES


def test_the_native_scores_the_docs_name_are_the_ones_of_a_predicted_pose():
    for name in _NATIVE_SCALARS:
        assert f"`{name}`" in POSES and f"`{name}`" in METRICS
    assert "chain_ptm.<chain>" in POSES and "chain_pair_iptm.<pair key>" in POSES


def test_the_members_of_every_pose_are_documented():
    for member in ("kind", "id", "structure_path", "atoms()", "chain_map", "provenance"):
        assert f"| `{member}` |" in POSES, member
    for member in ("content_hash()", "native_scores()"):
        assert f"| `{member}` |" in POSES, member
        assert callable(getattr(Pose, member[:-2]))


def test_the_deprecation_message_is_the_one_the_code_emits():
    assert MESSAGE in POSES


def test_the_old_name_is_always_given_with_the_new_one():
    """A line of the documentation that names ``PredictionRecord`` also names ``PredictedPose``."""
    for label, text in (("poses.md", POSES), ("metrics.md", METRICS), ("README.md", README)):
        for line in text.splitlines():
            if "PredictionRecord" in line and not line.startswith("#"):
                assert "PredictedPose" in line, (label, line[:80])
