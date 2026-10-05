"""What docs/poses.md, docs/metrics.md, the README and the CHANGELOG say about poses is true."""

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
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


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
    assert MESSAGE.startswith("PredictionRecord is deprecated and renamed PredictedPose")
    assert "(not before 0.4.0)" in MESSAGE
    assert MESSAGE in POSES
    assert MESSAGE in CHANGELOG


def test_the_pages_point_at_each_other():
    assert "docs/poses.md" in README
    assert "(poses.md)" in METRICS


def test_the_documentation_uses_the_new_name_for_the_class():
    for label, text in (("metrics.md", METRICS), ("README.md", README)):
        for line in text.splitlines():
            if "PredictionRecord" in line:
                assert "former name" in line or "deprecated" in line.lower(), (label, line[:80])


def test_the_changelog_records_the_rename_and_the_deprecation():
    section = CHANGELOG.split("### Deprecated", 1)[1].split("### Fixed", 1)[0]
    assert "`PredictionRecord`" in section and "`PredictedPose`" in section
    assert "### Changed" in CHANGELOG
    assert "`PredictionRecord` is renamed `PredictedPose`" in CHANGELOG
