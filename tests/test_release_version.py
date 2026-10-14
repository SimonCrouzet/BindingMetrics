"""The version of pyproject.toml and the latest release heading of the changelog agree."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_HEADING = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$", re.M
)


def _pyproject_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _changelog() -> str:
    return (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_the_latest_release_heading_is_the_pyproject_version():
    headings = RELEASE_HEADING.findall(_changelog())
    assert headings, "CHANGELOG.md has no '## [x.y.z] - date' release heading"
    latest_version, _date = headings[0]
    assert latest_version == _pyproject_version(), (
        f"pyproject.toml says {_pyproject_version()} but the latest release heading of "
        f"CHANGELOG.md is {latest_version}: bump both in one commit"
    )


def test_the_unreleased_section_comes_before_the_releases():
    text = _changelog()
    unreleased = text.find("## [Unreleased]")
    first_release = RELEASE_HEADING.search(text)
    assert unreleased != -1, "CHANGELOG.md lost its '## [Unreleased]' section"
    assert first_release is not None and unreleased < first_release.start()


def test_release_versions_do_not_go_backwards():
    versions = [
        tuple(int(part) for part in v.split(".")) for v, _ in RELEASE_HEADING.findall(_changelog())
    ]
    assert versions == sorted(versions, reverse=True)
