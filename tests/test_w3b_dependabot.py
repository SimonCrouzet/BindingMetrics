"""Dependabot config: GitHub Actions, monthly, all updates in one PR."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIG = Path(__file__).parent.parent / ".github" / "dependabot.yml"


@pytest.fixture(scope="module")
def actions_entry():
    if not CONFIG.exists():
        pytest.skip("no .github/dependabot.yml in this checkout")
    config = yaml.safe_load(CONFIG.read_text())
    (entry,) = [u for u in config["updates"] if u["package-ecosystem"] == "github-actions"]
    return entry


def test_runs_monthly(actions_entry):
    assert actions_entry["schedule"]["interval"] == "monthly"


def test_every_update_goes_into_one_group(actions_entry):
    groups = actions_entry["groups"]
    assert len(groups) == 1
    (group,) = groups.values()
    assert group["patterns"] == ["*"]
    # no update-types filter: minor, patch and major bumps share the PR
    assert "update-types" not in group


def test_only_github_actions_are_tracked():
    config = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {"updates": []}
    assert {u["package-ecosystem"] for u in config["updates"]} <= {"github-actions"}
