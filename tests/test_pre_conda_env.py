"""``--openfold-conda-env ""`` means the current environment, as its help says (#106)."""

from __future__ import annotations

import pytest

from binding_metrics.metrics import openfold


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(openfold, "_run_openfold_command", lambda cmd, out: calls.append(list(cmd)))
    monkeypatch.setattr(
        "binding_metrics.metrics._openfold_run._drop_removed_presets",
        lambda presets, conda_env=None: presets,
    )
    return calls


@pytest.fixture
def on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/run_openfold")


class TestAnEmptyEnvironmentIsTheCurrentOne:
    def test_the_command_is_not_wrapped_in_conda_run(self, tmp_path, recorded, on_path):
        openfold.run_openfold(tmp_path / "q.json", tmp_path / "o", conda_env="")
        (cmd,) = recorded
        assert cmd[0] == "run_openfold" and "conda" not in cmd and "-n" not in cmd

    def test_it_is_the_same_command_as_none(self, tmp_path, recorded, on_path):
        openfold.run_openfold(tmp_path / "q.json", tmp_path / "a", conda_env="")
        openfold.run_openfold(tmp_path / "q.json", tmp_path / "b", conda_env=None)
        first, second = recorded
        assert [a.replace("/a", "/b") for a in first] == second

    def test_the_path_check_applies_as_it_does_for_none(self, tmp_path, recorded, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="run_openfold not found on PATH"):
            openfold.run_openfold(tmp_path / "q.json", tmp_path / "o", conda_env="")
        assert recorded == []

    def test_a_named_environment_still_goes_through_conda_run(self, tmp_path, recorded):
        openfold.run_openfold(tmp_path / "q.json", tmp_path / "o", conda_env="of3")
        (cmd,) = recorded
        assert cmd[:6] == ["conda", "run", "-n", "of3", "--no-capture-output", "run_openfold"]

    def test_the_runner_yaml_writer_gets_no_environment(
        self, tmp_path, recorded, on_path, monkeypatch
    ):
        seen = {}
        original = openfold._write_runner_yaml

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(openfold, "_write_runner_yaml", spy)
        openfold.run_openfold(tmp_path / "q.json", tmp_path / "o", conda_env="")
        assert seen["conda_env"] is None
