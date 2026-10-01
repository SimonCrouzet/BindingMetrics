"""Seeds of an OpenFold3 run (#67).

OpenFold3 0.5.0 does not read a ``seeds`` field of the query JSON. It takes seeds from
``experiment_settings.seeds`` of the runner YAML, or generates them from ``--num_model_seeds``,
and that option replaces the YAML value. The tests below check the files and the command line
that the toolkit hands to OpenFold3 through a stub process; what OpenFold3 does with them rests
on its source (``experiment_runner.py`` and ``validator.py`` at tag v0.5.0), not on a run.
"""

import builtins
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_run, openfold
from binding_metrics.predictors.of3_runner import OpenFold3Runner

_P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
_LOGGER = "binding_metrics.metrics._openfold_run"


def _load_yaml(path: Path):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _openfold3_environment(tmp_path, monkeypatch):
    """OpenFold3 0.5.0 on this machine, and no user-default runner.yml."""
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "openfold_cache"))
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )


@pytest.fixture
def commands(monkeypatch):
    """The command lines that ``run_openfold`` would start, without starting them."""
    calls = []
    monkeypatch.setattr(
        openfold, "_run_openfold_command", lambda cmd, output_dir: calls.append(list(cmd))
    )
    return calls


def _yaml_path(command) -> Path:
    (option,) = [a for a in command if a.startswith("--runner_yaml=")]
    return Path(option.split("=", 1)[1])


def _seed_options(command) -> list[str]:
    return [a for a in command if a.startswith(("--num_model_seeds", "--num-model-seeds"))]


def _run(tmp_path, **kwargs):
    return openfold.run_openfold(tmp_path / "q.json", tmp_path / "out", conda_env="of3", **kwargs)


class TestGeneratedRunnerYaml:
    def test_the_seeds_are_written_to_experiment_settings(self, tmp_path):
        path = _openfold_run._write_runner_yaml(tmp_path, ["predict"], seeds=(7, 8))
        assert _load_yaml(path)["experiment_settings"] == {"seeds": [7, 8]}

    def test_without_seeds_the_file_has_no_experiment_settings(self, tmp_path):
        path = _openfold_run._write_runner_yaml(tmp_path, ["predict"])
        assert "experiment_settings" not in _load_yaml(path)

    def test_the_manual_writer_fallback_matches(self, tmp_path, monkeypatch):
        with_yaml = _load_yaml(
            _openfold_run._write_runner_yaml(tmp_path, ["predict"], seeds=[7, 8])
        )
        real_import = builtins.__import__

        def _no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        (tmp_path / "manual").mkdir()
        monkeypatch.setattr(builtins, "__import__", _no_yaml)
        manual = _openfold_run._write_runner_yaml(tmp_path / "manual", ["predict"], seeds=[7, 8])
        monkeypatch.undo()
        assert _load_yaml(manual) == with_yaml

    def test_a_seed_above_32_bits_survives(self, tmp_path):
        path = _openfold_run._write_runner_yaml(tmp_path, ["predict"], seeds=[2746317213])
        assert _load_yaml(path)["experiment_settings"] == {"seeds": [2746317213]}

    def test_the_default_run_pins_the_documented_seed_42(self, tmp_path, commands):
        _run(tmp_path)
        (command,) = commands
        assert _load_yaml(_yaml_path(command))["experiment_settings"] == {"seeds": [42]}
        assert _seed_options(command) == []

    def test_explicit_seeds_reach_the_yaml_and_the_command_has_no_seed_option(
        self, tmp_path, commands
    ):
        _run(tmp_path, seeds=(3, 5))
        (command,) = commands
        assert _load_yaml(_yaml_path(command))["experiment_settings"] == {"seeds": [3, 5]}
        assert _seed_options(command) == []

    def test_num_model_seeds_is_passed_and_no_seed_is_written(self, tmp_path, commands):
        _run(tmp_path, num_model_seeds=3)
        (command,) = commands
        assert _seed_options(command) == ["--num_model_seeds=3"]
        assert "experiment_settings" not in _load_yaml(_yaml_path(command))

    def test_the_seeds_do_not_depend_on_the_templates(self, tmp_path, commands):
        _run(tmp_path, seeds=(3,), template_dir=tmp_path / "templates")
        cfg = _load_yaml(_yaml_path(commands[0]))
        assert cfg["experiment_settings"] == {"seeds": [3]}
        assert cfg["template_preprocessor_settings"]["structure_directory"] == str(
            tmp_path / "templates"
        )


class TestSeedsAndNumModelSeedsExcludeEachOther:
    """OpenFold3 replaces the YAML seeds by the generated ones, so both would lose the first."""

    @pytest.mark.parametrize(
        "extra_args",
        [None, ["--num_model_seeds=2"], ["--num-model-seeds=2"], ["--num_model_seeds", "2"]],
    )
    def test_run_openfold_refuses_both(self, tmp_path, commands, extra_args):
        kwargs = {"num_model_seeds": 2} if extra_args is None else {"extra_args": extra_args}
        with pytest.raises(ValueError, match="cannot be combined") as info:
            _run(tmp_path, seeds=(1,), **kwargs)
        assert "num_model_seeds" in str(info.value) and "replace the seeds" in str(info.value)
        assert commands == []
        assert not (tmp_path / "out" / "runner_config.yaml").exists()

    def test_other_extra_arguments_are_fine(self, tmp_path, commands):
        _run(tmp_path, seeds=(1,), extra_args=["--use_tf32=false"])
        assert _seed_options(commands[0]) == []

    @pytest.mark.parametrize("bad", [(), []])
    def test_empty_seeds_are_refused(self, tmp_path, commands, bad):
        with pytest.raises(ValueError, match="at least one"):
            _run(tmp_path, seeds=bad)
        assert commands == []

    def test_a_string_is_not_a_seed_list(self, tmp_path, commands):
        with pytest.raises(TypeError, match="sequence of integers"):
            _run(tmp_path, seeds="42")

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_generated_count_below_one_is_refused(self, tmp_path, commands, count):
        with pytest.raises(ValueError, match="at least 1"):
            _run(tmp_path, num_model_seeds=count)
        assert commands == []


class TestUsersRunnerYaml:
    """An explicit ``seeds`` wins over the file; without one the file's seeds stay."""

    _USERS = "model_update:\n  presets: [predict]\nexperiment_settings:\n  seeds: [7, 8]\n"

    @pytest.fixture
    def users_yaml(self, tmp_path):
        path = tmp_path / "mine.yml"
        path.write_text(self._USERS, encoding="utf-8")
        return path

    def test_the_users_seeds_are_kept_when_no_seeds_are_given(self, tmp_path, users_yaml, commands):
        _run(tmp_path, runner_yaml=users_yaml)
        assert _yaml_path(commands[0]) == users_yaml  # nothing to add, so nothing is copied
        assert _seed_options(commands[0]) == []

    def test_the_users_seeds_are_kept_in_the_copy_made_for_the_templates(
        self, tmp_path, users_yaml, commands
    ):
        _run(tmp_path, runner_yaml=users_yaml, template_dir=tmp_path / "templates")
        merged = _load_yaml(_yaml_path(commands[0]))
        assert merged["experiment_settings"] == {"seeds": [7, 8]}

    def test_explicit_seeds_replace_the_users(self, tmp_path, users_yaml, commands):
        _run(tmp_path, runner_yaml=users_yaml, seeds=(1, 2, 3))
        used = _yaml_path(commands[0])
        assert used == tmp_path / "out" / "runner_config_merged.yaml"
        merged = _load_yaml(used)
        assert merged["experiment_settings"] == {"seeds": [1, 2, 3]}
        assert merged["model_update"] == {"presets": ["predict"]}
        assert "template_preprocessor_settings" not in merged  # no template directory was given
        assert _seed_options(commands[0]) == []
        assert users_yaml.read_text(encoding="utf-8") == self._USERS  # never modified

    def test_explicit_seeds_are_added_to_a_file_without_experiment_settings(
        self, tmp_path, commands
    ):
        mine = tmp_path / "mine.yml"
        mine.write_text("model_update:\n  presets: [predict]\n", encoding="utf-8")
        _run(tmp_path, runner_yaml=mine, seeds=(9,))
        merged = _load_yaml(_yaml_path(commands[0]))
        assert merged == {
            "model_update": {"presets": ["predict"]},
            "experiment_settings": {"seeds": [9]},
        }

    def test_other_experiment_settings_stay(self, tmp_path, commands):
        mine = tmp_path / "mine.yml"
        mine.write_text(
            "experiment_settings:\n  seeds: [7]\n  skip_existing: true\n", encoding="utf-8"
        )
        _run(tmp_path, runner_yaml=mine, seeds=(9,))
        merged = _load_yaml(_yaml_path(commands[0]))
        assert merged["experiment_settings"] == {"seeds": [9], "skip_existing": True}

    def test_a_copy_that_would_not_change_the_file_is_not_written(
        self, tmp_path, users_yaml, commands
    ):
        _run(tmp_path, runner_yaml=users_yaml, seeds=(7, 8))
        assert _yaml_path(commands[0]) == users_yaml
        assert not (tmp_path / "out" / "runner_config_merged.yaml").exists()

    def test_a_users_structure_directory_is_kept_and_the_seeds_still_apply(
        self, tmp_path, commands, caplog
    ):
        mine = tmp_path / "mine.yml"
        mine.write_text(
            f"template_preprocessor_settings:\n  structure_directory: {tmp_path / 'theirs'}\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            _run(tmp_path, runner_yaml=mine, seeds=(4,), template_dir=tmp_path / "ours")
        merged = _load_yaml(_yaml_path(commands[0]))
        assert merged["experiment_settings"] == {"seeds": [4]}
        assert merged["template_preprocessor_settings"] == {
            "structure_directory": str(tmp_path / "theirs")
        }
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert str(tmp_path / "theirs") in warning.getMessage()
        info = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
        assert "the seeds [4]" in info and "template directory" not in info

    def test_the_copy_is_logged_with_the_seeds(self, tmp_path, users_yaml, commands, caplog):
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            _run(tmp_path, runner_yaml=users_yaml, seeds=(1,))
        (record,) = [r for r in caplog.records if r.levelno == logging.INFO]
        assert "the seeds [1]" in record.getMessage()
        assert str(users_yaml) in record.getMessage()

    def test_num_model_seeds_leaves_the_users_file_alone(self, tmp_path, users_yaml, commands):
        _run(tmp_path, runner_yaml=users_yaml, num_model_seeds=2)
        assert _yaml_path(commands[0]) == users_yaml
        assert _seed_options(commands[0]) == ["--num_model_seeds=2"]

    def test_a_missing_file_with_explicit_seeds_raises_before_the_run(self, tmp_path, commands):
        with pytest.raises(FileNotFoundError, match="missing.yml"):
            _run(tmp_path, runner_yaml=tmp_path / "missing.yml", seeds=(1,))
        assert commands == []

    @pytest.fixture
    def no_pyyaml(self, monkeypatch):
        real_import = builtins.__import__

        def _no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_yaml)
        return monkeypatch

    def test_without_pyyaml_the_seeds_are_appended_as_text(self, tmp_path, commands, no_pyyaml):
        mine = tmp_path / "mine.yml"
        mine.write_text("model_update:\n  presets: [predict]\n", encoding="utf-8")
        _run(tmp_path, runner_yaml=mine, seeds=(5, 6), template_dir=tmp_path / "templates")
        used = _yaml_path(commands[0])
        no_pyyaml.undo()
        merged = _load_yaml(used)
        assert merged["experiment_settings"] == {"seeds": [5, 6]}
        assert merged["template_preprocessor_settings"] == {
            "structure_directory": str(tmp_path / "templates")
        }
        assert merged["model_update"] == {"presets": ["predict"]}

    def test_without_pyyaml_a_file_with_experiment_settings_cannot_take_seeds(
        self, tmp_path, users_yaml, commands, no_pyyaml
    ):
        with pytest.raises(ValueError, match="PyYAML") as info:
            _run(tmp_path, runner_yaml=users_yaml, seeds=(1,))
        assert str(users_yaml) in str(info.value) and "experiment_settings" in str(info.value)
        assert commands == []

    def test_without_pyyaml_the_users_seeds_are_kept_without_explicit_seeds(
        self, tmp_path, users_yaml, commands, no_pyyaml
    ):
        _run(tmp_path, runner_yaml=users_yaml)
        assert _yaml_path(commands[0]) == users_yaml


@pytest.fixture
def stub_conda(tmp_path, monkeypatch):
    """A ``conda`` on PATH that records its arguments and exits 0; returns a reader."""
    record = tmp_path / "argv.json"
    conda = tmp_path / "bin" / "conda"
    conda.parent.mkdir()
    conda.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(record)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)
    monkeypatch.setenv("PATH", f"{conda.parent}{os.pathsep}{os.environ['PATH']}")
    return lambda: json.loads(record.read_text(encoding="utf-8"))


class TestEndToEndWithAStubProcess:
    """The query builder and the run function together, with a fake ``conda run``."""

    @pytest.fixture(autouse=True)
    def _require_gemmi(self):
        pytest.importorskip("gemmi")

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    @pytest.mark.parametrize("seeds, expected", [(None, [42]), ((11, 12), [11, 12])])
    def test_wrappers_write_the_seeds_to_the_yaml_and_not_to_the_query(
        self, tmp_path, stub_conda, runner, seeds, expected
    ):
        out = tmp_path / "out"
        kwargs = {} if seeds is None else {"seeds": seeds}
        getattr(openfold, runner)(_P53, "A", "B", "q", out, conda_env="of3", **kwargs)
        argv = stub_conda()
        assert _seed_options(argv) == []
        assert _load_yaml(_yaml_path(argv))["experiment_settings"] == {"seeds": expected}
        query = json.loads((out / "query" / "q_query.json").read_text(encoding="utf-8"))
        assert "seeds" not in query

    def test_the_batched_run_writes_the_seeds_to_the_yaml(self, tmp_path, stub_conda):
        out = tmp_path / "out"
        sample = openfold._BatchSample("p53", _P53, "A", "B")
        openfold.run_openfold_batched([sample], out, mode="refold", conda_env="of3", seeds=(6,))
        argv = stub_conda()
        assert _seed_options(argv) == []
        assert _load_yaml(_yaml_path(argv))["experiment_settings"] == {"seeds": [6]}
        assert "seeds" not in json.loads((out / "query" / "batch_query.json").read_text("utf-8"))

    def test_num_model_seeds_reaches_the_command_of_a_wrapper(self, tmp_path, stub_conda):
        openfold.run_openfold_scoring(
            _P53, "A", "B", "q", tmp_path / "out", conda_env="of3", num_model_seeds=2
        )
        assert _seed_options(stub_conda()) == ["--num_model_seeds=2"]


class TestRequestKey:
    """The store key follows the seeds OpenFold3 will use."""

    @staticmethod
    def _request(tmp_path, **kwargs):
        complex_file = tmp_path / "c.pdb"
        complex_file.write_bytes(b"ATOM\n")
        return OpenFold3Runner().make_request(
            complex_file, name="q", binder_chain="B", receptor_chain="A", **kwargs
        )

    def test_the_default_request_names_seed_42_and_generates_none(self, tmp_path):
        request = self._request(tmp_path)
        assert request.seeds == (42,)
        assert request.options["num_model_seeds"] is None

    def test_a_changed_seed_changes_the_key(self, tmp_path):
        keys = {
            self._request(tmp_path, seeds=seeds).key() for seeds in ((42,), (7,), (7, 8), (8, 7))
        }
        assert len(keys) == 4

    def test_the_default_key_is_the_key_of_seed_42(self, tmp_path):
        assert self._request(tmp_path).key() == self._request(tmp_path, seeds=(42,)).key()

    def test_generated_seeds_are_another_run_than_the_same_number_of_listed_ones(self, tmp_path):
        generated = self._request(tmp_path, num_model_seeds=1)
        assert generated.seeds == () and generated.options["num_model_seeds"] == 1
        assert generated.key() != self._request(tmp_path).key()
        assert generated.key() != self._request(tmp_path, num_model_seeds=2).key()

    def test_both_options_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be combined"):
            self._request(tmp_path, seeds=(1,), num_model_seeds=2)

    def test_the_run_arguments_follow_the_request(self, tmp_path):
        runner = OpenFold3Runner()
        assert runner._run_arguments(self._request(tmp_path)) == {}
        assert runner._run_arguments(self._request(tmp_path, seeds=(7, 8))) == {"seeds": (7, 8)}
        assert runner._run_arguments(self._request(tmp_path, num_model_seeds=3)) == {
            "num_model_seeds": 3
        }
