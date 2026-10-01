"""Runner-side OpenFold3 behaviour: presets and the runner YAML.

No OpenFold3 install is needed. The tests use temporary files, stub subprocesses and
fake presets; what they cannot show is stated in the docstrings.
"""

import builtins
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_cli, _openfold_run, openfold


def _parse_runner_yaml(path):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestDefaultPresets:
    """``pae_enabled`` is gone from OpenFold3 0.4.1 on; PAE, pTM and ipTM are always written."""

    def test_the_default_is_predict_and_low_mem(self):
        assert _openfold_run._DEFAULT_MODEL_PRESETS == ("predict", "low_mem")

    @pytest.mark.parametrize(
        "command, target",
        [
            ("run", "run_openfold"),
            ("refold", "run_openfold_refolding"),
            ("score", "run_openfold_scoring"),
        ],
    )
    def test_command_line_default_has_no_pae_enabled(self, tmp_path, monkeypatch, command, target):
        seen = {}
        monkeypatch.setattr(openfold, target, lambda **kw: seen.update(kw) or tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(_openfold_cli, "_print_metrics", lambda *a, **kw: None)
        argv = ["prog", command, "--output-dir", str(tmp_path), "--query-name", "q"]
        if command == "run":
            argv += ["--query-json", "q.json"]
        else:
            argv += ["--complex", "c.cif", "--receptor-chain", "A", "--binder-chain", "B"]
        monkeypatch.setattr("sys.argv", argv)
        openfold.main()
        assert seen["model_presets"] == ["predict", "low_mem"]

    def test_help_names_the_default(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "run", "--help"])
        with pytest.raises(SystemExit):
            openfold.main()
        text = " ".join(capsys.readouterr().out.split())
        assert "(default: predict low_mem)" in text
        assert "pae_enabled" not in text

    def test_an_explicit_pae_enabled_still_runs(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: seen.update(kw) or tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(_openfold_cli, "_print_metrics", lambda *a, **kw: None)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "run", "--query-json", "q.json", "--output-dir", str(tmp_path)]
            + ["--query-name", "q", "--presets", "predict", "pae_enabled"],
        )
        openfold.main()
        assert seen["model_presets"] == ["predict", "pae_enabled"]


class TestWriteRunnerYamlPresets:
    @pytest.fixture(autouse=True)
    def _no_openfold3_in_this_interpreter(self, monkeypatch):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: None
        )

    def test_defaults_are_written_without_a_warning(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            path = _openfold_run._write_runner_yaml(
                tmp_path, list(_openfold_run._DEFAULT_MODEL_PRESETS)
            )
        assert _parse_runner_yaml(path) == {"model_update": {"presets": ["predict", "low_mem"]}}

    def test_pae_enabled_is_dropped_with_a_deprecation_warning(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="pae_enabled"):
            path = _openfold_run._write_runner_yaml(tmp_path, ["predict", "pae_enabled", "low_mem"])
        assert _parse_runner_yaml(path)["model_update"]["presets"] == ["predict", "low_mem"]

    def test_the_warning_is_also_logged_so_that_command_line_users_see_it(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger=_openfold_run.logger.name):
            with pytest.warns(DeprecationWarning):
                _openfold_run._write_runner_yaml(tmp_path, ["predict", "pae_enabled"])
        assert "pae_enabled" in caplog.text

    def test_the_input_list_is_not_modified(self, tmp_path):
        presets = ["predict", "pae_enabled"]
        with pytest.warns(DeprecationWarning):
            _openfold_run._write_runner_yaml(tmp_path, presets)
        assert presets == ["predict", "pae_enabled"]

    def test_the_manual_writer_fallback_matches(self, tmp_path, monkeypatch):
        """Without PyYAML the file is written by hand and holds the same presets."""
        import builtins

        real_import = builtins.__import__

        def _no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_yaml)
        with pytest.warns(DeprecationWarning):
            path = _openfold_run._write_runner_yaml(tmp_path, ["predict", "pae_enabled", "low_mem"])
        monkeypatch.undo()
        assert _parse_runner_yaml(path)["model_update"]["presets"] == ["predict", "low_mem"]


class TestVersionAwarePresets:
    """The preset is only needed before 0.4.0, where the PAE head is off by default."""

    def test_an_old_installation_keeps_the_preset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.3.1"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            path = _openfold_run._write_runner_yaml(tmp_path, ["predict", "pae_enabled"])
        assert _parse_runner_yaml(path)["model_update"]["presets"] == ["predict", "pae_enabled"]

    @pytest.mark.parametrize("version", ["0.4.0", "0.4.1", "0.5.0", "0.5.1.dev3"])
    def test_a_current_installation_drops_it(self, tmp_path, monkeypatch, version):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: version
        )
        with pytest.warns(DeprecationWarning):
            path = _openfold_run._write_runner_yaml(tmp_path, ["predict", "pae_enabled"])
        assert _parse_runner_yaml(path)["model_update"]["presets"] == ["predict"]

    @pytest.mark.parametrize(
        "text, expected",
        [("0.5.0", (0, 5, 0)), ("0.4.5.dev12+gabc", (0, 4, 5)), ("1.0", (1, 0)), ("dev", ())],
    )
    def test_version_tuple(self, text, expected):
        assert _openfold_run._version_tuple(text) == expected


class TestInstalledVersion:
    def test_the_current_interpreter_is_asked_without_a_subprocess(self, monkeypatch):
        import importlib.metadata

        monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.5.0")
        monkeypatch.setattr(
            _openfold_run.subprocess, "run", lambda *a, **k: pytest.fail("no process expected")
        )
        assert _openfold_run.installed_openfold3_version() == "0.5.0"

    def test_missing_package_gives_none(self, monkeypatch):
        import importlib.metadata

        def _missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _missing)
        assert _openfold_run.installed_openfold3_version() is None

    def test_another_interpreter_is_asked_through_its_command(self, monkeypatch):
        calls = []

        class _Done:
            returncode = 0
            stdout = "0.5.0\n"

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Done()

        monkeypatch.setattr(_openfold_run.subprocess, "run", _fake_run)
        version = _openfold_run.installed_openfold3_version(["conda", "run", "-n", "of3", "python"])
        assert version == "0.5.0"
        assert calls[0][:5] == ["conda", "run", "-n", "of3", "python"]
        assert calls[0][5] == "-c"

    @pytest.mark.parametrize("failure", ["exit", "missing-binary", "timeout"])
    def test_a_failing_probe_gives_none(self, monkeypatch, failure):
        class _Failed:
            returncode = 1
            stdout = ""

        def _fake_run(cmd, **kwargs):
            if failure == "missing-binary":
                raise FileNotFoundError("conda")
            if failure == "timeout":
                raise _openfold_run.subprocess.TimeoutExpired(cmd, 60)
            return _Failed()

        monkeypatch.setattr(_openfold_run.subprocess, "run", _fake_run)
        assert _openfold_run.installed_openfold3_version(["conda", "run", "python"]) is None


class TestInputsStayOnDisk:
    """OpenFold3 deletes ``structure_directory.parent`` after a run with the MSA server (#69)."""

    def test_toolkit_templates_switch_the_deletion_off(self, tmp_path):
        path = _openfold_run._write_runner_yaml(tmp_path, ["predict"], template_dir=tmp_path / "t")
        cfg = _parse_runner_yaml(path)
        assert cfg["msa_computation_settings"] == {"cleanup_msa_dir": False}
        assert cfg["template_preprocessor_settings"]["structure_directory"] == str(tmp_path / "t")

    def test_without_templates_the_file_is_as_before(self, tmp_path):
        path = _openfold_run._write_runner_yaml(tmp_path, ["predict"])
        assert set(_parse_runner_yaml(path)) == {"model_update"}

    def test_the_manual_writer_fallback_writes_the_same_settings(self, tmp_path, monkeypatch):
        import builtins

        with_yaml = _parse_runner_yaml(
            _openfold_run._write_runner_yaml(tmp_path, ["predict"], template_dir=tmp_path / "t")
        )
        real_import = builtins.__import__

        def _no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        (tmp_path / "manual").mkdir()
        monkeypatch.setattr(builtins, "__import__", _no_yaml)
        manual_path = _openfold_run._write_runner_yaml(
            tmp_path / "manual", ["predict"], template_dir=tmp_path / "t"
        )
        monkeypatch.undo()
        assert _parse_runner_yaml(manual_path) == with_yaml

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_the_yaml_of_a_scoring_or_refolding_run_keeps_the_query_directory(
        self, tmp_path, monkeypatch, runner
    ):
        """End to end with a stub ``conda`` that records its command line and exits 0."""
        p53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
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
        out = tmp_path / "out"
        getattr(openfold, runner)(p53, "A", "B", "q", out, conda_env="of3")

        argv = json.loads(record.read_text(encoding="utf-8"))
        yaml_arg = next(a for a in argv if a.startswith("--runner_yaml="))
        cfg = _parse_runner_yaml(Path(yaml_arg.split("=", 1)[1]))
        assert cfg["template_preprocessor_settings"]["structure_directory"] == str(
            out / "query" / "templates"
        )
        assert cfg["msa_computation_settings"] == {"cleanup_msa_dir": False}
        # the folder OpenFold3 would have removed still holds the inputs
        assert (out / "query" / "q_query.json").exists()
        assert (out / "query" / "templates" / "receptor.cif").exists()


class TestRunnerYamlKeepsTemplates:
    """A runner YAML given by the user still tells OpenFold3 where the templates are (#96).

    A stub replaces the OpenFold3 process, so what is checked is the file the command line names;
    that OpenFold3 then finds the template is not shown here.
    """

    _USERS_YAML = (
        "# settings of the user\n"
        "model_update:\n"
        "  presets:\n"
        "    - predict\n"
        "    - low_mem\n"
        "experiment_settings:\n"
        "  seeds: [7, 8]\n"
    )
    _LOGGER = "binding_metrics.metrics._openfold_run"
    _P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

    @pytest.fixture
    def commands(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            openfold, "_run_openfold_command", lambda cmd, output_dir: calls.append(list(cmd))
        )
        return calls

    @pytest.fixture
    def users_yaml(self, tmp_path):
        path = tmp_path / "mine.yml"
        path.write_text(self._USERS_YAML, encoding="utf-8")
        return path

    @staticmethod
    def _yaml_argument(command) -> Path:
        (option,) = [a for a in command if a.startswith("--runner_yaml=")]
        return Path(option.split("=", 1)[1])

    def _run(self, tmp_path, runner_yaml, template_dir, **kwargs):
        return openfold.run_openfold(
            tmp_path / "q.json",
            tmp_path / "out",
            conda_env="of3",
            runner_yaml=runner_yaml,
            template_dir=template_dir,
            **kwargs,
        )

    def test_the_merged_copy_holds_the_users_keys_and_the_template_directory(
        self, tmp_path, users_yaml, commands
    ):
        templates = tmp_path / "query" / "templates"
        self._run(tmp_path, users_yaml, templates)
        merged = _parse_runner_yaml(self._yaml_argument(commands[0]))
        assert merged == {
            "model_update": {"presets": ["predict", "low_mem"]},
            "experiment_settings": {"seeds": [7, 8]},
            "template_preprocessor_settings": {"structure_directory": str(templates)},
            # without it OpenFold3 deletes the folder that holds the query files
            "msa_computation_settings": {"cleanup_msa_dir": False},
        }

    def test_the_users_file_is_not_modified(self, tmp_path, users_yaml, commands):
        before = users_yaml.read_bytes()
        self._run(tmp_path, users_yaml, tmp_path / "templates")
        assert users_yaml.read_bytes() == before
        assert users_yaml.read_text(encoding="utf-8") == self._USERS_YAML

    def test_the_command_uses_the_merged_copy_in_the_output_directory(
        self, tmp_path, users_yaml, commands
    ):
        self._run(tmp_path, users_yaml, tmp_path / "templates")
        (command,) = commands
        used = self._yaml_argument(command)
        assert used != users_yaml
        assert used.parent == tmp_path / "out" and used.is_file()
        assert str(users_yaml) not in " ".join(command)

    def test_the_merge_is_logged_at_info_with_both_paths(
        self, tmp_path, users_yaml, commands, caplog
    ):
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            self._run(tmp_path, users_yaml, tmp_path / "templates")
        used = self._yaml_argument(commands[0])
        (record,) = [r for r in caplog.records if r.levelno == logging.INFO]
        assert str(used) in record.getMessage() and str(users_yaml) in record.getMessage()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_other_settings_of_the_users_template_section_stay(self, tmp_path, commands):
        mine = tmp_path / "mine.yml"
        mine.write_text(
            "template_preprocessor_settings:\n"
            "  fetch_missing_structures: true\n"
            "  structure_file_format: pdb\n"
            "msa_computation_settings:\n"
            "  cleanup_msa_dir: true\n",
            encoding="utf-8",
        )
        self._run(tmp_path, mine, tmp_path / "templates")
        merged = _parse_runner_yaml(self._yaml_argument(commands[0]))
        assert merged["template_preprocessor_settings"] == {
            "fetch_missing_structures": True,
            "structure_file_format": "pdb",
            "structure_directory": str(tmp_path / "templates"),
        }
        assert merged["msa_computation_settings"] == {"cleanup_msa_dir": True}

    def test_an_empty_file_gives_only_the_template_settings(self, tmp_path, commands):
        mine = tmp_path / "empty.yml"
        mine.write_text("", encoding="utf-8")
        self._run(tmp_path, mine, tmp_path / "templates")
        assert set(_parse_runner_yaml(self._yaml_argument(commands[0]))) == {
            "template_preprocessor_settings",
            "msa_computation_settings",
        }

    def test_a_different_structure_directory_is_kept_with_a_warning(
        self, tmp_path, commands, caplog
    ):
        mine = tmp_path / "mine.yml"
        mine.write_text(
            f"template_preprocessor_settings:\n  structure_directory: {tmp_path / 'theirs'}\n",
            encoding="utf-8",
        )
        ours = tmp_path / "query" / "templates"
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            self._run(tmp_path, mine, ours)
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert str(tmp_path / "theirs") in warning.getMessage()
        assert str(ours) in warning.getMessage()
        # nothing is added, so the user's own file is the one OpenFold3 reads
        assert self._yaml_argument(commands[0]) == mine
        assert not (tmp_path / "out" / "runner_config_merged.yaml").exists()
        assert "structure_directory: " + str(tmp_path / "theirs") in mine.read_text(
            encoding="utf-8"
        )

    def test_the_same_structure_directory_gives_no_warning(self, tmp_path, commands, caplog):
        ours = tmp_path / "query" / "templates"
        mine = tmp_path / "mine.yml"
        mine.write_text(
            f"template_preprocessor_settings:\n  structure_directory: {ours}/../templates\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            self._run(tmp_path, mine, ours)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        merged = _parse_runner_yaml(self._yaml_argument(commands[0]))
        assert merged["template_preprocessor_settings"]["structure_directory"].endswith(
            "/../templates"
        )

    def test_without_a_template_directory_the_users_path_is_passed_unchanged(
        self, tmp_path, users_yaml, commands
    ):
        self._run(tmp_path, users_yaml, None)
        assert self._yaml_argument(commands[0]) == users_yaml
        assert not (tmp_path / "out" / "runner_config_merged.yaml").exists()

    def test_without_a_template_directory_the_file_is_not_even_read(self, tmp_path, commands):
        self._run(tmp_path, tmp_path / "missing.yml", None)
        assert self._yaml_argument(commands[0]) == tmp_path / "missing.yml"

    def test_without_a_runner_yaml_the_generated_file_is_as_before_plus_the_seeds(
        self, tmp_path, commands
    ):
        templates = tmp_path / "templates"
        self._run(tmp_path, None, templates)
        used = self._yaml_argument(commands[0])
        assert used == tmp_path / "out" / "runner_config.yaml"
        assert (
            used.read_bytes()
            == _openfold_run._write_runner_yaml(
                tmp_path, ["predict", "low_mem"], template_dir=templates, seeds=[42]
            ).read_bytes()
        )
        assert not (tmp_path / "out" / "runner_config_merged.yaml").exists()

    @pytest.mark.parametrize(
        "content, message",
        [
            ("model_update: [unclosed\n", "not valid YAML"),
            ("- predict\n- low_mem\n", "mapping at the top level"),
            ("template_preprocessor_settings: [a, b]\n", "must be a mapping"),
        ],
    )
    def test_a_yaml_that_cannot_be_merged_raises_before_the_run(
        self, tmp_path, commands, content, message
    ):
        mine = tmp_path / "bad.yml"
        mine.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=message) as info:
            self._run(tmp_path, mine, tmp_path / "templates")
        assert str(mine) in str(info.value)
        assert commands == []

    def test_a_missing_file_with_templates_raises_before_the_run(self, tmp_path, commands):
        with pytest.raises(FileNotFoundError, match="missing.yml"):
            self._run(tmp_path, tmp_path / "missing.yml", tmp_path / "templates")
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

    def test_without_pyyaml_the_settings_are_appended_as_text(
        self, tmp_path, users_yaml, commands, no_pyyaml
    ):
        templates = tmp_path / 'we"ird: #dir' / "templates"  # needs the JSON quoting
        self._run(tmp_path, users_yaml, templates)
        used = self._yaml_argument(commands[0])
        no_pyyaml.undo()
        merged = _parse_runner_yaml(used)
        assert merged["template_preprocessor_settings"] == {"structure_directory": str(templates)}
        assert merged["msa_computation_settings"] == {"cleanup_msa_dir": False}
        assert merged["model_update"] == {"presets": ["predict", "low_mem"]}
        assert merged["experiment_settings"] == {"seeds": [7, 8]}
        assert used.read_text(encoding="utf-8").startswith(self._USERS_YAML)
        assert users_yaml.read_text(encoding="utf-8") == self._USERS_YAML

    def test_without_pyyaml_a_file_that_sets_the_sections_cannot_be_merged(
        self, tmp_path, commands, no_pyyaml
    ):
        mine = tmp_path / "mine.yml"
        mine.write_text(
            "template_preprocessor_settings:\n  structure_file_format: cif\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="PyYAML") as info:
            self._run(tmp_path, mine, tmp_path / "templates")
        assert str(mine) in str(info.value)
        assert commands == []

    def _stub_conda(self, tmp_path, monkeypatch) -> Path:
        """A ``conda`` on PATH that records its arguments and exits 0."""
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
        return record

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_scoring_and_refolding_find_their_templates_with_a_runner_yaml(
        self, tmp_path, monkeypatch, users_yaml, runner
    ):
        pytest.importorskip("gemmi")
        record = self._stub_conda(tmp_path, monkeypatch)
        out = tmp_path / "out"
        getattr(openfold, runner)(
            self._P53, "A", "B", "q", out, conda_env="of3", runner_yaml=users_yaml
        )
        argv = json.loads(record.read_text(encoding="utf-8"))
        used = self._yaml_argument(argv)
        assert used == out / "predictions" / "runner_config_merged.yaml"
        merged = _parse_runner_yaml(used)
        assert merged["template_preprocessor_settings"]["structure_directory"] == str(
            out / "query" / "templates"
        )
        assert merged["experiment_settings"] == {"seeds": [7, 8]}
        assert (out / "query" / "templates" / "receptor.cif").is_file()
        assert users_yaml.read_text(encoding="utf-8") == self._USERS_YAML

    def test_the_batched_run_finds_its_templates_with_a_runner_yaml(
        self, tmp_path, monkeypatch, users_yaml
    ):
        pytest.importorskip("gemmi")
        record = self._stub_conda(tmp_path, monkeypatch)
        out = tmp_path / "out"
        samples = [
            openfold._BatchSample(
                query_name="p53",
                complex_structure_path=self._P53,
                receptor_chain="A",
                binder_chain="B",
            )
        ]
        openfold.run_openfold_batched(
            samples, out, mode="score", conda_env="of3", runner_yaml=users_yaml
        )
        used = self._yaml_argument(json.loads(record.read_text(encoding="utf-8")))
        assert used == out / "predictions" / "runner_config_merged.yaml"
        merged = _parse_runner_yaml(used)
        assert merged["template_preprocessor_settings"]["structure_directory"] == str(
            out / "query" / "templates"
        )
        assert merged["model_update"] == {"presets": ["predict", "low_mem"]}
        # receptor and binder CIF of the one sample: the files the merged YAML points to
        assert len(list((out / "query" / "templates").glob("*.cif"))) == 2
        assert users_yaml.read_text(encoding="utf-8") == self._USERS_YAML


class TestQueryLayoutDocstrings:
    """The file layout that ``prepare_*_query`` documents is the layout it writes."""

    _P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

    @pytest.mark.parametrize(
        "function, files",
        [
            (
                openfold.prepare_refolding_query,
                ["{query_name}_query.json", "{query_name}_receptor.a3m", "templates/receptor.cif"],
            ),
            (
                openfold.prepare_scoring_query,
                [
                    "{query_name}_query.json",
                    "{query_name}_receptor.a3m",
                    "{query_name}_binder.a3m",
                    "templates/receptor.cif",
                    "templates/binder.cif",
                ],
            ),
        ],
    )
    def test_documented_files_are_written(self, tmp_path, function, files):
        pytest.importorskip("gemmi")
        function(self._P53, "A", "B", "q", tmp_path)
        for name in files:
            assert Path(name).name in function.__doc__  # the docstring draws a tree
            assert (tmp_path / name.replace("{query_name}", "q")).is_file()

    def test_the_refolding_docstring_does_not_send_users_to_a_missing_option(self):
        doc = openfold.prepare_refolding_query.__doc__
        assert "has no ``--template_mmcif_dir`` option" in doc
        assert "Pass::" not in doc


class TestTemplateChainMustExist:
    """A template source without the chain used to give a template CIF with no atoms (#98)."""

    _P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"

    @pytest.fixture(autouse=True)
    def _require_gemmi(self):
        pytest.importorskip("gemmi")

    @pytest.fixture
    def renamed_template(self, tmp_path):
        """A relaxed-looking template whose chains are X and Y instead of A and B."""
        from tests.test_of3_synth import _write

        return _write(tmp_path, {"X": ["ALA", "GLY"], "Y": ["SER", "LYS"]}, "relaxed.pdb")

    def test_the_direct_call_raises_and_writes_nothing(self, tmp_path):
        import gemmi

        st = gemmi.read_structure(str(self._P53))
        out = tmp_path / "t.cif"
        with pytest.raises(ValueError, match="Chain 'Z' not found in template structure"):
            _openfold_run._extract_chain_to_cif(st, "Z", out, sequence="AAAA")
        assert not out.exists()

    def test_the_message_names_the_source_and_the_chains_it_has(self, tmp_path):
        import gemmi

        st = gemmi.read_structure(str(self._P53))
        with pytest.raises(ValueError) as info:
            _openfold_run._extract_chain_to_cif(
                st, "Z", tmp_path / "t.cif", sequence="AAAA", source="relaxed.cif"
            )
        assert "relaxed.cif" in str(info.value)
        assert "chains in its first model: A, B" in str(info.value)

    @pytest.mark.parametrize(
        "function", [openfold.prepare_scoring_query, openfold.prepare_refolding_query]
    )
    def test_prepare_names_the_template_file_and_writes_nothing(
        self, tmp_path, renamed_template, function
    ):
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Chain 'A' not found in template structure") as info:
            function(self._P53, "A", "B", "q", out, template_cif_path=renamed_template)
        assert str(renamed_template) in str(info.value)
        assert not out.exists()

    def test_scoring_checks_the_binder_chain_too(self, tmp_path):
        from tests.test_of3_synth import _write

        template = _write(tmp_path, {"A": ["ALA", "GLY"]}, "only_a.pdb")
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Chain 'B' not found in template structure"):
            openfold.prepare_scoring_query(
                self._P53, "A", "B", "q", out, template_cif_path=template
            )
        assert not out.exists()

    def test_a_template_that_has_the_chains_still_works(self, tmp_path):
        out = tmp_path / "out"
        openfold.prepare_scoring_query(self._P53, "A", "B", "q", out, template_cif_path=self._P53)
        assert (out / "templates" / "receptor.cif").exists()
        assert (out / "templates" / "binder.cif").exists()


class TestManualYamlFallbackQuoting:
    """Without PyYAML the template path is written by hand and must survive YAML rules (#99)."""

    @pytest.mark.parametrize(
        "template_dir",
        [
            "/data/run: 1 #x/templates",
            "/data/plain/templates",
            '/data/quote"d/and\\back/templates',
            "/data/it's here/templates",
            "/data/ends with #",
        ],
    )
    def test_the_path_is_read_back_unchanged(self, tmp_path, monkeypatch, template_dir):
        import builtins

        real_import = builtins.__import__

        def _no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_yaml)
        path = _openfold_run._write_runner_yaml(
            tmp_path, ["predict", "low_mem"], template_dir=Path(template_dir)
        )
        monkeypatch.undo()
        cfg = _parse_runner_yaml(path)
        assert cfg["template_preprocessor_settings"]["structure_directory"] == str(
            Path(template_dir)
        )
        assert cfg["model_update"]["presets"] == ["predict", "low_mem"]
        assert cfg["msa_computation_settings"] == {"cleanup_msa_dir": False}


class TestTemplateChainIdWithUnderscore:
    """OpenFold3 splits ``<entry>_<chain>`` on one underscore, so a template chain ID may not
    contain one; the toolkit refuses such an ID before writing anything (#100)."""

    @pytest.fixture(autouse=True)
    def _require_gemmi(self):
        pytest.importorskip("gemmi")

    @pytest.fixture
    def complex_with_underscore(self, tmp_path):
        from tests.test_of3_synth import _structure

        st = _structure({"A_1": ["ALA", "GLY", "SER"], "B": ["LYS", "ARG"]})
        path = tmp_path / "underscore.cif"
        st.make_mmcif_document().write_file(str(path))
        return path

    @pytest.mark.parametrize(
        "function", [openfold.prepare_scoring_query, openfold.prepare_refolding_query]
    )
    def test_a_receptor_id_with_an_underscore_is_refused_before_any_file(
        self, tmp_path, complex_with_underscore, function
    ):
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="'A_1'.*splits it on one underscore"):
            function(complex_with_underscore, "A_1", "B", "q", out)
        assert not out.exists()

    def test_scoring_refuses_a_binder_id_with_an_underscore_too(self, tmp_path):
        from tests.test_of3_synth import _structure

        path = tmp_path / "c.cif"
        _structure({"A": ["ALA", "GLY"], "B_2": ["SER", "LYS"]}).make_mmcif_document().write_file(
            str(path)
        )
        with pytest.raises(ValueError, match="'B_2'"):
            openfold.prepare_scoring_query(path, "A", "B_2", "q", tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_refolding_leaves_the_binder_id_alone_because_it_has_no_template(self, tmp_path):
        from tests.test_of3_synth import _structure

        path = tmp_path / "c.cif"
        _structure({"A": ["ALA", "GLY"], "B_2": ["SER", "LYS"]}).make_mmcif_document().write_file(
            str(path)
        )
        query = openfold.prepare_refolding_query(path, "A", "B_2", "q", tmp_path / "out")
        chains = json.loads(query.read_text(encoding="utf-8"))["queries"]["q"]["chains"]
        assert chains[1]["chain_ids"] == ["B_2"]

    def test_the_batched_functions_name_the_sample_and_write_nothing(
        self, tmp_path, complex_with_underscore
    ):
        from binding_metrics.metrics._openfold_run import _BatchSample

        samples = [_BatchSample("s1", complex_with_underscore, "A_1", "B")]
        for function in (
            openfold.prepare_batched_scoring_queries,
            openfold.prepare_batched_refolding_queries,
        ):
            out = tmp_path / function.__name__
            with pytest.raises(ValueError, match="'A_1' in sample 's1'"):
                function(samples, out)
            assert not out.exists()

    def test_ids_without_an_underscore_are_unchanged(self, tmp_path):
        p53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"
        query = openfold.prepare_scoring_query(p53, "A", "B", "q", tmp_path)
        a3m = (tmp_path / "q_receptor.a3m").read_text(encoding="utf-8")
        assert ">receptor_A/1-" in a3m
        assert query.exists()

    def test_the_a3m_writer_refuses_it_as_well(self, tmp_path):
        out = tmp_path / "x.a3m"
        with pytest.raises(ValueError, match="underscore"):
            _openfold_run._write_a3m_self_alignment("AAA", "query_A_1", "receptor", "A_1", out)
        assert not out.exists()


class TestPresetVersionFromCondaEnv:
    """With a conda environment the version that decides is the environment's, not ours.

    A stub ``conda`` first on PATH answers the version probe (``conda run -n <env> python -c
    ...``) with ``$FAKE_OF3_VERSION`` and records any other call (the run itself).
    """

    _STUB = (
        "#!{python}\n"
        "import json, os, sys\n"
        "if '-c' in sys.argv:\n"
        "    if os.environ.get('FAKE_OF3_PROBE_EXIT', '0') != '0':\n"
        "        sys.exit(1)\n"
        "    print(os.environ['FAKE_OF3_VERSION'])\n"
        "else:\n"
        "    open(os.environ['FAKE_OF3_RUN_LOG'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )

    @pytest.fixture
    def conda(self, tmp_path, monkeypatch):
        stub = tmp_path / "bin" / "conda"
        stub.parent.mkdir()
        stub.write_text(self._STUB.format(python=sys.executable), encoding="utf-8")
        stub.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stub.parent}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_OF3_RUN_LOG", str(tmp_path / "run.json"))
        monkeypatch.setenv("FAKE_OF3_VERSION", "0.5.0")
        return monkeypatch

    def _write(self, tmp_path, conda_env="of3-old"):
        return _parse_runner_yaml(
            _openfold_run._write_runner_yaml(
                tmp_path, ["predict", "pae_enabled"], conda_env=conda_env
            )
        )["model_update"]["presets"]

    def test_an_environment_with_an_old_openfold3_keeps_the_preset(self, tmp_path, conda):
        conda.setenv("FAKE_OF3_VERSION", "0.3.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._write(tmp_path) == ["predict", "pae_enabled"]

    @pytest.mark.parametrize("version", ["0.4.0", "0.5.0"])
    def test_an_environment_with_a_current_openfold3_drops_it(self, tmp_path, conda, version):
        conda.setenv("FAKE_OF3_VERSION", version)
        with pytest.warns(DeprecationWarning, match="pae_enabled"):
            assert self._write(tmp_path) == ["predict"]

    def test_an_unreadable_environment_counts_as_current(self, tmp_path, conda):
        conda.setenv("FAKE_OF3_PROBE_EXIT", "1")
        with pytest.warns(DeprecationWarning):
            assert self._write(tmp_path) == ["predict"]

    def test_the_current_interpreter_is_not_asked_when_an_environment_is_named(
        self, tmp_path, conda
    ):
        asked = []

        def _version(python_cmd=None):
            asked.append(python_cmd)
            return "0.3.1"

        conda.setattr(_openfold_run, "installed_openfold3_version", _version)
        self._write(tmp_path)
        assert len(asked) == 1
        assert asked[0][1:] == ["run", "-n", "of3-old", "python"]

    def test_without_an_environment_the_current_interpreter_is_asked(self, tmp_path, conda):
        asked = []
        conda.setattr(
            _openfold_run,
            "installed_openfold3_version",
            lambda python_cmd=None: asked.append(python_cmd),
        )
        with pytest.warns(DeprecationWarning):
            self._write(tmp_path, conda_env=None)
        assert asked == [None]

    def test_no_probe_when_the_preset_is_not_named(self, tmp_path, conda):
        conda.setattr(
            _openfold_run,
            "installed_openfold3_version",
            lambda python_cmd=None: pytest.fail("probed"),
        )
        path = _openfold_run._write_runner_yaml(tmp_path, ["predict"], conda_env="of3-old")
        assert _parse_runner_yaml(path)["model_update"]["presets"] == ["predict"]

    @pytest.mark.parametrize("version, kept", [("0.3.1", True), ("0.5.0", False)])
    def test_run_openfold_passes_its_environment_to_the_writer(
        self, tmp_path, conda, version, kept
    ):
        conda.setenv("FAKE_OF3_VERSION", version)
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            openfold.run_openfold(
                "q.json", out, conda_env="of3-old", model_presets=["predict", "pae_enabled"]
            )
        presets = _parse_runner_yaml(out / "runner_config.yaml")["model_update"]["presets"]
        assert ("pae_enabled" in presets) is kept
        run_argv = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert run_argv[:5] == ["run", "-n", "of3-old", "--no-capture-output", "run_openfold"]
