"""Runner-side OpenFold3 behaviour: presets and the runner YAML.

No OpenFold3 install is needed. The tests use temporary files, stub subprocesses and
fake presets; what they cannot show is stated in the docstrings.
"""

import json
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
