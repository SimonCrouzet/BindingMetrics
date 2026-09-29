"""``--config``: option defaults from a TOML file for run, batch and relax (issue #32)."""

import argparse
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import add_config_arg, parse_args_with_config

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def _toml(tmp_path, text, name="config.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _parser():
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--md-duration-ps", type=float, default=200.0)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--skip-prep", action="store_true")
    parser.add_argument(
        "--energy-modes", nargs="+", choices=["raw", "relaxed"], default=["relaxed"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--format", choices=["json", "csv"], default="json", dest="fmt")
    parser.add_argument("--peptide-chain", "--binder-chain", type=str, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=int, default=None)
    group.add_argument("--all-models", action="store_true")
    add_config_arg(parser)
    return parser


def _parse(tmp_path, toml, *argv):
    config = _toml(tmp_path, toml)
    return parse_args_with_config(_parser(), ["--config", str(config), *argv])


def _fails(tmp_path, capsys, toml, *argv):
    with pytest.raises(SystemExit) as exit_request:
        _parse(tmp_path, toml, *argv)
    assert exit_request.value.code == 2
    return capsys.readouterr().err


class TestPrecedence:
    def test_file_beats_builtin_default_and_command_line_beats_file(self, tmp_path):
        toml = 'input = "a.cif"\nmd-duration-ps = 100\nph = 6.5\n'
        args = _parse(tmp_path, toml, "--ph", "8")
        assert args.md_duration_ps == 100.0  # from the file
        assert args.ph == 8.0  # command line
        assert args.device == "cuda"  # neither: built-in default
        assert args.input == Path("a.cif")

    def test_without_config_nothing_changes(self):
        args = parse_args_with_config(_parser(), ["--input", "x.cif"])
        assert (args.md_duration_ps, args.ph, args.config) == (200.0, 7.4, None)

    def test_config_equals_form_is_understood(self, tmp_path):
        config = _toml(tmp_path, 'input = "a.cif"\nph = 5\n')
        args = parse_args_with_config(_parser(), [f"--config={config}"])
        assert args.ph == 5.0

    def test_required_option_is_satisfied_by_the_file_only(self, tmp_path, capsys):
        assert _parse(tmp_path, 'input = "a.cif"\n').input == Path("a.cif")
        with pytest.raises(SystemExit):
            _parse(tmp_path, "ph = 5\n")
        assert "--input" in capsys.readouterr().err

    def test_command_line_overrides_a_required_value_from_the_file(self, tmp_path):
        args = _parse(tmp_path, 'input = "a.cif"\n', "-i", "b.cif")
        assert args.input == Path("b.cif")


class TestKeys:
    def test_dashes_and_underscores_are_the_same_key(self, tmp_path):
        args = _parse(tmp_path, 'input = "a.cif"\nmd_duration_ps = 50\nsample_id = "s"\n')
        assert (args.md_duration_ps, args.sample_id) == (50.0, "s")

    def test_key_follows_the_option_name_not_the_dest(self, tmp_path):
        assert _parse(tmp_path, 'input = "a"\nformat = "csv"\n').fmt == "csv"

    def test_alias_spelling_is_accepted(self, tmp_path):
        assert _parse(tmp_path, 'input = "a"\nbinder-chain = "B"\n').peptide_chain == "B"

    def test_unknown_key_is_an_error_naming_it(self, tmp_path, capsys):
        err = _fails(tmp_path, capsys, 'input = "a"\nmd-duraton-ps = 5\n')
        assert "unknown key 'md-duraton-ps'" in err
        assert "did you mean 'md-duration-ps'?" in err

    def test_unknown_key_without_a_close_match(self, tmp_path, capsys):
        err = _fails(tmp_path, capsys, 'input = "a"\nzzz = 1\n')
        assert "unknown key 'zzz'" in err and "did you mean" not in err

    @pytest.mark.parametrize("key", ["help", "config", "h"])
    def test_options_that_cannot_come_from_a_file_are_unknown(self, tmp_path, capsys, key):
        assert f"unknown key {key!r}" in _fails(tmp_path, capsys, f'input = "a"\n{key} = "x"\n')

    def test_two_spellings_of_one_option_are_refused(self, tmp_path, capsys):
        err = _fails(tmp_path, capsys, 'input = "a"\nmd-duration-ps = 5\nmd_duration_ps = 6\n')
        assert "'md-duration-ps' and 'md_duration_ps' set the same option" in err
        err = _fails(tmp_path, capsys, 'input = "a"\npeptide-chain = "B"\nbinder-chain = "B"\n')
        assert "set the same option" in err


class TestValues:
    def test_types_are_converted_like_command_line_text(self, tmp_path):
        args = _parse(tmp_path, 'input = "a"\nph = 7\nmd-duration-ps = 1e2\n')
        assert args.ph == 7.0 and isinstance(args.ph, float)
        assert args.md_duration_ps == 100.0

    def test_flags_take_booleans(self, tmp_path):
        assert _parse(tmp_path, 'input = "a"\nskip-prep = true\n').skip_prep is True
        assert _parse(tmp_path, 'input = "a"\nskip-prep = false\n').skip_prep is False
        args = _parse(tmp_path, 'input = "a"\nskip-prep = false\n', "--skip-prep")
        assert args.skip_prep is True  # the flag on the command line still switches it on

    def test_lists_for_options_with_several_values(self, tmp_path):
        args = _parse(tmp_path, 'input = "a"\nenergy-modes = ["raw", "relaxed"]\nseeds = [1, 2]\n')
        assert args.energy_modes == ["raw", "relaxed"]
        assert args.seeds == [1, 2]

    @pytest.mark.parametrize(
        "line, message",
        [
            ('ph = "acid"', "key 'ph': invalid value 'acid'"),
            ("ph = true", "key 'ph': invalid value True"),
            ("ph = [1, 2]", "key 'ph': expected a single value, not a list"),
            ('skip-prep = "yes"', "key 'skip-prep': expected true or false, got 'yes'"),
            ('device = "tpu"', "key 'device': 'tpu' is not one of cuda, cpu"),
            ('energy-modes = ["raw", "bogus"]', "'bogus' is not one of raw, relaxed"),
            ("seeds = [1, 2.5]", "key 'seeds': invalid value 2.5"),
            ("[ph]\nx = 1", "key 'ph': expected a value, not a table"),
        ],
    )
    def test_bad_values_name_the_key(self, tmp_path, capsys, line, message):
        assert message in _fails(tmp_path, capsys, f'input = "a"\n{line}\n')


class TestFile:
    def test_missing_file(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_request:
            parse_args_with_config(_parser(), ["--config", str(tmp_path / "nope.toml")])
        assert exit_request.value.code == 2
        assert "cannot read" in capsys.readouterr().err

    def test_invalid_toml(self, tmp_path, capsys):
        assert "not valid TOML" in _fails(tmp_path, capsys, "input = = 1\n")

    def test_empty_file_changes_nothing(self, tmp_path):
        args = _parse(tmp_path, "", "-i", "x.cif")
        assert args.ph == 7.4


class TestMutuallyExclusiveGroup:
    def test_file_value_is_used_alone(self, tmp_path):
        args = _parse(tmp_path, 'input = "a"\nall-models = true\n')
        assert args.all_models is True and args.model is None

    def test_command_line_member_beats_the_files_other_member(self, tmp_path):
        args = _parse(tmp_path, 'input = "a"\nall-models = true\n', "--model", "2")
        assert args.model == 2 and args.all_models is False


# ---------------------------------------------------------------------------
# The three CLIs
# ---------------------------------------------------------------------------


class TestRunCli:
    def _main(self, monkeypatch, tmp_path, toml, *argv):
        from binding_metrics.cli import run

        seen = {}

        def fake_pipeline(**kwargs):
            seen.update(kwargs)
            return {"sample_id": "x", "provenance": {}}

        monkeypatch.setattr(run, "run_pipeline", fake_pipeline)
        config = _toml(tmp_path, toml)
        monkeypatch.setattr(sys, "argv", ["binding-metrics-run", "--config", str(config), *argv])
        run.main()
        return seen

    def test_file_supplies_defaults_including_required_options(self, monkeypatch, tmp_path):
        toml = (
            f'input = "{EXAMPLE_1YCR}"\noutput-dir = "{tmp_path / "o"}"\n'
            'md-duration-ps = 5\nph = 6.5\nmetrics = "interface,geometry"\n'
            'energy-modes = ["raw", "relaxed"]\nskip-prep = true\nbinder-chain = "B"\n'
        )
        seen = self._main(monkeypatch, tmp_path, toml)
        assert seen["md_duration_ps"] == 5.0 and seen["ph"] == 6.5
        assert seen["metrics"] == frozenset({"interface", "geometry"})
        assert seen["energy_modes"] == ("raw", "relaxed")
        assert seen["skip_prep"] is True and seen["peptide_chain"] == "B"

    def test_command_line_overrides_the_file(self, monkeypatch, tmp_path):
        toml = f'input = "{EXAMPLE_1YCR}"\noutput-dir = "{tmp_path / "o"}"\nph = 6.5\n'
        seen = self._main(monkeypatch, tmp_path, toml, "--ph", "8.0", "--random-seed", "none")
        assert seen["ph"] == 8.0 and seen["random_seed"] is None

    def test_unknown_key_stops_the_run(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_request:
            self._main(monkeypatch, tmp_path, 'input = "x"\nphh = 1\n')
        assert exit_request.value.code == 2
        assert "unknown key 'phh'" in capsys.readouterr().err


class TestBatchCli:
    def test_file_supplies_the_directories_and_options(self, monkeypatch, tmp_path):
        from binding_metrics.cli import batch

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.cif").write_text("data_x\n")
        seen = {}

        def fake_run_one(input_path, **kwargs):
            seen.update(kwargs)
            return {"sample_id": input_path.stem, "batch_status": "ok"}

        monkeypatch.setattr(batch, "_run_one", fake_run_one)
        toml = (
            f'input-dir = "{input_dir}"\noutput-csv = "{tmp_path / "m.csv"}"\n'
            'workers = 1\nmd-duration-ps = 30\nskip-relax = true\nmetrics = "energy"\n'
        )
        config = _toml(tmp_path, toml)
        monkeypatch.setattr(
            sys, "argv", ["binding-metrics-batch", "--config", str(config), "--ph", "9"]
        )
        with pytest.raises(SystemExit) as exit_request:
            batch.main()
        assert exit_request.value.code == 0
        assert (tmp_path / "m.csv").exists()
        assert seen["md_duration_ps"] == 30.0 and seen["skip_relax"] is True
        assert seen["ph"] == 9.0  # the command line beats the default and the file


class TestRelaxCli:
    def test_file_supplies_the_relaxation_settings(self, monkeypatch, tmp_path):
        from binding_metrics.protocols import relaxation

        seen = {}

        class FakeRelaxer:
            def __init__(self, config):
                seen["config"] = config

        def fake_run_one(relaxer, *args, **kwargs):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "minimized_structure_path": None,
                    "md_final_structure_path": None,
                },
            )()

        monkeypatch.setattr(relaxation, "ImplicitRelaxation", FakeRelaxer)
        monkeypatch.setattr(relaxation, "_run_one", fake_run_one)
        toml = (
            f'input = "{EXAMPLE_1YCR}"\noutput-dir = "{tmp_path / "o"}"\n'
            'md-duration-ps = 50\nmd-save-interval-ps = 5\nsolvent-model = "gbn2"\n'
            'temperature = 310\nsmall-molecules = "none"\nrandom-seed = 7\n'
        )
        config = _toml(tmp_path, toml)
        monkeypatch.setattr(
            sys, "argv", ["binding-metrics-relax", "--config", str(config), "--device", "cpu"]
        )
        relaxation.main()
        c = seen["config"]
        assert (c.md_duration_ps, c.md_save_interval_ps) == (50.0, 5.0)
        assert (c.solvent_model, c.md_temperature_k) == ("gbn2", 310.0)
        assert c.small_molecules is None and c.random_seed == 7
        assert c.device == "cpu"  # command line

    def test_file_all_models_yields_to_a_command_line_model(self, monkeypatch, tmp_path):
        from binding_metrics.protocols import relaxation

        seen = {}
        monkeypatch.setattr(relaxation, "ImplicitRelaxation", lambda config: None)

        def fake_run_one(relaxer, input_path, output_dir, sample_id, results_json, model_num):
            seen["model"] = model_num
            return type("Result", (), {"success": True})()

        monkeypatch.setattr(relaxation, "_run_one", fake_run_one)
        config = _toml(
            tmp_path, f'input = "{EXAMPLE_1YCR}"\noutput-dir = "{tmp_path}"\nall-models = true\n'
        )
        monkeypatch.setattr(
            sys, "argv", ["binding-metrics-relax", "--config", str(config), "--model", "1"]
        )
        relaxation.main()
        assert seen["model"] == 1


@pytest.mark.parametrize(
    "module",
    [
        "binding_metrics.cli.run",
        "binding_metrics.cli.batch",
        "binding_metrics.protocols.relaxation",
    ],
)
def test_help_documents_the_option_and_the_example(module, monkeypatch, capsys):
    import importlib

    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit):
        importlib.import_module(module).main()
    out = capsys.readouterr().out
    assert "--config PATH" in out
    assert "override the file" in " ".join(out.split())
    assert "Configuration file:" in out and ".toml" in out
