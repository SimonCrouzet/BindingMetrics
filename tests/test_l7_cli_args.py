"""Argument parsers shared by the CLIs."""

import argparse

import pytest

from binding_metrics.cli import add_random_seed_arg, seed_arg, small_molecules_arg


class TestSmallMoleculesArg:
    def test_auto_is_kept(self):
        assert small_molecules_arg("auto") == "auto"

    def test_none_maps_to_python_none(self):
        assert small_molecules_arg("none") is None

    @pytest.mark.parametrize("value", ["AUTO", " Auto ", "None", "NONE"])
    def test_case_and_whitespace_are_forgiven(self, value):
        assert small_molecules_arg(value) in ("auto", None)

    @pytest.mark.parametrize("value", ["aut", "yes", "", "CCO", "NC(CS)C(=O)O", "auto,none"])
    def test_anything_else_is_rejected_with_the_valid_choices(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="expected one of auto, none"):
            small_molecules_arg(value)

    def test_argparse_reports_a_clear_error_and_exits(self, capsys):
        parser = argparse.ArgumentParser(prog="x")
        parser.add_argument("--small-molecules", type=small_molecules_arg, default="auto")
        assert parser.parse_args([]).small_molecules == "auto"
        assert parser.parse_args(["--small-molecules", "none"]).small_molecules is None
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--small-molecules", "aut"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "argument --small-molecules: invalid value 'aut': expected one of auto, none" in err


class TestSeedArg:
    @pytest.mark.parametrize("text,expected", [("7", 7), ("0", 0), ("-3", -3), (" 12 ", 12)])
    def test_integers(self, text, expected):
        assert seed_arg(text) == expected

    @pytest.mark.parametrize("text", ["none", "None", "random", "OFF"])
    def test_fresh_randomness_spellings(self, text):
        assert seed_arg(text) is None

    def test_garbage_is_rejected_by_argparse(self):
        parser = argparse.ArgumentParser()
        add_random_seed_arg(parser, "testing")
        with pytest.raises(SystemExit):
            parser.parse_args(["--random-seed", "seven"])

    def test_add_random_seed_arg_defaults_to_the_library_seed(self):
        from binding_metrics.core.system import DEFAULT_RANDOM_SEED

        parser = argparse.ArgumentParser()
        add_random_seed_arg(parser, "testing")
        assert parser.parse_args([]).random_seed == DEFAULT_RANDOM_SEED
        assert parser.parse_args(["--random-seed", "5"]).random_seed == 5
        assert parser.parse_args(["--random-seed", "none"]).random_seed is None

    @pytest.mark.parametrize(
        "formatter", [argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter]
    )
    def test_help_shows_the_default_once(self, formatter):
        parser = argparse.ArgumentParser(formatter_class=formatter)
        add_random_seed_arg(parser, "ion placement")
        assert parser.format_help().count("default:") == 1
