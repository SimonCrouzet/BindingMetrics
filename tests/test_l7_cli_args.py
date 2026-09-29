"""Argument parsers shared by the CLIs."""

import argparse

import pytest

from binding_metrics.cli import small_molecules_arg


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
