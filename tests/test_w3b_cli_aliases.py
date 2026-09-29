"""``--binder-chain`` / ``--target-chain`` on the metric command-line tools.

The new flags are aliases of the old ones: they fill the same value, and two
spellings with different IDs end the program with an argparse error. The
metric functions are replaced by a stub that records the call, so no structure
is read.
"""

import importlib
import sys

import pytest

pytest.importorskip("biotite")


class CapturedCallError(Exception):
    """Raised by the stub to hand the recorded call back to the test."""

    def __init__(self, args, kwargs):
        super().__init__("captured")
        self.call_args, self.call_kwargs = args, kwargs


def _stub(*args, **kwargs):
    raise CapturedCallError(args, kwargs)


def _invoke(monkeypatch, module_name, function_name, argv):
    """Run ``<module>.main()`` with ``argv``; return the (args, kwargs) the metric got."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, function_name, _stub)
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(CapturedCallError) as captured:
        module.main()
    return captured.value.call_args, captured.value.call_kwargs


# name, module, metric function, base argv, keyword for the binder, keyword for the target,
# old binder flag, old target flag. None where the CLI has no such role.
CLIS = [
    (
        "interface",
        "binding_metrics.metrics.interface",
        "compute_interface_metrics",
        ["--input", "x.cif"],
        "design_chain",
        "receptor_chain",
        "--design-chain",
        "--receptor-chain",
    ),
    (
        "electrostatics",
        "binding_metrics.metrics.electrostatics",
        "compute_coulomb_cross_chain",
        ["--input", "x.cif"],
        "peptide_chain",
        "receptor_chain",
        "--design-chain",
        "--receptor-chain",
    ),
    (
        "energy",
        "binding_metrics.metrics.energy",
        "compute_interaction_energy",
        ["--input", "x.cif"],
        "peptide_chain",
        "receptor_chain",
        "--peptide-chain",
        "--receptor-chain",
    ),
    (
        "geometry_sc",
        "binding_metrics.metrics.geometry",
        "compute_shape_complementarity",
        ["--input", "x.cif", "--metric", "sc"],
        "peptide_chain",
        "receptor_chain",
        "--peptide-chain",
        "--receptor-chain",
    ),
    (
        "geometry_void",
        "binding_metrics.metrics.geometry",
        "compute_buried_void_volume",
        ["--input", "x.cif", "--metric", "void"],
        "peptide_chain",
        "receptor_chain",
        "--peptide-chain",
        "--receptor-chain",
    ),
    (
        "geometry_ramachandran",
        "binding_metrics.metrics.geometry",
        "compute_ramachandran",
        ["--input", "x.cif", "--metric", "ramachandran"],
        "chain",
        None,
        "--chain",
        None,
    ),
    (
        "geometry_omega",
        "binding_metrics.metrics.geometry",
        "compute_omega_planarity",
        ["--input", "x.cif", "--metric", "omega"],
        "chain",
        None,
        "--chain",
        None,
    ),
    (
        "receptor_quality",
        "binding_metrics.metrics.receptor_quality",
        "compute_receptor_quality",
        ["--input", "x.cif"],
        None,
        "receptor_chain",
        None,
        "--receptor-chain",
    ),
]


@pytest.fixture(params=CLIS, ids=[c[0] for c in CLIS])
def cli(request):
    return request.param


def test_old_flags_still_work(cli, monkeypatch):
    _, module, function, base, binder_key, target_key, binder_flag, target_flag = cli
    argv = list(base)
    if binder_flag:
        argv += [binder_flag, "B"]
    if target_flag:
        argv += [target_flag, "A"]
    _, kwargs = _invoke(monkeypatch, module, function, argv)
    if binder_key:
        assert kwargs[binder_key] == "B"
    if target_key:
        assert kwargs[target_key] == "A"


def test_aliases_fill_the_same_values(cli, monkeypatch):
    _, module, function, base, binder_key, target_key, binder_flag, target_flag = cli
    argv = list(base)
    if binder_flag:
        argv += ["--binder-chain", "B"]
    if target_flag:
        argv += ["--target-chain", "A"]
    _, kwargs = _invoke(monkeypatch, module, function, argv)
    if binder_key:
        assert kwargs[binder_key] == "B"
    if target_key:
        assert kwargs[target_key] == "A"


def test_both_spellings_with_the_same_id_are_accepted(cli, monkeypatch):
    _, module, function, base, binder_key, target_key, binder_flag, target_flag = cli
    argv = list(base)
    if binder_flag:
        argv += [binder_flag, "B", "--binder-chain", "B"]
    if target_flag:
        argv += ["--target-chain", "A", target_flag, "A"]
    _, kwargs = _invoke(monkeypatch, module, function, argv)
    if binder_key:
        assert kwargs[binder_key] == "B"
    if target_key:
        assert kwargs[target_key] == "A"


@pytest.mark.parametrize("order", ["old_first", "alias_first"])
def test_conflicting_binder_spellings_are_a_usage_error(cli, monkeypatch, capsys, order):
    _, module, function, base, _, _, binder_flag, _ = cli
    if not binder_flag:
        pytest.skip("no binder role in this CLI")
    pair = [binder_flag, "B", "--binder-chain", "C"]
    if order == "alias_first":
        pair = ["--binder-chain", "C", binder_flag, "B"]
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, function, _stub)
    monkeypatch.setattr(sys, "argv", ["prog", *base, *pair])
    with pytest.raises(SystemExit) as exit_info:
        mod.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--binder-chain" in err and binder_flag in err


def test_conflicting_target_spellings_are_a_usage_error(cli, monkeypatch, capsys):
    _, module, function, base, _, _, _, target_flag = cli
    if not target_flag:
        pytest.skip("no target role in this CLI")
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, function, _stub)
    monkeypatch.setattr(sys, "argv", ["prog", *base, target_flag, "A", "--target-chain", "C"])
    with pytest.raises(SystemExit) as exit_info:
        mod.main()
    assert exit_info.value.code == 2
    assert "--target-chain" in capsys.readouterr().err


def test_help_names_the_aliases(cli, monkeypatch, capsys):
    _, module, _, _, _, _, binder_flag, target_flag = cli
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        mod.main()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    if binder_flag:
        assert "--binder-chain" in out
    if target_flag:
        assert "--target-chain" in out


class TestGeometryPicksTheOldFlagOfTheMetric:
    """--binder-chain is --chain for Ramachandran/omega and --peptide-chain for Sc/void."""

    def test_ramachandran_ignores_peptide_chain_as_before(self, monkeypatch):
        _, kwargs = _invoke(
            monkeypatch,
            "binding_metrics.metrics.geometry",
            "compute_ramachandran",
            ["--input", "x.cif", "--metric", "ramachandran", "--peptide-chain", "P"],
        )
        assert kwargs["chain"] is None

    def test_sc_ignores_chain_as_before(self, monkeypatch):
        _, kwargs = _invoke(
            monkeypatch,
            "binding_metrics.metrics.geometry",
            "compute_shape_complementarity",
            ["--input", "x.cif", "--metric", "sc", "--chain", "P", "--binder-chain", "B"],
        )
        assert kwargs["peptide_chain"] == "B"

    def test_conflict_is_checked_on_the_flag_the_metric_uses(self, monkeypatch, capsys):
        from binding_metrics.metrics import geometry

        argv = ["--input", "x.cif", "--metric", "void", "--peptide-chain", "P"]
        monkeypatch.setattr(sys, "argv", ["prog", *argv, "--binder-chain", "B"])

        with pytest.raises(SystemExit) as exit_info:
            geometry.main()
        assert exit_info.value.code == 2
        assert "--peptide-chain" in capsys.readouterr().err


class TestOpenFoldCli:
    """The OpenFold CLI already had --binder-chain; --target-chain is new."""

    def _prepare(self, monkeypatch, *chain_flags):
        from binding_metrics.metrics import openfold

        seen = {}

        def fake_prepare(**kwargs):
            seen.update(kwargs)
            return "query.json"

        monkeypatch.setattr(openfold, "prepare_scoring_query", fake_prepare)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "prepare-scoring-query",
                "--complex",
                "x.cif",
                "--query-name",
                "q",
                "--output-dir",
                "out",
                *chain_flags,
            ],
        )
        openfold.main()
        return seen

    def test_target_chain_is_the_receptor_chain(self, monkeypatch):
        seen = self._prepare(monkeypatch, "--binder-chain", "B", "--target-chain", "A")
        assert (seen["binder_chain"], seen["receptor_chain"]) == ("B", "A")

    def test_receptor_chain_still_works(self, monkeypatch):
        seen = self._prepare(monkeypatch, "--binder-chain", "B", "--receptor-chain", "A")
        assert (seen["binder_chain"], seen["receptor_chain"]) == ("B", "A")

    def test_one_of_the_two_is_still_required(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            self._prepare(monkeypatch, "--binder-chain", "B")
        assert exit_info.value.code == 2
        assert "--receptor-chain" in capsys.readouterr().err

    def test_conflict_is_a_usage_error(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            self._prepare(
                monkeypatch,
                "--binder-chain",
                "B",
                "--receptor-chain",
                "A",
                "--target-chain",
                "C",
            )
        assert exit_info.value.code == 2
        assert "--target-chain" in capsys.readouterr().err

    def test_parse_command_forwards_the_alias(self, monkeypatch):
        from binding_metrics.metrics import openfold

        def fake_metrics(**kwargs):
            raise CapturedCallError((), kwargs)

        monkeypatch.setattr(openfold, "compute_openfold_metrics", fake_metrics)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "parse",
                "--output-dir",
                "out",
                "--query-name",
                "q",
                "--binder-chain",
                "B",
                "--target-chain",
                "A",
            ],
        )
        with pytest.raises(CapturedCallError) as captured:
            openfold.main()
        assert captured.value.call_kwargs["receptor_chain"] == "A"
        assert captured.value.call_kwargs["binder_chain"] == "B"
