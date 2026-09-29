"""The command-line options of the pipeline CLIs are unchanged.

GOLDEN records, for every option of binding-metrics-run, -batch, -relax,
-prep and -solvate, what --help shows for it: group, metavar, choices, nargs,
required flag, default and the expanded help text. It was captured before the repeated
default literals (pH, device, MD duration, save interval) were replaced by the named
constants in binding_metrics._constants. New options may be added; an existing one
must not change.
"""

import argparse
import importlib
import sys

import pytest

from binding_metrics import _constants

CLI_MODULES = {
    "run": "binding_metrics.cli.run",
    "batch": "binding_metrics.cli.batch",
    "relax": "binding_metrics.protocols.relaxation",
    "prep": "binding_metrics.protocols.prep",
    "solvate": "binding_metrics.protocols.solvate",
}


class _ParserCapturedError(Exception):
    pass


def capture_parser(module_name, monkeypatch):
    """Run the module's main up to parse_args and return the parser it built."""
    holder = {}

    def stop(self, *args, **kwargs):
        holder["parser"] = self
        raise _ParserCapturedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(_ParserCapturedError):
        importlib.import_module(module_name).main()
    return holder["parser"]


def _stable_repr(value):
    """``repr`` with sets sorted: the iteration order of a frozenset varies between runs."""
    if isinstance(value, (set, frozenset)):
        return repr(sorted(value))
    return repr(value)


def describe_options(parser):
    """Map each option (its spellings joined by "/") to what --help shows for it."""
    formatter = parser._get_formatter()
    group_of = {
        id(action): group.title
        for group in parser._action_groups
        for action in group._group_actions
    }
    described = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        described["/".join(action.option_strings)] = (
            group_of.get(id(action)),
            action.metavar,
            list(action.choices) if action.choices else None,
            action.nargs,
            action.required,
            _stable_repr(action.default),
            formatter._expand_help(action) if action.help else None,
        )
    return described


GOLDEN = {
    "run": {
        "description": "Run the full binding-metrics pipeline on a single structure.",
        "options": {
            "--input/-i": ("options", None, None, None, True, "None", "Input CIF or PDB file"),
            "--output-dir/-o": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Directory to write all outputs",
            ),
            "--sample-id": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Sample identifier (defaults to input file stem)",
            ),
            "--device": (
                "options",
                None,
                ["cuda", "cpu"],
                None,
                False,
                "'cuda'",
                "Compute device (default: cuda)",
            ),
            "--peptide-chain": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Peptide chain ID (auto-detect if omitted)",
            ),
            "--receptor-chain": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Receptor chain ID (auto-detect if omitted)",
            ),
            "--reference/--native": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Reference/native complex for reference-based "
                "accuracy metrics (DockQ, fnat, i-RMSD, L-RMSD). "
                "Supplying this auto-enables the 'dockq' metric. "
                "Requires: pip install DockQ",
            ),
            "--skip-prep": (
                "Preparation",
                None,
                None,
                0,
                False,
                "False",
                "Skip PDBFixer prep; run relax on raw input",
            ),
            "--ph": (
                "Preparation",
                None,
                None,
                None,
                False,
                "7.4",
                "pH for hydrogen placement during prep (default: 7.4)",
            ),
            "--keep-water": (
                "Preparation",
                None,
                None,
                0,
                False,
                "False",
                "Retain crystallographic water molecules during prep",
            ),
            "--canonicalize": (
                "Preparation",
                None,
                None,
                0,
                False,
                "False",
                "Replace non-standard residues with standard "
                "equivalents during prep (e.g. MSE→MET, SEP→SER). By "
                "default they are preserved for GAFF2 parameterisation "
                "in relax (--small-molecules auto).",
            ),
            "--skip-relax": (
                "Relaxation",
                None,
                None,
                0,
                False,
                "False",
                "Skip relaxation; run metrics on raw input",
            ),
            "--md-duration-ps": (
                "Relaxation",
                None,
                None,
                None,
                False,
                "200.0",
                "MD duration in ps (0 = minimize only, default: 200)",
            ),
            "--random-seed": (
                "Relaxation",
                "INT|none",
                None,
                None,
                False,
                "1",
                "Seed for all stochastic steps (hydrogen placement, MD "
                "velocities and thermostat). A fixed integer makes the "
                "run reproducible (default: 1); pass 'none' for fresh "
                "randomness each run, e.g. to generate independent MD "
                "replicas.",
            ),
            "--metrics": (
                "Metrics",
                "METRICS",
                None,
                None,
                False,
                "['electrostatics', 'energy', 'geometry', 'interface', 'openfold']",
                "Comma-separated list of metrics to compute. Valid: dockq, "
                "electrostatics, energy, geometry, interface, openfold. "
                "Default: all reference-free metrics. 'dockq' also needs "
                "--reference.",
            ),
            "--energy-modes": (
                "Metrics",
                None,
                ["raw", "relaxed", "after_md"],
                "+",
                False,
                "['relaxed']",
                "Energy evaluation modes (default: relaxed)",
            ),
            "--openfold-mode": (
                "OpenFold",
                None,
                ["score", "refold"],
                None,
                False,
                "'score'",
                "score: both chains as templates (confidence); "
                "refold: binder predicted freely (refolding RMSD). "
                "Default: score",
            ),
            "--openfold-conda-env": (
                "OpenFold",
                None,
                None,
                None,
                False,
                "'openfold3'",
                "Conda environment name where OpenFold3 is "
                "installed (default: openfold3). Set to empty "
                "string to use the current environment if "
                "openfold3 is installed there.",
            ),
            "--openfold-seeds": (
                "OpenFold",
                "SEED",
                None,
                "+",
                False,
                "None",
                "Seed values written to the OpenFold3 query JSON "
                "(default: the OpenFold module default, 42). The "
                "first seed's first sample is scored.",
            ),
            "--format": (
                "Report",
                None,
                ["json", "csv"],
                None,
                False,
                "'json'",
                "Results output format (default: json)",
            ),
            "--summary": (
                "Report",
                None,
                None,
                0,
                False,
                "False",
                "Also write a human-readable summary (*_report.md or *_report.html)",
            ),
            "--summary-format": (
                "Report",
                None,
                ["md", "html"],
                None,
                False,
                "'md'",
                "Summary format (default: md)",
            ),
            "--log-file": (
                "Report",
                "PATH",
                None,
                None,
                False,
                "None",
                "Redirect all output (stdout + stderr) to this file",
            ),
        },
    },
    "batch": {
        "description": "Run the binding-metrics pipeline on all structures in a directory.",
        "options": {
            "--input-dir/-i": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Directory containing .cif / .pdb / .mmcif files",
            ),
            "--output-csv": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Path for the aggregated CSV results file",
            ),
            "--output-dir/-o": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Directory for per-sample outputs (default: same directory as --output-csv)",
            ),
            "--workers": (
                "options",
                None,
                None,
                None,
                False,
                "1",
                "Number of parallel worker processes (default: 1). Use "
                "--device cpu when workers > 1 on a single GPU.",
            ),
            "--glob": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Optional glob pattern to filter files within --input-dir "
                "(e.g. '*.cif'). Default: all .cif/.pdb/.mmcif files.",
            ),
            "--reference-dir": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Directory of native/reference structures for "
                "DockQ. Each sample is matched to a reference by "
                "filename stem (e.g. input 'target1.cif' → "
                "reference 'target1.pdb'). Supplying this "
                "auto-enables the 'dockq' metric. Requires: pip "
                "install DockQ",
            ),
            "--device": (
                "options",
                None,
                ["cuda", "cpu"],
                None,
                False,
                "'cuda'",
                "Compute device (default: cuda)",
            ),
            "--peptide-chain": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Peptide chain ID applied to all structures (auto-detect per structure if omitted)",
            ),
            "--receptor-chain": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Receptor chain ID applied to all structures "
                "(auto-detect per structure if omitted)",
            ),
            "--skip-prep": ("Preparation", None, None, 0, False, "False", "Skip PDBFixer prep"),
            "--ph": (
                "Preparation",
                None,
                None,
                None,
                False,
                "7.4",
                "pH for hydrogen placement during prep (default: 7.4)",
            ),
            "--keep-water": (
                "Preparation",
                None,
                None,
                0,
                False,
                "False",
                "Retain crystallographic water molecules during prep",
            ),
            "--canonicalize": (
                "Preparation",
                None,
                None,
                0,
                False,
                "False",
                "Replace non-standard residues with standard equivalents",
            ),
            "--skip-relax": ("Relaxation", None, None, 0, False, "False", "Skip relaxation"),
            "--md-duration-ps": (
                "Relaxation",
                None,
                None,
                None,
                False,
                "200.0",
                "MD duration in ps (0 = minimize only, default: 200)",
            ),
            "--random-seed": (
                "Relaxation",
                "INT|none",
                None,
                None,
                False,
                "1",
                "Seed for every stochastic step of each sample "
                "(hydrogen placement, MD velocities and thermostat); "
                "the same seed is used for all samples. A fixed "
                "integer makes the run reproducible (default: 1); "
                "pass 'none' for fresh randomness each run.",
            ),
            "--metrics": (
                "Metrics",
                "METRICS",
                None,
                None,
                False,
                "['electrostatics', 'energy', 'geometry', 'interface', 'openfold']",
                "Comma-separated list of metrics to compute. Valid: "
                "electrostatics, energy, geometry, interface, openfold. "
                "Default: all.",
            ),
            "--energy-modes": (
                "Metrics",
                None,
                ["raw", "relaxed", "after_md"],
                "+",
                False,
                "['relaxed']",
                "Energy evaluation modes (default: relaxed)",
            ),
            "--openfold-mode": (
                "OpenFold",
                None,
                ["score", "refold"],
                None,
                False,
                "'score'",
                "score: both chains as templates; refold: binder predicted freely. Default: score",
            ),
            "--openfold-conda-env": (
                "OpenFold",
                None,
                None,
                None,
                False,
                "'openfold3'",
                "Conda env where OpenFold3 is installed (default: openfold3)",
            ),
            "--openfold-seeds": (
                "OpenFold",
                "SEED",
                None,
                "+",
                False,
                "None",
                "Seed values written to the OpenFold3 query JSON "
                "(default: the OpenFold module default, 42). The "
                "first seed's first sample is scored.",
            ),
            "--log-file": (
                "Logging",
                "PATH",
                None,
                None,
                False,
                "None",
                "Redirect all output (stdout + stderr) to this file",
            ),
            "--per-sample-log": (
                "Logging",
                None,
                None,
                0,
                False,
                "False",
                "Write a separate .log file for each sample inside "
                "its output directory (always on when --log-file "
                "is not set, ignored when --log-file is "
                "provided)",
            ),
        },
    },
    "relax": {
        "description": "Implicit solvent MD relaxation for protein complexes",
        "options": {
            "--input/-i": ("options", None, None, None, True, "None", "Input CIF or PDB file"),
            "--output-dir/-o": ("options", None, None, None, True, "None", "Output directory"),
            "--md-duration-ps": (
                "options",
                None,
                None,
                None,
                False,
                "200.0",
                "MD duration in ps (0 to minimize only)",
            ),
            "--md-save-interval-ps": (
                "options",
                None,
                None,
                None,
                False,
                "10.0",
                "Frame save interval in ps",
            ),
            "--temperature": (
                "options",
                None,
                None,
                None,
                False,
                "300.0",
                "Simulation temperature in K",
            ),
            "--device": ("options", None, ["cuda", "cpu"], None, False, "'cuda'", "Compute device"),
            "--ph": (
                "options",
                None,
                None,
                None,
                False,
                "7.4",
                "pH for hydrogen addition (default 7.4)",
            ),
            "--solvent-model": (
                "options",
                None,
                ["obc2", "gbn2"],
                None,
                False,
                "'obc2'",
                "Implicit solvent model",
            ),
            "--peptide-chain": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Peptide chain ID (auto-detect if omitted)",
            ),
            "--receptor-chain": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Receptor chain ID (auto-detect if omitted)",
            ),
            "--sample-id": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Sample identifier (defaults to input file stem)",
            ),
            "--small-molecules": (
                "options",
                None,
                None,
                None,
                False,
                "'auto'",
                "Non-standard residue parameterisation. 'auto' "
                "(default) builds GAFF2 ExternalBond templates "
                "for every exotic NCAA; 'none' disables it.",
            ),
            "--random-seed": (
                "options",
                "INT|none",
                None,
                None,
                False,
                "'1'",
                "Seed for stochastic steps (hydrogen placement, MD "
                "velocities/thermostat). Default 1 (reproducible); "
                "'none' for fresh randomness each run.",
            ),
            "--model": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Extract and relax a single model from a multi-model CIF (1-based)",
            ),
            "--all-models": (
                "options",
                None,
                None,
                0,
                False,
                "False",
                "Relax every model in a multi-model CIF; errors on "
                "single-model files if given explicitly. sample-id is "
                "auto-set to <stem>_model<N> for each.",
            ),
            "--results-json": (
                "options",
                None,
                None,
                None,
                False,
                "None",
                "Path to write relax results as JSON. Omit to skip "
                "JSON output (ignored with --all-models).",
            ),
            "--log-file": (
                "options",
                "PATH",
                None,
                None,
                False,
                "None",
                "Redirect all output (stdout + stderr) to this file",
            ),
        },
    },
    "prep": {
        "description": "Fix and protonate a structure using PDBFixer.",
        "options": {
            "--input/-i": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Input structure (.pdb, .cif, .mmcif) (default: None)",
            ),
            "--output/-o": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Output structure (.pdb, .cif, .mmcif) (default: None)",
            ),
            "--ph": (
                "options",
                None,
                None,
                None,
                False,
                "7.4",
                "pH for hydrogen placement (default: 7.4)",
            ),
            "--keep-water": (
                "options",
                None,
                None,
                0,
                False,
                "False",
                "Retain crystallographic water molecules (default: False)",
            ),
            "--canonicalize": (
                "options",
                None,
                None,
                0,
                False,
                "False",
                "Replace non-standard residues with their nearest "
                "standard equivalents (e.g. MSE→MET, SEP→SER). By "
                "default they are preserved so they can be "
                "parameterised downstream with GAFF2 "
                "(--small-molecules auto in relax). (default: False)",
            ),
            "--no-rebuild-zero-coord-atoms": (
                "options",
                None,
                None,
                0,
                False,
                "False",
                "Disable detection and rebuild of "
                "zero-coordinate placeholder atoms. By "
                "default, atoms at the origin are "
                "removed and rebuilt by PDBFixer "
                "(needed for pipelines that output "
                "placeholders instead of modelled "
                "atoms). (default: False)",
            ),
            "--random-seed": (
                "options",
                "INT|none",
                None,
                None,
                False,
                "1",
                "Seed for hydrogen placement and PDBFixer's atom "
                "rebuilding. A fixed integer makes the run "
                "reproducible (default: 1); pass 'none' for fresh "
                "randomness each run.",
            ),
            "--log-file": (
                "options",
                "PATH",
                None,
                None,
                False,
                "None",
                "Redirect all output (stdout + stderr) to this file (default: None)",
            ),
        },
    },
    "solvate": {
        "description": "Add explicit solvent and ions to a structure.",
        "options": {
            "--input/-i": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Input structure (.pdb, .cif, .mmcif) (default: None)",
            ),
            "--output/-o": (
                "options",
                None,
                None,
                None,
                True,
                "None",
                "Output structure (.pdb, .cif, .mmcif) (default: None)",
            ),
            "--forcefield": (
                "options",
                None,
                ["amber", "charmm"],
                None,
                False,
                "'amber'",
                "Force field for solvent parameters (default: amber)",
            ),
            "--padding": (
                "options",
                None,
                None,
                None,
                False,
                "1.0",
                "Distance in nm between solute and box edge (default: 1.0)",
            ),
            "--ionic-strength": (
                "options",
                None,
                None,
                None,
                False,
                "0.15",
                "Salt concentration in M (default: 0.15)",
            ),
            "--positive-ion": (
                "options",
                None,
                None,
                None,
                False,
                "'Na+'",
                "Positive ion type (default: Na+)",
            ),
            "--negative-ion": (
                "options",
                None,
                None,
                None,
                False,
                "'Cl-'",
                "Negative ion type (default: Cl-)",
            ),
            "--random-seed": (
                "options",
                "INT|none",
                None,
                None,
                False,
                "1",
                "Seed for ion placement (which water molecules "
                "become ions). A fixed integer makes the run "
                "reproducible (default: 1); pass 'none' for fresh "
                "randomness each run.",
            ),
            "--log-file": (
                "options",
                "PATH",
                None,
                None,
                False,
                "None",
                "Redirect all output (stdout + stderr) to this file (default: None)",
            ),
        },
    },
}


@pytest.mark.parametrize("cli", sorted(GOLDEN))
def test_existing_options_are_unchanged(cli, monkeypatch):
    parser = capture_parser(CLI_MODULES[cli], monkeypatch)
    assert parser.description == GOLDEN[cli]["description"]
    described = describe_options(parser)
    for option, expected in GOLDEN[cli]["options"].items():
        assert option in described, f"{cli}: option {option} disappeared"
        assert described[option] == expected, f"{cli}: {option} changed"


@pytest.mark.parametrize("cli", sorted(GOLDEN))
def test_help_still_exits_cleanly(cli, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as exit_request:
        importlib.import_module(CLI_MODULES[cli]).main()
    assert exit_request.value.code == 0
    out = capsys.readouterr().out
    for option in GOLDEN[cli]["options"]:
        assert option.split("/")[0] in out


def test_default_constants_hold_the_documented_values():
    assert _constants.DEFAULT_PH == 7.4
    assert _constants.DEFAULT_DEVICE == "cuda"
    assert _constants.DEFAULT_MD_DURATION_PS == 200.0
    assert _constants.DEFAULT_MD_SAVE_INTERVAL_PS == 10.0


def test_library_and_cli_defaults_come_from_the_constants(monkeypatch):
    import inspect

    from binding_metrics.cli import run
    from binding_metrics.core import system
    from binding_metrics.protocols.relaxation import RelaxationConfig

    config = RelaxationConfig()
    assert config.ph == _constants.DEFAULT_PH
    assert config.device == _constants.DEFAULT_DEVICE
    assert config.md_duration_ps == _constants.DEFAULT_MD_DURATION_PS
    assert config.md_save_interval_ps == _constants.DEFAULT_MD_SAVE_INTERVAL_PS

    pipeline = inspect.signature(run.run_pipeline).parameters
    assert pipeline["ph"].default == _constants.DEFAULT_PH
    assert pipeline["device"].default == _constants.DEFAULT_DEVICE
    assert pipeline["md_duration_ps"].default == _constants.DEFAULT_MD_DURATION_PS

    for function in (system.prep_structure, system.prepare_system):
        assert inspect.signature(function).parameters["ph"].default == _constants.DEFAULT_PH

    for cli in ("run", "batch", "relax"):
        parser = capture_parser(CLI_MODULES[cli], monkeypatch)
        defaults = {a.dest: a.default for a in parser._actions}
        assert defaults["ph"] == _constants.DEFAULT_PH
        assert defaults["device"] == _constants.DEFAULT_DEVICE
        assert defaults["md_duration_ps"] == _constants.DEFAULT_MD_DURATION_PS
    prep = {a.dest: a.default for a in capture_parser(CLI_MODULES["prep"], monkeypatch)._actions}
    assert prep["ph"] == _constants.DEFAULT_PH
