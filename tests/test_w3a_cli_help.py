"""The command-line options of the pipeline CLIs are unchanged.

``GOLDEN`` records, for every option of ``binding-metrics-run``, ``-batch``, ``-relax``,
``-prep`` and ``-solvate``, what ``--help`` shows for it: group, metavar, choices, nargs,
required flag, default and the expanded help text. It was captured before the repeated
default literals (pH, device, MD duration, save interval) were replaced by the named
constants in ``binding_metrics._constants``. New options may be added; an existing one
must not change.

Four deliberate edits since the capture: ``--peptide-chain`` and ``--receptor-chain`` gained
the alias spellings ``--binder-chain`` and ``--target-chain``, the ``--metrics`` help of
``-batch`` lists ``dockq``, which the option already accepted, the ``--openfold-seeds``
help of both commands no longer says that the seeds go to the query JSON (OpenFold3 does not
read them there; they are written to its runner YAML) and says which sample is scored, and the
``--openfold-mode`` help of both commands no longer says that ``score`` gives OpenFold3 both
chains as templates "for the known conformation": a template carries the fold of one chain and
no cross-chain geometry, so OpenFold3 places the binder itself (the help names ``binder_ca_rmsd``
and ``delta_com_angstrom`` as the keys that say how far its pose is from the input pose). Two
options were added to both commands and are recorded here from now on: ``--openfold-cyclic``
and ``--openfold-no-msa-server``. Four more were added with the pre-flight check and are recorded
the same way: ``--binder-type``, ``--on-incompatible``, ``--preflight-only`` and
``--prediction-mode`` (the mode a model is used in: predict, refold, score or score-lock).
``--prediction-weights`` (custom weights of the model) was added to both commands afterwards and
is recorded the same way. When the runners of ColabFold, Boltz-2 and Protenix were registered, the
``--openfold-seeds`` help gained a sentence on what the seeds mean for those runners, the
``--prediction-mode`` help says what each runner can do and what its default is, and four options
were added to both commands and are recorded here: ``--prediction-cyclic``,
``--prediction-no-msa-server``, ``--prediction-conda-env`` and ``--prediction-lock-threshold``.
The help of ``--openfold-cyclic`` and ``--prediction-cyclic`` then gained the rule that ``auto``
leaves a head-to-tail binder with modified residues linear, and the one-complex measurement
behind it.
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
                "score: each chain is given its own structure from the "
                "input as a template and OpenFold3 places the binder "
                "itself, so its confidences refer to its own pose "
                "(binder_ca_rmsd and delta_com_angstrom show how far it "
                "is from the input pose); refold: only the receptor "
                "is templated and the binder is predicted from its "
                "sequence (binder_ca_rmsd is the refolding RMSD). "
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
                "Seed values OpenFold3 samples with, written to its runner "
                "YAML (default: 42). It makes one seed_<value> directory "
                "per seed; the first seed given, first sample, is scored. "
                "With --predictor MODEL the seeds go to that model's runner"
                " (ColabFold: consecutive integers from 0, default 0; "
                "Boltz-2: exactly one, default 42; Protenix: default 101).",
            ),
            "--openfold-cyclic": (
                "OpenFold",
                None,
                ["auto", "on", "off"],
                None,
                False,
                "'auto'",
                "Whether the binder chain of the OpenFold3 query gets 'cyclic: "
                "true' (OpenFold3 >= 0.4.5). auto (default): when the binder "
                "has a head-to-tail bond, consists of standard residues only "
                "and the installed OpenFold3 is new enough; on: always; off: "
                "never. OpenFold3 uses the flag only to wrap the relative "
                "positions of the chain: it does not enforce the closure bond, "
                "documents the flag only in an example query, and has published"
                " no accuracy benchmark for cyclic peptides. It builds the wrap"
                " from the token count of the chain and gives every atom of a "
                "modified residue its own token: for 1CWA (D-amino acid, "
                "N-methylated residues; one complex, three seeds) the flag "
                "lowered ipTM from 0.91-0.92 to 0.78-0.81 and raised the binder"
                " C-alpha RMSD from 0.5-0.7 A to 3.0-4.8 A, so auto leaves such"
                " a binder linear (on forces the flag); for SFTI-1 (standard "
                "residues; one seed) the flag closed the ring (C-N 7.40 A "
                "without it, 1.38 A with it). Disulfide, lactam and staple "
                "closures cannot be given to OpenFold3 and are not written.",
            ),
            "--openfold-no-msa-server": (
                "OpenFold",
                None,
                None,
                0,
                False,
                "False",
                "Do not use the ColabFold MSA server for OpenFold3: it then "
                "runs with a dummy MSA that holds only the query sequence of "
                "each chain (OpenFold3's input reference suggests this for "
                "MSA-free runs), which lowers accuracy for a natural receptor, "
                "but the template alignments written by the toolkit are no "
                "longer replaced by the server (issue #68). One complex (1YCR, "
                "OpenFold3 0.5.0, one seed), binder C-alpha RMSD against the "
                "crystal pose: 1.6 A with the server and no template (the "
                "server replaces the template), 21.6 A with no MSA and no "
                "template, 1.1 A with a working template and no MSA.",
            ),
            "--prediction-mode": (
                "Prediction",
                None,
                ["predict", "refold", "score", "score-lock"],
                None,
                False,
                "None",
                "How the model is used for the complex: predict (sequences "
                "only), refold (receptor templated, binder predicted "
                "freely), score (every chain templated on its own, the pose"
                " not given: re-docking) or score-lock (score, with the "
                "pose pinned to the input). It is checked against what the "
                "model supports and, for a run from here, against what its "
                "runner can do (of3: refold, score; boltz2: all four; af2 "
                "and protenix: predict), before anything runs, and "
                "recorded. Default: the runner's own mode, that is for "
                "--predictor of3 the value of --openfold-mode (score), "
                "score for boltz2 and predict for af2 and protenix; for an "
                "output read with --prediction-dir, not stated and not "
                "checked. Needs --predictor.",
            ),
            "--prediction-cyclic": (
                "Prediction",
                None,
                ["auto", "on", "off"],
                None,
                False,
                "None",
                "Whether the binder is given to the model as cyclic, for "
                "--predictor MODEL run from here. auto (default): when the "
                "binder has a head-to-tail bond (for OpenFold3 also only when "
                "it consists of standard residues: with a D-amino acid and "
                "N-methylated residues, 1CWA, the flag lowered ipTM from "
                "0.91-0.92 to 0.78-0.81 in one complex, three seeds, so auto "
                "leaves such a binder linear); on: always; off: never. "
                "OpenFold3 (>= 0.4.5) and Boltz-2 get 'cyclic: true' on the "
                "binder chain, which only wraps its relative positions and does"
                " not enforce the closure bond; Protenix gets the head-to-tail "
                "and disulfide bonds as covalent_bonds; ColabFold has no such "
                "setting and refuses a value. For --predictor of3 it is the "
                "setting of --openfold-cyclic (both given with different values"
                " is an error). Needs --predictor.",
            ),
            "--prediction-no-msa-server": (
                "Prediction",
                None,
                None,
                0,
                False,
                "False",
                "Do not use an MSA server, for --predictor MODEL run from "
                "here: ColabFold runs single_sequence, Boltz-2 writes 'msa:"
                " empty', Protenix runs with --use_msa false, OpenFold3 as "
                "--openfold-no-msa-server. The accuracy for a natural "
                "receptor drops; no sequence leaves the machine. Needs "
                "--predictor.",
            ),
            "--prediction-conda-env": (
                "Prediction",
                "NAME",
                None,
                None,
                False,
                "None",
                "Conda environment that has the model, for --predictor "
                "MODEL run from here (conda run -n NAME). Default: the "
                "model's executable on PATH, that is the current "
                "environment; an empty string says the same. For "
                "--predictor of3 it is the setting of --openfold-conda-env "
                "(default openfold3) and wins over its default; both given "
                "with different values is an error. Needs --predictor.",
            ),
            "--prediction-lock-threshold": (
                "Prediction",
                "ANGSTROM",
                None,
                None,
                False,
                "None",
                "Only for --prediction-mode score-lock: how far, in "
                "angstrom, a residue may move from the pinned template "
                "before the model pulls it back (the threshold of the "
                "forced template of Boltz-2). Default: 2.0, the choice of "
                "the Boltz-2 runner (Boltz-2 documents none). A runner or a"
                " mode that does not use it refuses it. Needs --predictor.",
            ),
            "--prediction-weights": (
                "Prediction",
                "PATH",
                None,
                None,
                False,
                "None",
                "Custom weights for the model, for instance a fine-tuned "
                "checkpoint: a file for a model that takes a checkpoint file "
                "(OpenFold3: --inference-ckpt-path), a directory for a model "
                "whose weights are a directory. It applies to --predictor "
                "MODEL run from here and to the OpenFold3 step without "
                "--predictor (binding-metrics-openfold names it --ckpt). The "
                "weights are identified by content (SHA-256) in the key of "
                "the prediction store and recorded in the results. A model "
                "whose runner cannot take custom weights is refused before "
                "anything runs. Cannot be combined with --prediction-dir: "
                "the weights are whatever made that output. Default: the "
                "model's own weights.",
            ),
            "--binder-type": (
                "Pre-flight check",
                None,
                ["auto", "peptide", "miniprotein", "nanobody", "antibody"],
                None,
                False,
                "'auto'",
                "What the binder is, for the checks that depend on it. auto "
                "(default) estimates it from the number of residues: at most "
                "40 a peptide, at most 100 a miniprotein, longer unknown, "
                "which skips the type checks. A nanobody or an antibody chain"
                " is never guessed: name it.",
            ),
            "--on-incompatible": (
                "Pre-flight check",
                None,
                ["error", "skip", "warn"],
                None,
                False,
                "'error'",
                "What to do when the input cannot go through a requested step"
                " or model, found before anything runs. error (default): "
                "refuse, listing every problem with its fix. skip: leave out "
                "the incompatible steps, record why, and run the rest. warn: "
                "log the problems and run everything. An output read with "
                "--prediction-dir only warns.",
            ),
            "--preflight-only": (
                "Pre-flight check",
                None,
                None,
                0,
                False,
                "False",
                "Print the pre-flight plan (what would run, what is "
                "incompatible and why) and stop, without preparing, relaxing "
                "or predicting anything. Exit status 1 when --on-incompatible"
                " is error and something is refused.",
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
                "Comma-separated list of metrics to compute. Valid: dockq, "
                "electrostatics, energy, geometry, interface, openfold. "
                "Default: all reference-free metrics; 'dockq' is enabled by "
                "--reference-dir.",
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
                "score: each chain is given its own structure from the "
                "input as a template and OpenFold3 places the binder "
                "itself, so its confidences refer to its own pose "
                "(binder_ca_rmsd and delta_com_angstrom show how far it "
                "is from the input pose); refold: only the receptor "
                "is templated and the binder is predicted from its "
                "sequence (binder_ca_rmsd is the refolding RMSD). "
                "Default: score",
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
                "Seed values OpenFold3 samples with, written to its runner "
                "YAML (default: 42). It makes one seed_<value> directory "
                "per seed; the first seed given, first sample, is scored. "
                "With --predictor MODEL the seeds go to that model's runner"
                " (ColabFold: consecutive integers from 0, default 0; "
                "Boltz-2: exactly one, default 42; Protenix: default 101).",
            ),
            "--openfold-cyclic": (
                "OpenFold",
                None,
                ["auto", "on", "off"],
                None,
                False,
                "'auto'",
                "Whether the binder chain of the OpenFold3 query gets 'cyclic: "
                "true' (OpenFold3 >= 0.4.5). auto (default): when the binder "
                "has a head-to-tail bond, consists of standard residues only "
                "and the installed OpenFold3 is new enough; on: always; off: "
                "never. OpenFold3 uses the flag only to wrap the relative "
                "positions of the chain: it does not enforce the closure bond, "
                "documents the flag only in an example query, and has published"
                " no accuracy benchmark for cyclic peptides. It builds the wrap"
                " from the token count of the chain and gives every atom of a "
                "modified residue its own token: for 1CWA (D-amino acid, "
                "N-methylated residues; one complex, three seeds) the flag "
                "lowered ipTM from 0.91-0.92 to 0.78-0.81 and raised the binder"
                " C-alpha RMSD from 0.5-0.7 A to 3.0-4.8 A, so auto leaves such"
                " a binder linear (on forces the flag); for SFTI-1 (standard "
                "residues; one seed) the flag closed the ring (C-N 7.40 A "
                "without it, 1.38 A with it). Disulfide, lactam and staple "
                "closures cannot be given to OpenFold3 and are not written.",
            ),
            "--openfold-no-msa-server": (
                "OpenFold",
                None,
                None,
                0,
                False,
                "False",
                "Do not use the ColabFold MSA server for OpenFold3: it then "
                "runs with a dummy MSA that holds only the query sequence of "
                "each chain (OpenFold3's input reference suggests this for "
                "MSA-free runs), which lowers accuracy for a natural receptor, "
                "but the template alignments written by the toolkit are no "
                "longer replaced by the server (issue #68). One complex (1YCR, "
                "OpenFold3 0.5.0, one seed), binder C-alpha RMSD against the "
                "crystal pose: 1.6 A with the server and no template (the "
                "server replaces the template), 21.6 A with no MSA and no "
                "template, 1.1 A with a working template and no MSA.",
            ),
            "--prediction-mode": (
                "Prediction",
                None,
                ["predict", "refold", "score", "score-lock"],
                None,
                False,
                "None",
                "How the model is used for the complex: predict (sequences "
                "only), refold (receptor templated, binder predicted "
                "freely), score (every chain templated on its own, the pose"
                " not given: re-docking) or score-lock (score, with the "
                "pose pinned to the input). It is checked against what the "
                "model supports and, for a run from here, against what its "
                "runner can do (of3: refold, score; boltz2: all four; af2 "
                "and protenix: predict), before anything runs, and "
                "recorded. Default: the runner's own mode, that is for "
                "--predictor of3 the value of --openfold-mode (score), "
                "score for boltz2 and predict for af2 and protenix; for an "
                "output read with --prediction-dir, not stated and not "
                "checked. Needs --predictor.",
            ),
            "--prediction-cyclic": (
                "Prediction",
                None,
                ["auto", "on", "off"],
                None,
                False,
                "None",
                "Whether the binder is given to the model as cyclic, for "
                "--predictor MODEL run from here. auto (default): when the "
                "binder has a head-to-tail bond (for OpenFold3 also only when "
                "it consists of standard residues: with a D-amino acid and "
                "N-methylated residues, 1CWA, the flag lowered ipTM from "
                "0.91-0.92 to 0.78-0.81 in one complex, three seeds, so auto "
                "leaves such a binder linear); on: always; off: never. "
                "OpenFold3 (>= 0.4.5) and Boltz-2 get 'cyclic: true' on the "
                "binder chain, which only wraps its relative positions and does"
                " not enforce the closure bond; Protenix gets the head-to-tail "
                "and disulfide bonds as covalent_bonds; ColabFold has no such "
                "setting and refuses a value. For --predictor of3 it is the "
                "setting of --openfold-cyclic (both given with different values"
                " is an error). Needs --predictor.",
            ),
            "--prediction-no-msa-server": (
                "Prediction",
                None,
                None,
                0,
                False,
                "False",
                "Do not use an MSA server, for --predictor MODEL run from "
                "here: ColabFold runs single_sequence, Boltz-2 writes 'msa:"
                " empty', Protenix runs with --use_msa false, OpenFold3 as "
                "--openfold-no-msa-server. The accuracy for a natural "
                "receptor drops; no sequence leaves the machine. Needs "
                "--predictor.",
            ),
            "--prediction-conda-env": (
                "Prediction",
                "NAME",
                None,
                None,
                False,
                "None",
                "Conda environment that has the model, for --predictor "
                "MODEL run from here (conda run -n NAME). Default: the "
                "model's executable on PATH, that is the current "
                "environment; an empty string says the same. For "
                "--predictor of3 it is the setting of --openfold-conda-env "
                "(default openfold3) and wins over its default; both given "
                "with different values is an error. Needs --predictor.",
            ),
            "--prediction-lock-threshold": (
                "Prediction",
                "ANGSTROM",
                None,
                None,
                False,
                "None",
                "Only for --prediction-mode score-lock: how far, in "
                "angstrom, a residue may move from the pinned template "
                "before the model pulls it back (the threshold of the "
                "forced template of Boltz-2). Default: 2.0, the choice of "
                "the Boltz-2 runner (Boltz-2 documents none). A runner or a"
                " mode that does not use it refuses it. Needs --predictor.",
            ),
            "--prediction-weights": (
                "Prediction",
                "PATH",
                None,
                None,
                False,
                "None",
                "Custom weights for the model, for instance a fine-tuned "
                "checkpoint: a file for a model that takes a checkpoint file "
                "(OpenFold3: --inference-ckpt-path), a directory for a model "
                "whose weights are a directory. It applies to --predictor "
                "MODEL run from here and to the OpenFold3 step without "
                "--predictor (binding-metrics-openfold names it --ckpt). The "
                "weights are identified by content (SHA-256) in the key of "
                "the prediction store and recorded in the results. A model "
                "whose runner cannot take custom weights is refused before "
                "anything runs. Cannot be combined with --prediction-dir: "
                "the weights are whatever made that output. Default: the "
                "model's own weights.",
            ),
            "--binder-type": (
                "Pre-flight check",
                None,
                ["auto", "peptide", "miniprotein", "nanobody", "antibody"],
                None,
                False,
                "'auto'",
                "What the binder is, for the checks that depend on it. auto "
                "(default) estimates it from the number of residues: at most "
                "40 a peptide, at most 100 a miniprotein, longer unknown, "
                "which skips the type checks. A nanobody or an antibody chain"
                " is never guessed: name it.",
            ),
            "--on-incompatible": (
                "Pre-flight check",
                None,
                ["error", "skip", "warn"],
                None,
                False,
                "'error'",
                "What to do when the input cannot go through a requested step"
                " or model, found before anything runs. error (default): "
                "refuse, listing every problem with its fix. skip: leave out "
                "the incompatible steps, record why, and run the rest. warn: "
                "log the problems and run everything. An output read with "
                "--prediction-dir only warns.",
            ),
            "--preflight-only": (
                "Pre-flight check",
                None,
                None,
                0,
                False,
                "False",
                "Print the pre-flight plan (what would run, what is "
                "incompatible and why) and stop, without preparing, relaxing "
                "or predicting anything. Exit status 1 when --on-incompatible"
                " is error and something is refused.",
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
        # A spelling may be appended to an option (``--peptide-chain`` gained
        # ``--binder-chain``); the old spellings stay first and everything else is equal.
        found = [key for key in described if key == option or key.startswith(option + "/")]
        assert len(found) == 1, f"{cli}: option {option} disappeared"
        assert described[found[0]] == expected, f"{cli}: {option} changed"


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
