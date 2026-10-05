"""Helpers of the ``test_runners_*`` modules: one stand-in model process for every runner.

No model runs. ``ModelStub`` puts a stand-in for the executable of each registered model on PATH
(``colabfold_batch``, ``boltz``, ``protenix``). It reads the command line the runner built,
records it with the input file, and writes the synthetic output of ``tests/predictors/
synth_<model>.py`` where the real model would write it. OpenFold3 is stubbed as in
``test_feat_c_support.StubOpenFold``, at the functions the runner calls. Everything else, the
runner (its request, its input files, its command line, its check of the output), the store, the
session, the adapter and the metrics, is the real code, so a setting that the command line passes
on shows up in what the stand-in recorded.

The synthetic prediction reproduces the coordinates of the input, so the binder RMSD against the
input is 0 and the adversarial check compares a structure with itself. ColabFold names the chains
``A`` (receptor) and ``B`` (binder) whatever the input calls them, and the stand-in does the same;
the other models keep the IDs of the input.
"""

import json
import os
import stat
import sys
from pathlib import Path

from binding_metrics.cli.prediction import RUNNERS
from binding_metrics.predictors.af2_runner import ColabFoldRunner
from tests.predictors import synth, synth_af2, synth_boltz2, synth_protenix
from tests.test_feat_c_support import EXAMPLE_1YCR, StubOpenFold, complex_from

#: Every registered model, so that a runner added to ``RUNNERS`` is tested without a new line.
MODELS = sorted(RUNNERS)

#: The stand-in executables: the name on PATH, and the kind the script is told.
EXECUTABLES = {"af2": "colabfold_batch", "boltz2": "boltz", "protenix": "protenix"}

#: Samples and seeds the stand-in has written, more than a run asks for.
CANNED_SEEDS = 2
CANNED_SAMPLES = 5

FAKE_MODEL = r"""
import json, os, re, shutil, sys
from pathlib import Path

kind, argv = sys.argv[1], sys.argv[2:]
canned = Path(os.environ["FAKE_MODEL_CANNED"])


def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


if os.environ.get("FAKE_MODEL_FAIL"):
    print("Traceback (most recent call last):", file=sys.stderr)
    print("RuntimeError: stand-in failure: the model cannot run", file=sys.stderr)
    sys.exit(1)
record = {"kind": kind, "argv": argv, "conda_env": os.environ.get("FAKE_MODEL_CONDA_ENV")}
if kind == "af2":
    fasta, out = Path(argv[0]), Path(argv[1])
    record["input"] = fasta.read_text(encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text("{}", encoding="utf-8")
    for seed in range(1, int(flag("--num-seeds", 1)) + 1):
        for sample in range(1, int(flag("--num-models", 5)) + 1):
            for path in (canned / f"seed{seed}" / f"sample{sample}").iterdir():
                shutil.copy2(path, out / path.name)
elif kind == "boltz2":
    yaml_path, out = Path(argv[1]), Path(flag("--out_dir"))
    record["input"] = yaml_path.read_text(encoding="utf-8")
    wanted = int(flag("--diffusion_samples", 1))
    source = canned / f"boltz_results_{yaml_path.stem}"
    for path in source.rglob("*"):
        match = re.search(r"_model_(\d+)", path.name)
        if not path.is_file() or (match and int(match.group(1)) >= wanted):
            continue
        target = out / f"boltz_results_{yaml_path.stem}" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
elif kind == "protenix":
    job = json.loads(Path(flag("--input")).read_text(encoding="utf-8"))
    record["input"] = job
    name, out = job[0]["name"], Path(flag("--out_dir"))
    template = next((canned / name).glob("seed_*")) / "predictions"
    for seed in (int(s) for s in flag("--seeds").split(",")):
        target = out / name / f"seed_{seed}" / "predictions"
        target.mkdir(parents=True, exist_ok=True)
        for rank in range(int(flag("--sample"))):
            for path in template.glob(f"*_sample_{rank}.*"):
                shutil.copy2(path, target / path.name)
with open(os.environ["FAKE_MODEL_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
"""

FAKE_CONDA = """#!/bin/sh
# conda run -n NAME --no-capture-output COMMAND...
export FAKE_MODEL_CONDA_ENV="$3"
shift 4
exec "$@"
"""


def renamed_input(path: Path, chains: dict[str, str]) -> Path:
    """A copy of ``EXAMPLE_1YCR`` with the chains renamed (``{"A": "R", "B": "L"}``)."""
    lines = []
    for line in EXAMPLE_1YCR.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) > 21:
            line = line[:21] + chains.get(line[21], line[21]) + line[22:]
        if line.startswith(("ATOM", "HETATM", "TER", "END")):
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def synthetic_for(model: str, input_path: Path, receptor: str, binder: str, rank: int = 0):
    """The synthetic complex the stand-in writes for ``model``: the input's own atoms.

    ColabFold calls the receptor ``A`` and the binder ``B``; the others keep the input's IDs.
    ``rank`` lowers the pLDDT so that the samples differ.
    """
    truth = complex_from(input_path, plddt_high=95.0 - 3.0 * rank)
    if model != "af2":
        return truth
    return synth.SyntheticComplex(
        atoms=synth.renamed_atoms(truth, {receptor: "A", binder: "B"}),
        plddt_per_atom=truth.plddt_per_atom,
        pae=truth.pae,
        pde=truth.pde,
        scalars=truth.scalars,
        chain_ptm=truth.chain_ptm,
        chain_pair_iptm=truth.chain_pair_iptm,
    )


class ModelStub:
    """The stand-in of one model, whichever it is; see the module docstring.

    Use ``ModelStub(model, tmp_path, monkeypatch, input_path, receptor, binder, names)`` once per
    test, where ``names`` are the sample IDs the test runs. ``calls`` are what the stand-in
    recorded (the executable's arguments and the input file it was given; for OpenFold3 the
    keyword arguments of the run function), ``starts`` how many times the model started.
    """

    def __init__(
        self,
        model,
        tmp_path,
        monkeypatch,
        input_path=EXAMPLE_1YCR,
        receptor="A",
        binder="B",
        names=None,
    ):
        self.model = model
        self.monkeypatch = monkeypatch
        self.tmp_path = Path(tmp_path)
        self.names = list(names or [Path(input_path).stem])
        if model == "of3":
            self._of3 = StubOpenFold(monkeypatch)
            return
        self._of3 = None
        self.log = self.tmp_path / "model_calls.jsonl"
        canned = self.tmp_path / "canned"
        bin_dir = self.tmp_path / "fakebin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        script = bin_dir / "fake_model.py"
        script.write_text(FAKE_MODEL, encoding="utf-8")
        wrapper = bin_dir / EXECUTABLES[model]
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" {model} "$@"\n', encoding="utf-8"
        )
        conda = bin_dir / "conda"
        conda.write_text(FAKE_CONDA, encoding="utf-8")
        for path in (wrapper, conda):
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_MODEL_LOG", str(self.log))
        monkeypatch.setenv("FAKE_MODEL_CANNED", str(canned))
        monkeypatch.setenv("PROTENIX_ROOT_DIR", str(self.tmp_path / "protenix_root"))
        self._no_installation_probe(model)
        for name in self.names:
            self._write_canned(canned, name, input_path, receptor, binder)

    def _no_installation_probe(self, model):
        """The version is fixed: no model is installed here, and a probe would start a process."""
        from binding_metrics.predictors import boltz2_runner, protenix_runner

        patch = self.monkeypatch
        if model == "af2":
            patch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.6.3")
        elif model == "boltz2":
            patch.setattr(boltz2_runner, "_installed_boltz_version", lambda conda_env: "2.2.1")
        elif model == "protenix":
            patch.setattr(protenix_runner, "_installed_version", lambda python_cmd=None: "2.0.0")

    def _write_canned(self, canned, name, input_path, receptor, binder):
        if self.model == "af2":
            for seed in range(1, CANNED_SEEDS + 1):
                for sample in range(1, CANNED_SAMPLES + 1):
                    synth_af2.write_colabfold(
                        canned / f"seed{seed}" / f"sample{sample}",
                        name,
                        synthetic_for("af2", input_path, receptor, binder, sample - 1),
                        seed_index=seed,
                        sample=sample,
                    )
            return
        writer = {"boltz2": synth_boltz2, "protenix": synth_protenix}[self.model]
        for sample in range(1, CANNED_SAMPLES + 1):
            writer.write_prediction(
                canned,
                name,
                synthetic_for(self.model, input_path, receptor, binder, sample - 1),
                sample=sample,
            )

    def fail(self) -> None:
        """Make every later start of the model fail with a status other than 0."""
        if self._of3 is not None:
            self._of3.error = RuntimeError("stand-in failure: the model cannot run")
        else:
            self.monkeypatch.setenv("FAKE_MODEL_FAIL", "1")

    @property
    def calls(self) -> list[dict]:
        if self._of3 is not None:
            return self._of3.calls
        if not self.log.is_file():
            return []
        text = self.log.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    @property
    def starts(self) -> int:
        """How many times the model was started."""
        return len(self.calls)

    def argument(self, flag: str):
        """The value after ``flag`` in the command line of the last start (not for OpenFold3)."""
        argv = self.calls[-1]["argv"]
        return argv[argv.index(flag) + 1] if flag in argv else None


def default_mode_of(model: str) -> str:
    """The default mode of the runner of ``model``, as documented (``--prediction-mode``)."""
    return {"af2": "predict", "boltz2": "score", "of3": "score", "protenix": "predict"}[model]
