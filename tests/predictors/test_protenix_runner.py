"""ProtenixRunner: the input JSON, the command line, the modes and the failures.

No model, GPU or network is used. The process is a stub ``protenix`` executable on PATH that logs
its arguments and writes a synthetic output in the layout of ``predictors/protenix.py``
(``tests/predictors/synth_protenix.py``). The tests that read the Protenix source run only when
``BINDING_METRICS_MODEL_SOURCES`` names a folder with a clone at ``Protenix/`` (commit 85767b8);
they pin the statements the runner's docstring cites.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from binding_metrics.predictors import protenix_runner
from binding_metrics.predictors.protenix import ProtenixParser
from binding_metrics.predictors.protenix_runner import (
    CONSTRAINT_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    ProtenixRunError,
    ProtenixRunner,
)
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import (
    PredictionFailedError,
    PredictionRequest,
    PredictionStore,
    PredictionUnavailableError,
)

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"
STAPLED = DATA / "example_staple_3V3B.pdb"


def head_to_tail_bond(length):
    """The documented head-to-tail bond inside the binder (entity 2, copy 1): C of the last."""
    return {
        "entity1": 2, "copy1": 1, "position1": length, "atom1": "C",
        "entity2": 2, "copy2": 1, "position2": 1, "atom2": "N",
    }  # fmt: skip


def disulfide_bond(first, second):
    return {
        "entity1": 2, "copy1": 1, "position1": first, "atom1": "SG",
        "entity2": 2, "copy2": 1, "position2": second, "atom2": "SG",
    }  # fmt: skip


MDM2 = "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
P53_PEPTIDE = "ETFSDLWKLLPEN"
CYCLOPHILIN = (
    "MVNPTVFFDIAVDGEPLGRVSFELFADKVPKTAENFRALSTGEKGFGYKGSCFHRIIPGFMCQGGDFTRHNGTGGKSIYGEKFEDEN"
    "FILKHTGPGILSMANAGPNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE"
)

STUB_SCRIPT = """#!{python}
import json, os, pathlib, shutil, sys
config = json.loads(pathlib.Path({config!r}).read_text(encoding="utf-8"))
argv = sys.argv[1:]
args = {{}}
for index, token in enumerate(argv):
    if token.startswith("--") and index + 1 < len(argv):
        args[token[2:]] = argv[index + 1]
job = json.loads(pathlib.Path(args["input"]).read_text(encoding="utf-8"))
with open({record!r}, "a", encoding="utf-8") as handle:
    root = os.environ.get("PROTENIX_ROOT_DIR", "")
    seen = {{"root": root}}
    if root and os.path.isdir(root):
        seen["entries"] = sorted(os.listdir(root))
        seen["links"] = sorted(e for e in os.listdir(root) if os.path.islink(os.path.join(root, e)))
        checkpoint = os.path.join(root, "checkpoint")
        if os.path.isdir(checkpoint):
            seen["checkpoint"] = sorted(os.listdir(checkpoint))
            seen["checkpoint_target"] = os.path.realpath(checkpoint)
        common = os.path.join(root, "common")
        seen["common"] = sorted(os.listdir(common)) if os.path.isdir(common) else None
    handle.write(json.dumps({{"argv": argv, "job": job, "cwd": os.getcwd(), "env": seen}}) + "\\n")
name = job[0]["name"]
out = pathlib.Path(args["out_dir"])
behavior = config["behavior"]
if behavior == "exit":
    sys.stderr.write("Traceback (most recent call last):\\n")
    sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\\n")
    sys.exit(1)
if behavior == "run_failed":
    sys.stderr.write("2026-10-01 ERROR Run inference failed: {{'in.json': 'MSA boom'}}\\n")
    sys.exit(0)
if behavior in ("err_file", "error_txt"):
    (out / "ERR").mkdir(parents=True, exist_ok=True)
    file_name = name + ".txt" if behavior == "err_file" else "error.txt"
    (out / "ERR" / file_name).write_text(
        "[Rank 0] " + name + " failed: boom\\nTraceback (most recent call last):\\n"
        '  File "x.py", line 1\\nValueError: boom\\n', encoding="utf-8")
    sys.exit(0)
if behavior == "msa_failure":
    print("MMSEQS2 failed with the following error message:")
    print("ConnectionError: server unreachable")
    print("Failed in searching MSA for")
    print("SEQ")
    print("using the sequence itself as MSA.")
seeds = [int(s) for s in args["seeds"].split(",")]
if behavior == "partial":
    seeds = seeds[:1]
    (out / "ERR").mkdir(parents=True, exist_ok=True)
    (out / "ERR" / (name + ".txt")).write_text("[Rank 0] failed: seed two\\n", encoding="utf-8")
template = pathlib.Path(config["template"]) / name / "seed_9" / "predictions"
samples = config.get("samples") or int(args["sample"])
for seed in seeds:
    target = out / name / ("seed_%d" % seed) / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for rank in range(samples):
        for path in template.glob("*_sample_%d.*" % rank):
            if behavior == "no_full_data" and "_full_data_" in path.name:
                continue
            shutil.copy2(path, target / path.name)
(out / "msa").mkdir(exist_ok=True)
"""

CONDA_SCRIPT = """#!{python}
import json, os, sys
arguments = sys.argv[1:]
with open({record!r}, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(arguments) + "\\n")
assert arguments[:2] == ["run", "-n"], arguments
rest = arguments[arguments.index("--no-capture-output") + 1:]
os.execvp(rest[0], rest)
"""


@pytest.fixture(autouse=True)
def protenix_environment(tmp_path, monkeypatch):
    """No Protenix install and no weights on this machine, whatever the machine has."""
    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(tmp_path / "protenix_root"))
    monkeypatch.delenv("MMSEQS_SERVICE_HOST_URL", raising=False)
    asked = []

    def fake_version(python_cmd=None):
        asked.append(python_cmd)
        return "2.0.0"

    monkeypatch.setattr(protenix_runner, "_installed_version", fake_version)
    return asked


@pytest.fixture
def structure(tmp_path):
    """A structure file of two tiny chains; the content differs per test through ``tmp_path``."""
    return tiny_structure(tmp_path / "tiny.pdb")


def pdb_line(serial, atom, residue, chain, number, x):
    return (
        f"ATOM  {serial:5d} {' ' + atom:<4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           {atom[0]:>1s}"
    )


def tiny_structure(path, residues_a=("ALA", "GLY", "ALA"), residues_b=("SER", "ALA")):
    """A PDB file with chain A and chain B; each residue has N, CA and C."""
    lines, serial = [], 0
    for chain, residues in (("A", residues_a), ("B", residues_b)):
        for number, residue in enumerate(residues, start=1):
            for atom, shift in (("N", 0.0), ("CA", 1.45), ("C", 2.5)):
                serial += 1
                lines.append(pdb_line(serial, atom, residue, chain, number, 3.8 * number + shift))
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


def make(runner=None, path=P53, **kwargs):
    runner = runner or ProtenixRunner()
    kwargs.setdefault("name", "p53")
    kwargs.setdefault("binder_chain", "B")
    kwargs.setdefault("receptor_chain", "A")
    return runner.make_request(path, **kwargs)


@pytest.fixture
def stub_protenix(tmp_path, monkeypatch):
    """A ``protenix`` on PATH that writes a synthetic output; returns the log of its calls."""
    pytest.importorskip("gemmi")
    from tests.predictors import synth, synth_protenix

    record = tmp_path / "calls.jsonl"
    config_path = tmp_path / "stub_config.json"
    template = tmp_path / "template_output"
    for sample in range(1, 4):  # ranks 0, 1, 2 of the seed 9; each sample has other pLDDTs
        synth_protenix.write_prediction(
            template, "p53", synth.synthetic_complex(plddt_shift=2.0 * (sample - 1)), sample=sample
        )

    def install(behavior="ok", **config):
        config_path.write_text(
            json.dumps({"behavior": behavior, "template": str(template), **config}),
            encoding="utf-8",
        )
        script = tmp_path / "bin" / "protenix"
        script.parent.mkdir(exist_ok=True)
        script.write_text(
            STUB_SCRIPT.format(python=sys.executable, config=str(config_path), record=str(record)),
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")

    def calls():
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]

    install()
    return type("Stub", (), {"install": staticmethod(install), "calls": staticmethod(calls)})


# ---------------------------------------------------------------------------- the input JSON


class TestInputJson:
    def test_a_linear_peptide_has_the_two_entities_with_the_exact_keys(self, tmp_path):
        runner = ProtenixRunner()
        path = runner.prepare(make(runner), tmp_path / "work")
        assert path == (tmp_path / "work" / "input" / "p53.json").resolve()
        assert json.loads(path.read_text(encoding="utf-8")) == [
            {
                "name": "p53",
                "sequences": [
                    {"proteinChain": {"sequence": MDM2, "count": 1, "id": ["A"]}},
                    {"proteinChain": {"sequence": P53_PEPTIDE, "count": 1, "id": ["B"]}},
                ],
            }
        ]

    def test_a_cyclic_peptide_with_modified_residues_names_them_by_ccd_code(self, tmp_path):
        runner = ProtenixRunner()
        request = make(runner, CYCLOSPORIN, name="cwa", binder_chain="C", receptor_chain="A")
        path = runner.prepare(request, tmp_path)
        (job,) = json.loads(path.read_text(encoding="utf-8"))
        receptor, binder = job["sequences"]
        assert receptor == {"proteinChain": {"sequence": CYCLOPHILIN, "count": 1, "id": ["A"]}}
        modifications = [
            ("DAL", 1), ("MLE", 2), ("MLE", 3), ("MVA", 4), ("BMT", 5),
            ("ABA", 6), ("SAR", 7), ("MLE", 8), ("MLE", 10),
        ]  # fmt: skip
        assert binder == {
            "proteinChain": {
                "sequence": "ALLVTAGLVLA",
                "count": 1,
                "id": ["C"],
                "modifications": [
                    {"ptmType": f"CCD_{code}", "ptmPosition": position}
                    for code, position in modifications
                ],
            }
        }
        assert job["covalent_bonds"] == [head_to_tail_bond(11)]
        assert set(job) == {"name", "sequences", "covalent_bonds"}

    def test_the_file_is_ascii_and_a_list_of_one_job(self, tmp_path):
        runner = ProtenixRunner()
        path = runner.prepare(make(runner, name="peptide ü"), tmp_path)
        text = path.read_bytes().decode("ascii")
        assert isinstance(json.loads(text), list) and len(json.loads(text)) == 1

    def test_the_constraint_section_is_written_as_given(self, tmp_path):
        runner = ProtenixRunner()
        pocket = {
            "pocket": {
                "binder_chain": {"entity": 2, "copy": 1},
                "contact_residues": [{"entity": 1, "copy": 1, "position": 26}],
                "max_distance": 8,
            }
        }
        request = make(runner, model_name=CONSTRAINT_MODEL_NAME, constraints=pocket)
        (job,) = json.loads(runner.prepare(request, tmp_path).read_text(encoding="utf-8"))
        assert job["constraint"] == pocket
        assert "covalent_bonds" not in job

    def test_the_covalent_bonds_are_written_as_given(self, tmp_path):
        runner = ProtenixRunner()
        bond = {
            "entity1": 2, "copy1": 1, "position1": 13, "atom1": "C",
            "entity2": 2, "copy2": 1, "position2": 1, "atom2": "N",
        }  # fmt: skip
        request = make(runner, covalent_bonds=[bond])
        (job,) = json.loads(runner.prepare(request, tmp_path).read_text(encoding="utf-8"))
        assert job["covalent_bonds"] == [bond]
        assert "constraint" not in job

    def test_selenocysteine_is_an_x_with_its_ccd_code(self):
        sequence, modifications = protenix_runner._protenix_chain("AUG", {})
        assert sequence == "AXG"
        assert modifications == [{"ptmType": "CCD_SEC", "ptmPosition": 2}]

    def test_a_letter_protenix_does_not_take_is_refused(self):
        with pytest.raises(ValueError, match="'B' at position 2"):
            protenix_runner._protenix_chain("ABG", {})

    def test_a_chain_the_structure_lacks_is_named_with_the_chains_it_has(self, tmp_path):
        runner = ProtenixRunner()
        with pytest.raises(ValueError, match=r"chain 'Z'.*chains in its first model: A, B"):
            runner.prepare(make(runner, receptor_chain="Z"), tmp_path)


class TestAResidueProtenixCannotTake:
    """It stops before a file is written, and the process is not started."""

    @pytest.fixture
    def odd_structure(self, tmp_path):
        return tiny_structure(
            tmp_path / "odd.pdb", residues_a=("ALA", "ZZZ", "ALA"), residues_b=("SER", "QQQ")
        )

    def test_prepare_names_every_chain_and_residue_and_writes_nothing(
        self, tmp_path, odd_structure
    ):
        pytest.importorskip("gemmi")
        runner = ProtenixRunner()
        request = make(runner, odd_structure)
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with pytest.raises(ValueError, match="Protenix cannot take these residues") as info:
            runner.prepare(request, work_dir)
        message = str(info.value)
        assert "chain 'A': ZZZ 2" in message and "chain 'B': QQQ 2" in message
        assert "docs/infer_json_format.md:60-65" in message
        assert list(work_dir.iterdir()) == []

    def test_run_does_not_start_the_process(self, tmp_path, odd_structure, stub_protenix):
        runner = ProtenixRunner()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with pytest.raises(ValueError, match="ZZZ 2"):
            runner.run(make(runner, odd_structure), work_dir)
        assert stub_protenix.calls() == []
        assert list(work_dir.iterdir()) == []

    def test_the_store_records_the_reason_and_starts_nothing(
        self, tmp_path, odd_structure, stub_protenix
    ):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        with pytest.raises(PredictionFailedError, match="ZZZ 2"):
            store.get_or_run(make(runner, odd_structure), runner)
        assert stub_protenix.calls() == []

    def test_x_sends_an_unknown_residue_in_their_place(self, tmp_path, odd_structure):
        pytest.importorskip("gemmi")
        runner = ProtenixRunner()
        request = make(runner, odd_structure, on_unmappable_residue="x")
        (job,) = json.loads(runner.prepare(request, tmp_path).read_text(encoding="utf-8"))
        sequences = [entity["proteinChain"]["sequence"] for entity in job["sequences"]]
        assert sequences == ["AXA", "SX"]


# ---------------------------------------------------------------------------- the modes


class TestModes:
    @pytest.mark.parametrize("mode", ["refold", "score", "score-lock"])
    def test_a_mode_other_than_predict_raises_with_the_documented_routes(self, mode):
        with pytest.raises(ValueError) as info:
            make(mode=mode)
        message = str(info.value)
        assert f"supports mode 'predict' only, not '{mode}'" in message
        assert "`templatesPath`" in message and "docs/infer_json_format.md:68" in message
        assert "`constraint` section with `contact` and `pocket`" in message
        assert "docs/infer_json_format.md:252-254" in message
        assert "docs/infer_json_format.md:256" in message
        assert (
            "a soft constraint: the model is encouraged, but not strictly required, to satisfy it"
            in message
        )
        assert "do not pin the complete pose of the chains" in message
        assert "not verified" in message

    def test_each_mode_is_described_by_what_it_needs(self):
        needs = {
            "refold": "gives the receptor as a template",
            "score": "gives every chain its own structure as a template",
            "score-lock": "pins the relative pose of the chains",
        }
        for mode, text in needs.items():
            with pytest.raises(ValueError, match=text):
                make(mode=mode)

    def test_an_unknown_mode_is_not_described_as_a_template_problem(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            make(mode="dock")

    @pytest.mark.parametrize("mode", ["refold", "score", "score-lock"])
    def test_a_request_made_elsewhere_in_another_mode_is_refused_by_prepare_and_run(
        self, tmp_path, stub_protenix, mode
    ):
        runner = ProtenixRunner()
        request = PredictionRequest(
            "protenix", "p53", mode=mode, input_path=P53, binder_chain="B", receptor_chain="A"
        )
        with pytest.raises(ValueError, match=f"not '{mode}'"):
            runner.prepare(request, tmp_path)
        with pytest.raises(ValueError, match=f"not '{mode}'"):
            runner.run(request, tmp_path)
        assert stub_protenix.calls() == []

    def test_constraints_need_the_model_that_reads_them(self):
        with pytest.raises(ValueError, match="read by the model protenix_base_constraint_v0.5.0"):
            make(constraints={"contact": []})
        request = make(model_name=CONSTRAINT_MODEL_NAME, constraints={"contact": []})
        assert request.options["constraints"] == {"contact": []}

    def test_an_empty_constraint_section_is_no_section_for_any_model(self):
        assert make(constraints={}).options["constraints"] is None
        assert make(constraints={}).key() == make().key()

    def test_constraints_that_are_not_a_dict_are_refused(self):
        with pytest.raises(ValueError, match="must be a dict"):
            make(model_name=CONSTRAINT_MODEL_NAME, constraints=[{"entity1": 1}])


# ---------------------------------------------------------------------------- the closures


class TestClosures:
    """``binder_cyclic`` writes the documented closures of the binder as ``covalent_bonds``."""

    def job(self, tmp_path, path, binder, receptor, **kwargs):
        runner = ProtenixRunner()
        request = make(runner, path, binder_chain=binder, receptor_chain=receptor, **kwargs)
        (job,) = json.loads(runner.prepare(request, tmp_path).read_text(encoding="utf-8"))
        return job

    def test_the_default_is_auto_and_a_head_to_tail_binder_gets_its_bond(self, tmp_path):
        assert make().options["binder_cyclic"] == "auto"
        job = self.job(tmp_path, CYCLOSPORIN, "C", "A", name="cwa")
        assert job["covalent_bonds"] == [head_to_tail_bond(11)]  # 11 residues: ALA 11 C to DAL 1 N

    def test_a_bicyclic_binder_gets_the_head_to_tail_bond_and_the_disulfide(self, tmp_path):
        job = self.job(tmp_path, SFTI1, "I", "A", name="sfti")
        (_, binder) = job["sequences"]
        assert len(binder["proteinChain"]["sequence"]) == 14  # the waters of the chain are not sent
        assert job["covalent_bonds"] == [head_to_tail_bond(14), disulfide_bond(3, 11)]

    def test_a_linear_binder_gets_none(self, tmp_path):
        assert "covalent_bonds" not in self.job(tmp_path, P53, "B", "A")

    @pytest.mark.parametrize(
        "path, binder, receptor",
        [(CYCLOSPORIN, "C", "A"), (SFTI1, "I", "A"), (P53, "B", "A")],
    )
    def test_false_writes_none(self, tmp_path, path, binder, receptor):
        assert "covalent_bonds" not in self.job(
            tmp_path, path, binder, receptor, binder_cyclic=False
        )

    def test_true_on_a_linear_binder_writes_the_head_to_tail_bond(self, tmp_path):
        job = self.job(tmp_path, P53, "B", "A", binder_cyclic=True)
        assert job["covalent_bonds"] == [head_to_tail_bond(13)]  # 13 residues; no ring in the input

    def test_true_on_a_closed_binder_does_not_write_the_bond_twice(self, tmp_path):
        job = self.job(tmp_path, SFTI1, "I", "A", binder_cyclic=True)
        assert job["covalent_bonds"] == [head_to_tail_bond(14), disulfide_bond(3, 11)]

    def test_a_binder_of_one_residue_cannot_be_closed_head_to_tail(self, structure, tmp_path):
        single = tiny_structure(tmp_path / "single.pdb", residues_b=("ALA",))
        runner = ProtenixRunner()
        request = make(runner, single, binder_cyclic=True)
        with pytest.raises(ValueError, match="at least two residues, chain B has 1"):
            runner.prepare(request, tmp_path / "work")
        assert not (tmp_path / "work").exists()

    @pytest.mark.parametrize("bad", ["on", 1, 0, None, "yes"])
    def test_a_value_that_is_not_true_false_or_auto_is_refused(self, bad):
        with pytest.raises(ValueError, match='binder_cyclic must be True, False or "auto"'):
            make(binder_cyclic=bad)

    def test_a_lactam_and_a_staple_are_not_written_and_the_run_says_so(self, caplog):
        pytest.importorskip("gemmi")
        for path, chain, expected in [
            (SOMATOSTATIN, "A", [disulfide_bond(2, 12)]),
            (STAPLED, "C", []),
        ]:
            (entity,) = protenix_runner._read_entities(path, [chain], "error")
            caplog.clear()
            with caplog.at_level("WARNING", logger=protenix_runner.logger.name):
                bonds = protenix_runner._closure_bonds(path, chain, entity[3], "auto")
            assert bonds == expected
            assert "not written as covalent_bonds" in caplog.text
            assert "docs/infer_json_format.md:224-229" in caplog.text
        assert "lactam" in caplog.text or "staple" in caplog.text

    def test_the_residues_of_the_sequence_are_the_ones_the_query_reader_counts(self):
        (entity,) = protenix_runner._read_entities(SFTI1, ["I"], "error")
        residues = entity[3]
        assert len(residues) == len(entity[1]) == 14
        assert residues[0] == ("GLY", 1) and residues[-1] == ("ASP", 14)
        assert residues[2] == ("CYS", 3) and residues[10] == ("CYS", 11)

    def test_the_structure_is_not_searched_for_closures_when_false(self, tmp_path, monkeypatch):
        import binding_metrics.capabilities as capabilities

        def fail(*args, **kwargs):
            raise AssertionError("the closures were searched")

        monkeypatch.setattr(capabilities, "detect_closures", fail)
        assert "covalent_bonds" not in self.job(
            tmp_path, CYCLOSPORIN, "C", "A", binder_cyclic=False
        )

    def test_a_search_that_fails_is_an_error_and_not_a_linear_prediction(
        self, tmp_path, monkeypatch
    ):
        import binding_metrics.capabilities as capabilities

        def broken(atoms, chain_id):
            raise RuntimeError("cannot read the bond table")

        monkeypatch.setattr(capabilities, "detect_closures", broken)
        runner = ProtenixRunner()
        with pytest.raises(ValueError, match="could not look for ring closures in chain C") as info:
            runner.prepare(make(runner, CYCLOSPORIN, binder_chain="C", name="c"), tmp_path)
        assert "cannot read the bond table" in str(info.value)
        assert "binder_cyclic=False" in str(info.value) and "binder_cyclic=True" in str(info.value)
        assert not (tmp_path / "input").exists()

    def closure(self, family, end1, end2):
        from binding_metrics.capabilities import Closure, ClosureEnd

        return Closure(family, family, ClosureEnd(*end1), ClosureEnd(*end2))

    @pytest.mark.parametrize(
        "family, end1, end2, message",
        [
            ("head_to_tail", (0, "DAL", 1, "C"), (10, "ALA", 11, "N"), "is not between C"),
            ("head_to_tail", (10, "ALA", 11, "O"), (0, "DAL", 1, "N"), "is not between C"),
            ("head_to_tail", (10, "ALA", 9, "C"), (0, "DAL", 1, "N"), "residue ALA 9 is not in"),
            ("disulfide", (1, "ALA", 11, "CB"), (2, "DAL", 1, "SG"), "not between two SG"),
        ],
    )
    def test_a_closure_that_cannot_be_placed_raises_and_names_the_way_out(
        self, tmp_path, monkeypatch, family, end1, end2, message
    ):
        import binding_metrics.capabilities as capabilities

        monkeypatch.setattr(
            capabilities, "detect_closures", lambda atoms, chain: [self.closure(family, end1, end2)]
        )
        runner = ProtenixRunner()
        request = make(runner, CYCLOSPORIN, binder_chain="C", name="c")
        with pytest.raises(ValueError, match=message) as info:
            runner.prepare(request, tmp_path)
        assert "binder_cyclic=False" in str(info.value)

    def test_a_residue_that_is_there_twice_is_not_guessed(self, monkeypatch):
        import binding_metrics.capabilities as capabilities

        monkeypatch.setattr(
            capabilities,
            "detect_closures",
            lambda atoms, chain: [
                self.closure("head_to_tail", (10, "ALA", 11, "C"), (0, "DAL", 1, "N"))
            ],
        )
        residues = [("DAL", 1), ("ALA", 11)] + [("ALA", 11)]
        with pytest.raises(ValueError, match="is twice in the 3 residues"):
            protenix_runner._closure_bonds(CYCLOSPORIN, "C", residues, "auto")

    def test_given_bonds_follow_the_closures_and_a_repeat_is_dropped(self, tmp_path):
        reversed_bond = {
            "entity1": 2, "copy1": 1, "position1": 1, "atom1": "N",
            "entity2": 2, "copy2": 1, "position2": 11, "atom2": "C",
        }  # fmt: skip
        extra = {
            "entity1": 1, "copy1": 1, "position1": 5, "atom1": "SG",
            "entity2": 2, "copy2": 1, "position2": 3, "atom2": "SG",
        }  # fmt: skip
        job = self.job(
            tmp_path, CYCLOSPORIN, "C", "A", name="cwa", covalent_bonds=[reversed_bond, extra]
        )
        assert job["covalent_bonds"] == [head_to_tail_bond(11), extra]

    def test_given_bonds_alone_are_written_when_the_binder_is_not_closed(self, tmp_path):
        job = self.job(tmp_path, P53, "B", "A", covalent_bonds=[disulfide_bond(1, 2)])
        assert job["covalent_bonds"] == [disulfide_bond(1, 2)]

    def test_the_choice_is_in_the_key_and_the_structure_decides_auto(self):
        keys = {make(binder_cyclic=value).key() for value in ("auto", True, False)}
        assert len(keys) == 3

    def test_a_request_made_elsewhere_without_the_option_means_auto(self, tmp_path):
        request = PredictionRequest(
            "protenix",
            "cwa",
            mode="predict",
            input_path=CYCLOSPORIN,
            binder_chain="C",
            receptor_chain="A",
        )
        (job,) = json.loads(ProtenixRunner().prepare(request, tmp_path).read_text(encoding="utf-8"))
        assert job["covalent_bonds"] == [head_to_tail_bond(11)]

    def test_the_stub_process_receives_the_bonds_in_its_input(self, tmp_path, stub_protenix):
        runner = ProtenixRunner()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        request = make(runner, CYCLOSPORIN, binder_chain="C", name="p53", seeds=(3,), num_samples=1)
        runner.run(request, work_dir)
        (call,) = stub_protenix.calls()
        assert call["job"][0]["covalent_bonds"] == [head_to_tail_bond(11)]


# ---------------------------------------------------------------------------- the weights


DATA_FILES = (
    "components.cif",
    "components.cif.rdkit_mol.pkl",
    "obsolete_release_date.csv",
    "clusters-by-entity-40.txt",
)


class TestWeights:
    """Weights are a directory plus a model name; a linked root carries them to `protenix pred`."""

    MODEL_FILE = f"{DEFAULT_MODEL_NAME}.pt"

    @pytest.fixture
    def real_root(self, tmp_path):
        """The data root of this machine: the CCD files, a template database, no weights."""
        root = tmp_path / "protenix_root"
        (root / "common").mkdir(parents=True)
        for name in DATA_FILES:
            (root / "common" / name).write_text(name, encoding="utf-8")
        (root / "search_database").mkdir()
        (root / "checkpoint").mkdir()
        (root / "checkpoint" / self.MODEL_FILE).write_bytes(b"default weights")
        return root

    @pytest.fixture
    def weights_dir(self, tmp_path):
        directory = tmp_path / "my_weights"
        directory.mkdir()
        (directory / self.MODEL_FILE).write_bytes(b"fine-tuned")
        return directory

    def request(self, weights, runner=None, **kwargs):
        return make(
            runner or ProtenixRunner(), weights=weights, seeds=(3,), num_samples=1, **kwargs
        )

    def test_the_runner_declares_what_protenix_takes(self):
        assert ProtenixRunner.supports_custom_weights is True
        assert ProtenixRunner.weights_kind == "directory"

    def test_the_request_holds_the_weights_by_content_and_not_the_default_file(
        self, real_root, weights_dir, tmp_path
    ):
        request = self.request(weights_dir)
        assert request.weights.kind == "directory" and request.weights.path == weights_dir
        assert request.options["checkpoint_size_bytes"] is None
        assert request.canonical()["weights"] == request.weights.key_fields()
        assert str(weights_dir) not in json.dumps(request.canonical())
        assert request.key() != make().key()
        copy = tmp_path / "moved"
        shutil.copytree(weights_dir, copy)
        assert self.request(copy).key() == request.key()
        (copy / self.MODEL_FILE).write_bytes(b"other")
        assert self.request(copy).key() != request.key()
        assert request.canonical()["options"]["model_name"] == DEFAULT_MODEL_NAME

    def test_a_reference_made_by_the_store_is_the_same_request(
        self, tmp_path, real_root, weights_dir
    ):
        reference = PredictionStore(tmp_path / "store").weights_reference(weights_dir)
        assert self.request(reference).key() == self.request(weights_dir).key()

    def test_the_model_name_chooses_the_file_in_the_directory(self, real_root, weights_dir):
        (weights_dir / "protenix-v2.pt").write_bytes(b"v2")
        assert (
            self.request(weights_dir, model_name="protenix-v2").key()
            != self.request(weights_dir).key()
        )

    def test_a_directory_without_the_model_file_is_refused_because_protenix_would_download_it(
        self, real_root, weights_dir
    ):
        with pytest.raises(ValueError) as info:
            self.request(weights_dir, model_name="protenix-v2")
        message = str(info.value)
        assert f"{weights_dir} has no protenix-v2.pt" in message
        assert "runner/inference.py:445-489" in message and "into this directory" in message

    @pytest.mark.parametrize(
        "model_name, extra",
        [
            ("protenix_mini_esm_v0.5.0", ["esm2_t36_3B_UR50D.pt"]),
            ("protenix_mini_ism_v0.5.0", ["esm2_t36_3B_UR50D_ism.pt"]),
        ],
    )
    def test_a_language_model_needs_its_own_files_in_the_directory(
        self, real_root, weights_dir, model_name, extra
    ):
        (weights_dir / f"{model_name}.pt").write_bytes(b"x")
        with pytest.raises(ValueError, match=extra[0].replace(".", r"\.")):
            self.request(weights_dir, model_name=model_name)

    def test_a_file_is_not_the_kind_of_weights_the_runner_takes(self, real_root, weights_dir):
        reference = PredictionStore(weights_dir / "store").weights_reference(
            weights_dir / self.MODEL_FILE
        )
        with pytest.raises(ValueError, match="is a file"):
            self.request(reference)
        with pytest.raises(ValueError, match="is not one"):
            self.request(weights_dir / self.MODEL_FILE)

    def test_a_real_root_without_the_ccd_files_is_refused_before_the_run(
        self, real_root, weights_dir
    ):
        (real_root / "common" / "components.cif").unlink()
        with pytest.raises(ValueError) as info:
            self.request(weights_dir)
        message = str(info.value)
        assert "components.cif missing" in message and "download them into it" in message
        assert "configs/configs_data.py:298-306" in message

    def test_without_any_data_root_the_message_names_the_missing_directory(
        self, tmp_path, weights_dir
    ):
        with pytest.raises(ValueError, match=r"protenix_root/common \(components\.cif"):
            self.request(weights_dir)

    def test_an_empty_root_variable_is_refused(self, monkeypatch, weights_dir):
        monkeypatch.setenv("PROTENIX_ROOT_DIR", "")
        with pytest.raises(ValueError, match="empty string"):
            self.request(weights_dir)

    def test_the_checks_are_made_again_by_check_weights_prepare_and_run(
        self, tmp_path, real_root, weights_dir, stub_protenix
    ):
        runner = ProtenixRunner()
        request = self.request(weights_dir)
        (weights_dir / self.MODEL_FILE).unlink()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        for call in (runner.check_weights, lambda r: runner.prepare(r, work_dir)):
            with pytest.raises(ValueError, match="has no protenix_base_default_v1.0.0.pt"):
                call(request)
        with pytest.raises(ValueError, match="has no protenix_base_default_v1.0.0.pt"):
            runner.run(request, work_dir)
        assert list(work_dir.iterdir()) == [] and stub_protenix.calls() == []

    def test_a_request_made_elsewhere_with_a_file_is_refused_by_the_check_of_the_base_class(
        self, real_root, weights_dir
    ):
        request = PredictionRequest(
            "protenix",
            "p53",
            mode="predict",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            weights=weights_dir / self.MODEL_FILE,
        )
        with pytest.raises(ValueError, match="takes its weights as a directory"):
            ProtenixRunner().check_weights(request)

    def test_the_run_gives_protenix_a_root_that_links_the_real_data_and_the_weights(
        self, tmp_path, real_root, weights_dir, stub_protenix
    ):
        runner = ProtenixRunner()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        runner.run(self.request(weights_dir), work_dir)
        (call,) = stub_protenix.calls()
        seen = call["env"]
        assert seen["root"] == str(work_dir.resolve() / "protenix_root")
        assert seen["entries"] == ["checkpoint", "common", "search_database"]
        assert seen["links"] == ["checkpoint", "common", "search_database"]
        assert seen["checkpoint"] == [self.MODEL_FILE]
        assert seen["checkpoint_target"] == str(weights_dir.resolve())
        assert seen["common"] == sorted(DATA_FILES)  # the real data, not a download
        argv = call["argv"]  # the command is the one without weights
        assert argv[argv.index("--model_name") + 1] == DEFAULT_MODEL_NAME
        assert "--load_checkpoint_dir" not in argv

    def test_the_linked_root_is_removed_and_nothing_real_is_touched(
        self, tmp_path, real_root, weights_dir, stub_protenix
    ):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        ProtenixRunner().run(self.request(weights_dir), work_dir)
        assert not (work_dir / "protenix_root").exists()
        assert sorted(os.listdir(weights_dir)) == [self.MODEL_FILE]
        assert (weights_dir / self.MODEL_FILE).read_bytes() == b"fine-tuned"
        assert sorted(os.listdir(real_root)) == ["checkpoint", "common", "search_database"]
        assert (real_root / "checkpoint" / self.MODEL_FILE).read_bytes() == b"default weights"
        assert sorted(os.listdir(real_root / "common")) == sorted(DATA_FILES)

    def test_the_linked_root_is_removed_when_the_run_fails(
        self, tmp_path, real_root, weights_dir, stub_protenix
    ):
        stub_protenix.install("exit")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with pytest.raises(ProtenixRunError) as info:
            ProtenixRunner().run(self.request(weights_dir), work_dir)
        assert not (work_dir / "protenix_root").exists()
        message = str(info.value)
        assert "Environment: PROTENIX_ROOT_DIR=<work_dir>/protenix_root" in message
        assert f"as checkpoint, to the weights {weights_dir}" in message

    def test_a_run_without_weights_leaves_the_environment_as_it_is(
        self, tmp_path, real_root, stub_protenix
    ):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        ProtenixRunner().run(make(num_samples=1, seeds=(3,)), work_dir)
        (call,) = stub_protenix.calls()
        assert call["env"]["root"] == str(real_root)
        assert not (work_dir / "protenix_root").exists()

    def test_the_store_runs_a_request_with_weights_and_the_adapter_reads_it(
        self, tmp_path, real_root, weights_dir, stub_protenix
    ):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = self.request(store.weights_reference(weights_dir))
        entry = store.get_or_run(request, runner)
        assert entry.status == "done"
        assert ProtenixParser().load(entry.prediction_dir, "p53").ptm == 0.88
        described = json.loads((store.path_for(request) / "request.json").read_text("utf-8"))
        assert described["weights"]["path"] == str(weights_dir)
        assert len(stub_protenix.calls()) == 1

    def test_a_file_system_without_links_is_a_clear_error(
        self, tmp_path, real_root, weights_dir, stub_protenix, monkeypatch
    ):
        def no_links(source, target, *args, **kwargs):
            raise OSError("symbolic links are not supported")

        monkeypatch.setattr(protenix_runner.os, "symlink", no_links)
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with pytest.raises(ValueError, match="Custom weights need symbolic links") as info:
            ProtenixRunner().run(self.request(weights_dir), work_dir)
        assert "symbolic links are not supported" in str(info.value)
        assert not (work_dir / "protenix_root").exists() and stub_protenix.calls() == []

    def test_a_request_without_weights_has_no_weights_in_its_key(self):
        request = make()
        assert request.weights is None and "weights" not in request.canonical()


# ---------------------------------------------------------------------------- the request


class TestMakeRequest:
    def test_every_default_is_written_out(self):
        request = make()
        assert request.model == "protenix" and request.mode == "predict"
        assert request.seeds == (101,) and request.num_samples == 5
        assert request.binder_chain == "B" and request.receptor_chain == "A"
        assert request.model_version == "2.0.0"
        assert dict(request.options) == {
            "model_name": "protenix_base_default_v1.0.0",
            "dtype": "bf16",
            "use_msa_server": True,
            "msa_server_mode": "protenix",
            "on_unmappable_residue": "error",
            "binder_cyclic": "auto",
            "constraints": None,
            "covalent_bonds": None,
            "extra_args": [],
            "need_atom_confidence": True,
            "checkpoint_size_bytes": None,
        }

    def test_the_same_run_gives_the_same_key(self):
        assert make().key() == make(ProtenixRunner("elsewhere")).key()
        assert make().key() == make(name="another name").key()

    @pytest.mark.parametrize(
        "change",
        [
            {"seeds": [1, 2]},
            {"num_samples": 2},
            {"model_name": "protenix-v2"},
            {"dtype": "fp32"},
            {"use_msa_server": False},
            {"msa_server_mode": "colabfold"},
            {"on_unmappable_residue": "x"},
            {"binder_cyclic": False},
            {"binder_cyclic": True},
            {"extra_args": ["--cycle", "4"]},
            {"binder_chain": "A", "receptor_chain": "B"},
        ],
    )
    def test_a_setting_that_changes_the_output_changes_the_key(self, change):
        assert make(**change).key() != make().key()

    def test_the_content_of_the_structure_is_in_the_key(self, structure, tmp_path):
        other = tmp_path / "other.pdb"
        other.write_text(structure.read_text(encoding="utf-8") + "REMARK x\n", encoding="utf-8")
        assert make(path=structure).key() != make(path=other).key()

    def test_another_protenix_version_is_another_key(self, monkeypatch):
        before = make().key()
        monkeypatch.setattr(protenix_runner, "_installed_version", lambda cmd=None: "2.1.0")
        assert make().key() != before

    def test_an_unknown_version_is_an_empty_string(self, monkeypatch):
        monkeypatch.setattr(protenix_runner, "_installed_version", lambda cmd=None: None)
        assert make().model_version == ""

    def test_the_size_of_the_weights_file_that_protenix_would_load_is_recorded(self, tmp_path):
        checkpoint = tmp_path / "protenix_root" / "checkpoint" / f"{DEFAULT_MODEL_NAME}.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"x" * 123)
        request = make()
        assert request.options["checkpoint_size_bytes"] == 123
        assert request.key() != make(model_name="protenix-v2").key()
        checkpoint.write_bytes(b"x" * 124)
        assert make().key() != request.key()

    @pytest.mark.parametrize(
        "seeds, message",
        [
            ([], "at least one seed"),
            ([-1], "must not be negative"),
            ([1, 1], "must be distinct"),
            ([1.5], "must be integers"),
            ([True], "must be integers"),
            ("12", "not a string"),
        ],
    )
    def test_a_seed_the_adapter_cannot_read_back_is_refused(self, seeds, message):
        with pytest.raises(ValueError, match=message):
            make(seeds=seeds)

    def test_seeds_given_as_numpy_integers_are_accepted(self):
        np = pytest.importorskip("numpy")
        assert make(seeds=np.array([3, 4])).seeds == (3, 4)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"dtype": "fp8"}, "dtype must be one of"),
            ({"msa_server_mode": "local"}, "msa_server_mode must be one of"),
            ({"on_unmappable_residue": "skip"}, "on_unmappable_residue must be one of"),
            ({"model_name": ""}, "model_name must be"),
            ({"model_name": "a/b"}, "model_name must be"),
            ({"binder_chain": None}, "needs binder_chain and receptor_chain"),
            ({"receptor_chain": None}, "needs binder_chain and receptor_chain"),
            ({"binder_chain": "A"}, "are both 'A'"),
            ({"name": "a/b"}, "path separator"),
            ({"name": ".."}, "path separator"),
            ({"covalent_bonds": ["C-N"]}, "list of dicts"),
            ({"extra_args": "--cycle 4"}, "not a string"),
        ],
    )
    def test_a_choice_that_is_not_offered_is_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make(**kwargs)

    @pytest.mark.parametrize(
        "argument",
        ["--seeds", "--seeds=5", "-s5", "--sample", "-e", "--dtype=fp32", "--model_name", "-n",
         "--use_msa", "--msa_server_mode", "--need_atom_confidence", "--out_dir", "-o",
         "--input", "--use_template", "--use_rna_msa", "--use_seeds_in_json"],
    )  # fmt: skip
    def test_extra_args_cannot_set_what_the_runner_sets(self, argument):
        with pytest.raises(ValueError, match="extra_args cannot hold"):
            make(extra_args=["--trimul_kernel", "torch", argument, "1"])

    def test_extra_args_that_the_runner_does_not_set_pass(self):
        request = make(extra_args=("--cycle", "4", "--step=5"))
        assert request.options["extra_args"] == ["--cycle", "4", "--step=5"]

    def test_the_name_must_not_leave_the_output_folder_when_the_request_is_made_elsewhere(
        self, tmp_path
    ):
        request = PredictionRequest(
            "protenix", "../x", mode="predict", input_path=P53, binder_chain="B", receptor_chain="A"
        )
        with pytest.raises(ValueError, match="path separator"):
            ProtenixRunner().prepare(request, tmp_path)

    def test_a_request_for_another_model_is_refused(self, tmp_path):
        request = PredictionRequest(
            "of3", "p53", mode="predict", input_path=P53, binder_chain="B", receptor_chain="A"
        )
        with pytest.raises(ValueError, match="cannot run a 'of3' request"):
            ProtenixRunner().prepare(request, tmp_path)


# ---------------------------------------------------------------------------- the command line


class TestCommandLine:
    JOB = Path("/w/input/p53.json")
    OUT = Path("/w/predictions")

    def command(self, runner=None, **kwargs):
        runner = runner or ProtenixRunner()
        return runner._command(make(runner, **kwargs), self.JOB, self.OUT)

    def test_the_defaults(self):
        command = self.command()
        assert command[0].endswith("protenix")
        assert command[1:] == [
            "pred",
            "--input", "/w/input/p53.json",
            "--out_dir", "/w/predictions",
            "--seeds", "101",
            "--sample", "5",
            "--dtype", "bf16",
            "--model_name", "protenix_base_default_v1.0.0",
            "--use_msa", "true",
            "--msa_server_mode", "protenix",
            "--need_atom_confidence", "true",
        ]  # fmt: skip

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"seeds": [7, 3, 12]}, ["--seeds", "7,3,12"]),
            ({"num_samples": 2}, ["--sample", "2"]),
            ({"dtype": "fp32"}, ["--dtype", "fp32"]),
            ({"model_name": "protenix-v2"}, ["--model_name", "protenix-v2"]),
            ({"use_msa_server": False}, ["--use_msa", "false"]),
            ({"msa_server_mode": "colabfold"}, ["--msa_server_mode", "colabfold"]),
        ],
    )
    def test_each_option_is_one_flag(self, kwargs, expected):
        command = self.command(**kwargs)
        flag = expected[0]
        assert command[command.index(flag) : command.index(flag) + 2] == expected
        assert command.count(flag) == 1

    def test_the_per_atom_confidence_is_always_asked_for(self):
        for kwargs in ({}, {"use_msa_server": False}, {"extra_args": ["--cycle", "4"]}):
            command = self.command(**kwargs)
            assert command[command.index("--need_atom_confidence") + 1] == "true"

    def test_extra_arguments_come_last_and_unchanged(self):
        command = self.command(extra_args=["--cycle", "4", "--step=5"])
        assert command[-3:] == ["--cycle", "4", "--step=5"]

    def test_a_conda_environment_runs_the_command_behind_conda_run(self):
        command = self.command(ProtenixRunner("px"))
        assert command[:7] == [
            "conda",
            "run",
            "-n",
            "px",
            "--no-capture-output",
            "protenix",
            "pred",
        ]
        assert command[7:9] == ["--input", "/w/input/p53.json"]

    def test_the_command_is_the_one_of_the_request_even_when_it_has_no_options(self):
        request = PredictionRequest(
            "protenix", "p53", mode="predict", input_path=P53, binder_chain="B", receptor_chain="A"
        )
        command = ProtenixRunner()._command(request, self.JOB, self.OUT)
        assert command[command.index("--seeds") + 1] == "42"  # the request's own default
        assert command[command.index("--model_name") + 1] == DEFAULT_MODEL_NAME


# ---------------------------------------------------------------------------- the machine


class TestAvailabilityAndVersion:
    def test_protenix_on_path(self, monkeypatch):
        real = shutil.which
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, *a, **k: "/fake/protenix" if name == "protenix" else real(name, *a, **k),
        )
        assert ProtenixRunner().is_available() is True
        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
        assert ProtenixRunner().is_available() is False

    def test_a_conda_environment_needs_the_package_in_it(self, monkeypatch):
        assert ProtenixRunner("px").is_available() is True
        monkeypatch.setattr(protenix_runner, "_installed_version", lambda cmd=None: None)
        assert ProtenixRunner("px").is_available() is False

    def test_the_version_is_asked_once_and_through_conda_when_an_environment_is_set(
        self, protenix_environment
    ):
        runner = ProtenixRunner("px")
        assert runner.version() == "2.0.0" and runner.version() == "2.0.0"
        assert protenix_environment == [["conda", "run", "-n", "px", "python"]]

    def test_the_current_interpreter_is_asked_without_an_environment(self, protenix_environment):
        assert ProtenixRunner().version() == "2.0.0"
        assert protenix_environment == [None]

    def test_the_current_interpreter_reads_the_package_metadata(self, monkeypatch):
        import importlib.metadata

        monkeypatch.undo()
        monkeypatch.setattr(importlib.metadata, "version", lambda name: f"{name}-9")
        assert protenix_runner._installed_version() == "protenix-9"

        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", missing)
        assert protenix_runner._installed_version() is None

    def test_another_interpreter_is_asked_through_its_command(self, monkeypatch):
        monkeypatch.undo()
        calls = []

        class Done:
            returncode = 0
            stdout = "some conda banner\n2.0.0\n"

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Done()

        monkeypatch.setattr(protenix_runner.subprocess, "run", fake_run)
        assert protenix_runner._installed_version(["conda", "run", "-n", "px", "python"]) == "2.0.0"
        assert calls[0][:5] == ["conda", "run", "-n", "px", "python"] and calls[0][5] == "-c"
        assert "version('protenix')" in calls[0][6]

    @pytest.mark.parametrize("failure", ["exit", "missing-binary", "timeout"])
    def test_a_failing_probe_gives_none(self, monkeypatch, failure):
        monkeypatch.undo()

        class Failed:
            returncode = 1
            stdout = ""

        def fake_run(cmd, **kwargs):
            if failure == "missing-binary":
                raise FileNotFoundError("conda")
            if failure == "timeout":
                raise protenix_runner.subprocess.TimeoutExpired(cmd, 60)
            return Failed()

        monkeypatch.setattr(protenix_runner.subprocess, "run", fake_run)
        assert protenix_runner._installed_version(["conda", "run", "-n", "px", "python"]) is None

    def test_the_runner_is_a_runner_without_capabilities_and_without_batches(self):
        runner = ProtenixRunner()
        assert isinstance(runner, PredictionRunner)
        assert ProtenixRunner.name == ProtenixParser.name == "protenix"
        assert ProtenixRunner.capabilities is None
        assert runner.supports_batch(make()) is False
        with pytest.raises(NotImplementedError, match="protenix runner has no batched mode"):
            runner.run_many([], Path("."))


# ---------------------------------------------------------------------------- with a stub process


class TestRunThroughTheStoreAndTheSession:
    def test_a_run_is_stored_once_and_the_adapter_reads_it(self, tmp_path, stub_protenix):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(runner, seeds=(3, 4), num_samples=2)
        entry = store.get_or_run(request, runner)
        store.get_or_run(request, runner)  # a second metric: no second process
        (call,) = stub_protenix.calls()
        argv = call["argv"]
        assert argv[0] == "pred"
        assert argv[argv.index("--input") + 1].endswith("/input/p53.json")
        assert argv[argv.index("--out_dir") + 1].endswith("/predictions")
        assert argv[argv.index("--seeds") + 1] == "3,4"
        assert argv[argv.index("--sample") + 1] == "2"
        assert argv[argv.index("--need_atom_confidence") + 1] == "true"
        assert call["cwd"].endswith(".tmp-" + call["cwd"].rsplit(".tmp-", 1)[1])
        assert [e["proteinChain"]["sequence"] for e in call["job"][0]["sequences"]] == [
            MDM2,
            P53_PEPTIDE,
        ]
        assert entry.status == "done" and entry.prediction_dir.name == "predictions"

        parser = ProtenixParser()
        first = parser.load(entry.prediction_dir, "p53", seed_index=1, sample=1)
        assert (first.ptm, first.iptm, first.avg_plddt) == pytest.approx(
            (0.88, 0.76, 82.0), abs=0.01
        )
        assert first.extras["seed_value"] == "3" and first.pae is not None
        assert first.plddt_per_atom is not None and first.pde is not None
        second = parser.load(entry.prediction_dir, "p53", seed_index=2, sample=2)
        assert second.extras["seed_value"] == "4"
        assert second.avg_plddt == pytest.approx(first.avg_plddt - 2.0, abs=0.01)

    def test_two_metrics_and_a_second_session_start_one_process(self, tmp_path, stub_protenix):
        from binding_metrics.predictors.session import PredictionSession

        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(runner, seeds=(3,), num_samples=2)
        session = PredictionSession(store, [runner])
        record = session.record(request)  # metric one
        assert session.record(request, sample=2).ptm == 0.88  # metric two
        again = PredictionSession(store, [runner])  # a restarted pipeline
        assert again.record(request).iptm == 0.76
        assert len(stub_protenix.calls()) == 1
        assert session.stats()["runs"] == 1 and again.stats()["hits"] == 1
        # the adapter completes the record: chains are named by their IDs, tokens are laid out
        assert set(record.chain_ptm) == {"A", "B"}
        assert record.tokens is not None and record.pae.shape == (7, 7)

    def test_the_conda_route_runs_the_same_command(self, tmp_path, monkeypatch, stub_protenix):
        record = tmp_path / "conda_calls.jsonl"
        script = tmp_path / "bin" / "conda"
        script.write_text(
            CONDA_SCRIPT.format(python=sys.executable, record=str(record)), encoding="utf-8"
        )
        script.chmod(0o755)
        runner = ProtenixRunner("px")
        request = make(runner, seeds=(5,), num_samples=1)
        entry = PredictionStore(tmp_path / "store").get_or_run(request, runner)
        assert entry.status == "done"
        (conda_call,) = [json.loads(line) for line in record.read_text("utf-8").splitlines()]
        assert conda_call[:4] == ["run", "-n", "px", "--no-capture-output"]
        assert conda_call[4:6] == ["protenix", "pred"]
        assert len(stub_protenix.calls()) == 1

    def test_protenix_that_is_not_installed_records_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(runner)
        assert runner.is_available() is False
        with pytest.raises(PredictionUnavailableError, match="cannot be started on this machine"):
            store.get_or_run(request, runner)
        assert not store.path_for(request).exists()

    def test_a_missing_executable_is_a_clear_error_when_run_directly(self, tmp_path, monkeypatch):
        pytest.importorskip("gemmi")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        runner = ProtenixRunner()
        with pytest.raises(FileNotFoundError, match="pip install protenix"):
            runner.run(make(runner), tmp_path / "work")

    def test_prepare_writes_the_input_and_starts_nothing(self, tmp_path, stub_protenix):
        runner = ProtenixRunner()
        path = runner.prepare(make(runner), tmp_path / "work")
        assert path.is_file() and stub_protenix.calls() == []


class TestFailures:
    def run(self, tmp_path, **kwargs):
        runner = ProtenixRunner()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        return runner.run(make(runner, seeds=(3, 4), num_samples=2, **kwargs), work_dir)

    def test_a_process_that_exits_non_zero_keeps_status_reason_and_hint(
        self, tmp_path, stub_protenix
    ):
        stub_protenix.install("exit")
        with pytest.raises(ProtenixRunError) as info:
            self.run(tmp_path)
        error = info.value
        assert error.returncode == 1 and error.cmd[0].endswith("protenix")
        message = str(error)
        assert message.startswith("Protenix exited with status 1: torch.OutOfMemoryError")
        assert "Hint: The GPU ran out of memory" in message
        assert "Last lines of output:" in message and "Command: " in message
        assert "CUDA out of memory" in error.output_tail

    def test_a_failed_run_through_the_store_is_recorded_with_the_reason_and_not_retried(
        self, tmp_path, stub_protenix
    ):
        stub_protenix.install("exit")
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(runner, num_samples=2)
        with pytest.raises(PredictionFailedError, match="exited with status 1") as info:
            store.get_or_run(request, runner)
        assert isinstance(info.value.__cause__, ProtenixRunError)
        with pytest.raises(PredictionFailedError):
            store.get_or_run(request, runner)
        assert len(stub_protenix.calls()) == 1
        stub_protenix.install("ok")
        assert store.get_or_run(request, runner, rerun=True).status == "done"
        assert len(stub_protenix.calls()) == 2

    def test_exit_status_zero_with_an_err_file_and_no_output_is_a_failure(
        self, tmp_path, stub_protenix
    ):
        stub_protenix.install("err_file")
        with pytest.raises(ProtenixRunError) as info:
            self.run(tmp_path)
        message = str(info.value)
        assert info.value.returncode is None
        assert "exited normally but wrote no complete output for 'p53'" in message
        assert "no output for seed 3; no output for seed 4" in message
        assert (
            "Recorded by Protenix: ERR/p53.txt: [Rank 0] p53 failed: boom ... ValueError: boom"
            in message
        )

    @pytest.mark.parametrize("behavior", ["exit", "err_file", "msa_failure", "partial"])
    def test_a_recorded_reason_does_not_name_the_work_directory_the_store_renames(
        self, tmp_path, stub_protenix, behavior
    ):
        stub_protenix.install(behavior)
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(runner, seeds=(3, 4), num_samples=2)
        with pytest.raises(PredictionFailedError) as info:
            store.get_or_run(request, runner)
        reason = info.value.reason
        assert ".tmp-" not in reason and str(tmp_path / "store") not in reason
        assert "<work_dir>/input/p53.json" in reason and "<work_dir>/predictions" in reason

    def test_the_error_file_of_the_dataloader_is_read_too(self, tmp_path, stub_protenix):
        stub_protenix.install("error_txt")
        with pytest.raises(ProtenixRunError, match=r"Recorded by Protenix: ERR/error\.txt: "):
            self.run(tmp_path)

    def test_an_error_that_protenix_only_logged_is_in_the_message(self, tmp_path, stub_protenix):
        stub_protenix.install("run_failed")
        with pytest.raises(ProtenixRunError) as info:
            self.run(tmp_path)
        assert "Protenix log: " in str(info.value)
        assert "Run inference failed: {'in.json': 'MSA boom'}" in str(info.value)

    def test_a_failed_msa_search_fails_the_run_even_though_output_exists(
        self, tmp_path, stub_protenix
    ):
        stub_protenix.install("msa_failure")
        with pytest.raises(ProtenixRunError) as info:
            self.run(tmp_path)
        message = str(info.value)
        assert message.startswith("The MSA search failed and Protenix continued with the sequence")
        assert "use_msa_server=False" in message and "MMSEQS2 failed" in message
        assert "ConnectionError: server unreachable" in message  # in the last lines

    def test_a_seed_that_is_missing_would_shift_the_others_and_fails_the_run(
        self, tmp_path, stub_protenix
    ):
        stub_protenix.install("partial")
        with pytest.raises(ProtenixRunError) as info:
            self.run(tmp_path)
        message = str(info.value)
        assert "no output for seed 4" in message and "seed 3" not in message.split(":")[1]
        assert "ERR/p53.txt: [Rank 0] failed: seed two" in message

    def test_a_sample_that_is_missing_is_named(self, tmp_path, stub_protenix):
        stub_protenix.install("ok", samples=1)
        with pytest.raises(ProtenixRunError) as info:
            self.run(tmp_path)
        assert "seed 3 sample 1 lacks its structure, summary confidence, full-data file" in str(
            info.value
        )

    def test_a_missing_full_data_file_is_named(self, tmp_path, stub_protenix):
        stub_protenix.install("no_full_data")
        with pytest.raises(ProtenixRunError, match="seed 3 sample 0 lacks its full-data file"):
            self.run(tmp_path)

    def test_a_complete_run_returns_the_directory_the_adapter_loads(self, tmp_path, stub_protenix):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        runner = ProtenixRunner()
        directory = runner.run(make(runner, seeds=(3, 4), num_samples=2), work_dir)
        assert directory == work_dir.resolve() / "predictions"
        assert ProtenixParser().find_files(directory, "p53", seed_index=2, sample=2).has_output()

    def test_the_hint_for_the_kernels_names_the_options_that_exist(self):
        message = protenix_runner._failure_message(
            "failed",
            recorded=[],
            output_tail="ModuleNotFoundError: No module named 'cuequivariance_ops_torch'",
            cmd=["protenix"],
        )
        assert "--trimul_kernel', 'torch', '--triatt_kernel', 'torch'" in message

    def test_the_hint_for_the_weights_names_where_they_are_loaded_from(self):
        message = protenix_runner._failure_message(
            "failed",
            recorded=["ERR/p53.txt: FileNotFoundError: Given checkpoint path not exist [/x.pt]"],
            output_tail="",
            cmd=["protenix"],
        )
        assert "<PROTENIX_ROOT_DIR or ~>/checkpoint/<model_name>.pt" in message


# ---------------------------------------------------------------------------- module facts


def test_importing_the_module_and_making_a_request_imports_only_the_standard_library():
    code = textwrap.dedent(
        f"""
        import sys, tempfile, pathlib
        import binding_metrics.predictors.protenix_runner as m
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "gemmi", "protenix")
        print(sorted(x for x in sys.modules if x.split(".")[0] in heavy))
        m.ProtenixRunner().make_request(
            {str(P53)!r}, name="p", binder_chain="B", receptor_chain="A")
        print(sorted(x for x in sys.modules if x.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:2] == ["[]", "[]"]


# ---------------------------------------------------------------------------- the source it cites

MODEL_SOURCES_ENVIRONMENT_VARIABLE = "BINDING_METRICS_MODEL_SOURCES"


def protenix_source(relative_path: str) -> str:
    """A file of the Protenix clone, or skip the test when the clone is not at hand."""
    root = os.environ.get(MODEL_SOURCES_ENVIRONMENT_VARIABLE)
    if not root:
        pytest.skip(f"{MODEL_SOURCES_ENVIRONMENT_VARIABLE} is not set")
    path = Path(root) / "Protenix" / relative_path
    if not path.is_file():
        pytest.skip(f"{path} is not there")
    return path.read_text(encoding="utf-8")


class TestTheSourceBehindTheRunner:
    """Re-read the Protenix commit 85767b8 that the module docstring cites."""

    def test_the_command_and_its_flags_exist_with_the_defaults_the_runner_writes_out(self):
        text = protenix_source("runner/batch_inference.py")
        assert 'protenix_cli.add_command(predict, name="pred")' in text
        for declaration in (
            '"-i", "--input"',
            '"-o", "--out_dir"',
            '"-s", "--seeds", type=str, default="101"',
            '"-e", "--sample", type=int, default=5',
            '"-d", "--dtype", type=str, default="bf16"',
            '"--model_name"',
            'default="protenix_base_default_v1.0.0"',
            '"--use_msa"',
            '"--msa_server_mode"',
            'default="protenix"',
            '"--need_atom_confidence"',
            '"--use_template"',
            '"--use_rna_msa"',
            '"--use_seeds_in_json"',
            '"--trimul_kernel"',
            '"--triatt_kernel"',
        ):
            assert declaration in text, declaration
        assert "msa_server_mode (str): MSA server mode." in text

    def test_the_option_lines_the_docstring_cites(self):
        lines = protenix_source("runner/batch_inference.py").splitlines()
        assert '"--seeds"' in lines[603] and '"--sample"' in lines[606]
        assert '"--dtype"' in lines[607] and '"--use_msa"' in lines[616]
        assert '"--use_template"' in lines[667] and '"--need_atom_confidence"' in lines[685]
        assert "def predict(" in lines[762]
        assert "protenix_cli.add_command(predict" in lines[1351]

    def test_the_dtypes_and_the_command_line_seeds(self):
        assert re.search(
            r'"fp32": torch\.float32,\s*"bf16": torch\.bfloat16,\s*"fp16": torch\.float16',
            protenix_source("runner/inference.py"),
        )
        assert 'seeds = list(map(int, seeds.split(",")))' in protenix_source(
            "runner/batch_inference.py"
        )

    def test_a_failed_sample_exits_zero_and_the_errors_are_written_where_the_runner_reads(self):
        inference = protenix_source("runner/inference.py")
        assert 'self.error_dir = opjoin(self.dump_dir, "ERR")' in inference
        assert 'opjoin(runner.error_dir, f"{sample_name}.txt")' in inference
        assert 'opjoin(runner.error_dir, "error.txt")' in inference
        batch = protenix_source("runner/batch_inference.py")
        assert "infer_errors[infer_json] = str(exc)" in batch
        assert 'logger.warning(f"Run inference failed: {infer_errors}")' in batch

    def test_the_input_format_the_runner_writes(self):
        lines = protenix_source("docs/infer_json_format.md").splitlines()
        assert "`templatesPath`" in lines[67]
        assert "`sequence`" in lines[59] and "X (UNK)" in lines[59]
        assert "`ptmType`" in lines[63] and "`ptmPosition`" in lines[64]
        assert "`id` (optional)" in lines[61] and "`count`" in lines[61]
        assert "`constraint` section" in lines[252] and "`contact` and `pocket`" in lines[252]
        assert "a **soft constraint**" in lines[255]
        assert "the model is encouraged, but not strictly required, to satisfy it" in lines[255]
        assert "head-to-tail amide bond" in " ".join(lines[221:229])
        assert "A head-to-tail amide bond connecting the N- and C-terminal residues" in lines[225]
        assert "A disulfide bond between cysteine residues" in lines[226]
        assert "not reliably handled" in lines[228]
        assert "`entity1`, `entity2`" in lines[232] and "starting from 1" in lines[233]
        assert "`copy1`, `copy2`" in lines[234] and "starting from 1" in lines[234]
        assert (
            "starts at 1" in lines[236] and "location of the residue in the sequence" in lines[237]
        )
        assert "consistent with those defined in the CCD" in lines[242]
        assert "order in which the entity appears in the `sequences` list" in lines[233]

    def test_the_polymer_builder_replaces_the_residue_by_the_ccd_component(self):
        text = protenix_source("protenix/data/inference/json_parser.py")
        assert 'index = m["ptmPosition"] - 1' in text and 'mtype = m["ptmType"]' in text
        assert "ccd_seqs[index] = mtype[4:]" in text
        letters = re.search(r"PROTEIN_1to3 = \{(.*?)\}", text, re.DOTALL).group(1)
        assert set(re.findall(r'"(\w)":', letters)) == set(protenix_runner._PROTEIN_LETTERS)

    def test_the_chain_ids_are_the_id_list(self):
        text = protenix_source("protenix/data/inference/json_to_feature.py")
        assert 'len(ids) != entity["count"]' in text
        assert "duplicated chain IDs across entities" in text
        assert 'self.input_dict.get("constraint", {})' in text

    def test_the_weights_directory_is_not_an_option_of_the_command(self):
        text = protenix_source("runner/batch_inference.py")
        assert "load_checkpoint_dir" not in text
        lines = text.splitlines()
        assert lines[564].startswith("CONTEXT_SETTINGS = dict(help_option_names=")
        assert "ignore_unknown_options" not in text and "allow_extra_args" not in text
        assert "parse_configs(" in lines[333] and "fill_required_with_null=True" in lines[335]
        assert "arg_str" not in "\n".join(lines[333:337])
        assert "preprocess_input(" in lines[539]
        configs = protenix_source("configs/configs_inference.py")
        assert 'os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))' in configs
        assert '"load_checkpoint_dir": os.path.join(PROTENIX_ROOT_DIR, "checkpoint")' in configs
        inference = protenix_source("runner/inference.py")
        assert 'f"{self.configs.model_name}.pt"' in inference

    def test_the_option_exists_only_in_the_other_entry_point_that_does_not_search_the_msa(self):
        inference = protenix_source("runner/inference.py")
        assert "def run() -> None:" in inference and "arg_str = parse_sys_args()" in inference
        assert "preprocess_input" not in inference
        assert "def parse_sys_args() -> str:" in protenix_source("protenix/config/config.py")
        parser = protenix_source("protenix/web_service/colab_request_parser.py")
        assert 'f"--load_checkpoint_dir {checkpoint_dir}"' in parser
        batch = protenix_source("runner/batch_inference.py").splitlines()
        assert "configs.model.N_cycle = n_cycle" in batch[361]
        assert "configs.sample_diffusion.N_step = n_step" in batch[363]

    def test_the_data_files_and_the_downloads_the_linked_root_must_avoid(self):
        data = protenix_source("configs/configs_data.py")
        assert 'os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))' in data
        for name in protenix_runner._DATA_FILES:
            assert f'"common/{name}"' in data, name
        inference = protenix_source("runner/inference.py")
        for name in ("ccd_components_file", "ccd_components_rdkit_mol_file", "pdb_cluster_file"):
            assert f'"{name}"' in inference
        assert "obsolete_release_data_csv" in inference
        assert "if not opexists(checkpoint_path):" in inference
        assert "download_from_url(tos_url, checkpoint_path)" in inference
        for name in ("esm2_t36_3B_UR50D.pt", "esm2_t36_3B_UR50D-contact-regression.pt"):
            assert f"{{checkpoint_dir}}/{name}" in inference
        for name in ("esm2_t36_3B_UR50D_ism.pt", "esm2_t36_3B_UR50D_ism-contact-regression.pt"):
            assert f"{{checkpoint_dir}}/{name}" in inference
        assert (
            '"esm" in configs.model_name' in inference
            and '"ism" in configs.model_name' in inference
        )

    def test_only_the_constraint_model_has_its_embedders_on(self):
        types = protenix_source("configs/configs_model_type.py")
        block = types.split('"protenix_base_constraint_v0.5.0": {', 1)[1].split(
            '"protenix_mini_default_v0.5.0"', 1
        )[0]
        assert block.count('"enable": True') >= 4
        assert types.count('"constraint_embedder"') == 1
        base = protenix_source("configs/configs_base.py")
        embedder = base.split('"constraint_embedder": {', 1)[1].split('"initialize_method"', 1)[0]
        assert embedder.count('"enable": False') == 4 and "True" not in embedder
        assert "if z_constraint is not None:" in protenix_source("protenix/model/protenix.py")
        assert CONSTRAINT_MODEL_NAME in types

    def test_the_msa_search_failure_is_printed_and_the_run_goes_on(self):
        text = protenix_source("protenix/web_service/colab_request_parser.py")
        assert "MMSEQS2 failed with the following error message" in text
        assert "using the sequence itself as MSA." in text
        for marker in protenix_runner._MSA_FAILURE_MARKERS:
            assert marker in text
        utils = protenix_source("protenix/web_service/colab_request_utils.py")
        assert 'assert host_url == "https://protenix-server.com/api/msa"' in utils

    def test_the_full_data_file_is_written_only_with_the_confidence_flag(self):
        text = protenix_source("runner/dumper.py")
        assert "if self.need_atom_confidence:" in text and "_full_data_sample_" in text

    def test_the_version_the_docstring_names(self):
        assert '__version__ = "2.0.0"' in protenix_source("protenix/version.py")
        assert 'name="protenix"' in protenix_source("setup.py")
