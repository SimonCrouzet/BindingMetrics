"""The Boltz-2 runner: the YAML and the template CIFs it writes, the command it runs, the failures.

No Boltz-2, no GPU and no network are needed. The "boltz" that the run tests start is a small
Python process behind a shell wrapper on PATH: it records its command line and the YAML it was
given and writes a synthetic Boltz-2 output (``tests/predictors/synth_boltz2.py``, the layout of
Boltz v2.2.1's writers) or fails the way the source says Boltz-2 fails. What the stub cannot
show is stated in the docstring of ``boltz2_runner.py``: the YAML and the templates were parsed
by Boltz-2's own reader outside the test suite, and no prediction of the model is tested here.
"""

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from binding_metrics.predictors import boltz2_runner  # noqa: E402
from binding_metrics.predictors.boltz2 import Boltz2Parser  # noqa: E402
from binding_metrics.predictors.boltz2_runner import (  # noqa: E402
    DEFAULT_LOCK_THRESHOLD_ANGSTROM,
    RUN_MODES,
    Boltz2RunError,
    Boltz2Runner,
)
from binding_metrics.predictors.runners import PredictionRunner  # noqa: E402
from binding_metrics.predictors.session import PredictionSession  # noqa: E402
from binding_metrics.predictors.store import (  # noqa: E402
    MODES,
    PredictionFailedError,
    PredictionStore,
)
from tests.predictors import synth, synth_boltz2  # noqa: E402
from tests.test_of3_synth import _write  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stubs are POSIX shell scripts")

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"

YCR_A = "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
YCR_B = "ETFSDLWKLLPEN"
CWA_A = (
    "MVNPTVFFDIAVDGEPLGRVSFELFADKVPKTAENFRALSTGEKGFGYKGSCFHRIIPGFMCQGGDFTRHNGTGGKSIYGEKFEDEN"
    "FILKHTGPGILSMANAGPNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE"
)
CWA_C = "ALLVTAGLVLA"
CWA_C_MODIFICATIONS = [
    (1, "DAL"), (2, "MLE"), (3, "MLE"), (4, "MVA"), (5, "BMT"), (6, "ABA"), (7, "SAR"),
    (8, "MLE"), (10, "MLE"),
]  # fmt: skip

_REAL_VERSION_PROBE = boltz2_runner._installed_boltz_version


@pytest.fixture(autouse=True)
def boltz_version(monkeypatch):
    """Boltz-2 is not installed here: the version is 2.2.1 whatever the machine has."""
    asked = []

    def fake_version(conda_env):
        asked.append(conda_env)
        return "2.2.1"

    monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", fake_version)
    return asked


def make_request(runner=None, path=YCR, **overrides):
    """A ``score`` request for 1YCR (binder B, receptor A) unless said otherwise."""
    keywords = {"name": "ycr", "binder_chain": "B", "receptor_chain": "A"}
    keywords.update(overrides)
    return (runner or Boltz2Runner()).make_request(path, **keywords)


def quoted(value):
    return json.dumps(str(value))


def read_yaml(path):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def prepared(tmp_path, request, runner=None):
    """``(work_dir, yaml_path)`` after ``prepare``."""
    work = tmp_path / "work"
    work.mkdir(parents=True)
    return work, (runner or Boltz2Runner()).prepare(request, work)


# ---------------------------------------------------------------------------- the request


class TestTheRequest:
    def test_every_setting_that_changes_the_output_is_written_out(self):
        request = make_request()
        assert request.model == "boltz2"
        assert request.mode == "score"
        assert (request.binder_chain, request.receptor_chain) == ("B", "A")
        assert request.seeds == (42,)
        assert request.num_samples == 5
        assert request.model_version == "2.2.1"
        assert request.weights is None
        assert dict(request.options) == {
            "use_msa_server": True,
            "binder_cyclic": "auto",
            "on_unmappable_residue": "error",
            "lock_threshold_angstrom": None,
            "extra_args": [],
        }

    def test_the_same_run_has_the_same_key_whatever_its_name_or_environment(self):
        key = make_request().key()
        assert make_request(name="another").key() == key
        assert make_request(Boltz2Runner("some_env")).key() == key
        assert make_request(seeds=[42], num_samples=5, use_msa_server=True).key() == key

    @pytest.mark.parametrize(
        "change",
        [
            {"mode": "refold"},
            {"mode": "predict"},
            {"mode": "score-lock"},
            {"seeds": [7]},
            {"num_samples": 1},
            {"use_msa_server": False},
            {"binder_cyclic": True},
            {"on_unmappable_residue": "x"},
            {"extra_args": ["--no_kernels"]},
            {"binder_chain": "A", "receptor_chain": "B"},
        ],
    )
    def test_a_setting_that_changes_the_output_changes_the_key(self, change):
        assert make_request(**change).key() != make_request().key()

    def test_the_lock_threshold_is_part_of_the_key_in_score_lock_only(self):
        default = make_request(mode="score-lock")
        assert default.options["lock_threshold_angstrom"] == DEFAULT_LOCK_THRESHOLD_ANGSTROM == 2.0
        assert make_request(mode="score-lock", lock_threshold_angstrom=3).key() != default.key()
        assert make_request(mode="score-lock", lock_threshold_angstrom=2).key() == default.key()

    def test_the_input_file_is_hashed_by_content(self, tmp_path):
        copy = tmp_path / "elsewhere.pdb"
        copy.write_bytes(YCR.read_bytes())
        assert make_request(path=copy).key() == make_request().key()
        copy.write_bytes(YCR.read_bytes() + b"REMARK changed\n")
        assert make_request(path=copy).key() != make_request().key()

    def test_the_modes_are_those_of_the_store_and_the_runner_names_them(self):
        assert set(RUN_MODES) == set(MODES)
        assert Boltz2Runner.run_modes == RUN_MODES

    @pytest.mark.parametrize(
        "keywords, message",
        [
            ({"mode": "dock"}, "mode must be one of"),
            ({"binder_chain": None}, "needs binder_chain and receptor_chain"),
            ({"receptor_chain": ""}, "needs binder_chain and receptor_chain"),
            ({"binder_chain": "A"}, "must be different chains"),
            ({"name": "a/b"}, "plain file name"),
            ({"name": ".."}, "plain file name"),
            ({"name": ""}, "non-empty"),
            ({"seeds": [1, 2]}, "exactly one seed"),
            ({"seeds": []}, "exactly one seed"),
            ({"on_unmappable_residue": "skip"}, "on_unmappable_residue must be one of"),
            ({"binder_cyclic": "yes"}, "binder_cyclic must be True, False or 'auto'"),
            ({"lock_threshold_angstrom": 2.0}, "only applies to mode 'score-lock'"),
            ({"mode": "score-lock", "lock_threshold_angstrom": 0}, "positive number"),
            ({"mode": "score-lock", "lock_threshold_angstrom": -1.0}, "positive number"),
            ({"mode": "score-lock", "lock_threshold_angstrom": math.nan}, "positive number"),
            ({"mode": "score-lock", "lock_threshold_angstrom": math.inf}, "positive number"),
            ({"extra_args": ["--seed", "3"]}, "repeats --seed"),
            ({"extra_args": ["--diffusion_samples=9"]}, "repeats --diffusion_samples"),
            ({"extra_args": ["--use_msa_server"]}, "repeats --use_msa_server"),
            ({"extra_args": ["--checkpoint", "x.ckpt"]}, "repeats --checkpoint"),
        ],
    )
    def test_a_request_that_cannot_be_run_is_refused(self, keywords, message):
        with pytest.raises(ValueError, match=message):
            make_request(**keywords)

    def test_seeds_must_not_be_a_string(self):
        with pytest.raises(TypeError, match="not a string"):
            make_request(seeds="42")

    def test_extra_args_must_not_be_a_string(self):
        with pytest.raises(TypeError, match="not a string"):
            make_request(extra_args="--no_kernels")

    def test_a_flag_that_the_runner_does_not_set_is_passed_on(self):
        request = make_request(extra_args=["--recycling_steps", "5", "--no_kernels"])
        assert request.options["extra_args"] == ["--recycling_steps", "5", "--no_kernels"]

    def test_an_unreadable_input_file_raises_os_error(self, tmp_path):
        with pytest.raises(OSError):
            make_request(path=tmp_path / "absent.pdb")

    def test_the_runner_declares_no_capabilities_of_its_own(self):
        assert Boltz2Runner.capabilities is None
        assert Boltz2Runner.name == Boltz2Parser.name == "boltz2"
        assert issubclass(Boltz2Runner, PredictionRunner)
        assert Boltz2Runner().supports_batch(make_request()) is False


# ---------------------------------------------------------------------------- the YAML


def ycr_head(msa=False):
    empty = '      msa: "empty"\n' if msa else ""
    return (
        "version: 1\nsequences:\n"
        f'  - protein:\n      id: "A"\n      sequence: "{YCR_A}"\n{empty}'
        f'  - protein:\n      id: "B"\n      sequence: "{YCR_B}"\n{empty}'
    )


class TestTheYamlOfEachMode:
    def test_predict_has_the_sequences_and_no_template(self, tmp_path):
        work, yaml_path = prepared(tmp_path, make_request(mode="predict"))
        assert yaml_path == work / "input" / "ycr.yaml"
        assert yaml_path.read_text(encoding="utf-8") == ycr_head()
        assert not (work / "input" / "templates").exists()

    def test_refold_templates_the_receptor_only(self, tmp_path):
        work, yaml_path = prepared(tmp_path, make_request(mode="refold"))
        templates = work / "input" / "templates"
        assert yaml_path.read_text(encoding="utf-8") == ycr_head() + (
            f"templates:\n  - cif: {quoted(templates / 'receptor.cif')}\n"
            '    chain_id: ["A"]\n    template_id: ["A"]\n'
        )
        assert [p.name for p in templates.iterdir()] == ["receptor.cif"]

    def test_score_templates_each_chain_from_its_own_file_without_force(self, tmp_path):
        work, yaml_path = prepared(tmp_path, make_request(mode="score"))
        templates = work / "input" / "templates"
        assert yaml_path.read_text(encoding="utf-8") == ycr_head() + (
            f"templates:\n  - cif: {quoted(templates / 'receptor.cif')}\n"
            '    chain_id: ["A"]\n    template_id: ["A"]\n'
            f"  - cif: {quoted(templates / 'binder.cif')}\n"
            '    chain_id: ["B"]\n    template_id: ["B"]\n'
        )
        assert sorted(p.name for p in templates.iterdir()) == ["binder.cif", "receptor.cif"]

    def test_score_lock_lists_one_file_for_both_chains_with_force_and_a_threshold(self, tmp_path):
        work, yaml_path = prepared(tmp_path, make_request(mode="score-lock"))
        templates = work / "input" / "templates"
        assert yaml_path.read_text(encoding="utf-8") == ycr_head() + (
            f"templates:\n  - cif: {quoted(templates / 'lock.cif')}\n"
            '    chain_id: ["A", "B"]\n    template_id: ["A", "B"]\n'
            "    force: true\n    threshold: 2.0\n"
        )
        assert [p.name for p in templates.iterdir()] == ["lock.cif"]

    def test_the_threshold_of_the_request_is_written(self, tmp_path):
        request = make_request(mode="score-lock", lock_threshold_angstrom=3.5)
        _, yaml_path = prepared(tmp_path, request)
        assert "    force: true\n    threshold: 3.5\n" in yaml_path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("mode", ["predict", "refold", "score"])
    def test_force_and_threshold_appear_in_score_lock_only(self, tmp_path, mode):
        _, yaml_path = prepared(tmp_path, make_request(mode=mode))
        text = yaml_path.read_text(encoding="utf-8")
        assert "    force:" not in text and "    threshold:" not in text

    @pytest.mark.parametrize("mode", RUN_MODES)
    def test_the_keys_are_exactly_the_ones_of_the_boltz_schema(self, tmp_path, mode):
        _, yaml_path = prepared(tmp_path, make_request(mode=mode, use_msa_server=False))
        document = read_yaml(yaml_path)
        assert set(document) == {"version", "sequences"} | (
            set() if mode == "predict" else {"templates"}
        )
        assert document["version"] == 1
        assert document["sequences"] == [
            {"protein": {"id": "A", "sequence": YCR_A, "msa": "empty"}},
            {"protein": {"id": "B", "sequence": YCR_B, "msa": "empty"}},
        ]
        for template in document.get("templates", []):
            keys = {"cif", "chain_id", "template_id"}
            assert set(template) == keys | (
                {"force", "threshold"} if mode == "score-lock" else set()
            )
            assert template["chain_id"] == template["template_id"]
        if mode == "score-lock":
            assert document["templates"][0]["force"] is True
            assert document["templates"][0]["threshold"] == 2.0

    def test_without_the_server_every_entity_has_an_empty_msa(self, tmp_path):
        _, yaml_path = prepared(tmp_path, make_request(mode="predict", use_msa_server=False))
        assert yaml_path.read_text(encoding="utf-8") == ycr_head(msa=True)

    def test_with_the_server_no_msa_key_is_written(self, tmp_path):
        _, yaml_path = prepared(tmp_path, make_request(mode="score", use_msa_server=True))
        assert "      msa:" not in yaml_path.read_text(encoding="utf-8")

    def test_the_input_is_written_below_the_work_dir_only(self, tmp_path):
        work, _ = prepared(tmp_path, make_request(mode="score-lock"))
        assert {p.name for p in work.iterdir()} == {"input"}

    def test_prepare_does_not_start_boltz(self, tmp_path, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Boltz-2 started"))
        prepared(tmp_path, make_request(mode="score"))


class TestModifiedResidues:
    """1CWA chain C is cyclosporin: D-Ala, N-methylated residues, MeBmt, Abu and sarcosine."""

    def test_the_sequence_and_the_modifications_are_those_of_the_chain_reader(self, tmp_path):
        request = make_request(path=CWA, name="cwa", binder_chain="C", receptor_chain="A")
        _, yaml_path = prepared(tmp_path, request)
        document = read_yaml(yaml_path)
        receptor, binder = (item["protein"] for item in document["sequences"])
        assert receptor == {"id": "A", "sequence": CWA_A}
        assert binder["id"] == "C" and binder["sequence"] == CWA_C
        assert [(m["position"], m["ccd"]) for m in binder["modifications"]] == CWA_C_MODIFICATIONS
        assert all(set(m) == {"position", "ccd"} for m in binder["modifications"])

    def test_the_exact_text_of_the_modifications(self, tmp_path):
        request = make_request(
            path=CWA, name="cwa", binder_chain="C", receptor_chain="A", mode="predict",
            binder_cyclic=False,
        )  # fmt: skip
        _, yaml_path = prepared(tmp_path, request)
        text = yaml_path.read_text(encoding="utf-8")
        assert text.endswith(
            f'  - protein:\n      id: "C"\n      sequence: "{CWA_C}"\n      modifications:\n'
            '        - position: 1\n          ccd: "DAL"\n'
            '        - position: 2\n          ccd: "MLE"\n'
            '        - position: 3\n          ccd: "MLE"\n'
            '        - position: 4\n          ccd: "MVA"\n'
            '        - position: 5\n          ccd: "BMT"\n'
            '        - position: 6\n          ccd: "ABA"\n'
            '        - position: 7\n          ccd: "SAR"\n'
            '        - position: 8\n          ccd: "MLE"\n'
            '        - position: 10\n          ccd: "MLE"\n'
        )

    def test_the_waters_of_the_chains_are_not_sent(self, tmp_path):
        request = make_request(path=CWA, name="cwa", binder_chain="C", receptor_chain="A")
        _, yaml_path = prepared(tmp_path, request)
        receptor, binder = (item["protein"] for item in read_yaml(yaml_path)["sequences"])
        assert len(receptor["sequence"]) == 165 and len(binder["sequence"]) == 11

    def test_selenocysteine_is_sent_as_cysteine_with_its_ccd_code(self, tmp_path):
        """Boltz-2 reads the letter U as an unknown residue (data/const.py:166)."""
        path = _write(tmp_path, {"A": ["ALA", "GLY", "SER", "LYS"], "B": ["ALA", "SEC", "GLY"]})
        _, yaml_path = prepared(tmp_path, make_request(path=path, mode="predict"))
        binder = read_yaml(yaml_path)["sequences"][1]["protein"]
        assert binder["sequence"] == "ACG"
        assert binder["modifications"] == [{"position": 2, "ccd": "SEC"}]


class TestTheCyclicFlag:
    def test_auto_writes_it_for_a_head_to_tail_binder_only(self, tmp_path):
        cwa = make_request(path=CWA, name="cwa", binder_chain="C", receptor_chain="A")
        _, yaml_path = prepared(tmp_path, cwa)
        receptor, binder = (item["protein"] for item in read_yaml(yaml_path)["sequences"])
        assert binder["cyclic"] is True and "cyclic" not in receptor
        _, yaml_path = prepared(tmp_path / "linear", make_request())
        assert "cyclic" not in yaml_path.read_text(encoding="utf-8")

    def test_true_writes_it_and_false_never_does(self, tmp_path):
        _, yaml_path = prepared(tmp_path / "on", make_request(binder_cyclic=True))
        binder = read_yaml(yaml_path)["sequences"][1]["protein"]
        assert binder["cyclic"] is True
        cwa = make_request(
            path=CWA, name="cwa", binder_chain="C", receptor_chain="A", binder_cyclic=False
        )
        _, yaml_path = prepared(tmp_path / "off", cwa)
        assert "cyclic" not in yaml_path.read_text(encoding="utf-8")

    def test_the_receptor_is_never_cyclic(self, tmp_path):
        _, yaml_path = prepared(tmp_path, make_request(binder_cyclic=True))
        assert "cyclic" not in read_yaml(yaml_path)["sequences"][0]["protein"]


class TestChainIdsAndEntities:
    def test_chain_ids_stay_strings_for_the_yaml_loader(self, tmp_path):
        """N and Y are booleans, and 1 an integer, for a YAML 1.1 loader unless quoted."""
        path = _write(tmp_path, {"N": ["ALA", "GLY", "SER", "LYS"], "Y": ["ALA", "GLY", "SER"]})
        request = make_request(path=path, binder_chain="Y", receptor_chain="N", mode="score")
        _, yaml_path = prepared(tmp_path, request)
        document = read_yaml(yaml_path)
        assert [item["protein"]["id"] for item in document["sequences"]] == ["N", "Y"]
        assert document["templates"][0]["chain_id"] == ["N"]
        assert document["templates"][1]["chain_id"] == ["Y"]

    def test_chains_with_one_sequence_and_one_chemistry_are_allowed(self, tmp_path):
        path = _write(tmp_path, {"A": ["ALA", "GLY", "SER"], "B": ["ALA", "GLY", "SER"]})
        _, yaml_path = prepared(tmp_path, make_request(path=path, mode="score"))
        assert len(read_yaml(yaml_path)["sequences"]) == 2

    def test_one_sequence_with_other_modified_residues_is_refused_before_a_file_is_written(
        self, tmp_path
    ):
        """Boltz-2 makes one entity of them and keeps the modifications of the first."""
        path = _write(tmp_path, {"A": ["ALA", "MLE", "GLY"], "B": ["ALA", "LEU", "GLY"]})
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(ValueError, match="'A' and 'B' have the same sequence"):
            Boltz2Runner().prepare(make_request(path=path), work)
        assert list(work.iterdir()) == []

    def test_one_sequence_with_another_cyclic_flag_is_refused(self, tmp_path):
        path = _write(tmp_path, {"A": ["ALA", "GLY", "SER"], "B": ["ALA", "GLY", "SER"]})
        with pytest.raises(ValueError, match="same cyclic flag"):
            Boltz2Runner().prepare(make_request(path=path, binder_cyclic=True), tmp_path)

    def test_a_chain_that_the_structure_lacks_is_named_with_the_chains_it_has(self, tmp_path):
        with pytest.raises(ValueError, match=r"binder chain 'Q' not found.*chains in its first"):
            Boltz2Runner().prepare(make_request(binder_chain="Q"), tmp_path)

    def test_an_unreadable_structure_is_a_value_error(self, tmp_path):
        garbage = tmp_path / "garbage.cif"
        garbage.write_text("data_x\nloop_\n_atom_site.id\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot read"):
            Boltz2Runner().prepare(make_request(path=garbage), tmp_path)


class TestYamlHelpers:
    @pytest.mark.parametrize(
        "value, text",
        [
            (2.0, "2.0"),
            (2, "2.0"),
            (2.5, "2.5"),
            (0.1, "0.1"),
            (1e-05, "0.00001"),
            (12.25, "12.25"),
        ],
    )
    def test_a_float_is_written_so_that_a_yaml_1_1_loader_reads_a_float(self, value, text):
        assert boltz2_runner._yaml_float(value) == text

    def test_a_text_is_always_double_quoted(self):
        assert boltz2_runner._quoted("N") == '"N"'
        assert boltz2_runner._quoted('a"b\\c') == '"a\\"b\\\\c"'


# ---------------------------------------------------------------------------- the template CIFs


def boltz_view(path):
    """What Boltz-2's mmCIF reader sees (``parse_mmcif``, gemmi calls only), per chain name.

    Returns ``{chain name: (polymer type, one-letter sequence, match string, residue names)}``.
    The chain name is the subchain ID, which is what Boltz-2 calls the name of a template chain.
    """
    block = gemmi.cif.read(str(path))[0]
    structure = gemmi.make_structure_from_block(block)
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()
    entities = {}
    for entity in structure.entities:
        if entity.entity_type.name == "Water":
            continue
        for subchain in entity.subchains:
            entities[subchain] = entity
    view = {}
    for raw in structure[0].subchains():
        entity = entities[raw.subchain_id()]
        sequence = [gemmi.Entity.first_mon(item) for item in entity.full_sequence]
        result = gemmi.align_sequence_to_polymer(
            sequence, raw, entity.polymer_type, gemmi.AlignmentScoring()
        )
        view[raw.subchain_id()] = (
            entity.polymer_type.name,
            gemmi.one_letter_code(sequence),
            result.match_string,
            [residue.name for residue in raw],
        )
    return view


def atom_positions(path):
    """``{(chain, residue number, atom name): (x, y, z)}`` of the first model of a file."""
    structure = gemmi.read_structure(str(path))
    return {
        (chain.name, residue.seqid.num, atom.name): (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in structure[0]
        for residue in chain
        for atom in residue
    }


class TestTheTemplateFiles:
    def test_boltz_sees_each_chain_under_its_own_name_with_every_residue_matched(self, tmp_path):
        work, _ = prepared(tmp_path, make_request(mode="score-lock"))
        view = boltz_view(work / "input" / "templates" / "lock.cif")
        assert set(view) == {"A", "B"}
        for chain, sequence in (("A", YCR_A), ("B", YCR_B)):
            polymer, letters, match, _ = view[chain]
            assert polymer == "PeptideL"
            assert letters == sequence
            assert match == "|" * len(sequence)

    def test_each_mode_writes_the_chains_it_lists(self, tmp_path):
        expected = {
            "refold": {"receptor": {"A"}},
            "score": {"receptor": {"A"}, "binder": {"B"}},
            "score-lock": {"lock": {"A", "B"}},
        }
        for mode, files in expected.items():
            work, _ = prepared(tmp_path / mode, make_request(mode=mode))
            for stem, chains in files.items():
                view = boltz_view(work / "input" / "templates" / f"{stem}.cif")
                assert set(view) == chains

    def test_the_lock_template_keeps_the_frame_of_the_input(self, tmp_path):
        work, _ = prepared(tmp_path, make_request(mode="score-lock"))
        written = atom_positions(work / "input" / "templates" / "lock.cif")
        original = atom_positions(YCR)
        assert written
        for key, position in written.items():
            assert position == pytest.approx(original[key], abs=1e-3)

    def test_a_chain_has_the_same_coordinates_in_the_lock_and_in_its_own_template(self, tmp_path):
        work, _ = prepared(tmp_path / "lock", make_request(mode="score-lock"))
        own, _ = prepared(tmp_path / "own", make_request(mode="score"))
        lock = atom_positions(work / "input" / "templates" / "lock.cif")
        binder = atom_positions(own / "input" / "templates" / "binder.cif")
        assert binder and {key: lock[key] for key in binder} == pytest.approx(binder)

    def test_waters_are_left_out_of_a_chain(self, tmp_path):
        """The waters of 1CWA carry the chain IDs A and C; the sequence has no place for them."""
        request = make_request(
            path=CWA, name="cwa", binder_chain="C", receptor_chain="A", mode="score-lock"
        )
        work, _ = prepared(tmp_path, request)
        path = work / "input" / "templates" / "lock.cif"
        assert "HOH" not in path.read_text(encoding="utf-8")
        view = boltz_view(path)
        assert len(view["A"][3]) == 165 and len(view["C"][3]) == 11

    def test_hydrogens_are_left_out(self, tmp_path):
        with_hydrogen = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("H", "H"), ("HA", "H"))
        path = _write(
            tmp_path,
            {"A": ["ALA", ("GLY", with_hydrogen), "SER"], "B": [("ALA", with_hydrogen), "GLY"]},
        )
        work, _ = prepared(tmp_path, make_request(path=path, mode="score-lock"))
        written = gemmi.read_structure(str(work / "input" / "templates" / "lock.cif"))
        elements = {atom.element.name for chain in written[0] for r in chain for atom in r}
        assert elements and "H" not in elements
        assert len(atom_positions(path)) > len(
            atom_positions(work / "input" / "templates" / "lock.cif")
        )

    def test_a_modified_residue_is_named_after_its_parent_so_the_sequences_align(self, tmp_path):
        """Boltz-2 reads a modified residue of a template as X, which cannot match the query.

        The local alignment of ALLVTAGLVLA against XXXXXXXXVXA (Boltz-2's ``get_local_alignments``)
        gives two short pieces that map different residues of the query onto the same template
        residues.
        """
        request = make_request(
            path=CWA, name="cwa", binder_chain="C", receptor_chain="A", mode="score"
        )
        work, _ = prepared(tmp_path, request)
        polymer, letters, match, names = boltz_view(work / "input" / "templates" / "binder.cif")[
            "C"
        ]
        assert (polymer, letters, match) == ("PeptideL", CWA_C, "|" * 11)
        assert names == [
            "ALA",
            "LEU",
            "LEU",
            "VAL",
            "THR",
            "ALA",
            "GLY",
            "LEU",
            "VAL",
            "LEU",
            "ALA",
        ]

    def test_an_unknown_parent_is_an_unk_residue_and_the_alignment_still_matches(self, tmp_path):
        """With ``on_unmappable_residue="x"`` the residue is sent as X and templated as UNK."""
        path = _write(tmp_path, {"A": ["ALA", "GLY", "SER"], "B": ["ALA", "QQQ", "GLY"]})
        with pytest.raises(ValueError, match="chain 'B': QQQ 2"):
            Boltz2Runner().prepare(make_request(path=path, mode="score"), tmp_path / "w0")
        request = make_request(path=path, mode="score", on_unmappable_residue="x")
        work, yaml_path = prepared(tmp_path, request)
        assert read_yaml(yaml_path)["sequences"][1]["protein"]["sequence"] == "AXG"
        _, letters, match, names = boltz_view(work / "input" / "templates" / "binder.cif")["B"]
        assert (letters, match, names) == ("AXG", "|||", ["ALA", "UNK", "GLY"])


# ---------------------------------------------------------------------------- input errors


class TestAnInputThatCannotBeExpressed:
    @pytest.fixture
    def unexpressible(self, tmp_path):
        """Chain B holds QQQ and ZZZ: they have a backbone and are no amino acid of the CCD."""
        return _write(
            tmp_path,
            {"A": ["ALA", "GLY", "SER", "LYS"], "B": ["ALA", "QQQ", "ZZZ", "GLY"]},
        )

    def test_the_error_names_the_chain_and_the_residues_and_nothing_is_written(
        self, tmp_path, unexpressible
    ):
        work = tmp_path / "work"
        work.mkdir()
        request = make_request(path=unexpressible, mode="score-lock")
        with pytest.raises(ValueError) as info:
            Boltz2Runner().prepare(request, work)
        message = str(info.value)
        assert "Boltz-2 cannot take these residues" in message
        assert "chain 'B': QQQ 2, ZZZ 3" in message
        assert "OpenFold3" not in message
        assert "on_unmappable_residue='x'" in message
        assert list(work.iterdir()) == []

    def test_run_refuses_it_and_the_model_is_never_started(
        self, tmp_path, unexpressible, fake_boltz
    ):
        with pytest.raises(ValueError, match="Boltz-2 cannot take"):
            Boltz2Runner().run(make_request(path=unexpressible), tmp_path / "work")
        assert fake_boltz.calls() == []
        assert not (tmp_path / "work").exists()

    def test_the_store_records_the_refusal_and_does_not_start_the_model(
        self, tmp_path, unexpressible, fake_boltz
    ):
        session = PredictionSession(PredictionStore(tmp_path / "store"), [Boltz2Runner()])
        request = make_request(path=unexpressible)
        with pytest.raises(PredictionFailedError, match="Boltz-2 cannot take these residues"):
            session.record(request)
        assert fake_boltz.calls() == []

    def test_a_structure_without_the_chain_is_refused_before_the_model_starts(
        self, tmp_path, fake_boltz
    ):
        with pytest.raises(ValueError, match="receptor chain 'Z' not found"):
            Boltz2Runner().run(make_request(receptor_chain="Z"), tmp_path / "work")
        assert fake_boltz.calls() == []


# ---------------------------------------------------------------------------- the command line


class TestTheCommandLine:
    def command(self, tmp_path, runner=None, **overrides):
        runner = runner or Boltz2Runner()
        request = make_request(runner, **overrides)
        return runner._command(request, tmp_path / "input" / "ycr.yaml", tmp_path)

    def test_the_default_run(self, tmp_path):
        assert self.command(tmp_path) == [
            "boltz", "predict", str(tmp_path / "input" / "ycr.yaml"),
            "--out_dir", str(tmp_path),
            "--model", "boltz2",
            "--diffusion_samples", "5",
            "--seed", "42",
            "--output_format", "mmcif",
            "--write_full_pae", "--write_full_pde",
            "--use_msa_server",
        ]  # fmt: skip

    def test_the_full_pae_and_pde_are_always_requested(self, tmp_path):
        for mode in RUN_MODES:
            command = self.command(tmp_path, mode=mode, use_msa_server=False)
            assert "--write_full_pae" in command and "--write_full_pde" in command

    def test_the_number_of_samples_is_diffusion_samples(self, tmp_path):
        command = self.command(tmp_path, num_samples=3)
        assert command[command.index("--diffusion_samples") + 1] == "3"

    def test_the_seed_is_the_one_of_the_request(self, tmp_path):
        command = self.command(tmp_path, seeds=[7])
        assert command[command.index("--seed") + 1] == "7"

    def test_the_msa_server_flag_follows_the_option(self, tmp_path):
        assert "--use_msa_server" in self.command(tmp_path, use_msa_server=True)
        assert "--use_msa_server" not in self.command(tmp_path, use_msa_server=False)

    def test_extra_arguments_come_last_and_are_passed_verbatim(self, tmp_path):
        command = self.command(tmp_path, extra_args=["--recycling_steps", "5", "--no_kernels"])
        assert command[-3:] == ["--recycling_steps", "5", "--no_kernels"]

    def test_the_mode_does_not_change_the_command(self, tmp_path):
        commands = {tuple(self.command(tmp_path, mode=mode)) for mode in RUN_MODES}
        assert len(commands) == 1

    def test_the_conda_environment_prefixes_the_command(self, tmp_path):
        command = self.command(tmp_path, runner=Boltz2Runner("boltz"))
        assert command[:7] == [
            "conda",
            "run",
            "-n",
            "boltz",
            "--no-capture-output",
            "boltz",
            "predict",
        ]

    def test_an_empty_environment_name_means_the_current_one(self):
        assert Boltz2Runner("").conda_env is None


# ---------------------------------------------------------------------------- custom weights


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "finetuned.ckpt"
    path.write_bytes(b"weights of a fine-tuned model")
    return path


class TestCustomWeights:
    def test_the_runner_takes_a_checkpoint_file(self):
        assert Boltz2Runner.supports_custom_weights is True
        assert Boltz2Runner.weights_kind == "file"

    def test_the_weights_are_in_the_key_by_content(self, tmp_path, checkpoint):
        plain = make_request()
        request = make_request(weights=checkpoint)
        assert request.weights.sha256 and request.weights.size == checkpoint.stat().st_size
        assert request.key() != plain.key()
        moved = tmp_path / "elsewhere" / "copy.ckpt"
        moved.parent.mkdir()
        moved.write_bytes(checkpoint.read_bytes())
        assert make_request(weights=moved).key() == request.key()
        moved.write_bytes(b"other weights")
        assert make_request(weights=moved).key() != request.key()

    def test_a_request_without_weights_keeps_its_key_and_has_no_checkpoint_option(self):
        request = make_request()
        assert "weights" not in request.canonical()
        assert "checkpoint" not in request.options

    def test_the_weights_reference_of_the_store_is_accepted(self, tmp_path, checkpoint):
        store = PredictionStore(tmp_path / "store")
        reference = store.weights_reference(checkpoint)
        assert make_request(weights=reference).key() == make_request(weights=checkpoint).key()

    def test_the_run_passes_the_file_as_checkpoint(self, tmp_path, checkpoint):
        runner = Boltz2Runner()
        request = make_request(runner, weights=checkpoint)
        command = runner._command(request, tmp_path / "x.yaml", tmp_path)
        assert command[command.index("--checkpoint") + 1] == str(request.weights.path)

    def test_no_checkpoint_flag_without_weights(self, tmp_path):
        runner = Boltz2Runner()
        command = runner._command(make_request(runner), tmp_path / "x.yaml", tmp_path)
        assert "--checkpoint" not in command

    def test_a_directory_is_refused_before_anything_is_written(self, tmp_path, fake_boltz):
        folder = tmp_path / "weights_dir"
        folder.mkdir()
        (folder / "model.ckpt").write_bytes(b"x")
        request = make_request(weights=folder)
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(ValueError, match="takes its weights as a file"):
            Boltz2Runner().prepare(request, work)
        with pytest.raises(ValueError, match="takes its weights as a file"):
            Boltz2Runner().run(request, work)
        assert list(work.iterdir()) == [] and fake_boltz.calls() == []

    def test_weights_that_vanished_before_the_run_are_refused(
        self, tmp_path, checkpoint, fake_boltz
    ):
        request = make_request(weights=checkpoint)
        checkpoint.unlink()
        with pytest.raises(FileNotFoundError, match="do not exist"):
            Boltz2Runner().run(request, tmp_path / "work")
        assert fake_boltz.calls() == []

    def test_a_missing_weights_file_is_refused_when_the_request_is_made(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            make_request(weights=tmp_path / "absent.ckpt")

    def test_the_checkpoint_reaches_the_process(self, tmp_path, checkpoint, fake_boltz):
        fake_boltz.canned("ycr")
        request = make_request(weights=checkpoint)
        Boltz2Runner().run(request, tmp_path / "work")
        argv = fake_boltz.calls()[0]["argv"]
        assert argv[argv.index("--checkpoint") + 1] == str(request.weights.path)

    def test_the_store_refuses_a_request_for_a_runner_without_the_support(
        self, tmp_path, checkpoint
    ):
        class Plain(Boltz2Runner):
            supports_custom_weights = False

        store = PredictionStore(tmp_path / "store")
        with pytest.raises(ValueError, match="does not take custom weights"):
            store.ensure(make_request(weights=checkpoint), Plain())


# ---------------------------------------------------------------------------- the version


class TestTheVersion:
    def test_it_is_asked_once_per_runner(self, boltz_version):
        runner = Boltz2Runner("some_env")
        assert runner.version() == "2.2.1" and runner.version() == "2.2.1"
        assert boltz_version == ["some_env"]

    def test_the_request_key_holds_it(self, monkeypatch):
        key = make_request().key()
        monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda env: "2.3.0")
        assert make_request().model_version == "2.3.0"
        assert make_request().key() != key

    def test_an_unknown_version_is_an_empty_string_in_the_request(self, monkeypatch):
        monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda env: None)
        assert make_request().model_version == ""

    def test_the_current_interpreter_is_asked_through_the_package_metadata(self, monkeypatch):
        import importlib.metadata as metadata

        seen = []
        monkeypatch.setattr(metadata, "version", lambda name: seen.append(name) or "2.2.1")
        assert _REAL_VERSION_PROBE(None) == "2.2.1" and seen == ["boltz"]

        def absent(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "version", absent)
        assert _REAL_VERSION_PROBE(None) is None

    def test_a_conda_environment_is_asked_to_import_boltz(self, monkeypatch):
        seen = {}

        def fake_run(command, **keywords):
            seen["command"] = command
            seen["keywords"] = keywords
            return subprocess.CompletedProcess(command, 0, stdout="note\n2.2.1\n", stderr="")

        monkeypatch.setattr(boltz2_runner.subprocess, "run", fake_run)
        assert _REAL_VERSION_PROBE("boltz") == "2.2.1"
        assert seen["command"][1:5] == ["run", "-n", "boltz", "python"]
        assert seen["command"][-1] == "import boltz; print(boltz.__version__)"
        assert seen["keywords"]["encoding"] == "utf-8"

    @pytest.mark.parametrize(
        "outcome",
        [
            subprocess.CompletedProcess([], 1, stdout="", stderr="ModuleNotFoundError"),
            subprocess.CompletedProcess([], 0, stdout="not a version\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            OSError("no conda"),
            subprocess.TimeoutExpired("conda", 60),
        ],
    )
    def test_an_environment_that_cannot_import_boltz_has_no_version(self, monkeypatch, outcome):
        def fake_run(command, **keywords):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(boltz2_runner.subprocess, "run", fake_run)
        assert _REAL_VERSION_PROBE("boltz") is None


class TestAvailability:
    def test_the_current_environment_needs_boltz_on_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        assert Boltz2Runner().is_available() is False
        executable = tmp_path / "boltz"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        assert Boltz2Runner().is_available() is True

    def test_a_conda_environment_needs_a_version(self, monkeypatch):
        monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda env: None)
        assert Boltz2Runner("boltz").is_available() is False
        monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda env: "2.2.1")
        assert Boltz2Runner("boltz").is_available() is True

    def test_a_missing_installation_is_reported_before_the_model_starts(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError, match="boltz not found on PATH"):
            Boltz2Runner().run(make_request(), tmp_path / "work")

    def test_the_store_does_not_record_a_failure_for_a_missing_installation(
        self, tmp_path, monkeypatch
    ):
        from binding_metrics.predictors.store import PredictionUnavailableError

        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        store = PredictionStore(tmp_path / "store")
        with pytest.raises(PredictionUnavailableError):
            store.ensure(make_request(), Boltz2Runner())
        assert not (tmp_path / "store").exists() or not any(
            (tmp_path / "store").rglob("STATUS.json")
        )


# ---------------------------------------------------------------------------- a stub Boltz-2

_STUB = r"""
import json, os, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
yaml_path = Path(argv[1])
templates = yaml_path.parent / "templates"
record = {
    "argv": argv,
    "yaml": yaml_path.read_text(encoding="utf-8"),
    "templates": sorted(p.name for p in templates.glob("*.cif")) if templates.is_dir() else [],
    "unbuffered": os.environ.get("PYTHONUNBUFFERED"),
}
with open(os.environ["FAKE_BOLTZ_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")

behaviour = os.environ.get("FAKE_BOLTZ_BEHAVIOUR", "ok")
out_dir = Path(argv[argv.index("--out_dir") + 1])
stem = yaml_path.stem
if behaviour == "ok":
    print("Running structure prediction for 1 input.")
    print("Predicting:  10%\rPredicting: 100%", file=sys.stderr)
    source = Path(os.environ["FAKE_BOLTZ_CANNED"]) / ("boltz_results_" + stem)
    shutil.copytree(source, out_dir / ("boltz_results_" + stem), dirs_exist_ok=True)
elif behaviour == "crash":
    sys.stderr.write("Predicting:  10%\rPredicting:  60%\n")
    print("Traceback (most recent call last):", file=sys.stderr)
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", file=sys.stderr)
    sys.exit(1)
elif behaviour == "broken_install":
    print("Traceback (most recent call last):", file=sys.stderr)
    print("ModuleNotFoundError: No module named 'boltz'", file=sys.stderr)
    sys.exit(1)
elif behaviour == "swallowed":
    # process_input of boltz/main.py prints this and goes on; the run ends with status 0
    print("Traceback (most recent call last):", file=sys.stderr)
    print("ValueError: Template lock must have threshold specified if force is set to True",
          file=sys.stderr)
    print("Failed to process " + str(yaml_path) + ". Skipping. Error: Template lock must have "
          "threshold specified if force is set to True.")
    print("Number of failed examples: 0")
elif behaviour == "oom_skipped":
    print("| WARNING: ran out of memory, skipping batch")
    print("Number of failed examples: 1")
elif behaviour == "partial_stdout":
    sys.stdout.write("no newline at the end")
"""

_CONDA = """#!/bin/sh
# conda run -n NAME --no-capture-output COMMAND...
echo "$@" >> "$FAKE_CONDA_LOG"
shift 4
exec "$@"
"""


class FakeBoltz:
    """A ``boltz`` (and a ``conda`` that runs what follows ``conda run -n NAME``) on PATH."""

    def __init__(self, tmp_path, monkeypatch):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.bin = tmp_path / "fakebin"
        self.bin.mkdir()
        self.log = tmp_path / "boltz_calls.jsonl"
        self.conda_log = tmp_path / "conda_calls.txt"
        script = self.bin / "boltz_stub.py"
        script.write_text(_STUB, encoding="utf-8")
        wrapper = self.bin / "boltz"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        wrapper.chmod(0o755)
        conda = self.bin / "conda"
        conda.write_text(_CONDA, encoding="utf-8")
        conda.chmod(0o755)
        monkeypatch.setenv("PATH", f"{self.bin}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_BOLTZ_LOG", str(self.log))
        monkeypatch.setenv("FAKE_CONDA_LOG", str(self.conda_log))
        self.behave("ok")

    def behave(self, behaviour):
        self.monkeypatch.setenv("FAKE_BOLTZ_BEHAVIOUR", behaviour)

    def canned(self, name, samples=1):
        """Have the stub write ``samples`` synthetic samples as the output of the input ``name``."""
        directory = self.tmp_path / "canned"
        for rank in range(samples):
            synth_boltz2.write_prediction(
                directory, name, synth.synthetic_complex(plddt_shift=5.0 * rank), sample=rank + 1
            )
        self.monkeypatch.setenv("FAKE_BOLTZ_CANNED", str(directory))
        return directory

    def calls(self):
        if not self.log.is_file():
            return []
        return [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


@pytest.fixture
def fake_boltz(tmp_path, monkeypatch):
    return FakeBoltz(tmp_path, monkeypatch)


class TestRunningTheStub:
    def test_run_returns_the_folder_that_the_adapter_loads(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr")
        work = tmp_path / "work"
        work.mkdir()
        directory = Boltz2Runner().run(make_request(mode="score-lock"), work)
        assert directory == work.absolute() / "boltz_results_ycr" / "predictions" / "ycr"
        assert Boltz2Parser().find_files(directory, "ycr").has_output()

    def test_the_process_gets_the_yaml_the_templates_and_the_arguments(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr")
        work = tmp_path / "work"
        work.mkdir()
        runner = Boltz2Runner()
        request = make_request(runner, mode="score-lock", seeds=[3], num_samples=2)
        runner.run(request, work)
        (call,) = fake_boltz.calls()
        argv = call["argv"]
        assert argv[:2] == ["predict", str(work.absolute() / "input" / "ycr.yaml")]
        assert argv[argv.index("--out_dir") + 1] == str(work.absolute())
        assert argv[argv.index("--seed") + 1] == "3"
        assert argv[argv.index("--diffusion_samples") + 1] == "2"
        assert {"--write_full_pae", "--write_full_pde", "--use_msa_server"} <= set(argv)
        assert call["templates"] == ["lock.cif"]
        assert "    force: true\n    threshold: 2.0\n" in call["yaml"]
        assert call["unbuffered"] == "1"

    def test_the_run_is_started_through_conda_when_an_environment_is_set(
        self, tmp_path, fake_boltz
    ):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner("myenv")
        runner.run(make_request(runner), tmp_path / "work")
        assert len(fake_boltz.calls()) == 1
        line = fake_boltz.conda_log.read_text(encoding="utf-8").splitlines()[0]
        assert line.startswith("run -n myenv --no-capture-output boltz predict ")

    def test_the_output_of_the_model_goes_to_stderr_and_stdout_stays_free(
        self, tmp_path, fake_boltz, capfd
    ):
        fake_boltz.canned("ycr")
        Boltz2Runner().run(make_request(), tmp_path / "work")
        captured = capfd.readouterr()
        assert "Running structure prediction for 1 input." in captured.err
        assert "Predicting: 100%" in captured.err
        assert captured.out == ""


class TestTheRoundTripThroughTheStore:
    def test_the_store_and_the_session_read_what_the_stub_wrote(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        store = PredictionStore(tmp_path / "store")
        session = PredictionSession(store, [runner])
        request = make_request(runner, mode="score-lock")
        record = session.record(request)

        assert record.model == "boltz2" and record.name == "ycr"
        truth = synth.synthetic_complex()
        assert record.ranking_score == pytest.approx(truth.scalars["ranking_score"], abs=1e-3)
        assert record.iptm == pytest.approx(truth.scalars["iptm"], abs=1e-3)
        assert record.ptm == pytest.approx(truth.scalars["ptm"], abs=1e-3)
        assert record.pae.shape == (7, 7) and record.pde.shape == (7, 7)
        assert record.pae == pytest.approx(truth.pae, abs=1e-2)
        assert record.pde == pytest.approx(truth.pde, abs=1e-2)
        assert set(record.chain_pair_iptm) == {"A-B", "B-A"}

        entry = session.entry(request)
        assert entry.ok and entry.executed_here
        assert entry.runner_name == "boltz2" and entry.runner_version == "2.2.1"
        assert entry.prediction_dir.name == "ycr"
        assert entry.prediction_dir.parent.parent.name == "boltz_results_ycr"
        assert session.stats()["runs"] == 1

    def test_the_model_runs_once_for_one_request(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        store = PredictionStore(tmp_path / "store")
        request = make_request(runner)
        PredictionSession(store, [runner]).record(request)
        PredictionSession(store, [runner]).record(request)
        assert len(fake_boltz.calls()) == 1

    def test_a_second_request_with_another_name_and_the_same_content_shares_the_run(
        self, tmp_path, fake_boltz
    ):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        session.record(make_request(runner))
        record = session.record(make_request(runner, name="other"))
        assert record.name == "ycr" and len(fake_boltz.calls()) == 1

    def test_the_samples_are_ranked_as_boltz_wrote_them(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr", samples=2)
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        request = make_request(runner, num_samples=2)
        best = session.record(request, sample=1)
        second = session.record(request, sample=2)
        assert best.avg_plddt > second.avg_plddt
        assert best.avg_plddt - second.avg_plddt == pytest.approx(5.0, abs=2.5)

    def test_prepare_then_run_in_one_work_dir_is_allowed(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        work = tmp_path / "work"
        work.mkdir()
        request = make_request(runner, mode="refold")
        runner.prepare(request, work)
        runner.run(request, work)
        assert len(fake_boltz.calls()) == 1


# ---------------------------------------------------------------------------- failures


class TestFailures:
    def test_a_non_zero_exit_keeps_the_reason_the_hint_and_the_command(self, tmp_path, fake_boltz):
        fake_boltz.behave("crash")
        with pytest.raises(subprocess.CalledProcessError) as info:
            Boltz2Runner().run(make_request(), tmp_path / "work")
        error = info.value
        assert isinstance(error, Boltz2RunError) and error.returncode == 1
        text = str(error)
        assert text.splitlines()[0].startswith(
            "Boltz-2 exited with status 1: torch.OutOfMemoryError: CUDA out of memory"
        )
        assert "Hint: The GPU ran out of memory" in text
        assert "Last lines of output" in text
        assert "Command: ['boltz', 'predict'" in text
        assert "Predicting:  60%" in text and "10%" not in text  # a bar keeps its last state
        assert error.hint.startswith("The GPU ran out of memory")
        assert "CUDA out of memory" in error.stderr

    def test_the_reason_survives_the_200_character_cut_of_the_pipeline(self, tmp_path, fake_boltz):
        """``cli.run._collect_failures`` keeps the first 200 characters of ``str(error)``."""
        fake_boltz.behave("crash")
        with pytest.raises(Boltz2RunError) as info:
            Boltz2Runner().run(make_request(), tmp_path / "work")
        assert "CUDA out of memory" in str(info.value)[:200]

    def test_the_store_records_the_reason_and_does_not_retry(self, tmp_path, fake_boltz):
        fake_boltz.behave("crash")
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        request = make_request(runner)
        with pytest.raises(PredictionFailedError) as info:
            session.record(request)
        assert info.value.reason.startswith("Boltz2RunError: Boltz-2 exited with status 1")
        assert "CUDA out of memory" in info.value.reason
        with pytest.raises(PredictionFailedError):
            PredictionSession(PredictionStore(tmp_path / "store"), [runner]).record(request)
        assert len(fake_boltz.calls()) == 1

    def test_a_failure_that_boltz_swallows_is_not_stored_as_done(self, tmp_path, fake_boltz):
        """``process_input`` prints the error and the run exits with status 0."""
        fake_boltz.behave("swallowed")
        with pytest.raises(RuntimeError) as info:
            Boltz2Runner().run(make_request(mode="score-lock"), tmp_path / "work")
        text = str(info.value)
        assert text.startswith("Boltz-2 exited normally but wrote no output for 'ycr' in ")
        assert "Failed to process" in text
        assert "must have threshold specified if force is set to True" in text

    def test_a_batch_skipped_for_memory_is_a_failure_with_the_advice(self, tmp_path, fake_boltz):
        fake_boltz.behave("oom_skipped")
        with pytest.raises(RuntimeError) as info:
            Boltz2Runner().run(make_request(), tmp_path / "work")
        text = str(info.value)
        assert "wrote no output for 'ycr'" in text
        assert "ran out of memory, skipping batch" in text
        assert "Hint: The GPU ran out of memory" in text

    def test_a_package_that_cannot_be_imported_is_explained(self, tmp_path, fake_boltz):
        fake_boltz.behave("broken_install")
        with pytest.raises(Boltz2RunError, match="package cannot be imported"):
            Boltz2Runner().run(make_request(), tmp_path / "work")

    def test_output_without_a_final_newline_is_kept(self, tmp_path, fake_boltz):
        fake_boltz.behave("partial_stdout")
        with pytest.raises(RuntimeError, match="no newline at the end"):
            Boltz2Runner().run(make_request(), tmp_path / "work")

    def test_a_run_interrupted_by_the_caller_kills_the_process(self, tmp_path, monkeypatch):
        killed = []

        def interrupted(size):
            raise KeyboardInterrupt

        class Process:
            returncode = None
            stdout = types.SimpleNamespace(read1=interrupted, close=lambda: None)

            def kill(self):
                killed.append(True)

            def wait(self):
                return 0

        monkeypatch.setattr(boltz2_runner.subprocess, "Popen", lambda *a, **k: Process())
        with pytest.raises(KeyboardInterrupt):
            boltz2_runner._run_boltz_command(["boltz"])
        assert killed

    @pytest.mark.parametrize(
        "output, advice",
        [
            ("torch.OutOfMemoryError: CUDA out of memory", "GPU ran out of memory"),
            ("RuntimeError: CUDA error: out of memory", "GPU ran out of memory"),
            ("ImportError: Error importing x from cuequivariance_ops_torch", "--no_kernels"),
            ("MisconfigurationException: No supported gpu backend found!", "No GPU is visible"),
            ("ModuleNotFoundError: No module named 'boltz'", "cannot be imported"),
            ("requests.exceptions.ConnectionError: Max retries exceeded", "network request failed"),
            ("ValueError: something nobody has seen", ""),
        ],
    )
    def test_known_failures_get_advice_and_unknown_ones_none(self, output, advice):
        hint = boltz2_runner._hint(output)
        assert (advice in hint) if advice else hint == ""

    def test_the_key_line_is_the_last_exception_or_the_last_line(self):
        lines = ["Traceback (most recent call last):", "  File x", "KeyError: 'a'", "cleanup done"]
        assert boltz2_runner._key_line(lines) == "KeyError: 'a'"
        assert boltz2_runner._key_line(["one", "two"]) == "two"
        assert boltz2_runner._key_line([]) == ""

    def test_the_no_output_message_names_the_directory_and_the_name(self, tmp_path):
        request = make_request()
        message = boltz2_runner._no_output_message(request, tmp_path, "")
        assert message == f"Boltz-2 exited normally but wrote no output for 'ycr' in {tmp_path}"


# ---------------------------------------------------------------------------- the source


@pytest.fixture(scope="module")
def boltz_source():
    from tests.test_pre_structures import model_source_text

    return {
        "main": model_source_text("boltz", "src/boltz/main.py"),
        "schema": model_source_text("boltz", "src/boltz/data/parse/schema.py"),
        "docs": model_source_text("boltz", "docs/prediction.md"),
        "const": model_source_text("boltz", "src/boltz/data/const.py"),
        "writer": model_source_text("boltz", "src/boltz/data/write/writer.py"),
        "boltz2": model_source_text("boltz", "src/boltz/model/models/boltz2.py"),
    }


class TestTheSourceBehindIt:
    """Re-read the Boltz-2 source when a clone is at hand (v2.2.1; see the module docstring)."""

    @pytest.mark.parametrize(
        "flag",
        [
            "--out_dir", "--model", "--diffusion_samples", "--seed", "--output_format",
            "--write_full_pae", "--write_full_pde", "--use_msa_server", "--checkpoint",
        ],
    )  # fmt: skip
    def test_every_flag_the_runner_sets_is_a_flag_of_boltz_predict(self, boltz_source, flag):
        assert f'    "{flag}",\n' in boltz_source["main"]
        assert flag in boltz2_runner._OWNED_FLAGS

    def test_the_output_folder_is_named_after_the_input_file(self, boltz_source):
        assert 'out_dir = out_dir / f"boltz_results_{data.stem}"' in boltz_source["main"]
        assert 'output_dir=out_dir / "predictions",' in boltz_source["main"]

    def test_the_files_are_those_that_the_adapter_reads(self, boltz_source):
        for pattern in ('f"{record.id}_model_{idx_to_rank[model_idx]}"', "confidence_{record.id}",
                        "plddt_{record.id}", "pae_{record.id}", "pde_{record.id}"):  # fmt: skip
            assert pattern in boltz_source["writer"]

    def test_a_failure_of_the_preparation_is_printed_and_the_run_goes_on(self, boltz_source):
        assert 'print(f"Failed to process {path}. Skipping. Error: {e}.")' in boltz_source["main"]

    def test_a_forced_template_needs_a_threshold_and_has_no_default(self, boltz_source):
        assert "must have threshold specified if force is set to True" in boltz_source["schema"]
        assert "threshold: DISTANCE_THRESHOLD" in boltz_source["docs"]

    def test_the_msa_can_be_left_empty(self, boltz_source):
        assert (
            "To force single-sequence mode" in boltz_source["docs"]
            and "`msa: empty`" in boltz_source["docs"]
        )

    def test_the_letter_u_is_an_unknown_residue(self, boltz_source):
        assert '"U": "UNK"' in boltz_source["const"]

    def test_entities_are_grouped_by_type_and_sequence(self, boltz_source):
        assert (
            "items_to_group.setdefault((entity_type, seq), []).append(item)"
            in boltz_source["schema"]
        )
        assert 'cyclic = items[0][entity_type].get("cyclic", False)' in boltz_source["schema"]

    def test_the_template_potential_is_on_without_use_potentials(self, boltz_source):
        assert "contact_guidance_update: bool = True" in boltz_source["main"]
        assert "steering_args.fk_steering = use_potentials" in boltz_source["main"]

    def test_the_pae_and_pde_flags_are_not_read_by_boltz2(self, boltz_source):
        assert (
            "write_full_pae" not in boltz_source["boltz2"]
            and "write_full_pde" not in boltz_source["boltz2"]
        )
        assert 'pred_dict["pde"] = out["pde"]' in boltz_source["boltz2"]

    def test_a_batch_out_of_memory_is_skipped_with_a_warning(self, boltz_source):
        assert "ran out of memory, skipping batch" in boltz_source["boltz2"]
        assert 'if prediction["exception"]:' in boltz_source["writer"]
