"""Boltz-2 with a given receptor MSA, and the MSA policy that forbids the server.

``Boltz2Runner.make_request(..., msa_policy=, receptor_msa=)`` writes ``msa: <file>`` on the
receptor entity and ``msa: "empty"`` on the binder, keeps the alignment in the key by content, and
never puts ``--use_msa_server`` on the command of a run under ``forbid-server``. No Boltz-2, no GPU
and no network are needed: the process is the stub of ``tests/predictors/test_boltz2_runner.py``.
What Boltz-2 does with the YAML is read from its source when a clone is at hand (the source tests
skip otherwise) and was run for real outside the test suite.
"""

import shutil
import sys

import pytest

pytest.importorskip("gemmi")
pytest.importorskip("yaml")

from binding_metrics.predictors import boltz2_runner  # noqa: E402
from binding_metrics.predictors.boltz2_runner import Boltz2Runner  # noqa: E402
from binding_metrics.predictors.session import PredictionSession  # noqa: E402
from binding_metrics.predictors.store import PredictionRequest, PredictionStore  # noqa: E402
from tests.predictors.test_boltz2_runner import (  # noqa: E402
    CWA,
    CWA_A,
    YCR,
    YCR_A,
    YCR_B,
    FakeBoltz,
    quoted,
    read_yaml,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stubs are POSIX shell scripts")

#: A ``boltz`` that records what the process was given, including the alignment files it can read,
#: and copies a canned output (a variant of the stub of ``test_boltz2_runner``).
_MSA_STUB = r"""
import json, os, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
yaml_path = Path(argv[1])
msa_dir = yaml_path.parent / "msa"
record = {
    "argv": argv,
    "yaml": yaml_path.read_text(encoding="utf-8"),
    "msa": {p.name: p.read_text(encoding="utf-8") for p in sorted(msa_dir.glob("*"))}
    if msa_dir.is_dir() else {},
}
with open(os.environ["FAKE_BOLTZ_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
out_dir = Path(argv[argv.index("--out_dir") + 1])
stem = yaml_path.stem
source = Path(os.environ["FAKE_BOLTZ_CANNED"]) / ("boltz_results_" + stem)
shutil.copytree(source, out_dir / ("boltz_results_" + stem), dirs_exist_ok=True)
"""

#: Keys of 1YCR requests (binder B, receptor A, version 2.2.1, defaults otherwise, mode ``score``
#: or ``predict``) as the runner computed them before it knew the two MSA settings.
KEYS_BEFORE = {
    ("score", True): "692fe603a5b9eed871027a1333345b5d9981b769475112c2ac1a60db64129028",
    ("score", False): "11208cf8635209c107737ab93561b9841fb15efa897959ec8ac49dec4057b3f2",
    ("predict", True): "ff15638f706ab934ac755157bfec5c7c827bd3a7930db0bb1d83e78204060b32",
    ("predict", False): "bca1d4b3dd349adf065ae18193a481b352f60ade1f1580eafae4653d22b5b09b",
}


@pytest.fixture(autouse=True)
def boltz_version(monkeypatch):
    """Boltz-2 is not installed here: the version is 2.2.1 whatever the machine has."""
    monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda conda_env: "2.2.1")


@pytest.fixture
def fake_boltz(tmp_path, monkeypatch):
    fake = FakeBoltz(tmp_path, monkeypatch)
    (fake.bin / "boltz_stub.py").write_text(_MSA_STUB, encoding="utf-8")
    return fake


def a3m_rows(query, extra=2):
    """An alignment of ``query``: the query first, then ``extra`` made-up hits with gaps."""
    rows = [(">query", query)]
    for number in range(extra):
        hit = list(query)
        for place in range(number, len(hit), 7):
            hit[place] = "A" if hit[place] != "A" else "G"
        hit[3 + number] = "-"
        rows.append((f">hit_{number}", "".join(hit[:20]) + "ac" + "".join(hit[20:])))
    return rows


@pytest.fixture
def alignment(tmp_path):
    """An a3m for the receptor of 1YCR."""
    path = tmp_path / "msa_files" / "mdm2.a3m"
    path.parent.mkdir()
    path.write_text(
        "".join(f"{header}\n{row}\n" for header, row in a3m_rows(YCR_A)), encoding="utf-8"
    )
    return path


def make_request(path=YCR, binder="B", receptor="A", **keywords):
    keywords.setdefault("name", "ycr")
    keywords.setdefault("mode", "predict")
    keywords.setdefault("use_msa_server", False)
    return Boltz2Runner().make_request(
        path, binder_chain=binder, receptor_chain=receptor, **keywords
    )


def prepare(tmp_path, request):
    """``(input directory, YAML path)`` after ``prepare``."""
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    yaml_path = Boltz2Runner().prepare(request, work)
    return work / "input", yaml_path


def hand_made(**options):
    """A request as another version or a caller could build it, without ``make_request``."""
    return PredictionRequest(
        "boltz2",
        "ycr",
        mode="predict",
        input_path=YCR,
        binder_chain="B",
        receptor_chain="A",
        options={"use_msa_server": False, **options},
    )


class TestTheRequest:
    @pytest.mark.parametrize("mode, server", sorted(KEYS_BEFORE))
    def test_a_request_without_the_new_fields_has_the_key_it_had(self, mode, server):
        request = make_request(mode=mode, use_msa_server=server)
        assert request.key() == KEYS_BEFORE[mode, server]
        assert "msa_policy" not in request.options and "receptor_msa" not in request.options
        assert dict(request.extra_files) == {}

    def test_the_defaults_spelled_out_change_nothing(self):
        spelled = make_request(msa_policy="default", receptor_msa=None)
        assert spelled.key() == make_request().key()

    def test_a_file_goes_to_the_extra_files_and_is_hashed_by_content(self, alignment):
        request = make_request(receptor_msa=alignment)
        assert request.extra_files["receptor_msa"] == alignment
        assert set(request.content_hashes) == {"input", "receptor_msa"}
        assert "receptor_msa" not in request.options  # a file is not an option

    def test_the_word_single_is_an_option_and_adds_no_file(self):
        request = make_request(receptor_msa="single")
        assert request.options["receptor_msa"] == "single"
        assert dict(request.extra_files) == {} and set(request.content_hashes) == {"input"}

    def test_the_policy_is_an_option_only_when_it_forbids(self, alignment):
        request = make_request(msa_policy="forbid-server", receptor_msa=alignment)
        assert request.options["msa_policy"] == "forbid-server"
        assert request.options["use_msa_server"] is False  # stays in the key as before

    def test_the_same_content_is_the_same_key_at_another_path(self, tmp_path, alignment):
        moved = tmp_path / "elsewhere" / "other_name.a3m"
        moved.parent.mkdir()
        shutil.copy(alignment, moved)
        assert make_request(receptor_msa=moved).key() == make_request(receptor_msa=alignment).key()

    def test_another_content_is_another_key(self, tmp_path, alignment):
        other = tmp_path / "other.a3m"
        other.write_text(
            alignment.read_text(encoding="utf-8") + ">more\n" + YCR_A + "\n", encoding="utf-8"
        )
        assert make_request(receptor_msa=other).key() != make_request(receptor_msa=alignment).key()

    def test_the_policy_a_file_and_the_single_sequence_are_four_requests(self, alignment):
        keys = {
            "none": make_request().key(),
            "file": make_request(receptor_msa=alignment).key(),
            "file, forbidden": make_request(
                receptor_msa=alignment, msa_policy="forbid-server"
            ).key(),
            "single": make_request(receptor_msa="single").key(),
            "single, forbidden": make_request(
                receptor_msa="single", msa_policy="forbid-server"
            ).key(),
        }
        assert len(set(keys.values())) == len(keys)

    def test_the_server_is_refused_under_the_policy(self, alignment):
        with pytest.raises(ValueError, match="would be sent to an MSA server") as caught:
            make_request(use_msa_server=True, msa_policy="forbid-server", receptor_msa=alignment)
        assert str(caught.value).startswith("boltz2: ")

    def test_the_default_use_of_the_server_is_refused_under_the_policy_too(self, alignment):
        with pytest.raises(ValueError, match="would be sent to an MSA server"):
            Boltz2Runner().make_request(
                YCR,
                name="ycr",
                binder_chain="B",
                receptor_chain="A",
                msa_policy="forbid-server",
                receptor_msa=alignment,
            )

    def test_a_receptor_without_a_source_is_refused_under_the_policy(self):
        with pytest.raises(ValueError, match="needs a source for the MSA of the receptor"):
            make_request(msa_policy="forbid-server")

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="msa_policy must be one of"):
            make_request(msa_policy="never")

    def test_a_file_that_is_absent_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="receptor_msa is not a file"):
            make_request(receptor_msa=tmp_path / "absent.a3m")

    @pytest.mark.parametrize(
        "name", ["receptor.fasta", "receptor.sto", "receptor", "receptor.a3m.gz"]
    )
    def test_only_the_two_formats_boltz_reads_are_taken(self, tmp_path, name):
        path = tmp_path / name
        path.write_text(f">q\n{YCR_A}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.a3m or \.csv"):
            make_request(receptor_msa=path)

    def test_the_first_sequence_must_be_the_receptor(self, tmp_path):
        wrong = tmp_path / "wrong.a3m"
        wrong.write_text(f">q\n{YCR_A[:-1]}\n>hit\n{YCR_A[:-1]}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="first sequence .* is not the receptor sequence"):
            make_request(receptor_msa=wrong)
        other = tmp_path / "other.a3m"
        other.write_text(f">q\n{'G' * len(YCR_A)}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="differs from the receptor at position 1"):
            make_request(receptor_msa=other)

    def test_the_word_single_needs_no_file_check(self):
        assert make_request(receptor_msa="single", msa_policy="forbid-server")


class TestTheYaml:
    def test_a_file_is_written_on_the_receptor_and_the_binder_is_single_sequence(
        self, tmp_path, alignment
    ):
        input_dir, yaml_path = prepare(tmp_path, make_request(receptor_msa=alignment))
        staged = input_dir / "msa" / "receptor.a3m"
        assert yaml_path.read_text(encoding="utf-8") == (
            "version: 1\nsequences:\n"
            f'  - protein:\n      id: "A"\n      sequence: "{YCR_A}"\n'
            f"      msa: {quoted(staged)}\n"
            f'  - protein:\n      id: "B"\n      sequence: "{YCR_B}"\n      msa: "empty"\n'
        )

    def test_the_file_is_staged_below_the_input_directory_with_its_content(
        self, tmp_path, alignment
    ):
        input_dir, _ = prepare(tmp_path, make_request(receptor_msa=alignment))
        staged = input_dir / "msa" / "receptor.a3m"
        assert staged.read_bytes() == alignment.read_bytes()
        assert [p.name for p in (input_dir / "msa").iterdir()] == ["receptor.a3m"]
        assert alignment.exists()  # the caller's file is copied, not moved

    def test_the_schema_of_boltz_sees_a_path_and_an_empty_msa(self, tmp_path, alignment):
        input_dir, yaml_path = prepare(tmp_path, make_request(receptor_msa=alignment))
        document = read_yaml(yaml_path)
        receptor, binder = (item["protein"] for item in document["sequences"])
        assert receptor["msa"] == str(input_dir / "msa" / "receptor.a3m")
        assert binder["msa"] == "empty"

    def test_a_csv_keeps_its_extension_and_an_upper_case_one_is_read_as_lower_case(self, tmp_path):
        path = tmp_path / "MDM2.CSV"
        path.write_text(f"key,sequence\n-1,{YCR_A}\n3,{YCR_A}\n", encoding="utf-8")
        input_dir, yaml_path = prepare(tmp_path, make_request(receptor_msa=path))
        assert [p.name for p in (input_dir / "msa").iterdir()] == ["receptor.csv"]
        assert read_yaml(yaml_path)["sequences"][0]["protein"]["msa"].endswith("receptor.csv")

    def test_the_policy_changes_nothing_in_the_yaml_of_a_given_file(self, tmp_path, alignment):
        _, free = prepare(tmp_path / "a", make_request(receptor_msa=alignment))
        _, forbidden = prepare(
            tmp_path / "b", make_request(receptor_msa=alignment, msa_policy="forbid-server")
        )
        assert free.read_text(encoding="utf-8").replace("/a/", "/x/") == (
            forbidden.read_text(encoding="utf-8").replace("/b/", "/x/")
        )

    def test_a_single_sequence_receptor_is_empty_like_the_binder(self, tmp_path):
        _, yaml_path = prepare(
            tmp_path, make_request(receptor_msa="single", msa_policy="forbid-server")
        )
        assert yaml_path.read_text(encoding="utf-8").count('      msa: "empty"\n') == 2

    def test_with_the_server_on_a_given_file_still_leaves_the_binder_single_sequence(
        self, tmp_path, alignment
    ):
        """Boltz-2 refuses a custom MSA next to an automatic one in one input."""
        _, yaml_path = prepare(tmp_path, make_request(receptor_msa=alignment, use_msa_server=True))
        receptor, binder = (item["protein"] for item in read_yaml(yaml_path)["sequences"])
        assert receptor["msa"].endswith("receptor.a3m") and binder["msa"] == "empty"

    def test_without_the_new_fields_the_yaml_is_what_it_was(self, tmp_path):
        _, with_server = prepare(tmp_path / "a", make_request(use_msa_server=True))
        _, without = prepare(tmp_path / "b", make_request(use_msa_server=False))
        assert "msa:" not in with_server.read_text(encoding="utf-8")
        assert without.read_text(encoding="utf-8").count('      msa: "empty"\n') == 2

    @pytest.mark.parametrize("mode", ["predict", "refold", "score", "score-lock", "refold-lock"])
    def test_every_mode_takes_the_alignment_and_keeps_its_templates(
        self, tmp_path, alignment, mode
    ):
        _, yaml_path = prepare(tmp_path, make_request(receptor_msa=alignment, mode=mode))
        document = read_yaml(yaml_path)
        assert document["sequences"][0]["protein"]["msa"].endswith("receptor.a3m")
        assert ("templates" in document) is (mode != "predict")

    def test_a_modified_cyclic_binder_keeps_its_lines_after_the_msa_line(self, tmp_path):
        alignment = tmp_path / "cyclophilin.a3m"
        alignment.write_text(f">q\n{CWA_A}\n", encoding="utf-8")
        request = make_request(CWA, binder="C", receptor="A", receptor_msa=alignment)
        _, yaml_path = prepare(tmp_path, request)
        binder = read_yaml(yaml_path)["sequences"][1]["protein"]
        assert binder["msa"] == "empty" and binder["cyclic"] is True
        assert len(binder["modifications"]) == 9

    def test_the_binder_and_the_receptor_may_not_share_a_sequence(self, tmp_path, alignment):
        """Boltz-2 reads chains with one sequence as one entity, which has one MSA."""
        structure = tmp_path / "twins.pdb"
        structure.write_text(_twin_chains(), encoding="utf-8")
        twin = tmp_path / "twin.a3m"
        twin.write_text(">q\nGGG\n", encoding="utf-8")
        with pytest.raises(ValueError, match="same sequence"):
            make_request(structure, receptor_msa=twin)
        request = make_request(structure)
        by_hand = PredictionRequest(
            "boltz2",
            "ycr",
            mode="predict",
            input_path=structure,
            binder_chain="B",
            receptor_chain="A",
            options=request.options,
            extra_files={"receptor_msa": twin},
        )
        with pytest.raises(ValueError, match="same sequence"):
            Boltz2Runner().prepare(by_hand, tmp_path / "work")
        assert not (tmp_path / "work").exists()

    def test_a_request_made_by_hand_with_a_wrong_first_sequence_writes_nothing(self, tmp_path):
        wrong = tmp_path / "wrong.a3m"
        wrong.write_text(f">q\n{YCR_B}\n", encoding="utf-8")
        request = PredictionRequest(
            "boltz2",
            "ycr",
            mode="predict",
            input_path=YCR,
            binder_chain="B",
            receptor_chain="A",
            options={"use_msa_server": False},
            extra_files={"receptor_msa": wrong},
        )
        with pytest.raises(ValueError, match="is not the receptor sequence"):
            Boltz2Runner().prepare(request, tmp_path / "work")
        assert not (tmp_path / "work").exists()


def _twin_chains():
    """A structure whose chains A and B both hold three glycines (backbone atoms only)."""
    rows = []
    serial = 1
    for chain_number, chain in enumerate("AB"):
        for number in (1, 2, 3):
            for atom, offset in (("N", 0.0), ("CA", 1.46), ("C", 2.5)):
                rows.append(
                    f"ATOM  {serial:5d} {atom:^4s} GLY {chain}{number:4d}    "
                    f"{3.8 * number + offset:8.3f}{10.0 * chain_number:8.3f}{0.0:8.3f}"
                    f"  1.00 20.00           {atom[0]:>2s}"
                )
                serial += 1
    return "\n".join(rows) + "\nEND\n"


class TestTheCommand:
    def command(self, tmp_path, runner=None, **keywords):
        runner = runner or Boltz2Runner()
        request = make_request(**keywords)
        return runner._command(request, tmp_path / "input" / "ycr.yaml", tmp_path)

    def test_under_the_policy_the_command_has_no_server_flag(self, tmp_path, alignment):
        for receptor_msa in (alignment, "single"):
            command = self.command(tmp_path, msa_policy="forbid-server", receptor_msa=receptor_msa)
            assert not [word for word in command if word.startswith("--use_msa")]

    def test_the_flag_follows_the_option_without_the_policy(self, tmp_path):
        assert "--use_msa_server" in self.command(tmp_path, use_msa_server=True)
        assert "--use_msa_server" not in self.command(tmp_path)

    def test_the_command_is_the_one_it_was_when_the_new_fields_are_absent(
        self, tmp_path, alignment
    ):
        plain = self.command(tmp_path)
        assert self.command(tmp_path, receptor_msa=alignment) == plain
        assert self.command(tmp_path, msa_policy="forbid-server", receptor_msa="single") == plain

    @pytest.mark.parametrize("extra", [["--use_msa_server"], ["--use_msa_server=true"]])
    def test_extra_arguments_cannot_bring_the_flag_back(self, extra):
        with pytest.raises(ValueError, match="repeats --use_msa_server"):
            make_request(msa_policy="forbid-server", receptor_msa="single", extra_args=extra)

    @pytest.mark.parametrize("step", ["prepare", "run"])
    def test_a_hand_made_request_that_asks_for_the_server_is_refused_before_anything(
        self, tmp_path, fake_boltz, step
    ):
        request = hand_made(msa_policy="forbid-server", use_msa_server=True, receptor_msa="single")
        with pytest.raises(ValueError, match="would be sent to an MSA server"):
            getattr(Boltz2Runner(), step)(request, tmp_path / "work")
        assert not (tmp_path / "work").exists() and fake_boltz.calls() == []

    @pytest.mark.parametrize("step", ["prepare", "run"])
    def test_a_hand_made_request_without_a_source_is_refused(self, tmp_path, fake_boltz, step):
        request = hand_made(msa_policy="forbid-server")
        with pytest.raises(ValueError, match="needs a source for the MSA of the receptor"):
            getattr(Boltz2Runner(), step)(request, tmp_path / "work")
        assert not (tmp_path / "work").exists() and fake_boltz.calls() == []

    @pytest.mark.parametrize("step", ["prepare", "run"])
    def test_a_command_with_a_server_flag_is_stopped_by_the_last_check(
        self, tmp_path, fake_boltz, monkeypatch, step
    ):
        original = Boltz2Runner._command

        def with_the_flag(self, request, yaml_path, work_dir):
            return [*original(self, request, yaml_path, work_dir), "--use_msa_server"]

        monkeypatch.setattr(Boltz2Runner, "_command", with_the_flag)
        request = make_request(msa_policy="forbid-server", receptor_msa="single")
        with pytest.raises(ValueError, match=r"would make the model use an MSA server"):
            getattr(Boltz2Runner(), step)(request, tmp_path / "work")
        assert not (tmp_path / "work").exists() and fake_boltz.calls() == []

    def test_the_conda_prefix_does_not_hide_a_flag_from_the_check(self, tmp_path, monkeypatch):
        runner = Boltz2Runner("boltz")
        request = make_request(msa_policy="forbid-server", receptor_msa="single")
        command = runner._command(request, tmp_path / "input" / "ycr.yaml", tmp_path)
        assert command[:5] == ["conda", "run", "-n", "boltz", "--no-capture-output"]
        assert "--use_msa_server" not in command


class TestRunningTheStub:
    def test_the_process_reads_the_staged_alignment_and_no_server_flag_is_passed(
        self, tmp_path, fake_boltz, alignment
    ):
        fake_boltz.canned("ycr")
        work = tmp_path / "work"
        work.mkdir()
        request = make_request(msa_policy="forbid-server", receptor_msa=alignment)
        Boltz2Runner().run(request, work)
        (call,) = fake_boltz.calls()
        assert not [word for word in call["argv"] if word.startswith("--use_msa")]
        assert call["msa"] == {"receptor.a3m": alignment.read_text(encoding="utf-8")}
        staged = work.absolute() / "input" / "msa" / "receptor.a3m"
        assert f"      msa: {quoted(staged)}\n" in call["yaml"]
        assert call["yaml"].count('      msa: "empty"\n') == 1

    def test_a_run_without_the_new_fields_passes_the_server_flag_as_before(
        self, tmp_path, fake_boltz
    ):
        fake_boltz.canned("ycr")
        Boltz2Runner().run(make_request(use_msa_server=True), tmp_path / "work")
        (call,) = fake_boltz.calls()
        assert "--use_msa_server" in call["argv"] and call["msa"] == {}

    def test_one_run_serves_the_same_alignment_at_another_path(
        self, tmp_path, fake_boltz, alignment
    ):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        moved = tmp_path / "elsewhere" / "copy.a3m"
        moved.parent.mkdir()
        shutil.copy(alignment, moved)
        first = session.record(make_request(msa_policy="forbid-server", receptor_msa=alignment))
        session.record(make_request(msa_policy="forbid-server", receptor_msa=moved))
        assert len(fake_boltz.calls()) == 1
        assert first.provenance["msa_policy"] == "forbid-server"
        assert first.provenance["msa_binder"] == "single"
        assert len(first.provenance["msa_receptor"]) == 12 and first.reasons == []

    def test_another_alignment_is_another_run(self, tmp_path, fake_boltz, alignment):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        other = tmp_path / "other.a3m"
        other.write_text(alignment.read_text(encoding="utf-8") + f">z\n{YCR_A}\n", encoding="utf-8")
        session.record(make_request(receptor_msa=alignment))
        session.record(make_request(receptor_msa=other))
        assert len(fake_boltz.calls()) == 2

    def test_the_stored_input_keeps_the_alignment_beside_the_yaml(
        self, tmp_path, fake_boltz, alignment
    ):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        request = make_request(receptor_msa=alignment)
        session.record(request)
        outputs = session.entry(request).prediction_dir.parents[2]
        assert (outputs / "input" / "msa" / "receptor.a3m").read_bytes() == alignment.read_bytes()


@pytest.fixture(scope="module")
def source():
    from tests.test_pre_structures import model_source_text

    return {
        "main": model_source_text("boltz", "src/boltz/main.py"),
        "schema": model_source_text("boltz", "src/boltz/data/parse/schema.py"),
    }


class TestTheSourceBehindIt:
    """What Boltz-2 does with these YAMLs, read from its source (a clone; v2.2.1)."""

    def test_an_msa_is_a_path_or_the_word_empty(self, source):
        text = source["schema"]
        assert 'msa = items[0][entity_type].get("msa", 0)' in text
        assert 'if msa == "empty":' in text and "msa = -1" in text

    def test_entities_with_one_sequence_share_one_msa(self, source):
        assert "All proteins with the same sequence must share the same MSA!" in source["schema"]

    def test_a_custom_msa_cannot_be_mixed_with_an_automatic_one(self, source):
        """The reason the binder is written as ``empty`` next to a given receptor MSA."""
        assert "Cannot mix custom and auto-generated MSAs in the same input!" in source["schema"]

    def test_only_a3m_and_csv_are_read(self, source):
        text = source["main"]
        assert 'msa_path.suffix == ".a3m"' in text and 'msa_path.suffix == ".csv"' in text
        assert "only a3m or csv" in text

    def test_an_input_without_msa_and_without_the_flag_fails_before_any_server_call(self, source):
        text = source["main"]
        assert "if to_generate and not use_msa_server:" in text
        assert "Missing MSA's in input and --use_msa_server flag not set." in text
        call = text.index("compute_msa(\n                data=to_generate")
        assert text.index("if to_generate and not use_msa_server:") < call

    def test_the_server_is_called_for_the_entities_without_an_msa_only(self, source):
        text = source["main"]
        assert "if (chain.mol_type == prot_id) and (chain.msa_id == 0):" in text
        assert "to_generate[msa_id] = target.sequences[entity_id]" in text
