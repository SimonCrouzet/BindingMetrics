"""Protenix and the MSA policy: ``msa_policy``, ``receptor_msa`` and the alignment files.

No Protenix, GPU or network is used. The process is a stub ``protenix`` executable on PATH that
records its command line and, for every ``proteinChain`` of its input, whether the alignment file
the chain names exists while the process runs and what it holds. These tests check what the
runner writes and refuses; they do not show that Protenix accepts the files, which no run has
confirmed (see the module docstring of ``predictors/protenix_runner.py``, "MSA").
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from binding_metrics.predictors import protenix_runner  # noqa: E402
from binding_metrics.predictors.protenix_runner import ProtenixRunner  # noqa: E402
from binding_metrics.predictors.store import (  # noqa: E402
    PredictionRequest,
    PredictionStore,
    file_sha256,
)

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"

MDM2 = "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
P53_PEPTIDE = "ETFSDLWKLLPEN"

#: The key and the input JSON of the three bundled cases as they were written before the MSA
#: arguments existed (``make_request`` of the bundled structure, Protenix 2.0.0, no weights).
OLD_KEYS = {
    "p53": "d1dd6a57856b339b40e783d515b149b3eb86d82511960a1eed21723d8c8bbcf8",
    "sfti": "f0e75b366b475a8d5d50347169490d3f4566018e2f39b6fa9aa242fe545c89ce",
    "cwa": "1f745f7824caf58f0e3ffe7cc8474ef5d785b0bd9ee9e58adc1b50251c938c9e",
}
OLD_JSON_SHA256 = {
    "p53": "b70c43b338150c4995340dfa47ae87e0c702687494a08d66cb762efb036f6efd",
    "sfti": "cf2f1abd8dbd7b325a35a36f1ec06beed1069c67611e9b4e9692738b31a26314",
    "cwa": "9e4cd7edaa9686d6ec415b91df4baa4569406d65af5ed6a25fc98874cdade5b0",
}
CASES = {
    "p53": (P53, "B", "A"),
    "sfti": (SFTI1, "I", "A"),
    "cwa": (CYCLOSPORIN, "C", "A"),
}

STUB_SCRIPT = """#!{python}
import json, os, pathlib, shutil, sys
argv = sys.argv[1:]
args = {{}}
for index, token in enumerate(argv):
    if token.startswith("--") and index + 1 < len(argv):
        args[token[2:]] = argv[index + 1]
job = json.loads(pathlib.Path(args["input"]).read_text(encoding="utf-8"))
files = []
for entry in job[0]["sequences"]:
    chain = entry["proteinChain"]
    path = chain.get("unpairedMsaPath")
    seen = {{"id": chain["id"], "path": path, "exists": bool(path) and os.path.isfile(path)}}
    if seen["exists"]:
        seen["text"] = pathlib.Path(path).read_text(encoding="utf-8")
    files.append(seen)
with open({record!r}, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": argv, "job": job, "files": files}}) + "\\n")
name = job[0]["name"]
out = pathlib.Path(args["out_dir"])
template = pathlib.Path({template!r}) / name / "seed_9" / "predictions"
for seed in [int(s) for s in args["seeds"].split(",")]:
    target = out / name / ("seed_%d" % seed) / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for rank in range(int(args["sample"])):
        for path in template.glob("*_sample_%d.*" % rank):
            shutil.copy2(path, target / path.name)
"""


@pytest.fixture(autouse=True)
def protenix_environment(tmp_path, monkeypatch):
    """Protenix 2.0.0 and no weights on every machine."""
    monkeypatch.setenv("PROTENIX_ROOT_DIR", str(tmp_path / "protenix_root"))
    monkeypatch.setattr(protenix_runner, "_installed_version", lambda python_cmd=None: "2.0.0")


@pytest.fixture
def stub_protenix(tmp_path, monkeypatch):
    """A ``protenix`` on PATH that writes a synthetic output; returns its log of calls."""
    from tests.predictors import synth, synth_protenix

    template = tmp_path / "template_output"
    for sample in (1, 2):
        synth_protenix.write_prediction(template, "p53", synth.synthetic_complex(), sample=sample)
    record = tmp_path / "calls.jsonl"
    script = tmp_path / "bin" / "protenix"
    script.parent.mkdir()
    script.write_text(
        STUB_SCRIPT.format(python=sys.executable, record=str(record), template=str(template)),
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")

    def calls():
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]

    return calls


def a3m_text(sequence=MDM2, homologs=2, insertions=False):
    """A receptor alignment: the query first, then ``homologs`` rows with a gap or an insertion."""
    rows = [(">query", sequence)]
    for index in range(homologs):
        row = list(sequence)
        row[3 + index] = "-"
        text = "".join(row)
        if insertions:
            text = text[:10] + "acg" + text[10:]
        rows.append((f">UniRef100_X{index}_9606/1-{len(sequence)}", text))
    return "".join(f"{header}\n{sequence}\n" for header, sequence in rows)


@pytest.fixture
def alignment(tmp_path):
    path = tmp_path / "mdm2.a3m"
    path.write_text(a3m_text(), encoding="utf-8")
    return path


def make(runner=None, path=P53, **kwargs):
    runner = runner or ProtenixRunner()
    kwargs.setdefault("name", "p53")
    kwargs.setdefault("binder_chain", "B")
    kwargs.setdefault("receptor_chain", "A")
    return runner.make_request(path, **kwargs)


def job_of(path):
    (job,) = json.loads(Path(path).read_text(encoding="utf-8"))
    return job


def chains_of(job):
    return [entry["proteinChain"] for entry in job["sequences"]]


# ---------------------------------------------------------------------------- old requests


class TestARequestWithoutTheNewFieldsIsUnchanged:
    @pytest.mark.parametrize("case", list(CASES))
    def test_the_key_is_the_one_it_was(self, case):
        path, binder, receptor = CASES[case]
        request = make(path=path, name=case, binder_chain=binder, receptor_chain=receptor)
        assert request.key() == OLD_KEYS[case]

    @pytest.mark.parametrize("case", list(CASES))
    def test_the_input_json_is_byte_for_byte_what_it_was(self, tmp_path, case):
        path, binder, receptor = CASES[case]
        runner = ProtenixRunner()
        request = make(runner, path, name=case, binder_chain=binder, receptor_chain=receptor)
        written = runner.prepare(request, tmp_path / "work")
        assert hashlib.sha256(written.read_bytes()).hexdigest() == OLD_JSON_SHA256[case]
        assert not (written.parent / "msa").exists()

    def test_the_options_and_extra_files_hold_nothing_of_the_policy(self):
        request = make()
        assert "msa_policy" not in request.options and "receptor_msa" not in request.options
        assert dict(request.extra_files) == {}

    def test_the_default_arguments_spelled_out_change_nothing(self):
        explicit = make(msa_policy="default", receptor_msa=None)
        assert explicit.key() == make().key() and explicit == make()

    def test_the_server_stays_on_by_default_and_the_command_asks_for_the_search(self):
        runner = ProtenixRunner()
        command = runner._command(make(runner), Path("in.json"), Path("out"))
        assert command[command.index("--use_msa") + 1] == "true"
        assert make().options["use_msa_server"] is True


# ---------------------------------------------------------------------------- the request


class TestTheRequest:
    def test_a_file_is_an_extra_file_hashed_by_content_and_not_an_option(self, alignment):
        request = make(use_msa_server=False, receptor_msa=alignment)
        assert request.extra_files["receptor_msa"] == alignment
        assert request.content_hashes["receptor_msa"] == file_sha256(alignment)
        assert "receptor_msa" not in request.options and "msa_policy" not in request.options

    def test_the_word_single_is_an_option_and_adds_no_file(self):
        request = make(use_msa_server=False, receptor_msa="single")
        assert request.options["receptor_msa"] == "single"
        assert dict(request.extra_files) == {}

    def test_the_same_alignment_at_another_path_is_the_same_request(self, alignment, tmp_path):
        moved = tmp_path / "elsewhere.a3m"
        moved.write_bytes(alignment.read_bytes())
        first = make(use_msa_server=False, receptor_msa=alignment)
        assert make(use_msa_server=False, receptor_msa=moved).key() == first.key()

    def test_another_alignment_is_another_request(self, alignment, tmp_path):
        other = tmp_path / "other.a3m"
        other.write_text(a3m_text(homologs=3), encoding="utf-8")
        keys = {
            make(use_msa_server=False, receptor_msa=alignment).key(),
            make(use_msa_server=False, receptor_msa=other).key(),
            make(use_msa_server=False, receptor_msa="single").key(),
            make(use_msa_server=False).key(),
        }
        assert len(keys) == 4

    def test_the_policy_is_in_the_key_only_when_it_forbids(self):
        plain = make(use_msa_server=False, receptor_msa="single")
        forbidding = make(use_msa_server=False, receptor_msa="single", msa_policy="forbid-server")
        assert forbidding.options["msa_policy"] == "forbid-server"
        assert forbidding.key() != plain.key()

    def test_a_receptor_msa_means_no_server_and_one_key_stands_for_the_run(self, alignment):
        asked_with_server = make(receptor_msa=alignment)  # use_msa_server is True by default
        assert asked_with_server.options["use_msa_server"] is False
        assert asked_with_server.key() == make(use_msa_server=False, receptor_msa=alignment).key()
        assert make(receptor_msa="single").options["use_msa_server"] is False

    def test_the_search_stays_in_the_key_without_a_receptor_msa(self):
        assert make(use_msa_server=False).options["use_msa_server"] is False
        assert make(use_msa_server=False).key() != make().key()

    def test_the_path_may_be_a_string_and_the_user_directory_is_expanded(self, alignment):
        assert make(use_msa_server=False, receptor_msa=str(alignment)).extra_files[
            "receptor_msa"
        ] == Path(alignment)

    def test_a_path_that_is_a_file_named_single_is_a_file(self, tmp_path):
        named = tmp_path / "single"
        named.write_text(a3m_text(), encoding="utf-8")
        request = make(use_msa_server=False, receptor_msa=named)
        assert request.extra_files["receptor_msa"] == named
        assert "receptor_msa" not in request.options


class TestForbidServer:
    def test_the_server_must_be_off(self, alignment):
        with pytest.raises(ValueError, match="would be sent to an MSA server"):
            make(msa_policy="forbid-server", use_msa_server=True, receptor_msa=alignment)

    def test_the_receptor_needs_a_source(self):
        with pytest.raises(ValueError, match="receptor_msa='single'"):
            make(msa_policy="forbid-server", use_msa_server=False)

    def test_a_file_and_the_word_single_are_both_accepted(self, alignment):
        for source in (alignment, "single"):
            request = make(msa_policy="forbid-server", use_msa_server=False, receptor_msa=source)
            assert request.options["msa_policy"] == "forbid-server"
            assert request.options["use_msa_server"] is False

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="msa_policy must be one of"):
            make(msa_policy="never")

    def test_a_request_made_by_hand_is_refused_by_prepare_and_run(self, tmp_path, stub_protenix):
        runner = ProtenixRunner()
        hand_made = [
            # the server on
            {"msa_policy": "forbid-server", "use_msa_server": True, "receptor_msa": "single"},
            # no source for the receptor
            {"msa_policy": "forbid-server", "use_msa_server": False},
        ]
        for options in hand_made:
            good = make(runner)
            request = PredictionRequest(
                "protenix",
                "p53",
                mode="predict",
                input_path=P53,
                binder_chain="B",
                receptor_chain="A",
                seeds=(101,),
                options={**good.options, **options},
            )
            for step in (runner.prepare, runner.run):
                with pytest.raises(ValueError, match="MSA server|msa_policy"):
                    step(request, tmp_path / "work")
        assert stub_protenix() == []
        assert not (tmp_path / "work").exists()


class TestTheAlignmentIsChecked:
    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="receptor_msa is not a file"):
            make(use_msa_server=False, receptor_msa=tmp_path / "none.a3m")

    def test_a_first_sequence_of_another_length_is_refused_with_both_lengths(self, tmp_path):
        short = tmp_path / "short.a3m"
        short.write_text(a3m_text(MDM2[:-5]), encoding="utf-8")
        with pytest.raises(ValueError, match=r"has 80 aligned columns.*85 residues"):
            make(use_msa_server=False, receptor_msa=short)

    def test_lowercase_letters_are_insertions_and_do_not_count(self, tmp_path):
        inserted = tmp_path / "inserted.a3m"
        text = a3m_text(insertions=True)
        inserted.write_text(
            text.replace(">query\n" + MDM2, ">query\n" + MDM2[:5] + "wy" + MDM2[5:]),
            encoding="utf-8",
        )
        assert make(use_msa_server=False, receptor_msa=inserted).extra_files["receptor_msa"]

    def test_a_gap_in_the_first_sequence_counts_as_a_column(self, tmp_path):
        gapped = tmp_path / "gapped.a3m"
        gapped.write_text(f">query\n{MDM2[:10]}-{MDM2[10:]}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"has 86 aligned columns"):
            make(use_msa_server=False, receptor_msa=gapped)

    @pytest.mark.parametrize("text", ["", "ETLVRP\n", ">only a header\n", "# a comment\n"])
    def test_a_file_without_a_record_is_refused(self, tmp_path, text):
        empty = tmp_path / "empty.a3m"
        empty.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="holds no sequence"):
            make(use_msa_server=False, receptor_msa=empty)

    def test_a_binary_file_is_refused(self, tmp_path):
        binary = tmp_path / "binary.a3m"
        binary.write_bytes(b"\xff\xfe\x00\x01")
        with pytest.raises(ValueError, match="not a UTF-8 text file"):
            make(use_msa_server=False, receptor_msa=binary)

    def test_the_sequence_may_span_lines_as_protenix_reads_it(self, tmp_path):
        wrapped = tmp_path / "wrapped.a3m"
        wrapped.write_text(f">query\n{MDM2[:40]}\n{MDM2[40:]}\n", encoding="utf-8")
        assert make(use_msa_server=False, receptor_msa=wrapped)

    def test_a_different_residue_is_allowed_and_logged(self, tmp_path, caplog):
        mutant = tmp_path / "mutant.a3m"
        mutant.write_text(f">query\nA{MDM2[1:]}\n", encoding="utf-8")
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa=mutant)
        with caplog.at_level(logging.WARNING, logger=protenix_runner.logger.name):
            runner.prepare(request, tmp_path / "work")
        assert any("differs from the receptor at 1 of 85" in r.getMessage() for r in caplog.records)

    def test_the_request_does_not_log_what_prepare_will(self, tmp_path, caplog):
        mutant = tmp_path / "mutant.a3m"
        mutant.write_text(f">query\nA{MDM2[1:]}\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=protenix_runner.logger.name):
            make(use_msa_server=False, receptor_msa=mutant)
        assert caplog.records == []

    def test_prepare_checks_a_request_made_by_hand_against_the_receptor(self, tmp_path):
        short = tmp_path / "short.a3m"
        short.write_text(a3m_text(MDM2[:-5]), encoding="utf-8")
        request = make(use_msa_server=False)
        hand_made = PredictionRequest(
            "protenix",
            "p53",
            mode="predict",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            seeds=(101,),
            options=dict(request.options),
            extra_files={"receptor_msa": short},
        )
        with pytest.raises(ValueError, match="aligned columns"):
            ProtenixRunner().prepare(hand_made, tmp_path / "work")
        assert not (tmp_path / "work" / "input" / "p53.json").exists()


# ---------------------------------------------------------------------------- the input


class TestTheInputJson:
    def test_every_chain_names_an_alignment_file_that_exists(self, alignment, tmp_path):
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa=alignment)
        written = runner.prepare(request, tmp_path / "work")
        receptor, binder = chains_of(job_of(written))
        for chain in (receptor, binder):
            path = Path(chain["unpairedMsaPath"])
            assert path.is_absolute() and path.is_file()
            assert path.parent == written.parent / "msa"
            assert "pairedMsaPath" not in chain

    def test_the_receptor_gets_the_file_and_the_binder_its_own_sequence(self, alignment, tmp_path):
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa=alignment)
        receptor, binder = chains_of(job_of(runner.prepare(request, tmp_path / "work")))
        assert Path(receptor["unpairedMsaPath"]).read_text(encoding="utf-8") == a3m_text()
        assert Path(binder["unpairedMsaPath"]).read_text(encoding="utf-8") == (
            f">query\n{P53_PEPTIDE}\n"
        )

    def test_the_word_single_gives_the_receptor_its_own_sequence_too(self, tmp_path):
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa="single")
        receptor, binder = chains_of(job_of(runner.prepare(request, tmp_path / "work")))
        assert Path(receptor["unpairedMsaPath"]).read_text(encoding="utf-8") == f">query\n{MDM2}\n"
        assert Path(binder["unpairedMsaPath"]).read_text(encoding="utf-8") == (
            f">query\n{P53_PEPTIDE}\n"
        )

    def test_the_rest_of_the_job_is_what_it_is_without_the_files(self, alignment, tmp_path):
        runner = ProtenixRunner()
        plain = job_of(runner.prepare(make(runner, use_msa_server=False), tmp_path / "a"))
        given = job_of(
            runner.prepare(
                make(runner, use_msa_server=False, receptor_msa=alignment), tmp_path / "b"
            )
        )
        for chain in chains_of(given):
            del chain["unpairedMsaPath"]
        assert given == plain

    def test_a_file_without_a_final_newline_is_given_one(self, tmp_path):
        bare = tmp_path / "bare.a3m"
        bare.write_text(f">query\n{MDM2}", encoding="utf-8")
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa=bare)
        receptor, _ = chains_of(job_of(runner.prepare(request, tmp_path / "work")))
        assert Path(receptor["unpairedMsaPath"]).read_bytes() == f">query\n{MDM2}\n".encode()

    def test_the_copy_does_not_depend_on_the_user_file_afterwards(self, alignment, tmp_path):
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa=alignment)
        receptor, _ = chains_of(job_of(runner.prepare(request, tmp_path / "work")))
        alignment.unlink()
        assert Path(receptor["unpairedMsaPath"]).read_text(encoding="utf-8") == a3m_text()

    def test_a_binder_with_modified_residues_gets_the_sequence_the_job_sends(self, tmp_path):
        runner = ProtenixRunner()
        request = make(
            runner,
            CYCLOSPORIN,
            name="cwa",
            binder_chain="C",
            receptor_chain="A",
            use_msa_server=False,
            receptor_msa="single",
        )
        written = runner.prepare(request, tmp_path / "work")
        _, binder = chains_of(job_of(written))
        assert binder["modifications"]  # the D-amino acid and N-methylated residues
        text = Path(binder["unpairedMsaPath"]).read_text(encoding="utf-8")
        assert text == f">query\n{binder['sequence']}\n"

    def test_a_cyclic_binder_keeps_its_bonds_next_to_the_files(self, tmp_path):
        runner = ProtenixRunner()
        request = make(
            runner,
            SFTI1,
            name="sfti",
            binder_chain="I",
            receptor_chain="A",
            use_msa_server=False,
            receptor_msa="single",
        )
        job = job_of(runner.prepare(request, tmp_path / "work"))
        assert len(job["covalent_bonds"]) == 2
        assert all("unpairedMsaPath" in chain for chain in chains_of(job))

    def test_nothing_is_written_when_the_input_is_refused(self, tmp_path):
        runner = ProtenixRunner()
        request = make(runner, use_msa_server=False, receptor_msa="single")
        broken = PredictionRequest(
            "protenix",
            "p53",
            mode="predict",
            input_path=P53,
            binder_chain="Z",  # not in the structure
            receptor_chain="A",
            seeds=(101,),
            options=dict(request.options),
        )
        with pytest.raises(ValueError, match="chain 'Z'"):
            runner.prepare(broken, tmp_path / "work")
        assert not (tmp_path / "work").exists()

    def test_the_job_dict_needs_a_folder_for_the_files(self):
        request = make(use_msa_server=False, receptor_msa="single")
        options = protenix_runner._resolve_options(request.options)
        with pytest.raises(ValueError, match="needs a directory for the MSA files"):
            protenix_runner._job_dict(request, options)


class TestTheCommandLine:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "true"),
            ({"use_msa_server": False}, "false"),
            ({"use_msa_server": False, "receptor_msa": "single"}, "true"),
            ({"receptor_msa": "single"}, "true"),
        ],
    )
    def test_use_msa_is_true_when_the_server_or_the_files_are_read(self, kwargs, expected):
        runner = ProtenixRunner()
        command = runner._command(make(runner, **kwargs), Path("in.json"), Path("out"))
        assert command[command.index("--use_msa") + 1] == expected

    def test_a_file_asks_for_the_same_command_as_the_word_single(self, alignment):
        runner = ProtenixRunner()
        words = [
            runner._command(
                make(runner, use_msa_server=False, receptor_msa=source), Path("i"), Path("o")
            )
            for source in (alignment, "single")
        ]
        assert words[0] == words[1]


# ---------------------------------------------------------------------------- the paths check


class TestChainsHaveMsaPaths:
    def write(self, tmp_path, chains):
        path = tmp_path / "job.json"
        sequences = [
            {"proteinChain": {"sequence": "AC", "count": 1, "id": ["A"], **c}} for c in chains
        ]
        path.write_text(json.dumps([{"name": "x", "sequences": sequences}]), encoding="utf-8")
        return path

    def test_every_chain_with_a_file_is_true(self, tmp_path):
        a, b = tmp_path / "a.a3m", tmp_path / "b.a3m"
        a.write_text(">q\nAC\n", encoding="utf-8")
        b.write_text(">q\nAC\n", encoding="utf-8")
        job = self.write(tmp_path, [{"unpairedMsaPath": str(a)}, {"unpairedMsaPath": str(b)}])
        assert protenix_runner._chains_have_msa_paths(job) is True

    def test_one_chain_without_a_path_is_false(self, tmp_path):
        a = tmp_path / "a.a3m"
        a.write_text(">q\nAC\n", encoding="utf-8")
        job = self.write(tmp_path, [{"unpairedMsaPath": str(a)}, {}])
        assert protenix_runner._chains_have_msa_paths(job) is False

    def test_a_path_that_is_not_a_file_is_false(self, tmp_path):
        job = self.write(tmp_path, [{"unpairedMsaPath": str(tmp_path / "gone.a3m")}])
        assert protenix_runner._chains_have_msa_paths(job) is False

    def test_a_paired_path_alone_counts_and_one_that_is_gone_does_not(self, tmp_path):
        a = tmp_path / "a.a3m"
        a.write_text(">q\nAC\n", encoding="utf-8")
        assert protenix_runner._chains_have_msa_paths(
            self.write(tmp_path, [{"pairedMsaPath": str(a)}])
        )
        assert not protenix_runner._chains_have_msa_paths(
            self.write(
                tmp_path, [{"pairedMsaPath": str(a), "unpairedMsaPath": str(tmp_path / "x")}]
            )
        )

    def test_a_job_without_a_protein_chain_is_false(self, tmp_path):
        path = tmp_path / "job.json"
        path.write_text(json.dumps([{"name": "x", "sequences": []}]), encoding="utf-8")
        assert protenix_runner._chains_have_msa_paths(path) is False


class TestTheCommandIsCheckedUnderTheirPolicy:
    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = []
        real = protenix_runner.assert_no_server_flag

        def record(command, name, *, chains_have_msa_paths=False):
            calls.append((list(command), name, chains_have_msa_paths))
            return real(command, name, chains_have_msa_paths=chains_have_msa_paths)

        monkeypatch.setattr(protenix_runner, "assert_no_server_flag", record)
        return calls

    def test_prepare_checks_the_command_and_says_the_chains_have_paths(self, tmp_path, recorded):
        runner = ProtenixRunner()
        request = make(
            runner, msa_policy="forbid-server", use_msa_server=False, receptor_msa="single"
        )
        runner.prepare(request, tmp_path / "work")
        (command, name, have_paths) = recorded[0]
        assert name == "protenix" and have_paths is True
        assert command[command.index("--use_msa") + 1] == "true"

    def test_the_default_policy_checks_nothing(self, tmp_path, alignment, recorded):
        runner = ProtenixRunner()
        runner.prepare(make(runner, receptor_msa=alignment), tmp_path / "a")
        runner.prepare(make(runner), tmp_path / "b")
        assert recorded == []

    def test_a_chain_left_without_a_path_is_refused_before_a_process_starts(
        self, tmp_path, monkeypatch, stub_protenix
    ):
        runner = ProtenixRunner()
        request = make(
            runner, msa_policy="forbid-server", use_msa_server=False, receptor_msa="single"
        )
        real = protenix_runner._job_dict

        def without_binder_path(request, options, msa_directory=None):
            job = real(request, options, msa_directory)
            del job["sequences"][1]["proteinChain"]["unpairedMsaPath"]
            return job

        monkeypatch.setattr(protenix_runner, "_job_dict", without_binder_path)
        with pytest.raises(ValueError, match=r"--use_msa true.*nothing was started"):
            runner.run(request, tmp_path / "work")
        assert stub_protenix() == []

    def test_a_server_flag_in_the_command_is_refused_whatever_the_request_says(
        self, tmp_path, monkeypatch, stub_protenix
    ):
        runner = ProtenixRunner()
        request = make(
            runner, msa_policy="forbid-server", use_msa_server=False, receptor_msa="single"
        )
        real = ProtenixRunner._command

        def with_server(self, request, job_path, output_dir):
            command = real(self, request, job_path, output_dir)
            command[command.index("--use_msa") + 1] = "true"
            return [*command, "--use_msa_server"]

        monkeypatch.setattr(ProtenixRunner, "_command", with_server)
        with pytest.raises(ValueError, match="--use_msa_server"):
            runner.run(request, tmp_path / "work")
        assert stub_protenix() == []

    def test_a_conda_environment_command_passes_the_check(self, tmp_path):
        runner = ProtenixRunner("fold")
        request = make(
            runner, msa_policy="forbid-server", use_msa_server=False, receptor_msa="single"
        )
        written = runner.prepare(request, tmp_path / "work")
        assert written.is_file()


# ---------------------------------------------------------------------------- through the store


class TestRunThroughTheStore:
    def test_the_process_reads_a_file_for_each_chain_and_no_chain_is_left_out(
        self, tmp_path, alignment, stub_protenix
    ):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(
            runner,
            msa_policy="forbid-server",
            use_msa_server=False,
            receptor_msa=alignment,
            seeds=(3, 4),
            num_samples=2,
        )
        entry = store.get_or_run(request, runner)
        assert entry.status == "done"
        (call,) = stub_protenix()
        argv = call["argv"]
        assert argv[argv.index("--use_msa") + 1] == "true"
        assert "--use_msa_server" not in argv
        receptor, binder = call["files"]
        assert receptor["exists"] and binder["exists"]
        assert receptor["text"] == a3m_text()
        assert binder["text"] == f">query\n{P53_PEPTIDE}\n"

    def test_a_default_run_without_a_receptor_msa_writes_no_path(self, tmp_path, stub_protenix):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        store.get_or_run(make(runner, num_samples=2), runner)
        (call,) = stub_protenix()
        assert [chain["path"] for chain in call["files"]] == [None, None]
        assert call["argv"][call["argv"].index("--use_msa") + 1] == "true"

    def test_a_run_with_the_search_off_and_no_file_keeps_use_msa_false(
        self, tmp_path, stub_protenix
    ):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        store.get_or_run(make(runner, use_msa_server=False, num_samples=2), runner)
        (call,) = stub_protenix()
        assert call["argv"][call["argv"].index("--use_msa") + 1] == "false"
        assert [chain["path"] for chain in call["files"]] == [None, None]

    def test_the_same_alignment_at_another_path_is_a_hit_and_starts_nothing(
        self, tmp_path, alignment, stub_protenix
    ):
        runner = ProtenixRunner()
        store = PredictionStore(tmp_path / "store")
        moved = tmp_path / "moved.a3m"
        moved.write_bytes(alignment.read_bytes())
        for source in (alignment, moved):
            request = make(runner, use_msa_server=False, receptor_msa=source, num_samples=2)
            store.get_or_run(request, runner)
        assert len(stub_protenix()) == 1
