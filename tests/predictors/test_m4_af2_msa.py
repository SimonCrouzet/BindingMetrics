"""ColabFold and the MSA policy: ``msa_policy`` and ``receptor_msa``.

ColabFold can run without its MSA server only in the single-sequence mode, which is global to the
job, so the runner takes ``receptor_msa="single"`` and refuses the path of an alignment. No
ColabFold, GPU or network is used: the process is a stub ``colabfold_batch`` on PATH that records
its command line and copies synthetic output (``synth_af2``).
"""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from binding_metrics.predictors import af2_runner  # noqa: E402
from binding_metrics.predictors.af2_runner import ColabFoldRunner  # noqa: E402
from binding_metrics.predictors.store import PredictionRequest, PredictionStore  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"

#: Keys of the requests of 1YCR as they were before the MSA arguments existed (ColabFold 1.6.3).
OLD_KEYS = {
    "default": "63ddb1f30d5907aa2f4fb14d3556487c2bba9ea203a1ebb7acc23d83a6762b50",
    "no server": "90055df74ec8bb56da77e3f860100604019ea4bf5a5ce3fe3754029eaf55a2bb",
    "two seeds": "415fb60014dcd6f0ed90e7bd1505126a34409d5ddd98fe205a0d646a86f938d1",
}

STUB_SCRIPT = """#!{python}
import json, pathlib, shutil, sys
argv = sys.argv[1:]
here = pathlib.Path(__file__).resolve().parent
fasta, results = pathlib.Path(argv[0]), pathlib.Path(argv[1])
with open(here / "calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": argv, "fasta": fasta.read_text(encoding="utf-8")}}) + "\\n")
results.mkdir(parents=True, exist_ok=True)
for source in (here / "synthetic").iterdir():
    shutil.copy2(source, results / source.name)
"""


@pytest.fixture(autouse=True)
def colabfold_version(monkeypatch):
    """ColabFold 1.6.3 on every machine."""
    monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.6.3")


@pytest.fixture
def stub_colabfold(tmp_path, monkeypatch):
    """A ``colabfold_batch`` on PATH that copies a synthetic job; returns its log of calls."""
    from tests.predictors import synth, synth_af2

    bin_dir = tmp_path / "bin"
    (bin_dir / "synthetic").mkdir(parents=True)
    for sample in (1, 2):
        synth_af2.write_colabfold(
            bin_dir / "synthetic", "p53", synth.synthetic_complex(), sample=sample
        )
    script = bin_dir / "colabfold_batch"
    script.write_text(STUB_SCRIPT.format(python=sys.executable), encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def calls():
        record = bin_dir / "calls.jsonl"
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]

    return calls


def make(runner=None, **kwargs):
    runner = runner or ColabFoldRunner()
    kwargs.setdefault("name", "p53")
    kwargs.setdefault("binder_chain", "B")
    kwargs.setdefault("receptor_chain", "A")
    return runner.make_request(P53, **kwargs)


def command_of(request, runner=None):
    runner = runner or ColabFoldRunner()
    return runner._command(request, Path("q.fasta"), Path("out"))


def msa_mode(command):
    return command[command.index("--msa-mode") + 1]


# ---------------------------------------------------------------------------- old requests


class TestARequestWithoutTheNewFieldsIsUnchanged:
    @pytest.mark.parametrize(
        "label, kwargs",
        [
            ("default", {}),
            ("no server", {"use_msa_server": False}),
            ("two seeds", {"seeds": (0, 1), "num_samples": 2}),
        ],
    )
    def test_the_key_is_the_one_it_was(self, label, kwargs):
        assert make(**kwargs).key() == OLD_KEYS[label]

    def test_the_options_hold_nothing_of_the_policy(self):
        request = make()
        assert "msa_policy" not in request.options and "receptor_msa" not in request.options
        assert dict(request.extra_files) == {}

    def test_the_default_arguments_spelled_out_change_nothing(self):
        explicit = make(msa_policy="default", receptor_msa=None)
        assert explicit == make() and explicit.key() == make().key()

    def test_the_server_stays_the_default_and_single_sequence_the_choice_to_leave_it(self):
        assert msa_mode(command_of(make())) == "mmseqs2_uniref_env"
        assert msa_mode(command_of(make(use_msa_server=False))) == "single_sequence"


# ---------------------------------------------------------------------------- the request


class TestTheWordSingle:
    def test_it_is_accepted_and_is_an_option_of_the_request(self):
        request = make(receptor_msa="single")
        assert request.options["receptor_msa"] == "single"
        assert dict(request.extra_files) == {}

    def test_it_maps_to_the_single_sequence_mode(self):
        assert msa_mode(command_of(make(receptor_msa="single"))) == "single_sequence"

    def test_it_means_no_server_so_the_request_records_that(self):
        request = make(receptor_msa="single")  # use_msa_server is True by default
        assert request.options["use_msa_server"] is False
        assert request.key() == make(use_msa_server=False, receptor_msa="single").key()

    def test_it_is_in_the_key(self):
        assert make(receptor_msa="single").key() != make(use_msa_server=False).key()
        assert make(receptor_msa="single").key() != make().key()

    def test_a_request_made_by_hand_with_the_server_flag_and_single_runs_single_sequence(self):
        """The safe side: the word asks for no alignment, whatever the server option says."""
        runner = ColabFoldRunner()
        good = make(runner, receptor_msa="single")
        request = PredictionRequest(
            "af2",
            "p53",
            mode="predict",
            binder_chain="B",
            receptor_chain="A",
            sequences=dict(good.sequences),
            seeds=(0,),
            options={**good.options, "use_msa_server": True},
        )
        assert msa_mode(command_of(request, runner)) == "single_sequence"


class TestAnAlignmentFileIsRefused:
    def test_a_path_is_refused_with_the_reason_and_what_is_offered(self, tmp_path):
        alignment = tmp_path / "mdm2.a3m"
        alignment.write_text(">query\nETLV\n", encoding="utf-8")
        with pytest.raises(ValueError) as caught:
            make(use_msa_server=False, receptor_msa=alignment)
        message = str(caught.value)
        assert message.startswith("af2 cannot be given the alignment of the receptor (")
        assert "FASTA file" in message and "single_sequence" in message
        assert message.endswith("it takes receptor_msa='single' only")

    def test_a_string_path_is_refused_the_same_way(self, tmp_path):
        alignment = tmp_path / "mdm2.a3m"
        alignment.write_text(">query\nETLV\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            make(receptor_msa=str(alignment))

    def test_a_file_that_is_named_single_is_still_a_file(self, tmp_path):
        named = tmp_path / "single"
        named.write_text(">query\nETLV\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            make(use_msa_server=False, receptor_msa=named)

    def test_the_refusal_comes_before_a_request_exists(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            make(receptor_msa=tmp_path / "does_not_exist.a3m")

    def test_a_request_made_by_hand_with_a_file_is_refused_by_prepare_and_run(
        self, tmp_path, stub_colabfold
    ):
        alignment = tmp_path / "mdm2.a3m"
        alignment.write_text(">query\nETLV\n", encoding="utf-8")
        runner = ColabFoldRunner()
        good = make(runner, use_msa_server=False)
        request = PredictionRequest(
            "af2",
            "p53",
            mode="predict",
            binder_chain="B",
            receptor_chain="A",
            sequences=dict(good.sequences),
            seeds=(0,),
            options=dict(good.options),
            extra_files={"receptor_msa": alignment},
        )
        for step in (runner.prepare, runner.run):
            with pytest.raises(ValueError, match="cannot be given the alignment"):
                step(request, tmp_path / "work")
        assert stub_colabfold() == []
        assert not (tmp_path / "work").exists()


class TestForbidServer:
    def test_single_without_the_server_is_accepted(self):
        request = make(msa_policy="forbid-server", use_msa_server=False, receptor_msa="single")
        assert request.options["msa_policy"] == "forbid-server"
        assert request.options["receptor_msa"] == "single"
        assert request.options["use_msa_server"] is False
        assert msa_mode(command_of(request)) == "single_sequence"

    def test_the_server_on_is_refused(self):
        with pytest.raises(ValueError, match="would be sent to an MSA server"):
            make(msa_policy="forbid-server", use_msa_server=True, receptor_msa="single")
        with pytest.raises(ValueError, match="would be sent to an MSA server"):
            make(msa_policy="forbid-server")

    def test_no_source_for_the_receptor_is_refused_and_single_is_suggested(self):
        with pytest.raises(ValueError, match="receptor_msa='single'"):
            make(msa_policy="forbid-server", use_msa_server=False)

    def test_a_file_is_refused_under_the_policy_too(self, tmp_path):
        alignment = tmp_path / "mdm2.a3m"
        alignment.write_text(">query\nETLV\n", encoding="utf-8")
        with pytest.raises(ValueError, match="takes receptor_msa='single' only"):
            make(msa_policy="forbid-server", use_msa_server=False, receptor_msa=alignment)

    def test_the_policy_is_in_the_key_only_when_it_forbids(self):
        plain = make(use_msa_server=False, receptor_msa="single")
        forbidding = make(use_msa_server=False, receptor_msa="single", msa_policy="forbid-server")
        assert forbidding.key() != plain.key()
        assert "msa_policy" not in plain.options

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="msa_policy must be one of"):
            make(msa_policy="never")

    def test_extra_args_cannot_bring_the_server_mode_back(self):
        for argument in ("--msa-mode=mmseqs2_uniref_env", "--msa-mode", "--msa=x"):
            with pytest.raises(ValueError, match="extra_args may not hold"):
                make(
                    msa_policy="forbid-server",
                    use_msa_server=False,
                    receptor_msa="single",
                    extra_args=[argument],
                )

    def test_a_request_made_by_hand_is_refused_by_prepare_and_run(self, tmp_path, stub_colabfold):
        runner = ColabFoldRunner()
        good = make(runner, use_msa_server=False, receptor_msa="single")
        hand_made = [
            {"msa_policy": "forbid-server", "use_msa_server": True, "receptor_msa": "single"},
            {"msa_policy": "forbid-server", "use_msa_server": False},
            {"msa_policy": "forbid-server"},  # the server is on when the option is absent
        ]
        base = {
            key: value
            for key, value in good.options.items()
            if key not in ("use_msa_server", "receptor_msa", "msa_policy")
        }
        for policy_options in hand_made:
            merged = {**base, **policy_options}
            request = PredictionRequest(
                "af2",
                "p53",
                mode="predict",
                binder_chain="B",
                receptor_chain="A",
                sequences=dict(good.sequences),
                seeds=(0,),
                options=merged,
            )
            for step in (runner.prepare, runner.run):
                with pytest.raises(ValueError, match="MSA server|msa_policy"):
                    step(request, tmp_path / "work")
        assert stub_colabfold() == []
        assert not (tmp_path / "work").exists()


class TestTheCommandIsChecked:
    """The last check before a process starts looks at the command line that was built."""

    @pytest.fixture
    def forbidding(self):
        return make(
            msa_policy="forbid-server", use_msa_server=False, receptor_msa="single", num_samples=2
        )

    @pytest.mark.parametrize("mode", ["mmseqs2_uniref_env", "mmseqs2_uniref"])
    def test_a_server_mode_in_the_command_is_refused_before_a_process_starts(
        self, tmp_path, monkeypatch, stub_colabfold, forbidding, mode
    ):
        real = ColabFoldRunner._command

        def with_server_mode(self, request, fasta, results):
            command = real(self, request, fasta, results)
            command[command.index("--msa-mode") + 1] = mode
            return command

        monkeypatch.setattr(ColabFoldRunner, "_command", with_server_mode)
        for step in (ColabFoldRunner().prepare, ColabFoldRunner().run):
            with pytest.raises(ValueError, match=rf"--msa-mode {mode}.*nothing was started"):
                step(forbidding, tmp_path / "work")
        assert stub_colabfold() == []
        assert not (tmp_path / "work").exists()

    def test_a_command_without_the_mode_is_refused_because_the_default_calls_the_server(
        self, tmp_path, monkeypatch, stub_colabfold, forbidding
    ):
        real = ColabFoldRunner._command

        def without_mode(self, request, fasta, results):
            command = real(self, request, fasta, results)
            index = command.index("--msa-mode")
            return command[:index] + command[index + 2 :]

        monkeypatch.setattr(ColabFoldRunner, "_command", without_mode)
        with pytest.raises(ValueError, match="no --msa-mode"):
            ColabFoldRunner().run(forbidding, tmp_path / "work")
        assert stub_colabfold() == []

    def test_the_default_policy_checks_nothing(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            af2_runner, "assert_no_server_flag", lambda *args, **kwargs: calls.append(args)
        )
        runner = ColabFoldRunner()
        runner.prepare(make(runner), tmp_path / "a")
        runner.prepare(make(runner, receptor_msa="single"), tmp_path / "b")
        assert calls == []

    def test_the_policy_checks_the_command_in_prepare_and_in_run(
        self, tmp_path, monkeypatch, stub_colabfold, forbidding
    ):
        calls = []
        real = af2_runner.assert_no_server_flag

        def record(command, name, **kwargs):
            calls.append((list(command), name))
            return real(command, name, **kwargs)

        monkeypatch.setattr(af2_runner, "assert_no_server_flag", record)
        runner = ColabFoldRunner()
        runner.prepare(forbidding, tmp_path / "work")
        assert len(calls) == 1 and calls[0][1] == "af2"
        assert msa_mode(calls[0][0]) == "single_sequence"
        runner.run(forbidding, tmp_path / "work2")
        assert len(calls) == 3  # prepare and the command of run, once more

    def test_a_conda_command_passes_the_check(self, tmp_path):
        runner = ColabFoldRunner("fold")
        request = make(
            runner, msa_policy="forbid-server", use_msa_server=False, receptor_msa="single"
        )
        assert runner.prepare(request, tmp_path / "work").is_file()


# ---------------------------------------------------------------------------- through the store


class TestRunThroughTheStore:
    def test_a_forbidden_server_run_starts_with_the_single_sequence_mode(
        self, tmp_path, stub_colabfold
    ):
        runner = ColabFoldRunner()
        store = PredictionStore(tmp_path / "store")
        request = make(
            runner,
            msa_policy="forbid-server",
            use_msa_server=False,
            receptor_msa="single",
            num_samples=2,
        )
        entry = store.get_or_run(request, runner)
        assert entry.status == "done"
        (call,) = stub_colabfold()
        assert msa_mode(call["argv"]) == "single_sequence"
        assert not any(word.startswith("mmseqs2") for word in call["argv"])

    def test_the_word_single_runs_the_same_command_as_the_server_off(
        self, tmp_path, stub_colabfold
    ):
        runner = ColabFoldRunner()
        store = PredictionStore(tmp_path / "store")
        store.get_or_run(make(runner, receptor_msa="single", num_samples=2), runner)
        store.get_or_run(make(runner, use_msa_server=False, num_samples=2), runner)
        first, second = stub_colabfold()
        assert first["argv"][2:] == second["argv"][2:]

    def test_a_default_run_still_asks_for_the_server_mode(self, tmp_path, stub_colabfold):
        runner = ColabFoldRunner()
        PredictionStore(tmp_path / "store").get_or_run(make(runner, num_samples=2), runner)
        (call,) = stub_colabfold()
        assert msa_mode(call["argv"]) == "mmseqs2_uniref_env"
