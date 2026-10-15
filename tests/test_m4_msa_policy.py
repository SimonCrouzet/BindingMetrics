"""The MSA policy of a prediction request: a given receptor MSA and no MSA server.

``binding_metrics.predictors._msa`` holds the pieces the runners share, and they are tested here
with stub runners that build a command line the way the four real runners do. The pre-flight
check refuses a run that would call a server under the policy ``forbid-server``, and refuses it
under every ``--on-incompatible`` value. The command line gets the two options, which reach a
runner only when they are not at their default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pytest

from binding_metrics.capabilities import (
    DEFAULT_MSA_POLICY,
    MSA_POLICIES,
    POLICIES,
    IncompatibleInputError,
    InputProfile,
    preflight,
)
from binding_metrics.cli import batch, prediction, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import run_pipeline
from binding_metrics.predictors import _msa
from binding_metrics.predictors._msa import (
    FORBID_SERVER,
    SINGLE_SEQUENCE,
    assert_no_server_flag,
    check_msa_of_request,
    check_msa_request,
)
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.store import PredictionRequest
from tests.test_pre_cli_run import LINEAR, Tripwires

PROFILE = InputProfile("B", "A", n_binder_residues=12, binder_type="peptide")

#: Keys of requests built before the MSA fields existed (see ``plain_request``).
KEY_WITH_SERVER_BEFORE = "56fb84455b4b2abc8de1dec6d098a312ee64b5540f14da35bb6f594fddc75f70"
KEY_WITHOUT_SERVER_BEFORE = "1fc5fb3645a09d02c2dabb9f796542bbb8a24733c5a393aa7483aacc9dffa634"

OPTIONS = {
    "use_msa_server": True,
    "binder_cyclic": "auto",
    "extra_args": [],
    "presets": ["predict"],
}

CAN_DO_ALL = dict(supports_given_msa=True, supports_single=True)
SINGLE_ONLY = dict(supports_given_msa=False, supports_single=True)
NOTHING = dict(supports_given_msa=False, supports_single=False)


@pytest.fixture
def alignment(tmp_path):
    path = tmp_path / "receptor.a3m"
    path.write_text(">query\nSQETFSDLWKLLPEN\n>hit\nSQETFSDLWKLLPEN\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------


class TestTheVocabulary:
    def test_the_policies(self):
        assert MSA_POLICIES == ("default", "forbid-server")
        assert _msa.MSA_POLICIES is MSA_POLICIES
        assert DEFAULT_MSA_POLICY == "default" == _msa.DEFAULT_MSA_POLICY
        assert FORBID_SERVER == "forbid-server" and SINGLE_SEQUENCE == "single"


class TestCheckMsaRequest:
    def test_a_request_that_asks_for_nothing_adds_nothing(self):
        fragments = check_msa_request("boltz2", use_msa_server=True, **CAN_DO_ALL)
        assert fragments.options == {} and fragments.extra_files == {}
        options, extra_files = fragments  # a named tuple
        assert options == {} and extra_files == {}

    def test_a_server_run_with_the_default_policy_is_allowed_with_a_file(self, alignment):
        fragments = check_msa_request(
            "boltz2", use_msa_server=True, receptor_msa=alignment, **CAN_DO_ALL
        )
        assert fragments.options == {}  # neither the policy nor a single sequence
        assert fragments.extra_files == {"receptor_msa": alignment}

    def test_forbid_server_with_a_file(self, alignment):
        fragments = check_msa_request(
            "boltz2",
            use_msa_server=False,
            msa_policy="forbid-server",
            receptor_msa=str(alignment),
            **CAN_DO_ALL,
        )
        assert fragments.options == {"msa_policy": "forbid-server"}
        assert fragments.extra_files == {"receptor_msa": alignment}

    def test_forbid_server_with_a_single_sequence_receptor(self):
        fragments = check_msa_request(
            "boltz2",
            use_msa_server=False,
            msa_policy="forbid-server",
            receptor_msa="single",
            **CAN_DO_ALL,
        )
        assert fragments.options == {"msa_policy": "forbid-server", "receptor_msa": "single"}
        assert fragments.extra_files == {}

    def test_a_server_run_is_refused_under_the_policy(self, alignment):
        with pytest.raises(ValueError) as caught:
            check_msa_request(
                "boltz2",
                use_msa_server=True,
                msa_policy="forbid-server",
                receptor_msa=alignment,
                **CAN_DO_ALL,
            )
        message = str(caught.value)
        assert "the sequences would be sent to an MSA server and the policy forbids it" in message
        assert "pass use_msa_server=False and receptor_msa=PATH" in message
        assert message.startswith("boltz2: ")

    def test_a_receptor_without_a_source_is_refused_and_single_must_be_asked_for(self):
        with pytest.raises(
            ValueError, match="needs a source for the MSA of the receptor"
        ) as caught:
            check_msa_request(
                "boltz2", use_msa_server=False, msa_policy="forbid-server", **CAN_DO_ALL
            )
        assert "receptor_msa='single'" in str(caught.value)

    def test_the_server_rule_comes_before_the_source_rule(self):
        with pytest.raises(ValueError, match="sent to an MSA server"):
            check_msa_request("boltz2", use_msa_server=True, msa_policy="forbid-server", **NOTHING)

    def test_colabfold_takes_a_single_sequence_only(self, alignment):
        reason = "colabfold_batch takes FASTA input only"
        with pytest.raises(ValueError) as caught:
            check_msa_request(
                "af2",
                use_msa_server=False,
                msa_policy="forbid-server",
                receptor_msa=alignment,
                unsupported_reason=reason,
                **SINGLE_ONLY,
            )
        message = str(caught.value)
        assert "af2 cannot be given the alignment of the receptor" in message
        assert reason in message and "receptor_msa='single' only" in message
        fragments = check_msa_request(
            "af2",
            use_msa_server=False,
            msa_policy="forbid-server",
            receptor_msa="single",
            **SINGLE_ONLY,
        )
        assert fragments.options == {"msa_policy": "forbid-server", "receptor_msa": "single"}

    def test_a_runner_that_takes_neither_refuses_both_under_any_policy(self, alignment):
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            check_msa_request("x", use_msa_server=True, receptor_msa=alignment, **NOTHING)
        with pytest.raises(ValueError, match="cannot run a receptor without an alignment"):
            check_msa_request("x", use_msa_server=True, receptor_msa="single", **NOTHING)

    def test_a_path_called_single_is_a_file(self, tmp_path):
        (tmp_path / "single").write_text(">q\nAAA\n", encoding="utf-8")
        fragments = check_msa_request(
            "boltz2", use_msa_server=False, receptor_msa=tmp_path / "single", **CAN_DO_ALL
        )
        assert fragments.options == {} and fragments.extra_files["receptor_msa"].is_file()

    def test_bad_arguments(self, tmp_path):
        with pytest.raises(ValueError, match="msa_policy must be one of"):
            check_msa_request("x", use_msa_server=False, msa_policy="never", **CAN_DO_ALL)
        with pytest.raises(FileNotFoundError, match="receptor_msa is not a file"):
            check_msa_request(
                "x", use_msa_server=False, receptor_msa=tmp_path / "none.a3m", **CAN_DO_ALL
            )
        with pytest.raises(TypeError, match="receptor_msa must be None"):
            check_msa_request("x", use_msa_server=False, receptor_msa=3, **CAN_DO_ALL)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


class TestAssertNoServerFlag:
    @pytest.mark.parametrize(
        "command",
        [
            ["boltz", "predict", "in.yaml", "--out_dir", "o"],
            ["boltz", "predict", "in.yaml", "--use_msa_server=false"],
            ["boltz", "predict", "in.yaml", "--use_msa_server", "false"],
            ["run_openfold", "predict", "--use_msa_server=False", "--query_json", "q.json"],
            ["protenix", "pred", "--use_msa", "false"],
            ["protenix", "pred", "--use_msa=false"],
            ["colabfold_batch", "--msa-mode", "single_sequence", "in.fasta", "out"],
            ["colabfold_batch", "--msa-mode=single_sequence", "in.fasta", "out"],
        ],
    )
    def test_a_command_without_a_server_flag_passes(self, command):
        assert_no_server_flag(command, "stub")

    @pytest.mark.parametrize(
        "command, flag",
        [
            (["boltz", "predict", "in.yaml", "--use_msa_server"], "--use_msa_server"),
            (["boltz", "predict", "--use_msa_server", "--out_dir", "o"], "--use_msa_server"),
            (["boltz", "predict", "--use_msa_server", "in.yaml"], "--use_msa_server"),
            (["run_openfold", "--use_msa_server=true"], "--use_msa_server true"),
            (["run_openfold", "--use_msa_server", "True"], "--use_msa_server True"),
            (["run_openfold", "--use_msa_server=maybe"], "--use_msa_server maybe"),
            (["protenix", "pred", "--use_msa", "true"], "--use_msa true"),
            (["protenix", "pred", "--use_msa=1"], "--use_msa 1"),
            (["colabfold_batch", "--msa-mode", "mmseqs2_uniref_env"], "--msa-mode mmseqs2_uni"),
            (["colabfold_batch", "--msa-mode=mmseqs2_uniref", "in", "out"], "mmseqs2_uniref"),
            (["colabfold_batch", "--msa-mode"], "--msa-mode"),
        ],
    )
    def test_a_server_flag_is_refused_and_named(self, command, flag):
        with pytest.raises(ValueError, match="nothing was started") as caught:
            assert_no_server_flag(command, "stub")
        assert flag in str(caught.value) and str(caught.value).startswith("stub: ")

    def test_use_msa_true_is_allowed_only_when_every_chain_has_a_path(self):
        command = ["protenix", "pred", "--use_msa", "true"]
        assert_no_server_flag(command, "protenix", chains_have_msa_paths=True)
        with pytest.raises(ValueError, match="--use_msa true"):
            assert_no_server_flag(command, "protenix")
        with pytest.raises(ValueError):  # the paths do not excuse a server flag of another kind
            assert_no_server_flag(
                ["protenix", "--use_msa", "maybe"], "protenix", chains_have_msa_paths=True
            )

    def test_colabfold_needs_its_flag_because_the_model_defaults_to_the_server(self):
        with pytest.raises(ValueError, match="has no --msa-mode"):
            assert_no_server_flag(["colabfold_batch", "in.fasta", "out"], "af2")
        assert_no_server_flag(["colabfold_batch", "in.fasta", "out"], "boltz2")  # absent is off

    def test_a_string_and_paths_are_read_like_a_command(self, tmp_path):
        assert_no_server_flag("boltz predict in.yaml --use_msa_server=false", "stub")
        with pytest.raises(ValueError, match="--use_msa_server"):
            assert_no_server_flag("boltz predict in.yaml --use_msa_server", "stub")
        assert_no_server_flag(["boltz", "predict", tmp_path / "in.yaml"], "stub")


# ---------------------------------------------------------------------------
# stub runners: no server flag reaches the command under the policy
# ---------------------------------------------------------------------------


class StubRunner(PredictionRunner):
    """A runner that builds the command line the way a real one does, and starts nothing."""

    supported_modes = frozenset({"predict", "score"})
    default_mode = "score"
    #: what the model accepts, as ``check_msa_request`` wants it
    capacity = CAN_DO_ALL

    def __init__(self):
        self.started: list[list[str]] = []

    def make_request(
        self,
        input_path,
        *,
        name,
        binder_chain="B",
        receptor_chain="A",
        mode="score",
        use_msa_server=True,
        msa_policy="default",
        receptor_msa=None,
    ):
        fragments = check_msa_request(
            self.name,
            use_msa_server=use_msa_server,
            msa_policy=msa_policy,
            receptor_msa=receptor_msa,
            **self.capacity,
        )
        return PredictionRequest(
            self.name,
            name,
            mode=mode,
            input_path=input_path,
            binder_chain=binder_chain,
            receptor_chain=receptor_chain,
            options={"use_msa_server": bool(use_msa_server), **fragments.options},
            extra_files=fragments.extra_files,
        )

    def command(self, request):
        raise NotImplementedError

    def _checked_command(self, request):
        check_msa_of_request(self.name, request, **self.capacity)
        command = self.command(request)
        if request.options.get("msa_policy", DEFAULT_MSA_POLICY) == FORBID_SERVER:
            assert_no_server_flag(command, self.name)
        return command

    def prepare(self, request, work_dir):
        command = self._checked_command(request)
        target = Path(work_dir) / "command.txt"
        target.write_text(" ".join(command), encoding="utf-8")
        return target

    def run(self, request, work_dir):
        self.started.append(self._checked_command(request))
        return Path(work_dir)


class BoltzStub(StubRunner):
    """``--use_msa_server`` is a bare flag; left out, no server."""

    name = "stubboltz"

    def command(self, request):
        command = ["boltz", "predict", str(request.input_path)]
        if request.options.get("use_msa_server", True):
            command.append("--use_msa_server")
        return command


class OpenFoldStub(StubRunner):
    """``--use_msa_server=false`` is written when the server is off."""

    name = "stubof"

    def command(self, request):
        flag = "true" if request.options.get("use_msa_server", True) else "false"
        return ["run_openfold", "predict", f"--use_msa_server={flag}"]


class ProtenixStub(StubRunner):
    """``--use_msa`` takes a value; every chain has a path in the JSON when the receptor has."""

    name = "stubpx"

    def command(self, request):
        server = request.options.get("use_msa_server", True)
        return ["protenix", "pred", "--use_msa", "true" if server else "false"]


class ColabFoldStub(StubRunner):
    """Only ``single_sequence`` does not call the server."""

    name = "af2"
    capacity = SINGLE_ONLY

    def command(self, request):
        server = request.options.get("use_msa_server", True)
        mode = "mmseqs2_uniref_env" if server else "single_sequence"
        return ["colabfold_batch", "--msa-mode", mode, "in.fasta", "out"]


RUNNERS = [BoltzStub, OpenFoldStub, ProtenixStub, ColabFoldStub]


@pytest.fixture
def complex_file(tmp_path):
    path = tmp_path / "complex.cif"
    path.write_bytes(b"data_x\n#a complex\n")
    return path


class TestNoServerFlagReachesTheCommand:
    @pytest.mark.parametrize("runner_class", RUNNERS)
    def test_under_the_policy_the_command_holds_no_server_flag(
        self, runner_class, complex_file, tmp_path, alignment
    ):
        runner = runner_class()
        receptor = "single" if runner.capacity is SINGLE_ONLY else alignment
        request = runner.make_request(
            complex_file,
            name="s",
            use_msa_server=False,
            msa_policy="forbid-server",
            receptor_msa=receptor,
        )
        command_file = runner.prepare(request, tmp_path)
        runner.run(request, tmp_path)
        assert runner.started and runner.started[0] == command_file.read_text(
            encoding="utf-8"
        ).split(" ")
        # the check sees the same command: it is clean
        assert_no_server_flag(runner.started[0], runner.name)

    @pytest.mark.parametrize("runner_class", RUNNERS)
    def test_the_server_flag_is_written_without_the_policy(self, runner_class, complex_file):
        runner = runner_class()
        request = runner.make_request(complex_file, name="s", use_msa_server=True)
        command = runner.command(request)
        with pytest.raises(ValueError):
            assert_no_server_flag(command, runner.name)

    @pytest.mark.parametrize("runner_class", RUNNERS)
    def test_a_request_with_the_server_on_is_refused_when_it_is_made(
        self, runner_class, complex_file, alignment
    ):
        with pytest.raises(ValueError, match="the policy forbids it"):
            runner_class().make_request(
                complex_file,
                name="s",
                use_msa_server=True,
                msa_policy="forbid-server",
                receptor_msa="single",
            )

    @pytest.mark.parametrize("runner_class", RUNNERS)
    @pytest.mark.parametrize("step", ["prepare", "run"])
    def test_a_request_made_by_hand_is_refused_before_a_process_starts(
        self, runner_class, step, complex_file, tmp_path
    ):
        runner = runner_class()
        request = PredictionRequest(
            runner.name,
            "s",
            mode="score",
            input_path=complex_file,
            binder_chain="B",
            receptor_chain="A",
            options={
                "use_msa_server": True,
                "msa_policy": "forbid-server",
                "receptor_msa": "single",
            },
        )
        with pytest.raises(ValueError, match="the policy forbids it"):
            getattr(runner, step)(request, tmp_path)
        assert runner.started == [] and not (tmp_path / "command.txt").exists()

    @pytest.mark.parametrize("runner_class", RUNNERS)
    def test_a_hand_made_request_without_a_receptor_source_is_refused(
        self, runner_class, complex_file, tmp_path
    ):
        runner = runner_class()
        request = PredictionRequest(
            runner.name,
            "s",
            mode="score",
            input_path=complex_file,
            binder_chain="B",
            receptor_chain="A",
            options={"use_msa_server": False, "msa_policy": "forbid-server"},
        )
        with pytest.raises(ValueError, match="needs a source for the MSA of the receptor"):
            runner.run(request, tmp_path)
        assert runner.started == []

    def test_a_command_with_a_server_flag_is_stopped_by_the_last_check(
        self, complex_file, tmp_path, alignment
    ):
        """A runner whose ``command`` writes a flag the request does not show is still refused."""

        class Leaky(BoltzStub):
            def command(self, request):
                return [*super().command(request), "--use_msa_server"]

        runner = Leaky()
        request = runner.make_request(
            complex_file,
            name="s",
            use_msa_server=False,
            msa_policy="forbid-server",
            receptor_msa=alignment,
        )
        with pytest.raises(ValueError, match="--use_msa_server"):
            runner.run(request, tmp_path)
        assert runner.started == []

    def test_a_hand_made_request_with_a_file_for_a_runner_that_cannot_take_it(
        self, complex_file, tmp_path, alignment
    ):
        runner = ColabFoldStub()
        request = PredictionRequest(
            runner.name,
            "s",
            mode="score",
            input_path=complex_file,
            binder_chain="B",
            receptor_chain="A",
            options={"use_msa_server": False, "msa_policy": "forbid-server"},
            extra_files={"receptor_msa": alignment},
        )
        with pytest.raises(ValueError, match="cannot be given the alignment"):
            runner.prepare(request, tmp_path)


# ---------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------


def plain_request(tmp_path, *, options=None, fragments=None, name="s1"):
    """The request of ``test_m4_refold_lock_vocabulary.build_request`` plus MSA fragments."""
    complex_file = tmp_path / "complex.cif"
    complex_file.write_bytes(b"data_x\n#pinned complex\n")
    merged = dict(OPTIONS if options is None else options)
    extra_files = {}
    if fragments is not None:
        merged.update(fragments.options)
        extra_files = fragments.extra_files
    return PredictionRequest(
        "stub",
        name,
        mode="refold",
        input_path=complex_file,
        binder_chain="B",
        receptor_chain="A",
        seeds=(42,),
        num_samples=5,
        model_version="1.0",
        options=merged,
        extra_files=extra_files,
    )


class TestTheKey:
    def test_a_request_without_the_new_fields_has_the_key_it_had(self, tmp_path):
        none_asked = check_msa_request("stub", use_msa_server=True, **CAN_DO_ALL)
        assert plain_request(tmp_path).key() == KEY_WITH_SERVER_BEFORE
        assert plain_request(tmp_path, fragments=none_asked).key() == KEY_WITH_SERVER_BEFORE
        off = check_msa_request("stub", use_msa_server=False, **CAN_DO_ALL)
        no_server = {**OPTIONS, "use_msa_server": False}
        assert plain_request(tmp_path, options=no_server, fragments=off).key() == (
            KEY_WITHOUT_SERVER_BEFORE
        )

    def forbidding(self, receptor_msa):
        return check_msa_request(
            "stub",
            use_msa_server=False,
            msa_policy="forbid-server",
            receptor_msa=receptor_msa,
            **CAN_DO_ALL,
        )

    def key_with(self, tmp_path, receptor_msa):
        no_server = {**OPTIONS, "use_msa_server": False}
        return plain_request(
            tmp_path, options=no_server, fragments=self.forbidding(receptor_msa)
        ).key()

    def test_the_same_content_gives_the_same_key_at_any_path(self, tmp_path, alignment):
        copy = tmp_path / "elsewhere" / "copy.a3m"
        copy.parent.mkdir()
        shutil.copy(alignment, copy)
        assert self.key_with(tmp_path, copy) == self.key_with(tmp_path, alignment)

    def test_another_content_gives_another_key(self, tmp_path, alignment):
        other = tmp_path / "other.a3m"
        other.write_text(
            alignment.read_text(encoding="utf-8") + ">more\nSQETFSDLWKLLPEN\n", encoding="utf-8"
        )
        assert self.key_with(tmp_path, other) != self.key_with(tmp_path, alignment)

    def test_the_policy_a_file_and_a_single_sequence_are_all_in_the_key(self, tmp_path, alignment):
        no_server = {**OPTIONS, "use_msa_server": False}
        keys = {
            "without": plain_request(tmp_path, options=no_server).key(),
            "single": self.key_with(tmp_path, "single"),
            "file": self.key_with(tmp_path, alignment),
            # the file alone, with the default policy
            "file, default policy": plain_request(
                tmp_path,
                options=no_server,
                fragments=check_msa_request(
                    "stub", use_msa_server=False, receptor_msa=alignment, **CAN_DO_ALL
                ),
            ).key(),
        }
        assert len(set(keys.values())) == len(keys)

    def test_the_file_is_hashed_by_content_under_its_role(self, tmp_path, alignment):
        request = plain_request(
            tmp_path,
            options={**OPTIONS, "use_msa_server": False},
            fragments=self.forbidding(alignment),
        )
        assert set(request.content_hashes) == {"input", "receptor_msa"}
        assert request.canonical()["options"]["msa_policy"] == "forbid-server"
        assert "receptor_msa" not in request.canonical()["options"]  # a file is not an option

    def test_the_format_of_the_key_is_unchanged(self, tmp_path):
        from binding_metrics.predictors.store import KEY_FORMAT

        assert KEY_FORMAT == 1
        assert plain_request(tmp_path).canonical()["format"] == 1


# ---------------------------------------------------------------------------
# pre-flight
# ---------------------------------------------------------------------------


class TestPreflight:
    @pytest.mark.parametrize("policy", POLICIES)
    @pytest.mark.parametrize("model", ["boltz2", "of3", "protenix", "af2"])
    def test_a_server_under_the_policy_is_refused_whatever_on_incompatible_says(
        self, model, policy
    ):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(
                PROFILE,
                [],
                model,
                policy=policy,
                mode="predict" if model in ("protenix", "af2") else None,
                msa_policy="forbid-server",
                msa_server=True,
            )
        (violation,) = [v for v in caught.value.violations if v.constraint == "msa_policy"]
        assert violation.fact == "the run would use an MSA server"
        assert violation.kind == "msa" and violation.name == model
        assert "forbid-server" in violation.requirement
        assert "use_msa_server=False" in violation.fix and "receptor_msa" in violation.fix
        assert "--prediction-no-msa-server" in violation.fix
        assert "every --on-incompatible value" in violation.reason
        assert caught.value.report.refused and caught.value.report.msa_policy == "forbid-server"
        assert "Pre-flight check failed" in str(caught.value)

    def test_the_policy_is_hard_for_a_predictor_that_only_warns(self):
        with pytest.raises(IncompatibleInputError):
            preflight(
                PROFILE,
                [],
                "boltz2",
                policy="warn",
                predictor_policy="warn",
                msa_policy="forbid-server",
                msa_server=True,
            )

    def test_a_run_without_a_server_passes_and_the_report_names_the_policy(self):
        report = preflight(PROFILE, [], "boltz2", msa_policy="forbid-server", msa_server=False)
        assert report.compatible and report.msa_policy == "forbid-server"
        assert "MSA policy: forbid-server" in report.format()
        assert report.to_dict()["msa_policy"] == "forbid-server"

    def test_the_default_policy_changes_nothing_in_the_report(self):
        plain = preflight(PROFILE, [], "boltz2", mode="score")
        server = preflight(
            PROFILE, [], "boltz2", mode="score", msa_policy="default", msa_server=True
        )
        assert plain.to_dict() == server.to_dict() and "msa_policy" not in plain.to_dict()
        assert plain.format() == server.format() and "MSA policy" not in plain.format()

    def test_a_server_that_is_not_known_gives_a_note_and_not_a_refusal(self):
        report = preflight(PROFILE, [], "boltz2", msa_policy="forbid-server")
        assert report.compatible
        assert any("not checked" in note and "MSA server" in note for note in report.notes)

    def test_without_a_predictor_there_is_nothing_to_refuse(self):
        report = preflight(PROFILE, [], None, msa_policy="forbid-server", msa_server=True)
        assert report.compatible and not report.notes

    def test_an_unknown_policy_is_a_value_error(self):
        with pytest.raises(ValueError, match="msa_policy must be one of"):
            preflight(PROFILE, [], "boltz2", msa_policy="never")


class TestPreflightOfTheCommandLine:
    """``check_input`` is what ``--preflight-only`` prints."""

    def check(self, **kwargs):
        from binding_metrics.preflight_cli import check_input

        kwargs.setdefault("model", ("boltz2", "prediction", False, "score"))
        return check_input(LINEAR, "B", "A", include_plan=True, **kwargs)

    @pytest.mark.parametrize("policy", POLICIES)
    def test_a_server_run_is_refused_under_every_policy(self, policy):
        outcome = self.check(on_incompatible=policy, msa_policy="forbid-server", msa_server=True)
        assert outcome.refused and outcome.error is not None
        assert outcome.block["status"] == "refused" and outcome.block["policy"] == policy
        assert "msa_policy: the run would use an MSA server" in outcome.block["reason"]
        assert outcome.block["report"]["msa_policy"] == "forbid-server"

    @pytest.mark.parametrize("policy", POLICIES)
    def test_the_plan_shows_the_policy_and_the_receptor_msa(self, policy, alignment):
        outcome = self.check(
            on_incompatible=policy,
            msa_policy="forbid-server",
            msa_server=False,
            receptor_msa=alignment,
        )
        assert not outcome.refused and outcome.block["status"] == "ok"
        plan = outcome.block["plan"].splitlines()
        assert "MSA policy: forbid-server (no MSA server may be called)" in plan
        assert plan[plan.index("MSA policy: forbid-server (no MSA server may be called)") + 1] == (
            f"Receptor MSA: {alignment}"
        )

    def test_an_output_made_elsewhere_runs_no_server(self):
        outcome = self.check(
            model=("boltz2", "prediction", True, None),
            msa_policy="forbid-server",
            msa_server=True,
        )
        assert not outcome.refused
        assert "MSA policy" not in outcome.block["plan"]

    def test_the_default_run_is_unchanged(self):
        assert self.check().block["status"] == "ok"
        assert "MSA" not in self.check().block["plan"]

    def test_a_run_without_a_model_step_is_not_concerned(self):
        outcome = self.check(model=None, msa_policy="forbid-server", msa_server=True)
        assert not outcome.refused


# ---------------------------------------------------------------------------
# the command line and the Python API
# ---------------------------------------------------------------------------


class MsaBoltz:
    """A Boltz-2 runner class that has the two settings (the real one is given them later)."""

    name = "boltz2"
    supported_modes = frozenset({"predict", "refold", "score", "score-lock"})
    default_mode = "score"

    def make_request(
        self,
        input_path,
        *,
        name,
        binder_chain,
        receptor_chain,
        mode="score",
        use_msa_server=True,
        msa_policy="default",
        receptor_msa=None,
    ):
        raise AssertionError("a request is not made in these tests")


class NoMsaBoltz(MsaBoltz):
    """A runner without the settings."""

    def make_request(
        self, input_path, *, name, binder_chain, receptor_chain, mode="score", use_msa_server=True
    ):
        raise AssertionError("a request is not made in these tests")


@pytest.fixture
def runner_with_msa(monkeypatch):
    original = prediction.runner_class
    monkeypatch.setattr(
        prediction, "runner_class", lambda model: MsaBoltz if model == "boltz2" else original(model)
    )


@pytest.fixture
def runner_without_msa(monkeypatch):
    original = prediction.runner_class
    monkeypatch.setattr(
        prediction,
        "runner_class",
        lambda model: NoMsaBoltz if model == "boltz2" else original(model),
    )


def prediction_parser():
    parser = argparse.ArgumentParser()
    prediction.add_prediction_args(parser)
    return parser


class TestTheOptions:
    def test_the_defaults_and_the_choices(self):
        args = prediction_parser().parse_args([])
        assert args.prediction_msa_policy == "default" and args.prediction_receptor_msa is None
        action = prediction_parser()._option_string_actions["--prediction-msa-policy"]
        assert tuple(action.choices) == MSA_POLICIES
        with pytest.raises(SystemExit):
            prediction_parser().parse_args(["--prediction-msa-policy", "never"])

    @pytest.mark.parametrize("command", ["run", "batch"])
    def test_both_commands_have_them_in_the_prediction_group(self, command, monkeypatch):
        from tests.test_w3a_cli_help import CLI_MODULES, capture_parser, describe_options

        described = describe_options(capture_parser(CLI_MODULES[command], monkeypatch))
        policy = described["--prediction-msa-policy"]
        receptor = described["--prediction-receptor-msa"]
        assert policy[0] == receptor[0] == "Prediction"
        assert policy[2] == ["default", "forbid-server"] and policy[5] == "'default'"
        assert receptor[1] == "PATH" and receptor[5] == "None"
        assert "forbid-server: no sequence may be sent to a server" in policy[6]
        assert "the word single" in receptor[6]

    def test_they_need_a_predictor(self):
        parser = prediction_parser()
        for argv in (
            ["--prediction-msa-policy", "forbid-server"],
            ["--prediction-receptor-msa", "x.a3m"],
        ):
            with pytest.raises(SystemExit):
                prediction.check_prediction_args(parser, parser.parse_args(argv))
        # the default policy is not "given"
        prediction.check_prediction_args(parser, parser.parse_args([]))

    def test_they_cannot_be_combined_with_a_directory(self, tmp_path, runner_with_msa):
        parser = prediction_parser()
        for extra in (
            ["--prediction-msa-policy", "forbid-server"],
            ["--prediction-receptor-msa", "single"],
        ):
            args = parser.parse_args(
                ["--predictor", "boltz2", "--prediction-dir", str(tmp_path), *extra]
            )
            with pytest.raises(SystemExit):
                prediction.check_prediction_args(parser, args)

    def test_the_receptor_msa_must_be_a_file_or_the_word_single(
        self, tmp_path, alignment, runner_with_msa
    ):
        parser = prediction_parser()
        good = parser.parse_args(
            ["--predictor", "boltz2", "--prediction-receptor-msa", str(alignment)]
        )
        prediction.check_prediction_args(parser, good)
        single = parser.parse_args(["--predictor", "boltz2", "--prediction-receptor-msa", "single"])
        prediction.check_prediction_args(parser, single)
        missing = parser.parse_args(
            ["--predictor", "boltz2", "--prediction-receptor-msa", str(tmp_path / "none.a3m")]
        )
        with pytest.raises(SystemExit):
            prediction.check_prediction_args(parser, missing)

    @pytest.mark.parametrize(
        "extra, option",
        [
            (["--prediction-msa-policy", "forbid-server"], "--prediction-msa-policy"),
            (["--prediction-receptor-msa", "single"], "--prediction-receptor-msa"),
        ],
    )
    def test_a_runner_without_the_setting_says_the_model_does_not_have_it(
        self, extra, option, runner_without_msa, capsys
    ):
        parser = prediction_parser()
        args = parser.parse_args(["--predictor", "boltz2", *extra])
        with pytest.raises(SystemExit):
            prediction.check_prediction_args(parser, args)
        assert f"{option} is not available for Boltz-2" in capsys.readouterr().err

    def test_the_python_api_checks_the_same(self, alignment, runner_with_msa):
        options = prediction.check_predictor(
            "boltz2", None, prediction_msa_policy="forbid-server", prediction_receptor_msa=alignment
        )
        assert options.mode == "score"
        with pytest.raises(ValueError, match="prediction_msa_policy must be one of"):
            prediction.check_predictor("boltz2", None, prediction_msa_policy="never")
        with pytest.raises(ValueError, match="needs --predictor"):
            prediction.check_predictor(None, None, prediction_msa_policy="forbid-server")
        with pytest.raises(ValueError, match="cannot be combined with --prediction-dir"):
            prediction.check_predictor("boltz2", alignment.parent, prediction_receptor_msa="single")

    def test_the_runner_without_the_setting_is_refused_by_the_python_api(self, runner_without_msa):
        with pytest.raises(
            ValueError, match="--prediction-msa-policy is not available for Boltz-2"
        ):
            prediction.check_predictor("boltz2", None, prediction_msa_policy="forbid-server")
        # a request that does not ask for them is accepted by the same runner
        prediction.check_predictor("boltz2", None)

    @pytest.mark.parametrize("model", ["af2", "boltz2", "of3", "protenix"])
    def test_a_runner_that_has_not_the_keywords_yet_refuses_the_option_and_not_the_default(
        self, model
    ):
        """The real runners: either they take the keyword, or the message names option and model."""
        from binding_metrics.cli.prediction import runner_class, takes_setting

        runner = runner_class(model)
        if takes_setting(runner, "msa_policy"):
            return
        with pytest.raises(
            ValueError, match="not available for .*: its runner has no 'msa_policy'"
        ):
            prediction.check_predictor(model, None, prediction_msa_policy="forbid-server")
        prediction.check_predictor(model, None)  # the default flows are unchanged


class RecordingRunner(MsaBoltz):
    """Records the keywords of the call to ``make_request``."""

    calls: list[dict] = []

    def make_request(self, input_path, *, name, binder_chain, receptor_chain, mode="score",
                     use_msa_server=True, msa_policy="default", receptor_msa=None):  # fmt: skip
        keywords = dict(
            mode=mode,
            use_msa_server=use_msa_server,
            msa_policy=msa_policy,
            receptor_msa=receptor_msa,
        )
        type(self).calls.append(keywords)
        return keywords


class TestMakeRequest:
    """``cli.prediction.make_request`` passes a setting only when it is not at its default."""

    def make(self, runner, **kwargs):
        return prediction.make_request(
            "boltz2", "s", "in.pdb", binder_chain="B", receptor_chain="A", runner=runner, **kwargs
        )

    def test_the_defaults_pass_nothing(self):
        class OnlyDefaults(MsaBoltz):
            def make_request(self, input_path, *, name, binder_chain, receptor_chain, mode="score"):
                return {"mode": mode}

        # a runner without the settings is not called with them
        assert self.make(OnlyDefaults()) == {"mode": "score"}
        assert self.make(OnlyDefaults(), prediction_msa_policy="default") == {"mode": "score"}

    def test_a_setting_that_is_given_is_passed(self, alignment):
        RecordingRunner.calls = []
        self.make(RecordingRunner())
        self.make(
            RecordingRunner(),
            openfold_use_msa_server=False,
            prediction_msa_policy="forbid-server",
            prediction_receptor_msa=alignment,
        )
        self.make(RecordingRunner(), prediction_receptor_msa="single")
        assert RecordingRunner.calls == [
            {
                "mode": "score",
                "use_msa_server": True,
                "msa_policy": "default",
                "receptor_msa": None,
            },
            {
                "mode": "score",
                "use_msa_server": False,
                "msa_policy": "forbid-server",
                "receptor_msa": alignment,
            },  # fmt: skip
            {
                "mode": "score",
                "use_msa_server": True,
                "msa_policy": "default",
                "receptor_msa": "single",
            },  # fmt: skip
        ]

    @pytest.mark.parametrize(
        "keywords, option",
        [
            ({"prediction_msa_policy": "forbid-server"}, "--prediction-msa-policy"),
            ({"prediction_receptor_msa": "single"}, "--prediction-receptor-msa"),
        ],
    )
    def test_a_runner_without_the_keyword_is_refused_with_the_option_and_the_model(
        self, keywords, option
    ):
        with pytest.raises(ValueError, match=f"{option} is not available for Boltz-2"):
            self.make(NoMsaBoltz(), **keywords)


class TestRunPipeline:
    def pipeline(self, out, **kwargs):
        kwargs.setdefault("skip_prep", True)
        kwargs.setdefault("skip_relax", True)
        kwargs.setdefault("metrics", frozenset({"openfold"}))
        kwargs.setdefault("openfold_conda_env", None)
        kwargs.setdefault("peptide_chain", "B")
        kwargs.setdefault("receptor_chain", "A")
        kwargs.setdefault("predictor", "boltz2")
        return run_pipeline(LINEAR, out, **kwargs)

    @pytest.mark.parametrize("policy", POLICIES)
    def test_a_server_run_is_refused_before_anything_starts(
        self, policy, tmp_path, monkeypatch, runner_with_msa
    ):
        tripwires = Tripwires(monkeypatch)
        out = tmp_path / "out"
        with pytest.raises(IncompatibleInputError, match="the run would use an MSA server"):
            self.pipeline(
                out,
                on_incompatible=policy,
                prediction_msa_policy="forbid-server",
                prediction_receptor_msa="single",
            )
        assert tripwires.calls == [] and not out.exists()

    @pytest.mark.parametrize("policy", POLICIES)
    def test_preflight_only_returns_the_refusal_and_the_plan(
        self, policy, tmp_path, runner_with_msa
    ):
        block = self.pipeline(
            tmp_path / "out",
            on_incompatible=policy,
            prediction_msa_policy="forbid-server",
            prediction_receptor_msa="single",
            preflight_only=True,
        )["preflight"]
        assert block["status"] == "refused"
        assert "MSA policy: forbid-server" in block["plan"]
        assert "Receptor MSA: single" in block["plan"]
        assert "the run would use an MSA server" in block["plan"]

    def test_without_a_server_the_plan_is_clean(self, tmp_path, alignment, runner_with_msa):
        block = self.pipeline(
            tmp_path / "out",
            prediction_msa_policy="forbid-server",
            prediction_use_msa_server=False,
            prediction_receptor_msa=alignment,
            preflight_only=True,
        )["preflight"]
        assert block["status"] == "ok"
        assert f"Receptor MSA: {alignment}" in block["plan"]

    def test_the_default_policy_does_not_look_at_the_server(self, tmp_path, runner_with_msa):
        block = self.pipeline(tmp_path / "out", preflight_only=True)["preflight"]
        assert block["status"] == "ok" and "MSA" not in block["plan"]

    def test_the_options_are_refused_for_a_runner_without_them(self, tmp_path, runner_without_msa):
        with pytest.raises(ValueError, match="not available for Boltz-2"):
            self.pipeline(tmp_path / "out", prediction_msa_policy="forbid-server")


class TestRunBatch:
    @pytest.fixture
    def samples(self, tmp_path):
        folder = tmp_path / "in"
        folder.mkdir()
        paths = []
        for name in ("a", "b"):
            path = folder / f"{name}.pdb"
            shutil.copy(LINEAR, path)
            paths.append(path)
        return paths

    def batch_of(self, samples, out, **kwargs):
        kwargs.setdefault("skip_prep", True)
        kwargs.setdefault("skip_relax", True)
        kwargs.setdefault("metrics", {"openfold"})
        kwargs.setdefault("openfold_conda_env", None)
        kwargs.setdefault("predictor", "boltz2")
        return run_batch(samples, out, **kwargs)

    @pytest.mark.parametrize("policy", POLICIES)
    def test_every_sample_is_refused_and_no_model_starts(
        self, policy, tmp_path, samples, monkeypatch, runner_with_msa
    ):
        tripwires = Tripwires(monkeypatch)
        rows = self.batch_of(
            samples,
            tmp_path / "out",
            on_incompatible=policy,
            prediction_msa_policy="forbid-server",
            prediction_receptor_msa="single",
        )
        assert tripwires.calls == []
        for row in rows:
            assert row["batch_status"] == "error" and row["preflight_status"] == "refused"
            assert "the run would use an MSA server" in row["preflight_reason"]

    @pytest.mark.parametrize("policy", POLICIES)
    def test_preflight_only_rows_say_the_same(self, policy, tmp_path, samples, runner_with_msa):
        rows = self.batch_of(
            samples,
            tmp_path / "out",
            on_incompatible=policy,
            prediction_msa_policy="forbid-server",
            prediction_receptor_msa="single",
            preflight_only=True,
        )
        assert {row["batch_status"] for row in rows} == {"error"}
        assert all("MSA policy: forbid-server" in row["preflight_plan"] for row in rows)
        assert not (tmp_path / "out").exists()

    def test_preflight_only_with_the_server_off_can_run(
        self, tmp_path, samples, alignment, runner_with_msa
    ):
        rows = self.batch_of(
            samples,
            tmp_path / "out",
            prediction_msa_policy="forbid-server",
            prediction_use_msa_server=False,
            prediction_receptor_msa=alignment,
            preflight_only=True,
        )
        assert {row["batch_status"] for row in rows} == {"ok"}

    def test_the_model_step_is_refused_under_warn_as_well(self, samples):
        allowed, why = batch._model_step_allowed(
            "a",
            samples[0],
            "B",
            "A",
            ("boltz2", "prediction", False, "score"),
            "auto",
            "warn",
            msa_policy="forbid-server",
            msa_server=True,
        )
        assert not allowed and "the run would use an MSA server" in why
        allowed, _ = batch._model_step_allowed(
            "a", samples[0], "B", "A", ("boltz2", "prediction", False, "score"), "auto", "warn"
        )
        assert allowed  # warn without the policy still lets everything through


class TestTheCommandLines:
    BASE = ["--metrics", "openfold", "--predictor", "boltz2", "--preflight-only"]

    def argv(self, monkeypatch, tool, *arguments):
        monkeypatch.setattr(sys, "argv", [tool, *arguments])

    @pytest.mark.parametrize("on_incompatible", POLICIES)
    def test_run_refuses_a_server_and_prints_the_plan(
        self, on_incompatible, tmp_path, monkeypatch, capsys, runner_with_msa
    ):
        self.argv(
            monkeypatch,
            "binding-metrics-run",
            "--input", str(LINEAR),
            "--output-dir", str(tmp_path / "out"),
            *self.BASE,
            "--on-incompatible", on_incompatible,
            "--prediction-msa-policy", "forbid-server",
            "--prediction-receptor-msa", "single",
        )  # fmt: skip
        with pytest.raises(SystemExit) as exit_info:
            run.main()
        out = capsys.readouterr().out
        assert exit_info.value.code == 1
        assert "Pre-flight check failed" in out and "MSA policy: forbid-server" in out
        assert "the run would use an MSA server" in out
        assert not (tmp_path / "out").exists()

    def test_run_accepts_the_policy_without_a_server(
        self, tmp_path, monkeypatch, capsys, alignment, runner_with_msa
    ):
        self.argv(
            monkeypatch,
            "binding-metrics-run",
            "--input", str(LINEAR),
            "--output-dir", str(tmp_path / "out"),
            *self.BASE,
            "--prediction-msa-policy", "forbid-server",
            "--prediction-no-msa-server",
            "--prediction-receptor-msa", str(alignment),
        )  # fmt: skip
        with pytest.raises(SystemExit) as exit_info:
            run.main()
        out = capsys.readouterr().out
        assert exit_info.value.code == 0
        assert "MSA policy: forbid-server" in out and f"Receptor MSA: {alignment}" in out

    def test_batch_refuses_a_server_in_the_plan_and_accepts_it_without(
        self, tmp_path, monkeypatch, capsys, alignment, runner_with_msa
    ):
        folder = tmp_path / "in"
        folder.mkdir()
        shutil.copy(LINEAR, folder / "a.pdb")
        common = [
            "--input-dir", str(folder),
            "--output-csv", str(tmp_path / "res" / "out.csv"),
            *self.BASE,
            "--prediction-msa-policy", "forbid-server",
            "--prediction-receptor-msa", str(alignment),
        ]  # fmt: skip
        self.argv(monkeypatch, "binding-metrics-batch", *common)
        with pytest.raises(SystemExit) as exit_info:
            batch.main()
        assert exit_info.value.code == 1
        assert "the run would use an MSA server" in capsys.readouterr().out
        self.argv(monkeypatch, "binding-metrics-batch", *common, "--prediction-no-msa-server")
        with pytest.raises(SystemExit) as exit_info:
            batch.main()
        assert exit_info.value.code == 0
        assert not (tmp_path / "res").exists()

    def test_a_usage_error_names_the_model_for_a_runner_without_the_setting(
        self, tmp_path, monkeypatch, capsys, runner_without_msa
    ):
        self.argv(
            monkeypatch,
            "binding-metrics-run",
            "--input", str(LINEAR),
            "--output-dir", str(tmp_path / "out"),
            *self.BASE,
            "--prediction-msa-policy", "forbid-server",
        )  # fmt: skip
        with pytest.raises(SystemExit) as exit_info:
            run.main()
        assert exit_info.value.code == 2
        assert "--prediction-msa-policy is not available for Boltz-2" in capsys.readouterr().err
