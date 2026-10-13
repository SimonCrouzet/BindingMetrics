"""The guard against a server flag reads the options the way the tools do.

``assert_no_server_flag`` is the last check before a process starts under the policy
``forbid-server``. The tools take the hyphen and the underscore spelling of an option alike
(OpenFold3 does), take the value as ``--name=value`` or as the next word, and let the last
occurrence win, so the guard has to read all of those, and the runners have to give it the final
command line, ``extra_args`` included.
"""

from __future__ import annotations

import dataclasses
import shutil

import pytest

from binding_metrics.predictors import af2_runner, boltz2_runner, of3_runner, protenix_runner
from binding_metrics.predictors._msa import assert_no_server_flag
from binding_metrics.predictors.af2_runner import ColabFoldRunner
from binding_metrics.predictors.boltz2_runner import Boltz2Runner
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.protenix_runner import ProtenixRunner
from tests.test_pre_cli_run import LINEAR


def spelled(flag: str, how: str) -> str:
    """``--use_msa_server`` as written with underscores (``us``) or hyphens (``hy``)."""
    return flag if how == "us" else "--" + flag[2:].replace("_", "-")


HOW = ["us", "hy"]


def forms(flag: str, value: str | None, how: str) -> list[str]:
    """The words of one occurrence: ``--flag=value``, ``--flag value`` or the bare flag."""
    name = spelled(flag, how)
    return [name] if value is None else [f"{name}={value}"]


def separate(flag: str, value: str, how: str) -> list[str]:
    return [spelled(flag, how), value]


def refuses(words, runner="stub", **kwargs) -> bool:
    try:
        assert_no_server_flag(words, runner, **kwargs)
    except ValueError as error:
        assert "nothing was started" in str(error)
        return True
    return False


# ---------------------------------------------------------------------------
# the table: spelling x form x flag
# ---------------------------------------------------------------------------


class TestOneOccurrence:
    @pytest.mark.parametrize("how", HOW)
    @pytest.mark.parametrize("value", [None, "true", "True", "1", "maybe"])
    def test_use_msa_server_on_is_refused_in_every_form(self, how, value):
        flag = "--use_msa_server"
        assert refuses(["tool", *forms(flag, value, how)])
        if value is not None:
            # a word that is no boolean is read as an argument of the command: the flag is bare
            assert refuses(["tool", *separate(flag, value, how)])

    @pytest.mark.parametrize("how", HOW)
    @pytest.mark.parametrize("value", ["false", "False", "0", "no"])
    def test_use_msa_server_off_is_accepted_in_every_form(self, how, value):
        flag = "--use_msa_server"
        assert not refuses(["tool", *forms(flag, value, how)])
        assert not refuses(["tool", *separate(flag, value, how), "input.yaml"])

    @pytest.mark.parametrize("how", HOW)
    def test_a_bare_flag_before_an_argument_or_another_option_is_on(self, how):
        name = spelled("--use_msa_server", how)
        assert refuses(["boltz", "predict", name, "input.yaml"])
        assert refuses(["boltz", "predict", name, "--out_dir", "o"])
        assert refuses(["boltz", "predict", "input.yaml", name])

    @pytest.mark.parametrize("how", HOW)
    @pytest.mark.parametrize("value", ["true", "True", "1", "maybe", None])
    def test_use_msa_is_refused_without_paths_and_when_it_is_no_boolean(self, how, value):
        words = ["protenix", *forms("--use_msa", value, how)]
        assert refuses(words)
        assert refuses(["protenix", *separate("--use_msa", value or "x", how)])
        # the paths excuse a true, and only a true
        assert refuses(words, chains_have_msa_paths=True) == (value in (None, "maybe"))

    @pytest.mark.parametrize("how", HOW)
    @pytest.mark.parametrize("value", ["false", "False", "0"])
    def test_use_msa_false_is_accepted(self, how, value):
        assert not refuses(["protenix", *forms("--use_msa", value, how)])
        assert not refuses(["protenix", *separate("--use_msa", value, how)])

    @pytest.mark.parametrize("how", HOW)
    @pytest.mark.parametrize("mode", ["mmseqs2_uniref_env", "mmseqs2_uniref", "mmseqs2", "custom"])
    def test_a_msa_mode_that_calls_the_server_is_refused(self, how, mode):
        assert refuses(["colabfold_batch", *forms("--msa_mode", mode, how), "in", "out"])
        assert refuses(["colabfold_batch", *separate("--msa_mode", mode, how), "in", "out"])

    @pytest.mark.parametrize("how", HOW)
    def test_single_sequence_is_accepted_and_a_missing_value_is_not(self, how):
        assert not refuses(["colabfold_batch", *forms("--msa_mode", "single_sequence", how)])
        assert not refuses(
            ["colabfold_batch", *separate("--msa_mode", "single_sequence", how), "in", "out"]
        )
        assert refuses(["colabfold_batch", spelled("--msa_mode", how)])
        assert refuses(["colabfold_batch", spelled("--msa_mode", how), "--num-models", "5"])

    @pytest.mark.parametrize("how", HOW)
    def test_colabfold_must_name_the_mode_in_either_spelling(self, how):
        assert refuses(["colabfold_batch", "in", "out"], "af2")
        assert not refuses(
            ["colabfold_batch", *forms("--msa_mode", "single_sequence", how), "in"], "af2"
        )

    def test_the_message_names_the_option_as_it_was_written(self):
        with pytest.raises(ValueError) as caught:
            assert_no_server_flag(["run_openfold", "--use-msa-server=true"], "of3")
        assert "--use-msa-server true" in str(caught.value)


# ---------------------------------------------------------------------------
# repeated options: the last one wins, as in click and argparse
# ---------------------------------------------------------------------------


class TestTheLastOccurrenceWins:
    @pytest.mark.parametrize("first", HOW)
    @pytest.mark.parametrize("second", HOW)
    def test_off_then_on_is_a_server_call_and_on_then_off_is_not(self, first, second):
        flag = "--use_msa_server"
        off_then_on = [*forms(flag, "false", first), "--x", *forms(flag, "true", second)]
        on_then_off = [*forms(flag, "true", first), "--x", *forms(flag, "false", second)]
        assert refuses(["run_openfold", *off_then_on])
        assert not refuses(["run_openfold", *on_then_off])

    def test_the_case_of_the_report_of_issue_153(self):
        assert refuses(["run_openfold", "--use_msa_server=false", "--use-msa-server=true"])
        assert refuses(["run_openfold", "--use-msa-server=true", "--use_msa_server"]) is True
        assert not refuses(["run_openfold", "--use-msa-server=true", "--use_msa_server=false"])

    @pytest.mark.parametrize("first", HOW)
    @pytest.mark.parametrize("second", HOW)
    def test_the_same_for_use_msa(self, first, second):
        flag = "--use_msa"
        assert refuses(["p", *forms(flag, "false", first), *separate(flag, "true", second)])
        assert not refuses(["p", *forms(flag, "true", first), *separate(flag, "false", second)])
        # with paths a last true is fine, a last unreadable value is not
        words = [*forms(flag, "false", first), *forms(flag, "true", second)]
        assert not refuses(["p", *words], chains_have_msa_paths=True)
        assert refuses(
            ["p", *forms(flag, "true", first), *forms(flag, "x", second)],
            chains_have_msa_paths=True,
        )

    @pytest.mark.parametrize("first", HOW)
    @pytest.mark.parametrize("second", HOW)
    def test_the_same_for_the_msa_mode(self, first, second):
        flag = "--msa_mode"
        single = forms(flag, "single_sequence", first)
        server = forms(flag, "mmseqs2_uniref_env", second)
        assert refuses(["c", *single, *server])
        assert not refuses(["c", *server, *forms(flag, "single_sequence", second)])

    def test_three_occurrences_and_words_in_between(self):
        words = [
            "--use_msa_server=true",
            "--use_msa_server=false",
            "--model", "boltz2",
            "--use-msa-server=true",
            "--use-msa-server", "false",
        ]  # fmt: skip
        assert not refuses(["boltz", "predict", *words])
        assert refuses(["boltz", "predict", *words, "--use-msa-server"])

    def test_each_flag_has_its_own_last_occurrence(self):
        words = ["--use_msa_server=false", "--use_msa=true", "--msa_mode=single_sequence"]
        assert refuses(["tool", *words])  # use_msa=true has no path
        assert not refuses(["tool", *words], chains_have_msa_paths=True)


# ---------------------------------------------------------------------------
# what is not read
# ---------------------------------------------------------------------------


class TestWhatIsNotAnOption:
    def test_the_case_of_a_name_is_kept(self):
        assert not refuses(["tool", "--USE_MSA_SERVER=true", "--Use-Msa-Server"])

    def test_other_options_and_values_are_left_alone(self):
        words = [
            "boltz", "predict", "use_msa_server", "--use_msa_servers=true",
            "--msa_pairing_strategy", "greedy", "--use_template", "--msa-only",
            "-u", "--diffusion_samples=5",
        ]  # fmt: skip
        assert not refuses(words)

    def test_a_value_that_looks_like_an_option_name_is_not_one(self):
        assert not refuses(["tool", "--note", "use_msa_server=true", "--model", "use-msa-server"])

    def test_an_abbreviation_of_a_watched_option_is_read_as_the_option(self):
        # argparse takes an unambiguous prefix of an option
        assert refuses(["colabfold_batch", "--msa-m", "mmseqs2_uniref_env"])
        assert refuses(["colabfold_batch", "--msa_mo=mmseqs2_uniref"])
        assert not refuses(["colabfold_batch", "--msa-mod=single_sequence"])
        assert refuses(["tool", "--use_msa_s"])
        assert not refuses(["tool", "--use_msa_ser=false"])
        assert refuses(["tool", "--use_ms", "true"])  # ambiguous: counts for both
        assert not refuses(["tool", "--us", "true"])  # too short to be read as one

    def test_a_string_command_and_path_words(self, tmp_path):
        assert refuses("boltz predict in.yaml --use-msa-server=true")
        assert not refuses("boltz predict in.yaml --use-msa-server=false")
        assert refuses(["boltz", tmp_path / "in.yaml", "--use-msa-server"])


# ---------------------------------------------------------------------------
# the runners give the guard the final command, extra_args included
# ---------------------------------------------------------------------------

ONE_BINDER = dict(binder_chain="B", receptor_chain="A")
FORBID = dict(msa_policy="forbid-server", receptor_msa="single")


@pytest.fixture(autouse=True)
def _tools_are_not_probed_and_never_start(monkeypatch):
    """No version probe, and every process starter records and fails the test."""
    started = []
    for owner in (Boltz2Runner, OpenFold3Runner, ProtenixRunner, ColabFoldRunner):
        monkeypatch.setattr(owner, "version", lambda self: "0")
    monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.6.3")
    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: f"/tools/{name}")
    monkeypatch.setattr(protenix_runner, "_installed_version", lambda python_cmd=None: "2.0.0")

    def never(name):
        def start(*args, **kwargs):
            started.append(name)
            raise AssertionError(f"{name} would have started a process")

        return start

    monkeypatch.setattr(boltz2_runner, "_run_boltz_command", never("boltz"))
    monkeypatch.setattr(protenix_runner, "_run_process", never("protenix"))
    monkeypatch.setattr(af2_runner, "_run_command", never("colabfold"))
    from binding_metrics.metrics import openfold

    monkeypatch.setattr(openfold, "_run_openfold_command", never("run_openfold"))
    return started


def boltz(**extra):
    return Boltz2Runner().make_request(
        LINEAR, name="ycr", mode="predict", use_msa_server=False, **ONE_BINDER, **FORBID, **extra
    )


def openfold3(**extra):
    return OpenFold3Runner().make_request(
        LINEAR, name="q", mode="score", use_msa_server=False, **ONE_BINDER, **FORBID, **extra
    )


def protenix(**extra):
    return ProtenixRunner().make_request(
        LINEAR, name="p53", use_msa_server=False, **ONE_BINDER, **FORBID, **extra
    )


def colabfold(**extra):
    return ColabFoldRunner().make_request(
        LINEAR, name="p53", use_msa_server=False, **ONE_BINDER, **FORBID, **extra
    )


RUNNERS = {
    "boltz2": (boltz, Boltz2Runner),
    "of3": (openfold3, OpenFold3Runner),
    "protenix": (protenix, ProtenixRunner),
    "af2": (colabfold, ColabFoldRunner),
}

#: extra arguments that would switch the server on, in the spellings that the tools take
SMUGGLED = {
    "boltz2": [
        ["--use_msa_server"],
        ["--use_msa_server=true"],
        ["--use-msa-server"],
        ["--use-msa-server=true"],
        ["--use-msa-server", "true"],
    ],
    "of3": [
        ["--use_msa_server=true"],
        ["--use-msa-server=true"],
        ["--use-msa-server", "true"],
        ["--use_msa_server"],
        ["--use-msa-server"],
    ],
    "protenix": [
        ["--use-msa", "maybe"],
        ["--use_msa=maybe"],
    ],
    "af2": [
        ["--msa-mode", "mmseqs2_uniref_env"],
        ["--msa_mode", "mmseqs2_uniref_env"],
        ["--msa-mode=mmseqs2_uniref"],
        ["--msa_mode=mmseqs2_uniref"],
        ["--msa-m", "mmseqs2_uniref_env"],
    ],
}


class TestNoRunnerLetsExtraArgumentsSwitchTheServerOn:
    @pytest.mark.parametrize("model", sorted(SMUGGLED))
    def test_every_spelling_is_stopped_before_a_process_starts(
        self, model, tmp_path, _tools_are_not_probed_and_never_start
    ):
        build, runner_class = RUNNERS[model]
        for extra in SMUGGLED[model]:
            try:
                request = build(extra_args=extra)
            except ValueError:
                continue  # refused when the request is made
            for step in ("prepare", "run"):
                with pytest.raises(ValueError):
                    getattr(runner_class(), step)(request, tmp_path / f"{step}-{'-'.join(extra)}")
        assert _tools_are_not_probed_and_never_start == []

    @pytest.mark.parametrize("model", sorted(SMUGGLED))
    def test_a_request_made_by_hand_with_them_is_stopped_too(
        self, model, tmp_path, _tools_are_not_probed_and_never_start
    ):
        build, runner_class = RUNNERS[model]
        good = build()
        for extra in SMUGGLED[model]:
            by_hand = dataclasses.replace(good, options={**good.options, "extra_args": extra})
            for step in ("prepare", "run"):
                with pytest.raises(ValueError):
                    getattr(runner_class(), step)(by_hand, tmp_path / f"{step}-{'-'.join(extra)}")
        assert _tools_are_not_probed_and_never_start == []

    @pytest.mark.parametrize("model", ["boltz2", "of3"])
    def test_the_guard_and_not_the_argument_check_stops_the_hyphen_spelling(self, model, tmp_path):
        """The hyphen spelling reaches the command of Boltz-2 and OpenFold3; the guard stops it."""
        build, runner_class = RUNNERS[model]
        extra = ["--use-msa-server=true"]
        if model == "boltz2":
            request = build(extra_args=extra)  # the argument check of the runner lets it by
            with pytest.raises(ValueError, match="would make the model use an MSA server"):
                runner_class().prepare(request, tmp_path / "work")
        else:
            with pytest.raises(ValueError, match="would make the model use an MSA server"):
                build(extra_args=extra)


class TestTheGuardGetsTheFinalCommand:
    HARMLESS = {
        "boltz2": ["--no_kernels"],
        "of3": ["--log_level=INFO"],
        "protenix": ["--triatt_kernel", "torch"],
        "af2": ["--num-ensemble", "1"],
    }

    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = []
        for module in (boltz2_runner, of3_runner, protenix_runner, af2_runner):
            real = module.assert_no_server_flag

            def record(command, name, *, _real=real, **kwargs):
                calls.append((list(command), name, kwargs))
                return _real(command, name, **kwargs)

            monkeypatch.setattr(module, "assert_no_server_flag", record)
        return calls

    @pytest.mark.parametrize("model", sorted(HARMLESS))
    def test_the_extra_arguments_are_in_the_checked_command(self, model, tmp_path, recorded):
        build, runner_class = RUNNERS[model]
        extra = self.HARMLESS[model]
        runner_class().prepare(build(extra_args=extra), tmp_path / "work")
        checked = [command for command, name, _ in recorded if name == model]
        assert checked, f"the {model} runner did not call the guard"
        for command in checked:
            index = command.index(extra[0])
            assert command[index : index + len(extra)] == extra

    @pytest.mark.parametrize("model", sorted(HARMLESS))
    def test_nothing_is_checked_under_the_default_policy(self, model, tmp_path, recorded):
        _, runner_class = RUNNERS[model]
        build, _ = RUNNERS[model]
        request = build()
        default = dataclasses.replace(
            request, options={k: v for k, v in request.options.items() if k != "msa_policy"}
        )
        recorded.clear()  # the request above was made under the policy and was checked then
        runner_class().prepare(default, tmp_path / "work")
        assert [call for call in recorded if call[1] == model] == []


class TestTheArgumentChecksReadBothSpellings:
    """``extra_args`` may not repeat a flag the runner owns, whichever spelling it uses."""

    @pytest.mark.parametrize(
        "extra", [["--use-msa", "true"], ["--use-msa=false"], ["--model-name=x"]]
    )
    def test_protenix(self, extra):
        with pytest.raises(ValueError, match="extra_args cannot hold"):
            protenix(extra_args=extra)

    @pytest.mark.parametrize(
        "extra", [["--msa_mode", "single_sequence"], ["--model_type=x"], ["--num_models", "1"]]
    )
    def test_colabfold(self, extra):
        with pytest.raises(ValueError, match="extra_args may not hold"):
            colabfold(extra_args=extra)
