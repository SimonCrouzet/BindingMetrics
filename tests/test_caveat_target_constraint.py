"""The caveat for a prediction whose target has neither a template nor an MSA.

A co-folding run has to give the model some constraint on the fold of the target: the target is a
template (pinned when locked) and a cached target MSA is optional. The one configuration that gives
none is the mode ``predict`` (no template on the target) with the MSA server off and a receptor MSA
that is absent or ``"single"``. The pre-flight check reports it as a caveat (a warning of the
report), never as a violation, so ``--on-incompatible error`` does not refuse because of it.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest
from conftest import DATA_DIR

from binding_metrics import capabilities
from binding_metrics.capabilities import (
    MODES,
    MSA_POLICIES,
    POLICIES,
    Capabilities,
    InputProfile,
    preflight,
)
from binding_metrics.cli import batch, run
from binding_metrics.cli.run import run_pipeline
from binding_metrics.predictors import _msa
from binding_metrics.predictors.registry import PARSERS

LINEAR = DATA_DIR / "example_linear_p53_1YCR.pdb"  # peptide B, receptor A

#: The caveat, without the name of the predictor that ``preflight`` puts in front.
CAVEAT = (
    "the target fold is unconstrained (no template and no MSA), so the binder placement and the "
    "confidence scores say little: give the target as a template (--prediction-mode refold or "
    "score), give a receptor MSA (--prediction-receptor-msa PATH), or leave the MSA server on if "
    "the sequence may be sent out"
)
#: Enough of the sentence to find it in a printed plan or a CLI output.
MARK = "the target fold is unconstrained (no template and no MSA)"

TEMPLATED_MODES = tuple(mode for mode in MODES if mode != "predict")

PROFILES = {
    "peptide": InputProfile("B", "A", n_binder_residues=12, binder_type="peptide"),
    "miniprotein": InputProfile("B", "A", n_binder_residues=80, binder_type="miniprotein"),
    "nanobody": InputProfile("B", "A", n_binder_residues=120, binder_type="nanobody"),
    "antibody": InputProfile(("H", "L"), "A", n_binder_residues=440, binder_type="antibody"),
    "unknown": InputProfile("B", "A", n_binder_residues=12),
}


@pytest.fixture
def alignment(tmp_path):
    path = tmp_path / "receptor.a3m"
    path.write_text(">query\nSQETFSDLWKLLPEN\n>hit\nSQETFSDLWKLLPEN\n", encoding="utf-8")
    return path


def caveats_of(report) -> list[str]:
    return [text for text in report.warnings if MARK in text]


# ---------------------------------------------------------------------------
# the condition
# ---------------------------------------------------------------------------


class TestTheConditionTable:
    """Every combination of mode, server, receptor MSA and binder type, against the rule."""

    RECEPTOR_MSAS = ("none", "single", "file", "path")

    @pytest.mark.parametrize(
        "binder_type, mode, server, kind",
        list(itertools.product(PROFILES, (None, *MODES), (None, True, False), RECEPTOR_MSAS)),
    )
    def test_the_caveat_is_there_exactly_for_predict_without_server_or_alignment(
        self, binder_type, mode, server, kind, alignment
    ):
        receptor_msa = {
            "none": None,
            "single": "single",
            "file": str(alignment),
            "path": alignment,
        }[kind]
        report = preflight(
            PROFILES[binder_type],
            predictor=Capabilities(),
            mode=mode,
            msa_server=server,
            receptor_msa=receptor_msa,
        )
        expected = mode == "predict" and server is False and kind in ("none", "single")
        assert bool(caveats_of(report)) is expected
        assert report.compatible and report.violations == ()  # a caveat, never a violation

    def test_the_default_run_is_not_a_caveat(self):
        # the server is on unless the user turns it off, and the default mode templates the target
        for mode in MODES:
            report = preflight(
                PROFILES["peptide"], predictor=Capabilities(), mode=mode, msa_server=True
            )
            assert caveats_of(report) == []

    @pytest.mark.parametrize("mode", TEMPLATED_MODES)
    def test_a_templated_or_pinned_mode_is_never_a_caveat(self, mode):
        report = preflight(
            PROFILES["peptide"],
            predictor=Capabilities(),
            mode=mode,
            msa_server=False,
            receptor_msa="single",
        )
        assert caveats_of(report) == []

    def test_an_unknown_server_or_mode_does_not_decide(self):
        # the caller did not say: nothing is claimed
        for mode, server in ((None, False), ("predict", None), (None, None)):
            report = preflight(
                PROFILES["peptide"], predictor=Capabilities(), mode=mode, msa_server=server
            )
            assert caveats_of(report) == []

    def test_a_check_without_a_predictor_has_no_target_to_constrain(self):
        report = preflight(
            PROFILES["peptide"], ["omega"], mode="predict", msa_server=False, receptor_msa=None
        )
        assert caveats_of(report) == []

    def test_a_file_that_is_called_single_is_an_alignment(self, tmp_path):
        # a str "single" is the word, a Path is a file whatever its name (as in _msa)
        named_single = tmp_path / "single"
        named_single.write_text(">q\nSQETF\n", encoding="utf-8")
        report = preflight(
            PROFILES["peptide"],
            predictor=Capabilities(),
            mode="predict",
            msa_server=False,
            receptor_msa=named_single,
        )
        assert caveats_of(report) == []

    def test_the_word_is_the_one_the_runners_use(self):
        assert capabilities._SINGLE_SEQUENCE == _msa.SINGLE_SEQUENCE == "single"

    @pytest.mark.parametrize("name", sorted(PARSERS))
    def test_every_registered_model_gets_it_in_predict_without_server(self, name):
        report = preflight(PROFILES["peptide"], predictor=name, policy="warn", mode="predict",
                           msa_server=False)  # fmt: skip
        found = caveats_of(report)
        assert len(found) == 1 and found[0].endswith(CAVEAT)
        assert found[0].startswith("predictor ")

    @pytest.mark.parametrize("name", sorted(PARSERS))
    @pytest.mark.parametrize("mode", TEMPLATED_MODES)
    def test_a_report_that_does_not_fire_is_the_report_of_a_caller_who_said_nothing(
        self, name, mode
    ):
        said = preflight(PROFILES["peptide"], predictor=name, policy="warn", mode=mode,
                         msa_server=False)  # fmt: skip
        silent = preflight(PROFILES["peptide"], predictor=name, policy="warn", mode=mode)
        assert said.to_dict() == silent.to_dict() and said.format() == silent.format()


# ---------------------------------------------------------------------------
# the report and the plan
# ---------------------------------------------------------------------------


class TestTheReport:
    def fired(self, **extra):
        return preflight(
            PROFILES["peptide"], predictor=Capabilities(), mode="predict", msa_server=False,
            **extra,
        )  # fmt: skip

    def test_the_dict_carries_it_with_the_other_warnings(self):
        data = self.fired().to_dict()
        assert data["warnings"] == [f"predictor (unnamed): {CAVEAT}"]
        assert data["compatible"] is True and data["violations"] == []
        assert data["mode"] == "predict" and data["notes"] == []

    def test_the_dict_has_the_keys_of_every_other_report(self):
        other = preflight(
            PROFILES["peptide"], predictor=Capabilities(), mode="refold", msa_server=False
        ).to_dict()
        assert list(self.fired().to_dict()) == list(other)

    def test_the_text_of_the_report_lists_it_under_warnings(self):
        text = self.fired().format()
        assert text.endswith(f"Warnings:\n  - predictor (unnamed): {CAVEAT}")
        assert text.startswith("Pre-flight check: the input fits everything requested")

    def test_a_given_alignment_takes_it_away(self, alignment):
        report = self.fired(receptor_msa=str(alignment))
        assert report.warnings == ()
        assert "Warnings" not in report.format()

    def test_the_sentence_says_what_is_missing_and_how_to_fix_it(self):
        assert CAVEAT.count(".") == 0  # one sentence
        for part in ("no template and no MSA", "binder placement", "confidence scores",
                     "--prediction-mode refold or score", "receptor MSA",
                     "MSA server on"):  # fmt: skip
            assert part in CAVEAT

    def test_it_is_a_warning_under_every_policy(self):
        for policy in POLICIES:
            report = preflight(
                PROFILES["peptide"], predictor=Capabilities(), policy=policy, mode="predict",
                msa_server=False,
            )  # fmt: skip
            assert caveats_of(report) and report.compatible and not report.refused


class TestForbidServer:
    def test_single_sequence_under_the_policy_is_allowed_and_unconstrained(self):
        report = preflight(
            PROFILES["peptide"],
            predictor=Capabilities(),
            mode="predict",
            msa_policy="forbid-server",
            msa_server=False,
            receptor_msa="single",
        )
        assert report.compatible and not report.refused  # the policy allows it
        assert [text.endswith(CAVEAT) for text in report.warnings] == [True]
        assert "MSA policy: forbid-server" in report.format()
        assert report.to_dict()["msa_policy"] == "forbid-server"

    def test_a_given_alignment_under_the_policy_is_constrained(self, alignment):
        report = preflight(
            PROFILES["peptide"],
            predictor=Capabilities(),
            mode="predict",
            msa_policy="forbid-server",
            msa_server=False,
            receptor_msa=str(alignment),
        )
        assert caveats_of(report) == []

    def test_a_server_under_the_policy_is_the_violation_and_not_the_caveat(self):
        from binding_metrics.capabilities import IncompatibleInputError

        with pytest.raises(IncompatibleInputError) as refused:
            preflight(
                PROFILES["peptide"], predictor=Capabilities(), mode="predict",
                msa_policy="forbid-server", msa_server=True, receptor_msa="single",
            )  # fmt: skip
        assert caveats_of(refused.value.report) == []
        assert [v.constraint for v in refused.value.violations] == ["msa_policy"]

    @pytest.mark.parametrize("policy", MSA_POLICIES)
    def test_the_rule_is_the_same_for_both_msa_policies(self, policy):
        report = preflight(
            PROFILES["peptide"], predictor=Capabilities(), mode="predict", msa_policy=policy,
            msa_server=False,
        )  # fmt: skip
        assert len(caveats_of(report)) == 1


# ---------------------------------------------------------------------------
# the command line and run_pipeline
# ---------------------------------------------------------------------------


def cli(monkeypatch, capsys, tmp_path, *arguments):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binding-metrics-run",
            "--input", str(LINEAR),
            "--output-dir", str(tmp_path / "out"),
            "--metrics", "openfold",
            "--predictor", "boltz2",
            "--preflight-only",
            *arguments,
        ],
    )  # fmt: skip
    try:
        run.main()
        code = 0
    except SystemExit as exit_info:
        code = exit_info.code
    out = capsys.readouterr().out
    assert not (tmp_path / "out").exists()  # --preflight-only creates nothing
    return code, out


class TestTheCommandLine:
    PREDICT_NO_SERVER = ("--prediction-mode", "predict", "--prediction-no-msa-server")

    def test_predict_without_server_prints_the_caveat_in_the_plan(
        self, monkeypatch, capsys, tmp_path
    ):
        code, out = cli(monkeypatch, capsys, tmp_path, *self.PREDICT_NO_SERVER)
        assert code == 0
        assert "Mode: predict (sequences only)" in out
        assert f"Warnings:\n  - predictor Boltz-2 2.2.1: {CAVEAT}" in out

    def test_the_default_run_prints_none(self, monkeypatch, capsys, tmp_path):
        code, out = cli(monkeypatch, capsys, tmp_path)
        assert code == 0 and MARK not in out and "Warnings" not in out

    def test_predict_with_the_server_on_prints_none(self, monkeypatch, capsys, tmp_path):
        code, out = cli(monkeypatch, capsys, tmp_path, "--prediction-mode", "predict")
        assert code == 0 and MARK not in out and "Mode: predict (sequences only)" in out

    @pytest.mark.parametrize("mode", TEMPLATED_MODES)
    def test_a_templated_mode_prints_none_even_without_the_server(
        self, mode, monkeypatch, capsys, tmp_path
    ):
        code, out = cli(
            monkeypatch, capsys, tmp_path, "--prediction-mode", mode, "--prediction-no-msa-server"
        )
        assert code == 0 and MARK not in out

    @pytest.mark.parametrize("policy", POLICIES)
    def test_no_value_of_on_incompatible_refuses_because_of_it(
        self, policy, monkeypatch, capsys, tmp_path
    ):
        code, out = cli(
            monkeypatch, capsys, tmp_path, *self.PREDICT_NO_SERVER, "--on-incompatible", policy
        )
        assert code == 0 and MARK in out and "Pre-flight check failed" not in out

    def test_a_receptor_alignment_takes_it_away(self, monkeypatch, capsys, tmp_path, alignment):
        code, out = cli(
            monkeypatch, capsys, tmp_path, *self.PREDICT_NO_SERVER,
            "--prediction-receptor-msa", str(alignment),
        )  # fmt: skip
        assert code == 0 and MARK not in out and f"Receptor MSA: {alignment}" in out

    def test_single_is_not_an_alignment(self, monkeypatch, capsys, tmp_path):
        code, out = cli(
            monkeypatch, capsys, tmp_path, *self.PREDICT_NO_SERVER,
            "--prediction-receptor-msa", "single",
        )  # fmt: skip
        assert code == 0 and MARK in out and "Receptor MSA: single" in out

    def test_forbid_server_with_single_allows_the_run_and_says_the_target_is_free(
        self, monkeypatch, capsys, tmp_path
    ):
        code, out = cli(
            monkeypatch, capsys, tmp_path, *self.PREDICT_NO_SERVER,
            "--prediction-msa-policy", "forbid-server", "--prediction-receptor-msa", "single",
        )  # fmt: skip
        assert code == 0
        assert "MSA policy: forbid-server" in out and MARK in out
        assert "Pre-flight check failed" not in out

    def test_a_model_whose_default_mode_is_predict_gets_it_too(self, monkeypatch, capsys, tmp_path):
        # ColabFold has no template mode of its own here: its default mode is predict
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-run", "--input", str(LINEAR), "--output-dir", str(tmp_path / "o"),
             "--metrics", "openfold", "--predictor", "af2", "--preflight-only",
             "--prediction-no-msa-server"],
        )  # fmt: skip
        with pytest.raises(SystemExit) as exit_info:
            run.main()
        out = capsys.readouterr().out
        assert exit_info.value.code == 0 and MARK in out


class TestRunPipeline:
    def block(self, **kwargs):
        return run_pipeline(
            LINEAR, Path("unused"), metrics=frozenset({"openfold"}), predictor="boltz2",
            preflight_only=True, **kwargs,
        )["preflight"]  # fmt: skip

    def test_the_block_is_a_warning_with_the_caveat_in_the_report_and_the_plan(self):
        block = self.block(prediction_mode="predict", prediction_use_msa_server=False)
        assert block["status"] == "warn" and block["mode"] == "predict"
        assert block["reason"].endswith(CAVEAT)
        assert [text.endswith(CAVEAT) for text in block["report"]["warnings"]] == [True]
        assert block["report"]["violations"] == []
        assert block["plan"].endswith(f"Warnings:\n  - predictor Boltz-2 2.2.1: {CAVEAT}")

    def test_the_default_block_is_unchanged(self):
        block = self.block()
        assert block["status"] == "ok" and block["reason"] == ""
        assert block["report"]["warnings"] == [] and "Warnings" not in block["plan"]

    def test_a_caller_that_runs_the_model_later_passes_the_alignment_with_the_policy(
        self, alignment
    ):
        # the batch checks the model of each sample in a worker that has no predictor of its own
        # and is told the policy, the server and now the receptor MSA of the model step
        def worker(preflight_msa):
            return run_pipeline(
                LINEAR, Path("unused"), metrics=frozenset({"interface"}), skip_relax=True,
                preflight_only=True, preflight_model=("boltz2", "prediction", False, "predict"),
                preflight_msa=preflight_msa,
            )["preflight"]  # fmt: skip

        without = worker(("default", False, None))
        assert without["status"] == "warn" and without["report"]["warnings"]
        assert worker(("default", False))["report"]["warnings"] == without["report"]["warnings"]
        given = worker(("default", False, str(alignment)))
        assert given["status"] == "ok" and given["report"]["warnings"] == []
        single = worker(("default", False, "single"))
        assert single["report"]["warnings"] == without["report"]["warnings"]
        assert worker(("default", True, None))["report"]["warnings"] == []


class TestTheBatch:
    def plan(self, tmp_path, **kwargs):
        rows = batch.run_batch(
            [LINEAR], tmp_path / "out", metrics=["openfold"], predictor="boltz2",
            preflight_only=True, **kwargs,
        )  # fmt: skip
        assert not (tmp_path / "out").exists()
        assert len(rows) == 1 and rows[0]["batch_status"] == "ok"
        return rows[0]

    def test_the_row_of_a_sample_says_it(self, tmp_path):
        row = self.plan(tmp_path, prediction_mode="predict", prediction_use_msa_server=False)
        assert row["preflight_status"] == "warn"
        assert row["preflight_plan"].endswith(f"Warnings:\n  - predictor Boltz-2 2.2.1: {CAVEAT}")

    def test_the_default_row_does_not(self, tmp_path):
        row = self.plan(tmp_path)
        assert row["preflight_status"] == "ok" and MARK not in row["preflight_plan"]

    def test_an_alignment_for_the_batch_takes_it_away(self, tmp_path, alignment):
        row = self.plan(
            tmp_path, prediction_mode="predict", prediction_use_msa_server=False,
            prediction_receptor_msa=str(alignment),
        )  # fmt: skip
        assert row["preflight_status"] == "ok" and MARK not in row["preflight_plan"]


# ---------------------------------------------------------------------------
# the pose set
# ---------------------------------------------------------------------------


class TestThePoseSet:
    def test_the_pose_set_pre_flight_does_not_know_the_mode_or_the_msa(self, monkeypatch):
        """``score_pose_set`` reads predictions made elsewhere.

        Its pre-flight call names no predictor, no mode and no MSA, and the poses do not record
        whether the MSA server was used (their provenance has the mode, the MSA policy and the
        receptor alignment of the request, not the server), so the condition cannot be decided and
        the check says nothing about the target.
        """
        from binding_metrics.poses.score import score_pose_set
        from binding_metrics.testing import make_synthetic_poses

        calls = []
        original = capabilities.preflight

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(capabilities, "preflight", spy)
        synthetic = make_synthetic_poses()
        result = score_pose_set(synthetic.poses[:2], "B", "A", metrics=["hbonds"])
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert not {"mode", "msa_server", "receptor_msa"} & set(kwargs)
        assert len(calls[0][0]) == 2  # the profile and the metrics: no predictor
        report = result["preflight"]
        assert report["mode"] is None and report["predictors"] == []
        assert not any(MARK in text for text in report["warnings"])
