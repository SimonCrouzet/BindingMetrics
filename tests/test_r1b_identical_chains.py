"""The binder and the receptor cannot be the same chain (#180).

The pre-flight check reports a violation of kind ``input`` and refuses it under the policy
``error``. ``run_pipeline`` and ``run_batch`` raise ``ValueError`` before any output exists,
whatever ``on_incompatible`` says, and ``binding-metrics-run`` and ``binding-metrics-batch`` print
the message to stderr and exit 1, as for a chain that is not in the file.
"""

import sys

import pytest
from conftest import DATA_DIR

from binding_metrics.capabilities import (
    IncompatibleInputError,
    InputProfile,
    preflight,
    profile_input,
)
from binding_metrics.cli import batch, run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics._common import require_distinct_chains
from binding_metrics.preflight_cli import check_input

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
STEPS = frozenset({"interface", "electrostatics"})
MESSAGE = "peptide chain and receptor chain are both 'B'"


class TestRequireDistinctChains:
    def test_the_same_chain_raises(self):
        with pytest.raises(ValueError, match=MESSAGE):
            require_distinct_chains("B", "B")

    @pytest.mark.parametrize(("binder", "receptor"), [("B", "A"), ("B", None), (None, None)])
    def test_two_chains_or_a_role_left_to_detection_pass(self, binder, receptor):
        require_distinct_chains(binder, receptor)


class TestPreflight:
    def test_check_input_refuses_the_same_chain_under_error(self):
        outcome = check_input(YCR, "B", "B", steps=STEPS)
        assert outcome.refused
        assert outcome.block["status"] == "refused"
        (violation,) = outcome.report.violations
        assert (violation.kind, violation.name, violation.constraint) == (
            "input",
            "chains",
            "chain_roles",
        )
        assert "'B'" in violation.fact and "binder" in violation.fact
        assert "receptor" in violation.fact
        assert violation.requirement and violation.fix
        assert "'B'" in outcome.block["reason"]

    def test_the_refusal_prints_found_requires_and_fix(self):
        text = str(check_input(YCR, "B", "B", steps=STEPS).error)
        assert "Pre-flight check failed: 1 incompatibility" in text
        assert "    found:    " in text and "    requires: " in text and "    fix:      " in text

    @pytest.mark.parametrize("policy", ["skip", "warn"])
    def test_the_other_policies_report_it_and_refuse_nothing(self, policy):
        outcome = check_input(YCR, "B", "B", steps=STEPS, on_incompatible=policy)
        assert not outcome.refused
        assert outcome.block["status"] == "warn"
        assert [v.kind for v in outcome.report.violations] == ["input"]
        assert outcome.skipped_steps == {}

    def test_two_chains_have_no_such_violation(self):
        outcome = check_input(YCR, "B", "A", steps=STEPS)
        assert not outcome.refused and outcome.block["status"] == "ok"
        assert all(v.kind != "input" for v in outcome.report.violations)

    def test_a_binder_that_spans_the_receptor_chain_is_found_by_preflight(self):
        profile = InputProfile(
            binder_chains=("B", "C"), receptor_chain="C", chain_ids=("A", "B", "C")
        )
        with pytest.raises(IncompatibleInputError) as refused:
            preflight(profile, ["interface"])
        (violation,) = refused.value.violations
        assert "'C'" in violation.fact

    def test_a_hand_built_profile_with_no_receptor_passes(self):
        profile = InputProfile(binder_chains=("B",), chain_ids=("A", "B"))
        assert preflight(profile, []).compatible

    def test_the_same_chain_through_profile_input(self):
        profile = profile_input(YCR, "B", "B")
        with pytest.raises(IncompatibleInputError):
            preflight(profile, ["interface"])


class TestRunPipeline:
    @pytest.mark.parametrize("policy", ["error", "skip", "warn"])
    def test_the_same_chain_is_a_usage_error_whatever_the_policy(self, tmp_path, policy):
        out = tmp_path / "out"
        with pytest.raises(ValueError, match=MESSAGE):
            run_pipeline(
                YCR,
                out,
                skip_prep=True,
                skip_relax=True,
                metrics=STEPS,
                peptide_chain="B",
                receptor_chain="B",
                on_incompatible=policy,
            )
        assert not out.exists()

    def test_the_role_spellings_are_checked_too(self, tmp_path):
        with pytest.raises(ValueError, match=MESSAGE):
            run_pipeline(YCR, tmp_path / "out", metrics=STEPS, binder_chain="B", target_chain="B")

    def test_preflight_only_raises_as_well(self, tmp_path):
        with pytest.raises(ValueError, match=MESSAGE):
            run_pipeline(
                YCR,
                tmp_path / "out",
                metrics=STEPS,
                peptide_chain="B",
                receptor_chain="B",
                preflight_only=True,
            )

    def test_two_chains_run(self, tmp_path):
        results = run_pipeline(
            YCR,
            tmp_path / "out",
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"interface"}),
            peptide_chain="B",
            receptor_chain="A",
        )
        assert results["preflight"]["status"] == "ok"


class TestCommandLine:
    @staticmethod
    def _run(monkeypatch, tmp_path, *chains):
        argv = [
            "binding-metrics-run", "-i", str(YCR), "-o", str(tmp_path / "out"),
            "--skip-prep", "--skip-relax", "--metrics", "interface", *chains,
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as stop:
            run.main()
        return stop.value.code

    @pytest.mark.parametrize(
        "spelling",
        [
            ("--peptide-chain", "B", "--receptor-chain", "B"),
            ("--binder-chain", "B", "--target-chain", "B"),
        ],
    )
    def test_run_exits_1_with_the_message_on_stderr(self, monkeypatch, tmp_path, capsys, spelling):
        assert self._run(monkeypatch, tmp_path, *spelling) == 1
        captured = capsys.readouterr()
        assert f"ERROR: {MESSAGE}" in captured.err
        assert "different chains" in captured.err
        assert not (tmp_path / "out").exists()

    def test_batch_exits_1_with_the_message_on_stderr(self, monkeypatch, tmp_path, capsys):
        folder = tmp_path / "in"
        folder.mkdir()
        (folder / "a.pdb").write_text(YCR.read_text(encoding="utf-8"), encoding="utf-8")
        argv = [
            "binding-metrics-batch", "--input-dir", str(folder),
            "--output-csv", str(tmp_path / "out" / "m.csv"),
            "--skip-prep", "--skip-relax", "--metrics", "interface",
            "--peptide-chain", "B", "--receptor-chain", "B",
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as stop:
            batch.main()
        assert stop.value.code == 1
        assert f"ERROR: {MESSAGE}" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()


class TestRunBatch:
    def test_the_same_chain_raises_before_any_sample(self, tmp_path):
        out = tmp_path / "out"
        with pytest.raises(ValueError, match=MESSAGE):
            run_batch(
                [YCR], out, skip_prep=True, skip_relax=True, metrics={"interface"},
                peptide_chain="B", receptor_chain="B",
            )  # fmt: skip
        assert not out.exists()

    def test_the_role_spellings_are_checked_too(self, tmp_path):
        with pytest.raises(ValueError, match=MESSAGE):
            run_batch([YCR], tmp_path / "out", binder_chain="B", target_chain="B")
