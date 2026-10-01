"""``run_pipeline`` and ``binding-metrics-run`` with the pre-flight check.

The check runs first: before the output directory, the provenance probe, preparation,
relaxation, any model run and the prediction store. Tripwires stand in for all of those and
must not be called when the input is refused. The model, when a test needs one, is the stub of
the FEAT-C tests.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from binding_metrics.capabilities import IncompatibleInputError
from binding_metrics.cli import run
from binding_metrics.cli.run import run_pipeline
from binding_metrics.preflight_cli import model_step_of, steps_that_run

DATA = Path(__file__).resolve().parent.parent / "data"
LINEAR = DATA / "example_linear_p53_1YCR.pdb"  # peptide B, receptor A
BICYCLE = DATA / "example_bicyclic_sfti1_3P8F.cif"  # head-to-tail and disulfide, binder I
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"  # head-to-tail, D and N-methyl, binder C
ONE_CHAIN = DATA / "example_lactam_somatostatin_1XY4.cif"  # a single chain


class Tripwires:
    """Replaces everything that must not run before a refusal and records any call."""

    def __init__(self, monkeypatch, allow=()):
        self.calls: list[str] = []
        targets = {
            "cli.run.collect_provenance": (run, "collect_provenance"),
            "core.system.prep_structure": ("binding_metrics.core.system", "prep_structure"),
            "protocols.relaxation.ImplicitRelaxation": (
                "binding_metrics.protocols.relaxation",
                "ImplicitRelaxation",
            ),
            "metrics.openfold.run_openfold_scoring": (
                "binding_metrics.metrics.openfold",
                "run_openfold_scoring",
            ),
            "metrics.openfold.run_openfold_refolding": (
                "binding_metrics.metrics.openfold",
                "run_openfold_refolding",
            ),
            "metrics.openfold.run_openfold_batched": (
                "binding_metrics.metrics.openfold",
                "run_openfold_batched",
            ),
            "cli.run.run_single_prediction": (run, "run_single_prediction"),
            "cli.prediction.make_store": ("binding_metrics.cli.prediction", "make_store"),
            "cli.prediction.make_session": ("binding_metrics.cli.prediction", "make_session"),
            "cli.prediction.make_runner": ("binding_metrics.cli.prediction", "make_runner"),
        }
        for label, (owner, name) in targets.items():
            if label in allow:
                continue
            module = importlib.import_module(owner) if isinstance(owner, str) else owner
            monkeypatch.setattr(module, name, self._wire(label), raising=True)

    def _wire(self, label):
        def tripwire(*args, **kwargs):
            self.calls.append(label)
            raise AssertionError(f"{label} was called before the pre-flight check refused")

        return tripwire


def pipeline(path, output_dir, **kwargs):
    kwargs.setdefault("skip_prep", True)
    kwargs.setdefault("skip_relax", True)
    kwargs.setdefault("openfold_conda_env", None)
    return run_pipeline(path, output_dir, **kwargs)


class TestNothingRunsBeforeARefusal:
    def test_a_disulfide_binder_is_refused_for_the_openfold3_step(self, tmp_path, monkeypatch):
        tripwires = Tripwires(monkeypatch)
        out = tmp_path / "out"
        with pytest.raises(IncompatibleInputError) as caught:
            run_pipeline(BICYCLE, out, metrics=frozenset({"openfold"}), openfold_conda_env=None)
        assert tripwires.calls == []
        assert not out.exists()  # not even the output directory
        message = str(caught.value)
        assert "predictor OpenFold3 0.5.0: closures" in message
        assert "a disulfide bond (CYS 3.SG - CYS 11.SG)" in message

    def test_the_relaxation_and_prep_do_not_start_either(self, tmp_path, monkeypatch):
        tripwires = Tripwires(monkeypatch)
        with pytest.raises(IncompatibleInputError):
            run_pipeline(
                BICYCLE,
                tmp_path / "out",
                metrics=frozenset({"openfold", "interface"}),
                openfold_conda_env=None,
            )
        assert tripwires.calls == []

    def test_the_predictor_route_is_refused_before_the_store_is_touched(
        self, tmp_path, monkeypatch
    ):
        tripwires = Tripwires(monkeypatch)
        out = tmp_path / "out"
        with pytest.raises(IncompatibleInputError) as caught:
            run_pipeline(
                BICYCLE,
                out,
                metrics=frozenset({"openfold"}),
                predictor="of3",
                skip_prep=True,
                skip_relax=True,
            )
        assert tripwires.calls == [] and not out.exists()
        assert "metric 'prediction'" not in str(caught.value)  # the metric itself has no problem
        assert [v.kind for v in caught.value.violations] == ["predictor"]

    def test_a_metric_that_needs_a_receptor_is_refused_for_a_single_chain(self, tmp_path):
        with pytest.raises(IncompatibleInputError) as caught:
            pipeline(
                ONE_CHAIN, tmp_path / "out", metrics=frozenset({"interface", "electrostatics"})
            )
        assert {v.name for v in caught.value.violations} == {"interface", "coulomb"}
        assert not (tmp_path / "out").exists()

    def test_every_problem_is_in_one_message(self, tmp_path):
        with pytest.raises(IncompatibleInputError) as caught:
            pipeline(
                ONE_CHAIN,
                tmp_path / "out",
                metrics=frozenset({"interface", "electrostatics", "geometry"}),
            )
        names = {v.name for v in caught.value.violations}
        assert names == {"interface", "coulomb", "shape_complementarity"}
        assert str(caught.value).count("no receptor chain was given") == 3

    def test_the_relaxation_is_checked_when_it_would_run(self, tmp_path, monkeypatch):
        tripwires = Tripwires(monkeypatch)
        from tests.test_pre_structures import build_chain

        # a sulfur-carbon link between two side chains cannot be patched by the force field
        atoms = build_chain(
            ["ALA", "CYS", "GLY", "GLY", "GLY", "XAA", "ALA"],
            "B",
            side_chain_atoms={"XAA": ("CB", "CE")},
            bonds=[(1, "SG", 5, "CE")],
        )
        import biotite.structure.io.pdb as pdb_io

        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, atoms)
        path = tmp_path / "thioether.pdb"
        pdb_file.write(str(path))
        with pytest.raises(IncompatibleInputError) as caught:
            run_pipeline(
                path, tmp_path / "out", skip_prep=True, skip_relax=False, metrics=frozenset()
            )
        (violation,) = caught.value.violations
        assert (violation.name, violation.constraint) == ("md_implicit", "closures")
        assert tripwires.calls == []
        # with the relaxation skipped there is nothing to refuse
        result = run_pipeline(
            path, tmp_path / "out2", skip_prep=True, skip_relax=True, metrics=frozenset(),
            preflight_only=True,
        )  # fmt: skip
        assert result["preflight"]["status"] == "ok"


class TestPolicies:
    def test_skip_records_the_left_out_steps_and_runs_the_rest(self, tmp_path):
        results = pipeline(
            ONE_CHAIN,
            tmp_path / "out",
            metrics=frozenset({"interface", "electrostatics", "geometry"}),
            on_incompatible="skip",
        )
        assert results["preflight"]["status"] == "skipped"
        for step in ("interface", "electrostatics"):
            assert results[step]["skipped"] is True
            assert "no receptor chain was given" in results[step]["reason"]
        geometry = results["geometry"]
        assert geometry["shape_complementarity"]["skipped"] is True
        assert "ramachandran_favoured_pct" in geometry["ramachandran"]  # computed for real
        assert "omega_mean_dev" in geometry["omega"]

    def test_a_step_left_out_is_not_a_failure(self, tmp_path):
        results = pipeline(
            ONE_CHAIN,
            tmp_path / "out",
            metrics=frozenset({"interface"}),
            on_incompatible="skip",
        )
        assert run._collect_failures(results) == []

    def test_warn_runs_everything_and_records_the_problem(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="binding_metrics.capabilities"):
            results = pipeline(
                ONE_CHAIN,
                tmp_path / "out",
                metrics=frozenset({"electrostatics"}),
                on_incompatible="warn",
            )
        assert results["preflight"]["status"] == "warn"
        assert "running anyway" in caplog.text
        assert "skipped" not in results["electrostatics"]  # it ran, as it did before

    def test_a_compatible_input_is_ok_and_carries_the_decision(self, tmp_path):
        results = pipeline(LINEAR, tmp_path / "out", metrics=frozenset({"interface", "geometry"}))
        block = results["preflight"]
        assert block["status"] == "ok" and block["reason"] == ""
        assert block["policy"] == "error"
        assert block["report"]["profile"]["binder_type"] == "peptide"
        json.dumps(block)  # JSON-ready

    def test_the_skip_of_a_model_step_is_recorded_and_the_model_never_starts(
        self, tmp_path, monkeypatch
    ):
        # the run goes on, so its provenance block is legitimate
        tripwires = Tripwires(monkeypatch, allow={"cli.run.collect_provenance"})
        results = pipeline(
            BICYCLE,
            tmp_path / "out",
            metrics=frozenset({"openfold", "geometry"}),
            on_incompatible="skip",
        )
        assert tripwires.calls == []
        assert results["openfold"]["skipped"] is True
        assert "a disulfide bond" in results["openfold"]["reason"]
        assert results["preflight"]["status"] == "skipped"
        assert "ramachandran_favoured_pct" in results["geometry"]["ramachandran"]

    def test_the_plan_does_not_list_the_model_step_as_run_when_its_predictor_is_left_out(
        self, tmp_path
    ):
        results = run_pipeline(
            BICYCLE,
            tmp_path / "out",
            skip_relax=True,
            metrics=frozenset({"openfold", "interface"}),
            on_incompatible="skip",
            preflight_only=True,
        )
        plan = results["preflight"]["plan"]
        runs = next(line for line in plan.splitlines() if line.startswith("Runs:"))
        assert "interface" in runs and "openfold" not in runs
        assert "Left out: predictor OpenFold3" in plan

    def test_the_binder_type_that_was_given_reaches_the_profile(self, tmp_path):
        results = pipeline(
            LINEAR, tmp_path / "out", metrics=frozenset({"interface"}), binder_type="nanobody"
        )
        profile = results["preflight"]["report"]["profile"]
        assert (profile["binder_type"], profile["binder_type_source"]) == ("nanobody", "given")

    def test_the_options_are_validated(self, tmp_path):
        with pytest.raises(ValueError, match="binder_type must be one of"):
            pipeline(LINEAR, tmp_path / "out", binder_type="protein")
        with pytest.raises(ValueError, match="on_incompatible must be one of"):
            pipeline(LINEAR, tmp_path / "out", on_incompatible="ignore")


class TestAnOutputMadeElsewhere:
    def test_a_prediction_dir_only_warns_whatever_the_policy(self, tmp_path, monkeypatch):
        # AlphaFold2 declares no ring closure; the prediction was made elsewhere, so the model
        # limits warn and the run goes on to read it
        calls = []

        def fake_prediction(*args, **kwargs):
            calls.append((args, kwargs))
            return {"model": "af2"}, {}

        monkeypatch.setattr(run, "run_single_prediction", fake_prediction)
        results = run_pipeline(
            CYCLOSPORIN,
            tmp_path / "out",
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            predictor="af2",
            prediction_dir=tmp_path / "given",
        )
        assert len(calls) == 1
        block = results["preflight"]
        assert block["status"] == "warn"
        assert block["report"]["predictor_policy"] == "warn"
        assert "AlphaFold2 / ColabFold" in block["reason"]

    def test_the_same_model_run_from_here_is_refused(self, tmp_path, monkeypatch):
        tripwires = Tripwires(monkeypatch)
        with pytest.raises(IncompatibleInputError):
            run_pipeline(
                BICYCLE,
                tmp_path / "out",
                skip_prep=True,
                skip_relax=True,
                metrics=frozenset({"openfold"}),
                predictor="of3",
            )
        assert tripwires.calls == []


class TestTheStepsThatAreChecked:
    def test_the_steps_that_run(self):
        assert steps_that_run({"interface", "geometry", "openfold"}, skip_relax=True) == {
            "interface",
            "geometry",
        }
        assert steps_that_run({"interface"}, skip_relax=False) == {"interface", "relax"}
        assert steps_that_run({"interface"}, skip_relax=False, custom_relaxer=True) == {"interface"}

    def test_the_model_step(self):
        openfold = frozenset({"openfold"})
        assert model_step_of(None, None, openfold) == ("of3", "openfold", False)
        assert model_step_of("boltz2", None, openfold) == ("boltz2", "prediction", False)
        assert model_step_of("boltz2", Path("d"), openfold) == ("boltz2", "prediction", True)
        assert model_step_of("boltz2", None, frozenset({"interface"})) is None

    def test_a_missing_dockq_reference_is_not_a_refusal(self, tmp_path):
        # the pipeline already skips dockq with a warning; the check leaves it alone
        results = pipeline(LINEAR, tmp_path / "out", metrics=frozenset({"dockq"}))
        assert results["dockq"] == {"skipped": True}
        assert results["preflight"]["status"] == "ok"


class TestPreflightOnly:
    def test_it_returns_the_plan_and_creates_nothing(self, tmp_path, monkeypatch):
        tripwires = Tripwires(monkeypatch)
        out = tmp_path / "out"
        results = run_pipeline(BICYCLE, out, metrics=frozenset({"openfold"}), preflight_only=True)
        assert set(results) == {"sample_id", "input", "preflight"}
        block = results["preflight"]
        assert block["status"] == "refused"
        assert block["plan"].startswith("Pre-flight check failed: 1 incompatibility")
        assert tripwires.calls == [] and not out.exists()

    def test_a_compatible_input_gets_a_plan_that_says_what_runs(self, tmp_path):
        results = run_pipeline(
            LINEAR,
            tmp_path / "out",
            skip_relax=True,
            metrics=frozenset({"interface", "geometry"}),
            preflight_only=True,
        )
        assert results["preflight"]["status"] == "ok"
        assert (
            "Runs: metrics interface, ramachandran, omega, shape_complementarity"
            in (results["preflight"]["plan"])
        )


class TestTheCommandLine:
    def _main(self, monkeypatch, capsys, *argv):
        monkeypatch.setattr(sys, "argv", ["binding-metrics-run", *argv])
        try:
            run.main()
            code = 0  # a finished run returns
        except SystemExit as exit_info:
            code = exit_info.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_the_flags_and_their_defaults(self):
        import argparse

        from binding_metrics.preflight_cli import add_preflight_args

        parser = argparse.ArgumentParser()
        add_preflight_args(parser)
        args = parser.parse_args([])
        assert (args.binder_type, args.on_incompatible, args.preflight_only) == (
            "auto",
            "error",
            False,
        )
        for choice in ("peptide", "miniprotein", "nanobody", "antibody"):
            assert parser.parse_args(["--binder-type", choice]).binder_type == choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--on-incompatible", "ignore"])

    def test_preflight_only_prints_the_plan_and_exits_1_when_something_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        code, out, _ = self._main(
            monkeypatch,
            capsys,
            "--input", str(BICYCLE),
            "--output-dir", str(tmp_path / "out"),
            "--metrics", "openfold",
            "--preflight-only",
        )  # fmt: skip
        assert code == 1
        assert "Pre-flight check failed" in out and "a disulfide bond" in out
        assert not (tmp_path / "out").exists()

    def test_preflight_only_exits_0_for_a_compatible_input(self, tmp_path, monkeypatch, capsys):
        code, out, _ = self._main(
            monkeypatch,
            capsys,
            "--input", str(LINEAR),
            "--output-dir", str(tmp_path / "out"),
            "--metrics", "interface",
            "--skip-relax",
            "--preflight-only",
        )  # fmt: skip
        assert code == 0 and "fits everything requested" in out

    def test_a_refusal_is_a_message_and_status_1_not_a_traceback(
        self, tmp_path, monkeypatch, capsys
    ):
        code, _, err = self._main(
            monkeypatch,
            capsys,
            "--input", str(ONE_CHAIN),
            "--output-dir", str(tmp_path / "out"),
            "--metrics", "interface",
            "--skip-prep", "--skip-relax",
        )  # fmt: skip
        assert code == 1
        assert err.startswith("ERROR: Pre-flight check failed") and "Traceback" not in err

    def test_skip_on_the_command_line_runs_the_rest_and_writes_the_reason(
        self, tmp_path, monkeypatch, capsys
    ):
        out_dir = tmp_path / "out"
        code, _, _ = self._main(
            monkeypatch,
            capsys,
            "--input", str(ONE_CHAIN),
            "--output-dir", str(out_dir),
            "--metrics", "interface,geometry",
            "--skip-prep", "--skip-relax",
            "--on-incompatible", "skip",
        )  # fmt: skip
        assert code == 0
        report = json.loads(
            (out_dir / f"{ONE_CHAIN.stem}_results.json").read_text(encoding="utf-8")
        )
        assert report["interface"]["skipped"] is True
        assert report["preflight"]["status"] == "skipped"
        assert report["geometry"]["shape_complementarity"]["skipped"] is True


class TestOnUnmappableResidue:
    """``--on-unmappable-residue x`` keeps sending an X, so the residue check does not refuse."""

    @pytest.fixture
    def custom_residue(self, tmp_path):
        import biotite.structure.io.pdb as pdb_io

        from tests.test_pre_structures import build_chain

        atoms = build_chain(["ALA", "GLY", "XYZ", "ALA", "GLY", "ALA"], "B") + build_chain(
            ["ALA"] * 30, "A", x_offset=900.0
        )
        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, atoms)
        path = tmp_path / "custom.pdb"
        pdb_file.write(str(path))
        return path

    def test_a_residue_the_builder_cannot_express_is_refused_by_default(
        self, tmp_path, custom_residue
    ):
        with pytest.raises(IncompatibleInputError) as caught:
            run_pipeline(
                custom_residue,
                tmp_path / "out",
                skip_prep=True,
                skip_relax=True,
                metrics=frozenset({"openfold"}),
                preflight_only=False,
                openfold_conda_env=None,
            )
        assert "OpenFold3 cannot take these residues" in str(caught.value)

    def test_x_lifts_the_residue_check_only(self, tmp_path, custom_residue):
        result = run_pipeline(
            custom_residue,
            tmp_path / "out",
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            on_unmappable_residue="x",
            preflight_only=True,
        )
        assert result["preflight"]["status"] == "ok"

    def test_x_does_not_lift_the_closure_limit(self, tmp_path):
        with pytest.raises(IncompatibleInputError) as caught:
            run_pipeline(
                BICYCLE,
                tmp_path / "out",
                skip_prep=True,
                skip_relax=True,
                metrics=frozenset({"openfold"}),
                on_unmappable_residue="x",
            )
        assert [v.constraint for v in caught.value.violations] == ["closures"]
