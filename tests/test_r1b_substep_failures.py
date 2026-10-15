"""A failed EvoBind score or adversarial check is a failed step (#178).

The model step (``prediction``, or ``openfold`` without ``--predictor``) writes
``evobind_error`` and ``adversarial_error`` when one of its sub-steps could not be computed.
``_collect_failures`` reads them, so ``binding-metrics-run`` exits 1 and a batch row is
``partial``. A check left out by the design-origin policy (``adversarial_skipped``) is not a
failure.
"""

import sys

import pytest

from binding_metrics.cli import run
from binding_metrics.cli.batch import run_batch
from binding_metrics.cli.run import _collect_failures, run_pipeline
from tests.predictors import synth, synth_af2, synth_of3
from tests.test_feat_c_support import (
    EXAMPLE_1YCR,
    PEPTIDE_CHAIN,
    RECEPTOR_CHAIN,
    StubOpenFold,
    complex_from,
)

STEM = EXAMPLE_1YCR.stem
BOOM = "the scoring stopped"


class TestCollectFailures:
    @pytest.mark.parametrize("step", ["prediction", "openfold"])
    def test_a_failed_score_fails_the_step(self, step):
        results = {step: {"avg_plddt": 80.0, "evobind_error": BOOM}}
        assert _collect_failures(results) == [(step, f"EvoBind score: {BOOM}")]

    @pytest.mark.parametrize("step", ["prediction", "openfold"])
    def test_a_failed_adversarial_check_fails_the_step(self, step):
        results = {step: {"avg_plddt": 80.0, "adversarial_error": BOOM}}
        assert _collect_failures(results) == [(step, f"EvoBind adversarial check: {BOOM}")]

    def test_both_sub_steps_give_one_entry_for_the_step(self):
        results = {"prediction": {"evobind_error": "a", "adversarial_error": "b"}}
        assert _collect_failures(results) == [
            ("prediction", "EvoBind score: a; EvoBind adversarial check: b")
        ]

    def test_the_reason_is_cut_like_the_other_failures(self):
        ((_, reason),) = _collect_failures({"prediction": {"evobind_error": "x" * 500}})
        assert len(reason) == 200

    @pytest.mark.parametrize(
        "block",
        [
            {"adversarial_skipped": "the design was made with this model", "evobind_score": 0.4},
            {"evobind_error": None, "adversarial_error": ""},
            {"evobind_score": 0.4, "delta_com_angstrom": 1.2},
            {"skipped": True, "evobind_error": BOOM},
        ],
    )
    def test_a_skipped_check_and_a_clean_block_are_not_failures(self, block):
        assert _collect_failures({"prediction": block}) == []

    def test_the_step_failures_are_read_as_before(self):
        results = {
            "geometry": {"error": "empty chain"},
            "energy": {"success": False, "error_message": "no template"},
            "interface": {"num_contacts": 3},
        }
        assert _collect_failures(results) == [
            ("geometry", "empty chain"),
            ("energy", "no template"),
        ]


def _renamed_output(directory, name):
    """A prediction of 1YCR whose chains are named P and R, not the B and A of the input."""
    truth = complex_from(EXAMPLE_1YCR)
    renamed = synth.SyntheticComplex(
        atoms=synth.renamed_atoms(truth, {"A": "R", "B": "P"}),
        plddt_per_atom=truth.plddt_per_atom,
        pae=truth.pae,
        pde=truth.pde,
        scalars=truth.scalars,
    )
    synth_of3.write_prediction(directory, name, renamed)
    return directory


def _argv(output_dir, *extra):
    return [
        "binding-metrics-run", "-i", str(EXAMPLE_1YCR), "-o", str(output_dir),
        "--skip-prep", "--skip-relax", "--metrics", "openfold",
        "--peptide-chain", PEPTIDE_CHAIN, "--receptor-chain", RECEPTOR_CHAIN,
        "--openfold-conda-env", "", *extra,
    ]  # fmt: skip


class TestRunPipeline:
    @pytest.mark.filterwarnings("ignore:summarize_prediction")
    def test_the_report_of_a_failed_run_names_the_sub_steps(self, tmp_path, monkeypatch, capsys):
        StubOpenFold(monkeypatch)
        out = _renamed_output(tmp_path / "out", STEM)
        argv = _argv(tmp_path / "run", "--predictor", "of3", "--prediction-dir", str(out))
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as stop:
            run.main()
        assert stop.value.code == 1
        shown = capsys.readouterr().out
        assert "1 step(s) did not complete" in shown
        assert "[x] prediction: EvoBind score: " in shown
        assert "EvoBind adversarial check: " in shown

    @pytest.mark.parametrize(
        ("function", "key", "label"),
        [
            ("compute_evobind_score", "evobind_error", "EvoBind score"),
            ("compute_evobind_adversarial_check", "adversarial_error", "EvoBind adversarial check"),
        ],
    )
    def test_the_openfold_step_without_a_predictor_fails_the_run(
        self, tmp_path, monkeypatch, function, key, label
    ):
        from binding_metrics.metrics import evobind

        def broken(*args, **kwargs):
            raise RuntimeError(BOOM)

        StubOpenFold(monkeypatch)
        monkeypatch.setattr(evobind, function, broken)
        results = run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain=PEPTIDE_CHAIN,
            receptor_chain=RECEPTOR_CHAIN,
            openfold_conda_env=None,
        )
        assert results["openfold"][key] == BOOM
        assert _collect_failures(results) == [("openfold", f"{label}: {BOOM}")]

    def test_a_check_left_out_by_the_design_origin_exits_0(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / "af2_out"
        out.mkdir()
        synth_af2.write_prediction(out, STEM, complex_from(EXAMPLE_1YCR))
        extra = ("--predictor", "af2", "--prediction-dir", str(out), "--designed-with", "bindcraft")
        monkeypatch.setattr(sys, "argv", _argv(tmp_path / "run", *extra))
        run.main()
        assert "DONE in" in capsys.readouterr().out


NAMES = ("a", "b")


@pytest.fixture
def samples(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    text = EXAMPLE_1YCR.read_text(encoding="utf-8")
    paths = []
    for name in NAMES:
        path = folder / f"{name}.pdb"
        path.write_text(f"REMARK   9 sample {name}\n{text}", encoding="utf-8")
        paths.append(path)
    return paths


def _batch(samples, out, **kwargs):
    return run_batch(
        samples,
        out,
        skip_prep=True,
        skip_relax=True,
        metrics={"openfold"},
        openfold_conda_env=None,
        n_workers=1,
        **kwargs,
    )


class TestBatch:
    @pytest.mark.filterwarnings("ignore:summarize_prediction")
    def test_the_prediction_step_marks_a_sample_whose_sub_step_failed(self, tmp_path, samples):
        root = tmp_path / "outputs"
        for name in NAMES:
            _renamed_output(root, name)
        rows = _batch(samples, tmp_path / "out", predictor="of3", prediction_dir=root)
        for row in rows:
            assert row["batch_status"] == "partial"
            assert row["batch_failed_steps"] == "prediction"
            assert "prediction: EvoBind score: " in row["batch_failed_reasons"]
            assert "prediction_evobind_error" in row and "prediction_adversarial_error" in row

    @pytest.mark.parametrize(
        ("function", "key", "label"),
        [
            ("compute_evobind_score", "evobind_error", "EvoBind score"),
            ("compute_evobind_adversarial_check", "adversarial_error", "EvoBind adversarial check"),
        ],
    )
    def test_the_batched_openfold_step_marks_a_sample_whose_sub_step_failed(
        self, tmp_path, samples, monkeypatch, function, key, label
    ):
        from binding_metrics.metrics import evobind

        def broken(*args, **kwargs):
            raise RuntimeError(BOOM)

        StubOpenFold(monkeypatch)
        monkeypatch.setattr(evobind, function, broken)
        rows = _batch(samples, tmp_path / "out")
        for row in rows:
            assert row["batch_status"] == "partial"
            assert row["batch_failed_steps"] == "openfold"
            assert row["batch_failed_reasons"] == f"openfold: {label}: {BOOM}"
            assert row[f"openfold_{key}"] == BOOM
            other = "adversarial_error" if key == "evobind_error" else "evobind_error"
            assert f"openfold_{other}" not in row

    def test_a_check_left_out_by_the_design_origin_keeps_the_row_ok(
        self, tmp_path, samples, monkeypatch
    ):
        StubOpenFold(monkeypatch)
        rows = _batch(samples, tmp_path / "out", predictor="of3", designed_with=("of3",))
        assert [row["batch_status"] for row in rows] == ["ok", "ok"]
        assert all("batch_failed_steps" not in row for row in rows)
