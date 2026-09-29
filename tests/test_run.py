"""Tests for the binding-metrics-run pipeline failure detection.

Regression: the pipeline used to print "DONE" and exit 0 even when metric
steps errored or produced no result (e.g. a failed relaxation leaving empty
energy / NaN interface). _collect_failures is what turns those into a non-zero
exit instead of a silent pass.
"""

import logging
from pathlib import Path

import pytest

from binding_metrics.cli.run import (
    ChainNotFoundError,
    _collect_failures,
    _require_chains_present,
    run_pipeline,
)

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


def test_no_failures_on_clean_results():
    results = {
        "energy": {"success": True, "relaxed_interaction_energy": -357.4},
        "interface": {"delta_sasa": 1574.0, "hbonds": 7},
        "geometry": {"ramachandran": {"ramachandran_favoured_pct": 100.0}},
        "openfold": {"skipped": True},
        "total_elapsed_s": 17.7,
    }
    assert _collect_failures(results) == []


def test_detects_error_key():
    results = {
        "geometry": {"error": "index -1 is out of bounds for axis 0 with size 0"},
        "interface": {"delta_sasa": 1200.0},
    }
    failures = _collect_failures(results)
    assert [s for s, _ in failures] == ["geometry"]


def test_detects_success_false():
    """A failed relaxation (success=False) must count as a failure."""
    results = {
        "relax": {"success": False, "error_message": "KeyError: 'N'"},
        "energy": {"success": True, "relaxed_interaction_energy": -1.0},
    }
    failures = _collect_failures(results)
    assert ("relax", "KeyError: 'N'") in failures


def test_skipped_steps_are_not_failures():
    results = {
        "energy": {"skipped": True},
        "openfold": {"skipped": True},
        "relax": {"skipped": True},
    }
    assert _collect_failures(results) == []


def test_multiple_failures_collected():
    results = {
        "relax": {"success": False, "error_message": "boom"},
        "energy": {"error": "no template"},
        "interface": {"delta_sasa": 1.0},  # ok
        "geometry": {"error": "empty chain"},
        "electrostatics": {"skipped": True},  # not a failure
    }
    steps = {s for s, _ in _collect_failures(results)}
    assert steps == {"relax", "energy", "geometry"}


def test_ignores_non_dict_and_scalars():
    results = {"total_elapsed_s": 12.3, "input": "x.cif", "energy": {"error": "e"}}
    assert [s for s, _ in _collect_failures(results)] == ["energy"]


# ---------------------------------------------------------------------------
# Requested chain IDs must exist in the structure
# ---------------------------------------------------------------------------

_CHAIN_INFO = {"all_chains": [{"id": "B", "n_residues": 13}, {"id": "A", "n_residues": 85}]}


class TestRequireChainsPresent:
    def test_known_and_auto_chains_pass(self):
        _require_chains_present(_CHAIN_INFO, "B", "A")
        _require_chains_present(_CHAIN_INFO, None, None)
        _require_chains_present(_CHAIN_INFO, "B", None)

    def test_missing_chain_names_it_and_lists_available(self):
        with pytest.raises(ValueError, match=r"chain 'Z' not found; available: B \(13\), A \(85\)"):
            _require_chains_present(_CHAIN_INFO, "Z", "A")

    def test_missing_receptor_is_caught(self):
        with pytest.raises(ChainNotFoundError, match="chain 'Q' not found"):
            _require_chains_present(_CHAIN_INFO, "B", "Q")

    def test_several_missing_chains_are_all_named(self):
        with pytest.raises(ValueError, match=r"chains 'Z', 'Q' not found; available: B \(13\)"):
            _require_chains_present(_CHAIN_INFO, "Z", "Q")


class TestRunPipelineChains:
    """1YCR has protein chains B (13 residues, p53 peptide) and A (85, MDM2)."""

    def test_unknown_peptide_chain_fails_before_any_step(self, tmp_path):
        with pytest.raises(ValueError, match=r"chain 'Z' not found; available: B \(13\), A \(85\)"):
            run_pipeline(EXAMPLE_1YCR, tmp_path, peptide_chain="Z", metrics=frozenset())
        assert not list(tmp_path.glob("*_cleaned.cif")), "prep must not have run"

    def test_unknown_receptor_chain_fails(self, tmp_path):
        with pytest.raises(ValueError, match="chain 'Q' not found"):
            run_pipeline(EXAMPLE_1YCR, tmp_path, receptor_chain="Q", metrics=frozenset())

    def test_explicit_valid_chains_are_accepted(self, tmp_path):
        results = run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            peptide_chain="B",
            receptor_chain="A",
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset(),
        )
        assert results["chains"]["peptide_n_residues"] == 13
        assert results["chains"]["receptor_n_residues"] == 85


class TestProvenanceInResults:
    def _run(self, tmp_path, **kwargs):
        return run_pipeline(
            EXAMPLE_1YCR, tmp_path, skip_prep=True, skip_relax=True, metrics=frozenset(), **kwargs
        )

    def test_provenance_block_is_recorded_with_the_seed(self, tmp_path):
        prov = self._run(tmp_path, random_seed=42)["provenance"]
        assert prov["seed"] == 42
        assert prov["schema_version"] == 1
        assert isinstance(prov["package_version"], str)

    def test_default_seed_is_recorded(self, tmp_path):
        from binding_metrics.core.system import DEFAULT_RANDOM_SEED

        assert self._run(tmp_path)["provenance"]["seed"] == DEFAULT_RANDOM_SEED

    def test_fresh_randomness_is_recorded_as_none(self, tmp_path):
        assert self._run(tmp_path, random_seed=None)["provenance"]["seed"] is None

    def test_provenance_is_not_a_failed_step(self, tmp_path):
        assert _collect_failures(self._run(tmp_path)) == []

    @pytest.mark.parametrize("fmt", ["json", "csv"])
    def test_report_writers_accept_the_block(self, tmp_path, fmt):
        import json

        from binding_metrics.protocols.report import write_report

        results = self._run(tmp_path)
        path = write_report(results, tmp_path, "s", fmt=fmt, summary=True)
        assert path.exists()
        if fmt == "json":
            assert (
                json.loads(path.read_text())["provenance"]["seed"] == results["provenance"]["seed"]
            )


class _RecordingRelaxer:
    """Stands in for ``ImplicitRelaxation``: records the config, relaxes nothing."""

    configs: list = []

    def __init__(self, config):
        self.config = config
        type(self).configs.append(config)

    def run(self, input_path, output_dir, sample_id=None):
        from binding_metrics.protocols.relaxation import RelaxationResult

        return RelaxationResult(sample_id=sample_id, success=False, error_message="stub")


@pytest.fixture
def recorded_configs(monkeypatch):
    monkeypatch.setattr(_RecordingRelaxer, "configs", [])
    monkeypatch.setattr(
        "binding_metrics.protocols.relaxation.ImplicitRelaxation", _RecordingRelaxer
    )
    return _RecordingRelaxer.configs


class TestShortMdDuration:
    """A short ``--md-duration-ps`` must not trip the RelaxationConfig validation."""

    def _relax(self, tmp_path, md_duration_ps):
        return run_pipeline(
            EXAMPLE_1YCR,
            tmp_path,
            skip_prep=True,
            md_duration_ps=md_duration_ps,
            metrics=frozenset(),
        )

    @pytest.mark.parametrize("duration", [1.0, 5.0, 9.5])
    def test_duration_below_the_default_interval_saves_one_frame(
        self, tmp_path, recorded_configs, duration
    ):
        self._relax(tmp_path, duration)
        (config,) = recorded_configs
        assert config.md_duration_ps == duration
        assert config.md_save_interval_ps == duration

    def test_minimise_only_stays_valid(self, tmp_path, recorded_configs):
        self._relax(tmp_path, 0.0)
        (config,) = recorded_configs
        assert config.md_duration_ps == 0.0
        assert config.md_save_interval_ps == 10.0

    @pytest.mark.parametrize("duration", [10.0, 200.0])
    def test_longer_runs_keep_the_default_interval(self, tmp_path, recorded_configs, duration):
        self._relax(tmp_path, duration)
        assert recorded_configs[0].md_save_interval_ps == 10.0


class TestStructuralQcWarning:
    """A failed advisory QC is logged once and never counts as a failed step."""

    @staticmethod
    def _relaxer_reporting(qc_passed, failed_checks=()):
        from binding_metrics.protocols.relaxation import RelaxationResult

        class _Relaxer:
            def __init__(self, config):
                pass

            def run(self, input_path, output_dir, sample_id=None):
                return RelaxationResult(
                    sample_id=sample_id,
                    success=True,
                    minimized_structure_path=str(input_path),
                    qc_passed=qc_passed,
                    qc={"passed": qc_passed, "failed": list(failed_checks), "checks": {}},
                )

        return _Relaxer

    def _run(self, tmp_path, monkeypatch, relaxer):
        monkeypatch.setattr("binding_metrics.protocols.relaxation.ImplicitRelaxation", relaxer)
        return run_pipeline(
            EXAMPLE_1YCR, tmp_path, skip_prep=True, md_duration_ps=0.0, metrics=frozenset()
        )

    def test_failed_qc_logs_one_warning_naming_the_checks(self, tmp_path, monkeypatch, caplog):
        relaxer = self._relaxer_reporting(False, ["bond_lengths", "clashes"])
        with caplog.at_level(logging.INFO, logger="binding_metrics"):
            results = self._run(tmp_path, monkeypatch, relaxer)
        warnings_ = [r for r in caplog.records if "Structural QC failed" in r.getMessage()]
        assert len(warnings_) == 1
        assert warnings_[0].levelno == logging.WARNING
        assert "bond_lengths,clashes" in warnings_[0].getMessage()
        assert _collect_failures(results) == []  # advisory: no exit-1 path

    @pytest.mark.parametrize("qc_passed", [True, None])
    def test_passing_or_missing_qc_is_silent(self, tmp_path, monkeypatch, caplog, qc_passed):
        with caplog.at_level(logging.INFO, logger="binding_metrics"):
            self._run(tmp_path, monkeypatch, self._relaxer_reporting(qc_passed))
        assert "Structural QC failed" not in caplog.text
