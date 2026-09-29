"""Tests for the protocol module."""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import BEST_PLATFORM

from binding_metrics.core.simulation import SimulationConfig
from binding_metrics.protocols.base import BaseProtocol, ProtocolResults
from binding_metrics.protocols.peptide import PeptideBindingProtocol


class TestProtocolResults:
    """Tests for ProtocolResults dataclass."""

    def test_default_values(self):
        """Should initialize with empty arrays and zero means."""
        results = ProtocolResults()

        assert len(results.sasa_buried) == 0
        assert results.sasa_buried_mean == 0.0
        assert results.sasa_buried_std == 0.0
        assert len(results.interface_contacts) == 0
        assert results.interface_contacts_mean == 0.0
        assert len(results.interaction_energy) == 0
        assert results.interaction_energy_mean == 0.0
        assert len(results.rmsd) == 0
        assert results.rmsd_mean == 0.0
        assert results.raw_data == {}

    def test_to_dict(self):
        """to_dict should return summary statistics."""
        results = ProtocolResults(
            sasa_buried=np.array([100.0, 110.0, 105.0]),
            sasa_buried_mean=105.0,
            sasa_buried_std=5.0,
            interface_contacts=np.array([10, 12, 11]),
            interface_contacts_mean=11.0,
            interaction_energy=np.array([-50.0, -55.0, -52.0]),
            interaction_energy_mean=-52.3,
            interaction_energy_std=2.5,
            rmsd=np.array([0.1, 0.15, 0.12]),
            rmsd_mean=0.123,
        )

        d = results.to_dict()

        assert "sasa_buried_mean" in d
        assert "interface_contacts_mean" in d
        assert "interaction_energy_mean" in d
        assert "rmsd_mean" in d
        assert "n_frames" in d
        assert d["n_frames"] == 3

    def test_summary_string(self):
        """summary should return formatted string."""
        results = ProtocolResults(
            sasa_buried=np.array([100.0]),
            sasa_buried_mean=100.0,
            sasa_buried_std=5.0,
            interface_contacts=np.array([10]),
            interface_contacts_mean=10.0,
            interaction_energy=np.array([-50.0]),
            interaction_energy_mean=-50.0,
            interaction_energy_std=2.0,
            rmsd=np.array([0.1]),
            rmsd_mean=0.1,
        )

        summary = results.summary()

        assert "Buried SASA" in summary
        assert "100.0" in summary
        assert "Interface contacts" in summary
        assert "10.0" in summary
        assert "Interaction energy" in summary
        assert "-50.0" in summary
        assert "RMSD" in summary
        assert "0.100" in summary

    def test_raw_data_storage(self):
        """raw_data should store arbitrary metadata."""
        results = ProtocolResults(
            raw_data={
                "ligand_chain": "B",
                "receptor_chains": ["A"],
                "custom_metric": 42.0,
            }
        )

        assert results.raw_data["ligand_chain"] == "B"
        assert results.raw_data["custom_metric"] == 42.0


class TestBaseProtocol:
    """Tests for BaseProtocol abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseProtocol is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseProtocol("test.pdb", "A", ["B"])

    def test_subclass_must_implement_run(self):
        """Subclass without run() should fail."""

        class IncompleteProtocol(BaseProtocol):
            def analyze(self, trajectory_path=None):
                return ProtocolResults()

        with pytest.raises(TypeError):
            IncompleteProtocol("test.pdb", "A", ["B"])

    def test_subclass_must_implement_analyze(self):
        """Subclass without analyze() should fail."""

        class IncompleteProtocol(BaseProtocol):
            def run(self, output_dir, **kwargs):
                return Path("traj.dcd")

        with pytest.raises(TypeError):
            IncompleteProtocol("test.pdb", "A", ["B"])

    def test_complete_subclass_works(self):
        """Complete subclass should instantiate."""

        class CompleteProtocol(BaseProtocol):
            def run(self, output_dir, **kwargs):
                return Path("traj.dcd")

            def analyze(self, trajectory_path=None):
                return ProtocolResults()

        protocol = CompleteProtocol("test.pdb", "A", ["B"])
        assert protocol.pdb_path == Path("test.pdb")
        assert protocol.ligand_chain == "A"
        assert protocol.receptor_chains == ["B"]


class TestPeptideBindingProtocol:
    """Tests for PeptideBindingProtocol."""

    def test_init_with_defaults(self, sample_pdb_path: Path):
        """Should initialize with default configuration."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        assert protocol.pdb_path == sample_pdb_path
        assert protocol.ligand_chain == "B"
        assert protocol.receptor_chains == ["A"]
        assert protocol.forcefield_name == "amber"
        assert protocol.simulation_config is not None

    def test_init_with_custom_config(self, sample_pdb_path: Path):
        """Should accept custom simulation config."""
        config = SimulationConfig(duration_ns=1.0, platform=BEST_PLATFORM)
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
            simulation_config=config,
        )

        assert protocol.simulation_config.duration_ns == 1.0
        assert protocol.simulation_config.platform == BEST_PLATFORM

    def test_init_with_charmm(self, sample_pdb_path: Path):
        """Should accept CHARMM force field."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
            forcefield="charmm",
        )

        assert protocol.forcefield_name == "charmm"

    def test_trajectory_path_initially_none(self, sample_pdb_path: Path):
        """trajectory_path should be None before run()."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        assert protocol.trajectory_path is None

    def test_results_initially_none(self, sample_pdb_path: Path):
        """results should be None before analyze()."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        assert protocol.results is None

    def test_analyze_without_run_raises(self, sample_pdb_path: Path):
        """analyze() without run() or trajectory_path should raise."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A"],
        )

        with pytest.raises(RuntimeError, match="No trajectory"):
            protocol.analyze()

    def test_multiple_receptor_chains(self, sample_pdb_path: Path):
        """Should accept multiple receptor chains."""
        protocol = PeptideBindingProtocol(
            pdb_path=sample_pdb_path,
            ligand_chain="B",
            receptor_chains=["A", "C", "D"],
        )

        assert protocol.receptor_chains == ["A", "C", "D"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_creates_output(
        self, example_pdb_path: Path, example_pdb_chains: dict, output_dir: Path
    ):
        """run() should create trajectory and topology files."""
        pytest.importorskip("openmm")
        pytest.importorskip("pdbfixer", reason="pdbfixer required to fix incomplete PDB structures")

        config = SimulationConfig(
            duration_ns=0.0001,  # Very short for testing
            equilibration_ns=0.00001,
            save_interval_ps=0.01,
            platform=BEST_PLATFORM,
        )

        protocol = PeptideBindingProtocol(
            pdb_path=example_pdb_path,
            ligand_chain=example_pdb_chains["ligand"],
            receptor_chains=example_pdb_chains["receptor"],
            simulation_config=config,
        )

        traj_path = protocol.run(output_dir)

        # Generic assertions - file existence only
        assert traj_path.exists()
        assert traj_path.suffix == ".dcd"
        assert (output_dir / "solvated.pdb").exists()
        assert (output_dir / "state.csv").exists()


class TestRunAndAnalyze:
    """Tests for the combined run_and_analyze workflow."""

    def test_run_and_analyze_returns_results(self, sample_pdb_path: Path):
        """run_and_analyze should return ProtocolResults."""

        class MockProtocol(BaseProtocol):
            def run(self, output_dir, **kwargs):
                self._trajectory_path = Path("mock.dcd")
                return self._trajectory_path

            def analyze(self, trajectory_path=None):
                self._results = ProtocolResults(
                    sasa_buried_mean=100.0,
                    interaction_energy_mean=-50.0,
                )
                return self._results

        protocol = MockProtocol(sample_pdb_path, "B", ["A"])
        results = protocol.run_and_analyze("/tmp/output")

        assert isinstance(results, ProtocolResults)
        assert results.sasa_buried_mean == 100.0
        assert results.interaction_energy_mean == -50.0


class TestReportNonFinite:
    """NaN and inf are N/A in the report, not a red score."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), np.float32("nan"), np.nan])
    def test_rag_treats_non_finite_as_not_available(self, value):
        from binding_metrics.protocols.report import _THRESHOLDS, _rag

        for spec in _THRESHOLDS:
            assert _rag(value, spec) == "⬜"

    def test_rag_still_grades_finite_values(self):
        from binding_metrics.protocols.report import _THRESHOLDS, _rag

        rmsd = next(spec for spec in _THRESHOLDS if spec["label"] == "MD RMSD")
        assert _rag(1.0, rmsd) == "🟢"
        assert _rag(np.float32(3.0), rmsd) == "🟡"
        assert _rag(9.0, rmsd) == "🔴"

    def test_fmt_shows_dash_for_none_and_non_finite(self):
        from binding_metrics.protocols.report import _fmt

        for value in (None, float("nan"), float("inf"), -np.inf, np.float32("nan")):
            assert _fmt(value) == "—"

    def test_fmt_formats_numpy_and_python_numbers(self):
        from binding_metrics.protocols.report import _fmt

        assert _fmt(np.float32(1.5), 2) == "1.50"
        assert _fmt(1.23456, 2) == "1.23"
        assert _fmt(3) == "3"
        assert _fmt(np.int64(7)) == "7"

    def test_scorecard_row_of_a_nan_metric_is_not_red(self):
        from binding_metrics.protocols.report import _build_summary

        summary = _build_summary({"sample_id": "s", "relax": {"rmsd_md_final": float("nan")}})
        row = next(line for line in summary.splitlines() if "MD RMSD" in line and "|" in line)
        assert "⬜" in row and "🔴" not in row
        assert "nan" not in row.lower()


class TestReportJson:
    """write_report keeps numbers as numbers and lists the non-finite fields."""

    @staticmethod
    def _write(tmp_path, results, fmt="json"):
        from binding_metrics.protocols.report import write_report

        return write_report(results, tmp_path, "sample", fmt=fmt)

    def test_numpy_scalars_become_numbers_and_paths_become_strings(self, tmp_path):
        results = {
            "sample_id": "s",
            "relax": {
                "energy": np.float32(1.5),
                "count": np.int64(3),
                "flag": np.bool_(True),
                "series": np.array([1.0, 2.0]),
                "path": Path("out/relaxed.cif"),
            },
        }
        data = json.loads(self._write(tmp_path, results).read_text())
        assert data["relax"]["energy"] == 1.5 and isinstance(data["relax"]["energy"], float)
        assert data["relax"]["count"] == 3 and isinstance(data["relax"]["count"], int)
        assert data["relax"]["flag"] is True
        assert data["relax"]["series"] == [1.0, 2.0]
        assert data["relax"]["path"] == str(Path("out/relaxed.cif"))

    def test_unknown_types_keep_the_str_fallback(self, tmp_path):
        class Odd:
            def __str__(self):
                return "odd-value"

        data = json.loads(self._write(tmp_path, {"sample_id": "s", "odd": Odd()}).read_text())
        assert data["odd"] == "odd-value"

    def test_nonfinite_fields_lists_dotted_paths(self, tmp_path):
        results = {
            "sample_id": "s",
            "relax": {"rmsd_md_final": float("nan"), "energy": -5.0},
            "geometry": {
                "ramachandran": {"per_residue": [{"phi": 1.0}, {"phi": np.float32("inf")}]}
            },
            "electrostatics": {"coulomb_energy_kJ": -np.inf},
        }
        path = self._write(tmp_path, results)
        data = json.loads(path.read_text())
        assert sorted(data["nonfinite_fields"]) == [
            "electrostatics.coulomb_energy_kJ",
            "geometry.ramachandran.per_residue[1].phi",
            "relax.rmsd_md_final",
        ]
        assert results["nonfinite_fields"] == data["nonfinite_fields"]

    def test_nan_tokens_are_left_as_they_were(self, tmp_path):
        text = self._write(tmp_path, {"sample_id": "s", "x": {"y": float("nan")}}).read_text()
        assert '"y": NaN' in text

    def test_no_nonfinite_gives_an_empty_list_and_rewriting_is_stable(self, tmp_path):
        results = {"sample_id": "s", "relax": {"energy": -5.0}}
        first = json.loads(self._write(tmp_path, results).read_text())
        second = json.loads(self._write(tmp_path, results).read_text())
        assert first["nonfinite_fields"] == [] == second["nonfinite_fields"]

    def test_csv_output_has_no_new_column(self, tmp_path):
        path = self._write(
            tmp_path, {"sample_id": "s", "relax": {"energy": float("nan")}}, fmt="csv"
        )
        header = path.read_text().splitlines()[0].split(",")
        assert "nonfinite_fields" not in header

    def test_relaxation_json_writer_uses_the_same_encoder(self, tmp_path):
        from binding_metrics.protocols.relaxation import RelaxationResult, _run_one

        class StubRelaxer:
            def run(self, input_path, output_dir, sample_id=None):
                return RelaxationResult(
                    sample_id="s", success=True, potential_energy_minimized=np.float32(-12.5)
                )

        target = tmp_path / "relax.json"
        _run_one(StubRelaxer(), tmp_path / "in.cif", tmp_path, "s", target, None)
        data = json.loads(target.read_text())
        assert data["potential_energy_minimized"] == -12.5
