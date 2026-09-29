"""Tests for the implicit solvent MD relaxation protocol."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_cuda

from binding_metrics.protocols.relaxation import (
    ImplicitRelaxation,
    RelaxationConfig,
    RelaxationResult,
)


class TestRelaxationConfig:
    """Tests for RelaxationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RelaxationConfig()
        assert config.md_duration_ps == 200.0
        assert config.md_timestep_fs == 2.0
        assert config.md_temperature_k == 300.0
        assert config.solvent_model == "obc2"
        assert config.device == "cuda"
        assert config.peptide_chain_id is None
        assert config.receptor_chain_id is None
        assert config.custom_bond_handler is None

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            solvent_model="gbn2",
            peptide_chain_id="A",
        )
        assert config.md_duration_ps == 0.0
        assert config.device == "cpu"
        assert config.solvent_model == "gbn2"
        assert config.peptide_chain_id == "A"

    def test_custom_bond_handler_callable(self):
        """Should accept a callable for custom_bond_handler."""

        def handler(topo, pos, chain):
            return topo, pos, []

        config = RelaxationConfig(custom_bond_handler=handler)
        assert callable(config.custom_bond_handler)


class TestRelaxationConfigValidation:
    """__post_init__ rejects MD settings that would save no frames."""

    def test_duration_shorter_than_interval_raises(self):
        with pytest.raises(ValueError, match="md_save_interval_ps"):
            RelaxationConfig(md_duration_ps=5.0, md_save_interval_ps=10.0)

    def test_non_positive_interval_raises_when_md_runs(self):
        with pytest.raises(ValueError, match="md_save_interval_ps"):
            RelaxationConfig(md_duration_ps=10.0, md_save_interval_ps=0.0)

    def test_minimize_only_ignores_the_interval(self):
        RelaxationConfig(md_duration_ps=0.0, md_save_interval_ps=10.0)
        RelaxationConfig(md_duration_ps=0.0, md_save_interval_ps=0.0)

    def test_non_multiple_duration_warns_with_simulated_length(self):
        with pytest.warns(UserWarning, match=r"not a multiple.*stops after 200 ps \(20 frames\)"):
            RelaxationConfig(md_duration_ps=205.0, md_save_interval_ps=10.0)

    @pytest.mark.parametrize(
        "duration, interval",
        [(200.0, 10.0), (10.0, 10.0), (0.3, 0.1), (2.0, 0.5)],
    )
    def test_whole_number_of_intervals_is_silent(self, duration, interval):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RelaxationConfig(md_duration_ps=duration, md_save_interval_ps=interval)

    def test_frame_count_absorbs_float_error(self):
        from binding_metrics.protocols.relaxation import _md_frame_count

        assert 0.3 / 0.1 < 3.0  # the plain int() of this would give 2
        assert _md_frame_count(0.3, 0.1) == 3
        assert _md_frame_count(205.0, 10.0) == 20


class TestRelaxationResult:
    """Tests for RelaxationResult dataclass."""

    def test_failed_result(self):
        """Should represent a failed run correctly."""
        result = RelaxationResult(sample_id="test", success=False, error_message="Boom")
        assert result.sample_id == "test"
        assert not result.success
        assert result.error_message == "Boom"
        assert result.potential_energy_minimized is None

    def test_to_dict_keys(self):
        """to_dict() should contain required keys."""
        result = RelaxationResult(sample_id="test", success=True)
        d = result.to_dict()
        assert "sample_id" in d
        assert "success" in d
        assert "potential_energy_minimized" in d
        assert "rmsd_md_final" in d
        assert "minimization_time_s" in d

    def test_to_dict_platform_keys_default_to_none(self):
        d = RelaxationResult(sample_id="test", success=True).to_dict()
        assert d["platform"] is None
        assert d["precision"] is None
        assert d["platform_fallback_reason"] is None

    def test_to_dict_rmsf_json(self):
        """to_dict() should serialize per-residue RMSF as JSON string."""
        result = RelaxationResult(
            sample_id="test",
            success=True,
            peptide_rmsf_per_residue=[1.0, 2.0, 3.0],
        )
        d = result.to_dict()
        assert "peptide_rmsf_per_residue" in d
        parsed = json.loads(d["peptide_rmsf_per_residue"])
        assert parsed == [1.0, 2.0, 3.0]

    def test_to_dict_no_rmsf_key_when_none(self):
        """to_dict() should omit rmsf key when not set."""
        result = RelaxationResult(sample_id="test", success=True)
        d = result.to_dict()
        assert "peptide_rmsf_per_residue" not in d


class TestImplicitRelaxation:
    """Tests for ImplicitRelaxation class."""

    def test_init(self):
        """Should initialize with config."""
        config = RelaxationConfig()
        relaxer = ImplicitRelaxation(config)
        assert relaxer.config is config
        assert not relaxer._openmm_imported

    def test_get_platform_records_cpu(self):
        relaxer = ImplicitRelaxation(RelaxationConfig(device="cpu", md_duration_ps=0.0))
        platform, _ = relaxer._get_platform()
        assert platform.getName() == "CPU"
        assert relaxer._platform_used == "CPU"
        assert relaxer._precision_used is None
        assert relaxer._platform_fallback_reason is None

    def test_cuda_failure_falls_back_to_cpu_and_is_recorded(self, monkeypatch):
        import openmm

        real_lookup = openmm.Platform.getPlatformByName

        def lookup(name):
            if name == "CUDA":
                raise RuntimeError("no CUDA driver in this test")
            return real_lookup(name)

        monkeypatch.setattr(openmm.Platform, "getPlatformByName", staticmethod(lookup))
        relaxer = ImplicitRelaxation(RelaxationConfig(device="cuda", md_duration_ps=0.0))
        platform, _ = relaxer._get_platform()
        assert platform.getName() == "CPU"
        assert relaxer._platform_used == "CPU"
        assert "no CUDA driver in this test" in relaxer._platform_fallback_reason

    def test_addhydrogens_failure_raises_and_logs(self, tmp_path, prepped_example_cif, caplog):
        """Both addHydrogens attempts failing stops the run with a clear error."""
        from unittest import mock

        from openmm import app

        config = RelaxationConfig(md_duration_ps=0.0, device="cpu")
        with mock.patch.object(app.Modeller, "addHydrogens", side_effect=ValueError("boom")):
            with caplog.at_level("ERROR", logger="binding_metrics.protocols.relaxation"):
                result = ImplicitRelaxation(config).run(prepped_example_cif, tmp_path / "out")
        assert not result.success
        assert "addHydrogens failed with and without the force field" in result.error_message
        assert "boom" in result.error_message
        assert "addHydrogens failed" in caplog.text

    @requires_cuda
    @pytest.mark.integration
    def test_run_minimize_only_cif(self, tmp_path: Path, prepped_example_cif):
        """Should successfully minimize a CIF structure (no MD)."""
        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=10,
            min_steps_restrained=5,
            min_steps_final=10,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(prepped_example_cif, tmp_path / "out")

        assert result.success, result.error_message
        assert result.potential_energy_minimized is not None
        assert result.minimized_structure_path is not None
        assert Path(result.minimized_structure_path).exists()
        assert result.md_final_structure_path is None
        assert result.platform == "CUDA"
        assert result.precision == "mixed"
        assert result.platform_fallback_reason is None

    @requires_cuda
    @pytest.mark.slow
    @pytest.mark.integration
    def test_run_with_md_cif(self, tmp_path: Path, prepped_example_cif):
        """Should run a very short MD simulation on a CIF structure."""
        config = RelaxationConfig(
            md_duration_ps=2.0,
            md_save_interval_ps=1.0,
            md_timestep_fs=1.0,  # 1 fs for stability after clash resolution
            min_steps_initial=200,
            min_steps_restrained=100,
            min_steps_final=200,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(prepped_example_cif, tmp_path / "out")

        assert result.success, result.error_message
        assert result.md_final_structure_path is not None
        assert result.rmsd_md_final is not None
        assert result.rmsd_md_final >= 0.0

    @requires_cuda
    @pytest.mark.integration
    def test_run_missing_file_fails_gracefully(self, tmp_path: Path):
        """Should return failed result for missing input."""
        config = RelaxationConfig(md_duration_ps=0.0)
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(tmp_path / "missing.pdb", tmp_path / "out")
        assert not result.success
        assert result.error_message is not None

    @requires_cuda
    @pytest.mark.integration
    def test_sample_id_defaults_to_file_stem(self, tmp_path: Path, prepped_example_cif):
        """sample_id should default to the input file stem."""
        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=5,
            min_steps_restrained=5,
            min_steps_final=5,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(prepped_example_cif, tmp_path / "out")
        assert result.sample_id == prepped_example_cif.stem

    @pytest.mark.integration
    def test_cpu_platform_available(self, tmp_path: Path, prepped_example_cif):
        """CPU platform should work as a fallback for non-GPU environments."""
        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            min_steps_initial=5,
            min_steps_restrained=5,
            min_steps_final=5,
        )
        relaxer = ImplicitRelaxation(config)
        result = relaxer.run(prepped_example_cif, tmp_path / "out")
        assert result.success, result.error_message
        assert result.platform == "CPU"
        assert result.platform_fallback_reason is None
        assert result.to_dict()["platform"] == "CPU"

    @pytest.mark.integration
    def test_kabsch_rmsd_identical(self):
        """_compute_rmsd should return 0 for identical positions."""
        config = RelaxationConfig()
        relaxer = ImplicitRelaxation(config)

        import openmm.unit as unit
        from openmm import Vec3

        pos = [Vec3(float(i) * 0.1 + 0.1, 0.0, 0.0) for i in range(5)] * unit.nanometers
        rmsd = relaxer._compute_rmsd(pos, pos)
        assert abs(rmsd) < 1e-6

    @pytest.mark.integration
    def test_kabsch_rmsd_rotated_copy_is_zero(self):
        """A rigidly rotated and translated copy superposes exactly (RMSD 0)."""
        from openmm import Vec3
        from scipy.spatial.transform import Rotation

        relaxer = ImplicitRelaxation(RelaxationConfig())
        rng = np.random.default_rng(0)
        coords_nm = rng.normal(size=(50, 3)) * 0.5
        rotation = Rotation.from_rotvec(np.deg2rad(60) * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0))
        moved_nm = rotation.apply(coords_nm) + np.array([0.3, -0.2, 0.1])

        pos1 = [Vec3(*row) for row in coords_nm]
        pos2 = [Vec3(*row) for row in moved_nm]
        assert relaxer._compute_rmsd(pos1, pos2) < 1e-4  # Angstrom
        assert relaxer._compute_rmsd(pos2, pos1) < 1e-4

    @pytest.mark.integration
    def test_kabsch_rmsd_matches_scipy_on_noisy_pair(self):
        """With coordinate noise the RMSD equals scipy's optimal-rotation RMSD."""
        from openmm import Vec3
        from scipy.spatial.transform import Rotation

        relaxer = ImplicitRelaxation(RelaxationConfig())
        rng = np.random.default_rng(1)
        coords_nm = rng.normal(size=(40, 3)) * 0.5
        rotation = Rotation.from_euler("xyz", [40.0, -25.0, 70.0], degrees=True)
        noisy_nm = rotation.apply(coords_nm) + rng.normal(size=(40, 3)) * 0.02

        centered_a = coords_nm - coords_nm.mean(axis=0)
        centered_b = noisy_nm - noisy_nm.mean(axis=0)
        _, scipy_rssd_nm = Rotation.align_vectors(centered_b, centered_a)
        expected_angstrom = scipy_rssd_nm / np.sqrt(len(coords_nm)) * 10.0

        pos1 = [Vec3(*row) for row in coords_nm]
        pos2 = [Vec3(*row) for row in noisy_nm]
        assert relaxer._compute_rmsd(pos1, pos2) == pytest.approx(expected_angstrom, rel=1e-6)
        # 0.02 nm sigma per axis -> about 0.02 * sqrt(3) nm = 3.5 A before fitting
        assert 0.0 < expected_angstrom < 3.5

    @pytest.mark.integration
    def test_rmsf_zero_for_static_trajectory(self):
        """_compute_rmsf should return 0 for identical frames."""
        config = RelaxationConfig()
        relaxer = ImplicitRelaxation(config)

        import openmm.unit as unit
        from openmm import Vec3

        pos = [Vec3(float(i) * 0.1 + 0.1, 0.0, 0.0) for i in range(5)] * unit.nanometers
        trajectory = [pos, pos, pos]
        rmsf = relaxer._compute_rmsf(trajectory, list(range(5)))
        assert np.allclose(rmsf, 0.0, atol=1e-6)


CYCLOSPORIN_CIF = Path(__file__).parent.parent / "data" / "example_ncaa_cyclosporin_1CWA.cif"


@pytest.fixture(scope="module")
def relaxed_cyclosporin(tmp_path_factory):
    """Minimize-only relaxation of the raw cyclosporin example (GAFF for BMT/ABA)."""
    config = RelaxationConfig(
        md_duration_ps=0.0,
        min_steps_initial=50,
        min_steps_restrained=20,
        min_steps_final=50,
        small_molecules="auto",
    )
    out = tmp_path_factory.mktemp("cyclosporin_names")
    result = ImplicitRelaxation(config).run(CYCLOSPORIN_CIF, out)
    assert result.success, result.error_message
    return result


@requires_cuda
@pytest.mark.integration
class TestRelaxedOutputKeepsNonstandardNames:
    """Relaxed cyclosporin keeps its D-alanine and sarcosine names.

    The force field needs DAL -> ALA and SAR -> NMG on the topology. Before the
    names were restored, the saved file had no DAL or SAR atoms, so every
    downstream step that keys on the residue name (Ramachandran D-residue
    handling among them) saw an all-L peptide.

    The D-alanine is the first residue of the cyclic peptide, where the linear
    phi angle is undefined, so ``compute_ramachandran`` cannot score it and its
    ``n_d_residues`` stays 0 for this structure. The internal sarcosine is
    scored, which shows the name reaching that metric.
    """

    @staticmethod
    def _peptide_residue_names(cif_path: str) -> list:
        import gemmi

        chains = gemmi.read_structure(str(cif_path))[0]
        peptide = min(chains, key=len)
        return [residue.name for residue in peptide]

    def test_saved_peptide_keeps_input_residue_names(self, relaxed_cyclosporin):
        names = self._peptide_residue_names(relaxed_cyclosporin.minimized_structure_path)
        assert names == [
            "DAL",
            "MLE",
            "MLE",
            "MVA",
            "BMT",
            "ABA",
            "SAR",
            "MLE",
            "VAL",
            "MLE",
            "ALA",
        ]

    def test_ramachandran_reads_the_restored_names(self, relaxed_cyclosporin):
        from binding_metrics.metrics.geometry import compute_ramachandran

        rama = compute_ramachandran(relaxed_cyclosporin.minimized_structure_path)
        scored = [entry["res_name"] for entry in rama["per_residue"]]
        assert "SAR" in scored
        assert "NMG" not in scored
