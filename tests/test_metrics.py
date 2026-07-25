"""Tests for the metrics module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestSASA:
    """Tests for SASA calculations."""

    def test_import_sasa_module(self):
        """Should be able to import SASA functions."""
        from binding_metrics.metrics.sasa import (
            calculate_buried_sasa,
            calculate_interface_sasa,
        )

        assert callable(calculate_buried_sasa)
        assert callable(calculate_interface_sasa)

    def test_calculate_buried_sasa_requires_mdtraj(self):
        """Should raise ImportError if mdtraj not available."""
        with patch.dict("sys.modules", {"mdtraj": None}):
            # Force reimport to trigger the ImportError path
            from binding_metrics.metrics import sasa

            # Manually set md to None to simulate missing mdtraj
            original_md = sasa.md
            sasa.md = None

            try:
                with pytest.raises(ImportError, match="mdtraj"):
                    sasa.calculate_buried_sasa("fake.dcd", "fake.pdb", [0, 1], [2, 3])
            finally:
                sasa.md = original_md

    @pytest.mark.integration
    def test_buried_sasa_returns_array(self, sample_pdb_path: Path, tmp_path: Path):
        """Buried SASA should return numpy array."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.sasa import calculate_buried_sasa

        # Create minimal trajectory from PDB
        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_buried_sasa(
            traj_path,
            sample_pdb_path,
            ligand_indices=list(range(23, 33)),
            receptor_indices=list(range(23)),
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == traj.n_frames

    @pytest.mark.integration
    def test_interface_sasa_returns_dict(self, sample_pdb_path: Path, tmp_path: Path):
        """Interface SASA should return dict with components."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.sasa import calculate_interface_sasa

        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_interface_sasa(
            traj_path,
            sample_pdb_path,
            ligand_indices=list(range(23, 33)),
            receptor_indices=list(range(23)),
        )

        assert isinstance(result, dict)
        assert "ligand" in result
        assert "receptor" in result
        assert "complex" in result
        assert "buried" in result


class TestContacts:
    """Tests for contact calculations."""

    def test_import_contacts_module(self):
        """Should be able to import contact functions."""
        from binding_metrics.metrics.contacts import (
            calculate_contact_residues,
            calculate_contacts,
        )

        assert callable(calculate_contacts)
        assert callable(calculate_contact_residues)

    def test_calculate_contacts_requires_mdtraj(self):
        """Should raise ImportError if mdtraj not available."""
        from binding_metrics.metrics import contacts

        original_md = contacts.md
        contacts.md = None

        try:
            with pytest.raises(ImportError, match="mdtraj"):
                contacts.calculate_contacts("fake.dcd", "fake.pdb", [0, 1], [2, 3])
        finally:
            contacts.md = original_md

    @pytest.mark.integration
    def test_contacts_returns_array(self, sample_pdb_path: Path, tmp_path: Path):
        """Contact count should return numpy array."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.contacts import calculate_contacts

        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_contacts(
            traj_path,
            sample_pdb_path,
            ligand_indices=list(range(23, 33)),
            receptor_indices=list(range(23)),
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == traj.n_frames
        assert result.dtype == np.float64

    @pytest.mark.integration
    def test_contacts_empty_indices_returns_zeros(self, sample_pdb_path: Path, tmp_path: Path):
        """Empty indices should return zeros."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.contacts import calculate_contacts

        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_contacts(
            traj_path, sample_pdb_path, ligand_indices=[], receptor_indices=[]
        )

        assert np.all(result == 0)


class TestEnergy:
    """Tests for energy calculations."""

    def test_import_energy_module(self):
        """Should be able to import energy functions."""
        from binding_metrics.metrics.energy import (
            calculate_component_energies,
            calculate_interaction_energy,
        )

        assert callable(calculate_interaction_energy)
        assert callable(calculate_component_energies)

    def test_calculate_energy_requires_mdtraj(self):
        """Should raise ImportError if mdtraj not available."""
        from binding_metrics.metrics import energy

        original_md = energy.md
        energy.md = None

        try:
            with pytest.raises(ImportError, match="mdtraj"):
                energy.calculate_interaction_energy("fake.dcd", "fake.pdb", [0, 1], [2, 3])
        finally:
            energy.md = original_md

    @pytest.mark.integration
    def test_interaction_energy_returns_array(
        self, prepped_example_pdb: Path, example_pdb_chains: dict, tmp_path: Path
    ):
        """Interaction energy over a trajectory of the bound MDM2-p53 complex.

        Uses the prepped fixture rather than the raw crystal: 1YCR has no
        hydrogens, which the force field cannot parameterise.
        """
        pytest.importorskip("mdtraj")
        pytest.importorskip("openmm")
        import mdtraj as md

        from binding_metrics.io.structures import get_chain_atom_indices
        from binding_metrics.metrics.energy import calculate_interaction_energy

        traj = md.load(str(prepped_example_pdb))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        ligand_indices = get_chain_atom_indices(prepped_example_pdb, [example_pdb_chains["ligand"]])
        receptor_indices = get_chain_atom_indices(
            prepped_example_pdb, example_pdb_chains["receptor"]
        )

        result = calculate_interaction_energy(
            traj_path,
            prepped_example_pdb,
            ligand_indices=ligand_indices,
            receptor_indices=receptor_indices,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == traj.n_frames
        assert result.dtype in (np.float64, np.float32)
        assert np.all(np.isfinite(result)), "non-finite interaction energy"
        # 1YCR is a bound complex, so the interaction must be favourable. A
        # sign flip here is the failure mode that a shape-only assertion would
        # have let through.
        assert result[0] < 0, f"bound complex gave repulsive energy {result[0]} kJ/mol"

    @pytest.mark.integration
    def test_component_energies_returns_dict(
        self, prepped_example_pdb: Path, example_pdb_chains: dict, tmp_path: Path
    ):
        """Electrostatic and vdW components must decompose the total exactly."""
        pytest.importorskip("mdtraj")
        pytest.importorskip("openmm")
        import mdtraj as md

        from binding_metrics.io.structures import get_chain_atom_indices
        from binding_metrics.metrics.energy import calculate_component_energies

        traj = md.load(str(prepped_example_pdb))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        ligand_indices = get_chain_atom_indices(prepped_example_pdb, [example_pdb_chains["ligand"]])
        receptor_indices = get_chain_atom_indices(
            prepped_example_pdb, example_pdb_chains["receptor"]
        )

        result = calculate_component_energies(
            traj_path,
            prepped_example_pdb,
            ligand_indices=ligand_indices,
            receptor_indices=receptor_indices,
        )

        assert isinstance(result, dict)
        assert set(result) >= {"electrostatic", "vdw", "total"}
        assert len(result["electrostatic"]) == len(result["vdw"])
        assert len(result["total"]) == traj.n_frames
        # The decomposition must be exact, not merely present: total is defined
        # as elec + vdw, so a drift here means one component is being computed
        # from different state than the other.
        np.testing.assert_allclose(
            result["total"], result["electrostatic"] + result["vdw"], rtol=0, atol=1e-9
        )
        assert np.all(np.isfinite(result["total"]))
        # Both terms are attractive across a packed protein-peptide interface.
        assert result["electrostatic"][0] < 0
        assert result["vdw"][0] < 0


class TestRMSD:
    """Tests for RMSD calculations."""

    def test_import_rmsd_module(self):
        """Should be able to import RMSD functions."""
        from binding_metrics.metrics.rmsd import (
            calculate_ligand_rmsd,
            calculate_rmsd,
            calculate_rmsf,
        )

        assert callable(calculate_rmsd)
        assert callable(calculate_rmsf)
        assert callable(calculate_ligand_rmsd)

    def test_calculate_rmsd_requires_mdtraj(self):
        """Should raise ImportError if mdtraj not available."""
        from binding_metrics.metrics import rmsd

        original_md = rmsd.md
        rmsd.md = None

        try:
            with pytest.raises(ImportError, match="mdtraj"):
                rmsd.calculate_rmsd("fake.dcd", "fake.pdb")
        finally:
            rmsd.md = original_md

    @pytest.mark.integration
    def test_rmsd_returns_array(self, sample_pdb_path: Path, tmp_path: Path):
        """RMSD should return numpy array."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import calculate_rmsd

        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_rmsd(traj_path, sample_pdb_path, atom_indices=list(range(33)))

        assert isinstance(result, np.ndarray)
        assert len(result) == traj.n_frames

    @pytest.mark.integration
    def test_rmsd_first_frame_is_zero(self, sample_pdb_path: Path, tmp_path: Path):
        """RMSD of reference frame should be zero."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import calculate_rmsd

        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_rmsd(
            traj_path, sample_pdb_path, atom_indices=list(range(33)), reference_frame=0
        )

        assert result[0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.integration
    def test_ligand_rmsd_returns_dict(self, sample_pdb_path: Path, tmp_path: Path):
        """Ligand RMSD should return dict with ligand and receptor."""
        pytest.importorskip("mdtraj")
        import mdtraj as md

        from binding_metrics.metrics.rmsd import calculate_ligand_rmsd

        traj = md.load(str(sample_pdb_path))
        traj_path = tmp_path / "test.dcd"
        traj.save_dcd(str(traj_path))

        result = calculate_ligand_rmsd(
            traj_path,
            sample_pdb_path,
            ligand_indices=list(range(23, 33)),
            receptor_indices=list(range(23)),
        )

        assert isinstance(result, dict)
        assert "ligand_rmsd" in result
        assert "receptor_rmsd" in result
