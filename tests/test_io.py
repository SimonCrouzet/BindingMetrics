"""Tests for the I/O module."""

from pathlib import Path

import pytest

from binding_metrics.io.structures import (
    get_chain_atom_indices,
    get_residue_info,
    load_complex,
)


class TestLoadComplex:
    """Tests for load_complex function."""

    @pytest.mark.integration
    def test_load_valid_pdb(self, sample_pdb_path: Path):
        """Should successfully load a valid PDB file."""
        pdb = load_complex(sample_pdb_path)

        assert pdb is not None
        assert pdb.topology is not None
        assert pdb.positions is not None

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing file."""
        fake_path = tmp_path / "nonexistent.pdb"

        with pytest.raises(FileNotFoundError, match="not found"):
            load_complex(fake_path)

    @pytest.mark.integration
    def test_load_invalid_pdb_raises(self, tmp_path: Path):
        """Should raise ValueError for invalid PDB content."""
        bad_pdb = tmp_path / "bad.pdb"
        bad_pdb.write_text("This is not valid PDB content\nJust random text")

        with pytest.raises(ValueError, match="Failed to parse"):
            load_complex(bad_pdb)

    @pytest.mark.integration
    def test_load_accepts_path_object(self, sample_pdb_path: Path):
        """Should accept Path objects."""
        pdb = load_complex(sample_pdb_path)
        assert pdb is not None

    @pytest.mark.integration
    def test_load_accepts_string_path(self, sample_pdb_path: Path):
        """Should accept string paths."""
        pdb = load_complex(str(sample_pdb_path))
        assert pdb is not None


class TestGetChainAtomIndices:
    """Tests for get_chain_atom_indices function."""

    @pytest.mark.integration
    def test_get_single_chain_indices(self, sample_pdb_path: Path):
        """Should return indices for a single chain."""
        indices = get_chain_atom_indices(sample_pdb_path, ["A"])

        assert isinstance(indices, list)
        assert len(indices) > 0
        assert all(isinstance(i, int) for i in indices)

    @pytest.mark.integration
    def test_get_multiple_chain_indices(self, sample_pdb_path: Path):
        """Should return indices for multiple chains."""
        indices_a = get_chain_atom_indices(sample_pdb_path, ["A"])
        indices_b = get_chain_atom_indices(sample_pdb_path, ["B"])
        indices_ab = get_chain_atom_indices(sample_pdb_path, ["A", "B"])

        assert len(indices_ab) == len(indices_a) + len(indices_b)

    @pytest.mark.integration
    def test_nonexistent_chain_returns_empty(self, sample_pdb_path: Path):
        """Should return empty list for non-existent chain."""
        indices = get_chain_atom_indices(sample_pdb_path, ["Z"])

        assert indices == []

    @pytest.mark.integration
    def test_indices_are_zero_based(self, sample_pdb_path: Path):
        """Atom indices should be 0-based."""
        indices = get_chain_atom_indices(sample_pdb_path, ["A", "B"])

        if indices:
            assert min(indices) >= 0

    @pytest.mark.integration
    def test_chain_a_has_23_atoms(self, sample_pdb_path: Path):
        """Chain A (receptor) should have 23 atoms (with hydrogens)."""
        indices = get_chain_atom_indices(sample_pdb_path, ["A"])
        assert len(indices) == 23

    @pytest.mark.integration
    def test_chain_b_has_10_atoms(self, sample_pdb_path: Path):
        """Chain B (ligand) should have 10 atoms (with hydrogens)."""
        indices = get_chain_atom_indices(sample_pdb_path, ["B"])
        assert len(indices) == 10


class TestGetResidueInfo:
    """Tests for get_residue_info function."""

    @pytest.mark.integration
    def test_returns_list_of_dicts(self, sample_pdb_path: Path):
        """Should return a list of residue dictionaries."""
        residues = get_residue_info(sample_pdb_path)

        assert isinstance(residues, list)
        assert len(residues) > 0
        assert all(isinstance(r, dict) for r in residues)

    @pytest.mark.integration
    def test_residue_dict_has_required_keys(self, sample_pdb_path: Path):
        """Each residue dict should have name, index, chain, n_atoms."""
        residues = get_residue_info(sample_pdb_path)

        required_keys = {"name", "index", "chain", "n_atoms"}
        for residue in residues:
            assert required_keys.issubset(residue.keys())

    @pytest.mark.integration
    def test_detects_correct_residue_names(self, sample_pdb_path: Path):
        """Should detect ALA and GLY residues."""
        residues = get_residue_info(sample_pdb_path)
        names = {r["name"] for r in residues}

        assert "ALA" in names
        assert "GLY" in names

    @pytest.mark.integration
    def test_detects_correct_chains(self, sample_pdb_path: Path):
        """Should detect chains A and B."""
        residues = get_residue_info(sample_pdb_path)
        chains = {r["chain"] for r in residues}

        assert "A" in chains
        assert "B" in chains

    @pytest.mark.integration
    def test_residue_count(self, sample_pdb_path: Path):
        """Should find 3 residues total (2 ALA + 1 GLY)."""
        residues = get_residue_info(sample_pdb_path)
        assert len(residues) == 3
