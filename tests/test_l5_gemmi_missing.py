"""Behaviour of the structure I/O helpers when the optional gemmi package is absent.

gemmi is imported lazily inside the helpers, so hiding it from ``sys.modules``
(a ``None`` entry makes ``import gemmi`` raise ImportError) reproduces an
environment without it.
"""

import logging
import sys
from pathlib import Path

import pytest
from openmm import Vec3, unit
from openmm.app import Topology, element

from binding_metrics.io.structures import (
    _patch_nonstd_bonds_in_cif,
    extract_model_to_tempfile,
    load_structure,
    save_cif,
)

DATA = Path(__file__).parent.parent / "data"
SFTI_CIF = DATA / "example_bicyclic_sfti1_3P8F.cif"
LOGGER_NAME = "binding_metrics.io.structures"


@pytest.fixture
def no_gemmi(monkeypatch):
    monkeypatch.setitem(sys.modules, "gemmi", None)


def _peptide_topology(ring: bool):
    """Four ALA residues (N and C atoms only); ``ring`` adds a bond from residue 3 to 0."""
    topology = Topology()
    chain = topology.addChain("A")
    atoms = []
    for _ in range(4):
        residue = topology.addResidue("ALA", chain)
        atoms.append(
            (
                topology.addAtom("N", element.nitrogen, residue),
                topology.addAtom("C", element.carbon, residue),
            )
        )
    for i in range(3):
        topology.addBond(atoms[i][1], atoms[i + 1][0])
    if ring:
        topology.addBond(atoms[3][1], atoms[0][0])
    return topology


class TestExtractModelWithoutGemmi:
    def test_cif_with_requested_model_raises(self, no_gemmi):
        if not SFTI_CIF.exists():
            pytest.skip(f"bundled example not found: {SFTI_CIF}")
        with pytest.raises(ImportError, match="gemmi.*model 2"):
            extract_model_to_tempfile(SFTI_CIF, 2)

    def test_pdb_needs_no_gemmi(self, no_gemmi, tmp_path):
        pdb = DATA / "example_linear_p53_1YCR.pdb"
        if not pdb.exists():
            pytest.skip(f"bundled example not found: {pdb}")
        assert extract_model_to_tempfile(pdb, 1) == pdb

    def test_with_gemmi_model_one_is_extracted(self):
        """Positive control: the same call succeeds when gemmi is present."""
        pytest.importorskip("gemmi")
        if not SFTI_CIF.exists():
            pytest.skip(f"bundled example not found: {SFTI_CIF}")
        extracted = extract_model_to_tempfile(SFTI_CIF, 1)
        try:
            topology, _ = load_structure(extracted)
            expected, _ = load_structure(SFTI_CIF)
            assert topology.getNumAtoms() == expected.getNumAtoms()
        finally:
            if extracted != SFTI_CIF:
                extracted.unlink(missing_ok=True)


class TestPatchBondsWithoutGemmi:
    def test_ring_closure_bond_dropped_is_warned(self, no_gemmi, tmp_path, caplog):
        cif = tmp_path / "ring.cif"
        cif.write_text("data_x\n")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _patch_nonstd_bonds_in_cif(cif, _peptide_topology(ring=True))
        messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
        assert len(messages) == 1
        assert "1 ring-closure" in messages[0] and "ring.cif" in messages[0]
        assert cif.read_text() == "data_x\n", "the file must be left untouched"

    def test_linear_topology_stays_quiet(self, no_gemmi, tmp_path, caplog):
        cif = tmp_path / "linear.cif"
        cif.write_text("data_x\n")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _patch_nonstd_bonds_in_cif(cif, _peptide_topology(ring=False))
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]


class TestSaveCifWithoutGemmi:
    def test_source_ids_not_restored_is_warned_and_file_still_written(
        self, no_gemmi, tmp_path, caplog
    ):
        if not SFTI_CIF.exists():
            pytest.skip(f"bundled example not found: {SFTI_CIF}")
        topology, positions = load_structure(SFTI_CIF)
        out = tmp_path / "out.cif"
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            save_cif(topology, positions, out, source_cif_path=SFTI_CIF)
        messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
        assert len(messages) == 1
        assert "sequential chain IDs" in messages[0]
        reloaded, reloaded_positions = load_structure(out)
        assert reloaded.getNumAtoms() == topology.getNumAtoms()
        assert len(reloaded_positions) == len(positions)

    def test_without_source_a_linear_structure_writes_quietly(self, no_gemmi, tmp_path, caplog):
        topology = _peptide_topology(ring=False)
        positions = unit.Quantity([Vec3(0.1 * i, 0, 0) for i in range(8)], unit.nanometer)
        out = tmp_path / "linear.cif"
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            save_cif(topology, positions, out)
        assert out.exists()
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]
