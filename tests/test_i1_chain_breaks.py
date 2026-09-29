"""Chain breaks are reported by prep and warned about by the relaxation.

OpenMM bonds every residue to the next one of its chain by atom name, whatever the
distance, so a gap in the model is closed by the minimisation without a trace. The
fixture is the bundled 1YCR (MDM2 chain A, p53 peptide chain B, no gap in either) with
the second half of the peptide moved 0.6 nm along x.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

from openmm import Vec3, unit
from openmm.app import Modeller, PDBFile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"


def _peptide_moved_after(residue_id: str, shift_nm: float = 0.6):
    """1YCR with every residue of chain B from ``residue_id`` on moved along x."""
    pdb = PDBFile(str(P53_MDM2))
    moving = False
    moved = []
    positions = pdb.positions.value_in_unit(unit.nanometer)
    for residue in pdb.topology.residues():
        if residue.chain.id == "B" and residue.id == residue_id:
            moving = True
        for atom in residue.atoms():
            x, y, z = positions[atom.index]
            shift = shift_nm if moving and residue.chain.id == "B" else 0.0
            moved.append(Vec3(x + shift, y, z))
    return pdb.topology, unit.Quantity(moved, unit.nanometer)


def _c_n_distance_angstrom(topology, positions, before: str, after: str) -> float:
    atoms = {
        (a.residue.chain.id, a.residue.id, a.name): np.array(
            positions[a.index].value_in_unit(unit.angstrom)
        )
        for a in topology.atoms()
    }
    return float(np.linalg.norm(atoms[("B", before, "C")] - atoms[("B", after, "N")]))


class TestFindChainBreaks:
    def test_an_unbroken_complex_has_none(self):
        from binding_metrics.core.system import find_chain_breaks

        pdb = PDBFile(str(P53_MDM2))
        assert find_chain_breaks(pdb.topology, pdb.positions) == []

    def test_a_moved_segment_is_reported_with_its_residues_and_distance(self):
        from binding_metrics.core.system import find_chain_breaks

        topology, positions = _peptide_moved_after("23")
        (gap,) = find_chain_breaks(topology, positions)
        assert gap["chain"] == "B"
        assert (gap["residue_before"], gap["residue_after"]) == ("22", "23")
        expected = _c_n_distance_angstrom(topology, positions, "22", "23")
        assert gap["c_n_distance_angstrom"] == pytest.approx(expected, abs=0.01)
        assert gap["c_n_distance_angstrom"] > 4.0  # 1.33 A bond stretched by about 6 A

    def test_a_missing_atom_or_an_origin_placeholder_is_not_judged(self):
        from binding_metrics.core.system import find_chain_breaks

        topology, positions = _peptide_moved_after("23")
        # The N atom of residue 23 sits at the origin, as a placeholder does.
        moved = list(positions.value_in_unit(unit.nanometer))
        n_atom = next(a for a in topology.atoms() if a.residue.id == "23" and a.name == "N")
        moved[n_atom.index] = Vec3(0.0, 0.0, 0.0)
        assert find_chain_breaks(topology, unit.Quantity(moved, unit.nanometer)) == []
        # The C atom of residue 22 is absent.
        modeller = Modeller(topology, positions)
        c_atom = next(
            a for a in modeller.topology.atoms() if a.residue.id == "22" and a.name == "C"
        )
        modeller.delete([c_atom])
        assert find_chain_breaks(modeller.topology, modeller.positions) == []

    def test_waters_between_two_residues_do_not_hide_a_break(self):
        from openmm.app import Topology, element

        from binding_metrics.core.system import find_chain_breaks

        topology, positions = _peptide_moved_after("23")
        positions_nm = positions.value_in_unit(unit.nanometer)
        # Rebuild the topology with a water inside chain B, after residue 22.
        rebuilt = Topology()
        new_positions = []
        for old_chain in topology.chains():
            new_chain = rebuilt.addChain(old_chain.id)
            for residue in old_chain.residues():
                new_residue = rebuilt.addResidue(residue.name, new_chain, residue.id)
                for atom in residue.atoms():
                    rebuilt.addAtom(atom.name, atom.element, new_residue)
                    new_positions.append(Vec3(*positions_nm[atom.index]))
                if old_chain.id == "B" and residue.id == "22":
                    water = rebuilt.addResidue("HOH", new_chain, "900")
                    rebuilt.addAtom("O", element.oxygen, water)
                    new_positions.append(Vec3(2.0, 2.0, 2.0))
        (gap,) = find_chain_breaks(rebuilt, unit.Quantity(new_positions, unit.nanometer))
        assert (gap["residue_before"], gap["residue_after"]) == ("22", "23")


class TestPrepReportsChainBreaks:
    def test_report_key_is_an_empty_list_for_an_unbroken_input(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure

        pdb = PDBFile(str(P53_MDM2))
        report: dict = {}
        prep_structure(pdb.topology, pdb.positions, report=report)
        assert report["chain_breaks"] == []

    def test_a_break_is_in_the_report_and_warned_about(self, caplog):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure

        topology, positions = _peptide_moved_after("23")
        report: dict = {}
        with caplog.at_level(logging.WARNING, logger="binding_metrics.core.system"):
            prep_structure(topology, positions, report=report)
        (gap,) = report["chain_breaks"]
        assert (gap["chain"], gap["residue_before"], gap["residue_after"]) == ("B", "22", "23")
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Chain break in chain B" in m and "22 and 23" in m for m in messages)

    def test_prep_still_leaves_the_two_segments_where_they_are(self):
        """The report is the whole change: prep does not close or rebuild the gap."""
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import find_chain_breaks, prep_structure

        topology, positions = _peptide_moved_after("23")
        prepped_topology, prepped_positions = prep_structure(topology, positions)
        (gap,) = find_chain_breaks(prepped_topology, prepped_positions)
        assert gap["c_n_distance_angstrom"] > 4.0


class TestRelaxationWarns:
    def test_the_setup_warns_and_keeps_going(self, tmp_path, caplog):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import save_cif
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        topology, positions = prep_structure(*_peptide_moved_after("23"))
        path = tmp_path / "broken.cif"
        save_cif(topology, positions, path)
        relaxer = ImplicitRelaxation(RelaxationConfig(peptide_chain_id="B", receptor_chain_id="A"))
        with caplog.at_level(logging.WARNING, logger="binding_metrics.protocols.relaxation"):
            system, _, _, _ = relaxer._setup_system(path)
        assert system.getNumParticles() > 0
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Chain break in chain B" in m and "22 and 23" in m for m in messages)
