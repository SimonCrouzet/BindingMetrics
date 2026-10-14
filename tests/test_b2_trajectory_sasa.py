"""Buried SASA of a trajectory: the complex term holds the two selections only (#58).

The systems are built by hand: one carbon atom for the receptor (chain A), one for the ligand
(chain B), and, in the solvated variant, a water molecule far from both (chain C). Two spheres of
equal radius R = r + probe (carbon 0.17 nm in mdtraj, probe 0.14 nm, so R = 0.31 nm) at distance d
bury a cap of height h = R - d/2 on each other, with area 2 pi R h per sphere, so

    buried = 2 * (2 pi R h) = 2 pi R (2 R - d)   for d < 2 R, and 0 beyond.

At d = 0.40 nm that is 0.4285 nm^2 (42.85 A^2). Shrake-Rupley with 960 points lands within
1 % of it.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from binding_metrics.metrics.sasa import calculate_buried_sasa, calculate_interface_sasa

md = pytest.importorskip("mdtraj")

CARBON_SPHERE_NM = 0.17 + 0.14  # mdtraj carbon radius plus the default probe
# 960 test points leave an error of 0.2 to 1 % on the cap: 0.005 nm^2 on a 0.43 nm^2 area.
CAP_TOLERANCE_NM2 = 0.005
# mdtraj gives a lone carbon 1.2076 nm^2 in most frames and 1.2091 nm^2 in some (0.13 %).
SPHERE_TOLERANCE_NM2 = 0.003
NM2_TO_ANGSTROM2 = 100.0


def two_sphere_buried_nm2(distance_nm: float) -> float:
    return max(0.0, 2 * math.pi * CARBON_SPHERE_NM * (2 * CARBON_SPHERE_NM - distance_nm))


def make_system(tmp_path: Path, distances_nm, *, solvent: bool):
    """Write a DCD and a PDB of receptor (index 0), ligand (1) and, optionally, a water (2-4).

    Returns the two paths and the receptor and ligand index lists. The atoms sit along the x axis,
    one frame per distance; the water is 5 nm away from both.
    """
    top = md.Topology()
    for residue_name in ("REC", "LIG"):
        residue = top.add_residue(residue_name, top.add_chain())
        top.add_atom("C", md.element.carbon, residue)
    if solvent:
        water = top.add_residue("HOH", top.add_chain())
        top.add_atom("O", md.element.oxygen, water)
        top.add_atom("H1", md.element.hydrogen, water)
        top.add_atom("H2", md.element.hydrogen, water)
    frames = []
    for distance in distances_nm:
        atoms = [[1.0, 1.0, 1.0], [1.0 + distance, 1.0, 1.0]]
        if solvent:
            atoms += [[6.0, 6.0, 6.0], [6.1, 6.0, 6.0], [5.9, 6.0, 6.0]]
        frames.append(atoms)
    traj = md.Trajectory(np.array(frames, dtype=np.float32), top)
    dcd_path, pdb_path = tmp_path / "frames.dcd", tmp_path / "topology.pdb"
    traj.save_dcd(str(dcd_path))
    traj[0].save_pdb(str(pdb_path))
    return dcd_path, pdb_path, [0], [1]


DISTANCES_NM = [0.40, 0.50, 0.70]


class TestTheComplexTermHoldsTheSelectionsOnly:
    @pytest.mark.parametrize("solvent", [False, True], ids=["dry", "with_water"])
    def test_buried_sasa_follows_the_two_sphere_formula_per_frame(self, tmp_path, solvent):
        dcd, pdb, receptor, ligand = make_system(tmp_path, DISTANCES_NM, solvent=solvent)

        buried = calculate_buried_sasa(dcd, pdb, ligand_indices=ligand, receptor_indices=receptor)

        expected = [two_sphere_buried_nm2(d) for d in DISTANCES_NM]
        assert expected[0] == pytest.approx(0.4285, abs=1e-4)  # the hand value of the docstring
        assert expected[2] == 0.0
        assert buried.shape == (len(DISTANCES_NM),)
        assert buried == pytest.approx(expected, abs=CAP_TOLERANCE_NM2)

    def test_a_water_far_from_the_complex_changes_nothing(self, tmp_path):
        (tmp_path / "dry").mkdir()
        (tmp_path / "wet").mkdir()
        dry_dcd, dry_pdb, receptor, ligand = make_system(
            tmp_path / "dry", DISTANCES_NM, solvent=False
        )
        wet_dcd, wet_pdb, _, _ = make_system(tmp_path / "wet", DISTANCES_NM, solvent=True)

        dry_buried = calculate_buried_sasa(dry_dcd, dry_pdb, ligand, receptor)
        wet_buried = calculate_buried_sasa(wet_dcd, wet_pdb, ligand, receptor)

        # The water adds about 0.5 nm^2 to the old complex term, which is the whole buried area.
        np.testing.assert_allclose(wet_buried, dry_buried, atol=1e-6)

    def test_interface_sasa_complex_and_buried_terms_leave_the_water_out(self, tmp_path):
        dcd, pdb, receptor, ligand = make_system(tmp_path, DISTANCES_NM, solvent=True)

        terms = calculate_interface_sasa(dcd, pdb, ligand, receptor)

        sphere_nm2 = 4 * math.pi * CARBON_SPHERE_NM**2
        expected_buried = [two_sphere_buried_nm2(d) for d in DISTANCES_NM]
        assert terms["ligand"] == pytest.approx([sphere_nm2] * 3, abs=SPHERE_TOLERANCE_NM2)
        assert terms["receptor"] == pytest.approx([sphere_nm2] * 3, abs=SPHERE_TOLERANCE_NM2)
        assert terms["complex"] == pytest.approx(
            [2 * sphere_nm2 - b for b in expected_buried],
            abs=CAP_TOLERANCE_NM2 + 2 * SPHERE_TOLERANCE_NM2,
        )
        assert terms["buried"] == pytest.approx(expected_buried, abs=CAP_TOLERANCE_NM2)

    def test_the_two_functions_agree(self, tmp_path):
        dcd, pdb, receptor, ligand = make_system(tmp_path, DISTANCES_NM, solvent=True)

        np.testing.assert_array_equal(
            calculate_buried_sasa(dcd, pdb, ligand, receptor),
            calculate_interface_sasa(dcd, pdb, ligand, receptor)["buried"],
        )


class TestThePeptideProtocolReportsSquareAngstrom:
    """``ProtocolResults`` documents Å^2; the trajectory function returns nm^2 (x100)."""

    @pytest.fixture
    def analysed(self, tmp_path, monkeypatch):
        from binding_metrics.protocols import peptide

        dcd, pdb, _, _ = make_system(tmp_path, DISTANCES_NM, solvent=True)
        frames = np.zeros(len(DISTANCES_NM))
        # Only the SASA runs for real; the other three metrics need OpenMM and a force field.
        monkeypatch.setattr(peptide, "calculate_contacts", lambda *a, **k: frames)
        monkeypatch.setattr(peptide, "calculate_interaction_energy", lambda *a, **k: frames)
        monkeypatch.setattr(peptide, "calculate_rmsd", lambda *a, **k: frames)
        protocol = peptide.PeptideBindingProtocol(
            pdb_path=pdb, ligand_chain="B", receptor_chains=["A"]
        )
        protocol._topology_path = pdb  # what run() sets: the topology that holds the water
        results = protocol.analyze(trajectory_path=dcd)
        return results, dcd, pdb

    def test_per_frame_values_are_the_function_values_times_100(self, analysed):
        results, dcd, pdb = analysed

        in_nm2 = calculate_buried_sasa(dcd, pdb, ligand_indices=[1], receptor_indices=[0])

        np.testing.assert_allclose(results.sasa_buried, NM2_TO_ANGSTROM2 * in_nm2, rtol=1e-6)

    def test_values_are_the_hand_computed_areas_in_square_angstrom(self, analysed):
        results, _, _ = analysed

        # 2 pi R (2 R - d) with R = 3.1 A: 42.85 A^2 at d = 4.0 A, 23.37 A^2 at 5.0 A, 0 at 7.0 A.
        expected = [NM2_TO_ANGSTROM2 * two_sphere_buried_nm2(d) for d in DISTANCES_NM]
        assert expected[0] == pytest.approx(42.85, abs=0.01)
        assert results.sasa_buried == pytest.approx(
            expected, abs=NM2_TO_ANGSTROM2 * CAP_TOLERANCE_NM2
        )
        assert results.sasa_buried_mean == pytest.approx(np.mean(expected), abs=0.5)
        assert results.sasa_buried_std == pytest.approx(np.std(results.sasa_buried))

    def test_the_summary_prints_the_converted_mean(self, analysed):
        results, _, _ = analysed

        mean, std = results.sasa_buried_mean, results.sasa_buried_std
        assert f"Buried SASA: {mean:.1f} +/- {std:.1f} A^2" in results.summary()
        assert 20.0 < mean < 25.0  # (42.85 + 23.37 + 0) / 3 = 22.1 A^2, not 0.22
