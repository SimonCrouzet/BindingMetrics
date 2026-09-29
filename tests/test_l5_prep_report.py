"""The optional ``report`` dict of ``prep_structure`` and ``strip_heterogens``.

The report records what prep removed, kept and rebuilt, so a result file can say
why an atom count changed. Behaviour with ``report=None`` must not change.
"""

from pathlib import Path

import numpy as np
import pytest
from openmm import Vec3, unit
from openmm.app import Modeller, Topology, element

from binding_metrics.core.system import _count_residue_gaps, prep_structure
from binding_metrics.io.structures import load_structure, strip_heterogens

DATA = Path(__file__).parent.parent / "data"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"


def _load(path):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return load_structure(path)


def _add_ion(topology, positions, name, symbol, chain_id, xyz_nm):
    """Return (topology, positions) with one single-atom residue in its own chain."""
    ion_top = Topology()
    residue = ion_top.addResidue(name, ion_top.addChain(chain_id))
    ion_top.addAtom(name, element.Element.getBySymbol(symbol), residue)
    modeller = Modeller(topology, positions)
    modeller.add(ion_top, [Vec3(*xyz_nm)] * unit.nanometer)
    return modeller.topology, modeller.positions


def _backbone_residue(topology, chain, name, res_id, origin_angstrom, with_c=True):
    """Add an N-CA-C residue whose N sits at ``origin_angstrom`` along x."""
    residue = topology.addResidue(name, chain, id=str(res_id))
    topology.addAtom("N", element.nitrogen, residue)
    topology.addAtom("CA", element.carbon, residue)
    if with_c:
        topology.addAtom("C", element.carbon, residue)
    x = origin_angstrom
    xyz = [(x, 0, 0), (x + 1.46, 0, 0), (x + 2.98, 0, 0)]
    return residue, xyz if with_c else xyz[:2]


def _chain_of_backbones(specs):
    """Build a one-chain topology from (res_id, origin_angstrom, with_c) triples."""
    topology = Topology()
    chain = topology.addChain("A")
    coords = []
    for res_id, origin, with_c in specs:
        _, xyz = _backbone_residue(topology, chain, "ALA", res_id, origin, with_c)
        coords.extend(xyz)
    positions = unit.Quantity([Vec3(*c) / 10.0 for c in coords], unit.nanometer)
    return topology, positions


class TestCountResidueGaps:
    """Synthetic backbones: C(i)-N(i+1) is 1.33 A when bonded, 3.9 A across a gap."""

    def test_continuous_chain_has_no_gap(self):
        top, pos = _chain_of_backbones([(1, 0.0, True), (2, 4.31, True), (3, 8.62, True)])
        assert _count_residue_gaps(top, pos) == 0

    def test_numbering_jump_with_broken_bond_is_a_gap(self):
        # residue 3 -> 7: four unresolved residues, neighbours 6.8 A apart
        top, pos = _chain_of_backbones([(1, 0.0, True), (2, 4.31, True), (7, 11.0, True)])
        assert _count_residue_gaps(top, pos) == 1

    def test_two_separate_gaps_are_counted_separately(self):
        specs = [(1, 0.0, True), (4, 8.0, True), (9, 16.0, True)]
        top, pos = _chain_of_backbones(specs)
        assert _count_residue_gaps(top, pos) == 2

    def test_renumbering_with_intact_bond_is_not_a_gap(self):
        # numbers jump 2 -> 101 but the peptide bond is intact (C-N = 1.33 A)
        top, pos = _chain_of_backbones([(1, 0.0, True), (2, 4.31, True), (101, 8.62, True)])
        assert _count_residue_gaps(top, pos) == 0

    def test_missing_carbon_falls_back_to_numbering(self):
        top, pos = _chain_of_backbones([(1, 0.0, False), (5, 4.31, True)])
        assert _count_residue_gaps(top, pos) == 1

    def test_non_peptide_residues_are_ignored(self):
        top = Topology()
        chain = top.addChain("A")
        coords = []
        for res_id, origin in ((1, 0.0), (2, 4.31)):
            _, xyz = _backbone_residue(top, chain, "ALA", res_id, origin)
            coords.extend(xyz)
        water = top.addResidue("HOH", chain, id="500")
        top.addAtom("O", element.oxygen, water)
        coords.append((30.0, 0, 0))
        pos = unit.Quantity([Vec3(*c) / 10.0 for c in coords], unit.nanometer)
        assert _count_residue_gaps(top, pos) == 0


class TestStripHeterogensReport:
    def test_report_is_optional_and_does_not_change_the_result(self):
        top, pos = _load(CYCLOSPORIN)
        plain_top, plain_pos = strip_heterogens(top, pos, "B", "A")
        report: dict = {}
        rep_top, rep_pos = strip_heterogens(top, pos, "B", "A", report=report)
        assert rep_top.getNumAtoms() == plain_top.getNumAtoms()
        np.testing.assert_allclose(
            np.array(rep_pos.value_in_unit(unit.nanometer)),
            np.array(plain_pos.value_in_unit(unit.nanometer)),
        )

    def test_waters_are_counted_on_cyclosporin(self):
        """1CWA carries 144 crystallographic waters in two solvent chains, no ligands."""
        top, pos = _load(CYCLOSPORIN)
        n_water_atoms = sum(len(list(r.atoms())) for r in top.residues() if r.name == "HOH")
        report: dict = {}
        out_top, _ = strip_heterogens(top, pos, "B", "A", report=report)
        assert report["n_removed_waters"] == 144
        assert report["removed_heterogens"] == []
        assert out_top.getNumAtoms() == top.getNumAtoms() - n_water_atoms

    def test_ion_and_ligand_are_listed_by_name_and_chain(self):
        top, pos = _load(CYCLOSPORIN)
        top, pos = _add_ion(top, pos, "CL", "Cl", "X", (0.1, 0.1, 0.1))
        top, pos = _add_ion(top, pos, "ZN", "Zn", "Y", (9.0, 9.0, 9.0))
        report: dict = {}
        strip_heterogens(top, pos, "B", "A", report=report)
        assert sorted(report["removed_heterogens"]) == ["CL (chain X)", "ZN (chain Y)"]
        assert report["n_removed_waters"] == 144

    def test_report_accumulates_over_calls(self):
        top, pos = _load(CYCLOSPORIN)
        report: dict = {}
        strip_heterogens(top, pos, "B", "A", report=report)
        strip_heterogens(top, pos, "B", "A", report=report)
        assert report["n_removed_waters"] == 288


@pytest.fixture(scope="module")
def cyclosporin_report():
    """One prep of the cyclic NCAA example (GAFF templates make it slow)."""
    pytest.importorskip("pdbfixer")
    top, pos = _load(CYCLOSPORIN)
    report: dict = {}
    prep_structure(top, pos, report=report)
    return report


@pytest.mark.integration
class TestPrepStructureReport:
    def test_cyclosporin_waters_and_kept_ncaas(self, cyclosporin_report):
        assert cyclosporin_report["n_removed_waters"] == 144
        assert cyclosporin_report["removed_heterogens"] == []
        kept = cyclosporin_report["kept_nonstandard"]
        assert sorted(set(kept)) == sorted(
            {f"{n} (chain B)" for n in ("DAL", "MLE", "MVA", "BMT", "ABA", "SAR")}
        )
        assert kept.count("MLE (chain B)") == 4

    def test_cyclosporin_has_no_gap(self, cyclosporin_report):
        assert cyclosporin_report["n_missing_residue_gaps"] == 0

    def test_keep_water_reports_zero_removed_waters(self):
        pytest.importorskip("pdbfixer")
        top, pos = _load(P53_MDM2)
        report: dict = {}
        prep_structure(top, pos, keep_water=True, report=report)
        assert report["n_removed_waters"] == 0

    def test_ions_are_split_into_kept_and_removed(self):
        """Chloride outside the protein chains is removed; sodium is kept as a metal."""
        pytest.importorskip("pdbfixer")
        top, pos = _load(P53_MDM2)
        top, pos = _add_ion(top, pos, "CL", "Cl", "X", (5.0, 5.0, 5.0))
        top, pos = _add_ion(top, pos, "NA", "Na", "Y", (6.0, 6.0, 6.0))
        report: dict = {}
        out_top, _ = prep_structure(top, pos, report=report)
        assert report["removed_heterogens"] == ["CL (chain X)"]
        assert report["kept_nonstandard"] == ["NA (metal, chain Y)"]
        names = {r.name for r in out_top.residues()}
        assert "CL" not in names and "NA" in names

    def test_deleted_side_chain_atoms_are_counted_as_rebuilt(self):
        """Removing the CG/CD1/CD2 of one leucine adds exactly three rebuilt atoms."""
        pytest.importorskip("pdbfixer")
        top, pos = _load(P53_MDM2)
        baseline: dict = {}
        prep_structure(top, pos, report=baseline)

        modeller = Modeller(top, pos)
        leucine = next(r for r in modeller.topology.residues() if r.name == "LEU")
        modeller.delete([a for a in leucine.atoms() if a.name in ("CG", "CD1", "CD2")])
        damaged: dict = {}
        out_top, _ = prep_structure(modeller.topology, modeller.positions, report=damaged)

        assert damaged["n_missing_atoms_rebuilt"] - baseline["n_missing_atoms_rebuilt"] == 3
        rebuilt = {a.name for a in list(out_top.residues())[leucine.index].atoms()}
        assert {"CG", "CD1", "CD2"} <= rebuilt

    def test_deleted_residue_is_reported_as_a_gap(self):
        pytest.importorskip("pdbfixer")
        top, pos = _load(P53_MDM2)
        baseline: dict = {}
        prep_structure(top, pos, report=baseline)
        assert baseline["n_missing_residue_gaps"] == 0

        modeller = Modeller(top, pos)
        chain_a = next(c for c in modeller.topology.chains() if c.id == "A")
        modeller.delete([list(chain_a.residues())[10]])
        gapped: dict = {}
        prep_structure(modeller.topology, modeller.positions, report=gapped)
        assert gapped["n_missing_residue_gaps"] == 1

    def test_prep_without_report_is_unchanged(self):
        pytest.importorskip("pdbfixer")
        top, pos = _load(P53_MDM2)
        plain_top, plain_pos = prep_structure(top, pos)
        rep_top, rep_pos = prep_structure(top, pos, report={})
        assert rep_top.getNumAtoms() == plain_top.getNumAtoms()
        heavy = np.array([a.element.symbol != "H" for a in plain_top.atoms()])
        delta = np.abs(
            np.array(rep_pos.value_in_unit(unit.nanometer))
            - np.array(plain_pos.value_in_unit(unit.nanometer))
        )
        assert delta[heavy].max() < 1e-6
