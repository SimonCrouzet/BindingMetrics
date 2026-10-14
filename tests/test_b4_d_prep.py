"""Prep keeps a peptide made only of D-amino acids.

``prep_structure`` took a chain for a protein chain when it held one of the AMBER standard
residues. That set has no D-amino acid, so 1YCR with its 13 peptide residues renamed to
DGL, DTH, DPN, ... lost the peptide: prep removed it as 13 heterogens. A chain is now a
protein chain also when it holds an amino acid by the definition of ``detect_chains`` and
``drop_other_protein_chains``.
"""

import pytest

pytest.importorskip("openmm")

from conftest import DATA_DIR
from openmm import Vec3, app, unit

from binding_metrics.core.nonstandard import D_AA_MAP
from binding_metrics.io.structures import load_structure

P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"

L_TO_D = {l_name: d_name for d_name, l_name in D_AA_MAP.items()}


def _flip_to_d(topology, chain_id, every=1):
    """Rename every ``every``-th residue of a chain to its D form; return how many changed."""
    changed = 0
    for index, residue in enumerate(
        r for c in topology.chains() if c.id == chain_id for r in c.residues()
    ):
        if index % every == 0 and residue.name in L_TO_D:
            residue.name = L_TO_D[residue.name]
            changed += 1
    return changed


def _chain_sizes(topology):
    return {c.id: len(list(c.residues())) for c in topology.chains()}


@pytest.mark.slow
class TestPrepKeepsADPeptide:
    def _prep(self, every):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure

        topology, positions = load_structure(P53)
        flipped = _flip_to_d(topology, "B", every=every)
        report: dict = {}
        prepped, _ = prep_structure(topology, positions, report=report)
        return flipped, prepped, report

    def test_a_peptide_of_d_residues_is_kept(self):
        flipped, prepped, report = self._prep(every=1)

        assert flipped == 13
        sizes = _chain_sizes(prepped)
        assert (sizes["A"], sizes["B"]) == (85, 13)
        assert report["removed_heterogens"] == []
        assert len(report["kept_nonstandard"]) == 13
        assert all(entry.endswith("(chain B)") for entry in report["kept_nonstandard"])

    @pytest.mark.parametrize("every", [2, 3, 13])
    def test_a_mixed_peptide_is_kept_as_before(self, every):
        flipped, prepped, report = self._prep(every=every)

        assert 0 < flipped < 13
        assert _chain_sizes(prepped)["B"] == 13
        assert report["removed_heterogens"] == []

    def test_a_d_peptide_with_a_glycine_is_kept_as_before(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure

        topology, positions = load_structure(P53)
        _flip_to_d(topology, "B")
        residues = [r for c in topology.chains() if c.id == "B" for r in c.residues()]
        residues[3].name = "GLY"  # the one residue with no D form
        report: dict = {}
        prepped, _ = prep_structure(topology, positions, report=report)

        assert _chain_sizes(prepped)["B"] == 13
        assert report["removed_heterogens"] == []

    def test_a_ligand_chain_is_still_removed(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure

        topology, positions = load_structure(P53)
        _flip_to_d(topology, "B")
        modeller = app.Modeller(topology, positions)
        ligand_topology = app.Topology()
        ligand = ligand_topology.addResidue("EDO", ligand_topology.addChain("L"))
        ligand_topology.addAtom("C1", app.element.carbon, ligand)
        modeller.add(ligand_topology, [Vec3(3.0, 3.0, 3.0)] * unit.nanometer)
        report: dict = {}
        prepped, _ = prep_structure(modeller.topology, modeller.positions, report=report)

        assert "L" not in _chain_sizes(prepped)
        assert report["removed_heterogens"] == ["EDO (chain L)"]
