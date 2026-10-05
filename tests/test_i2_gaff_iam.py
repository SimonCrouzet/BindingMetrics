"""IAM in 1XY4 takes its bond orders from the component dictionary.

After PDBFixer the topology lists the IAM-THR peptide bond twice. The capped
molecule then had two carbon caps on the backbone carbon, so the dictionary
bond orders (C=O) gave that carbon a valence of five, RDKit refused to
sanitise them, and IAM fell back to single bonds: a cyclohexane instead of the
aromatic ring, 24 hydrogens on the prepped residue instead of 18.
"""

import pytest

rdkit_chem = pytest.importorskip("rdkit.Chem")
pytest.importorskip("biotite")

from pathlib import Path  # noqa: E402

from rdkit.Chem import rdMolDescriptors  # noqa: E402

from binding_metrics.core import gaff_ncaa  # noqa: E402
from binding_metrics.core.gaff_ncaa import (  # noqa: E402
    BOND_ORDER_SOURCE_CCD,
    _build_capped_molecule,
    _perceive_residue_bond_orders,
)

SOMATOSTATIN = Path(__file__).parent.parent / "data" / "example_lactam_somatostatin_1XY4.cif"

# 4-[(isopropylamino)methyl]phenylalanine inside a chain, capped by a methyl
# carbon on each side (CH3-NH-CA(R)-C(=O)-CH3), in the neutral form of the template.
CAPPED_IAM_SMILES = "CNC(Cc1ccc(CNC(C)C)cc1)C(C)=O"
CAPPED_IAM_FORMULA = "C15H24N2O"
IAM_HYDROGENS_IN_CHAIN = 18  # C13H20N2O2 minus one amine H and the carboxyl OH


@pytest.fixture()
def somatostatin():
    """1XY4 as (topology, positions in angstrom); a fresh copy per test, as tests edit it."""
    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(SOMATOSTATIN)
    return topology, gaff_ncaa._pos_to_angstrom(positions)


def _iam(topology):
    return next(r for r in topology.residues() if r.name == "IAM")


def _list_peptide_bond_twice(topology) -> None:
    """Add a second copy of the IAM C to THR N bond, as PDBFixer leaves it."""
    c_iam = next(a for a in topology.atoms() if a.residue.name == "IAM" and a.name == "C")
    n_thr = next(a for a in topology.atoms() if a.residue.name == "THR" and a.name == "N")
    assert sum(1 for b in topology.bonds() if {b.atom1, b.atom2} == {c_iam, n_thr}) == 1
    topology.addBond(c_iam, n_thr)


class TestCappedMolecule:
    def test_a_bond_listed_twice_gets_one_cap(self, somatostatin):
        topology, pos_A = somatostatin
        _list_peptide_bond_twice(topology)

        mol, names, caps, external, partner = _build_capped_molecule(
            _iam(topology), topology, pos_A
        )

        assert len(caps) == 2  # one at N (previous residue), one at C (next residue)
        assert sorted(external) == ["C", "N"]
        backbone_c = next(i for i, name in names.items() if name == "C")
        assert mol.GetAtomWithIdx(backbone_c).GetDegree() == 3  # CA, O and one cap
        assert sorted(partner.values()) == ["C", "N"]


class TestPerception:
    def test_iam_is_perceived_with_its_aromatic_ring_and_a_carbonyl(self, somatostatin):
        topology, pos_A = somatostatin
        _list_peptide_bond_twice(topology)  # the duplicate that broke the dictionary route
        res = _iam(topology)
        mol, names, _, _, _ = _build_capped_molecule(res, topology, pos_A)

        perceived, fallback_reason = _perceive_residue_bond_orders(mol, names, res.name)

        assert fallback_reason == ""
        assert rdkit_chem.MolToSmiles(rdkit_chem.RemoveHs(perceived)) == CAPPED_IAM_SMILES
        assert sum(atom.GetIsAromatic() for atom in perceived.GetAtoms()) == 6
        assert rdMolDescriptors.CalcMolFormula(rdkit_chem.AddHs(perceived)) == CAPPED_IAM_FORMULA

    def test_a_single_listed_bond_gives_the_same_molecule(self, somatostatin):
        topology, pos_A = somatostatin
        res = _iam(topology)
        mol, names, _, _, _ = _build_capped_molecule(res, topology, pos_A)

        perceived, fallback_reason = _perceive_residue_bond_orders(mol, names, res.name)

        assert fallback_reason == ""
        assert rdkit_chem.MolToSmiles(rdkit_chem.RemoveHs(perceived)) == CAPPED_IAM_SMILES


@pytest.mark.integration
def test_prep_of_somatostatin_reports_iam_from_the_dictionary():
    """The prepped topology carries the 18 hydrogens of the real residue."""
    pytest.importorskip("pdbfixer")
    pytest.importorskip("openmmforcefields")
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(SOMATOSTATIN)
    report: dict = {}

    prepped, _ = prep_structure(topology, positions, report=report)

    assert report["ncaa_bond_order_source"] == {"IAM": BOND_ORDER_SOURCE_CCD}
    n_hydrogens = sum(
        atom.element is not None and atom.element.symbol == "H" for atom in _iam(prepped).atoms()
    )
    assert n_hydrogens == IAM_HYDROGENS_IN_CHAIN
