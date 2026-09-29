"""Bond orders of GAFF non-canonical residues come from the Chemical Component Dictionary.

The RDKit molecule that :mod:`binding_metrics.core.gaff_ncaa` builds for a residue
has no hydrogens, and valence-based perception cannot recover unsaturation without
them: every bond stays single, so a phenylalanine becomes cyclohexylalanine, an
aspartate becomes C(OH)2 and the C=C of MeBmt in cyclosporin A is lost. The bond
orders are therefore read from the dictionary bundled with biotite.

Everything here is fast and CPU-only (no antechamber, no force-field build); the
GAFF template of cyclosporin is checked in ``test_gaff_ncaa.py``.
"""

import logging
from pathlib import Path

import pytest

rdkit_chem = pytest.importorskip("rdkit.Chem")
pytest.importorskip("biotite")

from rdkit.Chem import rdMolDescriptors  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402

from binding_metrics.core import gaff_ncaa  # noqa: E402
from binding_metrics.core.gaff_ncaa import (  # noqa: E402
    BOND_ORDER_SOURCE_CCD,
    BOND_ORDER_SOURCE_SINGLE_BONDS,
    _build_capped_molecule,
    _ccd_bond_orders,
    _ccd_heavy_atom_chemistry,
    _perceive_bond_orders,
    _perceive_residue_bond_orders,
)

DATA = Path(__file__).parent.parent / "data"
P53_PDB = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN_CIF = DATA / "example_ncaa_cyclosporin_1CWA.cif"
LOGGER_NAME = "binding_metrics.core.gaff_ncaa"

# A residue code that is not in the dictionary, and one that is but names another compound.
UNKNOWN_CODE = "QQQQQ"

# Capped residue as the generator builds it: the neighbouring residue is a methyl carbon on
# each side (CH3-NH-CA(R)-C(=O)-CH3), written here in the neutral form of the template.
CAPPED_STANDARD_RESIDUES = {
    "PHE": "CNC(Cc1ccccc1)C(C)=O",
    "ASP": "CNC(CC(=O)O)C(C)=O",
    "ARG": "CNC(CCCNC(=N)N)C(C)=O",
    "HIS": "CNC(Cc1c[nH]cn1)C(C)=O",
}

_TAUTOMERS = rdMolStandardize.TautomerEnumerator()


def _tautomer_key(mol) -> str:
    """Canonical SMILES that does not depend on where a mobile proton sits (HIS, ARG)."""
    return rdkit_chem.MolToSmiles(_TAUTOMERS.Canonicalize(rdkit_chem.RemoveHs(mol)))


def _formula(mol) -> str:
    return rdMolDescriptors.CalcMolFormula(rdkit_chem.AddHs(mol))


def _bond_between(mol, names: dict, name1: str, name2: str):
    by_name = {name: idx for idx, name in names.items()}
    return mol.GetBondBetweenAtoms(by_name[name1], by_name[name2])


@pytest.fixture(scope="module")
def p53():
    """1YCR (heavy atoms only) as (topology, positions in angstrom)."""
    if not P53_PDB.exists():
        pytest.skip(f"bundled example not found: {P53_PDB}")
    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(P53_PDB)
    return topology, gaff_ncaa._pos_to_angstrom(positions)


def _capped(p53, residue_name: str):
    """Capped molecule of the first residue of that name with a neighbour on both sides."""
    topology, pos_A = p53
    for res in topology.residues():
        if res.name != residue_name:
            continue
        mol, names, caps, _, _ = _build_capped_molecule(res, topology, pos_A)
        if len(caps) == 2:
            return res, mol, names
    pytest.skip(f"no interior {residue_name} in the example")


class TestStandardResiduesRoundTrip:
    """The same function that handles BMT, on residues whose chemistry is known."""

    @pytest.mark.parametrize("residue_name", sorted(CAPPED_STANDARD_RESIDUES))
    def test_perceived_molecule_is_the_true_residue(self, p53, residue_name):
        res, mol, names = _capped(p53, residue_name)
        perceived, fallback_reason = _perceive_residue_bond_orders(mol, names, res.name)

        assert fallback_reason == ""
        expected = rdkit_chem.MolFromSmiles(CAPPED_STANDARD_RESIDUES[residue_name])
        assert _tautomer_key(perceived) == _tautomer_key(expected)
        assert _formula(perceived) == _formula(expected)

    def test_side_chains_keep_their_unsaturation(self, p53):
        """PHE aromatic, ASP a carboxylic acid, ARG a guanidine with one C=N."""
        _, phe, phe_names = _capped(p53, "PHE")
        phe, _ = _perceive_residue_bond_orders(phe, phe_names, "PHE")
        assert sum(atom.GetIsAromatic() for atom in phe.GetAtoms()) == 6

        _, asp, asp_names = _capped(p53, "ASP")
        asp, _ = _perceive_residue_bond_orders(asp, asp_names, "ASP")
        assert asp.HasSubstructMatch(rdkit_chem.MolFromSmarts("[CX3](=O)[OX2H1]"))

        _, arg, arg_names = _capped(p53, "ARG")
        arg, _ = _perceive_residue_bond_orders(arg, arg_names, "ARG")
        assert arg.HasSubstructMatch(rdkit_chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]"))

    def test_the_hydrogen_free_molecule_is_not_perceivable_from_valence(self, p53):
        """Why the dictionary is needed: valence leaves a hydrogen-free graph single-bonded."""
        _, mol, _ = _capped(p53, "PHE")
        assert not any(atom.GetIsAromatic() for atom in _perceive_bond_orders(mol).GetAtoms())

    def test_ionisable_groups_come_out_neutral(self, p53):
        """The template convention is total charge 0; ARG's deposited +1 is removed."""
        _, mol, names = _capped(p53, "ARG")
        perceived, _ = _perceive_residue_bond_orders(mol, names, "ARG")
        assert all(atom.GetFormalCharge() == 0 for atom in perceived.GetAtoms())


class TestFallbackWhenTheDictionaryDoesNotApply:
    def test_unknown_code_is_not_in_the_dictionary(self):
        assert _ccd_heavy_atom_chemistry(UNKNOWN_CODE) is None

    def test_unknown_code_falls_back_to_single_bonds_and_says_why(self, p53):
        res, mol, names = _capped(p53, "PHE")
        perceived, reason = _perceive_residue_bond_orders(mol, names, UNKNOWN_CODE)

        assert "not in the Chemical Component Dictionary" in reason
        assert rdkit_chem.MolToSmiles(perceived) == rdkit_chem.MolToSmiles(
            _perceive_bond_orders(mol)
        )

    def test_entry_of_another_compound_is_rejected(self, p53):
        """A residue code reused for a different molecule must not borrow that molecule's bonds."""
        _, mol, names = _capped(p53, "PHE")
        assert _ccd_bond_orders(mol, names, "ALA")[0] is None  # ALA has no ring atoms
        assert "not in the dictionary entry" in _ccd_bond_orders(mol, names, "ALA")[1]

    def test_element_mismatch_is_rejected(self, p53):
        _, mol, names = _capped(p53, "PHE")
        swapped = dict(names)
        n_index = next(i for i, name in names.items() if name == "N")
        swapped[n_index] = "CA"  # a nitrogen labelled as the alpha carbon
        perceived, reason = _ccd_bond_orders(mol, swapped, "PHE")
        assert perceived is None and "is not a C" in reason

    def test_extra_topology_bond_is_rejected(self, p53):
        _, mol, names = _capped(p53, "PHE")
        by_name = {name: idx for idx, name in names.items()}
        rw = rdkit_chem.RWMol(mol)
        rw.AddBond(by_name["CB"], by_name["CD1"], rdkit_chem.BondType.SINGLE)
        perceived, reason = _ccd_bond_orders(rw.GetMol(), names, "PHE")
        assert perceived is None and "CB-CD1" in reason

    def test_missing_topology_bond_is_rejected(self, p53):
        _, mol, names = _capped(p53, "PHE")
        by_name = {name: idx for idx, name in names.items()}
        rw = rdkit_chem.RWMol(mol)
        rw.RemoveBond(by_name["CG"], by_name["CD2"])
        perceived, reason = _ccd_bond_orders(rw.GetMol(), names, "PHE")
        assert perceived is None and "CD2-CG" in reason


@pytest.fixture(scope="module")
def cyclosporin():
    """Cyclosporin A after the pre-GAFF topology patches, as (topology, positions in angstrom)."""
    if not CYCLOSPORIN_CIF.exists():
        pytest.skip(f"bundled example not found: {CYCLOSPORIN_CIF}")
    pytest.importorskip("pdbfixer")
    from pdbfixer import PDBFixer

    from binding_metrics.core.cyclic import patch_cyclic_topology, rename_disulfide_cys_to_cyx
    from binding_metrics.core.nonstandard import detect_nonstandard, patch_nonstandard

    fixer = PDBFixer(filename=str(CYCLOSPORIN_CIF))
    topology, positions = fixer.topology, fixer.positions
    water = {"HOH", "WAT", "H2O"}
    sizes = sorted(
        (c.id, sum(1 for r in c.residues() if r.name not in water)) for c in topology.chains()
    )
    peptide_chain = min((s for s in sizes if s[1] > 0), key=lambda s: s[1])[0]
    ns_info = detect_nonstandard(topology, peptide_chain)
    topology, positions = patch_nonstandard(topology, positions, peptide_chain, ns_info)
    topology, positions, _ = patch_cyclic_topology(topology, positions, peptide_chain)
    topology, positions = rename_disulfide_cys_to_cyx(topology, positions)
    return topology, gaff_ncaa._pos_to_angstrom(positions)


def _perceived_cyclosporin_residue(cyclosporin, residue_name: str):
    topology, pos_A = cyclosporin
    res = next(r for r in topology.residues() if r.name == residue_name)
    mol, names, caps, _, _ = _build_capped_molecule(res, topology, pos_A)
    assert len(caps) == 2  # head-to-tail macrocycle: a neighbour on both sides
    perceived, fallback_reason = _perceive_residue_bond_orders(mol, names, res.name)
    assert fallback_reason == ""
    return perceived, names


class TestCyclosporinResidues:
    """BMT and ABA are the exotic residues of cyclosporin A (1CWA chain B)."""

    def test_bmt_keeps_its_alkene(self, cyclosporin):
        # MeBmt = (4R)-4-[(E)-2-butenyl]-4,N-dimethyl-L-threonine: the side chain
        # CD2-CE=CZ-CH3 carries a C=C, and the backbone C is a carbonyl.
        bmt, names = _perceived_cyclosporin_residue(cyclosporin, "BMT")

        assert _bond_between(bmt, names, "CE", "CZ").GetBondType() == rdkit_chem.BondType.DOUBLE
        assert _bond_between(bmt, names, "C", "O").GetBondType() == rdkit_chem.BondType.DOUBLE
        assert _bond_between(bmt, names, "CD2", "CE").GetBondType() == rdkit_chem.BondType.SINGLE
        assert rdkit_chem.MolToSmiles(bmt) == rdkit_chem.MolToSmiles(
            rdkit_chem.MolFromSmiles("CC=CCC(C)C(O)C(C(C)=O)N(C)C")
        )

    def test_bmt_has_17_hydrogens_not_19(self, cyclosporin):
        """Free MeBmt is C10H19NO3; in the chain the acid OH and the N-H are gone (17 H).

        The two methyl caps add C2H6, so the capped molecule is C12H23NO2. With every
        bond single it had two more H (the alkene CH=CH read as CH2-CH2).
        """
        bmt, _ = _perceived_cyclosporin_residue(cyclosporin, "BMT")
        assert _formula(bmt) == "C12H23NO2"

    def test_aba_is_saturated_with_a_carbonyl(self, cyclosporin):
        aba, names = _perceived_cyclosporin_residue(cyclosporin, "ABA")

        assert _bond_between(aba, names, "C", "O").GetBondType() == rdkit_chem.BondType.DOUBLE
        assert all(
            bond.GetBondType() == rdkit_chem.BondType.SINGLE
            for bond in aba.GetBonds()
            if {names.get(bond.GetBeginAtomIdx()), names.get(bond.GetEndAtomIdx())} != {"C", "O"}
        )
        assert rdkit_chem.MolToSmiles(aba) == rdkit_chem.MolToSmiles(
            rdkit_chem.MolFromSmiles("CCC(NC)C(C)=O")
        )
        assert _formula(aba) == "C6H13NO"  # free ABA is C4H9NO2; the caps add C2H6

    def test_perception_keeps_the_conformer(self, cyclosporin):
        """Hydrogens are placed from the 3D coordinates afterwards, so they must survive."""
        bmt, _ = _perceived_cyclosporin_residue(cyclosporin, "BMT")
        assert bmt.GetNumConformers() == 1
        assert bmt.GetConformer().GetNumAtoms() == bmt.GetNumAtoms()


class TestBondOrderSourceIsReported:
    """``parameterize_ncaa_residues`` records where each residue's bond orders came from."""

    @pytest.fixture(autouse=True)
    def _need_openmm_gaff(self):
        pytest.importorskip("openmmforcefields")

    @staticmethod
    def _stub_generator(monkeypatch, fallback_reason: str):
        """Replace template generation (needs antechamber) by a fixed one-atom template."""
        template = (
            "<ForceField><Residues><Residue name='QQQQQ'>"
            "<Atom name='CA' type='t' charge='0.0'/></Residue></Residues></ForceField>"
        )
        monkeypatch.setattr(
            gaff_ncaa,
            "_generate_residue_template",
            lambda *args, **kwargs: (template, [], [], fallback_reason),
        )
        monkeypatch.setattr(gaff_ncaa, "_load_ffxml", lambda ff, xml: None)

    @staticmethod
    def _renamed_p53(residue_name: str = UNKNOWN_CODE):
        from openmm.app import ForceField

        from binding_metrics.io.structures import load_structure

        if not P53_PDB.exists():
            pytest.skip(f"bundled example not found: {P53_PDB}")
        topology, positions = load_structure(P53_PDB)
        next(r for r in topology.residues() if r.name == "PHE").name = residue_name
        return topology, positions, ForceField("amber14-all.xml")

    def test_single_bond_fallback_is_recorded_and_warned_once(self, monkeypatch, caplog):
        self._stub_generator(monkeypatch, "'QQQQQ' is not in the Chemical Component Dictionary")
        topology, positions, ff = self._renamed_p53()
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _, _, templates = gaff_ncaa.parameterize_ncaa_residues(
                topology, positions, ff, verbose=False
            )

        assert templates.bond_order_source_by_residue == {
            UNKNOWN_CODE: BOND_ORDER_SOURCE_SINGLE_BONDS
        }
        messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
        assert len(messages) == 1
        assert "single bonds only" in messages[0] and "not in the Chemical Component" in messages[0]

    def test_dictionary_bond_orders_are_recorded_without_a_warning(self, monkeypatch, caplog):
        self._stub_generator(monkeypatch, "")
        topology, positions, ff = self._renamed_p53()
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _, _, templates = gaff_ncaa.parameterize_ncaa_residues(
                topology, positions, ff, verbose=False
            )

        assert templates.bond_order_source_by_residue == {UNKNOWN_CODE: BOND_ORDER_SOURCE_CCD}
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]
