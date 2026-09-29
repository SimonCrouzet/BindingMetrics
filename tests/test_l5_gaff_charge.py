"""Ionisable groups that GAFF template generation builds in the neutral form.

Bond-order perception in :mod:`binding_metrics.core.gaff_ncaa` runs at total charge
0, so a carboxylic acid, phosphate, sulfate, primary amine or guanidine comes out
uncharged even though it is ionised at pH 7.4. The generator warns about it and
records the residue in ``NcaaTemplateList.neutral_ionizable_groups``.
"""

import logging
from pathlib import Path

import pytest

from binding_metrics.core.gaff_ncaa import (
    NcaaTemplateList,
    _neutral_ionizable_groups,
    parameterize_ncaa_residues,
)

DATA = Path(__file__).parent.parent / "data"
LOGGER_NAME = "binding_metrics.core.gaff_ncaa"

rdkit_chem = pytest.importorskip("rdkit.Chem")


def _capped_residue(side_chain_smiles: str):
    """Capped residue ``C-N-CA(side chain)-C(=O)-C`` as (mol, names, caps).

    The two outer carbons are the caps the generator adds at the external bonds.
    ``side_chain_smiles`` is written from the atom bonded to CA; its atoms are
    named ``S0``, ``S1``, ...  Hydrogens are explicit, as in the generator.
    """
    n_side = rdkit_chem.MolFromSmiles(side_chain_smiles).GetNumAtoms()
    mol = rdkit_chem.MolFromSmiles(f"CN[C@@H]({side_chain_smiles})C(=O)C")
    mol = rdkit_chem.AddHs(mol)
    c_index = 3 + n_side
    names = {1: "N", 2: "CA", c_index: "C", c_index + 1: "O"}
    names.update({3 + i: f"S{i}" for i in range(n_side)})
    caps = {0, c_index + 2}
    return mol, names, caps


def _groups(side_chain_smiles: str) -> list:
    mol, names, caps = _capped_residue(side_chain_smiles)
    return _neutral_ionizable_groups(mol, names, caps)


class TestNeutralIonizableGroups:
    @pytest.mark.parametrize(
        "side_chain, expected",
        [
            ("CCCN", [("primary amine", 1)]),  # ornithine
            ("CC(=O)O", [("carboxylic acid", -1)]),  # aspartate
            ("CCCNC(=N)N", [("guanidine", 1)]),  # arginine
            # Perception on a hydrogen-free residue keeps every bond single:
            ("CC(O)O", [("carboxylic acid", -1)]),  # aspartate as C(OH)2
            ("CCCNC(N)N", [("guanidine", 1)]),  # arginine as C(NH2)3
            ("COP(O)(O)(O)O", [("phosphate (two acidic OH)", -2)]),  # phosphate, single bonds
            ("COP(=O)(O)O", [("phosphate (two acidic OH)", -2)]),  # phosphoserine
            ("COP(=O)(O)OC", [("phosphate (one acidic OH)", -1)]),  # phosphodiester
            ("Cc1ccc(OS(=O)(=O)O)cc1", [("sulfate or sulfonic acid", -1)]),  # sulfotyrosine
        ],
    )
    def test_group_is_recognised(self, side_chain, expected):
        assert _groups(side_chain) == expected

    @pytest.mark.parametrize(
        "side_chain",
        [
            "CC(C)C",  # leucine
            "CC(N)=O",  # asparagine: amide NH2 is not basic
            "Cc1ccc(N)cc1",  # aromatic amine: pKa about 5
            "CC(=O)[O-]",  # already carboxylate in the perceived graph
            "CCC[NH3+]",  # already ammonium in the perceived graph
            "CCCNC(=[NH2+])N",  # already guanidinium in the perceived graph
            "CCNC(=O)C",  # acylated amine
        ],
    )
    def test_no_group_no_warning(self, side_chain):
        assert _groups(side_chain) == []

    def test_backbone_carbonyl_is_never_reported(self):
        """With the C-terminal cap missing, the backbone C(=O)OH is not a side-chain acid."""
        mol = rdkit_chem.AddHs(rdkit_chem.MolFromSmiles("CN[C@@H](C)C(=O)O"))
        names = {1: "N", 2: "CA", 3: "S0", 4: "C", 5: "O", 6: "OXT"}
        assert _neutral_ionizable_groups(mol, names, {0}) == []

    def test_two_groups_are_both_listed(self):
        assert sorted(_groups("CC(N)CC(=O)O")) == [("carboxylic acid", -1), ("primary amine", 1)]


def _prepared_peptide(renames: dict):
    """1YCR with the named residues (index -> new name) renamed to non-canonical codes.

    The new names must be real Chemical Component Dictionary codes whose atoms match
    the residue (the D-amino acids DLY, DAS, DAR and DLE have the atom names of
    LYS, ASP, ARG and LEU), so that the bond orders come from the dictionary as they
    do for any exotic residue in a PDB file.
    """
    from openmm.app import ForceField

    from binding_metrics.io.structures import load_structure

    path = DATA / "example_linear_p53_1YCR.pdb"
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    topology, positions = load_structure(path)
    residues = list(topology.residues())
    for index, name in renames.items():
        residues[index].name = name
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    return topology, positions, ff


def _first_index(name: str) -> int:
    from binding_metrics.io.structures import load_structure

    topology, _ = load_structure(DATA / "example_linear_p53_1YCR.pdb")
    return next(r.index for r in topology.residues() if r.name == name)


@pytest.mark.integration
class TestParameterizeReportsCharge:
    """Real GAFF template generation (antechamber / AM1-BCC) on a renamed residue."""

    @pytest.fixture(autouse=True)
    def _need_gaff(self):
        pytest.importorskip("openmmforcefields")
        pytest.importorskip("pdbfixer")

    @pytest.mark.parametrize(
        "source_residue, ncaa_code, expected_group",
        [
            ("LYS", "DLY", "primary amine"),
            ("ASP", "DAS", "carboxylic acid"),
            ("ARG", "DAR", "guanidine"),
        ],
    )
    def test_ionisable_residue_warns_and_is_recorded(
        self, caplog, source_residue, ncaa_code, expected_group
    ):
        topology, positions, ff = _prepared_peptide({_first_index(source_residue): ncaa_code})
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _, _, templates = parameterize_ncaa_residues(topology, positions, ff, verbose=False)

        assert isinstance(templates, NcaaTemplateList) and isinstance(templates, list)
        assert len(templates) == 1
        assert templates.net_charge_by_residue == {ncaa_code: pytest.approx(0.0, abs=1e-9)}
        assert templates.neutral_ionizable_groups == {ncaa_code: [expected_group]}
        messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
        assert len(messages) == 1
        assert f"'{ncaa_code}'" in messages[0] and expected_group in messages[0]
        assert "assumed neutral" in messages[0]

    def test_neutral_side_chain_is_silent(self, caplog):
        topology, positions, ff = _prepared_peptide({_first_index("LEU"): "DLE"})
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _, _, templates = parameterize_ncaa_residues(topology, positions, ff, verbose=False)
        assert len(templates) == 1
        assert templates.net_charge_by_residue == {"DLE": pytest.approx(0.0, abs=1e-9)}
        assert templates.neutral_ionizable_groups == {}
        assert templates.bond_order_source_by_residue == {"DLE": "ccd"}
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_without_ncaa_the_list_is_empty_but_typed(self):
        topology, positions, ff = _prepared_peptide({})
        _, _, templates = parameterize_ncaa_residues(topology, positions, ff, verbose=False)
        assert templates == []
        assert templates.net_charge_by_residue == {}
