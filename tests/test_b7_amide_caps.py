"""Backbone charges of a non-canonical residue come from an amide, not an amine (#61).

The GAFF2 route computes the AM1-BCC charges of a residue with a cap at each external bond.
A methyl carbon on the backbone N turns the peptide amide into an amine, and the N of ALA then
carries a more negative charge than the amide N of ff14SB. A residue with an N-CA-C=O backbone
is capped with ACE and NME at its peptide bonds instead, which gives the N of an amide.

The fast tests need RDKit only: the caps of real residues of the bundled examples, the charge
arithmetic and the record of the scheme. The slow tests run antechamber (sqm) on ALA, on the
charged ASP and LYS, and on the cyclosporin residues of 1CWA, and compare the charges of the
ACE/NME-capped residue with those of the methyl-capped one and with ff14SB.
"""

import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import DATA_DIR

pytest.importorskip("rdkit")
pytest.importorskip("openff.toolkit")
from rdkit import Chem  # noqa: E402

from binding_metrics.core import gaff_ncaa  # noqa: E402
from binding_metrics.core.gaff_ncaa import (  # noqa: E402
    CAP_KIND_AMIDE,
    CAP_KIND_METHYL,
    CAP_KIND_NONE,
    CHARGE_SCHEME_VERSION,
    CapScheme,
    _build_capped_molecule,
    _cap_backbone_as_amide,
    _charges_with_integer_total,
    _perceive_residue_bond_orders,
)

LOGGER_NAME = "binding_metrics.core.gaff_ncaa"
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
STAPLE = DATA_DIR / "example_staple_3V3B.pdb"
SFTI1 = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"


def _load(path):
    """``(topology, positions in angstrom)`` of a bundled example; skip if it is not there."""
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(path)
    return topology, gaff_ncaa._pos_to_angstrom(positions)


@pytest.fixture(scope="module")
def p53():
    gaff_ncaa._ccd_heavy_atom_chemistry("SER")  # the first lookup reads the whole dictionary
    return _load(P53)


@pytest.fixture(scope="module")
def cyclosporin():
    return _load(CYCLOSPORIN)


def _residue(structure, name, n_caps, chain_id=None):
    """First residue called ``name`` (in ``chain_id``) that has ``n_caps`` external bonds."""
    topology, pos_A = structure
    for res in topology.residues():
        if res.name != name or (chain_id is not None and res.chain.id != chain_id):
            continue
        if len(_build_capped_molecule(res, topology, pos_A)[2]) == n_caps:
            return res
    pytest.skip(f"no {name} with {n_caps} external bond(s) in the example")


def _perceived(structure, res):
    """``(mol, names, caps, partner)`` of the residue with its bond orders perceived."""
    topology, pos_A = structure
    mol, names, caps, _, partner = _build_capped_molecule(res, topology, pos_A)
    mol, _ = _perceive_residue_bond_orders(mol, names, res.name)
    return mol, names, caps, partner


def _smiles(mol) -> str:
    return Chem.MolToSmiles(mol)


def _canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


class TestBackboneCapsAreAmideGroups:
    @pytest.mark.parametrize(
        "residue_name, methyl_capped, amide_capped",
        [
            ("SER", "CNC(CO)C(C)=O", "CC(=O)NC(CO)C(=O)NC"),
            ("PHE", "CNC(Cc1ccccc1)C(C)=O", "CC(=O)NC(Cc1ccccc1)C(=O)NC"),
        ],
    )
    def test_residue_between_two_residues_gets_ace_and_nme(
        self, p53, residue_name, methyl_capped, amide_capped
    ):
        res = _residue(p53, residue_name, 2)
        mol, names, caps, partner = _perceived(p53, res)
        assert _smiles(mol) == _canonical(methyl_capped)  # what the earlier code computed on

        amide, amide_caps, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme == CapScheme(CAP_KIND_AMIDE)
        assert _smiles(amide) == _canonical(amide_capped)
        assert Chem.GetFormalCharge(amide) == 0
        assert caps < amide_caps and len(amide_caps) - len(caps) == 3  # ACE: O, CH3; NME: CH3
        assert not amide_caps & set(names)  # no residue atom is turned into a cap atom

    def test_alanine_of_cyclophilin(self, cyclosporin):
        res = _residue(cyclosporin, "ALA", 2, chain_id="A")
        mol, names, caps, partner = _perceived(cyclosporin, res)

        amide, _, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_AMIDE
        assert _smiles(amide) == _canonical("CC(=O)NC(C)C(=O)NC")

    def test_the_cap_atoms_have_coordinates_and_the_residue_keeps_its_own(self, p53):
        res = _residue(p53, "SER", 2)
        mol, names, caps, partner = _perceived(p53, res)
        before = mol.GetConformer().GetPositions()

        amide, amide_caps, _ = _cap_backbone_as_amide(mol, names, caps, partner)

        after = amide.GetConformer().GetPositions()
        assert after.shape == (amide.GetNumAtoms(), 3) and np.isfinite(after).all()
        np.testing.assert_allclose(after[: len(before)], before)
        # ACE and NME atoms lie where a peptide bond puts them: 1.2 to 1.6 A from the cap atom.
        for added in sorted(amide_caps - caps):
            neighbours = [n.GetIdx() for n in amide.GetAtomWithIdx(added).GetNeighbors()]
            assert len(neighbours) == 1
            distance = np.linalg.norm(after[added] - after[neighbours[0]])
            assert 1.2 <= distance <= 1.6

    def test_the_input_molecule_and_sets_are_not_edited(self, p53):
        res = _residue(p53, "SER", 2)
        mol, names, caps, partner = _perceived(p53, res)
        smiles, n_atoms, cap_copy = _smiles(mol), mol.GetNumAtoms(), set(caps)

        _cap_backbone_as_amide(mol, names, caps, partner)

        assert (_smiles(mol), mol.GetNumAtoms(), caps) == (smiles, n_atoms, cap_copy)

    def test_a_residue_at_a_chain_end_gets_only_the_nme(self, p53):
        res = _residue(p53, "GLU", 1, chain_id="B")  # first residue of the peptide
        mol, names, caps, partner = _perceived(p53, res)

        amide, amide_caps, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_AMIDE
        assert _smiles(amide) == _canonical("NC(CCC(=O)O)C(=O)NC")  # the N stays a free amine
        assert len(amide_caps) - len(caps) == 1

    def test_a_side_chain_link_keeps_the_methyl_cap(self):
        """Residue 0EH of the staple has peptide bonds on both sides and a C-C cross-link."""
        structure = _load(STAPLE)
        res = _residue(structure, "0EH", 3)
        mol, names, caps, partner = _perceived(structure, res)

        amide, amide_caps, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_AMIDE
        assert len(amide_caps) == 3 + 2 + 1  # ACE, NME, and the methyl of the cross-link
        link = next(c for c, p in partner.items() if p not in ("C", "N"))
        linked_atom = amide.GetAtomWithIdx(link)
        assert linked_atom.GetAtomicNum() == 6 and linked_atom.GetDegree() == 1
        assert names[linked_atom.GetNeighbors()[0].GetIdx()] == "CAT"

    def test_a_residue_without_the_backbone_atoms_keeps_the_methyl_caps(self):
        """Glutathione of 3P8F is bound to a cysteine by its sulfur and has no N, CA, C, O."""
        structure = _load(SFTI1)
        res = _residue(structure, "GSH", 1)
        mol, names, caps, partner = _perceived(structure, res)

        same, same_caps, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_METHYL and "backbone" in scheme.reason
        assert same is mol and same_caps == caps

    def test_a_cap_for_a_side_chain_atom_is_not_completed(self, p53):
        """The rule is the partner: a cap on the N that stands for no C stays a methyl."""
        res = _residue(p53, "SER", 2)
        mol, names, caps, partner = _perceived(p53, res)
        n_cap = next(
            c for c in caps if names[mol.GetAtomWithIdx(c).GetNeighbors()[0].GetIdx()] == "N"
        )
        partner = {**partner, n_cap: "SG"}

        amide, amide_caps, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_AMIDE
        assert _smiles(amide) == _canonical("CNC(CO)C(=O)NC")  # only the NME
        assert len(amide_caps) - len(caps) == 1

    def test_no_peptide_bond_keeps_the_methyl_caps(self, p53):
        res = _residue(p53, "SER", 2)
        mol, names, caps, partner = _perceived(p53, res)
        partner = {cap: "SG" for cap in partner}

        same, same_caps, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_METHYL and "peptide bond" in scheme.reason
        assert same is mol and same_caps == caps

    def test_a_residue_without_external_bond_is_uncapped(self, p53):
        res = _residue(p53, "SER", 2)
        mol, names, _, _ = _perceived(p53, res)

        same, caps, scheme = _cap_backbone_as_amide(mol, names, set(), {})

        assert scheme == CapScheme(CAP_KIND_NONE) and same is mol and caps == set()

    def test_the_backbone_carbonyl_is_a_double_bond_after_the_single_bond_fallback(self, p53):
        """A code the component dictionary does not know is built with single bonds only."""
        res = _residue(p53, "SER", 2)
        topology, pos_A = p53
        mol, names, caps, _, partner = _build_capped_molecule(res, topology, pos_A)
        res.name = "QQQQQ"
        try:
            mol, fallback_reason = _perceive_residue_bond_orders(mol, names, res.name)
        finally:
            res.name = "SER"
        assert fallback_reason  # the premise: the dictionary did not apply

        amide, _, scheme = _cap_backbone_as_amide(mol, names, caps, partner)

        assert scheme.kind == CAP_KIND_AMIDE
        assert _smiles(amide) == _canonical("CC(=O)NC(CO)C(=O)NC")

    def test_caps_that_do_not_sanitise_fall_back_to_the_methyl_caps(self, p53):
        """A backbone C that already has an OXT cannot also take the NME: valence five."""
        res = _residue(p53, "SER", 2)
        mol, names, caps, partner = _perceived(p53, res)
        edited = Chem.RWMol(mol)
        backbone_c = next(i for i, name in names.items() if name == "C")
        edited.AddBond(backbone_c, edited.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
        edited.GetConformer().SetAtomPosition(edited.GetNumAtoms() - 1, (0.0, 0.0, 0.0))

        same, same_caps, scheme = _cap_backbone_as_amide(edited.GetMol(), names, caps, partner)

        assert scheme.kind == CAP_KIND_METHYL and "do not sanitise" in scheme.reason
        assert same_caps == caps


class TestIntegerTotal:
    @pytest.mark.parametrize("total", [0, -1, 1, 2])
    def test_sum_is_exact_and_the_shift_is_even(self, total):
        rng = np.random.default_rng(7)
        charges = {i: float(q) for i, q in enumerate(rng.normal(0.0, 0.3, 17))}
        shift = (total - sum(charges.values())) / len(charges)

        fixed = _charges_with_integer_total(charges, total)

        assert sum(fixed.values()) == pytest.approx(total, abs=1e-12)
        assert all(fixed[i] == pytest.approx(charges[i] + shift, abs=1.1e-6) for i in charges)

    def test_values_are_multiples_of_the_xml_precision(self):
        """The XML stores six decimals, so the stored charges are the ones that sum to the total."""
        rng = np.random.default_rng(3)
        charges = {i: float(q) for i, q in enumerate(rng.normal(0.0, 0.3, 40))}

        fixed = _charges_with_integer_total(charges, -1)

        stored = [float(f"{q:.6f}") for q in fixed.values()]
        assert sum(stored) == pytest.approx(-1.0, abs=1e-12)
        assert stored == pytest.approx(list(fixed.values()), abs=1e-12)

    def test_an_exact_input_is_unchanged(self):
        charges = {0: 0.25, 1: -0.5, 2: 0.25}
        assert _charges_with_integer_total(charges, 0) == charges


class TestSchemeRecord:
    def test_label_carries_the_version(self):
        assert CHARGE_SCHEME_VERSION >= 2  # version 1 is the methyl-cap scheme of earlier code
        assert CapScheme(CAP_KIND_AMIDE).label == f"ace-nme-caps-v{CHARGE_SCHEME_VERSION}"
        assert CapScheme(CAP_KIND_METHYL, "x").label == f"methyl-caps-v{CHARGE_SCHEME_VERSION}"
        assert CapScheme(CAP_KIND_NONE).label == f"uncapped-v{CHARGE_SCHEME_VERSION}"


class _Residue:
    def __init__(self, name, index):
        self.name, self.index = name, index


class _Topology:
    def __init__(self, residues):
        self._residues = residues

    def residues(self):
        return iter(self._residues)


_STUB_TEMPLATE = (
    '<ForceField><Residues><Residue name="X">'
    '<Atom name="C1" type="c3" charge="0.06"/><Atom name="C2" type="c3" charge="-0.06"/>'
    "</Residue></Residues></ForceField>"
)


@pytest.fixture
def stubbed_step(monkeypatch):
    """``parameterize_ncaa_residues`` on fake residues; ``run(*schemes)`` gives each its scheme."""
    pytest.importorskip("openmmforcefields")

    def run(names_and_schemes, **kwargs):
        def generate(res, topology, pos_A, gaff_version, backbone_amber, random_seed=None):
            scheme = names_and_schemes[res.name]
            return (_STUB_TEMPLATE, [], [], "") + ((scheme,) if scheme is not None else ())

        monkeypatch.setattr(gaff_ncaa, "_is_ncaa", lambda res: True)
        monkeypatch.setattr(gaff_ncaa, "_pos_to_angstrom", lambda positions: np.zeros((1, 3)))
        monkeypatch.setattr(gaff_ncaa, "_amber_backbone_types", lambda ff: {})
        monkeypatch.setattr(gaff_ncaa, "_generate_residue_template", generate)
        monkeypatch.setattr(gaff_ncaa, "_load_ffxml", lambda ff, xml: None)
        monkeypatch.setattr(gaff_ncaa, "_hydrogen_parents", lambda topology: {})
        monkeypatch.setattr(gaff_ncaa, "_has_template_hydrogens", lambda *args: True)
        topology = _Topology([_Residue(n, i) for i, n in enumerate(names_and_schemes)])
        return gaff_ncaa.parameterize_ncaa_residues(topology, [None], ff=None, **kwargs)[2]

    return run


class TestSchemeIsReported:
    def test_each_residue_records_its_scheme_and_logs_one_line(self, stubbed_step, caplog):
        schemes = {
            "BMT": CapScheme(CAP_KIND_AMIDE),
            "GSH": CapScheme(CAP_KIND_METHYL, "it has no N-CA-C=O backbone"),
        }
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            templates = stubbed_step(schemes)

        assert templates.charge_scheme_by_residue == {
            "BMT": f"ace-nme-caps-v{CHARGE_SCHEME_VERSION}",
            "GSH": f"methyl-caps-v{CHARGE_SCHEME_VERSION}",
        }
        lines = [r.getMessage() for r in caplog.records if "charges from" in r.getMessage()]
        assert lines == [
            "  Auto-GAFF2: 'BMT' charges from the ACE/NME-capped residue (AM1-BCC)",
            "  Auto-GAFF2: 'GSH' charges from the methyl-capped residue (AM1-BCC), "
            "because it has no N-CA-C=O backbone",
        ]

    def test_verbose_false_records_without_logging(self, stubbed_step, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            templates = stubbed_step({"BMT": CapScheme(CAP_KIND_AMIDE)}, verbose=False)

        assert templates.charge_scheme_by_residue == {"BMT": "ace-nme-caps-v2"}
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_a_generator_without_scheme_leaves_the_record_empty(self, stubbed_step):
        assert stubbed_step({"BMT": None}).charge_scheme_by_residue == {}


# ---------------------------------------------------------------------------------------------
# AM1-BCC charges (antechamber and sqm): ALA, charged ASP and LYS, the residues of 1CWA
# ---------------------------------------------------------------------------------------------


def _ff14sb_charges(residue_name: str) -> dict:
    """Atom charges of an internal residue in amber14/protein.ff14SB.xml."""
    import openmm.app

    path = os.path.join(
        os.path.dirname(openmm.app.__file__), "data", "amber14", "protein.ff14SB.xml"
    )
    for residue in ET.parse(path).getroot().find("Residues").findall("Residue"):
        if residue.get("name") == residue_name:
            return {a.get("name"): float(a.get("charge")) for a in residue.findall("Atom")}
    raise LookupError(residue_name)


def _template_charges(xml: str) -> dict:
    residue = ET.fromstring(xml).find(".//Residue")
    return {a.get("name"): float(a.get("charge")) for a in residue.findall("Atom")}


def _methyl_caps(mol, names, caps, partner):
    """``_cap_backbone_as_amide`` as it was before #61: the external bonds keep their methyl."""
    return mol, caps, CapScheme(CAP_KIND_METHYL, "forced")


#: Atoms of the peptide group in a template, with the name each has in an ff14SB residue.
_PEPTIDE_GROUP_IN_FF14SB = {"N": "N", "C": "C", "O": "O"}
_BACKBONE_IN_FF14SB = {**_PEPTIDE_GROUP_IN_FF14SB, "HN": "H", "CA": "CA"}


def _deviation_from_ff14sb(charges: dict, residue_name: str, atoms: dict) -> float:
    """Root-mean-square difference of the charges of ``atoms`` from the ff14SB residue's."""
    reference = _ff14sb_charges(residue_name)
    return float(
        np.sqrt(
            np.mean([(charges[ours] - reference[theirs]) ** 2 for ours, theirs in atoms.items()])
        )
    )


def _ionised_residue_charges(structure, residue_name, ionised, cap_backbone=_cap_backbone_as_amide):
    """Charges of a residue in an ionised state, through the steps of the template builder.

    ``ionised`` maps the name of an atom of the residue to its formal charge, which is set on
    the perceived (neutral) molecule: ``{"OD2": -1}`` makes ASP a carboxylate. Returns
    ``(charges of the heavy atoms by name, sum over all kept atoms, formal charge)``. The
    template builder always builds the neutral form, so this is the one place where a charged
    residue is tested. ``cap_backbone`` is the capping step: ``_methyl_caps`` gives the
    residue the caps of the earlier code.
    """
    from openff.toolkit import Molecule

    res = _residue(structure, residue_name, 2)
    topology, pos_A = structure
    mol, names, caps, _, partner = _build_capped_molecule(res, topology, pos_A)
    mol, _ = _perceive_residue_bond_orders(mol, names, res.name)
    by_name = {name: index for index, name in names.items()}
    for atom_name, charge in ionised.items():
        atom = mol.GetAtomWithIdx(by_name[atom_name])
        atom.SetNumExplicitHs(0)  # the neutraliser left the hydrogens of the group explicit
        atom.SetNoImplicit(False)
        atom.SetFormalCharge(charge)
    Chem.SanitizeMol(mol)
    mol, caps, _ = cap_backbone(mol, names, caps, partner)
    mh = Chem.AddHs(mol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(mh)
    off = Molecule.from_rdkit(mh, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    charges = gaff_ncaa._am1bcc_charges(off, 1)
    kept = [
        i
        for i, atom in enumerate(mh.GetAtoms())
        if i not in caps
        and not (atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetIdx() in caps)
    ]
    formal_charge = Chem.GetFormalCharge(mh)
    fixed = _charges_with_integer_total({i: float(charges[i]) for i in kept}, formal_charge)
    return {names[i]: fixed[i] for i in kept if i in names}, sum(fixed.values()), formal_charge


def _alanine_template(structure):
    """Template tuple of ALA in chain A of 1CWA, built by the template builder."""
    pytest.importorskip("openmmforcefields")
    from openmm.app import ForceField

    topology, pos_A = structure
    res = _residue(structure, "ALA", 2, chain_id="A")
    backbone_types = gaff_ncaa._amber_backbone_types(ForceField("amber14-all.xml"))
    return gaff_ncaa._generate_residue_template(res, topology, pos_A, "gaff-2.2.20", backbone_types)


@pytest.fixture(scope="module")
def alanine_template(cyclosporin):
    return _alanine_template(cyclosporin)


@pytest.fixture(scope="module")
def alanine_methyl_charges(cyclosporin):
    """Charges of the same ALA built with the methyl caps of the earlier code."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gaff_ncaa, "_cap_backbone_as_amide", _methyl_caps)
        return _template_charges(_alanine_template(cyclosporin)[0])


@pytest.mark.slow
class TestTemplateChargesOfAlanine:
    """ALA of cyclophilin (1CWA chain A) through the template builder, as for any residue."""

    def test_the_backbone_is_an_amide(self, alanine_template, alanine_methyl_charges):
        charges = _template_charges(alanine_template[0])

        assert alanine_template[4] == CapScheme(CAP_KIND_AMIDE)
        # The polarity of a peptide group: N and O negative, the H on N and the C=O carbon positive.
        assert charges["N"] < 0 < charges["HN"] and charges["O"] < 0 < charges["C"]
        # The caps of the earlier code left the backbone an amine; the ACE/NME caps must bring
        # the backbone charges closer to those of ALA in a chain, which is the ff14SB residue.
        amide = _deviation_from_ff14sb(charges, "ALA", _BACKBONE_IN_FF14SB)
        methyl = _deviation_from_ff14sb(alanine_methyl_charges, "ALA", _BACKBONE_IN_FF14SB)
        assert amide < methyl

    def test_the_total_is_the_integer(self, alanine_template):
        charges = _template_charges(alanine_template[0])
        assert sum(charges.values()) == pytest.approx(0.0, abs=1e-9)

    def test_methyl_caps_give_the_amine_nitrogen(self, alanine_template, alanine_methyl_charges):
        """The same residue with the earlier caps: the N is the too negative one (#61)."""
        new = _template_charges(alanine_template[0])
        reference = _ff14sb_charges("ALA")["N"]

        assert alanine_methyl_charges["N"] < new["N"]
        assert abs(new["N"] - reference) < abs(alanine_methyl_charges["N"] - reference)


@pytest.mark.slow
class TestChargedResidues:
    """ASP (-1) and LYS (+1) as ACE/NME model compounds: the integer total and an amide N."""

    @pytest.mark.parametrize(
        "residue_name, ionised, total",
        [("ASP", {"OD2": -1}, -1), ("LYS", {"NZ": +1}, +1)],
    )
    def test_total_is_exact_and_the_nitrogen_is_an_amide_nitrogen(
        self, p53, residue_name, ionised, total
    ):
        charges, kept_total, formal_charge = _ionised_residue_charges(p53, residue_name, ionised)
        methyl_capped, _, _ = _ionised_residue_charges(
            p53, residue_name, ionised, cap_backbone=_methyl_caps
        )
        reference = _ff14sb_charges(residue_name)

        assert formal_charge == total
        assert kept_total == pytest.approx(total, abs=1e-9)
        assert charges["N"] < 0 and charges["O"] < 0 < charges["C"]
        # The amine N of the methyl caps is further from the amide N of ff14SB, and so is the
        # peptide group as a whole.
        assert abs(charges["N"] - reference["N"]) < abs(methyl_capped["N"] - reference["N"])
        assert _deviation_from_ff14sb(
            charges, residue_name, _PEPTIDE_GROUP_IN_FF14SB
        ) < _deviation_from_ff14sb(methyl_capped, residue_name, _PEPTIDE_GROUP_IN_FF14SB)


@pytest.fixture(scope="module")
def cyclosporin_built():
    """``(topology, positions, force field, templates)`` from ``parameterize_ncaa_residues``."""
    pytest.importorskip("openmmforcefields")
    from openmm.app import ForceField

    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(CYCLOSPORIN)
    for res in topology.residues():
        # The prep renames these two to ALA and NMG (curated templates); do the same here.
        res.name = {"DAL": "ALA", "SAR": "NMG"}.get(res.name, res.name)
    ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    topology, positions, xmls = gaff_ncaa.parameterize_ncaa_residues(topology, positions, ff)
    return topology, positions, ff, xmls


@pytest.fixture(scope="module")
def cyclosporin_templates(cyclosporin_built):
    """``(topology, templates)`` of BMT and ABA from ``parameterize_ncaa_residues`` on 1CWA."""
    topology, _, _, xmls = cyclosporin_built
    return topology, xmls


@pytest.fixture(scope="module")
def cyclosporin_methyl_charges(cyclosporin_built):
    """Template charges of BMT and ABA built again with the methyl caps of the earlier code."""
    topology, positions, ff, _ = cyclosporin_built
    pos_A = gaff_ncaa._pos_to_angstrom(positions)
    backbone_types = gaff_ncaa._amber_backbone_types(ff)
    charges = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gaff_ncaa, "_cap_backbone_as_amide", _methyl_caps)
        for name in ("BMT", "ABA"):
            residue = next(r for r in topology.residues() if r.name == name)
            xml = gaff_ncaa._generate_residue_template(
                residue, topology, pos_A, "gaff-2.2.20", backbone_types
            )[0]
            charges[name] = _template_charges(xml)
    return charges


def _residue_charges(xmls, name):
    return {
        a.get("name"): float(a.get("charge")) for a in _residue_element(xmls, name).findall("Atom")
    }


def _residue_element(xmls, name):
    for xml in xmls:
        element = ET.fromstring(xml).find(".//Residue")
        if element.get("name") == name:
            return element
    raise AssertionError(f"no template for {name}")


@pytest.mark.slow
class TestCyclosporinResidues:
    """BMT and ABA of cyclosporin A (1CWA) through ``parameterize_ncaa_residues``.

    With the methyl caps of the earlier code the template N of N-methylated MeBmt and of ABA
    was the N of an amine, more negative than the amide N of ff14SB.
    """

    def test_both_residues_are_built_from_the_capped_amide(self, cyclosporin_templates):
        _, xmls = cyclosporin_templates
        label = f"ace-nme-caps-v{CHARGE_SCHEME_VERSION}"
        assert xmls.charge_scheme_by_residue == {"BMT": label, "ABA": label}

    def test_the_backbone_nitrogen_is_an_amide_nitrogen(
        self, cyclosporin_templates, cyclosporin_methyl_charges
    ):
        _, xmls = cyclosporin_templates
        amide_n = _ff14sb_charges("ALA")["N"]
        for name in ("BMT", "ABA"):
            n_charge = _residue_charges(xmls, name)["N"]
            methyl_n = cyclosporin_methyl_charges[name]["N"]
            assert n_charge < 0, name
            # The N of the methyl caps is an amine and further from the amide N of ff14SB.
            assert abs(n_charge - amide_n) < abs(methyl_n - amide_n), name

    def test_the_carbonyl_is_a_carbonyl(self, cyclosporin_templates):
        _, xmls = cyclosporin_templates
        for name in ("BMT", "ABA"):
            charges = _residue_charges(xmls, name)
            carbons = {atom: charge for atom, charge in charges.items() if atom.startswith("C")}
            others = [charge for atom, charge in carbons.items() if atom != "C"]
            # The carbonyl carbon is bound to O and N, so it is the most positive carbon; the
            # carbonyl O is more negative than every carbon.
            assert charges["C"] > max(others) and charges["C"] > 0, name
            assert charges["O"] < min(carbons.values()) and charges["O"] < 0, name

    def test_the_total_is_the_integer(self, cyclosporin_templates):
        _, xmls = cyclosporin_templates
        for name in ("BMT", "ABA"):
            atoms = _residue_element(xmls, name).findall("Atom")
            assert sum(float(a.get("charge")) for a in atoms) == pytest.approx(0.0, abs=1e-9)
        assert xmls.net_charge_by_residue == {"BMT": 0.0, "ABA": 0.0}

    def test_atoms_and_external_bonds_are_the_residue_s_own(self, cyclosporin_templates):
        topology, xmls = cyclosporin_templates
        for name in ("BMT", "ABA"):
            element = _residue_element(xmls, name)
            atom_names = {a.get("name") for a in element.findall("Atom")}
            residue = next(r for r in topology.residues() if r.name == name)
            assert {a.name for a in residue.atoms()} == atom_names  # no cap atom, all hydrogens
            assert {e.get("atomName") for e in element.findall("ExternalBond")} == {"N", "C"}
