"""``prep_structure`` keeps the lactam closure of a model that lists no bond (#199).

A model output has no ``_struct_conn``, so nothing in the loaded topology bonds the side-chain
amine to the carboxyl carbon of the other residue. Prep finds the bond from the distance, as it
finds a head-to-tail ring (``_head_to_tail_bonds``), and takes the cyclic route: the acid keeps one
carboxyl oxygen, the lysine one amide hydrogen, and the closure bond is in the result.

``_lactam_bonds`` is tested on small topologies built by hand and on the bundled structures. The
preparation of the bundled somatostatin analogue (1XY4: a Glu4-Lys10 lactam and a Cys2-Cys12
disulfide) runs once for the listed file and once for the same atoms written without any bond
record.
"""

import pytest
from conftest import DATA_DIR
from openmm import Vec3
from openmm.app import Element, Topology
from openmm.unit import nanometer

pytest.importorskip("pdbfixer")

from binding_metrics.core.cyclic import _AMIDE_BOND_THRESH
from binding_metrics.core.system import _head_to_tail_bonds, _lactam_bonds, prep_structure
from binding_metrics.io.structures import load_structure
from binding_metrics.metrics._common import load_structure as load_atoms
from binding_metrics.testing.synthetic import write_cif

SOMATOSTATIN = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"
SFTI = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
CWA = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"

LYS = ("LYS", ["N", "CA", "C", "CE", "NZ"])
GLU = ("GLU", ["N", "CA", "C", "CG", "CD"])
ASP = ("ASP", ["N", "CA", "C", "CB", "CG"])
GLY = ("GLY", ["N", "CA", "C"])


def bare(source, target):
    """The structure without waters, written as a model: no bond record of any kind."""
    atoms = load_atoms(source)
    write_cif(atoms[atoms.res_name != "HOH"], target)
    assert "_struct_conn" not in target.read_text(encoding="utf-8")
    return target


def make_peptide(residues, pair, gap_nm):
    """One chain of ``residues`` (name, atom names), no bond, ``gap_nm`` between two atoms.

    The atoms sit on a line 0.5 nm apart, except the second atom of ``pair`` ((residue index,
    atom name) twice), which is moved to ``gap_nm`` from the first.
    """
    topology = Topology()
    chain = topology.addChain("A")
    positions, atoms = [], {}
    for index, (name, atom_names) in enumerate(residues):
        residue = topology.addResidue(name, chain, str(index + 1))
        for atom_name in atom_names:
            element = Element.getBySymbol("N" if atom_name.startswith("N") else "C")
            atoms[(index, atom_name)] = topology.addAtom(atom_name, element, residue)
            positions.append(Vec3(0.5 * len(positions), 0.0, 0.0))
    anchor = positions[atoms[pair[0]].index]
    positions[atoms[pair[1]].index] = Vec3(anchor.x, anchor.y + gap_nm, anchor.z)
    return topology, positions * nanometer


def names(residue):
    return {a.name for a in residue.atoms()}


class TestTheDetectionByDistance:
    """``_lactam_bonds`` on small topologies: the closures that the cyclic builders know."""

    def test_a_side_chain_lactam_is_found_as_acid_carbon_then_amine_nitrogen(self):
        topology, positions = make_peptide(
            [GLY, GLU, GLY, LYS, GLY], ((3, "NZ"), (1, "CD")), gap_nm=0.15
        )

        assert _lactam_bonds(topology, positions) == [("A", 1, "CD", "A", 3, "NZ")]

    def test_the_aspartate_closure_is_found_too(self):
        topology, positions = make_peptide(
            [GLY, LYS, GLY, ASP, GLY], ((1, "NZ"), (3, "CG")), gap_nm=0.15
        )

        assert _lactam_bonds(topology, positions) == [("A", 3, "CG", "A", 1, "NZ")]

    def test_a_lysine_to_the_c_terminus_is_found(self):
        topology, positions = make_peptide([GLY, LYS, GLY, GLY], ((1, "NZ"), (3, "C")), 0.15)

        assert _lactam_bonds(topology, positions) == [("A", 3, "C", "A", 1, "NZ")]

    @pytest.mark.parametrize(
        "acid, carbon", [(ASP, "CG"), (GLU, "CD")], ids=["aspartate", "glutamate"]
    )
    def test_an_acid_side_chain_to_the_n_terminus_is_found(self, acid, carbon):
        topology, positions = make_peptide([GLY, GLY, acid, GLY], ((0, "N"), (2, carbon)), 0.15)

        assert _lactam_bonds(topology, positions) == [("A", 2, carbon, "A", 0, "N")]

    def test_the_cut_is_the_amide_distance_of_the_head_to_tail_ring(self):
        residues, pair = [GLY, GLU, GLY, LYS, GLY], ((3, "NZ"), (1, "CD"))

        inside = make_peptide(residues, pair, _AMIDE_BOND_THRESH - 0.005)
        outside = make_peptide(residues, pair, _AMIDE_BOND_THRESH + 0.005)

        assert len(_lactam_bonds(*inside)) == 1
        assert _lactam_bonds(*outside) == []

    def test_a_pair_that_the_topology_bonds_is_left_to_the_listed_bond(self):
        topology, positions = make_peptide(
            [GLY, GLU, GLY, LYS, GLY], ((3, "NZ"), (1, "CD")), gap_nm=0.15
        )
        atoms = {(a.residue.index, a.name): a for a in topology.atoms()}
        topology.addBond(atoms[(1, "CD")], atoms[(3, "NZ")])

        assert _lactam_bonds(topology, positions) == []

    def test_the_head_to_tail_pair_is_not_a_lactam(self):
        topology, positions = make_peptide([GLY, GLY, GLY], ((2, "C"), (0, "N")), gap_nm=0.15)

        assert _lactam_bonds(topology, positions) == []
        assert _head_to_tail_bonds(topology, positions) == [("A", 2, "C", "A", 0, "N")]

    @pytest.mark.parametrize(
        "residues, pair",
        [
            ([ASP, GLY, GLY], ((0, "N"), (0, "CG"))),
            ([GLY, GLY, LYS], ((2, "NZ"), (2, "C"))),
        ],
        ids=["acid_and_n_terminus_in_one_residue", "lysine_and_c_terminus_in_one_residue"],
    )
    def test_two_atoms_of_one_residue_are_not_a_closure(self, residues, pair):
        topology, positions = make_peptide(residues, pair, gap_nm=0.15)

        assert _lactam_bonds(topology, positions) == []

    def test_a_residue_with_the_wrong_name_is_not_a_partner(self):
        asparagine = ("ASN", ["N", "CA", "C", "CB", "CG"])
        topology, positions = make_peptide(
            [GLY, asparagine, GLY, LYS, GLY], ((3, "NZ"), (1, "CG")), gap_nm=0.15
        )

        assert _lactam_bonds(topology, positions) == []

    def test_a_chain_of_one_residue_has_none(self):
        topology, positions = make_peptide([LYS], ((0, "NZ"), (0, "C")), gap_nm=0.15)

        assert _lactam_bonds(topology, positions) == []


class TestTheBundledStructures:
    @pytest.mark.parametrize(
        "path", [P53, SFTI, CWA, SOMATOSTATIN], ids=["1YCR", "3P8F", "1CWA", "1XY4_listed"]
    )
    def test_a_structure_that_lists_its_bonds_gets_none(self, path):
        topology, positions = load_structure(path)

        assert _lactam_bonds(topology, positions) == []

    @pytest.mark.parametrize("path", [P53, SFTI, CWA], ids=["1YCR", "3P8F", "1CWA"])
    def test_a_bare_structure_without_a_lactam_gets_none(self, path, tmp_path):
        topology, positions = load_structure(bare(path, tmp_path / "bare.cif"))

        assert _lactam_bonds(topology, positions) == []

    def test_the_bare_somatostatin_has_its_lactam_and_no_head_to_tail_ring(self, tmp_path):
        topology, positions = load_structure(bare(SOMATOSTATIN, tmp_path / "bare.cif"))

        assert _lactam_bonds(topology, positions) == [("A", 3, "CD", "A", 9, "NZ")]
        assert _head_to_tail_bonds(topology, positions) == []


@pytest.fixture(scope="module")
def somatostatin_prepared(tmp_path_factory):
    """Prepared topologies of 1XY4: the listed file, and the same atoms with no bond record."""
    tmp_path = tmp_path_factory.mktemp("lactam")
    listed = prep_structure(*load_structure(SOMATOSTATIN))[0]
    raw = prep_structure(*load_structure(bare(SOMATOSTATIN, tmp_path / "bare.cif")))[0]
    return listed, raw


def residue_numbered(topology, number):
    return next(r for r in topology.residues() if r.id == number)


def bond_pairs(topology):
    """Bonds as pairs of (residue index, atom name), comparable between two equal topologies."""
    return {
        frozenset(((b.atom1.residue.index, b.atom1.name), (b.atom2.residue.index, b.atom2.name)))
        for b in topology.bonds()
    }


@pytest.mark.slow
@pytest.mark.integration
class TestPrepKeepsTheLactam:
    def test_the_raw_model_keeps_the_amide_bond_of_the_lactam(self, somatostatin_prepared):
        _, raw = somatostatin_prepared
        carbon = next(a for a in residue_numbered(raw, "4").atoms() if a.name == "CD")
        nitrogen = next(a for a in residue_numbered(raw, "10").atoms() if a.name == "NZ")

        assert any({b.atom1, b.atom2} == {carbon, nitrogen} for b in raw.bonds())

    def test_the_acid_keeps_one_carboxyl_oxygen_and_the_amine_one_hydrogen(
        self, somatostatin_prepared
    ):
        _, raw = somatostatin_prepared
        acid, amine = residue_numbered(raw, "4"), residue_numbered(raw, "10")

        assert "OE1" in names(acid) and "OE2" not in names(acid)
        assert {"NZ", "HZ1"} <= names(amine) and not {"HZ2", "HZ3"} & names(amine)

    def test_the_raw_model_is_prepared_as_the_listed_file(self, somatostatin_prepared):
        listed, raw = somatostatin_prepared

        assert [(r.name, sorted(names(r))) for r in raw.residues()] == [
            (r.name, sorted(names(r))) for r in listed.residues()
        ]
        assert bond_pairs(raw) == bond_pairs(listed)

    def test_the_ends_of_the_open_chain_stay_charged(self, somatostatin_prepared):
        _, raw = somatostatin_prepared
        first, *_, last = next(raw.chains()).residues()

        assert {"H", "H2", "H3"} <= names(first)
        assert "OXT" in names(last)

    def test_the_disulfide_is_kept(self, somatostatin_prepared):
        _, raw = somatostatin_prepared

        assert any(
            b.atom1.name == b.atom2.name == "SG" and b.atom1.residue is not b.atom2.residue
            for b in raw.bonds()
        )
