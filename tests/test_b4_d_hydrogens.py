"""The linear route of prep adds hydrogens to D-amino-acid residues (#164).

``Modeller.addHydrogens`` places hydrogens by residue name, and its definitions have no
D-amino acid: a linear peptide with DGL, DPN, ... came out of prep with no hydrogen on
those residues while their L neighbours had theirs. The cyclic route renames each D residue
to its L template before the hydrogens are added and gives the names back afterwards; the
linear route now does the same. The hydrogens are placed from the heavy-atom geometry, so a
D residue gets its hydrogens on the D side of the alpha carbon.
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

from conftest import DATA_DIR
from openmm import Vec3, unit
from openmm.app import Modeller

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


def _peptide_only(mirror: bool):
    """The 13 residues of chain B of 1YCR; with ``mirror`` every x coordinate is negated.

    The mirror image of an L peptide is a peptide of D residues, so its geometry is the one
    the D names stand for.
    """
    topology, positions = load_structure(P53)
    modeller = Modeller(topology, positions)
    modeller.delete([r for r in modeller.topology.residues() if r.chain.id != "B"])
    coordinates = modeller.positions.value_in_unit(unit.nanometer)
    if mirror:
        coordinates = [Vec3(-v.x, v.y, v.z) for v in coordinates]
    return modeller.topology, unit.Quantity(list(coordinates), unit.nanometer)


def _prep(topology, positions):
    pytest.importorskip("pdbfixer")
    from binding_metrics.core.system import prep_structure

    return prep_structure(topology, positions)


def _hydrogens_per_residue(topology, chain_id="B"):
    return [
        sum(1 for a in r.atoms() if a.element.symbol == "H")
        for c in topology.chains()
        if c.id == chain_id
        for r in c.residues()
    ]


def _names(topology, chain_id="B"):
    return [r.name for c in topology.chains() if c.id == chain_id for r in c.residues()]


def _heavy_atoms_and_hydrogens(topology, chain_id="B"):
    elements = [a.element.symbol for c in topology.chains() if c.id == chain_id for a in c.atoms()]
    n_hydrogens = elements.count("H")
    return len(elements) - n_hydrogens, n_hydrogens


def _signed_volume(positions_nm, residue, fourth: str) -> float:
    """Signed volume of N, C and ``fourth`` around CA: which face of N-CA-C the atom is on."""
    atoms = {a.name: np.array(positions_nm[a.index]) for a in residue.atoms()}
    ca = atoms["CA"]
    return float(np.dot(atoms["N"] - ca, np.cross(atoms["C"] - ca, atoms[fourth] - ca)))


@pytest.mark.slow
class TestHydrogensOfDResidues:
    @pytest.fixture(scope="class")
    def l_peptide(self):
        topology, positions = load_structure(P53)
        return _prep(topology, positions)

    @pytest.mark.parametrize("every", [1, 2, 3])
    def test_a_d_residue_gets_the_hydrogens_of_its_l_form(self, l_peptide, every):
        topology, positions = load_structure(P53)
        n_flipped = _flip_to_d(topology, "B", every=every)
        flipped_names = _names(topology)
        prepped, _ = _prep(topology, positions)

        assert n_flipped == len(range(0, 13, every))
        assert _hydrogens_per_residue(prepped) == _hydrogens_per_residue(l_peptide[0])
        assert _names(prepped) == flipped_names  # DGL, DPN, ... are back
        assert prepped.getNumAtoms() == l_peptide[0].getNumAtoms()

    def test_the_all_d_peptide_has_the_atoms_of_the_l_peptide(self, l_peptide):
        topology, positions = load_structure(P53)
        _flip_to_d(topology, "B")
        prepped, _ = _prep(topology, positions)

        # Same heavy atoms and the same number of hydrogens as the L peptide of the same sequence.
        assert _heavy_atoms_and_hydrogens(prepped) == _heavy_atoms_and_hydrogens(l_peptide[0])
        assert _hydrogens_per_residue(prepped) == _hydrogens_per_residue(l_peptide[0])
        # The failure of #164 left every D residue without a hydrogen, and an L peptide that
        # lost its hydrogens too would still satisfy the two equalities above.
        assert all(count > 0 for count in _hydrogens_per_residue(prepped))

    def test_the_hydrogens_are_bonded_to_the_atoms_of_the_l_template(self):
        topology, positions = load_structure(P53)
        _flip_to_d(topology, "B")
        prepped, _ = _prep(topology, positions)

        by_residue = {}
        for bond in prepped.bonds():
            for atom, other in ((bond.atom1, bond.atom2), (bond.atom2, bond.atom1)):
                if atom.element.symbol == "H" and atom.residue.chain.id == "B":
                    by_residue.setdefault(atom.residue.index, []).append((atom.name, other.name))
        ha = [bonds for bonds in by_residue.values() if ("HA", "CA") in bonds]
        assert len(ha) == 13  # every residue has its HA on CA

    @pytest.mark.parametrize("mirror", [False, True])
    def test_ha_and_cb_sit_on_opposite_faces_of_the_backbone_plane(self, mirror):
        """The same rule for L (geometry as it is) and D (mirror image with D names)."""
        topology, positions = _peptide_only(mirror=mirror)
        if mirror:
            _flip_to_d(topology, "B")
        prepped, prepped_positions = _prep(topology, positions)
        coordinates = prepped_positions.value_in_unit(unit.nanometer)

        checked = 0
        for residue in prepped.residues():
            names = {a.name for a in residue.atoms()}
            if not {"CB", "HA"} <= names:
                continue
            assert (
                _signed_volume(coordinates, residue, "CB")
                * _signed_volume(coordinates, residue, "HA")
                < 0
            ), residue.name
            checked += 1
        assert checked == 13

    def test_the_d_peptide_is_the_mirror_image_of_the_l_peptide_at_the_alpha_carbons(self):
        l_topology, l_positions = _peptide_only(mirror=False)
        d_topology, d_positions = _peptide_only(mirror=True)
        _flip_to_d(d_topology, "B")
        l_prepped, l_coordinates = _prep(l_topology, l_positions)
        d_prepped, d_coordinates = _prep(d_topology, d_positions)
        l_nm = l_coordinates.value_in_unit(unit.nanometer)
        d_nm = d_coordinates.value_in_unit(unit.nanometer)

        for l_residue, d_residue in zip(l_prepped.residues(), d_prepped.residues()):
            if "CB" not in {a.name for a in l_residue.atoms()}:
                continue
            for fourth in ("CB", "HA"):
                assert (
                    _signed_volume(l_nm, l_residue, fourth)
                    * _signed_volume(d_nm, d_residue, fourth)
                    < 0
                ), (l_residue.name, fourth)
            l_ha = next(a for a in l_residue.atoms() if a.name == "HA")
            d_ha = next(a for a in d_residue.atoms() if a.name == "HA")
            mirrored = np.array(l_nm[l_ha.index]) * np.array([-1.0, 1.0, 1.0])
            assert np.linalg.norm(mirrored - np.array(d_nm[d_ha.index])) < 0.03, l_residue.name
