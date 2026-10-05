"""A GAFF2 residue that already holds its template's hydrogens keeps them.

``parameterize_ncaa_residues`` used to drop the hydrogens of every GAFF2 residue and inject the
ones RDKit places for the heavy atoms, in prep, in relaxation and in the energy step. The energy
step then evaluated a relaxed structure with unrelaxed BMT and ABA hydrogens: in 1CWA the
complex energy came out 124 kJ/mol above the relaxation minimum.

These tests use a two-carbon residue and a stubbed template step, so no sqm run is needed.
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

from openmm import Vec3, unit  # noqa: E402
from openmm.app import Topology, element  # noqa: E402

from binding_metrics.core import gaff_ncaa  # noqa: E402

#: The template's hydrogens: (name, parent atom name, position in nm).
TEMPLATE_HYDROGENS = [
    ("H11", "C1", (0.10, 0.00, 0.00)),
    ("H12", "C1", (0.00, 0.10, 0.00)),
    ("H21", "C2", (0.30, 0.10, 0.00)),
]

#: Where the residue's hydrogens are in a relaxed structure (nm); none is a template position.
RELAXED_POSITIONS = {
    "H11": (0.11, 0.02, 0.01),
    "H12": (0.01, 0.12, 0.02),
    "H21": (0.31, 0.12, 0.01),
}


def _residue(hydrogens: list, bonded: bool = True):
    """Topology of residue XXX (C1, C2 and the ``(name, parent)`` hydrogens) and its positions."""
    topology = Topology()
    residue = topology.addResidue("XXX", topology.addChain("A"))
    carbons = {
        "C1": topology.addAtom("C1", element.carbon, residue),
        "C2": topology.addAtom("C2", element.carbon, residue),
    }
    topology.addBond(carbons["C1"], carbons["C2"])
    positions = [Vec3(0.0, 0.0, 0.0), Vec3(0.15, 0.0, 0.0)]
    for name, parent in hydrogens:
        atom = topology.addAtom(name, element.hydrogen, residue)
        if bonded:
            topology.addBond(atom, carbons[parent])
        positions.append(Vec3(*RELAXED_POSITIONS.get(name, (0.5, 0.5, 0.5))))
    return topology, positions


def _matches(hydrogens: list, bonded: bool = True) -> bool:
    topology, _ = _residue(hydrogens, bonded)
    residue = next(topology.residues())
    parents = gaff_ncaa._hydrogen_parents(topology)
    return gaff_ncaa._has_template_hydrogens(residue, parents, TEMPLATE_HYDROGENS)


TEMPLATE_PAIRS = [(name, parent) for name, parent, _ in TEMPLATE_HYDROGENS]


class TestHasTemplateHydrogens:
    def test_the_same_names_on_the_same_parents_match(self):
        assert _matches(TEMPLATE_PAIRS)

    def test_the_order_of_the_atoms_does_not_matter(self):
        assert _matches(list(reversed(TEMPLATE_PAIRS)))

    def test_a_residue_without_hydrogens_does_not_match(self):
        assert not _matches([])

    def test_a_missing_hydrogen_does_not_match(self):
        assert not _matches(TEMPLATE_PAIRS[:-1])

    def test_an_extra_hydrogen_does_not_match(self):
        assert not _matches([*TEMPLATE_PAIRS, ("H22", "C2")])

    def test_another_name_does_not_match(self):
        assert not _matches([("HA", "C1"), *TEMPLATE_PAIRS[1:]])

    def test_another_parent_does_not_match(self):
        assert not _matches([("H11", "C2"), *TEMPLATE_PAIRS[1:]])

    def test_a_hydrogen_without_a_bond_does_not_match(self):
        assert not _matches(TEMPLATE_PAIRS, bonded=False)


class TestParameterizeKeepsMatchingHydrogens:
    @pytest.fixture
    def template_step(self, monkeypatch):
        """Template step with the sqm calculation stubbed out: it returns TEMPLATE_HYDROGENS."""
        template = (
            '<ForceField><Residues><Residue name="XXX">'
            '<Atom name="C1" type="c3" charge="0.0"/><Atom name="C2" type="c3" charge="0.0"/>'
            "</Residue></Residues></ForceField>"
        )
        monkeypatch.setattr(
            gaff_ncaa,
            "_generate_residue_template",
            lambda *args, **kwargs: (template, list(TEMPLATE_HYDROGENS), [], None),
        )
        monkeypatch.setattr(gaff_ncaa, "_load_ffxml", lambda ff, ffxml: None)
        monkeypatch.setattr(gaff_ncaa, "_amber_backbone_types", lambda ff: None)

        def run(hydrogens):
            topology, positions = _residue(hydrogens)
            result = gaff_ncaa.parameterize_ncaa_residues(topology, positions, None, verbose=False)
            return topology, positions, result

        return run

    @staticmethod
    def _hydrogen_positions(topology, positions) -> dict:
        if hasattr(positions, "value_in_unit"):  # a rebuilt topology returns a Quantity
            xyz = np.asarray(positions.value_in_unit(unit.nanometer))
        else:
            xyz = np.array([[p.x, p.y, p.z] for p in positions])
        return {
            a.name: tuple(xyz[a.index]) for a in topology.atoms() if a.element == element.hydrogen
        }

    def test_matching_hydrogens_stay_where_they_are(self, template_step):
        topology, positions, (new_topology, new_positions, templates) = template_step(
            TEMPLATE_PAIRS
        )
        assert new_topology is topology and new_positions is positions
        assert self._hydrogen_positions(new_topology, new_positions) == RELAXED_POSITIONS
        assert len(templates) == 1

    def test_other_hydrogens_are_replaced_by_the_templates(self, template_step):
        _, _, (new_topology, new_positions, templates) = template_step(TEMPLATE_PAIRS[:-1])
        found = self._hydrogen_positions(new_topology, new_positions)
        assert set(found) == {name for name, _, _ in TEMPLATE_HYDROGENS}
        for name, _, position in TEMPLATE_HYDROGENS:
            np.testing.assert_allclose(found[name], position)
        assert len(templates) == 1

    def test_a_residue_without_hydrogens_gets_the_templates(self, template_step):
        _, _, (new_topology, new_positions, _) = template_step([])
        found = self._hydrogen_positions(new_topology, new_positions)
        for name, _, position in TEMPLATE_HYDROGENS:
            np.testing.assert_allclose(found[name], position)
