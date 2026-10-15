"""Van der Waals radii: which table serves which metric, and that none misses a halogen or Se.

Three tables coexist on purpose:

* ``geometry._BONDI_VDW_RADII`` (Bondi 1964, hydrogen 1.20 A) for the dots of the shape
  complementarity and the grid of the buried void volume;
* ``receptor_quality._VDW_RADII`` (Bondi 1964, heavy atoms) for the clash score;
* biotite's ``vdw_radius_single`` (hydrogen 1.10 A, bromine 1.83 A) for the SASA, interface
  and pose-descriptor metrics.

The geometry table used to stop at C, N, O, S, P and H, so F, Cl, Br, I and Se silently got
the 1.80 A fallback.
"""

import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.info import vdw_radius_single

from binding_metrics.metrics import geometry, receptor_quality

HALOGENS_AND_SELENIUM = ["F", "CL", "BR", "I", "SE"]

# Bondi (1964), J. Phys. Chem. 68, 441, for the elements the metrics meet in peptides.
BONDI = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


class TestGeometryTable:
    @pytest.mark.parametrize(("element", "radius"), sorted(BONDI.items()))
    def test_bondi_radius(self, element, radius):
        assert geometry._get_vdw(element) == radius

    @pytest.mark.parametrize("spelling", ["Br", " br ", "BR", "br"])
    def test_symbol_case_and_padding_are_ignored(self, spelling):
        assert geometry._get_vdw(spelling) == 1.85

    @pytest.mark.parametrize("element", HALOGENS_AND_SELENIUM)
    def test_halogens_and_selenium_are_not_the_fallback(self, element):
        assert element in geometry._BONDI_VDW_RADII
        assert geometry._get_vdw(element) != geometry._DEFAULT_VDW_RADIUS

    def test_an_element_without_an_entry_gets_the_fallback(self):
        assert geometry._get_vdw("FE") == geometry._DEFAULT_VDW_RADIUS == 1.80

    @pytest.mark.parametrize("element", ["F", "CL", "BR", "I", "SE", "C"])
    def test_surface_dots_lie_on_the_sphere_of_the_element(self, element):
        """The dots of an interface atom sit at its Bondi radius from its centre."""
        radius = BONDI[element]
        atom = AtomArray(1)
        atom.coord = np.zeros((1, 3), dtype=np.float32)
        atom.element = np.array([element])
        partner = AtomArray(1)
        partner.coord = np.array([[0.0, 0.0, radius + 1.5]], dtype=np.float32)
        partner.element = np.array(["C"])

        dots, _ = geometry._build_surface_dots(atom, atom, partner, 150, 2.4, 6.0)

        assert len(dots) > 0
        assert np.linalg.norm(dots, axis=1) == pytest.approx(radius, abs=1e-5)


class TestOtherTables:
    @pytest.mark.parametrize("element", HALOGENS_AND_SELENIUM)
    def test_receptor_quality_has_the_halogens_and_selenium(self, element):
        assert element in receptor_quality._VDW_RADII
        assert receptor_quality._vdw(element) == BONDI[element]

    @pytest.mark.parametrize("element", HALOGENS_AND_SELENIUM)
    def test_biotite_has_the_halogens_and_selenium(self, element):
        """The table behind SASA and interface (their code falls back to 1.8 for None)."""
        assert vdw_radius_single(element) is not None

    def test_the_two_bondi_tables_agree_where_both_have_an_element(self):
        shared = set(receptor_quality._VDW_RADII) & set(geometry._BONDI_VDW_RADII)
        assert shared >= set(HALOGENS_AND_SELENIUM) | {"C", "N", "O", "S", "P"}
        for element in shared:
            assert receptor_quality._VDW_RADII[element] == geometry._BONDI_VDW_RADII[element]

    def test_the_clash_score_table_has_no_hydrogen(self):
        """Hydrogens are left out of the clash score, so the table does not carry one."""
        assert "H" not in receptor_quality._VDW_RADII


class TestDeliberateDifferences:
    """Differences between the tables that are kept; a change here moves published numbers."""

    def test_hydrogen_is_1_20_for_packing_and_1_10_for_areas(self):
        assert geometry._get_vdw("H") == 1.20
        assert vdw_radius_single("H") == 1.10

    def test_bromine_is_1_85_in_the_bondi_tables_and_1_83_in_biotites(self):
        assert geometry._get_vdw("BR") == receptor_quality._vdw("BR") == 1.85
        assert vdw_radius_single("BR") == 1.83
