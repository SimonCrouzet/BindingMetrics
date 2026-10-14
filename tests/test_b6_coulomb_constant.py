"""One Coulomb constant for the three modules that use it.

``metrics/electrostatics.py`` (kJ A/(mol e^2)), ``metrics/energy.py`` (kJ nm/(mol e^2)) and
``metrics/polar_contacts.py`` (kcal A/(mol e^2)) each wrote the constant on its own, with
different roundings (relative differences of 3.3e-6 and 1.1e-5). They now import it from
``binding_metrics._constants``.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics import _constants  # noqa: E402
from binding_metrics.metrics import _common, electrostatics, energy, polar_contacts  # noqa: E402
from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain  # noqa: E402
from binding_metrics.metrics.polar_contacts import compute_saltbridges  # noqa: E402

SRC = Path(_constants.__file__).resolve().parent


def _atoms(records):
    """AtomArray from (chain, res_id, res_name, atom_name, element, xyz) tuples."""
    arr = struc.AtomArray(len(records))
    arr.chain_id = np.array([r[0] for r in records])
    arr.res_id = np.array([r[1] for r in records], dtype=int)
    arr.res_name = np.array([r[2] for r in records])
    arr.atom_name = np.array([r[3] for r in records])
    arr.element = np.array([r[4] for r in records])
    arr.coord = np.array([r[5] for r in records], dtype=float)
    return arr


def _lysine_aspartate(distance):
    """A lysine NZ (+1) of chain B facing an aspartate OD1 (-0.5) of chain A."""
    return _atoms(
        [
            ("A", 1, "ASP", "OD1", "O", (0.0, 0.0, 0.0)),
            ("B", 1, "LYS", "NZ", "N", (distance, 0.0, 0.0)),
        ]
    )


class TestTheConstant:
    def test_value_is_the_one_decided(self):
        assert _constants.COULOMB_KJ_ANG_MOL_E2 == 1389.35457

    def test_value_agrees_with_the_physical_constants(self):
        """e^2 N_A / (4 pi eps0) from CODATA 2018, in kJ A/(mol e^2)."""
        constants = pytest.importorskip("scipy.constants")
        k = (
            constants.e**2
            * constants.N_A
            / (4 * constants.pi * constants.epsilon_0)
            / 1000.0
            / constants.angstrom
        )
        assert _constants.COULOMB_KJ_ANG_MOL_E2 == pytest.approx(k, rel=1e-7)

    def test_other_units_are_derived_from_it(self):
        assert _constants.COULOMB_KCAL_ANG_MOL_E2 == _constants.COULOMB_KJ_ANG_MOL_E2 / 4.184
        assert _constants.COULOMB_KJ_NM_MOL_E2 == _constants.COULOMB_KJ_ANG_MOL_E2 / 10.0

    def test_the_calorie_is_the_one_of_the_metric_helpers(self):
        assert _common.KCAL_TO_KJ == 4.184


class TestModulesShareIt:
    def test_electrostatics_uses_the_kj_angstrom_object(self):
        assert electrostatics._COULOMB_KJ_ANG_MOL is _constants.COULOMB_KJ_ANG_MOL_E2

    def test_energy_uses_the_kj_nm_object(self):
        assert energy._COULOMB_K_KJ_NM_MOL_E2 is _constants.COULOMB_KJ_NM_MOL_E2

    def test_polar_contacts_uses_the_kcal_angstrom_object(self):
        assert polar_contacts._COULOMB_K is _constants.COULOMB_KCAL_ANG_MOL_E2


class TestNoLiteralCopyRemains:
    """No module of ``src`` writes the constant again, in code, a docstring or a comment."""

    PATTERN = re.compile(r"1389\.3|332\.06|138\.93")

    @staticmethod
    def _python_files():
        return sorted(p for p in SRC.rglob("*.py") if p.name != "_constants.py")

    def test_the_files_are_found(self):
        assert len(self._python_files()) > 50

    def test_no_text_copy(self):
        found = [
            f"{path.relative_to(SRC)}:{number}"
            for path in self._python_files()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if self.PATTERN.search(line)
        ]
        assert found == []

    def test_no_numeric_literal_close_to_it(self):
        """A rounding other than the three known ones (1389, 138.9, 332) is caught as well."""
        targets = (
            _constants.COULOMB_KJ_ANG_MOL_E2,
            _constants.COULOMB_KJ_NM_MOL_E2,
            _constants.COULOMB_KCAL_ANG_MOL_E2,
        )
        found = []
        for path in self._python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    if any(abs(node.value / t - 1.0) < 1e-3 for t in targets):
                        found.append(f"{path.relative_to(SRC)}:{node.lineno}")
        assert found == []


class TestCoulombMetric:
    def test_energy_is_k_q1_q2_over_epsilon_r(self, tmp_path):
        """A lysine NZ and an aspartate OD1 4 A apart at epsilon 4: K (+1)(-0.5) / (4 * 4)."""
        # ten inert alanines make chain A a receptor the chain detection accepts
        inert = [("A", 10 + i, "ALA", "CA", "C", (300.0 + 5 * i, 0.0, 0.0)) for i in range(10)]
        records = [
            ("A", 1, "ASP", "OD1", "O", (0.0, 0.0, 0.0)),
            ("B", 1, "LYS", "NZ", "N", (4.0, 0.0, 0.0)),
            *inert,
        ]
        file = pdb_io.PDBFile()
        pdb_io.set_structure(file, _atoms(records))
        path = tmp_path / "pair.pdb"
        file.write(str(path))

        result = compute_coulomb_cross_chain(path, "B", "A")

        expected = _constants.COULOMB_KJ_ANG_MOL_E2 * (+1.0) * (-0.5) / (4.0 * 4.0)
        assert result["coulomb_energy_kJ"] == pytest.approx(expected, rel=1e-12)
        assert result["coulomb_energy_kcal"] == pytest.approx(expected / 4.184, rel=1e-12)


class TestSaltBridgeScore:
    def test_energy_is_the_kcal_constant_over_epsilon_r(self):
        """Unit charges at epsilon 4 and 3 A: -K_kcal / (4 * 3)."""
        result = compute_saltbridges(_lysine_aspartate(3.0), "B", "A")
        expected = -_constants.COULOMB_KCAL_ANG_MOL_E2 / (4.0 * 3.0)
        assert result["saltbridge_energy"] == pytest.approx(expected, rel=1e-12)

    def test_same_energy_in_both_units(self):
        """The kcal score of the salt bridge is the kJ Coulomb sum divided by 4.184."""
        atoms = _lysine_aspartate(3.0)
        kcal = compute_saltbridges(atoms, "B", "A")["saltbridge_energy"]
        assert kcal * 4.184 == pytest.approx(
            -_constants.COULOMB_KJ_ANG_MOL_E2 / (4.0 * 3.0), rel=1e-12
        )


class TestPairSumInEnergyModule:
    """``calculate_component_energies`` and ``calculate_interaction_energy`` sum k q q / r."""

    @pytest.fixture
    def ion_pair(self, tmp_path):
        pytest.importorskip("openmm")
        pytest.importorskip("mdtraj")
        path = tmp_path / "ions.pdb"
        path.write_text(
            "HETATM    1 NA    NA A   1       0.000   0.000   0.000  1.00  0.00          NA\n"
            "HETATM    2 CL    CL B   2       5.000   0.000   0.000  1.00  0.00          CL\n"
            "END\n",
            encoding="utf-8",
        )
        return path

    def test_electrostatic_term_of_two_ions(self, ion_pair):
        components = energy.calculate_component_energies(ion_pair, ion_pair, [0], [1])
        expected = _constants.COULOMB_KJ_NM_MOL_E2 * (+1.0) * (-1.0) / 0.5
        assert components["electrostatic"][0] == pytest.approx(expected, rel=1e-12)

    def test_interaction_energy_includes_the_same_term(self, ion_pair):
        components = energy.calculate_component_energies(ion_pair, ion_pair, [0], [1])
        total = energy.calculate_interaction_energy(ion_pair, ion_pair, [0], [1])
        assert total[0] == pytest.approx(components["total"][0], rel=1e-12)
