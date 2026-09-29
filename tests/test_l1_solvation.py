"""Atom-type-aware Eisenberg-McLachlan solvation parameters in the interface metrics."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402

from binding_metrics.metrics import interface  # noqa: E402
from binding_metrics.metrics.interface import (  # noqa: E402
    _SOLVATION_PARAMS,
    _gamma_array,
    _solvation_types,
    compute_interface_metrics,
    load_biotite_structure,
)

DATA = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1_TRYPSIN = DATA / "example_bicyclic_sfti1_3P8F.cif"

# Eisenberg & McLachlan, Nature 319:199 (1986), cal/mol/Å² of accessible area,
# as tabulated in Table 1 (column A) of Krissinel & Henrick, JMB 372:774 (2007).
EISENBERG_MCLACHLAN_CAL = {"C": 16, "S": 21, "N/O": -6, "O-": -24, "N+": -50}


def _atoms(records):
    """AtomArray from (res_name, atom_name, element) tuples."""
    arr = struc.AtomArray(len(records))
    arr.res_name = np.array([r[0] for r in records])
    arr.atom_name = np.array([r[1] for r in records])
    arr.element = np.array([r[2] for r in records])
    return arr


class TestParameterTable:
    @pytest.mark.parametrize("atom_type", list(EISENBERG_MCLACHLAN_CAL))
    def test_value_is_minus_the_published_parameter_per_buried_area(self, atom_type):
        published_kcal = EISENBERG_MCLACHLAN_CAL[atom_type] / 1000.0
        assert _SOLVATION_PARAMS[atom_type] == pytest.approx(-published_kcal)

    def test_neutral_polar_atoms_are_far_cheaper_than_charged_ones(self):
        assert abs(_SOLVATION_PARAMS["N/O"]) < abs(_SOLVATION_PARAMS["O-"])
        assert abs(_SOLVATION_PARAMS["N/O"]) < abs(_SOLVATION_PARAMS["N+"])

    def test_burying_apolar_atoms_is_favorable_and_polar_atoms_is_not(self):
        assert _SOLVATION_PARAMS["C"] < 0.0
        assert _SOLVATION_PARAMS["S"] < 0.0
        assert all(_SOLVATION_PARAMS[t] > 0.0 for t in ("N/O", "O-", "N+"))


class TestAtomTyping:
    def test_types_follow_element_and_charge_state(self):
        atoms = _atoms(
            [
                ("ALA", "CA", "C"),
                ("ALA", "N", "N"),  # backbone amide: neutral
                ("ALA", "O", "O"),  # backbone carbonyl: neutral
                ("SER", "OG", "O"),
                ("TRP", "NE1", "N"),
                ("HIS", "NE2", "N"),  # plain histidine is neutral
                ("LYS", "NZ", "N"),
                ("ARG", "NH1", "N"),
                ("ARG", "NE", "N"),
                ("ASP", "OD1", "O"),
                ("GLU", "OE2", "O"),
                ("CYS", "SG", "S"),
                ("MET", "SD", "S"),
                ("ALA", "HA", "H"),
                ("SEP", "P", "P"),
            ]
        )
        assert list(_solvation_types(atoms)) == [
            "C",
            "N/O",
            "N/O",
            "N/O",
            "N/O",
            "N/O",
            "N+",
            "N+",
            "N+",
            "O-",
            "O-",
            "S",
            "S",
            "",
            "",
        ]

    def test_gamma_array_matches_the_table(self):
        atoms = _atoms([("LYS", "NZ", "N"), ("ALA", "N", "N"), ("ALA", "H", "H")])
        np.testing.assert_allclose(
            _gamma_array(atoms),
            [_SOLVATION_PARAMS["N+"], _SOLVATION_PARAMS["N/O"], 0.0],
        )

    def test_real_structure_has_all_five_types(self):
        atoms = load_biotite_structure(P53_MDM2)
        assert set(_solvation_types(atoms)) >= {"C", "S", "N/O", "O-", "N+"}


class TestDeltaGInt:
    def test_p53_mdm2_is_favorable(self):
        result = compute_interface_metrics(P53_MDM2)
        assert result["delta_g_int"] == pytest.approx(-11.05, abs=0.05)

    def test_cyclosporin_cyclophilin_with_waters_is_favorable(self):
        result = compute_interface_metrics(CYCLOSPORIN)
        assert result["delta_g_int"] == pytest.approx(-6.11, abs=0.05)
        assert result["delta_sasa"] == pytest.approx(985.4, abs=0.5)

    def test_sfti1_trypsin_is_favorable(self):
        result = compute_interface_metrics(SFTI1_TRYPSIN)
        assert result["delta_g_int"] == pytest.approx(-4.97, abs=0.05)

    def test_kj_value_is_the_kcal_value_converted(self):
        result = compute_interface_metrics(P53_MDM2)
        assert result["delta_g_int_kJ"] == pytest.approx(result["delta_g_int"] * 4.184, rel=1e-9)

    def test_per_residue_terms_add_up_to_the_total(self):
        result = compute_interface_metrics(P53_MDM2, interface_threshold=0.0)
        per_residue_total = sum(r["delta_g_res"] for r in result["per_residue"])
        assert per_residue_total == pytest.approx(result["delta_g_int"], abs=1e-6)

    def test_areas_are_independent_of_the_parameters(self, monkeypatch):
        reference = compute_interface_metrics(P53_MDM2)
        monkeypatch.setattr(interface, "_SOLVATION_PARAMS", {k: 0.0 for k in _SOLVATION_PARAMS})
        zeroed = compute_interface_metrics(P53_MDM2)
        assert zeroed["delta_g_int"] == 0.0
        assert zeroed["delta_sasa"] == reference["delta_sasa"]
        assert zeroed["polar_area"] == reference["polar_area"]
