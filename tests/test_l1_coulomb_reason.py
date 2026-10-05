"""Coulomb energy: the all-zero default for undetected chains carries a `reason`."""

from pathlib import Path

import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"


def test_evaluated_energy_has_no_reason():
    assert "reason" not in compute_coulomb_cross_chain(P53_MDM2)


class TestCoulombDefault:
    def test_single_chain_file_keeps_zeros_and_gains_a_reason(self):
        result = compute_coulomb_cross_chain(SOMATOSTATIN)
        assert result["coulomb_energy_kJ"] == 0.0
        assert result["n_charged_pairs"] == 0
        assert "receptor_chain=None" in result["reason"]

    def test_genuine_zero_has_no_reason(self):
        # Cyclosporin carries no charged atom, so 0.0 is the evaluated answer.
        result = compute_coulomb_cross_chain(CYCLOSPORIN)
        assert result["coulomb_energy_kJ"] == 0.0
        assert "reason" not in result
