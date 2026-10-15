"""A chain selection that names no chain of the structure, or one chain twice, is refused.

Before the check, ``compute_coulomb_cross_chain``, ``compute_hbonds`` and
``compute_saltbridges`` returned zeros for an ID that is not in the structure and
scored a chain against itself (#172).
"""

import sys

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("hydride")

from _reference_contacts import (  # noqa: E402
    coulomb_reference,
    hbond_reference,
    salt_bridge_reference,
)
from conftest import DATA_DIR  # noqa: E402

from binding_metrics.cli.run import ChainNotFoundError  # noqa: E402
from binding_metrics.metrics import electrostatics  # noqa: E402
from binding_metrics.metrics.electrostatics import compute_coulomb_cross_chain  # noqa: E402
from binding_metrics.metrics.interface import load_biotite_structure  # noqa: E402
from binding_metrics.metrics.polar_contacts import (  # noqa: E402
    compute_hbonds,
    compute_saltbridges,
)

P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"

# The package sums float32 coordinates, the references float64 ones.
RELATIVE = 1e-5


@pytest.fixture(scope="module")
def p53_atoms():
    return load_biotite_structure(P53_MDM2)


class TestCoulombChains:
    def test_two_present_chains_keep_the_energy(self):
        result = compute_coulomb_cross_chain(P53_MDM2, "B", "A")
        energy, n_pairs = coulomb_reference(P53_MDM2, "B", "A")
        # The reference must hold pairs, or the zeros of a wrongly refused selection would match it.
        assert n_pairs > 0
        assert result["coulomb_energy_kJ"] == pytest.approx(energy, rel=RELATIVE)
        assert result["n_charged_pairs"] == n_pairs
        assert "reason" not in result

    @pytest.mark.parametrize("peptide, receptor", [("missing", "A"), ("B", "missing")])
    def test_absent_chain_raises_with_the_ids(self, peptide, receptor):
        with pytest.raises(ChainNotFoundError) as caught:
            compute_coulomb_cross_chain(P53_MDM2, peptide, receptor)
        message = str(caught.value)
        assert "'missing'" in message
        assert "['A', 'B']" in message

    def test_the_error_names_the_role_of_the_absent_chain(self):
        with pytest.raises(ChainNotFoundError, match="peptide chain 'missing'"):
            compute_coulomb_cross_chain(P53_MDM2, "missing", "A")
        with pytest.raises(ChainNotFoundError, match="receptor chain 'missing'"):
            compute_coulomb_cross_chain(P53_MDM2, "B", "missing")

    def test_both_absent_names_both(self):
        with pytest.raises(ChainNotFoundError) as caught:
            compute_coulomb_cross_chain(P53_MDM2, "X", "Y")
        assert "'X'" in str(caught.value)
        assert "'Y'" in str(caught.value)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"peptide_chain": "missing"},
            {"receptor_chain": "missing"},
            {"binder_chain": "missing", "receptor_chain": "A"},
            {"peptide_chain": "B", "target_chain": "missing"},
        ],
    )
    def test_absent_chain_raises_when_the_other_is_detected_or_an_alias(self, kwargs):
        with pytest.raises(ChainNotFoundError, match="'missing'"):
            compute_coulomb_cross_chain(P53_MDM2, **kwargs)

    def test_the_same_chain_twice_raises(self):
        with pytest.raises(ValueError, match="both 'B'"):
            compute_coulomb_cross_chain(P53_MDM2, "B", "B")

    def test_a_detected_chain_equal_to_the_named_one_raises(self):
        # With the smaller chain B named as the receptor, detection puts B on the peptide too.
        with pytest.raises(ValueError, match="both 'B'"):
            compute_coulomb_cross_chain(P53_MDM2, receptor_chain="B")

    def test_neutral_partners_give_zero_without_a_reason(self, tmp_path, sample_pdb_content):
        path = tmp_path / "neutral.pdb"
        path.write_text(sample_pdb_content, encoding="utf-8")
        result = compute_coulomb_cross_chain(path, "B", "A")
        assert result["coulomb_energy_kJ"] == 0.0
        assert result["n_charged_pairs"] == 0
        assert "reason" not in result


class TestPolarContactChains:
    @pytest.mark.parametrize("peptide, receptor", [("missing", "A"), ("B", "missing")])
    def test_hbonds_absent_chain_raises_with_the_ids(self, p53_atoms, peptide, receptor):
        with pytest.raises(ChainNotFoundError) as caught:
            compute_hbonds(p53_atoms.copy(), peptide, receptor)
        assert "'missing'" in str(caught.value)
        assert "['A', 'B']" in str(caught.value)

    @pytest.mark.parametrize("peptide, receptor", [("missing", "A"), ("B", "missing")])
    def test_saltbridges_absent_chain_raises_with_the_ids(self, p53_atoms, peptide, receptor):
        with pytest.raises(ChainNotFoundError) as caught:
            compute_saltbridges(p53_atoms.copy(), peptide, receptor)
        assert "'missing'" in str(caught.value)
        assert "['A', 'B']" in str(caught.value)

    def test_the_same_chain_twice_raises(self, p53_atoms):
        with pytest.raises(ValueError, match="both 'B'"):
            compute_hbonds(p53_atoms.copy(), "B", "B")
        with pytest.raises(ValueError, match="both 'B'"):
            compute_saltbridges(p53_atoms.copy(), "B", "B")

    def test_two_present_chains_keep_the_counts(self, p53_atoms):
        hbonds = compute_hbonds(p53_atoms.copy(), "B", "A")
        saltbridges = compute_saltbridges(p53_atoms.copy(), "B", "A")
        hbond_count, _ = hbond_reference(p53_atoms, "B", "A")
        saltbridge_count, _ = salt_bridge_reference(p53_atoms, "B", "A")
        # The reference must hold bonds, or the zeros of a wrongly refused selection would match it.
        assert hbond_count > 0
        assert saltbridge_count > 0
        assert hbonds["hbonds"] == hbond_count
        assert saltbridges["saltbridges"] == saltbridge_count

    def test_hetero_ignore_judges_the_chains_that_are_left(self):
        atoms = load_biotite_structure(CYCLOSPORIN)
        atoms.chain_id[atoms.res_name == "HOH"] = "W"
        with pytest.raises(ChainNotFoundError, match="receptor chain 'W'"):
            compute_hbonds(atoms.copy(), "C", "W", hetero="ignore")
        with pytest.raises(ChainNotFoundError, match="receptor chain 'W'"):
            compute_saltbridges(atoms.copy(), "C", "W", hetero="ignore")
        assert np.any(atoms.chain_id == "W")
        # A chain of waters is present when the heteroatoms are kept: no error.
        assert compute_saltbridges(atoms.copy(), "C", "W", hetero="keep")["saltbridges"] == 0


class TestElectrostaticsCommandLine:
    def test_absent_chain_prints_the_error_and_exits_1(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-electrostatics",
                "--input",
                str(P53_MDM2),
                "--design-chain",
                "missing",
                "--receptor-chain",
                "A",
            ],
        )
        with pytest.raises(SystemExit) as caught:
            electrostatics.main()
        assert caught.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("ERROR: ")
        assert "'missing'" in err
        assert "Traceback" not in err

    def test_present_chains_still_print_the_summary(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "binding-metrics-electrostatics",
                "--input",
                str(P53_MDM2),
                "--design-chain",
                "B",
                "--receptor-chain",
                "A",
            ],
        )
        electrostatics.main()
        energy, _ = coulomb_reference(P53_MDM2, "B", "A")
        lines = capsys.readouterr().out.splitlines()
        printed = [line for line in lines if line.strip().startswith("coulomb_energy_kJ:")]
        assert len(printed) == 1
        assert float(printed[0].split(":")[1]) == pytest.approx(energy, rel=RELATIVE)
