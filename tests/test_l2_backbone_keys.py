"""Additive keys of the backbone-geometry metrics: cis count and ring-closure flags.

The expected omega values of the bundled examples are recomputed with
``biotite.structure.dihedral_backbone`` on the same file. For a head-to-tail ring the
closing amide is one of the evaluated bonds (``tests/test_w3b_cyclic_dihedrals.py``).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics.geometry import (  # noqa: E402
    compute_omega_planarity,
    compute_ramachandran,
)
from tests.test_w3b_cyclic_dihedrals import (  # noqa: E402
    CIS_OMEGA_DEG,
    OMEGA_OUTLIER_DEG,
    backbone_angles_deg,
    deviation_from_trans_deg,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"  # chain C: head-to-tail 11-mer
SFTI1 = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"  # chain I: head-to-tail, one cis-Pro
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"  # chain B: linear
SOMATOSTATIN = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"  # side-chain lactam only


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} not bundled")
    return path


class TestClosureFlags:
    @pytest.mark.parametrize(
        "path, chain, detected",
        [
            (CYCLOSPORIN, "C", True),
            (SFTI1, "I", True),
            (P53, "B", False),
            (SOMATOSTATIN, "A", False),
        ],
    )
    def test_detection_on_bundled_examples(self, path, chain, detected):
        _need(path)
        for result in (
            compute_omega_planarity(path, chain=chain),
            compute_ramachandran(path, chain=chain),
        ):
            assert result["cyclic_closure_detected"] is detected
            # a detected ring closure is always evaluated, a linear chain never is
            assert result["cyclic_closure_evaluated"] is detected

    def test_closing_bond_is_part_of_the_statistics(self):
        """Cyclosporin has 11 residues and 11 peptide bonds; all 11 are evaluated."""
        result = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        assert result["cyclic_closure_detected"] is True
        assert result["n_bonds_evaluated"] == 11

    def test_flags_are_false_when_no_chain_is_found(self, tmp_path):
        empty = tmp_path / "water.pdb"
        empty.write_text(
            "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n",
            encoding="utf-8",
        )
        for result in (compute_omega_planarity(empty), compute_ramachandran(empty)):
            assert result["cyclic_closure_detected"] is False
            assert result["cyclic_closure_evaluated"] is False


class TestCisCount:
    def test_cis_proline_is_counted_and_stays_an_outlier(self):
        result = compute_omega_planarity(_need(SFTI1), chain="I")
        _, _, reference = backbone_angles_deg(SFTI1, "I", ring=True)
        cis_bonds = np.abs(reference) < CIS_OMEGA_DEG
        assert cis_bonds.sum() == 1  # the cis proline
        assert result["omega_cis_count"] == cis_bonds.sum()
        # the outlier rule is unchanged: the cis bond still counts as an outlier, next to the
        # bonds that are more than the cut-off from trans
        outliers = deviation_from_trans_deg(reference) > OMEGA_OUTLIER_DEG
        assert result["omega_outlier_count"] == outliers.sum()
        cis = [r for r in result["per_residue"] if abs(r["omega"]) < CIS_OMEGA_DEG]
        assert len(cis) == cis_bonds.sum()
        assert cis[0]["is_outlier"] is True

    @pytest.mark.parametrize("path, chain", [(P53, "B"), (CYCLOSPORIN, "C"), (SOMATOSTATIN, "A")])
    def test_all_trans_chains_have_no_cis_bond(self, path, chain):
        assert compute_omega_planarity(_need(path), chain=chain)["omega_cis_count"] == 0

    def test_missing_chain_gives_zero(self, tmp_path):
        empty = tmp_path / "water.pdb"
        empty.write_text(
            "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n",
            encoding="utf-8",
        )
        assert compute_omega_planarity(empty)["omega_cis_count"] == 0


class TestExistingKeysUnchanged:
    # The reference is computed from the same file on the machine that runs the test: torsions
    # come from float32 coordinates and differ by up to 2e-5 degree between machines.
    def test_sfti1_omega(self):
        """13 sequential bonds plus the closing bond."""
        result = compute_omega_planarity(_need(SFTI1), chain="I")
        _, _, reference = backbone_angles_deg(SFTI1, "I", ring=True)
        deviations = deviation_from_trans_deg(reference)
        n_outliers = int((deviations > OMEGA_OUTLIER_DEG).sum())
        # the statistics cover the 13 sequential bonds and the closing one, none left out
        assert result["omega_mean_dev"] == pytest.approx(deviations.mean(), abs=1e-3)
        assert result["omega_max_dev"] == pytest.approx(deviations.max(), abs=1e-3)
        assert result["omega_outlier_fraction"] == pytest.approx(n_outliers / len(deviations))
        assert result["omega_outlier_count"] == n_outliers
        assert result["n_bonds_evaluated"] == len(deviations)
        sequential = [r["deviation"] for r in result["per_residue"][:-1]]
        assert len(sequential) == len(deviations) - 1
        assert sum(sequential) / len(sequential) == pytest.approx(deviations[:-1].mean(), abs=1e-3)

    def test_cyclosporin_omega(self):
        """10 sequential bonds plus the closing bond."""
        result = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        _, _, reference = backbone_angles_deg(CYCLOSPORIN, "C", ring=True)
        deviations = deviation_from_trans_deg(reference)
        # the statistics cover the 10 sequential bonds and the closing one, none left out
        assert result["omega_mean_dev"] == pytest.approx(deviations.mean(), abs=1e-3)
        assert result["omega_outlier_count"] == (deviations > OMEGA_OUTLIER_DEG).sum()
        sequential = [r["deviation"] for r in result["per_residue"][:-1]]
        assert len(sequential) == len(deviations) - 1
        assert sum(sequential) / len(sequential) == pytest.approx(deviations[:-1].mean(), abs=1e-3)

    def test_ramachandran_percentages_still_sum_to_100(self):
        result = compute_ramachandran(_need(P53), chain="B")
        total = (
            result["ramachandran_favoured_pct"]
            + result["ramachandran_allowed_pct"]
            + result["ramachandran_outlier_pct"]
        )
        assert total == pytest.approx(100.0)
