"""Additive keys of the backbone-geometry metrics: cis count and ring-closure flags.

The existing keys must not move. The reference values below were taken from
the bundled examples before the keys were added.
"""

from pathlib import Path

import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics.geometry import (  # noqa: E402
    compute_omega_planarity,
    compute_ramachandran,
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
            assert result["cyclic_closure_evaluated"] is False

    def test_closing_bond_is_indeed_missing_from_the_statistics(self):
        """Cyclosporin has 11 residues and 11 peptide bonds; only 10 are evaluated."""
        result = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        assert result["cyclic_closure_detected"] is True
        assert result["n_bonds_evaluated"] == 10

    def test_flags_are_false_when_no_chain_is_found(self, tmp_path):
        empty = tmp_path / "water.pdb"
        empty.write_text(
            "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n"
        )
        for result in (compute_omega_planarity(empty), compute_ramachandran(empty)):
            assert result["cyclic_closure_detected"] is False
            assert result["cyclic_closure_evaluated"] is False


class TestCisCount:
    def test_cis_proline_is_counted_and_stays_an_outlier(self):
        result = compute_omega_planarity(_need(SFTI1), chain="I")
        assert result["omega_cis_count"] == 1
        # the outlier rule is unchanged: the cis bond still counts as an outlier
        assert result["omega_outlier_count"] == 1
        cis = [r for r in result["per_residue"] if abs(r["omega"]) < 30.0]
        assert len(cis) == 1
        assert cis[0]["is_outlier"] is True

    @pytest.mark.parametrize("path, chain", [(P53, "B"), (CYCLOSPORIN, "C"), (SOMATOSTATIN, "A")])
    def test_all_trans_chains_have_no_cis_bond(self, path, chain):
        assert compute_omega_planarity(_need(path), chain=chain)["omega_cis_count"] == 0

    def test_missing_chain_gives_zero(self, tmp_path):
        empty = tmp_path / "water.pdb"
        empty.write_text(
            "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n"
        )
        assert compute_omega_planarity(empty)["omega_cis_count"] == 0


class TestExistingKeysUnchanged:
    def test_sfti1_omega(self):
        result = compute_omega_planarity(_need(SFTI1), chain="I")
        assert result["omega_mean_dev"] == pytest.approx(17.558478, abs=1e-5)
        assert result["omega_max_dev"] == pytest.approx(177.129636, abs=1e-5)
        assert result["omega_outlier_fraction"] == pytest.approx(1 / 13)
        assert result["omega_outlier_count"] == 1
        assert result["n_bonds_evaluated"] == 13

    def test_cyclosporin_omega(self):
        result = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        assert result["omega_mean_dev"] == pytest.approx(2.992090, abs=1e-5)
        assert result["omega_outlier_count"] == 0

    def test_ramachandran_percentages_still_sum_to_100(self):
        result = compute_ramachandran(_need(P53), chain="B")
        total = (
            result["ramachandran_favoured_pct"]
            + result["ramachandran_allowed_pct"]
            + result["ramachandran_outlier_pct"]
        )
        assert total == pytest.approx(100.0)
