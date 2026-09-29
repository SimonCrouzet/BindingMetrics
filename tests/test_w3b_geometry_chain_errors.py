"""An explicit chain that is not in the structure is a ValueError naming the chains present."""

from pathlib import Path

import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics.geometry import (  # noqa: E402
    compute_omega_planarity,
    compute_ramachandran,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"  # chains A (protein) and C (peptide)

WATER_ONLY = (
    "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
    "HETATM    2  O   HOH A   2       3.000   0.000   0.000  1.00  0.00           O\n"
    "END\n"
)


@pytest.mark.parametrize("metric", [compute_ramachandran, compute_omega_planarity])
def test_absent_chain_names_it_and_the_available_ones(metric):
    if not CYCLOSPORIN.exists():
        pytest.skip("cyclosporin example not bundled")
    with pytest.raises(ValueError) as excinfo:
        metric(CYCLOSPORIN, chain="Z")
    message = str(excinfo.value)
    assert "'Z'" in message
    assert "'A'" in message and "'C'" in message


@pytest.mark.parametrize("metric", [compute_ramachandran, compute_omega_planarity])
def test_present_chain_without_amino_acids_still_reports_a_reason(metric, tmp_path):
    """Only an absent chain raises; a chain of waters returns NaN with a reason."""
    path = tmp_path / "water.pdb"
    path.write_text(WATER_ONLY, encoding="utf-8")
    result = metric(path, chain="A")
    assert "chain 'A'" in result["reason"]


@pytest.mark.parametrize("metric", [compute_ramachandran, compute_omega_planarity])
def test_absent_chain_in_a_water_only_file_lists_what_exists(metric, tmp_path):
    path = tmp_path / "water.pdb"
    path.write_text(WATER_ONLY, encoding="utf-8")
    with pytest.raises(ValueError, match=r"'B'.*\['A'\]"):
        metric(path, chain="B")
