"""Waters that carry a protein chain ID must not count as receptor or peptide atoms.

The expected counts and scores come from ``tests/_reference_contacts.py``, which finds the
contacts with hydride and biotite called directly on the structure with and without its waters.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("hydride")

from _reference_contacts import hbond_reference, salt_bridge_reference  # noqa: E402

from binding_metrics.metrics.interface import (  # noqa: E402
    compute_interface_metrics,
    load_biotite_structure,
)
from binding_metrics.metrics.polar_contacts import (  # noqa: E402
    compute_hbonds,
    compute_saltbridges,
)

DATA = Path(__file__).resolve().parents[1] / "data"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1_TRYPSIN = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"

# The package sums float32 terms, the reference float64 ones.
SCORE_RELATIVE = 1e-5


@pytest.fixture(scope="module")
def cyclosporin():
    return load_biotite_structure(CYCLOSPORIN)


@pytest.fixture(scope="module")
def cyclosporin_dry(cyclosporin):
    """1CWA without its 144 waters, selected by residue name (independent of the hetero filter)."""
    return cyclosporin[cyclosporin.res_name != "HOH"]


@pytest.fixture(scope="module")
def cyclosporin_reference(cyclosporin, cyclosporin_dry):
    """(count, score) of the C-A hydrogen bonds of 1CWA with its waters (wet) and without (dry)."""
    return {
        "wet": hbond_reference(cyclosporin, "C", "A"),
        "dry": hbond_reference(cyclosporin_dry, "C", "A"),
    }


class TestHbondsHetero:
    def test_waters_add_a_spurious_hbond_when_kept(self, cyclosporin, cyclosporin_reference):
        (wet_count, wet_score), (dry_count, dry_score) = (
            cyclosporin_reference["wet"],
            cyclosporin_reference["dry"],
        )
        kept = compute_hbonds(cyclosporin.copy(), "C", "A", hetero="keep")
        ignored = compute_hbonds(cyclosporin.copy(), "C", "A", hetero="ignore")
        # A water must be able to add a bond, or a filter that fails to drop it goes unseen.
        assert wet_count > dry_count
        assert kept["hbonds"] == wet_count
        assert ignored["hbonds"] == dry_count
        assert kept["hbond_energy"] == pytest.approx(wet_score, rel=SCORE_RELATIVE)
        assert ignored["hbond_energy"] == pytest.approx(dry_score, rel=SCORE_RELATIVE)
        assert ignored["hbond_energy"] > kept["hbond_energy"]

    def test_ignore_on_wet_structure_equals_the_dry_structure(
        self, cyclosporin, cyclosporin_dry, cyclosporin_reference
    ):
        dry_count, dry_score = cyclosporin_reference["dry"]
        wet = compute_hbonds(cyclosporin.copy(), "C", "A")
        dry = compute_hbonds(cyclosporin_dry.copy(), "C", "A", hetero="keep")
        # Without a bond the equality of the two routes would be empty.
        assert dry_count > 0
        assert wet["hbonds"] == dry["hbonds"] == dry_count
        assert wet["hbond_energy"] == pytest.approx(dry["hbond_energy"], rel=1e-6)
        assert dry["hbond_energy"] == pytest.approx(dry_score, rel=SCORE_RELATIVE)

    def test_default_is_ignore(self, cyclosporin):
        default = compute_hbonds(cyclosporin.copy(), "C", "A")
        explicit = compute_hbonds(cyclosporin.copy(), "C", "A", hetero="ignore")
        assert default == explicit

    def test_sfti1_trypsin_water_and_glutathione_hbonds_drop_out(self):
        atoms = load_biotite_structure(SFTI1_TRYPSIN)
        wet_count, _ = hbond_reference(atoms, "I", "A")
        dry_count, _ = hbond_reference(atoms[~np.isin(atoms.res_name, ["HOH", "GSH"])], "I", "A")
        kept = compute_hbonds(atoms.copy(), "I", "A", hetero="keep")["hbonds"]
        ignored = compute_hbonds(atoms.copy(), "I", "A", hetero="ignore")["hbonds"]
        # The water and glutathione contacts must exist for their removal to be observable.
        assert wet_count > dry_count
        assert (kept, ignored) == (wet_count, dry_count)

    def test_dry_structure_is_unchanged(self):
        atoms = load_biotite_structure(P53_MDM2)
        count, score = hbond_reference(atoms, "B", "A")
        default = compute_hbonds(atoms.copy(), "B", "A")
        kept = compute_hbonds(atoms.copy(), "B", "A", hetero="keep")
        # Without a bond the equality of the two modes would be empty.
        assert count > 0
        assert default["hbonds"] == kept["hbonds"] == count
        assert default["hbond_energy"] == pytest.approx(score, rel=SCORE_RELATIVE)
        assert default["hbond_energy"] == kept["hbond_energy"]

    def test_ignore_leaves_the_callers_atom_array_untouched(self, cyclosporin):
        atoms = cyclosporin.copy()
        assert atoms.bonds is None
        compute_hbonds(atoms, "C", "A", hetero="ignore")
        assert atoms.bonds is None

    def test_unknown_mode_raises(self, cyclosporin):
        with pytest.raises(ValueError, match="hetero"):
            compute_hbonds(cyclosporin.copy(), "C", "A", hetero="drop")


class TestSaltbridgesHetero:
    def test_waters_and_ions_do_not_change_salt_bridges(self):
        atoms = load_biotite_structure(SFTI1_TRYPSIN)
        count, score = salt_bridge_reference(atoms, "I", "A")
        kept = compute_saltbridges(atoms, "I", "A", hetero="keep")
        ignored = compute_saltbridges(atoms, "I", "A", hetero="ignore")
        assert kept == ignored
        # Without a bridge the equality of the two modes would be empty.
        assert count > 0
        assert ignored["saltbridges"] == count
        assert ignored["saltbridge_energy"] == pytest.approx(score, rel=SCORE_RELATIVE)
        assert ignored["saltbridge_energy"] < 0.0

    def test_unknown_mode_raises(self, cyclosporin):
        with pytest.raises(ValueError, match="hetero"):
            compute_saltbridges(cyclosporin, "C", "A", hetero="drop")


class TestInterfaceMetricsForwardsHetero:
    def test_hbonds_follow_the_hetero_mode(self, cyclosporin_reference):
        ignored = compute_interface_metrics(CYCLOSPORIN)
        kept = compute_interface_metrics(CYCLOSPORIN, hetero="keep")
        assert ignored["hbonds"] == cyclosporin_reference["dry"][0]
        assert kept["hbonds"] == cyclosporin_reference["wet"][0]
        assert np.isfinite(ignored["hbond_energy"])
        assert ignored["hbond_energy"] > kept["hbond_energy"]
