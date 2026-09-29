"""Waters that carry a protein chain ID must not count as receptor or peptide atoms."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("hydride")

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


@pytest.fixture(scope="module")
def cyclosporin():
    return load_biotite_structure(CYCLOSPORIN)


@pytest.fixture(scope="module")
def cyclosporin_dry(cyclosporin):
    """1CWA without its 144 waters, selected by residue name (independent of the hetero filter)."""
    return cyclosporin[cyclosporin.res_name != "HOH"]


class TestHbondsHetero:
    def test_waters_add_a_spurious_hbond_when_kept(self, cyclosporin):
        kept = compute_hbonds(cyclosporin.copy(), "C", "A", hetero="keep")
        ignored = compute_hbonds(cyclosporin.copy(), "C", "A", hetero="ignore")
        assert kept["hbonds"] == 6
        assert ignored["hbonds"] == 5
        assert ignored["hbond_energy"] > kept["hbond_energy"]

    def test_ignore_on_wet_structure_equals_the_dry_structure(self, cyclosporin, cyclosporin_dry):
        wet = compute_hbonds(cyclosporin.copy(), "C", "A")
        dry = compute_hbonds(cyclosporin_dry.copy(), "C", "A", hetero="keep")
        assert wet["hbonds"] == dry["hbonds"] == 5
        assert wet["hbond_energy"] == pytest.approx(dry["hbond_energy"], rel=1e-6)

    def test_default_is_ignore(self, cyclosporin):
        default = compute_hbonds(cyclosporin.copy(), "C", "A")
        explicit = compute_hbonds(cyclosporin.copy(), "C", "A", hetero="ignore")
        assert default == explicit

    def test_sfti1_trypsin_water_and_glutathione_hbonds_drop_out(self):
        atoms = load_biotite_structure(SFTI1_TRYPSIN)
        kept = compute_hbonds(atoms.copy(), "I", "A", hetero="keep")["hbonds"]
        ignored = compute_hbonds(atoms.copy(), "I", "A", hetero="ignore")["hbonds"]
        assert (kept, ignored) == (11, 9)

    def test_dry_structure_is_unchanged(self):
        atoms = load_biotite_structure(P53_MDM2)
        default = compute_hbonds(atoms.copy(), "B", "A")
        kept = compute_hbonds(atoms.copy(), "B", "A", hetero="keep")
        assert default["hbonds"] == kept["hbonds"] == 3
        assert default["hbond_energy"] == pytest.approx(-5.09272, rel=1e-5)
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
        kept = compute_saltbridges(atoms, "I", "A", hetero="keep")
        ignored = compute_saltbridges(atoms, "I", "A", hetero="ignore")
        assert kept == ignored
        assert ignored["saltbridges"] == 1
        assert ignored["saltbridge_energy"] < 0.0

    def test_unknown_mode_raises(self, cyclosporin):
        with pytest.raises(ValueError, match="hetero"):
            compute_saltbridges(cyclosporin, "C", "A", hetero="drop")


class TestInterfaceMetricsForwardsHetero:
    def test_hbonds_follow_the_hetero_mode(self):
        ignored = compute_interface_metrics(CYCLOSPORIN)
        kept = compute_interface_metrics(CYCLOSPORIN, hetero="keep")
        assert ignored["hbonds"] == 5
        assert kept["hbonds"] == 6
        assert np.isfinite(ignored["hbond_energy"])
        assert ignored["hbond_energy"] > kept["hbond_energy"]
