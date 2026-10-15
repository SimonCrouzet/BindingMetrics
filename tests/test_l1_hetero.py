"""Waters, ions and other heteroatoms must not turn interface areas into NaN.

biotite reports NaN SASA for water and monoatomic ions. Summing that array used
to make delta_sasa and delta_g_int NaN on any structure that kept them.
"""

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402
from biotite.structure.info import vdw_radius_single  # noqa: E402
from biotite.structure.sasa import sasa as biotite_sasa  # noqa: E402

from binding_metrics.metrics.interface import (  # noqa: E402
    compute_interface_metrics,
    filter_hetero_atoms,
    load_biotite_structure,
)
from binding_metrics.metrics.sasa import compute_delta_sasa_static  # noqa: E402
from tests.test_l1_solvation import (  # noqa: E402
    RELATIVE,
    expected_interface,
    polymer_atoms,
    read_atoms,
)

DATA = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1_TRYPSIN = DATA / "example_bicyclic_sfti1_3P8F.cif"


# The expected interface values come from biotite and the published solvation parameters
# (helpers of tests/test_l1_solvation.py), on the water-free polymer of the file.
@pytest.fixture(scope="module")
def p53_expected():
    """delta_sasa and delta_g_int of 1YCR (chain B on chain A), which has no heteroatoms."""
    return expected_interface(polymer_atoms(read_atoms(P53_MDM2)), "B", "A")


@pytest.fixture(scope="module")
def cyclosporin_expected():
    """delta_sasa and delta_g_int of the 1CWA polymer (chain C on chain A), waters left out."""
    return expected_interface(polymer_atoms(read_atoms(CYCLOSPORIN)), "C", "A")


@pytest.fixture(scope="module")
def sfti1_expected():
    """delta_sasa and delta_g_int of the 3P8F polymer (chain I on chain A), without ligands."""
    return expected_interface(polymer_atoms(read_atoms(SFTI1_TRYPSIN)), "I", "A")


def assert_matches(result, expected):
    assert result["delta_sasa"] == pytest.approx(expected["delta_sasa"], rel=RELATIVE)
    assert result["delta_g_int"] == pytest.approx(expected["delta_g_int"], rel=RELATIVE)


@pytest.fixture(scope="module")
def p53_with_iodide_and_water(tmp_path_factory) -> Path:
    """1YCR with a monoatomic iodide and an O-only water labelled as receptor chain A."""
    atoms = pdb_io.get_structure(pdb_io.PDBFile.read(str(P53_MDM2)), model=1)
    extra = atoms[atoms.chain_id == "A"][:2].copy()
    extra.res_name[:] = ["IOD", "HOH"]
    extra.atom_name[:] = ["I", "O"]
    extra.element[:] = ["I", "O"]
    extra.hetero[:] = True
    extra.res_id[:] = [900, 901]
    far_away = atoms.coord.max(axis=0) + 8.0
    extra.coord[:] = [far_away, far_away + [3.0, 0.0, 0.0]]

    path = tmp_path_factory.mktemp("hetero") / "p53_iodide_water.pdb"
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, struc.concatenate([atoms, extra]))
    pdb_file.write(str(path))
    return path


def test_synthetic_structure_reproduces_the_nan_premise(p53_with_iodide_and_water):
    atoms = load_biotite_structure(p53_with_iodide_and_water)
    radii = np.array([vdw_radius_single(str(e)) or 1.8 for e in atoms.element], dtype=float)
    per_atom = biotite_sasa(atoms, vdw_radii=radii, point_number=960)
    assert np.isnan(per_atom[-2:]).all()
    assert np.isfinite(per_atom[:-2]).all()


class TestWaterAndIonRegression:
    @pytest.mark.parametrize("hetero", ["ignore", "keep"])
    def test_synthetic_iodide_and_water_do_not_poison_interface(
        self, p53_with_iodide_and_water, p53_expected, hetero
    ):
        result = compute_interface_metrics(p53_with_iodide_and_water, "B", "A", hetero=hetero)
        assert "reason" not in result
        # The added atoms sit far from the complex: the result is that of the file without them.
        assert_matches(result, p53_expected)
        assert np.isfinite(result["fraction_polar"])

    @pytest.mark.parametrize("hetero", ["ignore", "keep"])
    def test_synthetic_iodide_and_water_do_not_poison_static_sasa(
        self, p53_with_iodide_and_water, p53_expected, hetero
    ):
        result = compute_delta_sasa_static(p53_with_iodide_and_water, "B", "A", hetero=hetero)
        assert "reason" not in result
        assert result["delta_sasa"] == pytest.approx(p53_expected["delta_sasa"], rel=RELATIVE)
        assert all(np.isfinite(v) for v in result.values())

    @pytest.mark.parametrize("hetero", ["ignore", "keep"])
    def test_cyclosporin_with_144_waters_matches_the_dry_complex(
        self, hetero, cyclosporin_expected
    ):
        result = compute_interface_metrics(CYCLOSPORIN, hetero=hetero)
        assert_matches(result, cyclosporin_expected)
        assert np.isfinite(result["polar_area"] + result["apolar_area"])
        assert "reason" not in result

    def test_cyclosporin_static_sasa_is_finite(self, cyclosporin_expected):
        result = compute_delta_sasa_static(CYCLOSPORIN, "C", "A")
        assert result["delta_sasa"] == pytest.approx(
            cyclosporin_expected["delta_sasa"], rel=RELATIVE
        )
        assert "reason" not in result

    def test_sfti1_trypsin_with_waters_and_glutathione_is_finite(self, sfti1_expected):
        interface = compute_interface_metrics(SFTI1_TRYPSIN)
        static = compute_delta_sasa_static(SFTI1_TRYPSIN, "I", "A")
        # The polymer value excludes the waters and the glutathione of the file.
        assert_matches(interface, sfti1_expected)
        assert static["delta_sasa"] == pytest.approx(sfti1_expected["delta_sasa"], rel=RELATIVE)

    def test_dry_structure_is_unchanged(self, p53_expected):
        result = compute_interface_metrics(P53_MDM2)
        assert_matches(result, p53_expected)
        assert "reason" not in result


class TestFilterHeteroAtoms:
    def test_ignore_keeps_only_amino_acid_atoms(self):
        atoms = load_biotite_structure(CYCLOSPORIN)
        kept = filter_hetero_atoms(atoms, "ignore")
        assert len(kept) == len(atoms) - 144
        assert not np.isin(kept.res_name, ["HOH"]).any()
        # D-alanine and N-methyl leucine of cyclosporin are peptide-linking residues.
        assert np.isin("DAL", kept.res_name)
        assert np.isin("MLE", kept.res_name)

    def test_keep_returns_the_input(self):
        atoms = load_biotite_structure(CYCLOSPORIN)
        assert filter_hetero_atoms(atoms, "keep") is atoms

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="hetero"):
            filter_hetero_atoms(struc.AtomArray(0), "drop")

    def test_interface_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="hetero"):
            compute_interface_metrics(P53_MDM2, hetero="drop")


class TestPolymerVariantsSurviveIgnore:
    """AMBER protonation variants and terminal caps belong to the chain, not to the solvent."""

    def test_variants_and_caps_are_kept_and_solvent_is_dropped(self):
        atoms = load_biotite_structure(P53_MDM2)[:8].copy()
        atoms.res_name[:] = ["HIE", "HID", "CYX", "ASH", "ACE", "NME", "NH2", "HOH"]
        kept = filter_hetero_atoms(atoms, "ignore")
        assert list(kept.res_name) == ["HIE", "HID", "CYX", "ASH", "ACE", "NME", "NH2"]

    def test_histidine_variant_names_do_not_change_the_interface(self, tmp_path, p53_expected):
        atoms = load_biotite_structure(P53_MDM2)
        histidines = atoms.res_name == "HIS"
        assert histidines.sum() > 0
        atoms.res_name[histidines] = "HIE"
        path = tmp_path / "p53_hie.pdb"
        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, atoms)
        pdb_file.write(str(path))

        result = compute_interface_metrics(path, "B", "A")
        assert_matches(result, p53_expected)


class TestReason:
    def test_missing_chain_keeps_nan_and_explains(self):
        result = compute_interface_metrics(P53_MDM2, design_chain="Z", receptor_chain="A")
        assert np.isnan(result["delta_sasa"])
        assert np.isnan(result["delta_g_int"])
        assert "'Z'" in result["reason"]

    def test_static_missing_chain_keeps_zero_and_explains(self):
        result = compute_delta_sasa_static(P53_MDM2, "Z", "A")
        assert result["delta_sasa"] == 0.0
        assert "'Z'" in result["reason"]

    def test_chain_of_only_heteroatoms_is_empty_under_ignore(self, tmp_path):
        atoms = load_biotite_structure(P53_MDM2)
        ion = atoms[:1].copy()
        ion.chain_id[:] = "Z"
        ion.res_name[:] = "IOD"
        ion.atom_name[:] = "I"
        ion.element[:] = "I"
        ion.hetero[:] = True
        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, struc.concatenate([atoms, ion]))
        path = tmp_path / "ion_chain.pdb"
        pdb_file.write(str(path))

        result = compute_interface_metrics(path, design_chain="Z", receptor_chain="A")
        assert np.isnan(result["delta_sasa"])
        assert "'Z'" in result["reason"]
        assert result["reason"].endswith("undefined")


class TestHeteroCli:
    def test_flag_defaults_to_ignore_and_reaches_the_metric(
        self, monkeypatch, capsys, cyclosporin_expected
    ):
        from binding_metrics.metrics import interface

        seen = {}

        def fake_metrics(*args, **kwargs):
            seen.update(kwargs)
            return compute_interface_metrics(*args, **kwargs)

        monkeypatch.setattr(interface, "compute_interface_metrics", fake_metrics)

        monkeypatch.setattr("sys.argv", ["binding-metrics-interface", "--input", str(CYCLOSPORIN)])
        interface.main()
        assert seen["hetero"] == "ignore"
        printed = re.search(r"^\s+delta_sasa: (\S+)$", capsys.readouterr().out, re.MULTILINE)
        assert float(printed.group(1)) == pytest.approx(
            cyclosporin_expected["delta_sasa"], rel=RELATIVE
        )

        monkeypatch.setattr(
            "sys.argv",
            ["binding-metrics-interface", "--input", str(CYCLOSPORIN), "--hetero", "keep"],
        )
        interface.main()
        assert seen["hetero"] == "keep"

    def test_flag_rejects_unknown_choice(self, monkeypatch):
        from binding_metrics.metrics import interface

        monkeypatch.setattr(
            "sys.argv",
            ["binding-metrics-interface", "--input", str(P53_MDM2), "--hetero", "drop"],
        )
        with pytest.raises(SystemExit):
            interface.main()
