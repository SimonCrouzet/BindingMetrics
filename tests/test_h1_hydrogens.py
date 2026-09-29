"""Buried area, the polar/apolar partition and ΔG_int are computed on heavy atoms.

The Eisenberg-McLachlan parameters describe the accessible area of heavy atoms.
With explicit hydrogens the surface of a carbon or nitrogen is partly taken by
its attached H atoms, which carry no parameter, so the areas of the
parameterised atoms shrank while the total still counted the H atoms. On the
relaxed MDM2-p53 complex (1YCR, 851 H of 1670 atoms) that gave
delta_sasa 1552.4 with polar + apolar 651 and delta_g_int -0.95, against
1488.4, 1488.4 and -10.50 for the same coordinates without hydrogens.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
hydride = pytest.importorskip("hydride")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics import interface  # noqa: E402
from binding_metrics.metrics.interface import (  # noqa: E402
    compute_interface_metrics,
    filter_hydrogens,
)
from binding_metrics.metrics.sasa import compute_delta_sasa_static  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA / "example_linear_p53_1YCR.pdb"

SCALAR_KEYS = (
    "delta_sasa",
    "sasa_peptide",
    "sasa_receptor",
    "sasa_complex",
    "delta_g_int",
    "delta_g_int_kJ",
    "polar_area",
    "apolar_area",
    "fraction_polar",
)


@pytest.fixture(scope="module")
def p53_protonated(tmp_path_factory) -> Path:
    """1YCR with hydrogens added by hydride at ideal geometry (about 850 H)."""
    atoms = pdb_io.get_structure(pdb_io.PDBFile.read(str(P53_MDM2)), model=1, include_bonds=True)
    atoms.set_annotation("charge", np.zeros(len(atoms), dtype=int))
    protonated, _ = hydride.add_hydrogen(atoms)
    assert (protonated.element == "H").sum() > 500

    path = tmp_path_factory.mktemp("h1") / "p53_with_hydrogens.pdb"
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, protonated)
    pdb_file.write(str(path))
    return path


@pytest.fixture(scope="module")
def p53_deuterated(tmp_path_factory, p53_protonated) -> Path:
    """The protonated copy with every H relabelled as deuterium."""
    atoms = pdb_io.get_structure(pdb_io.PDBFile.read(str(p53_protonated)), model=1)
    atoms.element[atoms.element == "H"] = "D"
    assert (atoms.element == "D").sum() > 500

    path = tmp_path_factory.mktemp("h1_d") / "p53_with_deuterium.pdb"
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, atoms)
    pdb_file.write(str(path))
    return path


@pytest.fixture(scope="module")
def raw_interface():
    return compute_interface_metrics(P53_MDM2, "B", "A")


class TestRawStructureHasNoHydrogens:
    def test_both_modes_are_bit_identical_for_the_interface(self, raw_interface):
        kept = compute_interface_metrics(P53_MDM2, "B", "A", hydrogens="keep")
        for key in SCALAR_KEYS:
            assert kept[key] == raw_interface[key], key
        assert kept["per_residue"] == raw_interface["per_residue"]

    def test_both_modes_are_bit_identical_for_static_sasa(self):
        ignored = compute_delta_sasa_static(P53_MDM2, "B", "A", hydrogens="ignore")
        kept = compute_delta_sasa_static(P53_MDM2, "B", "A", hydrogens="keep")
        assert ignored == kept

    def test_default_is_ignore(self, raw_interface):
        explicit = compute_interface_metrics(P53_MDM2, "B", "A", hydrogens="ignore")
        for key in SCALAR_KEYS:
            assert explicit[key] == raw_interface[key], key


class TestProtonatedStructure:
    def test_interface_ignore_mode_matches_the_heavy_atom_file(self, p53_protonated, raw_interface):
        result = compute_interface_metrics(p53_protonated, "B", "A")
        assert result["delta_sasa"] == pytest.approx(raw_interface["delta_sasa"], rel=0.05)
        assert result["delta_g_int"] == pytest.approx(raw_interface["delta_g_int"], rel=0.05)
        # The heavy atoms keep their coordinates and order, so the match is exact.
        assert result["delta_sasa"] == pytest.approx(raw_interface["delta_sasa"], rel=1e-6)
        assert result["delta_g_int"] == pytest.approx(raw_interface["delta_g_int"], rel=1e-6)
        assert result["polar_area"] == pytest.approx(raw_interface["polar_area"], rel=1e-6)
        assert result["apolar_area"] == pytest.approx(raw_interface["apolar_area"], rel=1e-6)

    def test_static_ignore_mode_matches_the_heavy_atom_file(self, p53_protonated, raw_interface):
        result = compute_delta_sasa_static(p53_protonated, "B", "A")
        assert result["delta_sasa"] == pytest.approx(raw_interface["delta_sasa"], rel=0.05)
        assert result["delta_sasa"] == pytest.approx(raw_interface["delta_sasa"], rel=1e-6)
        assert result["sasa_complex"] == pytest.approx(raw_interface["sasa_complex"], rel=1e-6)

    def test_keep_mode_reproduces_the_inflated_area_and_the_collapsed_solvation_term(
        self, p53_protonated, raw_interface
    ):
        result = compute_interface_metrics(p53_protonated, "B", "A", hydrogens="keep")
        assert result["delta_sasa"] > 1.03 * raw_interface["delta_sasa"]
        # The H atoms are in neither mask, so the partition no longer adds up ...
        assert result["polar_area"] + result["apolar_area"] < 0.6 * result["delta_sasa"]
        # ... and the parameterised atoms lose the area their hydrogens took.
        assert abs(result["delta_g_int"]) < 0.5 * abs(raw_interface["delta_g_int"])

    def test_static_keep_mode_counts_the_hydrogens(self, p53_protonated, raw_interface):
        result = compute_delta_sasa_static(p53_protonated, "B", "A", hydrogens="keep")
        assert result["delta_sasa"] > 1.03 * raw_interface["delta_sasa"]

    def test_keep_mode_static_and_interface_agree(self, p53_protonated):
        interface_result = compute_interface_metrics(p53_protonated, "B", "A", hydrogens="keep")
        static_result = compute_delta_sasa_static(p53_protonated, "B", "A", hydrogens="keep")
        assert static_result["delta_sasa"] == pytest.approx(
            interface_result["delta_sasa"], rel=1e-6
        )

    def test_deuterium_is_dropped_like_hydrogen(self, p53_deuterated, raw_interface):
        result = compute_interface_metrics(p53_deuterated, "B", "A")
        assert result["delta_sasa"] == pytest.approx(raw_interface["delta_sasa"], rel=1e-6)
        static = compute_delta_sasa_static(p53_deuterated, "B", "A")
        assert static["delta_sasa"] == pytest.approx(raw_interface["delta_sasa"], rel=1e-6)

    def test_hydrogen_bond_and_salt_bridge_counts_do_not_depend_on_the_setting(
        self, p53_protonated
    ):
        ignored = compute_interface_metrics(p53_protonated, "B", "A", hydrogens="ignore")
        kept = compute_interface_metrics(p53_protonated, "B", "A", hydrogens="keep")
        for key in ("hbonds", "hbond_energy", "saltbridges", "saltbridge_energy"):
            assert ignored[key] == kept[key], key


class TestPolarApolarPartition:
    """In ignore mode every buried atom of a protein is C, S, N or O."""

    @pytest.mark.parametrize("structure", ["raw", "protonated", "deuterated"])
    def test_polar_plus_apolar_equals_delta_sasa(self, structure, p53_protonated, p53_deuterated):
        path = {"raw": P53_MDM2, "protonated": p53_protonated, "deuterated": p53_deuterated}[
            structure
        ]
        result = compute_interface_metrics(path, "B", "A")
        assert result["polar_area"] + result["apolar_area"] == pytest.approx(
            result["delta_sasa"], rel=1e-6
        )
        assert result["fraction_polar"] == pytest.approx(
            result["polar_area"] / result["delta_sasa"], rel=1e-6
        )

    def test_per_residue_areas_add_up_to_the_total(self, p53_protonated):
        result = compute_interface_metrics(p53_protonated, "B", "A", interface_threshold=0.0)
        residue_total = sum(r["buried_sasa"] for r in result["per_residue"])
        assert residue_total == pytest.approx(result["delta_sasa"], rel=1e-6)

    def test_keep_mode_breaks_the_partition_on_a_protonated_structure(self, p53_protonated):
        result = compute_interface_metrics(p53_protonated, "B", "A", hydrogens="keep")
        assert result["polar_area"] + result["apolar_area"] < result["delta_sasa"] - 100.0


class TestFilterHydrogens:
    @staticmethod
    def _atoms(elements):
        arr = struc.AtomArray(len(elements))
        arr.element = np.array(elements)
        return arr

    def test_ignore_drops_hydrogen_and_deuterium_only(self):
        atoms = self._atoms(["C", "H", "N", "D", "O", "h", "S"])
        kept = filter_hydrogens(atoms, "ignore")
        assert list(kept.element) == ["C", "N", "O", "S"]

    def test_keep_returns_the_input(self):
        atoms = self._atoms(["C", "H"])
        assert filter_hydrogens(atoms, "keep") is atoms

    def test_default_is_ignore(self):
        atoms = self._atoms(["C", "H"])
        assert list(filter_hydrogens(atoms).element) == ["C"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="hydrogens"):
            filter_hydrogens(struc.AtomArray(0), "drop")

    def test_interface_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="hydrogens"):
            compute_interface_metrics(P53_MDM2, "B", "A", hydrogens="drop")

    def test_static_sasa_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="hydrogens"):
            compute_delta_sasa_static(P53_MDM2, "B", "A", hydrogens="drop")


class TestCommandLine:
    @staticmethod
    def _delta_sasa_printed(capsys) -> float:
        for line in capsys.readouterr().out.splitlines():
            if line.strip().startswith("delta_sasa:"):
                return float(line.split(":")[1])
        raise AssertionError("no delta_sasa line in the output")

    def _run(self, monkeypatch, path, *extra):
        argv = ["binding-metrics-interface", "--input", str(path), *extra]
        monkeypatch.setattr("sys.argv", argv)
        interface.main()

    def test_default_ignores_hydrogens(self, monkeypatch, capsys, p53_protonated, raw_interface):
        self._run(monkeypatch, p53_protonated, "--design-chain", "B", "--receptor-chain", "A")
        assert self._delta_sasa_printed(capsys) == pytest.approx(
            raw_interface["delta_sasa"], abs=1e-3
        )

    def test_keep_flag_includes_hydrogens(self, monkeypatch, capsys, p53_protonated, raw_interface):
        self._run(
            monkeypatch,
            p53_protonated,
            "--design-chain",
            "B",
            "--receptor-chain",
            "A",
            "--hydrogens",
            "keep",
        )
        assert self._delta_sasa_printed(capsys) > 1.03 * raw_interface["delta_sasa"]

    def test_unknown_choice_is_rejected(self, monkeypatch, p53_protonated):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, p53_protonated, "--hydrogens", "drop")
