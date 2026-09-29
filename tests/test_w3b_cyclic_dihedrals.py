"""Ring-closing phi/psi/omega of head-to-tail cyclic peptides.

The closing amide bond links the last residue's C to the first residue's N.
Its angles must be scored like every other backbone angle, so a cyclic chain
has as many evaluated residues and peptide bonds as it has residues.
"""

from pathlib import Path

import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdbx as pdbx  # noqa: E402

from binding_metrics.metrics.geometry import (  # noqa: E402
    _load_structure,
    compute_omega_planarity,
    compute_ramachandran,
)

DATA_DIR = Path(__file__).parent.parent / "data"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"  # chain C: 11 residues, D-Ala first
SFTI1 = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"  # chain I: 14 residues, head-to-tail
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"  # chain B: 13 residues, linear
SOMATOSTATIN = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"  # 12 residues, side-chain lactam


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.name} not bundled")
    return path


class TestCyclosporin:
    def test_every_residue_is_evaluated(self):
        rama = compute_ramachandran(_need(CYCLOSPORIN), chain="C")
        assert rama["cyclic_closure_detected"] is True
        assert rama["cyclic_closure_evaluated"] is True
        assert rama["n_residues_evaluated"] == 11

    def test_d_alanine_at_position_one_is_scored(self):
        rama = compute_ramachandran(_need(CYCLOSPORIN), chain="C")
        assert rama["n_d_residues"] == 1
        first = rama["per_residue"][0]
        assert (first["res_id"], first["res_name"], first["is_d_aa"]) == (1, "DAL", True)
        # D-residue phi is positive; it is judged on the mirrored plot.
        assert 60.0 < first["phi"] < 120.0
        assert first["region"] == "favoured"

    def test_last_residue_psi_comes_from_the_closing_bond(self):
        rama = compute_ramachandran(_need(CYCLOSPORIN), chain="C")
        last = rama["per_residue"][-1]
        assert (last["res_id"], last["res_name"]) == (11, "ALA")
        assert last["psi"] == pytest.approx(159.43, abs=0.01)

    def test_closing_peptide_bond_is_evaluated(self):
        omega = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        assert omega["cyclic_closure_evaluated"] is True
        assert omega["n_bonds_evaluated"] == 11
        closing = omega["per_residue"][-1]
        assert closing["res_id"] == 11
        # trans within 15 degrees: the closing amide is planar in this structure
        assert closing["omega"] == pytest.approx(173.61, abs=0.01)
        assert omega["omega_outlier_count"] == 0

    def test_sequential_angles_are_untouched(self):
        """Dropping the closing bond reproduces the values from before the fix."""
        omega = compute_omega_planarity(_need(CYCLOSPORIN), chain="C")
        deviations = [r["deviation"] for r in omega["per_residue"][:-1]]
        assert sum(deviations) / len(deviations) == pytest.approx(2.992090, abs=1e-5)


class TestSfti1:
    def test_all_fourteen_residues_and_bonds_are_evaluated(self):
        rama = compute_ramachandran(_need(SFTI1), chain="I")
        omega = compute_omega_planarity(_need(SFTI1), chain="I")
        assert rama["cyclic_closure_evaluated"] is True
        assert rama["n_residues_evaluated"] == 14
        assert omega["n_bonds_evaluated"] == 14

    def test_closing_bond_is_a_slightly_twisted_trans_amide(self):
        omega = compute_omega_planarity(_need(SFTI1), chain="I")
        closing = omega["per_residue"][-1]
        assert (closing["res_id"], closing["res_name"]) == (14, "ASP")
        assert closing["omega"] == pytest.approx(-164.28, abs=0.01)
        assert closing["is_outlier"] is True  # 15.7 degrees from trans, cut-off is 15
        assert omega["omega_cis_count"] == 1  # the cis Pro is the only cis bond


class TestLinearPeptidesAreUnchanged:
    def test_p53_terminal_residues_stay_unscored(self):
        rama = compute_ramachandran(_need(P53), chain="B")
        omega = compute_omega_planarity(_need(P53), chain="B")
        assert rama["cyclic_closure_detected"] is False
        assert rama["cyclic_closure_evaluated"] is False
        assert rama["n_residues_evaluated"] == 11  # 13 residues minus the two termini
        assert omega["n_bonds_evaluated"] == 12
        assert omega["omega_mean_dev"] == pytest.approx(1.28, abs=0.01)

    def test_side_chain_lactam_is_not_a_backbone_closure(self):
        rama = compute_ramachandran(_need(SOMATOSTATIN), chain="A")
        omega = compute_omega_planarity(_need(SOMATOSTATIN), chain="A")
        assert rama["cyclic_closure_evaluated"] is False
        assert rama["n_residues_evaluated"] == 10
        assert omega["n_bonds_evaluated"] == 11


def _write_rotated_ring(source: Path, chain: str, shift: int, out: Path) -> Path:
    """Write the chain with its residues cyclically re-ordered by ``shift``.

    Coordinates and residue numbers are untouched, only the order in the file
    changes, so a different residue is "first" and the ring closes at another
    bond. Whatever the metrics say for a residue must not depend on that order.
    """
    atoms = _load_structure(source)
    peptide = atoms[(atoms.chain_id == chain) & struc.filter_amino_acids(atoms)]
    bounds = [*struc.get_residue_starts(peptide), peptide.array_length()]
    residues = [peptide[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    order = residues[shift:] + residues[:shift]
    rotated = order[0]
    for residue in order[1:]:
        rotated = rotated + residue
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, rotated)
    cif.write(str(out))
    return out


class TestRingClosureAgreesWithSequentialAngles:
    """The closing angles equal what biotite gives when that bond is not the closure."""

    @pytest.mark.parametrize("shift", [1, 5, 10])
    def test_cyclosporin_angles_do_not_depend_on_where_the_ring_is_opened(self, shift, tmp_path):
        _need(CYCLOSPORIN)
        rotated = _write_rotated_ring(CYCLOSPORIN, "C", shift, tmp_path / "rotated.cif")

        reference_rama = compute_ramachandran(CYCLOSPORIN, chain="C")
        reference_omega = compute_omega_planarity(CYCLOSPORIN, chain="C")
        rama = compute_ramachandran(rotated, chain="C")
        omega = compute_omega_planarity(rotated, chain="C")

        assert rama["cyclic_closure_evaluated"] is True
        assert rama["n_residues_evaluated"] == 11
        assert omega["n_bonds_evaluated"] == 11
        expected = {r["res_id"]: r for r in reference_rama["per_residue"]}
        for entry in rama["per_residue"]:
            assert entry["phi"] == pytest.approx(expected[entry["res_id"]]["phi"], abs=1e-3)
            assert entry["psi"] == pytest.approx(expected[entry["res_id"]]["psi"], abs=1e-3)
        expected_omega = {r["res_id"]: r["omega"] for r in reference_omega["per_residue"]}
        for entry in omega["per_residue"]:
            assert entry["omega"] == pytest.approx(expected_omega[entry["res_id"]], abs=1e-3)
        assert rama["n_d_residues"] == reference_rama["n_d_residues"] == 1
