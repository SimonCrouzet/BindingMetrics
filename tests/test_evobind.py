"""Behavioural tests for ``binding_metrics.metrics.evobind``.

Structures are built in-test from a few Cα/Cβ atoms per residue and written to
``tmp_path`` as PDB files, so every expected value follows from the geometry
constructed here (no bundled data, no network).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from binding_metrics.metrics.evobind import (  # noqa: E402
    _adversarial_from_atoms,
    _auto_interface_mask,
    _cb_atoms,
    _load_atoms,
    _pairwise_min_dists,
    _per_residue_plddt,
    _score_from_atoms,
    compute_evobind_adversarial_check,
    compute_evobind_score,
)

# ---------------------------------------------------------------------------
# Synthetic structure builders
# ---------------------------------------------------------------------------


def _helix_ca(n: int, radius: float = 2.3, rise: float = 1.5) -> np.ndarray:
    """Cα trace of an ideal-ish α-helix along z (non-collinear, so Kabsch is well posed)."""
    theta = np.deg2rad(100.0) * np.arange(n)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), rise * np.arange(n)])


def _residue_atoms(chain_id, res_id, res_name, ca, cb=None):
    """Cα (and Cβ when given) atoms of one residue."""
    atoms = [
        struc.Atom(
            ca, chain_id=chain_id, res_id=res_id, res_name=res_name, atom_name="CA", element="C"
        )
    ]
    if cb is not None:
        atoms.append(
            struc.Atom(
                cb, chain_id=chain_id, res_id=res_id, res_name=res_name, atom_name="CB", element="C"
            )
        )
    return atoms


def _chain(chain_id, ca_coords, cb_coords=None, first_res_id=1, res_names=None):
    """AtomArray for one chain; ``cb_coords=None`` makes every residue glycine-like (Cα only)."""
    atoms = []
    for i, ca in enumerate(ca_coords):
        cb = None if cb_coords is None else cb_coords[i]
        name = "GLY" if cb is None else "ALA"
        if res_names is not None:
            name = res_names[i]
        atoms.extend(_residue_atoms(chain_id, first_res_id + i, name, ca, cb))
    return struc.array(atoms)


def _write(path, *chains):
    """Write the concatenated chains as a PDB file and return the path."""
    atoms = chains[0]
    for extra in chains[1:]:
        atoms = atoms + extra
    pdb = pdb_io.PDBFile()
    pdb_io.set_structure(pdb, atoms)
    pdb.write(str(path))
    return path


def _line_complex(tmp_path, name="line.pdb"):
    """Receptor on the x axis (5 residues), binder 6 Å above residues 2-4.

    Receptor Cα at x = 3.8*i (res_id i+1), Cβ 1.5 Å above (y). Binder Cβ at
    y = 7.5, so each binder Cβ sits exactly 6.0 Å above the Cβ of receptor
    residues 2, 3 and 4; the other receptor Cβ atoms are 7.1 Å away.
    """
    rec_ca = np.array([[3.8 * i, 0.0, 0.0] for i in range(5)])
    rec_cb = rec_ca + np.array([0.0, 1.5, 0.0])
    pep_ca = np.array([[3.8 * (j + 1), 6.0, 0.0] for j in range(3)])
    pep_cb = pep_ca + np.array([0.0, 1.5, 0.0])
    path = _write(tmp_path / name, _chain("A", rec_ca, rec_cb), _chain("B", pep_ca, pep_cb))
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_pairwise_min_dists_matches_cdist(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(7, 3)) * 5
        b = rng.normal(size=(11, 3)) * 5
        np.testing.assert_allclose(_pairwise_min_dists(a, b), cdist(a, b).min(axis=1))

    def test_pairwise_min_dists_known_value(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
        assert _pairwise_min_dists(a, b)[0] == pytest.approx(5.0)

    def test_auto_interface_mask_uses_strict_cutoff(self):
        rec = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        pep = np.array([[0.0, 5.0, 0.0]])
        assert _auto_interface_mask(rec, pep, 8.0).tolist() == [True, False]
        # a residue exactly at the cutoff is not inside it
        assert _auto_interface_mask(rec, pep, 5.0).tolist() == [False, False]

    def test_cb_atoms_prefers_cb_and_falls_back_to_ca(self):
        ca = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]])
        atoms = struc.array(
            _residue_atoms("A", 1, "ALA", ca[0], ca[0] + [0, 1.5, 0])
            + _residue_atoms("A", 2, "GLY", ca[1])
            + _residue_atoms("A", 3, "ALA", ca[2], ca[2] + [0, 1.5, 0])
        )
        picked = _cb_atoms(atoms, "A")
        assert picked.atom_name.tolist() == ["CB", "CA", "CB"]
        assert picked.res_id.tolist() == [1, 2, 3]

    def test_cb_atoms_unknown_chain_is_empty(self):
        atoms = _chain("A", np.zeros((2, 3)) + [[0, 0, 0], [3.8, 0, 0]])
        assert _cb_atoms(atoms, "Z").array_length() == 0

    def test_per_residue_plddt_averages_atoms_of_the_chain(self):
        atoms = _chain(
            "A", np.array([[0.0, 0, 0], [3.8, 0, 0]]), np.array([[0.0, 1.5, 0], [3.8, 1.5, 0]])
        )
        atoms = atoms + _chain("B", np.array([[0.0, 9, 0]]), np.array([[0.0, 10.5, 0]]))
        plddt = np.array([10.0, 30.0, 50.0, 70.0, 99.0, 99.0])
        np.testing.assert_allclose(_per_residue_plddt(plddt, atoms, "A"), [20.0, 60.0])
        np.testing.assert_allclose(_per_residue_plddt(plddt, atoms, "B"), [99.0])


# ---------------------------------------------------------------------------
# compute_evobind_score
# ---------------------------------------------------------------------------


class TestEvobindScore:
    def test_auto_interface_distances_and_score(self, tmp_path):
        path = _line_complex(tmp_path)
        # receptor atoms (10) come first in the file, then the binder (6)
        plddt = np.concatenate([np.full(10, 50.0), np.full(6, 80.0)])

        res = compute_evobind_score(
            path, plddt, binder_chain="B", receptor_chain="A", interface_cutoff_angstrom=6.5
        )

        assert res["n_interface_receptor_residues"] == 3
        assert res["if_dist_pep_to_rec"] == pytest.approx(6.0, abs=1e-2)
        assert res["if_dist_rec_to_pep"] == pytest.approx(6.0, abs=1e-2)
        assert res["if_dist_symmetric"] == pytest.approx(6.0, abs=1e-2)
        # pLDDT is taken from the binder chain only
        assert res["mean_plddt_binder"] == pytest.approx(80.0)
        assert res["evobind_score"] == pytest.approx(6.0 / 0.8, abs=1e-2)

    def test_default_cutoff_pulls_in_the_whole_receptor_here(self, tmp_path):
        # with the 8 A default, all five receptor Cβ (6.0 and 7.1 A away) are interface
        path = _line_complex(tmp_path)
        res = compute_evobind_score(path, None, binder_chain="B", receptor_chain="A")
        assert res["n_interface_receptor_residues"] == 5

    def test_explicit_interface_residues_override_auto_detection(self, tmp_path):
        path = _line_complex(tmp_path)
        res = compute_evobind_score(
            path,
            None,
            binder_chain="B",
            receptor_chain="A",
            receptor_interface_residues=[2, 3, 4],
        )
        assert res["n_interface_receptor_residues"] == 3
        assert res["if_dist_rec_to_pep"] == pytest.approx(6.0, abs=1e-2)

        far = compute_evobind_score(
            path, None, binder_chain="B", receptor_chain="A", receptor_interface_residues=[1]
        )
        assert far["n_interface_receptor_residues"] == 1
        # res 1 at x=0: nearest binder Cβ is at x=3.8, 6 A up -> sqrt(3.8^2 + 6^2)
        assert far["if_dist_rec_to_pep"] == pytest.approx(np.hypot(3.8, 6.0), abs=1e-2)

    def test_no_residue_in_cutoff_falls_back_to_full_receptor(self, tmp_path):
        path = _line_complex(tmp_path)
        res = compute_evobind_score(
            path, None, binder_chain="B", receptor_chain="A", interface_cutoff_angstrom=1.0
        )
        assert res["n_interface_receptor_residues"] == 5

    def test_fallback_to_full_receptor_is_flagged(self, tmp_path):
        path = _line_complex(tmp_path)
        auto = compute_evobind_score(path, None, "B", "A", interface_cutoff_angstrom=6.5)
        assert auto["interface_fallback_used"] is False
        tight = compute_evobind_score(path, None, "B", "A", interface_cutoff_angstrom=1.0)
        assert tight["interface_fallback_used"] is True
        explicit = compute_evobind_score(
            path, None, "B", "A", receptor_interface_residues=[2, 3, 4]
        )
        assert explicit["interface_fallback_used"] is False

    def test_explicit_residues_absent_from_the_receptor_raise(self, tmp_path):
        path = _line_complex(tmp_path)
        with pytest.raises(ValueError, match="receptor_interface_residues"):
            compute_evobind_score(path, None, "B", "A", receptor_interface_residues=[900, 901])

    def test_without_plddt_only_distances_are_reported(self, tmp_path):
        path = _line_complex(tmp_path)
        res = compute_evobind_score(path, None, binder_chain="B", receptor_chain="A")
        assert res["mean_plddt_binder"] is None
        assert res["evobind_score"] is None
        assert np.isfinite(res["if_dist_pep_to_rec"])

    def test_zero_plddt_gives_no_score(self, tmp_path):
        path = _line_complex(tmp_path)
        res = compute_evobind_score(path, np.zeros(16), binder_chain="B", receptor_chain="A")
        assert res["mean_plddt_binder"] == 0.0
        assert res["evobind_score"] is None
        assert res["reason"] == "mean binder pLDDT is zero or not finite"

    def test_nan_plddt_gives_no_score_with_a_reason(self, tmp_path):
        path = _line_complex(tmp_path)
        res = compute_evobind_score(path, np.full(16, np.nan), "B", "A")
        assert res["evobind_score"] is None
        assert "reason" in res

    def test_no_reason_when_the_score_is_computed_or_not_requested(self, tmp_path):
        path = _line_complex(tmp_path)
        assert "reason" not in compute_evobind_score(path, np.full(16, 80.0), "B", "A")
        assert "reason" not in compute_evobind_score(path, None, "B", "A")

    def test_score_is_inversely_proportional_to_plddt(self, tmp_path):
        path = _line_complex(tmp_path)
        low = compute_evobind_score(path, np.full(16, 50.0), "B", "A")
        high = compute_evobind_score(path, np.full(16, 100.0), "B", "A")
        assert low["evobind_score"] == pytest.approx(2.0 * high["evobind_score"])

    def test_glycine_binder_uses_ca(self, tmp_path):
        rec_ca = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        rec_cb = rec_ca + [0.0, 1.5, 0.0]
        pep_ca = np.array([[0.0, 4.0, 0.0]])
        path = _write(tmp_path / "gly.pdb", _chain("A", rec_ca, rec_cb), _chain("B", pep_ca, None))
        res = compute_evobind_score(path, None, "B", "A")
        # binder Cα (y=4) to receptor Cβ (y=1.5) directly below: 2.5 A
        assert res["if_dist_pep_to_rec"] == pytest.approx(2.5, abs=1e-2)

    @pytest.mark.parametrize("missing", ["binder", "receptor"])
    def test_missing_chain_raises_value_error(self, tmp_path, missing):
        path = _line_complex(tmp_path)
        kwargs = {"binder_chain": "B", "receptor_chain": "A"}
        kwargs[f"{missing}_chain"] = "Z"
        with pytest.raises(ValueError, match=f"{missing}"):
            compute_evobind_score(path, None, **kwargs)


# ---------------------------------------------------------------------------
# compute_evobind_adversarial_check
# ---------------------------------------------------------------------------


def _helix_complex(
    tmp_path,
    name,
    n_rec=12,
    first_rec_res=1,
    first_pep_res=1,
    pep_shift=(0, 0, 0),
    rec_names=None,
    pep_names=None,
):
    """α-helix receptor (chain A) with a short extended binder (chain B) beside it.

    All coordinates depend only on the residue index, so two calls with
    different ``first_*_res`` describe the same geometry under different numbering.
    ``rec_names`` / ``pep_names`` set the residue names (default ALA).
    """
    rec_ca = _helix_ca(n_rec)
    radial = rec_ca[:, :2] / np.linalg.norm(rec_ca[:, :2], axis=1, keepdims=True)
    rec_cb = rec_ca + 1.5 * np.column_stack([radial, np.zeros(n_rec)])
    pep_ca = np.array([[9.0, 0.0, 2.0 + 3.8 * j] for j in range(4)]) + np.asarray(pep_shift)
    pep_cb = pep_ca + [-1.5, 0.0, 0.0]
    return _write(
        tmp_path / name,
        _chain("A", rec_ca, rec_cb, first_res_id=first_rec_res, res_names=rec_names),
        _chain("B", pep_ca, pep_cb, first_res_id=first_pep_res, res_names=pep_names),
    )


def _rigid_transform(path_in, path_out, angle_deg=70.0, translation=(15.0, -8.0, 4.0)):
    """Rewrite ``path_in`` rotated about z and translated, as a new PDB."""
    atoms = pdb_io.get_structure(pdb_io.PDBFile.read(str(path_in)), model=1)
    moved = struc.rotate(atoms, [0.0, 0.0, np.deg2rad(angle_deg)])
    moved = struc.translate(moved, translation)
    pdb = pdb_io.PDBFile()
    pdb_io.set_structure(pdb, moved)
    pdb.write(str(path_out))
    return path_out


class TestAdversarialCheck:
    def test_identical_structures_have_zero_delta_com(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")
        res = compute_evobind_adversarial_check(
            a, b, "B", "A", afm_plddt_per_atom=np.full(32, 90.0)
        )
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
        assert res["n_superposition_residues"] == 12
        assert res["evobind_adversarial_score"] == pytest.approx(0.0, abs=0.05)

    def test_rigid_motion_of_whole_complex_is_removed_by_superposition(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _rigid_transform(a, tmp_path / "afm_moved.pdb")
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=0.05)

    def test_shifted_binder_gives_known_com_displacement(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm_shift.pdb", pep_shift=(0.0, 3.0, 4.0))
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)

    def test_shift_is_measured_in_the_receptor_frame(self, tmp_path):
        # move the binder by 5 A AND the whole AFM complex rigidly: still 5 A
        a = _helix_complex(tmp_path, "design.pdb")
        shifted = _helix_complex(tmp_path, "afm_shift.pdb", pep_shift=(0.0, 3.0, 4.0))
        b = _rigid_transform(shifted, tmp_path / "afm_shift_moved.pdb")
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=0.05)

    def test_adversarial_score_formula(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm_shift.pdb", pep_shift=(0.0, 3.0, 4.0))
        # receptor atoms (24) first, then binder (8); binder pLDDT 80
        plddt = np.concatenate([np.full(24, 40.0), np.full(8, 80.0)])
        res = compute_evobind_adversarial_check(a, b, "B", "A", afm_plddt_per_atom=plddt)
        assert res["afm_mean_plddt_binder"] == pytest.approx(80.0)
        expected = res["afm_mean_if_dist"] * (100.0 / 80.0) * res["delta_com_angstrom"]
        assert res["evobind_adversarial_score"] == pytest.approx(expected)
        assert res["afm_mean_if_dist"] == pytest.approx(
            0.5 * (res["afm_if_dist_pep_to_rec"] + res["afm_if_dist_rec_to_pep"])
        )

    def test_without_plddt_only_geometry_is_reported(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["afm_mean_plddt_binder"] is None
        assert res["evobind_adversarial_score"] is None
        assert "reason" not in res

    def test_zero_plddt_gives_no_score_with_a_reason(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")
        res = compute_evobind_adversarial_check(a, b, "B", "A", afm_plddt_per_atom=np.zeros(32))
        assert res["evobind_adversarial_score"] is None
        assert "zero or not finite" in res["reason"]

    def test_partial_receptor_overlap_counts_common_residues(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", n_rec=10)
        # same helix, two more residues (11-12) that exist only in the AFM model
        b = _helix_complex(tmp_path, "afm_long.pdb", n_rec=12)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["n_superposition_residues"] == 10
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_renumbered_prediction_is_paired_by_position(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", first_rec_res=101, first_pep_res=201)
        b = _helix_complex(tmp_path, "afm.pdb", first_rec_res=1, first_pep_res=1)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_too_few_receptor_atoms_raises(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", n_rec=2)
        b = _helix_complex(tmp_path, "afm.pdb", n_rec=2)
        with pytest.raises(ValueError, match="Fewer than 3 receptor"):
            compute_evobind_adversarial_check(a, b, "B", "A")

    def test_missing_binder_chain_raises(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")
        with pytest.raises(ValueError, match="binder chain 'Z'"):
            compute_evobind_adversarial_check(a, b, "Z", "A")

    def test_interface_fallback_is_flagged(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")
        normal = compute_evobind_adversarial_check(a, b, "B", "A")
        assert normal["interface_fallback_used"] is False
        # no receptor residue within 0.5 A of the binder in the design
        tight = compute_evobind_adversarial_check(a, b, "B", "A", interface_cutoff_angstrom=0.5)
        assert tight["interface_fallback_used"] is True

    def test_interface_residues_missing_from_the_afm_model_are_flagged(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", first_rec_res=101, first_pep_res=201)
        b = _helix_complex(tmp_path, "afm.pdb", first_rec_res=1, first_pep_res=1)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["interface_fallback_used"] is True

    def test_superposition_atom_count_with_matching_numbering(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", n_rec=10)
        b = _helix_complex(tmp_path, "afm.pdb", n_rec=12)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["receptor_pairing"] == "residue_number"
        assert res["binder_pairing"] == "residue_number"
        assert res["n_superposition_residues"] == 10
        assert res["n_superposition_atoms"] == 10

    def test_superposition_atom_count_when_pairing_falls_back_to_position(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", first_rec_res=101, first_pep_res=201)
        b = _helix_complex(tmp_path, "afm.pdb")
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["receptor_pairing"] == "position"
        assert res["binder_pairing"] == "position"
        # the residue-number count keeps its meaning (numbers shared by both models)
        assert res["n_superposition_residues"] < 3
        # the atom count is what the superposition actually used
        assert res["n_superposition_atoms"] == 12

    def test_matching_residue_names_have_zero_mismatch(self, tmp_path):
        names = ["ALA", "GLY", "SER", "LEU", "VAL", "ILE", "PHE", "TYR", "LYS", "ARG", "GLU", "ASP"]
        a = _helix_complex(tmp_path, "design.pdb", rec_names=names)
        b = _helix_complex(tmp_path, "afm.pdb", rec_names=names)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["receptor_resname_mismatch_fraction"] == 0.0
        assert res["binder_resname_mismatch_fraction"] == 0.0

    def test_histidine_and_cysteine_variants_are_not_mismatches(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", rec_names=["HIS", "CYS"] * 6)
        b = _helix_complex(tmp_path, "afm.pdb", rec_names=["HIE", "CYX"] * 6)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["receptor_resname_mismatch_fraction"] == 0.0

    def test_a_few_mutations_are_reported_but_accepted(self, tmp_path):
        names = ["ALA", "GLY", "SER", "LEU", "VAL", "ILE", "PHE", "TYR", "LYS", "ARG", "GLU", "ASP"]
        mutated = list(names)
        mutated[4] = "TRP"
        a = _helix_complex(tmp_path, "design.pdb", rec_names=names)
        b = _helix_complex(tmp_path, "afm.pdb", rec_names=mutated)
        res = compute_evobind_adversarial_check(a, b, "B", "A")
        assert res["receptor_resname_mismatch_fraction"] == pytest.approx(1 / 12)
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_offset_numbering_with_different_residues_raises(self, tmp_path):
        # A design numbered 106-117 of a 17-residue receptor against a prediction of the
        # same protein renumbered from 1: numbers do not overlap, so residues are paired by
        # position, and position k of one model is residue k+5 of the other.
        sequence = ["ALA", "GLY", "SER", "LEU", "VAL", "ILE", "PHE", "TYR", "LYS", "ARG"]
        sequence += ["GLU", "ASP", "ASN", "GLN", "HIS", "TRP", "PRO"]
        a = _helix_complex(
            tmp_path, "design.pdb", first_rec_res=106, first_pep_res=1, rec_names=sequence[5:17]
        )
        b = _helix_complex(tmp_path, "afm.pdb", first_rec_res=1, rec_names=sequence[0:12])
        with pytest.raises(ValueError, match="different residue names"):
            compute_evobind_adversarial_check(a, b, "B", "A")

    def test_mismatch_limit_is_adjustable(self, tmp_path):
        sequence = ["ALA", "GLY", "SER", "LEU", "VAL", "ILE", "PHE", "TYR", "LYS", "ARG"]
        sequence += ["GLU", "ASP", "ASN", "GLN", "HIS", "TRP", "PRO"]
        a = _helix_complex(tmp_path, "design.pdb", first_rec_res=106, rec_names=sequence[5:17])
        b = _helix_complex(tmp_path, "afm.pdb", first_rec_res=1, rec_names=sequence[0:12])
        res = compute_evobind_adversarial_check(a, b, "B", "A", max_resname_mismatch_fraction=1.0)
        assert res["receptor_resname_mismatch_fraction"] == pytest.approx(1.0)

    def test_binder_with_different_residues_raises(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb", pep_names=["ALA", "GLY", "SER", "LEU"])
        b = _helix_complex(tmp_path, "afm.pdb", pep_names=["TRP", "PRO", "HIS", "GLN"])
        with pytest.raises(ValueError, match="binder residues"):
            compute_evobind_adversarial_check(a, b, "B", "A")


# ---------------------------------------------------------------------------
# The split at the load step: the path functions only load, then call these
# ---------------------------------------------------------------------------


class TestSplitAtTheLoadStep:
    def test_score_from_atoms_equals_the_path_function(self, tmp_path):
        path = _line_complex(tmp_path)
        plddt = np.concatenate([np.full(10, 50.0), np.full(6, 80.0)])
        from_path = compute_evobind_score(path, plddt, "B", "A", interface_cutoff_angstrom=6.5)
        from_atoms = _score_from_atoms(_load_atoms(path), plddt, "B", "A", None, 6.5)
        assert from_atoms == from_path

    def test_adversarial_from_atoms_equals_the_path_function(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm_shift.pdb", pep_shift=(0.0, 3.0, 4.0))
        plddt = np.concatenate([np.full(24, 40.0), np.full(8, 80.0)])
        from_path = compute_evobind_adversarial_check(a, b, "B", "A", afm_plddt_per_atom=plddt)
        from_atoms = _adversarial_from_atoms(
            _load_atoms(a), _load_atoms(b), plddt, "B", "A", 8.0, 0.5
        )
        assert from_atoms == from_path
        assert from_atoms["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)

    def test_the_adversary_label_names_the_second_structure_in_messages(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb", n_rec=2)
        design, second = _load_atoms(a), _load_atoms(b)
        with pytest.raises(ValueError, match=r"design 12, AFM 2"):
            _adversarial_from_atoms(design, second, None, "B", "A", 8.0, 0.5)
        with pytest.raises(ValueError, match=r"design 12, adversary 2"):
            _adversarial_from_atoms(
                design, second, None, "B", "A", 8.0, 0.5, adversary_label="adversary"
            )


# ---------------------------------------------------------------------------
# The per-residue pLDDT helper is the shared one (issue #92)
# ---------------------------------------------------------------------------


class TestSharedPlddtHelper:
    def test_wrong_length_array_is_a_value_error_naming_both_lengths(self):
        atoms = _chain("A", np.array([[0.0, 0, 0], [3.8, 0, 0]]))
        with pytest.raises(ValueError, match=r"plddt_per_atom length \(3\) != atom count .* \(2\)"):
            _per_residue_plddt(np.array([50.0, 60.0, 70.0]), atoms, "A")

    def test_score_with_a_wrong_length_array_raises_value_error(self, tmp_path):
        path = _line_complex(tmp_path)  # 16 atoms
        with pytest.raises(ValueError, match="plddt_per_atom length"):
            compute_evobind_score(path, np.full(6, 80.0), "B", "A")
        with pytest.raises(ValueError, match="plddt_per_atom length"):
            compute_evobind_score(path, np.full(20, 80.0), "B", "A")

    def test_adversarial_check_with_a_wrong_length_array_raises_value_error(self, tmp_path):
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")  # 32 atoms
        with pytest.raises(ValueError, match="plddt_per_atom length"):
            compute_evobind_adversarial_check(a, b, "B", "A", afm_plddt_per_atom=np.full(8, 80.0))

    def test_insertion_codes_make_separate_residues(self):
        # residue 1 and residue 1A are two residues: their pLDDT are not averaged together
        atoms = struc.array(
            [
                struc.Atom(
                    [0.0, 0, 0],
                    chain_id="B",
                    res_id=1,
                    ins_code="",
                    res_name="ALA",
                    atom_name="CA",
                    element="C",
                ),
                struc.Atom(
                    [3.8, 0, 0],
                    chain_id="B",
                    res_id=1,
                    ins_code="A",
                    res_name="ALA",
                    atom_name="CA",
                    element="C",
                ),
                struc.Atom(
                    [7.6, 0, 0],
                    chain_id="B",
                    res_id=2,
                    ins_code="",
                    res_name="ALA",
                    atom_name="CA",
                    element="C",
                ),
            ]
        )
        plddt = np.array([10.0, 30.0, 50.0])
        np.testing.assert_allclose(_per_residue_plddt(plddt, atoms, "B"), [10.0, 30.0, 50.0])

    def test_a_list_of_plddt_values_is_accepted(self, tmp_path):
        # a list used to fail with "only integer scalar arrays can be converted to a scalar index"
        path = _line_complex(tmp_path)
        as_array = np.concatenate([np.full(10, 50.0), np.full(6, 80.0)])
        from_list = compute_evobind_score(path, as_array.tolist(), "B", "A")
        assert from_list["mean_plddt_binder"] == pytest.approx(80.0)
        assert (
            from_list["evobind_score"]
            == compute_evobind_score(path, as_array, "B", "A")["evobind_score"]
        )
        a = _helix_complex(tmp_path, "design.pdb")
        b = _helix_complex(tmp_path, "afm.pdb")
        plddt = np.concatenate([np.full(24, 40.0), np.full(8, 80.0)])
        res = compute_evobind_adversarial_check(a, b, "B", "A", afm_plddt_per_atom=list(plddt))
        assert res["afm_mean_plddt_binder"] == pytest.approx(80.0)

    def test_a_list_of_the_wrong_length_is_still_a_value_error(self):
        atoms = _chain("A", np.array([[0.0, 0, 0], [3.8, 0, 0]]))
        with pytest.raises(ValueError, match="plddt_per_atom length"):
            _per_residue_plddt([50.0, 60.0, 70.0], atoms, "A")


# ---------------------------------------------------------------------------
# Residues that differ only by insertion code are two residues (issue #102)
# ---------------------------------------------------------------------------


def _insertion_atom(x, y, chain_id, res_id, ins_code, atom_name):
    return struc.Atom(
        [x, y, 0.0],
        chain_id=chain_id,
        res_id=res_id,
        ins_code=ins_code,
        res_name="ALA",
        atom_name=atom_name,
        element="C",
    )


def _insertion_code_complex(tmp_path):
    """Receptor A of two residues; binder B of residues 1, 1A and 2, the 1A one far away.

    Binder Cβ at (0, 7.5), (3.8, 13.5) [residue 1A] and (7.6, 7.5); receptor Cβ at (0, 1.5)
    and (3.8, 1.5). The nearest receptor Cβ is 6.0, 12.0 and hypot(3.8, 6.0) angstrom away.
    """
    atoms = struc.array(
        [
            _insertion_atom(0.0, 0.0, "A", 1, "", "CA"),
            _insertion_atom(0.0, 1.5, "A", 1, "", "CB"),
            _insertion_atom(3.8, 0.0, "A", 2, "", "CA"),
            _insertion_atom(3.8, 1.5, "A", 2, "", "CB"),
            _insertion_atom(0.0, 6.0, "B", 1, "", "CA"),
            _insertion_atom(0.0, 7.5, "B", 1, "", "CB"),
            _insertion_atom(3.8, 12.0, "B", 1, "A", "CA"),
            _insertion_atom(3.8, 13.5, "B", 1, "A", "CB"),
            _insertion_atom(7.6, 6.0, "B", 2, "", "CA"),
            _insertion_atom(7.6, 7.5, "B", 2, "", "CB"),
        ]
    )
    return _write(tmp_path / "insertion.pdb", atoms), atoms


class TestInsertionCodes:
    def test_cb_atoms_keeps_a_residue_that_differs_only_by_insertion_code(self, tmp_path):
        _, atoms = _insertion_code_complex(tmp_path)
        picked = _cb_atoms(atoms, "B")
        assert picked.array_length() == 3
        assert picked.res_id.tolist() == [1, 1, 2]
        assert picked.ins_code.tolist() == ["", "A", ""]
        assert picked.atom_name.tolist() == ["CB", "CB", "CB"]
        np.testing.assert_allclose(picked.coord[:, 0], [0.0, 3.8, 7.6])

    def test_cb_atoms_lists_the_residues_in_residue_number_then_insertion_code_order(self):
        # written out of order: 2, 1A, 1
        atoms = struc.array(
            [
                _insertion_atom(7.6, 0.0, "B", 2, "", "CA"),
                _insertion_atom(3.8, 0.0, "B", 1, "A", "CA"),
                _insertion_atom(0.0, 0.0, "B", 1, "", "CA"),
            ]
        )
        picked = _cb_atoms(atoms, "B")
        assert list(zip(picked.res_id.tolist(), picked.ins_code.tolist())) == [
            (1, ""),
            (1, "A"),
            (2, ""),
        ]

    def test_the_score_counts_the_insertion_code_residue(self, tmp_path):
        path, _ = _insertion_code_complex(tmp_path)
        res = compute_evobind_score(path, None, "B", "A")
        assert res["n_interface_receptor_residues"] == 2
        expected = (6.0 + 12.0 + np.hypot(3.8, 6.0)) / 3.0  # was (6.0 + hypot) / 2 without 1A
        assert res["if_dist_pep_to_rec"] == pytest.approx(expected, abs=1e-2)


class TestAdversarialPairingWithInsertionCodes:
    """Issue #103: residues are paired by residue number AND insertion code."""

    @staticmethod
    def _renumbered(tmp_path, name, chain, from_res, to_res, to_ins, n_rec=12):
        """The helix complex with one residue given another number and insertion code."""
        atoms = _load_atoms(_helix_complex(tmp_path, name, n_rec=n_rec))
        moved = (atoms.chain_id == chain) & (atoms.res_id == from_res)
        atoms.res_id[moved] = to_res
        atoms.ins_code[moved] = to_ins
        return _write(tmp_path / name, atoms)

    def test_a_receptor_residue_with_an_insertion_code_is_not_matched_by_its_number(self, tmp_path):
        # residue 5 of the prediction is called 4A: the design has residues 4 and 5, the
        # prediction 4 and 4A, so the two share 11 residues and 4A is not paired with 4
        design = _helix_complex(tmp_path, "design.pdb")
        second = self._renumbered(tmp_path, "afm.pdb", "A", 5, 4, "A")
        res = compute_evobind_adversarial_check(design, second, "B", "A")
        assert res["receptor_pairing"] == "residue_number"
        assert res["n_superposition_residues"] == 11
        assert res["n_superposition_atoms"] == 11
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_a_binder_residue_with_an_insertion_code_is_not_matched_by_its_number(self, tmp_path):
        design = _helix_complex(tmp_path, "design.pdb")
        second = self._renumbered(tmp_path, "afm.pdb", "B", 3, 2, "A")
        res = compute_evobind_adversarial_check(design, second, "B", "A")
        assert res["binder_pairing"] == "residue_number"
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_the_interface_is_mapped_by_number_and_insertion_code(self, tmp_path):
        # the prediction has a residue 5A where the design has residue 5: the design interface
        # residue 5 has no partner, so the prediction interface holds fewer residues
        design = _helix_complex(tmp_path, "design.pdb")
        second = self._renumbered(tmp_path, "afm.pdb", "A", 5, 5, "A")
        with_code = compute_evobind_adversarial_check(design, second, "B", "A")
        plain = compute_evobind_adversarial_check(
            design, _helix_complex(tmp_path, "plain.pdb"), "B", "A"
        )
        assert with_code["interface_fallback_used"] is False
        assert with_code["afm_if_dist_rec_to_pep"] != pytest.approx(plain["afm_if_dist_rec_to_pep"])

    @pytest.mark.parametrize(
        "chain, pairing_key", [("A", "receptor_pairing"), ("B", "binder_pairing")]
    )
    def test_a_repeated_residue_number_is_paired_by_position_when_the_lengths_agree(
        self, tmp_path, chain, pairing_key
    ):
        # two residues of one chain with the same number and no insertion code cannot be paired
        # by number; the chains have the same length, so they are paired by position
        design = _helix_complex(tmp_path, "design.pdb")
        second = self._renumbered(
            tmp_path, "afm.pdb", chain, 5 if chain == "A" else 3, 4 if chain == "A" else 2, ""
        )
        res = compute_evobind_adversarial_check(design, second, "B", "A")
        assert res[pairing_key] == "position"
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_a_repeated_residue_number_raises_when_the_lengths_differ(self, tmp_path):
        design = _helix_complex(tmp_path, "design.pdb", n_rec=10)
        second = self._renumbered(tmp_path, "afm.pdb", "A", 5, 4, "", n_rec=12)
        with pytest.raises(
            ValueError, match=r"receptor residues cannot be paired.*Cα atoms.*by position needs"
        ):
            compute_evobind_adversarial_check(design, second, "B", "A")


# ---------------------------------------------------------------------------
# A second model that numbers every chain from 1 (issue #108)
# ---------------------------------------------------------------------------


def renumber_from_one(atoms):
    """Copy of ``atoms`` with the residues of each chain numbered 1, 2, ... in file order.

    Boltz-2 numbers the chains of its output like this, whatever the input numbering.
    """
    out = atoms.copy()
    for chain in np.unique(out.chain_id):
        index = np.where(out.chain_id == chain)[0]
        keys = list(zip(out.res_id[index].tolist(), out.ins_code[index].tolist()))
        number: dict = {}
        for key in keys:
            number.setdefault(key, len(number) + 1)
        out.res_id[index] = [number[key] for key in keys]
        out.ins_code[index] = ""
    return out


class TestRenumberedFromOneSecondModel:
    """1YCR: receptor A is numbered 25-109 and binder B 17-29; the second model numbers both from 1.

    The receptor numbers overlap (25-85) but name other residues, so a pairing by number is
    name-inconsistent for 93% of the pairs and must give way to a pairing by position.
    """

    @pytest.fixture
    def design_atoms(self, example_pdb_path):
        return _load_atoms(example_pdb_path)

    def test_the_second_model_is_paired_by_position(self, tmp_path, example_pdb_path, design_atoms):
        second = _write(tmp_path / "boltz_like.pdb", renumber_from_one(design_atoms))
        res = compute_evobind_adversarial_check(example_pdb_path, second, "B", "A")
        assert res["receptor_pairing"] == "position"
        assert res["binder_pairing"] == "position"
        assert res["n_superposition_atoms"] == 85
        assert res["n_superposition_residues"] == 0  # none is paired by number
        assert res["receptor_resname_mismatch_fraction"] == 0.0
        assert res["binder_resname_mismatch_fraction"] == 0.0
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_the_interface_follows_the_positional_pairing(
        self, tmp_path, example_pdb_path, design_atoms
    ):
        # same coordinates and residues, so the interface distances are those of the design
        # against itself; mapping the interface residues by number would pick other residues
        second = _write(tmp_path / "boltz_like.pdb", renumber_from_one(design_atoms))
        renumbered = compute_evobind_adversarial_check(example_pdb_path, second, "B", "A")
        same_numbers = compute_evobind_adversarial_check(
            example_pdb_path, example_pdb_path, "B", "A"
        )
        assert renumbered["interface_fallback_used"] is False
        for key in ("afm_if_dist_pep_to_rec", "afm_if_dist_rec_to_pep", "afm_mean_if_dist"):
            assert renumbered[key] == pytest.approx(same_numbers[key], abs=1e-3)

    def test_a_displaced_binder_gives_the_known_delta_com(
        self, tmp_path, example_pdb_path, design_atoms
    ):
        moved = renumber_from_one(design_atoms)
        moved.coord[moved.chain_id == "B"] += [0.0, 3.0, 4.0]
        second = _write(tmp_path / "boltz_like_moved.pdb", moved)
        res = compute_evobind_adversarial_check(example_pdb_path, second, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)

    def test_matching_numbers_are_still_paired_by_number(self, example_pdb_path):
        res = compute_evobind_adversarial_check(example_pdb_path, example_pdb_path, "B", "A")
        assert res["receptor_pairing"] == "residue_number"
        assert res["binder_pairing"] == "residue_number"
        assert res["n_superposition_residues"] == 85

    def test_residues_that_name_other_residues_in_both_pairings_still_raise(
        self, tmp_path, example_pdb_path, design_atoms
    ):
        scrambled = renumber_from_one(design_atoms)
        receptor = scrambled.chain_id == "A"
        names = scrambled.res_name[receptor]
        # give every residue the name of the residue seven places on: no pairing agrees
        _, first, inverse = np.unique(
            scrambled.res_id[receptor], return_index=True, return_inverse=True
        )
        per_residue = names[first]
        scrambled.res_name[receptor] = np.roll(per_residue, 7)[inverse]
        second = _write(tmp_path / "scrambled.pdb", scrambled)
        with pytest.raises(
            ValueError, match=r"receptor residues cannot be paired.*by position have different"
        ):
            compute_evobind_adversarial_check(example_pdb_path, second, "B", "A")

    def test_chains_of_different_length_are_not_paired_by_position(
        self, tmp_path, example_pdb_path, design_atoms
    ):
        shorter = design_atoms[~((design_atoms.chain_id == "A") & (design_atoms.res_id == 60))]
        second = _write(tmp_path / "shorter.pdb", renumber_from_one(shorter))
        with pytest.raises(ValueError, match=r"receptor residues cannot be paired.*84"):
            compute_evobind_adversarial_check(example_pdb_path, second, "B", "A")
