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
    _auto_interface_mask,
    _cb_atoms,
    _pairwise_min_dists,
    _per_residue_plddt,
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


def _helix_complex(tmp_path, name, n_rec=12, first_rec_res=1, first_pep_res=1, pep_shift=(0, 0, 0)):
    """α-helix receptor (chain A) with a short extended binder (chain B) beside it.

    All coordinates depend only on the residue index, so two calls with
    different ``first_*_res`` describe the same geometry under different numbering.
    """
    rec_ca = _helix_ca(n_rec)
    radial = rec_ca[:, :2] / np.linalg.norm(rec_ca[:, :2], axis=1, keepdims=True)
    rec_cb = rec_ca + 1.5 * np.column_stack([radial, np.zeros(n_rec)])
    pep_ca = np.array([[9.0, 0.0, 2.0 + 3.8 * j] for j in range(4)]) + np.asarray(pep_shift)
    pep_cb = pep_ca + [-1.5, 0.0, 0.0]
    return _write(
        tmp_path / name,
        _chain("A", rec_ca, rec_cb, first_res_id=first_rec_res),
        _chain("B", pep_ca, pep_cb, first_res_id=first_pep_res),
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
