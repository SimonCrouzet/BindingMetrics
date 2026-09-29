"""Behavioural tests for ``binding_metrics.metrics.receptor_quality``.

Backbones are built in-test from ideal Engh & Huber internal coordinates
(NeRF construction) so that phi/psi, bond lengths and Cβ positions are known
exactly. Only the clashscore regression tests read bundled example structures
from ``data/``; nothing uses the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics import receptor_quality as rq  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Synthetic structure builders
# ---------------------------------------------------------------------------


def _place(a, b, c, bond, angle_deg, dihedral_deg):
    """NeRF step: position d with |cd| = bond, angle(b, c, d) and dihedral(a, b, c, d) as given."""
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    bc = (c - b) / np.linalg.norm(c - b)
    normal = np.cross(b - a, bc)
    normal /= np.linalg.norm(normal)
    binormal = np.cross(normal, bc)
    ang, dih = np.deg2rad(angle_deg), np.deg2rad(dihedral_deg)
    local = np.array(
        [-bond * np.cos(ang), bond * np.sin(ang) * np.cos(dih), bond * np.sin(ang) * np.sin(dih)]
    )
    return c + local[0] * bc + local[1] * binormal + local[2] * normal


def _backbone_residues(phi_psi):
    """Ideal-geometry N, CA, C, O, CB coordinates for a chain with the given (phi, psi) pairs.

    phi of the first and psi of the last residue are undefined in the structure
    (their value here only fixes where the chain starts and ends). All peptide
    bonds are trans. CB is on the L side (verified against the chirality of a
    real L-protein).
    """
    residues = []
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([1.458, 0.0, 0.0])
    c = np.array(
        [
            1.458 + 1.525 * np.cos(np.deg2rad(180 - 111.2)),
            1.525 * np.sin(np.deg2rad(180 - 111.2)),
            0,
        ]
    )
    for i, (phi, psi) in enumerate(phi_psi):
        if i > 0:
            prev = residues[-1]
            n = _place(prev["N"], prev["CA"], prev["C"], 1.336, 116.2, phi_psi[i - 1][1])
            ca = _place(prev["CA"], prev["C"], n, 1.458, 121.7, 180.0)
            c = _place(prev["C"], n, ca, 1.525, 111.2, phi)
        residues.append(
            {
                "N": n,
                "CA": ca,
                "C": c,
                "O": _place(n, ca, c, 1.229, 120.8, psi + 180.0),
                "CB": _place(c, n, ca, 1.521, 110.4, -122.6),
            }
        )
    return residues


def _chain_atoms(
    residues, res_names=None, chain_id="A", first_res_id=1, b_factor=None, res_ids=None
):
    """AtomArray from residue dicts; GLY residues lose their CB."""
    atoms = []
    for i, res in enumerate(residues):
        name = "ALA" if res_names is None else res_names[i]
        for atom_name, xyz in res.items():
            if name == "GLY" and atom_name == "CB":
                continue
            atoms.append(
                struc.Atom(
                    xyz,
                    chain_id=chain_id,
                    res_id=first_res_id + i if res_ids is None else res_ids[i],
                    res_name=name,
                    atom_name=atom_name,
                    element=atom_name[0],
                )
            )
    arr = struc.array(atoms)
    arr.set_annotation("b_factor", np.zeros(arr.array_length()))
    if b_factor is not None:
        arr.b_factor[arr.atom_name == "CA"] = b_factor
    return arr


def _helix(n=6, **kwargs):
    return _chain_atoms(_backbone_residues([(-63.0, -43.0)] * n), **kwargs)


def _serine_helix(n=6, chi1=-60.0, **kwargs):
    """Helix of serines, so that every residue has a chi1 and the rotamer term is defined."""
    residues = _backbone_residues([(-63.0, -43.0)] * n)
    for res in residues:
        res["OG"] = _place(res["N"], res["CA"], res["CB"], 1.43, 110.0, chi1)
    return _chain_atoms(residues, res_names=["SER"] * n, **kwargs)


def _atoms_at(*records):
    """AtomArray from (chain, res_id, res_name, atom_name, element, xyz) tuples."""
    return struc.array(
        [
            struc.Atom(xyz, chain_id=ch, res_id=rid, res_name=rn, atom_name=an, element=el)
            for ch, rid, rn, an, el, xyz in records
        ]
    )


# ---------------------------------------------------------------------------
# Builder self-check: the fixtures really have the geometry the tests rely on
# ---------------------------------------------------------------------------


def test_builder_reproduces_requested_dihedrals():
    arr = _chain_atoms(
        _backbone_residues([(0.0, -43.0), (-63.0, -43.0), (-120.0, 130.0), (-90.0, 0.0)])
    )
    phi, psi, omega = struc.dihedral_backbone(arr)
    np.testing.assert_allclose(np.degrees(phi[1:3]), [-63.0, -120.0], atol=1e-3)
    np.testing.assert_allclose(np.degrees(psi[1:3]), [-43.0, 130.0], atol=1e-3)
    assert np.all(np.abs(np.degrees(omega[:-1])) > 179.9)


# ---------------------------------------------------------------------------
# Ramachandran
# ---------------------------------------------------------------------------


class TestRamachandran:
    def test_alpha_helix_is_fully_favoured(self):
        res = rq._ramachandran(_helix(8))
        # first residue has no phi and last has no psi
        assert res["n_evaluated"] == 6
        assert res["favoured_count"] == 6
        assert res["favoured_pct"] == pytest.approx(100.0)
        assert res["outlier_count"] == 0
        assert "reason" not in res

    def test_region_counts_and_percentages(self):
        phi_psi = [
            (0.0, -43.0),  # phi undefined
            (-63.0, -43.0),  # alpha: favoured
            (-120.0, 130.0),  # beta: favoured
            (-100.0, 20.0),  # allowed
            (60.0, -120.0),  # outlier
            (-63.0, 0.0),  # psi undefined
        ]
        res = rq._ramachandran(_chain_atoms(_backbone_residues(phi_psi)))
        assert res["n_evaluated"] == 4
        assert (res["favoured_count"], res["allowed_count"], res["outlier_count"]) == (2, 1, 1)
        assert res["favoured_pct"] == pytest.approx(50.0)
        assert res["allowed_pct"] == pytest.approx(25.0)
        assert res["outlier_pct"] == pytest.approx(25.0)

    def test_d_residue_is_scored_on_the_mirrored_plot(self):
        # (phi, psi) = (120, -130) is an outlier for an L residue but the mirror image
        # of the L beta region, so it is favoured for a D residue.
        phi_psi = [(0.0, -130.0)] + [(120.0, -130.0)] * 4 + [(120.0, 0.0)]
        residues = _backbone_residues(phi_psi)
        as_l = rq._ramachandran(_chain_atoms(residues, res_names=["ALA"] * 6))
        as_d = rq._ramachandran(_chain_atoms(residues, res_names=["DAL"] * 6))
        assert as_l["outlier_count"] == 4 and as_l["favoured_count"] == 0
        assert as_d["favoured_count"] == 4 and as_d["outlier_count"] == 0

    def test_single_residue_has_nothing_to_evaluate(self):
        res = rq._ramachandran(_chain_atoms(_backbone_residues([(-63.0, -43.0)])))
        assert res["n_evaluated"] == 0
        assert np.isnan(res["favoured_pct"]) and np.isnan(res["outlier_pct"])
        assert res["outlier_count"] == 0
        assert res["reason"] == "no residue with both phi and psi"

    def test_missing_backbone_atoms_give_the_empty_record_with_a_reason(self):
        arr = _helix(4)
        no_carbonyl = arr[arr.atom_name != "C"]
        res = rq._ramachandran(no_carbonyl)
        assert res["n_evaluated"] == 0
        assert np.isnan(res["favoured_pct"])
        assert res["reason"] == "no residue with both phi and psi"

    def test_empty_input_gives_the_empty_record_with_a_reason(self):
        res = rq._ramachandran(_helix(4)[:0])
        assert res["n_evaluated"] == 0
        assert np.isnan(res["favoured_pct"])
        assert res["reason"].startswith("backbone dihedrals unavailable")

    def test_unexpected_errors_are_not_swallowed(self, monkeypatch):
        def _boom(atoms):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(struc, "dihedral_backbone", _boom)
        with pytest.raises(RuntimeError, match="unexpected"):
            rq._ramachandran(_helix(4))


# ---------------------------------------------------------------------------
# Clashscore
# ---------------------------------------------------------------------------


def _pair(distance, res_a=1, res_b=4, chain_b="A", elements=("C", "C"), names=("C1", "C2")):
    return _atoms_at(
        ("A", res_a, "ALA", names[0], elements[0], [0.0, 0.0, 0.0]),
        (chain_b, res_b, "ALA", names[1], elements[1], [distance, 0.0, 0.0]),
    )


class TestClashscore:
    def test_overlap_above_cutoff_is_a_clash(self):
        # C...C radii sum 3.4 A: 2.8 A is a 0.6 A overlap, 3.2 A only 0.2 A
        clash = rq._clashscore(_pair(2.8))
        assert clash["n_clashes"] == 1
        assert clash["n_heavy_atoms"] == 2
        assert clash["clashscore"] == pytest.approx(500.0)
        assert rq._clashscore(_pair(3.2))["n_clashes"] == 0

    def test_clash_cutoff_argument_is_honoured(self):
        assert rq._clashscore(_pair(2.8), clash_cutoff=0.7)["n_clashes"] == 0
        assert rq._clashscore(_pair(3.2), clash_cutoff=0.1)["n_clashes"] == 1

    def test_same_and_adjacent_residues_are_exempt(self):
        assert rq._clashscore(_pair(2.0, res_a=1, res_b=1))["n_clashes"] == 0
        assert rq._clashscore(_pair(2.0, res_a=1, res_b=2))["n_clashes"] == 0
        assert rq._clashscore(_pair(2.0, res_a=1, res_b=3))["n_clashes"] == 1

    def test_residue_adjacency_only_applies_within_a_chain(self):
        assert rq._clashscore(_pair(2.0, res_a=1, res_b=2, chain_b="B"))["n_clashes"] == 1

    def test_hydrogens_are_not_counted(self):
        arr = _atoms_at(
            ("A", 1, "ALA", "C1", "C", [0.0, 0.0, 0.0]),
            ("A", 5, "ALA", "C2", "C", [5.0, 0.0, 0.0]),
            ("A", 1, "ALA", "H1", "H", [4.9, 0.0, 0.0]),
        )
        res = rq._clashscore(arr)
        assert res["n_heavy_atoms"] == 2
        assert res["n_clashes"] == 0

    def test_per_thousand_atoms_normalisation(self):
        # one clashing pair among 4 well separated heavy atoms -> 1000 * 1 / 4
        arr = _atoms_at(
            ("A", 1, "ALA", "C1", "C", [0.0, 0.0, 0.0]),
            ("A", 4, "ALA", "C2", "C", [2.8, 0.0, 0.0]),
            ("A", 8, "ALA", "C3", "C", [30.0, 0.0, 0.0]),
            ("A", 12, "ALA", "C4", "C", [60.0, 0.0, 0.0]),
        )
        res = rq._clashscore(arr)
        assert res["n_clashes"] == 1
        assert res["clashscore"] == pytest.approx(250.0)

    def test_fewer_than_two_heavy_atoms_is_nan(self):
        arr = _atoms_at(("A", 1, "ALA", "C1", "C", [0.0, 0.0, 0.0]))
        res = rq._clashscore(arr)
        assert np.isnan(res["clashscore"])
        assert res["n_clashes"] == 0
        assert res["reason"] == "fewer than two heavy atoms"

    def test_ideal_helix_has_no_clashes(self):
        res = rq._clashscore(_helix(10))
        assert res["n_clashes"] == 0
        assert "reason" not in res


_LEGACY = {"exclude_bonded": False, "exempt_hbond_pairs": False}


def _disulfide_atoms(res_a=3, res_b=11):
    """Two cysteines joined by an S-S bond (2.04 A, CB-S-S 104 deg, dihedral 90 deg)."""
    cb_a = np.zeros(3)
    sg_a = np.array([1.82, 0.0, 0.0])
    sg_b = _place([0.0, 1.0, 0.0], cb_a, sg_a, 2.04, 104.0, 0.0)
    cb_b = _place(cb_a, sg_a, sg_b, 1.82, 104.0, 90.0)
    return _atoms_at(
        ("A", res_a, "CYS", "CB", "C", cb_a),
        ("A", res_a, "CYS", "SG", "S", sg_a),
        ("A", res_b, "CYS", "SG", "S", sg_b),
        ("A", res_b, "CYS", "CB", "C", cb_b),
    )


class TestClashscoreCovalentAndHbondPairs:
    def test_disulfide_and_its_neighbours_are_not_clashes(self):
        cys = _disulfide_atoms()
        # S-S overlaps by 1.56 A and each CB sits 3.04 A from the other sulfur
        assert rq._clashscore(cys, **_LEGACY)["n_clashes"] == 3
        assert rq._clashscore(cys)["n_clashes"] == 0
        assert rq._clashscore(cys, exclude_bonded=False)["n_clashes"] == 3

    def test_a_real_clash_next_to_a_disulfide_is_still_counted(self):
        far_pair = _atoms_at(
            ("A", 20, "ALA", "C1", "C", [10.0, 10.0, 10.0]),
            ("A", 30, "ALA", "C2", "C", [12.8, 10.0, 10.0]),
        )
        res = rq._clashscore(_disulfide_atoms() + far_pair)
        assert res["n_clashes"] == 1
        assert res["n_heavy_atoms"] == 6

    def test_non_bonded_sulfur_contact_between_cysteines_is_counted(self):
        # two free thiols at 2.9 A (overlap 0.7 A) are not a disulfide (limit 2.6 A)
        thiols = _atoms_at(
            ("A", 3, "CYS", "SG", "S", [0.0, 0.0, 0.0]),
            ("A", 11, "CYS", "SG", "S", [2.9, 0.0, 0.0]),
        )
        assert rq._clashscore(thiols)["n_clashes"] == 1

    def test_peptide_bond_across_a_numbering_gap_is_not_a_clash(self):
        # chymotrypsin-style numbering: residue 2 is followed by residue 4 (no 3)
        arr = _chain_atoms(_backbone_residues([(-63.0, -43.0)] * 4), res_ids=[1, 2, 4, 5])
        assert rq._clashscore(arr, **_LEGACY)["n_clashes"] > 0
        assert rq._clashscore(arr)["n_clashes"] == 0

    def test_hydrogen_bonded_heteroatom_pairs_are_exempt_from_2p5_angstrom(self):
        for elements in (("N", "O"), ("O", "N"), ("O", "O")):
            pair = _pair(2.6, elements=elements)
            assert rq._clashscore(pair, **_LEGACY)["n_clashes"] == 1
            assert rq._clashscore(pair)["n_clashes"] == 0
            # the two switches are independent
            assert rq._clashscore(pair, exclude_bonded=False)["n_clashes"] == 0
            assert rq._clashscore(pair, exempt_hbond_pairs=False)["n_clashes"] == 1

    def test_heteroatom_pair_below_2p5_angstrom_is_still_a_clash(self):
        assert rq._clashscore(_pair(2.3, elements=("N", "O")))["n_clashes"] == 1

    def test_carbon_oxygen_contact_is_not_exempt(self):
        # C...O radii sum 3.22 A: 2.7 A overlaps by 0.52 A; only N/O pairs are H-bond candidates
        assert rq._clashscore(_pair(2.7, elements=("C", "O")))["n_clashes"] == 1

    def test_vectorised_count_matches_the_pair_loop_when_exemptions_are_off(self):
        rng = np.random.default_rng(3)
        xyz = rng.uniform(0.0, 10.0, size=(300, 3))
        arr = struc.array(
            [
                struc.Atom(
                    x,
                    chain_id="AB"[k % 2],
                    res_id=k // 5,
                    res_name="ALA",
                    atom_name="C1",
                    element="CNOS"[k % 4],
                )
                for k, x in enumerate(xyz)
            ]
        )
        for cutoff in (0.2, 0.4, 0.8):
            expected = 0
            for a in range(len(arr)):
                for b in range(a + 1, len(arr)):
                    if (
                        arr.chain_id[a] == arr.chain_id[b]
                        and abs(arr.res_id[a] - arr.res_id[b]) <= 1
                    ):
                        continue
                    d = np.linalg.norm(arr.coord[a].astype(float) - arr.coord[b].astype(float))
                    overlap = rq._vdw(arr.element[a]) + rq._vdw(arr.element[b]) - d
                    expected += overlap >= cutoff
            assert rq._clashscore(arr, cutoff, **_LEGACY)["n_clashes"] == expected


def _amino_acid_chain(filename, chain):
    atoms = rq._load_all_models(DATA_DIR / filename)[0]
    return atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == chain)]


class TestClashscoreBundledStructures:
    # (file, chain, earlier count, count now, what the earlier pairs were)
    @pytest.mark.parametrize(
        "filename, chain, legacy, current",
        [
            # SFTI-1: head-to-tail closure (GLY1 N - ASP14 C 1.44 A) and CYS3-CYS11 disulfide
            ("example_bicyclic_sfti1_3P8F.cif", "I", 7, 0),
            # trypsin: 3 disulfides, 2 numbering gaps (149, 218), 19 N/O hydrogen bonds;
            # THR62 O - ARG84 NH2 at 2.28 A remains
            ("example_bicyclic_sfti1_3P8F.cif", "A", 34, 1),
            # somatostatin lactam GLU4 CD - LYS10 NZ (1.89 A) and CYS2-CYS12 disulfide
            ("example_lactam_somatostatin_1XY4.cif", "A", 6, 0),
            # cyclosporin head-to-tail closure and a BMT OG1...O hydrogen bond
            ("example_ncaa_cyclosporin_1CWA.cif", "C", 6, 0),
            # hydrocarbon staple 0EH20 CAT = MK827 CE (1.39 A)
            ("example_staple_3V3B.pdb", "C", 3, 0),
        ],
    )
    def test_covalent_links_and_hydrogen_bonds_are_no_longer_clashes(
        self, filename, chain, legacy, current
    ):
        atoms = _amino_acid_chain(filename, chain)
        assert rq._clashscore(atoms, **_LEGACY)["n_clashes"] == legacy
        assert rq._clashscore(atoms)["n_clashes"] == current

    def test_sfti1_bicycle_scores_zero_and_is_reported_through_the_public_api(self, monkeypatch):
        _fake_energy(monkeypatch)
        path = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
        now = rq.compute_receptor_quality(path, receptor_chain="I")["models"][0]
        before = rq.compute_receptor_quality(path, receptor_chain="I", exclude_bonded=False)[
            "models"
        ][0]
        assert now["clashes"]["clashscore"] == 0.0
        assert before["clashes"]["clashscore"] == pytest.approx(1000.0 * 7 / 105)
        assert now["molprobity_score"] < before["molprobity_score"]


# ---------------------------------------------------------------------------
# Rotamers, Cβ deviation, backbone geometry
# ---------------------------------------------------------------------------


class TestRotamers:
    @pytest.mark.parametrize(
        "chi1, expected",
        [
            (-60.0, 0.0),
            (60.0, 0.0),
            (180.0, 0.0),
            (-180.0, 0.0),
            (0.0, 60.0),
            (120.0, 60.0),
            (-90.0, 30.0),
        ],
    )
    def test_distance_to_canonical_chi1(self, chi1, expected):
        assert rq._chi1_dist_to_canonical(chi1) == pytest.approx(expected)

    def test_staggered_chi1_is_not_an_outlier_and_eclipsed_is(self):
        chi1_values = [-60.0, 180.0, 60.0, 0.0]
        residues = _backbone_residues([(-63.0, -43.0)] * len(chi1_values))
        for res, chi1 in zip(residues, chi1_values):
            res["OG"] = _place(res["N"], res["CA"], res["CB"], 1.43, 110.0, chi1)
        arr = _chain_atoms(residues, res_names=["SER"] * len(chi1_values))
        res = rq._rotamer_quality(arr)
        assert res["n_evaluated"] == 4
        assert res["outlier_count"] == 1  # chi1 = 0 deg is 60 deg from any canonical value
        assert res["outlier_pct"] == pytest.approx(25.0)

    def test_residues_without_chi1_are_skipped(self):
        res = rq._rotamer_quality(_helix(5))  # all ALA
        assert res["n_evaluated"] == 0
        assert np.isnan(res["outlier_pct"])
        assert res["reason"] == "no residue with a complete chi1 dihedral"


class TestCbetaDeviation:
    def test_ideal_cbeta_geometry(self):
        res = _backbone_residues([(-63.0, -43.0)] * 3)[1]
        cb = rq._ideal_cbeta(res["N"], res["CA"], res["C"])
        assert np.linalg.norm(cb - res["CA"]) == pytest.approx(1.521)
        # near-tetrahedral, symmetric about the N-CA-C bisector
        n_ca_cb = rq._angle_deg(res["N"], res["CA"], cb)
        c_ca_cb = rq._angle_deg(res["C"], res["CA"], cb)
        assert n_ca_cb == pytest.approx(c_ca_cb, abs=1e-6)
        assert 108.0 < n_ca_cb < 112.0
        # agrees with the independently constructed L-amino-acid Cβ to well under 0.25 A
        assert np.linalg.norm(cb - res["CB"]) < 0.15

    def test_degenerate_backbone_returns_none(self):
        p = np.array([0.0, 0.0, 0.0])
        assert rq._ideal_cbeta(p, p, p) is None
        # collinear N, CA, C
        assert rq._ideal_cbeta(np.array([-1.0, 0, 0]), p, np.array([1.0, 0, 0])) is None

    def test_ideal_backbone_has_no_deviations_and_gly_is_skipped(self):
        residues = _backbone_residues([(-63.0, -43.0)] * 5)
        arr = _chain_atoms(residues, res_names=["ALA", "GLY", "ALA", "ALA", "ALA"])
        res = rq._cbeta_deviations(arr)
        assert res["cb_n_evaluated"] == 4
        assert res["cb_deviation_count"] == 0
        assert res["cb_deviation_pct"] == 0.0

    def test_displaced_cbeta_is_flagged(self):
        residues = _backbone_residues([(-63.0, -43.0)] * 4)
        residues[2]["CB"] = residues[2]["CB"] + np.array([0.0, 0.0, 0.6])
        res = rq._cbeta_deviations(_chain_atoms(residues))
        assert res["cb_deviation_count"] == 1
        assert res["cb_deviation_pct"] == pytest.approx(25.0)

    def test_no_cbeta_residues_gives_nan_percentage(self):
        residues = _backbone_residues([(-63.0, -43.0)] * 3)
        res = rq._cbeta_deviations(_chain_atoms(residues, res_names=["GLY"] * 3))
        assert res["cb_n_evaluated"] == 0
        assert np.isnan(res["cb_deviation_pct"])
        assert res["reason"] == "no residue with N, CA, C and CB"

    def test_no_reason_when_residues_were_evaluated(self):
        assert "reason" not in rq._cbeta_deviations(_helix(4))


class TestBackboneGeometry:
    def test_ideal_backbone_has_no_outliers(self):
        n = 6
        res = rq._backbone_geometry(_helix(n))
        # per residue 3 bonds (N-CA, CA-C, C=O) and 2 angles; per junction 1 bond and 2 angles
        assert res["total_bonds"] == 3 * n + (n - 1)
        assert res["total_angles"] == 2 * n + 2 * (n - 1)
        assert res["bad_bonds"] == 0 and res["bad_angles"] == 0
        assert res["bad_bonds_pct"] == 0.0
        assert "reason" not in res

    def test_atoms_without_a_backbone_give_nan_with_a_reason(self):
        arr = _atoms_at(("A", 1, "ALA", "CB", "C", [0.0, 0.0, 0.0]))
        res = rq._backbone_geometry(arr)
        assert res["total_bonds"] == 0 and res["total_angles"] == 0
        assert np.isnan(res["bad_bonds_pct"]) and np.isnan(res["bad_angles_pct"])
        assert res["reason"] == "no complete backbone bond or angle found"

    def test_stretched_carbonyl_is_a_bad_bond(self):
        residues = _backbone_residues([(-63.0, -43.0)] * 4)
        c, o = residues[1]["C"], residues[1]["O"]
        residues[1]["O"] = c + 1.5 * (o - c) / np.linalg.norm(o - c)  # 1.229 -> 1.5 A
        res = rq._backbone_geometry(_chain_atoms(residues))
        assert res["bad_bonds"] == 1
        assert res["bad_bonds_pct"] == pytest.approx(100.0 / res["total_bonds"])

    def test_chain_break_skips_peptide_terms(self):
        residues = _backbone_residues([(-63.0, -43.0)] * 4)
        shift = np.array([20.0, 0.0, 0.0])
        for res in residues[2:]:
            for name in res:
                res[name] = res[name] + shift
        broken = rq._backbone_geometry(_chain_atoms(residues))
        # one junction (2->3) is dropped: minus 1 peptide bond and 2 inter-residue angles
        assert broken["total_bonds"] == 3 * 4 + 2
        assert broken["total_angles"] == 2 * 4 + 2 * 2
        assert broken["bad_bonds"] == 0 and broken["bad_angles"] == 0


# ---------------------------------------------------------------------------
# B-factors, chain detection, composite score
# ---------------------------------------------------------------------------


class TestBFactors:
    def test_statistics_over_calpha(self):
        arr = _helix(3, b_factor=np.array([10.0, 20.0, 90.0]))
        res = rq._bfactor_stats(arr)
        assert res["available"] is True
        assert res["mean_b_factor"] == pytest.approx(40.0)
        assert res["max_b_factor"] == 90.0 and res["min_b_factor"] == 10.0
        assert res["n_high_b_residues"] == 1  # only 90 exceeds 60 A^2

    def test_all_zero_b_factors_mean_unavailable(self):
        assert rq._bfactor_stats(_helix(3)) == {"available": False}


class TestDetectReceptorChain:
    def test_largest_amino_acid_chain_wins(self):
        arr = _helix(3, chain_id="A") + _helix(7, chain_id="B")
        assert rq._detect_receptor_chain(arr) == "B"

    def test_no_amino_acids_returns_none(self):
        arr = _atoms_at(("A", 1, "HOH", "O", "O", [0.0, 0.0, 0.0]))
        assert rq._detect_receptor_chain(arr) is None


class TestMolprobityScore:
    def test_clean_structure_sits_at_the_offset(self):
        assert rq._molprobity_score(0.0, 0.0, 0.0) == pytest.approx(0.5)

    def test_baseline_noise_is_not_penalised(self):
        # 0.2 % Ramachandran and 2 % rotamer outliers are the free allowance
        assert rq._molprobity_score(0.0, 0.2, 2.0) == pytest.approx(0.5)

    def test_formula(self):
        clash, rama, rota = 10.0, 1.2, 6.0
        expected = (
            0.426 * np.log(1 + clash)
            + 0.33 * np.log(1 + (rama - 0.2) / 0.2)
            + 0.25 * np.log(1 + (rota - 2.0) / 2.0)
            + 0.5
        )
        assert rq._molprobity_score(clash, rama, rota) == pytest.approx(expected)

    def test_worse_inputs_give_a_higher_score(self):
        base = rq._molprobity_score(5.0, 1.0, 5.0)
        assert rq._molprobity_score(20.0, 1.0, 5.0) > base
        assert rq._molprobity_score(5.0, 4.0, 5.0) > base
        assert rq._molprobity_score(5.0, 1.0, 15.0) > base

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_input_gives_nan(self, bad):
        assert np.isnan(rq._molprobity_score(bad, 0.0, 0.0))
        assert np.isnan(rq._molprobity_score(0.0, bad, 0.0))
        assert np.isnan(rq._molprobity_score(0.0, 0.0, bad))


# ---------------------------------------------------------------------------
# Energy wrapper: error path
# ---------------------------------------------------------------------------


class TestReceptorEnergyErrorPath:
    def test_missing_openmm_is_reported_not_raised(self, monkeypatch):
        def _no_openmm():
            raise ImportError("openmm is required for energy computation.")

        monkeypatch.setattr(rq, "_import_openmm", _no_openmm)
        res = rq._receptor_energy(_helix(3))
        assert np.isnan(res["energy_kJ_mol"]) and np.isnan(res["energy_per_residue_kJ_mol"])
        assert res["n_atoms_with_h"] is None
        assert "openmm is required" in res["error"]

    def test_unbuildable_input_is_reported_with_exception_type(self):
        res = rq._receptor_energy(_helix(3)[:0])
        assert np.isnan(res["energy_kJ_mol"])
        assert res["n_atoms_with_h"] is None
        assert isinstance(res["error"], str) and res["error"]


# ---------------------------------------------------------------------------
# Energy wrapper: seeding
# ---------------------------------------------------------------------------


class TestReceptorEnergySeeding:
    def test_default_seed_mirrors_the_package_default(self):
        pytest.importorskip("openmm")
        from binding_metrics.core.system import DEFAULT_RANDOM_SEED

        assert rq.DEFAULT_RANDOM_SEED == DEFAULT_RANDOM_SEED

    def test_repeated_calls_give_identical_energy(self):
        pytest.importorskip("openmm")
        atoms = _helix(6)
        first = rq._receptor_energy(atoms)
        second = rq._receptor_energy(atoms)
        assert first["error"] is None and second["error"] is None
        assert np.isfinite(first["energy_kJ_mol"])
        assert first["energy_kJ_mol"] == pytest.approx(second["energy_kJ_mol"], rel=1e-6)
        assert first["n_atoms_with_h"] == second["n_atoms_with_h"]

    def test_explicit_seed_is_repeatable_and_leaves_the_global_rng_alone(self):
        import random

        pytest.importorskip("openmm")
        atoms = _helix(6)
        first = rq._receptor_energy(atoms, random_seed=7)
        random.seed(123)
        state = random.getstate()
        second = rq._receptor_energy(atoms, random_seed=7)
        assert random.getstate() == state
        assert first["energy_kJ_mol"] == pytest.approx(second["energy_kJ_mol"], rel=1e-6)

    def test_seed_reaches_the_energy_term_from_the_public_api(self, tmp_path, monkeypatch):
        seen = []

        def _spy(atoms, solvent_model="obc2", device="cuda", random_seed="unset"):
            seen.append(random_seed)
            return {
                "energy_kJ_mol": -1.0,
                "energy_per_residue_kJ_mol": -1.0,
                "n_atoms_with_h": 1,
                "error": None,
            }

        monkeypatch.setattr(rq, "_receptor_energy", _spy)
        pdb = pdb_io.PDBFile()
        pdb_io.set_structure(pdb, _serine_helix(5))
        path = tmp_path / "h.pdb"
        pdb.write(str(path))

        rq.compute_receptor_quality(path)
        rq.compute_receptor_quality(path, random_seed=11)
        rq.compute_receptor_quality(path, random_seed=None)
        assert seen == [rq.DEFAULT_RANDOM_SEED, 11, None]

    @pytest.mark.parametrize(
        "argv, expected",
        [([], "default"), (["--random-seed", "9"], 9), (["--random-seed", "none"], None)],
    )
    def test_cli_random_seed_flag(self, tmp_path, monkeypatch, argv, expected):
        pytest.importorskip("openmm")
        seen = {}

        def _fake(path, **kwargs):
            seen.update(kwargs)
            return {"receptor_chain": None, "n_models": 0, "models": [], "error": "stop"}

        monkeypatch.setattr(rq, "compute_receptor_quality", _fake)
        monkeypatch.setattr("sys.argv", ["prog", "--input", str(tmp_path / "x.pdb"), *argv])
        rq.main()
        want = rq.DEFAULT_RANDOM_SEED if expected == "default" else expected
        assert seen["random_seed"] == want


# ---------------------------------------------------------------------------
# Aggregation, dispatcher and public API
# ---------------------------------------------------------------------------


def _fake_energy(monkeypatch):
    monkeypatch.setattr(
        rq,
        "_receptor_energy",
        lambda *a, **k: {
            "energy_kJ_mol": -100.0,
            "energy_per_residue_kJ_mol": -10.0,
            "n_atoms_with_h": 1,
            "error": None,
        },
    )


class TestAggregateSummary:
    def test_means_and_best_model(self):
        models = [
            {
                "model_index": 1,
                "molprobity_score": 2.0,
                "clashes": {"clashscore": 10.0},
                "ramachandran": {"favoured_pct": 90.0},
            },
            {
                "model_index": 2,
                "molprobity_score": 1.0,
                "clashes": {"clashscore": 4.0},
                "ramachandran": {"favoured_pct": 96.0},
            },
        ]
        summary = rq._aggregate_summary(models)
        assert summary["clashscore"] == pytest.approx(7.0)
        assert summary["ramachandran_favoured_pct"] == pytest.approx(93.0)
        assert summary["molprobity_score"] == pytest.approx(1.5)
        assert summary["best_model_index"] == 2

    def test_nan_and_missing_values_are_ignored(self):
        models = [
            {"model_index": 1, "molprobity_score": np.nan, "clashes": {"clashscore": np.nan}},
            {"model_index": 2, "molprobity_score": 3.0, "clashes": {"clashscore": 8.0}},
        ]
        summary = rq._aggregate_summary(models)
        assert summary["clashscore"] == pytest.approx(8.0)
        assert summary["best_model_index"] == 2
        assert np.isnan(summary["energy_kJ_mol"])

    def test_no_finite_score_means_no_best_model(self):
        summary = rq._aggregate_summary([{"model_index": 1, "molprobity_score": np.nan}])
        assert summary["best_model_index"] is None


class TestScoreModel:
    def test_receptor_chain_is_selected_and_scored(self, monkeypatch):
        _fake_energy(monkeypatch)
        atoms = _serine_helix(6, chain_id="A") + _serine_helix(4, chain_id="B")
        res = rq._score_model(atoms, "A", 0.4, "obc2", "cuda", 3)
        assert res["model_index"] == 3
        assert res["n_residues"] == 6
        assert res["n_heavy_atoms"] == 6 * 6  # N, CA, C, O, CB, OG per residue
        assert res["ramachandran"]["n_evaluated"] == 4
        assert res["clashes"]["n_clashes"] == 0
        assert res["rotamers"]["n_evaluated"] == 6
        assert np.isfinite(res["molprobity_score"])
        assert "reason" not in res

    def test_chain_without_chi1_residues_has_no_composite_score(self, monkeypatch):
        # the rotamer term is undefined for poly-Ala/Gly, and the composite needs all three terms
        _fake_energy(monkeypatch)
        res = rq._score_model(_helix(6), "A", 0.4, "obc2", "cuda", 1)
        assert np.isnan(res["rotamers"]["outlier_pct"])
        assert np.isnan(res["molprobity_score"])
        assert res["reason"].startswith("molprobity_score undefined")
        assert "rotamers: no residue with a complete chi1 dihedral" in res["reason"]

    def test_missing_receptor_chain_reports_an_error(self):
        res = rq._score_model(_helix(4), "Z", 0.4, "obc2", "cuda", 1)
        assert res["n_residues"] == 0
        assert res["energy"]["error"] == "no receptor atoms found"
        assert np.isnan(res["molprobity_score"])


class TestComputeReceptorQuality:
    def _write_stack(self, path, n_models):
        arr = _serine_helix(6)
        stack = struc.stack([arr] * n_models)
        pdb = pdb_io.PDBFile()
        pdb_io.set_structure(pdb, stack)
        pdb.write(str(path))

    def test_multi_model_file_scores_every_model(self, tmp_path, monkeypatch):
        _fake_energy(monkeypatch)
        path = tmp_path / "ensemble.pdb"
        self._write_stack(path, 2)
        res = rq.compute_receptor_quality(path)
        assert res["receptor_chain"] == "A"
        assert res["n_models"] == 2
        assert [m["model_index"] for m in res["models"]] == [1, 2]
        assert res["summary"]["best_model_index"] in (1, 2)
        assert res["summary"]["ramachandran_favoured_pct"] == pytest.approx(100.0)

    def test_file_without_protein_reports_an_error(self, tmp_path, monkeypatch):
        _fake_energy(monkeypatch)
        arr = _atoms_at(("A", 1, "HOH", "O", "O", [0.0, 0.0, 0.0]))
        arr.hetero[:] = True
        pdb = pdb_io.PDBFile()
        pdb_io.set_structure(pdb, arr)
        path = tmp_path / "water.pdb"
        pdb.write(str(path))
        res = rq.compute_receptor_quality(path)
        assert res["receptor_chain"] is None
        assert res["n_models"] == 0
        assert "No protein chains" in res["error"]


class TestExportHelpers:
    def test_csv_row_and_file(self, tmp_path, monkeypatch):
        _fake_energy(monkeypatch)
        result = rq.compute_receptor_quality(self._pdb(tmp_path))
        result["input_filename"] = "h.pdb"
        out = tmp_path / "out" / "q.csv"
        rq._write_csv(result, out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # header + one model
        header = lines[0].split(",")
        for column in (
            "filename",
            "clashscore",
            "molprobity_score",
            "energy_kJ_mol",
            "rama_favoured_pct",
        ):
            assert column in header

    def _pdb(self, tmp_path):
        pdb = pdb_io.PDBFile()
        pdb_io.set_structure(pdb, _serine_helix(6))
        path = tmp_path / "h.pdb"
        pdb.write(str(path))
        return path

    def test_json_default_handles_numpy_scalars(self):
        payload = {
            "a": np.int64(3),
            "b": np.float64(1.5),
            "c": np.float32("nan"),
            "d": np.arange(2),
        }
        assert json.loads(json.dumps(payload, default=rq._json_default)) == {
            "a": 3,
            "b": 1.5,
            "c": None,
            "d": [0, 1],
        }
