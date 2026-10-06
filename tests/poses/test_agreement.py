"""Pose agreement: all-pairs peptide RMSD, residue correspondence, symmetric atoms, rings.

The poses are built from the bundled examples (data/) and from hand-made arrays by rigid-body
displacements whose pairwise RMSD is known in closed form. Coordinates of an AtomArray are
float32, so the analytic comparisons use an absolute tolerance of 2e-5 angstrom.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
import biotite.structure as struc  # noqa: E402
import biotite.structure.io as strucio  # noqa: E402

from binding_metrics.poses import agreement  # noqa: E402
from binding_metrics.poses.agreement import (  # noqa: E402
    CLUSTER_LINKAGE,
    DEFAULT_CLUSTER_CUTOFFS_ANGSTROM,
    BinderView,
    InterfaceResidues,
    PairMatrix,
    binder_closure_families,
    complete_linkage_clusters,
    contact_map_jaccard,
    heavy_atom_contacts,
    interface_jaccard,
    interface_residues,
    jaccard_matrix,
    medoid,
    medoids,
    pair_statistics,
    pairwise_peptide_rmsd,
    peptide_rmsd,
    positional_spread,
    positional_spread_by_group,
)

DATA = Path(__file__).resolve().parents[2] / "data"
TOL = 2e-5

P53 = "example_linear_p53_1YCR.pdb"
CSA = "example_ncaa_cyclosporin_1CWA.cif"
SFTI = "example_bicyclic_sfti1_3P8F.cif"
PHOSPHO = "example_phospho_1QJB.pdb"


# ---------------------------------------------------------------------------
# builders: every helper returns a modified copy
# ---------------------------------------------------------------------------


def _load(name):
    return strucio.load_structure(DATA / name, model=1, include_bonds=True)


@pytest.fixture(scope="module")
def p53():
    return _load(P53)


@pytest.fixture(scope="module")
def csa():
    return _load(CSA)


@pytest.fixture(scope="module")
def sfti():
    return _load(SFTI)


@pytest.fixture(scope="module")
def phospho():
    return _load(PHOSPHO)


def _numbers(atoms, chain):
    """Residue numbers of a chain in file order (waters included)."""
    mask = atoms.chain_id == chain
    seen = []
    for number in atoms.res_id[mask]:
        if not seen or seen[-1] != number:
            seen.append(int(number))
    return seen


def moved(atoms, chain, vector):
    out = atoms.copy()
    out.coord[out.chain_id == chain] += np.asarray(vector, dtype=np.float32)
    return out


def rotated(atoms, chain, angle_degrees, shift=(0.0, 0.0, 0.0), about_ca_centroid=True):
    """The chain turned about the z axis through the centroid of its alpha carbons, then shifted."""
    out = atoms.copy()
    chain_mask = out.chain_id == chain
    selected = chain_mask & (out.atom_name == "CA") if about_ca_centroid else chain_mask
    center = out.coord[selected].astype(np.float64).mean(axis=0)
    angle = np.radians(angle_degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    coord = out.coord[chain_mask].astype(np.float64)
    out.coord[chain_mask] = ((coord - center) @ rotation.T + center + np.asarray(shift)).astype(
        np.float32
    )
    return out


def swapped_names(atoms, chain, number, *pairs):
    """Relabel atoms: every pair of names is exchanged in the residue (same coordinates)."""
    out = atoms.copy()
    in_residue = (out.chain_id == chain) & (out.res_id == number)
    for first, second in pairs:
        at_first = in_residue & (out.atom_name == first)
        at_second = in_residue & (out.atom_name == second)
        assert at_first.any() and at_second.any(), (number, first, second)
        out.atom_name[at_first] = second
        out.atom_name[at_second] = first
    return out


def without_residue(atoms, chain, number):
    return atoms[~((atoms.chain_id == chain) & (atoms.res_id == number))]


def without_atoms(atoms, chain, numbers, names=None):
    """Drop the named atoms (default: every atom that is not N, CA, C or O) of some residues."""
    drop = np.zeros(atoms.array_length(), dtype=bool)
    for number in numbers:
        in_residue = (atoms.chain_id == chain) & (atoms.res_id == number)
        if names is None:
            drop |= in_residue & ~np.isin(atoms.atom_name, ["N", "CA", "C", "O"])
        else:
            drop |= in_residue & np.isin(atoms.atom_name, list(names))
    return atoms[~drop]


def renumbered(atoms, chain, offset):
    out = atoms.copy()
    out.res_id[out.chain_id == chain] += offset
    return out


def renamed_residue(atoms, chain, number, new_name):
    out = atoms.copy()
    out.res_name[(out.chain_id == chain) & (out.res_id == number)] = new_name
    return out


def ring_origin_moved(atoms, chain, n_first):
    """A ring whose first residue is the ``n_first``-th: the residues are listed from there on,
    renumbered from 1 (waters of the chain are dropped)."""
    chain_atoms = atoms[(atoms.chain_id == chain) & (atoms.res_name != "HOH")]
    starts = list(struc.get_residue_starts(chain_atoms)) + [chain_atoms.array_length()]
    spans = [np.arange(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
    order = spans[n_first:] + spans[:n_first]
    out = chain_atoms[np.concatenate(order)]
    new_ids = np.concatenate([np.full(len(span), i + 1) for i, span in enumerate(order)])
    out.res_id = new_ids
    return out


def build_atoms(rows):
    """An AtomArray from ``(chain, number, insertion code, residue, atom, (x, y, z))`` rows."""
    atoms = struc.AtomArray(len(rows))
    for i, (chain, number, ins_code, residue, name, xyz) in enumerate(rows):
        atoms.chain_id[i] = chain
        atoms.res_id[i] = number
        atoms.ins_code[i] = ins_code
        atoms.res_name[i] = residue
        atoms.atom_name[i] = name
        atoms.element[i] = name[0]
        atoms.coord[i] = xyz
    return atoms


def ring_of_alanines(n_residues=4, radius=2.5):
    """A closed ring of Ala residues with the full symmetry of the regular polygon.

    Each residue is the first one turned by 2 pi i / n about z, so that turning the origin of
    the ring by one residue gives the same set of atoms; C of residue i lies 0.2 of a chord from
    N of residue i + 1, which closes the ring head to tail.
    """
    rows = []
    angles = 2 * np.pi * np.arange(n_residues + 1) / n_residues
    vertices = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], 1
    )
    for i in range(n_residues):
        start, step = vertices[i], vertices[i + 1] - vertices[i]
        offsets = {
            "N": start,
            "CA": start + 0.4 * step + [0, 0, 0.5],
            "C": start + 0.8 * step,
            "O": start + 0.8 * step + [0, 0, 1.2],
            "CB": start + 0.4 * step + [0.3, 0.3, 1.5],
        }
        rows += [("B", i + 1, "", "ALA", name, xyz) for name, xyz in offsets.items()]
    return build_atoms(rows)


def _json_strict(value):
    """Round-trip through JSON that refuses NaN and infinity."""
    return json.loads(json.dumps(value, allow_nan=False))


def _peptide_ids(atoms, chain):
    return sorted(set(_numbers(atoms, chain)))


# ---------------------------------------------------------------------------
# rigid-body displacements with analytic RMSD
# ---------------------------------------------------------------------------

OFFSETS = {"a1": 0.0, "a2": 0.4, "a3": 0.8, "b1": 2.0, "b2": 2.3, "b3": 2.9}


@pytest.fixture(scope="module")
def translated_set(p53):
    return {name: moved(p53, "B", [offset, 0.0, 0.0]) for name, offset in OFFSETS.items()}


class TestTranslations:
    @pytest.mark.parametrize("kind", ["ca", "backbone", "heavy"])
    def test_pairwise_rmsd_is_the_difference_of_the_offsets(self, translated_set, kind):
        result = pairwise_peptide_rmsd(translated_set, "B", kind=kind)
        for first, second in [(a, b) for a in OFFSETS for b in OFFSETS]:
            expected = abs(OFFSETS[first] - OFFSETS[second])
            assert result.value(first, second) == pytest.approx(expected, abs=TOL)

    def test_the_matrix_is_symmetric_with_a_zero_diagonal_and_sorted_ids(self, translated_set):
        result = pairwise_peptide_rmsd(translated_set, "B")
        assert result.pose_ids == tuple(sorted(OFFSETS))
        np.testing.assert_array_equal(result.values, result.values.T)
        np.testing.assert_array_equal(np.diag(result.values), np.zeros(len(OFFSETS)))
        assert not result.values.flags.writeable

    def test_the_order_of_the_poses_does_not_change_the_output(self, translated_set):
        forward = pairwise_peptide_rmsd(translated_set, "B", kind="heavy")
        backward = pairwise_peptide_rmsd(
            list(reversed(list(translated_set.items()))), "B", kind="heavy"
        )
        assert json.dumps(forward.to_dict(include_pairs=True)) == json.dumps(
            backward.to_dict(include_pairs=True)
        )

    def test_the_dictionary_is_json_ready_without_nan(self, translated_set):
        out = _json_strict(pairwise_peptide_rmsd(translated_set, "B").to_dict(include_pairs=True))
        assert out["atom_kind"] == "ca"
        assert out["unit"] == "angstrom"
        assert len(out["pairs"]) == 15
        assert out["undefined_pairs"] == []

    def test_a_pair_reports_how_it_was_computed(self, translated_set):
        pair = peptide_rmsd(translated_set["a1"], translated_set["b1"], "B", kind="heavy")
        assert pair.rmsd_angstrom == pytest.approx(2.0, abs=TOL)
        assert (pair.n_atoms, pair.n_residues, pair.shared_atom_fraction) == (113, 13, 1.0)
        assert (pair.pairing, pair.shift, pair.n_swapped_groups) == ("key", 0, 0)
        assert pair.unknown_residues == ()
        assert "reason" not in pair.to_dict()

    def test_the_receptor_does_not_matter(self, p53):
        other = p53.copy()
        other.coord[other.chain_id == "A"] += 50.0
        assert peptide_rmsd(p53, other, "B").rmsd_angstrom == pytest.approx(0.0, abs=TOL)

    def test_the_numbers_of_a_known_set_of_six_poses_are_sensible(self, translated_set):
        # the shape of the worked example: two tight groups that are about 2 angstrom apart
        stats = pair_statistics(
            pairwise_peptide_rmsd(translated_set, "B"), {k: k[0] for k in translated_set}
        )
        assert stats["within"]["a"]["max"] < 1.0 < stats["between"]["a|b"]["min"]


class TestRotation:
    def test_a_turn_with_a_shift_matches_the_closed_form_for_alpha_carbons(self, p53):
        # RMSD^2 = |t|^2 + 4 sin^2(theta / 2) <d_perp^2> when the axis passes through the centroid
        angle, shift = 20.0, np.array([1.0, 2.0, 0.0])
        pose = rotated(p53, "B", angle, shift)
        ca = p53.coord[(p53.chain_id == "B") & (p53.atom_name == "CA")].astype(np.float64)
        centered = ca - ca.mean(axis=0)
        perpendicular_squared = np.mean(centered[:, 0] ** 2 + centered[:, 1] ** 2)
        expected = np.sqrt(
            np.sum(shift**2) + 4 * np.sin(np.radians(angle) / 2) ** 2 * perpendicular_squared
        )
        assert peptide_rmsd(p53, pose, "B", kind="ca").rmsd_angstrom == pytest.approx(
            expected, abs=1e-4
        )

    @pytest.mark.parametrize("kind", ["ca", "backbone", "heavy"])
    def test_the_shape_rmsd_of_a_rigid_motion_is_zero(self, p53, kind):
        pose = rotated(p53, "B", 35.0, (3.0, -2.0, 1.0))
        plain = peptide_rmsd(p53, pose, "B", kind=kind).rmsd_angstrom
        shape = peptide_rmsd(p53, pose, "B", kind=kind, refit=True).rmsd_angstrom
        assert plain > 1.0
        assert shape == pytest.approx(0.0, abs=1e-4)

    def test_the_shape_rmsd_agrees_with_the_kabsch_rmsd_of_the_toolkit(self, p53):
        from binding_metrics.metrics.comparison import _kabsch_rmsd

        rng = np.random.default_rng(7)
        noisy = p53.copy()
        mask = noisy.chain_id == "B"
        noisy.coord[mask] += rng.normal(0.0, 0.6, size=noisy.coord[mask].shape).astype(np.float32)
        ca = (p53.chain_id == "B") & (p53.atom_name == "CA")
        expected = _kabsch_rmsd(
            p53.coord[ca].astype(np.float64), noisy.coord[ca].astype(np.float64)
        )
        got = peptide_rmsd(p53, noisy, "B", kind="ca", refit=True).rmsd_angstrom
        assert got == pytest.approx(expected, abs=1e-5)
        assert got < peptide_rmsd(p53, noisy, "B", kind="ca").rmsd_angstrom

    def test_a_fit_needs_three_atoms(self):
        one = build_atoms([("B", 1, "", "ALA", "CA", (0, 0, 0))])
        other = build_atoms([("B", 1, "", "ALA", "CA", (1, 0, 0))])
        pair = peptide_rmsd(one, other, "B", refit=True)
        assert pair.rmsd_angstrom is None
        assert "at least 3 atoms" in pair.reason
        assert peptide_rmsd(one, other, "B").rmsd_angstrom == pytest.approx(1.0)


class TestMetricProperties:
    def test_symmetry_nonnegativity_and_the_triangle_inequality_without_symmetry(self, p53):
        rng = np.random.default_rng(2026)
        poses = {}
        for index in range(6):
            pose = p53.copy()
            mask = pose.chain_id == "B"
            pose.coord[mask] += rng.normal(0.0, 0.5, size=pose.coord[mask].shape).astype(np.float32)
            poses[f"p{index}"] = pose
        values = pairwise_peptide_rmsd(poses, "B", kind="heavy", symmetry="off").values
        np.testing.assert_allclose(values, values.T, atol=1e-12)
        assert (values >= 0).all() and (np.diag(values) == 0).all()
        n = len(values)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert values[i, k] <= values[i, j] + values[j, k] + 1e-9


# ---------------------------------------------------------------------------
# symmetric atoms
# ---------------------------------------------------------------------------


class TestSymmetricAtoms:
    @pytest.fixture
    def residue_numbers(self, p53):
        numbers = _numbers(p53, "B")
        names = dict(zip(p53.res_id[p53.chain_id == "B"], p53.res_name[p53.chain_id == "B"]))
        return {name: [n for n in numbers if names[n] == name] for name in set(names.values())}

    def test_a_swapped_aspartate_is_the_same_pose_with_the_table_and_not_without(
        self, p53, residue_numbers
    ):
        (asp,) = residue_numbers["ASP"]
        pose = swapped_names(p53, "B", asp, ("OD1", "OD2"))
        standard = peptide_rmsd(p53, pose, "B", kind="heavy", symmetry="standard")
        off = peptide_rmsd(p53, pose, "B", kind="heavy", symmetry="off")
        assert standard.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
        assert standard.n_swapped_groups == 1
        # two atoms move by the distance between the oxygens (about 2.2 angstrom) over 113 atoms
        oxygens = p53[(p53.chain_id == "B") & (p53.res_id == asp)]
        gap = np.linalg.norm(
            oxygens.coord[oxygens.atom_name == "OD1"][0]
            - oxygens.coord[oxygens.atom_name == "OD2"][0]
        )
        assert off.rmsd_angstrom == pytest.approx(gap * np.sqrt(2 / 113), abs=1e-4)
        assert off.n_swapped_groups == 0

    def test_every_listed_kind_of_swap_is_undone(self, p53, residue_numbers):
        (asp,) = residue_numbers["ASP"]
        leucine = residue_numbers["LEU"][0]
        phenylalanine = residue_numbers["PHE"][0]
        glutamate = residue_numbers["GLU"][0]
        pose = swapped_names(p53, "B", asp, ("OD1", "OD2"))
        pose = swapped_names(pose, "B", leucine, ("CD1", "CD2"))
        pose = swapped_names(pose, "B", phenylalanine, ("CD1", "CD2"), ("CE1", "CE2"))
        pose = swapped_names(pose, "B", glutamate, ("OE1", "OE2"))
        standard = peptide_rmsd(p53, pose, "B", kind="heavy")
        assert standard.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
        assert standard.n_swapped_groups == 4
        assert peptide_rmsd(p53, pose, "B", kind="heavy", symmetry="off").rmsd_angstrom > 0.3

    def test_the_default_is_the_standard_table(self, p53, residue_numbers):
        (asp,) = residue_numbers["ASP"]
        pose = swapped_names(p53, "B", asp, ("OD1", "OD2"))
        assert peptide_rmsd(p53, pose, "B", kind="heavy").rmsd_angstrom == pytest.approx(
            0.0, abs=1e-6
        )

    def test_alpha_carbon_and_backbone_rmsd_do_not_depend_on_the_labels(self, p53, residue_numbers):
        pose = swapped_names(p53, "B", residue_numbers["LEU"][0], ("CD1", "CD2"))
        for kind in ("ca", "backbone"):
            for symmetry in ("standard", "off"):
                pair = peptide_rmsd(p53, pose, "B", kind=kind, symmetry=symmetry)
                assert pair.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
                assert pair.unknown_residues == ()

    def test_a_ring_flip_is_one_group_and_half_a_flip_is_not_undone(self, p53, residue_numbers):
        phenylalanine = residue_numbers["PHE"][0]
        half = swapped_names(p53, "B", phenylalanine, ("CD1", "CD2"))
        pair = peptide_rmsd(p53, half, "B", kind="heavy")
        assert pair.rmsd_angstrom > 0.05
        assert pair.n_swapped_groups == 0

    def test_the_best_of_two_labellings_is_chosen_not_just_any_swap(self, p53, residue_numbers):
        # a residue whose labels are already best is not swapped
        (asp,) = residue_numbers["ASP"]
        pair = peptide_rmsd(p53, p53, "B", kind="heavy")
        assert pair.n_swapped_groups == 0
        assert pair.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
        assert asp in _numbers(p53, "B")

    def test_the_shape_rmsd_also_uses_the_table(self, p53, residue_numbers):
        (asp,) = residue_numbers["ASP"]
        pose = swapped_names(rotated(p53, "B", 25.0, (1.0, 0.5, -2.0)), "B", asp, ("OD1", "OD2"))
        standard = peptide_rmsd(p53, pose, "B", kind="heavy", refit=True)
        off = peptide_rmsd(p53, pose, "B", kind="heavy", refit=True, symmetry="off")
        assert standard.rmsd_angstrom == pytest.approx(0.0, abs=2e-4)
        assert off.rmsd_angstrom > 0.1

    def test_phosphate_oxygens_and_a_tyrosine_ring_in_a_phosphopeptide(self, phospho):
        chain = "Q"
        numbers = _numbers(phospho, chain)
        names = dict(zip(phospho.res_id, phospho.res_name))
        (sep,) = [n for n in numbers if names[n] == "SEP"]
        (tyr,) = [n for n in numbers if names[n] == "TYR"]
        pose = swapped_names(phospho, chain, sep, ("O1P", "O2P"))
        pose = swapped_names(pose, chain, tyr, ("CD1", "CD2"), ("CE1", "CE2"))
        standard = peptide_rmsd(phospho, pose, chain, kind="heavy")
        assert standard.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
        assert standard.n_swapped_groups == 2
        assert peptide_rmsd(phospho, pose, chain, kind="heavy", symmetry="off").rmsd_angstrom > 0.1

    def test_a_d_residue_and_an_n_methyl_residue_use_the_entry_of_their_parent(self, csa):
        chain = "C"
        names = dict(zip(csa.res_id[csa.chain_id == chain], csa.res_name[csa.chain_id == chain]))
        (valine_methyl,) = [n for n, name in names.items() if name == "MVA"]
        leucines = [n for n, name in names.items() if name == "MLE"]
        pose = swapped_names(csa, chain, valine_methyl, ("CG1", "CG2"))
        pose = swapped_names(pose, chain, leucines[0], ("CD1", "CD2"))
        standard = peptide_rmsd(csa, pose, chain, kind="heavy")
        assert standard.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
        assert standard.n_swapped_groups == 2
        assert peptide_rmsd(csa, pose, chain, kind="heavy", symmetry="off").rmsd_angstrom > 0.05

    def test_a_d_leucine_is_swapped_like_a_leucine(self, p53):
        numbers = _numbers(p53, "B")
        names = dict(zip(p53.res_id[p53.chain_id == "B"], p53.res_name[p53.chain_id == "B"]))
        leucine = next(n for n in numbers if names[n] == "LEU")
        first = renamed_residue(p53, "B", leucine, "DLE")
        second = swapped_names(first, "B", leucine, ("CD1", "CD2"))
        pair = peptide_rmsd(first, second, "B", kind="heavy")
        assert pair.rmsd_angstrom == pytest.approx(0.0, abs=1e-6)
        assert pair.n_swapped_groups == 1

    def test_unknown_residues_are_compared_as_named_and_listed(self, csa):
        pose = moved(csa, "C", [0.5, 0.0, 0.0])
        heavy = pairwise_peptide_rmsd({"a": csa, "b": pose}, "C", kind="heavy")
        assert heavy.unknown_residues == ("ABA", "BMT")
        assert heavy.pairs[("a", "b")].unknown_residues == ("ABA", "BMT")
        assert heavy.value("a", "b") == pytest.approx(0.5, abs=TOL)
        assert _json_strict(heavy.to_dict())["unknown_residues"] == ["ABA", "BMT"]
        # nothing is listed where the table plays no part
        for kind in ("ca", "backbone"):
            assert (
                pairwise_peptide_rmsd({"a": csa, "b": pose}, "C", kind=kind).unknown_residues == ()
            )
        off = pairwise_peptide_rmsd({"a": csa, "b": pose}, "C", kind="heavy", symmetry="off")
        assert off.unknown_residues == ()

    def test_an_unknown_residue_is_not_swapped(self, csa):
        chain = "C"
        names = dict(zip(csa.res_id[csa.chain_id == chain], csa.res_name[csa.chain_id == chain]))
        (bmt,) = [n for n, name in names.items() if name == "BMT"]
        pose = swapped_names(csa, chain, bmt, ("CD1", "CD2"))
        pair = peptide_rmsd(csa, pose, chain, kind="heavy")
        assert pair.rmsd_angstrom > 0.1
        assert pair.n_swapped_groups == 0
        assert "BMT" in pair.unknown_residues


# ---------------------------------------------------------------------------
# residue correspondence and missing residues
# ---------------------------------------------------------------------------


class TestResidueCorrespondence:
    def test_the_binder_lists_residues_by_key_without_waters_or_hydrogens(self, csa):
        view = BinderView.from_atoms(csa, "C", "csa")
        assert [r.name for r in view.residues] == [
            "DAL", "MLE", "MLE", "MVA", "BMT", "ABA", "SAR", "MLE", "VAL", "MLE", "ALA",
        ]  # fmt: skip
        assert view.keys_unique
        assert view.residues[0].key == ("C", 1, "")
        expected_atoms = int(
            np.sum((csa.chain_id == "C") & (csa.res_name != "HOH") & (csa.element != "H"))
        )
        assert view.coord.shape == (expected_atoms, 3)

    def test_waters_of_the_binder_chain_do_not_take_part(self, csa):
        pose = moved(csa, "C", [0.0, 1.0, 0.0])
        pair = peptide_rmsd(csa, pose, "C", kind="heavy")
        waters = (csa.chain_id == "C") & (csa.res_name == "HOH")
        assert waters.any()
        assert pair.n_atoms == int(np.sum((csa.chain_id == "C") & ~waters))
        assert pair.rmsd_angstrom == pytest.approx(1.0, abs=TOL)

    def test_a_missing_residue_is_dropped_and_the_rest_compared(self, p53):
        numbers = _numbers(p53, "B")
        missing = numbers[6]
        pose = without_residue(moved(p53, "B", [0.0, 0.0, 1.5]), "B", missing)
        for kind in ("ca", "backbone", "heavy"):
            pair = peptide_rmsd(p53, pose, "B", kind=kind)
            assert pair.rmsd_angstrom == pytest.approx(1.5, abs=TOL)
            assert pair.pairing == "key"
            assert pair.n_residues == 12
            assert 0.8 <= pair.shared_atom_fraction < 1.0

    def test_the_share_of_missing_atoms_is_set_by_min_shared_atom_fraction(self, p53):
        numbers = _numbers(p53, "B")
        pose = without_residue(p53, "B", numbers[6])
        loose = peptide_rmsd(p53, pose, "B", kind="ca", min_shared_atom_fraction=0.9)
        assert loose.rmsd_angstrom is not None
        strict = peptide_rmsd(p53, pose, "B", kind="ca", min_shared_atom_fraction=0.95)
        assert strict.rmsd_angstrom is None
        assert "only 12 of 13 atoms are shared" in strict.reason
        assert strict.shared_atom_fraction == pytest.approx(12 / 13)

    def test_a_pose_that_lost_its_side_chains_has_no_heavy_atom_rmsd_but_has_a_backbone_one(
        self, p53
    ):
        numbers = _numbers(p53, "B")
        pose = without_atoms(p53, "B", numbers[:8])
        heavy = peptide_rmsd(p53, pose, "B", kind="heavy")
        assert heavy.rmsd_angstrom is None
        assert "atoms are shared" in heavy.reason and "min_shared_atom_fraction" in heavy.reason
        assert peptide_rmsd(p53, pose, "B", kind="backbone").rmsd_angstrom == pytest.approx(
            0.0, abs=TOL
        )
        assert peptide_rmsd(p53, pose, "B", kind="ca").rmsd_angstrom == pytest.approx(0.0, abs=TOL)

    def test_a_few_missing_atoms_are_dropped_from_the_pair(self, p53):
        numbers = _numbers(p53, "B")
        pose = without_atoms(moved(p53, "B", [2.0, 0.0, 0.0]), "B", numbers[:2], names=["CB", "CG"])
        pair = peptide_rmsd(p53, pose, "B", kind="heavy")
        assert pair.rmsd_angstrom == pytest.approx(2.0, abs=TOL)
        assert pair.n_atoms < 113

    def test_a_different_numbering_is_read_by_position_under_auto(self, p53):
        pose = renumbered(moved(p53, "B", [1.0, 0.0, 0.0]), "B", 100)
        pair = peptide_rmsd(p53, pose, "B", kind="heavy")
        assert pair.pairing == "position"
        assert pair.rmsd_angstrom == pytest.approx(1.0, abs=TOL)

    def test_a_numbering_shifted_by_one_is_not_read_by_key(self, p53):
        pose = renumbered(moved(p53, "B", [1.0, 0.0, 0.0]), "B", 1)
        assert peptide_rmsd(p53, pose, "B").pairing == "position"
        assert peptide_rmsd(
            p53, pose, "B", residue_pairing="position"
        ).rmsd_angstrom == pytest.approx(1.0, abs=TOL)

    def test_key_pairing_needs_shared_keys(self, p53):
        pose = renumbered(p53, "B", 100)
        pair = peptide_rmsd(p53, pose, "B", residue_pairing="key")
        assert pair.rmsd_angstrom is None
        assert "no residue key is shared" in pair.reason
        assert pair.pairing is None

    def test_position_pairing_needs_equal_lengths(self, p53):
        pose = without_residue(p53, "B", _numbers(p53, "B")[-1])
        pair = peptide_rmsd(p53, pose, "B", residue_pairing="position")
        assert pair.rmsd_angstrom is None
        assert "binder lengths differ (13 residues" in pair.reason

    def test_two_poses_that_each_lack_residues_have_no_correspondence_under_auto(self, p53):
        numbers = _numbers(p53, "B")
        first = without_residue(p53, "B", numbers[2])
        second = without_residue(p53, "B", numbers[8])
        pair = peptide_rmsd(first, second, "B")
        assert pair.rmsd_angstrom is None
        assert pair.reason.startswith("no residue pairing")
        # by key the shared residues can still be compared if the caller asks for it
        assert peptide_rmsd(
            first, second, "B", residue_pairing="key"
        ).rmsd_angstrom == pytest.approx(0.0, abs=TOL)

    def test_insertion_codes_tell_residues_apart(self, p53):
        numbers = _numbers(p53, "B")
        pose = p53.copy()
        target = (pose.chain_id == "B") & (pose.res_id == numbers[4])
        pose.ins_code[target] = "A"
        by_key = peptide_rmsd(p53, pose, "B", kind="ca", residue_pairing="key")
        assert (
            by_key.n_residues == 12
        )  # residue 5 and 5A are two residues, each missing in the other
        assert peptide_rmsd(p53, pose, "B", kind="ca").rmsd_angstrom == pytest.approx(0.0, abs=TOL)

    def test_a_changed_sequence_is_refused_with_the_residue_named(self, p53):
        numbers = _numbers(p53, "B")
        pose = renamed_residue(p53, "B", numbers[4], "GLU")
        pair = peptide_rmsd(p53, pose, "B")
        assert pair.rmsd_angstrom is None
        assert "is ASP" in pair.reason and "but" in pair.reason and "GLU" in pair.reason

    def test_a_d_form_and_a_protonation_variant_are_the_same_residue(self, phospho):
        chain = "Q"
        numbers = _numbers(phospho, chain)
        names = dict(zip(phospho.res_id, phospho.res_name))
        histidine = next(n for n in numbers if names[n] == "HIS")
        arginine = next(n for n in numbers if names[n] == "ARG")
        pose = renamed_residue(
            renamed_residue(phospho, chain, histidine, "HID"), chain, arginine, "DAR"
        )
        pair = peptide_rmsd(phospho, pose, chain, kind="heavy")
        assert pair.rmsd_angstrom == pytest.approx(0.0, abs=TOL)

    def test_an_n_methyl_residue_is_not_the_residue_it_methylates(self, p53):
        numbers = _numbers(p53, "B")
        names = dict(zip(p53.res_id[p53.chain_id == "B"], p53.res_name[p53.chain_id == "B"]))
        leucine = next(n for n in numbers if names[n] == "LEU")
        pair = peptide_rmsd(p53, renamed_residue(p53, "B", leucine, "MLE"), "B")
        assert pair.rmsd_angstrom is None
        assert "MLE" in pair.reason

    def test_chains_may_have_any_id_and_other_chains_are_ignored(self, p53):
        pose = p53.copy()
        pose.chain_id[pose.chain_id == "B"] = "Z"
        assert peptide_rmsd(pose, pose, "Z").n_atoms == 13
        assert peptide_rmsd(p53, p53, "B").n_atoms == 13


# ---------------------------------------------------------------------------
# cyclic peptides
# ---------------------------------------------------------------------------


class TestRings:
    def test_a_ring_is_closed_and_a_linear_peptide_is_not(self, csa, sfti, p53):
        assert BinderView.from_atoms(csa, "C").closed_head_to_tail
        assert BinderView.from_atoms(sfti, "I").closed_head_to_tail
        assert not BinderView.from_atoms(p53, "B").closed_head_to_tail

    def test_the_rings_have_no_terminal_residue_dropped(self, csa, sfti):
        assert len(BinderView.from_atoms(csa, "C").residues) == 11
        assert len(BinderView.from_atoms(sfti, "I").residues) == 14

    @pytest.mark.parametrize(("name", "chain"), [(CSA, "C"), (SFTI, "I")])
    @pytest.mark.parametrize("kind", ["ca", "backbone", "heavy"])
    def test_a_displaced_ring_has_the_offset_as_rmsd_in_every_kind(self, name, chain, kind):
        atoms = _load(name)
        pose = moved(atoms, chain, [0.0, 0.0, 1.25])
        pair = peptide_rmsd(atoms, pose, chain, kind=kind)
        assert pair.rmsd_angstrom == pytest.approx(1.25, abs=TOL)

    def test_the_atoms_that_close_the_ring_are_in_the_sets(self, csa, sfti):
        # C of the last residue and N of the first (head to tail), SG and SG (disulfide)
        view = BinderView.from_atoms(sfti, "I")
        assert "N" in view.residues[0].atom_rows and "C" in view.residues[-1].atom_rows
        cysteines = [r for r in view.residues if r.name == "CYS"]
        assert len(cysteines) == 2 and all("SG" in r.atom_rows for r in cysteines)
        # moving only these atoms changes the backbone and heavy-atom RMSD, not the alpha carbons
        pose = sfti.copy()
        chain = pose.chain_id == "I"
        last_c = (
            chain
            & (pose.res_id == pose.res_id[chain & (pose.res_name != "HOH")][-1])
            & (pose.atom_name == "C")
        )
        pose.coord[last_c] += np.array([0.0, 0.0, 3.0], dtype=np.float32)
        assert peptide_rmsd(sfti, pose, "I", kind="ca").rmsd_angstrom == pytest.approx(0.0, abs=TOL)
        assert peptide_rmsd(sfti, pose, "I", kind="backbone").rmsd_angstrom == pytest.approx(
            3.0 / np.sqrt(56), abs=TOL
        )
        sg = chain & (pose.atom_name == "SG")
        pose2 = sfti.copy()
        pose2.coord[sg] += np.array([0.0, 2.0, 0.0], dtype=np.float32)
        heavy = peptide_rmsd(sfti, pose2, "I", kind="heavy")
        assert heavy.rmsd_angstrom == pytest.approx(2.0 * np.sqrt(2 / heavy.n_atoms), abs=TOL)

    def test_the_ring_origin_matters_unless_the_shift_is_asked_for(self, csa):
        rotated_ring = ring_origin_moved(csa, "C", 3)
        reference = ring_origin_moved(csa, "C", 0)
        off = peptide_rmsd(reference, rotated_ring, "C", kind="heavy")
        assert off.rmsd_angstrom is None
        assert off.reason.startswith("no residue pairing")
        auto = peptide_rmsd(reference, rotated_ring, "C", kind="heavy", cyclic_shift="auto")
        assert auto.rmsd_angstrom == pytest.approx(0.0, abs=TOL)
        assert auto.shift == 11 - 3
        assert auto.pairing == "position"

    def test_a_bicyclic_ring_with_a_new_origin_is_found_for_every_kind(self, sfti):
        reference = ring_origin_moved(sfti, "I", 0)
        other = ring_origin_moved(sfti, "I", 5)
        for kind in ("ca", "backbone", "heavy"):
            pair = peptide_rmsd(reference, other, "I", kind=kind, cyclic_shift="auto")
            assert pair.rmsd_angstrom == pytest.approx(0.0, abs=TOL)
            assert pair.shift == 14 - 5

    def test_the_shift_is_the_smallest_rmsd_and_a_tie_goes_to_the_smallest_shift(self):
        ring = ring_of_alanines(4)
        assert BinderView.from_atoms(ring, "B").closed_head_to_tail
        # the regular ring is invariant under the shifts: every shift gives 0, so the first wins
        pair = peptide_rmsd(ring, ring, "B", kind="heavy", cyclic_shift="auto")
        assert pair.rmsd_angstrom == pytest.approx(0.0, abs=TOL)
        assert pair.shift == 0
        origin_moved = ring_origin_moved(ring, "B", 1)
        plain = peptide_rmsd(ring, origin_moved, "B", kind="heavy")
        assert plain.rmsd_angstrom > 1.0
        found = peptide_rmsd(ring, origin_moved, "B", kind="heavy", cyclic_shift="auto")
        assert found.rmsd_angstrom == pytest.approx(0.0, abs=TOL)
        assert found.shift == 3

    def test_a_linear_peptide_is_never_shifted(self, p53):
        reference = ring_origin_moved(p53, "B", 0)
        other = ring_origin_moved(p53, "B", 4)
        pair = peptide_rmsd(reference, other, "B", cyclic_shift="auto")
        assert pair.shift == 0
        assert pair.rmsd_angstrom is None  # a different origin is a different sequence

    def test_shifting_leaves_an_ordinary_pair_unchanged(self, csa):
        pose = moved(csa, "C", [0.7, 0.0, 0.0])
        off = peptide_rmsd(csa, pose, "C", kind="heavy")
        auto = peptide_rmsd(csa, pose, "C", kind="heavy", cyclic_shift="auto")
        assert auto.rmsd_angstrom == pytest.approx(off.rmsd_angstrom, abs=1e-9)
        assert auto.shift == 0

    def test_closure_families_of_the_examples(self, csa, sfti, p53):
        assert binder_closure_families(csa, "C") == {"head_to_tail"}
        assert binder_closure_families(sfti, "I") == {"head_to_tail", "disulfide"}
        assert binder_closure_families(p53, "B") == frozenset()

    def test_a_pair_with_different_closures_is_flagged_and_can_be_excluded(self, sfti):
        pose = moved(sfti, "I", [0.0, 1.0, 0.0])
        families = {
            "a": {"head_to_tail", "disulfide"},
            "b": {"head_to_tail"},
            "c": {"head_to_tail", "disulfide"},
        }
        poses = {"a": sfti, "b": pose, "c": pose}
        flagged = pairwise_peptide_rmsd(poses, "I", closure_families=families)
        assert flagged.pairs[("a", "b")].closure_mismatch is True
        assert flagged.pairs[("a", "c")].closure_mismatch is False
        assert flagged.value("a", "b") == pytest.approx(1.0, abs=TOL)
        excluded = pairwise_peptide_rmsd(
            poses, "I", closure_families=families, exclude_closure_mismatch=True
        )
        assert excluded.value("a", "b") is None
        assert "closure families differ" in excluded.reason("a", "b")
        assert excluded.value("a", "c") == pytest.approx(1.0, abs=TOL)

    def test_closure_mismatch_is_unknown_without_families(self, sfti):
        pair = peptide_rmsd(sfti, sfti, "I")
        assert pair.closure_mismatch is None


# ---------------------------------------------------------------------------
# failures, options, small sets
# ---------------------------------------------------------------------------


class TestFailuresAndOptions:
    def test_a_pose_without_the_binder_chain_has_no_value_and_the_others_are_computed(self, p53):
        no_binder = p53[p53.chain_id == "A"]
        poses = {"a": p53, "b": no_binder, "c": moved(p53, "B", [1.0, 0.0, 0.0])}
        result = pairwise_peptide_rmsd(poses, "B")
        assert "b" in result.pose_reasons
        assert "no atom of chain 'B'" in result.pose_reasons["b"]
        assert result.value("a", "b") is None and result.value("b", "c") is None
        assert "no atom of chain" in result.reason("a", "b")
        assert result.value("a", "c") == pytest.approx(1.0, abs=TOL)
        out = _json_strict(result.to_dict())
        assert out["matrix"][0][1] is None
        assert {tuple(item["pair"]) for item in out["undefined_pairs"]} == {("a", "b"), ("b", "c")}

    def test_a_stack_is_refused_for_that_pose_only(self, p53):
        stack = struc.stack([p53, p53])
        result = pairwise_peptide_rmsd({"a": p53, "b": stack}, "B")
        assert "stack" in result.pose_reasons["b"]
        assert result.value("a", "b") is None

    def test_a_binder_chain_of_waters_only_is_refused(self, p53):
        waters = p53.copy()
        waters.res_name[waters.chain_id == "B"] = "HOH"
        result = pairwise_peptide_rmsd({"a": p53, "b": waters}, "B")
        assert "no heavy atom outside water" in result.pose_reasons["b"]

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("kind", "all"),
            ("symmetry", "full"),
            ("residue_pairing", "index"),
            ("cyclic_shift", "on"),
            ("min_shared_atom_fraction", 0.0),
            ("min_shared_atom_fraction", 1.5),
        ],
    )
    def test_bad_options_raise_before_any_pose_is_read(self, p53, option, value):
        with pytest.raises(ValueError, match=option):
            pairwise_peptide_rmsd({"a": p53}, "B", **{option: value})

    def test_repeated_ids_raise(self, p53):
        with pytest.raises(ValueError, match="not unique"):
            pairwise_peptide_rmsd([("a", p53), ("a", p53)], "B")

    def test_ids_must_be_non_empty_strings(self, p53):
        with pytest.raises(TypeError, match="non-empty id"):
            pairwise_peptide_rmsd([(1, p53)], "B")

    def test_one_pose_gives_a_one_by_one_matrix(self, p53):
        result = pairwise_peptide_rmsd({"only": p53}, "B")
        assert result.values.shape == (1, 1)
        assert result.pairs == {}
        assert result.value("only", "only") == 0.0

    def test_no_pose_gives_an_empty_matrix(self):
        result = pairwise_peptide_rmsd({}, "B")
        assert result.values.shape == (0, 0)
        assert result.to_dict()["matrix"] == []

    def test_parameters_are_echoed(self, p53):
        result = pairwise_peptide_rmsd({"a": p53}, "B", kind="heavy", cyclic_shift="auto")
        assert result.parameters == {
            "kind": "heavy",
            "symmetry": "standard",
            "residue_pairing": "auto",
            "cyclic_shift": "auto",
            "refit": False,
            "min_shared_atom_fraction": 0.8,
            "exclude_closure_mismatch": False,
        }

    def test_the_matrix_lookup_names_a_missing_pose(self, p53):
        result = pairwise_peptide_rmsd({"a": p53}, "B")
        with pytest.raises(KeyError, match="no pose 'z'"):
            result.value("a", "z")

    def test_the_module_states_what_cyclic_aware_means(self):
        doc = agreement.__doc__
        for phrase in (
            "No terminus is assumed",
            "Closure atoms are ordinary atoms",
            "no chain break",
        ):
            assert phrase in doc
        assert "does not do these things" in doc


# ---------------------------------------------------------------------------
# statistics over groups
# ---------------------------------------------------------------------------


class TestPairStatistics:
    @pytest.fixture
    def matrix(self, p53):
        poses = {name: moved(p53, "B", [offset, 0.0, 0.0]) for name, offset in OFFSETS.items()}
        return pairwise_peptide_rmsd(poses, "B")

    def test_within_and_between_are_the_analytic_values(self, matrix):
        groups = {name: name[0] for name in OFFSETS}
        stats = pair_statistics(matrix, groups)
        a, b, between = stats["within"]["a"], stats["within"]["b"], stats["between"]["a|b"]
        assert (a["n_pairs"], b["n_pairs"], between["n_pairs"]) == (3, 3, 9)
        assert (a["min"], a["mean"], a["max"]) == pytest.approx((0.4, 0.8 / 1.5, 0.8), abs=TOL)
        assert (b["min"], b["mean"], b["max"]) == pytest.approx((0.3, 0.6, 0.9), abs=TOL)
        assert (between["min"], between["mean"], between["max"]) == pytest.approx(
            (1.2, 2.0, 2.9), abs=TOL
        )
        assert between["min_pair"] == ["a3", "b1"] and between["max_pair"] == ["a1", "b3"]
        assert a["max_pair"] == ["a1", "a3"]
        assert stats["overall"]["n_pairs"] == 15
        assert stats["between_reason"] is None

    def test_one_group_has_no_between_block_and_says_why(self, matrix):
        stats = pair_statistics(matrix)
        assert list(stats["within"]) == ["all"]
        assert stats["between"] == {}
        assert stats["between_reason"] == "only one group"
        assert stats["within"]["all"]["n_pairs"] == 15

    def test_a_group_of_one_pose_has_no_statistic_and_a_reason(self, matrix):
        groups = {name: ("big" if name != "b3" else "single") for name in OFFSETS}
        stats = pair_statistics(matrix, groups)
        single = stats["within"]["single"]
        assert single["n_pairs"] == 0
        assert (single["min"], single["mean"], single["max"]) == (None, None, None)
        assert single["reason"] == "needs at least 2 poses (group 'single' has 1)"
        assert stats["between"]["big|single"]["n_pairs"] == 5

    def test_one_pose_gives_none_with_a_reason(self, p53):
        stats = pair_statistics(pairwise_peptide_rmsd({"only": p53}, "B"))
        overall = stats["overall"]
        assert overall["mean"] is None
        assert overall["reason"] == "needs at least 2 poses (got 1)"
        assert stats["within"]["all"]["reason"].startswith("needs at least 2 poses")

    def test_two_poses_give_one_pair(self, p53):
        result = pairwise_peptide_rmsd({"a": p53, "b": moved(p53, "B", [0.0, 0.0, 0.6])}, "B")
        overall = pair_statistics(result)["overall"]
        assert (overall["n_pairs"], overall["n_defined"]) == (1, 1)
        assert overall["min"] == overall["mean"] == overall["max"]
        assert overall["min"] == pytest.approx(0.6, abs=TOL)
        assert overall["min_pair"] == overall["max_pair"] == ["a", "b"]
        split = pair_statistics(result, {"a": "x", "b": "y"})
        assert split["within"]["x"]["min"] is None
        assert split["between"]["x|y"]["min"] == pytest.approx(0.6, abs=TOL)

    def test_undefined_pairs_are_counted_and_left_out(self, p53):
        poses = {
            "a": p53,
            "b": moved(p53, "B", [1.0, 0.0, 0.0]),
            "c": p53[p53.chain_id == "A"],
        }
        stats = pair_statistics(pairwise_peptide_rmsd(poses, "B"))
        overall = stats["overall"]
        assert (overall["n_pairs"], overall["n_defined"]) == (3, 1)
        assert overall["mean"] == pytest.approx(1.0, abs=TOL)

    def test_a_group_with_no_defined_pair_says_why(self, p53):
        poses = {"a": p53, "b": p53[p53.chain_id == "A"], "c": p53[p53.chain_id == "A"]}
        stats = pair_statistics(pairwise_peptide_rmsd(poses, "B"), {"a": "x", "b": "y", "c": "y"})
        within = stats["within"]["y"]
        assert within["mean"] is None
        assert within["reason"].startswith("no pair has a value")
        _json_strict(stats)

    def test_every_pose_needs_a_group_label(self, matrix):
        with pytest.raises(ValueError, match="without a group label"):
            pair_statistics(matrix, {"a1": "a"})

    def test_subset_keeps_the_values_and_the_reasons(self, p53):
        poses = {"a": p53, "b": moved(p53, "B", [1.0, 0.0, 0.0]), "c": p53[p53.chain_id == "A"]}
        result = pairwise_peptide_rmsd(poses, "B")
        part = result.subset(["b", "a"])
        assert isinstance(part, PairMatrix) and part.pose_ids == ("a", "b")
        assert part.value("a", "b") == pytest.approx(1.0, abs=TOL)
        assert part.reasons == {}
        assert result.subset(["a", "c"]).reason("a", "c") is not None


def test_the_modules_import_without_openmm_biotite_or_mdtraj():
    script = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in ('openmm', 'simtk', 'mdtraj', 'pdbfixer', 'biotite'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import binding_metrics.poses.agreement, binding_metrics.poses.symmetry\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ---------------------------------------------------------------------------
# hand-built distance matrices (float64, so that ties are exact)
# ---------------------------------------------------------------------------


def line_matrix(positions, ids=None):
    """Distances of points on a line, as a distance ``PairMatrix`` (ids sorted as given)."""
    ids = tuple(ids or [f"p{i}" for i in range(len(positions))])
    points = np.asarray(positions, dtype=np.float64)
    values = np.abs(points[:, None] - points[None, :])
    return PairMatrix("test", "angstrom", "distance", ids, values, {})


OFFSET_IDS = tuple(OFFSETS)
OFFSET_MATRIX = line_matrix(list(OFFSETS.values()), OFFSET_IDS)


def partition(result):
    """The clusters of a ``ClusterResult`` as a set of frozensets."""
    return {frozenset(cluster) for cluster in result.clusters}


# ---------------------------------------------------------------------------
# medoid
# ---------------------------------------------------------------------------


class TestMedoid:
    def test_the_medoid_of_each_group_is_the_pose_with_the_smallest_distance_sum(self):
        groups = {name: name[0] for name in OFFSETS}
        result = medoids(OFFSET_MATRIX, groups)
        # sums inside the groups: a1 1.2, a2 0.8, a3 1.2 and b1 1.2, b2 0.9, b3 1.5
        assert result["per_group"]["a"]["pose_id"] == "a2"
        assert result["per_group"]["a"]["sum_distance_angstrom"] == pytest.approx(0.8)
        assert result["per_group"]["b"]["pose_id"] == "b2"
        assert result["per_group"]["b"]["mean_distance_angstrom"] == pytest.approx(0.45)
        between = result["between_groups"]["a|b"]
        assert (between["medoid_a"], between["medoid_b"]) == ("a2", "b2")
        assert between["distance_angstrom"] == pytest.approx(1.9)

    def test_an_exact_tie_goes_to_the_lowest_id_and_names_the_others(self):
        # a3 and b1 both have a distance sum of 6.0 to the other five poses
        result = medoid(OFFSET_MATRIX)
        assert result["pose_id"] == "a3"
        assert result["sum_distance_angstrom"] == pytest.approx(6.0)
        assert result["mean_distance_angstrom"] == pytest.approx(1.2)
        assert result["tied_with"] == ["b1"]

    def test_a_tie_that_only_rounding_hides_is_still_a_tie(self):
        # p1 and p2 mirror each other, but their floating-point sums differ by 1e-16
        matrix = line_matrix([0.1, 0.2, 0.3, 0.4])
        assert matrix.values[1].sum() != matrix.values[2].sum()
        result = medoid(matrix)
        assert result["pose_id"] == "p1"
        assert result["tied_with"] == ["p2"]

    def test_all_poses_equidistant_gives_the_lowest_id(self):
        values = np.ones((4, 4)) - np.eye(4)
        matrix = PairMatrix("test", "angstrom", "distance", ("a", "b", "c", "d"), values, {})
        result = medoid(matrix)
        assert result["pose_id"] == "a"
        assert result["tied_with"] == ["b", "c", "d"]

    def test_two_poses_tie_and_the_lower_id_wins(self):
        result = medoid(line_matrix([0.0, 0.7], ["x", "y"]))
        assert result["pose_id"] == "x"
        assert result["tied_with"] == ["y"]
        assert result["mean_distance_angstrom"] == pytest.approx(0.7)

    def test_one_pose_is_its_own_medoid_with_a_reason(self):
        result = medoid(line_matrix([0.0], ["only"]))
        assert result["pose_id"] == "only"
        assert result["mean_distance_angstrom"] is None
        assert "own medoid" in result["reason"]

    def test_no_pose_has_no_medoid(self):
        result = medoid(line_matrix([], []))
        assert result["pose_id"] is None and result["reason"] == "no pose"

    def test_a_pose_with_an_undefined_distance_is_not_a_candidate(self):
        values = line_matrix([0.0, 1.0, 2.0, 3.0]).values.copy()
        values[0, 3] = values[3, 0] = np.nan
        matrix = PairMatrix(
            "test", "angstrom", "distance", ("a", "b", "c", "d"), values, {("a", "d"): "why"}
        )
        result = medoid(matrix)
        assert result["pose_id"] == "b"
        assert result["excluded"] == ["a", "d"]
        # leaving a pose out restores the others
        assert medoid(matrix.subset(["a", "b", "c"]))["excluded"] == []

    def test_no_candidate_gives_none_with_a_reason(self):
        values = np.array([[0.0, np.nan], [np.nan, 0.0]])
        matrix = PairMatrix("t", "angstrom", "distance", ("a", "b"), values, {("a", "b"): "x"})
        result = medoid(matrix)
        assert result["pose_id"] is None
        assert "every other pose" in result["reason"]

    def test_the_medoid_distance_between_groups_can_be_undefined(self):
        values = np.array([[0.0, np.nan], [np.nan, 0.0]])
        matrix = PairMatrix(
            "t", "angstrom", "distance", ("a", "b"), values, {("a", "b"): "no atoms"}
        )
        entry = medoids(matrix, {"a": "g", "b": "h"})["between_groups"]["g|h"]
        assert entry["distance_angstrom"] is None and entry["reason"] == "no atoms"

    def test_a_similarity_matrix_has_no_medoid(self):
        matrix = jaccard_matrix({"a": {1}, "b": {1, 2}})
        with pytest.raises(ValueError, match="distance matrix"):
            medoid(matrix)
        with pytest.raises(ValueError, match="distance matrix"):
            medoids(matrix)

    def test_poses_need_a_group_label(self):
        with pytest.raises(ValueError, match="without a group label"):
            medoids(OFFSET_MATRIX, {"a1": "a"})

    def test_the_medoid_of_computed_poses(self, translated_set):
        matrix = pairwise_peptide_rmsd(translated_set, "B")
        result = medoids(matrix, {name: name[0] for name in translated_set})
        assert (result["per_group"]["a"]["pose_id"], result["per_group"]["b"]["pose_id"]) == (
            "a2",
            "b2",
        )
        _json_strict(result)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


class TestClustering:
    def test_defaults_and_the_fixed_linkage(self):
        assert DEFAULT_CLUSTER_CUTOFFS_ANGSTROM == (1.0, 2.0)
        assert CLUSTER_LINKAGE == "complete"
        results = complete_linkage_clusters(OFFSET_MATRIX)
        assert [r.cutoff_angstrom for r in results] == [1.0, 2.0]

    def test_two_groups_are_two_clusters_at_both_default_cutoffs(self):
        # diameters 0.8 and 0.9, farthest between-pair 2.9: complete linkage keeps them apart
        for result in complete_linkage_clusters(OFFSET_MATRIX):
            assert result.n_clusters == 2
            assert partition(result) == {
                frozenset("a1 a2 a3".split()),
                frozenset("b1 b2 b3".split()),
            }
            assert result.sizes == (3, 3)
            assert result.largest_cluster_fraction == 0.5

    def test_a_larger_cutoff_joins_everything(self):
        (result,) = complete_linkage_clusters(OFFSET_MATRIX, [3.0])
        assert (result.n_clusters, result.sizes, result.largest_cluster_fraction) == (1, (6,), 1.0)

    def test_the_tie_at_a_small_cutoff_goes_to_the_lowest_ranks(self):
        # a1-a2 and a2-a3 are both 0.4: the pair with the lowest ranks, (a1, a2), merges first;
        # the pair b1-b2 (0.3) merges before either
        (result,) = complete_linkage_clusters(OFFSET_MATRIX, [0.5])
        assert result.clusters == (("a1", "a2"), ("b1", "b2"), ("a3",), ("b3",))
        assert result.sizes == (2, 2, 1, 1)
        assert result.largest_cluster_fraction == pytest.approx(2 / 6)
        assert result.labels == {"a1": 0, "a2": 0, "b1": 1, "b2": 1, "a3": 2, "b3": 3}

    def test_the_tie_does_not_depend_on_floating_point_rounding(self):
        # 0.7 - 0.4 is 0.29999999999999993 and 0.4 - 0.1 is 0.30000000000000004
        matrix = line_matrix([0.1, 0.4, 0.7])
        (result,) = complete_linkage_clusters(matrix, [0.35])
        assert result.clusters == (("p0", "p1"), ("p2",))

    def test_labels_follow_size_then_lowest_member(self):
        matrix = line_matrix([0.0, 10.0, 10.1, 20.0, 20.1, 20.2], list("abcdef"))
        (result,) = complete_linkage_clusters(matrix, [0.5])
        assert result.clusters == (("d", "e", "f"), ("b", "c"), ("a",))
        assert result.sizes == (3, 2, 1)

    def test_the_cutoff_itself_is_inside_the_cluster(self):
        assert complete_linkage_clusters(line_matrix([0.0, 1.0]), [1.0])[0].n_clusters == 1
        assert complete_linkage_clusters(line_matrix([0.0, 1.000001]), [1.0])[0].n_clusters == 2

    def test_complete_linkage_does_not_chain(self):
        # 0 - 0.9 - 1.8: single linkage would give one cluster at 1.0, complete linkage two
        (result,) = complete_linkage_clusters(line_matrix([0.0, 0.9, 1.8]), [1.0])
        assert partition(result) == {frozenset({"p0", "p1"}), frozenset({"p2"})}

    def test_an_undefined_distance_keeps_two_poses_apart(self):
        values = np.array([[0.0, np.nan, 0.1], [np.nan, 0.0, 0.1], [0.1, 0.1, 0.0]])
        matrix = PairMatrix(
            "test", "angstrom", "distance", ("p0", "p1", "p2"), values, {("p0", "p1"): "no atoms"}
        )
        (result,) = complete_linkage_clusters(matrix, [1.0])
        assert result.clusters == (("p0", "p2"), ("p1",))
        assert result.n_undefined_pairs == 1

    def test_clusters_that_mix_groups_are_listed(self):
        groups = {"a1": "x", "a2": "x", "a3": "y", "b1": "y", "b2": "y", "b3": "y"}
        (result,) = complete_linkage_clusters(OFFSET_MATRIX, [1.0], groups=groups)
        # {a1, a2, a3} has x and y poses, {b1, b2, b3} only y
        assert result.mixed_clusters == (0,)
        assert complete_linkage_clusters(OFFSET_MATRIX, [1.0])[0].mixed_clusters is None

    def test_one_or_no_pose_has_no_clusters_and_says_why(self):
        for n_poses in (0, 1):
            (result,) = complete_linkage_clusters(line_matrix([0.0] * n_poses), [1.0])
            assert result.n_clusters is None and result.sizes is None
            assert result.largest_cluster_fraction is None and result.labels is None
            assert result.reason == f"needs at least 2 poses (got {n_poses})"
            _json_strict(result.to_dict())

    def test_two_poses_are_one_cluster_or_two(self):
        matrix = line_matrix([0.0, 0.5], ["x", "y"])
        near, far = complete_linkage_clusters(matrix, [1.0, 0.3])
        assert (near.n_clusters, near.largest_cluster_fraction) == (1, 1.0)
        assert (far.n_clusters, far.sizes, far.largest_cluster_fraction) == (2, (1, 1), 0.5)
        assert "reason" not in near.to_dict()

    @pytest.mark.parametrize("cutoffs", [[0.0], [-1.0], [float("nan")], [float("inf")], []])
    def test_a_cutoff_must_be_positive_and_finite(self, cutoffs):
        with pytest.raises(ValueError, match="cutoffs_angstrom"):
            complete_linkage_clusters(OFFSET_MATRIX, cutoffs)

    def test_a_single_number_is_accepted(self):
        (result,) = complete_linkage_clusters(OFFSET_MATRIX, 1.0)
        assert result.cutoff_angstrom == 1.0

    def test_a_similarity_matrix_cannot_be_clustered(self):
        with pytest.raises(ValueError, match="distance matrix"):
            complete_linkage_clusters(jaccard_matrix({"a": {1}, "b": {1}}))

    def test_the_dictionary_is_json_ready(self):
        out = _json_strict(complete_linkage_clusters(OFFSET_MATRIX, [0.5])[0].to_dict())
        assert out["linkage"] == "complete"
        assert out["labels"]["a3"] == 2 and out["clusters"][0] == ["a1", "a2"]

    def test_the_result_agrees_with_scipy_on_random_distances(self):
        scipy_cluster = pytest.importorskip("scipy.cluster.hierarchy")
        scipy_spatial = pytest.importorskip("scipy.spatial.distance")
        rng = np.random.default_rng(11)
        for n_poses in (5, 9, 14):
            points = rng.normal(0.0, 1.5, size=(n_poses, 3))
            ids = [f"p{i:02d}" for i in range(n_poses)]
            values = scipy_spatial.squareform(scipy_spatial.pdist(points))
            matrix = PairMatrix("test", "angstrom", "distance", tuple(ids), values, {})
            tree = scipy_cluster.linkage(scipy_spatial.pdist(points), method="complete")
            for cutoff in (0.8, 1.5, 2.5, 3.5):
                (ours,) = complete_linkage_clusters(matrix, [cutoff])
                labels = scipy_cluster.fcluster(tree, t=cutoff, criterion="distance")
                theirs = {
                    frozenset(ids[i] for i in np.flatnonzero(labels == label))
                    for label in set(labels)
                }
                assert partition(ours) == theirs, (n_poses, cutoff)

    def test_the_number_of_clusters_does_not_grow_with_the_cutoff(self):
        rng = np.random.default_rng(3)
        points = rng.normal(0.0, 2.0, size=(12, 3))
        values = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        matrix = PairMatrix(
            "test", "angstrom", "distance", tuple(f"p{i:02d}" for i in range(12)), values, {}
        )
        cutoffs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, float(values.max()) + 0.1]
        results = complete_linkage_clusters(matrix, cutoffs)
        counts = [r.n_clusters for r in results]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] == 1
        for result in results:
            assert sum(result.sizes) == 12
            assert sorted(result.labels.values()) == sorted(
                label for label, size in enumerate(result.sizes) for _ in range(size)
            )

    def test_clusters_of_computed_poses(self, translated_set):
        matrix = pairwise_peptide_rmsd(translated_set, "B")
        by_cutoff = {
            r.cutoff_angstrom: r for r in complete_linkage_clusters(matrix, [0.5, 1.0, 3.5])
        }
        assert by_cutoff[1.0].sizes == (3, 3)
        assert by_cutoff[3.5].sizes == (6,)
        assert by_cutoff[0.5].n_clusters == 4


# ---------------------------------------------------------------------------
# Jaccard indices, interface residues and contact maps
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_the_index_of_overlapping_sets(self):
        matrix = jaccard_matrix({"b": {2, 3, 4}, "a": {1, 2, 3}, "c": {9}})
        assert matrix.pose_ids == ("a", "b", "c")
        assert matrix.direction == "similarity" and matrix.unit is None
        assert matrix.value("a", "b") == pytest.approx(2 / 4)
        assert matrix.value("a", "c") == 0.0
        assert matrix.value("a", "a") == 1.0

    def test_two_empty_sets_have_no_value_and_one_empty_set_gives_zero(self):
        matrix = jaccard_matrix({"a": set(), "b": set(), "c": {1}}, empty_reason="both empty")
        assert matrix.value("a", "b") is None
        assert matrix.reason("a", "b") == "both empty"
        assert matrix.value("a", "c") == 0.0
        assert matrix.value("a", "a") is None
        _json_strict(matrix.to_dict())

    def test_statistics_of_a_similarity_matrix_use_the_same_groups(self):
        matrix = jaccard_matrix({"a1": {1, 2}, "a2": {1, 2}, "b1": {1, 3}, "b2": {3}})
        stats = pair_statistics(matrix, {"a1": "a", "a2": "a", "b1": "b", "b2": "b"})
        assert stats["within"]["a"]["mean"] == 1.0
        assert stats["within"]["b"]["mean"] == pytest.approx(0.5)
        assert stats["between"]["a|b"]["n_pairs"] == 4
        assert stats["between"]["a|b"]["max"] == pytest.approx(1 / 3)

    def test_interface_jaccard_gives_both_sides_and_each_side(self):
        interfaces = {
            "p": InterfaceResidues(frozenset({1, 2, 3}), frozenset({("A", 5, ""), ("A", 6, "")})),
            "q": InterfaceResidues(frozenset({2, 3}), frozenset({("A", 5, ""), ("A", 7, "")})),
            "r": InterfaceResidues(frozenset(), frozenset()),
            "s": InterfaceResidues(frozenset(), frozenset()),
        }
        out = interface_jaccard(interfaces)
        assert set(out) == {"all", "receptor", "binder"}
        assert out["binder"].value("p", "q") == pytest.approx(2 / 3)
        assert out["receptor"].value("p", "q") == pytest.approx(1 / 3)
        assert out["all"].value("p", "q") == pytest.approx(3 / 6)
        assert out["all"].value("r", "s") is None
        assert out["all"].reason("r", "s") == "both interfaces empty"
        assert out["all"].value("p", "r") == 0.0

    def test_a_receptor_key_and_a_binder_position_never_collide_in_the_combined_set(self):
        interface = InterfaceResidues(frozenset({5}), frozenset({5}))
        assert len(interface.combined) == 2

    def test_the_contact_map_jaccard_has_its_own_reason(self):
        matrix = contact_map_jaccard({"a": set(), "b": set(), "c": {(1, "x")}, "d": {(1, "x")}})
        assert matrix.name == "contact_map_jaccard"
        assert matrix.reason("a", "b") == "both contact maps empty"
        assert matrix.value("c", "d") == 1.0


def _contact_fixture():
    """A one-residue binder and a receptor with a near atom, a water and a hydrogen close by."""
    rows = [
        ("B", 1, "", "ALA", "N", (0.0, 0.0, 0.0)),
        ("B", 1, "", "ALA", "CA", (1.4, 0.0, 0.0)),
        ("B", 2, "", "GLY", "CA", (0.0, 3.0, 0.0)),
        ("A", 10, "", "GLY", "CA", (1.4, 4.0, 0.0)),
        ("A", 11, "A", "GLY", "CA", (1.4, 0.0, 5.0)),
        ("A", 12, "", "HOH", "O", (0.0, 1.5, 0.0)),
        ("A", 13, "", "GLY", "H", (0.0, 1.0, 1.0)),
        ("A", 14, "", "GLY", "CA", (1.4, 0.0, 8.0)),
    ]
    atoms = build_atoms(rows)
    atoms.element[atoms.atom_name == "H"] = "H"
    return atoms


class TestContacts:
    def test_residue_pairs_closer_than_the_cutoff(self):
        atoms = _contact_fixture()
        contacts = heavy_atom_contacts(atoms, "B", target_chain="A")
        # B:1 is 4.0 from A:10 and exactly 5.0 from A:11A (not closer than the cutoff); the
        # water and the hydrogen are ignored; B:2 sits 3.0 from the first atom of B:1 only
        assert contacts == frozenset({(1, ("A", 10, "")), (2, ("A", 10, ""))})
        wider = heavy_atom_contacts(atoms, "B", target_chain="A", cutoff_angstrom=5.01)
        assert (1, ("A", 11, "A")) in wider
        assert (1, ("A", 14, "")) not in wider

    def test_the_binder_position_counts_residues_not_numbers(self):
        atoms = _contact_fixture()
        atoms.res_id[atoms.chain_id == "B"] += 100
        assert {position for position, _ in heavy_atom_contacts(atoms, "B")} == {1, 2}

    def test_every_atom_outside_the_binder_is_the_receptor_by_default(self):
        atoms = _contact_fixture()
        assert heavy_atom_contacts(atoms, "B") == heavy_atom_contacts(atoms, "B", target_chain="A")

    def test_the_key_type_is_chosen_by_the_caller(self):
        from collections import namedtuple

        Key = namedtuple("Key", "chain number icode")
        contacts = heavy_atom_contacts(_contact_fixture(), "B", key_factory=Key)
        assert (1, Key("A", 10, "")) in contacts

    def test_an_unknown_target_chain_is_refused_and_an_empty_receptor_has_no_contact(self):
        atoms = _contact_fixture()
        with pytest.raises(ValueError, match="no atom of chain 'Z'"):
            heavy_atom_contacts(atoms, "B", target_chain="Z")
        assert heavy_atom_contacts(atoms[atoms.chain_id == "B"], "B") == frozenset()
        assert heavy_atom_contacts(atoms, "B", cutoff_angstrom=0.5) == frozenset()

    def test_a_missing_binder_chain_is_refused(self):
        with pytest.raises(ValueError, match="no atom of chain 'Q'"):
            heavy_atom_contacts(_contact_fixture(), "Q")

    def test_the_interface_residues_of_the_fixture(self):
        interface = interface_residues(_contact_fixture(), "B", target_chain="A")
        assert interface.binder == frozenset({1, 2})
        assert interface.receptor == frozenset({("A", 10, "")})

    def test_the_contacts_of_the_p53_peptide_match_a_brute_force_search(self, p53):
        contacts = heavy_atom_contacts(p53, "B", target_chain="A")
        binder = p53[(p53.chain_id == "B")]
        receptor = p53[(p53.chain_id == "A")]
        numbers = _numbers(p53, "B")
        expected = set()
        for i in range(binder.array_length()):
            distances = np.linalg.norm(
                receptor.coord.astype(np.float64) - binder.coord[i].astype(np.float64), axis=1
            )
            position = numbers.index(int(binder.res_id[i])) + 1
            for j in np.flatnonzero(distances < 5.0):
                expected.add((position, ("A", int(receptor.res_id[j]), str(receptor.ins_code[j]))))
        assert contacts == expected
        assert len(contacts) > 40

    def test_the_known_hot_spot_residues_of_p53_are_in_the_interface(self, p53):
        # Phe19, Trp23 and Leu26 of p53 bury themselves in MDM2: positions 3, 7 and 10 of 17-29
        interface = interface_residues(p53, "B", target_chain="A")
        assert {3, 7, 10} <= interface.binder
        assert all(key[0] == "A" and isinstance(key[1], int) for key in interface.receptor)

    def test_a_larger_cutoff_gives_a_superset(self, p53):
        small = interface_residues(p53, "B", cutoff_angstrom=4.0)
        large = interface_residues(p53, "B", cutoff_angstrom=6.0)
        assert small.binder <= large.binder and small.receptor <= large.receptor
        assert len(large.receptor) > len(small.receptor)

    def test_contact_maps_fall_apart_when_the_peptide_leaves(self, p53):
        near = heavy_atom_contacts(p53, "B", target_chain="A")
        far = heavy_atom_contacts(moved(p53, "B", [0.0, 0.0, 40.0]), "B", target_chain="A")
        assert far == frozenset()
        matrix = contact_map_jaccard({"near": near, "same": near, "far": far})
        assert matrix.value("near", "same") == 1.0
        assert matrix.value("far", "near") == 0.0

    def test_the_waters_of_a_ring_chain_are_not_binder_atoms(self, csa):
        contacts = heavy_atom_contacts(csa, "C", target_chain="A")
        positions = {position for position, _ in contacts}
        assert positions and positions <= set(range(1, 12))


# ---------------------------------------------------------------------------
# positional spread
# ---------------------------------------------------------------------------


class TestPositionalSpread:
    def test_translations_along_one_axis_give_the_population_standard_deviation(
        self, translated_set
    ):
        result = positional_spread(translated_set, "B")
        expected = float(np.std(list(OFFSETS.values())))  # divisor n
        assert len(result.positions) == 13
        for position in result.positions:
            assert position.spread_angstrom == pytest.approx(expected, abs=TOL)
            assert position.n_poses == 6
        assert result.mean_spread_angstrom == pytest.approx(expected, abs=TOL)
        assert result.pose_ids == tuple(sorted(OFFSETS)) and result.reference_id == "a1"
        assert result.excluded == {} and result.reason is None

    def test_the_positions_are_named_after_the_reference_residues(self, translated_set):
        positions = positional_spread(translated_set, "B").positions
        assert [p.binder_index for p in positions] == list(range(1, 14))
        assert (positions[0].chain, positions[0].number, positions[0].residue) == ("B", 17, "GLU")
        assert (positions[2].number, positions[2].insertion_code, positions[2].residue) == (
            19,
            "",
            "PHE",
        )

    def test_two_poses_scatter_by_half_their_distance(self, p53):
        result = positional_spread({"a": p53, "b": moved(p53, "B", [0.0, 3.0, 0.0])}, "B")
        assert result.mean_spread_angstrom == pytest.approx(1.5, abs=TOL)

    def test_identical_poses_do_not_scatter(self, p53):
        result = positional_spread({"a": p53, "b": p53.copy(), "c": p53.copy()}, "B")
        assert result.mean_spread_angstrom == pytest.approx(0.0, abs=1e-6)

    def test_one_pose_has_no_spread_and_says_why(self, p53):
        result = positional_spread({"only": p53}, "B")
        assert result.positions == () and result.mean_spread_angstrom is None
        assert result.reason == "needs at least 2 usable poses (got 1)"
        assert result.reference_id == "only"
        _json_strict(result.to_dict())

    def test_no_pose_has_no_spread(self):
        result = positional_spread({}, "B")
        assert result.reference_id is None and result.reason.startswith("needs at least 2")

    def test_a_pose_that_cannot_be_paired_or_read_is_excluded_with_its_reason(self, p53):
        different = renamed_residue(p53, "B", 21, "GLU")
        no_binder = p53[p53.chain_id == "A"]
        result = positional_spread(
            {"a": p53, "b": moved(p53, "B", [0.0, 2.0, 0.0]), "c": different, "d": no_binder}, "B"
        )
        assert result.pose_ids == ("a", "b")
        assert set(result.excluded) == {"c", "d"}
        assert "is ASP" in result.excluded["c"] and "no atom of chain" in result.excluded["d"]
        assert result.mean_spread_angstrom == pytest.approx(1.0, abs=TOL)

    def test_a_position_that_only_one_pose_has_gets_no_spread(self, p53):
        missing = _numbers(p53, "B")[6]
        poses = {"a": p53, "b": without_residue(moved(p53, "B", [0.0, 2.0, 0.0]), "B", missing)}
        result = positional_spread(poses, "B")
        by_index = {p.binder_index: p for p in result.positions}
        assert by_index[7].spread_angstrom is None
        assert by_index[7].n_poses == 1 and "at least 2 poses" in by_index[7].reason
        assert by_index[8].spread_angstrom == pytest.approx(1.0, abs=TOL)
        # the mean runs over the positions that have a value
        assert result.mean_spread_angstrom == pytest.approx(1.0, abs=TOL)
        out = _json_strict(result.to_dict())
        assert "reason" in out["positions"][6] and "reason" not in out["positions"][0]

    def test_another_numbering_is_paired_by_position(self, p53):
        renumbered_pose = renumbered(moved(p53, "B", [0.0, 2.0, 0.0]), "B", 100)
        result = positional_spread({"a": p53, "b": renumbered_pose}, "B")
        assert result.excluded == {}
        assert result.mean_spread_angstrom == pytest.approx(1.0, abs=TOL)
        assert (
            positional_spread(
                {"a": p53, "b": renumbered_pose}, "B", residue_pairing="key"
            ).excluded["b"]
            == "no residue key is shared"
        )

    def test_the_reference_pose_can_be_chosen(self, p53):
        renumbered_pose = renumbered(p53, "B", 100)
        result = positional_spread({"a": p53, "b": renumbered_pose}, "B", reference_id="b")
        assert result.reference_id == "b"
        assert result.positions[0].number == 117
        with pytest.raises(ValueError, match="not one of the poses"):
            positional_spread({"a": p53}, "B", reference_id="z")
        with pytest.raises(ValueError, match="no atom of chain"):
            positional_spread({"a": p53, "b": p53[p53.chain_id == "A"]}, "B", reference_id="b")

    def test_every_position_of_a_ring_has_a_spread(self, csa):
        poses = {
            "a": csa,
            "b": moved(csa, "C", [0.0, 0.0, 1.0]),
            "c": moved(csa, "C", [0.0, 0.0, -1.0]),
        }
        result = positional_spread(poses, "C")
        # every residue with an alpha carbon, in the 11-residue ring (waters are not residues)
        assert len(result.positions) == 11
        expected = float(np.std([0.0, 1.0, -1.0]))
        assert result.mean_spread_angstrom == pytest.approx(expected, abs=TOL)

    def test_the_spread_of_each_group(self, translated_set):
        groups = {name: name[0] for name in translated_set}
        by_group = positional_spread_by_group(translated_set, "B", groups)
        assert list(by_group) == ["a", "b"]
        assert by_group["a"].mean_spread_angstrom == pytest.approx(
            float(np.std([0.0, 0.4, 0.8])), abs=TOL
        )
        assert by_group["b"].mean_spread_angstrom == pytest.approx(
            float(np.std([2.0, 2.3, 2.9])), abs=TOL
        )
        with pytest.raises(ValueError, match="without a group label"):
            positional_spread_by_group(translated_set, "B", {"a1": "a"})

    def test_a_bad_pairing_option_raises(self, p53):
        with pytest.raises(ValueError, match="residue_pairing"):
            positional_spread({"a": p53}, "B", residue_pairing="index")


# ---------------------------------------------------------------------------
# the pieces together on one set
# ---------------------------------------------------------------------------


def test_the_agreement_of_two_groups_of_computed_poses_end_to_end(p53):
    poses = {
        "m1:s1": moved(p53, "B", [0.0, 0.0, 0.0]),
        "m1:s2": moved(p53, "B", [0.4, 0.0, 0.0]),
        "m1:s3": moved(p53, "B", [0.8, 0.0, 0.0]),
        "m2:s1": moved(p53, "B", [0.0, 2.0, 0.0]),
        "m2:s2": moved(p53, "B", [0.0, 2.3, 0.0]),
        "m2:s3": moved(p53, "B", [0.0, 2.9, 0.0]),
    }
    groups = {pose_id: pose_id.split(":")[0] for pose_id in poses}
    matrix = pairwise_peptide_rmsd(poses, "B", kind="ca")
    stats = pair_statistics(matrix, groups)
    clusters = complete_linkage_clusters(matrix, groups=groups)
    picks = medoids(matrix, groups)
    contacts = {
        pose_id: heavy_atom_contacts(atoms, "B", target_chain="A")
        for pose_id, atoms in poses.items()
    }
    overlap = contact_map_jaccard(contacts)
    spread = positional_spread_by_group(poses, "B", groups)
    assert stats["within"]["m1"]["max"] == pytest.approx(0.8, abs=TOL)
    assert [c.n_clusters for c in clusters] == [2, 2] and clusters[0].mixed_clusters == ()
    assert picks["per_group"]["m1"]["pose_id"] == "m1:s2"
    assert overlap.value("m1:s1", "m1:s1") == 1.0
    assert set(spread) == {"m1", "m2"}
    _json_strict(
        {
            "stats": stats,
            "clusters": [c.to_dict() for c in clusters],
            "medoids": picks,
            "jaccard": overlap.to_dict(),
            "spread": {k: v.to_dict() for k, v in spread.items()},
        }
    )
