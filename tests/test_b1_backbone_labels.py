"""Residue labels of the backbone dihedrals when the chain holds residues without a C-alpha.

``biotite.structure.dihedral_backbone`` returns one row per residue of the array, caps
(ACE, NME, NH2), waters and ions included, so a row is a residue index and not an index into
the C-alpha atoms. Pairing the rows with the C-alpha atoms by position labels every residue
that follows a cap with the angles of its predecessor, which moves the D-residue mirroring,
``n_d_residues`` and the per-residue listing of ``compute_ramachandran``,
``compute_omega_planarity`` and the Ramachandran term of the receptor quality.

The fixtures are the 13-residue peptide of the bundled 1YCR example (chain B), copied to a
temporary directory with an ACE and an NH2 cap added. The cap atoms are non-amino-acid atoms,
so ``dihedral_backbone`` gives NaN for the residue next to a cap, exactly as at a chain end:
the capped peptide has the same 11 evaluated residues as the uncapped one.
"""

import numpy as np
import pytest

pytest.importorskip("biotite")

import biotite.structure as struc  # noqa: E402 - after the importorskip
import biotite.structure.io.pdb as pdb_io  # noqa: E402
from conftest import DATA_DIR  # noqa: E402

from binding_metrics.metrics.geometry import (  # noqa: E402
    _classify_ramachandran,
    compute_omega_planarity,
    compute_ramachandran,
)
from binding_metrics.metrics.receptor_quality import _ramachandran  # noqa: E402

PDB_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
CYCLIC_3P8F = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"

# 1YCR chain B holds residues 17 to 29; the first and last have no complete phi/psi pair.
EVALUATED_IDS = list(range(18, 29))

# D name of the 1YCR peptide residues (GLU THR PHE SER ASP LEU TRP LYS LEU LEU PRO GLU ASN).
D_NAME = {
    "GLU": "DGL",
    "THR": "DTH",
    "PHE": "DPN",
    "SER": "DSN",
    "ASP": "DAS",
    "LEU": "DLE",
    "TRP": "DTR",
    "LYS": "DLY",
    "PRO": "DPR",
    "ASN": "DSG",
}


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def _load(path):
    from binding_metrics.metrics.geometry import _load_structure

    return _load_structure(path)


def _write_pdb(atoms, path):
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, atoms)
    pdb_file.write(str(path))
    return path


def _peptide():
    """Heavy atoms of the 1YCR peptide (chain B, residues 17 to 29)."""
    atoms = _load(PDB_1YCR)
    return atoms[(atoms.chain_id == "B") & ~atoms.hetero & (atoms.element != "H")]


def _cap(template, res_id, res_name, atom_names, elements, coords):
    """A cap residue that carries the annotations of ``template``."""
    cap = template[: len(atom_names)].copy()
    cap.coord = np.asarray(coords, dtype=np.float32)
    cap.res_id[:] = res_id
    cap.res_name[:] = res_name
    cap.atom_name = np.array(atom_names)
    cap.element = np.array(elements)
    cap.hetero[:] = False
    return cap


def _capped(peptide):
    """``peptide`` with an ACE before its first residue and an NH2 after its last.

    Only the identity of the cap residues matters to the dihedrals (they are not amino
    acids), so the cap atoms are placed on the N-CA and CA-C axes of the end residues.
    """
    first = peptide[peptide.res_id == peptide.res_id.min()]
    last = peptide[peptide.res_id == peptide.res_id.max()]
    n_first = first.coord[first.atom_name == "N"][0]
    ca_first = first.coord[first.atom_name == "CA"][0]
    c_last = last.coord[last.atom_name == "C"][0]
    ca_last = last.coord[last.atom_name == "CA"][0]
    toward_cap = n_first - ca_first
    ace = _cap(
        peptide,
        peptide.res_id.min() - 1,
        "ACE",
        ["C", "O", "CH3"],
        ["C", "O", "C"],
        [
            n_first + 0.9 * toward_cap,
            n_first + 1.5 * toward_cap + 1,
            n_first + 1.4 * toward_cap - 1,
        ],
    )
    nh2 = _cap(
        peptide,
        peptide.res_id.max() + 1,
        "NH2",
        ["N"],
        ["N"],
        [c_last + 0.9 * (c_last - ca_last)],
    )
    return ace + peptide + nh2


def _relabel(atoms, names_by_res_id):
    relabelled = atoms.copy()
    for res_id, name in names_by_res_id.items():
        relabelled.res_name[relabelled.res_id == res_id] = name
    return relabelled


@pytest.fixture(scope="module")
def peptide():
    return _peptide()


@pytest.fixture(scope="module")
def capped(peptide):
    return _capped(peptide)


@pytest.fixture
def write(tmp_path):
    def _write(atoms, name="peptide.pdb"):
        return _write_pdb(atoms, tmp_path / name)

    return _write


def _rows(result, *keys):
    return [tuple(row[k] for k in keys) for row in result["per_residue"]]


def _reference_angles_deg(atoms):
    """phi and psi in degrees of each residue of chain B, keyed by residue number, from biotite.

    ``dihedral_backbone`` has one row per residue of the array, caps included, and NaN where the
    angle is undefined; the row of a residue is found from the residue starts, not by position
    among the C-alpha atoms.
    """
    chain = atoms[atoms.chain_id == "B"]
    phi, psi, _ = (np.degrees(angles) for angles in struc.dihedral_backbone(chain))
    numbers = chain.res_id[struc.get_residue_starts(chain)]
    return {int(n): (float(f), float(s)) for n, f, s in zip(numbers, phi, psi)}


def _reference_regions(atoms, d_residue=None):
    """Ramachandran region of each residue of chain B that has a phi and a psi.

    The angles come from ``_reference_angles_deg``; the residue ``d_residue`` is classified on the
    mirrored plot. The region boxes are tested on their own in ``tests/test_geometry.py``.
    """
    regions = {
        number: _classify_ramachandran(phi, psi, is_d=number == d_residue)
        for number, (phi, psi) in _reference_angles_deg(atoms).items()
    }
    return {number: region for number, region in regions.items() if region is not None}


# ---------------------------------------------------------------------------
# compute_ramachandran
# ---------------------------------------------------------------------------


class TestRamachandranWithCaps:
    def test_the_fixture_has_the_caps_biotite_counts_as_residues(self, capped, peptide):
        phi, _, _ = struc.dihedral_backbone(capped)
        assert struc.get_residue_count(capped) == struc.get_residue_count(peptide) + 2
        assert len(phi) == struc.get_residue_count(capped)
        assert (capped.atom_name == "CA").sum() == (peptide.atom_name == "CA").sum()

    def test_labels_follow_the_residue(self, peptide, capped, write):
        plain = compute_ramachandran(write(peptide, "plain.pdb"), chain="B")
        with_caps = compute_ramachandran(write(capped, "caps.pdb"), chain="B")

        assert [p["res_id"] for p in with_caps["per_residue"]] == EVALUATED_IDS
        assert _rows(with_caps, "res_id", "res_name", "region") == _rows(
            plain, "res_id", "res_name", "region"
        )
        for key in ("phi", "psi"):
            assert [p[key] for p in with_caps["per_residue"]] == pytest.approx(
                [p[key] for p in plain["per_residue"]]
            )

    def test_the_first_row_is_the_residue_that_owns_its_angles(self, capped, write):
        path = write(capped)
        result = compute_ramachandran(path, chain="B")
        first = result["per_residue"][0]
        assert (first["res_id"], first["res_name"]) == (18, "THR")
        # a residue labelled with the angle of its predecessor (the ACE cap shifts every row by
        # one) would carry the phi of residue 17, which is undefined
        phi, _ = _reference_angles_deg(_load(path))[18]
        assert first["phi"] == pytest.approx(phi, abs=1e-3)
        assert result["n_residues_evaluated"] == 11
        assert "ACE" not in {p["res_name"] for p in result["per_residue"]}
        assert "NH2" not in {p["res_name"] for p in result["per_residue"]}

    @pytest.mark.parametrize("d_residue", [18, 19, 26, 28])
    def test_a_d_label_lands_on_the_residue_that_carries_it(
        self, peptide, capped, write, d_residue
    ):
        plain = compute_ramachandran(write(_relabel(peptide, {d_residue: "DAL"})), chain="B")
        path = write(_relabel(capped, {d_residue: "DAL"}))
        result = compute_ramachandran(path, chain="B")

        assert result["n_d_residues"] == 1
        assert [p["res_id"] for p in result["per_residue"] if p["is_d_aa"]] == [d_residue]
        # the regions of the residues, with only the labelled one mirrored: a label that landed on
        # the neighbouring row would mirror another residue
        regions = list(_reference_regions(_load(path), d_residue).values())
        assert result["n_residues_evaluated"] == len(regions)
        for region in ("favoured", "allowed"):
            assert result[f"ramachandran_{region}_pct"] == pytest.approx(
                100.0 * regions.count(region) / len(regions)
            )
        assert result["ramachandran_outlier_count"] == regions.count("outlier")
        # the capped peptide scores exactly like the same peptide without caps
        for key in (
            "ramachandran_favoured_pct",
            "ramachandran_allowed_pct",
            "ramachandran_outlier_count",
            "n_residues_evaluated",
            "n_d_residues",
        ):
            assert result[key] == pytest.approx(plain[key])

    def test_a_capped_d_peptide_is_listed_under_its_own_residues(self, peptide, capped, write):
        """The D mirror image of the capped peptide scores like the L peptide, row by row."""
        mirror = _relabel(capped, {})
        for res_id in np.unique(mirror.res_id):
            old = mirror.res_name[mirror.res_id == res_id][0]
            if old in D_NAME:
                mirror.res_name[mirror.res_id == res_id] = D_NAME[old]
        mirror.coord = mirror.coord * np.array([1.0, 1.0, -1.0], dtype=np.float32)

        l_result = compute_ramachandran(write(peptide, "l.pdb"), chain="B")
        d_result = compute_ramachandran(write(mirror, "d.pdb"), chain="B")

        assert d_result["n_d_residues"] == 11
        assert _rows(d_result, "res_id") == [(i,) for i in EVALUATED_IDS]
        names_by_id = {int(r): str(n) for r, n in zip(mirror.res_id, mirror.res_name)}
        assert [p["res_name"] for p in d_result["per_residue"]] == [
            names_by_id[i] for i in EVALUATED_IDS
        ]
        assert _rows(d_result, "region") == _rows(l_result, "region")
        assert d_result["ramachandran_favoured_pct"] == pytest.approx(
            l_result["ramachandran_favoured_pct"]
        )


class TestResidueWithoutCalpha:
    """A residue whose C-alpha is missing keeps its row but gets NaN angles."""

    def test_the_residues_after_the_gap_keep_their_own_angles(self, peptide, write):
        full = compute_ramachandran(write(peptide, "full.pdb"), chain="B")
        no_ca = peptide[~((peptide.res_id == 20) & (peptide.atom_name == "CA"))]
        result = compute_ramachandran(write(no_ca, "no_ca.pdb"), chain="B")

        by_id = {p["res_id"]: p for p in full["per_residue"]}
        assert [p["res_id"] for p in result["per_residue"]] == [i for i in EVALUATED_IDS if i != 20]
        for row in result["per_residue"]:
            assert row["res_name"] == by_id[row["res_id"]]["res_name"]
            assert row["phi"] == pytest.approx(by_id[row["res_id"]]["phi"], abs=1e-3)

    def test_omega_rows_after_the_gap_keep_their_own_angles(self, peptide, write):
        full = compute_omega_planarity(write(peptide, "full.pdb"), chain="B")
        no_ca = peptide[~((peptide.res_id == 20) & (peptide.atom_name == "CA"))]
        result = compute_omega_planarity(write(no_ca, "no_ca.pdb"), chain="B")

        by_id = {p["res_id"]: p for p in full["per_residue"]}
        assert {p["res_id"] for p in result["per_residue"]} >= {21, 22, 28}
        for row in result["per_residue"]:
            assert row["res_name"] == by_id[row["res_id"]]["res_name"]
            assert row["omega"] == pytest.approx(by_id[row["res_id"]]["omega"], abs=1e-3)

    def test_receptor_quality_counts_the_regions_of_the_residues_it_evaluates(self, peptide, write):
        """With a D label on residue 28, residue 20 alone drops out of the counts."""
        d_label = {28: "DAL"}
        full = _ramachandran(_relabel(peptide, d_label))
        region_of_20 = {
            p["res_id"]: p["region"]
            for p in compute_ramachandran(write(peptide, "full.pdb"), chain="B")["per_residue"]
        }[20]
        expected = {k: full[f"{k}_count"] for k in ("favoured", "allowed", "outlier")}
        expected[region_of_20] -= 1

        no_ca = peptide[~((peptide.res_id == 20) & (peptide.atom_name == "CA"))]
        result = _ramachandran(_relabel(no_ca, d_label))
        assert {k: result[f"{k}_count"] for k in expected} == expected


# ---------------------------------------------------------------------------
# compute_omega_planarity
# ---------------------------------------------------------------------------


class TestOmegaWithCaps:
    def test_labels_follow_the_residue(self, peptide, capped, write):
        plain = compute_omega_planarity(write(peptide, "plain.pdb"), chain="B")
        with_caps = compute_omega_planarity(write(capped, "caps.pdb"), chain="B")

        assert _rows(with_caps, "res_id", "res_name") == _rows(plain, "res_id", "res_name")
        assert [p["omega"] for p in with_caps["per_residue"]] == pytest.approx(
            [p["omega"] for p in plain["per_residue"]]
        )

    def test_omega_is_listed_under_the_residue_that_starts_the_bond(self, capped, write):
        result = compute_omega_planarity(write(capped), chain="B")
        first = result["per_residue"][0]
        assert (first["res_id"], first["res_name"]) == (17, "GLU")
        assert abs(first["omega"]) == pytest.approx(180.0, abs=5.0)
        assert result["n_bonds_evaluated"] == 12
        assert [p["res_id"] for p in result["per_residue"]] == list(range(17, 29))

    def test_the_statistics_do_not_change_with_the_caps(self, peptide, capped, write):
        plain = compute_omega_planarity(write(peptide, "plain.pdb"), chain="B")
        with_caps = compute_omega_planarity(write(capped, "caps.pdb"), chain="B")
        for key in (
            "omega_mean_dev",
            "omega_max_dev",
            "omega_outlier_count",
            "n_bonds_evaluated",
        ):
            assert with_caps[key] == pytest.approx(plain[key])


# ---------------------------------------------------------------------------
# receptor_quality._ramachandran
# ---------------------------------------------------------------------------


class TestReceptorQualityRamachandranWithCaps:
    """The Ramachandran term of the receptor quality, called on an array with caps."""

    @pytest.mark.parametrize("d_residue", [18, 19, 26, 28])
    def test_a_d_label_lands_on_the_residue_that_carries_it(self, peptide, capped, d_residue):
        plain = _ramachandran(_relabel(peptide, {d_residue: "DAL"}))
        result = _ramachandran(_relabel(capped, {d_residue: "DAL"}))

        # the regions of the residues, with only the labelled one mirrored
        regions = list(_reference_regions(peptide, d_residue).values())
        counts = tuple(regions.count(r) for r in ("favoured", "allowed", "outlier"))
        for scored in (plain, result):
            assert (
                scored["favoured_count"],
                scored["allowed_count"],
                scored["outlier_count"],
            ) == counts
            assert scored["n_evaluated"] == len(regions)

    def test_an_all_l_capped_peptide_matches_the_uncapped_one(self, peptide, capped):
        plain = _ramachandran(peptide)
        result = _ramachandran(capped)
        assert result == plain
        assert result["n_evaluated"] == 11


# ---------------------------------------------------------------------------
# Head-to-tail rings: the closing angles go on the first and last amino-acid rows
# ---------------------------------------------------------------------------


def _sfti_ring():
    """The amino-acid atoms of the cyclic peptide chain I of 3P8F and one of its waters."""
    atoms = _load(CYCLIC_3P8F)
    chain = atoms[atoms.chain_id == "I"]
    ring = chain[struc.filter_amino_acids(chain)]
    water = chain[chain.res_name == "HOH"]
    water = water[water.res_id == water.res_id[0]]
    return ring, water


class TestRingClosureRows:
    def test_a_leading_non_amino_acid_residue_does_not_shift_the_closing_angles(self, write):
        ring, water = _sfti_ring()
        plain = compute_ramachandran(write(ring, "ring.pdb"), chain="I")
        shifted = compute_ramachandran(write(water + ring, "water_first.pdb"), chain="I")

        assert plain["cyclic_closure_evaluated"] is True
        assert shifted["cyclic_closure_evaluated"] is True
        assert plain["n_residues_evaluated"] == 14
        assert shifted["n_residues_evaluated"] == 14
        assert _rows(shifted, "res_id", "res_name") == _rows(plain, "res_id", "res_name")
        assert [p["phi"] for p in shifted["per_residue"]] == pytest.approx(
            [p["phi"] for p in plain["per_residue"]]
        )
        assert [p["psi"] for p in shifted["per_residue"]] == pytest.approx(
            [p["psi"] for p in plain["per_residue"]]
        )

    def test_the_closing_omega_stays_under_the_last_residue(self, write):
        ring, water = _sfti_ring()
        plain = compute_omega_planarity(write(ring, "ring.pdb"), chain="I")
        shifted = compute_omega_planarity(write(water + ring, "water_first.pdb"), chain="I")

        assert shifted["cyclic_closure_evaluated"] is True
        assert shifted["n_bonds_evaluated"] == plain["n_bonds_evaluated"] == 14
        assert _rows(shifted, "res_id", "res_name") == _rows(plain, "res_id", "res_name")
        assert [p["omega"] for p in shifted["per_residue"]] == pytest.approx(
            [p["omega"] for p in plain["per_residue"]]
        )
        assert shifted["per_residue"][-1]["res_id"] == ring.res_id.max()

    def test_a_trailing_calcium_ion_named_ca_is_not_a_residue_of_the_ring(self, write):
        ring, _ = _sfti_ring()
        ion = ring[:1].copy()
        ion.res_id[:] = ring.res_id.max() + 1
        ion.res_name[:] = "CA"
        ion.atom_name[:] = "CA"
        ion.element[:] = "CA"
        ion.hetero[:] = True
        plain = compute_ramachandran(write(ring, "ring.pdb"), chain="I")
        with_ion = compute_ramachandran(write(ring + ion, "ion.pdb"), chain="I")

        assert with_ion["cyclic_closure_evaluated"] is True
        assert _rows(with_ion, "res_id", "res_name") == _rows(plain, "res_id", "res_name")
        for key in ("phi", "psi"):
            assert [p[key] for p in with_ion["per_residue"]] == pytest.approx(
                [p[key] for p in plain["per_residue"]]
            )
