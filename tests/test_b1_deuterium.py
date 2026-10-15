"""Deuterium is hydrogen for the van der Waals radii of the geometry and receptor metrics.

The package treats the elements H and D as hydrogen, and where the element column is empty it
takes an atom name that starts with H for a hydrogen (``poses.agreement._hydrogen_mask``,
``io.structures``). The Bondi table of the shape complementarity and the void volume had an
entry for H only, so every D atom got the 1.80 A fallback instead of 1.20 A: a protonated
structure and the same structure with its hydrogens labelled D scored differently.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biotite")
hydride = pytest.importorskip("hydride")

import biotite.structure as struc  # noqa: E402
import biotite.structure.io.pdb as pdb_io  # noqa: E402

from binding_metrics.metrics import geometry, receptor_quality  # noqa: E402

P53_MDM2 = Path(__file__).resolve().parents[1] / "data" / "example_linear_p53_1YCR.pdb"


def _write_pdb(atoms, path):
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, atoms)
    pdb_file.write(str(path))
    return path


@pytest.fixture(scope="module")
def protonated_atoms():
    """1YCR with hydrogens added by hydride at ideal geometry."""
    atoms = pdb_io.get_structure(pdb_io.PDBFile.read(str(P53_MDM2)), model=1, include_bonds=True)
    atoms.set_annotation("charge", np.zeros(len(atoms), dtype=int))
    protonated, _ = hydride.add_hydrogen(atoms)
    # Every residue carries an alpha hydrogen and every residue but proline an amide one: fewer
    # hydrogens mean residues left bare, and the H against D comparisons below would be empty.
    n_residues = struc.get_residue_count(protonated)
    n_proline = struc.get_residue_count(protonated[protonated.res_name == "PRO"])
    assert (protonated.element == "H").sum() >= 2 * n_residues - n_proline
    return protonated


@pytest.fixture(scope="module")
def structures(tmp_path_factory, protonated_atoms):
    """Paths of the protonated 1YCR, of the same with every H labelled D, and of its heavy atoms."""
    folder = tmp_path_factory.mktemp("b1_deuterium")
    deuterated = protonated_atoms.copy()
    deuterated.element[deuterated.element == "H"] = "D"
    assert (deuterated.element == "D").sum() == (protonated_atoms.element == "H").sum()
    return {
        "H": _write_pdb(protonated_atoms, folder / "with_h.pdb"),
        "D": _write_pdb(deuterated, folder / "with_d.pdb"),
        "heavy": _write_pdb(
            protonated_atoms[protonated_atoms.element != "H"], folder / "heavy.pdb"
        ),
    }


def _atoms(*pairs):
    """An array with one atom per (element, atom name) pair, 4 A apart along x."""
    atoms = struc.AtomArray(len(pairs))
    coord = np.zeros((len(pairs), 3), dtype=np.float32)
    coord[:, 0] = 4.0 * np.arange(len(pairs))
    atoms.coord = coord
    atoms.element = np.array([element for element, _ in pairs])
    atoms.atom_name = np.array([name for _, name in pairs])
    return atoms


class TestGeometryTable:
    def test_deuterium_has_the_hydrogen_radius(self):
        assert geometry._get_vdw("D") == geometry._get_vdw("H") == 1.20

    @pytest.mark.parametrize("spelling", ["d", " D ", "D"])
    def test_symbol_case_and_padding_are_ignored(self, spelling):
        assert geometry._get_vdw(spelling) == 1.20

    def test_a_deuterium_atom_is_dotted_on_the_hydrogen_sphere(self):
        atom = _atoms(("D", "D1"))
        partner = _atoms(("C", "CA"))
        partner.coord = np.array([[0.0, 0.0, 2.7]], dtype=np.float32)

        dots, _ = geometry._build_surface_dots(atom, atom, partner, 150, 2.4, 6.0)

        assert len(dots) > 0
        assert np.linalg.norm(dots, axis=1) == pytest.approx(1.20, abs=1e-5)


class TestElementFirstThenName:
    """Per-atom radii: the element column decides, the atom name only where it is empty."""

    def test_the_radius_of_each_atom(self):
        atoms = _atoms(
            ("H", "HA"),
            ("D", "DA"),
            ("C", "CA"),
            ("BR", "BR"),
            ("", "HD21"),  # no element: a name starting with H is a hydrogen
            ("", "CA"),  # no element, not a hydrogen name: the fallback
            ("", "DG1"),  # no element: a D prefix is not read as deuterium
            ("HG", "HG"),  # mercury: the element wins over the H in the name
            ("C", "H"),  # a carbon with an unusual name: the element wins
        )
        assert geometry._atom_radii(atoms).tolist() == [
            1.20,
            1.20,
            1.70,
            1.85,
            1.20,
            1.80,
            1.80,
            1.80,
            1.70,
        ]

    def test_an_empty_array_has_no_radii(self):
        assert geometry._atom_radii(_atoms()).shape == (0,)


class TestProtonatedAndDeuteratedStructuresAgree:
    """The same coordinates, hydrogens labelled H or D, give the same Sc and void volume."""

    @pytest.mark.slow
    def test_shape_complementarity(self, structures):
        kwargs = dict(peptide_chain="B", receptor_chain="A")
        with_h = geometry.compute_shape_complementarity(structures["H"], **kwargs)
        with_d = geometry.compute_shape_complementarity(structures["D"], **kwargs)
        heavy = geometry.compute_shape_complementarity(structures["heavy"], **kwargs)

        assert -1.0 <= with_h["sc"] <= 1.0
        assert with_d["sc"] == with_h["sc"]
        assert with_d["sc_A_to_B"] == with_h["sc_A_to_B"]
        assert with_d["sc_B_to_A"] == with_h["sc_B_to_A"]
        assert with_d["n_surface_dots_A"] == with_h["n_surface_dots_A"]
        assert with_d["n_surface_dots_B"] == with_h["n_surface_dots_B"]
        # Hydrogens and deuterium both being dropped would also make H and D agree: the surface
        # of the heavy atoms alone is a different one.
        assert with_d["n_surface_dots_A"] != heavy["n_surface_dots_A"]

    @pytest.mark.slow
    def test_void_volume(self, structures):
        kwargs = dict(peptide_chain="B", receptor_chain="A")
        with_h = geometry.compute_buried_void_volume(structures["H"], **kwargs)
        with_d = geometry.compute_buried_void_volume(structures["D"], **kwargs)
        heavy = geometry.compute_buried_void_volume(structures["heavy"], **kwargs)

        assert with_h["void_volume_A3"] >= 0.0
        assert with_d["void_volume_A3"] == with_h["void_volume_A3"]
        assert with_d["n_interface_atoms"] == with_h["n_interface_atoms"]
        # Hydrogens and deuterium both being dropped would also make H and D agree: the
        # interface of the heavy atoms alone has fewer atoms.
        assert with_d["n_interface_atoms"] > heavy["n_interface_atoms"]


class TestReceptorQualityTable:
    """The clash score leaves hydrogens out, deuterium included, so it needs no D radius."""

    def test_deuterium_and_hydrogen_get_the_same_radius(self):
        assert receptor_quality._vdw("D") == receptor_quality._vdw("H")

    @pytest.mark.slow
    def test_clashscore_is_the_same_with_h_or_d(self, protonated_atoms):
        chain_a = protonated_atoms[protonated_atoms.chain_id == "A"]
        deuterated = chain_a.copy()
        deuterated.element[deuterated.element == "H"] = "D"

        with_h = receptor_quality._clashscore(chain_a)
        with_d = receptor_quality._clashscore(deuterated)

        assert with_d == with_h
        heavy = int((chain_a.element != "H").sum())
        assert with_h["n_heavy_atoms"] == heavy
