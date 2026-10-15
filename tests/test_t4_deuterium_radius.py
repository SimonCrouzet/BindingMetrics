"""Deuterium takes the van der Waals radius of hydrogen when hydrogens are kept (#225)."""

import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.info import vdw_radius_single

from binding_metrics.metrics.interface import _get_vdw_radii, compute_interface_metrics
from binding_metrics.metrics.sasa import compute_delta_sasa_static
from binding_metrics.testing import bundled_complex_path


def _two_atoms(first: str, second: str) -> AtomArray:
    atoms = AtomArray(2)
    atoms.element = np.array([first, second])
    return atoms


def test_the_radius_of_deuterium_is_the_radius_of_hydrogen():
    radii = _get_vdw_radii(_two_atoms("H", "D"), vdw_radius_single)

    assert radii[0] == radii[1] == vdw_radius_single("H")


def test_an_element_without_an_entry_keeps_the_fallback_radius():
    # a deuterium fix must not change the radius of an element that the table lacks
    assert vdw_radius_single("XX") is None
    assert _get_vdw_radii(_two_atoms("XX", "H"), vdw_radius_single)[0] == pytest.approx(1.8)


@pytest.fixture(scope="module")
def protonated_and_deuterated(tmp_path_factory):
    hydride = pytest.importorskip("hydride")
    import biotite.structure.io.pdb as pdb

    folder = tmp_path_factory.mktemp("deuterium")
    atoms = pdb.PDBFile.read(str(bundled_complex_path())).get_structure(model=1, include_bonds=True)
    atoms = atoms[atoms.element != "H"]
    atoms.set_annotation("charge", np.zeros(atoms.array_length(), dtype=int))
    with_h, _ = hydride.add_hydrogen(atoms)
    paths = {}
    for label, symbol in (("H", "H"), ("D", "D")):
        copy = with_h.copy()
        copy.element[copy.element == "H"] = symbol
        file = pdb.PDBFile()
        file.set_structure(copy)
        paths[label] = folder / f"{label}.pdb"
        file.write(str(paths[label]))
    return paths


def test_the_interface_area_does_not_change_when_the_hydrogens_are_deuterium(
    protonated_and_deuterated,
):
    areas = {
        label: compute_interface_metrics(str(path), "B", "A", hydrogens="keep")
        for label, path in protonated_and_deuterated.items()
    }

    for key in ("delta_sasa", "polar_area", "apolar_area"):
        assert areas["D"][key] == pytest.approx(areas["H"][key], rel=1e-9), key


def test_the_static_area_does_not_change_when_the_hydrogens_are_deuterium(
    protonated_and_deuterated,
):
    areas = {
        label: compute_delta_sasa_static(str(path), "B", "A", hydrogens="keep")["delta_sasa"]
        for label, path in protonated_and_deuterated.items()
    }

    assert areas["D"] == pytest.approx(areas["H"], rel=1e-9)
