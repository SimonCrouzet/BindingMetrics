"""``metrics.energy`` and ``metrics.comparison`` use the shared backbone atom names.

Both carried their own ``{"N", "CA", "C", "O"}`` literal; the constant in
``core.residues`` replaces them and must equal it.
"""

from pathlib import Path

import pytest

from binding_metrics.core.residues import BACKBONE_HEAVY_ATOM_NAMES

OLD_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})
P53_MDM2 = Path(__file__).resolve().parents[1] / "data" / "example_linear_p53_1YCR.pdb"


def test_the_shared_constant_equals_the_old_literal():
    assert BACKBONE_HEAVY_ATOM_NAMES == OLD_BACKBONE_ATOM_NAMES


def test_the_energy_module_uses_the_shared_constant():
    pytest.importorskip("openmm")
    from binding_metrics.metrics import energy

    assert energy._BACKBONE_ATOM_NAMES == OLD_BACKBONE_ATOM_NAMES
    assert energy._BACKBONE_ATOM_NAMES is BACKBONE_HEAVY_ATOM_NAMES


def test_backbone_only_coordinates_keep_exactly_the_four_backbone_atoms():
    gemmi = pytest.importorskip("gemmi")
    from binding_metrics.metrics.comparison import _get_coords

    structure = gemmi.read_structure(str(P53_MDM2))
    coords, keys = _get_coords(structure, chain_filter="B", backbone_only=True)
    assert {atom_name for _, _, atom_name in keys} == OLD_BACKBONE_ATOM_NAMES
    assert len(keys) == 4 * 13  # p53 peptide: 13 residues
    assert coords.shape == (52, 3)
