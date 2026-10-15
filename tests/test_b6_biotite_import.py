"""The metric modules call the shared ``import_biotite`` and keep their install hints.

Each of ``electrostatics``, ``evobind``, ``geometry``, ``interface``, ``polar_contacts`` and
``receptor_quality`` used to wrap ``import_biotite(purpose)`` in a private
``_import_biotite``, and ``predictors/_confidence`` in ``_import_biotite_struc``. The wrappers
are gone; a function that needs biotite names the
same feature in the install hint as before.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics import (  # noqa: E402
    electrostatics,
    evobind,
    geometry,
    interface,
    polar_contacts,
    receptor_quality,
)
from binding_metrics.metrics._common import load_structure  # noqa: E402
from binding_metrics.predictors import _confidence  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"

BLOCKED = {
    "biotite": None,
    "biotite.structure": None,
    "biotite.structure.io": None,
    "biotite.structure.io.pdb": None,
    "biotite.structure.io.pdbx": None,
}

WRAPPERS_REMOVED = [
    electrostatics,
    evobind,
    geometry,
    interface,
    polar_contacts,
    receptor_quality,
    _confidence,
]


@pytest.fixture(scope="module")
def atoms():
    """The peptide of 1YCR and a few receptor residues around it (small enough for hydride)."""
    structure = load_structure(P53, purpose="tests")
    near_peptide = (structure.chain_id == "A") & (structure.res_id >= 54) & (structure.res_id <= 62)
    return structure[(structure.chain_id == "B") | near_peptide]


def _hint(purpose):
    return f"biotite is required for {purpose}. Install with: pip install binding-metrics[biotite]"


# (id, call of the function that reaches the import with a loaded structure, purpose)
CALLS = [
    (
        "electrostatics",
        lambda atoms: electrostatics._collect_charged_atoms(atoms),
        "electrostatics metrics",
    ),
    (
        "evobind",
        lambda atoms: evobind._adversarial_from_atoms(atoms, atoms, None, "B", "A", 8.0, 0.5),
        "EvoBind metrics",
    ),
    (
        "geometry",
        lambda atoms: geometry._terminal_backbone_coords(atoms[atoms.chain_id == "B"]),
        "geometry metrics",
    ),
    (
        "interface",
        lambda atoms: interface._amino_acid_mask(atoms),
        "interface metrics",
    ),
    (
        "confidence",
        lambda atoms: _confidence._binder_ca_rmsd(atoms, atoms, "B"),
        "per-chain structural analysis",
    ),
    (
        "polar_contacts",
        lambda atoms: polar_contacts.compute_hbonds(atoms, "B", "A"),
        "H-bond/salt bridge metrics",
    ),
    (
        "receptor_quality",
        lambda atoms: list(receptor_quality._iter_residues(atoms)),
        "receptor quality metrics",
    ),
]


@pytest.mark.parametrize("module", WRAPPERS_REMOVED, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_the_private_wrapper_is_gone(module):
    assert not hasattr(module, "_import_biotite")
    assert not hasattr(module, "_import_biotite_struc")


@pytest.mark.parametrize("call, purpose", [c[1:] for c in CALLS], ids=[c[0] for c in CALLS])
def test_a_missing_biotite_names_the_same_feature_as_before(atoms, call, purpose):
    with mock.patch.dict(sys.modules, BLOCKED):
        with pytest.raises(ImportError) as excinfo:
            call(atoms)
    assert str(excinfo.value) == _hint(purpose)


# compute_hbonds adds hydrogens with hydride, which takes seconds; the tests of the polar
# contacts run it with biotite present.
@pytest.mark.parametrize(
    "call",
    [c[1] for c in CALLS if c[0] != "polar_contacts"],
    ids=[c[0] for c in CALLS if c[0] != "polar_contacts"],
)
def test_the_functions_still_run_with_biotite(atoms, call):
    """The same calls succeed when biotite is there."""
    call(atoms)
