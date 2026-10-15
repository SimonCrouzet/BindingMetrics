"""``prep_structure`` keeps the head-to-tail ring of a model that lists no bond (#196).

A model output has no ``_struct_conn``, so nothing in the loaded topology says that the last
carbonyl carbon is bonded to the first nitrogen. The ring is found from the distance, as the
energy and the relaxation find it, and prep takes the cyclic route: the last residue gets no
OXT, the first one a single N-terminal hydrogen, and the closure bond is in the result.
"""

import os
from pathlib import Path

import pytest
from conftest import DATA_DIR

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

from binding_metrics.core.system import prep_structure
from binding_metrics.io.structures import load_structure
from binding_metrics.metrics._common import load_structure as load_atoms
from binding_metrics.testing.synthetic import write_cif

SFTI = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
CWA = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
SOMATOSTATIN = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"
P53 = DATA_DIR / "example_linear_p53_1YCR.pdb"


def _bare(source: Path, target: Path) -> Path:
    """The structure without waters, written as a model: no bond record of any kind."""
    atoms = load_atoms(source)
    write_cif(atoms[atoms.res_name != "HOH"], target)
    assert "_struct_conn" not in target.read_text(encoding="utf-8")
    return target


def _prepared(path: Path):
    topology, positions = load_structure(path)
    return prep_structure(topology, positions)[0]


def _residues(topology, n_residues: int) -> list:
    """The residues of the chain that has ``n_residues`` of them."""
    chain = next(c for c in topology.chains() if len(list(c.residues())) == n_residues)
    return list(chain.residues())


def _atom_names(topology, n_residues: int) -> list:
    return [[a.name for a in r.atoms()] for r in _residues(topology, n_residues)]


def _is_closed(topology, n_residues: int) -> bool:
    residues = _residues(topology, n_residues)
    carbon = next(a for a in residues[-1].atoms() if a.name == "C")
    nitrogen = next(a for a in residues[0].atoms() if a.name == "N")
    return any({b.atom1, b.atom2} == {carbon, nitrogen} for b in topology.bonds())


def _names(residue) -> set:
    return {a.name for a in residue.atoms()}


@pytest.mark.slow
def test_a_raw_ring_is_prepared_as_the_listed_one(tmp_path):
    raw = _prepared(_bare(SFTI, tmp_path / "bare.cif"))
    listed = _prepared(SFTI)

    assert _is_closed(raw, 14)
    assert _atom_names(raw, 14) == _atom_names(listed, 14)
    residues = _residues(raw, 14)
    assert _names(residues[0]) & {"H", "H1", "H2", "H3"} == {"H"}
    assert "OXT" not in _names(residues[-1])


def test_a_disulfide_ring_keeps_its_two_termini(tmp_path):
    """Somatostatin is closed by a disulfide, not head to tail: both ends stay charged."""
    topology = _prepared(_bare(SOMATOSTATIN, tmp_path / "bare.cif"))

    first, *_, last = _residues(topology, 12)

    assert {"H", "H2", "H3"} <= _names(first)
    assert "OXT" in _names(last)
    assert not _is_closed(topology, 12)


def test_a_linear_peptide_keeps_its_two_termini(tmp_path):
    topology = _prepared(P53)

    first, *_, last = _residues(topology, 13)

    assert {"H", "H2", "H3"} <= _names(first)
    assert "OXT" in _names(last)
    assert not _is_closed(topology, 13)


@pytest.mark.slow
@pytest.mark.integration
def test_a_raw_ring_with_non_standard_residues_is_prepared(tmp_path):
    topology = _prepared(_bare(CWA, tmp_path / "bare.cif"))

    first, *_, last = _residues(topology, 11)

    assert _is_closed(topology, 11)
    assert "OXT" not in _names(last)
    assert [a.name for a in first.atoms()] == "N H CA HA CB HB1 HB2 HB3 C O".split()


SAVED = os.environ.get("BM_SAVED_PREDICTIONS")
SAVED_1CWA = (
    "boltz2/*/*/outputs/boltz_results_cwa_auto_s42/predictions/cwa_auto_s42/"
    "cwa_auto_s42_model_0.cif"
)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not SAVED, reason="BM_SAVED_PREDICTIONS names no directory of saved predictions"
)
def test_a_saved_prediction_of_the_ring_is_prepared():
    found = sorted(Path(SAVED).glob(SAVED_1CWA))
    if not found:
        pytest.skip(f"no saved prediction matches {SAVED_1CWA}")

    topology = _prepared(found[0])

    first, *_, last = _residues(topology, 11)
    assert _is_closed(topology, 11)
    assert "OXT" not in _names(last)
    assert _names(first) & {"H", "H1", "H2", "H3"} == {"H"}
