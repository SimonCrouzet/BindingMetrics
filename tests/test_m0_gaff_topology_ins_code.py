"""The topology rebuilt around injected hydrogens keeps the insertion codes of its residues."""

import numpy as np
import pytest

pytest.importorskip("openmm")

from openmm.app import Topology  # noqa: E402
from openmm.app import element as elem  # noqa: E402

from binding_metrics.core.gaff_ncaa import _rebuild_topology_with_injected_h  # noqa: E402


def _topology_with_insertion_codes():
    """Chain A with residues 52, 52A and 53, one carbon each, and the matching positions (nm)."""
    topology = Topology()
    chain = topology.addChain("A")
    for number, ins_code in ((52, ""), (52, "A"), (53, "")):
        residue = topology.addResidue("ALA", chain, id=str(number), insertionCode=ins_code)
        topology.addAtom("CA", elem.carbon, residue)
    return topology, np.arange(9, dtype=float).reshape(3, 3) * 0.1


def test_insertion_codes_survive_the_rebuild():
    """Regression: only ``id`` was copied, so residues 52 and 52A both came out as "52"."""
    topology, positions = _topology_with_insertion_codes()
    rebuilt, _ = _rebuild_topology_with_injected_h(topology, positions, {})
    assert [(r.id, r.insertionCode) for r in rebuilt.residues()] == [
        ("52", ""),
        ("52", "A"),
        ("53", ""),
    ]


def test_insertion_codes_survive_when_hydrogens_are_injected():
    topology, positions = _topology_with_insertion_codes()
    residue_52a = list(topology.residues())[1]
    injected = {residue_52a.index: [("HA", "CA", np.array([0.5, 0.5, 0.5]))]}
    rebuilt, new_positions = _rebuild_topology_with_injected_h(topology, positions, injected)
    assert [(r.id, r.insertionCode) for r in rebuilt.residues()] == [
        ("52", ""),
        ("52", "A"),
        ("53", ""),
    ]
    assert [a.name for a in rebuilt.atoms()] == ["CA", "CA", "HA", "CA"]
    assert len(new_positions) == 4


def test_residues_without_insertion_codes_stay_without():
    topology = Topology()
    chain = topology.addChain("A")
    for number in (1, 2):
        residue = topology.addResidue("GLY", chain, id=str(number))
        topology.addAtom("CA", elem.carbon, residue)
    rebuilt, _ = _rebuild_topology_with_injected_h(topology, np.zeros((2, 3)), {})
    assert [(r.id, r.insertionCode.strip()) for r in rebuilt.residues()] == [("1", ""), ("2", "")]
