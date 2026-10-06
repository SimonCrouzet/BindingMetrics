"""Residue labels in logs and results carry the insertion code (#128).

OpenMM keeps the insertion code in ``Residue.insertionCode`` and not in ``Residue.id``, so a
label built from ``id`` names residues 60 and 60A alike.
"""

import logging

import pytest

pytest.importorskip("openmm")

from openmm import Vec3, unit  # noqa: E402
from openmm.app import Topology  # noqa: E402
from openmm.app import element as elem  # noqa: E402

from binding_metrics.core.residues import residue_label, residue_number  # noqa: E402
from binding_metrics.core.system import find_chain_breaks  # noqa: E402
from binding_metrics.io.structures import strip_heterogens  # noqa: E402
from binding_metrics.protocols.qc import AtomSnapshot, _format_residue  # noqa: E402
from binding_metrics.protocols.relaxation import _cyclic_atom_label  # noqa: E402

_BACKBONE = (("N", elem.nitrogen, 0.0), ("CA", elem.carbon, 0.15), ("C", elem.carbon, 0.25))


def _glycine_chain(numbering, gaps_nm=None):
    """Chain A of glycines (N, CA, C) numbered by ``(id, insertion code)``; a peptide bond
    (0.133 nm) joins neighbours except where ``gaps_nm`` maps the residue index to a distance."""
    gaps_nm = gaps_nm or {}
    topology = Topology()
    chain = topology.addChain("A")
    positions = []
    x_start = 0.0
    for index, (number, ins_code) in enumerate(numbering):
        residue = topology.addResidue("GLY", chain, id=str(number), insertionCode=ins_code)
        for name, element, offset in _BACKBONE:
            topology.addAtom(name, element, residue)
            positions.append(Vec3(x_start + offset, 0.0, 0.0))
        x_start += 0.25 + gaps_nm.get(index + 1, 0.133)
    return topology, unit.Quantity(positions, unit.nanometer)


class TestHelpers:
    def test_number_and_label_append_the_insertion_code(self):
        topology, _ = _glycine_chain([(60, ""), (60, "A"), (61, " ")])
        plain, coded, blank = topology.residues()
        assert (residue_number(plain), residue_label(plain)) == ("60", "GLY60")
        assert (residue_number(coded), residue_label(coded)) == ("60A", "GLY60A")
        assert (residue_number(blank), residue_label(blank)) == ("61", "GLY61")


class TestChainBreaks:
    def test_break_between_60B_and_60C_is_reported_as_such(self):
        """Regression: it was reported as 60 -> 60."""
        topology, positions = _glycine_chain(
            [(59, ""), (60, ""), (60, "A"), (60, "B"), (60, "C"), (61, "")], gaps_nm={4: 0.5}
        )
        breaks = find_chain_breaks(topology, positions)
        assert [(b["residue_before"], b["residue_after"]) for b in breaks] == [("60B", "60C")]

    def test_break_without_insertion_codes_keeps_its_labels(self):
        topology, positions = _glycine_chain([(22, ""), (23, ""), (24, "")], gaps_nm={1: 0.5})
        breaks = find_chain_breaks(topology, positions)
        assert [(b["residue_before"], b["residue_after"]) for b in breaks] == [("22", "23")]


class TestClosureAtomLabel:
    def test_label_names_chain_number_with_code_and_atom(self):
        numbers = {"B": ["60", "60A", "61"]}
        assert _cyclic_atom_label(("B", 1, "SG"), numbers, {"B": "C"}) == "C:60A:SG"
        assert _cyclic_atom_label(("B", 2, "N"), numbers, {"B": "C"}) == "C:61:N"

    def test_index_beyond_the_chain_and_unknown_chain_are_printed_as_they_are(self):
        assert _cyclic_atom_label(("B", 7, "C"), {"B": ["1"]}, {}) == "B:7:C"
        assert _cyclic_atom_label(("Z", 0, "C"), {}, {}) == "Z:0:C"


class TestSnapshotKeys:
    def test_topology_snapshot_tells_60_and_60A_apart_and_prints_the_code(self):
        topology, positions = _glycine_chain([(60, ""), (60, "A")])
        snapshot = AtomSnapshot.from_topology(topology, positions)
        keys = sorted(set(snapshot.residue_keys))
        assert [_format_residue(key) for key in keys] == ["A:60", "A:60A"]
        assert len(keys) == 2


class TestStripHeterogensLog:
    def test_removed_heterogen_is_logged_with_its_insertion_code(self, caplog):
        topology, positions = _glycine_chain([(1, "")])
        chain = topology.addChain("B")
        ion = topology.addResidue("MG", chain, id="101", insertionCode="A")
        topology.addAtom("MG", elem.magnesium, ion)
        coordinates = list(positions.value_in_unit(unit.nanometer)) + [Vec3(5.0, 0.0, 0.0)]
        positions = unit.Quantity(coordinates, unit.nanometer)
        with caplog.at_level(logging.INFO, logger="binding_metrics.io.structures"):
            strip_heterogens(topology, positions, "A", None)
        messages = [record.getMessage() for record in caplog.records]
        assert any("MG101A" in message for message in messages), messages
