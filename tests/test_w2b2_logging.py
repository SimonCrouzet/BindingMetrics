"""Library output that used to be printed now goes through ``logging``.

Each converted message is checked twice: it reaches the module logger (caplog),
and, once ``configure_logging()`` has run, it appears on stdout with exactly the
text the former ``print`` wrote.
"""

import logging

import numpy as np
import pytest
from openmm import Vec3, unit
from openmm.app import Topology, element

from binding_metrics.utils import _CurrentStreamHandler, configure_logging


@pytest.fixture
def console_logging():
    """Undo ``configure_logging`` after the test so later tests see the default setup."""
    package_logger = logging.getLogger("binding_metrics")
    saved_level = package_logger.level
    yield configure_logging
    for name in ("binding_metrics", "__main__"):
        target = logging.getLogger(name)
        for handler in [h for h in target.handlers if isinstance(h, _CurrentStreamHandler)]:
            target.removeHandler(handler)
    package_logger.setLevel(saved_level)


# ---------------------------------------------------------------------------
# core/system.py
# ---------------------------------------------------------------------------


def _alanine_with_ha_on_the_cb_face():
    """One residue whose HA sits on the same face as CB, which the repair must fix."""
    tetra = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
    ]
    unit_vectors = [v / np.linalg.norm(v) for v in tetra]
    coords = {
        "N": unit_vectors[0] * 0.145,
        "CA": np.zeros(3),
        "C": unit_vectors[1] * 0.152,
        "CB": unit_vectors[2] * 0.153,
        "HA": unit_vectors[2] * 0.109,
    }
    topology = Topology()
    residue = topology.addResidue("ALA", topology.addChain("A"))
    elements = {
        "N": element.nitrogen,
        "CA": element.carbon,
        "C": element.carbon,
        "CB": element.carbon,
        "HA": element.hydrogen,
    }
    for name in coords:
        topology.addAtom(name, elements[name], residue)
    positions = unit.Quantity([Vec3(*coords[a.name]) for a in topology.atoms()], unit.nanometer)
    return topology, positions


class TestRepairCaHydrogenChiralityLogging:
    expected = "  Repaired 1 wrong-side Cα hydrogen(s): ALA1/A"

    def test_message_reaches_the_module_logger(self, caplog):
        from binding_metrics.core.system import repair_ca_hydrogen_chirality

        topology, positions = _alanine_with_ha_on_the_cb_face()
        with caplog.at_level(logging.INFO, logger="binding_metrics.core.system"):
            repair_ca_hydrogen_chirality(topology, positions)
        assert [r.getMessage() for r in caplog.records] == [self.expected]
        assert caplog.records[0].levelno == logging.INFO

    def test_stdout_text_is_the_former_print(self, console_logging, capsys):
        from binding_metrics.core.system import repair_ca_hydrogen_chirality

        console_logging()
        topology, positions = _alanine_with_ha_on_the_cb_face()
        repair_ca_hydrogen_chirality(topology, positions)
        assert capsys.readouterr().out == self.expected + "\n"

    def test_verbose_false_stays_silent(self, caplog):
        from binding_metrics.core.system import repair_ca_hydrogen_chirality

        topology, positions = _alanine_with_ha_on_the_cb_face()
        with caplog.at_level(logging.DEBUG, logger="binding_metrics"):
            repair_ca_hydrogen_chirality(topology, positions, verbose=False)
        assert caplog.records == []
