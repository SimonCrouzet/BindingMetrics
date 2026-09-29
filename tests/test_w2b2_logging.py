"""Library output that used to be printed now goes through ``logging``.

Each converted message is checked twice: it reaches the module logger (caplog),
and, once ``configure_logging()`` has run, it appears on stdout with exactly the
text the former ``print`` wrote.
"""

import logging
from pathlib import Path

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


# ---------------------------------------------------------------------------
# io/structures.py
# ---------------------------------------------------------------------------

STRUCTURES_LOGGER = "binding_metrics.io.structures"
SFTI_CIF = Path(__file__).parent.parent / "data" / "example_bicyclic_sfti1_3P8F.cif"


class TestDetectChainsLogging:
    expected_lines = [
        "  Chain detection (example_bicyclic_sfti1_3P8F.cif):",
        "    chain I: 14 residues  ← peptide [auto]",
        "    chain A: 227 residues  ← receptor [auto]",
    ]

    @pytest.fixture(autouse=True)
    def _example(self):
        if not SFTI_CIF.exists():
            pytest.skip(f"bundled example not found: {SFTI_CIF}")

    def test_chain_table_reaches_the_module_logger(self, caplog):
        from binding_metrics.io.structures import detect_chains_from_file

        with caplog.at_level(logging.INFO, logger=STRUCTURES_LOGGER):
            detect_chains_from_file(SFTI_CIF)
        assert [r.getMessage() for r in caplog.records] == self.expected_lines
        assert {r.levelno for r in caplog.records} == {logging.INFO}

    def test_stdout_text_is_the_former_print(self, console_logging, capsys):
        from binding_metrics.io.structures import detect_chains_from_file

        console_logging()
        detect_chains_from_file(SFTI_CIF)
        assert capsys.readouterr().out == "\n".join(self.expected_lines) + "\n"

    def test_verbose_false_stays_silent(self, caplog):
        from binding_metrics.io.structures import detect_chains_from_file

        with caplog.at_level(logging.DEBUG, logger="binding_metrics"):
            detect_chains_from_file(SFTI_CIF, verbose=False)
        assert caplog.records == []


def _protein_with_ions(ion_distances_angstrom):
    """One alanine (chain A, CA at the origin) plus one single-atom ion per chain.

    Ion ``k`` is residue ``k + 2`` in its own chain (X, Y, ...) at the given
    distance from the alanine along x.
    """
    from openmm.app import Topology, element

    topology = Topology()
    protein = topology.addResidue("ALA", topology.addChain("A"), id="1")
    topology.addAtom("CA", element.carbon, protein)
    coords = [Vec3(0.0, 0.0, 0.0)]
    for k, distance in enumerate(ion_distances_angstrom):
        chain = topology.addChain("XYZ"[k])
        ion = topology.addResidue(("ZN", "CL", "MG")[k], chain, id=str(k + 2))
        topology.addAtom(("ZN", "CL", "MG")[k], element.zinc, ion)
        coords.append(Vec3(distance / 10.0, 0.0, 0.0))
    return topology, unit.Quantity(coords, unit.nanometer)


class TestStripHeterogensLogging:
    close = (
        "  Warning: removing heterogen ZN2 (chain X) which is 3.0 Å from the protein — "
        "it may be a functional cofactor or ion. "
        "Parametrize it via custom_bond_handler to keep it."
    )
    distant = "  Removing distant heterogen CL3 (chain Y, 20.0 Å from protein)"
    no_protein = "  Removing heterogen ZN2 (chain X)"

    def test_close_and_distant_heterogens_reach_the_logger_with_their_levels(self, caplog):
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = _protein_with_ions([3.0, 20.0])
        with caplog.at_level(logging.INFO, logger=STRUCTURES_LOGGER):
            strip_heterogens(topology, positions, "A", None)
        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (logging.WARNING, self.close),
            (logging.INFO, self.distant),
        ]

    def test_heterogen_without_a_protein_chain_is_logged_at_info(self, caplog):
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = _protein_with_ions([3.0])
        with caplog.at_level(logging.INFO, logger=STRUCTURES_LOGGER):
            strip_heterogens(topology, positions, None, None)
        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (logging.INFO, self.no_protein)
        ]

    def test_stdout_text_is_the_former_print(self, console_logging, capsys):
        from binding_metrics.io.structures import strip_heterogens

        console_logging()
        topology, positions = _protein_with_ions([3.0, 20.0])
        strip_heterogens(topology, positions, "A", None)
        assert capsys.readouterr().out == f"{self.close}\n{self.distant}\n"
