"""Prep names the chains of an mmCIF by their author IDs (#64).

OpenMM names the chains of 1CWA A (protein), B (peptide), C and D (waters) where the file's
author IDs are A (protein and waters) and C (peptide and waters); 3P8F has A, B (peptide),
C (GSH), D and E where the author IDs are A and I.
"""

import logging
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")
pytest.importorskip("gemmi")

from openmm import app  # noqa: E402

from binding_metrics.core import system  # noqa: E402
from binding_metrics.core.system import find_chain_breaks, prep_structure  # noqa: E402
from binding_metrics.io.structures import (  # noqa: E402
    attach_author_chain_ids,
    author_chain_ids,
    load_structure,
    openmm_chain_id,
)

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"


def _load(path):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return load_structure(path)


@pytest.fixture
def skip_hydrogens(monkeypatch):
    """Leave out the hydrogen step of a cyclic peptide (GAFF templates, about 20 s).

    The chain names of the report are fixed before that step.
    """
    monkeypatch.setattr(
        system, "_add_hydrogens_cyclic", lambda top, pos, *args, **kwargs: (top, pos)
    )


def _delete_residue(topology, positions, chain_id, index_in_chain, source):
    """The topology without one residue, annotated again from ``source``."""
    chain = next(c for c in topology.chains() if c.id == chain_id)
    modeller = app.Modeller(topology, positions)
    modeller.delete([list(chain.residues())[index_in_chain]])
    assert attach_author_chain_ids(modeller.topology, source)
    return modeller.topology, modeller.positions


class TestPrepReport:
    def test_1cwa_kept_residues_are_in_author_chain_c(self, skip_hydrogens):
        topology, positions = _load(CWA)
        report: dict = {}
        prepped, _ = prep_structure(topology, positions, report=report)
        kept = report["kept_nonstandard"]
        assert sorted(set(kept)) == sorted(
            {f"{name} (chain C)" for name in ("DAL", "MLE", "MVA", "BMT", "ABA", "SAR")}
        )
        assert kept.count("MLE (chain C)") == 4
        assert report["removed_heterogens"] == []
        # The topology keeps OpenMM's IDs, and says which author chain each one is.
        assert [chain.id for chain in prepped.chains()] == ["A", "B"]
        assert author_chain_ids(prepped) == ["A", "C"]
        assert openmm_chain_id(prepped, "C") == "B"

    def test_3p8f_ligand_is_in_author_chain_a(self):
        topology, positions = _load(SFTI)
        report: dict = {}
        prepped, _ = prep_structure(topology, positions, report=report)
        # GSH has label ID C in the file and author ID A (the trypsin chain it binds)
        assert report["removed_heterogens"] == ["GSH (chain A)"]
        assert author_chain_ids(prepped) == ["A", "I"]

    def test_the_log_names_the_author_chain(self, skip_hydrogens, caplog):
        topology, positions = _load(CWA)
        with caplog.at_level(logging.INFO, logger="binding_metrics.core.system"):
            prep_structure(topology, positions)
        assert "DAL (chain C)" in caplog.text
        assert "chain B" not in caplog.text

    def test_a_pdb_input_is_unchanged(self):
        topology, positions = _load(P53)
        report: dict = {}
        prepped, _ = prep_structure(topology, positions, report=report)
        assert report["removed_heterogens"] == []
        assert author_chain_ids(prepped) == [chain.id for chain in prepped.chains()]


class TestChainBreaks:
    def test_a_break_in_the_1cwa_peptide_is_in_author_chain_c(self, skip_hydrogens, caplog):
        topology, positions = _delete_residue(*_load(CWA), "B", 4, CWA)
        breaks = find_chain_breaks(topology, positions)
        assert [(b["chain"], b["residue_before"], b["residue_after"]) for b in breaks] == [
            ("C", "4", "6")
        ]
        report: dict = {}
        with caplog.at_level(logging.WARNING, logger="binding_metrics.core.system"):
            prep_structure(topology, positions, report=report)
        assert [b["chain"] for b in report["chain_breaks"]] == ["C"]
        assert "Chain break in chain C" in caplog.text

    def test_a_topology_without_author_ids_names_its_own_chains(self):
        topology, positions = _load(CWA)
        chain = next(c for c in topology.chains() if c.id == "B")
        modeller = app.Modeller(topology, positions)
        modeller.delete([list(chain.residues())[4]])
        breaks = find_chain_breaks(modeller.topology, modeller.positions)
        assert [b["chain"] for b in breaks] == ["B"]
