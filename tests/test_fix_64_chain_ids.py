"""Author chain IDs of an OpenMM topology read from an mmCIF (#64).

OpenMM's ``PDBxFile`` names the chains by ``label_asym_id`` when the file has more label IDs
than author IDs, and every option and result of the package uses the author IDs. 1CWA has
OpenMM chains A (protein), B (peptide), C and D (waters) for the author chains A (protein and
140 waters) and C (peptide and 4 waters); 3P8F has A, B (peptide), C (GSH), D and E for the
author chains A and I.
"""

import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

from openmm import app  # noqa: E402

from binding_metrics.io.structures import (  # noqa: E402
    attach_author_chain_ids,
    author_chain_ids,
    copy_author_chain_ids,
    drop_other_protein_chains,
    load_structure,
    openmm_chain_id,
    strip_heterogens,
)

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"

_ATOMS = {"GLY": ("N", "CA", "C", "O"), "HOH": ("O",)}
_HEADER = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""


def _write_cif(
    path: Path, residues: list[tuple[str, str, str]], *, author_column: bool = True
) -> Path:
    """A minimal mmCIF with one residue per ``(residue name, label_asym_id, auth_asym_id)``.

    ``author_column=False`` leaves out ``_atom_site.auth_asym_id``.
    """
    header = _HEADER if author_column else _HEADER.replace("_atom_site.auth_asym_id\n", "")
    rows = []
    for number, (comp, label, auth) in enumerate(residues, start=1):
        for atom in _ATOMS[comp]:
            serial = len(rows) + 1
            x = 1.5 * serial
            author = f" {auth}" if author_column else ""
            rows.append(
                f"ATOM {serial} {atom[0]} {atom} . {comp} {label} 1 {number} ? {x} 0.0 0.0 "
                f"1.00 10.0 {number} {comp}{author} {atom} 1"
            )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _example(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return path


class TestBundledExamples:
    def test_1cwa_peptide_is_author_chain_c(self):
        topology, _ = load_structure(_example(CWA))
        assert [chain.id for chain in topology.chains()] == ["A", "B", "C", "D"]
        assert author_chain_ids(topology) == ["A", "C", "A", "C"]
        assert openmm_chain_id(topology, "C") == "B"

    def test_1cwa_author_id_shared_with_waters_names_the_protein_chain(self):
        topology, _ = load_structure(_example(CWA))
        # chain C of the topology holds the 140 waters of author chain A, D those of author chain C
        assert openmm_chain_id(topology, "A") == "A"
        assert openmm_chain_id(topology, "C") != "D"

    def test_3p8f_peptide_is_author_chain_i(self):
        topology, _ = load_structure(_example(SFTI))
        assert [chain.id for chain in topology.chains()] == ["A", "B", "C", "D", "E"]
        assert author_chain_ids(topology) == ["A", "I", "A", "A", "I"]
        assert openmm_chain_id(topology, "I") == "B"
        # the GSH ligand and the waters share author ID A with the protein
        assert openmm_chain_id(topology, "A") == "A"

    def test_unknown_or_water_only_author_id_is_none(self):
        topology, _ = load_structure(_example(CWA))
        assert openmm_chain_id(topology, "Z") is None
        # "B" and "D" are topology IDs, not author IDs, of this file
        assert openmm_chain_id(topology, "B") is None
        assert openmm_chain_id(topology, "D") is None

    def test_label_and_author_ids_equal(self):
        topology, _ = load_structure(_example(SOMATOSTATIN))
        assert author_chain_ids(topology) == [chain.id for chain in topology.chains()] == ["A"]
        assert openmm_chain_id(topology, "A") == "A"

    def test_pdb_input_is_unchanged(self):
        topology, _ = load_structure(_example(P53))
        assert author_chain_ids(topology) == [chain.id for chain in topology.chains()]
        assert openmm_chain_id(topology, "B") == "B"
        assert openmm_chain_id(topology, "A") == "A"
        assert openmm_chain_id(topology, "W") is None


class TestSyntheticFiles:
    def test_swapped_letters_are_resolved_by_author_id(self, tmp_path):
        """Label A is author B and label B is author A: author A names the second chain."""
        path = _write_cif(
            tmp_path / "swapped.cif",
            [("GLY", "A", "B")] * 3 + [("GLY", "B", "A")] * 3 + [("HOH", "C", "A")],
        )
        topology, _ = load_structure(path)
        assert [chain.id for chain in topology.chains()] == ["A", "B", "C"]
        assert author_chain_ids(topology) == ["B", "A", "A"]
        assert openmm_chain_id(topology, "A") == "B"
        assert openmm_chain_id(topology, "B") == "A"

    def test_waters_never_answer_while_the_protein_chain_exists(self, tmp_path):
        path = _write_cif(
            tmp_path / "shared.cif",
            [("GLY", "A", "A")] * 3
            + [("GLY", "B", "C")] * 2
            + [("HOH", "C", "A")] * 4
            + [("HOH", "D", "C")] * 2,
        )
        topology, _ = load_structure(path)
        assert [chain.id for chain in topology.chains()] == ["A", "B", "C", "D"]
        assert openmm_chain_id(topology, "A") == "A"
        assert openmm_chain_id(topology, "C") == "B"

    def test_a_waters_only_author_id_is_none(self, tmp_path):
        path = _write_cif(
            tmp_path / "waters.cif",
            [("GLY", "A", "A")] * 3 + [("HOH", "B", "W")] * 2 + [("HOH", "C", "A")],
        )
        topology, _ = load_structure(path)
        assert author_chain_ids(topology) == ["A", "W", "A"]
        assert openmm_chain_id(topology, "W") is None

    def test_amino_acids_of_one_author_id_in_two_chains_raise(self, tmp_path):
        path = _write_cif(
            tmp_path / "ambiguous.cif",
            [("GLY", "A", "A")] * 3 + [("GLY", "B", "A")] * 3 + [("HOH", "C", "A")],
        )
        topology, _ = load_structure(path)
        with pytest.raises(ValueError, match="author chain 'A'.*'A', 'B'"):
            openmm_chain_id(topology, "A")

    def test_a_file_without_author_ids_is_named_by_its_label_ids(self, tmp_path):
        path = _write_cif(
            tmp_path / "labels.cif",
            [("GLY", "A", "A")] * 3 + [("GLY", "B", "B")] * 2 + [("HOH", "C", "C")],
            author_column=False,
        )
        topology, _ = load_structure(path)
        assert author_chain_ids(topology) == ["A", "B", "C"]
        assert openmm_chain_id(topology, "B") == "B"

    def test_a_label_chain_split_into_two_runs_is_one_answer(self, tmp_path):
        """Two chains of the topology with the same ID name one chain, not an ambiguity."""
        path = _write_cif(
            tmp_path / "split.cif",
            [("GLY", "A", "A")] * 2 + [("HOH", "B", "A")] + [("GLY", "A", "A")] * 2,
        )
        topology, _ = load_structure(path)
        assert openmm_chain_id(topology, "A") == "A"


class TestAttachment:
    def test_a_derived_topology_gets_the_author_ids_by_chain_id(self):
        topology, positions = load_structure(_example(CWA))
        waters = [r for c in topology.chains() if c.id in ("C", "D") for r in c.residues()]
        modeller = app.Modeller(topology, positions)
        modeller.delete(waters)
        derived = modeller.topology
        assert [chain.id for chain in derived.chains()] == ["A", "B"]
        assert author_chain_ids(derived) == ["A", "B"]  # the copy made by Modeller carries none
        copy_author_chain_ids(topology, derived)
        assert author_chain_ids(derived) == ["A", "C"]
        assert openmm_chain_id(derived, "C") == "B"

    def test_a_derived_topology_is_annotated_again_from_the_file(self):
        topology, positions = load_structure(_example(CWA))
        modeller = app.Modeller(topology, positions)
        modeller.delete([r for c in topology.chains() if c.id == "C" for r in c.residues()])
        assert attach_author_chain_ids(modeller.topology, CWA)
        assert author_chain_ids(modeller.topology) == ["A", "C", "C"]

    def test_a_renamed_chain_voids_the_record(self):
        topology, _ = load_structure(_example(CWA))
        next(iter(topology.chains())).id = "Z"
        assert author_chain_ids(topology) == ["Z", "B", "C", "D"]

    def test_a_topology_of_another_file_is_not_annotated(self, caplog):
        topology, _ = load_structure(_example(P53))
        with caplog.at_level(logging.WARNING, logger="binding_metrics.io.structures"):
            assert not attach_author_chain_ids(topology, _example(CWA))
        assert "not read from this file" in caplog.text
        assert author_chain_ids(topology) == [chain.id for chain in topology.chains()]

    def test_without_gemmi_the_topology_keeps_its_own_ids(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gemmi", None)
        topology, _ = load_structure(_example(CWA))
        assert author_chain_ids(topology) == ["A", "B", "C", "D"]
        assert openmm_chain_id(topology, "B") == "B"


class TestRemovalMessages:
    """What strip_heterogens and drop_other_protein_chains say names author chains.

    Both still select by the IDs of the topology: their chain arguments are OpenMM IDs.
    """

    def test_strip_heterogens_names_the_ligand_chain_by_author_id(self, caplog):
        topology, positions = load_structure(_example(SFTI))
        report: dict = {}
        with caplog.at_level(logging.INFO, logger="binding_metrics.io.structures"):
            stripped, _ = strip_heterogens(topology, positions, "B", "A", report=report)
        # GSH is chain C of the topology (label ID) and chain A of the file (author ID)
        assert report["removed_heterogens"] == ["GSH (chain A)"]
        assert report["n_removed_waters"] == 101
        assert "GSH1001 (chain A)" in caplog.text
        assert "chain C" not in caplog.text
        assert [chain.id for chain in stripped.chains()] == ["A", "B"]

    def test_the_relaxation_wrapper_reports_the_same_names(self):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        topology, positions = load_structure(_example(SFTI))
        report: dict = {}
        ImplicitRelaxation(RelaxationConfig())._strip_heterogens(
            topology, positions, "B", "A", report=report
        )
        assert report["removed_heterogens"] == ["GSH (chain A)"]

    def test_dropped_protein_chains_are_author_ids(self, tmp_path, caplog):
        path = _write_cif(
            tmp_path / "three.cif",
            [("GLY", "A", "A")] * 4
            + [("GLY", "B", "C")] * 3
            + [("GLY", "C", "E")] * 2
            + [("HOH", "D", "A")] * 2,
        )
        topology, positions = load_structure(path)
        assert [chain.id for chain in topology.chains()] == ["A", "B", "C", "D"]
        report: dict = {}
        with caplog.at_level(logging.WARNING, logger="binding_metrics.io.structures"):
            kept, _ = drop_other_protein_chains(topology, positions, "B", "A", report=report)
        assert report["dropped_protein_chains"] == ["E"]
        assert "Removing protein chain(s) E: neither the peptide (C) nor the receptor (A)" in (
            caplog.text
        )
        assert [chain.id for chain in kept.chains()] == ["A", "B", "D"]

    def test_a_pdb_input_names_its_own_chains(self, caplog):
        topology, positions = load_structure(_example(P53))
        report: dict = {}
        with caplog.at_level(logging.WARNING, logger="binding_metrics.io.structures"):
            drop_other_protein_chains(topology, positions, "B", "A", report=report)
        assert report["dropped_protein_chains"] == []
