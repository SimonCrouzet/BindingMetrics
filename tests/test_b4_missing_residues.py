"""Prep reports the residues a model is missing and adds none.

Missing residues come from the sequence records of the input when it has them
(SEQRES of a PDB file, ``_entity_poly_seq`` of an mmCIF) and from the residue numbers
and the chain geometry otherwise. Terminal tails need the records; a gap in the
numbering does not. Nothing is rebuilt: the report says where the model is truncated.
"""

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from openmm import Vec3, unit
from openmm.app import Modeller, Topology, element

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"
SFTI_TRYPSIN = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"
CYCLOSPORIN = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
LACTAM = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"
PHOSPHO = DATA_DIR / "example_phospho_1QJB.pdb"
STAPLED = DATA_DIR / "example_staple_3V3B.pdb"

_BONDED_STEP_ANGSTROM = 4.31  # N to the next N when C(i)-N(i+1) is 1.33 A


def _backbones(specs, chain_id="A"):
    """One chain of N-CA-C backbones: ``specs`` holds (number, insertion code, x in A)."""
    topology = Topology()
    chain = topology.addChain(chain_id)
    coords = []
    for number, insertion_code, x in specs:
        residue = topology.addResidue("ALA", chain, str(number), insertion_code)
        for name, symbol, dx in (("N", element.nitrogen, 0.0), ("CA", element.carbon, 1.46)):
            topology.addAtom(name, symbol, residue)
            coords.append(Vec3(x + dx, 0, 0) / 10.0)
        topology.addAtom("C", element.carbon, residue)
        coords.append(Vec3(x + 2.98, 0, 0) / 10.0)
    return topology, unit.Quantity(coords, unit.nanometer)


def _bonded_run(numbers_and_codes, start_x=0.0):
    """Backbones that follow each other at peptide-bond distance."""
    return [
        (number, code, start_x + k * _BONDED_STEP_ANGSTROM)
        for k, (number, code) in enumerate(numbers_and_codes)
    ]


def _missing(topology, positions):
    from binding_metrics.core.system import find_missing_residues

    return find_missing_residues(topology, positions)


class TestNumberingAndGeometry:
    """No sequence records: the numbers and the C-N distances decide."""

    def test_a_continuous_chain_has_nothing_missing(self):
        topology, positions = _backbones(_bonded_run([(1, ""), (2, ""), (3, "")]))
        assert _missing(topology, positions) == []

    def test_a_numbering_gap_is_listed_with_its_neighbours_and_size(self):
        specs = _bonded_run([(1, ""), (2, ""), (3, "")]) + [(7, "", 20.0)]
        topology, positions = _backbones(specs)
        (entry,) = _missing(topology, positions)
        assert entry == {
            "chain": "A",
            "kind": "gap",
            "terminus": None,
            "n_residues": 3,
            "residue_before": "3",
            "residue_after": "7",
            "sequence_positions": None,
            "detected_by": "numbering",
        }

    def test_residues_with_insertion_codes_are_not_a_gap(self):
        numbered = [(59, ""), (60, ""), (60, "A"), (60, "B"), (61, "")]
        topology, positions = _backbones(_bonded_run(numbered))
        assert _missing(topology, positions) == []

    def test_a_gap_after_an_insertion_code_names_it(self):
        specs = _bonded_run([(59, ""), (60, ""), (60, "A")]) + [(62, "", 20.0)]
        topology, positions = _backbones(specs)
        (entry,) = _missing(topology, positions)
        assert (entry["residue_before"], entry["residue_after"]) == ("60A", "62")
        assert entry["n_residues"] == 1

    def test_a_renumbered_chain_with_intact_bonds_is_not_a_gap(self):
        topology, positions = _backbones(_bonded_run([(1, ""), (2, ""), (101, ""), (102, "")]))
        assert _missing(topology, positions) == []

    def test_a_chain_break_with_consecutive_numbers_is_listed_without_a_size(self):
        specs = _bonded_run([(1, ""), (2, "")]) + [(3, "", 20.0)]
        topology, positions = _backbones(specs)
        (entry,) = _missing(topology, positions)
        assert entry["kind"] == "gap"
        assert entry["detected_by"] == "chain_break"
        assert entry["n_residues"] is None
        assert (entry["residue_before"], entry["residue_after"]) == ("2", "3")

    def test_a_break_that_is_also_a_numbering_jump_is_listed_once(self):
        specs = _bonded_run([(1, ""), (2, "")]) + [(6, "", 20.0)]
        topology, positions = _backbones(specs)
        (entry,) = _missing(topology, positions)
        assert entry["detected_by"] == "numbering"
        assert entry["n_residues"] == 3

    def test_waters_between_residues_are_not_residues_of_the_chain(self):
        topology, positions = _backbones(_bonded_run([(1, ""), (2, "")]))
        chain = next(topology.chains())
        water = topology.addResidue("HOH", chain, "500")
        topology.addAtom("O", element.oxygen, water)
        coords = list(positions.value_in_unit(unit.nanometer)) + [Vec3(5.0, 5.0, 5.0)]
        assert _missing(topology, unit.Quantity(coords, unit.nanometer)) == []

    def test_a_calcium_ion_named_ca_is_not_a_residue_of_the_chain(self):
        specs = _bonded_run([(1, ""), (2, ""), (3, "")])
        topology, positions = _backbones(specs)
        # An ion in the middle of the chain, as a HETATM of the same chain ID is read.
        # Its atom is called CA, like the alpha carbon of a residue.
        residues = list(topology.residues())
        positions_nm = positions.value_in_unit(unit.nanometer)
        rebuilt = Topology()
        chain = rebuilt.addChain("A")
        coords = []
        for index, residue in enumerate(residues):
            new_residue = rebuilt.addResidue(residue.name, chain, residue.id)
            for atom in residue.atoms():
                rebuilt.addAtom(atom.name, atom.element, new_residue)
                coords.append(Vec3(*positions_nm[atom.index]))
            if index == 1:
                ion = rebuilt.addResidue("CA", chain, "900")
                rebuilt.addAtom("CA", element.calcium, ion)
                coords.append(Vec3(5.0, 5.0, 5.0))
        assert _missing(rebuilt, unit.Quantity(coords, unit.nanometer)) == []

    def test_the_count_of_numbering_gaps_ignores_a_calcium_ion_named_ca(self):
        from binding_metrics.core.system import _count_residue_gaps

        topology, positions = _backbones(_bonded_run([(1, ""), (2, ""), (3, "")]))
        residues = list(topology.residues())
        positions_nm = positions.value_in_unit(unit.nanometer)
        rebuilt = Topology()
        chain = rebuilt.addChain("A")
        coords = []
        for index, residue in enumerate(residues):
            new_residue = rebuilt.addResidue(residue.name, chain, residue.id)
            for atom in residue.atoms():
                rebuilt.addAtom(atom.name, atom.element, new_residue)
                coords.append(Vec3(*positions_nm[atom.index]))
            if index == 1:
                ion = rebuilt.addResidue("CA", chain, "900")
                rebuilt.addAtom("CA", element.calcium, ion)
                coords.append(Vec3(5.0, 5.0, 5.0))
        assert _count_residue_gaps(rebuilt, unit.Quantity(coords, unit.nanometer)) == 0


# ---------------------------------------------------------------- sequence records

_ATOM_NAMES = {"N": " N  ", "CA": " CA ", "C": " C  "}


def _write_pdb(path: Path, residues, seqres=None, chain_id="A") -> Path:
    """A PDB file of N-CA-C backbones; ``residues`` holds (name, number, insertion code)."""
    lines = []
    if seqres:
        names = list(seqres)
        for k in range(0, len(names), 13):
            lines.append(
                f"SEQRES{k // 13 + 1:4d} {chain_id} {len(names):4d}  " + " ".join(names[k : k + 13])
            )
    serial = 0
    for k, (name, number, code) in enumerate(residues):
        for atom, dx in (("N", 0.0), ("CA", 1.46), ("C", 2.98)):
            serial += 1
            x = k * _BONDED_STEP_ANGSTROM + dx
            lines.append(
                f"ATOM  {serial:5d} {_ATOM_NAMES[atom]} {name:>3s} {chain_id}{number:4d}{code:1s}"
                f"   {x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom[0]:>2s}"
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _records(path: Path):
    from binding_metrics.io.structures import load_structure, sequence_records

    topology, positions = load_structure(path)
    return topology, positions, sequence_records(topology)


class TestSequenceRecordsOfAPdbFile:
    def test_seqres_is_aligned_to_the_residues_of_each_chain(self):
        topology, _, records = _records(P53_MDM2)
        by_chain = {c.id: r for c, r in zip(topology.chains(), records)}
        receptor, peptide = by_chain["A"], by_chain["B"]
        assert (receptor.source, len(receptor.residues)) == ("SEQRES", 109)
        assert (peptide.source, len(peptide.residues)) == ("SEQRES", 15)
        # MDM2 is modelled from position 9 to 93 (numbers 25 to 109), p53 from 3 to 15.
        placed = [p for p in receptor.positions if p is not None]
        assert (placed[0], placed[-1], len(placed)) == (9, 93, 85)
        placed = [p for p in peptide.positions if p is not None]
        assert (placed[0], placed[-1], len(placed)) == (3, 15, 13)

    def test_a_pdb_file_without_seqres_has_no_records(self):
        topology, _, records = _records(PHOSPHO)
        assert records == [None] * topology.getNumChains()

    def test_a_residue_the_sequence_does_not_have_leaves_the_chain_without_records(self, tmp_path):
        residues = [("ALA", k, "") for k in range(1, 5)]
        path = _write_pdb(tmp_path / "x.pdb", residues, seqres=["GLY", "SER", "THR", "PHE"])
        assert _records(path)[2] == [None]

    def test_a_sequence_that_fits_in_two_places_is_not_guessed(self, tmp_path):
        residues = [("GLY", 5, ""), ("GLY", 6, "")]
        path = _write_pdb(tmp_path / "x.pdb", residues, seqres=["GLY"] * 5)
        assert _records(path)[2] == [None]

    def test_insertion_codes_take_a_place_each_in_the_sequence(self, tmp_path):
        residues = [("SER", 81, ""), ("ALA", 82, ""), ("GLY", 82, "A"), ("THR", 82, "B")]
        residues += [("LEU", 83, "")]
        sequence = ["MET", "SER", "ALA", "GLY", "THR", "LEU", "VAL"]
        path = _write_pdb(tmp_path / "x.pdb", residues, seqres=sequence)
        (record,) = _records(path)[2]
        assert list(record.positions) == [2, 3, 4, 5, 6]

    def test_a_record_does_not_survive_a_change_of_the_topology(self):
        from binding_metrics.io.structures import load_structure, sequence_records

        topology, positions = load_structure(P53_MDM2)
        modeller = Modeller(topology, positions)
        modeller.delete([r for r in modeller.topology.residues() if r.chain.id == "B"])
        assert sequence_records(modeller.topology) == [None]


class TestSequenceRecordsOfAnMmcif:
    def test_entity_poly_seq_gives_the_position_of_each_residue(self):
        topology, _, records = _records(SFTI_TRYPSIN)
        by_chain = {c.id: r for c, r in zip(topology.chains(), records)}
        trypsin, peptide = by_chain["A"], by_chain["B"]
        assert (trypsin.source, len(trypsin.residues)) == ("_entity_poly_seq", 241)
        assert [p for p in trypsin.positions if p is not None] == list(range(1, 242))
        assert (peptide.source, len(peptide.residues)) == ("_entity_poly_seq", 14)
        assert [p for p in peptide.positions if p is not None] == list(range(1, 15))
        assert by_chain["C"] is None  # the glutathione
        assert by_chain["D"] is None and by_chain["E"] is None  # waters

    def test_a_cif_without_sequence_records_has_none(self, tmp_path):
        import gemmi

        document = gemmi.cif.read(str(LACTAM))
        document[0].find_loop_item("_entity_poly_seq.num").erase()
        path = tmp_path / "no_records.cif"
        document.write_file(str(path))
        topology, _, records = _records(path)
        assert records == [None] * topology.getNumChains()


class TestUnreadableRecords:
    def test_loading_goes_on_without_records_when_they_cannot_be_read(self, tmp_path, caplog):
        import gemmi

        document = gemmi.cif.read(str(LACTAM))
        document[0].find_loop("_entity_poly_seq.num")[0] = "one"
        path = tmp_path / "bad_records.cif"
        document.write_file(str(path))

        with caplog.at_level(logging.WARNING, logger="binding_metrics.io.structures"):
            topology, positions, records = _records(path)

        assert records == [None]
        assert topology.getNumAtoms() == len(positions)
        assert any("cannot read the sequence records" in r.getMessage() for r in caplog.records)
        assert _missing(topology, positions) == []  # numbering and geometry still work


def _truncated_cif(source: Path, destination: Path, label_asym_id: str, drop_seq_ids) -> Path:
    """Copy of an mmCIF without the atoms of some polymer positions of one chain."""
    import gemmi

    document = gemmi.cif.read(str(source))
    table = document[0].find("_atom_site.", ["label_asym_id", "label_seq_id"])
    rows = [
        k
        for k, row in enumerate(table)
        if row.str(0) == label_asym_id and row.str(1) in {str(n) for n in drop_seq_ids}
    ]
    for k in reversed(rows):
        table.remove_row(k)
    document.write_file(str(destination))
    return destination


class TestMissingFromSequenceRecords:
    def test_the_bundled_examples_are_complete_except_1ycr(self):
        from binding_metrics.io.structures import load_structure

        for path in (SFTI_TRYPSIN, CYCLOSPORIN, LACTAM, PHOSPHO, STAPLED):
            topology, positions = load_structure(path)
            assert _missing(topology, positions) == [], path.name

    def test_1ycr_misses_two_tails_of_the_receptor_and_one_of_the_peptide(self):
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(P53_MDM2)
        entries = _missing(topology, positions)
        assert [
            (e["chain"], e["kind"], e["terminus"], e["n_residues"], e["sequence_positions"])
            for e in entries
        ] == [
            ("A", "terminal_tail", "N", 8, [1, 8]),
            ("A", "terminal_tail", "C", 16, [94, 109]),
            ("B", "terminal_tail", "N", 2, [1, 2]),
        ]
        assert {e["detected_by"] for e in entries} == {"SEQRES"}
        first, last, peptide = entries
        assert (first["residue_before"], first["residue_after"]) == (None, "25")
        assert (last["residue_before"], last["residue_after"]) == ("109", None)
        assert (peptide["residue_before"], peptide["residue_after"]) == (None, "17")

    def test_a_chain_break_without_a_missing_position_is_not_a_missing_residue(self):
        """The records show that no residue is absent; the break stays in the break list."""
        from binding_metrics.core.system import find_chain_breaks
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(P53_MDM2)
        moved = []
        shifting = False
        positions_nm = positions.value_in_unit(unit.nanometer)
        for residue in topology.residues():
            if residue.chain.id == "B" and residue.id == "23":
                shifting = True
            for atom in residue.atoms():
                x, y, z = positions_nm[atom.index]
                moved.append(Vec3(x + (0.6 if shifting and residue.chain.id == "B" else 0.0), y, z))
        moved_positions = unit.Quantity(moved, unit.nanometer)

        assert len(find_chain_breaks(topology, moved_positions)) == 1
        entries = _missing(topology, moved_positions)
        assert [(e["chain"], e["kind"], e["terminus"]) for e in entries] == [
            ("A", "terminal_tail", "N"),
            ("A", "terminal_tail", "C"),
            ("B", "terminal_tail", "N"),
        ]

    def test_a_pdb_copy_without_three_residues_has_a_gap_from_seqres(self, tmp_path):
        text = P53_MDM2.read_text(encoding="utf-8").splitlines()
        kept = [
            line
            for line in text
            if not (
                line.startswith(("ATOM", "HETATM"))
                and line[21] == "B"
                and line[22:26].strip() in {"20", "21", "22"}
            )
        ]
        path = tmp_path / "1ycr_gap.pdb"
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        topology, positions = _records(path)[:2]
        entries = [e for e in _missing(topology, positions) if e["chain"] == "B"]
        assert [(e["kind"], e["n_residues"]) for e in entries] == [
            ("terminal_tail", 2),
            ("gap", 3),
        ]
        gap = entries[1]
        assert (gap["residue_before"], gap["residue_after"]) == ("19", "23")
        assert gap["sequence_positions"] == [6, 8]
        assert gap["detected_by"] == "SEQRES"

    def test_a_truncated_mmcif_lists_both_tails_and_the_gap(self, tmp_path):
        path = _truncated_cif(CYCLOSPORIN, tmp_path / "1cwa_cut.cif", "A", [1, 2, 3, 50, 164, 165])
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(path)
        entries = _missing(topology, positions)
        assert [
            (e["chain"], e["kind"], e["terminus"], e["n_residues"], e["sequence_positions"])
            for e in entries
        ] == [
            ("A", "terminal_tail", "N", 3, [1, 3]),
            ("A", "gap", None, 1, [50, 50]),
            ("A", "terminal_tail", "C", 2, [164, 165]),
        ]
        assert {e["detected_by"] for e in entries} == {"_entity_poly_seq"}
        n_tail, gap, c_tail = entries
        assert (n_tail["residue_before"], n_tail["residue_after"]) == (None, "4")
        assert (gap["residue_before"], gap["residue_after"]) == ("49", "51")
        assert (c_tail["residue_before"], c_tail["residue_after"]) == ("163", None)

    def test_the_author_chain_id_names_the_chain(self, tmp_path):
        """The cyclosporin is chain B in the OpenMM topology and chain C for the user."""
        path = _truncated_cif(CYCLOSPORIN, tmp_path / "1cwa_cut.cif", "B", [10, 11])
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(path)
        (entry,) = _missing(topology, positions)
        assert entry["chain"] == "C"
        assert (entry["kind"], entry["terminus"], entry["n_residues"]) == ("terminal_tail", "C", 2)


@pytest.mark.slow
class TestPrepStructure:
    def test_prep_reports_the_tails_warns_and_adds_no_residue(self, caplog, tmp_path):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure, save_structure

        topology, positions = load_structure(P53_MDM2)
        n_polymer_residues = sum(1 for r in topology.residues() if r.name != "HOH")
        report: dict = {}
        with caplog.at_level(logging.WARNING, logger="binding_metrics.core.system"):
            prepped, prepped_positions = prep_structure(topology, positions, report=report)

        assert [
            (e["chain"], e["terminus"], e["n_residues"]) for e in report["missing_residues"]
        ] == [("A", "N", 8), ("A", "C", 16), ("B", "N", 2)]
        assert sum(1 for r in prepped.residues() if r.name != "HOH") == n_polymer_residues
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("chain A" in m and "8" in m and "16" in m and "SEQRES" in m for m in warned)
        assert any("chain B" in m and "2" in m for m in warned)
        json.dumps(report)  # the report goes into a JSON result file
        output = tmp_path / "prepped.cif"
        save_structure(prepped, prepped_positions, output, source_path=P53_MDM2)
        reloaded, _ = load_structure(output)
        assert sum(1 for r in reloaded.residues() if r.name != "HOH") == n_polymer_residues
        assert report["n_missing_residue_gaps"] == 0  # tails are not gaps

    def test_a_complete_input_reports_an_empty_list(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(PHOSPHO)
        report: dict = {}
        prep_structure(topology, positions, report=report)
        assert report["missing_residues"] == []

    def test_a_deleted_stretch_is_one_gap_and_one_chain_break(self, caplog):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure

        pdb_topology, pdb_positions = _records(P53_MDM2)[:2]
        modeller = Modeller(pdb_topology, pdb_positions)
        modeller.delete(
            [
                r
                for r in modeller.topology.residues()
                if r.chain.id == "B" and r.id in {"22", "23", "24"}
            ]
        )
        report: dict = {}
        with caplog.at_level(logging.WARNING, logger="binding_metrics.core.system"):
            prep_structure(modeller.topology, modeller.positions, report=report)
        (entry,) = report["missing_residues"]
        assert (entry["chain"], entry["kind"], entry["n_residues"]) == ("B", "gap", 3)
        assert (entry["residue_before"], entry["residue_after"]) == ("21", "25")
        assert entry["detected_by"] == "numbering"
        assert len(report["chain_breaks"]) == 1
        assert report["n_missing_residue_gaps"] == 1
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Missing residues in chain B" in m and "21 and 25" in m for m in warned)
