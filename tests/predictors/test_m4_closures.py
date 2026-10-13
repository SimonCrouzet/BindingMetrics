"""The closure-to-position mapping that Protenix and Boltz-2 share (``predictors/_closures.py``).

The Protenix cases pin the output of ``ProtenixRunner`` on the three bundled structures that have
ring closures: the bicyclic SFTI-1 of 3P8F (chain I, a head-to-tail amide and a disulfide), the
cyclosporin of 1CWA (chain C, a head-to-tail amide and nine modified residues) and the
somatostatin analogue of 1XY4 (chain A, a disulfide and a lactam). The values were recorded
before the mapping was moved out of ``protenix_runner.py`` and must not change with it.
"""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

from binding_metrics.predictors import protenix_runner  # noqa: E402
from binding_metrics.predictors.protenix_runner import ProtenixRunner  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"


def protenix_bond(position1, atom1, position2, atom2):
    return {
        "entity1": 2, "copy1": 1, "position1": position1, "atom1": atom1,
        "entity2": 2, "copy2": 1, "position2": position2, "atom2": atom2,
    }  # fmt: skip


HEAD_TO_TAIL_14 = protenix_bond(14, "C", 1, "N")
HEAD_TO_TAIL_11 = protenix_bond(11, "C", 1, "N")
HEAD_TO_TAIL_12 = protenix_bond(12, "C", 1, "N")
SFTI1_DISULFIDE = protenix_bond(3, "SG", 11, "SG")
SOMATOSTATIN_DISULFIDE = protenix_bond(2, "SG", 12, "SG")

#: SHA-256 of the input JSON that ``ProtenixRunner.prepare`` wrote before the mapping was shared,
#: by (job name, ``binder_cyclic``); the receptor is chain A and the job is named as the key says.
INPUT_JSON_SHA256 = {
    ("sfti", "auto"): "cf2f1abd8dbd7b325a35a36f1ec06beed1069c67611e9b4e9692738b31a26314",
    ("sfti", True): "cf2f1abd8dbd7b325a35a36f1ec06beed1069c67611e9b4e9692738b31a26314",
    ("sfti", False): "eb2d9b201797326667db6c4c4f577f2d03c00cff33f3dbab86fe9ad9dc6bc559",
    ("cwa", "auto"): "9e4cd7edaa9686d6ec415b91df4baa4569406d65af5ed6a25fc98874cdade5b0",
    ("cwa", True): "9e4cd7edaa9686d6ec415b91df4baa4569406d65af5ed6a25fc98874cdade5b0",
    ("cwa", False): "048a9310aeecdc4205d95188a2472f85caf3224da051c3dc04fde732413c67da",
}


class TestProtenixOutputIsUnchangedByTheSharedMapping:
    """``(path, binder, receptor)`` and the ``covalent_bonds`` Protenix wrote for each setting."""

    @pytest.mark.parametrize(
        "path, binder, expected",
        [
            (SFTI1, "I", {"auto": [HEAD_TO_TAIL_14, SFTI1_DISULFIDE], False: []}),
            (CYCLOSPORIN, "C", {"auto": [HEAD_TO_TAIL_11], False: []}),
            (SOMATOSTATIN, "A", {"auto": [SOMATOSTATIN_DISULFIDE], False: []}),
        ],
    )
    def test_closure_bonds_of_the_bundled_binders(self, path, binder, expected):
        (entity,) = protenix_runner._read_entities(path, [binder], "error")
        for setting, bonds in expected.items():
            assert protenix_runner._closure_bonds(path, binder, entity[3], setting) == bonds

    def test_true_adds_the_head_to_tail_bond_only_where_the_input_has_none(self):
        (entity,) = protenix_runner._read_entities(SOMATOSTATIN, ["A"], "error")
        bonds = protenix_runner._closure_bonds(SOMATOSTATIN, "A", entity[3], True)
        assert bonds == [HEAD_TO_TAIL_12, SOMATOSTATIN_DISULFIDE]
        (entity,) = protenix_runner._read_entities(SFTI1, ["I"], "error")
        bonds = protenix_runner._closure_bonds(SFTI1, "I", entity[3], True)
        assert bonds == [HEAD_TO_TAIL_14, SFTI1_DISULFIDE]

    @pytest.mark.parametrize(
        "name, path, binder, setting",
        [
            ("sfti", SFTI1, "I", "auto"),
            ("sfti", SFTI1, "I", True),
            ("sfti", SFTI1, "I", False),
            ("cwa", CYCLOSPORIN, "C", "auto"),
            ("cwa", CYCLOSPORIN, "C", True),
            ("cwa", CYCLOSPORIN, "C", False),
        ],
    )
    def test_the_input_json_is_byte_for_byte_what_it_was(
        self, tmp_path, name, path, binder, setting
    ):
        runner = ProtenixRunner()
        request = runner.make_request(
            path, name=name, binder_chain=binder, receptor_chain="A", binder_cyclic=setting
        )
        text = runner.prepare(request, tmp_path).read_bytes()
        assert hashlib.sha256(text).hexdigest() == INPUT_JSON_SHA256[name, setting]
        assert ("covalent_bonds" in json.loads(text)[0]) == (setting is not False)


FAMILIES = ("head_to_tail", "disulfide", "lactam")


def closure(family, end1, end2, kind=None):
    from binding_metrics.capabilities import Closure, ClosureEnd

    return Closure(kind or family, family, ClosureEnd(*end1), ClosureEnd(*end2))


def residues_of(path, chain):
    (entity,) = protenix_runner._read_entities(path, [chain], "error")
    return entity[3]


class TestPlacingClosures:
    """``place_closures``: closure ends to 1-based positions in the sequence that was sent."""

    def test_the_closures_of_the_bundled_binders_get_the_positions_of_the_sequence(self):
        from binding_metrics.predictors._closures import PlacedBond, find_closures, place_closures

        expected = {
            (SFTI1, "I"): [
                PlacedBond("head_to_tail", 14, "C", 1, "N"),
                PlacedBond("disulfide", 3, "SG", 11, "SG"),
            ],
            (CYCLOSPORIN, "C"): [PlacedBond("head_to_tail", 11, "C", 1, "N")],
            (SOMATOSTATIN, "A"): [
                PlacedBond("disulfide", 2, "SG", 12, "SG"),
                PlacedBond("lactam", 10, "NZ", 4, "CD"),
            ],
        }
        for (path, chain), bonds in expected.items():
            found = find_closures(path, chain, fallback_advice="none")
            placed, skipped = place_closures(
                found, residues_of(path, chain), chain, families=FAMILIES, way_out="out"
            )
            assert placed == bonds and skipped == []

    def test_a_family_that_is_not_asked_for_comes_back_as_text(self):
        from binding_metrics.predictors._closures import find_closures, place_closures

        found = find_closures(SOMATOSTATIN, "A", fallback_advice="none")
        placed, skipped = place_closures(
            found,
            residues_of(SOMATOSTATIN, "A"),
            "A",
            families=("head_to_tail", "disulfide"),
            way_out="out",
        )
        assert [bond.family for bond in placed] == ["disulfide"]
        assert skipped == ["lactam sc lys glu LYS 10.NZ - GLU 4.CD"]

    def test_positions_follow_the_sequence_and_not_the_numbers_of_the_file(self):
        from binding_metrics.predictors._closures import PlacedBond, place_closures

        residues = [("CYS", 41), ("ALA", 42), ("CYS", 43), ("GLY", 44)]
        found = [closure("disulfide", (2, "CYS", 43, "SG"), (0, "CYS", 41, "SG"))]
        placed, _ = place_closures(found, residues, "B", families=FAMILIES, way_out="out")
        assert placed == [PlacedBond("disulfide", 1, "SG", 3, "SG")]

    @pytest.mark.parametrize(
        "found, residues, message",
        [
            (
                closure("head_to_tail", (3, "GLY", 44, "C"), (0, "CYS", 41, "N")),
                [("CYS", 41), ("ALA", 42), ("GLY", 43)],
                "residue GLY 44 is not in the 3 residues",
            ),
            (
                closure("head_to_tail", (2, "GLY", 43, "C"), (0, "CYS", 41, "N")),
                [("CYS", 41), ("GLY", 43), ("GLY", 43)],
                "residue GLY 43 is twice in the 3 residues",
            ),
            (
                closure("head_to_tail", (1, "ALA", 42, "C"), (0, "CYS", 41, "N")),
                [("CYS", 41), ("ALA", 42), ("GLY", 43)],
                "is not between C of the last residue \\(3\\) and N of the first",
            ),
            (
                closure("disulfide", (0, "CYS", 41, "SG"), (1, "ALA", 42, "CB")),
                [("CYS", 41), ("ALA", 42), ("GLY", 43)],
                "is not between two SG atoms",
            ),
            (
                closure("lactam", (0, "CYS", 41, "N"), (1, "ASP", 99, "CG")),
                [("CYS", 41), ("ALA", 42), ("GLY", 43)],
                "residue ASP 99 is not in the 3 residues",
            ),
        ],
    )
    def test_a_closure_that_cannot_be_placed_raises_with_the_way_out(
        self, found, residues, message
    ):
        from binding_metrics.predictors._closures import place_closures

        with pytest.raises(ValueError, match=message) as info:
            place_closures([found], residues, "B", families=FAMILIES, way_out="Use the way out")
        assert str(info.value).endswith("Use the way out")

    def test_a_search_that_fails_is_an_error_with_the_advice(self, monkeypatch):
        import binding_metrics.capabilities as capabilities
        from binding_metrics.predictors._closures import find_closures

        def broken(atoms, chain_id):
            raise RuntimeError("no bond table")

        monkeypatch.setattr(capabilities, "detect_closures", broken)
        with pytest.raises(ValueError, match="could not look for ring closures in chain I") as info:
            find_closures(SFTI1, "I", fallback_advice="pass nothing")
        assert "no bond table" in str(info.value)
        assert str(info.value).endswith("is not the fallback: pass nothing")
