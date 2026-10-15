"""Ring-closure bonds: the mapping Protenix and Boltz-2 share, and the Boltz-2 ``constraints``.

``predictors/_closures.py`` places the ends of a closure in the sequence that a runner sends. The
Protenix cases pin the output of ``ProtenixRunner`` on the bundled structures that have ring
closures: the bicyclic SFTI-1 of 3P8F (chain I, a head-to-tail amide and a disulfide), the
cyclosporin of 1CWA (chain C, a head-to-tail amide and nine modified residues) and the
somatostatin analogue of 1XY4 (chain A, a disulfide and a lactam). The values were recorded
before the mapping was moved out of ``protenix_runner.py`` and must not change with it.

The Boltz-2 cases build the YAML on the same structures and need no Boltz-2: that its own parser
accepts the block was checked outside the test suite, on the installed package.
"""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

from binding_metrics.predictors import boltz2_runner, protenix_runner  # noqa: E402
from binding_metrics.predictors.boltz2_runner import Boltz2Runner  # noqa: E402
from binding_metrics.predictors.protenix_runner import ProtenixRunner  # noqa: E402
from binding_metrics.predictors.store import PredictionRequest  # noqa: E402

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


# ---------------------------------------------------------------------------- Boltz-2 bonds

P53 = DATA / "example_linear_p53_1YCR.pdb"
STAPLED = DATA / "example_staple_3V3B.pdb"


@pytest.fixture(autouse=True)
def boltz_version(monkeypatch):
    """Boltz-2 is not installed here: the version is 2.2.1 whatever the machine has."""
    monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda conda_env: "2.2.1")


def make_boltz2(path, binder, receptor="A", **keywords):
    keywords.setdefault("name", "job")
    keywords.setdefault("mode", "predict")
    keywords.setdefault("use_msa_server", False)
    return Boltz2Runner().make_request(
        path, binder_chain=binder, receptor_chain=receptor, **keywords
    )


def written_yaml(tmp_path, request):
    """``(text, parsed)`` of the YAML that ``prepare`` writes for ``request``."""
    yaml = pytest.importorskip("yaml")
    text = Boltz2Runner().prepare(request, tmp_path / "work").read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def bond(chain, position1, atom1, position2, atom2):
    return {"bond": {"atom1": [chain, position1, atom1], "atom2": [chain, position2, atom2]}}


def with_receptor(path, tmp_path):
    """``path`` plus a chain R (the MDM2 chain of 1YCR): the runner needs a receptor chain.

    The somatostatin file holds the peptide alone. The receptor is only a sequence in
    ``predict`` mode, so its coordinates do not matter here.
    """
    import gemmi

    structure = gemmi.read_structure(str(path))
    receptor = gemmi.Chain("R")
    for residue in gemmi.read_structure(str(P53))[0]["A"]:
        receptor.add_residue(residue)
    structure[0].add_chain(receptor)
    structure.setup_entities()
    out = tmp_path / f"{path.stem}_with_receptor.cif"
    structure.make_mmcif_document().write_file(str(out))
    return out


def stapled_with_receptor(tmp_path):
    """The stapled peptide (PDB text, so its CONECT records stay) plus the MDM2 atoms as chain R."""
    receptor_atoms = []
    for line in P53.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM") and line[21] == "A":
            serial = 5000 + len(receptor_atoms)
            receptor_atoms.append(f"{line[:6]}{serial:5d}{line[11:21]}R{line[22:]}")
    lines = STAPLED.read_text(encoding="utf-8").splitlines()
    first_link = next(i for i, line in enumerate(lines) if line.startswith("CONECT"))
    out = tmp_path / "stapled_with_receptor.pdb"
    text = "\n".join(lines[:first_link] + receptor_atoms + lines[first_link:]) + "\n"
    out.write_text(text, encoding="utf-8")
    return out


class TestBoltz2WritesTheClosureBonds:
    def test_a_bicyclic_binder_gets_the_ring_and_the_disulfide(self, tmp_path):
        request = make_boltz2(SFTI1, "I", closure_bonds="auto")
        text, document = written_yaml(tmp_path, request)
        assert (
            'constraints:\n  - bond:\n      atom1: ["I", 14, "C"]\n      atom2: ["I", 1, "N"]\n'
            '  - bond:\n      atom1: ["I", 3, "SG"]\n      atom2: ["I", 11, "SG"]\n'
        ) in text
        assert document["constraints"] == [bond("I", 14, "C", 1, "N"), bond("I", 3, "SG", 11, "SG")]
        binder = document["sequences"][1]["protein"]
        assert binder["id"] == "I" and binder["cyclic"] is True  # the positional wrap stays
        assert "constraints" not in {k for k in document["sequences"][0]}

    def test_a_head_to_tail_ring_with_modified_residues_is_placed_by_sequence_position(
        self, tmp_path
    ):
        request = make_boltz2(CYCLOSPORIN, "C", closure_bonds="auto")
        _, document = written_yaml(tmp_path, request)
        assert document["constraints"] == [bond("C", 11, "C", 1, "N")]
        binder = document["sequences"][1]["protein"]
        assert binder["sequence"] == "ALLVTAGLVLA"
        assert len(binder["modifications"]) == 9 and binder["modifications"][0]["ccd"] == "DAL"

    def test_a_lactam_and_a_disulfide_next_to_modified_residues(self, tmp_path):
        path = with_receptor(SOMATOSTATIN, tmp_path)
        request = make_boltz2(path, "A", receptor="R", closure_bonds="auto")
        _, document = written_yaml(tmp_path, request)
        assert document["constraints"] == [
            bond("A", 2, "SG", 12, "SG"),
            bond("A", 10, "NZ", 4, "CD"),
        ]
        binder = next(e["protein"] for e in document["sequences"] if e["protein"]["id"] == "A")
        # D-Trp and the C-terminal cap sit before the lysine: the positions are those of the
        # sequence sent, and each points at the residue the atom belongs to
        assert [m["position"] for m in binder["modifications"]] == [6, 7]
        sequence = binder["sequence"]
        assert (sequence[1], sequence[11], sequence[9], sequence[3]) == ("C", "C", "K", "E")

    def test_waters_after_the_chain_do_not_shift_the_last_position(self, tmp_path):
        request = make_boltz2(SFTI1, "I", closure_bonds="auto")
        _, document = written_yaml(tmp_path, request)
        assert document["constraints"][0] == bond("I", 14, "C", 1, "N")  # 6 waters follow ASP 14

    def test_a_staple_is_not_written_and_a_warning_names_it(self, tmp_path, caplog):
        path = stapled_with_receptor(tmp_path)
        with caplog.at_level("WARNING", logger=boltz2_runner.logger.name):
            request = make_boltz2(path, "C", receptor="R", closure_bonds="auto")
        assert "hydrocarbon staple" in caplog.text and "not written as a bond" in caplog.text
        assert "closure_bonds" not in request.options
        _, document = written_yaml(tmp_path, request)
        assert "constraints" not in document

    def test_a_linear_binder_has_no_bond_and_keeps_its_key(self, tmp_path):
        request = make_boltz2(P53, "B", closure_bonds="auto")
        assert request.key() == make_boltz2(P53, "B").key()
        _, document = written_yaml(tmp_path, request)
        assert "constraints" not in document

    def test_the_bond_is_written_whatever_the_cyclic_flag_says(self, tmp_path):
        request = make_boltz2(CYCLOSPORIN, "C", closure_bonds="auto", binder_cyclic=False)
        text, document = written_yaml(tmp_path, request)
        assert "cyclic" not in text
        assert document["constraints"] == [bond("C", 11, "C", 1, "N")]

    def test_every_mode_writes_the_block_before_the_templates(self, tmp_path):
        request = make_boltz2(SFTI1, "I", mode="score-lock", closure_bonds="auto")
        text, document = written_yaml(tmp_path, request)
        assert list(document) == ["version", "sequences", "constraints", "templates"]
        assert text.index("constraints:") < text.index("templates:")


class TestTheDefaultKeepsTheYamlAndTheKey:
    #: Keys of ``make_boltz2(path, binder, mode=...)`` as the runner computed them before
    #: ``closure_bonds`` existed (version 2.2.1, defaults otherwise).
    OLD_KEYS = {
        (P53, "B", "score"): "692fe603a5b9eed871027a1333345b5d9981b769475112c2ac1a60db64129028",
        (SFTI1, "I", "score"): "02eef864983fc7819f6311f214f4a407d7fae8265b07ee245d2cbc9d0dea94da",
        (CYCLOSPORIN, "C", "score"): (
            "4076fbf7e03576d6c89eb8d29239b766d2cfee9206a2be9e768eb59b54d60e35"
        ),
        (P53, "B", "predict"): "ff15638f706ab934ac755157bfec5c7c827bd3a7930db0bb1d83e78204060b32",
        (SFTI1, "I", "predict"): "dc0a9bca6b3efcd138b991f26bc4812a07d2d3d8615d38b928d876603e3d32da",
        (CYCLOSPORIN, "C", "predict"): (
            "82179c2be5b973643ae31df8c2f9b1868ac80c829529c830314ae00fbd02647d"
        ),
    }
    #: SHA-256 of the ``predict`` YAML of the same runner, by (path, chain, ``use_msa_server``).
    OLD_YAML_SHA256 = {
        (P53, "B", True): "c0782b13c28aed9f5252c44e8223cf450e6bbd0a847204fbfdd1003cc50e2368",
        (P53, "B", False): "2b765c8c3bc91c886172b0aa9d39c55f2f9de10bf1d397c1e464bb947d492b07",
        (SFTI1, "I", True): "9980ab85fc9b7050c28cc9abd049f8ec5338c3c6cb48acdb99cc340f22700dd3",
        (SFTI1, "I", False): "d045f835210e56630eaa8185b752312aa53a535b3f406f944e6e97a605a0a6f1",
        (CYCLOSPORIN, "C", True): (
            "399e131069f21bc884d2a0cf23b52da23afb8950ba251e476ac49e72ad6b7429"
        ),
        (CYCLOSPORIN, "C", False): (
            "5b14796239bde52d5026b60a77f9504216591339be1812d5a182f3b146edbb19"
        ),
    }

    @pytest.mark.parametrize("closure_bonds", [None, "none", []])
    @pytest.mark.parametrize("path, binder, mode", list(OLD_KEYS))
    def test_an_old_request_keeps_its_key(self, path, binder, mode, closure_bonds):
        keywords = {} if closure_bonds is None else {"closure_bonds": closure_bonds}
        request = make_boltz2(path, binder, mode=mode, use_msa_server=True, **keywords)
        assert request.key() == self.OLD_KEYS[path, binder, mode]
        assert "closure_bonds" not in request.options

    @pytest.mark.parametrize("closure_bonds", [None, "none"])
    @pytest.mark.parametrize("path, binder, use_msa_server", list(OLD_YAML_SHA256))
    def test_the_yaml_is_what_it_was(self, tmp_path, path, binder, use_msa_server, closure_bonds):
        keywords = {} if closure_bonds is None else {"closure_bonds": closure_bonds}
        request = make_boltz2(path, binder, use_msa_server=use_msa_server, **keywords)
        yaml_path = Boltz2Runner().prepare(request, tmp_path / "work")
        digest = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
        assert digest == self.OLD_YAML_SHA256[path, binder, use_msa_server]

    def test_a_request_made_without_the_option_means_none(self, tmp_path):
        request = PredictionRequest(
            "boltz2",
            "job",
            mode="predict",
            input_path=SFTI1,
            binder_chain="I",
            receptor_chain="A",
            options={"use_msa_server": False},
        )
        yaml_path = Boltz2Runner().prepare(request, tmp_path)
        assert "constraints" not in yaml_path.read_text(encoding="utf-8")


class TestTheBondsAreInTheKey:
    def test_auto_and_the_same_bonds_given_as_a_list_share_a_key(self):
        auto = make_boltz2(SFTI1, "I", closure_bonds="auto")
        listed = make_boltz2(
            SFTI1,
            "I",
            closure_bonds=[
                {"atom1": ("I", 14, "C"), "atom2": ("I", 1, "N")},
                {"atom1": ("I", 3, "SG"), "atom2": ("I", 11, "SG")},
            ],
        )
        assert auto.options["closure_bonds"] == listed.options["closure_bonds"]
        assert auto.key() == listed.key()
        assert auto.key() != make_boltz2(SFTI1, "I").key()

    def test_two_different_lists_give_two_keys(self):
        one = make_boltz2(
            SFTI1, "I", closure_bonds=[{"atom1": ["I", 3, "SG"], "atom2": ["I", 11, "SG"]}]
        )
        other = make_boltz2(
            SFTI1, "I", closure_bonds=[{"atom1": ["I", 3, "SG"], "atom2": ["I", 12, "SG"]}]
        )
        assert len({one.key(), other.key(), make_boltz2(SFTI1, "I").key()}) == 3

    def test_the_options_are_json_lists_and_the_resolved_bonds_are_stored(self):
        request = make_boltz2(CYCLOSPORIN, "C", closure_bonds="auto")
        assert request.options["closure_bonds"] == [
            {"atom1": ["C", 11, "C"], "atom2": ["C", 1, "N"]}
        ]


class TestAListOfBonds:
    def test_a_list_is_written_as_given_and_may_join_the_two_chains(self, tmp_path):
        bonds = [
            {"atom1": ["B", 1, "N"], "atom2": ["B", 13, "C"]},
            {"atom1": ["A", 5, "SG"], "atom2": ["B", 7, "SG"]},
        ]
        _, document = written_yaml(tmp_path, make_boltz2(P53, "B", closure_bonds=bonds))
        assert document["constraints"] == [{"bond": bond_} for bond_ in bonds]

    @pytest.mark.parametrize(
        "closure_bonds, error, message",
        [
            ("everything", ValueError, "must be 'none', 'auto' or a list of bonds"),
            (3, TypeError, "got int"),
            ({"atom1": ["B", 1, "N"]}, TypeError, "got dict"),
            ([["B", 1, "N"]], ValueError, "item 1 is"),
            ([{"atom1": ["B", 1, "N"]}], ValueError, "item 1 is"),
            ([{"atom1": ["B", 1, "N"], "atom2": ["B", 2, "C"], "type": "x"}], ValueError, "item 1"),
            ([{"atom1": "B1N", "atom2": ["B", 2, "C"]}], ValueError, "atom1 is 'B1N'"),
            ([{"atom1": ["B", 1], "atom2": ["B", 2, "C"]}], ValueError, "atom1 is \\['B', 1\\]"),
            ([{"atom1": ["B", 0, "N"], "atom2": ["B", 2, "C"]}], ValueError, "position 0"),
            ([{"atom1": ["B", "1", "N"], "atom2": ["B", 2, "C"]}], ValueError, "position '1'"),
            ([{"atom1": ["B", True, "N"], "atom2": ["B", 2, "C"]}], ValueError, "position True"),
            ([{"atom1": ["", 1, "N"], "atom2": ["B", 2, "C"]}], ValueError, "chain ''"),
            ([{"atom1": ["B", 1, ""], "atom2": ["B", 2, "C"]}], ValueError, "atom ''"),
            ([{"atom1": ["B", 1, "N"], "atom2": ["B", 1, "N"]}], ValueError, "to itself"),
        ],
    )
    def test_a_badly_shaped_request_is_refused_when_it_is_made(self, closure_bonds, error, message):
        with pytest.raises(error, match=message):
            make_boltz2(P53, "B", closure_bonds=closure_bonds)

    @pytest.mark.parametrize(
        "bonds, message",
        [
            ([{"atom1": ["Z", 1, "N"], "atom2": ["B", 2, "C"]}], "chain 'Z' is not the receptor"),
            ([{"atom1": ["B", 1, "N"], "atom2": ["B", 14, "C"]}], "position 14 is past the end"),
        ],
    )
    def test_a_chain_or_position_the_input_lacks_is_refused_before_anything_is_written(
        self, tmp_path, bonds, message
    ):
        request = make_boltz2(P53, "B", closure_bonds=bonds)
        with pytest.raises(ValueError, match=message):
            Boltz2Runner().prepare(request, tmp_path / "work")
        assert not (tmp_path / "work").exists()

    def test_a_request_made_elsewhere_with_a_bad_option_is_refused_by_prepare(self, tmp_path):
        request = PredictionRequest(
            "boltz2",
            "job",
            mode="predict",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            options={"closure_bonds": "some"},
        )
        with pytest.raises(ValueError, match="must be 'none', 'auto' or a list"):
            Boltz2Runner().prepare(request, tmp_path / "work")
        assert not (tmp_path / "work").exists()

    def test_a_request_made_elsewhere_with_auto_is_resolved_by_prepare(self, tmp_path):
        request = PredictionRequest(
            "boltz2",
            "job",
            mode="predict",
            input_path=CYCLOSPORIN,
            binder_chain="C",
            receptor_chain="A",
            options={"closure_bonds": "auto", "use_msa_server": False},
        )
        _, document = written_yaml(tmp_path, request)
        assert document["constraints"] == [bond("C", 11, "C", 1, "N")]


class TestAClosureThatCannotBePlaced:
    def unplaceable(self, monkeypatch):
        import binding_metrics.capabilities as capabilities

        monkeypatch.setattr(
            capabilities,
            "detect_closures",
            lambda atoms, chain: [
                closure("disulfide", (1, "CYS", 41, "SG"), (3, "CYS", 43, "SG")),
            ],
        )

    def test_the_request_raises_and_names_the_way_out(self, monkeypatch):
        self.unplaceable(monkeypatch)
        with pytest.raises(ValueError, match="residue CYS 41 is not in the 14 residues") as info:
            make_boltz2(SFTI1, "I", closure_bonds="auto")
        assert "closure_bonds='none'" in str(info.value)

    def test_prepare_raises_before_any_file_is_written(self, tmp_path, monkeypatch):
        request = PredictionRequest(
            "boltz2",
            "job",
            mode="score",
            input_path=SFTI1,
            binder_chain="I",
            receptor_chain="A",
            options={"closure_bonds": "auto", "use_msa_server": False},
        )
        self.unplaceable(monkeypatch)
        with pytest.raises(ValueError, match="cannot place the disulfide"):
            Boltz2Runner().prepare(request, tmp_path / "work")
        assert not (tmp_path / "work").exists()

    def test_a_search_that_fails_is_an_error_and_not_a_prediction_without_the_ring(
        self, monkeypatch
    ):
        import binding_metrics.capabilities as capabilities

        def broken(atoms, chain_id):
            raise RuntimeError("cannot read the bond table")

        monkeypatch.setattr(capabilities, "detect_closures", broken)
        with pytest.raises(ValueError, match="could not look for ring closures in chain I") as info:
            make_boltz2(SFTI1, "I", closure_bonds="auto")
        assert "closure_bonds='none'" in str(info.value)

    def test_nothing_is_searched_when_the_option_is_none(self, tmp_path, monkeypatch):
        import binding_metrics.capabilities as capabilities

        def fail(*args, **kwargs):
            raise AssertionError("the closures were searched")

        monkeypatch.setattr(capabilities, "detect_closures", fail)
        request = make_boltz2(SFTI1, "I")
        written_yaml(tmp_path, request)
