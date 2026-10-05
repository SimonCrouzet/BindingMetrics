"""Cyclization is found for the chain a user names by its author ID (#64).

For 1CWA the peptide is author chain C and chain B of the OpenMM topology, whose chain C
holds the 140 waters of author chain A. For 3P8F the peptide is author chain I and chain B
of the topology, which has no chain I.
"""

import logging
import warnings
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

from openmm import app  # noqa: E402

from binding_metrics.core.cyclic import detect_cyclization, patch_cyclic_topology  # noqa: E402
from binding_metrics.io.structures import (  # noqa: E402
    attach_author_chain_ids,
    author_chain_ids,
    load_structure,
    topology_chain_id,
)

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"


def _load(path):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return load_structure(path)


def _detect(topology, positions, chain_id):
    """The cyclic bonds, with the warnings of the detection turned into errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return detect_cyclization(topology, positions, chain_id)


def _swap_author_ids(tmp_path, source, mapping):
    """A copy of ``source`` whose author chain IDs are renamed by ``mapping``."""
    import gemmi

    document = gemmi.cif.read(str(source))
    block = document.sole_block()
    for row in block.find("_atom_site.", ["auth_asym_id"]):
        row[0] = mapping.get(row.str(0), row.str(0))
    path = tmp_path / source.name
    document.write_file(str(path))
    return path


def _links(found):
    return sorted((b.cyclic_type, b.atom1_id, b.atom2_id) for b in found)


class TestAuthorChainId:
    def test_1cwa_peptide_by_author_id(self):
        topology, positions = _load(CWA)
        found = _detect(topology, positions, "C")
        assert [b.cyclic_type for b in found] == ["head_to_tail"]
        # the entries name the chain as the topology does, which later steps look up
        assert found[0].atom1_id == ("B", 10, "C")
        assert found[0].atom2_id == ("B", 0, "N")

    def test_1cwa_topology_id_still_works(self):
        topology, positions = _load(CWA)
        assert _links(_detect(topology, positions, "B")) == _links(
            _detect(topology, positions, "C")
        )

    def test_3p8f_peptide_by_author_id(self):
        topology, positions = _load(SFTI)
        found = _detect(topology, positions, "I")
        assert [b.cyclic_type for b in found] == ["head_to_tail", "disulfide"]
        assert _links(found) == _links(_detect(topology, positions, "B"))

    def test_the_receptor_is_found_by_its_author_id(self):
        """Author A is chain A of the topology, not the GSH or the water chains that share it."""
        topology, positions = _load(SFTI)
        found = _detect(topology, positions, "A")
        assert [b.cyclic_type for b in found] == ["disulfide"] * 3
        assert {b.atom1_id[0] for b in found} == {"A"}

    def test_a_linear_peptide_in_a_pdb_file_is_unchanged(self):
        topology, positions = _load(P53)
        assert _detect(topology, positions, "B") == []

    def test_an_unknown_chain_still_raises(self):
        topology, positions = _load(CWA)
        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            detect_cyclization(topology, positions, "Z")

    def test_the_warnings_name_the_author_chain(self):
        topology, positions = _load(CWA)
        peptide = next(c for c in topology.chains() if c.id == "B")
        n_atom = next(a for a in next(iter(peptide.residues())).atoms() if a.name == "N")
        modeller = app.Modeller(topology, positions)
        modeller.delete([n_atom])
        assert attach_author_chain_ids(modeller.topology, CWA)
        with pytest.warns(UserWarning, match=r"atom N not found.*\(chain C\)"):
            detect_cyclization(modeller.topology, modeller.positions, "C")


class TestPatch:
    def test_patch_cyclic_topology_takes_the_topology_id(self):
        topology, positions = _load(CWA)
        patched, _, info = patch_cyclic_topology(topology, positions, "B")
        assert [b.cyclic_type for b in info] == ["head_to_tail"]
        assert info[0].atom1_id[0] == "B"
        # the patched topology is a new one, still with OpenMM's chain IDs
        assert [chain.id for chain in patched.chains()][:2] == ["A", "B"]

    def test_the_id_of_the_topology_is_never_read_as_an_author_id(self, tmp_path):
        """Label A is author B and label B is author A: topology ID B is the author A chain.

        The relaxation and the energy hand this function the IDs of the topology, of a
        topology that still carries its author IDs.
        """
        _load(SFTI)
        swapped = _swap_author_ids(tmp_path, SFTI, {"A": "B", "I": "A"})
        topology, positions = load_structure(swapped)
        assert author_chain_ids(topology)[:2] == ["B", "A"]
        # chain B of the topology is the peptide (author A), chain A the trypsin (author B)
        _, _, info = patch_cyclic_topology(topology, positions, "B")
        assert [b.cyclic_type for b in info] == ["head_to_tail", "disulfide"]
        _, _, receptor_info = patch_cyclic_topology(topology, positions, "A")
        assert {b.cyclic_type for b in receptor_info} == {"disulfide"}
        # the entry point that takes the caller's ID reads author A as the peptide
        assert [b.cyclic_type for b in detect_cyclization(topology, positions, "A")] == [
            "head_to_tail",
            "disulfide",
        ]


class TestResolution:
    def test_an_ambiguous_author_id_raises(self, tmp_path):
        from tests.test_fix_64_chain_ids import _write_cif

        path = _write_cif(
            tmp_path / "ambiguous.cif",
            [("GLY", "A", "A")] * 3 + [("GLY", "B", "A")] * 3,
        )
        topology, positions = load_structure(path)
        with pytest.raises(ValueError, match="author chain 'A'"):
            detect_cyclization(topology, positions, "A")

    def test_letters_swapped_between_label_and_author_ids(self, tmp_path):
        """Author A is chain B of the topology and author B is chain A."""
        from tests.test_fix_64_chain_ids import _write_cif

        path = _write_cif(
            tmp_path / "swapped.cif",
            [("GLY", "A", "B")] * 3 + [("GLY", "B", "A")] * 3 + [("HOH", "C", "A")],
        )
        topology, _ = load_structure(path)
        assert topology_chain_id(topology, "A") == "B"
        assert topology_chain_id(topology, "B") == "A"
        assert topology_chain_id(topology, "C") == "C"  # waters only: as named


def test_no_warning_is_logged_for_the_examples(caplog):
    topology, positions = _load(CWA)
    with caplog.at_level(logging.WARNING):
        detect_cyclization(topology, positions, "C")
    assert not caplog.records
