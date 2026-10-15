"""The non-standard residue steps find the chain a caller names by its author ID (#113).

1CWA: the peptide, with DAL, MLE, MVA and SAR, is author chain C and chain B of the OpenMM
topology, whose chain C holds 140 waters.
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

from binding_metrics.core.nonstandard import (  # noqa: E402
    _detect_nonstandard,
    detect_nonstandard,
    patch_nonstandard,
    restore_nonstandard_names,
)
from binding_metrics.io.structures import load_structure  # noqa: E402
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig  # noqa: E402
from tests.test_fix_64_cyclic import _swap_author_ids  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"

try:
    import openmmforcefields  # noqa: F401

    HAS_OMMFF = True
except ImportError:
    HAS_OMMFF = False
requires_ommff = pytest.mark.skipif(not HAS_OMMFF, reason="openmmforcefields not installed")


def _load(path):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return load_structure(path)


def _peptide_names(topology, chain_id):
    chain = next(c for c in topology.chains() if c.id == chain_id)
    return [residue.name for residue in chain.residues()]


class TestAuthorChainId:
    def test_detect_by_author_id(self):
        topology, _ = _load(CWA)
        info = detect_nonstandard(topology, "C")
        assert [e["original_name"] for e in info.d_residues] == ["DAL"]
        assert {e["original_name"] for e in info.nmethyl_residues} >= {"MLE", "MVA", "SAR"}
        assert info.chain_id == "B"  # the ID in the topology, which the later steps look up

    def test_the_id_of_the_topology_gives_the_same_entries(self):
        topology, _ = _load(CWA)
        by_author, by_topology = (
            detect_nonstandard(topology, "C"),
            detect_nonstandard(topology, "B"),
        )
        assert by_author.d_residues == by_topology.d_residues
        assert by_author.nmethyl_residues == by_topology.nmethyl_residues

    def test_patch_and_restore_by_author_id(self):
        topology, positions = _load(CWA)
        original = _peptide_names(topology, "B")
        info = detect_nonstandard(topology, "C")
        topology, positions = patch_nonstandard(topology, positions, "C", info)
        patched = _peptide_names(topology, "B")
        assert patched[0] == "ALA" and patched != original  # DAL renamed
        assert restore_nonstandard_names(topology, info, "C") == len(info.d_residues) + len(
            info.nmethyl_residues
        )
        assert _peptide_names(topology, "B") == original

    def test_restore_takes_the_id_of_the_info_as_a_topology_id(self):
        topology, positions = _load(CWA)
        original = _peptide_names(topology, "B")
        info = detect_nonstandard(topology, "C")
        topology, positions = patch_nonstandard(topology, positions, "C", info)
        assert restore_nonstandard_names(topology, info) > 0
        assert _peptide_names(topology, "B") == original

    def test_an_unknown_chain_still_raises(self):
        topology, _ = _load(CWA)
        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            detect_nonstandard(topology, "Z")


class TestSwappedLetters:
    """Author IDs A and C of 1CWA renamed to B and A: the peptide is author A, topology B."""

    @pytest.fixture
    def swapped(self, tmp_path):
        _load(CWA)
        return _swap_author_ids(tmp_path, CWA, {"A": "B", "C": "A"})

    def test_the_caller_names_the_peptide_by_its_author_id(self, swapped):
        topology, _ = load_structure(swapped)
        assert detect_nonstandard(topology, "A").chain_id == "B"
        assert [e["original_name"] for e in detect_nonstandard(topology, "A").d_residues] == ["DAL"]

    def test_the_code_that_holds_a_topology_id_is_not_redirected(self, swapped):
        """Topology chain B is the peptide; read as an author ID it would be the receptor."""
        topology, _ = load_structure(swapped)
        assert [e["original_name"] for e in _detect_nonstandard(topology, "B").d_residues] == [
            "DAL"
        ]
        assert detect_nonstandard(topology, "B").d_residues == []  # author B is the receptor


@pytest.mark.integration
@requires_ommff
def test_the_relaxation_patches_the_peptide_of_a_swapped_file(tmp_path):
    """The relaxation holds topology IDs on a topology that still carries author IDs."""
    _load(CWA)
    swapped = _swap_author_ids(tmp_path, CWA, {"A": "B", "C": "A"})
    config = RelaxationConfig(peptide_chain_id="A", receptor_chain_id="B", small_molecules="auto")
    relaxer = ImplicitRelaxation(config)
    _, topology, _, bond_info = relaxer._setup_system(swapped)
    assert relaxer._chain_ids == ("B", "A")
    assert [e["original_name"] for e in relaxer._ns_info.d_residues] == ["DAL"]
    assert [b.cyclic_type for b in bond_info] == ["head_to_tail"]
    assert relaxer._dropped_protein_chains == []
