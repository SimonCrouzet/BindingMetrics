"""A raw structure whose chain ends without OXT is refused before the force field (#114).

The final VAL of the trypsin chain of 3P8F has no OXT in the mmCIF; prep (PDBFixer) adds it.
Relaxation and energy used to stop inside OpenMM with "No template found for residue 241
(VAL) ... the bonds are different".
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

from openmm.app import Topology, element  # noqa: E402

from binding_metrics.core.cyclic import patch_cyclic_topology  # noqa: E402
from binding_metrics.core.system import (  # noqa: E402
    find_open_c_termini,
    require_closed_c_termini,
)
from binding_metrics.io.structures import load_structure  # noqa: E402
from binding_metrics.metrics import energy  # noqa: E402
from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig  # noqa: E402
from tests.test_fix_64_cyclic import _swap_author_ids  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"
STAPLE = DATA / "example_staple_3V3B.pdb"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"


def _load(path):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return load_structure(path)


def _chain(last_name, *, oxt=False, closure=False, carbonyl=True):
    """A two-residue chain; ``last_name`` is the second residue."""
    topology = Topology()
    chain = topology.addChain("A")
    first = topology.addResidue("ALA", chain, id="1")
    n1 = topology.addAtom("N", element.nitrogen, first)
    topology.addAtom("CA", element.carbon, first)
    c1 = topology.addAtom("C", element.carbon, first)
    last = topology.addResidue(last_name, chain, id="2")
    n2 = topology.addAtom("N", element.nitrogen, last)
    topology.addAtom("CA", element.carbon, last)
    c2 = topology.addAtom("C", element.carbon, last) if carbonyl else None
    topology.addAtom("O", element.oxygen, last)
    if oxt:
        topology.addAtom("OXT", element.oxygen, last)
    topology.addBond(c1, n2)
    if closure and c2 is not None:
        topology.addBond(c2, n1)
    return topology


class TestFindOpenCTermini:
    def test_a_standard_residue_without_oxt_is_open(self):
        assert find_open_c_termini(_chain("VAL")) == [("A", "VAL", "2")]

    def test_a_protonation_variant_is_open_too(self):
        assert find_open_c_termini(_chain("CYX")) == [("A", "CYX", "2")]

    def test_oxt_closes_the_chain(self):
        assert find_open_c_termini(_chain("VAL", oxt=True)) == []

    def test_a_head_to_tail_bond_closes_the_chain(self):
        assert find_open_c_termini(_chain("VAL", closure=True)) == []

    def test_a_cap_is_not_an_amino_acid(self):
        assert find_open_c_termini(_chain("NME")) == []

    def test_a_non_standard_last_residue_is_left_to_the_force_field(self):
        assert find_open_c_termini(_chain("BMT")) == []

    def test_a_residue_without_a_carbonyl_carbon_is_not_judged(self):
        assert find_open_c_termini(_chain("VAL", carbonyl=False)) == []

    def test_an_empty_topology_has_none(self):
        assert find_open_c_termini(Topology()) == []


class TestExamples:
    def test_3p8f_trypsin_chain_is_open(self):
        topology, positions = _load(SFTI)
        patch_cyclic_topology(topology, positions, "B")
        # the SFTI-1 peptide closes head to tail; the last VAL of the trypsin chain has no OXT
        assert find_open_c_termini(topology) == [("A", "VAL", "244")]

    def test_the_cyclic_peptide_of_1cwa_is_closed_once_patched(self):
        topology, positions = _load(CWA)
        topology, positions, _ = patch_cyclic_topology(topology, positions, "B")
        assert find_open_c_termini(topology) == []

    def test_the_capped_staple_example_is_closed(self):
        topology, _ = _load(STAPLE)
        assert find_open_c_termini(topology) == []

    def test_the_prepped_example_is_closed(self, prepped_example_cif):
        topology, _ = load_structure(prepped_example_cif)
        assert find_open_c_termini(topology) == []


class TestMessage:
    def test_it_names_the_residue_and_says_to_prepare(self):
        topology, _ = _load(SFTI)
        with pytest.raises(ValueError) as caught:
            require_closed_c_termini(topology)
        message = str(caught.value)
        assert "chain A ends in VAL244 without its terminal oxygen (OXT)" in message
        assert "not prepared" in message
        assert "binding-metrics-prep" in message and "--skip-prep" in message

    def test_it_names_the_author_chain(self, tmp_path):
        """Author IDs A and I renamed to B and A: the trypsin is author B (topology chain A)."""
        _load(SFTI)
        topology, _ = load_structure(_swap_author_ids(tmp_path, SFTI, {"A": "B", "I": "A"}))
        with pytest.raises(ValueError, match="chain B ends in VAL244"):
            require_closed_c_termini(topology)

    def test_a_closed_topology_passes(self):
        require_closed_c_termini(_chain("VAL", oxt=True))


class TestRefusedBeforeTheForceField:
    @pytest.mark.parametrize(
        "path, peptide, receptor, residue",
        [(SFTI, "I", "A", "VAL244"), (P53, "B", "A", "VAL109")],
        ids=["3P8F", "1YCR"],
    )
    def test_the_relaxation_refuses_a_raw_input(self, path, peptide, receptor, residue):
        _load(path)
        config = RelaxationConfig(peptide_chain_id=peptide, receptor_chain_id=receptor)
        with pytest.raises(ValueError, match=f"chain A ends in {residue} without its terminal"):
            ImplicitRelaxation(config)._setup_system(path)

    def test_a_failed_relaxation_reports_the_reason(self, tmp_path):
        _load(SFTI)
        config = RelaxationConfig(peptide_chain_id="I", receptor_chain_id="A", md_duration_ps=0)
        result = ImplicitRelaxation(config).run(SFTI, tmp_path)
        assert not result.success
        assert "chain A ends in VAL244" in result.error_message
        assert "No template found" not in result.error_message

    def test_the_energy_refuses_a_raw_input(self):
        _load(SFTI)
        result = energy.compute_interaction_energy(
            SFTI, peptide_chain="I", receptor_chain="A", modes=("raw",)
        )
        assert not result["success"]
        assert "chain A ends in VAL244" in result["error_message"]
        assert "No template found" not in result["error_message"]

    def test_the_energy_names_the_author_chain(self, tmp_path):
        _load(SFTI)
        swapped = _swap_author_ids(tmp_path, SFTI, {"A": "B", "I": "A"})
        result = energy.compute_interaction_energy(
            swapped, peptide_chain="A", receptor_chain="B", modes=("raw",)
        )
        assert "chain B ends in VAL244" in result["error_message"]

    def test_the_pipeline_with_skip_prep_says_so(self, tmp_path):
        pytest.importorskip("biotite")
        from binding_metrics.cli.run import run_pipeline

        results = run_pipeline(
            SFTI,
            tmp_path,
            skip_prep=True,
            md_duration_ps=0,
            metrics=frozenset({"energy"}),
        )
        assert results["relax"]["success"] is False
        assert "not prepared" in results["relax"]["error_message"]
        assert "not prepared" in results["energy"]["error_message"]
