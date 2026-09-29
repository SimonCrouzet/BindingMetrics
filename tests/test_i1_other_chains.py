"""A third protein chain must not enter the relaxation or the interaction energy.

Without the drop, E_complex holds the bystander chain while the isolated peptide and
receptor do not, so E_int is inconsistent, and a bystander that lost its caps in
``strip_heterogens`` cannot be built by the force field at all.
"""

import logging
from pathlib import Path

import pytest
from conftest import requires_cuda

pytest.importorskip("openmm")

from openmm import Vec3, unit
from openmm.app import Modeller, PDBFile, Topology, element

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
P53_MDM2 = DATA_DIR / "example_linear_p53_1YCR.pdb"  # chain A: MDM2 (85), chain B: p53 (13)


def _copy_of_chain(topology, positions, chain_id: str, new_id: str, shift_nm: float):
    """(topology, positions) holding one chain of the input, renamed and moved along x."""
    modeller = Modeller(topology, positions)
    others = [c for c in topology.chains() if c.id != chain_id]
    modeller.delete(others)
    for chain in modeller.topology.chains():
        chain.id = new_id
    moved = [Vec3(p.x + shift_nm, p.y, p.z) for p in modeller.positions]
    return modeller.topology, unit.Quantity(moved, unit.nanometer)


def _with_bystander(pair=None, chain_to_copy: str = "B", new_id: str = "C", shift_nm: float = 6.0):
    """The pair (``(topology, positions)``, default the raw 1YCR) plus a copy of one chain
    as a third chain, 6 nm away."""
    topology, positions = pair or _load_pair()
    extra_topology, extra_positions = _copy_of_chain(
        topology, positions, chain_to_copy, new_id, shift_nm
    )
    modeller = Modeller(topology, positions)
    modeller.add(extra_topology, extra_positions)
    return modeller.topology, modeller.positions


def _load_pair():
    pdb = PDBFile(str(P53_MDM2))
    return pdb.topology, pdb.positions


@pytest.fixture(scope="module")
def prepped_pair(tmp_path_factory):
    """The prepped 1YCR complex as a two-chain file and as a three-chain file (a bystander copy)."""
    pytest.importorskip("pdbfixer")
    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_cif

    directory = tmp_path_factory.mktemp("prepped_pair")
    topology, positions = prep_structure(*load_structure(P53_MDM2))
    pair, triple = directory / "pair.cif", directory / "triple.cif"
    save_cif(topology, positions, pair)
    save_cif(*_with_bystander((topology, positions)), triple)
    return pair, triple


def _chain_ids(topology):
    return [chain.id for chain in topology.chains()]


class TestDropOtherProteinChains:
    def test_the_bystander_chain_is_removed_and_reported(self):
        from binding_metrics.io.structures import drop_other_protein_chains

        topology, positions = _with_bystander()
        report: dict = {}
        top, pos = drop_other_protein_chains(topology, positions, "B", "A", report=report)
        assert _chain_ids(top) == ["A", "B"]
        assert len(pos) == top.getNumAtoms()
        assert report == {"dropped_protein_chains": ["C"]}

    def test_the_warning_names_the_dropped_chain_and_the_kept_pair(self, caplog):
        from binding_metrics.io.structures import drop_other_protein_chains

        topology, positions = _with_bystander()
        with caplog.at_level(logging.WARNING, logger="binding_metrics.io.structures"):
            drop_other_protein_chains(topology, positions, "B", "A")
        (record,) = [r for r in caplog.records if "protein chain" in r.getMessage()]
        message = record.getMessage()
        assert record.levelno == logging.WARNING
        assert "C" in message and "peptide (B)" in message and "receptor (A)" in message

    def test_nothing_to_drop_leaves_the_topology_and_an_empty_report(self):
        from binding_metrics.io.structures import drop_other_protein_chains

        pdb = PDBFile(str(P53_MDM2))
        report: dict = {}
        top, pos = drop_other_protein_chains(pdb.topology, pdb.positions, "B", "A", report=report)
        assert top is pdb.topology and pos is pdb.positions
        assert report == {"dropped_protein_chains": []}

    def test_report_accumulates_over_calls(self):
        from binding_metrics.io.structures import drop_other_protein_chains

        report: dict = {}
        for _ in range(2):
            drop_other_protein_chains(*_with_bystander(), "B", "A", report=report)
        assert report["dropped_protein_chains"] == ["C", "C"]

    @pytest.mark.parametrize(
        ("peptide", "receptor"),
        [("B", None), (None, "A"), (None, None), ("B", "Z"), ("Z", "A")],
        ids=["no receptor", "no peptide", "no roles", "unknown receptor", "unknown peptide"],
    )
    def test_nothing_is_removed_unless_both_chains_are_named_and_present(self, peptide, receptor):
        """A wrong chain ID must not delete the real receptor."""
        from binding_metrics.io.structures import drop_other_protein_chains

        topology, positions = _with_bystander()
        top, _ = drop_other_protein_chains(topology, positions, peptide, receptor)
        assert _chain_ids(top) == ["A", "B", "C"]

    def test_a_chain_of_d_amino_acids_counts_as_protein(self):
        from binding_metrics.io.structures import drop_other_protein_chains

        topology, positions = _with_bystander()
        for residue in topology.residues():
            if residue.chain.id == "C":
                residue.name = "DAL"
        top, _ = drop_other_protein_chains(topology, positions, "B", "A")
        assert _chain_ids(top) == ["A", "B"]

    def test_a_chain_without_amino_acids_is_left_to_strip_heterogens(self):
        from binding_metrics.io.structures import drop_other_protein_chains

        pdb = PDBFile(str(P53_MDM2))
        ion = Topology()
        residue = ion.addResidue("ZN", ion.addChain("Z"))
        ion.addAtom("ZN", element.zinc, residue)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.add(ion, unit.Quantity([Vec3(9.0, 9.0, 9.0)], unit.nanometer))
        top, _ = drop_other_protein_chains(modeller.topology, modeller.positions, "B", "A")
        assert _chain_ids(top) == ["A", "B", "Z"]

    def test_strip_heterogens_alone_keeps_the_other_protein_chain(self):
        from binding_metrics.io.structures import strip_heterogens

        topology, positions = _with_bystander()
        top, _ = strip_heterogens(topology, positions, "B", "A")
        assert _chain_ids(top) == ["A", "B", "C"]


class TestRelaxationDropsTheBystander:
    def test_strip_step_forwards_the_report(self):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        relaxer = ImplicitRelaxation(RelaxationConfig())
        report: dict = {}
        top, _ = relaxer._strip_heterogens(*_with_bystander(), "B", "A", report=report)
        assert _chain_ids(top) == ["A", "B"]
        assert report["dropped_protein_chains"] == ["C"]

    def test_result_carries_the_dropped_chains(self):
        from binding_metrics.protocols.relaxation import RelaxationResult

        assert (
            RelaxationResult(sample_id="x", success=True).to_dict()["dropped_protein_chains"] == []
        )
        result = RelaxationResult(sample_id="x", success=True, dropped_protein_chains=["C"])
        assert result.to_dict()["dropped_protein_chains"] == ["C"]

    def test_the_system_is_built_from_the_pair_only(self, prepped_pair, tmp_path):
        """A 3-chain file gives the system of the 2-chain file, and the run says what it dropped."""
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        pair, triple = prepped_pair
        relaxer = ImplicitRelaxation(RelaxationConfig(peptide_chain_id="B", receptor_chain_id="A"))
        system_pair, top_pair, _, _ = relaxer._setup_system(pair)
        assert relaxer._dropped_protein_chains == []
        system_triple, top_triple, _, _ = relaxer._setup_system(triple)
        assert relaxer._dropped_protein_chains == ["C"]
        assert system_triple.getNumParticles() == system_pair.getNumParticles()
        assert _chain_ids(top_triple) == _chain_ids(top_pair) == ["A", "B"]


class TestEnergyDropsTheBystander:
    def test_the_energy_function_drops_after_stripping(self, monkeypatch):
        """Wiring only: the strip step runs first, then the drop, with the resolved chains."""
        import binding_metrics.io.structures as structures
        from binding_metrics.metrics.energy import compute_interaction_energy

        calls: list = []

        def fake_strip(topology, positions, peptide_chain, receptor_chain):
            calls.append(("strip", peptide_chain, receptor_chain))
            return topology, positions

        def fake_drop(topology, positions, peptide_chain, receptor_chain):
            calls.append(("drop", peptide_chain, receptor_chain))
            raise RuntimeError("stop after the drop")

        monkeypatch.setattr(structures, "strip_heterogens", fake_strip)
        monkeypatch.setattr(structures, "drop_other_protein_chains", fake_drop)
        compute_interaction_energy(P53_MDM2, peptide_chain="B", receptor_chain="A", modes=("raw",))
        assert calls == [("strip", "B", "A"), ("drop", "B", "A")]

    @requires_cuda
    @pytest.mark.integration
    def test_e_int_of_a_bystander_complex_equals_that_of_the_pair(self, prepped_pair):
        from binding_metrics.metrics.energy import compute_interaction_energy

        pair, triple = prepped_pair

        e_pair = compute_interaction_energy(
            pair, peptide_chain="B", receptor_chain="A", modes=("raw",)
        )
        e_triple = compute_interaction_energy(
            triple, peptide_chain="B", receptor_chain="A", modes=("raw",)
        )
        assert e_pair["success"] and e_triple["success"]
        assert e_triple["raw_interaction_energy"] == pytest.approx(
            e_pair["raw_interaction_energy"], abs=0.5
        )
        assert e_triple["raw_e_complex"] == pytest.approx(e_pair["raw_e_complex"], abs=0.5)
