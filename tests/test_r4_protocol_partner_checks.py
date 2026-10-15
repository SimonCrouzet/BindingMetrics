"""Empty and overlapping partner selections are rejected at the protocol boundary (#171)."""

from pathlib import Path

import pytest
from conftest import DATA_DIR
from openmm.app import Modeller, PDBFile
from test_r4_protocol_chain_ids import HYDROGEN, OXYGEN, small_chain

from binding_metrics.protocols import peptide
from binding_metrics.protocols.base import BaseProtocol, ProtocolResults

EXAMPLE = DATA_DIR / "example_linear_p53_1YCR.pdb"


class MinimalProtocol(BaseProtocol):
    def run(self, output_dir, **kwargs):
        return Path(output_dir) / "trajectory.dcd"

    def analyze(self, trajectory_path=None):
        return ProtocolResults()


class TestPartnerIdsGivenToTheConstructor:
    @pytest.mark.parametrize("protocol_class", [MinimalProtocol, peptide.PeptideBindingProtocol])
    def test_a_chain_in_both_partners_is_rejected(self, protocol_class):
        with pytest.raises(
            ValueError, match=r"ligand_chain 'A' is also in receptor_chains \['A', 'B'\]"
        ):
            protocol_class(EXAMPLE, "A", ["A", "B"])

    @pytest.mark.parametrize("protocol_class", [MinimalProtocol, peptide.PeptideBindingProtocol])
    def test_an_empty_receptor_selection_is_rejected(self, protocol_class):
        with pytest.raises(ValueError, match="receptor_chains is empty"):
            protocol_class(EXAMPLE, "B", [])

    @pytest.mark.parametrize("receptors", [[""], ["A", ""]])
    def test_an_empty_receptor_id_is_rejected(self, receptors):
        with pytest.raises(ValueError, match="receptor_chains .* empty chain ID"):
            MinimalProtocol(EXAMPLE, "B", receptors)

    def test_an_empty_ligand_id_is_rejected(self):
        with pytest.raises(ValueError, match="ligand_chain is empty"):
            MinimalProtocol(EXAMPLE, "", ["A"])

    def test_distinct_partners_are_accepted_and_kept_as_given(self):
        receptors = ["A", "C"]
        protocol = MinimalProtocol(EXAMPLE, "B", receptors)

        assert protocol.ligand_chain == "B"
        assert protocol.receptor_chains is receptors


class TestPartnersAgainstTheStructure:
    @pytest.fixture
    def preparation_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(peptide, "get_forcefield", lambda name: calls.append("forcefield"))
        monkeypatch.setattr(
            peptide, "prepare_system", lambda pdb, forcefield: calls.append("prepare")
        )
        return calls

    def test_run_names_the_missing_chain_and_the_chains_present(self, preparation_calls, tmp_path):
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, "L", ["A"])

        with pytest.raises(ValueError, match=r"ligand_chain 'L' not in .*1YCR.pdb; chain IDs"):
            protocol.run(tmp_path)

        assert preparation_calls == []

    def test_run_names_every_missing_receptor_chain(self, preparation_calls, tmp_path):
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, "B", ["A", "R", "S"])

        with pytest.raises(
            ValueError, match=r"receptor_chains 'R', 'S' not in .*; chain IDs present: 'A', 'B'"
        ):
            protocol.run(tmp_path)

    def test_run_ignores_a_chain_of_water_when_it_lists_the_chains_present(
        self, preparation_calls, tmp_path, monkeypatch
    ):
        pdb = PDBFile(str(EXAMPLE))
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.add(*small_chain("W", "HOH", [("O", OXYGEN), ("H1", HYDROGEN)]))
        monkeypatch.setattr(peptide, "load_complex", lambda path: modeller)
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, "W", ["A"])

        with pytest.raises(
            ValueError, match=r"ligand_chain 'W' not in .*; chain IDs present: 'A', 'B'"
        ):
            protocol.run(tmp_path)

    def test_analyze_without_run_names_the_empty_selection(self, tmp_path):
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, "L", ["R"])

        with pytest.raises(
            ValueError,
            match=r"ligand_chain 'L' selects no atoms in .*1YCR.pdb; chain IDs present: 'A', 'B'",
        ):
            protocol.analyze(trajectory_path=tmp_path / "never_read.dcd")

    def test_analyze_names_the_empty_receptor_selection(self, tmp_path):
        protocol = peptide.PeptideBindingProtocol(EXAMPLE, "B", ["R"])

        with pytest.raises(
            ValueError,
            match=r"receptor_chains 'R' selects no atoms in .*; chain IDs present: 'A', 'B'",
        ):
            protocol.analyze(trajectory_path=tmp_path / "never_read.dcd")
