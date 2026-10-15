"""``compute_interaction_energy`` takes the chain IDs as author IDs (#112).

1CWA: the peptide is author chain C and chain B of the OpenMM topology, whose chain C holds 140
waters. With ``peptide_chain="C"`` the function took the waters for the peptide, removed the
peptide residues as heterogens and dropped chain B as a third protein chain, and reported
success: 7542 contacts and -674.7 kJ/mol where the peptide gives 2288 contacts and -260.4 kJ/mol.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("gemmi")

sys.path.insert(0, str(Path(__file__).parent))
from conftest import requires_cuda  # noqa: E402

from binding_metrics.metrics import energy  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
CWA = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"

try:
    import openmmforcefields  # noqa: F401

    HAS_OMMFF = True
except ImportError:
    HAS_OMMFF = False
requires_ommff = pytest.mark.skipif(not HAS_OMMFF, reason="openmmforcefields not installed")


class _StopError(Exception):
    """Raised by the stand-in for the system builder, after it has recorded its arguments."""


@pytest.fixture
def builder_calls(monkeypatch):
    """Replace the force-field step by a recorder of the chain and topology it is given."""
    calls: list = []

    def recording(topology, positions, solvent_model, peptide_chain=None, **kwargs):
        chains = {chain.id: sum(1 for _ in chain.residues()) for chain in topology.chains()}
        calls.append({"peptide_chain": peptide_chain, "chains": chains})
        raise _StopError

    monkeypatch.setattr(energy, "_create_implicit_system", recording)
    return calls


def _run(path, peptide, receptor):
    if not path.exists():
        pytest.skip(f"bundled example not found: {path}")
    return energy.compute_interaction_energy(
        path, peptide_chain=peptide, receptor_chain=receptor, modes=("raw",)
    )


class TestChainSelection:
    def test_1cwa_author_id_selects_the_peptide(self, builder_calls):
        result = _run(CWA, "C", "A")
        (call,) = builder_calls
        # topology chain B is the peptide; the waters (chains C and D) are stripped
        assert call["peptide_chain"] == "B"
        assert call["chains"] == {"A": 165, "B": 11}
        assert result["num_contacts"] == 2288

    def test_1cwa_topology_id_gives_the_same_selection(self, builder_calls):
        by_author = _run(CWA, "C", "A")
        by_topology = _run(CWA, "B", "A")
        assert builder_calls[0] == builder_calls[1]
        assert by_author["num_contacts"] == by_topology["num_contacts"] == 2288
        assert by_author["num_close_contacts"] == by_topology["num_close_contacts"]

    def test_auto_detection_is_unchanged(self, builder_calls):
        result = _run(CWA, None, None)
        assert builder_calls[0]["peptide_chain"] == "B"
        assert result["num_contacts"] == 2288

    def test_3p8f_author_id_selects_the_peptide(self, builder_calls):
        result = _run(SFTI, "I", "A")
        (call,) = builder_calls
        assert call["peptide_chain"] == "B"
        assert call["chains"] == {"A": 241, "B": 14}
        assert result["num_contacts"] == _run(SFTI, "B", "A")["num_contacts"]

    def test_a_pdb_input_is_unchanged(self, builder_calls):
        _run(P53, "B", "A")
        assert builder_calls[0]["peptide_chain"] == "B"
        assert builder_calls[0]["chains"] == {"A": 85, "B": 13}

    def test_letters_swapped_between_label_and_author_ids(self, builder_calls, tmp_path):
        """Author A is chain B of the topology and author B is chain A; waters share author A."""
        from tests.test_fix_64_chain_ids import _write_cif

        path = _write_cif(
            tmp_path / "swapped.cif",
            [("GLY", "A", "B")] * 6 + [("GLY", "B", "A")] * 3 + [("HOH", "C", "A")] * 2,
        )
        # the peptide is the 3-residue author chain A, which is chain B of the topology
        result = _run(path, "A", "B")
        (call,) = builder_calls
        assert call["peptide_chain"] == "B"
        assert call["chains"] == {"A": 6, "B": 3}
        assert result["num_contacts"] > 0

    def test_the_log_names_the_author_ids(self, builder_calls, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="binding_metrics.metrics.energy"):
            _run(CWA, "C", "A")
        assert "peptide=C, receptor=A" in caplog.text
        assert "peptide=B" not in caplog.text

    def test_the_command_line_takes_author_ids(self, builder_calls, tmp_path, monkeypatch):
        pytest.importorskip("pandas")
        monkeypatch.setattr(
            sys,
            "argv",
            ["binding-metrics-energy", "--input", str(CWA), "--output", str(tmp_path / "e.csv")]
            + ["--peptide-chain", "C", "--receptor-chain", "A", "--modes", "raw"],
        )
        energy.main()
        assert builder_calls[0]["peptide_chain"] == "B"


@requires_cuda
@requires_ommff
@pytest.mark.integration
class TestRawEnergyOfTheRawFile:
    @pytest.fixture(scope="class")
    def energies(self):
        if not CWA.exists():
            pytest.skip(f"bundled example not found: {CWA}")
        return {
            chain: energy.compute_interaction_energy(
                CWA, peptide_chain=chain, receptor_chain="A", modes=("raw",), device="cuda"
            )
            for chain in ("C", "B")
        }

    def test_the_author_id_gives_the_energy_of_the_peptide(self, energies):
        assert energies["C"]["success"], energies["C"]["error_message"]
        assert energies["C"]["num_contacts"] == 2288
        assert energies["C"]["raw_interaction_energy"] == pytest.approx(
            energies["B"]["raw_interaction_energy"], rel=1e-3
        )
        assert energies["C"]["raw_interaction_energy"] == pytest.approx(-260.4, abs=2.0)


@requires_cuda
@requires_ommff
@pytest.mark.integration
def test_the_energy_of_a_file_with_swapped_letters(tmp_path):
    """1CWA with author IDs A and C renamed to B and A: the peptide is author A, topology B.

    The function holds topology IDs while it patches the peptide, on a topology that still
    carries its author IDs; the energy is that of the original file.
    """
    from tests.test_fix_64_cyclic import _swap_author_ids

    if not CWA.exists():
        pytest.skip(f"bundled example not found: {CWA}")
    swapped = _swap_author_ids(tmp_path, CWA, {"A": "B", "C": "A"})
    result = energy.compute_interaction_energy(
        swapped, peptide_chain="A", receptor_chain="B", modes=("raw",), device="cuda"
    )
    assert result["success"], result["error_message"]
    assert result["num_contacts"] == 2288
    assert result["raw_interaction_energy"] == pytest.approx(-260.4, abs=2.0)
