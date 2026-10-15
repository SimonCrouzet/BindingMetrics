"""``add_terminal_oxygens`` on a chain that has not the atoms to place the OXT (#198).

The OXT is placed in the plane of CA, C and O of the last residue. A last residue that lacks
CA or O gives no plane: the chain stays open, so that ``require_closed_c_termini`` names it, as
it does for the default call of the interaction energy.
"""

import pytest
from conftest import DATA_DIR

pytest.importorskip("openmm")

from openmm.app import Modeller

from binding_metrics.core.system import (
    add_terminal_oxygens,
    find_open_c_termini,
    require_closed_c_termini,
)
from binding_metrics.io.structures import load_structure
from binding_metrics.metrics.energy import compute_interaction_energy

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"


def _without_atom(tmp_path, atom_name: str):
    """1YCR without one atom of VAL 109 of chain A and without the OXT of chain B."""
    kept = [
        line
        for line in YCR.read_text(encoding="utf-8").splitlines(keepends=True)
        if not (line[12:16].strip() == atom_name and line[21] == "A" and line[22:26] == " 109")
        and not (line[12:16].strip() == "OXT" and line[21] == "B")
    ]
    path = tmp_path / f"ycr_no_{atom_name}.pdb"
    path.write_text("".join(kept), encoding="utf-8")
    return path


@pytest.mark.parametrize("atom_name", ["O", "CA"])
def test_a_chain_without_the_atoms_for_the_plane_is_left_open(tmp_path, atom_name):
    topology, positions = load_structure(_without_atom(tmp_path, atom_name))
    assert [chain for chain, _, _ in find_open_c_termini(topology)] == ["A", "B"]

    completed, _ = add_terminal_oxygens(topology, positions)

    assert find_open_c_termini(completed) == [("A", "VAL", "109")]  # chain B got its OXT
    with pytest.raises(ValueError, match="chain A ends in VAL109 without its terminal oxygen"):
        require_closed_c_termini(completed)


def test_the_arguments_come_back_when_no_chain_can_be_completed(tmp_path):
    topology, positions = load_structure(_without_atom(tmp_path, "O"))
    modeller = Modeller(topology, positions)
    modeller.delete([c for c in topology.chains() if c.id == "B"])

    completed, completed_positions = add_terminal_oxygens(modeller.topology, modeller.positions)

    assert completed is modeller.topology
    assert completed_positions is modeller.positions


@pytest.mark.integration
def test_the_energy_names_the_chain_that_cannot_be_completed(tmp_path):
    result = compute_interaction_energy(
        _without_atom(tmp_path, "O"),
        "B",
        "A",
        modes=("raw",),
        device="cpu",
        complete_terminal_oxygens=True,
    )

    assert result["success"] is False
    assert "chain A ends in VAL109 without its terminal oxygen (OXT)" in result["error_message"]
