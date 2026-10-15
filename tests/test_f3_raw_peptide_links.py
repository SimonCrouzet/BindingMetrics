"""The peptide links next to non-standard residues of a raw mmCIF (#195).

A model output has no ``_struct_conn``: OpenMM bonds the atoms inside the residues it knows and
nothing next to a residue it does not (D-alanine, N-methyl-leucine, BMT), so the neighbour of such
a residue has an open external bond and the force field reports ``No template found``. The links
are restored by distance when the topology is patched, whether or not the residue was bare.
"""

import math
import os
from pathlib import Path

import pytest
from conftest import DATA_DIR

pytest.importorskip("openmm")

from binding_metrics.core.cyclic import reconstruct_nonstandard_residue_bonds
from binding_metrics.io.structures import load_structure
from binding_metrics.metrics.energy import compute_interaction_energy
from binding_metrics.testing.synthetic import write_cif

CWA = DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif"
CPU = {"modes": ("raw",), "device": "cpu"}


def _bare_cwa(path: Path, *, drop_residue: int | None = None) -> Path:
    """1CWA without waters, written as a bare model; optionally without one peptide residue."""
    from binding_metrics.metrics._common import load_structure as load_atoms

    atoms = load_atoms(CWA)
    keep = atoms.res_name != "HOH"
    if drop_residue is not None:
        keep &= ~((atoms.chain_id == "C") & (atoms.res_id == drop_residue))
    write_cif(atoms[keep], path)
    assert "_struct_conn" not in path.read_text(encoding="utf-8")
    return path


def _open_junctions(topology) -> list:
    """(chain, name of residue i, name of residue i+1) of each C-N junction without a bond."""
    bonded = {frozenset((b.atom1.index, b.atom2.index)) for b in topology.bonds()}
    found = []
    for chain in topology.chains():
        residues = list(chain.residues())
        for previous, following in zip(residues, residues[1:]):
            c_atoms = [a.index for a in previous.atoms() if a.name == "C"]
            n_atoms = [a.index for a in following.atoms() if a.name == "N"]
            if c_atoms and n_atoms and frozenset((c_atoms[0], n_atoms[0])) not in bonded:
                found.append((chain.id, previous.name, following.name))
    return found


def test_the_links_next_to_non_standard_residues_are_restored_after_loading(tmp_path):
    topology, positions = load_structure(_bare_cwa(tmp_path / "bare.cif"))
    assert len(_open_junctions(topology)) == 8  # DAL, MLE, MVA, BMT, ABA and SAR have no link

    added = reconstruct_nonstandard_residue_bonds(topology, positions)

    assert added == 8
    assert _open_junctions(topology) == []
    assert reconstruct_nonstandard_residue_bonds(topology, positions) == 0


def test_a_gap_in_the_chain_is_not_bridged(tmp_path):
    topology, positions = load_structure(_bare_cwa(tmp_path / "gap.cif", drop_residue=5))

    reconstruct_nonstandard_residue_bonds(topology, positions)

    assert len(_open_junctions(topology)) == 1  # the junction over the missing residue


@pytest.mark.parametrize(
    "name",
    [
        "example_ncaa_cyclosporin_1CWA.cif",
        "example_bicyclic_sfti1_3P8F.cif",
        "example_lactam_somatostatin_1XY4.cif",
        "example_phospho_1QJB.pdb",
        "example_linear_p53_1YCR.pdb",
        "example_staple_3V3B.pdb",
    ],
)
def test_a_structure_that_lists_its_bonds_gains_none(name):
    topology, positions = load_structure(DATA_DIR / name)

    assert reconstruct_nonstandard_residue_bonds(topology, positions) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_the_energy_of_a_bare_model_is_that_of_the_bundled_file(tmp_path):
    bare = compute_interaction_energy(_bare_cwa(tmp_path / "bare.cif"), "C", "A", **CPU)
    bundled = compute_interaction_energy(CWA, "C", "A", **CPU)

    assert bare["success"], bare["error_message"]
    assert bundled["success"], bundled["error_message"]
    # The CPU platform sums on several threads in no fixed order: repeated calls differ by up
    # to about a kJ/mol, and the faults checked here move the energy by tens of kJ/mol.
    assert bare["raw_interaction_energy"] == pytest.approx(
        bundled["raw_interaction_energy"], abs=1.5
    )


SAVED = os.environ.get("BM_SAVED_PREDICTIONS")
SAVED_1CWA = (
    "boltz2/*/*/outputs/boltz_results_cwa_auto_s42/predictions/cwa_auto_s42/"
    "cwa_auto_s42_model_0.cif"
)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not SAVED, reason="BM_SAVED_PREDICTIONS names no directory of saved predictions"
)
def test_the_energy_of_a_saved_prediction_of_the_ring():
    found = sorted(Path(SAVED).glob(SAVED_1CWA))
    if not found:
        pytest.skip(f"no saved prediction matches {SAVED_1CWA}")

    result = compute_interaction_energy(found[0], "C", "A", complete_terminal_oxygens=True, **CPU)

    assert result["success"], result["error_message"]
    assert math.isfinite(result["raw_interaction_energy"])
    assert result["raw_interaction_energy"] < 0
