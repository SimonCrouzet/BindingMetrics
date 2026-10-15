"""A raw interaction energy of a structure that PDBFixer rebuilt atoms of carries a warning (#65).

PDBFixer puts a missing heavy atom where the template says, without looking at its
neighbours. On 1YCR, deleting the side chain of the receptor residue nearest to the peptide and
preparing again rebuilds the missing atoms and raises the raw interaction energy by the clash
energy of the rebuilt side chain, while the relaxed energy stays negative. The raw mode evaluates
the structure as it is, so on a structure that came straight out of preparation the number says
more about the clashes than about the binding, and nothing in the result said so.

``run_pipeline`` evaluates ``raw`` on the prepared structure when the relaxation is skipped or
fails. It then logs a warning and records the sentence under ``results["energy"]["warnings"]``,
the list of sentences that the pre-flight report and the pose-set report also call ``warnings``.
The energies themselves are not touched. The first class uses stubs for the preparation, the
relaxation and the energy; the last test runs the real thing on a GPU.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR, requires_cuda

from binding_metrics.cli.run import _collect_failures, run_pipeline
from binding_metrics.protocols.report import _flatten

EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"

RAW_ENERGY = 5054.0
RELAXED_ENERGY = -316.0


def _stub_prep(monkeypatch, n_rebuilt: int):
    """Preparation that changes nothing and reports ``n_rebuilt`` rebuilt heavy atoms."""
    from binding_metrics.core import system

    def prep(topology, positions, report=None, **kwargs):
        report["n_missing_atoms_rebuilt"] = n_rebuilt
        return topology, positions

    monkeypatch.setattr(system, "HAS_PDBFIXER", True)
    monkeypatch.setattr(system, "prep_structure", prep)


def _stub_energy(monkeypatch):
    """Energy step that records the file it was given and returns two modes."""
    from binding_metrics.metrics import energy

    calls: list[dict] = []

    def compute(path, **kwargs):
        calls.append({"path": Path(path), **kwargs})
        return {
            "success": True,
            "raw_interaction_energy": RAW_ENERGY,
            "relaxed_interaction_energy": RELAXED_ENERGY,
        }

    monkeypatch.setattr(energy, "compute_interaction_energy", compute)
    return calls


class _Relaxation:
    """Relaxation that succeeds (``structure`` given) or fails, without running OpenMM."""

    def __init__(self, structure: Path | None):
        self.structure = structure

    def run(self, path, output_dir, sample_id=None):
        structure = self.structure
        return _RelaxationResult(structure is not None, structure)


class _RelaxationResult:
    def __init__(self, success: bool, structure: Path | None):
        self.success = success
        self.error_message = None if success else "boom"
        self.md_final_structure_path = structure
        self.minimized_structure_path = structure

    def to_dict(self):
        return {"success": self.success}


def _run(tmp_path, **kwargs):
    kwargs.setdefault("skip_relax", True)
    kwargs.setdefault("energy_modes", ("raw",))
    return run_pipeline(EXAMPLE_1YCR, tmp_path, metrics=frozenset({"energy"}), **kwargs)


def _warnings_logged(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "raw interaction energy" in record.getMessage()
    ]


class TestRawEnergyOnRebuiltStructure:
    def test_raw_energy_of_a_prepared_structure_carries_a_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        _stub_prep(monkeypatch, 107)
        calls = _stub_energy(monkeypatch)
        results = _run(tmp_path)

        (note,) = results["energy"]["warnings"]
        assert "107 heavy atoms" in note
        assert "raw interaction energy" in note
        assert calls[0]["path"].name.endswith("_cleaned.cif"), "the energy reads the prepared file"
        assert _warnings_logged(caplog) == [f"  [warning] {note}"]

    def test_the_energies_are_not_changed(self, tmp_path, monkeypatch):
        _stub_prep(monkeypatch, 107)
        _stub_energy(monkeypatch)
        energy = _run(tmp_path, energy_modes=("raw", "relaxed"))["energy"]
        assert energy["raw_interaction_energy"] == RAW_ENERGY
        assert energy["relaxed_interaction_energy"] == RELAXED_ENERGY
        assert set(energy) == {
            "success",
            "raw_interaction_energy",
            "relaxed_interaction_energy",
            "warnings",
        }

    def test_even_one_rebuilt_atom_is_reported_with_the_right_count(self, tmp_path, monkeypatch):
        _stub_prep(monkeypatch, 1)
        _stub_energy(monkeypatch)
        (note,) = _run(tmp_path)["energy"]["warnings"]
        assert "1 heavy atom " in note

    def test_a_failed_relaxation_leaves_the_prepared_structure_to_the_energy(
        self, tmp_path, monkeypatch, caplog
    ):
        _stub_prep(monkeypatch, 12)
        _stub_energy(monkeypatch)
        results = _run(tmp_path, skip_relax=False, relaxer=_Relaxation(None))
        assert len(results["energy"]["warnings"]) == 1
        assert len(_warnings_logged(caplog)) == 1

    def test_no_rebuilt_atom_gives_no_warning(self, tmp_path, monkeypatch, caplog):
        _stub_prep(monkeypatch, 0)
        _stub_energy(monkeypatch)
        results = _run(tmp_path)
        assert "warnings" not in results["energy"]
        assert _warnings_logged(caplog) == []

    def test_relaxed_mode_alone_gives_no_warning(self, tmp_path, monkeypatch, caplog):
        _stub_prep(monkeypatch, 107)
        _stub_energy(monkeypatch)
        results = _run(tmp_path, energy_modes=("relaxed",))
        assert "warnings" not in results["energy"]
        assert _warnings_logged(caplog) == []

    def test_a_successful_relaxation_gives_no_warning(self, tmp_path, monkeypatch, caplog):
        _stub_prep(monkeypatch, 107)
        calls = _stub_energy(monkeypatch)
        relaxed = tmp_path / "relaxed.cif"
        results = _run(tmp_path, skip_relax=False, relaxer=_Relaxation(relaxed))
        assert calls[0]["path"] == relaxed, "the energy reads the relaxed file"
        assert "warnings" not in results["energy"]
        assert _warnings_logged(caplog) == []

    def test_skipped_prep_gives_no_warning(self, tmp_path, monkeypatch, caplog):
        _stub_energy(monkeypatch)
        results = _run(tmp_path, skip_prep=True)
        assert "warnings" not in results["energy"]
        assert _warnings_logged(caplog) == []

    def test_a_failed_energy_step_keeps_its_error_alone(self, tmp_path, monkeypatch, caplog):
        from binding_metrics.metrics import energy

        _stub_prep(monkeypatch, 107)

        def fail(path, **kwargs):
            raise RuntimeError("no platform")

        monkeypatch.setattr(energy, "compute_interaction_energy", fail)
        results = _run(tmp_path)
        assert results["energy"] == {"error": "no platform"}
        assert _warnings_logged(caplog) == []

    def test_the_warning_is_not_a_failure_and_adds_no_csv_column(self, tmp_path, monkeypatch):
        _stub_prep(monkeypatch, 107)
        _stub_energy(monkeypatch)
        results = _run(tmp_path, energy_modes=("raw", "relaxed"))
        assert _collect_failures(results) == []
        flat = _flatten(results)
        assert not [column for column in flat if "warning" in column]
        assert flat["energy_raw_interaction_energy"] == RAW_ENERGY


def _residue_nearest_to_chain(topology, positions, chain_id: str, other_chain_id: str):
    """The non-glycine, non-alanine residue of ``chain_id`` closest to ``other_chain_id``."""
    xyz = np.array([[p.x, p.y, p.z] for p in positions])
    other = [a.index for a in topology.atoms() if a.residue.chain.id == other_chain_id]

    def distance(residue):
        own = [a.index for a in residue.atoms()]
        return np.linalg.norm(xyz[own][:, None] - xyz[other][None], axis=-1).min()

    candidates = [
        r for r in topology.residues() if r.chain.id == chain_id and r.name not in ("GLY", "ALA")
    ]
    return min(candidates, key=distance)


@pytest.mark.slow
@pytest.mark.integration
@requires_cuda
def test_real_interface_side_chain_rebuilt_by_pdbfixer(tmp_path):
    """The raw value of 1YCR with one interface side chain rebuilt is unchanged by the warning."""
    from openmm.app import Modeller, PDBFile

    from binding_metrics.io.structures import load_structure
    from binding_metrics.metrics.energy import compute_interaction_energy

    topology, positions = load_structure(EXAMPLE_1YCR)
    residue = _residue_nearest_to_chain(topology, positions, "A", "B")
    keep = {"N", "CA", "C", "O", "CB", "OXT"}
    modeller = Modeller(topology, positions)
    modeller.delete(
        [
            atom
            for atom in modeller.topology.atoms()
            if atom.residue.index == residue.index
            and atom.name not in keep
            and not atom.name.startswith("H")
        ]
    )
    damaged = tmp_path / "damaged.pdb"
    with open(damaged, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle)

    results = run_pipeline(
        damaged,
        tmp_path / "run",
        skip_relax=True,
        metrics=frozenset({"energy"}),
        energy_modes=("raw", "relaxed"),
        peptide_chain="B",
        receptor_chain="A",
    )
    energy = results["energy"]
    n_rebuilt = results["prep"]["n_missing_atoms_rebuilt"]
    assert n_rebuilt > 1
    (note,) = energy["warnings"]
    assert f"{n_rebuilt} heavy atoms" in note
    # the rebuilt side chain clashes with its neighbours until the relaxation removes the overlap,
    # so the relaxed complex binds more favourably than the raw one
    assert energy["raw_interaction_energy"] > energy["relaxed_interaction_energy"]
    assert energy["relaxed_interaction_energy"] < 0.0

    direct = compute_interaction_energy(
        results["prep"]["output"], peptide_chain="B", receptor_chain="A", modes=("raw",)
    )
    assert energy["raw_interaction_energy"] == pytest.approx(
        direct["raw_interaction_energy"], rel=1e-6
    )
