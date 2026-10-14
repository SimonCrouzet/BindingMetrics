"""Warnings of the structural QC about chain breaks (#60).

A minimisation that bonded the two sides of a gap closed it, and the structural QC passed:
the pulled-together residues have an ordinary geometry (the C-alpha pair of a 3-residue
gap went from 5.7 to 3.9 A, and a 20 A gap ends at 3.9 A as well), so none of the seven
checks has anything to flag. ``check_relaxed_structure`` therefore adds a list
``warnings`` beside ``checks``. A warning is advisory: it never changes ``passed``. Two
keys exist:

* ``chain_break_closed``: C and N of two consecutive residues are more than 2.0 A apart
  in the input and bonded in the result;
* ``ca_ca_distance``: two consecutive residues are bonded in the result and their
  C-alphas are outside 2.5-4.2 A (a trans peptide has 3.8 A, a cis peptide 2.9 A).

Pairs across a break that is still open are not judged.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA_DIR

from binding_metrics.protocols import qc
from binding_metrics.protocols.qc import AtomSnapshot

gemmi = pytest.importorskip("gemmi", reason="gemmi reads the example structures")
pytest.importorskip("biotite", reason="biotite supplies the residue templates of a file's bonds")

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
EXAMPLES = sorted(DATA_DIR.glob("example_*"))


def _warning_keys(result: dict) -> list:
    return [warning["key"] for warning in result["warnings"]]


def _rows(snapshot: AtomSnapshot, chain: str, number: int, name: str) -> int:
    for i in np.nonzero(snapshot.heavy_mask)[0]:
        key = snapshot.residue_keys[i]
        if key[0] == chain and key[1] == number and snapshot.atom_names[i] == name:
            return int(i)
    raise KeyError((chain, number, name))


def _shifted_after(snapshot: AtomSnapshot, chain: str, first_number: int, dx: float):
    """The snapshot with every atom of ``chain`` from residue ``first_number`` on moved along x."""
    coords = snapshot.coords.copy()
    for i, key in enumerate(snapshot.residue_keys):
        if key[0] == chain and key[1] >= first_number:
            coords[i, 0] += dx
    return dataclasses.replace(snapshot, coords=coords)


@pytest.fixture(scope="module")
def ycr() -> AtomSnapshot:
    return AtomSnapshot.from_file(YCR)


@pytest.fixture(scope="module")
def ycr_with_break(ycr) -> AtomSnapshot:
    """1YCR with the second half of the peptide (chain B from residue 23) 0.6 nm away."""
    return _shifted_after(ycr, "B", 23, 6.0)


class TestChainBreakClosed:
    def test_a_break_that_the_result_bonds_is_warned_about(self, ycr, ycr_with_break):
        result = qc.check_relaxed_structure(ycr_with_break, ycr)

        assert _warning_keys(result) == ["chain_break_closed"]
        (warning,) = result["warnings"]
        assert warning["value"] == 1
        assert "B:22" in warning["detail"] and "B:23" in warning["detail"]
        # Advisory: the seven checks decide `passed`.
        assert result["passed"] is True
        assert result["failed"] == []

    def test_the_warning_is_logged_at_warning_level(self, ycr, ycr_with_break, caplog):
        with caplog.at_level(logging.WARNING, logger="binding_metrics.protocols.qc"):
            qc.check_relaxed_structure(ycr_with_break, ycr)
        records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("chain_break_closed" in r.getMessage() for r in records)

    def test_a_break_that_stays_open_gives_no_warning(self, ycr_with_break):
        result = qc.check_relaxed_structure(ycr_with_break, ycr_with_break)
        assert result["warnings"] == []

    def test_an_unbroken_structure_gives_no_warning(self, ycr):
        assert qc.check_relaxed_structure(ycr, ycr)["warnings"] == []

    def test_the_break_limit_is_the_one_of_find_chain_breaks(self):
        pytest.importorskip("openmm")
        from binding_metrics.core.system import _PEPTIDE_BOND_MAX_ANGSTROM

        assert qc.CHAIN_BREAK_MIN_C_N_ANGSTROM == _PEPTIDE_BOND_MAX_ANGSTROM


class TestCaCaDistance:
    def test_a_stretched_pair_is_warned_about(self, ycr):
        ca_before = _rows(ycr, "B", 25, "CA")
        ca_after = _rows(ycr, "B", 26, "CA")
        direction = ycr.coords[ca_after] - ycr.coords[ca_before]
        direction /= np.linalg.norm(direction)
        coords = ycr.coords.copy()
        coords[ca_after] += 1.0 * direction  # 3.8 -> 4.8 A, C and N keep their bond
        result = qc.check_relaxed_structure(ycr, dataclasses.replace(ycr, coords=coords))

        assert _warning_keys(result) == ["ca_ca_distance"]
        assert "B:25" in result["warnings"][0]["detail"]

    def test_a_compressed_pair_is_warned_about(self, ycr):
        ca_before = _rows(ycr, "B", 25, "CA")
        ca_after = _rows(ycr, "B", 26, "CA")
        direction = ycr.coords[ca_after] - ycr.coords[ca_before]
        direction /= np.linalg.norm(direction)
        coords = ycr.coords.copy()
        coords[ca_after] -= 1.5 * direction  # 3.8 -> 2.3 A
        result = qc.check_relaxed_structure(ycr, dataclasses.replace(ycr, coords=coords))

        assert _warning_keys(result) == ["ca_ca_distance"]

    def test_a_pair_across_an_open_break_is_not_judged(self, ycr_with_break):
        """Its C-alphas are 6 A or more apart, and the break is the declared reason."""
        result = qc.check_relaxed_structure(ycr_with_break, ycr_with_break)
        assert "ca_ca_distance" not in _warning_keys(result)

    @pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem.split("_")[-1])
    def test_no_bundled_example_is_warned_about(self, path):
        """Cis proline (3P8F, 2.85 A), a stapled residue (3V3B, 3.96 A), N-methylated peptides."""
        snapshot = AtomSnapshot.from_file(path)
        assert qc.check_relaxed_structure(snapshot, snapshot)["warnings"] == []


class TestRelaxationResult:
    def test_the_warnings_are_a_column_of_the_result(self):
        pytest.importorskip("openmm")
        from binding_metrics.protocols.relaxation import RelaxationResult

        qc_block = {
            "passed": True,
            "failed": [],
            "checks": {},
            "warnings": [{"key": "chain_break_closed"}],
            "md_final": {"passed": True, "failed": [], "checks": {}, "warnings": [{"key": "x"}]},
        }
        row = RelaxationResult(sample_id="x", success=True, qc=qc_block, qc_passed=True).to_dict()
        assert row["qc_warnings"] == "chain_break_closed,md_final:x"
        assert row["qc_passed"] is True

    def test_the_column_is_empty_when_the_qc_did_not_run_or_found_nothing(self):
        pytest.importorskip("openmm")
        from binding_metrics.protocols.relaxation import RelaxationResult

        assert RelaxationResult(sample_id="x", success=False).to_dict()["qc_warnings"] == ""
        found_nothing = {"passed": True, "failed": [], "checks": {}, "warnings": []}
        assert (
            RelaxationResult(sample_id="x", success=True, qc=found_nothing).to_dict()["qc_warnings"]
            == ""
        )

    def test_a_qc_that_could_not_run_has_the_same_keys(self):
        pytest.importorskip("openmm")
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        skipped = ImplicitRelaxation(RelaxationConfig())._structural_qc("x", None, None, None)
        assert skipped["warnings"] == []
        assert skipped["passed"] is None


@pytest.fixture(scope="module")
def gapped(tmp_path_factory):
    """1YCR without residues 60-62 of chain A, prepared and saved as PDB."""
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    from openmm import app

    from binding_metrics.core.system import prep_structure
    from binding_metrics.io.structures import load_structure, save_structure

    topology, positions = load_structure(YCR)
    modeller = app.Modeller(topology, positions)
    modeller.delete(
        [r for r in modeller.topology.residues() if r.chain.id == "A" and 60 <= int(r.id) <= 62]
    )
    prepped, prepped_positions = prep_structure(modeller.topology, modeller.positions)
    path = tmp_path_factory.mktemp("gap") / "1YCR_gap.pdb"
    save_structure(prepped, prepped_positions, path)
    return path


@pytest.mark.slow
class TestRelaxedGap:
    """The behaviour of the relaxation before and after: bonded, the gap closes; unbonded, not."""

    def test_the_old_behaviour_is_caught(self, gapped):
        """Plain createSystem bonds the gap by name and the minimisation closes it."""
        import openmm
        from openmm import app

        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(gapped)
        force_field = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml", "implicit/obc2.xml")
        system = force_field.createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        simulation = app.Simulation(
            topology,
            system,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("CPU"),
        )
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy(maxIterations=300)
        relaxed = simulation.context.getState(getPositions=True).getPositions()

        before = AtomSnapshot.from_topology(topology, positions)
        after = AtomSnapshot.from_topology(topology, relaxed)
        result = qc.check_relaxed_structure(before, after)

        assert _warning_keys(result) == ["chain_break_closed"]
        # The geometry the closed gap ends in is an ordinary one: nothing else flags it.
        assert result["passed"] is True

    def test_the_relaxation_keeps_the_gap_and_the_qc_has_nothing_to_warn_about(
        self, gapped, tmp_path
    ):
        from binding_metrics.protocols.relaxation import ImplicitRelaxation, RelaxationConfig

        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            min_steps_initial=200,
            min_steps_restrained=100,
            min_steps_final=200,
            peptide_chain_id="B",
            receptor_chain_id="A",
        )
        result = ImplicitRelaxation(config).run(gapped, tmp_path / "out")

        assert result.success, result.error_message
        assert result.qc["warnings"] == []
        assert result.to_dict()["qc_warnings"] == ""
        assert result.qc_passed is True
        assert result.qc["checks"]["bond_lengths"]["passed"]
        after = AtomSnapshot.from_file(Path(result.minimized_structure_path))
        assert max(_c_n_distances(after)) > 3.0


def _c_n_distances(snapshot: AtomSnapshot) -> list:
    """C-N distance of every pair of consecutive residues of one chain."""
    from binding_metrics.protocols.qc import _backbone_pairs

    return [
        float(np.linalg.norm(snapshot.coords[rows_a["C"]] - snapshot.coords[rows_b["N"]]))
        for (_, rows_a), (_, rows_b) in _backbone_pairs(snapshot)
    ]
