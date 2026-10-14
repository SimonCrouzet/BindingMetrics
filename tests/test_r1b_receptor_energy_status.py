"""A failed energy term of ``receptor_quality`` in ``score_pose_set`` (#177).

The energy term reports its failure in ``error`` and, as every term does, in ``reason``; the
block names the failed term in ``error_message`` (the shape of a metric that failed in part, so
the pose row is ``partial`` and the terms that were computed stay in the aggregates).
"""

import contextlib

import numpy as np
import pytest

pytest.importorskip("biotite.structure")

import biotite.structure as struc
import biotite.structure.io as sio
from conftest import DATA_DIR

from binding_metrics.metrics import receptor_quality as rq
from binding_metrics.poses.score import score_pose_set
from binding_metrics.testing import make_synthetic_poses

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
NO_OPENMM = "openmm is not importable here"


def _no_openmm():
    raise ImportError(NO_OPENMM)


def _ycr_receptor(n_residues: int = 12):
    atoms = sio.load_structure(str(YCR))
    atoms = atoms[0] if isinstance(atoms, struc.AtomArrayStack) else atoms
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.chain_id == "A")]
    first = atoms.res_id[0]
    return atoms[atoms.res_id < first + n_residues]


def _assert_failed_term(term: dict) -> None:
    assert np.isnan(term["energy_kJ_mol"]) and np.isnan(term["energy_per_residue_kJ_mol"])
    assert isinstance(term["error"], str) and term["error"]
    assert term["reason"] == term["error"]


class TestEveryFailurePathHasAReason:
    def test_openmm_not_importable(self, monkeypatch):
        monkeypatch.setattr(rq, "_import_openmm", _no_openmm)
        term = rq._receptor_energy(_ycr_receptor())
        _assert_failed_term(term)
        assert term["error"] == NO_OPENMM

    def test_biotite_not_importable(self, monkeypatch):
        pytest.importorskip("openmm")

        def no_biotite(purpose):
            raise ImportError(f"biotite is required for {purpose}")

        monkeypatch.setattr(rq, "import_biotite", no_biotite)
        term = rq._receptor_energy(_ycr_receptor())
        _assert_failed_term(term)
        assert "biotite is required" in term["error"]

    def test_failure_while_building_the_system(self, monkeypatch):
        pytest.importorskip("openmm")
        import binding_metrics.core.system as system

        @contextlib.contextmanager
        def broken_placement(seed):
            raise RuntimeError("the hydrogen placement stopped")
            yield  # pragma: no cover

        monkeypatch.setattr(system, "deterministic_hydrogen_placement", broken_placement)
        term = rq._receptor_energy(_ycr_receptor())
        _assert_failed_term(term)
        assert term["error"].startswith("RuntimeError: the hydrogen placement stopped")

    @pytest.mark.slow
    def test_non_finite_energy(self, monkeypatch):
        pytest.importorskip("openmm")
        pytest.importorskip("pdbfixer")
        real = rq._import_openmm()

        class InfiniteEnergy:
            def __init__(self, *args, **kwargs):
                self.context = self

            def setPositions(self, positions):
                pass

            def getState(self, **kwargs):
                return self

            def getPotentialEnergy(self):
                return self

            def value_in_unit(self, unit):
                return float("inf")

        monkeypatch.setattr(rq, "_import_openmm", lambda: (*real[:5], InfiniteEnergy))
        term = rq._receptor_energy(_ycr_receptor(), device="cpu")
        _assert_failed_term(term)
        assert "non-finite energy" in term["error"]

    def test_chain_without_atoms(self):
        term = rq._score_model(_ycr_receptor(), "Z", 0.4, "obc2", "cpu", 1)["energy"]
        _assert_failed_term(term)
        assert term["error"] == "no receptor atoms found"


class TestBlockNamesTheFailedTerm:
    def test_the_failed_energy_is_named_in_error_message(self, monkeypatch):
        monkeypatch.setattr(rq, "_import_openmm", _no_openmm)
        result = rq.compute_receptor_quality(YCR, "A")
        assert result["error_message"] == f"energy of model 1: {NO_OPENMM}"
        assert "reason" not in result and "error" not in result
        assert np.isfinite(result["summary"]["ramachandran_favoured_pct"])

    def test_every_failed_model_is_named_once(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rq, "_import_openmm", _no_openmm)
        path = tmp_path / "ensemble.pdb"
        sio.save_structure(str(path), struc.stack([_ycr_receptor()] * 2))
        result = rq.compute_receptor_quality(path, "A")
        assert result["error_message"] == f"energy of models 1, 2: {NO_OPENMM}"

    def test_a_computed_energy_leaves_the_block_without_the_key(self, monkeypatch):
        monkeypatch.setattr(
            rq,
            "_receptor_energy",
            lambda *a, **k: {
                "energy_kJ_mol": -100.0,
                "energy_per_residue_kJ_mol": -10.0,
                "n_atoms_with_h": 1,
                "error": None,
            },
        )
        result = rq.compute_receptor_quality(YCR, "A")
        assert "error_message" not in result and "reason" not in result


@pytest.fixture(scope="module")
def synthetic_poses():
    return make_synthetic_poses(backends=["a"], seeds=[1, 2]).poses


def _score(poses):
    return score_pose_set(poses, metrics=["receptor_quality"], heavy="all", preflight="off")


class TestPoseSet:
    def test_a_failed_energy_makes_the_pose_partial_with_the_reason(
        self, monkeypatch, synthetic_poses
    ):
        monkeypatch.setattr(rq, "_import_openmm", _no_openmm)
        result = _score(synthetic_poses)
        assert [row["status"] for row in result.rows] == ["partial", "partial"]
        for row in result.rows:
            assert NO_OPENMM in row["reasons"]["receptor_quality"]
            assert row["reason"] is None
            _assert_failed_term(row["receptor_quality"]["models"][0]["energy"])

    def test_the_terms_that_were_computed_stay_in_the_aggregates(
        self, monkeypatch, synthetic_poses
    ):
        monkeypatch.setattr(rq, "_import_openmm", _no_openmm)
        aggregates = _score(synthetic_poses)["aggregates"]["all"]
        for key in ("ramachandran_favoured_pct", "clashscore", "bad_bonds_pct"):
            entry = aggregates[f"receptor_quality.summary.{key}"]
            assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (2, 2, True)
        for key in ("energy_kJ_mol", "energy_per_residue_kJ_mol"):
            entry = aggregates[f"receptor_quality.summary.{key}"]
            assert (entry["n_ok"], entry["n_total"], entry["complete"]) == (0, 2, False)

    def test_a_computed_energy_leaves_the_pose_ok(self, monkeypatch, synthetic_poses):
        monkeypatch.setattr(
            rq,
            "_receptor_energy",
            lambda *a, **k: {
                "energy_kJ_mol": -100.0,
                "energy_per_residue_kJ_mol": -10.0,
                "n_atoms_with_h": 1,
                "error": None,
            },
        )
        result = _score(synthetic_poses)
        assert [row["status"] for row in result.rows] == ["ok", "ok"]
        assert all(row["reasons"] == {} for row in result.rows)
