"""The interaction energy of a pose without terminal oxygens, in ``score_pose_set`` (#185).

The energy function refuses a chain that ends without OXT unless it is asked to complete it,
and the pose layer asks. The calls run on the CPU platform: they check that the energy is
obtained, not how fast.
"""

import math
import os
from pathlib import Path

import pytest
from conftest import DATA_DIR

from binding_metrics.metrics.energy import compute_interaction_energy
from binding_metrics.poses import ExperimentalPose, Reference, score_pose_set

pytest.importorskip("openmm")
pytestmark = pytest.mark.integration

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
ENERGY = "structure_interaction_energy"
CPU = {"modes": ("raw",), "device": "cpu"}


def _without_oxt(tmp_path: Path) -> Path:
    """1YCR with the OXT of chain B removed; chain A of the file ends without one as well."""
    lines = YCR.read_text(encoding="utf-8").splitlines(keepends=True)
    path = tmp_path / "ycr_no_oxt.pdb"
    path.write_text("".join(line for line in lines if " OXT " not in line), encoding="utf-8")
    return path


def _with_oxt_on_chain_a(tmp_path: Path) -> Path:
    from openmm.app import PDBFile

    from binding_metrics.core.system import add_terminal_oxygens
    from binding_metrics.io.structures import load_structure

    topology, positions = load_structure(YCR)
    topology, positions = add_terminal_oxygens(topology, positions)
    path = tmp_path / "ycr_oxt.pdb"
    with open(path, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)
    return path


def _score(path, **options):
    pose = ExperimentalPose.from_file(path, id="ycr")
    result = score_pose_set(
        [pose], "B", "A", metrics=[ENERGY], heavy="all", metric_kwargs={ENERGY: CPU}, **options
    )
    return result["poses"][0], result


class TestEnergyFunction:
    def test_a_chain_without_its_terminal_oxygen_is_refused_by_default(self):
        result = compute_interaction_energy(YCR, "B", "A", **CPU)

        assert result["success"] is False
        assert "chain A ends in VAL109 without its terminal oxygen (OXT)" in result["error_message"]
        assert "terminal_oxygens_added" not in result

    def test_the_chains_that_lack_one_get_it_on_request(self):
        result = compute_interaction_energy(YCR, "B", "A", complete_terminal_oxygens=True, **CPU)

        assert result["success"], result["error_message"]
        assert result["terminal_oxygens_added"] == ["A:VAL109"]
        assert math.isfinite(result["raw_interaction_energy"])
        assert result["raw_interaction_energy"] < 0  # the peptide binds the receptor

    def test_both_chains_are_completed_when_both_lack_one(self, tmp_path):
        path = _without_oxt(tmp_path)

        result = compute_interaction_energy(path, "B", "A", complete_terminal_oxygens=True, **CPU)

        assert result["success"], result["error_message"]
        assert result["terminal_oxygens_added"] == ["A:VAL109", "B:ASN29"]

    def test_a_chain_that_has_its_oxygen_is_left_as_it_is(self, tmp_path):
        closed = _with_oxt_on_chain_a(tmp_path)

        plain = compute_interaction_energy(closed, "B", "A", **CPU)
        asked = compute_interaction_energy(closed, "B", "A", complete_terminal_oxygens=True, **CPU)

        assert plain["success"], plain["error_message"]
        assert asked["terminal_oxygens_added"] == []
        # The CPU platform sums on several threads in no fixed order: repeated calls differ by up
        # to about a kJ/mol; an added or removed oxygen moves the energy by more.
        assert asked["raw_interaction_energy"] == pytest.approx(
            plain["raw_interaction_energy"], abs=1.5
        )


class TestPoseSet:
    def test_the_energy_runs_on_a_pose_without_terminal_oxygens(self, tmp_path):
        row, _ = _score(_without_oxt(tmp_path))

        assert row["status"] == "ok", row["reasons"]
        block = row[ENERGY]
        assert block["success"], block["error_message"]
        assert block["terminal_oxygens_added"] == ["A:VAL109", "B:ASN29"]
        assert block["raw_interaction_energy"] < 0

    def test_the_energy_of_the_completed_pose_is_the_energy_of_the_closed_file(self, tmp_path):
        row, _ = _score(YCR)
        direct = compute_interaction_energy(_with_oxt_on_chain_a(tmp_path), "B", "A", **CPU)

        assert row[ENERGY]["terminal_oxygens_added"] == ["A:VAL109"]
        assert row[ENERGY]["raw_interaction_energy"] == pytest.approx(
            direct["raw_interaction_energy"], abs=1.0
        )

    def test_the_other_metrics_read_the_placed_file_without_the_oxygens(self, tmp_path):
        pose = ExperimentalPose.from_file(_without_oxt(tmp_path), id="ycr")
        result = score_pose_set(
            [pose],
            "B",
            "A",
            metrics=["interface", ENERGY],
            heavy="all",
            metric_kwargs={ENERGY: CPU},
            keep_tmp=True,
            tmp_dir=tmp_path,
        )

        placed = Path(result["provenance"]["temp_dir"]) / "pose_0000.cif"
        assert " OXT " not in placed.read_text(encoding="utf-8")
        assert result["poses"][0][ENERGY]["terminal_oxygens_added"] == ["A:VAL109", "B:ASN29"]

    def test_metric_kwargs_can_turn_the_completion_off(self, tmp_path):
        pose = ExperimentalPose.from_file(YCR, id="ycr")
        result = score_pose_set(
            [pose],
            "B",
            "A",
            metrics=[ENERGY],
            heavy="all",
            metric_kwargs={ENERGY: {**CPU, "complete_terminal_oxygens": False}},
        )

        row = result["poses"][0]
        assert row["status"] == "partial"
        assert "without its terminal oxygen" in row["reasons"][ENERGY]


SAVED = os.environ.get("BM_SAVED_PREDICTIONS")


@pytest.mark.skipif(
    not SAVED, reason="BM_SAVED_PREDICTIONS names no directory of saved predictions"
)
@pytest.mark.parametrize(
    "pattern",
    [
        "boltz2/*/*/outputs/boltz_results_ycr_predict_s42/predictions/ycr_predict_s42/"
        "ycr_predict_s42_model_0.cif",
        "of3/*/*/outputs/predictions/server_seed42/seed_42/server_seed42_seed_42_sample_1_model.cif",
    ],
    ids=["boltz2", "openfold3"],
)
def test_the_energy_of_a_saved_prediction_of_1ycr(pattern):
    found = sorted(Path(SAVED).glob(pattern))
    if not found:
        pytest.skip(f"no saved prediction matches {pattern}")
    assert " OXT " not in found[0].read_text(encoding="utf-8")
    pose = ExperimentalPose.from_file(found[0], id="prediction")
    reference = Reference(YCR, receptor_chain="A", binder_chain="B")

    result = score_pose_set(
        [pose],
        "B",
        "A",
        metrics=[ENERGY],
        reference=reference,
        heavy="all",
        metric_kwargs={ENERGY: CPU},
    )

    row = result["poses"][0]
    assert row["status"] == "ok", row["reasons"]
    assert row[ENERGY]["terminal_oxygens_added"] == ["A:VAL109", "B:ASN29"]
    assert math.isfinite(row[ENERGY]["raw_interaction_energy"])
