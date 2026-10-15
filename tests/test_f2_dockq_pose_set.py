"""``dockq`` reads the structure that ``score_pose_set`` writes for the metrics (#184)."""

import shutil

import numpy as np
import pytest

from binding_metrics.metrics._common import load_structure
from binding_metrics.poses import Reference, score_pose_set
from binding_metrics.poses.score import _write_structure
from binding_metrics.testing import bundled_complex_path, make_synthetic_poses

requires_dockq = pytest.mark.skipif(shutil.which("DockQ") is None, reason="DockQ CLI not installed")


def _atom_site(path):
    import biotite.structure.io.pdbx as pdbx

    return pdbx.CIFFile.read(str(path)).block["atom_site"]


def test_an_array_without_b_factor_and_occupancy_is_written_with_neutral_values(tmp_path):
    atoms = load_structure(bundled_complex_path())
    assert not {"b_factor", "occupancy"} & set(atoms.get_annotation_categories())
    target = tmp_path / "pose.cif"

    _write_structure(atoms, target)

    site = _atom_site(target)
    assert set(site["B_iso_or_equiv"].as_array(float)) == {0.0}
    assert set(site["occupancy"].as_array(float)) == {1.0}
    assert "b_factor" not in atoms.get_annotation_categories()  # the caller's array is not changed


def test_the_values_of_the_array_are_written_as_they_are(tmp_path):
    atoms = load_structure(bundled_complex_path())
    atoms.add_annotation("b_factor", dtype=float)
    atoms.add_annotation("occupancy", dtype=float)
    atoms.b_factor[:] = 12.5
    atoms.occupancy[:] = 0.5
    target = tmp_path / "pose.cif"

    _write_structure(atoms, target)

    site = _atom_site(target)
    assert np.allclose(site["B_iso_or_equiv"].as_array(float), 12.5)
    assert np.allclose(site["occupancy"].as_array(float), 0.5)


@pytest.mark.slow
@requires_dockq
def test_dockq_scores_every_pose_of_a_set_against_a_reference():
    synthetic = make_synthetic_poses()
    reference = Reference(bundled_complex_path(), receptor_chain="A", binder_chain="B")

    result = score_pose_set(
        synthetic.poses, "B", "A", metrics=["dockq"], reference=reference, preflight="skip"
    )

    assert len(result["poses"]) == len(synthetic.poses)
    for row in result["poses"]:
        assert row["status"] == "ok", row["reasons"]
        assert 0.0 <= row["dockq"]["dockq"] <= 1.0
        assert row["dockq"]["n_interfaces"] == 1
