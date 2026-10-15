"""A pose is a view of its structure file as it is when read: a rewrite is read again.

The cache of ``Pose.atoms()`` and ``Pose.content_hash()`` is keyed by the file's size and
modification time as well as by its path, so a pose that outlives a rewrite of its file at
the same path (a ``--rerun-predictions`` run in one process, a relaxed structure written back
over its input) does not return the values of the first read.
"""

import os
import threading

import numpy as np
import pytest

from binding_metrics.poses import DockedPose, ExperimentalPose
from binding_metrics.poses import pose as pose_module
from binding_metrics.predictors.record import PredictedPose
from binding_metrics.predictors.store import file_sha256
from tests.predictors import synth

SHIFT_ANGSTROM = 5.0
ONE_SECOND_NS = 1_000_000_000


def _make_pose(kind, path):
    if kind == "docked":
        return DockedPose.from_file(path)
    if kind == "experimental":
        return ExperimentalPose.from_file(path)
    return PredictedPose("of3", "q", structure_path=path)


KINDS = ["docked", "experimental", "predicted"]


@pytest.fixture(params=["cif", "pdb", "pdb.gz"])
def structure_path(request, tmp_path):
    return synth.write_structure(synth.synthetic_complex().atoms, tmp_path / f"s.{request.param}")


def _rewrite_shifted(path, *, atoms=None, new_mtime=True):
    """Write the synthetic complex shifted along x over ``path``; the mtime moves on by 1 s.

    A coarse file system (or a fast test) can give a rewrite the old timestamp, so the new
    one is set explicitly: a test then depends on the cache key and not on the clock.
    """
    before = path.stat()
    atoms = (synth.synthetic_complex().atoms if atoms is None else atoms).copy()
    atoms.coord[:, 0] += SHIFT_ANGSTROM
    synth.write_structure(atoms, path)
    stamp = before.st_mtime_ns + (ONE_SECOND_NS if new_mtime else 0)
    os.utime(path, ns=(stamp, stamp))
    return atoms


@pytest.mark.parametrize("kind", KINDS)
class TestARewrittenFileIsReadAgain:
    def test_atoms_return_the_coordinates_of_the_new_file(self, kind, structure_path):
        pose = _make_pose(kind, structure_path)
        first = pose.atoms().coord.copy()
        shifted = _rewrite_shifted(structure_path)
        second = pose.atoms().coord
        np.testing.assert_allclose(second[:, 0], first[:, 0] + SHIFT_ANGSTROM, atol=2e-3)
        np.testing.assert_allclose(second[:, 0], shifted.coord[:, 0], atol=2e-3)

    def test_the_content_hash_is_that_of_the_new_file(self, kind, structure_path):
        pose = _make_pose(kind, structure_path)
        before = pose.content_hash()
        _rewrite_shifted(structure_path)
        after = pose.content_hash()
        assert after != before
        assert after == file_sha256(structure_path)

    def test_a_rewrite_of_another_size_is_seen_under_the_old_modification_time(
        self, kind, structure_path
    ):
        pose = _make_pose(kind, structure_path)
        n_before = pose.atoms().array_length()
        before = pose.content_hash()
        stat = structure_path.stat()
        smaller = synth.synthetic_complex().atoms[:-3]
        synth.write_structure(smaller, structure_path)
        os.utime(structure_path, ns=(stat.st_mtime_ns, stat.st_mtime_ns))
        assert structure_path.stat().st_size != stat.st_size
        assert pose.atoms().array_length() == n_before - 3
        assert pose.content_hash() != before

    def test_the_chain_map_still_applies_to_the_new_read(self, kind, structure_path):
        pose = _make_pose(kind, structure_path)
        chains = sorted(set(pose.atoms().chain_id.tolist()))
        renamed = _make_pose_with_map(kind, structure_path, {chains[0]: "Z"})
        assert "Z" in set(renamed.atoms().chain_id.tolist())
        _rewrite_shifted(structure_path)
        assert "Z" in set(renamed.atoms().chain_id.tolist())


def _make_pose_with_map(kind, path, chain_map):
    if kind == "docked":
        return DockedPose.from_file(path, chain_map=chain_map)
    if kind == "experimental":
        return ExperimentalPose.from_file(path, chain_map=chain_map)
    return PredictedPose("of3", "q", structure_path=path, chain_map=chain_map)


@pytest.mark.parametrize("kind", KINDS)
class TestAnUnchangedFileIsReadOnce:
    def test_the_atoms_are_the_same_object_and_the_hash_the_same_string(self, kind, structure_path):
        pose = _make_pose(kind, structure_path)
        atoms = pose.atoms()
        digest = pose.content_hash()
        assert pose.atoms() is atoms
        assert pose.content_hash() == digest == file_sha256(structure_path)

    def test_a_touch_without_a_change_reads_again_to_the_same_values(self, kind, structure_path):
        pose = _make_pose(kind, structure_path)
        atoms = pose.atoms()
        digest = pose.content_hash()
        stat = structure_path.stat()
        later = stat.st_mtime_ns + ONE_SECOND_NS
        os.utime(structure_path, ns=(later, later))
        np.testing.assert_array_equal(pose.atoms().coord, atoms.coord)
        assert pose.content_hash() == digest

    def test_a_hot_call_costs_one_stat_and_no_read(self, kind, structure_path, monkeypatch):
        pose = _make_pose(kind, structure_path)
        pose.atoms()
        pose.content_hash()
        stats, opens = [], []
        real_stat = os.stat

        def counting_stat(path, *args, **kwargs):
            if os.fspath(path) == os.fspath(structure_path):
                stats.append(path)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", counting_stat)
        monkeypatch.setattr(pose_module, "file_sha256", lambda path: opens.append(path))
        monkeypatch.setattr(pose_module, "_read_structure", lambda path: opens.append(path))
        for _ in range(3):
            pose.atoms()
        assert len(stats) == 3
        stats.clear()
        for _ in range(3):
            pose.content_hash()
        assert len(stats) == 3
        assert opens == []


class TestTheStampIsTakenBeforeTheRead:
    def test_a_rewrite_during_a_read_is_picked_up_by_the_next_call(self, tmp_path, monkeypatch):
        """The cache never pairs the new stamp with the old content."""
        path = synth.write_structure(synth.synthetic_complex().atoms, tmp_path / "s.cif")
        pose = DockedPose.from_file(path)
        real_read = pose_module._read_structure
        rewritten = threading.Event()

        def read_then_get_rewritten(target):
            atoms = real_read(target)
            if not rewritten.is_set():
                rewritten.set()
                _rewrite_shifted(path)
            return atoms

        monkeypatch.setattr(pose_module, "_read_structure", read_then_get_rewritten)
        stale = pose.atoms().coord[:, 0].copy()
        fresh = pose.atoms().coord[:, 0]
        np.testing.assert_allclose(fresh, stale + SHIFT_ANGSTROM, atol=2e-3)


class TestAnInMemoryStructureIsNotAFile:
    def test_atoms_and_hash_are_cached_without_any_stat(self, monkeypatch):
        atoms = synth.synthetic_complex().atoms
        pose = DockedPose.from_atoms(atoms)
        digest = pose.content_hash()
        first = pose.atoms()
        with monkeypatch.context() as patch:
            patch.setattr(os, "stat", lambda *a, **k: pytest.fail("stat called"))
            assert pose.content_hash() == digest
            assert pose.atoms() is first
