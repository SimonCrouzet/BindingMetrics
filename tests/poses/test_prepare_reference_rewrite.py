"""``prepare_reference`` follows a rewrite of the reference structure file.

The prepared reference (receptor table, binding site) is cached on the ``Reference``; the entry
is checked against the content hash of the structure, which follows the size and modification
time of the file, so a reference file rewritten at the same path is read again.
"""

import os
import pickle

import numpy as np
import pytest

from binding_metrics.poses import Reference
from binding_metrics.poses import pose as pose_module
from binding_metrics.poses.frame import prepare_reference
from tests.predictors import synth

SHIFT_ANGSTROM = 50.0


@pytest.fixture
def reference_file(tmp_path):
    return synth.write_structure(synth.synthetic_complex().atoms, tmp_path / "ref.cif")


def _rewrite_binder_far_away(path):
    """Write the complex again with receptor shifted along x and binder moved off the receptor."""
    before = path.stat()
    atoms = synth.synthetic_complex().atoms.copy()
    atoms.coord[atoms.chain_id == "A", 0] += SHIFT_ANGSTROM
    atoms.coord[atoms.chain_id == "B", 1] += SHIFT_ANGSTROM
    synth.write_structure(atoms, path)
    stamp = before.st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


class TestARewrittenReferenceIsPreparedAgain:
    def test_the_table_and_the_site_are_those_of_the_new_file(self, reference_file):
        reference = Reference(reference_file, receptor_chain="A", binder_chain="B")
        first = prepare_reference(reference)
        assert len(first.site) > 0
        x_before = first.table.ca[:, 0].copy()

        _rewrite_binder_far_away(reference_file)
        second = prepare_reference(reference)

        assert second is not first
        assert second.site == ()
        np.testing.assert_allclose(second.table.ca[:, 0], x_before + SHIFT_ANGSTROM, atol=2e-3)

    def test_every_cutoff_is_checked_on_its_own(self, reference_file):
        reference = Reference(reference_file, receptor_chain="A", binder_chain="B")
        tight = prepare_reference(reference, site_cutoff_angstrom=3.5)
        wide = prepare_reference(reference, site_cutoff_angstrom=4.5)
        assert len(tight.site) < len(wide.site)

        _rewrite_binder_far_away(reference_file)

        assert prepare_reference(reference, site_cutoff_angstrom=3.5) is not tight
        assert prepare_reference(reference, site_cutoff_angstrom=4.5) is not wide
        assert prepare_reference(reference, site_cutoff_angstrom=4.5).site == ()

    def test_the_content_hash_of_the_reference_follows_the_file(self, reference_file):
        reference = Reference(reference_file, receptor_chain="A", binder_chain="B")
        before = reference.content_hash()
        _rewrite_binder_far_away(reference_file)
        assert reference.content_hash() != before


class TestAnUnchangedReferenceIsPreparedOnce:
    def test_the_prepared_reference_is_the_same_object(self, reference_file):
        reference = Reference(reference_file, receptor_chain="A", binder_chain="B")
        first = prepare_reference(reference)
        assert prepare_reference(reference) is first

    def test_a_cached_call_costs_one_stat_of_the_structure_and_no_read(
        self, reference_file, monkeypatch
    ):
        reference = Reference(reference_file, receptor_chain="A", binder_chain="B")
        prepare_reference(reference)
        stats, reads = [], []
        real_stat = os.stat

        def counting_stat(path, *args, **kwargs):
            if os.fspath(path) == os.fspath(reference_file):
                stats.append(path)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", counting_stat)
        monkeypatch.setattr(pose_module, "file_sha256", lambda path: reads.append(path))
        monkeypatch.setattr(pose_module, "_read_structure", lambda path: reads.append(path))
        for _ in range(3):
            prepare_reference(reference)
        assert len(stats) == 3
        assert reads == []

    def test_an_in_memory_reference_is_not_stat_ed(self, monkeypatch):
        reference = Reference(synth.synthetic_complex().atoms, "A", "B")
        first = prepare_reference(reference)
        with monkeypatch.context() as patch:
            patch.setattr(os, "stat", lambda *a, **k: pytest.fail("stat called"))
            assert prepare_reference(reference) is first


class TestThePickleStillDropsTheCache:
    def test_a_copy_prepares_from_the_current_file(self, reference_file):
        reference = Reference(reference_file, receptor_chain="A", binder_chain="B")
        prepare_reference(reference)
        assert "_prepared_cache" in reference.__dict__

        clone = pickle.loads(pickle.dumps(reference))
        assert "_prepared_cache" not in clone.__dict__
        assert len(prepare_reference(clone).site) > 0

        _rewrite_binder_far_away(reference_file)
        assert prepare_reference(clone).site == ()
        assert prepare_reference(reference).site == ()
