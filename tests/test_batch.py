"""Tests for binding-metrics-batch helpers.

Focus on the DockQ reference-matching logic (input sample → native structure by
filename stem), which is pure and testable without running the pipeline.
"""

from pathlib import Path

from binding_metrics.cli.batch import _build_reference_map


def _touch(path: Path) -> Path:
    path.write_text("REMARK dummy\n")
    return path


class TestBuildReferenceMap:
    def test_matches_structures_by_stem(self, tmp_path):
        _touch(tmp_path / "target1.pdb")
        _touch(tmp_path / "target2.cif")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"target1", "target2"}
        assert refs["target1"].name == "target1.pdb"
        assert refs["target2"].name == "target2.cif"

    def test_ignores_non_structure_files(self, tmp_path):
        _touch(tmp_path / "target1.pdb")
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / "scores.csv")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"target1"}

    def test_ignores_subdirectories(self, tmp_path):
        _touch(tmp_path / "target1.pdb")
        (tmp_path / "subdir").mkdir()
        _touch(tmp_path / "subdir" / "target2.pdb")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"target1"}

    def test_duplicate_stem_is_deterministic(self, tmp_path):
        # Two references share the stem 'target1'; .cif sorts before .pdb, so it wins.
        _touch(tmp_path / "target1.pdb")
        _touch(tmp_path / "target1.cif")
        refs = _build_reference_map(tmp_path)
        assert refs["target1"].suffix == ".cif"

    def test_suffix_matching_is_case_insensitive(self, tmp_path):
        _touch(tmp_path / "TARGET.PDB")
        refs = _build_reference_map(tmp_path)
        assert set(refs) == {"TARGET"}

    def test_empty_dir_gives_empty_map(self, tmp_path):
        assert _build_reference_map(tmp_path) == {}

    def test_resolution_against_input_stems(self, tmp_path):
        """A sample is matched iff a reference shares its stem."""
        _touch(tmp_path / "sampleA.pdb")
        _touch(tmp_path / "sampleC.pdb")
        refs = _build_reference_map(tmp_path)
        inputs = [Path("in/sampleA.cif"), Path("in/sampleB.cif"), Path("in/sampleC.cif")]
        resolved = {p.stem: refs.get(p.stem) for p in inputs}
        assert resolved["sampleA"] is not None
        assert resolved["sampleB"] is None  # no matching reference
        assert resolved["sampleC"] is not None
