"""Tests for binding-metrics-batch helpers.

Focus on the DockQ reference-matching logic (input sample → native structure by
filename stem), which is pure and testable without running the pipeline.
"""

from pathlib import Path

from binding_metrics.cli.batch import _build_reference_map, _run_one

EXAMPLE_1YCR = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


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


def _worker_kwargs(tmp_path, **overrides):
    """Arguments for ``_run_one`` that reach the pipeline's chain check and stop."""
    kwargs = dict(
        input_path=EXAMPLE_1YCR,
        output_dir=tmp_path,
        sample_id=None,
        skip_prep=True,
        ph=7.4,
        keep_water=False,
        canonicalize=False,
        skip_relax=True,
        md_duration_ps=0.0,
        device="cuda",
        peptide_chain=None,
        receptor_chain=None,
        metrics=frozenset(),
        energy_modes=("relaxed",),
        openfold_mode="score",
        openfold_conda_env=None,
        log_file=None,
    )
    kwargs.update(overrides)
    return kwargs


class TestRunOneChains:
    def test_unknown_chain_marks_the_sample_as_error(self, tmp_path):
        row = _run_one(**_worker_kwargs(tmp_path, peptide_chain="Z"))
        assert row["batch_status"] == "error"
        assert "chain 'Z' not found; available: B (13), A (85)" in row["batch_error"]

    def test_valid_chains_run_to_completion(self, tmp_path):
        row = _run_one(**_worker_kwargs(tmp_path, peptide_chain="B", receptor_chain="A"))
        assert row["batch_status"] == "ok"
        assert row["sample_id"] == "example_linear_p53_1YCR"
