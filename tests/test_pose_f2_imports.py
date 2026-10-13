"""The pose-set layer works without OpenMM, MDTraj, simtk and PDBFixer (requirement F2).

The static metrics read structures with biotite and need none of the simulation stack, so an
engine that installs only the ``static`` extra must be able to import ``binding_metrics.poses``
and score a set of poses. Each test runs a fresh interpreter in which the four packages are hidden
by an import blocker (design 9.1): a wrapper around every finder on ``sys.meta_path`` that answers
"not found" for those names, so ``import openmm`` raises ``ModuleNotFoundError`` and
``importlib.util.find_spec("openmm")`` is ``None``, exactly as when the package is not installed.
The blocker records every attempt.

On the static path (importing the layer, scoring with the static metrics, a reference, the
descriptors and the writers) the tests assert that nothing raises, that no blocked module is
imported (``sys.modules``) and that none is even looked for: ``attempts`` is empty. A guarded
``try: import openmm`` would load OpenMM, about a second and its plugins, when it is installed.
A metric that needs OpenMM is the one place where an attempt is expected.
"""

import importlib.metadata as metadata
import json
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("biotite.structure")

BLOCKER = """
import importlib.util, json, sys

BLOCKED = ("openmm", "simtk", "mdtraj", "pdbfixer")
attempts = []


class Hide:
    def __init__(self, finder):
        self._finder = finder

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            attempts.append(name)
            return None
        return self._finder.find_spec(name, path, target)

    def __getattr__(self, attribute):
        return getattr(self._finder, attribute)


sys.meta_path[:] = [Hide(finder) for finder in sys.meta_path]


def report(**values):
    imported = [name for name in sorted(sys.modules) if name.split(".")[0] in BLOCKED]
    values.update(imported=imported, attempts=sorted(set(attempts)))
    print("REPORT " + json.dumps(values))
"""


def run_blocked(code: str) -> dict:
    """Run ``code`` with the four packages hidden and return what it passed to ``report``."""
    completed = subprocess.run(
        [sys.executable, "-c", BLOCKER + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = [line for line in completed.stdout.splitlines() if line.startswith("REPORT ")]
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0][len("REPORT ") :])


def test_the_blocker_hides_the_four_packages_like_missing_ones():
    out = run_blocked(
        """
        outcome = {}
        for name in BLOCKED:
            try:
                __import__(name)
                outcome[name] = "imported"
            except ModuleNotFoundError:
                outcome[name] = "not found"
            outcome[name + " spec"] = importlib.util.find_spec(name)
        outcome["json"] = importlib.util.find_spec("json") is not None
        report(outcome=outcome)
        """
    )
    assert out["outcome"] == {
        "openmm": "not found",
        "openmm spec": None,
        "simtk": "not found",
        "simtk spec": None,
        "mdtraj": "not found",
        "mdtraj spec": None,
        "pdbfixer": "not found",
        "pdbfixer spec": None,
        "json": True,
    }
    assert out["imported"] == []


def test_importing_the_pose_layer_and_the_fixtures_needs_none_of_them_and_does_not_look():
    out = run_blocked(
        """
        import binding_metrics.poses as poses
        import binding_metrics.testing as testing

        resolved = [name for name in poses.__all__ if getattr(poses, name) is not None]
        fixtures = [name for name in testing.__all__ if getattr(testing, name) is not None]
        report(n_names=len(resolved), n_all=len(poses.__all__), fixtures=fixtures)
        """
    )
    assert out["n_names"] == out["n_all"] > 50
    assert sorted(out["fixtures"]) == [
        "SyntheticPoses",
        "SyntheticTruth",
        "bundled_complex_path",
        "make_synthetic_poses",
    ]
    assert out["imported"] == []
    assert out["attempts"] == []


@pytest.mark.slow
def test_a_pose_set_call_with_every_static_metric_a_reference_and_the_writers_works(tmp_path):
    out = run_blocked(
        f"""
        from pathlib import Path

        from binding_metrics.poses import (
            Partner,
            Reference,
            ResidueKey,
            score_pose_set,
            write_pose_report,
        )
        from binding_metrics.testing import make_synthetic_poses

        synthetic = make_synthetic_poses(backends=("a", "b"), seeds=(1,))
        reference = Reference(
            synthetic.reference,
            "A",
            "B",
            residue_sets={{"pocket": [ResidueKey("A", n) for n in (25, 26, 50, 51, 54)]}},
            partners={{"crystal": Partner(synthetic.reference, chains=["B"])}},
        )
        metrics = (
            "interface", "delta_sasa_static", "hbonds", "saltbridges", "coulomb", "ramachandran",
            "omega", "shape_complementarity", "void_volume", "structure_rmsd",
        )
        result = score_pose_set(
            synthetic.poses, "B", "A", metrics=metrics, reference=reference, residue_table=True
        )
        written = [
            write_pose_report(result, fmt, Path({str(tmp_path)!r}) / ("report." + fmt)).name
            for fmt in ("json", "csv", "md")
        ]
        row = result["poses"][0]
        report(
            statuses=[r["status"] for r in result["poses"]],
            reasons=[r["reasons"] for r in result["poses"]],
            blocks=[name for name in metrics if name in row],
            descriptors=sorted(row["descriptors"]["reference"]),
            table_rows=len(row["residue_table"]),
            representative=result["representatives"]["all"],
            n_flat_rows=len(result.to_flat_rows()),
            summary_lines=len(result.summary().splitlines()),
            written=written,
        )
        """
    )
    assert out["statuses"] == ["ok", "ok"] and out["reasons"] == [{}, {}]
    assert len(out["blocks"]) == 10
    assert out["descriptors"] == ["partners", "sets"]
    assert out["table_rows"] > 13
    assert out["representative"] in ("a-s1", "b-s1")
    assert out["n_flat_rows"] == 2 and out["summary_lines"] > 10
    assert out["written"] == ["report.json", "report.csv", "report.md"]
    assert out["imported"] == []
    assert out["attempts"] == []
    assert (tmp_path / "report.json").stat().st_size > 0


@pytest.mark.slow
def test_a_metric_that_needs_openmm_says_so_and_leaves_the_rest_of_the_result_intact():
    out = run_blocked(
        """
        from binding_metrics.poses import score_pose_set
        from binding_metrics.testing import make_synthetic_poses

        synthetic = make_synthetic_poses(backends=("a", "b"), seeds=(1,))
        score_pose_set(synthetic.poses, "B", "A", metrics=["delta_sasa_static"])
        attempts_static = sorted(set(attempts))
        result = score_pose_set(
            synthetic.poses, "B", "A", metrics=["delta_sasa_static", "structure_interaction_energy"]
        )
        report(
            attempts_static=attempts_static,
            statuses=[row["status"] for row in result["poses"]],
            reasons=[row["reasons"].get("structure_interaction_energy") for row in result["poses"]],
            has_area=[
                "delta_sasa" in row["delta_sasa_static"] for row in result["poses"]
            ],
            n_ok=result["aggregates"]["a"]["delta_sasa_static.delta_sasa"]["n_ok"],
        )
        """
    )
    assert out["attempts_static"] == []
    asked = [reason for reason in out["reasons"] if reason is not None]
    assert asked, "the energy metric should have run on a representative and said why it failed"
    assert all("OpenMM" in reason for reason in asked)
    assert out["has_area"] == [True, True] and out["n_ok"] == 1
    assert "partial" in out["statuses"]
    assert out["imported"] == []
    assert "openmm" in out["attempts"]  # the metric that needs it is where it is looked for


def test_the_provenance_of_a_result_reads_the_openmm_version_without_importing_it():
    out = run_blocked(
        """
        from binding_metrics.provenance import collect_provenance

        report(openmm_version=collect_provenance()["openmm_version"])
        """
    )
    try:
        installed = metadata.version("openmm")  # the blocker hides imports, not the metadata
    except metadata.PackageNotFoundError:
        installed = None
    assert out["openmm_version"] == installed
    assert out["imported"] == [] and out["attempts"] == []
