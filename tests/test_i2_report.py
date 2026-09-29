"""The cyclic topology section of the summary report.

The pipeline stores the closure bonds of a cyclic peptide under
``results["relax"]["peptide_cyclic_bonds"]`` (``RelaxationResult.to_dict``); the
summary used to read a key of another name and never rendered the section.
"""

import json

from binding_metrics.protocols.relaxation import RelaxationResult
from binding_metrics.protocols.report import _build_summary, write_report

# SFTI-1 in 3P8F (chain B, 14 residues) is bicyclic: the head-to-tail amide bond
# and the Cys3-Cys11 disulfide, in the atom format the relaxation records.
SFTI1_CLOSURES = [
    {"type": "head_to_tail", "atom1": "B:14:C", "atom2": "B:1:N"},
    {"type": "disulfide", "atom1": "B:3:SG", "atom2": "B:11:SG"},
]


def _table_rows(summary: str) -> set:
    """Lines of the summary with the column padding collapsed."""
    return {" ".join(line.split()) for line in summary.splitlines()}


def _pipeline_results(bonds) -> dict:
    """Results as the pipeline assembles them: the relax section is ``to_dict()``."""
    relax = RelaxationResult(sample_id="3P8F", success=True, peptide_cyclic_bonds=bonds).to_dict()
    return {"sample_id": "3P8F", "relax": relax}


class TestCyclicTopologySection:
    def test_pipeline_bonds_are_rendered(self):
        summary = _build_summary(_pipeline_results(SFTI1_CLOSURES))

        assert "## Cyclic topology" in summary
        rows = _table_rows(summary)
        assert "| head_to_tail | B:14:C | B:1:N |" in rows
        assert "| disulfide | B:3:SG | B:11:SG |" in rows
        assert "2 closure bond(s) detected" in summary

    def test_section_follows_relaxation_and_precedes_interaction_energy(self):
        results = _pipeline_results(SFTI1_CLOSURES)
        results["energy"] = {"skipped": True}

        summary = _build_summary(results)

        assert (
            summary.index("## Relaxation")
            < summary.index("## Cyclic topology")
            < summary.index("## Interaction Energy")
        )

    def test_a_linear_peptide_has_no_cyclic_section(self):
        assert "## Cyclic topology" not in _build_summary(_pipeline_results(None))
        assert "## Cyclic topology" not in _build_summary(_pipeline_results([]))

    def test_the_older_key_still_renders(self):
        results = {"sample_id": "s", "relax": {"success": True, "cyclic_bonds": SFTI1_CLOSURES}}
        assert "| head_to_tail | B:14:C | B:1:N |" in _table_rows(_build_summary(results))

    def test_written_summary_file_carries_the_section(self, tmp_path):
        results = _pipeline_results(SFTI1_CLOSURES)

        results_path = write_report(results, tmp_path, "3P8F", fmt="json", summary=True)

        assert "## Cyclic topology" in (tmp_path / "3P8F_report.md").read_text()
        assert json.loads(results_path.read_text())["relax"]["peptide_cyclic_bonds"] == (
            SFTI1_CLOSURES
        )
