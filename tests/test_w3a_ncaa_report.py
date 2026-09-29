"""Which non-canonical residues fell back to single bonds shows up in the results (issue #33).

The GAFF template step records ``bond_order_source_by_residue`` (``"ccd"`` or
``"single_bonds"``); the relaxation result and the prep report now carry it.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from binding_metrics.cli.run import run_pipeline  # noqa: E402
from binding_metrics.core import gaff_ncaa  # noqa: E402
from binding_metrics.core.gaff_ncaa import (  # noqa: E402
    BOND_ORDER_SOURCE_CCD,
    BOND_ORDER_SOURCE_SINGLE_BONDS,
    NcaaTemplateList,
)
from binding_metrics.protocols.relaxation import (  # noqa: E402
    ImplicitRelaxation,
    RelaxationConfig,
    RelaxationResult,
)
from binding_metrics.protocols.report import _flatten, write_report  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
EXAMPLE_1YCR = DATA / "example_linear_p53_1YCR.pdb"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
SOMATOSTATIN = DATA / "example_lactam_somatostatin_1XY4.cif"

SOURCES = {"BMT": BOND_ORDER_SOURCE_CCD, "XYZ": BOND_ORDER_SOURCE_SINGLE_BONDS}


def _template_list(sources):
    templates = NcaaTemplateList()
    templates.bond_order_source_by_residue.update(sources)
    return templates


class TestResultField:
    def test_default_is_empty_and_serialises(self):
        row = RelaxationResult(sample_id="x", success=True).to_dict()
        assert row["ncaa_bond_order_source"] == {}
        assert not [key for key in _flatten({"relax": row}) if "ncaa" in key]

    def test_to_dict_carries_a_copy_of_the_mapping(self):
        result = RelaxationResult(sample_id="x", success=True, ncaa_bond_order_source=dict(SOURCES))
        row = result.to_dict()
        assert row["ncaa_bond_order_source"] == SOURCES
        row["ncaa_bond_order_source"]["BMT"] = "changed"
        assert result.ncaa_bond_order_source["BMT"] == BOND_ORDER_SOURCE_CCD

    def test_json_report_and_csv_columns(self, tmp_path):
        row = RelaxationResult(sample_id="x", success=True, ncaa_bond_order_source=SOURCES)
        results = {"sample_id": "x", "relax": row.to_dict()}
        path = write_report(results, tmp_path, "x", fmt="json")
        assert (
            json.loads(path.read_text(encoding="utf-8"))["relax"]["ncaa_bond_order_source"]
            == SOURCES
        )
        flat = _flatten(results)
        assert flat["relax_ncaa_bond_order_source_XYZ"] == "single_bonds"
        assert flat["relax_ncaa_bond_order_source_BMT"] == "ccd"


class TestRelaxationRecordsTheSources:
    @pytest.fixture
    def stub_template_step(self, monkeypatch):
        """1YCR has no exotic residue, so the template step is stubbed to report some."""

        def fake(topology, positions, ff, **kwargs):
            return topology, positions, _template_list(SOURCES)

        monkeypatch.setattr(gaff_ncaa, "parameterize_ncaa_residues", fake)

    @pytest.mark.integration
    def test_run_puts_the_mapping_in_the_result(
        self, tmp_path, prepped_example_cif, stub_template_step
    ):
        config = RelaxationConfig(
            md_duration_ps=0.0,
            min_steps_initial=5,
            min_steps_restrained=5,
            min_steps_final=5,
            small_molecules="auto",
        )
        result = ImplicitRelaxation(config).run(prepped_example_cif, tmp_path / "out")
        assert result.success, result.error_message
        assert result.ncaa_bond_order_source == SOURCES
        assert result.to_dict()["ncaa_bond_order_source"] == SOURCES

    @pytest.mark.integration
    def test_without_the_auto_route_the_mapping_stays_empty(self, tmp_path, prepped_example_cif):
        config = RelaxationConfig(
            md_duration_ps=0.0, min_steps_initial=5, min_steps_restrained=5, min_steps_final=5
        )
        result = ImplicitRelaxation(config).run(prepped_example_cif, tmp_path / "out")
        assert result.success, result.error_message
        assert result.ncaa_bond_order_source == {}


class TestPrepReport:
    def test_cyclic_prep_records_the_sources(self, monkeypatch):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        real = gaff_ncaa.parameterize_ncaa_residues

        def with_sources(topology, positions, ff, **kwargs):
            topology, positions, templates = real(topology, positions, ff, **kwargs)
            templates.bond_order_source_by_residue.update(SOURCES)
            return topology, positions, templates

        monkeypatch.setattr(gaff_ncaa, "parameterize_ncaa_residues", with_sources)
        topology, positions = load_structure(SFTI1)
        report = {}
        prep_structure(topology, positions, report=report)
        assert report["ncaa_bond_order_source"] == SOURCES

    def test_linear_prep_has_no_such_key(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(EXAMPLE_1YCR)
        report = {}
        prep_structure(topology, positions, report=report)
        assert "ncaa_bond_order_source" not in report

    @pytest.mark.integration
    def test_somatostatin_iam_takes_its_bond_orders_from_the_dictionary(self):
        """1XY4: IAM matches its dictionary entry once a bond listed twice is capped once."""
        pytest.importorskip("pdbfixer")
        pytest.importorskip("openmmforcefields")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(SOMATOSTATIN)
        report = {}
        prep_structure(topology, positions, report=report)
        assert report["ncaa_bond_order_source"] == {"IAM": "ccd"}


class TestPipelineResults:
    def test_prep_and_relax_sections_carry_the_mapping(self, tmp_path, monkeypatch):
        from binding_metrics.core import system
        from binding_metrics.protocols.relaxer import Relaxer

        def fake_prep(topology, positions, report=None, **kwargs):
            report["ncaa_bond_order_source"] = dict(SOURCES)
            return topology, positions

        class StubRelaxer(Relaxer):
            def run(self, input_path, output_dir, sample_id=None):
                return RelaxationResult(
                    sample_id=sample_id,
                    success=True,
                    minimized_structure_path=str(input_path),
                    ncaa_bond_order_source=dict(SOURCES),
                )

        monkeypatch.setattr(system, "prep_structure", fake_prep)
        results = run_pipeline(EXAMPLE_1YCR, tmp_path, relaxer=StubRelaxer(), metrics=frozenset())
        assert results["prep"]["ncaa_bond_order_source"] == SOURCES
        assert results["relax"]["ncaa_bond_order_source"] == SOURCES
