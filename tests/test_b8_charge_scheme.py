"""The charge scheme of the non-canonical residues is part of the results (#61).

The GAFF template step records how the capped molecule of each residue ended at its
external bonds, with the version of the scheme (``charge_scheme_by_residue``, for example
``"ace-nme-caps-v2"``). Beside ``ncaa_bond_order_source``, the prep report and the
relaxation result carry it as ``ncaa_charge_scheme``: a key inside those blocks, not a
new top-level key, and absent when no residue was parameterised this way.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from conftest import DATA_DIR

from binding_metrics.cli.run import run_pipeline
from binding_metrics.core import gaff_ncaa
from binding_metrics.core.gaff_ncaa import NcaaTemplateList
from binding_metrics.protocols.relaxation import (
    ImplicitRelaxation,
    RelaxationConfig,
    RelaxationResult,
)
from binding_metrics.protocols.report import _flatten, write_report

EXAMPLE_1YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
SFTI1 = DATA_DIR / "example_bicyclic_sfti1_3P8F.cif"

SCHEMES = {"BMT": "ace-nme-caps-v2", "XYZ": "methyl-caps-v2"}


def _template_list(schemes):
    templates = NcaaTemplateList()
    templates.charge_scheme_by_residue.update(schemes)
    return templates


class TestResultField:
    def test_default_is_empty_and_gives_no_column(self):
        row = RelaxationResult(sample_id="x", success=True).to_dict()
        assert row["ncaa_charge_scheme"] == {}
        assert not [key for key in _flatten({"relax": row}) if "charge_scheme" in key]

    def test_to_dict_carries_a_copy_of_the_mapping(self):
        result = RelaxationResult(sample_id="x", success=True, ncaa_charge_scheme=dict(SCHEMES))
        row = result.to_dict()
        assert row["ncaa_charge_scheme"] == SCHEMES
        row["ncaa_charge_scheme"]["BMT"] = "changed"
        assert result.ncaa_charge_scheme["BMT"] == "ace-nme-caps-v2"

    def test_json_report_and_csv_columns(self, tmp_path):
        row = RelaxationResult(sample_id="x", success=True, ncaa_charge_scheme=SCHEMES)
        results = {"sample_id": "x", "relax": row.to_dict()}
        path = write_report(results, tmp_path, "x", fmt="json")
        assert (
            json.loads(path.read_text(encoding="utf-8"))["relax"]["ncaa_charge_scheme"] == SCHEMES
        )
        flat = _flatten(results)
        assert flat["relax_ncaa_charge_scheme_BMT"] == "ace-nme-caps-v2"
        assert flat["relax_ncaa_charge_scheme_XYZ"] == "methyl-caps-v2"


class TestRelaxationRecordsTheScheme:
    @pytest.fixture
    def stub_template_step(self, monkeypatch):
        """1YCR has no exotic residue, so the template step is stubbed to report some."""

        def fake(topology, positions, ff, **kwargs):
            return topology, positions, _template_list(SCHEMES)

        monkeypatch.setattr(gaff_ncaa, "parameterize_ncaa_residues", fake)

    @pytest.mark.integration
    def test_run_puts_the_mapping_in_the_result(
        self, tmp_path, prepped_example_cif, stub_template_step
    ):
        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            min_steps_initial=5,
            min_steps_restrained=5,
            min_steps_final=5,
            small_molecules="auto",
        )
        result = ImplicitRelaxation(config).run(prepped_example_cif, tmp_path / "out")
        assert result.success, result.error_message
        assert result.ncaa_charge_scheme == SCHEMES
        assert result.to_dict()["ncaa_charge_scheme"] == SCHEMES

    @pytest.mark.integration
    def test_without_the_auto_route_the_mapping_stays_empty(self, tmp_path, prepped_example_cif):
        config = RelaxationConfig(
            md_duration_ps=0.0,
            device="cpu",
            min_steps_initial=5,
            min_steps_restrained=5,
            min_steps_final=5,
        )
        result = ImplicitRelaxation(config).run(prepped_example_cif, tmp_path / "out")
        assert result.success, result.error_message
        assert result.ncaa_charge_scheme == {}

    def test_a_second_run_does_not_inherit_the_mapping(self, monkeypatch, prepped_example_cif):
        """The relaxer is reused for several samples: the mapping is reset in every setup."""
        relaxer = ImplicitRelaxation(RelaxationConfig(small_molecules="auto"))
        monkeypatch.setattr(
            gaff_ncaa,
            "parameterize_ncaa_residues",
            lambda topology, positions, ff, **kw: (topology, positions, _template_list(SCHEMES)),
        )
        relaxer._setup_system(Path(prepped_example_cif))
        assert relaxer._ncaa_charge_scheme == SCHEMES

        monkeypatch.setattr(
            gaff_ncaa,
            "parameterize_ncaa_residues",
            lambda topology, positions, ff, **kw: (topology, positions, NcaaTemplateList()),
        )
        relaxer._setup_system(Path(prepped_example_cif))
        assert relaxer._ncaa_charge_scheme == {}


@pytest.mark.slow
class TestPrepReport:
    def test_cyclic_prep_records_the_scheme(self, monkeypatch):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        real = gaff_ncaa.parameterize_ncaa_residues

        def with_schemes(topology, positions, ff, **kwargs):
            topology, positions, templates = real(topology, positions, ff, **kwargs)
            templates.charge_scheme_by_residue.update(SCHEMES)
            return topology, positions, templates

        monkeypatch.setattr(gaff_ncaa, "parameterize_ncaa_residues", with_schemes)
        topology, positions = load_structure(SFTI1)
        report = {}
        prep_structure(topology, positions, report=report)
        assert report["ncaa_charge_scheme"] == SCHEMES

    def test_prep_of_cyclosporin_records_the_scheme_of_its_gaff_residues(self):
        """1CWA: BMT and ABA are parameterised from ACE/NME-capped molecules (scheme version 2)."""
        pytest.importorskip("pdbfixer")
        if shutil.which("antechamber") is None:
            pytest.skip("antechamber is not on PATH (the GAFF2 route needs it)")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(DATA_DIR / "example_ncaa_cyclosporin_1CWA.cif")
        report = {}
        prep_structure(topology, positions, report=report)
        assert report["ncaa_charge_scheme"] == {"BMT": "ace-nme-caps-v2", "ABA": "ace-nme-caps-v2"}
        assert report["ncaa_bond_order_source"].keys() == report["ncaa_charge_scheme"].keys()

    def test_cyclic_prep_without_a_gaff_residue_has_no_such_key(self, monkeypatch):
        """3P8F has none: the glutathione is stripped before the templates are built."""
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(SFTI1)
        report = {}
        prep_structure(topology, positions, report=report)
        assert "ncaa_charge_scheme" not in report

    def test_linear_prep_has_no_such_key(self):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(EXAMPLE_1YCR)
        report = {}
        prep_structure(topology, positions, report=report)
        assert "ncaa_charge_scheme" not in report


class TestPipelineResults:
    def test_prep_and_relax_sections_carry_the_mapping(self, tmp_path, monkeypatch):
        from binding_metrics.core import system
        from binding_metrics.protocols.relaxer import Relaxer

        def fake_prep(topology, positions, report=None, **kwargs):
            report["ncaa_charge_scheme"] = dict(SCHEMES)
            return topology, positions

        class StubRelaxer(Relaxer):
            def run(self, input_path, output_dir, sample_id=None):
                return RelaxationResult(
                    sample_id=sample_id,
                    success=True,
                    minimized_structure_path=str(input_path),
                    ncaa_charge_scheme=dict(SCHEMES),
                )

        monkeypatch.setattr(system, "prep_structure", fake_prep)
        results = run_pipeline(EXAMPLE_1YCR, tmp_path, relaxer=StubRelaxer(), metrics=frozenset())
        assert results["prep"]["ncaa_charge_scheme"] == SCHEMES
        assert results["relax"]["ncaa_charge_scheme"] == SCHEMES
