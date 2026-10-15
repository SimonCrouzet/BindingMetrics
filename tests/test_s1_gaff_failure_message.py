"""A residue whose GAFF2 template fails is named in the error that follows (#200).

``parameterize_ncaa_residues`` skips a residue whose template cannot be generated and logs a
warning. The force field then fails on the first residue it cannot match: for a model of 1XY4
whose IAM atoms have collapsed, that is the D-Trp before it ("No template found for residue 91
(TRP)"). The error now starts with the residue that failed and the reason, then the text of the
force field.

The collapsed model is the bundled 1XY4 written without bond records, with the IAM atom CI moved
next to NH (0.069 nm, as in a saved Boltz-2 model), so that the bonds perceived from the distances
give the residue 33 bonds and its template fails.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import DATA_DIR

pytest.importorskip("openmm")

from binding_metrics.core import gaff_ncaa
from binding_metrics.core.gaff_ncaa import NcaaTemplateList, explain_missing_templates

SOMATOSTATIN = DATA_DIR / "example_lactam_somatostatin_1XY4.cif"
OPENMM_TEXT = (
    "No template found for residue 91 (TRP).  The set of atoms matches TRP, but the bonds are "
    "different."
)
COLLAPSE_NM = 0.069


def failing_templates(monkeypatch, failure):
    """The list that ``parameterize_ncaa_residues`` returns when every template raises."""
    pytest.importorskip("openmmforcefields")
    residue = SimpleNamespace(name="IAM", index=0)
    topology = SimpleNamespace(residues=lambda: iter([residue]))

    def broken(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(gaff_ncaa, "_is_ncaa", lambda res: True)
    monkeypatch.setattr(gaff_ncaa, "_pos_to_angstrom", lambda positions: np.zeros((1, 3)))
    monkeypatch.setattr(gaff_ncaa, "_amber_backbone_types", lambda ff: None)
    monkeypatch.setattr(gaff_ncaa, "_generate_residue_template", broken)
    return gaff_ncaa.parameterize_ncaa_residues(topology, [None], ff=None, verbose=False)[2]


class TestTheFailedResiduesAreRecorded:
    def test_the_reason_is_kept_by_residue_name(self, monkeypatch):
        templates = failing_templates(
            monkeypatch, ValueError("Explicit valence for atom # 6 C, 5, is greater than permitted")
        )

        assert len(templates) == 0
        assert templates.failed_by_residue == {
            "IAM": "Explicit valence for atom # 6 C, 5, is greater than permitted"
        }

    def test_a_list_without_failures_has_none(self):
        assert NcaaTemplateList().failed_by_residue == {}


class TestTheForceFieldErrorNamesTheResidue:
    @staticmethod
    def templates(**failed):
        templates = NcaaTemplateList()
        templates.failed_by_residue.update(failed)
        return templates

    def test_the_message_starts_with_the_residue_and_the_reason_then_the_openmm_text(self):
        original = ValueError(OPENMM_TEXT)

        with pytest.raises(ValueError) as caught:
            with explain_missing_templates(self.templates(IAM="Explicit valence")):
                raise original

        message = str(caught.value)
        assert message.startswith(
            "GAFF2 template generation failed for IAM (IAM: Explicit valence)"
        )
        assert message.endswith(OPENMM_TEXT)
        assert caught.value.__cause__ is original

    def test_every_failed_residue_is_named(self):
        with pytest.raises(ValueError, match=r"for IAM, DTR \(IAM: one; DTR: two\)"):
            with explain_missing_templates(self.templates(IAM="one", DTR="two")):
                raise ValueError(OPENMM_TEXT)

    def test_a_long_multi_line_reason_is_cut_to_one_line(self):
        reason = "antechamber failed\n" + "Running: sqm -O -i sqm.in\n" * 40

        with pytest.raises(ValueError) as caught:
            with explain_missing_templates(self.templates(IAM=reason)):
                raise ValueError(OPENMM_TEXT)

        first, _, rest = str(caught.value).partition("), so the force field")
        assert "\n" not in first and len(first) < 400
        assert "(IAM: antechamber failed Running: sqm -O -i sqm.in" in first
        assert first.endswith("...")
        assert rest.endswith(OPENMM_TEXT)

    def test_without_a_failed_template_the_error_is_the_same_object(self):
        original = ValueError(OPENMM_TEXT)

        with pytest.raises(ValueError) as caught:
            with explain_missing_templates(NcaaTemplateList()):
                raise original

        assert caught.value is original

    def test_another_error_is_not_changed(self):
        original = ValueError("Particle coordinate is NaN")

        with pytest.raises(ValueError) as caught:
            with explain_missing_templates(self.templates(IAM="Explicit valence")):
                raise original

        assert caught.value is original

    def test_nothing_is_raised_when_nothing_fails(self):
        with explain_missing_templates(self.templates(IAM="Explicit valence")):
            pass

    def test_an_object_that_records_nothing_leaves_the_error_alone(self):
        original = ValueError(OPENMM_TEXT)

        with pytest.raises(ValueError) as caught:
            with explain_missing_templates([]):
                raise original

        assert caught.value is original


@pytest.fixture(scope="module")
def collapsed_somatostatin(tmp_path_factory):
    """The bundled 1XY4 as a model (no bond records) with the IAM atom CI next to NH."""
    from binding_metrics.metrics._common import load_structure as load_atoms
    from binding_metrics.testing.synthetic import write_cif

    atoms = load_atoms(SOMATOSTATIN)
    atoms = atoms[atoms.res_name != "HOH"]
    iam = atoms.res_name == "IAM"
    nitrogen = np.flatnonzero(iam & (atoms.atom_name == "NH"))[0]
    carbon = np.flatnonzero(iam & (atoms.atom_name == "CI"))[0]
    atoms.coord[carbon] = atoms.coord[nitrogen] + np.array([10 * COLLAPSE_NM, 0.0, 0.0])
    path = tmp_path_factory.mktemp("collapsed") / "collapsed.cif"
    write_cif(atoms, path)
    return path


@pytest.mark.slow
@pytest.mark.integration
class TestThePrepOfACollapsedResidue:
    def test_prep_names_iam_and_the_reason(self, collapsed_somatostatin):
        pytest.importorskip("pdbfixer")
        from binding_metrics.core.system import prep_structure
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(collapsed_somatostatin)

        with pytest.raises(ValueError) as caught:
            prep_structure(topology, positions)

        message = str(caught.value)
        assert message.startswith("GAFF2 template generation failed for IAM (IAM: ")
        assert "No template found for residue" in message
        assert isinstance(caught.value.__cause__, ValueError)
        assert "No template found for residue" in str(caught.value.__cause__)
