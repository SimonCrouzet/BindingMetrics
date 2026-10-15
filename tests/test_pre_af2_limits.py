"""The input limits that the AlphaFold2 / ColabFold adapter declares, and the code they rest on."""

from __future__ import annotations

from pathlib import Path

import pytest

from binding_metrics.capabilities import (
    Capabilities,
    IncompatibleInputError,
    preflight,
    profile_input,
)
from binding_metrics.predictors.registry import PARSERS
from tests.test_pre_structures import build_chain, model_source_text

DATA = Path(__file__).resolve().parent.parent / "data"

EXAMPLES = {
    "1YCR": ("example_linear_p53_1YCR.pdb", "B", "A"),
    "1CWA": ("example_ncaa_cyclosporin_1CWA.cif", "C", "A"),
    "3P8F": ("example_bicyclic_sfti1_3P8F.cif", "I", "A"),
    "1XY4": ("example_lactam_somatostatin_1XY4.cif", "A", None),
    "3V3B": ("example_staple_3V3B.pdb", "C", None),
    "1QJB": ("example_phospho_1QJB.pdb", "Q", None),
}


def _profile(entry):
    name, binder, receptor = EXAMPLES[entry]
    return profile_input(DATA / name, binder, receptor)


def _refusal(profile):
    with pytest.raises(IncompatibleInputError) as caught:
        preflight(profile, [], "af2")
    return caught.value


class TestTheDeclaration:
    caps = PARSERS["af2"].load_capabilities()

    def test_it_is_a_capabilities(self):
        assert isinstance(self.caps, Capabilities)

    def test_closures_are_limited_to_none(self):
        assert self.caps.closures == frozenset({"none"})

    def test_residues_are_limited_to_the_canonical_ones_and_what_a_sequence_leaves_out(self):
        assert self.caps.residue_classes == frozenset({"canonical", "cap", "ligand"})

    def test_the_reasons_cite_the_code(self):
        closures = self.caps.reason_for("closures", "head_to_tail")
        assert "build_monomer_feature" in closures and "make_sequence_features" in closures
        residues = self.caps.reason_for("residue_classes", "d_amino")
        assert "restypes_with_x" in residues and "residue_constants.py" in residues

    def test_nothing_else_is_limited(self):
        assert self.caps.binder_types == frozenset()
        assert self.caps.min_binder_residues is None and self.caps.max_binder_residues is None
        assert self.caps.multi_chain_binder is True
        assert self.caps.needs == frozenset() and self.caps.extra_checks == ()


class TestPreflight:
    def test_a_linear_canonical_peptide_passes(self):
        assert preflight(_profile("1YCR"), ["interface"], "af2").compatible

    def test_a_ring_is_refused_even_head_to_tail(self):
        atoms = build_chain(
            ["ALA", "GLY", "ALA", "GLY", "ALA"], "B", close=[(4, "C", 0, "N", 1.33)]
        )
        error = _refusal(profile_input(atoms, "B"))
        (violation,) = error.violations
        assert violation.constraint == "closures"
        assert "a head-to-tail amide closure" in violation.fact
        assert "closures limited to: none" in str(error)
        assert "build_monomer_feature" in str(error)

    def test_cyclosporin_is_refused_for_its_ring_and_its_d_n_methyl_and_ncaa_residues(self):
        error = _refusal(_profile("1CWA"))
        facts = [v.fact for v in error.violations]
        assert [v.constraint for v in error.violations] == [
            "closures",
            "residue_classes",
            "residue_classes",
            "residue_classes",
        ]
        assert any("D-amino acids (DAL)" in f for f in facts)
        assert any("N-methylated residues (MLE, MVA, SAR)" in f for f in facts)
        assert any("other non-canonical amino acids (ABA, BMT)" in f for f in facts)
        assert "predictor AlphaFold2 / ColabFold: residue_classes" in str(error)

    def test_a_phosphopeptide_is_refused(self):
        (violation,) = _refusal(_profile("1QJB")).violations
        assert "phosphorylated residues (SEP)" in violation.fact

    def test_a_bicycle_lists_both_links(self):
        facts = [v.fact for v in _refusal(_profile("3P8F")).violations]
        assert len(facts) == 2
        assert any("a head-to-tail amide closure" in f for f in facts)
        assert any("a disulfide bond" in f for f in facts)

    def test_caps_and_ligands_are_warnings_because_a_sequence_leaves_them_out(self):
        atoms = build_chain(
            ["ACE", "ALA", "GLY", "ALA", "NME", "GSH"],
            "B",
            residue_atoms={
                "ACE": ("C", "O", "CH3"),
                "NME": ("N", "C"),
                "GSH": ("S1", "C1", "O1"),
            },
        )
        report = preflight(profile_input(atoms, "B"), [], "af2")
        assert report.compatible
        assert any("capping group" in w for w in report.warnings)
        assert any("ligand or glycan" in w for w in report.warnings)

    def test_the_skip_policy_marks_the_model_unusable_and_keeps_the_metrics(self):
        report = preflight(_profile("1CWA"), ["interface"], "af2", policy="skip")
        assert report.predictor_usable is False and report.metrics_to_run == ("interface",)


class TestTheCodeBehindIt:
    """Re-read the AlphaFold and ColabFold code when the clones are at hand."""

    def test_the_residue_types_are_the_20_amino_acids_and_x(self):
        text = model_source_text("alphafold", "alphafold/common/residue_constants.py")
        assert "restypes_with_x = restypes + ['X']" in text
        assert "restype_num = len(restypes)  # := 20." in text

    def test_the_sequence_features_map_any_other_letter_to_x(self):
        text = model_source_text("alphafold", "alphafold/data/pipeline.py")
        assert "mapping=residue_constants.restype_order_with_x," in text
        assert "map_unknown_to_x=True," in text

    def test_colabfold_builds_its_features_from_sequence_msa_and_templates_only(self):
        text = model_source_text("ColabFold", "colabfold/batch.py")
        start = text.index("def build_monomer_feature(")
        function = text[start : text.index("def build_multimer_feature", start)]
        assert "pipeline.make_sequence_features(" in function
        assert "pipeline.make_msa_features([msa])" in function
        assert "**template_features" in function
        assert "bond" not in function and "cyclic" not in function

    def test_colabfold_has_no_word_cyclic_in_its_modules(self):
        # absence in the code read; the claim rests on the feature list above
        import os
        from pathlib import Path as _Path

        text = model_source_text("ColabFold", "colabfold/batch.py")
        assert "cyclic" not in text.lower()
        root = _Path(os.environ["BINDING_METRICS_MODEL_SOURCES"]) / "ColabFold" / "colabfold"
        assert all("cyclic" not in p.read_text(encoding="utf-8").lower() for p in root.glob("*.py"))
