"""The input limits that the Protenix adapter declares, and the documentation they rest on."""

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
from tests.test_pre_structures import model_source_text

DATA = Path(__file__).resolve().parent.parent / "data"

# (file, binder chain, receptor chain or None): the peptide of each bundled example.
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


class TestTheDeclaration:
    caps = PARSERS["protenix"].load_capabilities()

    def test_it_is_a_capabilities_for_version_2_0_0(self):
        assert isinstance(self.caps, Capabilities)
        assert self.caps.version == "2.0.0"

    def test_closures_are_limited_to_none_head_to_tail_and_disulfide(self):
        assert self.caps.closures == frozenset({"none", "head_to_tail", "disulfide"})

    def test_the_reason_quotes_the_documentation(self):
        reason = self.caps.reason_for("closures", "lactam")
        for cited in ("covalent_bonds", "not reliably handled", "docs/infer_json_format.md"):
            assert cited in reason

    def test_nothing_else_is_limited(self):
        assert self.caps.residue_classes == frozenset()
        assert self.caps.binder_types == frozenset()
        assert self.caps.min_binder_residues is None and self.caps.max_binder_residues is None
        assert self.caps.multi_chain_binder is True
        assert self.caps.needs == frozenset()
        assert self.caps.extra_checks == ()


class TestPreflight:
    @pytest.mark.parametrize("entry", ["1YCR", "1CWA", "3P8F", "1QJB"])
    def test_none_head_to_tail_and_disulfide_pass_whatever_the_residues(self, entry):
        # 3P8F is bicyclic (head-to-tail and disulfide); 1CWA holds D, N-methyl and CCD residues
        assert preflight(_profile(entry), [], "protenix").compatible

    def test_a_lactam_is_refused_with_the_model_the_link_and_the_reason(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(_profile("1XY4"), [], "protenix")
        message = str(caught.value)
        assert "predictor Protenix 2.0.0: closures" in message
        assert "a lactam bridge (LYS 10.NZ - GLU 4.CD)" in message
        assert "closures limited to: none, head_to_tail, disulfide" in message
        assert "not reliably handled" in message
        # the disulfide of somatostatin is accepted, only the lactam is reported
        assert [v.constraint for v in caught.value.violations] == ["closures"]

    def test_a_staple_is_refused(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(_profile("3V3B"), [], "protenix")
        (violation,) = caught.value.violations
        assert "a hydrocarbon staple" in violation.fact

    def test_the_rejected_model_is_not_offered_as_an_alternative(self):
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(_profile("3V3B"), [], "protenix")
        assert "Protenix (protenix)" not in str(caught.value)


class TestTheDocumentationBehindIt:
    """Re-read the Protenix documentation when a clone is at hand (commit 85767b8)."""

    def test_the_covalent_bond_section_says_what_the_reason_says(self):
        text = model_source_text("Protenix", "docs/infer_json_format.md")
        for sentence in (
            "Covalent bonds between two polymer residues",
            "are generally not supported",
            "A head-to-tail amide bond connecting the N- and C-terminal residues",
            "A disulfide bond between cysteine residues",
            "not reliably handled by the current model",
        ):
            assert sentence in text

    def test_the_source_takes_any_atom_pair_so_the_limit_is_about_reliability(self):
        text = model_source_text("Protenix", "protenix/data/inference/json_to_feature.py")
        assert "atom_array.bonds.add_bond(atom_idx1, atom_idx2, 1)" in text
