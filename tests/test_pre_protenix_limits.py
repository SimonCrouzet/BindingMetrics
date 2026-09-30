"""The input limits that the Protenix adapter declares, and the documentation they rest on."""

from __future__ import annotations

from pathlib import Path

import pytest

from binding_metrics.capabilities import Capabilities, preflight, profile_input
from binding_metrics.predictors.registry import PARSERS
from tests.test_pre_structures import build_chain, model_source_text

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

    def test_it_refuses_nothing(self):
        # "not reliably handled" is a reliability statement, so it is a caveat and not a limit
        assert self.caps.is_unconstrained
        assert self.caps.closures == frozenset()

    def test_the_caveats_are_the_closures_the_documentation_does_not_list(self):
        assert sorted(self.caps.caveats) == ["closures:lactam", "closures:other", "closures:staple"]

    def test_the_caveat_quotes_the_documentation_with_its_file_and_section(self):
        text = self.caps.caveats["closures:lactam"]
        for quoted in (
            "not reliably handled by the current model",
            "may tend to be positioned in close proximity",
            "docs/infer_json_format.md, section covalent_bonds",
        ):
            assert quoted in text


class TestPreflight:
    @pytest.mark.parametrize("entry", sorted(EXAMPLES))
    def test_every_example_passes(self, entry):
        assert preflight(_profile(entry), [], "protenix").compatible

    @pytest.mark.parametrize("entry", ["1YCR", "1CWA", "3P8F", "1QJB"])
    def test_none_head_to_tail_and_disulfide_give_no_warning(self, entry):
        # 3P8F is bicyclic (head-to-tail and disulfide); 1CWA holds D, N-methyl and CCD residues
        assert preflight(_profile(entry), [], "protenix").warnings == ()

    def test_a_lactam_gives_the_warning_and_no_refusal(self):
        report = preflight(_profile("1XY4"), [], "protenix", policy="error")
        assert report.compatible
        (warning,) = report.warnings
        assert warning.startswith("predictor Protenix 2.0.0: ")
        assert "not reliably handled" in warning

    def test_a_staple_gives_the_warning(self):
        (warning,) = preflight(_profile("3V3B"), [], "protenix").warnings
        assert "docs/infer_json_format.md" in warning

    def test_another_cross_link_gives_the_warning(self):
        atoms = build_chain(
            ["ALA", "CYS", "GLY", "GLY", "GLY", "XAA", "ALA"],
            "B",
            side_chain_atoms={"XAA": ("CB", "CE")},
            bonds=[(1, "SG", 5, "CE")],
        )
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"other"})
        assert len(preflight(profile, [], "protenix").warnings) == 1


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

    def test_the_source_takes_any_atom_pair_so_it_is_a_reliability_statement(self):
        text = model_source_text("Protenix", "protenix/data/inference/json_to_feature.py")
        assert "atom_array.bonds.add_bond(atom_idx1, atom_idx2, 1)" in text
