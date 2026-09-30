"""The input limits that the Boltz-2 adapter declares, and the source they rest on.

Boltz-2 refuses no closure, so the declaration holds one warning. The evidence tests pin why no
closure limit is declared: the model takes more than head-to-tail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from binding_metrics.capabilities import Capabilities, preflight, profile_input
from binding_metrics.predictors.registry import PARSERS
from tests.test_pre_structures import model_source_text

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


class TestTheDeclaration:
    caps = PARSERS["boltz2"].load_capabilities()

    def test_it_is_a_capabilities_for_version_2_2_1(self):
        assert isinstance(self.caps, Capabilities)
        assert self.caps.version == "2.2.1"

    def test_it_limits_nothing(self):
        assert self.caps.is_unconstrained
        assert self.caps.closures == frozenset()  # head-to-tail is not the only closure it takes

    def test_the_one_caveat_is_about_a_staple_and_cites_the_documentation(self):
        assert list(self.caps.caveats) == ["closures:staple"]
        assert "docs/prediction.md" in self.caps.caveats["closures:staple"]


class TestPreflight:
    @pytest.mark.parametrize("entry", sorted(EXAMPLES))
    def test_every_example_passes(self, entry):
        assert preflight(_profile(entry), [], "boltz2").compatible

    def test_a_staple_passes_with_the_warning(self):
        report = preflight(_profile("3V3B"), [], "boltz2")
        (warning,) = report.warnings
        assert warning.startswith("predictor Boltz-2 2.2.1: ")
        assert "canonical residues only" in warning

    @pytest.mark.parametrize("entry", ["1YCR", "1CWA", "3P8F", "1XY4"])
    def test_a_disulfide_a_lactam_and_a_ring_give_no_warning(self, entry):
        assert preflight(_profile(entry), [], "boltz2").warnings == ()


class TestTheSourceBehindIt:
    """Re-read the Boltz-2 source and documentation when a clone is at hand (v2.2.1)."""

    def test_the_documentation_limits_the_bond_constraint_to_canonical_residues(self):
        text = model_source_text("boltz", "docs/prediction.md")
        assert "It is currently only supported for CCD ligands and canonical residues" in text
        assert (
            "The `cyclic` flag indicates whether a polymer chain (not ligands) is cyclic." in text
        )

    def test_the_cyclic_flag_is_a_period_equal_to_the_sequence_length(self):
        text = model_source_text("boltz", "src/boltz/data/parse/schema.py")
        assert "cyclic_period = len(sequence)" in text

    def test_a_bond_constraint_takes_any_two_atoms_of_the_input(self):
        text = model_source_text("boltz", "src/boltz/data/parse/schema.py")
        assert "atom_idx_map[(c1, r1 - 1, a1)]" in text
        assert "atom_idx_map[(c2, r2 - 1, a2)]" in text

    def test_a_bond_between_the_ends_of_a_chain_sets_the_cyclic_period(self):
        text = model_source_text("boltz", "src/boltz/data/feature/featurizerv2.py")
        assert "cyclic period is either computed from the bonds or given as input flag" in text
