"""The input limits that the OpenFold3 adapter declares, and the checks that back them.

The closure limit comes from OpenFold3 0.5.0 itself (documentation and source, cited in the
reasons); the residue limit is the rule of this package's own query builder, called through
``check_openfold3_residues`` and compared here with ``_extract_query_chain`` on the same input.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from binding_metrics.capabilities import (
    Capabilities,
    IncompatibleInputError,
    check_openfold3_residues,
    preflight,
    profile_input,
)
from binding_metrics.predictors.registry import PARSERS
from tests.test_pre_structures import build_chain

gemmi = pytest.importorskip("gemmi")
struc = pytest.importorskip("biotite.structure")

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


def _refusal(profile, **kwargs):
    with pytest.raises(IncompatibleInputError) as caught:
        preflight(profile, [], "of3", **kwargs)
    return caught.value


class TestTheDeclaration:
    caps = PARSERS["of3"].load_capabilities()

    def test_it_is_a_capabilities_for_version_0_5_0(self):
        assert isinstance(self.caps, Capabilities)
        assert self.caps.version == "0.5.0"

    def test_closures_are_limited_to_none_and_head_to_tail(self):
        assert self.caps.closures == frozenset({"none", "head_to_tail"})

    def test_the_closure_reason_cites_the_source_it_rests_on(self):
        reason = self.caps.reason_for("closures", "disulfide")
        for cited in ("cyclic: true", "relpos.py", "covalent_bonds", "inference_query_format.py"):
            assert cited in reason

    def test_nothing_is_limited_that_no_source_shows(self):
        assert self.caps.residue_classes == frozenset()
        assert self.caps.binder_types == frozenset()
        assert self.caps.min_binder_residues is None and self.caps.max_binder_residues is None
        assert self.caps.multi_chain_binder is True
        assert self.caps.needs == frozenset()

    def test_the_residue_limit_is_the_query_builders_check(self):
        assert self.caps.extra_checks == (check_openfold3_residues,)

    def test_declaring_it_imports_neither_the_query_builder_nor_a_heavy_package(self):
        code = textwrap.dedent(
            """
            import sys
            from binding_metrics.predictors.registry import PARSERS
            caps = PARSERS["of3"].load_capabilities()
            print(sorted(m for m in sys.modules
                         if m.split(".")[0] in ("openmm", "simtk", "torch", "biotite", "gemmi")
                         or m == "binding_metrics.metrics._openfold_run"))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]"


class TestClosures:
    def test_a_linear_peptide_and_a_phosphopeptide_pass(self):
        assert preflight(_profile("1YCR"), ["interface"], "of3").compatible
        assert preflight(_profile("1QJB"), [], "of3").compatible

    def test_a_head_to_tail_peptide_passes_with_the_warning_that_the_builder_sends_it_linear(self):
        report = preflight(_profile("1CWA"), [], "of3")
        assert report.compatible
        (warning,) = [w for w in report.warnings if "cyclic: true" in w]
        assert warning.startswith("predictor OpenFold3 0.5.0: ")
        assert "folded as a linear chain" in warning

    def test_the_caveat_is_true_of_the_query_builder(self):
        # If this fails, the builder now writes `cyclic` into the chain: remove the caveat
        # "closures:head_to_tail" from the OpenFold3 declaration, its sentence is no longer true.
        from binding_metrics.metrics import _openfold_run

        tree = ast.parse(textwrap.dedent(inspect.getsource(_openfold_run._query_chain)))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "cyclic" not in strings

    def test_a_disulfide_is_refused_with_the_model_the_link_and_the_reason(self):
        error = _refusal(_profile("3P8F"))
        message = str(error)
        assert "predictor OpenFold3 0.5.0: closures" in message
        assert "a disulfide bond (CYS 3.SG - CYS 11.SG)" in message
        assert "closures limited to: none, head_to_tail" in message
        assert "covalent_bonds" in message and "relpos.py" in message
        assert [v.constraint for v in error.violations] == ["closures"]  # the ring is accepted

    def test_the_disulfide_and_the_lactam_of_somatostatin_are_both_reported(self):
        error = _refusal(_profile("1XY4"))
        facts = [v.fact for v in error.violations]
        assert len(facts) == 2
        assert any("a disulfide bond" in f for f in facts)
        assert any("a lactam bridge" in f and "LYS 10.NZ" in f for f in facts)

    def test_a_staple_is_refused(self):
        (violation,) = _refusal(_profile("3V3B")).violations
        assert "a hydrocarbon staple" in violation.fact

    def test_the_skip_policy_marks_the_model_unusable_and_keeps_the_metrics(self):
        report = preflight(_profile("3P8F"), ["interface"], "of3", policy="skip")
        assert report.predictor_usable is False and report.metrics_to_run == ("interface",)

    def test_caps_and_ligands_are_left_out_of_the_query_and_are_warnings(self):
        atoms = build_chain(
            ["ACE", "ALA", "GLY", "ALA", "NME", "GSH"],
            "B",
            residue_atoms={
                "ACE": ("C", "O", "CH3"),
                "NME": ("N", "C"),
                "GSH": ("S1", "C1", "O1"),
            },
        )
        report = preflight(profile_input(atoms, "B"), [], "of3")
        assert report.compatible
        assert any("terminal capping groups" in w for w in report.warnings)
        assert any("ligand or glycan" in w for w in report.warnings)


class TestResiduesThroughTheRegistry:
    def test_a_residue_the_builder_cannot_express_is_refused_for_the_registered_model(self):
        atoms = build_chain(["ALA", "GLY", "XYZ", "ALA"], "B")
        error = _refusal(profile_input(atoms, "B"))
        (violation,) = error.violations
        assert violation.constraint == "residue_classes"
        assert "predictor OpenFold3 0.5.0: residue_classes" in str(error)

    @pytest.mark.parametrize("entry", ["1YCR", "1CWA", "1QJB"])
    def test_the_d_n_methyl_and_phospho_examples_pass(self, entry):
        assert preflight(_profile(entry), [], "of3").compatible
