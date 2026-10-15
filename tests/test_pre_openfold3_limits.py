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

    def test_a_head_to_tail_peptide_passes_with_the_warning_about_what_the_flag_does_not_do(self):
        report = preflight(_profile("1CWA"), [], "of3")
        assert report.compatible
        (warning,) = [w for w in report.warnings if "cyclic: true" in w]
        assert warning.startswith("predictor OpenFold3 0.5.0: ")
        # what remains true: no enforced bond, an example only, no published benchmark
        assert "does not enforce the closure bond" in warning
        assert "example query" in warning
        assert "no accuracy benchmark for cyclic peptides" in warning
        # the builder writes the flag by default now, so the old sentence must be gone
        assert "folded as a linear chain" not in warning
        assert "do not write" not in warning

    def test_the_warning_gives_what_the_runs_showed_as_one_example_per_case(self):
        (warning,) = [w for w in preflight(_profile("1CWA"), [], "of3").warnings if "cyclic" in w]
        # the sources of the mechanism
        for cited in ("relpos.py", "cyclic_offset", "tokenize_atom_array", "tokenization.py"):
            assert cited in warning
        assert "one token per heavy atom" in warning
        # the measurements, and that they are examples (OpenFold3 0.5.0, no MSA)
        assert "one example of each case, not a benchmark" in warning
        assert "C-N 1.38 A with it, 7.40 A without" in warning
        assert "(3P8F, one seed)" in warning and "(three seeds)" in warning
        assert "ipTM 0.78-0.81 against 0.91-0.92" in warning
        assert "3.0-4.8 A against 0.5-0.7 A" in warning
        # what they do not show
        assert "no ablation of the model was done" in warning
        assert "the cause is not established" in warning

    def test_the_warning_says_which_binders_get_the_flag_by_default(self):
        (warning,) = [w for w in preflight(_profile("1CWA"), [], "of3").warnings if "cyclic" in w]
        assert "only for a binder of standard residues" in warning
        assert "binder_cyclic=True (`--openfold-cyclic on`) writes it for any binder" in warning
        assert "binder_cyclic=False (`--openfold-cyclic off`) never does" in warning

    def test_the_old_wording_is_gone(self):
        (warning,) = [w for w in preflight(_profile("1CWA"), [], "of3").warnings if "cyclic" in w]
        assert "so there is no published check of its confidence values" not in warning
        assert "when binder_cyclic is" not in warning

    def test_the_warning_names_the_switch_that_turns_the_flag_off(self):
        (warning,) = [w for w in preflight(_profile("1CWA"), [], "of3").warnings if "cyclic" in w]
        assert "binder_cyclic" in warning and "--openfold-cyclic off" in warning

    def test_the_caveat_is_true_of_the_query_builder(self):
        # The caveat "closures:head_to_tail" says that the query builders write `cyclic: true` on
        # a head-to-tail binder of standard residues by default, and on any binder when asked. If
        # this fails, the default changed (or the builder no longer writes the field): reword the
        # caveat in the OpenFold3 declaration.
        from binding_metrics.metrics import _openfold_run, openfold

        tree = ast.parse(textwrap.dedent(inspect.getsource(_openfold_run._query_chain)))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "cyclic" in strings
        for builder in (openfold.prepare_scoring_query, openfold.prepare_refolding_query):
            assert inspect.signature(builder).parameters["binder_cyclic"].default == "auto"

    def _chains_of_the_query(self, tmp_path, monkeypatch, entry, **kwargs):
        import json

        from binding_metrics.metrics import _openfold_run, openfold

        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
        )
        name, binder, receptor = EXAMPLES[entry]
        path = openfold.prepare_refolding_query(
            DATA / name, receptor, binder, "q", tmp_path, **kwargs
        )
        chains = {
            c["chain_ids"][0]: c
            for c in json.loads(path.read_text(encoding="utf-8"))["queries"]["q"]["chains"]
        }
        return chains[binder], chains[receptor]

    def test_the_builder_writes_the_flag_by_default_on_a_binder_of_standard_residues(
        self, tmp_path, monkeypatch
    ):
        # SFTI-1 of 3P8F: 14 standard residues closed head to tail (and a disulfide, never written)
        binder, receptor = self._chains_of_the_query(tmp_path, monkeypatch, "3P8F")
        assert binder.get("cyclic") is True and "cyclic" not in receptor

    def test_the_builder_writes_the_flag_on_the_binder_with_modified_residues_when_asked(
        self, tmp_path, monkeypatch
    ):
        binder, receptor = self._chains_of_the_query(
            tmp_path, monkeypatch, "1CWA", binder_cyclic=True
        )
        assert binder.get("cyclic") is True and "cyclic" not in receptor
        assert binder["non_canonical_residues"]  # nine modified residues go to the CCD route

    @pytest.mark.parametrize("entry", ["1CWA", "3P8F"])
    def test_the_builder_never_writes_the_flag_when_turned_off(self, tmp_path, monkeypatch, entry):
        binder, _ = self._chains_of_the_query(tmp_path, monkeypatch, entry, binder_cyclic=False)
        assert "cyclic" not in binder

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
