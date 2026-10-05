"""``check_openfold3_residues``: the residues that the OpenFold3 query builder cannot express.

The check calls the builder's own rule (``_residue_letter_and_ccd``) and its reason is the text of
the ``UnmappableResidueError`` that ``_extract_query_chain`` raises for the same chain; this file
compares the two on the bundled examples and on synthetic chains.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from binding_metrics.capabilities import (
    Capabilities,
    IncompatibleInputError,
    check_openfold3_residues,
    preflight,
    profile_input,
)
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
        preflight(profile, [], _OpenFold3Like, **kwargs)
    return caught.value


class _OpenFold3Like:
    """A predictor stand-in with the declaration of OpenFold3 that these tests need."""

    display_name = "OpenFold3"
    name = "of3_like"
    capabilities = Capabilities(
        closures={"none", "head_to_tail"},
        extra_checks=(check_openfold3_residues,),
        reasons={"closures": "Only a head-to-tail closure can be given."},
        version="0.5.0",
    )


class TestCheck:
    @pytest.mark.parametrize("entry", sorted(EXAMPLES))
    def test_the_check_agrees_with_the_query_builder_on_the_examples(self, entry):
        from binding_metrics.metrics._openfold_run import (
            UnmappableResidueError,
            _extract_query_chain,
        )

        name, binder, _ = EXAMPLES[entry]
        structure = gemmi.read_structure(str(DATA / name))
        try:
            _extract_query_chain(structure, binder)
            builder_refuses = False
        except UnmappableResidueError:
            builder_refuses = True
        assert bool(check_openfold3_residues(_profile(entry))) is builder_refuses
        assert builder_refuses is False  # OpenFold3 takes D, N-methyl, phospho and CCD residues

    def _synthetic(self, tmp_path):
        names = ["ALA", "GLY", "XYZ", "DAL", "ALA", "QQQ"]
        atoms = build_chain(names, "B", first_number=4)
        atoms.ins_code[atoms.res_id == 9] = "A"
        import biotite.structure.io.pdb as pdb_io

        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, atoms)
        path = tmp_path / "custom.pdb"
        pdb_file.write(str(path))
        return atoms, path

    def test_a_residue_the_builder_cannot_express_is_refused_with_the_builders_text(self, tmp_path):
        from binding_metrics.metrics._openfold_run import (
            UnmappableResidueError,
            _extract_query_chain,
        )

        atoms, path = self._synthetic(tmp_path)
        with pytest.raises(UnmappableResidueError) as builder:
            _extract_query_chain(gemmi.read_structure(str(path)), "B")
        error = _refusal(profile_input(atoms, "B"))
        (violation,) = error.violations
        assert violation.constraint == "residue_classes"
        assert "XYZ 6, QQQ 9A" in violation.fact
        # the reason is the text of the error the query builder raises for the same chain
        assert violation.reason == str(builder.value)
        assert "predictor OpenFold3 0.5.0: residue_classes" in str(error)

    def test_a_d_residue_and_a_ccd_residue_are_not_refused(self):
        atoms = build_chain(["ALA", "DAL", "SAR", "SEP", "BMT", "GLY"], "B")
        assert check_openfold3_residues(profile_input(atoms, "B")) == []

    def test_caps_and_ligands_are_left_out_of_the_query_and_are_not_checked(self):
        atoms = build_chain(
            ["ACE", "ALA", "GLY", "ALA", "NME", "GSH"],
            "B",
            residue_atoms={
                "ACE": ("C", "O", "CH3"),
                "NME": ("N", "C"),
                "GSH": ("S1", "C1", "O1"),
            },
        )
        assert check_openfold3_residues(profile_input(atoms, "B")) == []

    def test_a_hand_built_profile_without_labels_is_checked_by_name(self):
        from binding_metrics.capabilities import InputProfile

        profile = InputProfile(
            "B",
            residue_classes=frozenset({"canonical", "other_ncaa"}),
            residue_names={"canonical": ("ALA",), "other_ncaa": ("XYZ",)},
        )
        (violation,) = check_openfold3_residues(profile)
        assert "XYZ" in violation.fact

    def test_all_the_problems_of_a_sample_are_listed_at_once(self):
        atoms = build_chain(
            ["ALA", "CYS", "XYZ", "GLY", "CYS", "ALA"], "B", close=[(4, "SG", 1, "SG", 2.05)]
        )
        error = _refusal(profile_input(atoms, "B"))
        assert sorted(v.constraint for v in error.violations) == ["closures", "residue_classes"]
