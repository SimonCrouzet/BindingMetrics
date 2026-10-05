"""CYM (AMBER deprotonated cysteine) is a standard residue of the preparation code.

The standard-variant set left CYM out, so prep listed it under ``kept_nonstandard`` and
the GAFF step and the heterogen scan of the relaxation tried to parameterise it, although
amber14 has a CYM template. The old sets are pinned here; the current ones are the old
ones plus CYM.
"""

import pytest

from binding_metrics.core import residues

# core.residues.AMBER_STANDARD_VARIANTS before CYM was added.
OLD_AMBER_STANDARD_VARIANTS = frozenset("HID HIE HIN HIP CYX ASH GLH LYN".split())

# core.residues.AMBER_STANDARD_RESIDUES before CYM was added (36 names).
OLD_AMBER_STANDARD_RESIDUES = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "HIE HID HIP CYX ASH GLH LYN "
        "DA DC DG DT A C G T U"
    ).split()
)

# ImplicitRelaxation._AMBER_STANDARD before CYM was added (44 names).
OLD_RELAXATION_AMBER_STANDARD = frozenset(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
        "HID HIE HIN HIP CYX ASH GLH LYN "
        "ASPL GLUL LYSL "
        "ACE NME FOR NMA "
        "HOH WAT H2O "
        "NA CL K MG CA ZN"
    ).split()
)


class TestSetsAreTheOldOnesPlusCym:
    def test_amber_standard_variants(self):
        assert residues.AMBER_STANDARD_VARIANTS == OLD_AMBER_STANDARD_VARIANTS | {"CYM"}

    def test_amber_standard_residues(self):
        assert residues.AMBER_STANDARD_RESIDUES == OLD_AMBER_STANDARD_RESIDUES | {"CYM"}
        assert "HIN" not in residues.AMBER_STANDARD_RESIDUES  # left out on purpose, as before

    def test_relaxation_standard_set(self):
        pytest.importorskip("openmm")
        from binding_metrics.protocols.relaxation import ImplicitRelaxation

        assert ImplicitRelaxation._AMBER_STANDARD == OLD_RELAXATION_AMBER_STANDARD | {"CYM"}

    def test_protein_residues_and_disulfide_names_are_unchanged(self):
        """CYM is a thiolate: no disulfide, and the chain-ranking set is not widened here."""
        assert residues.CYSTEINE_NAMES == {"CYS", "CYX"}
        assert "CYM" not in residues.PROTEIN_RESIDUES

    def test_the_gaff_skip_set_follows_the_shared_constant(self):
        pytest.importorskip("openmm")
        from binding_metrics.core import gaff_ncaa

        assert "CYM" in gaff_ncaa.GAFF_SKIP_RESIDUES


class TestPrepTreatsCymAsStandard:
    def test_a_cym_residue_is_not_listed_as_kept_nonstandard(self):
        pytest.importorskip("pdbfixer")
        from pathlib import Path

        from openmm.app import PDBFile

        from binding_metrics.core.system import prep_structure

        pdb = PDBFile(str(Path(__file__).resolve().parents[1] / "data/example_linear_p53_1YCR.pdb"))
        cysteine = next(r for r in pdb.topology.residues() if r.name == "CYS")
        cysteine.name = "CYM"
        report: dict = {}
        topology, _ = prep_structure(pdb.topology, pdb.positions, report=report)
        assert report["kept_nonstandard"] == []
        assert sum(1 for r in topology.residues() if r.name == "CYM") == 1
