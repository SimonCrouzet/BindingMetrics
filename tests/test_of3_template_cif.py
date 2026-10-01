"""The template CIF that the OpenFold3 query builders write (single chain, one per template).

OpenFold3 0.5.0 reads these files with the parsers it uses for PDB entries. A bare gemmi chain
was rejected in score mode and for the receptor of refold mode: every query failed with exit
status 0 (``invalid literal for int() with base 10: '.'``). The tests parse the written file
with gemmi and biotite and assert that each defect is absent; one test, marked ``openfold``,
runs OpenFold3's own functions in the conda environment ``openfold3`` and is skipped when that
environment or the package is missing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")
pdbx = pytest.importorskip("biotite.structure.io.pdbx")

from binding_metrics.metrics import _openfold_run, openfold  # noqa: E402
from tests.test_of3_synth import _structure, _write  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"

#: The component types OpenFold3 maps to the protein molecule type
#: (``CHEM_COMP_TYPE_TO_MOLECULE_TYPE``, openfold3/core/data/resources/residues.py).
PROTEIN_COMPONENT_TYPES = {"PEPTIDE LINKING", "L-PEPTIDE LINKING", "D-PEPTIDE LINKING"}


def _written(structure_path: Path, chain_id: str, tmp_path: Path, name: str = "t.cif"):
    """Write the template of one chain; return (sequence, non_canonical, cif path)."""
    structure = gemmi.read_structure(str(structure_path))
    sequence, non_canonical = _openfold_run._extract_query_chain(structure, chain_id)
    out = tmp_path / name
    _openfold_run._extract_chain_to_cif(structure, chain_id, out, sequence=sequence)
    return sequence, non_canonical, out


def _column(block, tag: str) -> list[str]:
    return [str(value).strip("'\"") for value in block.find_values(tag)]


CASES = [
    pytest.param(P53, "A", id="1YCR-receptor"),
    pytest.param(P53, "B", id="1YCR-binder"),
    pytest.param(CYCLOSPORIN, "C", id="1CWA-binder-d-and-n-methyl"),
    pytest.param(CYCLOSPORIN, "A", id="1CWA-receptor-with-waters"),
    pytest.param(SFTI1, "I", id="3P8F-binder"),
]


class TestEachDefectIsAbsent:
    """Each assertion is one defect found by running OpenFold3's parser on the old file."""

    @pytest.mark.parametrize("path, chain", CASES)
    def test_the_entity_ids_are_integers(self, tmp_path, path, chain):
        _, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        for tag in ("_atom_site.label_entity_id", "_struct_asym.entity_id", "_entity.id"):
            values = _column(block, tag)
            assert values and all(value == "1" for value in values), tag

    @pytest.mark.parametrize("path, chain", CASES)
    def test_entity_poly_seq_lists_the_residues_in_order(self, tmp_path, path, chain):
        sequence, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        numbers = [int(n) for n in _column(block, "_entity_poly_seq.num")]
        assert numbers == list(range(1, len(sequence) + 1))
        assert set(_column(block, "_entity_poly_seq.entity_id")) == {"1"}
        assert len(_column(block, "_entity_poly_seq.mon_id")) == len(sequence)

    @pytest.mark.parametrize("path, chain", CASES)
    def test_the_canonical_sequence_is_kept(self, tmp_path, path, chain):
        sequence, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        assert _column(block, "_entity_poly.pdbx_seq_one_letter_code_can") == [sequence]
        assert _column(block, "_entity_poly.entity_id") == ["1"]

    @pytest.mark.parametrize("path, chain", CASES)
    def test_label_seq_id_counts_the_residues_from_one(self, tmp_path, path, chain):
        """The A3M indexes the template residues 1..N; author numbers (17..29) would not match."""
        sequence, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        label_seq = _column(block, "_atom_site.label_seq_id")
        assert "." not in label_seq and "?" not in label_seq
        assert sorted({int(n) for n in label_seq}) == list(range(1, len(sequence) + 1))
        # the numbers follow the order of the residues in the file
        in_order = [int(n) for n in label_seq]
        assert in_order == sorted(in_order)

    @pytest.mark.parametrize("path, chain", CASES)
    def test_every_component_has_a_protein_type(self, tmp_path, path, chain):
        _, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        types = _column(block, "_chem_comp.type")
        assert types and set(types) <= PROTEIN_COMPONENT_TYPES
        assert "." not in types

    @pytest.mark.parametrize("path, chain", CASES)
    def test_the_release_date_and_the_chain_to_entity_table_are_there(self, tmp_path, path, chain):
        _, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        assert _column(block, "_pdbx_audit_revision_history.revision_date") == ["1900-01-01"]
        assert _column(block, "_pdbx_poly_seq_scheme.asym_id") == [chain]
        assert _column(block, "_pdbx_poly_seq_scheme.entity_id") == ["1"]

    @pytest.mark.parametrize("path, chain", CASES)
    def test_the_chain_keeps_its_own_id_as_label_asym_id(self, tmp_path, path, chain):
        _, _, out = _written(path, chain, tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        assert set(_column(block, "_atom_site.label_asym_id")) == {chain}
        assert set(_column(block, "_atom_site.auth_asym_id")) == {chain}
        assert _column(block, "_struct_asym.id") == [chain]


class TestReadAsBiotiteReadsIt:
    """OpenFold3 builds its atom array with biotite, from the label fields."""

    @pytest.mark.parametrize("path, chain", CASES)
    def test_the_label_residue_ids_run_from_one_to_the_sequence_length(self, tmp_path, path, chain):
        sequence, _, out = _written(path, chain, tmp_path)
        cif = pdbx.CIFFile.read(str(out))
        atoms = pdbx.get_structure(
            cif, model=1, use_author_fields=False, extra_fields=["label_entity_id"]
        )
        residue_ids = sorted(set(int(i) for i in atoms.res_id))
        assert residue_ids == list(range(1, len(sequence) + 1))
        assert set(atoms.chain_id) == {chain}
        assert set(int(e) for e in atoms.label_entity_id) == {1}

    def test_waters_and_other_hetero_groups_of_the_chain_are_not_written(self, tmp_path):
        """1CWA chain C holds 11 residues and 4 waters; the query has 11 letters."""
        sequence, _, out = _written(CYCLOSPORIN, "C", tmp_path)
        atoms = pdbx.get_structure(pdbx.CIFFile.read(str(out)), model=1, use_author_fields=False)
        assert len(sequence) == 11
        assert "HOH" not in set(atoms.res_name)


class TestNonCanonicalBinder:
    def test_the_components_are_the_ccd_codes_of_the_query(self, tmp_path):
        _, non_canonical, out = _written(CYCLOSPORIN, "C", tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        names = _column(block, "_entity_poly_seq.mon_id")
        for position, code in non_canonical.items():
            assert names[position - 1] == code

    def test_d_amino_acids_and_achiral_residues_get_their_ccd_types(self, tmp_path):
        _, _, out = _written(CYCLOSPORIN, "C", tmp_path)
        block = gemmi.cif.read(str(out)).sole_block()
        types = dict(zip(_column(block, "_chem_comp.id"), _column(block, "_chem_comp.type")))
        assert types["DAL"] == "D-PEPTIDE LINKING"
        assert types["SAR"] == "PEPTIDE LINKING"
        assert types["MLE"] == "L-PEPTIDE LINKING"
        assert types["VAL"] == "L-PEPTIDE LINKING"


class TestResiduesThatTheQueryRenames:
    """Variants are sent as their parent residue; the template says the same in CCD names."""

    def test_variants_caps_and_waters(self, tmp_path):
        ace = ("ACE", (("C", "C"), ("O", "O"), ("CH3", "C")))
        nme = ("NME", (("N", "N"), ("C", "C")))
        water = ("HOH", (("O", "O"),))
        structure = _structure(
            {"A": [ace, "ALA", "HID", "CYX", "GLY", nme, water], "B": ["LYS", "ARG"]}
        )
        path = tmp_path / "variants.pdb"
        structure.write_pdb(str(path))
        sequence, _, out = _written(path, "A", tmp_path)
        assert sequence == "AHCG"
        block = gemmi.cif.read(str(out)).sole_block()
        assert _column(block, "_entity_poly_seq.mon_id") == ["ALA", "HIS", "CYS", "GLY"]
        assert set(_column(block, "_atom_site.label_comp_id")) == {"ALA", "HIS", "CYS", "GLY"}
        assert _column(block, "_entity_poly.pdbx_seq_one_letter_code_can") == ["AHCG"]

    def test_n_methyl_template_names_become_ccd_codes(self, tmp_path):
        path = _write(tmp_path, {"A": ["ALA", "NMG", "NMA", "GLY"], "B": ["LYS"]}, "n.pdb")
        sequence, non_canonical, out = _written(path, "A", tmp_path)
        assert sequence == "AGAG"
        assert non_canonical == {2: "SAR", 3: "MAA"}
        block = gemmi.cif.read(str(out)).sole_block()
        assert _column(block, "_entity_poly_seq.mon_id") == ["ALA", "SAR", "MAA", "GLY"]

    def test_the_source_structure_is_not_changed(self, tmp_path):
        structure = _structure({"A": ["ALA", "HID", ("HOH", (("O", "O"),))], "B": ["LYS"]})
        before = [(r.name, r.label_seq, r.subchain) for r in structure[0]["A"]]
        _openfold_run._extract_chain_to_cif(structure, "A", tmp_path / "t.cif", sequence="AH")
        assert [(r.name, r.label_seq, r.subchain) for r in structure[0]["A"]] == before


class TestWhatIsRefused:
    def test_a_sequence_of_another_length_than_the_residues_writes_nothing(self, tmp_path):
        structure = gemmi.read_structure(str(P53))
        out = tmp_path / "t.cif"
        with pytest.raises(
            ValueError, match="13 amino-acid residues but the query sequence has 12"
        ):
            _openfold_run._extract_chain_to_cif(structure, "B", out, sequence="ETFSDLWKLLPE")
        assert not out.exists()

    def test_a_chain_without_amino_acids_writes_nothing(self, tmp_path):
        water = ("HOH", (("O", "O"),))
        structure = _structure({"A": ["ALA"], "W": [water, water]})
        out = tmp_path / "t.cif"
        with pytest.raises(ValueError, match="no amino-acid residue"):
            _openfold_run._extract_chain_to_cif(structure, "W", out)
        assert not out.exists()

    def test_without_a_sequence_it_is_read_from_the_residues(self, tmp_path):
        structure = gemmi.read_structure(str(P53))
        out = tmp_path / "t.cif"
        _openfold_run._extract_chain_to_cif(structure, "B", out)
        block = gemmi.cif.read(str(out)).sole_block()
        assert _column(block, "_entity_poly.pdbx_seq_one_letter_code_can") == ["ETFSDLWKLLPEN"]


class TestQueryBuildersWriteTheRepairedFiles:
    @pytest.mark.parametrize(
        "function", [openfold.prepare_scoring_query, openfold.prepare_refolding_query]
    )
    def test_the_receptor_template_has_its_sequence_and_numbering(self, tmp_path, function):
        function(P53, "A", "B", "q", tmp_path, binder_cyclic=False)
        block = gemmi.cif.read(str(tmp_path / "templates" / "receptor.cif")).sole_block()
        assert len(_column(block, "_entity_poly.pdbx_seq_one_letter_code_can")[0]) == 85
        assert _column(block, "_atom_site.label_entity_id")[0] == "1"

    def test_the_binder_template_of_a_score_query(self, tmp_path):
        openfold.prepare_scoring_query(CYCLOSPORIN, "A", "C", "q", tmp_path, binder_cyclic=False)
        block = gemmi.cif.read(str(tmp_path / "templates" / "binder.cif")).sole_block()
        assert _column(block, "_entity_poly.pdbx_seq_one_letter_code_can") == ["ALLVTAGLVLA"]

    def test_the_batched_queries_write_it_too(self, tmp_path):
        samples = [openfold._BatchSample("s1", P53, "A", "B")]
        openfold.prepare_batched_scoring_queries(samples, tmp_path, binder_cyclic=False)
        for name in ("s1rec.cif", "s1bnd.cif"):
            block = gemmi.cif.read(str(tmp_path / "templates" / name)).sole_block()
            assert _column(block, "_entity_poly_seq.num")[0] == "1"


class TestTheTemplateCacheKeyFollowsTheWriter:
    """OpenFold3 keys its template cache on the chain sequence and the content of the A3M only."""

    def _a3m(self, tmp_path):
        out = tmp_path / "x.a3m"
        _openfold_run._write_a3m_self_alignment("ETFSDLWKLLPEN", "query_B", "binder", "B", out)
        return out.read_text(encoding="utf-8")

    def test_the_query_row_names_the_builder_version(self, tmp_path):
        text = self._a3m(tmp_path)
        version = _openfold_run.QUERY_BUILDER_VERSION
        assert text.splitlines()[0] == f">query-b{version}_B/1-13"
        assert text.splitlines()[2] == ">binder_B/1-13"

    def test_the_header_still_splits_in_two_as_openfold3_reads_it(self, tmp_path):
        """OpenFold3 does ``entry_id, chain_id = header.split('_')`` on every row."""
        for header in self._a3m(tmp_path).splitlines()[::2]:
            entry, chain = header.lstrip(">").split("/")[0].split("_")
            assert chain == "B"
            assert entry

    def test_another_builder_version_gives_another_file(self, tmp_path, monkeypatch):
        before = self._a3m(tmp_path)
        monkeypatch.setattr(
            _openfold_run, "QUERY_BUILDER_VERSION", _openfold_run.QUERY_BUILDER_VERSION + 1
        )
        assert self._a3m(tmp_path) != before

    def test_the_version_is_importable_from_the_openfold_module(self):
        assert openfold.QUERY_BUILDER_VERSION is _openfold_run.QUERY_BUILDER_VERSION
        assert isinstance(openfold.QUERY_BUILDER_VERSION, int)


# ---------------------------------------------------------------------------
# OpenFold3's own parser (optional)
# ---------------------------------------------------------------------------

#: What the probe runs in the OpenFold3 environment: the functions the template preprocessor
#: and the data loader of 0.5.0 call on a template CIF, on CPU, without the model.
_PROBE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from openfold3.core.data.io.structure.cif import _load_ciffile
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict, get_cif_block, get_release_date)
from openfold3.core.data.primitives.structure.template import parse_template_structure
directory, entry, chain = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cif = _load_ciffile(directory / f"{entry}.cif")
sequences = get_asym_id_to_canonical_seq_dict(cif)
release = str(get_release_date(get_cif_block(cif)).date())
atoms = parse_template_structure(directory, None, f"{entry}_{chain}", "cif",
                                 BiotiteCCDWrapper(), cif_assembly_cache={})
ids = np.unique(atoms.res_id.astype(int))
print("PROBE " + json.dumps({"sequences": sequences, "release": release,
      "first": int(ids.min()), "last": int(ids.max()), "n": int(len(ids)),
      "chains": sorted({str(c) for c in atoms.chain_id})}))
"""


def _openfold3_python():
    """The command that starts Python in the OpenFold3 environment, or a reason to skip."""
    import os

    env = os.environ.get("BM_OPENFOLD3_ENV", "openfold3")
    conda = shutil.which("conda")
    if conda is None:
        pytest.skip("conda is not on PATH")
    try:
        check = subprocess.run(
            [conda, "run", "-n", env, "python", "-c", "import openfold3"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"cannot start conda run -n {env}: {exc}")
    if check.returncode != 0:
        pytest.skip(f"the conda environment {env!r} has no openfold3")
    return [conda, "run", "-n", env, "python"]


@pytest.mark.openfold
@pytest.mark.parametrize("path, chain", CASES)
def test_openfold3_parses_the_written_file(tmp_path, path, chain):
    """The parsers of OpenFold3 itself accept the file and see residues 1..N (CPU only)."""
    python = _openfold3_python()
    sequence, _, written = _written(path, chain, tmp_path)
    directory = tmp_path / "templates"
    directory.mkdir()
    shutil.copy(written, directory / "tpl.cif")
    probe = subprocess.run(
        [*python, "-c", _PROBE, str(directory), "tpl", chain],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    lines = [line for line in probe.stdout.splitlines() if line.startswith("PROBE ")]
    assert lines, probe.stdout[-2000:]
    found = json.loads(lines[-1][len("PROBE ") :])
    assert found["sequences"] == {chain: sequence}
    assert found["release"] == "1900-01-01"
    assert (found["first"], found["last"], found["n"]) == (1, len(sequence), len(sequence))
    assert found["chains"] == [chain]
