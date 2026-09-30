"""Template entry IDs of batched OpenFold3 queries (issue #86).

OpenFold3 finds a template as ``<structure_directory>/<entry>.cif`` from the entry name in
the A3M header. Two samples that map to one entry name overwrite each other's CIF, and one
query is then predicted from the other sample's template.
"""

import re

import pytest

gemmi = pytest.importorskip("gemmi")

from binding_metrics.metrics import _openfold_run, openfold  # noqa: E402
from binding_metrics.metrics._openfold_run import (  # noqa: E402
    _BatchSample,
    _safe_entry_id,
    _unique_entry_ids,
)
from tests.test_of3_synth import _write  # noqa: E402


class TestUniqueEntryIds:
    def test_ids_that_do_not_collide_are_the_old_ids(self):
        ids = ["s1", "s_2", "sample-3", "x_y_z"]
        assert _unique_entry_ids(ids, "rec") == {sid: _safe_entry_id(sid, "rec") for sid in ids}
        assert _unique_entry_ids(["s_2"], "bnd") == {"s_2": "s-2bnd"}

    def test_underscore_and_hyphen_no_longer_collide(self):
        entries = _unique_entry_ids(["a_b", "a-b", "other"], "rec")
        assert len(set(entries.values())) == 3
        assert entries["other"] == "otherrec"
        assert entries["a_b"] != entries["a-b"]

    def test_the_colliding_ids_keep_their_readable_stem_and_role_suffix(self):
        entries = _unique_entry_ids(["a_b", "a-b"], "bnd")
        for entry in entries.values():
            assert re.fullmatch(r"a-b-[0-9a-f]{8}bnd", entry)

    def test_every_entry_is_valid_for_openfold3(self):
        """OpenFold3 splits ``<entry>_<chain>`` on the underscore, so entries have none."""
        entries = _unique_entry_ids(["a_b", "a-b", "a_b_c", "a-b-c", "a_b-c", "a-b_c"], "rec")
        assert all("_" not in entry and "/" not in entry for entry in entries.values())
        assert len(set(entries.values())) == 6

    def test_case_differences_collide_on_case_insensitive_file_systems(self):
        entries = _unique_entry_ids(["Sample", "sample", "third"], "rec")
        assert len({entry.casefold() for entry in entries.values()}) == 3
        assert entries["third"] == "thirdrec"

    def test_an_id_does_not_depend_on_the_order_of_the_batch(self):
        forward = _unique_entry_ids(["a_b", "a-b"], "rec")
        backward = _unique_entry_ids(["a-b", "a_b"], "rec")
        assert forward == backward

    def test_a_repeated_sample_id_is_one_sample(self):
        assert _unique_entry_ids(["a_b", "a_b"], "rec") == {"a_b": "a-brec"}

    def test_an_unresolvable_clash_is_reported(self, monkeypatch):
        """If even the widest hash clashed, the run must stop rather than overwrite a CIF."""

        class _Constant:
            def __init__(self, data):
                pass

            def hexdigest(self):
                return "0" * 64

        monkeypatch.setattr(_openfold_run.hashlib, "sha256", _Constant)
        with pytest.raises(ValueError, match="distinct template entry IDs"):
            _unique_entry_ids(["a_b", "a-b"], "rec")

    def test_safe_entry_id_is_unchanged(self):
        assert _safe_entry_id("a_b", "rec") == "a-brec"


def _receptor_sequence(cif_path) -> str:
    block = gemmi.cif.read(str(cif_path)).sole_block()
    return block.find_value("_entity_poly.pdbx_seq_one_letter_code_can").strip("'\"")


def _template_entry(a3m_path) -> str:
    """Entry name of the template line of a self-alignment, as OpenFold3 reads it."""
    header = a3m_path.read_text(encoding="utf-8").splitlines()[2]
    return header.lstrip(">").split("/")[0].rsplit("_", 1)[0]


class TestBatchedQueries:
    @pytest.fixture
    def colliding_samples(self, tmp_path):
        first = _write(tmp_path, {"A": ["ALA", "GLY", "SER"], "B": ["LYS", "ARG"]}, "first.pdb")
        second = _write(tmp_path, {"A": ["TRP", "TYR", "PHE", "VAL"], "B": ["ASP"]}, "second.pdb")
        return [
            _BatchSample("a_b", first, "A", "B"),
            _BatchSample("a-b", second, "A", "B"),
            _BatchSample("plain", first, "A", "B"),
        ]

    @pytest.mark.parametrize(
        "function, roles",
        [
            (openfold.prepare_batched_scoring_queries, ("receptor", "binder")),
            (openfold.prepare_batched_refolding_queries, ("receptor",)),
        ],
    )
    def test_each_sample_keeps_its_own_template(self, tmp_path, colliding_samples, function, roles):
        out = tmp_path / "out"
        function(colliding_samples, out)
        expected_sequences = {"a_b": "AGS", "a-b": "WYFV", "plain": "AGS"}
        for sample in colliding_samples:
            entry = _template_entry(out / f"{sample.query_name}_receptor.a3m")
            cif = out / "templates" / f"{entry}.cif"
            assert cif.exists(), f"{sample.query_name}: no {cif.name}"
            assert _receptor_sequence(cif) == expected_sequences[sample.query_name]
        files = sorted(p.name for p in (out / "templates").glob("*.cif"))
        assert len(files) == len(set(files)) == len(colliding_samples) * len(roles)

    def test_the_binder_templates_of_colliding_samples_differ_too(
        self, tmp_path, colliding_samples
    ):
        out = tmp_path / "out"
        openfold.prepare_batched_scoring_queries(colliding_samples, out)
        sequences = {}
        for name in ("a_b", "a-b"):
            entry = _template_entry(out / f"{name}_binder.a3m")
            sequences[name] = _receptor_sequence(out / "templates" / f"{entry}.cif")
        assert sequences == {"a_b": "KR", "a-b": "D"}

    def test_a_batch_without_collisions_writes_the_same_file_names_as_before(self, tmp_path):
        complex_path = _write(tmp_path, {"A": ["ALA", "GLY"], "B": ["SER", "LYS"]})
        samples = [
            _BatchSample("s_1", complex_path, "A", "B"),
            _BatchSample("s2", complex_path, "A", "B"),
        ]
        out = tmp_path / "out"
        openfold.prepare_batched_scoring_queries(samples, out)
        assert sorted(p.name for p in (out / "templates").glob("*.cif")) == [
            "s-1bnd.cif", "s-1rec.cif", "s2bnd.cif", "s2rec.cif",
        ]  # fmt: skip
        assert _template_entry(out / "s_1_receptor.a3m") == "s-1rec"
