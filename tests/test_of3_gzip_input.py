"""A gzip-compressed complex is read by the pre-flight and by ``binder_cyclic="auto"``.

The OpenFold3 query builders read the complex with gemmi, which decompresses ``.cif.gz`` and
``.pdb.gz`` itself; ``capabilities._read_atoms`` (biotite) used to parse any file whose name did
not end in ``.cif`` or ``.mmcif`` as PDB, so a ``.cif.gz`` failed there with ``OSError``.
"""

import gzip
import json
import logging
import shutil
from pathlib import Path

import pytest

from binding_metrics.capabilities import _read_atoms as read_atoms
from binding_metrics.capabilities import detect_closures, profile_input
from binding_metrics.metrics import _openfold_run, openfold

pytest.importorskip("biotite")
pytest.importorskip("gemmi")

DATA = Path(__file__).parent.parent / "data"
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"  # chain I: head to tail, standard residues
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"  # chain C: head-to-tail
P53 = DATA / "example_linear_p53_1YCR.pdb"  # chain B: linear


def gzip_copy(source: Path, directory: Path, name: str) -> Path:
    target = directory / name
    with open(source, "rb") as plain, gzip.open(target, "wb") as packed:
        shutil.copyfileobj(plain, packed)
    return target


@pytest.fixture(autouse=True)
def _openfold3_version(monkeypatch):
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )
    monkeypatch.setattr(_openfold_run, "_VERSION_BY_PYTHON", {})


@pytest.mark.parametrize("name", ["c.cif.gz", "c.mmcif.gz", "C.CIF.GZ"])
def test_a_gzipped_mmcif_gives_the_profile_of_the_plain_file(tmp_path, name):
    packed = gzip_copy(CYCLOSPORIN, tmp_path, name)
    profile = profile_input(packed, "C", "A")
    plain = profile_input(CYCLOSPORIN, "C", "A")
    assert profile.closures == plain.closures == frozenset({"head_to_tail"})
    assert profile.n_binder_residues == plain.n_binder_residues
    assert profile.chain_ids == plain.chain_ids


def test_a_gzipped_pdb_gives_the_profile_of_the_plain_file(tmp_path):
    packed = gzip_copy(P53, tmp_path, "p.pdb.gz")
    assert profile_input(packed, "B", "A").closures == frozenset({"none"})
    assert (
        profile_input(packed, "B", "A").n_binder_residues
        == profile_input(P53, "B", "A").n_binder_residues
    )


def test_the_atoms_of_a_gzip_equal_those_of_the_plain_file(tmp_path):
    packed = gzip_copy(CYCLOSPORIN, tmp_path, "c.cif.gz")
    gz_atoms, plain_atoms = read_atoms(packed), read_atoms(CYCLOSPORIN)
    assert gz_atoms.array_length() == plain_atoms.array_length()
    assert [c.kind for c in detect_closures(gz_atoms, "C")] == ["head_to_tail"]


def test_a_missing_gzip_is_a_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_atoms(tmp_path / "missing.cif.gz")


def test_decide_binder_cyclic_auto_sees_the_bond_in_a_gzip(tmp_path):
    packed = gzip_copy(SFTI1, tmp_path, "s.cif.gz")
    decision = openfold.decide_binder_cyclic(packed, "I")
    assert decision == openfold.BinderCyclicDecision(True)
    assert openfold.decide_binder_cyclic(gzip_copy(P53, tmp_path, "p.pdb.gz"), "B").cyclic is False


def test_decide_binder_cyclic_auto_sees_the_modified_residues_in_a_gzip(tmp_path):
    packed = gzip_copy(CYCLOSPORIN, tmp_path, "c.cif.gz")
    decision = openfold.decide_binder_cyclic(packed, "C")
    assert decision.cyclic is False and "modified residues (ABA, BMT, DAL" in decision.reason


def test_a_cyclic_binder_in_a_gzip_gets_the_flag_and_no_warning(tmp_path, caplog):
    packed = gzip_copy(SFTI1, tmp_path, "s.cif.gz")
    with caplog.at_level(logging.WARNING, logger=_openfold_run.logger.name):
        query = openfold.prepare_refolding_query(packed, "A", "I", "q", tmp_path / "out")
    chains = {
        c["chain_ids"][0]: c
        for c in json.loads(query.read_text(encoding="utf-8"))["queries"]["q"]["chains"]
    }
    assert chains["I"]["cyclic"] is True and "cyclic" not in chains["A"]
    assert caplog.records == [] or all(r.levelno < logging.WARNING for r in caplog.records)
