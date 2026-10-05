"""Failures in the CIF helpers are logged with a cause instead of swallowed.

Each helper used to fall back silently (`except Exception: pass`). The fallback
is kept, because the caller can still proceed, but the reason is now logged and
only the exceptions the fallback was written for are caught.
"""

import logging
import sys
from pathlib import Path

import pytest

pdbx = pytest.importorskip("biotite.structure.io.pdbx")

from binding_metrics.io.structures import (  # noqa: E402
    detect_chains_from_file,
    detect_models,
    load_structure,
    merge_cif_models,
    save_cif,
)
from binding_metrics.utils import backfill_auth_columns  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
SFTI_CIF = DATA / "example_bicyclic_sfti1_3P8F.cif"
STRUCTURES_LOGGER = "binding_metrics.io.structures"
UTILS_LOGGER = "binding_metrics.utils"


@pytest.fixture
def sfti_cif():
    if not SFTI_CIF.exists():
        pytest.skip(f"bundled example not found: {SFTI_CIF}")
    return SFTI_CIF


def _cif_without_column(source: Path, column: str, target: Path) -> Path:
    cif = pdbx.CIFFile.read(str(source))
    del cif.block["atom_site"][column]
    cif.write(str(target))
    return target


def _records(caplog, name):
    return [r for r in caplog.records if r.name == name]


class TestBackfillAuthColumns:
    def test_missing_source_columns_are_logged_at_debug_and_left_alone(self, caplog):
        class NoAtomSite:
            @property
            def block(self):
                raise KeyError("atom_site")

        with caplog.at_level(logging.DEBUG, logger=UTILS_LOGGER):
            backfill_auth_columns(NoAtomSite())
        records = _records(caplog, UTILS_LOGGER)
        assert len(records) == 1 and records[0].levelno == logging.DEBUG
        assert "atom_site" in records[0].getMessage()

    def test_several_blocks_are_logged_not_raised(self, caplog):
        class ManyBlocks:
            @property
            def block(self):
                raise ValueError("There are multiple blocks in the file")

        with caplog.at_level(logging.DEBUG, logger=UTILS_LOGGER):
            backfill_auth_columns(ManyBlocks())
        assert "multiple blocks" in _records(caplog, UTILS_LOGGER)[0].getMessage()

    def test_unexpected_errors_are_not_swallowed(self):
        class Broken:
            @property
            def block(self):
                raise RuntimeError("disk on fire")

        with pytest.raises(RuntimeError, match="disk on fire"):
            backfill_auth_columns(Broken())

    def test_missing_auth_columns_are_filled_from_label_columns(self, sfti_cif, tmp_path):
        stripped = _cif_without_column(sfti_cif, "auth_atom_id", tmp_path / "s.cif")
        cif = pdbx.CIFFile.read(str(stripped))
        assert "auth_atom_id" not in cif.block["atom_site"]
        backfill_auth_columns(cif)
        site = cif.block["atom_site"]
        assert site["auth_atom_id"].as_array().tolist() == site["label_atom_id"].as_array().tolist()


class TestDetectModels:
    def test_unreadable_file_warns_and_returns_one(self, tmp_path, caplog):
        bad = tmp_path / "garbage.cif"
        bad.write_text("this is not a CIF file {{{", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            assert detect_models(bad) == [1]
        records = _records(caplog, STRUCTURES_LOGGER)
        assert len(records) == 1 and "garbage.cif" in records[0].getMessage()

    def test_non_integer_model_numbers_warn(self, tmp_path, caplog):
        bad = tmp_path / "models.cif"
        bad.write_text(
            "data_x\nloop_\n_atom_site.id\n_atom_site.pdbx_PDB_model_num\n1 first\n2 second\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            assert detect_models(bad) == [1]
        assert "assuming a single model" in _records(caplog, STRUCTURES_LOGGER)[0].getMessage()

    def test_missing_model_column_is_a_normal_single_model_file(self, tmp_path, caplog):
        plain = tmp_path / "single.cif"
        plain.write_text(
            "data_x\nloop_\n_atom_site.id\n_atom_site.label_asym_id\n1 A\n", encoding="utf-8"
        )
        with caplog.at_level(logging.DEBUG, logger=STRUCTURES_LOGGER):
            assert detect_models(plain) == [1]
        assert not _records(caplog, STRUCTURES_LOGGER)

    def test_pdb_file_is_single_model(self, tmp_path):
        assert detect_models(tmp_path / "anything.pdb") == [1]

    def test_multi_model_file_lists_every_model(self, sfti_cif, tmp_path, caplog):
        pytest.importorskip("gemmi")
        merged = tmp_path / "multi.cif"
        merge_cif_models([(1, sfti_cif), (2, sfti_cif)], merged)
        with caplog.at_level(logging.DEBUG, logger=STRUCTURES_LOGGER):
            assert detect_models(merged) == [1, 2]
        assert not _records(caplog, STRUCTURES_LOGGER)

    def test_missing_biotite_warns_instead_of_hiding_models(self, sfti_cif, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "biotite", None)
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            assert detect_models(sfti_cif) == [1]
        assert "biotite is not installed" in _records(caplog, STRUCTURES_LOGGER)[0].getMessage()


class TestDetectChainsLabelMapping:
    def test_label_ids_come_from_the_label_column(self, sfti_cif, caplog):
        """3P8F: author chain I is label chain B, so OpenMM-based steps need 'B'."""
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            info = detect_chains_from_file(sfti_cif, verbose=False)
        assert info["peptide_chain"] == "I"
        assert info["peptide_chain_label"] == "B"
        assert info["receptor_chain"] == info["receptor_chain_label"] == "A"
        assert not _records(caplog, STRUCTURES_LOGGER)

    def test_missing_label_column_warns_and_falls_back_to_author_ids(
        self, sfti_cif, tmp_path, caplog
    ):
        stripped = _cif_without_column(sfti_cif, "label_asym_id", tmp_path / "nolabel.cif")
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            info = detect_chains_from_file(stripped, verbose=False)
        assert info["peptide_chain_label"] == info["peptide_chain"] == "I"
        records = _records(caplog, STRUCTURES_LOGGER)
        assert len(records) == 1
        assert "label_asym_id" in records[0].getMessage()


class TestSaveCifSourceColumns:
    def test_source_without_auth_atom_id_warns_and_still_saves(self, sfti_cif, tmp_path, caplog):
        pytest.importorskip("gemmi")
        source = _cif_without_column(sfti_cif, "auth_atom_id", tmp_path / "source.cif")
        topology, positions = load_structure(sfti_cif)
        out = tmp_path / "out.cif"
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            save_cif(topology, positions, out, source_cif_path=source)
        records = _records(caplog, STRUCTURES_LOGGER)
        assert len(records) == 1
        message = records[0].getMessage()
        assert "auth_atom_id" in message and "source.cif" in message
        reloaded, _ = load_structure(out)
        assert reloaded.getNumAtoms() == topology.getNumAtoms()

    def test_complete_source_is_quiet(self, sfti_cif, tmp_path, caplog):
        pytest.importorskip("gemmi")
        topology, positions = load_structure(sfti_cif)
        with caplog.at_level(logging.WARNING, logger=STRUCTURES_LOGGER):
            save_cif(topology, positions, tmp_path / "out.cif", source_cif_path=sfti_cif)
        assert not _records(caplog, STRUCTURES_LOGGER)
