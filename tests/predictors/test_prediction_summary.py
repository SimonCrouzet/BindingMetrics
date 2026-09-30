"""summarize_prediction: the result dictionary of a record, and its parity with OpenFold3."""

import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from binding_metrics.metrics import openfold
from binding_metrics.metrics.prediction import summarize_prediction
from binding_metrics.predictors.record import PredictionFiles, PredictionRecord
from tests.predictors import contract, synth, synth_of3
from tests.predictors.test_openfold_golden import _KEYS, _write_reference, _write_run

NAME = contract.NAME
QUERY = "gold"


def _run(tmp_path):
    return _write_run(tmp_path / "run")


class TestParityWithComputeOpenfoldMetrics:
    def test_the_only_difference_is_the_model_key(self, tmp_path):
        root = _run(tmp_path)
        reference = _write_reference(tmp_path)
        kwargs = dict(binder_chain="B", receptor_chain="A", reference_structure_path=reference)
        legacy = openfold.compute_openfold_metrics(root, QUERY, include_matrices=True, **kwargs)
        record = openfold.get_parser("of3").load(root, QUERY)
        summary = summarize_prediction(record, include_matrices=True, **kwargs)
        assert list(summary) == ["model", *_KEYS]
        assert summary.pop("model") == "of3"
        np.testing.assert_equal(summary, legacy)

    def test_keys_of_the_legacy_function_have_no_model(self, tmp_path):
        assert "model" not in openfold.compute_openfold_metrics(_run(tmp_path), QUERY)


class TestOptions:
    def test_target_chain_is_an_alias_of_receptor_chain(self, tmp_path):
        record = openfold.get_parser("of3").load(_run(tmp_path), QUERY)
        via_alias = summarize_prediction(record, binder_chain="B", target_chain="A")
        assert via_alias["mean_interface_pde"] == pytest.approx(1.9375)
        with pytest.raises(ValueError, match="target_chain"):
            summarize_prediction(record, binder_chain="B", receptor_chain="A", target_chain="C")

    def test_without_matrices_the_arrays_are_withheld_but_the_statistics_are_kept(self, tmp_path):
        record = openfold.get_parser("of3").load(_run(tmp_path), QUERY)
        summary = summarize_prediction(record, binder_chain="B", receptor_chain="A")
        assert summary["pde"] is None and summary["pae_interface"] is None
        assert summary["max_pae"] == pytest.approx(5.5)
        assert summary["mean_interface_pae"] == pytest.approx(3.4375)

    def test_a_record_built_by_hand_gives_nan_and_none_for_what_it_lacks(self):
        summary = summarize_prediction(PredictionRecord("boltz2", "q", avg_plddt=81.5))
        assert summary["model"] == "boltz2" and summary["query_name"] == "q"
        assert summary["avg_plddt"] == 81.5
        assert np.isnan(summary["ptm"]) and np.isnan(summary["max_pae"])
        assert summary["bespoke_iptm"] == {} and summary["n_atoms"] == 0
        assert "reason" not in summary

    def test_the_parsers_reasons_come_first_and_stay(self):
        record = PredictionRecord("boltz2", "q", reasons=["pae file not found"])
        assert summarize_prediction(record)["reason"] == "pae file not found"


class TestReasonsForAFileThatWasReadButLacksAValue:
    def _record(self, tmp_path, files):
        atoms_run = _run(tmp_path)
        record = openfold.get_parser("of3").load(atoms_run, QUERY)
        record.plddt_per_atom = None
        record.pde = None
        record.pae = None
        record.files = files
        return record

    def test_a_confidence_file_that_was_read_explains_each_missing_value(self, tmp_path):
        files = PredictionFiles(directory=tmp_path, arrays=tmp_path / "confidences.json")
        summary = summarize_prediction(
            self._record(tmp_path, files), binder_chain="B", receptor_chain="A"
        )
        assert summary["reason"] == (
            "binder pLDDT: no per-atom pLDDT in the confidences file; "
            "interface PDE: no PDE matrix in the confidences file; "
            "interface PAE: no PAE matrix in the confidences file"
        )

    @pytest.mark.parametrize("files", [None, PredictionFiles(directory=Path("."))])
    def test_no_confidence_file_means_the_parser_has_already_said_why(self, tmp_path, files):
        summary = summarize_prediction(
            self._record(tmp_path, files), binder_chain="B", receptor_chain="A"
        )
        assert "reason" not in summary


class TestWarnings:
    def test_the_default_prefix_and_caller(self, tmp_path):
        root = _write_run(tmp_path / "run", plddt=np.full(10, 90.0))
        record = openfold.get_parser("of3").load(root, QUERY)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summarize_prediction(record, binder_chain="B", receptor_chain="A")
        assert [str(w.message).split(":")[0] for w in caught] == ["summarize_prediction"]
        assert Path(caught[0].filename).name == Path(__file__).name

    def test_a_wrapper_names_itself_and_its_own_caller(self, tmp_path):
        root = _write_run(tmp_path / "run", plddt=np.full(10, 90.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            openfold.compute_openfold_metrics(root, QUERY, binder_chain="B", receptor_chain="A")
        assert str(caught[0].message).startswith("compute_openfold_metrics: ")
        assert Path(caught[0].filename).name == Path(__file__).name


class TestReferenceRmsd:
    def test_receptors_of_different_length_give_nan_and_a_reason_not_a_frame_dependent_number(
        self, tmp_path
    ):
        from tests.predictors.test_openfold_golden import _dimer, _write_cif

        atoms = _dimer()
        short = atoms[~((atoms.chain_id == "A") & (atoms.res_id == 4))]
        reference = tmp_path / "short_receptor.cif"
        _write_cif(short, reference)
        record = openfold.get_parser("of3").load(_run(tmp_path), QUERY)
        with pytest.warns(UserWarning, match="binder RMSD skipped"):
            summary = summarize_prediction(
                record, binder_chain="B", receptor_chain="A", reference_structure_path=reference
            )
        assert np.isnan(summary["binder_ca_rmsd"])
        assert summary["reason"].startswith("binder RMSD: Receptor Cα count mismatch (chain 'A')")

    def test_without_a_receptor_chain_the_binder_shape_is_compared(self, tmp_path):
        record = openfold.get_parser("of3").load(_run(tmp_path), QUERY)
        reference = _write_reference(tmp_path, binder_shift_z=1.0)
        summary = summarize_prediction(record, binder_chain="B", reference_structure_path=reference)
        assert summary["binder_ca_rmsd"] == pytest.approx(0.0, abs=1e-4)  # a rigid shift only


class TestMissingBiotite:
    def test_the_install_hint_names_the_structural_analysis(self, tmp_path):
        root = _run(tmp_path)
        blocked = {
            name: None
            for name in (
                "biotite",
                "biotite.structure",
                "biotite.structure.io",
                "biotite.structure.io.pdb",
                "biotite.structure.io.pdbx",
            )
        }
        with mock.patch.dict(sys.modules, blocked):
            with pytest.warns(UserWarning, match="structural analysis failed"):
                metrics = openfold.compute_openfold_metrics(
                    root, QUERY, binder_chain="B", receptor_chain="A"
                )
        assert "biotite is required for per-chain structural analysis" in metrics["reason"]
        assert metrics["avg_plddt"] == pytest.approx(87.5)


class TestOpenFold30Layouts:
    """What the adapter adds for OpenFold3 0.5.0, seen through compute_openfold_metrics."""

    def test_a_gzipped_structure_gives_the_same_numbers(self, tmp_path, monkeypatch):
        synth_of3.write_prediction(tmp_path / "plain", NAME, synth.synthetic_complex())
        monkeypatch.setattr(synth_of3, "STRUCTURE_SUFFIX", ".cif.gz")
        synth_of3.write_prediction(tmp_path / "gz", NAME, synth.synthetic_complex())
        plain = openfold.compute_openfold_metrics(
            tmp_path / "plain", NAME, binder_chain="B", receptor_chain="A"
        )
        packed = openfold.compute_openfold_metrics(
            tmp_path / "gz", NAME, binder_chain="B", receptor_chain="A"
        )
        assert packed["structure_path"].endswith("_model.cif.gz")
        assert "reason" not in packed
        assert packed["binder_avg_plddt"] == pytest.approx(76.0)
        for key in ("mean_interface_pde", "mean_interface_pae", "binder_avg_plddt"):
            assert packed[key] == pytest.approx(plain[key]), key

    def test_seed_positions_are_numeric(self, tmp_path):
        synth_of3.write_prediction(tmp_path, NAME, synth.synthetic_complex())  # seed_9
        synth_of3.write_prediction(
            tmp_path, NAME, synth.synthetic_complex(plddt_shift=10.0), seed_index=2
        )  # seed_10
        first = openfold.compute_openfold_metrics(tmp_path, NAME, seed=1)
        second = openfold.compute_openfold_metrics(tmp_path, NAME, seed=2)
        assert (first["avg_plddt"], second["avg_plddt"]) == (82.0, 72.0)
        assert (first["seed"], second["seed"]) == (1, 2)

    def test_npz_confidences(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_of3, "CONFIDENCE_FORMAT", "npz")
        synth_of3.write_prediction(tmp_path, NAME, synth.synthetic_complex())
        metrics = openfold.compute_openfold_metrics(
            tmp_path, NAME, binder_chain="B", receptor_chain="A"
        )
        assert metrics["mean_interface_pae"] == pytest.approx(3.4375)

    def test_interface_pae_reads_npz_without_pickle(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_of3, "CONFIDENCE_FORMAT", "npz")
        synth_of3.write_prediction(tmp_path, NAME, synth.synthetic_complex())
        seed_dir = tmp_path / NAME / "seed_9"
        result = openfold.compute_interface_pae(
            seed_dir / f"{NAME}_seed_9_sample_1_confidences.npz",
            seed_dir / f"{NAME}_seed_9_sample_1_model.cif",
            binder_chain="B",
            receptor_chain="A",
        )
        assert result["mean_interface_pae"] == pytest.approx(3.4375)
