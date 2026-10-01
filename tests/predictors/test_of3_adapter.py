"""The OpenFold3 adapter: file discovery, parsing, and the 0.5.0 conventions."""

import json

import numpy as np
import pytest

from binding_metrics.predictors import of3
from binding_metrics.predictors.of3 import OpenFold3Parser
from binding_metrics.predictors.registry import PARSERS, get_parser
from tests.predictors import contract, synth, synth_of3

NAME = contract.NAME


def _write(tmp_path, **kwargs):
    synth_of3.write_prediction(tmp_path, NAME, synth.synthetic_complex(), **kwargs)
    return tmp_path


def _seed_dir(tmp_path, seed):
    return tmp_path / NAME / f"seed_{seed}"


class TestRegistration:
    def test_of3_is_registered_and_lazy(self):
        spec = PARSERS["of3"]
        assert spec.import_path == "binding_metrics.predictors.of3:OpenFold3Parser"
        assert isinstance(get_parser("of3"), OpenFold3Parser)
        assert (spec.display_name, spec.family) == ("OpenFold3", "af3")

    def test_the_input_limits_are_declared_in_tests_pre_openfold3_limits(self):
        # tests/test_pre_openfold3_limits.py holds the contents of the declaration
        from binding_metrics.capabilities import Capabilities

        assert isinstance(OpenFold3Parser.capabilities, Capabilities)


class TestFindFiles:
    def test_locates_the_files_of_one_sample(self, tmp_path):
        _write(tmp_path)
        files = OpenFold3Parser().find_files(tmp_path, NAME)
        seed_dir = _seed_dir(tmp_path, 9)
        assert files.structure == seed_dir / f"{NAME}_seed_9_sample_1_model.cif"
        assert files.scores == seed_dir / f"{NAME}_seed_9_sample_1_confidences_aggregated.json"
        assert files.arrays == seed_dir / f"{NAME}_seed_9_sample_1_confidences.json"
        assert files.timing == seed_dir / "timing.json"
        assert files.directory == tmp_path

    def test_nothing_is_found_in_an_empty_or_absent_directory(self, tmp_path):
        parser = OpenFold3Parser()
        assert not parser.find_files(tmp_path, NAME).any_found()
        assert not parser.find_files(tmp_path / "nowhere", NAME).any_found()

    def test_the_sample_number_selects_the_files_and_counts_from_one(self, tmp_path):
        _write(tmp_path, sample=1)
        _write(tmp_path, sample=2)
        parser = OpenFold3Parser()
        assert parser.find_files(tmp_path, NAME, sample=2).structure.name.endswith(
            "_sample_2_model.cif"
        )
        assert not parser.find_files(tmp_path, NAME, sample=0).has_output()
        assert not parser.find_files(tmp_path, NAME, sample=3).has_output()

    @pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".pdb"])
    def test_structure_formats(self, tmp_path, monkeypatch, suffix):
        monkeypatch.setattr(synth_of3, "STRUCTURE_SUFFIX", suffix)
        _write(tmp_path)
        files = OpenFold3Parser().find_files(tmp_path, NAME)
        assert files.structure.name == f"{NAME}_seed_9_sample_1_model{suffix}"

    def test_structure_preference_is_cif_then_gzipped_cif_then_pdb(self, tmp_path):
        seed_dir = _seed_dir(tmp_path, 5)
        seed_dir.mkdir(parents=True)
        parser = OpenFold3Parser()
        for suffix in (".pdb", ".cif.gz", ".cif"):
            (seed_dir / f"{NAME}_seed_5_sample_1_model{suffix}").write_bytes(b"")
            assert parser.find_files(tmp_path, NAME).structure.name.endswith(suffix)

    def test_json_confidences_are_preferred_to_npz(self, tmp_path):
        seed_dir = _seed_dir(tmp_path, 5)
        seed_dir.mkdir(parents=True)
        parser = OpenFold3Parser()
        (seed_dir / f"{NAME}_seed_5_sample_1_confidences.npz").write_bytes(b"")
        assert parser.find_files(tmp_path, NAME).arrays.suffix == ".npz"
        (seed_dir / f"{NAME}_seed_5_sample_1_confidences.json").write_bytes(b"")
        assert parser.find_files(tmp_path, NAME).arrays.suffix == ".json"


class TestSeedDirectories:
    """The seed part of a directory name is a value; ``seed_index`` is a position."""

    def _three_seeds(self, tmp_path):
        for seed in (100, 9, 10):  # written out of order on purpose
            (_seed_dir(tmp_path, seed)).mkdir(parents=True)
            (
                _seed_dir(tmp_path, seed)
                / f"{NAME}_seed_{seed}_sample_1_confidences_aggregated.json"
            ).write_text(json.dumps({"avg_plddt": float(seed)}), encoding="utf-8")
        return tmp_path

    def test_positions_follow_the_numeric_order_of_the_seed_values(self, tmp_path):
        root = self._three_seeds(tmp_path)
        parser = OpenFold3Parser()
        values = [parser.load(root, NAME, seed_index=i).avg_plddt for i in (1, 2, 3)]
        assert values == [9.0, 10.0, 100.0]  # string order would give 10, 100, 9

    def test_the_seed_value_of_the_directory_is_recorded(self, tmp_path):
        root = self._three_seeds(tmp_path)
        record = OpenFold3Parser().load(root, NAME, seed_index=2)
        assert record.extras["seed_value"] == "10"
        assert record.seed_index == 2

    def test_a_seed_that_openfold3_generated_is_named_by_its_directory_only(self, tmp_path):
        from binding_metrics.metrics.prediction import compute_prediction_metrics

        # `--num_model_seeds=1` on OpenFold3 0.5.0: the directory is seed_2746317213 and
        # experiment_config.json holds `seeds: [42]` and `num_seeds: null`, not the seed it made
        _write(tmp_path)  # seed_9
        generated = 2746317213
        (_seed_dir(tmp_path, 9)).rename(_seed_dir(tmp_path, generated))
        for path in _seed_dir(tmp_path, generated).glob(f"{NAME}_seed_9_*"):
            path.rename(path.with_name(path.name.replace("_seed_9_", f"_seed_{generated}_")))
        config = {
            "experiment_settings": {"seeds": [42], "num_seeds": None},
            "inference_ckpt_path": "/w/of3-ob-2025-06-30-174k.pt",
            "inference_ckpt_name": "openbind-2025-06-30-174k",
        }
        (tmp_path / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")

        record = OpenFold3Parser().load(tmp_path, NAME)
        assert record.extras["seed_value"] == str(generated)  # not "42"
        assert record.avg_plddt == 82.0  # the files of that directory were read
        assert record.extras["inference_ckpt_name"] == "openbind-2025-06-30-174k"
        summary = compute_prediction_metrics(tmp_path, "of3", NAME)
        assert summary["seed_value"] == generated and summary["seed"] == 1
        refs = OpenFold3Parser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1)]

    def test_the_seed_value_does_not_come_from_the_config_when_it_lists_other_seeds(self, tmp_path):
        # two directories of a run with seeds 7 and 11: each sample reports its own directory
        for position, seed in ((1, 7), (2, 11)):
            _write(tmp_path, seed_index=position)  # written as seed_9 and seed_10
            (_seed_dir(tmp_path, 8 + position)).rename(_seed_dir(tmp_path, seed))
            for path in _seed_dir(tmp_path, seed).glob(f"{NAME}_seed_{8 + position}_*"):
                path.rename(
                    path.with_name(path.name.replace(f"_seed_{8 + position}_", f"_seed_{seed}_"))
                )
        config = {"experiment_settings": {"seeds": [42], "num_seeds": None}}
        (tmp_path / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
        values = [
            OpenFold3Parser().load(tmp_path, NAME, seed_index=i).extras["seed_value"]
            for i in (1, 2)
        ]
        assert values == ["7", "11"]

    def test_a_directory_that_is_not_numeric_sorts_after_the_numeric_ones(self, tmp_path):
        root = self._three_seeds(tmp_path)
        odd = _seed_dir(root, "final")
        odd.mkdir()
        (odd / f"{NAME}_seed_final_sample_1_confidences_aggregated.json").write_text(
            json.dumps({"avg_plddt": 1.0}), encoding="utf-8"
        )
        parser = OpenFold3Parser()
        assert parser.load(root, NAME, seed_index=3).avg_plddt == 100.0
        assert parser.load(root, NAME, seed_index=4).avg_plddt == 1.0

    def test_a_file_called_seed_something_is_not_a_seed_directory(self, tmp_path):
        root = self._three_seeds(tmp_path)
        (root / NAME / "seed_0.log").write_text("x", encoding="utf-8")
        assert OpenFold3Parser().load(root, NAME, seed_index=1).avg_plddt == 9.0

    def test_a_position_beyond_the_last_directory_is_taken_as_the_seed_value(self, tmp_path):
        _write(tmp_path)  # seed_9
        (_seed_dir(tmp_path, 2)).mkdir()
        parser = OpenFold3Parser()
        # two directories, seeds 2 and 9; position 3 does not exist, so "seed_3" is looked up
        assert not parser.find_files(tmp_path, NAME, seed_index=3).has_output()
        assert parser.find_files(tmp_path, NAME, seed_index=2).structure is not None  # seed 9

    def test_list_samples_follows_the_same_order(self, tmp_path):
        _write(tmp_path, seed_index=2)  # seed_10
        _write(tmp_path, seed_index=1)  # seed_9
        _write(tmp_path, seed_index=1, sample=2)
        refs = OpenFold3Parser().list_samples(tmp_path, NAME)
        assert [(r.seed_index, r.sample) for r in refs] == [(1, 1), (1, 2), (2, 1)]
        assert refs[0].ranking_score == pytest.approx(0.82)


class TestParse:
    def test_the_aggregated_keys_go_to_the_record_fields(self, tmp_path):
        record = OpenFold3Parser().load(_write(tmp_path), NAME)
        assert record.model == "of3" and record.name == NAME
        assert (record.avg_plddt, record.gpde) == (82.0, 1.23)
        assert (record.ptm, record.iptm, record.disorder, record.has_clash) == (
            0.88,
            0.76,
            0.12,
            0.0,
        )
        assert record.ranking_score == 0.82
        assert record.ranking_score_name == "sample_ranking_score"
        assert record.timing == {"runtime_s": 12.5}
        assert record.extras["seed_value"] == "9"

    def test_the_record_keeps_the_files_it_was_parsed_from(self, tmp_path):
        parser = OpenFold3Parser()
        root = _write(tmp_path)
        assert parser.parse(parser.find_files(root, NAME), name=NAME).files.arrays.suffix == ".json"

    def test_chain_pair_keys_are_strings_as_written(self, tmp_path):
        record = OpenFold3Parser().load(_write(tmp_path), NAME)
        assert record.chain_ptm == {"A": 0.88, "B": 0.80}
        assert set(record.chain_pair_iptm) == {"(A, B)", "(B, A)"}
        assert all(isinstance(key, str) for key in record.chain_pair_iptm)
        assert record.extras["bespoke_iptm"] == {"(A, B)": 0.74}

    def test_the_full_confidence_arrays(self, tmp_path):
        truth = synth.synthetic_complex()
        record = OpenFold3Parser().load(_write(tmp_path), NAME)
        np.testing.assert_allclose(record.plddt_per_atom, truth.plddt_per_atom)
        np.testing.assert_allclose(record.pae, truth.pae)
        np.testing.assert_allclose(record.pde, truth.pde)
        assert record.tokens is None  # OpenFold3 writes no token layout
        record.validate(check_structure=True)

    def test_the_gzipped_structure_is_read_through_the_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(synth_of3, "STRUCTURE_SUFFIX", ".cif.gz")
        record = OpenFold3Parser().load(_write(tmp_path), NAME, chain_map={"A": "R"})
        assert record.structure_path.name.endswith("_model.cif.gz")
        assert set(record.atoms().chain_id) == {"R", "B"}
        record.validate(check_structure=True)

    def test_avg_plddt_falls_back_to_the_per_atom_mean_and_gpde_does_not(self, tmp_path):
        root = _write(tmp_path)
        (_seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences_aggregated.json").unlink()
        record = OpenFold3Parser().load(root, NAME)
        assert record.avg_plddt == pytest.approx(82.0)
        assert np.isnan(record.gpde) and np.isnan(record.ptm)
        assert record.reasons == ["aggregated confidences file not found"]

    def test_a_missing_full_confidence_file_names_the_option_that_controls_it(self, tmp_path):
        root = _write(tmp_path)
        (_seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences.json").unlink()
        record = OpenFold3Parser().load(root, NAME)
        assert record.plddt_per_atom is None and record.pae is None and record.pde is None
        assert record.avg_plddt == 82.0
        assert record.reasons == [
            "per-atom confidences file not found; OpenFold3 writes it only when "
            "write_full_confidence_scores is true"
        ]
        assert "pae_enabled" not in record.reasons[0]

    def test_no_confidence_files_at_all(self, tmp_path):
        record = OpenFold3Parser().load(tmp_path, NAME, seed_index=2, sample=3)
        assert record.reasons == [
            f"no confidence files found for query '{NAME}' (seed index 2, sample 3) in {tmp_path}"
        ]
        assert "seed_value" not in record.extras

    def test_only_a_structure_counts_as_no_confidence_files(self, tmp_path):
        root = _write(tmp_path)
        for suffix in ("_confidences.json", "_confidences_aggregated.json"):
            (_seed_dir(root, 9) / f"{NAME}_seed_9_sample_1{suffix}").unlink()
        record = OpenFold3Parser().load(root, NAME)
        assert record.structure_path is not None
        assert record.reasons[0].startswith("no confidence files found")

    def test_a_corrupt_json_raises(self, tmp_path):
        root = _write(tmp_path)
        (_seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences.json").write_text(
            "{", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            OpenFold3Parser().load(root, NAME)

    def test_parsing_does_not_open_the_structure_file(self, tmp_path):
        root = _write(tmp_path)
        (_seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_model.cif").write_text(
            "# stub CIF\n", encoding="utf-8"
        )
        assert OpenFold3Parser().load(root, NAME).avg_plddt == 82.0


def _summary_text(total, failed):
    """``summary.txt`` as openfold3 v0.5.0 writes it (``core/runners/writer.py``)."""
    lines = [
        "=" * 50,
        f"Total Queries Processed: {total}",
        f"  - Successful Queries:  {total - len(failed)}",
        f"  - Failed Queries:      {len(failed)}",
    ]
    if failed:
        lines.append(f"\nFailed Queries: {', '.join(failed)}")
    return "\n".join(lines) + "\n"


def _error_log(query_ids, kind, message):
    """One entry of ``logs/predict_err_rank0.log`` (``projects/of3_all_atom/runner.py``)."""
    return "\n".join(
        [
            "=" * 50,
            f"Query ID(s): {', '.join(query_ids)}",
            f"Error Type: {kind}",
            f"Error Message: {message}",
            "-" * 50,
            "Traceback:Traceback (most recent call last):",
            "=" * 50,
        ]
    )


class TestRunProvenance:
    """The checkpoint and the user-default runner YAML are kept in ``record.extras`` (issue 85)."""

    @pytest.fixture(autouse=True)
    def _no_user_default_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "empty_cache"))

    def _config(self, root, **overrides):
        config = {
            "experiment_settings": {"mode": "predict"},
            "inference_ckpt_path": "/w/of3-ob-2025-06-30-174k.pt",
            "inference_ckpt_name": "openbind-2025-06-30-174k",
            "user_default_runner_yaml_path": None,
        }
        config.update(overrides)
        (root / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_the_checkpoint_is_read_from_the_experiment_config(self, tmp_path):
        root = _write(tmp_path)
        self._config(root)
        extras = OpenFold3Parser().load(root, NAME).extras
        assert extras["inference_ckpt_path"] == "/w/of3-ob-2025-06-30-174k.pt"
        assert extras["inference_ckpt_name"] == "openbind-2025-06-30-174k"
        assert extras["user_default_runner_yaml"] is None

    def test_the_runner_yaml_the_run_merged_is_read_from_the_config(self, tmp_path):
        root = _write(tmp_path)
        self._config(root, user_default_runner_yaml_path="/home/u/.openfold3/runner.yml")
        extras = OpenFold3Parser().load(root, NAME).extras
        assert extras["user_default_runner_yaml"] == "/home/u/.openfold3/runner.yml"

    def test_a_null_checkpoint_name_is_kept_as_none(self, tmp_path):
        root = _write(tmp_path)
        self._config(root, inference_ckpt_name=None)
        assert OpenFold3Parser().load(root, NAME).extras["inference_ckpt_name"] is None

    def test_without_the_config_nothing_is_known_about_the_checkpoint(self, tmp_path):
        extras = OpenFold3Parser().load(_write(tmp_path), NAME).extras
        assert "inference_ckpt_path" not in extras and "inference_ckpt_name" not in extras
        assert "user_default_runner_yaml" not in extras

    def test_without_the_config_the_user_default_runner_yaml_of_this_machine_is_probed(
        self, tmp_path, monkeypatch
    ):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "runner.yml").write_text("experiment_settings: {}\n", encoding="utf-8")
        monkeypatch.setenv("OPENFOLD_CACHE", str(cache))
        extras = OpenFold3Parser().load(_write(tmp_path / "run"), NAME).extras
        assert extras["user_default_runner_yaml"] == str(cache / "runner.yml")

    @pytest.mark.parametrize("content", ["{", "[1, 2]", ""])
    def test_an_unreadable_config_is_ignored(self, tmp_path, content):
        root = _write(tmp_path)
        (root / "experiment_config.json").write_text(content, encoding="utf-8")
        record = OpenFold3Parser().load(root, NAME)
        assert record.reasons == [] and "inference_ckpt_name" not in record.extras

    def test_a_directory_with_no_output_has_no_checkpoint_information(self, tmp_path):
        record = OpenFold3Parser().load(tmp_path, "q")
        assert record.reasons and "inference_ckpt_path" not in record.extras


class TestTemplateExtras:
    """What became of the templates is kept in ``record.extras["templates"]`` (OpenFold3 0.5.0)."""

    def _query_set(self, root, entries_by_chain):
        chains = [
            {"chain_ids": [chain_id], "template_entry_chain_ids": entries}
            for chain_id, entries in entries_by_chain.items()
        ]
        body = {"queries": {NAME: {"query_name": NAME, "chains": chains}}}
        (root / "inference_query_set.json").write_text(json.dumps(body), encoding="utf-8")

    def test_the_query_set_of_the_run_says_which_chains_kept_a_template(self, tmp_path):
        root = _write(tmp_path)
        self._query_set(root, {"A": ["receptor_A"], "B": []})
        templates = OpenFold3Parser().load(root, NAME).extras["templates"]
        assert templates["A"]["used"] is True and templates["A"]["entry_ids"] == ["receptor_A"]
        assert templates["B"]["used"] is False and templates["B"]["cause"] == "not_recorded"

    def test_the_accounting_file_of_the_toolkit_gives_the_cause(self, tmp_path):
        from binding_metrics.metrics._openfold_templates import write_template_accounting

        root = _write(tmp_path)
        self._query_set(root, {"A": [], "B": []})
        record = {"requested": True, "source": "alignment", "used": False}
        record["cause"] = "replaced_by_msa_server"
        write_template_accounting(root, {NAME: {"A": record, "B": dict(record)}})
        templates = OpenFold3Parser().load(root, NAME).extras["templates"]
        assert templates["A"]["cause"] == "replaced_by_msa_server"
        assert templates["B"]["requested"] is True

    def test_an_output_without_either_file_has_no_templates_key(self, tmp_path):
        assert "templates" not in OpenFold3Parser().load(_write(tmp_path), NAME).extras

    def test_an_unreadable_query_set_is_ignored(self, tmp_path):
        root = _write(tmp_path)
        (root / "inference_query_set.json").write_text("{", encoding="utf-8")
        record = OpenFold3Parser().load(root, NAME)
        assert "templates" not in record.extras and record.reasons == []

    def test_a_query_set_of_another_query_is_not_taken_for_this_one(self, tmp_path):
        root = _write(tmp_path)
        body = {
            "queries": {
                "other": {"chains": [{"chain_ids": ["A"], "template_entry_chain_ids": ["x"]}]}
            }
        }
        (root / "inference_query_set.json").write_text(json.dumps(body), encoding="utf-8")
        assert "templates" not in OpenFold3Parser().load(root, NAME).extras


class TestFailedQueries:
    """OpenFold3 exits with status 0 when a query fails; the run's files say why (issue 84)."""

    def _failed_run(self, tmp_path, failed=("gone",), message="CUDA out of memory"):
        (tmp_path / "summary.txt").write_text(_summary_text(2, list(failed)), encoding="utf-8")
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "predict_err_rank0.log").write_text(
            _error_log(list(failed), "OutOfMemoryError", message), encoding="utf-8"
        )
        return tmp_path

    def test_a_failed_query_adds_its_reason_after_the_missing_files(self, tmp_path):
        root = self._failed_run(tmp_path)
        record = OpenFold3Parser().load(root, "gone")
        assert len(record.reasons) == 2
        assert record.reasons[0].startswith("no confidence files found for query 'gone'")
        assert record.reasons[1].startswith(
            "OpenFold3 failed on this query: OutOfMemoryError: CUDA out of memory"
        )
        # the log is named relative to the run: the directory of a stored run is renamed
        assert "(logs/predict_err_rank0.log in the output of the run)" in record.reasons[1]
        assert str(tmp_path) not in record.reasons[1]

    def test_only_the_query_that_failed_gets_the_reason(self, tmp_path):
        root = self._failed_run(tmp_path)
        other = OpenFold3Parser().load(root, "fine")
        assert len(other.reasons) == 1 and "failed on this query" not in other.reasons[0]

    def test_a_summary_without_the_log_still_names_the_summary(self, tmp_path):
        (tmp_path / "summary.txt").write_text(_summary_text(1, ["gone"]), encoding="utf-8")
        record = OpenFold3Parser().load(tmp_path, "gone")
        assert "reported this query as failed" in record.reasons[1]
        assert "summary.txt" in record.reasons[1]

    def test_a_query_with_confidence_files_is_not_reported_as_failed(self, tmp_path):
        root = self._failed_run(tmp_path, failed=(NAME,))
        _write(root)
        record = OpenFold3Parser().load(root, NAME)
        assert record.reasons == []

    def test_a_run_without_a_summary_adds_nothing(self, tmp_path):
        assert len(OpenFold3Parser().load(tmp_path, "q").reasons) == 1

    def test_the_parser_still_imports_no_biotite_when_it_explains_a_failure(self, tmp_path):
        import subprocess
        import sys

        self._failed_run(tmp_path)
        code = (
            "import sys; sys.modules['biotite'] = None\n"
            "from binding_metrics.predictors.registry import get_parser\n"
            f"r = get_parser('of3').load({str(tmp_path)!r}, 'gone')\n"
            "print(len(r.reasons))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8"
        )
        assert done.returncode == 0 and done.stdout.strip() == "2", done.stderr


class TestNpzConfidences:
    """``.npz`` files are read without pickle and only for the arrays OpenFold3 writes."""

    def _npz_run(self, tmp_path, monkeypatch, **extra_arrays):
        monkeypatch.setattr(synth_of3, "CONFIDENCE_FORMAT", "npz")
        root = _write(tmp_path)
        path = _seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences.npz"
        truth = synth.synthetic_complex()
        with open(path, "wb") as fh:
            np.savez(
                fh,
                plddt=truth.plddt_per_atom.astype(np.float16),
                pde=truth.pde.astype(np.float16),
                pae=truth.pae.astype(np.float16),
                **extra_arrays,
            )
        return root

    def test_float16_arrays_are_read_as_floats(self, tmp_path, monkeypatch):
        record = OpenFold3Parser().load(self._npz_run(tmp_path, monkeypatch), NAME)
        truth = synth.synthetic_complex()
        assert record.plddt_per_atom.dtype == np.float64
        np.testing.assert_allclose(record.pae, truth.pae)
        record.validate(check_structure=True)

    def test_the_stray_allow_pickle_array_that_numpy_1_26_adds_is_ignored(
        self, tmp_path, monkeypatch
    ):
        root = self._npz_run(tmp_path, monkeypatch, allow_pickle=np.array(False))
        record = OpenFold3Parser().load(root, NAME)
        assert record.pae is not None

    def test_an_object_array_that_is_not_read_is_never_unpickled(self, tmp_path, monkeypatch):
        marker = tmp_path / "executed"

        class Payload:
            def __reduce__(self):
                return (marker.write_text, ("x",))

        evil = np.empty(1, dtype=object)
        evil[0] = Payload()
        root = self._npz_run(tmp_path, monkeypatch, evil=evil)
        OpenFold3Parser().load(root, NAME)
        assert not marker.exists()

    def test_an_object_array_under_a_key_that_is_read_raises_a_clear_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(synth_of3, "CONFIDENCE_FORMAT", "npz")
        root = _write(tmp_path)
        path = _seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences.npz"
        bad = np.empty(1, dtype=object)
        bad[0] = {"not": "numeric"}
        with open(path, "wb") as fh:
            np.savez(fh, plddt=bad)
        with pytest.raises(ValueError, match="cannot be read without pickle.*plain numeric arrays"):
            OpenFold3Parser().load(root, NAME)

    def test_a_corrupt_npz_raises(self, tmp_path, monkeypatch):
        root = self._npz_run(tmp_path, monkeypatch)
        path = _seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences.npz"
        path.write_bytes(b"not an archive")
        with pytest.raises((ValueError, OSError, EOFError)):
            OpenFold3Parser().load(root, NAME)

    def test_the_helper_functions_read_the_same_values_as_the_adapter(self, tmp_path, monkeypatch):
        root = self._npz_run(tmp_path, monkeypatch)
        parsed = of3.parse_full_confidences(
            _seed_dir(root, 9) / f"{NAME}_seed_9_sample_1_confidences.npz"
        )
        assert set(parsed) == {"plddt_per_atom", "pde", "pae", "gpde"}
        assert np.isnan(parsed["gpde"])


class TestContractVariants:
    """The layouts OpenFold3 0.5.0 can write, through the same contract checks."""

    @pytest.mark.parametrize("suffix", [".cif.gz", ".pdb"])
    @pytest.mark.parametrize("fmt", ["json", "npz"])
    def test_structure_and_confidence_formats(self, tmp_path, monkeypatch, suffix, fmt):
        monkeypatch.setattr(synth_of3, "STRUCTURE_SUFFIX", suffix)
        monkeypatch.setattr(synth_of3, "CONFIDENCE_FORMAT", fmt)
        for check in (
            contract.check_load_valid_record,
            contract.check_sample_and_seed_selection,
            contract.check_missing_files,
            contract.check_corrupt_files,
            contract.check_chain_map,
        ):
            workdir = tmp_path / check.__name__
            workdir.mkdir()
            check("of3", workdir)
