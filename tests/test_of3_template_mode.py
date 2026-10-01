"""``template_mode``: the template as an alignment (the default) or as a structure (#68).

OpenFold3's CIF Direct Template Mode (``template_cif_paths`` and ``template_cif_chain_ids`` of a
chain, protein chains only) gives the template CIF itself; the ColabFold MSA server overwrites an
alignment path and leaves CIF paths alone (``colabfold_msa_server.py``). Nothing here runs
OpenFold3: the query builders, the request key and the command line are the real code, the model is
a stub. What OpenFold3 does with such a query is in the validation runs of the documentation.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

from binding_metrics.cli import (
    add_openfold_templates_arg,
    batch,
    check_openfold_templates,
    openfold_template_kwargs,
    run,
)
from binding_metrics.cli.prediction import check_templates_option, make_request, make_runner
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import _openfold_cli, _openfold_run, openfold
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from tests.test_feat_c_support import StubOpenFold

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

DATA = Path(__file__).parent.parent / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"


@pytest.fixture(autouse=True)
def _openfold3_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "openfold_cache"))
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )
    monkeypatch.setattr(_openfold_run, "_VERSION_BY_PYTHON", {})


def _chains(query_json: Path, name: str = "q") -> dict:
    query = json.loads(Path(query_json).read_text(encoding="utf-8"))
    return {c["chain_ids"][0]: c for c in query["queries"][name]["chains"]}


def _refold(out, **kwargs):
    return openfold.prepare_refolding_query(P53, "A", "B", "q", out, **kwargs)


def _score(out, **kwargs):
    return openfold.prepare_scoring_query(P53, "A", "B", "q", out, **kwargs)


class TestStructureQueries:
    def test_refold_gives_the_receptor_its_cif_and_writes_no_alignment(self, tmp_path):
        chains = _chains(_refold(tmp_path, template_mode="structure"))
        receptor, binder = chains["A"], chains["B"]
        assert receptor["template_cif_paths"] == [str(tmp_path / "templates" / "receptor.cif")]
        assert receptor["template_cif_chain_ids"] == ["A"]
        assert "template_alignment_file_path" not in receptor
        assert not any(key.startswith("template") for key in binder)
        assert list(tmp_path.glob("*.a3m")) == []
        assert (tmp_path / "templates" / "receptor.cif").is_file()

    def test_score_gives_each_chain_its_cif(self, tmp_path):
        chains = _chains(_score(tmp_path, template_mode="structure"))
        assert chains["A"]["template_cif_paths"] == [str(tmp_path / "templates" / "receptor.cif")]
        assert chains["B"]["template_cif_paths"] == [str(tmp_path / "templates" / "binder.cif")]
        assert chains["A"]["template_cif_chain_ids"] == ["A"]
        assert chains["B"]["template_cif_chain_ids"] == ["B"]
        assert all("template_alignment_file_path" not in c for c in chains.values())
        assert list(tmp_path.glob("*.a3m")) == []

    def test_the_default_is_the_alignment_of_before(self, tmp_path):
        explicit = _chains(_score(tmp_path / "e", template_mode="alignment"))
        default = _chains(_score(tmp_path / "d"))
        assert [
            c.get("template_alignment_file_path", "").rsplit("/", 1)[-1] for c in default.values()
        ] == [
            "q_receptor.a3m",
            "q_binder.a3m",
        ]
        assert all("template_cif_paths" not in c for c in default.values())
        assert {k: sorted(v) for k, v in explicit.items()} == {
            k: sorted(v) for k, v in default.items()
        }

    def test_the_cif_in_the_query_is_the_repaired_one(self, tmp_path):
        import gemmi

        chains = _chains(_score(tmp_path, template_mode="structure"))
        block = gemmi.cif.read(chains["B"]["template_cif_paths"][0]).sole_block()
        assert (
            block.find_value("_entity_poly.pdbx_seq_one_letter_code_can").strip("'\"")
            == (chains["B"]["sequence"])
        )
        assert set(block.find_values("_atom_site.label_entity_id")) == {"1"}

    def test_a_mode_that_does_not_exist_is_refused_before_anything_is_written(self, tmp_path):
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="template_mode must be one of"):
            _refold(out, template_mode="cif")
        assert not out.exists()

    def test_the_binder_of_a_modified_residue_keeps_its_residues(self, tmp_path):
        query = openfold.prepare_scoring_query(
            CYCLOSPORIN, "A", "C", "q", tmp_path, template_mode="structure", binder_cyclic=False
        )
        chain = _chains(query)["C"]
        assert chain["non_canonical_residues"]["1"] == "DAL"
        assert chain["template_cif_chain_ids"] == ["C"]

    def test_an_underscore_in_a_template_chain_id_is_still_refused(self, tmp_path):
        """OpenFold3 reads the template as <entry>_<chain> and splits on the underscore."""
        with pytest.raises(ValueError, match="underscore"):
            _openfold_run._template_fields(
                "structure",
                sequence="AG",
                chain_id="A_1",
                entry_id="receptor",
                cif_path=tmp_path / "receptor.cif",
                a3m_path=tmp_path / "x.a3m",
            )

    def test_a_chain_cannot_have_both_kinds_of_template(self):
        with pytest.raises(ValueError, match="not both"):
            _openfold_run._query_chain(
                "A",
                "AAA",
                {},
                template_alignment_file_path="x.a3m",
                template_cif_paths=["x.cif"],
            )

    @pytest.mark.parametrize(
        "function",
        [openfold.prepare_batched_scoring_queries, openfold.prepare_batched_refolding_queries],
    )
    def test_the_batched_queries_take_the_mode_too(self, tmp_path, function):
        samples = [openfold._BatchSample("s1", P53, "A", "B")]
        path = function(samples, tmp_path, template_mode="structure")
        chains = {
            c["chain_ids"][0]: c
            for c in json.loads(path.read_text(encoding="utf-8"))["queries"]["s1"]["chains"]
        }
        assert chains["A"]["template_cif_paths"] == [str(tmp_path / "templates" / "s1rec.cif")]
        assert ("template_cif_paths" in chains["B"]) is (
            function.__name__.endswith("scoring_queries")
        )
        assert list(tmp_path.glob("*.a3m")) == []


class TestWrappers:
    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_the_run_functions_pass_the_mode_to_the_query_builder_only_when_it_is_set(
        self, tmp_path, monkeypatch, runner
    ):
        builder = {
            "run_openfold_scoring": "prepare_scoring_query",
            "run_openfold_refolding": "prepare_refolding_query",
        }[runner]
        seen = {}

        def fake_prepare(**kwargs):
            seen.clear()
            seen.update(kwargs)
            return tmp_path / "q.json"

        monkeypatch.setattr(openfold, builder, fake_prepare)
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: tmp_path)
        getattr(openfold, runner)(P53, "A", "B", "q", tmp_path)
        assert "template_mode" not in seen  # the default is left out
        getattr(openfold, runner)(P53, "A", "B", "q", tmp_path, template_mode="structure")
        assert seen["template_mode"] == "structure"

    def test_the_batched_run_passes_it_on(self, tmp_path, monkeypatch):
        seen = {}

        def fake_prepare(samples, output_dir, **kwargs):
            seen.update(kwargs)
            return Path(output_dir) / "batch_query.json"

        monkeypatch.setattr(openfold, "prepare_batched_scoring_queries", fake_prepare)
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: tmp_path)
        samples = [openfold._BatchSample("s", P53, "A", "B")]
        openfold.run_openfold_batched(samples, tmp_path, template_mode="structure")
        assert seen["template_mode"] == "structure"

    def test_a_mode_that_does_not_exist_stops_before_the_process(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            openfold, "run_openfold", lambda **kw: pytest.fail("OpenFold3 must not start")
        )
        with pytest.raises(ValueError, match="template_mode must be one of"):
            openfold.run_openfold_scoring(P53, "A", "B", "q", tmp_path, template_mode="x")

    def test_the_names_are_importable_from_the_openfold_module(self):
        assert openfold.TEMPLATE_MODES == ("alignment", "structure")
        assert openfold._check_template_mode is _openfold_run._check_template_mode


class TestRequestKey:
    @staticmethod
    def _request(**kwargs):
        return OpenFold3Runner().make_request(
            P53, name="q", binder_chain="B", receptor_chain="A", **kwargs
        )

    def test_the_default_request_says_alignment(self):
        assert self._request().options["template_mode"] == "alignment"

    def test_each_mode_is_another_key(self):
        assert self._request().key() == self._request(template_mode="alignment").key()
        assert self._request().key() != self._request(template_mode="structure").key()

    def test_the_query_arguments_leave_the_default_out(self):
        arguments = OpenFold3Runner._query_arguments
        assert "template_mode" not in arguments(self._request())
        assert arguments(self._request(template_mode="structure"))["template_mode"] == "structure"

    def test_a_query_file_names_its_own_templates(self, tmp_path):
        query = tmp_path / "query.json"
        query.write_text("{}", encoding="utf-8")
        runner = OpenFold3Runner()
        assert runner.make_request(query, name="q", mode="predict").options["template_mode"] is None
        with pytest.raises(ValueError, match="names its own templates"):
            runner.make_request(query, name="q", mode="predict", template_mode="structure")

    def test_a_mode_that_does_not_exist_is_refused(self):
        with pytest.raises(ValueError, match="template_mode must be one of"):
            self._request(template_mode="cif")

    def test_the_runner_prepares_and_runs_with_the_mode(self, tmp_path, monkeypatch):
        seen = {}

        def fake_prepare(**kwargs):
            seen["prepare"] = kwargs
            return Path(kwargs["output_dir"]) / "q_query.json"

        def fake_run(**kwargs):
            seen["run"] = kwargs
            out = Path(kwargs["output_dir"]) / "predictions"
            seed_dir = out / "q" / "seed_42"
            seed_dir.mkdir(parents=True)
            (seed_dir / "q_seed_42_sample_1_confidences_aggregated.json").write_text(
                "{}", encoding="utf-8"
            )
            return out

        monkeypatch.setattr(openfold, "prepare_scoring_query", fake_prepare)
        monkeypatch.setattr(openfold, "run_openfold_scoring", fake_run)
        runner = OpenFold3Runner()
        request = self._request(template_mode="structure")
        (tmp_path / "w").mkdir()
        runner.prepare(request, tmp_path / "w")
        runner.run(request, tmp_path / "w")
        assert seen["prepare"]["template_mode"] == "structure"
        assert seen["run"]["template_mode"] == "structure"


class TestOptionsOfTheCommands:
    def test_the_choices_and_the_helper(self):
        assert check_openfold_templates("structure") == "structure"
        with pytest.raises(ValueError, match="openfold_templates must be one of"):
            check_openfold_templates("cif")
        assert openfold_template_kwargs("alignment") == {}
        assert openfold_template_kwargs("structure") == {"template_mode": "structure"}

    def test_the_help_says_what_the_option_does_and_that_it_is_openfold3s(self):
        parser = argparse.ArgumentParser()
        add_openfold_templates_arg(parser)
        text = " ".join(parser.format_help().split())
        assert "--openfold-templates {alignment,structure}" in text
        for stated in (
            "CIF Direct Template Mode",
            "overwrites it",
            "does not overwrite",
            "OpenFold3 only",
            "--openfold-no-msa-server",
            "'templates'",
        ):
            assert stated in text

    @pytest.mark.parametrize("module", ["run", "batch"])
    @pytest.mark.parametrize(
        "extra, expected",
        [([], "alignment"), (["--openfold-templates", "structure"], "structure")],
    )
    def test_the_choice_reaches_the_api(self, monkeypatch, tmp_path, module, extra, expected):
        captured = {}
        target = {"run": "run_pipeline", "batch": "run_batch"}[module]
        monkeypatch.setattr(
            {"run": run, "batch": batch}[module],
            target,
            lambda *a, **kwargs: captured.update(kwargs) or ({} if module == "run" else []),
        )
        input_args = ["-i", str(P53), "-o", str(tmp_path / "o")]
        if module == "batch":
            input_args = ["-i", str(P53.parent), "--output-csv", str(tmp_path / "m.csv")]
        monkeypatch.setattr(sys, "argv", ["prog", *input_args, *extra])
        try:
            {"run": run, "batch": batch}[module].main()
        except SystemExit:
            pass
        assert captured["openfold_templates"] == expected

    @pytest.mark.parametrize("command", ["prepare-query", "prepare-scoring-query"])
    def test_binding_metrics_openfold_prepares_a_structure_query(
        self, tmp_path, monkeypatch, command
    ):
        argv = [
            "prog", command, "--complex", str(P53), "--receptor-chain", "A",
            "--binder-chain", "B", "--query-name", "q", "--output-dir", str(tmp_path / "out"),
            "--openfold-templates", "structure",
        ]  # fmt: skip
        monkeypatch.setattr("sys.argv", argv)
        openfold.main()
        chains = _chains(tmp_path / "out" / "q_query.json")
        assert "template_cif_paths" in chains["A"]

    @pytest.mark.parametrize(
        "command, target",
        [("score", "run_openfold_scoring"), ("refold", "run_openfold_refolding")],
    )
    @pytest.mark.parametrize(
        "extra, expected", [([], None), (["--openfold-templates", "structure"], "structure")]
    )
    def test_the_run_commands_pass_the_mode(
        self, tmp_path, monkeypatch, command, target, extra, expected
    ):
        seen = {}
        monkeypatch.setattr(openfold, target, lambda **kw: seen.update(kw) or tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(_openfold_cli, "_print_metrics", lambda *a, **kw: None)
        argv = [
            "prog", command, "--complex", str(P53), "--receptor-chain", "A",
            "--binder-chain", "B", "--query-name", "q", "--output-dir", str(tmp_path), *extra,
        ]  # fmt: skip
        monkeypatch.setattr("sys.argv", argv)
        openfold.main()
        assert seen.get("template_mode") == expected  # left out for the default

    def test_a_bad_choice_is_a_usage_error(self, tmp_path, monkeypatch, capsys):
        argv = [
            "prog", "refold", "--complex", str(P53), "--receptor-chain", "A",
            "--binder-chain", "B", "--query-name", "q", "--output-dir", str(tmp_path),
            "--openfold-templates", "maybe",
        ]  # fmt: skip
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as info:
            openfold.main()
        assert info.value.code == 2 and "invalid choice" in capsys.readouterr().err


class TestPipeline:
    @staticmethod
    def _pipeline(tmp_path, **kwargs):
        return run_pipeline(
            P53,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain="B",
            receptor_chain="A",
            openfold_conda_env=None,
            **kwargs,
        )

    def test_the_predictor_route_passes_the_mode_to_the_run_function(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        self._pipeline(tmp_path, predictor="of3", openfold_templates="structure")
        assert stub.calls[0]["kwargs"]["template_mode"] == "structure"

    def test_the_default_leaves_the_argument_out(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        self._pipeline(tmp_path, predictor="of3")
        assert "template_mode" not in stub.calls[0]["kwargs"]

    def test_the_mode_is_part_of_the_key_of_the_stored_prediction(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        store = tmp_path / "shared"
        keys = {
            mode: self._pipeline(
                tmp_path / mode, predictor="of3", prediction_cache=store, openfold_templates=mode
            )["prediction"]["cache"]["request_key"]
            for mode in ("alignment", "structure")
        }
        assert len(set(keys.values())) == 2

    def test_the_openfold_step_without_a_predictor_passes_it_too(self, tmp_path, monkeypatch):
        seen = {}

        def record(**kwargs):
            seen.update(kwargs)
            return tmp_path

        monkeypatch.setattr(openfold, "run_openfold_scoring", record)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        self._pipeline(tmp_path, openfold_templates="structure")
        assert seen["template_mode"] == "structure"
        seen.clear()
        self._pipeline(tmp_path / "default")
        assert "template_mode" not in seen

    def test_a_bad_value_is_refused_before_anything_runs(self, tmp_path):
        with pytest.raises(ValueError, match="openfold_templates must be one of"):
            self._pipeline(tmp_path, openfold_templates="cif")
        assert not (tmp_path / "openfold").exists()

    def test_another_model_is_refused_and_the_message_says_it_is_openfold3s(self, tmp_path):
        with pytest.raises(ValueError, match="does not apply to --predictor boltz2") as info:
            self._pipeline(tmp_path, predictor="boltz2", openfold_templates="structure")
        assert "--prediction- spelling" in str(info.value)

    def test_an_adopted_output_ignores_it(self, tmp_path):
        check_templates_option("boltz2", tmp_path, "structure")  # nothing is run
        check_templates_option(None, None, "structure")  # the OpenFold3 step
        check_templates_option("of3", None, "structure")

    def test_make_request_refuses_a_runner_without_the_setting(self):
        runner = make_runner("boltz2")
        with pytest.raises(ValueError, match="--openfold-templates is not available"):
            make_request(
                "boltz2",
                "q",
                P53,
                binder_chain="B",
                receptor_chain="A",
                runner=runner,
                openfold_templates="structure",
            )

    def test_make_request_gives_the_openfold3_runner_the_mode_only_when_set(self):
        runner = OpenFold3Runner()
        plain = make_request("of3", "q", P53, binder_chain="B", receptor_chain="A", runner=runner)
        structure = make_request(
            "of3",
            "q",
            P53,
            binder_chain="B",
            receptor_chain="A",
            runner=runner,
            openfold_templates="structure",
        )
        assert plain.options["template_mode"] == "alignment"
        assert structure.options["template_mode"] == "structure"
        assert plain.key() != structure.key()


class TestBatchedCommandLine:
    def test_the_batched_openfold_call_gets_the_mode(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            openfold, "run_openfold_batched", lambda **kw: seen.update(kw) or tmp_path
        )
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(
            batch, "_detect_sample_chains", lambda *a, **k: [(0, "s", P53, "B", "A")]
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [{"sample_id": "s", "batch_status": "ok"}]
        batch._run_batched_openfold(
            rows=rows,
            sid_to_input={"s": P53},
            output_dir=tmp_path,
            openfold_mode="score",
            openfold_conda_env=None,
            peptide_chain=None,
            receptor_chain=None,
            openfold_templates="structure",
        )
        assert seen["template_mode"] == "structure"

    def test_the_batched_prediction_gives_each_request_the_mode(self, tmp_path, monkeypatch):
        stub = StubOpenFold(monkeypatch)
        monkeypatch.setattr(
            batch, "_detect_sample_chains", lambda *a, **k: [(0, "s", P53, "B", "A")]
        )
        monkeypatch.setattr(batch, "_model_step_allowed", lambda *a, **k: (True, None))
        rows = [{"sample_id": "s", "batch_status": "ok"}]
        batch._run_batched_prediction(
            rows=rows,
            sid_to_input={"s": P53},
            output_dir=tmp_path,
            predictor="of3",
            peptide_chain=None,
            receptor_chain=None,
            openfold_templates="structure",
        )
        assert stub.calls[0]["kwargs"]["template_mode"] == "structure"
