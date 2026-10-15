"""OpenFold3 with a receptor MSA and no MSA server (the OpenFold3 side of the MSA policy).

A receptor alignment goes to the receptor chain of the query as ``colabfold_main.a3m`` (the name
that OpenFold3 0.5.0 parses), the binder keeps the dummy MSA, and a request that forbids the
server can only be built, prepared and run with the server off. The model is replaced by a stub:
the tests look at the files the query builders write, at the arguments the runner passes on and
at the command line it would start. One optional test (``openfold`` marker) runs OpenFold3's own
MSA parser on the staged file.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_run, openfold
from binding_metrics.predictors._msa import assert_no_server_flag
from binding_metrics.predictors.of3_runner import OpenFold3Runner
from binding_metrics.predictors.store import PredictionRequest

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
RECEPTOR = "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
BINDER = "ETFSDLWKLLPEN"
CHAINS = {"binder_chain": "B", "receptor_chain": "A"}


@pytest.fixture(autouse=True)
def _openfold3_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "openfold_cache"))
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )
    monkeypatch.setattr(_openfold_run, "_VERSION_BY_PYTHON", {})
    monkeypatch.setattr(OpenFold3Runner, "version", lambda runner: "0.5.0")


def write_a3m(path: Path, query: str = RECEPTOR, hits: int = 2) -> Path:
    """An alignment: the query, then hits with gaps and lower-case insertions."""
    rows = [(">101", query)]
    for number in range(hits):
        hit = query[:10] + "ac" + query[10:30] + "-" * (number + 3) + query[33 + number :]
        rows.append((f">hit_{number}", hit))
    path.write_text("".join(f"{header}\n{row}\n" for header, row in rows), encoding="utf-8")
    return path


@pytest.fixture
def alignment(tmp_path):
    return write_a3m(tmp_path / "mdm2.a3m")


def chains_of(query_json) -> dict:
    query = json.loads(Path(query_json).read_text(encoding="utf-8"))
    return {c["chain_ids"][0]: c for c in query["queries"]["q"]["chains"]}


def make_request(**kwargs) -> PredictionRequest:
    kwargs.setdefault("mode", "score")
    return OpenFold3Runner().make_request(P53, name="q", **CHAINS, **kwargs)


# ---------------------------------------------------------------------------- the query builders


class TestTheStagedFile:
    @pytest.mark.parametrize("build", ["prepare_scoring_query", "prepare_refolding_query"])
    def test_the_alignment_becomes_colabfold_main_in_a_folder_of_the_chain(
        self, tmp_path, alignment, build
    ):
        query = getattr(openfold, build)(
            P53, "A", "B", "q", tmp_path / "out", dummy_msa=True, receptor_msa_path=alignment
        )
        (path,) = chains_of(query)["A"]["main_msa_file_paths"]
        staged = Path(path)
        assert staged == tmp_path / "out" / "msas" / "q_A" / "colabfold_main.a3m"
        assert staged.stem == "colabfold_main"  # a key of MSASettings.max_seq_counts
        assert staged.read_bytes() == alignment.read_bytes()
        assert staged != alignment  # a copy: the run does not depend on the original

    @pytest.mark.parametrize("build", ["prepare_scoring_query", "prepare_refolding_query"])
    def test_the_binder_keeps_the_dummy_msa_and_never_gets_the_receptors_file(
        self, tmp_path, alignment, build
    ):
        query = getattr(openfold, build)(
            P53, "A", "B", "q", tmp_path / "out", dummy_msa=True, receptor_msa_path=alignment
        )
        chains = chains_of(query)
        (binder_path,) = chains["B"]["main_msa_file_paths"]
        assert Path(binder_path) == tmp_path / "out" / "msas" / "q_B" / "colabfold_main.a3m"
        assert Path(binder_path).read_text(encoding="utf-8") == f">query_B\n{BINDER}\n"
        assert chains["A"]["main_msa_file_paths"] != chains["B"]["main_msa_file_paths"]

    def test_without_the_dummy_the_binder_has_no_msa_file_at_all(self, tmp_path, alignment):
        query = openfold.prepare_scoring_query(
            P53, "A", "B", "q", tmp_path / "out", receptor_msa_path=alignment
        )
        chains = chains_of(query)
        assert "main_msa_file_paths" in chains["A"]
        assert "main_msa_file_paths" not in chains["B"]
        assert not (tmp_path / "out" / "msas" / "q_B").exists()

    def test_without_an_alignment_the_query_is_what_it_was(self, tmp_path):
        with_none = openfold.prepare_scoring_query(
            P53, "A", "B", "q", tmp_path / "a", dummy_msa=True, receptor_msa_path=None
        )
        default = openfold.prepare_scoring_query(P53, "A", "B", "q", tmp_path / "b", dummy_msa=True)
        for query in (with_none, default):
            (path,) = chains_of(query)["A"]["main_msa_file_paths"]
            assert Path(path).read_text(encoding="utf-8") == f">query_A\n{RECEPTOR}\n"

    def test_the_templates_stay_with_the_receptor_chain(self, tmp_path, alignment):
        query = openfold.prepare_scoring_query(
            P53, "A", "B", "q", tmp_path / "out", dummy_msa=True, receptor_msa_path=alignment
        )
        chains = chains_of(query)
        assert "template_cif_paths" in chains["A"] and "template_cif_paths" in chains["B"]

    @pytest.mark.parametrize(
        "function",
        [openfold.prepare_batched_scoring_queries, openfold.prepare_batched_refolding_queries],
    )
    def test_a_batch_gets_one_copy_per_sample_for_the_receptor_only(
        self, tmp_path, alignment, function
    ):
        samples = [
            openfold._BatchSample("s1", P53, "A", "B"),
            openfold._BatchSample("s2", P53, "A", "B"),
        ]
        path = function(samples, tmp_path / "out", dummy_msa=True, receptor_msa_path=alignment)
        queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
        for sample in ("s1", "s2"):
            by_chain = {c["chain_ids"][0]: c for c in queries[sample]["chains"]}
            staged = Path(by_chain["A"]["main_msa_file_paths"][0])
            assert staged == tmp_path / "out" / "msas" / f"{sample}_A" / "colabfold_main.a3m"
            assert staged.read_bytes() == alignment.read_bytes()
            dummy = Path(by_chain["B"]["main_msa_file_paths"][0])
            assert dummy.read_text(encoding="utf-8") == f">query_B\n{BINDER}\n"


class TestTheFirstSequenceIsTheQuery:
    """OpenFold3 0.5.0 takes the first row as the query row and never compares it."""

    def test_an_alignment_of_another_sequence_is_refused_before_anything_is_written(self, tmp_path):
        other = write_a3m(tmp_path / "other.a3m", query=RECEPTOR[:-1] + "W")
        with pytest.raises(ValueError, match="first differ at position 85"):
            openfold.prepare_scoring_query(
                P53, "A", "B", "q", tmp_path / "out", receptor_msa_path=other
            )
        assert not (tmp_path / "out").exists()

    def test_a_shorter_alignment_is_refused_with_both_lengths(self, tmp_path):
        short = write_a3m(tmp_path / "short.a3m", query=RECEPTOR[:60])
        with pytest.raises(ValueError, match=r"it has 60 residues, the chain has 85"):
            openfold.prepare_refolding_query(
                P53, "A", "B", "q", tmp_path / "out", receptor_msa_path=short
            )

    def test_the_insertions_of_the_query_row_do_not_count(self, tmp_path):
        path = tmp_path / "lower.a3m"
        path.write_text(f">q\n{RECEPTOR[:20]}xyz{RECEPTOR[20:]}\n", encoding="utf-8")
        assert _openfold_run._check_receptor_msa(path, RECEPTOR, "A") == path

    def test_a_file_that_is_not_an_a3m_is_refused(self, tmp_path):
        stockholm = tmp_path / "x.sto"
        stockholm.write_text(f"# STOCKHOLM 1.0\nquery {RECEPTOR}\n//\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no sequence in A3M format"):
            _openfold_run._check_receptor_msa(stockholm, RECEPTOR, "A")

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a file"):
            openfold.prepare_scoring_query(
                P53, "A", "B", "q", tmp_path / "out", receptor_msa_path=tmp_path / "none.a3m"
            )

    def test_a_batch_names_the_sample_whose_receptor_differs(self, tmp_path):
        other = write_a3m(tmp_path / "other.a3m", query="W" * 85)
        samples = [openfold._BatchSample("s1", P53, "A", "B")]
        with pytest.raises(ValueError, match="chain A of sample 's1'"):
            openfold.prepare_batched_scoring_queries(
                samples, tmp_path / "out", receptor_msa_path=other
            )


# ---------------------------------------------------------------------------- the run functions


class TestTheRunFunctions:
    @staticmethod
    def _capture(monkeypatch, tmp_path, name="prepare_scoring_query"):
        seen = {}

        def prepare(*args, **kwargs):
            seen.clear()
            seen.update(kwargs)
            return tmp_path / "q.json"

        monkeypatch.setattr(openfold, name, prepare)
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: tmp_path)
        return seen

    @pytest.mark.parametrize(
        "function, builder",
        [
            ("run_openfold_scoring", "prepare_scoring_query"),
            ("run_openfold_refolding", "prepare_refolding_query"),
        ],
    )
    def test_the_path_reaches_the_query_builder_when_the_server_is_off(
        self, tmp_path, monkeypatch, alignment, function, builder
    ):
        seen = self._capture(monkeypatch, tmp_path, builder)
        getattr(openfold, function)(
            P53, "A", "B", "q", tmp_path, use_msa_server=False, receptor_msa_path=alignment
        )
        assert seen["receptor_msa_path"] == alignment
        assert seen["dummy_msa"] is True  # the binder keeps the dummy

    def test_the_batched_run_passes_it_on_too(self, tmp_path, monkeypatch, alignment):
        seen = {}

        def prepare(samples, output_dir, **kwargs):
            seen.update(kwargs)
            return Path(output_dir) / "batch_query.json"

        monkeypatch.setattr(openfold, "prepare_batched_scoring_queries", prepare)
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: tmp_path)
        samples = [openfold._BatchSample("s", P53, "A", "B")]
        openfold.run_openfold_batched(
            samples, tmp_path, use_msa_server=False, receptor_msa_path=alignment
        )
        assert seen["receptor_msa_path"] == alignment

    def test_without_a_path_the_query_builder_is_called_as_before(self, tmp_path, monkeypatch):
        seen = self._capture(monkeypatch, tmp_path)
        openfold.run_openfold_scoring(P53, "A", "B", "q", tmp_path, use_msa_server=False)
        assert "receptor_msa_path" not in seen

    @pytest.mark.parametrize(
        "function", ["run_openfold_scoring", "run_openfold_refolding", "run_openfold_batched"]
    )
    def test_the_server_would_replace_the_alignment_so_the_combination_is_refused(
        self, tmp_path, monkeypatch, alignment, function
    ):
        started = []
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: started.append(kw))
        arguments = (
            ([openfold._BatchSample("s", P53, "A", "B")], tmp_path / "out")
            if function == "run_openfold_batched"
            else (P53, "A", "B", "q", tmp_path / "out")
        )
        with pytest.raises(ValueError, match="needs use_msa_server=False"):
            getattr(openfold, function)(*arguments, receptor_msa_path=alignment)
        assert not (tmp_path / "out").exists() and started == []


# ---------------------------------------------------------------------------- the request


class TestAnOldRequestKeepsItsKey:
    def test_the_defaults_add_nothing_to_options_or_files(self):
        request = make_request()
        assert not {"msa_policy", "receptor_msa"} & set(request.options)
        assert "receptor_msa" not in request.extra_files

    def test_naming_the_defaults_is_the_same_request(self):
        assert make_request(msa_policy="default", receptor_msa=None).key() == make_request().key()

    @pytest.mark.parametrize("mode", ["score", "refold"])
    def test_the_key_is_the_one_the_request_had_before_the_fields_existed(self, mode):
        run_module = _openfold_run
        before = PredictionRequest(
            "of3",
            "q",
            mode=mode,
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            seeds=(42,),
            num_samples=5,
            model_version="0.5.0",
            options={
                "presets": ["predict", "low_mem"],
                "use_msa_server": True,
                "num_model_seeds": None,
                "on_unmappable_residue": "error",
                "binder_cyclic": "auto",
                "template_mode": "structure",
                "query_builder_version": run_module.QUERY_BUILDER_VERSION,
                "extra_args": [],
                "inference_ckpt_path": None,
                "inference_ckpt_size_bytes": None,
            },
        )
        assert make_request(mode=mode).key() == before.key()

    def test_the_query_builder_version_is_not_raised_by_the_receptor_msa(self):
        # the query of a request without the field is unchanged, so its key must not move
        assert _openfold_run.QUERY_BUILDER_VERSION == 3
        assert make_request().options["query_builder_version"] == 3


class TestTheKeyOfAReceptorMsa:
    def test_the_file_is_an_extra_file_of_the_request_and_nothing_goes_to_the_options(
        self, alignment
    ):
        request = make_request(use_msa_server=False, receptor_msa=alignment)
        assert request.extra_files["receptor_msa"] == alignment
        assert "receptor_msa" not in request.options and "msa_policy" not in request.options
        assert request.options["use_msa_server"] is False

    def test_the_same_content_is_the_same_key(self, tmp_path, alignment):
        copy = tmp_path / "elsewhere" / "other_name.a3m"
        copy.parent.mkdir()
        shutil.copyfile(alignment, copy)
        first = make_request(use_msa_server=False, receptor_msa=alignment)
        assert make_request(use_msa_server=False, receptor_msa=copy).key() == first.key()

    def test_a_moved_identical_file_is_the_same_key(self, tmp_path, alignment):
        before = make_request(use_msa_server=False, receptor_msa=alignment).key()
        moved = tmp_path / "moved.a3m"
        alignment.rename(moved)
        assert make_request(use_msa_server=False, receptor_msa=moved).key() == before

    def test_another_content_is_another_key(self, tmp_path, alignment):
        other = write_a3m(tmp_path / "more.a3m", hits=3)
        first = make_request(use_msa_server=False, receptor_msa=alignment)
        assert make_request(use_msa_server=False, receptor_msa=other).key() != first.key()

    def test_a_given_alignment_is_not_the_request_without_one(self, alignment):
        without = make_request(use_msa_server=False)
        assert make_request(use_msa_server=False, receptor_msa=alignment).key() != without.key()

    def test_single_is_an_option_of_its_own(self):
        single = make_request(use_msa_server=False, receptor_msa="single")
        assert single.options["receptor_msa"] == "single"
        assert "receptor_msa" not in single.extra_files
        assert single.key() != make_request(use_msa_server=False).key()

    def test_the_policy_is_in_the_key(self):
        forbidding = make_request(
            use_msa_server=False, receptor_msa="single", msa_policy="forbid-server"
        )
        assert forbidding.options["msa_policy"] == "forbid-server"
        assert forbidding.key() != make_request(use_msa_server=False, receptor_msa="single").key()


class TestWhatMakeRequestRefuses:
    def test_the_server_would_replace_a_given_alignment(self, alignment):
        with pytest.raises(ValueError, match="needs use_msa_server=False"):
            make_request(receptor_msa=alignment)  # use_msa_server is True by default

    def test_the_server_would_search_a_receptor_that_was_asked_to_be_single(self):
        with pytest.raises(ValueError, match="needs use_msa_server=False"):
            make_request(receptor_msa="single")

    def test_forbid_server_with_the_server_on_raises(self, alignment):
        with pytest.raises(ValueError, match="the policy forbids it"):
            make_request(msa_policy="forbid-server", use_msa_server=True, receptor_msa=alignment)

    def test_forbid_server_needs_a_source_for_the_receptor(self):
        with pytest.raises(ValueError, match="needs a source for the MSA of the receptor"):
            make_request(msa_policy="forbid-server", use_msa_server=False)

    def test_an_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="msa_policy must be one of"):
            make_request(msa_policy="no-server")

    def test_a_missing_alignment_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="receptor_msa is not a file"):
            make_request(use_msa_server=False, receptor_msa=tmp_path / "none.a3m")

    @pytest.mark.parametrize("extra", [["--use_msa_server=true"], ["--use-msa-server=true"]])
    def test_extra_arguments_cannot_switch_the_server_on_again(self, extra):
        with pytest.raises(ValueError, match="use_msa_server"):
            make_request(
                msa_policy="forbid-server",
                use_msa_server=False,
                receptor_msa="single",
                extra_args=extra,
            )

    def test_a_query_file_of_mode_predict_names_its_own_msa_files(self, tmp_path, alignment):
        query = tmp_path / "query.json"
        query.write_text("{}", encoding="utf-8")
        runner = OpenFold3Runner()
        for kwargs in ({"receptor_msa": alignment}, {"receptor_msa": "single"}):
            with pytest.raises(ValueError, match="names its own MSA files"):
                runner.make_request(query, name="q", mode="predict", use_msa_server=False, **kwargs)
        with pytest.raises(ValueError):
            runner.make_request(
                query, name="q", mode="predict", use_msa_server=False, msa_policy="forbid-server"
            )
        runner.make_request(query, name="q", mode="predict", use_msa_server=False)  # as before


# ---------------------------------------------------------------------------- prepare and run


@pytest.fixture
def commands(monkeypatch):
    """Replace the OpenFold3 process: the command lines it would have been started with."""
    calls = []
    monkeypatch.setattr(
        openfold, "_run_openfold_command", lambda cmd, output_dir: calls.append(list(cmd))
    )
    # nothing is written by the stub, so there is no output to find
    monkeypatch.setattr(OpenFold3Runner, "_require_output", staticmethod(lambda r, p: None))
    return calls


def forbidding_request(alignment, **kwargs):
    return make_request(
        use_msa_server=False, receptor_msa=alignment, msa_policy="forbid-server", **kwargs
    )


class TestPrepare:
    def test_the_query_holds_the_alignment_for_the_receptor_and_the_dummy_for_the_binder(
        self, tmp_path, alignment
    ):
        request = make_request(use_msa_server=False, receptor_msa=alignment)
        query = OpenFold3Runner().prepare(request, tmp_path / "w")
        chains = chains_of(query)
        receptor = Path(chains["A"]["main_msa_file_paths"][0])
        binder = Path(chains["B"]["main_msa_file_paths"][0])
        assert receptor.read_bytes() == alignment.read_bytes()
        assert binder.read_text(encoding="utf-8") == f">query_B\n{BINDER}\n"
        assert receptor.parent.name == "q_A" and binder.parent.name == "q_B"

    def test_a_single_receptor_gets_the_dummy_like_the_binder(self, tmp_path):
        request = make_request(use_msa_server=False, receptor_msa="single")
        chains = chains_of(OpenFold3Runner().prepare(request, tmp_path / "w"))
        for chain_id, sequence in (("A", RECEPTOR), ("B", BINDER)):
            path = Path(chains[chain_id]["main_msa_file_paths"][0])
            assert path.read_text(encoding="utf-8") == f">query_{chain_id}\n{sequence}\n"

    def test_a_request_without_the_fields_is_prepared_as_before(self, tmp_path):
        request = make_request(use_msa_server=False)
        chains = chains_of(OpenFold3Runner().prepare(request, tmp_path / "w"))
        for chain_id, sequence in (("A", RECEPTOR), ("B", BINDER)):
            path = Path(chains[chain_id]["main_msa_file_paths"][0])
            assert path.read_text(encoding="utf-8") == f">query_{chain_id}\n{sequence}\n"

    def test_an_alignment_of_another_sequence_stops_the_preparation(self, tmp_path):
        other = write_a3m(tmp_path / "other.a3m", query="G" * 85)
        request = make_request(use_msa_server=False, receptor_msa=other)
        with pytest.raises(ValueError, match="not the sequence of chain A"):
            OpenFold3Runner().prepare(request, tmp_path / "w")

    def test_an_alignment_that_was_deleted_stops_it_too(self, tmp_path, alignment):
        request = make_request(use_msa_server=False, receptor_msa=alignment)
        alignment.unlink()
        with pytest.raises(FileNotFoundError):
            OpenFold3Runner().prepare(request, tmp_path / "w")

    def test_a_request_built_by_hand_with_the_server_on_is_refused(self, tmp_path, alignment):
        good = make_request(use_msa_server=False, receptor_msa=alignment)
        options = {**good.options, "use_msa_server": True}
        by_hand = PredictionRequest(
            "of3",
            "q",
            mode="score",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            extra_files=good.extra_files,
            options=options,
        )
        with pytest.raises(ValueError, match="needs use_msa_server=False"):
            OpenFold3Runner().prepare(by_hand, tmp_path / "w")

    def test_forbid_server_refuses_a_hand_built_request_with_the_server_on(self, tmp_path):
        good = make_request(use_msa_server=False, receptor_msa="single", msa_policy="forbid-server")
        by_hand = PredictionRequest(
            "of3",
            "q",
            mode="score",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            options={**good.options, "use_msa_server": True},
        )
        with pytest.raises(ValueError, match="the policy forbids it"):
            OpenFold3Runner().prepare(by_hand, tmp_path / "w")


class TestRun:
    def test_the_command_has_no_server_flag(self, tmp_path, alignment, commands):
        runner = OpenFold3Runner(conda_env="of3")
        runner.run(forbidding_request(alignment), tmp_path)
        (command,) = commands
        assert "--use_msa_server=false" in command
        assert not [word for word in command if "msa_server=true" in word.lower()]
        assert_no_server_flag(command, "of3")  # the helper of the policy reads it the same way

    def test_the_query_of_the_run_names_the_staged_alignment(self, tmp_path, alignment, commands):
        runner = OpenFold3Runner(conda_env="of3")
        runner.run(make_request(use_msa_server=False, receptor_msa=alignment), tmp_path)
        query = tmp_path / "query" / "q_query.json"
        staged = Path(chains_of(query)["A"]["main_msa_file_paths"][0])
        assert staged == tmp_path / "query" / "msas" / "q_A" / "colabfold_main.a3m"
        assert staged.read_bytes() == alignment.read_bytes()

    @pytest.mark.parametrize(
        "extra",
        [
            ["--use_msa_server=true"],
            ["--use-msa-server=true"],
            ["--use-msa-server", "true"],
            ["--use_msa_server"],
        ],
    )
    def test_extra_arguments_that_switch_the_server_on_stop_the_run_before_it_starts(
        self, tmp_path, alignment, commands, extra
    ):
        good = forbidding_request(alignment)
        by_hand = PredictionRequest(
            "of3",
            "q",
            mode="score",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            extra_files=good.extra_files,
            options={**good.options, "extra_args": extra},
        )
        with pytest.raises(ValueError, match="use_msa_server"):
            OpenFold3Runner(conda_env="of3").run(by_hand, tmp_path)
        assert commands == []
        assert not (tmp_path / "query").exists()

    def test_the_policy_is_checked_on_the_hand_built_request_in_run_too(
        self, tmp_path, alignment, commands
    ):
        good = forbidding_request(alignment)
        by_hand = PredictionRequest(
            "of3",
            "q",
            mode="score",
            input_path=P53,
            binder_chain="B",
            receptor_chain="A",
            extra_files=good.extra_files,
            options={**good.options, "use_msa_server": True},
        )
        with pytest.raises(ValueError, match="the policy forbids it"):
            OpenFold3Runner(conda_env="of3").run(by_hand, tmp_path)
        assert commands == []

    def test_a_request_without_a_policy_is_run_as_before(self, tmp_path, commands):
        OpenFold3Runner(conda_env="of3").run(make_request(use_msa_server=False), tmp_path)
        (command,) = commands
        assert "--use_msa_server=false" in command

    def test_the_default_policy_does_not_look_at_extra_arguments(self, tmp_path, commands):
        # the caller who chose the default policy may set the server flag by hand
        request = make_request(use_msa_server=False, extra_args=["--use_msa_server=true"])
        OpenFold3Runner(conda_env="of3").run(request, tmp_path)
        assert commands and "--use_msa_server=true" in commands[0]


class TestABatch:
    def test_the_requests_can_share_a_batch_and_the_path_reaches_the_batched_run(
        self, tmp_path, alignment, monkeypatch
    ):
        runner = OpenFold3Runner(conda_env="of3")
        requests = [
            runner.make_request(
                P53, name=name, **CHAINS, use_msa_server=False, receptor_msa=alignment
            )
            for name in ("one", "two")
        ]
        assert all(runner.supports_batch(request) for request in requests)
        seen = {}

        class StoppedError(Exception):
            pass

        def stop(**kwargs):
            seen.update(kwargs)
            raise StoppedError

        monkeypatch.setattr(openfold, "run_openfold_batched", stop)
        with pytest.raises(StoppedError):
            runner.run_many(requests, tmp_path)
        assert seen["receptor_msa_path"] == alignment
        assert seen["use_msa_server"] is False


# ---------------------------------------------------------------------------- OpenFold3's parser

_PROBE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from openfold3.core.data.io.sequence.msa import parse_msas_direct
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
path = Path(sys.argv[1])
msas = parse_msas_direct([path], max_seq_counts=MSASettings().max_seq_counts)
msa = next(iter(msas.values())).msa if msas else None
print("PROBE " + json.dumps({
    "keys": sorted(msas),
    "rep_id": path.parent.stem,
    "shape": [int(n) for n in msa.shape] if msa is not None else [],
    "query": "".join(msa[0]) if msa is not None else "",
}))
"""


def _openfold3_python():
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
def test_openfold3_reads_the_staged_alignment_with_the_receptor_as_its_query_row(
    tmp_path, alignment
):
    """The parser keeps the file (colabfold_main) and its first row is the receptor sequence."""
    python = _openfold3_python()
    query = openfold.prepare_scoring_query(
        P53, "A", "B", "q", tmp_path / "out", dummy_msa=True, receptor_msa_path=alignment
    )
    staged = Path(chains_of(query)["A"]["main_msa_file_paths"][0])
    probe = subprocess.run(
        [*python, "-c", _PROBE, str(staged)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    lines = [line for line in probe.stdout.splitlines() if line.startswith("PROBE ")]
    found = json.loads(lines[-1][len("PROBE ") :])
    assert found["keys"] == ["colabfold_main"] and found["rep_id"] == "q_A"
    assert found["shape"] == [3, len(RECEPTOR)]  # the insertions are not columns
    assert found["query"] == RECEPTOR
