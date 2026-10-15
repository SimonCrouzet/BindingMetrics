"""The dummy MSA of an OpenFold3 query for a run without the ColabFold MSA server.

OpenFold3's input reference suggests MSA-free inference through a dummy MSA that holds only the
query sequence and discourages leaving the MSA input out. OpenFold3 0.5.0 builds the same dummy
for a chain without MSA files (with a warning), so the query written with it predicts the same
(one complex, three seeds, no template: within 0.03 ipTM and 0.1 A of binder RMSD of the run that
leaves the input out). It reads an MSA file only when its name is a key of
``MSASettings.max_seq_counts``: a file named otherwise is skipped and the chain fails with
``IndexError`` while its features are built, which a real run showed. Nothing here runs
OpenFold3 except the optional test marked ``openfold``, which runs its MSA parser on CPU.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_cli, _openfold_run, openfold
from binding_metrics.predictors.of3_runner import OpenFold3Runner

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

DATA = Path(__file__).parent.parent / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
RECEPTOR_SEQUENCE = (
    "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
)
BINDER_SEQUENCE = "ETFSDLWKLLPEN"


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


class TestTheFile:
    def test_one_folder_per_chain_with_the_name_that_openfold3_parses(self, tmp_path):
        path = _openfold_run._write_dummy_msa("ETFSDLWKLLPEN", "B", tmp_path / "msas" / "q_B")
        assert path == tmp_path / "msas" / "q_B" / "colabfold_main.a3m"
        # a key of MSASettings.max_seq_counts in OpenFold3 0.5.0; any other name is skipped
        assert path.stem == "colabfold_main"

    def test_the_file_holds_the_query_and_nothing_else(self, tmp_path):
        path = _openfold_run._write_dummy_msa("ETFSDLWKLLPEN", "B", tmp_path / "x")
        assert path.read_text(encoding="utf-8") == ">query_B\nETFSDLWKLLPEN\n"

    def test_the_fields_are_empty_when_no_dummy_is_wanted(self, tmp_path):
        assert (
            _openfold_run._msa_fields(False, sequence="AG", chain_id="A", directory=tmp_path) == {}
        )
        assert not (tmp_path / "colabfold_main.a3m").exists()


class TestQueries:
    @pytest.mark.parametrize("build", ["prepare_refolding_query", "prepare_scoring_query"])
    def test_each_chain_gets_its_own_dummy_msa(self, tmp_path, build):
        query = getattr(openfold, build)(P53, "A", "B", "q", tmp_path, dummy_msa=True)
        chains = _chains(query)
        for chain_id, sequence in (("A", RECEPTOR_SEQUENCE), ("B", BINDER_SEQUENCE)):
            (path,) = chains[chain_id]["main_msa_file_paths"]
            assert Path(path) == tmp_path / "msas" / f"q_{chain_id}" / "colabfold_main.a3m"
            assert Path(path).read_text(encoding="utf-8").splitlines()[1] == sequence
        # the folder name is what OpenFold3 takes as the id of the alignment: one per chain
        folders = {Path(c["main_msa_file_paths"][0]).parent.name for c in chains.values()}
        assert len(folders) == 2

    @pytest.mark.parametrize("build", ["prepare_refolding_query", "prepare_scoring_query"])
    def test_the_default_leaves_the_msa_input_out_as_before(self, tmp_path, build):
        query = getattr(openfold, build)(P53, "A", "B", "q", tmp_path)
        assert all("main_msa_file_paths" not in c for c in _chains(query).values())
        assert not (tmp_path / "msas").exists()

    def test_the_templates_and_the_msa_go_together(self, tmp_path):
        for mode in ("alignment", "structure"):
            chains = _chains(
                openfold.prepare_scoring_query(
                    P53, "A", "B", "q", tmp_path / mode, template_mode=mode, dummy_msa=True
                )
            )
            assert "main_msa_file_paths" in chains["A"]
            assert ("template_cif_paths" in chains["A"]) is (mode == "structure")
            assert ("template_alignment_file_path" in chains["A"]) is (mode == "alignment")

    def test_a_binder_with_modified_residues_gets_the_parent_letters(self, tmp_path):
        query = openfold.prepare_refolding_query(
            CYCLOSPORIN, "A", "C", "q", tmp_path, dummy_msa=True, binder_cyclic=False
        )
        chain = _chains(query)["C"]
        text = Path(chain["main_msa_file_paths"][0]).read_text(encoding="utf-8")
        assert text.splitlines()[1] == chain["sequence"] == "ALLVTAGLVLA"

    @pytest.mark.parametrize(
        "function",
        [openfold.prepare_batched_scoring_queries, openfold.prepare_batched_refolding_queries],
    )
    def test_the_batched_queries_write_a_pair_of_files_per_sample(self, tmp_path, function):
        samples = [
            openfold._BatchSample("s1", P53, "A", "B"),
            openfold._BatchSample("s2", P53, "A", "B"),
        ]
        path = function(samples, tmp_path, dummy_msa=True)
        queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
        folders = {
            Path(c["main_msa_file_paths"][0]).parent.name
            for q in queries.values()
            for c in q["chains"]
        }
        assert folders == {"s1_A", "s1_B", "s2_A", "s2_B"}
        assert all(
            (tmp_path / "msas" / folder / "colabfold_main.a3m").is_file() for folder in folders
        )


class TestWrappers:
    @staticmethod
    def _capture(monkeypatch, tmp_path):
        seen = {}

        def prepare(**kwargs):
            seen.clear()
            seen.update(kwargs)
            return tmp_path / "q.json"

        monkeypatch.setattr(openfold, "prepare_scoring_query", prepare)
        monkeypatch.setattr(openfold, "prepare_refolding_query", prepare)
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: tmp_path)
        return seen

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_a_run_without_the_server_gets_the_dummy_msa(self, tmp_path, monkeypatch, runner):
        seen = self._capture(monkeypatch, tmp_path)
        getattr(openfold, runner)(P53, "A", "B", "q", tmp_path, use_msa_server=False)
        assert seen["dummy_msa"] is True

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_a_run_with_the_server_does_not(self, tmp_path, monkeypatch, runner):
        seen = self._capture(monkeypatch, tmp_path)
        getattr(openfold, runner)(P53, "A", "B", "q", tmp_path)
        assert "dummy_msa" not in seen  # the server would overwrite it

    def test_the_choice_can_be_made_by_hand(self, tmp_path, monkeypatch):
        seen = self._capture(monkeypatch, tmp_path)
        openfold.run_openfold_scoring(
            P53, "A", "B", "q", tmp_path, use_msa_server=False, dummy_msa=False
        )
        assert "dummy_msa" not in seen
        openfold.run_openfold_scoring(P53, "A", "B", "q", tmp_path, dummy_msa=True)
        assert seen["dummy_msa"] is True

    def test_the_batched_run_follows_the_server_too(self, tmp_path, monkeypatch):
        seen = {}

        def prepare(samples, output_dir, **kwargs):
            seen.clear()
            seen.update(kwargs)
            return Path(output_dir) / "batch_query.json"

        monkeypatch.setattr(openfold, "prepare_batched_scoring_queries", prepare)
        monkeypatch.setattr(openfold, "run_openfold", lambda **kw: tmp_path)
        samples = [openfold._BatchSample("s", P53, "A", "B")]
        openfold.run_openfold_batched(samples, tmp_path, use_msa_server=False)
        assert seen["dummy_msa"] is True
        openfold.run_openfold_batched(samples, tmp_path)
        assert "dummy_msa" not in seen

    def test_the_real_query_of_a_run_without_the_server_has_the_files(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run(query_json, **kwargs):
            captured["chains"] = _chains(Path(query_json))
            return Path(kwargs["output_dir"])

        monkeypatch.setattr(openfold, "run_openfold", fake_run)
        openfold.run_openfold_scoring(P53, "A", "B", "q", tmp_path, use_msa_server=False)
        for chain in captured["chains"].values():
            assert Path(chain["main_msa_file_paths"][0]).is_file()


class TestRunner:
    @staticmethod
    def _request(**kwargs):
        return OpenFold3Runner().make_request(
            P53, name="q", binder_chain="B", receptor_chain="A", **kwargs
        )

    def test_prepare_tells_the_query_builder_when_the_server_is_off(self, tmp_path, monkeypatch):
        seen = {}

        def prepare(**kwargs):
            seen.clear()
            seen.update(kwargs)
            return Path(kwargs["output_dir"]) / "q_query.json"

        monkeypatch.setattr(openfold, "prepare_scoring_query", prepare)
        runner = OpenFold3Runner()
        runner.prepare(self._request(use_msa_server=False), tmp_path / "off")
        assert seen["dummy_msa"] is True
        runner.prepare(self._request(), tmp_path / "on")
        assert "dummy_msa" not in seen

    def test_a_run_calls_the_wrapper_as_before(self, tmp_path, monkeypatch):
        """The wrapper decides from use_msa_server, so the call does not name the dummy MSA."""
        seen = {}

        def fake_run(**kwargs):
            seen.update(kwargs)
            out = Path(kwargs["output_dir"]) / "predictions"
            seed_dir = out / "q" / "seed_42"
            seed_dir.mkdir(parents=True)
            (seed_dir / "q_seed_42_sample_1_confidences_aggregated.json").write_text(
                "{}", encoding="utf-8"
            )
            return out

        monkeypatch.setattr(openfold, "run_openfold_scoring", fake_run)
        (tmp_path / "w").mkdir()
        OpenFold3Runner().run(self._request(use_msa_server=False), tmp_path / "w")
        assert seen["use_msa_server"] is False and "dummy_msa" not in seen


class TestCommandLine:
    @staticmethod
    def _argv(command, out, *extra):
        return [
            "prog", command, "--complex", str(P53), "--receptor-chain", "A",
            "--binder-chain", "B", "--query-name", "q", "--output-dir", str(out), *extra,
        ]  # fmt: skip

    @pytest.mark.parametrize("command", ["prepare-query", "prepare-scoring-query"])
    @pytest.mark.parametrize("extra, written", [([], False), (["--dummy-msa"], True)])
    def test_the_prepare_commands_take_the_option(
        self, tmp_path, monkeypatch, command, extra, written
    ):
        monkeypatch.setattr("sys.argv", self._argv(command, tmp_path / "out", *extra))
        openfold.main()
        chains = _chains(tmp_path / "out" / "q_query.json")
        assert ("main_msa_file_paths" in chains["A"]) is written

    @pytest.mark.parametrize(
        "command, target",
        [("score", "run_openfold_scoring"), ("refold", "run_openfold_refolding")],
    )
    def test_the_run_commands_follow_no_msa_server(self, tmp_path, monkeypatch, command, target):
        seen = {}
        monkeypatch.setattr(openfold, target, lambda **kw: seen.update(kw) or tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(_openfold_cli, "_print_metrics", lambda *a, **kw: None)
        monkeypatch.setattr("sys.argv", self._argv(command, tmp_path, "--no-msa-server"))
        openfold.main()
        assert seen["use_msa_server"] is False  # the wrapper adds the dummy MSA from this

    def test_the_pipeline_with_the_server_off_reaches_the_wrapper_with_it(
        self, tmp_path, monkeypatch
    ):
        from binding_metrics.cli.run import run_pipeline

        seen = {}
        monkeypatch.setattr(
            openfold, "run_openfold_scoring", lambda **kw: seen.update(kw) or tmp_path
        )
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        run_pipeline(
            P53,
            tmp_path,
            skip_prep=True,
            skip_relax=True,
            metrics=frozenset({"openfold"}),
            peptide_chain="B",
            receptor_chain="A",
            openfold_conda_env=None,
            openfold_use_msa_server=False,
        )
        assert seen["use_msa_server"] is False


# ---------------------------------------------------------------------------
# OpenFold3's own MSA parser (optional)
# ---------------------------------------------------------------------------

_PROBE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from openfold3.core.data.io.sequence.msa import parse_msas_direct
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
path = Path(sys.argv[1])
msas = parse_msas_direct([path], max_seq_counts=MSASettings().max_seq_counts)
shape = [int(n) for n in next(iter(msas.values())).msa.shape] if msas else []
print("PROBE " + json.dumps({"keys": sorted(msas), "rep_id": path.parent.stem, "shape": shape}))
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
def test_openfold3_parses_the_dummy_msa_to_one_row(tmp_path):
    """OpenFold3's parser keeps the file (its name is a key of max_seq_counts) as one row."""
    python = _openfold3_python()
    path = _openfold_run._write_dummy_msa(BINDER_SEQUENCE, "B", tmp_path / "msas" / "q_B")
    probe = subprocess.run(
        [*python, "-c", _PROBE, str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    lines = [line for line in probe.stdout.splitlines() if line.startswith("PROBE ")]
    assert lines, probe.stdout[-2000:]
    found = json.loads(lines[-1][len("PROBE ") :])
    assert found["keys"] == ["colabfold_main"]
    assert found["rep_id"] == "q_B"
    assert found["shape"][0] == 1 and found["shape"][1] == len(BINDER_SEQUENCE)


@pytest.mark.openfold
def test_a_file_with_another_name_would_be_skipped_by_openfold3(tmp_path):
    """Why the file is called colabfold_main.a3m: OpenFold3 skips any other name."""
    python = _openfold3_python()
    folder = tmp_path / "q_B"
    folder.mkdir()
    path = folder / "my_msa.a3m"
    path.write_text(f">query_B\n{BINDER_SEQUENCE}\n", encoding="utf-8")
    probe = subprocess.run(
        [*python, "-c", _PROBE, str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    lines = [line for line in probe.stdout.splitlines() if line.startswith("PROBE ")]
    assert json.loads(lines[-1][len("PROBE ") :])["keys"] == []
