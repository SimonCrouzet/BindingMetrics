"""What became of the templates of an OpenFold3 run (accounting of ``inference_query_set.json``).

OpenFold3 0.5.0 goes on without a template when it cannot use one: it exits with status 0 and
reports "Successful Queries". The files and messages written here are the ones of real runs of
0.5.0 (1YCR, score and refold, MSA server on and off): ``inference_query_set.json`` after the
template preprocessing, the ``UserWarning`` about the alignment that the MSA server overwrites
(stderr), and the ``Failed to preprocess template alignment`` message (stdout). The "OpenFold3"
that the run tests start is a tiny Python process that writes them.
"""

import json
import logging
import sys
import textwrap
from pathlib import Path

import pytest

from binding_metrics.cli import prediction as cli_prediction
from binding_metrics.cli.run import run_pipeline
from binding_metrics.metrics import _openfold_run, openfold
from binding_metrics.metrics import _openfold_templates as templates
from binding_metrics.metrics._openfold_run import OpenFoldRunInfo, _run_openfold_command
from binding_metrics.metrics._openfold_templates import (
    NO_TEMPLATE_KEPT,
    NOT_RECORDED,
    NOT_REQUESTED,
    PREPROCESSING_FAILED,
    REPLACED_BY_MSA_SERVER,
    StreamNotes,
    account_for_templates,
    describe_missing,
    read_template_accounting,
)
from binding_metrics.protocols.report import _flatten
from tests.test_feat_c_support import StubOpenFold

RECEPTOR_SEQUENCE = (
    "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
)
BINDER_SEQUENCE = "ETFSDLWKLLPEN"


def _overwrite_warning(query: str, chain_id: str, sequence: str) -> str:
    """The UserWarning of colabfold_msa_server.py:1370 as 0.5.0 prints it (one long line)."""
    return (
        "/envs/openfold3/lib/python3.13/site-packages/openfold3/core/data/tools/"
        f"colabfold_msa_server.py:1370: UserWarning: Query {query} chain "
        f"molecule_type=<MoleculeType.PROTEIN: 0> chain_ids=['{chain_id}'] description=None "
        f"sequence='{sequence}' non_canonical_residues=None smiles=None ccd_codes=None "
        "paired_msa_file_paths=[PosixPath('/o/msas/paired/x.npz')] "
        "main_msa_file_paths=[PosixPath('/o/msas/main/x.npz')] "
        "template_alignment_file_path=PosixPath('/o/query/q_receptor.a3m') "
        "template_entry_chain_ids=None template_cif_paths=None template_cif_chain_ids=None "
        "sdf_file_path=None cyclic=False already has itstemplate_alignment_file_path set. "
        "This are now overwritten with a path to the template alignment filefrom the "
        "ColabFold MSA server."
    )


def _failed_preprocessing(path: str, kind: str = "KeyError", message: str = "'x'") -> str:
    """The message of ``preprocess_templates`` (template.py:2042-2060), printed on stdout."""
    return (
        f"Failed to preprocess template alignment {path}:\n\nException:\n{message}\n\n"
        f"Type:\n{kind}\n\nTraceback:\nTraceback (most recent call last):\n"
        f'  File "template.py", line 2042, in preprocess_templates\n{kind}: {message}\n\n'
    )


def _chain(chain_id, entries, alignment=None):
    """A chain of ``inference_query_set.json`` after the template preprocessing."""
    return {
        "molecule_type": "PROTEIN",
        "chain_ids": [chain_id],
        "description": None,
        "sequence": "AAA",
        "template_alignment_file_path": alignment,
        "template_entry_chain_ids": entries,
        "template_cif_paths": None,
        "template_cif_chain_ids": None,
        "cyclic": False,
    }


def _write_query_set(directory, name, chains):
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "seeds": [42],
        "queries": {name: {"query_name": name, "chains": chains, "use_msas": True}},
    }
    (directory / "inference_query_set.json").write_text(json.dumps(body), encoding="utf-8")


def _write_input_query(path, name, alignments):
    """The query file the toolkit hands over: ``alignments`` maps a chain ID to an A3M path."""
    chains = []
    for chain_id, alignment in alignments.items():
        chain = {"molecule_type": "protein", "chain_ids": [chain_id], "sequence": "AAA"}
        if alignment:
            chain["template_alignment_file_path"] = alignment
        chains.append(chain)
    path.write_text(json.dumps({"queries": {name: {"chains": chains}}}), encoding="utf-8")
    return path


class TestStreamNotes:
    def test_the_overwritten_alignment_of_each_chain_is_noted(self):
        notes = StreamNotes()
        notes.feed(_overwrite_warning("q", "A", RECEPTOR_SEQUENCE) + "\n")
        notes.feed("  warnings.warn(\n" + _overwrite_warning("q", "B", BINDER_SEQUENCE) + "\n")
        assert notes.overwritten == {("q", "A"), ("q", "B")}

    def test_a_line_split_across_chunks_is_read_whole(self):
        line = _overwrite_warning("q", "A", RECEPTOR_SEQUENCE) + "\n"
        notes = StreamNotes()
        for start in range(0, len(line), 97):
            notes.feed(line[start : start + 97])
        assert notes.overwritten == {("q", "A")}

    def test_a_failed_preprocessing_gives_the_error_per_alignment(self):
        notes = StreamNotes()
        notes.feed(_failed_preprocessing("/o/query/q_receptor.a3m", "KeyError", "'pdbx_seq'"))
        notes.finish()
        assert notes.failed_templates == {"/o/query/q_receptor.a3m": "KeyError: 'pdbx_seq'"}

    def test_two_failures_in_a_row_are_both_noted(self):
        notes = StreamNotes()
        notes.feed(_failed_preprocessing("/o/q_receptor.a3m", "KeyError", "'a'"))
        notes.feed(_failed_preprocessing("/o/q_binder.a3m", "IndexError", "boom"))
        notes.finish()
        assert notes.failed_templates == {
            "/o/q_receptor.a3m": "KeyError: 'a'",
            "/o/q_binder.a3m": "IndexError: boom",
        }

    def test_a_timeout_is_a_failure(self):
        notes = StreamNotes()
        notes.feed("\n Template preprocessing TIMED OUT after 300s for /o/q_binder.a3m.\n")
        notes.finish()
        assert notes.failed_templates == {"/o/q_binder.a3m": "timed out after 300 s"}

    def test_progress_bars_and_other_lines_note_nothing(self):
        notes = StreamNotes()
        notes.feed("Preprocessing templates:  50%|#####     | 1/2\rPreprocessing templates: 100%\n")
        notes.feed("Warning: No template data provided for chain ['B'] ... skipping...\n")
        notes.finish()
        assert not notes.overwritten and not notes.failed_templates

    def test_a_block_that_ends_with_the_stream_is_kept(self):
        notes = StreamNotes()
        notes.feed("Failed to preprocess template alignment /o/q.a3m:\n\nException:\nboom")
        notes.finish()
        assert notes.failed_templates == {"/o/q.a3m": "boom"}

    def test_notes_of_two_streams_merge(self):
        out, err = StreamNotes(), StreamNotes()
        out.feed(_failed_preprocessing("/o/q.a3m"))
        err.feed(_overwrite_warning("q", "A", "AAA") + "\n")
        out.finish()
        err.finish()
        merged = out.merge(err)
        assert merged.overwritten == {("q", "A")} and "/o/q.a3m" in merged.failed_templates


class TestAccounting:
    """One test per state that the real runs showed."""

    @pytest.fixture
    def alignments(self, tmp_path):
        return {
            "A": str(tmp_path / "query" / "q_receptor.a3m"),
            "B": str(tmp_path / "query" / "q_binder.a3m"),
        }

    def test_score_with_both_templates_read(self, tmp_path, alignments):
        out = tmp_path / "predictions"
        _write_query_set(
            out,
            "q",
            [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", ["binder_B"], "/c/b.npz")],
        )
        query = _write_input_query(tmp_path / "q.json", "q", alignments)
        accounting = account_for_templates(out, query_json=query, use_msa_server=False)
        assert accounting["q"]["A"] == {
            "requested": True,
            "source": "alignment",
            "used": True,
            "cause": None,
            "detail": None,
            "entry_ids": ["receptor_A"],
        }
        assert accounting["q"]["B"]["used"] and accounting["q"]["B"]["cause"] is None
        assert describe_missing(accounting["q"]) is None

    def test_a_failed_preprocessing_is_named_with_its_error(self, tmp_path, alignments):
        """Seen: the binder CIF lacked a column; exit 0, 'Successful Queries: 1', no template."""
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", [], None), _chain("B", [], None)])
        query = _write_input_query(tmp_path / "q.json", "q", alignments)
        notes = StreamNotes()
        notes.feed(
            _failed_preprocessing(alignments["A"], "KeyError", "'pdbx_seq_one_letter_code_can'")
        )
        notes.finish()
        accounting = account_for_templates(out, query_json=query, use_msa_server=False, notes=notes)
        assert accounting["q"]["A"]["used"] is False
        assert accounting["q"]["A"]["cause"] == PREPROCESSING_FAILED
        assert accounting["q"]["A"]["detail"] == "KeyError: 'pdbx_seq_one_letter_code_can'"
        # the other chain asked for a template too, and nothing says why it has none
        assert accounting["q"]["B"]["cause"] == NO_TEMPLATE_KEPT

    def test_the_msa_server_replaces_both_alignments(self, tmp_path, alignments):
        """Seen with the default settings: both warnings, template_entry_chain_ids [] twice."""
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", [], None), _chain("B", [], None)])
        query = _write_input_query(tmp_path / "q.json", "q", alignments)
        notes = StreamNotes()
        notes.feed(
            _overwrite_warning("q", "A", "AAA") + "\n" + _overwrite_warning("q", "B", "AAA") + "\n"
        )
        accounting = account_for_templates(out, query_json=query, use_msa_server=True, notes=notes)
        for chain_id in "AB":
            record = accounting["q"][chain_id]
            assert record["used"] is False and record["cause"] == REPLACED_BY_MSA_SERVER
            assert "overwrote" in record["detail"]

    def test_with_the_server_on_the_cause_is_inferred_when_the_warning_was_not_seen(
        self, tmp_path, alignments
    ):
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", [], None), _chain("B", [], None)])
        query = _write_input_query(tmp_path / "q.json", "q", alignments)
        accounting = account_for_templates(out, query_json=query, use_msa_server=True)
        assert accounting["q"]["A"]["cause"] == REPLACED_BY_MSA_SERVER
        assert "was not seen" in accounting["q"]["A"]["detail"]

    def test_refold_asks_for_the_receptor_template_only(self, tmp_path):
        out = tmp_path / "predictions"
        # OpenFold3 leaves template_entry_chain_ids null for a chain that declared no source
        _write_query_set(
            out, "q", [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", None, None)]
        )
        query = _write_input_query(
            tmp_path / "q.json", "q", {"A": str(tmp_path / "q_receptor.a3m"), "B": None}
        )
        accounting = account_for_templates(out, query_json=query, use_msa_server=False)
        assert accounting["q"]["A"]["used"] is True
        assert accounting["q"]["B"] == {
            "requested": False,
            "source": None,
            "used": False,
            "cause": NOT_REQUESTED,
            "detail": None,
            "entry_ids": [],
        }
        assert describe_missing(accounting["q"]) is None

    def test_without_the_query_file_the_cause_is_not_recorded(self, tmp_path):
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", [], None)])
        accounting = account_for_templates(out)
        assert accounting["q"]["A"]["requested"] is None and accounting["q"]["A"]["used"] is True
        assert accounting["q"]["B"]["cause"] == NOT_RECORDED
        assert describe_missing(accounting["q"]) is None  # whether it was asked for is not known

    def test_an_unreadable_query_file_is_treated_as_unknown(self, tmp_path):
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("B", [], None)])
        (tmp_path / "q.json").write_text("{not json", encoding="utf-8")
        accounting = account_for_templates(out, query_json=tmp_path / "q.json")
        assert accounting["q"]["B"]["requested"] is None
        assert accounting["q"]["B"]["cause"] == NOT_RECORDED

    def test_hits_that_all_fail_the_filters_leave_no_cause_to_name(self, tmp_path, alignments):
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", [], None)])
        query = _write_input_query(tmp_path / "q.json", "q", {"A": alignments["A"]})
        record = account_for_templates(out, query_json=query, use_msa_server=False)["q"]["A"]
        assert record["cause"] == NO_TEMPLATE_KEPT and record["detail"] is None

    def test_a_structure_template_that_was_not_kept(self, tmp_path):
        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", [], None)])
        query = tmp_path / "q.json"
        query.write_text(
            json.dumps(
                {
                    "queries": {
                        "q": {
                            "chains": [
                                {
                                    "chain_ids": ["A"],
                                    "template_cif_paths": ["/t/receptor.cif"],
                                    "template_cif_chain_ids": ["A"],
                                }
                            ]
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        record = account_for_templates(out, query_json=query, use_msa_server=True)["q"]["A"]
        assert record["source"] == "structure" and record["used"] is False
        # the server does not overwrite a CIF-direct template
        assert record["cause"] == NO_TEMPLATE_KEPT

    def test_a_homomer_chain_entry_lists_every_chain_id(self, tmp_path):
        out = tmp_path / "predictions"
        chain = _chain("A", ["x_A"], "/c/a.npz")
        chain["chain_ids"] = ["A", "C"]
        _write_query_set(out, "q", [chain])
        accounting = account_for_templates(out)
        assert set(accounting["q"]) == {"A", "C"}

    def test_a_query_set_of_an_earlier_run_is_ignored(self, tmp_path):
        import os

        out = tmp_path / "predictions"
        _write_query_set(out, "q", [_chain("A", ["x_A"], "/c/a.npz")])
        stale = out / "inference_query_set.json"
        os.utime(stale, (1_000_000, 1_000_000))
        assert account_for_templates(out, not_before=2_000_000) == {}
        assert account_for_templates(tmp_path / "missing") == {}

    def test_the_sentence_names_each_chain_and_its_cause(self):
        chains = {
            "A": {"requested": True, "used": False, "cause": REPLACED_BY_MSA_SERVER},
            "B": {"requested": True, "used": False, "cause": PREPROCESSING_FAILED},
            "C": {"requested": True, "used": True, "cause": None},
        }
        sentence = describe_missing(chains)
        assert "chain A" in sentence and "MSA server" in sentence
        assert "chain B" in sentence and "preprocess" in sentence
        assert "chain C" not in sentence

    def test_chains_with_the_same_cause_share_a_clause(self):
        lost = {"requested": True, "used": False, "cause": REPLACED_BY_MSA_SERVER}
        sentence = describe_missing({"A": lost, "B": lost})
        assert "chains A, B: " in sentence
        assert sentence.count("MSA server") == 1
        failed = {"requested": True, "used": False, "cause": PREPROCESSING_FAILED}
        different = describe_missing(
            {"A": {**failed, "detail": "KeyError: 'a'"}, "B": {**failed, "detail": "IndexError: b"}}
        )
        assert "chain A: " in different and "(KeyError: 'a')" in different
        assert "chain B: " in different and "(IndexError: b)" in different


class TestStoredAccounting:
    def test_the_file_of_the_run_is_read_back(self, tmp_path):
        record = {"requested": True, "used": False, "cause": REPLACED_BY_MSA_SERVER}
        accounting = {"q": {"A": record}}
        assert templates.write_template_accounting(tmp_path, accounting) == (
            tmp_path / "template_accounting.json"
        )
        assert read_template_accounting(tmp_path, "q") == {"A": record}
        assert read_template_accounting(tmp_path) == accounting

    def test_without_the_file_it_is_derived_from_the_query_set(self, tmp_path):
        _write_query_set(tmp_path, "q", [_chain("A", ["x_A"], "/c/a.npz"), _chain("B", [], None)])
        chains = read_template_accounting(tmp_path, "q")
        assert chains["A"]["used"] is True and chains["A"]["requested"] is None
        assert chains["B"]["used"] is False and chains["B"]["cause"] == NOT_RECORDED

    def test_a_query_that_the_file_does_not_have_is_derived(self, tmp_path):
        templates.write_template_accounting(tmp_path, {"other": {"A": {"used": True}}})
        _write_query_set(tmp_path, "q", [_chain("A", ["x_A"], "/c/a.npz")])
        assert read_template_accounting(tmp_path, "q")["A"]["used"] is True
        assert set(read_template_accounting(tmp_path)) == {"other", "q"}

    def test_nothing_known_gives_nothing(self, tmp_path):
        assert read_template_accounting(tmp_path, "q") == {}
        assert read_template_accounting(tmp_path) == {}

    def test_the_names_are_importable_from_the_openfold_module(self):
        assert openfold.read_template_accounting is read_template_accounting
        assert openfold.TEMPLATE_ACCOUNTING_FILE == "template_accounting.json"


# ---------------------------------------------------------------------------
# The run: both streams are read, the file is written, the warning is logged
# ---------------------------------------------------------------------------

_FAKE_RUN = """
import json, pathlib, sys
out = pathlib.Path([a for a in sys.argv if a.startswith("--output_dir=")][0].split("=", 1)[1])
out.mkdir(parents=True, exist_ok=True)
spec = json.loads(sys.argv[1])
(out / "inference_query_set.json").write_text(json.dumps(spec["query_set"]), encoding="utf-8")
sys.stdout.write(spec["stdout"])
sys.stdout.flush()
sys.stderr.write(spec["stderr"])
sys.stderr.flush()
"""


def _fake_run(
    tmp_path, *, query_set, stdout="", stderr="", use_msa_server="false", alignments=None
):
    """Run the fake process and return ``(info, output dir)``."""
    out = tmp_path / "predictions"
    query = _write_input_query(
        tmp_path / "q.json",
        "q",
        alignments or {"A": str(tmp_path / "q_receptor.a3m"), "B": str(tmp_path / "q_binder.a3m")},
    )
    spec = json.dumps({"query_set": query_set, "stdout": stdout, "stderr": stderr})
    cmd = [
        sys.executable,
        "-c",
        textwrap.dedent(_FAKE_RUN),
        spec,
        f"--query_json={query}",
        f"--output_dir={out}",
        f"--use_msa_server={use_msa_server}",
    ]
    return _run_openfold_command(cmd, out), out


def _query_set(chains):
    return {
        "seeds": [42],
        "queries": {"q": {"query_name": "q", "chains": chains, "use_msas": True}},
    }


class TestRunAccounting:
    def test_both_templates_used_is_quiet_and_recorded(self, tmp_path, caplog):
        chains = [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", ["binder_B"], "/c/b.npz")]
        with caplog.at_level(logging.WARNING, logger=templates.logger.name):
            info, out = _fake_run(tmp_path, query_set=_query_set(chains))
        assert isinstance(info, OpenFoldRunInfo)
        assert info.templates["q"]["A"]["used"] and info.templates["q"]["B"]["used"]
        assert "no template" not in caplog.text
        stored = json.loads((out / "template_accounting.json").read_text(encoding="utf-8"))
        assert stored["queries"]["q"]["A"]["entry_ids"] == ["receptor_A"]

    def test_the_server_overwrite_is_read_from_stderr_and_warned_about(self, tmp_path, caplog):
        stderr = (
            _overwrite_warning("q", "A", "AAA")
            + "\n  warnings.warn(\n"
            + _overwrite_warning("q", "B", "AAA")
            + "\n"
            + "x" * 20000  # the warnings are far from the end of stderr
        )
        with caplog.at_level(logging.WARNING, logger=templates.logger.name):
            info, _ = _fake_run(
                tmp_path,
                query_set=_query_set([_chain("A", [], None), _chain("B", [], None)]),
                stderr=stderr,
                use_msa_server="true",
            )
        assert info.templates["q"]["A"]["cause"] == REPLACED_BY_MSA_SERVER
        assert info.templates["q"]["B"]["cause"] == REPLACED_BY_MSA_SERVER
        warnings_text = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("chain A" in text and "MSA server" in text for text in warnings_text)
        assert any("chain B" in text and "MSA server" in text for text in warnings_text)

    def test_a_failed_preprocessing_is_read_from_stdout(self, tmp_path, caplog):
        alignments = {"A": str(tmp_path / "q_receptor.a3m"), "B": str(tmp_path / "q_binder.a3m")}
        with caplog.at_level(logging.WARNING, logger=templates.logger.name):
            info, _ = _fake_run(
                tmp_path,
                query_set=_query_set(
                    [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", [], None)]
                ),
                stdout=_failed_preprocessing(alignments["B"], "KeyError", "'cif key'"),
                alignments=alignments,
            )
        assert info.templates["q"]["A"]["used"] is True
        assert info.templates["q"]["B"]["cause"] == PREPROCESSING_FAILED
        assert info.templates["q"]["B"]["detail"] == "KeyError: 'cif key'"
        message = next(r.getMessage() for r in caplog.records if "chain B" in r.getMessage())
        assert "KeyError: 'cif key'" in message and "'q'" in message

    def test_a_run_without_a_query_set_file_has_no_accounting(self, tmp_path):
        out = tmp_path / "o"
        info = _run_openfold_command([sys.executable, "-c", "pass", f"--output_dir={out}"], out)
        assert info.templates == {}
        assert not (out / "template_accounting.json").exists()

    def test_the_streams_still_reach_the_console(self, tmp_path, capfd):
        _fake_run(
            tmp_path,
            query_set=_query_set([_chain("A", ["x_A"], "/c/a.npz")]),
            stdout="to stdout\n",
            stderr="to stderr\n",
        )
        captured = capfd.readouterr()
        assert "to stdout" in captured.out and "to stderr" in captured.err

    def test_an_option_is_found_in_either_spelling(self):
        cmd = ["run_openfold", "predict", "--query-json=a.json", "--use_msa_server", "true"]
        assert _openfold_run._option_value(cmd, "query_json") == "a.json"
        assert _openfold_run._option_value(cmd, "use_msa_server") == "true"
        assert _openfold_run._option_value(cmd, "runner_yaml") is None
        assert _openfold_run._option_value(["x", "--a=1", "--a=2"], "a") == "2"


# ---------------------------------------------------------------------------
# The result block, the CSV row and the batch
# ---------------------------------------------------------------------------

P53 = Path(__file__).parent.parent / "data" / "example_linear_p53_1YCR.pdb"


class StubWithTemplateFiles(StubOpenFold):
    """The stub model, which also writes the two files of a run that lost its templates."""

    def __init__(self, monkeypatch, chains, *, accounting=None):
        super().__init__(monkeypatch)
        self.chains = chains
        self.accounting = accounting

    def _single(self, kind):
        base = super()._single(kind)

        def run(**kwargs):
            predictions = base(**kwargs)
            name = kwargs["query_name"]
            _write_query_set(predictions, name, self.chains)
            if self.accounting is not None:
                templates.write_template_accounting(predictions, {name: self.accounting})
            return predictions

        return run


_LOST = {
    "A": {
        "requested": True,
        "source": "alignment",
        "used": False,
        "cause": REPLACED_BY_MSA_SERVER,
        "detail": "OpenFold3 warned that it overwrote the alignment path",
        "entry_ids": [],
    },
    "B": {
        "requested": True,
        "source": "alignment",
        "used": False,
        "cause": REPLACED_BY_MSA_SERVER,
        "detail": "OpenFold3 warned that it overwrote the alignment path",
        "entry_ids": [],
    },
}
_KEPT = {
    "A": {**_LOST["A"], "used": True, "cause": None, "detail": None, "entry_ids": ["receptor_A"]},
    "B": {**_LOST["B"], "used": True, "cause": None, "detail": None, "entry_ids": ["binder_B"]},
}


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


class TestResultBlock:
    @pytest.fixture(autouse=True)
    def _version(self, monkeypatch):
        monkeypatch.setattr(
            _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
        )

    def test_a_score_run_that_lost_its_templates_says_so(self, tmp_path, monkeypatch):
        StubWithTemplateFiles(
            monkeypatch,
            [_chain("A", [], None), _chain("B", [], None)],
            accounting=_LOST,
        )
        block = _pipeline(tmp_path, predictor="of3")["prediction"]
        assert block["templates"]["A"]["used"] is False
        assert block["templates"]["A"]["cause"] == REPLACED_BY_MSA_SERVER
        assert block["templates"]["B"]["requested"] is True
        assert (
            "templates: OpenFold3 used no template where the query asked for one" in block["reason"]
        )
        assert "chains A, B: " in block["reason"]

    def test_the_columns_of_the_csv_row(self, tmp_path, monkeypatch):
        StubWithTemplateFiles(
            monkeypatch, [_chain("A", [], None), _chain("B", [], None)], accounting=_LOST
        )
        results = _pipeline(tmp_path, predictor="of3")
        row = _flatten(results)
        assert row["prediction_templates_A_used"] is False
        assert row["prediction_templates_A_cause"] == REPLACED_BY_MSA_SERVER
        assert row["prediction_templates_B_requested"] is True
        assert "prediction_templates_A_entry_ids" not in row  # lists stay in the JSON

    def test_a_run_that_used_its_templates_has_no_reason(self, tmp_path, monkeypatch):
        StubWithTemplateFiles(
            monkeypatch,
            [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", ["binder_B"], "/c/b.npz")],
            accounting=_KEPT,
        )
        block = _pipeline(tmp_path, predictor="of3")["prediction"]
        assert block["templates"]["A"]["entry_ids"] == ["receptor_A"]
        assert "templates" not in block.get("reason", "")

    def test_the_stored_run_gives_the_same_answer_the_second_time(self, tmp_path, monkeypatch):
        stub = StubWithTemplateFiles(
            monkeypatch, [_chain("A", [], None), _chain("B", [], None)], accounting=_LOST
        )
        first = _pipeline(tmp_path, predictor="of3")["prediction"]
        second = _pipeline(tmp_path, predictor="of3")["prediction"]
        assert stub.starts == 1 and second["cache"]["hits"] >= 1
        assert second["templates"] == first["templates"] and second["reason"] == first["reason"]

    def test_an_output_without_the_accounting_file_gives_what_the_query_set_shows(
        self, tmp_path, monkeypatch
    ):
        StubWithTemplateFiles(
            monkeypatch, [_chain("A", ["receptor_A"], "/c/a.npz"), _chain("B", [], None)]
        )
        block = _pipeline(tmp_path, predictor="of3")["prediction"]
        assert block["templates"]["A"]["used"] is True
        assert block["templates"]["B"] == {
            "requested": None,
            "source": None,
            "used": False,
            "cause": NOT_RECORDED,
            "detail": None,
            "entry_ids": [],
        }
        assert "templates" not in block.get("reason", "")  # whether it asked is not known

    def test_the_markdown_report_has_a_templates_row(self, tmp_path, monkeypatch):
        from binding_metrics.protocols.report import write_report

        StubWithTemplateFiles(
            monkeypatch, [_chain("A", [], None), _chain("B", [], None)], accounting=_LOST
        )
        results = _pipeline(tmp_path, predictor="of3")
        write_report(results, tmp_path, "s", fmt="json", summary=True)
        report = (tmp_path / "s_report.md").read_text(encoding="utf-8")
        row = next(line for line in report.splitlines() if line.startswith("| Templates"))
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells == [
            "Templates",
            "A: not used (replaced_by_msa_server), B: not used (replaced_by_msa_server)",
        ]

    def test_a_run_without_accounting_has_no_templates_row(self, tmp_path, monkeypatch):
        from binding_metrics.protocols.report import write_report

        StubOpenFold(monkeypatch)
        results = _pipeline(tmp_path, predictor="of3")
        write_report(results, tmp_path, "s", fmt="json", summary=True)
        assert "Templates" not in (tmp_path / "s_report.md").read_text(encoding="utf-8")

    def test_the_label_names_each_chain(self):
        from binding_metrics.protocols.report import _templates_label

        label = _templates_label(
            {
                "A": {"requested": True, "used": True, "cause": None},
                "B": {"requested": False, "used": False, "cause": NOT_REQUESTED},
                "C": {"requested": True, "used": False, "cause": PREPROCESSING_FAILED},
                "D": {"requested": None, "used": False, "cause": None},
            }
        )
        assert label == (
            f"A: used, B: none asked for, C: not used ({PREPROCESSING_FAILED}), D: not used"
        )

    def test_a_stub_run_without_any_file_adds_nothing(self, tmp_path, monkeypatch):
        StubOpenFold(monkeypatch)
        block = _pipeline(tmp_path, predictor="of3")["prediction"]
        assert "templates" not in block

    def test_the_openfold_step_without_a_predictor_records_it_too(self, tmp_path, monkeypatch):
        StubWithTemplateFiles(
            monkeypatch, [_chain("A", [], None), _chain("B", [], None)], accounting=_LOST
        )
        block = _pipeline(tmp_path)["openfold"]
        assert block["templates"]["A"]["cause"] == REPLACED_BY_MSA_SERVER
        assert "chains A, B: " in block["reason"]

    def test_record_templates_leaves_a_missing_folder_alone(self, tmp_path):
        block = {"model": "of3"}
        cli_prediction.record_templates(block, None, "q")
        cli_prediction.record_templates(block, tmp_path / "nowhere", "q")
        assert block == {"model": "of3"}

    def test_the_reason_is_joined_to_an_existing_one(self, tmp_path):
        templates.write_template_accounting(tmp_path, {"q": _LOST})
        block = {"reason": "interface PAE: no matrix"}
        cli_prediction.record_templates(block, tmp_path, "q")
        assert block["reason"].startswith("interface PAE: no matrix; templates: ")
