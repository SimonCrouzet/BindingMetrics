"""The ``cyclic`` flag of the binder chain in an OpenFold3 query (#77).

OpenFold3 0.4.5 added a per-chain ``cyclic`` field that wraps the relative positions of a chain,
which describes a head-to-tail closure. The field is read from the source of OpenFold3 v0.5.0
(``inference_query_format.py``, ``relpos.py``) and an example query; nothing here runs OpenFold3,
so what the model predicts for a cyclic binder is not tested. The version probe is stubbed.
"""

import json
import logging
import warnings
from pathlib import Path

import pytest

from binding_metrics.metrics import _openfold_cli, _openfold_run, openfold
from binding_metrics.predictors.of3_runner import OpenFold3Runner

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

DATA = Path(__file__).parent.parent / "data"
# chain C is closed head to tail and has D-Ala and N-methylated residues
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
# chain I (SFTI-1) is closed head to tail, has standard residues only, and a disulfide
SFTI1 = DATA / "example_bicyclic_sfti1_3P8F.cif"
P53 = DATA / "example_linear_p53_1YCR.pdb"  # chain B is linear
LOGGER = "binding_metrics.metrics._openfold_run"


@pytest.fixture(autouse=True)
def _openfold3_environment(tmp_path, monkeypatch):
    """OpenFold3 0.5.0 in this interpreter, no user-default runner.yml, an empty version cache."""
    monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "openfold_cache"))
    monkeypatch.setattr(
        _openfold_run, "installed_openfold3_version", lambda python_cmd=None: "0.5.0"
    )
    monkeypatch.setattr(_openfold_run, "_VERSION_BY_PYTHON", {})


@pytest.fixture
def set_version(monkeypatch):
    """Make the version probe answer ``version`` (None: unreadable); returns the list of calls."""
    calls = []

    def _set(version):
        def probe(python_cmd=None):
            calls.append(python_cmd)
            return version

        monkeypatch.setattr(_openfold_run, "installed_openfold3_version", probe)
        return calls

    return _set


def _chains(query_json: Path, query_name: str = "q") -> dict:
    query = json.loads(query_json.read_text(encoding="utf-8"))
    return {c["chain_ids"][0]: c for c in query["queries"][query_name]["chains"]}


def _refold(tmp_path, structure=SFTI1, receptor="A", binder="I", **kwargs):
    return openfold.prepare_refolding_query(structure, receptor, binder, "q", tmp_path, **kwargs)


def _score(tmp_path, structure=SFTI1, receptor="A", binder="I", **kwargs):
    return openfold.prepare_scoring_query(structure, receptor, binder, "q", tmp_path, **kwargs)


class TestQueryChain:
    def test_the_flag_is_written_only_when_asked_for(self):
        plain = _openfold_run._query_chain("C", "AGLV", {})
        assert "cyclic" not in plain
        assert _openfold_run._query_chain("C", "AGLV", {}, cyclic=True)["cyclic"] is True
        assert "cyclic" not in _openfold_run._query_chain("C", "AGLV", {}, cyclic=False)


class TestDefaultIsAuto:
    @pytest.mark.parametrize(
        "function", [openfold.prepare_refolding_query, openfold.prepare_scoring_query]
    )
    def test_the_signatures_default_to_auto(self, function):
        import inspect

        assert inspect.signature(function).parameters["binder_cyclic"].default == "auto"

    @pytest.mark.parametrize(
        "function",
        [
            openfold.run_openfold_scoring,
            openfold.run_openfold_refolding,
            openfold.run_openfold_batched,
            openfold.prepare_batched_scoring_queries,
            openfold.prepare_batched_refolding_queries,
        ],
    )
    def test_the_other_functions_default_to_auto_too(self, function):
        import inspect

        assert inspect.signature(function).parameters["binder_cyclic"].default == "auto"

    def test_the_runner_default_is_auto(self):
        import inspect

        parameters = inspect.signature(OpenFold3Runner.make_request).parameters
        assert parameters["binder_cyclic"].default == "auto"


class TestAuto:
    """``"auto"`` writes the flag for a head-to-tail binder of standard residues, no other chain."""

    @pytest.mark.parametrize("build", [_refold, _score])
    def test_a_head_to_tail_binder_gets_the_flag_and_the_receptor_does_not(self, tmp_path, build):
        chains = _chains(build(tmp_path))
        assert chains["I"]["cyclic"] is True
        assert "cyclic" not in chains["A"]

    @pytest.mark.parametrize("build", [_refold, _score])
    def test_a_linear_binder_gets_nothing(self, tmp_path, build, caplog):
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            chains = _chains(build(tmp_path, structure=P53, receptor="A", binder="B"))
        assert all("cyclic" not in chain for chain in chains.values())
        assert caplog.records == []  # a linear binder is not worth a word

    def test_a_linear_binder_does_not_ask_for_the_version(self, tmp_path, set_version):
        calls = set_version("0.5.0")
        # "alignment" does not look at the version; the default "structure" does (it needs 0.4.2)
        _refold(tmp_path, structure=P53, receptor="A", binder="B", template_mode="alignment")
        assert calls == []

    def test_the_sequence_and_the_modified_residues_are_those_of_a_query_without_the_flag(
        self, tmp_path
    ):
        flagged = _chains(_refold(tmp_path / "auto"))["I"]
        plain = _chains(_refold(tmp_path / "off", binder_cyclic=False))["I"]
        assert {k: v for k, v in flagged.items() if k != "cyclic"} == plain

    def test_the_flag_is_logged_at_info(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            _refold(tmp_path)
        (record,) = [r for r in caplog.records if "cyclic: true" in r.getMessage()]
        assert record.levelno == logging.INFO and "0.5.0" in record.getMessage()
        assert "standard residues only" in record.getMessage()

    @pytest.mark.parametrize("version", ["0.4.5", "0.5.0", "0.5.1.dev3", "1.0"])
    def test_a_new_enough_openfold3_gets_it(self, tmp_path, set_version, version):
        set_version(version)
        assert _chains(_refold(tmp_path))["I"]["cyclic"] is True

    def test_a_bicyclic_peptide_with_a_head_to_tail_bond_gets_it(self, tmp_path):
        """SFTI-1: a backbone ring and a disulfide; only the ring has an OpenFold3 input."""
        chains = _chains(_refold(tmp_path, structure=SFTI1, receptor="A", binder="I"))
        assert chains["I"]["cyclic"] is True
        assert chains["I"]["sequence"] == "GRCTKSIPPICFPD"
        assert "non_canonical_residues" not in chains["I"]

    def test_a_protonation_or_cross_link_variant_is_still_standard(self, tmp_path):
        """HID, CYX and the like are sent as their parent letter: one token each, as a standard
        residue (the SFTI-1 cysteines renamed CYX stay a standard-residue binder)."""
        import gemmi

        structure = gemmi.read_structure(str(SFTI1))
        for residue in structure[0]["I"]:
            if residue.name == "CYS":
                residue.name = "CYX"
        path = tmp_path / "cyx.pdb"
        structure.write_pdb(str(path))
        decision = openfold.decide_binder_cyclic(path, "I")
        assert decision == openfold.BinderCyclicDecision(True)


class TestModifiedResiduesLeaveTheBinderLinear:
    """Measured once: with the flag the cyclosporin got worse (see decide_binder_cyclic)."""

    @pytest.mark.parametrize("build", [_refold, _score])
    def test_auto_writes_no_flag_for_cyclosporin(self, tmp_path, build):
        chains = _chains(build(tmp_path, structure=CYCLOSPORIN, binder="C"))
        assert "cyclic" not in chains["C"] and "cyclic" not in chains["A"]
        # the modified residues are still sent, as before
        assert chains["C"]["non_canonical_residues"]["1"] == "DAL"

    def test_the_decision_says_why_and_states_the_measurement_as_one_complex(self, set_version):
        set_version("0.5.0")
        decision = openfold.decide_binder_cyclic(CYCLOSPORIN, "C")
        assert decision.cyclic is False
        for stated in (
            "chain C is closed head to tail but has modified residues",
            "ABA, BMT, DAL, MLE, MVA, SAR",
            "linear chain",
            "token of its own",
            "one complex",
            "0.91-0.92 to 0.78-0.81",
            "0.5-0.7 A to 3.0-4.8 A",
            "SFTI-1",
            "7.40 A",
            "1.38 A",
            "--openfold-cyclic on",
        ):
            assert stated in decision.reason

    def test_the_log_warns_with_the_reason(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _refold(tmp_path, structure=CYCLOSPORIN, binder="C")
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "modified residues" in warning.getMessage()
        assert not [
            r for r in caplog.records if "cyclic: true" in r.getMessage() and r.levelno < 30
        ]

    def test_the_record_of_the_pipeline_is_decided_quietly(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            decision = openfold.decide_binder_cyclic(CYCLOSPORIN, "C", log=False)
        assert decision.cyclic is False and "modified residues" in decision.reason
        assert caplog.text == ""

    def test_true_still_forces_the_flag_and_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            chains = _chains(
                _refold(tmp_path, structure=CYCLOSPORIN, binder="C", binder_cyclic=True)
            )
        assert chains["C"]["cyclic"] is True
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "binder_cyclic=True writes 'cyclic: true' on chain C" in warning.getMessage()
        assert "DAL" in warning.getMessage() and "one complex" in warning.getMessage()

    def test_true_for_a_standard_binder_does_not_warn(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            chains = _chains(_refold(tmp_path, binder_cyclic=True))
        assert chains["I"]["cyclic"] is True and caplog.records == []

    def test_false_never_writes_it(self, tmp_path):
        chains = _chains(_refold(tmp_path, structure=CYCLOSPORIN, binder="C", binder_cyclic=False))
        assert "cyclic" not in chains["C"]

    def test_a_binder_with_selenocysteine_counts_as_modified(self, tmp_path):
        """OpenFold3 tokenises SEC per atom: the letter U is not a standard residue."""
        import gemmi

        structure = gemmi.read_structure(str(SFTI1))
        for residue in structure[0]["I"]:
            if residue.seqid.num == 3:
                residue.name = "SEC"
        path = tmp_path / "sec.pdb"
        structure.write_pdb(str(path))
        decision = openfold.decide_binder_cyclic(path, "I")
        assert decision.cyclic is False and "SEC" in decision.reason

    def test_the_old_version_reason_comes_first(self, set_version):
        """A version that cannot take the field is the first thing to say."""
        set_version("0.4.4")
        decision = openfold.decide_binder_cyclic(CYCLOSPORIN, "C")
        assert "predates the 'cyclic' chain field" in decision.reason
        assert "modified residues" not in decision.reason

    def test_a_structure_that_cannot_be_classified_leaves_the_binder_linear(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(_openfold_run, "_modified_binder_residues", lambda *a, **k: None)
        chains = _chains(_refold(tmp_path))
        assert "cyclic" not in chains["I"]
        decision = openfold.decide_binder_cyclic(SFTI1, "I")
        assert "could not be classified" in decision.reason

    def test_the_classification_is_the_one_of_the_query(self):
        modified = _openfold_run._modified_binder_residues
        assert modified(SFTI1, "I") == []
        assert modified(P53, "B") == []
        assert modified(CYCLOSPORIN, "C") == ["ABA", "BMT", "DAL", "MLE", "MVA", "SAR"]
        assert modified(CYCLOSPORIN, "Z") is None  # no such chain


class TestOtherClosuresAreNeverWritten:
    @pytest.mark.parametrize(
        "name, chain",
        [("example_lactam_somatostatin_1XY4.cif", "A"), ("example_staple_3V3B.pdb", "C")],
    )
    def test_a_disulfide_a_lactam_and_a_staple_are_not_a_cyclic_flag(self, name, chain, caplog):
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            decision = openfold.decide_binder_cyclic(DATA / name, chain)
        assert decision == openfold.BinderCyclicDecision(False)
        assert caplog.records == []


class TestOff:
    @pytest.mark.parametrize("build", [_refold, _score])
    def test_false_writes_nothing_even_for_a_head_to_tail_binder(
        self, tmp_path, build, set_version
    ):
        calls = set_version("0.5.0")
        chains = _chains(build(tmp_path, binder_cyclic=False, template_mode="alignment"))
        assert all("cyclic" not in chain for chain in chains.values())
        assert calls == []  # neither the structure nor the version is looked at


class TestOn:
    @pytest.mark.parametrize("build", [_refold, _score])
    def test_true_writes_it_on_the_binder_only_whatever_the_structure_says(self, tmp_path, build):
        chains = _chains(
            build(tmp_path, structure=P53, receptor="A", binder="B", binder_cyclic=True)
        )
        assert chains["B"]["cyclic"] is True
        assert "cyclic" not in chains["A"]

    def test_true_is_refused_before_anything_is_written_when_openfold3_is_too_old(
        self, tmp_path, set_version
    ):
        set_version("0.4.4")
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="0.4.5") as info:
            _refold(out, binder_cyclic=True)
        assert "0.4.4" in str(info.value)
        assert not out.exists()

    def test_true_with_an_unreadable_version_warns_and_writes(self, tmp_path, set_version, caplog):
        set_version(None)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            chains = _chains(_refold(tmp_path, binder_cyclic=True))
        assert chains["I"]["cyclic"] is True
        assert "Could not read the installed OpenFold3 version" in caplog.text

    def test_true_with_an_unreadable_version_is_decided_quietly_when_log_is_off(
        self, set_version, caplog
    ):
        """The pipeline repeats the decision with ``log=False`` to record it in its result."""
        set_version(None)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            decision = _openfold_run.decide_binder_cyclic(CYCLOSPORIN, "C", True, log=False)
        assert decision.cyclic is True and not decision.reason
        assert caplog.text == ""


class TestVersionBelowTheField:
    """OpenFold3 older than 0.4.5 rejects the field, so "auto" leaves the binder linear."""

    def test_the_binder_stays_linear_and_the_log_says_why(self, tmp_path, set_version, caplog):
        set_version("0.4.4")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            chains = _chains(_refold(tmp_path))
        assert "cyclic" not in chains["I"]
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "0.4.4" in warning.getMessage() and "0.4.5" in warning.getMessage()
        assert "linear" in warning.getMessage()

    def test_the_decision_carries_the_reason(self, set_version):
        set_version("0.4.4")
        decision = openfold.decide_binder_cyclic(CYCLOSPORIN, "C")
        assert decision.cyclic is False
        assert "0.4.4" in decision.reason and "head to tail" in decision.reason

    def test_a_linear_binder_has_no_reason(self, set_version):
        set_version("0.4.4")
        assert openfold.decide_binder_cyclic(P53, "B") == openfold.BinderCyclicDecision(False)


class TestUnreadableVersion:
    def test_the_binder_stays_linear_and_the_warning_names_the_way_to_force_it(
        self, tmp_path, set_version, caplog
    ):
        set_version(None)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            chains = _chains(_refold(tmp_path))
        assert "cyclic" not in chains["I"]
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "--openfold-cyclic on" in warning.getMessage()
        assert "could not be read" in warning.getMessage()

    def test_the_decision_carries_the_reason(self, set_version):
        set_version(None)
        decision = openfold.decide_binder_cyclic(CYCLOSPORIN, "C")
        assert decision.cyclic is False and "could not be read" in decision.reason


class TestVersionProbe:
    def test_the_conda_environment_is_the_one_asked(self, set_version):
        calls = set_version("0.5.0")
        openfold.decide_binder_cyclic(CYCLOSPORIN, "C", conda_env="of3")
        (command,) = calls
        assert command[1:] == ["run", "-n", "of3", "python"]

    def test_the_current_interpreter_is_asked_without_an_environment(self, set_version):
        calls = set_version("0.5.0")
        openfold.decide_binder_cyclic(CYCLOSPORIN, "C")
        assert calls == [None]

    def test_an_empty_environment_name_means_the_current_interpreter(self, set_version):
        calls = set_version("0.5.0")
        openfold.decide_binder_cyclic(CYCLOSPORIN, "C", conda_env="")
        assert calls == [None]

    def test_a_batch_asks_a_conda_environment_once(self, tmp_path, set_version):
        calls = set_version("0.5.0")
        for index in range(3):
            _refold(tmp_path / str(index), conda_env="of3")
        assert len(calls) == 1

    def test_an_unreadable_answer_is_asked_again(self, set_version):
        calls = set_version(None)
        for _ in range(2):
            openfold.decide_binder_cyclic(CYCLOSPORIN, "C", conda_env="of3")
        assert len(calls) == 2

    def test_two_environments_are_asked_separately(self, set_version):
        calls = set_version("0.5.0")
        openfold.decide_binder_cyclic(CYCLOSPORIN, "C", conda_env="one")
        openfold.decide_binder_cyclic(CYCLOSPORIN, "C", conda_env="two")
        assert len(calls) == 2


class TestSomethingElseThanAChoice:
    @pytest.mark.parametrize("value", ["yes", "on", None, 1, "True"])
    def test_a_value_that_is_not_true_false_or_auto_is_refused_before_anything_is_written(
        self, tmp_path, value
    ):
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="True, False or 'auto'"):
            _refold(out, binder_cyclic=value)
        assert not out.exists()

    def test_an_unreadable_structure_does_not_fail_an_auto_run(self, tmp_path, caplog, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("cannot read")

        monkeypatch.setattr(_openfold_run, "_binder_is_head_to_tail", broken)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            chains = _chains(_refold(tmp_path))
        assert "cyclic" not in chains["I"]
        assert "cannot read" in caplog.text


class TestBatched:
    @staticmethod
    def _samples():
        return [
            openfold._BatchSample("sfti", SFTI1, "A", "I"),
            openfold._BatchSample("cyclo", CYCLOSPORIN, "A", "C"),
            openfold._BatchSample("p53", P53, "A", "B"),
        ]

    @pytest.mark.parametrize(
        "function",
        [openfold.prepare_batched_scoring_queries, openfold.prepare_batched_refolding_queries],
    )
    def test_each_sample_is_decided_on_its_own_binder(self, tmp_path, function):
        path = function(self._samples(), tmp_path)
        query = json.loads(path.read_text(encoding="utf-8"))["queries"]
        sfti = {c["chain_ids"][0]: c for c in query["sfti"]["chains"]}
        cyclo = {c["chain_ids"][0]: c for c in query["cyclo"]["chains"]}
        p53 = {c["chain_ids"][0]: c for c in query["p53"]["chains"]}
        assert sfti["I"]["cyclic"] is True and "cyclic" not in sfti["A"]
        assert all("cyclic" not in chain for chain in cyclo.values())  # modified residues
        assert all("cyclic" not in chain for chain in p53.values())

    @pytest.mark.parametrize(
        "function",
        [openfold.prepare_batched_scoring_queries, openfold.prepare_batched_refolding_queries],
    )
    def test_true_flags_every_binder_and_false_none(self, tmp_path, function):
        on = json.loads(
            function(self._samples(), tmp_path / "on", binder_cyclic=True).read_text(
                encoding="utf-8"
            )
        )["queries"]
        off = json.loads(
            function(self._samples(), tmp_path / "off", binder_cyclic=False).read_text(
                encoding="utf-8"
            )
        )["queries"]
        assert [c.get("cyclic") for c in on["p53"]["chains"]] == [None, True]
        assert all("cyclic" not in c for q in off.values() for c in q["chains"])

    def test_a_batch_asks_the_version_once(self, tmp_path, set_version):
        calls = set_version("0.5.0")
        samples = [openfold._BatchSample(f"s{i}", SFTI1, "A", "I") for i in range(3)]
        openfold.prepare_batched_refolding_queries(samples, tmp_path, conda_env="of3")
        assert len(calls) == 1

    def test_true_with_an_old_openfold3_writes_nothing(self, tmp_path, set_version):
        set_version("0.4.4")
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="0.4.5"):
            openfold.prepare_batched_scoring_queries(self._samples(), out, binder_cyclic=True)
        assert not out.exists()


class TestWrappers:
    """The run functions pass the choice and the environment on to the query builder."""

    @pytest.mark.parametrize("runner", ["run_openfold_scoring", "run_openfold_refolding"])
    def test_a_run_writes_the_flag_into_the_query_it_hands_over(
        self, tmp_path, monkeypatch, runner
    ):
        captured = {}

        def fake_run(query_json, **kwargs):
            captured["chains"] = _chains(Path(query_json))
            return Path(kwargs["output_dir"])

        monkeypatch.setattr(openfold, "run_openfold", fake_run)
        getattr(openfold, runner)(SFTI1, "A", "I", "q", tmp_path / "a")
        assert captured["chains"]["I"]["cyclic"] is True
        getattr(openfold, runner)(SFTI1, "A", "I", "q", tmp_path / "b", binder_cyclic=False)
        assert "cyclic" not in captured["chains"]["I"]

    def test_the_environment_of_the_run_is_the_one_asked_for_the_version(
        self, tmp_path, monkeypatch, set_version
    ):
        calls = set_version("0.4.4")
        monkeypatch.setattr(openfold, "run_openfold", lambda query_json, **kw: tmp_path)
        openfold.run_openfold_refolding(SFTI1, "A", "I", "q", tmp_path / "o", conda_env="of3")
        assert calls[0][1:] == ["run", "-n", "of3", "python"]

    def test_the_batched_run_passes_it_on(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run(query_json, **kwargs):
            captured["query"] = json.loads(Path(query_json).read_text(encoding="utf-8"))
            return Path(kwargs["output_dir"])

        monkeypatch.setattr(openfold, "run_openfold", fake_run)
        openfold.run_openfold_batched(
            [openfold._BatchSample("sfti", SFTI1, "A", "I")], tmp_path, mode="score"
        )
        assert captured["query"]["queries"]["sfti"]["chains"][1]["cyclic"] is True

    def test_a_forced_flag_on_an_old_openfold3_stops_before_the_process(
        self, tmp_path, monkeypatch, set_version
    ):
        set_version("0.4.4")
        monkeypatch.setattr(
            openfold, "run_openfold", lambda **kw: pytest.fail("OpenFold3 must not start")
        )
        with pytest.raises(ValueError, match="0.4.5"):
            openfold.run_openfold_scoring(
                CYCLOSPORIN, "A", "C", "q", tmp_path / "out", binder_cyclic=True
            )
        assert not (tmp_path / "out").exists()


class TestRequestKey:
    @staticmethod
    def _request(tmp_path, **kwargs):
        runner = OpenFold3Runner()
        return runner.make_request(
            CYCLOSPORIN, name="q", binder_chain="C", receptor_chain="A", **kwargs
        )

    def test_the_default_request_says_auto(self, tmp_path):
        assert self._request(tmp_path).options["binder_cyclic"] == "auto"

    def test_each_choice_is_another_key(self, tmp_path):
        keys = {self._request(tmp_path, binder_cyclic=v).key() for v in ("auto", True, False)}
        assert len(keys) == 3

    def test_the_default_key_is_the_key_of_auto(self, tmp_path):
        assert self._request(tmp_path).key() == self._request(tmp_path, binder_cyclic="auto").key()

    def test_the_query_arguments_leave_the_default_out(self, tmp_path):
        arguments = OpenFold3Runner._query_arguments
        assert "binder_cyclic" not in arguments(self._request(tmp_path))
        assert arguments(self._request(tmp_path, binder_cyclic=True))["binder_cyclic"] is True
        assert arguments(self._request(tmp_path, binder_cyclic=False))["binder_cyclic"] is False

    def test_a_query_file_is_not_given_a_cyclic_choice(self, tmp_path):
        query = tmp_path / "query.json"
        query.write_text("{}", encoding="utf-8")
        runner = OpenFold3Runner()
        default = runner.make_request(query, name="q", mode="predict")
        assert default.options["binder_cyclic"] is None
        with pytest.raises(ValueError, match="names its own chains"):
            runner.make_request(query, name="q", mode="predict", binder_cyclic=True)

    def test_a_value_that_is_no_choice_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="True, False or 'auto'"):
            self._request(tmp_path, binder_cyclic="yes")

    def test_the_runner_prepares_with_its_environment_and_the_choice(self, tmp_path, monkeypatch):
        seen = {}

        def fake_prepare(**kwargs):
            seen.update(kwargs)
            return Path(kwargs["output_dir"]) / "q_query.json"

        monkeypatch.setattr(openfold, "prepare_scoring_query", fake_prepare)
        runner = OpenFold3Runner(conda_env="of3")
        runner.prepare(self._request(tmp_path, binder_cyclic=False), tmp_path / "w")
        assert seen["binder_cyclic"] is False and seen["conda_env"] == "of3"

    def test_the_run_passes_the_choice_to_the_wrapper(self, tmp_path, monkeypatch):
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
        OpenFold3Runner().run(self._request(tmp_path, binder_cyclic=True), tmp_path / "w")
        assert seen["binder_cyclic"] is True


class TestCommandLine:
    @staticmethod
    def _argv(command, out, *extra):
        return [
            "prog", command, "--complex", str(SFTI1), "--receptor-chain", "A",
            "--binder-chain", "I", "--query-name", "q", "--output-dir", str(out), *extra,
        ]  # fmt: skip

    @pytest.mark.parametrize("command", ["prepare-query", "prepare-scoring-query"])
    @pytest.mark.parametrize("extra, written", [([], True), (["--openfold-cyclic", "off"], False)])
    def test_the_prepare_commands_take_the_choice(
        self, tmp_path, monkeypatch, command, extra, written
    ):
        monkeypatch.setattr("sys.argv", self._argv(command, tmp_path / "out", *extra))
        openfold.main()
        chains = _chains(tmp_path / "out" / "q_query.json")
        assert ("cyclic" in chains["I"]) is written

    def test_a_forced_flag_reaches_the_query_of_a_linear_binder(self, tmp_path, monkeypatch):
        argv = self._argv("prepare-query", tmp_path / "out", "--openfold-cyclic", "on")
        argv[argv.index(str(SFTI1))] = str(P53)
        argv[argv.index("I")] = "B"
        monkeypatch.setattr("sys.argv", argv)
        openfold.main()
        assert _chains(tmp_path / "out" / "q_query.json")["B"]["cyclic"] is True

    @pytest.mark.parametrize(
        "command, target",
        [("score", "run_openfold_scoring"), ("refold", "run_openfold_refolding")],
    )
    @pytest.mark.parametrize(
        "extra, expected",
        [([], None), (["--openfold-cyclic", "on"], True), (["--openfold-cyclic", "off"], False)],
    )
    def test_the_run_commands_pass_the_choice(
        self, tmp_path, monkeypatch, command, target, extra, expected
    ):
        seen = {}
        monkeypatch.setattr(openfold, target, lambda **kw: seen.update(kw) or tmp_path)
        monkeypatch.setattr(openfold, "compute_openfold_metrics", lambda **kw: {})
        monkeypatch.setattr(_openfold_cli, "_print_metrics", lambda *a, **kw: None)
        monkeypatch.setattr("sys.argv", self._argv(command, tmp_path, *extra))
        openfold.main()
        assert seen.get("binder_cyclic") is expected  # left out for the default, auto

    def test_the_help_states_what_the_flag_does_and_does_not(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "refold", "--help"])
        with pytest.raises(SystemExit):
            openfold.main()
        text = " ".join(capsys.readouterr().out.split())
        assert "--openfold-cyclic {auto,on,off}" in text
        for stated in ("head-to-tail", "does not enforce the closure bond", "example query"):
            assert stated in text
        # the rule and what it rests on, stated as one complex each
        for stated in (
            "standard residues only",
            "modified residue its own token",
            "one complex",
            "0.91-0.92 to 0.78-0.81",
            "0.5-0.7 A to 3.0-4.8 A",
            "7.40 A without it, 1.38 A with it",
        ):
            assert stated in text
        assert "no accuracy benchmark for cyclic peptides" in text.replace("published no", "no")

    def test_a_bad_choice_is_a_usage_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv", self._argv("refold", tmp_path, "--openfold-cyclic", "maybe")
        )
        with pytest.raises(SystemExit) as info:
            openfold.main()
        assert info.value.code == 2
        assert "invalid choice" in capsys.readouterr().err


def test_the_default_query_builders_raise_no_deprecation_warning(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _refold(tmp_path / "refold")
        _score(tmp_path / "score")
