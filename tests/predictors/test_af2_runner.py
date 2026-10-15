"""ColabFoldRunner: the request, the FASTA, the command line and the run through the store.

No ColabFold, GPU or network is used. The run tests put a stub ``colabfold_batch`` on PATH that
records its arguments and copies synthetic output (``synth_af2``) into the results directory;
what that stub cannot show is stated in ``predictors/af2_runner.py``. The tests that re-read the
ColabFold source need ``BINDING_METRICS_MODEL_SOURCES`` to point at a folder with a ``ColabFold``
clone and are skipped otherwise.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from binding_metrics.predictors import af2_runner
from binding_metrics.predictors.af2 import AlphaFold2Parser
from binding_metrics.predictors.af2_runner import ColabFoldRunError, ColabFoldRunner
from binding_metrics.predictors.registry import get_parser
from binding_metrics.predictors.runners import PredictionRunner
from binding_metrics.predictors.session import PredictionSession
from binding_metrics.predictors.store import (
    PredictionFailedError,
    PredictionRequest,
    PredictionStore,
    PredictionUnavailableError,
)
from binding_metrics.predictors.weights import WeightsRef

DATA = Path(__file__).resolve().parents[2] / "data"
P53 = DATA / "example_linear_p53_1YCR.pdb"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"
PHOSPHO = DATA / "example_phospho_1QJB.pdb"

#: The two chains of 1YCR (MDM2 and the p53 peptide), as the query builder reads them.
MDM2 = "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
P53_PEPTIDE = "ETFSDLWKLLPEN"

_PROBE_OF_THE_RUNNER = ColabFoldRunner._probe_version


@pytest.fixture(autouse=True)
def colabfold_version(monkeypatch):
    """A ColabFold 1.6.3 on every machine, whatever the machine has."""
    monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.6.3")


def p53_request(runner=None, **kwargs):
    runner = runner or ColabFoldRunner()
    kwargs.setdefault("binder_chain", "B")
    kwargs.setdefault("receptor_chain", "A")
    kwargs.setdefault("name", "p53")
    return runner.make_request(kwargs.pop("input_path", P53), **kwargs)


def pdb_text(chains):
    """A PDB file with N, CA and C per residue; ``chains`` maps a chain ID to residue names."""
    lines = []
    serial = 0
    for chain, residues in chains.items():
        for number, residue in enumerate(residues, start=1):
            for index, atom in enumerate(("N", "CA", "C")):
                serial += 1
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} {residue:>3s} {chain}{number:4d}    "
                    f"{3.8 * number + index:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}"
                    f"           {atom[0]}"
                )
    return "\n".join(lines) + "\nEND\n"


def write_pdb(path, chains, extra_lines=()):
    path.write_text(
        pdb_text(chains).replace("END\n", "") + "\n".join(extra_lines) + "\nEND\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------- the request


class TestMakeRequest:
    def test_every_default_is_written_out(self):
        request = p53_request()
        assert (request.model, request.name, request.mode) == ("af2", "p53", "predict")
        assert request.model_version == "1.6.3"
        assert request.seeds == (0,) and request.num_samples == 5
        assert (request.binder_chain, request.receptor_chain) == ("B", "A")
        assert request.options == {
            "use_msa_server": True,
            "model_type": "alphafold2_multimer_v3",
            "num_recycles": None,
            "num_relax": 0,
            "data_dir": None,
            "extra_args": [],
        }
        assert request.input_path is None and set(request.extra_files) == set()

    def test_the_request_holds_the_sequences_of_the_two_chains(self):
        assert dict(p53_request().sequences) == {"A": MDM2, "B": P53_PEPTIDE}

    def test_the_same_run_gives_the_same_key(self):
        assert p53_request().key() == p53_request().key()

    def test_the_name_labels_the_files_and_is_not_in_the_key(self):
        assert p53_request(name="other").key() == p53_request().key()

    def test_two_poses_of_the_same_sequences_share_one_key(self, tmp_path):
        """A prediction from sequences does not depend on the coordinates."""
        moved = tmp_path / "moved.pdb"
        moved.write_text(
            P53.read_text(encoding="utf-8").replace("ATOM      1", "REMARK x\nATOM      1", 1)
            + "REMARK another file\n",
            encoding="utf-8",
        )
        assert moved.read_bytes() != P53.read_bytes()
        assert p53_request(input_path=moved).key() == p53_request().key()

    def test_another_sequence_is_another_key(self, tmp_path):
        mutated = tmp_path / "mutated.pdb"
        lines = P53.read_text(encoding="utf-8").splitlines()
        mutated.write_text(
            "\n".join(
                line.replace(" GLU B", " ASP B", 1) if line.startswith("ATOM") else line
                for line in lines
            )
            + "\n",
            encoding="utf-8",
        )
        assert dict(p53_request(input_path=mutated).sequences)["B"][0] == "D"
        assert p53_request(input_path=mutated).key() != p53_request().key()

    @pytest.mark.parametrize(
        "change",
        [
            {"seeds": (1, 2)},
            {"num_samples": 3},
            {"num_recycles": 6},
            {"use_msa_server": False},
            {"model_type": "alphafold2_multimer_v2"},
            {"num_relax": 1},
            {"extra_args": ["--calc-extra-ptm"]},
            {"binder_chain": "A", "receptor_chain": "B"},
        ],
        ids=lambda change: next(iter(change)),
    )
    def test_a_setting_that_changes_the_output_changes_the_key(self, change):
        assert p53_request(**change).key() != p53_request().key()

    def test_the_weights_directory_is_in_the_key(self, tmp_path):
        (tmp_path / "weights").mkdir()
        request = p53_request(data_dir=tmp_path / "weights")
        assert request.options["data_dir"] == str(tmp_path / "weights")
        assert request.key() != p53_request().key()

    def test_another_colabfold_version_is_another_key(self, monkeypatch):
        before = p53_request().key()
        monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.5.5")
        assert p53_request().key() != before

    def test_an_unknown_version_is_an_empty_string(self, monkeypatch):
        monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: None)
        assert p53_request().model_version == ""

    def test_the_conda_environment_is_not_part_of_the_key(self):
        assert p53_request(ColabFoldRunner("colabfold")).key() == p53_request().key()

    def test_seeds_are_sorted_into_one_request(self):
        assert p53_request(seeds=(2, 0, 1)).seeds == (0, 1, 2)
        assert p53_request(seeds=(2, 0, 1)).key() == p53_request(seeds=(0, 1, 2)).key()

    @pytest.mark.parametrize("seeds", [(0, 2), (1, 1), (-1, 0), ()])
    def test_seeds_that_colabfold_cannot_iterate_are_refused(self, seeds):
        with pytest.raises(ValueError, match="consecutive integers from 0 up"):
            p53_request(seeds=seeds)

    def test_seeds_may_start_above_zero(self):
        assert p53_request(seeds=(40, 41)).seeds == (40, 41)

    @pytest.mark.parametrize("count", [0, 6])
    def test_the_number_of_models_is_one_to_five(self, count):
        with pytest.raises(ValueError, match="num_samples|1 to 5"):
            p53_request(num_samples=count)

    def test_an_unknown_model_type_is_refused(self):
        with pytest.raises(ValueError, match="model_type must be one of"):
            p53_request(model_type="auto")

    def test_a_name_that_colabfold_would_change_is_refused(self):
        with pytest.raises(ValueError, match="would write the job 'my sample' as 'my_sample'"):
            p53_request(name="my sample")
        assert p53_request(name="p53_v2.1-a").name == "p53_v2.1-a"

    def test_the_chain_roles_are_checked(self):
        with pytest.raises(ValueError, match="needs binder_chain and receptor_chain"):
            p53_request(binder_chain=None)
        with pytest.raises(ValueError, match="two different chains"):
            p53_request(binder_chain="A", receptor_chain="A")
        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            p53_request(binder_chain="Z")

    def test_the_weights_directory_must_exist(self, tmp_path):
        with pytest.raises(ValueError, match="is not a directory"):
            p53_request(data_dir=tmp_path / "no_such_directory")

    def test_a_residue_choice_other_than_error_is_refused(self):
        with pytest.raises(ValueError, match="on_unmappable_residue must be 'error'"):
            p53_request(on_unmappable_residue="x")
        assert p53_request(on_unmappable_residue="error").key() == p53_request().key()

    @pytest.mark.parametrize(
        "argument",
        ["--num-models", "--num-models=3", "--msa-mode", "--random-seed=1", "--data", "--amber"],
    )
    def test_a_flag_the_runner_writes_cannot_be_given_again(self, argument):
        with pytest.raises(ValueError, match="the runner writes it"):
            p53_request(extra_args=[argument, "3"])

    def test_an_abbreviated_flag_counts_because_argparse_accepts_it(self):
        with pytest.raises(ValueError, match="the runner writes it"):
            p53_request(extra_args=["--num-mod", "3"])

    @pytest.mark.parametrize("argument", ["--zip", "--jobname-prefix", "--msa-only"])
    def test_a_flag_that_moves_the_output_away_from_the_adapter_is_refused(self, argument):
        with pytest.raises(ValueError, match="changes the files the adapter reads"):
            p53_request(extra_args=[argument])

    @pytest.mark.parametrize(
        "argument", ["--templates", "--custom-template-path=/t", "--initial-guess"]
    )
    def test_a_flag_that_adds_templates_is_refused_in_predict_mode(self, argument):
        with pytest.raises(ValueError, match="folds from sequences only"):
            p53_request(extra_args=[argument])

    def test_other_flags_pass(self):
        request = p53_request(extra_args=["--calc-extra-ptm", "--num-ensemble", "2"])
        assert request.options["extra_args"] == ["--calc-extra-ptm", "--num-ensemble", "2"]


# ---------------------------------------------------------------------------- the FASTA


class TestTheFasta:
    def test_one_record_with_the_receptor_then_the_binder(self):
        request = p53_request()
        assert ColabFoldRunner.fasta_text(request) == f">p53\n{MDM2}:{P53_PEPTIDE}\n"

    def test_the_order_follows_the_roles_and_not_the_chain_ids(self):
        swapped = p53_request(binder_chain="A", receptor_chain="B")
        assert ColabFoldRunner.fasta_text(swapped) == f">p53\n{P53_PEPTIDE}:{MDM2}\n"

    def test_prepare_writes_the_record_below_the_work_directory(self, tmp_path):
        request = p53_request()
        path = ColabFoldRunner().prepare(request, tmp_path)
        assert path == tmp_path / "query" / "p53.fasta"
        assert path.read_text(encoding="utf-8") == f">p53\n{MDM2}:{P53_PEPTIDE}\n"
        assert [p.name for p in tmp_path.iterdir()] == ["query"]

    def test_the_chains_of_the_prediction_are_a_and_b_in_the_order_of_the_record(self, tmp_path):
        """The receptor is chain A and the binder B, whatever their IDs in the input."""
        path = write_pdb(tmp_path / "rp.pdb", {"R": ["ALA", "GLY", "SER"], "P": ["TRP", "PHE"]})
        request = ColabFoldRunner().make_request(
            path, name="rp", binder_chain="P", receptor_chain="R"
        )
        assert ColabFoldRunner.fasta_text(request) == ">rp\nAGS:WF\n"
        assert ColabFoldRunner.output_chain_map(request) == {"A": "R", "B": "P"}

    def test_the_chain_map_of_1ycr_is_the_identity(self):
        assert ColabFoldRunner.output_chain_map(p53_request()) == {"A": "A", "B": "B"}

    def test_protonation_variants_take_their_parent_and_unknown_residues_x(self, tmp_path):
        path = write_pdb(
            tmp_path / "variants.pdb",
            {"A": ["ALA", "HID", "CYX", "UNK", "GLY"], "B": ["HIE", "HIP"]},
        )
        request = ColabFoldRunner().make_request(
            path, name="v", binder_chain="B", receptor_chain="A"
        )
        assert dict(request.sequences) == {"A": "AHCXG", "B": "HH"}

    def test_waters_and_ligands_of_a_chain_are_left_out(self, tmp_path):
        path = write_pdb(
            tmp_path / "wet.pdb",
            {"A": ["ALA", "GLY"], "B": ["SER", "THR"]},
            extra_lines=[
                "HETATM  900  O   HOH A 901       9.000   9.000   9.000  1.00  0.00           O",
                "HETATM  901  C1  XYZ B 902       9.000   9.000   9.000  1.00  0.00           C",
            ],
        )
        request = ColabFoldRunner().make_request(
            path, name="w", binder_chain="B", receptor_chain="A"
        )
        assert dict(request.sequences) == {"A": "AG", "B": "ST"}


# ---------------------------------------------------------------------------- residues


class TestNonStandardResidues:
    def test_a_cyclosporin_binder_is_refused_naming_the_chain_and_the_residues(self):
        with pytest.raises(ValueError) as error:
            p53_request(input_path=CYCLOSPORIN, binder_chain="C", receptor_chain="A")
        text = str(error.value)
        assert "AlphaFold2 / ColabFold cannot take these residues" in text
        assert "chain 'C': DAL 1, MLE 2, MLE 3, MVA 4, BMT 5, ABA 6, SAR 7, MLE 8, MLE 10" in text
        assert "chain 'A'" not in text

    def test_a_phosphorylated_residue_is_refused(self):
        with pytest.raises(ValueError, match="chain 'Q': SEP 7"):
            p53_request(input_path=PHOSPHO, binder_chain="Q", receptor_chain="Z")

    def test_both_chains_are_named_when_both_have_residues_it_cannot_take(self, tmp_path):
        path = write_pdb(tmp_path / "both.pdb", {"A": ["ALA", "MSE"], "B": ["SEC", "DAL"]})
        with pytest.raises(ValueError) as error:
            ColabFoldRunner().make_request(path, name="b", binder_chain="B", receptor_chain="A")
        assert "chain 'A': MSE 2" in str(error.value)
        assert "chain 'B': SEC 1, DAL 2" in str(error.value)

    def test_a_residue_with_no_letter_is_named_too(self, tmp_path):
        path = write_pdb(tmp_path / "odd.pdb", {"A": ["ALA", "ZZZ"], "B": ["GLY"]})
        with pytest.raises(ValueError, match="chain 'A': ZZZ 2"):
            ColabFoldRunner().make_request(path, name="o", binder_chain="B", receptor_chain="A")

    def test_a_request_built_by_hand_is_checked_as_well(self, tmp_path):
        request = PredictionRequest(
            "af2",
            "q",
            mode="predict",
            binder_chain="B",
            receptor_chain="A",
            sequences={"A": "ACDE", "B": "AUG"},
        )
        with pytest.raises(ValueError, match=r"chain 'B' must be written with the 20 amino acid"):
            ColabFoldRunner().prepare(request, tmp_path)
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------- modes


class TestModes:
    @pytest.mark.parametrize("mode", ["refold", "score"])
    def test_a_templated_mode_is_refused_with_the_route_that_exists(self, mode):
        with pytest.raises(ValueError) as error:
            p53_request(mode=mode)
        text = str(error.value)
        assert f"runs mode 'predict' only, not '{mode}'" in text
        assert "--custom-template-path" in text and "one directory for the whole job" in text
        assert "--prediction-dir" in text

    def test_score_lock_is_refused_because_nothing_fixes_the_pose(self):
        with pytest.raises(ValueError, match="no input of ColabFold fixes the relative pose"):
            p53_request(mode="score-lock")

    @pytest.mark.parametrize("mode", ["refold", "score", "score-lock"])
    def test_a_request_built_by_hand_is_refused_by_prepare_and_run(self, tmp_path, mode):
        request = PredictionRequest(
            "af2",
            "q",
            mode=mode,
            binder_chain="B",
            receptor_chain="A",
            sequences={"A": "ACDE", "B": "AGK"},
        )
        runner = ColabFoldRunner()
        with pytest.raises(ValueError, match=f"not '{mode}'"):
            runner.prepare(request, tmp_path)
        with pytest.raises(ValueError, match=f"not '{mode}'"):
            runner.run(request, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_predict_is_the_default(self):
        assert p53_request().mode == "predict"


# ---------------------------------------------------------------------------- command line


class TestCommandLine:
    @staticmethod
    def command(request=None, runner=None):
        runner = runner or ColabFoldRunner()
        return runner._command(request or p53_request(), Path("/w/query/p53.fasta"), Path("/w/out"))

    def test_the_defaults_are_all_written_out(self):
        assert self.command() == [
            "colabfold_batch",
            "/w/query/p53.fasta",
            "/w/out",
            "--msa-mode",
            "mmseqs2_uniref_env",
            "--model-type",
            "alphafold2_multimer_v3",
            "--num-models",
            "5",
            "--random-seed",
            "0",
            "--num-seeds",
            "1",
        ]

    @pytest.mark.parametrize(
        "kwargs, flags",
        [
            ({"use_msa_server": False}, ["--msa-mode", "single_sequence"]),
            ({"seeds": (3, 4, 5)}, ["--random-seed", "3", "--num-seeds", "3"]),
            ({"num_samples": 2}, ["--num-models", "2"]),
            ({"model_type": "alphafold2_multimer_v2"}, ["--model-type", "alphafold2_multimer_v2"]),
            ({"num_recycles": 6}, ["--num-recycle", "6"]),
            ({"num_recycles": 0}, ["--num-recycle", "0"]),
            ({"num_relax": 2}, ["--num-relax", "2"]),
        ],
        ids=lambda value: next(iter(value)) if isinstance(value, dict) else None,
    )
    def test_each_option_reaches_its_flag(self, kwargs, flags):
        command = self.command(p53_request(**kwargs))
        for flag, value in zip(flags[::2], flags[1::2]):
            assert command[command.index(flag) + 1] == value

    def test_the_weights_directory_is_passed_as_data(self, tmp_path):
        (tmp_path / "w").mkdir()
        command = self.command(p53_request(data_dir=tmp_path / "w"))
        assert command[command.index("--data") + 1] == str(tmp_path / "w")

    def test_nothing_that_was_not_asked_for_is_written(self):
        command = self.command()
        for flag in (
            "--templates",
            "--custom-template-path",
            "--amber",
            "--num-relax",
            "--num-recycle",
            "--data",
            "--zip",
            "--save-all",
            "--initial-guess",
            "--calc-extra-ptm",
        ):
            assert flag not in command

    def test_extra_arguments_come_last_and_unchanged(self):
        command = self.command(p53_request(extra_args=["--calc-extra-ptm", "--num-ensemble", "2"]))
        assert command[-3:] == ["--calc-extra-ptm", "--num-ensemble", "2"]

    def test_the_positionals_come_first_so_a_flag_with_an_optional_value_cannot_take_them(self):
        command = self.command(p53_request(num_relax=1, extra_args=["--calc-extra-ptm"]))
        assert command[1:3] == ["/w/query/p53.fasta", "/w/out"]

    def test_a_conda_environment_runs_the_command_through_conda_without_capture(self):
        command = self.command(runner=ColabFoldRunner("colabfold"))
        assert command[1:6] == ["run", "-n", "colabfold", "--no-capture-output", "colabfold_batch"]
        assert command[6:] == self.command()[1:]


# ---------------------------------------------------------------------------- the machine


def install_script(directory, name, text):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


class TestTheMachine:
    def test_the_runner_is_the_af2_model_and_declares_no_limit(self):
        assert ColabFoldRunner.name == "af2" == AlphaFold2Parser.name
        assert get_parser(ColabFoldRunner.name) is not None
        assert ColabFoldRunner.capabilities is None and issubclass(
            ColabFoldRunner, PredictionRunner
        )

    def test_a_request_is_not_batched(self):
        runner = ColabFoldRunner()
        assert runner.supports_batch(p53_request()) is False
        with pytest.raises(NotImplementedError, match="no batched mode"):
            runner.run_many([], Path("."))

    def test_available_when_the_executable_is_on_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        assert ColabFoldRunner().is_available() is False
        install_script(tmp_path / "bin", "colabfold_batch", "#!/bin/sh\nexit 0\n")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        assert ColabFoldRunner().is_available() is True

    def test_a_conda_environment_is_available_when_it_has_colabfold(self, monkeypatch):
        monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: None)
        assert ColabFoldRunner("cf").is_available() is False
        monkeypatch.setattr(ColabFoldRunner, "_probe_version", lambda self: "1.6.3")
        assert ColabFoldRunner("cf").is_available() is True

    def test_the_version_is_asked_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ColabFoldRunner, "_probe_version", lambda self: calls.append(1) or "1.6.3"
        )
        runner = ColabFoldRunner()
        assert (runner.version(), runner.version()) == ("1.6.3", "1.6.3") and len(calls) == 1

    def test_the_version_comes_from_the_interpreter_of_the_script_on_path(
        self, tmp_path, monkeypatch
    ):
        """LocalColabFold is not the environment that runs this process."""
        interpreter = install_script(tmp_path / "env" / "bin", "python", "#!/bin/sh\necho 1.5.5\n")
        install_script(tmp_path / "bin", "colabfold_batch", f"#!{interpreter}\nprint('x')\n")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        assert _PROBE_OF_THE_RUNNER(ColabFoldRunner()) == "1.5.5"

    def test_a_script_with_a_long_interpreter_path_names_it_on_its_second_line(
        self, tmp_path, monkeypatch
    ):
        interpreter = install_script(
            tmp_path / "long env" / "bin", "python3", "#!/bin/sh\necho 1.5.4\n"
        )
        script = f"#!/bin/sh\n'''exec' \"{interpreter}\" \"$0\" \"$@\"\n' '''\nimport sys\n"
        install_script(tmp_path / "bin", "colabfold_batch", script)
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        assert _PROBE_OF_THE_RUNNER(ColabFoldRunner()) == "1.5.4"

    def test_script_interpreters(self, tmp_path, monkeypatch):
        python = install_script(tmp_path / "bin", "python3.11", "#!/bin/sh\n")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        read = af2_runner._script_interpreter
        assert read(install_script(tmp_path, "a", f"#!{python}\n")) == [str(python)]
        assert read(install_script(tmp_path, "b", "#!/usr/bin/env python3.11\n")) == [str(python)]
        assert read(install_script(tmp_path, "c", "#!/usr/bin/env -S python3.11\n")) == [
            str(python)
        ]
        assert read(install_script(tmp_path, "d", "#!/bin/sh\necho hi\n")) is None
        assert read(install_script(tmp_path, "e", "print('no shebang')\n")) is None
        assert read(tmp_path / "missing") is None

    def test_a_conda_environment_is_asked_through_conda(self, tmp_path, monkeypatch):
        install_script(
            tmp_path / "bin",
            "conda",
            '#!/bin/sh\n[ "$1 $2 $3 $4" = "run -n cf python" ] && echo 1.6.3 && exit 0\nexit 3\n',
        )
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
        assert _PROBE_OF_THE_RUNNER(ColabFoldRunner("cf")) == "1.6.3"
        assert _PROBE_OF_THE_RUNNER(ColabFoldRunner("other")) is None

    def test_a_missing_script_and_package_give_no_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(
            af2_runner.importlib.metadata,
            "version",
            lambda name: (_ for _ in ()).throw(
                af2_runner.importlib.metadata.PackageNotFoundError(name)
            ),
        )
        assert _PROBE_OF_THE_RUNNER(ColabFoldRunner()) is None

    def test_the_package_of_this_interpreter_is_the_last_resort(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(af2_runner.importlib.metadata, "version", lambda name: "1.2.3")
        assert _PROBE_OF_THE_RUNNER(ColabFoldRunner()) == "1.2.3"


# ---------------------------------------------------------------------------- a stub executable


STUB_COLABFOLD = """#!{python}
import json, pathlib, shutil, sys

argv = sys.argv[1:]
here = pathlib.Path(__file__).resolve().parent
behaviour = json.loads((here / "behaviour.json").read_text(encoding="utf-8"))
fasta, results = pathlib.Path(argv[0]), pathlib.Path(argv[1])
with open(here / "calls.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"argv": argv, "fasta": fasta.read_text(encoding="utf-8")}}) + "\\n")
if behaviour["mode"] == "fail":
    sys.stdout.write("2026-10-01 10:00:00,000 Running colabfold 1.6.3\\n")
    sys.stdout.flush()
    sys.stderr.write("Traceback (most recent call last):\\n")
    sys.stderr.write("RuntimeError: Error downloading files\\n")
    sys.exit(1)
results.mkdir(parents=True, exist_ok=True)
(results / "config.json").write_text("{{}}", encoding="utf-8")
if behaviour.get("log"):
    (results / "log.txt").write_text(behaviour["log"], encoding="utf-8")
if behaviour["mode"] == "ok":
    for source in pathlib.Path(behaviour["source"]).iterdir():
        shutil.copy2(source, results / source.name)
"""

STUB_CONDA = """#!{python}
import json, os, sys

argv = sys.argv[1:]
with open({log!r}, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\\n")
assert argv[:2] == ["run", "-n"], argv
rest = argv[3:]
if rest[0] == "--no-capture-output":
    rest = rest[1:]
if rest[0] == "python":
    print("1.6.3")
    sys.exit(0)
os.execvp(rest[0], rest)
"""


@pytest.fixture
def stub_colabfold(tmp_path, monkeypatch):
    """A ``colabfold_batch`` on PATH that copies synthetic output; the stub's controls."""
    from tests.predictors import synth, synth_af2

    bin_dir = tmp_path / "bin"
    install_script(bin_dir, "colabfold_batch", STUB_COLABFOLD.format(python=sys.executable))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    class Stub:
        directory = bin_dir

        @staticmethod
        def set(mode="ok", log="", name="p53", seeds=1, samples=5, relaxed=False):
            source = tmp_path / "synthetic"
            if source.exists():
                for old in source.iterdir():
                    old.unlink()
            if mode == "ok":
                complex_ = synth.synthetic_complex()
                for seed_index in range(1, seeds + 1):
                    for sample in range(1, samples + 1):
                        synth_af2.write_colabfold(
                            source,
                            name,
                            complex_,
                            seed_index=seed_index,
                            sample=sample,
                            relaxed=relaxed,
                        )
            (bin_dir / "behaviour.json").write_text(
                json.dumps({"mode": mode, "log": log, "source": str(source)}), encoding="utf-8"
            )

        @staticmethod
        def calls():
            record = bin_dir / "calls.jsonl"
            if not record.exists():
                return []
            return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]

    Stub.set()
    return Stub


class TestWithAStubExecutable:
    def test_a_run_is_stored_and_the_adapter_reads_it(self, tmp_path, stub_colabfold):
        stub_colabfold.set(seeds=2, samples=2)
        store = PredictionStore(tmp_path / "store")
        runner = ColabFoldRunner()
        request = p53_request(runner, seeds=(1, 2), num_samples=2, use_msa_server=False)
        entry = store.get_or_run(request, runner)
        store.get_or_run(request, runner)  # a second metric: no second process
        (call,) = stub_colabfold.calls()
        argv = call["argv"]
        assert argv[0].endswith("/outputs/query/p53.fasta")
        assert argv[1].endswith("/outputs/predictions")
        assert argv[argv.index("--msa-mode") + 1] == "single_sequence"
        assert argv[argv.index("--num-models") + 1] == "2"
        assert argv[argv.index("--random-seed") + 1] == "1"
        assert argv[argv.index("--num-seeds") + 1] == "2"
        assert call["fasta"] == f">p53\n{MDM2}:{P53_PEPTIDE}\n"
        parser = AlphaFold2Parser()
        assert len(parser.list_samples(entry.prediction_dir, "p53")) == 4
        record = parser.load(entry.prediction_dir, "p53")
        assert (record.iptm, record.ptm) == (0.76, 0.88)
        assert record.chain_ptm == {"A": 0.88, "B": 0.8}

    def test_the_stored_directory_is_the_one_the_runner_returned(self, tmp_path, stub_colabfold):
        store = PredictionStore(tmp_path / "store")
        entry = store.get_or_run(p53_request(), ColabFoldRunner())
        assert entry.prediction_dir == entry.directory / "outputs" / "predictions"
        assert (entry.prediction_dir / "config.json").is_file()
        assert (entry.directory / "outputs" / "query" / "p53.fasta").is_file()

    def test_a_relaxed_run_is_read_as_relaxed(self, tmp_path, stub_colabfold):
        stub_colabfold.set(relaxed=True, samples=1)
        runner = ColabFoldRunner()
        request = p53_request(runner, num_relax=1, num_samples=1)
        entry = PredictionStore(tmp_path / "store").get_or_run(request, runner)
        names = sorted(p.name for p in entry.prediction_dir.iterdir())
        assert any("_relaxed_rank_001" in name for name in names)
        assert AlphaFold2Parser().load(entry.prediction_dir, "p53").iptm == 0.76
        assert "--num-relax" in stub_colabfold.calls()[0]["argv"]

    def test_the_session_runs_the_model_once_for_two_metrics_and_two_poses(
        self, tmp_path, stub_colabfold
    ):
        """The same sequences in another pose, under another name, share the run."""
        other = tmp_path / "other_pose.pdb"
        other.write_text(
            P53.read_text(encoding="utf-8") + "REMARK another pose\n", encoding="utf-8"
        )
        runner = ColabFoldRunner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        request = p53_request(runner)
        chain_map = ColabFoldRunner.output_chain_map(request)
        assert session.record(request, chain_map=chain_map).iptm == 0.76  # metric one
        assert session.record(request, chain_map=chain_map).avg_plddt > 0  # metric two
        again = p53_request(runner, input_path=other, name="pose2")
        assert session.record(again, chain_map=chain_map).iptm == 0.76
        assert len(stub_colabfold.calls()) == 1
        assert session.stats()["runs"] == 1

    def test_a_second_session_finds_the_run(self, tmp_path, stub_colabfold):
        runner = ColabFoldRunner()
        store = PredictionStore(tmp_path / "store")
        request = p53_request(runner)
        PredictionSession(store, [runner]).record(request)
        again = PredictionSession(store, [runner])
        assert again.record(request).iptm == 0.76
        assert len(stub_colabfold.calls()) == 1 and again.stats()["hits"] == 1

    def test_nothing_runs_and_nothing_is_written_for_a_residue_it_cannot_take(
        self, tmp_path, stub_colabfold
    ):
        runner = ColabFoldRunner()
        with pytest.raises(ValueError, match="chain 'C': DAL 1"):
            p53_request(runner, input_path=CYCLOSPORIN, binder_chain="C", receptor_chain="A")
        assert stub_colabfold.calls() == []
        assert not (tmp_path / "store").exists()

    def test_nothing_runs_for_a_request_built_by_hand_with_a_residue_it_cannot_take(
        self, tmp_path, stub_colabfold
    ):
        request = PredictionRequest(
            "af2",
            "hand",
            mode="predict",
            binder_chain="B",
            receptor_chain="A",
            sequences={"A": "ACDE", "B": "AUG"},
        )
        store = PredictionStore(tmp_path / "store")
        entry = store.ensure(request, ColabFoldRunner())
        assert entry.status == "failed"
        assert "must be written with the 20 amino acid" in entry.reason
        assert stub_colabfold.calls() == []

    def test_a_mode_it_cannot_run_never_starts_the_process(self, tmp_path, stub_colabfold):
        request = PredictionRequest(
            "af2",
            "hand",
            mode="score",
            binder_chain="B",
            receptor_chain="A",
            sequences={"A": "ACDE", "B": "AGK"},
        )
        entry = PredictionStore(tmp_path / "store").ensure(request, ColabFoldRunner())
        assert entry.status == "failed" and "runs mode 'predict' only" in entry.reason
        assert stub_colabfold.calls() == []

    def test_a_process_that_fails_is_recorded_with_its_reason_and_not_run_again(
        self, tmp_path, stub_colabfold
    ):
        stub_colabfold.set(mode="fail")
        store = PredictionStore(tmp_path / "store")
        runner = ColabFoldRunner()
        request = p53_request(runner)
        with pytest.raises(PredictionFailedError, match="Error downloading files") as info:
            store.get_or_run(request, runner)
        assert isinstance(info.value.__cause__, ColabFoldRunError)
        assert "colabfold_batch exited with status 1: RuntimeError: Error downloading files" in (
            info.value.reason
        )
        assert "params/" in info.value.reason  # the hint
        assert "Last lines of output:" in info.value.reason
        with pytest.raises(PredictionFailedError):
            store.get_or_run(request, runner)
        assert len(stub_colabfold.calls()) == 1
        stub_colabfold.set()
        assert store.get_or_run(request, runner, rerun=True).status == "done"
        assert len(stub_colabfold.calls()) == 2

    def test_a_job_that_colabfold_skipped_with_status_zero_is_a_failure_with_its_reason(
        self, tmp_path, stub_colabfold
    ):
        log = (
            "2026-10-01 10:00:00,000 Running colabfold 1.6.3\n"
            "2026-10-01 10:00:05,123 Could not get MSA/templates for p53: ConnectionError\n"
            "Traceback (most recent call last):\n"
        )
        stub_colabfold.set(mode="no_output", log=log)
        store = PredictionStore(tmp_path / "store")
        runner = ColabFoldRunner()
        with pytest.raises(PredictionFailedError) as info:
            store.get_or_run(p53_request(runner), runner)
        reason = info.value.reason
        assert "ColabFold wrote 0 of 5 structures for job 'p53'" in reason
        assert "It logged: Could not get MSA/templates for p53: ConnectionError" in reason
        assert "predictions/log.txt" in reason and "/tmp" not in reason
        assert "2026-10-01" not in reason.split("\n")[0]  # the timestamp is dropped
        assert "use_msa_server=False" in reason  # the hint

    def test_a_job_with_fewer_structures_than_asked_is_a_failure(self, tmp_path, stub_colabfold):
        stub_colabfold.set(samples=1)  # one structure written
        runner = ColabFoldRunner()
        request = p53_request(runner, num_samples=2)
        with pytest.raises(PredictionFailedError, match="wrote 1 of 2 structures"):
            PredictionStore(tmp_path / "store").get_or_run(request, runner)

    def test_a_colabfold_that_is_not_installed_records_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        store = PredictionStore(tmp_path / "store")
        runner = ColabFoldRunner()
        request = p53_request(runner)
        with pytest.raises(PredictionUnavailableError, match="cannot be started"):
            store.get_or_run(request, runner)
        assert store.lookup(request) is None and not store.path_for(request).exists()

    def test_run_without_the_executable_says_what_to_do(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError, match="bin directory"):
            ColabFoldRunner().run(p53_request(), tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_conda_environment_runs_the_stub_through_conda(
        self, tmp_path, stub_colabfold, monkeypatch
    ):
        log = tmp_path / "conda_calls.jsonl"
        install_script(
            stub_colabfold.directory,
            "conda",
            STUB_CONDA.format(python=sys.executable, log=str(log)),
        )
        monkeypatch.setattr(ColabFoldRunner, "_probe_version", _PROBE_OF_THE_RUNNER)
        runner = ColabFoldRunner("cf")
        request = p53_request(runner)
        assert request.model_version == "1.6.3" and runner.is_available()
        entry = PredictionStore(tmp_path / "store").get_or_run(request, runner)
        assert entry.status == "done"
        conda_calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert conda_calls[0][:4] == ["run", "-n", "cf", "python"]  # the version probe
        assert conda_calls[-1][:4] == ["run", "-n", "cf", "--no-capture-output"]
        assert conda_calls[-1][4] == "colabfold_batch"
        assert len(stub_colabfold.calls()) == 1

    def test_the_output_of_the_process_is_shown_on_stderr(self, tmp_path, stub_colabfold, capfd):
        stub_colabfold.set(mode="fail")
        runner = ColabFoldRunner()
        with pytest.raises(ColabFoldRunError):
            runner.run(p53_request(runner), tmp_path)
        assert "Error downloading files" in capfd.readouterr().err


class TestTheRunError:
    def test_the_message_names_the_status_the_exception_line_and_the_last_lines(self):
        tail = "\n".join(f"line {i}" for i in range(20)) + "\nValueError: bad input\nline last"
        error = ColabFoldRunError(2, ["colabfold_batch"], tail)
        text = str(error)
        assert text.startswith("colabfold_batch exited with status 2: ValueError: bad input")
        assert "line last" in text and "line 0" not in text
        assert error.returncode == 2 and error.output == tail

    def test_an_empty_output_still_gives_a_message(self):
        assert (
            str(ColabFoldRunError(1, ["colabfold_batch"])) == "colabfold_batch exited with status 1"
        )

    def test_progress_bars_are_split_into_lines(self):
        error = ColabFoldRunError(
            1, ["x"], "10%|#\r50%|###\r100%|######\nRuntimeError: out of memory"
        )
        assert "RuntimeError: out of memory" in str(error)


# ---------------------------------------------------------------------------- the source


def colabfold_source(relative_path):
    """The text of a file of a ColabFold clone, or skip when the clone is not there."""
    root = os.environ.get("BINDING_METRICS_MODEL_SOURCES")
    if not root:
        pytest.skip("BINDING_METRICS_MODEL_SOURCES is not set")
    path = Path(root) / "ColabFold" / relative_path
    if not path.is_file():
        pytest.skip(f"{path} is not there")
    return path.read_text(encoding="utf-8")


def argparse_flags(text):
    """``{flag: {keyword: literal value}}`` of every ``add_argument`` call in ``text``."""
    flags = {}
    for node in ast.walk(ast.parse(text)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        names = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--")
        ]
        keywords = {}
        for keyword in node.keywords:
            try:
                keywords[keyword.arg] = ast.literal_eval(keyword.value)
            except ValueError:
                pass  # a name, a call: not a literal
        for name in names:
            flags[name] = keywords
    return flags


@pytest.fixture(scope="module")
def flags():
    return argparse_flags(colabfold_source("colabfold/batch.py"))


class TestAgainstTheColabFoldSource:
    """What the runner writes, re-read in ``colabfold/batch.py`` (v1.6.3) when a clone is there."""

    def test_the_flags_the_runner_writes_exist_with_the_defaults_it_writes_out(self, flags):
        assert flags["--msa-mode"]["default"] == af2_runner._MSA_SERVER_MODE
        assert af2_runner._NO_MSA_MODE in flags["--msa-mode"]["choices"]
        assert flags["--num-models"]["default"] == af2_runner._DEFAULT_NUM_SAMPLES
        assert flags["--num-models"]["choices"] == list(range(1, af2_runner._MAX_NUM_SAMPLES + 1))
        assert flags["--random-seed"]["default"] == af2_runner._DEFAULT_SEED
        assert flags["--num-seeds"]["default"] == 1
        assert flags["--num-recycle"]["default"] is None
        assert flags["--num-relax"]["default"] == 0
        assert "--data" in flags

    def test_the_model_types_are_the_choices_of_the_flag(self, flags):
        choices = set(flags["--model-type"]["choices"])
        assert flags["--model-type"]["default"] == "auto"
        assert choices - {"auto"} == set(af2_runner._MODEL_TYPES)
        assert af2_runner._DEFAULT_MODEL_TYPE in choices

    def test_every_flag_the_runner_refuses_in_extra_args_exists(self, flags):
        refused = af2_runner._OWNED_FLAGS + af2_runner._LAYOUT_FLAGS + af2_runner._TEMPLATE_FLAGS
        assert [flag for flag in refused if flag not in flags] == []

    def test_templates_are_off_unless_asked_for(self, flags):
        assert flags["--templates"]["default"] is False
        assert flags["--templates"]["action"] == "store_true"
        assert flags["--zip"]["default"] is False

    def test_seeds_are_a_range_from_the_random_seed(self):
        assert "enumerate(range(random_seed, random_seed+num_seeds))" in colabfold_source(
            "colabfold/batch.py"
        )

    def test_the_job_name_is_made_safe_and_chains_are_lettered_in_order(self):
        text = colabfold_source("colabfold/batch.py")
        assert "jobname = safe_filename(raw_jobname)" in text
        assert "features_for_chain[protein.PDB_CHAIN_IDS[chain_cnt]] = feature_dict" in text
        rule = colabfold_source("colabfold/input.py")
        assert 'c if c.isalnum() or c in ["_", ".", "-"] else "_"' in rule

    def test_the_templates_of_a_job_are_one_directory_for_every_distinct_sequence(self):
        text = colabfold_source("colabfold/batch.py")
        assert "template_paths[index] = custom_template_path" in text
        assert "def mk_hhsearch_db(" in text and "def mk_template(" in text

    def test_relaxation_is_set_by_num_relax_and_amber_only_fills_it_in(self):
        text = colabfold_source("colabfold/batch.py")
        assert "if args.amber and args.num_relax == 0:" in text

    def test_data_is_the_directory_that_holds_params(self):
        assert 'params_dir = data_dir.joinpath("params")' in colabfold_source(
            "colabfold/download.py"
        )

    def test_a_job_that_fails_is_logged_and_skipped_with_status_zero(self):
        text = colabfold_source("colabfold/batch.py")
        for message in ("Could not get MSA/templates for", "Could not generate input features"):
            start = text.index(message)
            assert "continue" in text[start : start + 200]
        assert "Could not predict {jobname}. Not Enough GPU memory?" in text

    def test_the_marker_files_and_the_download_that_the_weights_check_guards_against(self):
        text = colabfold_source("colabfold/download.py")
        for model_type, marker in af2_runner._PARAMS_MARKERS.items():
            assert f'"{marker}"' in text, model_type
        assert text.count("success_marker.is_file()") == 1
        assert "file.extractall(path=params_dir)" in text
        batch = colabfold_source("colabfold/batch.py")
        start = batch.index("if args.num_models > 0:")
        assert "download_alphafold_params(model_type, data_dir)" in batch[start : start + 120]
        assert "data_dir = Path(args.data or default_data_dir)" in batch

    def test_the_parameter_files_are_read_from_params_below_data(self):
        text = colabfold_source("colabfold/alphafold/models.py")
        assert 'path = os.path.join(data_dir, "params", file)' in text
        assert 'file = f"params_model_{model_number}_multimer_v3.npz"' in text

    def test_the_script_is_named_colabfold_batch(self):
        pyproject = colabfold_source("pyproject.toml")
        assert "colabfold_batch = 'colabfold.batch:main'" in pyproject


# ---------------------------------------------------------------------------- custom weights


def parameter_directory(base, name="weights", *, model_type="alphafold2_multimer_v3", content=b"a"):
    """A ``--data`` directory: ``params/`` with one parameter file and the marker file."""
    params = Path(base) / name / "params"
    params.mkdir(parents=True, exist_ok=True)
    (params / "params_model_1_multimer_v3.npz").write_bytes(content)
    (params / af2_runner._PARAMS_MARKERS[model_type]).write_bytes(b"")
    return params.parent


class TestCustomWeights:
    def test_the_runner_takes_a_directory_of_weights(self):
        assert ColabFoldRunner.supports_custom_weights is True
        assert ColabFoldRunner.weights_kind == "directory"

    def test_a_request_without_weights_has_no_weights_field(self):
        request = p53_request()
        assert request.weights is None and "weights" not in request.canonical()

    def test_the_weights_are_in_the_key_by_content_and_not_by_path(self, tmp_path):
        first = p53_request(weights=parameter_directory(tmp_path, "one"))
        moved = p53_request(weights=parameter_directory(tmp_path, "two"))
        changed = p53_request(weights=parameter_directory(tmp_path, "three", content=b"b"))
        assert first.weights.kind == "directory"
        assert first.key() == moved.key() != p53_request().key()
        assert changed.key() != first.key()
        assert "one" not in json.dumps(first.canonical())
        assert first.describe()["weights"]["path"] == str(tmp_path / "one")

    def test_a_weights_reference_of_the_store_gives_the_same_key(self, tmp_path):
        directory = parameter_directory(tmp_path)
        store = PredictionStore(tmp_path / "store")
        reference = store.weights_reference(directory)
        assert isinstance(reference, WeightsRef)
        assert p53_request(weights=reference).key() == p53_request(weights=directory).key()

    def test_the_weights_directory_is_passed_as_data(self, tmp_path):
        directory = parameter_directory(tmp_path)
        command = TestCommandLine.command(p53_request(weights=directory))
        assert command[command.index("--data") + 1] == str(directory)
        assert command.count("--data") == 1

    def test_weights_and_data_dir_cannot_be_combined(self, tmp_path):
        directory = parameter_directory(tmp_path)
        with pytest.raises(ValueError, match="weights or data_dir, not both"):
            p53_request(weights=directory, data_dir=directory)

    def test_data_dir_keeps_its_meaning(self, tmp_path):
        directory = parameter_directory(tmp_path)
        request = p53_request(data_dir=directory)
        assert request.weights is None and request.options["data_dir"] == str(directory)
        assert TestCommandLine.command(request).count("--data") == 1

    def test_a_file_is_not_a_weights_directory(self, tmp_path):
        checkpoint = tmp_path / "params.npz"
        checkpoint.write_bytes(b"x")
        with pytest.raises(ValueError, match="are a directory"):
            p53_request(weights=checkpoint)
        with pytest.raises(ValueError, match="takes its weights as a directory"):
            ColabFoldRunner().prepare(
                self.by_hand(WeightsRef(checkpoint, "file", "0" * 64, 1)), tmp_path
            )

    def test_a_missing_weights_path_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="are a directory"):
            p53_request(weights=tmp_path / "nothing")

    def test_the_params_directory_itself_is_refused(self, tmp_path):
        params = parameter_directory(tmp_path) / "params"
        with pytest.raises(ValueError, match="no params/ directory"):
            p53_request(weights=params)

    def test_a_directory_without_the_marker_is_refused_because_colabfold_would_download(
        self, tmp_path
    ):
        directory = parameter_directory(tmp_path)
        (directory / "params" / "download_complexes_multimer_v3_finished.txt").unlink()
        with pytest.raises(ValueError) as error:
            p53_request(weights=directory)
        text = str(error.value)
        assert "download_complexes_multimer_v3_finished.txt" in text
        assert "downloads the default parameters" in text and "over the files of the same" in text

    def test_the_marker_is_the_one_of_the_model_type(self, tmp_path):
        directory = parameter_directory(tmp_path, model_type="alphafold2_multimer_v2")
        with pytest.raises(ValueError, match="download_complexes_multimer_v3_finished.txt"):
            p53_request(weights=directory)
        request = p53_request(weights=directory, model_type="alphafold2_multimer_v2")
        assert request.options["model_type"] == "alphafold2_multimer_v2"

    def test_an_unknown_model_type_is_refused_before_the_directory_is_read(self, tmp_path):
        with pytest.raises(ValueError, match="model_type must be one of"):
            p53_request(weights=parameter_directory(tmp_path), model_type="auto")

    def test_a_request_built_by_hand_is_checked_when_it_is_prepared(self, tmp_path):
        directory = parameter_directory(tmp_path)
        request = self.by_hand(WeightsRef(directory, "directory", "0" * 64, 1))
        assert ColabFoldRunner().prepare(request, tmp_path / "w").is_file()
        (directory / "params" / "download_complexes_multimer_v3_finished.txt").unlink()
        with pytest.raises(ValueError, match="download_complexes_multimer_v3_finished.txt"):
            ColabFoldRunner().prepare(request, tmp_path / "w2")
        assert not (tmp_path / "w2").exists()

    @staticmethod
    def by_hand(weights):
        return PredictionRequest(
            "af2",
            "q",
            mode="predict",
            binder_chain="B",
            receptor_chain="A",
            sequences={"A": "ACDE", "B": "AGK"},
            weights=weights,
        )

    def test_a_run_with_weights_reaches_the_process_and_a_moved_copy_shares_the_entry(
        self, tmp_path, stub_colabfold
    ):
        store = PredictionStore(tmp_path / "store")
        runner = ColabFoldRunner()
        first = parameter_directory(tmp_path, "tuned")
        entry = store.get_or_run(
            p53_request(runner, weights=store.weights_reference(first)), runner
        )
        (call,) = stub_colabfold.calls()
        assert call["argv"][call["argv"].index("--data") + 1] == str(first)
        again = store.get_or_run(
            p53_request(runner, weights=parameter_directory(tmp_path, "copy")), runner
        )
        assert again.key == entry.key and len(stub_colabfold.calls()) == 1
        other = parameter_directory(tmp_path, "another", content=b"fine-tuned")
        store.get_or_run(p53_request(runner, weights=other), runner)
        assert len(stub_colabfold.calls()) == 2

    def test_the_store_refuses_the_weights_of_a_runner_that_cannot_take_them(
        self, tmp_path, stub_colabfold
    ):
        """The contract of the ABC: a request with weights never runs with the default model."""

        class Plain(ColabFoldRunner):
            supports_custom_weights = False

        request = p53_request(weights=parameter_directory(tmp_path))
        with pytest.raises(ValueError, match="does not take custom weights"):
            PredictionStore(tmp_path / "store").get_or_run(request, Plain())
        assert stub_colabfold.calls() == []


# ---------------------------------------------------------------------------- module facts


def test_importing_the_module_imports_only_the_standard_library():
    code = textwrap.dedent(
        """
        import sys
        import binding_metrics.predictors.af2_runner
        heavy = ("numpy", "scipy", "biotite", "torch", "openmm", "gemmi", "jax")
        print(sorted(m for m in sys.modules if m.split(".")[0] in heavy))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
