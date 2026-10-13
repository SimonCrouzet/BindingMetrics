"""Boltz-2 embeddings: the request, the command, and the file the adapter registers.

``--write_embeddings`` makes Boltz-2 write ``embeddings_<name>.npz`` (arrays ``s`` and ``z``)
beside the structures, once per input. No Boltz-2 is needed: the output is synthetic
(``tests/predictors/synth_boltz2.py``) and the arrays are tiny, because the adapter never reads
them. Whether the session records a reason when the file is missing is the session's business
(it knows what was asked); here the adapter is shown to leave no trace of a file that is not there.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
pytest.importorskip("biotite")

from binding_metrics.predictors import boltz2_runner  # noqa: E402
from binding_metrics.predictors.boltz2 import Boltz2Parser  # noqa: E402
from binding_metrics.predictors.boltz2_runner import _OWNED_FLAGS, Boltz2Runner  # noqa: E402
from binding_metrics.predictors.session import PredictionSession  # noqa: E402
from binding_metrics.predictors.store import PredictionRequest, PredictionStore  # noqa: E402
from tests.predictors import synth, synth_boltz2  # noqa: E402
from tests.predictors.test_boltz2_runner import FakeBoltz  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
YCR = DATA / "example_linear_p53_1YCR.pdb"


@pytest.fixture(autouse=True)
def boltz_version(monkeypatch):
    """Boltz-2 is not installed here: the version is 2.2.1 whatever the machine has."""
    monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda conda_env: "2.2.1")


@pytest.fixture
def fake_boltz(tmp_path, monkeypatch):
    return FakeBoltz(tmp_path, monkeypatch)


def make_request(**keywords):
    keywords.setdefault("name", "ycr")
    return Boltz2Runner().make_request(YCR, binder_chain="B", receptor_chain="A", **keywords)


def write_samples(directory, name="ycr", samples=1, embeddings=True):
    """A synthetic output of ``samples`` samples, with ``embeddings_<name>.npz`` if asked.

    Returns the folder that holds the sample files. ``s`` is (1, tokens, 8) and ``z`` is
    (1, tokens, tokens, 4): the shapes are made up, since nothing reads them.
    """
    for rank in range(samples):
        synth_boltz2.write_prediction(
            directory, name, synth.synthetic_complex(plddt_shift=5.0 * rank), sample=rank + 1
        )
    folder = synth_boltz2.prediction_directory(directory, name)
    if embeddings:
        np.savez_compressed(
            folder / f"embeddings_{name}.npz",
            s=np.zeros((1, 7, 8), dtype=np.float32),
            z=np.zeros((1, 7, 7, 4), dtype=np.float32),
        )
    return folder


class TestTheRequest:
    def test_the_option_is_absent_unless_the_embeddings_are_asked_for(self):
        assert "write_embeddings" not in make_request().options
        assert "write_embeddings" not in make_request(write_embeddings=False).options
        assert make_request(write_embeddings=True).options["write_embeddings"] is True

    def test_false_keeps_the_key_and_true_changes_it(self):
        key = make_request().key()
        assert make_request(write_embeddings=False).key() == key
        assert make_request(write_embeddings=True).key() != key

    @pytest.mark.parametrize("bad", [1, 0, "yes", None, "True"])
    def test_a_value_that_is_not_a_bool_is_refused(self, bad):
        with pytest.raises(TypeError, match="write_embeddings must be True or False"):
            make_request(write_embeddings=bad)

    def test_extra_args_may_not_repeat_the_flag(self):
        assert "--write_embeddings" in _OWNED_FLAGS
        for extra in (["--write_embeddings"], ["--write_embeddings=true"]):
            with pytest.raises(ValueError, match="repeats --write_embeddings"):
                make_request(extra_args=extra)

    def test_a_request_made_elsewhere_with_a_bad_option_is_refused(self, tmp_path):
        request = PredictionRequest(
            "boltz2",
            "ycr",
            mode="predict",
            input_path=YCR,
            binder_chain="B",
            receptor_chain="A",
            options={"write_embeddings": "yes"},
        )
        with pytest.raises(TypeError, match="write_embeddings must be True or False"):
            Boltz2Runner().prepare(request, tmp_path / "work")
        assert not (tmp_path / "work").exists()


class TestTheCommand:
    def command(self, tmp_path, **keywords):
        request = make_request(**keywords)
        return Boltz2Runner()._command(request, tmp_path / "ycr.yaml", tmp_path)

    def test_the_flag_is_passed_once_when_asked_for(self, tmp_path):
        assert self.command(tmp_path, write_embeddings=True).count("--write_embeddings") == 1

    @pytest.mark.parametrize("keywords", [{}, {"write_embeddings": False}])
    def test_no_flag_otherwise(self, tmp_path, keywords):
        assert "--write_embeddings" not in self.command(tmp_path, **keywords)

    def test_the_other_flags_stay_where_they_were(self, tmp_path):
        plain = self.command(tmp_path, use_msa_server=False)
        asked = self.command(tmp_path, use_msa_server=False, write_embeddings=True)
        assert [word for word in asked if word != "--write_embeddings"] == plain


class TestTheFileTheAdapterRegisters:
    def test_the_file_is_registered_for_each_sample_that_exists(self, tmp_path):
        folder = write_samples(tmp_path, samples=2)
        parser = Boltz2Parser()
        for sample in (1, 2):
            files = parser.find_files(folder, "ycr", sample=sample)
            assert files.extra["embeddings"] == folder / "embeddings_ycr.npz"
            assert files.has_output()
        assert set(parser.find_files(folder, "ycr").extra) == {"plddt", "pae", "pde", "embeddings"}

    def test_a_sample_number_past_the_last_one_finds_nothing(self, tmp_path):
        folder = write_samples(tmp_path, samples=2)
        parser = Boltz2Parser()
        files = parser.find_files(folder, "ycr", sample=3)
        assert not files.has_output() and "embeddings" not in files.extra
        assert [ref.sample for ref in parser.list_samples(folder, "ycr")] == [1, 2]

    def test_the_embeddings_alone_are_no_sample(self, tmp_path):
        folder = tmp_path / "boltz_results_ycr" / "predictions" / "ycr"
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "embeddings_ycr.npz", s=np.zeros(1), z=np.zeros(1))
        files = Boltz2Parser().find_files(folder, "ycr")
        assert not files.has_output() and files.extra == {}

    def test_the_path_is_found_from_the_directory_the_sample_is_in(self, tmp_path):
        write_samples(tmp_path / "run")
        parser = Boltz2Parser()
        files = parser.find_files(tmp_path / "run", "ycr")  # the folder is two levels down
        folder = synth_boltz2.prediction_directory(tmp_path / "run", "ycr")
        assert files.extra["embeddings"] == folder / "embeddings_ycr.npz"

    def test_a_file_of_another_name_is_not_taken(self, tmp_path):
        folder = write_samples(tmp_path, embeddings=False)
        np.savez_compressed(folder / "embeddings_other.npz", s=np.zeros(1), z=np.zeros(1))
        assert "embeddings" not in Boltz2Parser().find_files(folder, "ycr").extra

    def test_parse_puts_the_path_in_the_extras(self, tmp_path):
        folder = write_samples(tmp_path)
        parser = Boltz2Parser()
        record = parser.parse(parser.find_files(folder, "ycr"), name="ycr")
        assert record.extras["embeddings_path"] == str(folder / "embeddings_ycr.npz")
        assert record.reasons == []

    def test_the_arrays_are_never_read(self, tmp_path):
        folder = write_samples(tmp_path)
        (folder / "embeddings_ycr.npz").write_bytes(b"this is not an npz file")
        parser = Boltz2Parser()
        record = parser.parse(parser.find_files(folder, "ycr"), name="ycr")
        assert record.extras["embeddings_path"].endswith("embeddings_ycr.npz")

    def test_without_the_file_the_record_has_no_trace_of_it(self, tmp_path):
        folder = write_samples(tmp_path, embeddings=False)
        parser = Boltz2Parser()
        files = parser.find_files(folder, "ycr")
        record = parser.parse(files, name="ycr")
        assert "embeddings" not in files.extra
        assert "embeddings_path" not in record.extras
        assert record.reasons == []

    def test_the_path_follows_a_moved_directory(self, tmp_path):
        folder = write_samples(tmp_path / "before")
        moved = tmp_path / "after"
        shutil.move(str(tmp_path / "before"), str(moved))
        parser = Boltz2Parser()
        record = parser.parse(parser.find_files(moved, "ycr"), name="ycr")
        assert record.extras["embeddings_path"] == str(
            moved / "boltz_results_ycr" / "predictions" / "ycr" / "embeddings_ycr.npz"
        )
        assert not folder.exists()
        assert Path(record.extras["embeddings_path"]).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="the stub is a POSIX shell script")
class TestTheRoundTripThroughTheStore:
    def run(self, tmp_path, fake_boltz, *, embeddings, **keywords):
        write_samples(fake_boltz.tmp_path / "canned", embeddings=embeddings)
        fake_boltz.monkeypatch.setenv("FAKE_BOLTZ_CANNED", str(fake_boltz.tmp_path / "canned"))
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        request = runner.make_request(
            YCR, name="ycr", binder_chain="B", receptor_chain="A", **keywords
        )
        return session, request

    def test_the_stored_entry_keeps_the_file_and_the_record_points_at_it(
        self, tmp_path, fake_boltz
    ):
        session, request = self.run(tmp_path, fake_boltz, embeddings=True, write_embeddings=True)
        record = session.record(request)
        (call,) = fake_boltz.calls()
        assert "--write_embeddings" in call["argv"]
        path = Path(record.extras["embeddings_path"])
        assert path.is_file() and path.name == "embeddings_ycr.npz"
        assert path.parent == session.entry(request).prediction_dir  # the stored place

    def test_an_entry_made_without_the_flag_is_no_hit_for_a_request_that_asks(
        self, tmp_path, fake_boltz
    ):
        session, plain = self.run(tmp_path, fake_boltz, embeddings=False)
        record = session.record(plain)
        assert "embeddings_path" not in record.extras
        assert "--write_embeddings" not in fake_boltz.calls()[0]["argv"]

        asking = Boltz2Runner().make_request(
            YCR, name="ycr", binder_chain="B", receptor_chain="A", write_embeddings=True
        )
        assert asking.key() != plain.key()
        write_samples(fake_boltz.tmp_path / "canned", embeddings=True)
        session.record(asking)
        assert len(fake_boltz.calls()) == 2
