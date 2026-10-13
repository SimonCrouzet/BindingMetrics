"""Boltz-2 in the mode ``refold-lock``: the receptor pinned by a forced template, the binder free.

The YAML and the template file that the runner writes for the mode, the threshold in the key, the
keys of the requests that existed before the mode, the declaration of the parser, and one run
through the stub of ``boltz`` (``tests/predictors/test_boltz2_runner.py``). No Boltz-2, no GPU and
no network are needed. What the guidance does to a prediction is a measurement of the real model,
not a property of the files written here.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("gemmi")
pytest.importorskip("yaml")

from binding_metrics.capabilities import MODES, preflight, profile_input  # noqa: E402
from binding_metrics.predictors import boltz2_runner  # noqa: E402
from binding_metrics.predictors.boltz2_runner import (  # noqa: E402
    DEFAULT_LOCK_THRESHOLD_ANGSTROM,
    RUN_MODES,
    Boltz2Runner,
)
from binding_metrics.predictors.registry import PARSERS  # noqa: E402
from binding_metrics.predictors.session import (  # noqa: E402
    PredictionSession,
    receptor_forced,
    receptor_templated,
)
from binding_metrics.predictors.store import PredictionStore  # noqa: E402
from tests.predictors.test_boltz2_runner import (  # noqa: E402
    CWA,
    CWA_A,
    CWA_C,
    CWA_C_MODIFICATIONS,
    YCR,
    YCR_A,
    YCR_B,
    FakeBoltz,
    atom_positions,
    boltz_view,
    quoted,
    read_yaml,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stubs are POSIX shell scripts")

DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(autouse=True)
def boltz_version(monkeypatch):
    """Boltz-2 is not installed here: the version is 2.2.1 whatever the machine has."""
    monkeypatch.setattr(boltz2_runner, "_installed_boltz_version", lambda conda_env: "2.2.1")


@pytest.fixture
def fake_boltz(tmp_path, monkeypatch):
    return FakeBoltz(tmp_path, monkeypatch)


def make_request(path=YCR, binder="B", receptor="A", **keywords):
    keywords.setdefault("name", "ycr")
    keywords.setdefault("mode", "refold-lock")
    return Boltz2Runner().make_request(
        path, binder_chain=binder, receptor_chain=receptor, **keywords
    )


def prepare(tmp_path, request):
    """``(input directory, YAML path)`` after ``prepare``."""
    work = tmp_path / "work"
    work.mkdir(parents=True)
    yaml_path = Boltz2Runner().prepare(request, work)
    return work / "input", yaml_path


class TestTheYaml:
    def test_the_receptor_has_one_forced_template_and_the_binder_none(self, tmp_path):
        input_dir, yaml_path = prepare(tmp_path, make_request())
        assert yaml_path.read_text(encoding="utf-8") == (
            "version: 1\nsequences:\n"
            f'  - protein:\n      id: "A"\n      sequence: "{YCR_A}"\n'
            f'  - protein:\n      id: "B"\n      sequence: "{YCR_B}"\n'
            f"templates:\n  - cif: {quoted(input_dir / 'templates' / 'receptor.cif')}\n"
            '    chain_id: ["A"]\n    template_id: ["A"]\n'
            "    force: true\n    threshold: 2.0\n"
        )
        assert [p.name for p in (input_dir / "templates").iterdir()] == ["receptor.cif"]

    def test_the_schema_of_boltz_sees_one_template_for_chain_a_alone(self, tmp_path):
        _, yaml_path = prepare(tmp_path, make_request(use_msa_server=False))
        document = read_yaml(yaml_path)
        assert set(document) == {"version", "sequences", "templates"}
        (template,) = document["templates"]
        assert set(template) == {"cif", "chain_id", "template_id", "force", "threshold"}
        assert template["chain_id"] == template["template_id"] == ["A"]
        assert template["force"] is True and template["threshold"] == 2.0
        assert Path(template["cif"]).name == "receptor.cif"

    @pytest.mark.parametrize("threshold, text", [(0.5, "0.5"), (1, "1.0"), (3.5, "3.5")])
    def test_the_threshold_of_the_request_is_written(self, tmp_path, threshold, text):
        request = make_request(lock_threshold_angstrom=threshold)
        _, yaml_path = prepare(tmp_path, request)
        assert yaml_path.read_text(encoding="utf-8").endswith(
            f"    force: true\n    threshold: {text}\n"
        )

    def test_it_differs_from_refold_by_the_force_and_the_threshold_only(self, tmp_path):
        _, locked = prepare(tmp_path / "lock", make_request())
        _, plain = prepare(tmp_path / "plain", make_request(mode="refold"))
        lines = locked.read_text(encoding="utf-8").splitlines()
        assert [line for line in lines if "force" not in line and "threshold" not in line] == [
            line.replace(str(tmp_path / "plain"), str(tmp_path / "lock"))
            for line in plain.read_text(encoding="utf-8").splitlines()
        ]

    def test_it_differs_from_score_lock_by_the_chains_it_lists(self, tmp_path):
        _, receptor_lock = prepare(tmp_path / "a", make_request())
        _, both_lock = prepare(tmp_path / "b", make_request(mode="score-lock"))
        assert read_yaml(receptor_lock)["templates"][0]["chain_id"] == ["A"]
        assert read_yaml(both_lock)["templates"][0]["chain_id"] == ["A", "B"]

    def test_the_msa_setting_is_written_on_both_entities_as_in_the_other_modes(self, tmp_path):
        _, yaml_path = prepare(tmp_path, make_request(use_msa_server=False))
        assert yaml_path.read_text(encoding="utf-8").count('      msa: "empty"\n') == 2


class TestTheTemplateFile:
    def test_it_holds_the_receptor_chain_alone_with_every_residue_matched(self, tmp_path):
        input_dir, _ = prepare(tmp_path, make_request())
        view = boltz_view(input_dir / "templates" / "receptor.cif")
        assert set(view) == {"A"}
        polymer, letters, match, _ = view["A"]
        assert polymer == "PeptideL" and letters == YCR_A and match == "|" * len(YCR_A)

    def test_it_keeps_the_frame_of_the_input(self, tmp_path):
        input_dir, _ = prepare(tmp_path, make_request())
        written = atom_positions(input_dir / "templates" / "receptor.cif")
        original = atom_positions(YCR)
        assert written and {chain for chain, _, _ in written} == {"A"}
        for key, position in written.items():
            assert position == pytest.approx(original[key], abs=1e-3)

    def test_it_is_the_template_that_refold_writes(self, tmp_path):
        locked, _ = prepare(tmp_path / "lock", make_request())
        plain, _ = prepare(tmp_path / "plain", make_request(mode="refold"))
        assert atom_positions(locked / "templates" / "receptor.cif") == pytest.approx(
            atom_positions(plain / "templates" / "receptor.cif")
        )


class TestAModifiedBinder:
    """1CWA: a cyclic binder with nine modified residues, which the mode leaves free."""

    def test_the_binder_keeps_its_modifications_and_its_ring_and_has_no_template(self, tmp_path):
        request = make_request(CWA, binder="C", receptor="A", use_msa_server=False)
        input_dir, yaml_path = prepare(tmp_path, request)
        document = read_yaml(yaml_path)
        receptor, binder = (item["protein"] for item in document["sequences"])
        assert receptor["sequence"] == CWA_A and "modifications" not in receptor
        assert binder["sequence"] == CWA_C and binder["cyclic"] is True
        assert [(m["position"], m["ccd"]) for m in binder["modifications"]] == (CWA_C_MODIFICATIONS)
        (template,) = document["templates"]
        assert template["chain_id"] == ["A"] and template["force"] is True
        assert [p.name for p in (input_dir / "templates").iterdir()] == ["receptor.cif"]
        assert set(boltz_view(input_dir / "templates" / "receptor.cif")) == {"A"}


class TestTheThresholdInTheKey:
    def test_the_default_is_the_package_choice_and_is_in_the_options(self):
        request = make_request()
        assert DEFAULT_LOCK_THRESHOLD_ANGSTROM == 2.0
        assert request.options["lock_threshold_angstrom"] == 2.0
        assert request.mode == "refold-lock"

    def test_a_threshold_changes_the_key_and_the_default_spelled_out_does_not(self):
        default = make_request().key()
        assert make_request(lock_threshold_angstrom=2).key() == default
        assert make_request(lock_threshold_angstrom=2.0).key() == default
        keys = {make_request(lock_threshold_angstrom=value).key() for value in (0.5, 1.0, 2.0, 3.0)}
        assert len(keys) == 4

    def test_the_mode_is_in_the_key(self):
        keys = {mode: make_request(mode=mode).key() for mode in MODES}
        assert len(set(keys.values())) == len(MODES)

    def test_both_lock_modes_take_the_threshold_and_the_others_refuse_it(self):
        for mode in ("score-lock", "refold-lock"):
            assert make_request(mode=mode, lock_threshold_angstrom=1.5).options[
                "lock_threshold_angstrom"
            ] == pytest.approx(1.5)
        for mode in ("predict", "refold", "score"):
            with pytest.raises(ValueError, match="only applies to the lock modes"):
                make_request(mode=mode, lock_threshold_angstrom=1.5)
            assert make_request(mode=mode).options["lock_threshold_angstrom"] is None

    def test_a_request_made_by_hand_without_the_threshold_is_refused(self, tmp_path):
        request = make_request()
        options = {k: v for k, v in request.options.items() if k != "lock_threshold_angstrom"}
        by_hand = type(request)(
            "boltz2",
            "ycr",
            mode="refold-lock",
            input_path=YCR,
            binder_chain="B",
            receptor_chain="A",
            options=options,
        )
        with pytest.raises(TypeError, match="lock_threshold_angstrom must be a number"):
            Boltz2Runner().prepare(by_hand, tmp_path / "work")
        assert not (tmp_path / "work").exists()


class TestTheOlderRequestsKeepTheirKeys:
    """Keys of 1YCR requests (binder B, receptor A, version 2.2.1, defaults otherwise) as the
    runner computed them before ``refold-lock`` was a mode."""

    OLD_KEYS = {
        ("predict", True): "ff15638f706ab934ac755157bfec5c7c827bd3a7930db0bb1d83e78204060b32",
        ("predict", False): "bca1d4b3dd349adf065ae18193a481b352f60ade1f1580eafae4653d22b5b09b",
        ("refold", True): "a1732c7d26aa71db50e418ac70bf7a98b6d42efb4cd8463f80ecff993dc2a13a",
        ("refold", False): "68305dc08f9714b64deb20765ba99fec55612b71e0ae1f98e0a6ee5b012218b2",
        ("score", True): "692fe603a5b9eed871027a1333345b5d9981b769475112c2ac1a60db64129028",
        ("score", False): "11208cf8635209c107737ab93561b9841fb15efa897959ec8ac49dec4057b3f2",
        ("score-lock", True): "db31325aa968a4f49e9011345e47a597440072387a3c2f2307e9978af8e32489",
        ("score-lock", False): "f35b6f09c5dfe972cafad4c535f40f1fb8a94672129a525aa06b993f6970a644",
    }

    @pytest.mark.parametrize("mode, server", sorted(OLD_KEYS))
    def test_the_key_is_the_one_it_was(self, mode, server):
        assert make_request(mode=mode, use_msa_server=server).key() == self.OLD_KEYS[mode, server]

    def test_a_score_lock_request_with_a_threshold_keeps_its_key(self):
        request = make_request(
            mode="score-lock",
            lock_threshold_angstrom=1.0,
            seeds=[43],
            num_samples=2,
            use_msa_server=False,
        )
        assert request.key() == "66a7718b42b1a301196785aba7833949048da72526aa3d808a20c3d12d05bb44"

    def test_refold_has_no_lock_option_in_its_options(self):
        assert make_request(mode="refold").options["lock_threshold_angstrom"] is None
        assert set(make_request(mode="refold").options) == {
            "use_msa_server",
            "binder_cyclic",
            "on_unmappable_residue",
            "lock_threshold_angstrom",
            "extra_args",
        }


class TestTheDeclarationAndThePreflightCheck:
    caps = PARSERS["boltz2"].load_capabilities()

    def test_boltz2_declares_every_mode_and_the_runner_builds_every_one(self):
        assert self.caps.modes == frozenset(MODES)
        assert set(RUN_MODES) == set(MODES) and Boltz2Runner.run_modes == RUN_MODES
        assert self.caps.is_unconstrained and "modes" not in self.caps.reasons

    def test_the_caveat_is_that_of_a_guidance_term_and_names_the_mode(self):
        sentence = self.caps.caveats["modes:refold-lock"]
        for part in ("force: true", "TemplateReferencePotential", "weight 0.1", "not a hard"):
            assert part in sentence
        assert "receptor" in sentence and "measured once" in sentence

    def test_the_check_accepts_boltz2_for_a_peptide_and_warns_once(self):
        profile = profile_input(YCR, "B", "A")
        report = preflight(profile, [], "boltz2", mode="refold-lock")
        assert report.compatible
        (warning,) = report.warnings
        assert warning == f"predictor Boltz-2 2.2.1: {self.caps.caveats['modes:refold-lock']}"

    def test_the_other_models_are_still_refused(self):
        from binding_metrics.capabilities import IncompatibleInputError
        from binding_metrics.cli import prediction

        profile = profile_input(YCR, "B", "A")
        with pytest.raises(IncompatibleInputError, match="mode 'refold-lock' was requested"):
            preflight(profile, [], "of3", mode="refold-lock")
        for model in ("af2", "protenix"):
            with pytest.raises(ValueError, match="cannot be run from here"):
                prediction.check_predictor(model, None, "refold-lock")

    def test_the_lock_modes_force_the_receptor_and_every_template_mode_templates_it(self):
        request = make_request()
        assert receptor_forced(request) is True and receptor_templated(request) is True
        assert receptor_forced(make_request(mode="refold")) is False


class TestRunningTheStub:
    def test_the_process_gets_the_forced_receptor_template(self, tmp_path, fake_boltz):
        fake_boltz.canned("ycr")
        work = tmp_path / "work"
        work.mkdir()
        Boltz2Runner().run(make_request(lock_threshold_angstrom=1.0), work)
        (call,) = fake_boltz.calls()
        assert call["templates"] == ["receptor.cif"]
        assert call["yaml"].endswith('    chain_id: ["A"]\n    template_id: ["A"]\n'
                                     "    force: true\n    threshold: 1.0\n")  # fmt: skip
        assert "--use_msa_server" in call["argv"]

    def test_the_store_keeps_one_entry_per_threshold_and_the_pose_says_what_was_asked(
        self, tmp_path, fake_boltz
    ):
        fake_boltz.canned("ycr")
        runner = Boltz2Runner()
        session = PredictionSession(PredictionStore(tmp_path / "store"), [runner])
        tight, loose = make_request(lock_threshold_angstrom=0.5), make_request()
        pose = session.record(tight)
        session.record(tight)
        assert len(fake_boltz.calls()) == 1
        session.record(loose)
        assert len(fake_boltz.calls()) == 2
        assert pose.provenance["mode"] == "refold-lock"
        assert pose.provenance["receptor_forced"] is True
        assert pose.provenance["receptor_templated"] is True
        assert pose.reasons == []
