"""The record-based EvoBind functions: the adversarial check and the primary score.

``compute_evobind_adversarial_from_records`` is tested first, ``compute_evobind_score_from_record``
at the end of the file; both read predictions through the predictor adapters.

The cross-model tests write one geometry (an alpha-helix receptor with a short binder beside
it) in the layout of every registered model, plus the made-up "stub" model whose pLDDT is on
0-1, with chain names, residue numbering and pLDDT scale that differ from the design file.
Whatever the model, the check must then give the delta COM that follows from the geometry.
A model that is registered later is picked up here by ``sorted(PARSERS)`` and needs only its
``tests/predictors/synth_<model>.py`` writer.

Nothing here is real model output; every expected value follows from the coordinates built
below.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("biotite")

from binding_metrics.metrics.evobind import (  # noqa: E402
    compute_evobind_adversarial_check,
    compute_evobind_adversarial_from_records,
    compute_evobind_score,
    compute_evobind_score_from_record,
)
from binding_metrics.predictors.record import PredictionRecord  # noqa: E402
from binding_metrics.predictors.registry import (  # noqa: E402
    PARSERS,
    ParserSpec,
    get_parser,
    register_parser,
)
from tests.predictors import synth  # noqa: E402
from tests.predictors.contract import writer_module  # noqa: E402
from tests.test_evobind import _chain, _helix_ca, renumber_from_one  # noqa: E402

NAME = "cmplx"
N_RECEPTOR = 12
N_BINDER = 4
RECEPTOR_PLDDT = 40.0
BINDER_PLDDT = 80.0
#: Binder displacement (3, 4) in y and z: 5 angstrom, the Pythagorean triple.
DISPLACEMENT = (0.0, 3.0, 4.0)

MODELS = sorted(set(PARSERS) | {"stub"})


def _complex_atoms(
    *,
    binder_shift=(0.0, 0.0, 0.0),
    receptor_chain="A",
    binder_chain="B",
    first_receptor_res=1,
    first_binder_res=1,
    receptor_names=None,
    binder_names=None,
):
    """The helix complex of ``tests/test_evobind.py`` with the given chain names and numbering.

    Returns ``(atoms, plddt)``; the atoms are receptor (CA, CB per residue) then binder, and
    the coordinates depend only on the residue index, so two calls that differ in names or
    numbering describe the same geometry.
    """
    receptor_ca = _helix_ca(N_RECEPTOR)
    radial = receptor_ca[:, :2] / np.linalg.norm(receptor_ca[:, :2], axis=1, keepdims=True)
    receptor_cb = receptor_ca + 1.5 * np.column_stack([radial, np.zeros(N_RECEPTOR)])
    binder_ca = np.array([[9.0, 0.0, 2.0 + 3.8 * j] for j in range(N_BINDER)]) + np.asarray(
        binder_shift
    )
    binder_cb = binder_ca + [-1.5, 0.0, 0.0]
    atoms = _chain(
        receptor_chain,
        receptor_ca,
        receptor_cb,
        first_res_id=first_receptor_res,
        res_names=receptor_names,
    ) + _chain(
        binder_chain,
        binder_ca,
        binder_cb,
        first_res_id=first_binder_res,
        res_names=binder_names,
    )
    plddt = np.concatenate(
        [np.full(2 * N_RECEPTOR, RECEPTOR_PLDDT), np.full(2 * N_BINDER, BINDER_PLDDT)]
    )
    atoms.set_annotation("b_factor", plddt.copy())
    return atoms, plddt


def _as_synthetic_complex(atoms, plddt):
    """A ``SyntheticComplex`` around ``atoms``, for a model's writer (one token per residue)."""
    base = synth.synthetic_complex()
    n_tokens = int((atoms.atom_name == "CA").sum())
    i = np.arange(n_tokens)[:, None]
    j = np.arange(n_tokens)[None, :]
    return synth.SyntheticComplex(
        atoms=atoms,
        plddt_per_atom=plddt,
        pae=1.0 + 0.5 * i + 0.25 * j,
        pde=0.5 + 0.25 * i + 0.125 * j,
        scalars={**base.scalars, "avg_plddt": float(plddt.mean())},
        chain_ptm=base.chain_ptm,
        chain_pair_iptm=base.chain_pair_iptm,
    )


def _design_path(tmp_path, **kwargs):
    """The design (the input pose): a PDB file with chains A and B."""
    atoms, _ = _complex_atoms(**kwargs)
    return synth.write_structure(atoms, tmp_path / "design.pdb")


def _model_record(model, directory, chain_map, **kwargs):
    """Write the geometry in ``model``'s layout under ``directory`` and load it as a record."""
    atoms, plddt = _complex_atoms(**kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    writer_module(model).write_prediction(directory, NAME, _as_synthetic_complex(atoms, plddt))
    return get_parser(model).load(directory, NAME, chain_map=chain_map)


def _plddt_atol(model):
    module = writer_module(model)
    return getattr(module, "PLDDT_ATOL", 0.05)


@pytest.fixture(autouse=True)
def _stub_is_registered():
    """The stub model (pLDDT on 0-1 in its files) takes part in the cross-model tests."""
    saved = dict(PARSERS)
    register_parser(
        ParserSpec(
            name="stub",
            import_path="tests.predictors.synth_stub:StubParser",
            display_name="Stub model",
            family="af3",
        ),
        replace=True,
    )
    yield
    PARSERS.clear()
    PARSERS.update(saved)


# ---------------------------------------------------------------------------
# The same coordinates written by every model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
class TestSameGeometryInEveryModelFormat:
    """Different chain names, numbering and pLDDT scale, one delta COM."""

    def test_identical_coordinates_give_zero_delta_com(self, tmp_path, model):
        design = _design_path(tmp_path)
        adversary = _model_record(
            model, tmp_path / "adv", {"X": "A", "Y": "B"}, receptor_chain="X", binder_chain="Y"
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
        assert res["n_superposition_residues"] == N_RECEPTOR
        assert res["receptor_pairing"] == "residue_number"
        assert res["binder_pairing"] == "residue_number"
        assert res["evobind_adversarial_score"] == pytest.approx(0.0, abs=0.05)

    def test_the_binder_plddt_is_on_the_0_100_scale_whatever_the_file_scale(self, tmp_path, model):
        design = _design_path(tmp_path)
        adversary = _model_record(
            model, tmp_path / "adv", {"X": "A", "Y": "B"}, receptor_chain="X", binder_chain="Y"
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["afm_mean_plddt_binder"] == pytest.approx(BINDER_PLDDT, abs=_plddt_atol(model))

    def test_a_displaced_binder_gives_the_known_delta_com(self, tmp_path, model):
        design = _design_path(tmp_path)
        adversary = _model_record(
            model,
            tmp_path / "adv",
            {"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            binder_shift=DISPLACEMENT,
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)
        expected = res["afm_mean_if_dist"] * (100.0 / res["afm_mean_plddt_binder"]) * 5.0
        assert res["evobind_adversarial_score"] == pytest.approx(expected, rel=2e-2)

    def test_the_result_equals_the_path_function_on_plain_files(self, tmp_path, model):
        design = _design_path(tmp_path)
        adversary = _model_record(
            model,
            tmp_path / "adv",
            {"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            binder_shift=DISPLACEMENT,
        )
        atoms, plddt = _complex_atoms(binder_shift=DISPLACEMENT)
        plain = synth.write_structure(atoms, tmp_path / "plain.pdb")
        expected = compute_evobind_adversarial_check(
            design, plain, "B", "A", afm_plddt_per_atom=plddt
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert set(res) == set(expected) | {"design_model", "adversary_model"}
        for key, value in expected.items():
            if isinstance(value, float):
                assert res[key] == pytest.approx(value, abs=0.1 if "score" in key else 1e-2), key
            else:
                assert res[key] == value, key

    def test_a_renumbered_model_is_paired_by_position(self, tmp_path, model):
        # the design keeps the numbers of the input (101 and 201), the model renumbers from 1
        design = _design_path(tmp_path, first_receptor_res=101, first_binder_res=201)
        adversary = _model_record(
            model,
            tmp_path / "adv",
            {"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            binder_shift=DISPLACEMENT,
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["receptor_pairing"] == "position"
        assert res["binder_pairing"] == "position"
        assert res["n_superposition_atoms"] == N_RECEPTOR
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)

    def test_the_design_may_be_a_record_with_other_chain_names(self, tmp_path, model):
        design = _model_record(
            model,
            tmp_path / "design",
            {"P": "B", "R": "A"},
            receptor_chain="R",
            binder_chain="P",
            first_receptor_res=301,
            first_binder_res=401,
        )
        adversary = _model_record(
            model,
            tmp_path / "adv",
            {"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            binder_shift=DISPLACEMENT,
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["design_model"] == model
        assert res["adversary_model"] == model
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)


# ---------------------------------------------------------------------------
# The arguments and the results of the wrapper, on hand-built records
# ---------------------------------------------------------------------------


def _record(tmp_path, name="adv", *, model="hand", chain_map=None, plddt="truth", **kwargs):
    """A record over a PDB file written from ``_complex_atoms(**kwargs)``.

    ``plddt`` is ``"truth"`` (the pLDDT of the geometry), None or an array.
    """
    atoms, truth = _complex_atoms(**kwargs)
    path = synth.write_structure(atoms, tmp_path / f"{name}.pdb")
    return PredictionRecord(
        model,
        name,
        structure_path=path,
        chain_map=dict(chain_map or {}),
        plddt_per_atom=truth if isinstance(plddt, str) else plddt,
    )


class TestResultKeys:
    def test_a_record_pair_returns_the_path_keys_and_the_two_model_names(self, tmp_path):
        design = _record(tmp_path, "design", model="alpha")
        adversary = _record(tmp_path, "adv", model="beta", binder_shift=DISPLACEMENT)
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        expected = compute_evobind_adversarial_check(
            design.structure_path,
            adversary.structure_path,
            "B",
            "A",
            afm_plddt_per_atom=adversary.plddt_per_atom,
        )
        assert set(res) == set(expected) | {"design_model", "adversary_model"}
        assert (res["design_model"], res["adversary_model"]) == ("alpha", "beta")
        for key, value in expected.items():
            assert res[key] == pytest.approx(value), key

    def test_a_design_path_has_no_model_name(self, tmp_path):
        design = _design_path(tmp_path)
        res = compute_evobind_adversarial_from_records(design, _record(tmp_path), "B", "A")
        assert res["design_model"] is None
        assert res["adversary_model"] == "hand"

    def test_a_string_path_is_accepted_as_the_design(self, tmp_path):
        design = str(_design_path(tmp_path))
        res = compute_evobind_adversarial_from_records(design, _record(tmp_path), "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)

    def test_the_score_divides_by_the_adversary_binder_plddt(self, tmp_path):
        design = _record(tmp_path, "design")
        binder_only_60 = np.concatenate(
            [np.full(2 * N_RECEPTOR, 10.0), np.full(2 * N_BINDER, 60.0)]
        )
        adversary = _record(tmp_path, binder_shift=DISPLACEMENT, plddt=binder_only_60)
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["afm_mean_plddt_binder"] == pytest.approx(60.0)
        assert res["evobind_adversarial_score"] == pytest.approx(
            res["afm_mean_if_dist"] * (100.0 / 60.0) * 5.0, rel=1e-2
        )

    def test_no_reason_when_the_score_is_computed(self, tmp_path):
        res = compute_evobind_adversarial_from_records(
            _record(tmp_path, "design"), _record(tmp_path), "B", "A"
        )
        assert "reason" not in res


class TestAdversaryWithoutPlddt:
    def test_the_geometry_is_returned_with_a_reason(self, tmp_path):
        design = _record(tmp_path, "design")
        adversary = _record(tmp_path, binder_shift=DISPLACEMENT, plddt=None)
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)
        assert np.isfinite(res["afm_mean_if_dist"])
        assert res["afm_mean_plddt_binder"] is None
        assert res["evobind_adversarial_score"] is None
        assert res["reason"] == "adversary has no per-atom pLDDT"

    def test_the_adapter_reasons_are_appended(self, tmp_path):
        adversary = _record(tmp_path, plddt=None)
        adversary.reasons.append("the confidences file is missing")
        res = compute_evobind_adversarial_from_records(
            _record(tmp_path, "design"), adversary, "B", "A"
        )
        assert res["reason"] == ("adversary has no per-atom pLDDT: the confidences file is missing")

    def test_zero_plddt_gives_no_score_and_names_the_adversary(self, tmp_path):
        adversary = _record(tmp_path, plddt=np.zeros(2 * (N_RECEPTOR + N_BINDER)))
        res = compute_evobind_adversarial_from_records(
            _record(tmp_path, "design"), adversary, "B", "A"
        )
        assert res["evobind_adversarial_score"] is None
        assert res["reason"] == "mean binder pLDDT in the adversary model is zero or not finite"


class TestChainIds:
    def test_chain_ids_are_the_user_ids_after_the_chain_map(self, tmp_path):
        # the model calls the receptor "X" and the binder "Y"; the user calls them A and B
        design = _record(tmp_path, "design")
        adversary = _record(
            tmp_path,
            chain_map={"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            binder_shift=DISPLACEMENT,
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)

    def test_a_chain_map_that_swaps_the_two_names_is_honoured(self, tmp_path):
        # the model writes the receptor as "B" and the binder as "A": swapped against the user's
        design = _record(tmp_path, "design")
        adversary = _record(
            tmp_path,
            chain_map={"B": "A", "A": "B"},
            receptor_chain="B",
            binder_chain="A",
            binder_shift=DISPLACEMENT,
        )
        res = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert res["delta_com_angstrom"] == pytest.approx(5.0, abs=1e-2)

    def test_the_model_ids_are_refused_with_the_chains_that_exist(self, tmp_path):
        # the design file uses X and Y; the adversary renames its X and Y to A and B
        design = _record(tmp_path, "design", receptor_chain="X", binder_chain="Y")
        adversary = _record(
            tmp_path, chain_map={"X": "A", "Y": "B"}, receptor_chain="X", binder_chain="Y"
        )
        with pytest.raises(
            ValueError,
            match=r"binder chain 'Y' is not in the adversary structure.*it has \['A', 'B'\]",
        ):
            compute_evobind_adversarial_from_records(design, adversary, "Y", "X")
        with pytest.raises(
            ValueError, match=r"binder chain 'B' is not in the design structure.*\['X', 'Y'\]"
        ):
            compute_evobind_adversarial_from_records(design, adversary, "B", "A")

    def test_target_chain_is_an_alias_of_receptor_chain(self, tmp_path):
        design, adversary = _record(tmp_path, "design"), _record(tmp_path)
        by_alias = compute_evobind_adversarial_from_records(
            design, adversary, "B", target_chain="A"
        )
        by_name = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert by_alias == by_name
        with pytest.raises(ValueError, match="different chains"):
            compute_evobind_adversarial_from_records(design, adversary, "B", "A", target_chain="B")
        with pytest.raises(TypeError, match="receptor_chain"):
            compute_evobind_adversarial_from_records(design, adversary, "B")


class TestInputChecks:
    def test_the_adversary_must_be_a_record(self, tmp_path):
        design = _record(tmp_path, "design")
        with pytest.raises(TypeError, match="compute_evobind_adversarial_check"):
            compute_evobind_adversarial_from_records(design, design.structure_path, "B", "A")

    def test_the_design_must_be_a_record_or_a_path(self, tmp_path):
        with pytest.raises(
            TypeError, match="design must be a PredictionRecord or a structure path"
        ):
            compute_evobind_adversarial_from_records(42, _record(tmp_path), "B", "A")

    def test_a_record_without_a_structure_file_is_refused(self, tmp_path):
        empty = PredictionRecord("hand", "none", reasons=["no output found"])
        with pytest.raises(ValueError, match="no structure file.*no output found"):
            compute_evobind_adversarial_from_records(_record(tmp_path, "design"), empty, "B", "A")

    def test_a_pLDDT_array_of_the_wrong_length_is_a_value_error(self, tmp_path):
        adversary = _record(tmp_path, plddt=np.full(5, 80.0))
        with pytest.raises(ValueError, match="plddt_per_atom length"):
            compute_evobind_adversarial_from_records(
                _record(tmp_path, "design"), adversary, "B", "A"
            )

    def test_offset_numbering_with_different_residues_is_refused(self, tmp_path):
        # the design numbered 106-117 of a longer receptor, the model renumbered from 1: pairing
        # by position pairs residue k with residue k+5, which the residue names catch
        names = ["ALA", "GLY", "SER", "LEU", "VAL", "ILE", "PHE", "TYR", "LYS", "ARG"]
        names += ["GLU", "ASP", "ASN", "GLN", "HIS", "TRP", "PRO"]
        design = _record(tmp_path, "design", first_receptor_res=106, receptor_names=names[5:17])
        adversary = _record(tmp_path, receptor_names=names[0:12])
        with pytest.raises(ValueError, match="different residue names"):
            compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        res = compute_evobind_adversarial_from_records(
            design, adversary, "B", "A", max_resname_mismatch_fraction=1.0
        )
        assert res["receptor_resname_mismatch_fraction"] == pytest.approx(1.0)

    def test_the_interface_cutoff_is_passed_on(self, tmp_path):
        design, adversary = _record(tmp_path, "design"), _record(tmp_path)
        normal = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        tight = compute_evobind_adversarial_from_records(
            design, adversary, "B", "A", interface_cutoff_angstrom=0.5
        )
        assert normal["interface_fallback_used"] is False
        assert tight["interface_fallback_used"] is True


class TestRecordsAreNotModified:
    def test_the_cached_atoms_keep_their_chain_ids_and_a_second_call_agrees(self, tmp_path):
        design = _record(tmp_path, "design")
        adversary = _record(
            tmp_path,
            chain_map={"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            binder_shift=DISPLACEMENT,
        )
        before = adversary.atoms().coord.copy()
        chains_before = adversary.atoms().chain_id.copy()
        first = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        second = compute_evobind_adversarial_from_records(design, adversary, "B", "A")
        assert first == second
        np.testing.assert_array_equal(adversary.atoms().coord, before)
        np.testing.assert_array_equal(adversary.atoms().chain_id, chains_before)


# ---------------------------------------------------------------------------
# A model that numbers every chain from 1, on the 1YCR coordinates (issue #108)
# ---------------------------------------------------------------------------


def _ycr_second_model_atoms(example_pdb_path):
    """1YCR renumbered from 1 per chain, with pLDDT 40 on the receptor and 80 on the binder."""
    from binding_metrics.metrics.evobind import _load_atoms

    atoms = renumber_from_one(_load_atoms(example_pdb_path))
    plddt = np.where(atoms.chain_id == "B", BINDER_PLDDT, RECEPTOR_PLDDT)
    atoms.set_annotation("b_factor", plddt.copy())
    return atoms, plddt


class TestRenumberedFromOneSecondModel:
    """1YCR: receptor 25-109, binder 17-29 in the design; the second model counts from 1."""

    @pytest.mark.parametrize("model", MODELS)
    def test_every_model_layout_of_the_renumbered_coordinates_agrees_with_the_design(
        self, tmp_path, example_pdb_path, model
    ):
        atoms, plddt = _ycr_second_model_atoms(example_pdb_path)
        directory = tmp_path / "adv"
        directory.mkdir()
        writer_module(model).write_prediction(directory, NAME, _as_synthetic_complex(atoms, plddt))
        adversary = get_parser(model).load(directory, NAME)
        res = compute_evobind_adversarial_from_records(example_pdb_path, adversary, "B", "A")
        assert res["receptor_pairing"] == "position"
        assert res["binder_pairing"] == "position"
        assert res["n_superposition_atoms"] == 85
        assert res["delta_com_angstrom"] == pytest.approx(0.0, abs=1e-2)
        assert res["afm_mean_plddt_binder"] == pytest.approx(BINDER_PLDDT, abs=_plddt_atol(model))

    def test_a_record_and_the_path_function_agree(self, tmp_path, example_pdb_path):
        atoms, plddt = _ycr_second_model_atoms(example_pdb_path)
        path = synth.write_structure(atoms, tmp_path / "boltz_like.pdb")
        record = PredictionRecord("hand", "1ycr", structure_path=path, plddt_per_atom=plddt)
        from_record = compute_evobind_adversarial_from_records(example_pdb_path, record, "B", "A")
        from_paths = compute_evobind_adversarial_check(
            example_pdb_path, path, "B", "A", afm_plddt_per_atom=plddt
        )
        for key, value in from_paths.items():
            assert from_record[key] == pytest.approx(value), key

    def test_a_wrong_pairing_still_raises_for_a_record(self, tmp_path, example_pdb_path):
        atoms, plddt = _ycr_second_model_atoms(example_pdb_path)
        receptor = atoms.chain_id == "A"
        _, first, inverse = np.unique(
            atoms.res_id[receptor], return_index=True, return_inverse=True
        )
        atoms.res_name[receptor] = np.roll(atoms.res_name[receptor][first], 7)[inverse]
        path = synth.write_structure(atoms, tmp_path / "scrambled.pdb")
        record = PredictionRecord("hand", "1ycr", structure_path=path, plddt_per_atom=plddt)
        with pytest.raises(ValueError, match="receptor residues cannot be paired"):
            compute_evobind_adversarial_from_records(example_pdb_path, record, "B", "A")


# ---------------------------------------------------------------------------
# compute_evobind_score_from_record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
class TestScoreFromRecordInEveryModelFormat:
    """The same coordinates in each model's layout give the same score, whatever the chain names."""

    def _record_and_plain_file(self, tmp_path, model, **kwargs):
        record = _model_record(
            model,
            tmp_path / "adv",
            {"X": "A", "Y": "B"},
            receptor_chain="X",
            binder_chain="Y",
            **kwargs,
        )
        atoms, _ = _complex_atoms(**kwargs)
        return record, synth.write_structure(atoms, tmp_path / "plain.pdb")

    def test_the_score_equals_the_path_function_on_a_plain_file(self, tmp_path, model):
        record, plain = self._record_and_plain_file(tmp_path, model)
        _, plddt = _complex_atoms()
        expected = compute_evobind_score(plain, plddt, "B", "A")
        res = compute_evobind_score_from_record(record, "B", "A")
        assert res["model"] == model
        assert set(res) == set(expected) | {"model"}
        for key, value in expected.items():
            if isinstance(value, float):
                assert res[key] == pytest.approx(value, abs=1e-2), key
            else:
                assert res[key] == value, key

    def test_the_binder_plddt_is_on_the_0_100_scale_and_divides_the_distance(self, tmp_path, model):
        record, _ = self._record_and_plain_file(tmp_path, model)
        res = compute_evobind_score_from_record(record, "B", "A")
        assert res["mean_plddt_binder"] == pytest.approx(BINDER_PLDDT, abs=_plddt_atol(model))
        assert res["evobind_score"] == pytest.approx(
            res["if_dist_pep_to_rec"] / (BINDER_PLDDT / 100.0), rel=1e-3
        )


class TestScoreFromRecord:
    def test_a_hand_built_record_gives_the_path_keys_and_the_model_name(self, tmp_path):
        record = _record(tmp_path, model="beta")
        expected = compute_evobind_score(record.structure_path, record.plddt_per_atom, "B", "A")
        res = compute_evobind_score_from_record(record, "B", "A")
        assert res == {"model": "beta", **expected}
        assert res["evobind_score"] == pytest.approx(
            res["if_dist_pep_to_rec"] / (BINDER_PLDDT / 100.0)
        )

    def test_chain_ids_are_the_user_ids_after_the_chain_map(self, tmp_path):
        record = _record(
            tmp_path, chain_map={"X": "A", "Y": "B"}, receptor_chain="X", binder_chain="Y"
        )
        swapped = _record(
            tmp_path,
            "swapped",
            chain_map={"B": "A", "A": "B"},
            receptor_chain="B",
            binder_chain="A",
        )
        plain = _record(tmp_path, "plain")
        reference = compute_evobind_score_from_record(plain, "B", "A")
        for candidate in (record, swapped):
            res = compute_evobind_score_from_record(candidate, "B", "A")
            assert res["if_dist_pep_to_rec"] == pytest.approx(reference["if_dist_pep_to_rec"])
            assert res["evobind_score"] == pytest.approx(reference["evobind_score"])

    def test_the_model_ids_are_refused_with_the_chains_that_exist(self, tmp_path):
        record = _record(
            tmp_path, chain_map={"X": "A", "Y": "B"}, receptor_chain="X", binder_chain="Y"
        )
        with pytest.raises(
            ValueError,
            match=r"binder chain 'Y' is not in the prediction structure.*it has \['A', 'B'\]",
        ):
            compute_evobind_score_from_record(record, "Y", "A")

    def test_the_interface_arguments_are_passed_on(self, tmp_path):
        record = _record(tmp_path)
        default = compute_evobind_score_from_record(record, "B", "A")
        tight = compute_evobind_score_from_record(record, "B", "A", interface_cutoff_angstrom=0.5)
        explicit = compute_evobind_score_from_record(
            record, "B", "A", receptor_interface_residues=[1, 2, 3]
        )
        assert default["interface_fallback_used"] is False
        assert tight["interface_fallback_used"] is True
        assert explicit["n_interface_receptor_residues"] == 3

    def test_target_chain_is_an_alias_of_receptor_chain(self, tmp_path):
        record = _record(tmp_path)
        assert compute_evobind_score_from_record(
            record, "B", target_chain="A"
        ) == compute_evobind_score_from_record(record, "B", "A")
        with pytest.raises(ValueError, match="different chains"):
            compute_evobind_score_from_record(record, "B", "A", target_chain="B")
        with pytest.raises(TypeError, match="receptor_chain"):
            compute_evobind_score_from_record(record, "B")

    def test_a_record_without_plddt_gives_the_distances_and_a_reason(self, tmp_path):
        record = _record(tmp_path, plddt=None)
        record.reasons.append("the confidences file is missing")
        res = compute_evobind_score_from_record(record, "B", "A")
        assert np.isfinite(res["if_dist_pep_to_rec"])
        assert res["mean_plddt_binder"] is None
        assert res["evobind_score"] is None
        assert res["reason"] == "prediction has no per-atom pLDDT: the confidences file is missing"

    def test_zero_plddt_gives_no_score_and_the_path_reason(self, tmp_path):
        record = _record(tmp_path, plddt=np.zeros(2 * (N_RECEPTOR + N_BINDER)))
        res = compute_evobind_score_from_record(record, "B", "A")
        assert res["evobind_score"] is None
        assert res["reason"] == "mean binder pLDDT is zero or not finite"

    def test_no_reason_when_the_score_is_computed(self, tmp_path):
        assert "reason" not in compute_evobind_score_from_record(_record(tmp_path), "B", "A")

    def test_a_wrong_length_plddt_array_is_a_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="plddt_per_atom length"):
            compute_evobind_score_from_record(_record(tmp_path, plddt=np.full(5, 80.0)), "B", "A")

    def test_the_record_must_be_a_record_with_a_structure(self, tmp_path):
        with pytest.raises(TypeError, match="compute_evobind_score"):
            compute_evobind_score_from_record(_record(tmp_path).structure_path, "B", "A")
        empty = PredictionRecord("hand", "none", reasons=["no output found"])
        with pytest.raises(ValueError, match="no structure file.*no output found"):
            compute_evobind_score_from_record(empty, "B", "A")

    def test_the_record_is_not_modified(self, tmp_path):
        record = _record(
            tmp_path, chain_map={"X": "A", "Y": "B"}, receptor_chain="X", binder_chain="Y"
        )
        before = record.atoms().chain_id.copy()
        first = compute_evobind_score_from_record(record, "B", "A")
        assert compute_evobind_score_from_record(record, "B", "A") == first
        np.testing.assert_array_equal(record.atoms().chain_id, before)
