"""Which keys the aggregates list (issue #191).

The aggregates summarised every numeric leaf as a metric, the identity of a prediction
(``seed``, ``sample``, ``n_atoms``, ``seed_value``) among them; listed a key that only one
back-end writes for the other back-end as n_ok 0 of n_total.
"""

import numpy as np
import pytest

from binding_metrics.metrics._common import load_structure
from binding_metrics.poses.aggregate import aggregate_poses
from binding_metrics.poses.reference import Reference
from binding_metrics.poses.score import score_pose_set
from tests.conftest import DATA_DIR

pytest.importorskip("biotite.structure")

YCR = DATA_DIR / "example_linear_p53_1YCR.pdb"
IDENTITY = ("seed", "sample", "n_atoms", "seed_value")


def _predicted_poses(root):
    """Two samples each of a Boltz-2 and an OpenFold3 run of 1YCR, with made-up confidences.

    The two models write the pair scores under different keys ("A-B" and "(A, B)").
    """
    from binding_metrics.predictors import get_parser
    from tests.predictors import synth, synth_boltz2, synth_of3

    atoms = load_structure(YCR)
    atoms = atoms[(~atoms.hetero) & np.isin(atoms.chain_id, ["A", "B"]) & (atoms.element != "H")]
    n_tokens = int(synth_boltz2.token_index_of_atoms(atoms).max()) + 1
    poses = []
    for model, writer in (("boltz2", synth_boltz2), ("of3", synth_of3)):
        for name, shift in (("sampleone", 0.0), ("sampletwo", 8.0)):
            plddt = np.linspace(60.0, 95.0, atoms.array_length()) - shift
            sample_atoms = atoms.copy()
            sample_atoms.set_annotation("b_factor", plddt)
            i, j = np.arange(n_tokens)[:, None], np.arange(n_tokens)[None, :]
            truth = synth.SyntheticComplex(
                atoms=sample_atoms,
                plddt_per_atom=plddt,
                pae=1.0 + 0.01 * i + 0.01 * j,
                pde=0.5 + 0.01 * i + 0.005 * j,
                scalars={
                    "avg_plddt": float(plddt.mean()),
                    "ptm": 0.88,
                    "iptm": 0.76,
                    "gpde": 1.2,
                    "ranking_score": 0.82 - shift / 100,
                    "has_clash": 0.0,
                    "disorder": 0.1,
                },
                chain_ptm={"A": 0.88, "B": 0.8},
                chain_pair_iptm={"A-B": 0.76, "B-A": 0.74},
            )
            writer.write_prediction(root, name, truth)
            poses.append(get_parser(model).load(root, name))
    return poses


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    poses = _predicted_poses(tmp_path_factory.mktemp("two_models"))
    return score_pose_set(
        poses, "B", "A", metrics=["prediction"], reference=Reference(YCR, "A", "B")
    )


def _with(entries, prefix):
    return sorted(key for key in entries if key.startswith(prefix))


class TestScorePoseSet:
    def test_the_poses_come_from_two_back_ends(self, result):
        assert list(result["aggregates"]) == ["boltz2", "of3", "all"]
        assert [row["status"] for row in result["poses"]] == ["ok"] * 4

    @pytest.mark.parametrize("group", ["boltz2", "of3", "all"])
    def test_the_identity_of_a_prediction_is_not_aggregated(self, result, group):
        listed = result["aggregates"][group]
        for block in ("prediction", "native"):
            for field in IDENTITY:
                assert f"{block}.{field}" not in listed

    def test_the_confidences_of_a_prediction_still_are(self, result):
        listed = result["aggregates"]["boltz2"]
        assert listed["prediction.avg_plddt"]["n_ok"] == 2
        assert listed["prediction.iptm"]["n_ok"] == 2

    def test_a_pair_score_of_one_model_is_not_listed_for_the_other(self, result):
        boltz, of3 = result["aggregates"]["boltz2"], result["aggregates"]["of3"]
        assert "native.chain_pair_iptm.A-B" in boltz
        assert "native.chain_pair_iptm.(A, B)" not in boltz
        assert "prediction.chain_pair_iptm.A-B" not in of3
        assert "prediction.bespoke_iptm.(A, B)" not in boltz
        assert "prediction.bespoke_iptm.(A, B)" in of3

    def test_a_key_that_one_model_lacks_counts_the_other_model_in_all(self, result):
        entry = result["aggregates"]["all"]["prediction.bespoke_iptm.(A, B)"]
        assert (entry["n_ok"], entry["n_total"]) == (2, 4)

    def test_no_listed_key_is_absent_from_every_pose_of_its_group(self, result):
        for group, listed in result["aggregates"].items():
            for key, entry in listed.items():
                nothing_but_absent = set(entry["reasons"]) == {"no value for this key"}
                assert entry["n_ok"] > 0 or not nothing_but_absent, (group, key)


def _row(pose_id, backend, **blocks):
    return {
        "id": pose_id,
        "status": "ok",
        "provenance": {"backend": backend},
        "reasons": {},
        **blocks,
    }


class TestAggregatePoses:
    def test_an_identity_key_is_left_out_by_default_and_can_be_asked_for(self):
        rows = [
            _row("p1", "m", prediction={"seed": 1, "sample": 1, "n_atoms": 8, "avg_plddt": 80.0}),
            _row("p2", "m", prediction={"seed": 2, "sample": 1, "n_atoms": 8, "avg_plddt": 90.0}),
        ]
        assert _with(aggregate_poses(rows)["m"], "prediction.") == ["prediction.avg_plddt"]
        kept = aggregate_poses(rows, exclude_keys=())["m"]
        assert "prediction.seed" in kept and "prediction.n_atoms" in kept

    def test_a_key_absent_from_every_pose_of_a_group_is_not_listed_for_it(self):
        rows = [
            _row("a1", "a", native={"iptm": 0.5, "only_a": 3.0}),
            _row("b1", "b", native={"iptm": 0.6}),
            _row("b2", "b", native={"iptm": 0.7}),
        ]
        aggregates = aggregate_poses(rows)
        assert "native.only_a" in aggregates["a"]
        assert "native.only_a" not in aggregates["b"]
        assert (
            aggregates["all"]["native.only_a"]["n_ok"],
            aggregates["all"]["native.only_a"]["n_total"],
        ) == (1, 3)

    def test_a_key_with_no_valid_value_in_a_group_is_still_listed(self):
        rows = [_row("a1", "a", prediction={"binder_ca_rmsd": float("nan"), "iptm": 0.5})]
        entry = aggregate_poses(rows)["a"]["prediction.binder_ca_rmsd"]
        assert (entry["n_ok"], entry["n_total"]) == (0, 1)

    def test_a_group_of_failed_poses_still_counts_them(self):
        failed = {
            "id": "f1",
            "status": "failed",
            "reason": "no frame",
            "provenance": {"backend": "b"},
        }
        rows = [_row("a1", "a", interface={"delta_sasa": 100.0}), failed]
        entry = aggregate_poses(rows)["b"]["interface.delta_sasa"]
        assert (entry["n_ok"], entry["n_total"]) == (0, 1)

    def test_a_key_that_is_asked_for_is_listed_even_when_absent(self):
        rows = [_row("a1", "a", native={"iptm": 0.5})]
        entry = aggregate_poses(rows, keys=["native.gpde"])["a"]["native.gpde"]
        assert (entry["n_ok"], entry["n_total"]) == (0, 1)
