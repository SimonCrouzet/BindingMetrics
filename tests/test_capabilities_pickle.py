"""Capabilities, InputProfile and the registry specs survive pickle and deepcopy (issue #136).

A process pool (``score_pose_set(n_workers > 1)``, ``run_batch``) sends them to its workers, and
a ``spawn`` worker needs them pickled. The read-only mapping views that these classes hold
cannot be pickled themselves, so they travel as plain dicts and come back as read-only views.
"""

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import MappingProxyType

import pytest

from binding_metrics.capabilities import (
    Capabilities,
    InputProfile,
    check_openfold3_residues,
    preflight,
    profile_input,
)
from binding_metrics.metrics.registry import METRICS, get_metric

DATA = Path(__file__).resolve().parents[1] / "data"
SFTI = DATA / "example_bicyclic_sfti1_3P8F.cif"
CYCLOSPORIN = DATA / "example_ncaa_cyclosporin_1CWA.cif"


def round_trips(value):
    """The value after pickle and after deepcopy."""
    return pickle.loads(pickle.dumps(value)), copy.deepcopy(value)


def declared() -> Capabilities:
    return Capabilities(
        binder_types={"peptide"},
        closures={"none", "head_to_tail"},
        residue_classes={"canonical", "d_amino"},
        min_binder_residues=3,
        max_binder_residues=40,
        multi_chain_binder=False,
        needs={"receptor_chain"},
        modes={"predict", "score"},
        extra_checks=(check_openfold3_residues,),
        reasons={
            "binder_types": "only peptides were validated",
            "closures": "a disulfide cannot be patched",
            "closures:disulfide": "the S-S bond has no template",
            "residue_classes": "no force-field entry",
            "min_binder_residues": "too short",
            "max_binder_residues": "too long",
            "multi_chain_binder": "one chain only",
            "needs": "needs a receptor",
            "modes": "refold is not supported",
        },
        caveats={"closures:head_to_tail": "never validated on rings"},
        version="1.2.3",
    )


@pytest.fixture(scope="module")
def profile():
    from binding_metrics.metrics._common import load_structure

    return profile_input(load_structure(SFTI), "I", "A")


class TestCapabilities:
    def test_a_declared_object_round_trips_equal_and_with_the_same_hash(self):
        original = declared()
        for clone in round_trips(original):
            assert clone == original
            assert hash(clone) == hash(original)
            assert clone is not original

    def test_the_reasons_and_the_caveats_are_read_only_views_again(self):
        for clone in round_trips(declared()):
            assert isinstance(clone.reasons, MappingProxyType)
            assert isinstance(clone.caveats, MappingProxyType)
            assert clone.reasons["closures:disulfide"] == "the S-S bond has no template"
            assert clone.caveats == {"closures:head_to_tail": "never validated on rings"}
            with pytest.raises(TypeError):
                clone.reasons["x"] = "y"  # type: ignore[index]

    def test_the_fields_keep_their_types(self):
        for clone in round_trips(declared()):
            assert isinstance(clone.closures, frozenset) and "head_to_tail" in clone.closures
            assert clone.extra_checks == (check_openfold3_residues,)
            assert (clone.min_binder_residues, clone.max_binder_residues) == (3, 40)
            assert clone.version == "1.2.3" and clone.multi_chain_binder is False

    def test_a_clone_checks_an_input_like_the_original(self, profile):
        original = declared()
        for clone in round_trips(original):
            assert [v.to_dict() for v in clone.check(profile)] == [
                v.to_dict() for v in original.check(profile)
            ]

    def test_the_default_object_round_trips(self):
        for clone in round_trips(Capabilities()):
            assert clone == Capabilities() and clone.is_unconstrained

    def test_the_frozen_dataclass_is_still_frozen(self):
        for clone in round_trips(declared()):
            with pytest.raises(AttributeError):
                clone.version = "other"  # type: ignore[misc]

    def test_the_derived_values_do_not_change(self):
        # the repr of a frozenset depends on how it was built, so it is not compared
        original = declared()
        for clone in round_trips(original):
            assert clone.constrained_fields() == original.constrained_fields()
            assert clone.reason_for("closures", "disulfide") == original.reason_for(
                "closures", "disulfide"
            )

    def test_a_predictor_declaration_with_a_check_function_round_trips(self):
        from binding_metrics.predictors.of3 import OpenFold3Parser

        original = OpenFold3Parser.capabilities
        clone = pickle.loads(pickle.dumps(original))
        assert clone == original and clone.extra_checks == original.extra_checks


class TestInputProfile:
    def test_a_profile_of_a_bundled_complex_round_trips_equal(self, profile):
        for clone in round_trips(profile):
            assert isinstance(clone, InputProfile)
            assert clone == profile and hash(clone) == hash(profile)
            assert clone.to_dict() == profile.to_dict()

    def test_the_mappings_are_read_only_views_again(self, profile):
        for clone in round_trips(profile):
            assert isinstance(clone.residue_names, MappingProxyType)
            assert isinstance(clone.residue_labels, MappingProxyType)
            assert clone.residue_names == profile.residue_names
            with pytest.raises(TypeError):
                clone.residue_labels["x"] = ()  # type: ignore[index]

    def test_the_closures_and_the_derived_values_survive(self, profile):
        clone = pickle.loads(pickle.dumps(profile))
        assert clone.closures == profile.closures == {"head_to_tail", "disulfide"}
        assert [c.describe() for c in clone.closure_bonds] == [
            c.describe() for c in profile.closure_bonds
        ]
        assert clone.binder_chain == "I" and clone.describe() == profile.describe()

    def test_a_profile_with_non_canonical_residues_round_trips(self):
        from binding_metrics.metrics._common import load_structure

        original = profile_input(load_structure(CYCLOSPORIN), "C", "A")
        assert "n_methyl" in original.residue_classes
        clone = pickle.loads(pickle.dumps(original))
        assert clone == original and clone.residue_names == original.residue_names

    def test_a_preflight_report_round_trips(self, profile):
        report = preflight(profile, ["interface", "omega", "coulomb"], policy="warn")
        for clone in round_trips(report):
            assert clone == report and clone.to_dict() == report.to_dict()
            assert clone.format() == report.format()


class TestRegistry:
    @pytest.mark.parametrize("spec", METRICS, ids=lambda spec: spec.name)
    def test_every_registered_spec_round_trips_equal(self, spec):
        for clone in round_trips(spec):
            assert clone == spec and hash(clone) == hash(spec)
            assert clone.capabilities == spec.capabilities
            assert clone.import_path == spec.import_path

    def test_a_spec_with_a_declared_limit_keeps_it(self):
        spec = get_metric("interface")
        assert spec.capabilities is not None
        clone = pickle.loads(pickle.dumps(spec))
        assert clone.capabilities.needs == {"receptor_chain"}
        assert clone.capabilities.reasons["needs"] == spec.capabilities.reasons["needs"]

    def test_a_cloned_spec_still_calls_its_function(self, tmp_path):
        clone = pickle.loads(pickle.dumps(get_metric("ramachandran")))
        assert clone.load() is get_metric("ramachandran").load()


def _back(value):
    """Return what a worker received, and a value computed from it there."""
    if isinstance(value, InputProfile):
        derived = value.describe()
    elif isinstance(value, Capabilities):
        derived = sorted(value.reasons)
    else:
        derived = (value.name, value.capabilities is not None)
    return value, derived


class TestWorkerProcess:
    @pytest.mark.parametrize("method", multiprocessing.get_all_start_methods())
    def test_the_objects_make_the_round_trip_to_a_worker_and_back(self, method, profile):
        values = [declared(), profile, get_metric("interface"), get_metric("hbonds")]
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context(method)
        ) as pool:
            results = list(pool.map(_back, values))
        for original, (clone, _derived) in zip(values, results):
            assert clone == original
        assert results[0][1] == sorted(declared().reasons)
        assert results[1][1] == profile.describe()
        assert results[2][1] == ("interface", True)
        assert isinstance(results[0][0].reasons, MappingProxyType)
