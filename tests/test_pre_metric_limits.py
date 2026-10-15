"""The input limits declared on the registry entries, and the behaviour that justifies each one.

A limit is declared only where the code of the metric shows it. The "evidence" tests run the
metric on an input that breaks the limit and pin what happens, so that a metric that starts to
handle the input makes its test fail and its entry has to go. The rest of the file checks what
``preflight`` does with the declared limits on the bundled examples.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path

import pytest

from binding_metrics.capabilities import (
    Capabilities,
    IncompatibleInputError,
    preflight,
    profile_input,
)
from binding_metrics.metrics.registry import METRICS, MetricSpec, get_metric

DATA = Path(__file__).resolve().parent.parent / "data"
TWO_CHAINS = DATA / "example_linear_p53_1YCR.pdb"  # peptide B, receptor A
ONE_CHAIN = DATA / "example_lactam_somatostatin_1XY4.cif"  # a single chain A

RECEPTOR_METRICS = (
    "interface",
    "coulomb",
    "shape_complementarity",
    "void_volume",
    "delta_sasa_static",
    "hbonds",
    "saltbridges",
    "evobind_score",
    "evobind_adversarial",
    "interface_pae",
    "structure_interaction_energy",
)
REFERENCE_METRICS = ("dockq",)
PREDICTION_METRICS = (
    "evobind_score",
    "evobind_adversarial",
    "interface_pae",
    "openfold",
    "prediction",
)
FORCE_FIELD_METRICS = ("md_implicit", "structure_interaction_energy")

DECLARED = sorted(
    set(RECEPTOR_METRICS + REFERENCE_METRICS + PREDICTION_METRICS + FORCE_FIELD_METRICS)
)


class TestTheRegistryField:
    def test_exactly_the_listed_metrics_declare_limits(self):
        assert sorted(m.name for m in METRICS if m.capabilities is not None) == DECLARED

    def test_the_field_is_optional_and_keyword_only(self):
        spec = MetricSpec(
            "x", "binding_metrics.metrics.geometry:compute_omega_planarity", "d", "static_structure"
        )
        assert spec.capabilities is None
        field = next(f for f in dataclasses.fields(MetricSpec) if f.name == "capabilities")
        assert field.kw_only is True
        with pytest.raises(TypeError):  # 20 positional arguments do not reach it
            MetricSpec(*range(30))

    def test_a_spec_with_a_limit_keeps_its_other_fields(self):
        spec = get_metric("dockq")
        assert (spec.input_type, spec.secondary_path_arg, spec.path_arg) == (
            "static_structure",
            "reference_path",
            "model_path",
        )

    @pytest.mark.parametrize("name", DECLARED)
    def test_every_limit_names_the_function_that_shows_it(self, name):
        spec = get_metric(name)
        function = spec.import_path.split(":")[1]
        caps = spec.capabilities
        assert isinstance(caps, Capabilities)
        for field in caps.constrained_fields():
            assert function in caps.reasons[field], f"{name}: the {field} reason cites no function"

    def test_no_declared_limit_is_a_gpu_need(self):
        # requires_gpu is a scheduling hint, not a limit of the input
        assert all("gpu" not in m.capabilities.needs for m in METRICS if m.capabilities)


class TestTheDeclaredNeeds:
    @pytest.mark.parametrize("name", RECEPTOR_METRICS)
    def test_the_receptor_metrics(self, name):
        assert "receptor_chain" in get_metric(name).capabilities.needs

    @pytest.mark.parametrize("name", REFERENCE_METRICS)
    def test_the_reference_metric(self, name):
        assert get_metric(name).capabilities.needs == frozenset({"reference_structure"})

    @pytest.mark.parametrize("name", PREDICTION_METRICS)
    def test_the_prediction_metrics(self, name):
        assert "predicted_structure" in get_metric(name).capabilities.needs

    @pytest.mark.parametrize("name", FORCE_FIELD_METRICS)
    def test_the_force_field_metrics(self, name):
        assert get_metric(name).capabilities.closures == frozenset(
            {"none", "head_to_tail", "disulfide", "lactam", "staple"}
        )


class TestPreflightOnTheExamples:
    def test_a_complex_passes_every_receptor_metric(self):
        profile = profile_input(TWO_CHAINS, "B", "A")
        assert preflight(profile, RECEPTOR_METRICS).compatible

    def test_a_complex_whose_receptor_is_not_named_still_has_one(self):
        # 1YCR chain A is a protein chain, and the metrics pick it when none is given
        profile = profile_input(TWO_CHAINS, "B")
        assert profile.receptor_chain is None and profile.other_protein_chains == ("A",)
        assert preflight(profile, RECEPTOR_METRICS).compatible

    def test_a_single_chain_is_refused_by_every_receptor_metric_at_once(self):
        profile = profile_input(ONE_CHAIN, "A")
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(profile, list(RECEPTOR_METRICS))
        refused = {v.name for v in caught.value.violations if v.constraint == "needs"}
        assert refused == set(RECEPTOR_METRICS)
        message = str(caught.value)
        assert "metric 'coulomb': needs" in message
        assert "no receptor chain was given and the structure has no other protein chain" in message
        assert "0.0 kJ/mol" in message  # the reason says what coulomb would have done

    def test_the_skip_policy_keeps_the_metrics_that_need_no_receptor(self):
        profile = profile_input(ONE_CHAIN, "A")
        report = preflight(
            profile,
            ["ramachandran", "omega", "interface", "receptor_quality", "coulomb"],
            policy="skip",
        )
        assert report.metrics_to_run == ("ramachandran", "omega", "receptor_quality")
        assert report.skipped_metrics == ("interface", "coulomb")

    def test_reference_and_prediction_needs_follow_what_the_caller_provides(self):
        profile = profile_input(TWO_CHAINS, "B", "A")
        wanted = ["dockq", "openfold", "prediction", "interface_pae"]
        assert preflight(profile, wanted).compatible  # nothing said: not checked
        assert preflight(
            profile, wanted, provided={"reference_structure", "predicted_structure"}
        ).compatible
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(profile, wanted, provided=set())
        assert {v.name for v in caught.value.violations} == set(wanted)

    def test_the_force_field_steps_take_every_closure_but_an_unsupported_link(self):
        for example, binder in (
            ("example_bicyclic_sfti1_3P8F.cif", "I"),
            ("example_lactam_somatostatin_1XY4.cif", "A"),
            ("example_staple_3V3B.pdb", "C"),
            ("example_ncaa_cyclosporin_1CWA.cif", "C"),
        ):
            profile = profile_input(
                DATA / example, binder, "A" if example != "example_staple_3V3B.pdb" else None
            )
            assert preflight(profile, ["md_implicit"]).compatible, example

    def test_an_unsupported_cross_link_is_refused_with_the_reason(self):
        from tests.test_pre_structures import build_chain

        atoms = build_chain(
            ["ALA", "CYS", "GLY", "GLY", "GLY", "XAA", "ALA"],
            "B",
            side_chain_atoms={"XAA": ("CB", "CE")},
            bonds=[(1, "SG", 5, "CE")],
        )
        profile = profile_input(atoms, "B")
        assert profile.closures == frozenset({"other"})
        with pytest.raises(IncompatibleInputError) as caught:
            preflight(profile, ["md_implicit", "ramachandran"])
        (violation,) = caught.value.violations
        assert violation.name == "md_implicit" and violation.constraint == "closures"
        assert "CyclizationError" in violation.reason and "thioether" in violation.reason


class TestEvidenceForTheReceptorNeed:
    """What each metric does with one protein chain and no receptor."""

    def test_interface_raises(self):
        with pytest.raises(ValueError, match="Chain auto-detection failed"):
            get_metric("interface").call(cif_path=str(ONE_CHAIN))

    def test_coulomb_returns_zero_energy_and_no_pair(self):
        result = get_metric("coulomb").call(cif_path=str(ONE_CHAIN))
        assert result["coulomb_energy_kJ"] == 0.0 and result["n_charged_pairs"] == 0

    def test_shape_complementarity_returns_nan(self):
        result = get_metric("shape_complementarity").call(cif_path=str(ONE_CHAIN))
        assert math.isnan(result["sc"]) and result["n_surface_dots_A"] == 0

    def test_void_volume_returns_nan_with_the_reason(self):
        result = get_metric("void_volume").call(cif_path=str(ONE_CHAIN))
        assert math.isnan(result["void_volume_A3"])
        assert "fewer than two protein chains" in result["reason"]

    def test_delta_sasa_hbonds_and_saltbridges_need_the_receptor_argument(self):
        from binding_metrics.metrics._common import load_structure

        with pytest.raises(TypeError, match="receptor_chain"):
            get_metric("delta_sasa_static").call(cif_path=str(ONE_CHAIN), peptide_chain="A")
        atoms = load_structure(ONE_CHAIN)
        for name in ("hbonds", "saltbridges"):
            with pytest.raises(TypeError, match="receptor_chain"):
                get_metric(name).call(atoms=atoms, peptide_chain="A")

    def test_the_interaction_energy_reports_that_it_found_no_second_chain(self):
        pytest.importorskip("openmm")
        result = get_metric("structure_interaction_energy").call(input_path=str(ONE_CHAIN))
        assert result["success"] is False
        assert "Could not identify two protein chains" in result["error_message"]

    @pytest.mark.parametrize(
        "name, argument",
        [
            ("evobind_score", "receptor_chain"),
            ("evobind_adversarial", "receptor_chain"),
            ("interface_pae", "receptor_chain"),
        ],
    )
    def test_the_receptor_is_required_by_the_functions_that_take_a_prediction(self, name, argument):
        spec = get_metric(name)
        with pytest.raises(TypeError, match=argument):
            spec.call(**{spec.path_arg: "a", spec.peptide_chain_arg: "B", **_placeholders(spec)})


def _placeholders(spec):
    """Values for the arguments the call needs besides the chains (the receptor is left out)."""
    extra = {}
    if spec.secondary_path_arg:
        extra[spec.secondary_path_arg] = "b"
    if spec.name == "evobind_score":
        extra["plddt_per_atom"] = None
    return extra


class TestEvidenceForTheReferenceAndPredictionNeeds:
    @pytest.mark.parametrize(
        "name, argument",
        [
            ("dockq", "reference_path"),
            ("evobind_adversarial", "afm_structure_path"),
            ("interface_pae", "confidences_path"),
            ("openfold", "output_dir"),
            ("prediction", "prediction_dir"),
        ],
    )
    def test_the_argument_is_required(self, name, argument):
        parameter = inspect.signature(get_metric(name).load()).parameters[argument]
        assert parameter.default is inspect.Parameter.empty

    def test_the_evobind_score_is_none_without_a_pLDDT(self):
        result = get_metric("evobind_score").call(
            structure_path=str(TWO_CHAINS),
            plddt_per_atom=None,
            binder_chain="B",
            receptor_chain="A",
        )
        assert result["evobind_score"] is None


class TestEvidenceForTheClosureLimit:
    @pytest.fixture
    def thioether_topology(self):
        pytest.importorskip("openmm")
        from binding_metrics.io.structures import load_structure

        topology, positions = load_structure(DATA / "example_bicyclic_sfti1_3P8F.cif")
        chain = next(c for c in topology.chains() if c.id == "B")
        residues = list(chain.residues())
        sulfur = next(a for a in residues[2].atoms() if a.name == "SG")
        carbon = next(a for a in residues[6].atoms() if a.name == "CB")
        topology.addBond(sulfur, carbon)  # a sulfur-carbon link between two side chains
        return topology, positions

    def test_detect_cyclization_raises_for_a_link_that_is_no_supported_pattern(
        self, thioether_topology
    ):
        from binding_metrics.core.cyclic import CyclizationError, detect_cyclization

        topology, positions = thioether_topology
        with pytest.raises(CyclizationError, match="Unsupported cyclization"):
            detect_cyclization(topology, positions, "B")

    def test_patch_cyclic_topology_raises_the_same(self, thioether_topology):
        from binding_metrics.core.cyclic import CyclizationError, patch_cyclic_topology

        topology, positions = thioether_topology
        with pytest.raises(CyclizationError):
            patch_cyclic_topology(topology, positions, "B")

    def test_both_force_field_steps_call_it_without_a_switch(self):
        pytest.importorskip("openmm")
        from binding_metrics.metrics import energy
        from binding_metrics.protocols import relaxation

        # _create_implicit_system builds the systems of compute_interaction_energy
        for function in (
            energy._create_implicit_system,
            relaxation.ImplicitRelaxation._setup_system,
        ):
            source = inspect.getsource(function)
            assert "patch_cyclic_topology(" in source
            assert "if is_cyclic" not in source and "if self.config.is_cyclic" not in source
        assert "_create_implicit_system(" in inspect.getsource(energy.compute_interaction_energy)
