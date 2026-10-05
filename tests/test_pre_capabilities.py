"""``Capabilities``: defaults, validation, and the matrix of constraints against profiles."""

from __future__ import annotations

import dataclasses

import pytest

from binding_metrics.capabilities import (
    CLOSURE_FAMILIES,
    Capabilities,
    Closure,
    ClosureEnd,
    InputProfile,
    Violation,
)


def _closure(kind, family, a=(2, "CYS", 3, "SG"), b=(8, "CYS", 9, "SG")):
    return Closure(kind, family, ClosureEnd(*a), ClosureEnd(*b))


DISULFIDE = _closure("disulfide", "disulfide")
RING = _closure("head_to_tail", "head_to_tail", (9, "ALA", 10, "C"), (0, "GLY", 1, "N"))

LINEAR = InputProfile("B", "A", n_binder_residues=12, binder_type="peptide")
RING_PROFILE = InputProfile(
    "B",
    "A",
    n_binder_residues=12,
    binder_type="peptide",
    closures=frozenset({"head_to_tail"}),
    closure_bonds=(RING,),
)
BICYCLE = dataclasses.replace(
    RING_PROFILE, closures=frozenset({"head_to_tail", "disulfide"}), closure_bonds=(RING, DISULFIDE)
)
DISULFIDE_ONLY = dataclasses.replace(
    RING_PROFILE, closures=frozenset({"disulfide"}), closure_bonds=(DISULFIDE,)
)
D_AND_N_METHYL = dataclasses.replace(
    LINEAR,
    residue_classes=frozenset({"canonical", "d_amino", "n_methyl"}),
    residue_names={"canonical": ("ALA",), "d_amino": ("DAL",), "n_methyl": ("MLE", "SAR")},
)


class TestDefaults:
    def test_nothing_is_constrained_by_default(self):
        caps = Capabilities()
        assert caps.is_unconstrained and caps.constrained_fields() == ()
        assert caps.multi_chain_binder is True and caps.version == ""

    @pytest.mark.parametrize(
        "profile", [LINEAR, RING_PROFILE, BICYCLE, D_AND_N_METHYL, InputProfile("B")]
    )
    def test_an_unconstrained_step_accepts_every_input(self, profile):
        assert Capabilities().check(profile) == []
        assert Capabilities().accepts(profile, provided=set())

    def test_equal_and_hashable_and_frozen(self):
        assert Capabilities() == Capabilities()
        assert len({Capabilities(), Capabilities()}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            Capabilities().version = "1"  # type: ignore[misc]

    def test_collections_become_frozensets_and_mappings_read_only(self):
        caps = Capabilities(
            closures=["none", "head_to_tail"], reasons={"closures": "It reads no other ring."}
        )
        assert caps.closures == frozenset({"none", "head_to_tail"})
        with pytest.raises(TypeError):
            caps.reasons["closures"] = "changed"  # type: ignore[index]

    def test_the_vocabulary_of_closures_starts_with_none(self):
        assert CLOSURE_FAMILIES[0] == "none"


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"binder_types": {"protein"}},
            {"closures": {"macrocycle"}},
            {"residue_classes": {"peptoid"}},
            {"needs": {"internet"}},
        ],
    )
    def test_an_unknown_word_is_refused(self, kwargs):
        with pytest.raises(ValueError, match="unknown value"):
            Capabilities(**kwargs)

    def test_a_string_is_not_a_collection_of_words(self):
        with pytest.raises(ValueError, match="not the string"):
            Capabilities(closures="disulfide")

    @pytest.mark.parametrize("bad", [-1, 2.5, "10"])
    def test_a_size_bound_is_a_non_negative_integer(self, bad):
        with pytest.raises(ValueError, match="non-negative integer"):
            Capabilities(min_binder_residues=bad, reasons={"min_binder_residues": "Because."})

    def test_the_bounds_must_be_ordered(self):
        with pytest.raises(ValueError, match="exceeds"):
            Capabilities(
                min_binder_residues=10,
                max_binder_residues=5,
                reasons={"min_binder_residues": "a", "max_binder_residues": "b"},
            )

    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"closures": {"none"}}, "closures"),
            ({"binder_types": {"peptide"}}, "binder_types"),
            ({"residue_classes": {"canonical"}}, "residue_classes"),
            ({"min_binder_residues": 5}, "min_binder_residues"),
            ({"max_binder_residues": 50}, "max_binder_residues"),
            ({"multi_chain_binder": False}, "multi_chain_binder"),
            ({"needs": {"gpu"}}, "needs"),
        ],
    )
    def test_a_constraint_without_a_reason_is_refused(self, kwargs, field):
        with pytest.raises(ValueError, match=f"{field} is constrained but reasons has no sentence"):
            Capabilities(**kwargs)
        # a sentence about one value does not stand in for the sentence about the field
        if field in ("closures", "binder_types", "residue_classes", "needs"):
            value = next(iter(kwargs[field]))
            with pytest.raises(ValueError, match="no sentence"):
                Capabilities(**kwargs, reasons={f"{field}:{value}": "A sentence."})

    def test_an_unknown_reason_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown key 'closure'"):
            Capabilities(reasons={"closure": "typo"})
        with pytest.raises(ValueError, match="unknown key 'closures:ring'"):
            Capabilities(reasons={"closures:ring": "typo"})

    def test_a_caveat_names_a_field_and_a_value(self):
        with pytest.raises(ValueError, match="caveats has an unknown key 'closures'"):
            Capabilities(caveats={"closures": "no value"})
        with pytest.raises(ValueError, match="unknown key 'needs:gpu'"):
            Capabilities(caveats={"needs:gpu": "not a validation state"})

    def test_a_sentence_is_not_empty(self):
        with pytest.raises(ValueError, match="non-empty sentence"):
            Capabilities(closures={"none"}, reasons={"closures": "  "})

    def test_reason_for_prefers_the_sentence_about_the_value(self):
        caps = Capabilities(
            closures={"none"},
            reasons={"closures": "General.", "closures:disulfide": "About disulfides."},
        )
        assert caps.reason_for("closures", "disulfide") == "About disulfides."
        assert caps.reason_for("closures", "staple") == "General."
        assert caps.reason_for("closures") == "General."
        assert caps.reason_for("needs") == ""


OPENFOLD_LIKE = Capabilities(
    closures={"none", "head_to_tail"},
    reasons={
        "closures": "The query builder can only express a head-to-tail closure.",
        "closures:disulfide": "A disulfide cannot be written into the query.",
    },
    version="0.5.0",
)


class TestClosures:
    @pytest.mark.parametrize("profile", [LINEAR, RING_PROFILE])
    def test_the_allowed_closures_pass(self, profile):
        assert OPENFOLD_LIKE.check(profile) == []

    def test_a_disulfide_is_refused_with_the_fact_the_requirement_and_the_reason(self):
        (violation,) = OPENFOLD_LIKE.check(DISULFIDE_ONLY)
        assert violation.constraint == "closures"
        assert "a disulfide bond" in violation.fact
        assert "CYS 3.SG - CYS 9.SG" in violation.fact
        assert violation.requirement == "closures limited to: none, head_to_tail"
        assert violation.reason == "A disulfide cannot be written into the query."

    def test_only_the_offending_closure_of_a_bicycle_is_reported(self):
        (violation,) = OPENFOLD_LIKE.check(BICYCLE)
        assert "disulfide" in violation.fact and "head-to-tail" not in violation.fact

    def test_a_step_that_needs_a_ring_refuses_a_linear_binder(self):
        cyclic_only = Capabilities(
            closures=set(CLOSURE_FAMILIES) - {"none"},
            reasons={"closures": "The metric scores the closing bond."},
        )
        (violation,) = cyclic_only.check(LINEAR)
        assert "linear" in violation.fact
        assert violation.requirement.startswith("needs a cyclic binder")
        assert cyclic_only.check(RING_PROFILE) == []
        assert cyclic_only.check(BICYCLE) == []


class TestResidueClasses:
    caps = Capabilities(
        residue_classes={"canonical", "d_amino"},
        reasons={"residue_classes": "Only these are in the query grammar."},
    )

    def test_a_class_outside_the_set_is_named_with_its_residue_codes(self):
        (violation,) = self.caps.check(D_AND_N_METHYL)
        assert violation.constraint == "residue_classes"
        assert "N-methylated residues (MLE, SAR)" in violation.fact
        assert violation.requirement == "residue classes limited to: canonical, d_amino"

    def test_the_listed_classes_pass(self):
        assert self.caps.check(LINEAR) == []
        profile = dataclasses.replace(D_AND_N_METHYL, residue_classes=frozenset({"d_amino"}))
        assert self.caps.check(profile) == []


class TestBinderTypes:
    caps = Capabilities(
        binder_types={"nanobody", "antibody"},
        reasons={"binder_types": "The numbering scheme is for immunoglobulin domains."},
    )

    def test_an_estimated_type_outside_the_set_is_refused_and_says_it_was_estimated(self):
        (violation,) = self.caps.check(LINEAR)
        assert "peptide (estimated from size)" in violation.fact
        assert violation.requirement == "binder type one of: antibody, nanobody"

    def test_a_given_type_is_reported_as_given(self):
        profile = dataclasses.replace(LINEAR, binder_type="miniprotein", binder_type_source="given")
        (violation,) = self.caps.check(profile)
        assert "miniprotein (given)" in violation.fact

    def test_the_listed_types_pass(self):
        profile = dataclasses.replace(LINEAR, binder_type="nanobody", binder_type_source="given")
        assert self.caps.check(profile) == []

    def test_an_unknown_type_never_blocks(self):
        assert self.caps.check(dataclasses.replace(LINEAR, binder_type="unknown")) == []


class TestSizeAndChains:
    caps = Capabilities(
        min_binder_residues=5,
        max_binder_residues=20,
        multi_chain_binder=False,
        reasons={
            "min_binder_residues": "Too short to fold.",
            "max_binder_residues": "Out of memory beyond this.",
            "multi_chain_binder": "One chain only.",
        },
    )

    @pytest.mark.parametrize("n, fits", [(4, False), (5, True), (20, True), (21, False)])
    def test_the_bounds_include_their_ends(self, n, fits):
        profile = dataclasses.replace(LINEAR, n_binder_residues=n)
        assert self.caps.accepts(profile) is fits

    def test_the_constraint_and_the_numbers_are_named(self):
        (short,) = self.caps.check(dataclasses.replace(LINEAR, n_binder_residues=4))
        assert short.constraint == "min_binder_residues"
        assert short.fact == "the binder has 4 residues"
        assert short.requirement == "at least 5 residues"
        (long,) = self.caps.check(dataclasses.replace(LINEAR, n_binder_residues=21))
        assert long.requirement == "at most 20 residues"

    def test_a_binder_of_several_chains(self):
        profile = dataclasses.replace(LINEAR, binder_chains=("H", "L"))
        (violation,) = self.caps.check(profile)
        assert violation.constraint == "multi_chain_binder"
        assert "2 chains (H, L)" in violation.fact


class TestNeeds:
    receptor = Capabilities(needs={"receptor_chain"}, reasons={"needs": "It scores an interface."})
    reference = Capabilities(
        needs={"reference_structure", "gpu"}, reasons={"needs": "It compares to a reference."}
    )

    def test_the_receptor_chain_is_read_from_the_profile(self):
        assert self.receptor.accepts(LINEAR)
        (violation,) = self.receptor.check(dataclasses.replace(LINEAR, receptor_chain=None))
        assert violation.fact == (
            "no receptor chain was given and the structure has no other protein chain"
        )
        assert violation.requirement == "needs a receptor chain"

    def test_another_protein_chain_of_the_structure_meets_the_need(self):
        # the metrics take the largest other protein chain when none is given
        profile = dataclasses.replace(LINEAR, receptor_chain=None, other_protein_chains=("A",))
        assert self.receptor.accepts(profile)

    def test_the_other_needs_are_checked_only_when_the_caller_says_what_it_provides(self):
        assert self.reference.check(LINEAR) == []
        assert self.reference.check(LINEAR, provided={"reference_structure", "gpu"}) == []
        violations = self.reference.check(LINEAR, provided={"gpu"})
        assert [v.fact for v in violations] == ["no reference structure was provided"]
        both = self.reference.check(LINEAR, provided=set())
        assert [v.requirement for v in both] == ["needs a reference structure", "needs a GPU"]


class TestSeveralProblemsAtOnce:
    def test_every_violation_is_returned_not_the_first(self):
        caps = Capabilities(
            closures={"none"},
            residue_classes={"canonical"},
            max_binder_residues=10,
            binder_types={"antibody"},
            needs={"receptor_chain"},
            reasons={
                "closures": "a",
                "residue_classes": "b",
                "max_binder_residues": "c",
                "binder_types": "d",
                "needs": "e",
            },
        )
        profile = dataclasses.replace(
            BICYCLE,
            receptor_chain=None,
            residue_classes=frozenset({"canonical", "d_amino"}),
            residue_names={"canonical": ("ALA",), "d_amino": ("DAL",)},
        )
        assert [v.constraint for v in caps.check(profile)] == [
            "binder_types",
            "closures",  # head-to-tail: "none" is the only closure allowed
            "closures",  # disulfide
            "residue_classes",
            "max_binder_residues",
            "needs",
        ]


class TestCaveats:
    caps = Capabilities(
        caveats={
            "closures:disulfide": "Never benchmarked with a disulfide.",
            "residue_classes:d_amino": "D residues are outside the benchmark.",
            "binder_types:peptide": "Validated on proteins only.",
        }
    )

    def test_a_caveat_applies_when_the_input_has_the_value(self):
        assert self.caps.caveats_for(DISULFIDE_ONLY) == [
            "Never benchmarked with a disulfide.",
            "Validated on proteins only.",
        ]
        assert self.caps.caveats_for(D_AND_N_METHYL) == [
            "D residues are outside the benchmark.",
            "Validated on proteins only.",
        ]

    def test_a_caveat_does_not_refuse(self):
        assert self.caps.check(DISULFIDE_ONLY) == []

    def test_no_caveat_for_an_input_without_the_value(self):
        assert self.caps.caveats_for(dataclasses.replace(RING_PROFILE, binder_type="unknown")) == []


class TestViolation:
    def test_format_lists_the_fact_the_requirement_the_reason_and_the_fix(self):
        (violation,) = OPENFOLD_LIKE.check(DISULFIDE_ONLY)
        text = dataclasses.replace(
            violation, subject="predictor OpenFold3 0.5.0", fix="use another model"
        ).format()
        assert text.splitlines()[0] == "predictor OpenFold3 0.5.0: closures"
        for part in ("found:", "requires:", "why:", "fix:    ", "use another model"):
            assert part in text

    def test_a_violation_without_reason_or_fix_omits_those_lines(self):
        text = Violation("closures", "the fact", "the requirement").format()
        assert "why:" not in text and "fix:" not in text

    def test_to_dict_carries_every_field(self):
        (violation,) = OPENFOLD_LIKE.check(DISULFIDE_ONLY)
        assert set(violation.to_dict()) == {
            "kind",
            "name",
            "subject",
            "constraint",
            "fact",
            "requirement",
            "reason",
            "fix",
        }


def _refuse_short_chains(profile):
    if profile.n_binder_residues < 20:
        return [
            Violation(
                constraint="residue_classes",
                fact=f"the binder has {profile.n_binder_residues} residues",
                requirement="at least 20 residues",
                reason="A rule that only the model's own code can state.",
            )
        ]
    return []


class TestExtraChecks:
    caps = Capabilities(extra_checks=(_refuse_short_chains,))

    def test_an_extra_check_is_a_constraint_that_needs_no_reason_entry(self):
        assert self.caps.constrained_fields() == ("extra_checks",)
        assert not self.caps.is_unconstrained

    def test_its_violations_are_returned_with_the_others(self):
        (violation,) = self.caps.check(LINEAR)
        assert violation.fact == "the binder has 12 residues"
        assert violation.reason == "A rule that only the model's own code can state."
        assert self.caps.accepts(dataclasses.replace(LINEAR, n_binder_residues=30))

    def test_it_runs_after_the_built_in_constraints(self):
        caps = Capabilities(
            closures={"none"},
            reasons={"closures": "No ring."},
            extra_checks=(_refuse_short_chains,),
        )
        assert [v.constraint for v in caps.check(DISULFIDE_ONLY)] == ["closures", "residue_classes"]

    def test_a_list_of_functions_becomes_a_tuple_and_the_object_stays_hashable(self):
        caps = Capabilities(extra_checks=[_refuse_short_chains])
        assert caps.extra_checks == (_refuse_short_chains,)
        assert hash(caps) == hash(Capabilities(extra_checks=(_refuse_short_chains,)))

    def test_something_that_is_not_a_function_is_refused(self):
        with pytest.raises(ValueError, match="extra_checks must be functions"):
            Capabilities(extra_checks=("not a function",))

    def test_a_check_must_return_violations(self):
        caps = Capabilities(extra_checks=(lambda profile: ["a string"],))
        with pytest.raises(TypeError, match="not Violation"):
            caps.check(LINEAR)
