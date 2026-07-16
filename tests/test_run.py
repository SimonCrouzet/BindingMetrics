"""Tests for the binding-metrics-run pipeline failure detection.

Regression: the pipeline used to print "DONE" and exit 0 even when metric
steps errored or produced no result (e.g. a failed relaxation leaving empty
energy / NaN interface). _collect_failures is what turns those into a non-zero
exit instead of a silent pass.
"""

from binding_metrics.cli.run import _collect_failures


def test_no_failures_on_clean_results():
    results = {
        "energy": {"success": True, "relaxed_interaction_energy": -357.4},
        "interface": {"delta_sasa": 1574.0, "hbonds": 7},
        "geometry": {"ramachandran": {"ramachandran_favoured_pct": 100.0}},
        "openfold": {"skipped": True},
        "total_elapsed_s": 17.7,
    }
    assert _collect_failures(results) == []


def test_detects_error_key():
    results = {
        "geometry": {"error": "index -1 is out of bounds for axis 0 with size 0"},
        "interface": {"delta_sasa": 1200.0},
    }
    failures = _collect_failures(results)
    assert [s for s, _ in failures] == ["geometry"]


def test_detects_success_false():
    """A failed relaxation (success=False) must count as a failure."""
    results = {
        "relax": {"success": False, "error_message": "KeyError: 'N'"},
        "energy": {"success": True, "relaxed_interaction_energy": -1.0},
    }
    failures = _collect_failures(results)
    assert ("relax", "KeyError: 'N'") in failures


def test_skipped_steps_are_not_failures():
    results = {
        "energy": {"skipped": True},
        "openfold": {"skipped": True},
        "relax": {"skipped": True},
    }
    assert _collect_failures(results) == []


def test_multiple_failures_collected():
    results = {
        "relax": {"success": False, "error_message": "boom"},
        "energy": {"error": "no template"},
        "interface": {"delta_sasa": 1.0},          # ok
        "geometry": {"error": "empty chain"},
        "electrostatics": {"skipped": True},        # not a failure
    }
    steps = {s for s, _ in _collect_failures(results)}
    assert steps == {"relax", "energy", "geometry"}


def test_ignores_non_dict_and_scalars():
    results = {"total_elapsed_s": 12.3, "input": "x.cif", "energy": {"error": "e"}}
    assert [s for s, _ in _collect_failures(results)] == ["energy"]
